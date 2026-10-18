#!/usr/bin/env python3
"""Prepare/freeze locally; execute ONLY within one separately authorized H200 allocation.

No SSH, fetch, push, sbatch, allocation, install, retry, cleanup or publication command.
"""
import argparse
import ast
import datetime
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from policy import *
import science

HERE = Path(__file__).resolve().parent
PY = E + "/bin/python"


def literal_assignments(path, names):
    values = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == set(names), "literal source metadata unavailable")
    return values


def prepare(args):
    local = dict(baseline=Path(args.baseline_local).resolve(), candidate=Path(args.candidate_local).resolve())
    harness = HERE.parents[1]
    hp = git(harness, "rev-parse", "HEAD")
    check_source(harness, hp)
    full_pin(hp)
    candidate_pin = args.approved_candidate_pin if args.command == "freeze" else git(local["candidate"], "rev-parse", "HEAD")
    full_pin(candidate_pin)
    arms = {}
    for arm, pin in [("baseline", BASE), ("candidate", candidate_pin)]:
        check_source(local[arm], pin)
        files = file_manifest(local[arm])
        arms[arm] = dict(source=SOURCES[arm], pin=pin, tree=git(local[arm], "rev-parse", "HEAD^{tree}"),
                         backend="gpu", build=SOURCES[arm] + "/build_major2_cuda", files=files, files_digest=digest(files))
    require(subprocess.run(["git", "-C", str(local["candidate"]), "merge-base", "--is-ancestor", BASE, candidate_pin]).returncode == 0,
            "candidate not descended from merged baseline")
    changed = set(git(local["candidate"], "diff", "--name-only", BASE, candidate_pin).splitlines())
    require(changed == OWNED_DIFF, "candidate diff not the five reviewed paths; new review/harness policy required")
    require(tree_manifest(local["baseline"] / "psi4/driver") == tree_manifest(local["candidate"] / "psi4/driver"), "scientific drivers differ")
    additions = git(harness, "diff", "--name-status", BASE, hp).splitlines()
    require(additions and all(x.startswith("A\tdevtools/major2/") for x in additions), "harness must only add its own helpers from BASE")
    hfiles = file_manifest(harness)
    # Freeze all shared toolchain/dependency small input files by path/hash before job.
    # Missing remote-only compiler files are rehashed and recorded inside the GPU allocation.
    m = dict(schema=1, route=ROUTE, gates=GATES, run_root=RUN, scratch_parent="/storage/scratch1/3/awallace43/cpks-major2",
             suite=H0 + "/geometries", suite_sha256=SUITE_SHA, systems=SYSTEMS, sequence=sequence(), arms=arms,
             harness=dict(path=HARNESS, pin=hp, files=hfiles, files_digest=digest(hfiles)),
             dependency_hashes=LIBS, inspection_tool=INSPECTION_TOOL,
             common_selectors=COMMON, fixed_selector=FIXED, fixed_k=FIXED_K,
             selectors_ast=selectors_ast(local["candidate"]),
             runtime_policy=dict(threads=8, memory="100 GiB", cache="enabled-shared-job-1GiB",
                                 prewarm=False, print_override=False, affinity_override=False),
             approval=dict(state="APPROVED" if args.command == "freeze" else "PENDING",
                           review_id=args.review_id if args.command == "freeze" else "",
                           candidate_pin=candidate_pin if args.command == "freeze" else None),
             observed_candidate_pin=candidate_pin,
             rhf_tolerance_note="Reviewed focused tests remain stricter: atol=2e-13 rtol=2e-13; response 1e-12/1e-10. No weakening.",
             build_budget_seconds=[3600, 3600], test_smoke_budget_seconds=1800,
             production_budget_seconds=3600, per_process_timeout_seconds=600)
    validate_manifest(m, executable=args.command == "freeze")
    out = Path(args.output_dir).resolve()
    out.mkdir(exist_ok=False)
    write_json(out / "manifest.json", m)
    mh = sha(out / "manifest.json")
    batch = batch_text(mh)
    (out / "campaign.sbatch").write_text(batch)
    write_json(out / "bundle-receipt.json", dict(manifest_sha256=mh, batch_sha256=sha(out / "campaign.sbatch"),
               harness_pin=hp, executable_approval=m["approval"]["state"], generated_only=True))
    print(f"{args.command.upper()} {out} manifest_sha256={mh}; NOT submitted")


def batch_text(manifest_hash):
    require(re.fullmatch("[0-9a-f]{64}", manifest_hash), "manifest digest")
    return f"""#!/usr/bin/env bash
# Generated data; tracked code moves only via Git after separate approval.
#SBATCH --job-name=major2-gpu-ab
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --constraint=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@300
#SBATCH --no-requeue
#SBATCH --chdir={RUN}
#SBATCH --output={RUN}/logs/%x-%j.out
#SBATCH --error={RUN}/logs/%x-%j.err
set -euo pipefail
# logs/ must preexist, no source-root or spool-relative path resolution.
exec bash {HARNESS}/devtools/major2/supervise.sh {RUN} \\
  {PY} -B {HARNESS}/devtools/major2/campaign.py run \\
  --manifest {RUN}/manifest.json --manifest-sha256 {manifest_hash}
"""


def allocation(job_text, env):
    fields = dict(re.findall(r"(\w+)=([^\s]+)", job_text))
    for key, value in dict(Account=ROUTE["account"], QOS="embers", NumCPUs="8",
                           MinMemoryNode="128G", NumNodes="1", NumTasks="1", TimeLimit="04:00:00").items():
        # Slurm versions may normalize 128G to 131072M.
        allowed = {value, "131072M"} if key == "MinMemoryNode" else {value}
        require(fields.get(key) in allowed, "allocation route mismatch: " + key)
    require("gpu-h200" in fields.get("Partition", "").split(","), "no H200 partition")
    require("H200" in fields.get("Features", "") and re.search(r"gres/gpu(?::h200)?=1\b", fields.get("AllocTRES", "")), "H200 GRES/constraint")
    require(env.get("SLURM_CPUS_PER_TASK") == "8" and env.get("SLURM_JOB_ACCOUNT") == ROUTE["account"] and
            env.get("SLURM_JOB_QOS") == "embers" and env.get("SLURM_JOB_NUM_NODES") == "1", "allocation environment")
    start, end = int(env["SLURM_JOB_START_TIME"]), int(env["SLURM_JOB_END_TIME"])
    require(0 < end-start <= 14400, "allocation wall cap")
    return fields


def gpu_line(text):
    rows = [x.strip() for x in text.splitlines() if x.strip()]
    require(len(rows) == 1, "exactly one visible GPU required")
    fields = [x.strip() for x in rows[0].split(",")]
    require(len(fields) == 5 and "H200" in fields[0] and
            re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", fields[1]) and
            130000 <= float(fields[2]) < 200000 and fields[4].isdigit(), "H200/UUID/VRAM/index")
    return dict(name=fields[0], uuid=fields[1], memory_mib=float(fields[2]), driver=fields[3],
                nvml_index=int(fields[4]))


class Engine:
    def __init__(self, m, manifest_path):
        self.m = m
        self.manifest_path = Path(manifest_path)
        self.root = Path(m["run_root"])
        self.raw = self.root / "raw"
        self.sampler = None
        self.active = None
        self.signalled = False
        self.records = []
        self.probes = {}
        self.drivers = {}
        self.executables = {}
        self.configurations = {}
        self.job = os.environ["SLURM_JOB_ID"]
        require(re.fullmatch(r"\d+", self.job), "real job id required")
        self.scratch = Path(m["scratch_parent"]) / self.job
        self.end = int(os.environ["SLURM_JOB_END_TIME"])
        self.gpu = None
        self.scheduler_gpu = self.env_gpu_selector()
        self.gpu_selector = None  # never a scheduler/global numerical-index guess

    @staticmethod
    def env_gpu_selector():
        value = os.environ.get("SLURM_JOB_GPUS", "")
        require(re.fullmatch(r"(?:\d+|GPU-[0-9a-fA-F-]+)", value), "one physical SLURM_JOB_GPUS device required")
        return value

    def stop(self, *_):
        if self.signalled:
            return  # do not interrupt partial-evidence cleanup with a second signal
        self.signalled = True
        # Supervisor forwards to our whole group, including active children.
        if self.active and self.active.poll() is None:
            try:
                os.killpg(self.active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise RuntimeError("TERM/INT/HUP: campaign interrupted, preserve partials")

    def run_command(self, argv, name, env=None, cwd=None, timeout=600, accept_failure=False):
        require(not self.signalled, "signal already received")
        log = self.raw / (name + ".log")
        write_json(self.raw / (name + ".argv.json"), dict(argv=list(map(str, argv)), cwd=str(cwd or self.scratch),
                   timeout=timeout, start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        started = time.monotonic()
        with log.open("xb") as stream:
            proc = subprocess.Popen(list(map(str, argv)), cwd=cwd or self.scratch,
                                    env=env or self.env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            self.active = proc
            try:
                rc = proc.wait(timeout=min(timeout, max(1, self.end-time.time()-240)))
            except BaseException as exc:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                write_json(self.raw / (name + ".exit.json"), dict(exit=proc.returncode,
                           process_wall_s=time.monotonic()-started, interrupted=type(exc).__name__,
                           signalled=self.signalled))
                raise
            finally:
                self.active = None
        write_json(self.raw / (name + ".exit.json"), dict(exit=rc, process_wall_s=time.monotonic()-started))
        require(not self.signalled and (rc == 0 or accept_failure), "command failed: " + name)
        return rc

    def query(self, argv):
        return subprocess.check_output(argv, env=self.env, text=True, timeout=30)

    def identify_gpu(self):
        visible = self.env.get("CUDA_VISIBLE_DEVICES", "")
        require(re.fullmatch(r"(?:\d+|GPU-[0-9a-fA-F-]+)", visible), "one CUDA-visible device")
        helper = self.scratch / "gpu-identity"
        # The very same module initialization and include/compiler isolation as
        # the builds, but only one tiny host translation unit, no dependency build.
        self.run_command(["bash", HERE/"build.sh", P+"/software/toolchains/conda-g++",
                          "-std=c++17", "-O0", "-H", "-I"+CUDA+"/targets/x86_64-linux/include",
                          HERE/"gpu_identity.cc", E+"/lib/libcudart.so",
                          "-Wl,-rpath,"+E+"/lib", "-o", helper],
                         "gpu-identity-compile", timeout=60)
        # A separate executable process cannot prewarm any production energy/J/K/XC.
        self.run_command([helper], "gpu-identity-run", timeout=30)
        proof = read_json(self.raw/"gpu-identity-run.log")
        require(proof["count"] == 1 and proof["logical_device"] == 0, "one logical CUDA device required")
        require(proof["header_version"] == proof["runtime_version"] == 13030,
                "identity CUDA13 header/runtime version mismatch")
        require("H200" in proof["name"] and 130000*1048576 <= proof["memory_bytes"] < 200000*1048576,
                "identity H200/VRAM")
        uuid = proof["uuid"]
        require(isinstance(uuid, str) and
                re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid),
                "invalid CUDA GPU UUID")
        for provenance in [visible, self.scheduler_gpu]:
            if provenance.startswith("GPU-"):
                require(provenance.lower() == uuid.lower(), "visible/scheduler UUID mismatch")
        write_json(self.raw/"gpu-identity.json", proof)
        self.gpu_selector = uuid
        selected = gpu_line(self.query(["nvidia-smi", "--id="+uuid,
                            "--query-gpu=name,uuid,memory.total,driver_version,index",
                            "--format=csv,noheader,nounits"]))
        require(selected["uuid"] == uuid, "CUDA/NVML UUID mismatch")
        self.gpu = selected

    def pins(self):
        for item in self.m["arms"].values():
            check_source(item["source"], item["pin"], item["files"])
        check_source(self.m["harness"]["path"], self.m["harness"]["pin"], self.m["harness"]["files"])
        require(sha(self.manifest_path) == self.manifest_hash, "manifest changed")
        require(sha(Path(self.m["suite"]) / "systems.py") == SUITE_SHA, "geometry module changed")
        require(gpu_line(self.query(["nvidia-smi", "--id="+self.gpu_selector,
                                    "--query-gpu=name,uuid,memory.total,driver_version,index", "--format=csv,noheader,nounits"])) == self.gpu,
                "GPU/UUID/driver changed")
        for p, h in LIBS.items():
            require(sha(p) == h, "shared dependency changed")
        einsums_coherence()
        for arm in self.probes:
            stage = Path(self.m["arms"][arm]["build"]) / "stage"
            require(sha(self.probes[arm]["core"]) == self.probes[arm]["core_sha256"], "core changed")
            require(tree_manifest(stage / "lib/psi4/driver") == self.drivers[arm], "staged driver changed")
            require(sha(stage/"bin/psi4") == self.executables[arm], "staged executable changed")

    def prolog(self):
        self.raw.mkdir()
        self.scratch.mkdir(parents=True, exist_ok=False)
        self.env = runtime_env(os.environ, self.scratch)
        self.env["CUDA_CACHE_PATH"] = str(self.scratch / "cuda-cache")
        Path(self.env["CUDA_CACHE_PATH"]).mkdir()
        env_keys = [
            "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
            "CUDA_CACHE_PATH", "CUDA_CACHE_DISABLE", "CUDA_CACHE_MAXSIZE",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
            "TMPDIR", "SCRATCH", "PSI_SCRATCH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
            "CPKS_CUDA_ARCHS", "CPKS_CUDA_LIBXC", "CPKS_CUEST_SDK", "CPKS_BUILD_DIR",
            "SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS", "SLURM_JOB_PARTITION",
            "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "SLURM_JOB_START_TIME", "SLURM_JOB_END_TIME",
            "SLURM_JOB_NUM_NODES",
        ]
        write_json(self.raw / "environment.json", {key: self.env.get(key) for key in env_keys})
        info = self.query(["scontrol", "show", "job", "-o", self.job])
        (self.raw / "scheduler.txt").write_text(info)
        allocation(info, self.env)
        self.identify_gpu()
        write_json(self.raw / "gpu.json", self.gpu)
        for name, argv in [("lscpu", ["lscpu"]), ("affinity", ["taskset", "-pc", str(os.getpid())]),
                           ("gpu-full", ["nvidia-smi", "--id="+self.gpu_selector, "-q"]),
                           ("gpu-tenants", ["nvidia-smi", "--id="+self.gpu_selector,
                            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"])]:
            self.run_command(argv, name, timeout=30)
        self.pins()
        for arm, item in self.m["arms"].items():
            require(not Path(item["build"]).exists() and not Path(item["build"]).is_symlink(), "new build root already exists: " + arm)
            require(not (self.root / ("build-" + arm)).exists(), "new build metadata root exists")
        selectors = selectors_ast(self.m["arms"]["candidate"]["source"])
        require(selectors == self.m["selectors_ast"], "test selector source changed")
        write_json(self.raw / "selectors-ast.json", selectors)
        # Record read-only toolchain/config/library inputs. Their runtime hashes must
        # remain identical through the allocation, including after both builds.
        paths = [
            INSPECTION_TOOL["path"],
            P + "/software/toolchains/conda-gcc", P + "/software/toolchains/conda-g++",
            CUDA + "/bin/nvcc", E + "/bin/python", E + "/bin/cmake", E + "/bin/ninja",
            XC + "/lib64/cmake/Libxc/LibxcConfig.cmake",
            P + "/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage/share/cmake/Einsums/EinsumsConfig.cmake",
        ]
        libint_configs = sorted(Path(E + "/lib/cmake/libint2").glob("*.cmake"))
        require(libint_configs, "Libint2 config directory missing")
        paths += list(map(str, libint_configs))
        self.inputs = {p: sha(p) for p in paths}
        require(self.inputs[INSPECTION_TOOL["path"]] == INSPECTION_TOOL["sha256"], "inspection tool changed")
        write_json(self.raw / "inspection-tool.json", INSPECTION_TOOL)
        self.inputs.update({DEPS + "/" + p: sha(Path(DEPS) / p) for p in EINSUMS_INPUTS})
        write_json(self.raw / "einsums-coherence.json", einsums_coherence())
        write_json(self.raw / "toolchain-inputs.json", self.inputs)

    def arm_command(self, arm, argv):
        stage = Path(self.m["arms"][arm]["build"]) / "stage"
        runtime = self.root / ("build-" + arm) / "runtime.env"
        return ["bash", str(HERE / "arm.sh"), str(stage), str(runtime), *map(str, argv)]

    def build(self, arm, remaining_builds):
        needed = remaining_builds*3600 + 1800 + 3600
        admission(self.end-time.time(), needed)
        self.pins()
        item = self.m["arms"][arm]
        out = self.root / ("build-" + arm)
        out.mkdir()
        self.run_command(["bash", HERE/"build.sh",
                          *build_argv(item["source"], item["pin"], out)], "build-" + arm, timeout=3600)
        require((out / "BUILD_OK").read_text().strip() == item["pin"] + " cuda", "BUILD_OK pin")
        inner = Path(item["build"]) / "psi4-core-prefix/src/psi4-core-build"
        cache = (inner / "CMakeCache.txt").read_text()
        stage = Path(item["build"]) / "stage"
        self.configurations[arm] = cmake_policy((Path(item["build"])/"CMakeCache.txt").read_text(), cache, stage)
        write_json(self.raw/(arm+"-cmake-policy.json"), self.configurations[arm])
        for token in ["PSI4_CUEST_HAS_NONSYMMETRIC_XC:INTERNAL=1", "CMAKE_BUILD_TYPE:STRING=Release",
                      "CMAKE_INSTALL_PREFIX:PATH=" + str(stage)]:
            require(token in cache, "inner CMake gate: " + token)
        require("Einsums_DIR:" in cache and DEPS + "/share/cmake/Einsums" in cache,
                "inner Einsums configuration prefix")
        require(DEPS + "/lib/libEinsums.so.1.1.3" in (inner / "build.ninja").read_text(),
                "new core must link the matching Einsums imported library")
        write_json(self.raw / (arm + "-runtime-selection.json"),
                   dict(helper_runtime_env=(out / "runtime.env").read_text(),
                        helper_runtime_sha256=sha(out / "runtime.env"),
                        helper_ldd=(out / "ldd.txt").read_text(),
                        effective_ld_library_path=RUNTIME_PATH,
                        note="Helper ldd precedes arm override; actual tested maps are in probes."))
        self.run_command(self.arm_command(arm, ["ldd", *stage.glob("lib/psi4/core*.so")]),
                         arm + "-effective-ldd", timeout=30)
        for arch in ["sm_80", "sm_89", "sm_90", "sm_120"]:
            require(arch in (out / "core_sass.txt").read_text().split(), "missing embedded SASS")
        abi = read_json(out / "abi-after-build/result.json")
        require(abi["ok"] and abi["no_gpu_calls"], "CUDA13 ABI probe")
        require(abi["cudart_version"] == 13030, "CUDA13 header macro evidence absent")
        self.run_command(self.arm_command(arm, [PY, "-B", HERE / "probe.py", self.raw / (arm + "-probe.json")]),
                         arm + "-probe", timeout=60)
        self.probes[arm] = read_json(self.raw / (arm + "-probe.json"))
        self.executables[arm] = sha(stage/"bin/psi4")
        self.drivers[arm] = tree_manifest(stage / "lib/psi4/driver")
        expected = {k.removeprefix("psi4/driver/"): v["sha256"] for k, v in item["files"].items()
                    if k.startswith("psi4/driver/") and k.endswith(".py")}
        require(self.drivers[arm] == expected, "staged drivers not exact merged source")
        write_json(self.raw / (arm + "-drivers.json"), self.drivers[arm])
        write_json(self.raw / (arm + "-stage.json"), {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*"))
                   if p.is_file()})
        # Complete cache/compile database retained in the new build, additionally hashed.
        write_json(self.raw / (arm + "-build-files.json"),
                   {str(p): sha(p) for p in [Path(item["build"])/"CMakeCache.txt", inner/"CMakeCache.txt",
                                            inner/"compile_commands.json", stage/"bin/psi4"]})
        for src, name in [(Path(item["build"])/"CMakeCache.txt", "CMakeCache.outer.txt"),
                          (inner/"CMakeCache.txt", "CMakeCache.inner.txt"),
                          (inner/"compile_commands.json", "compile_commands.json")]:
            shutil.copy2(src, out/name)
        self.pins()
        require(self.inputs == {p: sha(p) for p in self.inputs}, "toolchain/config changed during build")

    def tests(self, arm, label, selectors, expression=None, red=False):
        admission(self.end-time.time(), 3600 + 600)
        self.pins()
        directory = self.raw / label
        directory.mkdir()
        temp = self.scratch / label
        temp.mkdir()
        env = runtime_env(self.env, temp)
        # Plugin-only harness path, never a source package or old driver overlay.
        # arm.sh clears PYTHONPATH; pytest's --pyargs is not used.
        env["MAJOR2_PYTEST_EVIDENCE"] = str(directory / "execution.json")
        expected_count = 1 if red else (4 if expression == FIXED_K else (48 if selectors == COMMON else 16))
        env["MAJOR2_EXPECTED_COUNT"] = str(expected_count)
        env["MAJOR2_PROCESS_PROBE"] = str(directory / "process-probe.json")
        argv = [PY, "-B", HERE / "run_pytest.py", HERE, "--", "-c",
                Path(self.m["arms"]["candidate"]["source"]) / "pytest.ini",
                "-q", "-ra", "-x", "--maxfail=1", "--tb=short", "-p", "no:cacheprovider",
                "--basetemp=" + str(temp / "pytest"), "--junitxml=" + str(directory / "junit.xml"),
                *[str(Path(self.m["arms"]["candidate"]["source"]) / "tests/pytests" / s) for s in selectors]]
        if expression:
            argv += ["-k", expression]
        rc = self.run_command(self.arm_command(arm, argv), label, env=env, cwd=temp, timeout=600, accept_failure=red)
        evidence = read_json(directory / "execution.json")
        require(evidence["exitstatus"] == rc, "pytest exit receipt")
        verdict = check_junit(directory / "junit.xml", evidence["collected"], rc, red)
        require(set(evidence["phases"]) == set(evidence["collected"]), "collected test did not execute")
        require(all(p.get("setup") == p.get("teardown") == "passed" and
                    p.get("call") == ("failed" if red else "passed") for p in evidence["phases"].values()),
                "required test phase skipped/error/xfail")
        require(len(evidence["collected"]) <= 200, "unbounded correctness selector expansion")
        require(len(evidence["collected"]) == expected_count, "required test matrix collection count changed; review before execution")
        write_json(directory / "VERIFIED.json", verdict)

    def start_sampler(self):
        self.sampler = subprocess.Popen([PY, "-B", str(HERE / "sample.py"),
                                        str(self.raw / "telemetry.jsonl"), self.gpu["uuid"]],
                                       env=self.env, cwd=self.scratch)

    def case(self, row):
        self.pins()
        admission(self.end-time.time(), 600)
        arm, label = row["arm"], row["label"]
        item = self.m["arms"][arm]
        out = self.raw / "sessions" / label
        scratch = self.scratch / label
        scratch.mkdir()
        before = self.raw / (label + ".before.json")
        after = self.raw / (label + ".maps.json")
        argv = [PY, "-P", HERE / "execute_runner.py", before, after,
                Path(item["source"]) / "devtools/saptdft_cpks_suite_run.py",
                "--arm", "gpu", "--suite", self.m["suite"], "--system", row["system"],
                "--output", out, "--threads", "8", "--memory", "100 GiB",
                "--initial", "35x194", "--final", "75x302", "--factor", "1", "--minimum", "2",
                "--source-commit", item["pin"]]
        if row["smoke"]:
            argv.append("--smoke")
        # no shell interpolation of chemistry argv
        meta = dict(row, source_pin=item["pin"], manifest_sha256=self.manifest_hash,
                    job=self.job, host=socket.gethostname(), gpu=self.gpu,
                    scratch=str(scratch), stage=item["build"]+"/stage",
                    sampler_start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(self.raw / (label + ".meta.json"), meta)
        env = runtime_env(self.env, scratch)
        env["CUDA_CACHE_PATH"] = self.env["CUDA_CACHE_PATH"]
        self.run_command(["/usr/bin/time", "-v", "-o", str(self.raw / (label + ".time.txt")),
                          *self.arm_command(arm, argv)], label, env=env, cwd=scratch, timeout=600)
        d = read_json(out / "result.json")
        geometry_source = literal_assignments(Path(self.m["suite"]) / "systems.py", ["SYSTEMS"])["SYSTEMS"]
        runner = literal_assignments(Path(item["source"]) / "devtools/saptdft_cpks_suite_run.py", ["SCIENCE", "WATER_DIMER"])
        geom = runner["WATER_DIMER"] if row["smoke"] else geometry_source[row["system"]]
        options = {**runner["SCIENCE"], "USE_CUEST": True, "CUEST_XC": True, "CUEST_SAD": True,
                   "DFT_RADIAL_POINTS": 75, "DFT_SPHERICAL_POINTS": 302,
                   "DFT_RADIAL_POINTS_INITIAL": 35, "DFT_SPHERICAL_POINTS_INITIAL": 194, "DFT_GRID_SWITCH_FACTOR": 1.0}
        stage = Path(item["build"]) / "stage"
        expected = dict(source_commit=item["pin"], runner_sha256=RUNNER_SHA, suite_systems_sha256=SUITE_SHA,
                        geometry_sha256=hashlib.sha256(geom.encode()).hexdigest(), slurm_job_id=self.job,
                        hostname=socket.gethostname(), system="water_dimer_smoke" if row["smoke"] else row["system"],
                        suite_system=row["system"], psi4_module=self.probes[arm]["module"],
                        method="sapt(dft)-d4(i)", basis="6-31+g**",
                        science_options_sha256=hashlib.sha256(json.dumps(runner["SCIENCE"], sort_keys=True).encode()).hexdigest(),
                        grid_policy=dict(initial="35x194", final="75x302", switch_factor=1.0, minimum_final_iterations=2),
                        policy_checks=dict(problems=[], switched_scfs=2),
                        iterator=dict(path=str(stage/"lib/psi4/driver/procrouting/scf_proc/scf_iterator.py"),
                                      sha256=self.drivers[arm]["procrouting/scf_proc/scf_iterator.py"]))
        if not row["smoke"]:
            sizes = SYSTEMS[row["system"]]
            expected.update(zip(["natom", "nbf", "naux_jkfit", "naux_ri", "nbf_monomer_a", "nbf_monomer_b"], sizes[:6]))
            require(expected["geometry_sha256"] == sizes[6], "geometry identity")
        else:
            expected.update(natom=6, nbf=56, naux_jkfit=226, naux_ri=236, nbf_monomer_a=28, nbf_monomer_b=28)
        verdict = science.validate_record(d, (out/"psi4.out").read_text(), (out/"timer.dat").read_text(),
                                         (out/"profile.txt").read_text(), expected, options, geom,
                                         candidate=arm == "candidate")
        actual_maps = read_json(after)
        before_probe = read_json(before)
        require(before_probe["core_sha256"] == self.probes[arm]["core_sha256"], "process imported other core")
        require(all(actual_maps.get(p) == h for p, h in d["libraries_sha256"].items()), "runner/library mapping hash disagreement")
        require(str(Path(self.probes[arm]["core"]).resolve()) in d["libraries_sha256"], "runner core mapping absent")
        check_runtime_maps(actual_maps)
        for prefix in ("libcudart.so", "libmkl_rt.so", "libstdc++.so"):
            found = [p for p in actual_maps if Path(p).name.startswith(prefix)]
            require(found and all(Path(p).is_relative_to(E + "/lib") for p in found), "actual runtime " + prefix)
        other = self.m["arms"]["candidate" if arm == "baseline" else "baseline"]["build"]
        require(not any(p.startswith(other) for p in actual_maps), "other-arm runtime mapped")
        if not row["smoke"]:
            telemetry = [json.loads(line) for line in (self.raw/"telemetry.jsonl").read_text().splitlines()]
            require(any(any(str(app.get("pid")) == str(before_probe["pid"]) and app.get("gpu_uuid") == self.gpu["uuid"]
                            for app in sample.get("compute_apps", [])) for sample in telemetry),
                    "actual chemistry PID not observed on allocated UUID")
        self.pins()
        require(self.sampler is None or self.sampler.poll() is None, "production sampler stopped")
        write_json(self.raw / (label + ".VERIFIED.json"), dict(verdict, end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        return d

    def inventory(self):
        # Independent second pass verifies every new raw byte, never scans old archives.
        def collect():
            roots = [self.raw, self.root/"build-baseline", self.root/"build-candidate"]
            return {str(p.relative_to(self.root)): dict(sha256=sha(p), bytes=p.stat().st_size)
                    for root in roots for p in sorted(root.rglob("*")) if p.is_file()}
        inventory = collect()
        write_json(self.root / "raw-inventory.json", inventory)
        require(inventory == collect(), "raw inventory changed")
        write_json(self.root / "raw-inventory-recheck.json", dict(sha256=sha(self.root/"raw-inventory.json"),
                                                                files=len(inventory), verified=True))

    def execute(self):
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, self.stop)
        self.manifest_hash = sha(self.manifest_path)
        require(not (self.root/"STARTED.json").exists(), "campaign cannot resume/retry")
        allowed = {"manifest.json", "campaign.sbatch", "bundle-receipt.json", "logs"}
        require(set(p.name for p in self.root.iterdir()) <= allowed and
                not any(p.is_symlink() for p in self.root.iterdir()), "new run root contains unrelated/prior data")
        write_json(self.root/"STARTED.json", dict(job=self.job, manifest_sha256=self.manifest_hash))
        try:
            self.prolog()
            self.build("baseline", 2)
            self.build("candidate", 1)
            require(self.drivers["baseline"] == self.drivers["candidate"], "merged drivers must match exactly")
            require(self.configurations["baseline"] == self.configurations["candidate"], "build policies differ")
            for arm in ("baseline", "candidate"):
                self.tests(arm, arm+"-common", COMMON)
                self.tests(arm, arm+"-hf-pbe0-fixed", [FIXED], FIXED_K)
            self.tests("baseline", "baseline-expected-red", [RED], red=True)
            self.tests("candidate", "candidate-focused-plan", [PLAN])
            (self.raw / "sessions").mkdir()
            smoke = {}
            for arm in ("baseline", "candidate"):
                smoke[arm] = self.case(dict(arm=arm, backend="gpu", label="major2-smoke-"+arm,
                                           system="peptide", repeat=0, smoke=True))
            science.compare(smoke["baseline"], smoke["candidate"])
            admission(self.end-time.time(), 3600)
            self.start_sampler()
            seen = {}
            for offset in range(0, 24, 2):
                admission(self.end-time.time(), max(1200, 3600 - offset*125))
                pair = []
                for row in self.m["sequence"][offset:offset+2]:
                    d = self.case(row)
                    pair.append(d)
                    self.records.append(dict(row, data=d))
                    if row["system"] in seen:
                        science.compare(seen[row["system"]], d)
                    else:
                        seen[row["system"]] = d
                science.compare(*pair)  # first numerical/iteration change stops further production
            self.sampler.terminate()
            require(self.sampler.wait(timeout=15) == 0, "sampler failed")
            self.sampler = None
            self.pins()
            require(self.inputs == {p: sha(p) for p in self.inputs}, "toolchain inputs changed")
            write_json(self.root/"gpu-ab-summary.json", science.summary(self.records))
        except BaseException as exc:
            write_json(self.root/"ENGINE_FAILED.json", dict(error=f"{type(exc).__name__}: {exc}",
                       signalled=self.signalled, production_records=len(self.records)))
            raise
        finally:
            if self.sampler and self.sampler.poll() is None:
                self.sampler.terminate()
                self.sampler.wait(timeout=15)
            if self.raw.exists():
                self.inventory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("template", "freeze"):
        p = sub.add_parser(mode)
        p.add_argument("--baseline-local", required=True)
        p.add_argument("--candidate-local", required=True)
        p.add_argument("--output-dir", required=True)
        if mode == "freeze":
            p.add_argument("--approved-candidate-pin", required=True)
            p.add_argument("--review-id", required=True)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command in ("template", "freeze"):
        prepare(args)
    else:
        require(sha(args.manifest) == args.manifest_sha256, "frozen manifest SHA mismatch")
        m = read_json(args.manifest)
        validate_manifest(m)
        require(Path(args.manifest) == Path(RUN)/"manifest.json", "manifest outside new cohort root")
        Engine(m, args.manifest).execute()


if __name__ == "__main__":
    main()
