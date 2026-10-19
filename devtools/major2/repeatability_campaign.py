"""Isolated six-fresh-process diagnostic. No builds, no primary acceptance, no timing report."""
import argparse
import datetime
import itertools
import os
from pathlib import Path
import re
import signal
import sys
import time

import numpy as np

import artifacts as a
import campaign as c
import handoff as h
import observe as o
import policy as p
import repeatability as r
import science

HERE = Path(__file__).resolve().parent
IDENTITY = r.IDENTITY


def batch_text(manifest_hash):
    # Exact previously reviewed route/supervisor; only new harness/run/entrypoint.
    return c.batch_text(manifest_hash, hybrid=True).replace(c.n.RUN, r.RUN).replace(
        c.n.HARNESS, r.HARNESS).replace("devtools/major2/campaign.py run",
                                     "devtools/major2/repeatability_campaign.py run")


def freeze(args):
    payload = p.read_json(args.collection)
    pair = payload["pair"]
    r.validate_pair(pair)
    local = dict(baseline=Path(args.baseline_local), candidate=Path(args.candidate_local))
    arms = {}
    for arm in r.ALLOWLIST:
        item = r.ALLOWLIST[arm]
        p.check_source(local[arm], item["pin"])
        files = p.file_manifest(local[arm])
        p.require(p.digest(files) == pair["arms"][arm]["source_files_digest"], "local original full source")
        arms[arm] = dict(source=item["source"], build=item["build"], pin=item["pin"], files=files)
    harness = HERE.parents[1]
    pin = p.git(harness, "rev-parse", "HEAD")
    p.check_source(harness, pin)
    p.require(p.git(harness, "merge-base", pin, r.PARENT_PIN) == r.PARENT_PIN, "new harness ancestry")
    additions = p.git(harness, "diff", "--name-status", r.PARENT_PIN, pin).splitlines()
    p.require(additions and all(x.startswith("A\tdevtools/major2/") for x in additions), "add-only diagnostic")
    files = p.file_manifest(harness)
    m = dict(schema=1, mode=r.MODE, approval=r.REVIEW_ID, diagnostic_pair=pair,
        run_root=r.RUN, route=p.ROUTE, gates=p.GATES, suite=p.H0+"/geometries", suite_sha256=p.SUITE_SHA,
        scratch_parent="/storage/scratch1/3/awallace43/cpks-major2-p157", arms=arms,
        harness=dict(path=r.HARNESS, pin=pin, files=files, files_digest=p.digest(files)),
        sequence=r.sequence(), matrix_cap_bytes=r.MATRIX_CAP, build_budget_seconds=[0, 0],
        per_process_timeout_seconds=1800, accepted_speedup_data=False, identity_helper=IDENTITY,
        preservation=payload["preservation"])
    r.validate(m)
    out = Path(args.output_dir)
    out.mkdir(exist_ok=False)
    o.atomic(out/"manifest.json", m)
    mh = p.sha(out/"manifest.json")
    h.durable(out/"campaign.sbatch", batch_text(mh))
    o.atomic(out/"bundle-receipt.json", dict(manifest_sha256=mh, batch_sha256=p.sha(out/"campaign.sbatch"),
                                           harness_pin=pin, generated_only=True, no_submission=True))
    print("FROZEN numerical diagnostic; no submission; raw timings blinded")


def observations(raw, row):
    root = raw/(row["label"]+".before.matrices")
    value = p.read_json(root/"COMPLETE.json")
    p.require(value["pid"] == p.read_json(raw/(row["label"]+".before.json"))["pid"], "observation PID")
    p.require([x["stage"] for x in value["stages"]] == science.STAGES, "observation stage count/order")
    p.require(value["finalized"] == list(range(5)), "all five native finalize calls")
    p.require(value["cap_bytes"] == r.PROCESS_CAP and value["stored_bytes"] <= r.PROCESS_CAP, "process storage")
    for index, (stage, proof) in enumerate(zip(value["stages"], row["data"]["proofs"])):
        p.require(stage["calls"] == stage["native_calls"] == stage["native_returns"] == proof["iterations"]+1,
                  "actual form_G count does not match SCF trajectory")
        p.require(stage["functional"] == proof["functional"] and stage["nbf"] == proof["nbf"] and
                  stage["pid"] == value["pid"] and stage["iteration"] == 0 and stage["sad"], "class/stage/SAD stamp")
        file = root/stage["file"]
        p.require(file.name == f"stage-{index}.npz" and p.sha(file) == stage["file_sha256"], "snapshot bytes")
        with np.load(file, allow_pickle=False) as arrays:
            p.require(set(arrays.files) == {"Da", "J", "K"}, "matrix set")
            for name in arrays.files:
                value_array = arrays[name]
                p.require(o.array_info(value_array) == stage["arrays"][name], "array exact bytes/shape/dtype")
    return value


def matrix_report(raw, records):
    # Called only inside this allocated GPU job, never a login-side analysis CLI.
    p.require(os.environ.get("SLURM_JOB_ID", "").isdigit(), "analysis requires allocation")
    captures = {row["label"]: observations(raw, row) for row in records}
    p.require(len({v["pid"] for v in captures.values()}) == 6, "six fresh process PIDs")
    p.require(sum(v["stored_bytes"] for v in captures.values()) <= r.MATRIX_CAP, "total persistent cap")
    comparisons = []
    for left, right in itertools.combinations(records, 2):
        label_a, label_b = left["label"], right["label"]
        stages = []
        first = None
        for index, stage in enumerate(science.STAGES):
            items = []
            roots = [raw/(label+".before.matrices")/f"stage-{index}.npz" for label in (label_a, label_b)]
            with np.load(roots[0], allow_pickle=False) as aa, np.load(roots[1], allow_pickle=False) as bb:
                for name in ("Da", "J", "K"):
                    hashes = [captures[label]["stages"][index]["arrays"][name]["sha256"] for label in (label_a, label_b)]
                    diff = o.matrix_difference(aa[name], bb[name])
                    p.require((hashes[0] == hashes[1]) == diff["bitwise_equal"], "exact hash/equality mismatch")
                    if not diff["bitwise_equal"] and first is None:
                        first = dict(stage=stage, matrix=name)
                    items.append(dict(matrix=name, hash_a=hashes[0], hash_b=hashes[1], **diff))
            stages.append(dict(stage=stage, matrices=items))
        comparisons.append(dict(a=label_a, b=label_b, within_arm=left["arm"] == right["arm"],
                                first_divergence=first, stages=stages))
    return dict(comparisons=comparisons, total_stored_bytes=sum(v["stored_bytes"] for v in captures.values()))


class Engine(c.Engine):
    """Reuses reviewed command/TERM/probe/math verification, not primary orchestration."""
    def identify_gpu(self):
        # Reuse the already compiled CUDA13 identity executable; never compile.
        p.require(self.m["identity_helper"] == IDENTITY and p.sha(IDENTITY["path"]) == IDENTITY["sha256"],
                  "original identity executable missing/changed: STOP, no rebuild")
        self.run_command([IDENTITY["path"]], "gpu-identity-run", timeout=30)
        proof = p.read_json(self.raw/"gpu-identity-run.log")
        p.require(proof["count"] == 1 and proof["logical_device"] == 0 and
                  proof["header_version"] == proof["runtime_version"] == 13030, "CUDA13 identity")
        uuid = proof["uuid"]
        p.require("H200" in proof["name"] and 130000*1048576 <= proof["memory_bytes"] < 200000*1048576 and
                  re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid), "H200 identity")
        visible = self.env.get("CUDA_VISIBLE_DEVICES", "")
        p.require(re.fullmatch(r"(?:\d+|GPU-[0-9a-fA-F-]+)", visible), "one visible device")
        for value in (visible, self.scheduler_gpu):
            if value.startswith("GPU-"):
                p.require(value.lower() == uuid.lower(), "CUDA/scheduler device")
        self.gpu_selector = uuid
        self.gpu = c.gpu_line(self.query(["nvidia-smi", "--id="+uuid,
            "--query-gpu=name,uuid,memory.total,driver_version,index", "--format=csv,noheader,nounits"]))
        p.require(self.gpu["uuid"] == uuid, "NVML UUID")
        o.atomic(self.raw/"gpu-identity.json", proof)

    def pins(self):
        super().pins()
        r.verify_shared(self.m["diagnostic_pair"])

    def prolog(self):
        self.raw.mkdir()
        self.scratch.mkdir(parents=True, exist_ok=False)
        self.env = p.runtime_env(os.environ, self.scratch)
        p.require(not any(k.startswith(("CUBLAS_", "CUDNN_", "NVIDIA_TF32")) or
                          k == "CUDA_LAUNCH_BLOCKING" for k in self.env),
                  "unexpected numerical/determinism environment knob; STOP, never change it")
        self.env["CUDA_CACHE_PATH"] = str(self.scratch/"cuda-cache")
        Path(self.env["CUDA_CACHE_PATH"]).mkdir()
        info = self.query(["scontrol", "show", "job", "-o", self.job])
        h.durable(self.raw/"scheduler.txt", info)
        c.allocation(info, self.env)
        o.atomic(self.raw/"environment.json", {k: v for k, v in self.env.items()
            if k.startswith(("SLURM_", "CUDA_", "OMP_", "MKL_", "OPENBLAS_", "CPKS_")) or
            k in ("LD_LIBRARY_PATH", "PATH", "TMPDIR", "SCRATCH", "PSI_SCRATCH", "PYTHONPATH", "PYTHONNOUSERSITE")})
        self.identify_gpu()
        self.pins()
        p.admission(self.end-time.time(), 6*1800+1200)
        pair = self.m["diagnostic_pair"]
        self.inputs = dict(pair["shared_inputs"])
        self.inputs.update(p.read_json(c.n.RUN+"/raw/toolchain-inputs.json"))
        p.require(all(p.sha(path) == sha for path, sha in self.inputs.items()), "toolchain proof")
        # Full current source/stage/config/ABI/SASS/143driver verification on GPU host.
        for arm, item in self.m["arms"].items():
            r.verify_artifact(pair, arm, item["files"])
            origin = pair["arms"][arm]
            out = self.root/("build-"+arm)
            out.mkdir()
            for rel, sha in origin["metadata_files"].items():
                src, dst = Path(origin["metadata"])/rel, out/rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                p.require(p.sha(src) == sha, "origin changed during copy")
                with dst.open("xb") as stream:
                    stream.write(src.read_bytes())
            # Prior candidate has BUILD_ORIGIN; preserve it, write new diagnostic receipt separately.
            o.atomic(out/"DIAGNOSTIC_ORIGIN.json", dict(origin_job=origin["origin_job"],
                compiled_in_this_job=False, original_metadata=origin["metadata"], copied_receipts=True,
                pair_digest=r.PAIR_DIGEST, compiled_source_pin=origin["pin"], core_sha256=origin["core_sha256"]))
            self.verify_build(arm)  # unchanged config/import/ABI/driver tail, no build
            p.require(self.probes[arm]["core_sha256"] == origin["core_sha256"] and
                      self.probes[arm]["pyeinsums_core"] == pair["pyeinsums_core"]["path"] and
                      self.probes[arm]["pyeinsums_core_sha256"] == pair["pyeinsums_core"]["sha256"], "actual imported pair")
            self.run_command([p.INSPECTION_TOOL["path"], "-lelf", self.probes[arm]["core"]], arm+"-sass", timeout=30)
            p.require(sorted(set(re.findall(r"sm_[0-9]+", (self.raw/(arm+"-sass.log")).read_text()))) ==
                      sorted((out/"core_sass.txt").read_text().split()), "actual SASS")
        p.require(self.drivers["baseline"] == self.drivers["candidate"] and
                  self.configurations["baseline"] == self.configurations["candidate"], "pair driver/config mismatch")
        # Validate the existing 122 positive gates + specific expected-red, never rerun chemistry on login.
        count = 0
        for label, expected, red in [("baseline-common", 48, False), ("candidate-common", 48, False),
                ("baseline-hf-pbe0-fixed", 4, False), ("candidate-hf-pbe0-fixed", 4, False),
                ("candidate-focused-plan", 18, False), ("baseline-expected-red", 1, True)]:
            directory = Path(c.n.RUN)/"raw"/label
            execution = p.read_json(directory/"execution.json")
            p.require(len(execution["collected"]) == expected, "original gate count")
            p.check_junit(directory/"junit.xml", execution["collected"], execution["exitstatus"], red)
            p.require(set(execution["phases"]) == set(execution["collected"]) and all(
                row.get("setup") == row.get("teardown") == "passed" and row.get("call") == ("failed" if red else "passed")
                for row in execution["phases"].values()), "original gate phases")
            count += 0 if red else expected
        p.require(count == 122, "122 positive gates")
        o.atomic(self.raw/"pair-VERIFIED.json", dict(pair_digest=r.PAIR_DIGEST,
                 full_same_allocation_verification=True, existing_positive_gates=122, no_compilation=True,
                 numerical_environment_overrides={}))

    def arm_command(self, arm, argv):
        argv = list(argv)
        target = str(c.HERE/"execute_runner.py")
        if len(argv) > 2 and str(argv[2]) == target:
            argv[2] = HERE/"observe_runner.py"
        return super().arm_command(arm, argv)

    def run_command(self, argv, name, **kwargs):
        if name in {row["label"] for row in r.sequence()}:
            kwargs["timeout"] = 1800
        return super().run_command(argv, name, **kwargs)

    def case(self, row):
        p.admission(self.end-time.time(), 1800)
        d = super().case(row)  # original record math/options/residual/runtime/PID checks unchanged
        observation = observations(self.raw, dict(row, data=d))
        o.atomic(self.raw/(row["label"]+".NUMERICAL.json"), dict(label=row["label"],
            matrix_receipt_sha256=p.sha(self.raw/(row["label"]+".before.matrices/COMPLETE.json")),
            pid=observation["pid"], trajectory=r.trajectory(
                (self.raw/"sessions"/row["label"]/"psi4.out").read_text())))
        return d

    def execute(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self.stop)
        self.manifest_hash = p.sha(self.manifest_path)
        allowed = {"manifest.json", "campaign.sbatch", "bundle-receipt.json", "logs"}
        p.require(set(x.name for x in self.root.iterdir()) <= allowed and
                  not any(x.is_symlink() for x in self.root.iterdir()), "new diagnostic root only; no resume")
        o.atomic(self.root/"STARTED.json", dict(job=self.job, manifest_sha256=self.manifest_hash, mode=r.MODE))
        try:
            self.prolog()
            (self.raw/"sessions").mkdir()
            self.start_sampler()
            for row in r.sequence():
                d = self.case(row)
                self.records.append(dict(row, data=d))
                # Individual component gates remain FATAL. Retain all finalized records before comparison.
                o.atomic(self.raw/(row["label"]+".RETAINED.json"), dict(label=row["label"],
                    record_sha256=p.sha(self.raw/"sessions"/row["label"]/"result.json"), numerical_diagnostic=True))
                for previous in self.records[:-1]:
                    for key in science.COMPONENTS:
                        p.require(abs(previous["data"]["components_hartree"][key]-d["components_hartree"][key])
                                  <= p.GATES["component_abs_eh"], "fatal diagnostic component difference")
            self.sampler.terminate()
            p.require(self.sampler.wait(timeout=15) == 0, "sampler failed")
            self.sampler = None
            numerical = r.numerical_report(self.records)
            numerical["matrix_analysis"] = matrix_report(self.raw, self.records)
            numerical["trajectories"] = {row["label"]: p.read_json(
                self.raw/(row["label"]+".NUMERICAL.json"))["trajectory"] for row in self.records}
            for comparison in numerical["comparisons"]:
                comparison["first_printed_trajectory_difference"] = r.first_trajectory_difference(
                    numerical["trajectories"][comparison["a"]], numerical["trajectories"][comparison["b"]])
            self.pins()
            for arm, item in self.m["arms"].items():
                r.verify_artifact(self.m["diagnostic_pair"], arm, item["files"])
            p.require(all(p.sha(path) == sha for path, sha in self.inputs.items()), "toolchain changed")
            o.atomic(self.root/"numerical-repeatability.json", numerical)
        except BaseException as exc:
            # No log/timer content in reports; a fatal error is never an accepted partial cohort.
            h.durable(self.raw/"engine-error.txt", f"{type(exc).__name__}: {exc}\n")
            o.atomic(self.root/"ENGINE_FAILED.json", dict(error_type=type(exc).__name__,
                error_detail="See blinded raw/engine-error.txt; no timing values copied into report",
                signalled=self.signalled, finalized_diagnostic_records=len(self.records), accepted_speedup_data=False))
            raise
        finally:
            if self.sampler and self.sampler.poll() is None:
                self.sampler.terminate()
                self.sampler.wait(timeout=15)
            if self.raw.exists():
                self.inventory()  # raw timer/result bytes retained unmodified, hashes only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    freeze_parser = sub.add_parser("freeze")
    for key in ("collection", "baseline-local", "candidate-local", "output-dir"):
        freeze_parser.add_argument("--"+key, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args)
    else:
        p.require(Path(args.manifest) == Path(r.RUN)/"manifest.json" and
                  p.sha(args.manifest) == args.manifest_sha256, "frozen diagnostic manifest")
        m = p.read_json(args.manifest)
        r.validate(m)
        Engine(m, args.manifest).execute()


if __name__ == "__main__":
    main()
