"""Allocation-only whole cohort. No scientific reruns/builds/observer/sampler."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import config as s
import campaign as c
import policy as p
import repeatability as r
import repeatability_campaign as rc
import science
from trace import parse, timer_regions


def allocation(info, env):
    """Original route guard with explicitly preregistered six-hour limit only."""
    fields = dict(re.findall(r"(\w+)=([^\s]+)", info))
    expected = dict(Account="gts-cs207-chemx",QOS="embers",NumCPUs="8",
        MinMemoryNode="128G",NumNodes="1",NumTasks="1",TimeLimit="06:00:00")
    for key,value in expected.items():
        s.require(fields.get(key) in ({value,"131072M"} if key == "MinMemoryNode" else {value}),
                  "allocation route: "+key)
    s.require("gpu-h200" in fields.get("Partition","").split(",") and
              "H200" in fields.get("Features","") and
              re.search(r"gres/gpu(?::h200)?=1\b",fields.get("AllocTRES","")), "H200 route")
    s.require(env.get("SLURM_CPUS_PER_TASK") == "8" and
              env.get("SLURM_JOB_ACCOUNT") == "gts-cs207-chemx" and
              env.get("SLURM_JOB_QOS") == "embers" and
              env.get("SLURM_JOB_NUM_NODES") == "1" and
              0 < int(env["SLURM_JOB_START_TIME"]) < int(env["SLURM_JOB_END_TIME"]), "route ENV")
    return fields


def deadline(*_):
    raise RuntimeError("whole phase budget exhausted; retain partial receipts")


class Engine(rc.Engine):
    def __init__(self, m, path):
        super().__init__(m, path)
        self.end = min(int(os.environ["SLURM_JOB_END_TIME"]),
                       int(os.environ["SLURM_JOB_START_TIME"])+s.ROUTE["seconds"])
        self.manifest_hash = s.sha(path)

    # Crucially, bypass rc.arm_command: it substitutes the RHF matrix observer.
    arm_command = c.Engine.arm_command

    def pins(self):
        s.n.check_env(os.environ)
        super().pins()  # exact source, drivers, libraries, runtime, pair shared guards
        s.source_check(self.m)
        compiler = self.m["compiler"]
        s.require(str(Path(compiler["actual"]).resolve(strict=True)) == compiler["actual_resolved"],
                  "compiler alias target")
        extra = dict(self.m["sdk_headers"], **self.m["native_libraries"])
        extra.update({compiler["actual"]:compiler["actual_sha256"],
                      compiler["frontend"]:compiler["frontend_sha256"],
                      compiler["wrapper"]:compiler["wrapper_sha256"]})
        for path,digest in extra.items():
            s.require(s.sha(path) == digest, "full toolchain/native identity: "+path)

    def run_command(self, argv, name, env=None, cwd=None, timeout=600, **kwargs):
        s.require(not self.signalled and self.end-time.time() >= timeout+300, "command admission")
        s.durable(self.raw/(name+".argv.json"), dict(argv=list(map(str,argv)),
                  cwd=str(cwd or self.scratch), timeout=timeout))
        started = time.perf_counter_ns()
        with (self.raw/(name+".log")).open("xb") as stream:
            proc = subprocess.Popen(list(map(str,argv)), cwd=cwd or self.scratch,
                env=env or self.env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.active = proc
            try:
                code = proc.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                s.durable(self.raw/(name+".exit.json"), dict(exit=proc.returncode,
                    pid=proc.pid, interrupted=True, process_wall_ns=time.perf_counter_ns()-started))
                raise
            finally:
                self.active = None
        s.durable(self.raw/(name+".exit.json"), dict(exit=code, pid=proc.pid,
            interrupted=False, process_wall_ns=time.perf_counter_ns()-started))
        s.require(code == 0 and not self.signalled, "command failed: "+name)
        s.require((self.raw/(name+".log")).stat().st_size < s.POLICY["output_cap_bytes"], "log cap")
        return proc.pid

    def prolog(self):
        """Owned copy of rc prolog, same guards; new route/budget, no observer."""
        self.raw.mkdir()
        self.scratch.mkdir(parents=True, exist_ok=False)
        self.env = p.runtime_env(os.environ, self.scratch)
        s.n.check_env(self.env)
        self.env["CUDA_CACHE_PATH"] = str(self.scratch/"cuda-cache")
        Path(self.env["CUDA_CACHE_PATH"]).mkdir()
        info = self.query(["scontrol","show","job","-o",self.job])
        s.durable(self.raw/"scheduler.txt", info)
        fields = allocation(info, self.env)
        s.durable(self.raw/"deadline.json", dict(effective_end=self.end,
            controller_time_limit=fields["TimeLimit"], policy_seconds=s.ROUTE["seconds"]))
        s.durable(self.raw/"environment.json", {k:v for k,v in self.env.items()
            if k.startswith(("SLURM_","CUDA_","OMP_","MKL_","OPENBLAS_","CPKS_")) or
            k in ("LD_LIBRARY_PATH","PATH","TMPDIR","SCRATCH","PSI_SCRATCH","PYTHONPATH")})
        self.identify_gpu()  # original compiled CUDA13 helper, never rebuilt
        self.pins()
        pair = self.m["diagnostic_pair"]
        self.inputs = dict(pair["shared_inputs"])
        self.inputs.update(p.read_json(c.n.RUN+"/raw/toolchain-inputs.json"))
        s.require(all(p.sha(path) == h for path,h in self.inputs.items()), "toolchain proof")
        for arm,item in self.m["arms"].items():
            r.verify_artifact(pair, arm, item["files"])  # ABI, configs, SASS, 143 drivers
            origin = pair["arms"][arm]
            out = self.root/("build-"+arm)
            out.mkdir()
            for rel,digest in origin["metadata_files"].items():
                src,dst = Path(origin["metadata"])/rel,out/rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                s.require(p.sha(src) == digest, "origin receipt")
                with dst.open("xb") as stream:
                    stream.write(src.read_bytes())
            s.durable(out/"WHOLE_ORIGIN.json", dict(origin_job=origin["origin_job"],
                compiled_in_this_job=False, pair_digest=r.PAIR_DIGEST,
                compiled_source_pin=origin["pin"], core_sha256=origin["core_sha256"]))
            self.verify_build(arm)  # unchanged original config/import/ABI/driver guard
            s.require(self.probes[arm]["core_sha256"] == origin["core_sha256"] and
                      self.probes[arm]["pyeinsums_core"] == pair["pyeinsums_core"]["path"] and
                      self.probes[arm]["pyeinsums_core_sha256"] == pair["pyeinsums_core"]["sha256"],
                      "actual imported pair")
            self.run_command([p.INSPECTION_TOOL["path"],"-lelf",self.probes[arm]["core"]],
                             arm+"-sass",timeout=30)
            s.require(sorted(set(re.findall(r"sm_[0-9]+",
                (self.raw/(arm+"-sass.log")).read_text()))) ==
                sorted((out/"core_sass.txt").read_text().split()), "actual SASS")
        s.require(self.drivers["baseline"] == self.drivers["candidate"] and
                  self.configurations["baseline"] == self.configurations["candidate"], "pair config/drivers")
        count = 0
        for label,expected,red in [("baseline-common",48,False),("candidate-common",48,False),
                ("baseline-hf-pbe0-fixed",4,False),("candidate-hf-pbe0-fixed",4,False),
                ("candidate-focused-plan",18,False),("baseline-expected-red",1,True)]:
            directory = Path(c.n.RUN)/"raw"/label
            execution = p.read_json(directory/"execution.json")
            s.require(len(execution["collected"]) == expected, "original gate count")
            p.check_junit(directory/"junit.xml",execution["collected"],execution["exitstatus"],red)
            s.require(set(execution["phases"]) == set(execution["collected"]) and all(
                v.get("setup") == v.get("teardown") == "passed" and
                v.get("call") == ("failed" if red else "passed")
                for v in execution["phases"].values()), "original gate phases")
            count += 0 if red else expected
        s.require(count == 122, "122 positive existing gates")
        s.durable(self.raw/"pair-VERIFIED.json", dict(pair_digest=r.PAIR_DIGEST,
            existing_positive_gates=122,no_compilation=True,full_same_allocation_verification=True))

    def case(self, row):
        arm,label = row["arm"],row["label"]
        self.pins()
        s.require(not self.query(["nvidia-smi","--id="+self.gpu_selector,
            "--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"]).strip(),
            "no concurrent GPU holder")
        scratch = self.scratch/label
        scratch.mkdir()
        env = p.runtime_env(self.env, scratch)
        env["CUDA_CACHE_PATH"] = self.env["CUDA_CACHE_PATH"]
        argv = self.arm_command(arm, [s.E+"/bin/python","-B",s.HERE/"execute.py",
            self.manifest_path,self.manifest_hash,label,self.gpu_selector])
        pid = self.run_command(argv,label,env=env,cwd=scratch,timeout=s.CAPS[row["system"]])
        self.complete_case(row,pid)
        return self.verify_case(row,pid)

    def complete_case(self, row, pid):
        """Parent-only custody after wait(): capture final atexit bytes before science."""
        label = row["label"]
        complete = self.raw/(label+".COMPLETE.json")
        if complete.exists():
            raise FileExistsError(complete)
        child_exit = s.load(self.raw/(label+".exit.json"))
        s.require(child_exit["pid"] == pid and child_exit["exit"] == 0 and
                  child_exit["interrupted"] is False and not self.signalled,
                  "successful actual child exit required")
        endpoint_path = self.raw/(label+".ENDPOINT-PRE-EXIT.json")
        s.require(endpoint_path.is_file() and not endpoint_path.is_symlink() and
                  not endpoint_path.stat().st_mode & 0o222, "readonly endpoint evidence")
        endpoint = s.load(endpoint_path)
        s.require(endpoint["evidence_kind"] ==
                  "pre-exit endpoint evidence; native artifacts not yet verified" and
                  "output_hashes" not in endpoint and
                  endpoint["label"] == label and endpoint["pid"] == pid and
                  endpoint["job"] == self.job and endpoint["uuid"] == self.gpu_selector and
                  endpoint["early"]["pid"] == pid and
                  endpoint["early"]["manifest_sha256"] == self.manifest_hash and
                  endpoint["actual_gpu_execution"] is True and endpoint["energy_calls"] == 1 and
                  endpoint["pre_gpu"] == endpoint["post_gpu"] ==
                      dict(pid=pid,uuid=self.gpu_selector), "pre-exit endpoint identity")
        out = self.raw/"sessions"/label
        s.require(all(v.is_dir() and not v.is_symlink()
                      for v in (self.raw,self.raw/"sessions",out)) and
                  out.resolve().parent == (self.raw/"sessions").resolve(), "output path custody")
        # Validate all entries before sealing any; required files may only appear at exit.
        required = {"result.json","psi4.out","timer.dat","profile.txt"}
        entries = list(out.iterdir())
        s.require(required <= {v.name for v in entries}, "missing completed output")
        for file in entries:
            info = file.lstat()
            s.require(stat.S_ISREG(info.st_mode) and
                      info.st_size <= s.POLICY["output_cap_bytes"], "output cap/type/path")
        s.require((out/"result.json").stat().st_size <= s.POLICY["record_cap_bytes"], "record cap")
        files = {}
        for name in sorted(required):
            file = out/name
            fd = os.open(file, os.O_RDONLY|os.O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                s.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                          info.st_size <= s.POLICY["output_cap_bytes"] and
                          (name != "result.json" or info.st_size <= s.POLICY["record_cap_bytes"]),
                          "ordinary exclusive output file")
                os.fchmod(fd, 0o444)  # Parent seals completed bytes, never before native exit.
                os.fsync(fd)
                with os.fdopen(os.dup(fd), "rb") as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda:stream.read(1024*1024), b""):
                        digest.update(block)
                    files[name] = digest.hexdigest()
                final = os.fstat(fd)
                actual = file.lstat()
                s.require(not final.st_mode & 0o222 and
                          (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns) ==
                          (final.st_dev,final.st_ino,final.st_size,final.st_mtime_ns) and
                          (actual.st_dev,actual.st_ino,actual.st_mode,actual.st_size,actual.st_mtime_ns) ==
                          (final.st_dev,final.st_ino,final.st_mode,final.st_size,final.st_mtime_ns),
                          "stable readonly completed output")
            finally:
                os.close(fd)
        s.n.fsync_dir(out)
        s.durable(complete, dict(endpoint, evidence_kind="parent post-exit completed output custody",
            endpoint_evidence=endpoint, endpoint_sha256=s.sha(endpoint_path),
            child_exit=child_exit, output_hashes=files))

    def verify_case(self, row, pid):
        arm,label = row["arm"],row["label"]
        out = self.raw/"sessions"/label
        receipt = s.load(self.raw/(label+".COMPLETE.json"))
        endpoint_path = self.raw/(label+".ENDPOINT-PRE-EXIT.json")
        s.require(receipt["evidence_kind"] == "parent post-exit completed output custody" and
                  receipt["endpoint_sha256"] == s.sha(endpoint_path) and
                  receipt["endpoint_evidence"] == s.load(endpoint_path) and
                  all(receipt[key] == value for key,value in receipt["endpoint_evidence"].items()
                      if key != "evidence_kind") and
                  receipt["child_exit"] == s.load(self.raw/(label+".exit.json")) and
                  receipt["child_exit"]["exit"] == 0 and
                  receipt["child_exit"]["pid"] == pid and
                  receipt["child_exit"]["interrupted"] is False, "parent post-exit custody")
        s.require(receipt["label"] == label and receipt["pid"] == pid and
                  receipt["job"] == self.job and receipt["uuid"] == self.gpu_selector and
                  receipt["actual_gpu_execution"] is True and receipt["energy_calls"] == 1 and
                  receipt["pre_gpu"] == receipt["post_gpu"] == dict(pid=pid,uuid=self.gpu_selector),
                  "actual process/UUID/one-call proof")
        for name,digest in receipt["output_hashes"].items():
            file = out/name
            s.require(not file.is_symlink() and not file.stat().st_mode & 0o222 and
                      s.sha(file) == digest, "readonly post-exit output hash")
        s.require(set(receipt["output_hashes"]) == {"result.json","psi4.out","timer.dat","profile.txt"},
                  "complete artifacts")
        for file in out.iterdir():
            s.require(file.is_file() and not file.is_symlink() and
                      file.stat().st_size <= s.POLICY["output_cap_bytes"], "output cap/type")
        s.require((out/"result.json").stat().st_size <= s.POLICY["record_cap_bytes"], "record cap")
        d = s.load(out/"result.json")
        runner = c.literal_assignments(s.HERE/"runner.py",["SCIENCE","WATER_DIMER"])
        geom = c.literal_assignments(Path(self.m["suite"])/"systems.py",["SYSTEMS"])["SYSTEMS"][row["system"]]
        options = {**runner["SCIENCE"],"USE_CUEST":True,"CUEST_XC":True,"CUEST_SAD":True,
            "DFT_RADIAL_POINTS":75,"DFT_SPHERICAL_POINTS":302,"DFT_RADIAL_POINTS_INITIAL":35,
            "DFT_SPHERICAL_POINTS_INITIAL":194,"DFT_GRID_SWITCH_FACTOR":1.0}
        item = self.m["arms"][arm]
        stage = Path(item["build"])/"stage"
        expected = dict(source_commit=item["pin"],runner_sha256=s.sha(s.HERE/"runner.py"),
            suite_systems_sha256=p.SUITE_SHA,geometry_sha256=hashlib.sha256(geom.encode()).hexdigest(),
            slurm_job_id=self.job,hostname=socket.gethostname(),system=row["system"],
            suite_system=row["system"],psi4_module=self.probes[arm]["module"],method="sapt(dft)-d4(i)",
            basis="6-31+g**",trace_print_options={"global":3,"scf":3,"sapt":3},
            science_options_sha256=hashlib.sha256(json.dumps(runner["SCIENCE"],sort_keys=True).encode()).hexdigest(),
            grid_policy=dict(initial="35x194",final="75x302",switch_factor=1.0,minimum_final_iterations=2),
            policy_checks=dict(problems=[],switched_scfs=2),
            iterator=dict(path=str(stage/"lib/psi4/driver/procrouting/scf_proc/scf_iterator.py"),
                          sha256=self.drivers[arm]["procrouting/scf_proc/scf_iterator.py"]))
        sizes = p.SYSTEMS[row["system"]]
        expected.update(zip(["natom","nbf","naux_jkfit","naux_ri","nbf_monomer_a","nbf_monomer_b"],sizes[:6]))
        s.require(expected["geometry_sha256"] == sizes[6], "geometry identity")
        text = (out/"psi4.out").read_text()
        verdict = science.validate_record(d,text,(out/"timer.dat").read_text(),
            (out/"profile.txt").read_text(),expected,options,geom,candidate=arm == "candidate")
        # Do not retain inherited UNAVAILABLE-PRINT note for this new explicit PRINT3 cohort.
        verdict = dict(verdict,plan_counts="partial observed native events; see trace coverage")
        trajectory = r.trajectory(text)  # post-energy numeric SCF convergence, no observer
        s.require([len(v["rows"]) for v in trajectory] ==
                  [v["iterations"]+1 for v in d["proofs"]], "complete finite SCF trajectory")
        verdict["scf_terminal_rows"] = [dict(stage=v["stage"], **v["rows"][-1]) for v in trajectory]
        traced = parse(text,arm)
        s.require(traced["observed_created"] > 0, "missing observable molecular plan creation")
        regions = timer_regions(d)
        s.require(regions["SCF: JK native initialize"]["n_calls"] > 0, "initialize call coverage")
        before = s.load(self.raw/(label+".before.json"))
        early = s.load(self.raw/(label+".effective-env.json"))
        s.require(before["pid"] == pid and before["core_sha256"] == self.probes[arm]["core_sha256"] and
                  receipt["early"] == early and early["pid"] == pid and
                  early["manifest_sha256"] == self.manifest_hash, "import/early ENV custody")
        runtime = s.inherited("planreuse_speed","runtime")
        for mapped in (receipt["maps_after_CDLL"],receipt["final_maps"]):
            p.check_runtime_maps(mapped)
            p.check_required_runtime_maps(mapped)
            runtime.native_maps(mapped,self.m["native_libraries"])
        actual = s.load(self.raw/(label+".maps.json"))
        s.require(actual == receipt["final_maps"] and
                  all(actual.get(path) == digest for path,digest in d["libraries_sha256"].items()) and
                  str(Path(self.probes[arm]["core"]).resolve()) in d["libraries_sha256"],
                  "runner/map agreement")
        other = self.m["arms"]["candidate" if arm == "baseline" else "baseline"]["build"]
        s.require(not any(path.startswith(other) for path in actual), "other arm mapped")
        s.durable(self.raw/(label+".VERIFIED.json"), dict(validation=verdict,trace=traced,regions=regions))
        return dict(row,data=d,validation=verdict,trace=traced,pid=pid)

    def execute(self):
        from custody import preserve
        from statistics_report import summary
        s.allocated()
        s.source_check(self.m)
        preserve(self.m["preservation"])
        s.require(set(v.name for v in self.root.iterdir()) ==
                  {"manifest.json","campaign.sbatch","bundle-receipt.json","logs"} and
                  not any(v.is_symlink() for v in self.root.iterdir()), "pristine start; no resume")
        s.durable(self.root/"STARTED.json",dict(job=self.job,pin=self.m["pin"],manifest_sha256=self.manifest_hash))
        s.admit(self.end-time.time(),0,prolog=True)
        signal.signal(signal.SIGALRM,deadline)
        signal.alarm(s.POLICY["prolog_seconds"])
        self.prolog()
        signal.alarm(0)
        (self.raw/"sessions").mkdir()
        for index,row in enumerate(s.SEQUENCE):
            s.admit(self.end-time.time(),index)
            # Full pins/launch/verification must all fit this record's reserved cap+validation.
            started = time.monotonic()
            signal.alarm(s.CAPS[row["system"]]+s.POLICY["validation_seconds"])
            value = self.case(row)
            signal.alarm(0)
            s.require(time.monotonic()-started <= s.CAPS[row["system"]]+s.POLICY["validation_seconds"],
                      "per-record total budget cap")
            self.records.append(value)
            s.durable(self.raw/(row["label"]+".RETAINED.json"),
                dict(label=row["label"],result_sha256=s.sha(self.raw/"sessions"/row["label"]/"result.json"),
                     individually_valid=True, cross_arm_gates="findings only, no filtering"))
        s.require(len({v["pid"] for v in self.records}) == 24, "24 distinct fresh process PIDs")
        s.admit(self.end-time.time(),24)
        signal.alarm(s.POLICY["analysis_seconds"])
        self.pins()
        for arm,item in self.m["arms"].items():
            r.verify_artifact(self.m["diagnostic_pair"],arm,item["files"])
        s.require(all(s.sha(path) == h for path,h in self.inputs.items()), "toolchain changed")
        preserve(self.m["preservation"])
        report = summary(self.records)
        report.update(actual_gpu_results=True,uuid=self.gpu_selector,pin=self.m["pin"],
                      manifest_sha256=self.manifest_hash)
        s.durable(self.root/"WHOLE-REPORT.json",report)
        self.inventory()
        s.durable(self.root/"WHOLE-COMPLETE.json",dict(processes=24,energy_calls=24,
            individually_valid_records=24,observational_only=True,no_dropped_samples=True))
        signal.alarm(0)


def main():
    s.allocated()
    path,mh = sys.argv[1:]
    s.require(Path(path) == Path(s.RUN)/"manifest.json", "allocation manifest path")
    m = s.manifest(path,mh)
    engine = Engine(m,path)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,engine.stop)
    try:
        engine.execute()
    except BaseException as exc:
        signal.alarm(0)
        s.durable(Path(s.RUN)/"WHOLE-FAILED.json",dict(error_type=type(exc).__name__,
            error=str(exc),signalled=engine.signalled,individually_valid_records=len(engine.records)))
        raise


if __name__ == "__main__":
    main()
