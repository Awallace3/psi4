"""Allocation-only orchestration; inherits complete reviewed pair/probe/UUID gates."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import config as s
import policy as p
import repeatability as r
import repeatability_campaign as rc
from protocol import expected_trace


def records_check(value, row, pid, uuid, job):
    from statistics_report import summarize
    s.require(value["label"] == row["label"] and value["arm"] == row["arm"] and
              value["pid"] == pid and value["job"] == job and value["gpu_uuid"] == uuid and
              value["actual_gpu_execution"] is True and value["backend"] == "cuESTJK" and
              value["calls"] == s.CALLS and value["trace"] == expected_trace(row["arm"], 6),
              "actual process identity/count/backend")
    s.require(value["core_sha256"] == r.ALLOWLIST[row["arm"]]["core_sha256"] and
              value["numerical_overrides"] == {} and
              value["cudart"] == p.RUNTIME_LIBS["libcudart.so"], "exact process core/runtime")
    p.check_runtime_maps(value["maps_after_CDLL"])
    p.check_required_runtime_maps(value["maps_after_CDLL"])
    p.check_runtime_maps(value["final_runtime_maps"])
    p.check_required_runtime_maps(value["final_runtime_maps"])
    expected_schedule = [("warmup", 0)]+[(phase, i) for i in range(1, 6)
                                       for phase in ("initialize-compute", "control")]
    s.require([(v["phase"], v["round"]) for v in value["records"]] == expected_schedule,
              "all eleven samples, no dropped output")
    for i, record in enumerate(value["records"]):
        initializations = 1+(i+1)//2
        expected_calls = dict(initialize=initializations, compute=i+1, finalize=0,
                              C_left_add=1, C_right_add=0)
        s.require(record["calls"] == expected_calls and
                  record["trace"] == expected_trace(row["arm"], initializations) and
                  record["gpu"] == dict(pid=pid, uuid=uuid) and
                  record["C"] == value["factor_C"]["array"] and record["matched_D"],
                  "per-call input/count/GPU identity")


def verify_process(raw, value, row, pid, uuid, m):
    import fixed_jk as f
    import observe as o
    import numpy as np
    from runtime import native_maps
    records_check(value, row, pid, uuid, os.environ["SLURM_JOB_ID"])
    native_maps(value["maps_after_CDLL"], m["native_libraries"])
    native_maps(value["final_runtime_maps"], m["native_libraries"])
    s.require(value["settings"] == m["settings"], "actual settings")
    root = raw/row["label"]
    early = s.load(raw/(row["label"]+".effective-env.json"))
    s.require(value["early_env"] == early and early["pid"] == pid and
              early["job"] == value["job"] and early["label"] == row["label"] and
              early["gpu_uuid"] == uuid and early["numerical_overrides"] == {} and
              early["manifest_sha256"] == s.sha(Path(s.RUN)/"manifest.json"), "early ENV proof")
    s.require(s.sha(root/"plan.out") == value["trace_sha256"] and
              f.traces((root/"plan.out").read_text(), row["arm"]) == expected_trace(row["arm"], 6),
              "actual full plan trace")
    factor = s.load(Path(f.FROZEN_FACTOR["root"])/"FACTOR.json")
    s.require(value["factor_C"] == factor["C"], "frozen factor for all calls")
    stored = []
    for i, rec in enumerate(value["records"]):
        label = "warmup" if i == 0 else f"round{rec['round']}-{rec['phase']}"
        s.require(s.load(root/(label+".json")) == rec, "every raw snapshot metadata persisted")
        if i not in (0, 10):
            s.require(not rec["saved"], "no intervening arrays retained")
        else:
            s.require(set(rec["saved"]) == {"J", "K"}, "saved warmup/final array set")
        for key, meta in rec["saved"].items():
            file = root/meta["file"]
            s.require(file.name in (label+"-"+key+".npy",) and
                      not file.stat().st_mode & 0o222 and s.sha(file) == meta["file_sha256"],
                      "readonly stored matrix")
            a = np.load(file, allow_pickle=False)
            f.finite_array(a, (1786, 1786))
            s.require(o.array_info(a) == meta["array"] == rec["arrays"][key], "snapshot readback")
            stored.append(file)
    s.require(len(stored) == 4 and set(stored) == set(root.glob("*.npy")) and
              sum(x.stat().st_size for x in stored) == value["stored_bytes"] <= s.PROCESS_CAP,
              "four-array process cap")
    return value


def accept_inputs(values):
    ds = [(v["label"], r["phase"], r["round"], r["actual_D"])
          for v in values for r in v["records"]]
    s.require(len(ds) == 66 and len({p.digest(row[3]) for row in ds}) == 1,
              "actual reconstructed density differs across processes/calls; NO speedup acceptance")
    s.require(len({p.digest(v["actual_geometry"]) for v in values}) == 1, "live geometry difference")
    s.require(len({p.digest(v["factor_C"]) for v in values}) == 1, "coefficient byte difference")


def accept_candidate_stability(values):
    s.require(all(rec["stable_within_original_roundoff"] for v in values
                  if v["arm"] == "candidate" for rec in v["records"]),
              "candidate unchanged-initialize variation beyond prior roundoff; findings retained")


class Engine(rc.Engine):
    def pins(self):
        s.n.check_env(os.environ)
        super().pins()
        s.source_check(self.m)
        compiler = self.m["compiler"]
        s.require(str(Path(compiler["actual"]).resolve(strict=True)) == compiler["actual_resolved"],
                  "compiler target alias changed")
        extra = dict(self.m["sdk_headers"], **self.m["native_libraries"])
        extra.update({compiler["actual"]:compiler["actual_sha256"],
                      compiler["frontend"]:compiler["frontend_sha256"],
                      compiler["wrapper"]:compiler["wrapper_sha256"]})
        for path, digest in extra.items():
            s.require(s.sha(path) == digest, "full compiler/native runtime identity: "+path)

    def run_command(self, argv, name, env=None, cwd=None, timeout=600, **kwargs):
        """Reviewed process-group/TERM pattern; exact PID and startup elapsed retained."""
        s.require(not self.signalled and self.end-time.time() >= timeout+300, "admission budget")
        s.durable(self.raw/(name+".argv.json"), dict(argv=list(map(str, argv)),
                                                   cwd=str(cwd or self.scratch)))
        started = time.perf_counter_ns()
        with (self.raw/(name+".log")).open("xb") as stream:
            proc = subprocess.Popen(list(map(str, argv)), cwd=cwd or self.scratch,
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
        s.require((self.raw/(name+".log")).stat().st_size < 8*1024**2, "bounded command log")
        return proc.pid

    def execute_speed(self):
        from custody import preserve
        from numerics import total_storage
        from statistics_report import summarize
        s.allocated()
        s.source_check(self.m)
        preserve(self.m["preservation"])
        s.durable(self.root/"STARTED.json", dict(job=self.job, pin=self.m["pin"]))
        self.prolog()  # full source/stage/config/ABI/SASS/143-driver/import checks, ZERO builds
        p.admission(self.end-time.time(), 6*1800+1200+300)
        values = []
        for index, row in enumerate(s.SEQUENCE):
            p.admission(self.end-time.time(), (6-index)*1800+1200+300)
            self.pins()
            s.require(not self.query(["nvidia-smi", "--id="+self.gpu_selector,
                "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]).strip(),
                "no concurrent GPU holder before fresh process")
            argv = self.arm_command(row["arm"], [s.E+"/bin/python", "-B", s.HERE/"runner.py",
                self.manifest_path, self.manifest_hash, row["label"], self.gpu_selector])
            pid = self.run_command(argv, row["label"], timeout=1800)
            value = s.load(self.raw/row["label"]/"COMPLETE.json")
            verify_process(self.raw, value, row, pid, self.gpu_selector, self.m)
            values.append(value)
            total_storage(self.raw)
        s.require(len({v["pid"] for v in values}) == 6, "six distinct processes")
        # Always retain all numeric findings/input identities before performance gate.
        s.durable(self.root/"NUMERICAL-FINDINGS.json", dict(
            bounds=s.POLICY["findings_bounds"],
            records=[dict(label=v["label"], records=v["records"]) for v in values],
            original_full_SAPT_gate="unchanged/unresolved"))
        try:
            accept_inputs(values)
        except BaseException:
            s.durable(self.root/"INPUT-DIFFERENCES.json", dict(actual_D=[
                dict(label=v["label"], phase=r["phase"], round=r["round"],
                     actual_D=r["actual_D"], difference=r["actual_D_difference"])
                for v in values for r in v["records"]]))
            raise
        accept_candidate_stability(values)
        p.admission(self.end-time.time(), 1200+300)
        def analysis_deadline(*_):
            raise RuntimeError("final verification/analysis budget exhausted")
        signal.signal(signal.SIGALRM, analysis_deadline)
        signal.alarm(1200)
        self.pins()
        for arm, item in self.m["arms"].items():
            r.verify_artifact(self.m["diagnostic_pair"], arm, item["files"])
        s.require(all(p.sha(path) == h for path, h in self.inputs.items()), "toolchain changed")
        import fixed_jk as f
        f.frozen_factor_files(f.FROZEN_FACTOR["root"])
        preserve(self.m["preservation"])
        report = summarize(values)
        report.update(actual_gpu_execution=True, scope=s.POLICY["scope"], computes=66,
            creation_reuse={v["label"]:v["trace"] for v in values},
            actual_D=values[0]["records"][0]["actual_D"],
            total_stored_bytes=total_storage(self.raw), uuid=self.gpu_selector,
            pin=self.m["pin"], manifest_sha256=self.manifest_hash,
            process_startup=[dict(label=v["label"], setup_ns=v["setup_ns"],
                full_child_execution=s.load(self.raw/(v["label"]+".exit.json"))) for v in values])
        s.durable(self.root/"SPEED-REPORT.json", report)
        s.durable(self.root/"SPEED-COMPLETE.json", dict(processes=6, computes=66,
            stored_arrays=24, matched_actual_D=True, plan_reuse_benchmark_only=True))
        signal.alarm(0)


def main():
    s.allocated()
    path, mh = sys.argv[1:]
    m = s.manifest(path, mh)
    s.require(Path(path) == Path(s.RUN)/"manifest.json", "exact allocation manifest path")
    engine = Engine(m, path)
    engine.manifest_hash = mh
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, engine.stop)
    try:
        engine.execute_speed()
    except BaseException as exc:
        s.durable(Path(s.RUN)/"SPEED-FAILED.json", dict(error_type=type(exc).__name__,
            error=str(exc), signalled=engine.signalled, accepted_speedup_data=False))
        raise


if __name__ == "__main__":
    main()
