"""No-build allocated orchestration, isolated from molecular/performance comparison."""
import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

import campaign as c
import fixed_jk as f
import handoff as h
import observe as o
import policy as p
import repeatability as r
import repeatability_campaign as rc
import workspace_control as w

HERE = Path(__file__).resolve().parent


def batch_text(mh):
    return rc.batch_text(mh).replace(r.RUN, f.RUN).replace(r.HARNESS, f.HARNESS).replace(
        "repeatability_campaign.py run", "fixed_jk_campaign.py run")


def source_review(roots):
    """Fail if reviewed SAD, symmetric-factor or RHF contraction semantics change."""
    checks = {
        "psi4/src/psi4/libscf_solver/sad.cc": [
            'Ca_ = Da_->partial_cholesky_factorize(options_.get_double("SAD_CHOL_TOLERANCE"))',
            "SharedMatrix Ca_sad = guess->Ca()", "nalpha_ = sad_dim.sum()"],
        "psi4/src/read_options.cc": ['options.add_double("SAD_CHOL_TOLERANCE", 1E-7)',
            'options.add_int("CUEST_DFK_SLICES", 5)', 'options.add_int("CUEST_DFK_MODULI", 8)',
            'options.add_double("DF_FITTING_CONDITION", 1.0E-10)'],
        "psi4/src/psi4/libscf_solver/rhf.cc": ['C.push_back(Ca_subset("SO", "OCC"))',
            "G_->axpy(2.0, J_)", "G_->axpy(-alpha, K_)",
            "double coulomb_E = 2.0 * Da_->vector_dot(J_)",
            "exchange_E -= alpha * Da_->vector_dot(K_)",
            'energies_["Two-Electron"] = coulomb_E + exchange_E'],
        "psi4/src/psi4/libfock/jk.cc": ["void JK::initialize() { preiterations(); }",
            "void JK::finalize() { postiterations(); }", "C_right_ = C_left_", "C_right_.clear()",
            *[line+"\\n" for line in f.SIMPLE_DF_WARNING.splitlines()],
            'options.get_bool("USE_CUEST")', "std::make_shared<cuESTJK>"],
        "psi4/src/export_mints.cc": ['"partial_cholesky_factorize", &Matrix::partial_cholesky_factorize'],
        "psi4/src/export_fock.cc": ['.def("C_left_add"', '.def("set_do_wK"', '.def("set_omega"'],
    }
    result = {}
    for arm, root in roots.items():
        files = {}
        for rel, tokens in checks.items():
            file = Path(root)/rel
            text = file.read_text()
            p.require(all(token in text for token in tokens), "source semantic review: "+rel)
            files[rel] = p.sha(file)
        result[arm] = files
    return result


def freeze(args):
    payload = p.read_json(args.collection)
    old = p.read_json(args.old_manifest)
    r.validate(old)
    r.validate_pair(payload["pair"])
    local = dict(baseline=Path(args.baseline_local), candidate=Path(args.candidate_local))
    arms = {}
    for arm, root in local.items():
        origin = r.ALLOWLIST[arm]
        p.check_source(root, origin["pin"])
        files = p.file_manifest(root)
        p.require(p.digest(files) == payload["pair"]["arms"][arm]["source_files_digest"], "whole scientific source")
        arms[arm] = dict(source=origin["source"], build=origin["build"], pin=origin["pin"], files=files)
    harness = HERE.parents[1]
    pin = p.git(harness, "rev-parse", "HEAD")
    p.check_source(harness, pin)
    additions = p.git(harness, "diff", "--name-status", w.PARENT, pin).splitlines()
    owned = {"devtools/major2/"+name for name in (
        "fixed_jk.py", "fixed_jk_runner.py", "fixed_jk_campaign.py", "fixed_jk_handoff.py", "FIXED_JK.md")}
    p.require(additions and all(
        line.startswith("A\tdevtools/major2/") or
        (line.startswith("M\t") and line.split("\t")[1] in owned)
        for line in additions), "narrow owned warning/factor repair only")
    files = p.file_manifest(harness)
    m = dict(schema=1, mode=f.MODE, approval=f.REVIEW_ID, diagnostic_pair=payload["pair"],
        run_root=f.RUN, route=p.ROUTE, gates=p.GATES, suite=p.H0+"/geometries", suite_sha256=p.SUITE_SHA,
        scratch_parent="/storage/scratch1/3/awallace43/cpks-major2-workspace", arms=arms,
        harness=dict(path=f.HARNESS, pin=pin, files=files, files_digest=p.digest(files)),
        sequence=f.sequence(), matrix_cap_bytes=f.CAP, settings=f.SETTINGS, findings_bounds=f.BOUNDS,
        input_path=f.INPUT, input_sha256=f.INPUT_SHA, input_Da=f.DA, factor_tolerance=f.TOL,
        frozen_factor=f.FROZEN_FACTOR,
        build_budget_seconds=[0,0], per_process_timeout_seconds=1800, factor_budget_seconds=600,
        analysis_budget_seconds=1200, teardown_reserve_seconds=300, accepted_speedup_data=False,
        identity_helper=f.IDENTITY, numerical_control=w.POLICY, completed_control=payload["completed_control"],
        preservation=payload["preservation"], prior_staging=payload["prior_staging"],
        input_complete_sha256=payload["input_complete_sha256"], source_review=source_review(local))
    f.validate(m)
    out = Path(args.output_dir)
    out.mkdir(exist_ok=False)
    o.atomic(out/"manifest.json", m)
    mh = p.sha(out/"manifest.json")
    h.durable(out/"campaign.sbatch", batch_text(mh))
    o.atomic(out/"bundle-receipt.json", dict(manifest_sha256=mh, batch_sha256=p.sha(out/"campaign.sbatch"),
        harness_pin=pin, generated_only=True, no_submission=True))
    print("FROZEN fixed-factor endpoint, NOT submitted")


def storage(raw):
    files = list(raw.glob("fixed-*/*.npz")) + list((raw/"factor").glob("*.npy"))
    p.require(len(files) <= 26, "bounded output file count")
    used = sum(file.stat().st_size for file in files)
    p.require(used <= f.CAP, "actual total array storage cap")
    return used


def verify_process(raw, row, factor_sha, uuid):
    root = raw/row["label"]
    value = p.read_json(root/"COMPLETE.json")
    env_file = raw/(row["label"]+".effective-env.json")
    env_receipt = p.read_json(env_file)
    w.verify_receipt(env_receipt, row, value["pid"], os.environ["SLURM_JOB_ID"], uuid,
                     p.sha(Path(f.RUN)/"manifest.json"))
    p.require(not env_file.stat().st_mode & 0o222 and
              value["numerical_environment"] == w.CONTROL and
              value["effective_env_receipt"] == env_receipt and
              value["effective_env_sha256"] == p.sha(env_file), "raw effective ENV receipt")
    p.require(value["label"] == row["label"] and value["arm"] == row["arm"] and
              value["settings"] == f.SETTINGS and value["factor_receipt_sha256"] == factor_sha,
              "process exact fixed input/mode")
    p.require(value["calls"] == dict(initialize=3, compute=4, finalize=2, C_left_add=1, C_right_add=0),
              "exact four-compute dispatch")
    f.verify_reused_factor(raw/"factor")
    p.require(value["actual_backend"] == "cuESTJK" and
              value["reused_factor_origin_sha256"] == p.sha(raw/"factor/REUSED_FACTOR_ORIGIN.json"),
              "live backend and frozen origin")
    p.require(value["gpu_uuid"] == uuid and value["core_sha256"] == r.ALLOWLIST[row["arm"]]["core_sha256"],
              "process GPU/core identity")
    p.check_runtime_maps(value["runtime_maps"])
    p.check_required_runtime_maps(value["runtime_maps"])
    p.require(p.sha(root/"plan.out") == value["trace_sha256"] and
              f.traces((root/"plan.out").read_bytes().decode("utf-8"), row["arm"]) == f.expected_trace(row["arm"], "D"),
              "complete backend trace")
    p.require([v["state"] for v in value["records"]] == list(f.STATES), "four output states")
    factor = p.read_json(raw/"factor/FACTOR.json")
    p.require(value["factor_C"] == factor["C"], "same factor bytes all processes")
    for v in value["records"]:
        p.require(v["numerical_environment"] == w.CONTROL, "per-output ENV proof")
        p.require(v["actual_backend"] == "cuESTJK" and
                  v["C_left_add"] == factor["C"]["array"] and v["rank"] == f.RANK and
                  v["after"] == f.expected_trace(row["arm"], v["state"]) and
                  v["gpu"] == dict(pid=value["pid"], gpu_uuid=uuid), "actual input/count/PID")
        file = root/(v["state"]+".npz")
        p.require(v["file"] == file.name and p.sha(file) == v["file_sha256"] and
                  file.stat().st_size == v["file_bytes"], "exact output bytes")
        with np.load(file, allow_pickle=False) as arrays:
            p.require(set(arrays.files) == {"J", "K"}, "exact J/K only")
            for key in ("J", "K"):
                f.finite_array(arrays[key], (1786,1786))
                p.require(o.array_info(arrays[key]) == v["arrays"][key], "output array integrity")
    p.require(value["stored_bytes"] == sum(v["file_bytes"] for v in value["records"]) <= f.PROCESS_CAP,
              "actual process storage")
    storage(raw)
    return value


def analyze(raw, values):
    f.allocated()
    p.require(len(values) == 6 and len({v["pid"] for v in values}) == 6, "six distinct fresh PIDs")
    p.require([v["label"] for v in values] == [v["label"] for v in f.sequence()], "balanced six order")
    p.require(len({p.digest(v["molecule_geometry"]) for v in values}) == 1, "identical live geometry all six")
    factor = p.read_json(raw/"factor/FACTOR.json")
    _, density = f.load_factor(raw/"factor", factor)
    outputs = []
    for process in values:
        for row in process["records"]:
            file = raw/process["label"]/row["file"]
            with np.load(file, allow_pickle=False) as arrays:
                p.require(f.scalar(density, arrays["J"], arrays["K"]) == row["scalar_eh"], "scalar readback")
            outputs.append(dict(label=process["label"], arm=process["arm"], pid=process["pid"],
                                state=row["state"], **{k:row[k] for k in (
                                    "scalar_eh", "symmetry_maxabs", "symmetry_findings", "file_sha256", "arrays")}))
    p.require(len(outputs) == 24, "all24 outputs")
    differences = []
    for left, right in itertools.combinations(outputs, 2):
        lfile = raw/left["label"]/(left["state"]+".npz")
        rfile = raw/right["label"]/(right["state"]+".npz")
        same = left["label"] == right["label"]
        category = ("same-plan" if {left["state"], right["state"]} == {"A","B"} else
                    "unchanged-reinitialize" if right["state"] == "C" else
                    "finalize-new-plan") if same else ("fresh-process-within-arm" if left["arm"] == right["arm"]
                                                     else "fresh-process-across-arm")
        with np.load(lfile, allow_pickle=False) as aa, np.load(rfile, allow_pickle=False) as bb:
            delta = {k:f.difference(aa[k], bb[k]) for k in ("J","K")}
        scalar_delta = right["scalar_eh"]-left["scalar_eh"]
        p.require(np.isfinite(scalar_delta), "nonfinite scalar difference")
        differences.append(dict(a=left["label"]+"/"+left["state"], b=right["label"]+"/"+right["state"],
            category=category, matrices=delta, scalar_delta_eh=scalar_delta,
            scalar_exceeds_bound=abs(scalar_delta)>f.BOUNDS["scalar_abs_eh"]))
    result = dict(mode=f.MODE, exploratory_endpoint_only=True, accepted_speedup_data=False,
        original_primary_gate_unchanged_and_failed=True, findings_bounds=f.BOUNDS,
        factor_receipt_sha256=p.sha(raw/"factor/FACTOR.json"), C=factor["C"], D=factor["D"],
        reconstruction=factor["reconstruction"], outputs=outputs, differences=differences,
        total_stored_bytes=storage(raw))
    f.no_timing(result)
    p.require(len(differences) == 276, "all output pairs")
    return result


class Engine(rc.Engine):
    def run_command(self, argv, name, env=None, cwd=None, timeout=600, accept_failure=False):
        """Reviewed TERM/process-group cleanup with NO measured durations."""
        p.require(not self.signalled, "signal already received")
        o.atomic(self.raw/(name+".argv.json"), dict(argv=list(map(str,argv)), cwd=str(cwd or self.scratch)))
        with (self.raw/(name+".log")).open("xb") as stream:
            proc = subprocess.Popen(list(map(str,argv)), cwd=cwd or self.scratch, env=env or self.env,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.active = proc
            try:
                rc = proc.wait(timeout=min(timeout,max(1,self.end-time.time()-300)))
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
                o.atomic(self.raw/(name+".exit.json"), dict(exit=proc.returncode, interrupted=True))
                raise
            finally:
                self.active = None
        o.atomic(self.raw/(name+".exit.json"), dict(exit=rc, interrupted=False))
        p.require(not self.signalled and (rc == 0 or accept_failure), "command failed: "+name)
        return rc

    def fixed_child(self, arm, mode, name, extra=(), budget=1800):
        w.check_env(self.env)  # Never mutate controller or remove inherited overrides.
        argv = self.arm_command(arm, [c.PY, "-B", HERE/"fixed_jk_runner.py", mode,
            self.manifest_path, self.manifest_hash, *extra])
        env = None
        if mode == "jk":
            p.require(name in {row["label"] for row in f.sequence()}, "only actual JK children")
            p.require(argv[0] == "bash" and Path(argv[1]).name == "arm.sh", "reviewed child arm")
            argv[1] = str(HERE/"child-arm.sh")
            env = w.child_env(self.env)
        else:
            p.require(mode == "reuse-factor", "no factor generation")
        self.run_command(argv, name, timeout=budget, env=env)
        w.check_env(self.env)

    def execute_fixed(self):
        f.allocated()
        w.check_env(os.environ)
        self.prolog()  # exact reviewed allocation, pair/ABI/config/SASS/import/143-driver checks; no builds
        w.check_env(self.env)
        import fixed_jk_handoff
        fixed_jk_handoff.preserve(self.m)
        p.require(source_review({arm:item["source"] for arm,item in self.m["arms"].items()}) ==
                  self.m["source_review"], "same-allocation source semantics")
        p.admission(self.end-time.time(), 6*1800+600+1200+300)
        self.fixed_child("baseline", "reuse-factor", "factor-reuse", budget=600)
        factor_sha = p.sha(self.raw/"factor/FACTOR.json")
        factor = p.read_json(self.raw/"factor/FACTOR.json")
        f.verify_reused_factor(self.raw/"factor")
        f.load_frozen_factor(self.raw/"factor")
        values = []
        for index, row in enumerate(f.sequence()):
            p.admission(self.end-time.time(), (6-index)*1800+1200+300)
            self.pins()
            f.verify_reused_factor(self.raw/"factor")
            self.fixed_child(row["arm"], "jk", row["label"], (row["label"],self.gpu_selector))
            values.append(verify_process(self.raw,row,factor_sha,self.gpu_selector))
            o.atomic(self.raw/(row["label"]+".VERIFIED.json"), dict(label=row["label"], pid=values[-1]["pid"],
                     factor_receipt_sha256=factor_sha, outputs=4, storage_bytes=storage(self.raw)))
        self.pins()
        p.admission(self.end-time.time(), 1200+300)
        # Analysis is finite and bounded by the allocation/supervisor; no chemistry.
        def analysis_deadline(*_):
            raise RuntimeError("analysis budget exhausted")
        prior = signal.signal(signal.SIGALRM, analysis_deadline)
        signal.alarm(1200)
        try:
            result = analyze(self.raw, values)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prior)
        p.require(p.sha(self.raw/"factor/FACTOR.json") == factor_sha and p.sha(f.INPUT) == f.INPUT_SHA,
                  "immutable inputs after all six processes")
        f.verify_reused_factor(self.raw/"factor")
        result["reused_factor_origin"] = p.read_json(self.raw/"factor/REUSED_FACTOR_ORIGIN.json")
        result["numerical_control"] = w.POLICY
        for output in result["outputs"]:
            output["numerical_environment"] = w.CONTROL
        result["effective_environments"] = [
            dict(label=v["label"], arm=v["arm"], pid=v["pid"],
                 receipt=v["effective_env_receipt"], sha256=v["effective_env_sha256"]) for v in values]
        result["within_cohort_summary"] = w.summary(result)
        old = self.m["completed_control"]
        p.require(p.sha(Path(old["root"])/"ENDPOINT-DIAGNOSTIC.json") ==
                  old["hashes"]["ENDPOINT-DIAGNOSTIC.json"], "old numerical report unchanged")
        result["old_unset_comparison"] = dict(job=old["job"],
            report_sha256=old["hashes"]["ENDPOINT-DIAGNOSTIC.json"],
            summary=w.summary(old["report"]),
            interpretation="Compare bounds and variation only; one control, no unique-cause or SAPT claim")
        f.no_timing(result)
        o.atomic(self.root/"ENDPOINT-DIAGNOSTIC.json", result)


def run(args):
    f.allocated()
    p.require(p.sha(args.manifest) == args.manifest_sha256, "manifest exact bytes")
    m = p.read_json(args.manifest)
    f.validate(m)
    engine = Engine(m,args.manifest)
    engine.manifest_hash = args.manifest_sha256
    o.atomic(Path(f.RUN)/"STARTED.json", dict(job=engine.job,manifest_sha256=args.manifest_sha256,mode=f.MODE))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,engine.stop)
    try:
        engine.execute_fixed()
        o.atomic(Path(f.RUN)/"ENDPOINT-COMPLETE.json", dict(outputs=24, computes=24, accepted_speedup_data=False))
    except BaseException as exc:
        o.atomic(Path(f.RUN)/"ENDPOINT-FAILED.json", dict(error_type=type(exc).__name__, error=str(exc),
            signalled=engine.signalled, accepted_speedup_data=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("freeze")
    for arg in ("collection","old-manifest","baseline-local","candidate-local","output-dir"):
        q.add_argument("--"+arg, required=True)
    q = sub.add_parser("run")
    q.add_argument("--manifest", required=True)
    q.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)(args)


if __name__ == "__main__":
    main()
