# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Synthetic-only metadata preparation / guarded payload. NEVER invokes scheduler."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import stat
import sys
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_pfit_smoke_bootstrap", HERE/"integrated_runtime.py")
bootstrap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bootstrap
spec.loader.exec_module(bootstrap)
R = bootstrap.for_plan(HERE/"pfit_smoke_plan.json")
BASE = "d1b23d019d59d30b6ca9cff9018177cac285d341"


def argv(commit, tree, runtime_sha, procedure_sha):
    for value, length in ((commit, 40), (tree, 40), (runtime_sha, 64), (procedure_sha, 64)):
        if not isinstance(value, str) or re.fullmatch("[0-9a-f]{"+str(length)+"}", value) is None:
            raise ValueError("concrete reviewed source/runtime/procedure pins required")
    return ["/opt/slurm/current/bin/sbatch", "--test-only", "--export=NONE",
        "--chdir="+R.PLAN["run"],
        str(Path(R.PLAN["source"])/"devtools/ccasp_phoenix_benzene/pfit_smoke.sbatch"),
        commit, tree, runtime_sha, procedure_sha]


def source_binding(commit, tree):
    binding = R.source_binding(commit, tree)
    # Stronger than the accepted native/routing allowlist: this procedure may
    # add tools only, and must retain the ENTIRE current production subtree.
    if R.OLD.git(R.ROOT, "rev-parse", commit+":psi4") != R.OLD.git(R.ROOT, "rev-parse", BASE+":psi4"):
        raise ValueError("complete production subtree differs from smoke base")
    changes = R.OLD.git(R.ROOT, "diff", "--name-status", "--no-renames", BASE, commit).splitlines()
    if any(not line.startswith("A\tdevtools/ccasp_phoenix_benzene/") for line in changes):
        raise ValueError("only new devtools additions allowed relative to base")
    return binding


def procedure_document(binding, runtime_sha):
    return dict(schema="ccasp-synthetic-pfit-procedure-admission/1",
        binding=binding, runtime_manifest_sha256=runtime_sha,
        original_declaration=binding["effective_profile"]["declaration_provenance"],
        basis_admission="UNTESTED; no molecular basis/fixture read or rebuilt",
        authorization="one synthetic smoke only; parent review/publication/staging/submission separate",
        no_full_driver=True)


def small_write(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > R.PLAN["procedure"]["max_snapshot_bytes"]:
        raise ValueError("smoke snapshot byte bound")
    return R.write(path, value)


def output_inventory(root, *, reserve=0):
    p = R.PLAN["procedure"]
    count = total = 0
    for parent, dirs, names in os.walk(root, followlinks=False):
        if any((Path(parent)/name).is_symlink() for name in dirs):
            raise ValueError("output directory symlink")
        for name in names:
            file = Path(parent)/name
            info = file.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("nonregular output")
            # SLURM stdout/stderr have a separate 1MiB per-file hard cap.
            cap = 1024**2 if file.parent == Path(root)/"logs" else p["max_snapshot_bytes"]
            if info.st_size > cap:
                raise ValueError("individual output byte bound")
            count += 1
            total += info.st_size
            if count+reserve > p["max_output_files"] or total+reserve*p["max_snapshot_bytes"] > p["max_output_bytes"]:
                raise ValueError("aggregate output file/byte bound")
    return dict(files=count, bytes=total, remaining_reserved_files=reserve)


def prepare(commit, tree):
    binding = source_binding(commit, tree)
    root = Path(R.PLAN["run"]).parent
    runtime, run, scratch = (Path(R.PLAN[k]) for k in ("runtime", "run", "scratch"))
    if any(path.exists() or path.is_symlink() for path in (root, runtime, run, scratch)):
        raise ValueError("all smoke destinations/root must be fresh; retain partial failures")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    runtime.mkdir(mode=0o700, exist_ok=False)
    run.mkdir(mode=0o700, exist_ok=False)
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ("metadata", "logs"):
        (run/name).mkdir(mode=0o700)
    manifest = R.compose(R.ROOT, R.OLD.STAGE, runtime/"stage", binding)
    runtime_sha = R.write(runtime/"runtime-manifest.json", manifest)
    R.verify_runtime(runtime/"runtime-manifest.json", runtime_sha, binding)
    procedure_sha = small_write(runtime/"procedure-admission.json", procedure_document(binding, runtime_sha))
    receipt = dict(schema="ccasp-synthetic-pfit-preparation/1", binding=binding,
        runtime_manifest_sha256=runtime_sha, procedure_sha256=procedure_sha,
        script_sha256=R.sha(HERE/"pfit_smoke.sbatch"),
        test_only_argv=argv(commit, tree, runtime_sha, procedure_sha),
        no_native_import_build_calculation_or_scheduler_invocation=True,
        basis_or_private_geometry_read=False)
    small_write(run/"metadata/preparation.json", receipt)
    output_inventory(run)
    print(json.dumps(dict(status="PREPARED_METADATA_ONLY", preparation=str(run/"metadata/preparation.json"),
        runtime_sha256=runtime_sha, procedure_sha256=procedure_sha)))


def compute_guard():
    R.compute_guard()
    if (os.environ.get("SLURM_CPUS_PER_TASK") != "2" or
            any(os.environ.get(name) != "1" for name in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))):
        raise ValueError("two allocated CPUs and numerical threads one required")
    R.effective_profile()  # exact closed resource/procedure identity, stdlib only


def run(commit, tree, runtime_sha, procedure_sha):
    compute_guard()  # strict before results writes, scientific imports or CDLL
    p = R.PLAN
    root = Path(p["run"])
    out = root/"results"
    out.mkdir(mode=0o700, exist_ok=False)
    # Hard per-file guard also bounds inherited map snapshots and trace output.
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    record = dict(schema="ccasp-synthetic-pfit-result/1", status="FAIL", smoke_pass=False,
        scientific_acceptance=False, commit=commit, tree=tree,
        native_commit=R.OLD.BUILD_COMMIT, native_core_sha256=R.OLD.CORE_SHA,
        runtime_manifest_sha256=runtime_sha, procedure_sha256=procedure_sha,
        job=os.environ["SLURM_JOB_ID"])
    handles = []
    binding = None
    snapshots = {}

    def emit(name, value):
        if not re.fullmatch(r"(design|negative-cases|model-\d{2}|solution-\d{2})\.json", name):
            raise ValueError("unexpected snapshot path")
        output_inventory(root, reserve=5)
        snapshots[name] = small_write(out/name, value)

    try:
        binding = source_binding(commit, tree)
        record["profile"] = binding["effective_profile"]
        preparation = R.pinned_json(root/"metadata/preparation.json", R.sha(root/"metadata/preparation.json"))
        if (preparation["binding"] != binding or preparation["runtime_manifest_sha256"] != runtime_sha
                or preparation["procedure_sha256"] != procedure_sha
                or preparation["script_sha256"] != R.sha(HERE/"pfit_smoke.sbatch")
                or preparation["test_only_argv"] != argv(commit, tree, runtime_sha, procedure_sha)):
            raise ValueError("reviewed preparation differs")
        if R.pinned_json(Path(p["runtime"])/"procedure-admission.json", procedure_sha) != procedure_document(binding, runtime_sha):
            raise ValueError("procedure identity differs before native imports")
        psi4, _ = R.initialize(binding, Path(p["runtime"])/"runtime-manifest.json", runtime_sha, root, handles)
        psi4.set_memory(p["procedure"]["max_bytes"])
        driver = R.module("_pfit_smoke_native_driver", "pfit_smoke_driver.py")
        record["checks"] = driver.run_native(R, emit)
        if (record["checks"].get("status") != "SYNTHETIC_CHECKS_PASSED_PENDING_FINAL_CLOSURE"
                or record["checks"].get("solved") != 11):
            raise ValueError("synthetic checks incomplete")
    except BaseException as error:
        record.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-8192:])
    finally:
        failures = []
        # Independent closure attempts: a map failure cannot skip source/runtime
        # checks, nor overwrite the primary procedure failure.
        if handles:
            for name, close in (
                ("loader", lambda: R.OLD.snapshot("after-smoke", receipt_run=root)),
                ("runtime", lambda: R.verify_runtime(Path(p["runtime"])/"runtime-manifest.json",
                    runtime_sha, binding, hash_core=True)),
                ("source", lambda: _same_binding(commit, tree, binding)),
                ("procedure", lambda: _same_procedure(binding, runtime_sha, procedure_sha))):
                try:
                    close()
                except BaseException as error:
                    failures.append(dict(stage=name, error=f"{type(error).__name__}: {error}"))
            record["closure"] = "PASS" if not failures else "FAIL"
        else:
            record["closure"] = "NOT_INITIALIZED"
        if failures:
            record["closure_errors"] = failures
    try:
        record["output_inventory_before_terminal"] = output_inventory(root, reserve=3)
    except BaseException as error:
        record["output_bound_error"] = str(error)
    record["snapshots_sha256"] = snapshots
    passed = ("checks" in record and "error" not in record and record["closure"] == "PASS"
              and "output_bound_error" not in record)
    record.update(status="PASS" if passed else "FAIL", smoke_pass=passed,
        native_smoke_only=True, basis_LW_molecular_admission_untested=True)
    small_write(out/"result.json", record)
    small_write(root/"metadata/payload-terminal.json", dict(exit_code=0 if passed else 1,
        complete=passed, smoke_pass=passed, final_closure=record["closure"],
        scientific_acceptance=False, result_sha256=R.sha(out/"result.json")))
    print(json.dumps(dict(status=record["status"], smoke_pass=passed, scientific_acceptance=False)))
    return 0 if passed else 1  # retained handles outlive EVERY closure and receipt


def _same_binding(commit, tree, binding):
    if source_binding(commit, tree) != binding:
        raise ValueError("final source/profile differs")


def _same_procedure(binding, runtime_sha, procedure_sha):
    if R.pinned_json(Path(R.PLAN["runtime"])/"procedure-admission.json", procedure_sha) != procedure_document(binding, runtime_sha):
        raise ValueError("final procedure identity differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--runtime-sha256")
    parser.add_argument("--procedure-sha256")
    a = parser.parse_args()
    if a.action == "prepare":
        # Validate source pins without inventing runtime pins.
        for value in (a.commit, a.tree):
            if re.fullmatch("[0-9a-f]{40}", value) is None:
                parser.error("concrete source pins required")
        prepare(a.commit, a.tree)
        return 0
    argv(a.commit, a.tree, a.runtime_sha256, a.procedure_sha256)
    return run(a.commit, a.tree, a.runtime_sha256, a.procedure_sha256)


if __name__ == "__main__":
    # An uncaught bootstrap/receipt failure remains nonzero. Never manufacture
    # a success receipt here; the batch independently records its child status.
    sys.exit(main())
