# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Prepare/run ONE four-case study. Author uses only engine-free tests."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE/file)
    item = importlib.util.module_from_spec(spec)
    sys.modules[name] = item
    spec.loader.exec_module(item)
    return item


Q = module("_ccasp_convergence_original_probe", "qgrid_job.py")
C = module("_ccasp_convergence_contracts", "qgrid_convergence.py")
R = Q.GENERIC.for_plan(HERE/"convergence_plan.json")
PLAN = R.PLAN


def validate_plan():
    work = C.accounting(PLAN["cases"])
    if (R.sha(HERE/"qgrid_plan.json") != PLAN["original_capture_plan_sha256"]
            or PLAN["bounds"]["max_values"] != work["values"]
            or PLAN["bounds"]["tile_points"] != 1024
            or PLAN["bounds"]["deadline_seconds"] != 1500
            or PLAN["bounds"]["final_receipt_reservation_files"] != 3
            or PLAN["bounds"]["final_receipt_reservation_bytes"] != 2*4*1024**2+65536
            or PLAN["bounds"]["max_output_bytes"] < work["estimated_output_bytes"]
            or PLAN["allocation"] != Q.PLAN["allocation"]):
        raise ValueError("exact study plan/cumulative bounds/admitted allocation differ")
    return work


def argv(commit, tree, runtime_sha, derivative_sha, *, test_only=True):
    Q.arguments(commit, tree, runtime_sha, derivative_sha)
    if runtime_sha is None or derivative_sha is None:
        raise ValueError("actual reviewed generated manifest pins required")
    return ["/opt/slurm/current/bin/sbatch", "--test-only" if test_only else "--parsable",
        "--export=NONE", "--chdir="+PLAN["run"],
        str(Path(PLAN["source"])/"devtools/ccasp_phoenix_benzene/convergence.sbatch"),
        commit, tree, runtime_sha, derivative_sha]


def binding_for(commit, tree):
    Q.arguments(commit, tree)
    validate_plan()
    binding = R.source_binding(commit, tree)
    changed = R.OLD.git(R.ROOT, "diff", "--name-only", PLAN["base"], commit).splitlines()
    if any(not p.startswith("devtools/ccasp_phoenix_benzene/") for p in changed):
        raise ValueError("changes outside convergence tools-only grant")
    # Every pre-existing tool, launcher, config and production file stays frozen.
    previous = R.OLD.git(R.ROOT, "ls-tree", "-r", "--name-only", PLAN["base"]).splitlines()
    if any(p in previous for p in changed):
        raise ValueError("convergence must not change existing files")
    if R.OLD.git(R.ROOT, "rev-parse", commit+":psi4") != R.OLD.git(
            R.ROOT, "rev-parse", PLAN["base"]+":psi4"):
        raise ValueError("production Python/native/GBS differs")
    return binding


def verify_inputs(capture_root=None, baseline_root=None):
    """Stdlib-only lossless saved-byte validation; no numerical engine."""
    validate_plan()
    capture = Q.verify_capture(capture_root or Q.PLAN["capture"]["root"])
    root = Path(baseline_root or PLAN["baseline_probe"]["root"])
    pins = PLAN["baseline_probe"]["files"]
    raw = {name: Q.read_pinned(root/name, pin) for name, pin in pins.items()}
    docs = {name: json.loads(data) for name, data in raw.items() if name.endswith(".json")}
    result = docs["results/result.json"]
    diagnostic = docs["results/grid-diagnostic.json"]
    grid = docs["results/grid-identity.json"]
    batch, terminal = (docs["metadata/"+n+"-terminal.json"] for n in ("batch", "payload"))
    runtime, preparation = docs["metadata/runtime.json"], docs["metadata/preparation.json"]
    if (result["job"] != "14308581" or result["probe_complete"] is not True
            or result["properties_accepted"] is not False or result["after_runtime_closure"] != "passed"
            or result["commit"] != PLAN["base"]
            or result["tree"] != "1fae89606b94f8d54dbf9f621f96ab03e114657f"
            or result["native_commit"] != R.OLD.BUILD_COMMIT
            or result["native_core_sha256"] != R.OLD.CORE_SHA
            or result["diagnostic_sha256"] != pins["results/grid-diagnostic.json"]
            or result["runtime_manifest_sha256"] != preparation["runtime_manifest_sha256"]
            or result["runtime_manifest_sha256"] != runtime["runtime_manifest_sha256"]
            or result["derivative_admission_sha256"] != preparation["derivative_sha256"]
            or runtime["binding"] != preparation["binding"]
            or batch["exit_code"] != 0 or batch["term_received"] != 0
            or batch["child_unsignalled_zero"] != 1 or batch["complete"] is not True
            or terminal["complete"] is not True or terminal["after_runtime_closure"] != "passed"
            or raw["metadata/before-capture.json"] != raw["metadata/after-capture.json"]
            or grid["measured_sha256"] != C.BASELINE_HASH
            or grid["hash_verified_before_integration"] is not True
            or diagnostic["grid"] != Q.PLAN["grid"]
            or diagnostic["original_marker_sha256"] != capture["marker_sha256"]
            or diagnostic["original_anchor_sha256"] != capture["anchor_sha256"]
            or docs["results/basis-identities-grid-only.json"]["auxiliary"] != capture["basis"]["auxiliary"]):
        raise ValueError("actual14308581 source/runtime/grid/capture closure chain differs")
    vectors = {}
    for key in C.VECTOR_KEYS:
        relative = "results/"+key+".f64"
        if diagnostic["vectors"][key] != dict(file=key+".f64", dtype="<f8",
                shape=[1146], nbytes=9168, sha256=pins[relative]):
            raise ValueError("original baseline vector descriptor differs")
        vectors[key] = struct.unpack("<1146d", raw[relative])
        if not all(map(math.isfinite, vectors[key])):
            raise ValueError("nonfinite baseline vector")
    return dict(capture=capture, baseline_vectors=vectors,
                baseline_runtime=runtime, baseline_diagnostic=diagnostic)


def receipt(binding, runtime_sha, derivative_sha):
    return dict(schema="ccasp-convergence-preparation/1", binding=binding,
        plan_sha256=R.sha(HERE/"convergence_plan.json"),
        script_sha256=R.sha(HERE/"convergence.sbatch"),
        original_capture=Q.PLAN["capture"], baseline_probe=PLAN["baseline_probe"],
        cases=PLAN["cases"], cumulative=validate_plan(), bounds=PLAN["bounds"],
        runtime_manifest_sha256=runtime_sha, derivative_sha256=derivative_sha,
        test_only_argv=argv(binding["commit"], binding["tree"], runtime_sha, derivative_sha),
        future_submit_argv_not_executed=argv(binding["commit"], binding["tree"],
            runtime_sha, derivative_sha, test_only=False),
        no_native_import_build_or_submission=True)


def prepare(commit, tree):
    """FUTURE coordinator metadata-only action, never run by author."""
    binding = binding_for(commit, tree)
    verify_inputs()
    R.prior_probe()
    cfg = json.loads((HERE/"pilot.json").read_text())
    if R.sha(PLAN["fixture"]) != cfg["fixture_sha256"]:
        raise ValueError("private geometry hash differs")
    runtime, run, scratch = (Path(PLAN[k]) for k in ("runtime", "run", "scratch"))
    if any(p.exists() or p.is_symlink() for p in (runtime, run, scratch)):
        raise ValueError("exclusive new paths must not be reused")
    runtime.mkdir(mode=0o700, parents=True, exist_ok=False)
    run.mkdir(mode=0o700, parents=True, exist_ok=False)
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ("logs", "metadata"):
        (run/name).mkdir(mode=0o700)
    manifest = R.compose(R.ROOT, R.OLD.STAGE, runtime/"stage", binding)
    runtime_sha = R.write(runtime/"runtime-manifest.json", manifest)
    R.verify_runtime(runtime/"runtime-manifest.json", runtime_sha, binding)
    derivative_sha = R.write(runtime/"basis-derivative-admission.json",
                             R.derived_document(binding, runtime_sha))
    document = receipt(binding, runtime_sha, derivative_sha)
    R.write(run/"metadata/preparation.json", document)
    print(json.dumps(document, indent=2))


def verify_preparation(binding, runtime_sha, derivative_sha):
    if json.loads((Path(PLAN["run"])/"metadata/preparation.json").read_text()) != receipt(
            binding, runtime_sha, derivative_sha):
        raise ValueError("actual reviewed preparation differs")
    if R.pinned_json(Path(PLAN["runtime"])/"basis-derivative-admission.json",
            derivative_sha) != R.derived_document(binding, runtime_sha):
        raise ValueError("original probe/new derivative binding differs")
    verify_inputs()


def storage_bounds(*, reserve=0, reserve_files=0):
    """Aggregate, regular-only study/scratch/log budgets, not reset per case."""
    bounds = PLAN["bounds"]
    roots = {role: Path(PLAN[role]) for role in ("run", "scratch")}
    def validate_roots():
        for root in roots.values():
            if not stat.S_ISDIR(root.lstat().st_mode):
                raise ValueError("study/scratch root must remain a real directory")
    def traversal_error(error):
        raise error
    validate_roots()
    output = logs = scratch = files = 0
    for role, root in roots.items():
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=traversal_error):
            for name in dirs+names:
                path = Path(directory)/name
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("nonregular/symlink study artifact")
                files += 1
                if files > bounds["max_files"]:
                    raise ValueError("whole study artifact count bound")
                is_log = role == "run" and path.is_relative_to(root/"logs")
                if info.st_size > bounds["max_log_bytes" if is_log else "max_file_bytes"]:
                    raise ValueError("whole study per-file bound")
                if role == "scratch":
                    scratch += info.st_size
                elif is_log:
                    logs += info.st_size
                else:
                    output += info.st_size
    validate_roots()
    if (files+reserve_files > bounds["max_files"]
            or output+reserve > bounds["max_output_bytes"] or scratch > bounds["max_scratch_bytes"]
            or logs > 2*bounds["max_log_bytes"]
            or output+reserve+logs+scratch > bounds["max_total_bytes"]):
        raise ValueError("whole study output/scratch/log aggregate bound")
    return dict(output_bytes=output, log_bytes=logs, scratch_bytes=scratch, files=files)


def basis_only(psi4, binding, runtime_sha, derivative_sha, capture, out):
    from psi4.driver.p4util import OptionsState
    cfg = json.loads((HERE/"pilot.json").read_text())
    if R.sha(PLAN["fixture"]) != cfg["fixture_sha256"]:
        raise ValueError("original fixture changed")
    raw = json.loads(Path(PLAN["fixture"]).read_text())
    if (raw["source_sha256"] != cfg["source_dataframe_sha256"]
            or raw["system_id"] != "24_Benzene-Benzene_pi-pi_1.00"
            or raw["dataframe_index"] != 3806 or raw["fragment"] != "A"
            or raw["charge"] != 0 or raw["multiplicity"] != 1):
        raise ValueError("original geometry attribution differs")
    psi4.set_output_file(str(out/"psi4.out"), False)
    psi4.set_memory(PLAN["allocation"]["psi4_memory_bytes"])
    psi4.set_num_threads(1)
    prior = module("_ccasp_convergence_geometry_only", "pilot.py")
    molecule, xyz = prior.make_molecule(psi4, cfg, raw)
    psi4.set_options(dict(basis=cfg["scf"]["options"]["basis"], puream=True))
    admission = {}
    try:
        _, auxiliary = R.admit_basis(psi4.core, OptionsState, molecule, admission, binding,
            runtime_sha, Path(PLAN["runtime"])/"basis-derivative-admission.json",
            derivative_sha, lambda aux: aux)
        if admission["auxiliary"] != capture["basis"]["auxiliary"]:
            raise ValueError("rebuilt exact AUX snapshot differs from capture")
    finally:
        R.write(out/"basis-identities-grid-only.json", admission)
    snapshot = admission["auxiliary"]
    centres = [[float.fromhex(x) for x in c] for c in snapshot["centres_bohr_hex"]]
    if centres != xyz:
        raise ValueError("absolute geometry/order changed")
    shells = []
    for shell in snapshot["ordered_shells"]:
        item = psi4.core.IsaGaussianShell()
        item.centre, item.l = shell["centre"], shell["l"]
        item.exponents = [float.fromhex(x) for x in shell["exponents_hex"]]
        item.coefficients = [float.fromhex(x) for x in shell["native_normalized_coefficients_hex"]]
        shells.append(item)
    explicit = psi4.core.IsaExplicitBasis(psi4.core.IsaBasisRole.MolecularAux,
        psi4.core.IsaBasisRepresentation.Cartesian, centres, shells)
    Q.P.all_centres(explicit, 12, 1146)
    del auxiliary
    return molecule, explicit, xyz


def one_case(psi4, molecule, explicit, case, data, xyz, binding, check, previous):
    """Only this stack frame owns a grid. It returns compact vectors, never grid."""
    import numpy as np
    out = Path(PLAN["run"])/case["output"]
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    capture = data["capture"]
    check()
    options = psi4.core.IsaGridOptions()
    C.configure(options, case)
    native = psi4.core.IsaGrid(molecule.clone(), options)
    actual = C.layout(native, case)
    # One component list at a time; don't keep four multi-million Python lists.
    grid = np.empty((case["points"], 4), dtype="<f8")
    for column, name in enumerate(("x", "y", "z", "w")):
        check()
        grid[:, column] = getattr(native, name)()
    del native
    digest = C.canonical_hash(grid, case["points"])
    identity = dict(case=case, measured_sha256=digest, actual_layout=actual,
        selected_points=C.selection(actual), geometry_bohr=xyz, source=binding,
        runtime=data["runtime_identity"], basis_fingerprints=PLAN["fingerprints"],
        canonical=Q.PLAN["grid"]["canonical"], original_capture=Q.PLAN["capture"],
        baseline_probe=PLAN["baseline_probe"], numerical_threads=1, tile_points=1024)
    # Preserve actual identity even when gate refuses; no collocation yet.
    R.write(out/"grid-identity.json", identity)
    C.hash_gate(case, digest, previous)
    measured = Q.P.stream(explicit, grid, capture["rows"], dict(PLAN["bounds"],
        max_values=case["points"]*1146),
        selected_points=tuple(p["index"] for p in identity["selected_points"]),
        selected_columns=Q.P.selected_columns(capture["rows"]), check_deadline=check)
    del grid
    check()
    descriptors = {key: Q.P.write_vector(out/(key+".f64"), measured[key], expected=1146)
                   for key in C.VECTOR_KEYS}
    error = measured["U"]-np.asarray(capture["vectors"]["q"])
    descriptors["U_minus_q"] = Q.P.write_vector(out/"U_minus_q.f64", error, expected=1146)
    reproducibility = None
    if case["name"] == "baseline":
        reproducibility = C.remeasurement(measured, data["baseline_vectors"])
        R.write(out/"remeasurement.json", reproducibility)
    comparisons = {key: C.diagnostics(measured[key], capture["vectors"]["q"],
        capture["vectors"]["QT_c"], capture["rows"], baseline=case["name"] == "baseline")
        for key in ("U", "U_compensated")}
    report = dict(schema="ccasp-convergence-case/1", identity=identity, vectors=descriptors,
        comparison=comparisons, collocation_checks=measured["collocation_checks"],
        accumulation=measured["accumulation"], error_bound_scope=measured["error_bound_scope"],
        tiles=measured["tiles"], elapsed_seconds=time.monotonic()-started,
        remeasurement_exact=None if reproducibility is None else reproducibility["all_exact"],
        collocation_admission="sampled measured differences only; no new acceptance tolerance",
        science_pass=False, properties_accepted=False, no_scf_isa_response=True)
    storage_bounds(reserve=4*1024**2, reserve_files=1)
    report_sha = R.write(out/"diagnostic.json", report)
    check()
    if reproducibility is not None and not reproducibility["all_exact"]:
        raise ValueError("baseline vector byte reproduction failed; retained exact differences; no fallback")
    # No grid/native/matrix or large report escapes to the next case.
    return dict(name=case["name"], grid_sha256=digest, diagnostic_sha256=report_sha,
                U=measured["U"], U_compensated=measured["U_compensated"])


def sequential(execute, check):
    """All-case completion, no retry, retaining only compact results."""
    completed = []
    for case in PLAN["cases"]:
        check()
        result = execute(case, [r["grid_sha256"] for r in completed])
        if result["name"] != case["name"]:
            raise ValueError("sequential case identity differs")
        completed.append(result)
        check()
    return completed


def science(psi4, binding, runtime_sha, derivative_sha, check):
    R.compute_guard()
    validate_plan()
    data = verify_inputs()
    root, out = Path(PLAN["run"]), Path(PLAN["run"])/"results"
    # Same numerical runtime is a declared reproducibility condition.
    runtime = json.loads((root/"metadata/runtime.json").read_text())
    if any(runtime[k] != data["baseline_runtime"][k] for k in ("python", "numpy", "adcc", "node")):
        raise ValueError("baseline numerical runtime identity differs")
    for role, key in (("scratch", "required_free_scratch_bytes"), ("run", "required_free_run_bytes")):
        free = os.statvfs(PLAN[role])
        if free.f_bavail*free.f_frsize < PLAN["bounds"][key]:
            raise ValueError("whole study free storage admission")
    scratch = Path(PLAN["scratch"])/("payload-"+os.environ["SLURM_JOB_ID"])
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ("TMPDIR", "SCRATCH", "PSI_SCRATCH"):
        os.environ[name] = str(scratch)
    molecule, explicit, xyz = basis_only(psi4, binding, runtime_sha, derivative_sha,
                                        data["capture"], out)
    data["runtime_identity"] = dict(runtime_manifest_sha256=runtime_sha,
        derivative_admission_sha256=derivative_sha,
        runtime_receipt_sha256=R.sha(root/"metadata/runtime.json"),
        basis_receipt_sha256=R.sha(out/"basis-identities-grid-only.json"))
    completed = sequential(lambda case, previous: one_case(psi4, molecule, explicit, case,
        data, xyz, binding, check, previous), check)
    comparisons = {}
    for key in ("U", "U_compensated"):
        effect = C.effects({r["name"]: r[key] for r in completed})
        comparisons[key] = {name: dict(values_hex=[v.hex() for v in values],
            summary=C.summary(values, data["capture"]["rows"])) for name, values in effect.items()}
    R.write(out/"comparisons.json", dict(effects=comparisons,
        interpretation="signed grid-only effects; no monotonicity assumption; no new Q/partition/response",
        cases=[{k: v for k, v in r.items() if k not in ("U", "U_compensated")} for r in completed],
        source=binding, original_capture=Q.PLAN["capture"], baseline_probe=PLAN["baseline_probe"]))
    del explicit, molecule
    check()
    return dict(study_complete=True, completed_cases=[r["name"] for r in completed],
                comparisons_sha256=R.sha(out/"comparisons.json"))


class Terminated(BaseException):
    pass


def run(commit, tree, runtime_sha, derivative_sha):
    Q.arguments(commit, tree, runtime_sha, derivative_sha)
    if runtime_sha is None or derivative_sha is None:
        raise ValueError("run requires reviewed manifest pins")
    R.compute_guard()
    started = time.monotonic()
    deadline = C.deadline(PLAN["bounds"]["deadline_seconds"])
    last_scan = [float("-inf")]
    def check():
        deadline()
        if time.monotonic()-last_scan[0] >= 1:
            storage_bounds(reserve=65536)
            last_scan[0] = time.monotonic()
    def terminate(signum, frame):
        raise Terminated("allocation signal "+str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, terminate)
    resource.setrlimit(resource.RLIMIT_FSIZE, (PLAN["bounds"]["max_file_bytes"],)*2)
    root = Path(PLAN["run"])
    out = root/"results"
    out.mkdir(mode=0o700, exist_ok=False)
    record = dict(status="failed", study_complete=False, science_pass=False, properties_accepted=False,
        commit=commit, tree=tree, runtime_manifest_sha256=runtime_sha,
        derivative_admission_sha256=derivative_sha, job=os.environ["SLURM_JOB_ID"],
        native_commit=R.OLD.BUILD_COMMIT, native_core_sha256=R.OLD.CORE_SHA)
    handles = []
    try:
        binding = binding_for(commit, tree)
        verify_preparation(binding, runtime_sha, derivative_sha)
        check()
        psi4, _ = R.initialize(binding, Path(PLAN["runtime"])/"runtime-manifest.json",
                               runtime_sha, root, handles)
        check()
        record.update(science(psi4, binding, runtime_sha, derivative_sha, check))
        if (record["study_complete"] is not True
                or record["completed_cases"] != [c["name"] for c in C.CASES]):
            raise ValueError("whole study requires all four successful cases")
        check()
        record["status"] = "four_grid_diagnostic_complete"
    except BaseException as error:
        record.update(status="failed", study_complete=False, error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        if handles:
            try:
                R.OLD.snapshot("after-capture", receipt_run=root)
                R.verify_runtime(Path(PLAN["runtime"])/"runtime-manifest.json", runtime_sha,
                                 binding, hash_core=True)
                record["after_runtime_closure"] = "passed"
            except BaseException as error:
                record.update(status="failed", study_complete=False, after_runtime_closure="failed",
                    closure_error=f"{type(error).__name__}: {error}", closure_traceback=traceback.format_exc())
        else:
            record.update(study_complete=False, after_runtime_closure="not initialized")
    # Never throttle the final scan: include closure artifacts and fresh logs/
    # scratch. Reserve two JSON writer caps (4MiB each), plus 64KiB for the
    # shell's fixed batch-terminal.json: reserve all THREE before admitting success.
    try:
        deadline()
        record["final_storage"] = storage_bounds(
            reserve=PLAN["bounds"]["final_receipt_reservation_bytes"],
            reserve_files=PLAN["bounds"]["final_receipt_reservation_files"])
        deadline()
        record["final_budgets"] = "passed"
    except BaseException as error:
        record.update(status="failed", study_complete=False, final_budgets="failed",
            budget_error=f"{type(error).__name__}: {error}",
            budget_traceback=traceback.format_exc())
    record["elapsed_seconds"] = time.monotonic()-started
    R.write(out/"result.json", record)
    code = 0 if record["study_complete"] else 1
    R.write(root/"metadata/payload-terminal.json", dict(exit_code=code, complete=code == 0,
        study_complete=record["study_complete"], science_pass=False, properties_accepted=False,
        after_runtime_closure=record["after_runtime_closure"]))
    return code  # retained exact CDLL handle alive through final receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--runtime-sha256")
    parser.add_argument("--derivative-sha256")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.runtime_sha256 or args.derivative_sha256:
            parser.error("prepare cannot accept generated hashes")
        prepare(args.commit, args.tree)
        return 0
    if not args.runtime_sha256 or not args.derivative_sha256:
        parser.error("run requires actual reviewed manifest hashes")
    return run(args.commit, args.tree, args.runtime_sha256, args.derivative_sha256)


if __name__ == "__main__":
    sys.exit(main())
