# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""New exact-grid-only allocation path. Prepare is metadata-only, never submits."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
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


GENERIC = module("_ccasp_grid_runtime_factory", "integrated_runtime.py")
R = GENERIC.for_plan(HERE/"qgrid_plan.json")
P = module("_ccasp_grid_reductions", "qgrid_probe.py")
PLAN = R.PLAN


def arguments(commit, tree, runtime_sha=None, derivative_sha=None):
    for name, value, length in (("commit", commit, 40), ("tree", tree, 40),
                               ("runtime", runtime_sha, 64), ("derivative", derivative_sha, 64)):
        if value is None and length == 64:
            continue
        if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{"+str(length)+"}", value):
            raise ValueError("invalid "+name+" identity argument")


def argv(commit, tree, runtime_sha, derivative_sha, *, test_only=True):
    arguments(commit, tree, runtime_sha, derivative_sha)
    if runtime_sha is None or derivative_sha is None:
        raise ValueError("actual reviewed manifest hashes required")
    return ["/opt/slurm/current/bin/sbatch", "--test-only" if test_only else "--parsable",
        "--export=NONE", "--chdir="+PLAN["run"],
        str(Path(PLAN["source"])/"devtools/ccasp_phoenix_benzene/qgrid_probe.sbatch"),
        commit, tree, runtime_sha, derivative_sha]


def binding_for(commit, tree):
    arguments(commit, tree)
    binding = R.source_binding(commit, tree)
    # Production Python is byte-identical to the capture base, as are native/GBS.
    changed = R.OLD.git(R.ROOT, "diff", "--name-only", PLAN["base"], commit).splitlines()
    if any(not name.startswith("devtools/ccasp_phoenix_benzene/") for name in changed):
        raise ValueError("qgrid probe changed source outside tools-only grant")
    if R.OLD.git(R.ROOT, "rev-parse", commit+":psi4") != R.OLD.git(
            R.ROOT, "rev-parse", PLAN["base"]+":psi4"):
        raise ValueError("scientific Python/C++/GBS tree differs from exact capture")
    return binding


def read_pinned(path, pin):
    """Small no-follow/nonblocking regular input; never copy/overwrite old capture."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 4*1024**2:
            raise ValueError("capture input type/size mismatch")
        data = stream.read(info.st_size+1)
        after = os.fstat(stream.fileno())
    if (len(data) != info.st_size or info.st_size != after.st_size
            or info.st_mtime_ns != after.st_mtime_ns
            or hashlib.sha256(data).hexdigest() != pin):
        raise ValueError("immutable capture input hash mismatch")
    return data


def verify_capture(root):
    root = Path(root)
    pins = PLAN["capture"]["files"]
    raw = {name: read_pinned(root/name, pin) for name, pin in pins.items()}
    marker = json.loads(raw["results/diagnostic/node-01/pre-lw-complete.json"])
    context = json.loads(raw["results/diagnostic/context.json"])
    anchor = json.loads(raw["results/diagnostic/anchor/manifest.json"])
    basis = json.loads(raw["results/basis-identities-before-scf.json"])
    source = ast.literal_eval(marker["source_repr"])
    diagnostics = source["partition"]["diagnostics"]
    cfg = json.loads((HERE/"pilot.json").read_text())
    if (marker["anchor_manifest_sha256"] != pins["results/diagnostic/anchor/manifest.json"]
            or marker["properties_accepted"] is not False
            or diagnostics["integration_grid_sha256"] != PLAN["grid"]["sha256"]
            or diagnostics["integration_grid_points"] != PLAN["grid"]["points"]
            or context["context"]["basis_receipt_sha256"] != pins["results/basis-identities-before-scf.json"]
            or context["identities"]["driver"]["source_commit"] != PLAN["base"]
            or context["identities"]["fixture_sha256"] != cfg["fixture_sha256"]
            or context["identities"]["config_sha256"] != R.sha(HERE/"pilot.json")
            or context["context"]["original_probe_sha256"] != PLAN["prior_probe_sha256"]
            or anchor["metadata"]["Q_sha256"] != source["partition"]["q_sha256"]
            or anchor["metadata"]["Q_shape"] != [300, 1146]):
        raise ValueError("original marker/anchor/grid/basis/fixture chain mismatch")
    for role, pin in PLAN["fingerprints"].items():
        snap = basis[role]
        if (snap["fingerprint_sha256"] != pin
                or R.BASIS.digest({k: v for k, v in snap.items() if k != "fingerprint_sha256"}) != pin):
            raise ValueError("captured exact normalized-shell fingerprint differs")
    rows = P.columns(basis["auxiliary"])
    if len(rows) != 1146 or {r["centre"] for r in rows} != set(range(12)):
        raise ValueError("capture does not declare all1146/12 AUX")
    vectors = {}
    for name in ("q", "QT_c", "deltaq"):
        desc = anchor["arrays"][name]
        relative = "results/diagnostic/anchor/"+name+".f64"
        if desc != dict(dtype="<f8", file=name+".f64", nbytes=9168,
                        sha256=pins[relative], shape=[1146]):
            raise ValueError("exact q/S/delta vector descriptor mismatch")
        values = struct.unpack("<1146d", raw[relative])
        if not all(map(math.isfinite, values)):
            raise ValueError("nonfinite saved capture reduction")
        vectors[name] = values
    if any(s-q != d for s, q, d in zip(vectors["QT_c"], vectors["q"], vectors["deltaq"])):
        raise ValueError("saved S-q does not reproduce captured deltaq")
    return dict(basis=basis, rows=rows, vectors=vectors, marker_sha256=pins[
        "results/diagnostic/node-01/pre-lw-complete.json"], anchor_sha256=pins[
        "results/diagnostic/anchor/manifest.json"])


def prepare(commit, tree):
    """Coordinator-only future metadata action after publication. No imports of engines."""
    binding = binding_for(commit, tree)
    verify_capture(PLAN["capture"]["root"])
    R.prior_probe()
    cfg = json.loads((HERE/"pilot.json").read_text())
    if R.sha(PLAN["fixture"]) != cfg["fixture_sha256"]:
        raise ValueError("private geometry hash differs")
    runtime, run, scratch = (Path(PLAN[k]) for k in ("runtime", "run", "scratch"))
    if any(path.exists() or path.is_symlink() for path in (runtime, run, scratch)):
        raise ValueError("new source/runtime/run/scratch paths must not be reused")
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
    receipt = dict(schema="ccasp-qgrid-preparation/1", binding=binding,
        plan_sha256=R.sha(HERE/"qgrid_plan.json"), script_sha256=R.sha(HERE/"qgrid_probe.sbatch"),
        capture=PLAN["capture"], grid=PLAN["grid"], bounds=PLAN["bounds"],
        runtime_manifest_sha256=runtime_sha, derivative_sha256=derivative_sha,
        test_only_argv=argv(commit, tree, runtime_sha, derivative_sha),
        future_submit_argv_not_executed=argv(commit, tree, runtime_sha, derivative_sha, test_only=False),
        no_native_import_build_or_submission=True)
    R.write(run/"metadata/preparation.json", receipt)
    print(json.dumps(receipt, indent=2))


def science(psi4, binding, runtime_sha, derivative_sha, check_deadline):
    """One grid/collocation diagnostic, no molecular electronic calculation."""
    R.compute_guard()
    import numpy as np
    from psi4.driver.p4util import OptionsState
    capture = verify_capture(PLAN["capture"]["root"])
    root, out = Path(PLAN["run"]), Path(PLAN["run"])/"results"
    cfg = json.loads((HERE/"pilot.json").read_text())
    if R.sha(PLAN["fixture"]) != cfg["fixture_sha256"]:
        raise ValueError("original geometry fixture changed")
    raw = json.loads(Path(PLAN["fixture"]).read_text())
    if (raw["source_sha256"] != cfg["source_dataframe_sha256"]
            or raw["system_id"] != "24_Benzene-Benzene_pi-pi_1.00"
            or raw["dataframe_index"] != 3806 or raw["fragment"] != "A"
            or raw["charge"] != 0 or raw["multiplicity"] != 1):
        raise ValueError("original geometry attribution differs")
    scratch = Path(PLAN["scratch"])/("payload-"+os.environ["SLURM_JOB_ID"])
    free = os.statvfs(scratch.parent)
    if free.f_bavail*free.f_frsize < PLAN["bounds"]["required_free_scratch_bytes"]:
        raise ValueError("grid-only free scratch resource bound")
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ("TMPDIR", "SCRATCH", "PSI_SCRATCH"):
        os.environ[name] = str(scratch)
    psi4.set_output_file(str(out/"psi4.out"), False)
    psi4.set_memory(PLAN["allocation"]["psi4_memory_bytes"])
    psi4.set_num_threads(1)
    prior = R.module("_ccasp_grid_geometry_only", "pilot.py")
    molecule, xyz = prior.make_molecule(psi4, cfg, raw)  # pure original geometry helper, NOT main()
    psi4.set_options(dict(basis=cfg["scf"]["options"]["basis"], puream=True))
    admission = {}
    try:
        _, auxiliary = R.admit_basis(psi4.core, OptionsState, molecule, admission, binding, runtime_sha,
            Path(PLAN["runtime"])/"basis-derivative-admission.json", derivative_sha, lambda aux: aux)
        if admission["auxiliary"] != capture["basis"]["auxiliary"]:
            raise ValueError("rebuilt AUX differs from actual captured effective snapshot")
    finally:
        R.write(out/"basis-identities-grid-only.json", admission)
    snapshot = admission["auxiliary"]
    centres = [[float.fromhex(x) for x in c] for c in snapshot["centres_bohr_hex"]]
    if centres != xyz:
        raise ValueError("captured absolute frame/centre order differs")
    # The declared explicit API uses these SAME admitted native effective coef.
    shells = []
    for shell in snapshot["ordered_shells"]:
        item = psi4.core.IsaGaussianShell()
        item.centre, item.l = shell["centre"], shell["l"]
        item.exponents = [float.fromhex(x) for x in shell["exponents_hex"]]
        item.coefficients = [float.fromhex(x) for x in shell["native_normalized_coefficients_hex"]]
        shells.append(item)
    explicit = psi4.core.IsaExplicitBasis(psi4.core.IsaBasisRole.MolecularAux,
        psi4.core.IsaBasisRepresentation.Cartesian, centres, shells)
    P.all_centres(explicit, 12, 1146)
    check_deadline()
    options = psi4.core.IsaGridOptions()
    P.configure_grid(options, PLAN["grid"]["recipe"])
    native_grid = psi4.core.IsaGrid(molecule.clone(), options)
    if (native_grid.npoints() != PLAN["grid"]["points"] or native_grid.spherical_points() != 590
            or native_grid.radial_points() != 200
            or any(native_grid.atom_start(a) != a*199*590 for a in range(12))):
        raise ValueError("exact Q native grid layout differs")
    grid = np.column_stack((native_grid.x(), native_grid.y(), native_grid.z(), native_grid.w()))
    del native_grid
    grid_sha = P.validate_grid(grid, PLAN["grid"]["points"], PLAN["grid"]["sha256"])
    # No collocation/integration/comparison until the exact canonical hash passes.
    R.write(out/"grid-identity.json", dict(grid=PLAN["grid"], measured_sha256=grid_sha,
        geometry_bohr=xyz, native_core_sha256=R.OLD.CORE_SHA, hash_verified_before_integration=True))
    measured = P.stream(explicit, grid, capture["rows"], PLAN["bounds"],
        selected_points=P.selected_points(), selected_columns=P.selected_columns(capture["rows"]),
        check_deadline=check_deadline)
    del grid, explicit, auxiliary
    vectors = {}
    for key in ("U", "U_compensated", "absolute_sum", "roundoff_bound"):
        vectors[key] = P.write_vector(out/(key+".f64"), measured[key], expected=1146)
    comparison = P.compare(measured["U"], capture["vectors"]["q"], capture["vectors"]["QT_c"],
                           capture["rows"])
    comparison_compensated = P.compare(measured["U_compensated"], capture["vectors"]["q"],
                                       capture["vectors"]["QT_c"], capture["rows"])
    checks = measured["collocation_checks"]
    checks_peak = max(checks, key=lambda c: abs(c["native_minus_oracle"]))
    report = dict(schema="ccasp-exact-qgrid-diagnostic/1", properties_accepted=False,
        capture=PLAN["capture"], grid=PLAN["grid"], vectors=vectors,
        columns=capture["rows"], comparison=comparison, comparison_compensated=comparison_compensated,
        ordinary_minus_compensated_maxabs=float(np.max(np.abs(measured["U"]-measured["U_compensated"]))),
        collocation_checks=checks, worst_collocation_check=checks_peak,
        collocation_admission="measured differences only; no invented tolerance or production admission",
        accumulation=measured["accumulation"], error_bound_scope=measured["error_bound_scope"],
        tiles=measured["tiles"], original_marker_sha256=capture["marker_sha256"],
        original_anchor_sha256=capture["anchor_sha256"],
        no_scf_isa_response=True, scratch_files_created_by_probe=0)
    report_bytes = len(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    output_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    scratch_bytes = sum(p.stat().st_size for p in Path(PLAN["scratch"]).rglob("*") if p.is_file())
    if (output_bytes+report_bytes+65536 > PLAN["bounds"]["max_output_bytes"]
            or scratch_bytes > PLAN["bounds"]["max_scratch_bytes"]):
        raise ValueError("grid probe final output/scratch resource bound")
    report_sha = R.write(out/"grid-diagnostic.json", report)
    return dict(probe_complete=True, diagnostic_sha256=report_sha,
                grid_sha256=grid_sha, properties_accepted=False)


class Terminated(BaseException):
    pass


def run(commit, tree, runtime_sha, derivative_sha):
    arguments(commit, tree, runtime_sha, derivative_sha)
    if runtime_sha is None or derivative_sha is None:
        raise ValueError("run requires reviewed manifest pins")
    R.compute_guard()  # before any output creation/import/native work, even on failure
    started = time.monotonic()
    def deadline():
        if time.monotonic()-started > PLAN["bounds"]["deadline_seconds"]:
            raise TimeoutError("grid-only1500s payload work budget exceeded; no retry")
    def terminate(signum, frame):
        raise Terminated("allocation signal "+str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, terminate)
    resource.setrlimit(resource.RLIMIT_FSIZE, (PLAN["bounds"]["max_file_bytes"],)*2)
    root = Path(PLAN["run"])
    out = root/"results"
    out.mkdir(mode=0o700, exist_ok=False)
    record = dict(status="failed", probe_complete=False, science_pass=False, properties_accepted=False,
        commit=commit, tree=tree, runtime_manifest_sha256=runtime_sha,
        derivative_admission_sha256=derivative_sha, job=os.environ["SLURM_JOB_ID"],
        native_commit=R.OLD.BUILD_COMMIT, native_core_sha256=R.OLD.CORE_SHA)
    handles = []
    try:
        binding = binding_for(commit, tree)
        preparation = json.loads((root/"metadata/preparation.json").read_text())
        if (preparation["binding"] != binding
                or preparation["runtime_manifest_sha256"] != runtime_sha
                or preparation["derivative_sha256"] != derivative_sha
                or preparation["plan_sha256"] != R.sha(HERE/"qgrid_plan.json")
                or preparation["script_sha256"] != R.sha(HERE/"qgrid_probe.sbatch")
                or preparation["capture"] != PLAN["capture"] or preparation["grid"] != PLAN["grid"]
                or preparation["bounds"] != PLAN["bounds"]):
            raise ValueError("reviewed grid-only preparation/source/argv pins differ")
        verify_capture(PLAN["capture"]["root"])
        if R.pinned_json(Path(PLAN["runtime"])/"basis-derivative-admission.json",
                         derivative_sha) != R.derived_document(binding, runtime_sha):
            raise ValueError("grid-only original probe/new-driver derivative differs before imports")
        psi4, _ = R.initialize(binding, Path(PLAN["runtime"])/"runtime-manifest.json",
                               runtime_sha, root, handles)
        deadline()
        record.update(science(psi4, binding, runtime_sha, derivative_sha, deadline))
        record["status"] = "grid_diagnostic_complete"
    except BaseException as error:
        record.update(status="failed", probe_complete=False, error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        if handles:
            try:
                R.OLD.snapshot("after-capture", receipt_run=root)
                R.verify_runtime(Path(PLAN["runtime"])/"runtime-manifest.json", runtime_sha,
                                 binding, hash_core=True)
                record["after_runtime_closure"] = "passed"
            except BaseException as error:
                record.update(status="failed", probe_complete=False, after_runtime_closure="failed",
                              closure_error=f"{type(error).__name__}: {error}",
                              closure_traceback=traceback.format_exc())
        else:
            record["after_runtime_closure"] = "not initialized"
    record["elapsed_seconds"] = time.monotonic()-started
    R.write(out/"result.json", record)
    code = 0 if record["probe_complete"] else 1
    R.write(root/"metadata/payload-terminal.json", dict(exit_code=code,
        complete=code == 0, probe_complete=record["probe_complete"], science_pass=False,
        properties_accepted=False, after_runtime_closure=record["after_runtime_closure"]))
    return code  # exact retained CDLL handle remains alive through final closure/receipts


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
            parser.error("prepare does not accept generated manifest pins")
        prepare(args.commit, args.tree)
        return 0
    if not args.runtime_sha256 or not args.derivative_sha256:
        parser.error("run requires actual reviewed manifest SHA256 pins")
    return run(args.commit, args.tree, args.runtime_sha256, args.derivative_sha256)


if __name__ == "__main__":
    sys.exit(main())
