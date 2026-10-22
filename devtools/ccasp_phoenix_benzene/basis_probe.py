# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""One separately granted BASIS-ONLY native metadata probe. No SCF/integrals/ISA."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path("/storage/project/r-cs207-0/awallace43/gits/"
            "psi4.ccasp-phoenix-benzene-pilot-20261021-science-launch-v3")
RUN = Path("/storage/project/r-cs207-0/awallace43/runs/"
           "ccasp-phoenix-benzene-pilot-20261021/basis-probe-001")
SCRATCH = Path("/storage/scratch1/3/awallace43/ccasp-phoenix-benzene-basis-probe-001")


def load_job():
    spec = importlib.util.spec_from_file_location("ccasp_basis_probe_runtime", HERE / "benzene_job.py")
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job


def prepare(commit, tree):
    job = load_job()
    if HERE.parent.parent != ROOT:
        raise ValueError("probe must use the new exact-commit science-launch-v3 source")
    job.identity(commit, tree)
    if RUN.exists() or SCRATCH.exists():
        raise ValueError("probe run/scratch must be new; no overwrite, reuse or cleanup")
    if job.sha(job.FIXTURE) != job.FIXTURE_SHA:
        raise ValueError("read-only P002 fixture identity mismatch")
    os.umask(0o027)
    for sub in ("logs", "metadata"):
        (RUN / sub).mkdir(parents=True)
    SCRATCH.mkdir()
    job.write(RUN / "metadata/preparation.json", dict(
        driver_source=str(ROOT), driver_commit=commit, driver_tree=tree,
        native_source=str(job.SOURCE), native_commit=job.BUILD_COMMIT, native_tree=job.BUILD_TREE,
        core_sha256=job.CORE_SHA, fixture=str(job.FIXTURE), fixture_sha256=job.FIXTURE_SHA,
        native_probe_executed=False))


class ControlledBasisProbeFailure(Exception):
    pass


def run(commit, tree):
    job = load_job()
    if HERE.parent.parent != ROOT:
        raise ValueError("wrong dedicated probe/driver source")
    prep = json.loads((RUN / "metadata/preparation.json").read_text())
    if prep["driver_commit"] != commit or prep["driver_tree"] != tree:
        raise ValueError("probe preparation source pin mismatch")
    if job.sha(job.FIXTURE) != job.FIXTURE_SHA:
        raise ValueError("read-only P002 fixture changed")
    psi4, handle = job.initialize_runtime(commit, tree, receipt_run=RUN, operation="basis-probe")
    from psi4.driver.p4util import OptionsState
    spec = importlib.util.spec_from_file_location("corrected_probe_pilot", HERE / "pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    tools, core = pilot.basis_tools(), psi4.core
    cfg_path = HERE / "pilot.json"
    cfg = json.loads(cfg_path.read_text())
    raw = json.loads(job.FIXTURE.read_text())
    args = SimpleNamespace(source=str(job.SOURCE), commit=job.BUILD_COMMIT,
        native_tree=job.BUILD_TREE, stage=str(job.STAGE), core_sha256=job.CORE_SHA,
        driver_source=str(ROOT), driver_commit=commit, driver_tree=tree, fixture=str(job.FIXTURE))
    binding = pilot.basis_binding(args, cfg_path)
    receipt = {}
    try:
        psi4.set_output_file(str(RUN / "metadata/native-basis.log"), False)
        psi4.set_memory(4 * 1024**3)
        molecule, _ = pilot.make_molecule(psi4, cfg, raw)
        psi4.set_options(cfg["scf"]["options"])
        tools.basis_files(job.SOURCE, job.STAGE, receipt)
        receipt["override_before"] = tools.puream_state(core)
        # Intentionally observe the documented precedence once, BASIS METADATA ONLY.
        unscoped = core.BasisSet.build(molecule, "DF_BASIS_SCF", "aug-cc-pvtz-ri", puream=0)
        receipt["unscoped_requested_puream0_actual"] = tools.native_snapshot(unscoped)
        receipt["override_after"] = tools.puream_state(core)
        tools.check_shape(receipt["unscoped_requested_puream0_actual"],
                          tools.contract()["unscoped_auxiliary_prediction"], "override demonstration")
        if receipt["override_before"] != receipt["override_after"]:
            raise ValueError("override demonstration changed PUREAM")
        main, auxiliary = tools.build_bases(core, OptionsState, molecule, receipt)
        # Controlled callback seam only: never patch a production native function.
        errors = []
        for changed in (True, False):
            stash = OptionsState(["PUREAM"])
            audit = {}
            error = ControlledBasisProbeFailure("controlled error before any native AUX call")
            def fail():
                raise error
            try:
                core.set_global_option("PUREAM", True)
                if not changed:
                    core.revoke_global_option_changed("PUREAM")
                try:
                    tools.scoped_auxiliary(core, OptionsState, molecule, audit, builder=fail)
                except ControlledBasisProbeFailure as observed:
                    if observed is not error:
                        raise ValueError("controlled error identity was changed")
                else:
                    raise ValueError("controlled error did not propagate")
                errors.append(audit)
            finally:
                stash.restore()
        receipt["controlled_error_restoration"] = errors
        job.snapshot("after-basis-probe", receipt_run=RUN)
        manifest = dict(schema="ccasp-benzene-native-basis-admission/1",
            status="basis_metadata_only_complete", no_scf=True, binding=binding, admission=receipt,
            measurement="native BasisSet metadata and normalization only; not molecular parity or binary/source attestation",
            numerical_threads=core.get_num_threads(), psi4_memory_bytes=int(core.get_memory()),
            einsums_handle_retained=bool(handle))
        path = RUN / "basis-admission.json"
        job.write(path, manifest)
        job.write(RUN / "metadata/probe-result.json", dict(
            status="basis_metadata_only_complete", no_scf=True, manifest=str(path),
            manifest_sha256=job.sha(path), driver_commit=commit, native_commit=job.BUILD_COMMIT))
        return 0
    except (Exception, SystemExit):
        job.write(RUN / "metadata/basis-probe-failure.json",
                  dict(traceback=traceback.format_exc(), partial_admission=receipt, no_scf=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            prepare(args.commit, args.tree)
            return 0
        return run(args.commit, args.tree)
    except (Exception, SystemExit):
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if args.mode == "run" and (RUN / "metadata").is_dir():
            load_job().write(RUN / "metadata/bootstrap-failure.json", dict(traceback=error, no_scf=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
