import argparse
import json
import traceback
from pathlib import Path

import psi4
from psi4 import core
from psi4.driver.procrouting import proc, proc_util
from psi4.driver.procrouting.sapt import sapt_proc


GEOMETRY = """0 1
Ne 0 0 0
--
0 1
Ne 0 0 3.0
units angstrom
symmetry c1
no_reorient
no_com
"""


def _safe_variable(name):
    try:
        return core.variable(name)
    except Exception:
        return None


def _molecule_signature(molecule):
    payload = molecule.to_schema(dtype=2)
    return json.dumps(payload, sort_keys=True)


def _configure(scenario, name):
    core.clean()
    core.clean_options()
    psi4.set_memory("1 GiB")
    psi4.set_num_threads(1)
    core.set_output_file("/dev/null", False)
    mol = psi4.geometry(GEOMETRY)
    options = {
        "basis": "sto-3g",
        "freeze_core": False,
        "guess": "sad",
        "orbital_optimizer_package": "internal",
        "sapt_dft_do_hybrid": False,
        "sapt_dft_grac_shift_a": 0.0,
        "sapt_dft_grac_shift_b": 0.0,
        "sapt_dft_use_einsums": False,
        "scf_type": "df",
        "sapt_dft_do_ddft": True,
        "sapt_dft_do_dhf": True,
        "sapt_dft_do_disp": False,
        "sapt_dft_do_fsapt": "none",
        "sapt_dft_functional": "svwn",
    }
    if "-d3" in name.lower() or "-d4" in name.lower():
        options["sapt_dft_functional"] = "pbe0"
    if scenario == "localization":
        options.update(
            {
                "sapt_dft_do_dhf": False,
                "sapt_dft_do_ddft": False,
                "sapt_dft_do_fsapt": "FISAPT",
                "fisapt_fsapt_filepath": "none",
            }
        )
    psi4.set_options(options)
    sapt_dimer, monomer_a, monomer_b = proc_util.prepare_sapt_molecule(mol, "dimer")
    return mol, {
        "dimer": _molecule_signature(sapt_dimer),
        "monomer_a": _molecule_signature(monomer_a),
        "monomer_b": _molecule_signature(monomer_b),
    }


def _manifest_summary(checkpoint_dir: str):
    if not checkpoint_dir:
        return None
    manifest_path = Path(checkpoint_dir) / "saptdft_state.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    return {
        "path": str(manifest_path),
        "completed_stages": sorted(manifest.get("completed_stages", {}).keys()),
        "manifest": manifest,
    }


def _install_scf_guards(*, checkpoint_dir: str, guard_jk: bool, forbidden_banners, molecule_signatures):
    manifest = _manifest_summary(checkpoint_dir)
    completed_stages = set(manifest["completed_stages"] if manifest is not None else [])
    current_stage = {"name": None}
    run_scf_counts = {"dimer": 0, "monomer_a": 0, "monomer_b": 0}

    banner_stage_map = {
        "SAPT(DFT): delta HF Dimer": "hf_dimer_scf",
        "SAPT(DFT): delta HF Monomer A": "hf_monomer_a_scf",
        "SAPT(DFT): delta HF Monomer B": "hf_monomer_b_scf",
        "SAPT(DFT): Dimer for Localization": "dimer_localization_scf",
        "SAPT(DFT): DFT Monomer A": "monomer_a_dft_scf",
        "SAPT(DFT): DFT Monomer B": "monomer_b_dft_scf",
    }

    def boom(message):
        raise AssertionError(message)

    def _classify_run_scf_stage(molecule):
        signature = _molecule_signature(molecule)
        if signature == molecule_signatures["dimer"]:
            run_scf_counts["dimer"] += 1
            return "delta_dft_dimer_scf"
        if signature == molecule_signatures["monomer_a"]:
            run_scf_counts["monomer_a"] += 1
            return "delta_dft_monomer_a_scf"
        if signature == molecule_signatures["monomer_b"]:
            run_scf_counts["monomer_b"] += 1
            return "delta_dft_monomer_b_scf"
        boom("Unclassified run_scf molecule encountered during checkpoint restart")

    original_proc_scf_helper = proc.scf_helper
    original_proc_run_scf = proc.run_scf

    forbidden_banner_set = set(forbidden_banners)

    def guarded_scf_helper(*args, **kwargs):
        banner = kwargs.get("banner")
        stage = banner_stage_map.get(banner)
        restore_stage = current_stage["name"]
        if stage is None and banner is None and restore_stage is not None:
            stage = restore_stage
        if stage is None:
            boom(f"Unclassified scf_helper banner encountered during checkpoint restart: {banner!r}")
        if banner in forbidden_banner_set or stage in completed_stages:
            boom(f"SCF helper replayed completed checkpoint stage {stage}: {banner}")
        current_stage["name"] = stage
        try:
            return original_proc_scf_helper(*args, **kwargs)
        finally:
            current_stage["name"] = restore_stage

    def guarded_run_scf(*args, **kwargs):
        molecule = kwargs.get("molecule")
        if molecule is None:
            boom("run_scf called without molecule during checkpoint restart")
        stage = _classify_run_scf_stage(molecule)
        if stage in completed_stages:
            boom(f"run_scf replayed completed checkpoint stage {stage}")
        current_stage["name"] = stage
        try:
            return original_proc_run_scf(*args, **kwargs)
        finally:
            current_stage["name"] = None

    proc.scf_helper = guarded_scf_helper
    proc.run_scf = guarded_run_scf
    sapt_proc.scf_helper = guarded_scf_helper
    sapt_proc.run_scf = guarded_run_scf

    for attr in ["compute_energy", "guess", "diis"]:
        original = getattr(core.HF, attr)

        def guarded(self, *args, _original=original, _attr=attr, **kwargs):
            stage = current_stage["name"]
            if stage in completed_stages:
                boom(f"core.HF.{_attr} replayed completed checkpoint stage {stage}")
            return _original(self, *args, **kwargs)

        setattr(core.HF, attr, guarded)

    if guard_jk:
        def guarded_jk_build(*args, **kwargs):
            boom("JK.build called during guarded final checkpoint restart")
        core.JK.build = guarded_jk_build
        sapt_proc._saptdft_prepare_restored_scf = guarded_jk_build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "stop", "restart", "restart_with_guards"])
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--stop-after")
    parser.add_argument("--name", default="sapt(dft)")
    parser.add_argument("--scenario", default="default", choices=["default", "localization"])
    parser.add_argument("--guard-jk", action="store_true")
    parser.add_argument("--forbid-banner", action="append", default=[])
    args = parser.parse_args()

    mol, molecule_signatures = _configure(args.scenario, args.name)
    if args.mode == "restart_with_guards":
        _install_scf_guards(
            checkpoint_dir=args.checkpoint_dir,
            guard_jk=args.guard_jk,
            forbidden_banners=args.forbid_banner,
            molecule_signatures=molecule_signatures,
        )

    kwargs = {"molecule": mol}
    if args.checkpoint_dir:
        kwargs["checkpoint_dir"] = args.checkpoint_dir
    if args.stop_after:
        kwargs["checkpoint_stop_after"] = args.stop_after

    summary = {
        "mode": args.mode,
        "checkpoint_dir": args.checkpoint_dir,
        "stop_after": args.stop_after,
        "name": args.name,
        "scenario": args.scenario,
    }
    try:
        psi4.energy(args.name, **kwargs)
        summary["status"] = "ok"
    except RuntimeError as exc:
        if args.stop_after and str(exc) == f"SAPT(DFT) checkpoint stop after {args.stop_after}":
            summary["status"] = "stopped"
            summary["message"] = str(exc)
        else:
            summary["status"] = "error"
            summary["error_type"] = type(exc).__name__
            summary["error"] = str(exc)
            summary["traceback"] = traceback.format_exc()
    except Exception as exc:
        summary["status"] = "error"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()

    summary["current_energy"] = _safe_variable("CURRENT ENERGY")
    summary["sapt_total_energy"] = _safe_variable("SAPT TOTAL ENERGY")
    summary["saptdft_total_energy"] = _safe_variable("SAPT(DFT) TOTAL ENERGY")
    summary["elst10_r"] = _safe_variable("Elst10,r")
    summary["energy"] = summary["sapt_total_energy"]

    manifest = _manifest_summary(args.checkpoint_dir)
    if manifest is not None:
        summary["manifest_path"] = manifest["path"]
        summary["completed_stages"] = manifest["completed_stages"]
        summary["manifest"] = manifest["manifest"]

    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if summary.get("status") in {"ok", "stopped"} else 1)


if __name__ == "__main__":
    main()
