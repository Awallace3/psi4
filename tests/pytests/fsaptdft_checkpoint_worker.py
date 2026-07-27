import argparse
import json
import traceback
from pathlib import Path

import psi4
from psi4 import core
from psi4.driver.procrouting import proc
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


def _configure():
    core.clean()
    core.clean_options()
    psi4.set_memory("1 GiB")
    psi4.set_num_threads(1)
    core.set_output_file("/dev/null", False)
    mol = psi4.geometry(GEOMETRY)
    psi4.set_options(
        {
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
    )
    return mol


def _install_scf_guards(*, guard_jk: bool, forbidden_banners):
    def boom(*args, **kwargs):
        raise AssertionError("SCF convergence entry point called during checkpoint restart")

    if forbidden_banners:
        forbidden = set(forbidden_banners)
        original_proc_scf_helper = proc.scf_helper

        def guarded_scf_helper(*args, **kwargs):
            if kwargs.get("banner") in forbidden:
                raise AssertionError(f"Forbidden SCF helper replayed during checkpoint restart: {kwargs.get('banner')}")
            return original_proc_scf_helper(*args, **kwargs)

        proc.scf_helper = guarded_scf_helper
        sapt_proc.scf_helper = guarded_scf_helper
    else:
        proc.scf_helper = boom
        proc.run_scf = boom
        sapt_proc.scf_helper = boom
        sapt_proc.run_scf = boom
        for attr in ["compute_energy", "guess", "diis"]:
            setattr(core.HF, attr, boom)
    if guard_jk:
        core.JK.build = boom
        sapt_proc._saptdft_prepare_restored_scf = boom


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "stop", "restart", "restart_with_guards"])
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--stop-after")
    parser.add_argument("--name", default="sapt(dft)")
    parser.add_argument("--guard-jk", action="store_true")
    parser.add_argument("--forbid-banner", action="append", default=[])
    args = parser.parse_args()

    if args.mode == "restart_with_guards":
        _install_scf_guards(guard_jk=args.guard_jk, forbidden_banners=args.forbid_banner)

    mol = _configure()
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
    }
    try:
        psi4.energy(args.name, **kwargs)
        summary["status"] = "ok"
    except RuntimeError as exc:
        if args.mode == "stop" and str(exc) == f"SAPT(DFT) checkpoint stop after {args.stop_after}":
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
