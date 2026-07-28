import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import psi4
import qcelemental as qcel
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

RAW_IDENTITY_GEOMETRY = """0 1
He 1.25 -0.40 0.30
--
0 1
Ne 3.10 1.70 2.80
units angstrom
"""


def _safe_variable(name):
    try:
        return core.variable(name)
    except Exception:
        return None



def _collect_scalar_qcvars():
    qcvars = {}
    for key, value in core.variables().items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, int, float)):
            qcvars[str(key)] = value
    return dict(sorted(qcvars.items()))


def _serialize_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "np"):
        return np.asarray(value.np).tolist()
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _molecule_signature(molecule):
    payload = molecule.to_schema(dtype=2)
    return json.dumps(payload, sort_keys=True)


def _raw_identity_qcschema_molecule():
    geometry_angstrom = np.array(
        [
            [1.25, -0.40, 0.30],
            [3.10, 1.70, 2.80],
        ]
    )
    return {
        "symbols": ["He", "Ne"],
        "geometry": (geometry_angstrom / qcel.constants.bohr2angstroms).ravel().tolist(),
        "molecular_charge": 0,
        "molecular_multiplicity": 1,
        "fragments": [[0], [1]],
        "fragment_charges": [0, 0],
        "fragment_multiplicities": [1, 1],
    }



def _configure(scenario, name):
    core.clean()
    core.clean_options()
    psi4.set_memory("1 GiB")
    psi4.set_num_threads(1)
    core.set_output_file("/dev/null", False)
    geometry = RAW_IDENTITY_GEOMETRY if scenario == "raw_identity" else GEOMETRY
    mol = psi4.geometry(geometry)
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
        "fisapt_fsapt_filepath": "none",
    }
    if "-d3" in name.lower() or "-d4" in name.lower():
        options["sapt_dft_functional"] = "pbe0"
    if scenario == "localization":
        options.update(
            {
                "sapt_dft_do_dhf": False,
                "sapt_dft_do_ddft": False,
                "sapt_dft_do_fsapt": "FISAPT",
            }
        )
    elif scenario == "fsapt_einsums":
        options.update(
            {
                "sapt_dft_do_dhf": False,
                "sapt_dft_do_ddft": False,
                "sapt_dft_do_disp": True,
                "sapt_dft_do_fsapt": "SAPTDFT",
                "sapt_dft_functional": "HF",
                "sapt_dft_mp2_disp_alg": "FISAPT",
                "sapt_dft_use_einsums": True,
            }
        )
    elif scenario == "fsapt_fisapt":
        options.update(
            {
                "sapt_dft_do_dhf": False,
                "sapt_dft_do_ddft": False,
                "sapt_dft_do_disp": True,
                "sapt_dft_do_fsapt": "FISAPT",
                "sapt_dft_functional": "HF",
                "sapt_dft_mp2_disp_alg": "FISAPT",
                "sapt_dft_use_einsums": False,
            }
        )
    elif scenario == "disp":
        options.update(
            {
                "sapt_dft_do_ddft": False,
                "sapt_dft_do_disp": True,
            }
        )
    elif scenario == "lrc":
        options.update(
            {
                "sapt_dft_functional": "wb97x",
                "dft_radial_points": 50,
                "dft_spherical_points": 110,
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


def _build_qcschema_input(mol, *, name, checkpoint_dir, scenario, protocols=None, extras=None):
    keywords = {k.lower(): v for k, v in psi4.driver.p4util.prepare_options_for_set_options().items()}
    basis = keywords.pop("basis", core.get_global_option("BASIS"))
    function_kwargs = {}
    if checkpoint_dir:
        function_kwargs["checkpoint_dir"] = str(checkpoint_dir)
    if function_kwargs:
        keywords["function_kwargs"] = function_kwargs
    molecule = _raw_identity_qcschema_molecule() if scenario == "raw_identity" else mol.to_schema(dtype=3)
    return {
        "schema_name": "qcschema_atomic_input",
        "schema_version": 2,
        "molecule": molecule,
        "specification": {
            "driver": "energy",
            "model": {"method": name, "basis": basis},
            "keywords": keywords,
            "protocols": dict(protocols or {}),
            "extras": dict(extras or {}),
        },
    }


def _install_restart_guards(
    *,
    summary,
    checkpoint_dir: str,
    guard_jk: bool,
    count_jk_builds: bool,
    capture_jk_settings: bool,
    forbidden_banners,
    molecule_signatures,
):
    manifest = _manifest_summary(checkpoint_dir)
    completed_stages = set(manifest["completed_stages"] if manifest is not None else [])
    current_stage = {"name": None}
    current_context = {"name": None}
    run_scf_counts = {"dimer": 0, "monomer_a": 0, "monomer_b": 0}
    summary.setdefault("scf_helper_call_count", 0)
    summary.setdefault("run_scf_call_count", 0)
    summary.setdefault("stage_routine_call_count", 0)
    summary.setdefault("guarded_call_count", 0)
    summary.setdefault("guarded_call_sentinel", None)
    summary.setdefault("jk_build_count", 0)
    if capture_jk_settings:
        summary.setdefault("jk_builds", [])

    banner_stage_map = {
        "SAPT(DFT): delta HF Dimer": "hf_dimer_scf",
        "SAPT(DFT): delta HF Monomer A": "hf_monomer_a_scf",
        "SAPT(DFT): delta HF Monomer B": "hf_monomer_b_scf",
        "SAPT(DFT): Dimer for Localization": "dimer_localization_scf",
        "SAPT(DFT): DFT Monomer A": "monomer_a_dft_scf",
        "SAPT(DFT): DFT Monomer B": "monomer_b_dft_scf",
    }
    timer_stage_map = {
        "SAPT(HF):elst": "hf_sapt_elst",
        "SAPT(HF):exch": "hf_sapt_exch",
        "SAPT(HF):ind": "hf_sapt_ind",
        "SAPT(DFT):delta DFT": "delta_dft",
        "SAPT(DFT):elst": "elst",
        "SAPT(DFT):exch": "exch",
        "SAPT(DFT):ind": "ind",
        "SAPT(DFT):disp": "disp",
    }

    def boom(message):
        summary["guarded_call_sentinel"] = message
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

    def _with_context(label):
        previous = current_context["name"]
        current_context["name"] = label
        return previous

    def _restore_context(previous):
        current_context["name"] = previous

    def _guard_completed_stage(stage, label):
        summary["guarded_call_count"] += 1
        if stage in completed_stages:
            boom(f"{label} replayed completed checkpoint stage {stage}")

    original_proc_scf_helper = proc.scf_helper
    original_proc_run_scf = proc.run_scf
    original_timer_on = core.timer_on
    original_timer_off = core.timer_off
    original_prepare_restored_scf = sapt_proc._saptdft_prepare_restored_scf
    original_commit_stage = sapt_proc._saptdft_commit_stage
    forbidden_banner_set = set(forbidden_banners)
    timer_context_stack = []

    def guarded_scf_helper(*args, **kwargs):
        summary["scf_helper_call_count"] += 1
        summary["guarded_call_count"] += 1
        banner = kwargs.get("banner")
        stage = banner_stage_map.get(banner)
        restore_stage = current_stage["name"]
        previous_context = _with_context(restore_stage)
        if stage is None and banner is None and restore_stage is not None:
            stage = restore_stage
        if stage is None:
            boom(f"Unclassified scf_helper banner encountered during checkpoint restart: {banner!r}")
        if banner in forbidden_banner_set or stage in completed_stages:
            boom(f"SCF helper replayed completed checkpoint stage {stage}: {banner}")
        current_stage["name"] = stage
        current_context["name"] = stage
        try:
            return original_proc_scf_helper(*args, **kwargs)
        finally:
            current_stage["name"] = restore_stage
            _restore_context(previous_context)

    def guarded_run_scf(*args, **kwargs):
        summary["run_scf_call_count"] += 1
        summary["guarded_call_count"] += 1
        molecule = kwargs.get("molecule")
        if molecule is None:
            boom("run_scf called without molecule during checkpoint restart")
        stage = _classify_run_scf_stage(molecule)
        if stage in completed_stages:
            boom(f"run_scf replayed completed checkpoint stage {stage}")
        current_stage["name"] = stage
        previous_context = _with_context(stage)
        try:
            return original_proc_run_scf(*args, **kwargs)
        finally:
            current_stage["name"] = None
            _restore_context(previous_context)

    def guarded_timer_on(label):
        timer_context_stack.append(current_context["name"])
        stage = timer_stage_map.get(label)
        if stage is not None:
            current_context["name"] = stage
        return original_timer_on(label)

    def guarded_timer_off(label):
        try:
            return original_timer_off(label)
        finally:
            if timer_context_stack:
                current_context["name"] = timer_context_stack.pop()

    def guarded_prepare_restored_scf(wfn, *args, **kwargs):
        functional_name = "unknown"
        try:
            functional_name = wfn.functional().name().lower()
        except Exception:
            pass
        previous_context = _with_context(f"restore:{functional_name}")
        try:
            return original_prepare_restored_scf(wfn, *args, **kwargs)
        finally:
            _restore_context(previous_context)

    def guarded_commit_stage(checkpoint, stage, *args, **kwargs):
        if stage == "delta_dft":
            _guard_completed_stage(stage, "delta_dft commit")
        return original_commit_stage(checkpoint, stage, *args, **kwargs)

    proc.scf_helper = guarded_scf_helper
    proc.run_scf = guarded_run_scf
    sapt_proc.scf_helper = guarded_scf_helper
    sapt_proc.run_scf = guarded_run_scf
    core.timer_on = guarded_timer_on
    core.timer_off = guarded_timer_off
    sapt_proc._saptdft_prepare_restored_scf = guarded_prepare_restored_scf
    sapt_proc._saptdft_commit_stage = guarded_commit_stage

    for attr in ["compute_energy", "guess", "diis"]:
        original = getattr(core.HF, attr)

        def guarded(self, *args, _original=original, _attr=attr, **kwargs):
            summary["guarded_call_count"] += 1
            stage = current_stage["name"]
            if stage in completed_stages:
                boom(f"core.HF.{_attr} replayed completed checkpoint stage {stage}")
            return _original(self, *args, **kwargs)

        setattr(core.HF, attr, guarded)

    from psi4.driver.procrouting.sapt import sapt_jk_terms, sapt_mp2_terms

    guarded_modules = [sapt_jk_terms]
    guarded_mp2_modules = [sapt_mp2_terms]
    try:
        from psi4.driver.procrouting.sapt import sapt_jk_terms_ein, sapt_mp2_terms_ein
    except ImportError:
        pass
    else:
        guarded_modules.append(sapt_jk_terms_ein)
        guarded_mp2_modules.append(sapt_mp2_terms_ein)

    for module in guarded_modules:
        original_electrostatics = module.electrostatics
        original_exchange = module.exchange
        original_induction = module.induction

        def guarded_electrostatics(*args, _original=original_electrostatics, **kwargs):
            summary["stage_routine_call_count"] += 1
            stage = current_context["name"]
            if stage in {"hf_sapt_elst", "elst"}:
                _guard_completed_stage(stage, "electrostatics")
            return _original(*args, **kwargs)

        def guarded_exchange(*args, _original=original_exchange, **kwargs):
            summary["stage_routine_call_count"] += 1
            stage = current_context["name"]
            if stage in {"hf_sapt_exch", "exch"}:
                _guard_completed_stage(stage, "exchange")
            return _original(*args, **kwargs)

        def guarded_induction(*args, _original=original_induction, **kwargs):
            summary["stage_routine_call_count"] += 1
            stage = current_context["name"]
            if stage in {"hf_sapt_ind", "ind"}:
                _guard_completed_stage(stage, "induction")
            return _original(*args, **kwargs)

        module.electrostatics = guarded_electrostatics
        module.exchange = guarded_exchange
        module.induction = guarded_induction

    for module in guarded_mp2_modules:
        original_fdds = module.df_fdds_dispersion
        original_mp2_fisapt = module.df_mp2_fisapt_dispersion
        original_mp2_sapt = module.df_mp2_sapt_dispersion

        def guarded_fdds(*args, _original=original_fdds, **kwargs):
            summary["stage_routine_call_count"] += 1
            stage = current_context["name"]
            if stage == "disp":
                _guard_completed_stage(stage, "df_fdds_dispersion")
            return _original(*args, **kwargs)

        def guarded_mp2_fisapt(*args, _original=original_mp2_fisapt, **kwargs):
            summary["stage_routine_call_count"] += 1
            stage = current_context["name"]
            if stage == "disp":
                _guard_completed_stage(stage, "df_mp2_fisapt_dispersion")
            return _original(*args, **kwargs)

        def guarded_mp2_sapt(*args, _original=original_mp2_sapt, **kwargs):
            summary["stage_routine_call_count"] += 1
            stage = current_context["name"]
            if stage == "disp":
                _guard_completed_stage(stage, "df_mp2_sapt_dispersion")
            return _original(*args, **kwargs)

        module.df_fdds_dispersion = guarded_fdds
        module.df_mp2_fisapt_dispersion = guarded_mp2_fisapt
        module.df_mp2_sapt_dispersion = guarded_mp2_sapt

    if guard_jk:
        def guarded_jk_build(*args, **kwargs):
            summary["guarded_call_count"] += 1
            boom("JK.build called during guarded final checkpoint restart")

        core.JK.build = guarded_jk_build
    elif count_jk_builds or capture_jk_settings:
        jk_states = {}
        original_jk_build = core.JK.build
        original_set_do_j = core.JK.set_do_J
        original_set_do_k = core.JK.set_do_K
        original_set_do_wk = core.JK.set_do_wK
        original_set_memory = core.JK.set_memory
        original_set_omega = core.JK.set_omega
        original_set_omega_alpha = core.JK.set_omega_alpha
        original_set_omega_beta = core.JK.set_omega_beta
        original_initialize = core.JK.initialize

        def _jk_state(jk):
            return jk_states.setdefault(
                id(jk),
                {
                    "built": False,
                    "context": current_context["name"],
                    "orbital_basis": None,
                    "aux_basis": None,
                    "memory": None,
                    "do_J": None,
                    "do_K": None,
                    "do_wK": None,
                    "omega": None,
                    "omega_alpha": None,
                    "omega_beta": None,
                    "initialized": False,
                },
            )

        def counted_jk_build(*args, **kwargs):
            jk = original_jk_build(*args, **kwargs)
            summary["jk_build_count"] += 1
            state = _jk_state(jk)
            state["built"] = True
            orbital_basis = kwargs.get("orbital") if "orbital" in kwargs else (args[0] if args else None)
            aux_basis = kwargs.get("aux") if "aux" in kwargs else (args[1] if len(args) > 1 else None)
            state["context"] = current_context["name"]
            state["orbital_basis"] = orbital_basis.name() if hasattr(orbital_basis, "name") else None
            state["aux_basis"] = aux_basis.name() if hasattr(aux_basis, "name") else None
            if "memory" in kwargs:
                state["memory"] = kwargs["memory"]
            if "do_wK" in kwargs:
                state["do_wK"] = kwargs["do_wK"]
            return jk

        def wrapped_set_do_j(self, value):
            _jk_state(self)["do_J"] = bool(value)
            return original_set_do_j(self, value)

        def wrapped_set_do_k(self, value):
            _jk_state(self)["do_K"] = bool(value)
            return original_set_do_k(self, value)

        def wrapped_set_do_wk(self, value):
            _jk_state(self)["do_wK"] = bool(value)
            return original_set_do_wk(self, value)

        def wrapped_set_memory(self, value):
            _jk_state(self)["memory"] = int(value)
            return original_set_memory(self, value)

        def wrapped_set_omega(self, value):
            _jk_state(self)["omega"] = float(value)
            return original_set_omega(self, value)

        def wrapped_set_omega_alpha(self, value):
            _jk_state(self)["omega_alpha"] = float(value)
            return original_set_omega_alpha(self, value)

        def wrapped_set_omega_beta(self, value):
            _jk_state(self)["omega_beta"] = float(value)
            return original_set_omega_beta(self, value)

        def wrapped_initialize(self):
            _jk_state(self)["initialized"] = True
            return original_initialize(self)

        core.JK.build = counted_jk_build
        core.JK.set_do_J = wrapped_set_do_j
        core.JK.set_do_K = wrapped_set_do_k
        core.JK.set_do_wK = wrapped_set_do_wk
        core.JK.set_memory = wrapped_set_memory
        core.JK.set_omega = wrapped_set_omega
        core.JK.set_omega_alpha = wrapped_set_omega_alpha
        core.JK.set_omega_beta = wrapped_set_omega_beta
        core.JK.initialize = wrapped_initialize
        summary["_jk_states"] = jk_states


def _install_fsapt_guards(forbid_stages):
    if not forbid_stages:
        return

    forbid_stages = set(forbid_stages)

    from psi4.driver.procrouting.sapt import saptdft_fisapt, sapt_jk_terms_ein

    def boom(label):
        raise AssertionError(f"F-SAPT routine replayed completed checkpoint stage {label}")

    if "setup" in forbid_stages:
        original_setup = saptdft_fisapt.setup_fisapt_object

        def guarded_setup(*args, **kwargs):
            if kwargs.get("do_flocalize", False):
                boom("fsapt_setup")
            return original_setup(*args, **kwargs)

        saptdft_fisapt.setup_fisapt_object = guarded_setup
        sapt_jk_terms_ein.localization = lambda *args, **kwargs: boom("fsapt_setup")
        sapt_jk_terms_ein.partition = lambda *args, **kwargs: boom("fsapt_setup")
        sapt_jk_terms_ein.flocalization = lambda *args, **kwargs: boom("fsapt_setup")

    if "elst" in forbid_stages:
        sapt_jk_terms_ein.felst = lambda *args, **kwargs: boom("fsapt_elst")
        core.FISAPT.felst = lambda self, *args, **kwargs: boom("fsapt_elst")
    if "exch" in forbid_stages:
        sapt_jk_terms_ein.fexch = lambda *args, **kwargs: boom("fsapt_exch")
        core.FISAPT.fexch = lambda self, *args, **kwargs: boom("fsapt_exch")
    if "ind" in forbid_stages:
        sapt_jk_terms_ein.find = lambda *args, **kwargs: boom("fsapt_ind")
        core.FISAPT.find = lambda self, *args, **kwargs: boom("fsapt_ind")
    if "disp" in forbid_stages:
        sapt_jk_terms_ein.fdisp0 = lambda *args, **kwargs: boom("fsapt_disp")
        core.FISAPT.fdisp = lambda self, *args, **kwargs: boom("fsapt_disp")


def _capture_fsapt_variables(summary):
    labels = [
        "FSAPT_QA",
        "FSAPT_QB",
        "FSAPT_ELST_AB",
        "FSAPT_EXCH_AB",
        "FSAPT_INDAB_AB",
        "FSAPT_INDBA_AB",
        "FSAPT_DISP_AB",
        "FSAPT_EMPIRICAL_DISP",
        "FSAPT_AB_SIZE",
    ]
    summary["fsapt_variables"] = {
        label: _serialize_value(_safe_variable(label)) for label in labels if _safe_variable(label) is not None
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "stop", "restart", "restart_with_guards", "qcschema", "qcschema_restart_with_guards"])
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--stop-after")
    parser.add_argument("--name", default="sapt(dft)")
    parser.add_argument(
        "--scenario",
        default="default",
        choices=["default", "disp", "localization", "fsapt_einsums", "fsapt_fisapt", "lrc", "raw_identity"],
    )
    parser.add_argument("--guard-jk", action="store_true")
    parser.add_argument("--count-jk-builds", action="store_true")
    parser.add_argument("--capture-jk-settings", action="store_true")
    parser.add_argument("--capture-fsapt", action="store_true")
    parser.add_argument("--forbid-banner", action="append", default=[])
    parser.add_argument("--forbid-fsapt-stage", action="append", default=[])
    parser.add_argument("--qcschema-protocols-json")
    parser.add_argument("--qcschema-extras-json")
    args = parser.parse_args()

    mol, molecule_signatures = _configure(args.scenario, args.name)
    summary = {
        "mode": args.mode,
        "checkpoint_dir": args.checkpoint_dir,
        "stop_after": args.stop_after,
        "name": args.name,
        "scenario": args.scenario,
    }

    if args.mode in {"restart_with_guards", "qcschema_restart_with_guards"}:
        _install_restart_guards(
            summary=summary,
            checkpoint_dir=args.checkpoint_dir,
            guard_jk=args.guard_jk,
            count_jk_builds=args.count_jk_builds,
            capture_jk_settings=args.capture_jk_settings,
            forbidden_banners=args.forbid_banner,
            molecule_signatures=molecule_signatures,
        )
    if args.forbid_fsapt_stage:
        _install_fsapt_guards(args.forbid_fsapt_stage)

    kwargs = {"molecule": mol}
    if args.checkpoint_dir:
        kwargs["checkpoint_dir"] = args.checkpoint_dir
    if args.stop_after:
        kwargs["checkpoint_stop_after"] = args.stop_after

    try:
        if args.mode in {"qcschema", "qcschema_restart_with_guards"}:
            protocols = json.loads(args.qcschema_protocols_json) if args.qcschema_protocols_json else None
            extras = json.loads(args.qcschema_extras_json) if args.qcschema_extras_json else None
            result = psi4.schema_wrapper.run_qcschema(
                _build_qcschema_input(
                    mol,
                    name=args.name,
                    checkpoint_dir=args.checkpoint_dir,
                    scenario=args.scenario,
                    protocols=protocols,
                    extras=extras,
                )
            )
            if getattr(result, "success", False):
                summary["status"] = "ok"
                summary["qcschema_success"] = True
                summary["sapt_total_energy"] = result.extras["qcvars"].get("SAPT TOTAL ENERGY")
                summary["current_energy"] = result.extras["qcvars"].get("CURRENT ENERGY")
                summary["energy"] = summary["sapt_total_energy"]
            else:
                summary["status"] = "error"
                summary["error_type"] = type(result).__name__
                summary["error"] = result.error.error_message
        else:
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

    if "_jk_states" in summary:
        summary["jk_builds"] = list(summary.pop("_jk_states").values())

    summary["current_energy"] = summary.get("current_energy", _safe_variable("CURRENT ENERGY"))
    summary["sapt_total_energy"] = summary.get("sapt_total_energy", _safe_variable("SAPT TOTAL ENERGY"))
    summary["saptdft_total_energy"] = summary.get("saptdft_total_energy", _safe_variable("SAPT(DFT) TOTAL ENERGY"))
    summary["elst10_r"] = summary.get("elst10_r", _safe_variable("Elst10,r"))
    summary["energy"] = summary.get("energy", summary["sapt_total_energy"])

    if args.capture_fsapt:
        _capture_fsapt_variables(summary)

    manifest = _manifest_summary(args.checkpoint_dir)
    if manifest is not None:
        summary["manifest_path"] = manifest["path"]
        summary["completed_stages"] = manifest["completed_stages"]
        summary["manifest"] = manifest["manifest"]
        summary["qcvars"] = {
            str(key): _serialize_value(value)
            for key, value in manifest["manifest"].get("scalars", {}).items()
            if isinstance(_serialize_value(value), (bool, int, float))
        }
    else:
        summary["qcvars"] = _collect_scalar_qcvars()

    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if summary.get("status") in {"ok", "stopped"} else 1)


if __name__ == "__main__":
    main()
