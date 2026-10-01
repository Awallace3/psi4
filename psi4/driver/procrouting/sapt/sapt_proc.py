#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2026 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import numpy as np

from psi4 import core

from ... import p4util
from ...constants import constants
from ...p4util.exceptions import ValidationError
from ..empirical_disp import edisp_interaction_energy
from .. import proc_util
from ..proc import (
    _set_external_potentials_to_wavefunction,
    run_scf,
    scf_helper,
    validate_external_potential,
)
from . import (
    sapt_jk_terms,
    sapt_mp2_terms,
    sapt_sf_terms,
    saptdft_fisapt,
)
from .saptdft_checkpoint import (
    CheckpointSession,
    functional_value,
    prepare_restored_scf,
    wfn_jk,
)
from .sapt_util import (
    print_sapt_dft_summary,
    print_sapt_hf_induction_summary,
    print_sapt_hf_summary,
    print_sapt_var,
)
import qcelemental as qcel
from ...p4util.exceptions import ConvergenceError

try:
    import einsums as ein
    from . import (
        sapt_jk_terms_ein,
        sapt_mp2_terms_ein,
    )

    einsums_available = True
except ImportError:
    einsums_available = False

# Only export the run_ scripts
__all__ = ["run_sapt_dft", "sapt_dft", "run_sf_sapt"]


# SAPT(DFT) computational stages, in the order the driver reaches them. The
# checkpoint session uses these to answer "what is still left to do?", which is
# what lets the JK object and SAPT cache be skipped entirely on a late restart.
_FSAPT_STAGE_ORDER = ("fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind")
_FSAPT_CACHE_STAGES = _FSAPT_STAGE_ORDER + ("fsapt_disp",)

# Checkpoint scalar prefixes for values that must survive a restart but are not
# SAPT(DFT) results. Keys starting with "_" never reach ``data`` or QCVariables.
_HF_DATA_PREFIX = "_hf_data::"
_HF_IND_PREFIX = "_hf_ind::"
_GRAC_PREFIX = "_grac::"


def _sapt_stage_order(*, do_disp, do_fsapt, do_dft=True, induction_type="CPKS"):
    """Computational stages of :func:`sapt_dft` for the requested SAPT(DFT) flavour."""
    stages = ["elst", "exch"]
    if _sapt_dft_runs_induction(do_dft=do_dft, induction_type=induction_type):
        stages.append("ind")
    if do_disp:
        stages.append("disp")
    if do_fsapt:
        stages.extend(_fsapt_stage_order(do_disp=do_disp))
    return stages


def _sapt_dft_runs_induction(*, do_dft, induction_type):
    """Whether :func:`sapt_dft` solves the coupled induction equations itself."""
    induction_type = induction_type.upper()
    return induction_type == "CPKS" or (induction_type == "CPHF" and not do_dft)


def _fsapt_stage_order(*, do_disp):
    """F-SAPT sub-stages of :func:`sapt_dft` for the requested SAPT(DFT) flavour."""
    stages = list(_FSAPT_STAGE_ORDER)
    if do_disp:
        stages.append("fsapt_disp")
    stages.append("fsapt_final")
    return stages


# QCVariables that :func:`sapt_util.print_sapt_dft_summary` and its SAPT(HF)
# counterpart publish directly to Psi4 rather than through the SAPT ``data`` dict.
_SAPT_SUMMARY_QCVARIABLES = (
    "SAPT ELST ENERGY",
    "SAPT EXCH ENERGY",
    "SAPT IND ENERGY",
    "SAPT DISP ENERGY",
    "SAPT0 TOTAL ENERGY",
    "SAPT(DFT) TOTAL ENERGY",
    "SAPT TOTAL ENERGY",
    "CURRENT ENERGY",
)


def _sapt_summary_qcvariables():
    """Snapshot the summary QCVariables so a restart can republish them."""
    return {
        label: core.variable(label) for label in _SAPT_SUMMARY_QCVARIABLES if core.has_variable(label)
    }


def _replay_final_checkpoint(ckpt, data, molecule):
    """Return the stored result of a run whose ``final`` stage is complete.

    The stored wavefunction carries the matrix QCVariables (F-SAPT partitions,
    pairwise dispersion) that the manifest cannot hold; the manifest carries the
    scalars the summary published straight to Psi4.
    """
    dimer_wfn = ckpt.restore_final(molecule, core.get_global_option("BASIS"))
    for k, v in dimer_wfn.variables().items():
        core.set_variable(k, v)
    for k, v in data.items():
        core.set_variable(k, v)
        dimer_wfn.set_variable(k, v)
    current_energy = data.get("CURRENT ENERGY", data.get("SAPT TOTAL ENERGY", dimer_wfn.energy()))
    core.set_variable("CURRENT ENERGY", current_energy)
    dimer_wfn.set_variable("CURRENT ENERGY", current_energy)
    dimer_wfn.set_energy(current_energy)
    return dimer_wfn


def _cache_fisapt_localization_aliases(cache):
    """FISAPT names the localized orbitals differently than the einsums path does."""
    aliases = {
        "Locc0A": "Locc_A",
        "Locc0B": "Locc_B",
        "Uocc0A": "Uocc_A",
        "Uocc0B": "Uocc_B",
    }
    for source_key, target_key in aliases.items():
        if source_key in cache and target_key not in cache:
            cache[target_key] = cache[source_key]
    return cache


def _rebuild_einsums_fsapt_elst_cache(cache, dimer_wfn):
    """Rebuild the DFHelper that a restored F-SAPT electrostatics cache does not carry."""
    if "dfh" in cache or "Locc_A" not in cache or "Locc_B" not in cache:
        return cache
    aux_basis = dimer_wfn.get_basisset("DF_BASIS_SCF")
    dfh = core.DFHelper(dimer_wfn.basisset(), aux_basis)
    dfh.set_memory(core.get_memory() // 8)
    dfh.set_method("DIRECT_iaQ")
    dfh.set_nthreads(core.get_num_threads())
    dfh.initialize()
    dfh.add_space("a", core.Matrix.from_array(cache["Locc_A"].np))
    dfh.add_space("b", core.Matrix.from_array(cache["Locc_B"].np))
    dfh.add_transformation("Aaa", "a", "a")
    dfh.add_transformation("Abb", "b", "b")
    dfh.transform()
    dfh.clear_spaces()
    cache["dfh"] = dfh
    return cache


def _absorb_fisapt_matrices(cache, FISAPT_obj):
    """Copy the FISAPT object's matrices into the SAPT cache so a stage can store them."""
    for key, value in FISAPT_obj.matrices().items():
        cache[key] = value
    return cache


def run_sapt_dft(name: str, **kwargs) -> core.Wavefunction:
    """Run SAPT(DFT) while restoring all driver-managed options and timers."""
    optstash = p4util.OptionsState(
        ["DF_INTS_IO"],
        ["SCF_TYPE"],
        ["SCF", "REFERENCE"],
        ["SCF", "DFT_GRAC_SHIFT"],
        ["SCF", "SAVE_JK"],
        ["SAPT", "SAPT_DFT_DO_DISP"],
        ["SAPT", "SAPT_DFT_DO_DDFT"],
        ["SAPT", "SAPT_DFT_D3_IE"],
        ["SAPT", "SAPT_DFT_D4_IE"],
        ["SAPT", "SAPT_DFT_D_TYPE"],
    )
    core.timer_on("SAPT(DFT) Energy")
    try:
        return _run_sapt_dft(name, **kwargs)
    finally:
        core.timer_off("SAPT(DFT) Energy")
        optstash.restore()


def _run_sapt_dft(name: str, **kwargs) -> core.Wavefunction:
    """Run the SAPT(DFT) interaction energy calculation.

    Top-level driver function for SAPT(DFT). Sets up monomer SCF
    calculations with optional GRAC shifts, builds the SAPT JK cache,
    and calls :func:`sapt_dft` to compute the interaction energy
    components.

    Parameters
    ----------
    name : str
        Name of the SAPT(DFT) method (e.g., ``'sapt(dft)'``).
    **kwargs
        Additional keyword arguments. Recognized keys include ``ref_wfn``
        (reference wavefunction), ``molecule`` (molecular system).

    Returns
    -------
    core.Wavefunction
        The dimer wavefunction with SAPT(DFT) results stored as variables.
    """
    core.prepare_options_for_module("SAPT")
    induction_type = core.get_option("SAPT", "SAPT_DFT_INDUCTION_TYPE").upper()
    do_delta_hf = core.get_option("SAPT", "SAPT_DFT_DO_DHF")
    fsapt_type = core.get_option("SAPT", "SAPT_DFT_DO_FSAPT").upper()
    do_fsapt = fsapt_type != "NONE"
    if induction_type == "NONE" and do_fsapt:
        raise ValidationError("F-SAPT requires induction; SAPT_DFT_INDUCTION_TYPE=NONE is unavailable.")
    if induction_type == "CPHF" and fsapt_type == "SAPTDFT":
        raise ValidationError(
            "SAPTDFT F-SAPT requires SAPT(DFT) fragment induction; "
            "use SAPT_DFT_DO_FSAPT=FISAPT with SAPT_DFT_INDUCTION_TYPE=CPHF."
        )

    use_einsums = core.get_option("SAPT", "SAPT_DFT_USE_EINSUMS")

    # Build SAPT cache
    if einsums_available and use_einsums:
        jk_terms = sapt_jk_terms_ein
        ein.initialize()
    else:
        # If einsums is not available, need to conditionally stop einsums
        # without adding einsums_available and use_einsums to every check.
        use_einsums = False
        jk_terms = sapt_jk_terms

    # Alter default algorithm
    if not core.has_global_option_changed("SCF_TYPE"):
        core.set_global_option("SCF_TYPE", "DF")

    # Get the molecule of interest
    ref_wfn = kwargs.get("ref_wfn", None)
    if ref_wfn is None:
        sapt_dimer_initial = kwargs.pop("molecule", core.get_active_molecule())
    else:
        core.print_out(
            'Warning! SAPT argument "ref_wfn" is only able to use molecule information.'
        )
        sapt_dimer_initial = ref_wfn.molecule()

    sapt_dimer, monomerA, monomerB = proc_util.prepare_sapt_molecule(
        sapt_dimer_initial, "dimer"
    )

    if getattr(sapt_dimer_initial, "_initial_cartesian", None) is not None:
        sapt_dimer._initial_cartesian = sapt_dimer_initial._initial_cartesian
        monomerA._initial_cartesian = core.Matrix.from_array(
            sapt_dimer._initial_cartesian.np.copy()
        )
        monomerB._initial_cartesian = core.Matrix.from_array(
            sapt_dimer._initial_cartesian.np.copy()
        )

    data = {}
    # Grab overall settings
    do_mon_grac_shift_A = False
    do_mon_grac_shift_B = False
    mon_a_shift = core.get_option("SAPT", "SAPT_DFT_GRAC_SHIFT_A")
    mon_b_shift = core.get_option("SAPT", "SAPT_DFT_GRAC_SHIFT_B")
    grac_compute = core.get_option("SAPT", "SAPT_DFT_GRAC_COMPUTE")
    shift_only = core.get_option("SAPT", "SAPT_DFT_GRAC_SHIFT_ONLY")
    grac_use_ext_pot = core.get_option("SAPT", "SAPT_DFT_GRAC_USE_EXT_POT")

    if (
        not core.has_option_changed("SAPT", "SAPT_DFT_GRAC_SHIFT_A")
        and grac_compute.upper() != "NONE"
    ):
        do_mon_grac_shift_A = True
    if (
        not core.has_option_changed("SAPT", "SAPT_DFT_GRAC_SHIFT_B")
        and grac_compute.upper() != "NONE"
    ):
        do_mon_grac_shift_B = True

    sapt_dft_functional = core.get_option("SAPT", "SAPT_DFT_FUNCTIONAL")
    e_disp_param_name = None
    supported_functionals_edisp = ["hf", "pbe0", "b3lyp"]

    # SAPT_DFT_D4_IE and SAPT_DFT_D3_IE control whether to run -D3/-D4.
    # Explicit user settings are honored for plain SAPT(DFT); method aliases
    # below temporarily override them and the public wrapper restores them.
    # Need to identify which flavor of -D we are using
    if "-D4" in name.upper():
        d4_type = core.get_option("SAPT", "SAPT_DFT_D_TYPE").lower()
        if "-D4(S)" in name.upper():
            core.print_out(r"SAPT(DFT)-D4(S): -D4(S) for dispersion")
            e_disp_param_name = (
                f"sapt({sapt_dft_functional.lower()})(s)"
                if sapt_dft_functional.lower() != "hf"
                else "hf"
            )
            if sapt_dft_functional.lower() not in supported_functionals_edisp:
                raise ValueError(
                    "SAPT(DFT)-D4 with D4(S) parameters is currently only available for PBE0 and B3LYP."
                    f" Functional {sapt_dft_functional.lower()} does not have D4(S) parameters defined."
                )
            core.set_global_option("SAPT_DFT_DO_DISP", 0)
            core.set_global_option("SAPT_DFT_D4_IE", 1)
            core.set_global_option("SAPT_DFT_DO_DDFT", 0)
            core.set_global_option("SAPT_DFT_D_TYPE", "supermolecular")
        elif "-D4(I)" in name.upper():
            core.print_out(r"SAPT(DFT)-D4(I): -D4(I) for dispersion")
            # D4(I) uses intermolecular atom-pair summation together with the
            # SAPT functional's dedicated (I) damping-parameter record.
            e_disp_param_name = (
                f"sapt({sapt_dft_functional.lower()})(i)"
                if sapt_dft_functional.lower() != "hf"
                else "hf"
            )
            if sapt_dft_functional.lower() not in supported_functionals_edisp:
                raise ValueError(
                    "SAPT(DFT)-D4 with D4(I) parameters is currently only available for PBE0 and B3LYP."
                    f" Functional {sapt_dft_functional.lower()} does not have D4(I) parameters defined."
                )
            core.set_global_option("SAPT_DFT_DO_DISP", 0)
            core.set_global_option("SAPT_DFT_D4_IE", 1)
            core.set_global_option("SAPT_DFT_DO_DDFT", 0)
            core.set_global_option("SAPT_DFT_D_TYPE", "intermolecular")
        elif "DFT-D4" in name.upper():
            core.print_out(r"DFT-D4(SAPT): $\Delta$-DFT+D4 for dispersion")
            core.set_global_option("SAPT_DFT_DO_DISP", 0)
            core.set_global_option("SAPT_DFT_D4_IE", 1)
            core.set_global_option("SAPT_DFT_DO_DDFT", 1)
            core.set_global_option("SAPT_DFT_D_TYPE", "gd4_supermolecular")
            e_disp_param_name = (
                sapt_dft_functional.lower()
                if sapt_dft_functional.lower() != "hf"
                else "hf"
            )
        else:
            raise ValueError(
                "SAPT(DFT)-D4 must be specified as 'SAPT(DFT)-D4(S)' or "
                "'SAPT(DFT)-D4(I)' through setting SAPT_DFT_D_TYPE to "
                "'supermolecular' or 'intermolecular'."
            )
    elif "-D3" in name.upper():
        d4_type = core.get_option("SAPT", "SAPT_DFT_D_TYPE").lower()
        if "-D3(S)" in name.upper():
            core.print_out(r"SAPT(DFT)-D3(S): -D3(S) for dispersion")
            e_disp_param_name = (
                f"sapt({sapt_dft_functional.lower()})(s)"
                if sapt_dft_functional.lower() != "hf"
                else "hf"
            )
            if sapt_dft_functional.lower() not in supported_functionals_edisp:
                raise ValueError(
                    "SAPT(DFT)-D3 with D3(S) parameters is currently only available for PBE0 and B3LYP."
                    f" Functional {sapt_dft_functional.lower()} does not have D3(S) parameters defined."
                )
            core.set_global_option("SAPT_DFT_DO_DISP", 0)
            core.set_global_option("SAPT_DFT_D3_IE", 1)
            core.set_global_option("SAPT_DFT_DO_DDFT", 0)
            core.set_global_option("SAPT_DFT_D_TYPE", "supermolecular")
        elif "-D3(I)" in name.upper():
            core.print_out(r"SAPT(DFT)-D3(I): -D3(I) for dispersion")
            e_disp_param_name = (
                f"sapt({sapt_dft_functional.lower()})(i)"
                if sapt_dft_functional.lower() != "hf"
                else "hf"
            )
            if sapt_dft_functional.lower() not in supported_functionals_edisp:
                raise ValueError(
                    "SAPT(DFT)-D3 with D3(I) parameters is currently only available for PBE0 and B3LYP."
                    f" Functional {sapt_dft_functional.lower()} does not have D3(I) parameters defined."
                )
            core.set_global_option("SAPT_DFT_DO_DISP", 0)
            core.set_global_option("SAPT_DFT_D3_IE", 1)
            core.set_global_option("SAPT_DFT_DO_DDFT", 0)
            core.set_global_option("SAPT_DFT_D_TYPE", "intermolecular")
        elif "DFT-D3" in name.upper():
            core.print_out(r"DFT-D3(SAPT): $\Delta$-DFT+D3 for dispersion")
            core.set_global_option("SAPT_DFT_DO_DISP", 0)
            core.set_global_option("SAPT_DFT_D3_IE", 1)
            core.set_global_option("SAPT_DFT_DO_DDFT", 1)
            core.set_global_option("SAPT_DFT_D_TYPE", "gd3_supermolecular")
            e_disp_param_name = (
                sapt_dft_functional.lower()
                if sapt_dft_functional.lower() != "hf"
                else "hf"
            )
        else:
            raise ValueError(
                "SAPT(DFT)-D3 must be specified as 'SAPT(DFT)-D3(S)' or "
                "'SAPT(DFT)-D3(I)' through setting SAPT_DFT_D_TYPE to "
                "'supermolecular' or 'intermolecular'."
            )
        # # Re-prepare options after local option changes
        # core.prepare_options_for_module("SAPT")

    do_delta_dft = core.get_option("SAPT", "SAPT_DFT_DO_DDFT")
    do_disp = core.get_option("SAPT", "SAPT_DFT_DO_DISP")
    sapt_dft_D4_IE = core.get_option("SAPT", "SAPT_DFT_D4_IE")
    sapt_dft_D3_IE = core.get_option("SAPT", "SAPT_DFT_D3_IE")
    do_dft = sapt_dft_functional != "HF"
    if shift_only and grac_compute == "NONE":
        raise ValidationError("SAPT_DFT_GRAC_SHIFT_ONLY=true contradicts SAPT_DFT_GRAC_COMPUTE=NONE.")
    if shift_only and not do_dft:
        raise ValidationError("SAPT_DFT_GRAC_SHIFT_ONLY requires a non-HF SAPT_DFT_FUNCTIONAL.")
    if not do_dft:
        do_mon_grac_shift_A = do_mon_grac_shift_B = False

    if do_fsapt and (sapt_dft_D4_IE or sapt_dft_D3_IE):
        dispersion_type = core.get_option("SAPT", "SAPT_DFT_D_TYPE").lower()
        if do_delta_dft or dispersion_type != "intermolecular":
            core.print_out(
                "\n    Warning: the empirical F-SAPT dispersion breakdown uses the "
                "intermolecular atom-pair contributions from the dimer calculation. "
                "For supermolecular and delta-DFT D3/D4 methods, this qualitative "
                "breakdown does not in general sum to the scalar dispersion or total "
                "interaction energy; the reported scalar energies remain authoritative.\n\n"
            )

    # CPHF needs the HF segment for the SAPT0 induction terms, even without delta HF.
    run_hf_segment = do_delta_hf or (induction_type == "CPHF" and do_dft)
    hf_segment_label = "delta HF" if do_delta_hf else "SAPT0 induction"

    # Because SAPT(DFT) FDDS Dispersion doesn't have FSAPT support currently,
    # catch this case when FISAPT is requested with SAPT_DFT_DO_DISP false
    if do_fsapt and do_disp and sapt_dft_functional != "HF":
        raise ValidationError(
            "FSAPT(DFT) currently only supported with empirical dispersion methods "
            "(like SAPT(DFT)-D4(I)) or with SAPT_DFT_FUNCTIONAL=HF."
        )

    raw_external_potentials = kwargs.pop("external_potentials", None)
    do_ext_potential = raw_external_potentials is not None
    external_potentials = (
        validate_external_potential(raw_external_potentials)
        if do_ext_potential
        else {}
    )
    if do_ext_potential:
        kwargs["external_potentials"] = {}
    if grac_use_ext_pot and external_potentials.get("C"):
        core.print_out(
            "\n   Warning: SAPT_DFT_GRAC_USE_EXT_POT excludes C from GRAC shifts. "
            "Place charges belonging in the shift in A/B; the consuming monomer DFT still includes C.\n"
        )


    if (
        do_dft
        and (
            (not core.has_option_changed("SAPT", "SAPT_DFT_GRAC_SHIFT_A"))
            or (not core.has_option_changed("SAPT", "SAPT_DFT_GRAC_SHIFT_B"))
        )
        and grac_compute == "NONE"
    ):
        raise ValidationError(
            'SAPT(DFT): User must set both "SAPT_DFT_GRAC_SHIFT_A" and "_B".  Or, to automatically compute the GRAC shift, set SAPT_DFT_GRAC_COMPUTE to "ITERATIVE" or "SINGLE".'
        )

    if core.get_option("SCF", "REFERENCE") != "RHF":
        raise ValidationError(
            "SAPT(DFT) currently only supports restricted references."
        )


    if do_mon_grac_shift_A or do_mon_grac_shift_B:
        grac_dimer = sapt_dimer
        if grac_use_ext_pot and getattr(sapt_dimer, "_initial_cartesian", None) is not None:
            # Embedding coordinates are in the original input frame, just as
            # in scf_helper. Do not derive a shift in a reoriented QM frame.
            grac_dimer = sapt_dimer.clone()
            grac_dimer.set_geometry(sapt_dimer._initial_cartesian)
            grac_dimer.fix_com(True)
            grac_dimer.fix_orientation(True)
            grac_dimer.update_geometry()
        monomerA_mon_only_bf = grac_dimer.extract_subsets(1)
        monomerB_mon_only_bf = grac_dimer.extract_subsets(2)

    # Print out the title and some information
    core.print_out("\n")
    core.print_out(
        "         ---------------------------------------------------------\n"
    )
    core.print_out("         " + "SAPT(DFT) Procedure".center(58) + "\n")
    core.print_out("\n")
    core.print_out(
        "         " + "by Daniel G. A. Smith, Yi Xie, and Austin M. Wallace".center(58) + "\n"
    )
    core.print_out(
        "         ---------------------------------------------------------\n"
    )
    core.print_out("\n")

    core.print_out(
        "Warning! The default value of SAPT_DFT_EXCH_DISP_SCALE_SCHEME has changed from DISP to FIXED. Please be careful comparing results with earlier versions. \n\n"
    )

    core.print_out("  ==> Algorithm <==\n\n")
    core.print_out("   SAPT DFT Functional     %12s\n" % str(sapt_dft_functional))
    # fmt: off
    core.print_out("   Delta HF                %12s\n" % ("True" if do_delta_hf else "False"))
    core.print_out("   Induction Type          %12s\n" % induction_type)
    core.print_out("   JK Algorithm            %12s\n" % core.get_global_option("SCF_TYPE"))
    # fmt: on
    core.print_out("\n")
    core.print_out("   Required computations:\n")
    if run_hf_segment and not shift_only:
        core.print_out("     HF   (Dimer)\n")
        core.print_out("     HF   (Monomer A)\n")
        core.print_out("     HF   (Monomer B)\n")
    if do_dft and not shift_only:
        core.print_out("     DFT  (Monomer A)\n")
        core.print_out("     DFT  (Monomer B)\n")
    if do_mon_grac_shift_A:
        core.print_out("     GRAC (Monomer A)\n")
    if do_mon_grac_shift_B:
        core.print_out("     GRAC (Monomer B)\n")
    if do_delta_dft and not shift_only:
        core.print_out("     Delta DFT Correction:\n")
        core.print_out("       DFT (Dimer)\n")
        core.print_out("       DFT (Monomer A: No Asymptotic Correction)\n")
        core.print_out("       DFT (Monomer B: No Asymptotic Correction)\n")

    core.print_out("\n")

    identity_atomic_input = kwargs.pop(p4util.SAPTDFT_IDENTITY_ATOMIC_INPUT_KEY, None)
    checkpoint_dir, checkpoint_stop_after = CheckpointSession.controls(kwargs)
    if checkpoint_dir:
        if do_ext_potential:
            raise ValidationError(
                "SAPT(DFT) checkpointing is not supported with external_potentials; "
                "run embedded SAPT(DFT) jobs without a checkpoint directory."
            )
        if induction_type == "CPHF" and do_dft and fsapt_type == "FISAPT":
            raise ValidationError(
                "SAPT(DFT) checkpointing is not supported with SAPT_DFT_INDUCTION_TYPE=CPHF "
                "and SAPT_DFT_DO_FSAPT=FISAPT; the HF-backed fragment induction cannot be restored."
            )

    ckpt = CheckpointSession.start(
        name,
        sapt_dimer_initial,
        kwargs,
        directory=checkpoint_dir,
        stop_after=checkpoint_stop_after,
        atomic_input=identity_atomic_input,
        data=data,
    )
    with ckpt:
        restored_scalars = ckpt.restored_scalars()
        data.update(restored_scalars)
        if "Delta HF Correction" in restored_scalars:
            core.set_variable("SAPT(DFT) Delta HF", restored_scalars["Delta HF Correction"])
        if "Delta DFT Correction" in restored_scalars:
            core.set_variable("SAPT(DFT) Delta DFT", restored_scalars["Delta DFT Correction"])
        if ckpt.done("final"):
            # Everything was computed by an earlier run; replay the stored result.
            return _replay_final_checkpoint(ckpt, data, sapt_dimer)

        core.print_out("   Beginning setup computations\n")

        if do_mon_grac_shift_A:
            core.print_out("     GRAC (Monomer A)\n")
            mon_a_shift = compute_GRAC_shift(
                monomerA_mon_only_bf,
                grac_compute,
                "A",
                results=data,
                external_potentials=external_potentials.get("A") if grac_use_ext_pot else None,
                checkpoint=ckpt,
            )
        if do_mon_grac_shift_B:
            core.print_out("     GRAC (Monomer B)\n")
            mon_b_shift = compute_GRAC_shift(
                monomerB_mon_only_bf,
                grac_compute,
                "B",
                results=data,
                external_potentials=external_potentials.get("B") if grac_use_ext_pot else None,
                checkpoint=ckpt,
            )

        core.set_variable("SAPT DFT GRAC SHIFT A", mon_a_shift)  # P::e SAPT
        core.set_variable("SAPT DFT GRAC SHIFT B", mon_b_shift)  # P::e SAPT
        data["SAPT DFT GRAC SHIFT A"] = mon_a_shift
        data["SAPT DFT GRAC SHIFT B"] = mon_b_shift
        core.print_out("\n  ==> SAPT(DFT) GRAC Shifts <==\n\n")
        core.print_out("   Monomer   E(monomer) [Eh]   E(ionized) [Eh]     HOMO [Eh]     IP [Eh]   GRAC shift [Eh]\n")
        for label, shift, computed in (("A", mon_a_shift, do_mon_grac_shift_A),
                                        ("B", mon_b_shift, do_mon_grac_shift_B)):
            if computed:
                values = [data[f"SAPT DFT GRAC {quantity} {label}"] for quantity in
                          ("MONOMER ENERGY", "IONIZED MONOMER ENERGY", "HOMO", "IP")]
                core.print_out(f"         {label} {values[0]:18.8f} {values[1]:18.8f} {values[2]:13.8f} {values[3]:11.8f} {shift:17.8f}\n")
            else:
                core.print_out(f"         {label} {'':63s} {shift:17.8f}\n")
                if do_dft:
                    core.print_out(f"   Monomer {label} GRAC shift supplied by the user (not computed).\n")
                else:
                    core.print_out(f"   Monomer {label} GRAC shift not applicable for HF.\n")
        core.print_out("   Monomer A GRAC Shift    %12.6f\n" % mon_a_shift)
        core.print_out("   Monomer B GRAC Shift    %12.6f\n" % mon_b_shift)
        data["SAPT DFT GRAC SHIFT ONLY"] = float(shift_only)  # P::e SAPT
        core.set_variable("SAPT DFT GRAC SHIFT ONLY", float(shift_only))  # P::e SAPT
        if shift_only:
            wfn = core.Wavefunction.build(sapt_dimer, core.get_global_option("BASIS"))
            for key, value in data.items():
                wfn.set_variable(key, value)
            core.set_variable("CURRENT ENERGY", 0.0)
            wfn.set_variable("CURRENT ENERGY", 0.0)
            core.print_out("\n   SAPT(DFT) stopped early by request: GRAC shifts only; no interaction energy computed.\n")
            ckpt.commit_final(wfn, scalars={**data, "CURRENT ENERGY": 0.0})
            return wfn
        core.print_out("\n")
        # Save integrals
        # We want to try to re-use itegrals for the dimer and monomer SCF's. If we
        # are using Disk based DF (DISK_DF) then we can use the DF_INTS_IO option.
        # MemDF does not know about this option but setting it will be harmless
        # there.
        core.set_global_option("DF_INTS_IO", "SAVE")

        # Compute dimer wavefunction
        dimer_wfn = None
        wfn_A = None
        wfn_B = None
        hf_wfn_dimer = None
        fsapt_induction_data = None

        # Need to collect external potentials (if exist) to properly set on each
        # SCF correctly. To use scf_helper for SAPT external potentials, we have to
        # manually set external potentials to kwargs["external_potentials"]["C"],
        # before each scf_helper call. This is done with
        # construct_external_potential_in_field_C to combine potentials. This
        # happens for both delta_HF and DFT scf's.
        ext_pot_C = external_potentials.get("C")
        if isinstance(ext_pot_C, np.ndarray):
            ext_pot_C = [np.array(x) for x in ext_pot_C]
        ext_pot_A = external_potentials.get("A")
        ext_pot_B = external_potentials.get("B")
        # A charge may be copied into A/B to reach a GRAC shift while it also sits
        # in C. Trim those copies once, here, so every union below stays additive
        # and the dimer field remains the sum of the two monomer fields.
        ext_pot_A_not_in_C = drop_rows_carried_by(ext_pot_A, ext_pot_C)
        ext_pot_B_not_in_C = drop_rows_carried_by(ext_pot_B, ext_pot_C)
        if run_hf_segment:
            core.set_global_option("DF_INTS_IO", "SAVE")
            core.timer_on("SAPT(DFT):Dimer SCF")
            # The SAPT0 terms share their names with the SAPT(DFT) ones, so the
            # checkpoint keeps them apart under a private prefix.
            hf_data = ckpt.private_scalars(_HF_DATA_PREFIX)
            ind = ckpt.private_scalars(_HF_IND_PREFIX)

            core.set_local_option("SCF", "SAVE_JK", True)

            def run_hf_dimer():
                if do_ext_potential:
                    kwargs["external_potentials"]["C"] = (
                        construct_external_potential_in_field_C(
                            [ext_pot_C, ext_pot_A_not_in_C, ext_pot_B_not_in_C]
                        )
                    )
                wfn = scf_helper(
                    "SCF", molecule=sapt_dimer, banner=f"SAPT(DFT): {hf_segment_label} Dimer", **kwargs
                )
                if do_ext_potential:
                    kwargs.pop("external_potentials")
                return wfn

            def run_hf_monomer(molecule, label, ext_pot, ext_pot_not_in_C):
                if do_ext_potential and (ext_pot is not None or ext_pot_C is not None):
                    kwargs["external_potentials"] = {
                        "C": construct_external_potential_in_field_C([ext_pot_C, ext_pot_not_in_C])
                    }
                # Monomers reuse the dimer JK object; a restored dimer wavefunction
                # arrives without one, so build it on first use only.
                wfn = scf_helper(
                    "SCF",
                    molecule=molecule,
                    banner=f"SAPT(DFT): {hf_segment_label} Monomer {label}",
                    jk=wfn_jk(prepare_restored_scf(hf_wfn_dimer)),
                    **kwargs,
                )
                if do_ext_potential and kwargs.get("external_potentials"):
                    kwargs.pop("external_potentials")
                return wfn

            hf_scf = dict(method="hf", reference="RHF", energies=hf_data, scalar_prefix=_HF_DATA_PREFIX)
            hf_wfn_dimer = ckpt.scf_stage(
                "hf_dimer_scf", run_hf_dimer, molecule=sapt_dimer, energy_key="HF DIMER", **hf_scf
            )
            core.timer_off("SAPT(DFT):Dimer SCF")

            core.timer_on("SAPT(DFT):Monomer A SCF")
            hf_wfn_A = ckpt.scf_stage(
                "hf_monomer_a_scf",
                lambda: run_hf_monomer(monomerA, "A", ext_pot_A, ext_pot_A_not_in_C),
                molecule=monomerA,
                energy_key="HF MONOMER A",
                **hf_scf,
            )
            core.timer_off("SAPT(DFT):Monomer A SCF")

            core.timer_on("SAPT(DFT):Monomer B SCF")
            # core.IO.change_file_namespace(97, "monomerA", "monomerB")

            hf_wfn_B = ckpt.scf_stage(
                "hf_monomer_b_scf",
                lambda: run_hf_monomer(monomerB, "B", ext_pot_B, ext_pot_B_not_in_C),
                molecule=monomerB,
                energy_key="HF MONOMER B",
                **hf_scf,
            )
            core.set_global_option("SAVE_JK", False)
            core.timer_off("SAPT(DFT):Monomer B SCF")

            # After HF scf_helper calls, reconstruct original external_potential
            # dictionary.
            if do_ext_potential:
                kwargs["external_potentials"] = {}
            if ext_pot_C is not None:
                kwargs["external_potentials"]["C"] = ext_pot_C
            if ext_pot_A is not None:
                kwargs["external_potentials"]["A"] = ext_pot_A
            if ext_pot_B is not None:
                kwargs["external_potentials"]["B"] = ext_pot_B

            if do_dft:  # For SAPT(HF) do the JK terms in sapt_dft()
                # Grab JK object and set to A (so we do not save many JK objects)
                sapt_jk = wfn_jk(hf_wfn_B)
                if sapt_jk is not None:
                    hf_wfn_A.set_jk(sapt_jk)
                core.set_global_option("SAVE_JK", False)

                # Move it back to monomer A
                # core.IO.change_file_namespace(97, "monomerB", "dimer")

                core.print_out("\n")
                core.print_out(
                    "         ---------------------------------------------------------\n"
                )
                segment_name = f"SAPT(DFT): {hf_segment_label} Segment"
                core.print_out("         " + segment_name.center(58) + "\n")
                core.print_out("\n")
                core.print_out(
                    "         " + "by Daniel G. A. Smith and Rob Parrish".center(58) + "\n"
                )
                core.print_out(
                    "         ---------------------------------------------------------\n"
                )
                core.print_out("\n")

                # Need to properly set external potentials for SAPT0 terms
                if do_ext_potential:
                    kwargs["external_potentials"] = {}
                    hf_wfn_dimer.del_potential_variable("C")
                    _set_external_potentials_to_wavefunction(
                        construct_external_potential_in_field_C([ext_pot_A, ext_pot_B]),
                        hf_wfn_dimer,
                    )
                    if ext_pot_C is not None:
                        kwargs["external_potentials"]["C"] = ext_pot_C
                    if ext_pot_A is not None:
                        kwargs["external_potentials"]["A"] = ext_pot_A
                        _set_external_potentials_to_wavefunction(ext_pot_A, hf_wfn_A)
                    if ext_pot_B is not None:
                        kwargs["external_potentials"]["B"] = ext_pot_B
                        _set_external_potentials_to_wavefunction(ext_pot_B, hf_wfn_B)

                def commit_hf_sapt(stage):
                    ckpt.commit(
                        stage,
                        scalars={
                            **{_HF_DATA_PREFIX + k: v for k, v in hf_data.items()},
                            **{_HF_IND_PREFIX + k: v for k, v in ind.items()},
                        },
                    )

                # The SAPT0 cache and JK are only rebuilt when a SAPT0 stage is
                # still outstanding; once the last one is stored nothing here
                # needs them.
                hf_sapt_stages = ["hf_sapt_elst", "hf_sapt_exch"]
                if induction_type != "NONE":
                    hf_sapt_stages.append("hf_sapt_ind")
                hf_cache_ein = None
                if ckpt.next_stage(hf_sapt_stages) is not None:
                    if sapt_jk is None:
                        sapt_jk = wfn_jk(prepare_restored_scf(hf_wfn_B))
                        hf_wfn_A.set_jk(sapt_jk)
                    # Build the SAPT0 cache needed for electrostatics and exchange.
                    hf_cache_ein = jk_terms.build_sapt_jk_cache(
                        hf_wfn_dimer,
                        hf_wfn_A,
                        hf_wfn_B,
                        sapt_jk,
                        True,
                        external_potentials=kwargs.get("external_potentials", None),
                    )

                    # Electrostatics
                    core.timer_on("SAPT(HF):elst")
                    if ckpt.pending("hf_sapt_elst"):
                        elst, extern_extern_IE = jk_terms.electrostatics(hf_cache_ein, True)
                        hf_data["extern_extern_IE"] = extern_extern_IE
                        hf_data.update(elst)
                        commit_hf_sapt("hf_sapt_elst")
                    core.timer_off("SAPT(HF):elst")

                    # Exchange
                    core.timer_on("SAPT(HF):exch")
                    if ckpt.pending("hf_sapt_exch"):
                        exch = jk_terms.exchange(hf_cache_ein, sapt_jk, True)
                        hf_data.update(exch)
                        commit_hf_sapt("hf_sapt_exch")
                    core.timer_off("SAPT(HF):exch")

                    if induction_type != "NONE" and ckpt.pending("hf_sapt_ind"):
                        core.timer_on("SAPT(HF):ind")
                        ind = jk_terms.induction(
                            hf_cache_ein,
                            sapt_jk,
                            True,
                            maxiter=core.get_option("SAPT", "MAXITER"),
                            conv=core.get_option("SAPT", "CPHF_R_CONVERGENCE"),
                            Sinf=core.get_option("SAPT", "DO_IND_EXCH_SINF"),
                        )
                        hf_data.update(ind)
                        commit_hf_sapt("hf_sapt_ind")
                        core.timer_off("SAPT(HF):ind")

                dhf_value = (
                    hf_data["HF DIMER"] - hf_data["HF MONOMER A"] - hf_data["HF MONOMER B"]
                )
                if do_delta_hf:
                    data["DHF VALUE"] = dhf_value

                core.print_out("\n")
                if induction_type == "NONE":
                    if do_delta_hf:
                        data["Delta HF Correction"] = (
                            dhf_value - hf_data["Elst10,r"] - hf_data["Exch10"]
                        )
                    core.print_out("   SAPT0 induction skipped; induction will be assigned from delta HF.\n")
                elif do_delta_hf:
                    core.print_out(
                        print_sapt_hf_summary(
                            hf_data,
                            "SAPT(HF)",
                            dimer_wfn=hf_wfn_dimer,
                            delta_hf=dhf_value,
                        )
                    )
                    data["Delta HF Correction"] = core.variable("SAPT(DFT) Delta HF")
                else:
                    core.print_out(print_sapt_hf_induction_summary(hf_data, "SAPT(HF)"))
                if induction_type == "CPHF":
                    data.update(ind)
                    if do_delta_hf:
                        hf_data["Delta HF Correction"] = data["Delta HF Correction"]
                    if fsapt_type == "FISAPT":
                        # The einsums exchange-induction path retains JK-owned
                        # J_P matrices. Clone them before finalizing the HF JK
                        # object so the later FISAPT::find() cannot dereference
                        # released storage.
                        hf_cache_ein["J_P_A"] = hf_cache_ein["J_P_A"].clone()
                        hf_cache_ein["J_P_B"] = hf_cache_ein["J_P_B"].clone()

                        # Retain the SAPT0 cache and wavefunctions so FISAPT::find()
                        # can partition the same HF induction used by CPHF.
                        fsapt_induction_data = (
                            hf_wfn_A,
                            hf_wfn_B,
                            hf_cache_ein,
                            hf_data.copy(),
                        )
                if sapt_jk is not None:
                    sapt_jk.finalize()

                del hf_wfn_A, hf_wfn_B, sapt_jk, hf_cache_ein
                # The DFT segment below allocates its own caches; without this the arenas these
                # wavefunctions leave behind are still resident when it does.
                core.release_freed_memory()

            else:
                wfn_A = hf_wfn_A
                wfn_B = hf_wfn_B
                data["DFT MONOMER A"] = hf_data["HF MONOMER A"]
                data["DFT MONOMER B"] = hf_data["HF MONOMER B"]
                dhf_value = (
                    hf_data["HF DIMER"] - hf_data["HF MONOMER A"] - hf_data["HF MONOMER B"]
                )
                data["DHF VALUE"] = dhf_value

        if hf_wfn_dimer is None and not do_fsapt:
            dimer_wfn = core.Wavefunction.build(sapt_dimer, core.get_global_option("BASIS"))
        # If we did not compute HF wavefunction, we still need orbital coefficients
        # for IBOLocalizer2
        elif hf_wfn_dimer is None and do_fsapt:
            dimer_wfn = ckpt.scf_stage(
                "dimer_localization_scf",
                lambda: scf_helper(
                    "SCF",
                    molecule=sapt_dimer,
                    banner="SAPT(DFT): Dimer for Localization",
                    **kwargs,
                ),
                method="hf",
                reference="RHF",
                molecule=sapt_dimer,
            )
        else:
            dimer_wfn = hf_wfn_dimer

        if do_dft or not do_delta_hf:
            # Set the primary functional
            core.set_local_option("SCF", "REFERENCE", "RKS")
            # An HF "functional" still builds an RHF wavefunction.
            dft_scf = dict(method=sapt_dft_functional.lower(), reference="RKS" if do_dft else "RHF")

            # Compute Monomer A wavefunction
            core.timer_on("SAPT(DFT): Monomer A DFT")
            if mon_a_shift:
                core.set_global_option("DFT_GRAC_SHIFT", mon_a_shift)

            def run_dft_monomer_a():
                if do_ext_potential and (ext_pot_A is not None or ext_pot_C is not None):
                    kwargs["external_potentials"] = {
                        "C": construct_external_potential_in_field_C([ext_pot_C, ext_pot_A_not_in_C])
                    }
                elif do_ext_potential:
                    kwargs["external_potentials"] = {}

                wfn = scf_helper(
                    sapt_dft_functional,
                    post_scf=False,
                    molecule=monomerA,
                    banner="SAPT(DFT): DFT Monomer A",
                    **kwargs,
                )
                if do_ext_potential and kwargs.get("external_potentials"):
                    kwargs.pop("external_potentials")
                return wfn

            core.set_global_option("SAVE_JK", True)
            wfn_A = ckpt.scf_stage(
                "monomer_a_dft_scf",
                run_dft_monomer_a,
                molecule=monomerA,
                energy_key="DFT MONOMERA",
                **dft_scf,
            )

            core.set_global_option("DFT_GRAC_SHIFT", 0.0)
            core.timer_off("SAPT(DFT): Monomer A DFT")

            # Compute Monomer B wavefunction
            core.timer_on("SAPT(DFT): Monomer B DFT")

            if mon_b_shift:
                core.set_global_option("DFT_GRAC_SHIFT", mon_b_shift)

            def run_dft_monomer_b():
                if do_ext_potential and (ext_pot_B is not None or ext_pot_C is not None):
                    kwargs["external_potentials"] = {
                        "C": construct_external_potential_in_field_C([ext_pot_C, ext_pot_B_not_in_C])
                    }
                return scf_helper(
                    sapt_dft_functional,
                    post_scf=False,
                    molecule=monomerB,
                    banner="SAPT(DFT): DFT Monomer B",
                    jk=wfn_jk(prepare_restored_scf(wfn_A)),
                    **kwargs,
                )

            core.set_global_option("SAVE_JK", True)
            wfn_B = ckpt.scf_stage(
                "monomer_b_dft_scf",
                run_dft_monomer_b,
                molecule=monomerB,
                energy_key="DFT MONOMERB",
                **dft_scf,
            )
            core.timer_off("SAPT(DFT): Monomer B DFT")
            if do_ext_potential:
                kwargs["external_potentials"] = {}
            if ext_pot_C is not None:
                kwargs["external_potentials"]["C"] = ext_pot_C
            if ext_pot_A is not None:
                kwargs["external_potentials"]["A"] = ext_pot_A
            if ext_pot_B is not None:
                kwargs["external_potentials"]["B"] = ext_pot_B
        # Reset external potentials on kwargs['external_potentials']
        kwargs["external_potentials"] = {}
        if do_ext_potential:
            dimer_wfn.del_potential_variable("C")
            _set_external_potentials_to_wavefunction(
                construct_external_potential_in_field_C([ext_pot_A, ext_pot_B]),
                dimer_wfn,
            )
            if ext_pot_C is not None:
                kwargs["external_potentials"]["C"] = ext_pot_C
            if ext_pot_A is not None:
                kwargs["external_potentials"]["A"] = ext_pot_A
                _set_external_potentials_to_wavefunction(ext_pot_A, wfn_A)
            if ext_pot_B is not None:
                kwargs["external_potentials"]["B"] = ext_pot_B
                _set_external_potentials_to_wavefunction(ext_pot_B, wfn_B)

        # Save JK object when available; restored wavefunctions carry none.
        # Rebuilding one is expensive, so only do it when a stage that actually
        # needs a JK object is still outstanding.
        sapt_jk = wfn_jk(wfn_B)
        pending_sapt_stage = ckpt.next_stage(
            _sapt_stage_order(do_disp=do_disp, do_fsapt=do_fsapt, do_dft=do_dft, induction_type=induction_type)
        )
        needs_restored_sapt_jk = sapt_jk is None and (
            (do_delta_dft and do_dft and ckpt.pending("delta_dft")) or pending_sapt_stage not in {None, "fsapt_final"}
        )
        if needs_restored_sapt_jk:
            sapt_jk = wfn_jk(prepare_restored_scf(wfn_B))
        if sapt_jk is not None:
            wfn_A.set_jk(sapt_jk)

        if do_delta_dft and do_dft:
            optstash2 = p4util.OptionsState(
                ["SCF_TYPE"],
                ["SCF", "REFERENCE"],
                ["SCF", "DFT_GRAC_SHIFT"],
                ["SCF", "SAVE_JK"],
            )
            core.set_local_option("SCF", "DFT_GRAC_SHIFT", 0.0)
            # Enable SAVE_JK so JK objects can be reused across calculations
            core.set_local_option("SCF", "SAVE_JK", True)
            core.print_out("\n")
            core.print_out(
                "         ---------------------------------------------------------\n"
            )
            core.print_out("         " + "SAPT(DFT): delta DFT Segment".center(58) + "\n")
            core.print_out("\n")
            core.timer_on("SAPT(DFT):delta DFT")

            dimer_dft_kwargs = {}
            monomer_a_dft_kwargs = {}
            monomer_b_dft_kwargs = {}
            if do_ext_potential:
                dimer_dft_kwargs["external_potentials"] = {
                    "C": construct_external_potential_in_field_C(
                        [ext_pot_C, ext_pot_A_not_in_C, ext_pot_B_not_in_C]
                    )
                }
                monomer_a_dft_kwargs["external_potentials"] = {
                    "C": construct_external_potential_in_field_C([ext_pot_C, ext_pot_A_not_in_C])
                }
                monomer_b_dft_kwargs["external_potentials"] = {
                    "C": construct_external_potential_in_field_C([ext_pot_C, ext_pot_B_not_in_C])
                }

            def delta_dft_scf(stage, molecule, timer, energy_key, scf_kwargs):
                """Supermolecular SCF for one term of the delta-DFT correction.

                Only its energy is used downstream, so only the energy is stored.
                """
                core.timer_on(timer)
                try:
                    ckpt.energy_stage(
                        stage,
                        lambda: run_scf(sapt_dft_functional.lower(), molecule=molecule, jk=sapt_jk, **scf_kwargs),
                        energy_key=energy_key,
                    )
                finally:
                    core.timer_off(timer)

            if ckpt.pending("delta_dft"):
                delta_dft_scf("delta_dft_dimer_scf", sapt_dimer, "SAPT(DFT):Dimer DFT", "DFT DIMER ENERGY", dimer_dft_kwargs)
                delta_dft_scf("delta_dft_monomer_a_scf", monomerA, "SAPT(DFT):Monomer A DFT", "DFT MONOMER A ENERGY", monomer_a_dft_kwargs)
                delta_dft_scf("delta_dft_monomer_b_scf", monomerB, "SAPT(DFT):Monomer B DFT", "DFT MONOMER B ENERGY", monomer_b_dft_kwargs)

                data["DFT IE"] = (
                    data["DFT DIMER ENERGY"]
                    - data["DFT MONOMER A ENERGY"]
                    - data["DFT MONOMER B ENERGY"]
                )
                ckpt.commit("delta_dft", scalars={"DFT IE": data["DFT IE"]})

            core.timer_off("SAPT(DFT):delta DFT")
            core.print_out("\n")
            optstash2.restore()
        elif do_delta_dft and not do_dft:
            raise ValueError(
                "SAPT(DFT): delta DFT correction requested when running HF. Set SAPT_DFT_DO_DDFT to False or use a DFT functional."
            )

        # If a -D dispersion requested above, we now compute those values
        if sapt_dft_D4_IE:
            core.print_out("\n")
            core.print_out(
                "         ---------------------------------------------------------\n"
            )
            core.print_out(
                "         " + "SAPT(DFT): D4 Interaction Energy".center(58) + "\n"
            )
            core.print_out("\n")
            core.timer_on("SAPT(DFT):D4 Interaction Energy")
            d4_type = core.get_option("SAPT", "SAPT_DFT_D_TYPE").lower()

            if ckpt.pending("d4"):
                edisp_interaction_energy.sapt_dft_d4_interaction_energy(
                    sapt_dimer=sapt_dimer,
                    monomerA=monomerA,
                    monomerB=monomerB,
                    dimer_wfn=dimer_wfn,
                    dftd4_functional_name=e_disp_param_name,
                    d4_type=d4_type,
                    data=data,
                )
                ckpt.commit_data_arrays("d4")
            else:
                ckpt.restore_data_arrays("d4")
            core.timer_off("SAPT(DFT):D4 Interaction Energy")
        elif sapt_dft_D3_IE:
            core.print_out("\n")
            core.print_out(
                "         ---------------------------------------------------------\n"
            )
            core.print_out(
                "         " + "SAPT(DFT): D3 Interaction Energy".center(58) + "\n"
            )
            core.print_out("\n")
            core.timer_on("SAPT(DFT):D3 Interaction Energy")
            d3_type = core.get_option("SAPT", "SAPT_DFT_D_TYPE").lower()

            if ckpt.pending("d3"):
                edisp_interaction_energy.sapt_dft_d3_interaction_energy(
                    sapt_dimer=sapt_dimer,
                    monomerA=monomerA,
                    monomerB=monomerB,
                    dimer_wfn=dimer_wfn,
                    dftd3_functional_name=e_disp_param_name,
                    d3_type=d3_type,
                    data=data,
                )
                ckpt.commit_data_arrays("d3")
            else:
                ckpt.restore_data_arrays("d3")
            core.timer_off("SAPT(DFT):D3 Interaction Energy")

        core.set_global_option("SAVE_JK", False)
        core.set_global_option("DFT_GRAC_SHIFT", 0.0)

        # Write out header
        scf_alg = core.get_global_option("SCF_TYPE")
        sapt_dft_header(
            sapt_dft_functional, mon_a_shift, mon_b_shift, bool(do_delta_hf), scf_alg
        )

        # Call SAPT(DFT)
        sapt_jk = wfn_jk(wfn_B)
        sapt_dft(
            dimer_wfn,
            wfn_A,
            wfn_B,
            do_dft=do_dft,
            sapt_jk=sapt_jk,
            data=data,
            print_header=False,
            delta_hf=do_delta_hf,
            cleanup_jk=True,
            external_potentials=kwargs.get("external_potentials", None),
            do_delta_dft=do_delta_dft,
            do_disp=do_disp,
            fsapt_induction_data=fsapt_induction_data,
            checkpoint=ckpt,
        )

        # Copy data back into globals
        for k, v in data.items():
            core.set_variable(k, v)
            dimer_wfn.set_variable(k, v)

        # The per-component QCVariables are published by the summary printer rather
        # than collected in `data`, so store them alongside it; a "final" restart has
        # nothing else to republish them from.
        ckpt.commit_final(dimer_wfn, scalars={**data, **_sapt_summary_qcvariables()})
        return dimer_wfn


def drop_rows_carried_by(potential, reference):
    """Drop point/diffuse rows of *potential* that *reference* already carries.

    Used to copy a charge into A/B (so it enters that monomer's GRAC shift)
    while it also sits in the environment field C, without charging the
    consuming monomer twice. Only rows that are exactly equal to a reference
    row are removed, and only against C: rows shared between A and B must be
    left alone, since A and B partition the field the dimer sees.

    Matrix operators are opaque and pass through untouched.
    """
    if not potential or not reference:
        return potential
    trimmed = {}
    for mode, values in potential.items():
        if mode == "matrix":
            trimmed[mode] = values
            continue
        carried = {tuple(row) for row in reference.get(mode, [])}
        trimmed[mode] = [row for row in values if tuple(row) not in carried]
    return trimmed


def construct_external_potential_in_field_C(potentials):
    """Concatenate point/diffuse rows; sum opaque matrix operators.

    Rows are never dropped here. The field the dimer sees has to equal the
    sum of the fields the monomers see or the embedding contribution fails
    to cancel in the SAPT decomposition, and two equal rows mean two equal
    charges. Use :py:func:`drop_rows_carried_by` to remove a monomer's rows
    that C already carries before combining.
    """
    combined = {}
    for potential in potentials:
        if not potential:
            continue
        for mode, values in potential.items():
            if mode == "matrix":
                matrix = np.asarray(values)
                if mode in combined:
                    if np.asarray(combined[mode]).shape != matrix.shape:
                        raise ValidationError(
                            "SAPT(DFT): external-potential matrices must have identical dimensions before combination."
                        )
                    combined[mode] = (np.asarray(combined[mode]) + matrix).tolist()
                else:
                    combined[mode] = matrix.tolist()
            else:
                combined.setdefault(mode, []).extend(values)
    return combined


sapt_dft_grac_convergence_tier_options = {
    "SINGLE": [
        {
            "SCF_INITIAL_ACCELERATOR": "ADIIS",
        }
    ],
    "ITERATIVE": [
        {
            "SCF_INITIAL_ACCELERATOR": "ADIIS",
        },
        {
            "LEVEL_SHIFT": 0.1,
            "LEVEL_SHIFT_CUTOFF": 1e-05,
            "SCF_INITIAL_ACCELERATOR": "ADIIS",
            "MAXITER": 200,
        },
        {
            "LEVEL_SHIFT": 0.5,
            "LEVEL_SHIFT_CUTOFF": 1e-3,
            "SCF_INITIAL_ACCELERATOR": "ADIIS",
            "MAXITER": 200,
        },
        {
            "LEVEL_SHIFT": 0.01,
            "LEVEL_SHIFT_CUTOFF": 1e-2,
            "SCF_INITIAL_ACCELERATOR": "ADIIS",
            "MAXITER": 200,
        },
    ],
}


def compute_GRAC_shift(
    molecule: core.Molecule,
    sapt_dft_grac_convergence_tier: str,
    label: str,
    jk_obj: core.JK | None = None,
    results: dict | None = None,
    external_potentials: dict | None = None,
    checkpoint: CheckpointSession | None = None,
) -> float:
    """Compute the GRAC (gradient-regulated asymptotic correction) shift for a monomer.

    Estimates the ionization energy of the monomer by computing neutral
    and cation SCF energies, then determines the GRAC shift as the
    difference between the ionization energy and the DFT HOMO energy.
    Uses a tiered convergence strategy for robustness.

    Parameters
    ----------
    molecule : core.Molecule
        The monomer molecule.
    sapt_dft_grac_convergence_tier : str
        Convergence tier for GRAC computation (controls SCF settings).
    label : str
        Label identifying the monomer (e.g., ``'A'`` or ``'B'``).
    jk_obj : core.JK or None, optional
        Pre-built JK object, by default None.
    results : dict or None, optional
        Destination for computed intermediate QCVariables.
    external_potentials : dict or None, optional
        Monomer's own normalized potential, applied to both charge states.
    checkpoint : CheckpointSession or None, optional
        Open checkpoint session. The neutral and the electron-removed SCF are
        separate stages, so a restart skips whichever of the two has finished.

    Returns
    -------
    float
        The GRAC shift value in Hartree.
    """
    ckpt = checkpoint or CheckpointSession.disabled()
    monomer_label = label
    stage = f"grac_monomer_{monomer_label.lower()}"
    neutral_stage = f"{stage}_neutral"
    private_prefix = f"{_GRAC_PREFIX}{monomer_label}::"
    label = f"Monomer {label}"

    def publish(E_given, E_cation, HOMO):
        values = {
            f"SAPT DFT GRAC MONOMER ENERGY {monomer_label}": E_given,  # P::e SAPT
            f"SAPT DFT GRAC IONIZED MONOMER ENERGY {monomer_label}": E_cation,  # P::e SAPT
            f"SAPT DFT GRAC HOMO {monomer_label}": HOMO,  # P::e SAPT
            f"SAPT DFT GRAC IP {monomer_label}": E_cation - E_given,  # P::e SAPT
        }
        for key, value in values.items():
            core.set_variable(key, value)
            if results is not None:
                results[key] = value
        return values

    if ckpt.done(stage):
        stored = ckpt.private_scalars(private_prefix)
        grac = stored["E_cation"] - stored["E_given"] + stored["HOMO"]
        core.print_out(f"\n   GRAC shift for {label} restored from checkpoint: {grac:.8f}\n")
        publish(stored["E_given"], stored["E_cation"], stored["HOMO"])
        return grac

    # A neutral SCF stored by an earlier run converged at a known tier. The
    # ladder resumes there, re-applying the earlier tiers' options so the
    # electron-removed SCF sees exactly the settings it would have seen.
    neutral = ckpt.private_scalars(private_prefix) if ckpt.done(neutral_stage) else {}
    resume_tier = neutral.get("tier")

    optstash = p4util.OptionsState(
        ["SCF_TYPE"],
        ["SCF", "REFERENCE"],
        ["SCF", "DFT_GRAC_SHIFT"],
        ["SCF", "SAVE_JK"],
        ["SCF", "MAXITER"],
        ["SCF", "LEVEL_SHIFT"],
        ["SCF", "LEVEL_SHIFT_CUTOFF"],
        ["SCF", "SCF_INITIAL_ACCELERATOR"],
        ["SCF", "ORBITAL_OPTIMIZER_PACKAGE"],
        ["BASIS"],
    )

    scf_kwargs = {}
    if external_potentials is not None:
        scf_kwargs["external_potentials"] = construct_external_potential_in_field_C([external_potentials])
    core.timer_on("SAPT(DFT):GRAC Shift " + label)
    try:
        dft_functional = core.get_option("SAPT", "SAPT_DFT_FUNCTIONAL")
        grac_basis = core.get_option("SAPT", "SAPT_DFT_GRAC_BASIS")
        if grac_basis != "AUTO":
            core.set_global_option("BASIS", grac_basis)

        core.print_out(
            f"Computing GRAC shift for {label} using {sapt_dft_grac_convergence_tier}..."
        )
        grac_options = sapt_dft_grac_convergence_tier_options[
            sapt_dft_grac_convergence_tier
        ]
        grac = None
        for tier, options in enumerate(grac_options):
            for key, val in options.items():
                core.set_local_option("SCF", key, val)
            core.set_local_option("SCF", "ORBITAL_OPTIMIZER_PACKAGE", "INTERNAL")
            if resume_tier is not None and tier < resume_tier:
                continue
            # Need to get the initial and cation to estimate ionization energy for
            # GRAC shift
            mol_qcel_dict = molecule.to_schema(dtype=3)
            del mol_qcel_dict["fragment_charges"]
            del mol_qcel_dict["fragment_multiplicities"]
            del mol_qcel_dict["molecular_multiplicity"]

            mol_given = core.Molecule.from_schema(mol_qcel_dict)
            mol_qcel_dict["molecular_charge"] += 1
            mol_cation = core.Molecule.from_schema(mol_qcel_dict)

            core.print_out(
                f"\n\n  ==> GRAC {label} Given Molecule: charge={mol_given.molecular_charge()} mult={mol_given.multiplicity()} <==\n\n"
            )
            try:
                if tier == resume_tier:
                    E_given = neutral["E_given"]
                    HOMO = neutral["HOMO"]
                    core.print_out(f"   Neutral SCF restored from checkpoint: E = {E_given:.12f}\n")
                else:
                    if mol_given.multiplicity() != 1:
                        core.set_local_option("SCF", "REFERENCE", "UHF")
                    else:
                        core.set_local_option("SCF", "REFERENCE", "RHF")
                    # Set SAVE_JK=True so we can reuse the JK object for the cation calc
                    core.set_local_option("SCF", "SAVE_JK", True)
                    wfn_given = run_scf(
                        dft_functional.lower(),
                        molecule=mol_given,
                        jk=jk_obj,
                        **scf_kwargs,
                    )
                    # We don't want to keep re-computing JK objects if we can avoid it
                    if jk_obj is None:
                        jk_obj = wfn_given.jk()
                    occ_given = wfn_given.epsilon_a_subset(basis="SO", subset="OCC").to_array(
                        dense=True
                    )
                    HOMO = np.amax(occ_given)
                    E_given = wfn_given.energy()
                    wfn_given = None
                    ckpt.commit(
                        neutral_stage,
                        scalars={
                            private_prefix + "tier": tier,
                            private_prefix + "E_given": E_given,
                            private_prefix + "HOMO": HOMO,
                        },
                    )
                if mol_cation.multiplicity() != 1:
                    core.set_local_option("SCF", "REFERENCE", "UHF")
                else:
                    core.set_local_option("SCF", "REFERENCE", "RHF")
                core.print_out(
                    f"\n\n  ==> GRAC {label} Electron Removed Molecule: charge={mol_cation.molecular_charge()} mult={mol_cation.multiplicity()} <==\n\n"
                )
                wfn_cation = run_scf(
                    dft_functional.lower(),
                    molecule=mol_cation,
                    jk=jk_obj,
                    **scf_kwargs,
                )
            except ConvergenceError:
                # A failed attempt's wavefunctions are still bound in this
                # function's scope, so without dropping them here they stay
                # resident -- grid data, collocation cache and all -- while the
                # next convergence tier allocates its own from the full budget.
                wfn_given = None
                wfn_cation = None
                if len(grac_options) == 1:
                    raise Exception(
                        "Convergence error in GRAC shift calculation, please try a different convergence tier."
                    )
                else:
                    core.print_out("Convergence error, trying next GRAC iteration...")
                continue

            E_cation = wfn_cation.energy()
            wfn_cation = None
            grac = E_cation - E_given + HOMO
            if grac >= 1 or grac <= -1:
                raise ValueError(
                    f"The computed GRAC shift ({grac} [E_h]) for {label} exceeds the bounds of -1 < x < 1 and should not be used to approximate the ionization potential."
                    + (" Try disabling SAPT_DFT_GRAC_USE_EXT_POT."
                       if core.get_option("SAPT", "SAPT_DFT_GRAC_USE_EXT_POT") else "")
                )
            break
        if grac is None:
            raise ValueError(
                "Failed to converge the input monomer or its cation for computing the GRAC shift for Monomer" + label
            )
        core.print_out(f" GRAC shift {label}: {grac:.8f}\n")
        core.print_out(f" {E_given = :.8f}, {E_cation = :.8f}, {HOMO = :.8f}\n")
        values = publish(E_given, E_cation, HOMO)
        ckpt.commit(
            stage,
            scalars={
                **values,
                f"SAPT DFT GRAC SHIFT {monomer_label}": grac,
                private_prefix + "E_given": E_given,
                private_prefix + "E_cation": E_cation,
                private_prefix + "HOMO": HOMO,
            },
        )
        return grac
    finally:
        optstash.restore()
        core.timer_off("SAPT(DFT):GRAC Shift " + label)


def sapt_dft_header(
    sapt_dft_functional: str = "unknown",
    mon_a_shift: float | None = None,
    mon_b_shift: float | None = None,
    do_delta_hf: str = "N/A",
    jk_alg: str = "N/A",
) -> None:
    """Print the SAPT(DFT) calculation header and algorithm summary.

    Outputs a formatted banner and algorithm settings to the Psi4 output
    file, including the DFT functional, GRAC shifts, delta-HF flag, and
    JK algorithm.

    Parameters
    ----------
    sapt_dft_functional : str, optional
        Name of the DFT functional, by default ``'unknown'``.
    mon_a_shift : float or None, optional
        GRAC shift for monomer A, by default None.
    mon_b_shift : float or None, optional
        GRAC shift for monomer B, by default None.
    do_delta_hf : str, optional
        Whether delta-HF correction is applied, by default ``'N/A'``.
    jk_alg : str, optional
        JK algorithm type, by default ``'N/A'``.

    Returns
    -------
    None
    """
    # Print out the title and some information
    core.print_out("\n")
    core.print_out(
        "         ---------------------------------------------------------\n"
    )
    core.print_out(
        "         " + "SAPT(DFT): Intermolecular Interaction Segment".center(58) + "\n"
    )
    core.print_out("\n")
    core.print_out(
        "         " + "by Daniel G. A. Smith and Rob Parrish".center(58) + "\n"
    )
    core.print_out(
        "         ---------------------------------------------------------\n"
    )
    core.print_out("\n")

    core.print_out("  ==> Algorithm <==\n\n")
    core.print_out("   SAPT DFT Functional     %12s\n" % str(sapt_dft_functional))
    if mon_a_shift:
        core.print_out("   Monomer A GRAC Shift    %12.6f\n" % mon_a_shift)
    if mon_b_shift:
        core.print_out("   Monomer B GRAC Shift    %12.6f\n" % mon_b_shift)
    core.print_out("   Delta HF                %12s\n" % do_delta_hf)
    core.print_out("   JK Algorithm            %12s\n" % jk_alg)


def sapt_dft(
    dimer_wfn: core.Wavefunction,
    wfn_A: core.Wavefunction,
    wfn_B: core.Wavefunction,
    do_dft: bool = True,
    sapt_jk: core.JK | None = None,
    sapt_jk_B: core.JK | None = None,
    data: dict | None = None,
    print_header: bool = True,
    cleanup_jk: bool = True,
    delta_hf: bool = False,
    external_potentials: dict | None = None,
    do_delta_dft: bool = False,
    do_disp: bool = True,
    fsapt_induction_data: tuple | None = None,
    checkpoint: CheckpointSession | None = None,
) -> dict:
    """Compute the SAPT(DFT) interaction energy components.

    Primary algorithm for computing SAPT(DFT) interaction energy once
    monomer wavefunctions have been built. Computes electrostatics,
    exchange, induction, and dispersion components.

    Parameters
    ----------
    dimer_wfn : core.Wavefunction
        Dimer wavefunction providing the dimer basis set.
    wfn_A : core.Wavefunction
        Converged monomer A wavefunction.
    wfn_B : core.Wavefunction
        Converged monomer B wavefunction.
    do_dft : bool, optional
        Whether to use DFT-based exchange-correlation, by default True.
    sapt_jk : core.JK or None, optional
        Pre-built JK object for the dimer basis, by default None (built internally).
    sapt_jk_B : core.JK or None, optional
        Separate JK object for monomer B, by default None (uses ``sapt_jk``).
    data : dict or None, optional
        Pre-existing data dictionary to update, by default None.
    print_header : bool, optional
        Whether to print the SAPT(DFT) header, by default True.
    cleanup_jk : bool, optional
        Whether to finalize and clean up the JK object, by default True.
    delta_hf : bool, optional
        Whether to include the delta-HF correction, by default False.
    external_potentials : dict or None, optional
        External potentials for embedding calculations, by default None.
    do_delta_dft : bool, optional
        Whether to compute delta-DFT correction, by default False.
    do_disp : bool, optional
        Whether to compute dispersion, by default True.
    fsapt_induction_data : tuple or None, optional
        Private transport of SAPT0 monomer wavefunctions, cache, and scalar data
        used to partition CPHF induction with the FISAPT implementation.
    checkpoint : CheckpointSession or None, optional
        Open checkpoint session from :func:`run_sapt_dft`. Stages it already
        holds are restored instead of recomputed. By default None, meaning
        every stage runs and nothing is stored.

    Returns
    -------
    dict
        Dictionary of SAPT(DFT) interaction energy components (in Hartree).

    Examples
    --------
    >>> dimer = psi4.geometry('''
    ...   Ne
    ...   --
    ...   Ar 1 6.5
    ...   units bohr
    ... ''')
    >>> psi4.set_options({"BASIS": "aug-cc-pVDZ"})
    >>> sapt_dimer, monomerA, monomerB = psi4.proc_util.prepare_sapt_molecule(
    ...     sapt_dimer, "dimer"
    ... )
    >>> psi4.set_options({"DFT_GRAC_SHIFT": 0.203293})
    >>> wfnA, energyA = psi4.energy("PBE0", molecule=monomerA, return_wfn=True)
    >>> psi4.set_options({"DFT_GRAC_SHIFT": 0.138264})
    >>> wfnB, energyB = psi4.energy("PBE0", molecule=monomerB, return_wfn=True)
    >>> wfnD = psi4.core.Wavefunction.build(sapt_dimer)
    >>> data = psi4.procrouting.sapt.sapt_dft(wfnD, wfnA, wfnB)
    """

    # Handle the input options
    if data is None:
        data = {}

    induction_type = core.get_option("SAPT", "SAPT_DFT_INDUCTION_TYPE").upper()
    fsapt_type = core.get_option("SAPT", "SAPT_DFT_DO_FSAPT").upper()
    if induction_type == "CPHF" and do_dft and "Ind20,r" not in data:
        raise ValidationError(
            "SAPT_DFT_INDUCTION_TYPE=CPHF reuses the SAPT0 induction terms computed by the "
            "SAPT(DFT) delta HF segment, which sapt_dft() does not run. Call "
            "energy('sapt(dft)') instead of sapt_dft() directly, or supply the SAPT0 "
            "induction terms in `data`."
        )
    if (
        induction_type == "CPHF"
        and do_dft
        and fsapt_type == "FISAPT"
        and fsapt_induction_data is None
    ):
        raise ValidationError(
            "SAPT_DFT_INDUCTION_TYPE=CPHF with FISAPT requires HF-backed fragment induction "
            "data from the SAPT0 segment. Call energy('sapt(dft)') instead of sapt_dft() directly."
        )
    if induction_type == "NONE" and delta_hf and "Delta HF Correction" not in data:
        # For SAPT(DFT) the delta HF segment owns the SAPT0 terms, so the
        # correction can only come from there. For SAPT(HF) the SAPT0
        # electrostatics and exchange are computed below in this very function,
        # so only the total HF interaction energy has to be handed in.
        if do_dft:
            raise ValidationError(
                "SAPT_DFT_INDUCTION_TYPE=NONE with delta HF needs the total SAPT0 induction "
                "produced by the SAPT(DFT) delta HF segment, which sapt_dft() does not run. "
                "Call energy('sapt(dft)') instead of sapt_dft() directly, or supply that "
                "total as 'Delta HF Correction' in `data`."
            )
        if "DHF VALUE" not in data:
            raise ValidationError(
                "SAPT_DFT_INDUCTION_TYPE=NONE with delta HF needs the total HF interaction "
                "energy produced by the delta HF segment, which sapt_dft() does not run. "
                "Call energy('sapt(dft)') instead of sapt_dft() directly, or supply it as "
                "'DHF VALUE' in `data`."
            )

    if print_header:
        sapt_dft_header()

    ckpt = (checkpoint or CheckpointSession.disabled()).bind(data=data)
    do_fsapt = fsapt_type != "NONE"
    do_ind = _sapt_dft_runs_induction(do_dft=do_dft, induction_type=induction_type)
    pending_stage = ckpt.next_stage(
        _sapt_stage_order(do_disp=do_disp, do_fsapt=do_fsapt, do_dft=do_dft, induction_type=induction_type)
    )
    pending_fsapt_stage = ckpt.next_stage(_fsapt_stage_order(do_disp=do_disp)) if do_fsapt else None
    use_einsums = core.get_option("SAPT", "SAPT_DFT_USE_EINSUMS")

    # Build SAPT cache only when a remaining computational stage still needs it.
    if einsums_available and use_einsums:
        jk_terms = sapt_jk_terms_ein
        sapt_mp2 = sapt_mp2_terms_ein
    else:
        # If einsums is not available, need to conditionally stop einsums
        # without adding einsums_available and use_einsums to every check.
        use_einsums = False
        jk_terms = sapt_jk_terms
        sapt_mp2 = sapt_mp2_terms

    cache = {}
    if pending_stage is not None and pending_stage != "fsapt_final":
        core.timer_on("SAPT(DFT):Build JK")
        if sapt_jk is None:
            core.print_out("\n   => Building SAPT JK object <= \n\n")
            sapt_jk = core.JK.build(dimer_wfn.basisset())
            sapt_jk.set_do_J(True)
            sapt_jk.set_do_K(True)
            wfn_A_is_lrc = functional_value(wfn_A, "is_x_lrc", False)
            wfn_A_omega = functional_value(wfn_A, "x_omega", 0.0)
            wfn_B_is_lrc = functional_value(wfn_B, "is_x_lrc", False)
            wfn_B_omega = functional_value(wfn_B, "x_omega", 0.0)
            if wfn_A_is_lrc:
                sapt_jk.set_do_wK(True)
                sapt_jk.set_omega(wfn_A_omega)
            sapt_jk.initialize()
            sapt_jk.print_header()
            if wfn_B_is_lrc and (wfn_A_omega != wfn_B_omega):
                core.print_out("   => Monomer B: Building SAPT JK object <= \n\n")
                core.print_out("      Reason: MonomerA Omega != MonomerB Omega\n\n")
                sapt_jk_B = core.JK.build(dimer_wfn.basisset())
                sapt_jk_B.set_do_J(True)
                sapt_jk_B.set_do_K(True)
                sapt_jk_B.set_do_wK(True)
                sapt_jk_B.set_omega(wfn_B_omega)
                sapt_jk_B.initialize()
                sapt_jk_B.print_header()
        else:
            sapt_jk.set_do_K(True)

        sapt_jk.set_do_J(True)
        sapt_jk.set_do_K(True)

        if functional_value(wfn_A, "is_x_lrc", False):
            sapt_jk.set_do_wK(True)
            sapt_jk.set_omega(functional_value(wfn_A, "x_omega", 0.0))

        cache = jk_terms.build_sapt_jk_cache(
            dimer_wfn, wfn_A, wfn_B, sapt_jk, True, external_potentials
        )
        ckpt.bind(cache=cache)
        core.timer_off("SAPT(DFT):Build JK")

    # Electrostatics
    core.timer_on("SAPT(DFT):elst")
    if ckpt.pending("elst"):
        elst, extern_extern_IE = jk_terms.electrostatics(cache, True)
        data["extern_extern_IE"] = extern_extern_IE
        data.update(elst)
        ckpt.commit("elst")
    else:
        elst = {"Elst10,r": data["Elst10,r"]}
        extern_extern_IE = data.get("extern_extern_IE", 0.0)
    core.timer_off("SAPT(DFT):elst")

    # Exchange
    core.timer_on("SAPT(DFT):exch")
    if ckpt.pending("exch"):
        exch = jk_terms.exchange(cache, sapt_jk, True)
        data.update(exch)
        ckpt.commit("exch")
    else:
        exch = {"Exch10": data["Exch10"], "Exch10(S^2)": data.get("Exch10(S^2)", data["Exch10"])}
    ckpt.restore_cache(cache, ("exch",), only_missing=True)
    core.timer_off("SAPT(DFT):exch")

    # Induction
    core.timer_on("SAPT(DFT):ind")
    if do_ind and ckpt.pending("ind"):
        ind = jk_terms.induction(
            cache,
            sapt_jk,
            True,
            sapt_jk_B=sapt_jk_B,
            maxiter=core.get_option("SAPT", "MAXITER"),
            conv=core.get_option("SAPT", "CPHF_R_CONVERGENCE"),
            Sinf=core.get_option("SAPT", "DO_IND_EXCH_SINF"),
        )
        data.update(ind)
        ckpt.commit("ind")
    elif not do_ind:
        core.print_out(f"\n   SAPT(DFT) induction skipped ({induction_type}).\n")
    ckpt.restore_cache(cache, ("exch",), only_missing=True)

    if induction_type == "NONE":
        if delta_hf:
            if "Delta HF Correction" not in data:
                # SAPT(HF): the delta HF segment computed only the HF dimer and
                # monomer energies, so build the induction-carrying correction
                # here from the electrostatics and exchange just computed. This
                # is the same expression the SAPT(DFT) delta HF segment uses.
                data["Delta HF Correction"] = data["DHF VALUE"] - (
                    data["Elst10,r"] + data["Exch10"]
                )
            core.set_variable("SAPT(DFT) Delta HF", data["Delta HF Correction"])
    elif delta_hf and "Delta HF Correction" not in data:
        total_sapt = (
            data["Elst10,r"] + data["Exch10"] + data["Ind20,r"] + data["Exch-Ind20,r"]
        )
        sapt_hf_delta = data["DHF VALUE"] - total_sapt
        core.set_variable("SAPT(DFT) Delta HF", sapt_hf_delta)
        data["Delta HF Correction"] = sapt_hf_delta

    # Set Delta DFT for SAPT(DFT) if requested
    if do_delta_dft:
        base_ind = (
            0.0
            if induction_type == "NONE"
            else data["Ind20,r"] + data["Exch-Ind20,r"]
        )
        sapt_dft_elst_exch_indu = data["Elst10,r"] + data["Exch10"] + base_ind
        sapt_dft_delta = data["DFT IE"] - sapt_dft_elst_exch_indu
        core.set_variable("SAPT(DFT) Delta DFT", sapt_dft_delta)
        data["Delta DFT Correction"] = core.variable("SAPT(DFT) Delta DFT")

    core.timer_off("SAPT(DFT):ind")

    ckpt.restore_cache(cache, _FSAPT_CACHE_STAGES)
    if do_fsapt and fsapt_type == "SAPTDFT" and use_einsums and ckpt.done("fsapt_elst"):
        _rebuild_einsums_fsapt_elst_cache(cache, dimer_wfn)

    # Use DFHelper before deleting the JK object for dispersion
    FISAPT_obj = None
    if do_fsapt and fsapt_type == "SAPTDFT" and use_einsums and pending_fsapt_stage not in {None, "fsapt_final"}:
        if ckpt.pending("fsapt_setup"):
            core.timer_on("SAPT(DFT):Localize Orbitals")
            jk_terms.localization(cache, dimer_wfn)
            core.timer_off("SAPT(DFT):Localize Orbitals")
            core.timer_on("SAPT(DFT):Partition")
            cache = jk_terms.partition(cache, dimer_wfn)
            core.timer_off("SAPT(DFT):Partition")

            core.timer_on("SAPT(DFT): F-SAPT Localization (IBO)")
            jk_terms.flocalization(cache, dimer_wfn)
            ckpt.commit("fsapt_setup")
            core.timer_off("SAPT(DFT): F-SAPT Localization (IBO)")

        if ckpt.pending("fsapt_elst"):
            core.timer_on("SAPT(DFT): F-SAPT Electrostatics")
            cache = jk_terms.felst(
                cache,
                elst["Elst10,r"] + extern_extern_IE,
                dimer_wfn,
                wfn_A,
                wfn_B,
                sapt_jk,
                True,
            )
            ckpt.commit("fsapt_elst")
            core.timer_off("SAPT(DFT): F-SAPT Electrostatics")

        if ckpt.pending("fsapt_exch"):
            core.timer_on("SAPT(DFT): F-SAPT Exchange")
            cache = jk_terms.fexch(
                cache,
                exch["Exch10(S^2)"],
                exch["Exch10"],
                dimer_wfn,
                wfn_A,
                wfn_B,
                sapt_jk,
                True,
            )
            ckpt.commit("fsapt_exch")
            core.timer_off("SAPT(DFT): F-SAPT Exchange")

        if ckpt.pending("fsapt_ind"):
            core.timer_on("SAPT(DFT): F-SAPT Induction")
            cache = jk_terms.find(cache, data, dimer_wfn, wfn_A, wfn_B, sapt_jk, True)
            ckpt.commit("fsapt_ind")
            core.timer_off("SAPT(DFT): F-SAPT Induction")

    elif do_fsapt and pending_fsapt_stage not in {None, "fsapt_final"}:
        if fsapt_type == "SAPTDFT":
            core.print_out(
                "\n  => Einsums is not available, switching to using FISAPT0 object for FSAPT <= \n\n"
            )

        # Build auxiliary basis for FISAPT
        aux_basis = core.BasisSet.build(
            dimer_wfn.molecule(),
            "DF_BASIS_MP2",
            core.get_option("DFMP2", "DF_BASIS_MP2"),
            "RIFIT",
            core.get_global_option("BASIS"),
        )

        # Partition CPHF induction with a short-lived HF-backed FISAPT object.
        # Destroy it before constructing the DFT-backed object so their DFHelper
        # scratch tensors cannot alias one another.
        induction_matrices = None
        if fsapt_induction_data is not None:
            hf_wfn_A, hf_wfn_B, hf_cache, hf_scalars = fsapt_induction_data
            core.timer_on("SAPT(DFT): F-SAPT Induction")
            hf_FISAPT_obj = saptdft_fisapt.setup_fisapt_object(
                dimer_wfn,
                hf_wfn_A,
                hf_wfn_B,
                hf_cache,
                hf_scalars,
                aux_basis,
                do_flocalize=True,
            )
            # felst() initializes the shared DFHelper state consumed by find();
            # fexch() mirrors the complete FISAPT setup used by an HF-functional run.
            hf_FISAPT_obj.felst()
            hf_FISAPT_obj.fexch()
            hf_FISAPT_obj.find()
            hf_matrices = hf_FISAPT_obj.matrices()
            induction_matrices = {
                key: hf_matrices[key].clone()
                for key in ("IndAB_AB", "IndBA_AB", "sIndAB_AB", "sIndBA_AB")
                if key in hf_matrices
            }
            del hf_FISAPT_obj
            core.timer_off("SAPT(DFT): F-SAPT Induction")

            # FISAPT setup expects these exchange-induction Coulomb
            # intermediates even though the DFT-backed find() is skipped.
            cache["J_P_A"] = hf_cache["J_P_A"]
            cache["J_P_B"] = hf_cache["J_P_B"]

        # Create single FISAPT object with do_flocalize=True to handle IBO
        # localization internally, unless a checkpoint already holds it.
        do_flocalize = ckpt.pending("fsapt_setup")
        if do_flocalize:
            core.timer_on("SAPT(DFT): F-SAPT Setup + Localization (IBO)")
        FISAPT_obj = saptdft_fisapt.setup_fisapt_object(
            dimer_wfn, wfn_A, wfn_B, cache, data, aux_basis, do_flocalize=do_flocalize
        )
        if do_flocalize:
            _cache_fisapt_localization_aliases(_absorb_fisapt_matrices(cache, FISAPT_obj))
            ckpt.commit("fsapt_setup")
            core.timer_off("SAPT(DFT): F-SAPT Setup + Localization (IBO)")

        if ckpt.pending("fsapt_elst"):
            core.timer_on("SAPT(DFT): F-SAPT Electrostatics")
            FISAPT_obj.felst()
            _absorb_fisapt_matrices(cache, FISAPT_obj)
            ckpt.commit("fsapt_elst")
            core.timer_off("SAPT(DFT): F-SAPT Electrostatics")

        if ckpt.pending("fsapt_exch"):
            core.timer_on("SAPT(DFT): F-SAPT Exchange")
            FISAPT_obj.fexch()
            _absorb_fisapt_matrices(cache, FISAPT_obj)
            ckpt.commit("fsapt_exch")
            core.timer_off("SAPT(DFT): F-SAPT Exchange")

        if ckpt.pending("fsapt_ind"):
            core.timer_on("SAPT(DFT): F-SAPT Induction")
            if induction_matrices is None:
                FISAPT_obj.find()
            else:
                FISAPT_obj.set_matrix(induction_matrices)
            _absorb_fisapt_matrices(cache, FISAPT_obj)
            ckpt.commit("fsapt_ind")
            core.timer_off("SAPT(DFT): F-SAPT Induction")
        if FISAPT_obj is not None:
            _cache_fisapt_localization_aliases(_absorb_fisapt_matrices(cache, FISAPT_obj))

    # Blow away JK object before doing MP2 for memory considerations
    if cleanup_jk and sapt_jk is not None:
        core.print_out("\n   => Finalizing SAPT JK object to free memory <= \n\n")
        sapt_jk.finalize()
        if sapt_jk_B is not None:
            sapt_jk_B.finalize()
        core.release_freed_memory()

    if do_disp and ckpt.pending("disp"):
        # Hybrid xc kernel check
        do_hybrid = core.get_option("SAPT", "SAPT_DFT_DO_HYBRID")
        is_x_hybrid = functional_value(wfn_B, "is_x_hybrid", False)
        is_x_lrc = functional_value(wfn_B, "is_x_lrc", False)
        hybrid_specified = core.has_option_changed("SAPT", "SAPT_DFT_DO_HYBRID")
        if is_x_lrc:
            if do_hybrid:
                if hybrid_specified:
                    raise ValidationError(
                        "SAPT(DFT): Hybrid xc kernel not yet implemented for range-separated funtionals."
                    )
                else:
                    core.print_out(
                        "Warning: Hybrid xc kernel not yet implemented for range-separated funtionals; hybrid kernel capability is turned off.\n"
                    )
            is_hybrid = False
        else:
            if do_hybrid:
                is_hybrid = is_x_hybrid
            else:
                is_hybrid = False

        # Dispersion
        core.timer_on("SAPT(DFT):disp")

        primary_basis = wfn_A.basisset()
        aux_basis = core.BasisSet.build(
            dimer_wfn.molecule(),
            "DF_BASIS_MP2",
            core.get_option("DFMP2", "DF_BASIS_MP2"),
            "RIFIT",
            core.get_global_option("BASIS"),
        )

        if do_dft:
            core.timer_on("FDDS disp")
            core.print_out("\n")
            x_alpha = functional_value(wfn_B, "x_alpha", 0.0)
            if not is_hybrid:
                x_alpha = 0.0
            fdds_disp = sapt_mp2.df_fdds_dispersion(
                primary_basis, aux_basis, cache, is_hybrid, x_alpha
            )
            data.update(fdds_disp)
            nfrozen_A = 0
            nfrozen_B = 0
            core.timer_off("FDDS disp")
        else:
            # this is where we actually need to figure out the number of
            # frozen-core orbitals
            # if SAPT_DFT_MP2_DISP_ALG == FISAPT, the code will not figure it
            # out on its own
            nfrozen_A = wfn_A.basisset().n_frozen_core(
                core.get_global_option("FREEZE_CORE"), wfn_A.molecule()
            )
            nfrozen_B = wfn_B.basisset().n_frozen_core(
                core.get_global_option("FREEZE_CORE"), wfn_B.molecule()
            )

        if not do_fsapt:
            core.timer_on("MP2 disp")
            if core.get_option("SAPT", "SAPT_DFT_MP2_DISP_ALG") == "FISAPT":
                mp2_disp = sapt_mp2.df_mp2_fisapt_dispersion(
                    wfn_A,
                    primary_basis,
                    aux_basis,
                    cache,
                    nfrozen_A,
                    nfrozen_B,
                    do_print=True,
                )
            else:
                mp2_disp = sapt_mp2.df_mp2_sapt_dispersion(
                    dimer_wfn,
                    wfn_A,
                    wfn_B,
                    primary_basis,
                    aux_basis,
                    cache,
                    do_print=True,
                )
            core.timer_off("MP2 disp")
            data.update(mp2_disp)

        # Exchange-dispersion scaling
        if do_dft:
            exch_disp_scheme = core.get_option(
                "SAPT", "SAPT_DFT_EXCH_DISP_SCALE_SCHEME"
            )
            core.print_out("    %-33s % s\n" % ("Scaling Scheme", exch_disp_scheme))
            if exch_disp_scheme == "NONE":
                data["Exch-Disp20,r"] = data["Exch-Disp20,u"]
            elif exch_disp_scheme == "FIXED":
                exch_disp_scale = core.get_option(
                    "SAPT", "SAPT_DFT_EXCH_DISP_FIXED_SCALE"
                )
                core.print_out(
                    "    %-28s % 10.3f\n" % ("Scaling Factor", exch_disp_scale)
                )
                data["Exch-Disp20,r"] = exch_disp_scale * data["Exch-Disp20,u"]
            elif exch_disp_scheme == "DISP":
                exch_disp_scale = data["Disp20"] / data["Disp20,u"]
                data["Exch-Disp20,r"] = exch_disp_scale * data["Exch-Disp20,u"]
            if exch_disp_scheme != "NONE":
                core.print_out(
                    print_sapt_var(
                        "Est. Exch-Disp20,r", data["Exch-Disp20,r"], short=True
                    )
                    + "\n"
                )

        ckpt.commit("disp")
        core.timer_off("SAPT(DFT):disp")

    # Now do F-SAPT on dispersion if requested
    if do_fsapt and fsapt_type == "SAPTDFT" and use_einsums:
        # Because dispersion is defined differently between SAPT0 (E_disp20 =
        # -4\sigma_{abrs} |(ar|bs)|^2 / (epsilon_a + epsilon_b)) and SAPT(DFT)
        # with FDDS dispersion, we will only implement F-SAPT for the SAPT0
        # case. Practically speaking, -D3/-D4 dispersion is preferred for
        # SAPT(DFT) due to computational costs, so those are the only currently
        # supported dispersion method for F-SAPT in SAPT(DFT). Hence,
        # FSAPT_DISP_AB will be set to zero if SAPT(DFT) is requested with FDDS
        # dispersion with DO_FSAPT.

        if do_disp and ckpt.pending("fsapt_disp"):
            core.timer_on("SAPT(DFT): F-SAPT Dispersion")
            cache = jk_terms.fdisp0(
                cache, data, dimer_wfn, wfn_A, wfn_B, sapt_jk, do_print=True
            )
            data["Exch-Disp20,u"] = cache["Exch-Disp20,u"]
            data["Disp20,u"] = cache["Disp20,u"]
            ckpt.commit("fsapt_disp")
            core.timer_off("SAPT(DFT): F-SAPT Dispersion")

    elif do_fsapt and do_disp:
        if ckpt.pending("fsapt_disp"):
            core.timer_on("SAPT(DFT): F-SAPT Dispersion")
            FISAPT_obj.fdisp()
            core.timer_off("SAPT(DFT): F-SAPT Dispersion")
            FISAPT_obj.fdrop(external_potentials)
            scalars = FISAPT_obj.scalars()
            data["Exch-Disp20,u"] = scalars["Exch-Disp20"]
            data["Disp20,u"] = scalars["Disp20"]
            _absorb_fisapt_matrices(cache, FISAPT_obj)
            ckpt.commit("fsapt_disp")
    elif do_fsapt and fsapt_type == "FISAPT" and FISAPT_obj is not None:
        _absorb_fisapt_matrices(cache, FISAPT_obj)
        FISAPT_obj.fdrop(external_potentials)

    sapt_dft_D4_IE = core.get_option("SAPT", "SAPT_DFT_D4_IE")
    sapt_dft_D3_IE = core.get_option("SAPT", "SAPT_DFT_D3_IE")
    if do_fsapt and (
        sapt_dft_D4_IE or sapt_dft_D3_IE
    ):
        cache["FSAPT_EMPIRICAL_DISP"] = core.Matrix.from_array(
            data["FSAPT_EMPIRICAL_DISP"]
        )

    # Print out final data
    core.print_out("\n")
    core.print_out(
        print_sapt_dft_summary(
            data,
            "SAPT(DFT)",
            dimer_wfn=dimer_wfn,
            do_dft=do_dft,
            do_disp=do_disp,
            do_delta_dft=do_delta_dft,
            induction_type=induction_type,
        )
    )

    # because FISAPT_obj drop sets core variables, avoid setting them twice
    if core.get_option("FISAPT", "FISAPT_FSAPT_FILEPATH").upper() != "NONE" and do_fsapt:
        FISAPT_obj = saptdft_fisapt.drop_saptdft_variables(
            dimer_wfn, wfn_A, wfn_B, cache, data
        )
    elif do_fsapt:
        def _set_fsapt_var(label, value):
            core.set_variable(label, value)
            dimer_wfn.set_variable(label, value)

        _set_fsapt_var("FSAPT_QA", cache["Qocc0A"])
        _set_fsapt_var("FSAPT_QB", cache["Qocc0B"])
        _set_fsapt_var("FSAPT_ELST_AB", cache["Elst_AB"])
        _set_fsapt_var(
            "FSAPT_AB_SIZE", np.array(cache["Elst_AB"].np.shape).reshape(1, -1)
        )
        _set_fsapt_var("FSAPT_EXCH_AB", cache["Exch_AB"])
        _set_fsapt_var("FSAPT_INDAB_AB", cache["IndAB_AB"])
        _set_fsapt_var("FSAPT_INDBA_AB", cache["IndBA_AB"])
        if sapt_dft_D4_IE or sapt_dft_D3_IE:
            disp_ab = cache["Elst_AB"].clone()
            disp_ab.zero()
            _set_fsapt_var("FSAPT_DISP_AB", disp_ab)
            _set_fsapt_var("FSAPT_EMPIRICAL_DISP", cache["FSAPT_EMPIRICAL_DISP"])
        else:
            disp_ab = cache.get("Disp_AB")
            if disp_ab is None:
                disp_ab = cache["Elst_AB"].clone()
                disp_ab.zero()
                cache["Disp_AB"] = disp_ab
            _set_fsapt_var("FSAPT_DISP_AB", disp_ab)

    # F-SAPT is finished either way above; the "final" stage depends on this one,
    # so it has to be recorded on the drop-to-file path too.
    if do_fsapt and ckpt.pending("fsapt_final"):
        ckpt.commit("fsapt_final")
    return data


def run_sf_sapt(name: str, **kwargs) -> core.Wavefunction:
    """Run the spin-flip SAPT (SF-SAPT) interaction energy calculation.

    Top-level driver for SF-SAPT calculations. Prepares the dimer
    molecule, runs monomer SCF computations, and computes the SF-SAPT
    interaction energy components.

    Parameters
    ----------
    name : str
        Name of the SF-SAPT method (e.g., ``'sf-sapt'``).
    **kwargs
        Additional keyword arguments. Recognized keys include ``ref_wfn``
        (reference wavefunction), ``molecule`` (molecular system).

    Returns
    -------
    core.Wavefunction
        The dimer wavefunction with SF-SAPT results stored as variables.
    """
    optstash = p4util.OptionsState(
        ["SCF_TYPE"],
        ["SCF", "REFERENCE"],
        ["SCF", "DFT_GRAC_SHIFT"],
        ["SCF", "SAVE_JK"],
    )

    core.tstart()

    # Alter default algorithm
    if not core.has_global_option_changed("SCF_TYPE"):
        core.set_global_option("SCF_TYPE", "DF")

    core.prepare_options_for_module("SAPT")

    # Get the molecule of interest
    ref_wfn = kwargs.get("ref_wfn", None)
    if ref_wfn is None:
        sapt_dimer = kwargs.pop("molecule", core.get_active_molecule())
    else:
        core.print_out(
            'Warning! SAPT argument "ref_wfn" is only able to use molecule information.'
        )
        sapt_dimer = ref_wfn.molecule()

    sapt_dimer, monomerA, monomerB = proc_util.prepare_sapt_molecule(
        sapt_dimer, "dimer"
    )

    # Print out the title and some information
    core.print_out("\n")
    core.print_out(
        "         ---------------------------------------------------------\n"
    )
    core.print_out("         " + "Spin-Flip SAPT Procedure".center(58) + "\n")
    core.print_out("\n")
    core.print_out(
        "         " + "by Daniel G. A. Smith and Konrad Patkowski".center(58) + "\n"
    )
    core.print_out(
        "         ---------------------------------------------------------\n"
    )
    core.print_out("\n")

    core.print_out("  ==> Algorithm <==\n\n")
    core.print_out(
        "   JK Algorithm            %12s\n" % core.get_option("SCF", "SCF_TYPE")
    )
    core.print_out("\n")
    core.print_out("   Required computations:\n")
    core.print_out("     HF  (Monomer A)\n")
    core.print_out("     HF  (Monomer B)\n")
    core.print_out("\n")

    if core.get_option("SCF", "REFERENCE") != "ROHF":
        raise ValidationError(
            "Spin-Flip SAPT currently only supports restricted open-shell references."
        )

    # Run the two monomer computations
    core.IO.set_default_namespace("dimer")

    core.set_global_option("DF_INTS_IO", "SAVE")

    # Compute dimer wavefunction
    wfn_A = scf_helper(
        "SCF", molecule=monomerA, banner="SF-SAPT: HF Monomer A", **kwargs
    )

    core.set_global_option("SAVE_JK", True)
    wfn_B = scf_helper(
        "SCF", molecule=monomerB, banner="SF-SAPT: HF Monomer B", **kwargs
    )
    sapt_jk = wfn_B.jk()
    core.set_global_option("SAVE_JK", False)
    core.print_out("\n")
    core.print_out(
        "         ---------------------------------------------------------\n"
    )
    core.print_out(
        "         " + "Spin-Flip SAPT Exchange and Electrostatics".center(58) + "\n"
    )
    core.print_out("\n")
    core.print_out(
        "         " + "by Daniel G. A. Smith and Konrad Patkowski".center(58) + "\n"
    )
    core.print_out(
        "         ---------------------------------------------------------\n"
    )
    core.print_out("\n")

    sf_data = sapt_sf_terms.compute_sapt_sf(sapt_dimer, sapt_jk, wfn_A, wfn_B)

    # Print the results
    core.print_out("   Spin-Flip SAPT Results\n")
    core.print_out("  " + "-" * 103 + "\n")

    for key, value in sf_data.items():
        value = sf_data[key]
        print_vals = (
            key,
            value * 1000,
            value * constants.hartree2kcalmol,
            value * constants.hartree2kJmol,
        )
        string = (
            "    %-26s % 15.8f [mEh] % 15.8f [kcal/mol] % 15.8f [kJ/mol]\n" % print_vals
        )
        core.print_out(string)
    core.print_out("  " + "-" * 103 + "\n\n")

    dimer_wfn = core.Wavefunction.build(sapt_dimer, wfn_A.basisset())

    # Set variables
    psivar_tanslator = {
        "Elst10": "SAPT ELST ENERGY",
        "Exch10(S^2) [diagonal]": "SAPT EXCH10(S^2),DIAGONAL ENERGY",
        "Exch10(S^2) [off-diagonal]": "SAPT EXCH10(S^2),OFF-DIAGONAL ENERGY",
        "Exch10(S^2) [highspin]": "SAPT EXCH10(S^2),HIGHSPIN ENERGY",
    }

    for k, v in sf_data.items():
        psi_k = psivar_tanslator[k]

        dimer_wfn.set_variable(psi_k, v)
        core.set_variable(psi_k, v)

    # Copy over highspin
    core.set_variable("SAPT EXCH ENERGY", sf_data["Exch10(S^2) [highspin]"])

    core.tstop()
    optstash.restore()

    return dimer_wfn
