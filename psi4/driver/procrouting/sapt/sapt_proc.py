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
)
from .sapt_util import (
    print_sapt_dft_summary,
    print_sapt_hf_induction_summary,
    print_sapt_hf_summary,
    print_sapt_var,
)
import qcelemental as qcel
from ...p4util.exceptions import ConvergenceError


# Only export the run_ scripts
__all__ = ["run_sapt_dft", "sapt_dft", "run_sf_sapt"]

# A user-supplied SAPT_DFT_GRAC_SHIFT_A/B at or below this value switches GRAC
# off for that monomer. A real shift (IP + HOMO) is rejected by
# compute_GRAC_shift unless -1 < shift < 1, so the sentinel cannot collide
# with a physical value.
GRAC_SHIFT_DISABLED = -1.0


def run_sapt_dft(name: str, **kwargs) -> core.Wavefunction:
    """Run SAPT(DFT) while restoring all driver-managed options and timers."""
    optstash = p4util.OptionsState(
        ["DF_INTS_IO"],
        ["SCF_TYPE"],
        ["SCF", "REFERENCE"],
        ["SCF", "DFT_GRAC_SHIFT"],
        ["SCF", "SAVE_JK"],
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
    # A shift <= -1 is a request to run that monomer without GRAC. Replace it
    # by 0 here, before anything can forward it to DFT_GRAC_SHIFT: a zero
    # shift is never applied to the monomer SCF.
    grac_disabled_A = (
        core.has_option_changed("SAPT", "SAPT_DFT_GRAC_SHIFT_A")
        and mon_a_shift <= GRAC_SHIFT_DISABLED
    )
    grac_disabled_B = (
        core.has_option_changed("SAPT", "SAPT_DFT_GRAC_SHIFT_B")
        and mon_b_shift <= GRAC_SHIFT_DISABLED
    )
    if grac_disabled_A:
        mon_a_shift = 0.0
    if grac_disabled_B:
        mon_b_shift = 0.0
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
    do_dft = sapt_dft_functional != "HF"
    if shift_only and grac_compute == "NONE":
        raise ValidationError("SAPT_DFT_GRAC_SHIFT_ONLY=true contradicts SAPT_DFT_GRAC_COMPUTE=NONE.")
    if shift_only and not do_dft:
        raise ValidationError("SAPT_DFT_GRAC_SHIFT_ONLY requires a non-HF SAPT_DFT_FUNCTIONAL.")
    if not do_dft:
        do_mon_grac_shift_A = do_mon_grac_shift_B = False

    # CPHF needs the HF segment for the SAPT0 induction terms, even without delta HF.
    run_hf_segment = do_delta_hf or (induction_type == "CPHF" and do_dft)
    hf_segment_label = "delta HF" if do_delta_hf else "SAPT0 induction"

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

    core.print_out("\n")
    core.print_out("   Beginning setup computations\n")

    if do_mon_grac_shift_A:
        core.print_out("     GRAC (Monomer A)\n")
        mon_a_shift = compute_GRAC_shift(
            monomerA_mon_only_bf,
            grac_compute,
            "A",
            results=data,
            external_potentials=external_potentials.get("A") if grac_use_ext_pot else None,
        )
    if do_mon_grac_shift_B:
        core.print_out("     GRAC (Monomer B)\n")
        mon_b_shift = compute_GRAC_shift(
            monomerB_mon_only_bf,
            grac_compute,
            "B",
            results=data,
            external_potentials=external_potentials.get("B") if grac_use_ext_pot else None,
        )

    core.set_variable("SAPT DFT GRAC SHIFT A", mon_a_shift)  # P::e SAPT
    core.set_variable("SAPT DFT GRAC SHIFT B", mon_b_shift)  # P::e SAPT
    data["SAPT DFT GRAC SHIFT A"] = mon_a_shift
    data["SAPT DFT GRAC SHIFT B"] = mon_b_shift
    core.print_out("\n  ==> SAPT(DFT) GRAC Shifts <==\n\n")
    core.print_out("   Monomer   E(monomer) [Eh]   E(ionized) [Eh]     HOMO [Eh]     IP [Eh]   GRAC shift [Eh]\n")
    for label, shift, computed, disabled in (
        ("A", mon_a_shift, do_mon_grac_shift_A, grac_disabled_A),
        ("B", mon_b_shift, do_mon_grac_shift_B, grac_disabled_B),
    ):
        if computed:
            values = [data[f"SAPT DFT GRAC {quantity} {label}"] for quantity in
                      ("MONOMER ENERGY", "IONIZED MONOMER ENERGY", "HOMO", "IP")]
            core.print_out(f"         {label} {values[0]:18.8f} {values[1]:18.8f} {values[2]:13.8f} {values[3]:11.8f} {shift:17.8f}\n")
        else:
            core.print_out(f"         {label} {'':63s} {shift:17.8f}\n")
            if disabled and do_dft:
                core.print_out(
                    f"   Monomer {label} GRAC disabled by SAPT_DFT_GRAC_SHIFT_{label} <= {GRAC_SHIFT_DISABLED:.0f};"
                    " its DFT runs without an asymptotic correction.\n"
                )
            elif do_dft:
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
        return wfn
    core.print_out("\n")
    # Save integrals
    # We want to try to re-use itegrals for the dimer and monomer SCF's. If we
    # are using Disk based DF (DISK_DF) then we can use the DF_INTS_IO option.
    # MemDF does not know about this option but setting it will be harmless
    # there.
    core.set_global_option("DF_INTS_IO", "SAVE")

    # Compute dimer wavefunction
    hf_wfn_dimer = None

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
        hf_data = {}

        core.set_local_option("SCF", "SAVE_JK", True)
        if do_ext_potential:
            kwargs["external_potentials"]["C"] = (
                construct_external_potential_in_field_C(
                    [ext_pot_C, ext_pot_A_not_in_C, ext_pot_B_not_in_C]
                )
            )
        hf_wfn_dimer = scf_helper(
            "SCF", molecule=sapt_dimer, banner=f"SAPT(DFT): {hf_segment_label} Dimer", **kwargs
        )
        if do_ext_potential:
            kwargs.pop("external_potentials")
        hf_data["HF DIMER"] = core.variable("CURRENT ENERGY")
        core.timer_off("SAPT(DFT):Dimer SCF")

        core.timer_on("SAPT(DFT):Monomer A SCF")

        jk_obj = hf_wfn_dimer.jk()
        if do_ext_potential and (ext_pot_A is not None or ext_pot_C is not None):
            kwargs["external_potentials"] = {}
            kwargs["external_potentials"]["C"] = (
                construct_external_potential_in_field_C([ext_pot_C, ext_pot_A_not_in_C])
            )
        hf_wfn_A = scf_helper(
            "SCF",
            molecule=monomerA,
            banner=f"SAPT(DFT): {hf_segment_label} Monomer A",
            jk=jk_obj,
            **kwargs,
        )
        if do_ext_potential and kwargs.get("external_potentials"):
            kwargs.pop("external_potentials")
        hf_data["HF MONOMER A"] = core.variable("CURRENT ENERGY")
        core.timer_off("SAPT(DFT):Monomer A SCF")

        core.timer_on("SAPT(DFT):Monomer B SCF")
        # core.IO.change_file_namespace(97, "monomerA", "monomerB")

        if do_ext_potential and (ext_pot_B is not None or ext_pot_C is not None):
            kwargs["external_potentials"] = {}
            kwargs["external_potentials"]["C"] = (
                construct_external_potential_in_field_C([ext_pot_C, ext_pot_B_not_in_C])
            )
        hf_wfn_B = scf_helper(
            "SCF",
            molecule=monomerB,
            banner=f"SAPT(DFT): {hf_segment_label} Monomer B",
            jk=jk_obj,
            **kwargs,
        )
        hf_data["HF MONOMER B"] = core.variable("CURRENT ENERGY")
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
            sapt_jk = hf_wfn_B.jk()
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

            # Build the SAPT0 cache needed for electrostatics and exchange.
            hf_cache = jk_terms.build_sapt_jk_cache(
                hf_wfn_dimer,
                hf_wfn_A,
                hf_wfn_B,
                sapt_jk,
                True,
                external_potentials=kwargs.get("external_potentials", None),
            )

            # Electrostatics
            core.timer_on("SAPT(HF):elst")
            elst, extern_extern_IE = jk_terms.electrostatics(hf_cache, True)
            hf_data["extern_extern_IE"] = extern_extern_IE
            hf_data.update(elst)
            core.timer_off("SAPT(HF):elst")

            # Exchange
            core.timer_on("SAPT(HF):exch")
            exch = jk_terms.exchange(hf_cache, sapt_jk, True)
            hf_data.update(exch)
            core.timer_off("SAPT(HF):exch")

            if induction_type != "NONE":
                core.timer_on("SAPT(HF):ind")
                ind = jk_terms.induction(
                    hf_cache,
                    sapt_jk,
                    True,
                    maxiter=core.get_option("SAPT", "MAXITER"),
                    conv=core.get_option("SAPT", "CPHF_R_CONVERGENCE"),
                    Sinf=core.get_option("SAPT", "DO_IND_EXCH_SINF"),
                )
                hf_data.update(ind)
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
            sapt_jk.finalize()

            del hf_wfn_A, hf_wfn_B, sapt_jk

        else:
            wfn_A = hf_wfn_A
            wfn_B = hf_wfn_B
            data["DFT MONOMER A"] = hf_data["HF MONOMER A"]
            data["DFT MONOMER B"] = hf_data["HF MONOMER B"]
            dhf_value = (
                hf_data["HF DIMER"] - hf_data["HF MONOMER A"] - hf_data["HF MONOMER B"]
            )
            data["DHF VALUE"] = dhf_value

    if hf_wfn_dimer is None:
        dimer_wfn = core.Wavefunction.build(sapt_dimer, core.get_global_option("BASIS"))
    else:
        dimer_wfn = hf_wfn_dimer

    if do_dft or not do_delta_hf:
        # Set the primary functional
        core.set_local_option("SCF", "REFERENCE", "RKS")

        # Compute Monomer A wavefunction
        core.timer_on("SAPT(DFT): Monomer A DFT")
        if mon_a_shift:
            core.set_global_option("DFT_GRAC_SHIFT", mon_a_shift)

        core.set_global_option("SAVE_JK", True)
        if do_ext_potential and (ext_pot_A is not None or ext_pot_C is not None):
            kwargs["external_potentials"] = {}
            kwargs["external_potentials"]["C"] = (
                construct_external_potential_in_field_C([ext_pot_C, ext_pot_A_not_in_C])
            )
        elif do_ext_potential:
            kwargs["external_potentials"] = {}

        wfn_A = scf_helper(
            sapt_dft_functional,
            post_scf=False,
            molecule=monomerA,
            banner="SAPT(DFT): DFT Monomer A",
            **kwargs,
        )
        if do_ext_potential and kwargs.get("external_potentials"):
            kwargs.pop("external_potentials")
        data["DFT MONOMERA"] = core.variable("CURRENT ENERGY")

        core.set_global_option("DFT_GRAC_SHIFT", 0.0)
        core.timer_off("SAPT(DFT): Monomer A DFT")

        # Compute Monomer B wavefunction
        core.timer_on("SAPT(DFT): Monomer B DFT")

        if mon_b_shift:
            core.set_global_option("DFT_GRAC_SHIFT", mon_b_shift)

        core.set_global_option("SAVE_JK", True)
        if do_ext_potential and (ext_pot_B is not None or ext_pot_C is not None):
            kwargs["external_potentials"] = {}
            kwargs["external_potentials"]["C"] = (
                construct_external_potential_in_field_C([ext_pot_C, ext_pot_B_not_in_C])
            )
        wfn_B = scf_helper(
            sapt_dft_functional,
            post_scf=False,
            molecule=monomerB,
            banner="SAPT(DFT): DFT Monomer B",
            jk=wfn_A.jk(),
            **kwargs,
        )
        data["DFT MONOMERB"] = core.variable("CURRENT ENERGY")
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

    # Save JK object
    sapt_jk = wfn_B.jk()
    wfn_A.set_jk(sapt_jk)

    core.set_global_option("SAVE_JK", False)
    core.set_global_option("DFT_GRAC_SHIFT", 0.0)

    # Write out header
    scf_alg = core.get_global_option("SCF_TYPE")
    sapt_dft_header(
        sapt_dft_functional, mon_a_shift, mon_b_shift, bool(do_delta_hf), scf_alg
    )

    # Call SAPT(DFT)
    sapt_jk = wfn_B.jk()
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
    )

    # Copy data back into globals
    for k, v in data.items():
        core.set_variable(k, v)
        dimer_wfn.set_variable(k, v)

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

    Returns
    -------
    float
        The GRAC shift value in Hartree.
    """
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

    monomer_label = label
    label = f"Monomer {label}"
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
        for options in grac_options:
            for key, val in options.items():
                core.set_local_option("SCF", key, val)
            core.set_local_option("SCF", "ORBITAL_OPTIMIZER_PACKAGE", "INTERNAL")
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
            occ_given = wfn_given.epsilon_a_subset(basis="SO", subset="OCC").to_array(
                dense=True
            )
            HOMO = np.amax(occ_given)

            E_given = wfn_given.energy()
            E_cation = wfn_cation.energy()
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
        def set_variable(key, value):
            core.set_variable(key, value)
            wfn_given.set_variable(key, value)
            if results is not None:
                results[key] = value

        if monomer_label == "A":
            set_variable("SAPT DFT GRAC MONOMER ENERGY A", E_given)  # P::e SAPT
            set_variable("SAPT DFT GRAC IONIZED MONOMER ENERGY A", E_cation)  # P::e SAPT
            set_variable("SAPT DFT GRAC HOMO A", HOMO)  # P::e SAPT
            set_variable("SAPT DFT GRAC IP A", E_cation - E_given)  # P::e SAPT
        elif monomer_label == "B":
            set_variable("SAPT DFT GRAC MONOMER ENERGY B", E_given)  # P::e SAPT
            set_variable("SAPT DFT GRAC IONIZED MONOMER ENERGY B", E_cation)  # P::e SAPT
            set_variable("SAPT DFT GRAC HOMO B", HOMO)  # P::e SAPT
            set_variable("SAPT DFT GRAC IP B", E_cation - E_given)  # P::e SAPT
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
    if induction_type == "CPHF" and do_dft and "Ind20,r" not in data:
        raise ValidationError(
            "SAPT_DFT_INDUCTION_TYPE=CPHF reuses the SAPT0 induction terms computed by the "
            "SAPT(DFT) delta HF segment, which sapt_dft() does not run. Call "
            "energy('sapt(dft)') instead of sapt_dft() directly, or supply the SAPT0 "
            "induction terms in `data`."
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

    core.timer_on("SAPT(DFT):Build JK")
    if print_header:
        sapt_dft_header()

    if sapt_jk is None:
        core.print_out("\n   => Building SAPT JK object <= \n\n")
        sapt_jk = core.JK.build(dimer_wfn.basisset())
        sapt_jk.set_do_J(True)
        sapt_jk.set_do_K(True)
        if wfn_A.functional().is_x_lrc():
            sapt_jk.set_do_wK(True)
            sapt_jk.set_omega(wfn_A.functional().x_omega())
        sapt_jk.initialize()
        sapt_jk.print_header()
        if wfn_B.functional().is_x_lrc() and (
            wfn_A.functional().x_omega() != wfn_B.functional().x_omega()
        ):
            core.print_out("   => Monomer B: Building SAPT JK object <= \n\n")
            core.print_out("      Reason: MonomerA Omega != MonomerB Omega\n\n")
            sapt_jk_B = core.JK.build(dimer_wfn.basisset())
            sapt_jk_B.set_do_J(True)
            sapt_jk_B.set_do_K(True)
            sapt_jk_B.set_do_wK(True)
            sapt_jk_B.set_omega(wfn_B.functional().x_omega())
            sapt_jk_B.initialize()
            sapt_jk_B.print_header()

    else:
        sapt_jk.set_do_K(True)

    sapt_jk.set_do_J(True)
    sapt_jk.set_do_K(True)

    if wfn_A.functional().is_x_lrc():
        sapt_jk.set_do_wK(True)
        sapt_jk.set_omega(wfn_A.functional().x_omega())

    # Build SAPT cache
    jk_terms = sapt_jk_terms
    sapt_mp2 = sapt_mp2_terms
    cache = jk_terms.build_sapt_jk_cache(
        dimer_wfn, wfn_A, wfn_B, sapt_jk, True, external_potentials
    )
    core.timer_off("SAPT(DFT):Build JK")

    # Electrostatics
    core.timer_on("SAPT(DFT):elst")
    elst, extern_extern_IE = jk_terms.electrostatics(cache, True)
    data["extern_extern_IE"] = extern_extern_IE
    data.update(elst)
    core.timer_off("SAPT(DFT):elst")

    # Exchange
    core.timer_on("SAPT(DFT):exch")
    exch = jk_terms.exchange(cache, sapt_jk, True)
    data.update(exch)
    core.timer_off("SAPT(DFT):exch")

    # Induction
    core.timer_on("SAPT(DFT):ind")
    if induction_type == "CPKS" or (induction_type == "CPHF" and not do_dft):
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
    else:
        core.print_out(f"\n   SAPT(DFT) induction skipped ({induction_type}).\n")

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

    core.timer_off("SAPT(DFT):ind")

    # Blow away JK object before doing MP2 for memory considerations
    if cleanup_jk:
        core.print_out("\n   => Finalizing SAPT JK object to free memory <= \n\n")
        sapt_jk.finalize()

    # Hybrid xc kernel check
    do_hybrid = core.get_option("SAPT", "SAPT_DFT_DO_HYBRID")
    is_x_hybrid = wfn_B.functional().is_x_hybrid()
    is_x_lrc = wfn_B.functional().is_x_lrc()
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
        x_alpha = wfn_B.functional().x_alpha()
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
        # this is where we actually need to figure out the number of frozen-core orbitals
        # if SAPT_DFT_MP2_DISP_ALG == FISAPT, the code will not figure it out on its own
        nfrozen_A = wfn_A.basisset().n_frozen_core(
            core.get_global_option("FREEZE_CORE"), wfn_A.molecule()
        )
        nfrozen_B = wfn_B.basisset().n_frozen_core(
            core.get_global_option("FREEZE_CORE"), wfn_B.molecule()
        )

    core.timer_on("MP2 disp")
    if core.get_option("SAPT", "SAPT_DFT_MP2_DISP_ALG") == "FISAPT":
        mp2_disp = sapt_mp2.df_mp2_fisapt_dispersion(
            wfn_A, primary_basis, aux_basis, cache, nfrozen_A, nfrozen_B, do_print=True
        )
    else:
        mp2_disp = sapt_mp2.df_mp2_sapt_dispersion(
            dimer_wfn, wfn_A, wfn_B, primary_basis, aux_basis, cache, do_print=True
        )
    data.update(mp2_disp)

    # Exchange-dispersion scaling
    if do_dft:
        exch_disp_scheme = core.get_option(
            "SAPT", "SAPT_DFT_EXCH_DISP_SCALE_SCHEME")
        core.print_out("    %-33s % s\n" %
                       ("Scaling Scheme", exch_disp_scheme))
        if exch_disp_scheme == "NONE":
            data["Exch-Disp20,r"] = data["Exch-Disp20,u"]
        elif exch_disp_scheme == "FIXED":
            exch_disp_scale = core.get_option(
                "SAPT", "SAPT_DFT_EXCH_DISP_FIXED_SCALE")
            core.print_out("    %-28s % 10.3f\n" %
                           ("Scaling Factor", exch_disp_scale))
            data["Exch-Disp20,r"] = exch_disp_scale * data["Exch-Disp20,u"]
        elif exch_disp_scheme == "DISP":
            exch_disp_scale = data["Disp20"] / data["Disp20,u"]
            data["Exch-Disp20,r"] = exch_disp_scale * data["Exch-Disp20,u"]
        if exch_disp_scheme != "NONE":
            core.print_out(
                print_sapt_var("Est. Exch-Disp20,r",
                               data["Exch-Disp20,r"], short=True)
                + "\n"
            )

    core.timer_off("MP2 disp")
    core.timer_off("SAPT(DFT):disp")

    # Print out final data
    core.print_out("\n")
    core.print_out(
        print_sapt_dft_summary(
            data,
            "SAPT(DFT)",
            dimer_wfn=dimer_wfn,
            do_dft=do_dft,
            induction_type=induction_type,
        )
    )

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
