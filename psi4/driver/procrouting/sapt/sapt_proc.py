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
    saptdft_fisapt,
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
    fsapt_type = core.get_option("SAPT", "SAPT_DFT_DO_FSAPT").upper()
    do_fsapt = fsapt_type != "NONE"
    if induction_type == "NONE" and do_fsapt:
        raise ValidationError("F-SAPT requires induction; SAPT_DFT_INDUCTION_TYPE=NONE is unavailable.")
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

    # CPHF needs the HF segment for the SAPT0 induction terms, even without delta HF.
    run_hf_segment = do_delta_hf or (induction_type == "CPHF" and do_dft)
    hf_segment_label = "delta HF" if do_delta_hf else "SAPT0 induction"

    # SAPT(DFT) FDDS dispersion has no fragment partition, so F-SAPT is
    # limited to SAPT(HF), whose dispersion FISAPT partitions directly.
    if do_fsapt and do_dft:
        raise ValidationError(
            "FSAPT(DFT) currently only supported with SAPT_DFT_FUNCTIONAL=HF; "
            "FDDS dispersion has no F-SAPT partition."
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
        monomerA_mon_only_bf = sapt_dimer.extract_subsets(1)
        monomerB_mon_only_bf = sapt_dimer.extract_subsets(2)

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
    core.print_out("   Monomer A GRAC Shift    %12.6f\n" % mon_a_shift)
    core.print_out("   Monomer B GRAC Shift    %12.6f\n" % mon_b_shift)
    # fmt: off
    core.print_out("   Delta HF                %12s\n" % ("True" if do_delta_hf else "False"))
    core.print_out("   Induction Type          %12s\n" % induction_type)
    core.print_out("   JK Algorithm            %12s\n" % core.get_global_option("SCF_TYPE"))
    # fmt: on
    core.print_out("\n")
    core.print_out("   Required computations:\n")
    if run_hf_segment:
        core.print_out("     HF   (Dimer)\n")
        core.print_out("     HF   (Monomer A)\n")
        core.print_out("     HF   (Monomer B)\n")
    if do_dft:
        core.print_out("     DFT  (Monomer A)\n")
        core.print_out("     DFT  (Monomer B)\n")
    if do_mon_grac_shift_A:
        core.print_out("     GRAC (Monomer A)\n")
        mon_a_shift = compute_GRAC_shift(
            monomerA_mon_only_bf,
            grac_compute,
            "Monomer A",
        )
    if do_mon_grac_shift_B:
        core.print_out("     GRAC (Monomer B)\n")
        mon_b_shift = compute_GRAC_shift(
            monomerB_mon_only_bf,
            grac_compute,
            "Monomer B",
        )

    core.set_variable("SAPT DFT GRAC SHIFT A", mon_a_shift)  # P::e SAPT
    core.set_variable("SAPT DFT GRAC SHIFT B", mon_b_shift)  # P::e SAPT
    core.print_out("\n")

    # Save integrals
    # We want to try to re-use itegrals for the dimer and monomer SCF's. If we
    # are using Disk based DF (DISK_DF) then we can use the DF_INTS_IO option.
    # MemDF does not know about this option but setting it will be harmless
    # there.
    core.set_global_option("DF_INTS_IO", "SAVE")

    # Compute dimer wavefunction
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
    if run_hf_segment:
        core.set_global_option("DF_INTS_IO", "SAVE")
        core.timer_on("SAPT(DFT):Dimer SCF")
        hf_data = {}

        core.set_local_option("SCF", "SAVE_JK", True)
        if do_ext_potential:
            kwargs["external_potentials"]["C"] = (
                construct_external_potential_in_field_C(
                    [ext_pot_C, ext_pot_A, ext_pot_B]
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
                construct_external_potential_in_field_C([ext_pot_C, ext_pot_A])
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
                construct_external_potential_in_field_C([ext_pot_C, ext_pot_B])
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
                if fsapt_type == "FISAPT":
                    # Keep the J_P matrices independent of the HF JK object,
                    # which is finalized below, so the later FISAPT::find()
                    # cannot dereference released storage.
                    hf_cache["J_P_A"] = hf_cache["J_P_A"].clone()
                    hf_cache["J_P_B"] = hf_cache["J_P_B"].clone()

                    # Retain the SAPT0 cache and wavefunctions so FISAPT::find()
                    # can partition the same HF induction used by CPHF.
                    fsapt_induction_data = (
                        hf_wfn_A,
                        hf_wfn_B,
                        hf_cache,
                        hf_data.copy(),
                    )
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

    if hf_wfn_dimer is None and not do_fsapt:
        dimer_wfn = core.Wavefunction.build(sapt_dimer, core.get_global_option("BASIS"))
    # If we did not compute HF wavefunction, we still need orbital coefficients
    # for IBOLocalizer2
    elif hf_wfn_dimer is None and do_fsapt:
        dimer_wfn = scf_helper(
            "SCF",
            molecule=sapt_dimer,
            banner="SAPT(DFT): Dimer for Localization",
            **kwargs,
        )
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
                construct_external_potential_in_field_C([ext_pot_C, ext_pot_A])
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
                construct_external_potential_in_field_C([ext_pot_C, ext_pot_B])
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
        fsapt_induction_data=fsapt_induction_data,
    )

    # Copy data back into globals
    for k, v in data.items():
        core.set_variable(k, v)
        dimer_wfn.set_variable(k, v)

    return dimer_wfn


def construct_external_potential_in_field_C(potentials):
    """Concatenate point/diffuse rows; sum opaque matrix operators.

    Rows are never dropped here. The field the dimer sees has to equal the
    sum of the fields the monomers see or the embedding contribution fails
    to cancel in the SAPT decomposition, and two equal rows mean two equal
    charges.
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
    ],
}


def compute_GRAC_shift(molecule, sapt_dft_grac_convergence_tier, label):
    optstash = p4util.OptionsState(
        ["SCF_TYPE"],
        ["SCF", "REFERENCE"],
        ["SCF", "DFT_GRAC_SHIFT"],
        ["SCF", "SAVE_JK"],
        ["SCF", "MAXITER"],
        ["SCF", "LEVEL_SHIFT"],
        ["SCF", "LEVEL_SHIFT_CUTOFF"],
        ["SCF", "SCF_INITIAL_ACCELERATOR"],
        ["BASIS"],
    )

    dft_functional = core.get_option("SAPT", "SAPT_DFT_FUNCTIONAL")
    grac_basis = core.get_option("SAPT", "SAPT_DFT_GRAC_BASIS")
    if grac_basis != "AUTO":
        core.set_global_option("BASIS", grac_basis)

    core.print_out(f"Computing GRAC shift for {label} using {sapt_dft_grac_convergence_tier}...")
    grac_options = sapt_dft_grac_convergence_tier_options[
        sapt_dft_grac_convergence_tier
    ]
    for options in grac_options:
        for key, val in options.items():
            core.set_local_option("SCF", key, val)
        # Need to get the initial and cation to estimate ionization energy for
        # GRAC shift
        mol_qcel_dict = molecule.to_schema(dtype=3)
        del mol_qcel_dict["fragment_charges"]
        del mol_qcel_dict["fragment_multiplicities"]
        del mol_qcel_dict["molecular_multiplicity"]
        given_charge = mol_qcel_dict["molecular_charge"]

        mol_given = core.Molecule.from_schema(mol_qcel_dict)
        mol_qcel_dict["molecular_charge"] += 1
        mol_cation = core.Molecule.from_schema(mol_qcel_dict)

        core.print_out(f"\n\n  ==> GRAC {label} Given Molecule: charge={mol_given.molecular_charge()} mult={mol_given.multiplicity()} <==\n\n")
        try:
            if mol_given.multiplicity() != 1:
                core.set_local_option("SCF", "REFERENCE", "UHF")
            else:
                core.set_local_option("SCF", "REFERENCE", "RHF")
            wfn_given = run_scf(
                dft_functional.lower(),
                molecule=mol_given,
            )
            if mol_cation.multiplicity() != 1:
                core.set_local_option("SCF", "REFERENCE", "UHF")
            else:
                core.set_local_option("SCF", "REFERENCE", "RHF")
            core.print_out(f"\n\n  ==> GRAC {label} Electron Removed Molecule: charge={mol_cation.molecular_charge()} mult={mol_cation.multiplicity()} <==\n\n")
            wfn_cation = run_scf(
                dft_functional.lower(),
                molecule=mol_cation,
            )
        except ConvergenceError:
            if len(options) == 1:
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
            raise Exception(f"The computed GRAC shift ({grac} [E_h]) for {label} exceeds the bounds of -1 < x < 1 and should not be used to approximate the ionization potential.")
    optstash.restore()
    return grac


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
    fsapt_induction_data: tuple | None = None,
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
    fsapt_induction_data : tuple or None, optional
        Private transport of SAPT0 monomer wavefunctions, cache, and scalar data
        used to partition CPHF induction with the FISAPT implementation.

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
    do_fsapt = fsapt_type != "NONE"
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

    if do_fsapt:
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

        # Create single FISAPT object with do_flocalize=True to handle IBO localization internally.
        core.timer_on("SAPT(DFT): F-SAPT Setup + Localization (IBO)")
        FISAPT_obj = saptdft_fisapt.setup_fisapt_object(
            dimer_wfn, wfn_A, wfn_B, cache, data, aux_basis, do_flocalize=True
        )
        core.timer_off("SAPT(DFT): F-SAPT Setup + Localization (IBO)")

        core.timer_on("SAPT(DFT): F-SAPT Electrostatics")
        FISAPT_obj.felst()
        core.timer_off("SAPT(DFT): F-SAPT Electrostatics")
        core.timer_on("SAPT(DFT): F-SAPT Exchange")
        FISAPT_obj.fexch()
        core.timer_off("SAPT(DFT): F-SAPT Exchange")
        core.timer_on("SAPT(DFT): F-SAPT Induction")
        if induction_matrices is None:
            FISAPT_obj.find()
        else:
            FISAPT_obj.set_matrix(induction_matrices)
        core.timer_off("SAPT(DFT): F-SAPT Induction")
        matrices = FISAPT_obj.matrices()
        for k, v in matrices.items():
            cache[k] = v

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

    # F-SAPT computes the HF dispersion itself (fdisp below), partitioned.
    if not do_fsapt:
        core.timer_on("MP2 disp")
        if core.get_option("SAPT", "SAPT_DFT_MP2_DISP_ALG") == "FISAPT":
            mp2_disp = sapt_mp2.df_mp2_fisapt_dispersion(
                wfn_A, primary_basis, aux_basis, cache, nfrozen_A, nfrozen_B, do_print=True
            )
        else:
            mp2_disp = sapt_mp2.df_mp2_sapt_dispersion(
                dimer_wfn, wfn_A, wfn_B, primary_basis, aux_basis, cache, do_print=True
            )
        core.timer_off("MP2 disp")
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

    core.timer_off("SAPT(DFT):disp")

    # Now do F-SAPT on dispersion if requested
    if do_fsapt:
        core.timer_on("SAPT(DFT): F-SAPT Dispersion")
        FISAPT_obj.fdisp()
        core.timer_off("SAPT(DFT): F-SAPT Dispersion")
        FISAPT_obj.fdrop(external_potentials)
        scalars = FISAPT_obj.scalars()
        data["Exch-Disp20,u"] = scalars["Exch-Disp20"]
        data["Disp20,u"] = scalars["Disp20"]
        matrices = FISAPT_obj.matrices()
        for k, v in matrices.items():
            cache[k] = v

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
        _set_fsapt_var("FSAPT_DISP_AB", cache["Disp_AB"])
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
