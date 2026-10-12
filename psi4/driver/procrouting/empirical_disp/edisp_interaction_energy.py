#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2025 The Psi4 Developers.
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

from ...p4util.exceptions import ValidationError
from . import empirical_dispersion
from .xdm_counterpoise import ghost_superposition_correction


def _sapt_dft_dispersion_interaction_energy(
    sapt_dimer: core.Molecule,
    monomerA: core.Molecule,
    monomerB: core.Molecule,
    dimer_wfn: core.Wavefunction,
    functional_name: str,
    d_type: str,
    data: dict[str, object],
    *,
    disp_label: str,
    gd_type: str,
    supermolecular_text: str,
    gd_supermolecular_text: str,
    two_body_level_hint: str,
    atm_level_hint: str,
) -> dict[str, object]:
    """Compute SAPT(DFT) empirical dispersion interaction terms.

    Parameters
    ----------
    sapt_dimer : core.Molecule
        Dimer molecule used for the SAPT(DFT) dispersion evaluation.
    monomerA : core.Molecule
        Monomer A molecule.
    monomerB : core.Molecule
        Monomer B molecule.
    dimer_wfn : core.Wavefunction
        Dimer wavefunction object used to store/read pairwise variables.
    functional_name : str
        Dispersion parameter lookup key (usually functional name).
    d_type : str
        Dispersion mode selector (e.g., ``supermolecular``).
    data : dict[str, object]
        Result dictionary updated in place.
    disp_label : str
        Label prefix for output keys (e.g., ``D3`` or ``D4``).
    gd_type : str
        Selector value for Grimme ATM-enabled supermolecular mode.
    supermolecular_text : str
        Header text for the regular supermolecular branch.
    gd_supermolecular_text : str
        Header text for the ATM-enabled supermolecular branch.
    two_body_level_hint : str
        Dispersion level hint for two-body-only calculations.
    atm_level_hint : str
        Dispersion level hint for ATM-enabled calculations.

    Returns
    -------
    dict[str, object]
        Updated ``data`` dictionary containing dispersion terms.
    """

    if core.has_option_changed("SCF", "DFT_DISPERSION_PARAMETERS"):
        modified_disp_params = core.get_option("SCF", "DFT_DISPERSION_PARAMETERS")
    else:
        modified_disp_params = None

    if d_type == "supermolecular":
        core.print_out(
            "         "
            + supermolecular_text.center(58)
            + "\n\n"
        )
        _disp_functor = empirical_dispersion.EmpiricalDispersion(
            name_hint=functional_name,
            level_hint=two_body_level_hint,
            param_tweaks=modified_disp_params,
            save_pairwise_disp=True,
        )
        dimer_disp = _disp_functor.compute_energy(sapt_dimer, dimer_wfn)
        monA_disp = _disp_functor.compute_energy(monomerA)
        monB_disp = _disp_functor.compute_energy(monomerB)
        data[f"{disp_label} DIMER"] = dimer_disp
        data[f"{disp_label} MONOMER A"] = monA_disp
        data[f"{disp_label} MONOMER B"] = monB_disp
        data[f"{disp_label} IE"] = dimer_disp - monA_disp - monB_disp
        data['FSAPT_EMPIRICAL_DISP'] = dimer_wfn.variable("PAIRWISE DISPERSION CORRECTION ANALYSIS")
        _disp_functor.print_out()
    elif d_type == "intermolecular":
        monAs = np.array([i for i in range(monomerA.natom()) if monomerA.Z(i) > 0])
        monBs = np.array([i for i in range(monomerB.natom()) if monomerB.Z(i) > 0])
        _disp_functor = empirical_dispersion.EmpiricalDispersion(
            name_hint=functional_name,
            level_hint=two_body_level_hint,
            param_tweaks=modified_disp_params,
            save_pairwise_disp=True,
        )
        E_disp = 0.0
        _ = _disp_functor.compute_energy(sapt_dimer, dimer_wfn)
        pairwise_energies = dimer_wfn.variable("PAIRWISE DISPERSION CORRECTION ANALYSIS").np
        FSAPT_EMPIRICAL_DISP = np.zeros_like(pairwise_energies)
        for A in monAs:
            for B in monBs:
                # account for A-B and B-A with factor of 2
                E_disp += 2 * pairwise_energies[A, B]
                FSAPT_EMPIRICAL_DISP[A, B] = pairwise_energies[A, B]
                FSAPT_EMPIRICAL_DISP[B, A] = pairwise_energies[A, B]
        data[f"{disp_label} IE"] = E_disp
        data['FSAPT_EMPIRICAL_DISP'] = FSAPT_EMPIRICAL_DISP
        _disp_functor.print_out()
    elif d_type == gd_type:
        # This uses the default Grimme parameters for the given
        # functional with BJ damping and ATM three-body terms. Only use
        # this option in compbination with another baseline form of
        # dispersion like the delta_DFT correction:
        # "SAPT_DFT_DO_DDFT = True"
        core.print_out(
            "         "
            + gd_supermolecular_text.center(58)
            + "\n\n"
        )
        _disp_functor = empirical_dispersion.EmpiricalDispersion(
            name_hint=functional_name,
            level_hint=atm_level_hint,
            param_tweaks=modified_disp_params,
            save_pairwise_disp=True,
        )

        dimer_disp = _disp_functor.compute_energy(sapt_dimer, dimer_wfn)
        monA_disp = _disp_functor.compute_energy(monomerA)
        monB_disp = _disp_functor.compute_energy(monomerB)
        data[f"{disp_label} DIMER"] = dimer_disp
        data[f"{disp_label} MONOMER A"] = monA_disp
        data[f"{disp_label} MONOMER B"] = monB_disp
        data[f"{disp_label} IE"] = dimer_disp - monA_disp - monB_disp
        data['FSAPT_EMPIRICAL_DISP'] = dimer_wfn.variable("PAIRWISE DISPERSION CORRECTION ANALYSIS")
        _disp_functor.print_out()
    else:
        raise ValueError(
            "SAPT(DFT): d_type must be 'supermolecular' or 'intermolecular'."
        )
    return data


def sapt_dft_d4_interaction_energy(
    sapt_dimer: core.Molecule,
    monomerA: core.Molecule,
    monomerB: core.Molecule,
    dimer_wfn: core.Wavefunction,
    dftd4_functional_name: str,
    d4_type: str,
    data: dict[str, object],
) -> dict[str, object]:
    """Compute SAPT(DFT) D4 interaction energy contributions.

    Parameters
    ----------
    sapt_dimer : core.Molecule
        Dimer molecule used for the SAPT(DFT) dispersion evaluation.
    monomerA : core.Molecule
        Monomer A molecule.
    monomerB : core.Molecule
        Monomer B molecule.
    dimer_wfn : core.Wavefunction
        Dimer wavefunction object used to store/read pairwise variables.
    dftd4_functional_name : str
        Dispersion parameter lookup key for D4.
    d4_type : str
        D4 mode selector (e.g., ``supermolecular``, ``intermolecular``, or ``gd4_supermolecular``).
    data : dict[str, object]
        Result dictionary updated in place.

    Returns
    -------
    dict[str, object]
        Updated ``data`` dictionary containing D4 terms.
    """

    return _sapt_dft_dispersion_interaction_energy(
        sapt_dimer,
        monomerA,
        monomerB,
        dimer_wfn,
        dftd4_functional_name,
        d4_type,
        data,
        disp_label="D4",
        gd_type="gd4_supermolecular",
        supermolecular_text="Supermolecular D4 Interaction Energy E_IE = E_IJ - E_I - E_J",
        gd_supermolecular_text="Supermolecular GD4(BJ)+ATM Interaction Energy E_IE = E_IJ - E_I - E_J",
        two_body_level_hint='d4bj2b',
        atm_level_hint='d4bjeeqatm',
    )


def sapt_dft_d3_interaction_energy(
    sapt_dimer: core.Molecule,
    monomerA: core.Molecule,
    monomerB: core.Molecule,
    dimer_wfn: core.Wavefunction,
    dftd3_functional_name: str,
    d3_type: str,
    data: dict[str, object],
) -> dict[str, object]:
    """Compute SAPT(DFT) D3 interaction energy contributions.

    Parameters
    ----------
    sapt_dimer : core.Molecule
        Dimer molecule used for the SAPT(DFT) dispersion evaluation.
    monomerA : core.Molecule
        Monomer A molecule.
    monomerB : core.Molecule
        Monomer B molecule.
    dimer_wfn : core.Wavefunction
        Dimer wavefunction object used to store/read pairwise variables.
    dftd3_functional_name : str
        Dispersion parameter lookup key for D3.
    d3_type : str
        D3 mode selector (e.g., ``supermolecular``, ``intermolecular``, or ``gd3_supermolecular``).
    data : dict[str, object]
        Result dictionary updated in place.

    Returns
    -------
    dict[str, object]
        Updated ``data`` dictionary containing D3 terms.
    """

    return _sapt_dft_dispersion_interaction_energy(
        sapt_dimer,
        monomerA,
        monomerB,
        dimer_wfn,
        dftd3_functional_name,
        d3_type,
        data,
        disp_label="D3",
        gd_type="gd3_supermolecular",
        supermolecular_text="Supermolecular -D3MBJ Interaction Energy E_IE = E_IJ - E_I - E_J",
        gd_supermolecular_text="Supermolecular GD3(BJ)+ATM Interaction Energy E_IE = E_IJ - E_I - E_J",
        two_body_level_hint='d3mbj2b',
        atm_level_hint='d3mbjatm',
    )


def sapt_dft_xdm_interaction_energy(
    sapt_dimer: core.Molecule,
    monomerA: core.Molecule,
    monomerB: core.Molecule,
    dimer_wfn: core.Wavefunction,
    monomerA_wfn: core.Wavefunction,
    monomerB_wfn: core.Wavefunction,
    xdm_functional_name: str,
    xdm_model: str,
    data: dict[str, object],
) -> dict[str, object]:
    """Compute the SAPT(DFT) counterpoise XDM interaction energy.

    The delta-DFT dimer and monomer wavefunctions are all in the dimer basis, so
    the monomer XDM energies are counterpoise subsystems. Each monomer's XDM
    energy is evaluated with its partner's ghost-centre moments superposed onto
    its atoms (see :mod:`xdm_counterpoise`), so the supermolecular difference
    keeps only intermolecular pairs and the true density deformation.

    Parameters
    ----------
    sapt_dimer : core.Molecule
        Dimer molecule.
    monomerA : core.Molecule
        Monomer A with monomer B as ghosts (dimer basis).
    monomerB : core.Molecule
        Monomer B with monomer A as ghosts (dimer basis).
    dimer_wfn : core.Wavefunction
        Converged delta-DFT dimer wavefunction.
    monomerA_wfn : core.Wavefunction
        Converged delta-DFT monomer A wavefunction in the dimer basis.
    monomerB_wfn : core.Wavefunction
        Converged delta-DFT monomer B wavefunction in the dimer basis.
    xdm_functional_name : str
        XC functional of the delta-DFT wavefunctions, without ``-xdm``.
    xdm_model : str
        XDM damping-parameter model (``kb49`` or ``los-ii``), used when
        ``XDM_DISPERSION_PARAMETERS`` is not set.
    data : dict[str, object]
        Result dictionary updated in place.

    Returns
    -------
    dict[str, object]
        Updated ``data`` dictionary containing XDM terms.
    """

    if core.has_option_changed("SCF", "XDM_DISPERSION_PARAMETERS"):
        params = core.get_option("SCF", "XDM_DISPERSION_PARAMETERS")
        if len(params) != 2 or not all(np.isfinite(float(p)) for p in params):
            raise ValidationError("XDM_DISPERSION_PARAMETERS must contain exactly two finite values: [a1, a2_angstrom].")
        a1, a2_ang = (float(p) for p in params)
        xdm_functor = empirical_dispersion.XDMDispersionFunctor(
            functional_name=xdm_functional_name, a1=a1, a2_ang=a2_ang, model=xdm_model)
    else:
        xdm_functor = empirical_dispersion.XDMDispersionFunctor(functional_name=xdm_functional_name,
                                                                basis_name=core.get_global_option("BASIS"),
                                                                model=xdm_model,
                                                                counterpoise=True)

    core.print_out("         " + "Supermolecular XDM Interaction Energy E_IE = E_IJ - E_I - E_J".center(58) + "\n")
    core.print_out("         " + "(counterpoise monomers carry superposed ghost-centre moments)".center(58) + "\n\n")

    dimer_disp = xdm_functor.compute_energy(sapt_dimer, dimer_wfn)
    monA_disp = xdm_functor.compute_energy(monomerA, monomerA_wfn)
    monB_disp = xdm_functor.compute_energy(monomerB, monomerB_wfn)

    coords = sapt_dimer.geometry().np
    props_A = monomerA_wfn.array_variable("XDM ATOMIC PROPERTIES").np
    props_B = monomerB_wfn.array_variable("XDM ATOMIC PROPERTIES").np
    for mol in (monomerA, monomerB):
        if mol.natom() != sapt_dimer.natom() or not np.allclose(mol.geometry().np, coords, atol=1.0e-8):
            raise ValidationError("SAPT(DFT) XDM: monomers must keep every dimer centre (as ghosts) in dimer order.")
    corr_A = ghost_superposition_correction(xdm_functor.xdm, coords, props_A, [props_B])
    corr_B = ghost_superposition_correction(xdm_functor.xdm, coords, props_B, [props_A])

    data["XDM DIMER"] = dimer_disp
    data["XDM MONOMER A"] = monA_disp + corr_A
    data["XDM MONOMER B"] = monB_disp + corr_B
    data["XDM IE"] = data["XDM DIMER"] - data["XDM MONOMER A"] - data["XDM MONOMER B"]
    core.set_variable("SAPT(DFT) XDM IE", data["XDM IE"])

    core.print_out(f"    {'XDM dimer':<34} {dimer_disp:18.12f} [Eh]\n")
    core.print_out(f"    {'XDM monomer A (bare CP)':<34} {monA_disp:18.12f} [Eh]\n")
    core.print_out(f"    {'XDM monomer A ghost superposition':<34} {corr_A:18.12f} [Eh]\n")
    core.print_out(f"    {'XDM monomer B (bare CP)':<34} {monB_disp:18.12f} [Eh]\n")
    core.print_out(f"    {'XDM monomer B ghost superposition':<34} {corr_B:18.12f} [Eh]\n")
    core.print_out(f"    {'XDM IE':<34} {data['XDM IE']:18.12f} [Eh]\n\n")

    # F-SAPT breakdown: intermolecular atom pairs of the dimer evaluation.
    pairwise_energies = dimer_wfn.array_variable("XDM PAIRWISE ENERGY").np
    fsapt_empirical_disp = np.zeros((sapt_dimer.natom(), sapt_dimer.natom()))
    real_A = [i for i in range(monomerA.natom()) if monomerA.Z(i) > 0]
    real_B = [i for i in range(monomerB.natom()) if monomerB.Z(i) > 0]
    for a in real_A:
        for b in real_B:
            fsapt_empirical_disp[a, b] = fsapt_empirical_disp[b, a] = pairwise_energies[a, b]
    data["FSAPT_EMPIRICAL_DISP"] = fsapt_empirical_disp
    xdm_functor.print_out()
    return data
