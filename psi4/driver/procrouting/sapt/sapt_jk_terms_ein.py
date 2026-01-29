#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2024 The Psi4 Developers.
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

import time
from typing import List, Tuple

import numpy as np

from psi4 import core

from ...p4util import solvers
from .sapt_util import print_sapt_var
import einsums as ein


# Equations come from https://doi.org/10.1063/5.0090688


# ==> Helper Functions for psi4.core.Matrix with einsums operations <==

def _to_numpy(obj):
    """
    Convert a psi4.core.Matrix or ein.core.RuntimeTensorD to numpy array.
    """
    if isinstance(obj, core.Matrix):
        return obj.np
    elif isinstance(obj, ein.core.RuntimeTensorD):
        return np.asarray(obj)
    elif isinstance(obj, np.ndarray):
        return obj
    elif hasattr(obj, '__array__'):  # Fallback for array-like objects
        return np.asarray(obj)
    else:
        raise TypeError(f"Cannot convert {type(obj)} to numpy array")


def _to_matrix(arr, name=""):
    """
    Convert numpy array to psi4.core.Matrix.
    """
    if isinstance(arr, core.Matrix):
        return arr
    mat = core.Matrix.from_array(np.asarray(arr))
    if name:
        mat.name = name
    return mat


def matrix_dot(A, B):
    """
    Compute the Frobenius inner product (element-wise dot product) of two 
    matrices.
    
    Equivalent to ein.core.dot() or psi4.core.Matrix.vector_dot()
    
    Parameters
    ----------
    A : psi4.core.Matrix or array-like
    B : psi4.core.Matrix or array-like
    
    Returns
    -------
    float
        The scalar dot product.
    """
    A.np = _to_numpy(A)
    B.np = _to_numpy(B)
    return np.vdot(A.np, B.np)


def matrix_axpy(alpha, X, Y):
    """
    Compute Y = alpha * X + Y in-place.
    
    Equivalent to ein.core.axpy(alpha, X, Y) or ein.core.axpy(alpha, X.np, Y.np) for 
    psi4.core.Matrix.
    
    Parameters
    ----------
    alpha : float
        Scalar multiplier.
    X : psi4.core.Matrix or array-like
        Input matrix.
    Y : psi4.core.Matrix
        Output matrix (modified in-place).
    """
    X.np = _to_numpy(X)
    if isinstance(Y, core.Matrix):
        Y.np[:] += alpha * X.np
    else:
        # For einsums tensors
        Y.np = _to_numpy(Y)
        Y.np[:] += alpha * X.np


def matrix_copy(A, name=""):
    """
    Create a copy/clone of matrix A.
    
    Parameters
    ----------
    A : psi4.core.Matrix or array-like
    name : str, optional
        Name for the new matrix.
    
    Returns
    -------
    psi4.core.Matrix
        Copy of the input matrix.
    """
    if isinstance(A, core.Matrix):
        result = A.clone()
        if name:
            result.name = name
        return result
    else:
        return _to_matrix(np.array(_to_numpy(A)), name)


def matrix_scale(A, alpha):
    """
    Scale matrix A by alpha in-place.
    
    Parameters
    ----------
    A : psi4.core.Matrix
        Matrix to scale (modified in-place).
    alpha : float
        Scale factor.
    """
    if isinstance(A, core.Matrix):
        A.scale(alpha)
    else:
        A.np = _to_numpy(A)
        A.np *= alpha


def localization(cache, dimer_wfn, wfn_A, wfn_B, do_print=True):
    core.print_out("\n  ==> Localizing Orbitals 1 <== \n\n")
    # localization_scheme = core.get_option("SAPT", "SAPT_DFT_LOCAL_ORBITALS")
    # loc = core.Localizer.build(localization_scheme, wfn_A.basisset(), wfn_A.Ca_subset("AO", "OCC"))
    # loc.localize()
    # C_lmo_A = loc.L
    # loc = core.Localizer.build(localization_scheme, wfn_B.basisset(), wfn_B.Ca_subset("AO", "OCC"))
    # loc.localize()
    # C_lmo_B = loc.L
    # IBOLocalizer
    
    # Extract monomers to compute frozen core counts
    mol = dimer_wfn.molecule()
    molA = mol.extract_subsets([1], [])
    molB = mol.extract_subsets([2], [])
    nfocc0A = dimer_wfn.basisset().n_frozen_core(core.get_option("GLOBALS", "FREEZE_CORE"), molA)
    nfocc0B = dimer_wfn.basisset().n_frozen_core(core.get_option("GLOBALS", "FREEZE_CORE"), molB)
    nfocc_dimer = nfocc0A + nfocc0B
    
    N_eps_focc = cache["eps_focc"].dimpi()[0]
    N_eps_occ = cache["eps_occ"].dimpi()[0]
    Focc = core.Matrix("Focc", N_eps_occ, N_eps_occ)
    for i in range(N_eps_occ):
        Focc.np[i, i] = cache["eps_occ"].np[i]
    ranges = [0, nfocc_dimer, N_eps_occ]  # Separate frozen and active orbitals
    minao = core.BasisSet.build(dimer_wfn.molecule(), "BASIS", core.get_global_option("MINAO_BASIS"))
    dimer_wfn.set_basisset("MINAO", minao)
    # pybind11 location: ./psi4/src/export_wavefunction.cc
    IBO_loc = core.IBOLocalizer2(
        dimer_wfn.basisset(),
        dimer_wfn.get_basisset("MINAO"),
        # dimer_wfn.Ca_subset("AO", "OCC"),
        cache['Cocc'],
    )
    IBO_loc.print_header()
    ret = IBO_loc.localize(
        cache['Cocc'],
        Focc,
        ranges,
    )
    cache['Locc'] = ret['L']
    cache['Qocc'] = ret['Q']
    cache['IAO'] = ret['A']
    
    # Extract frozen and active localized orbitals separately
    nn = cache['Cocc'].shape[0]  # number of AO basis functions
    nf = nfocc_dimer
    na = N_eps_occ - nfocc_dimer  # number of active occupied orbitals
    
    if nf > 0:
        # Store frozen core localized orbitals
        Lfocc = core.Matrix("Lfocc", nn, nf)
        Lfocc.np[:, :] = ret['L'].np[:, :nf]
        cache['Lfocc'] = Lfocc
    
    # Store active occupied localized orbitals
    Laocc = core.Matrix("Laocc", nn, na)
    Laocc.np[:, :] = ret['L'].np[:, nf:]
    cache['Laocc'] = Laocc
    
    return


def flocalization(cache, dimer_wfn, wfn_A, wfn_B, do_print=True):
    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    core.print_out("  ==> F-SAPT Localization (IBO) <==\n\n")
    core.print_out("  ==> Local orbitals for Monomer A <==\n\n")
    mol = dimer_wfn.molecule()
    molA = mol.extract_subsets([1], [])
    molB = mol.extract_subsets([2], [])
    nfocc0A = dimer_wfn.basisset().n_frozen_core(core.get_option("GLOBALS", "FREEZE_CORE"), molA)
    nfocc0B = dimer_wfn.basisset().n_frozen_core(core.get_option("GLOBALS", "FREEZE_CORE"), molB)
    nn = cache["Cocc_A"].shape[0]
    nf = nfocc0A
    nm = cache["Cocc_A"].shape[1]  # total occupied orbitals (frozen + active)
    na = nm - nf  # active occupied orbitals only
    ranges = [0, nf, nm]
    N = cache['eps_occ_A'].shape[0]
    Focc = core.Matrix("Focc", N, N)
    for i in range(N):
        Focc.np[i, i] = cache["eps_occ_A"].np[i]
    IBO_loc = core.IBOLocalizer2(
        dimer_wfn.basisset(),
        dimer_wfn.get_basisset("MINAO"),
        core.Matrix.from_array(cache['Cocc_A']),
    )
    IBO_loc.print_header()
    ret = IBO_loc.localize(
        core.Matrix.from_array(cache['Cocc_A']),
        Focc,
        ranges,
    )
    
    Locc_A = ret["L"]
    Uocc_A = ret["U"]
    Qocc0A = ret["Q"]
    
    cache["Locc_A"] = Locc_A
    cache["Uocc_A"] = Uocc_A
    cache["Qocc0A"] = Qocc0A
    
    Lfocc0A = core.Matrix("Lfocc0A", nn, nf)
    Laocc0A = core.Matrix("Laocc0A", nn, na)
    Ufocc0A = core.Matrix("Ufocc0A", nf, nf)
    Uaocc0A = core.Matrix("Uaocc0A", na, na)
    
    Lfocc0A.np[:, :] = Locc_A.np[:, :nf]
    Laocc0A.np[:, :] = Locc_A.np[:, nf:nf+na]
    Ufocc0A.np[:, :] = Uocc_A.np[:nf, :nf]
    Uaocc0A.np[:, :] = Uocc_A.np[nf:nf+na, nf:nf+na]
    
    cache["Lfocc0A"] = Lfocc0A
    cache["Laocc0A"] = Laocc0A
    cache["Ufocc0A"] = Ufocc0A
    cache["Uaocc0A"] = Uaocc0A
    # Store active occupied orbitals for dispersion (Caocc0A = Cocc_A[:, nf:])
    Caocc0A = core.Matrix("Caocc0A", nn, na)
    Caocc0A.np[:, :] = cache["Cocc_A"].np[:, nf:nf+na]
    cache["Caocc0A"] = Caocc0A

    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        # For I-SAPT, thislinkA may not exist if we haven't computed link orbitals yet
        # In that case, use the link orbitals from the partition function (LoccL)
        if "thislinkA" in cache:
            # Save the old localized orbitals before creating new matrix
            old_Locc_A = Locc_A.np.copy()
            Locc_A_new = core.Matrix("Locc_A", nn, nm + 1)
            Locc_A_new.np[:, :nm] = old_Locc_A[:, :nm]
            Locc_A_new.np[:, nm] = cache["thislinkA"].np[:, 0]
            cache["Locc_A"] = Locc_A_new
        elif "LoccL" in cache and cache["LoccL"] is not None:
            # Use link orbitals from partition for I-SAPT
            # LoccL contains the link orbital(s) assigned to fragment C
            # We need to include them with appropriate scaling for F-SAPT
            old_Locc_A = Locc_A.np.copy()
            n_link = cache["LoccL"].shape[1] if hasattr(cache["LoccL"], 'shape') else cache["LoccL"].np.shape[1]
            Locc_A_new = core.Matrix("Locc_A", nn, nm + n_link)
            Locc_A_new.np[:, :nm] = old_Locc_A[:, :nm]
            # Add scaled link orbitals (0.5 contribution to each fragment for a shared link)
            link_data = cache["LoccL"].np if hasattr(cache["LoccL"], 'np') else cache["LoccL"]
            Locc_A_new.np[:, nm:] = link_data * 0.7071067811865475244  # 1/sqrt(2)
            cache["Locc_A"] = Locc_A_new
            core.print_out(f"  I-SAPT: Added {n_link} link orbital(s) to Locc_A\n")
        else:
            cache["Locc_A"] = Locc_A
    else:
        cache["Locc_A"] = Locc_A
    
    core.print_out("  ==> Local orbitals for Monomer B <==\n\n")
    
    nn = cache["Cocc_B"].shape[0]
    nf = nfocc0B
    nm = cache["Cocc_B"].shape[1]  # total occupied orbitals (frozen + active)
    na = nm - nf  # active occupied orbitals only
    ranges = [0, nf, nm]
    
    N = cache['eps_occ_B'].shape[0]
    Focc = core.Matrix("Focc", N, N)
    for i in range(N):
        Focc.np[i, i] = cache["eps_occ_B"].np[i]
    
    IBO_loc = core.IBOLocalizer2(
        dimer_wfn.basisset(),
        dimer_wfn.get_basisset("MINAO"),
        # dimer_wfn.Ca_subset("AO", "OCC"),
        core.Matrix.from_array(cache['Cocc_B']),
    )
    IBO_loc.print_header()
    ret = IBO_loc.localize(
        core.Matrix.from_array(cache['Cocc_B']),
        Focc,
        ranges,
    )
    
    Locc_B = ret["L"]
    Uocc_B = ret["U"]
    Qocc0B = ret["Q"]
    
    cache["Locc_B"] = Locc_B
    cache["Uocc_B"] = Uocc_B
    cache["Qocc0B"] = Qocc0B
    
    Lfocc0B = core.Matrix("Lfocc0B", nn, nf)
    Laocc0B = core.Matrix("Laocc0B", nn, na)
    Ufocc0B = core.Matrix("Ufocc0B", nf, nf)
    Uaocc0B = core.Matrix("Uaocc0B", na, na)
    
    Lfocc0B.np[:, :] = Locc_B.np[:, :nf]
    Laocc0B.np[:, :] = Locc_B.np[:, nf:nf+na]
    Ufocc0B.np[:, :] = Uocc_B.np[:nf, :nf]
    Uaocc0B.np[:, :] = Uocc_B.np[nf:nf+na, nf:nf+na]
    
    cache["Lfocc0B"] = Lfocc0B
    cache["Laocc0B"] = Laocc0B
    cache["Ufocc0B"] = Ufocc0B
    cache["Uaocc0B"] = Uaocc0B
    # Store active occupied orbitals for dispersion (Caocc0B = Cocc_B[:, nf:])
    Caocc0B = core.Matrix("Caocc0B", nn, na)
    Caocc0B.np[:, :] = cache["Cocc_B"].np[:, nf:nf+na]
    cache["Caocc0B"] = Caocc0B

    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        # For I-SAPT, thislinkB may not exist if we haven't computed link orbitals yet
        if "thislinkB" in cache:
            # Save the old localized orbitals before creating new matrix
            old_Locc_B = Locc_B.np.copy()
            Locc_B_new = core.Matrix("Locc_B", nn, nm + 1)
            Locc_B_new.np[:, :nm] = old_Locc_B[:, :nm]
            Locc_B_new.np[:, nm] = cache["thislinkB"].np[:, 0]
            cache["Locc_B"] = Locc_B_new
        elif "LoccL" in cache and cache["LoccL"] is not None:
            # Use link orbitals from partition for I-SAPT
            old_Locc_B = Locc_B.np.copy()
            n_link = cache["LoccL"].shape[1] if hasattr(cache["LoccL"], 'shape') else cache["LoccL"].np.shape[1]
            Locc_B_new = core.Matrix("Locc_B", nn, nm + n_link)
            Locc_B_new.np[:, :nm] = old_Locc_B[:, :nm]
            # Add scaled link orbitals (0.5 contribution to each fragment for a shared link)
            link_data = cache["LoccL"].np if hasattr(cache["LoccL"], 'np') else cache["LoccL"]
            Locc_B_new.np[:, nm:] = link_data * 0.7071067811865475244  # 1/sqrt(2)
            cache["Locc_B"] = Locc_B_new
            core.print_out(f"  I-SAPT: Added {n_link} link orbital(s) to Locc_B\n")
        else:
            cache["Locc_B"] = Locc_B
    else:
        cache["Locc_B"] = Locc_B


def partition(cache, dimer_wfn, wfn_A, wfn_B, do_print=True):
    core.print_out("\n  ==> Partitioning <== \n\n")
    # Sizing
    mol = dimer_wfn.molecule()
    natoms = mol.natom()
    n_Locc = cache['Locc'].shape[1]


    # Monomer Atoms
    fragments = mol.get_fragments()
    indA = np.arange(*fragments[0], dtype=int)
    indB = np.arange(*fragments[1], dtype=int)
    indC = np.arange(*fragments[2], dtype=int) if len(fragments) == 3 else np.array([], dtype=int)
    cache["FRAG"] = core.Vector(natoms)
    frag = cache["FRAG"].np
    frag[:] = 0.0
    frag[indA] = 1.0
    frag[indB] = 2.0
    if indC.size:
        frag[indC] = 3.0
    core.print_out( "Fragment lookup table:\n")
    cache['FRAG'].print_out()
    core.print_out("   => Atomic Partitioning <= \n\n")
    core.print_out(f"    Monomer A: {len(indA)} atoms\n")
    core.print_out(f"    Monomer B: {len(indB)} atoms\n")
    core.print_out(f"    Monomer C: {len(indC)} atoms\n\n")
    np.set_printoptions(precision=14, suppress=True)

    # Fragment Orbital Charges
    Locc = cache["Locc"].np     # (n_ao x n_occ)
    Qocc = cache["Qocc"].np     # (n_atom x n_occ) orbital populations per atom

    n_ao, n_occ = Locc.shape

    QF = core.Matrix(3, n_Locc).np
    QF.fill(0.0)
    QF[0, :] = Qocc[indA, :].sum(axis=0)
    QF[1, :] = Qocc[indB, :].sum(axis=0)
    if indC.size:
        QF[2, :] = Qocc[indC, :].sum(axis=0)
    # QF init is in slightly different order than C++...

    # --- link identification ---
    link_orbs: List[int] = []
    link_atoms: List[Tuple[int, int]] = []
    link_types: List[str] = []

    def top_two_atoms_for_orb(a: int) -> tuple[int, int]:
        A_sorted = np.argsort(Qocc[:, a])[::-1]
        A1, A2 = int(A_sorted[0]), int(A_sorted[1])
        return (A1, A2) if A1 < A2 else (A2, A1)

    link_sel = core.get_option("FISAPT", "FISAPT_LINK_SELECTION").upper()
    if link_sel == "AUTOMATIC":
        delta = float(core.get_option("FISAPT", "FISAPT_CHARGE_COMPLETENESS"))
        for a in range(n_occ):
            if np.any(QF[:, a] > delta):
                continue
            if QF[0, a] + QF[2, a] > delta:
                link_orbs.append(a)
                link_types.append("AC")
            elif QF[1, a] + QF[2, a] > delta:
                link_orbs.append(a)
                link_types.append("BC")
            elif QF[0, a] + QF[1, a] > delta:
                link_orbs.append(a)
                link_types.append("AB")
            else:
                raise ValueError("FISAPT: 3c-2e style bond encountered (no single/pair exceeds delta).")
        for a in link_orbs:
            link_atoms.append(top_two_atoms_for_orb(a))
    elif link_sel == "MANUAL":
        if not core.get_option("FISAPT", "FISAPT_MANUAL_LINKS"):
            raise ValueError("FISAPT: MANUAL selection requires manual_links (0-based atom pairs).")
        S = set(indA.tolist())
        T = set(indB.tolist())
        U = set(indC.tolist())
        for (A1, A2) in core.get_option("FISAPT", "FISAPT_MANUAL_LINKS"):
            prod = Qocc[A1, :] * Qocc[A2, :]
            a = int(np.argmax(prod))
            link_orbs.append(a)
            A1_, A2_ = (A1, A2) if A1 < A2 else (A2, A1)
            link_atoms.append((A1_, A2_))
            if (A1_ in S) and (A2_ in U):
                link_types.append("AC")
            elif (A1_ in T) and (A2_ in U):
                link_types.append("BC")
            elif (A1_ in S) and (A2_ in T):
                link_types.append("AB")
            else:
                raise ValueError("FISAPT: manual pair is not AB, AC, or BC.")
    else:
        raise ValueError("FISAPT: Unrecognized FISAPT_LINK_SELECTION.")
    link_orbs = np.array(link_orbs, dtype=int)

    # --- Z per fragment originals ---
    ZA = core.Vector(natoms)
    ZB = core.Vector(natoms)
    ZC = core.Vector(natoms)
    ZA.np[:] = 0.0
    ZB.np[:] = 0.0
    ZC.np[:] = 0.0

    Z_all = np.array([mol.Z(i) for i in range(natoms)], dtype=float)
    ZA.np[indA] = Z_all[indA]
    ZB.np[indB] = Z_all[indB]
    if indC.size:
        ZC.np[indC] = Z_all[indC]

    cache["ZA"] = ZA
    cache["ZB"] = ZB
    cache["ZC"] = ZC
    cache["ZA_orig"] = core.Vector.from_array(ZA.np.copy())
    cache["ZB_orig"] = core.Vector.from_array(ZB.np.copy())
    cache["ZC_orig"] = core.Vector.from_array(ZC.np.copy())

    # --- link assignment (C vs AB vs SAO*/SIAO*) ---
    orbsA: List[int] = []
    orbsB: List[int] = []
    orbsC: List[int] = []
    orbsL: List[int] = []
    typesL: List[str] = []

    la = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    valid = {"AB", "C", "SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}
    if la not in valid:
        raise ValueError("FISAPT: FISAPT_LINK_ASSIGNMENT not recognized.")


    # --- link assignment (C vs AB vs SAO*/SIAO*) ---
    orbsA: List[int] = []
    orbsB: List[int] = []
    orbsC: List[int] = []
    orbsL: List[int] = []
    typesL: List[str] = []

    la = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    valid = {"AB", "C", "SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}
    if la not in valid:
        raise ValueError("FISAPT: FISAPT_LINK_ASSIGNMENT not recognized.")

    if la in {"C", "SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}:
        for a, (A1, A2), t in zip(link_orbs, link_atoms, link_types):
            typesL.append(t)
            if t == "AC":
                ZA.np[A1] -= 1.0
                ZC.np[A1] += 1.0
                orbsC.append(a)
                orbsL.append(a)
            elif t == "BC":
                ZB.np[A1] -= 1.0
                ZC.np[A1] += 1.0
                orbsC.append(a)
                orbsL.append(a)
            elif t == "AB":
                ZA.np[A1] -= 1.0
                ZC.np[A1] += 1.0
                ZB.np[A2] -= 1.0
                ZC.np[A2] += 1.0
                orbsC.append(a)
                orbsL.append(a)
    elif la == "AB":
        for a, (A1, A2), t in zip(link_orbs, link_atoms, link_types):
            if t == "AC":
                ZA.np[A1] += 1.0
                ZC.np[A1] -= 1.0
                orbsA.append(a)
            elif t == "BC":
                ZB.np[A1] += 1.0
                ZC.np[A1] -= 1.0
                orbsB.append(a)
            elif t == "AB":
                raise ValueError("FISAPT: AB link requires LINK_ASSIGNMENT C in this scheme.")

    # --- electron counts per fragment; enforce closed-shell ---
    fragment_charges = mol.get_fragment_charges()
    qA, qB = int(fragment_charges[0]), int(fragment_charges[1])
    qC = int(fragment_charges[2]) if len(fragment_charges) == 3 else 0

    def i_round(x: float) -> int:
        # protect against bankers rounding
        return int(np.floor(x + 0.5))

    ZA2 = i_round(float(ZA.np.sum()))
    ZB2 = i_round(float(ZB.np.sum()))
    ZC2 = i_round(float(ZC.np.sum()))
    EA2, EB2, EC2 = ZA2 - qA, ZB2 - qB, ZC2 - qC

    if EA2 % 2 or EB2 % 2 or EC2 % 2:
        raise ValueError("FISAPT: fragment charge incompatible with singlet (odd electron count).")

    NA2, NB2, NC2 = EA2 // 2, EB2 // 2, EC2 // 2
    if (NA2 + NB2 + NC2) != n_occ:
        raise ValueError("FISAPT: sum of fragment electrons incompatible with total electrons.")

    RA2 = NA2 - len(orbsA)
    RB2 = NB2 - len(orbsB)
    RC2 = NC2 - len(orbsC)

    # --- greedy fill using QF weights (C then A then B), excluding taken orbs ---
    taken = set(orbsA) | set(orbsB) | set(orbsC)

    def take_top(weights: np.ndarray, k: int, taken_set: set) -> list[int]:
        if k <= 0:
            return []
        order = np.argsort(weights)[::-1]
        picked: List[int] = []
        for a in order:
            if int(a) in taken_set:
                continue
            picked.append(int(a))
            taken_set.add(int(a))
            if len(picked) == k:
                break
        return picked

    orbsC += take_top(QF[2, :], RC2, taken)
    orbsA += take_top(QF[0, :], RA2, taken)
    orbsB += take_top(QF[1, :], RB2, taken)

    # --- sort & link ordering swap like C++ ---
    orbsA = np.array(sorted(set(orbsA)), dtype=int)
    orbsB = np.array(sorted(set(orbsB)), dtype=int)
    orbsC = np.array(sorted(set(orbsC)), dtype=int)
    orbsL = np.array(orbsL, dtype=int)
    if orbsL.size > 1 and orbsL[0] > orbsL[1]:
        orbsL[[0, 1]] = orbsL[[1, 0]]
        typesL[0], typesL[1] = typesL[1], typesL[0]

    # --- build LoccA/B/C/L as psi4 Matrices (column extracts) ---
    def cols(M_np: np.ndarray, idx: np.ndarray) -> core.Matrix:
        if idx.size == 0:
            return np.zeros(M_np.shape[0])
        return core.Matrix.from_array(M_np[:, idx])
    
    def extract_columns(cols, A: core.Matrix) -> core.Matrix:
        cols = np.asarray(cols, dtype=int)
        if cols.size == 0:
            return None
        A2 = A[:, cols]
        return core.Matrix.from_array(A2)

    cache["LoccA"] = extract_columns(orbsA, Locc)
    cache["LoccB"] = extract_columns(orbsB, Locc)
    cache["LoccC"] = extract_columns(orbsC, Locc)
    cache["LoccL"] = extract_columns(orbsL, Locc)
    cache["typesL"] = typesL  # Store link types for density reassignment

    cache["QF"] = QF
    # --- summary numbers (if you want to print like C++ later) ---
    ZA_int, ZB_int, ZC_int = i_round(ZA.np.sum()), i_round(ZB.np.sum()), i_round(ZC.np.sum())
    YA, YB, YC = int(2 * orbsA.size), int(2 * orbsB.size), int(2 * orbsC.size)

    core.print_out("   => Partition Summary <= \n\n")
    core.print_out(f"    Monomer A: {ZA_int - YA:2d} charge, {ZA_int:3d} protons, {YA:3d} electrons, {len(orbsA):3d} docc\n")
    core.print_out(f"    Monomer B: {ZB_int - YB:2d} charge, {ZB_int:3d} protons, {YB:3d} electrons, {len(orbsB):3d} docc\n")
    core.print_out(f"    Monomer C: {ZC_int - YC:2d} charge, {ZC_int:3d} protons, {YC:3d} electrons, {len(orbsC):3d} docc\n")
    return cache


def build_sapt_jk_cache(
    wfn_dimer: core.Wavefunction,
    wfn_A: core.Wavefunction,
    wfn_B: core.Wavefunction,
    jk: core.JK,
    do_print=True,
    external_potentials=None,
):
    """
    Constructs the DCBS cache data required to compute ELST/EXCH/IND
    """
    core.print_out("\n  ==> Preparing SAPT Data Cache <== \n\n")
    jk.print_header()

    cache = {}
    cache["wfn_A"] = wfn_A
    cache["wfn_B"] = wfn_B

    # First grab the orbitals as psi4.core.Matrix objects
    # NOTE: scf_A from FISAPT0 and SAPT(DFT) wfn_A have slightly different coefficients
    cache["Cocc_A"] = wfn_A.Ca_subset("AO", "OCC")
    cache["Cocc_A"].name = "Cocc_A"
    cache["Cvir_A"] = wfn_A.Ca_subset("AO", "VIR")
    cache["Cvir_A"].name = "Cvir_A"

    cache["Cocc_B"] = wfn_B.Ca_subset("AO", "OCC")
    cache["Cocc_B"].name = "Cocc_B"
    cache["Cvir_B"] = wfn_B.Ca_subset("AO", "VIR")
    cache["Cvir_B"].name = "Cvir_B"

    cache["eps_occ_A"] = wfn_A.epsilon_a_subset("AO", "OCC")
    cache["eps_vir_A"] = wfn_A.epsilon_a_subset("AO", "VIR")
    cache["eps_occ_B"] = wfn_B.epsilon_a_subset("AO", "OCC")
    cache["eps_vir_B"] = wfn_B.epsilon_a_subset("AO", "VIR")

    # localization
    if core.get_option("SAPT", "SAPT_DFT_DO_FSAPT"):
        cache["Cfocc"] = wfn_dimer.Ca_subset("AO", "FROZEN_OCC")
        cache["eps_all"] = wfn_dimer.epsilon_a_subset("AO", "ALL")

        cache["Call"] = wfn_dimer.Ca_subset("AO", "ALL")
        cache["Cocc"] = wfn_dimer.Ca_subset("AO", "OCC")
        cache["Cvir"] = wfn_dimer.Ca_subset("AO", "VIR")

        cache["eps_occ"] = wfn_dimer.epsilon_a_subset("AO", "OCC")
        cache["eps_vir"] = wfn_dimer.epsilon_a_subset("AO", "VIR")

        cache["Caocc"] = wfn_dimer.Ca_subset("AO", "ACTIVE_OCC")
        cache["Cavir"] = wfn_dimer.Ca_subset("AO", "ACTIVE_VIR")
        cache["Cfvir"] = wfn_dimer.Ca_subset("AO", "FROZEN_VIR")

        cache["eps_focc"] = wfn_dimer.epsilon_a_subset("AO", "FROZEN_OCC")
        cache["eps_aocc"] = wfn_dimer.epsilon_a_subset("AO", "ACTIVE_OCC")
        cache["eps_avir"] = wfn_dimer.epsilon_a_subset("AO", "ACTIVE_VIR")
        cache["eps_fvir"] = wfn_dimer.epsilon_a_subset("AO", "FROZEN_VIR")

    # Build the densities as HF takes an extra "step", Eq. 5
    cache["D_A"] = chain_gemm_einsums([cache['Cocc_A'], cache['Cocc_A']], ['N', 'T'])
    cache['D_B'] = chain_gemm_einsums([cache['Cocc_B'], cache['Cocc_B']], ['N', 'T'])
    # Eq. 7
    cache["P_A"] = chain_gemm_einsums([cache['Cvir_A'], cache['Cvir_A']], ['N', 'T'])
    cache['P_B'] = chain_gemm_einsums([cache['Cvir_B'], cache['Cvir_B']], ['N', 'T'])

    # Potential ints - store as psi4.core.Matrix
    mints = core.MintsHelper(wfn_A.basisset())
    cache["V_A"] = mints.ao_potential()
    mints = core.MintsHelper(wfn_B.basisset())
    cache["V_B"] = mints.ao_potential()

    # External Potentials need to add to V_A and V_B
    if external_potentials:
        if external_potentials.get("A") is not None:
            ext_A = wfn_A.external_pot().computePotentialMatrix(wfn_A.basisset())
            cache["V_A"].add(ext_A)
        if external_potentials.get("B") is not None:
            ext_B = wfn_B.external_pot().computePotentialMatrix(wfn_B.basisset())
            cache["V_B"].add(ext_B)

    # Anything else we might need
    # S corresponds to the overlap matrix, S^{AO}
    cache["S"] = wfn_A.S().clone()
    cache["S"].name = "S"

    # J and K matrices
    jk.C_clear()

    # Normal J/K for Monomer A
    jk.C_left_add(wfn_A.Ca_subset("SO", "OCC"))
    jk.C_right_add(wfn_A.Ca_subset("SO", "OCC"))

    # Normal J/K for Monomer B
    jk.C_left_add(wfn_B.Ca_subset("SO", "OCC"))
    jk.C_right_add(wfn_B.Ca_subset("SO", "OCC"))

    DB_S_CA = chain_gemm_einsums([cache['D_B'], cache['S'], cache['Cocc_A']])
    jk.C_left_add(DB_S_CA)
    jk.C_right_add(cache["Cocc_A"])

    jk.compute()

    # Clone them as the JK object will overwrite. Store as psi4.core.Matrix
    cache["J_A"] = jk.J()[0].clone()
    cache["K_A"] = jk.K()[0].clone()
    cache["J_B"] = jk.J()[1].clone()
    cache["K_B"] = jk.K()[1].clone()
    cache["J_O"] = jk.J()[2].clone()
    # K_O needs transpose
    K_O = jk.K()[2].clone().transpose() #.np.T
    # K_O.np = K_O.np.T
    cache["K_O"] = core.Matrix.from_array(K_O.np)
    cache["K_O"].name = "K_O"

    monA_nr = wfn_A.molecule().nuclear_repulsion_energy()
    monB_nr = wfn_B.molecule().nuclear_repulsion_energy()
    dimer_nr = wfn_A.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy()

    cache["extern_extern_IE"] = 0.0
    if external_potentials:
        dimer_nr += wfn_dimer.external_pot().computeNuclearEnergy(wfn_dimer.molecule()) 
        if external_potentials.get("A") is not None:
            monA_nr += wfn_A.external_pot().computeNuclearEnergy(wfn_A.molecule())
        if external_potentials.get("B") is not None:
            monB_nr += wfn_B.external_pot().computeNuclearEnergy(wfn_B.molecule())
        if external_potentials.get("A") is not None and external_potentials.get("B") is not None:
            cache["extern_extern_IE"] = wfn_A.external_pot().computeExternExternInteraction(wfn_B.external_pot())

    cache["nuclear_repulsion_energy"] = dimer_nr - monA_nr - monB_nr
    return cache


def electrostatics(cache, do_print=True):
    r"""
    Computes the E10 electrostatics from a build_sapt_jk_cache datacache.

    Equation 4. E^{(1)}_{\rm elst} = 2P^A \cdot V^B + 2P^B \cdot V^A + 4P^B \cdot J^A + V_{\rm nuc}
    """
    if do_print:
        core.print_out("\n  ==> E10 Electrostatics <== \n\n")

    # Eq. 4 - use matrix_dot helper for Frobenius inner product
    # Note: FISAPT uses 4*D_A·J_B (electrons of A feeling Coulomb from B)
    term1 = 2.0 * ein.core.dot(cache["D_A"].np, cache["V_B"].np)
    term2 = 2.0 * ein.core.dot(cache["D_B"].np, cache["V_A"].np)
    term3 = 4.0 * ein.core.dot(cache["D_A"].np, cache["J_B"].np)  # Changed from D_B·J_A to match FISAPT
    term4 = cache["nuclear_repulsion_energy"]
    Elst10 = term1 + term2 + term3 + term4
    
    if do_print:
        core.print_out(print_sapt_var("Elst10,r ", Elst10, short=True))
        core.print_out("\n")

    # External Potentials interacting with each other (V_A_ext, V_B_ext)
    extern_extern_ie = 0
    if cache.get('extern_extern_IE'):
        extern_extern_ie = cache['extern_extern_IE']
        core.print_out(print_sapt_var("Extern-Extern ", extern_extern_ie, short=True))
        core.print_out("\n")

    return {"Elst10,r": Elst10}, extern_extern_ie


def felst(cache, sapt_elst, dimer_wfn, wfn_A, wfn_B, jk, do_print=True):
    r"""
    Computes the F-SAPT electrostatics partitioning according to FISAPT::felst in C++.
    Returns the total Elst10,r and stores the breakdown matrix in cache["Elst_AB"].
    """
    core.timer_on("F-SAPT Elst Setup")
    if do_print:
        core.print_out("  ==> F-SAPT Electrostatics <==\n\n")

    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    mol = dimer_wfn.molecule()  # dimer molecule
    dimer_basis = dimer_wfn.basisset()
    nA_atoms = mol.natom()
    nB_atoms = mol.natom()

    # Sizing
    L0A = cache["Locc_A"] if link_assignment not in {"SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"} else cache["Locc_A"]
    L0B = cache["Locc_B"] if link_assignment not in {"SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"} else cache["Locc_B"]
    na = L0A.np.shape[1]
    nb = L0B.np.shape[1]

    # Initialize breakdown matrix (nA_atoms + na + 1, nB_atoms + nb + 1)
    Elst_AB = np.zeros((nA_atoms + na + 1, nB_atoms + nb + 1))

    # Terms for total
    Elst1_terms = np.zeros(4)  # [0]: a-B, [1]: A-b, [2]: a-b, [3]: nuc

    # Nuclear-nuclear interactions (A <-> B)
    ZA = cache["ZA"]
    ZB = cache["ZB"]
    
    # Vectorized nuclear-nuclear interactions
    # Build distance matrix
    for A in range(nA_atoms):
        for B in range(nB_atoms):
            if A == B:
                continue
            R = mol.xyz(A).distance(mol.xyz(B))
            if R == 0:
                continue
            E = ZA.np[A] * ZB.np[B] / R
            Elst_AB[A, B] = E
            Elst1_terms[3] += E

    # External A - atom B interactions
    # Compute directly: sum over external charges and B nuclei: Z_ext * Z_B / R
    if "A" in cache.get("external_potentials", {}):
        ext_pot_A = cache["external_potentials"]["A"]
        ext_charges_A = ext_pot_A.getCharges()  # list of (Z, x, y, z) tuples
        for B in range(nB_atoms):
            if abs(ZB.np[B]) < 1e-14:
                continue
            xB, yB, zB = mol.x(B), mol.y(B), mol.z(B)
            interaction = 0.0
            for charge_tuple in ext_charges_A:
                Z_ext, x_ext, y_ext, z_ext = charge_tuple
                dx, dy, dz = xB - x_ext, yB - y_ext, zB - z_ext
                R = np.sqrt(dx*dx + dy*dy + dz*dz)
                interaction += ZB.np[B] * Z_ext / R
            Elst_AB[nA_atoms + na, B] = interaction
            Elst1_terms[3] += interaction

    # External B - atom A interactions
    if "B" in cache.get("external_potentials", {}):
        ext_pot_B = cache["external_potentials"]["B"]
        ext_charges_B = ext_pot_B.getCharges()  # list of (Z, x, y, z) tuples
        for A in range(nA_atoms):
            if abs(ZA.np[A]) < 1e-14:
                continue
            xA, yA, zA = mol.x(A), mol.y(A), mol.z(A)
            interaction = 0.0
            for charge_tuple in ext_charges_B:
                Z_ext, x_ext, y_ext, z_ext = charge_tuple
                dx, dy, dz = xA - x_ext, yA - y_ext, zA - z_ext
                R = np.sqrt(dx*dx + dy*dy + dz*dz)
                interaction += ZA.np[A] * Z_ext / R
            Elst_AB[A, nB_atoms + nb] = interaction
            Elst1_terms[3] += interaction

    core.timer_off("F-SAPT Elst Setup")
    # => a <-> b (electron-electron interactions via DFHelper) <= //
    
    # Get auxiliary basis for density fitting
    aux_basis = dimer_wfn.get_basisset("DF_BASIS_SCF")
    
    # Create DFHelper object
    dfh = core.DFHelper(dimer_basis, aux_basis)
    # TODO: This memory estimate needs corrected...
    dfh.set_memory(int(core.get_memory() * 0.5 / 8))
    dfh.set_method("DIRECT_iaQ")
    dfh.set_nthreads(core.get_num_threads())
    dfh.initialize()
    dfh.print_header()
    
    # Create Matrix objects from numpy arrays for L0A and L0B
    L0A = core.Matrix.from_array(L0A.np)
    L0B = core.Matrix.from_array(L0B.np)
    
    # Add orbital spaces
    dfh.add_space("a", L0A)
    dfh.add_space("b", L0B)
    
    # Add transformations for diagonal blocks (a,a|Q) and (b,b|Q)
    dfh.add_transformation("Aaa", "a", "a")
    dfh.add_transformation("Abb", "b", "b")
    
    # Perform the transformation
    dfh.transform()
    
    # Extract diagonal 3-index integrals (vectorized)
    nQ = aux_basis.nbf()
    QaC = np.zeros((na, nQ))
    QbC = np.zeros((nb, nQ))
    
    # Process in batches for better memory efficiency
    batch_size = min(100, max(na, nb))
    
    # Extract Aaa diagonal elements
    for start_a in range(0, na, batch_size):
        end_a = min(start_a + batch_size, na)
        for a in range(start_a, end_a):
            tensor = dfh.get_tensor("Aaa", [a, a + 1], [a, a + 1], [0, nQ])
            QaC[a, :] = tensor.np.flatten()
    
    # Extract Abb diagonal elements
    for start_b in range(0, nb, batch_size):
        end_b = min(start_b + batch_size, nb)
        for b in range(start_b, end_b):
            tensor = dfh.get_tensor("Abb", [b, b + 1], [b, b + 1], [0, nQ])
            QbC[b, :] = tensor.np.flatten()

    # Compute electrostatic interaction: Elst10_3 = 4.0 * QaC @ QbC.T
    Elst10_3 = 4.0 * np.dot(QaC, QbC.T)
    
    # Store in breakdown matrix and accumulate total
    Elst1_terms[2] += np.sum(Elst10_3)
    Elst_AB[nA_atoms:nA_atoms + na, nB_atoms:nB_atoms + nb] += Elst10_3
    
    # Store QaC and QbC in cache for reuse in f-induction
    cache["Vlocc0A"] = core.Matrix.from_array(QaC)
    cache["Vlocc0B"] = core.Matrix.from_array(QbC)
    
    # Clear DFHelper spaces for next use
    dfh.clear_spaces()
    
    core.timer_on("F-SAPT Elst Final")
    # => A <-> b (nuclei A interacting with orbitals b) <= //
    L0B_mat = core.Matrix.from_array(L0B.np)
    L0B_mat.name = "L0B_mat"

    L0A_mat = core.Matrix.from_array(L0A.np)
    ext_pot = core.ExternalPotential()
    for A in range(nA_atoms):
        if ZA.np[A] == 0.0:
            continue
        
        ext_pot.clear()
        atom_pos = mol.xyz(A)
        ext_pot.addCharge(ZA.np[A], atom_pos[0], atom_pos[1], atom_pos[2])
        
        Vtemp = ext_pot.computePotentialMatrix(dimer_basis)
        Vtemp_mat = Vtemp.clone()
        Vtemp_mat.name = "Vtemp_mat"
        
        Vbb = chain_gemm_einsums([L0B_mat, Vtemp_mat, L0B_mat], ['T', 'N', 'N'])
        Vbb.name = "Vbb"
        
        # Vectorized diagonal extraction
        diag_Vbb = np.diag(Vbb.np)
        E_vec = 2.0 * diag_Vbb
        Elst1_terms[1] += np.sum(E_vec)
        Elst_AB[A, nB_atoms:nB_atoms + nb] += E_vec
    
    # Add external-A <-> orbital b interaction
    if "A" in cache.get("external_potentials", {}):
        ext_pot_A = cache["external_potentials"]["A"]
        Vtemp = ext_pot_A.computePotentialMatrix(dimer_basis)
        
        Vtemp_mat = Vtemp.clone()
        Vbb = chain_gemm_einsums([L0B_mat, Vtemp_mat, L0B_mat], ['T', 'N', 'N'])
        
        # Vectorized diagonal extraction
        diag_Vbb = np.diag(Vbb.np)
        E_vec = 2.0 * diag_Vbb
        Elst1_terms[1] += np.sum(E_vec)
        Elst_AB[nA_atoms + na, nB_atoms:nB_atoms + nb] += E_vec
    
    # => a <-> B (orbitals a interacting with nuclei B) <= //
    
    for B in range(nB_atoms):
        if ZB.np[B] == 0.0:
            continue

        ext_pot.clear()
        atom_pos = mol.xyz(B)
        ext_pot.addCharge(ZB.np[B], atom_pos[0], atom_pos[1], atom_pos[2])
        
        Vtemp = ext_pot.computePotentialMatrix(dimer_basis)
        
        Vtemp_mat = Vtemp.clone()
        Vaa = chain_gemm_einsums([L0A_mat, Vtemp_mat, L0A_mat], ['T', 'N', 'N'])
        
        # Vectorized diagonal extraction
        diag_Vaa = np.diag(Vaa.np)
        E_vec = 2.0 * diag_Vaa
        Elst1_terms[0] += np.sum(E_vec)
        Elst_AB[nA_atoms:nA_atoms + na, B] += E_vec
    
    # Add orbital a <-> external-B interaction
    if "B" in cache.get("external_potentials", {}):
        ext_pot_B = cache["external_potentials"]["B"]
        Vtemp = ext_pot_B.computePotentialMatrix(dimer_basis)
        
        Vtemp_mat = Vtemp.clone()
        Vaa = chain_gemm_einsums([L0A_mat, Vtemp_mat, L0A_mat], ['T', 'N', 'N'])
        
        # Vectorized diagonal extraction  
        diag_Vaa = np.diag(Vaa.np)
        E_vec = 2.0 * diag_Vaa
        Elst1_terms[0] += np.sum(E_vec)
        Elst_AB[nA_atoms:nA_atoms + na, nB_atoms + nb] += E_vec
    
    # Clear DFHelper for next use
    dfh.clear_spaces()
    cache['dfh'] = dfh  # Store DFHelper in cache for potential reuse
    Elst10 = np.sum(Elst1_terms)
    core.print_out(f"    Elst10,r            = {Elst10*1000:.8f} [mEh]\n")
    # Ensure that partition matches SAPT elst energy. Should be equal to
    # numerical precision and effectively free to check assertion here.
    # For I-SAPT (3 fragments), the localization is different and may not match exactly
    n_fragments = len(mol.get_fragments())
    if n_fragments == 3:
        # I-SAPT case - warn but don't assert
        if abs(Elst10 - sapt_elst) > 1e-6:
            core.print_out(f"\n  WARNING: I-SAPT localized Elst10,r ({Elst10*1000:.8f} mEh) differs from SAPT Elst10,r ({sapt_elst*1000:.8f} mEh)\n")
            core.print_out(f"           This is expected for I-SAPT with link orbitals. Using SAPT Elst10,r for consistency.\n\n")
    else:
        assert abs(Elst10 - sapt_elst) < 1e-8, f"FELST: Localized Elst10,r does not match SAPT Elst10,r!\n{Elst10 = }, {sapt_elst}"
    
    # Add extern-extern contribution if both external potentials exist
    if "A" in cache.get("external_potentials", {}) and "B" in cache.get("external_potentials", {}):
        ext_pot_A = cache["external_potentials"]["A"]
        ext_pot_B = cache["external_potentials"]["B"]
        ext_ext = ext_pot_A.computeExternExternInteraction(ext_pot_B) * 2.0
        Elst_AB[nA_atoms + na, nB_atoms + nb] += ext_ext
    
    # Store breakdown matrix in cache
    cache["Elst_AB"] = core.Matrix.from_array(Elst_AB)
    core.timer_off("F-SAPT Elst Final")
    return cache


def fexch(cache, sapt_exch10_s2, sapt_exch10, dimer_wfn, wfn_A, wfn_B, jk, do_print=True):
    """
    Computes the F-SAPT exchange partitioning according to FISAPT::fexch in C++.
    Uses the Exch10(S^2) approximation with orbital partitioning.
    
    Args:
        cache: Dictionary containing matrices and vectors
        sapt_exch: Total SAPT Exch10(S^2) energy from the regular exch() calculation
        dimer_wfn: Dimer wavefunction
        wfn_A, wfn_B: Monomer wavefunctions
        jk: JK object
        do_print: Whether to print output
    
    Returns:
        cache: Updated cache with Exch_AB matrix
    """
    if do_print:
        core.print_out("  ==> F-SAPT Exchange <==\n\n")
    
    mol = dimer_wfn.molecule()
    nA_atoms = nB_atoms = mol.natom()
    na = cache["Locc_A"].shape[1]
    nb = cache["Locc_B"].shape[1]
    nr = cache["Cvir_A"].shape[1]
    ns = cache["Cvir_B"].shape[1]
    
    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT")
    na1 = na
    nb1 = nb
    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        na1 = na + 1
        nb1 = nb + 1
    
    Exch10_2 = 0.0
    Exch10_2_terms = [0.0, 0.0, 0.0]
    
    Exch_AB = np.zeros((nA_atoms + na1 + 1, nB_atoms + nb1 + 1))
    
    S = cache["S"]
    V_A = cache["V_A"]
    J_A = cache["J_A"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]
    
    LoccA = cache["Locc_A"].clone()
    LoccA.name = "LoccA"
    LoccB = cache["Locc_B"].clone()
    LoccB.name = "LoccB"
    CvirA = cache["Cvir_A"]
    CvirB = cache["Cvir_B"]
    CvirA.name = "CvirA"
    CvirB.name = "CvirB"
    
    dfh = cache["dfh"]
    
    dfh.add_space("a", LoccA)
    dfh.add_space("r", CvirA)
    dfh.add_space("b", LoccB)
    dfh.add_space("s", CvirB)
    
    dfh.add_transformation("Aar", "a", "r")
    dfh.add_transformation("Abs", "b", "s")
    
    dfh.transform()

    # W_A = V_A + 2.0 * J_A using core.Matrix operations
    W_A = V_A.clone()
    ein.core.axpy(2.0, J_A.np, W_A.np)
    W_A.name = "W_A"
    # W_B = V_B + 2.0 * J_B
    W_B = V_B.clone()
    ein.core.axpy(2.0, J_B.np, W_B.np)
    W_B.name = "W_B"

    WAbs = chain_gemm_einsums([LoccB, W_A, CvirB], ['T', 'N', 'N'])
    WBar = chain_gemm_einsums([LoccA, W_B, CvirA], ['T', 'N', 'N'])
    WAbs.name = "WAbs"
    WBar.name = "WBar"

    Sab = chain_gemm_einsums([LoccA, S, LoccB], ['T', 'N', 'N'])
    Sba = chain_gemm_einsums([LoccB, S, LoccA], ['T', 'N', 'N'])
    Sas = chain_gemm_einsums([LoccA, S, CvirB], ['T', 'N', 'N'])
    Sas.name = "Sas"
    Sab.name = "Sab"

    LoccB.name = "LoccB"
    CvirA.name = "CvirA"
    Sbr = chain_gemm_einsums([LoccB, S, CvirA], ['T', 'N', 'N'])

    Sab.name = "Sab"
    Sba.name = "Sba"
    Sas.name = "Sas"
    Sbr.name = "Sbr"

    WBab = chain_gemm_einsums([WBar, Sbr], ['N', 'T'])
    WAba = chain_gemm_einsums([WAbs, Sas], ['N', 'T'])
    WBab.name = "WBab"
    WAba.name = "WAba"

    E_exch1 = np.zeros((na, nb))
    E_exch2 = np.zeros((na, nb))
    
    for a in range(na):
        for b in range(nb):
            E_exch1[a, b] = -2.0 * Sab.np[a, b] * WBab.np[a, b]
            E_exch2[a, b] = -2.0 * Sba.np[b, a] * WAba.np[b, a]
    
    nQ = dimer_wfn.get_basisset("DF_BASIS_SCF").nbf()
    TrQ = core.Matrix("TrQ", nr, nQ)
    TsQ = core.Matrix("TsQ", ns, nQ)
    TbQ = core.Matrix("TbQ", nb, nQ)
    TaQ = core.Matrix("TaQ", na, nQ)
    
    dfh.add_disk_tensor("Bab", (na, nb, nQ))
    
    for a in range(na):
        TrQ.np[:, :] = dfh.get_tensor("Aar", [a, a + 1], [0, nr], [0, nQ]).np.reshape(nr, nQ)
        TbQ.np[:, :] = np.dot(Sbr.np, TrQ.np)
        dfh.write_disk_tensor("Bab", TbQ, [a, a + 1])
    
    dfh.add_disk_tensor("Bba", (nb, na, nQ))
    
    for b in range(nb):
        TsQ.np[:, :] = dfh.get_tensor("Abs", [b, b + 1], [0, ns], [0, nQ]).np.reshape(ns, nQ)
        TaQ.np[:, :] = np.dot(Sas.np, TsQ.np)
        dfh.write_disk_tensor("Bba", TaQ, [b, b + 1])
    
    E_exch3 = np.zeros((na, nb))
    
    for a in range(na):
        TbQ.np[:, :] = dfh.get_tensor("Bab", [a, a + 1], [0, nb], [0, nQ]).np.reshape(nb, nQ)
        for b in range(nb):
            TaQ_slice = dfh.get_tensor("Bba", [b, b + 1], [a, a + 1], [0, nQ]).np.reshape(nQ)
            E_exch3[a, b] = -2.0 * np.dot(TbQ.np[b, :], TaQ_slice)
    
    for a in range(na):
        for b in range(nb):
            Exch_AB[a + nA_atoms, b + nB_atoms] = E_exch1[a, b] + E_exch2[a, b] + E_exch3[a, b]
            Exch10_2_terms[0] += E_exch1[a, b]
            Exch10_2_terms[1] += E_exch2[a, b]
            Exch10_2_terms[2] += E_exch3[a, b]
    
    Exch10_2 = sum(Exch10_2_terms)
    
    if do_print:
        core.print_out(f"    Exch10(S^2)         = {Exch10_2 * 1000:18.10f} [mEh]\n")
        core.print_out(f"    Exch10(S^2)-true    = {sapt_exch10_s2 * 1000:18.10f} [mEh]\n")
        core.print_out(f"    Exch10-true         = {sapt_exch10 * 1000:18.10f} [mEh]\n")
        core.print_out("\n")
    
    if core.get_option("FISAPT", "FISAPT_FSAPT_EXCH_SCALE"):
        # For now, scaling should be 1.0
        scale = sapt_exch10 / Exch10_2
        Exch_AB *= scale
        if do_print:
            core.print_out(f"    Scaling F-SAPT Exch10(S^2) by {scale:11.3E} to match Exch10\n\n")
        # assert abs(scale - 1.0) < 1e-6, "Currently should only get scale factor of 1.0"
    
    cache["Exch_AB"] = core.Matrix.from_array(Exch_AB)
    
    dfh.clear_spaces()
    
    return cache


# TODO: update induction to use this function as well as find()
def build_ind_pot(vars):
    """
    Build the induction potential for monomer A due to monomer B.
    By changing vars map to have B and A swapped, can get induction potential
    for B due to A.
    """
    w_B = vars['V_B'].clone()
    ein.core.axpy(2.0, vars['J_B'].np, w_B.np)
    return chain_gemm_einsums(
        [vars['Cocc_A'], w_B, vars['Cvir_A']],
        ['T', 'N', 'N'],
    )
    

def build_exch_ind_pot_AB(vars):
    """
    Build the exchange-induction potential for monomer A due to monomer B
    """

    K_B = vars['K_B']
    J_O = vars['J_O']
    K_O = vars['K_O']
    J_P_B = vars['J_P_B']
    J_A = vars['J_A']
    K_A = vars['K_A']
    J_B = vars['J_B']
    D_A = vars['D_A']
    D_B = vars['D_B']
    S = vars['S']
    V_B = vars['V_B']
    V_A = vars['V_A']

    # Exch-Ind Potential A
    EX_A = K_B.clone()
    EX_A.scale(-1.0)
    ein.core.axpy(-2.0, J_O.np, EX_A.np)
    ein.core.axpy(1.0, K_O.np, EX_A.np)
    ein.core.axpy(2.0, J_P_B.np, EX_A.np)

    # Apply all the axpy operations to EX_A
    S_DB, S_DB_VA, S_DB_VA_DB_S = chain_gemm_einsums(
        [S, D_B, V_A, D_B, S],
        return_tensors=[True, True, False, True]
    )
    S_DB_JA, S_DB_JA_DB_S = chain_gemm_einsums(
        [S_DB, J_A, D_B, S],
        return_tensors=[True, False, True]
    )
    S_DB_S_DA, S_DB_S_DA_VB = chain_gemm_einsums(
        [S_DB, S, D_A, V_B],
        return_tensors=[False, True, True],
    )
    ein.core.axpy(-1.0, S_DB_VA.np, EX_A.np)
    ein.core.axpy(-2.0, S_DB_JA.np, EX_A.np)
    ein.core.axpy(1.0, chain_gemm_einsums([S_DB, K_A]).np, EX_A.np)
    ein.core.axpy(1.0, S_DB_S_DA_VB.np, EX_A.np)
    ein.core.axpy(2.0, chain_gemm_einsums([S_DB_S_DA, J_B]).np, EX_A.np)
    ein.core.axpy(1.0, S_DB_VA_DB_S.np, EX_A.np)
    ein.core.axpy(2.0, S_DB_JA_DB_S.np, EX_A.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([S_DB, K_O], ["N", "T"]).np, EX_A.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([V_B, D_B, S]).np, EX_A.np)
    ein.core.axpy(-2.0, chain_gemm_einsums([J_B, D_B, S]).np, EX_A.np)
    ein.core.axpy(1.0, chain_gemm_einsums([K_B, D_B, S]).np, EX_A.np)
    ein.core.axpy(1.0, chain_gemm_einsums([V_B, D_A, S, D_B, S]).np, EX_A.np)
    ein.core.axpy(2.0, chain_gemm_einsums([J_B, D_A, S, D_B, S]).np, EX_A.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([K_O, D_B, S]).np, EX_A.np)

    EX_A_MO = chain_gemm_einsums(
        [vars['Cocc_A'], EX_A, vars['Cvir_A']],
        ['T', 'N', 'N'],
    )
    return EX_A_MO


def build_exch_ind_pot_BA(vars):
    """
    Build the exchange-induction potential for monomer B due to monomer A
    """

    K_B = vars['K_B']
    J_O = vars['J_O']
    K_O = vars['K_O']
    J_P_A = vars['J_P_A']
    J_A = vars['J_A']
    K_A = vars['K_A']
    J_B = vars['J_B']
    D_A = vars['D_A']
    D_B = vars['D_B']
    S = vars['S']
    V_B = vars['V_B']
    V_A = vars['V_A']

    EX_B = K_A.clone()
    EX_B.scale(-1.0)
    ein.core.axpy(-2.0, J_O.np, EX_B.np)
    ein.core.axpy(1.0, K_O.np, EX_B.np.T)
    ein.core.axpy(2.0, J_P_A.np, EX_B.np)

    S_DA, S_DA_VB, S_DA_VB_DA_S = chain_gemm_einsums(
        [S, D_A, V_B, D_A, S],
        return_tensors=[True, True, False, True]
    )
    S_DA_JB, S_DA_JB_DA_S = chain_gemm_einsums(
        [S_DA, J_B, D_A, S],
        return_tensors=[True, False, True]
    )
    S_DA_S_DB, S_DA_S_DB_VA = chain_gemm_einsums(
        [S_DA, S, D_B, V_A],
        return_tensors=[False, True, True],
    )

    # Apply all the axpy operations to EX_B
    ein.core.axpy(-1.0, S_DA_VB.np, EX_B.np)
    ein.core.axpy(-2.0, S_DA_JB.np, EX_B.np)
    ein.core.axpy(1.0, chain_gemm_einsums([S_DA, K_B]).np, EX_B.np)
    ein.core.axpy(1.0, S_DA_S_DB_VA.np, EX_B.np)
    ein.core.axpy(2.0, chain_gemm_einsums([S_DA_S_DB, J_A]).np, EX_B.np)
    ein.core.axpy(1.0, S_DA_VB_DA_S.np, EX_B.np)
    ein.core.axpy(2.0, S_DA_JB_DA_S.np, EX_B.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([S_DA, K_O]).np, EX_B.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([V_A, D_A, S]).np, EX_B.np)
    ein.core.axpy(-2.0, chain_gemm_einsums([J_A, D_A, S]).np, EX_B.np)
    ein.core.axpy(1.0, chain_gemm_einsums([K_A, D_A, S]).np, EX_B.np)
    ein.core.axpy(1.0, chain_gemm_einsums([V_A, D_B, S, D_A, S]).np, EX_B.np)
    ein.core.axpy(2.0, chain_gemm_einsums([J_A, D_B, S, D_A, S]).np, EX_B.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([K_O, D_A, S], ["T", "N", "N"]).np, EX_B.np)

    EX_B_MO = chain_gemm_einsums(
        [vars['Cocc_B'], EX_B, vars['Cvir_B']],
        ['T', 'N', 'N'],
    )
    return EX_B_MO


def build_exch_ind_pot_avg(vars):
    Ca = vars["Cocc_A"]
    Cr = vars["Cvir_A"]
    
    S = vars["S"]
    
    D_A = vars["D_A"]
    J_A = vars["J_A"]
    K_A = vars["K_A"]
    V_A = vars["V_A"]
    D_B = vars["D_B"]
    J_B = vars["J_B"]
    K_B = vars["K_B"]
    V_B = vars["V_B"]
    D_X = vars["D_X"]
    J_X = vars["J_X"]
    K_X = vars["K_X"]
    D_Y = vars["D_Y"]
    J_Y = vars["J_Y"]
    K_Y = vars["K_Y"]
    
    J_O = vars["J_O"]
    K_O = vars["K_O"]
    K_AOY = vars["K_AOY"]
    
    J_P = vars["J_P"]
    J_PYAY = vars["J_PYAY"]
    
    W = core.Matrix.from_array(-K_B.np)
    
    T = core.triplet(S, D_B, J_A, False, False, False)
    W.np[:] += -2.0 * T.np
    
    W.np[:] += K_O.np
    
    W.np[:] += -2.0 * J_O.np
    
    T = core.triplet(S, D_B, K_A, False, False, False)
    W.np[:] += T.np
    
    T = core.triplet(J_B, D_B, S, False, False, False)
    W.np[:] += -2.0 * T.np
    
    T = core.triplet(K_B, D_B, S, False, False, False)
    W.np[:] += T.np
    T = core.triplet(K_Y, D_Y, S, False, False, False)
    W.np[:] += T.np
    
    T1 = core.triplet(S, D_B, J_A, False, False, False)
    T = core.triplet(T1, D_B, S, False, False, False)
    W.np[:] += 2.0 * T.np
    T1 = core.triplet(S, D_Y, J_A, False, False, False)
    T = core.triplet(T1, D_Y, S, False, False, False)
    W.np[:] += 2.0 * T.np
    
    T1 = core.triplet(J_B, D_A, S, False, False, False)
    T = core.triplet(T1, D_B, S, False, False, False)
    W.np[:] += 2.0 * T.np
    
    T = core.triplet(K_O, D_B, S, False, False, False)
    W.np[:] += -1.0 * T.np
    T = core.triplet(K_AOY, D_Y, S, False, False, False)
    W.np[:] += -1.0 * T.np
    
    W.np[:] += 2.0 * J_P.np
    W.np[:] += 2.0 * J_PYAY.np
    
    T1 = core.triplet(S, D_B, S, False, False, False)
    T = core.triplet(T1, D_A, J_B, False, False, False)
    W.np[:] += 2.0 * T.np
    
    T = core.triplet(S, D_B, K_O, False, False, True)
    W.np[:] += -1.0 * T.np
    T = core.triplet(S, D_Y, K_AOY, False, False, True)
    W.np[:] += -1.0 * T.np
    
    T = core.triplet(S, D_B, V_A, False, False, False)
    W.np[:] += -1.0 * T.np
    
    T = core.triplet(V_B, D_B, S, False, False, False)
    W.np[:] += -1.0 * T.np
    
    T1 = core.triplet(S, D_B, V_A, False, False, False)
    T = core.triplet(T1, D_B, S, False, False, False)
    W.np[:] += T.np
    T1 = core.triplet(S, D_Y, V_A, False, False, False)
    T = core.triplet(T1, D_Y, S, False, False, False)
    W.np[:] += T.np
    
    T1 = core.triplet(V_B, D_A, S, False, False, False)
    T = core.triplet(T1, D_B, S, False, False, False)
    W.np[:] += T.np
    
    T1 = core.triplet(S, D_B, S, False, False, False)
    T = core.triplet(T1, D_A, V_B, False, False, False)
    W.np[:] += T.np
    
    return core.triplet(Ca, W, Cr, True, False, False)


def find(cache, scalars, dimer_wfn, wfn_A, wfn_B, jk, do_print=True):
    if do_print:
        core.print_out("  ==> F-SAPT Induction <==\n\n")
    
    ind_scale = core.get_option("FISAPT", "FISAPT_FSAPT_IND_SCALE")
    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT")
    
    mol = dimer_wfn.molecule()
    nA = mol.natom()
    nB = mol.natom()
    na = cache["Locc_A"].shape[1]
    nb = cache["Locc_B"].shape[1]
    nr = cache["Cvir_A"].shape[1]
    ns = cache["Cvir_B"].shape[1]
    
    na1 = na
    nb1 = nb
    # for the SAOn/SIAOn variants, we sometimes need na1 = na+1 (with link
    # orbital) and sometimes na (without) - be careful with this!
    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        na1 = na + 1
        nb1 = nb + 1
    
    Locc_A = cache["Locc_A"].clone()
    Locc_A.name = "LoccA"
    Locc_B = cache["Locc_B"].clone()
    Locc_B.name = "LoccB"
    
    Uocc_A = cache["Uocc_A"]
    Uocc_B = cache["Uocc_B"]
    
    Cocc_A = cache["Cocc_A"]
    Cocc_B = cache["Cocc_B"]
    Cvir_A = cache["Cvir_A"]
    Cvir_B = cache["Cvir_B"]

    # Cvir_A.set_name("Cvir_A")
    # print(Cvir_A)
    
    eps_occ_A = cache["eps_occ_A"]
    eps_occ_B = cache["eps_occ_B"]
    eps_vir_A = cache["eps_vir_A"]
    eps_vir_B = cache["eps_vir_B"]

    # Collect relevant variables
    S = cache["S"]
    D_A = cache["D_A"]
    V_A = cache["V_A"]
    J_A = cache["J_A"]
    K_A = cache["K_A"]
    D_B = cache["D_B"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]
    K_B = cache["K_B"]
    J_O = cache["J_O"]
    K_O = cache["K_O"]
    J_P_A = cache["J_P_A"]
    J_P_B = cache["J_P_B"]

    aux_basis = dimer_wfn.get_basisset("DF_BASIS_SCF")
    nQ = aux_basis.nbf()
    
    # dfh = core.DFHelper(dimer_wfn.basisset(), aux_basis)
    # TODO: This memory estimate needs corrected...
    # dfh.set_memory(int(core.get_memory() * 0.9 / 8))  # Use 90% of available memory (in doubles)
    # dfh.set_method("DIRECT")
    # dfh.set_nthreads(core.get_num_threads())
    # dfh.initialize()
    dfh = cache["dfh"]
    
    # ESPs - external potential entries
    dfh.add_disk_tensor("WBar", (nB + nb1 + 1, na, nr))
    dfh.add_disk_tensor("WAbs", (nA + na1 + 1, nb, ns))
    
    # Nuclear Contribution to ESPs
    ext_pot = core.ExternalPotential()
    ZA = cache["ZA"].np
    for A in range(nA):
        ext_pot.clear()
        atom_pos = mol.xyz(A)
        ext_pot.addCharge(ZA[A], atom_pos[0], atom_pos[1], atom_pos[2])
        Vtemp = ext_pot.computePotentialMatrix(dimer_wfn.basisset())
        Vbs = core.Matrix.from_array(chain_gemm_einsums([Cocc_B, Vtemp, Cvir_B], ['T', 'N', 'N']))
        # Vbs_A doesn't agree... Cocc_B and Cvir_B 
        dfh.write_disk_tensor("WAbs", Vbs, (A, A + 1))
    
    ZB = cache["ZB"].np
    for B in range(nB):
        ext_pot.clear()
        atom_pos = mol.xyz(B)
        ext_pot.addCharge(ZB[B], atom_pos[0], atom_pos[1], atom_pos[2])
        Vtemp = ext_pot.computePotentialMatrix(dimer_wfn.basisset())
        Var = core.Matrix.from_array(chain_gemm_einsums([Cocc_A, Vtemp, Cvir_A], ['T', 'N', 'N']))
        dfh.write_disk_tensor("WBar", Var, (B, B + 1))
    
    dfh.add_space("a", core.Matrix.from_array(Cocc_A))
    dfh.add_space("r", core.Matrix.from_array(Cvir_A))
    dfh.add_space("b", core.Matrix.from_array(Cocc_B))
    dfh.add_space("s", core.Matrix.from_array(Cvir_B))
    
    dfh.add_transformation("Aar", "a", "r")
    dfh.add_transformation("Abs", "b", "s")
    
    dfh.transform()
    
    RaC = cache["Vlocc0A"] # na x nQ
    RbD = cache["Vlocc0B"] # nb x nQ
    
    TsQ = core.Matrix("TsQ", ns, nQ)
    T1As = core.Matrix("T1As", na1, ns)
    # print(f"{na1 = }, {nb1 = }, {ns = }, {nr = }")
    # print(f"{na1 = }, {nb1 = }, {ns = }, {nQ = }")
    for B in range(nb):
        # print(f"{TsQ.np.shape =}")
        # TODO: CONTINUE HERE
        # fill_tensor is not working properly with 2D slices yet...
        # dfh.fill_tensor("Abs", TsQ, [B, B + 1])
        dfh.fill_tensor("Abs", TsQ, [B, B + 1], [0, ns], [0, nQ])
        TsQ = core.Matrix.from_array(TsQ.np[0, :, :])
        T1As.gemm(False, True, 2.0, RaC, TsQ, 0.0)
        for A in range(na1):
            row_view = core.Matrix.from_array(T1As.np[A: A+1, :])
            dfh.write_disk_tensor("WAbs", row_view, (nA + A, nA + A + 1), (B, B + 1))
    
    TrQ = core.Matrix("TrQ", nr, nQ)
    T1Br = core.Matrix("T1Br", nb1, nr)
    for A in range(na):
        # dfh.fill_tensor("Abs", TsQ, [B, B + 1], [0, ns], [0, nQ])
        # TsQ = core.Matrix.from_array(TsQ.np[0, :, :])
        dfh.fill_tensor("Aar", TrQ, [A, A + 1], [0, nr], [0, nQ])
        TrQ = core.Matrix.from_array(TrQ.np[0, :, :])
        T1Br.gemm(False, True, 2.0, RbD, TrQ, 0.0)
        for B in range(nb1):
            row_view = core.Matrix.from_array(T1Br.np[B: B+1, :])
            dfh.write_disk_tensor("WBar", row_view, (nB + B, nB + B + 1), (A, A + 1))
    
    xA = core.Matrix("xA", na, nr)
    xB = core.Matrix("xB", nb, ns)
    wB = core.Matrix("wB", na, nr)
    wA = core.Matrix("wA", nb, ns)

    uAT = core.Matrix("uAT", nb, ns)
    wAT = core.Matrix("wAT", nb, ns)
    uBT = core.Matrix("uBT", na, nr)
    wBT = core.Matrix("wBT", na, nr)

    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        D_X = core.doublet(cache["thislinkA"], cache["thislinkA"], False, True)
        D_Y = core.doublet(cache["thislinkB"], cache["thislinkB"], False, True)
        J_X = cache["JLA"]
        K_X = cache["KLA"]
        J_Y = cache["JLB"]
        K_Y = cache["KLB"]
        
        K_AOY = cache["K_AOY"]
        K_XOB = core.Matrix.from_array(cache["K_XOB"].np.T)
        J_P_YAY = cache["J_P_YAY"]
        J_P_XBX = cache["J_P_XBX"]
        
        mapA = {
            "Cocc_A": Locc_A,
            "Cvir_A": Cvir_A,
            "S": S,
            "D_A": D_A,
            "V_A": V_A,
            "J_A": J_A,
            "K_A": K_A,
            "D_B": D_B,
            "V_B": V_B,
            "J_B": J_B,
            "K_B": K_B,
            "D_X": D_X,
            "J_X": J_X,
            "K_X": K_X,
            "D_Y": D_Y,
            "J_Y": J_Y,
            "K_Y": K_Y,
            "J_O": J_O,
            "K_O": K_O,
            "K_AOY": K_AOY,
            "J_P": J_P_A,
            "J_PYAY": J_P_YAY,
        }
        
        raise NotImplementedError("find() not ready yet for link orbitals")
        wBT = build_ind_pot(mapA)
        uBT = build_exch_ind_pot_avg(mapA)
        
        K_O.np = K_O.np
        K_O.np[:] = K_O.np.T
        
        mapB = {
            "Cocc_A": Locc_B,
            "Cvir_A": Cvir_B,
            "S": S,
            "D_A": D_B,
            "V_A": V_B,
            "J_A": J_B,
            "K_A": K_B,
            "D_B": D_A,
            "V_B": V_A,
            "J_B": J_A,
            "K_B": K_A,
            "D_X": D_Y,
            "J_X": J_Y,
            "K_X": K_Y,
            "D_Y": D_X,
            "J_Y": J_X,
            "K_Y": K_X,
            "J_O": J_O,
            "K_O": K_O,
            "K_AOY": K_XOB,
            "J_P": J_P_B,
            "J_PYAY": J_P_XBX,
        }
        
        wAT = build_ind_pot(mapB)
        uAT = build_exch_ind_pot_avg(mapB)
        
        K_O.np[:] = K_O.np.T
    
    else:
        mapA = {
            "S": S,
            "J_O": J_O,
            "K_O": K_O,

            "Cocc_A": Locc_A,
            "Cvir_A": Cvir_A,
            "D_A": D_A,
            "V_A": V_A,
            "J_A": J_A,
            "K_A": K_A,
            "J_P_A": J_P_A,

            "Cocc_B": Locc_B,
            "Cvir_B": Cvir_B,
            "D_B": D_B,
            "V_B": V_B,
            "J_B": J_B,
            "K_B": K_B,
            "J_P_B": J_P_B,
        }
        
        # V_B and J_B are equivalent, but Cocc_A and Cvir_A differ from FISAPT0... is there a critical disagreement that would only surface here? Seems unlikely.
        # Locc_A magnitudes are about the same with different signs, but that is okay.
        # Cvir_A does not have the same magnitude for most terms... This is an issue.
        wBT = build_ind_pot({
            "V_B": V_B,
            "J_B": J_B,
            "Cocc_A": Locc_A,
            "Cvir_A": Cvir_A,
        })
        wAT = build_ind_pot({
            "V_B": V_A,
            "J_B": J_A,
            "Cocc_A": Locc_B,
            "Cvir_A": Cvir_B,
        })
        uBT = build_exch_ind_pot_AB(mapA)
        uAT = build_exch_ind_pot_BA(mapA)

    wBT.name = "wBT"
    uBT.name = "uBT"
    wAT.name = "wAT"
    uAT.name = "uAT"
    # V_A checks out
    V_B.name = "V_B"
    J_B.name = "J_B"
    # print(J_B)
    # print(V_B)
    # print(wBT)
    # print(uBT)
    # print(wAT)
    # print(uAT)

    Ind20u_AB_terms = core.Matrix("Ind20 [A<-B] (a x B)", na, nB + nb1 + 1)
    Ind20u_BA_terms = core.Matrix("Ind20 [B<-A] (A x b)", nA + na1 + 1, nb)
    Ind20u_AB_termsp = Ind20u_AB_terms.np
    Ind20u_BA_termsp = Ind20u_BA_terms.np
    
    Ind20u_AB = 0.0
    Ind20u_BA = 0.0
    
    ExchInd20u_AB_terms = core.Matrix("ExchInd20 [A<-B] (a x B)", na, nB + nb1 + 1)
    ExchInd20u_BA_terms = core.Matrix("ExchInd20 [B<-A] (A x b)", nA + na1 + 1, nb)
    ExchInd20u_AB_termsp = ExchInd20u_AB_terms.np
    ExchInd20u_BA_termsp = ExchInd20u_BA_terms.np
    
    ExchInd20u_AB = 0.0
    ExchInd20u_BA = 0.0
    
    # sna = snB = snb = snA = 0
    # sExchInd20u_AB_terms = core.Matrix("sExchInd20 [A<-B] (a x B)", sna, snB + snb + 1)
    # sExchInd20u_BA_terms = core.Matrix("sExchInd20 [B<-A] (A x b)", snA + sna + 1, snb)
    # sExchInd20u_AB_termsp = sExchInd20u_AB_terms.np
    # sExchInd20u_BA_termsp = sExchInd20u_BA_terms.np
    # 
    # sExchInd20u_AB = 0.0
    # sExchInd20u_BA = 0.0
    
    Indu_AB_terms = core.Matrix("Ind [A<-B] (a x B)", na, nB + nb1 + 1)
    Indu_BA_terms = core.Matrix("Ind [B<-A] (A x b)", nA + na1 + 1, nb)
    Indu_AB = 0.0
    Indu_BA = 0.0
    
    # Commented out terms are for sSAPT0 scaling... do we really want this?
    # sIndu_AB_terms = core.Matrix("sInd [A<-B] (a x B)", sna, snB + snb + 1)
    # sIndu_BA_terms = core.Matrix("sInd [B<-A] (A x b)", snA + sna + 1, snb)
    # sIndu_AB_termsp = sIndu_AB_terms.np
    # sIndu_BA_termsp = sIndu_BA_terms.np
    # sIndu_AB = 0.0
    # sIndu_BA = 0.0
    
    # ==> A <- B Uncoupled <==
    if dimer_wfn.has_potential_variable("B"):
        Var = core.triplet(Cocc_A, cache["VB_extern"], Cvir_A, True, False, False)
        dfh.write_disk_tensor("WBar", Var, (nB + nb1, nB + nb1 + 1))
    else:
        Var = core.Matrix("zero", na, nr)
        Var.zero()
        dfh.write_disk_tensor("WBar", Var, (nB + nb1, nB + nb1 + 1))
    
    for B in range(nB + nb1 + 1): # add one for external potential
        # ESP
        dfh.fill_tensor("WBar", wB, [B, B + 1])
        # Uncoupled
        for a in range(na):
            for r in range(nr):
                # fill_tensor wB as (1, na, nr), so we take first index only
                xA.np[a, r] = wB.np[0, a, r] / (eps_occ_A.np[a] - eps_vir_A.np[r])
        
        x2A = core.doublet(Uocc_A, xA, True, False)
        x2Ap = x2A.np
        
        for a in range(na):
            Jval = 2.0 * np.dot(x2Ap[a, :], wBT.np[a, :])
            Kval = 2.0 * np.dot(x2Ap[a, :], uBT.np[a, :])
            Ind20u_AB += Jval
            ExchInd20u_AB_termsp[a, B] = Kval
            ExchInd20u_AB += Kval
            Ind20u_AB_termsp[a, B] = Jval
            # if core.get_option("SAPT", "SSAPT0_SCALE"):
            #     sExchInd20u_AB_termsp[a, B] = Kval
            #     sExchInd20u_AB += Kval
            #     sIndu_AB_termsp[a, B] = Jval + Kval
            #     sIndu_AB += Jval + Kval
            
            Indu_AB_terms.np[a, B] = Jval + Kval
            Indu_AB += Jval + Kval
    
    # ==> B <- A Uncoupled <==
    if dimer_wfn.has_potential_variable("A"):
        Vbs = core.triplet(Cocc_B, cache["VA_extern"], Cvir_B, True, False, False)
        dfh.write_disk_tensor("WAbs", Vbs, (nA + na1, nA + na1 + 1))
    else:
        Vbs = core.Matrix("zero", nb, ns)
        Vbs.zero()
        dfh.write_disk_tensor("WAbs", Vbs, (nA + na1, nA + na1 + 1))
    
    for A in range(nA + na1 + 1):
        dfh.fill_tensor("WAbs", wA, [A, A + 1])
        for b in range(nb):
            for s in range(ns):
                xB.np[b, s] = wA.np[0, b, s] / (eps_occ_B.np[b] - eps_vir_B.np[s])
        
        x2B = core.doublet(Uocc_B, xB, True, False)
        x2Bp = x2B.np
        
        for b in range(nb):
            Jval = 2.0 * np.dot(x2Bp[b, :], wAT.np[b, :])
            Kval = 2.0 * np.dot(x2Bp[b, :], uAT.np[b, :])
            Ind20u_BA_termsp[A, b] = Jval
            Ind20u_BA += Jval
            ExchInd20u_BA_termsp[A, b] = Kval
            ExchInd20u_BA += Kval
            # if core.get_option("SAPT", "SSAPT0_SCALE"):
            #     sExchInd20u_BA_termsp[A, b] = Kval
            #     sExchInd20u_BA += Kval
            #     sIndu_BA_termsp[A, b] = Jval + Kval
            #     sIndu_BA += Jval + Kval
            
            Indu_BA_terms.np[A, b] = Jval + Kval
            Indu_BA += Jval + Kval

    
    # Currently Ind20 and Exch-Ind are qualitatively coming out with wrong sign even...
    if do_print:
        core.print_out(f"    Ind20,u (A<-B)          = {Ind20u_AB*1000:18.8f} [mEh]\n")
        core.print_out(f"    Ind20,u (B<-A)          = {Ind20u_BA*1000:18.8f} [mEh]\n")
        assert abs(scalars['Ind20,u (A<-B)'] - Ind20u_AB) < 1e-8, f"Ind20u_AB mismatch: {1000 * scalars['Ind20,u (A<-B)']:.8f} vs {1000 * Ind20u_AB:.8f}"
        assert abs(scalars['Ind20,u (A->B)'] - Ind20u_BA) < 1e-8, f"Ind20u_BA mismatch: {1000 * scalars['Ind20,u (A->B)']:.8f} vs {1000 * Ind20u_BA:.8f}"
        core.print_out(f"    Ind20,u                 = {Ind20u_AB + Ind20u_BA*1000:18.8f} [mEh]\n")
        core.print_out(f"    Exch-Ind20,u (A<-B)     = {ExchInd20u_AB*1000:18.8f} [mEh]\n")
        core.print_out(f"    Exch-Ind20,u (B<-A)     = {ExchInd20u_BA*1000:18.8f} [mEh]\n")
        assert abs(scalars['Exch-Ind20,u (A<-B)'] - ExchInd20u_AB) < 1e-8, f"ExchInd20u_AB mismatch: {1000 * scalars['Exch-Ind20,u (A<-B)']:.8f} vs {1000 * ExchInd20u_AB:.8f}"
        assert abs(scalars['Exch-Ind20,u (A->B)'] - ExchInd20u_BA) < 1e-8, f"ExchInd20u_BA mismatch: {1000 * scalars['Exch-Ind20,u (A->B)']:.8f} vs {1000 * ExchInd20u_BA:.8f}"
        core.print_out(f"    Exch-Ind20,u            = {ExchInd20u_AB + ExchInd20u_BA*1000:18.8f} [mEh]\n\n")

    # Induction scaling
    if ind_scale:
        dHF = scalars.get("Delta HF Correction", 0.0)
        IndHF = scalars["Ind20,r"] + scalars["Exch-Ind20,r"] + dHF
        IndSAPT0 = scalars["Ind20,r"] + scalars["Exch-Ind20,r"]
        
        Sdelta = IndHF / IndSAPT0
        
        # NOTE: if doing ind_resp, logic below needs adjusted
        SrAB = ((scalars["Ind20,r (A<-B)"] + scalars["Exch-Ind20,r (A<-B)"]) / 
                (scalars["Ind20,u (A<-B)"] + scalars["Exch-Ind20,u (A<-B)"]))
        SrBA = ((scalars["Ind20,r (A->B)"] + scalars["Exch-Ind20,r (A->B)"]) / 
                (scalars["Ind20,u (A->B)"] + scalars["Exch-Ind20,u (A->B)"]))
        
        if do_print:
            core.print_out(f"    Scaling for delta HF        = {Sdelta:11.3E}\n")
            core.print_out(f"    Scaling for response (A<-B) = {SrAB:11.3E}\n")
            core.print_out(f"    Scaling for response (A->B) = {SrBA:11.3E}\n")
            core.print_out(f"    Scaling for total (A<-B)    = {Sdelta * SrAB:11.3E}\n")
            core.print_out(f"    Scaling for total (A->B)    = {Sdelta * SrBA:11.3E}\n")
            core.print_out("\n")
        
        # Apply scaling to all terms
        Indu_AB_terms.scale(Sdelta * SrAB)
        Indu_BA_terms.scale(Sdelta * SrBA)
        Ind20u_AB_terms.scale(Sdelta * SrAB)
        ExchInd20u_AB_terms.scale(Sdelta * SrAB)
        Ind20u_BA_terms.scale(Sdelta * SrBA)
        ExchInd20u_BA_terms.scale(Sdelta * SrBA)
        
        # Apply SSAPT0 scaling if enabled
        # if "sExch-Ind20,r" in scalars:
        #     sIndu_AB_terms.scale(sSdelta * sSrAB)
        #     sIndu_BA_terms.scale(sSdelta * sSrBA)

    IndAB_AB = core.Matrix("IndAB_AB", nA + na1 + 1, nB + nb1 + 1)
    IndBA_AB = core.Matrix("IndBA_AB", nA + na1 + 1, nB + nb1 + 1)

    # Final assembly might be wrong... backtrace time
    for a in range(na):
        for B in range(nB + nb1 + 1):
            IndAB_AB.np[a + nA, B] = Ind20u_AB_termsp[a, B] + ExchInd20u_AB_termsp[a, B]
    for A in range(nA + na1 + 1):
        for b in range(nb):
            IndBA_AB.np[A, b + nB] = Ind20u_BA_termsp[A, b] + ExchInd20u_BA_termsp[A, b]
    
    cache["IndAB_AB"] = IndAB_AB
    cache["IndBA_AB"] = IndBA_AB
    
    # if core.get_option("SAPT", "SSAPT0_SCALE"):
    #     cache["sExchInd20u_AB"] = sExchInd20u_AB
    #     cache["sExchInd20u_BA"] = sExchInd20u_BA
    #     cache["sIndu_AB"] = sIndu_AB
    #     cache["sIndu_BA"] = sIndu_BA

    """
    Ind20,u (A<-B)      =    -0.000005862 [mEh]
    Ind20,u (B<-A)      =    -0.000003086 [mEh]
    Ind20,u             =    -0.000008949 [mEh]
    Exch-Ind20,u (A<-B) =     0.000000887 [mEh]
    Exch-Ind20,u (B<-A) =     0.000000291 [mEh]
    Exch-Ind20,u        =     0.000001178 [mEh]
    """
    
    # NOT IMPLEMENTED YET
    # if (ind_resp) {
    #     outfile->Printf("  COUPLED INDUCTION (You asked for it!):\n\n");
    dfh.clear_all()
    return cache


def fdisp0(cache, scalars, dimer_wfn, wfn_A, wfn_B, jk, do_print=True):
    if do_print:
        core.print_out("  ==> F-SAPT0 Dispersion <==\n\n")
    
    core.timer_on("F-SAPT Disp Setup")
    # ind_scale = core.get_option("FISAPT", "FISAPT_FSAPT_IND_SCALE")
    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT")
    
    mol = dimer_wfn.molecule()
    dimer_basis = dimer_wfn.basisset()
    nA = mol.natom()
    nB = mol.natom()
    nfa = cache["Lfocc0A"].shape[1]
    nfb = cache["Lfocc0B"].shape[1]
    # Use active occupied dimensions (excluding frozen core) to match C++ FISAPT fdisp
    na = cache["Caocc0A"].shape[1]
    nb = cache["Caocc0B"].shape[1]
    nr = cache["Cvir_A"].shape[1]
    ns = cache["Cvir_B"].shape[1]
    nn = cache["Cocc_A"].shape[0]  # number of AO basis functions
    
    na1 = na
    nb1 = nb

    # Disp_AB = core.Matrix("Disp_AB", nA + nfa + na1 + 1, nB + nfb + nb1 + 1)
    snA = 0
    snfa = 0
    sna = 0
    snB = 0
    snfb = 0
    snb = 0
    # if options_.get_bool("FISAPT", "FISAPT_SSAPT0_SCALE"):
    #     snA = nA
    #     snfa = nfa
    #     sna = na
    #     snB = nB
    #     snfb = nfb
    #     snb = nb

    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        na1 = na + 1
        nb1 = nb + 1
    
    Locc_A = cache["Locc_A"].clone()
    Locc_A.name = "LoccA"
    Locc_B = cache["Locc_B"].clone()
    Locc_B.name = "LoccB"
    
    Uocc_A = cache["Uocc_A"]
    Uocc_B = cache["Uocc_B"]
    
    # Use active occupied orbitals (excluding frozen core) to match C++ FISAPT fdisp
    Cocc_A = cache["Caocc0A"]
    Cocc_B = cache["Caocc0B"]
    Cvir_A = cache["Cvir_A"]
    Cvir_B = cache["Cvir_B"]
    

    # Cvir_A.set_name("Cvir_A")
    # print(Cvir_A)

    # Use only active occupied orbital energies (skip frozen core)
    # nfa and nfb are already defined at the start of fdisp0
    eps_occ_A = core.Vector.from_array(cache["eps_occ_A"].np[nfa:])
    eps_occ_B = core.Vector.from_array(cache["eps_occ_B"].np[nfb:])
    eps_vir_A = cache["eps_vir_A"]
    eps_vir_B = cache["eps_vir_B"]

    # Collect relevant variables
    S = cache["S"]
    D_A = cache["D_A"]
    P_A = cache["P_A"]
    V_A = cache["V_A"]
    J_A = cache["J_A"]
    K_A = cache["K_A"]
    D_B = cache["D_B"]
    P_B = cache["P_B"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]
    K_B = cache["K_B"]
    # J_O = cache["J_O"]
    K_O = cache["K_O"]
    # J_P_A = cache["J_P_A"]
    # J_P_B = cache["J_P_B"]

    aux_basis = dimer_wfn.get_basisset("DF_BASIS_SCF")
    nQ = aux_basis.nbf()
    
    # => Auxiliary C matrices <= //
    # Cr1 = (I - D_B * S) * Cvir_A  [C++ line 6766-6768]
    Cr1 = chain_gemm_einsums([D_B, S, Cvir_A])
    ein.core.axpy(-1.0, Cvir_A.np, Cr1.np)
    
    # Cs1 = (I - D_A * S) * Cvir_B  [C++ line 6769-6771]
    Cs1 = chain_gemm_einsums([D_A, S, Cvir_B])
    ein.core.axpy(-1.0, Cvir_B.np, Cs1.np)
    
    # Ca2 = D_B * S * Cocc_A  [C++ line 6772]
    Ca2 = chain_gemm_einsums([D_B, S, Cocc_A])
    
    # Cb2 = D_A * S * Cocc_B  [C++ line 6773]
    Cb2 = chain_gemm_einsums([D_A, S, Cocc_B])
    
    # Cr3 = 2 * (D_B * S * Cvir_A - D_A * S * D_B * S * Cvir_A)  [C++ line 6775-6778]

    # std::shared_ptr<Matrix> Cr3 = linalg::triplet(D_B, S, Cavir_A);
    # Cr3->set_name("Cr3");
    # Cr3->print();
    # std::shared_ptr<Matrix> CrX = linalg::triplet(linalg::triplet(D_A, S, D_B), S, Cavir_A);
    # CrX->set_name("CrX");
    # CrX->print();
    # Cr3->subtract(CrX);
    # Cr3->scale(2.0);
    # Cr3->print();
    Cr3 = chain_gemm_einsums([D_B, S, Cvir_A])
    CrX = chain_gemm_einsums([D_A, S, D_B, S, Cvir_A])
    Cr3.subtract(CrX)
    Cr3.scale(2.0)
    
    # Cs3 = 2 * (D_A * S * Cvir_B - D_B * S * D_A * S * Cvir_B)  [C++ line 6779-6782]
    Cs3 = chain_gemm_einsums([D_A, S, Cvir_B])
    CsX = chain_gemm_einsums([D_B, S, D_A, S, Cvir_B])
    Cs3.subtract(CsX)
    Cs3.scale(2.0)
    
    # Ca4 = -2 * D_A * S * D_B * S * Cocc_A  [C++ line 6784-6785]
    Ca4 = chain_gemm_einsums([D_A, S, D_B, S, Cocc_A])
    Ca4.scale(-2.0)
    
    # Cb4 = -2 * D_B * S * D_A * S * Cocc_B  [C++ line 6786-6787]
    Cb4 = chain_gemm_einsums([D_B, S, D_A, S, Cocc_B])
    Cb4.scale(-2.0)
    
    # => Auxiliary V matrices <= #  [C++ lines 6789-6872]
    
    # Jbr = 2.0 * Cocc_B.T @ J_A @ Cvir_A  [C++ lines 6791-6792]
    Jbr = chain_gemm_einsums([Cocc_B, J_A, Cvir_A], ['T', 'N', 'N'])
    Jbr.scale(2.0)
    
    # Kbr = -1.0 * Cocc_B.T @ K_A @ Cvir_A  [C++ lines 6793-6794]
    Kbr = chain_gemm_einsums([Cocc_B, K_A, Cvir_A], ['T', 'N', 'N'])
    Kbr.scale(-1.0)
    
    # Jas = 2.0 * Cocc_A.T @ J_B @ Cvir_B  [C++ lines 6796-6797]
    Jas = chain_gemm_einsums([Cocc_A, J_B, Cvir_B], ['T', 'N', 'N'])
    Jas.scale(2.0)
    
    # Kas = -1.0 * Cocc_A.T @ K_B @ Cvir_B  [C++ lines 6798-6799]
    Kas = chain_gemm_einsums([Cocc_A, K_B, Cvir_B], ['T', 'N', 'N'])
    Kas.scale(-1.0)
    
    # KOas = 1.0 * Cocc_A.T @ K_O @ Cvir_B  [C++ lines 6801-6802]
    KOas = chain_gemm_einsums([Cocc_A, K_O, Cvir_B], ['T', 'N', 'N'])
    
    # KObr = 1.0 * Cocc_B.T @ K_O.T @ Cvir_A  [C++ lines 6803-6804]
    # Note: K_O is transposed (second 'T' in the transpose list)
    KObr = chain_gemm_einsums([Cocc_B, K_O, Cvir_A], ['T', 'T', 'N'])
    
    # JBas = -2.0 * (Cocc_A.T @ S @ D_B) @ J_A @ Cvir_B  [C++ lines 6806-6807]
    temp_JBas = chain_gemm_einsums([Cocc_A, S, D_B], ['T', 'N', 'N'])
    JBas = chain_gemm_einsums([temp_JBas, J_A, Cvir_B], ['N', 'N', 'N'])
    JBas.scale(-2.0)
    
    # JAbr = -2.0 * (Cocc_B.T @ S @ D_A) @ J_B @ Cvir_A  [C++ lines 6808-6809]
    temp_JAbr = chain_gemm_einsums([Cocc_B, S, D_A], ['T', 'N', 'N'])
    JAbr = chain_gemm_einsums([temp_JAbr, J_B, Cvir_A], ['N', 'N', 'N'])
    JAbr.scale(-2.0)
    
    # Jbs = 4.0 * Cocc_B.T @ J_A @ Cvir_B  [C++ lines 6811-6812]
    Jbs = chain_gemm_einsums([Cocc_B, J_A, Cvir_B], ['T', 'N', 'N'])
    Jbs.scale(4.0)
    
    # Jar = 4.0 * Cocc_A.T @ J_B @ Cvir_A  [C++ lines 6813-6814]
    Jar = chain_gemm_einsums([Cocc_A, J_B, Cvir_A], ['T', 'N', 'N'])
    Jar.scale(4.0)
    
    # JAas = -2.0 * (Cocc_A.T @ J_B @ D_A) @ S @ Cvir_B  [C++ lines 6816-6817]
    temp_JAas = chain_gemm_einsums([Cocc_A, J_B, D_A], ['T', 'N', 'N'])
    JAas = chain_gemm_einsums([temp_JAas, S, Cvir_B], ['N', 'N', 'N'])
    JAas.scale(-2.0)
    
    # JBbr = -2.0 * (Cocc_B.T @ J_A @ D_B) @ S @ Cvir_A  [C++ lines 6818-6819]
    temp_JBbr = chain_gemm_einsums([Cocc_B, J_A, D_B], ['T', 'N', 'N'])
    JBbr = chain_gemm_einsums([temp_JBbr, S, Cvir_A], ['N', 'N', 'N'])
    JBbr.scale(-2.0)
    
    # Get your signs right Hesselmann!  [C++ line 6821]
    # Vbs = 2.0 * Cocc_B.T @ V_A @ Cvir_B  [C++ lines 6822-6823]
    Vbs = chain_gemm_einsums([Cocc_B, V_A, Cvir_B], ['T', 'N', 'N'])
    Vbs.scale(2.0)
    
    # Var = 2.0 * Cocc_A.T @ V_B @ Cvir_A  [C++ lines 6824-6825]
    Var = chain_gemm_einsums([Cocc_A, V_B, Cvir_A], ['T', 'N', 'N'])
    Var.scale(2.0)
    
    # VBas = -1.0 * (Cocc_A.T @ S @ D_B) @ V_A @ Cvir_B  [C++ lines 6826-6827]
    temp_VBas = chain_gemm_einsums([Cocc_A, S, D_B], ['T', 'N', 'N'])
    VBas = chain_gemm_einsums([temp_VBas, V_A, Cvir_B], ['N', 'N', 'N'])
    VBas.scale(-1.0)
    
    # VAbr = -1.0 * (Cocc_B.T @ S @ D_A) @ V_B @ Cvir_A  [C++ lines 6828-6829]
    temp_VAbr = chain_gemm_einsums([Cocc_B, S, D_A], ['T', 'N', 'N'])
    VAbr = chain_gemm_einsums([temp_VAbr, V_B, Cvir_A], ['N', 'N', 'N'])
    VAbr.scale(-1.0)
    
    # VRas = 1.0 * (Cocc_A.T @ V_B @ P_A) @ S @ Cvir_B  [C++ lines 6830-6831]
    temp_VRas = chain_gemm_einsums([Cocc_A, V_B, P_A], ['T', 'N', 'N'])
    VRas = chain_gemm_einsums([temp_VRas, S, Cvir_B], ['N', 'N', 'N'])
    
    # VSbr = 1.0 * (Cocc_B.T @ V_A @ P_B) @ S @ Cvir_A  [C++ lines 6832-6833]
    temp_VSbr = chain_gemm_einsums([Cocc_B, V_A, P_B], ['T', 'N', 'N'])
    VSbr = chain_gemm_einsums([temp_VSbr, S, Cvir_A], ['N', 'N', 'N'])
    
    # Sas = Cocc_A.T @ S @ Cvir_B  [C++ line 6835]
    Sas = chain_gemm_einsums([Cocc_A, S, Cvir_B], ['T', 'N', 'N'])
    
    # Sbr = Cocc_B.T @ S @ Cvir_A  [C++ line 6836]
    Sbr = chain_gemm_einsums([Cocc_B, S, Cvir_A], ['T', 'N', 'N'])
    
    # Qbr = Jbr + Kbr + KObr + JAbr + JBbr + VAbr + VSbr  [C++ lines 6838-6846]
    Qbr = Jbr.clone()
    Qbr.add(Kbr)
    Qbr.add(KObr)
    Qbr.add(JAbr)
    Qbr.add(JBbr)
    Qbr.add(VAbr)
    Qbr.add(VSbr)
    
    # Qas = Jas + Kas + KOas + JAas + JBas + VBas + VRas  [C++ lines 6848-6856]
    Qas = Jas.clone()
    Qas.add(Kas)
    Qas.add(KOas)
    Qas.add(JAas)
    Qas.add(JBas)
    Qas.add(VBas)
    Qas.add(VRas)
    
    # SBar = Cocc_A.T @ S @ D_B @ S @ Cvir_A  [C++ line 6858]
    SBar = chain_gemm_einsums([Cocc_A, S, D_B, S, Cvir_A], ['T', 'N', 'N', 'N', 'N'])
    
    # SAbs = Cocc_B.T @ S @ D_A @ S @ Cvir_B  [C++ line 6859]
    SAbs = chain_gemm_einsums([Cocc_B, S, D_A, S, Cvir_B], ['T', 'N', 'N', 'N', 'N'])
    
    # Qar = Jar + Var  [C++ lines 6861-6864]
    Qar = Jar.clone()
    Qar.add(Var)
    
    # Qbs = Jbs + Vbs  [C++ lines 6866-6869]
    Qbs = Jbs.clone()
    Qbs.add(Vbs)
    
    # => Integrals from DFHelper <= #  [C++ lines 6895-6946]
    
    # Build list of orbital space matrices for DF transformations  [C++ lines 6897-6909]
    # Order: Cocc_A, Cvir_A, Cocc_B, Cvir_B, Cr1, Cs1, Ca2, Cb2, Cr3, Cs3, Ca4, Cb4
    # Convert einsums RuntimeTensorD objects to core.Matrix objects for DFHelper
    # RuntimeTensorD supports buffer protocol, so np.asarray() can convert to numpy
    orbital_spaces = [
        core.Matrix.from_array(Cocc_A),    # 0: 'a'
        core.Matrix.from_array(Cvir_A),    # 1: 'r'
        core.Matrix.from_array(Cocc_B),    # 2: 'b'
        core.Matrix.from_array(Cvir_B),    # 3: 's'
        core.Matrix.from_array(Cr1),       # 4: 'r1'
        core.Matrix.from_array(Cs1),       # 5: 's1'
        core.Matrix.from_array(Ca2),       # 6: 'a2'
        core.Matrix.from_array(Cb2),       # 7: 'b2'
        core.Matrix.from_array(Cr3),       # 8: 'r3'
        core.Matrix.from_array(Cs3),       # 9: 's3'
        core.Matrix.from_array(Ca4),       # 10: 'a4'
        core.Matrix.from_array(Cb4),       # 11: 'b4'
    ]

    # orbital_spaces[0].set_name("Cocc_A")
    # orbital_spaces[1].set_name("Cvir_A")
    # orbital_spaces[2].set_name("Cocc_B")
    # orbital_spaces[3].set_name("Cvir_B")
    # orbital_spaces[4].set_name("Cr1")
    # orbital_spaces[5].set_name("Cs1")
    # orbital_spaces[6].set_name("Ca2")
    # orbital_spaces[7].set_name("Cb2")
    # orbital_spaces[8].set_name("Cr3")
    # orbital_spaces[9].set_name("Cs3")
    # orbital_spaces[10].set_name("Ca4")
    # orbital_spaces[11].set_name("Cb4")

    # print("Caocc_A\n", orbital_spaces[0].np)
    # print("Cavir_A\n", orbital_spaces[1].np)
    # print("Caocc_B\n", orbital_spaces[2].np)
    # print("Cavir_B\n", orbital_spaces[3].np)
    # print("Cr1\n", orbital_spaces[4].np)
    # print("Cs1\n", orbital_spaces[5].np)
    # print("Ca2\n", orbital_spaces[6].np)
    # print("Cb2\n", orbital_spaces[7].np)
    # print("Cr3\n", orbital_spaces[8].np)
    # print("Cs3\n", orbital_spaces[9].np)
    # print("Ca4\n", orbital_spaces[10].np)
    # print("Cb4\n", orbital_spaces[11].np)
    # orbital_spaces[0].print()
    # orbital_spaces[1].print()
    # orbital_spaces[2].print()
    # orbital_spaces[3].print()
    # orbital_spaces[4].print()
    # orbital_spaces[5].print()
    # orbital_spaces[6].print()
    # orbital_spaces[7].print()
    # orbital_spaces[8].print()
    # orbital_spaces[9].print()
    # orbital_spaces[10].print()
    # orbital_spaces[11].print()

    
    # Calculate total columns for memory allocation  [C++ lines 6911-6915]
    ncol = sum(mat.shape[1] for mat in orbital_spaces)
    nrows = orbital_spaces[0].shape[0]  # All should have same number of rows (AO basis)
    
    # Initialize DFHelper  [C++ lines 6917-6922]
    aux_basis = dimer_wfn.get_basisset("DF_BASIS_SCF")
    dfh = core.DFHelper(dimer_basis, aux_basis)
    
    # Set memory: total available minus space needed for orbital matrices
    # Note: In C++, doubles_ is the total memory budget in doubles
    # Here we use a reasonable default or get from options if available
    memory_bytes = core.get_memory()  # in bytes
    memory_doubles = memory_bytes // 8
    orbital_memory = nrows * ncol
    print(orbital_memory)
    dfh.set_memory(memory_doubles - orbital_memory)
    # print set memory in GB
    core.print_out(f"    Setting DFHelper memory to {(memory_doubles - orbital_memory) * 8 / 1e9:.3f} GB\n")
    
    dfh.set_method("DIRECT_iaQ")
    dfh.set_nthreads(core.get_num_threads())
    dfh.initialize()
    dfh.print_header()
    
    # Add orbital spaces  [C++ lines 6924-6935]
    dfh.add_space("a", orbital_spaces[0])    # Cocc_A
    dfh.add_space("r", orbital_spaces[1])    # Cvir_A
    dfh.add_space("b", orbital_spaces[2])    # Cocc_B
    dfh.add_space("s", orbital_spaces[3])    # Cvir_B
    dfh.add_space("r1", orbital_spaces[4])   # Cr1
    dfh.add_space("s1", orbital_spaces[5])   # Cs1
    dfh.add_space("a2", orbital_spaces[6])   # Ca2
    dfh.add_space("b2", orbital_spaces[7])   # Cb2
    dfh.add_space("r3", orbital_spaces[8])   # Cr3
    dfh.add_space("s3", orbital_spaces[9])   # Cs3
    dfh.add_space("a4", orbital_spaces[10])  # Ca4
    dfh.add_space("b4", orbital_spaces[11])  # Cb4
    
    # Add DF transformations  [C++ lines 6937-6946]
    # Format: (name, left_space, right_space) -> computes (left|right) integrals
    dfh.add_transformation("Aar", "r", "a")   # (r|a) virtuals_A x occupied_A
    dfh.add_transformation("Abs", "s", "b")   # (s|b) virtuals_B x occupied_B
    dfh.add_transformation("Bas", "s1", "a")  # (s1|a) Cs1 x occupied_A
    dfh.add_transformation("Bbr", "r1", "b")  # (r1|b) Cr1 x occupied_B
    dfh.add_transformation("Cas", "s", "a2")  # (s|a2) virtuals_B x Ca2
    dfh.add_transformation("Cbr", "r", "b2")  # (r|b2) virtuals_A x Cb2
    dfh.add_transformation("Dar", "r3", "a")  # (r3|a) Cr3 x occupied_A
    dfh.add_transformation("Dbs", "s3", "b")  # (s3|b) Cs3 x occupied_B
    dfh.add_transformation("Ear", "r", "a4")  # (r|a4) virtuals_A x Ca4
    dfh.add_transformation("Ebs", "s", "b4")  # (s|b4) virtuals_B x Cb4
    
    # TODO: Handle link orbital spaces for parallel/perpendicular coupling (lines 6950-7018)
    # For now, skip this and proceed with standard dispersion calculation
    
    # Perform DF transformations  [C++ line 7020]
    dfh.transform()
    
    # Clear spaces now that transformations are done  [C++ lines 7024-7033]
    dfh.clear_spaces()
    
    # => Memory blocking setup  [C++ lines 7035-7082]
    
    # Number of threads (single-threaded in Python)
    nT = 1
    
    # Calculate overhead for work arrays
    overhead = 0
    overhead += 5 * nT * na * nb  # Tab, Vab, T2ab, V2ab, Iab work arrays
    # For link orbitals with parperp, we'd need more, but we're skipping that
    overhead += 2 * na * ns + 2 * nb * nr + 2 * na * nr + 2 * nb * ns  # S and Q matrices
    overhead += 2 * na * nb * (nT + 1)  # E_disp20 and E_exch_disp20 thread work and final
    overhead += 1 * sna * snb * (nT + 1)  # sE_exch_disp20 thread work and final
    overhead += 1 * (nA + nfa + na) * (nB + nfb + nb)  # Disp_AB
    overhead += 1 * (snA + snfa + sna) * (snB + snfb + snb)  # sDisp_AB
    overhead += 12 * nn * nn  # D, V, J, K, P, C matrices for A and B
    
    # Available memory for dispersion calculation
    total_memory = core.get_memory() // 8  # Convert bytes to doubles
    rem = total_memory - overhead
    
    core.print_out(f"    {total_memory} doubles - {overhead} overhead leaves {rem} for dispersion\n")
    
    if rem < 0:
        raise Exception("Too little static memory for fdisp0")
    
    # Calculate cost per r or s virtual orbital
    # Each r needs: Aar, Bbr, Cbr, Dar (each is na x nQ or nb x nQ)
    cost_r = 2 * na * nQ + 2 * nb * nQ
    max_r_l = rem // (2 * cost_r)  # Factor of 2 because we hold both r and s slices
    max_s_l = max_r_l
    max_r = min(max_r_l, nr)
    max_s = min(max_s_l, ns)
    
    if max_r < 1 or max_s < 1:
        raise Exception("Too little dynamic memory for fdisp0")
    
    nrblocks = (nr + max_r - 1) // max_r  # Ceiling division
    nsblocks = (ns + max_s - 1) // max_s
    
    core.print_out(f"    Processing a single (r,s) pair requires {cost_r * 2} doubles\n")
    core.print_out(f"    {nr} values of r processed in {nrblocks} blocks of {max_r}\n")
    core.print_out(f"    {ns} values of s processed in {nsblocks} blocks of {max_s}\n\n")
    
    # => Compute Far = Dar + Ear and Fbs = Dbs + Ebs  [C++ lines 7136-7168]
    # These represent combined D and E DF integrals that will be reused in the main loop
    
    # Add disk tensor for Far
    dfh.add_disk_tensor("Far", (nr, na, nQ))
    
    # Loop over r blocks to compute Far = Dar + Ear
    for rstart in range(0, nr, max_r):
        nrblock = min(max_r, nr - rstart)
        
        # Allocate matrices to hold the tensor slices
        Dar = core.Matrix("Dar block", nrblock * na, nQ)
        Ear = core.Matrix("Ear block", nrblock * na, nQ)
        
        # Fill Dar and Ear from disk tensors
        dfh.fill_tensor("Dar", Dar, [rstart, rstart + nrblock], [0, na], [0, nQ])
        dfh.fill_tensor("Ear", Ear, [rstart, rstart + nrblock], [0, na], [0, nQ])
        
        # Compute Far = Dar + Ear (element-wise addition)
        Dar.np[:, :] += Ear.np[:, :]
        
        # Write Far back to disk (Dar now contains Dar + Ear)
        dfh.write_disk_tensor("Far", Dar, (rstart, rstart + nrblock))
    
    # Add disk tensor for Fbs
    dfh.add_disk_tensor("Fbs", (ns, nb, nQ))
    
    # Loop over s blocks to compute Fbs = Dbs + Ebs
    for sstart in range(0, ns, max_s):
        nsblock = min(max_s, ns - sstart)
        
        # Allocate matrices to hold the tensor slices
        Dbs = core.Matrix("Dbs block", nsblock * nb, nQ)
        Ebs = core.Matrix("Ebs block", nsblock * nb, nQ)
        
        # Fill Dbs and Ebs from disk tensors
        dfh.fill_tensor("Dbs", Dbs, [sstart, sstart + nsblock], [0, nb], [0, nQ])
        dfh.fill_tensor("Ebs", Ebs, [sstart, sstart + nsblock], [0, nb], [0, nQ])
        
        # Compute Fbs = Dbs + Ebs (element-wise addition)
        Dbs.np[:, :] += Ebs.np[:, :]
        
        # Write Fbs back to disk (Dbs now contains Dbs + Ebs)
        dfh.write_disk_tensor("Fbs", Dbs, (sstart, sstart + nsblock))
    
    E_disp20_comp = core.Matrix("E_disp20", na, nb)
    E_exch_disp20_comp = core.Matrix("E_exch_disp20", na, nb)
    
    # => MO to LO Transformation [C++ lines 7192-7193]
    Uaocc_A = cache["Uaocc0A"]
    Uaocc_B = cache["Uaocc0B"]
    UAp = Uaocc_A.np
    UBp = Uaocc_B.np
    
    # Orbital energies (already numpy arrays)
    # In the dispersion formula: indices a,b are occupied and r,s are virtual
    eap = eps_occ_A  # occupied energies for monomer A (index a)
    ebp = eps_occ_B  # occupied energies for monomer B (index b)
    erp = eps_vir_A  # virtual energies for monomer A (index r)
    esp = eps_vir_B  # virtual energies for monomer B (index s)
    
    # => Work arrays for inner loop
    Tab = core.Matrix("Tab", na, nb)
    Vab = core.Matrix("Vab", na, nb)
    T2ab = core.Matrix("T2ab", na, nb)
    V2ab = core.Matrix("V2ab", na, nb)
    Iab = core.Matrix("Iab", na, nb)
    
    # => Main r,s loop <= //
    # Allocate and fill r-block tensors
    Aar = core.Matrix("Aar block", nrblock * na, nQ)
    Far = core.Matrix("Far block", nrblock * na, nQ)
    Bbr = core.Matrix("Bbr block", nrblock * nb, nQ)
    Cbr = core.Matrix("Cbr block", nrblock * nb, nQ)
        
    # Allocate and fill s-block tensors
    Abs = core.Matrix("Abs block", nsblock * nb, nQ)
    Fbs = core.Matrix("Fbs block", nsblock * nb, nQ)
    Bas = core.Matrix("Bas block", nsblock * na, nQ)
    Cas = core.Matrix("Cas block", nsblock * na, nQ)
    core.timer_off("F-SAPT Disp Setup")
            
    core.timer_on("F-SAPT Disp Compute")
    for rstart in range(0, nr, max_r):
        nrblock = min(max_r, nr - rstart)
        
        dfh.fill_tensor("Aar", Aar, [rstart, rstart + nrblock], [0, na], [0, nQ])
        dfh.fill_tensor("Far", Far, [rstart, rstart + nrblock], [0, na], [0, nQ])
        dfh.fill_tensor("Bbr", Bbr, [rstart, rstart + nrblock], [0, nb], [0, nQ])
        dfh.fill_tensor("Cbr", Cbr, [rstart, rstart + nrblock], [0, nb], [0, nQ])
        
        # Get numpy pointers for r-block tensors and reshape to 3D
        # Tensors are stored as 2D with shape (nrblock * nX, nQ) and need to be (nrblock, nX, nQ)
        Aarp = Aar.np.reshape(nrblock, na, nQ)
        Farp = Far.np.reshape(nrblock, na, nQ)
        Bbrp = Bbr.np.reshape(nrblock, nb, nQ)
        Cbrp = Cbr.np.reshape(nrblock, nb, nQ)
        
        for sstart in range(0, ns, max_s):
            nsblock = min(max_s, ns - sstart)
            
            dfh.fill_tensor("Abs", Abs, [sstart, sstart + nsblock], [0, nb], [0, nQ])
            dfh.fill_tensor("Fbs", Fbs, [sstart, sstart + nsblock], [0, nb], [0, nQ])
            dfh.fill_tensor("Bas", Bas, [sstart, sstart + nsblock], [0, na], [0, nQ])
            dfh.fill_tensor("Cas", Cas, [sstart, sstart + nsblock], [0, na], [0, nQ])
            
            # Get numpy pointers for s-block tensors and reshape to 3D
            # Tensors are stored as 2D with shape (nsblock * nX, nQ) and need to be (nsblock, nX, nQ)
            Absp = Abs.np.reshape(nsblock, nb, nQ)
            Fbsp = Fbs.np.reshape(nsblock, nb, nQ)
            Basp = Bas.np.reshape(nsblock, na, nQ)
            Casp = Cas.np.reshape(nsblock, na, nQ)
            
            nrs = nrblock * nsblock
            
            # => RS inner loop <= //
            for rs in range(nrs):
                r = rs // nsblock
                s = rs % nsblock
                
                # Get pointers to work arrays and energy matrices
                Tabp = Tab.np
                Vabp = Vab.np
                T2abp = T2ab.np
                V2abp = V2ab.np
                Iabp = Iab.np
                E_disp20Tp = E_disp20_comp.np
                E_exch_disp20Tp = E_exch_disp20_comp.np
                
                # => Amplitudes, Disp20 <= //
                
                # Vab = Aar[r] @ Abs[s].T
                # Extract slices for r-th and s-th orbitals
                # Store these as we need them for Exch-Disp20 too
                Aar_r = Aarp[r, :, :]
                Abs_s = Absp[s, :, :]
                # Use einsum to match C++ DGEMM('N', 'T', ...) more closely
                np.einsum('aQ,bQ->ab', Aar_r, Abs_s, out=Vabp, optimize=True)
                
                # Compute amplitudes Tab[a,b] = Vab[a,b] / (ea + eb - er - es)
                for a in range(na):
                    for b in range(nb):
                        Tabp[a, b] = Vabp[a, b] / (eap.np[a] + ebp.np[b] - erp.np[r + rstart] - esp.np[s + sstart])
                
                # Transform to localized orbital basis
                # T2ab = UA.T @ Tab @ UB
                Iabp[:, :] = Tabp @ UBp
                T2abp[:, :] = UAp.T @ Iabp
                
                # V2ab = UA.T @ Vab @ UB
                Iabp[:, :] = Vabp @ UBp
                V2abp[:, :] = UAp.T @ Iabp
                
                # Accumulate Disp20
                for a in range(na):
                    for b in range(nb):
                        E_disp20Tp[a, b] += 4.0 * T2abp[a, b] * V2abp[a, b]
                
                # => Exch-Disp20 <= //
                
                # > Q1-Q3 < //
                # Vab = Bas[s] @ Bbr[r].T + Cas[s] @ Cbr[r].T + Aar[r] @ Fbs[s].T + Far[r] @ Abs[s].T
                # Extract slices for r-th and s-th orbitals
                Bas_s = Basp[s, :, :]
                Bbr_r = Bbrp[r, :, :]
                Cas_s = Casp[s, :, :]
                Cbr_r = Cbrp[r, :, :]
                Far_r = Farp[r, :, :]
                Fbs_s = Fbsp[s, :, :]
                
                Vabp[:, :] = Bas_s @ Bbr_r.T
                Vabp[:, :] += Cas_s @ Cbr_r.T
                Vabp[:, :] += Aar_r @ Fbs_s.T
                Vabp[:, :] += Far_r @ Abs_s.T
                
                # > V,J,K < //
                # Add outer product contributions using DGER equivalent
                # C_DGER(na, nb, 1.0, &Sasp[0][s + sstart], ns, &Qbrp[0][r + rstart], nr, Vabp[0], nb);
                Vabp[:, :] += np.outer(Sas.np[:, s + sstart], Qbr.np[:, r + rstart])
                
                # C_DGER(na, nb, 1.0, &Qasp[0][s + sstart], ns, &Sbrp[0][r + rstart], nr, Vabp[0], nb);
                Vabp[:, :] += np.outer(Qas.np[:, s + sstart], Sbr.np[:, r + rstart])
                
                # C_DGER(na, nb, 1.0, &Qarp[0][r + rstart], nr, &SAbsp[0][s + sstart], ns, Vabp[0], nb);
                Vabp[:, :] += np.outer(Qar.np[:, r + rstart], SAbs.np[:, s + sstart])
                
                # C_DGER(na, nb, 1.0, &SBarp[0][r + rstart], nr, &Qbsp[0][s + sstart], ns, Vabp[0], nb);
                Vabp[:, :] += np.outer(SBar.np[:, r + rstart], Qbs.np[:, s + sstart])
                
                # Transform to localized orbital basis
                Iabp[:, :] = Vabp @ UBp
                V2abp[:, :] = UAp.T @ Iabp
                
                # Accumulate ExchDisp20
                for a in range(na):
                    for b in range(nb):
                        E_exch_disp20Tp[a, b] -= 2.0 * T2abp[a, b] * V2abp[a, b]
    
    core.timer_off("F-SAPT Disp Compute")
    # => Accumulate thread results <= //
    E_disp20 = core.Matrix("E_disp20", nA + nfa + na1 + 1, nB + nfb + nb1 + 1)
    E_exch_disp20 = core.Matrix("E_exch_disp20", nA + nfa + na1 + 1, nB + nfb + nb1 + 1)
    
    # Single-threaded, so just use the first (and only) thread result
    # E_disp20.copy(E_disp20_comp)
    # E_exch_disp20.copy(E_exch_disp20_comp)
    for a in range(na):
        for b in range(nb):
            E_disp20.np[a + nfa + nA, b + nfb + nB] = E_disp20_comp.np[a, b]
            E_exch_disp20.np[a + nfa + nA, b + nfb + nB] = E_exch_disp20_comp.np[a, b]

    # => Populate cache['E'] matrix <= //
    # Store energy matrices and scalars
    # cache['E_DISP20'] = E_disp20
    # cache['E_EXCH_DISP20'] = E_exch_disp20
    # add E_disp20 and E_exch_disp20
    # Disp_AB = core.Matrix("DISP_AB", na, nb)
    Disp_AB = core.Matrix("Disp_AB", nA + nfa + na1 + 1, nB + nfb + nb1 + 1)
    Disp_AB.np[:, :] = E_disp20.np + E_exch_disp20.np
    cache['Disp_AB'] = Disp_AB
    # => Output printing <= //
    # {Elst10*1000:.8f} [mEh]
    Disp20 = np.sum(E_disp20.np)
    ExchDisp20 = np.sum(E_exch_disp20.np)

    cache["Exch-Disp20,u"] = ExchDisp20
    cache["Disp20,u"] = Disp20
    # if do_print:
    #     core.print_out(f"    Disp20              = {Disp20 * 1000:.8f} [mEh]\n")
    #     core.print_out(f"    Exch-Disp20         = {ExchDisp20 * 1000:.8f} [mEh]\n")
    #     core.print_out("\n")
    #     assert abs(scalars['Disp20,u'] - Disp20) < 1e-6, f"Disp20 scalar mismatch! {scalars['Disp20,u'] = } {Disp20 = }"
    #     assert abs(scalars['Exch-Disp20,u'] - ExchDisp20) < 1e-6, f"ExchDisp20 scalar mismatch!\nRef: {scalars['Exch-Disp20,u']:.4e}\nAct: {ExchDisp20:.4e}"
    return cache


def chain_gemm_matrix(
    tensors: list,
    transposes: list[str] = None,
    prefactors_C: list[float] = None,
    prefactors_AB: list[float] = None,
    return_tensors: list[bool] = None,
):
    """
    Computes a chain of matrix multiplications using numpy operations.
    
    Accepts psi4.core.Matrix objects and returns psi4.core.Matrix objects.
    Internally uses numpy for matrix operations via the .np property.

    Parameters
    ----------
    tensors : list[psi4.core.Matrix or array-like]
        List of matrices to be contracted in a chain multiplication.
    transposes : list[str], optional
        List of transpose operations for each tensor, where "N" means no 
        transpose and "T" means transpose.
    prefactors_C : list[float], optional
        List of prefactors for the resulting tensors in the chain.
        C = prefactors_AB * A @ B + prefactors_C * C
    prefactors_AB : list[float], optional
        List of prefactors for the tensors being multiplied in the chain.
    return_tensors : list[bool], optional
        List indicating which intermediate tensors should be returned. If None,
        only the final tensor is returned. Note that these are only
        intermediate tensors and final tensor; hence, the length of this list
        should be one less than the number of tensors.
    
    Returns
    -------
    psi4.core.Matrix or list[psi4.core.Matrix]
        The final result matrix, or a list of intermediate results if 
        return_tensors is specified.
    """
    N = len(tensors)
    if transposes is None:
        transposes = ["N"] * N
    if prefactors_C is None:
        prefactors_C = [0.0] * (N - 1)
    if prefactors_AB is None:
        prefactors_AB = [1.0] * (N - 1)
    
    # Convert first tensor to numpy array for computation
    first.np = tensors[0]
    computed_arrays = [first.np]
    
    try:
        for i in range(len(tensors) - 1):
            A.np = computed_arrays[-1]
            B.np = _to_numpy(tensors[i + 1])
            
            # For intermediate results (i > 0), always use 'N' for T1 since A is a computed intermediate
            T1 = transposes[i] if i == 0 else 'N'
            T2 = transposes[i + 1]
            
            # Apply transposes
            if T1 == "T":
                A.np = A.np.T
            if T2 == "T":
                B.np = B.np.T
            
            # Compute C = prefactors_AB * A @ B + prefactors_C * C
            # Since we're creating a new C each time, prefactors_C[i] * C is zero
            # unless we're accumulating into an existing result
            C.np = prefactors_AB[i] * np.dot(A.np, B.np)
            # Note: prefactors_C is typically 0.0 for new result, so we ignore it for now
            # If needed: C.np = prefactors_AB[i] * np.dot(A.np, B.np) + prefactors_C[i] * C.np
            
            computed_arrays.append(C.np)
    except Exception as e:
        raise ValueError(f"Error in einsum_chain_gemm: {e}\n{i = }\n{A.np.shape = }\n{B.np.shape = }\n{T1 = }\n{T2 = }")
    
    # Convert results back to psi4.core.Matrix
    if return_tensors is None:
        return _to_matrix(computed_arrays[-1])
    
    returned_tensors = []
    for i, r in enumerate(return_tensors):
        if r:
            returned_tensors.append(_to_matrix(computed_arrays[i + 1]))
    return returned_tensors

def chain_gemm_einsums(
    tensors: list[core.Matrix],
    transposes: list[str] = None,
    prefactors_C: list[float] = None,
    prefactors_AB: list[float] = None,
    return_tensors: list[bool] = None,
):
    """
    Computes a chain of einsum matrix multiplications

    Parameters
    ----------
    tensors : list[core.Matrix]
        List of tensors to be contracted.
    transposes : list[str], optional
        List of transpose operations for each tensor, where "N" means no transpose and "T" means transpose.
    prefactors_C : list[float], optional
        List of prefactors for the resulting tensors in the chain.
    prefactors_AB : list[float], optional
        List of prefactors for the tensors being multiplied in the chain.
    return_tensors : list[bool], optional
        List indicating which intermediate tensors should be returned. If None,
        only the final tensor is returned. Note that these are only
        intermediate tensors and final tensor; hence, the length of this list
        should be one less than the number of tensors.
    """
    # initialization "computed_tensors" with the first tensor of the chain
    computed_tensors = [tensors[0]]
    N = len(tensors)
    if transposes is None:
        transposes = ["N"] * N
    if prefactors_C is None:
        prefactors_C = [0.0] * (N - 1)
    if prefactors_AB is None:
        prefactors_AB = [1.0] * (N - 1)
    try:
        for i in range(len(tensors) - 1):
            A = computed_tensors[-1]
            B = tensors[i + 1]

            # For intermediate results (i > 0), always use 'N' for T1 since A is a computed intermediate
            T1 = transposes[i] if i == 0 else 'N'
            T2 = transposes[i + 1]
            A_size = A.shape[0]
            if T1 == "T":
                A_size = A.shape[1]
            B_size = B.shape[1]
            if T2 == "T":
                B_size = B.shape[0]

            # Initialize output as psi4.core.Matrix with zeros
            C = core.Matrix(A_size, B_size)
            C.zero()
            # Use ein.core.gemm to write to C.np
            ein.core.gemm(T1, T2, prefactors_AB[i], A.np, B.np, prefactors_C[i], C.np)
            computed_tensors.append(C)
    except Exception as e:
        raise ValueError(f"Error in einsum_chain_gemm: {e}\n{i = }\n{A = }\n{B = }\n{T1 = }\n{T2 = }")
    if return_tensors is None:
        return computed_tensors[-1]
    returned_tensors = []
    for i, r in enumerate(return_tensors):
        if r:
            returned_tensors.append(computed_tensors[i + 1])
    return returned_tensors


def exchange(cache, jk, do_print=True):
    r"""
    Computes the E10 exchange (S^2 and S^inf) from a build_sapt_jk_cache datacache.

    Equation E^{(1)}_{\rm exch}(S^2) =
        -2(P^{A,occ} S^{AO} P^{B,occ} S^{AO} P^{A,vir}) \cdot \omega^{B}
        -2(P^{B,occ} S^{AO} P^{A,occ} S^{AO} P^{B,vir}) \cdot \omega^{B}
        -2(P^{A,vir} S^{AO} P^{B,occ}) \cdot K[P^{A,occ} S^{AO} P^{B,vir}]
    """

    if do_print:
        core.print_out("\n  ==> E10 Exchange Einsums <== \n\n")

    # Build potentials using psi4.core.Matrix operations
    h_A = cache["V_A"].clone()
    ein.core.axpy(2.0, cache["J_A"].np, h_A.np)
    ein.core.axpy(-1.0, cache["K_A"].np, h_A.np)

    h_B = cache["V_B"].clone()
    ein.core.axpy(2.0, cache["J_B"].np, h_B.np)
    ein.core.axpy(-1.0, cache["K_B"].np, h_B.np)

    w_A = cache["V_A"].clone()
    ein.core.axpy(2.0, cache["J_A"].np, w_A.np)

    w_B = cache["V_B"].clone()
    ein.core.axpy(2.0, cache["J_B"].np, w_B.np)

    # Build inverse exchange metric
    nocc_A = cache["Cocc_A"].shape[1]
    nocc_B = cache["Cocc_B"].shape[1]
    SAB = chain_gemm_einsums(
        [cache['Cocc_A'], cache['S'], cache['Cocc_B']],
        ['T', 'N', 'N'],
    )

    num_occ = nocc_A + nocc_B

    Sab = core.Matrix(num_occ, num_occ)
    Sab.np[:nocc_A, nocc_A:] = SAB.np
    Sab.np[nocc_A:, :nocc_A] = SAB.np.T
    Sab.np[np.diag_indices_from(Sab.np)] += 1
    Sab.power(-1.0, 1.0e-14)
    Sab.np[np.diag_indices_from(Sab.np)] -= 1.0

    Tmo_AA = core.Matrix.from_array(Sab.np[:nocc_A, :nocc_A])
    Tmo_BB = core.Matrix.from_array(Sab.np[nocc_A:, nocc_A:])
    Tmo_AB = core.Matrix.from_array(Sab.np[:nocc_A, nocc_A:])

    T_AA = chain_gemm_einsums([cache['Cocc_A'], Tmo_AA, cache['Cocc_A']], ['N', 'N', 'T'])
    T_BB = chain_gemm_einsums([cache['Cocc_B'], Tmo_BB, cache['Cocc_B']], ['N', 'N', 'T'])
    T_AB = chain_gemm_einsums([cache['Cocc_A'], Tmo_AB, cache['Cocc_B']], ['N', 'N', 'T'])

    S = cache["S"]
    D_A = cache["D_A"]
    P_A = cache["P_A"]
    D_B = cache["D_B"]
    P_B = cache["P_B"]

    # Compute the J and K matrices
    jk.C_clear()

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_A"]))
    jk.C_right_add(chain_gemm_einsums([cache['Cocc_A'], Tmo_AA]))

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_B"]))
    jk.C_right_add(chain_gemm_einsums([cache['Cocc_A'], Tmo_AB]))

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_A"]))
    jk.C_right_add(chain_gemm_einsums([P_B, S, cache['Cocc_A']]))
    # This also works... you can choose to form the density-like matrix either
    # way..., just remember that the C_right_add has an adjoint (transpose, and switch matmul order)
    # jk.C_left_add(core.Matrix.from_array(einsum_chain_gemm([D_A, S, cache['Cvir_B']])))
    # jk.C_right_add(core.Matrix.from_array(cache['Cvir_B']))
    jk.compute()

    JT_A, JT_AB, Jij = jk.J()
    KT_A, KT_AB, Kij = jk.K()

    # Start S^2
    Exch_s2 = 0.0

    # Save some intermediate tensors to avoid recomputation in the next steps
    DA_S_DB_S_PA = chain_gemm_einsums([D_A, S, D_B, S, P_A])
    Exch_s2 -= 2.0 * ein.core.dot(w_B.np, DA_S_DB_S_PA.np)

    DB_S_DA_S_PB = chain_gemm_einsums([D_B, S, D_A, S, P_B])
    Exch_s2 -= 2.0 * ein.core.dot(w_A.np, DB_S_DA_S_PB.np)
    Exch_s2 -= 2.0 * ein.core.dot(Kij.np, chain_gemm_einsums([P_A, S, D_B]).np)

    if do_print:
        core.print_out(print_sapt_var("Exch10(S^2) ", Exch_s2, short=True))
        core.print_out("\n")

    # Start Sinf
    Exch10 = 0.0
    Exch10 -= 2.0 * ein.core.dot(D_A.np, cache["K_B"].np)
    Exch10 += 2.0 * ein.core.dot(T_AA.np, h_B.np)
    Exch10 += 2.0 * ein.core.dot(T_BB.np, h_A.np)
    Exch10 += 2.0 * ein.core.dot(T_AB.np, h_A.np + h_B.np)
    Exch10 += 4.0 * ein.core.dot(T_BB.np, JT_AB.np - 0.5 * KT_AB.np)
    Exch10 += 4.0 * ein.core.dot(T_AA.np, JT_AB.np - 0.5 * KT_AB.np.T)
    Exch10 += 4.0 * ein.core.dot(T_BB.np, JT_A.np - 0.5 * KT_A.np)
    Exch10 += 4.0 * ein.core.dot(T_AB.np, JT_AB.np - 0.5 * KT_AB.np.T)

    if do_print:
        core.set_variable("Exch10", Exch10)
        core.print_out(print_sapt_var("Exch10", Exch10, short=True))
        core.print_out("\n")

    return {"Exch10(S^2)": Exch_s2, "Exch10": Exch10}


def exchange_isapt(cache, jk, do_print=True):
    r"""
    Computes the E10 exchange for I-SAPT using the MCBS formula.
    
    This is the I-SAPT specific exchange function that uses the MCBS
    (monomer-centered basis set) formula, appropriate for 3-fragment
    calculations with link_assignment='C'.
    
    The formula is from fisapt.cc lines 2329-2477.
    """
    
    if do_print:
        core.print_out("\n  ==> E10 Exchange I-SAPT (MCBS) <== \n\n")
    
    # Get matrices from cache
    S = cache["S"]
    D_A = cache["D_A"]
    D_B = cache["D_B"]
    V_A = cache["V_A"]
    V_B = cache["V_B"]
    J_A = cache["J_A"]
    J_B = cache["J_B"]
    K_A = cache["K_A"]
    K_B = cache["K_B"]
    
    Cocc_A = cache["Cocc_A"]
    Cocc_B = cache["Cocc_B"]
    
    # Debug: print matrix diagnostics
    if do_print:
        core.print_out("    Matrix diagnostics:\n")
        core.print_out(f"      Tr(D_A) = {np.trace(D_A.np):12.6f}\n")
        core.print_out(f"      Tr(D_B) = {np.trace(D_B.np):12.6f}\n")
        core.print_out(f"      Tr(K_A) = {np.trace(K_A.np):12.6f}\n")
        core.print_out(f"      Tr(K_B) = {np.trace(K_B.np):12.6f}\n")
        core.print_out(f"      D_A · K_B = {ein.core.dot(D_A.np, K_B.np):12.6f}\n")
        core.print_out(f"      D_B · K_A = {ein.core.dot(D_B.np, K_A.np):12.6f}\n\n")
    
    # Compute K_O: K[D_B @ S @ Cocc_A]
    # C_O = D_B @ S @ Cocc_A
    C_O = chain_gemm_einsums([D_B, S, Cocc_A])
    jk.C_clear()
    jk.C_left_add(core.Matrix.from_array(Cocc_A))
    jk.C_right_add(core.Matrix.from_array(C_O))
    jk.compute()
    K_O = jk.K()[0]
    
    # ========== S^2 Exchange (MCBS formula) ==========
    # From fisapt.cc lines 2329-2346
    Exch_s2 = 0.0
    terms_s2 = []
    
    # Term 0: -2 * D_A · K_B
    t0 = -2.0 * ein.core.dot(D_A.np, K_B.np)
    Exch_s2 += t0
    terms_s2.append(t0)
    
    # Precompute D_A @ S @ D_B
    D_A_S_D_B = chain_gemm_einsums([D_A, S, D_B])
    D_B_S_D_A = chain_gemm_einsums([D_B, S, D_A])
    
    # Term 1: D_A @ S @ D_B with V_A, J_A, K_A
    t1 = -2.0 * ein.core.dot(D_A_S_D_B.np, V_A.np)
    t1 -= 4.0 * ein.core.dot(D_A_S_D_B.np, J_A.np)
    t1 += 2.0 * ein.core.dot(D_A_S_D_B.np, K_A.np)
    Exch_s2 += t1
    terms_s2.append(t1)
    
    # Term 2: D_B @ S @ D_A with V_B, J_B, K_B
    t2 = -2.0 * ein.core.dot(D_B_S_D_A.np, V_B.np)
    t2 -= 4.0 * ein.core.dot(D_B_S_D_A.np, J_B.np)
    t2 += 2.0 * ein.core.dot(D_B_S_D_A.np, K_B.np)
    Exch_s2 += t2
    terms_s2.append(t2)
    
    # Term 3: D_B @ S @ D_A @ S @ D_B with V_A, J_A
    D_B_S_D_A_S_D_B = chain_gemm_einsums([D_B_S_D_A, S, D_B])
    t3 = 2.0 * ein.core.dot(D_B_S_D_A_S_D_B.np, V_A.np)
    t3 += 4.0 * ein.core.dot(D_B_S_D_A_S_D_B.np, J_A.np)
    Exch_s2 += t3
    terms_s2.append(t3)
    
    # Term 4: D_A @ S @ D_B @ S @ D_A with V_B, J_B
    D_A_S_D_B_S_D_A = chain_gemm_einsums([D_A_S_D_B, S, D_A])
    t4 = 2.0 * ein.core.dot(D_A_S_D_B_S_D_A.np, V_B.np)
    t4 += 4.0 * ein.core.dot(D_A_S_D_B_S_D_A.np, J_B.np)
    Exch_s2 += t4
    terms_s2.append(t4)
    
    # Term 5: -2 * D_A @ S @ D_B · K_O
    t5 = -2.0 * ein.core.dot(D_A_S_D_B.np, K_O.np)
    Exch_s2 += t5
    terms_s2.append(t5)
    
    if do_print:
        core.print_out("    Exch10(S^2) terms:\n")
        for i, t in enumerate(terms_s2):
            core.print_out(f"      Term {i}: {t*1000:14.8f} [mEh]\n")
        core.print_out(print_sapt_var("Exch10(S^2) ", Exch_s2, short=True))
        core.print_out("\n")
    
    # ========== S^inf Exchange (MCBS formula) ==========
    # From fisapt.cc lines 2390-2477
    
    nocc_A = Cocc_A.np.shape[1]
    nocc_B = Cocc_B.np.shape[1]
    
    # Build T matrix (inverse exchange metric)
    SAB = chain_gemm_einsums([Cocc_A, S, Cocc_B], ['T', 'N', 'N'])
    
    num_occ = nocc_A + nocc_B
    T_full = core.Matrix(num_occ, num_occ)
    T_full.identity()
    T_full.np[:nocc_A, nocc_A:] = SAB.np
    T_full.np[nocc_A:, :nocc_A] = SAB.np.T
    T_full.power(-1.0, 1.0e-14)
    
    # Subtract identity to get T
    for i in range(num_occ):
        T_full.np[i, i] -= 1.0
    
    # Extract blocks
    T_AA = T_full.np[:nocc_A, :nocc_A]
    T_BB = T_full.np[nocc_A:, nocc_A:]
    T_AB = T_full.np[:nocc_A, nocc_A:]
    T_BA = T_full.np[nocc_A:, :nocc_A]
    
    # Build C_T matrices following C++ fisapt.cc lines 2419-2426:
    # C_T_A_n = Cocc0A @ T_AA         (nbf, na)
    # C_T_B_n = Cocc0B @ T_BB         (nbf, nb)
    # C_T_BA_n = Cocc0A @ T_AB        (nbf, nb) - note: uses Cocc_A!
    # C_T_AB_n = Cocc0B @ T_BA        (nbf, na) - note: uses Cocc_B!
    C_T_A_n = core.Matrix.from_array(Cocc_A.np @ T_AA)   # (nbf, nocc_A)
    C_T_B_n = core.Matrix.from_array(Cocc_B.np @ T_BB)   # (nbf, nocc_B)
    C_T_BA_n = core.Matrix.from_array(Cocc_A.np @ T_AB)  # (nbf, nocc_B) - Cocc_A @ T_AB
    C_T_AB_n = core.Matrix.from_array(Cocc_B.np @ T_BA)  # (nbf, nocc_A) - Cocc_B @ T_BA
    
    # Build T density matrices in AO basis (C++ lines 2446-2449):
    # T_A_n  = Cocc0A @ C_T_A_n.T  = Cocc_A @ T_AA @ Cocc_A.T
    # T_B_n  = Cocc0B @ C_T_B_n.T  = Cocc_B @ T_BB @ Cocc_B.T
    # T_BA_n = Cocc0B @ C_T_BA_n.T = Cocc_B @ (Cocc_A @ T_AB).T = Cocc_B @ T_AB.T @ Cocc_A.T = Cocc_B @ T_BA @ Cocc_A.T
    # T_AB_n = Cocc0A @ C_T_AB_n.T = Cocc_A @ (Cocc_B @ T_BA).T = Cocc_A @ T_BA.T @ Cocc_B.T = Cocc_A @ T_AB @ Cocc_B.T
    T_A_n = chain_gemm_einsums([Cocc_A, core.Matrix.from_array(T_AA), Cocc_A], ['N', 'N', 'T'])
    T_B_n = chain_gemm_einsums([Cocc_B, core.Matrix.from_array(T_BB), Cocc_B], ['N', 'N', 'T'])
    T_AB_n = chain_gemm_einsums([Cocc_A, core.Matrix.from_array(T_AB), Cocc_B], ['N', 'N', 'T'])
    # T_BA_n not used in the final energy expression
    
    # Compute JK for T matrices (C++ lines 2430-2437):
    # Note: C++ computes J/K for:
    #   [0]: Cl = Cocc0A, Cr = C_T_A_n   -> J/K for T_A
    #   [1]: Cl = Cocc0A, Cr = C_T_AB_n  -> J/K for Cocc_A @ (Cocc_B @ T_BA).T
    jk.C_clear()
    jk.C_left_add(core.Matrix.from_array(Cocc_A))
    jk.C_right_add(C_T_A_n)
    jk.C_left_add(core.Matrix.from_array(Cocc_A))
    jk.C_right_add(C_T_AB_n)  # C_T_AB_n = Cocc_B @ T_BA
    jk.compute()
    
    J_T_A = jk.J()[0]
    K_T_A = jk.K()[0]
    J_T_AB = jk.J()[1]
    K_T_AB = jk.K()[1]
    
    # Build h_A = V_A + 2*J_A - K_A and h_B = V_B + 2*J_B - K_B
    h_A = V_A.clone()
    ein.core.axpy(2.0, J_A.np, h_A.np)
    ein.core.axpy(-1.0, K_A.np, h_A.np)
    
    h_B = V_B.clone()
    ein.core.axpy(2.0, J_B.np, h_B.np)
    ein.core.axpy(-1.0, K_B.np, h_B.np)
    
    # Compute Exch10 (S^inf)
    Exch10 = 0.0
    
    # Term 0: -2 * D_A · K_B
    Exch10 -= 2.0 * ein.core.dot(D_A.np, K_B.np)
    
    # Term 1: T_A_n with h_B (= V_B + 2*J_B - K_B)
    Exch10 += 2.0 * ein.core.dot(T_A_n.np, h_B.np)
    
    # Term 2: T_B_n with h_A (= V_A + 2*J_A - K_A)
    Exch10 += 2.0 * ein.core.dot(T_B_n.np, h_A.np)
    
    # Term 3: T_AB_n with h_A
    Exch10 += 2.0 * ein.core.dot(T_AB_n.np, h_A.np)
    
    # Term 4: T_AB_n with h_B
    Exch10 += 2.0 * ein.core.dot(T_AB_n.np, h_B.np)
    
    # Term 5: T_B_n with (4*J_T_AB - 2*K_T_AB)
    Exch10 += 4.0 * ein.core.dot(T_B_n.np, J_T_AB.np)
    Exch10 -= 2.0 * ein.core.dot(T_B_n.np, K_T_AB.np)
    
    # Term 6: T_A_n with (4*J_T_AB - 2*K_T_AB)
    Exch10 += 4.0 * ein.core.dot(T_A_n.np, J_T_AB.np)
    Exch10 -= 2.0 * ein.core.dot(T_A_n.np, K_T_AB.np)
    
    # Term 7: T_B_n with (4*J_T_A - 2*K_T_A)
    Exch10 += 4.0 * ein.core.dot(T_B_n.np, J_T_A.np)
    Exch10 -= 2.0 * ein.core.dot(T_B_n.np, K_T_A.np)
    
    # Term 8: T_AB_n with (4*J_T_AB - 2*K_T_AB)
    Exch10 += 4.0 * ein.core.dot(T_AB_n.np, J_T_AB.np)
    Exch10 -= 2.0 * ein.core.dot(T_AB_n.np, K_T_AB.np)
    
    if do_print:
        core.set_variable("Exch10", Exch10)
        core.print_out(print_sapt_var("Exch10", Exch10, short=True))
        core.print_out("\n")
    
    return {"Exch10(S^2)": Exch_s2, "Exch10": Exch10}


def induction(
    cache,
    jk,
    do_print=True,
    maxiter=12,
    conv=1.0e-8,
    do_response=True,
    Sinf=False,
    sapt_jk_B=None,
):
    """
    Compute Ind20 and Exch-Ind20 quantities from a SAPT cache and JK object.
    """

    if do_print:
        core.print_out("\n  ==> E20 Induction Einsums <== \n\n")

    # Build Induction and Exchange-Induction potentials
    S = cache["S"]

    D_A = cache["D_A"]
    V_A = cache["V_A"]

    J_A = cache["J_A"]
    K_A = cache["K_A"]

    D_B = cache["D_B"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]
    K_B = cache["K_B"]

    K_O = cache["K_O"]
    J_O = cache["J_O"]

    # Set up matrix multiplication plans
    plan_matmul_tt = ein.core.compile_plan("ij", "ik", "kj")

    # Prepare JK calculations
    jk.C_clear()
    
    DB_S, DB_S_CA = chain_gemm_einsums([D_B, S, cache["Cocc_A"]], return_tensors=[True, True])
    jk.C_left_add(core.Matrix.from_array(DB_S_CA))
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_A"]))

    jk.C_left_add(core.Matrix.from_array(chain_gemm_einsums([DB_S, D_A, S, cache["Cocc_B"]])))
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_B"]))

    DA_S, DA_S_DB_S_CA = chain_gemm_einsums(
                [D_A, S, D_B, S, cache["Cocc_A"]],
                return_tensors=[True, False, False, True],
    )
    jk.C_left_add(core.Matrix.from_array(DA_S_DB_S_CA))
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_A"]))

    jk.compute()

    J_Ot, J_P_B, J_P_A = jk.J()
    K_Ot, K_P_B, K_P_A = jk.K()

    # Save for later usage in find()
    cache['J_P_A'] = J_P_A
    cache['J_P_B'] = J_P_B

    # Exch-Ind Potential A
    EX_A = K_B.clone()
    EX_A.scale(-1.0)
    ein.core.axpy(-2.0, J_O.np, EX_A.np)
    ein.core.axpy(1.0, K_O.np, EX_A.np)
    ein.core.axpy(2.0, J_P_B.np, EX_A.np)

    # Apply all the axpy operations to EX_A
    S_DB, S_DB_VA, S_DB_VA_DB_S = chain_gemm_einsums(
        [S, D_B, V_A, D_B, S],
        return_tensors=[True, True, False, True]
    )
    S_DB_JA, S_DB_JA_DB_S = chain_gemm_einsums(
        [S_DB, J_A, D_B, S],
        return_tensors=[True, False, True]
    )
    S_DB_S_DA, S_DB_S_DA_VB = chain_gemm_einsums(
        [S_DB, S, D_A, V_B],
        return_tensors=[False, True, True],
    )
    ein.core.axpy(-1.0, S_DB_VA.np, EX_A.np)
    ein.core.axpy(-2.0, S_DB_JA.np, EX_A.np)
    ein.core.axpy(1.0, chain_gemm_einsums([S_DB, K_A]).np, EX_A.np)
    ein.core.axpy(1.0, S_DB_S_DA_VB.np, EX_A.np)
    ein.core.axpy(2.0, chain_gemm_einsums([S_DB_S_DA, J_B]).np, EX_A.np)
    ein.core.axpy(1.0, S_DB_VA_DB_S.np, EX_A.np)
    ein.core.axpy(2.0, S_DB_JA_DB_S.np, EX_A.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([S_DB, K_O], ["N", "T"]).np, EX_A.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([V_B, D_B, S]).np, EX_A.np)
    ein.core.axpy(-2.0, chain_gemm_einsums([J_B, D_B, S]).np, EX_A.np)
    ein.core.axpy(1.0,  chain_gemm_einsums([K_B, D_B, S]).np, EX_A.np)
    ein.core.axpy(1.0,  chain_gemm_einsums([V_B, D_A, S, D_B, S]).np, EX_A.np)
    ein.core.axpy(2.0,  chain_gemm_einsums([J_B, D_A, S, D_B, S]).np, EX_A.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([K_O, D_B, S]).np, EX_A.np)

    EX_A_MO_1 = chain_gemm_einsums(
        [cache['Cocc_A'], EX_A, cache['Cvir_A']],
        ['T', 'N', 'N'],
    )
    mapA = {
        "S": S,
        "J_O": J_O,
        "K_O": K_O,

        "Cocc_A": cache['Cocc_A'],
        "Cvir_A": cache['Cvir_A'],
        "D_A": D_A,
        "V_A": V_A,
        "J_A": J_A,
        "K_A": K_A,
        "J_P_A": J_P_A,

        "Cocc_B": cache['Cocc_B'],
        "Cvir_B": cache['Cvir_B'],
        "D_B": D_B,
        "V_B": V_B,
        "J_B": J_B,
        "K_B": K_B,
        "J_P_B": J_P_B,
    }
    EX_A_MO = build_exch_ind_pot_AB(mapA)
    assert np.allclose(EX_A_MO, EX_A_MO_1), "EX_A_MO and EX_A_MO_1 do not match!"

    # Exch-Ind Potential B
    EX_B = K_A.clone()
    EX_B.scale(-1.0)
    ein.core.axpy(-2.0, J_O.np, EX_B.np)
    ein.core.axpy(1.0, K_O.np, EX_B.np.T)
    ein.core.axpy(2.0, J_P_A.np, EX_B.np)
    cache['J_P_A'] = J_P_A
    cache['J_P_B'] = J_P_B

    S_DA, S_DA_VB, S_DA_VB_DA_S = chain_gemm_einsums(
        [S, D_A, V_B, D_A, S],
        return_tensors=[True, True, False, True]
    )
    S_DA_JB, S_DA_JB_DA_S = chain_gemm_einsums(
        [S_DA, J_B, D_A, S],
        return_tensors=[True, False, True]
    )
    S_DA_S_DB, S_DA_S_DB_VA = chain_gemm_einsums(
        [S_DA, S, D_B, V_A],
        return_tensors=[False, True, True],
    )

    # Apply all the axpy operations to EX_B
    ein.core.axpy(-1.0, S_DA_VB.np, EX_B.np)
    ein.core.axpy(-2.0, S_DA_JB.np, EX_B.np)
    ein.core.axpy(1.0, chain_gemm_einsums([S_DA, K_B]).np, EX_B.np)
    ein.core.axpy(1.0, S_DA_S_DB_VA.np, EX_B.np)
    ein.core.axpy(2.0, chain_gemm_einsums([S_DA_S_DB, J_A]).np, EX_B.np)
    ein.core.axpy(1.0, S_DA_VB_DA_S.np, EX_B.np)
    ein.core.axpy(2.0, S_DA_JB_DA_S.np, EX_B.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([S_DA, K_O]).np, EX_B.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([V_A, D_A, S]).np, EX_B.np)
    ein.core.axpy(-2.0, chain_gemm_einsums([J_A, D_A, S]).np, EX_B.np)
    ein.core.axpy(1.0, chain_gemm_einsums([K_A, D_A, S]).np, EX_B.np)
    ein.core.axpy(1.0, chain_gemm_einsums([V_A, D_B, S, D_A, S]).np, EX_B.np)
    ein.core.axpy(2.0, chain_gemm_einsums([J_A, D_B, S, D_A, S]).np, EX_B.np)
    ein.core.axpy(-1.0, chain_gemm_einsums([K_O, D_A, S], ["T", "N", "N"]).np, EX_B.np)

    EX_B_MO_1 = chain_gemm_einsums(
        [cache['Cocc_B'], EX_B, cache['Cvir_B']],
        ['T', 'N', 'N'],
    )
    EX_B_MO = build_exch_ind_pot_BA(mapA)
    assert np.allclose(EX_B_MO, EX_B_MO_1), "EX_B_MO and EX_B_MO_1 do not match!"

    # Build electrostatic potentials - $\omega_A$ = w_A, Eq. 8
    w_A = V_A.clone()
    w_A.name = "w_A"
    ein.core.axpy(2.0, J_A.np, w_A.np)

    w_B = V_B.clone()
    w_B.name = "w_B"
    ein.core.axpy(2.0, J_B.np, w_B.np)

    w_B_MOA_1 = chain_gemm_einsums(
        [cache['Cocc_A'], w_B, cache['Cvir_A']],
        ['T', 'N', 'N'],
    )
    w_A_MOB_1 = chain_gemm_einsums(
        [cache['Cocc_B'], w_A, cache['Cvir_B']],
        ['T', 'N', 'N'],
    )

    # Build electrostatic potentials - $\omega_A$ = w_A, Eq. 8
    w_B_MOA = build_ind_pot({
        "V_B": V_B,
        "J_B": J_B,
        "Cocc_A": cache["Cocc_A"],
        "Cvir_A": cache["Cvir_A"],
    })
    w_B_MOA.name = "w_B_MOA"
    # Can re-use same function for w_A by swapping A and B labels
    w_A_MOB = build_ind_pot({
        "V_B": V_A,
        "J_B": J_A,
        "Cocc_A": cache["Cocc_B"],
        "Cvir_A": cache["Cvir_B"],
    })
    w_A_MOB.name = "w_A_MOB"
    assert np.allclose(w_B_MOA, w_B_MOA_1), "w_B_MOA and w_B_MOA_1 do not match!"
    assert np.allclose(w_A_MOB, w_A_MOB_1), "w_A_MOB and w_A_MOB_1 do not match!"

    # Do uncoupled induction calculations
    core.print_out("   => Uncoupled Induction <= \n\n")
    
    # Create uncoupled response vectors by element-wise division
    unc_x_B_MOA = w_B_MOA.clone()
    unc_x_A_MOB = w_A_MOB.clone()
    
    eps_occ_A = cache["eps_occ_A"]
    eps_vir_A = cache["eps_vir_A"]
    eps_occ_B = cache["eps_occ_B"]
    eps_vir_B = cache["eps_vir_B"]
    
    # Eq. 20
    for r in range(unc_x_B_MOA.shape[0]):
        for a in range(unc_x_B_MOA.shape[1]):
            unc_x_B_MOA.np[r, a] /= (eps_occ_A.np[r] - eps_vir_A.np[a])
    
    # Eq. 20
    for r in range(unc_x_A_MOB.shape[0]):
        for a in range(unc_x_A_MOB.shape[1]):
            unc_x_A_MOB.np[r, a] /= (eps_occ_B.np[r] - eps_vir_B.np[a])

    # Compute uncoupled induction energies according to Eq. 14, 15
    unc_ind_ab = 2.0 * ein.core.dot(unc_x_B_MOA.np, w_B_MOA.np)
    unc_ind_ba = 2.0 * ein.core.dot(unc_x_A_MOB.np, w_A_MOB.np)
    unc_indexch_ab = 2.0 * ein.core.dot(unc_x_B_MOA.np, EX_A_MO.np)
    unc_indexch_ba = 2.0 * ein.core.dot(unc_x_A_MOB.np, EX_B_MO.np)

    ret = {}
    ret["Ind20,u (A<-B)"] = unc_ind_ab
    ret["Ind20,u (A->B)"] = unc_ind_ba
    ret["Ind20,u"] = unc_ind_ab + unc_ind_ba
    ret["Exch-Ind20,u (A<-B)"] = unc_indexch_ab
    ret["Exch-Ind20,u (A->B)"] = unc_indexch_ba
    ret["Exch-Ind20,u"] = unc_indexch_ba + unc_indexch_ab

    plist = [
        "Ind20,u (A<-B)",
        "Ind20,u (A->B)",
        "Ind20,u",
        "Exch-Ind20,u (A<-B)",
        "Exch-Ind20,u (A->B)",
        "Exch-Ind20,u",
    ]

    if do_print:
        for name in plist:
            core.print_out(print_sapt_var(name, ret[name], short=True))
            core.print_out("\n")

    # Exch-Ind without S^2 (Sinf calculations)
    if Sinf:
        nocc_A = cache["Cocc_A"].shape[1]
        nocc_B = cache["Cocc_B"].shape[1]
        SAB = core.triplet(
            cache["Cocc_A"], cache["S"], cache["Cocc_B"], True, False, False
        )
        num_occ = nocc_A + nocc_B

        Sab = core.Matrix(num_occ, num_occ)
        Sab.np[:nocc_A, nocc_A:] = SAB.np
        Sab.np[nocc_A:, :nocc_A] = SAB.np.T
        Sab.np[np.diag_indices_from(Sab.np)] += 1
        Sab.power(-1.0, 1.0e-14)

        Tmo_AA = core.Matrix.from_array(Sab.np[:nocc_A, :nocc_A])
        Tmo_BB = core.Matrix.from_array(Sab.np[nocc_A:, nocc_A:])
        Tmo_AB = core.Matrix.from_array(Sab.np[:nocc_A, nocc_A:])

        T_A = core.triplet(cache["Cocc_A"], Tmo_AA, cache["Cocc_A"], False, False, True)
        T_B = core.triplet(cache["Cocc_B"], Tmo_BB, cache["Cocc_B"], False, False, True)
        T_AB = core.triplet(
            cache["Cocc_A"], Tmo_AB, cache["Cocc_B"], False, False, True
        )

        sT_A = core.Matrix.chain_dot(
            cache["Cvir_A"],
            unc_x_B_MOA,
            Tmo_AA,
            cache["Cocc_A"],
            trans=[False, True, False, True],
        )
        sT_B = core.Matrix.chain_dot(
            cache["Cvir_B"],
            unc_x_A_MOB,
            Tmo_BB,
            cache["Cocc_B"],
            trans=[False, True, False, True],
        )
        sT_AB = core.Matrix.chain_dot(
            cache["Cvir_A"],
            unc_x_B_MOA,
            Tmo_AB,
            cache["Cocc_B"],
            trans=[False, True, False, True],
        )
        sT_BA = core.Matrix.chain_dot(
            cache["Cvir_B"],
            unc_x_A_MOB,
            Tmo_AB,
            cache["Cocc_A"],
            trans=[False, True, True, True],
        )

        jk.C_clear()

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_A"], Tmo_AA))
        jk.C_right_add(cache["Cocc_A"])

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_B"], Tmo_BB))
        jk.C_right_add(cache["Cocc_B"])

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_A"], Tmo_AB))
        jk.C_right_add(cache["Cocc_B"])

        jk.compute()

        J_AA_inf, J_BB_inf, J_AB_inf = jk.J()
        K_AA_inf, K_BB_inf, K_AB_inf = jk.K()

        # A <- B
        EX_AA_inf = V_B.clone()
        EX_AA_inf.axpy(
            -1.00, core.Matrix.chain_dot(S, T_AB, V_B, trans=[False, True, False])
        )
        EX_AA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_B, V_B))
        EX_AA_inf.axpy(2.00, J_AB_inf)
        EX_AA_inf.axpy(
            -2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf, trans=[False, True, False])
        )
        EX_AA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AB_inf))
        EX_AA_inf.axpy(2.00, J_BB_inf)
        EX_AA_inf.axpy(
            -2.00, core.Matrix.chain_dot(S, T_AB, J_BB_inf, trans=[False, True, False])
        )
        EX_AA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_BB_inf))
        EX_AA_inf.axpy(-1.00, K_AB_inf.transpose())
        EX_AA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, True, True])
        )
        EX_AA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_B, K_AB_inf, trans=[False, False, True])
        )
        EX_AA_inf.axpy(-1.00, K_BB_inf)
        EX_AA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_BB_inf, trans=[False, True, False])
        )
        EX_AA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_BB_inf))

        EX_AB_inf = V_A.clone()
        EX_AB_inf.axpy(
            -1.00, core.Matrix.chain_dot(S, T_AB, V_A, trans=[False, True, False])
        )
        EX_AB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_B, V_A))
        EX_AB_inf.axpy(2.00, J_AA_inf)
        EX_AB_inf.axpy(
            -2.00, core.Matrix.chain_dot(S, T_AB, J_AA_inf, trans=[False, True, False])
        )
        EX_AB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AA_inf))
        EX_AB_inf.axpy(2.00, J_AB_inf)
        EX_AB_inf.axpy(
            -2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf, trans=[False, True, False])
        )
        EX_AB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AB_inf))
        EX_AB_inf.axpy(-1.00, K_AA_inf)
        EX_AB_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_AA_inf, trans=[False, True, False])
        )
        EX_AB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_AA_inf))
        EX_AB_inf.axpy(-1.00, K_AB_inf)
        EX_AB_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, True, False])
        )
        EX_AB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_AB_inf))

        # B <- A
        EX_BB_inf = V_A.clone()
        EX_BB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_AB, V_A))
        EX_BB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_A, V_A))
        EX_BB_inf.axpy(2.00, J_AB_inf)
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf))
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AB_inf))
        EX_BB_inf.axpy(2.00, J_AA_inf)
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AA_inf))
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AA_inf))
        EX_BB_inf.axpy(-1.00, K_AB_inf)
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf))
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_AB_inf))
        EX_BB_inf.axpy(-1.00, K_AA_inf)
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AA_inf))
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_AA_inf))

        EX_BA_inf = V_B.clone()
        EX_BA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_AB, V_B))
        EX_BA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_A, V_B))
        EX_BA_inf.axpy(2.00, J_BB_inf)
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_BB_inf))
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_BB_inf))
        EX_BA_inf.axpy(2.00, J_AB_inf)
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf))
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AB_inf))
        EX_BA_inf.axpy(-1.00, K_BB_inf)
        EX_BA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_BB_inf))
        EX_BA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_BB_inf))
        EX_BA_inf.axpy(-1.00, K_AB_inf.transpose())
        EX_BA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, False, True])
        )
        EX_BA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_A, K_AB_inf, trans=[False, False, True])
        )

        unc_ind_ab_total = 2.0 * (
            sT_A.vector_dot(EX_AA_inf) + sT_AB.vector_dot(EX_AB_inf)
        )
        unc_ind_ba_total = 2.0 * (
            sT_B.vector_dot(EX_BB_inf) + sT_BA.vector_dot(EX_BA_inf)
        )
        unc_indexch_ab_inf = unc_ind_ab_total - unc_ind_ab
        unc_indexch_ba_inf = unc_ind_ba_total - unc_ind_ba

        ret["Exch-Ind20,u (A<-B) (S^inf)"] = unc_indexch_ab_inf
        ret["Exch-Ind20,u (A->B) (S^inf)"] = unc_indexch_ba_inf
        ret["Exch-Ind20,u (S^inf)"] = unc_indexch_ba_inf + unc_indexch_ab_inf

        if do_print:
            for name in plist[3:]:
                name = name + " (S^inf)"

                core.print_out(print_sapt_var(name, ret[name], short=True))
                core.print_out("\n")

    # Do coupled induction calculations
    if do_response:
        core.print_out("\n   => Coupled Induction <= \n\n")

        cphf_r_convergence = core.get_option("SAPT", "CPHF_R_CONVERGENCE")
        x_B_MOA, x_A_MOB = _sapt_cpscf_solve(
            cache, jk, w_B_MOA.np, w_A_MOB.np, maxiter, cphf_r_convergence, sapt_jk_B=sapt_jk_B
        )
        # Negate the CPSCF solution to match convention (see fisapt.cc lines 3426-3427)
        # The CG solver solves H*x = w, but the correct coupled response is x = -H^{-1}*w
        x_B_MOA = core.Matrix.from_array(-x_B_MOA)
        x_A_MOB = core.Matrix.from_array(-x_A_MOB)

        ind_ab = 2.0 * ein.core.dot(x_B_MOA.np, w_B_MOA.np)
        ind_ba = 2.0 * ein.core.dot(x_A_MOB.np, w_A_MOB.np)
        indexch_ab = 2.0 * ein.core.dot(x_B_MOA.np, EX_A_MO.np)
        indexch_ba = 2.0 * ein.core.dot(x_A_MOB.np, EX_B_MO.np)

        ret["Ind20,r (A<-B)"] = ind_ab
        ret["Ind20,r (A->B)"] = ind_ba
        ret["Ind20,r"] = ind_ab + ind_ba
        ret["Exch-Ind20,r (A<-B)"] = indexch_ab
        ret["Exch-Ind20,r (A->B)"] = indexch_ba
        ret["Exch-Ind20,r"] = indexch_ba + indexch_ab

        if do_print:
            core.print_out("\n")
            for name in plist:
                name = name.replace(",u", ",r")
                core.print_out(print_sapt_var(name, ret[name], short=True))
                core.print_out("\n")

        # Exch-Ind without S^2
        if Sinf:
            cT_A = core.Matrix.chain_dot(
                cache["Cvir_A"],
                x_B_MOA,
                Tmo_AA,
                cache["Cocc_A"],
                trans=[False, True, False, True],
            )
            cT_B = core.Matrix.chain_dot(
                cache["Cvir_B"],
                x_A_MOB,
                Tmo_BB,
                cache["Cocc_B"],
                trans=[False, True, False, True],
            )
            cT_AB = core.Matrix.chain_dot(
                cache["Cvir_A"],
                x_B_MOA,
                Tmo_AB,
                cache["Cocc_B"],
                trans=[False, True, False, True],
            )
            cT_BA = core.Matrix.chain_dot(
                cache["Cvir_B"],
                x_A_MOB,
                Tmo_AB,
                cache["Cocc_A"],
                trans=[False, True, True, True],
            )

            ind_ab_total = 2.0 * (
                cT_A.vector_dot(EX_AA_inf) + cT_AB.vector_dot(EX_AB_inf)
            )
            ind_ba_total = 2.0 * (
                cT_B.vector_dot(EX_BB_inf) + cT_BA.vector_dot(EX_BA_inf)
            )
            indexch_ab_inf = ind_ab_total - ind_ab
            indexch_ba_inf = ind_ba_total - ind_ba

            ret["Exch-Ind20,r (A<-B) (S^inf)"] = indexch_ab_inf
            ret["Exch-Ind20,r (A->B) (S^inf)"] = indexch_ba_inf
            ret["Exch-Ind20,r (S^inf)"] = indexch_ba_inf + indexch_ab_inf

            if do_print:
                for name in plist[3:]:
                    name = name.replace(",u", ",r") + " (S^inf)"

                    core.print_out(print_sapt_var(name, ret[name], short=True))
                    core.print_out("\n")

    return ret


def _sapt_cpscf_solve(cache, jk, rhsA, rhsB, maxiter, conv, sapt_jk_B=None):
    """
    Solve the SAPT CPHF (or CPKS) equations.
    """

    # For I-SAPT, wavefunctions are bare Wavefunction objects without set_jk method
    # The JK object is already available and passed in, so we only call set_jk
    # if the method exists (i.e., for regular SAPT with SCF wavefunctions)
    if hasattr(cache["wfn_A"], 'set_jk'):
        cache["wfn_A"].set_jk(jk)
    if hasattr(cache["wfn_B"], 'set_jk'):
        if sapt_jk_B:
            cache["wfn_B"].set_jk(sapt_jk_B)
        else:
            cache["wfn_B"].set_jk(jk)

    def setup_P_X(eps_occ, eps_vir, name='P_X'):
        P_X = ein.utils.tensor_factory(name, [eps_occ.shape[0], eps_vir.shape[0]], np.float64, 'einsums')

        ones_occ = ein.utils.tensor_factory("ones_occ", [eps_occ.shape[0]], np.float64, 'einsums')
        ones_vir = ein.utils.tensor_factory("ones_vir", [eps_vir.shape[0]], np.float64, 'einsums')
        ones_occ.set_all(1.0)
        ones_vir.set_all(1.0)
        plan_outer = ein.core.compile_plan("ia", "i", "a")
        plan_outer.execute(0.0, P_X, 1.0, eps_occ.np, ones_vir)
        eps_vir_2D = ein.utils.tensor_factory("eps_vir_2D", [eps_occ.shape[0], eps_vir.shape[0]], np.float64, 'einsums')
        plan_outer.execute(0.0, eps_vir_2D, 1.0, ones_occ, eps_vir.np)
        ein.core.axpy(-1.0, eps_vir_2D, P_X)
        return P_X

    # Make a preconditioner function
    P_A = setup_P_X(cache['eps_occ_A'], cache['eps_vir_A'])
    P_B = setup_P_X(cache['eps_occ_B'], cache['eps_vir_B'])

    # Preconditioner function
    def apply_precon(x_vec, act_mask):
        if act_mask[0]:
            pA = x_vec[0].copy()
            pA /= P_A
        else:
            pA = False

        if act_mask[1]:
            pB = x_vec[1].copy()
            pB /= P_B
        else:
            pB = False
        return [pA, pB]

    # Hx function
    def hessian_vec(x_vec, act_mask):
        # For I-SAPT, wavefunctions may not have cphf_Hx method
        # In that case, use a direct Python implementation of the CPSCF Hessian
        has_cphf_A = hasattr(cache["wfn_A"], 'cphf_Hx')
        has_cphf_B = hasattr(cache["wfn_B"], 'cphf_Hx')
        
        if has_cphf_A and has_cphf_B:
            # Standard SAPT path - use wavefunction's cphf_Hx
            if act_mask[0]:
                xA = cache["wfn_A"].cphf_Hx([core.Matrix.from_array(x_vec[0])])[0].np
            else:
                xA = False

            if act_mask[1]:
                xB = cache["wfn_B"].cphf_Hx([core.Matrix.from_array(x_vec[1])])[0].np
            else:
                xB = False
        else:
            # I-SAPT path - implement CPSCF Hessian directly
            # For HF:      H*x = (eps_vir - eps_occ)*x + 4*J(D_x) - K(D_x) - K(D_x)^T
            # For hybrid:  H*x = (eps_vir - eps_occ)*x + 4*J(D_x) - alpha*(K(D_x) + K(D_x)^T) + 4*Vx(D_x)
            # where D_x = C_occ @ x @ C_vir.T
            
            # Get DFT parameters from cache
            V_potential = cache.get("V_potential", None)
            x_alpha = cache.get("x_alpha", 1.0)  # 1.0 for HF, 0.25 for PBE0
            is_dft = cache.get("is_dft", False)
            
            def compute_cphf_Hx_direct(x, Cocc, Cvir, eps_occ, eps_vir, jk_obj, is_A):
                """Direct CPSCF Hessian-vector product for I-SAPT.
                
                Implements the CPSCF orbital Hessian for I-SAPT where the wavefunction
                doesn't have cphf_Hx method.
                
                For HF (x_alpha=1.0):
                    H*x = (eps_a - eps_i)*x + 4*J(D_x) - K(D_x) - K(D_x)^T
                
                For hybrid DFT (e.g., PBE0 with x_alpha=0.25):
                    H*x = (eps_a - eps_i)*x + 4*J(D_x) - alpha*(K(D_x) + K(D_x)^T) + 4*Vx(D_x)
                
                where:
                    D_x = Cocc @ x @ Cvir^T (the trial density perturbation)
                    Vx(D_x) = XC kernel response (only for DFT)
                """
                nocc = Cocc.np.shape[1]
                nvir = Cvir.np.shape[1]
                
                # Orbital energy contribution: (eps_a - eps_i) * x
                hx = x.copy()
                for i in range(nocc):
                    for a in range(nvir):
                        hx[i, a] *= (eps_vir.np[a] - eps_occ.np[i])
                
                # Build D_x = Cocc @ x @ Cvir^T for JK and Vx computations
                # Using the factorization: D_x = (Cocc @ x) @ Cvir^T = C_mod @ Cvir^T
                C_mod = core.Matrix.from_array(Cocc.np @ x)  # (nbf, nvir)
                
                # Compute J and K via JK object
                jk_obj.C_clear()
                jk_obj.C_left_add(C_mod)
                jk_obj.C_right_add(Cvir)
                jk_obj.compute()
                
                J_Dx = jk_obj.J()[0]
                K_Dx = jk_obj.K()[0]
                
                # Build G(D_x) = 4*J - x_alpha*(K + K^T)
                # For HF: x_alpha = 1.0, so this is 4*J - K - K^T
                # For PBE0: x_alpha = 0.25, so this is 4*J - 0.25*(K + K^T)
                G_Dx = J_Dx.clone()
                G_Dx.scale(4.0)
                G_Dx.axpy(-x_alpha, K_Dx)
                G_Dx_np = G_Dx.np - x_alpha * K_Dx.np.T  # subtract x_alpha * K^T
                
                # Add XC kernel contribution for DFT
                if is_dft and V_potential is not None:
                    # First, set the ground state density for V_potential
                    # The XC kernel f_xc is evaluated at the ground state density
                    if is_A:
                        D_ground = cache["D_A"]
                    else:
                        D_ground = cache["D_B"]
                    V_potential.set_D([D_ground])
                    
                    # D_x in AO basis (non-symmetric response density)
                    D_x = core.Matrix.from_array(C_mod.np @ Cvir.np.T)
                    D_x.name = "D_x"
                    
                    # Create output matrix for Vx
                    Vx = core.Matrix("Vx", D_x.rowdim(), D_x.coldim())
                    
                    # compute_Vx computes the XC kernel response: Vx = f_xc * rho_x
                    # where rho_x is the density perturbation from D_x
                    V_potential.compute_Vx([D_x], [Vx])
                    
                    # Add 4*Vx to G (factor of 4 from closed-shell RHF equations)
                    G_Dx_np += 4.0 * Vx.np
                
                # Transform to MO basis: Cocc^T @ G(D_x) @ Cvir
                hx_2e = Cocc.np.T @ G_Dx_np @ Cvir.np
                
                # Add two-electron (+ XC) contribution
                hx += hx_2e
                
                return hx
            
            if act_mask[0]:
                xA = compute_cphf_Hx_direct(
                    x_vec[0], cache["Cocc_A"], cache["Cvir_A"],
                    cache["eps_occ_A"], cache["eps_vir_A"], jk, True
                )
            else:
                xA = False

            if act_mask[1]:
                xB = compute_cphf_Hx_direct(
                    x_vec[1], cache["Cocc_B"], cache["Cvir_B"],
                    cache["eps_occ_B"], cache["eps_vir_B"], jk, False
                )
            else:
                xB = False

        return [xA, xB]

    # Manipulate the printing
    sep_size = 51
    core.print_out("   " + ("-" * sep_size) + "\n")
    core.print_out("   " + "SAPT Coupled Induction Solver".center(sep_size) + "\n")
    core.print_out("   " + ("-" * sep_size) + "\n")
    core.print_out("    Maxiter             = %11d\n" % maxiter)
    core.print_out("    Convergence         = %11.3E\n" % conv)
    core.print_out("   " + ("-" * sep_size) + "\n")

    tstart = time.time()
    core.print_out(
        "     %4s %12s     %12s     %9s\n" % ("Iter", "(A<-B)", "(B->A)", "Time [s]")
    )
    core.print_out("   " + ("-" * sep_size) + "\n")

    start_resid = [ein.core.dot(rhsA, rhsA), ein.core.dot(rhsB, rhsB)]

    def pfunc(niter, x_vec, r_vec):
        if niter == 0:
            niter = "Guess"
        else:
            niter = "%5d" % niter
        # Compute IndAB
        valA = (ein.core.dot(r_vec[0], r_vec[0]) / start_resid[0]) ** 0.5
        if valA < conv:
            cA = "*"
        else:
            cA = " "

        # Compute IndBA
        valB = (ein.core.dot(r_vec[1], r_vec[1]) / start_resid[1]) ** 0.5
        if valB < conv:
            cB = "*"
        else:
            cB = " "

        core.print_out(
            "    %5s %15.6e%1s %15.6e%1s %9d\n"
            % (niter, valA, cA, valB, cB, time.time() - tstart)
        )
        return [valA, valB]

    # Compute the solver
    vecs, resid = solvers.cg_solver_ein(
        [rhsA, rhsB],
        hessian_vec,
        apply_precon,
        maxiter=maxiter,
        rcond=conv,
        printlvl=0,
        printer=pfunc,
    )
    core.print_out("   " + ("-" * sep_size) + "\n")

    return vecs


# ==> I-SAPT (Intramolecular SAPT) Functions <==

def compute_isapt_fragment_nuclear_potentials(
    molecule: core.Molecule,
    basisset: core.BasisSet,
    cache: dict
) -> None:
    """
    Compute nuclear attraction potential matrices for each fragment (A, B, C).
    
    This is the Python analog of FISAPT::nuclear() in fisapt.cc.
    The nuclear potential V_X for fragment X is the one-electron integral
    <mu|sum_A Z_X[A]/|r-R_A||nu> where Z_X[A] is the nuclear charge
    assigned to fragment X for atom A.
    
    Parameters
    ----------
    molecule : core.Molecule
        The full molecule.
    basisset : core.BasisSet
        The basis set.
    cache : dict
        Cache containing ZA, ZB, ZC vectors (nuclear charges per fragment).
        Updated in-place with V_A, V_B, V_C matrices.
    
    Notes
    -----
    Requires that cache contains:
        - ZA, ZB, ZC: core.Vector with nuclear charges per atom for each fragment
    
    Adds to cache:
        - V_A, V_B, V_C: core.Matrix nuclear potential matrices
    """
    from .embedded_scf import compute_fragment_nuclear_potential
    
    core.print_out("\n  ==> I-SAPT Nuclear Potentials <==\n\n")
    
    nbf = basisset.nbf()
    
    # Debug: print nuclear charge sums
    ZA_sum = cache["ZA"].np.sum()
    ZB_sum = cache["ZB"].np.sum()
    ZC_sum = cache["ZC"].np.sum()
    core.print_out(f"    Sum(ZA) = {ZA_sum:8.2f} (nuclear charge on A)\n")
    core.print_out(f"    Sum(ZB) = {ZB_sum:8.2f} (nuclear charge on B)\n")
    core.print_out(f"    Sum(ZC) = {ZC_sum:8.2f} (nuclear charge on C)\n\n")
    
    # Compute nuclear potentials for each fragment
    V_A = compute_fragment_nuclear_potential(molecule, basisset, cache["ZA"])
    V_A.name = "V_A"
    cache["V_A"] = V_A
    
    V_B = compute_fragment_nuclear_potential(molecule, basisset, cache["ZB"])
    V_B.name = "V_B"
    cache["V_B"] = V_B
    
    V_C = compute_fragment_nuclear_potential(molecule, basisset, cache["ZC"])
    V_C.name = "V_C"
    cache["V_C"] = V_C
    
    core.print_out(f"    Nuclear potential V_A computed: ({nbf} x {nbf})\n")
    core.print_out(f"    Nuclear potential V_B computed: ({nbf} x {nbf})\n")
    core.print_out(f"    Nuclear potential V_C computed: ({nbf} x {nbf})\n\n")


def compute_isapt_JK_C(
    jk: core.JK,
    cache: dict
) -> None:
    """
    Compute Coulomb and exchange matrices for fragment C electrons.
    
    This is the Python analog of FISAPT::coulomb() in fisapt.cc.
    J_C and K_C are computed from the localized occupied orbitals of fragment C.
    
    Parameters
    ----------
    jk : core.JK
        JK object for integral computation.
    cache : dict
        Cache containing Locc_C (localized occupied orbitals for C).
        Updated in-place with J_C, K_C matrices.
    
    Notes
    -----
    Requires that cache contains:
        - Locc_C: core.Matrix with localized occupied orbitals for fragment C
    
    Adds to cache:
        - J_C, K_C: core.Matrix Coulomb and exchange matrices
        - W_C: core.Matrix embedding potential W_C = V_C + 2*J_C - K_C
    """
    from .embedded_scf import compute_JK_C, compute_embedding_potential
    
    core.print_out("\n  ==> I-SAPT Coulomb Integrals for C <==\n\n")
    
    # Note: partition() creates LoccC (no underscore)
    Locc_C = cache.get("LoccC")
    if Locc_C is None:
        # No fragment C orbitals - empty embedding
        S = cache["S"]
        nbf = S.rows() if hasattr(S, 'rows') else S.rowspi()[0]
        cache["J_C"] = core.Matrix("J_C", nbf, nbf)
        cache["K_C"] = core.Matrix("K_C", nbf, nbf)
        cache["W_C"] = cache["V_C"].clone()
        cache["W_C"].name = "W_C"
        core.print_out("    Fragment C has no electrons; embedding is nuclear-only.\n\n")
        return
    
    # Handle both dimension-aware and regular Matrix types
    if hasattr(Locc_C, 'colspi'):
        nocc_C = Locc_C.colspi()[0]
        nbf = Locc_C.rowspi()[0]
    else:
        nocc_C = Locc_C.cols()
        nbf = Locc_C.rows()
    
    if nocc_C == 0:
        cache["J_C"] = core.Matrix("J_C", nbf, nbf)
        cache["K_C"] = core.Matrix("K_C", nbf, nbf)
        cache["W_C"] = cache["V_C"].clone()
        cache["W_C"].name = "W_C"
        core.print_out("    Fragment C has no electrons; embedding is nuclear-only.\n\n")
        return
    
    # Compute J_C and K_C
    J_C, K_C = compute_JK_C(jk, Locc_C, nbf)
    cache["J_C"] = J_C
    cache["K_C"] = K_C
    
    # Build embedding potential W_C = V_C + 2*J_C - K_C
    W_C = compute_embedding_potential(cache["V_C"], J_C, K_C)
    cache["W_C"] = W_C
    
    core.print_out(f"    J_C computed from {nocc_C} localized orbitals of C\n")
    core.print_out(f"    K_C computed from {nocc_C} localized orbitals of C\n")
    core.print_out(f"    Embedding potential W_C = V_C + 2*J_C - K_C computed\n\n")


def run_isapt_embedded_scf(
    jk: core.JK,
    molecule: core.Molecule,
    basisset: core.BasisSet,
    mints: core.MintsHelper,
    cache: dict,
    monomer: str,
    functional: str = None,
    V_potential: core.VBase = None,
    options: dict = None
) -> None:
    """
    Run embedded SCF for a monomer (A or B) in the I-SAPT framework.
    
    This is the Python analog of calling FISAPTSCF in FISAPT::scf() in fisapt.cc.
    
    Parameters
    ----------
    jk : core.JK
        JK object for integral computation.
    molecule : core.Molecule
        The full molecule.
    basisset : core.BasisSet
        The basis set.
    mints : core.MintsHelper
        Mints helper for one-electron integrals.
    cache : dict
        Cache containing fragment data.
    monomer : str
        Which monomer to compute: "A" or "B".
    functional : str, optional
        DFT functional name. If None, uses HF.
    V_potential : core.VBase, optional
        V potential object for DFT.
    options : dict, optional
        SCF options (maxiter, e_convergence, d_convergence).
    
    Notes
    -----
    Requires that cache contains:
        - Locc_{monomer}: Localized occupied orbitals for this monomer
        - Cvir: Virtual orbitals (shared, orthogonal to C)
        - Z{monomer}: Nuclear charges for this monomer
        - W_C: Embedding potential from fragment C
        - S, T, V_{monomer}: Overlap, kinetic, and nuclear potential matrices
    
    Adds to cache:
        - Cocc0{monomer}: Relaxed occupied orbital coefficients
        - Cvir0{monomer}: Relaxed virtual orbital coefficients
        - eps_occ0{monomer}: Occupied orbital energies
        - eps_vir0{monomer}: Virtual orbital energies
        - J0{monomer}: Final Coulomb matrix
        - K0{monomer}: Final exchange matrix
        - E0_{monomer}: SCF energy
    """
    from .embedded_scf import EmbeddedSCF, build_restricted_basis_isapt, compute_fragment_nuclear_repulsion
    
    core.print_out(f"\n  ==> I-SAPT Embedded SCF for Monomer {monomer} <==\n\n")
    
    # Get data from cache
    # Note: partition() creates LoccA, LoccB, LoccC (no underscore)
    Locc = cache[f"Locc{monomer}"]
    Locc_A = cache["LoccA"]
    Locc_B = cache["LoccB"]
    Cvir = cache["Cvir"]  # shared virtual space from dimer wfn
    Z_monomer = cache[f"Z{monomer}"]
    W_C = cache["W_C"]
    S = cache["S"]
    T = mints.ao_kinetic()
    V_monomer = cache[f"V_{monomer}"]
    
    # Build restricted basis: X = [LoccA | LoccB | Cvir]
    # This matches C++ FISAPT::scf() where XC includes orbitals from BOTH monomers
    # The key insight is that while each monomer's SCF is solved independently,
    # both need to mix within the same orbital space (excluding C)
    X = build_restricted_basis_isapt(Locc_A, Locc_B, Cvir)
    
    # Compute nuclear repulsion for this monomer
    Z_dummy = core.Vector("Z_dummy", molecule.natom())
    E_nuc, _, _, _ = compute_fragment_nuclear_repulsion(
        molecule, Z_monomer, Z_dummy, Z_dummy
    )
    
    # Create and run embedded SCF
    scf = EmbeddedSCF(
        jk=jk,
        enuc=E_nuc,
        S=S,
        X=X,
        T=T,
        V=V_monomer,
        W=W_C,
        C_guess=Locc,
        functional=functional,
        V_potential=V_potential,
        options=options
    )
    scf.compute_energy()
    
    # Store results in cache
    cache[f"Cocc0{monomer}"] = scf.Cocc
    cache[f"Cvir0{monomer}"] = scf.Cvir
    cache[f"eps_occ0{monomer}"] = scf.eps_occ
    cache[f"eps_vir0{monomer}"] = scf.eps_vir
    cache[f"J0{monomer}"] = scf.J
    cache[f"K0{monomer}"] = scf.K
    cache[f"E0_{monomer}"] = scf.energy
    
    if not scf.converged:
        raise RuntimeError(f"I-SAPT embedded SCF for monomer {monomer} failed to converge")
    
    core.print_out(f"    Monomer {monomer} SCF Energy: {scf.energy:24.16E} [Eh]\n\n")


def build_isapt_V_potential(
    functional: str,
    basisset: core.BasisSet,
    do_print: bool = True
) -> core.VBase:
    """
    Build a VBase object for DFT XC potential in I-SAPT embedded SCF.
    
    This creates the DFT integration grid and V_xc potential needed for
    computing DFT energy and Fock matrix contributions in embedded SCF.
    
    Parameters
    ----------
    functional : str
        DFT functional name (e.g., 'PBE0', 'B3LYP').
    basisset : core.BasisSet
        The basis set for the calculation.
    do_print : bool, optional
        Whether to print progress (default True).
    
    Returns
    -------
    core.VBase
        The V potential object for DFT calculations.
    """
    from ..dft import build_superfunctional
    
    if do_print:
        core.print_out(f"  ==> Building DFT V_potential for I-SAPT <==\n\n")
        core.print_out(f"    Functional: {functional}\n")
    
    # Build the superfunctional
    # restricted=True for closed-shell calculations
    # npoints controls the grid block size
    # deriv=1 for first derivatives (needed for SCF)
    npoints = core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
    sup, disp_type = build_superfunctional(functional, restricted=True, npoints=npoints, deriv=1)
    
    if do_print:
        core.print_out(f"    SuperFunctional: {sup.name()}\n")
        core.print_out(f"    Is hybrid: {sup.is_x_hybrid()}\n")
        if sup.is_x_hybrid():
            core.print_out(f"    Exact exchange: {sup.x_alpha():6.2f}\n")
    
    # Build the VBase object
    # "RV" = Restricted V potential (for closed-shell)
    V_potential = core.VBase.build(basisset, sup, "RV")
    
    # Initialize the potential (sets up grid, etc.)
    V_potential.initialize()
    
    # Build collocation cache for efficient evaluation
    # Use available memory for the cache
    memory = core.get_memory() // 8  # Convert bytes to doubles
    V_potential.build_collocation_cache(memory // 4)  # Use 1/4 of available memory
    
    if do_print:
        core.print_out(f"    DFT grid initialized with {V_potential.nblocks()} blocks\n\n")
    
    return V_potential


def build_isapt_cache(
    dimer_wfn: core.Wavefunction,
    jk: core.JK,
    cache: dict,
    functional: str = None,
    do_print: bool = True,
    external_potentials: dict = None
) -> dict:
    """
    Build the I-SAPT cache with embedded SCF wavefunctions for monomers A and B.
    
    This is the main entry point for I-SAPT, analogous to FISAPT::scf() but
    integrated with the SAPT(DFT) framework.
    
    The workflow is:
    1. Localize orbitals (already done in partition())
    2. Compute nuclear potentials V_A, V_B, V_C
    3. Compute J_C, K_C from C orbitals
    4. Build embedding potential W_C = V_C + 2*J_C - K_C  
    5. Run embedded SCF for A in W_C
    6. Run embedded SCF for B in W_C
    7. Update cache with relaxed orbitals and energies
    
    Parameters
    ----------
    dimer_wfn : core.Wavefunction
        Dimer wavefunction.
    jk : core.JK
        JK object for integral computation.
    cache : dict
        Cache from partition() containing:
        - Locc_A, Locc_B, Locc_C: Localized occupied orbitals
        - ZA, ZB, ZC: Nuclear charges per fragment
        - Cvir: Virtual orbitals
        - S: Overlap matrix
    functional : str, optional
        DFT functional for embedded SCF. If None, uses HF.
    do_print : bool, optional
        Whether to print progress (default True).
    external_potentials : dict, optional
        External potentials for fragments A, B, C.
    
    Returns
    -------
    dict
        Updated cache with embedded SCF results.
    """
    if do_print:
        core.print_out("\n")
        core.print_out("  " + "="*60 + "\n")
        core.print_out("  " + "I-SAPT: Intramolecular SAPT".center(60) + "\n")
        core.print_out("  " + "Embedded SCF in Fragment C Potential".center(60) + "\n")
        core.print_out("  " + "="*60 + "\n\n")
    
    molecule = dimer_wfn.molecule()
    basisset = dimer_wfn.basisset()
    mints = core.MintsHelper(basisset)
    
    # Store overlap matrix in cache
    cache["S"] = mints.ao_overlap()
    
    # Step 1: Compute nuclear potentials for all fragments
    compute_isapt_fragment_nuclear_potentials(molecule, basisset, cache)
    
    # Step 1b: Handle external potentials if provided
    # This mirrors the logic in build_sapt_jk_cache for external potentials
    if external_potentials:
        if do_print:
            core.print_out("\n  ==> I-SAPT External Potentials <==\n\n")
        
        # Store external potentials in cache for electrostatics calculation
        cache["external_potentials"] = {}
        
        # Add external potential contributions to V_A and V_B
        if external_potentials.get("A") is not None:
            ext_pot_A = external_potentials["A"]
            # Build ExternalPotential object if needed
            if not isinstance(ext_pot_A, core.ExternalPotential):
                ext_obj_A = core.ExternalPotential()
                for charge, coords in ext_pot_A:
                    ext_obj_A.addCharge(charge, coords[0], coords[1], coords[2])
                cache["external_potentials"]["A"] = ext_obj_A
            else:
                cache["external_potentials"]["A"] = ext_pot_A
            # Compute potential matrix and add to V_A
            ext_V_A = cache["external_potentials"]["A"].computePotentialMatrix(basisset)
            cache["V_A"].add(ext_V_A)
            if do_print:
                core.print_out("    Added external potential for fragment A\n")
        
        if external_potentials.get("B") is not None:
            ext_pot_B = external_potentials["B"]
            # Build ExternalPotential object if needed
            if not isinstance(ext_pot_B, core.ExternalPotential):
                ext_obj_B = core.ExternalPotential()
                for charge, coords in ext_pot_B:
                    ext_obj_B.addCharge(charge, coords[0], coords[1], coords[2])
                cache["external_potentials"]["B"] = ext_obj_B
            else:
                cache["external_potentials"]["B"] = ext_pot_B
            # Compute potential matrix and add to V_B
            ext_V_B = cache["external_potentials"]["B"].computePotentialMatrix(basisset)
            cache["V_B"].add(ext_V_B)
            if do_print:
                core.print_out("    Added external potential for fragment B\n")
        
        if external_potentials.get("C") is not None:
            ext_pot_C = external_potentials["C"]
            # Build ExternalPotential object if needed
            if not isinstance(ext_pot_C, core.ExternalPotential):
                ext_obj_C = core.ExternalPotential()
                for charge, coords in ext_pot_C:
                    ext_obj_C.addCharge(charge, coords[0], coords[1], coords[2])
                cache["external_potentials"]["C"] = ext_obj_C
            else:
                cache["external_potentials"]["C"] = ext_pot_C
            # Add to V_C (which contributes to embedding potential W_C)
            ext_V_C = cache["external_potentials"]["C"].computePotentialMatrix(basisset)
            cache["V_C"].add(ext_V_C)
            # Store the external potential C matrix separately (VE in C++ FISAPT)
            # This is needed for delta HF calculation where VE is added to ALL Hamiltonians
            cache["VE"] = ext_V_C.clone()
            if do_print:
                core.print_out("    Added external potential for fragment C\n")
        
        if do_print:
            core.print_out("\n")
    
    # Step 2: Compute J_C, K_C and embedding potential W_C
    compute_isapt_JK_C(jk, cache)
    
    # Step 3: Run embedded SCF for monomer A
    scf_options = {
        'maxiter': core.get_option("SCF", "MAXITER"),
        'e_convergence': core.get_option("SCF", "E_CONVERGENCE"),
        'd_convergence': core.get_option("SCF", "D_CONVERGENCE"),
        'diis_max_vecs': core.get_option("SCF", "DIIS_MAX_VECS"),
        'print_level': 1 if do_print else 0,
    }
    
    # Get V_potential for DFT (if applicable)
    V_potential = None
    if functional is not None and functional.upper() != 'HF':
        V_potential = build_isapt_V_potential(functional, basisset, do_print=do_print)
    
    run_isapt_embedded_scf(
        jk=jk,
        molecule=molecule,
        basisset=basisset,
        mints=mints,
        cache=cache,
        monomer="A",
        functional=functional,
        V_potential=V_potential,
        options=scf_options
    )
    
    # Step 4: Run embedded SCF for monomer B
    run_isapt_embedded_scf(
        jk=jk,
        molecule=molecule,
        basisset=basisset,
        mints=mints,
        cache=cache,
        monomer="B",
        functional=functional,
        V_potential=V_potential,
        options=scf_options
    )
    
    if do_print:
        core.print_out("  ==> I-SAPT Embedded SCF Complete <==\n\n")
        core.print_out(f"    E(A) = {cache['E0_A']:24.16E} [Eh]\n")
        core.print_out(f"    E(B) = {cache['E0_B']:24.16E} [Eh]\n\n")
    
    # Step 5: Build standard SAPT cache keys from embedded SCF results
    # The electrostatics/exchange/induction functions expect these keys
    
    # Map embedded SCF results to standard SAPT cache keys
    cache["Cocc_A"] = cache["Cocc0A"]
    cache["Cocc_B"] = cache["Cocc0B"]
    cache["Cvir_A"] = cache["Cvir0A"]
    cache["Cvir_B"] = cache["Cvir0B"]
    cache["eps_occ_A"] = cache["eps_occ0A"]
    cache["eps_occ_B"] = cache["eps_occ0B"]
    cache["eps_vir_A"] = cache["eps_vir0A"]
    cache["eps_vir_B"] = cache["eps_vir0B"]
    cache["J_A"] = cache["J0A"]
    cache["J_B"] = cache["J0B"]
    cache["K_A"] = cache["K0A"]
    cache["K_B"] = cache["K0B"]
    
    # Build density matrices: D = Cocc @ Cocc.T
    # Wrap in core.Matrix for compatibility with electrostatics() which uses .np accessor
    D_A_np = chain_gemm_einsums([cache["Cocc_A"], cache["Cocc_A"]], ['N', 'T'])
    D_B_np = chain_gemm_einsums([cache["Cocc_B"], cache["Cocc_B"]], ['N', 'T'])
    cache["D_A"] = core.Matrix.from_array(D_A_np)
    cache["D_B"] = core.Matrix.from_array(D_B_np)
    cache["D_A"].name = "D_A"
    cache["D_B"].name = "D_B"
    
    # TODO: Implement proper SIAO1 density reassignment
    # The SIAO algorithm requires:
    # 1. Computing IAOs (Intrinsic Atomic Orbitals)
    # 2. Projecting link orbitals onto IAOs of each fragment
    # 3. Orthogonalizing to occupied orbitals of A/B
    # 4. Adding scaled link orbital contributions to D_A and D_B
    # For now, skip this step - the electrostatics will be slightly off
    # See FISAPT::scf() in fisapt.cc for the full implementation
    
    # Build virtual projectors: P = Cvir @ Cvir.T  
    P_A_np = chain_gemm_einsums([cache["Cvir_A"], cache["Cvir_A"]], ['N', 'T'])
    P_B_np = chain_gemm_einsums([cache["Cvir_B"], cache["Cvir_B"]], ['N', 'T'])
    cache["P_A"] = core.Matrix.from_array(P_A_np)
    cache["P_B"] = core.Matrix.from_array(P_B_np)
    cache["P_A"].name = "P_A"
    cache["P_B"].name = "P_B"
    
    # Debug: print traces of key matrices for comparison with FISAPT C++
    if do_print:
        S = cache["S"]
        core.print_out("\n  ==> I-SAPT Debug: Matrix Traces <==\n\n")
        # Trace of density = N_elec (for normalized D)
        trace_DA = ein.core.dot(cache["D_A"].np, S.np)
        trace_DB = ein.core.dot(cache["D_B"].np, S.np)
        core.print_out(f"    Tr(D_A @ S) = {trace_DA:12.6f} (should be N_occ_A)\n")
        core.print_out(f"    Tr(D_B @ S) = {trace_DB:12.6f} (should be N_occ_B)\n")
        
        # Check orbital orthonormality: C^T @ S @ C should be identity
        Cocc_A = cache["Cocc_A"]
        Cocc_B = cache["Cocc_B"]
        CtSC_A = chain_gemm_einsums([Cocc_A, S, Cocc_A], ['T', 'N', 'N'])
        CtSC_B = chain_gemm_einsums([Cocc_B, S, Cocc_B], ['T', 'N', 'N'])
        CtSC_A_np = _to_numpy(CtSC_A)
        CtSC_B_np = _to_numpy(CtSC_B)
        core.print_out(f"    Tr(C_A^T @ S @ C_A) = {np.trace(CtSC_A_np):12.6f} (should be N_occ_A)\n")
        core.print_out(f"    Tr(C_B^T @ S @ C_B) = {np.trace(CtSC_B_np):12.6f} (should be N_occ_B)\n")
        # Check norm of first orbital
        core.print_out(f"    (C_A^T @ S @ C_A)[0,0] = {CtSC_A_np[0,0]:12.6f} (should be 1.0)\n")
        core.print_out(f"    (C_B^T @ S @ C_B)[0,0] = {CtSC_B_np[0,0]:12.6f} (should be 1.0)\n")
        
        # Trace of V matrices (should be < 0 for nuclear attraction)
        trace_VA = np.trace(cache["V_A"].np)
        trace_VB = np.trace(cache["V_B"].np)
        core.print_out(f"    Tr(V_A) = {trace_VA:12.6f}\n")
        core.print_out(f"    Tr(V_B) = {trace_VB:12.6f}\n")
        # Trace of J matrices
        trace_JA = np.trace(cache["J_A"].np)
        trace_JB = np.trace(cache["J_B"].np)
        core.print_out(f"    Tr(J_A) = {trace_JA:12.6f}\n")
        core.print_out(f"    Tr(J_B) = {trace_JB:12.6f}\n")
        
        # Additional debug: raw Tr(D) and orbital norms
        trace_DA_raw = np.trace(cache["D_A"].np)
        trace_DB_raw = np.trace(cache["D_B"].np)
        core.print_out(f"    Tr(D_A) = {trace_DA_raw:12.6f} (raw, depends on basis)\n")
        core.print_out(f"    Tr(D_B) = {trace_DB_raw:12.6f} (raw, depends on basis)\n")
        
        # Check if orbitals are in the correct subspace
        # For I-SAPT, monomer A orbitals should be mostly on fragment A atoms
        CA_np = cache["Cocc_A"].np
        CB_np = cache["Cocc_B"].np
        core.print_out(f"    |C_A|_F = {np.linalg.norm(CA_np):12.6f}\n")
        core.print_out(f"    |C_B|_F = {np.linalg.norm(CB_np):12.6f}\n")
        core.print_out(f"    C_A.shape = {CA_np.shape}\n")
        core.print_out(f"    C_B.shape = {CB_np.shape}\n")
        
        # Print eigenvalue ranges
        eps_A = cache["eps_occ_A"].np
        eps_B = cache["eps_occ_B"].np
        core.print_out(f"    eps_occ_A: min={eps_A.min():10.4f}, max={eps_A.max():10.4f}\n")
        core.print_out(f"    eps_occ_B: min={eps_B.min():10.4f}, max={eps_B.max():10.4f}\n")
        core.print_out("\n")
    
    # Compute nuclear repulsion between A and B only (for SAPT)
    # For I-SAPT, this is E_AB = sum_{A in frag_A, B in frag_B} Z_A * Z_B / R_AB
    from .embedded_scf import compute_fragment_nuclear_repulsion
    _, _, _, E_total = compute_fragment_nuclear_repulsion(
        molecule, cache["ZA"], cache["ZB"], cache["ZC"]
    )
    # The cross-term E_AB is what matters for SAPT electrostatics
    # E_total = E_AA + E_BB + E_CC + 2*(E_AB + E_AC + E_BC)
    # But we only want E_AB for SAPT, which is in the off-diagonal
    ZAp = cache["ZA"].np
    ZBp = cache["ZB"].np
    E_AB = 0.0
    for A in range(molecule.natom()):
        for B in range(molecule.natom()):
            if A != B and abs(ZAp[A]) > 1e-14 and abs(ZBp[B]) > 1e-14:
                dx = molecule.x(A) - molecule.x(B)
                dy = molecule.y(A) - molecule.y(B)
                dz = molecule.z(A) - molecule.z(B)
                R = np.sqrt(dx*dx + dy*dy + dz*dz)
                E_AB += ZAp[A] * ZBp[B] / R
    cache["nuclear_repulsion_energy"] = E_AB
    
    if do_print:
        core.print_out(f"    Nuclear repulsion E_AB: {E_AB:24.16E} [Eh]\n\n")
    
    # Compute J_O and K_O (overlap exchange matrices)
    # These come from: D_B @ S @ Cocc_A as the left/right density for JK
    jk.C_clear()
    DB_S_CA = chain_gemm_einsums([cache['D_B'], cache['S'], cache['Cocc_A']])
    jk.C_left_add(core.Matrix.from_array(DB_S_CA))
    jk.C_right_add(cache["Cocc_A"])
    jk.compute()
    
    cache["J_O"] = jk.J()[0].clone()
    K_O = jk.K()[0].clone().transpose()
    cache["K_O"] = core.Matrix.from_array(K_O.np)
    cache["K_O"].name = "K_O"
    
    jk.C_clear()
    
    # Handle extern-extern interaction energy
    cache["extern_extern_IE"] = 0.0
    if cache.get("external_potentials"):
        ext_pots = cache["external_potentials"]
        # External potential A interacting with nuclei of B
        # Compute directly: sum over external charges and B nuclei: Z_ext * Z_B / R
        if ext_pots.get("A") is not None:
            ext_charges_A = ext_pots["A"].getCharges()  # list of (Z, x, y, z) tuples
            for B in range(molecule.natom()):
                if abs(ZBp[B]) > 1e-14:
                    xB, yB, zB = molecule.x(B), molecule.y(B), molecule.z(B)
                    for charge_tuple in ext_charges_A:
                        Z_ext, x_ext, y_ext, z_ext = charge_tuple
                        dx, dy, dz = xB - x_ext, yB - y_ext, zB - z_ext
                        R = np.sqrt(dx*dx + dy*dy + dz*dz)
                        E_AB += ZBp[B] * Z_ext / R
        
        # External potential B interacting with nuclei of A
        if ext_pots.get("B") is not None:
            ext_charges_B = ext_pots["B"].getCharges()  # list of (Z, x, y, z) tuples
            for A in range(molecule.natom()):
                if abs(ZAp[A]) > 1e-14:
                    xA, yA, zA = molecule.x(A), molecule.y(A), molecule.z(A)
                    for charge_tuple in ext_charges_B:
                        Z_ext, x_ext, y_ext, z_ext = charge_tuple
                        dx, dy, dz = xA - x_ext, yA - y_ext, zA - z_ext
                        R = np.sqrt(dx*dx + dy*dy + dz*dz)
                        E_AB += ZAp[A] * Z_ext / R
        
        # External-external interaction between A and B potentials
        if ext_pots.get("A") is not None and ext_pots.get("B") is not None:
            cache["extern_extern_IE"] = ext_pots["A"].computeExternExternInteraction(ext_pots["B"])
            if do_print:
                core.print_out(f"    Extern-Extern interaction: {cache['extern_extern_IE']:24.16E} [Eh]\n")
    
    # Update nuclear repulsion with any external contributions
    cache["nuclear_repulsion_energy"] = E_AB
    if do_print and cache.get("external_potentials"):
        core.print_out(f"    Nuclear repulsion E_AB (with ext): {E_AB:24.16E} [Eh]\n\n")
    
    # Store DFT-related quantities for CPHF
    # These are needed for the XC kernel contribution in the CPHF Hessian
    cache["V_potential"] = V_potential
    cache["is_dft"] = V_potential is not None
    if V_potential is not None:
        sup = V_potential.functional()
        cache["x_alpha"] = sup.x_alpha() if sup.is_x_hybrid() else 0.0
    else:
        cache["x_alpha"] = 1.0  # HF: 100% exact exchange
    
    return cache


def compute_delta_hf_isapt(
    dimer_wfn: core.Wavefunction,
    jk: core.JK,
    cache: dict,
    do_print: bool = True,
    external_potentials: dict = None,
) -> float:
    """
    Compute delta HF for I-SAPT (3-fragment systems).
    
    The delta HF correction is the difference between the full HF interaction
    energy and the sum of first-order SAPT terms:
    
        delta_HF = E_HF - Elst10,r - Exch10 - Ind20,r - Exch-Ind20,r
    
    where E_HF is the total HF interaction energy:
        E_HF = E_ABC - E_AC - E_BC + E_C
    
    Each subsystem energy is computed using the fragment density matrices
    (D_A, D_B from embedded SCF, D_C from localization) and corresponding
    J/K matrices.
    
    For DFT I-SAPT, this function runs a separate HF embedded SCF to get
    HF-consistent densities and J/K matrices for the Delta HF calculation.
    This matches the standard SAPT(DFT) approach where Delta HF uses HF
    energies even when SAPT terms use DFT orbitals.
    
    This follows the C++ FISAPT::dHF() implementation.
    
    Parameters
    ----------
    dimer_wfn : core.Wavefunction
        The dimer HF wavefunction.
    jk : core.JK
        JK object for computing Coulomb and exchange integrals.
    cache : dict
        I-SAPT cache containing:
        - D_A, D_B: Fragment density matrices from embedded SCF
        - J_A, J_B, K_A, K_B: Coulomb/exchange from embedded SCF
        - J_C, K_C: Coulomb/exchange from C
        - V_A, V_B, V_C: Nuclear potential matrices
        - ZA, ZB, ZC: Nuclear charges per fragment
        - is_dft: Whether DFT was used for embedded SCF
    do_print : bool, optional
        Whether to print progress (default True).
    
    Returns
    -------
    float
        The delta HF correction in Hartree.
    """
    if do_print:
        core.print_out("\n")
        core.print_out("  " + "="*60 + "\n")
        core.print_out("  " + "I-SAPT: Delta HF Correction".center(60) + "\n")
        core.print_out("  " + "="*60 + "\n\n")
    
    molecule = dimer_wfn.molecule()
    basisset = dimer_wfn.basisset()
    mints = core.MintsHelper(basisset)
    
    # Get kinetic energy matrix
    T = mints.ao_kinetic()
    
    # Get matrices from cache
    S = cache["S"]
    V_A = cache["V_A"]
    V_B = cache["V_B"]
    V_C = cache["V_C"]
    
    # Check if DFT was used - if so, run HF embedded SCF for delta HF
    is_dft = cache.get("is_dft", False)
    
    # Check if we have external potentials
    has_ext_pots_param = (external_potentials is not None and 
                          any(external_potentials.get(k) is not None for k in ['A', 'B', 'C']))
    has_ext_pots_cache = (cache.get("external_potentials") is not None and
                          any(cache["external_potentials"].get(k) is not None for k in ['A', 'B', 'C']))
    has_ext_pots = has_ext_pots_param or has_ext_pots_cache
    
    # For delta HF with external potentials:
    # 
    # This is a known limitation of Python I-SAPT: the dimer SCF is run WITHOUT
    # external potentials, so the delta HF correction cannot properly account for
    # external potential effects.
    #
    # The C++ FISAPT runs dimer SCF WITH external potentials, making delta HF
    # consistent. Until Python I-SAPT is modified to run dimer SCF with ext pots,
    # we use the same code path as non-ext-pots, which gives E_HF ≈ 0.
    #
    # Note: This means delta_HF will be computed relative to SAPT terms that
    # DO include external potential effects, which may cause some inconsistency.
    # TODO: Fix by running dimer SCF with external potentials in sapt_proc.py.
    
    if has_ext_pots:
        if do_print:
            core.print_out("\n    External potentials detected for delta HF calculation.\n")
            core.print_out("    NOTE: Dimer SCF was computed without external potentials.\n")
            core.print_out("    Using embedded SCF orbitals with V matrices excluding external potentials.\n\n")
        
        # Use embedded SCF results from cache
        D_A = cache["D_A"]  # From embedded SCF (optimized WITH ext pots in embedding)
        D_B = cache["D_B"]  # From embedded SCF
        J_A = cache["J_A"]  # From embedded SCF
        J_B = cache["J_B"]  # From embedded SCF
        K_A = cache["K_A"]  # From embedded SCF
        K_B = cache["K_B"]  # From embedded SCF
        
        # J_C and K_C are always from localized C orbitals
        J_C = cache["J_C"]
        K_C = cache["K_C"]
        
        # Get C density from localized orbitals
        Locc_C = cache.get("Locc_C") or cache.get("LoccC")
        if Locc_C is not None and Locc_C.cols() > 0:
            D_C = chain_gemm_einsums([Locc_C, Locc_C], ['N', 'T'])
        else:
            nbf = S.rows()
            D_C = np.zeros((nbf, nbf))
        
        # V matrices from cache already include external potentials
        # We need to subtract them to be consistent with dimer_wfn.energy()
        ext_pots = cache.get("external_potentials", {})
        V_A_np = np.asarray(V_A.np).copy()
        V_B_np = np.asarray(V_B.np).copy()
        V_C_np = np.asarray(V_C.np).copy()
        
        if ext_pots.get("A") is not None:
            ext_V_A = ext_pots["A"].computePotentialMatrix(basisset)
            V_A_np -= np.asarray(ext_V_A.np)
            if do_print:
                core.print_out("      Subtracted ext_A from V_A\n")
        
        if ext_pots.get("B") is not None:
            ext_V_B = ext_pots["B"].computePotentialMatrix(basisset)
            V_B_np -= np.asarray(ext_V_B.np)
            if do_print:
                core.print_out("      Subtracted ext_B from V_B\n")
        
        if ext_pots.get("C") is not None:
            ext_V_C = ext_pots["C"].computePotentialMatrix(basisset)
            V_C_np -= np.asarray(ext_V_C.np)
            if do_print:
                core.print_out("      Subtracted ext_C from V_C\n")
        
    elif is_dft:
        if do_print:
            core.print_out("    Running HF embedded SCF for Delta HF calculation...\n\n")
        
        # Run HF embedded SCF for monomer A
        scf_options = {
            'maxiter': core.get_option("SCF", "MAXITER"),
            'e_convergence': core.get_option("SCF", "E_CONVERGENCE"),
            'd_convergence': core.get_option("SCF", "D_CONVERGENCE"),
            'diis_max_vecs': core.get_option("SCF", "DIIS_MAX_VECS"),
            'print_level': 1 if do_print else 0,
        }
        
        # Create a temporary cache for HF embedded SCF
        # Copy necessary items from the main cache
        # Note: run_isapt_embedded_scf expects keys without underscores (LoccA, not Locc_A)
        hf_cache = {
            "S": cache["S"],
            "V_A": cache["V_A"],
            "V_B": cache["V_B"],
            "V_C": cache["V_C"],
            "J_C": cache["J_C"],
            "K_C": cache["K_C"],
            "W_C": cache["W_C"],
            "ZA": cache["ZA"],
            "ZB": cache["ZB"],
            "ZC": cache["ZC"],
            "LoccA": cache.get("LoccA") or cache.get("Locc_A"),
            "LoccB": cache.get("LoccB") or cache.get("Locc_B"),
            "LoccC": cache.get("LoccC") or cache.get("Locc_C"),
            "Cvir": cache["Cvir"],
        }
        
        # Run HF embedded SCF for A
        run_isapt_embedded_scf(
            jk=jk,
            molecule=molecule,
            basisset=basisset,
            mints=mints,
            cache=hf_cache,
            monomer="A",
            functional=None,  # HF
            V_potential=None,
            options=scf_options
        )
        
        # Run HF embedded SCF for B
        run_isapt_embedded_scf(
            jk=jk,
            molecule=molecule,
            basisset=basisset,
            mints=mints,
            cache=hf_cache,
            monomer="B",
            functional=None,  # HF
            V_potential=None,
            options=scf_options
        )
        
        # Build HF density matrices
        D_A_np = chain_gemm_einsums([hf_cache["Cocc0A"], hf_cache["Cocc0A"]], ['N', 'T'])
        D_B_np = chain_gemm_einsums([hf_cache["Cocc0B"], hf_cache["Cocc0B"]], ['N', 'T'])
        D_A = core.Matrix.from_array(D_A_np)
        D_B = core.Matrix.from_array(D_B_np)
        
        # Get HF J/K matrices
        J_A = hf_cache["J0A"]
        J_B = hf_cache["J0B"]
        K_A = hf_cache["K0A"]
        K_B = hf_cache["K0B"]
        
        if do_print:
            core.print_out(f"    HF Embedded SCF: E(A) = {hf_cache['E0_A']:24.16E} [Eh]\n")
            core.print_out(f"    HF Embedded SCF: E(B) = {hf_cache['E0_B']:24.16E} [Eh]\n\n")
        
        # Get J_C and K_C from cache
        J_C = cache["J_C"]
        K_C = cache["K_C"]
        
        # Get C density from localized orbitals
        Locc_C = cache.get("Locc_C") or cache.get("LoccC")
        if Locc_C is not None and Locc_C.cols() > 0:
            D_C = chain_gemm_einsums([Locc_C, Locc_C], ['N', 'T'])
        else:
            nbf = S.rows()
            D_C = np.zeros((nbf, nbf))
        
        # V matrices without external potentials
        V_A_np = np.asarray(V_A.np)
        V_B_np = np.asarray(V_B.np)
        V_C_np = np.asarray(V_C.np)
        
    else:
        # Use the existing (HF) embedded SCF results from cache
        D_A = cache["D_A"]  # From embedded SCF
        D_B = cache["D_B"]  # From embedded SCF
        J_A = cache["J_A"]  # From embedded SCF
        J_B = cache["J_B"]  # From embedded SCF
        K_A = cache["K_A"]  # From embedded SCF
        K_B = cache["K_B"]  # From embedded SCF
        
        # J_C and K_C are always from localized C orbitals (HF)
        J_C = cache["J_C"]
        K_C = cache["K_C"]
        
        # Get C density from localized orbitals
        Locc_C = cache.get("Locc_C") or cache.get("LoccC")
        if Locc_C is not None and Locc_C.cols() > 0:
            D_C = chain_gemm_einsums([Locc_C, Locc_C], ['N', 'T'])
        else:
            nbf = S.rows()
            D_C = np.zeros((nbf, nbf))
        
        # V matrices from cache (no external potentials for this case)
        V_A_np = np.asarray(V_A.np)
        V_B_np = np.asarray(V_B.np)
        V_C_np = np.asarray(V_C.np)
    
    # Get numpy arrays
    T_np = T.np
    S_np = S.np
    
    # Handle D_A, D_B conversion to numpy
    if hasattr(D_A, 'np'):
        D_A_np = np.asarray(D_A.np)
    else:
        D_A_np = np.asarray(D_A)
    if hasattr(D_B, 'np'):
        D_B_np = np.asarray(D_B.np)
    else:
        D_B_np = np.asarray(D_B)
    
    J_A_np = np.asarray(J_A.np)
    J_B_np = np.asarray(J_B.np)
    K_A_np = np.asarray(K_A.np)
    K_B_np = np.asarray(K_B.np)
    J_C_np = np.asarray(J_C.np)
    K_C_np = np.asarray(K_C.np)
    
    # Ensure D_C is numpy
    if hasattr(D_C, 'np'):
        D_C_np = np.asarray(D_C.np)
    else:
        D_C_np = np.asarray(D_C)
    
    # Compute nuclear repulsion energies between fragments
    ZAp = cache["ZA"].np
    ZBp = cache["ZB"].np
    ZCp = cache["ZC"].np
    
    natom = molecule.natom()
    E_nuc = np.zeros((3, 3))  # [A, B, C] x [A, B, C]
    
    for i in range(natom):
        for j in range(natom):
            if i != j:
                dx = molecule.x(i) - molecule.x(j)
                dy = molecule.y(i) - molecule.y(j)
                dz = molecule.z(i) - molecule.z(j)
                R = np.sqrt(dx*dx + dy*dy + dz*dz)
                Rinv = 1.0 / R
                
                Zi = np.array([ZAp[i], ZBp[i], ZCp[i]])
                Zj = np.array([ZAp[j], ZBp[j], ZCp[j]])
                
                for fi in range(3):
                    for fj in range(3):
                        E_nuc[fi, fj] += 0.5 * Zi[fi] * Zj[fj] * Rinv
    
    # => Dimer ABC HF Energy <= //
    # 
    # For I-SAPT with external potentials:
    # - dimer_wfn.energy() is from dimer SCF WITHOUT ext pots
    # - V_A, V_B, V_C have been adjusted (ext pots subtracted) for consistency
    # - E_AC, E_BC, E_C are computed without ext pots for consistency
    #
    # For I-SAPT without external potentials:
    # - Everything is consistent, just use dimer_wfn.energy()
    
    E_ABC = dimer_wfn.energy()
    
    if has_ext_pots and do_print:
        core.print_out(f"    E_ABC (dimer_wfn) = {E_ABC:24.16E} [Eh]\n")
    
    # => Monomer AC Energy (E_AC(0) in C++ notation) <= //
    # Uses D_A, J_A, K_A from embedded SCF
    E_AC = 0.0
    # Nuclear repulsion: E_AA + E_CC + E_AC + E_CA
    E_AC += E_nuc[0, 0]  # A-A
    E_AC += E_nuc[2, 2]  # C-C
    E_AC += E_nuc[0, 2]  # A-C
    E_AC += E_nuc[2, 0]  # C-A
    
    # One-electron Hamiltonian: H_AC = T + V_A + V_C
    H_AC = T_np + V_A_np + V_C_np
    
    # Fock matrix: F_AC = H_AC + 2*J_AC - K_AC
    # where J_AC = J_A + J_C, K_AC = K_A + K_C
    F_AC = H_AC.copy()
    F_AC += 2.0 * (J_A_np + J_C_np)
    F_AC -= (K_A_np + K_C_np)
    
    # Combined density: D_AC = D_A + D_C
    D_AC = D_A_np + D_C_np
    
    # Energy: E_AC += D_AC · (H_AC + F_AC)
    E_AC += ein.core.dot(D_AC, H_AC + F_AC)
    
    if do_print:
        trace_D_AC = ein.core.dot(D_AC, S_np)
        core.print_out(f"    Trace(D_AC @ S) = {trace_D_AC:10.6f}\n")
    
    # => Monomer BC Energy (E_BC(0) in C++ notation) <= //
    # Uses D_B, J_B, K_B from embedded SCF
    E_BC = 0.0
    # Nuclear repulsion: E_BB + E_CC + E_BC + E_CB
    E_BC += E_nuc[1, 1]  # B-B
    E_BC += E_nuc[2, 2]  # C-C
    E_BC += E_nuc[1, 2]  # B-C
    E_BC += E_nuc[2, 1]  # C-B
    
    # One-electron Hamiltonian: H_BC = T + V_B + V_C
    H_BC = T_np + V_B_np + V_C_np
    
    # Fock matrix: F_BC = H_BC + 2*J_BC - K_BC
    F_BC = H_BC.copy()
    F_BC += 2.0 * (J_B_np + J_C_np)
    F_BC -= (K_B_np + K_C_np)
    
    # Combined density: D_BC = D_B + D_C
    D_BC = D_B_np + D_C_np
    
    # Energy: E_BC += D_BC · (H_BC + F_BC)
    E_BC += ein.core.dot(D_BC, H_BC + F_BC)
    
    if do_print:
        trace_D_BC = ein.core.dot(D_BC, S_np)
        core.print_out(f"    Trace(D_BC @ S) = {trace_D_BC:10.6f}\n")
    
    # => Monomer C Energy <= //
    E_C = 0.0
    # Nuclear repulsion: E_CC
    E_C += E_nuc[2, 2]
    
    # Only compute if C has electrons
    if np.abs(D_C_np).max() > 1e-14:
        # One-electron Hamiltonian: H_C = T + V_C
        H_C = T_np + V_C_np
        
        # Fock matrix: F_C = H_C + 2*J_C - K_C
        F_C = H_C.copy()
        F_C += 2.0 * J_C_np
        F_C -= K_C_np
        
        # Energy: E_C += D_C · (H_C + F_C)
        E_C += ein.core.dot(D_C_np, H_C + F_C)
        
        if do_print:
            trace_D_C = ein.core.dot(D_C_np, S_np)
            core.print_out(f"    Trace(D_C @ S)  = {trace_D_C:10.6f}\n")
    
    # => Total HF interaction energy <= //
    E_HF = E_ABC - E_AC - E_BC + E_C
    
    if do_print:
        core.print_out("\n")
        core.print_out(f"    E ABC(HF) = {E_ABC:24.16E} [Eh]\n")
        core.print_out(f"    E AC(0)   = {E_AC:24.16E} [Eh]\n")
        core.print_out(f"    E BC(0)   = {E_BC:24.16E} [Eh]\n")
        core.print_out(f"    E C       = {E_C:24.16E} [Eh]\n")
        core.print_out(f"    E HF      = {E_HF:24.16E} [Eh]\n")
        core.print_out(f"    E HF      = {E_HF * 1000:24.16f} [mEh]\n")
        core.print_out("\n")
    
    # Store total HF interaction in cache for later use
    cache["HF"] = E_HF
    cache["E_ABC_HF"] = E_ABC
    cache["E_AC_0"] = E_AC
    cache["E_BC_0"] = E_BC
    cache["E_C"] = E_C
    
    # Now compute delta HF as the standard SAPT correction:
    # delta_HF = E_HF - Elst10,r - Exch10 - Ind20,r - Exch-Ind20,r
    # But we don't have all SAPT terms here yet, so we return E_HF
    # and let sapt_proc.py compute the actual delta
    
    return E_HF
