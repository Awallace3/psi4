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

import numpy as np

from psi4 import core

from ...p4util import solvers
from ...p4util.exceptions import *
from .sapt_util import print_sapt_var
from pprint import pprint as pp
import einsums as ein


# Equations come from https://doi.org/10.1063/5.0090688


def localization(cache, dimer_wfn, wfn_A, wfn_B, jk, do_print=True):
    print("\n  ==> Localizing Orbitals <== \n\n")
    # localization_scheme = core.get_option("SAPT", "SAPT_DFT_LOCAL_ORBITALS")
    # loc = core.Localizer.build(localization_scheme, wfn_A.basisset(), wfn_A.Ca_subset("AO", "OCC"))
    # loc.localize()
    # C_lmo_A = loc.L
    # loc = core.Localizer.build(localization_scheme, wfn_B.basisset(), wfn_B.Ca_subset("AO", "OCC"))
    # loc.localize()
    # C_lmo_B = loc.L
    # IBOLocalizer
    N = cache["eps_occ"].dimpi()[0]
    Focc = core.Matrix("Focc", N, N)
    for i in range(N):
        Focc.np[i, i] = cache["eps_occ"].np[i]
    ranges = [0, N, N]
    minao = core.BasisSet.build(dimer_wfn.molecule(), "BASIS", core.get_global_option("MINAO_BASIS"))
    dimer_wfn.set_basisset("MINAO", minao)
    # pybind11 location: ./psi4/src/export_wavefunction.cc
    IBO_loc = core.IBOLocalizer2(
        dimer_wfn.basisset(),
        dimer_wfn.get_basisset("MINAO"),
        dimer_wfn.Ca_subset("AO", "OCC"),
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
    return

def _split_L_U_blocks(cache, tag: str, link_assignment: str):
    """
    Split localized occupied matrices for monomer tag ('A' or 'B'):
      - Locc0X -> Lfocc0X (core) + Laocc0X (valence)
      - Uocc0X -> Ufocc0X (core-core) + Uaocc0X (valence-valence)
      - (optional) LLocc0X augmented with the link orbital column thislinkX

    Assumes matrices_ contains:
      Caocc0X (n_ao x n_act), Cfocc0X (n_ao x n_core), Cocc0X (n_ao x (n_core+n_act))
      eps_occ0X in vectors_
    And that L/U/Q for monomer X have just been written at keys Locc0X/Uocc0X/Qocc0X.
    """
    # Dimensions from coefficient blocks (match your C++ rowspi/colspi usage)
    nn = cache[f"Caocc0{tag}"].np.shape[0]      # rows (AO dimension)
    nf = cache[f"Cfocc0{tag}"].np.shape[1]      # core occ count
    na = cache[f"Caocc0{tag}"].np.shape[1]      # valence/active occ count
    nm = nf + na

    # Grab NumPy views
    L_np = cache[f"Locc0{tag}"].np              # (nn x nm)
    U_np = cache[f"Uocc0{tag}"].np             # (nm x nm)

    # Core/valence splits for L (by columns) and U (by 2x2 block)
    Lf_np = L_np[:, :nf]                            # (nn x nf)
    La_np = L_np[:, nf:nm]                          # (nn x na)
    Uf_np = U_np[:nf, :nf]                          # (nf x nf)
    Ua_np = U_np[nf:nm, nf:nm]                      # (na x na)

    cache[f"Lfocc0{tag}"] = psi4.core.Matrix.from_array(Lf_np)
    cache[f"Laocc0{tag}"] = psi4.core.Matrix.from_array(La_np)
    cache[f"Ufocc0{tag}"] = psi4.core.Matrix.from_array(Uf_np)
    cache[f"Uaocc0{tag}"] = psi4.core.Matrix.from_array(Ua_np)

    cache[f"Locc0{tag}"].set_name(f"Locc0{tag}")
    cache[f"Lfocc0{tag}"].set_name(f"Lfocc0{tag}")
    cache[f"Laocc0{tag}"].set_name(f"Laocc0{tag}")
    cache[f"Uocc0{tag}"].set_name(f"Uocc0{tag}")
    cache[f"Ufocc0{tag}"].set_name(f"Ufocc0{tag}")
    cache[f"Uaocc0{tag}"].set_name(f"Uaocc0{tag}")
    cache[f"Qocc0{tag}"].set_name(f"Qocc0{tag}")

    # Optional: augmented L with the link orbital column
    if link_assignment in {"SAO0","SAO1","SAO2","SIAO0","SIAO1","SIAO2"}:
        Laug = psi4.core.Matrix.zeros(nn, nm + 1)
        Laug_np = Laug.np
        Laug_np[:, :nm] = L_np  # copy full L
        # Append the (normalized-to-1/2) link column thislinkX (shape nn x 1)
        link_col = cache[f"thislink{tag}"].np[:, 0]
        Laug_np[:, nm] = link_col
        cache[f"LLocc0{tag}"] = Laug
        cache[f"LLocc0{tag}"].set_name(f"LLocc0{tag}")


def flocalization(cache, dimer_wfn, wfn_A, wfn_B, jk, do_print=True):
    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    core.print_out("  ==> F-SAPT Localization (IBO) <==\n\n")
    core.print_out("  ==> Local orbitals for Monomer A <==\n\n")
    mol = dimer_wfn.molecule()
    molA = mol.extract_subsets([1], [])
    molB = mol.extract_subsets([2], [])
    nfocc0A = dimer_wfn.basisset().n_frozen_core(core.get_option("GLOBALS","FREEZE_CORE"), molA)
    nfocc0B = dimer_wfn.basisset().n_frozen_core(core.get_option("GLOBALS","FREEZE_CORE"), molB)
    nn = cache["Cocc_A"].shape[0]
    nf = nfocc0A
    na = cache["Cocc_A"].shape[1]
    nm = nf + na
    ranges = [0, nf, nm]
    print(ranges)
    N = cache['eps_occ_A'].shape[0]
    Focc = core.Matrix("Focc", N, N)
    for i in range(N):
        Focc.np[i, i] = cache["eps_occ_A"][i]
    IBO_loc = core.IBOLocalizer2(
        dimer_wfn.basisset(),
        dimer_wfn.get_basisset("MINAO"),
        dimer_wfn.Ca_subset("AO", "OCC"),
    )
    IBO_loc.print_header()
    ret = IBO_loc.localize(
        core.Matrix.from_array(cache['Cocc_A']),
        Focc,
        ranges,
    )
    print(ret)
    raise Exception("Continue Coding!")
    return


def partition(cache, dimer_wfn, wfn_A, wfn_B, jk, do_print=True):
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
    frag_np = cache["FRAG"].np
    frag_np[:] = 0.0
    frag_np[indA] = 1.0
    frag_np[indB] = 2.0
    if indC.size:
        frag_np[indC] = 3.0
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
    ZA_np, ZB_np, ZC_np = ZA.np, ZB.np, ZC.np
    ZA_np[:] = 0.0
    ZB_np[:] = 0.0
    ZC_np[:] = 0.0

    Z_all = np.array([mol.Z(i) for i in range(natoms)], dtype=float)
    ZA_np[indA] = Z_all[indA]
    ZB_np[indB] = Z_all[indB]
    if indC.size:
        ZC_np[indC] = Z_all[indC]

    cache["ZA"] = ZA
    cache["ZB"] = ZB
    cache["ZC"] = ZC
    cache["ZA_orig"] = core.Vector.from_array(ZA_np.copy())
    cache["ZB_orig"] = core.Vector.from_array(ZB_np.copy())
    cache["ZC_orig"] = core.Vector.from_array(ZC_np.copy())

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
                ZA_np[A1] -= 1.0
                ZC_np[A1] += 1.0
                orbsC.append(a)
                orbsL.append(a)
            elif t == "BC":
                ZB_np[A1] -= 1.0
                ZC_np[A1] += 1.0
                orbsC.append(a)
                orbsL.append(a)
            elif t == "AB":
                ZA_np[A1] -= 1.0
                ZC_np[A1] += 1.0
                ZB_np[A2] -= 1.0
                ZC_np[A2] += 1.0
                orbsC.append(a)
                orbsL.append(a)
    elif la == "AB":
        for a, (A1, A2), t in zip(link_orbs, link_atoms, link_types):
            if t == "AC":
                ZA_np[A1] += 1.0
                ZC_np[A1] -= 1.0
                orbsA.append(a)
            elif t == "BC":
                ZB_np[A1] += 1.0
                ZC_np[A1] -= 1.0
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

    ZA2 = i_round(float(ZA_np.sum()))
    ZB2 = i_round(float(ZB_np.sum()))
    ZC2 = i_round(float(ZC_np.sum()))
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

    cache["QF"] = QF
    # --- summary numbers (if you want to print like C++ later) ---
    ZA_int, ZB_int, ZC_int = i_round(ZA_np.sum()), i_round(ZB_np.sum()), i_round(ZC_np.sum())
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

    # Connor said using block tensor, but could create tensorView
    # to assign to memory
    # First grab the orbitals
    cache["Cocc_A"] = ein.core.RuntimeTensorD(wfn_A.Ca_subset("AO", "OCC").np)
    cache['Cocc_A'].set_name("Cocc_A")
    cache["Cvir_A"] = ein.core.RuntimeTensorD(wfn_A.Ca_subset("AO", "VIR").np)
    cache['Cvir_A'].set_name("Cvir_A")

    cache["Cocc_B"] = ein.core.RuntimeTensorD(wfn_B.Ca_subset("AO", "OCC").np)
    cache['Cocc_B'].set_name("Cocc_B")
    cache["Cvir_B"] = ein.core.RuntimeTensorD(wfn_B.Ca_subset("AO", "VIR").np)
    cache['Cvir_B'].set_name("Cvir_B")

    cache["eps_occ_A"] = ein.core.RuntimeTensorD(wfn_A.epsilon_a_subset("AO", "OCC").np)
    cache["eps_vir_A"] = ein.core.RuntimeTensorD(wfn_A.epsilon_a_subset("AO", "VIR").np)
    cache["eps_occ_B"] = ein.core.RuntimeTensorD(wfn_B.epsilon_a_subset("AO", "OCC").np)
    cache["eps_vir_B"] = ein.core.RuntimeTensorD(wfn_B.epsilon_a_subset("AO", "VIR").np)

    cache["eps_occ_A"].set_name("eps_occ_A")
    cache["eps_vir_A"].set_name("eps_vir_A")
    cache["eps_occ_B"].set_name("eps_occ_B")
    cache["eps_vir_B"].set_name("eps_vir_B")

    # localization
    if core.get_option("SAPT", "SAPT_DFT_LOCAL_ORBITALS") != "None":
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
    cache["D_A"] = einsum_chain_gemm([cache['Cocc_A'], cache['Cocc_A']], ['N', 'T'])
    cache['D_B'] = einsum_chain_gemm([cache['Cocc_B'], cache['Cocc_B']], ['N', 'T'])
    # Eq. 7
    cache["P_A"] = einsum_chain_gemm([cache['Cvir_A'], cache['Cvir_A']], ['N', 'T'])
    cache['P_B'] = einsum_chain_gemm([cache['Cvir_B'], cache['Cvir_B']], ['N', 'T'])

    # Potential ints
    mints = core.MintsHelper(wfn_A.basisset())
    cache["V_A"] = ein.core.RuntimeTensorD(mints.ao_potential().np)
    mints = core.MintsHelper(wfn_B.basisset())
    cache["V_B"] = ein.core.RuntimeTensorD(mints.ao_potential().np)

    # External Potentials need to add to V_A and V_B
    # TODO: update this for einsums adding
    if external_potentials:
        if external_potentials.get("A") is not None:
            ext_A = wfn_A.external_pot().computePotentialMatrix(wfn_A.basisset())
            cache["V_A"].add(ext_A)
        if external_potentials.get("B") is not None:
            ext_B = wfn_B.external_pot().computePotentialMatrix(wfn_B.basisset())
            cache["V_B"].add(ext_B)

    # Anything else we might need
    # S corresponds to the overlap matrix, S^{AO}
    cache["S"] = ein.core.RuntimeTensorD(wfn_A.S().clone().np)
    cache["S"].set_name("S")

    # J and K matrices
    jk.C_clear()

    # Normal J/K for Monomer A
    jk.C_left_add(wfn_A.Ca_subset("SO", "OCC"))
    jk.C_right_add(wfn_A.Ca_subset("SO", "OCC"))

    # Normal J/K for Monomer B
    jk.C_left_add(wfn_B.Ca_subset("SO", "OCC"))
    jk.C_right_add(wfn_B.Ca_subset("SO", "OCC"))

    DB_S_CA = core.Matrix.from_array(einsum_chain_gemm([cache['D_B'], cache['S'], cache['Cocc_A']]))
    jk.C_left_add(DB_S_CA)
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_A"]))

    jk.compute()

    # Clone them as the JK object will overwrite.
    cache["J_A"] = ein.core.RuntimeTensorD(jk.J()[0].clone().np)
    cache["K_A"] = ein.core.RuntimeTensorD(jk.K()[0].clone().np)
    cache["J_B"] = ein.core.RuntimeTensorD(jk.J()[1].clone().np)
    cache["K_B"] = ein.core.RuntimeTensorD(jk.K()[1].clone().np)
    cache["J_O"] = ein.core.RuntimeTensorD(jk.J()[2].clone().np)
    cache["K_O"] = ein.core.RuntimeTensorD(jk.K()[2].clone().np.T)

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

    # Eq. 4
    Elst10 = 2.0 * ein.core.dot(cache["D_A"], cache["V_B"])
    Elst10 += 2.0 * ein.core.dot(cache["D_B"], cache["V_A"])
    Elst10 += 4.0 * ein.core.dot(cache["D_B"], cache["J_A"])
    Elst10 += cache["nuclear_repulsion_energy"]

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


def einsum_chain_gemm(
    tensors: list[ein.core.RuntimeTensorD],
    transposes: list[str] = None,
    prefactors_C: list[float] = None,
    prefactors_AB: list[float] = None,
    return_tensors: list[bool] = None,
):
    """
    Computes a chain of einsum matrix multiplications

    Parameters
    ----------
    tensors : list[ein.core.RuntimeTensorD]
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
            T1, T2 = transposes[i], transposes[i + 1]
            A_size = A.shape[0]
            if T1 == "T":
                A_size = A.shape[1]
            B_size = B.shape[1]
            if T2 == "T":
                B_size = B.shape[0]
            C = ein.utils.tensor_factory(f"{A.name} @ {B.name}", [A_size, B_size], np.float64, 'einsums')
            ein.core.gemm(T1, T2, prefactors_AB[i], A, B, prefactors_C[i], C)
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

    # Build potenitals
    h_A = cache["V_A"].copy()
    print("EINSUMS EXCHANGE")
    ein.core.axpy(2.0, cache["J_A"], h_A)
    ein.core.axpy(-1.0, cache["K_A"], h_A)

    h_B = cache["V_B"].copy()
    ein.core.axpy(2.0, cache["J_B"], h_B)
    ein.core.axpy(-1.0, cache["K_B"], h_B)

    w_A = ein.core.RuntimeTensorD(cache["V_A"].copy())
    ein.core.axpy(2.0, cache["J_A"], w_A)

    w_B = ein.core.RuntimeTensorD(cache["V_B"])
    ein.core.axpy(2.0, cache["J_B"], w_B)

    # Build inverse exchange metric
    nocc_A = cache["Cocc_A"].shape[1]
    nocc_B = cache["Cocc_B"].shape[1]
    SAB = einsum_chain_gemm(
        [cache['Cocc_A'], cache['S'], cache['Cocc_B']],
        ['T', 'N', 'N'],
    )

    num_occ = nocc_A + nocc_B

    Sab = core.Matrix(num_occ, num_occ)
    Sab.np[:nocc_A, nocc_A:] = SAB
    Sab.np[nocc_A:, :nocc_A] = SAB.T
    Sab.np[np.diag_indices_from(Sab.np)] += 1
    Sab.power(-1.0, 1.0e-14)
    Sab.np[np.diag_indices_from(Sab.np)] -= 1.0

    Tmo_AA = ein.core.RuntimeTensorD(Sab.np[:nocc_A, :nocc_A])
    Tmo_BB = ein.core.RuntimeTensorD(Sab.np[nocc_A:, nocc_A:])
    Tmo_AB = ein.core.RuntimeTensorD(Sab.np[:nocc_A, nocc_A:])

    T_AA = einsum_chain_gemm([cache['Cocc_A'], Tmo_AA, cache['Cocc_A']], ['N', 'N', 'T'])
    T_BB = einsum_chain_gemm([cache['Cocc_B'], Tmo_BB, cache['Cocc_B']], ['N', 'N', 'T'])
    T_AB = einsum_chain_gemm([cache['Cocc_A'], Tmo_AB, cache['Cocc_B']], ['N', 'N', 'T'])

    S = cache["S"]
    D_A = cache["D_A"]
    P_A = cache["P_A"]
    D_B = cache["D_B"]
    P_B = cache["P_B"]

    # Compute the J and K matrices
    jk.C_clear()

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_A"]))
    jk.C_right_add(core.Matrix.from_array(einsum_chain_gemm([cache['Cocc_A'], Tmo_AA])))

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_B"]))
    jk.C_right_add(core.Matrix.from_array(einsum_chain_gemm([cache['Cocc_A'], Tmo_AB])))

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_A"]))
    jk.C_right_add(core.Matrix.from_array(einsum_chain_gemm([P_B, S, cache['Cocc_A']])))
    # This also works... you can choose to form the density-like matrix either
    # way..., just remember that the C_right_add has an adjoint (transpose, and switch matmul order)
    # jk.C_left_add(core.Matrix.from_array(einsum_chain_gemm([D_A, S, cache['Cvir_B']])))
    # jk.C_right_add(core.Matrix.from_array(cache['Cvir_B']))
    jk.compute()

    JT_A, JT_AB, Jij = jk.J()
    KT_A, KT_AB, Kij = jk.K()
    JT_A = ein.core.RuntimeTensorD(JT_A.np)
    JT_AB = ein.core.RuntimeTensorD(JT_AB.np)
    Jij = ein.core.RuntimeTensorD(Jij.np)
    KT_A = ein.core.RuntimeTensorD(KT_A.np)
    KT_AB = ein.core.RuntimeTensorD(KT_AB.np)
    Kij = ein.core.RuntimeTensorD(Kij.np)

    # Start S^2
    Exch_s2 = 0.0

    # Save some intermediate tensors to avoid recomputation in the next steps
    DA_S_DB_S_PA = einsum_chain_gemm([D_A, S, D_B, S, P_A])
    Exch_s2 -= 2.0 * ein.core.dot(w_B, DA_S_DB_S_PA)

    DB_S_DA_S_PB = einsum_chain_gemm([D_B, S, D_A, S, P_B])
    Exch_s2 -= 2.0 * ein.core.dot(w_A, DB_S_DA_S_PB)
    Exch_s2 -= 2.0 * ein.core.dot(Kij, einsum_chain_gemm([P_A, S, D_B]))

    if do_print:
        core.print_out(print_sapt_var("Exch10(S^2) ", Exch_s2, short=True))
        core.print_out("\n")

    # Start Sinf
    Exch10 = 0.0
    Exch10 -= 2.0 * ein.core.dot(D_A, cache["K_B"])
    Exch10 += 2.0 * ein.core.dot(T_AA, h_B)
    Exch10 += 2.0 * ein.core.dot(T_BB, h_A)
    Exch10 += 2.0 * ein.core.dot(T_AB, h_A + h_B)
    Exch10 += 4.0 * ein.core.dot(T_BB, JT_AB - 0.5 * KT_AB)
    Exch10 += 4.0 * ein.core.dot(T_AA, JT_AB - 0.5 * KT_AB.T)
    Exch10 += 4.0 * ein.core.dot(T_BB, JT_A - 0.5 * KT_A)
    Exch10 += 4.0 * ein.core.dot(T_AB, JT_AB - 0.5 * KT_AB.T)

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
    V_A = ein.core.RuntimeTensorD(cache["V_A"].copy())
    J_A = cache["J_A"]
    K_A = cache["K_A"]

    D_B = cache["D_B"]
    V_B = ein.core.RuntimeTensorD(cache["V_B"].copy())
    J_B = cache["J_B"]
    K_B = cache["K_B"]

    K_O = cache["K_O"]
    J_O = cache["J_O"]

    # Set up matrix multiplication plans
    plan_matmul_tt = ein.core.compile_plan("ij", "ik", "kj")

    # Prepare JK calculations
    jk.C_clear()
    
    DB_S, DB_S_CA = einsum_chain_gemm([D_B, S, cache["Cocc_A"]], return_tensors=[True, True])
    jk.C_left_add(core.Matrix.from_array(DB_S_CA))
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_A"]))

    jk.C_left_add(core.Matrix.from_array(einsum_chain_gemm([DB_S, D_A, S, cache["Cocc_B"]])))
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_B"]))

    DA_S, DA_S_DB_S_CA = einsum_chain_gemm(
                [D_A, S, D_B, S, cache["Cocc_A"]],
                return_tensors=[True, False, False, True],
    )
    jk.C_left_add(core.Matrix.from_array(DA_S_DB_S_CA))
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_A"]))

    jk.compute()

    J_Ot, J_P_B, J_P_A = jk.J()
    K_Ot, K_P_B, K_P_A = jk.K()

    J_Ot = ein.core.RuntimeTensorD(J_Ot.np)
    J_P_B = ein.core.RuntimeTensorD(J_P_B.np)
    J_P_A = ein.core.RuntimeTensorD(J_P_A.np)
    K_Ot = ein.core.RuntimeTensorD(K_Ot.np)
    K_P_B = ein.core.RuntimeTensorD(K_P_B.np)
    K_P_A = ein.core.RuntimeTensorD(K_P_A.np)

    # Exch-Ind Potential A
    EX_A = K_B.copy()
    EX_A *= -1.0
    ein.core.axpy(-2.0, J_O, EX_A)
    ein.core.axpy(1.0, K_O, EX_A)
    ein.core.axpy(2.0, J_P_B, EX_A)

    # Apply all the axpy operations to EX_A
    S_DB, S_DB_VA, S_DB_VA_DB_S = einsum_chain_gemm(
        [S, D_B, V_A, D_B, S],
        return_tensors=[True, True, False, True]
    )
    S_DB_JA, S_DB_JA_DB_S = einsum_chain_gemm(
        [S_DB, J_A, D_B, S],
        return_tensors=[True, False, True]
    )
    S_DB_S_DA, S_DB_S_DA_VB = einsum_chain_gemm(
        [S_DB, S, D_A, V_B],
        return_tensors=[False, True, True],
    )
    ein.core.axpy(-1.0, S_DB_VA, EX_A)
    ein.core.axpy(-2.0, S_DB_JA, EX_A)
    ein.core.axpy(1.0, einsum_chain_gemm([S_DB, K_A]), EX_A)
    ein.core.axpy(1.0, S_DB_S_DA_VB, EX_A)
    ein.core.axpy(2.0, einsum_chain_gemm([S_DB_S_DA, J_B]), EX_A)
    ein.core.axpy(1.0, S_DB_VA_DB_S, EX_A)
    ein.core.axpy(2.0, S_DB_JA_DB_S, EX_A)
    ein.core.axpy(-1.0, einsum_chain_gemm([S_DB, K_O], ["N", "T"]), EX_A)
    ein.core.axpy(-1.0, einsum_chain_gemm([V_B, D_B, S]), EX_A)
    ein.core.axpy(-2.0, einsum_chain_gemm([J_B, D_B, S]), EX_A)
    ein.core.axpy(1.0,  einsum_chain_gemm([K_B, D_B, S]), EX_A)
    ein.core.axpy(1.0,  einsum_chain_gemm([V_B, D_A, S, D_B, S]), EX_A)
    ein.core.axpy(2.0,  einsum_chain_gemm([J_B, D_A, S, D_B, S]), EX_A)
    ein.core.axpy(-1.0, einsum_chain_gemm([K_O, D_B, S]), EX_A)

    EX_A_MO = einsum_chain_gemm(
        [cache['Cocc_A'], EX_A, cache['Cvir_A']],
        ['T', 'N', 'N'],
    )

    # Exch-Ind Potential B
    EX_B = K_A.copy()
    EX_B *= -1.0
    ein.core.axpy(-2.0, J_O, EX_B)
    ein.core.axpy(1.0, K_O.T, EX_B)
    ein.core.axpy(2.0, J_P_A, EX_B)

    S_DA, S_DA_VB, S_DA_VB_DA_S = einsum_chain_gemm(
        [S, D_A, V_B, D_A, S],
        return_tensors=[True, True, False, True]
    )
    S_DA_JB, S_DA_JB_DA_S = einsum_chain_gemm(
        [S_DA, J_B, D_A, S],
        return_tensors=[True, False, True]
    )
    S_DA_S_DB, S_DA_S_DB_VA = einsum_chain_gemm(
        [S_DA, S, D_B, V_A],
        return_tensors=[False, True, True],
    )

    # Bpply all the axpy operations to EX_B
    ein.core.axpy(-1.0, S_DA_VB, EX_B)
    ein.core.axpy(-2.0, S_DA_JB, EX_B)
    ein.core.axpy(1.0, einsum_chain_gemm([S_DA, K_B]), EX_B)
    ein.core.axpy(1.0, S_DA_S_DB_VA, EX_B)
    ein.core.axpy(2.0, einsum_chain_gemm([S_DA_S_DB, J_A]), EX_B)
    ein.core.axpy(1.0, S_DA_VB_DA_S, EX_B)
    ein.core.axpy(2.0, S_DA_JB_DA_S, EX_B)
    ein.core.axpy(-1.0, einsum_chain_gemm([S_DA, K_O]), EX_B)
    ein.core.axpy(-1.0, einsum_chain_gemm([V_A, D_A, S]), EX_B)
    ein.core.axpy(-2.0, einsum_chain_gemm([J_A, D_A, S]), EX_B)
    ein.core.axpy(1.0,  einsum_chain_gemm([K_A, D_A, S]), EX_B)
    ein.core.axpy(1.0,  einsum_chain_gemm([V_A, D_B, S, D_A, S]), EX_B)
    ein.core.axpy(2.0,  einsum_chain_gemm([J_A, D_B, S, D_A, S]), EX_B)
    ein.core.axpy(-1.0, einsum_chain_gemm([K_O, D_A, S], ["T", "N", "N"]), EX_B)

    EX_B_MO = einsum_chain_gemm(
        [cache['Cocc_B'], EX_B, cache['Cvir_B']],
        ['T', 'N', 'N'],
    )

    # Build electrostatic potentials - $\omega_A$ = w_A, Eq. 8
    w_A = V_A.copy()
    w_A.set_name("w_A")
    ein.core.axpy(2.0, J_A, w_A)
    
    w_B = V_B.copy()
    w_B.set_name("w_B")
    ein.core.axpy(2.0, J_B, w_B)

    w_B_MOA = einsum_chain_gemm(
        [cache['Cocc_A'], w_B, cache['Cvir_A']],
        ['T', 'N', 'N'],
    )
    w_A_MOB = einsum_chain_gemm(
        [cache['Cocc_B'], w_A, cache['Cvir_B']],
        ['T', 'N', 'N'],
    )

    # Do uncoupled induction calculations
    core.print_out("   => Uncoupled Induction <= \n\n")
    
    # Create uncoupled response vectors by element-wise division
    unc_x_B_MOA = w_B_MOA.copy()
    unc_x_A_MOB = w_A_MOB.copy()
    
    eps_occ_A = cache["eps_occ_A"]
    eps_vir_A = cache["eps_vir_A"]
    eps_occ_B = cache["eps_occ_B"]
    eps_vir_B = cache["eps_vir_B"]
    
    # Eq. 20
    for r in range(unc_x_B_MOA.shape[0]):
        for a in range(unc_x_B_MOA.shape[1]):
            unc_x_B_MOA[r, a] /= (eps_occ_A[r] - eps_vir_A[a])
    
    # Eq. 20
    for r in range(unc_x_A_MOB.shape[0]):
        for a in range(unc_x_A_MOB.shape[1]):
            unc_x_A_MOB[r, a] /= (eps_occ_B[r] - eps_vir_B[a])

    # Compute uncoupled induction energies according to Eq. 14, 15
    unc_ind_ab = 2.0 * ein.core.dot(unc_x_B_MOA, w_B_MOA)
    unc_ind_ba = 2.0 * ein.core.dot(unc_x_A_MOB, w_A_MOB)
    unc_indexch_ab = 2.0 * ein.core.dot(unc_x_B_MOA, EX_A_MO)
    unc_indexch_ba = 2.0 * ein.core.dot(unc_x_A_MOB, EX_B_MO)

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
        
        # Compute SAB using einsums
        SAB_tmp = ein.utils.tensor_factory("SAB_tmp", [nocc_A, S.shape[1]], np.float64, 'numpy')
        SAB = ein.utils.tensor_factory("SAB", [nocc_A, nocc_B], np.float64, 'numpy')
        plan_matmul_tt.execute(0.0, SAB_tmp, 1.0, cache["Cocc_A"].T, S)
        plan_matmul_tt.execute(0.0, SAB, 1.0, SAB_tmp, cache["Cocc_B"])
        
        num_occ = nocc_A + nocc_B

        # Build Sab matrix (still using psi4 Matrix for matrix operations like power)
        Sab = core.Matrix(num_occ, num_occ)
        Sab.np[:nocc_A, nocc_A:] = SAB
        Sab.np[nocc_A:, :nocc_A] = SAB.T
        Sab.np[np.diag_indices_from(Sab.np)] += 1
        Sab.power(-1.0, 1.0e-14)

        Tmo_AA = ein.core.RuntimeTensorD(Sab.np[:nocc_A, :nocc_A])
        Tmo_BB = ein.core.RuntimeTensorD(Sab.np[nocc_A:, nocc_A:])
        Tmo_AB = ein.core.RuntimeTensorD(Sab.np[:nocc_A, nocc_A:])

        # Compute T matrices using einsums
        T_A_tmp = ein.utils.tensor_factory("T_A_tmp", [cache["Cocc_A"].shape[0], Tmo_AA.shape[1]], np.float64, 'numpy')
        T_A = ein.utils.tensor_factory("T_A", [cache["Cocc_A"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')
        plan_matmul_tt.execute(0.0, T_A_tmp, 1.0, cache["Cocc_A"], Tmo_AA)
        plan_matmul_tt.execute(0.0, T_A, 1.0, T_A_tmp, cache["Cocc_A"].T)

        T_B_tmp = ein.utils.tensor_factory("T_B_tmp", [cache["Cocc_B"].shape[0], Tmo_BB.shape[1]], np.float64, 'numpy')
        T_B = ein.utils.tensor_factory("T_B", [cache["Cocc_B"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
        plan_matmul_tt.execute(0.0, T_B_tmp, 1.0, cache["Cocc_B"], Tmo_BB)
        plan_matmul_tt.execute(0.0, T_B, 1.0, T_B_tmp, cache["Cocc_B"].T)

        T_AB_tmp = ein.utils.tensor_factory("T_AB_tmp", [cache["Cocc_A"].shape[0], Tmo_AB.shape[1]], np.float64, 'numpy')
        T_AB = ein.utils.tensor_factory("T_AB", [cache["Cocc_A"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
        plan_matmul_tt.execute(0.0, T_AB_tmp, 1.0, cache["Cocc_A"], Tmo_AB)
        plan_matmul_tt.execute(0.0, T_AB, 1.0, T_AB_tmp, cache["Cocc_B"].T)

        # Compute sT matrices using einsums
        sT_A_tmp1 = ein.utils.tensor_factory("sT_A_tmp1", [cache["Cvir_A"].shape[1], unc_x_B_MOA.shape[0]], np.float64, 'numpy')
        sT_A_tmp2 = ein.utils.tensor_factory("sT_A_tmp2", [cache["Cvir_A"].shape[1], Tmo_AA.shape[1]], np.float64, 'numpy')
        sT_A = ein.utils.tensor_factory("sT_A", [cache["Cvir_A"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')
        plan_matmul_tt.execute(0.0, sT_A_tmp1, 1.0, cache["Cvir_A"].T, unc_x_B_MOA.T)
        plan_matmul_tt.execute(0.0, sT_A_tmp2, 1.0, sT_A_tmp1, Tmo_AA)
        plan_matmul_tt.execute(0.0, sT_A, 1.0, sT_A_tmp2.T, cache["Cocc_A"].T)

        sT_B_tmp1 = ein.utils.tensor_factory("sT_B_tmp1", [cache["Cvir_B"].shape[1], unc_x_A_MOB.shape[0]], np.float64, 'numpy')
        sT_B_tmp2 = ein.utils.tensor_factory("sT_B_tmp2", [cache["Cvir_B"].shape[1], Tmo_BB.shape[1]], np.float64, 'numpy')
        sT_B = ein.utils.tensor_factory("sT_B", [cache["Cvir_B"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
        plan_matmul_tt.execute(0.0, sT_B_tmp1, 1.0, cache["Cvir_B"].T, unc_x_A_MOB.T)
        plan_matmul_tt.execute(0.0, sT_B_tmp2, 1.0, sT_B_tmp1, Tmo_BB)
        plan_matmul_tt.execute(0.0, sT_B, 1.0, sT_B_tmp2.T, cache["Cocc_B"].T)

        sT_AB_tmp1 = ein.utils.tensor_factory("sT_AB_tmp1", [cache["Cvir_A"].shape[1], unc_x_B_MOA.shape[0]], np.float64, 'numpy')
        sT_AB_tmp2 = ein.utils.tensor_factory("sT_AB_tmp2", [cache["Cvir_A"].shape[1], Tmo_AB.shape[1]], np.float64, 'numpy')
        sT_AB = ein.utils.tensor_factory("sT_AB", [cache["Cvir_A"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
        plan_matmul_tt.execute(0.0, sT_AB_tmp1, 1.0, cache["Cvir_A"].T, unc_x_B_MOA.T)
        plan_matmul_tt.execute(0.0, sT_AB_tmp2, 1.0, sT_AB_tmp1, Tmo_AB)
        plan_matmul_tt.execute(0.0, sT_AB, 1.0, sT_AB_tmp2.T, cache["Cocc_B"].T)

        sT_BA_tmp1 = ein.utils.tensor_factory("sT_BA_tmp1", [cache["Cvir_B"].shape[1], unc_x_A_MOB.shape[0]], np.float64, 'numpy')
        sT_BA_tmp2 = ein.utils.tensor_factory("sT_BA_tmp2", [cache["Cvir_B"].shape[1], Tmo_AB.shape[0]], np.float64, 'numpy')
        sT_BA = ein.utils.tensor_factory("sT_BA", [cache["Cvir_B"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')
        plan_matmul_tt.execute(0.0, sT_BA_tmp1, 1.0, cache["Cvir_B"].T, unc_x_A_MOB.T)
        plan_matmul_tt.execute(0.0, sT_BA_tmp2, 1.0, sT_BA_tmp1, Tmo_AB.T)
        plan_matmul_tt.execute(0.0, sT_BA, 1.0, sT_BA_tmp2.T, cache["Cocc_A"].T)

        # Compute JK matrices for Sinf
        jk.C_clear()
        
        CA_Tmo_AA = ein.utils.tensor_factory("CA_Tmo_AA", [cache["Cocc_A"].shape[0], Tmo_AA.shape[1]], np.float64, 'numpy')
        CB_Tmo_BB = ein.utils.tensor_factory("CB_Tmo_BB", [cache["Cocc_B"].shape[0], Tmo_BB.shape[1]], np.float64, 'numpy')
        CA_Tmo_AB = ein.utils.tensor_factory("CA_Tmo_AB", [cache["Cocc_A"].shape[0], Tmo_AB.shape[1]], np.float64, 'numpy')
        
        plan_matmul_tt.execute(0.0, CA_Tmo_AA, 1.0, cache["Cocc_A"], Tmo_AA)
        plan_matmul_tt.execute(0.0, CB_Tmo_BB, 1.0, cache["Cocc_B"], Tmo_BB)
        plan_matmul_tt.execute(0.0, CA_Tmo_AB, 1.0, cache["Cocc_A"], Tmo_AB)

        jk.C_left_add(core.Matrix.from_array(CA_Tmo_AA))
        jk.C_right_add(core.Matrix.from_array(cache["Cocc_A"]))

        jk.C_left_add(core.Matrix.from_array(CB_Tmo_BB))
        jk.C_right_add(core.Matrix.from_array(cache["Cocc_B"]))

        jk.C_left_add(core.Matrix.from_array(CA_Tmo_AB))
        jk.C_right_add(core.Matrix.from_array(cache["Cocc_B"]))

        jk.compute()

        J_AA_inf, J_BB_inf, J_AB_inf = jk.J()
        K_AA_inf, K_BB_inf, K_AB_inf = jk.K()

        J_AA_inf = ein.core.RuntimeTensorD(J_AA_inf.np)
        J_BB_inf = ein.core.RuntimeTensorD(J_BB_inf.np)
        J_AB_inf = ein.core.RuntimeTensorD(J_AB_inf.np)
        K_AA_inf = ein.core.RuntimeTensorD(K_AA_inf.np)
        K_BB_inf = ein.core.RuntimeTensorD(K_BB_inf.np)
        K_AB_inf = ein.core.RuntimeTensorD(K_AB_inf.np)

        # Build EX_AA_inf (A <- B)
        EX_AA_inf = V_B.copy()
        
        # Compute all intermediate tensors for EX_AA_inf
        S_TAB_T_VB = ein.utils.tensor_factory("S_TAB_T_VB", [S.shape[0], V_B.shape[1]], np.float64, 'numpy')
        S_TB_VB = ein.utils.tensor_factory("S_TB_VB", [S.shape[0], V_B.shape[1]], np.float64, 'numpy')
        S_TAB_T_JAB_inf = ein.utils.tensor_factory("S_TAB_T_JAB_inf", [S.shape[0], J_AB_inf.shape[1]], np.float64, 'numpy')
        S_TB_JAB_inf = ein.utils.tensor_factory("S_TB_JAB_inf", [S.shape[0], J_AB_inf.shape[1]], np.float64, 'numpy')
        S_TAB_T_JBB_inf = ein.utils.tensor_factory("S_TAB_T_JBB_inf", [S.shape[0], J_BB_inf.shape[1]], np.float64, 'numpy')
        S_TB_JBB_inf = ein.utils.tensor_factory("S_TB_JBB_inf", [S.shape[0], J_BB_inf.shape[1]], np.float64, 'numpy')
        S_TAB_T_KAB_inf_T = ein.utils.tensor_factory("S_TAB_T_KAB_inf_T", [S.shape[0], K_AB_inf.shape[0]], np.float64, 'numpy')
        S_TB_KAB_inf_T = ein.utils.tensor_factory("S_TB_KAB_inf_T", [S.shape[0], K_AB_inf.shape[0]], np.float64, 'numpy')
        S_TAB_T_KBB_inf = ein.utils.tensor_factory("S_TAB_T_KBB_inf", [S.shape[0], K_BB_inf.shape[1]], np.float64, 'numpy')
        S_TB_KBB_inf = ein.utils.tensor_factory("S_TB_KBB_inf", [S.shape[0], K_BB_inf.shape[1]], np.float64, 'numpy')

        plan_matmul_tt.execute(0.0, S_TAB_T_VB, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_VB, 1.0, S_TAB_T_VB, V_B)
        plan_matmul_tt.execute(0.0, S_TB_VB, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_VB, 1.0, S_TB_VB, V_B)
        
        plan_matmul_tt.execute(0.0, S_TAB_T_JAB_inf, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_JAB_inf, 1.0, S_TAB_T_JAB_inf, J_AB_inf)
        plan_matmul_tt.execute(0.0, S_TB_JAB_inf, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_JAB_inf, 1.0, S_TB_JAB_inf, J_AB_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_T_JBB_inf, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_JBB_inf, 1.0, S_TAB_T_JBB_inf, J_BB_inf)
        plan_matmul_tt.execute(0.0, S_TB_JBB_inf, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_JBB_inf, 1.0, S_TB_JBB_inf, J_BB_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_T_KAB_inf_T, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_KAB_inf_T, 1.0, S_TAB_T_KAB_inf_T, K_AB_inf.T)
        plan_matmul_tt.execute(0.0, S_TB_KAB_inf_T, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_KAB_inf_T, 1.0, S_TB_KAB_inf_T, K_AB_inf.T)
        
        plan_matmul_tt.execute(0.0, S_TAB_T_KBB_inf, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_KBB_inf, 1.0, S_TAB_T_KBB_inf, K_BB_inf)
        plan_matmul_tt.execute(0.0, S_TB_KBB_inf, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_KBB_inf, 1.0, S_TB_KBB_inf, K_BB_inf)

        # Apply operations to EX_AA_inf
        ein.core.axpy(-1.0, S_TAB_T_VB, EX_AA_inf)
        ein.core.axpy(-1.0, S_TB_VB, EX_AA_inf)
        ein.core.axpy(2.0, J_AB_inf, EX_AA_inf)
        ein.core.axpy(-2.0, S_TAB_T_JAB_inf, EX_AA_inf)
        ein.core.axpy(-2.0, S_TB_JAB_inf, EX_AA_inf)
        ein.core.axpy(2.0, J_BB_inf, EX_AA_inf)
        ein.core.axpy(-2.0, S_TAB_T_JBB_inf, EX_AA_inf)
        ein.core.axpy(-2.0, S_TB_JBB_inf, EX_AA_inf)
        ein.core.axpy(-1.0, K_AB_inf.T, EX_AA_inf)
        ein.core.axpy(1.0, S_TAB_T_KAB_inf_T, EX_AA_inf)
        ein.core.axpy(1.0, S_TB_KAB_inf_T, EX_AA_inf)
        ein.core.axpy(-1.0, K_BB_inf, EX_AA_inf)
        ein.core.axpy(1.0, S_TAB_T_KBB_inf, EX_AA_inf)
        ein.core.axpy(1.0, S_TB_KBB_inf, EX_AA_inf)

        # Build EX_AB_inf
        EX_AB_inf = V_A.copy()
        
        # Compute all intermediate tensors for EX_AB_inf
        S_TAB_T_VA = ein.utils.tensor_factory("S_TAB_T_VA", [S.shape[0], V_A.shape[1]], np.float64, 'numpy')
        S_TB_VA = ein.utils.tensor_factory("S_TB_VA", [S.shape[0], V_A.shape[1]], np.float64, 'numpy')
        S_TAB_T_JAA_inf = ein.utils.tensor_factory("S_TAB_T_JAA_inf", [S.shape[0], J_AA_inf.shape[1]], np.float64, 'numpy')
        S_TB_JAA_inf = ein.utils.tensor_factory("S_TB_JAA_inf", [S.shape[0], J_AA_inf.shape[1]], np.float64, 'numpy')
        S_TAB_T_JAB_inf_2 = ein.utils.tensor_factory("S_TAB_T_JAB_inf_2", [S.shape[0], J_AB_inf.shape[1]], np.float64, 'numpy')
        S_TB_JAB_inf_2 = ein.utils.tensor_factory("S_TB_JAB_inf_2", [S.shape[0], J_AB_inf.shape[1]], np.float64, 'numpy')
        S_TAB_T_KAA_inf = ein.utils.tensor_factory("S_TAB_T_KAA_inf", [S.shape[0], K_AA_inf.shape[1]], np.float64, 'numpy')
        S_TB_KAA_inf = ein.utils.tensor_factory("S_TB_KAA_inf", [S.shape[0], K_AA_inf.shape[1]], np.float64, 'numpy')
        S_TAB_T_KAB_inf_2 = ein.utils.tensor_factory("S_TAB_T_KAB_inf_2", [S.shape[0], K_AB_inf.shape[1]], np.float64, 'numpy')
        S_TB_KAB_inf_2 = ein.utils.tensor_factory("S_TB_KAB_inf_2", [S.shape[0], K_AB_inf.shape[1]], np.float64, 'numpy')

        plan_matmul_tt.execute(0.0, S_TAB_T_VA, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_VA, 1.0, S_TAB_T_VA, V_A)
        plan_matmul_tt.execute(0.0, S_TB_VA, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_VA, 1.0, S_TB_VA, V_A)
        
        plan_matmul_tt.execute(0.0, S_TAB_T_JAA_inf, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_JAA_inf, 1.0, S_TAB_T_JAA_inf, J_AA_inf)
        plan_matmul_tt.execute(0.0, S_TB_JAA_inf, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_JAA_inf, 1.0, S_TB_JAA_inf, J_AA_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_T_JAB_inf_2, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_JAB_inf_2, 1.0, S_TAB_T_JAB_inf_2, J_AB_inf)
        plan_matmul_tt.execute(0.0, S_TB_JAB_inf_2, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_JAB_inf_2, 1.0, S_TB_JAB_inf_2, J_AB_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_T_KAA_inf, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_KAA_inf, 1.0, S_TAB_T_KAA_inf, K_AA_inf)
        plan_matmul_tt.execute(0.0, S_TB_KAA_inf, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_KAA_inf, 1.0, S_TB_KAA_inf, K_AA_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_T_KAB_inf_2, 1.0, S, T_AB.T)
        plan_matmul_tt.execute(0.0, S_TAB_T_KAB_inf_2, 1.0, S_TAB_T_KAB_inf_2, K_AB_inf)
        plan_matmul_tt.execute(0.0, S_TB_KAB_inf_2, 1.0, S, T_B)
        plan_matmul_tt.execute(0.0, S_TB_KAB_inf_2, 1.0, S_TB_KAB_inf_2, K_AB_inf)

        # Apply operations to EX_AB_inf
        ein.core.axpy(-1.0, S_TAB_T_VA, EX_AB_inf)
        ein.core.axpy(-1.0, S_TB_VA, EX_AB_inf)
        ein.core.axpy(2.0, J_AA_inf, EX_AB_inf)
        ein.core.axpy(-2.0, S_TAB_T_JAA_inf, EX_AB_inf)
        ein.core.axpy(-2.0, S_TB_JAA_inf, EX_AB_inf)
        ein.core.axpy(2.0, J_AB_inf, EX_AB_inf)
        ein.core.axpy(-2.0, S_TAB_T_JAB_inf_2, EX_AB_inf)
        ein.core.axpy(-2.0, S_TB_JAB_inf_2, EX_AB_inf)
        ein.core.axpy(-1.0, K_AA_inf, EX_AB_inf)
        ein.core.axpy(1.0, S_TAB_T_KAA_inf, EX_AB_inf)
        ein.core.axpy(1.0, S_TB_KAA_inf, EX_AB_inf)
        ein.core.axpy(-1.0, K_AB_inf, EX_AB_inf)
        ein.core.axpy(1.0, S_TAB_T_KAB_inf_2, EX_AB_inf)
        ein.core.axpy(1.0, S_TB_KAB_inf_2, EX_AB_inf)

        # Build EX_BB_inf (B <- A)
        EX_BB_inf = V_A.copy()
        
        # Compute all intermediate tensors for EX_BB_inf
        S_TAB_VA = ein.utils.tensor_factory("S_TAB_VA", [S.shape[0], V_A.shape[1]], np.float64, 'numpy')
        S_TA_VA = ein.utils.tensor_factory("S_TA_VA", [S.shape[0], V_A.shape[1]], np.float64, 'numpy')
        S_TAB_JAB_inf = ein.utils.tensor_factory("S_TAB_JAB_inf", [S.shape[0], J_AB_inf.shape[1]], np.float64, 'numpy')
        S_TA_JAB_inf = ein.utils.tensor_factory("S_TA_JAB_inf", [S.shape[0], J_AB_inf.shape[1]], np.float64, 'numpy')
        S_TAB_JAA_inf = ein.utils.tensor_factory("S_TAB_JAA_inf", [S.shape[0], J_AA_inf.shape[1]], np.float64, 'numpy')
        S_TA_JAA_inf = ein.utils.tensor_factory("S_TA_JAA_inf", [S.shape[0], J_AA_inf.shape[1]], np.float64, 'numpy')
        S_TAB_KAB_inf = ein.utils.tensor_factory("S_TAB_KAB_inf", [S.shape[0], K_AB_inf.shape[1]], np.float64, 'numpy')
        S_TA_KAB_inf = ein.utils.tensor_factory("S_TA_KAB_inf", [S.shape[0], K_AB_inf.shape[1]], np.float64, 'numpy')
        S_TAB_KAA_inf = ein.utils.tensor_factory("S_TAB_KAA_inf", [S.shape[0], K_AA_inf.shape[1]], np.float64, 'numpy')
        S_TA_KAA_inf = ein.utils.tensor_factory("S_TA_KAA_inf", [S.shape[0], K_AA_inf.shape[1]], np.float64, 'numpy')

        plan_matmul_tt.execute(0.0, S_TAB_VA, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_VA, 1.0, S_TAB_VA, V_A)
        plan_matmul_tt.execute(0.0, S_TA_VA, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_VA, 1.0, S_TA_VA, V_A)
        
        plan_matmul_tt.execute(0.0, S_TAB_JAB_inf, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_JAB_inf, 1.0, S_TAB_JAB_inf, J_AB_inf)
        plan_matmul_tt.execute(0.0, S_TA_JAB_inf, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_JAB_inf, 1.0, S_TA_JAB_inf, J_AB_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_JAA_inf, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_JAA_inf, 1.0, S_TAB_JAA_inf, J_AA_inf)
        plan_matmul_tt.execute(0.0, S_TA_JAA_inf, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_JAA_inf, 1.0, S_TA_JAA_inf, J_AA_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_KAB_inf, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_KAB_inf, 1.0, S_TAB_KAB_inf, K_AB_inf)
        plan_matmul_tt.execute(0.0, S_TA_KAB_inf, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_KAB_inf, 1.0, S_TA_KAB_inf, K_AB_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_KAA_inf, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_KAA_inf, 1.0, S_TAB_KAA_inf, K_AA_inf)
        plan_matmul_tt.execute(0.0, S_TA_KAA_inf, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_KAA_inf, 1.0, S_TA_KAA_inf, K_AA_inf)

        # Apply operations to EX_BB_inf
        ein.core.axpy(-1.0, S_TAB_VA, EX_BB_inf)
        ein.core.axpy(-1.0, S_TA_VA, EX_BB_inf)
        ein.core.axpy(2.0, J_AB_inf, EX_BB_inf)
        ein.core.axpy(-2.0, S_TAB_JAB_inf, EX_BB_inf)
        ein.core.axpy(-2.0, S_TA_JAB_inf, EX_BB_inf)
        ein.core.axpy(2.0, J_AA_inf, EX_BB_inf)
        ein.core.axpy(-2.0, S_TAB_JAA_inf, EX_BB_inf)
        ein.core.axpy(-2.0, S_TA_JAA_inf, EX_BB_inf)
        ein.core.axpy(-1.0, K_AB_inf, EX_BB_inf)
        ein.core.axpy(1.0, S_TAB_KAB_inf, EX_BB_inf)
        ein.core.axpy(1.0, S_TA_KAB_inf, EX_BB_inf)
        ein.core.axpy(-1.0, K_AA_inf, EX_BB_inf)
        ein.core.axpy(1.0, S_TAB_KAA_inf, EX_BB_inf)
        ein.core.axpy(1.0, S_TA_KAA_inf, EX_BB_inf)

        # Build EX_BA_inf
        EX_BA_inf = V_B.copy()
        
        # Compute all intermediate tensors for EX_BA_inf
        S_TAB_VB = ein.utils.tensor_factory("S_TAB_VB", [S.shape[0], V_B.shape[1]], np.float64, 'numpy')
        S_TA_VB = ein.utils.tensor_factory("S_TA_VB", [S.shape[0], V_B.shape[1]], np.float64, 'numpy')
        S_TAB_JBB_inf_2 = ein.utils.tensor_factory("S_TAB_JBB_inf_2", [S.shape[0], J_BB_inf.shape[1]], np.float64, 'numpy')
        S_TA_JBB_inf_2 = ein.utils.tensor_factory("S_TA_JBB_inf_2", [S.shape[0], J_BB_inf.shape[1]], np.float64, 'numpy')
        S_TAB_JAB_inf_3 = ein.utils.tensor_factory("S_TAB_JAB_inf_3", [S.shape[0], J_AB_inf.shape[1]], np.float64, 'numpy')
        S_TA_JAB_inf_3 = ein.utils.tensor_factory("S_TA_JAB_inf_3", [S.shape[0], J_AB_inf.shape[1]], np.float64, 'numpy')
        S_TAB_KBB_inf_2 = ein.utils.tensor_factory("S_TAB_KBB_inf_2", [S.shape[0], K_BB_inf.shape[1]], np.float64, 'numpy')
        S_TA_KBB_inf_2 = ein.utils.tensor_factory("S_TA_KBB_inf_2", [S.shape[0], K_BB_inf.shape[1]], np.float64, 'numpy')
        S_TAB_KAB_inf_T_2 = ein.utils.tensor_factory("S_TAB_KAB_inf_T_2", [S.shape[0], K_AB_inf.shape[0]], np.float64, 'numpy')
        S_TA_KAB_inf_T_2 = ein.utils.tensor_factory("S_TA_KAB_inf_T_2", [S.shape[0], K_AB_inf.shape[0]], np.float64, 'numpy')

        plan_matmul_tt.execute(0.0, S_TAB_VB, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_VB, 1.0, S_TAB_VB, V_B)
        plan_matmul_tt.execute(0.0, S_TA_VB, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_VB, 1.0, S_TA_VB, V_B)
        
        plan_matmul_tt.execute(0.0, S_TAB_JBB_inf_2, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_JBB_inf_2, 1.0, S_TAB_JBB_inf_2, J_BB_inf)
        plan_matmul_tt.execute(0.0, S_TA_JBB_inf_2, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_JBB_inf_2, 1.0, S_TA_JBB_inf_2, J_BB_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_JAB_inf_3, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_JAB_inf_3, 1.0, S_TAB_JAB_inf_3, J_AB_inf)
        plan_matmul_tt.execute(0.0, S_TA_JAB_inf_3, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_JAB_inf_3, 1.0, S_TA_JAB_inf_3, J_AB_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_KBB_inf_2, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_KBB_inf_2, 1.0, S_TAB_KBB_inf_2, K_BB_inf)
        plan_matmul_tt.execute(0.0, S_TA_KBB_inf_2, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_KBB_inf_2, 1.0, S_TA_KBB_inf_2, K_BB_inf)
        
        plan_matmul_tt.execute(0.0, S_TAB_KAB_inf_T_2, 1.0, S, T_AB)
        plan_matmul_tt.execute(0.0, S_TAB_KAB_inf_T_2, 1.0, S_TAB_KAB_inf_T_2, K_AB_inf.T)
        plan_matmul_tt.execute(0.0, S_TA_KAB_inf_T_2, 1.0, S, T_A)
        plan_matmul_tt.execute(0.0, S_TA_KAB_inf_T_2, 1.0, S_TA_KAB_inf_T_2, K_AB_inf.T)

        # Apply operations to EX_BA_inf
        ein.core.axpy(-1.0, S_TAB_VB, EX_BA_inf)
        ein.core.axpy(-1.0, S_TA_VB, EX_BA_inf)
        ein.core.axpy(2.0, J_BB_inf, EX_BA_inf)
        ein.core.axpy(-2.0, S_TAB_JBB_inf_2, EX_BA_inf)
        ein.core.axpy(-2.0, S_TA_JBB_inf_2, EX_BA_inf)
        ein.core.axpy(2.0, J_AB_inf, EX_BA_inf)
        ein.core.axpy(-2.0, S_TAB_JAB_inf_3, EX_BA_inf)
        ein.core.axpy(-2.0, S_TA_JAB_inf_3, EX_BA_inf)
        ein.core.axpy(-1.0, K_BB_inf, EX_BA_inf)
        ein.core.axpy(1.0, S_TAB_KBB_inf_2, EX_BA_inf)
        ein.core.axpy(1.0, S_TA_KBB_inf_2, EX_BA_inf)
        ein.core.axpy(-1.0, K_AB_inf.T, EX_BA_inf)
        ein.core.axpy(1.0, S_TAB_KAB_inf_T_2, EX_BA_inf)
        ein.core.axpy(1.0, S_TA_KAB_inf_T_2, EX_BA_inf)

        # Compute uncoupled Sinf energies using vector dot products
        unc_ind_ab_total_tensor_1 = ein.core.dot(sT_A, EX_AA_inf)
        unc_ind_ab_total_tensor_2 = ein.core.dot(sT_AB, EX_AB_inf)
        unc_ind_ba_total_tensor_1 = ein.core.dot(sT_B, EX_BB_inf)
        unc_ind_ba_total_tensor_2 = ein.core.dot(sT_BA, EX_BA_inf)

        unc_ind_ab_total = 2.0 * (unc_ind_ab_total_tensor_1[0] + unc_ind_ab_total_tensor_2[0])
        unc_ind_ba_total = 2.0 * (unc_ind_ba_total_tensor_1[0] + unc_ind_ba_total_tensor_2[0])
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
            cache, jk, w_B_MOA, w_A_MOB, 20, cphf_r_convergence, sapt_jk_B=sapt_jk_B
        )

        ind_ab = 2.0 * ein.core.dot(x_B_MOA, w_B_MOA)
        ind_ba = 2.0 * ein.core.dot(x_A_MOB, w_A_MOB)
        indexch_ab = 2.0 * ein.core.dot(x_B_MOA, EX_A_MO)
        indexch_ba = 2.0 * ein.core.dot(x_A_MOB, EX_B_MO)

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

        # Coupled Exch-Ind without S^2 (if Sinf)
        if Sinf:
            # TODO: need a test for Sinf... highly certain Einsums are wrong here...
            # Compute cT matrices using coupled amplitudes
            cT_A_tmp1 = ein.utils.tensor_factory("cT_A_tmp1", [cache["Cvir_A"].shape[1], x_B_MOA.shape[0]], np.float64, 'numpy')
            cT_A_tmp2 = ein.utils.tensor_factory("cT_A_tmp2", [cache["Cvir_A"].shape[1], Tmo_AA.shape[1]], np.float64, 'numpy')
            cT_A = ein.utils.tensor_factory("cT_A", [cache["Cvir_A"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')
            plan_matmul_tt.execute(0.0, cT_A_tmp1, 1.0, cache["Cvir_A"].T, x_B_MOA.T)
            plan_matmul_tt.execute(0.0, cT_A_tmp2, 1.0, cT_A_tmp1, Tmo_AA)
            plan_matmul_tt.execute(0.0, cT_A, 1.0, cT_A_tmp2.T, cache["Cocc_A"].T)

            cT_B_tmp1 = ein.utils.tensor_factory("cT_B_tmp1", [cache["Cvir_B"].shape[1], x_A_MOB.shape[0]], np.float64, 'numpy')
            cT_B_tmp2 = ein.utils.tensor_factory("cT_B_tmp2", [cache["Cvir_B"].shape[1], Tmo_BB.shape[1]], np.float64, 'numpy')
            cT_B = ein.utils.tensor_factory("cT_B", [cache["Cvir_B"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
            plan_matmul_tt.execute(0.0, cT_B_tmp1, 1.0, cache["Cvir_B"].T, x_A_MOB.T)
            plan_matmul_tt.execute(0.0, cT_B_tmp2, 1.0, cT_B_tmp1, Tmo_BB)
            plan_matmul_tt.execute(0.0, cT_B, 1.0, cT_B_tmp2.T, cache["Cocc_B"].T)

            cT_AB_tmp1 = ein.utils.tensor_factory("cT_AB_tmp1", [cache["Cvir_A"].shape[1], x_B_MOA.shape[0]], np.float64, 'numpy')
            cT_AB_tmp2 = ein.utils.tensor_factory("cT_AB_tmp2", [cache["Cvir_A"].shape[1], Tmo_AB.shape[1]], np.float64, 'numpy')
            cT_AB = ein.utils.tensor_factory("cT_AB", [cache["Cvir_A"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
            plan_matmul_tt.execute(0.0, cT_AB_tmp1, 1.0, cache["Cvir_A"].T, x_B_MOA.T)
            plan_matmul_tt.execute(0.0, cT_AB_tmp2, 1.0, cT_AB_tmp1, Tmo_AB)
            plan_matmul_tt.execute(0.0, cT_AB, 1.0, cT_AB_tmp2.T, cache["Cocc_B"].T)

            cT_BA_tmp1 = ein.utils.tensor_factory("cT_BA_tmp1", [cache["Cvir_B"].shape[1], x_A_MOB.shape[0]], np.float64, 'numpy')
            cT_BA_tmp2 = ein.utils.tensor_factory("cT_BA_tmp2", [cache["Cvir_B"].shape[1], Tmo_AB.shape[0]], np.float64, 'numpy')
            cT_BA = ein.utils.tensor_factory("cT_BA", [cache["Cvir_B"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')
            plan_matmul_tt.execute(0.0, cT_BA_tmp1, 1.0, cache["Cvir_B"].T, x_A_MOB.T)
            plan_matmul_tt.execute(0.0, cT_BA_tmp2, 1.0, cT_BA_tmp1, Tmo_AB.T)
            plan_matmul_tt.execute(0.0, cT_BA, 1.0, cT_BA_tmp2.T, cache["Cocc_A"].T)

            ind_ab_total = 2.0 * (ein.core.dot(cT_A, EX_AA_inf) + ein.core.dot(cT_AB, EX_AB_inf))
            ind_ba_total = 2.0 * (ein.core.dot(cT_B, EX_BB_inf) + ein.core.dot(cT_BA, EX_BA_inf))
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

    cache["wfn_A"].set_jk(jk)
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
        plan_outer.execute(0.0, P_X, 1.0, eps_occ, ones_vir)
        eps_vir_2D = ein.utils.tensor_factory("eps_vir_2D", [eps_occ.shape[0], eps_vir.shape[0]], np.float64, 'einsums')
        plan_outer.execute(0.0, eps_vir_2D, 1.0, ones_occ, eps_vir)
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
        # TODO: to convert to einsums fully here, would need to re-write
        # cphf_HX, onel_Hx, and twoel_Hx functions in libscf_solver/uhf.cc
        for i in range(len(x_vec) // 2):
            print(x_vec[0][2 * i])
        if act_mask[0]:
            xA = cache["wfn_A"].cphf_Hx([core.Matrix.from_array(x_vec[0])])[0].np
        else:
            xA = False

        if act_mask[1]:
            xB = cache["wfn_B"].cphf_Hx([core.Matrix.from_array(x_vec[1])])[0].np
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

    # print function
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
