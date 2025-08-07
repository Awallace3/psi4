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

    # First grab the orbitals
    cache["Cocc_A"] = ein.core.RuntimeTensorD(wfn_A.Ca_subset("AO", "OCC").np)
    cache['Cocc_A'].set_name("Cocc_A")
    cache["Cvir_A"] = ein.core.RuntimeTensorD(wfn_A.Ca_subset("AO", "VIR").np)
    cache['Cvir_A'].set_name("Cvir_A")

    cache["Cocc_B"] =  ein.core.RuntimeTensorD(wfn_B.Ca_subset("AO", "OCC").np)
    cache['Cocc_B'].set_name("Cocc_B")
    cache["Cvir_B"] =  ein.core.RuntimeTensorD(wfn_B.Ca_subset("AO", "VIR").np)
    cache['Cvir_B'].set_name("Cvir_B")

    cache["eps_occ_A"] = ein.core.RuntimeTensorD(wfn_A.epsilon_a_subset("AO", "OCC").np)
    cache["eps_vir_A"] = ein.core.RuntimeTensorD(wfn_A.epsilon_a_subset("AO", "VIR").np)
    cache["eps_occ_B"] = ein.core.RuntimeTensorD(wfn_B.epsilon_a_subset("AO", "OCC").np)
    cache["eps_vir_B"] = ein.core.RuntimeTensorD(wfn_B.epsilon_a_subset("AO", "VIR").np)

    cache["eps_occ_A"].set_name("eps_occ_A")
    cache["eps_vir_A"].set_name("eps_vir_A")
    cache["eps_occ_B"].set_name("eps_occ_B")
    cache["eps_vir_B"].set_name("eps_vir_B")

    # Build the densities as HF takes an extra "step"

    # Should be fine, but only fills in half the matrix... is it because of the symmetry? How can I fix this to use plan_matmul_tT?
    # plan_matmul_tT = ein.core.compile_plan("ij", "ik", "jk")
    plan_matmul_tt = ein.core.compile_plan("ij", "ik", "kj")
    # D_X corresponds to P^{X,occ}
    cache["D_A"] = ein.utils.tensor_factory("D_A", [cache["Cocc_A"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')
    cache["D_B"] = ein.utils.tensor_factory("D_B", [cache["Cocc_B"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, cache['D_A'], 1.0, cache['Cocc_A'], cache['Cocc_A'].T)
    plan_matmul_tt.execute(0.0, cache['D_B'], 1.0, cache['Cocc_B'], cache['Cocc_B'].T)

    # P_X corresponds to P^{X,vir}
    cache["P_A"] = ein.utils.tensor_factory("P_A", [cache["Cvir_A"].shape[0], cache["Cvir_A"].shape[0]], np.float64, 'numpy')
    cache["P_B"] = ein.utils.tensor_factory("P_B", [cache["Cvir_B"].shape[0], cache["Cvir_B"].shape[0]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, cache['P_A'], 1.0, cache['Cvir_A'], cache['Cvir_A'].T)
    plan_matmul_tt.execute(0.0, cache['P_B'], 1.0, cache['Cvir_B'], cache['Cvir_B'].T)

    # Potential ints
    mints = core.MintsHelper(wfn_A.basisset())
    # cache["V_A"] = ein.core.RuntimeTensorD(mints.ao_potential().np)
    tmp = mints.ao_potential().np
    cache["V_A"] = ein.utils.tensor_factory("V_A", [tmp.shape[0], tmp.shape[1]], np.float64, 'numpy')
    cache["V_A"][:] = tmp[:]
    mints = core.MintsHelper(wfn_B.basisset())
    tmp = mints.ao_potential().np
    cache["V_B"] = ein.utils.tensor_factory("V_B", [tmp.shape[0], tmp.shape[1]], np.float64, 'numpy')
    cache["V_B"][:] = tmp[:]

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

    # J and K matrices
    jk.C_clear()

    # Normal J/K for Monomer A
    jk.C_left_add(wfn_A.Ca_subset("SO", "OCC"))
    jk.C_right_add(wfn_A.Ca_subset("SO", "OCC"))

    # Normal J/K for Monomer B
    jk.C_left_add(wfn_B.Ca_subset("SO", "OCC"))
    jk.C_right_add(wfn_B.Ca_subset("SO", "OCC"))

    # K_O J/K
    # C_O_A = core.triplet(cache["D_B"], cache["S"], cache["Cocc_A"], False, False, False)
    C_O_A = ein.utils.tensor_factory("C_O_A", [cache["D_B"].shape[0], cache["Cocc_A"].shape[1]], np.float64, 'numpy')
    DB_S = ein.utils.tensor_factory("DB_S", [cache["D_B"].shape[0], cache["S"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, DB_S, 1.0, cache["D_B"], cache["S"])
    plan_matmul_tt.execute(0.0, C_O_A, 1.0, DB_S, cache["Cocc_A"])

    C_O_A_matrix = core.Matrix.from_array(C_O_A)
    # Q: How can I convert jk to use RuntimeTensorD?
    jk.C_left_add(C_O_A_matrix)
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_A"]))

    jk.compute()

    # Clone them as the JK object will overwrite.
    cache["J_A"] = ein.core.RuntimeTensorD(jk.J()[0].clone().np)
    cache["K_A"] = ein.core.RuntimeTensorD(jk.K()[0].clone().np)
    cache["J_B"] = ein.core.RuntimeTensorD(jk.J()[1].clone().np)
    cache["K_B"] = ein.core.RuntimeTensorD(jk.K()[1].clone().np)
    cache["J_O"] = ein.core.RuntimeTensorD(jk.J()[2].clone().np)
    cache["K_O"] = ein.core.RuntimeTensorD(jk.K()[2].clone().np).T

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


def exchange(cache, jk, do_print=True):
    r"""
    Computes the E10 exchange (S^2 and S^inf) from a build_sapt_jk_cache datacache.

    Equation E^{(1)}_{\rm exch}(S^2) = 
        -2(P^{A,occ} S^{AO} P^{B,occ} S^{AO} P^{A,vir}) \cdot \omega^{B}
        -2(P^{B,occ} S^{AO} P^{A,occ} S^{AO} P^{B,vir}) \cdot \omega^{B}
        -2(P^{A,vir} S^{AO} P^{B,occ}) \cdot K[P^{A,occ} S^{AO} P^{B,vir}]

    """

    if do_print:
        core.print_out("\n  ==> E10 Exchange <== \n\n")

    plan_matmul_tt = ein.core.compile_plan("ij", "ik", "kj")
    plan_vector_dot = ein.core.compile_plan("", "ij", "ij")

    # Build potenitals
    h_A = cache["V_A"].copy()
    print("EINSUMS EXCHANGE")
    ein.core.axpy(2.0, cache["J_A"], h_A)
    ein.core.axpy(-1.0, cache["K_A"], h_A)

    h_B = cache["V_B"].copy()
    ein.core.axpy(2.0, cache["J_B"], h_B)
    ein.core.axpy(-1.0, cache["K_B"], h_B)

    w_A = cache["V_A"].copy()
    ein.core.axpy(2.0, cache["J_A"], w_A)

    w_B = cache["V_B"].copy()
    ein.core.axpy(2.0, cache["J_B"], w_B)

    # Build inverse exchange metric
    nocc_A = cache["Cocc_A"].shape[1]
    nocc_B = cache["Cocc_B"].shape[1]
    SA = ein.utils.tensor_factory("SAB", [cache["Cocc_A"].shape[1], cache["S"].shape[1]], np.float64, 'numpy')
    SAB = ein.utils.tensor_factory("SAB", [cache["Cocc_A"].shape[1], cache["Cocc_B"].shape[1]], np.float64, 'numpy')
    ein.core.gemm("T", "N", 1.0, cache["Cocc_A"], cache["S"], 0.0, SA)
    ein.core.gemm("N", "N", 1.0, SA, cache["Cocc_B"], 0.0, SAB)
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

    S = cache["S"]

    D_A = cache["D_A"]
    P_A = cache["P_A"]

    D_B = cache["D_B"]
    P_B = cache["P_B"]

    # Compute the J and K matrices
    jk.C_clear()

    # QA: I am getting the right answer, but I do not understand exactly how
    # this maps to equations 11, 12, 13, and 9... Shouldn't jk_C_right_tmp be
    # (Tmo_AA @ Cocc_A.T) instead of the (Cocc_A @ Tmo_AA)? Also, does it
    # matter if we put Tmo_AA on the left or right of Cocc_A?
    jk.C_left_add(core.Matrix.from_array(cache["Cocc_A"])) 
    jk_C_right_tmp = ein.utils.tensor_factory("jk_C_right_tmp", [cache["Cocc_A"].shape[0], Tmo_AA.shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, jk_C_right_tmp, 1.0, cache["Cocc_A"], Tmo_AA)
    jk.C_right_add(core.Matrix.from_array(jk_C_right_tmp))

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_B"]))
    jk_C_right_tmp = ein.utils.tensor_factory("jk_C_right_tmp", [cache["Cocc_A"].shape[0], Tmo_AB.shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, jk_C_right_tmp, 1.0, cache["Cocc_A"], Tmo_AB)
    jk.C_right_add(core.Matrix.from_array(jk_C_right_tmp))

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_A"]))
    PB_S = ein.utils.tensor_factory("PB_S", [P_B.shape[0], S.shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, PB_S, 1.0, P_B, S)
    PB_S_CA = ein.utils.tensor_factory("PB_S", [P_B.shape[0], cache["Cocc_A"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, PB_S_CA, 1.0, PB_S, cache['Cocc_A'])
    jk.C_right_add(core.Matrix.from_array(PB_S_CA))
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

    DA_S = ein.utils.tensor_factory("DA_S", [D_A.shape[0], S.shape[1]], np.float64, 'numpy')
    DA_S_DB = ein.utils.tensor_factory("DA_S_DB", [D_A.shape[0], D_B.shape[-1]], np.float64, 'numpy')
    DA_S_DB_S = ein.utils.tensor_factory("DA_S_DB_S", [D_A.shape[0], S.shape[1]], np.float64, 'numpy')
    DA_S_DB_S_PA = ein.utils.tensor_factory("DA_S_DB_S_PA", [D_A.shape[0], P_A.shape[0]], np.float64, 'numpy')
    wB_DA_S_DB_S_PA = ein.utils.tensor_factory("wB_DA_S_DB_S_PA", [1], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, DA_S, 1.0, D_A, S)
    plan_matmul_tt.execute(0.0, DA_S_DB, 1.0, DA_S, D_B)
    plan_matmul_tt.execute(0.0, DA_S_DB_S, 1.0, DA_S_DB, S)
    plan_matmul_tt.execute(0.0, DA_S_DB_S_PA, 1.0, DA_S_DB_S, P_A)
    plan_vector_dot.execute(0.0, wB_DA_S_DB_S_PA, 1.0, w_B, DA_S_DB_S_PA)
    Exch_s2 -= 2.0 * wB_DA_S_DB_S_PA[0]

    # tmp = core.Matrix.chain_dot(D_B, S, D_A, S, P_B)
    DB_S = ein.utils.tensor_factory("DB_S", [D_B.shape[0], S.shape[1]], np.float64, 'numpy')
    DB_S_DA = ein.utils.tensor_factory("DB_S_DA", [D_B.shape[0], D_A.shape[-1]], np.float64, 'numpy')
    DB_S_DA_S = ein.utils.tensor_factory("DB_S_DA_S", [D_B.shape[0], S.shape[1]], np.float64, 'numpy')
    DB_S_DA_S_PB = ein.utils.tensor_factory("DB_S_DA_S_PB", [D_B.shape[0], P_B.shape[0]], np.float64, 'numpy')
    wA_DB_S_DA_S_PB = ein.utils.tensor_factory("wA_DB_S_DA_S_PB", [1], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, DB_S, 1.0, D_B, S)
    plan_matmul_tt.execute(0.0, DB_S_DA, 1.0, DB_S, D_A)
    plan_matmul_tt.execute(0.0, DB_S_DA_S, 1.0, DB_S_DA, S)
    plan_matmul_tt.execute(0.0, DB_S_DA_S_PB, 1.0, DB_S_DA_S, P_B)
    plan_vector_dot.execute(0.0, wA_DB_S_DA_S_PB, 1.0, w_A, DB_S_DA_S_PB)
    Exch_s2 -= 2.0 * wA_DB_S_DA_S_PB[0]

    # NOTE: Kij is wrong, but PA_S_DB is correct, looking above...
    PA_S = ein.utils.tensor_factory("PA_S", [P_A.shape[0], S.shape[1]], np.float64, 'numpy')
    PA_S_DB = ein.utils.tensor_factory("PA_S_DB", [P_A.shape[0], D_B.shape[1]], np.float64, 'numpy')
    Kij_PA_S_DB = ein.utils.tensor_factory("Kij_PA_S_DB", [1], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, PA_S, 1.0, P_A, S)
    plan_matmul_tt.execute(0.0, PA_S_DB, 1.0, PA_S, D_B)
    plan_vector_dot.execute(0.0, Kij_PA_S_DB, 1.0, Kij, PA_S_DB)
    Exch_s2 -= 2.0 * Kij_PA_S_DB[0]

    if do_print:
        core.print_out(print_sapt_var("Exch10(S^2) ", Exch_s2, short=True))
        core.print_out("\n")

    # Start Sinf
    Exch10 = 0.0
    DA_KB = ein.utils.tensor_factory("DA_KB", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, DA_KB, 1.0, D_A, cache["K_B"])
    Exch10 -= 2.0 * DA_KB[0]
    TA_hB = ein.utils.tensor_factory("TA_hB", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, TA_hB, 1.0, T_A, h_B)
    Exch10 += 2.0 * TA_hB[0]
    TB_hA = ein.utils.tensor_factory("TB_hA", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, TB_hA, 1.0, T_B, h_A)
    Exch10 += 2.0 * TB_hA[0]
    T_hAphB = ein.utils.tensor_factory("T_hAphB", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, T_hAphB, 1.0, T_AB, h_A + h_B)
    Exch10 += 2.0 * T_hAphB[0]
    T_B_JT_ABmKT_AB = ein.utils.tensor_factory("T_B_JT_ABmKT_AB", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, T_B_JT_ABmKT_AB, 1.0, T_B, JT_AB - 0.5 * KT_AB)
    Exch10 += 4.0 * T_B_JT_ABmKT_AB[0]
    T_A_JT_ABmKT_AB = ein.utils.tensor_factory("T_A_JT_ABmKT_AB", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, T_A_JT_ABmKT_AB, 1.0, T_A, JT_AB - 0.5 * KT_AB.T)
    Exch10 += 4.0 * T_A_JT_ABmKT_AB[0]
    TB_JTAmKTA = ein.utils.tensor_factory("TB_JTAmKTA", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, TB_JTAmKTA, 1.0, T_B, JT_A - 0.5 * KT_A)
    Exch10 += 4.0 * TB_JTAmKTA[0]
    TAB_JTABmKTAB = ein.utils.tensor_factory("TAB_JTABmKTAB", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, TAB_JTABmKTAB, 1.0, T_AB, JT_AB - 0.5 * KT_AB.T)
    Exch10 += 4.0 * TAB_JTABmKTAB[0]

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
    
    DB_S = ein.utils.tensor_factory("DB_S", [D_B.shape[0], S.shape[1]], np.float64, 'numpy')
    DB_S_CA = ein.utils.tensor_factory("DB_S_CA", [D_B.shape[0], cache["Cocc_A"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, DB_S, 1.0, D_B, S)
    plan_matmul_tt.execute(0.0, DB_S_CA, 1.0, DB_S, cache["Cocc_A"])
    jk.C_left_add(core.Matrix.from_array(DB_S_CA))
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_A"]))

    DB_S_DA = ein.utils.tensor_factory("DB_S_DA", [D_B.shape[0], D_A.shape[1]], np.float64, 'numpy')
    DB_S_DA_S = ein.utils.tensor_factory("DB_S_DA_S", [D_B.shape[0], S.shape[1]], np.float64, 'numpy')
    DB_S_DA_S_CB = ein.utils.tensor_factory("DB_S_DA_S_CB", [D_B.shape[0], cache["Cocc_B"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, DB_S_DA, 1.0, DB_S, D_A)
    plan_matmul_tt.execute(0.0, DB_S_DA_S, 1.0, DB_S_DA, S)
    plan_matmul_tt.execute(0.0, DB_S_DA_S_CB, 1.0, DB_S_DA_S, cache["Cocc_B"])
    jk.C_left_add(core.Matrix.from_array(DB_S_DA_S_CB))
    jk.C_right_add(core.Matrix.from_array(cache["Cocc_B"]))

    DA_S = ein.utils.tensor_factory("DA_S", [D_A.shape[0], S.shape[1]], np.float64, 'numpy')
    DA_S_DB = ein.utils.tensor_factory("DA_S_DB", [D_A.shape[0], D_B.shape[1]], np.float64, 'numpy')
    DA_S_DB_S = ein.utils.tensor_factory("DA_S_DB", [D_A.shape[0], S.shape[1]], np.float64, 'numpy')
    DA_S_DB_S_CA = ein.utils.tensor_factory("DA_S_DB_S_CA", [D_A.shape[0], cache["Cocc_A"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, DA_S, 1.0, D_A, S)
    plan_matmul_tt.execute(0.0, DA_S_DB, 1.0, DA_S, D_B)
    plan_matmul_tt.execute(0.0, DA_S_DB_S, 1.0, DA_S_DB, S)
    plan_matmul_tt.execute(0.0, DA_S_DB_S_CA, 1.0, DA_S_DB_S, cache["Cocc_A"])
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

    # Create intermediate tensors for EX_A chain operations
    S_DB = ein.utils.tensor_factory("S_DB", [S.shape[0], D_B.shape[1]], np.float64, 'numpy')
    S_DB_VA = ein.utils.tensor_factory("S_DB_VA", [S.shape[0], V_A.shape[1]], np.float64, 'numpy')
    S_DB_JA = ein.utils.tensor_factory("S_DB_JA", [S.shape[0], J_A.shape[1]], np.float64, 'numpy') 
    S_DB_KA = ein.utils.tensor_factory("S_DB_KA", [S.shape[0], K_A.shape[1]], np.float64, 'numpy')
    S_DB_S = ein.utils.tensor_factory("S_DB_S", [S.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DB_S_DA = ein.utils.tensor_factory("S_DB_S_DA", [S.shape[0], D_A.shape[1]], np.float64, 'numpy')
    S_DB_S_DA_VB = ein.utils.tensor_factory("S_DB_S_DA_VB", [S.shape[0], V_B.shape[1]], np.float64, 'numpy')
    S_DB_S_DA_JB = ein.utils.tensor_factory("S_DB_S_DA_JB", [S.shape[0], J_B.shape[1]], np.float64, 'numpy')
    S_DB_VA_DB_S = ein.utils.tensor_factory("S_DB_VA_DB_S", [S.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DB_JA_DB_S = ein.utils.tensor_factory("S_DB_JA_DB_S", [S.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DB_KO = ein.utils.tensor_factory("S_DB_KO", [S.shape[0], K_O.shape[0]], np.float64, 'numpy')
    VB_DB_S = ein.utils.tensor_factory("VB_DB_S", [V_B.shape[0], S.shape[1]], np.float64, 'numpy')
    JB_DB_S = ein.utils.tensor_factory("JB_DB_S", [J_B.shape[0], S.shape[1]], np.float64, 'numpy')
    KB_DB_S = ein.utils.tensor_factory("KB_DB_S", [K_B.shape[0], S.shape[1]], np.float64, 'numpy')
    VB_DA_S_DB_S = ein.utils.tensor_factory("VB_DA_S_DB_S", [V_B.shape[0], S.shape[1]], np.float64, 'numpy')
    JB_DA_S_DB_S = ein.utils.tensor_factory("JB_DA_S_DB_S", [J_B.shape[0], S.shape[1]], np.float64, 'numpy')
    KO_DB_S = ein.utils.tensor_factory("KO_DB_S", [K_O.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DB_VA_DB = ein.utils.tensor_factory("S_DB_VA_DB", [S_DB.shape[0], D_B.shape[1]], np.float64, 'numpy')
    S_DB_VA_DB_S = ein.utils.tensor_factory("S_DB_VA_DB_S", [S_DB.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DB_JA_DB = ein.utils.tensor_factory("S_DB_JA_DB", [J_A.shape[0], D_B.shape[1]], np.float64, 'numpy')
    S_DB_JA_DB_S = ein.utils.tensor_factory("S_DB_JA_DB_S", [J_A.shape[0], S.shape[1]], np.float64, 'numpy')
    VB_DA_S = ein.utils.tensor_factory("VB_DA_S", [V_B.shape[0], S.shape[1]], np.float64, 'numpy')
    JB_DA_S = ein.utils.tensor_factory("JB_DA_S", [J_B.shape[0], S.shape[1]], np.float64, 'numpy')

    # Compute all the EX_A chain operations using einsums
    plan_matmul_tt.execute(0.0, S_DB, 1.0, S, D_B)
    plan_matmul_tt.execute(0.0, S_DB_VA, 1.0, S_DB, V_A)
    plan_matmul_tt.execute(0.0, S_DB_JA, 1.0, S_DB, J_A)
    plan_matmul_tt.execute(0.0, S_DB_KA, 1.0, S_DB, K_A)
    plan_matmul_tt.execute(0.0, S_DB_S, 1.0, S_DB, S)
    plan_matmul_tt.execute(0.0, S_DB_S_DA, 1.0, S_DB_S, D_A)
    plan_matmul_tt.execute(0.0, S_DB_S_DA_VB, 1.0, S_DB_S_DA, V_B)
    plan_matmul_tt.execute(0.0, S_DB_S_DA_JB, 1.0, S_DB_S_DA, J_B)
    plan_matmul_tt.execute(0.0, S_DB_VA_DB, 1.0, S_DB_VA, D_B)
    plan_matmul_tt.execute(0.0, S_DB_VA_DB_S, 1.0, S_DB_VA_DB, S)
    plan_matmul_tt.execute(0.0, S_DB_JA_DB, 1.0, S_DB_JA, D_B)
    plan_matmul_tt.execute(0.0, S_DB_JA_DB_S, 1.0, S_DB_JA_DB, S)
    plan_matmul_tt.execute(0.0, S_DB_KO, 1.0, S_DB, K_O.T)
    plan_matmul_tt.execute(0.0, VB_DB_S, 1.0, V_B, DB_S)
    plan_matmul_tt.execute(0.0, JB_DB_S, 1.0, J_B, DB_S)
    plan_matmul_tt.execute(0.0, KB_DB_S, 1.0, K_B, DB_S)
    plan_matmul_tt.execute(0.0, VB_DA_S, 1.0, V_B, DA_S)
    plan_matmul_tt.execute(0.0, VB_DA_S_DB_S, 1.0, VB_DA_S, DB_S)
    plan_matmul_tt.execute(0.0, JB_DA_S, 1.0, J_B, DA_S)
    plan_matmul_tt.execute(0.0, JB_DA_S_DB_S, 1.0, JB_DA_S, DB_S)
    plan_matmul_tt.execute(0.0, KO_DB_S, 1.0, K_O, DB_S)

    # Apply all the axpy operations to EX_A
    ein.core.axpy(-1.0, S_DB_VA, EX_A)
    ein.core.axpy(-2.0, S_DB_JA, EX_A)
    ein.core.axpy(1.0, S_DB_KA, EX_A)
    ein.core.axpy(1.0, S_DB_S_DA_VB, EX_A)
    ein.core.axpy(2.0, S_DB_S_DA_JB, EX_A)
    ein.core.axpy(1.0, S_DB_VA_DB_S, EX_A)
    ein.core.axpy(2.0, S_DB_JA_DB_S, EX_A)
    ein.core.axpy(-1.0, S_DB_KO, EX_A)
    ein.core.axpy(-1.0, VB_DB_S, EX_A)
    ein.core.axpy(-2.0, JB_DB_S, EX_A)
    ein.core.axpy(1.0, KB_DB_S, EX_A)
    ein.core.axpy(1.0, VB_DA_S_DB_S, EX_A)
    ein.core.axpy(2.0, JB_DA_S_DB_S, EX_A)
    ein.core.axpy(-1.0, KO_DB_S, EX_A)

    # Transform EX_A to MO basis: C_A^T @ EX_A @ C_vir_A
    EX_A_tmp = ein.utils.tensor_factory("EX_A_tmp", [cache["Cocc_A"].shape[1], EX_A.shape[1]], np.float64, 'numpy')
    EX_A_MO = ein.utils.tensor_factory("EX_A_MO", [cache["Cocc_A"].shape[1], cache["Cvir_A"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, EX_A_tmp, 1.0, cache["Cocc_A"].T, EX_A)
    plan_matmul_tt.execute(0.0, EX_A_MO, 1.0, EX_A_tmp, cache["Cvir_A"])
    # This shows that there is an error in the einsums chain above...

    # Complete EX_B construction using einsums
    EX_B = K_A.copy()
    EX_B *= -1.0
    ein.core.axpy(-2.0, J_O, EX_B)
    ein.core.axpy(1.0, K_O.T, EX_B)
    # ISSUE WITH J_P_A...
    ein.core.axpy(2.0, J_P_A, EX_B)

    # Create intermediate tensors for EX_A chain operations
    S_DA = ein.utils.tensor_factory("S_DA", [S.shape[0], D_A.shape[1]], np.float64, 'numpy')
    DA_S = ein.utils.tensor_factory("DA_S", [D_A.shape[1], S.shape[0]], np.float64, 'numpy')
    S_DA_VB = ein.utils.tensor_factory("S_DA_VB", [S.shape[0], V_B.shape[1]], np.float64, 'numpy')
    S_DA_JB = ein.utils.tensor_factory("S_DA_JB", [S.shape[0], J_B.shape[1]], np.float64, 'numpy') 
    S_DA_KB = ein.utils.tensor_factory("S_DA_KB", [S.shape[0], K_B.shape[1]], np.float64, 'numpy')
    S_DA_S = ein.utils.tensor_factory("S_DA_S", [S.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DA_S_DB = ein.utils.tensor_factory("S_DA_S_DB", [S.shape[0], D_B.shape[1]], np.float64, 'numpy')
    S_DA_S_DB_VA = ein.utils.tensor_factory("S_DA_S_DB_VA", [S.shape[0], V_A.shape[1]], np.float64, 'numpy')
    S_DA_S_DB_JA = ein.utils.tensor_factory("S_DA_S_DB_JA", [S.shape[0], J_A.shape[1]], np.float64, 'numpy')
    S_DA_VB_DA_S = ein.utils.tensor_factory("S_DA_VB_DA_S", [S.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DA_JB_DA_S = ein.utils.tensor_factory("S_DA_JB_DA_S", [S.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DA_KO = ein.utils.tensor_factory("S_DA_KO", [S.shape[0], K_O.shape[0]], np.float64, 'numpy')
    VA_DA_S = ein.utils.tensor_factory("VA_DA_S", [V_A.shape[0], S.shape[1]], np.float64, 'numpy')
    JA_DA = ein.utils.tensor_factory("JA_DA", [J_A.shape[0], D_A.shape[1]], np.float64, 'numpy')
    JA_DA_S = ein.utils.tensor_factory("JA_DA_S", [J_A.shape[0], S.shape[1]], np.float64, 'numpy')
    KA_DA_S = ein.utils.tensor_factory("KA_DA_S", [K_A.shape[0], S.shape[1]], np.float64, 'numpy')
    VA_DB_S_DA_S = ein.utils.tensor_factory("VA_DB_S_DA_S", [V_A.shape[0], S.shape[1]], np.float64, 'numpy')
    JA_DB_S_DA_S = ein.utils.tensor_factory("JA_DB_S_DA_S", [J_A.shape[0], S.shape[1]], np.float64, 'numpy')
    KO_DA = ein.utils.tensor_factory("KO_DA", [K_O.shape[0], D_A.shape[1]], np.float64, 'numpy')
    KO_DA_S = ein.utils.tensor_factory("KO_DA_S", [K_O.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DA_VB_DA = ein.utils.tensor_factory("S_DA_VB_DA", [S_DA.shape[0], D_A.shape[1]], np.float64, 'numpy')
    S_DA_VB_DA_S = ein.utils.tensor_factory("S_DA_VB_DA_S", [S_DA.shape[0], S.shape[1]], np.float64, 'numpy')
    S_DA_JB_DA = ein.utils.tensor_factory("S_DA_JB_DA", [J_B.shape[0], D_A.shape[1]], np.float64, 'numpy')
    S_DA_JB_DA_S = ein.utils.tensor_factory("S_DA_JB_DA_S", [J_B.shape[0], S.shape[1]], np.float64, 'numpy')
    VA_DB_S = ein.utils.tensor_factory("VA_DB_S", [V_A.shape[0], S.shape[1]], np.float64, 'numpy')
    JA_DB_S = ein.utils.tensor_factory("JA_DB_S", [J_A.shape[0], S.shape[1]], np.float64, 'numpy')

    # Compute all the EX_B chain operations using einsums
    plan_matmul_tt.execute(0.0, S_DA, 1.0, S, D_A)
    plan_matmul_tt.execute(0.0, DA_S, 1.0, D_A, S)
    plan_matmul_tt.execute(0.0, S_DA_VB, 1.0, S_DA, V_B)
    plan_matmul_tt.execute(0.0, S_DA_JB, 1.0, S_DA, J_B)
    plan_matmul_tt.execute(0.0, S_DA_KB, 1.0, S_DA, K_B)
    plan_matmul_tt.execute(0.0, S_DA_S, 1.0, S_DA, S)
    plan_matmul_tt.execute(0.0, S_DA_S_DB, 1.0, S_DA_S, D_B)
    plan_matmul_tt.execute(0.0, S_DA_S_DB_VA, 1.0, S_DA_S_DB, V_A)
    plan_matmul_tt.execute(0.0, S_DA_S_DB_JA, 1.0, S_DA_S_DB, J_A)
    plan_matmul_tt.execute(0.0, S_DA_VB_DA, 1.0, S_DA_VB, D_A)
    plan_matmul_tt.execute(0.0, S_DA_VB_DA_S, 1.0, S_DA_VB_DA, S)
    plan_matmul_tt.execute(0.0, S_DA_JB_DA, 1.0, S_DA_JB, D_A)
    plan_matmul_tt.execute(0.0, S_DA_JB_DA_S, 1.0, S_DA_JB_DA, S)
    plan_matmul_tt.execute(0.0, S_DA_KO, 1.0, S_DA, K_O)
    plan_matmul_tt.execute(0.0, VA_DA_S, 1.0, V_A, DA_S)
    plan_matmul_tt.execute(0.0, JA_DA, 1.0, J_A, D_A)
    plan_matmul_tt.execute(0.0, JA_DA_S, 1.0, JA_DA, S)
    plan_matmul_tt.execute(0.0, KA_DA_S, 1.0, K_A, DA_S)
    plan_matmul_tt.execute(0.0, VA_DB_S, 1.0, V_A, DB_S)
    plan_matmul_tt.execute(0.0, VA_DB_S_DA_S, 1.0, VA_DB_S, DA_S)
    plan_matmul_tt.execute(0.0, JA_DB_S, 1.0, J_A, DB_S)
    plan_matmul_tt.execute(0.0, JA_DB_S_DA_S, 1.0, JA_DB_S, DA_S)
    plan_matmul_tt.execute(0.0, KO_DA, 1.0, K_O.T, D_A)
    plan_matmul_tt.execute(0.0, KO_DA_S, 1.0, KO_DA, S)

    # Bpply all the axpy operations to EX_B
    ein.core.axpy(-1.0, S_DA_VB, EX_B)
    ein.core.axpy(-2.0, S_DA_JB, EX_B)
    ein.core.axpy(1.0, S_DA_KB, EX_B)
    ein.core.axpy(1.0, S_DA_S_DB_VA, EX_B)
    ein.core.axpy(2.0, S_DA_S_DB_JA, EX_B)
    ein.core.axpy(1.0, S_DA_VB_DA_S, EX_B)
    ein.core.axpy(2.0, S_DA_JB_DA_S, EX_B)
    ein.core.axpy(-1.0, S_DA_KO, EX_B)

    ein.core.axpy(-1.0, VA_DA_S, EX_B)
    ein.core.axpy(-2.0, JA_DA_S, EX_B)
    ein.core.axpy(1.0, KA_DA_S, EX_B)
    ein.core.axpy(1.0, VA_DB_S_DA_S, EX_B)
    ein.core.axpy(2.0, JA_DB_S_DA_S, EX_B)
    ein.core.axpy(-1.0, KO_DA_S, EX_B)

    # Transform EX_B to MO basis: C_B^T @ EX_B @ C_vir_B
    EX_B_tmp = ein.utils.tensor_factory("EX_B_tmp", [cache["Cocc_B"].shape[1], EX_B.shape[1]], np.float64, 'numpy')
    EX_B_MO = ein.utils.tensor_factory("EX_B_MO", [cache["Cocc_B"].shape[1], cache["Cvir_B"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, EX_B_tmp, 1.0, cache["Cocc_B"].T, EX_B)
    plan_matmul_tt.execute(0.0, EX_B_MO, 1.0, EX_B_tmp, cache["Cvir_B"])

    # Build electrostatic potential using einsums
    w_A = V_A.copy()
    ein.core.axpy(2.0, J_A, w_A)
    
    w_B = V_B.copy()
    ein.core.axpy(2.0, J_B, w_B)

    # Transform to MO basis: C_A^T @ w_B @ C_vir_A
    w_B_tmp = ein.utils.tensor_factory("w_B_tmp", [cache["Cocc_A"].shape[1], w_B.shape[1]], np.float64, 'numpy')
    w_B_MOA = ein.utils.tensor_factory("w_B_MOA", [cache["Cocc_A"].shape[1], cache["Cvir_A"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, w_B_tmp, 1.0, cache["Cocc_A"].T, w_B)
    plan_matmul_tt.execute(0.0, w_B_MOA, 1.0, w_B_tmp, cache["Cvir_A"])

    w_A_tmp = ein.utils.tensor_factory("w_A_tmp", [cache["Cocc_B"].shape[1], w_A.shape[1]], np.float64, 'numpy')
    w_A_MOB = ein.utils.tensor_factory("w_A_MOB", [cache["Cocc_B"].shape[1], cache["Cvir_B"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, w_A_tmp, 1.0, cache["Cocc_B"].T, w_A)
    plan_matmul_tt.execute(0.0, w_A_MOB, 1.0, w_A_tmp, cache["Cvir_B"])

    # Do uncoupled induction calculations
    core.print_out("   => Uncoupled Induction <= \n\n")
    
    # Create uncoupled response vectors by element-wise division
    unc_x_B_MOA = w_B_MOA.copy()
    unc_x_A_MOB = w_A_MOB.copy()
    
    # Convert to numpy for element-wise operations, then back to einsums
    eps_occ_A_np = cache["eps_occ_A"]
    eps_vir_A_np = cache["eps_vir_A"]
    eps_occ_B_np = cache["eps_occ_B"]
    eps_vir_B_np = cache["eps_vir_B"]
    
    # Eq. 20
    for r in range(unc_x_B_MOA.shape[0]):
        for a in range(unc_x_B_MOA.shape[1]):
            unc_x_B_MOA[r, a] /= (eps_occ_A_np[r] - eps_vir_A_np[a])
    
    # Eq. 20
    for r in range(unc_x_A_MOB.shape[0]):
        for a in range(unc_x_A_MOB.shape[1]):
            unc_x_A_MOB[r, a] /= (eps_occ_B_np[r] - eps_vir_B_np[a])

    # Compute uncoupled induction energies using vector dot products
    plan_vector_dot = ein.core.compile_plan("", "ij", "ij")
    
    unc_ind_ab_tensor = ein.utils.tensor_factory("unc_ind_ab", [1], np.float64, 'numpy')
    unc_ind_ba_tensor = ein.utils.tensor_factory("unc_ind_ba", [1], np.float64, 'numpy')
    unc_indexch_ab_tensor = ein.utils.tensor_factory("unc_indexch_ab", [1], np.float64, 'numpy')
    unc_indexch_ba_tensor = ein.utils.tensor_factory("unc_indexch_ba", [1], np.float64, 'numpy')
    
    plan_vector_dot.execute(0.0, unc_ind_ab_tensor, 1.0, unc_x_B_MOA, w_B_MOA)
    plan_vector_dot.execute(0.0, unc_ind_ba_tensor, 1.0, unc_x_A_MOB, w_A_MOB)
    plan_vector_dot.execute(0.0, unc_indexch_ab_tensor, 1.0, unc_x_B_MOA, EX_A_MO)
    plan_vector_dot.execute(0.0, unc_indexch_ba_tensor, 1.0, unc_x_A_MOB, EX_B_MO)

    unc_ind_ab = 2.0 * unc_ind_ab_tensor[0]
    unc_ind_ba = 2.0 * unc_ind_ba_tensor[0]
    unc_indexch_ab = 2.0 * unc_indexch_ab_tensor[0]
    unc_indexch_ba = 2.0 * unc_indexch_ba_tensor[0]

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
        unc_ind_ab_total_tensor_1 = ein.utils.tensor_factory("unc_ind_ab_total_1", [1], np.float64, 'numpy')
        unc_ind_ab_total_tensor_2 = ein.utils.tensor_factory("unc_ind_ab_total_2", [1], np.float64, 'numpy')
        unc_ind_ba_total_tensor_1 = ein.utils.tensor_factory("unc_ind_ba_total_1", [1], np.float64, 'numpy')
        unc_ind_ba_total_tensor_2 = ein.utils.tensor_factory("unc_ind_ba_total_2", [1], np.float64, 'numpy')
        
        plan_vector_dot.execute(0.0, unc_ind_ab_total_tensor_1, 1.0, sT_A, EX_AA_inf)
        plan_vector_dot.execute(0.0, unc_ind_ab_total_tensor_2, 1.0, sT_AB, EX_AB_inf)
        plan_vector_dot.execute(0.0, unc_ind_ba_total_tensor_1, 1.0, sT_B, EX_BB_inf)
        plan_vector_dot.execute(0.0, unc_ind_ba_total_tensor_2, 1.0, sT_BA, EX_BA_inf)

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

        # Convert einsums tensors back to psi4 Matrices for CPHF solver
        w_B_MOA_matrix = core.Matrix.from_array(w_B_MOA)
        w_A_MOB_matrix = core.Matrix.from_array(w_A_MOB)

        x_B_MOA, x_A_MOB = _sapt_cpscf_solve(
            cache, jk, w_B_MOA_matrix, w_A_MOB_matrix, 20, cphf_r_convergence, sapt_jk_B=sapt_jk_B
        )

        # Convert solution vectors back to einsums tensors
        x_B_MOA_ein = ein.core.RuntimeTensorD(x_B_MOA.np)
        x_A_MOB_ein = ein.core.RuntimeTensorD(x_A_MOB.np)

        # Compute coupled induction energies
        ind_ab_tensor = ein.utils.tensor_factory("ind_ab", [1], np.float64, 'numpy')
        ind_ba_tensor = ein.utils.tensor_factory("ind_ba", [1], np.float64, 'numpy')
        indexch_ab_tensor = ein.utils.tensor_factory("indexch_ab", [1], np.float64, 'numpy')
        indexch_ba_tensor = ein.utils.tensor_factory("indexch_ba", [1], np.float64, 'numpy')

        plan_vector_dot.execute(0.0, ind_ab_tensor, 1.0, x_B_MOA_ein, w_B_MOA)
        plan_vector_dot.execute(0.0, ind_ba_tensor, 1.0, x_A_MOB_ein, w_A_MOB)
        plan_vector_dot.execute(0.0, indexch_ab_tensor, 1.0, x_B_MOA_ein, EX_A_MO)
        plan_vector_dot.execute(0.0, indexch_ba_tensor, 1.0, x_A_MOB_ein, EX_B_MO)

        ind_ab = 2.0 * ind_ab_tensor[0]
        ind_ba = 2.0 * ind_ba_tensor[0]
        indexch_ab = 2.0 * indexch_ab_tensor[0]
        indexch_ba = 2.0 * indexch_ba_tensor[0]

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
            cT_A_tmp1 = ein.utils.tensor_factory("cT_A_tmp1", [cache["Cvir_A"].shape[1], x_B_MOA_ein.shape[0]], np.float64, 'numpy')
            cT_A_tmp2 = ein.utils.tensor_factory("cT_A_tmp2", [cache["Cvir_A"].shape[1], Tmo_AA.shape[1]], np.float64, 'numpy')
            cT_A = ein.utils.tensor_factory("cT_A", [cache["Cvir_A"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')
            plan_matmul_tt.execute(0.0, cT_A_tmp1, 1.0, cache["Cvir_A"].T, x_B_MOA_ein.T)
            plan_matmul_tt.execute(0.0, cT_A_tmp2, 1.0, cT_A_tmp1, Tmo_AA)
            plan_matmul_tt.execute(0.0, cT_A, 1.0, cT_A_tmp2.T, cache["Cocc_A"].T)

            cT_B_tmp1 = ein.utils.tensor_factory("cT_B_tmp1", [cache["Cvir_B"].shape[1], x_A_MOB_ein.shape[0]], np.float64, 'numpy')
            cT_B_tmp2 = ein.utils.tensor_factory("cT_B_tmp2", [cache["Cvir_B"].shape[1], Tmo_BB.shape[1]], np.float64, 'numpy')
            cT_B = ein.utils.tensor_factory("cT_B", [cache["Cvir_B"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
            plan_matmul_tt.execute(0.0, cT_B_tmp1, 1.0, cache["Cvir_B"].T, x_A_MOB_ein.T)
            plan_matmul_tt.execute(0.0, cT_B_tmp2, 1.0, cT_B_tmp1, Tmo_BB)
            plan_matmul_tt.execute(0.0, cT_B, 1.0, cT_B_tmp2.T, cache["Cocc_B"].T)

            cT_AB_tmp1 = ein.utils.tensor_factory("cT_AB_tmp1", [cache["Cvir_A"].shape[1], x_B_MOA_ein.shape[0]], np.float64, 'numpy')
            cT_AB_tmp2 = ein.utils.tensor_factory("cT_AB_tmp2", [cache["Cvir_A"].shape[1], Tmo_AB.shape[1]], np.float64, 'numpy')
            cT_AB = ein.utils.tensor_factory("cT_AB", [cache["Cvir_A"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
            plan_matmul_tt.execute(0.0, cT_AB_tmp1, 1.0, cache["Cvir_A"].T, x_B_MOA_ein.T)
            plan_matmul_tt.execute(0.0, cT_AB_tmp2, 1.0, cT_AB_tmp1, Tmo_AB)
            plan_matmul_tt.execute(0.0, cT_AB, 1.0, cT_AB_tmp2.T, cache["Cocc_B"].T)

            cT_BA_tmp1 = ein.utils.tensor_factory("cT_BA_tmp1", [cache["Cvir_B"].shape[1], x_A_MOB_ein.shape[0]], np.float64, 'numpy')
            cT_BA_tmp2 = ein.utils.tensor_factory("cT_BA_tmp2", [cache["Cvir_B"].shape[1], Tmo_AB.shape[0]], np.float64, 'numpy')
            cT_BA = ein.utils.tensor_factory("cT_BA", [cache["Cvir_B"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')
            plan_matmul_tt.execute(0.0, cT_BA_tmp1, 1.0, cache["Cvir_B"].T, x_A_MOB_ein.T)
            plan_matmul_tt.execute(0.0, cT_BA_tmp2, 1.0, cT_BA_tmp1, Tmo_AB.T)
            plan_matmul_tt.execute(0.0, cT_BA, 1.0, cT_BA_tmp2.T, cache["Cocc_A"].T)

            # Compute coupled Sinf energies using vector dot products
            ind_ab_total_tensor_1 = ein.utils.tensor_factory("ind_ab_total_1", [1], np.float64, 'numpy')
            ind_ab_total_tensor_2 = ein.utils.tensor_factory("ind_ab_total_2", [1], np.float64, 'numpy')
            ind_ba_total_tensor_1 = ein.utils.tensor_factory("ind_ba_total_1", [1], np.float64, 'numpy')
            ind_ba_total_tensor_2 = ein.utils.tensor_factory("ind_ba_total_2", [1], np.float64, 'numpy')
            
            plan_vector_dot.execute(0.0, ind_ab_total_tensor_1, 1.0, cT_A, EX_AA_inf)
            plan_vector_dot.execute(0.0, ind_ab_total_tensor_2, 1.0, cT_AB, EX_AB_inf)
            plan_vector_dot.execute(0.0, ind_ba_total_tensor_1, 1.0, cT_B, EX_BB_inf)
            plan_vector_dot.execute(0.0, ind_ba_total_tensor_2, 1.0, cT_BA, EX_BA_inf)

            ind_ab_total = 2.0 * (ind_ab_total_tensor_1[0] + ind_ab_total_tensor_2[0])
            ind_ba_total = 2.0 * (ind_ba_total_tensor_1[0] + ind_ba_total_tensor_2[0])
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
        P_X = ein.utils.tensor_factory(name, [eps_occ.shape[0], eps_vir.shape[0]], np.float64, 'numpy')

        ones_occ = ein.utils.tensor_factory("ones_occ", [eps_occ.shape[0]], np.float64, 'numpy')
        ones_vir = ein.utils.tensor_factory("ones_vir", [eps_vir.shape[0]], np.float64, 'numpy')
        ones_occ.fill(1.0)
        ones_vir.fill(1.0)
        plan_outer = ein.core.compile_plan("ia", "i", "a")
        plan_outer.execute(0.0, P_X, 1.0, eps_occ, ones_vir)
        eps_vir_2D = ein.utils.tensor_factory("eps_vir_2D", [eps_occ.shape[0], eps_vir.shape[0]], np.float64, 'numpy')
        plan_outer.execute(0.0, eps_vir_2D, 1.0, ones_occ, eps_vir)
        ein.core.axpy(-1.0, eps_vir_2D, P_X)
        return P_X

    # Make a preconditioner function
    P_A = setup_P_X(cache['eps_occ_A'], cache['eps_vir_A'])
    P_B = setup_P_X(cache['eps_occ_B'], cache['eps_vir_B'])
    P_A = core.Matrix.from_array(P_A, "P_A")
    P_B = core.Matrix.from_array(P_B, "P_B")

    print(P_A.np)

    # Preconditioner function
    def apply_precon(x_vec, act_mask):
        if act_mask[0]:
            pA = x_vec[0].clone()
            pA.apply_denominator(P_A)
        else:
            pA = False

        if act_mask[1]:
            pB = x_vec[1].clone()
            pB.apply_denominator(P_B)
        else:
            pB = False

        return [pA, pB]

    # Hx function
    def hessian_vec(x_vec, act_mask):
        if act_mask[0]:
            xA = cache["wfn_A"].cphf_Hx([x_vec[0]])[0]
        else:
            xA = False

        if act_mask[1]:
            xB = cache["wfn_B"].cphf_Hx([x_vec[1]])[0]
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

    start_resid = [rhsA.sum_of_squares(), rhsB.sum_of_squares()]

    # print function
    def pfunc(niter, x_vec, r_vec):
        if niter == 0:
            niter = "Guess"
        else:
            niter = "%5d" % niter

        # Compute IndAB
        valA = (r_vec[0].sum_of_squares() / start_resid[0]) ** 0.5
        if valA < conv:
            cA = "*"
        else:
            cA = " "

        # Compute IndBA
        valB = (r_vec[1].sum_of_squares() / start_resid[1]) ** 0.5
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
    vecs, resid = solvers.cg_solver(
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
