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
    print("Cocc_A.shape:", cache["Cocc_A"].shape)
    print(cache["Cocc_A"])
    print(wfn_A.Ca_subset("AO", "OCC").np)
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
    cache["D_A"] = ein.utils.tensor_factory("D_A", [cache["Cocc_A"].shape[0], cache["Cocc_A"].shape[0]], np.float64, 'numpy')

    # Should be fine, but only fills in half the matrix... is it because of the symmetry? How can I fix this to use plan_matmul_tT?
    # plan_matmul_tT = ein.core.compile_plan("ij", "ik", "jk")
    plan_matmul_tt = ein.core.compile_plan("ij", "ik", "kj")
    plan_matmul_tt.execute(0.0, cache['D_A'], 1.0, cache['Cocc_A'], cache['Cocc_A'].T)

    cache["D_B"] = ein.utils.tensor_factory("D_B", [cache["Cocc_B"].shape[0], cache["Cocc_B"].shape[0]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, cache['D_B'], 1.0, cache['Cocc_B'], cache['Cocc_B'].T)
    print("D_B:", cache['D_B'].shape)
    print(cache['D_B'])
    print((wfn_B.Ca_subset("AO", "OCC").np @ wfn_B.Ca_subset("AO", "OCC").np.T))

    assert np.allclose(cache["D_A"], (wfn_A.Ca_subset("AO", "OCC").np @ wfn_A.Ca_subset("AO", "OCC").np.T))
    assert np.allclose(cache["D_B"], (wfn_B.Ca_subset("AO", "OCC").np @ wfn_B.Ca_subset("AO", "OCC").np.T))

    cache["P_A"] = ein.utils.tensor_factory("P_A", [cache["Cvir_A"].shape[0], cache["Cvir_A"].shape[0]], np.float64, 'numpy')
    cache["P_B"] = ein.utils.tensor_factory("P_B", [cache["Cvir_B"].shape[0], cache["Cvir_B"].shape[0]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, cache['P_A'], 1.0, cache['Cvir_A'], cache['Cvir_A'].T)
    plan_matmul_tt.execute(0.0, cache['P_B'], 1.0, cache['Cvir_B'], cache['Cvir_B'].T)

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
    D_B__S = ein.utils.tensor_factory("D_B__S", [cache["D_B"].shape[0], cache["S"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, D_B__S, 1.0, cache["D_B"], cache["S"])
    plan_matmul_tt.execute(0.0, C_O_A, 1.0, D_B__S, cache["Cocc_A"])

    C_O_A_matrix = core.Matrix.from_array(C_O_A)
    # Q: How can I convert jk to use RuntimeTensorD?
    print(type(jk), dir(jk))
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
    """
    Computes the E10 electrostatics from a build_sapt_jk_cache datacache.
    """
    if do_print:
        core.print_out("\n  ==> E10 Electostatics <== \n\n")

    # ELST
    Elst10 = 0.0
    plan_vector_dot = ein.core.compile_plan("", "i", "i")
    Elst10_tmp = ein.utils.tensor_factory("Elst10_tmp", [1], np.float64, 'numpy')
    plan_vector_dot.execute(0.0, Elst10_tmp, 1.0, cache["D_A"], cache["V_B"])
    Elst10 += 2.0 * Elst10_tmp[0]
    plan_vector_dot.execute(0.0, Elst10_tmp, 1.0, cache["D_B"], cache["V_A"])
    Elst10 += 2.0 * Elst10_tmp[0]
    plan_vector_dot.execute(0.0, Elst10_tmp, 1.0, cache["D_B"], cache["J_A"])
    Elst10 += 4.0 * Elst10_tmp[0]
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
    """
    Computes the E10 exchange (S^2 and S^inf) from a build_sapt_jk_cache datacache.
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
    plan_matmul_tt.execute(0.0, SA, 1.0, cache["Cocc_A"].T, cache["S"])
    SAB = ein.utils.tensor_factory("SAB", [cache["Cocc_A"].shape[1], cache["Cocc_B"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, SAB, 1.0, SA, cache["Cocc_B"])
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
        core.print_out("\n  ==> E20 Induction <== \n\n")

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

    jk.C_clear()
    DB_S = ein.utils.tensor_factory("DB_S", [D_B.shape[0], S.shape[1]], np.float64, 'numpy')
    DB_S_CA = ein.utils.tensor_factory("DB_S_CA", [D_B.shape[0], cache["Cocc_A"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt = ein.core.compile_plan("ij", "ik", "kj")
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

    D_A_S = ein.utils.tensor_factory("D_A_S", [D_A.shape[0], S.shape[1]], np.float64, 'numpy')
    D_A_S_DB = ein.utils.tensor_factory("D_A_S_CB", [D_A.shape[0], cache["D_B"].shape[1]], np.float64, 'numpy')
    DA_S_DB_CA = ein.utils.tensor_factory("D_A_S_CB", [D_A.shape[0], cache["Cocc_A"].shape[1]], np.float64, 'numpy')
    plan_matmul_tt.execute(0.0, D_A_S, 1.0, D_A, S)
    plan_matmul_tt.execute(0.0, D_A_S_DB, 1.0, D_A_S, D_B)
    plan_matmul_tt.execute(0.0, DA_S_DB_CA, 1.0, D_A_S_DB, cache["Cocc_A"])
    jk.C_left_add(core.Matrix.from_array(DA_S_DB_CA))
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
    EX_A.scale(-1.0)
    EX_A.axpy(-2.0, J_O)
    EX_A.axpy(1.0, K_O)
    EX_A.axpy(2.0, J_P_B)

    EX_A.axpy(-1.0, core.Matrix.chain_dot(S, D_B, V_A))
    EX_A.axpy(-2.0, core.Matrix.chain_dot(S, D_B, J_A))
    EX_A.axpy(1.0, core.Matrix.chain_dot(S, D_B, K_A))
    EX_A.axpy(1.0, core.Matrix.chain_dot(S, D_B, S, D_A, V_B))
    EX_A.axpy(2.0, core.Matrix.chain_dot(S, D_B, S, D_A, J_B))
    EX_A.axpy(1.0, core.Matrix.chain_dot(S, D_B, V_A, D_B, S))
    EX_A.axpy(2.0, core.Matrix.chain_dot(S, D_B, J_A, D_B, S))
    EX_A.axpy(-1.0, core.Matrix.chain_dot(S, D_B, K_O, trans=[False, False, True]))

    EX_A.axpy(-1.0, core.Matrix.chain_dot(V_B, D_B, S))
    EX_A.axpy(-2.0, core.Matrix.chain_dot(J_B, D_B, S))
    EX_A.axpy(1.0, core.Matrix.chain_dot(K_B, D_B, S))
    EX_A.axpy(1.0, core.Matrix.chain_dot(V_B, D_A, S, D_B, S))
    EX_A.axpy(2.0, core.Matrix.chain_dot(J_B, D_A, S, D_B, S))
    EX_A.axpy(-1.0, core.Matrix.chain_dot(K_O, D_B, S))

    EX_A = core.Matrix.chain_dot(
        cache["Cocc_A"], EX_A, cache["Cvir_A"], trans=[True, False, False]
    )

    # Exch-Ind Potential B
    EX_B = K_A.clone()
    EX_B.scale(-1.0)
    EX_B.axpy(-2.0, J_O)
    EX_B.axpy(1.0, K_O.transpose())
    EX_B.axpy(2.0, J_P_A)

    EX_B.axpy(-1.0, core.Matrix.chain_dot(S, D_A, V_B))
    EX_B.axpy(-2.0, core.Matrix.chain_dot(S, D_A, J_B))
    EX_B.axpy(1.0, core.Matrix.chain_dot(S, D_A, K_B))
    EX_B.axpy(1.0, core.Matrix.chain_dot(S, D_A, S, D_B, V_A))
    EX_B.axpy(2.0, core.Matrix.chain_dot(S, D_A, S, D_B, J_A))
    EX_B.axpy(1.0, core.Matrix.chain_dot(S, D_A, V_B, D_A, S))
    EX_B.axpy(2.0, core.Matrix.chain_dot(S, D_A, J_B, D_A, S))
    EX_B.axpy(-1.0, core.Matrix.chain_dot(S, D_A, K_O))

    EX_B.axpy(-1.0, core.Matrix.chain_dot(V_A, D_A, S))
    EX_B.axpy(-2.0, core.Matrix.chain_dot(J_A, D_A, S))
    EX_B.axpy(1.0, core.Matrix.chain_dot(K_A, D_A, S))
    EX_B.axpy(1.0, core.Matrix.chain_dot(V_A, D_B, S, D_A, S))
    EX_B.axpy(2.0, core.Matrix.chain_dot(J_A, D_B, S, D_A, S))
    EX_B.axpy(-1.0, core.Matrix.chain_dot(K_O, D_A, S, trans=[True, False, False]))

    EX_B = core.Matrix.chain_dot(
        cache["Cocc_B"], EX_B, cache["Cvir_B"], trans=[True, False, False]
    )

    # Build electrostatic potenital
    w_A = cache["V_A"].clone()
    w_A.axpy(2.0, cache["J_A"])

    w_B = cache["V_B"].clone()
    w_B.axpy(2.0, cache["J_B"])

    w_B_MOA = core.triplet(cache["Cocc_A"], w_B, cache["Cvir_A"], True, False, False)
    w_A_MOB = core.triplet(cache["Cocc_B"], w_A, cache["Cvir_B"], True, False, False)

    # Do uncoupled
    core.print_out("   => Uncoupled Induction <= \n\n")
    unc_x_B_MOA = w_B_MOA.clone()
    unc_x_B_MOA.np[:] /= cache["eps_occ_A"].np.reshape(-1, 1) - cache["eps_vir_A"].np
    unc_x_A_MOB = w_A_MOB.clone()
    unc_x_A_MOB.np[:] /= cache["eps_occ_B"].np.reshape(-1, 1) - cache["eps_vir_B"].np

    unc_ind_ab = 2.0 * unc_x_B_MOA.vector_dot(w_B_MOA)
    unc_ind_ba = 2.0 * unc_x_A_MOB.vector_dot(w_A_MOB)
    unc_indexch_ab = 2.0 * unc_x_B_MOA.vector_dot(EX_A)
    unc_indexch_ba = 2.0 * unc_x_A_MOB.vector_dot(EX_B)

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
            # core.set_variable(name, ret[name])
            core.print_out(print_sapt_var(name, ret[name], short=True))
            core.print_out("\n")

    # Exch-Ind without S^2
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

    # Do coupled
    if do_response:
        core.print_out("\n   => Coupled Induction <= \n\n")

        cphf_r_convergence = core.get_option("SAPT", "CPHF_R_CONVERGENCE")

        x_B_MOA, x_A_MOB = _sapt_cpscf_solve(
            cache, jk, w_B_MOA, w_A_MOB, 20, cphf_r_convergence, sapt_jk_B=sapt_jk_B
        )

        ind_ab = 2.0 * x_B_MOA.vector_dot(w_B_MOA)
        ind_ba = 2.0 * x_A_MOB.vector_dot(w_A_MOB)
        indexch_ab = 2.0 * x_B_MOA.vector_dot(EX_A)
        indexch_ba = 2.0 * x_A_MOB.vector_dot(EX_B)

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

                # core.set_variable(name, ret[name])
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

    cache["wfn_A"].set_jk(jk)
    if sapt_jk_B:
        cache["wfn_B"].set_jk(sapt_jk_B)
    else:
        cache["wfn_B"].set_jk(jk)

    # Make a preconditioner function
    P_A = core.Matrix(cache["eps_occ_A"].shape[0], cache["eps_vir_A"].shape[0])
    P_A.np[:] = cache["eps_occ_A"].np.reshape(-1, 1) - cache["eps_vir_A"].np

    P_B = core.Matrix(cache["eps_occ_B"].shape[0], cache["eps_vir_B"].shape[0])
    P_B.np[:] = cache["eps_occ_B"].np.reshape(-1, 1) - cache["eps_vir_B"].np

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
