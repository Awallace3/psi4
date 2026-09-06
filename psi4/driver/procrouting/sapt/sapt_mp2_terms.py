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

import sys
import time

import numpy as np

from psi4 import core

from ... import p4util
from ... import psifiles as psif
from ...p4util.exceptions import *
from .sapt_util import print_sapt_var
from .fdds_response import _compute_fxc, prepare_fdds_hybrid_transform, solve_fdds_response


def df_fdds_dispersion(primary, auxiliary, cache, is_hybrid, x_alpha, leg_points=10, leg_lambda=0.3, do_print=True):

    rho_thresh = core.get_option("SAPT", "SAPT_FDDS_V2_RHO_CUTOFF")
    if do_print:
        core.print_out("\n  ==> E20 Dispersion (CHF FDDS) <== \n\n")
        core.print_out("   Legendre Points:  % 10d\n" % leg_points)
        core.print_out("   Lambda Shift:     % 10.3f\n" % leg_lambda)
        core.print_out("   Fxc Kernal:       % 10s\n" % "ALDA")
        core.print_out("   (P|Fxc|Q) Thresh: % 8.3e\n" % rho_thresh)

    # Build object
    core.timer_on("Form FDDS object")
    df_matrix_keys = ["Cocc_A", "Cvir_A", "Cocc_B", "Cvir_B"]
    fdds_matrix_cache = {key: cache[key] for key in df_matrix_keys}

    df_vector_keys = ["eps_occ_A", "eps_vir_A", "eps_occ_B", "eps_vir_B"]
    fdds_vector_cache = {key: cache[key] for key in df_vector_keys}

    fdds_obj = core.FDDS_Dispersion(primary, auxiliary, fdds_matrix_cache, fdds_vector_cache, is_hybrid)
    core.timer_off("Form FDDS object")

    # Aux Densities
    core.timer_on("Form xc kernel")
    D = fdds_obj.project_densities([cache["D_A"], cache["D_B"]])

    # Temps
    half_Saux = fdds_obj.aux_overlap().clone()
    half_Saux.power(-0.5, 1.e-12)

    halfp_Saux = fdds_obj.aux_overlap().clone()
    halfp_Saux.power(0.5, 1.e-12)

    # Builds potentials
    W_A = fdds_obj.metric().clone()
    W_A.axpy(1.0, _compute_fxc(D[0], half_Saux, halfp_Saux, x_alpha, rho_thresh=rho_thresh))
    W_A = W_A.to_array()

    W_B = fdds_obj.metric().clone()
    W_B.axpy(1.0, _compute_fxc(D[1], half_Saux, halfp_Saux, x_alpha, rho_thresh=rho_thresh))
    W_B = W_B.to_array()

    # Nuke the densities
    del D
    core.timer_off("Form xc kernel")

    metric = fdds_obj.metric().clone().to_array()
    metric_inv = fdds_obj.metric_inv().clone().to_array()

    # Integrate
    core.timer_on("Time Integration")
    core.print_out("\n   => Time Integration <= \n\n")

    val_pack = ("Omega", "Weight", "Disp20,u", "Disp20", "time [s]")
    core.print_out("% 12s % 12s % 14s % 14s % 10s\n" % val_pack)
    start_time = time.time()

    total_uc = 0
    total_c = 0

    # Read R
    if is_hybrid:
        R_A = fdds_obj.R_A().to_array()
        R_B = fdds_obj.R_B().to_array()
        Rtinv_A = prepare_fdds_hybrid_transform(R_A)
        Rtinv_B = prepare_fdds_hybrid_transform(R_B)

    for point, weight in zip(*np.polynomial.legendre.leggauss(leg_points)):

        omega = leg_lambda * (1.0 - point) / (1.0 + point)
        lambda_scale = ((2.0 * leg_lambda) / (point + 1.0)**2)

        responses = {}
        for label, W in (("A", W_A), ("B", W_B)):
            if is_hybrid:
                aux_dict = fdds_obj.form_aux_matrices(label, omega)
                aux_dict = {k: v.to_array() for k, v in aux_dict.items()}
                uncoupled = aux_dict.pop("amp")  # Hybrid amp is already negative.
                aux_dict["Rtinv"] = Rtinv_A if label == "A" else Rtinv_B
            else:
                uncoupled = fdds_obj.form_unc_amplitude(label, omega)
                uncoupled.scale(-1.0)
                uncoupled = uncoupled.to_array()
                aux_dict = None
            responses[label] = solve_fdds_response(
                metric=metric, metric_inv=metric_inv, W=W, uncoupled=uncoupled,
                x_alpha=x_alpha, hybrid=aux_dict)
            del uncoupled, aux_dict

        X_A_uc, X_B_uc = responses["A"].uncoupled, responses["B"].uncoupled
        X_A_coupled, X_B_coupled = responses["A"].coupled, responses["B"].coupled

        # Combine
        tmp_uc = metric_inv.dot(X_A_uc).dot(metric_inv)
        value_uc = np.dot(tmp_uc.flatten(), X_B_uc.flatten())
        del tmp_uc

        tmp_c = metric_inv.dot(X_A_coupled).dot(metric_inv)
        value_c = np.dot(tmp_c.flatten(), X_B_coupled.flatten())

        # Tally
        total_uc += value_uc * weight * lambda_scale
        total_c += value_c * weight * lambda_scale

        if do_print:
            tmp_disp_unc = value_uc * weight * lambda_scale
            tmp_disp = value_c * weight * lambda_scale
            fdds_time = time.time() - start_time

            val_pack = (omega, weight, tmp_disp_unc, tmp_disp, fdds_time)
            core.print_out("% 12.3e % 12.3e % 14.3e % 14.3e %10d\n" % val_pack)

    Disp20_uc = -1.0 / (2.0 * np.pi) * total_uc
    Disp20_c = -1.0 / (2.0 * np.pi) * total_c
    core.timer_off("Time Integration")

    core.print_out("\n")
    core.print_out(print_sapt_var("Disp20,u", Disp20_uc, short=True) + "\n")
    core.print_out(print_sapt_var("Disp20", Disp20_c, short=True) + "\n")

    return {"Disp20,FDDS (unc)": Disp20_uc, "Disp20": Disp20_c}


def df_mp2_fisapt_dispersion(wfn, primary, auxiliary, cache, nfrozen_A, nfrozen_B, do_print=True):

    if do_print:
        core.print_out("\n  ==> E20 Dispersion (MP2) <== \n\n")

    # Build object
    df_matrix_keys = ["Cocc_A", "Cvir_A", "Cocc_B", "Cvir_B"]
    df_mfisapt_keys = ["Caocc0A", "Cvir0A", "Caocc0B", "Cvir0B"]
    matrix_cache = {fkey: cache[ckey] for ckey, fkey in zip(df_matrix_keys, df_mfisapt_keys)}

    other_keys = ["S", "D_A", "P_A", "V_A", "J_A", "K_A", "D_B", "P_B", "V_B", "J_B", "K_B", "K_O"]
    for key in other_keys:
        matrix_cache[key] = cache[key]

    # matrix_cache["K_O"] = matrix_cache["K_O"].transpose()

    df_vector_keys = ["eps_occ_A", "eps_vir_A", "eps_occ_B", "eps_vir_B"]
    df_vfisapt_keys = ["eps_aocc0A", "eps_vir0A", "eps_aocc0B", "eps_vir0B"]
    vector_cache = {fkey: cache[ckey] for ckey, fkey in zip(df_vector_keys, df_vfisapt_keys)}

    # If frozen core, trim the appropriate matrices and vectors. We can do it with NumPy slicing.
    if nfrozen_A > 0:
        matrix_cache["Caocc0A"] = core.Matrix.from_array(np.asarray(matrix_cache["Caocc0A"])[:,nfrozen_A:])
        vector_cache["eps_aocc0A"] = core.Vector.from_array(np.asarray(vector_cache["eps_aocc0A"])[nfrozen_A:])
    if nfrozen_B > 0:
        matrix_cache["Caocc0B"] = core.Matrix.from_array(np.asarray(matrix_cache["Caocc0B"])[:,nfrozen_B:])
        vector_cache["eps_aocc0B"] = core.Vector.from_array(np.asarray(vector_cache["eps_aocc0B"])[nfrozen_B:])

    wfn.set_basisset("DF_BASIS_SAPT", auxiliary)
    fisapt = core.FISAPT(wfn)

    # Compute!
    fisapt.disp(matrix_cache, vector_cache, False)
    scalars = fisapt.scalars()

    core.print_out("\n")
    core.print_out(print_sapt_var("Disp20 (MP2)", scalars["Disp20"], short=True) + "\n")
    core.print_out(print_sapt_var("Exch-Disp20,u", scalars["Exch-Disp20"], short=True) + "\n")

    ret = {}
    ret["Exch-Disp20,u"] = scalars["Exch-Disp20"]
    ret["Disp20,u"] = scalars["Disp20"]

    if core.get_option("SAPT", "DO_DISP_EXCH_SINF"):
        fisapt.sinf_disp(matrix_cache, vector_cache, True)
        scalars = fisapt.scalars()

    return ret


def df_mp2_sapt_dispersion(dimer_wfn, wfn_A, wfn_B, primary_basis, aux_basis, cache, do_print=True):

    if do_print:
        core.print_out("\n  ==> E20 Dispersion (MP2) <== \n\n")

    optstash = p4util.OptionsState(['SAPT', 'SAPT0_E10'], ['SAPT', 'SAPT0_E20IND'], ['SAPT', 'SAPT0_E20DISP'],
                                   ['SAPT', 'SAPT_QUIET'])

    core.set_local_option("SAPT", "SAPT0_E10", False)
    core.set_local_option("SAPT", "SAPT0_E20IND", False)
    core.set_local_option("SAPT", "SAPT0_E20DISP", True)
    core.set_local_option("SAPT", "SAPT_QUIET", True)

    if core.get_option('SCF', 'REFERENCE') == 'RHF':
        core.IO.change_file_namespace(psif.PSIF_SAPT_MONOMERA, 'monomerA', 'dimer')
        core.IO.change_file_namespace(psif.PSIF_SAPT_MONOMERB, 'monomerB', 'dimer')

    core.IO.set_default_namespace('dimer')

    dimer_wfn.set_basisset("DF_BASIS_SAPT", aux_basis)
    dimer_wfn.set_basisset("DF_BASIS_ELST", aux_basis)
    e_sapt = core.sapt(dimer_wfn, wfn_A, wfn_B)

    optstash.restore()

    svars = dimer_wfn.variables()

    core.print_out("\n")
    core.print_out(print_sapt_var("Disp20 (MP2)", svars["E DISP20"], short=True) + "\n")
    core.print_out(print_sapt_var("Exch-Disp20,u", svars["E EXCH-DISP20"], short=True) + "\n")

    ret = {}
    ret["Exch-Disp20,u"] = svars["E EXCH-DISP20"]
    ret["Disp20,u"] = svars["E DISP20"]
    return ret
