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
# Need to import FISAPT to set fdrop, plot, save_fsapt_variables methods
from . import fisapt_proc
from pprint import pprint as pp


def setup_fisapt_object(wfn, wfn_A, wfn_B, cache, scalars, basis_set=None):
    # Setup FISAPT object
    if basis_set is None:
        basis_set = wfn.basisset()
    wfn.set_basisset("DF_BASIS_SAPT", basis_set)
    fisapt = core.FISAPT(wfn)

    # Used to slice arrays later if frozen core is requested
    nfrozen_A = wfn_A.basisset().n_frozen_core(
        core.get_global_option("FREEZE_CORE"), wfn_A.molecule()
    )
    nfrozen_B = wfn_B.basisset().n_frozen_core(
        core.get_global_option("FREEZE_CORE"), wfn_B.molecule()
    )
    # Gather SAPT(DFT) cache matrices
    # Map cache keys (from SAPT(DFT) cache) to FISAPT object keys
    matrix_keys = {
        "Cocc_A": "Cocc0A",
        "Cvir_A": "Cvir0A",
        "Cocc_B": "Cocc0B",
        "Cvir_B": "Cvir0B",
        "Locc_A": "Locc0A",
        "Locc_B": "Locc0B",
        "Uocc_A": "Uocc0A",
        "Uocc_B": "Uocc0B",
        "Qocc0A": "Qocc0A",
        "Qocc0B": "Qocc0B",
        "Laocc0A": "Laocc0A",
        "Laocc0B": "Laocc0B",
        "Lfocc0A": "Lfocc0A",
        "Lfocc0B": "Lfocc0B",
        "Uaocc0A": "Uaocc0A",
        "Uaocc0B": "Uaocc0B",
        # "Cocc_A": "Caocc0A",
        # "Cocc_B": "Caocc0B",
    }
    matrix_cache = {
        fisapt_key: core.Matrix.from_array(cache[sdft_key])
        for sdft_key, fisapt_key in matrix_keys.items()
    }
    other_keys = [
        "S",
        "D_A",
        "P_A",
        "V_A",
        "J_A",
        "K_A",
        "D_B",
        "P_B",
        "V_B",
        "J_B",
        "J_O",
        "K_B",
        "K_O",
        "J_P_A",
        "J_P_B",
    ]
    for key in other_keys:
        matrix_cache[key] = core.Matrix.from_array(cache[key])
        # matrix_cache[key] = cache[key]

    # Map cache keys (from SAPT(DFT) cache) to FISAPT object keys for vectors
    # Gather SAPT(DFT) cache vectors
    vector_keys = {
        "eps_occ_A": "eps_occ0A",
        "eps_vir_A": "eps_vir0A",
        "eps_occ_B": "eps_occ0B",
        "eps_vir_B": "eps_vir0B",
    }
    vector_cache = {
        fisapt_key: core.Vector.from_array(cache[sdft_key])
        for sdft_key, fisapt_key in vector_keys.items()
    }
    other_vector_keys = [
        "ZA",
        "ZA_orig",
        "ZB",
        "ZB_orig",
        "ZC",
        "ZC_orig",
    ]
    for key in other_vector_keys:
        vector_cache[key] = core.Vector.from_array(cache[key])

    # If frozen core, trim the appropriate matrices and vectors. We can do it with NumPy slicing.
    if nfrozen_A > 0:
        matrix_cache["Caocc0A"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Cocc0A"])[:, nfrozen_A:]
        )
        vector_cache["eps_aocc0A"] = core.Vector.from_array(
            np.asarray(vector_cache["eps_occ0A"])[nfrozen_A:]
        )
        matrix_cache["Uaocc0A"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Uocc0A"])[:, nfrozen_A:]
        )
    else:
        matrix_cache["Caocc0A"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Cocc0A"]).copy()
        )
        vector_cache["eps_aocc0A"] = core.Vector.from_array(
            np.asarray(vector_cache["eps_occ0A"]).copy()
        )
        matrix_cache["Uaocc0A"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Uocc0A"]).copy()
        )
    if nfrozen_B > 0:
        matrix_cache["Caocc0B"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Cocc0B"])[:, nfrozen_B:]
        )
        vector_cache["eps_aocc0B"] = core.Vector.from_array(
            np.asarray(vector_cache["eps_occ0B"])[nfrozen_B:]
        )
        matrix_cache["Uaocc0B"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Uocc0B"])[:, nfrozen_B:]
        )
    else:
        matrix_cache["Caocc0B"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Cocc0B"]).copy()
        )
        vector_cache["eps_aocc0B"] = core.Vector.from_array(
            np.asarray(vector_cache["eps_occ0B"]).copy()
        )
        matrix_cache["Uaocc0B"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Uocc0B"]).copy()
        )
    fisapt.set_matrix(matrix_cache)
    fisapt.set_vector(vector_cache)
    scalar_keys = {
        "Ind20,r (A<-B)": "Ind20,r (A<-B)",
        "Ind20,r (A->B)": "Ind20,r (B<-A)",
        "Ind20,u (A<-B)": "Ind20,u (A<-B)",
        "Ind20,u (A->B)": "Ind20,u (B<-A)",
        "DHF VALUE": "HF",
        "Exch10": "Exch10",
        "Exch10(S^2)": "Exch10(S^2)",
        "Elst10,r": "Elst10,r",
        "Ind20,r": "Ind20,r",
        "Exch-Ind20,r": "Exch-Ind20,r",
    }
    scalar_cache = {
        fisapt_key: scalars[sdft_key] for sdft_key, fisapt_key in scalar_keys.items()
    }
    fisapt.set_scalar(scalar_cache)
    # fisapt.fdrop = fisapt_fdrop
    # fisapt.plot = fisapt_plot
    # fisapt.save_fsapt_variables = fisapt_save_fsapt_variables
    return fisapt
