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


def setup_fisapt_object(wfn, wfn_A, wfn_B, cache):
    print("CACHE START")
    pp(cache)
    basis_set = wfn.basisset()
    nfrozen_A = wfn_A.basisset().n_frozen_core(
        core.get_global_option("FREEZE_CORE"), wfn_A.molecule()
    )
    nfrozen_B = wfn_B.basisset().n_frozen_core(
        core.get_global_option("FREEZE_CORE"), wfn_B.molecule()
    )
    # Build object
    # Map cache keys (from SAPT(DFT) cache) to FISAPT object keys
    df_matrix_keys = {
        "Cocc_A": "Cocc0A",
        "Cvir_A": "Cvir0A",
        "Cocc_B": "Cocc0B",
        "Cvir_B": "Cvir0B",
        "Locc_A": "Locc0A",
        "Locc_B": "Locc0B",
        "Uocc_A": "Uocc0A",
        "Uocc_B": "Uocc0B",
    }
    matrix_cache = {
        fkey: core.Matrix.from_array(cache[ckey])
        for ckey, fkey in df_matrix_keys.items()
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
    df_vector_keys = {
        "eps_occ_A": "eps_occ0A",
        "eps_vir_A": "eps_vir0A",
        "eps_occ_B": "eps_occ0B",
        "eps_vir_B": "eps_vir0B",
    }
    vector_cache = {
        fkey: core.Vector.from_array(cache[ckey])
        for ckey, fkey in df_vector_keys.items()
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
            np.asarray(matrix_cache["Caocc0A"])[:, nfrozen_A:]
        )
        vector_cache["eps_aocc0A"] = core.Vector.from_array(
            np.asarray(vector_cache["eps_aocc0A"])[nfrozen_A:]
        )
    if nfrozen_B > 0:
        matrix_cache["Caocc0B"] = core.Matrix.from_array(
            np.asarray(matrix_cache["Caocc0B"])[:, nfrozen_B:]
        )
        vector_cache["eps_aocc0B"] = core.Vector.from_array(
            np.asarray(vector_cache["eps_aocc0B"])[nfrozen_B:]
        )
    wfn.set_basisset("DF_BASIS_SAPT", basis_set)
    fisapt = core.FISAPT(wfn)
    # return fisapt, matrix_cache, vector_cache
    pp(fisapt.matrices())
    pp(matrix_cache)
    fisapt.set_matrix(matrix_cache)
    print("vector cache:")
    pp(vector_cache)
    fisapt.set_vector(vector_cache)
    pp(fisapt.matrices())
    return fisapt
