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
    basis_set = wfn.basisset()
    nfrozen_A = wfn_A.basisset().n_frozen_core(core.get_global_option("FREEZE_CORE"),wfn_A.molecule())
    nfrozen_B = wfn_B.basisset().n_frozen_core(core.get_global_option("FREEZE_CORE"),wfn_B.molecule())
    # Build object
    df_matrix_keys = ["Cocc_A", "Cvir_A", "Cocc_B", "Cvir_B"]
    df_mfisapt_keys = ["Caocc0A", "Cvir0A", "Caocc0B", "Cvir0B"]
    matrix_cache = {
        fkey: cache[ckey] for ckey, fkey in zip(df_matrix_keys, df_mfisapt_keys)
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
        "K_B",
        "K_O",
    ]
    for key in other_keys:
        matrix_cache[key] = cache[key]

    df_vector_keys = ["eps_occ_A", "eps_vir_A", "eps_occ_B", "eps_vir_B"]
    df_vfisapt_keys = ["eps_aocc0A", "eps_vir0A", "eps_aocc0B", "eps_vir0B"]
    vector_cache = {
        fkey: cache[ckey] for ckey, fkey in zip(df_vector_keys, df_vfisapt_keys)
    }

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
    pp(fisapt.matrices())
    return fisapt
