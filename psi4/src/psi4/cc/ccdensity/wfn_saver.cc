/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2022 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

/*! \file
    \ingroup CCDENSITY
    \brief Save variables to wavefunction.
*/

#include "ccdensity.h"
#include "MOInfo.h"
#define EXTERN
#include "globals.h"

namespace psi {
namespace ccdensity {

void scalar_saver_ground(SharedWavefunction wfn, struct TD_Params *S, const std::string suffix, double val) {
    auto target_sym = moinfo.sym ^ S->irrep;
    auto idx_num = S->root + static_cast<int>(S->irrep == 0);
    auto varname = "CC ROOT 0 (" + moinfo.labels[moinfo.sym] + ") -> ROOT " + std::to_string(idx_num) + " (" + moinfo.labels[target_sym] + ") " + suffix;
    wfn->set_scalar_variable(varname, val);
    if (params.wfn == "EOM_CCSD") {
        auto varname = "CCSD ROOT 0 (" + moinfo.labels[moinfo.sym] + ") -> ROOT " + std::to_string(idx_num) + " (" + moinfo.labels[target_sym] + ") " + suffix;
        wfn->set_scalar_variable(varname, val);
    } else if (params.wfn == "EOM_CC2") {
        auto varname = "CC2 ROOT 0 (" + moinfo.labels[moinfo.sym] + ") -> ROOT " + std::to_string(idx_num) + " (" + moinfo.labels[target_sym] + ") "  + suffix;
        wfn->set_scalar_variable(varname, val);
    } else {
        throw PSIEXCEPTION("Unknown wfn type");
    }
}

void scalar_saver_excited(SharedWavefunction wfn, struct TD_Params *S, struct TD_Params *U, const std::string suffix, double val) {
    auto S_sym = moinfo.sym ^ S->irrep;
    auto U_sym = moinfo.sym ^ U->irrep;
    auto S_idx = S->root + static_cast<int>(S->irrep == 0);
    auto U_idx = U->root + static_cast<int>(U->irrep == 0);
    auto varname = "CC ROOT " + std::to_string(S_idx) + " (" + moinfo.labels[S_sym] + ") -> ROOT " + std::to_string(U_idx) + " (" + moinfo.labels[U_sym] + ") " + suffix;
    wfn->set_scalar_variable(varname, val);
    if (params.wfn == "EOM_CCSD") {
        auto varname = "CCSD ROOT " + std::to_string(S_idx) + " (" + moinfo.labels[S_sym] + ") -> ROOT " + std::to_string(U_idx) + " (" + moinfo.labels[U_sym] + ") " + suffix;
        wfn->set_scalar_variable(varname, val);
    } else if (params.wfn == "EOM_CC2") {
        auto varname = "CC2 ROOT " + std::to_string(S_idx) + " (" + moinfo.labels[S_sym] + ") -> ROOT " + std::to_string(U_idx) + " (" + moinfo.labels[U_sym] + ") " + suffix;
        wfn->set_scalar_variable(varname, val);
    } else {
        throw PSIEXCEPTION("Unknown wfn type");
    }
}

}
}