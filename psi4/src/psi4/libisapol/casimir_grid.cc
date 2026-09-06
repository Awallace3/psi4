/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
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

/*
 * libisapol: ISA partitioning, distributed polarizabilities and dispersion
 * coefficients, ported with permission from CamCASP 6.0 by Alston J. Misquitta
 * and Anthony J. Stone (http://gitlab.com/anthonyjstone/camcasp).  Numerical
 * conventions follow the CamCASP source, referenced inline by file and line.
 *
 * See SPEC.md in this directory for the full specification.
 */

#include "casimir_grid.h"

#include "psi4/libpsi4util/exception.h"

#include <cmath>
#include <string>

namespace psi {
namespace isapol {

namespace {

/// Squares of the positive Gauss-Legendre roots on [-1, 1] for orders
/// 2, 4, 6, ..., 18, packed end to end: order 2n starts at n(n-1)/2.
/// Transcribed verbatim from casimir.f90:43-58.
const double rlow[45] = {
    0.33333333333333e+00, 0.74155574714581e+00, 0.11558710999705e+00, 0.86949939491826e+00, 0.43719785275109e+00,
    0.56939115967007e-01, 0.92215660849206e+00, 0.63467747623464e+00, 0.27618431387246e+00, 0.33648268067507e-01,
    0.94849392628837e+00, 0.74833462838728e+00, 0.46159736149627e+00, 0.18783156765245e+00, 0.22163568807218e-01,
    0.96346127870282e+00, 0.81742801326687e+00, 0.59275012773154e+00, 0.34494237942742e+00, 0.13530001165525e+00,
    0.15683406607401e-01, 0.97275575129749e+00, 0.86199133320339e+00, 0.68426201565315e+00, 0.47237153700448e+00,
    0.26548115726894e+00, 0.10183270400277e+00, 0.11675871940146e-01, 0.97891421016235e+00, 0.89222197421380e+00,
    0.74931737854740e+00, 0.57063582016217e+00, 0.38177105339712e+00, 0.20977936861551e+00, 0.79300559811486e-01,
    0.90273770256471e-02, 0.98320148322563e+00, 0.91359942257427e+00, 0.79673916319752e+00, 0.64594166107702e+00,
    0.47843096553757e+00, 0.31334338332122e+00, 0.16953901896600e+00, 0.63446670693112e-01, 0.71868028362264e-02};

/// Gauss-Legendre weights matching `rlow`, casimir.f90:59-74.
const double wlow[45] = {
    0.10000000000000e+01, 0.34785484513745e+00, 0.65214515486255e+00, 0.17132449237917e+00, 0.36076157304814e+00,
    0.46791393457269e+00, 0.10122853629038e+00, 0.22238103445337e+00, 0.31370664587789e+00, 0.36268378337836e+00,
    0.66671344308689e-01, 0.14945134915058e+00, 0.21908636251598e+00, 0.26926671931000e+00, 0.29552422471475e+00,
    0.47175336386513e-01, 0.10693932599532e+00, 0.16007832854335e+00, 0.20316742672307e+00, 0.23349253653835e+00,
    0.24914704581340e+00, 0.35119460331752e-01, 0.80158087159761e-01, 0.12151857068790e+00, 0.15720316715819e+00,
    0.18553839747794e+00, 0.20519846372130e+00, 0.21526385346316e+00, 0.27152459411755e-01, 0.62253523938648e-01,
    0.95158511682493e-01, 0.12462897125553e+00, 0.14959598881658e+00, 0.16915651939500e+00, 0.18260341504492e+00,
    0.18945061045507e+00, 0.21616013526485e-01, 0.49714548894970e-01, 0.76425730254889e-01, 0.10094204410629e+00,
    0.12255520671148e+00, 0.14064291467065e+00, 0.15468467512627e+00, 0.16427648374583e+00, 0.16914238296314e+00};

/// The literal CamCASP writes into `cpint` (casimir.f90:413).  This is the
/// correctly rounded double nearest pi, i.e. the same value as M_PI, but it is
/// spelled out here so the provenance is not a matter of trusting <cmath>.
constexpr double kPi = 3.141592653589793;

}  // namespace

CasimirGrid::CasimirGrid(int n_freq, double omega0) : n_freq_(n_freq), omega0_(omega0) {
    if (n_freq < 2 || n_freq > kMaxCasimirFrequencies) {
        throw PSIEXCEPTION("libisapol: CasimirGrid needs 2 to " + std::to_string(kMaxCasimirFrequencies) +
                           " frequencies, got " + std::to_string(n_freq));
    }
    if (n_freq % 2 != 0) {
        throw PSIEXCEPTION("libisapol: CasimirGrid needs an even number of frequencies, got " +
                           std::to_string(n_freq) + "; CamCASP tabulates only even-order Gauss-Legendre rules");
    }

    omega_.assign(n_freq + 1, 0.0);
    tm1sq_.assign(n_freq + 1, 0.0);
    weight_.assign(n_freq + 1, 0.0);

    // casimir.f90:443-456, index for index.  omega(0) = 0 is the static point,
    // already in place from the assign() above.
    const int halfn = n_freq / 2;
    const int base = halfn * (halfn - 1) / 2;
    for (int i = 1; i <= halfn; ++i) {
        const double t = std::sqrt(rlow[base + i - 1]);
        // Written to match Fortran's left-to-right evaluation of
        // `omega0*(1d0+t)/(1d0-t)`; the module is compiled -ffp-contract=off so
        // the compiler may not reassociate these into an FMA either.
        omega_[n_freq - i + 1] = omega0 * (1.0 + t) / (1.0 - t);
        tm1sq_[n_freq - i + 1] = (1.0 - t) * (1.0 - t);
        omega_[i] = omega0 * (1.0 - t) / (1.0 + t);
        tm1sq_[i] = (1.0 + t) * (1.0 + t);
        weight_[i] = wlow[base + i - 1];
        weight_[n_freq - i + 1] = wlow[base + i - 1];
    }
}

void CasimirGrid::check(int k) const {
    if (k < 0 || k > n_freq_) {
        throw PSIEXCEPTION("libisapol: CasimirGrid frequency index " + std::to_string(k) + " out of range [0, " +
                           std::to_string(n_freq_) + "]");
    }
}

double CasimirGrid::omega(int k) const {
    check(k);
    return omega_[k];
}

double CasimirGrid::tm1sq(int k) const {
    check(k);
    return tm1sq_[k];
}

double CasimirGrid::weight(int k) const {
    check(k);
    return weight_[k];
}

double CasimirGrid::wsq(int k) const {
    check(k);
    return -omega_[k] * omega_[k];
}

double CasimirGrid::cp_weight(int k) const {
    check(k);
    if (k == 0) return 0.0;
    // casimir.f90:419-420.
    return weight_[k] * (omega0_ / (kPi * tm1sq_[k]));
}

}  // namespace isapol
}  // namespace psi
