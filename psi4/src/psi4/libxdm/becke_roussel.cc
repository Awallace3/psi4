/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2024 The Psi4 Developers.
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

#include "becke_roussel.h"

#include <cmath>
#include <stdexcept>

namespace psi {
namespace xdm {

static constexpr double PI = 3.141592653589793;
static constexpr double THIRD = 1.0 / 3.0;
static constexpr double TWO_THIRDS = 2.0 / 3.0;

double bhole(double rho, double quad, double hnorm) {
    // Compute the right-hand side of the BR equation
    double rhs = TWO_THIRDS * std::pow(PI * rho / hnorm, TWO_THIRDS) * rho / quad;

    // Bracket the solution via bisection-like search
    double x0 = 2.0;
    double shift = 1.0;
    double x = x0;
    double f, df;

    if (rhs < 0.0) {
        // Search x < 2
        for (int i = 0; i < 16; i++) {
            x = x0 - shift;
            xfuncs(x, rhs, f, df);
            if (f < 0.0) goto newton;
            shift *= 0.1;
        }
        throw std::runtime_error("bhole: Newton algorithm fails to initialize (rhs < 0)");
    } else {
        // Search x > 2
        for (int i = 0; i < 16; i++) {
            x = x0 + shift;
            xfuncs(x, rhs, f, df);
            if (f > 0.0) goto newton;
            shift *= 0.1;
        }
        throw std::runtime_error("bhole: Newton algorithm fails to initialize (rhs > 0)");
    }

newton:
    // Newton-Raphson iteration
    for (int i = 0; i < 100; i++) {
        xfuncs(x, rhs, f, df);
        double x1 = x - f / df;
        if (std::abs(x1 - x) < 1.0e-10) {
            x = x1;
            // Compute b from converged x
            double expo = std::exp(-x);
            double prefac = rho / expo;
            double alf = std::cbrt(8.0 * PI * prefac / hnorm);
            return x / alf;
        }
        x = x1;
    }
    throw std::runtime_error("bhole: Newton algorithm fails to converge");
}

}  // namespace xdm
}  // namespace psi
