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

#ifndef PSI4_SRC_PSI4_LIBXDM_BECKE_ROUSSEL_H
#define PSI4_SRC_PSI4_LIBXDM_BECKE_ROUSSEL_H

#include <algorithm>
#include <cmath>

namespace psi {
namespace xdm {

/// Evaluate the Becke-Roussel exchange-hole function and its derivative.
/// f(x) = x * exp(-2x/3) / (x - 2) - rhs
/// df(x) = 2/3 * (2x - x^2 - 3) / (x - 2)^2 * exp(-2x/3)
inline void xfuncs(double x, double rhs, double& f, double& df) {
    double expo23 = std::exp(-2.0 / 3.0 * x);
    double xm2 = x - 2.0;
    f = x * expo23 / xm2 - rhs;
    df = 2.0 / 3.0 * (2.0 * x - x * x - 3.0) / (xm2 * xm2) * expo23;
}

/// Compute the exchange-hole curvature Q from density properties.
/// Q = (laplacian - 2 * dsigs) / 6
/// dsigs = tau - 0.25 * |grad_rho|^2 / rho (positive-definite KE density)
///
/// @param rho          spin density rho_sigma
/// @param grad_rho_sq  |grad rho_sigma|^2
/// @param tau          kinetic energy density for spin sigma
/// @param laplacian    Laplacian of rho_sigma
/// @return exchange-hole curvature Q
inline double exchange_hole_curvature(double rho, double grad_rho_sq, double tau, double laplacian) {
    double dsigs = tau - 0.25 * grad_rho_sq / std::max(rho, 1.0e-30);
    return (laplacian - 2.0 * dsigs) / 6.0;
}

/// Solve the Becke-Roussel exchange-hole equation for the dipole moment parameter b.
///
/// Based on: Becke & Roussel, Phys. Rev. A 39, 3761 (1989)
///
/// Solves: f(x) = x * exp(-2x/3) / (x - 2) - rhs = 0
/// where rhs = (2/3) * (pi * rho / hnorm)^(2/3) * rho / Q
///
/// Then: alpha = (8*pi*rho/(hnorm*exp(-x)))^(1/3), b = x / alpha
///
/// @param rho    spin density
/// @param quad   exchange-hole curvature Q
/// @param hnorm  normalization factor (1.0 for single spin channel)
/// @return exchange-hole dipole moment b
double bhole(double rho, double quad, double hnorm = 1.0);

}  // namespace xdm
}  // namespace psi

#endif
