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

#ifndef PSI4_LIBISAPOL_CASIMIR_GRID_H
#define PSI4_LIBISAPOL_CASIMIR_GRID_H

#include <vector>

namespace psi {
namespace isapol {

/// Most imaginary frequencies CamCASP will accept (`MAXF`, casimir.f90:40).
constexpr int kMaxCasimirFrequencies = 10;

/// Default quadrature scale, CamCASP's `omega0` (casimir.f90:82).  This is the same
/// parameter that `quadrature.f90` calls `Beta`, and it is 0.3 only in the standalone
/// `casimir` program; the ISA-Pol protocols set `SET QUAD / Beta 0.5`, so the Psi4
/// keyword default is kIsaPolOmega0 below.  See SPEC.md 9.1.
constexpr double kCasimirOmega0 = 0.3;

/// Quadrature scale used by the ISA-Pol protocols (`methods/isa-pol-from-isa-A`).
constexpr double kIsaPolOmega0 = 0.5;

/// Gauss-Legendre quadrature on the imaginary frequency axis, for the
/// Casimir-Polder integral
///
///     C_6 = (3/pi) \int_0^inf alpha_A(i w) alpha_B(i w) dw.
///
/// This is a transliteration of `SUBROUTINE frequencies` (casimir.f90:437-462).
/// The integral is mapped onto t in [-1, 1] by w = omega0 (1 + t) / (1 - t), and
/// evaluated with an `n_freq`-point Gauss-Legendre rule whose roots CamCASP stores
/// *squared*, in the packed table `rlow`, for orders 2, 4, ..., 18.  Only the
/// positive half of each rule is tabulated; the two halves are unpacked here in
/// CamCASP's order, so `omega(1) < ... < omega(n_freq)`.
///
/// Indices are 1-based, matching CamCASP, with index 0 reserved for the static
/// point w = 0.  That point is not part of the quadrature -- it carries no weight
/// -- but the response code evaluates it anyway (CamCASP's `SKIP` directive
/// controls whether it appears in a `.pol` file), so it is convenient to have it
/// in the same array.
///
/// `n_freq` must be even.  CamCASP fills only i = 1 ... n_freq/2 and its mirror
/// image, so an odd count would silently leave the middle frequency uninitialised;
/// we reject it instead.
class CasimirGrid {
   public:
    /// @param n_freq number of imaginary frequencies; even, in [2, kMaxCasimirFrequencies]
    /// @param omega0 quadrature scale in hartree
    explicit CasimirGrid(int n_freq, double omega0 = kCasimirOmega0);

    int n_freq() const { return n_freq_; }
    double omega0() const { return omega0_; }

    /// Imaginary frequency w_k in hartree, k = 0 ... n_freq.  `omega(0)` is 0.
    double omega(int k) const;

    /// (1 - t_k)^2 for the mapped root t_k, CamCASP's `tm1sq`.  Zero at k = 0.
    double tm1sq(int k) const;

    /// Raw Gauss-Legendre weight of point k on t in [-1, 1].  Zero at k = 0.
    double weight(int k) const;

    /// -w_k^2, the argument the FDDS propagator wants (CamCASP's `wsq`).
    double wsq(int k) const;

    /// Weight of point k in the Casimir-Polder integral itself,
    /// `weight(k) * omega0 / (pi * tm1sq(k))`: the Gauss-Legendre weight, times
    /// the Jacobian 2 omega0 / (1 - t)^2 of the mapping, times the 1 / (2 pi) of
    /// the Casimir-Polder formula (casimir.f90:418-421).  With this, CamCASP's
    /// `cpint` is the bare sum
    ///
    ///     sum_k cp_weight(k) alpha_A(i w_k) alpha_B(i w_k);
    ///
    /// the remaining rank-dependent factors that turn `cpint` into a dispersion
    /// coefficient are the recoupling coefficients of SPEC.md 9.3, not part of
    /// the quadrature.  Zero at k = 0.
    double cp_weight(int k) const;

    /// All n_freq + 1 frequencies, index 0 being the static point.
    const std::vector<double>& omegas() const { return omega_; }

   private:
    int n_freq_;
    double omega0_;
    std::vector<double> omega_;   ///< [0 ... n_freq]
    std::vector<double> tm1sq_;   ///< [0 ... n_freq]
    std::vector<double> weight_;  ///< [0 ... n_freq]

    void check(int k) const;
};

}  // namespace isapol
}  // namespace psi

#endif
