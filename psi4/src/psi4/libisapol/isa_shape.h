/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_ISA_SHAPE_H
#define PSI4_LIBISAPOL_ISA_SHAPE_H
#include "explicit_basis.h"
#include <string>
namespace psi { namespace isapol {
/// Explicit Func-1 tail. Signed amplitude is allowed; no positivity is implied.
struct IsaExponentialTail {
    bool defined = false;
    double amplitude = 0.0, exponent = 0.0, cutoff = 0.0;
};
struct IsaTailFitResult {
    IsaExponentialTail tail;
    bool used_previous_exponent = false;
    double gaussian_tail_charge = 0.0;
    double ionization_potential = 0.0;
    std::string status;
};
/// Owned co-centred effective s-Gaussian expansion; no normalization or clipping.
class IsaGaussianShape {
   public:
    IsaGaussianShape(const IsaExplicitBasis& basis, const std::vector<double>& coefficients);
    double value(double radius) const;
    double exterior_charge(double radius) const;
    /// Func-1, Fit-Type 3 only. Uses centered finite difference step 1e-8
    /// via translated z-axis Cartesian points and ordered effective shell samples.
    /// Invalid slope uses a previous valid 1<b<4 exponent or returns undefined.
    /// No historical stale saved-A sign gate or undefined IP assignment is reproduced.
    IsaTailFitResult fit_tail(double cutoff, const IsaExponentialTail& previous) const;
    /// Active defined tail replaces only r>cutoff; interior is NOT clipped.
    /// Disabled/undefined tail selects the no-tail max(w,0) branch.
    std::vector<double> sample(const std::vector<std::array<double,3>>& points,
                               const IsaExponentialTail& tail, bool apply_tail) const;
   private:
    friend class IsaASweep;
    std::vector<double> sample_counted(const std::vector<std::array<double,3>>& points,
                                     const IsaExponentialTail& tail, bool apply_tail, int* clipped) const;
    double value_squared(double r2) const;
    IsaExplicitBasis basis_;
    std::vector<double> coefficients_;
    std::array<double,3> centre_;
    std::vector<double> exponents_, amplitudes_;
};
} }
#endif
