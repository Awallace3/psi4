/*
 * Psi4: an open-source quantum chemistry software package
 * Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ISA-A numerical conventions follow CamCASP by Alston J. Misquitta and
 * Anthony J. Stone. Independently implemented from the fitting equations;
 * see SPEC.md and CamCASP stockholder.F90 / num_integrals.F90.
 */
#ifndef PSI4_LIBISAPOL_ISA_FIT_H
#define PSI4_LIBISAPOL_ISA_FIT_H

#include <memory>
#include <vector>

namespace psi {
class Matrix;
namespace isapol {

/// Active settings for ONE frozen update, not an iteration/activation controller.
struct IsaAFitOptions {
    double w_eps = 0.0;
    bool s_block_only = true;
    double damping = 0.0;
    double positive_lambda = 0.0;
    double positive_max_alpha = 0.2;
    bool positive_auto = true;
    double density_cutoff = 1.0e-36;
};

/// All arrays use a common, caller-specified point and basis-function order.
/// Samples are already neighbor-screened; shape samples are already tail-corrected
/// or clamped by the selected upstream policy. Signed samples are not clipped here.
/// This initial boundary accepts primitive (uncontracted) atomic bases only. The
/// caller must decontract before sampling; basis values cannot establish this fact.
struct IsaAFitData {
    std::vector<double> weights, density, shape, shape_sum, radius_squared;
    std::shared_ptr<Matrix> basis_values;  ///< (npoint, nfunction), values including normalization
    std::shared_ptr<Matrix> overlap;       ///< Analytic W-Eps-weighted metric, BEFORE damping/ridge
    std::vector<double> previous;          ///< Old full atomic expansion coefficients
    std::vector<int> angular_momenta;      ///< Per function; s functions need not be contiguous
    std::vector<double> exponents;         ///< Positive primitive exponent for EVERY function, including non-s
};

struct IsaAFitResult {
    std::shared_ptr<Matrix> metric;        ///< Modified metric, not the overwritten LU factors
    std::shared_ptr<Matrix> rhs;           ///< (nfunction, 1)
    std::shared_ptr<Matrix> coefficients;  ///< (nfunction, 1), no renormalization
    double population = 0.0;              ///< Integral of rho * shape / shape_sum, before damping
    double relative_residual = 0.0;       ///< ||S D - T||inf / (||S||inf ||D||inf + ||T||inf)
    int excluded_points = 0;              ///< abs(shape_sum) <= density_cutoff
};

/// Input objects are never mutated or retained. Requires a C1, symmetric metric
/// constructed with the SAME active w_eps/s_block_only settings as the RHS.
IsaAFitResult isa_a_fit_step(const IsaAFitData& data, const IsaAFitOptions& options);

/// CamCASP W/RHO overlap-angle diagnostic. The metric must be UNWEIGHTED.
/// Scalar multiples (even opposite signs) have zero angle; this is not a charge test.
double isa_overlap_change(const std::vector<double>& current, const std::vector<double>& previous,
                          const std::shared_ptr<Matrix>& overlap);

}  // namespace isapol
}  // namespace psi
#endif
