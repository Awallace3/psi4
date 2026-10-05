/*
 * Psi4: an open-source quantum chemistry software package
 * Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ISA-A numerical conventions follow CamCASP by Alston J. Misquitta and
 * Anthony J. Stone. Independently implemented from the fitting equations;
 * see CamCASP stockholder.F90 / num_integrals.F90.
 */
#ifndef PSI4_LIBISAPOL_ISA_FIT_H
#define PSI4_LIBISAPOL_ISA_FIT_H

#include "explicit_basis.h"

#include <array>
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

/// Validated shape-shell -> AtomAux-shell map. Indices are zero-based, no padding.
/// Each shape shell must exactly match its unique s-shell target (centre, ordered
/// exponents and effective coefficients). An explicit subset/permutation is allowed.
class IsaShapeMap {
   public:
    IsaShapeMap(const IsaExplicitBasis& atomic, const IsaExplicitBasis& shape,
                const std::vector<int>& shell_map);
    std::vector<int> function_indices() const { return columns_; }
    /// Select raw coefficients only: no normalization, DIIS, mixing or tail policy.
    std::vector<double> project(const std::vector<double>& atomic_coefficients) const;
   private:
    int atomic_nfunction_;
    std::vector<int> columns_;
};

/// Supplied molecular AUX expansion, NOT AO density or a native Drho-C fit.
/// The caller owns provenance (e.g. Drho-C). No clipping or charge rescaling.
class IsaFixedDensity {
   public:
    IsaFixedDensity(const IsaExplicitBasis& basis, const std::vector<double>& coefficients);
    /// Explicit zero-based unique active sites, no padding. Empty means all screened out.
    std::vector<double> evaluate(const std::vector<std::array<double, 3>>& points,
                                 const std::vector<int>& sites) const;
   private:
    IsaExplicitBasis basis_;
    std::vector<double> coefficients_;
};

/// Caller-owned sampling policy: shapes are already screened and tail-corrected.
/// Density sites are unique zero-based active sites for this entire point batch.
struct IsaAFitSamples {
    std::vector<std::array<double, 3>> points;
    std::vector<double> weights, shape, shape_sum, previous;
    std::vector<int> density_sites;
};

/// Input objects are never mutated or retained. Requires a C1, symmetric metric
/// constructed with the SAME active w_eps/s_block_only settings as the RHS.
IsaAFitResult isa_a_fit_step(const IsaAFitData& data, const IsaAFitOptions& options);

/// Owned explicit-input provider, not a native basis/DF recipe or ISA controller.
class IsaAFitProvider {
   public:
    IsaAFitProvider(const IsaExplicitBasis& atomic, const IsaFixedDensity& density);
    /// Fresh inspectable data; use the same W-Eps/s-block settings in a later solve.
    IsaAFitData assemble(const IsaAFitSamples& samples, const IsaAFitOptions& options) const;
    /// Assembly and frozen solve with identical options; no retained mutable samples.
    IsaAFitResult fit(const IsaAFitSamples& samples, const IsaAFitOptions& options) const;
   private:
    friend class IsaASweep;
    IsaExplicitBasis atomic_;
    IsaFixedDensity density_;
    std::array<double, 3> centre_;
    std::vector<int> angular_;
    std::vector<double> exponents_;
};

/// CamCASP W/RHO overlap-angle diagnostic. The metric must be UNWEIGHTED.
/// Scalar multiples (even opposite signs) have zero angle; this is not a charge test.
double isa_overlap_change(const std::vector<double>& current, const std::vector<double>& previous,
                          const std::shared_ptr<Matrix>& overlap);

}  // namespace isapol
}  // namespace psi
#endif
