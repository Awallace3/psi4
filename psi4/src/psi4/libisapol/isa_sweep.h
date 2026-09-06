/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_ISA_SWEEP_H
#define PSI4_LIBISAPOL_ISA_SWEEP_H
#include "explicit_basis.h"
#include "isa_fit.h"
#include "isa_shape.h"
namespace psi { namespace isapol {
/// Explicit per-atom quadrature. Shape sites index sweep atoms; density sites
/// index the independent molecular AUX centres. Both are zero-based, no padding.
struct IsaNoTailGrid {
    std::vector<std::array<double,3>> points;
    std::vector<double> weights;
    std::vector<int> density_sites, shape_sites;
};
/// Old state may have independently supplied atomic and shape expansions (e.g.
/// initialization); no implicit projection or normalization occurs on input.
struct IsaSweepState {
    std::vector<std::vector<double>> atomic_coefficients, shape_coefficients;
};
struct IsaNoTailSweepResult {
    IsaSweepState next;
    std::vector<IsaAFitResult> fits;
    std::vector<int> clipped_shape_points;  ///< Selected old shape on its own grid, before clipping
};
/// One synchronous sweep, not an activation or convergence controller.
/// Every atom sees the same frozen old state. run() is explicitly no-tail;
/// run_with_tails() applies the supplied per-site policy. Raw output stays signed.
class IsaASweep {
   public:
    IsaASweep(const std::vector<IsaExplicitBasis>& atomic,
                   const std::vector<IsaExplicitBasis>& shape,
                   const std::vector<std::vector<int>>& shell_maps,
                   const IsaFixedDensity& density);
    IsaNoTailSweepResult run(const IsaSweepState& old, const std::vector<IsaNoTailGrid>& grids,
                             const IsaAFitOptions& options) const;
    IsaNoTailSweepResult run_with_tails(const IsaSweepState& old, const std::vector<IsaNoTailGrid>& grids,
                                       const std::vector<IsaExponentialTail>& tails,
                                       const std::vector<bool>& apply_tail, const IsaAFitOptions& options) const;
   private:
    std::vector<IsaExplicitBasis> shapes_;
    std::vector<IsaAFitProvider> providers_;
    std::vector<IsaShapeMap> maps_;
    std::vector<int> atomic_sizes_;
};
using IsaNoTailSweep = IsaASweep;  ///< Backward-compatible developer name; run() remains no-tail.
} }
#endif
