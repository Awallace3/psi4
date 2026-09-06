/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_ISA_CONTROLLER_H
#define PSI4_LIBISAPOL_ISA_CONTROLLER_H
#include "isa_sweep.h"
namespace psi { namespace isapol {
/// Explicit ordinary-A, W-convergence controller. No DIIS, symmetry, decoupled
/// subiterations or self-consistent postconvergence tail loop are implemented.
struct IsaAControllerOptions {
    IsaAFitOptions fit;
    double convergence = 1.e-9;
    double w_eps_activation = 1.e-5, positive_activation = 1.e-5, tail_activation = 1.e-5;
    double mixing = 0.0;
    int mixing_skip = 20, tail_iteration_limit = 20, max_iterations = 120;
    bool fix_tails = true;
    std::vector<double> tail_cutoffs;  ///< Explicit bohr radii, not inferred Slater tables
    std::vector<bool> tail_allowed, convergence_included;  ///< Empty means all sites
};
/// Restart cursor for the SAME controller settings, bases, density and grids.
/// saved_shape_charges follow reference pre-mixing bookkeeping, not necessarily
/// integrals of the stored mixed Gaussian coefficients.
struct IsaAControllerState {
    IsaSweepState coefficients;
    std::vector<IsaExponentialTail> tails;
    std::vector<double> saved_shape_charges;
    int iteration = 0;
    double active_w_eps = 0.0, active_positive_lambda = 0.0, max_delta = 0.0;
    bool apply_tails = false, converged = false;
};
struct IsaAControllerStep {
    IsaAControllerState next;
    IsaNoTailSweepResult raw_sweep;
    std::vector<double> deltas, shape_charges;
    std::vector<bool> atom_converged;
    std::vector<IsaTailFitResult> tail_fits;
};
struct IsaAControllerResult {
    IsaAControllerState state;
    std::vector<IsaAControllerStep> history;
    std::string termination;
};
class IsaAController {
   public:
    IsaAController(const std::vector<IsaExplicitBasis>& atomic,
                   const std::vector<IsaExplicitBasis>& shape,
                   const std::vector<std::vector<int>>& shell_maps, const IsaFixedDensity& density,
                   const std::vector<IsaNoTailGrid>& grids, const IsaAControllerOptions& options);
    IsaAControllerState initialize(const IsaSweepState& coefficients) const;
    IsaAControllerStep step(const IsaAControllerState& old) const;
    IsaAControllerResult run(const IsaAControllerState& initial) const;
   private:
    void validate(const IsaAControllerState& state) const;
    IsaASweep sweep_;
    std::vector<IsaExplicitBasis> shapes_;
    std::vector<int> atomic_sizes_;
    std::vector<std::shared_ptr<Matrix>> overlaps_;
    std::vector<IsaNoTailGrid> grids_;
    IsaAControllerOptions options_;
};
} }
#endif
