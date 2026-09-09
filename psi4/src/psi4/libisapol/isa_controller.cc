/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "isa_controller.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void controller_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void controller_vector(const std::vector<double>& v, size_t n) {
    controller_require(v.size() == n, "Controller state vector dimension mismatch");
    for (double x : v) controller_require(std::isfinite(x), "Controller state values must be finite");
}
}
IsaAController::IsaAController(const std::vector<IsaExplicitBasis>& atomic,
        const std::vector<IsaExplicitBasis>& shape, const std::vector<std::vector<int>>& shell_maps,
        const IsaFixedDensity& density, const std::vector<IsaNoTailGrid>& grids,
        const IsaAControllerOptions& options)
    : sweep_(atomic,shape,shell_maps,density), shapes_(shape), grids_(grids), options_(options) {
    const size_t n = shape.size();
    controller_require(grids.size() == n, "Controller grid count mismatch");
    for (double x : {options.convergence,options.w_eps_activation,options.positive_activation,
                     options.tail_activation,options.fit.w_eps,options.fit.positive_lambda,
                     options.fit.damping,options.fit.positive_max_alpha,options.fit.density_cutoff})
        controller_require(std::isfinite(x) && x >= 0, "Controller thresholds/settings must be finite and nonnegative");
    controller_require(options.convergence > 0, "Controller convergence threshold must be positive");
    controller_require(std::isfinite(options.mixing) && options.mixing >= 0 && options.mixing <= 1,
                       "Controller mixing must be in [0,1]");
    controller_require(options.max_iterations > 0 && options.mixing_skip >= 0 && options.tail_iteration_limit >= 0,
                       "Controller iteration limits must be valid");
    if (options_.tail_allowed.empty()) options_.tail_allowed.assign(n,true);
    if (options_.convergence_included.empty()) options_.convergence_included.assign(n,true);
    controller_require(options_.tail_allowed.size() == n && options_.convergence_included.size() == n,
                       "Controller site mask dimension mismatch");
    controller_require(std::any_of(options_.convergence_included.begin(),options_.convergence_included.end(),
                                  [](bool v) { return v; }), "Controller MaxDelta requires an included site");
    if (options.fix_tails || !options.tail_cutoffs.empty()) {
        controller_require(options.tail_cutoffs.size() == n, "Controller requires explicit per-site tail cutoffs");
        for (double r : options.tail_cutoffs)
            controller_require(std::isfinite(r) && r >= 1e-8, "Controller tail cutoffs must be finite and at least 1e-8");
    }
    for (size_t a = 0; a < n; ++a) {
        atomic_sizes_.push_back(atomic[a].nfunction());
        overlaps_.push_back(shape[a].overlap());
    }
    prepared_ = sweep_.prepare(grids_,options_.cache_max_bytes);
}
std::shared_ptr<IsaAController> IsaAController::without_prepared_cache() const {
    auto snapshot = std::make_shared<IsaAController>(*this);
    snapshot->prepared_.clear();
    return snapshot;
}
void IsaAController::validate(const IsaAControllerState& state) const {
    const size_t n = shapes_.size();
    controller_require(state.coefficients.atomic_coefficients.size() == n &&
                       state.coefficients.shape_coefficients.size() == n && state.tails.size() == n,
                       "Controller state atom count mismatch");
    controller_require(state.iteration >= 0 && state.iteration <= options_.max_iterations, "Invalid controller iteration");
    controller_require(std::isfinite(state.max_delta) && state.max_delta >= 0, "Invalid controller MaxDelta");
    controller_require((state.active_w_eps == 0 || state.active_w_eps == options_.fit.w_eps) &&
                       (state.active_positive_lambda == 0 || state.active_positive_lambda == options_.fit.positive_lambda),
                       "Controller active settings disagree with configuration");
    controller_vector(state.saved_shape_charges,n);
    for (size_t a = 0; a < n; ++a) {
        controller_vector(state.coefficients.atomic_coefficients[a],atomic_sizes_[a]);
        controller_vector(state.coefficients.shape_coefficients[a],shapes_[a].nfunction());
        IsaGaussianShape(shapes_[a],state.coefficients.shape_coefficients[a]).sample({},state.tails[a],false);
    }
}
IsaAControllerState IsaAController::initialize(const IsaSweepState& coefficients) const {
    IsaAControllerState state;
    state.coefficients = coefficients;
    state.tails.resize(shapes_.size());
    state.saved_shape_charges.assign(shapes_.size(),0.0);
    state.active_w_eps = options_.w_eps_activation > 0 ? 0 : options_.fit.w_eps;
    state.active_positive_lambda = options_.positive_activation > 0 ? 0 : options_.fit.positive_lambda;
    validate(state);
    for (size_t a = 0; a < shapes_.size(); ++a)
        state.saved_shape_charges[a] = IsaGaussianShape(shapes_[a],coefficients.shape_coefficients[a]).exterior_charge(0);
    return state;
}
IsaAControllerStep IsaAController::step(const IsaAControllerState& old) const {
    validate(old);
    controller_require(!old.converged, "Controller is already converged");
    controller_require(old.iteration < options_.max_iterations, "Controller maximum iterations reached");
    IsaAFitOptions active = options_.fit;
    active.w_eps = old.active_w_eps; active.positive_lambda = old.active_positive_lambda;
    std::vector<bool> apply(shapes_.size(),false);
    for (size_t a = 0; a < shapes_.size(); ++a) apply[a] = old.apply_tails && options_.tail_allowed[a];
    IsaAControllerStep result;
    result.raw_sweep = sweep_.run_prepared(old.coefficients,grids_,old.tails,apply,active,prepared_);
    result.next = old;
    result.next.iteration = old.iteration+1;
    result.next.coefficients = result.raw_sweep.next;
    result.next.max_delta = 0;
    result.next.converged = true;
    for (size_t a = 0; a < shapes_.size(); ++a) {
        const auto& raw = result.raw_sweep.next.shape_coefficients[a];
        const auto& previous = old.coefficients.shape_coefficients[a];
        const double delta = isa_overlap_change(raw,previous,overlaps_[a]);
        const bool converged = delta < options_.convergence;
        result.deltas.push_back(delta); result.atom_converged.push_back(converged);
        result.shape_charges.push_back(IsaGaussianShape(shapes_[a],raw).exterior_charge(0));
        result.next.saved_shape_charges[a] = result.shape_charges.back();
        if (options_.convergence_included[a]) {
            result.next.converged = result.next.converged && converged;
            result.next.max_delta = std::max(result.next.max_delta,delta);
        }
        // Convergence/charge are deliberately PRE-mixing; full D is never mixed here.
        if (!converged && options_.mixing > 0 && result.next.iteration > options_.mixing_skip)
            for (size_t k = 0; k < raw.size(); ++k)
                result.next.coefficients.shape_coefficients[a][k] = (1-options_.mixing)*raw[k]+options_.mixing*previous[k];
    }
    if (options_.w_eps_activation > 0)
        result.next.active_w_eps = result.next.max_delta <= options_.w_eps_activation ? options_.fit.w_eps : 0;
    if (options_.positive_activation > 0)
        result.next.active_positive_lambda = result.next.max_delta <= options_.positive_activation ? options_.fit.positive_lambda : 0;
    result.next.apply_tails = options_.fix_tails &&
        (result.next.max_delta <= options_.tail_activation || result.next.iteration > options_.tail_iteration_limit);
    // Source lag: fit NEXT tails from OLD Gaussian shapes, before committing the
    // next mixed shapes. Do not silently replace this with fitting the new state.
    if (!options_.tail_cutoffs.empty())
        for (size_t a = 0; a < shapes_.size(); ++a) {
            auto tail = IsaGaussianShape(shapes_[a],old.coefficients.shape_coefficients[a]).fit_tail(options_.tail_cutoffs[a],old.tails[a]);
            result.next.tails[a] = tail.tail;
            result.tail_fits.push_back(std::move(tail));
        }
    return result;
}
IsaAControllerResult IsaAController::run(const IsaAControllerState& initial) const {
    validate(initial);
    IsaAControllerResult result;
    result.state = initial;
    while (!result.state.converged && result.state.iteration < options_.max_iterations) {
        auto iteration = step(result.state);
        result.state = iteration.next;
        result.history.push_back(std::move(iteration));
    }
    result.termination = result.state.converged ? "converged" : "max_iterations";
    return result;
}
} }
