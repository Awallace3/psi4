/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "isa_sweep.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void sweep_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void sweep_vector(const std::vector<double>& v, size_t size) {
    sweep_require(v.size() == size, "Sweep coefficient dimension mismatch");
    for (double c : v) sweep_require(std::isfinite(c), "Sweep coefficients must be finite");
}
}
IsaASweep::IsaASweep(const std::vector<IsaExplicitBasis>& atomic,
                             const std::vector<IsaExplicitBasis>& shape,
                             const std::vector<std::vector<int>>& shell_maps,
                             const IsaFixedDensity& density) : shapes_(shape) {
    sweep_require(!atomic.empty() && atomic.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
                  "Sweep requires a nonempty bounded atom set");
    sweep_require(atomic.size() == shape.size() && atomic.size() == shell_maps.size(),
                  "Sweep basis/map count mismatch");
    for (size_t a = 0; a < atomic.size(); ++a) {
        providers_.emplace_back(atomic[a],density);
        maps_.emplace_back(atomic[a],shape[a],shell_maps[a]);
        atomic_sizes_.push_back(atomic[a].nfunction());
    }
}
IsaNoTailSweepResult IsaASweep::run(const IsaSweepState& old, const std::vector<IsaNoTailGrid>& grids,
                                       const IsaAFitOptions& options) const {
    return run_with_tails(old,grids,std::vector<IsaExponentialTail>(shapes_.size()),
                          std::vector<bool>(shapes_.size(),false),options);
}
IsaNoTailSweepResult IsaASweep::run_with_tails(const IsaSweepState& old, const std::vector<IsaNoTailGrid>& grids,
                                             const std::vector<IsaExponentialTail>& tails,
                                             const std::vector<bool>& apply_tail, const IsaAFitOptions& options) const {
    return run_prepared(old,grids,tails,apply_tail,options,{});
}
std::vector<IsaAFitData> IsaASweep::prepare(const std::vector<IsaNoTailGrid>& grids, size_t max_bytes) const {
    // Checked numerical-payload admission: basis + twelve point vectors + two
    // metrics and metadata per atom, plus the density block's 8 MiB scratch.
    // Preparation temporarily owns copied xyz/weights/shape/shape_sum (6*np)
    // together with assembled weights/shape/shape_sum/radii/density (5*np).
    // Twelve vectors conservatively cover that eleven-vector construction peak.
    // Container overhead, owned input grids and shape reuse are separate; this
    // is not a process RSS cap. No partial persistent cache on failed admission.
    constexpr size_t budget = 192 * 1024 * 1024, scratch = 8 * 1024 * 1024;
    const size_t available = std::min(budget,max_bytes);
    if (available < scratch) return {};
    size_t remaining = (available-scratch)/sizeof(double);
    for (size_t a = 0; a < grids.size(); ++a) {
        const size_t nf = atomic_sizes_[a], np = grids[a].points.size();
        if (nf > remaining / nf / 2) return {};
        remaining -= 2*nf*nf;
        if (nf > remaining/3) return {};
        remaining -= 3*nf;
        if (np > remaining/(nf+12)) return {};
        remaining -= np*(nf+12);
    }
    std::vector<IsaAFitData> prepared;
    for (size_t a = 0; a < grids.size(); ++a) {
        IsaAFitSamples samples;
        samples.points = grids[a].points;
        samples.weights = grids[a].weights;
        samples.density_sites = grids[a].density_sites;
        samples.shape.assign(samples.points.size(),0.0);
        samples.shape_sum.assign(samples.points.size(),0.0);
        samples.previous.assign(atomic_sizes_[a],0.0);
        auto data = providers_[a].assemble(samples,IsaAFitOptions{});
        // Only immutable inputs survive construction. Overlap is keyed at use
        // by actual w_eps/s_block_only, never by an activation assumption.
        std::vector<double>().swap(data.shape);
        std::vector<double>().swap(data.shape_sum);
        std::vector<double>().swap(data.previous);
        prepared.push_back(std::move(data));
    }
    return prepared;
}
IsaNoTailSweepResult IsaASweep::run_prepared(const IsaSweepState& old, const std::vector<IsaNoTailGrid>& grids,
        const std::vector<IsaExponentialTail>& tails, const std::vector<bool>& apply_tail,
        const IsaAFitOptions& options, const std::vector<IsaAFitData>& prepared) const {
    const size_t n = shapes_.size();
    sweep_require(tails.size() == n && apply_tail.size() == n, "Sweep tail policy count mismatch");
    sweep_require(grids.size() == n && old.atomic_coefficients.size() == n && old.shape_coefficients.size() == n,
                  "Sweep grid/state atom count mismatch");
    // Validate all old state before computing any atom. Nothing is mutated or
    // partially published if a later atom fails its sampling or solve checks.
    std::vector<IsaGaussianShape> gaussians;
    for (size_t a = 0; a < n; ++a) {
        sweep_vector(old.atomic_coefficients[a],atomic_sizes_[a]);
        sweep_vector(old.shape_coefficients[a],shapes_[a].nfunction());
        gaussians.emplace_back(shapes_[a],old.shape_coefficients[a]);
    }
    // Exact ordered coordinate equality only. Weights/screens are NOT part of
    // shape collocation, and neighbour sums below retain each grid's order.
    std::vector<size_t> representative(n);
    for (size_t a = 0; a < n; ++a) {
        representative[a] = a;
        for (size_t b = 0; b < a; ++b)
            if (grids[a].points == grids[b].points) { representative[a] = representative[b]; break; }
    }
    struct ShapeSamples { size_t grid, site; std::vector<double> values; int clipped; };
    std::vector<ShapeSamples> shared;
    size_t shape_budget = 32 * 1024 * 1024 / sizeof(double);
    IsaNoTailSweepResult result;
    for (size_t a = 0; a < n; ++a) {
        const auto& grid = grids[a];
        sweep_require(!grid.points.empty() && grid.weights.size() == grid.points.size(), "Sweep grid dimension mismatch");
        std::vector<bool> seen(n,false);
        for (int b : grid.shape_sites) {
            sweep_require(b >= 0 && static_cast<size_t>(b) < n, "Sweep shape neighbour out of range");
            sweep_require(!seen[b], "Duplicate sweep shape neighbour");
            seen[b] = true;
        }
        sweep_require(seen[a], "Sweep shape neighbours must include the selected atom");
        IsaAFitSamples samples;
        if (prepared.empty()) {
            samples.points = grid.points; samples.weights = grid.weights; samples.density_sites = grid.density_sites;
        }
        samples.previous = old.atomic_coefficients[a];
        samples.shape_sum.assign(grid.points.size(),0.0);
        int clipped = 0;
        for (int b : grid.shape_sites) {
            const ShapeSamples* cached = nullptr;
            for (const auto& entry : shared)
                if (entry.grid == representative[a] && entry.site == static_cast<size_t>(b)) { cached = &entry; break; }
            std::vector<double> temporary;
            int negative = 0;
            if (!cached) {
                temporary = gaussians[b].sample_counted(grid.points,tails[b],apply_tail[b],&negative);
                if (temporary.size() <= shape_budget) {
                    shape_budget -= temporary.size();
                    shared.push_back({representative[a],static_cast<size_t>(b),std::move(temporary),negative});
                    cached = &shared.back();
                }
            }
            const auto& w = cached ? cached->values : temporary;
            for (size_t p = 0; p < grid.points.size(); ++p) {
                samples.shape_sum[p] += w[p];
                sweep_require(std::isfinite(samples.shape_sum[p]), "Nonfinite sweep shape sum");
            }
            if (static_cast<size_t>(b) == a) {
                samples.shape = w;
                clipped = cached ? cached->clipped : negative;
            }
        }
        IsaAFitResult fit;
        if (prepared.empty()) {
            fit = providers_[a].fit(samples,options);
        } else {
            auto data = prepared[a]; // Private matrices shared read-only; vectors owned per call.
            data.shape = std::move(samples.shape); data.shape_sum = std::move(samples.shape_sum);
            data.previous = std::move(samples.previous);
            if (options.w_eps != 0.0)
                data.overlap = providers_[a].atomic_.overlap(options.w_eps,options.s_block_only);
            fit = isa_a_fit_step(data,options);
        }
        std::vector<double> coefficients(atomic_sizes_[a]);
        for (int k = 0; k < atomic_sizes_[a]; ++k) coefficients[k] = fit.coefficients->get(k,0);
        result.next.shape_coefficients.push_back(maps_[a].project(coefficients));
        result.next.atomic_coefficients.push_back(std::move(coefficients));
        result.fits.push_back(std::move(fit));
        result.clipped_shape_points.push_back(clipped);
    }
    return result;
}
} }
