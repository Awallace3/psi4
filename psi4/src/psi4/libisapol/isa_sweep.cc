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
        samples.points = grid.points; samples.weights = grid.weights; samples.density_sites = grid.density_sites;
        samples.previous = old.atomic_coefficients[a];
        samples.shape_sum.assign(grid.points.size(),0.0);
        int clipped = 0;
        for (int b : grid.shape_sites) {
            auto w = gaussians[b].sample(grid.points,tails[b],apply_tail[b]);
            if (static_cast<size_t>(b) == a && !(apply_tail[b] && tails[b].defined)) {
                auto values = shapes_[b].evaluate(grid.points);
                for (size_t p = 0; p < grid.points.size(); ++p) {
                    double raw = 0.0;
                    for (int k = 0; k < shapes_[b].nfunction(); ++k)
                        raw += values->get(p,k)*old.shape_coefficients[b][k];
                    if (raw < 0.0) ++clipped;
                }
            }
            for (size_t p = 0; p < grid.points.size(); ++p) {
                samples.shape_sum[p] += w[p];
                sweep_require(std::isfinite(samples.shape_sum[p]), "Nonfinite sweep shape sum");
            }
            if (static_cast<size_t>(b) == a) samples.shape = std::move(w);
        }
        auto fit = providers_[a].fit(samples,options);
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
