/*
 * Psi4: an open-source quantum chemistry software package
 * Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ISA-A numerical conventions follow CamCASP by Alston J. Misquitta and
 * Anthony J. Stone. See stockholder.F90:2985-3125,3971-4094 and
 * num_integrals.F90:622-870. No CamCASP source is linked into this module.
 */
#include "isa_fit.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "psi4/libmints/matrix.h"
#include "psi4/libqt/qt.h"

namespace psi {
namespace isapol {
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("ISA-A fit: " + message);
}
void finite_vector(const std::vector<double>& v, size_t n, const std::string& name) {
    require(v.size() == n, name + " has the wrong size");
    for (double x : v) require(std::isfinite(x), name + " contains a nonfinite value");
}
void finite_matrix(const std::shared_ptr<Matrix>& m, int rows, int cols, const std::string& name) {
    require(m && m->nirrep() == 1 && m->symmetry() == 0, name + " must be a C1 matrix");
    require(m->nrow() == rows && m->ncol() == cols, name + " has the wrong dimensions");
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            require(std::isfinite(m->get(i, j)), name + " contains a nonfinite value");
}
double symmetric_metric(const std::shared_ptr<Matrix>& m, int n) {
    finite_matrix(m, n, n, "overlap");
    double scale = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) scale = std::max(scale, std::abs(m->get(i, j)));
    require(scale > 0.0, "overlap diagonal must be positive");
    for (int i = 0; i < n; ++i) {
        require(m->get(i, i) > 0.0, "overlap diagonal must be positive");
        for (int j = 0; j < i; ++j)
            require(std::abs(m->get(i, j) / scale - m->get(j, i) / scale) <= 1.e-12,
                    "overlap must be symmetric");
    }
    return scale;
}
}  // namespace

IsaAFitResult isa_a_fit_step(const IsaAFitData& d, const IsaAFitOptions& o) {
    const size_t np = d.weights.size(), nf = d.previous.size();
    require(np > 0 && nf > 0 && np <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                nf <= static_cast<size_t>(std::numeric_limits<int>::max()), "invalid point/function count");
    const int n = static_cast<int>(nf);
    finite_vector(d.weights, np, "weights");
    finite_vector(d.density, np, "density");
    finite_vector(d.shape, np, "shape");
    finite_vector(d.shape_sum, np, "shape_sum");
    finite_vector(d.radius_squared, np, "radius_squared");
    finite_vector(d.previous, nf, "previous");
    finite_vector(d.exponents, nf, "exponents");
    require(d.angular_momenta.size() == nf, "angular_momenta has the wrong size");
    for (size_t k = 0; k < nf; ++k) {
        require(d.angular_momenta[k] >= 0, "angular momenta must be nonnegative");
        require(d.exponents[k] > 0.0, "primitive exponents must be positive");
    }
    for (double r2 : d.radius_squared) require(r2 >= 0.0, "radius_squared must be nonnegative");
    for (double x : {o.w_eps, o.damping, o.positive_lambda, o.positive_max_alpha, o.density_cutoff})
        require(std::isfinite(x) && x >= 0.0, "options must be finite and nonnegative");
    finite_matrix(d.basis_values, static_cast<int>(np), n, "basis_values");
    symmetric_metric(d.overlap, n);

    IsaAFitResult out;
    out.metric = d.overlap->clone();
    out.rhs = std::make_shared<Matrix>("ISA-A RHS", n, 1);
    // The weighted overlap is supplied by the integral provider. Do not weight it
    // a second time here. CamCASP modifies only its s/s block and diffuse s diagonal.
    for (int k = 0; k < n; ++k) {
        if (d.angular_momenta[k] != 0) continue;
        for (int l = 0; l < n; ++l)
            if (d.angular_momenta[l] == 0)
                out.metric->set(k, l, d.overlap->get(k, l) * (1.0 + o.damping));
        if (d.exponents[k] <= o.positive_max_alpha && (!o.positive_auto || d.previous[k] < 0.0))
            out.metric->add(k, k, o.positive_lambda);
    }

    auto phi = d.basis_values->pointer();
    auto rhs = out.rhs->pointer();
    for (size_t p = 0; p < np; ++p) {
        double rho_a = 0.0;
        if (std::abs(d.shape_sum[p]) > o.density_cutoff) {
            rho_a = d.density[p] * (d.shape[p] / d.shape_sum[p]);
            out.population += rho_a * d.weights[p];
        } else {
            ++out.excluded_points;
        }
        const double tail_weight = o.w_eps > 0.0 ? std::exp(std::min(o.w_eps * d.radius_squared[p], 230.0)) : 1.0;
        for (int k = 0; k < n; ++k) {
            double term;
            if (d.angular_momenta[k] == 0) {
                // Damping survives the denominator cutoff, just as in the source.
                term = d.weights[p] * phi[p][k] * (rho_a + o.damping * d.shape[p]) * tail_weight;
            } else {
                term = d.weights[p] * phi[p][k] * rho_a;
                if (!o.s_block_only) term *= tail_weight;
            }
            rhs[k][0] += term;
        }
    }
    finite_matrix(out.metric, n, n, "modified metric");
    finite_matrix(out.rhs, n, 1, "accumulated RHS");
    require(std::isfinite(out.population), "population overflow");

    // libqt DGESV is Fortran column-major. Explicit packing avoids relying on
    // symmetry to disguise a transposition error and leaves diagnostics untouched.
    std::vector<double> a(nf * nf), b(nf);
    std::vector<int> pivots(nf);
    for (int k = 0; k < n; ++k) {
        b[k] = rhs[k][0];
        for (int l = 0; l < n; ++l) a[k + static_cast<size_t>(l) * n] = out.metric->get(k, l);
    }
    const int info = C_DGESV(n, 1, a.data(), n, pivots.data(), b.data(), n);
    require(info == 0, "LU solve failed (info=" + std::to_string(info) + "); metric may be singular");
    finite_vector(b, nf, "solved coefficients");
    out.coefficients = std::make_shared<Matrix>("ISA-A coefficients", n, 1);
    double residual = 0.0, snorm = 0.0, dnorm = 0.0, tnorm = 0.0;
    for (int k = 0; k < n; ++k) {
        out.coefficients->set(k, 0, b[k]);
        double sd = 0.0, rownorm = 0.0;
        for (int l = 0; l < n; ++l) {
            sd += out.metric->get(k, l) * b[l];
            rownorm += std::abs(out.metric->get(k, l));
        }
        residual = std::max(residual, std::abs(sd - rhs[k][0]));
        snorm = std::max(snorm, rownorm);
        dnorm = std::max(dnorm, std::abs(b[k]));
        tnorm = std::max(tnorm, std::abs(rhs[k][0]));
    }
    const double denominator = snorm * dnorm + tnorm;
    require(std::isfinite(residual) && std::isfinite(denominator), "linear residual overflow");
    out.relative_residual = denominator > 0.0 ? residual / denominator : 0.0;
    return out;
}

double isa_overlap_change(const std::vector<double>& current, const std::vector<double>& previous,
                          const std::shared_ptr<Matrix>& overlap) {
    require(!current.empty() && current.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "invalid coefficient count");
    const int n = static_cast<int>(current.size());
    finite_vector(current, n, "current");
    finite_vector(previous, n, "previous");
    const double metric_scale = symmetric_metric(overlap, n);
    // Scale the metric and coefficients to avoid overflowing the norm product. Scaling cannot
    // change the normalized angle. This is deterministic, not a bitwise BLAS gate.
    double cs = 0.0, ps = 0.0;
    for (int i = 0; i < n; ++i) {
        cs = std::max(cs, std::abs(current[i]));
        ps = std::max(ps, std::abs(previous[i]));
    }
    require(cs > 0.0 && ps > 0.0, "overlap angle is undefined for a zero expansion");
    double cc = 0.0, pp = 0.0, cp = 0.0;
    for (int i = 0; i < n; ++i) {
        double sc = 0.0, sp = 0.0;
        for (int j = 0; j < n; ++j) {
            sc += (overlap->get(i, j) / metric_scale) * (current[j] / cs);
            sp += (overlap->get(i, j) / metric_scale) * (previous[j] / ps);
        }
        cc += (current[i] / cs) * sc;
        pp += (previous[i] / ps) * sp;
        cp += (current[i] / cs) * sp;
    }
    require(std::isfinite(cc) && std::isfinite(pp) && std::isfinite(cp) && cc > 0.0 && pp > 0.0,
            "overlap angle requires finite positive norms");
    const double cosine = std::abs(cp) / std::sqrt(cc) / std::sqrt(pp);
    require(cosine <= 1.0 + 1.e-10, "overlap violates the Cauchy-Schwarz bound");
    return std::abs(1.0 - cosine);
}

}  // namespace isapol
}  // namespace psi
