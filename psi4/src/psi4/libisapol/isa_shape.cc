/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 * ISA Func-1/Fit-3 equations; see SPEC.md and CamCASP stockholder.F90.
 */
#include "isa_shape.h"
#include "parallel_work.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void shape_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void shape_radius(double r) {
    shape_require(std::isfinite(r) && r >= 0, "Shape radius must be finite and nonnegative");
}
void shape_tail_valid(const IsaExponentialTail& tail) {
    if (!tail.defined) return;
    shape_require(std::isfinite(tail.amplitude), "Tail amplitude must be finite");
    shape_require(std::isfinite(tail.exponent) && tail.exponent > 0, "Tail exponent must be finite and positive");
    shape_require(std::isfinite(tail.cutoff) && tail.cutoff >= 0, "Tail cutoff must be finite and nonnegative");
}
}
IsaGaussianShape::IsaGaussianShape(const IsaExplicitBasis& basis, const std::vector<double>& coefficients)
    : basis_(basis), coefficients_(coefficients), centre_(basis.centres_[basis.shells_[0].centre]) {
    shape_require(basis.role_ == IsaBasisRole::Shape, "Gaussian shape requires Shape basis role");
    shape_require(coefficients.size() == static_cast<size_t>(basis.nfunction()), "Shape coefficient dimension mismatch");
    for (size_t s = 0; s < basis.shells_.size(); ++s) {
        const auto& shell = basis.shells_[s];
        shape_require(basis.centres_[shell.centre] == centre_, "Gaussian shape requires co-centred shells");
        shape_require(std::isfinite(coefficients[s]), "Shape coefficients must be finite");
        for (size_t p = 0; p < shell.exponents.size(); ++p) {
            const double amplitude = coefficients[s]*shell.coefficients[p];
            shape_require(std::isfinite(amplitude), "Nonfinite effective shape amplitude");
            exponents_.push_back(shell.exponents[p]); amplitudes_.push_back(amplitude);
        }
    }
}
double IsaGaussianShape::value(double radius) const {
    shape_radius(radius);
    return value_squared(radius*radius);
}
double IsaGaussianShape::value_squared(double r2) const {
    shape_require(std::isfinite(r2), "Nonfinite squared shape radius");
    double value = 0.0;
    // Match function-expansion sampling: contract each shell before applying
    // its expansion coefficient. Flattened amplitudes are for charge integrals,
    // not evaluation; reassociation perturbs subsequent finite-difference tails.
    for (size_t s = 0; s < basis_.shells_.size(); ++s) {
        const auto& shell = basis_.shells_[s];
        double contracted = 0.0;
        for (size_t p = 0; p < shell.exponents.size(); ++p)
            contracted += shell.coefficients[p]*std::exp(-shell.exponents[p]*r2);
        value += coefficients_[s]*contracted;
    }
    shape_require(std::isfinite(value), "Nonfinite Gaussian shape value");
    return value;
}
double IsaGaussianShape::exterior_charge(double radius) const {
    shape_radius(radius);
    const double pi = std::acos(-1.0), r2 = radius*radius;
    shape_require(std::isfinite(r2), "Nonfinite squared shape radius");
    // Match integrate_function_expansion_SphGTOs: ordered double accumulation.
    double charge = 0.0;
    for (size_t p = 0; p < exponents_.size(); ++p) {
        const double a = exponents_[p];
        const double integral = std::pow(pi/a,1.5)*std::erfc(radius*std::sqrt(a)) +
                                (2*pi*radius/a)*std::exp((-a*radius)*radius);
        shape_require(std::isfinite(integral), "Nonfinite exterior Gaussian integral");
        charge += amplitudes_[p]*integral;
    }
    shape_require(std::isfinite(charge), "Nonfinite exterior shape charge");
    return charge;
}
IsaTailFitResult IsaGaussianShape::fit_tail(double cutoff, const IsaExponentialTail& previous) const {
    constexpr double step = 1.e-8;
    shape_require(std::isfinite(cutoff) && cutoff >= step, "Tail fit cutoff must be finite and at least 1e-8");
    shape_tail_valid(previous);
    if (previous.defined)
        shape_require(previous.exponent > 1 && previous.exponent < 4, "Previous Fit-3 exponent must satisfy 1<b<4");
    IsaTailFitResult result;
    result.tail.cutoff = cutoff;
    result.gaussian_tail_charge = exterior_charge(cutoff);
    // The 1e-8 difference amplifies even one-ulp reassociation errors. Keep
    // reference order: translate z-axis points, evaluate contracted shells,
    // THEN multiply/sum expansion coefficients. Do not flatten (d*c)*exp.
    std::vector<std::array<double,3>> points(3, centre_);
    points[0][2] += cutoff;
    points[1][2] += cutoff-step;
    points[2][2] += cutoff+step;
    const auto samples = basis_.evaluate(points);
    std::array<double,3> values{{0.0,0.0,0.0}};
    for (size_t k=0; k<coefficients_.size(); ++k)
        for (size_t p=0; p<3; ++p)
            values[p] += coefficients_[k]*samples->get(p,k);
    const double w = values[0];
    const double derivative = (values[2]-values[1])/(2*step);
    double b = w != 0.0 ? -derivative/w : 0.0;
    if (!(std::isfinite(b) && b > 1 && b < 4)) {
        if (!previous.defined) { result.status = "undefined_slope"; return result; }
        b = previous.exponent;
        result.used_previous_exponent = true;
    }
    // Analytic integral of unit-amplitude exp(-b r), not a continuity fit.
    const double pi = std::acos(-1.0);
    const double q = 4*pi*std::pow(1/b,3)*(b*b*cutoff*cutoff+2*b*cutoff+2)*std::exp(-b*cutoff);
    shape_require(std::isfinite(q), "Nonfinite exponential tail integral");
    if (!(q > 0)) { result.status = "undefined_tail_integral"; return result; }
    const double amplitude = result.gaussian_tail_charge/q;
    shape_require(std::isfinite(amplitude), "Nonfinite fitted tail amplitude");
    result.tail.defined = true; result.tail.amplitude = amplitude; result.tail.exponent = b;
    result.ionization_potential = b*b/8;
    result.status = result.used_previous_exponent ? "previous_exponent" : "fitted";
    return result;
}
std::vector<double> IsaGaussianShape::sample(const std::vector<std::array<double,3>>& points,
                                            const IsaExponentialTail& tail, bool apply_tail) const {
    return sample_counted(points,tail,apply_tail,nullptr);
}
std::vector<double> IsaGaussianShape::sample_counted(const std::vector<std::array<double,3>>& points,
        const IsaExponentialTail& tail, bool apply_tail, int* clipped) const {
    shape_tail_valid(tail);
    const bool active = apply_tail && tail.defined;
    std::vector<double> samples(points.size());
    std::vector<unsigned char> negative(clipped && !active ? points.size() : 0);
    detail::parallel_work(points.size(),256,[&](size_t i) {
        const auto& p = points[i];
        double r2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            shape_require(std::isfinite(p[axis]), "Shape sample points must be finite");
            const double d = p[axis]-centre_[axis]; r2 += d*d;
        }
        shape_require(std::isfinite(r2), "Nonfinite shape sample distance");
        const double r = std::sqrt(r2);
        double w = active && r > tail.cutoff ? tail.amplitude*std::exp(-tail.exponent*r) : value_squared(r2);
        if (!active) {
            if (clipped) negative[i] = w < 0.0;
            w = std::max(w,0.0);
        }
        shape_require(std::isfinite(w), "Nonfinite shape sample");
        samples[i] = w;
    });
    if (clipped) *clipped = static_cast<int>(std::count(negative.begin(),negative.end(),1));
    return samples;
}
} }
