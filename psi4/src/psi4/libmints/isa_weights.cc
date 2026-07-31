/*
 * Native real-space iterated-stockholder (ISA) weights.
 *
 * This implementation deliberately solves the defining stockholder fixed point
 * from the frozen AO density.  It does not use MBIS, nearest-centre, or uniform
 * production fallbacks.
 */

#include "psi4/libmints/atomic_polarizability.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/exception.h"

namespace psi {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kLogFloor = -676.3964185322641;  // log(DBL_MIN) + 32
constexpr double kCoincidentTolerance = 1.0e-12;
constexpr double kUnityTolerance = 1.0e-13;

bool finite_site(const SitePosition& p) {
    return std::all_of(p.begin(), p.end(), [](double value) { return std::isfinite(value); });
}

double distance(const SitePosition& a, const SitePosition& b) {
    return std::hypot(std::hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]);
}

struct KahanSum {
    double sum{};
    double correction{};
    void add(double value) {
        const double adjusted = value - correction;
        const double next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
};

struct Quadrature {
    std::vector<double> nodes;
    std::vector<double> weights;
};

Quadrature gauss_legendre(std::size_t count, double lower, double upper) {
    Quadrature result;
    result.nodes.resize(count);
    result.weights.resize(count);
    const std::size_t half = (count + 1) / 2;
    for (std::size_t i = 0; i < half; ++i) {
        double root = std::cos(kPi * (static_cast<double>(i) + 0.75) /
                               (static_cast<double>(count) + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p0 = 1.0;
            double p1 = root;
            for (std::size_t order = 2; order <= count; ++order) {
                const double p2 = ((2.0 * order - 1.0) * root * p1 - (order - 1.0) * p0) /
                                  static_cast<double>(order);
                p0 = p1;
                p1 = p2;
            }
            derivative = static_cast<double>(count) * (root * p1 - p0) / (root * root - 1.0);
            const double next = root - p1 / derivative;
            if (std::abs(next - root) <= 4.0 * std::numeric_limits<double>::epsilon()) {
                root = next;
                break;
            }
            root = next;
            if (iteration == 99) throw PSIEXCEPTION("ISA: Gauss-Legendre root generation did not converge");
        }
        const double midpoint = 0.5 * (lower + upper);
        const double half_width = 0.5 * (upper - lower);
        const double weight = half_width * 2.0 / ((1.0 - root * root) * derivative * derivative);
        result.nodes[i] = midpoint - half_width * root;
        result.nodes[count - 1 - i] = midpoint + half_width * root;
        result.weights[i] = weight;
        result.weights[count - 1 - i] = weight;
    }
    return result;
}

Quadrature mapped_radial(std::size_t count, double scale) {
    const auto unit = gauss_legendre(count, 0.0, 1.0);
    Quadrature result;
    result.nodes.reserve(count + 1);
    result.weights.reserve(count + 1);
    result.nodes.push_back(0.0);
    result.weights.push_back(0.0);
    for (std::size_t i = 0; i < count; ++i) {
        const double denominator = 1.0 - unit.nodes[i];
        const double radius = scale * unit.nodes[i] / denominator;
        const double drdx = scale / (denominator * denominator);
        const double weight = unit.weights[i] * drdx;
        if (!std::isfinite(radius) || !std::isfinite(weight) || radius <= 0.0 || weight <= 0.0)
            throw PSIEXCEPTION("ISA: mapped radial quadrature is invalid");
        result.nodes.push_back(radius);
        result.weights.push_back(weight);
    }
    return result;
}

struct AngularGrid {
    std::vector<SitePosition> directions;
    std::vector<double> weights;
};

AngularGrid product_spherical_grid(std::size_t polar_count, std::size_t azimuthal_count) {
    const auto polar = gauss_legendre(polar_count, -1.0, 1.0);
    AngularGrid grid;
    grid.directions.reserve(polar_count * azimuthal_count);
    grid.weights.reserve(polar_count * azimuthal_count);
    for (std::size_t i = 0; i < polar_count; ++i) {
        const double z = polar.nodes[i];
        const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
        for (std::size_t j = 0; j < azimuthal_count; ++j) {
            const double phi = 2.0 * kPi * static_cast<double>(j) / static_cast<double>(azimuthal_count);
            grid.directions.push_back({radial * std::cos(phi), radial * std::sin(phi), z});
            grid.weights.push_back(0.5 * polar.weights[i] / static_cast<double>(azimuthal_count));
        }
    }
    KahanSum sum;
    for (double weight : grid.weights) sum.add(weight);
    if (std::abs(sum.sum - 1.0) > 2.0e-14)
        throw PSIEXCEPTION("ISA: normalized spherical quadrature failed unity");
    return grid;
}

std::vector<double> pchip_slopes(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.size();
    if (n < 2 || y.size() != n) throw PSIEXCEPTION("ISA: PCHIP requires matching nodes and values");
    std::vector<double> h(n - 1), delta(n - 1), slope(n, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        h[i] = x[i + 1] - x[i];
        if (!std::isfinite(h[i]) || h[i] <= 0.0 || !std::isfinite(y[i]) || !std::isfinite(y[i + 1]))
            throw PSIEXCEPTION("ISA: PCHIP nodes/log values must be finite and strictly ordered");
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }
    if (n == 2) {
        slope[0] = slope[1] = delta[0];
        return slope;
    }
    for (std::size_t i = 1; i + 1 < n; ++i) {
        if (delta[i - 1] * delta[i] <= 0.0) {
            slope[i] = 0.0;
        } else {
            const double w1 = 2.0 * h[i] + h[i - 1];
            const double w2 = h[i] + 2.0 * h[i - 1];
            slope[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }
    const auto endpoint = [](double h0, double h1, double d0, double d1) {
        double value = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
        if (value * d0 <= 0.0) value = 0.0;
        else if (d0 * d1 < 0.0 && std::abs(value) > 3.0 * std::abs(d0)) value = 3.0 * d0;
        return value;
    };
    slope.front() = endpoint(h[0], h[1], delta[0], delta[1]);
    slope.back() = endpoint(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    return slope;
}

struct Profile {
    std::vector<double> nodes;
    std::vector<double> logs;
    std::vector<double> slopes;
    bool gaussian{};
    double gaussian_alpha{};
    double gaussian_log_norm{};
    bool has_tail{};
    double tail_join{};
    double tail_alpha{};
    double tail_log_amplitude{};

    Profile() = default;
    Profile(std::vector<double> radial_nodes, std::vector<double> log_values)
        : nodes(std::move(radial_nodes)), logs(std::move(log_values)) {
        // Only sampled profile storage is floored. Analytical Gaussian and tail
        // evaluations remain unbounded in log space for correct far-field ratios.
        for (double& value : logs) value = std::max(kLogFloor, value);
        slopes = pchip_slopes(nodes, logs);
    }

    static Profile initial(const std::vector<double>& nodes, double alpha) {
        std::vector<double> logs(nodes.size());
        const double norm = 1.5 * std::log(alpha / kPi);
        for (std::size_t i = 0; i < nodes.size(); ++i) logs[i] = std::max(kLogFloor, norm - alpha * nodes[i] * nodes[i]);
        Profile profile(nodes, std::move(logs));
        profile.gaussian = true;
        profile.gaussian_alpha = alpha;
        profile.gaussian_log_norm = norm;
        return profile;
    }

    double pchip(double radius) const {
        if (radius <= nodes.front()) return logs.front();
        if (radius >= nodes.back())
            return std::max(kLogFloor, logs.back() + slopes.back() * (radius - nodes.back()));
        const auto upper = std::upper_bound(nodes.begin(), nodes.end(), radius);
        const std::size_t i = static_cast<std::size_t>(upper - nodes.begin() - 1);
        const double h = nodes[i + 1] - nodes[i];
        const double t = (radius - nodes[i]) / h;
        const double t2 = t * t;
        const double t3 = t2 * t;
        double value = (2.0 * t3 - 3.0 * t2 + 1.0) * logs[i] +
                       (t3 - 2.0 * t2 + t) * h * slopes[i] +
                       (-2.0 * t3 + 3.0 * t2) * logs[i + 1] +
                       (t3 - t2) * h * slopes[i + 1];
        value = std::min(std::max(value, std::min(logs[i], logs[i + 1])),
                         std::max(logs[i], logs[i + 1]));
        return std::max(kLogFloor, value);
    }

    double eval(double radius) const {
        if (!std::isfinite(radius) || radius < 0.0) throw PSIEXCEPTION("ISA: profile radius is invalid");
        double value;
        if (has_tail && radius > tail_join) value = tail_log_amplitude - tail_alpha * radius;
        else if (gaussian) value = gaussian_log_norm - gaussian_alpha * radius * radius;
        else value = pchip(radius);
        if (!std::isfinite(value)) throw PSIEXCEPTION("ISA: radial log profile is not finite");
        return value;
    }
};

double tail_charge_for_alpha(double value_at_join, double join, double alpha) {
    return 4.0 * kPi * value_at_join *
           (join * join / alpha + 2.0 * join / (alpha * alpha) + 2.0 / (alpha * alpha * alpha));
}

double solve_tail_alpha(double value_at_join, double join, double charge) {
    if (!std::isfinite(value_at_join) || value_at_join <= 0.0 || !std::isfinite(join) || join <= 0.0 ||
        !std::isfinite(charge) || charge <= 0.0)
        throw PSIEXCEPTION("ISA: exponential tail fit has invalid value or charge");
    double lower = 1.0e-12;
    double upper = 1.0;
    while (tail_charge_for_alpha(value_at_join, join, upper) > charge) {
        upper *= 2.0;
        if (!std::isfinite(upper)) throw PSIEXCEPTION("ISA: exponential tail root could not be bracketed");
    }
    for (int iteration = 0; iteration < 256; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        const double computed = tail_charge_for_alpha(value_at_join, join, middle);
        if (computed > charge) lower = middle;
        else upper = middle;
        if ((upper - lower) <= 1.0e-13 * std::max(1.0, upper)) break;
    }
    const double alpha = 0.5 * (lower + upper);
    const double relative = std::abs(tail_charge_for_alpha(value_at_join, join, alpha) - charge) / charge;
    if (!std::isfinite(alpha) || alpha <= 0.0 || !std::isfinite(relative) || relative > 2.0e-12)
        throw PSIEXCEPTION("ISA: exponential tail root did not meet its residual tolerance");
    return alpha;
}

void fit_tail(Profile& profile, const Quadrature& radial, double scale, double join) {
    const double log_value = profile.pchip(join);
    const double value = std::exp(log_value);
    // Integrate on [join, infinity) directly.  Filtering a quadrature built for
    // [0, infinity) introduces a moving step at join and makes the fitted tail
    // charge oscillate strongly as the radial point count changes.
    const auto tail_radial = mapped_radial(radial.nodes.size() - 1, scale);
    long double charge = 0.0L;
    for (std::size_t i = 1; i < tail_radial.nodes.size(); ++i) {
        const double radius_double = join + tail_radial.nodes[i];
        const long double radius = radius_double;
        charge += 4.0L * static_cast<long double>(kPi) * radius * radius * tail_radial.weights[i] *
                  std::exp(static_cast<long double>(profile.pchip(radius_double)));
    }
    const double charge_double = static_cast<double>(charge);
    const double alpha = solve_tail_alpha(value, join, charge_double);
    profile.gaussian = false;
    profile.has_tail = true;
    profile.tail_join = join;
    profile.tail_alpha = alpha;
    profile.tail_log_amplitude = log_value + alpha * join;
}

std::vector<double> probabilities(const SitePosition& point, const std::vector<SitePosition>& sites,
                                  const std::vector<Profile>& profiles) {
    std::vector<double> logs(sites.size());
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t site = 0; site < sites.size(); ++site) {
        logs[site] = profiles[site].eval(distance(point, sites[site]));
        maximum = std::max(maximum, logs[site]);
    }
    if (!std::isfinite(maximum)) throw PSIEXCEPTION("ISA: no finite pro-atom log shape at a point");
    KahanSum denominator;
    for (double& value : logs) {
        value = std::exp(value - maximum);
        denominator.add(value);
    }
    if (!std::isfinite(denominator.sum) || denominator.sum <= 0.0)
        throw PSIEXCEPTION("ISA: log-sum-exp promolecule is invalid");
    for (double& value : logs) value /= denominator.sum;
    return logs;
}

long double exponential_overlap_tail(double log_amplitude, double exponent, double join) {
    const long double beta = exponent;
    const long double r0 = join;
    const long double polynomial = r0 * r0 / beta + 2.0L * r0 / (beta * beta) +
                                   2.0L / (beta * beta * beta);
    return 4.0L * static_cast<long double>(kPi) *
           std::exp(static_cast<long double>(log_amplitude) - beta * r0) * polynomial;
}

double normalized_overlap(const Profile& first_input, const Profile& second_input,
                          const Quadrature& radial, double radial_scale, double default_tail_join,
                          std::size_t inner_integration_points, bool complete_missing_tails) {
    Profile first = first_input;
    Profile second = second_input;
    if (complete_missing_tails) {
        // At first tail activation the old profile is still raw while the new
        // profile has a fitted tail. Fit a comparison-only tail to the old
        // profile so both functions have their own continuous piecewise form.
        if (!first.has_tail) fit_tail(first, radial, radial_scale, default_tail_join);
        if (!second.has_tail) fit_tail(second, radial, radial_scale, default_tail_join);
    }
    const bool analytic_tail = first.has_tail && second.has_tail;
    long double cross = 0.0L, norm_first = 0.0L, norm_second = 0.0L;
    if (!analytic_tail) {
        // Before activation, both profiles are raw and use the complete mapped
        // rule. Canonical convergence cannot be declared until tails activate.
        for (std::size_t i = 1; i < radial.nodes.size(); ++i) {
            const long double radius = radial.nodes[i];
            const long double measure = 4.0L * static_cast<long double>(kPi) * radius * radius * radial.weights[i];
            const long double a = std::exp(static_cast<long double>(first.eval(radial.nodes[i])));
            const long double b = std::exp(static_cast<long double>(second.eval(radial.nodes[i])));
            cross += measure * a * b;
            norm_first += measure * a * a;
            norm_second += measure * b * b;
        }
    } else {
        // Integrate every inner piece on its exact interval. This handles equal
        // or distinct join radii and mixed old/new tail status without ever
        // filtering a quadrature constructed for [0, infinity).
        std::vector<double> boundaries{0.0, first.tail_join, second.tail_join};
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        for (std::size_t interval = 0; interval + 1 < boundaries.size(); ++interval) {
            if (boundaries[interval + 1] <= boundaries[interval]) continue;
            const auto inner = gauss_legendre(inner_integration_points, boundaries[interval],
                                              boundaries[interval + 1]);
            for (std::size_t i = 0; i < inner.nodes.size(); ++i) {
                const long double radius = inner.nodes[i];
                const long double measure =
                    4.0L * static_cast<long double>(kPi) * radius * radius * inner.weights[i];
                const long double a = std::exp(static_cast<long double>(first.eval(inner.nodes[i])));
                const long double b = std::exp(static_cast<long double>(second.eval(inner.nodes[i])));
                cross += measure * a * b;
                norm_first += measure * a * a;
                norm_second += measure * b * b;
            }
        }
        const double outer_join = boundaries.back();
        cross += exponential_overlap_tail(first.tail_log_amplitude + second.tail_log_amplitude,
                                           first.tail_alpha + second.tail_alpha, outer_join);
        norm_first += exponential_overlap_tail(2.0 * first.tail_log_amplitude,
                                                2.0 * first.tail_alpha, outer_join);
        norm_second += exponential_overlap_tail(2.0 * second.tail_log_amplitude,
                                                 2.0 * second.tail_alpha, outer_join);
    }
    if (!(cross > 0.0L) || !(norm_first > 0.0L) || !(norm_second > 0.0L))
        throw PSIEXCEPTION("ISA: normalized radial overlap is undefined");
    long double overlap = cross / std::sqrt(norm_first * norm_second);
    overlap = std::min(1.0L, std::max(0.0L, overlap));
    return std::abs(1.0 - static_cast<double>(overlap));
}

// Bragg-Slater radii in bohr, Slater JCP 41 (1964) 3199.  This versioned
// table intentionally has no out-of-range guess.
double slater_radius(int atomic_number) {
    static const std::vector<double> radii = {
        1.000, 0.661, 0.661,
        2.740, 1.984, 1.606, 1.323, 1.228, 1.134, 0.945, 0.900,
        3.402, 2.835, 2.362, 2.079, 1.890, 1.890, 1.890, 1.890,
        4.157, 3.402, 3.024, 2.656, 2.551, 2.656, 2.656, 2.656, 2.551, 2.551, 2.551, 2.551,
        2.457, 2.362, 2.173, 2.173, 2.173, 2.173,
        4.441, 3.780, 3.402, 2.929, 2.740, 2.740, 2.551, 2.457, 2.551, 2.646, 3.024, 2.929,
        2.929, 2.740, 2.740, 2.646, 2.646, 2.646,
        4.913, 4.063, 3.685, 3.496, 3.496, 3.496, 3.496, 3.496, 3.496, 3.402, 3.307, 3.307,
        3.307, 3.307, 3.307, 3.307, 3.307,
        2.929, 2.740, 2.551, 2.551, 2.457, 2.551, 2.551, 2.551, 2.835, 3.591, 3.024, 3.024,
        3.591, 3.591, 3.591,
        4.063, 4.063, 3.685, 3.401, 3.401, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307,
        3.307, 3.307, 3.307, 3.307, 3.307};
    if (atomic_number <= 0 || static_cast<std::size_t>(atomic_number) >= radii.size())
        throw PSIEXCEPTION("ISA: atomic number is absent from radius table Slater-1964-bohr-v1");
    return radii[static_cast<std::size_t>(atomic_number)];
}

void validate_inputs(const std::vector<SitePosition>& sites, const std::vector<SitePosition>& points,
                     const std::vector<double>& weights, const std::vector<int>& atomic_numbers) {
    if (sites.empty()) throw PSIEXCEPTION("ISA: at least one real nuclear site is required");
    if (points.empty() || points.size() != weights.size())
        throw PSIEXCEPTION("ISA: output points and integration weights are inconsistent");
    if (atomic_numbers.size() != sites.size()) throw PSIEXCEPTION("ISA: atomic numbers do not match sites");
    for (std::size_t i = 0; i < sites.size(); ++i) {
        if (!finite_site(sites[i])) throw PSIEXCEPTION("ISA: site coordinates must be finite");
        (void)slater_radius(atomic_numbers[i]);
        for (std::size_t j = 0; j < i; ++j)
            if (distance(sites[i], sites[j]) < kCoincidentTolerance)
                throw PSIEXCEPTION("ISA: coincident stockholder sites are not uniquely partitionable");
    }
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!finite_site(points[i]) || !std::isfinite(weights[i]))
            throw PSIEXCEPTION("ISA: output grid points/weights must be finite");
        if (weights[i] < 0.0)
            throw PSIEXCEPTION("ISA: output grid integration weights must be nonnegative");
    }
}

void clamp_density(std::vector<double>& density) {
    double maximum = 0.0;
    for (double value : density) {
        if (!std::isfinite(value)) throw PSIEXCEPTION("ISA: molecular density must be finite");
        maximum = std::max(maximum, std::abs(value));
    }
    const double tolerance = 1.0e-14 * std::max(1.0, maximum);
    for (double& value : density) {
        if (value < -tolerance) throw PSIEXCEPTION("ISA: molecular density is negative beyond roundoff tolerance");
        if (value < 0.0) value = 0.0;
    }
}

struct CoreResult {
    std::vector<double> weights;
    ISADiagnostics diagnostics;
};

CoreResult solve(const std::vector<SitePosition>& sites, const std::vector<SitePosition>& output_points,
                 const std::vector<double>& output_weights, const std::vector<int>& atomic_numbers,
                 const std::vector<double>& supplied_output_density,
                 const std::function<double(const SitePosition&)>& density_evaluator,
                 const ISAOptions& options, double formal_electron_count, bool enforce_electron_count,
                 std::size_t inject_tail_fit_failure_iteration = 0) {
    validate_inputs(sites, output_points, output_weights, atomic_numbers);
    std::vector<double> output_density = supplied_output_density;
    if (output_density.size() != output_points.size()) throw PSIEXCEPTION("ISA: output density cardinality is invalid");
    clamp_density(output_density);

    const std::size_t nsite = sites.size();
    const auto angular = product_spherical_grid(options.angular_polar_points(), options.angular_azimuthal_points());
    const std::size_t nangular = angular.directions.size();
    std::vector<Quadrature> radial(nsite);
    std::vector<double> scales(nsite), joins(nsite);
    std::vector<Profile> profiles;
    profiles.reserve(nsite);
    for (std::size_t site = 0; site < nsite; ++site) {
        scales[site] = slater_radius(atomic_numbers[site]);
        joins[site] = options.tail_join_factor() * scales[site];
        radial[site] = mapped_radial(options.radial_points(), scales[site]);
        profiles.push_back(Profile::initial(radial[site].nodes, options.initial_alpha()));
    }

    // Seal the auxiliary density once. Ordering is (site, origin; radial, angular).
    std::vector<std::vector<double>> shell_density(nsite);
    for (std::size_t site = 0; site < nsite; ++site) {
        auto& values = shell_density[site];
        values.reserve(1 + options.radial_points() * nangular);
        values.push_back(density_evaluator(sites[site]));
        for (std::size_t r = 1; r < radial[site].nodes.size(); ++r) {
            const double radius = radial[site].nodes[r];
            for (const auto& direction : angular.directions) {
                SitePosition point{sites[site][0] + radius * direction[0],
                                   sites[site][1] + radius * direction[1],
                                   sites[site][2] + radius * direction[2]};
                values.push_back(density_evaluator(point));
            }
        }
        clamp_density(values);
    }

    KahanSum electron_sum;
    for (std::size_t point = 0; point < output_points.size(); ++point)
        electron_sum.add(output_weights[point] * output_density[point]);
    if (!std::isfinite(electron_sum.sum) || electron_sum.sum <= 0.0)
        throw PSIEXCEPTION("ISA: molecular-grid electron integration is not finite and positive");
    const double electron_error = std::abs(electron_sum.sum - formal_electron_count);
    if (enforce_electron_count && electron_error > options.electron_count_tolerance())
        throw PSIEXCEPTION("ISA: molecular-grid electron count exceeds the configured integration tolerance");

    ISADiagnostics diagnostics;
    diagnostics.electron_count = electron_sum.sum;
    diagnostics.formal_electron_count = formal_electron_count;
    diagnostics.electron_count_absolute_error = electron_error;
    diagnostics.electron_count_relative_error = electron_error / std::max(1.0, std::abs(formal_electron_count));
    diagnostics.grid_profile.radial_points = options.radial_points() + 1;
    diagnostics.grid_profile.angular_points = nangular;
    diagnostics.grid_profile.shell_point_count = nsite * (1 + options.radial_points() * nangular);
    diagnostics.grid_profile.angular_rule = "Gauss-Legendre-polar x uniform-azimuth exact product rule";
    diagnostics.grid_profile.radial_rule = "mapped Gauss-Legendre r=s*x/(1-x), explicit origin";
    diagnostics.grid_profile.radius_table = "Slater-1964-bohr-v1";
    diagnostics.grid_profile.atom_scales = scales;

    std::vector<double> previous_populations(nsite, 0.0);
    std::vector<double> previous_output_weights(output_points.size() * nsite, 0.0);
    bool tail_active = false;

    for (std::size_t iteration = 0; iteration < options.max_iterations(); ++iteration) {
        std::vector<Profile> updated;
        updated.reserve(nsite);
        for (std::size_t site = 0; site < nsite; ++site) {
            std::vector<double> logs(radial[site].nodes.size(), kLogFloor);
            std::size_t shell_index = 0;
            {
                const auto p = probabilities(sites[site], sites, profiles);
                const double average = shell_density[site][shell_index++] * p[site];
                logs[0] = average > 0.0 ? std::max(kLogFloor, std::log(average)) : profiles[site].eval(0.0);
                if (average <= 0.0) ++diagnostics.underflow_fallbacks;
            }
            for (std::size_t r = 1; r < radial[site].nodes.size(); ++r) {
                KahanSum average;
                const double radius = radial[site].nodes[r];
                for (std::size_t q = 0; q < nangular; ++q) {
                    const auto& direction = angular.directions[q];
                    SitePosition point{sites[site][0] + radius * direction[0],
                                       sites[site][1] + radius * direction[1],
                                       sites[site][2] + radius * direction[2]};
                    const auto p = probabilities(point, sites, profiles);
                    average.add(angular.weights[q] * shell_density[site][shell_index++] * p[site]);
                }
                if (!std::isfinite(average.sum) || average.sum < 0.0)
                    throw PSIEXCEPTION("ISA: spherical stockholder average is not finite and nonnegative");
                if (average.sum > 0.0) logs[r] = std::max(kLogFloor, std::log(average.sum));
                else {
                    logs[r] = profiles[site].eval(radius);
                    ++diagnostics.underflow_fallbacks;
                }
            }
            if (shell_index != shell_density[site].size()) throw PSIEXCEPTION("ISA: shell-grid indexing is inconsistent");

            if (options.mix_fraction() < 1.0) {
                const double eta = options.mix_fraction();
                for (std::size_t r = 0; r < logs.size(); ++r) {
                    const double old_log = profiles[site].eval(radial[site].nodes[r]);
                    const double maximum = std::max(old_log, logs[r]);
                    logs[r] = maximum + std::log((1.0 - eta) * std::exp(old_log - maximum) +
                                                 eta * std::exp(logs[r] - maximum));
                }
            }
            updated.emplace_back(radial[site].nodes, std::move(logs));
        }

        double provisional_residual = 0.0;
        for (std::size_t site = 0; site < nsite; ++site)
            provisional_residual = std::max(
                provisional_residual,
                normalized_overlap(profiles[site], updated[site], radial[site], scales[site], joins[site],
                                   options.radial_points(), false));
        tail_active = tail_active || iteration + 1 >= options.tail_activation_iteration() ||
                      provisional_residual <= options.tail_activation_convergence();
        if (tail_active) {
            for (std::size_t site = 0; site < nsite; ++site) {
                try {
                    if (inject_tail_fit_failure_iteration == iteration + 1)
                        throw PSIEXCEPTION("ISA test injection: tail fit failure");
                    fit_tail(updated[site], radial[site], scales[site], joins[site]);
                } catch (const std::exception&) {
                    ++diagnostics.tail_fit_failures;
                    if (profiles[site].has_tail) {
                        // Retain the complete prior profile. Combining a new
                        // inner profile with an old tail is discontinuous at r0.
                        updated[site] = profiles[site];
                        ++diagnostics.tail_failure_reused_profiles;
                    } else {
                        throw PSIEXCEPTION("ISA: no valid exponential tail was available by activation");
                    }
                }
            }
        }

        diagnostics.max_overlap_residual = 0.0;
        for (std::size_t site = 0; site < nsite; ++site) {
            const double residual = normalized_overlap(
                profiles[site], updated[site], radial[site], scales[site], joins[site],
                options.radial_points(), tail_active);
            diagnostics.max_overlap_residual = std::max(diagnostics.max_overlap_residual, residual);
        }

        std::vector<double> current_weights(output_points.size() * nsite);
        std::vector<double> populations(nsite, 0.0);
        diagnostics.max_weight_change = 0.0;
        for (std::size_t point = 0; point < output_points.size(); ++point) {
            const auto p = probabilities(output_points[point], sites, updated);
            for (std::size_t site = 0; site < nsite; ++site) {
                const std::size_t index = point * nsite + site;
                current_weights[index] = p[site];
                diagnostics.max_weight_change = std::max(
                    diagnostics.max_weight_change, std::abs(p[site] - previous_output_weights[index]));
                populations[site] += output_weights[point] * output_density[point] * p[site];
            }
        }
        diagnostics.max_population_change = 0.0;
        for (std::size_t site = 0; site < nsite; ++site)
            diagnostics.max_population_change = std::max(
                diagnostics.max_population_change, std::abs(populations[site] - previous_populations[site]));
        previous_populations = std::move(populations);
        previous_output_weights = std::move(current_weights);
        profiles = std::move(updated);
        diagnostics.iterations = iteration + 1;
        if (diagnostics.max_overlap_residual <= options.convergence()) {
            diagnostics.converged = true;
            break;
        }
    }
    if (!diagnostics.converged) throw PSIEXCEPTION("ISA: real-space stockholder iteration did not converge");

    CoreResult result;
    result.weights.resize(output_points.size() * nsite);
    result.diagnostics = std::move(diagnostics);
    result.diagnostics.atomic_populations.assign(nsite, 0.0);
    result.diagnostics.max_unity_residual = 0.0;
    for (std::size_t point = 0; point < output_points.size(); ++point) {
        auto p = probabilities(output_points[point], sites, profiles);
        long double sum = 0.0L;
        std::size_t largest = 0;
        for (std::size_t site = 0; site < nsite; ++site) {
            sum += p[site];
            if (p[site] > p[largest]) largest = site;
        }
        const double residual = static_cast<double>(1.0L - sum);
        if (p[largest] + residual >= 0.0 && p[largest] + residual <= 1.0) p[largest] += residual;
        else {
            KahanSum closure;
            for (double value : p) closure.add(value);
            for (double& value : p) value /= closure.sum;
        }
        KahanSum check;
        for (double value : p) check.add(value);
        const double unity = std::abs(check.sum - 1.0);
        result.diagnostics.max_unity_residual = std::max(result.diagnostics.max_unity_residual, unity);
        if (unity > kUnityTolerance) throw PSIEXCEPTION("ISA: final pointwise partition unity failed");
        KahanSum partitioned_density;
        for (std::size_t site = 0; site < nsite; ++site) {
            const double value = p[site];
            if (!std::isfinite(value) || value < 0.0 || value > 1.0)
                throw PSIEXCEPTION("ISA: final partition weight is not finite and bounded");
            result.weights[point * nsite + site] = value;
            partitioned_density.add(output_density[point] * value);
            result.diagnostics.atomic_populations[site] +=
                output_weights[point] * output_density[point] * value;
        }
        const double density_residual = std::abs(partitioned_density.sum - output_density[point]);
        const double density_bound = 32.0 * std::numeric_limits<double>::epsilon() *
                                     std::max(1.0, std::abs(output_density[point]));
        if (density_residual > density_bound) throw PSIEXCEPTION("ISA: pointwise density partition failed");
    }
    KahanSum population_sum;
    for (double population : result.diagnostics.atomic_populations) population_sum.add(population);
    result.diagnostics.total_charge_residual = population_sum.sum - electron_sum.sum;
    const double charge_bound = 64.0 * std::numeric_limits<double>::epsilon() *
                                std::max(1.0, std::abs(electron_sum.sum)) * output_points.size();
    if (std::abs(result.diagnostics.total_charge_residual) > charge_bound)
        throw PSIEXCEPTION("ISA: integrated stockholder population conservation failed");
    for (const auto& rule : radial) result.diagnostics.radial_nodes.push_back(rule.nodes);
    for (const auto& profile : profiles) {
        result.diagnostics.log_profiles.push_back(profile.logs);
        result.diagnostics.tail_join_radii.push_back(profile.has_tail ? profile.tail_join : 0.0);
        result.diagnostics.tail_alphas.push_back(profile.has_tail ? profile.tail_alpha : 0.0);
    }
    return result;
}

std::vector<double> ao_density(const FrozenResponseContext& context,
                               const std::vector<SitePosition>& points,
                               const std::vector<std::vector<int>>* maps = nullptr) {
    const auto& basis_const = context.basis();
    if (!basis_const) throw PSIEXCEPTION("ISA: frozen AO basis is null");
    const std::size_t nbf = static_cast<std::size_t>(basis_const->nbf());
    const auto& da = context.Da();
    const auto& db = context.Db();
    if (!da || !db || da->nirrep() != 1 || db->nirrep() != 1 || da->nrow() != static_cast<int>(nbf) ||
        da->ncol() != static_cast<int>(nbf) || db->nrow() != static_cast<int>(nbf) ||
        db->ncol() != static_cast<int>(nbf))
        throw PSIEXCEPTION("ISA: frozen density matrices are not a complete C1 AO density");
    auto* basis = const_cast<BasisSet*>(basis_const.get());  // compute_phi is logically const but legacy API is not.
    std::vector<double> phi(nbf), density(points.size());
    for (std::size_t point = 0; point < points.size(); ++point) {
        basis->compute_phi(phi.data(), points[point][0], points[point][1], points[point][2]);
        KahanSum contraction;
        if (maps) {
            for (int mu : (*maps)[point]) {
                if (mu < 0 || static_cast<std::size_t>(mu) >= nbf) throw PSIEXCEPTION("ISA: frozen AO map index is invalid");
                for (int nu : (*maps)[point]) {
                    if (nu < 0 || static_cast<std::size_t>(nu) >= nbf) throw PSIEXCEPTION("ISA: frozen AO map index is invalid");
                    contraction.add((da->get(mu, nu) + db->get(mu, nu)) * phi[mu] * phi[nu]);
                }
            }
        } else {
            for (std::size_t mu = 0; mu < nbf; ++mu)
                for (std::size_t nu = 0; nu < nbf; ++nu)
                    contraction.add((da->get(mu, nu) + db->get(mu, nu)) * phi[mu] * phi[nu]);
        }
        density[point] = contraction.sum;
    }
    clamp_density(density);
    return density;
}

struct DigestBuilder {
    std::uint64_t hash{1469598103934665603ULL};

    void bytes(const void* data, std::size_t size) {
        const auto* values = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= values[i];
            hash *= 1099511628211ULL;
        }
    }
    template <typename T>
    void scalar(const T& value) {
        bytes(&value, sizeof(T));
    }
    void boolean(bool value) { scalar(static_cast<unsigned char>(value ? 1 : 0)); }
    void string(const std::string& value) {
        scalar(static_cast<std::uint64_t>(value.size()));
        bytes(value.data(), value.size());
    }
    template <typename T>
    void vector(const std::vector<T>& values) {
        scalar(static_cast<std::uint64_t>(values.size()));
        for (const auto& value : values) scalar(value);
    }
    void shell(const BasisShellSnapshot& value) {
        scalar(value.shell_type);
        scalar(value.angular_momentum);
        scalar(value.puream);
        scalar(value.center);
        scalar(value.start);
        scalar(value.nfunction);
        scalar(value.ncartesian);
        vector(value.coordinates);
        vector(value.exponents);
        vector(value.coefficients);
        vector(value.original_coefficients);
        vector(value.erd_coefficients);
        vector(value.radial_powers);
    }
    void basis(const BasisSetStructuralSnapshot& value) {
        string(value.name);
        string(value.key);
        string(value.target);
        scalar(static_cast<std::uint64_t>(value.shells.size()));
        for (const auto& item : value.shells) shell(item);
        scalar(static_cast<std::uint64_t>(value.ecp_shells.size()));
        for (const auto& item : value.ecp_shells) shell(item);
        scalar(static_cast<std::uint64_t>(value.core_electrons.size()));
        for (const auto& item : value.core_electrons) {
            string(item.first);
            scalar(item.second);
        }
        vector(value.scalar_state);
        boolean(value.puream);
        vector(value.n_prim_per_shell);
        vector(value.shell_first_ao);
        vector(value.shell_first_exponent);
        vector(value.shell_first_basis_function);
        vector(value.shell_center);
        vector(value.ecp_shell_center);
        vector(value.function_to_shell);
        vector(value.ao_to_shell);
        vector(value.function_center);
        vector(value.center_to_nshell);
        vector(value.center_to_shell);
        vector(value.center_to_ecp_nshell);
        vector(value.center_to_ecp_shell);
        vector(value.unique_exponents);
        vector(value.unique_coefficients);
        vector(value.unique_original_coefficients);
        vector(value.unique_ecp_exponents);
        vector(value.unique_ecp_coefficients);
        vector(value.unique_ecp_radial_powers);
        vector(value.erd_coefficients);
        vector(value.centers);
    }
    void matrix(const Matrix& value) {
        scalar(value.nirrep());
        for (int h = 0; h < value.nirrep(); ++h) {
            scalar(value.rowspi()[h]);
            scalar(value.colspi()[h]);
            for (int row = 0; row < value.rowspi()[h]; ++row)
                for (int column = 0; column < value.colspi()[h]; ++column)
                    scalar(value.get(h, row, column));
        }
    }
};

std::string context_digest(const FrozenResponseContext& context, const ISAOptions& options) {
    DigestBuilder digest;
    digest.string("native-real-space-isa-context-v2");
    digest.scalar(static_cast<std::uint64_t>(context.sites().size()));
    for (const auto& site : context.sites())
        for (double coordinate : site) digest.scalar(coordinate);
    digest.vector(context.grid_points());
    digest.vector(context.grid_weights());
    digest.scalar(static_cast<std::uint64_t>(context.grid_blocks().size()));
    for (const auto& block : context.grid_blocks()) {
        digest.scalar(static_cast<std::uint64_t>(block.point_offset));
        digest.scalar(static_cast<std::uint64_t>(block.point_count));
        digest.vector(block.functions_local_to_global);
    }
    digest.matrix(*context.Da());
    digest.matrix(*context.Db());
    digest.basis(context.basis()->structural_snapshot());
    digest.string("Slater-1964-bohr-v1");
    digest.scalar(context.molecule()->molecular_charge());
    digest.scalar(context.molecule()->natom());
    for (int atom = 0; atom < context.molecule()->natom(); ++atom) {
        const int atomic_number = static_cast<int>(context.molecule()->Z(atom));
        digest.scalar(atomic_number);
        digest.scalar(slater_radius(atomic_number));
    }
    digest.scalar(static_cast<std::uint64_t>(options.radial_points()));
    digest.scalar(static_cast<std::uint64_t>(options.angular_polar_points()));
    digest.scalar(static_cast<std::uint64_t>(options.angular_azimuthal_points()));
    digest.scalar(static_cast<std::uint64_t>(options.max_iterations()));
    digest.scalar(options.convergence());
    digest.scalar(options.mix_fraction());
    digest.scalar(options.initial_alpha());
    digest.scalar(options.tail_join_factor());
    digest.scalar(static_cast<std::uint64_t>(options.tail_activation_iteration()));
    digest.scalar(options.tail_activation_convergence());
    digest.scalar(options.electron_count_tolerance());
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << digest.hash;
    return stream.str();
}

}  // namespace

ISAOptions::ISAOptions(std::size_t radial_points, std::size_t angular_polar_points,
                       std::size_t angular_azimuthal_points, std::size_t max_iterations,
                       double convergence, double mix_fraction, double initial_alpha,
                       double tail_join_factor, std::size_t tail_activation_iteration,
                       double tail_activation_convergence, double electron_count_tolerance)
    : radial_points_(radial_points), angular_polar_points_(angular_polar_points),
      angular_azimuthal_points_(angular_azimuthal_points), max_iterations_(max_iterations),
      convergence_(convergence), mix_fraction_(mix_fraction), initial_alpha_(initial_alpha),
      tail_join_factor_(tail_join_factor), tail_activation_iteration_(tail_activation_iteration),
      tail_activation_convergence_(tail_activation_convergence),
      electron_count_tolerance_(electron_count_tolerance) {
    if (radial_points_ < 4 || angular_polar_points_ < 2 || angular_azimuthal_points_ < 4 ||
        max_iterations_ == 0 || tail_activation_iteration_ == 0 || !std::isfinite(convergence_) ||
        convergence_ <= 0.0 || !std::isfinite(mix_fraction_) || mix_fraction_ <= 0.0 ||
        mix_fraction_ > 1.0 || !std::isfinite(initial_alpha_) || initial_alpha_ <= 0.0 ||
        !std::isfinite(tail_join_factor_) || tail_join_factor_ <= 0.0 ||
        !std::isfinite(tail_activation_convergence_) || tail_activation_convergence_ <= 0.0 ||
        !std::isfinite(electron_count_tolerance_) || electron_count_tolerance_ < 0.0)
        throw PSIEXCEPTION("ISAOptions: grid, iteration, tail, mixing, and tolerance values are invalid");
    if (angular_polar_points_ > std::numeric_limits<std::size_t>::max() / angular_azimuthal_points_)
        throw PSIEXCEPTION("ISAOptions: angular grid cardinality overflows");
}

ISAWeights compute_isa_weights(std::shared_ptr<const FrozenResponseContext> context, const ISAOptions& options) {
    if (!context) throw PSIEXCEPTION("ISA: frozen response context is null");
    context->verify_basis_unchanged();
    const std::size_t npoint = context->grid_point_count();
    if (npoint == 0 || context->grid_points().size() != 3 * npoint || context->grid_weights().size() != npoint)
        throw PSIEXCEPTION("ISA: frozen ordered response grid is inconsistent");
    std::vector<SitePosition> points(npoint);
    for (std::size_t point = 0; point < npoint; ++point)
        points[point] = {context->grid_points()[3 * point], context->grid_points()[3 * point + 1],
                         context->grid_points()[3 * point + 2]};
    std::vector<std::vector<int>> maps(npoint);
    std::size_t offset = 0;
    for (const auto& block : context->grid_blocks()) {
        if (block.point_offset != offset || block.point_count == 0 ||
            block.point_offset + block.point_count > npoint || block.functions_local_to_global.empty())
            throw PSIEXCEPTION("ISA: frozen response-grid block coverage/map is inconsistent");
        int previous_function = -1;
        for (int function : block.functions_local_to_global) {
            if (function <= previous_function || function < 0 || function >= context->basis()->nbf())
                throw PSIEXCEPTION("ISA: frozen response-grid AO map is not strictly ordered and in range");
            previous_function = function;
        }
        for (std::size_t point = block.point_offset; point < block.point_offset + block.point_count; ++point)
            maps[point] = block.functions_local_to_global;
        offset += block.point_count;
    }
    if (offset != npoint) throw PSIEXCEPTION("ISA: frozen response-grid blocks do not cover the exact grid");
    std::vector<int> atomic_numbers;
    double formal_electrons = -context->molecule()->molecular_charge();
    for (int atom = 0; atom < context->molecule()->natom(); ++atom) {
        const double z = context->molecule()->Z(atom);
        if (!std::isfinite(z) || z != std::round(z)) throw PSIEXCEPTION("ISA: only real integer-Z nuclear sites are supported");
        atomic_numbers.push_back(static_cast<int>(z));
        formal_electrons += z;
    }
    if (atomic_numbers.size() != context->sites().size()) throw PSIEXCEPTION("ISA: molecule/site cardinality is inconsistent");
    if (!std::isfinite(formal_electrons) || formal_electrons <= 0.0)
        throw PSIEXCEPTION("ISA: formal frozen electron count is not finite and positive");
    validate_inputs(context->sites(), points, context->grid_weights(), atomic_numbers);
    auto output_density = ao_density(*context, points, &maps);
    // Reuse one AO buffer for the deterministic serial auxiliary-grid pass.
    // Matrix/basis dimensions were validated by ao_density above.
    auto* shell_basis = const_cast<BasisSet*>(context->basis().get());
    const auto shell_da = context->Da();
    const auto shell_db = context->Db();
    std::vector<double> shell_phi(static_cast<std::size_t>(shell_basis->nbf()));
    const auto evaluator = [shell_basis, shell_da, shell_db, shell_phi = std::move(shell_phi)](
                               const SitePosition& point) mutable {
        shell_basis->compute_phi(shell_phi.data(), point[0], point[1], point[2]);
        KahanSum contraction;
        for (std::size_t mu = 0; mu < shell_phi.size(); ++mu)
            for (std::size_t nu = 0; nu < shell_phi.size(); ++nu)
                contraction.add((shell_da->get(mu, nu) + shell_db->get(mu, nu)) *
                                shell_phi[mu] * shell_phi[nu]);
        return contraction.sum;
    };
    auto result = solve(context->sites(), points, context->grid_weights(), atomic_numbers, output_density,
                        evaluator, options, formal_electrons, true);
    result.diagnostics.context_digest = context_digest(*context, options);
    context->verify_basis_unchanged();
    return ISAWeights(std::move(context), std::move(result.weights), std::move(result.diagnostics));
}

namespace detail {

SyntheticISAResult compute_synthetic_isa(const std::vector<SitePosition>& sites,
                                         const std::vector<SitePosition>& output_points,
                                         const std::vector<double>& output_weights,
                                         const std::vector<int>& atomic_numbers,
                                         const std::vector<SyntheticGaussianDensity>& terms,
                                         const ISAOptions& options,
                                         std::size_t inject_tail_fit_failure_iteration) {
    if (terms.empty()) throw PSIEXCEPTION("ISA synthetic fixture: at least one Gaussian density term is required");
    const auto evaluator = [&terms](const SitePosition& point) {
        KahanSum density;
        for (const auto& term : terms) {
            if (!finite_site(term.center) || !std::isfinite(term.coefficient) || term.coefficient < 0.0 ||
                !std::isfinite(term.exponent) || term.exponent <= 0.0)
                throw PSIEXCEPTION("ISA synthetic fixture: Gaussian density parameters must be finite and positive");
            const double r = distance(point, term.center);
            density.add(term.coefficient * std::exp(-term.exponent * r * r));
        }
        return density.sum;
    };
    std::vector<double> density(output_points.size());
    for (std::size_t point = 0; point < output_points.size(); ++point) density[point] = evaluator(output_points[point]);
    KahanSum count;
    for (std::size_t point = 0; point < density.size(); ++point) count.add(output_weights[point] * density[point]);
    auto core = solve(sites, output_points, output_weights, atomic_numbers, density, evaluator,
                      options, count.sum, false, inject_tail_fit_failure_iteration);
    return {sites.size(), std::move(core.weights), std::move(core.diagnostics)};
}

ISAProfileTestResult test_isa_profile(const std::vector<double>& nodes,
                                      const std::vector<double>& log_values,
                                      const std::vector<double>& queries,
                                      double tail_join, double tail_charge) {
    Profile profile(nodes, log_values);
    const double join_log = profile.pchip(tail_join);
    profile.tail_join = tail_join;
    profile.tail_alpha = solve_tail_alpha(std::exp(join_log), tail_join, tail_charge);
    profile.tail_log_amplitude = join_log + profile.tail_alpha * tail_join;
    profile.has_tail = true;
    ISAProfileTestResult result;
    result.tail_alpha = profile.tail_alpha;
    result.tail_log_amplitude = profile.tail_log_amplitude;
    result.tail_charge = tail_charge_for_alpha(std::exp(join_log), tail_join, profile.tail_alpha);
    result.join_log_left = profile.pchip(tail_join);
    result.join_log_right = profile.tail_log_amplitude - profile.tail_alpha * tail_join;
    for (double query : queries) result.log_values.push_back(profile.eval(query));
    return result;
}

GaussianFixedPointTestResult test_isa_gaussian_fixed_point(
    const std::vector<SitePosition>& sites, const std::vector<SitePosition>& output_points,
    const std::vector<SyntheticGaussianDensity>& terms, std::size_t radial_points,
    std::size_t angular_polar_points, std::size_t angular_azimuthal_points) {
    if (sites.empty() || terms.size() != sites.size())
        throw PSIEXCEPTION("ISA Gaussian fixed-point fixture: terms must match sites");
    const auto angular = product_spherical_grid(angular_polar_points, angular_azimuthal_points);
    std::vector<Quadrature> radial;
    std::vector<Profile> profiles;
    for (std::size_t site = 0; site < sites.size(); ++site) {
        const auto& term = terms[site];
        if (!finite_site(term.center) || distance(term.center, sites[site]) != 0.0 ||
            !std::isfinite(term.coefficient) || term.coefficient <= 0.0 ||
            !std::isfinite(term.exponent) || term.exponent <= 0.0)
            throw PSIEXCEPTION("ISA Gaussian fixed-point fixture: each positive Gaussian must be site-centred");
        radial.push_back(mapped_radial(radial_points, slater_radius(1)));
        std::vector<double> logs(radial.back().nodes.size());
        for (std::size_t r = 0; r < logs.size(); ++r)
            logs[r] = std::log(term.coefficient) - term.exponent * radial.back().nodes[r] * radial.back().nodes[r];
        Profile profile(radial.back().nodes, std::move(logs));
        profile.gaussian = true;
        profile.gaussian_alpha = term.exponent;
        profile.gaussian_log_norm = std::log(term.coefficient);
        profiles.push_back(std::move(profile));
    }
    const auto density = [&terms](const SitePosition& point) {
        KahanSum value;
        for (const auto& term : terms) {
            const double radius = distance(point, term.center);
            value.add(term.coefficient * std::exp(-term.exponent * radius * radius));
        }
        return value.sum;
    };

    GaussianFixedPointTestResult result;
    for (const auto& point : output_points) {
        const auto p = probabilities(point, sites, profiles);
        result.weights.insert(result.weights.end(), p.begin(), p.end());
    }
    for (std::size_t site = 0; site < sites.size(); ++site) {
        for (std::size_t r = 0; r < radial[site].nodes.size(); ++r) {
            const double radius = radial[site].nodes[r];
            const double expected_log = std::log(terms[site].coefficient) - terms[site].exponent * radius * radius;
            if (expected_log < -600.0) continue;
            KahanSum average;
            if (r == 0) {
                const auto p = probabilities(sites[site], sites, profiles);
                average.add(density(sites[site]) * p[site]);
            } else {
                for (std::size_t q = 0; q < angular.directions.size(); ++q) {
                    const auto& direction = angular.directions[q];
                    SitePosition point{sites[site][0] + radius * direction[0],
                                       sites[site][1] + radius * direction[1],
                                       sites[site][2] + radius * direction[2]};
                    const auto p = probabilities(point, sites, profiles);
                    average.add(angular.weights[q] * density(point) * p[site]);
                }
            }
            const double expected = std::exp(expected_log);
            result.max_profile_relative_error = std::max(
                result.max_profile_relative_error, std::abs(average.sum - expected) / expected);
        }
    }
    return result;
}

ISAOverlapTestResult test_isa_overlap(
    const std::vector<double>& first_nodes, const std::vector<double>& first_logs,
    double first_tail_alpha, double first_tail_log_amplitude,
    const std::vector<double>& second_nodes, const std::vector<double>& second_logs,
    double second_tail_alpha, double second_tail_log_amplitude,
    double tail_join, std::size_t integration_points) {
    Profile first(first_nodes, first_logs);
    first.has_tail = first_tail_alpha > 0.0;
    first.tail_join = tail_join;
    first.tail_alpha = first_tail_alpha;
    first.tail_log_amplitude = first_tail_log_amplitude;
    Profile second(second_nodes, second_logs);
    second.has_tail = second_tail_alpha > 0.0;
    second.tail_join = tail_join;
    second.tail_alpha = second_tail_alpha;
    second.tail_log_amplitude = second_tail_log_amplitude;
    const auto unused_mapped_rule = mapped_radial(std::max<std::size_t>(4, integration_points), 1.0);
    return {normalized_overlap(first, second, unused_mapped_rule, 1.0, tail_join,
                               integration_points, true)};
}

std::vector<double> test_isa_tail_probabilities(const std::vector<double>& tail_log_amplitudes,
                                                const std::vector<double>& tail_alphas,
                                                const std::vector<double>& distances) {
    if (tail_log_amplitudes.empty() || tail_log_amplitudes.size() != tail_alphas.size() ||
        tail_log_amplitudes.size() != distances.size())
        throw PSIEXCEPTION("ISA tail probability fixture: dimensions are inconsistent");
    std::vector<double> scaled(distances.size());
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t site = 0; site < distances.size(); ++site) {
        Profile profile({0.0, 1.0}, {0.0, -1.0});
        profile.has_tail = true;
        profile.tail_join = 0.0;
        profile.tail_alpha = tail_alphas[site];
        profile.tail_log_amplitude = tail_log_amplitudes[site];
        scaled[site] = profile.eval(distances[site]);
        maximum = std::max(maximum, scaled[site]);
    }
    KahanSum denominator;
    for (double& value : scaled) {
        value = std::exp(value - maximum);
        denominator.add(value);
    }
    for (double& value : scaled) value /= denominator.sum;
    return scaled;
}

}  // namespace detail
}  // namespace psi
