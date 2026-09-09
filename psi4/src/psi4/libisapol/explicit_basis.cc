/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 * CamCASP-compatible conventions; independent solid-harmonic recurrence.
 */
#include "explicit_basis.h"
#include "isa_fit.h"
#include "parallel_work.h"
#include "psi4/libmints/matrix.h"
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void basis_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
int count(int l, IsaBasisRepresentation r) {
    return r == IsaBasisRepresentation::Cartesian ? (l + 1) * (l + 2) / 2 : 2 * l + 1;
}
double df(int n) { double v = 1; for (; n > 0; n -= 2) v *= n; return v; }
double factorial(int n) { double v = 1; for (; n > 1; --n) v *= n; return v; }
const std::vector<std::array<int, 3>>& basis_cartesian_powers(int l) {
    // GAMINT component order, not Psi4 Cartesian order.
    static const std::vector<std::array<int, 3>> powers[] = {
        {{0,0,0}}, {{1,0,0},{0,1,0},{0,0,1}},
        {{2,0,0},{0,2,0},{0,0,2},{1,1,0},{1,0,1},{0,1,1}},
        {{3,0,0},{0,3,0},{0,0,3},{2,1,0},{2,0,1},{1,2,0},{0,2,1},{1,0,2},{0,1,2},{1,1,1}},
        {{4,0,0},{0,4,0},{0,0,4},{3,1,0},{3,0,1},{1,3,0},{0,3,1},{1,0,3},{0,1,3},
         {2,2,0},{2,0,2},{0,2,2},{2,1,1},{1,2,1},{1,1,2}}};
    return powers[l];
}
double basis_cartesian_factor(int l, const std::array<int, 3>& p) {
    return std::sqrt(df(2*l-1)/(df(2*p[0]-1)*df(2*p[1]-1)*df(2*p[2]-1)));
}
std::array<double,15> angular(int l, IsaBasisRepresentation rep, double x, double y, double z) {
    // Both representations' l=0 recurrence is exactly the constant one.
    if (l == 0) return {1.0};
    std::array<double,15> out{};
    int index = 0;
    if (rep == IsaBasisRepresentation::Cartesian) {
        for (auto p : basis_cartesian_powers(l))
            out[index++] = basis_cartesian_factor(l,p) *
                          std::pow(x,p[0])*std::pow(y,p[1])*std::pow(z,p[2]);
        return out;
    }
    // Unnormalized regular associated harmonics, with no Condon--Shortley phase:
    // H_mm=(2m-1)!!(x+iy)^m; H_(m+1,m)=(2m+1)z H_mm;
    // (l-m)H_lm=(2l-1)z H_(l-1,m)-(l+m-1)r^2 H_(l-2,m).
    std::complex<double> h[5];
    const double r2 = x*x+y*y+z*z;
    for (int m = 0; m <= l; ++m) {
        auto prev = df(2*m-1)*std::pow(std::complex<double>(x,y),m);
        auto value = prev;
        if (l > m) {
            value = double(2*m+1)*z*prev;
            for (int n = m+2; n <= l; ++n) {
                auto next = (double(2*n-1)*z*value-double(n+m-1)*r2*prev)/double(n-m);
                prev = value; value = next;
            }
        }
        h[m] = value*std::sqrt((m ? 2.0 : 1.0)*factorial(l-m)/factorial(l+m));
    }
    if (l == 1) return {h[1].real(),h[1].imag(),h[0].real()};
    for (int m = l; m > 0; --m) out[index++] = h[m].imag();
    out[index++] = h[0].real();
    for (int m = 1; m <= l; ++m) out[index++] = h[m].real();
    return out;
}
}
std::vector<double> isa_regular_multipoles(int rank, const std::array<double,3>& r) {
    basis_require(rank >= 0 && rank <= 4, "Multipole rank must be between 0 and 4");
    for (double v : r) basis_require(std::isfinite(v), "Multipole displacement must be finite");
    std::vector<double> result;
    for (int l = 0; l <= rank; ++l) {
        auto a = angular(l, IsaBasisRepresentation::Spherical, r[0], r[1], r[2]);
        if (l == 1) {
            result.insert(result.end(), {a[2], a[0], a[1]});
        } else {
            result.push_back(a[l]);
            for (int m = 1; m <= l; ++m) {
                result.push_back(a[l+m]);
                result.push_back(a[l-m]);
            }
        }
    }
    for (double v : result) basis_require(std::isfinite(v), "Nonfinite regular multipole");
    return result;
}
const std::vector<std::array<int,3>>& IsaExplicitBasis::cartesian_powers(int l) {
    return basis_cartesian_powers(l);
}
IsaExplicitBasis::IsaExplicitBasis(IsaBasisRole role, IsaBasisRepresentation representation,
        const std::vector<std::array<double,3>>& centres, const std::vector<IsaGaussianShell>& shells)
    : role_(role), representation_(representation), centres_(centres), shells_(shells) {
    basis_require(role == IsaBasisRole::MolecularAux || role == IsaBasisRole::AtomAux || role == IsaBasisRole::Shape || role == IsaBasisRole::Orbital,
            "Invalid ISA basis role");
    basis_require(representation == IsaBasisRepresentation::Cartesian || representation == IsaBasisRepresentation::Spherical,
            "Invalid ISA basis representation");
    basis_require(role != IsaBasisRole::Orbital || representation == IsaBasisRepresentation::Spherical,
                  "Orbital basis currently requires DALTON spherical representation");
    basis_require(!centres.empty() && centres.size() <= std::numeric_limits<int>::max(), "Basis requires centres");
    basis_require(!shells.empty(), "Basis requires shells");
    for (auto c : centres) for (double v : c) basis_require(std::isfinite(v), "Basis centres must be finite");
    for (const auto& s : shells) {
        basis_require(s.centre >= 0 && s.centre < ncentre(), "Shell centre out of range");
        basis_require(s.l >= 0 && s.l <= 4, "Only S through G shells supported");
        basis_require(role != IsaBasisRole::Shape || s.l == 0, "Shape basis requires s shells");
        basis_require(!s.exponents.empty() && s.exponents.size() == s.coefficients.size(), "Invalid shell primitive dimensions");
        for (double a : s.exponents) basis_require(std::isfinite(a) && a > 0, "Exponents must be finite and positive");
        for (double c : s.coefficients) basis_require(std::isfinite(c), "Effective coefficients must be finite");
        basis_require(nfunction_ <= std::numeric_limits<int>::max()-count(s.l,representation), "Too many functions");
        nfunction_ += count(s.l,representation);
    }
}
std::shared_ptr<Matrix> IsaExplicitBasis::evaluate(const std::vector<std::array<double,3>>& points) const {
    std::vector<int> sites(ncentre());
    std::iota(sites.begin(),sites.end(),0);
    return evaluate_screened(points,sites);
}
std::shared_ptr<Matrix> IsaExplicitBasis::evaluate_screened(const std::vector<std::array<double,3>>& points,
                                                          const std::vector<int>& sites) const {
    basis_require(points.size() <= std::numeric_limits<int>::max(), "Too many points");
    std::vector<bool> active(ncentre(),false);
    for (int site : sites) {
        basis_require(site >= 0 && site < ncentre(), "Neighbour site out of range (zero-based, no padding)");
        basis_require(!active[site], "Duplicate neighbour site");
        active[site] = true;
    }
    for (auto p : points) for (double v : p) basis_require(std::isfinite(v), "Points must be finite");
    basis_require(points.size() <= std::numeric_limits<size_t>::max()/sizeof(double)/static_cast<size_t>(nfunction_),
                  "Explicit basis sample byte size overflow");
    auto result = std::make_shared<Matrix>("Explicit ISA basis samples", static_cast<int>(points.size()),nfunction_);
    std::vector<int> starts;
    int column = 0;
    for (const auto& s : shells_) {
        starts.push_back(column);
        column += count(s.l,representation_);
    }
    auto output = result->pointer();
    detail::parallel_work(points.size(),256,[&](size_t i) {
        for (size_t shell = 0; shell < shells_.size(); ++shell) {
            const auto& s = shells_[shell];
            if (!active[s.centre]) continue;
            const int n = count(s.l,representation_);
            const int column = starts[shell];
            double x = points[i][0]-centres_[s.centre][0], y = points[i][1]-centres_[s.centre][1],
                   z = points[i][2]-centres_[s.centre][2];
            const double r2 = x*x+y*y+z*z;
            basis_require(std::isfinite(r2), "Nonfinite squared distance");
            double radial = 0;
            for (size_t p = 0; p < s.exponents.size(); ++p)
                radial += s.coefficients[p]*std::exp(-s.exponents[p]*r2);
            auto values = angular(s.l,representation_,x,y,z);
            for (int k = 0; k < n; ++k) {
                double value = radial*values[k];
                basis_require(std::isfinite(value), "Nonfinite basis sample");
                output[i][column+k] = value;
            }
        }
    });
    return result;
}
std::shared_ptr<Matrix> IsaExplicitBasis::overlap(double w_eps, bool s_block_only) const {
    basis_require(role_ == IsaBasisRole::AtomAux || role_ == IsaBasisRole::Shape,
                  "Atomic overlap requires AtomAux or Shape, not molecular AUX or Orbital");
    basis_require(std::isfinite(w_eps) && w_eps >= 0, "W-Eps must be finite and nonnegative");
    const auto& centre = centres_[shells_[0].centre];
    for (const auto& shell : shells_)
        basis_require(centres_[shell.centre] == centre, "Atomic overlap requires co-centred shells");
    auto metric = std::make_shared<Matrix>("Analytic ISA atomic overlap",nfunction_,nfunction_);
    const double pi = std::acos(-1.0);
    int row = 0;
    for (size_t a = 0; a < shells_.size(); ++a) {
        const auto& sa = shells_[a];
        const int na = count(sa.l,representation_);
        int col = 0;
        for (size_t b = 0; b <= a; ++b) {
            const auto& sb = shells_[b];
            const int nb = count(sb.l,representation_);
            const double shift = (!s_block_only || (sa.l == 0 && sb.l == 0)) ? w_eps : 0.0;
            // Validate even angularly orthogonal or zero-coefficient pairs. No
            // conditional angular cancellation is accepted for a divergent radial integral.
            long double radial = 0.0L;
            const double power = 0.5*(sa.l+sb.l)+1.5;
            for (size_t p = 0; p < sa.exponents.size(); ++p)
                for (size_t q = 0; q < sb.exponents.size(); ++q) {
                    const double beta = sa.exponents[p]+sb.exponents[q]-shift;
                    basis_require(std::isfinite(beta) && beta > 0, "Nonintegrable or nonfinite weighted primitive overlap");
                    radial += static_cast<long double>(sa.coefficients[p])*sb.coefficients[q] /
                              std::pow(static_cast<long double>(beta),power);
                }
            basis_require(std::isfinite(radial), "Nonfinite contracted overlap");
            for (int i = 0; i < na; ++i) for (int j = 0; j < nb; ++j) {
                if (a == b && j > i) continue;
                double angular_factor = 0.0;
                if (representation_ == IsaBasisRepresentation::Spherical) {
                    // Real Racah-normalized harmonics are angularly orthogonal.
                    // Gaussian radial integral gives (2l-1)!! pi^(3/2) / 2^l.
                    if (sa.l == sb.l && i == j)
                        angular_factor = df(2*sa.l-1)*std::pow(pi,1.5)/std::pow(2.0,sa.l);
                } else {
                    const auto& pa = basis_cartesian_powers(sa.l)[i];
                    const auto& pb = basis_cartesian_powers(sb.l)[j];
                    angular_factor = basis_cartesian_factor(sa.l,pa)*basis_cartesian_factor(sb.l,pb);
                    for (int axis = 0; axis < 3; ++axis) {
                        const int n = pa[axis]+pb[axis];
                        // Integral x^(2k) exp(-beta*x*x) dx = Gamma(k+1/2)/beta^(k+1/2).
                        if (n % 2) { angular_factor = 0.0; break; }
                        angular_factor *= std::tgamma(0.5*(n+1));
                    }
                }
                const double value = static_cast<double>(radial*angular_factor);
                basis_require(std::isfinite(value), "Nonfinite atomic overlap");
                metric->set(row+i,col+j,value);
                metric->set(col+j,row+i,value);
            }
            col += nb;
        }
        row += na;
    }
    return metric;
}
IsaShapeMap::IsaShapeMap(const IsaExplicitBasis& atomic, const IsaExplicitBasis& shape,
                         const std::vector<int>& shell_map) : atomic_nfunction_(atomic.nfunction()) {
    basis_require(atomic.role_ == IsaBasisRole::AtomAux && shape.role_ == IsaBasisRole::Shape,
                  "Shape map requires AtomAux and Shape roles");
    basis_require(shell_map.size() == shape.shells_.size(), "Shape map dimension mismatch (no padding)");
    std::vector<int> starts;
    int column = 0;
    for (const auto& s : atomic.shells_) {
        starts.push_back(column);
        column += count(s.l,atomic.representation_);
    }
    std::vector<bool> used(atomic.shells_.size(),false);
    for (size_t i = 0; i < shell_map.size(); ++i) {
        const int target = shell_map[i];
        basis_require(target >= 0 && static_cast<size_t>(target) < atomic.shells_.size(), "Shape map target out of range");
        basis_require(!used[target], "Duplicate shape map target");
        used[target] = true;
        const auto& a = atomic.shells_[target];
        const auto& s = shape.shells_[i];
        basis_require(a.l == 0, "Shape map target must be an s shell");
        basis_require(atomic.centres_[a.centre] == shape.centres_[s.centre] &&
                      a.exponents == s.exponents && a.coefficients == s.coefficients,
                      "Shape map requires exactly matching shell descriptors");
        columns_.push_back(starts[target]);
    }
}
std::vector<double> IsaShapeMap::project(const std::vector<double>& atomic_coefficients) const {
    basis_require(atomic_coefficients.size() == static_cast<size_t>(atomic_nfunction_), "Atomic coefficient dimension mismatch");
    for (double c : atomic_coefficients) basis_require(std::isfinite(c), "Atomic coefficients must be finite");
    std::vector<double> result;
    for (int column : columns_) result.push_back(atomic_coefficients[column]);
    return result;
}
IsaFixedDensity::IsaFixedDensity(const IsaExplicitBasis& basis, const std::vector<double>& coefficients)
    : basis_(basis), coefficients_(coefficients) {
    basis_require(basis.role() == IsaBasisRole::MolecularAux, "Fixed density requires molecular AUX, not AtomAux/shape");
    basis_require(coefficients.size() == static_cast<size_t>(basis.nfunction()), "Density coefficient dimension mismatch");
    for (double c : coefficients) basis_require(std::isfinite(c), "Density coefficients must be finite");
}
std::vector<double> IsaFixedDensity::evaluate(const std::vector<std::array<double,3>>& points,
                                             const std::vector<int>& sites) const {
    // Bound AUX scratch independently of the molecular point count. Even
    // screened zero columns participate in the original ordered k contraction.
    basis_require(points.size() <= static_cast<size_t>(std::numeric_limits<int>::max()), "Too many density points");
    constexpr size_t scratch_bytes = 8 * 1024 * 1024;
    const size_t row_bytes = sizeof(double)*static_cast<size_t>(basis_.nfunction());
    basis_require(row_bytes <= scratch_bytes, "Density sample row exceeds 8 MiB scratch budget");
    const size_t blocksize = std::min(size_t(4096),scratch_bytes/row_bytes);
    std::vector<double> density(points.size(),0.0);
    // Validate the screening policy even for an empty point set.
    if (points.empty()) basis_.evaluate_screened({},sites);
    for (size_t start = 0; start < points.size(); start += blocksize) {
        const size_t count = std::min(blocksize,points.size()-start);
        std::vector<std::array<double,3>> block(points.begin()+start,points.begin()+start+count);
        auto samples = basis_.evaluate_screened(block,sites);
        auto phi = samples->pointer();
        detail::parallel_work(count,256,[&](size_t i) {
            for (int k = 0; k < basis_.nfunction(); ++k) density[start+i] += phi[i][k]*coefficients_[k];
            basis_require(std::isfinite(density[start+i]), "Nonfinite density sample");
        });
    }
    return density;
}
IsaAFitProvider::IsaAFitProvider(const IsaExplicitBasis& atomic, const IsaFixedDensity& density)
    : atomic_(atomic), density_(density), centre_(atomic.centres_[atomic.shells_[0].centre]) {
    basis_require(atomic.role_ == IsaBasisRole::AtomAux, "Fit provider requires AtomAux");
    for (const auto& shell : atomic.shells_) {
        basis_require(atomic.centres_[shell.centre] == centre_, "Fit provider requires co-centred AtomAux");
        basis_require(shell.exponents.size() == 1, "Fit provider requires primitive (uncontracted) AtomAux shells");
        const int n = count(shell.l,atomic.representation_);
        angular_.insert(angular_.end(),n,shell.l);
        exponents_.insert(exponents_.end(),n,shell.exponents[0]);
    }
}
IsaAFitData IsaAFitProvider::assemble(const IsaAFitSamples& samples, const IsaAFitOptions& options) const {
    const size_t n = samples.points.size();
    basis_require(n > 0 && n <= static_cast<size_t>(std::numeric_limits<int>::max()), "Fit samples require a nonempty bounded point set");
    auto check = [](const std::vector<double>& v, size_t size) {
        basis_require(v.size() == size, "Fit sample/vector dimension mismatch");
        for (double x : v) basis_require(std::isfinite(x), "Fit sample/vector values must be finite");
    };
    check(samples.weights,n); check(samples.shape,n); check(samples.shape_sum,n);
    check(samples.previous,atomic_.nfunction());
    for (double v : {options.w_eps,options.damping,options.positive_lambda,
                     options.positive_max_alpha,options.density_cutoff})
        basis_require(std::isfinite(v) && v >= 0, "Fit options must be finite and nonnegative");
    IsaAFitData data;
    data.weights = samples.weights; data.shape = samples.shape; data.shape_sum = samples.shape_sum;
    data.previous = samples.previous; data.angular_momenta = angular_; data.exponents = exponents_;
    data.radius_squared.reserve(n);
    for (auto p : samples.points) {
        double r2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            basis_require(std::isfinite(p[axis]), "Fit points must be finite");
            const double d = p[axis]-centre_[axis];
            r2 += d*d;
        }
        basis_require(std::isfinite(r2), "Nonfinite fit squared distance");
        data.radius_squared.push_back(r2);
    }
    data.overlap = atomic_.overlap(options.w_eps,options.s_block_only);
    data.density = density_.evaluate(samples.points,samples.density_sites);
    data.basis_values = atomic_.evaluate(samples.points);
    return data;
}
IsaAFitResult IsaAFitProvider::fit(const IsaAFitSamples& samples, const IsaAFitOptions& options) const {
    return isa_a_fit_step(assemble(samples,options),options);
}
} }
