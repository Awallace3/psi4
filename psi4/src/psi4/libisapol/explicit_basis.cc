/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 * CamCASP-compatible conventions; independent solid-harmonic recurrence.
 */
#include "explicit_basis.h"
#include "parallel_work.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
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
std::shared_ptr<Matrix> IsaExplicitBasis::screening_s_overlap(std::size_t max_bytes) const {
    const auto n = shells_.size();
    basis_require(max_bytes > 0 &&
                  n > 0 && n <= max_bytes/sizeof(double)/n,
                  "Shell screening matrix byte resource limit");
    auto result = std::make_shared<Matrix>("Signed shell normalized-s surrogate",
                                          static_cast<int>(n), static_cast<int>(n));
    for (std::size_t i=0; i<n; ++i) {
        const auto& a = shells_[i];
        for (std::size_t j=i; j<n; ++j) {
            const auto& b = shells_[j];
            double r2 = 0.;
            for (int xyz=0; xyz<3; ++xyz) {
                const double d = centres_[a.centre][xyz]-centres_[b.centre][xyz];
                r2 += d*d;
            }
            basis_require(std::isfinite(r2), "Nonfinite shell screening distance");
            double value = 0.;
            for (std::size_t p=0; p<a.exponents.size(); ++p)
                for (std::size_t q=0; q<b.exponents.size(); ++q) {
                    const double alpha = a.exponents[p], beta = b.exponents[q];
                    const double sum = alpha+beta;
                    basis_require(std::isfinite(sum), "Nonfinite shell screening exponent sum");
                    const double overlap = std::pow(4.*(alpha/sum)*(beta/sum), .75) *
                                           std::exp(-(alpha/sum)*beta*r2);
                    value += a.coefficients[p]*b.coefficients[q]*overlap;
                    basis_require(std::isfinite(value), "Nonfinite shell screening contraction");
                }
            result->set(i,j,value);
            result->set(j,i,value);
        }
    }
    return result;
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
int IsaExplicitBasis::shell_size(int l, IsaBasisRepresentation representation) {
    return count(l,representation);
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
        // The angular factor depends only on (centre, l): evaluate it once per
        // point for each, instead of once per shell. Same arithmetic, same bits.
        std::array<double,15> values[5];
        int cached_centre = -1;
        bool cached[5] = {};
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
            if (s.centre != cached_centre) {
                cached_centre = s.centre;
                std::fill(std::begin(cached),std::end(cached),false);
            }
            if (!cached[s.l]) {
                values[s.l] = angular(s.l,representation_,x,y,z);
                cached[s.l] = true;
            }
            for (int k = 0; k < n; ++k) {
                double value = radial*values[s.l][k];
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
} }
