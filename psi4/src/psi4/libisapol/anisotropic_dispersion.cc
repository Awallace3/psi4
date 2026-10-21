/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 * Independent regular-harmonic differential contraction; no reference tables.
 */
#include "anisotropic_dispersion.h"
#include "multipole_transform.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace psi { namespace isapol {
namespace {
// Distinct names also avoid collisions in CMake unity translation units.
using ADPower = std::array<int, 3>;
using ADPoly = std::map<ADPower, long double>;
void ad_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
long double ad_checked(long double x) {
    ad_require(std::isfinite(x) && std::abs(x) <= std::numeric_limits<double>::max(),
               "Nonfinite or overflow anisotropic arithmetic");
    return x;
}
bool ad_nonblank(const std::string& s) { return s.find_first_not_of(" \t\n\r\f\v") != std::string::npos; }
long double ad_factorial(int n) {
    long double v = 1; for (int i = 2; i <= n; ++i) v *= i; return v;
}
long double ad_df(int l) {
    long double v = 1; for (int i = 1; i <= l; ++i) v *= 2*i-1; return v;
}
ADPoly ad_product(const ADPoly& a, const ADPoly& b) {
    ADPoly out;
    for (const auto& x : a) for (const auto& y : b) {
        ADPower p{x.first[0]+y.first[0], x.first[1]+y.first[1], x.first[2]+y.first[2]};
        if (p[0]+p[1]+p[2] <= 8) out[p] += x.second*y.second;
    }
    return out;
}
ADPoly ad_power(const ADPoly& a, int n) {
    ADPoly out{{{0,0,0}, 1.L}};
    for (int i = 0; i < n; ++i) out = ad_product(out, a);
    return out;
}
std::vector<ADPoly> ad_harmonics(int l) {
    // Rodrigues: P_l(t)=2^-l sum_j (-1)^j (2l-2j)! /
    // [j!(l-j)!(l-2j)!] t^(l-2j). Differentiate m times, then multiply
    // (x+iy)^m r^(2j) z^(l-2j-m), with real Racah/no-CS normalization.
    const ADPoly r2{{{2,0,0},1}, {{0,2,0},1}, {{0,0,2},1}};
    std::vector<ADPoly> result;
    for (int m = 0; m <= l; ++m) {
        ADPoly radial;
        for (int j = 0; 2*j <= l-m; ++j) {
            long double c = (j%2 ? -1.L : 1.L)*ad_factorial(2*l-2*j)/
                (std::pow(2.L,l)*ad_factorial(j)*ad_factorial(l-j)*ad_factorial(l-2*j-m));
            for (const auto& term : ad_power(r2,j)) {
                auto p = term.first; p[2] += l-2*j-m; radial[p] += c*term.second;
            }
        }
        ADPoly real_xy, imag_xy;
        for (int y = 0; y <= m; ++y) {
            long double c = ad_factorial(m)/(ad_factorial(y)*ad_factorial(m-y));
            if (y%2 == 0) real_xy[{m-y,y,0}] = (y%4 == 0 ? c : -c);
            else imag_xy[{m-y,y,0}] = (y%4 == 1 ? c : -c);
        }
        long double norm = std::sqrt((m ? 2.L : 1.L)*ad_factorial(l-m)/ad_factorial(l+m));
        auto real = ad_product(real_xy, radial), imag = ad_product(imag_xy, radial);
        for (auto& t : real) t.second *= norm;
        for (auto& t : imag) t.second *= norm;
        result.push_back(real);
        if (m) result.push_back(imag);
    }
    return result;
}
struct ADGeometry {
    std::array<double,3> direction;
    double distance;
};
ADGeometry ad_geometry(const std::array<double,3>& r) {
    for (double x : r) ad_require(std::isfinite(x), "Anisotropic displacement must be finite");
    // hypot avoids both squaring overflow and small-distance norm underflow.
    const long double radius = std::hypot(std::hypot(static_cast<long double>(r[0]),
                                                   static_cast<long double>(r[1])),
                                                   static_cast<long double>(r[2]));
    ad_require(radius > 0, "Coincident anisotropic sites");
    ADGeometry g; g.distance = static_cast<double>(ad_checked(radius));
    for (int d = 0; d < 3; ++d) g.direction[d] = r[d]/radius;
    return g;
}
ADPoly ad_derivatives(const std::array<double,3>& u) {
    // Taylor jet at unit direction: |u+h|^-1 =
    // s^-1/2 sum_j binomial(-1/2,j) [(2u.h+h.h)/s]^j, s=u.u.
    // Keeping s (rather than rounding it to 1) accounts for normalization ulps.
    long double s = 0; for (double x : u) s += static_cast<long double>(x)*x;
    ADPoly q;
    for (int d = 0; d < 3; ++d) {
        ADPower p{0,0,0}; p[d] = 1; q[p] = 2.L*u[d]/s;
        p[d] = 2; q[p] = 1.L/s;
    }
    ADPoly power{{{0,0,0},1}}, jet;
    long double c = 1.L/std::sqrt(s);
    for (int j = 0; j <= 8; ++j) {
        for (const auto& t : power) jet[t.first] += c*t.second;
        power = ad_product(power,q);
        c *= (-.5L-j)/(j+1);
    }
    for (auto& t : jet)
        t.second *= ad_factorial(t.first[0])*ad_factorial(t.first[1])*ad_factorial(t.first[2]);
    return jet;
}
std::vector<long double> ad_tau(int l, int k, const ADPoly& derivatives) {
    const auto a = ad_harmonics(l), b = ad_harmonics(k);
    std::vector<long double> out(a.size()*b.size());
    const long double factor = (l%2 ? -1.L : 1.L)/(ad_df(l)*ad_df(k));
    for (size_t i = 0; i < a.size(); ++i) for (size_t j = 0; j < b.size(); ++j) {
        long double v = 0;
        for (const auto& t : ad_product(a[i],b[j])) {
            const auto it = derivatives.find(t.first);
            if (it != derivatives.end()) v += t.second*it->second;
        }
        out[i*b.size()+j] = ad_checked(factor*v);
    }
    return out;
}
std::vector<int> ad_axes(const std::vector<int>& ranks) {
    ad_require(!ranks.empty() && ranks.size() <= 4, "Anisotropic ranks must be explicit increasing ranks in [1,4]");
    std::vector<int> axes;
    for (size_t i = 0; i < ranks.size(); ++i) {
        int l = ranks[i];
        ad_require(l >= 1 && l <= 4 && (!i || l > ranks[i-1]), "Anisotropic ranks must be increasing in [1,4]");
        for (int m = 0; m < 2*l+1; ++m) axes.push_back(l*l+m);
    }
    return axes;
}
std::vector<long double> ad_rotate(const Matrix& local, const Matrix& d) {
    const int n = local.nrow();
    std::vector<long double> temp(n*n), global(n*n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            temp[i*n+j] = ad_checked(temp[i*n+j]+ad_checked(static_cast<long double>(d.get(i,k))*local.get(k,j)));
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            global[i*n+j] = ad_checked(global[i*n+j]+ad_checked(temp[i*n+k]*d.get(j,k)));
    // Deliberately neither enforce exact symmetry nor average global entries.
    // Exact reciprocity is an INPUT policy; these are direct D alpha D^T values.
    return global;
}
std::vector<int> ad_component_ranks(const std::vector<int>& ranks) {
    std::vector<int> out;
    for (int l : ranks) for (int m = 0; m < 2*l+1; ++m) out.push_back(l);
    return out;
}
}
std::vector<std::string> IsaAnisotropicSite::components() const {
    ad_axes(ranks);
    std::vector<std::string> out;
    for (int l : ranks) {
        out.push_back(std::to_string(l)+"0");
        for (int m = 1; m <= l; ++m) {
            out.push_back(std::to_string(l)+std::to_string(m)+"c");
            out.push_back(std::to_string(l)+std::to_string(m)+"s");
        }
    }
    return out;
}
IsaAnisotropicModel::IsaAnisotropicModel(const std::vector<double>& frequencies,
        const std::vector<IsaAnisotropicSite>& sites, const std::string& declaration,
        const std::string& provenance) : provenance_(provenance) {
    ad_require(declaration == "supplied_local_response", "Explicit supplied_local_response declaration required; no automatic localization");
    ad_require(ad_nonblank(provenance), "Explicit anisotropic model provenance required");
    ad_require(!frequencies.empty() && frequencies.size() <= 65536, "Anisotropic frequency resource limit or empty grid");
    for (size_t f = 0; f < frequencies.size(); ++f)
        ad_require(std::isfinite(frequencies[f]) && frequencies[f] >= 0 &&
                   (!f || frequencies[f] > frequencies[f-1]), "Anisotropic frequencies must be finite, nonnegative and strictly increasing");
    ad_require(!sites.empty() && sites.size() <= 4096, "Anisotropic site resource limit or empty model");
    ad_require(sites.size() <= 65536/frequencies.size(), "Anisotropic response matrix resource limit exceeded");
    size_t elements = 0;
    std::set<std::string> labels;
    // Validate dimensions/resource use BEFORE any tensor cloning.
    for (const auto& site : sites) {
        ad_require(ad_nonblank(site.label) && labels.insert(site.label).second, "Anisotropic site labels must be nonblank and unique");
        for (double x : site.origin) ad_require(std::isfinite(x), "Anisotropic origins must be finite");
        const auto axes = ad_axes(site.ranks);
        const size_t square = axes.size()*axes.size(); // <= 576
        ad_require(frequencies.size() <= (8000000-elements)/square, "Anisotropic tensor resource limit exceeded");
        elements += square*frequencies.size();
        ad_require(site.responses.size() == frequencies.size(), "Anisotropic response frequency dimensions disagree");
        for (const auto& m : site.responses) {
            ad_require(m != nullptr, "Null anisotropic response");
            ad_require(m->nirrep() == 1, "Anisotropic response requires one symmetry block");
            ad_require(m->nrow() == static_cast<int>(axes.size()) && m->ncol() == static_cast<int>(axes.size()),
                       "Anisotropic response rank dimensions disagree; all blocks required");
            for (int i = 0; i < m->nrow(); ++i) for (int j = 0; j < m->ncol(); ++j) {
                ad_require(std::isfinite(m->get(i,j)), "Nonfinite local anisotropic response");
                ad_require(m->get(i,j) == m->get(j,i), "Local anisotropic responses must be exactly symmetric");
            }
        }
        auto full = isa_multipole_rotation(site.ranks.back(), site.frame);
        auto d = std::make_shared<Matrix>("Declared-rank local-to-global rotation", axes.size(), axes.size());
        for (size_t i = 0; i < axes.size(); ++i) for (size_t j = 0; j < axes.size(); ++j)
            d->set(i,j,full->get(axes[i],axes[j]));
        rotations_.push_back(d);
    }
    frequencies_ = frequencies;
    for (const auto& site : sites) {
        sites_.push_back(site);
        for (auto& m : sites_.back().responses) m = m->clone();
    }
}
std::vector<IsaAnisotropicSite> IsaAnisotropicModel::sites() const {
    auto result = sites_;
    for (auto& s : result) for (auto& m : s.responses) m = m->clone();
    return result;
}
std::shared_ptr<Matrix> isa_anisotropic_interaction(int l, int k, const std::array<double,3>& r) {
    ad_require(l >= 1 && l <= 4 && k >= 1 && k <= 4, "Anisotropic interaction ranks must be in [1,4]");
    auto g = ad_geometry(r);
    auto tau = ad_tau(l,k,ad_derivatives(g.direction));
    // Repeated inverse powers, never R^n in double (which could spuriously overflow).
    long double scale = 1;
    for (int i = 0; i < l+k+1; ++i) scale = ad_checked(scale/g.distance);
    auto out = std::make_shared<Matrix>("Physical real Racah Coulomb interaction",2*l+1,2*k+1);
    for (int i = 0; i < 2*l+1; ++i) for (int j = 0; j < 2*k+1; ++j)
        out->set(i,j,static_cast<double>(ad_checked(tau[i*(2*k+1)+j]*scale)));
    return out;
}
IsaAnisotropicDispersionResult isa_anisotropic_dispersion(const IsaAnisotropicModel& a,
        const IsaAnisotropicModel& b, const std::vector<double>& weights, int max_order) {
    ad_require(max_order >= 6 && max_order <= 12, "Anisotropic maximum order must be in [6,12]");
    ad_require(a.frequencies_ == b.frequencies_, "A/B frequency grids must match exactly");
    ad_require(weights.size() == a.frequencies_.size(), "Anisotropic CP weight dimensions disagree");
    bool positive = false;
    for (size_t f = 0; f < weights.size(); ++f) {
        ad_require(std::isfinite(weights[f]) && weights[f] >= 0, "Anisotropic CP weights must be finite and nonnegative");
        ad_require(a.frequencies_[f] != 0 || weights[f] == 0, "Static frequency must have zero integration weight");
        positive = positive || weights[f] > 0;
    }
    ad_require(positive, "At least one positive integration weight required");
    ad_require(a.sites_.size() <= 4096/b.sites_.size(), "Anisotropic pair resource limit exceeded");
    IsaAnisotropicDispersionResult result(a,b);
    result.frequencies = a.frequencies_; result.cp_weights = weights; result.max_order = max_order;
    long double total_energy = 0;
    for (size_t ia = 0; ia < a.sites_.size(); ++ia) for (size_t ib = 0; ib < b.sites_.size(); ++ib) {
        const auto& sa = a.sites_[ia]; const auto& sb = b.sites_[ib];
        const auto ra = ad_component_ranks(sa.ranks), rb = ad_component_ranks(sb.ranks);
        const int na = ra.size(), nb = rb.size();
        IsaAnisotropicPair pair; pair.site_a = ia; pair.site_b = ib;
        for (int d = 0; d < 3; ++d)
            pair.displacement[d] = static_cast<double>(ad_checked(static_cast<long double>(sb.origin[d])-sa.origin[d]));
        auto g = ad_geometry(pair.displacement);
        pair.distance = g.distance; pair.direction = g.direction;
        auto derivatives = ad_derivatives(g.direction);
        std::vector<long double> tau(na*nb);
        int oa = 0;
        for (int l : sa.ranks) {
            int ob = 0;
            for (int k : sb.ranks) {
                auto block = ad_tau(l,k,derivatives);
                for (int i = 0; i < 2*l+1; ++i) for (int j = 0; j < 2*k+1; ++j)
                    tau[(oa+i)*nb+ob+j] = block[i*(2*k+1)+j];
                ob += 2*k+1;
            }
            oa += 2*l+1;
        }
        for (int n = 6; n <= max_order; ++n) {
            IsaAnisotropicCoefficient c; c.order = n;
            for (int l = 1; l <= n-5; ++l) for (int lp = 1; lp <= n-5; ++lp)
                for (int k = 1; k <= n-5; ++k) {
                    int kp = n-2-l-lp-k; if (kp < 1) continue;
                    bool present = std::binary_search(sa.ranks.begin(),sa.ranks.end(),l) &&
                        std::binary_search(sa.ranks.begin(),sa.ranks.end(),lp) &&
                        std::binary_search(sb.ranks.begin(),sb.ranks.end(),k) &&
                        std::binary_search(sb.ranks.begin(),sb.ranks.end(),kp);
                    (present ? c.included_rank_quadruples : c.missing_rank_quadruples).push_back({l,lp,k,kp});
                }
            c.unrestricted_complete = c.missing_rank_quadruples.empty();
            pair.coefficients.push_back(c);
        }
        std::array<long double,13> sums{};
        for (size_t f = 0; f < weights.size(); ++f) {
            if (weights[f] == 0) continue; // also avoid rotation overflow at excluded static nodes
            const auto aa = ad_rotate(*sa.responses[f],*a.rotations_[ia]);
            const auto bb = ad_rotate(*sb.responses[f],*b.rotations_[ib]);
            // Ordered component quadruples, once each. Never double off-diagonals.
            // Scratch stays O(na^2+nb^2+na*nb), independent of frequency count.
            for (int i = 0; i < na; ++i) for (int ip = 0; ip < na; ++ip) {
                if (aa[i*na+ip] == 0) continue;
                for (int j = 0; j < nb; ++j) for (int jp = 0; jp < nb; ++jp) {
                    int n = ra[i]+ra[ip]+rb[j]+rb[jp]+2;
                    if (n > max_order || bb[j*nb+jp] == 0 || tau[i*nb+j] == 0 || tau[ip*nb+jp] == 0) continue;
                    long double term = ad_checked(static_cast<long double>(weights[f])*aa[i*na+ip]);
                    term = ad_checked(term*bb[j*nb+jp]);
                    term = ad_checked(term*tau[i*nb+j]);
                    term = ad_checked(term*tau[ip*nb+jp]);
                    sums[n] = ad_checked(sums[n]+term);
                }
            }
        }
        long double pair_energy = 0;
        for (auto& c : pair.coefficients) {
            c.value = static_cast<double>(sums[c.order]);
            long double energy = -sums[c.order];
            for (int i = 0; i < c.order; ++i) energy = ad_checked(energy/g.distance);
            c.energy = static_cast<double>(energy);
            pair_energy = ad_checked(pair_energy+energy);
        }
        pair.truncated_energy = static_cast<double>(pair_energy);
        total_energy = ad_checked(total_energy+pair_energy);
        result.pairs.push_back(std::move(pair));
    }
    result.truncated_energy = static_cast<double>(total_energy);
    return result;
}
} }
