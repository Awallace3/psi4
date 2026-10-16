/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "isotropic_dispersion.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void isotropic_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
double isotropic_binomial(int n, int k) {
    int v = 1;
    for (int i = 1; i <= k; ++i) v = v*(n-k+i)/i;
    return v;
}
}
IsaIsotropicModel::IsaIsotropicModel(const std::vector<double>& frequencies,
        const std::vector<IsaIsotropicSite>& sites, const std::string& provenance)
    : frequencies_(frequencies), provenance_(provenance) {
    isotropic_require(!provenance.empty(), "Explicit isotropic model provenance required");
    isotropic_require(!frequencies.empty() && frequencies.size() <= std::numeric_limits<int>::max(),
                      "Isotropic model requires frequencies");
    for (size_t f = 0; f < frequencies.size(); ++f)
        isotropic_require(std::isfinite(frequencies[f]) && frequencies[f] >= 0 &&
                          (f == 0 || frequencies[f] > frequencies[f-1]),
                          "Isotropic frequencies must be finite, nonnegative and strictly increasing");
    isotropic_require(!sites.empty() && sites.size() <= std::numeric_limits<int>::max(), "Isotropic model requires sites");
    std::set<std::string> labels;
    for (const auto& site : sites) {
        isotropic_require(!site.label.empty() && labels.insert(site.label).second, "Isotropic site labels must be nonempty and unique");
        for (double r : site.origin) isotropic_require(std::isfinite(r), "Isotropic site origins must be finite");
        isotropic_require(!site.ranks.empty(), "Isotropic site requires explicit rank coverage");
        for (size_t l = 0; l < site.ranks.size(); ++l)
            isotropic_require(site.ranks[l] >= 1 && site.ranks[l] <= 4 &&
                              (l == 0 || site.ranks[l] > site.ranks[l-1]), "Isotropic ranks must be increasing in [1,4]");
        isotropic_require(site.polarizabilities != nullptr, "Null isotropic polarizability matrix");
        const auto& m = *site.polarizabilities;
        isotropic_require(m.nirrep() == 1, "Isotropic polarizabilities require one symmetry block");
        isotropic_require(m.nrow() == static_cast<int>(frequencies.size()) && m.ncol() == static_cast<int>(site.ranks.size()),
                          "Isotropic polarizability frequency/rank dimensions disagree");
        for (int i = 0; i < m.nrow(); ++i) for (int j = 0; j < m.ncol(); ++j)
            isotropic_require(std::isfinite(m.get(i,j)), "Nonfinite isotropic polarizability");
        sites_.push_back(site);
        sites_.back().polarizabilities = site.polarizabilities->clone();
    }
}
std::vector<IsaIsotropicSite> IsaIsotropicModel::sites() const {
    auto result = sites_;
    for (auto& site : result) site.polarizabilities = site.polarizabilities->clone();
    return result;
}
IsaIsotropicDispersionResult isa_isotropic_dispersion(const IsaIsotropicModel& a,
        const IsaIsotropicModel& b, const std::vector<double>& weights, int max_order) {
    isotropic_require(max_order >= 6 && max_order <= 12 && max_order%2 == 0, "Isotropic maximum order must be 6,8,10 or12");
    auto frequencies = a.frequencies();
    isotropic_require(frequencies == b.frequencies(), "A/B frequency grids must match exactly; no implicit interpolation");
    isotropic_require(weights.size() == frequencies.size(), "Casimir-Polder weight dimensions disagree");
    bool positive = false;
    for (size_t f = 0; f < weights.size(); ++f) {
        isotropic_require(std::isfinite(weights[f]) && weights[f] >= 0, "Casimir-Polder weights must be finite and nonnegative");
        isotropic_require(frequencies[f] != 0 || weights[f] == 0, "Static frequency must have zero integration weight");
        positive = positive || weights[f] > 0;
    }
    isotropic_require(positive, "At least one positive integration weight required");
    auto sites_a = a.sites(), sites_b = b.sites();
    IsaIsotropicDispersionResult result;
    result.frequencies = frequencies; result.cp_weights = weights;
    result.provenance_a = a.provenance(); result.provenance_b = b.provenance();
    for (const auto& s : sites_a) { result.labels_a.push_back(s.label); result.origins_a.push_back(s.origin); }
    for (const auto& s : sites_b) { result.labels_b.push_back(s.label); result.origins_b.push_back(s.origin); }
    for (size_t ia = 0; ia < sites_a.size(); ++ia) for (size_t ib = 0; ib < sites_b.size(); ++ib) {
        const auto& sa = sites_a[ia]; const auto& sb = sites_b[ib];
        IsaIsotropicPair pair; pair.site_a = ia; pair.site_b = ib;
        for (int n = 6; n <= max_order; n += 2) {
            IsaIsotropicCoefficient coefficient; coefficient.order = n;
            for (int la = 1; la < n/2-1; ++la) {
                const int lb = n/2-1-la;
                auto ca = std::find(sa.ranks.begin(), sa.ranks.end(), la);
                auto cb = std::find(sb.ranks.begin(), sb.ranks.end(), lb);
                if (ca == sa.ranks.end() || cb == sb.ranks.end()) {
                    coefficient.missing_rank_pairs.push_back({la,lb});
                    continue;
                }
                coefficient.included_rank_pairs.push_back({la,lb});
                double integral = 0.;
                for (size_t f = 0; f < frequencies.size(); ++f) {
                    if (weights[f] == 0.) continue; // static values are not part of the integral
                    double term = weights[f]*sa.polarizabilities->get(f, ca-sa.ranks.begin())*
                                  sb.polarizabilities->get(f, cb-sb.ranks.begin());
                    isotropic_require(std::isfinite(term), "Nonfinite isotropic quadrature product");
                    integral += term;
                    isotropic_require(std::isfinite(integral), "Nonfinite isotropic quadrature sum");
                }
                coefficient.value += isotropic_binomial(2*la+2*lb, 2*la)*integral;
                isotropic_require(std::isfinite(coefficient.value), "Nonfinite isotropic dispersion coefficient");
            }
            coefficient.complete = coefficient.missing_rank_pairs.empty();
            pair.coefficients.push_back(coefficient);
        }
        result.pairs.push_back(pair);
    }
    return result;
}
} }
