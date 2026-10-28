/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "partitioned_response.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void property_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void property_finite(const Matrix& m) {
    property_require(m.nirrep() == 1, "Response matrix must have one symmetry block");
    for (int i = 0; i < m.nrow(); ++i) for (int j = 0; j < m.ncol(); ++j)
        property_require(std::isfinite(m.get(i,j)), "Nonfinite response or multipole matrix");
}
}
/// Row axes implied by the declared sites.
void IsaPartitionedMultipoles::declare_sites(const std::vector<IsaMultipoleSite>& sites) {
    property_require(!sites.empty(), "Partitioned multipoles require sites");
    std::set<std::string> seen;
    offsets_.push_back(0);
    for (const auto& site : sites) {
        property_require(!site.label.empty() && seen.insert(site.label).second, "Site labels must be nonempty and unique");
        property_require(site.rank >= 0 && site.rank <= 4, "Multipole rank must be between 0 and 4");
        for (double v : site.origin) property_require(std::isfinite(v), "Site origins must be finite");
        int n = (site.rank+1)*(site.rank+1);
        property_require(offsets_.back() <= std::numeric_limits<int>::max()-n, "Too many multipole components");
        offsets_.push_back(offsets_.back()+n);
        ranks_.push_back(site.rank); labels_.push_back(site.label); origins_.push_back(site.origin);
        for (int l = 0; l <= site.rank; ++l) {
            components_.push_back(std::to_string(l)+"0");
            for (int m = 1; m <= l; ++m) {
                components_.push_back(std::to_string(l)+std::to_string(m)+"c");
                components_.push_back(std::to_string(l)+std::to_string(m)+"s");
            }
        }
    }
}
IsaPartitionedMultipoles::IsaPartitionedMultipoles(const IsaExplicitBasis& auxiliary,
        const std::vector<IsaMultipoleSite>& sites, const std::string& provenance, double cutoff)
    : provenance_(provenance), cutoff_(cutoff) {
    const int columns = auxiliary.nfunction();
    property_require(auxiliary.role() == IsaBasisRole::MolecularAux,
                     "Partitioned multipoles require molecular AUX");
    property_require(!provenance.empty(), "Explicit partition provenance is required");
    property_require(std::isfinite(cutoff) && cutoff >= 0, "Invalid partition denominator cutoff");
    declare_sites(sites);
    q_ = std::make_shared<Matrix>("Partitioned molecular AUX multipoles", offsets_.back(), columns);
    for (size_t a = 0; a < sites.size(); ++a) {
        const auto& site = sites[a]; const auto& s = site.samples;
        property_require(s.points.size() == s.weights.size() && s.points.size() == s.shape.size() &&
                         s.points.size() == s.shape_sum.size(), "Partition sample dimensions disagree");
        // The explicit provider validates every point and the complete neighbour list,
        // including batches whose denominator is excluded at every point.
        property_require(s.points.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
                         "Too many partition sample points");
        size_t excluded = 0, negative = 0;
        // AUX contraction needs only a bounded collocation block. Preserve the
        // original point-order running sums; never reduce independent block sums.
        const size_t batch_size = std::min<size_t>(4096,(8*1024*1024)/sizeof(double)/columns);
        property_require(batch_size > 0, "AUX multipole row exceeds collocation scratch budget");
        for (size_t begin = 0; begin < std::max<size_t>(1,s.points.size()); begin += batch_size) {
            const size_t end = std::min(s.points.size(),begin+batch_size);
            std::vector<std::array<double,3>> block(s.points.begin()+begin,s.points.begin()+end);
            auto basis = auxiliary.evaluate_screened(block, s.auxiliary_sites);
        for (size_t p = begin; p < end; ++p) {
            property_require(std::isfinite(s.weights[p]) && std::isfinite(s.shape[p]) &&
                             std::isfinite(s.shape_sum[p]), "Nonfinite partition sample");
            if (std::abs(s.shape_sum[p]) <= cutoff) { ++excluded; continue; }
            double ratio = s.shape[p]/s.shape_sum[p];
            property_require(std::isfinite(ratio), "Nonfinite partition ratio");
            if (ratio < 0) ++negative;
            double weight = s.weights[p]*ratio;
            property_require(std::isfinite(weight), "Nonfinite partition integration weight");
            std::array<double,3> r;
            for (int d = 0; d < 3; ++d) r[d] = s.points[p][d]-site.origin[d];
            auto harmonics = isa_regular_multipoles(site.rank, r);
            for (size_t t = 0; t < harmonics.size(); ++t) {
                double prefactor = weight*harmonics[t];
                property_require(std::isfinite(prefactor), "Nonfinite weighted multipole");
                for (int k = 0; k < columns; ++k)
                    q_->add(offsets_[a]+t, k, prefactor*basis->get(p-begin, k));
            }
        }
        }
        excluded_.push_back(excluded); negative_.push_back(negative);
    }
    property_finite(*q_);
}
std::shared_ptr<Matrix> IsaPartitionedMultipoles::values() const { return q_->clone(); }
} }
