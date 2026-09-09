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
IsaPartitionedMultipoles::IsaPartitionedMultipoles(const IsaExplicitBasis& auxiliary,
        const std::vector<IsaMultipoleSite>& sites, const std::string& provenance, double cutoff)
    : IsaPartitionedMultipoles(auxiliary, sites, provenance, cutoff, nullptr, 0) {}
IsaPartitionedMultipoles::IsaPartitionedMultipoles(const IsaExplicitBasis& auxiliary,
        const std::vector<IsaMultipoleSite>& sites, const std::string& provenance, double cutoff,
        std::shared_ptr<Matrix> orbitals, int nocc)
    : provenance_(provenance), cutoff_(cutoff) {
    int columns = auxiliary.nfunction();
    int nvir = 0;
    if (orbitals) {
        property_require(auxiliary.role() == IsaBasisRole::Orbital, "Direct OV requires Orbital basis");
        property_finite(*orbitals);
        property_require(orbitals->nrow() == auxiliary.nfunction() && nocc > 0 &&
                         nocc < orbitals->ncol(), "Invalid direct OV orbital dimensions/occupation");
        nvir = orbitals->ncol() - nocc;
        property_require(nocc <= 4096 / nvir, "Direct OV limited to 4096 pairs");
        columns = nocc * nvir;
        representation_ = "direct_ov";
    } else {
        property_require(nocc == 0 && auxiliary.role() == IsaBasisRole::MolecularAux,
                         "Partitioned multipoles require molecular AUX");
    }
    property_require(!provenance.empty(), "Explicit partition provenance is required");
    property_require(std::isfinite(cutoff) && cutoff >= 0, "Invalid partition denominator cutoff");
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
    q_ = std::make_shared<Matrix>("Partitioned molecular AUX multipoles", offsets_.back(), columns);
    for (size_t a = 0; a < sites.size(); ++a) {
        const auto& site = sites[a]; const auto& s = site.samples;
        property_require(s.points.size() == s.weights.size() && s.points.size() == s.shape.size() &&
                         s.points.size() == s.shape_sum.size(), "Partition sample dimensions disagree");
        // The explicit provider validates every point and the complete neighbour list,
        // including batches whose denominator is excluded at every point.
        property_require(s.points.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
                         "Too many partition sample points");
        std::vector<double> values(columns);
        size_t excluded = 0, negative = 0;
        // AUX contraction needs only a bounded collocation block. Preserve the
        // original point-order running sums; never reduce independent block sums.
        // Keep the direct-MO GEMM shape unchanged to retain its BLAS arithmetic.
        const size_t auxiliary_batch = std::min<size_t>(4096,(8*1024*1024)/sizeof(double)/columns);
        property_require(orbitals || auxiliary_batch > 0, "AUX multipole row exceeds collocation scratch budget");
        const size_t batch_size = orbitals ? std::max<size_t>(1,s.points.size()) : auxiliary_batch;
        for (size_t begin = 0; begin < std::max<size_t>(1,s.points.size()); begin += batch_size) {
            const size_t end = std::min(s.points.size(),begin+batch_size);
            std::vector<std::array<double,3>> block;
            if (!orbitals) block.assign(s.points.begin()+begin,s.points.begin()+end);
            auto basis = auxiliary.evaluate_screened(orbitals ? s.points : block, s.auxiliary_sites);
            std::shared_ptr<Matrix> mo;
            if (orbitals) {
                mo = std::make_shared<Matrix>("Direct MO samples", basis->nrow(), orbitals->ncol());
                mo->gemm(false, false, 1.0, basis, orbitals, 0.0);
                property_finite(*mo);
            }
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
            for (int k = 0; k < columns; ++k)
                values[k] = mo ? mo->get(p-begin, k % nocc) * mo->get(p-begin, nocc + k / nocc)
                               : basis->get(p-begin, k);
            for (size_t t = 0; t < harmonics.size(); ++t) {
                double prefactor = weight*harmonics[t];
                property_require(std::isfinite(prefactor), "Nonfinite weighted multipole");
                for (int k = 0; k < columns; ++k)
                    q_->add(offsets_[a]+t, k, prefactor*values[k]);
            }
        }
        }
        excluded_.push_back(excluded); negative_.push_back(negative);
    }
    property_finite(*q_);
}
std::shared_ptr<Matrix> IsaPartitionedMultipoles::values() const { return q_->clone(); }
IsaDistributedResponse::IsaDistributedResponse(const IsaPartitionedMultipoles& partition,
        const std::vector<double>& frequencies, const std::vector<std::shared_ptr<Matrix>>& responses,
        const std::string& representation, const std::string& provenance)
    : partition_(partition), frequencies_(frequencies), provenance_(provenance) {
    property_require(representation == partition.representation_, "Distributed response representation must match partition columns; no implicit metric conversion");
    property_require(!provenance.empty(), "Explicit response provenance is required");
    property_require(!frequencies.empty() && frequencies.size() == responses.size(), "Frequency/response dimensions disagree");
    const int n = partition.q_->nrow(), k = partition.q_->ncol();
    for (size_t f = 0; f < frequencies.size(); ++f) {
        property_require(std::isfinite(frequencies[f]) && frequencies[f] >= 0, "Imaginary frequencies must be finite and nonnegative");
        property_require(responses[f] != nullptr, "Null coefficient response");
        property_finite(*responses[f]);
        property_require(responses[f]->nrow() == k && responses[f]->ncol() == k, "Coefficient response AUX dimensions disagree");
        auto qc = std::make_shared<Matrix>("Q C_DF", n, k);
        qc->gemm(false, false, 1.0, partition.q_, responses[f], 0.0);
        property_finite(*qc);
        auto alpha = std::make_shared<Matrix>("Distributed multipole response", n, n);
        alpha->gemm(false, true, -1.0, qc, partition.q_, 0.0);
        property_finite(*alpha);
        double scale = 1.0, defect = 0.0;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            scale = std::max(scale, std::abs(alpha->get(i,j)));
            defect = std::max(defect, std::abs(alpha->get(i,j)-alpha->get(j,i)));
        }
        property_require(std::isfinite(defect), "Nonfinite reciprocity diagnostic");
        reciprocity_.push_back(defect/scale); alpha_.push_back(alpha);
    }
}
std::shared_ptr<Matrix> IsaDistributedResponse::at_index(size_t index) const {
    if (index >= alpha_.size()) throw std::out_of_range("Distributed response frequency index out of range");
    return alpha_[index]->clone();
}
} }
