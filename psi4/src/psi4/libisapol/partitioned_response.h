/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_PARTITIONED_RESPONSE_H
#define PSI4_LIBISAPOL_PARTITIONED_RESPONSE_H
#include "explicit_basis.h"
#include <string>
namespace psi { namespace isapol {
/// Explicit site quadrature; shapes already include the caller's screening/tails.
/// Signed weights/shapes are retained. AUX neighbours are zero-based, unpadded;
/// an empty list screens all AUX functions in this batch.
struct IsaMultipoleSamples {
    std::vector<std::array<double,3>> points;
    std::vector<double> weights, shape, shape_sum;
    std::vector<int> auxiliary_sites;
};
struct IsaMultipoleSite {
    std::string label;
    std::array<double,3> origin = {0,0,0};
    int rank = 0;
    IsaMultipoleSamples samples;
};
/// Supplied-partition Q, NOT an ISA fixed point or a native property adapter.
/// Rows concatenate sites and Racah regular real components 00,10,11c,11s,...;
/// columns are molecular AUX functions. Global Cartesian axes only.
/// All inputs are consumed/copied; getters return independent owned snapshots.
class IsaPartitionedMultipoles {
 public:
    IsaPartitionedMultipoles(const IsaExplicitBasis& auxiliary,
        const std::vector<IsaMultipoleSite>& sites, const std::string& provenance,
        double denominator_cutoff = 1.e-36);
    /// Supplied, unsampled Q (e.g. the analytic DF-centre rule), used as given: no
    /// quadrature, denominator, renormalization or reordering. Rows must already
    /// be the concatenated (site, Racah component) axes, and sites must carry no
    /// samples.
    IsaPartitionedMultipoles(std::shared_ptr<Matrix> values,
        const std::vector<IsaMultipoleSite>& sites, const std::string& representation,
        const std::string& provenance);
    std::string representation() const { return representation_; }
    std::shared_ptr<Matrix> values() const;
    std::vector<int> offsets() const { return offsets_; }
    std::vector<int> ranks() const { return ranks_; }
    std::vector<std::string> labels() const { return labels_; }
    std::vector<std::string> components() const { return components_; }
    std::vector<std::array<double,3>> origins() const { return origins_; }
    std::vector<size_t> excluded_denominators() const { return excluded_; }
    std::vector<size_t> negative_ratios() const { return negative_; }
    std::string provenance() const { return provenance_; }
    double denominator_cutoff() const { return cutoff_; }
 private:
    void declare_sites(const std::vector<IsaMultipoleSite>& sites);
    std::shared_ptr<Matrix> q_;
    std::vector<int> offsets_, ranks_;
    std::vector<std::string> labels_, components_;
    std::vector<std::array<double,3>> origins_;
    std::vector<size_t> excluded_, negative_;
    std::string provenance_;
    double cutoff_;
    std::string representation_ = "fitted_density_coefficients";
};
} }
#endif
