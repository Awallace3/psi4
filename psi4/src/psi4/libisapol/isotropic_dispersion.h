/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_ISOTROPIC_DISPERSION_H
#define PSI4_LIBISAPOL_ISOTROPIC_DISPERSION_H
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace psi {
class Matrix;
namespace isapol {
/// Supplied local scalar alpha_l = trace(alpha_ll)/(2l+1), Racah/atomic units.
/// No localization or isotropization is performed here. Matrix rows: frequencies;
/// columns: explicit strictly increasing ranks in [1,4]. Signed values retained.
struct IsaIsotropicSite {
    std::string label;
    std::array<double,3> origin = {0,0,0};
    std::vector<int> ranks;
    std::shared_ptr<Matrix> polarizabilities;
};
class IsaIsotropicModel {
 public:
    IsaIsotropicModel(const std::vector<double>& frequencies,
                     const std::vector<IsaIsotropicSite>& sites, const std::string& provenance);
    std::vector<double> frequencies() const { return frequencies_; }
    std::vector<IsaIsotropicSite> sites() const;
    std::string provenance() const { return provenance_; }
 private:
    std::vector<double> frequencies_;
    std::vector<IsaIsotropicSite> sites_;
    std::string provenance_;
};
struct IsaIsotropicCoefficient {
    int order = 0;
    double value = 0.;
    bool complete = false;
    std::vector<std::array<int,2>> included_rank_pairs, missing_rank_pairs;
};
struct IsaIsotropicPair {
    int site_a = 0, site_b = 0;
    std::vector<IsaIsotropicCoefficient> coefficients;
};
struct IsaIsotropicDispersionResult {
    std::vector<IsaIsotropicPair> pairs;
    std::vector<double> frequencies, cp_weights;
    std::vector<std::string> labels_a, labels_b;
    std::vector<std::array<double,3>> origins_a, origins_b;
    std::string provenance_a, provenance_b;
};
/// C_(2la+2lb+2) += binomial(2la+2lb,2la) sum_f cp_weight[f]*alphaA_la*alphaB_lb.
/// cp_weights ALREADY include 1/(2*pi). A static frequency must have zero weight.
/// All A/B site pairs (not a self-model assumption). Max order is 6,8,10 or12.
/// Missing ranks never become available zero tensors: each coefficient reports
/// included/missing pairs and completeness, separately from its within-model value.
IsaIsotropicDispersionResult isa_isotropic_dispersion(const IsaIsotropicModel& a,
    const IsaIsotropicModel& b, const std::vector<double>& cp_weights, int max_order = 12);
} }
#endif
