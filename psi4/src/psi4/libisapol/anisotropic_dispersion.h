/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_ANISOTROPIC_DISPERSION_H
#define PSI4_LIBISAPOL_ANISOTROPIC_DISPERSION_H
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace psi {
class Matrix;
namespace isapol {
using IsaAnisotropicFrame = std::array<std::array<double, 3>, 3>;
/// Supplied reciprocal real LOCAL response, not a distributed-response adapter.
/// Explicit increasing ranks in [1,4]; every block among these ranks is supplied.
/// Axes concatenate 10,11c,11s,20,... for the declared ranks (no rank zero).
/// One exactly symmetric finite matrix per frequency; signed/indefinite is legal.
struct IsaAnisotropicSite {
    std::string label;
    std::array<double, 3> origin = {0, 0, 0}; // bohr
    IsaAnisotropicFrame frame{}; // required proper local-to-global frame, NOT a default identity
    std::vector<int> ranks;
    std::vector<std::shared_ptr<Matrix>> responses;
    std::vector<std::string> components() const;
};
struct IsaAnisotropicDispersionResult;
class IsaRecoupledModel;
struct IsaRecoupledDispersionResult;
class IsaAnisotropicModel {
 public:
    /// declaration must be "supplied_local_response"; provenance must be nonblank.
    /// Owns snapshots; all matrix getters return copies. Frame tolerance is 1e-12.
    /// Resource ceilings per model: 8,000,000 tensor elements, 65,536 response
    /// matrices (including static nodes), and 4096 sites.
    IsaAnisotropicModel(const std::vector<double>& frequencies,
        const std::vector<IsaAnisotropicSite>& sites, const std::string& declaration,
        const std::string& provenance);
    std::vector<double> frequencies() const { return frequencies_; }
    std::vector<IsaAnisotropicSite> sites() const;
    std::string declaration() const { return "supplied_local_response"; }
    std::string provenance() const { return provenance_; }
 private:
    std::vector<double> frequencies_;
    std::vector<IsaAnisotropicSite> sites_;
    std::vector<std::shared_ptr<Matrix>> rotations_;
    std::string provenance_;
    // Trusted read-only recoupling access avoids cloning electronic tensors
    // merely to validate ranks and resource bounds. Public getters still copy.
    friend class IsaRecoupledModel;
    friend IsaRecoupledDispersionResult isa_recoupled_dispersion(const IsaRecoupledModel&,
        const IsaRecoupledModel&, const std::vector<double>&, int);
    friend IsaAnisotropicDispersionResult isa_anisotropic_dispersion(const IsaAnisotropicModel&,
        const IsaAnisotropicModel&, const std::vector<double>&, int);
};
struct IsaAnisotropicCoefficient {
    int order = 0;
    double value = 0., energy = 0.; // scalar C_n and -C_n/R^n, atomic units
    bool declared_model_complete = true, unrestricted_complete = false;
    // Ordered (la,la',lb,lb'); missing refers to unrestricted ranks>=1,
    // including theoretical ranks 5..7. Explicit zero blocks remain included.
    std::vector<std::array<int, 4>> included_rank_quadruples, missing_rank_quadruples;
};
struct IsaAnisotropicPair {
    int site_a = 0, site_b = 0; // indices into result.model_a/b.sites
    std::array<double, 3> displacement{}, direction{}; // B - A
    double distance = 0., truncated_energy = 0.;
    std::vector<IsaAnisotropicCoefficient> coefficients;
};
struct IsaAnisotropicDispersionResult {
    IsaAnisotropicDispersionResult(const IsaAnisotropicModel& a, const IsaAnisotropicModel& b)
        : model_a(a), model_b(b) {}
    IsaAnisotropicModel model_a, model_b; // immutable owned model snapshots
    std::vector<double> frequencies, cp_weights;
    std::vector<IsaAnisotropicPair> pairs;
    int max_order = 12;
    double truncated_energy = 0.;
};
/// Undamped, nonretarded orientation-resolved scalar coefficients, NOT C_n(t,u,J).
/// All A/B pairs (at most 4096); exact matching grids; CP weights already 1/(2*pi).
/// Static nodes require zero weight; at least one weight positive. All orders 6..12,
/// including odd orders. Truncated energy has no general positivity guarantee.
IsaAnisotropicDispersionResult isa_anisotropic_dispersion(const IsaAnisotropicModel& a,
    const IsaAnisotropicModel& b, const std::vector<double>& cp_weights, int max_order = 12);
/// Expert electrostatic diagnostic: T_lm,kn(R)=(-1)^l H_lm(d)H_kn(d)(1/R)/(d_l d_k).
/// R=B-A in bohr; ranks each 1..4; rows/columns use 0,1c,1s,... within that rank.
/// Physical inverse powers included (not tau); analytic derivatives through degree8.
/// No frames, responses, damping, or rank8 transform call. Returns an owned matrix.
std::shared_ptr<Matrix> isa_anisotropic_interaction(int rank_a, int rank_b,
    const std::array<double, 3>& displacement);
} }
#endif
