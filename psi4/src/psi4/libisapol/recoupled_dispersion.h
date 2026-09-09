/* CamCASP conventions: Alston J. Misquitta and Anthony J. Stone.
 * Copyright (c) 2019 Anthony Stone. MIT: see RECOUPLED_CAMCASP_LICENSE.
 * Psi4 additions: Copyright (c) 2026 The Psi4 Developers. LGPL-3.0-only.
 */
#ifndef PSI4_LIBISAPOL_RECOUPLED_DISPERSION_H
#define PSI4_LIBISAPOL_RECOUPLED_DISPERSION_H
#include "anisotropic_dispersion.h"
#include <complex>
#include <cstddef>
namespace psi { namespace isapol {
struct IsaRecoupledBlock {
    int la = 0, lap = 0, first_component = 0, last_component = 0;
    // frequency-major, then global components first..last, inclusive (one-based).
    std::vector<std::complex<double>> values;
    std::vector<std::string> components() const;
};
class IsaRecoupledModel {
 public:
    explicit IsaRecoupledModel(const IsaAnisotropicModel& local);
    IsaAnisotropicModel source() const { return source_; }
    std::vector<double> frequencies() const { return source_.frequencies(); }
    std::vector<std::vector<IsaRecoupledBlock>> sites() const { return sites_; }
    std::complex<double> value(std::size_t site, std::size_t frequency,
                              int la, int lap, int component) const;
 private:
    friend IsaRecoupledDispersionResult isa_recoupled_dispersion(const IsaRecoupledModel&,
        const IsaRecoupledModel&, const std::vector<double>&, int);
    IsaAnisotropicModel source_;
    std::size_t nfrequency_ = 0;
    std::vector<std::vector<IsaRecoupledBlock>> sites_;
};
struct IsaRecoupledCoefficient {
    int order = 0, t = 0, u = 0, J = 0;
    double value = 0.;
};
struct IsaRecoupledCoverage {
    int order = 0;
    bool table_complete = false, unrestricted_complete = false;
    std::vector<std::array<int,4>> included_rank_quadruples;
    std::vector<std::array<int,4>> missing_table_rank_quadruples;
    std::vector<std::array<int,4>> missing_unrestricted_rank_quadruples;
};
struct IsaRecoupledPair {
    int site_a = 0, site_b = 0;
    std::vector<IsaRecoupledCoefficient> coefficients;
    std::vector<IsaRecoupledCoverage> coverage;
    /// Structural absence is zero, never a print-threshold operation.
    double coefficient(int order, int t, int u, int J) const;
};
struct IsaRecoupledDispersionResult {
    IsaRecoupledDispersionResult(const IsaRecoupledModel& a, const IsaRecoupledModel& b)
        : model_a(a), model_b(b) {}
    IsaRecoupledModel model_a, model_b;
    std::vector<double> frequencies, cp_weights;
    std::vector<IsaRecoupledPair> pairs;
    int max_order = 12;
};
/// Local-axis Cn(t,u,J), NOT orientation-resolved scalar energies. No rotations,
/// conjugations, distance checks, damping, spin factors or extra CP prefactors.
/// Limits checked before output allocation: 4096 pairs, 2M coefficient records,
/// 100M frequency-term operations. Recoupling: 8M complex elements, 100M ops.
IsaRecoupledDispersionResult isa_recoupled_dispersion(const IsaRecoupledModel& a,
    const IsaRecoupledModel& b, const std::vector<double>& cp_weights, int max_order = 12);
} }
#endif
