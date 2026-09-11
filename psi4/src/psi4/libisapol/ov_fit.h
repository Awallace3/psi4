/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_OV_FIT_H
#define PSI4_LIBISAPOL_OV_FIT_H
#include <memory>
#include <string>
#include <vector>
namespace psi {
class Matrix;
namespace isapol {
/// Owned native-integral/supplied-orbital fit, NOT native SCF or response.
/// All matrix accessors clone; no mutable storage escapes, even from C++ copies.
class IsaOvFitResult {
   public:
    std::shared_ptr<Matrix> coulomb_metric() const;
    std::shared_ptr<Matrix> metric() const;
    std::shared_ptr<Matrix> rhs() const;
    std::shared_ptr<Matrix> coefficients() const;
    std::vector<double> charges() const { return charges_; }
    int nmain() const { return nmain_; }
    int naux() const { return naux_; }
    int noccupied() const { return noccupied_; }
    int nvirtual() const { return nvirtual_; }
    int ntransition() const { return noccupied_ * nvirtual_; }
    double charge_penalty() const { return penalty_; }
    /// Declared eta: off-centre metric elements were scaled by (1-eta). Zero is
    /// the undamped fit; any other value is a separately declared model.
    double offsite_metric_damping() const { return damping_; }
    /// ||A D^T - T^T||_F / (||A||_F ||D||_F + ||T||_F), long-double accumulation.
    double relative_backward_residual() const { return residual_; }
    int lapack_info() const { return info_; } // success is NOT a rank certificate
    std::string provenance() const { return provenance_; }
    std::string representation() const { return "fitted_density_coefficients"; }
    std::string order() const { return "p=a+noccupied*r; occupied-fast"; }
    std::string solver() const { return "C_DGESV; general LU; all RHS; no refinement"; }
   private:
    friend class IsaAuxCoulomb;
    IsaOvFitResult() = default;
    std::shared_ptr<Matrix> j_, a_, t_, d_;
    std::vector<double> charges_;
    int nmain_=0, naux_=0, noccupied_=0, nvirtual_=0, info_=0;
    double penalty_=0., damping_=0., residual_=0.;
    std::string provenance_;
};
} }
#endif
