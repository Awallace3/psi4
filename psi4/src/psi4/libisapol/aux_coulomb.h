/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_AUX_COULOMB_H
#define PSI4_LIBISAPOL_AUX_COULOMB_H
#include "explicit_basis.h"
#include <string>
namespace psi { namespace isapol {
class IsaOvFitResult;
struct IsaDrhoCResult {
    std::shared_ptr<Matrix> coulomb_metric, metric;
    std::vector<double> charges, raw_rhs, rhs, coefficients;
    double charge_penalty=0., relative_residual=0., fitted_electrons=0.;
};
/// Native molecular-AUX integrals from explicit effective coefficients.
/// Initial scope: Cartesian GAMINT S-G AUX only. Not a DF solve or basis recipe.
/// Psi4 owns Libint2 global initialization. No global ordering/normalization changes.
class IsaAuxCoulomb {
   public:
    explicit IsaAuxCoulomb(const IsaExplicitBasis& auxiliary);
    /// Analytic integrals of AUX functions, including all even Cartesian powers.
    std::vector<double> charges() const;
    /// Fresh Coulomb metric from raw Cartesian Libint2 shells and true unit shells.
    /// Precision zero, no additional shell screening. No coefficient renormalization.
    std::shared_ptr<Matrix> metric() const;
    /// Native (AUX|MAIN MAIN), rows AUX, column mu*nmain+nu (nu fastest).
    /// MAIN must be explicit DALTON spherical Orbital S-G. No MO/charge factors.
    std::shared_ptr<Matrix> three_center(const IsaExplicitBasis& orbital) const;
    /// 2 sum_occ C_i^T B_k C_i BEFORE solving; every spatial occupation is 2.
    /// No charge penalty here; no AO-density sampling or fitted-density substitute.
    std::vector<double> closed_shell_rhs(const IsaExplicitBasis& orbital,
                                          const Matrix& occupied_coefficients) const;
    /// Native Coulomb/closed-shell finite charge-penalty fit, LU without refinement.
    /// Penalty enters each occupied pair diagonal before tracing. No rescaling.
    IsaDrhoCResult fit_drho_c(const IsaExplicitBasis& orbital,
                             const Matrix& occupied_coefficients, double charge_penalty=1000.) const;
    /// Native-integral/supplied-MAIN-orbital OV fit; no SCF or response construction.
    IsaOvFitResult fit_ov(const IsaExplicitBasis& orbital, const Matrix& occupied,
                         const Matrix& virtuals, const std::string& provenance,
                         double charge_penalty=1.0) const;
   private:
    IsaExplicitBasis basis_;
};
} }
#endif
