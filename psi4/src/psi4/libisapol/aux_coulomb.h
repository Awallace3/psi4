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
/// Scope: Cartesian GAMINT or spherical DALTON S-G AUX. The two are DIFFERENT
/// declared bases even when built from one exponent set -- a spherical shell
/// spans 2l+1 functions, a Cartesian one (l+1)(l+2)/2 -- so they give different
/// fits, different partitions, and results that may never be quoted as agreeing.
/// Not a DF solve or basis recipe.
/// Psi4 owns Libint2 global initialization. No global ordering/normalization changes.
class IsaAuxCoulomb {
   public:
    explicit IsaAuxCoulomb(const IsaExplicitBasis& auxiliary);
    /// Analytic integrals of AUX functions, including all even Cartesian powers.
    std::vector<double> charges() const;
    /// Fresh Coulomb metric from raw Cartesian Libint2 shells and true unit shells,
    /// harmonically contracted afterwards when the declared AUX is spherical.
    /// Precision zero, no additional shell screening. No coefficient renormalization.
    std::shared_ptr<Matrix> metric() const;
    /// Positive Coulomb potentials P(k,p) = integral chi_k(r)/|r-R_p| dr,
    /// rows in this explicit AUX basis's own order, columns at caller points
    /// (npoint x 3, bohr). Effective contractions and Cartesian GAMINT factors
    /// or spherical DALTON transforms are preserved, with no renormalization.
    /// No multipole truncation, nuclear term, electron sign or response solve.
    /// This is an operand for -P^T C_aux(iw) P, NOT a fitted response producer.
    /// Finite points at a basis centre are valid; no distance screening/repair.
    /// At most 512 points; max_bytes bounds the returned dense matrix only,
    /// not Libint shell workspace. No global option or wavefunction mutation.
    std::shared_ptr<Matrix> point_potentials(const Matrix& points,
                                            size_t max_bytes=512UL*1024*1024) const;
    /// Native (AUX|MAIN MAIN), rows AUX in the declared AUX representation,
    /// column mu*nmain+nu (nu fastest).
    /// MAIN must be explicit DALTON spherical Orbital S-G. No MO/charge factors.
    std::shared_ptr<Matrix> three_center(const IsaExplicitBasis& orbital) const;
    /** Exact contiguous AUX-shell slice of three_center; local rows start at zero.
     * first_shell is zero-based; shell_count must be positive and entirely in
     * range. At most 512 AUX functions may occur in one block. MAIN columns
     * remain mu*nmain+nu with every MAIN function retained. No full AUX tensor
     * is formed or sliced. max_bytes (<=512 MiB) bounds the returned matrix,
     * not caller basis storage, copied shell descriptors or Libint workspace.
     * A streaming consumer must account those and its other live buffers.
     */
    std::shared_ptr<Matrix> three_center_shell_block(const IsaExplicitBasis& orbital,
        std::size_t first_shell, std::size_t shell_count,
        std::size_t max_bytes = 512UL*1024*1024) const;
    /// 2 sum_occ C_i^T B_k C_i BEFORE solving; every spatial occupation is 2.
    /// No charge penalty here; no AO-density sampling or fitted-density substitute.
    std::vector<double> closed_shell_rhs(const IsaExplicitBasis& orbital,
                                          const Matrix& occupied_coefficients) const;
    /// Native Coulomb/closed-shell finite charge-penalty fit, LU without refinement.
    /// Penalty enters each occupied pair diagonal before tracing. No rescaling.
    IsaDrhoCResult fit_drho_c(const IsaExplicitBasis& orbital,
                             const Matrix& occupied_coefficients, double charge_penalty=1000.) const;
    /// Native-integral/supplied-MAIN-orbital OV fit; no SCF or response construction.
    /// ``offsite_metric_damping`` is the declared eta of the constrained fit: every
    /// metric element whose two AUX functions sit on DIFFERENT centres is scaled by
    /// (1-eta) before the charge penalty is added. It is a model declaration, not a
    /// tolerance or a conditioning repair -- eta!=0 is a different fit and a
    /// different partition, and its result may never be quoted against an eta=0 one.
    IsaOvFitResult fit_ov(const IsaExplicitBasis& orbital, const Matrix& occupied,
                         const Matrix& virtuals, const std::string& provenance,
                         double charge_penalty=1.0, double offsite_metric_damping=0.0) const;
   private:
    IsaExplicitBasis basis_;
};
} }
#endif
