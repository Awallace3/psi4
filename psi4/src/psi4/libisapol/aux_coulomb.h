/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_AUX_COULOMB_H
#define PSI4_LIBISAPOL_AUX_COULOMB_H
#include "explicit_basis.h"
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace psi {
class BasisSet;
namespace isapol {
struct IsaDrhoCResult {
    std::shared_ptr<Matrix> coulomb_metric, metric;
    std::vector<double> charges, raw_rhs, rhs, coefficients;
    double charge_penalty=0., relative_residual=0., fitted_electrons=0.;
    /// Corrections applied by LU refinement (0 when none was requested).
    int refinement_iterations=0;
    /// max|x - x_LU| / max|x|: how far refinement moved the plain LU solution.
    /// An observed change, not an error estimate or bound.
    double refinement_displacement=0.;
};
/// Correctly rounded (round-to-nearest-even) r = b - A x for a finite row-major
/// n x n A and finite x, b, from exact integer products and an exact integer
/// long accumulator: independent of term order, threads, BLAS, FP rounding
/// mode, FTZ/DAZ and contraction. An exact zero is +0. Refuses nonfinite
/// inputs, n outside [1, 2^30-2] and any row whose rounded value overflows.
std::vector<double> isa_exact_residual(const double* a_rowmajor, const double* x, const double* b, std::size_t n);
struct IsaRefinedSolve {
    std::vector<double> x;
    int iterations=0;
    double displacement=0.;
};
/// LAPACK DGESV of the column-major copy of A, then up to max_iterations
/// (0..32) corrections x += A^-1 r on the same LU factors with the exact
/// residual r above. Converged when max|d| <= 2^-52 max|x|; that bounds the
/// last correction, it does not prove forward accuracy. Refuses (no fallback
/// to the plain LU x) on stagnation (a step larger than half the previous),
/// on reaching the cap, on nonfinite corrections or updates, on residual
/// overflow, and when max_iterations > 0 outside round-to-nearest with
/// gradual underflow on the calling thread. max_iterations = 0 is plain DGESV.
IsaRefinedSolve isa_refined_lu_solve(const double* a_rowmajor, const double* b, std::size_t n, int max_iterations);
/// Native molecular-AUX integrals from explicit effective coefficients.
/// Scope: Cartesian GAMINT or spherical DALTON S-G AUX. The two are DIFFERENT
/// declared bases even when built from one exponent set -- a spherical shell
/// spans 2l+1 functions, a Cartesian one (l+1)(l+2)/2 -- so they give different
/// fits, different partitions, and results that may never be quoted as agreeing.
/// Not a DF solve or basis recipe.
/// Integrals come from native Psi4 Libint2ERI/PotentialInt objects over raw
/// Cartesian twins of the explicit bases (see native_basis); the GAMINT factor
/// and DALTON transforms are applied here. No global ordering/normalization changes.
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
     * is formed or sliced. max_bytes bounds the returned matrix,
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
    /// Native Coulomb/closed-shell finite charge-penalty fit, solved by
    /// isa_refined_lu_solve: plain LU by default (max_refinement_iterations=0),
    /// exact-residual LU refinement when requested.
    /// Penalty enters each occupied pair diagonal before tracing. No rescaling.
    IsaDrhoCResult fit_drho_c(const IsaExplicitBasis& orbital,
                             const Matrix& occupied_coefficients, double charge_penalty=1000.,
                             int max_refinement_iterations=0) const;
    /// The raw Cartesian twin used by metric(), with the declared map T (declared x raw):
    /// a declared function is T times the raw functions (GAMINT factor at the Libint
    /// Cartesian index, or the DALTON rows), so the declared metric is T J_raw T^T.
    /// Both objects are freshly built, independent of this provider and owned by the
    /// caller. The BasisSet is an input for read-only consumers (e.g. FDDS_Monomer with
    /// aux_transform=T) and must not be modified.
    std::pair<std::shared_ptr<BasisSet>, std::shared_ptr<Matrix>> native_auxiliary() const;
   private:
    /// Native Cartesian twin: one Psi4 atom per explicit centre (coincident
    /// centres stay distinct), stored effective coefficients used unchanged.
    /// Native shells are grouped by centre; shells[i] is the native index of
    /// explicit shell i.
    struct NativeBasis {
        std::shared_ptr<BasisSet> basis;
        std::vector<int> shells;
    };
    static NativeBasis native_basis(const IsaExplicitBasis& explicit_basis);
    IsaExplicitBasis basis_;
};
} }
#endif
