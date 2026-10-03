/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#ifndef FDDS_DISP_H
#define FDDS_DISP_H

#include "psi4/libmints/typedefs.h"

#include <array>
#include <map>
#include <string>

namespace psi {

class BasisSet;
class DFHelper;

namespace sapt {

/// Explicit per-instance resources for the declared FDDS_Monomer path; every field is required.
struct FDDSResources {
    size_t memory_bytes = 0;  ///< cap on accounted numerical storage (matrices, blocks, LAPACK work, integral objects)
    size_t disk_bytes = 0;    ///< cap on this instance's peak scratch bytes
    std::string scratch_dir;  ///< existing writable caller-owned directory; only this instance's files are removed
    std::string subalgo = "OUT_OF_CORE";  ///< DFHelper AO integrals: INCORE (held) or OUT_OF_CORE (recomputed)
    size_t nthread = 0;       ///< OpenMP threads for integral generation and the declared blocking loops; the
                              ///< per-frequency form_unc_amplitude/form_aux_matrices loops and BLAS use the
                              ///< process OpenMP/vendor thread counts, so this is not a total-thread cap
};

/// Construction record of a declared FDDS_Monomer. Ratios are relative to the largest |eigenvalue|.
struct FDDSModel {
    size_t naux_raw = 0, naux = 0, nocc = 0, nvir = 0;
    bool is_hybrid = false;
    size_t metric_dropped = 0;        ///< J_d eigenvalues dropped by the native 1e-12 rule (T1/T2)
    double metric_max_dropped = 0.0;  ///< largest dropped ratio, 0 if none
    double metric_min_kept = 0.0;     ///< smallest kept ratio
    double metric_lu_rcond = 0.0;     ///< LAPACK 1-norm reciprocal condition estimate of J_d
    size_t qr_rank = 0;               ///< singular values of R kept by pinv(R, 1e-13); hybrid only
    size_t nthread = 0;
    size_t memory_bytes = 0, disk_bytes = 0, required_memory_bytes = 0, required_disk_bytes = 0;
    /// Declared-pass loop counts per raw stream: dispatched blocks and peak block rows (not allocation/RSS)
    std::map<std::string, std::array<size_t, 2>> declared_pass_blocks;
};

/// One frequency of the declared coefficient response.
struct FDDSResponse {
    SharedMatrix response;       ///< chi_c = sym((I - A)^-1 chi0), declared AUX coefficients, full J_d^-1 legs
    double omega = 0.0, x_alpha = 0.0;
    double native_dyson_ratio = 0.0;  ///< sigma_min/sigma_max of J_d - XSW formed as the legacy helper forms it
    double s2_dyson_ratio = 0.0;      ///< sigma_min/sigma_max of J_d - J_d A from the S2 factors
    double solve_residual = 0.0; ///< ||(I - A) chi - chi0||_F / ||chi0||_F before symmetrization
    size_t masked_transitions = 0;  ///< native non-hybrid 4D/(D^2+w^2) < 1e-14 mask count
};

class FDDS_Dispersion {
   protected:
    // BasisSets
    std::shared_ptr<BasisSet> primary_;
    std::shared_ptr<BasisSet> auxiliary_;

    // Coulomb metric inverse
    SharedMatrix metric_inv_;

    // Coulomb metric
    SharedMatrix metric_;

    // Coulomb metric -0.5 power
    SharedMatrix metric_half_inv_;

    // Auxiliary overlap matrix
    SharedMatrix aux_overlap_;

    // DFHelper object
    std::shared_ptr<DFHelper> dfh_;

    // Cache map
    std::map<std::string, SharedMatrix> matrix_cache_;
    std::map<std::string, SharedVector> vector_cache_;

    // Is hybrid functional? 
    bool is_hybrid_;

    // QR factorization result
    SharedMatrix R_A_, R_B_;

    bool single_monomer_;

    // Response auxiliary dimension (auxiliary_->nbf() on the legacy path, the declared size otherwise)
    size_t naux_;
    // Explicit path only: fixed work budget and blocking thread count; 0 selects the legacy global reads
    size_t work_doubles_ = 0;
    size_t nthread_ = 0;
    // Declared path: QR reads the declared (ar|Q) tensor instead of the legacy (Q|ar) stream
    bool qr_from_arQ_ = false;

    struct Deferred {};
    FDDS_Dispersion(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                    std::map<std::string, SharedMatrix> matrix_cache,
                    std::map<std::string, SharedVector> vector_cache, bool is_hybrid, bool single_monomer);
    // Validates and stores the orbital data only; the derived class prepares everything else.
    FDDS_Dispersion(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                    std::map<std::string, SharedMatrix> matrix_cache,
                    std::map<std::string, SharedVector> vector_cache, bool is_hybrid, Deferred);
    void validate_orbitals() const;
    void check_monomer(const std::string& monomer, bool needs_hybrid = false) const;
    size_t budget_doubles() const;
    int blocking_threads() const;

    // (ar|X|in) and (ar|Y|in) for two streamed inputs; the legacy form_X/form_Y call these unchanged
    void exchange_X(const std::string& arR, const std::array<std::string, 2>& in,
                    const std::array<std::string, 2>& out, size_t nocc, size_t nvir);
    void exchange_Y(const std::string& aaR, const std::string& rrR, const std::array<std::string, 2>& in_ra,
                    const std::array<std::string, 2>& out, size_t nocc, size_t nvir);

   public:
    /**
     * Constructs the FDDS_Dispersion object.
     * @param primary   The primary basis
     * @param auxiliary The auxiliary basis
     * @param cache     A data cache containing "Cocc_A", "Cvir_A", "eps_occ_A", "eps_vir_A", "Cocc_B",
     * "Cvir_B", "eps_occ_B", "eps_vir_B" quantities
     * @param is_hybrid Flag of hybrid functional
     */
    FDDS_Dispersion(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                    std::map<std::string, SharedMatrix> matrix_cache, std::map<std::string, SharedVector> vector_cache,
                    bool is_hybrid);

    ~FDDS_Dispersion();

    /**
     * Projects the densities from primary AO space to auxiliary AO space
     * @param  dens     Vector of densities to transform
     * @return aux_dens Vector of transformed densities
     */
    std::vector<SharedMatrix> project_densities(std::vector<SharedMatrix> dens);

    /**
     * Forms the uncoupled amplitude Qia,Eia,Pia->PQ and projector Qia,Pia->PQ
     * @param  monomer Monomer "A" or "B"
     * @param  omega   Time dependent value
     * @return         "PQ" amplitude 
     */
    SharedMatrix form_unc_amplitude(std::string monomer, double omega);

    /**
     * Forms the uncoupled amplitude and other PQ matrices in hybrid FDDS dispersion
     * @param  monomer Monomer "A" or "B"
     * @param  omega   Time dependent value
     * @return         Dictionary of PQ matrics 
     * @return ret["amp"]  PQ uncoupled amplitude
     * @return ret["K1LD"]  K1(lambda * d) = (P|ar) LDar (ar|X+Y|Q)
     * @return ret["K2LD"]  K2(lambda * d) = (P|ar) LDar (ar|X-Y|Q)
     * @return ret["K2L"]  K2(lambda) = (P|ar) Lar (ar|X-Y|Q)
     * @return ret["K21L"]  K21(lambda) = (P|X-Y|ar) Lar (ar|X+Y|Q)
     */
    std::map<std::string, SharedMatrix> form_aux_matrices(std::string monomer, double omega);

    /**
     * Returns the metric matrix. Borrowed instance state (not a copy): callers must treat it as
     * read-only, because every later operation of this instance uses it.
     * @return Metric
     */
    SharedMatrix metric() { return metric_; }

    /**
     * Returns the metric_inv matrix. Borrowed, read-only instance state, as for metric().
     * @return Metric
     */
    SharedMatrix metric_inv() { return metric_inv_; }

    /**
     * Returns the auxiliary overlap matrix. Borrowed, read-only instance state, as for metric().
     * @return Overlap
     */
    SharedMatrix aux_overlap() { return aux_overlap_; }

    /**
     * Returns R for QR decomposition of (ar|Q)
     * @return R_A
     */
    SharedMatrix R_A() { return R_A_; }

    /**
     * Returns R for QR decomposition of (bs|Q)
     * @return R_B
     */
    SharedMatrix R_B() { return R_B_; }

    /**
     * Forms X-type 3-index exchange integral tensor (ar|X|Q) = (ar'|a'r)(a'r'|Q)
     * @param monomer Monomer "A" or "B"
     */
    void form_X(std::string monomer);

    /**
     * Forms Y-type 3-index exchange integral tensor (ar|Y|Q) = (aa'|rr')(a'r'|Q)
     * @param monomer Monomer "A" or "B"
     */
    void form_Y(std::string monomer);

    /**
     * Performs QR factorization of the occupied-virtual three-index integrals.
     * Economy factors are zero-padded to naux columns/rows; the caller handles rank deficiency by pseudoinverse.
     * @param monomer Monomer "A" or "B"
     * @return R (not inverted)
     */
    SharedMatrix QR(std::string monomer);

    SharedMatrix get_tensor_pqQ(std::string name, std::tuple<size_t, size_t, size_t> dimensions);
    void print_tensor_pqQ(std::string tensor_name, std::string file_name, std::tuple<size_t, size_t, size_t> dimensions);

};  // End FDDS_Dispersion

/// One monomer, using the same integral preparation and response intermediates as SAPT.
class FDDS_Monomer : public FDDS_Dispersion {
   public:
    FDDS_Monomer(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                 SharedMatrix Cocc, SharedMatrix Cvir, SharedVector eps_occ, SharedVector eps_vir, bool is_hybrid)
        : FDDS_Dispersion(primary, auxiliary, {{"Cocc_A", Cocc}, {"Cvir_A", Cvir}},
                          {{"eps_occ_A", eps_occ}, {"eps_vir_A", eps_vir}}, is_hybrid, true) {}

    /**
     * Declared-basis path with explicit resources. The raw auxiliary basis is transformed by
     * aux_transform T (p_d x p_r, declared = T raw; nullptr = identity) on every auxiliary index
     * before any metric factorization, QR or response. Orbital inputs and T are copied; basis sets
     * are borrowed and must not be modified. Inputs are validated and storage admitted before any copy.
     * Global memory, options, threads and the PSIO path are never read or written for storage
     * decisions; BLAS threading stays the vendor's. ERIs are Libint2 whatever INTEGRAL_PACKAGE says,
     * and their storage is charged. Sequential frequency calls only.
     */
    FDDS_Monomer(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                 SharedMatrix Cocc, SharedMatrix Cvir, SharedVector eps_occ, SharedVector eps_vir, bool is_hybrid,
                 const FDDSResources& resources, SharedMatrix aux_transform = nullptr);

    /// Accounted memory and peak disk in bytes, with per-stage terms, for the declared path.
    static std::map<std::string, size_t> requirement(std::shared_ptr<BasisSet> primary,
                                                     std::shared_ptr<BasisSet> auxiliary, size_t nocc, size_t nvir,
                                                     size_t naux, bool is_hybrid, const std::string& subalgo,
                                                     size_t nthread);

    SharedMatrix form_unc_amplitude(double omega) { return FDDS_Dispersion::form_unc_amplitude("A", omega); }
    std::map<std::string, SharedMatrix> form_aux_matrices(double omega) {
        return FDDS_Dispersion::form_aux_matrices("A", omega);
    }
    SharedMatrix R() {
        check_monomer("A", true);
        return R_A_;
    }
    /// Legacy path only; the declared path refuses it.
    std::vector<SharedMatrix> project_densities(std::vector<SharedMatrix> dens);

    /**
     * Declared coefficient response at imaginary frequency omega for the explicit kernel W (J + fxc,
     * p_d x p_d, used as given) and exact-exchange fraction x_alpha (0 unless hybrid). Throws unless
     * both the natively formed J_d - XSW and the S2-formed J_d - J_d A have sigma_min/sigma_max
     * > 2e-13. This is an empirical policy, not an accuracy, forward-error or sign bound: declared
     * metrics resolved only to machine precision can pass both checks with an inaccurate, even
     * indefinite, response (see model() metric_max_dropped and metric_lu_rcond). Declared path only.
     */
    FDDSResponse form_coefficient_response(double omega, double x_alpha, SharedMatrix kernel);
    const FDDSModel& model() const;

    static constexpr double metric_cutoff = 1.e-12;   // native Matrix::power rule (T1/T2)
    static constexpr double r_rcond = 1.e-13;         // native pinv(R) rule (T3)
    static constexpr double dyson_rcond = 1.e-13;     // native pinv(J - XSW) rule (T4)
    static constexpr double dyson_refusal = 2.e-13;   // approved refusal threshold for both T4 ratios

   private:
    bool declared_ = false;
    bool has_transform_ = false;
    FDDSModel model_;
    SharedMatrix metric_lu_;  // LU factors of J_d (column-major view of the symmetric matrix)
    std::vector<int> metric_ipiv_;
    SharedMatrix Rtinv_;      // pinv(R, 1e-13)^T
    void prepare_declared(const FDDSResources& resources, SharedMatrix T);
    void declared_pass(std::shared_ptr<DFHelper> raw, const SharedMatrix& T, const SharedMatrix& half_inv,
                       size_t raw_retained);
    double native_dyson_ratio(double omega, double x_alpha, const SharedMatrix& W);
};
}  // namespace sapt
}  // namespace psi

#endif
