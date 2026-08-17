/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */

#ifndef PSI4_SRC_PSI4_LIBMINTS_ATOMIC_POLARIZABILITY_H
#define PSI4_SRC_PSI4_LIBMINTS_ATOMIC_POLARIZABILITY_H

#include <array>
#include <cstddef>
#include <map>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "psi4/psi4-dec.h"
#include "psi4/libmints/typedefs.h"

namespace psi {

class BasisSet;
struct BasisSetStructuralSnapshot;
class Matrix;
class Molecule;
class SuperFunctional;
class Vector;
class Wavefunction;
class ISAWeights;

/** Imaginary-frequency points and transformed Gauss-Legendre weights. */
struct PSI_API FrequencyGrid {
    std::vector<double> frequencies;
    std::vector<double> weights;
};

/** Complete rank-3 real-spherical response matrix (3 + 5 + 7 components). */
using L3Matrix = std::array<std::array<double, 15>, 15>;

/** Rank-0-through-rank-3 real-spherical workspace (1 + 3 + 5 + 7 components). */
using L3WorkingVector = std::array<double, 16>;
using L3WorkingMatrix = std::array<std::array<double, 16>, 16>;
using SitePosition = std::array<double, 3>;

/** Dense site-pair response; blocks use row-major (first ISA FDDS response coordinate, second ISA FDDS potential coordinate) order. */
struct PSI_API SitePairResponse {
    double frequency;
    std::vector<SitePosition> positions;
    std::vector<L3WorkingMatrix> blocks;
};

namespace detail {
/** Pure internal validation of cation-state and complete-basis vertical-protocol facts. */
void validate_vertical_protocol(bool cation_state_valid, bool complete_basis_valid);

/** Immutable dense amplitudes inseparably bound to their per-RHS solve diagnostics. */
class DenseRestrictedResponse {
   public:
    DenseRestrictedResponse(const DenseRestrictedResponse&) = default;
    DenseRestrictedResponse& operator=(const DenseRestrictedResponse&) = default;

    SharedMatrix P_clone() const;
    SharedMatrix Q_clone() const;
    std::size_t transition_count() const;
    double reciprocal_condition() const { return reciprocal_condition_; }
    double reciprocal_pivot_growth() const { return reciprocal_pivot_growth_; }
    const std::vector<double>& forward_error() const { return forward_error_; }
    const std::vector<double>& backward_error() const { return backward_error_; }
    const std::vector<double>& scaled_residual() const { return scaled_residual_; }
    const std::vector<double>& solution_column_scales() const { return solution_column_scales_; }

   private:
    friend DenseRestrictedResponse solve_dense_restricted_response(
        const Matrix&, const Matrix&, double, const Matrix&);
    DenseRestrictedResponse(SharedMatrix P, SharedMatrix Q, double reciprocal_condition,
                            double reciprocal_pivot_growth,
                            std::vector<double> forward_error,
                            std::vector<double> backward_error,
                            std::vector<double> scaled_residual,
                            std::vector<double> solution_column_scales);

    SharedMatrix P_;
    SharedMatrix Q_;
    double reciprocal_condition_{};
    double reciprocal_pivot_growth_{};
    std::vector<double> forward_error_;
    std::vector<double> backward_error_;
    std::vector<double> scaled_residual_;
    std::vector<double> solution_column_scales_;
};

/**
 * Enforce the dense solve's scientific-quality budget: RCOND and reciprocal
 * pivot growth >= 1e-12, FERR <= 1e-8, and BERR and independently recomputed
 * scale-aware residual <= 1e-11. All diagnostics must be finite.
 *
 * The forward-error limit leaves four decimal orders beneath downstream 1e-4
 * parity; the backward/residual limits leave seven. The RCOND and pivot-growth
 * floors reject condition amplification or LU growth beyond 1e12, where the
 * LAPACK error estimates themselves cease to be a dependable fail-closed gate.
 */
void validate_dense_response_diagnostics(double reciprocal_condition,
                                         double reciprocal_pivot_growth,
                                         const std::vector<double>& forward_error,
                                         const std::vector<double>& backward_error,
                                         const std::vector<double>& scaled_residual);

/**
 * Solve the amplitude algebra using the native convention H1 = A + B, H2 = A - B:
 *   [[H1, omega I], [-omega I, H2]] [P, Q]^T = [rhs, 0]^T.
 *
 * The RHS convention is deliberately the positive rhs shown above. P and Q are
 * algebraic amplitudes only; this function assigns no polarizability sign or factor.
 * At exactly zero frequency only H1 P = rhs is solved and Q is exactly zero.
 */
DenseRestrictedResponse solve_dense_restricted_response(const Matrix& H1, const Matrix& H2,
                                                         double omega, const Matrix& rhs);
}  // namespace detail

/** Explicit, deterministic controls for the native real-space ISA fixed point. */
class PSI_API ISAOptions {
   public:
    ISAOptions(std::size_t radial_points = 100, std::size_t angular_polar_points = 18,
               std::size_t angular_azimuthal_points = 24, std::size_t max_iterations = 120,
               double convergence = 1.0e-9, double mix_fraction = 1.0,
               double initial_alpha = 1.0, double tail_join_factor = 1.5,
               std::size_t tail_activation_iteration = 20,
               double tail_activation_convergence = 1.0e-6,
               double electron_count_tolerance = 0.1);

    std::size_t radial_points() const { return radial_points_; }
    std::size_t angular_polar_points() const { return angular_polar_points_; }
    std::size_t angular_azimuthal_points() const { return angular_azimuthal_points_; }
    std::size_t max_iterations() const { return max_iterations_; }
    double convergence() const { return convergence_; }
    double mix_fraction() const { return mix_fraction_; }
    double initial_alpha() const { return initial_alpha_; }
    double tail_join_factor() const { return tail_join_factor_; }
    std::size_t tail_activation_iteration() const { return tail_activation_iteration_; }
    double tail_activation_convergence() const { return tail_activation_convergence_; }
    double electron_count_tolerance() const { return electron_count_tolerance_; }

   private:
    std::size_t radial_points_;
    std::size_t angular_polar_points_;
    std::size_t angular_azimuthal_points_;
    std::size_t max_iterations_;
    double convergence_;
    double mix_fraction_;
    double initial_alpha_;
    double tail_join_factor_;
    std::size_t tail_activation_iteration_;
    double tail_activation_convergence_;
    double electron_count_tolerance_;
};

struct PSI_API ISAGridProfile {
    std::size_t radial_points{};
    std::size_t angular_points{};
    std::size_t shell_point_count{};
    std::string angular_rule;
    std::string radial_rule;
    std::string radius_table;
    std::vector<double> atom_scales;
};

/** Deterministic diagnostics for a successfully converged native ISA partition. */
struct PSI_API ISADiagnostics {
    double electron_count{};
    double formal_electron_count{};
    double electron_count_absolute_error{};
    double electron_count_relative_error{};
    std::size_t iterations{};
    bool converged{};
    double max_overlap_residual{};
    double max_population_change{};
    double max_weight_change{};
    double max_unity_residual{};
    double total_charge_residual{};
    std::size_t tail_fit_failures{};
    std::size_t tail_failure_reused_profiles{};
    std::size_t underflow_fallbacks{};
    std::vector<double> atomic_populations;
    ISAGridProfile grid_profile;
    std::vector<std::vector<double>> radial_nodes;
    std::vector<std::vector<double>> log_profiles;
    std::vector<double> tail_join_radii;
    std::vector<double> tail_alphas;
    std::string context_digest;
};

/** Immutable protocol response-kernel selection, independent of the ground-state functional. */
class PSI_API ResponseKernel {
   public:
    ResponseKernel(double chf_exchange, double alda_kernel);

    double chf_exchange() const { return chf_exchange_; }
    double alda_kernel() const { return alda_kernel_; }

   private:
    double chf_exchange_;
    double alda_kernel_;
};

namespace detail {
/** Pure internal/test-only restricted-singlet H1=A+B and H2=A-B carrier. */
struct RestrictedSingletHessian {
    SharedMatrix H1;
    SharedMatrix H2;
};

/**
 * Assemble restricted-singlet response Hessians in the supplied nov transition order.
 * The primitive indexing follows driver/procrouting/response/scf_products.py:
 *   J[ia,jb] = (ia|jb), K_direct[ia,jb] = (ij|ab), and
 *   K_transpose[ia,jb] = (aj|bi).
 *
 * K_transpose names the alternate native exchange contraction, not the ordinary
 * matrix transpose of K_direct. Each transition-space primitive is independently
 * required to be finite and symmetric; it is never silently symmetrized.
 * full_alda is the complete exchange-plus-correlation ALDA kernel constructed
 * separately from the frozen physical grid by the restricted C2b primitive.
 */
RestrictedSingletHessian assemble_restricted_singlet_hessian(
    const std::vector<double>& orbital_gaps, const Matrix& coulomb,
    const Matrix& exchange_direct, const Matrix& exchange_transpose,
    const Matrix& full_alda, const ResponseKernel& kernel);
}  // namespace detail

/** Exact ordered effective DFT-grid state retained by a frozen response context. */
struct PSI_API FrozenGridBlock {
    std::size_t point_offset{};
    std::size_t point_count{};
    std::vector<int> functions_local_to_global;
};

/** Verified GRAC provenance from actual converged neutral, precursor, and cation SCFs. */
struct PSI_API GRACProvenance {
    double neutral_precursor_energy{};
    double cation_energy{};
    double homo_energy{};
    double ionization_potential{};
    double applied_shift{};
    std::string cation_reference;
    int cation_charge{};
    int cation_multiplicity{};
};

/**
 * Frozen single-thread response state with cloned electronic/functional/molecular/grid data.
 * The orbital BasisSet is deliberately retained by const alias under an explicit no-mutation
 * contract and rechecked before future use. Production compute must resolve exclusive ownership
 * across check/use; this alias is not a current response-success claim.
 */
class PSI_API FrozenResponseContext {
   public:
    static std::shared_ptr<FrozenResponseContext> create(
        const std::shared_ptr<Wavefunction>& grac_wfn,
        const std::shared_ptr<Wavefunction>& neutral_precursor_wfn,
        const std::shared_ptr<Wavefunction>& cation_wfn);

    const std::shared_ptr<const Matrix>& Ca() const { return Ca_; }
    const std::shared_ptr<const Matrix>& Cb() const { return Cb_; }
    const std::shared_ptr<const Vector>& epsilon_a() const { return epsilon_a_; }
    const std::shared_ptr<const Vector>& epsilon_b() const { return epsilon_b_; }
    const std::shared_ptr<const Vector>& occupation_a() const { return occupation_a_; }
    const std::shared_ptr<const Vector>& occupation_b() const { return occupation_b_; }
    const std::shared_ptr<const Matrix>& Da() const { return Da_; }
    const std::shared_ptr<const Matrix>& Db() const { return Db_; }
    double energy() const { return energy_; }
    const std::shared_ptr<const Molecule>& molecule() const { return molecule_; }
    const std::shared_ptr<const BasisSet>& basis() const { return basis_; }
    const std::shared_ptr<const SuperFunctional>& functional() const { return functional_; }
    const std::vector<SitePosition>& sites() const { return sites_; }
    const std::vector<double>& grid_points() const { return grid_points_; }
    const std::vector<double>& grid_weights() const { return grid_weights_; }
    const std::vector<FrozenGridBlock>& grid_blocks() const { return grid_blocks_; }
    const GRACProvenance& grac() const { return grac_; }
    const std::string& functional_name() const { return functional_name_; }
    const std::string& grac_x_name() const { return grac_x_name_; }
    const std::string& grac_c_name() const { return grac_c_name_; }
    double functional_density_tolerance() const { return functional_density_tolerance_; }
    std::size_t grid_point_count() const { return grid_weights_.size(); }
    /** Enforce the documented single-thread/no-mutation contract for the retained basis alias. */
    void verify_basis_unchanged() const;

   private:
    FrozenResponseContext(SharedMatrix Ca, SharedMatrix Cb, SharedVector epsilon_a,
                          SharedVector epsilon_b, SharedVector occupation_a,
                          SharedVector occupation_b, SharedMatrix Da, SharedMatrix Db,
                          double energy, std::shared_ptr<const Molecule> molecule,
                          std::shared_ptr<const BasisSet> basis,
                          std::shared_ptr<const BasisSetStructuralSnapshot> basis_snapshot,
                          std::shared_ptr<const SuperFunctional> functional,
                          std::vector<SitePosition> sites, std::vector<double> grid_points,
                          std::vector<double> grid_weights, std::vector<FrozenGridBlock> grid_blocks,
                          GRACProvenance grac, std::string functional_name,
                          std::string grac_x_name, std::string grac_c_name,
                          double functional_density_tolerance);

    std::shared_ptr<const Matrix> Ca_;
    std::shared_ptr<const Matrix> Cb_;
    std::shared_ptr<const Vector> epsilon_a_;
    std::shared_ptr<const Vector> epsilon_b_;
    std::shared_ptr<const Vector> occupation_a_;
    std::shared_ptr<const Vector> occupation_b_;
    std::shared_ptr<const Matrix> Da_;
    std::shared_ptr<const Matrix> Db_;
    double energy_{};
    std::shared_ptr<const Molecule> molecule_;
    // Deliberate alias: safe only under the single-thread/no-mutation contract checked before response.
    std::shared_ptr<const BasisSet> basis_;
    std::shared_ptr<const BasisSetStructuralSnapshot> basis_snapshot_;
    std::shared_ptr<const SuperFunctional> functional_;
    std::vector<SitePosition> sites_;
    std::vector<double> grid_points_;
    std::vector<double> grid_weights_;
    std::vector<FrozenGridBlock> grid_blocks_;
    GRACProvenance grac_{};
    std::string functional_name_;
    std::string grac_x_name_;
    std::string grac_c_name_;
    double functional_density_tolerance_{};
};

/** Up-front storage and integral-work gate for caller-supplied point response. */
struct PSI_API PointResponsePlan {
    std::size_t frequency_count{};
    std::size_t nbf{};
    std::size_t nocc{};
    std::size_t nvir{};
    std::size_t transition_count{};
    std::size_t point_count{};
    std::size_t max_point_count{};
    std::size_t max_frequency_count{};
    std::size_t configured_memory_bytes{};
    std::size_t reserved_memory_bytes{};
    std::size_t ao_matrix_bytes{};
    std::size_t transition_potential_bytes{};
    std::size_t output_bytes{};
    std::size_t output_clone_bytes{};
    std::size_t retained_frequency_bytes{};
    std::size_t retained_points_bytes{};
    std::size_t native_diagnostic_record_bytes{};
    std::size_t native_diagnostics_bytes{};
    std::size_t container_overhead_bytes{};
    std::size_t retained_metadata_bytes{};
    std::size_t python_scalar_diagnostic_overhead_bytes{};
    std::size_t python_metadata_overhead_bytes{};
    std::size_t python_export_overhead_bytes{};
    std::size_t dense_solve_peak_bytes{};
    std::size_t scratch_bytes{};
    std::size_t c1_plan_estimated_bytes{};
    std::size_t alda_plan_estimated_bytes{};
    std::size_t retained_c1_bytes{};
    std::size_t retained_alda_bytes{};
    std::size_t hessian_bytes{};
    std::size_t transition_metadata_bytes{};
    std::size_t conservative_overhead_bytes{};
    std::size_t c1_stage_peak_bytes{};
    std::size_t alda_stage_peak_bytes{};
    std::size_t point_potential_stage_peak_bytes{};
    std::size_t dense_solve_stage_peak_bytes{};
    std::size_t output_clone_stage_peak_bytes{};
    std::size_t estimated_bytes{};
    std::size_t integral_work_terms{};
    std::string algorithm;
    std::string memory_semantics;
};

/**
 * Per-frequency scalar dense-solve and reciprocity summaries. Per-RHS vectors
 * remain transient in DenseRestrictedResponse and are never retained here.
 */
struct PSI_API PointResponseDiagnostics {
    double frequency{};
    double reciprocal_condition{};
    double reciprocal_pivot_growth{};
    double max_forward_error{};
    double max_backward_error{};
    double max_scaled_residual{};
    double max_solution_scale{};
    double allowed_antisymmetry{};
    double symmetry_residual{};
    double max_normalized_antisymmetry{};
    bool reciprocity_enforced{};
};

namespace detail {
struct PointResponseBuilder;
}

/**
 * Immutable, frequency-major external-point response carrier. Matrix access is
 * clone-only so amplitudes/results cannot be mutated through an exported alias.
 */
class PSI_API PointResponseData {
   public:
    PointResponseData(const PointResponseData&) = default;
    PointResponseData& operator=(const PointResponseData&) = default;

    const std::vector<SitePosition>& points() const { return points_; }
    const std::vector<double>& frequencies() const { return frequencies_; }
    std::size_t frequency_count() const { return frequencies_.size(); }
    SharedMatrix response_clone(std::size_t frequency) const;
    std::vector<SharedMatrix> response_clones() const;
    /** Underscored binding support only; not part of the production data model. */
    SharedMatrix transition_potentials_clone_test_only() const;
    const std::vector<PointResponseDiagnostics>& diagnostics() const { return diagnostics_; }
    const PointResponsePlan& plan() const { return plan_; }

   private:
    friend struct detail::PointResponseBuilder;
    PointResponseData(std::vector<SitePosition> points,
                      std::vector<double> frequencies,
                      std::vector<SharedMatrix> responses,
                      std::vector<PointResponseDiagnostics> diagnostics,
                      PointResponsePlan plan, SharedMatrix transition_potentials);

    std::vector<SitePosition> points_;
    std::vector<double> frequencies_;
    std::vector<SharedMatrix> responses_;
    SharedMatrix transition_potentials_test_only_;
    std::vector<PointResponseDiagnostics> diagnostics_;
    PointResponsePlan plan_;
};

/**
 * Evaluate Pi(g,h;omega)=4 sum_ia v(g,ia) P(ia,h) at caller-supplied points.
 * C1, full ALDA, and H1/H2 are constructed internally in the canonical
 * occupied-major transition order from the frozen context and reviewed kernel.
 * The native electronic AO multipole-potential sign is retained in v; because
 * the response is bilinear in v, a global potential-sign convention cancels.
 * No points are generated or refined. Exact duplicate points are rejected;
 * minimum_site_distance_bohr=0 deliberately permits evaluation at nuclei.
 */
PSI_API PointResponseData evaluate_point_response(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const ResponseKernel& kernel, const std::vector<double>& frequencies,
    const std::vector<SitePosition>& points,
    double minimum_site_distance_bohr = 0.0);

namespace detail {
PointResponsePlan plan_point_response_provider(
    std::size_t frequency_count, std::size_t nbf, std::size_t nocc,
    std::size_t nvir, std::size_t point_count,
    const std::vector<FrozenGridBlock>& blocks, bool has_dynamic_frequency,
    std::size_t memory_bytes, double density_cutoff);
/** Explicitly unprovenanced raw-operator seam; production must not call this. */
PointResponseData evaluate_raw_point_response_test_only(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const Matrix& H1, const Matrix& H2, const std::vector<double>& frequencies,
    const std::vector<SitePosition>& points,
    const std::vector<std::size_t>& transition_permutation,
    double minimum_site_distance_bohr = 0.0);

/* Standalone incremental point-stage planner used by the aggregate planner. */
PointResponsePlan plan_point_response(
    std::size_t frequency_count, std::size_t nbf, std::size_t nocc,
    std::size_t nvir, std::size_t point_count, bool has_dynamic_frequency,
    std::size_t memory_bytes);

/**
 * Overflow-checked storage diagnostics for canonical nonsymmetric DirectJK.
 *
 * This intentionally supported envelope is canonical closed-shell water-sized
 * response spaces (at most 512 occupied-virtual transitions, covering water/
 * aug-cc-pVTZ). Three retained dense nov-by-nov matrices impose an unavoidable
 * 24*nov^2-byte payload. DirectJK has no supported peak-memory estimator, so
 * only that retained payload is hard-gated against half the configured process
 * memory; workspace components are conservative protocol diagnostics, not a
 * claim that JK obeys set_memory or a process peak-memory guarantee.
 */
struct RestrictedC1JKPlan {
    std::size_t nbf{};
    std::size_t nocc{};
    std::size_t nvir{};
    std::size_t nov{};
    std::size_t batch_size{};
    std::size_t jk_threads{};
    std::size_t max_supported_nov{};
    std::size_t configured_memory_bytes{};
    std::size_t reserved_memory_bytes{};
    std::size_t retained_payload_bytes{};
    std::size_t metadata_bytes{};
    std::size_t coefficient_bytes{};
    std::size_t matrix_overhead_bytes{};
    std::size_t jk_coefficient_bytes{};
    std::size_t jk_ao_bytes{};
    std::size_t direct_jk_scratch_bytes{};
    std::size_t integral_engine_allowance_bytes{};
    std::size_t projection_bytes{};
    std::size_t estimated_bytes{};
    double integral_cutoff{};
    bool incfock{};
    std::string screening;
    std::string memory_semantics;
    std::string algorithm;
};

/** Plan batch-one DirectJK and reject payload beyond the reserved/envelope limits. */
RestrictedC1JKPlan plan_restricted_c1_jk(std::size_t nbf, std::size_t nocc,
                                         std::size_t nvir, std::size_t memory_bytes);

/** Native C1 restricted transition-space ERI primitives; ALDA is constructed separately. */
struct RestrictedC1Primitives {
    std::vector<std::pair<std::size_t, std::size_t>> transitions;
    std::vector<double> orbital_gaps;
    SharedMatrix coulomb;
    SharedMatrix exchange_direct;
    SharedMatrix exchange_transpose;
    RestrictedC1JKPlan jk_plan;
    std::size_t integral_engine_thread_count{};
};

/** Construct primitives from the immutable orbitals and retained basis of one frozen context. */
RestrictedC1Primitives construct_restricted_c1_primitives(
    const std::shared_ptr<const FrozenResponseContext>& context);

/** Context-bound test path for exercising validation with derived orbital-state variants. */
RestrictedC1Primitives construct_restricted_c1_primitives_test_only(
    const std::shared_ptr<const FrozenResponseContext>& context, const Matrix& Ca, const Matrix& Cb,
    const Vector& epsilon_a, const Vector& epsilon_b, const Vector& occupation_a,
    const Vector& occupation_b);

struct RestrictedALDAPlan {
    std::size_t nbf{};
    std::size_t nocc{};
    std::size_t nvir{};
    std::size_t nov{};
    std::size_t point_count{};
    std::size_t max_block_points{};
    std::size_t max_supported_nov{};
    std::size_t configured_memory_bytes{};
    std::size_t reserved_memory_bytes{};
    std::size_t retained_payload_bytes{};
    std::size_t block_transition_bytes{};
    std::size_t block_weighted_transition_bytes{};
    std::size_t block_mo_scratch_bytes{};
    std::size_t collocation_bytes{};
    std::size_t block_coordinate_weight_bytes{};
    std::size_t block_density_kernel_bytes{};
    std::size_t functional_workspace_bytes{};
    std::size_t point_scratch_bytes{};
    std::size_t metadata_bytes{};
    std::size_t validation_scratch_bytes{};
    std::size_t conservative_overhead_bytes{};
    std::size_t diagnostics_payload_bytes{};
    std::size_t estimated_bytes{};
    std::size_t density_work_terms{};
    std::size_t mo_transition_work_terms{};
    std::size_t ao_collocation_work_terms{};
    std::size_t libxc_work_terms{};
    std::size_t dgemm_work_terms{};
    std::size_t work_terms{};
    std::size_t max_work_terms{};
    double density_cutoff{};
    bool retain_test_diagnostics{};
    std::string density_cutoff_source;
    std::string algorithm;
    std::string memory_semantics;
};

struct RestrictedALDADiagnostics {
    std::string exchange_component;
    std::string correlation_component;
    int exchange_libxc_id{};
    int correlation_libxc_id{};
    std::string exchange_libxc_canonical_name;
    std::string correlation_libxc_canonical_name;
    std::map<std::string, double> exchange_effective_parameters;
    std::map<std::string, double> correlation_effective_parameters;
    double exchange_coefficient{};
    double correlation_coefficient{};
    int derivative_order{};
    double density_cutoff{};
    std::string density_cutoff_source;
    std::size_t point_count{};
    std::string restricted_normalization;
    RestrictedALDAPlan plan;
};

/** Full exchange-plus-correlation ALDA primitive on the exact sealed ordered grid. */
struct RestrictedALDAPrimitive {
    std::vector<std::pair<std::size_t, std::size_t>> transitions;
    SharedMatrix full_alda;
    RestrictedALDADiagnostics diagnostics;
    /** Populated only by the explicitly memory-gated test diagnostics mode. */
    std::vector<double> densities;
    std::vector<double> fxc;
    std::vector<double> transition_values;
};

RestrictedALDAPlan plan_restricted_alda(std::size_t nbf, std::size_t nocc,
                                        std::size_t nvir, std::size_t point_count,
                                        const std::vector<FrozenGridBlock>& blocks,
                                        std::size_t memory_bytes,
                                        bool retain_test_diagnostics,
                                        double density_cutoff);
RestrictedALDAPrimitive construct_restricted_alda_kernel(
    const std::shared_ptr<const FrozenResponseContext>& context,
    bool retain_test_diagnostics = false);
SharedMatrix contract_restricted_alda_test_only(
    const std::vector<double>& weights, const Matrix& transition_values,
    const std::vector<double>& densities, const std::vector<double>& fxc,
    double density_cutoff);
std::pair<std::vector<double>, RestrictedALDADiagnostics> evaluate_restricted_alda_fxc_test_only(
    const std::vector<double>& densities, bool include_correlation,
    double density_cutoff);
void validate_restricted_alda_grid_test_only(std::size_t nbf, std::size_t point_count,
                                             const std::vector<double>& weights,
                                             const std::vector<FrozenGridBlock>& blocks);
std::size_t validate_restricted_alda_work_bound_test_only(std::size_t work_terms);
struct RestrictedALDACollocationTestResult {
    std::size_t point_count{};
    std::size_t nbf{};
    std::vector<double> ao_values;
};
RestrictedALDACollocationTestResult collocate_restricted_alda_ao_target_test_only(
    const std::shared_ptr<const FrozenResponseContext>& context);

/** Resource-gated C3a projection plan; production streams tau one sealed block at a time. */
struct TransitionMultipoleProjectionPlan {
    std::size_t point_count{};
    std::size_t site_count{};
    std::size_t transition_count{};
    std::size_t max_block_points{};
    std::size_t output_bytes{};
    std::size_t block_scratch_bytes{};
    std::size_t estimated_bytes{};
    std::size_t work_terms{};
    std::size_t max_work_terms{};
    std::size_t max_site_count{};
    std::string algorithm;
};

/** Site-major B[A,t,k], with t in 00;10,11c,11s;...;33c,33s order. */
struct TransitionMultipoleProjection {
    std::vector<std::pair<std::size_t, std::size_t>> transitions;
    SharedMatrix values;
    TransitionMultipoleProjectionPlan plan;
};

TransitionMultipoleProjectionPlan plan_transition_multipole_projection(
    std::size_t point_count, std::size_t site_count, std::size_t transition_count,
    std::size_t max_block_points, std::size_t nbf, std::size_t nmo,
    std::size_t memory_bytes);

/** Pure C3a equation evaluator over caller-supplied tau[p,k]. */
TransitionMultipoleProjection project_transition_multipoles(
    const std::vector<SitePosition>& points, const std::vector<double>& weights,
    const std::vector<double>& partition, const std::vector<SitePosition>& sites,
    const Matrix& transition_values);

/** Internal friend carrier enforcing ISA/context identity before production projection. */
struct TransitionMultipoleProjector {
    static TransitionMultipoleProjection project(
        const std::shared_ptr<const FrozenResponseContext>& context,
        const ISAWeights& isa_weights);
};

/** Resource-gated plan for the pure C3b site-pair response contraction. */
struct SitePairResponseContractionPlan {
    std::size_t site_count{};
    std::size_t transition_count{};
    std::size_t component_count{};
    std::size_t output_bytes{};
    std::size_t scratch_bytes{};
    std::size_t estimated_bytes{};
    std::size_t work_terms{};
    std::size_t max_work_terms{};
    std::size_t max_site_count{};
    std::string algorithm;
    /** Incremental numeric payload only; caller-owned B and dense response are excluded. */
    std::string memory_semantics;
};

/** Ordered alpha[(response site,t),(source site,u)] in unchanged ISA component order. */
struct SitePairResponseContraction {
    SharedMatrix values;
    SitePairResponseContractionPlan plan;
    double restricted_factor{};
    double response_map_forward_error_bound{};
    double response_map_solution_scale{};
    double response_map_allowed_antisymmetry{};
    double response_map_symmetry_residual{};
    double response_map_max_normalized_antisymmetry{};
    bool reciprocity_enforced{};
};

SitePairResponseContractionPlan plan_site_pair_response_contraction(
    std::size_t site_count, std::size_t transition_count, std::size_t memory_bytes);

/** Pure response-map symmetry diagnostics shared by production and a test-only validator. */
struct ResponseMapSymmetryDiagnostics {
    double solution_scale{};
    double allowed_antisymmetry{};
    double symmetry_residual{};
    double max_normalized_antisymmetry{};
};

/**
 * Pure C3b evaluator: alpha[A,B](t,u) = 4 B[A,t,:] G B[B,u,:]^T.
 * G and its validated solve diagnostics are inseparable in response; its maximum
 * FERR bounds averaging of solver roundoff before reciprocity is enforced.
 */
SitePairResponseContraction contract_site_pair_response(
    std::size_t site_count, const Matrix& projection,
    const DenseRestrictedResponse& response);

/** Pure synthetic diagnostics seam; it does not construct or contract a response. */
ResponseMapSymmetryDiagnostics validate_response_map_symmetry_test_only(
    const Matrix& response_map, const Matrix& conjugate_map,
    const std::vector<double>& forward_error);

/** Up-front stagewise simultaneous-live storage gate for physical C4 wiring. */
struct ISAPolResponsePlan {
    std::size_t frequency_count{};
    std::size_t site_count{};
    std::size_t nbf{};
    std::size_t nocc{};
    std::size_t nvir{};
    std::size_t transition_count{};
    std::size_t point_count{};
    std::size_t max_block_points{};
    std::size_t component_count{};
    std::size_t configured_memory_bytes{};
    std::size_t reserved_memory_bytes{};
    std::size_t c1_plan_estimated_bytes{};
    std::size_t alda_plan_estimated_bytes{};
    std::size_t projection_plan_estimated_bytes{};
    std::size_t contraction_plan_estimated_bytes{};
    std::size_t retained_c1_bytes{};
    std::size_t retained_alda_bytes{};
    std::size_t hessian_bytes{};
    std::size_t retained_projection_bytes{};
    std::size_t identity_bytes{};
    std::size_t retained_output_bytes{};
    std::size_t dense_solve_peak_bytes{};
    std::size_t response_carrier_bytes{};
    std::size_t transition_metadata_bytes{};
    std::size_t conservative_overhead_bytes{};
    std::size_t c1_stage_peak_bytes{};
    std::size_t alda_stage_peak_bytes{};
    std::size_t projection_stage_peak_bytes{};
    std::size_t dense_solve_stage_peak_bytes{};
    std::size_t contraction_stage_peak_bytes{};
    std::size_t estimated_bytes{};
    std::string algorithm;
    std::string memory_semantics;
};

ISAPolResponsePlan plan_isapol_response_provider(
    std::size_t frequency_count, std::size_t site_count,
    std::size_t nbf, std::size_t nocc, std::size_t nvir,
    std::size_t point_count, const std::vector<FrozenGridBlock>& blocks,
    bool has_dynamic_frequency, std::size_t memory_bytes,
    double density_cutoff);

/** Policy and deterministic numerical gates for the pure refinement solve. */
struct ConstrainedLeastSquaresOptions {
    double column_cutoff{1.0e-4};
    bool prune_below_cutoff{true};
    double maximum_condition_number{1.0e12};
    double rank_tolerance{1.0e-12};
    std::size_t maximum_workspace_elements{std::numeric_limits<std::size_t>::max()};
};

/** Auditable economy-SVD allocation counts, in scalar elements. */
struct ConstrainedLeastSquaresAllocationPlan {
    std::size_t constraint_rows{};
    std::size_t constraint_columns{};
    std::size_t constraint_u_elements{};
    std::size_t constraint_vt_elements{};
    std::size_t fit_rows{};
    std::size_t fit_columns{};
    std::size_t fit_u_elements{};
    std::size_t fit_vt_elements{};
};

/** Complete solution and diagnostics; constructed only after every gate succeeds. */
struct ConstrainedLeastSquaresResult {
    std::vector<double> solution;
    std::vector<std::size_t> kept_columns;
    std::vector<std::size_t> pruned_columns;
    std::vector<int> full_to_reduced;
    std::vector<double> column_weighted_norms;
    std::vector<double> singular_values;
    std::size_t rank{};
    std::size_t constraint_rank{};
    std::size_t free_dimension{};
    double condition_number{};
    double weighted_residual_norm{};
    double anchor_residual_norm{};
    double constraint_residual_norm{};
    double objective_residual_norm{};
    double lambda{};
    double row_weight_min{};
    double row_weight_max{};
    std::string row_weight_source;
    ConstrainedLeastSquaresAllocationPlan allocation_plan;
};

/**
 * Solve min ||W(Ax-b)||^2 + lambda ||D(x-x0)||^2 subject to Cx=d.
 * W is supplied as row_weights and D as diagonal_anchor. No normal equations
 * are formed; equality elimination and the reduced fit both use direct SVDs.
 */
ConstrainedLeastSquaresResult solve_constrained_least_squares(
    const Matrix& design, const std::vector<double>& observations,
    const std::vector<double>& row_weights, double lambda,
    const std::vector<double>& diagonal_anchor, const std::vector<double>& reference,
    const Matrix& constraints, const std::vector<double>& constraint_targets,
    const ConstrainedLeastSquaresOptions& options);
}  // namespace detail

/** Actual ISA data structurally bound to one exact frozen context and its ordered grid/sites. */
class PSI_API ISAWeights {
   public:
    /** Existing arbitrary-array seam, deliberately named and restricted to tests. */
    static ISAWeights create_test_only(std::shared_ptr<const FrozenResponseContext> context,
                                       std::vector<double> partition_weights);

    std::size_t point_count() const;
    std::size_t site_count() const;
    const std::vector<double>& partition_weights() const { return partition_weights_; }
    const ISADiagnostics& diagnostics() const { return diagnostics_; }

   private:
    friend class ISAPolResponseProvider;
    friend struct detail::TransitionMultipoleProjector;
    friend ISAWeights compute_isa_weights(std::shared_ptr<const FrozenResponseContext>, const ISAOptions&);
    ISAWeights(std::shared_ptr<const FrozenResponseContext> context,
               std::vector<double> partition_weights, ISADiagnostics diagnostics);

    std::shared_ptr<const FrozenResponseContext> context_;
    std::vector<double> partition_weights_;
    ISADiagnostics diagnostics_;
};

/** Compute native real-space ISA probabilities on the exact sealed response grid. */
PSI_API ISAWeights compute_isa_weights(std::shared_ptr<const FrozenResponseContext> context,
                                       const ISAOptions& options = ISAOptions());

/** Production C3a projection bound to the exact ISA context/grid; tau is block-streamed. */
PSI_API detail::TransitionMultipoleProjection project_transition_multipoles(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const ISAWeights& isa_weights);

namespace detail {
struct PSI_API SyntheticGaussianDensity {
    SitePosition center;
    double coefficient{};
    double exponent{};
};
struct PSI_API SyntheticISAResult {
    std::size_t site_count{};
    std::vector<double> weights;
    ISADiagnostics diagnostics;
};
struct PSI_API ISAProfileTestResult {
    std::vector<double> log_values;
    double tail_alpha{};
    double tail_log_amplitude{};
    double tail_charge{};
    double join_log_left{};
    double join_log_right{};
};
struct PSI_API GaussianFixedPointTestResult {
    std::vector<double> weights;
    double max_profile_relative_error{};
};
struct PSI_API ISAOverlapTestResult {
    double overlap_residual{};
};
SyntheticISAResult compute_synthetic_isa(const std::vector<SitePosition>& sites,
                                         const std::vector<SitePosition>& output_points,
                                         const std::vector<double>& output_weights,
                                         const std::vector<int>& atomic_numbers,
                                         const std::vector<SyntheticGaussianDensity>& terms,
                                         const ISAOptions& options,
                                         std::size_t inject_tail_fit_failure_iteration = 0,
                                         std::size_t test_min_iterations = 0);
ISAProfileTestResult test_isa_profile(const std::vector<double>& nodes,
                                      const std::vector<double>& log_values,
                                      const std::vector<double>& queries,
                                      double tail_join, double tail_charge);
GaussianFixedPointTestResult test_isa_gaussian_fixed_point(
    const std::vector<SitePosition>& sites, const std::vector<SitePosition>& output_points,
    const std::vector<SyntheticGaussianDensity>& terms, std::size_t radial_points,
    std::size_t angular_polar_points, std::size_t angular_azimuthal_points);
ISAOverlapTestResult test_isa_overlap(
    const std::vector<double>& first_nodes, const std::vector<double>& first_logs,
    double first_tail_alpha, double first_tail_log_amplitude,
    const std::vector<double>& second_nodes, const std::vector<double>& second_logs,
    double second_tail_alpha, double second_tail_log_amplitude,
    double tail_join, std::size_t integration_points);
std::vector<double> test_isa_tail_probabilities(const std::vector<double>& tail_log_amplitudes,
                                                const std::vector<double>& tail_alphas,
                                                const std::vector<double>& distances);
}  // namespace detail

/** Production response interface; no mutable Wavefunction is retained or revalidated. */
class PSI_API ISAPolResponseProvider {
   public:
    ISAPolResponseProvider(std::shared_ptr<const FrozenResponseContext> context,
                           ResponseKernel kernel, ISAWeights isa_weights);

    std::size_t expected_response_count(const FrequencyGrid& frequencies) const;
    std::vector<SitePairResponse> compute_isapol_response(const FrequencyGrid& frequencies) const;

   private:
    std::shared_ptr<const FrozenResponseContext> context_;
    ResponseKernel kernel_;
    ISAWeights isa_weights_;
};

/** Unweighted undirected graph over the polarizable sites. */
struct PSI_API BondGraph {
    std::size_t site_count;
    std::vector<std::array<std::size_t, 2>> bonds;
};

struct PSI_API BondTransfer {
    std::size_t first;
    std::size_t second;
    std::size_t first_component;
    std::size_t second_component;
    std::size_t fixed_site;
    double amount;
};

struct PSI_API LocalizationResiduals {
    double off_site;
    double charge_sum;
    double reciprocity;
    double molecular_sum;
    double local_charge;
};

/** Fully localized rank-1-through-rank-3 response and deterministic diagnostics. */
struct PSI_API LocalizedResponse {
    /** Exact physical frequency inherited from the site-pair response. */
    double frequency;
    /** Ordered physical sites inherited from the localized site-pair response. */
    std::vector<SitePosition> positions;
    std::vector<L3Matrix> local;
    std::vector<BondTransfer> transfers;
    LocalizationResiduals residuals;
    /** Deterministic diagnostics consumed only by the underscored math-test seam. */
    std::vector<L3WorkingMatrix> refined_pairs;
    std::vector<std::array<std::size_t, 2>> omitted_component_pairs;
    std::size_t omitted_transfer_count;
};

/** Full site-major upper-triangle variable policy and optional equality Cx=d. */
struct PSI_API RefinementConstraints {
    /** Empty means all 120 variables per site are active; false fixes a variable to zero. */
    std::vector<bool> active_variables;
    SharedMatrix equality;
    std::vector<double> equality_targets;
};

/** The parsed polarizability-definition constraints used by WSM refinement. */
using PDefConstraints = RefinementConstraints;

/** Local-to-global site axis frame; column j holds local axis j in molecular coordinates. */
using SiteAxes = std::array<std::array<double, 3>, 3>;

/** Exact point-group classification of the 15 L3 components at one site's local frame. */
struct PSI_API SiteSymmetry {
    /** libmints SymmOps bit mask of the operations that leave this site fixed. */
    unsigned char point_group_bits{};
    /** Directional Schoenflies label from PointGroup::bits_to_full_name. */
    std::string point_group;
    /** Diagonal local-frame sign triples of the site-group operations, in char-table order. */
    std::vector<std::array<int, 3>> operation_signs;
    /** Irreducible-representation class index of each L3 component, in first-seen order. */
    std::array<std::size_t, 15> component_class{};
    std::size_t class_count{};
    /** Upper-triangle (t, u) component pairs that remain fit variables. */
    std::vector<std::array<std::size_t, 2>> active_pairs;
    /** Orbit representative whose variables this site copies; the site itself when independent. */
    std::size_t symmetry_source{};
    /** Local-frame sign triple of the operation carrying symmetry_source onto this site. */
    std::array<int, 3> copy_signs{{1, 1, 1}};
};

/** Auditable PDef derivation: the mask, the symmetry-copy equalities, and their provenance. */
struct PSI_API PDefDerivation {
    PDefConstraints constraints;
    std::vector<SiteSymmetry> sites;
    /** Directional label of the largest D2h subgroup realized by the current molecular frame. */
    std::string molecular_point_group;
    std::size_t variable_count{};
    std::size_t active_variable_count{};
    std::size_t equality_row_count{};
    std::size_t independent_variable_count{};
    double geometry_tolerance{};
};

/**
 * Derive the WSM active-variable mask and symmetry-copy equalities from site symmetry.
 *
 * An upper-triangle pair (t, u) is a fit variable exactly when the real solid harmonics t
 * and u carry the same one-dimensional irreducible representation of the site's point group
 * in that site's local axis frame; every other pair is frozen at zero by omitting its column
 * from the design matrix. Sites related by a molecular symmetry operation are tied to the
 * lowest-indexed member of their orbit by exact +-1 equality rows instead of being fitted.
 *
 * The site group is the subgroup of the molecular point group fixing the site. Only the D2h
 * subgroups libmints itself detects occur, so every operation is a diagonal sign triple and
 * every character is an integer product: the mask is bit-reproducible with no tolerance test.
 *
 * site_axes is empty for the molecular frame or supplies one right-handed orthonormal
 * local-to-global frame per site. Fails closed when the molecular frame does not realize the
 * detected point group, when a local frame is not orthonormal or right-handed, and when a
 * local frame fails to diagonalize a site-group operation.
 *
 * The returned mask and equalities index variables in whichever frame site_axes selects, so
 * refine_wsm - whose design matrix uses molecular-frame harmonics - requires the empty
 * default. The sixteen-site derivation envelope is looser than refine_wsm's own three-site
 * variable envelope, which still applies downstream.
 */
PSI_API PDefDerivation derive_pdef_constraints(const Molecule& molecule,
                                              const std::vector<SiteAxes>& site_axes = {});

/**
 * Documented covalent-bonding scale factor for derive_bond_graph.
 *
 * Sites bond when their separation is at most this factor times the sum of their
 * Bragg-Slater radii. Against the Slater-1964-bohr-v1 table this factor sits above every
 * first- and second-row single-bond ratio the table produces (the largest is the peroxide
 * O-O bond at 1.23) and below the tightest ordinary nonbonded contact ratio (a four-
 * membered-ring 1,3 C...C diagonal at about 1.53, water 1,3 H...H at 2.17). Homonuclear
 * F2 sits at 1.41 because the table's fluorine radius is anomalously small; that case is
 * reported as a disconnected graph rather than silently guessed.
 */
constexpr double kCovalentBondScale = 1.3;

/** Deterministic covalent bond-graph derivation with its auditable distance record. */
struct PSI_API BondGraphDerivation {
    BondGraph graph;
    double covalent_scale{};
    std::string radius_table;
    std::vector<double> radii;
    /** Bond-ordered separations and acceptance thresholds, in bohr. */
    std::vector<double> bond_distances;
    std::vector<double> bond_thresholds;
    std::size_t component_count{};
    std::vector<std::size_t> component_labels;
};

/**
 * Derive a connected covalent bond graph from the molecular geometry.
 *
 * Bonds are the site pairs i < j with |r_i - r_j| <= covalent_scale * (R_i + R_j), using the
 * existing versioned libmints Bragg-Slater radius table. The bond list is sorted and
 * orientation-independent because only interatomic distances enter. LW localization over a
 * disconnected graph is not meaningful, so a graph with more than one connected component
 * fails closed instead of yielding isolated sites.
 */
PSI_API BondGraphDerivation derive_bond_graph(const Molecule& molecule,
                                             double covalent_scale = kCovalentBondScale);

namespace detail {
/** Bragg-Slater radius in bohr from the versioned Slater-1964-bohr-v1 table. */
PSI_API double slater_radius(int atomic_number);

/** Pure geometry seam shared by the molecular derivation and the math tests. */
BondGraphDerivation derive_bond_graph(const std::vector<SitePosition>& sites,
                                     const std::vector<int>& atomic_numbers,
                                     double covalent_scale);
}  // namespace detail

/** Exact reviewed physical WSM policy; only the condition gate is caller-tunable. */
struct PSI_API RefinementOptions {
    unsigned int wsm_rank{3};
    unsigned int hydrogen_rank{3};
    unsigned int weight_type{4};
    double weight_coefficient{0.001};
    double cutoff{1.0e-4};
    double maximum_condition_number{1.0e12};
};

/** Up-front dense-design resource accounting. */
struct PSI_API WSMRefinementPlan {
    std::size_t point_count{};
    std::size_t pair_rows{};
    std::size_t site_count{};
    std::size_t variable_count{};
    std::size_t active_variable_count{};
    std::size_t constraint_rows{};
    std::size_t irregular_elements{};
    std::size_t response_clone_bytes{};
    std::size_t design_elements{};
    std::size_t design_bytes{};
    std::size_t constraint_matrix_bytes{};
    std::size_t null_space_elements{};
    std::size_t null_space_bytes{};
    std::size_t workspace_elements{};
    std::size_t workspace_bytes{};
    std::size_t constraint_svd_peak_bytes{};
    std::size_t fit_svd_peak_bytes{};
    std::size_t estimated_bytes{};
    std::size_t configured_memory_bytes{};
    std::size_t reserved_memory_bytes{};
    std::string algorithm;
    std::string memory_semantics;
};

/** Auditable result metadata for one physical WSM fit. */
struct PSI_API RefinementDiagnostics {
    std::size_t point_count{};
    std::size_t pair_rows{};
    std::size_t variable_count{};
    std::size_t active_variable_count{};
    std::size_t anchor_variable_count{};
    std::vector<double> solution;
    std::vector<std::size_t> kept_variables;
    std::vector<std::size_t> pruned_variables;
    double condition_number{};
    double weighted_residual_norm{};
    double anchor_residual_norm{};
    double constraint_residual_norm{};
    double objective_residual_norm{};
    double max_point_residual{};
    double max_output_asymmetry{};
    std::string row_weight_source;
    WSMRefinementPlan plan;
};

/** Symmetric site-local L3 model at one physical response frequency. */
struct PSI_API RefinedL3Model {
    double frequency{};
    std::vector<SitePosition> positions;
    std::vector<L3Matrix> tensors;
    RefinementDiagnostics diagnostics;
};

/** Refine exactly one frequency; the PointResponseData carrier must contain one response. */
PSI_API RefinedL3Model refine_wsm(const LocalizedResponse& localized,
                                  const PointResponseData& point_response,
                                  const PDefConstraints& constraints = PDefConstraints(),
                                  const RefinementOptions& options = RefinementOptions());

/** Frequency-major wrapper; localized responses and point responses must have equal counts. */
PSI_API std::vector<RefinedL3Model> refine_wsm(
    const std::vector<LocalizedResponse>& localized,
    const PointResponseData& point_response,
    const PDefConstraints& constraints = PDefConstraints(),
    const RefinementOptions& options = RefinementOptions());

namespace detail {
WSMRefinementPlan plan_wsm_refinement(std::size_t point_count, std::size_t site_count,
                                      std::size_t active_variable_count,
                                      std::size_t constraint_rows,
                                      std::size_t memory_bytes);
L3WorkingVector irregular_harmonics_test_only(const SitePosition& point,
                                               const SitePosition& site);
std::vector<RefinedL3Model> refine_wsm_test_only(
    const std::vector<SitePosition>& points, const std::vector<double>& frequencies,
    const std::vector<SharedMatrix>& responses, const std::vector<SitePosition>& sites,
    const std::vector<std::vector<L3Matrix>>& localized,
    const std::vector<double>& localized_frequencies,
    const PDefConstraints& constraints, const RefinementOptions& options);
}  // namespace detail

PSI_API Matrix lw_graph_operator(const BondGraph& graph);
PSI_API std::pair<Matrix, std::vector<double>> lw_graph_pseudoinverse(const BondGraph& graph);
PSI_API L3WorkingVector translate_l3_multipoles(const L3WorkingVector& source,
                                                 const SitePosition& source_minus_target);
/** Localize independently within every graph component; reject component-inconsistent flow. */
PSI_API LocalizedResponse localize_lw(const SitePairResponse& response, const BondGraph& graph,
                                      double residual_tolerance);

/** Build the required static plus ten-point imaginary-frequency grid. */
PSI_API FrequencyGrid make_casimir_grid(unsigned int nonzero_count, double scale);

/** Extract and reorder the real-spherical dipole block as Cartesian x, y, z. */
PSI_API Matrix local_spherical_dipole_to_cartesian(const L3Matrix& spherical);

/** Rotate a symmetric tensor from a right-handed local frame into the global frame. */
PSI_API Matrix rotate_tensor(const Matrix& local, const Matrix& local_to_global);

/** Pack a symmetric Cartesian tensor as xx, xy, xz, yy, yz, zz. */
PSI_API std::array<double, 6> pack_symmetric_tensor(const Matrix& tensor);

/** Native atomic-polarizability pipeline entry point. */
class PSI_API AtomicPolarizabilityCalculator {
   public:
    explicit AtomicPolarizabilityCalculator(std::shared_ptr<Wavefunction> wfn);

    /** Compute and publish the atomic polarizability and dispersion arrays. */
    void compute();

   private:
    void validate_wavefunction_prerequisites() const;

    std::shared_ptr<Wavefunction> wfn_;
};

}  // namespace psi

#endif  // PSI4_SRC_PSI4_LIBMINTS_ATOMIC_POLARIZABILITY_H
