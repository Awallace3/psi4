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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libmints/typedefs.h"

namespace psi {

/**
 * Named fail-closed error for every native atomic-polarizability prerequisite gate.
 *
 * Every message carries the class name, so a caller can distinguish a missing or
 * inconsistent prerequisite from an ordinary numerical failure. The pipeline publishes
 * nothing at all when this is thrown; partial output is never produced.
 */
class PSI_API AtomicPolarizabilityPrerequisiteError : public PsiException {
   public:
    AtomicPolarizabilityPrerequisiteError(const std::string& message, const char* file,
                                          int line) noexcept
        : PsiException("AtomicPolarizabilityPrerequisiteError: " + message, file, line) {}
};

#define ATOMIC_POLARIZABILITY_PREREQUISITE(message) \
    AtomicPolarizabilityPrerequisiteError(message, __FILE__, __LINE__)

class BasisSet;
struct BasisSetStructuralSnapshot;
class Matrix;
class Molecule;
class Options;
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

    /**
     * Second-stage factory that additionally seals an auxiliary basis.
     *
     * auxiliary_key must already resolve on the GRAC wavefunction; an empty key
     * is exactly the three-argument factory. The auxiliary basis is snapshotted the
     * same way the orbital basis is and rechecked by verify_basis_unchanged, because
     * a partition comparison whose auxiliary basis is unattested cannot support the
     * claim that only the partition differs.
     */
    static std::shared_ptr<FrozenResponseContext> create(
        const std::shared_ptr<Wavefunction>& grac_wfn,
        const std::shared_ptr<Wavefunction>& neutral_precursor_wfn,
        const std::shared_ptr<Wavefunction>& cation_wfn,
        const std::string& auxiliary_key);

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
    /** Sealed auxiliary basis, or null when none was attached. */
    const std::shared_ptr<const BasisSet>& auxiliary_basis() const { return auxiliary_basis_; }
    const std::string& auxiliary_basis_key() const { return auxiliary_basis_key_; }
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
                          double functional_density_tolerance,
                          std::shared_ptr<const BasisSet> auxiliary_basis,
                          std::shared_ptr<const BasisSetStructuralSnapshot> auxiliary_basis_snapshot,
                          std::string auxiliary_basis_key);

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
    // Deliberate alias under the same contract as basis_, and rechecked alongside it.
    std::shared_ptr<const BasisSet> auxiliary_basis_;
    std::shared_ptr<const BasisSetStructuralSnapshot> auxiliary_basis_snapshot_;
    std::string auxiliary_basis_key_;
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
 * This routine only evaluates; it never builds or refines a point set. The
 * production point source is generate_wsm_fit_points below. Exact duplicate
 * points are rejected; minimum_site_distance_bohr=0 deliberately permits
 * evaluation at nuclei.
 */
PSI_API PointResponseData evaluate_point_response(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const ResponseKernel& kernel, const std::vector<double>& frequencies,
    const std::vector<SitePosition>& points,
    double minimum_site_distance_bohr = 0.0);

/* ==> Symmetry-faithful WSM fit-point generation <== */

/** Row-major Cartesian 3 by 3 symmetry operation acting as p -> S p. */
using FitPointOperation = std::array<double, 9>;

/**
 * Radial convention for the fit-point shell limits.
 *
 * Bohr treats the limits as absolute distances from the nearest nucleus.
 * VanDerWaals scales them by the Bondi radius of the nearest nucleus.
 * Bohr is the reviewed protocol's convention: the reviewed point-to-point grid
 * spans 4.63 to 11.46 bohr from the nearest nucleus, an absolute band that is
 * not a fixed multiple of either the O or the H Bondi radius.
 *
 * A previous revision justified Bohr by observing that the van der Waals reading
 * pushes the rank-3 design columns below the fixed 1e-4 WSM cutoff. That was a
 * symptom of comparing the cutoff against an *absolute* weighted column norm and
 * is no longer relevant: refine_wsm now scales the cutoff by the largest weighted
 * column norm, so it is a rank threshold rather than an atomic-unit magnitude.
 */
enum class FitPointRadialUnits { Bohr, VanDerWaals };

/**
 * Deterministic nested-equidistant-surface fit-point policy.
 *
 * The shell limits must keep every fit point outside the molecular charge
 * density. A rank-3 distributed multipole model cannot represent the
 * point-to-point response of a point that penetrates the density, and fitting to
 * such points drives the fitted polarizabilities far below the response they were
 * derived from. Measured on H2O/aug-cc-pVDZ with PBE0 and the reviewed kernel:
 * a conserving LW-localized model overpredicts the ab initio point response by a
 * factor of 5.6 at 2.0 bohr, and reproduces it to 1.3 percent at 4.0 bohr and to
 * 0.4 percent beyond 6 bohr. The 4.5 to 11.5 bohr default brackets the reviewed
 * grid's measured 4.63 to 11.46 bohr span.
 */
struct PSI_API FitPointOptions {
    /** Lebedev nodes per atom per shell; must be a supported Lebedev size. */
    std::size_t spherical_points{50};
    /** Shells spanning the closed interval [inner_limit, outer_limit]. */
    std::size_t radial_shells{5};
    double inner_limit{4.5};
    double outer_limit{11.5};
    FitPointRadialUnits radial_units{FitPointRadialUnits::Bohr};
    /** Hard ceiling; the WSM refinement envelope is 500 points. */
    std::size_t maximum_points{500};
    /** Coincident-point merge radius in bohr. */
    double merge_tolerance_bohr{1.0e-8};
};

/** Up-front candidate and storage bound, computed before any point is built. */
struct PSI_API FitPointPlan {
    std::size_t atom_count{};
    std::size_t spherical_points{};
    std::size_t radial_shells{};
    std::size_t lebedev_order{};
    std::size_t symmetry_operation_count{};
    std::size_t candidate_count{};
    std::size_t point_count{};
    std::size_t maximum_points{};
    std::size_t candidate_bytes{};
    std::size_t retained_metadata_bytes{};
    std::size_t estimated_bytes{};
    std::vector<double> shell_offsets;
    std::string radial_units;
    std::string algorithm;
};

/** Generated fit points with the provenance needed to audit shell membership. */
struct PSI_API FitPointSet {
    std::vector<SitePosition> points;
    /** min over atoms of |p - R_A| / scaling_radii[A]; equals the point's shell offset. */
    std::vector<double> nearest_offsets;
    std::vector<std::size_t> shell_index;
    std::vector<std::size_t> generator_atom;
    /** Per-atom radial scale in bohr; all ones under the bohr convention. */
    std::vector<double> scaling_radii;
    /** Verified largest displacement when every operation maps the set onto itself. */
    double max_symmetry_deviation{};
    /** Verified largest |F^T S F| departure from a signed coordinate permutation. */
    double max_octahedral_deviation{};
    FitPointPlan plan;
};

/** Bondi (1964)/Mantina (2009) van der Waals radius in bohr; throws off table. */
PSI_API double bondi_vdw_radius_bohr(int atomic_number);

/** Identity angular frame, i.e. Lebedev axes aligned with the Cartesian axes. */
PSI_API FitPointOperation identity_fit_point_frame();

/** Bound the candidate set and storage before generating anything. */
PSI_API FitPointPlan plan_fit_points(std::size_t atom_count, const FitPointOptions& options);

/**
 * Build the union over shells and atoms of the Lebedev-sampled surfaces at each
 * shell offset, keeping only the nodes no closer to any other nucleus, and merge
 * coincident points.
 *
 * angular_frame is the proper rotation carrying the Lebedev axes into the frame
 * the molecule is expressed in; the Lebedev node u is placed along
 * angular_frame * u. Every symmetry operation must be orthogonal, must map the
 * nuclear framework onto itself, and must be a signed coordinate permutation in
 * the angular frame, i.e. an element of O_h there. Those three conditions make
 * the node set exactly invariant, which is then verified as a postcondition:
 * an arbitrary point set would inject symmetry-violating residuals into the
 * fitted anisotropy, so anything else fails closed rather than fitting.
 *
 * The result carries no RNG, hash-order, or iteration-order dependence, and
 * because the enumeration is purely index driven, rotating centers, operations
 * and angular_frame by R reproduces the same points transformed by R.
 */
PSI_API FitPointSet generate_fit_points(const std::vector<int>& atomic_numbers,
                                       const std::vector<SitePosition>& centers,
                                       const std::vector<FitPointOperation>& symmetry_operations,
                                       const FitPointOperation& angular_frame,
                                       const FitPointOptions& options);

/** Read the ATOMIC_POLARIZABILITY_FIT_* keywords into a validated policy. */
PSI_API FitPointOptions fit_point_options_from(Options& options);

/**
 * Production fit-point source for refine_wsm: reads the ATOMIC_POLARIZABILITY_FIT_*
 * keywords and closes over the symmetry operations of the molecule's own point group.
 */
PSI_API FitPointSet generate_wsm_fit_points(const Molecule& molecule, Options& options);

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

/** Rank-0-through-rank-3 real-spherical component count of one auxiliary moment row. */
constexpr std::size_t kAuxiliaryMomentComponents = 16;

/** How the auxiliary-space fit treats its linear constraint set. */
enum class PSI_API CDFConstraintPolicy {
    /**
     * Minimise Delta + penalty * ||C d - n||^2. The reviewed reference calculation
     * uses this form with a finite weight, and its fitted transition densities
     * therefore violate the charge condition by a small but nonzero amount. A hard
     * constraint would give machine zero and a different partition.
     */
    QuadraticPenalty,
    /** Minimise Delta subject to C d = n exactly; the penalty -> infinity limit. */
    HardConstraint,
};

/**
 * Localisation quadratic form added to the Coulomb metric before the fit.
 *
 * The auxiliary-space fit minimises the Coulomb self-repulsion of the fit error
 * alone. That alone does not distribute the fitted density over sites in any
 * particular way, so a distributed-polarizability fit adds a second quadratic
 * form built from the site-blocked fitted densities
 *
 *   E^ab[d] = ( rho~^a || rho~^b ),   rho~^a(r) = sum_{k on a} d_k chi_k(r)
 *
 * whose two published variants are, with weight eta,
 *
 *   InterSite:          Delta - eta * sum_{a, b != a} E^ab   (the reviewed default)
 *   SiteSelfRepulsion:  Delta + eta * sum_a           E^aa
 *
 * Because ( chi_k || chi_l ) is exactly the Coulomb metric J, both forms are a
 * masked rescaling of J itself: sum_{a, b != a} E^ab = d^T K_inter d with
 * K_inter = J masked to different sites, and sum_a E^aa = d^T K_self d with
 * K_self = J masked to the same site. J = K_self + K_inter exactly, and that
 * identity is what the unit tests for the assembler check.
 *
 * The published prose for the inter-site form says it "minimizes the inter-site
 * repulsion" while the displayed equation subtracts the term at the recommended
 * positive eta, which rewards it. Both readings are carried here rather than
 * guessed at: the sign is a data choice through the sign of eta, and whichever
 * is used is recorded in the diagnostics.
 */
enum class PSI_API CDFLocalisation {
    /** No localisation form; the normal matrix is the bare Coulomb metric. */
    None,
    /** Delta - eta * sum_{a, b != a} E^ab. */
    InterSite,
    /** Delta + eta * sum_a E^aa. */
    SiteSelfRepulsion,
};

/** Deterministic numerical policy for the auxiliary-space constrained density fit. */
struct PSI_API CDFOptions {
    /** Auxiliary basis label; must resolve through MintsHelper::get_basisset. */
    std::string auxiliary_basis;
    /** Localisation quadratic form folded into the normal matrix. */
    CDFLocalisation localisation{CDFLocalisation::InterSite};
    /** Weight of the localisation form. The reviewed reference protocol is 5.0e-4. */
    double localisation_weight{5.0e-4};
    /** Constraint treatment; the penalty form generalises the hard one. */
    CDFConstraintPolicy constraints{CDFConstraintPolicy::QuadraticPenalty};
    /** Quadratic penalty weight on ||C d - n||^2; ignored under HardConstraint. */
    double constraint_penalty{1.0};
    // The two gates below are set by measurement, not by taste, and they are
    // deliberately looser than the values first proposed for this stage.
    //
    // The reviewed reference calculation fits a 246-function Cartesian auxiliary
    // basis whose bare Coulomb metric was measured at lam_min = 5.4657e-10,
    // lam_max = 1.0393e+03, condition number 1.902e+12 -- its 48 Cartesian
    // contaminant functions are very nearly linearly dependent on the rest. With
    // the reviewed localisation and charge-penalty terms applied the normal matrix
    // is measured at 7.798e+12, and the reference solved that system by a plain LU
    // factorisation with no truncation whatsoever. A maximum_condition_number of
    // 1.0e+12 -- the value originally proposed for this stage -- therefore fails
    // closed on the very calculation this stage exists to reproduce, and a
    // metric_relative_cutoff of 1.0e-10 puts the threshold at 1.04e-07, roughly
    // thirty spectral directions above lam_min, discarding every one of them.
    //
    // So: admit the reviewed protocol (1.0e+14 > 7.798e+12), and put the cutoff at
    // 1.0e-14, which retains the whole spectrum of that matrix and makes
    // truncation an explicitly requested diagnostic rather than a silent default.
    //
    // The cutoff is RELATIVE and is applied inside the solver as
    // metric_relative_cutoff * lam_max. It is never compared against an absolute
    // eigenvalue magnitude, because the auxiliary exponents alone would then decide
    // the retained rank -- the same lesson already recorded for the WSM refinement
    // column cutoff, and reinforced here: an absolute 1.0e-10 sits at the very
    // bottom of the Cartesian spectrum and nowhere near the spherical one.
    double metric_relative_cutoff{1.0e-14};
    double maximum_condition_number{1.0e14};
    /** Hard cap on any single LAPACK workspace request, in scalar elements. */
    std::size_t maximum_workspace_elements{std::numeric_limits<std::size_t>::max()};
};

/** Complete fit diagnostics; assigned only after every gate has passed. */
struct PSI_API CDFDiagnostics {
    std::size_t auxiliary_count{};
    std::size_t transition_count{};
    std::size_t constraint_count{};
    std::size_t retained_rank{};
    std::size_t discarded_directions{};
    double smallest_eigenvalue{};
    double largest_eigenvalue{};
    double condition_number{};
    double retained_condition_number{};
    double effective_cutoff{};
    double max_constraint_residual{};
    double max_stationarity_residual{};
    double max_coefficient_magnitude{};
    std::string policy;
    std::string algorithm;
};

namespace detail {
/**
 * Pure evaluator: analytic Racah regular real solid-harmonic moments of every
 * auxiliary function about its assigned site.
 *
 * Returns (auxiliary.nbf(), kAuxiliaryMomentComponents) in the component order
 * 00; 10 11c 11s; 20 21c 21s 22c 22s; 30 31c 31s 32c 32s 33c 33s, matching the
 * convention of the projection stage exactly. Column 0 is the function charge
 * integral chi_k dr. The auxiliary basis must be Cartesian.
 */
PSI_API Matrix auxiliary_multipole_moments(const BasisSet& auxiliary,
                                           const std::vector<SitePosition>& sites,
                                           const std::vector<std::size_t>& function_to_site);

/**
 * Pure evaluator: constrained auxiliary-space fit coefficients d[k, (ia)].
 *
 * metric is the (naux, naux) symmetric normal matrix of the fit functional -- the
 * Coulomb metric already carrying any localisation quadratic form the caller wants
 * -- rhs is (naux, transitions), constraints is (rows, naux) and constraint_targets
 * is one target per row shared by every transition. The constraint term is added
 * here, either as the quadratic penalty or as a hard equality. J^-1 is never formed:
 * the solve is a symmetric eigendecomposition with an explicit relative spectral
 * cutoff, applied to the right-hand sides by two matrix products.
 */
PSI_API Matrix solve_constrained_density_fit(const Matrix& metric, const Matrix& rhs,
                                             const Matrix& constraints,
                                             const std::vector<double>& constraint_targets,
                                             const CDFOptions& options,
                                             CDFDiagnostics* diagnostics);

/**
 * Pure evaluator: the two-centre Coulomb metric J_kl = ( chi_k || chi_l ).
 *
 * Serial by construction, unlike the shared fitting-metric utility, because every
 * stage of this pipeline runs under a single-thread contract.
 */
PSI_API Matrix auxiliary_coulomb_metric(const std::shared_ptr<const BasisSet>& auxiliary_basis);

/**
 * Pure evaluator: the fit's symmetric normal matrix, Coulomb metric plus the
 * selected localisation quadratic form.
 *
 * K_inter and K_self are J masked by site coincidence, so the assembled matrix is
 * elementwise J_kl scaled by 1 (same site) and 1 - eta (different sites) under
 * InterSite, and by 1 + eta (same site) and 1 (different sites) under
 * SiteSelfRepulsion. The mask is derived from function_to_site, never from the
 * basis, so a caller may site-group auxiliary functions however it likes.
 */
PSI_API Matrix cdf_localised_normal_matrix(const Matrix& coulomb_metric,
                                           const std::vector<std::size_t>& function_to_site,
                                           std::size_t site_count,
                                           CDFLocalisation localisation, double weight);
}  // namespace detail

/** Which definition distributes the frequency-dependent density susceptibility over sites. */
enum class PSI_API ResponsePartition {
    /** Real-space iterated stockholder weights on the sealed response grid. */
    RealSpaceISA,
    /** Constrained density fitting onto an atom-centred auxiliary basis. */
    ConstrainedDF,
};

/** Up-front storage and work accounting for the auxiliary-space partition. */
struct PSI_API CDFPartitionPlan {
    std::size_t nbf{};
    std::size_t naux{};
    std::size_t nocc{};
    std::size_t nvir{};
    std::size_t site_count{};
    std::size_t transition_count{};
    std::size_t metric_bytes{};
    std::size_t three_index_bytes{};
    std::size_t coefficient_bytes{};
    std::size_t moment_bytes{};
    std::size_t projection_bytes{};
    std::size_t estimated_bytes{};
    std::size_t configured_memory_bytes{};
    std::size_t reserved_memory_bytes{};
    std::size_t work_terms{};
    std::size_t max_work_terms{};
    std::size_t max_auxiliary_count{};
    std::string algorithm;
    std::string memory_semantics;
};

/**
 * Resource gate for the auxiliary-space partition, evaluated before any dense
 * allocation. Three-index integrals are streamed one auxiliary shell at a time, so
 * three_index_bytes accounts for one shell block rather than the whole (P|mu nu)
 * tensor; every other term is a retained payload and is hard-gated against the
 * documented half-memory reservation.
 */
PSI_API CDFPartitionPlan plan_cdf_partition(std::size_t nbf, std::size_t naux, std::size_t nocc,
                                            std::size_t nvir, std::size_t site_count,
                                            std::size_t memory_bytes);

/** Measured fit quality of one auxiliary-space partition, reported after every gate. */
struct PSI_API CDFPartitionDiagnostics {
    CDFPartitionPlan plan;
    CDFDiagnostics fit;
    /** max over transitions of |sum_k q_k d_k|, the charge-condition violation. */
    double max_charge_residual{};
    /** Bound the charge residual was gated against. */
    double charge_residual_bound{};
    /** Auxiliary functions carrying nonzero charge integral chi_k dr. */
    std::size_t charged_auxiliary_count{};
    /** Sign and weight of the localisation form actually applied. */
    double localisation_weight{};
    std::string localisation;
};

/**
 * Production auxiliary-space projection bound to the frozen context's sealed
 * auxiliary basis. Returns exactly the layout project_transition_multipoles
 * returns: site-major B[A * 16 + t, (ia)] in the component order
 * 00; 10 11c 11s; 20 21c 21s 22c 22s; 30 31c 31s 32c 32s 33c 33s, columns in the
 * canonical occupied-major transition order.
 *
 * The charge condition is a finite quadratic penalty, not a hard constraint, so
 * sum_k q_k d_k is small but nonzero and the rank-0 rows sum to that residual
 * rather than to machine zero. The residual is measured, gated, and reported.
 */
PSI_API detail::TransitionMultipoleProjection project_transition_multipoles_cdf(
    const std::shared_ptr<const FrozenResponseContext>& context, const CDFOptions& options,
    CDFPartitionDiagnostics* diagnostics = nullptr);

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
    /** Real-space partition arm: the caller supplies converged partition weights. */
    ISAPolResponseProvider(std::shared_ptr<const FrozenResponseContext> context,
                           ResponseKernel kernel, ISAWeights isa_weights);
    /**
     * Auxiliary-space partition arm. No partition weights exist on this arm, and
     * handing it any is rejected rather than ignored: the two arms must not be able
     * to silently blend.
     */
    ISAPolResponseProvider(std::shared_ptr<const FrozenResponseContext> context,
                           ResponseKernel kernel, CDFOptions cdf_options);

    ResponsePartition partition() const { return partition_; }
    /** Auxiliary-fit provenance from the most recent response, if the arm produced any. */
    const std::optional<CDFPartitionDiagnostics>& cdf_diagnostics() const {
        return cdf_diagnostics_;
    }
    std::size_t expected_response_count(const FrequencyGrid& frequencies) const;
    std::vector<SitePairResponse> compute_isapol_response(const FrequencyGrid& frequencies) const;

   private:
    std::shared_ptr<const FrozenResponseContext> context_;
    ResponseKernel kernel_;
    ResponsePartition partition_{ResponsePartition::RealSpaceISA};
    std::optional<ISAWeights> isa_weights_;
    CDFOptions cdf_options_;
    // compute_isapol_response is const because it publishes nothing; this records the
    // fit provenance of the partition it just built so the caller can gate on it.
    mutable std::optional<CDFPartitionDiagnostics> cdf_diagnostics_;
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
    /** Largest weighted design-column norm; the policy cutoff is relative to it. */
    double maximum_weighted_column_norm{};
    /** The absolute threshold actually handed to the least-squares kernel. */
    double applied_column_cutoff{};
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



/** One ordered rank pair of the isotropic `00 00 0` recoupling table. */
struct PSI_API DispersionRankPair {
    unsigned int coefficient_order{};
    unsigned int first_rank{};
    unsigned int second_rank{};
    double prefactor{};
};

/** Up-front storage and work accounting for the isotropic recoupling sum. */
struct PSI_API DispersionPlan {
    std::size_t frequency_count{};
    std::size_t site_count{};
    std::size_t max_frequency_count{};
    std::size_t max_site_count{};
    std::size_t coefficient_count{};
    std::size_t rank_pair_count{};
    std::size_t isotropic_elements{};
    std::size_t isotropic_bytes{};
    std::size_t coefficient_elements{};
    std::size_t coefficient_bytes{};
    std::size_t contribution_elements{};
    std::size_t contribution_bytes{};
    std::size_t rank_pair_table_bytes{};
    std::size_t metadata_bytes{};
    std::size_t estimated_bytes{};
    std::size_t configured_memory_bytes{};
    std::size_t reserved_memory_bytes{};
    std::size_t work_terms{};
    std::size_t max_work_terms{};
    std::string algorithm;
    std::string memory_semantics;
};

/**
 * Auditable isotropic-recoupling metadata. rank_pair_contributions is
 * rank-pair-major over rank_pair_terms and then row-major over ordered site
 * pairs, so the ordered term t contribution for sites (A,B) lives at
 * t*site_count^2 + A*site_count + B.
 */
struct PSI_API DispersionDiagnostics {
    std::size_t frequency_count{};
    std::size_t weighted_frequency_count{};
    std::size_t site_count{};
    double quadrature_weight_sum{};
    double min_isotropic_polarizability{};
    double max_isotropic_polarizability{};
    std::size_t nonpositive_isotropic_count{};
    double inferred_scale{};
    double max_protocol_grid_deviation{};
    bool protocol_grid_enforced{};
    std::vector<DispersionRankPair> rank_pair_terms;
    std::vector<double> rank_pair_contributions;
    DispersionPlan plan;
};

/**
 * Isotropic `00 00 0` dispersion coefficients as (site, site) matrices.
 * Only the trace of each diagonal rank block of the L3 model enters, so no
 * real Clebsch-Gordan contraction table is required for these outputs. C8
 * through C12 are reviewed L3-model parity, not rank-complete physics: the
 * rank-4 terms of C12 are absent from an L3 model by construction.
 */
struct PSI_API DispersionMatrices {
    SharedMatrix c6;
    SharedMatrix c8;
    SharedMatrix c10;
    SharedMatrix c12;
    DispersionDiagnostics diagnostics;
};

/**
 * Recouple one frequency-major set of refined L3 models into C6/C8/C10/C12.
 * The grid must be the protocol grid produced by make_casimir_grid at some
 * positive scale, one model per grid frequency; the static zero frequency
 * carries no quadrature weight and is excluded from the dispersion sum.
 */
PSI_API DispersionMatrices compute_dispersion(const std::vector<RefinedL3Model>& models,
                                              const FrequencyGrid& frequencies);

namespace detail {
DispersionPlan plan_dispersion(std::size_t frequency_count, std::size_t site_count,
                               std::size_t memory_bytes);
/** Isotropic rank-l polarizability Tr(alpha^{ll})/(2l+1); rank must be 1, 2, or 3. */
double isotropic_rank_polarizability(const L3Matrix& tensor, unsigned int rank);
/** Ordered rank-pair prefactor binom(2*la + 2*lb, 2*la)/(2*pi). */
double dispersion_rank_prefactor(unsigned int first_rank, unsigned int second_rank);
/** The validated ordered rank-pair table; every entry has n = 2*(la + lb + 1). */
const std::vector<DispersionRankPair>& dispersion_rank_pairs();
/**
 * Quadrature-convergence seam: identical recoupling on a caller-supplied
 * ascending half-line grid instead of the eleven-point protocol grid.
 */
DispersionMatrices compute_dispersion_test_only(const std::vector<RefinedL3Model>& models,
                                                const FrequencyGrid& frequencies);
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

/** Read the ATOMIC_POLARIZABILITY_ISA_* keywords into a validated ISA policy. */
PSI_API ISAOptions isa_options_from(Options& options);

/** The exact reviewed protocol response kernel, 25 percent CHF plus 75 percent ALDA. */
PSI_API ResponseKernel reviewed_response_kernel();

/** Read ATOMIC_POLARIZABILITY_PARTITION into a validated partition selection. */
PSI_API ResponsePartition response_partition_from(Options& options);

/** Read the ATOMIC_POLARIZABILITY_CDF_* keywords into a validated auxiliary-fit policy. */
PSI_API CDFOptions cdf_options_from(Options& options);

/** Basis-set map key under which the auxiliary partition's basis must be attached. */
PSI_API const char* auxiliary_partition_basis_key();

/**
 * Complete, validated pipeline output. Nothing here is published until every stage gate
 * has passed, so a caller either receives all seven arrays or an exception.
 *
 * static_polarizabilities is (sites, 6) and dynamic_polarizabilities is
 * (frequencies * sites, 6), both packed xx, xy, xz, yy, yz, zz in the global Cartesian
 * frame and frequency-major over site-major blocks. frequencies is (frequencies, 1).
 */
struct PSI_API AtomicPolarizabilityPublication {
    FrequencyGrid grid;
    SharedMatrix static_polarizabilities;
    SharedMatrix dynamic_polarizabilities;
    SharedMatrix frequencies;
    DispersionMatrices dispersion;
    /** Which definition partitioned the response. */
    ResponsePartition partition{ResponsePartition::RealSpaceISA};
    /** Stage provenance, in chain order, for auditing a parity mismatch. */
    ISADiagnostics isa;
    /** Present only under the auxiliary partition. */
    std::optional<CDFPartitionDiagnostics> cdf;
    BondGraphDerivation bond_graph;
    FitPointPlan fit_points;
    PDefDerivation pdef;
    std::vector<LocalizationResiduals> localization_residuals;
    std::vector<RefinementDiagnostics> refinement;
};

/**
 * Native atomic-polarizability pipeline entry point.
 *
 * FrozenResponseContext::create needs the GRAC-corrected reference together with the
 * neutral precursor and cation wavefunctions that fix the applied shift, so this class
 * takes all three. Per the end-to-end wiring specification the Python driver runs the
 * three SCFs; a property class does not drive SCF. The single-wavefunction constructor is
 * retained only so a bare OEProp call keeps failing closed with a clear message.
 */
class PSI_API AtomicPolarizabilityCalculator {
   public:
    AtomicPolarizabilityCalculator(std::shared_ptr<Wavefunction> grac_wfn,
                                   std::shared_ptr<Wavefunction> neutral_precursor_wfn,
                                   std::shared_ptr<Wavefunction> cation_wfn);
    /** Bare-OEProp seam: retains no SCF triple, so compute() fails closed. */
    explicit AtomicPolarizabilityCalculator(std::shared_ptr<Wavefunction> wfn);

    /** Run every stage, then publish the seven arrays only if all of them passed. */
    void compute();
    /** Run every stage and return the complete result without publishing anything. */
    AtomicPolarizabilityPublication run() const;

   private:
    void validate_wavefunction_prerequisites() const;

    std::shared_ptr<Wavefunction> wfn_;
    std::shared_ptr<Wavefunction> neutral_precursor_wfn_;
    std::shared_ptr<Wavefunction> cation_wfn_;
};

}  // namespace psi

#endif  // PSI4_SRC_PSI4_LIBMINTS_ATOMIC_POLARIZABILITY_H
