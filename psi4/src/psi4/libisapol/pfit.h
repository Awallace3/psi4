/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_PFIT_H
#define PSI4_LIBISAPOL_PFIT_H
#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>
namespace psi { namespace isapol {
/// Value-owned row-major storage. Validated at solve entry; no borrowed arrays.
struct IsaPfitMatrix {
    size_t rows = 0, cols = 0;
    std::vector<double> values;
};
/// NativeDirectActualPointResponse is this code's own direct-OV point-charge
/// response. It is deliberately NOT interchangeable with the Supplied* origins:
/// the historical target is a constrained-NN/fitted-propagator quantity.
/// NativeFittedPointResponse contracts native AUX coefficient responses with
/// exact AUX Coulomb point potentials. It requires a declared AUX identity and
/// fitted_density_coefficients representation; it is not a supplied target or
/// a certification of CamCASP parity.
/// Appended, never inserted: the underlying values of the existing enumerators
/// must not change for anything already compiled against this header.
enum class IsaPfitTargetOrigin { Unspecified, SuppliedActualPointResponse,
    SuppliedFittedPropagatorPointResponse, SyntheticAnalyticTest,
    NativeDirectActualPointResponse, NativeFittedPointResponse };
enum class IsaPfitTargetConvention { Unspecified, NegativeInducedPotentialPerUnitSourceChargeAtomicUnits };
struct IsaPfitTargetProvenance {
    IsaPfitTargetOrigin origin = IsaPfitTargetOrigin::Unspecified;
    IsaPfitTargetConvention convention = IsaPfitTargetConvention::Unspecified;
    std::string source_id, response_representation, auxiliary_basis_id, generation_record;
};
struct IsaPfitBatch {
    std::string label;
    std::vector<std::array<double, 3>> points_bohr;
    IsaPfitMatrix fields;
    std::vector<double> targets; // i*(i+1)/2+j, every j<=i once
};
struct IsaPfitModel {
    std::vector<std::string> channel_labels, parameter_labels, parameter_units;
    std::vector<IsaPfitMatrix> parameter_tensors; // full symmetric channel tensors
    std::vector<bool> fixed;
    std::vector<double> fixed_values;
    std::string provenance;
};
struct IsaPfitMatrixPenalty { IsaPfitMatrix matrix; std::vector<double> anchor; };
struct IsaPfitLinearPenalty {
    std::vector<double> coefficients;
    double target = 0, strength = 0;
};
struct IsaPfitProblem {
    double frequency_au = 0;
    IsaPfitTargetProvenance target_provenance;
    IsaPfitModel model;
    std::vector<IsaPfitBatch> batches;
    IsaPfitMatrixPenalty penalty;
    std::vector<IsaPfitLinearPenalty> linear_penalties;
};
/// Metadata for supplied design rows: no fabricated fields or channel tensors.
struct IsaPfitRowModel {
    std::vector<std::string> channel_labels, parameter_labels, parameter_units;
    std::vector<bool> fixed;
    std::vector<double> fixed_values;
    std::string provenance;
};
struct IsaPfitCloudRows {
    std::string label;
    size_t points = 0, full_row_count = 0, maximum_block_rows = 256;
};
struct IsaPfitRowProblem {
    double frequency_au = 0;
    IsaPfitTargetProvenance target_provenance;
    IsaPfitRowModel model;
    IsaPfitCloudRows cloud;
    IsaPfitMatrixPenalty penalty;
    std::vector<IsaPfitLinearPenalty> linear_penalties;
};
struct IsaPfitRowBlockInfo { size_t packed_start = 0, rows = 0; };
/// Native callbacks are trusted bounded-buffer producers. Destinations are
/// borrowed only during next(), never retained or overrun. rows=0 denotes EOF.
/// finish_pass must digest actual row/target content, independently of block
/// partition. Successful fits require identical assembly/residual digests.
/// The Python adapter computes digests itself and checks arrays before copying.
struct IsaPfitRowSource {
    std::function<void(size_t)> begin_pass;
    std::function<IsaPfitRowBlockInfo(size_t, size_t, double*, double*)> next;
    std::function<std::array<unsigned char, 32>()> finish_pass;
};
namespace detail { struct IsaPfitSolverImpl; }
enum class IsaPfitSolver { NormalEquationsDSYSV, StreamingQR };
enum class IsaPfitStatus { Solved, AllFixed, RankDeficient, IllConditioned, NumericalFailure };
struct IsaPfitOptions {
    IsaPfitSolver solver = IsaPfitSolver::StreamingQR;
    size_t qr_chunk_rows = 256;
    /// Limit on conservative kernel numerical-buffer budget only, NOT process peak.
    /// Excludes caller-owned problem, strings/container overhead and subsequent getter copies.
    size_t maximum_work_bytes = 256 * 1024 * 1024;
    double rank_relative_tolerance = 1.e-12;
    double minimum_solver_rcond = 1.e-14;
    bool retain_pair_predictions = false;
};
struct IsaPfitBatchDiagnostics {
    size_t points = 0, rows = 0;
    double sse = 0, rms = 0, max_residual = 0;
};
struct IsaPfitDiagnostics {
    size_t data_rows = 0, augmented_rows = 0, numerical_rank = 0, work_budget_bytes = 0;
    std::vector<size_t> free_indices;
    std::vector<IsaPfitBatchDiagnostics> batches;
    double data_sse = 0, data_rms = 0, data_max_residual = 0;
    double matrix_objective = 0, lc_objective = 0, total_objective = 0;
    double stationarity_inf = 0, backward_residual = 0;
    double normal_h_rcond = 0, normal_h_norm1 = 0, qr_r_rcond = 0, rank_smallest = 0, rank_largest = 0;
    double penalty_min_eigenvalue = 0, penalty_asymmetry = 0, penalty_correction_max = 0;
    double qr_discarded_rhs_sse = 0;
    int lapack_info = 0;
    std::string rank_method, psd_policy = "strict symmetry; reject negative computed eigenvalues; effective P=B^T B";
    bool objective_available = false, native_verified = false, condition_estimate_available = false;
};
class IsaPfitResult {
 public:
    IsaPfitStatus status() const { return status_; }
    std::vector<double> parameters() const;
    IsaPfitMatrix normal_matrix() const { return normal_; }
    std::vector<double> normal_rhs() const { return rhs_; }
    IsaPfitMatrix effective_penalty_matrix() const;
    IsaPfitMatrix matrix_penalty_matrix() const { return penalty_; }
    IsaPfitMatrix lc_penalty_matrix() const { return lc_; }
    std::vector<double> matrix_penalty_rhs() const { return matrix_rhs_; }
    std::vector<double> lc_penalty_rhs() const { return lc_rhs_; }
    std::vector<double> effective_penalty_rhs() const;
    IsaPfitDiagnostics diagnostics() const { return diagnostics_; }
    IsaPfitOptions settings() const { return settings_; }
    double frequency_au() const { return frequency_; }
    IsaPfitTargetProvenance target_provenance() const { return provenance_; }
    std::string model_provenance() const { return model_provenance_; }
    std::vector<std::string> parameter_labels() const { return parameter_labels_; }
    std::vector<std::string> parameter_units() const { return parameter_units_; }
    std::vector<std::string> channel_labels() const { return channel_labels_; }
    std::vector<std::string> batch_labels() const { return batch_labels_; }
    std::vector<std::vector<double>> predictions() const { return predictions_; }
 private:
    friend struct detail::IsaPfitSolverImpl;
    friend IsaPfitResult isa_pfit_solve(const IsaPfitProblem&, const IsaPfitOptions&);
    IsaPfitStatus status_ = IsaPfitStatus::NumericalFailure;
    IsaPfitMatrix normal_, penalty_, lc_;
    std::vector<double> rhs_, parameters_, matrix_rhs_, lc_rhs_;
    IsaPfitDiagnostics diagnostics_;
    IsaPfitOptions settings_;
    double frequency_ = 0;
    IsaPfitTargetProvenance provenance_;
    std::string model_provenance_;
    std::vector<std::string> parameter_labels_, parameter_units_, channel_labels_, batch_labels_;
    std::vector<std::vector<double>> predictions_;
};
/// Supplied declaration only: v=-d(phi_induced)/dq, Eh/e^2, no energy 1/2 or bare electrostatics.
/// No native target generation, localization, pruning, scaling or regularization.
IsaPfitResult isa_pfit_solve(const IsaPfitProblem&, const IsaPfitOptions& = IsaPfitOptions());
/// One complete cloud, all N(N+1)/2 supplied rows, one global penalty and solve.
/// At most two traversals; failed rank/conditioning fits retain old early-return
/// semantics. Kernel budget includes bounded destination blocks, not producer
/// storage or snapshotted problem metadata. Problem/options/source functions are
/// copied before callbacks; no native/provenance certification is inferred.
IsaPfitResult isa_pfit_solve_rows(const IsaPfitRowProblem&, const IsaPfitRowSource&,
                                const IsaPfitOptions& = IsaPfitOptions());
}} // namespace psi::isapol
#endif
