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

#include "psi4/pybind11.h"

#include "psi4/libisapol/native_response.h"
#include "psi4/libisapol/point_response.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/vector.h"
#include "psi4/libisapol/casimir_grid.h"
#include "psi4/libisapol/pfit.h"
#include "psi4/libisapol/fit_points.h"
#include "psi4/libisapol/isa_grid.h"
#include "psi4/libisapol/isa_fit.h"
#include "psi4/libisapol/explicit_basis.h"
#include "psi4/libisapol/partitioned_response.h"
#include "psi4/libisapol/isotropic_dispersion.h"
#include "psi4/libisapol/anisotropic_dispersion.h"
#include "psi4/libisapol/multipole_transform.h"
#include "psi4/libisapol/t_functions.h"
#include "psi4/libisapol/lw_localization.h"
#include "psi4/libisapol/aux_coulomb.h"
#include "psi4/libisapol/ov_fit.h"
#include "psi4/libisapol/isa_sweep.h"
#include "psi4/libisapol/isa_shape.h"
#include "psi4/libisapol/isa_controller.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libisapol/recoupling_tables.h"
#include "psi4/libisapol/realcg_tables.h"
#include "psi4/libisapol/recoupled_dispersion.h"
#include <pybind11/complex.h>
#include "psi4/libisapol/tables.h"
#include "psi4/libmints/molecule.h"

#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#ifdef USING_LAPACK_MKL
#include <mkl.h>
#endif

using namespace psi;
using namespace psi::isapol;

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

/// Zero-copy-free view of one of IsaGrid's coordinate arrays as a numpy array.
py::array_t<double> grid_column(const IsaGrid& grid, const double* data) {
    return py::array_t<double>(static_cast<py::ssize_t>(grid.npoints()), data);
}

namespace lw_binding_private {

template <std::size_t N>
py::list matrix_copies(const std::vector<std::array<std::array<double, N>, N>>& blocks) {
    py::list result;
    for (const auto& block : blocks) {
        auto matrix = std::make_shared<Matrix>(N, N);
        for (std::size_t row = 0; row < N; ++row)
            for (std::size_t column = 0; column < N; ++column)
                (*matrix)(row, column) = block[row][column];
        result.append(matrix);
    }
    return result;
}

IsaLocalizedResponse localize(const Matrix& positions, const py::sequence& blocks,
                             double frequency, const py::sequence& bonds, double tolerance,
                             double input_sum_rule_tolerance, int rank_limit) {
    // Declared, not inferred: the caller states the rank the localization runs at.
    if (rank_limit < 1 || rank_limit > 3)
        throw std::runtime_error(
            "localize_lw: declared rank_limit must be 1, 2 or 3; rank 4 needs a rank-4 working matrix");
    if (!std::isfinite(frequency))
        throw std::runtime_error("localize_lw: response frequency must be finite");
    if (frequency < 0.0)
        throw std::runtime_error("localize_lw: response frequency must be nonnegative");
    if (!std::isfinite(tolerance) || tolerance <= 0.0)
        throw std::runtime_error("localize_lw: residual tolerance must be finite and positive");
    // Negative inherits `tolerance`; positive finite gates separately; infinity reports only.
    if (std::isnan(input_sum_rule_tolerance) || input_sum_rule_tolerance == 0.0)
        throw std::runtime_error(
            "localize_lw: input sum-rule tolerance must be positive or infinite, or negative to inherit");
    if (positions.nirrep() != 1 || positions.ncol() != 3 || positions.nrow() <= 0)
        throw std::runtime_error("localize_lw: positions must be an N by 3 single-block matrix");
    const auto count = static_cast<std::size_t>(positions.nrow());
    // Do not use automatic vector conversion on unbounded Python collections.
    // Check all lengths and the native budget before allocating owned input arrays.
    const std::size_t bond_count = py::len(bonds);
    isa_lw_validate_workspace(count, bond_count);
    if (py::len(blocks) != count * count)
        throw std::runtime_error("localize_lw: expected one 16 by 16 block for every ordered site pair");
    IsaBondGraph graph{count, {}};
    graph.bonds.reserve(bond_count);
    for (std::size_t edge = 0; edge < bond_count; ++edge) {
        const auto bond = py::cast<py::sequence>(bonds[edge]);
        if (py::len(bond) != 2)
            throw std::runtime_error("localize_lw: each bond must contain two site indices");
        graph.bonds.push_back({py::cast<std::size_t>(bond[0]), py::cast<std::size_t>(bond[1])});
    }
    // Reuse the validated graph helper; reject invalid edges before response allocation.
    isa_lw_graph_operator(graph);
    std::vector<std::shared_ptr<Matrix>> checked;
    checked.reserve(count * count);
    // Indexed, bounded access also prevents dishonest custom sequence iterators
    // from growing these vectors beyond the validated lengths.
    for (std::size_t block = 0; block < count * count; ++block)
        checked.push_back(py::cast<std::shared_ptr<Matrix>>(blocks[block]));
    // Finish Python callbacks before inspecting/copying matrices: a custom
    // sequence may mutate a Matrix already returned by an earlier __getitem__.
    if (positions.nirrep() != 1 || positions.ncol() != 3 ||
        positions.nrow() != static_cast<int>(count))
        throw std::runtime_error("localize_lw: positions changed dimensions during conversion");
    for (std::size_t site = 0; site < count; ++site)
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (!std::isfinite(positions(site, axis)))
                throw std::runtime_error("localize_lw: site positions must be finite");
    for (const auto& matrix : checked) {
        if (!matrix || matrix->nirrep() != 1 || matrix->nrow() != 16 || matrix->ncol() != 16)
            throw std::runtime_error("localize_lw: expected 16 by 16 single-block matrices");
        for (std::size_t row = 0; row < 16; ++row)
            for (std::size_t column = 0; column < 16; ++column)
                if (!std::isfinite((*matrix)(row, column)))
                    throw std::runtime_error("localize_lw: response values must be finite");
    }
    IsaSitePairResponse response;
    response.frequency = frequency;
    response.positions.resize(count);
    for (std::size_t site = 0; site < count; ++site)
        for (std::size_t axis = 0; axis < 3; ++axis)
            response.positions[site][axis] = positions(site, axis);
    response.blocks.resize(count * count);
    for (std::size_t block = 0; block < checked.size(); ++block)
        for (std::size_t row = 0; row < 16; ++row)
            for (std::size_t column = 0; column < 16; ++column)
                response.blocks[block][row][column] = (*checked[block])(row, column);
    return isa_localize_lw(response, graph, tolerance, input_sum_rule_tolerance, rank_limit);
}
}  // namespace lw_binding_private
}  // namespace

void export_isapol(py::module& m) {
    // Internal orchestration boundary: isolate tiny, conditioning-sensitive AO
    // adaptation from MKL's thread-dependent SVD arithmetic. A thread-local
    // override cannot affect other Python/OpenMP callers and is restored on every
    // exit (including nested calls and Python exceptions). Other BLAS backends
    // are deliberately untouched; do not emulate this with process-wide setters.
    m.def("_isa_serial_blas_call", [](const py::function& callback) -> py::object {
#ifdef USING_LAPACK_MKL
        struct Guard {
            int previous = mkl_set_num_threads_local(1);
            ~Guard() { mkl_set_num_threads_local(previous); }
        } guard;
#endif
        return callback();
    });
    m.def("_isa_blas_max_threads", []() {
#ifdef USING_LAPACK_MKL
        return mkl_get_max_threads();
#else
        return 0;  // no supported thread-local override in this build
#endif
    });
    py::class_<NativeResponseProvider, std::shared_ptr<NativeResponseProvider>>(m, "NativeResponseProvider")
        .def(py::init<std::shared_ptr<Wavefunction>, bool, const std::string&, double, double,
                      SharedMatrix, double, std::size_t, std::size_t, const std::string&>(),
             "wavefunction"_a, "caller_converged"_a, "kernel"_a, "exact_exchange"_a,
             "local_scale"_a, "grid"_a, "density_cutoff"_a, "max_bytes"_a, "max_nov"_a,
             "algorithm"_a = "ordered_pairwise")
        .def("h1", &NativeResponseProvider::h1)
        .def("h2", &NativeResponseProvider::h2)
        .def("coulomb", &NativeResponseProvider::coulomb)
        .def("exchange_direct", &NativeResponseProvider::exchange_direct)
        .def("exchange_transpose", &NativeResponseProvider::exchange_transpose)
        .def("local_primitive", &NativeResponseProvider::local_primitive)
        .def("orbitals", &NativeResponseProvider::orbitals)
        .def("energies", &NativeResponseProvider::energies)
        .def("density_alpha", &NativeResponseProvider::density_alpha)
        .def_property_readonly("nocc", &NativeResponseProvider::nocc)
        .def_property_readonly("nvir", &NativeResponseProvider::nvir)
        .def_property_readonly("kernel", &NativeResponseProvider::kernel)
        .def_property_readonly("algorithm", &NativeResponseProvider::algorithm)
        .def_property_readonly("exact_exchange", &NativeResponseProvider::exact_exchange)
        .def_property_readonly("local_scale", &NativeResponseProvider::local_scale)
        .def_property_readonly("density_cutoff", &NativeResponseProvider::density_cutoff)
        .def_property_readonly("planned_bytes", &NativeResponseProvider::planned_bytes)
        .def_property_readonly("ov_order", [](const NativeResponseProvider&) { return "occupied_fast: t=a*nocc+i"; })
        .def_property_readonly("caller_converged", [](const NativeResponseProvider&) { return true; })
        .def_property_readonly("convergence_evidence", [](const NativeResponseProvider&) {
            return "caller declaration only; restricted metadata, density and orthonormality checked";
        });

    py::class_<IsaAldaGridScreen, std::shared_ptr<IsaAldaGridScreen>>(m, "IsaAldaGridScreen")
        .def(py::init<std::shared_ptr<Wavefunction>, bool, const std::string&, SharedMatrix, double,
                      std::size_t>(),
             "wavefunction"_a, "caller_converged"_a, "kernel"_a, "grid"_a, "density_cutoff"_a,
             "max_bytes"_a)
        .def("values", &IsaAldaGridScreen::values)
        .def("retained_rows", &IsaAldaGridScreen::retained_rows, "threshold"_a)
        .def("retained", &IsaAldaGridScreen::retained, "threshold"_a)
        .def("omitted_bound", &IsaAldaGridScreen::omitted_bound, "threshold"_a)
        .def("omitted_count", &IsaAldaGridScreen::omitted_count, "threshold"_a)
        .def("threshold_for_rows", &IsaAldaGridScreen::threshold_for_rows, "max_rows"_a)
        .def_property_readonly("total", &IsaAldaGridScreen::total)
        .def_property_readonly("maximum", &IsaAldaGridScreen::maximum)
        .def_property_readonly("rows", &IsaAldaGridScreen::rows)
        .def_property_readonly("exact_zero_rows", &IsaAldaGridScreen::exact_zero_rows)
        .def_property_readonly("kernel", &IsaAldaGridScreen::kernel)
        .def_property_readonly("density_cutoff", &IsaAldaGridScreen::density_cutoff)
        .def_property_readonly("planned_bytes", &IsaAldaGridScreen::planned_bytes)
        .def_property_readonly("bound_norms", [](const IsaAldaGridScreen&) {
            return "omitted_bound bounds both maxabs and Frobenius deviation of the local primitive";
        })
        .def_property_readonly("quadrature_policy", [](const IsaAldaGridScreen&) {
            return "row subset only: original coordinates, weights and order; no renormalization";
        });

    py::class_<IsaPointChargeOperators, std::shared_ptr<IsaPointChargeOperators>>(m, "IsaPointChargeOperators")
        .def(py::init<std::shared_ptr<Wavefunction>, bool, SharedMatrix, std::size_t, std::size_t>(),
             "wavefunction"_a, "caller_converged"_a, "points_bohr"_a, "max_bytes"_a, "max_points"_a)
        .def("operators", &IsaPointChargeOperators::operators)
        .def("points", &IsaPointChargeOperators::points)
        .def_property_readonly("nocc", &IsaPointChargeOperators::nocc)
        .def_property_readonly("nvir", &IsaPointChargeOperators::nvir)
        .def_property_readonly("ntransition", &IsaPointChargeOperators::ntransition)
        .def_property_readonly("npoint", &IsaPointChargeOperators::npoint)
        .def_property_readonly("ov_order", &IsaPointChargeOperators::ov_order)
        .def_property_readonly("representation", &IsaPointChargeOperators::representation)
        .def_property_readonly("convention", &IsaPointChargeOperators::convention)
        .def_property_readonly("minimum_nuclear_distance_bohr", &IsaPointChargeOperators::minimum_nuclear_distance_bohr)
        .def_property_readonly("minimum_point_separation_bohr", &IsaPointChargeOperators::minimum_point_separation_bohr)
        .def_property_readonly("maximum_absolute_element", &IsaPointChargeOperators::maximum_absolute_element)
        .def_property_readonly("planned_bytes", &IsaPointChargeOperators::planned_bytes);

    // PFIT value containers: every property read is an independent snapshot,
    // including nested objects. Assign modified snapshots back explicitly.
#define PFIT_CLASS(T) py::class_<T>(m, #T).def(py::init<>())
#define PFIT_FIELD(T, F) .def_property(#F, [](const T& v) { return v.F; }, [](T& v, const decltype(T::F)& x) { v.F = x; })
#define PFIT_GET(F) .def_property_readonly(#F, &IsaPfitResult::F)
    py::enum_<IsaPfitTargetOrigin>(m, "IsaPfitTargetOrigin")
        .value("Unspecified", IsaPfitTargetOrigin::Unspecified)
        .value("SuppliedActualPointResponse", IsaPfitTargetOrigin::SuppliedActualPointResponse)
        .value("SuppliedFittedPropagatorPointResponse", IsaPfitTargetOrigin::SuppliedFittedPropagatorPointResponse)
        .value("NativeDirectActualPointResponse", IsaPfitTargetOrigin::NativeDirectActualPointResponse)
        .value("SyntheticAnalyticTest", IsaPfitTargetOrigin::SyntheticAnalyticTest);
    py::enum_<IsaPfitTargetConvention>(m, "IsaPfitTargetConvention")
        .value("Unspecified", IsaPfitTargetConvention::Unspecified)
        .value("NegativeInducedPotentialPerUnitSourceChargeAtomicUnits", IsaPfitTargetConvention::NegativeInducedPotentialPerUnitSourceChargeAtomicUnits);
    py::enum_<IsaPfitSolver>(m, "IsaPfitSolver")
        .value("NormalEquationsDSYSV", IsaPfitSolver::NormalEquationsDSYSV)
        .value("StreamingQR", IsaPfitSolver::StreamingQR);
    py::enum_<IsaPfitStatus>(m, "IsaPfitStatus")
        .value("Solved", IsaPfitStatus::Solved).value("AllFixed", IsaPfitStatus::AllFixed)
        .value("RankDeficient", IsaPfitStatus::RankDeficient).value("IllConditioned", IsaPfitStatus::IllConditioned)
        .value("NumericalFailure", IsaPfitStatus::NumericalFailure);
    PFIT_CLASS(IsaPfitMatrix)
        PFIT_FIELD(IsaPfitMatrix, rows) PFIT_FIELD(IsaPfitMatrix, cols) PFIT_FIELD(IsaPfitMatrix, values);
    PFIT_CLASS(IsaPfitTargetProvenance)
        PFIT_FIELD(IsaPfitTargetProvenance, origin) PFIT_FIELD(IsaPfitTargetProvenance, convention)
        PFIT_FIELD(IsaPfitTargetProvenance, source_id) PFIT_FIELD(IsaPfitTargetProvenance, response_representation)
        PFIT_FIELD(IsaPfitTargetProvenance, auxiliary_basis_id) PFIT_FIELD(IsaPfitTargetProvenance, generation_record);
    PFIT_CLASS(IsaPfitBatch)
        PFIT_FIELD(IsaPfitBatch, label) PFIT_FIELD(IsaPfitBatch, points_bohr)
        PFIT_FIELD(IsaPfitBatch, fields) PFIT_FIELD(IsaPfitBatch, targets);
    PFIT_CLASS(IsaPfitModel)
        PFIT_FIELD(IsaPfitModel, channel_labels) PFIT_FIELD(IsaPfitModel, parameter_labels)
        PFIT_FIELD(IsaPfitModel, parameter_units) PFIT_FIELD(IsaPfitModel, parameter_tensors)
        PFIT_FIELD(IsaPfitModel, fixed) PFIT_FIELD(IsaPfitModel, fixed_values) PFIT_FIELD(IsaPfitModel, provenance);
    PFIT_CLASS(IsaPfitMatrixPenalty) PFIT_FIELD(IsaPfitMatrixPenalty, matrix) PFIT_FIELD(IsaPfitMatrixPenalty, anchor);
    PFIT_CLASS(IsaPfitLinearPenalty) PFIT_FIELD(IsaPfitLinearPenalty, coefficients)
        PFIT_FIELD(IsaPfitLinearPenalty, target) PFIT_FIELD(IsaPfitLinearPenalty, strength);
    PFIT_CLASS(IsaPfitProblem) PFIT_FIELD(IsaPfitProblem, frequency_au) PFIT_FIELD(IsaPfitProblem, target_provenance)
        PFIT_FIELD(IsaPfitProblem, model) PFIT_FIELD(IsaPfitProblem, batches)
        PFIT_FIELD(IsaPfitProblem, penalty) PFIT_FIELD(IsaPfitProblem, linear_penalties);
    PFIT_CLASS(IsaPfitOptions) PFIT_FIELD(IsaPfitOptions, solver) PFIT_FIELD(IsaPfitOptions, qr_chunk_rows)
        PFIT_FIELD(IsaPfitOptions, maximum_work_bytes) PFIT_FIELD(IsaPfitOptions, rank_relative_tolerance)
        PFIT_FIELD(IsaPfitOptions, minimum_solver_rcond) PFIT_FIELD(IsaPfitOptions, retain_pair_predictions);
    PFIT_CLASS(IsaPfitBatchDiagnostics) PFIT_FIELD(IsaPfitBatchDiagnostics, points) PFIT_FIELD(IsaPfitBatchDiagnostics, rows)
        PFIT_FIELD(IsaPfitBatchDiagnostics, sse) PFIT_FIELD(IsaPfitBatchDiagnostics, rms) PFIT_FIELD(IsaPfitBatchDiagnostics, max_residual);
    PFIT_CLASS(IsaPfitDiagnostics)
        PFIT_FIELD(IsaPfitDiagnostics, data_rows) PFIT_FIELD(IsaPfitDiagnostics, augmented_rows)
        PFIT_FIELD(IsaPfitDiagnostics, numerical_rank) PFIT_FIELD(IsaPfitDiagnostics, work_budget_bytes)
        PFIT_FIELD(IsaPfitDiagnostics, free_indices) PFIT_FIELD(IsaPfitDiagnostics, batches)
        PFIT_FIELD(IsaPfitDiagnostics, data_sse) PFIT_FIELD(IsaPfitDiagnostics, data_rms) PFIT_FIELD(IsaPfitDiagnostics, data_max_residual)
        PFIT_FIELD(IsaPfitDiagnostics, matrix_objective) PFIT_FIELD(IsaPfitDiagnostics, lc_objective) PFIT_FIELD(IsaPfitDiagnostics, total_objective)
        PFIT_FIELD(IsaPfitDiagnostics, stationarity_inf) PFIT_FIELD(IsaPfitDiagnostics, backward_residual)
        PFIT_FIELD(IsaPfitDiagnostics, normal_h_rcond) PFIT_FIELD(IsaPfitDiagnostics, normal_h_norm1)
        PFIT_FIELD(IsaPfitDiagnostics, condition_estimate_available) PFIT_FIELD(IsaPfitDiagnostics, qr_r_rcond)
        PFIT_FIELD(IsaPfitDiagnostics, rank_smallest) PFIT_FIELD(IsaPfitDiagnostics, rank_largest)
        PFIT_FIELD(IsaPfitDiagnostics, penalty_min_eigenvalue) PFIT_FIELD(IsaPfitDiagnostics, penalty_asymmetry)
        PFIT_FIELD(IsaPfitDiagnostics, penalty_correction_max) PFIT_FIELD(IsaPfitDiagnostics, qr_discarded_rhs_sse)
        PFIT_FIELD(IsaPfitDiagnostics, lapack_info) PFIT_FIELD(IsaPfitDiagnostics, rank_method)
        PFIT_FIELD(IsaPfitDiagnostics, psd_policy) PFIT_FIELD(IsaPfitDiagnostics, objective_available)
        PFIT_FIELD(IsaPfitDiagnostics, native_verified);
    py::class_<IsaPfitResult>(m, "IsaPfitResult")
        PFIT_GET(status) PFIT_GET(parameters) PFIT_GET(normal_matrix) PFIT_GET(normal_rhs)
        PFIT_GET(effective_penalty_matrix) PFIT_GET(matrix_penalty_matrix) PFIT_GET(lc_penalty_matrix) PFIT_GET(matrix_penalty_rhs)
        PFIT_GET(lc_penalty_rhs) PFIT_GET(effective_penalty_rhs) PFIT_GET(diagnostics) PFIT_GET(settings)
        PFIT_GET(frequency_au) PFIT_GET(target_provenance) PFIT_GET(model_provenance) PFIT_GET(predictions)
        PFIT_GET(parameter_labels) PFIT_GET(parameter_units) PFIT_GET(channel_labels) PFIT_GET(batch_labels);
    m.def("isa_pfit_solve", &isa_pfit_solve, "problem"_a, "options"_a = IsaPfitOptions(),
          "Supplied single-frequency PFIT; v=-d(phi_induced)/dq in Eh/e^2. Caller provenance is not native verification.");
#undef PFIT_GET
#undef PFIT_FIELD
#undef PFIT_CLASS
    py::class_<IsaBondTransfer>(m, "IsaBondTransfer")
        .def_readonly("first", &IsaBondTransfer::first)
        .def_readonly("second", &IsaBondTransfer::second)
        .def_readonly("first_component", &IsaBondTransfer::first_component)
        .def_readonly("second_component", &IsaBondTransfer::second_component)
        .def_readonly("fixed_site", &IsaBondTransfer::fixed_site)
        .def_readonly("amount", &IsaBondTransfer::amount);
    py::class_<IsaLocalizationResiduals>(m, "IsaLocalizationResiduals")
        .def_readonly("off_site", &IsaLocalizationResiduals::off_site)
        .def_readonly("charge_sum", &IsaLocalizationResiduals::charge_sum)
        .def_readonly("reciprocity", &IsaLocalizationResiduals::reciprocity)
        .def_readonly("molecular_sum", &IsaLocalizationResiduals::molecular_sum)
        .def_readonly("local_charge", &IsaLocalizationResiduals::local_charge)
        .def_readonly("input_sum_rule", &IsaLocalizationResiduals::input_sum_rule)
        .def_readonly("charge_sum_transport", &IsaLocalizationResiduals::charge_sum_transport);
    py::class_<IsaLocalizedResponse>(m, "IsaLocalizedResponse",
        "Owned supplied-response LW localization, not native-upstream verification or PFIT refinement")
        .def_property_readonly("frequency", [](const IsaLocalizedResponse& r) { return r.frequency; })
        .def_property_readonly("positions", [](const IsaLocalizedResponse& r) {
            auto matrix = std::make_shared<Matrix>(r.positions.size(), 3);
            for (std::size_t site = 0; site < r.positions.size(); ++site)
                for (std::size_t axis = 0; axis < 3; ++axis) (*matrix)(site, axis) = r.positions[site][axis];
            return matrix;
        })
        .def_property_readonly("local", [](const IsaLocalizedResponse& r) {
            return lw_binding_private::matrix_copies(r.local);
        })
        .def_property_readonly("refined_pairs", [](const IsaLocalizedResponse& r) {
            return lw_binding_private::matrix_copies(r.refined_pairs);
        }, "Full 16x16 LW workspace; NOT externally PFIT-refined pairs")
        .def_property_readonly("transfers", [](const IsaLocalizedResponse& r) { return r.transfers; })
        .def_property_readonly("residuals", [](const IsaLocalizedResponse& r) { return r.residuals; })
        .def_property_readonly("omitted_component_pairs", [](const IsaLocalizedResponse& r) {
            return r.omitted_component_pairs;
        })
        .def_property_readonly("omitted_transfer_count", [](const IsaLocalizedResponse& r) {
            return r.omitted_transfer_count;
        })
        .def_property_readonly("localization_rank_limit", [](const IsaLocalizedResponse& r) {
            return r.localization_rank_limit;
        }, "Declared localization rank; every higher-rank component is identically zero")
        .def_property_readonly("truncated_input_maxabs", [](const IsaLocalizedResponse& r) {
            return r.truncated_input_maxabs;
        }, "Largest supplied value the declared rank limit discarded; a report, not a residual");
    m.def("isa_localize_lw", &lw_binding_private::localize,
          "positions"_a, "blocks"_a, "frequency"_a, "bonds"_a, "residual_tolerance"_a = 1.0e-6,
          "input_sum_rule_tolerance"_a = -1.0, "rank_limit"_a = 3,
          "Supplied atomic-unit ordered-pair response; finite nonnegative frequency required. "
          "Positions: N x 3 Matrix in bohr; blocks: N*N single 16x16 Matrices, real Racah 00,10,11c,11s,... . "
          "Explicit zero-based graph; source-minus-target translations; local output ranks 1..3. "
          "residual_tolerance is a postcondition gate, not iteration; it always holds the "
          "algorithm-controlled residuals off_site, reciprocity, molecular_sum and charge_sum_transport. "
          "input_sum_rule_tolerance gates the supplied data's charge-flow sum-rule defect separately: "
          "negative inherits residual_tolerance (one combined gate), positive finite sets an explicit "
          "threshold, and infinity measures and reports the defect without gating it. LW transports such a "
          "defect exactly (measured <= 2.2e-16) and cannot repair it. At most 256 sites, 1000000 retained transfers, "
          "768 MiB native workspace budget (caller inputs/getter copies additional). "
          "rank_limit declares the rank the localization runs at, in 1..3, default 3 (the full working "
          "space, for which this routine is unchanged). A declared limit L truncates the supplied blocks "
          "to the leading (L+1)^2 real Racah components in both index slots first -- reported through "
          "truncated_input_maxabs -- and then runs the component-pair loop, the translated transfer "
          "application, the molecular-sum conservation check and the local output inside that space. The "
          "restriction is exact because multipole translation is rank-raising, so no tolerance is relaxed "
          "and every residual is gated at the same threshold as at rank 3. A limited localization is a "
          "DIFFERENT MODEL from the rank-3 one, but an exactly consistent one: because translation is "
          "rank-raising and the pair loop is ordered, no pair above the limit can write below it, so the "
          "result equals the rank-3 result restricted to the declared space, bitwise (measured 0.0). A "
          "declared limit therefore cannot change any rank <= L observable. "
          "No solver fallback, native-upstream verification, external PFIT refinement, or parity claim.");
    m.def("isa_lw_graph_math",
          [](std::size_t site_count, const std::vector<std::array<std::size_t, 2>>& bonds) {
              IsaBondGraph graph{site_count, bonds};
              auto operator_matrix = std::make_shared<Matrix>(isa_lw_graph_operator(graph));
              auto inverse_and_values = isa_lw_graph_pseudoinverse(graph);
              auto pseudoinverse = std::make_shared<Matrix>(std::move(inverse_and_values.first));
              return py::make_tuple(operator_matrix, pseudoinverse, inverse_and_values.second);
          }, "site_count"_a, "bonds"_a,
          "Owned (negative Laplacian, pseudoinverse, eigenvalues); explicit undirected zero-based edges, "
          "1..256 sites. Dimensionless graph math only, not localization or ORIENT parity.");
    m.def("isa_multipole_translation", &isa_multipole_translation, "rank"_a, "displacement"_a,
          "R(x+d)=T(d)R(x); source-minus-target displacement in bohr, Racah 00,10,11c,11s,...");
    m.def("isa_multipole_rotation", &isa_multipole_rotation, "rank"_a, "frame"_a,
          "R(F x)=D(F)R(x); finite proper orthogonal local-to-global Cartesian frame");
    m.def("isa_regular_multipoles", &isa_regular_multipoles, "rank"_a, "displacement"_a,
          "r^k C_kq for every rank through rank, Racah 00,10,11c,11s,...; rank zero is 1");
    m.def("isa_irregular_solid_harmonics", &isa_irregular_solid_harmonics, "rank"_a, "displacement"_a,
          "r^(-k-1) C_kq for every rank through rank, Racah 00,10,11c,11s,...; rank zero is 1/r, "
          "not 1, and the displacement (bohr) must be nonzero");
    m.def("isa_t_function_damping", &isa_t_function_damping, "rank"_a, "br"_a,
          "Tang-Toennies factor 1-exp(-br)*sum_{n<=rank+1} br^n/n! for a whole rank block; the "
          "reference protocol leaves damping off, so this branch has no reference artifact behind it");
    m.def("isa_t_functions", &isa_t_functions, "rank"_a, "point_bohr"_a, "site_bohr"_a, "frame"_a,
          "damping"_a = 0.0,
          "One pfit T row: unit-charge interaction functions for a site's multipole components in "
          "the site's LOCAL axes. frame maps local to global Cartesian (columns are the local axes), "
          "as in isa_multipole_rotation; damping is CamCASP's Damping keyword in bohr^-1");
    py::class_<IsaAnisotropicSite>(m, "IsaAnisotropicSite", "All declared-rank local response blocks; explicit local-to-global frame required")
        .def(py::init<>())
        .def_readwrite("label", &IsaAnisotropicSite::label)
        .def_readwrite("origin", &IsaAnisotropicSite::origin)
        .def_readwrite("frame", &IsaAnisotropicSite::frame)
        .def_readwrite("ranks", &IsaAnisotropicSite::ranks)
        .def_readwrite("responses", &IsaAnisotropicSite::responses)
        .def_property_readonly("components", &IsaAnisotropicSite::components);
    py::class_<IsaAnisotropicModel>(m, "IsaAnisotropicModel", "Owned supplied-local reciprocal real Racah responses; no localization or model inference")
        .def(py::init<const std::vector<double>&, const std::vector<IsaAnisotropicSite>&,
                      const std::string&, const std::string&>(),
             "frequencies"_a, "sites"_a, "declaration"_a, "provenance"_a)
        .def_property_readonly("frequencies", &IsaAnisotropicModel::frequencies)
        .def_property_readonly("sites", &IsaAnisotropicModel::sites)
        .def_property_readonly("declaration", &IsaAnisotropicModel::declaration)
        .def_property_readonly("provenance", &IsaAnisotropicModel::provenance)
        .def_property_readonly("units", [](const IsaAnisotropicModel&) { return "atomic_units"; });
    py::class_<IsaAnisotropicCoefficient>(m, "IsaAnisotropicCoefficient")
        .def_readonly("order", &IsaAnisotropicCoefficient::order)
        .def_readonly("value", &IsaAnisotropicCoefficient::value)
        .def_readonly("energy", &IsaAnisotropicCoefficient::energy)
        .def_readonly("declared_model_complete", &IsaAnisotropicCoefficient::declared_model_complete)
        .def_readonly("unrestricted_complete", &IsaAnisotropicCoefficient::unrestricted_complete)
        .def_readonly("included_rank_quadruples", &IsaAnisotropicCoefficient::included_rank_quadruples)
        .def_readonly("missing_rank_quadruples", &IsaAnisotropicCoefficient::missing_rank_quadruples);
    py::class_<IsaAnisotropicPair>(m, "IsaAnisotropicPair")
        .def_readonly("site_a", &IsaAnisotropicPair::site_a)
        .def_readonly("site_b", &IsaAnisotropicPair::site_b)
        .def_readonly("displacement", &IsaAnisotropicPair::displacement)
        .def_readonly("direction", &IsaAnisotropicPair::direction)
        .def_readonly("distance", &IsaAnisotropicPair::distance)
        .def_readonly("coefficients", &IsaAnisotropicPair::coefficients)
        .def_readonly("truncated_energy", &IsaAnisotropicPair::truncated_energy);
    py::class_<IsaAnisotropicDispersionResult>(m, "IsaAnisotropicDispersionResult")
        .def_property_readonly("model_a", [](const IsaAnisotropicDispersionResult& r) { return r.model_a; })
        .def_property_readonly("model_b", [](const IsaAnisotropicDispersionResult& r) { return r.model_b; })
        .def_readonly("frequencies", &IsaAnisotropicDispersionResult::frequencies)
        .def_readonly("cp_weights", &IsaAnisotropicDispersionResult::cp_weights)
        .def_readonly("pairs", &IsaAnisotropicDispersionResult::pairs)
        .def_readonly("max_order", &IsaAnisotropicDispersionResult::max_order)
        .def_readonly("truncated_energy", &IsaAnisotropicDispersionResult::truncated_energy)
        .def_property_readonly("energy_is_truncated", [](const IsaAnisotropicDispersionResult&) { return true; })
        .def_property_readonly("units", [](const IsaAnisotropicDispersionResult&) { return "atomic_units"; })
        .def_property_readonly("method", [](const IsaAnisotropicDispersionResult&) { return "supplied_local_anisotropic"; })
        .def_property_readonly("coefficient_representation", [](const IsaAnisotropicDispersionResult&) { return "orientation_resolved_scalar"; });
    m.def("isa_anisotropic_dispersion", &isa_anisotropic_dispersion,
          "model_a"_a, "model_b"_a, "cp_weights"_a, "max_order"_a=12,
          "All A/B pairs, undamped/nonretarded; scalar C_n for every order 6..max_order, not C_n(t,u,J). CP weights already include 1/(2*pi).");
    m.def("isa_anisotropic_interaction", &isa_anisotropic_interaction,
          "rank_a"_a, "rank_b"_a, "displacement"_a,
          "Expert physical Coulomb T=(-1)^l H_l(grad)H_k(grad)(1/R)/(d_l*d_k), ranks 1..4, R=B-A in bohr; rank axes 0,1c,1s,...; not scaled tau.");
    py::class_<IsaIsotropicSite>(m, "IsaIsotropicSite")
        .def(py::init<>())
        .def_readwrite("label", &IsaIsotropicSite::label)
        .def_readwrite("origin", &IsaIsotropicSite::origin)
        .def_readwrite("ranks", &IsaIsotropicSite::ranks)
        .def_readwrite("polarizabilities", &IsaIsotropicSite::polarizabilities);
    py::class_<IsaIsotropicModel>(m, "IsaIsotropicModel", "Owned supplied scalar alpha_l=trace(alpha_ll)/(2*l+1); atomic units, no localization")
        .def(py::init<const std::vector<double>&, const std::vector<IsaIsotropicSite>&, const std::string&>(),
             "frequencies"_a, "sites"_a, "provenance"_a)
        .def_property_readonly("frequencies", &IsaIsotropicModel::frequencies)
        .def_property_readonly("sites", &IsaIsotropicModel::sites)
        .def_property_readonly("provenance", &IsaIsotropicModel::provenance)
        .def_property_readonly("units", [](const IsaIsotropicModel&) { return "atomic_units"; });
    py::class_<IsaIsotropicCoefficient>(m, "IsaIsotropicCoefficient")
        .def_readonly("order", &IsaIsotropicCoefficient::order)
        .def_readonly("value", &IsaIsotropicCoefficient::value)
        .def_readonly("complete", &IsaIsotropicCoefficient::complete)
        .def_readonly("included_rank_pairs", &IsaIsotropicCoefficient::included_rank_pairs)
        .def_readonly("missing_rank_pairs", &IsaIsotropicCoefficient::missing_rank_pairs);
    py::class_<IsaIsotropicPair>(m, "IsaIsotropicPair")
        .def_readonly("site_a", &IsaIsotropicPair::site_a)
        .def_readonly("site_b", &IsaIsotropicPair::site_b)
        .def_readonly("coefficients", &IsaIsotropicPair::coefficients);
    py::class_<IsaIsotropicDispersionResult>(m, "IsaIsotropicDispersionResult")
        .def_readonly("pairs", &IsaIsotropicDispersionResult::pairs)
        .def_readonly("frequencies", &IsaIsotropicDispersionResult::frequencies)
        .def_readonly("cp_weights", &IsaIsotropicDispersionResult::cp_weights)
        .def_readonly("labels_a", &IsaIsotropicDispersionResult::labels_a)
        .def_readonly("labels_b", &IsaIsotropicDispersionResult::labels_b)
        .def_readonly("origins_a", &IsaIsotropicDispersionResult::origins_a)
        .def_readonly("origins_b", &IsaIsotropicDispersionResult::origins_b)
        .def_readonly("provenance_a", &IsaIsotropicDispersionResult::provenance_a)
        .def_readonly("provenance_b", &IsaIsotropicDispersionResult::provenance_b)
        .def_property_readonly("units", [](const IsaIsotropicDispersionResult&) { return "atomic_units"; })
        .def_property_readonly("method", [](const IsaIsotropicDispersionResult&) { return "supplied_local_isotropic"; });
    m.def("isa_isotropic_dispersion", &isa_isotropic_dispersion,
          "model_a"_a, "model_b"_a, "cp_weights"_a, "max_order"_a=12,
          "All A/B site pairs; cp_weights already contain 1/(2*pi); explicit rank completeness");
    py::class_<IsaMultipoleSamples>(m, "IsaMultipoleSamples")
        .def(py::init<>())
        .def_readwrite("points", &IsaMultipoleSamples::points)
        .def_readwrite("weights", &IsaMultipoleSamples::weights)
        .def_readwrite("shape", &IsaMultipoleSamples::shape)
        .def_readwrite("shape_sum", &IsaMultipoleSamples::shape_sum)
        .def_readwrite("auxiliary_sites", &IsaMultipoleSamples::auxiliary_sites);
    py::class_<IsaMultipoleSite>(m, "IsaMultipoleSite")
        .def(py::init<>())
        .def_readwrite("label", &IsaMultipoleSite::label)
        .def_readwrite("origin", &IsaMultipoleSite::origin)
        .def_readwrite("rank", &IsaMultipoleSite::rank)
        .def_readwrite("samples", &IsaMultipoleSite::samples);
    py::class_<IsaPartitionedMultipoles>(m, "IsaPartitionedMultipoles",
            "Supplied-partition Racah Q; global axes, bohr origins, atomic units; owned snapshots")
        .def(py::init<const IsaExplicitBasis&, const std::vector<IsaMultipoleSite>&, const std::string&, double>(),
            "auxiliary"_a, "sites"_a, "provenance"_a, "denominator_cutoff"_a=1.e-36)
        .def(py::init<const IsaExplicitBasis&, const std::vector<IsaMultipoleSite>&, const std::string&, double,
         std::shared_ptr<Matrix>, int>(), "orbital"_a, "sites"_a, "provenance"_a,
         "denominator_cutoff"_a, "orbitals"_a, "nocc"_a)
    .def_property_readonly("representation", &IsaPartitionedMultipoles::representation)
    .def_property_readonly("values", &IsaPartitionedMultipoles::values)
        .def_property_readonly("offsets", &IsaPartitionedMultipoles::offsets)
        .def_property_readonly("ranks", &IsaPartitionedMultipoles::ranks)
        .def_property_readonly("labels", &IsaPartitionedMultipoles::labels)
        .def_property_readonly("components", &IsaPartitionedMultipoles::components)
        .def_property_readonly("origins", &IsaPartitionedMultipoles::origins)
        .def_property_readonly("excluded_denominators", &IsaPartitionedMultipoles::excluded_denominators)
        .def_property_readonly("negative_ratios", &IsaPartitionedMultipoles::negative_ratios)
        .def_property_readonly("provenance", &IsaPartitionedMultipoles::provenance)
        .def_property_readonly("denominator_cutoff", &IsaPartitionedMultipoles::denominator_cutoff)
        .def_property_readonly("frame", [](const IsaPartitionedMultipoles&) { return "global_cartesian"; })
        .def_property_readonly("units", [](const IsaPartitionedMultipoles&) { return "atomic_units"; });
    py::class_<IsaDistributedResponse>(m, "IsaDistributedResponse",
            "Raw -Q C_DF Q^T; site/component axes from partition, atomic units; no symmetrization")
        .def(py::init<const IsaPartitionedMultipoles&, const std::vector<double>&,
             const std::vector<std::shared_ptr<Matrix>>&, const std::string&, const std::string&>(),
             "partition"_a, "frequencies"_a, "coefficient_responses"_a, "representation"_a, "provenance"_a)
        .def("at_index", &IsaDistributedResponse::at_index, "index"_a)
        .def_property_readonly("frequencies", &IsaDistributedResponse::frequencies)
        .def_property_readonly("reciprocity_errors", &IsaDistributedResponse::reciprocity_errors)
        .def_property_readonly("partition", &IsaDistributedResponse::partition)
        .def_property_readonly("provenance", &IsaDistributedResponse::provenance)
        .def_property_readonly("units", [](const IsaDistributedResponse&) { return "atomic_units"; });
    py::enum_<IsaBasisRole>(m, "IsaBasisRole")
        .value("MolecularAux", IsaBasisRole::MolecularAux)
        .value("AtomAux", IsaBasisRole::AtomAux)
        .value("Shape", IsaBasisRole::Shape)
        .value("Orbital", IsaBasisRole::Orbital);
    py::enum_<IsaBasisRepresentation>(m, "IsaBasisRepresentation")
        .value("Cartesian", IsaBasisRepresentation::Cartesian)
        .value("Spherical", IsaBasisRepresentation::Spherical);
    py::class_<IsaGaussianShell>(m, "IsaGaussianShell", "Zero-based centre; effective coefficients, no renormalization")
        .def(py::init<>())
        .def_readwrite("centre", &IsaGaussianShell::centre)
        .def_readwrite("l", &IsaGaussianShell::l)
        .def_readwrite("exponents", &IsaGaussianShell::exponents)
        .def_readwrite("coefficients", &IsaGaussianShell::coefficients);
    py::class_<IsaExplicitBasis>(m, "IsaExplicitBasis", "Owned immutable exported-input basis; bohr, GAMINT/DALTON S-G")
        .def(py::init<IsaBasisRole, IsaBasisRepresentation, const std::vector<std::array<double,3>>&,
                     const std::vector<IsaGaussianShell>&>(), "role"_a, "representation"_a, "centres"_a, "shells"_a)
        .def_property_readonly("nfunction", &IsaExplicitBasis::nfunction)
        .def_property_readonly("role", &IsaExplicitBasis::role)
        .def("overlap", &IsaExplicitBasis::overlap, "w_eps"_a = 0.0, "s_block_only"_a = true,
             "New co-centred AtomAux/Shape metric before damping/ridge; no exponent cap")
        .def("evaluate", &IsaExplicitBasis::evaluate, "points"_a,
             "Return new (point,function) Matrix; no retained input views")
        .def("evaluate_screened", &IsaExplicitBasis::evaluate_screened, "points"_a, "sites"_a,
             "Unique zero-based active sites without padding; empty means no sites");
    py::class_<IsaDrhoCResult>(m, "IsaDrhoCResult", "Native explicit-input finite-penalty Drho-C result; no charge rescaling")
        .def_readonly("coulomb_metric", &IsaDrhoCResult::coulomb_metric)
        .def_readonly("metric", &IsaDrhoCResult::metric)
        .def_readonly("charges", &IsaDrhoCResult::charges)
        .def_readonly("raw_rhs", &IsaDrhoCResult::raw_rhs)
        .def_readonly("rhs", &IsaDrhoCResult::rhs)
        .def_readonly("coefficients", &IsaDrhoCResult::coefficients)
        .def_readonly("charge_penalty", &IsaDrhoCResult::charge_penalty)
        .def_readonly("relative_residual", &IsaDrhoCResult::relative_residual)
        .def_readonly("fitted_electrons", &IsaDrhoCResult::fitted_electrons);
    py::class_<IsaOvFitResult>(m, "IsaOvFitResult", "Owned native-integral/supplied-orbital OV fit; not native SCF/response")
        .def_property_readonly("coulomb_metric", &IsaOvFitResult::coulomb_metric)
        .def_property_readonly("metric", &IsaOvFitResult::metric)
        .def_property_readonly("rhs", &IsaOvFitResult::rhs)
        .def_property_readonly("coefficients", &IsaOvFitResult::coefficients)
        .def_property_readonly("charges", &IsaOvFitResult::charges)
        .def_property_readonly("nmain", &IsaOvFitResult::nmain)
        .def_property_readonly("naux", &IsaOvFitResult::naux)
        .def_property_readonly("noccupied", &IsaOvFitResult::noccupied)
        .def_property_readonly("nvirtual", &IsaOvFitResult::nvirtual)
        .def_property_readonly("ntransition", &IsaOvFitResult::ntransition)
        .def_property_readonly("charge_penalty", &IsaOvFitResult::charge_penalty)
        .def_property_readonly("relative_backward_residual", &IsaOvFitResult::relative_backward_residual)
        .def_property_readonly("lapack_info", &IsaOvFitResult::lapack_info)
        .def_property_readonly("provenance", &IsaOvFitResult::provenance)
        .def_property_readonly("representation", &IsaOvFitResult::representation)
        .def_property_readonly("order", &IsaOvFitResult::order)
        .def_property_readonly("solver", &IsaOvFitResult::solver);
    py::class_<IsaAuxCoulomb>(m, "IsaAuxCoulomb", "Native Libint2 Coulomb metric and analytic charges; explicit Cartesian molecular AUX only")
        .def(py::init<const IsaExplicitBasis&>(), "auxiliary"_a)
        .def("fit_ov", &IsaAuxCoulomb::fit_ov, "orbital"_a, "occupied"_a, "virtuals"_a,
             "provenance"_a, "charge_penalty"_a=1.0)
        .def("charges", &IsaAuxCoulomb::charges)
        .def("metric", &IsaAuxCoulomb::metric)
        .def("three_center", &IsaAuxCoulomb::three_center, "orbital"_a)
        .def("closed_shell_rhs", &IsaAuxCoulomb::closed_shell_rhs, "orbital"_a, "occupied_coefficients"_a)
        .def("fit_drho_c", &IsaAuxCoulomb::fit_drho_c, "orbital"_a, "occupied_coefficients"_a, "charge_penalty"_a=1000.);
    py::class_<IsaShapeMap>(m, "IsaShapeMap", "Validated zero-based shape-shell to AtomAux-shell map; exact descriptors")
        .def(py::init<const IsaExplicitBasis&, const IsaExplicitBasis&, const std::vector<int>&>(),
             "atomic"_a, "shape"_a, "shell_map"_a)
        .def_property_readonly("function_indices", &IsaShapeMap::function_indices)
        .def("project", &IsaShapeMap::project, "atomic_coefficients"_a,
             "New raw shape coefficient vector, before mixing/DIIS or tails");
    py::class_<IsaExponentialTail>(m, "IsaExponentialTail", "Explicit Func-1 parameters; signed amplitude permitted")
        .def(py::init<>())
        .def_readwrite("defined", &IsaExponentialTail::defined)
        .def_readwrite("amplitude", &IsaExponentialTail::amplitude)
        .def_readwrite("exponent", &IsaExponentialTail::exponent)
        .def_readwrite("cutoff", &IsaExponentialTail::cutoff);
    py::class_<IsaTailFitResult>(m, "IsaTailFitResult")
        .def_property_readonly("tail", [](const IsaTailFitResult& r) { return r.tail; })
        .def_readonly("used_previous_exponent", &IsaTailFitResult::used_previous_exponent)
        .def_readonly("gaussian_tail_charge", &IsaTailFitResult::gaussian_tail_charge)
        .def_readonly("ionization_potential", &IsaTailFitResult::ionization_potential)
        .def_readonly("status", &IsaTailFitResult::status);
    py::class_<IsaGaussianShape>(m, "IsaGaussianShape", "Owned effective s-Gaussian expansion; explicit Func-1/Fit-3 tail policy")
        .def(py::init<const IsaExplicitBasis&, const std::vector<double>&>(), "basis"_a, "coefficients"_a)
        .def("value", &IsaGaussianShape::value, "radius"_a)
        .def("exterior_charge", &IsaGaussianShape::exterior_charge, "radius"_a)
        .def("fit_tail", &IsaGaussianShape::fit_tail, "cutoff"_a, "previous"_a = IsaExponentialTail())
        .def("sample", &IsaGaussianShape::sample, "points"_a, "tail"_a = IsaExponentialTail(), "apply_tail"_a = false);
    py::class_<IsaFixedDensity>(m, "IsaFixedDensity", "Owned supplied molecular AUX expansion, not native Drho-C fitting")
        .def(py::init<const IsaExplicitBasis&, const std::vector<double>&>(), "basis"_a, "coefficients"_a)
        .def("evaluate", &IsaFixedDensity::evaluate, "points"_a, "sites"_a,
             "Signed density samples; explicit zero-based active sites, empty screens all");
    py::class_<IsaAFitOptions>(m, "IsaAFitOptions", "Active settings for one frozen ISA-A update")
        .def(py::init<>())
        .def_readwrite("w_eps", &IsaAFitOptions::w_eps)
        .def_readwrite("s_block_only", &IsaAFitOptions::s_block_only)
        .def_readwrite("damping", &IsaAFitOptions::damping)
        .def_readwrite("positive_lambda", &IsaAFitOptions::positive_lambda)
        .def_readwrite("positive_max_alpha", &IsaAFitOptions::positive_max_alpha)
        .def_readwrite("positive_auto", &IsaAFitOptions::positive_auto)
        .def_readwrite("density_cutoff", &IsaAFitOptions::density_cutoff);
    py::class_<IsaAFitData>(m, "IsaAFitData", "Explicit samples and weighted metric; no basis construction or activation scheduling")
        .def(py::init<>())
        .def_readwrite("weights", &IsaAFitData::weights)
        .def_readwrite("density", &IsaAFitData::density)
        .def_readwrite("shape", &IsaAFitData::shape)
        .def_readwrite("shape_sum", &IsaAFitData::shape_sum)
        .def_readwrite("radius_squared", &IsaAFitData::radius_squared)
        .def_readwrite("basis_values", &IsaAFitData::basis_values)
        .def_readwrite("overlap", &IsaAFitData::overlap, "Already W-Eps-weighted overlap, before damping/ridge")
        .def_readwrite("previous", &IsaAFitData::previous)
        .def_readwrite("angular_momenta", &IsaAFitData::angular_momenta)
        .def_readwrite("exponents", &IsaAFitData::exponents,
                       "Positive primitive exponent per function, including non-s. Caller must supply an uncontracted basis.");
    py::class_<IsaAFitResult>(m, "IsaAFitResult", "One fitted update, not a converged ISA partition")
        .def_property_readonly("metric", [](const IsaAFitResult& r) { return r.metric->clone(); })
        .def_property_readonly("rhs", [](const IsaAFitResult& r) { return r.rhs->clone(); })
        .def_property_readonly("coefficients", [](const IsaAFitResult& r) { return r.coefficients->clone(); })
        .def_readonly("population", &IsaAFitResult::population)
        .def_readonly("relative_residual", &IsaAFitResult::relative_residual)
        .def_readonly("excluded_points", &IsaAFitResult::excluded_points);
    py::class_<IsaAFitSamples>(m, "IsaAFitSamples", "Explicit quadrature and already screened/tail-processed old shapes")
        .def(py::init<>())
        .def_readwrite("points", &IsaAFitSamples::points)
        .def_readwrite("weights", &IsaAFitSamples::weights)
        .def_readwrite("shape", &IsaAFitSamples::shape)
        .def_readwrite("shape_sum", &IsaAFitSamples::shape_sum)
        .def_readwrite("previous", &IsaAFitSamples::previous)
        .def_readwrite("density_sites", &IsaAFitSamples::density_sites);
    py::class_<IsaAFitProvider>(m, "IsaAFitProvider", "Owned primitive AtomAux and explicit fixed molecular-AUX density")
        .def(py::init<const IsaExplicitBasis&, const IsaFixedDensity&>(), "atomic"_a, "density"_a)
        .def("assemble", &IsaAFitProvider::assemble, "samples"_a, "options"_a = IsaAFitOptions(),
             "Fresh fit data; subsequent solve must use identical weighting settings")
        .def("fit", &IsaAFitProvider::fit, "samples"_a, "options"_a = IsaAFitOptions(),
             "Assemble and perform one frozen fit using the same options, not an ISA iteration");
    py::class_<IsaNoTailGrid>(m, "IsaNoTailGrid", "Explicit atom grid with distinct shape and molecular-density neighbour indices")
        .def(py::init<>())
        .def_readwrite("points", &IsaNoTailGrid::points)
        .def_readwrite("weights", &IsaNoTailGrid::weights)
        .def_readwrite("density_sites", &IsaNoTailGrid::density_sites)
        .def_readwrite("shape_sites", &IsaNoTailGrid::shape_sites);
    py::class_<IsaSweepState>(m, "IsaSweepState", "Explicit atomic/shape expansions in sweep atom order")
        .def(py::init<>())
        .def_readwrite("atomic_coefficients", &IsaSweepState::atomic_coefficients)
        .def_readwrite("shape_coefficients", &IsaSweepState::shape_coefficients);
    py::class_<IsaNoTailSweepResult>(m, "IsaNoTailSweepResult")
        .def_property_readonly("next", [](const IsaNoTailSweepResult& r) { return r.next; })
        .def_property_readonly("fits", [](const IsaNoTailSweepResult& r) { return r.fits; })
        .def_readonly("clipped_shape_points", &IsaNoTailSweepResult::clipped_shape_points);
    py::class_<IsaASweep>(m, "IsaASweep", "Owned synchronous sweep with explicit no-tail or supplied-tail sampling")
        .def(py::init<const std::vector<IsaExplicitBasis>&, const std::vector<IsaExplicitBasis>&,
                     const std::vector<std::vector<int>>&, const IsaFixedDensity&>(),
             "atomic"_a, "shape"_a, "shell_maps"_a, "density"_a)
        .def("run", &IsaASweep::run, "old"_a, "grids"_a, "options"_a = IsaAFitOptions())
        .def("run_with_tails", &IsaASweep::run_with_tails, "old"_a, "grids"_a, "tails"_a,
             "apply_tail"_a, "options"_a = IsaAFitOptions());
    m.attr("IsaNoTailSweep") = m.attr("IsaASweep");
    py::class_<IsaAControllerOptions>(m, "IsaAControllerOptions", "Ordinary A / W convergence, no DIIS or symmetry; explicit tail cutoffs")
        .def(py::init<>())
        .def_readwrite("fit", &IsaAControllerOptions::fit)
        .def_readwrite("convergence", &IsaAControllerOptions::convergence)
        .def_readwrite("w_eps_activation", &IsaAControllerOptions::w_eps_activation)
        .def_readwrite("positive_activation", &IsaAControllerOptions::positive_activation)
        .def_readwrite("tail_activation", &IsaAControllerOptions::tail_activation)
        .def_readwrite("mixing", &IsaAControllerOptions::mixing)
        .def_readwrite("mixing_skip", &IsaAControllerOptions::mixing_skip)
        .def_readwrite("cache_max_bytes", &IsaAControllerOptions::cache_max_bytes)
        .def_readwrite("tail_iteration_limit", &IsaAControllerOptions::tail_iteration_limit)
        .def_readwrite("max_iterations", &IsaAControllerOptions::max_iterations)
        .def_readwrite("fix_tails", &IsaAControllerOptions::fix_tails)
        .def_readwrite("tail_cutoffs", &IsaAControllerOptions::tail_cutoffs)
        .def_readwrite("tail_allowed", &IsaAControllerOptions::tail_allowed)
        .def_readwrite("convergence_included", &IsaAControllerOptions::convergence_included);
    py::class_<IsaAControllerState>(m, "IsaAControllerState", "Restart cursor for identical controller inputs/settings")
        .def(py::init<>())
        .def_readwrite("coefficients", &IsaAControllerState::coefficients)
        .def_readwrite("tails", &IsaAControllerState::tails)
        .def_readwrite("saved_shape_charges", &IsaAControllerState::saved_shape_charges)
        .def_readwrite("iteration", &IsaAControllerState::iteration)
        .def_readwrite("active_w_eps", &IsaAControllerState::active_w_eps)
        .def_readwrite("active_positive_lambda", &IsaAControllerState::active_positive_lambda)
        .def_readwrite("max_delta", &IsaAControllerState::max_delta)
        .def_readwrite("apply_tails", &IsaAControllerState::apply_tails)
        .def_readwrite("converged", &IsaAControllerState::converged);
    py::class_<IsaAControllerStep>(m, "IsaAControllerStep")
        .def_property_readonly("next", [](const IsaAControllerStep& r) { return r.next; })
        .def_property_readonly("raw_sweep", [](const IsaAControllerStep& r) { return r.raw_sweep; })
        .def_readonly("deltas", &IsaAControllerStep::deltas)
        .def_readonly("shape_charges", &IsaAControllerStep::shape_charges)
        .def_readonly("atom_converged", &IsaAControllerStep::atom_converged)
        .def_readonly("tail_fits", &IsaAControllerStep::tail_fits);
    py::class_<IsaAControllerResult>(m, "IsaAControllerResult")
        .def_property_readonly("state", [](const IsaAControllerResult& r) { return r.state; })
        .def_readonly("history", &IsaAControllerResult::history)
        .def_readonly("termination", &IsaAControllerResult::termination);
    py::class_<IsaAController, std::shared_ptr<IsaAController>>(m, "IsaAController", "Explicit-input ordinary-A controller, not native wavefunction-to-property parity")
        .def(py::init<const std::vector<IsaExplicitBasis>&, const std::vector<IsaExplicitBasis>&,
                     const std::vector<std::vector<int>>&, const IsaFixedDensity&,
                     const std::vector<IsaNoTailGrid>&, const IsaAControllerOptions&>(),
             "atomic"_a, "shape"_a, "shell_maps"_a, "density"_a, "grids"_a, "options"_a)
        .def_property_readonly("prepared_cache_enabled", &IsaAController::prepared_cache_enabled,
                               "Whether the bounded immutable preparation was admitted")
        .def("without_prepared_cache", &IsaAController::without_prepared_cache,
             "Return an independent rerunnable controller without retained preparation")
        .def("initialize", &IsaAController::initialize, "coefficients"_a)
        .def("step", &IsaAController::step, "old"_a)
        .def("run", &IsaAController::run, "initial"_a);
    m.def("isa_a_fit_step", &isa_a_fit_step, "data"_a, "options"_a = IsaAFitOptions(),
          "Fit one atom against frozen samples. Inputs are not mutated/retained; result matrices are copies.");
    m.def("isa_overlap_change", &isa_overlap_change, "current"_a, "previous"_a, "overlap"_a,
          "Normalized overlap-angle change using an unweighted metric; insensitive to amplitude scaling.");

    py::class_<IsaGridOptions>(m, "IsaGridOptions", "Options controlling the ISA integration grid")
        .def(py::init<>())
        .def_readwrite("radial_points", &IsaGridOptions::radial_points,
                       "CamCASP n_r; the Euler-MacLaurin map yields n_r - 1 shells")
        .def_readwrite("spherical_points", &IsaGridOptions::spherical_points,
                       "Requested Lebedev order, rounded up to the next tabulated size")
        .def_readwrite("becke_smoothing", &IsaGridOptions::becke_smoothing, "Becke k_mu")
        .def_readwrite("radius_scaling", &IsaGridOptions::radius_scaling, "CamCASP rscale");

    py::class_<IsaGrid, std::shared_ptr<IsaGrid>>(m, "IsaGrid",
                                                  "Atom-centred integration grid used by the ISA machinery")
        .def(py::init<std::shared_ptr<Molecule>, const IsaGridOptions&>(), "molecule"_a, "options"_a)
        .def("natom", &IsaGrid::natom)
        .def("npoints", &IsaGrid::npoints)
        .def("spherical_points", &IsaGrid::spherical_points)
        .def("radial_points", &IsaGrid::radial_points)
        .def("atom_start", &IsaGrid::atom_start, "A"_a)
        .def("atom_npoints", &IsaGrid::atom_npoints, "A"_a)
        .def("alpha", &IsaGrid::alpha, "A"_a)
        .def("print_header", &IsaGrid::print_header)
        .def("x", [](const IsaGrid& g) { return grid_column(g, g.x()); })
        .def("y", [](const IsaGrid& g) { return grid_column(g, g.y()); })
        .def("z", [](const IsaGrid& g) { return grid_column(g, g.z()); })
        .def("w", [](const IsaGrid& g) { return grid_column(g, g.w()); });

    py::class_<CasimirGrid, std::shared_ptr<CasimirGrid>>(
        m, "CasimirGrid", "Gauss-Legendre quadrature on the imaginary frequency axis")
        .def(py::init<int, double>(), "n_freq"_a, "omega0"_a = kCasimirOmega0)
        .def("n_freq", &CasimirGrid::n_freq)
        .def("omega0", &CasimirGrid::omega0)
        .def("omega", &CasimirGrid::omega, "k"_a, "Imaginary frequency k in hartree; k = 0 is the static point")
        .def("tm1sq", &CasimirGrid::tm1sq, "k"_a, "(1 - t_k)^2 for the mapped Gauss-Legendre root")
        .def("weight", &CasimirGrid::weight, "k"_a, "Raw Gauss-Legendre weight of point k")
        .def("wsq", &CasimirGrid::wsq, "k"_a, "-omega_k^2")
        .def("cp_weight", &CasimirGrid::cp_weight, "k"_a, "Weight of point k in the Casimir-Polder integral")
        .def("omegas", [](const CasimirGrid& g) { return py::array_t<double>(g.omegas().size(), g.omegas().data()); });

    py::class_<MaclarenRng, std::shared_ptr<MaclarenRng>>(m, "MaclarenRng",
                                                          "Maclaren (1992) generator, CamCASP's sdprnd/dprand")
        .def(py::init<int>(), "seed"_a = 0)
        .def("seed", &MaclarenRng::seed, "seed"_a)
        .def("next", &MaclarenRng::next, "Next uniform deviate on (0, 1)")
        .def("take", [](MaclarenRng& r, int n) {
            std::vector<double> v(n);
            for (int i = 0; i < n; ++i) v[i] = r.next();
            return py::array_t<double>(v.size(), v.data());
        }, "n"_a, "Next n deviates");

    py::class_<FitPointsOptions>(m, "FitPointsOptions", "Options controlling the ISA-Pol fit-point cloud")
        .def(py::init<>())
        .def_readwrite("npoints", &FitPointsOptions::npoints, "CamCASP nlat; number of points to accept")
        .def_readwrite("lolim", &FitPointsOptions::lolim, "Inner cutoff in van der Waals radii")
        .def_readwrite("hilim", &FitPointsOptions::hilim, "Outer cutoff in van der Waals radii")
        .def_readwrite("seed", &FitPointsOptions::seed, "Seed for the point generator");

    py::class_<FitPoints, std::shared_ptr<FitPoints>>(m, "FitPoints",
                                                      "Points at which the point-response refinement samples the potential")
        .def(py::init<std::shared_ptr<Molecule>, const FitPointsOptions&>(), "molecule"_a, "options"_a)
        .def("npoints", &FitPoints::npoints)
        .def("ncandidates", &FitPoints::ncandidates)
        .def("dmax", &FitPoints::dmax)
        .def("print_header", &FitPoints::print_header)
        .def("centre", [](const FitPoints& f) { return py::array_t<double>(3, f.centre()); })
        .def("x", [](const FitPoints& f) { return py::array_t<double>(f.npoints(), f.x()); })
        .def("y", [](const FitPoints& f) { return py::array_t<double>(f.npoints(), f.y()); })
        .def("z", [](const FitPoints& f) { return py::array_t<double>(f.npoints(), f.z()); });

    py::class_<RecouplingTerm>(m, "RecouplingTerm",
                               "One Casimir-Polder term of one anisotropic dispersion coefficient")
        .def_readonly("p", &RecouplingTerm::p, "Numerator of the rational factor; carries the sign")
        .def_readonly("q", &RecouplingTerm::q, "Denominator of the rational factor")
        .def_readonly("r", &RecouplingTerm::r, "Numerator under the square root")
        .def_readonly("s", &RecouplingTerm::s, "Denominator under the square root")
        .def_readonly("la", &RecouplingTerm::la, "First rank label of alpha^A")
        .def_readonly("lap", &RecouplingTerm::lap, "Second rank label of alpha^A")
        .def_readonly("lb", &RecouplingTerm::lb, "First rank label of alpha^B")
        .def_readonly("lbp", &RecouplingTerm::lbp, "Second rank label of alpha^B")
        .def_readonly("ipow", &RecouplingTerm::ipow, "Power of i to fold in, 0 or 1")
        .def_property_readonly("coefficient", &RecouplingTerm::coefficient, "(p/q) sqrt(r/s)");

    m.def("isapol_recoupling_block",
          [](int n, int L1, int L2, int J) {
              RecouplingBlock block = recoupling_block(n, L1, L2, J);
              return std::vector<RecouplingTerm>(block.begin(), block.end());
          },
          "n"_a, "L1"_a, "L2"_a, "J"_a,
          "Terms of C_n(t, u, J) for t of rank L1 and u of rank L2; empty if the block vanishes");
    m.def("isapol_recoupling_blocks",
          []() {
              std::vector<std::tuple<int, int, int, int, std::vector<RecouplingTerm>>> all;
              for (int i = 0; i < num_recoupling_blocks(); ++i) {
                  int n, L1, L2, J;
                  RecouplingBlock block = recoupling_block_at(i, &n, &L1, &L2, &J);
                  all.emplace_back(n, L1, L2, J, std::vector<RecouplingTerm>(block.begin(), block.end()));
              }
              return all;
          },
          "Every tabulated block as (n, L1, L2, J, terms), in increasing key order");
    m.def("isapol_component_rank", &component_rank, "t"_a, "Rank L of real spherical-tensor component t");
    m.def("isapol_component_first", &component_first, "L"_a, "First component index of rank L");
    m.def("isapol_component_last", &component_last, "L"_a, "Last component index of rank L");
    m.def("isapol_component_label", &component_label, "t"_a, "CamCASP label for component t");

    py::class_<RealCGTerm>(m, "IsaRealCGTerm")
        .def_readonly("la", &RealCGTerm::la).def_readonly("lap", &RealCGTerm::lap)
        .def_readonly("k", &RealCGTerm::k).def_readonly("q", &RealCGTerm::q)
        .def_readonly("v", &RealCGTerm::v).def_readonly("p", &RealCGTerm::p)
        .def_readonly("denominator", &RealCGTerm::denominator)
        .def_readonly("r", &RealCGTerm::r).def_readonly("s", &RealCGTerm::s)
        .def_property_readonly("value", &RealCGTerm::value);
    m.def("isapol_realcg_terms", &realcg_terms, "la"_a, "lap"_a);
    m.def("isapol_realcg_defined", &realcg_defined, "la"_a, "lap"_a,
          "True for the ordered rank pairs upstream defines: 1<=la,lap<=4 with la+lap<=6. "
          "casimir.f90 read_cg/recouple skip j1+j2>6, so (3,4), (4,3) and (4,4) have no "
          "table and no initialized coupled tensor; they are structurally absent.");
    py::class_<IsaRecoupledBlock>(m, "IsaRecoupledBlock")
        .def_readonly("la", &IsaRecoupledBlock::la).def_readonly("lap", &IsaRecoupledBlock::lap)
        .def_readonly("first_component", &IsaRecoupledBlock::first_component)
        .def_readonly("last_component", &IsaRecoupledBlock::last_component)
        .def_property_readonly("components", &IsaRecoupledBlock::components)
        .def_property_readonly("values", [](const IsaRecoupledBlock& b) { return b.values; });
    py::class_<IsaRecoupledModel>(m, "IsaRecoupledModel")
        .def(py::init<const IsaAnisotropicModel&>(), "local_model"_a)
        .def_property_readonly("source", &IsaRecoupledModel::source)
        .def_property_readonly("frequencies", &IsaRecoupledModel::frequencies)
        .def_property_readonly("sites", &IsaRecoupledModel::sites)
        .def("value", &IsaRecoupledModel::value,
             "site"_a, "frequency"_a, "la"_a, "lap"_a, "component"_a);
    py::class_<IsaRecoupledCoefficient>(m, "IsaRecoupledCoefficient")
        .def_readonly("order", &IsaRecoupledCoefficient::order)
        .def_readonly("t", &IsaRecoupledCoefficient::t).def_readonly("u", &IsaRecoupledCoefficient::u)
        .def_readonly("J", &IsaRecoupledCoefficient::J).def_readonly("value", &IsaRecoupledCoefficient::value);
    py::class_<IsaRecoupledCoverage>(m, "IsaRecoupledCoverage")
        .def_readonly("order", &IsaRecoupledCoverage::order)
        .def_readonly("table_complete", &IsaRecoupledCoverage::table_complete)
        .def_readonly("unrestricted_complete", &IsaRecoupledCoverage::unrestricted_complete)
        .def_readonly("included_rank_quadruples", &IsaRecoupledCoverage::included_rank_quadruples)
        .def_readonly("missing_table_rank_quadruples", &IsaRecoupledCoverage::missing_table_rank_quadruples)
        .def_readonly("missing_unrestricted_rank_quadruples", &IsaRecoupledCoverage::missing_unrestricted_rank_quadruples);
    py::class_<IsaRecoupledPair>(m, "IsaRecoupledPair")
        .def_readonly("site_a", &IsaRecoupledPair::site_a).def_readonly("site_b", &IsaRecoupledPair::site_b)
        .def_readonly("coefficients", &IsaRecoupledPair::coefficients)
        .def_readonly("coverage", &IsaRecoupledPair::coverage)
        .def("coefficient", &IsaRecoupledPair::coefficient, "order"_a, "t"_a, "u"_a, "J"_a);
    py::class_<IsaRecoupledDispersionResult>(m, "IsaRecoupledDispersionResult")
        .def_readonly("model_a", &IsaRecoupledDispersionResult::model_a)
        .def_readonly("model_b", &IsaRecoupledDispersionResult::model_b)
        .def_readonly("frequencies", &IsaRecoupledDispersionResult::frequencies)
        .def_readonly("cp_weights", &IsaRecoupledDispersionResult::cp_weights)
        .def_readonly("pairs", &IsaRecoupledDispersionResult::pairs)
        .def_readonly("max_order", &IsaRecoupledDispersionResult::max_order);
    m.def("isa_recoupled_dispersion", [](const IsaRecoupledModel& a, const IsaRecoupledModel& b,
                                            const py::sequence& weights, int max_order) {
              if (py::len(weights) != a.frequencies().size())
                  throw std::invalid_argument("recoupled: CP weight dimensions disagree");
              return isa_recoupled_dispersion(a,b,weights.cast<std::vector<double>>(),max_order);
          },
          "model_a"_a, "model_b"_a, "cp_weights"_a, "max_order"_a=12,
          "Local-axis C6..C12(t,u,J), not scalar orientation energies; CP weights already 1/(2*pi)");

    m.attr("ISAPOL_MIN_DISPERSION_ORDER") = kMinDispersionOrder;
    m.attr("ISAPOL_MAX_DISPERSION_ORDER") = kMaxDispersionOrder;
    m.attr("ISAPOL_MAX_DISPERSION_RANK") = kMaxDispersionRank;
    m.attr("ISAPOL_MAX_POLARIZABILITY_RANK") = kMaxPolarizabilityRank;
    m.attr("ISAPOL_CASIMIR_OMEGA0") = kCasimirOmega0;
    m.attr("ISAPOL_OMEGA0") = kIsaPolOmega0;

    m.def("isapol_slater_radius", &slater_radius, "Z"_a,
          "CamCASP Bragg-Slater radius in bohr (a_o = 0.529177249)");
    m.def("isapol_vdw_radius_bondi", &vdw_radius_bondi, "Z"_a,
          "Bondi van der Waals radius in bohr, from AtomProp (float32-rounded)");
    m.def("isapol_vdw_radius", &vdw_radius, "Z"_a,
          "Bondi van der Waals radius in bohr, from MODULE radii (double); used by the fit points");
    m.def("isapol_vdw_radius_grimme", &vdw_radius_grimme, "Z"_a, "Grimme van der Waals radius in bohr");
    m.def("isapol_c6_grimme", &c6_grimme, "Z"_a, "Grimme C6 coefficient in atomic units");
    m.def("isapol_covalent_radius", &covalent_radius, "Z"_a, "Covalent radius in bohr");
    m.def("isapol_element_symbol", &element_symbol, "Z"_a);
}
