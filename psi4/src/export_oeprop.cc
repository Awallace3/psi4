/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
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

#include <algorithm>
#include <limits>

#include "psi4/libfunctional/LibXCfunctional.h"
#include "psi4/libfunctional/functional.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libfock/cubature.h"
#include "psi4/libmints/oeprop.h"
#include "psi4/libmints/atomic_polarizability.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libscf_solver/hf.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/process.h"

using namespace psi;
namespace py = pybind11;
using namespace pybind11::literals;

void export_oeprop(py::module &m) {
    const auto isa_options_from_dict = [](const py::dict& values) {
        for (const auto& entry : values) {
            const auto key = py::cast<std::string>(entry.first);
            if (key != "radial_points" && key != "angular_polar_points" &&
                key != "angular_azimuthal_points" && key != "max_iterations" &&
                key != "convergence" && key != "mix_fraction" && key != "initial_alpha" &&
                key != "tail_join_factor" && key != "tail_activation_iteration" &&
                key != "tail_activation_convergence" && key != "electron_count_tolerance")
                throw PSIEXCEPTION("ISAOptions: unknown option '" + key + "'");
        }
        const auto size_value = [&values](const char* key, std::size_t fallback) {
            return values.contains(key) ? values[key].cast<std::size_t>() : fallback;
        };
        const auto double_value = [&values](const char* key, double fallback) {
            return values.contains(key) ? values[key].cast<double>() : fallback;
        };
        return ISAOptions(size_value("radial_points", 100), size_value("angular_polar_points", 18),
                          size_value("angular_azimuthal_points", 24), size_value("max_iterations", 120),
                          double_value("convergence", 1.0e-9), double_value("mix_fraction", 1.0),
                          double_value("initial_alpha", 1.0), double_value("tail_join_factor", 1.5),
                          size_value("tail_activation_iteration", 20),
                          double_value("tail_activation_convergence", 1.0e-6),
                          double_value("electron_count_tolerance", 0.1));
    };
    const auto isa_diagnostics_dict = [](const ISADiagnostics& diagnostics) {
        py::dict grid;
        grid["radial_points"] = diagnostics.grid_profile.radial_points;
        grid["angular_points"] = diagnostics.grid_profile.angular_points;
        grid["shell_point_count"] = diagnostics.grid_profile.shell_point_count;
        grid["angular_rule"] = diagnostics.grid_profile.angular_rule;
        grid["radial_rule"] = diagnostics.grid_profile.radial_rule;
        grid["radius_table"] = diagnostics.grid_profile.radius_table;
        grid["atom_scales"] = diagnostics.grid_profile.atom_scales;
        py::dict result;
        result["electron_count"] = diagnostics.electron_count;
        result["formal_electron_count"] = diagnostics.formal_electron_count;
        result["electron_count_absolute_error"] = diagnostics.electron_count_absolute_error;
        result["electron_count_relative_error"] = diagnostics.electron_count_relative_error;
        result["iterations"] = diagnostics.iterations;
        result["converged"] = diagnostics.converged;
        result["max_overlap_residual"] = diagnostics.max_overlap_residual;
        result["max_population_change"] = diagnostics.max_population_change;
        result["max_weight_change"] = diagnostics.max_weight_change;
        result["max_unity_residual"] = diagnostics.max_unity_residual;
        result["total_charge_residual"] = diagnostics.total_charge_residual;
        result["tail_fit_failures"] = diagnostics.tail_fit_failures;
        result["tail_failure_reused_profiles"] = diagnostics.tail_failure_reused_profiles;
        result["underflow_fallbacks"] = diagnostics.underflow_fallbacks;
        result["atomic_populations"] = diagnostics.atomic_populations;
        result["grid_profile"] = std::move(grid);
        result["radial_nodes"] = diagnostics.radial_nodes;
        result["log_profiles"] = diagnostics.log_profiles;
        result["tail_join_radii"] = diagnostics.tail_join_radii;
        result["tail_alphas"] = diagnostics.tail_alphas;
        result["context_digest"] = diagnostics.context_digest;
        return result;
    };
    const auto isa_result_dict = [isa_diagnostics_dict](std::size_t site_count,
                                                         const std::vector<double>& weights,
                                                         const ISADiagnostics& diagnostics) {
        py::dict result;
        result["site_count"] = site_count;
        result["weights"] = weights;
        result["diagnostics"] = isa_diagnostics_dict(diagnostics);
        return result;
    };
    // Underscored pure-math seams keep protocol tests on the native implementation
    // without expanding the supported public API.
    m.def("_atomic_polarizability_make_casimir_grid", [](unsigned int nonzero_count, double scale) {
        const auto grid = make_casimir_grid(nonzero_count, scale);
        return py::make_tuple(grid.frequencies, grid.weights);
    });
    m.def("_atomic_polarizability_test_irregular_harmonics",
          &detail::irregular_harmonics_test_only, "point"_a, "site"_a);
    m.def("_atomic_polarizability_plan_wsm_refinement",
          [](std::size_t point_count, std::size_t site_count,
             std::size_t active_variable_count, std::size_t constraint_rows,
             std::size_t memory_bytes) {
              const auto plan = detail::plan_wsm_refinement(
                  point_count, site_count, active_variable_count, constraint_rows, memory_bytes);
              py::dict values;
              values["point_count"] = plan.point_count;
              values["pair_rows"] = plan.pair_rows;
              values["site_count"] = plan.site_count;
              values["variable_count"] = plan.variable_count;
              values["active_variable_count"] = plan.active_variable_count;
              values["constraint_rows"] = plan.constraint_rows;
              values["irregular_elements"] = plan.irregular_elements;
              values["response_clone_bytes"] = plan.response_clone_bytes;
              values["design_elements"] = plan.design_elements;
              values["design_bytes"] = plan.design_bytes;
              values["constraint_matrix_bytes"] = plan.constraint_matrix_bytes;
              values["null_space_elements"] = plan.null_space_elements;
              values["null_space_bytes"] = plan.null_space_bytes;
              values["workspace_elements"] = plan.workspace_elements;
              values["workspace_bytes"] = plan.workspace_bytes;
              values["constraint_svd_peak_bytes"] = plan.constraint_svd_peak_bytes;
              values["fit_svd_peak_bytes"] = plan.fit_svd_peak_bytes;
              values["estimated_bytes"] = plan.estimated_bytes;
              values["configured_memory_bytes"] = plan.configured_memory_bytes;
              values["reserved_memory_bytes"] = plan.reserved_memory_bytes;
              values["algorithm"] = plan.algorithm;
              values["memory_semantics"] = plan.memory_semantics;
              return values;
          },
          "point_count"_a, "site_count"_a, "active_variable_count"_a,
          "constraint_rows"_a, "memory_bytes"_a);

    // ==> Symmetry-faithful WSM fit-point generation seams <== //
    const auto fit_point_options_from_dict = [](const py::dict& values) {
        FitPointOptions options;
        for (const auto& entry : values) {
            const auto key = py::cast<std::string>(entry.first);
            if (key == "spherical_points")
                options.spherical_points = entry.second.cast<std::size_t>();
            else if (key == "radial_shells")
                options.radial_shells = entry.second.cast<std::size_t>();
            else if (key == "inner_limit")
                options.inner_limit = entry.second.cast<double>();
            else if (key == "outer_limit")
                options.outer_limit = entry.second.cast<double>();
            else if (key == "maximum_points")
                options.maximum_points = entry.second.cast<std::size_t>();
            else if (key == "merge_tolerance_bohr")
                options.merge_tolerance_bohr = entry.second.cast<double>();
            else if (key == "radial_units") {
                const auto units = entry.second.cast<std::string>();
                if (units == "BOHR")
                    options.radial_units = FitPointRadialUnits::Bohr;
                else if (units == "VDW")
                    options.radial_units = FitPointRadialUnits::VanDerWaals;
                else
                    throw PSIEXCEPTION("WSM fit points: unsupported radial units '" + units + "'");
            } else
                throw PSIEXCEPTION("WSM fit points: unknown option '" + key + "'");
        }
        return options;
    };
    const auto fit_point_plan_dict = [](const FitPointPlan& plan) {
        py::dict values;
        values["atom_count"] = plan.atom_count;
        values["spherical_points"] = plan.spherical_points;
        values["radial_shells"] = plan.radial_shells;
        values["lebedev_order"] = plan.lebedev_order;
        values["symmetry_operation_count"] = plan.symmetry_operation_count;
        values["candidate_count"] = plan.candidate_count;
        values["point_count"] = plan.point_count;
        values["maximum_points"] = plan.maximum_points;
        values["candidate_bytes"] = plan.candidate_bytes;
        values["retained_metadata_bytes"] = plan.retained_metadata_bytes;
        values["estimated_bytes"] = plan.estimated_bytes;
        values["shell_offsets"] = plan.shell_offsets;
        values["radial_units"] = plan.radial_units;
        values["algorithm"] = plan.algorithm;
        return values;
    };
    const auto fit_point_set_dict = [fit_point_plan_dict](const FitPointSet& set) {
        auto points = std::make_shared<Matrix>("WSM fit points", static_cast<int>(set.points.size()), 3);
        for (std::size_t point = 0; point < set.points.size(); ++point)
            for (std::size_t axis = 0; axis < 3; ++axis)
                (*points)(point, axis) = set.points[point][axis];
        py::dict values;
        values["points"] = std::move(points);
        values["nearest_offsets"] = set.nearest_offsets;
        values["shell_index"] = set.shell_index;
        values["generator_atom"] = set.generator_atom;
        values["scaling_radii"] = set.scaling_radii;
        values["max_symmetry_deviation"] = set.max_symmetry_deviation;
        values["max_octahedral_deviation"] = set.max_octahedral_deviation;
        values["plan"] = fit_point_plan_dict(set.plan);
        return values;
    };
    m.def("_atomic_polarizability_lebedev_unit_sphere", [](int npoints) {
        const auto nodes = lebedev_spherical_grid(npoints);
        auto grid = std::make_shared<Matrix>("Lebedev unit sphere",
                                             static_cast<int>(nodes.size()), 4);
        for (std::size_t node = 0; node < nodes.size(); ++node) {
            (*grid)(node, 0) = nodes[node].x;
            (*grid)(node, 1) = nodes[node].y;
            (*grid)(node, 2) = nodes[node].z;
            (*grid)(node, 3) = nodes[node].w;
        }
        return grid;
    }, "npoints"_a);
    m.def("_atomic_polarizability_plan_fit_points",
          [fit_point_options_from_dict, fit_point_plan_dict](std::size_t atom_count,
                                                             const py::dict& option_values) {
              return fit_point_plan_dict(
                  plan_fit_points(atom_count, fit_point_options_from_dict(option_values)));
          },
          "atom_count"_a, "options"_a = py::dict());
    m.def("_atomic_polarizability_generate_fit_points",
          [fit_point_options_from_dict, fit_point_set_dict](
              const std::vector<int>& atomic_numbers, const Matrix& center_matrix,
              const std::vector<SharedMatrix>& symmetry_operations,
              const SharedMatrix& angular_frame, const py::dict& option_values) {
              if (center_matrix.nirrep() != 1 || center_matrix.ncol() != 3 ||
                  center_matrix.nrow() <= 0)
                  throw PSIEXCEPTION("WSM fit points: centers must be a nonempty N by 3 matrix");
              std::vector<SitePosition> centers(static_cast<std::size_t>(center_matrix.nrow()));
              for (std::size_t atom = 0; atom < centers.size(); ++atom)
                  for (std::size_t axis = 0; axis < 3; ++axis)
                      centers[atom][axis] = center_matrix(atom, axis);
              const auto to_operation = [](const SharedMatrix& matrix, const char* what) {
                  if (!matrix || matrix->nirrep() != 1 || matrix->nrow() != 3 || matrix->ncol() != 3)
                      throw PSIEXCEPTION(std::string("WSM fit points: ") + what +
                                         " must be a 3 by 3 matrix");
                  FitPointOperation operation{};
                  for (std::size_t row = 0; row < 3; ++row)
                      for (std::size_t column = 0; column < 3; ++column)
                          operation[3 * row + column] = (*matrix)(row, column);
                  return operation;
              };
              std::vector<FitPointOperation> operations;
              operations.reserve(symmetry_operations.size());
              for (const auto& matrix : symmetry_operations)
                  operations.push_back(to_operation(matrix, "each symmetry operation"));
              const auto frame = angular_frame ? to_operation(angular_frame, "the angular frame")
                                               : identity_fit_point_frame();
              return fit_point_set_dict(generate_fit_points(
                  atomic_numbers, centers, operations, frame,
                  fit_point_options_from_dict(option_values)));
          },
          "atomic_numbers"_a, "centers"_a, "symmetry_operations"_a,
          "angular_frame"_a = SharedMatrix(), "options"_a = py::dict());
    m.def("_atomic_polarizability_wsm_fit_points",
          [fit_point_set_dict](const std::shared_ptr<Molecule>& molecule) {
              if (!molecule) throw PSIEXCEPTION("WSM fit points: a molecule is required");
              return fit_point_set_dict(
                  generate_wsm_fit_points(*molecule, Process::environment.options));
          },
          "molecule"_a);
    m.def("_atomic_polarizability_bondi_vdw_radius", &bondi_vdw_radius_bohr, "atomic_number"_a);

    m.def("_atomic_polarizability_test_refine_wsm",
          [](const Matrix& point_matrix, const std::vector<double>& frequencies,
             const std::vector<SharedMatrix>& responses, const Matrix& site_matrix,
             const std::vector<SharedMatrix>& localized_matrices,
             const std::vector<double>& localized_frequencies,
             const std::vector<bool>& active_variables, const SharedMatrix& equality,
             const std::vector<double>& equality_targets, const py::dict& option_values) {
              if (point_matrix.nirrep() != 1 || point_matrix.ncol() != 3 || point_matrix.nrow() <= 0 ||
                  site_matrix.nirrep() != 1 || site_matrix.ncol() != 3 || site_matrix.nrow() <= 0)
                  throw PSIEXCEPTION("WSM refinement: points and sites must be nonempty N by 3 matrices");
              std::vector<SitePosition> points(static_cast<std::size_t>(point_matrix.nrow()));
              std::vector<SitePosition> sites(static_cast<std::size_t>(site_matrix.nrow()));
              for (std::size_t point = 0; point < points.size(); ++point)
                  for (std::size_t axis = 0; axis < 3; ++axis)
                      points[point][axis] = point_matrix(point, axis);
              for (std::size_t site = 0; site < sites.size(); ++site)
                  for (std::size_t axis = 0; axis < 3; ++axis)
                      sites[site][axis] = site_matrix(site, axis);
              if (localized_matrices.size() != frequencies.size() * sites.size())
                  throw PSIEXCEPTION("WSM refinement: expected one frequency-major localized 15 by 15 tensor per site");
              std::vector<std::vector<L3Matrix>> localized(
                  frequencies.size(), std::vector<L3Matrix>(sites.size()));
              for (std::size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
                  for (std::size_t site = 0; site < sites.size(); ++site) {
                      const auto& matrix = localized_matrices[frequency * sites.size() + site];
                      if (!matrix || matrix->nirrep() != 1 || matrix->nrow() != 15 || matrix->ncol() != 15)
                          throw PSIEXCEPTION("WSM refinement: expected frequency-major localized 15 by 15 tensors");
                      for (std::size_t row = 0; row < 15; ++row)
                          for (std::size_t column = 0; column < 15; ++column)
                              localized[frequency][site][row][column] = (*matrix)(row, column);
                  }
              }
              RefinementOptions options;
              for (const auto& entry : option_values) {
                  const auto key = py::cast<std::string>(entry.first);
                  if (key == "wsm_rank") options.wsm_rank = entry.second.cast<unsigned int>();
                  else if (key == "hydrogen_rank") options.hydrogen_rank = entry.second.cast<unsigned int>();
                  else if (key == "weight_type") options.weight_type = entry.second.cast<unsigned int>();
                  else if (key == "weight_coefficient") options.weight_coefficient = entry.second.cast<double>();
                  else if (key == "cutoff") options.cutoff = entry.second.cast<double>();
                  else if (key == "maximum_condition_number")
                      options.maximum_condition_number = entry.second.cast<double>();
                  else throw PSIEXCEPTION("WSM refinement: unknown policy option '" + key + "'");
              }
              PDefConstraints constraints{active_variables, equality, equality_targets};
              const auto models = detail::refine_wsm_test_only(
                  points, frequencies, responses, sites, localized, localized_frequencies,
                  constraints, options);
              py::list result;
              for (const auto& model : models) {
                  py::dict values;
                  values["frequency"] = model.frequency;
                  values["positions"] = model.positions;
                  py::list tensors;
                  for (const auto& tensor : model.tensors) {
                      auto matrix = std::make_shared<Matrix>(15, 15);
                      for (std::size_t row = 0; row < 15; ++row)
                          for (std::size_t column = 0; column < 15; ++column)
                              (*matrix)(row, column) = tensor[row][column];
                      tensors.append(std::move(matrix));
                  }
                  values["tensors"] = std::move(tensors);
                  values["solution"] = model.diagnostics.solution;
                  values["kept_variables"] = model.diagnostics.kept_variables;
                  values["pruned_variables"] = model.diagnostics.pruned_variables;
                  values["point_count"] = model.diagnostics.point_count;
                  values["pair_rows"] = model.diagnostics.pair_rows;
                  values["variable_count"] = model.diagnostics.variable_count;
                  values["active_variable_count"] = model.diagnostics.active_variable_count;
                  values["anchor_variable_count"] = model.diagnostics.anchor_variable_count;
                  values["condition_number"] = model.diagnostics.condition_number;
                  values["weighted_residual_norm"] = model.diagnostics.weighted_residual_norm;
                  values["anchor_residual_norm"] = model.diagnostics.anchor_residual_norm;
                  values["constraint_residual_norm"] = model.diagnostics.constraint_residual_norm;
                  values["objective_residual_norm"] = model.diagnostics.objective_residual_norm;
                  values["max_point_residual"] = model.diagnostics.max_point_residual;
                  values["max_output_asymmetry"] = model.diagnostics.max_output_asymmetry;
                  values["maximum_weighted_column_norm"] =
                      model.diagnostics.maximum_weighted_column_norm;
                  values["applied_column_cutoff"] = model.diagnostics.applied_column_cutoff;
                  values["row_weight_source"] = model.diagnostics.row_weight_source;
                  py::dict policy;
                  policy["wsm_rank"] = options.wsm_rank;
                  policy["hydrogen_rank"] = options.hydrogen_rank;
                  policy["weight_type"] = options.weight_type;
                  policy["weight_coefficient"] = options.weight_coefficient;
                  policy["cutoff"] = options.cutoff;
                  policy["weight_type_definition"] =
                      "inherited protocol: anchor the site-local rank-1 dipole block to LocalizedResponse.local";
                  policy["external_oracle_parity_claimed"] = false;
                  values["policy"] = std::move(policy);
                  result.append(std::move(values));
              }
              return result;
          },
          "points"_a, "frequencies"_a, "responses"_a, "sites"_a,
          "localized"_a, "localized_frequencies"_a, "active_variables"_a, "equality"_a,
          "equality_targets"_a, "options"_a = py::dict());
    m.def("_atomic_polarizability_test_constrained_least_squares",
          [](const Matrix& design, const std::vector<double>& observations,
             const std::vector<double>& row_weights, double lambda,
             const std::vector<double>& diagonal_anchor, const std::vector<double>& reference,
             const Matrix& constraints, const std::vector<double>& constraint_targets,
             const py::dict& option_values) {
              detail::ConstrainedLeastSquaresOptions options;
              for (const auto& entry : option_values) {
                  const auto key = py::cast<std::string>(entry.first);
                  if (key == "column_cutoff")
                      options.column_cutoff = entry.second.cast<double>();
                  else if (key == "prune_below_cutoff")
                      options.prune_below_cutoff = entry.second.cast<bool>();
                  else if (key == "maximum_condition_number")
                      options.maximum_condition_number = entry.second.cast<double>();
                  else if (key == "rank_tolerance")
                      options.rank_tolerance = entry.second.cast<double>();
                  else if (key == "maximum_workspace_elements")
                      options.maximum_workspace_elements = entry.second.cast<std::size_t>();
                  else
                      throw PSIEXCEPTION("constrained least squares: unknown option '" + key + "'");
              }
              const auto result = detail::solve_constrained_least_squares(
                  design, observations, row_weights, lambda, diagonal_anchor, reference,
                  constraints, constraint_targets, options);
              py::dict input_metadata;
              input_metadata["lambda"] = result.lambda;
              input_metadata["row_weight_min"] = result.row_weight_min;
              input_metadata["row_weight_max"] = result.row_weight_max;
              input_metadata["row_weight_source"] = result.row_weight_source;
              py::dict allocation_plan;
              allocation_plan["constraint_rows"] = result.allocation_plan.constraint_rows;
              allocation_plan["constraint_columns"] = result.allocation_plan.constraint_columns;
              allocation_plan["constraint_u_elements"] = result.allocation_plan.constraint_u_elements;
              allocation_plan["constraint_vt_elements"] = result.allocation_plan.constraint_vt_elements;
              allocation_plan["fit_rows"] = result.allocation_plan.fit_rows;
              allocation_plan["fit_columns"] = result.allocation_plan.fit_columns;
              allocation_plan["fit_u_elements"] = result.allocation_plan.fit_u_elements;
              allocation_plan["fit_vt_elements"] = result.allocation_plan.fit_vt_elements;
              py::dict values;
              values["solution"] = result.solution;
              values["kept_columns"] = result.kept_columns;
              values["pruned_columns"] = result.pruned_columns;
              values["full_to_reduced"] = result.full_to_reduced;
              values["column_weighted_norms"] = result.column_weighted_norms;
              values["singular_values"] = result.singular_values;
              values["rank"] = result.rank;
              values["constraint_rank"] = result.constraint_rank;
              values["free_dimension"] = result.free_dimension;
              values["condition_number"] = result.condition_number;
              values["weighted_residual_norm"] = result.weighted_residual_norm;
              values["anchor_residual_norm"] = result.anchor_residual_norm;
              values["constraint_residual_norm"] = result.constraint_residual_norm;
              values["objective_residual_norm"] = result.objective_residual_norm;
              values["input_metadata"] = std::move(input_metadata);
              values["allocation_plan"] = std::move(allocation_plan);
              return values;
          },
          "design"_a, "observations"_a, "row_weights"_a, "lambda"_a,
          "diagonal_anchor"_a, "reference"_a, "constraints"_a,
          "constraint_targets"_a, "options"_a = py::dict());
    m.def("_atomic_polarizability_solve_restricted_response",
          [](const Matrix& H1, const Matrix& H2, double omega, const Matrix& rhs) {
              const auto result = detail::solve_dense_restricted_response(H1, H2, omega, rhs);
              py::dict values;
              values["P"] = result.P_clone();
              values["Q"] = result.Q_clone();
              values["reciprocal_condition"] = result.reciprocal_condition();
              values["reciprocal_pivot_growth"] = result.reciprocal_pivot_growth();
              values["forward_error"] = result.forward_error();
              values["backward_error"] = result.backward_error();
              values["scaled_residual"] = result.scaled_residual();
              values["solution_column_scales"] = result.solution_column_scales();
              values["max_forward_error"] = *std::max_element(
                  result.forward_error().begin(), result.forward_error().end());
              values["max_backward_error"] = *std::max_element(
                  result.backward_error().begin(), result.backward_error().end());
              values["max_scaled_residual"] = *std::max_element(
                  result.scaled_residual().begin(), result.scaled_residual().end());
              return values;
          },
          "H1"_a, "H2"_a, "omega"_a, "rhs"_a);
    const auto point_response_plan_dict = [](const PointResponsePlan& plan) {
        py::dict values;
        values["frequency_count"] = plan.frequency_count;
        values["nbf"] = plan.nbf;
        values["nocc"] = plan.nocc;
        values["nvir"] = plan.nvir;
        values["transition_count"] = plan.transition_count;
        values["point_count"] = plan.point_count;
        values["max_point_count"] = plan.max_point_count;
        values["max_frequency_count"] = plan.max_frequency_count;
        values["configured_memory_bytes"] = plan.configured_memory_bytes;
        values["reserved_memory_bytes"] = plan.reserved_memory_bytes;
        values["ao_matrix_bytes"] = plan.ao_matrix_bytes;
        values["transition_potential_bytes"] = plan.transition_potential_bytes;
        values["output_bytes"] = plan.output_bytes;
        values["output_clone_bytes"] = plan.output_clone_bytes;
        values["retained_frequency_bytes"] = plan.retained_frequency_bytes;
        values["retained_points_bytes"] = plan.retained_points_bytes;
        values["native_diagnostic_record_bytes"] = plan.native_diagnostic_record_bytes;
        values["native_diagnostics_bytes"] = plan.native_diagnostics_bytes;
        values["container_overhead_bytes"] = plan.container_overhead_bytes;
        values["retained_metadata_bytes"] = plan.retained_metadata_bytes;
        values["python_scalar_diagnostic_overhead_bytes"] =
            plan.python_scalar_diagnostic_overhead_bytes;
        values["python_metadata_overhead_bytes"] = plan.python_metadata_overhead_bytes;
        values["python_export_overhead_bytes"] = plan.python_export_overhead_bytes;
        values["dense_solve_peak_bytes"] = plan.dense_solve_peak_bytes;
        values["scratch_bytes"] = plan.scratch_bytes;
        values["c1_plan_estimated_bytes"] = plan.c1_plan_estimated_bytes;
        values["alda_plan_estimated_bytes"] = plan.alda_plan_estimated_bytes;
        values["retained_c1_bytes"] = plan.retained_c1_bytes;
        values["retained_alda_bytes"] = plan.retained_alda_bytes;
        values["hessian_bytes"] = plan.hessian_bytes;
        values["transition_metadata_bytes"] = plan.transition_metadata_bytes;
        values["conservative_overhead_bytes"] = plan.conservative_overhead_bytes;
        values["c1_stage_peak_bytes"] = plan.c1_stage_peak_bytes;
        values["alda_stage_peak_bytes"] = plan.alda_stage_peak_bytes;
        values["point_potential_stage_peak_bytes"] = plan.point_potential_stage_peak_bytes;
        values["dense_solve_stage_peak_bytes"] = plan.dense_solve_stage_peak_bytes;
        values["output_clone_stage_peak_bytes"] = plan.output_clone_stage_peak_bytes;
        values["estimated_bytes"] = plan.estimated_bytes;
        values["integral_work_terms"] = plan.integral_work_terms;
        values["algorithm"] = plan.algorithm;
        values["memory_semantics"] = plan.memory_semantics;
        return values;
    };
    m.def("_atomic_polarizability_estimate_point_response",
          [point_response_plan_dict](std::size_t frequency_count, std::size_t nbf,
                                     std::size_t nocc, std::size_t nvir,
                                     std::size_t point_count, bool has_dynamic_frequency,
                                     std::size_t memory_bytes) {
              return point_response_plan_dict(detail::plan_point_response(
                  frequency_count, nbf, nocc, nvir, point_count,
                  has_dynamic_frequency, memory_bytes));
          },
          "frequency_count"_a, "nbf"_a, "nocc"_a, "nvir"_a,
          "point_count"_a, "has_dynamic_frequency"_a, "memory_bytes"_a);
    const auto point_response_minimum_distance = [](const py::dict& options) {
        const std::vector<std::string> allowed{"minimum_site_distance_bohr"};
        for (const auto& item : options) {
            const auto key = py::cast<std::string>(item.first);
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                throw PSIEXCEPTION("point response: unknown option " + key);
        }
        return options.contains("minimum_site_distance_bohr")
            ? py::cast<double>(options["minimum_site_distance_bohr"])
            : 0.0;
    };
    const auto point_response_result_dict =
        [point_response_plan_dict](const PointResponseData& data,
                                   const std::string& operator_provenance) {
            py::dict result;
            result["points"] = data.points();
            result["frequencies"] = data.frequencies();
            result["responses"] = data.response_clones();
            const auto stored = data.transition_potentials_clone_test_only();
            auto point_major = std::make_shared<Matrix>(stored->ncol(), stored->nrow());
            for (int point = 0; point < stored->ncol(); ++point)
                for (int transition = 0; transition < stored->nrow(); ++transition)
                    (*point_major)(point, transition) = (*stored)(transition, point);
            result["transition_potentials"] = std::move(point_major);
            result["potential_convention"] =
                "native electronic AO multipole-potential sign; the sign cancels in the bilinear response";
            result["transition_order"] = "(i,a) occupied-major/virtual-minor";
            result["frequency_order"] = "frequency-major";
            result["operator_provenance"] = operator_provenance;
            py::list diagnostics;
            for (const auto& diagnostic : data.diagnostics()) {
                py::dict values;
                values["frequency"] = diagnostic.frequency;
                values["reciprocal_condition"] = diagnostic.reciprocal_condition;
                values["reciprocal_pivot_growth"] = diagnostic.reciprocal_pivot_growth;
                values["max_forward_error"] = diagnostic.max_forward_error;
                values["max_backward_error"] = diagnostic.max_backward_error;
                values["max_scaled_residual"] = diagnostic.max_scaled_residual;
                values["max_solution_scale"] = diagnostic.max_solution_scale;
                values["allowed_antisymmetry"] = diagnostic.allowed_antisymmetry;
                values["symmetry_residual"] = diagnostic.symmetry_residual;
                values["max_normalized_antisymmetry"] =
                    diagnostic.max_normalized_antisymmetry;
                values["symmetry_policy"] =
                    "AVERAGE_WITHIN_SOLVER_DERIVED_CONTRACTION_ERROR_BOUND";
                values["reciprocity_enforced"] = diagnostic.reciprocity_enforced;
                diagnostics.append(std::move(values));
            }
            result["diagnostics"] = std::move(diagnostics);
            result["plan"] = point_response_plan_dict(data.plan());
            return result;
        };
    m.def("_atomic_polarizability_evaluate_point_response",
          [point_response_minimum_distance, point_response_result_dict](
              const std::shared_ptr<FrozenResponseContext>& context,
              double chf_exchange, double alda_kernel,
              const std::vector<SitePosition>& points,
              const std::vector<double>& frequencies, const py::dict& options) {
              const ResponseKernel kernel(chf_exchange, alda_kernel);
              const auto data = evaluate_point_response(
                  context, kernel, frequencies, points,
                  point_response_minimum_distance(options));
              return point_response_result_dict(
                  data, "CANONICAL_C1_PLUS_FULL_ALDA");
          },
          "context"_a, "chf_exchange"_a, "alda_kernel"_a, "points"_a,
          "frequencies"_a, "options"_a = py::dict());
    m.def("_test_only_raw_point_response",
          [point_response_minimum_distance, point_response_result_dict](
              const std::shared_ptr<FrozenResponseContext>& context,
              const std::vector<SitePosition>& points,
              const std::vector<double>& frequencies, const Matrix& H1,
              const Matrix& H2,
              const std::vector<std::size_t>& transition_permutation,
              const py::dict& options) {
              const auto data = detail::evaluate_raw_point_response_test_only(
                  context, H1, H2, frequencies, points, transition_permutation,
                  point_response_minimum_distance(options));
              return point_response_result_dict(
                  data, "TEST_ONLY_UNPROVENANCED_RAW_H1_H2");
          },
          "context"_a, "points"_a, "frequencies"_a, "H1"_a, "H2"_a,
          "transition_permutation"_a = std::vector<std::size_t>{},
          "options"_a = py::dict());
    m.def("_atomic_polarizability_assemble_restricted_hessian",
          [](const std::vector<double>& orbital_gaps, const Matrix& coulomb,
             const Matrix& exchange_direct, const Matrix& exchange_transpose,
             const Matrix& full_alda, double chf_exchange, double alda_coefficient) {
              const ResponseKernel kernel(chf_exchange, alda_coefficient);
              const auto result = detail::assemble_restricted_singlet_hessian(
                  orbital_gaps, coulomb, exchange_direct, exchange_transpose, full_alda, kernel);
              py::dict values;
              values["H1"] = result.H1;
              values["H2"] = result.H2;
              return values;
          },
          "orbital_gaps"_a, "coulomb"_a, "exchange_direct"_a,
          "exchange_transpose"_a, "full_alda"_a, "chf_exchange"_a,
          "alda_coefficient"_a);
    m.def("_atomic_polarizability_validate_response_diagnostics",
          [](double reciprocal_condition, double reciprocal_pivot_growth,
             const std::vector<double>& forward_error, const std::vector<double>& backward_error,
             const std::vector<double>& scaled_residual) {
              detail::validate_dense_response_diagnostics(
                  reciprocal_condition, reciprocal_pivot_growth, forward_error, backward_error,
                  scaled_residual);
          },
          "reciprocal_condition"_a, "reciprocal_pivot_growth"_a, "forward_error"_a,
          "backward_error"_a, "scaled_residual"_a);
    m.def("_atomic_polarizability_validate_response_kernel", [](double chf_exchange, double alda_kernel) {
        const ResponseKernel kernel(chf_exchange, alda_kernel);
        return py::make_tuple(kernel.chf_exchange(), kernel.alda_kernel());
    });
    m.def("_atomic_polarizability_validate_vertical_protocol",
          [](bool cation_state_valid, bool complete_basis_valid) {
              detail::validate_vertical_protocol(cation_state_valid, complete_basis_valid);
          },
          "cation_state_valid"_a, "complete_basis_valid"_a);
    m.def("_atomic_polarizability_test_isa",
          [isa_options_from_dict, isa_result_dict](
              const std::vector<SitePosition>& sites, const std::vector<SitePosition>& points,
              const std::vector<double>& weights, const std::vector<int>& atomic_numbers,
              const std::vector<std::array<double, 5>>& gaussian_terms, const py::dict& option_values) {
              std::vector<detail::SyntheticGaussianDensity> terms;
              terms.reserve(gaussian_terms.size());
              for (const auto& term : gaussian_terms)
                  terms.push_back({{term[0], term[1], term[2]}, term[3], term[4]});
              py::dict isa_option_values;
              for (const auto& entry : option_values) isa_option_values[entry.first] = entry.second;
              std::size_t inject_tail_fit_failure_iteration = 0;
              std::size_t test_min_iterations = 0;
              if (isa_option_values.contains("inject_tail_fit_failure_iteration")) {
                  inject_tail_fit_failure_iteration =
                      isa_option_values["inject_tail_fit_failure_iteration"].cast<std::size_t>();
                  isa_option_values.attr("pop")("inject_tail_fit_failure_iteration");
              }
              if (isa_option_values.contains("test_min_iterations")) {
                  test_min_iterations = isa_option_values["test_min_iterations"].cast<std::size_t>();
                  isa_option_values.attr("pop")("test_min_iterations");
              }
              const auto result = detail::compute_synthetic_isa(
                  sites, points, weights, atomic_numbers, terms,
                  isa_options_from_dict(isa_option_values), inject_tail_fit_failure_iteration,
                  test_min_iterations);
              return isa_result_dict(result.site_count, result.weights, result.diagnostics);
          },
          "sites"_a, "points"_a, "weights"_a, "atomic_numbers"_a,
          "gaussian_terms"_a, "options"_a = py::dict());
    m.def("_atomic_polarizability_test_isa_profile",
          [](const std::vector<double>& nodes, const std::vector<double>& log_values,
             const std::vector<double>& queries, double tail_join, double tail_charge) {
              const auto result = detail::test_isa_profile(nodes, log_values, queries, tail_join, tail_charge);
              py::dict values;
              values["log_values"] = result.log_values;
              values["tail_alpha"] = result.tail_alpha;
              values["tail_log_amplitude"] = result.tail_log_amplitude;
              values["tail_charge"] = result.tail_charge;
              values["join_log_left"] = result.join_log_left;
              values["join_log_right"] = result.join_log_right;
              return values;
          });
    m.def("_atomic_polarizability_test_isa_gaussian_fixed_point",
          [](const std::vector<SitePosition>& sites, const std::vector<SitePosition>& points,
             const std::vector<std::array<double, 5>>& gaussian_terms, std::size_t radial_points,
             std::size_t angular_polar_points, std::size_t angular_azimuthal_points) {
              std::vector<detail::SyntheticGaussianDensity> terms;
              terms.reserve(gaussian_terms.size());
              for (const auto& term : gaussian_terms)
                  terms.push_back({{term[0], term[1], term[2]}, term[3], term[4]});
              const auto result = detail::test_isa_gaussian_fixed_point(
                  sites, points, terms, radial_points, angular_polar_points, angular_azimuthal_points);
              py::dict values;
              values["weights"] = result.weights;
              values["max_profile_relative_error"] = result.max_profile_relative_error;
              return values;
          });
    m.def("_atomic_polarizability_test_isa_overlap",
          [](const std::vector<double>& first_nodes, const std::vector<double>& first_logs,
             double first_tail_alpha, double first_tail_log_amplitude,
             const std::vector<double>& second_nodes, const std::vector<double>& second_logs,
             double second_tail_alpha, double second_tail_log_amplitude,
             double tail_join, std::size_t integration_points) {
              const auto result = detail::test_isa_overlap(
                  first_nodes, first_logs, first_tail_alpha, first_tail_log_amplitude,
                  second_nodes, second_logs, second_tail_alpha, second_tail_log_amplitude,
                  tail_join, integration_points);
              py::dict values;
              values["overlap_residual"] = result.overlap_residual;
              return values;
          });
    m.def("_atomic_polarizability_test_isa_tail_probabilities", &detail::test_isa_tail_probabilities);
    py::class_<FrozenResponseContext, std::shared_ptr<FrozenResponseContext>>(
        m, "_AtomicPolarizabilityFrozenResponseContext")
        .def("summary", [](const FrozenResponseContext& context) {
            py::dict result;
            const auto& grac = context.grac();
            const auto& functional = context.functional();
            result["reference"] = "RKS";
            result["functional"] = context.functional_name();
            result["needs_grac"] = functional->needs_grac();
            result["applied_shift"] = grac.applied_shift;
            result["derived_shift"] = grac.ionization_potential + grac.homo_energy;
            result["grac_x_functional"] = context.grac_x_name();
            result["grac_c_functional"] = context.grac_c_name();
            const auto grac_x = std::dynamic_pointer_cast<const LibXCFunctional>(functional->grac_x_functional());
            const auto grac_c = std::dynamic_pointer_cast<const LibXCFunctional>(functional->grac_c_functional());
            result["grac_x_parameters"] = grac_x->effective_parameter_map();
            result["grac_c_parameters"] = grac_c->effective_parameter_map();
            result["grac_alpha"] = functional->grac_alpha();
            result["grac_beta"] = functional->grac_beta();
            result["cation_reference"] = grac.cation_reference;
            result["neutral_precursor_energy"] = grac.neutral_precursor_energy;
            result["cation_energy"] = grac.cation_energy;
            result["site_count"] = context.sites().size();
            result["grid_point_count"] = context.grid_point_count();
            result["functional_density_tolerance"] = context.functional_density_tolerance();
            result["single_thread_no_basis_mutation_contract"] = true;
            result["basis_detached"] = false;
            return result;
        })
        .def("grid_snapshot", [](const FrozenResponseContext& context) {
            py::list blocks;
            for (const auto& block : context.grid_blocks())
                blocks.append(py::make_tuple(block.point_offset, block.point_count,
                                             block.functions_local_to_global));
            return py::make_tuple(context.grid_points(), context.grid_weights(), blocks);
        })
        .def("state_checksum", [](const FrozenResponseContext& context) {
            const auto matrix_sum_squares = [](const std::shared_ptr<const Matrix>& matrix) {
                double result = 0.0;
                for (int h = 0; h < matrix->nirrep(); ++h)
                    for (int row = 0; row < matrix->rowspi()[h]; ++row)
                        for (int column = 0; column < matrix->colspi()[h]; ++column) {
                            const double value = matrix->get(h, row, column);
                            result += value * value;
                        }
                return result;
            };
            const auto vector_sum_squares = [](const std::shared_ptr<const Vector>& vector) {
                double result = 0.0;
                for (int h = 0; h < vector->nirrep(); ++h)
                    for (int element = 0; element < vector->dim(h); ++element) {
                        const double value = vector->get(h, element);
                        result += value * value;
                    }
                return result;
            };
            return std::vector<double>{matrix_sum_squares(context.Ca()), matrix_sum_squares(context.Cb()),
                                       vector_sum_squares(context.epsilon_a()), vector_sum_squares(context.epsilon_b()),
                                       vector_sum_squares(context.occupation_a()), vector_sum_squares(context.occupation_b()),
                                       matrix_sum_squares(context.Da()), matrix_sum_squares(context.Db()), context.energy(),
                                       context.functional_density_tolerance()};
        });
    m.def("_atomic_polarizability_test_restricted_c1_primitives",
          [](const std::shared_ptr<FrozenResponseContext>& context, const py::dict& overrides) {
              if (!context) throw PSIEXCEPTION("restricted C1 transition primitives: frozen response context is null");
              const std::vector<std::string> allowed{
                  "orbital_order", "epsilon_a", "epsilon_b", "occupation_a", "occupation_b",
                  "beta_orbital_delta"};
              for (const auto& item : overrides) {
                  const auto key = py::cast<std::string>(item.first);
                  if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                      throw PSIEXCEPTION("restricted C1 transition primitives: unknown test override " + key);
              }
              auto Ca = context->Ca()->clone();
              auto Cb = context->Cb()->clone();
              auto epsilon_a = std::make_shared<Vector>(context->epsilon_a()->clone());
              auto epsilon_b = std::make_shared<Vector>(context->epsilon_b()->clone());
              auto occupation_a = std::make_shared<Vector>(context->occupation_a()->clone());
              auto occupation_b = std::make_shared<Vector>(context->occupation_b()->clone());
              const int nbf = Ca->nrow();
              const int nmo = Ca->ncol();
              if (overrides.contains("orbital_order")) {
                  const auto order = py::cast<std::vector<int>>(overrides["orbital_order"]);
                  std::vector<bool> seen(nmo, false);
                  if (order.size() != static_cast<std::size_t>(nmo))
                      throw PSIEXCEPTION("restricted C1 transition primitives: orbital permutation has wrong dimension");
                  for (int source : order) {
                      if (source < 0 || source >= nmo || seen[source])
                          throw PSIEXCEPTION("restricted C1 transition primitives: orbital order must be a permutation");
                      seen[source] = true;
                  }
                  auto permuted_Ca = std::make_shared<Matrix>(nbf, nmo);
                  auto permuted_Cb = std::make_shared<Matrix>(nbf, nmo);
                  auto permuted_epsilon_a = std::make_shared<Vector>("permuted epsilon a", nmo);
                  auto permuted_epsilon_b = std::make_shared<Vector>("permuted epsilon b", nmo);
                  auto permuted_occupation_a = std::make_shared<Vector>("permuted occupation a", nmo);
                  auto permuted_occupation_b = std::make_shared<Vector>("permuted occupation b", nmo);
                  for (int target = 0; target < nmo; ++target) {
                      const int source = order[target];
                      for (int mu = 0; mu < nbf; ++mu) {
                          (*permuted_Ca)(mu, target) = (*Ca)(mu, source);
                          (*permuted_Cb)(mu, target) = (*Cb)(mu, source);
                      }
                      permuted_epsilon_a->set(0, target, epsilon_a->get(0, source));
                      permuted_epsilon_b->set(0, target, epsilon_b->get(0, source));
                      permuted_occupation_a->set(0, target, occupation_a->get(0, source));
                      permuted_occupation_b->set(0, target, occupation_b->get(0, source));
                  }
                  Ca = std::move(permuted_Ca);
                  Cb = std::move(permuted_Cb);
                  epsilon_a = std::move(permuted_epsilon_a);
                  epsilon_b = std::move(permuted_epsilon_b);
                  occupation_a = std::move(permuted_occupation_a);
                  occupation_b = std::move(permuted_occupation_b);
              }
              const auto replace_vector = [nmo, &overrides](const char* key,
                                                             const std::shared_ptr<Vector>& target) {
                  if (!overrides.contains(key)) return;
                  const auto values = py::cast<std::vector<double>>(overrides[key]);
                  if (values.size() != static_cast<std::size_t>(nmo))
                      throw PSIEXCEPTION(std::string("restricted C1 transition primitives: ") + key +
                                         " override has wrong dimension");
                  for (int orbital = 0; orbital < nmo; ++orbital)
                      target->set(0, orbital, values[orbital]);
              };
              replace_vector("epsilon_a", epsilon_a);
              replace_vector("epsilon_b", epsilon_b);
              replace_vector("occupation_a", occupation_a);
              replace_vector("occupation_b", occupation_b);
              if (overrides.contains("beta_orbital_delta"))
                  (*Cb)(0, 0) += py::cast<double>(overrides["beta_orbital_delta"]);

              const auto primitives = overrides.empty()
                  ? detail::construct_restricted_c1_primitives(context)
                  : detail::construct_restricted_c1_primitives_test_only(
                        context, *Ca, *Cb, *epsilon_a, *epsilon_b, *occupation_a, *occupation_b);
              auto zero_alda = std::make_shared<Matrix>(primitives.orbital_gaps.size(),
                                                        primitives.orbital_gaps.size());
              const auto hessian = detail::assemble_restricted_singlet_hessian(
                  primitives.orbital_gaps, *primitives.coulomb, *primitives.exchange_direct,
                  *primitives.exchange_transpose, *zero_alda, ResponseKernel(0.25, 0.75));
              py::dict result;
              result["transition_order"] = "(i,a) occupied-major/virtual-minor";
              result["algorithm"] = primitives.jk_plan.algorithm;
              result["batch_size"] = primitives.jk_plan.batch_size;
              result["jk_threads"] = primitives.jk_plan.jk_threads;
              result["integral_engine_thread_count"] = primitives.integral_engine_thread_count;
              result["screening"] = primitives.jk_plan.screening;
              result["integral_cutoff"] = primitives.jk_plan.integral_cutoff;
              result["incfock"] = primitives.jk_plan.incfock;
              result["estimated_bytes"] = primitives.jk_plan.estimated_bytes;
              result["nbf"] = primitives.jk_plan.nbf;
              result["nocc"] = primitives.jk_plan.nocc;
              result["nvir"] = primitives.jk_plan.nvir;
              result["transitions"] = primitives.transitions;
              result["orbital_gaps"] = primitives.orbital_gaps;
              result["coulomb"] = primitives.coulomb;
              result["exchange_direct"] = primitives.exchange_direct;
              result["exchange_transpose"] = primitives.exchange_transpose;
              result["H1_zero_alda"] = hessian.H1;
              result["H2_zero_alda"] = hessian.H2;
              return result;
          },
          "context"_a, "test_overrides"_a = py::dict());
    const auto alda_plan_dict = [](const detail::RestrictedALDAPlan& plan) {
        py::dict values;
        values["algorithm"] = plan.algorithm;
        values["memory_semantics"] = plan.memory_semantics;
        values["nbf"] = plan.nbf;
        values["nocc"] = plan.nocc;
        values["nvir"] = plan.nvir;
        values["nov"] = plan.nov;
        values["point_count"] = plan.point_count;
        values["max_block_points"] = plan.max_block_points;
        values["max_supported_nov"] = plan.max_supported_nov;
        values["configured_memory_bytes"] = plan.configured_memory_bytes;
        values["reserved_memory_bytes"] = plan.reserved_memory_bytes;
        values["retained_payload_bytes"] = plan.retained_payload_bytes;
        values["block_transition_bytes"] = plan.block_transition_bytes;
        values["block_weighted_transition_bytes"] = plan.block_weighted_transition_bytes;
        values["block_mo_scratch_bytes"] = plan.block_mo_scratch_bytes;
        values["collocation_bytes"] = plan.collocation_bytes;
        values["block_coordinate_weight_bytes"] = plan.block_coordinate_weight_bytes;
        values["block_density_kernel_bytes"] = plan.block_density_kernel_bytes;
        values["functional_workspace_bytes"] = plan.functional_workspace_bytes;
        values["point_scratch_bytes"] = plan.point_scratch_bytes;
        values["metadata_bytes"] = plan.metadata_bytes;
        values["validation_scratch_bytes"] = plan.validation_scratch_bytes;
        values["conservative_overhead_bytes"] = plan.conservative_overhead_bytes;
        values["diagnostics_payload_bytes"] = plan.diagnostics_payload_bytes;
        values["estimated_bytes"] = plan.estimated_bytes;
        values["density_work_terms"] = plan.density_work_terms;
        values["mo_transition_work_terms"] = plan.mo_transition_work_terms;
        values["ao_collocation_work_terms"] = plan.ao_collocation_work_terms;
        values["libxc_work_terms"] = plan.libxc_work_terms;
        values["dgemm_work_terms"] = plan.dgemm_work_terms;
        values["work_terms"] = plan.work_terms;
        values["max_work_terms"] = plan.max_work_terms;
        values["density_cutoff"] = plan.density_cutoff;
        values["density_cutoff_source"] = plan.density_cutoff_source;
        values["retain_test_diagnostics"] = plan.retain_test_diagnostics;
        return values;
    };
    const auto alda_diagnostics_dict = [alda_plan_dict](const detail::RestrictedALDADiagnostics& diagnostics) {
        py::dict values;
        values["exchange_component"] = diagnostics.exchange_component;
        values["correlation_component"] = diagnostics.correlation_component;
        values["exchange_libxc_id"] = diagnostics.exchange_libxc_id;
        values["correlation_libxc_id"] = diagnostics.correlation_libxc_id;
        values["exchange_libxc_canonical_name"] = diagnostics.exchange_libxc_canonical_name;
        values["correlation_libxc_canonical_name"] = diagnostics.correlation_libxc_canonical_name;
        values["exchange_effective_parameters"] = diagnostics.exchange_effective_parameters;
        values["correlation_effective_parameters"] = diagnostics.correlation_effective_parameters;
        values["exchange_coefficient"] = diagnostics.exchange_coefficient;
        values["correlation_coefficient"] = diagnostics.correlation_coefficient;
        values["derivative_order"] = diagnostics.derivative_order;
        values["density_cutoff"] = diagnostics.density_cutoff;
        values["density_cutoff_source"] = diagnostics.density_cutoff_source;
        values["point_count"] = diagnostics.point_count;
        values["restricted_normalization"] = diagnostics.restricted_normalization;
        values["plan"] = alda_plan_dict(diagnostics.plan);
        return values;
    };
    m.def("_atomic_polarizability_test_contract_restricted_alda",
          [](const std::vector<double>& weights, const Matrix& transition_values,
             const std::vector<double>& densities, const std::vector<double>& fxc,
             double density_cutoff) {
              return detail::contract_restricted_alda_test_only(
                  weights, transition_values, densities, fxc, density_cutoff);
          }, "weights"_a, "transition_values"_a, "densities"_a, "fxc"_a,
          "density_cutoff"_a);
    m.def("_atomic_polarizability_test_validate_restricted_alda_grid",
          [](std::size_t nbf, std::size_t point_count, const std::vector<double>& weights,
             const std::vector<std::size_t>& offsets, const std::vector<std::size_t>& counts,
             const std::vector<std::vector<int>>& maps) {
              if (offsets.size() != counts.size() || offsets.size() != maps.size())
                  throw PSIEXCEPTION("restricted ALDA kernel: malformed sealed block metadata");
              std::vector<FrozenGridBlock> blocks;
              for (std::size_t block = 0; block < offsets.size(); ++block)
                  blocks.push_back({offsets[block], counts[block], maps[block]});
              detail::validate_restricted_alda_grid_test_only(nbf, point_count, weights, blocks);
          }, "nbf"_a, "point_count"_a, "weights"_a, "offsets"_a, "counts"_a, "maps"_a);
    m.def("_atomic_polarizability_test_restricted_alda_fxc",
          [alda_diagnostics_dict](const std::vector<double>& densities, bool include_correlation,
                                  double density_cutoff) {
              const auto result = detail::evaluate_restricted_alda_fxc_test_only(
                  densities, include_correlation, density_cutoff);
              py::dict values;
              values["fxc"] = result.first;
              values["diagnostics"] = alda_diagnostics_dict(result.second);
              return values;
          }, "densities"_a, "include_correlation"_a, "density_cutoff"_a);
    m.def("_atomic_polarizability_test_restricted_alda_kernel",
          [alda_diagnostics_dict](const std::shared_ptr<FrozenResponseContext>& context,
                                  bool retain_test_diagnostics) {
              const auto result = detail::construct_restricted_alda_kernel(context, retain_test_diagnostics);
              py::dict values;
              values["transition_order"] = "(i,a) occupied-major/virtual-minor";
              values["transitions"] = result.transitions;
              values["full_alda"] = result.full_alda;
              values["densities"] = result.densities;
              values["fxc"] = result.fxc;
              values["transition_values"] = result.transition_values;
              values["diagnostics"] = alda_diagnostics_dict(result.diagnostics);
              return values;
          }, "context"_a, "retain_test_diagnostics"_a = false);
    m.def("_atomic_polarizability_test_restricted_alda_ao_collocation_target",
          [](const std::shared_ptr<FrozenResponseContext>& context) {
              const auto result = detail::collocate_restricted_alda_ao_target_test_only(context);
              py::dict values;
              values["point_count"] = result.point_count;
              values["nbf"] = result.nbf;
              values["ao_values"] = result.ao_values;
              return values;
          }, "context"_a);
    m.def("_atomic_polarizability_test_validate_restricted_alda_work_bound",
          &detail::validate_restricted_alda_work_bound_test_only, "work_terms"_a);
    const auto projection_result_dict = [](const detail::TransitionMultipoleProjection& projection) {
        py::dict plan;
        plan["algorithm"] = projection.plan.algorithm;
        plan["point_count"] = projection.plan.point_count;
        plan["site_count"] = projection.plan.site_count;
        plan["transition_count"] = projection.plan.transition_count;
        plan["max_block_points"] = projection.plan.max_block_points;
        plan["output_bytes"] = projection.plan.output_bytes;
        plan["block_scratch_bytes"] = projection.plan.block_scratch_bytes;
        plan["estimated_bytes"] = projection.plan.estimated_bytes;
        plan["work_terms"] = projection.plan.work_terms;
        plan["max_work_terms"] = projection.plan.max_work_terms;
        plan["max_site_count"] = projection.plan.max_site_count;
        py::dict result;
        result["component_order"] =
            "00;10,11c,11s;20,21c,21s,22c,22s;30,31c,31s,32c,32s,33c,33s";
        result["transition_order"] = "(i,a) occupied-major/virtual-minor";
        result["transitions"] = projection.transitions;
        result["values"] = projection.values;
        result["plan"] = std::move(plan);
        return result;
    };
    m.def("_atomic_polarizability_test_project_transition_multipoles",
          [projection_result_dict](const std::vector<SitePosition>& points,
                                   const std::vector<double>& weights,
                                   const std::vector<double>& partition,
                                   const std::vector<SitePosition>& sites,
                                   const Matrix& transition_values) {
              return projection_result_dict(detail::project_transition_multipoles(
                  points, weights, partition, sites, transition_values));
          }, "points"_a, "weights"_a, "partition"_a, "sites"_a,
          "transition_values"_a);
    m.def("_atomic_polarizability_test_project_transition_multipoles_context",
          [projection_result_dict](const std::shared_ptr<FrozenResponseContext>& context,
                                   const std::shared_ptr<FrozenResponseContext>& isa_context,
                                   std::vector<double> partition) {
              if (!isa_context)
                  throw PSIEXCEPTION(
                      "transition multipole projection: ISA weights must belong to the same frozen response context");
              auto isa = ISAWeights::create_test_only(isa_context, std::move(partition));
              return projection_result_dict(project_transition_multipoles(context, isa));
          }, "context"_a, "isa_context"_a, "partition"_a);
    m.def("_atomic_polarizability_estimate_transition_multipole_projection",
          [](std::size_t point_count, std::size_t site_count,
             std::size_t transition_count, std::size_t max_block_points,
             std::size_t nbf, std::size_t nmo, std::size_t memory_bytes) {
              const auto plan = detail::plan_transition_multipole_projection(
                  point_count, site_count, transition_count, max_block_points,
                  nbf, nmo, memory_bytes);
              py::dict values;
              values["algorithm"] = plan.algorithm;
              values["estimated_bytes"] = plan.estimated_bytes;
              values["work_terms"] = plan.work_terms;
              return values;
          }, "point_count"_a, "site_count"_a, "transition_count"_a,
          "max_block_points"_a, "nbf"_a, "nmo"_a, "memory_bytes"_a);
    const auto site_pair_contraction_result_dict =
        [](const detail::SitePairResponseContraction& contraction) {
            py::dict plan;
            plan["algorithm"] = contraction.plan.algorithm;
            plan["memory_semantics"] = contraction.plan.memory_semantics;
            plan["site_count"] = contraction.plan.site_count;
            plan["transition_count"] = contraction.plan.transition_count;
            plan["component_count"] = contraction.plan.component_count;
            plan["output_bytes"] = contraction.plan.output_bytes;
            plan["scratch_bytes"] = contraction.plan.scratch_bytes;
            plan["estimated_bytes"] = contraction.plan.estimated_bytes;
            plan["work_terms"] = contraction.plan.work_terms;
            plan["max_work_terms"] = contraction.plan.max_work_terms;
            plan["max_site_count"] = contraction.plan.max_site_count;
            py::dict result;
            result["values"] = contraction.values;
            result["site_count"] = contraction.plan.site_count;
            result["transition_count"] = contraction.plan.transition_count;
            result["restricted_factor"] = contraction.restricted_factor;
            result["component_order"] =
                "00;10,11c,11s;20,21c,21s,22c,22s;30,31c,31s,32c,32s,33c,33s";
            result["block_order"] =
                "row=(response_site,ISA_component); column=(source_site,ISA_component)";
            result["response_map_symmetry_policy"] =
                "AVERAGE_WITHIN_SOLVER_FORWARD_ERROR_BOUND";
            result["response_map_forward_error_bound"] =
                contraction.response_map_forward_error_bound;
            result["response_map_solution_scale"] = contraction.response_map_solution_scale;
            result["response_map_allowed_antisymmetry"] =
                contraction.response_map_allowed_antisymmetry;
            result["response_map_symmetry_residual"] =
                contraction.response_map_symmetry_residual;
            result["response_map_max_normalized_antisymmetry"] =
                contraction.response_map_max_normalized_antisymmetry;
            result["reciprocity_enforced"] = contraction.reciprocity_enforced;
            result["plan"] = std::move(plan);
            return result;
        };
    m.def("_atomic_polarizability_test_solve_and_contract_site_pair_response",
          [site_pair_contraction_result_dict](std::size_t site_count,
                                               const Matrix& projection,
                                               const Matrix& H1, const Matrix& H2,
                                               double omega) {
              Matrix identity(H1.nrow(), H1.nrow());
              for (int row = 0; row < H1.nrow(); ++row) identity(row, row) = 1.0;
              const auto response = detail::solve_dense_restricted_response(
                  H1, H2, omega, identity);
              auto result = site_pair_contraction_result_dict(
                  detail::contract_site_pair_response(site_count, projection, response));
              result["P"] = response.P_clone();
              result["Q"] = response.Q_clone();
              result["reciprocal_condition"] = response.reciprocal_condition();
              result["reciprocal_pivot_growth"] = response.reciprocal_pivot_growth();
              result["forward_error"] = response.forward_error();
              result["backward_error"] = response.backward_error();
              result["scaled_residual"] = response.scaled_residual();
              result["solution_column_scales"] = response.solution_column_scales();
              result["max_forward_error"] = *std::max_element(
                  response.forward_error().begin(), response.forward_error().end());
              result["max_backward_error"] = *std::max_element(
                  response.backward_error().begin(), response.backward_error().end());
              result["max_scaled_residual"] = *std::max_element(
                  response.scaled_residual().begin(), response.scaled_residual().end());
              return result;
          }, "site_count"_a, "projection"_a, "H1"_a, "H2"_a, "omega"_a);
    m.def("_atomic_polarizability_test_validate_response_map_symmetry",
          [](const Matrix& response_map, const Matrix& conjugate_map,
             const std::vector<double>& forward_error) {
              const auto diagnostics = detail::validate_response_map_symmetry_test_only(
                  response_map, conjugate_map, forward_error);
              py::dict result;
              result["response_map_forward_error_bound"] =
                  *std::max_element(forward_error.begin(), forward_error.end());
              result["response_map_solution_scale"] = diagnostics.solution_scale;
              result["response_map_allowed_antisymmetry"] = diagnostics.allowed_antisymmetry;
              result["response_map_symmetry_residual"] = diagnostics.symmetry_residual;
              result["response_map_max_normalized_antisymmetry"] =
                  diagnostics.max_normalized_antisymmetry;
              return result;
          }, "response_map"_a, "conjugate_map"_a, "forward_error"_a);
    m.def("_atomic_polarizability_estimate_site_pair_response_contraction",
          [](std::size_t site_count, std::size_t transition_count,
             std::size_t memory_bytes) {
              const auto plan = detail::plan_site_pair_response_contraction(
                  site_count, transition_count, memory_bytes);
              py::dict values;
              values["algorithm"] = plan.algorithm;
              values["memory_semantics"] = plan.memory_semantics;
              values["component_count"] = plan.component_count;
              values["output_bytes"] = plan.output_bytes;
              values["scratch_bytes"] = plan.scratch_bytes;
              values["estimated_bytes"] = plan.estimated_bytes;
              values["work_terms"] = plan.work_terms;
              return values;
          }, "site_count"_a, "transition_count"_a, "memory_bytes"_a);
    m.def("_atomic_polarizability_estimate_isapol_response_provider",
          [](std::size_t frequency_count, std::size_t site_count,
             std::size_t nbf, std::size_t nocc, std::size_t nvir,
             const std::vector<std::size_t>& block_point_counts,
             const std::vector<std::size_t>& block_map_sizes,
             bool has_dynamic_frequency, std::size_t memory_bytes,
             double density_cutoff) {
              if (block_point_counts.empty() ||
                  block_point_counts.size() != block_map_sizes.size())
                  throw PSIEXCEPTION(
                      "ISAPolResponseProvider: inconsistent block metadata arrays");
              std::vector<FrozenGridBlock> blocks;
              blocks.reserve(block_point_counts.size());
              std::size_t point_count = 0;
              for (std::size_t block = 0; block < block_point_counts.size(); ++block) {
                  if (block_map_sizes[block] == 0 || block_map_sizes[block] > nbf)
                      throw PSIEXCEPTION(
                          "ISAPolResponseProvider: block map size exceeds basis dimension");
                  if (block_point_counts[block] == 0 ||
                      block_point_counts[block] >
                          std::numeric_limits<std::size_t>::max() - point_count)
                      throw PSIEXCEPTION(
                          "ISAPolResponseProvider: block point count is invalid");
                  std::vector<int> map(block_map_sizes[block]);
                  blocks.push_back(
                      {point_count, block_point_counts[block], std::move(map)});
                  point_count += block_point_counts[block];
              }
              const auto plan = detail::plan_isapol_response_provider(
                  frequency_count, site_count, nbf, nocc, nvir, point_count,
                  blocks, has_dynamic_frequency, memory_bytes, density_cutoff);
              py::dict values;
              values["algorithm"] = plan.algorithm;
              values["memory_semantics"] = plan.memory_semantics;
              values["frequency_count"] = plan.frequency_count;
              values["site_count"] = plan.site_count;
              values["nbf"] = plan.nbf;
              values["nocc"] = plan.nocc;
              values["nvir"] = plan.nvir;
              values["transition_count"] = plan.transition_count;
              values["point_count"] = plan.point_count;
              values["max_block_points"] = plan.max_block_points;
              values["component_count"] = plan.component_count;
              values["configured_memory_bytes"] = plan.configured_memory_bytes;
              values["reserved_memory_bytes"] = plan.reserved_memory_bytes;
              values["c1_plan_estimated_bytes"] = plan.c1_plan_estimated_bytes;
              values["alda_plan_estimated_bytes"] = plan.alda_plan_estimated_bytes;
              values["projection_plan_estimated_bytes"] =
                  plan.projection_plan_estimated_bytes;
              values["contraction_plan_estimated_bytes"] =
                  plan.contraction_plan_estimated_bytes;
              values["retained_c1_bytes"] = plan.retained_c1_bytes;
              values["retained_alda_bytes"] = plan.retained_alda_bytes;
              values["hessian_bytes"] = plan.hessian_bytes;
              values["retained_projection_bytes"] = plan.retained_projection_bytes;
              values["identity_bytes"] = plan.identity_bytes;
              values["retained_output_bytes"] = plan.retained_output_bytes;
              values["dense_solve_peak_bytes"] = plan.dense_solve_peak_bytes;
              values["response_carrier_bytes"] = plan.response_carrier_bytes;
              values["transition_metadata_bytes"] = plan.transition_metadata_bytes;
              values["conservative_overhead_bytes"] =
                  plan.conservative_overhead_bytes;
              values["c1_stage_peak_bytes"] = plan.c1_stage_peak_bytes;
              values["alda_stage_peak_bytes"] = plan.alda_stage_peak_bytes;
              values["projection_stage_peak_bytes"] =
                  plan.projection_stage_peak_bytes;
              values["dense_solve_stage_peak_bytes"] =
                  plan.dense_solve_stage_peak_bytes;
              values["contraction_stage_peak_bytes"] =
                  plan.contraction_stage_peak_bytes;
              values["estimated_bytes"] = plan.estimated_bytes;
              return values;
          }, "frequency_count"_a, "site_count"_a, "nbf"_a, "nocc"_a,
          "nvir"_a, "block_point_counts"_a, "block_map_sizes"_a,
          "has_dynamic_frequency"_a, "memory_bytes"_a, "density_cutoff"_a);
    m.def("_atomic_polarizability_estimate_restricted_alda",
          [alda_plan_dict](std::size_t nbf, std::size_t nocc, std::size_t nvir,
                           const std::vector<std::size_t>& block_point_counts,
                           const std::vector<std::size_t>& block_map_sizes,
                           std::size_t memory_bytes, bool retain_test_diagnostics,
                           double density_cutoff) {
              if (block_point_counts.size() != block_map_sizes.size() || block_point_counts.empty())
                  throw PSIEXCEPTION("restricted ALDA kernel: inconsistent block metadata arrays");
              std::vector<FrozenGridBlock> blocks;
              blocks.reserve(block_point_counts.size());
              std::size_t point_count = 0;
              for (std::size_t block = 0; block < block_point_counts.size(); ++block) {
                  if (block_map_sizes[block] > nbf)
                      throw PSIEXCEPTION("restricted ALDA kernel: block metadata exceeds basis dimension");
                  if (block_point_counts[block] > std::numeric_limits<std::size_t>::max() - point_count)
                      throw PSIEXCEPTION("restricted ALDA kernel: block point count overflow");
                  std::vector<int> map(block_map_sizes[block]);
                  blocks.push_back({point_count, block_point_counts[block], std::move(map)});
                  point_count += block_point_counts[block];
              }
              return alda_plan_dict(detail::plan_restricted_alda(
                  nbf, nocc, nvir, point_count, blocks, memory_bytes,
                  retain_test_diagnostics, density_cutoff));
          }, "nbf"_a, "nocc"_a, "nvir"_a, "block_point_counts"_a,
          "block_map_sizes"_a, "memory_bytes"_a, "retain_test_diagnostics"_a,
          "density_cutoff"_a);
    m.def("_atomic_polarizability_estimate_restricted_c1_jk",
          [](std::size_t nbf, std::size_t nocc, std::size_t nvir, std::size_t memory_bytes) {
              const auto plan = detail::plan_restricted_c1_jk(nbf, nocc, nvir, memory_bytes);
              py::dict result;
              result["algorithm"] = plan.algorithm;
              result["nbf"] = plan.nbf;
              result["nocc"] = plan.nocc;
              result["nvir"] = plan.nvir;
              result["nov"] = plan.nov;
              result["batch_size"] = plan.batch_size;
              result["jk_threads"] = plan.jk_threads;
              result["max_supported_nov"] = plan.max_supported_nov;
              result["configured_memory_bytes"] = plan.configured_memory_bytes;
              result["reserved_memory_bytes"] = plan.reserved_memory_bytes;
              result["retained_payload_bytes"] = plan.retained_payload_bytes;
              result["metadata_bytes"] = plan.metadata_bytes;
              result["coefficient_bytes"] = plan.coefficient_bytes;
              result["matrix_overhead_bytes"] = plan.matrix_overhead_bytes;
              result["jk_coefficient_bytes"] = plan.jk_coefficient_bytes;
              result["jk_ao_bytes"] = plan.jk_ao_bytes;
              result["direct_jk_scratch_bytes"] = plan.direct_jk_scratch_bytes;
              result["integral_engine_allowance_bytes"] = plan.integral_engine_allowance_bytes;
              result["projection_bytes"] = plan.projection_bytes;
              result["estimated_bytes"] = plan.estimated_bytes;
              result["integral_cutoff"] = plan.integral_cutoff;
              result["incfock"] = plan.incfock;
              result["screening"] = plan.screening;
              result["memory_semantics"] = plan.memory_semantics;
              return result;
          },
          "nbf"_a, "nocc"_a, "nvir"_a, "memory_bytes"_a);
    const auto site_pair_response_list =
        [](const std::vector<SitePairResponse>& responses) {
            py::list output;
            for (const auto& response : responses) {
                auto positions = std::make_shared<Matrix>(response.positions.size(), 3);
                for (std::size_t site = 0; site < response.positions.size(); ++site)
                    for (std::size_t axis = 0; axis < 3; ++axis)
                        (*positions)(site, axis) = response.positions[site][axis];
                py::list blocks;
                for (const auto& block : response.blocks) {
                    auto values = std::make_shared<Matrix>(16, 16);
                    for (std::size_t row = 0; row < 16; ++row)
                        for (std::size_t column = 0; column < 16; ++column)
                            (*values)(row, column) = block[row][column];
                    blocks.append(values);
                }
                py::dict item;
                item["positions"] = positions;
                item["blocks"] = std::move(blocks);
                item["chf_exchange_coefficient"] = 0.25;
                item["alda_kernel_coefficient"] = 0.75;
                item["restricted_factor"] = 4.0;
                item["component_order"] =
                    "00;10,11c,11s;20,21c,21s,22c,22s;30,31c,31s,32c,32s,33c,33s";
                output.append(std::move(item));
            }
            return output;
        };
    py::class_<ISAPolResponseProvider, std::shared_ptr<ISAPolResponseProvider>>(
        m, "_AtomicPolarizabilityTestResponseProvider")
        .def("expected_response_count",
             [](const ISAPolResponseProvider& provider, std::vector<double> frequencies,
                std::vector<double> weights) {
                 return provider.expected_response_count(FrequencyGrid{std::move(frequencies), std::move(weights)});
             })
        .def("compute", [site_pair_response_list](const ISAPolResponseProvider& provider,
                                                   std::vector<double> frequencies,
                                                   std::vector<double> weights) {
            return site_pair_response_list(provider.compute_isapol_response(
                FrequencyGrid{std::move(frequencies), std::move(weights)}));
        });
    m.def("_atomic_polarizability_make_frozen_response_context", &FrozenResponseContext::create,
          "grac_wfn"_a, "neutral_precursor_wfn"_a, "cation_wfn"_a);
    m.def("_atomic_polarizability_compute_isa_weights",
          [isa_options_from_dict, isa_result_dict](const std::shared_ptr<FrozenResponseContext>& context,
                                                     const py::dict& option_values) {
              const auto result = compute_isa_weights(context, isa_options_from_dict(option_values));
              return isa_result_dict(result.site_count(), result.partition_weights(), result.diagnostics());
          },
          "context"_a, "options"_a = py::dict());
    m.def("_atomic_polarizability_make_test_response_provider",
          [](const std::shared_ptr<FrozenResponseContext>& context,
             const std::shared_ptr<FrozenResponseContext>& isa_context) {
              const auto count = isa_context->grid_point_count() * isa_context->sites().size();
              std::vector<double> weights(count, 1.0 / static_cast<double>(isa_context->sites().size()));
              auto isa = ISAWeights::create_test_only(isa_context, std::move(weights));
              return std::make_shared<ISAPolResponseProvider>(context, ResponseKernel(0.25, 0.75), std::move(isa));
          }, "context"_a, "isa_context"_a);
    m.def("_atomic_polarizability_make_native_response_provider",
          [isa_options_from_dict](const std::shared_ptr<FrozenResponseContext>& context,
                                  const py::dict& option_values) {
              auto isa = compute_isa_weights(context, isa_options_from_dict(option_values));
              return std::make_shared<ISAPolResponseProvider>(
                  context, ResponseKernel(0.25, 0.75), std::move(isa));
          }, "context"_a, "options"_a = py::dict());
    m.def("_atomic_polarizability_local_spherical_dipole_to_cartesian",
          [](const Matrix& spherical) {
              if (spherical.nirrep() != 1 || spherical.nrow() != 15 || spherical.ncol() != 15) {
                  throw PSIEXCEPTION(
                      "local_spherical_dipole_to_cartesian: expected a 15 by 15 rank-3 matrix");
              }
              L3Matrix values{};
              for (std::size_t row = 0; row < values.size(); ++row) {
                  for (std::size_t column = 0; column < values[row].size(); ++column) {
                      values[row][column] = spherical(row, column);
                  }
              }
              return std::make_shared<Matrix>(local_spherical_dipole_to_cartesian(values));
          });
    m.def("_atomic_polarizability_rotate_tensor",
          [](const Matrix& local, const Matrix& local_to_global) {
              return std::make_shared<Matrix>(rotate_tensor(local, local_to_global));
          });
    m.def("_atomic_polarizability_pack_symmetric_tensor", &pack_symmetric_tensor);
    m.def("_atomic_polarizability_lw_graph_math",
          [](std::size_t site_count, const std::vector<std::array<std::size_t, 2>>& bonds) {
              BondGraph graph{site_count, bonds};
              auto operator_matrix = std::make_shared<Matrix>(lw_graph_operator(graph));
              auto inverse_and_values = lw_graph_pseudoinverse(graph);
              auto pseudoinverse = std::make_shared<Matrix>(std::move(inverse_and_values.first));
              return py::make_tuple(operator_matrix, pseudoinverse, inverse_and_values.second);
          });
    m.def("_atomic_polarizability_translate_l3", &translate_l3_multipoles);
    m.def("_atomic_polarizability_localize_lw",
          [](const Matrix& positions, const std::vector<std::shared_ptr<Matrix>>& block_matrices,
             double frequency, const std::vector<std::array<std::size_t, 2>>& bonds,
             double residual_tolerance) {
              if (positions.nirrep() != 1 || positions.ncol() != 3) {
                  throw PSIEXCEPTION("localize_lw: positions must be an N by 3 matrix");
              }
              SitePairResponse response;
              response.frequency = frequency;
              response.positions.resize(positions.nrow());
              for (std::size_t site = 0; site < response.positions.size(); ++site) {
                  for (std::size_t axis = 0; axis < 3; ++axis) {
                      response.positions[site][axis] = positions(site, axis);
                  }
              }
              if (block_matrices.size() != response.positions.size() * response.positions.size()) {
                  throw PSIEXCEPTION("localize_lw: expected one 16 by 16 block for every ordered site pair");
              }
              response.blocks.resize(block_matrices.size());
              for (std::size_t block = 0; block < block_matrices.size(); ++block) {
                  if (!block_matrices[block] || block_matrices[block]->nirrep() != 1 ||
                      block_matrices[block]->nrow() != 16 || block_matrices[block]->ncol() != 16) {
                      throw PSIEXCEPTION("localize_lw: expected 16 by 16 rank-0-through-rank-3 blocks");
                  }
                  for (std::size_t row = 0; row < 16; ++row) {
                      for (std::size_t column = 0; column < 16; ++column) {
                          response.blocks[block][row][column] = (*block_matrices[block])(row, column);
                      }
                  }
              }
              const auto localized =
                  localize_lw(response, BondGraph{response.positions.size(), bonds}, residual_tolerance);
              py::dict result;
              py::list local;
              for (const auto& block : localized.local) {
                  auto matrix = std::make_shared<Matrix>(15, 15);
                  for (std::size_t row = 0; row < 15; ++row) {
                      for (std::size_t column = 0; column < 15; ++column) {
                          (*matrix)(row, column) = block[row][column];
                      }
                  }
                  local.append(matrix);
              }
              py::list refined;
              for (const auto& block : localized.refined_pairs) {
                  auto matrix = std::make_shared<Matrix>(16, 16);
                  for (std::size_t row = 0; row < 16; ++row) {
                      for (std::size_t column = 0; column < 16; ++column) {
                          (*matrix)(row, column) = block[row][column];
                      }
                  }
                  refined.append(matrix);
              }
              py::list transfers;
              for (const auto& transfer : localized.transfers) {
                  transfers.append(py::make_tuple(
                      transfer.first, transfer.second, transfer.first_component,
                      transfer.second_component, transfer.fixed_site, transfer.amount));
              }
              result["frequency"] = localized.frequency;
              result["local"] = local;
              result["refined"] = refined;
              result["transfers"] = transfers;
              result["omitted_component_pairs"] = localized.omitted_component_pairs;
              result["omitted_transfer_count"] = localized.omitted_transfer_count;
              result["residuals"] = py::make_tuple(
                  localized.residuals.off_site, localized.residuals.charge_sum,
                  localized.residuals.reciprocity, localized.residuals.molecular_sum,
                  localized.residuals.local_charge);
              return result;
          });

    const auto l3_tensor_from_matrix = [](const SharedMatrix& matrix, const char* context) {
        if (!matrix || matrix->nirrep() != 1 || matrix->nrow() != 15 || matrix->ncol() != 15)
            throw PSIEXCEPTION(std::string(context) + ": expected 15 by 15 rank-3 L3 tensors");
        L3Matrix values{};
        for (std::size_t row = 0; row < values.size(); ++row)
            for (std::size_t column = 0; column < values[row].size(); ++column)
                values[row][column] = (*matrix)(row, column);
        return values;
    };
    const auto refined_models_from_python =
        [l3_tensor_from_matrix](const Matrix& site_matrix, const std::vector<double>& frequencies,
                               const std::vector<SharedMatrix>& tensors) {
            if (site_matrix.nirrep() != 1 || site_matrix.ncol() != 3 || site_matrix.nrow() <= 0)
                throw PSIEXCEPTION("dispersion: sites must be a nonempty N by 3 matrix");
            const auto site_count = static_cast<std::size_t>(site_matrix.nrow());
            std::vector<SitePosition> sites(site_count);
            for (std::size_t site = 0; site < site_count; ++site)
                for (std::size_t axis = 0; axis < 3; ++axis) sites[site][axis] = site_matrix(site, axis);
            if (tensors.size() != frequencies.size() * site_count)
                throw PSIEXCEPTION(
                    "dispersion: expected one frequency-major 15 by 15 tensor per site");
            std::vector<RefinedL3Model> models(frequencies.size());
            for (std::size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
                models[frequency].frequency = frequencies[frequency];
                models[frequency].positions = sites;
                models[frequency].tensors.resize(site_count);
                for (std::size_t site = 0; site < site_count; ++site)
                    models[frequency].tensors[site] =
                        l3_tensor_from_matrix(tensors[frequency * site_count + site], "dispersion");
            }
            return models;
        };
    const auto dispersion_plan_dict = [](const DispersionPlan& plan) {
        py::dict plan_values;
        plan_values["frequency_count"] = plan.frequency_count;
        plan_values["site_count"] = plan.site_count;
        plan_values["max_frequency_count"] = plan.max_frequency_count;
        plan_values["max_site_count"] = plan.max_site_count;
        plan_values["coefficient_count"] = plan.coefficient_count;
        plan_values["rank_pair_count"] = plan.rank_pair_count;
        plan_values["isotropic_elements"] = plan.isotropic_elements;
        plan_values["isotropic_bytes"] = plan.isotropic_bytes;
        plan_values["coefficient_elements"] = plan.coefficient_elements;
        plan_values["coefficient_bytes"] = plan.coefficient_bytes;
        plan_values["contribution_elements"] = plan.contribution_elements;
        plan_values["contribution_bytes"] = plan.contribution_bytes;
        plan_values["rank_pair_table_bytes"] = plan.rank_pair_table_bytes;
        plan_values["metadata_bytes"] = plan.metadata_bytes;
        plan_values["estimated_bytes"] = plan.estimated_bytes;
        plan_values["configured_memory_bytes"] = plan.configured_memory_bytes;
        plan_values["reserved_memory_bytes"] = plan.reserved_memory_bytes;
        plan_values["work_terms"] = plan.work_terms;
        plan_values["max_work_terms"] = plan.max_work_terms;
        plan_values["algorithm"] = plan.algorithm;
        plan_values["memory_semantics"] = plan.memory_semantics;
        return plan_values;
    };
    const auto dispersion_result_dict = [dispersion_plan_dict](const DispersionMatrices& dispersion) {
        const auto& diagnostics = dispersion.diagnostics;
        py::list rank_pair_terms;
        for (const auto& term : diagnostics.rank_pair_terms) {
            py::dict values;
            values["coefficient_order"] = term.coefficient_order;
            values["first_rank"] = term.first_rank;
            values["second_rank"] = term.second_rank;
            values["prefactor"] = term.prefactor;
            rank_pair_terms.append(std::move(values));
        }
        py::dict result;
        result["c6"] = dispersion.c6;
        result["c8"] = dispersion.c8;
        result["c10"] = dispersion.c10;
        result["c12"] = dispersion.c12;
        result["frequency_count"] = diagnostics.frequency_count;
        result["weighted_frequency_count"] = diagnostics.weighted_frequency_count;
        result["site_count"] = diagnostics.site_count;
        result["quadrature_weight_sum"] = diagnostics.quadrature_weight_sum;
        result["min_isotropic_polarizability"] = diagnostics.min_isotropic_polarizability;
        result["max_isotropic_polarizability"] = diagnostics.max_isotropic_polarizability;
        result["nonpositive_isotropic_count"] = diagnostics.nonpositive_isotropic_count;
        result["inferred_scale"] = diagnostics.inferred_scale;
        result["max_protocol_grid_deviation"] = diagnostics.max_protocol_grid_deviation;
        result["protocol_grid_enforced"] = diagnostics.protocol_grid_enforced;
        result["rank_pair_terms"] = std::move(rank_pair_terms);
        result["rank_pair_contributions"] = diagnostics.rank_pair_contributions;
        result["plan"] = dispersion_plan_dict(diagnostics.plan);
        return result;
    };
    m.def("_atomic_polarizability_dispersion_rank_prefactor", &detail::dispersion_rank_prefactor,
          "first_rank"_a, "second_rank"_a);
    m.def("_atomic_polarizability_dispersion_isotropic_rank",
          [l3_tensor_from_matrix](const SharedMatrix& tensor, unsigned int rank) {
              return detail::isotropic_rank_polarizability(
                  l3_tensor_from_matrix(tensor, "isotropic rank polarizability"), rank);
          },
          "tensor"_a, "rank"_a);
    m.def("_atomic_polarizability_plan_dispersion",
          [dispersion_plan_dict](std::size_t frequency_count, std::size_t site_count,
                                 std::size_t memory_bytes) {
              return dispersion_plan_dict(
                  detail::plan_dispersion(frequency_count, site_count, memory_bytes));
          },
          "frequency_count"_a, "site_count"_a, "memory_bytes"_a);
    m.def("_atomic_polarizability_compute_dispersion",
          [refined_models_from_python, dispersion_result_dict](
              const Matrix& sites, const std::vector<double>& frequencies,
              const std::vector<SharedMatrix>& tensors,
              const std::vector<double>& grid_frequencies,
              const std::vector<double>& grid_weights) {
              return dispersion_result_dict(compute_dispersion(
                  refined_models_from_python(sites, frequencies, tensors),
                  FrequencyGrid{grid_frequencies, grid_weights}));
          },
          "sites"_a, "frequencies"_a, "tensors"_a, "grid_frequencies"_a, "grid_weights"_a);
    m.def("_atomic_polarizability_test_compute_dispersion",
          [refined_models_from_python, dispersion_result_dict](
              const Matrix& sites, const std::vector<double>& frequencies,
              const std::vector<SharedMatrix>& tensors,
              const std::vector<double>& grid_frequencies,
              const std::vector<double>& grid_weights) {
              return dispersion_result_dict(detail::compute_dispersion_test_only(
                  refined_models_from_python(sites, frequencies, tensors),
                  FrequencyGrid{grid_frequencies, grid_weights}));
          },
          "sites"_a, "frequencies"_a, "tensors"_a, "grid_frequencies"_a, "grid_weights"_a);
    m.def("_atomic_polarizability_derive_pdef_constraints",
          [](const Molecule& molecule, const std::vector<SharedMatrix>& axis_matrices) {
              std::vector<SiteAxes> site_axes(axis_matrices.size());
              for (std::size_t site = 0; site < axis_matrices.size(); ++site) {
                  const auto& frame = axis_matrices[site];
                  if (!frame || frame->nirrep() != 1 || frame->nrow() != 3 || frame->ncol() != 3)
                      throw PSIEXCEPTION("PDef constraints: local axes must be 3 by 3 matrices");
                  for (std::size_t row = 0; row < 3; ++row)
                      for (std::size_t column = 0; column < 3; ++column)
                          site_axes[site][row][column] =
                              (*frame)(static_cast<int>(row), static_cast<int>(column));
              }
              const auto derived = derive_pdef_constraints(molecule, site_axes);
              py::list sites;
              for (const auto& record : derived.sites) {
                  py::dict values;
                  values["point_group"] = record.point_group;
                  values["point_group_bits"] = record.point_group_bits;
                  values["operation_signs"] = record.operation_signs;
                  values["component_class"] = record.component_class;
                  values["class_count"] = record.class_count;
                  values["active_pairs"] = record.active_pairs;
                  values["symmetry_source"] = record.symmetry_source;
                  values["copy_signs"] = record.copy_signs;
                  sites.append(std::move(values));
              }
              py::dict result;
              result["molecular_point_group"] = derived.molecular_point_group;
              result["variable_count"] = derived.variable_count;
              result["active_variable_count"] = derived.active_variable_count;
              result["equality_row_count"] = derived.equality_row_count;
              result["independent_variable_count"] = derived.independent_variable_count;
              result["geometry_tolerance"] = derived.geometry_tolerance;
              result["active_variables"] = derived.constraints.active_variables;
              result["equality"] = derived.constraints.equality;
              result["equality_targets"] = derived.constraints.equality_targets;
              result["sites"] = std::move(sites);
              return result;
          },
          "molecule"_a, "site_axes"_a = std::vector<SharedMatrix>());
    m.def("_atomic_polarizability_derive_bond_graph",
          [](const Molecule& molecule, double covalent_scale) {
              const auto derived = derive_bond_graph(molecule, covalent_scale);
              py::dict result;
              result["site_count"] = derived.graph.site_count;
              result["bonds"] = derived.graph.bonds;
              result["covalent_scale"] = derived.covalent_scale;
              result["radius_table"] = derived.radius_table;
              result["radii"] = derived.radii;
              result["bond_distances"] = derived.bond_distances;
              result["bond_thresholds"] = derived.bond_thresholds;
              result["component_count"] = derived.component_count;
              result["component_labels"] = derived.component_labels;
              return result;
          },
          "molecule"_a, "covalent_scale"_a = kCovalentBondScale);
    m.def("_atomic_polarizability_derive_bond_graph_from_sites",
          [](const Matrix& site_matrix, const std::vector<int>& atomic_numbers,
             double covalent_scale) {
              if (site_matrix.nirrep() != 1 || site_matrix.ncol() != 3 || site_matrix.nrow() <= 0)
                  throw PSIEXCEPTION("Bond graph: sites must be a nonempty N by 3 matrix");
              std::vector<SitePosition> sites(static_cast<std::size_t>(site_matrix.nrow()));
              for (std::size_t site = 0; site < sites.size(); ++site)
                  for (std::size_t axis = 0; axis < 3; ++axis)
                      sites[site][axis] = site_matrix(site, axis);
              const auto derived = detail::derive_bond_graph(sites, atomic_numbers, covalent_scale);
              py::dict result;
              result["site_count"] = derived.graph.site_count;
              result["bonds"] = derived.graph.bonds;
              result["covalent_scale"] = derived.covalent_scale;
              result["radius_table"] = derived.radius_table;
              result["radii"] = derived.radii;
              result["bond_distances"] = derived.bond_distances;
              result["bond_thresholds"] = derived.bond_thresholds;
              result["component_count"] = derived.component_count;
              result["component_labels"] = derived.component_labels;
              return result;
          },
          "sites"_a, "atomic_numbers"_a, "covalent_scale"_a = kCovalentBondScale);

    py::class_<AtomicPolarizabilityCalculator>(m, "AtomicPolarizabilityCalculator",
                                               "Native atomic-polarizability pipeline entry point")
        .def(py::init<std::shared_ptr<Wavefunction>>(),
             "Single-wavefunction seam without the SCF triple; compute() fails closed.")
        .def(py::init<std::shared_ptr<Wavefunction>, std::shared_ptr<Wavefunction>,
                      std::shared_ptr<Wavefunction>>(),
             "grac_wfn"_a, "neutral_precursor_wfn"_a, "cation_wfn"_a)
        .def("compute", &AtomicPolarizabilityCalculator::compute,
             "Validate prerequisites and compute native atomic polarizabilities.");

    py::class_<Prop, std::shared_ptr<Prop> >(m, "Prop", "docstring");

    py::class_<ESPPropCalc, std::shared_ptr<ESPPropCalc>, Prop>(
        m, "ESPPropCalc", "ESPPropCalc gives access to routines calculating the ESP on a grid")
        .def(py::init<std::shared_ptr<Wavefunction> >())
        .def("compute_esp_over_grid_in_memory", &ESPPropCalc::compute_esp_over_grid_in_memory,
             "Computes ESP on specified grid Nx3 (as SharedMatrix, in input units)")
        .def("compute_field_over_grid_in_memory", &ESPPropCalc::compute_field_over_grid_in_memory,
             "Computes field on specified grid Nx3 (as SharedMatrix, in input units)");

    py::class_<OEProp, std::shared_ptr<OEProp>>(m, "OEProp", "docstring")
        .def(py::init<std::shared_ptr<Wavefunction> >())
        .def("add", py::overload_cast<const std::string&>(&OEProp::add), "Append the given task to the list of properties to compute.")
        .def("compute", &OEProp::compute, "Compute the properties.")
        .def("clear", &OEProp::clear, "Clear the list of properties to compute.")
        .def("set_Da_ao", &OEProp::set_Da_ao, "docstring", "Da"_a, "symmetry"_a = 0)
        .def("set_Db_ao", &OEProp::set_Db_ao, "docstring", "Db"_a, "symmetry"_a = 0)
        .def("set_Da_so", &OEProp::set_Da_so, "docstring")
        .def("set_Db_so", &OEProp::set_Db_so, "docstring")
        .def("set_Da_mo", &OEProp::set_Da_mo, "docstring")
        .def("set_Db_mo", &OEProp::set_Db_mo, "docstring")
        .def("Vvals", &OEProp::Vvals, "The electrostatic potential (in a.u.) at each grid point")
        .def("Exvals", &OEProp::Exvals, "The x component of the field (in a.u.) at each grid point")
        .def("Eyvals", &OEProp::Eyvals, "The y component of the field (in a.u.) at each grid point")
        .def("Ezvals", &OEProp::Ezvals, "The z component of the field (in a.u.) at each grid point")
        .def("set_title", &OEProp::set_title,
             "Title OEProp for print purposes. As a side effect, saves variables as title + propertyname and only that. "
             "Follow up with side names, if the side effect is undesired,", "title"_a)
        .def("set_names", &OEProp::set_names,
             "Instruct OEProp to save variables under all specified names. The property name will "
             "be inserted at every occurrence of {}, like Python format strings. Wipes other names-to-save-by.")
        .def("set_atomic_polarizability_references", &OEProp::set_atomic_polarizability_references,
             "Supply the neutral precursor and cation SCF wavefunctions required by the native "
             "ATOMIC_POLARIZABILITIES task alongside this object's GRAC-corrected reference. "
             "Without them the task fails closed and publishes nothing.",
             "neutral_precursor_wfn"_a, "cation_wfn"_a);

    // class_<GridProp, std::shared_ptr<GridProp> >("GridProp", "docstring").
    //    def("add", &GridProp::gridpy_add, "docstring").
    //    def("set_filename", &GridProp::set_filename, "docstring").
    //    def("add_alpha_mo", &GridProp::add_alpha_mo, "docstring").
    //    def("add_beta_mo", &GridProp::add_beta_mo, "docstring").
    //    def("add_basis_fun", &GridProp::add_basis_fun, "docstring").
    //    def("build_grid_overages", &GridProp::build_grid_overages, "docstring").
    //    def("set_n", &GridProp::set_n, "docstring").
    //    def("set_o", &GridProp::set_o, "docstring").
    //    def("set_l", &GridProp::set_l, "docstring").
    //    def("get_n", &GridProp::get_n, "docstring").
    //    def("get_o", &GridProp::get_o, "docstring").
    //    def("get_l", &GridProp::get_l, "docstring").
    //    def("set_caxis", &GridProp::set_caxis, "docstring").
    //    def("set_format", &GridProp::set_format, "docstring").
    //    def("compute", &GridProp::gridpy_compute, "docstring");

    // ------------------------------------------------------------------
    // Anisotropic distributed dispersion coefficients. Appended as one block so
    // the isotropic exports above stay untouched.
    // ------------------------------------------------------------------
    const auto anisotropic_site_position = [](const std::vector<double>& values,
                                              const char* context) {
        if (values.size() != 3)
            throw PSIEXCEPTION(std::string(context) + ": expected exactly three components");
        SitePosition position{};
        for (std::size_t axis = 0; axis < 3; ++axis) position[axis] = values[axis];
        return position;
    };
    const auto anisotropic_site_axes = [](const SharedMatrix& matrix, const char* context) {
        if (!matrix || matrix->nirrep() != 1 || matrix->nrow() != 3 || matrix->ncol() != 3)
            throw PSIEXCEPTION(std::string(context) + ": expected a 3 by 3 frame");
        SiteAxes axes{};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 3; ++column)
                axes[row][column] = (*matrix)(static_cast<int>(row), static_cast<int>(column));
        return axes;
    };
    const auto anisotropic_l3_matrix = [](const L3Matrix& values, const char* name) {
        auto matrix = std::make_shared<Matrix>(name, 15, 15);
        for (std::size_t row = 0; row < values.size(); ++row)
            for (std::size_t column = 0; column < values[row].size(); ++column)
                matrix->set(static_cast<int>(row), static_cast<int>(column), values[row][column]);
        return matrix;
    };
    m.def("_atomic_polarizability_anisotropic_component_order",
          []() { return anisotropic_component_order(); });
    m.def("_atomic_polarizability_test_multipole_interaction_tensor",
          [anisotropic_site_position, anisotropic_l3_matrix](const std::vector<double>& separation) {
              return anisotropic_l3_matrix(
                  detail::multipole_interaction_tensor(
                      anisotropic_site_position(separation, "multipole interaction tensor")),
                  "Multipole interaction tensor");
          },
          "separation"_a);
    m.def("_atomic_polarizability_test_l3_rank_rotation",
          [anisotropic_site_axes, anisotropic_l3_matrix](const SharedMatrix& rotation) {
              return anisotropic_l3_matrix(
                  detail::l3_rank_rotation(anisotropic_site_axes(rotation, "L3 rank rotation")),
                  "L3 rank rotation");
          },
          "rotation"_a);
    const auto anisotropic_tensor_series = [l3_tensor_from_matrix](
                                               const std::vector<SharedMatrix>& tensors,
                                               const char* context) {
        std::vector<L3Matrix> series(tensors.size());
        for (std::size_t point = 0; point < tensors.size(); ++point)
            series[point] = l3_tensor_from_matrix(tensors[point], context);
        return series;
    };
    m.def("_atomic_polarizability_test_anisotropic_block_product",
          [anisotropic_tensor_series](const std::vector<SharedMatrix>& first,
                                     const std::vector<SharedMatrix>& second,
                                     const std::vector<double>& weights) {
              const auto product = detail::anisotropic_block_product(
                  anisotropic_tensor_series(first, "anisotropic block product"),
                  anisotropic_tensor_series(second, "anisotropic block product"), weights);
              const auto isotropic = detail::isotropic_from_anisotropic_block_product(product);
              py::dict result;
              result["values"] = product;
              result["isotropic"] = std::vector<double>(isotropic.begin(), isotropic.end());
              return result;
          },
          "first"_a, "second"_a, "weights"_a);
    m.def("_atomic_polarizability_test_direct_anisotropic_energy",
          [anisotropic_tensor_series, anisotropic_site_position](
              const std::vector<SharedMatrix>& first, const std::vector<SharedMatrix>& second,
              const std::vector<double>& weights, const std::vector<double>& separation) {
              const auto product = detail::anisotropic_block_product(
                  anisotropic_tensor_series(first, "direct anisotropic dispersion energy"),
                  anisotropic_tensor_series(second, "direct anisotropic dispersion energy"),
                  weights);
              return detail::direct_anisotropic_energy(
                  product,
                  anisotropic_site_position(separation, "direct anisotropic dispersion energy"));
          },
          "first"_a, "second"_a, "weights"_a, "separation"_a);
    const auto anisotropic_label_from_python = [](const std::vector<unsigned int>& values) {
        if (values.size() != 6)
            throw PSIEXCEPTION(
                "anisotropic dispersion label: expected (n, l1, k1, l2, k2, j)");
        AnisotropicDispersionLabel label;
        label.order = values[0];
        label.first_rank = values[1];
        label.first_component = values[2];
        label.second_rank = values[3];
        label.second_component = values[4];
        label.coupled_rank = values[5];
        return label;
    };
    m.def("_atomic_polarizability_anisotropic_recoupling_table", []() {
        const auto& table = detail::anisotropic_recoupling_table();
        py::list labels;
        for (const auto& label : table.labels)
            labels.append(std::vector<unsigned int>{
                label.order, label.first_rank, label.first_component, label.second_rank,
                label.second_component, label.coupled_rank});
        py::list entries;
        for (const auto& entry : table.entries) {
            py::dict values;
            values["first_site_rank"] = entry.first_site_rank;
            values["first_site_rank_prime"] = entry.first_site_rank_prime;
            values["second_site_rank"] = entry.second_site_rank;
            values["second_site_rank_prime"] = entry.second_site_rank_prime;
            values["order"] = entry.order;
            values["first_rank"] = entry.first_rank;
            values["second_rank"] = entry.second_rank;
            values["coupled_rank"] = entry.coupled_rank;
            values["scalar"] = entry.scalar;
            entries.append(std::move(values));
        }
        py::list conventions;
        for (const auto& convention : table.conventions)
            conventions.append(py::make_tuple(convention.first, convention.second));
        py::dict result;
        result["version"] = table.version;
        result["generator"] = table.generator;
        result["component_order"] = table.component_order;
        result["conventions"] = std::move(conventions);
        result["entry_count"] = table.entries.size();
        result["label_count"] = table.labels.size();
        result["coupling_matrix_count"] = table.coupling_matrices.size();
        result["labels"] = std::move(labels);
        result["entries"] = std::move(entries);
        result["max_collapse_residual"] = table.max_collapse_residual;
        result["max_rotation_orthogonality_deviation"] =
            table.max_rotation_orthogonality_deviation;
        return result;
    });
    m.def("_atomic_polarizability_test_dense_anisotropic_recoupling",
          [anisotropic_label_from_python](const std::vector<unsigned int>& label) {
              return detail::dense_anisotropic_recoupling(anisotropic_label_from_python(label));
          },
          "label"_a);
    m.def("_atomic_polarizability_test_anisotropic_coefficients",
          [anisotropic_tensor_series](const std::vector<SharedMatrix>& first,
                                     const std::vector<SharedMatrix>& second,
                                     const std::vector<double>& weights) {
              return detail::anisotropic_coefficients_from_block_product(
                  detail::anisotropic_block_product(
                      anisotropic_tensor_series(first, "anisotropic dispersion coefficients"),
                      anisotropic_tensor_series(second, "anisotropic dispersion coefficients"),
                      weights));
          },
          "first"_a, "second"_a, "weights"_a);
    m.def("_atomic_polarizability_test_anisotropic_s_functions",
          [anisotropic_site_axes, anisotropic_site_position](const SharedMatrix& first_frame,
                                                            const SharedMatrix& second_frame,
                                                            const std::vector<double>& direction) {
              return detail::anisotropic_s_functions(
                  anisotropic_site_axes(first_frame, "anisotropic S functions"),
                  anisotropic_site_axes(second_frame, "anisotropic S functions"),
                  anisotropic_site_position(direction, "anisotropic S functions"));
          },
          "first_frame"_a, "second_frame"_a, "direction"_a);
    m.def("_atomic_polarizability_test_anisotropic_orientational_average",
          [anisotropic_tensor_series](const std::vector<SharedMatrix>& first,
                                     const std::vector<SharedMatrix>& second,
                                     const std::vector<double>& weights) {
              const auto& table = detail::anisotropic_recoupling_table();
              const auto averaged = detail::anisotropic_orientational_average_test_only();
              const auto coefficients = detail::anisotropic_coefficients_from_block_product(
                  detail::anisotropic_block_product(
                      anisotropic_tensor_series(first, "anisotropic orientational average"),
                      anisotropic_tensor_series(second, "anisotropic orientational average"),
                      weights));
              double worst_s_function = 0.0;
              std::map<unsigned int, double> per_order;
              std::map<unsigned int, double> isotropic;
              for (std::size_t index = 0; index < table.labels.size(); ++index) {
                  const auto& label = table.labels[index];
                  const bool scalar = label.first_rank == 0 && label.second_rank == 0 &&
                                      label.coupled_rank == 0;
                  worst_s_function =
                      std::max(worst_s_function, std::abs(averaged[index] - (scalar ? 1.0 : 0.0)));
                  per_order[label.order] += coefficients[index] * averaged[index];
                  if (scalar) isotropic[label.order] = coefficients[index];
              }
              double worst_isotropic = 0.0;
              for (const auto& item : isotropic)
                  worst_isotropic = std::max(
                      worst_isotropic,
                      std::abs(per_order.at(item.first) / item.second - 1.0));
              py::dict result;
              result["max_s_function_deviation"] = worst_s_function;
              result["max_isotropic_deviation"] = worst_isotropic;
              result["label_count"] = table.labels.size();
              return result;
          },
          "first"_a, "second"_a, "weights"_a);
    m.def("_atomic_polarizability_test_anisotropic_table_rejection",
          [](const std::string& mutation) {
              // Every mutation below breaks exactly one structural invariant. The
              // loader must refuse the table rather than trust it, following the
              // precedent of validate_dispersion_rank_pairs.
              auto table = detail::anisotropic_recoupling_table();
              if (mutation == "none") {
                  detail::validate_anisotropic_recoupling_table(table);
                  return true;
              }
              if (table.entries.empty() || table.labels.empty())
                  throw PSIEXCEPTION("anisotropic table rejection: the table is empty");
              auto& entry = table.entries.front();
              auto& label = table.labels.front();
              if (mutation == "version") {
                  table.version = "partB-recoupling-0";
              } else if (mutation == "generator") {
                  table.generator = "hand edited";
              } else if (mutation == "component_order") {
                  table.component_order.at(4) = "12c";
              } else if (mutation == "conventions") {
                  table.conventions.at(5).second = "M = sum_k w_k alpha^A alpha^B";
              } else if (mutation == "nonfinite_scalar") {
                  entry.scalar = std::numeric_limits<double>::quiet_NaN();
              } else if (mutation == "site_rank") {
                  entry.first_site_rank = 4;
              } else if (mutation == "order_mismatch") {
                  entry.order += 2;
              } else if (mutation == "order_range") {
                  for (auto& item : table.entries) {
                      item.first_site_rank = 1;
                      item.first_site_rank_prime = 1;
                      item.second_site_rank = 1;
                      item.second_site_rank_prime = 1;
                      item.order = 6;
                  }
              } else if (mutation == "first_rank_triangle") {
                  entry.first_rank = entry.first_site_rank + entry.first_site_rank_prime + 1;
              } else if (mutation == "second_rank_triangle") {
                  entry.second_rank = entry.second_site_rank + entry.second_site_rank_prime + 1;
              } else if (mutation == "capital_triangle") {
                  entry.coupled_rank = entry.first_site_rank + entry.first_site_rank_prime +
                                       entry.second_site_rank + entry.second_site_rank_prime + 2;
              } else if (mutation == "coupled_triangle") {
                  entry.first_rank = 0;
                  entry.second_rank = 0;
                  entry.coupled_rank = 2;
              } else if (mutation == "parity") {
                  entry.coupled_rank += 1;
              } else if (mutation == "zero_scalar") {
                  entry.scalar = 0.0;
              } else if (mutation == "duplicate_entry") {
                  table.entries.insert(table.entries.begin(), table.entries.front());
              } else if (mutation == "missing_site_rank") {
                  table.entries.erase(
                      std::remove_if(table.entries.begin(), table.entries.end(),
                                     [](const AnisotropicRecouplingEntry& item) {
                                         return item.first_site_rank == 3;
                                     }),
                      table.entries.end());
              } else if (mutation == "component_index") {
                  label.first_component = 2 * label.first_rank + 1;
              } else if (mutation == "label_triangle") {
                  label.coupled_rank = label.first_rank + label.second_rank + 1;
              } else if (mutation == "label_order") {
                  std::swap(table.labels.front(), table.labels.back());
              } else if (mutation == "duplicate_label") {
                  table.labels.insert(table.labels.begin(), table.labels.front());
                  table.label_entry_offsets.insert(table.label_entry_offsets.begin(), 0);
              } else if (mutation == "exchange_closure") {
                  table.labels.pop_back();
                  table.label_entry_offsets.pop_back();
                  table.label_entry_indices.resize(table.label_entry_offsets.back());
              } else if (mutation == "exchange_scalar") {
                  entry.scalar *= 1.5;
              } else if (mutation == "empty_label") {
                  for (auto& matrix : table.coupling_matrices)
                      std::fill(matrix.values.begin(), matrix.values.end(), 0.0);
              } else if (mutation == "offset_count") {
                  table.label_entry_offsets.pop_back();
              } else if (mutation == "offset_monotonic") {
                  std::swap(table.label_entry_offsets.at(1), table.label_entry_offsets.at(2));
              } else if (mutation == "entry_index_range") {
                  table.label_entry_indices.front() = table.entries.size();
              } else if (mutation == "missing_coupling_matrix") {
                  table.coupling_matrices.pop_back();
              } else if (mutation == "coupling_matrix_shape") {
                  table.coupling_matrices.front().values.pop_back();
              } else if (mutation == "isotropic_reduction") {
                  for (auto& item : table.entries)
                      if (item.order == 6) item.scalar *= 1.0 + 1.0e-6;
              } else if (mutation == "rotation_orthogonality") {
                  table.max_rotation_orthogonality_deviation = 1.0;
              } else if (mutation == "collapse_residual") {
                  table.max_collapse_residual = 1.0;
              } else {
                  throw PSIEXCEPTION("anisotropic table rejection: unknown mutation '" +
                                     mutation + "'");
              }
              detail::validate_anisotropic_recoupling_table(table);
              throw PSIEXCEPTION("anisotropic table rejection: mutation '" + mutation +
                                 "' was accepted by the loader");
          },
          "mutation"_a);
    m.def("_atomic_polarizability_test_anisotropic_energy_reconstruction",
          [anisotropic_tensor_series, anisotropic_site_axes, anisotropic_site_position](
              const std::vector<SharedMatrix>& first, const std::vector<SharedMatrix>& second,
              const std::vector<double>& weights, const SharedMatrix& first_frame,
              const SharedMatrix& second_frame, const std::vector<double>& direction,
              double distance) {
              const auto reconstruction = detail::anisotropic_energy_reconstruction(
                  anisotropic_tensor_series(first, "anisotropic energy reconstruction"),
                  anisotropic_tensor_series(second, "anisotropic energy reconstruction"), weights,
                  anisotropic_site_axes(first_frame, "anisotropic energy reconstruction"),
                  anisotropic_site_axes(second_frame, "anisotropic energy reconstruction"),
                  anisotropic_site_position(direction, "anisotropic energy reconstruction"),
                  distance);
              py::dict result;
              result["direct_energy"] = reconstruction.direct_energy;
              result["full_energy"] = reconstruction.full_energy;
              result["published_energy"] = reconstruction.published_energy;
              result["full_relative_deviation"] = reconstruction.full_relative_deviation;
              result["published_relative_deviation"] =
                  reconstruction.published_relative_deviation;
              result["max_s_function_imaginary"] = reconstruction.max_s_function_imaginary;
              result["full_label_count"] = reconstruction.full_label_count;
              result["published_label_count"] = reconstruction.published_label_count;
              return result;
          },
          "first"_a, "second"_a, "weights"_a, "first_frame"_a, "second_frame"_a, "direction"_a,
          "distance"_a);
}
