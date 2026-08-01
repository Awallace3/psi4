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
#include "psi4/libmints/oeprop.h"
#include "psi4/libmints/atomic_polarizability.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libscf_solver/hf.h"
#include "psi4/libpsi4util/exception.h"

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
    m.def("_atomic_polarizability_solve_restricted_response",
          [](const Matrix& H1, const Matrix& H2, double omega, const Matrix& rhs) {
              const auto result = detail::solve_dense_restricted_response(H1, H2, omega, rhs);
              py::dict values;
              values["P"] = result.P;
              values["Q"] = result.Q;
              values["reciprocal_condition"] = result.reciprocal_condition;
              values["reciprocal_pivot_growth"] = result.reciprocal_pivot_growth;
              values["max_forward_error"] = result.max_forward_error;
              values["max_backward_error"] = result.max_backward_error;
              values["max_scaled_residual"] = result.max_scaled_residual;
              return values;
          },
          "H1"_a, "H2"_a, "omega"_a, "rhs"_a);
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
            result["reciprocity_enforced"] = contraction.reciprocity_enforced;
            result["plan"] = std::move(plan);
            return result;
        };
    m.def("_atomic_polarizability_test_contract_site_pair_response",
          [site_pair_contraction_result_dict](std::size_t site_count,
                                               const Matrix& projection,
                                               const Matrix& response_map) {
              if (response_map.nirrep() != 1 || response_map.nrow() <= 0 ||
                  response_map.nrow() != response_map.ncol())
                  throw PSIEXCEPTION(
                      "site-pair response contraction: exact test response map must be nonempty and square");
              for (int row = 0; row < response_map.nrow(); ++row)
                  for (int column = row + 1; column < response_map.ncol(); ++column)
                      if (response_map(row, column) != response_map(column, row))
                          throw PSIEXCEPTION(
                              "site-pair response contraction: test response map must be exactly symmetric");
              auto q = std::make_shared<Matrix>(response_map.nrow(), response_map.ncol());
              detail::DenseRestrictedResponse response{
                  response_map.clone(), std::move(q), 1.0, 1.0, 0.0, 0.0, 0.0};
              return site_pair_contraction_result_dict(
                  detail::contract_site_pair_response(site_count, projection, response));
          }, "site_count"_a, "projection"_a, "response_map"_a);
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
              result["P"] = response.P;
              result["Q"] = response.Q;
              result["reciprocal_condition"] = response.reciprocal_condition;
              result["reciprocal_pivot_growth"] = response.reciprocal_pivot_growth;
              result["max_forward_error"] = response.max_forward_error;
              result["max_backward_error"] = response.max_backward_error;
              result["max_scaled_residual"] = response.max_scaled_residual;
              return result;
          }, "site_count"_a, "projection"_a, "H1"_a, "H2"_a, "omega"_a);
    m.def("_atomic_polarizability_test_validate_response_map_symmetry",
          [](const Matrix& response_map, const Matrix& conjugate_map,
             double response_map_forward_error_bound) {
              const auto diagnostics = detail::validate_response_map_symmetry_test_only(
                  response_map, conjugate_map, response_map_forward_error_bound);
              py::dict result;
              result["response_map_forward_error_bound"] = response_map_forward_error_bound;
              result["response_map_solution_scale"] = diagnostics.solution_scale;
              result["response_map_allowed_antisymmetry"] = diagnostics.allowed_antisymmetry;
              result["response_map_symmetry_residual"] = diagnostics.symmetry_residual;
              return result;
          }, "response_map"_a, "conjugate_map"_a,
          "response_map_forward_error_bound"_a);
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
    py::class_<ISAPolResponseProvider, std::shared_ptr<ISAPolResponseProvider>>(
        m, "_AtomicPolarizabilityTestResponseProvider")
        .def("expected_response_count",
             [](const ISAPolResponseProvider& provider, std::vector<double> frequencies,
                std::vector<double> weights) {
                 return provider.expected_response_count(FrequencyGrid{std::move(frequencies), std::move(weights)});
             })
        .def("compute", [](const ISAPolResponseProvider& provider, std::vector<double> frequencies,
                           std::vector<double> weights) {
            return provider.compute_isapol_response(FrequencyGrid{std::move(frequencies), std::move(weights)});
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
             const std::vector<std::array<std::size_t, 2>>& bonds, double residual_tolerance) {
              if (positions.nirrep() != 1 || positions.ncol() != 3) {
                  throw PSIEXCEPTION("localize_lw: positions must be an N by 3 matrix");
              }
              SitePairResponse response;
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

    py::class_<AtomicPolarizabilityCalculator>(m, "AtomicPolarizabilityCalculator",
                                               "Native atomic-polarizability pipeline entry point")
        .def(py::init<std::shared_ptr<Wavefunction>>())
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
             "be inserted at every occurrence of {}, like Python format strings. Wipes other names-to-save-by.");

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
}
