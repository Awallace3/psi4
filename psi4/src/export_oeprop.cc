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
              if (isa_option_values.contains("inject_tail_fit_failure_iteration")) {
                  inject_tail_fit_failure_iteration =
                      isa_option_values["inject_tail_fit_failure_iteration"].cast<std::size_t>();
                  isa_option_values.attr("pop")("inject_tail_fit_failure_iteration");
              }
              const auto result = detail::compute_synthetic_isa(
                  sites, points, weights, atomic_numbers, terms,
                  isa_options_from_dict(isa_option_values), inject_tail_fit_failure_iteration);
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
                                       matrix_sum_squares(context.Da()), matrix_sum_squares(context.Db()), context.energy()};
        });
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
