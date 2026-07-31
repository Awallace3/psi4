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
    m.def("_atomic_polarizability_make_test_response_provider",
          [](const std::shared_ptr<FrozenResponseContext>& context,
             const std::shared_ptr<FrozenResponseContext>& isa_context) {
              const auto count = isa_context->grid_point_count() * isa_context->sites().size();
              std::vector<double> weights(count, 1.0 / static_cast<double>(isa_context->sites().size()));
              auto isa = ISAWeights::create(isa_context, std::move(weights));
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
