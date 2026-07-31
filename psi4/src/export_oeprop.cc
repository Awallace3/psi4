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

#include "psi4/libmints/oeprop.h"
#include "psi4/libmints/atomic_polarizability.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
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
    m.def(
        "_atomic_polarizability_validate_response_prerequisites",
        [](double chf_exchange, double alda_kernel, double neutral_energy, double cation_energy,
           double homo_energy, double ionization_potential, double grac_shift,
           const std::string& method_fingerprint, const std::string& functional_fingerprint,
           const std::string& basis_fingerprint, const std::string& grid_fingerprint,
           std::size_t point_count, std::size_t grid_dimension, std::size_t site_count,
           std::vector<double> points, std::vector<double> quadrature_weights,
           std::vector<double> partition_weights) {
            const ResponseKernel kernel(chf_exchange, alda_kernel);
            const GRACProvenance grac(neutral_energy, cation_energy, homo_energy,
                                      ionization_potential, grac_shift, method_fingerprint,
                                      functional_fingerprint, basis_fingerprint, grid_fingerprint);
            const ISAWeights isa(point_count, grid_dimension, site_count, std::move(points),
                                 std::move(quadrature_weights), std::move(partition_weights));
            py::dict result;
            result["kernel"] = py::make_tuple(kernel.chf_exchange(), kernel.alda_kernel());
            result["grac"] = py::make_tuple(grac.neutral_energy(), grac.cation_energy(),
                                            grac.homo_energy(), grac.ionization_potential(),
                                            grac.shift());
            result["fingerprints"] =
                py::make_tuple(grac.method_fingerprint(), grac.functional_fingerprint(),
                               grac.basis_fingerprint(), grac.grid_fingerprint());
            result["isa_dimensions"] =
                py::make_tuple(isa.point_count(), isa.grid_dimension(), isa.site_count());
            result["isa_data_sizes"] =
                py::make_tuple(isa.points().size(), isa.quadrature_weights().size(),
                               isa.partition_weights().size());
            return result;
        },
        "chf_exchange"_a, "alda_kernel"_a, "neutral_energy"_a, "cation_energy"_a,
        "homo_energy"_a, "ionization_potential"_a, "grac_shift"_a,
        "method_fingerprint"_a, "functional_fingerprint"_a, "basis_fingerprint"_a,
        "grid_fingerprint"_a, "point_count"_a, "grid_dimension"_a, "site_count"_a,
        "points"_a, "quadrature_weights"_a, "partition_weights"_a);
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
