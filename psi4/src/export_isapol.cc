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

#include "psi4/libisapol/isa_grid.h"
#include "psi4/libisapol/explicit_basis.h"
#include "psi4/libisapol/partitioned_response.h"
#include "psi4/libmints/matrix.h"
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
    m.def("isa_regular_multipoles", &isa_regular_multipoles, "rank"_a, "displacement"_a,
          "r^k C_kq for every rank through rank, Racah 00,10,11c,11s,...; rank zero is 1");
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

    m.def("isapol_slater_radius", &slater_radius, "Z"_a,
          "CamCASP Bragg-Slater radius in bohr (a_o = 0.529177249)");
    m.def("isapol_vdw_radius_bondi", &vdw_radius_bondi, "Z"_a,
          "Bondi van der Waals radius in bohr, from AtomProp (float32-rounded)");
    m.def("isapol_vdw_radius_grimme", &vdw_radius_grimme, "Z"_a, "Grimme van der Waals radius in bohr");
    m.def("isapol_c6_grimme", &c6_grimme, "Z"_a, "Grimme C6 coefficient in atomic units");
    m.def("isapol_covalent_radius", &covalent_radius, "Z"_a, "Covalent radius in bohr");
    m.def("isapol_element_symbol", &element_symbol, "Z"_a);
}
