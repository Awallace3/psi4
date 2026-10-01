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

#include "psi4/libisapol/casimir_grid.h"
#include "psi4/libisapol/isa_grid.h"
#include "psi4/libisapol/explicit_basis.h"
#include "psi4/libisapol/partitioned_response.h"
#include "psi4/libisapol/multipole_transform.h"
#include "psi4/libisapol/lw_localization.h"
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
    if (rank_limit < 1 || rank_limit > static_cast<int>(kIsaLwMaxRank))
        throw std::runtime_error("localize_lw: declared rank_limit must be 1, 2, 3 or 4");
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
        throw std::runtime_error(
            "localize_lw: expected one 16 by 16 or 25 by 25 block for every ordered site pair");
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
    // The supplied block width DECLARES the rank of the caller's data: 16 for rank 3,
    // 25 for rank 4. It is not a storage detail to be inferred away -- a rank-3 caller
    // may not silently declare a rank-4 localization on data that has no rank-4
    // components. Rank-3 input is zero-extended into the wider working matrix, which
    // leaves every rank <= 3 declaration bitwise unchanged by the widening.
    if (checked.empty() || !checked.front())
        throw std::runtime_error("localize_lw: expected 16 by 16 or 25 by 25 single-block matrices");
    const int supplied_width = checked.front()->nrow();
    if (supplied_width != 16 && supplied_width != static_cast<int>(kIsaLwWorkingComponents))
        throw std::runtime_error("localize_lw: expected 16 by 16 or 25 by 25 single-block matrices");
    const auto width = static_cast<std::size_t>(supplied_width);
    const int supplied_rank = supplied_width == 16 ? 3 : static_cast<int>(kIsaLwMaxRank);
    if (rank_limit > supplied_rank)
        throw std::runtime_error(
            "localize_lw: declared rank_limit exceeds the rank of the supplied blocks; "
            "rank 4 localization requires 25 by 25 supplied blocks");
    for (const auto& matrix : checked) {
        if (!matrix || matrix->nirrep() != 1 || matrix->nrow() != supplied_width ||
            matrix->ncol() != supplied_width)
            throw std::runtime_error(
                "localize_lw: expected 16 by 16 or 25 by 25 single-block matrices, all the same width");
        for (std::size_t row = 0; row < width; ++row)
            for (std::size_t column = 0; column < width; ++column)
                if (!std::isfinite((*matrix)(row, column)))
                    throw std::runtime_error("localize_lw: response values must be finite");
    }
    IsaSitePairResponse response;
    response.frequency = frequency;
    response.positions.resize(count);
    for (std::size_t site = 0; site < count; ++site)
        for (std::size_t axis = 0; axis < 3; ++axis)
            response.positions[site][axis] = positions(site, axis);
    // Value-initialized: components the supplied width does not reach stay at zero.
    response.blocks.resize(count * count);
    for (std::size_t block = 0; block < checked.size(); ++block)
        for (std::size_t row = 0; row < width; ++row)
            for (std::size_t column = 0; column < width; ++column)
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
        }, "Full 25x25 LW workspace; NOT externally PFIT-refined pairs")
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
          "Positions: N x 3 Matrix in bohr; blocks: N*N single 16x16 (rank 3) or 25x25 (rank 4) "
          "Matrices, all the same width, real Racah 00,10,11c,11s,... . The width declares the rank of "
          "the supplied data and rank_limit may not exceed it; rank-3 blocks are zero-extended. "
          "Explicit zero-based graph; source-minus-target translations; local output is 24x24 with "
          "ranks above the declared limit identically zero. "
          "residual_tolerance is a postcondition gate, not iteration; it always holds the "
          "algorithm-controlled residuals off_site, reciprocity, molecular_sum and charge_sum_transport. "
          "input_sum_rule_tolerance gates the supplied data's charge-flow sum-rule defect separately: "
          "negative inherits residual_tolerance (one combined gate), positive finite sets an explicit "
          "threshold, and infinity measures and reports the defect without gating it. LW transports such a "
          "defect exactly (measured <= 2.2e-16) and cannot repair it. At most 256 sites, 1000000 retained transfers, "
          "768 MiB native workspace budget (caller inputs/getter copies additional). The budget is unchanged "
          "by the rank-4 widening and the 25x25 working matrix therefore lowers the largest admissible graph "
          "from 256 to about 174 sites; that cost is reported, not hidden by raising the budget. "
          "rank_limit declares the rank the localization runs at, in 1..4, default 3 (for which this "
          "routine is unchanged; 4 is the full working space). A declared limit L truncates the supplied blocks "
          "to the leading (L+1)^2 real Racah components in both index slots first -- reported through "
          "truncated_input_maxabs -- and then runs the component-pair loop, the translated transfer "
          "application, the molecular-sum conservation check and the local output inside that space. The "
          "restriction is exact because multipole translation is rank-raising, so no tolerance is relaxed "
          "and every residual is gated at the same threshold as at the full rank. A localization at one limit is a "
          "DIFFERENT MODEL from one at another, but an exactly consistent one: because translation is "
          "rank-raising and the pair loop is ordered, no pair above the limit can write below it, so the "
          "result at L equals the result at any higher L' restricted to the declared space, bitwise "
          "(measured 0.0). A "
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

    m.def("isapol_slater_radius", &slater_radius, "Z"_a,
          "CamCASP Bragg-Slater radius in bohr (a_o = 0.529177249)");
    m.def("isapol_vdw_radius_bondi", &vdw_radius_bondi, "Z"_a,
          "Bondi van der Waals radius in bohr, from AtomProp (float32-rounded)");
    m.def("isapol_vdw_radius_grimme", &vdw_radius_grimme, "Z"_a, "Grimme van der Waals radius in bohr");
    m.def("isapol_c6_grimme", &c6_grimme, "Z"_a, "Grimme C6 coefficient in atomic units");
    m.def("isapol_covalent_radius", &covalent_radius, "Z"_a, "Covalent radius in bohr");
    m.def("isapol_element_symbol", &element_symbol, "Z"_a);
}
