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

#include "psi4/libisapol/casimir_grid.h"
#include "psi4/libisapol/fit_points.h"
#include "psi4/libisapol/isa_grid.h"
#include "psi4/libisapol/isa_fit.h"
#include "psi4/libisapol/explicit_basis.h"
#include "psi4/libisapol/aux_coulomb.h"
#include "psi4/libisapol/isa_sweep.h"
#include "psi4/libisapol/isa_shape.h"
#include "psi4/libisapol/isa_controller.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libisapol/recoupling_tables.h"
#include "psi4/libisapol/tables.h"
#include "psi4/libmints/molecule.h"

#include <pybind11/numpy.h>

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
    py::class_<IsaAuxCoulomb>(m, "IsaAuxCoulomb", "Native Libint2 Coulomb metric and analytic charges; explicit Cartesian molecular AUX only")
        .def(py::init<const IsaExplicitBasis&>(), "auxiliary"_a)
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
    py::class_<IsaAController>(m, "IsaAController", "Explicit-input ordinary-A controller, not native wavefunction-to-property parity")
        .def(py::init<const std::vector<IsaExplicitBasis>&, const std::vector<IsaExplicitBasis>&,
                     const std::vector<std::vector<int>>&, const IsaFixedDensity&,
                     const std::vector<IsaNoTailGrid>&, const IsaAControllerOptions&>(),
             "atomic"_a, "shape"_a, "shell_maps"_a, "density"_a, "grids"_a, "options"_a)
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
