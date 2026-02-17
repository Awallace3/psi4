/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2024 The Psi4 Developers.
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

// XDM dispersion correction — main driver.
// Translates postg's evalwfn + propts + edisp subroutines (wfnmod.f90).

#include "xdm.h"
#include "becke_roussel.h"
#include "hirshfeld.h"
#include "xdm_data.h"

#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace psi {
namespace xdm {

static constexpr double PI = 3.141592653589793;
static constexpr double BOHR_TO_ANGSTROM = 0.52917720859;

// ============================================================================
// BJ parameter lookup table: (functional, basis) -> (a1, a2 in angstrom)
// From postg reference data (xdm.param).
// ============================================================================
struct BJParams {
    double a1;
    double a2_ang;
};

static const std::map<std::string, BJParams>& bj_param_table() {
    static const std::map<std::string, BJParams> table = {
        // B3LYP
        {"b3lyp/6-31+g*", {0.6356, 1.5119}},
        {"b3lyp/6-311+g(2d,2p)", {0.6535, 1.4499}},
        {"b3lyp/aug-cc-pvdz", {0.6514, 1.4399}},
        {"b3lyp/aug-cc-pvtz", {0.6356, 1.5119}},
        {"b3lyp/aug-cc-pvqz", {0.6352, 1.5306}},

        // PW86PBE
        {"pw86pbe/6-31+g*", {0.6836, 1.5045}},
        {"pw86pbe/6-311+g(2d,2p)", {0.7017, 1.4502}},
        {"pw86pbe/aug-cc-pvdz", {0.6985, 1.4407}},
        {"pw86pbe/aug-cc-pvtz", {0.6836, 1.5045}},
        {"pw86pbe/aug-cc-pvqz", {0.6827, 1.5228}},

        // PBE
        {"pbe/6-31+g*", {0.4492, 2.2834}},
        {"pbe/6-311+g(2d,2p)", {0.4687, 2.1673}},
        {"pbe/aug-cc-pvdz", {0.4675, 2.1574}},
        {"pbe/aug-cc-pvtz", {0.4492, 2.2834}},
        {"pbe/aug-cc-pvqz", {0.4480, 2.3087}},

        // PBE0
        {"pbe0/6-31+g*", {0.4186, 2.2648}},
        {"pbe0/6-311+g(2d,2p)", {0.4378, 2.1442}},
        {"pbe0/aug-cc-pvdz", {0.4360, 2.1344}},
        {"pbe0/aug-cc-pvtz", {0.4186, 2.2648}},
        {"pbe0/aug-cc-pvqz", {0.4175, 2.2903}},

        // BLYP
        {"blyp/6-31+g*", {0.6512, 1.4155}},
        {"blyp/6-311+g(2d,2p)", {0.6660, 1.3527}},
        {"blyp/aug-cc-pvdz", {0.6639, 1.3432}},
        {"blyp/aug-cc-pvtz", {0.6512, 1.4155}},
        {"blyp/aug-cc-pvqz", {0.6499, 1.4338}},

        // BHAHLYP (BHandHLYP, 50% HF)
        {"bhahlyp/aug-cc-pvtz", {0.5956, 1.5972}},
        {"bhandh/aug-cc-pvtz", {0.5956, 1.5972}},
        {"bhandhlyp/aug-cc-pvtz", {0.5956, 1.5972}},

        // CAM-B3LYP
        {"cam-b3lyp/aug-cc-pvtz", {0.5929, 1.4933}},

        // LC-wPBE
        {"lc-wpbe/aug-cc-pvtz", {0.3984, 2.2747}},
        {"lcwpbe/aug-cc-pvtz", {0.3984, 2.2747}},

        // B971 (B97-1)
        {"b971/aug-cc-pvtz", {0.5991, 1.5700}},
        {"b97-1/aug-cc-pvtz", {0.5991, 1.5700}},
    };
    return table;
}

// ============================================================================
// Constructor and factory methods
// ============================================================================

XDMDispersion::XDMDispersion(double a1, double a2_bohr, const std::string& functional_name)
    : a1_(a1), a2_(a2_bohr), functional_name_(functional_name) {}

std::shared_ptr<XDMDispersion> XDMDispersion::build(const std::string& functional, const std::string& basis) {
    // Normalize names to lowercase
    std::string func_lower = functional;
    std::string basis_lower = basis;
    std::transform(func_lower.begin(), func_lower.end(), func_lower.begin(), ::tolower);
    std::transform(basis_lower.begin(), basis_lower.end(), basis_lower.begin(), ::tolower);

    std::string key = func_lower + "/" + basis_lower;
    const auto& table = bj_param_table();
    auto it = table.find(key);
    if (it == table.end()) {
        throw PSIEXCEPTION("XDMDispersion::build: No fitted BJ parameters for " + key +
                           ". Use build(functional, a1, a2) to specify parameters explicitly.");
    }

    double a2_bohr = it->second.a2_ang / BOHR_TO_ANGSTROM;
    return std::make_shared<XDMDispersion>(it->second.a1, a2_bohr, func_lower);
}

std::shared_ptr<XDMDispersion> XDMDispersion::build(const std::string& functional, double a1,
                                                     double a2_angstrom) {
    std::string func_lower = functional;
    std::transform(func_lower.begin(), func_lower.end(), func_lower.begin(), ::tolower);
    double a2_bohr = a2_angstrom / BOHR_TO_ANGSTROM;
    return std::make_shared<XDMDispersion>(a1, a2_bohr, func_lower);
}

// ============================================================================
// Energy and gradient
// ============================================================================

double XDMDispersion::compute_energy(std::shared_ptr<Wavefunction> wfn) {
    auto atoms = integrate_properties(wfn);
    return pairwise_energy(wfn->molecule(), atoms);
}

SharedMatrix XDMDispersion::compute_gradient(std::shared_ptr<Wavefunction> wfn) {
    auto atoms = integrate_properties(wfn);
    auto mol = wfn->molecule();
    int natom = mol->natom();
    auto grad = std::make_shared<Matrix>("XDM Gradient", natom, 3);
    pairwise_energy(mol, atoms, grad);
    return grad;
}

// ============================================================================
// Core: integrate Hirshfeld-weighted atomic properties
// ============================================================================

std::vector<AtomicData> XDMDispersion::integrate_properties(std::shared_ptr<Wavefunction> wfn) {
    auto mol = wfn->molecule();
    auto primary = wfn->basisset();
    auto& options = wfn->options();
    int natom = mol->natom();
    bool restricted = wfn->same_a_b_dens();

    // Get density matrices in AO basis (wfn stores them in SO basis)
    SharedMatrix Da = wfn->Da_subset("AO");
    SharedMatrix Db = restricted ? Da : wfn->Db_subset("AO");

    // --- Build DFT grid with ATOMIC blockscheme ---
    std::map<std::string, int> int_opts;
    std::map<std::string, std::string> str_opts;
    str_opts["DFT_BLOCK_SCHEME"] = "ATOMIC";

    auto grid = std::make_shared<DFTGrid>(mol, primary, int_opts, str_opts, options);

    int max_points = grid->max_points();
    int max_functions = grid->max_functions();

    // --- Build PointFunctions for density evaluation ---
    std::shared_ptr<PointFunctions> pf;
    if (restricted) {
        auto rks = std::make_shared<RKSFunctions>(primary, max_points, max_functions);
        rks->set_ansatz(2);  // meta-GGA: gives rho, gamma, tau + second derivs of basis
        rks->set_pointers(Da);
        pf = rks;
    } else {
        auto uks = std::make_shared<UKSFunctions>(primary, max_points, max_functions);
        uks->set_ansatz(2);
        uks->set_pointers(Da, Db);
        pf = uks;
    }

    // --- Precompute Hirshfeld weights for all grid points ---
    int npoints_total = grid->npoints();

    // Collect atom data
    std::vector<int> atomic_nums(natom);
    std::vector<std::array<double, 3>> atom_coords(natom);
    for (int a = 0; a < natom; a++) {
        atomic_nums[a] = mol->true_atomic_number(a);
        atom_coords[a][0] = mol->x(a);
        atom_coords[a][1] = mol->y(a);
        atom_coords[a][2] = mol->z(a);
    }

    // Compute Hirshfeld weights over all grid points
    ProatomDensity proatom;
    std::vector<std::vector<double>> hirshfeld_weights;
    compute_hirshfeld_weights(proatom, natom, atomic_nums.data(),
                              reinterpret_cast<const double(*)[3]>(atom_coords.data()), npoints_total,
                              grid->x(), grid->y(), grid->z(), hirshfeld_weights);

    // --- Allocate per-atom accumulators ---
    std::vector<AtomicData> atom_data(natom);

    // Temporary buffers for Laplacian computation (reused across blocks)
    auto T_lapl = std::make_shared<Matrix>("T_lapl", max_points, max_functions);
    auto D_local = std::make_shared<Matrix>("D_local", max_functions, max_functions);

    // --- Loop over grid blocks ---
    const auto& blocks = grid->blocks();
    int point_offset = 0;

    for (size_t ib = 0; ib < blocks.size(); ib++) {
        auto block = blocks[ib];
        int npoints = block->npoints();
        int nlocal = block->local_nbf();
        const auto& function_map = block->functions_local_to_global();

        // Compute density at grid points (rho, gamma, tau)
        pf->compute_points(block);

        // Get grid coordinates and weights
        double* xp = block->x();
        double* yp = block->y();
        double* zp = block->z();
        double* wp = block->w();

        // Get density quantities from PointFunctions
        // RKS: RHO_A = rho_total, GAMMA_AA = |grad rho_alpha|^2, TAU_A = tau_nohalf_alpha
        // UKS: RHO_A = rho_alpha, GAMMA_AA = |grad rho_alpha|^2, TAU_A = 0.5*tau_nohalf_alpha
        double* rho_a_p;
        double* rho_b_p;
        double* gamma_aa_p;
        double* gamma_bb_p;
        double* tau_a_p;
        double* tau_b_p;

        if (restricted) {
            rho_a_p = pf->point_value("RHO_A")->pointer();
            rho_b_p = rho_a_p;
            gamma_aa_p = pf->point_value("GAMMA_AA")->pointer();
            gamma_bb_p = gamma_aa_p;
            tau_a_p = pf->point_value("TAU_A")->pointer();
            tau_b_p = tau_a_p;
        } else {
            rho_a_p = pf->point_value("RHO_A")->pointer();
            rho_b_p = pf->point_value("RHO_B")->pointer();
            gamma_aa_p = pf->point_value("GAMMA_AA")->pointer();
            gamma_bb_p = pf->point_value("GAMMA_BB")->pointer();
            tau_a_p = pf->point_value("TAU_A")->pointer();
            tau_b_p = pf->point_value("TAU_B")->pointer();
        }

        // --- Compute Laplacian of spin density ---
        // lapl_sigma = 2 * sum_{c=x,y,z} phi_cc . (D_sigma * phi) + 2 * tau_nohalf_sigma
        // where tau_nohalf = sum D_sigma * phi_c * phi_c (without the 0.5 factor)

        double** phip = pf->basis_value("PHI")->pointer();
        double** phi_xx = pf->basis_value("PHI_XX")->pointer();
        double** phi_yy = pf->basis_value("PHI_YY")->pointer();
        double** phi_zz = pf->basis_value("PHI_ZZ")->pointer();
        size_t coll_funcs = pf->basis_value("PHI")->ncol();
        int nglobal = max_functions;

        // Per-point storage for b parameters and spin densities
        std::vector<double> b_alpha(npoints, 0.0);
        std::vector<double> b_beta(npoints, 0.0);
        std::vector<double> rho_alpha(npoints, 0.0);
        std::vector<double> rho_beta(npoints, 0.0);

        // Process alpha spin (and beta if unrestricted)
        for (int spin = 0; spin < (restricted ? 1 : 2); spin++) {
            SharedMatrix D_AO = (spin == 0) ? Da : Db;

            // Build local density matrix (sieve to relevant functions in this block)
            double** Dp = D_AO->pointer();
            double** D2p = D_local->pointer();
            for (int ml = 0; ml < nlocal; ml++) {
                int mg = function_map[ml];
                for (int nl = 0; nl <= ml; nl++) {
                    int ng = function_map[nl];
                    double Dval = Dp[mg][ng];
                    D2p[ml][nl] = Dval;
                    D2p[nl][ml] = Dval;
                }
            }

            // T0 = D * phi^T  (half-transform)
            double** Tp = T_lapl->pointer();
            C_DGEMM('N', 'N', npoints, nlocal, nlocal, 1.0, phip[0], coll_funcs, D2p[0], nglobal, 0.0,
                    Tp[0], nglobal);

            // Diagonal Laplacian contribution: 2 * sum_c phi_cc . T0
            std::vector<double> lapl(npoints, 0.0);
            for (int P = 0; P < npoints; P++) {
                lapl[P] = 2.0 * (C_DDOT(nlocal, phi_xx[P], 1, Tp[P], 1) +
                                  C_DDOT(nlocal, phi_yy[P], 1, Tp[P], 1) +
                                  C_DDOT(nlocal, phi_zz[P], 1, Tp[P], 1));
            }

            // Add cross terms: 2 * tau_nohalf
            // For RKS alpha: TAU_A = tau_nohalf_alpha, so add 2*TAU_A
            // For UKS: TAU = 0.5*tau_nohalf, so tau_nohalf = 2*TAU, add 2*(2*TAU) = 4*TAU
            double* tau_p = (spin == 0) ? tau_a_p : tau_b_p;
            double tau_factor = restricted ? 2.0 : 4.0;
            for (int P = 0; P < npoints; P++) {
                lapl[P] += tau_factor * tau_p[P];
            }

            // Compute per-point spin density, gradient squared, tau (in postg convention)
            double* rho_p = (spin == 0) ? rho_a_p : rho_b_p;
            double* gamma_p = (spin == 0) ? gamma_aa_p : gamma_bb_p;

            for (int P = 0; P < npoints; P++) {
                double rho_spin, grad_sq, tau_nohalf;

                if (restricted) {
                    // RKS: RHO_A = rho_total, GAMMA_AA = |grad rho_alpha|^2, TAU_A = tau_nohalf_alpha
                    rho_spin = rho_p[P] * 0.5;
                    grad_sq = gamma_p[P];        // already |grad rho_alpha|^2
                    tau_nohalf = tau_p[P];       // already tau_nohalf for alpha
                } else {
                    // UKS: RHO_A = rho_alpha, GAMMA_AA = |grad rho_alpha|^2, TAU_A = 0.5*tau_nohalf
                    rho_spin = rho_p[P];
                    grad_sq = gamma_p[P];
                    tau_nohalf = 2.0 * tau_p[P]; // convert from half to nohalf convention
                }

                if (rho_spin < 1.0e-15) {
                    if (spin == 0) {
                        rho_alpha[P] = rho_spin;
                        b_alpha[P] = 0.0;
                    } else {
                        rho_beta[P] = rho_spin;
                        b_beta[P] = 0.0;
                    }
                    continue;
                }

                // Exchange-hole curvature Q
                double Q = exchange_hole_curvature(rho_spin, grad_sq, tau_nohalf, lapl[P]);

                // Becke-Roussel b parameter
                double b;
                try {
                    b = bhole(rho_spin, Q, 1.0);
                } catch (...) {
                    b = 0.0;
                }

                if (spin == 0) {
                    rho_alpha[P] = rho_spin;
                    b_alpha[P] = b;
                } else {
                    rho_beta[P] = rho_spin;
                    b_beta[P] = b;
                }
            }

            // For restricted: beta = alpha
            if (restricted) {
                rho_beta = rho_alpha;
                b_beta = b_alpha;
            }
        }

        // --- Integrate XDM moments using Hirshfeld weights ---
        for (int a = 0; a < natom; a++) {
            if (atomic_nums[a] < 1) continue;

            double ax = atom_coords[a][0];
            double ay = atom_coords[a][1];
            double az = atom_coords[a][2];

            for (int P = 0; P < npoints; P++) {
                int gp = point_offset + P;  // global point index for Hirshfeld weights
                double h = hirshfeld_weights[a][gp];
                double w = wp[P];

                double dx = xp[P] - ax;
                double dy = yp[P] - ay;
                double dz = zp[P] - az;
                double r = std::sqrt(dx * dx + dy * dy + dz * dz);

                double r1_a = std::max(0.0, r - b_alpha[P]);
                double r1_b = std::max(0.0, r - b_beta[P]);

                double rho_a = rho_alpha[P];
                double rho_b = rho_beta[P];
                double rho_tot = rho_a + rho_b;

                // M1^2: exchange-hole dipole moment squared
                atom_data[a].mm1 += w * h * (rho_a * (r - r1_a) * (r - r1_a) +
                                              rho_b * (r - r1_b) * (r - r1_b));

                // M2^2: exchange-hole quadrupole moment squared
                atom_data[a].mm2 += w * h * (rho_a * (r * r - r1_a * r1_a) * (r * r - r1_a * r1_a) +
                                              rho_b * (r * r - r1_b * r1_b) * (r * r - r1_b * r1_b));

                // M3^2: exchange-hole octupole moment squared
                double r3 = r * r * r;
                double r1a3 = r1_a * r1_a * r1_a;
                double r1b3 = r1_b * r1_b * r1_b;
                atom_data[a].mm3 += w * h * (rho_a * (r3 - r1a3) * (r3 - r1a3) +
                                              rho_b * (r3 - r1b3) * (r3 - r1b3));

                // Effective atomic volume
                atom_data[a].vol += w * h * rho_tot * r3;

                // Integrated charge
                atom_data[a].charge += w * h * rho_tot;
            }
        }

        point_offset += npoints;
    }

    // Print summary
    outfile->Printf("\n  ==> XDM Atomic Properties <==\n\n");
    outfile->Printf("    %5s %8s %12s %12s %12s %12s %12s\n", "Atom", "Z", "Charge", "Volume", "M1^2", "M2^2",
                    "M3^2");
    for (int a = 0; a < natom; a++) {
        outfile->Printf("    %5d %8d %12.6f %12.6f %12.6f %12.6f %12.6f\n", a + 1, atomic_nums[a],
                        atom_data[a].charge, atom_data[a].vol, atom_data[a].mm1, atom_data[a].mm2,
                        atom_data[a].mm3);
    }
    outfile->Printf("\n");

    return atom_data;
}

// ============================================================================
// Pairwise BJ-damped dispersion energy and gradient
// ============================================================================

double XDMDispersion::pairwise_energy(std::shared_ptr<Molecule> mol, const std::vector<AtomicData>& atoms,
                                       SharedMatrix gradient) {
    int natom = mol->natom();

    // Compute effective atomic polarizabilities
    std::vector<double> atpol(natom, 0.0);
    for (int i = 0; i < natom; i++) {
        int Z = mol->true_atomic_number(i);
        if (Z < 1) continue;

        double alpha_free = get_free_polarizability(Z);
        double vol_free = get_free_volume(Z, functional_name_);

        if (vol_free > 1.0e-10 && atoms[i].vol > 1.0e-10) {
            atpol[i] = atoms[i].vol * alpha_free / vol_free;
        }
    }

    // Print polarizabilities
    outfile->Printf("  ==> XDM Atomic Polarizabilities <==\n\n");
    outfile->Printf("    %5s %8s %16s\n", "Atom", "Z", "Polarizability");
    for (int i = 0; i < natom; i++) {
        outfile->Printf("    %5d %8d %16.6f\n", i + 1, mol->true_atomic_number(i), atpol[i]);
    }
    outfile->Printf("\n");

    // Compute pairwise dispersion
    double e_disp = 0.0;
    double** gp = gradient ? gradient->pointer() : nullptr;

    outfile->Printf("  ==> XDM Pairwise Coefficients <==\n\n");
    outfile->Printf("    %4s %4s %12s %16s %16s %16s %12s %12s\n", "i", "j", "dij", "C6", "C8", "C10", "Rc",
                    "Rvdw");

    for (int i = 0; i < natom; i++) {
        int Zi = mol->true_atomic_number(i);
        if (Zi < 1) continue;
        for (int j = i + 1; j < natom; j++) {
            int Zj = mol->true_atomic_number(j);
            if (Zj < 1) continue;

            // Interatomic distance
            double xij = mol->x(j) - mol->x(i);
            double yij = mol->y(j) - mol->y(i);
            double zij = mol->z(j) - mol->z(i);
            double d = std::sqrt(xij * xij + yij * yij + zij * zij);

            if (d < 1.0e-10) continue;

            // Dispersion coefficients
            double denom = atoms[i].mm1 * atpol[j] + atoms[j].mm1 * atpol[i];
            if (denom < 1.0e-30) continue;
            double fac = atpol[i] * atpol[j] / denom;

            double c6 = fac * atoms[i].mm1 * atoms[j].mm1;
            double c8 = 1.5 * fac * (atoms[i].mm1 * atoms[j].mm2 + atoms[i].mm2 * atoms[j].mm1);
            double c10 = 2.0 * fac * (atoms[i].mm1 * atoms[j].mm3 + atoms[i].mm3 * atoms[j].mm1) +
                         4.2 * fac * atoms[i].mm2 * atoms[j].mm2;

            // Critical radius and vdW radius (BJ damping)
            double rc = (std::sqrt(c8 / c6) + std::sqrt(std::sqrt(c10 / c6)) + std::sqrt(c10 / c8)) / 3.0;
            double rvdw = a1_ * rc + a2_;

            double rvdw6 = std::pow(rvdw, 6);
            double rvdw8 = std::pow(rvdw, 8);
            double rvdw10 = std::pow(rvdw, 10);

            double d2 = d * d;
            double d6 = d2 * d2 * d2;
            double d8 = d6 * d2;
            double d10 = d8 * d2;

            // Dispersion energy
            e_disp -= c6 / (d6 + rvdw6) + c8 / (d8 + rvdw8) + c10 / (d10 + rvdw10);

            // Gradient (geometry-only, coefficients treated as constants)
            if (gp) {
                double c6com = 6.0 * c6 * d2 * d2 / ((d6 + rvdw6) * (d6 + rvdw6));
                double c8com = 8.0 * c8 * d6 / ((d8 + rvdw8) * (d8 + rvdw8));
                double c10com = 10.0 * c10 * d8 / ((d10 + rvdw10) * (d10 + rvdw10));

                double fgrad = c6com + c8com + c10com;

                // Force on i: +fgrad * (xj - xi)/d^2... but actually the derivative of
                // -Cn/(d^n + rvdw^n) w.r.t. xi gives: n*Cn*d^(n-2)/(d^n+rvdw^n)^2 * (xj-xi)
                // which is the force pointing from i toward j (positive contribution to gradient of i)
                // Wait: gradient = dE/dR_i. E = -Cn/(d^n + rvdw^n)
                // dE/dx_i = n*Cn*d^(n-2)/(d^n+rvdw^n)^2 * (x_i - x_j)
                // = -fgrad * xij (since xij = xj - xi)
                gp[i][0] -= fgrad * xij;
                gp[i][1] -= fgrad * yij;
                gp[i][2] -= fgrad * zij;
                gp[j][0] += fgrad * xij;
                gp[j][1] += fgrad * yij;
                gp[j][2] += fgrad * zij;
            }

            outfile->Printf("    %4d %4d %12.6f %16.9E %16.9E %16.9E %12.6f %12.6f\n", i + 1, j + 1, d, c6, c8,
                            c10, rc, rvdw);
        }
    }

    outfile->Printf("\n");
    outfile->Printf("  XDM Dispersion Energy: %20.12f [Eh]\n\n", e_disp);

    return e_disp;
}

}  // namespace xdm
}  // namespace psi
