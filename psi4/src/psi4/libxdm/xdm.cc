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
#include "xdm_gpu.h"

#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/physconst.h"
#include "psi4/psi4-dec.h"

#ifdef USING_cuEST
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/vector3.h"

#include "gau2grid/gau2grid.h"  // GG_SPHERICAL_* orders
#include <libint2/config.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace psi {
namespace xdm {

namespace {
// psi4 timers stay "on" if an exception skips timer_off, which breaks the next call.
struct ScopedTimer {
    const char* name;
    explicit ScopedTimer(const char* n) : name(n) { timer_on(name); }
    ~ScopedTimer() { timer_off(name); }
};

#ifdef USING_cuEST
XDMDeviceBasis describe_basis(const BasisSet& basis) {
    XDMDeviceBasis d;
    d.nbf = basis.nbf();
    d.puream = basis.has_puream();
    for (int Q = 0; Q < basis.nshell(); Q++) {
        const GaussianShell& shell = basis.shell(Q);
        const Vector3 v = shell.center();
        d.center.insert(d.center.end(), {v[0], v[1], v[2]});
        d.am.push_back(shell.am());
        d.nprim.push_back(shell.nprimitive());
        d.prim_offset.push_back(static_cast<int>(d.exps.size()));
        d.func_offset.push_back(shell.function_index());
        d.nfunc.push_back(shell.nfunction());
        for (int K = 0; K < shell.nprimitive(); K++) {
            d.exps.push_back(shell.exps()[K]);
            d.coefs.push_back(shell.coefs()[K]);
        }
    }
    if (d.puream) {
        // BasisFunctions::compute_functions picks the same order.
#if psi4_SHGSHELL_ORDERING == LIBINT_SHGSHELL_ORDERING_STANDARD
        const int order = GG_SPHERICAL_CCA;
#elif psi4_SHGSHELL_ORDERING == LIBINT_SHGSHELL_ORDERING_GAUSSIAN
        const int order = GG_SPHERICAL_GAUSSIAN;
#else
#error "unknown value of macro psi4_SHGSHELL_ORDERING"
#endif
        d.pure_order = order;
    }
    return d;
}

// Device counterpart of the host block loop in integrate_properties: same grid blocks,
// same screened shells, moments returned into ``atom_data``. With ``device_blocks`` the
// points never leave the device and ``grid`` has no host blocks.
void integrate_on_device(std::shared_ptr<Wavefunction> wfn, const DFTGrid& grid, const DeviceGridBlocks* device_blocks,
                         const ProatomDensity& proatom, const std::vector<int>& atomic_nums,
                         const std::vector<std::array<double, 3>>& atom_coords, SharedMatrix Da, SharedMatrix Db,
                         std::vector<AtomicData>& atom_data, size_t& nbhole_failed, size_t& nbhole_total) {
    const int natom = static_cast<int>(atomic_nums.size());
    const std::size_t budget = static_cast<std::size_t>(wfn->options().get_int("XDM_GPU_MEMORY")) << 20;
    std::unique_ptr<DeviceXDMIntegrator> device;
    {
        ScopedTimer t("XDM: device setup");
        device = std::make_unique<DeviceXDMIntegrator>(natom, atomic_nums.data(), atom_coords.data(), proatom,
                                                       describe_basis(*wfn->basisset()), Da->pointer()[0],
                                                       Db ? Db->pointer()[0] : nullptr, budget,
                                                       wfn->options().get_bool("XDM_GPU_RESIDENT_DENSITY"));
    }
    {
        ScopedTimer t("XDM: device queue");
        if (device_blocks) {
            device->add_device_blocks(*device_blocks);
        } else {
            for (const auto& block : grid.blocks()) {
                device->add_block(block->npoints(), block->x(), block->y(), block->z(), block->w(),
                                  block->shells_local_to_global());
            }
        }
    }
    std::vector<std::array<double, DeviceXDMIntegrator::NMOMENT>> sums;
    {
        ScopedTimer t("XDM: device finish");
        sums = device->finish(nbhole_failed, nbhole_total);
    }
    for (int a = 0; a < natom; a++) {
        atom_data[a].mm1 = sums[a][DeviceXDMIntegrator::MM1];
        atom_data[a].mm2 = sums[a][DeviceXDMIntegrator::MM2];
        atom_data[a].mm3 = sums[a][DeviceXDMIntegrator::MM3];
        atom_data[a].vol = sums[a][DeviceXDMIntegrator::VOLUME];
        atom_data[a].charge = sums[a][DeviceXDMIntegrator::CHARGE];
    }
    const double mib = 1.0 / (1024.0 * 1024.0);
    outfile->Printf("\n  XDM integrated on the GPU: %zu points in %zu chunks.\n", device->npoints(),
                    device->nchunks());
    outfile->Printf("    Device buffers %.1f MiB; device memory grew %.1f MiB over %.1f MiB already in use.\n",
                    device->device_bytes() * mib, device->device_growth() * mib,
                    device->device_in_use_at_start() * mib);
    outfile->Printf("    Density matrices %s.\n", device->density_resident() ? "resident on the device"
                                                                          : "gathered on the host per chunk");
    double gpu_total = 0.0;
    std::string stages;
    for (const auto& [name, sec] : device->stage_seconds()) {
        gpu_total += sec;
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %.3f", sec);
        stages += (stages.empty() ? "" : ", ") + name + buf;
    }
    outfile->Printf("    Device stage time %.3f s: %s.\n", gpu_total, stages.c_str());
}
#endif

}  // namespace

// ============================================================================
// Constructor and factory methods
// ============================================================================

XDMDispersion::XDMDispersion(double a1, double a2_bohr, const std::string& functional_name)
    : a1_(a1), a2_(a2_bohr), functional_name_(functional_name) {
    if (!std::isfinite(a1_) || !std::isfinite(a2_)) {
        throw std::invalid_argument("XDM damping parameters must be finite.");
    }
    std::transform(functional_name_.begin(), functional_name_.end(), functional_name_.begin(),
                   [](unsigned char c) { return std::tolower(c); });
}

std::shared_ptr<XDMDispersion> XDMDispersion::build(const std::string& functional, double a1, double a2_angstrom) {
    double a2_bohr = a2_angstrom / pc_bohr2angstroms;
    return std::make_shared<XDMDispersion>(a1, a2_bohr, functional);
}

// ============================================================================
// Energy
// ============================================================================

double XDMDispersion::compute_energy(std::shared_ptr<Wavefunction> wfn, double hf_fraction) {
    if (!wfn) {
        throw std::invalid_argument("XDM compute_energy requires a non-null wavefunction.");
    }
    if (!std::isfinite(hf_fraction)) {
        throw std::invalid_argument("XDM exact-exchange fraction must be finite.");
    }
    if (hf_fraction < 0.0 || hf_fraction > 1.0) {
        throw std::invalid_argument("XDM exact-exchange fraction must be between 0 and 1.");
    }
    validate_xdm_hf_fraction(functional_name_, hf_fraction);
    ScopedTimer t("XDM: total");
    auto atoms = integrate_properties(wfn);
    return pairwise_energy(wfn->molecule(), atoms, hf_fraction);
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

    // XDM_GPU moves everything per grid point to the device; the grid, its blocks and the
    // basis-function screening are the host's in both paths.
    const bool use_gpu = options.get_bool("XDM_GPU");
#ifndef USING_cuEST
    if (use_gpu) {
        throw PSIEXCEPTION("XDM_GPU requires a Psi4 built with cuEST (ENABLE_cuEST).");
    }
#endif

    // Get density matrices in AO basis (wfn stores them in SO basis)
    SharedMatrix Da = wfn->Da_subset("AO");
    SharedMatrix Db = restricted ? Da : wfn->Db_subset("AO");

    // --- Grid: the host DFT grid, or cuEST's points blocked on the host ---
    std::string grid_kind = options.get_str("XDM_GRID");
    if (grid_kind == "AUTO") {
        grid_kind = use_gpu && options.get_bool("USE_CUEST") && !options["DFT_GRID_ORIENTATION"].has_changed()
                        ? "CUEST"
                        : "PSI4";
    }
#ifndef USING_cuEST
    if (grid_kind == "CUEST") {
        throw PSIEXCEPTION("XDM_GRID CUEST requires a Psi4 built with cuEST (ENABLE_cuEST).");
    }
#endif
    std::map<std::string, int> int_opts;
    std::map<std::string, std::string> str_opts;
    // ATOMIC blocking needs the host atomic grids, which a cuEST grid never builds.
    str_opts["DFT_BLOCK_SCHEME"] = grid_kind == "CUEST" ? "OCTREE" : "ATOMIC";

    // Where cuEST's points are blocked: HOST copies them out and blocks with OCTREE; DEVICE
    // sorts and blocks them on the GPU, where a GPU integration then reads them in place.
    std::string blocking = options.get_str("XDM_GRID_BLOCKING");
    if (blocking == "AUTO") blocking = grid_kind == "CUEST" && use_gpu ? "DEVICE" : "HOST";
    if (blocking == "DEVICE" && grid_kind != "CUEST") {
        throw PSIEXCEPTION("XDM_GRID_BLOCKING DEVICE needs XDM_GRID CUEST: only cuEST's points live on the device.");
    }

    std::shared_ptr<DFTGrid> grid;
#ifdef USING_cuEST
    std::unique_ptr<DeviceGridBlocks> device_blocks;
#endif
    {
        ScopedTimer t("XDM: grid");
        grid = std::make_shared<DFTGrid>(mol, primary, int_opts, str_opts, options, grid_kind == "CUEST");
#ifdef USING_cuEST
        if (grid_kind == "CUEST" && blocking == "HOST") grid->host_blocks_from_cuest();
        if (blocking == "DEVICE") {
            double *d_xyz = nullptr, *d_w = nullptr;
            size_t ncuest = 0;
            {
                ScopedTimer tp("cuEST points");
                ncuest = grid->cuest_device_points(&d_xyz, &d_w);
            }
            ScopedTimer tb("device blocking");
            std::vector<std::array<double, 3>> centres(natom);
            for (int a = 0; a < natom; a++) centres[a] = {mol->x(a), mol->y(a), mol->z(a)};
            auto extents = grid->cuest_extents();
            const auto Rp = extents->shell_extents();
            std::vector<double> shell_extents(Rp->pointer(), Rp->pointer() + primary->nshell());
            device_blocks = std::make_unique<DeviceGridBlocks>(
                d_xyz, d_w, ncuest, grid->weights_cutoff(), grid->removes_distant_points() ? extents->maxR() : -1.0,
                natom, centres.data(), grid->cuest_max_points(), describe_basis(*primary), shell_extents);
            if (!use_gpu) {
                // The host integrator walks host blocks: hand it the device partition.
                std::vector<double> x, y, z, w;
                device_blocks->copy_to_host(x, y, z, w);
                grid->adopt_host_blocks(std::move(x), std::move(y), std::move(z), std::move(w),
                                        device_blocks->block_start(), device_blocks->block_npoints());
                device_blocks.reset();
            }
        }
#endif
    }
#ifdef USING_cuEST
    if (device_blocks) {
        const double mib = 1.0 / (1024.0 * 1024.0);
        outfile->Printf("\n  XDM grid: cuEST, %zu points in %zu blocks (%zu with shells), blocked on the device.\n",
                        device_blocks->npoints(), device_blocks->nblocks(), device_blocks->nblocks_with_shells());
        outfile->Printf("    Device blocking peak %.1f MiB; %.1f MiB of points kept for the integration.\n",
                        device_blocks->peak_device_bytes() * mib, device_blocks->resident_device_bytes() * mib);
    } else
#endif
    {
        outfile->Printf("\n  XDM grid: %s, %d points in %zu blocks%s.\n", grid_kind == "CUEST" ? "cuEST" : "psi4",
                        grid->npoints(), grid->blocks().size(), blocking == "DEVICE" ? ", blocked on the device" : "");
    }

    int max_points = grid->max_points();
    int max_functions = grid->max_functions();

    // --- Build PointFunctions for density evaluation ---
    std::shared_ptr<PointFunctions> pf;
    if (use_gpu) {
        // The device evaluates the basis functions itself.
    } else if (restricted) {
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

    // Collect atom data
    std::vector<int> atomic_nums(natom);
    std::vector<std::array<double, 3>> atom_coords(natom);
    for (int a = 0; a < natom; a++) {
        const double effective_z = mol->Z(a);
        const int true_z = mol->true_atomic_number(a);
        if (effective_z > 0.0 && std::abs(effective_z - true_z) > 1.0e-8) {
            throw PSIEXCEPTION(
                "XDM: effective-core potentials are not supported because all-electron densities are required (atom " +
                std::to_string(a + 1) + ", " + mol->symbol(a) + ").");
        }
        // Ghost centres stay in the Hirshfeld promolecule: dropping them hands the
        // ghost-basis density tail to the real atoms, whose r^4/r^6-weighted
        // moments then inflate C10 and Rc at long range. Only the pair sum
        // excludes ghosts.
        atomic_nums[a] = true_z;
        if (atomic_nums[a] > FTOT_NELEM) {
            throw PSIEXCEPTION("XDM: no tabulated free-atom density for atom " + std::to_string(a + 1) + " (" +
                               mol->symbol(a) + ", Z = " + std::to_string(atomic_nums[a]) +
                               "). XDM is parametrized for Z <= " + std::to_string(FTOT_NELEM) + ".");
        }
        atom_coords[a][0] = mol->x(a);
        atom_coords[a][1] = mol->y(a);
        atom_coords[a][2] = mol->z(a);
    }

    ProatomDensity proatom;
    proatom.prepare(natom, atomic_nums.data());
    std::vector<std::vector<double>> hirshfeld_weights;

    // --- Allocate per-atom accumulators ---
    std::vector<AtomicData> atom_data(natom);

    size_t nbhole_failed = 0;
    size_t nbhole_total = 0;

#ifdef USING_cuEST
    if (use_gpu) {
        integrate_on_device(wfn, *grid, device_blocks.get(), proatom, atomic_nums, atom_coords, Da,
                            restricted ? nullptr : Db, atom_data, nbhole_failed, nbhole_total);
    } else
#endif
    {
        // Temporary buffers for Laplacian computation (reused across blocks)
        auto T_lapl = std::make_shared<Matrix>("T_lapl", max_points, max_functions);
        auto D_local = std::make_shared<Matrix>("D_local", max_functions, max_functions);

        // Per-point spin densities and exchange-hole curvatures of the current block
        std::vector<double> rho_spin_buf[2], q_spin_buf[2];
        for (int s = 0; s < 2; s++) {
            rho_spin_buf[s].assign(max_points, 0.0);
            q_spin_buf[s].assign(max_points, 0.0);
        }


        // --- Loop over grid blocks ---
        const auto& blocks = grid->blocks();

        int num_threads = Process::environment.get_n_threads();

        for (size_t ib = 0; ib < blocks.size(); ib++) {
            auto block = blocks[ib];
            int npoints = block->npoints();
            int nlocal = block->local_nbf();
            const auto& function_map = block->functions_local_to_global();

            // Compute density at grid points (rho, gamma, tau)
            {
                ScopedTimer t("XDM: density points");
                pf->compute_points(block);
            }

            // Get grid coordinates and weights
            double* xp = block->x();
            double* yp = block->y();
            double* zp = block->z();
            double* wp = block->w();

            // Get density quantities from PointFunctions
            // RKS: RHO_A = rho_total, GAMMA_AA = |grad rho_total|^2, TAU_A = tau_nohalf_alpha
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

            // Per-spin density (postg convention) and exchange-hole curvature Q; Q is left
            // at zero where the spin density is below the 1e-15 cutoff and b is not solved.
            for (int spin = 0; spin < (restricted ? 1 : 2); spin++) {
                ScopedTimer t("XDM: laplacian");
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
                C_DGEMM('N', 'N', npoints, nlocal, nlocal, 1.0, phip[0], coll_funcs, D2p[0], nglobal, 0.0, Tp[0], nglobal);

                // Add cross terms: 2 * tau_nohalf
                // For RKS alpha: TAU_A = tau_nohalf_alpha, so add 2*TAU_A
                // For UKS: TAU = 0.5*tau_nohalf, so tau_nohalf = 2*TAU, add 2*(2*TAU) = 4*TAU
                double* tau_p = (spin == 0) ? tau_a_p : tau_b_p;
                double tau_factor = restricted ? 2.0 : 4.0;
                double* rho_p = (spin == 0) ? rho_a_p : rho_b_p;
                double* gamma_p = (spin == 0) ? gamma_aa_p : gamma_bb_p;
                double* rho_out = rho_spin_buf[spin].data();
                double* q_out = q_spin_buf[spin].data();

    #pragma omp parallel for schedule(static) num_threads(num_threads)
                for (int P = 0; P < npoints; P++) {
                    // Diagonal Laplacian contribution: 2 * sum_c phi_cc . T0, plus the cross terms
                    double lapl = 2.0 * (C_DDOT(nlocal, phi_xx[P], 1, Tp[P], 1) + C_DDOT(nlocal, phi_yy[P], 1, Tp[P], 1) +
                                         C_DDOT(nlocal, phi_zz[P], 1, Tp[P], 1));
                    lapl += tau_factor * tau_p[P];

                    // Spin density, gradient squared, tau (in postg convention)
                    double rho_spin, grad_sq, tau_nohalf;
                    if (restricted) {
                        // RKS: RHO_A = rho_total, GAMMA_AA = |grad rho_total|^2, TAU_A = tau_nohalf_alpha
                        rho_spin = rho_p[P] * 0.5;
                        grad_sq = 0.25 * gamma_p[P];
                        tau_nohalf = tau_p[P];  // already tau_nohalf for alpha
                    } else {
                        // UKS: RHO_A = rho_alpha, GAMMA_AA = |grad rho_alpha|^2, TAU_A = 0.5*tau_nohalf
                        rho_spin = rho_p[P];
                        grad_sq = gamma_p[P];
                        tau_nohalf = 2.0 * tau_p[P];  // convert from half to nohalf convention
                    }
                    rho_out[P] = rho_spin;
                    // Exchange-hole curvature Q
                    q_out[P] = rho_spin < 1.0e-15 ? 0.0 : exchange_hole_curvature(rho_spin, grad_sq, tau_nohalf, lapl);
                }
            }


            {
                ScopedTimer t("XDM: hirshfeld");
                compute_hirshfeld_weights(proatom, natom, atomic_nums.data(),
                                          reinterpret_cast<const double (*)[3]>(atom_coords.data()), npoints, xp, yp, zp,
                                          hirshfeld_weights);
            }

            // Per-point b parameters and spin densities
            std::vector<double> b_alpha(npoints, 0.0);
            std::vector<double> b_beta(npoints, 0.0);
            std::vector<double> rho_alpha(rho_spin_buf[0].begin(), rho_spin_buf[0].begin() + npoints);
            std::vector<double> rho_beta;

            {
                ScopedTimer t("XDM: becke-roussel");
                for (int spin = 0; spin < (restricted ? 1 : 2); spin++) {
                    const double* rho_in = rho_spin_buf[spin].data();
                    const double* q_in = q_spin_buf[spin].data();
                    double* b_out = (spin == 0) ? b_alpha.data() : b_beta.data();
    #pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+ : nbhole_failed, nbhole_total)
                    for (int P = 0; P < npoints; P++) {
                        double rho_spin = rho_in[P];
                        if (rho_spin < 1.0e-15) {
                            b_out[P] = 0.0;
                            continue;
                        }

                        // Becke-Roussel b parameter. Points where the solve fails fall back to
                        // b = 0 and are tallied for the warning printed after the block loop.
                        double b;
                        nbhole_total += 1;
                        try {
                            b = bhole(rho_spin, q_in[P], 1.0);
                        } catch (...) {
                            b = 0.0;
                            nbhole_failed += 1;
                        }
                        b_out[P] = b;
                    }
                }

                // For restricted: beta = alpha
                if (restricted) {
                    rho_beta = rho_alpha;
                    b_beta = b_alpha;
                } else {
                    rho_beta.assign(rho_spin_buf[1].begin(), rho_spin_buf[1].begin() + npoints);
                }
            }

            // --- Integrate XDM moments using Hirshfeld weights ---
            ScopedTimer t("XDM: moments");
    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
            for (int a = 0; a < natom; a++) {
                if (atomic_nums[a] < 1) continue;

                double ax = atom_coords[a][0];
                double ay = atom_coords[a][1];
                double az = atom_coords[a][2];

                for (int P = 0; P < npoints; P++) {
                    double h = hirshfeld_weights[a][P];
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
                    atom_data[a].mm1 += w * h * (rho_a * (r - r1_a) * (r - r1_a) + rho_b * (r - r1_b) * (r - r1_b));

                    // M2^2: exchange-hole quadrupole moment squared
                    atom_data[a].mm2 += w * h *
                                        (rho_a * (r * r - r1_a * r1_a) * (r * r - r1_a * r1_a) +
                                         rho_b * (r * r - r1_b * r1_b) * (r * r - r1_b * r1_b));

                    // M3^2: exchange-hole octupole moment squared
                    double r3 = r * r * r;
                    double r1a3 = r1_a * r1_a * r1_a;
                    double r1b3 = r1_b * r1_b * r1_b;
                    atom_data[a].mm3 += w * h * (rho_a * (r3 - r1a3) * (r3 - r1a3) + rho_b * (r3 - r1b3) * (r3 - r1b3));

                    // Effective atomic volume
                    atom_data[a].vol += w * h * rho_tot * r3;

                    // Integrated charge
                    atom_data[a].charge += w * h * rho_tot;
                }
            }
        }
    }

    if (nbhole_failed > 0) {
        outfile->Printf("\n  WARNING: the Becke-Roussel b parameter could not be solved at %zu of %zu grid points\n",
                        nbhole_failed, nbhole_total);
        outfile->Printf("           (%.3f%%). Those points were treated as b = 0, so the XDM moments below\n",
                        100.0 * static_cast<double>(nbhole_failed) / static_cast<double>(nbhole_total));
        outfile->Printf("           are approximate.\n");
    }

    // Print summary
    outfile->Printf("\n  ==> XDM Atomic Properties <==\n\n");
    outfile->Printf("    %5s %8s %12s %12s %12s %12s %12s\n", "Atom", "Z", "Charge", "Volume", "M1^2", "M2^2", "M3^2");
    for (int a = 0; a < natom; a++) {
        outfile->Printf("    %5d %5d%-3s %12.6f %12.6f %12.6f %12.6f %12.6f\n", a + 1, atomic_nums[a],
                        mol->Z(a) > 0.0 ? "" : " Gh", atom_data[a].charge, atom_data[a].vol, atom_data[a].mm1,
                        atom_data[a].mm2, atom_data[a].mm3);
    }
    outfile->Printf("\n");

    return atom_data;
}

// ============================================================================
// Pairwise BJ-damped dispersion energy
// ============================================================================

double XDMDispersion::pairwise_energy(std::shared_ptr<Molecule> mol, const std::vector<AtomicData>& atoms,
                                      double hf_fraction) {
    int natom = mol->natom();

    // Per-centre properties. Ghost centres are part of the Hirshfeld promolecule, so
    // their rows carry the moments of the density they hold; only real rows enter the
    // pair sum, but the ghost rows let a counterpoise driver superpose fragments.
    auto props = std::make_shared<Matrix>("XDM ATOMIC PROPERTIES", natom, XDM_NPROP);
    auto coords = std::make_shared<Matrix>("XDM COORDINATES", natom, 3);
    for (int i = 0; i < natom; i++) {
        int Z = mol->true_atomic_number(i);
        double atpol = 0.0;
        if (Z >= 1) {
            double alpha_free = get_free_polarizability(Z);
            double vol_free = get_free_volume(Z, functional_name_, hf_fraction);
            if (vol_free > 1.0e-10 && atoms[i].vol > 1.0e-10) {
                atpol = atoms[i].vol * alpha_free / vol_free;
            }
        }
        props->set(i, XDM_PROP_Z, Z);
        props->set(i, XDM_PROP_REAL, mol->Z(i) > 0.0 ? 1.0 : 0.0);
        props->set(i, XDM_PROP_CHARGE, atoms[i].charge);
        props->set(i, XDM_PROP_VOLUME, atoms[i].vol);
        props->set(i, XDM_PROP_M1, atoms[i].mm1);
        props->set(i, XDM_PROP_M2, atoms[i].mm2);
        props->set(i, XDM_PROP_M3, atoms[i].mm3);
        props->set(i, XDM_PROP_POLARIZABILITY, atpol);
        coords->set(i, 0, mol->x(i));
        coords->set(i, 1, mol->y(i));
        coords->set(i, 2, mol->z(i));
    }

    // Print polarizabilities
    outfile->Printf("  ==> XDM Atomic Polarizabilities <==\n\n");
    outfile->Printf("    %5s %8s %16s\n", "Atom", "Z", "Polarizability");
    for (int i = 0; i < natom; i++) {
        outfile->Printf("    %5d %8d %16.6f\n", i + 1, static_cast<int>(std::lround(mol->Z(i))),
                        props->get(i, XDM_PROP_POLARIZABILITY));
    }
    outfile->Printf("\n");

    Process::environment.arrays["XDM ATOMIC PROPERTIES"] = props;
    return pair_sum(*coords, *props, true);
}

double XDMDispersion::energy_from_properties(SharedMatrix coords, SharedMatrix props) const {
    if (!coords || !props) {
        throw std::invalid_argument("XDM energy_from_properties requires coordinates and properties.");
    }
    if (coords->nirrep() != 1 || props->nirrep() != 1 || coords->colspi(0) != 3 ||
        props->colspi(0) != XDM_NPROP || coords->rowspi(0) != props->rowspi(0)) {
        throw std::invalid_argument("XDM energy_from_properties expects (natom, 3) coordinates [bohr] and (natom, " +
                                    std::to_string(XDM_NPROP) + ") properties.");
    }
    return pair_sum(*coords, *props, false);
}

double XDMDispersion::pair_sum(const Matrix& coords, const Matrix& props, bool record) const {
    int natom = coords.rowspi(0);

    std::vector<int> real_atoms;
    real_atoms.reserve(natom);
    for (int i = 0; i < natom; i++) {
        if (props.get(i, XDM_PROP_REAL) > 0.5) real_atoms.push_back(i);
    }
    int nreal = static_cast<int>(real_atoms.size());

    double e_disp = 0.0, e_disp_tmp = 0.0;

    // Create matrices to store pairwise coefficients
    auto e_disp_pairs = std::make_shared<Matrix>("XDM PAIRWISE ENERGY", nreal, nreal);
    auto c6_mat = std::make_shared<Matrix>("XDM C6 Coefficients", nreal, nreal);
    auto c8_mat = std::make_shared<Matrix>("XDM C8 Coefficients", nreal, nreal);
    auto c10_mat = std::make_shared<Matrix>("XDM C10 Coefficients", nreal, nreal);
    auto rc_mat = std::make_shared<Matrix>("XDM Rc Coefficients", nreal, nreal);
    double** c6p = c6_mat->pointer();
    double** c8p = c8_mat->pointer();
    double** c10p = c10_mat->pointer();
    double** rcp = rc_mat->pointer();

    if (record) {
        outfile->Printf("  ==> XDM Pairwise Coefficients <==\n\n");
        outfile->Printf("    %4s %4s %6s %6s %12s %16s %16s %16s %12s %12s %12s\n", "i", "j", "mol_i", "mol_j", "dij",
                        "C6", "C8", "C10", "Rc", "Rvdw", "E_disp");
    }

    for (int ii = 0; ii < nreal; ii++) {
        int i = real_atoms[ii];
        double m1i = props.get(i, XDM_PROP_M1), m2i = props.get(i, XDM_PROP_M2), m3i = props.get(i, XDM_PROP_M3);
        double pi = props.get(i, XDM_PROP_POLARIZABILITY);
        for (int jj = ii; jj < nreal; jj++) {
            int j = real_atoms[jj];
            double m1j = props.get(j, XDM_PROP_M1), m2j = props.get(j, XDM_PROP_M2), m3j = props.get(j, XDM_PROP_M3);
            double pj = props.get(j, XDM_PROP_POLARIZABILITY);

            // Dispersion coefficients
            double denom = m1i * pj + m1j * pi;
            if (denom < 1.0e-30) continue;
            double fac = pi * pj / denom;

            double c6 = fac * m1i * m1j;
            double c8 = 1.5 * fac * (m1i * m2j + m2i * m1j);
            double c10 = 2.0 * fac * (m1i * m3j + m3i * m1j) + 4.2 * fac * m2i * m2j;

            // Critical radius (BJ damping)
            double rc = (std::sqrt(c8 / c6) + std::sqrt(std::sqrt(c10 / c6)) + std::sqrt(c10 / c8)) / 3.0;

            // Store coefficients in symmetric matrices
            c6p[ii][jj] = c6;
            c6p[jj][ii] = c6;
            c8p[ii][jj] = c8;
            c8p[jj][ii] = c8;
            c10p[ii][jj] = c10;
            c10p[jj][ii] = c10;
            rcp[ii][jj] = rc;
            rcp[jj][ii] = rc;

            if (ii == jj) continue;

            // Interatomic distance
            double xij = coords.get(j, 0) - coords.get(i, 0);
            double yij = coords.get(j, 1) - coords.get(i, 1);
            double zij = coords.get(j, 2) - coords.get(i, 2);
            double d = std::sqrt(xij * xij + yij * yij + zij * zij);

            if (d < 1.0e-10) continue;

            double rvdw = a1_ * rc + a2_;
            double rvdw6 = std::pow(rvdw, 6);
            double rvdw8 = std::pow(rvdw, 8);
            double rvdw10 = std::pow(rvdw, 10);

            double d2 = d * d;
            double d6 = d2 * d2 * d2;
            double d8 = d6 * d2;
            double d10 = d8 * d2;

            // Dispersion energy
            e_disp_tmp = -(c6 / (d6 + rvdw6) + c8 / (d8 + rvdw8) + c10 / (d10 + rvdw10));
            e_disp += e_disp_tmp;
            e_disp_pairs->set(ii, jj, 0.5 * e_disp_tmp);
            e_disp_pairs->set(jj, ii, 0.5 * e_disp_tmp);

            if (record) {
                outfile->Printf("    %4d %4d %6d %6d %12.6f %16.9E %16.9E %16.9E %12.6f %12.6f %12.6f\n", ii + 1,
                                jj + 1, i + 1, j + 1, d, c6, c8, c10, rc, rvdw, e_disp_tmp);
            }
        }
    }

    if (record) {
        // Store coefficient matrices as QC variables
        Process::environment.arrays["XDM C6 COEFFICIENTS"] = c6_mat;
        Process::environment.arrays["XDM C8 COEFFICIENTS"] = c8_mat;
        Process::environment.arrays["XDM C10 COEFFICIENTS"] = c10_mat;
        Process::environment.arrays["XDM RC COEFFICIENTS"] = rc_mat;
        Process::environment.arrays["XDM PAIRWISE ENERGY"] = e_disp_pairs;

        outfile->Printf("\n");
        outfile->Printf("  XDM Dispersion Energy: %20.12f [Eh]\n\n", e_disp);
    }

    return e_disp;
}

}  // namespace xdm
}  // namespace psi
