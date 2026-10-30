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
#include "psi4/libfock/v.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/libscf_solver/hf.h"
#include "psi4/physconst.h"
#include "psi4/psi4-dec.h"

#include "psi4/libmints/gshell.h"
#include "psi4/libmints/vector3.h"

#include "gau2grid/gau2grid.h"
#include <libint2/config.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
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

// gau2grid function order; BasisFunctions::compute_functions picks the same one.
int gg_order(bool puream) {
#if psi4_SHGSHELL_ORDERING == LIBINT_SHGSHELL_ORDERING_STANDARD
    return puream ? GG_SPHERICAL_CCA : GG_CARTESIAN_CCA;
#elif psi4_SHGSHELL_ORDERING == LIBINT_SHGSHELL_ORDERING_GAUSSIAN
    return puream ? GG_SPHERICAL_GAUSSIAN : GG_CARTESIAN_CCA;
#else
#error "unknown value of macro psi4_SHGSHELL_ORDERING"
#endif
}

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
    if (d.puream) d.pure_order = gg_order(true);
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
    outfile->Printf("    Stream time per stage %.3f s: %s.\n", gpu_total, stages.c_str());
    const auto work = device->gemm_work();
    outfile->Printf("    GEMM multiply-adds %.3e over chunk columns, %.3e over block columns.\n", work[0], work[1]);
}
#endif

// The SCF's XC potential when its grid is exactly the one XDM would build (same basis, final grid,
// equal DFTGrid::spec()), else null. Its grid lives as long as the wavefunction's potential.
std::shared_ptr<VBase> scf_potential_on_grid(const std::shared_ptr<Wavefunction>& wfn, const BasisSet& basis,
                                             const std::string& spec) {
    auto hf = std::dynamic_pointer_cast<scf::HF>(wfn);
    auto V = hf ? hf->V_potential() : nullptr;
    if (!V || !V->grid() || V->initial_grid() || V->basis().get() != &basis) return nullptr;
    return V->grid()->spec() == spec ? V : nullptr;
}

// One thread's scratch for integrate_on_host. Collocation is function-major
// ([function][point], row stride = the block's point count), the layout gau2grid writes,
// so D . phi is one GEMM with no transpose.
struct HostScratch {
    std::vector<double> xyz, phi[5], hess, D, T;  // phi: value, x, y, z, Laplacian
    std::vector<double> rho, grad[3], lapl, tau;  // one spin at the block's points
    std::vector<double> rho_s[2], b_s[2];         // per spin, kept for the moments
    std::vector<double> arho, r;                  // promolecule atoms at one point
    std::vector<std::array<double, 5>> sums;      // mm1, mm2, mm3, vol, charge per atom
    double seconds[5] = {};                       // collocation, GEMM, points, Becke-Roussel, moments
};

// Host counterpart of integrate_on_device. Threads take whole blocks, as psi4's V builds do,
// so BLAS runs single-threaded inside the region. Per spin sigma, with T = D_sigma . phi:
//   rho = phi.T, grad rho = 2 phi_c.T, tau (no 1/2) = sum_c phi_c.(D_sigma phi_c),
//   lapl rho = 2 (lapl phi).T + 2 tau,
// four GEMMs per spin. Of the Hessian only lapl phi is kept. Hirshfeld weights are formed
// per point inside the moment loop.
void integrate_on_host(const DFTGrid& grid, const BasisSet& basis, const ProatomDensity& proatom,
                       const std::vector<int>& atomic_nums, const std::vector<std::array<double, 3>>& atom_coords,
                       SharedMatrix Da, SharedMatrix Db, std::vector<AtomicData>& atom_data, size_t& nbhole_failed,
                       size_t& nbhole_total) {
    using clock = std::chrono::steady_clock;
    const int natom = static_cast<int>(atomic_nums.size());
    const int nspin = Db ? 2 : 1;
    const int nbf = basis.nbf();
    const int order = gg_order(basis.has_puream());
    const auto& blocks = grid.blocks();
    const size_t mp = grid.max_points();
    const size_t mf = grid.max_functions();
    size_t max_nfunc = 1;  // widest gau2grid shell output; the Cartesian count bounds the spherical one
    for (int Q = 0; Q < basis.nshell(); Q++) {
        const size_t L = basis.shell(Q).am();
        max_nfunc = std::max(max_nfunc, (L + 1) * (L + 2) / 2);
    }
    const size_t plane = (mf + max_nfunc) * mp;
    const size_t per_thread =
        sizeof(double) * (6 * plane + mf * mf + 6 * max_nfunc * mp + 13 * mp + 7 * static_cast<size_t>(natom));

    // Each thread holds its own scratch; fewer threads when psi4's memory cannot hold them all.
    int nthreads = std::max(1, std::min<int>(Process::environment.get_n_threads(), static_cast<int>(blocks.size())));
    const size_t budget = static_cast<size_t>(Process::environment.get_memory());
    while (nthreads > 1 && per_thread * nthreads > budget) nthreads--;

    std::vector<HostScratch> scratch(nthreads);
    for (auto& s : scratch) {
        s.xyz.resize(3 * mp);
        for (auto& v : s.phi) v.resize(plane);
        s.hess.resize(6 * max_nfunc * mp);
        s.D.resize(mf * mf);
        s.T.resize(plane);
        for (auto* v : {&s.rho, &s.lapl, &s.tau, &s.grad[0], &s.grad[1], &s.grad[2], &s.rho_s[0], &s.rho_s[1],
                        &s.b_s[0], &s.b_s[1]})
            v->resize(mp);
        s.arho.resize(natom);
        s.r.resize(natom);
        s.sums.assign(natom, {0.0, 0.0, 0.0, 0.0, 0.0});
    }
    const double* Dg[2] = {Da->pointer()[0], Db ? Db->pointer()[0] : nullptr};

    size_t failed = 0, total = 0;
    auto wall0 = clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(nthreads) reduction(+ : failed, total)
    for (size_t ib = 0; ib < blocks.size(); ib++) {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        HostScratch& s = scratch[rank];
        const auto& block = blocks[ib];
        const size_t np = block->npoints();
        const auto& fmap = block->functions_local_to_global();
        const int nl = static_cast<int>(fmap.size());
        if (np == 0 || nl == 0) continue;  // no basis function reaches these points: no density
        auto t0 = clock::now();

        // Collocation: value, gradient and Laplacian of every local function.
        std::copy(block->x(), block->x() + np, s.xyz.begin());
        std::copy(block->y(), block->y() + np, s.xyz.begin() + np);
        std::copy(block->z(), block->z() + np, s.xyz.begin() + 2 * np);
        const size_t hs = max_nfunc * np;
        double* h = s.hess.data();
        size_t row = 0;
        for (int Q : block->shells_local_to_global()) {
            const GaussianShell& shell = basis.shell(Q);
            const Vector3 v = shell.center();
            const double center[3] = {v[0], v[1], v[2]};
            const size_t off = row * np;
            gg_collocation_deriv2(shell.am(), np, s.xyz.data(), 1, shell.nprimitive(), shell.coefs(), shell.exps(),
                                  center, order, s.phi[0].data() + off, s.phi[1].data() + off, s.phi[2].data() + off,
                                  s.phi[3].data() + off, h, h + hs, h + 2 * hs, h + 3 * hs, h + 4 * hs, h + 5 * hs);
            const size_t n = static_cast<size_t>(shell.nfunction()) * np;
            double* lap = s.phi[4].data() + off;
            for (size_t k = 0; k < n; k++) lap[k] = h[k] + h[3 * hs + k] + h[5 * hs + k];
            row += shell.nfunction();
        }
        auto t1 = clock::now();
        s.seconds[0] += std::chrono::duration<double>(t1 - t0).count();

        for (int sp = 0; sp < nspin; sp++) {
            auto ta = clock::now();
            for (int m = 0; m < nl; m++) {
                const double* Drow = Dg[sp] + static_cast<size_t>(fmap[m]) * nbf;
                double* out = s.D.data() + static_cast<size_t>(m) * nl;
                for (int n = 0; n < nl; n++) out[n] = Drow[fmap[n]];
            }
            C_DGEMM('N', 'N', nl, np, nl, 1.0, s.D.data(), nl, s.phi[0].data(), np, 0.0, s.T.data(), np);
            auto tb = clock::now();
            std::fill_n(s.rho.begin(), np, 0.0);
            std::fill_n(s.lapl.begin(), np, 0.0);
            std::fill_n(s.tau.begin(), np, 0.0);
            for (auto& g : s.grad) std::fill_n(g.begin(), np, 0.0);
            for (int m = 0; m < nl; m++) {
                const size_t o = static_cast<size_t>(m) * np;
                const double* t = s.T.data() + o;
                const double* p0 = s.phi[0].data() + o;
                const double* px = s.phi[1].data() + o;
                const double* py = s.phi[2].data() + o;
                const double* pz = s.phi[3].data() + o;
                const double* pl = s.phi[4].data() + o;
                for (size_t P = 0; P < np; P++) {
                    s.rho[P] += p0[P] * t[P];
                    s.grad[0][P] += px[P] * t[P];
                    s.grad[1][P] += py[P] * t[P];
                    s.grad[2][P] += pz[P] * t[P];
                    s.lapl[P] += pl[P] * t[P];
                }
            }
            auto tc = clock::now();
            double gemm = std::chrono::duration<double>(tb - ta).count();
            double points = std::chrono::duration<double>(tc - tb).count();
            for (int c = 1; c <= 3; c++) {
                auto td = clock::now();
                C_DGEMM('N', 'N', nl, np, nl, 1.0, s.D.data(), nl, s.phi[c].data(), np, 0.0, s.T.data(), np);
                auto te = clock::now();
                for (int m = 0; m < nl; m++) {
                    const size_t o = static_cast<size_t>(m) * np;
                    const double* t = s.T.data() + o;
                    const double* pc = s.phi[c].data() + o;
                    for (size_t P = 0; P < np; P++) s.tau[P] += pc[P] * t[P];
                }
                gemm += std::chrono::duration<double>(te - td).count();
                points += std::chrono::duration<double>(clock::now() - te).count();
            }
            s.seconds[1] += gemm;
            s.seconds[2] += points;

            // Becke-Roussel b; points where the solve fails count toward the warning and use b = 0.
            auto tf = clock::now();
            double* rho_out = s.rho_s[sp].data();
            double* b_out = s.b_s[sp].data();
            for (size_t P = 0; P < np; P++) {
                const double rho = s.rho[P];
                rho_out[P] = rho;
                b_out[P] = 0.0;
                if (rho < 1.0e-15) continue;
                const double grad_sq =
                    4.0 * (s.grad[0][P] * s.grad[0][P] + s.grad[1][P] * s.grad[1][P] + s.grad[2][P] * s.grad[2][P]);
                const double q = exchange_hole_curvature(rho, grad_sq, s.tau[P], 2.0 * s.lapl[P] + 2.0 * s.tau[P]);
                total += 1;
                try {
                    b_out[P] = bhole(rho, q, 1.0);
                } catch (...) {
                    failed += 1;
                }
            }
            s.seconds[3] += std::chrono::duration<double>(clock::now() - tf).count();
        }

        // Hirshfeld-weighted moments; for a restricted density beta = alpha.
        auto tg = clock::now();
        const double* xp = block->x();
        const double* yp = block->y();
        const double* zp = block->z();
        const double* wp = block->w();
        const int sb = nspin - 1;
        for (size_t P = 0; P < np; P++) {
            double promol = 0.0;
            for (int a = 0; a < natom; a++) {
                s.arho[a] = 0.0;
                if (atomic_nums[a] < 1) continue;
                const double dx = xp[P] - atom_coords[a][0];
                const double dy = yp[P] - atom_coords[a][1];
                const double dz = zp[P] - atom_coords[a][2];
                s.r[a] = std::sqrt(dx * dx + dy * dy + dz * dz);
                s.arho[a] = proatom.evaluate(atomic_nums[a], s.r[a]);
                promol += s.arho[a];
            }
            const double wpro = wp[P] / std::max(promol, 1.0e-40);
            const double rho_a = s.rho_s[0][P];
            const double rho_b = s.rho_s[sb][P];
            const double b_a = s.b_s[0][P];
            const double b_b = s.b_s[sb][P];
            for (int a = 0; a < natom; a++) {
                if (atomic_nums[a] < 1) continue;
                const double wh = wpro * s.arho[a];
                const double r = s.r[a];
                const double r1_a = std::max(0.0, r - b_a);
                const double r1_b = std::max(0.0, r - b_b);
                const double r2 = r * r;
                const double r3 = r2 * r;
                const double d1a = r - r1_a, d1b = r - r1_b;
                const double d2a = r2 - r1_a * r1_a, d2b = r2 - r1_b * r1_b;
                const double d3a = r3 - r1_a * r1_a * r1_a, d3b = r3 - r1_b * r1_b * r1_b;
                auto& acc = s.sums[a];
                acc[0] += wh * (rho_a * d1a * d1a + rho_b * d1b * d1b);
                acc[1] += wh * (rho_a * d2a * d2a + rho_b * d2b * d2b);
                acc[2] += wh * (rho_a * d3a * d3a + rho_b * d3b * d3b);
                acc[3] += wh * (rho_a + rho_b) * r3;
                acc[4] += wh * (rho_a + rho_b);
            }
        }
        s.seconds[4] += std::chrono::duration<double>(clock::now() - tg).count();
    }
    const double wall = std::chrono::duration<double>(clock::now() - wall0).count();
    nbhole_failed += failed;
    nbhole_total += total;

    double seconds[5] = {};
    for (const auto& s : scratch) {
        for (int a = 0; a < natom; a++) {
            atom_data[a].mm1 += s.sums[a][0];
            atom_data[a].mm2 += s.sums[a][1];
            atom_data[a].mm3 += s.sums[a][2];
            atom_data[a].vol += s.sums[a][3];
            atom_data[a].charge += s.sums[a][4];
        }
        for (int k = 0; k < 5; k++) seconds[k] += s.seconds[k];
    }
    outfile->Printf("\n  XDM integrated on the host: %d points in %zu blocks, %d threads over blocks.\n", grid.npoints(),
                    blocks.size(), nthreads);
    outfile->Printf("    Scratch %.1f MiB per thread; block loop %.3f s wall.\n", per_thread / (1024.0 * 1024.0), wall);
    outfile->Printf(
        "    Thread time per stage %.3f s: collocation %.3f, GEMM %.3f, density %.3f, Becke-Roussel %.3f, Hirshfeld + "
        "moments %.3f.\n",
        seconds[0] + seconds[1] + seconds[2] + seconds[3] + seconds[4], seconds[0], seconds[1], seconds[2], seconds[3],
        seconds[4]);
}

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
    // A psi4 grid is blocked by DFT_BLOCK_SCHEME, as the SCF's is; a cuEST grid with OCTREE, since
    // ATOMIC needs the host atomic grids, which a cuEST grid never builds.
    if (grid_kind == "CUEST") str_opts["DFT_BLOCK_SCHEME"] = "OCTREE";

    // Where cuEST's points are blocked: HOST copies them out and blocks with OCTREE; DEVICE
    // sorts and blocks them on the GPU, where a GPU integration then reads them in place.
    std::string blocking = options.get_str("XDM_GRID_BLOCKING");
    if (blocking == "AUTO") blocking = grid_kind == "CUEST" && use_gpu ? "DEVICE" : "HOST";
    if (blocking == "DEVICE" && grid_kind != "CUEST") {
        throw PSIEXCEPTION("XDM_GRID_BLOCKING DEVICE needs XDM_GRID CUEST: only cuEST's points live on the device.");
    }

    // Borrow the SCF's grid when it is the one XDM would build. Host blocking of cuEST's points,
    // and device blocks handed to the host integrator, write host blocks into the grid, so those
    // routes always build their own.
    const bool writes_grid = grid_kind == "CUEST" && (blocking == "HOST" || !use_gpu);
    std::shared_ptr<VBase> scf_V;
    if (options.get_bool("XDM_GRID_REUSE") && !writes_grid) {
        scf_V = scf_potential_on_grid(wfn, *primary,
                                      DFTGrid::spec_for(options, int_opts, str_opts, {}, grid_kind == "CUEST"));
    }
    const char* origin = scf_V ? ", the SCF's" : "";

    std::shared_ptr<DFTGrid> grid;
#ifdef USING_cuEST
    std::unique_ptr<DeviceGridBlocks> device_blocks;
#endif
    {
        ScopedTimer t("XDM: grid");
        grid = scf_V ? scf_V->grid()
                     : std::make_shared<DFTGrid>(mol, primary, int_opts, str_opts, options, grid_kind == "CUEST");
#ifdef USING_cuEST
        if (grid_kind == "CUEST" && blocking == "HOST") grid->host_blocks_from_cuest();
        if (blocking == "DEVICE") {
            double *d_xyz = nullptr, *d_w = nullptr;
            size_t ncuest = 0;
            {
                ScopedTimer tp("cuEST points");
                ncuest = grid->cuest_device_points(&d_xyz, &d_w, scf_V ? scf_V->cuest_xc_plan() : nullptr);
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
        outfile->Printf("\n  XDM grid: cuEST%s, %zu points in %zu blocks (%zu with shells), blocked on the device.\n",
                        origin, device_blocks->npoints(), device_blocks->nblocks(), device_blocks->nblocks_with_shells());
        outfile->Printf("    Device blocking peak %.1f MiB; %.1f MiB of points kept for the integration.\n",
                        device_blocks->peak_device_bytes() * mib, device_blocks->resident_device_bytes() * mib);
    } else
#endif
    {
        outfile->Printf("\n  XDM grid: %s%s, %d points in %zu blocks%s.\n", grid_kind == "CUEST" ? "cuEST" : "psi4",
                        origin, grid->npoints(), grid->blocks().size(),
                        blocking == "DEVICE" ? ", blocked on the device" : "");
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
        ScopedTimer t("XDM: host blocks");
        integrate_on_host(*grid, *primary, proatom, atomic_nums, atom_coords, Da, restricted ? nullptr : Db, atom_data,
                          nbhole_failed, nbhole_total);
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
