/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */

// Device XDM moment integration. Each formula below mirrors the host code it
// replaces -- gau2grid's collocation as BasisFunctions calls it, the density
// and Laplacian in RKSFunctions/UKSFunctions and xdm.cc, becke_roussel.{h,cc},
// hirshfeld.cc and the moment loop in xdm.cc -- so the two paths differ only
// by rounding and summation order.

#include "xdm_gpu.h"
#include "xdm_gpu_blocking.h"

#include "hirshfeld.h"
#include "xdm_data.h"

#include "psi4/libpsi4util/exception.h"

#include "gau2grid/gau2grid.h"

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

// Process-global handles owned by cuest_runtime.cc; cublas_handle runs on stream_handle.
extern cublasHandle_t cublas_handle;
extern cudaStream_t stream_handle;

namespace psi {
// libqt/qt.h; declared here for the same reason as below.
void timer_on(const std::string& key);
void timer_off(const std::string& key);

namespace cuest_common {
// cuESTCommon.h; declared here so nvcc does not parse the libmints headers it pulls in.
void ensure_cuest_initialized();
}  // namespace cuest_common

namespace xdm {
namespace {

constexpr int block_size = 256;
// Blocks per atom in the moment kernel. Fixed, so the partial sums are always
// combined in the same order and the result does not depend on the device.
constexpr int moment_slots = 32;
constexpr int nmoment = DeviceXDMIntegrator::NMOMENT;
// Collocation buffers: phi, phi_x, phi_y, phi_z and phi_xx + phi_yy + phi_zz.
constexpr int ncolloc = 5;
// GEMM outputs: D phi, D phi_x, D phi_y, D phi_z.
constexpr int ngemm = 4;
constexpr int max_am = 6;
// Points per chunk; bounds the per-point arrays independently of the workspace.
constexpr std::size_t max_chunk_points = std::size_t{1} << 16;
constexpr int colloc_threads = 128;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw PSIEXCEPTION(std::string("XDM_GPU: ") + what + " failed: " + cudaGetErrorString(err));
    }
}

// Scoped psi4 timer that is closed on the way out of an exception too.
struct QTimer {
    const char* key;
    explicit QTimer(const char* k) : key(k) { timer_on(key); }
    ~QTimer() { timer_off(key); }
    QTimer(const QTimer&) = delete;
    QTimer& operator=(const QTimer&) = delete;
};

void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw PSIEXCEPTION(std::string("XDM_GPU: ") + what + " failed (cuBLAS status " +
                           std::to_string(static_cast<int>(status)) + ")");
    }
}

struct DeviceAtoms {
    int natom;
    const int* Z;
    const double* x;
    const double* y;
    const double* z;
    const int* knot_offset;  // into the packed {f, a, b, c} spline table
    const int* ndata;
    const double* rmid;
    const double4* table;
};

struct DeviceBasis {
    bool puream;
    const double* center;
    const int* am;
    const int* nprim;
    const int* prim_offset;
    const double* exps;
    const double* coefs;
    const double* transform;         // per L, row-major (2L+1) x ncart(L)
    const int* transform_offset;     // [max_am + 1]
};

// Per chunk: grid blocks and the union of their shells, which defines the GEMM columns.
struct ChunkLayout {
    int npoints;
    int ncols;
    const int* blk_point_offset;  // [nblk]
    const int* blk_npoints;       // [nblk]
    const int* blk_shell_offset;  // [nblk + 1] into blk_shells
    const int* blk_shells;        // union-shell indices
    const int* ushell_global;     // [nushell] global shell index
    const int* ushell_col;        // [nushell] first column
};

// ==> Collocation (gau2grid with deriv 2, as BasisFunctions::compute_functions) <== //

// Writes the shell's functions at phi[f * fstride + v * vstride], v over the ncolloc values.
// L is a template parameter so the Cartesian and spherical accumulators stay in registers.
template <int L>
__device__ void eval_shell(const DeviceBasis& B, int shell, double px, double py, double pz, double* phi,
                           std::size_t fstride, std::size_t vstride) {
    constexpr int ncart = (L + 1) * (L + 2) / 2;
    constexpr int nsph = 2 * L + 1;
    const double dx = px - B.center[3 * shell + 0];
    const double dy = py - B.center[3 * shell + 1];
    const double dz = pz - B.center[3 * shell + 2];
    const double r2 = dx * dx + dy * dy + dz * dz;

    // R = sum c e^{-a r^2}; dR/dx = -2 x S1; d2R/dx2 = -2 S1 + 4 x^2 S2
    double S0 = 0.0, S1 = 0.0, S2 = 0.0;
    const int p0 = B.prim_offset[shell];
    for (int k = 0; k < B.nprim[shell]; k++) {
        const double a = B.exps[p0 + k];
        const double ce = B.coefs[p0 + k] * exp(-a * r2);
        S0 += ce;
        S1 += a * ce;
        S2 += a * a * ce;
    }

    double xp[L + 1], yp[L + 1], zp[L + 1];
    xp[0] = yp[0] = zp[0] = 1.0;
#pragma unroll
    for (int i = 1; i <= L; i++) {
        xp[i] = xp[i - 1] * dx;
        yp[i] = yp[i - 1] * dy;
        zp[i] = zp[i - 1] * dz;
    }

    // Spherical functions accumulate as each Cartesian one is formed; Cartesian ones are written out.
    const double* M = B.transform + B.transform_offset[L];
    double sph[ncolloc][nsph];
#pragma unroll
    for (int v = 0; v < ncolloc; v++) {
#pragma unroll
        for (int s = 0; s < nsph; s++) sph[v][s] = 0.0;
    }
    int c = 0;
    // CCA Cartesian order: x power from L down, then y power from L - i down.
#pragma unroll
    for (int i = L; i >= 0; i--) {
#pragma unroll
        for (int j = L - i; j >= 0; j--, c++) {
            const int k = L - i - j;
            const double fx = xp[i], fy = yp[j], fz = zp[k];
            const double fx1 = i > 0 ? i * xp[i > 0 ? i - 1 : 0] : 0.0;
            const double fy1 = j > 0 ? j * yp[j > 0 ? j - 1 : 0] : 0.0;
            const double fz1 = k > 0 ? k * zp[k > 0 ? k - 1 : 0] : 0.0;
            const double fx2 = i > 1 ? i * (i - 1) * xp[i > 1 ? i - 2 : 0] : 0.0;
            const double fy2 = j > 1 ? j * (j - 1) * yp[j > 1 ? j - 2 : 0] : 0.0;
            const double fz2 = k > 1 ? k * (k - 1) * zp[k > 1 ? k - 2 : 0] : 0.0;

            const double gx = fx1 * S0 - 2.0 * dx * fx * S1;
            const double gy = fy1 * S0 - 2.0 * dy * fy * S1;
            const double gz = fz1 * S0 - 2.0 * dz * fz * S1;
            const double hx = fx2 * S0 - 4.0 * dx * fx1 * S1 + fx * (4.0 * dx * dx * S2 - 2.0 * S1);
            const double hy = fy2 * S0 - 4.0 * dy * fy1 * S1 + fy * (4.0 * dy * dy * S2 - 2.0 * S1);
            const double hz = fz2 * S0 - 4.0 * dz * fz1 * S1 + fz * (4.0 * dz * dz * S2 - 2.0 * S1);

            const double cart[ncolloc] = {fx * fy * fz * S0, gx * fy * fz, fx * gy * fz, fx * fy * gz,
                                          hx * fy * fz + fx * hy * fz + fx * fy * hz};
            if (!B.puream) {
#pragma unroll
                for (int v = 0; v < ncolloc; v++) phi[c * fstride + v * vstride] = cart[v];
            } else {
#pragma unroll
                for (int s = 0; s < nsph; s++) {
                    const double m = M[s * ncart + c];
#pragma unroll
                    for (int v = 0; v < ncolloc; v++) sph[v][s] += m * cart[v];
                }
            }
        }
    }
    if (!B.puream) return;
#pragma unroll
    for (int s = 0; s < nsph; s++) {
#pragma unroll
        for (int v = 0; v < ncolloc; v++) phi[s * fstride + v * vstride] = sph[v][s];
    }
}

// Grid blocks of a chunk run on gridDim.x; gridDim.y thread blocks share each one's
// (point, shell) pairs, consecutive threads taking consecutive points of one shell.
// phi is column-major with the points fastest: phi[col * ncolloc * P + v * P + p], so those
// threads write contiguously. Columns of shells the host screening dropped for this block
// stay zero, so the chunk GEMM over the shell union reproduces each block's local density exactly.
__global__ void collocation_kernel(DeviceBasis B, ChunkLayout C, const double* __restrict__ px,
                                   const double* __restrict__ py, const double* __restrict__ pz, double* phi) {
    const int blk = blockIdx.x;
    const int p0 = C.blk_point_offset[blk];
    const int np = C.blk_npoints[blk];
    const int s0 = C.blk_shell_offset[blk];
    const int ns = C.blk_shell_offset[blk + 1] - s0;
    const std::size_t vstride = C.npoints;
    const std::size_t fstride = ncolloc * vstride;
    for (int item = blockIdx.y * blockDim.x + threadIdx.x; item < np * ns; item += gridDim.y * blockDim.x) {
        const int ip = item % np;
        const int us = C.blk_shells[s0 + item / np];
        const int shell = C.ushell_global[us];
        const int p = p0 + ip;
        double* out = phi + C.ushell_col[us] * fstride + p;
        switch (B.am[shell]) {
            case 0: eval_shell<0>(B, shell, px[p], py[p], pz[p], out, fstride, vstride); break;
            case 1: eval_shell<1>(B, shell, px[p], py[p], pz[p], out, fstride, vstride); break;
            case 2: eval_shell<2>(B, shell, px[p], py[p], pz[p], out, fstride, vstride); break;
            case 3: eval_shell<3>(B, shell, px[p], py[p], pz[p], out, fstride, vstride); break;
            case 4: eval_shell<4>(B, shell, px[p], py[p], pz[p], out, fstride, vstride); break;
            case 5: eval_shell<5>(B, shell, px[p], py[p], pz[p], out, fstride, vstride); break;
            default: eval_shell<max_am>(B, shell, px[p], py[p], pz[p], out, fstride, vstride); break;
        }
    }
}

// ==> Spin density stage (RKSFunctions/UKSFunctions + the Laplacian in xdm.cc) <== //

// Thread blocks of density_tx points by density_ty column slices, the slices summed in shared
// memory. phi: [ncols][ncolloc][npoints]; T: [ncols][ngemm][npoints], T_v = phi_v D.
// Writes the spin density and exchange-hole curvature in the postg convention.
constexpr int density_tx = 32;
constexpr int density_ty = 8;
__global__ void density_kernel(int npoints, int ncols, const double* __restrict__ phi, const double* __restrict__ T,
                               double* __restrict__ rho_out, double* __restrict__ q_out) {
    __shared__ double red[6][density_ty][density_tx];
    const int p = blockIdx.x * density_tx + threadIdx.x;
    const std::size_t P = npoints;
    double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // rho, gx, gy, gz, lap, tau
    if (p < npoints) {
        for (int u = threadIdx.y; u < ncols; u += density_ty) {
            const double* f = phi + u * ncolloc * P + p;
            const double* t = T + u * ngemm * P + p;
            const double d = t[0];
            acc[0] += f[0] * d;
            acc[1] += f[P] * d;
            acc[2] += f[2 * P] * d;
            acc[3] += f[3 * P] * d;
            acc[4] += f[4 * P] * d;
            acc[5] += f[P] * t[P] + f[2 * P] * t[2 * P] + f[3 * P] * t[3 * P];
        }
    }
    for (int k = 0; k < 6; k++) red[k][threadIdx.y][threadIdx.x] = acc[k];
    __syncthreads();
    if (threadIdx.y != 0 || p >= npoints) return;
    for (int k = 0; k < 6; k++) {
        acc[k] = 0.0;
        for (int y = 0; y < density_ty; y++) acc[k] += red[k][y][threadIdx.x];
    }
    const double rho = acc[0], gx = acc[1], gy = acc[2], gz = acc[3], lap = acc[4], tau = acc[5];

    // rho_sigma = phi D phi, grad = 2 phi_c D phi, tau_nohalf = sum_c phi_c D phi_c,
    // lapl = 2 sum_c phi_cc D phi + 2 tau_nohalf
    const double rx = 2.0 * gx, ry = 2.0 * gy, rz = 2.0 * gz;
    const double grad_sq = rx * rx + ry * ry + rz * rz;
    const double lapl = 2.0 * lap + 2.0 * tau;
    rho_out[p] = rho;
    if (rho < 1.0e-15) {
        q_out[p] = 0.0;
    } else {
        // exchange_hole_curvature in becke_roussel.h
        const double dsigs = tau - 0.25 * grad_sq / fmax(rho, 1.0e-30);
        q_out[p] = (lapl - 2.0 * dsigs) / 6.0;
    }
}

// ==> Becke-Roussel, Hirshfeld and moments <== //

// ProatomDensity::evaluate
__device__ double proatom_density(const DeviceAtoms& at, int a, double r) {
    if (r < 1.0e-15) return 0.0;
    const int ndata = at.ndata[a];
    const double h = 1.0 / (ndata + 1);
    const double q = r / (r + at.rmid[a]);
    int intq = static_cast<int>((ndata + 1) * q);
    intq = min(intq, ndata);
    const double dq = q - intq * h;
    const double4 c = at.table[at.knot_offset[a] + intq];
    const double fval = c.x + dq * (c.y + dq * (c.z + dq * c.w));
    return fabs(fval) / (r * r);
}

// xfuncs in becke_roussel.h
__device__ void br_xfuncs(double x, double rhs, double& f, double& df) {
    const double expo23 = exp(-2.0 / 3.0 * x);
    const double xm2 = x - 2.0;
    f = x * expo23 / xm2 - rhs;
    df = 2.0 / 3.0 * (2.0 * x - x * x - 3.0) / (xm2 * xm2) * expo23;
}

// bhole in becke_roussel.cc with hnorm = 1; returns false where the host throws.
__device__ bool br_bhole(double rho, double quad, double& b) {
    constexpr double PI = 3.141592653589793;
    constexpr double TWO_THIRDS = 2.0 / 3.0;
    const double rhs = TWO_THIRDS * pow(PI * rho / 1.0, TWO_THIRDS) * rho / quad;

    const double x0 = 2.0;
    double shift = 1.0;
    double x = x0;
    double f, df;
    bool bracketed = false;
    for (int i = 0; i < 16; i++) {
        x = rhs < 0.0 ? x0 - shift : x0 + shift;
        br_xfuncs(x, rhs, f, df);
        if (rhs < 0.0 ? f < 0.0 : f > 0.0) {
            bracketed = true;
            break;
        }
        shift *= 0.1;
    }
    if (!bracketed) return false;

    for (int i = 0; i < 100; i++) {
        br_xfuncs(x, rhs, f, df);
        const double x1 = x - f / df;
        if (fabs(x1 - x) < 1.0e-10) {
            x = x1;
            const double expo = exp(-x);
            const double prefac = rho / expo;
            const double alf = cbrt(8.0 * PI * prefac / 1.0);
            b = x / alf;
            return true;
        }
        x = x1;
    }
    return false;
}

// Per spin channel, the host's per-point branch: tiny densities get b = 0 without a
// solve, failed solves get b = 0 and are counted.
__device__ double spin_b(double rho, double q, unsigned long long& total, unsigned long long& failed) {
    if (rho < 1.0e-15) return 0.0;
    total += 1;
    double b;
    if (br_bhole(rho, q, b)) return b;
    failed += 1;
    return 0.0;
}

// One thread per point: Becke-Roussel b for each spin and the inverse promolecule density.
__global__ void point_kernel(int n, bool restricted, DeviceAtoms at, const double* __restrict__ px,
                             const double* __restrict__ py, const double* __restrict__ pz,
                             const double* __restrict__ rho_a, const double* __restrict__ rho_b,
                             const double* __restrict__ q_a, const double* __restrict__ q_b,
                             double* __restrict__ b_a, double* __restrict__ b_b, double* __restrict__ promol_inv,
                             unsigned long long* counts) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = 0, failed = 0;
    if (p < n) {
        const double gx = px[p], gy = py[p], gz = pz[p];
        double promol = 0.0;
        for (int a = 0; a < at.natom; a++) {
            if (at.Z[a] < 1) continue;
            const double dx = gx - at.x[a];
            const double dy = gy - at.y[a];
            const double dz = gz - at.z[a];
            promol += proatom_density(at, a, sqrt(dx * dx + dy * dy + dz * dz));
        }
        promol_inv[p] = 1.0 / fmax(promol, 1.0e-40);

        const double ba = spin_b(rho_a[p], q_a[p], total, failed);
        b_a[p] = ba;
        if (!restricted) b_b[p] = spin_b(rho_b[p], q_b[p], total, failed);
    }
    // Integer sums are exact, so atomics keep the counts deterministic.
    if (total) atomicAdd(&counts[0], total);
    if (failed) atomicAdd(&counts[1], failed);
}

// Block (slot, atom): strided partial sums of the five moments for one atom over the
// chunk, reduced in a fixed tree and added to that (atom, slot)'s running total.
__global__ void moment_kernel(int n, bool restricted, DeviceAtoms at, const int* __restrict__ active,
                              const double* __restrict__ px, const double* __restrict__ py,
                              const double* __restrict__ pz, const double* __restrict__ pw,
                              const double* __restrict__ rho_a_in, const double* __restrict__ rho_b_in,
                              const double* __restrict__ b_a_in, const double* __restrict__ b_b_in,
                              const double* __restrict__ promol_inv, double* __restrict__ partials) {
    __shared__ double red[nmoment][block_size];
    const int a = active[blockIdx.y];
    const double ax = at.x[a], ay = at.y[a], az = at.z[a];
    double s[nmoment] = {0.0, 0.0, 0.0, 0.0, 0.0};

    const int stride = gridDim.x * blockDim.x;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += stride) {
        const double dx = px[p] - ax;
        const double dy = py[p] - ay;
        const double dz = pz[p] - az;
        const double r = sqrt(dx * dx + dy * dy + dz * dz);
        const double h = proatom_density(at, a, r) * promol_inv[p];
        const double w = pw[p];

        const double ba = b_a_in[p];
        const double bb = restricted ? ba : b_b_in[p];
        const double r1_a = fmax(0.0, r - ba);
        const double r1_b = fmax(0.0, r - bb);

        const double rho_a = rho_a_in[p];
        const double rho_b = restricted ? rho_a : rho_b_in[p];
        const double rho_tot = rho_a + rho_b;

        s[0] += w * h * (rho_a * (r - r1_a) * (r - r1_a) + rho_b * (r - r1_b) * (r - r1_b));
        s[1] += w * h *
                (rho_a * (r * r - r1_a * r1_a) * (r * r - r1_a * r1_a) +
                 rho_b * (r * r - r1_b * r1_b) * (r * r - r1_b * r1_b));
        const double r3 = r * r * r;
        const double r1a3 = r1_a * r1_a * r1_a;
        const double r1b3 = r1_b * r1_b * r1_b;
        s[2] += w * h * (rho_a * (r3 - r1a3) * (r3 - r1a3) + rho_b * (r3 - r1b3) * (r3 - r1b3));
        s[3] += w * h * rho_tot * r3;
        s[4] += w * h * rho_tot;
    }

    for (int k = 0; k < nmoment; k++) red[k][threadIdx.x] = s[k];
    __syncthreads();
    for (int half = block_size / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            for (int k = 0; k < nmoment; k++) red[k][threadIdx.x] += red[k][threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double* out = partials + (static_cast<std::size_t>(blockIdx.y) * moment_slots + blockIdx.x) * nmoment;
        for (int k = 0; k < nmoment; k++) out[k] += red[k][0];
    }
}

// ==> Density gather on the device (density_resident) <== //

// Du (ncols x ncols, row-major) = D[col_func, col_func] for the chunk's union columns.
__global__ void gather_density_kernel(int ncols, int nbf, const int* __restrict__ col_func,
                                      const double* __restrict__ D, double* __restrict__ Du) {
    const int i = blockIdx.y;
    const std::size_t row = static_cast<std::size_t>(col_func[i]) * nbf;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < ncols; j += gridDim.x * blockDim.x) {
        Du[static_cast<std::size_t>(i) * ncols + j] = D[row + col_func[j]];
    }
}

// ==> Device blocking (XDM_GRID_BLOCKING DEVICE) <== //

// Per thread block: min and max of x, y, z over its grid-stride share of the points.
__global__ void bbox_kernel(std::size_t n, const double* __restrict__ xyz, double* __restrict__ partial) {
    __shared__ double red[6][block_size];
    double v[6] = {DBL_MAX, DBL_MAX, DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX};
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < n; p += stride) {
        for (int k = 0; k < 3; k++) {
            const double c = xyz[3 * p + k];
            v[k] = fmin(v[k], c);
            v[3 + k] = fmax(v[3 + k], c);
        }
    }
    for (int k = 0; k < 6; k++) red[k][threadIdx.x] = v[k];
    __syncthreads();
    for (int half = block_size / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            for (int k = 0; k < 3; k++) red[k][threadIdx.x] = fmin(red[k][threadIdx.x], red[k][threadIdx.x + half]);
            for (int k = 3; k < 6; k++) red[k][threadIdx.x] = fmax(red[k][threadIdx.x], red[k][threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 6; k++) partial[6 * blockIdx.x + k] = red[k][0];
    }
}

// The host build's sieves (DFTGrid::host_blocks_from_cuest) and the Morton key of each kept
// point; dropped points get dropped_key and sort to the end. index[p] = p for the sort.
__global__ void key_kernel(std::size_t n, const double* __restrict__ xyz, const double* __restrict__ w,
                           double weights_cutoff, bool distance_sieve, double max_distance2, int natom,
                           const double* __restrict__ atom_xyz, double lox, double loy, double loz, double scale,
                           std::uint64_t* __restrict__ keys, unsigned int* __restrict__ index,
                           unsigned long long* __restrict__ nkept) {
    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool keep = false;
    if (p < n) {
        const double x = xyz[3 * p], y = xyz[3 * p + 1], z = xyz[3 * p + 2];
        keep = fabs(w[p]) >= weights_cutoff;
        if (keep && distance_sieve) {
            double R2 = DBL_MAX;
            for (int a = 0; a < natom; a++) {
                const double dx = x - atom_xyz[3 * a], dy = y - atom_xyz[3 * a + 1], dz = z - atom_xyz[3 * a + 2];
                // No FMA contraction: the same rounding as the host sieve.
                R2 = fmin(R2, __dadd_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(dz, dz)));
            }
            keep = R2 <= max_distance2;
        }
        keys[p] = keep ? blocking::morton_key(blocking::quantize(x, lox, scale), blocking::quantize(y, loy, scale),
                                              blocking::quantize(z, loz, scale))
                       : blocking::dropped_key;
        index[p] = static_cast<unsigned int>(p);
    }
    const unsigned int ballot = __ballot_sync(0xffffffffu, keep);
    if ((threadIdx.x & 31) == 0 && ballot) atomicAdd(nkept, static_cast<unsigned long long>(__popc(ballot)));
}

// Sorted structure-of-arrays copy of the kept points.
__global__ void gather_points_kernel(std::size_t n, const unsigned int* __restrict__ index,
                                     const double* __restrict__ xyz, const double* __restrict__ w,
                                     double* __restrict__ out) {
    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const std::size_t i = index[p];
    out[p] = xyz[3 * i];
    out[n + p] = xyz[3 * i + 1];
    out[2 * n + p] = xyz[3 * i + 2];
    out[3 * n + p] = w[i];
}

// One thread block per grid block: BlockOPoints::bound() (mean centre, largest distance from
// it), then BlockOPoints::populate()'s two-pass test for every shell, one warp per 32-shell
// mask word. Bit k of word j is shell 32 j + k.
__global__ void screen_kernel(const double* __restrict__ x, const double* __restrict__ y,
                              const double* __restrict__ z, const unsigned long long* __restrict__ bstart,
                              const int* __restrict__ bnp, int nshell, const double* __restrict__ center,
                              const double* __restrict__ extent, int nword, unsigned int* __restrict__ mask) {
    __shared__ double red[3][block_size];
    __shared__ double cR[4];
    const int b = blockIdx.x;
    const std::size_t s0 = bstart[b];
    const int np = bnp[b];

    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int q = threadIdx.x; q < np; q += blockDim.x) {
        sx += x[s0 + q];
        sy += y[s0 + q];
        sz += z[s0 + q];
    }
    red[0][threadIdx.x] = sx;
    red[1][threadIdx.x] = sy;
    red[2][threadIdx.x] = sz;
    __syncthreads();
    for (int half = block_size / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            for (int k = 0; k < 3; k++) red[k][threadIdx.x] += red[k][threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 3; k++) cR[k] = red[k][0] / np;
    }
    __syncthreads();
    const double cx = cR[0], cy = cR[1], cz = cR[2];
    double r2 = 0.0;
    for (int q = threadIdx.x; q < np; q += blockDim.x) {
        const double dx = x[s0 + q] - cx, dy = y[s0 + q] - cy, dz = z[s0 + q] - cz;
        r2 = fmax(r2, dx * dx + dy * dy + dz * dz);
    }
    red[0][threadIdx.x] = r2;
    __syncthreads();
    for (int half = block_size / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) red[0][threadIdx.x] = fmax(red[0][threadIdx.x], red[0][threadIdx.x + half]);
        __syncthreads();
    }
    if (threadIdx.x == 0) cR[3] = sqrt(red[0][0]);
    __syncthreads();
    const double R = cR[3];

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int nwarps = blockDim.x / 32;
    for (int wd = warp; wd < nword; wd += nwarps) {
        unsigned int bits = 0;
        for (int k = 0; k < 32; k++) {
            const int P = 32 * wd + k;
            if (P >= nshell) break;
            const double vx = center[3 * P], vy = center[3 * P + 1], vz = center[3 * P + 2];
            const double Reff = sqrt((vx - cx) * (vx - cx) + (vy - cy) * (vy - cy) + (vz - cz) * (vz - cz));
            if (Reff > R + extent[P]) continue;
            const double Rp2 = extent[P] * extent[P];
            bool found = false;
            for (int q0 = 0; q0 < np; q0 += 32) {
                const int q = q0 + lane;
                bool hit = false;
                if (q < np) {
                    const double dx = x[s0 + q] - vx, dy = y[s0 + q] - vy, dz = z[s0 + q] - vz;
                    hit = dx * dx + dy * dy + dz * dz < Rp2;
                }
                if (__any_sync(0xffffffffu, hit)) {
                    found = true;
                    break;
                }
            }
            if (found) bits |= 1u << k;
        }
        if (lane == 0) mask[static_cast<std::size_t>(b) * nword + wd] = bits;
    }
}

template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    std::size_t count = 0;
};

}  // namespace

std::vector<double> xdm_pure_transform(int L, int order) {
    const int ncart = (L + 1) * (L + 2) / 2;
    const int nsph = 2 * L + 1;
    const int npts = 4 * ncart;
    // Unit contraction, zero exponent: the collocation is the bare polynomial part.
    const double coef = 1.0, expo = 0.0;
    const double center[3] = {0.0, 0.0, 0.0};
    std::vector<double> xyz(3 * npts);
    for (int n = 0; n < npts; n++) {
        // Scattered points of both signs on a 1/32 lattice in [-1, 1): six significant
        // bits per coordinate keep every monomial up to degree 8 exact in double.
        xyz[n] = -1.0 + ((37 * n + 11) % 64) / 32.0;
        xyz[npts + n] = -1.0 + ((53 * n + 29) % 64) / 32.0;
        xyz[2 * npts + n] = -1.0 + ((17 * n * n + 5 * n + 3) % 64) / 32.0;
    }
    std::vector<double> cart(ncart * npts), sph(nsph * npts);
    gg_collocation(L, npts, xyz.data(), 1, 1, &coef, &expo, center, GG_CARTESIAN_CCA, cart.data());
    gg_collocation(L, npts, xyz.data(), 1, 1, &coef, &expo, center, order, sph.data());

    // sph[s][n] = sum_c M[s][c] cart[c][n]: least squares A M^T = S^T, A[n][c] = cart[c][n],
    // by Householder QR in extended precision.
    std::vector<long double> A(static_cast<std::size_t>(npts) * ncart), B(static_cast<std::size_t>(npts) * nsph);
    for (int n = 0; n < npts; n++) {
        for (int c = 0; c < ncart; c++) A[n * ncart + c] = cart[c * npts + n];
        for (int q = 0; q < nsph; q++) B[n * nsph + q] = sph[q * npts + n];
    }
    std::vector<long double> v(npts);
    for (int k = 0; k < ncart; k++) {
        long double norm = 0.0L;
        for (int n = k; n < npts; n++) norm += A[n * ncart + k] * A[n * ncart + k];
        norm = std::sqrt(norm);
        if (norm < 1.0e-12L) {
            throw PSIEXCEPTION("XDM_GPU: degenerate point set while reading the pure-function transform.");
        }
        const long double alpha = A[k * ncart + k] > 0 ? -norm : norm;
        long double vnorm = 0.0L;
        for (int n = k; n < npts; n++) {
            v[n] = A[n * ncart + k] - (n == k ? alpha : 0.0L);
            vnorm += v[n] * v[n];
        }
        for (int c = k; c < ncart; c++) {
            long double d = 0.0L;
            for (int n = k; n < npts; n++) d += v[n] * A[n * ncart + c];
            d = 2.0L * d / vnorm;
            for (int n = k; n < npts; n++) A[n * ncart + c] -= d * v[n];
        }
        for (int q = 0; q < nsph; q++) {
            long double d = 0.0L;
            for (int n = k; n < npts; n++) d += v[n] * B[n * nsph + q];
            d = 2.0L * d / vnorm;
            for (int n = k; n < npts; n++) B[n * nsph + q] -= d * v[n];
        }
    }
    std::vector<double> M(static_cast<std::size_t>(nsph) * ncart);
    for (int q = 0; q < nsph; q++) {
        std::vector<long double> x(ncart);
        for (int c = ncart - 1; c >= 0; c--) {
            long double r = B[c * nsph + q];
            for (int k = c + 1; k < ncart; k++) r -= A[c * ncart + k] * x[k];
            x[c] = r / A[c * ncart + c];
        }
        for (int c = 0; c < ncart; c++) {
            // Drop the solve's rounding residue on structurally zero coefficients.
            M[q * ncart + c] = std::fabs(x[c]) < 1.0e-13L ? 0.0 : static_cast<double>(x[c]);
        }
    }
    return M;
}

struct DeviceXDMIntegrator::Impl {
    bool restricted = true;
    int nspin = 1;
    int natom = 0;
    int nactive = 0;
    std::vector<int> active_atoms;
    const double* D[2] = {nullptr, nullptr};
    XDMDeviceBasis basis;

    std::size_t bytes = 0;
    std::size_t peak_bytes = 0;
    std::size_t sent = 0;
    std::size_t chunks = 0;
    std::size_t workspace_doubles = 0;

    std::vector<void*> device_allocs;
    DeviceAtoms atoms{};
    DeviceBasis dbasis{};
    int* d_active = nullptr;

    // Per-chunk device buffers
    double* d_points = nullptr;  // x, y, z, w, rho_a, rho_b, q_a, q_b, b_a, b_b, promol_inv: 11 * max_chunk_points
    double* d_work = nullptr;    // collocation, GEMM outputs, gathered density matrices
    double* d_partials = nullptr;
    unsigned long long* d_counts = nullptr;
    int* d_layout = nullptr;
    std::size_t layout_capacity = 0;

    // Host staging (pinned), double-buffered so one chunk is filled while the previous one's
    // copies are in flight: x, y, z, w and the layout ints per slot. The gathered density
    // matrices (only when the density is not resident) have one buffer and their own event.
    static constexpr int nslot = 2;
    int slot = 0;                           // slot of the chunk under construction
    double* h_points[nslot] = {nullptr, nullptr};  // 4 * max_chunk_points each
    int* h_layout[nslot] = {nullptr, nullptr};
    std::size_t h_layout_capacity[nslot] = {0, 0};
    cudaEvent_t copied[nslot] = {nullptr, nullptr};
    bool copy_pending[nslot] = {false, false};
    double* h_dens = nullptr;  // nspin * max_cols^2
    cudaEvent_t dens_copied = nullptr;
    bool dens_pending = false;
    std::size_t max_cols = 0;

    // Full AO density matrices on the device, gathered per chunk by gather_density_kernel.
    bool resident_D = false;
    double* d_Dfull = nullptr;

    // Stage timing: an event on the stream at the start of each chunk and after each stage;
    // the time between two consecutive events of a chunk belongs to the later one's stage.
    // That is stream time, not GPU busy time: host work between two marks while the stream
    // is idle (e.g. the host density gather) is charged to the later stage.
    enum Stage { COPY, GATHER, COLLOC, GEMM, DENSITY, POINT, MOMENT, NSTAGE };
    std::vector<cudaEvent_t> events;
    std::vector<int> event_stage;  // -1 opens a chunk
    std::size_t nevent = 0;
    double stage_ms[NSTAGE] = {};
    // GEMM multiply-adds: done over each chunk's union columns, and needed by each block's own.
    double gemm_done = 0.0, gemm_needed = 0.0;
    void mark(int stage) {
        if (nevent == events.size()) {
            cudaEvent_t e = nullptr;
            check_cuda(cudaEventCreate(&e), "cudaEventCreate");
            events.push_back(e);
            event_stage.push_back(stage);
        }
        event_stage[nevent] = stage;
        check_cuda(cudaEventRecord(events[nevent], stream_handle), "cudaEventRecord");
        nevent++;
    }
    // Called with the stream idle: fold the recorded intervals into stage_ms and reuse the events.
    void collect_stages() {
        for (std::size_t i = 1; i < nevent; i++) {
            if (event_stage[i] < 0) continue;
            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, events[i - 1], events[i]), "cudaEventElapsedTime");
            stage_ms[event_stage[i]] += ms;
        }
        nevent = 0;
    }

    // Device-blocked grid (add_device_blocks): the points are read in place, and a chunk is
    // the contiguous point range [fill_start, fill_start + fill_points).
    const double* dev_points[4] = {nullptr, nullptr, nullptr, nullptr};
    std::size_t fill_start = 0;

    // Chunk under construction
    int fill_points = 0;
    int fill_cols = 0;
    std::size_t max_block_items = 0;  // largest (point, shell) count of a block in the chunk
    int nsm = 1;                      // multiprocessors on the device
    std::vector<int> blk_point_offset, blk_npoints, blk_shell_offset{0}, blk_shells;
    std::vector<int> ushell_global, ushell_col;
    std::vector<int> union_index;  // global shell -> union index, -1 if absent
    std::vector<int> col_func;     // union column -> global AO index

    template <typename T>
    T* alloc(std::size_t count) {
        void* p = nullptr;
        check_cuda(cudaMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "cudaMalloc");
        device_allocs.push_back(p);
        bytes += std::max<std::size_t>(count, 1) * sizeof(T);
        peak_bytes = std::max(peak_bytes, bytes);
        return static_cast<T*>(p);
    }

    template <typename T>
    T* upload(const std::vector<T>& v) {
        T* p = alloc<T>(v.size());
        if (!v.empty()) check_cuda(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
        return p;
    }

    void release() {
        if (stream_handle) cudaStreamSynchronize(stream_handle);
        for (void* p : device_allocs) cudaFree(p);
        device_allocs.clear();
        if (d_layout) cudaFree(d_layout);
        d_layout = nullptr;
        for (int k = 0; k < nslot; k++) {
            if (h_points[k]) cudaFreeHost(h_points[k]);
            if (h_layout[k]) cudaFreeHost(h_layout[k]);
            if (copied[k]) cudaEventDestroy(copied[k]);
            h_points[k] = nullptr;
            h_layout[k] = nullptr;
            copied[k] = nullptr;
        }
        if (h_dens) cudaFreeHost(h_dens);
        h_dens = nullptr;
        for (cudaEvent_t e : events) cudaEventDestroy(e);
        events.clear();
        event_stage.clear();
        nevent = 0;
        if (dens_copied) cudaEventDestroy(dens_copied);
        dens_copied = nullptr;
    }

    // Device memory in use by the whole process, lowest free seen since construction.
    std::size_t free_at_start = 0;
    std::size_t min_free = 0;
    std::size_t total_device = 0;
    void sample_device_memory() {
        std::size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
            min_free = std::min(min_free, f);
            total_device = t;
        }
    }

    // Doubles the chunk needs in d_work for a given point count and column count.
    std::size_t work_need(std::size_t npts, std::size_t ncols) const {
        return (ncolloc + ngemm) * npts * ncols + nspin * ncols * ncols;
    }

    // A staging slot is reused every other chunk; wait for the last copy out of it.
    void wait_staging(int k) {
        if (copy_pending[k]) {
            QTimer t("XDM: staging wait");
            check_cuda(cudaEventSynchronize(copied[k]), "cudaEventSynchronize");
            copy_pending[k] = false;
        }
    }
    void wait_density_staging() {
        if (dens_pending) {
            QTimer t("XDM: staging wait");
            check_cuda(cudaEventSynchronize(dens_copied), "cudaEventSynchronize");
            dens_pending = false;
        }
    }

    // Add one block. Host blocks are copied into the slot's staging; a device block (dev_start
    // set) is its points' position in dev_points, and the chunk stretches to cover it.
    void add(std::size_t npoints, const double* x, const double* y, const double* z, const double* w,
             std::size_t dev_start, const int* shells, std::size_t nshells) {
        if (npoints == 0) return;
        if (npoints > max_chunk_points) {
            throw PSIEXCEPTION("XDM_GPU: grid block larger than the device chunk.");
        }
        const bool device = dev_points[0] != nullptr;
        // Columns this block would add to the chunk's shell union.
        int new_cols = 0;
        for (std::size_t i = 0; i < nshells; i++) {
            if (union_index[shells[i]] < 0) new_cols += basis.nfunc[shells[i]];
        }
        const std::size_t P = device && fill_points > 0 ? dev_start + npoints - fill_start : fill_points + npoints;
        const std::size_t ncols = fill_cols + new_cols;
        if (fill_points > 0 && (P > max_chunk_points || work_need(P, ncols) > workspace_doubles)) {
            flush();
        }
        if (fill_points == 0) {
            wait_staging(slot);
            std::size_t cols = 0;
            for (std::size_t i = 0; i < nshells; i++) cols += basis.nfunc[shells[i]];
            if (work_need(npoints, cols) > workspace_doubles) {
                throw PSIEXCEPTION("XDM_GPU: one grid block does not fit the device workspace; raise XDM_GPU_MEMORY.");
            }
            fill_start = dev_start;
        }

        const int p0 = device ? static_cast<int>(dev_start - fill_start) : fill_points;
        if (!device) {
            const std::size_t cap = max_chunk_points;
            double* h = h_points[slot];
            std::copy(x, x + npoints, h + p0);
            std::copy(y, y + npoints, h + cap + p0);
            std::copy(z, z + npoints, h + 2 * cap + p0);
            std::copy(w, w + npoints, h + 3 * cap + p0);
        }

        std::size_t own_cols = 0;
        for (std::size_t i = 0; i < nshells; i++) own_cols += basis.nfunc[shells[i]];
        gemm_needed += static_cast<double>(npoints) * own_cols * own_cols;
        blk_point_offset.push_back(p0);
        blk_npoints.push_back(static_cast<int>(npoints));
        max_block_items = std::max(max_block_items, npoints * nshells);
        for (std::size_t i = 0; i < nshells; i++) {
            const int g = shells[i];
            int u = union_index[g];
            if (u < 0) {
                u = static_cast<int>(ushell_global.size());
                union_index[g] = u;
                ushell_global.push_back(g);
                ushell_col.push_back(fill_cols);
                for (int f = 0; f < basis.nfunc[g]; f++) col_func.push_back(basis.func_offset[g] + f);
                fill_cols += basis.nfunc[g];
            }
            blk_shells.push_back(u);
        }
        blk_shell_offset.push_back(static_cast<int>(blk_shells.size()));
        // A device chunk's points between blocks belong to blocks no shell reaches: their
        // collocation stays zero, so they carry no density and add nothing to any moment.
        fill_points = p0 + static_cast<int>(npoints);
    }

    void flush() {
        if (fill_points == 0) return;
        QTimer timer("XDM: chunk flush");
        const int P = fill_points;
        const int ncols = fill_cols;
        const int nblk = static_cast<int>(blk_npoints.size());
        const int nus = static_cast<int>(ushell_global.size());
        const bool device = dev_points[0] != nullptr;
        const int k = slot;

        // Layout ints: blk_point_offset, blk_npoints, blk_shell_offset, blk_shells, ushell_global,
        // ushell_col, and col_func when the density is gathered on the device.
        const std::size_t ncolf = resident_D ? static_cast<std::size_t>(ncols) : 0;
        const std::size_t nlayout = 3 * static_cast<std::size_t>(nblk) + 1 + blk_shells.size() + 2 * nus + ncolf;
        if (nlayout > h_layout_capacity[k]) {
            if (h_layout[k]) cudaFreeHost(h_layout[k]);
            h_layout[k] = nullptr;
            h_layout_capacity[k] = std::max(nlayout, 2 * h_layout_capacity[k]);
            void* h = nullptr;
            check_cuda(cudaMallocHost(&h, h_layout_capacity[k] * sizeof(int)), "cudaMallocHost");
            h_layout[k] = static_cast<int*>(h);
        }
        if (nlayout > layout_capacity) {
            check_cuda(cudaStreamSynchronize(stream_handle), "cudaStreamSynchronize");
            if (d_layout) {
                cudaFree(d_layout);
                bytes -= layout_capacity * sizeof(int);
            }
            layout_capacity = std::max(nlayout, 2 * layout_capacity);
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_layout), layout_capacity * sizeof(int)), "cudaMalloc");
            bytes += layout_capacity * sizeof(int);
            peak_bytes = std::max(peak_bytes, bytes);
        }
        int* h = h_layout[k];
        std::copy(blk_point_offset.begin(), blk_point_offset.end(), h);
        std::copy(blk_npoints.begin(), blk_npoints.end(), h + nblk);
        std::copy(blk_shell_offset.begin(), blk_shell_offset.end(), h + 2 * nblk);
        std::copy(blk_shells.begin(), blk_shells.end(), h + 3 * nblk + 1);
        std::copy(ushell_global.begin(), ushell_global.end(), h + 3 * nblk + 1 + blk_shells.size());
        std::copy(ushell_col.begin(), ushell_col.end(), h + 3 * nblk + 1 + blk_shells.size() + nus);
        const std::size_t col_func_at = 3 * static_cast<std::size_t>(nblk) + 1 + blk_shells.size() + 2 * nus;
        if (resident_D) std::copy(col_func.begin(), col_func.end(), h + col_func_at);

        cudaStream_t st = stream_handle;
        // Bound the event list: fold in what has run once it grows (the sync costs one chunk's overlap).
        if (nevent > 4096) {
            check_cuda(cudaStreamSynchronize(st), "cudaStreamSynchronize");
            collect_stages();
        }
        mark(-1);
        const std::size_t cap = max_chunk_points;
        const double* d_x = device ? dev_points[0] + fill_start : d_points;
        const double* d_y = device ? dev_points[1] + fill_start : d_points + cap;
        const double* d_z = device ? dev_points[2] + fill_start : d_points + 2 * cap;
        const double* d_w = device ? dev_points[3] + fill_start : d_points + 3 * cap;
        double* d_rho[2] = {d_points + 4 * cap, d_points + 5 * cap};
        double* d_q[2] = {d_points + 6 * cap, d_points + 7 * cap};
        double* d_b[2] = {d_points + 8 * cap, d_points + 9 * cap};
        double* d_pinv = d_points + 10 * cap;

        if (!device) {
            for (int c = 0; c < 4; c++) {
                check_cuda(cudaMemcpyAsync(d_points + c * cap, h_points[k] + c * cap, P * sizeof(double),
                                           cudaMemcpyHostToDevice, st),
                           "cudaMemcpyAsync");
            }
        }
        check_cuda(cudaMemcpyAsync(d_layout, h, nlayout * sizeof(int), cudaMemcpyHostToDevice, st), "cudaMemcpyAsync");
        check_cuda(cudaEventRecord(copied[k], st), "cudaEventRecord");
        copy_pending[k] = true;
        mark(COPY);

        const std::size_t plane = static_cast<std::size_t>(P) * ncols;
        double* d_phi = d_work;
        double* d_T = d_work + ncolloc * plane;
        double* d_Du = d_T + ngemm * plane;
        if (resident_D) {
            if (ncols > 0) {
                const dim3 grid((ncols + 255) / 256, ncols);
                for (int s = 0; s < nspin; s++) {
                    gather_density_kernel<<<grid, 256, 0, st>>>(
                        ncols, basis.nbf, d_layout + col_func_at,
                        d_Dfull + static_cast<std::size_t>(s) * basis.nbf * basis.nbf,
                        d_Du + static_cast<std::size_t>(s) * ncols * ncols);
                    check_cuda(cudaGetLastError(), "XDM density gather kernel launch");
                }
            }
            mark(GATHER);
        } else {
            // Gather the union-column density matrices on the host.
            wait_density_staging();
            for (int s = 0; s < nspin; s++) {
                double* dst = h_dens + static_cast<std::size_t>(s) * ncols * ncols;
                const double* src = D[s];
                for (int i = 0; i < ncols; i++) {
                    const double* srow = src + static_cast<std::size_t>(col_func[i]) * basis.nbf;
                    double* drow = dst + static_cast<std::size_t>(i) * ncols;
                    for (int j = 0; j < ncols; j++) drow[j] = srow[col_func[j]];
                }
            }
            check_cuda(cudaMemcpyAsync(d_Du, h_dens, nspin * static_cast<std::size_t>(ncols) * ncols * sizeof(double),
                                       cudaMemcpyHostToDevice, st),
                       "cudaMemcpyAsync");
            check_cuda(cudaEventRecord(dens_copied, st), "cudaEventRecord");
            dens_pending = true;
            mark(COPY);
        }

        ChunkLayout C;
        C.npoints = P;
        C.ncols = ncols;
        C.blk_point_offset = d_layout;
        C.blk_npoints = d_layout + nblk;
        C.blk_shell_offset = d_layout + 2 * nblk;
        C.blk_shells = d_layout + 3 * nblk + 1;
        C.ushell_global = C.blk_shells + blk_shells.size();
        C.ushell_col = C.ushell_global + nus;

        check_cuda(cudaMemsetAsync(d_phi, 0, ncolloc * plane * sizeof(double), st), "cudaMemsetAsync");
        // Enough thread blocks to fill the device several times over, but no more than the
        // largest grid block has (point, shell) pairs for.
        const unsigned int colloc_split = static_cast<unsigned int>(std::max<std::size_t>(
            1, std::min<std::size_t>((max_block_items + colloc_threads - 1) / colloc_threads,
                                     (8 * static_cast<std::size_t>(nsm) + nblk - 1) / nblk)));
        collocation_kernel<<<dim3(nblk, colloc_split), colloc_threads, 0, st>>>(dbasis, C, d_x, d_y, d_z, d_phi);
        check_cuda(cudaGetLastError(), "XDM collocation kernel launch");
        mark(COLLOC);

        const double one = 1.0, zero = 0.0;
        const unsigned int density_blocks = static_cast<unsigned int>((P + density_tx - 1) / density_tx);
        for (int s = 0; s < nspin && ncols == 0; s++) {
            // No shell survives the screening anywhere in the chunk: zero density, as on the host.
            check_cuda(cudaMemsetAsync(d_rho[s], 0, P * sizeof(double), st), "cudaMemsetAsync");
            check_cuda(cudaMemsetAsync(d_q[s], 0, P * sizeof(double), st), "cudaMemsetAsync");
        }
        for (int s = 0; s < nspin && ncols > 0; s++) {
            // Column-major T (4P x ncols) = Phi (4P x ncols, leading dimension 5P: the first four values) D.
            check_cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, ngemm * P, ncols, ncols, &one, d_phi,
                                     ncolloc * P, d_Du + static_cast<std::size_t>(s) * ncols * ncols, ncols, &zero,
                                     d_T, ngemm * P),
                         "cublasDgemm");
            mark(GEMM);
            density_kernel<<<density_blocks, dim3(density_tx, density_ty), 0, st>>>(P, ncols, d_phi, d_T, d_rho[s], d_q[s]);
            check_cuda(cudaGetLastError(), "XDM density kernel launch");
            mark(DENSITY);
        }

        const unsigned int point_blocks = static_cast<unsigned int>((P + block_size - 1) / block_size);
        point_kernel<<<point_blocks, block_size, 0, st>>>(P, restricted, atoms, d_x, d_y, d_z, d_rho[0], d_rho[1],
                                                          d_q[0], d_q[1], d_b[0], d_b[1], d_pinv, d_counts);
        check_cuda(cudaGetLastError(), "XDM point kernel launch");
        mark(POINT);
        if (nactive > 0) {
            dim3 grid(moment_slots, nactive);
            moment_kernel<<<grid, block_size, 0, st>>>(P, restricted, atoms, d_active, d_x, d_y, d_z, d_w, d_rho[0],
                                                       d_rho[1], d_b[0], d_b[1], d_pinv, d_partials);
            check_cuda(cudaGetLastError(), "XDM moment kernel launch");
            mark(MOMENT);
        }

        sent += P;
        chunks += 1;
        gemm_done += static_cast<double>(P) * ncols * ncols;
        sample_device_memory();
        slot = (slot + 1) % nslot;
        fill_points = 0;
        fill_cols = 0;
        max_block_items = 0;
        blk_point_offset.clear();
        blk_npoints.clear();
        blk_shell_offset.assign(1, 0);
        blk_shells.clear();
        for (int g : ushell_global) union_index[g] = -1;
        ushell_global.clear();
        ushell_col.clear();
        col_func.clear();
    }
};

DeviceXDMIntegrator::DeviceXDMIntegrator(int natom, const int* atomic_nums, const std::array<double, 3>* atom_coords,
                                         const ProatomDensity& proatom, const XDMDeviceBasis& basis, const double* Da,
                                         const double* Db, std::size_t workspace_bytes, bool resident_density)
    : impl_(new Impl) {
    cuest_common::ensure_cuest_initialized();
    Impl& m = *impl_;
    {
        std::size_t f = 0, t = 0;
        check_cuda(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
        m.free_at_start = m.min_free = f;
        m.total_device = t;
        int dev = 0;
        check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
        check_cuda(cudaDeviceGetAttribute(&m.nsm, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
    }
    m.restricted = (Db == nullptr);
    m.nspin = m.restricted ? 1 : 2;
    m.natom = natom;
    m.D[0] = Da;
    m.D[1] = m.restricted ? Da : Db;
    m.basis = basis;

    try {
        // ==> Atoms and the promolecule splines <== //
        std::vector<double> table;
        std::vector<int> offset(natom, 0), ndata(natom, 0), elem_offset(FTOT_NELEM + 1, -1), active;
        std::vector<double> rmid(natom, 1.0), xyz(3 * static_cast<std::size_t>(std::max(natom, 1)), 0.0);
        const std::size_t na = std::max(natom, 1);
        for (int a = 0; a < natom; a++) {
            const int Z = atomic_nums[a];
            xyz[a] = atom_coords[a][0];
            xyz[na + a] = atom_coords[a][1];
            xyz[2 * na + a] = atom_coords[a][2];
            if (Z < 1 || Z > FTOT_NELEM) continue;
            if (elem_offset[Z] < 0) {
                elem_offset[Z] = static_cast<int>(table.size() / 4);
                proatom.pack_spline(Z, table);
            }
            offset[a] = elem_offset[Z];
            ndata[a] = ndata_for_z(Z);
            rmid[a] = 1.0 / std::cbrt(static_cast<double>(Z));
            active.push_back(a);
        }
        m.nactive = static_cast<int>(active.size());
        m.active_atoms = active;
        std::vector<int> Zv(atomic_nums, atomic_nums + natom);
        const int* d_Z = m.upload(Zv);
        const double* d_xyz = m.upload(xyz);
        const int* d_offset = m.upload(offset);
        const int* d_ndata = m.upload(ndata);
        const double* d_rmid = m.upload(rmid);
        const double* d_table = m.upload(table);
        m.d_active = m.upload(active);
        m.atoms = DeviceAtoms{natom,  d_Z,     d_xyz,  d_xyz + na, d_xyz + 2 * na,
                              d_offset, d_ndata, d_rmid, reinterpret_cast<const double4*>(d_table)};

        // ==> Basis <== //
        int max_l = 0;
        int max_nfunc = 0;
        for (int s = 0; s < basis.nshell(); s++) {
            max_l = std::max(max_l, basis.am[s]);
            max_nfunc = std::max(max_nfunc, basis.nfunc[s]);
        }
        if (max_l > max_am) {
            throw PSIEXCEPTION("XDM_GPU: basis functions with angular momentum above " + std::to_string(max_am) +
                               " are not supported on the GPU; use the host path (XDM_GPU false).");
        }
        std::vector<double> transform;
        std::vector<int> transform_offset(max_am + 1, 0);
        if (basis.puream) {
            for (int l = 0; l <= max_am; l++) {
                transform_offset[l] = static_cast<int>(transform.size());
                if (l <= max_l) {
                    const std::vector<double> M = xdm_pure_transform(l, basis.pure_order);
                    transform.insert(transform.end(), M.begin(), M.end());
                }
            }
        }
        m.dbasis.puream = basis.puream;
        m.dbasis.center = m.upload(basis.center);
        m.dbasis.am = m.upload(basis.am);
        m.dbasis.nprim = m.upload(basis.nprim);
        m.dbasis.prim_offset = m.upload(basis.prim_offset);
        m.dbasis.exps = m.upload(basis.exps);
        m.dbasis.coefs = m.upload(basis.coefs);
        m.dbasis.transform = m.upload(transform);
        m.dbasis.transform_offset = m.upload(transform_offset);
        m.union_index.assign(basis.nshell(), -1);

        // ==> Workspace <== //
        // At least one grid block (at most max_chunk_points points) with every function must fit.
        m.workspace_doubles = workspace_bytes / sizeof(double);
        const std::size_t one_block = m.work_need(256, basis.nbf);
        m.workspace_doubles = std::max(m.workspace_doubles, one_block);
        m.max_cols = basis.nbf;
        // The full density matrices move to the device when they take at most half the budget,
        // and come out of it: the device footprint is the budget either way.
        const std::size_t dfull = static_cast<std::size_t>(m.nspin) * basis.nbf * basis.nbf;
        if (resident_density && dfull > 0 && 2 * dfull <= m.workspace_doubles &&
            m.workspace_doubles - dfull >= one_block) {
            m.resident_D = true;
            m.workspace_doubles -= dfull;
        }
        m.d_work = m.alloc<double>(m.workspace_doubles);
        if (m.resident_D) {
            const std::size_t nbf2 = static_cast<std::size_t>(basis.nbf) * basis.nbf;
            m.d_Dfull = m.alloc<double>(dfull);
            for (int s = 0; s < m.nspin; s++) {
                check_cuda(cudaMemcpy(m.d_Dfull + s * nbf2, m.D[s], nbf2 * sizeof(double), cudaMemcpyHostToDevice),
                           "cudaMemcpy");
            }
        }
        m.d_points = m.alloc<double>(11 * max_chunk_points);
        const std::size_t npartial = static_cast<std::size_t>(std::max(m.nactive, 1)) * moment_slots * nmoment;
        m.d_partials = m.alloc<double>(npartial);
        m.d_counts = m.alloc<unsigned long long>(2);
        check_cuda(cudaMemset(m.d_partials, 0, npartial * sizeof(double)), "cudaMemset");
        check_cuda(cudaMemset(m.d_counts, 0, 2 * sizeof(unsigned long long)), "cudaMemset");

        for (int k = 0; k < Impl::nslot; k++) {
            void* h = nullptr;
            check_cuda(cudaMallocHost(&h, 4 * max_chunk_points * sizeof(double)), "cudaMallocHost");
            m.h_points[k] = static_cast<double*>(h);
            check_cuda(cudaEventCreateWithFlags(&m.copied[k], cudaEventDisableTiming), "cudaEventCreate");
        }
        if (!m.resident_D) {
            void* h = nullptr;
            const std::size_t dens_doubles =
                std::min<std::size_t>(m.nspin * m.max_cols * m.max_cols, m.workspace_doubles);
            check_cuda(cudaMallocHost(&h, dens_doubles * sizeof(double)), "cudaMallocHost");
            m.h_dens = static_cast<double*>(h);
            check_cuda(cudaEventCreateWithFlags(&m.dens_copied, cudaEventDisableTiming), "cudaEventCreate");
        }
    } catch (...) {
        m.release();
        throw;
    }
}

DeviceXDMIntegrator::~DeviceXDMIntegrator() {
    if (impl_) impl_->release();
}

void DeviceXDMIntegrator::add_block(std::size_t npoints, const double* x, const double* y, const double* z,
                                    const double* w, const std::vector<int>& shells) {
    Impl& m = *impl_;
    if (m.dev_points[0]) throw PSIEXCEPTION("XDM_GPU: add_block after add_device_blocks.");
    m.add(npoints, x, y, z, w, 0, shells.data(), shells.size());
}

void DeviceXDMIntegrator::add_device_blocks(const DeviceGridBlocks& grid) {
    Impl& m = *impl_;
    if (m.fill_points > 0 || m.sent > 0) throw PSIEXCEPTION("XDM_GPU: add_device_blocks after add_block.");
    m.dev_points[0] = grid.x();
    m.dev_points[1] = grid.y();
    m.dev_points[2] = grid.z();
    m.dev_points[3] = grid.w();
    const auto& start = grid.block_start();
    const auto& count = grid.block_npoints();
    const auto& off = grid.shell_offset();
    const auto& shells = grid.shells();
    for (std::size_t b = 0; b < start.size(); b++) {
        const std::size_t nsh = off[b + 1] - off[b];
        if (nsh == 0) continue;
        m.add(count[b], nullptr, nullptr, nullptr, nullptr, start[b], shells.data() + off[b], nsh);
    }
}

bool DeviceXDMIntegrator::density_resident() const { return impl_->resident_D; }

std::vector<std::array<double, DeviceXDMIntegrator::NMOMENT>> DeviceXDMIntegrator::finish(
    std::size_t& nbhole_failed, std::size_t& nbhole_total) {
    Impl& m = *impl_;
    m.flush();
    check_cuda(cudaStreamSynchronize(stream_handle), "cudaStreamSynchronize");
    m.collect_stages();
    for (int k = 0; k < Impl::nslot; k++) m.copy_pending[k] = false;
    m.dens_pending = false;

    std::vector<double> partials(static_cast<std::size_t>(std::max(m.nactive, 1)) * moment_slots * nmoment);
    unsigned long long counts[2] = {0, 0};
    check_cuda(cudaMemcpy(partials.data(), m.d_partials, partials.size() * sizeof(double), cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    check_cuda(cudaMemcpy(counts, m.d_counts, sizeof(counts), cudaMemcpyDeviceToHost), "cudaMemcpy");

    m.sample_device_memory();
    std::vector<std::array<double, NMOMENT>> sums(m.natom);
    for (auto& s : sums) s.fill(0.0);
    for (int i = 0; i < m.nactive; i++) {
        auto& s = sums[m.active_atoms[i]];
        for (int slot = 0; slot < moment_slots; slot++) {
            const double* src = partials.data() + (static_cast<std::size_t>(i) * moment_slots + slot) * nmoment;
            for (int k = 0; k < nmoment; k++) s[k] += src[k];
        }
    }
    nbhole_total += static_cast<std::size_t>(counts[0]);
    nbhole_failed += static_cast<std::size_t>(counts[1]);
    return sums;
}

std::size_t DeviceXDMIntegrator::device_bytes() const { return impl_->peak_bytes; }
std::size_t DeviceXDMIntegrator::device_growth() const { return impl_->free_at_start - impl_->min_free; }
std::size_t DeviceXDMIntegrator::device_in_use_at_start() const {
    return impl_->total_device - impl_->free_at_start;
}
std::size_t DeviceXDMIntegrator::npoints() const { return impl_->sent + impl_->fill_points; }
std::size_t DeviceXDMIntegrator::nchunks() const { return impl_->chunks; }
std::vector<std::pair<std::string, double>> DeviceXDMIntegrator::stage_seconds() const {
    static const char* names[Impl::NSTAGE] = {"copies", "density gather", "collocation", "GEMM",
                                              "density",  "Becke-Roussel + promolecule", "moments"};
    std::vector<std::pair<std::string, double>> out;
    for (int k = 0; k < Impl::NSTAGE; k++) out.emplace_back(names[k], impl_->stage_ms[k] * 1.0e-3);
    return out;
}
std::array<double, 2> DeviceXDMIntegrator::gemm_work() const {
    const double spin_planes = static_cast<double>(impl_->nspin) * ngemm;
    return {spin_planes * impl_->gemm_done, spin_planes * impl_->gemm_needed};
}

// ==> DeviceGridBlocks <== //

struct DeviceGridBlocks::Impl {
    double* points = nullptr;  // x, y, z, w of the kept points, sorted: 4 * n
    std::size_t n = 0;
    std::vector<std::size_t> start, count;
    std::vector<int> shell_offset{0}, shells;
    std::size_t nwith = 0;
    std::size_t peak = 0;
    std::size_t resident = 0;
    ~Impl() {
        if (points) cudaFree(points);
    }
};

namespace {
// Device buffers of one blocking pass, freed in reverse on every exit and counted for the peak.
struct Scratch {
    std::vector<std::pair<void*, std::size_t>> live;
    std::size_t bytes = 0, peak = 0;
    void* get(std::size_t n) {
        void* p = nullptr;
        check_cuda(cudaMalloc(&p, std::max<std::size_t>(n, 1)), "cudaMalloc");
        live.emplace_back(p, std::max<std::size_t>(n, 1));
        bytes += live.back().second;
        peak = std::max(peak, bytes);
        return p;
    }
    void adopt(void* p, std::size_t n) {
        live.emplace_back(p, n);
        bytes += n;
        peak = std::max(peak, bytes);
    }
    void put(void* p) {
        for (auto it = live.begin(); it != live.end(); ++it) {
            if (it->first == p) {
                cudaFree(p);
                bytes -= it->second;
                live.erase(it);
                return;
            }
        }
    }
    ~Scratch() {
        for (auto it = live.rbegin(); it != live.rend(); ++it) cudaFree(it->first);
    }
};
}  // namespace

DeviceGridBlocks::DeviceGridBlocks(double* d_xyz, double* d_w, std::size_t npoints, double weights_cutoff,
                                   double max_distance, int natom, const std::array<double, 3>* atom_coords,
                                   int max_points, const XDMDeviceBasis& basis,
                                   const std::vector<double>& shell_extents)
    : impl_(new Impl) {
    Scratch scratch;
    // cuEST's arrays are ours from here on, so they are released on every exit.
    scratch.adopt(d_xyz, 3 * npoints * sizeof(double));
    scratch.adopt(d_w, npoints * sizeof(double));
    cuest_common::ensure_cuest_initialized();
    Impl& m = *impl_;
    cudaStream_t st = stream_handle;
    if (npoints > static_cast<std::size_t>(INT_MAX)) {
        throw PSIEXCEPTION("XDM_GRID_BLOCKING DEVICE: more than INT_MAX grid points.");
    }
    if (static_cast<int>(shell_extents.size()) != basis.nshell()) {
        throw PSIEXCEPTION("XDM_GRID_BLOCKING DEVICE: shell extents do not match the basis.");
    }
    const int n = static_cast<int>(npoints);
    std::vector<std::uint64_t> keys;
    unsigned int* d_index = nullptr;
    {
        std::optional<QTimer> stage;
        stage.emplace("XDM: block keys");
        // Lattice box from all points; the sieves run in the key kernel.
        double lo[3] = {0.0, 0.0, 0.0};
        double extent = 0.0;
        if (n > 0) {
            const int nb = std::min((n + block_size - 1) / block_size, 1024);
            auto* d_partial = static_cast<double*>(scratch.get(6 * nb * sizeof(double)));
            bbox_kernel<<<nb, block_size, 0, st>>>(npoints, d_xyz, d_partial);
            check_cuda(cudaGetLastError(), "XDM bounding box kernel launch");
            std::vector<double> partial(6 * nb);
            check_cuda(cudaMemcpyAsync(partial.data(), d_partial, partial.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost, st),
                       "cudaMemcpyAsync");
            check_cuda(cudaStreamSynchronize(st), "cudaStreamSynchronize");
            scratch.put(d_partial);
            double hi[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
            lo[0] = lo[1] = lo[2] = DBL_MAX;
            for (int b = 0; b < nb; b++) {
                for (int k = 0; k < 3; k++) {
                    lo[k] = std::min(lo[k], partial[6 * b + k]);
                    hi[k] = std::max(hi[k], partial[6 * b + 3 + k]);
                }
            }
            for (int k = 0; k < 3; k++) extent = std::max(extent, hi[k] - lo[k]);
        }
        const double scale = extent > 0.0 ? static_cast<double>(blocking::morton_max) / extent : 1.0;

        std::vector<double> axyz(3 * static_cast<std::size_t>(std::max(natom, 1)), 0.0);
        for (int a = 0; a < natom; a++) {
            for (int k = 0; k < 3; k++) axyz[3 * a + k] = atom_coords[a][k];
        }
        auto* d_atoms = static_cast<double*>(scratch.get(axyz.size() * sizeof(double)));
        check_cuda(cudaMemcpyAsync(d_atoms, axyz.data(), axyz.size() * sizeof(double), cudaMemcpyHostToDevice, st),
                   "cudaMemcpyAsync");
        auto* d_keys_in = static_cast<std::uint64_t*>(scratch.get(npoints * sizeof(std::uint64_t)));
        auto* d_index_in = static_cast<unsigned int*>(scratch.get(npoints * sizeof(unsigned int)));
        auto* d_kept = static_cast<unsigned long long*>(scratch.get(sizeof(unsigned long long)));
        check_cuda(cudaMemsetAsync(d_kept, 0, sizeof(unsigned long long), st), "cudaMemsetAsync");
        if (n > 0) {
            key_kernel<<<(n + block_size - 1) / block_size, block_size, 0, st>>>(
                npoints, d_xyz, d_w, weights_cutoff, max_distance >= 0.0, max_distance * max_distance, natom, d_atoms,
                lo[0], lo[1], lo[2], scale, d_keys_in, d_index_in, d_kept);
            check_cuda(cudaGetLastError(), "XDM Morton key kernel launch");
        }
        unsigned long long kept = 0;
        check_cuda(cudaMemcpyAsync(&kept, d_kept, sizeof(kept), cudaMemcpyDeviceToHost, st), "cudaMemcpyAsync");
        check_cuda(cudaStreamSynchronize(st), "cudaStreamSynchronize");
        scratch.put(d_kept);
        scratch.put(d_atoms);
        m.n = static_cast<std::size_t>(kept);

        stage.reset();
        stage.emplace("XDM: block sort");
        // Stable radix sort: equal keys keep their cuEST order, so the result is deterministic.
        auto* d_keys = static_cast<std::uint64_t*>(scratch.get(npoints * sizeof(std::uint64_t)));
        d_index = static_cast<unsigned int*>(scratch.get(npoints * sizeof(unsigned int)));
        std::size_t temp_bytes = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, d_keys_in, d_keys, d_index_in, d_index, n, 0,
                                                   64, st),
                   "cub::DeviceRadixSort::SortPairs");
        void* d_temp = scratch.get(temp_bytes);
        check_cuda(cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_in, d_keys, d_index_in, d_index, n, 0,
                                                   64, st),
                   "cub::DeviceRadixSort::SortPairs");
        keys.resize(m.n);
        check_cuda(cudaMemcpyAsync(keys.data(), d_keys, m.n * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, st),
                   "cudaMemcpyAsync");
        check_cuda(cudaStreamSynchronize(st), "cudaStreamSynchronize");
        scratch.put(d_temp);
        scratch.put(d_keys_in);
        scratch.put(d_index_in);
        scratch.put(d_keys);
    }
    {
        QTimer t("XDM: block gather");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&m.points), std::max<std::size_t>(4 * m.n, 1) * sizeof(double)),
                   "cudaMalloc");
        scratch.bytes += 4 * m.n * sizeof(double);
        scratch.peak = std::max(scratch.peak, scratch.bytes);
        if (m.n > 0) {
            gather_points_kernel<<<static_cast<unsigned int>((m.n + block_size - 1) / block_size), block_size, 0, st>>>(
                m.n, d_index, d_xyz, d_w, m.points);
            check_cuda(cudaGetLastError(), "XDM point gather kernel launch");
        }
        check_cuda(cudaStreamSynchronize(st), "cudaStreamSynchronize");
        scratch.put(d_index);
        scratch.put(d_xyz);
        scratch.put(d_w);
    }
    {
        QTimer t("XDM: block split");
        blocking::split_sorted_keys(keys.data(), m.n, static_cast<std::size_t>(std::max(max_points, 1)), m.start,
                                    m.count);
        keys.clear();
        keys.shrink_to_fit();
    }
    {
        QTimer t("XDM: block screen");
        const int nblk = static_cast<int>(m.start.size());
        const int nshell = basis.nshell();
        const int nword = (nshell + 31) / 32;
        std::vector<unsigned int> mask(static_cast<std::size_t>(nblk) * nword, 0u);
        if (nblk > 0 && nshell > 0) {
            std::vector<unsigned long long> bstart(m.start.begin(), m.start.end());
            std::vector<int> bnp(m.count.begin(), m.count.end());
            auto* d_bstart = static_cast<unsigned long long*>(scratch.get(bstart.size() * sizeof(unsigned long long)));
            auto* d_bnp = static_cast<int*>(scratch.get(bnp.size() * sizeof(int)));
            auto* d_center = static_cast<double*>(scratch.get(basis.center.size() * sizeof(double)));
            auto* d_extent = static_cast<double*>(scratch.get(shell_extents.size() * sizeof(double)));
            auto* d_mask = static_cast<unsigned int*>(scratch.get(mask.size() * sizeof(unsigned int)));
            check_cuda(cudaMemcpyAsync(d_bstart, bstart.data(), bstart.size() * sizeof(unsigned long long),
                                       cudaMemcpyHostToDevice, st),
                       "cudaMemcpyAsync");
            check_cuda(cudaMemcpyAsync(d_bnp, bnp.data(), bnp.size() * sizeof(int), cudaMemcpyHostToDevice, st),
                       "cudaMemcpyAsync");
            check_cuda(cudaMemcpyAsync(d_center, basis.center.data(), basis.center.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, st),
                       "cudaMemcpyAsync");
            check_cuda(cudaMemcpyAsync(d_extent, shell_extents.data(), shell_extents.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, st),
                       "cudaMemcpyAsync");
            screen_kernel<<<nblk, block_size, 0, st>>>(m.points, m.points + m.n, m.points + 2 * m.n, d_bstart, d_bnp,
                                                       nshell, d_center, d_extent, nword, d_mask);
            check_cuda(cudaGetLastError(), "XDM shell screening kernel launch");
            check_cuda(cudaMemcpyAsync(mask.data(), d_mask, mask.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost,
                                       st),
                       "cudaMemcpyAsync");
            check_cuda(cudaStreamSynchronize(st), "cudaStreamSynchronize");
        }
        // Shells in ascending order, as populate() lists them.
        for (int b = 0; b < nblk; b++) {
            const unsigned int* row = mask.data() + static_cast<std::size_t>(b) * nword;
            for (int wd = 0; wd < nword; wd++) {
                for (unsigned int bits = row[wd]; bits; bits &= bits - 1) {
                    m.shells.push_back(32 * wd + __builtin_ctz(bits));
                }
            }
            if (static_cast<int>(m.shells.size()) > m.shell_offset.back()) m.nwith++;
            m.shell_offset.push_back(static_cast<int>(m.shells.size()));
        }
    }
    m.peak = scratch.peak;
    m.resident = 4 * m.n * sizeof(double);
}

DeviceGridBlocks::~DeviceGridBlocks() = default;

std::size_t DeviceGridBlocks::npoints() const { return impl_->n; }
std::size_t DeviceGridBlocks::nblocks() const { return impl_->start.size(); }
std::size_t DeviceGridBlocks::nblocks_with_shells() const { return impl_->nwith; }
const std::vector<std::size_t>& DeviceGridBlocks::block_start() const { return impl_->start; }
const std::vector<std::size_t>& DeviceGridBlocks::block_npoints() const { return impl_->count; }
const std::vector<int>& DeviceGridBlocks::shell_offset() const { return impl_->shell_offset; }
const std::vector<int>& DeviceGridBlocks::shells() const { return impl_->shells; }
const double* DeviceGridBlocks::x() const { return impl_->points; }
const double* DeviceGridBlocks::y() const { return impl_->points + impl_->n; }
const double* DeviceGridBlocks::z() const { return impl_->points + 2 * impl_->n; }
const double* DeviceGridBlocks::w() const { return impl_->points + 3 * impl_->n; }
std::size_t DeviceGridBlocks::peak_device_bytes() const { return impl_->peak; }
std::size_t DeviceGridBlocks::resident_device_bytes() const { return impl_->resident; }

void DeviceGridBlocks::copy_to_host(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z,
                                    std::vector<double>& w) const {
    const std::size_t n = impl_->n;
    std::vector<double>* out[4] = {&x, &y, &z, &w};
    for (int k = 0; k < 4; k++) {
        out[k]->resize(n);
        if (n) {
            check_cuda(cudaMemcpy(out[k]->data(), impl_->points + k * n, n * sizeof(double), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
        }
    }
}

}  // namespace xdm
}  // namespace psi
