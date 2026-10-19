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

#include "cuest_xc_gpu.h"
#include "cuest_xc_response_math.h"

#include <cmath>

namespace psi {
namespace {

constexpr int block_size = 256;

__global__ void prepare_inputs_kernel(std::size_t npoints, std::size_t ncomponents, const double* density,
                                      double* rho, double* gamma, double* tau) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;

    const double* values = density + ncomponents * point;
    rho[point] = 2.0 * values[0];
    if (ncomponents >= 4) {
        gamma[point] =
            4.0 * (values[1] * values[1] + values[2] * values[2] + values[3] * values[3]);
    }
    if (ncomponents >= 5) tau[point] = 2.0 * values[4];
}

__global__ void prepare_inputs_polarized_kernel(std::size_t npoints, std::size_t ncomponents,
                                                const double* density_a, const double* density_b, double* rho,
                                                double* gamma, double* tau) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;

    const double* a = density_a + ncomponents * point;
    const double* b = density_b + ncomponents * point;
    rho[2 * point + 0] = a[0];
    rho[2 * point + 1] = b[0];
    if (ncomponents >= 4) {
        gamma[3 * point + 0] = a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
        gamma[3 * point + 1] = a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        gamma[3 * point + 2] = b[1] * b[1] + b[2] * b[2] + b[3] * b[3];
    }
    if (ncomponents >= 5) {
        tau[2 * point + 0] = a[4];
        tau[2 * point + 1] = b[4];
    }
}

// One thread per grid point, so the all-or-nothing finiteness rule of the host
// path -- reject a point outright rather than accumulate half of it -- carries
// over unchanged across however many spin components the field has.
__global__ void accumulate_kernel(std::size_t npoints, double scale, bool has_exc, std::size_t nrho,
                                  std::size_t ngamma, std::size_t ntau, const double* f, const double* f_rho,
                                  const double* f_gamma, const double* f_tau, double* full_f, double* full_f_rho,
                                  double* full_f_gamma, double* full_f_tau) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;

    // f holds nothing for a potential-only functional; it was never written.
    bool finite = true;
    if (has_exc) finite = isfinite(f[point]);
    for (std::size_t i = 0; finite && i < nrho; ++i) finite = isfinite(f_rho[nrho * point + i]);
    for (std::size_t i = 0; finite && i < ngamma; ++i) finite = isfinite(f_gamma[ngamma * point + i]);
    for (std::size_t i = 0; finite && i < ntau; ++i) finite = isfinite(f_tau[ntau * point + i]);
    if (!finite) return;

    if (has_exc) full_f[point] += scale * f[point];
    for (std::size_t i = 0; i < nrho; ++i) full_f_rho[nrho * point + i] += scale * f_rho[nrho * point + i];
    for (std::size_t i = 0; i < ngamma; ++i) full_f_gamma[ngamma * point + i] += scale * f_gamma[ngamma * point + i];
    for (std::size_t i = 0; i < ntau; ++i) full_f_tau[ntau * point + i] += scale * f_tau[ntau * point + i];
}

__global__ void apply_grac_kernel(std::size_t npoints, double alpha, double beta, double shift, const double* rho,
                                  const double* gamma, const double* grac_v_rho, double* v_rho,
                                  double* v_gamma) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;

    const double denx = rho[point] < 1.e-16 ? 1.e2 : sqrt(gamma[point]) / pow(rho[point], 4.0 / 3.0);
    const double grac_fx = 1.0 / (1.0 + exp(-alpha * (denx - beta)));
    const double sr_grac_fx = 1.0 - grac_fx;
    v_rho[point] = sr_grac_fx * (v_rho[point] - shift) + grac_fx * grac_v_rho[point];
    v_gamma[point] *= sr_grac_fx;
}

__global__ void pack_potential_kernel(std::size_t npoints, std::size_t ncomponents, const double* density,
                                      const double* weights, const double* v_rho, const double* v_gamma,
                                      const double* v_tau, double* potential) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;

    const double weight = weights[point];
    const double* density_values = density + ncomponents * point;
    double* values = potential + ncomponents * point;
    values[0] = weight * v_rho[point];
    if (ncomponents >= 4) {
        values[1] = 4.0 * weight * v_gamma[point] * density_values[1];
        values[2] = 4.0 * weight * v_gamma[point] * density_values[2];
        values[3] = 4.0 * weight * v_gamma[point] * density_values[3];
    }
    if (ncomponents >= 5) values[4] = weight * v_tau[point];
}

__global__ void pack_potential_polarized_kernel(std::size_t npoints, std::size_t ncomponents,
                                                const double* density_a, const double* density_b,
                                                const double* weights, const double* v_rho, const double* v_gamma,
                                                const double* v_tau, double* potential_a, double* potential_b) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;

    const double weight = weights[point];
    const double* a = density_a + ncomponents * point;
    const double* b = density_b + ncomponents * point;
    double* out_a = potential_a + ncomponents * point;
    double* out_b = potential_b + ncomponents * point;
    out_a[0] = weight * v_rho[2 * point + 0];
    out_b[0] = weight * v_rho[2 * point + 1];
    if (ncomponents >= 4) {
        // The cross term sigma_ab is shared, so each spin's gradient potential
        // picks up the other spin's density gradient.
        const double v_aa = v_gamma[3 * point + 0];
        const double v_ab = v_gamma[3 * point + 1];
        const double v_bb = v_gamma[3 * point + 2];
        for (int k = 1; k <= 3; ++k) {
            out_a[k] = weight * (2.0 * v_aa * a[k] + v_ab * b[k]);
            out_b[k] = weight * (2.0 * v_bb * b[k] + v_ab * a[k]);
        }
    }
    if (ncomponents >= 5) {
        out_a[4] = weight * v_tau[2 * point + 0];
        out_b[4] = weight * v_tau[2 * point + 1];
    }
}

// => XC response (second-derivative) kernels <= //

__global__ void response_axpy_kernel(std::size_t n, double alpha, const double* x, double* y) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

__global__ void response_count_nonfinite_kernel(std::size_t n, const double* values, unsigned long long* count) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(values[i])) atomicAdd(count, 1ULL);
}

__global__ void response_project_triplet_kernel(std::size_t npoints, bool gga, const double* v2rho2,
                                                const double* v2rhosigma, const double* v2sigma2,
                                                const double* vsigma, double* A, double* B, double* C, double* S) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;
    const auto k = cuest_xc_response::project_triplet(v2rho2 + 3 * point, gga ? v2rhosigma + 6 * point : nullptr,
                                                      gga ? v2sigma2 + 6 * point : nullptr,
                                                      gga ? vsigma + 3 * point : nullptr, gga);
    A[point] = k.A;
    if (gga) {
        B[point] = k.B;
        C[point] = k.C;
        S[point] = k.S;
    }
}

__global__ void response_pack_rks_kernel(std::size_t npoints, bool gga, double cutoff, const double* weights,
                                         const double* reference, const double* A, const double* B, const double* C,
                                         const double* S, const double* response, double* potential) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;
    const std::size_t ncomponents = gga ? 4 : 1;
    cuest_xc_response::RKSKernel k{A[point], 0.0, 0.0, 0.0};
    if (gga) {
        k.B = B[point];
        k.C = C[point];
        k.S = S[point];
    }
    cuest_xc_response::pack_rks(gga, weights[point], cutoff, reference + ncomponents * point, k,
                                response + ncomponents * point, potential + ncomponents * point);
}

__global__ void response_pack_uks_kernel(std::size_t npoints, bool gga, double cutoff, const double* weights,
                                         const double* reference_a, const double* reference_b,
                                         const double* v2rho2, const double* v2rhosigma, const double* v2sigma2,
                                         const double* vsigma, const double* response_a, const double* response_b,
                                         double* potential_a, double* potential_b) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;
    const std::size_t ncomponents = gga ? 4 : 1;
    const std::size_t offset = ncomponents * point;
    cuest_xc_response::pack_uks(gga, weights[point], cutoff, reference_a + offset, reference_b + offset,
                                v2rho2 + 3 * point, gga ? v2rhosigma + 6 * point : nullptr,
                                gga ? v2sigma2 + 6 * point : nullptr, gga ? vsigma + 3 * point : nullptr,
                                response_a + offset, response_b + offset, potential_a + offset,
                                potential_b + offset);
}

inline int blocks(std::size_t npoints) { return static_cast<int>((npoints + block_size - 1) / block_size); }

// A fixed launch shape for the quadrature reduction: the partial count does not
// track npoints, so the summation tree -- and therefore the sum -- is the same on
// every call for a given grid.
constexpr int reduce_block_size = 256;
constexpr int reduce_nblocks = 256;
// reduce_finish_kernel gathers one partial per thread of a single block, so the
// block has to be wide enough to hold them all.
static_assert(reduce_nblocks <= reduce_block_size, "the final reduction would drop partials");

template <int NV>
__device__ void reduce_within_block(double (&acc)[NV], double* out_base) {
    __shared__ double shared[NV][reduce_block_size];
    for (int v = 0; v < NV; ++v) shared[v][threadIdx.x] = acc[v];
    __syncthreads();
    for (int stride = reduce_block_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int v = 0; v < NV; ++v) shared[v][threadIdx.x] += shared[v][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int v = 0; v < NV; ++v) out_base[NV * blockIdx.x + v] = shared[v][0];
    }
}

__global__ void reduce_rks_kernel(std::size_t npoints, const double* weights, const double* full_f,
                                  const double* rho, double* partials) {
    double acc[2] = {0.0, 0.0};
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x; point < npoints; point += stride) {
        const double weighted_rho = weights[point] * rho[point];
        acc[0] += weighted_rho * full_f[point];
        acc[1] += weighted_rho;
    }
    reduce_within_block<2>(acc, partials);
}

__global__ void reduce_uks_kernel(std::size_t npoints, const double* weights, const double* full_f,
                                  const double* rho, double* partials) {
    double acc[3] = {0.0, 0.0, 0.0};
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x; point < npoints; point += stride) {
        const double weight = weights[point];
        const double rho_a = rho[2 * point + 0];
        const double rho_b = rho[2 * point + 1];
        acc[0] += weight * full_f[point] * (rho_a + rho_b);
        acc[1] += weight * rho_a;
        acc[2] += weight * rho_b;
    }
    reduce_within_block<3>(acc, partials);
}

template <int NV>
__global__ void reduce_finish_kernel(const double* partials, double* out) {
    double acc[NV];
    for (int v = 0; v < NV; ++v) {
        acc[v] = (threadIdx.x < reduce_nblocks) ? partials[NV * threadIdx.x + v] : 0.0;
    }
    reduce_within_block<NV>(acc, out);
}

}  // namespace

cudaError_t cuest_xc_prepare_inputs(std::size_t npoints, std::size_t ncomponents, const double* density,
                                    double* rho, double* gamma, double* tau) {
    prepare_inputs_kernel<<<blocks(npoints), block_size>>>(npoints, ncomponents, density, rho, gamma, tau);
    return cudaGetLastError();
}

cudaError_t cuest_xc_prepare_inputs_polarized(std::size_t npoints, std::size_t ncomponents,
                                              const double* density_a, const double* density_b, double* rho,
                                              double* gamma, double* tau) {
    prepare_inputs_polarized_kernel<<<blocks(npoints), block_size>>>(npoints, ncomponents, density_a, density_b,
                                                                     rho, gamma, tau);
    return cudaGetLastError();
}

cudaError_t cuest_xc_accumulate(std::size_t npoints, double scale, bool has_exc, std::size_t nrho,
                                std::size_t ngamma, std::size_t ntau, const double* f, const double* f_rho,
                                const double* f_gamma, const double* f_tau, double* full_f, double* full_f_rho,
                                double* full_f_gamma, double* full_f_tau) {
    accumulate_kernel<<<blocks(npoints), block_size>>>(npoints, scale, has_exc, nrho, ngamma, ntau, f, f_rho,
                                                       f_gamma, f_tau, full_f, full_f_rho, full_f_gamma,
                                                       full_f_tau);
    return cudaGetLastError();
}

cudaError_t cuest_xc_apply_grac(std::size_t npoints, double alpha, double beta, double shift, const double* rho,
                               const double* gamma, const double* grac_v_rho, double* v_rho, double* v_gamma) {
    apply_grac_kernel<<<blocks(npoints), block_size>>>(npoints, alpha, beta, shift, rho, gamma, grac_v_rho, v_rho,
                                                       v_gamma);
    return cudaGetLastError();
}

cudaError_t cuest_xc_pack_potential(std::size_t npoints, std::size_t ncomponents, const double* density,
                                   const double* weights, const double* v_rho, const double* v_gamma,
                                   const double* v_tau, double* potential) {
    pack_potential_kernel<<<blocks(npoints), block_size>>>(npoints, ncomponents, density, weights, v_rho, v_gamma,
                                                           v_tau, potential);
    return cudaGetLastError();
}

cudaError_t cuest_xc_pack_potential_polarized(std::size_t npoints, std::size_t ncomponents, const double* density_a,
                                              const double* density_b, const double* weights, const double* v_rho,
                                              const double* v_gamma, const double* v_tau, double* potential_a,
                                              double* potential_b) {
    pack_potential_polarized_kernel<<<blocks(npoints), block_size>>>(npoints, ncomponents, density_a, density_b,
                                                                     weights, v_rho, v_gamma, v_tau, potential_a,
                                                                     potential_b);
    return cudaGetLastError();
}

std::size_t cuest_xc_reduce_scratch(std::size_t nvalues) { return nvalues * reduce_nblocks; }

cudaError_t cuest_xc_reduce_rks(std::size_t npoints, const double* weights, const double* full_f, const double* rho,
                                double* scratch, double* out) {
    reduce_rks_kernel<<<reduce_nblocks, reduce_block_size>>>(npoints, weights, full_f, rho, scratch);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    reduce_finish_kernel<2><<<1, reduce_block_size>>>(scratch, out);
    return cudaGetLastError();
}

cudaError_t cuest_xc_reduce_uks(std::size_t npoints, const double* weights, const double* full_f, const double* rho,
                                double* scratch, double* out) {
    reduce_uks_kernel<<<reduce_nblocks, reduce_block_size>>>(npoints, weights, full_f, rho, scratch);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    reduce_finish_kernel<3><<<1, reduce_block_size>>>(scratch, out);
    return cudaGetLastError();
}

cudaError_t cuest_xc_response_axpy(std::size_t n, double alpha, const double* x, double* y) {
    if (n == 0) return cudaSuccess;
    response_axpy_kernel<<<blocks(n), block_size>>>(n, alpha, x, y);
    return cudaGetLastError();
}

cudaError_t cuest_xc_response_count_nonfinite(std::size_t n, const double* values, unsigned long long* count) {
    if (n == 0) return cudaSuccess;
    response_count_nonfinite_kernel<<<blocks(n), block_size>>>(n, values, count);
    return cudaGetLastError();
}

cudaError_t cuest_xc_response_project_triplet(std::size_t npoints, bool gga, const double* v2rho2,
                                              const double* v2rhosigma, const double* v2sigma2, const double* vsigma,
                                              double* A, double* B, double* C, double* S) {
    if (npoints == 0) return cudaSuccess;
    response_project_triplet_kernel<<<blocks(npoints), block_size>>>(npoints, gga, v2rho2, v2rhosigma, v2sigma2,
                                                                     vsigma, A, B, C, S);
    return cudaGetLastError();
}

cudaError_t cuest_xc_response_pack_rks(std::size_t npoints, bool gga, double cutoff, const double* weights,
                                       const double* reference, const double* A, const double* B, const double* C,
                                       const double* S, const double* response, double* potential) {
    if (npoints == 0) return cudaSuccess;
    response_pack_rks_kernel<<<blocks(npoints), block_size>>>(npoints, gga, cutoff, weights, reference, A, B, C, S,
                                                              response, potential);
    return cudaGetLastError();
}

cudaError_t cuest_xc_response_pack_uks(std::size_t npoints, bool gga, double cutoff, const double* weights,
                                       const double* reference_a, const double* reference_b, const double* v2rho2,
                                       const double* v2rhosigma, const double* v2sigma2, const double* vsigma,
                                       const double* response_a, const double* response_b, double* potential_a,
                                       double* potential_b) {
    if (npoints == 0) return cudaSuccess;
    response_pack_uks_kernel<<<blocks(npoints), block_size>>>(npoints, gga, cutoff, weights, reference_a,
                                                              reference_b, v2rho2, v2rhosigma, v2sigma2, vsigma,
                                                              response_a, response_b, potential_a, potential_b);
    return cudaGetLastError();
}

}  // namespace psi
