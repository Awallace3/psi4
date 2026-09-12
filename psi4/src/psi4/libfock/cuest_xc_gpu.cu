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

__global__ void accumulate_kernel(std::size_t npoints, double scale, bool has_exc, bool has_gamma, bool has_tau,
                                  const double* f, const double* f_rho, const double* f_gamma, const double* f_tau,
                                  double* full_f, double* full_f_rho, double* full_f_gamma, double* full_f_tau) {
    const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoints) return;

    // f holds nothing for a potential-only functional; it was never written.
    bool finite = isfinite(f_rho[point]);
    if (has_exc) finite = finite && isfinite(f[point]);
    if (has_gamma) finite = finite && isfinite(f_gamma[point]);
    if (has_tau) finite = finite && isfinite(f_tau[point]);
    if (!finite) return;

    if (has_exc) full_f[point] += scale * f[point];
    full_f_rho[point] += scale * f_rho[point];
    if (has_gamma) full_f_gamma[point] += scale * f_gamma[point];
    if (has_tau) full_f_tau[point] += scale * f_tau[point];
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

inline int blocks(std::size_t npoints) { return static_cast<int>((npoints + block_size - 1) / block_size); }

}  // namespace

cudaError_t cuest_xc_prepare_inputs(std::size_t npoints, std::size_t ncomponents, const double* density,
                                    double* rho, double* gamma, double* tau) {
    prepare_inputs_kernel<<<blocks(npoints), block_size>>>(npoints, ncomponents, density, rho, gamma, tau);
    return cudaGetLastError();
}

cudaError_t cuest_xc_accumulate(std::size_t npoints, double scale, bool has_exc, bool has_gamma, bool has_tau,
                                const double* f, const double* f_rho, const double* f_gamma, const double* f_tau,
                                double* full_f, double* full_f_rho, double* full_f_gamma, double* full_f_tau) {
    accumulate_kernel<<<blocks(npoints), block_size>>>(npoints, scale, has_exc, has_gamma, has_tau, f, f_rho,
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

}  // namespace psi
