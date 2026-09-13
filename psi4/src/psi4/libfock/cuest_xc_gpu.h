/*
 * Device-side packing helpers for the CUDA LibXC/cuEST XC path.
 */

#ifndef PSI4_LIBFOCK_CUEST_XC_GPU_H
#define PSI4_LIBFOCK_CUEST_XC_GPU_H

#include <cstddef>
#include <cuda_runtime_api.h>

namespace psi {

cudaError_t cuest_xc_prepare_inputs(std::size_t npoints, std::size_t ncomponents, const double* density,
                                    double* rho, double* gamma, double* tau);

// Spin-polarized inputs in LibXC's interleaved layout: rho[2p+s], sigma[3p+{aa,ab,bb}],
// tau[2p+s], built from cuEST's two point-major spin densities.
cudaError_t cuest_xc_prepare_inputs_polarized(std::size_t npoints, std::size_t ncomponents,
                                              const double* density_a, const double* density_b, double* rho,
                                              double* gamma, double* tau);

// nrho/ngamma/ntau are components per point: 1/1/1 unpolarized, 2/3/2 polarized.
// Zero means the ansatz does not carry that field.
cudaError_t cuest_xc_accumulate(std::size_t npoints, double scale, bool has_exc, std::size_t nrho,
                                std::size_t ngamma, std::size_t ntau, const double* f, const double* f_rho,
                                const double* f_gamma, const double* f_tau, double* full_f, double* full_f_rho,
                                double* full_f_gamma, double* full_f_tau);

cudaError_t cuest_xc_apply_grac(std::size_t npoints, double alpha, double beta, double shift, const double* rho,
                               const double* gamma, const double* grac_v_rho, double* v_rho, double* v_gamma);

cudaError_t cuest_xc_pack_potential(std::size_t npoints, std::size_t ncomponents, const double* density,
                                   const double* weights, const double* v_rho, const double* v_gamma,
                                   const double* v_tau, double* potential);

cudaError_t cuest_xc_pack_potential_polarized(std::size_t npoints, std::size_t ncomponents, const double* density_a,
                                              const double* density_b, const double* weights, const double* v_rho,
                                              const double* v_gamma, const double* v_tau, double* potential_a,
                                              double* potential_b);

// Doubles of device scratch cuest_xc_reduce_* needs; nvalues is 2 for RKS, 3 for UKS.
std::size_t cuest_xc_reduce_scratch(std::size_t nvalues);

// Quadrature sums done on the device, so only the handful of resulting scalars
// crosses the bus instead of the energy density and the density over the whole
// grid. The reduction runs over a fixed block count and a fixed tree, so a given
// grid gives the same sum every time rather than whatever order atomics landed in.
//
// RKS out: {Exc, integrated density}, with rho spin-summed as prepare_inputs leaves it.
cudaError_t cuest_xc_reduce_rks(std::size_t npoints, const double* weights, const double* full_f, const double* rho,
                                double* scratch, double* out);

// UKS out: {Exc, integrated alpha density, integrated beta density}, rho interleaved as rho[2p+s].
cudaError_t cuest_xc_reduce_uks(std::size_t npoints, const double* weights, const double* full_f, const double* rho,
                                double* scratch, double* out);

}  // namespace psi

#endif
