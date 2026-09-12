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

cudaError_t cuest_xc_accumulate(std::size_t npoints, double scale, bool has_exc, bool has_gamma, bool has_tau,
                                const double* f, const double* f_rho, const double* f_gamma, const double* f_tau,
                                double* full_f, double* full_f_rho, double* full_f_gamma, double* full_f_tau);

cudaError_t cuest_xc_apply_grac(std::size_t npoints, double alpha, double beta, double shift, const double* rho,
                               const double* gamma, const double* grac_v_rho, double* v_rho, double* v_gamma);

cudaError_t cuest_xc_pack_potential(std::size_t npoints, std::size_t ncomponents, const double* density,
                                   const double* weights, const double* v_rho, const double* v_gamma,
                                   const double* v_tau, double* potential);

}  // namespace psi

#endif
