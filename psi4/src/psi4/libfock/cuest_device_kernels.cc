/* SPDX-License-Identifier: LGPL-3.0-only
 *
 * Device-resident CG follows the cuEST/cuBLAS usage patterns in NVIDIA's
 * CUDALibrarySamples/cuEST/cuest_scf_examples/examples/cphf-1. The algebra
 * below follows Psi4's signs and independent, coupled-spin RHS convention.
 * No secondary CUDA context or default-stream launches are introduced.
 */
#include "cuest_device_kernels.h"
#ifdef PSI4_CUEST_DEVICE_CG
#include "cuESTCommon.h"
#include "cuest_device_math_source.h"
#include <cuda.h>
#include <nvrtc.h>
#include <limits>
#include <map>
#include <string>
#include <vector>

extern cudaStream_t stream_handle;
namespace psi {
namespace {
void driver_check(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* text = nullptr;
    cuGetErrorString(result, &text);
    throw PSIEXCEPTION(std::string("cuEST device CG: ") + operation + ": " + (text ? text : "CUDA driver error"));
}
const char* source = R"CUDA(
#include "cuest_xc_response_math.h"
typedef unsigned long long U;
__device__ bool finite_value(double x) {
    union { double d; U u; } v; v.d = x;
    return (v.u & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
extern "C" __global__ void cg_init(U n, U stride, double tol2, const double* gap,
    const double* b, const double* bn, double* x, double* r, double* z, double* p, int* active, int* error) {
    U i = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    U v = i / stride;
    double denom = bn[v] > 1.e-14 ? bn[v] : 1.e-14;
    bool valid = finite_value(bn[v]);
    bool on = valid && bn[v] / denom > tol2;
    if (i % stride == 0) { active[v] = on; if (!valid) atomicOr(error, 1); }
    x[i] = 0.; r[i] = b[i]; z[i] = b[i] / gap[i % stride]; p[i] = on ? z[i] : 0.;
}
extern "C" __global__ void cg_step(U n, U stride, double* x, double* r, const double* p,
    const double* ap, const double* rho, const double* sigma, const int* active, int* error) {
    U i = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !active[i / stride]) return;
    U v = i / stride;
    double a = rho[v] / sigma[v];
    if (!(rho[v] > 0.) || !finite_value(rho[v]) ||
        !(sigma[v] > 0.) || !finite_value(sigma[v]) || !finite_value(a)) {
        atomicOr(error, 2); return;
    }
    x[i] += a * p[i]; r[i] -= a * ap[i];
}
extern "C" __global__ void cg_precondition(U n, U stride, const double* gap, const double* r, double* z) {
    U i = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = r[i] / gap[i % stride];
}
extern "C" __global__ void cg_metrics(U nrhs, double tol2, const double* rr, const double* bn,
    double* metric, int* active, int* error) {
    U v = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nrhs) return;
    double denom = bn[v] > 1.e-14 ? bn[v] : 1.e-14;
    double value = rr[v] / denom;
    bool valid = finite_value(value) && value >= 0.;
    if (!valid) atomicOr(error, 4);
    metric[v] = valid ? value : 1.e300;
    active[v] = valid && value > tol2;
}
extern "C" __global__ void cg_direction(U n, U stride, int restart, const double* z, double* p,
    const double* oldrho, const double* newrho, const int* active, int* error) {
    U i = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    U v = i / stride;
    if (!active[v]) { p[i] = 0.; return; }
    double beta = restart ? 0. : newrho[v] / oldrho[v];
    if (!(newrho[v] > 0.) || !finite_value(newrho[v]) ||
        (!restart && (!(oldrho[v] > 0.) || !finite_value(oldrho[v]))) || !finite_value(beta)) {
        atomicOr(error, 8); p[i] = 0.; return;
    }
    p[i] = z[i] + beta * p[i];
}
extern "C" __global__ void cg_true_residual(U n, U stride, const double* b, const double* ax,
    const double* gap, double* r, double* z) {
    U i = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { r[i] = b[i] - ax[i]; z[i] = r[i] / gap[i % stride]; }
}
extern "C" __global__ void cg_symmetric_density(U n, U nbf, const double* d, double* s) {
    U i = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    U pq = i % (nbf * nbf);
    U transpose = i - pq + (pq % nbf) * nbf + pq / nbf;
    s[i] = .5 * d[i] + .5 * d[transpose];
}
extern "C" __global__ void cg_assemble(U nbf, U nrhs, U nspin, int use_k, int use_xc,
    double scale_k, const double* j, const double* k, const double* v, double* g) {
    U i = (U)blockIdx.x * blockDim.x + threadIdx.x, n2 = nbf * nbf;
    if (i >= n2 * nrhs * nspin) return;
    U within = i % (nrhs * n2), pq = i % n2;
    double value = nspin == 1 ? 4. * j[i] : 2. * (j[within] + j[nrhs * n2 + within]);
    if (use_k) value -= scale_k * (k[i] + k[i - pq + (pq % nbf) * nbf + pq / nbf]);
    if (use_xc) value += (nspin == 1 ? 4. : 2.) * v[i];
    g[i] = value;
}
extern "C" __global__ void cg_pack_rks(U n, int gga, double cutoff, const double* weights,
    const double* ref, const double* a, const double* b, const double* c, const double* s,
    const double* response, double* potential) {
    U p = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    psi::cuest_xc_response::RKSKernel kernel{a[p], gga ? b[p] : 0., gga ? c[p] : 0., gga ? s[p] : 0.};
    U o = (gga ? 4 : 1) * p;
    psi::cuest_xc_response::pack_rks(gga, weights[p], cutoff, ref + o, kernel, response + o, potential + o);
}
extern "C" __global__ void cg_pack_uks(U n, int gga, double cutoff, const double* weights,
    const double* ra, const double* rb, const double* r2, const double* rs, const double* s2,
    const double* s1, const double* da, const double* db, double* va, double* vb) {
    U p = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    U o = (gga ? 4 : 1) * p;
    psi::cuest_xc_response::pack_uks(gga, weights[p], cutoff, ra + o, rb + o, r2 + 3*p,
        gga ? rs + 6*p : 0, gga ? s2 + 6*p : 0, gga ? s1 + 3*p : 0,
        da + o, db + o, va + o, vb + o);
}
)CUDA";
}
struct CuestDeviceKernels::Impl {
    CUmodule module = nullptr;
    int device = -1;
    std::map<std::string, CUfunction> functions;
    ~Impl() { if (module) cuModuleUnload(module); }
};
CuestDeviceKernels& CuestDeviceKernels::instance() {
    static CuestDeviceKernels kernels;
    return kernels;
}
CuestDeviceKernels::CuestDeviceKernels() : impl_(std::make_unique<Impl>()) {
    cuest_common::ensure_cuest_initialized();
    cudaDeviceProp prop{};
    if (cudaGetDevice(&impl_->device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, impl_->device) != cudaSuccess)
        throw PSIEXCEPTION("cuEST device CG: cannot query the active CUDA device.");
    nvrtcProgram program = nullptr;
    const char* header_name = "cuest_xc_response_math.h";
    auto status = nvrtcCreateProgram(&program, source, "psi4_device_cg.cu", 1,
                                      &cuest_response_math_source, &header_name);
    if (status != NVRTC_SUCCESS) throw PSIEXCEPTION("cuEST device CG: nvrtcCreateProgram failed.");
    struct ProgramGuard { nvrtcProgram& p; ~ProgramGuard() { nvrtcDestroyProgram(&p); } } guard{program};
    const std::string arch = "--gpu-architecture=compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
    const char* options[] = {"--std=c++14", arch.c_str()};
    status = nvrtcCompileProgram(program, 2, options);
    if (status != NVRTC_SUCCESS) {
        std::size_t size = 0;
        nvrtcGetProgramLogSize(program, &size);
        std::vector<char> log(size + 1, 0);
        nvrtcGetProgramLog(program, log.data());
        throw PSIEXCEPTION(std::string("cuEST device CG: NVRTC compilation failed: ") + log.data());
    }
    std::size_t size = 0;
    if (nvrtcGetPTXSize(program, &size) != NVRTC_SUCCESS)
        throw PSIEXCEPTION("cuEST device CG: cannot query PTX size.");
    std::vector<char> ptx(size);
    if (nvrtcGetPTX(program, ptx.data()) != NVRTC_SUCCESS)
        throw PSIEXCEPTION("cuEST device CG: cannot retrieve PTX.");
    driver_check(cuModuleLoadData(&impl_->module, ptx.data()), "load device kernels");
    for (const char* name : {"cg_init", "cg_step", "cg_precondition", "cg_metrics", "cg_direction",
                              "cg_true_residual", "cg_symmetric_density", "cg_assemble", "cg_pack_rks", "cg_pack_uks"})
        driver_check(cuModuleGetFunction(&impl_->functions[name], impl_->module, name), name);
}
CuestDeviceKernels::~CuestDeviceKernels() = default;
void CuestDeviceKernels::launch_impl(const char* name, std::size_t count, void** parameters) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != impl_->device)
        throw PSIEXCEPTION("cuEST device CG: active CUDA device changed during a solve.");
    const auto blocks = (count + 255) / 256;
    if (blocks > std::numeric_limits<unsigned int>::max())
        throw PSIEXCEPTION("cuEST device CG: kernel launch exceeds CUDA grid limits.");
    driver_check(cuLaunchKernel(impl_->functions.at(name), static_cast<unsigned int>(blocks), 1, 1,
                               256, 1, 1, 0, reinterpret_cast<CUstream>(stream_handle), parameters, nullptr), name);
}
}  // namespace psi
#endif
