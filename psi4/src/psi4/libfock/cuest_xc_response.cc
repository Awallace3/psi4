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

/*
 * LDA/GGA XC response kernel (compute_Vx) on the cuEST grid.
 *
 * The response density never enters LibXC. The first and second derivatives are
 * evaluated at the physical reference density -- on the host with the native
 * SuperFunctional workers, or on the device with CUDA LibXC -- and contracted with
 * the signed (possibly nonsymmetric) response density by cuest_xc_response_math.h,
 * which reproduces RV::compute_Vx_full and UV::compute_Vx point by point.
 *
 * cuestXCNonsymmetricDensityCompute evaluates rho[L R^T] = sum_i L_i(r) R_i(r) and its
 * gradient for factors L, R (nao x rank). Solver factors are used as given; a dense
 * response matrix D is passed as D = I D, in identity-column tiles, so arbitrary
 * signed and nonsymmetric matrices are reproduced without any factorization.
 */

#include "psi4/libfock/v.h"

#ifdef USING_cuEST

#include "psi4/libfock/cubature.h"
#include "psi4/libfock/cuESTCommon.h"
#include "psi4/libfock/cuest_xc_response.h"
#include "psi4/libfock/cuest_xc_response_math.h"
#include "psi4/libfunctional/LibXCfunctional.h"
#include "psi4/libfunctional/functional.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libfunctional/xc_timing.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/liboptions/liboptions.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <xc.h>

#ifdef USING_Libxc_CUDA
#include "psi4/libfock/cuest_xc_gpu.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern cuestHandle_t cuest_handle;

namespace psi {

// => Workspace <= //

void* CuestDeviceArray::ensure_bytes(std::size_t bytes, const char* what) {
    if (bytes == 0) return pointer_;
    if (bytes > capacity_) {
        release();
        void* pointer = nullptr;
        const cudaError_t status = cudaMalloc(&pointer, bytes);
        if (status != cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            const bool known = cuest_common::device_memory(&free_bytes, &total_bytes);
            std::string message = "cuEST XC response: device allocation of " +
                                  std::to_string(bytes * cuest_common::to_gib) + " GiB for " + what +
                                  " failed: " + cudaGetErrorString(status);
            if (known) {
                message += " (" + std::to_string(free_bytes * cuest_common::to_gib) + " of " +
                           std::to_string(total_bytes * cuest_common::to_gib) + " GiB free)";
            }
            throw PSIEXCEPTION(message);
        }
        pointer_ = pointer;
        capacity_ = bytes;
    }
    return pointer_;
}

void CuestDeviceArray::release() {
    if (pointer_) cudaFree(pointer_);
    pointer_ = nullptr;
    capacity_ = 0;
}

CuestXCResponseWorkspace::~CuestXCResponseWorkspace() { std::free(sdk_host); }

cuestWorkspace_t* CuestXCResponseWorkspace::sdk_workspace(const cuestWorkspaceDescriptor_t& descriptor) {
    void* device = sdk_device.ensure_bytes(descriptor.deviceBufferSizeInBytes, "the cuEST temporary workspace");
    if (descriptor.hostBufferSizeInBytes > sdk_host_capacity) {
        std::free(sdk_host);
        sdk_host = nullptr;
        sdk_host_capacity = 0;
        sdk_host = std::malloc(descriptor.hostBufferSizeInBytes);
        if (!sdk_host) throw PSIEXCEPTION("cuEST XC response: host workspace allocation failed.");
        sdk_host_capacity = descriptor.hostBufferSizeInBytes;
    }
    // Describe exactly the requested sizes; the buffers behind them may be larger.
    sdk_view.deviceBufferSizeInBytes = descriptor.deviceBufferSizeInBytes;
    sdk_view.hostBufferSizeInBytes = descriptor.hostBufferSizeInBytes;
    sdk_view.deviceBuffer = descriptor.deviceBufferSizeInBytes ? reinterpret_cast<uintptr_t>(device) : 0;
    sdk_view.hostBuffer = descriptor.hostBufferSizeInBytes ? reinterpret_cast<uintptr_t>(sdk_host) : 0;
    return &sdk_view;
}

bool VBase::cuest_Cocc_reproduces_D() {
    if (cuest_Cocc_matches_D_ >= 0) return cuest_Cocc_matches_D_ == 1;
    bool matches = cuest_Cocc_.size() == D_AO_.size() && d_Cocc_noccs_.size() == D_AO_.size() && d_Coccs_AO_;
    for (size_t s = 0; matches && s < D_AO_.size(); ++s) {
        const auto& C = cuest_Cocc_[s];
        const auto& D = D_AO_[s];
        if (C->nirrep() != 1 || C->rowdim() != D->rowdim() || D->rowdim() != D->coldim() ||
            static_cast<uint64_t>(C->coldim()) != d_Cocc_noccs_[s]) {
            matches = false;
            break;
        }
        const auto CCt = linalg::doublet(C, C, false, true);
        const double scale = std::max(1.0, D->absmax());
        CCt->subtract(D);
        matches = CCt->absmax() <= 1.0e-12 * scale;
    }
    cuest_Cocc_matches_D_ = matches ? 1 : 0;
    return matches;
}

namespace {

#ifdef PSI4_CUEST_NONSYMMETRIC_XC

#define RESPONSE_CHECK_CUEST(operation)                                                                \
    xc_timing::call(Process::environment.options.get_bool("CUEST_XC_TIMING"), #operation,              \
                    [&]() { CHECK_CUEST(operation); }, true)

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw PSIEXCEPTION(std::string("cuEST XC response: ") + operation + " failed: " + cudaGetErrorString(status));
    }
}

/// Stage timers that include device completion, so asynchronous work is charged to
/// the stage that launched it rather than to whichever stage synchronizes next.
class ResponseStages {
    xc_timing::CoarseScope scope_;
    bool active_ = false;

   public:
    void start(const char* label) {
        finish();
        scope_.start(label);
        active_ = true;
    }
    void finish() {
        if (!active_) return;
        check_cuda(cudaDeviceSynchronize(), "device synchronization");
        scope_.stop();
        active_ = false;
    }
};

/// RAII ownership of one cuEST compute-parameter object.
template <typename Handle>
class CuestParameters {
    cuestParametersType_t type_;
    Handle handle_{};

   public:
    explicit CuestParameters(cuestParametersType_t type) : type_(type) {
        CHECK_CUEST(cuestParametersCreate(type_, &handle_));
    }
    CuestParameters(const CuestParameters&) = delete;
    CuestParameters& operator=(const CuestParameters&) = delete;
    ~CuestParameters() { cuestParametersDestroy(type_, handle_); }
    Handle get() const { return handle_; }
};

/// Everything the grid-side helpers share for one response call.
struct GridContext {
    cuestXCIntPlan_t plan;
    cuestXCAdvancedComputeParametersApproximation_t approximation;
    size_t npoints;
    size_t ncomponents;
    size_t nbf;
    CuestXCResponseWorkspace& ws;
    cuestWorkspaceDescriptor_t budget;
    bool verbose;
};

/// Signed, nonsymmetric grid density of D = left_t^T right_t, where left_t/right_t are
/// device (rank x nao) row-major arrays, i.e. the cuEST orbital layout of nao x rank factors.
void nonsymmetric_density(GridContext& g, uint64_t rank, const double* left_t, const double* right_t,
                          double* out) {
    if (rank == 0) {
        check_cuda(cudaMemset(out, 0, g.npoints * g.ncomponents * sizeof(double)), "zero-rank density");
        return;
    }
    CuestParameters<cuestXCNonsymmetricDensityComputeParameters_t> parameters(
        CUEST_XCNONSYMMETRICDENSITYCOMPUTE_PARAMETERS);
    cuestWorkspaceDescriptor_t temporary{};
    RESPONSE_CHECK_CUEST(cuestXCNonsymmetricDensityComputeWorkspaceQuery(
        cuest_handle, g.plan, g.approximation, parameters.get(), &g.budget, &temporary, rank, left_t, right_t, out));
    if (g.verbose) {
        outfile->Printf("    cuEST Vx: nonsymmetric density rank %8llu workspace %10.3f MiB\n",
                        static_cast<unsigned long long>(rank), temporary.deviceBufferSizeInBytes / 1048576.0);
    }
    RESPONSE_CHECK_CUEST(cuestXCNonsymmetricDensityCompute(cuest_handle, g.plan, g.approximation, parameters.get(),
                                                           &g.budget, g.ws.sdk_workspace(temporary), rank, left_t,
                                                           right_t, out));
}

/// Symmetric occupied-orbital density, exactly as the SCF potential evaluates it.
void occupied_density(GridContext& g, uint64_t nocc, const double* Cocc_t, double* out) {
    if (nocc == 0) {
        check_cuda(cudaMemset(out, 0, g.npoints * g.ncomponents * sizeof(double)), "empty-spin density");
        return;
    }
    CuestParameters<cuestXCDensityComputeParameters_t> parameters(CUEST_XCDENSITYCOMPUTE_PARAMETERS);
    cuestWorkspaceDescriptor_t temporary{};
    RESPONSE_CHECK_CUEST(cuestXCDensityComputeWorkspaceQuery(cuest_handle, g.plan, g.approximation, parameters.get(),
                                                             &g.budget, &temporary, nocc, Cocc_t, out));
    RESPONSE_CHECK_CUEST(cuestXCDensityCompute(cuest_handle, g.plan, g.approximation, parameters.get(), &g.budget,
                                               g.ws.sdk_workspace(temporary), nocc, Cocc_t, out));
}

/// Columns of the identity (as cuEST rows) per tile, bounding the two staged factors.
size_t identity_tile(size_t nbf) {
    const size_t budget = size_t(256) << 20;
    return std::max<size_t>(1, std::min(nbf, budget / (2 * sizeof(double) * std::max<size_t>(nbf, 1))));
}

/// Where a density result has to end up: on the device (CUDA LibXC packing) or the host.
struct DensityTarget {
    double* device = nullptr;      ///< CUDA route: final device density.
    std::vector<double>* host = nullptr;  ///< Host route: final host density.
};

void download(const double* device, size_t count, double* host) {
    check_cuda(cudaMemcpy(host, device, count * sizeof(double), cudaMemcpyDeviceToHost), "density download");
}

/// Grid density of an arbitrary dense C1 AO matrix D, written as sum_k e_k D[k,:].
/// Tiles of identity rows k0..k1 (left) and the matching rows of D (right) are summed
/// after each density, so nothing is truncated, symmetrized or factorized.
void dense_density(GridContext& g, const Matrix& D, const DensityTarget& target) {
    const size_t nbf = g.nbf;
    const size_t field = g.npoints * g.ncomponents;
    if (D.nirrep() != 1 || static_cast<size_t>(D.rowdim()) != nbf || static_cast<size_t>(D.coldim()) != nbf) {
        throw PSIEXCEPTION("cuEST XC response: dense response density must be a C1 nbf x nbf AO matrix.");
    }
    const size_t tile = identity_tile(nbf);
    double* left = g.ws.left.ensure(tile * nbf, "identity factor tile");
    double* right = g.ws.right.ensure(tile * nbf, "response factor tile");
    std::vector<double> ones(tile, 1.0);
    std::vector<double> scratch;
    bool first = true;
    for (size_t k0 = 0; k0 < nbf; k0 += tile) {
        const size_t rank = std::min(tile, nbf - k0);
        check_cuda(cudaMemset(left, 0, rank * nbf * sizeof(double)), "identity tile reset");
        // Element (k, k0 + k) of the row-major rank x nbf tile.
        check_cuda(cudaMemcpy2D(left + k0, (nbf + 1) * sizeof(double), ones.data(), sizeof(double), sizeof(double),
                                rank, cudaMemcpyHostToDevice),
                   "identity tile upload");
        check_cuda(cudaMemcpy(right, D.pointer()[k0], rank * nbf * sizeof(double), cudaMemcpyHostToDevice),
                   "dense response upload");
        if (target.device) {
            double* out = first ? target.device : g.ws.tile.ensure(field, "density tile");
            nonsymmetric_density(g, rank, left, right, out);
#ifdef USING_Libxc_CUDA
            if (!first) check_cuda(cuest_xc_response_axpy(field, 1.0, out, target.device), "density tile sum");
#else
            if (!first) throw PSIEXCEPTION("cuEST XC response: device tile accumulation requires CUDA LibXC.");
#endif
        } else {
            double* out = g.ws.tile.ensure(field, "density tile");
            nonsymmetric_density(g, rank, left, right, out);
            if (first) {
                target.host->resize(field);
                download(out, field, target.host->data());
            } else {
                scratch.resize(field);
                download(out, field, scratch.data());
                double* sum = target.host->data();
                for (size_t i = 0; i < field; ++i) sum[i] += scratch[i];
            }
        }
        first = false;
    }
}

/// Grid density of a C1 AO factor pair, Dx = L R^T.
void factored_density(GridContext& g, const Matrix& L, const Matrix& R, const DensityTarget& target) {
    const size_t nbf = g.nbf;
    const size_t field = g.npoints * g.ncomponents;
    if (L.nirrep() != 1 || R.nirrep() != 1 || static_cast<size_t>(L.rowdim()) != nbf ||
        static_cast<size_t>(R.rowdim()) != nbf || L.coldim() != R.coldim()) {
        throw PSIEXCEPTION("cuEST XC response: factors must be C1 nbf x rank AO matrices of equal rank.");
    }
    const size_t rank = L.coldim();
    double* out = target.device ? target.device : g.ws.tile.ensure(field, "density");
    if (rank == 0) {
        nonsymmetric_density(g, 0, nullptr, nullptr, out);
    } else {
        auto Lt = L.transpose();
        auto Rt = R.transpose();
        double* left = g.ws.left.ensure(rank * nbf, "left response factor");
        double* right = g.ws.right.ensure(rank * nbf, "right response factor");
        check_cuda(cudaMemcpy(left, Lt->pointer()[0], rank * nbf * sizeof(double), cudaMemcpyHostToDevice),
                   "left factor upload");
        check_cuda(cudaMemcpy(right, Rt->pointer()[0], rank * nbf * sizeof(double), cudaMemcpyHostToDevice),
                   "right factor upload");
        nonsymmetric_density(g, rank, left, right, out);
    }
    if (!target.device) {
        target.host->resize(field);
        download(out, field, target.host->data());
    }
}

/// AO matrix of a weighted grid potential (cuEST adds the basis-function products only).
SharedMatrix ao_potential(GridContext& g, const double* grid_potential) {
    double* ao = g.ws.ao.ensure(g.nbf * g.nbf, "AO response matrix");
    CuestParameters<cuestXCPotentialComputeParameters_t> parameters(CUEST_XCPOTENTIALCOMPUTE_PARAMETERS);
    cuestWorkspaceDescriptor_t temporary{};
    RESPONSE_CHECK_CUEST(cuestXCPotentialComputeWorkspaceQuery(cuest_handle, g.plan, g.approximation, parameters.get(),
                                                               &g.budget, &temporary, grid_potential, ao));
    if (g.verbose) {
        outfile->Printf("    cuEST Vx: AO integration workspace %10.3f MiB\n",
                        temporary.deviceBufferSizeInBytes / 1048576.0);
    }
    RESPONSE_CHECK_CUEST(cuestXCPotentialCompute(cuest_handle, g.plan, g.approximation, parameters.get(), &g.budget,
                                                 g.ws.sdk_workspace(temporary), grid_potential, ao));
    auto V = std::make_shared<Matrix>("Vx AO", g.nbf, g.nbf);
    if (g.nbf) {
        check_cuda(cudaMemcpy(V->pointer()[0], ao, g.nbf * g.nbf * sizeof(double), cudaMemcpyDeviceToHost),
                   "AO response download");
    }
    return V;
}

// => Functional derivatives <= //

void require_response_derivatives(const std::vector<std::shared_ptr<Functional>>& components) {
    for (const auto& component : components) {
        auto* libxc = dynamic_cast<LibXCFunctional*>(component.get());
        if (!libxc) {
            throw PSIEXCEPTION("cuEST XC response requires LibXC components; '" + component->name() + "' is not.");
        }
        if (component->is_meta()) {
            throw PSIEXCEPTION("Vx: cuEST response does not support rotated V builds for MGGA's (component '" +
                               component->name() + "').");
        }
        const auto flags = libxc->xc_functional()->info->flags;
        if (!(flags & XC_FLAGS_HAVE_FXC) || (component->is_gga() && !(flags & XC_FLAGS_HAVE_VXC))) {
            throw PSIEXCEPTION("cuEST XC response: LibXC component '" + component->name() +
                               "' does not provide the derivatives the response kernel needs.");
        }
    }
}

/// Fields of the reference derivative kernel, in host memory.
struct HostKernel {
    std::vector<double> A, B, C, S;                              // RKS
    std::vector<double> v2rho2, v2rhosigma, v2sigma2, vsigma;    // UKS (polarized LibXC layouts)
};

/// Second derivatives with the native SuperFunctional workers (deriv = 2) on host copies of
/// the cuEST reference density. `base` is the unpolarized functional for an RKS singlet, its
/// polarized clone for a triplet (evaluated at rho_a = rho_b and projected exactly as
/// SuperFunctional::compute_functional does), and the polarized functional itself for UKS. GRAC is never applied at deriv = 2, so a GRAC
/// functional contributes its uncorrected semilocal kernel, as natively.
void evaluate_host_kernel(const std::shared_ptr<SuperFunctional>& base, VxSpin spin, bool gga, size_t npoints,
                          size_t ncomponents, const std::vector<double>& ref_a, const std::vector<double>& ref_b,
                          int nthreads, HostKernel& out) {
    const bool polarized = spin != VxSpin::RKSSinglet;

    nthreads = std::max(1, nthreads);
    std::vector<std::shared_ptr<SuperFunctional>> workers;
    for (int t = 0; t < nthreads; ++t) {
        auto worker = base->build_worker();
        worker->set_deriv(2);
        worker->allocate();
        workers.push_back(worker);
    }
    const size_t block = static_cast<size_t>(workers.front()->max_points());
    if (block == 0) throw PSIEXCEPTION("cuEST XC response: functional workers have zero max_points.");

    if (spin == VxSpin::UKS) {
        out.v2rho2.assign(3 * npoints, 0.0);
        if (gga) {
            out.v2rhosigma.assign(6 * npoints, 0.0);
            out.v2sigma2.assign(6 * npoints, 0.0);
            out.vsigma.assign(3 * npoints, 0.0);
        }
    } else {
        out.A.assign(npoints, 0.0);
        if (gga) {
            out.B.assign(npoints, 0.0);
            out.C.assign(npoints, 0.0);
            out.S.assign(npoints, 0.0);
        }
    }

    std::exception_ptr failure;
#pragma omp parallel num_threads(nthreads)
    {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        std::map<std::string, SharedVector> inputs;
        for (const char* key : {"RHO_A", "RHO_B", "GAMMA_AA", "GAMMA_AB", "GAMMA_BB"}) {
            inputs[key] = std::make_shared<Vector>(key, block);
        }
        double* rho_a = inputs["RHO_A"]->pointer();
        double* rho_b = inputs["RHO_B"]->pointer();
        double* gamma_aa = inputs["GAMMA_AA"]->pointer();
        double* gamma_ab = inputs["GAMMA_AB"]->pointer();
        double* gamma_bb = inputs["GAMMA_BB"]->pointer();

#pragma omp for schedule(static)
        for (size_t start = 0; start < npoints; start += block) {
            if (failure) continue;
            try {
                const size_t count = std::min(block, npoints - start);
                for (size_t i = 0; i < count; ++i) {
                    const double* a = ref_a.data() + ncomponents * (start + i);
                    const double* b = (spin == VxSpin::UKS ? ref_b.data() : ref_a.data()) + ncomponents * (start + i);
                    if (polarized) {
                        rho_a[i] = a[0];
                        rho_b[i] = b[0];
                        if (gga) {
                            gamma_aa[i] = a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
                            gamma_ab[i] = a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
                            gamma_bb[i] = b[1] * b[1] + b[2] * b[2] + b[3] * b[3];
                        }
                    } else {
                        // RKSFunctions: spin-summed density and gradient.
                        rho_a[i] = 2.0 * a[0];
                        if (gga) gamma_aa[i] = 4.0 * (a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
                    }
                }
                const auto& values = workers[rank]->compute_functional(inputs, static_cast<int>(count));
                const auto get = [&](const char* key) { return values.at(key)->pointer(); };
                if (spin == VxSpin::RKSSinglet) {
                    std::copy_n(get("V_RHO_A_RHO_A"), count, out.A.data() + start);
                    if (gga) {
                        std::copy_n(get("V_RHO_A_GAMMA_AA"), count, out.B.data() + start);
                        std::copy_n(get("V_GAMMA_AA_GAMMA_AA"), count, out.C.data() + start);
                        std::copy_n(get("V_GAMMA_AA"), count, out.S.data() + start);
                    }
                } else {
                    const double* r2[3] = {get("V_RHO_A_RHO_A"), get("V_RHO_A_RHO_B"), get("V_RHO_B_RHO_B")};
                    const double *rs[6] = {}, *s2[6] = {}, *s1[3] = {};
                    if (gga) {
                        const char* rs_keys[6] = {"V_RHO_A_GAMMA_AA", "V_RHO_A_GAMMA_AB", "V_RHO_A_GAMMA_BB",
                                                  "V_RHO_B_GAMMA_AA", "V_RHO_B_GAMMA_AB", "V_RHO_B_GAMMA_BB"};
                        const char* s2_keys[6] = {"V_GAMMA_AA_GAMMA_AA", "V_GAMMA_AA_GAMMA_AB",
                                                  "V_GAMMA_AA_GAMMA_BB", "V_GAMMA_AB_GAMMA_AB",
                                                  "V_GAMMA_AB_GAMMA_BB", "V_GAMMA_BB_GAMMA_BB"};
                        const char* s1_keys[3] = {"V_GAMMA_AA", "V_GAMMA_AB", "V_GAMMA_BB"};
                        for (int k = 0; k < 6; ++k) rs[k] = get(rs_keys[k]);
                        for (int k = 0; k < 6; ++k) s2[k] = get(s2_keys[k]);
                        for (int k = 0; k < 3; ++k) s1[k] = get(s1_keys[k]);
                    }
                    for (size_t i = 0; i < count; ++i) {
                        double l_r2[3], l_rs[6] = {}, l_s2[6] = {}, l_s1[3] = {};
                        for (int k = 0; k < 3; ++k) l_r2[k] = r2[k][i];
                        if (gga) {
                            for (int k = 0; k < 6; ++k) l_rs[k] = rs[k][i];
                            for (int k = 0; k < 6; ++k) l_s2[k] = s2[k][i];
                            for (int k = 0; k < 3; ++k) l_s1[k] = s1[k][i];
                        }
                        const size_t p = start + i;
                        if (spin == VxSpin::RKSTriplet) {
                            const auto k = cuest_xc_response::project_triplet(l_r2, l_rs, l_s2, l_s1, gga);
                            out.A[p] = k.A;
                            if (gga) {
                                out.B[p] = k.B;
                                out.C[p] = k.C;
                                out.S[p] = k.S;
                            }
                        } else {
                            std::copy_n(l_r2, 3, out.v2rho2.data() + 3 * p);
                            if (gga) {
                                std::copy_n(l_rs, 6, out.v2rhosigma.data() + 6 * p);
                                std::copy_n(l_s2, 6, out.v2sigma2.data() + 6 * p);
                                std::copy_n(l_s1, 3, out.vsigma.data() + 3 * p);
                            }
                        }
                    }
                }
            } catch (...) {
#pragma omp critical(cuest_xc_response_failure)
                if (!failure) failure = std::current_exception();
            }
        }
    }
    if (failure) std::rethrow_exception(failure);

    const auto require_finite = [](const std::vector<double>& field, const char* name) {
        for (double value : field) {
            if (!std::isfinite(value)) {
                throw PSIEXCEPTION(std::string("Vx: the XC functional derivative ") + name +
                                   " is not finite on the cuEST grid. The functional is not numerically stable.");
            }
        }
    };
    for (const auto& [field, name] : {std::pair<const std::vector<double>*, const char*>{&out.A, "f_rho,rho"},
                                      {&out.B, "f_rho,gamma"}, {&out.C, "f_gamma,gamma"}, {&out.S, "f_gamma"},
                                      {&out.v2rho2, "v2rho2"}, {&out.v2rhosigma, "v2rhosigma"},
                                      {&out.v2sigma2, "v2sigma2"}, {&out.vsigma, "vsigma"}}) {
        require_finite(*field, name);
    }
}

#ifdef USING_Libxc_CUDA
/// Device counterpart of evaluate_host_kernel, with CUDA LibXC on cuEST's device density.
/// Components are evaluated one at a time and accumulated with their mixing coefficients;
/// triplet components are projected immediately, since the projection is linear.
void evaluate_device_kernel(const std::shared_ptr<SuperFunctional>& base, VxSpin spin, bool gga, GridContext& g,
                            const double* ref_a, const double* ref_b) {
    auto& ws = g.ws;
    const size_t n = g.npoints;
    const bool polarized = spin != VxSpin::RKSSinglet;
    const size_t nrho = polarized ? 2 : 1, nsigma = polarized ? 3 : 1;
    const size_t n2r = polarized ? 3 : 1, nrs = polarized ? 6 : 1, n2s = polarized ? 6 : 1;

    double* rho = ws.xc_rho.ensure(nrho * n, "LibXC density");
    double* sigma = gga ? ws.xc_sigma.ensure(nsigma * n, "LibXC sigma") : nullptr;
    if (polarized) {
        check_cuda(cuest_xc_prepare_inputs_polarized(n, g.ncomponents, ref_a, spin == VxSpin::UKS ? ref_b : ref_a,
                                                     rho, sigma, nullptr),
                   "LibXC input preparation");
    } else {
        check_cuda(cuest_xc_prepare_inputs(n, g.ncomponents, ref_a, rho, sigma, nullptr), "LibXC input preparation");
    }

    double* t_vrho = gga ? ws.t_vrho.ensure(nrho * n, "LibXC vrho") : nullptr;
    double* t_vsigma = gga ? ws.t_vsigma.ensure(nsigma * n, "LibXC vsigma") : nullptr;
    double* t_v2rho2 = ws.t_v2rho2.ensure(n2r * n, "LibXC v2rho2");
    double* t_v2rhosigma = gga ? ws.t_v2rhosigma.ensure(nrs * n, "LibXC v2rhosigma") : nullptr;
    double* t_v2sigma2 = gga ? ws.t_v2sigma2.ensure(n2s * n, "LibXC v2sigma2") : nullptr;

    // Accumulated fields: RKS (A, B, C, S) or UKS polarized blocks.
    std::vector<std::pair<double*, size_t>> totals;
    double *A = nullptr, *B = nullptr, *C = nullptr, *S = nullptr;
    double *v2rho2 = nullptr, *v2rhosigma = nullptr, *v2sigma2 = nullptr, *vsigma = nullptr;
    if (spin == VxSpin::UKS) {
        v2rho2 = ws.v2rho2.ensure(3 * n, "v2rho2");
        totals.push_back({v2rho2, 3 * n});
        if (gga) {
            v2rhosigma = ws.v2rhosigma.ensure(6 * n, "v2rhosigma");
            v2sigma2 = ws.v2sigma2.ensure(6 * n, "v2sigma2");
            vsigma = ws.vsigma.ensure(3 * n, "vsigma");
            totals.push_back({v2rhosigma, 6 * n});
            totals.push_back({v2sigma2, 6 * n});
            totals.push_back({vsigma, 3 * n});
        }
    } else {
        A = ws.kernel_A.ensure(n, "f_rho,rho");
        totals.push_back({A, n});
        if (gga) {
            B = ws.kernel_B.ensure(n, "f_rho,gamma");
            C = ws.kernel_C.ensure(n, "f_gamma,gamma");
            S = ws.kernel_S.ensure(n, "f_gamma");
            totals.push_back({B, n});
            totals.push_back({C, n});
            totals.push_back({S, n});
        }
    }
    for (const auto& [pointer, count] : totals) {
        check_cuda(cudaMemset(pointer, 0, count * sizeof(double)), "derivative initialization");
    }

    for (const auto& components : {base->x_functionals(), base->c_functionals()}) {
        for (const auto& component : components) {
            auto* libxc = static_cast<LibXCFunctional*>(component.get());
            const auto* device = libxc->xc_functional_device();
            const double alpha = component->alpha();
            const bool component_gga = component->is_gga();
            if (component_gga) {
                xc_gga_vxc_fxc(device, n, rho, sigma, t_vrho, t_vsigma, t_v2rho2, t_v2rhosigma, t_v2sigma2);
            } else {
                xc_lda_fxc(device, n, rho, t_v2rho2);
            }
            check_cuda(cudaGetLastError(), "CUDA LibXC derivative evaluation");
            if (spin == VxSpin::RKSTriplet) {
                double* tA = ws.t_A.ensure(n, "triplet projection");
                double* tB = gga ? ws.t_B.ensure(n, "triplet projection") : nullptr;
                double* tC = gga ? ws.t_C.ensure(n, "triplet projection") : nullptr;
                double* tS = gga ? ws.t_S.ensure(n, "triplet projection") : nullptr;
                check_cuda(cuest_xc_response_project_triplet(n, component_gga, t_v2rho2, t_v2rhosigma, t_v2sigma2,
                                                             t_vsigma, tA, tB, tC, tS),
                           "triplet projection");
                check_cuda(cuest_xc_response_axpy(n, alpha, tA, A), "derivative accumulation");
                if (component_gga) {
                    check_cuda(cuest_xc_response_axpy(n, alpha, tB, B), "derivative accumulation");
                    check_cuda(cuest_xc_response_axpy(n, alpha, tC, C), "derivative accumulation");
                    check_cuda(cuest_xc_response_axpy(n, alpha, tS, S), "derivative accumulation");
                }
            } else if (spin == VxSpin::UKS) {
                check_cuda(cuest_xc_response_axpy(3 * n, alpha, t_v2rho2, v2rho2), "derivative accumulation");
                if (component_gga) {
                    check_cuda(cuest_xc_response_axpy(6 * n, alpha, t_v2rhosigma, v2rhosigma),
                               "derivative accumulation");
                    check_cuda(cuest_xc_response_axpy(6 * n, alpha, t_v2sigma2, v2sigma2), "derivative accumulation");
                    check_cuda(cuest_xc_response_axpy(3 * n, alpha, t_vsigma, vsigma), "derivative accumulation");
                }
            } else {
                check_cuda(cuest_xc_response_axpy(n, alpha, t_v2rho2, A), "derivative accumulation");
                if (component_gga) {
                    check_cuda(cuest_xc_response_axpy(n, alpha, t_v2rhosigma, B), "derivative accumulation");
                    check_cuda(cuest_xc_response_axpy(n, alpha, t_v2sigma2, C), "derivative accumulation");
                    check_cuda(cuest_xc_response_axpy(n, alpha, t_vsigma, S), "derivative accumulation");
                }
            }
        }
    }

    auto* counter = static_cast<unsigned long long*>(ws.counter.ensure_bytes(sizeof(unsigned long long), "counter"));
    check_cuda(cudaMemset(counter, 0, sizeof(unsigned long long)), "finite check reset");
    for (const auto& [pointer, count] : totals) {
        check_cuda(cuest_xc_response_count_nonfinite(count, pointer, counter), "finite check");
    }
    unsigned long long nonfinite = 0;
    check_cuda(cudaMemcpy(&nonfinite, counter, sizeof(nonfinite), cudaMemcpyDeviceToHost), "finite check download");
    if (nonfinite) {
        throw PSIEXCEPTION("Vx: " + std::to_string(nonfinite) +
                           " XC functional derivative values are not finite on the cuEST grid. "
                           "The functional is not numerically stable.");
    }
}
#endif

/// What the derivative kernel depends on in the functional: its semilocal components.
/// (GRAC is never applied at deriv = 2; exact exchange does not enter the XC kernel.)
std::string kernel_fingerprint(SuperFunctional& functional) {
    std::ostringstream key;
    key.precision(17);
    key << static_cast<const void*>(&functional);
    for (const auto& components : {functional.x_functionals(), functional.c_functionals()}) {
        for (const auto& component : components) {
            key << '|' << component->name() << ',' << component->alpha() << ',' << component->omega();
        }
    }
    return key.str();
}

#endif  // PSI4_CUEST_NONSYMMETRIC_XC

}  // namespace

void VBase::compute_Vx_cuest(const std::vector<VxDensity>& inputs, std::vector<SharedMatrix>& ret, VxSpin spin) {
#ifndef PSI4_CUEST_NONSYMMETRIC_XC
    throw_if_cuest_unsupported(
        "The XC response kernel (this cuEST SDK has no cuestXCNonsymmetricDensityCompute)");
    throw PSIEXCEPTION("unreachable");
#else
    const bool uks = spin == VxSpin::UKS;
    xc_timing::CoarseScope form_timer(uks ? "UV: Form Vx (cuEST)" : "RV: Form Vx (cuEST)");
    ResponseStages stages;
    stages.start("cuEST Vx: setup");

    // => Validate, with the native messages where a native check exists <= //
    if (D_AO_.size() != (uks ? 2u : 1u)) {
        throw PSIEXCEPTION(uks ? "Vx: UKS should have two D matrices." : "Vx: RKS should have only one D Matrix");
    }
    if (inputs.size() != ret.size() || inputs.empty()) {
        throw PSIEXCEPTION(uks ? "Vx: UKS input and output sizes should be the same."
                               : "Vx: RKS input and output sizes should be the same.");
    }
    if (uks && inputs.size() % 2) throw PSIEXCEPTION("Vx: UKS input should occur in alpha and beta pairs.");
    if (functional_->needs_vv10()) {
        throw PSIEXCEPTION(uks ? "V: UKS cannot compute VV10 Vx contribution."
                               : "Vx: RKS cannot compute VV10 Vx contribution.");
    }
    const int ansatz = functional_->ansatz();
    if (ansatz >= 2) {
        throw PSIEXCEPTION(uks ? "Vx: UKS does not support rotated V builds for MGGA's"
                               : "Vx: RKS does not support rotated V builds for MGGA's");
    }
    if (functional_->needs_grac() && spin != VxSpin::RKSSinglet) {
        // Native: the polarized clone a triplet or UKS kernel needs cannot carry GRAC.
        throw PSIEXCEPTION("GRAC is not implemented for UKS functionals (" +
                           std::string(uks ? "UKS" : "RKS triplet") + " XC response).");
    }
    if (initial_grid_) {
        throw PSIEXCEPTION("Vx: the XC response kernel must be built on the final DFT grid, not the initial SCF grid.");
    }
    if (!grid_ || !cuest_xcint_plan_) {
        throw PSIEXCEPTION(
            "Vx: this potential was not initialized on the cuEST grid (USE_CUEST/CUEST_XC changed after "
            "initialize()). Rebuild the potential, or set CUEST_XC false.");
    }
    const auto polarized_base = spin == VxSpin::RKSTriplet ? functional_->build_polarized() : functional_;
    require_response_derivatives(polarized_base->x_functionals());
    require_response_derivatives(polarized_base->c_functionals());

    const size_t nbf = static_cast<size_t>(nbf_);
    for (size_t i = 0; i < ret.size(); ++i) {
        const auto& out = ret[i];
        if (!out) throw PSIEXCEPTION("Vx: output matrix is missing.");
        if (out->nirrep() != 1) {
            if (!AO2USO_) throw PSIEXCEPTION("Vx: symmetry-blocked output requires a symmetry-blocked reference density.");
        } else if (static_cast<size_t>(out->rowdim()) != nbf || static_cast<size_t>(out->coldim()) != nbf) {
            throw PSIEXCEPTION("Vx: C1 output matrices must be nbf x nbf.");
        }
    }

    cuest_common::ensure_cuest_initialized();
    const bool gga = ansatz == 1;
    const size_t ncomponents = gga ? 4 : 1;
    uint64_t npoints = 0;
    RESPONSE_CHECK_CUEST(cuestQuery(cuest_handle, CUEST_MOLECULARGRID, grid_->cuest_grid(),
                                    CUEST_MOLECULARGRID_NUM_POINT, &npoints, sizeof(uint64_t)));
    const int budget_mb = options_.get_int("CUEST_XC_RESPONSE_BUDGET_MB");
    if (budget_mb <= 0) throw PSIEXCEPTION("CUEST_XC_RESPONSE_BUDGET_MB must be positive.");
    if (!cuest_response_) cuest_response_ = std::make_unique<CuestXCResponseWorkspace>();
    auto& ws = *cuest_response_;
    GridContext g{cuest_xcint_plan_,
                  gga ? CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_GGA : CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_LDA,
                  npoints,
                  ncomponents,
                  nbf,
                  ws,
                  cuestWorkspaceDescriptor_t{0, static_cast<size_t>(budget_mb) * 1000000},
                  print_ > 2 || debug_ > 0};
    const size_t field = npoints * ncomponents;

#ifdef USING_Libxc_CUDA
    const bool device_functional = !options_.get_bool("CUEST_XC_RESPONSE_HOST_LIBXC");
#else
    const bool device_functional = false;
#endif
    const double cutoff = v2_rho_cutoff_;

    const size_t nspin = uks ? 2 : 1;
    double* d_weights = ws.weights.ensure(npoints, "integration weights");
    // The host route keeps the reference on the host; the device copies are only scratch then.
    double* d_ref[2] = {ws.reference_a.ensure(field, "reference density"),
                        uks ? ws.reference_b.ensure(field, "reference density") : nullptr};
    std::vector<double> h_weights;
    std::vector<double> h_ref[2];
    HostKernel host_kernel;

    // => Reuse the reference state of the previous call, when nothing it depends on changed <= //
    const std::string fingerprint = kernel_fingerprint(*functional_);
    const bool reuse = device_functional && ws.reference_valid && ws.reference_epoch == cuest_reference_epoch_ &&
                       ws.reference_spin == static_cast<int>(spin) && ws.reference_npoints == npoints &&
                       ws.reference_functional == fingerprint;
    if (reuse) {
        if (g.verbose) {
            outfile->Printf("    cuEST Vx: %s, %zu points, reference and kernel reused, CUDA LibXC\n",
                            uks ? "UKS" : (spin == VxSpin::RKSSinglet ? "RKS singlet" : "RKS triplet"),
                            static_cast<size_t>(npoints));
        }
    } else {
        // Invalid until the derivatives below complete, so a failure can never leave a partial state.
        ws.reference_valid = false;

        // => Integration weights (applied exactly once, in the packed potential) <= //
        stages.start("cuEST Vx: weights");
        {
            CuestParameters<cuestXCIntegrationWeightComputeParameters_t> parameters(
                CUEST_XCINTEGRATIONWEIGHTCOMPUTE_PARAMETERS);
            cuestWorkspaceDescriptor_t temporary{};
            RESPONSE_CHECK_CUEST(cuestXCIntegrationWeightComputeWorkspaceQuery(
                cuest_handle, g.plan, CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_TOTAL, parameters.get(), &temporary,
                d_weights));
            RESPONSE_CHECK_CUEST(cuestXCIntegrationWeightCompute(cuest_handle, g.plan,
                                                                 CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_TOTAL,
                                                                 parameters.get(), ws.sdk_workspace(temporary), d_weights));
        }
        if (!device_functional) {
            h_weights.resize(npoints);
            download(d_weights, npoints, h_weights.data());
        }

        // => Reference density: verified occupied factors, else the dense D_AO_ <= //
        stages.start("cuEST Vx: reference density");
        const bool occupied = cuest_Cocc_reproduces_D();
        uint64_t offset = 0;
        for (size_t s = 0; s < nspin; ++s) {
            if (occupied) {
                occupied_density(g, d_Cocc_noccs_[s], d_Coccs_AO_ + offset, d_ref[s]);
                offset += d_Cocc_noccs_[s] * nbf;
                if (!device_functional) {
                    h_ref[s].resize(field);
                    download(d_ref[s], field, h_ref[s].data());
                }
            } else {
                DensityTarget target;
                if (device_functional) {
                    target.device = d_ref[s];
                } else {
                    target.host = &h_ref[s];
                }
                dense_density(g, *D_AO_[s], target);
            }
        }
        if (g.verbose) {
            outfile->Printf("    cuEST Vx: %s, %zu points, reference from %s, %s LibXC\n",
                            uks ? "UKS" : (spin == VxSpin::RKSSinglet ? "RKS singlet" : "RKS triplet"),
                            static_cast<size_t>(npoints), occupied ? "occupied factors" : "dense D",
                            device_functional ? "CUDA" : "host");
        }

        // => Reference first/second derivatives <= //
        stages.start("cuEST Vx: derivatives");
        if (device_functional) {
#ifdef USING_Libxc_CUDA
            evaluate_device_kernel(polarized_base, spin, gga, g, d_ref[0], d_ref[1]);
#endif
            ws.reference_epoch = cuest_reference_epoch_;
            ws.reference_spin = static_cast<int>(spin);
            ws.reference_npoints = npoints;
            ws.reference_functional = fingerprint;
            ws.reference_valid = true;
        } else {
            evaluate_host_kernel(polarized_base, spin, gga, npoints, ncomponents, h_ref[0], uks ? h_ref[1] : h_ref[0],
                                 num_threads_, host_kernel);
        }
    }

    // => One RHS (one alpha/beta pair) at a time <= //
    const size_t nrhs = uks ? inputs.size() / 2 : inputs.size();
    std::vector<double> h_resp[2], h_pot[2];
    double* d_resp[2] = {nullptr, nullptr};
    double* d_pot[2] = {ws.potential_a.ensure(field, "grid potential"),
                        uks ? ws.potential_b.ensure(field, "grid potential") : nullptr};
    for (size_t r = 0; r < nrhs; ++r) {
        stages.start("cuEST Vx: response density");
        for (size_t s = 0; s < nspin; ++s) {
            const auto& in = inputs[uks ? 2 * r + s : r];
            DensityTarget target;
            if (device_functional) {
                d_resp[s] = (s == 0 ? ws.response_a : ws.response_b).ensure(field, "response density");
                target.device = d_resp[s];
            } else {
                target.host = &h_resp[s];
            }
            if (in.left && in.right && in.left->nirrep() == 1 && in.right->nirrep() == 1) {
                factored_density(g, *in.left, *in.right, target);
            } else {
                // Symmetry-blocked factors and dense inputs reach the same dense adapter.
                SharedMatrix D = in.dense;
                if (in.left && in.right) D = linalg::doublet(in.left, in.right, false, true);
                if (!D) throw PSIEXCEPTION("Vx: response density is missing.");
                if (D->nirrep() != 1) {
                    if (!USO2AO_) {
                        throw PSIEXCEPTION("Vx: symmetry-blocked response requires a symmetry-blocked reference.");
                    }
                    auto D_AO = std::make_shared<Matrix>("Dx AO", nbf_, nbf_);
                    D_AO->remove_symmetry(D, USO2AO_);
                    D = D_AO;
                }
                dense_density(g, *D, target);
            }
        }

        stages.start("cuEST Vx: contraction");
        if (device_functional) {
#ifdef USING_Libxc_CUDA
            if (uks) {
                check_cuda(cuest_xc_response_pack_uks(npoints, gga, cutoff, d_weights, d_ref[0], d_ref[1],
                                                      ws.v2rho2.ensure(3 * npoints, "v2rho2"),
                                                      gga ? ws.v2rhosigma.ensure(6 * npoints, "v2rhosigma") : nullptr,
                                                      gga ? ws.v2sigma2.ensure(6 * npoints, "v2sigma2") : nullptr,
                                                      gga ? ws.vsigma.ensure(3 * npoints, "vsigma") : nullptr,
                                                      d_resp[0], d_resp[1], d_pot[0], d_pot[1]),
                           "UKS response contraction");
            } else {
                check_cuda(cuest_xc_response_pack_rks(npoints, gga, cutoff, d_weights, d_ref[0],
                                                      ws.kernel_A.ensure(npoints, "A"),
                                                      gga ? ws.kernel_B.ensure(npoints, "B") : nullptr,
                                                      gga ? ws.kernel_C.ensure(npoints, "C") : nullptr,
                                                      gga ? ws.kernel_S.ensure(npoints, "S") : nullptr, d_resp[0],
                                                      d_pot[0]),
                           "RKS response contraction");
            }
#endif
        } else {
            for (size_t s = 0; s < nspin; ++s) h_pot[s].resize(field);
            const auto& K = host_kernel;
#pragma omp parallel for schedule(static) num_threads(num_threads_)
            for (size_t p = 0; p < npoints; ++p) {
                const size_t o = ncomponents * p;
                if (uks) {
                    cuest_xc_response::pack_uks(gga, h_weights[p], cutoff, h_ref[0].data() + o, h_ref[1].data() + o,
                                                K.v2rho2.data() + 3 * p, gga ? K.v2rhosigma.data() + 6 * p : nullptr,
                                                gga ? K.v2sigma2.data() + 6 * p : nullptr,
                                                gga ? K.vsigma.data() + 3 * p : nullptr, h_resp[0].data() + o,
                                                h_resp[1].data() + o, h_pot[0].data() + o, h_pot[1].data() + o);
                } else {
                    cuest_xc_response::RKSKernel k{K.A[p], 0.0, 0.0, 0.0};
                    if (gga) k = {K.A[p], K.B[p], K.C[p], K.S[p]};
                    cuest_xc_response::pack_rks(gga, h_weights[p], cutoff, h_ref[0].data() + o, k,
                                                h_resp[0].data() + o, h_pot[0].data() + o);
                }
            }
            for (size_t s = 0; s < nspin; ++s) {
                check_cuda(cudaMemcpy(d_pot[s], h_pot[s].data(), field * sizeof(double), cudaMemcpyHostToDevice),
                           "grid potential upload");
            }
        }

        for (size_t s = 0; s < nspin; ++s) {
            stages.start("cuEST Vx: AO integration");
            auto V_AO = ao_potential(g, d_pot[s]);
            stages.start("cuEST Vx: output symmetry");
            auto& out = ret[uks ? 2 * r + s : r];
            if (out->nirrep() != 1) {
                out->apply_symmetry(V_AO, AO2USO_);
            } else {
                out->copy(V_AO);
            }
        }
    }
    stages.finish();
#endif
}

}  // namespace psi

#endif  // USING_cuEST
