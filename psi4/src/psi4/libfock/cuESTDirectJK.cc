/*
 * Psi4: an open-source quantum chemistry software package
 * Copyright (c) 2007-2026 The Psi4 Developers.
 * Licensed under the GNU Lesser General Public License, version 3.
 * This software is distributed WITHOUT ANY WARRANTY; see the repository COPYING.
 *
 * This adapter calls the separately licensed NVIDIA cuEST library. No SDK
 * implementation is included here. The SDK requires the following notice:
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THESE LICENSED
 * DELIVERABLES FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE
 * LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. NOTWITHSTANDING ANY
 * TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE AGREEMENT, IN NO EVENT SHALL
 * NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
 * DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
 * OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THESE LICENSED
 * DELIVERABLES.
 *
 * U.S. Government End Users. These Licensed Deliverables are a "commercial item"
 * as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting of "commercial
 * computer software" and "commercial computer software documentation" as such
 * terms are used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S.
 * Government only as a commercial end item. Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all U.S. Government End
 * Users acquire the Licensed Deliverables with only those rights set forth
 * herein.
 */
#include "cuESTDirectJK.h"

#if defined(USING_cuEST) && defined(USING_cuEST_DIRECT_JK)
#include "cuESTCommon.h"
#include "psi4/libmints/matrix.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace psi {
namespace cuest_direct_detail {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw PSIEXCEPTION(std::string("cuESTDirectJK: ") + operation + ": " + cudaGetErrorString(status));
    }
}

void cleanup_status(cuestStatus_t status, const char* operation) noexcept {
    if (status != CUEST_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuESTDirectJK cleanup: %s failed (status %d)\n", operation, static_cast<int>(status));
    }
}

struct WorkspaceDeleter {
    void operator()(cuestWorkspace_t* value) const noexcept {
        if (!value) return;
        if (value->deviceBuffer) {
            auto status = cudaFree(reinterpret_cast<void*>(value->deviceBuffer));
            if (status != cudaSuccess)
                std::fprintf(stderr, "cuESTDirectJK cleanup: workspace cudaFree: %s\n", cudaGetErrorString(status));
        }
        std::free(reinterpret_cast<void*>(value->hostBuffer));
        delete value;
    }
};
using Workspace = std::unique_ptr<cuestWorkspace_t, WorkspaceDeleter>;

Workspace workspace(const cuestWorkspaceDescriptor_t& descriptor) {
    Workspace value(new cuestWorkspace_t{});
    value->hostBufferSizeInBytes = descriptor.hostBufferSizeInBytes;
    value->deviceBufferSizeInBytes = descriptor.deviceBufferSizeInBytes;
    if (descriptor.hostBufferSizeInBytes) {
        auto host = std::malloc(descriptor.hostBufferSizeInBytes);
        if (!host) throw PSIEXCEPTION("cuESTDirectJK: cannot allocate host workspace");
        value->hostBuffer = reinterpret_cast<uintptr_t>(host);
    }
    if (descriptor.deviceBufferSizeInBytes) {
        void* device = nullptr;
        check_cuda(cudaMalloc(&device, descriptor.deviceBufferSizeInBytes), "allocate device workspace");
        value->deviceBuffer = reinterpret_cast<uintptr_t>(device);
    }
    return value;
}

struct Parameters {
    cuestParametersType_t type;
    void* value = nullptr;
    explicit Parameters(cuestParametersType_t type_) : type(type_) {
        CHECK_CUEST(cuestParametersCreate(type, &value));
    }
    ~Parameters() { if (value) cleanup_status(cuestParametersDestroy(type, value), "parameters destroy"); }
    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete;
    template <class T> void configure(uint64_t attribute, const T& setting) {
        CHECK_CUEST(cuestParametersConfigure(type, value, attribute, &setting, sizeof(setting)));
    }
};

class DeviceMatrix {
    double* pointer_ = nullptr;
   public:
    explicit DeviceMatrix(size_t bytes) {
        if (bytes) check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes), "allocate matrix");
    }
    ~DeviceMatrix() {
        if (pointer_) {
            auto status = cudaFree(pointer_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "cuESTDirectJK cleanup: cudaFree: %s\n", cudaGetErrorString(status));
            }
        }
    }
    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;
    double* get() const { return pointer_; }
};

}  // namespace cuest_direct_detail

struct cuESTDirectJK::State {
    cuestAOPairList_t pairs = nullptr;
    cuestDirectIntPlan_t plan = nullptr;
    cuest_direct_detail::Workspace pair_workspace;
    cuest_direct_detail::Workspace plan_workspace;
    cuest_direct_detail::Parameters j_parameters{CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS};
    cuest_direct_detail::Parameters k_parameters{CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS};
    double alpha = 0., beta = 0., omega = 0., cutoff = 0.;
    bool scratch_reported = false;
    ~State() {
        // The plan views the pair list and its persistent storage.
        if (plan) cuest_direct_detail::cleanup_status(cuestDirectIntPlanDestroy(plan), "direct plan destroy");
        if (pairs) cuest_direct_detail::cleanup_status(cuestAOPairListDestroy(pairs), "pair list destroy");
    }
};

cuESTDirectJK::cuESTDirectJK(std::shared_ptr<BasisSet> primary, Options& options)
    : JK(primary), options_(options) {
    const auto screening = options.get_str("SCREENING");
    if (screening != "SCHWARZ" && screening != "DENSITY" && screening != "NONE") {
        throw PSIEXCEPTION("cuESTDirectJK supports SCREENING=SCHWARZ, DENSITY or NONE, not " + screening);
    }
    cuest_common::ensure_cuest_initialized();
}

cuESTDirectJK::~cuESTDirectJK() = default;

void cuESTDirectJK::preiterations() {
    using namespace cuest_direct_detail;
    if (!std::isfinite(omega_alpha_) || !std::isfinite(omega_beta_) || !std::isfinite(omega_) ||
        omega_alpha_ < 0. || omega_alpha_ + omega_beta_ < 0. || omega_ < 0.) {
        throw PSIEXCEPTION("cuESTDirectJK: invalid exchange fractions or range-separation omega");
    }
    if (!std::isfinite(cutoff_) || cutoff_ < 0.) {
        throw PSIEXCEPTION("cuESTDirectJK: integral threshold must be finite and nonnegative");
    }
    CHECK_CUEST(cuestSetMathMode(cuest_handle, options_.get_bool("CUEST_MIXED_PRECISION")
        ? CUEST_DEFAULT_MATH_MODE : CUEST_NATIVE_FP64_MATH_MODE));
    auto next = std::make_unique<State>();
    next->alpha = omega_alpha_;
    next->beta = omega_beta_;
    next->omega = omega_;
    next->cutoff = cutoff_;
    auto basis = primary_->cuest_basis();
    auto mol = primary_->molecule();
    auto geometry = mol->geometry();
    cuestWorkspaceDescriptor_t persistent{}, temporary{};
    Parameters pair_parameters(CUEST_AOPAIRLIST_PARAMETERS);
    auto pair_params = reinterpret_cast<cuestAOPairListParameters_t>(pair_parameters.value);
    CHECK_CUEST(cuestAOPairListCreateWorkspaceQuery(cuest_handle, basis,
        static_cast<uint64_t>(mol->natom()), geometry.pointer()[0], cutoff_, pair_params,
        &persistent, &temporary, nullptr));
    next->pair_workspace = workspace(persistent);
    {
        auto scratch = workspace(temporary);
        CHECK_CUEST(cuestAOPairListCreate(cuest_handle, basis,
            static_cast<uint64_t>(mol->natom()), geometry.pointer()[0], cutoff_, pair_params,
            next->pair_workspace.get(), scratch.get(), &next->pairs));
    }
    Parameters plan_parameters(CUEST_DIRECTINTPLAN_PARAMETERS);
    plan_parameters.configure(CUEST_DIRECTINTPLAN_PARAMETERS_EXCHANGE_FRACTION, omega_alpha_);
    plan_parameters.configure(CUEST_DIRECTINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION, omega_beta_);
    plan_parameters.configure(CUEST_DIRECTINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA, omega_);
    auto plan_params = reinterpret_cast<cuestDirectIntPlanParameters_t>(plan_parameters.value);
    CHECK_CUEST(cuestDirectIntPlanCreateWorkspaceQuery(cuest_handle, basis, next->pairs,
        plan_params, &persistent, &temporary, nullptr));
    next->plan_workspace = workspace(persistent);
    {
        auto scratch = workspace(temporary);
        CHECK_CUEST(cuestDirectIntPlanCreate(cuest_handle, basis, next->pairs, plan_params,
            next->plan_workspace.get(), scratch.get(), &next->plan));
    }
    next->j_parameters.configure(CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_SCHWARZ_THRESHOLD, cutoff_);
    next->k_parameters.configure(CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS_SCHWARZ_THRESHOLD, cutoff_);
    auto jit = options_.get_str("CUEST_DIRECT_JIT");
    if (jit != "AUTO") {
        auto mode = jit == "ON" ? CUEST_JIT_USAGE_MODE_ON : CUEST_JIT_USAGE_MODE_OFF;
        next->j_parameters.configure(CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_JIT_USAGE_MODE, mode);
        next->k_parameters.configure(CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS_JIT_USAGE_MODE, mode);
    }
    check_cuda(cudaDeviceSynchronize(), "initialize direct plan");
    state_ = std::move(next);
    if (print_) {
        outfile->Printf("    Device memory, AO pair list: %8.3f [GiB]\n",
            state_->pair_workspace->deviceBufferSizeInBytes * cuest_common::to_gib);
        outfile->Printf("    Device memory, direct plan:  %8.3f [GiB]\n\n",
            state_->plan_workspace->deviceBufferSizeInBytes * cuest_common::to_gib);
    }
}

void cuESTDirectJK::compute_JK() {
    using namespace cuest_direct_detail;
    // The direct SDK has a symmetric-density K interface only. Never substitute
    // K(C_left,C_left) for a requested nonsymmetric exchange contraction.
    if (do_K_ && !lr_symmetric_) {
        throw PSIEXCEPTION("cuESTDirectJK: nonsymmetric exchange is not supported by the linked direct interface");
    }
    if (D_ao_.empty() || (!do_J_ && !do_K_)) return;
    // set_omega is not virtual in JK. Detect all operator/threshold changes here,
    // including modifications after initialize(), before consuming the plan.
    if (!state_ || state_->alpha != omega_alpha_ || state_->beta != omega_beta_ ||
        state_->omega != omega_ || state_->cutoff != cutoff_) {
        preiterations();
    }
    // The handle is shared with other cuEST builders. Do not inherit their mode.
    CHECK_CUEST(cuestSetMathMode(cuest_handle, options_.get_bool("CUEST_MIXED_PRECISION")
        ? CUEST_DEFAULT_MATH_MODE : CUEST_NATIVE_FP64_MATH_MODE));
    const auto nbf = static_cast<size_t>(primary_->nbf());
    const auto bytes = nbf * nbf * sizeof(double);
    DeviceMatrix density(bytes), coulomb(do_J_ ? bytes : 0), exchange(do_K_ ? bytes : 0);
    cuestWorkspaceDescriptor_t j_scratch{}, k_scratch{}, maximum{};
    auto j_params = reinterpret_cast<cuestDirectCoulombComputeParameters_t>(state_->j_parameters.value);
    auto k_params = reinterpret_cast<cuestDirectSymmetricExchangeComputeParameters_t>(state_->k_parameters.value);
    if (do_J_) CHECK_CUEST(cuestDirectCoulombComputeWorkspaceQuery(cuest_handle, state_->plan,
        j_params, &j_scratch, 1, density.get(), coulomb.get()));
    if (do_K_) CHECK_CUEST(cuestDirectSymmetricExchangeComputeWorkspaceQuery(cuest_handle, state_->plan,
        k_params, &k_scratch, 1, density.get(), exchange.get()));
    maximum.hostBufferSizeInBytes = std::max(j_scratch.hostBufferSizeInBytes, k_scratch.hostBufferSizeInBytes);
    maximum.deviceBufferSizeInBytes = std::max(j_scratch.deviceBufferSizeInBytes, k_scratch.deviceBufferSizeInBytes);
    auto scratch = workspace(maximum);
    if (print_ && !state_->scratch_reported) {
        outfile->Printf("    Device memory, direct J/K scratch: %8.3f [GiB]\n\n",
            maximum.deviceBufferSizeInBytes * cuest_common::to_gib);
        state_->scratch_reported = true;
    }
    timer_on("cuEST Direct J/K");
    try {
        for (size_t n = 0; n < D_ao_.size(); ++n) {
            const double* host_density = D_ao_[n]->get_pointer();
            std::vector<double> symmetric;
            if (!lr_symmetric_) {
                // (pq|rs) is symmetric in r,s: antisymmetric D contributes zero
                // to J. Make that exact rather than assuming an SDK convention.
                symmetric.resize(nbf * nbf);
                for (size_t i = 0; i < nbf; ++i) {
                    for (size_t j = 0; j < nbf; ++j) {
                        symmetric[i * nbf + j] = .5 * (host_density[i * nbf + j] + host_density[j * nbf + i]);
                    }
                }
                host_density = symmetric.data();
            }
            check_cuda(cudaMemcpy(density.get(), host_density, bytes, cudaMemcpyHostToDevice), "upload density");
            if (do_J_) {
                CHECK_CUEST(cuestDirectCoulombCompute(cuest_handle, state_->plan, j_params, scratch.get(),
                    1, density.get(), coulomb.get()));
                check_cuda(cudaDeviceSynchronize(), "compute direct J");
                check_cuda(cudaMemcpy(J_ao_[n]->get_pointer(), coulomb.get(), bytes, cudaMemcpyDeviceToHost),
                    "download direct J");
            }
            if (do_K_) {
                CHECK_CUEST(cuestDirectSymmetricExchangeCompute(cuest_handle, state_->plan, k_params, scratch.get(),
                    1, density.get(), exchange.get()));
                check_cuda(cudaDeviceSynchronize(), "compute direct K");
                check_cuda(cudaMemcpy(K_ao_[n]->get_pointer(), exchange.get(), bytes, cudaMemcpyDeviceToHost),
                    "download direct K");
            }
            if (do_wK_) wK_ao_[n]->zero();  // Weighted range-separated exchange is already in K.
        }
    } catch (...) {
        timer_off("cuEST Direct J/K");
        throw;
    }
    timer_off("cuEST Direct J/K");
}

void cuESTDirectJK::postiterations() { state_.reset(); }

void cuESTDirectJK::print_header() const {
    if (!print_) return;
    outfile->Printf("  ==> cuESTDirectJK: GPU-Accelerated Four-Center Direct J/K Matrices <==\n\n");
    outfile->Printf("    Density fitting:         No (no auxiliary basis)\n");
    outfile->Printf("    J tasked:                %s\n", do_J_ ? "Yes" : "No");
    outfile->Printf("    K tasked:                %s\n", do_K_ ? "Yes" : "No");
    outfile->Printf("    Exchange operator:       %.12g/r + %.12g erf(%.12g r)/r\n", omega_alpha_, omega_beta_, omega_);
    outfile->Printf("    Integral threshold:      %.3e\n", cutoff_);
    outfile->Printf("    Direct JIT:              %s (AUTO uses SDK/toolkit default)\n\n",
        options_.get_str("CUEST_DIRECT_JIT").c_str());
}

}  // namespace psi
#endif
