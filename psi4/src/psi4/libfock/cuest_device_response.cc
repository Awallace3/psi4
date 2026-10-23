/* SPDX-License-Identifier: LGPL-3.0-only */
#include "cuest_device_response.h"
#ifdef USING_cuEST
#include "cuESTCommon.h"
#include "cuest_response_profile.h"
#include "psi4/liboptions/liboptions.h"
#include <algorithm>
#include <utility>
extern cudaStream_t stream_handle;
namespace psi {
CuestJKDeviceResponse::CuestJKDeviceResponse(cuESTJK& jk, CuestDeviceJKView view)
    : jk_(jk), view_(std::move(view)) {
    if (!jk_.initialized_)
        throw PSIEXCEPTION("cuEST device CG requires an initialized JK; call initialize() after finalize/failure.");
    jk_.preiterations();
    if (!view_.nrhs || view_.nocc.empty() || view_.nocc.size() > 2 ||
        !view_.density || !view_.coulomb || (view_.compute_exchange && !view_.exchange))
        throw PSIEXCEPTION("cuEST device CG: invalid resident JK buffers or batch shape.");
    if (!jk_.do_J_ || (view_.compute_exchange && !jk_.do_K_))
        throw PSIEXCEPTION("cuEST device CG: attached JK task mask does not supply the required J/K operator.");
    plan_key_ = jk_.plan_key_;
    variable_.deviceBufferSizeInBytes = 8000000000ULL;
    const auto ndensity = view_.nrhs * view_.nocc.size();
    const auto n2 = static_cast<std::size_t>(jk_.primary_->nbf()) * jk_.primary_->nbf();
    const auto grow = [&](const cuestWorkspaceDescriptor_t& d) {
        scratch_.hostBufferSizeInBytes = std::max(scratch_.hostBufferSizeInBytes, d.hostBufferSizeInBytes);
        scratch_.deviceBufferSizeInBytes = std::max(scratch_.deviceBufferSizeInBytes, d.deviceBufferSizeInBytes);
    };
    cuestWorkspaceDescriptor_t d{};
    if (jk_.directj_) {
#ifdef PSI_CUEST_HAS_DIRECTJ
        CHECK_CUEST(cuestDirectCoulombComputeWorkspaceQuery(
            cuest_handle, jk_.cuest_directj_plan_, jk_.cuest_directj_params_, &d,
            ndensity, view_.density, view_.coulomb));
#endif
    } else {
        jk_.query_coulomb_workspace(&d, view_.density, view_.coulomb);
    }
    grow(d);
    if (view_.compute_exchange) {
        if (jk_.gridk_) {
#ifdef PSI_CUEST_HAS_GRIDK
            CHECK_CUEST(cuestGridKNonsymmetricExchangeComputeWorkspaceQuery(
                cuest_handle, jk_.cuest_gridk_plan_, jk_.cuest_gridk_nonsym_exchange_params_,
                &variable_, &d, ndensity, view_.density, view_.exchange));
            grow(d);
#endif
        } else {
            for (std::size_t s = 0; s < view_.nocc.size(); ++s) {
                if (!view_.nocc[s]) continue;
                CHECK_CUEST(cuestDFNonsymmetricExchangeComputeWorkspaceQuery(
                    cuest_handle, jk_.cuest_df_plan_, jk_.cuest_nonsym_exchange_compute_params_,
                    &variable_, &d, view_.nrhs, view_.nocc[s], view_.left[s], view_.right[s],
                    view_.exchange + s * view_.nrhs * n2));
                grow(d);
            }
        }
    }
    work_ = workspace_.sdk_workspace(scratch_);
}
void CuestJKDeviceResponse::compute() {
    if (!jk_.initialized_ || jk_.plan_key_ != plan_key_)
        throw PSIEXCEPTION("cuEST device CG: JK plan changed during a resident response session.");
    jk_.configure_math_mode();
    const auto n2 = static_cast<std::size_t>(jk_.primary_->nbf()) * jk_.primary_->nbf();
    const auto ndensity = view_.nrhs * view_.nocc.size();
    if (jk_.directj_) {
#ifdef PSI_CUEST_HAS_DIRECTJ
        CuestResponseProfile profile(jk_.options_.get_bool("CUEST_RESPONSE_PROFILE"), "cuEST response: J");
        CHECK_CUEST(cuestDirectCoulombCompute(
            cuest_handle, jk_.cuest_directj_plan_, jk_.cuest_directj_params_, work_,
            ndensity, view_.density, view_.coulomb));
#endif
    } else {
        for (std::size_t n = 0; n < ndensity; ++n)
            jk_.compute_coulomb(work_, view_.density + n * n2, view_.coulomb + n * n2);
    }
    if (!view_.compute_exchange) return;
    CuestResponseProfile profile(jk_.options_.get_bool("CUEST_RESPONSE_PROFILE"), "cuEST response: K");
    if (jk_.gridk_) {
#ifdef PSI_CUEST_HAS_GRIDK
        CHECK_CUEST(cuestGridKNonsymmetricExchangeCompute(
            cuest_handle, jk_.cuest_gridk_plan_, jk_.cuest_gridk_nonsym_exchange_params_,
            &variable_, work_, ndensity, view_.density, view_.exchange));
#endif
    } else {
        for (std::size_t s = 0; s < view_.nocc.size(); ++s) {
            double* out = view_.exchange + s * view_.nrhs * n2;
            if (!view_.nocc[s]) {
                const auto status = cudaMemsetAsync(out, 0, view_.nrhs * n2 * sizeof(double), stream_handle);
                if (status != cudaSuccess) throw PSIEXCEPTION("cuEST device CG: cannot zero empty-spin exchange.");
                continue;
            }
            CHECK_CUEST(cuestDFNonsymmetricExchangeCompute(
                cuest_handle, jk_.cuest_df_plan_, jk_.cuest_nonsym_exchange_compute_params_, &variable_, work_,
                view_.nrhs, view_.nocc[s], view_.left[s], view_.right[s], out));
        }
    }
}
}  // namespace psi
#endif
