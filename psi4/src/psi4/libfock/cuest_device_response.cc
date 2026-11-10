/* SPDX-License-Identifier: LGPL-3.0-only */
#include "cuest_device_response.h"
#ifdef USING_cuEST
#include "cuESTCommon.h"
#include "cuest_response_profile.h"
#include "cuest_device_kernels.h"
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
    variable_.deviceBufferSizeInBytes = jk_.gridk_ ? jk_.gridk_workspace_bytes() : 8000000000ULL;
    const auto ndensity = view_.nrhs * view_.nocc.size();
    const int requested_batch = jk_.options_.get_int("CUEST_RESPONSE_BATCH_SIZE");
    if (requested_batch < 0)
        throw PSIEXCEPTION("CUEST_RESPONSE_BATCH_SIZE must be nonnegative.");
    batch_size_ = requested_batch ? std::min<std::size_t>(requested_batch, ndensity) : ndensity;
    if (!view_.last_nrhs) view_.last_nrhs = view_.nrhs;
    if (view_.last_nrhs > view_.nrhs)
        throw PSIEXCEPTION("cuEST device CG: last tile exceeds resident RHS capacity.");
    full_batches_ = cuest_response::density_batches(view_.nrhs, view_.nrhs, view_.nocc.size(), batch_size_);
    last_batches_ = cuest_response::density_batches(view_.nrhs, view_.last_nrhs, view_.nocc.size(), batch_size_);
    const auto nbf = static_cast<std::size_t>(jk_.primary_->nbf());
    const auto n2 = static_cast<std::size_t>(jk_.primary_->nbf()) * jk_.primary_->nbf();
    symmetric_gridk_ = view_.compute_exchange && view_.exchange_sum_only && jk_.gridk_ &&
                      jk_.options_.get_bool("CUEST_RESPONSE_SYMMETRIC_K");
    const double* grid_density = view_.density;
    if (symmetric_gridk_) {
#ifndef PSI4_CUEST_DEVICE_CG
        throw PSIEXCEPTION("Symmetric CPKS Grid-K requires the device CG NVRTC kernels.");
#else
        grid_density = symmetric_density_.ensure(ndensity * n2, "symmetric CPKS density");
#endif
    }
    const auto grow = [&](const cuestWorkspaceDescriptor_t& d) {
        scratch_.hostBufferSizeInBytes = std::max(scratch_.hostBufferSizeInBytes, d.hostBufferSizeInBytes);
        scratch_.deviceBufferSizeInBytes = std::max(scratch_.deviceBufferSizeInBytes, d.deviceBufferSizeInBytes);
    };
    cuestWorkspaceDescriptor_t d{};
    if (jk_.directj_) {
#ifdef PSI_CUEST_HAS_DIRECTJ
        for (const auto* batches : {&full_batches_, &last_batches_})
            for (const auto& batch : *batches) {
                CHECK_CUEST(cuestDirectCoulombComputeWorkspaceQuery(
                    cuest_handle, jk_.cuest_directj_plan_, jk_.cuest_directj_params_, &d,
                    batch.count, view_.density + batch.first*n2, view_.coulomb + batch.first*n2));
                grow(d);
                coulomb_workspace_bytes_ = std::max(coulomb_workspace_bytes_, d.deviceBufferSizeInBytes);
            }
#endif
    } else {
        jk_.query_coulomb_workspace(&d, view_.density, view_.coulomb);
        grow(d);
        coulomb_workspace_bytes_ = d.deviceBufferSizeInBytes;
    }
    if (view_.compute_exchange) {
        if (jk_.gridk_) {
#ifdef PSI_CUEST_HAS_GRIDK
            for (const auto* batches : {&full_batches_, &last_batches_})
              for (const auto& batch : *batches) {
                if (symmetric_gridk_) {
                    CHECK_CUEST(cuestGridKSymmetricExchangeComputeWorkspaceQuery(
                        cuest_handle, jk_.cuest_gridk_plan_, jk_.cuest_gridk_exchange_params_,
                        &variable_, &d, batch.count,
                        grid_density + batch.first*n2, view_.exchange + batch.first*n2));
                } else {
                    CHECK_CUEST(cuestGridKNonsymmetricExchangeComputeWorkspaceQuery(
                        cuest_handle, jk_.cuest_gridk_plan_, jk_.cuest_gridk_nonsym_exchange_params_,
                        &variable_, &d, batch.count,
                        grid_density + batch.first*n2, view_.exchange + batch.first*n2));
                }
                grow(d);
                exchange_workspace_bytes_ = std::max(exchange_workspace_bytes_, d.deviceBufferSizeInBytes);
            }
#endif
        } else {
            for (std::size_t s = 0; s < view_.nocc.size(); ++s) {
                if (!view_.nocc[s]) continue;
                for (const auto actual : {view_.nrhs, view_.last_nrhs})
                  for (std::size_t r = 0; r < actual; r += batch_size_) {
                    CHECK_CUEST(cuestDFNonsymmetricExchangeComputeWorkspaceQuery(
                        cuest_handle, jk_.cuest_df_plan_, jk_.cuest_nonsym_exchange_compute_params_,
                        &variable_, &d, std::min(batch_size_, actual-r), view_.nocc[s], view_.left[s],
                        view_.right[s] + r*view_.nocc[s]*nbf,
                        view_.exchange + (s * view_.nrhs + r) * n2));
                    grow(d);
                    exchange_workspace_bytes_ = std::max(exchange_workspace_bytes_, d.deviceBufferSizeInBytes);
                }
            }
        }
    }
    work_ = workspace_.sdk_workspace(scratch_);
}
void CuestJKDeviceResponse::compute(std::size_t actual_rhs) {
    if (!jk_.initialized_ || jk_.plan_key_ != plan_key_ || jk_.current_plan_key() != plan_key_)
        throw PSIEXCEPTION("cuEST device CG: JK plan changed during a resident response session.");
    if (jk_.gridk_ && variable_.deviceBufferSizeInBytes != jk_.gridk_workspace_bytes())
        throw PSIEXCEPTION("cuEST device CG: Grid-K workspace budget changed during a resident response session.");
    if (!actual_rhs) actual_rhs = view_.nrhs;
    if (actual_rhs != view_.nrhs && actual_rhs != view_.last_nrhs)
        throw PSIEXCEPTION("cuEST device CG: RHS shape was not workspace-queried during setup.");
    const auto& batches = actual_rhs == view_.nrhs ? full_batches_ : last_batches_;
    jk_.configure_math_mode();
    const auto n2 = static_cast<std::size_t>(jk_.primary_->nbf()) * jk_.primary_->nbf();
    const auto nbf = static_cast<std::size_t>(jk_.primary_->nbf());
    const auto ndensity = view_.nrhs * view_.nocc.size();
    const auto submitted_before = density_slots_submitted_;
    if (jk_.directj_) {
#ifdef PSI_CUEST_HAS_DIRECTJ
        CuestResponseProfile profile(jk_.options_.get_bool("CUEST_RESPONSE_PROFILE"), "cuEST response: J");
        for (const auto& batch : batches) {
            CHECK_CUEST(cuestDirectCoulombCompute(
                cuest_handle, jk_.cuest_directj_plan_, jk_.cuest_directj_params_, work_,
                batch.count, view_.density + batch.first*n2, view_.coulomb + batch.first*n2));
            ++coulomb_calls_;
            density_slots_submitted_ += batch.count;
        }
#endif
    } else {
        for (const auto& batch : batches)
            for (std::size_t n = batch.first; n < batch.first + batch.count; ++n) {
                jk_.compute_coulomb(work_, view_.density + n * n2, view_.coulomb + n * n2);
                ++coulomb_calls_;
                ++density_slots_submitted_;
            }
    }
    // Count actual SDK density arguments, not the requested shape. A regression
    // dispatching padding must be visible even when its zero density hides it
    // from numerical parity checks.
    padding_slots_avoided_ += ndensity - (density_slots_submitted_ - submitted_before);
    if (!view_.compute_exchange) return;
    const char* k_profile_label = !jk_.gridk_ ? "cuEST response: K" :
        symmetric_gridk_ ? (batch_size_ == 1 ? "cuEST response: K sym batch1" : "cuEST response: K sym batched") :
                          (batch_size_ == 1 ? "cuEST response: K gen batch1" : "cuEST response: K gen batched");
    CuestResponseProfile profile(jk_.options_.get_bool("CUEST_RESPONSE_PROFILE"), k_profile_label);
    if (jk_.gridk_) {
#ifdef PSI_CUEST_HAS_GRIDK
        const double* grid_density = view_.density;
        if (symmetric_gridk_) {
#ifdef PSI4_CUEST_DEVICE_CG
            double* symmetric = symmetric_density_.ensure(ndensity * n2, "symmetric CPKS density");
            const bool full = actual_rhs == view_.nrhs;
            const auto panels = full ? 1 : view_.nocc.size();
            const auto count = actual_rhs * n2 * (full ? view_.nocc.size() : 1);
            for (std::size_t s = 0; s < panels; ++s) {
                const auto offset = s * view_.nrhs * n2;
                CuestDeviceKernels::instance().launch("cg_symmetric_density", count,
                    count, nbf, view_.density + offset, symmetric + offset);
            }
            grid_density = symmetric;
#endif
        }
        // cuEST's nonsymmetric kernel restores operator symmetry with two
        // Grid-K builds. CPKS consumes only the Hermitian sum, for which
        // K(D)+K(D)^T = 2*K_sym((D+D^T)/2). Keep general K unchanged elsewhere.
        for (const auto& batch : batches) {
            if (symmetric_gridk_) {
                CHECK_CUEST(cuestGridKSymmetricExchangeCompute(
                    cuest_handle, jk_.cuest_gridk_plan_, jk_.cuest_gridk_exchange_params_,
                    &variable_, work_, batch.count,
                    grid_density + batch.first*n2, view_.exchange + batch.first*n2));
            } else {
                CHECK_CUEST(cuestGridKNonsymmetricExchangeCompute(
                    cuest_handle, jk_.cuest_gridk_plan_, jk_.cuest_gridk_nonsym_exchange_params_,
                    &variable_, work_, batch.count,
                    grid_density + batch.first*n2, view_.exchange + batch.first*n2));
            }
            ++exchange_calls_;
            exchange_density_slots_submitted_ += batch.count;
        }
#endif
    } else {
        for (std::size_t s = 0; s < view_.nocc.size(); ++s) {
            double* out = view_.exchange + s * view_.nrhs * n2;
            if (!view_.nocc[s]) {
                const auto status = cudaMemsetAsync(out, 0, actual_rhs * n2 * sizeof(double), stream_handle);
                if (status != cudaSuccess) throw PSIEXCEPTION("cuEST device CG: cannot zero empty-spin exchange.");
                continue;
            }
            for (std::size_t r = 0; r < actual_rhs; r += batch_size_) {
                CHECK_CUEST(cuestDFNonsymmetricExchangeCompute(
                    cuest_handle, jk_.cuest_df_plan_, jk_.cuest_nonsym_exchange_compute_params_, &variable_, work_,
                    std::min(batch_size_, actual_rhs-r), view_.nocc[s], view_.left[s],
                    view_.right[s] + r*view_.nocc[s]*nbf, out + r*n2));
                ++exchange_calls_;
                exchange_density_slots_submitted_ += std::min(batch_size_, actual_rhs-r);
            }
        }
    }
}
}  // namespace psi
#endif
