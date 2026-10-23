/* Opt-in synchronizing response diagnostics. SPDX-License-Identifier: LGPL-3.0-only */
#ifndef PSI4_CUEST_RESPONSE_PROFILE_H
#define PSI4_CUEST_RESPONSE_PROFILE_H
#ifdef USING_cuEST
#include <cuda_runtime.h>
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/exception.h"
extern cudaStream_t stream_handle;
namespace psi {
/// These wall timers deliberately fence the shared stream to attribute work.
/// They are for bottleneck diagnosis, NOT unperturbed speedup measurements.
class CuestResponseProfile {
    bool active_ = false;
    const char* label_;
   public:
    CuestResponseProfile(bool enabled, const char* label) : label_(label) {
        if (!enabled) return;
        if (cudaStreamSynchronize(::stream_handle) != cudaSuccess)
            throw PSIEXCEPTION("cuEST response profile: CUDA stream failed before a timed phase.");
        timer_on(label_);
        active_ = true;
    }
    CuestResponseProfile(const CuestResponseProfile&) = delete;
    CuestResponseProfile& operator=(const CuestResponseProfile&) = delete;
    void stop() {
        if (!active_) return;
        const auto status = cudaStreamSynchronize(::stream_handle);
        timer_off(label_);
        active_ = false;
        if (status != cudaSuccess)
            throw PSIEXCEPTION("cuEST response profile: CUDA stream failed after a timed phase.");
    }
    ~CuestResponseProfile() {
        if (active_) {
            cudaStreamSynchronize(::stream_handle);
            timer_off(label_);
        }
    }
};
}  // namespace psi
#endif
#endif
