/*
 * Opt-in XC diagnostic timing. Device boundaries intentionally serialize work.
 * Host worker timers report thread seconds and must not be summed into wall time.
 */
#ifndef PSI4_XC_TIMING_H
#define PSI4_XC_TIMING_H

#include <string>
#include <algorithm>
#include <type_traits>
#include <memory>
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/exception.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USING_cuEST
#include <cuda_runtime.h>
#endif

namespace psi {
namespace xc_timing {

/// Preserve existing parent labels while balancing them on exceptions.
/// Declare before nested Stages so those close first during stack unwinding.
class CoarseScope {
    std::string label_;
    bool active_ = false;

   public:
    CoarseScope() = default;
    explicit CoarseScope(const char* label) { start(label); }
    CoarseScope(const CoarseScope&) = delete;
    CoarseScope& operator=(const CoarseScope&) = delete;
    void start(const char* label) {
        stop();
        label_ = label;
        timer_on(label_);
        active_ = true;
    }
    void stop() {
        if (!active_) return;
        timer_off(label_);
        active_ = false;
    }
    ~CoarseScope() { stop(); }
};

class Scope {
    bool active_ = false;
    bool device_;
    int rank_;
    std::string label_;

    void close() {
        if (rank_ < 0) timer_off(label_);
        else parallel_timer_off(label_, rank_);
        active_ = false;
    }

   public:
    Scope(const char* label, bool device, int rank) : device_(device), rank_(rank) {
        // Stringified API calls may contain arguments. Keep only the function name.
        label_ = "XC micro: " + std::string(label);
        label_.resize(label_.find('(') == std::string::npos ? label_.size() : label_.find('('));
        std::replace(label_.begin(), label_.end(), ';', ':');  // reserved timer-path delimiter
#ifdef USING_cuEST
        if (device_ && cudaDeviceSynchronize() != cudaSuccess) {
            throw PSIEXCEPTION("CUDA synchronization failed before " + label_);
        }
#endif
        if (rank_ < 0) timer_on(label_);
        else parallel_timer_on(label_, rank_);
        active_ = true;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    void stop() {
#ifdef USING_cuEST
        const auto status = device_ ? cudaDeviceSynchronize() : cudaSuccess;
#endif
        close();
#ifdef USING_cuEST
        if (status != cudaSuccess) throw PSIEXCEPTION("CUDA synchronization failed after " + label_);
#endif
    }
    ~Scope() {
        if (!active_) return;
#ifdef USING_cuEST
        if (device_) cudaDeviceSynchronize();
#endif
        close();
    }
};

template <typename F>
auto call(bool enabled, const char* label, F&& operation, bool device = false, int rank = -1) {
    if (!enabled) return operation();
    Scope timer(label, device, rank);
    if constexpr (std::is_void_v<decltype(operation())>) {
        operation();
        timer.stop();
    } else {
        auto result = operation();
        timer.stop();
        return result;
    }
}

inline int worker_rank() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/// Sequential inclusive phase timers, with nested API calls providing the detail.
class Stages {
    bool enabled_;
    int rank_;
    std::unique_ptr<Scope> scope_;

   public:
    explicit Stages(bool enabled, int rank = -1) : enabled_(enabled), rank_(rank) {}
    void start(const char* label) {
        if (!enabled_) return;
        if (scope_) scope_->stop();
        scope_ = std::make_unique<Scope>(label, false, rank_);
    }
    void stop() {
        if (scope_) scope_->stop();
        scope_.reset();
    }
};

}  // namespace xc_timing
}  // namespace psi
#endif
