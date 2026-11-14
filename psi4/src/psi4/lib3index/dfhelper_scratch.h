/*
 * Reusable, double-sized DFHelper scratch with explicit scalar object lifetimes.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_DFHELPER_SCRATCH_H
#define PSI4_DFHELPER_SCRATCH_H

#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace psi {

// A byte array provides storage for either scalar array, without accessing live
// doubles through float lvalues. Non-allocating placement array new starts the
// requested array's lifetime; switching precision neither allocates nor copies.
// Size/capacity remain in doubles, matching DFHelper's memory accounting.
class DFHelperScratch {
   public:
    void reset() {
        storage_.reset();
        size_ = 0;
        active_ = nullptr;
        single_ = false;
    }
    void resize(size_t size) {
        if (size <= size_) return;
        if (size > std::numeric_limits<size_t>::max() / sizeof(double)) throw std::bad_array_new_length();
        reset();  // Do not overlap the old and new high-water allocations.
        storage_.reset(new std::byte[size * sizeof(double)]);
        size_ = size;
    }
    size_t size() const { return size_; }
    size_t capacity() const { return size_; }

    template <typename T>
    T* data_as() {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
        constexpr bool single = std::is_same_v<T, float>;
        if (!size_) return nullptr;
        if (!active_ || single_ != single) {
#if defined(__has_feature)
#if __has_feature(type_sanitizer)
            // Experimental TySan does not see trivial placement-array-new
            // lifetime changes. Its documented memset interceptor resets the
            // shadow type. Only sanitizer builds touch the backing bytes here.
            std::memset(storage_.get(), 0, size_ * sizeof(double));
#endif
#endif
            active_ = ::new (static_cast<void*>(storage_.get())) T[size_ * sizeof(double) / sizeof(T)];
            single_ = single;
        }
        return static_cast<T*>(active_);
    }
    double* data() { return data_as<double>(); }
    double* get() { return data(); }
    float* float_data() { return data_as<float>(); }

   private:
    std::unique_ptr<std::byte[]> storage_;
    size_t size_ = 0;
    void* active_ = nullptr;
    bool single_ = false;
};

}  // namespace psi
#endif
