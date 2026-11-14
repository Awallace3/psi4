// Standalone optimized/sanitized test; needs no Psi4 or BLAS installation.
#include "../../psi4/src/psi4/lib3index/dfhelper_scratch.h"
#include "../../psi4/include/psi4/restrict.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

static void scale(float* PSI_RESTRICT out, const float* PSI_RESTRICT in, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = 2.0f * in[i];
}

int main() {
    psi::DFHelperScratch buffer;
    assert(buffer.data() == nullptr);
    for (size_t n : {size_t(1), size_t(7), size_t(4097)}) {
        buffer.resize(n);
        assert(buffer.size() == n && buffer.capacity() == n);
        double* dp = buffer.data();
        assert(reinterpret_cast<std::uintptr_t>(dp) % alignof(double) == 0);
        for (size_t i = 0; i < n; ++i) dp[i] = static_cast<double>(i);
        assert(dp == buffer.data());
        const void* storage = dp;
        for (int repeat = 0; repeat < 10; ++repeat) {
            float* fp = buffer.float_data();
            assert(static_cast<void*>(fp) == storage);
            for (size_t i = 0; i < 2 * n; ++i) fp[i] = static_cast<float>(i);
            std::vector<float> output(2 * n);
            scale(output.data(), fp, 2 * n);
            for (size_t i = 0; i < 2 * n; ++i) assert(output[i] == 2.0f * i);
            assert(fp == buffer.float_data());
            dp = buffer.data();
            assert(static_cast<void*>(dp) == storage);
            for (size_t i = 0; i < n; ++i) dp[i] = static_cast<double>(i + repeat);
            for (size_t i = 0; i < n; ++i) assert(dp[i] == i + repeat);
        }
        buffer.resize(1);
        assert(buffer.size() == n);  // grow-only, exact high-water accounting
    }
    auto moved = std::move(buffer);
    assert(moved.data() != nullptr);
    moved.reset();
    assert(moved.size() == 0 && moved.float_data() == nullptr);
}
