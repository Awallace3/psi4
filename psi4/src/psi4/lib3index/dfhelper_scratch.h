#pragma once

#include <cstddef>
#include <memory>

namespace psi {
/// Exactly sized, move-only double scratch. Growth releases the previous
/// allocation first; capacities are charged to the DFHelper memory planner.
class DFHelperScratch {
    std::unique_ptr<double[]> data_;
    size_t size_ = 0;
public:
    size_t size() const { return size_; }
    size_t capacity() const { return size_; }
    void reset() { data_.reset(); size_ = 0; }
    void resize(size_t words) {
        if (words == size_) return;
        reset();
        data_.reset(new double[words]);
        size_ = words;
    }
    double* data() { return data_.get(); }
    double* get() { return data(); }
};
}  // namespace psi
