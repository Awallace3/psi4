/*
 * Psi4: an open-source quantum chemistry software package
 * Copyright (c) 2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#ifndef PSI4_SAPT_FDDS_TENSOR_PROVIDER_H
#define PSI4_SAPT_FDDS_TENSOR_PROVIDER_H

#include "psi4/libmints/typedefs.h"

#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace psi {
class BasisSet;
namespace sapt {

/// Logical contiguous FP64 tensor: offset(i,j,k) = (i*d1+j)*d2+k.
using FDDSTensorShape = std::array<size_t, 3>;
/// Half-open first-axis slice. All second/third-axis elements are included.
using FDDSTensorSlice = std::array<size_t, 2>;
enum class FDDSTensorKind { RawCoulomb, MetricInverseHalf, Derived };

/**
 * Integral production and derived-tensor storage seam, deliberately independent
 * of DFHelper's space registration and transformation/lifetime protocol.
 *
 * This first interface transfers HOST buffers. It is not a device lease and
 * does not claim residency. Raw Coulomb tensors and metric(-1/2) tensors retain
 * different names/semantics; a fitted DFMO result is NOT a raw tensor.
 * No backend retry is performed by this interface.
 */
class FDDSTensorProvider {
   public:
    virtual ~FDDSTensorProvider() = default;

    const FDDSTensorShape& shape(const std::string& name) const {
        const auto found = tensors_.find(name);
        if (found == tensors_.end()) fail("tensor not found: " + name);
        return found->second.shape;
    }
    FDDSTensorKind kind(const std::string& name) const {
        shape(name);  // Same missing-name preflight as every other operation.
        return tensors_.at(name).kind;
    }

    static size_t elements(FDDSTensorShape dimensions) {
        for (auto value : dimensions)
            if (!value) return 0;
        size_t count = 1;
        for (auto value : dimensions) {
            if (value > std::numeric_limits<size_t>::max() / count) fail("tensor size overflow");
            count *= value;
        }
        if (count > std::numeric_limits<size_t>::max() / sizeof(double)) fail("tensor byte size overflow");
        return count;
    }

    void read(const std::string& name, double* destination, size_t capacity, FDDSTensorSlice slice) {
        const auto count = slice_elements(name, slice);
        if (count > capacity || (count && !destination)) fail("read buffer too small");
        if (count) read_impl(name, destination, slice);
    }

    void write(const std::string& name, const double* source, size_t capacity, FDDSTensorSlice slice) {
        const auto count = slice_elements(name, slice);
        if (kind(name) != FDDSTensorKind::Derived) fail("cannot overwrite an integral tensor");
        if (count > capacity || (count && !source)) fail("write buffer too small");
        if (count) write_impl(name, source, slice);
    }

    // Matrix overloads check the actual backing capacity, not the requested
    // shape. Backing storage is not reshaped and padded tails are untouched;
    // the NumPy view metadata follows the requested logical slice.
    void read(const std::string& name, SharedMatrix destination, FDDSTensorSlice slice);
    void write(const std::string& name, SharedMatrix source, FDDSTensorSlice slice);

    void create(const std::string& name, FDDSTensorShape dimensions) {
        if (name.empty() || tensors_.count(name)) fail("duplicate or empty tensor name");
        elements(dimensions);  // Before a backend allocation or file side effect.
        create_impl(name, dimensions);
        tensors_.emplace(name, Tensor{dimensions, FDDSTensorKind::Derived});
    }

   protected:
    void register_integral(const std::string& name, FDDSTensorShape dimensions, FDDSTensorKind kind) {
        if (name.empty() || tensors_.count(name)) fail("duplicate or empty integral name");
        if (kind == FDDSTensorKind::Derived) fail("integral registration requires an explicit Coulomb factor kind");
        elements(dimensions);
        tensors_.emplace(name, Tensor{dimensions, kind});
    }

   private:
    struct Tensor {
        FDDSTensorShape shape;
        FDDSTensorKind kind;
    };
    std::map<std::string, Tensor> tensors_;

    [[noreturn]] static void fail(const std::string& message) {
        throw std::runtime_error("FDDS tensor provider: " + message);
    }
    size_t slice_elements(const std::string& name, FDDSTensorSlice slice) const {
        auto dimensions = shape(name);
        if (slice[0] > slice[1] || slice[1] > dimensions[0]) fail("invalid first-axis slice");
        dimensions[0] = slice[1] - slice[0];
        return elements(dimensions);
    }
    virtual void read_impl(const std::string&, double*, FDDSTensorSlice) = 0;
    virtual void write_impl(const std::string&, const double*, FDDSTensorSlice) = 0;
    virtual void create_impl(const std::string&, FDDSTensorShape) = 0;
};

struct FDDSTensorPreparation {
    std::shared_ptr<BasisSet> primary, auxiliary;
    // A occupied/virtual, B occupied/virtual; exactly the existing FDDS ordering.
    std::array<SharedMatrix, 4> coefficients;
    std::array<bool, 2> empty;
    bool hybrid;
    size_t memory_doubles;
    size_t threads;
    bool metric_gpu = false;
    int metric_gpu_min_dim = 256;
};

/// Native adapter: unchanged integral operator, screening, fitting and two-round
/// transform policy. No cuEST integral-production claim is implied.
std::unique_ptr<FDDSTensorProvider> make_native_fdds_tensors(const FDDSTensorPreparation&);

}  // namespace sapt
}  // namespace psi
#endif
