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
// The interface to cuEST was contributed by NVIDIA under the following terms:
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LGPL-3.0-only

/*! \file cuest_gemm.cc
 *  \brief cuBLAS matrix-multiplication chains for the SAPT(DFT) tensor code.
 *
 *  The SAPT tensor code uses ``chain_gemm_einsums`` in
 *  psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py for left-associated matrix
 *  products. Its CPU path stores each intermediate in a host ``core.Matrix``.
 *
 *  ``cuest_chain_gemm`` below is the GPU counterpart: it uploads the operands
 *  once each, keeps intermediate products on the device across the whole
 *  chain, and copies back only the products the caller requested. Unlike a
 *  per-link GPU implementation, it does not upload/download the running
 *  product at every link. This offloads eligible SAPT(DFT)-D4(I) tensor chains;
 *  small chains and other host-side SAPT work remain on the CPU.
 *
 *  This file is compiled only when cuEST is enabled -- not because the code
 *  needs cuEST itself, but because that is the switch that guarantees a CUDA
 *  toolchain and, more to the point, the process-global ``cublas_handle`` that
 *  cuest_runtime.cc creates and binds to cuEST's stream.  Sharing that handle
 *  keeps these GEMMs on the same stream as the J/K builds instead of racing a
 *  second one.
 */

#ifdef USING_cuEST

#include <algorithm>
#include <cstddef>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "psi4/pybind11.h"

#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libqt/qt.h"
#include "psi4/libfock/cuESTCommon.h"
#include "cuest_gemm.h"

using namespace psi;
namespace py = pybind11;
using namespace pybind11::literals;

extern cublasHandle_t cublas_handle;

namespace psi {

namespace {

/*! A device allocation that frees itself.
 *
 *  The chain below throws on any CUDA or cuBLAS failure, and it holds up to
 *  three live allocations at once (the running product, the next operand, and
 *  the product being formed).  Unwinding through raw ``cudaFree`` calls would
 *  need a try/catch around the whole loop; this needs nothing.
 */
class DeviceBuffer {
   public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t elements, const char* what) {
        if (elements == 0) return;
        if (elements > std::numeric_limits<size_t>::max() / sizeof(double))
            throw PSIEXCEPTION("cuEST GEMM: device allocation size overflow");
        cudaError_t err = cudaMalloc((void**)&ptr_, elements * sizeof(double));
        if (err != cudaSuccess) {
            std::ostringstream msg;
            msg << "cuest_chain_gemm: cudaMalloc for " << what << " (" << elements
                << " doubles) failed: " << cudaGetErrorString(err);
            throw PSIEXCEPTION(msg.str());
        }
    }

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    double* get() const { return ptr_; }

   private:
    double* ptr_ = nullptr;
};

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream msg;
        msg << "cuest_chain_gemm: " << what << " failed: " << cudaGetErrorString(err);
        throw PSIEXCEPTION(msg.str());
    }
}

void check_cublas(cublasStatus_t stat, const char* what) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream msg;
        msg << "cuest_chain_gemm: " << what << " failed (cuBLAS status " << static_cast<int>(stat) << ")";
        throw PSIEXCEPTION(msg.str());
    }
}

cublasOperation_t to_cublas_op(const std::string& t, size_t position) {
    if (t == "N") return CUBLAS_OP_N;
    if (t == "T") return CUBLAS_OP_T;
    std::ostringstream msg;
    msg << "cuest_chain_gemm: transposes[" << position << "] is \"" << t << "\"; expected \"N\" or \"T\"";
    throw PSIEXCEPTION(msg.str());
}

/// Shape of a matrix as it is stored, before any transpose is applied.
struct Shape {
    int rows;
    int cols;
};

/// One multiplication of the chain, resolved to its cuBLAS arguments.
struct Link {
    cublasOperation_t op_a;
    cublasOperation_t op_b;
    int m;  ///< rows of the product
    int n;  ///< columns of the product
    int k;  ///< contracted dimension
    int lda;  ///< stored column count of the left operand
    int ldb;  ///< stored column count of the right operand
};

}  // namespace

namespace {
size_t fdds_product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw PSIEXCEPTION("cuEST FDDS: size overflow");
    return a * b;
}
size_t fdds_sum(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b)
        throw PSIEXCEPTION("cuEST FDDS: size overflow");
    return a + b;
}
size_t exchange_storage(size_t q, size_t v, size_t k, size_t outer, size_t inner) {
    const size_t left = fdds_product(outer, fdds_sum(k, fdds_product(2, v)));
    const size_t right = fdds_product(inner, fdds_sum(v, fdds_product(2, k)));
    return fdds_sum(fdds_product(q, fdds_sum(left, right)), fdds_product(v, k));
}
struct FDDSEnqueueGuard {
    cudaStream_t stream;
    bool complete = false;
    ~FDDSEnqueueGuard() {
        if (!complete) cudaStreamSynchronize(stream);
    }
};
struct MetricTimer {
    const char* name;
    explicit MetricTimer(const char* name_) : name(name_) { timer_on(name); }
    ~MetricTimer() { timer_off(name); }
};
}

size_t CuestDFMetric::capacity(size_t q, size_t requested) {
    if (!q || !requested || q > static_cast<size_t>(std::numeric_limits<int>::max())) return 0;
    const size_t fixed = fdds_product(q, q);
    fdds_product(fdds_sum(fixed, fdds_product(2, requested)), sizeof(double));
    cuest_common::ensure_cuest_initialized();
    size_t free = 0, total = 0;
    check_cuda(cudaMemGetInfo(&free, &total), "DF metric capacity");
    const size_t budget = (free / sizeof(double) / 5) * 4;
    return fixed < budget ? std::min(requested, (budget - fixed) / 2) : 0;
}

struct CuestDFMetric::Impl {
    size_t q, count;
    int min_dim;
    cudaStream_t stream = nullptr;
    DeviceBuffer metric, input, output;
    Impl(size_t q_, size_t count_, const double* metp, int threshold)
        : q(q_), count(count_), min_dim(threshold) {
        if (!q || !count || !metp || min_dim < 0 ||
            q > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw PSIEXCEPTION("cuEST DF metric: invalid dimensions, threshold or metric");
        fdds_product(fdds_sum(fdds_product(q, q), fdds_product(2, count)), sizeof(double));
        cuest_common::ensure_cuest_initialized();
        cublasPointerMode_t mode;
        check_cublas(cublasGetPointerMode(cublas_handle, &mode), "DF metric pointer mode");
        if (mode != CUBLAS_POINTER_MODE_HOST)
            throw PSIEXCEPTION("cuEST DF metric: shared cuBLAS handle requires HOST pointer mode");
        check_cublas(cublasGetStream(cublas_handle, &stream), "DF metric stream");
        MetricTimer timer("cuEST FDDS: metric setup");
        metric = DeviceBuffer(fdds_product(q, q), "DF metric");
        input = DeviceBuffer(count, "DF metric input");
        output = DeviceBuffer(count, "DF metric output");
        // A failed constructor does not run ~Impl: drain before its members or
        // the caller's borrowed metric can unwind.
        FDDSEnqueueGuard guard{stream};
        check_cuda(cudaMemcpyAsync(metric.get(), metp, fdds_product(fdds_product(q, q), sizeof(double)),
                                   cudaMemcpyHostToDevice, stream), "DF metric upload");
        check_cuda(cudaStreamSynchronize(stream), "DF metric upload completion");
        guard.complete = true;
    }
    ~Impl() { cudaStreamSynchronize(stream); }
};

CuestDFMetric::CuestDFMetric(size_t q, size_t count, const double* metric, int threshold)
    : impl_(new Impl(q, count, metric, threshold)) {}
CuestDFMetric::~CuestDFMetric() = default;
size_t CuestDFMetric::elements() const { return impl_->count; }
bool CuestDFMetric::eligible(size_t p, size_t r) const {
    const auto& d = *impl_;
    if (!p || !r || r > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        p > static_cast<size_t>(std::numeric_limits<int>::max()) / r) return false;
    const size_t rows = p * r;
    return rows <= d.count / d.q &&
        static_cast<long double>(d.q) * d.q * rows >=
            static_cast<long double>(d.min_dim) * d.min_dim * d.min_dim;
}

void CuestDFMetric::fold(Layout layout, const double* input, double* output, size_t p, size_t r) {
    auto& d = *impl_;
    if (!input || !output || !eligible(p, r))
        throw PSIEXCEPTION("cuEST DF metric: invalid or ineligible block");
    if (layout != Layout::pqQ && layout != Layout::Qpq && layout != Layout::pQq &&
        layout != Layout::Qpq_to_pqQ)
        throw PSIEXCEPTION("cuEST DF metric: unsupported layout");
    const size_t rows = p * r, bytes = fdds_product(fdds_product(rows, d.q), sizeof(double));
    FDDSEnqueueGuard guard{d.stream};
    MetricTimer timer("cuEST FDDS: metric fold");  // includes all transfers and completion
    check_cuda(cudaMemcpyAsync(d.input.get(), input, bytes, cudaMemcpyHostToDevice, d.stream),
               "DF metric block upload");
    const double one = 1.0, zero = 0.0;
    // Row-major products use reversed operands in column-major cuBLAS.
    if (layout == Layout::pqQ || layout == Layout::Qpq_to_pqQ) {
        check_cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N,
                         layout == Layout::pqQ ? CUBLAS_OP_N : CUBLAS_OP_T,
                         d.q, rows, d.q, &one, d.metric.get(), d.q,
                         d.input.get(), layout == Layout::pqQ ? d.q : rows,
                         &zero, d.output.get(), d.q), "DF metric right GEMM");
    } else if (layout == Layout::Qpq) {
        check_cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         rows, d.q, d.q, &one, d.input.get(), rows,
                         d.metric.get(), d.q, &zero, d.output.get(), rows), "DF metric left GEMM");
    } else {
        // Never enter the native OpenMP loop with a shared cuBLAS handle.
        for (size_t i = 0; i < p; ++i) {
            check_cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             r, d.q, d.q, &one, d.input.get() + i * d.q * r, r,
                             d.metric.get(), d.q, &zero, d.output.get() + i * d.q * r, r),
                         "DF metric slice GEMM");
        }
    }
    check_cuda(cudaMemcpyAsync(output, d.output.get(), bytes, cudaMemcpyDeviceToHost, d.stream),
               "DF metric block download");
    check_cuda(cudaStreamSynchronize(d.stream), "DF metric block completion");
    guard.complete = true;
}

std::pair<size_t, size_t> CuestFDDSExchange::capacity(size_t q, size_t v, size_t k,
                                                    size_t outer, size_t inner) {
    if (!q || !v || !k || !outer || !inner ||
        std::max({q, v, k}) > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {0, 0};
    cuest_common::ensure_cuest_initialized();
    size_t free = 0, total = 0;
    check_cuda(cudaMemGetInfo(&free, &total), "FDDS exchange memory preflight");
    const size_t budget = (free / sizeof(double) / 5) * 4;
    while (exchange_storage(q, v, k, outer, inner) > budget) {
        if (outer == 1 && inner == 1) return {0, 0};
        const size_t left = fdds_product(outer, fdds_sum(k, fdds_product(2, v)));
        const size_t right = fdds_product(inner, fdds_sum(v, fdds_product(2, k)));
        if (outer > 1 && (inner == 1 || left >= right)) outer = outer / 2 + outer % 2;
        else inner = inner / 2 + inner % 2;
    }
    return {outer, inner};
}

struct CuestFDDSExchange::Impl {
    size_t q, v, k, max_outer, max_inner, outer_count = 0;
    cudaStream_t stream = nullptr;
    DeviceBuffer outer, integrals, raw, orthogonal, out_raw, out_orthogonal, scratch;
    Impl(size_t q_, size_t v_, size_t k_, size_t bo, size_t bi)
        : q(q_), v(v_), k(k_), max_outer(bo), max_inner(bi) {
        if (!q || !v || !k || !bo || !bi ||
            std::max({q, v, k}) > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw PSIEXCEPTION("cuEST FDDS exchange: nonempty BLAS-int dimensions required");
        // Validate all sums/products before initialization or allocation.
        fdds_product(exchange_storage(q, v, k, bo, bi), sizeof(double));
        cuest_common::ensure_cuest_initialized();
        cublasPointerMode_t mode;
        check_cublas(cublasGetPointerMode(cublas_handle, &mode), "FDDS exchange pointer mode");
        if (mode != CUBLAS_POINTER_MODE_HOST)
            throw PSIEXCEPTION("cuEST FDDS exchange: shared cuBLAS handle requires HOST pointer mode");
        check_cublas(cublasGetStream(cublas_handle, &stream), "FDDS exchange stream");
        outer = DeviceBuffer(fdds_product(fdds_product(bo, k), q), "FDDS exchange outer");
        integrals = DeviceBuffer(fdds_product(fdds_product(bi, v), q), "FDDS exchange inner");
        raw = DeviceBuffer(fdds_product(fdds_product(bi, k), q), "FDDS exchange raw");
        orthogonal = DeviceBuffer(fdds_product(fdds_product(bi, k), q), "FDDS exchange orthogonal");
        out_raw = DeviceBuffer(fdds_product(fdds_product(bo, v), q), "FDDS exchange output");
        out_orthogonal = DeviceBuffer(fdds_product(fdds_product(bo, v), q), "FDDS exchange Q output");
        scratch = DeviceBuffer(fdds_product(v, k), "FDDS exchange pair");
    }
    ~Impl() {
        // Drain queued transfers even while unwinding; never let caller-owned
        // host scratch die while an asynchronous upload still references it.
        cudaStreamSynchronize(stream);
    }
    void upload(double* target, const double* source, size_t count) {
        if (!source) throw PSIEXCEPTION("cuEST FDDS exchange: null input");
        check_cuda(cudaMemcpyAsync(target, source, fdds_product(count, sizeof(double)),
                                   cudaMemcpyHostToDevice, stream), "FDDS exchange upload");
    }
};

CuestFDDSExchange::CuestFDDSExchange(size_t q, size_t v, size_t k, size_t outer, size_t inner)
    : impl_(new Impl(q, v, k, outer, inner)) {}
CuestFDDSExchange::~CuestFDDSExchange() = default;

void CuestFDDSExchange::begin(const double* integrals, size_t count) {
    auto& d = *impl_;
    FDDSEnqueueGuard guard{d.stream};
    if (!count || count > d.max_outer || d.outer_count)
        throw PSIEXCEPTION("cuEST FDDS exchange: invalid outer block");
    d.outer_count = count;
    d.upload(d.outer.get(), integrals, fdds_product(fdds_product(count, d.k), d.q));
    const size_t bytes = fdds_product(fdds_product(fdds_product(count, d.v), d.q), sizeof(double));
    check_cuda(cudaMemsetAsync(d.out_raw.get(), 0, bytes, d.stream), "FDDS exchange output zero");
    check_cuda(cudaMemsetAsync(d.out_orthogonal.get(), 0, bytes, d.stream), "FDDS exchange Q output zero");
    check_cuda(cudaStreamSynchronize(d.stream), "FDDS exchange outer upload completion");
    guard.complete = true;
}

void CuestFDDSExchange::accumulate(const double* integrals, const double* raw,
                                  const double* orthogonal, size_t count) {
    auto& d = *impl_;
    FDDSEnqueueGuard guard{d.stream};
    if (!d.outer_count || !count || count > d.max_inner)
        throw PSIEXCEPTION("cuEST FDDS exchange: invalid inner block");
    d.upload(d.integrals.get(), integrals, fdds_product(fdds_product(count, d.v), d.q));
    const size_t right = fdds_product(fdds_product(count, d.k), d.q);
    d.upload(d.raw.get(), raw, right);
    d.upload(d.orthogonal.get(), orthogonal, right);
    const double one = 1.0, zero = 0.0;
    for (size_t a = 0; a < d.outer_count; ++a) {
        const double* outer = d.outer.get() + a * d.k * d.q;
        double* output = d.out_raw.get() + a * d.v * d.q;
        double* qoutput = d.out_orthogonal.get() + a * d.v * d.q;
        for (size_t b = 0; b < count; ++b) {
            // Row-major V(v,k) = inner(v,q) outer(k,q)^T.
            // Column-major V^T(k,v) = outer_col^T(k,q) inner_col(q,v).
            check_cublas(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     d.k, d.v, d.q, &one, outer, d.q,
                                     d.integrals.get() + b * d.v * d.q, d.q,
                                     &zero, d.scratch.get(), d.k), "FDDS exchange pair GEMM");
            // Output_col(q,v) += right_col(q,k) V_col(k,v).
            check_cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     d.q, d.v, d.k, &one, d.raw.get() + b * d.k * d.q, d.q,
                                     d.scratch.get(), d.k, &one, output, d.q), "FDDS exchange raw GEMM");
            check_cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     d.q, d.v, d.k, &one, d.orthogonal.get() + b * d.k * d.q, d.q,
                                     d.scratch.get(), d.k, &one, qoutput, d.q), "FDDS exchange Q GEMM");
        }
    }
    check_cuda(cudaStreamSynchronize(d.stream), "FDDS exchange inner completion");
    guard.complete = true;
}

void CuestFDDSExchange::finish(double* raw, double* orthogonal) {
    auto& d = *impl_;
    FDDSEnqueueGuard guard{d.stream};
    if (!d.outer_count || !raw || !orthogonal)
        throw PSIEXCEPTION("cuEST FDDS exchange: invalid output block");
    const size_t bytes = fdds_product(fdds_product(fdds_product(d.outer_count, d.v), d.q), sizeof(double));
    check_cuda(cudaMemcpyAsync(raw, d.out_raw.get(), bytes, cudaMemcpyDeviceToHost, d.stream),
               "FDDS exchange raw download");
    check_cuda(cudaMemcpyAsync(orthogonal, d.out_orthogonal.get(), bytes, cudaMemcpyDeviceToHost, d.stream),
               "FDDS exchange Q download");
    check_cuda(cudaStreamSynchronize(d.stream), "FDDS exchange outer completion");
    d.outer_count = 0;
    guard.complete = true;
}

size_t CuestFDDSContractions::capacity(size_t naux, size_t requested_rows, bool hybrid) {
    if (!naux || !requested_rows || naux > std::numeric_limits<int>::max()) return 0;
    const size_t fixed = fdds_product(fdds_product(naux, naux), hybrid ? 5 : 1);
    const size_t per_row = fdds_product(naux, hybrid ? 5 : 1);
    cuest_common::ensure_cuest_initialized();
    size_t free = 0, total = 0;
    check_cuda(cudaMemGetInfo(&free, &total), "FDDS device memory preflight");
    const size_t budget = (free / sizeof(double) / 5) * 4;
    if (fixed >= budget) return 0;
    return std::min({requested_rows, (budget - fixed) / per_row,
                     static_cast<size_t>(std::numeric_limits<int>::max())});
}

struct CuestFDDSContractions::Impl {
    size_t q, max_rows;
    bool is_hybrid;
    cudaStream_t stream = nullptr;
    std::vector<DeviceBuffer> inputs, sums;
    Impl(size_t naux, size_t rows, bool hybrid) : q(naux), max_rows(rows), is_hybrid(hybrid) {
        if (!q || !rows || q > std::numeric_limits<int>::max() || rows > std::numeric_limits<int>::max())
            throw PSIEXCEPTION("cuEST FDDS: nonempty BLAS-int dimensions required");
        cuest_common::ensure_cuest_initialized();
        // cuEST also receives this shared handle. Our alpha/beta addresses are
        // host pointers: check the actual mode after any prior J/K use, without
        // changing handle ownership/state or silently retrying on the CPU.
        cublasPointerMode_t pointer_mode;
        check_cublas(cublasGetPointerMode(cublas_handle, &pointer_mode), "FDDS cublasGetPointerMode");
        if (pointer_mode != CUBLAS_POINTER_MODE_HOST)
            throw PSIEXCEPTION("cuEST FDDS: shared cuBLAS handle requires HOST pointer mode.");
        check_cublas(cublasGetStream(cublas_handle, &stream), "FDDS cublasGetStream");
        const size_t ni = hybrid ? 5 : 1;
        inputs.reserve(ni);
        sums.reserve(ni);
        for (size_t i = 0; i < ni; ++i) {
            inputs.emplace_back(fdds_product(rows, q), "FDDS block");
            sums.emplace_back(fdds_product(q, q), "FDDS accumulator");
            check_cuda(cudaMemsetAsync(sums.back().get(), 0, fdds_product(fdds_product(q, q), sizeof(double)),
                                      stream), "FDDS accumulator zero");
        }
    }
    void upload(const std::vector<const double*>& pointers, size_t rows) {
        if (!rows || rows > max_rows || pointers.size() != inputs.size())
            throw PSIEXCEPTION("cuEST FDDS: invalid input block");
        for (size_t i = 0; i < pointers.size(); ++i) {
            if (!pointers[i]) throw PSIEXCEPTION("cuEST FDDS: null input");
            check_cuda(cudaMemcpyAsync(inputs[i].get(), pointers[i], fdds_product(fdds_product(rows, q), sizeof(double)),
                                       cudaMemcpyHostToDevice, stream), "FDDS block upload");
        }
    }
    void product(size_t out, size_t left, size_t right, size_t rows) {
        const double one = 1.0;
        // Row-major left^T right -> column-major right^T left.
        check_cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, q, q, rows, &one,
                                 inputs[right].get(), q, inputs[left].get(), q, &one, sums[out].get(), q),
                     "FDDS cublasDgemm");
    }
};

CuestFDDSContractions::CuestFDDSContractions(size_t q, size_t rows, bool hybrid)
    : impl_(new Impl(q, rows, hybrid)) {}
CuestFDDSContractions::~CuestFDDSContractions() = default;

void CuestFDDSContractions::uncoupled(const double* ovQ, size_t rows) {
    if (impl_->is_hybrid) throw PSIEXCEPTION("cuEST FDDS: wrong contraction mode");
    try {
        impl_->upload({ovQ}, rows);
        impl_->product(0, 0, 0, rows);
        // Host scratch is reused by DFHelper on the next iteration. Async pageable
        // uploads are not a lifetime guarantee; explicitly drain before returning.
        check_cuda(cudaStreamSynchronize(impl_->stream), "FDDS block synchronization");
    } catch (...) {
        // Earlier uploads may still reference the caller's host scratch.
        // Drain before that scratch can unwind; preserve the original error.
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
}

void CuestFDDSContractions::hybrid(const double* ld, const double* ovQ, const double* qx,
                                  const double* qy, const double* yl, size_t rows) {
    if (!impl_->is_hybrid) throw PSIEXCEPTION("cuEST FDDS: wrong contraction mode");
    try {
        impl_->upload({ld, ovQ, qx, qy, yl}, rows);
        impl_->product(0, 0, 1, rows);
        impl_->product(1, 0, 2, rows);
        impl_->product(2, 0, 3, rows);
        impl_->product(3, 1, 4, rows);
        impl_->product(4, 4, 2, rows);
        check_cuda(cudaStreamSynchronize(impl_->stream), "FDDS block synchronization");
    } catch (...) {
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
}

void CuestFDDSContractions::finish(const std::vector<SharedMatrix>& outputs) {
    if (outputs.size() != impl_->sums.size()) throw PSIEXCEPTION("cuEST FDDS: invalid outputs");
    for (const auto& m : outputs)
        if (!m || m->nirrep() != 1 || m->symmetry() || m->rowdim() != impl_->q || m->coldim() != impl_->q)
            throw PSIEXCEPTION("cuEST FDDS: outputs must be dense Q x Q");
    try {
        for (size_t i = 0; i < outputs.size(); ++i)
            check_cuda(cudaMemcpyAsync(outputs[i]->get_pointer(0), impl_->sums[i].get(),
                                       fdds_product(fdds_product(impl_->q, impl_->q), sizeof(double)),
                                       cudaMemcpyDeviceToHost, impl_->stream), "FDDS accumulator download");
        check_cuda(cudaStreamSynchronize(impl_->stream), "FDDS final synchronization");
    } catch (...) {
        // Keep every destination alive even if a later copy failed to enqueue.
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
}

/*! Multiply a list of matrices left to right on the GPU.
 *
 *  Follows ``chain_gemm_einsums``'s transpose and product-scaling conventions:
 *  each input tensor's transpose flag is applied when it enters the chain.
 *  Later left operands are already-oriented intermediate products, so they
 *  are not transposed again. Link i scales its product by ``prefactors_AB[i]``.
 *
 *  \param tensors        the operands, at least two, all with one irrep.
 *  \param transposes     one flag per tensor, "N" or "T".
 *  \param prefactors_AB  one scale factor per link, so ``tensors.size() - 1``.
 *  \param return_tensors one flag per link; the products flagged true are
 *                        copied back, in chain order.
 *
 *  There is deliberately no counterpart to ``chain_gemm_einsums``'s
 *  ``prefactors_C``.  That helper allocates and zeroes its output matrix
 *  immediately before every GEMM, so beta always multiplies zero; carrying the
 *  factor here would only mean uploading a zero matrix to scale it.
 */
std::vector<SharedMatrix> cuest_chain_gemm(const std::vector<SharedMatrix>& tensors,
                                           const std::vector<std::string>& transposes,
                                           const std::vector<double>& prefactors_AB,
                                           const std::vector<bool>& return_tensors) {
    const size_t N = tensors.size();
    if (N < 2) {
        throw PSIEXCEPTION("cuest_chain_gemm: need at least two tensors to multiply");
    }
    if (transposes.size() != N) {
        throw PSIEXCEPTION("cuest_chain_gemm: expected one transpose flag per tensor");
    }
    if (prefactors_AB.size() != N - 1 || return_tensors.size() != N - 1) {
        throw PSIEXCEPTION("cuest_chain_gemm: expected one prefactor and one return flag per link");
    }
    for (size_t i = 0; i < N; ++i) {
        if (!tensors[i]) {
            throw PSIEXCEPTION("cuest_chain_gemm: null tensor in chain");
        }
        if (tensors[i]->nirrep() != 1) {
            throw PSIEXCEPTION(
                "cuest_chain_gemm: only C1 (one-irrep) matrices are supported.\n"
                "  Callers must use native BLAS for symmetry-blocked products.");
        }
    }

    // Walk the chain once on the host to fix every shape before touching the
    // GPU.  Two things come out of this: the caller learns about a
    // non-conformable chain without a device allocation having happened, and
    // the peak intermediate size is known, which is what lets the loop below
    // run without a single mid-chain cudaFree.  Freeing device memory
    // synchronizes the whole device, so a free per link would serialize the
    // chain against itself and give back much of what the GPU won.
    std::vector<Link> links;
    links.reserve(N - 1);
    {
        Shape a{tensors[0]->rowdim(), tensors[0]->coldim()};
        for (size_t i = 0; i + 1 < N; ++i) {
            // Only the first link may transpose its left operand; every later
            // one is an intermediate that came out of cuBLAS already oriented.
            const std::string t1 = (i == 0) ? transposes[0] : "N";
            const std::string& t2 = transposes[i + 1];
            const Shape b{tensors[i + 1]->rowdim(), tensors[i + 1]->coldim()};

            Link link;
            link.op_a = (i == 0) ? to_cublas_op(transposes[0], 0) : CUBLAS_OP_N;
            link.op_b = to_cublas_op(t2, i + 1);
            link.m = (t1 == "T") ? a.cols : a.rows;
            link.k = (t1 == "T") ? a.rows : a.cols;
            link.n = (t2 == "T") ? b.rows : b.cols;
            link.lda = a.cols;
            link.ldb = b.cols;

            const int k_b = (t2 == "T") ? b.cols : b.rows;
            if (link.k != k_b) {
                std::ostringstream msg;
                msg << "cuest_chain_gemm: link " << i << " is not conformable: (" << link.m << " x " << link.k << ") "
                    << t1 << " times (" << k_b << " x " << link.n << ") " << t2;
                throw PSIEXCEPTION(msg.str());
            }

            links.push_back(link);
            a = Shape{link.m, link.n};
        }
    }

    size_t product_elements = static_cast<size_t>(tensors[0]->rowdim()) * tensors[0]->coldim();
    size_t operand_elements = 0;
    for (size_t i = 0; i + 1 < N; ++i) {
        product_elements = std::max(product_elements, static_cast<size_t>(links[i].m) * links[i].n);
        operand_elements =
            std::max(operand_elements, static_cast<size_t>(tensors[i + 1]->rowdim()) * tensors[i + 1]->coldim());
    }

    cuest_common::ensure_cuest_initialized();

    // Everything runs on cuEST's stream, the same one the J/K builds use, so
    // these GEMMs queue behind the integrals rather than racing them.
    cudaStream_t stream = nullptr;
    check_cublas(cublasGetStream(cublas_handle, &stream), "cublasGetStream");

    // Two product buffers, ping-ponged: one holds the running product, the
    // other receives the next one.  Plus one buffer sized for the largest
    // right-hand operand.  Three allocations for the whole chain.
    DeviceBuffer buffer_even(product_elements, "a chain product");
    DeviceBuffer buffer_odd(product_elements, "a chain product");
    DeviceBuffer buffer_operand(operand_elements, "a chain operand");

    double* d_a = buffer_even.get();
    double* d_c = buffer_odd.get();
    double* d_b = buffer_operand.get();

    check_cuda(cudaMemcpyAsync(d_a, tensors[0]->get_pointer(0),
                               static_cast<size_t>(tensors[0]->rowdim()) * tensors[0]->coldim() * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "upload of the first operand");

    std::vector<SharedMatrix> returned;
    const double beta = 0.0;

    for (size_t i = 0; i + 1 < N; ++i) {
        const Link& link = links[i];
        const SharedMatrix& B = tensors[i + 1];

        check_cuda(cudaMemcpyAsync(d_b, B->get_pointer(0),
                                   static_cast<size_t>(B->rowdim()) * B->coldim() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream),
                   "upload of a chain operand");

        // Psi4 stores row-major, cuBLAS reads column-major, and a row-major
        // buffer read as column-major is its own transpose.  Since
        // (A B)^T = B^T A^T, asking cuBLAS for the column-major product of the
        // operands *in reverse order* -- with the transpose flags unchanged --
        // lands exactly the row-major product in d_c.  No explicit transposes,
        // no repacking; the leading dimension of each operand is simply its
        // stored column count.
        const double alpha = prefactors_AB[i];
        check_cublas(cublasDgemm(cublas_handle, link.op_b, link.op_a, link.n, link.m, link.k, &alpha, d_b, link.ldb,
                                 d_a, link.lda, &beta, d_c, link.n),
                     "cublasDgemm");

        std::swap(d_a, d_c);

        if (return_tensors[i]) {
            auto C = std::make_shared<Matrix>("cuest_chain_gemm", link.m, link.n);
            check_cuda(cudaMemcpyAsync(C->get_pointer(0), d_a,
                                       static_cast<size_t>(link.m) * link.n * sizeof(double), cudaMemcpyDeviceToHost,
                                       stream),
                       "download of a chain product");
            returned.push_back(C);
        }
    }

    // The downloads above are stream-ordered, so the host matrices are not
    // populated -- and the buffers about to be freed are not idle -- until the
    // stream drains.
    check_cuda(cudaStreamSynchronize(stream), "stream synchronization");

    return returned;
}

}  // namespace psi

void export_cuest_gemm(py::module& m) {
    m.def("cuest_chain_gemm", &psi::cuest_chain_gemm, "tensors"_a, "transposes"_a, "prefactors_AB"_a,
          "return_tensors"_a,
          "Multiply a chain of matrices on the GPU with cuBLAS, keeping intermediates device-resident. "
          "Only the links flagged in return_tensors are copied back to the host.");
    // Narrow internal regression seam: the same resident metric serves
    // differently sized, changing input blocks. Not a new routing option.
    m.def("_cuest_df_metric_fold", [](const psi::SharedMatrix& metric,
                                      const std::vector<psi::SharedMatrix>& inputs,
                                      const std::string& layout, size_t r) {
        using Engine = psi::CuestDFMetric;
        Engine::Layout kind;
        if (layout == "pqQ") kind = Engine::Layout::pqQ;
        else if (layout == "Qpq") kind = Engine::Layout::Qpq;
        else if (layout == "pQq") kind = Engine::Layout::pQq;
        else if (layout == "Qpq_to_pqQ") kind = Engine::Layout::Qpq_to_pqQ;
        else throw PSIEXCEPTION("cuEST DF metric: unsupported layout");
        if (!metric || metric->nirrep() != 1 || metric->symmetry() != 0 ||
            metric->rowdim() != metric->coldim() || !r)
            throw PSIEXCEPTION("cuEST DF metric: square one-irrep metric and positive r required");
        const size_t q = metric->rowdim();
        std::vector<size_t> ps;
        std::vector<psi::SharedMatrix> result;
        size_t requested = 0;
        for (const auto& input : inputs) {
            if (!input || input->nirrep() != 1 || input->symmetry() != 0)
                throw PSIEXCEPTION("cuEST DF metric: one-irrep input required");
            const size_t rows = input->rowdim(), cols = input->coldim();
            size_t p = 0;
            if (kind == Engine::Layout::pqQ && cols == q && rows % r == 0) p = rows / r;
            else if ((kind == Engine::Layout::Qpq || kind == Engine::Layout::Qpq_to_pqQ) &&
                     rows == q && cols % r == 0) p = cols / r;
            else if (kind == Engine::Layout::pQq && cols == r && q && rows % q == 0) p = rows / q;
            else throw PSIEXCEPTION("cuEST DF metric: incompatible input shape");
            const size_t count = psi::fdds_product(rows, cols);
            requested = std::max(requested, count);
            ps.push_back(p);
            result.push_back(std::make_shared<psi::Matrix>("DF metric regression",
                kind == Engine::Layout::Qpq_to_pqQ ? cols : rows,
                kind == Engine::Layout::Qpq_to_pqQ ? rows : cols));
        }
        if (!requested) return result;  // no CUDA initialization for empty inputs
        if (Engine::capacity(q, requested) < requested)
            throw PSIEXCEPTION("cuEST DF metric: regression blocks exceed device capacity");
        Engine engine(q, requested, metric->get_pointer(0), 0);
        for (size_t i = 0; i < inputs.size(); ++i)
            if (ps[i]) engine.fold(kind, inputs[i]->get_pointer(0), result[i]->get_pointer(0), ps[i], r);
        return result;
    }, "metric"_a, "inputs"_a, "layout"_a, "r"_a,
       "Internal FP64 DF metric block regression seam; explicit GPU execution, no CPU retry.");
}

#endif  // USING_cuEST
