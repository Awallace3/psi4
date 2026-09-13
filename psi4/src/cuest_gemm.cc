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
 *  Every GEMM in the SAPT electrostatics and exchange terms goes through one
 *  Python helper, ``chain_gemm_einsums`` in
 *  psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py, which multiplies a list
 *  of matrices left to right.  That helper builds each intermediate as a host
 *  ``core.Matrix``, so a chain of L links costs L round trips through host
 *  memory even though only the last product (or a named few) is wanted.
 *
 *  ``cuest_chain_gemm`` below is the GPU counterpart: it uploads the operands
 *  once, keeps the running product resident on the device across the whole
 *  chain, and copies back only the links the caller asked for.  The J/K builds
 *  that surround these terms are already on the GPU through cuEST, so this
 *  closes the remaining CPU gap in a SAPT(DFT)-D4(I) run.
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
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "psi4/pybind11.h"

#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libfock/cuESTCommon.h"

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

/*! Multiply a list of matrices left to right on the GPU.
 *
 *  Mirrors ``chain_gemm_einsums`` exactly, including its quirks: the transpose
 *  flag of the left operand only applies to the first link, because every later
 *  left operand is an intermediate that is already oriented; and the *i*-th
 *  link scales its product by ``prefactors_AB[i]``.
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
                "  The SAPT tensor code never uses symmetry, so a symmetry-blocked matrix\n"
                "  reaching here means the caller is wrong rather than the GPU path.");
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
}

#endif  // USING_cuEST
