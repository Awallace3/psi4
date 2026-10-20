#ifndef PSI4_CUEST_GEMM_H
#define PSI4_CUEST_GEMM_H

#include <string>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "psi4/libmints/typedefs.h"

namespace psi {
#ifdef USING_cuEST
/// Left-associated FP64 cuBLAS chain on the shared cuEST stream.
/// Only requested intermediate products are downloaded to host matrices.
std::vector<SharedMatrix> cuest_chain_gemm(const std::vector<SharedMatrix>& tensors,
                                         const std::vector<std::string>& transposes,
                                         const std::vector<double>& prefactors_AB,
                                         const std::vector<bool>& return_tensors);

/// Blocked FDDS A^T B sums. Inputs are contiguous row-major (ov,Q).
/// Outputs stay device-resident until finish(); no additional host matrices.
class CuestFDDSContractions {
   public:
    /// Capacity in rows under 80% of currently free device memory. Zero means
    /// the fixed outputs plus one row cannot fit; callers use native BLAS.
    /// CUDA/runtime failures throw, rather than masquerading as lack of space.
    static size_t capacity(size_t naux, size_t requested_rows, bool hybrid);
    CuestFDDSContractions(size_t naux, size_t max_rows, bool hybrid);
    ~CuestFDDSContractions();
    CuestFDDSContractions(const CuestFDDSContractions&) = delete;
    CuestFDDSContractions& operator=(const CuestFDDSContractions&) = delete;
    void uncoupled(const double* weighted_ovQ, size_t rows);
    void hybrid(const double* ovQLD, const double* ovQ, const double* QX,
                const double* QY, const double* YL, size_t rows);
    /// Order: amp, K1LD, K2LD, K2L, K21L (hybrid); amp (uncoupled).
    void finish(const std::vector<SharedMatrix>& outputs);
   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Hybrid FDDS exchange preprocessing, with two outputs resident across all
/// inner blocks. Each pair forms V = inner(v,Q) outer(k,Q)^T, then accumulates
/// V raw(k,Q) and V orthogonal(k,Q). k=v for X; k=nocc for Y.
class CuestFDDSExchange {
   public:
    /// Reduce, never enlarge, caller-budgeted outer/inner block sizes to fit
    /// 80% of free device memory. {0,0} means native execution before enqueue.
    static std::pair<size_t, size_t> capacity(size_t q, size_t v, size_t k,
                                            size_t outer, size_t inner);
    CuestFDDSExchange(size_t q, size_t v, size_t k, size_t outer, size_t inner);
    ~CuestFDDSExchange();
    CuestFDDSExchange(const CuestFDDSExchange&) = delete;
    CuestFDDSExchange& operator=(const CuestFDDSExchange&) = delete;
    void begin(const double* outer_integrals, size_t count);
    /// Drains the stream before returning so caller-owned DFHelper scratch
    /// can safely be refilled for the next inner block.
    void accumulate(const double* inner_integrals, const double* raw,
                    const double* orthogonal, size_t count);
    void finish(double* raw_output, double* orthogonal_output);
   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif
}  // namespace psi
#endif
