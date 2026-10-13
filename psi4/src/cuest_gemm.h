#ifndef PSI4_CUEST_GEMM_H
#define PSI4_CUEST_GEMM_H

#include <string>
#include <cstddef>
#include <memory>
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
#endif
}  // namespace psi
#endif
