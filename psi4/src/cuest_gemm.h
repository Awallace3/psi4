#ifndef PSI4_CUEST_GEMM_H
#define PSI4_CUEST_GEMM_H

#include <string>
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
#endif
}  // namespace psi
#endif
