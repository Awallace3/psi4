#ifndef PSI4_CPSCF_GEMM_H
#define PSI4_CPSCF_GEMM_H

#include <algorithm>
#include <vector>
#include "psi4/libmints/matrix.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libqt/qt.h"
#ifdef USING_cuEST
#include "cuest_gemm.h"
#endif

namespace psi {
namespace scf {
namespace cpscf {

/// Keep small, empty, and symmetry-blocked response products on native BLAS.
/// Match the SAPT chain threshold: the largest link's m*n*k >= min_dim^3.
inline bool use_gpu(Options& options, const std::vector<SharedMatrix>& operands,
                    const std::vector<bool>& transpose) {
#ifdef USING_cuEST
    if (!options.get_bool("USE_CUEST")) return false;
    const int min_dim = options.get_int("CUEST_GEMM_MIN_DIM");
    if (min_dim < 0) throw PSIEXCEPTION("CUEST_GEMM_MIN_DIM must be nonnegative.");
    for (const auto& m : operands) {
        if (m->nirrep() != 1 || m->symmetry() != 0 || !m->rowdim() || !m->coldim()) return false;
    }
    int rows = transpose[0] ? operands[0]->coldim() : operands[0]->rowdim();
    int cols = transpose[0] ? operands[0]->rowdim() : operands[0]->coldim();
    long double work = 0;
    for (size_t i = 1; i < operands.size(); ++i) {
        const int inner = transpose[i] ? operands[i]->coldim() : operands[i]->rowdim();
        const int next = transpose[i] ? operands[i]->rowdim() : operands[i]->coldim();
        if (cols != inner) return false;  // Native linalg reports invalid shapes.
        work = std::max(work, static_cast<long double>(rows) * cols * next);
        cols = next;
    }
    return work >= static_cast<long double>(min_dim) * min_dim * min_dim;
#else
    return false;
#endif
}

#ifdef USING_cuEST
inline SharedMatrix gpu_chain(const std::vector<SharedMatrix>& operands, const std::vector<bool>& transpose) {
    struct Timer {
        Timer() { timer_on("cuEST CPSCF: cuBLAS"); }
        ~Timer() { timer_off("cuEST CPSCF: cuBLAS"); }
    } timer;
    std::vector<std::string> flags;
    for (bool t : transpose) flags.push_back(t ? "T" : "N");
    std::vector<bool> returned(operands.size() - 1, false);
    returned.back() = true;
    return cuest_chain_gemm(operands, flags, std::vector<double>(returned.size(), 1.0), returned).back();
}
#endif

inline SharedMatrix doublet(Options& options, SharedMatrix a, SharedMatrix b, bool ta, bool tb) {
#ifdef USING_cuEST
    if (use_gpu(options, {a, b}, {ta, tb})) return gpu_chain({a, b}, {ta, tb});
#endif
    return linalg::doublet(a, b, ta, tb);
}

inline SharedMatrix triplet(Options& options, SharedMatrix a, SharedMatrix b, SharedMatrix c,
                            bool ta, bool tb, bool tc) {
#ifdef USING_cuEST
    if (use_gpu(options, {a, b, c}, {ta, tb, tc})) return gpu_chain({a, b, c}, {ta, tb, tc});
#endif
    return linalg::triplet(a, b, c, ta, tb, tc);
}

/// Co^T F Co X - X Cv^T F Cv. Retain each four-factor intermediate on device.
inline SharedMatrix onel(Options& options, SharedMatrix co, SharedMatrix cv, SharedMatrix f, SharedMatrix x) {
#ifdef USING_cuEST
    if (use_gpu(options, {co, f, co, x}, {true, false, false, false}) &&
        use_gpu(options, {x, cv, f, cv}, {false, true, false, false})) {
        auto result = gpu_chain({co, f, co, x}, {true, false, false, false});
        result->subtract(gpu_chain({x, cv, f, cv}, {false, true, false, false}));
        return result;
    }
#endif
    auto occupied = linalg::triplet(co, f, co, true, false, false);
    auto result = linalg::doublet(occupied, x, false, false);
    auto virtuals = linalg::triplet(x, cv, f, false, true, false);
    result->gemm(false, false, -1.0, virtuals, cv, 1.0);
    return result;
}

}  // namespace cpscf
}  // namespace scf
}  // namespace psi
#endif
