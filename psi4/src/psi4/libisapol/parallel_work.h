/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_PARALLEL_WORK_H
#define PSI4_LIBISAPOL_PARALLEL_WORK_H
#include <algorithm>
#include <exception>
#include <vector>
#include "psi4/libpsi4util/process.h"
#ifdef _OPENMP
#include <omp.h>
#endif
namespace psi { namespace isapol { namespace detail {
// Disjoint outputs only. Each output retains its serial arithmetic order. No
// ambient OpenMP settings are changed and nested invocations stay serial.
// Retain the exception at the lowest failing logical index, not the first
// racing worker. Slots are allocated before entering the region.
template<class F> void parallel_work(size_t count, size_t threshold, F work) {
    int threads = 1;
#ifdef _OPENMP
    if (count >= threshold && !omp_in_parallel())
        threads = static_cast<int>(std::min(count,static_cast<size_t>(std::max(1, Process::environment.get_n_threads()))));
#endif
    if (threads == 1) {
        for (size_t i = 0; i < count; ++i) work(i);
        return;
    }
    std::vector<std::exception_ptr> errors(threads);
    std::vector<size_t> failed(threads,count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (size_t i = 0; i < count; ++i) {
        int worker = 0;
#ifdef _OPENMP
        worker = omp_get_thread_num();
#endif
        try { work(i); }
        catch (...) {
            if (i < failed[worker]) {
                failed[worker] = i;
                errors[worker] = std::current_exception();
            }
        }
    }
    const auto first = std::min_element(failed.begin(),failed.end());
    if (*first != count) std::rethrow_exception(errors[first-failed.begin()]);
}
} } }
#endif
