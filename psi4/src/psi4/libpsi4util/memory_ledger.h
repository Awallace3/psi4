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

#ifndef _psi_src_lib_libpsi4util_memory_ledger_h_
#define _psi_src_lib_libpsi4util_memory_ledger_h_

#include <atomic>
#include <cstddef>
#include <cstdint>

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace psi {

/*! \brief Process-wide tally of live integral and grid buffers, in doubles.
 *
 * SCF subtracts these claims before allocating for another live wavefunction.
 * Claims report ownership; they neither allocate nor enforce a memory limit.
 */
class MemoryClaim {
   public:
    MemoryClaim() = default;
    ~MemoryClaim() { set(0); }

    /// Each buffer has one owner and must not be counted twice.
    MemoryClaim(const MemoryClaim&) = delete;
    MemoryClaim& operator=(const MemoryClaim&) = delete;

    /// Report the doubles this holder has resident *now*; pass 0 once the buffer is gone.
    void set(size_t doubles) {
        total().fetch_add(static_cast<std::int64_t>(doubles) - static_cast<std::int64_t>(held_),
                          std::memory_order_relaxed);
        held_ = doubles;
    }

    /// Doubles this claim last reported.
    size_t held() const { return held_; }

    /// Doubles reported by every live claim in the process.
    static size_t committed() {
        const std::int64_t tally = total().load(std::memory_order_relaxed);
        return tally > 0 ? static_cast<size_t>(tally) : 0;
    }

   private:
    size_t held_ = 0;

    static std::atomic<std::int64_t>& total() {
        static std::atomic<std::int64_t> tally{0};
        return tally;
    }
};

/*! \brief Return free glibc arena pages to the operating system.
 *
 * Freeing many small cache matrices need not reduce RSS. Trim after releasing
 * a large store, not inside allocation loops; this is a no-op on other libc implementations.
 */
inline void release_freed_memory() {
#ifdef __GLIBC__
    malloc_trim(0);
#endif
}

}  // namespace psi

#endif  // _psi_src_lib_libpsi4util_memory_ledger_h_
