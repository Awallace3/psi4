/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace psi {
namespace detail {

// Internal planner without tensor allocations; memory quantities are in doubles.
inline std::tuple<size_t, size_t> dfhelper_jk_blocks(
    std::vector<std::pair<size_t, size_t>>& blocks, const std::vector<size_t>& shell_offsets,
    size_t nbf, size_t nthreads, size_t max_nocc, bool lr_symmetric, bool ao_core,
    size_t ao_size, size_t function_pairs, size_t memory) {
    if (shell_offsets.size() < 2) return std::make_tuple(0, 0);
    const size_t nshells = shell_offsets.size() - 1;
    const size_t T1 = nbf * max_nocc;
    const size_t T2 = lr_symmetric ? nbf * nbf : nbf * max_nocc;
    const size_t T3 = std::max(nthreads * nbf * nbf, nthreads * nbf * max_nocc);
    const size_t resident_AO = ao_core ? ao_size : 0;
    size_t total_AO_buffer = resident_AO;
    size_t block_size = 0, largest = 0;
    for (size_t i = 0, tmpbs = 0, count = 1; i < nshells; i++, count++) {
        const size_t current = (shell_offsets[i + 1] - shell_offsets[i]) * function_pairs;
        total_AO_buffer += ao_core ? 0 : current;
        tmpbs += shell_offsets[i + 1] - shell_offsets[i];
        size_t constraint = total_AO_buffer + T1 * tmpbs + T3;
        constraint += lr_symmetric ? T2 : T2 * tmpbs;
        if (constraint > memory || i == nshells - 1) {
            if (count == 1 && constraint > memory) {
                throw std::runtime_error("DFHelper: not enough memory for JK blocking!");
            }
            if (constraint > memory) {
                if (!ao_core) total_AO_buffer -= current;
                tmpbs -= shell_offsets[i + 1] - shell_offsets[i];
                blocks.emplace_back(i - count + 1, i - 1);
                i--;
            } else if (i == nshells - 1) {
                blocks.emplace_back(i - count + 1, i);
            }
            if (block_size < tmpbs) {
                largest = total_AO_buffer;
                block_size = tmpbs;
            }
            count = tmpbs = 0;
            // In-core AOs remain resident between auxiliary blocks.
            total_AO_buffer = resident_AO;
        }
    }
    return std::make_tuple(largest, block_size);
}

}  // namespace detail
}  // namespace psi
