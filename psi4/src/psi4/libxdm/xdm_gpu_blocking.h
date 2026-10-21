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
 * @END LICENSE
 */

#ifndef PSI4_SRC_PSI4_LIBXDM_XDM_GPU_BLOCKING_H
#define PSI4_SRC_PSI4_LIBXDM_XDM_GPU_BLOCKING_H

// The pure parts of XDM_GRID_BLOCKING DEVICE, shared by the device code and the host
// unit test: Morton keys on a 21-bit lattice and the split of a sorted key array into
// blocks along octant boundaries.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __CUDACC__
#define XDM_HD __host__ __device__
#else
#define XDM_HD
#endif

namespace psi {
namespace xdm {
namespace blocking {

constexpr int morton_bits = 21;
constexpr std::uint32_t morton_max = (std::uint32_t{1} << morton_bits) - 1;
// Sieved-out points get this key, which sorts after every 63-bit Morton key.
constexpr std::uint64_t dropped_key = ~std::uint64_t{0};

// The low 21 bits of v, moved to every third bit.
XDM_HD inline std::uint64_t spread_bits(std::uint32_t v) {
    std::uint64_t x = v & morton_max;
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8) & 0x100f00f00f00f00fULL;
    x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2) & 0x1249249249249249ULL;
    return x;
}

XDM_HD inline std::uint64_t morton_key(std::uint32_t qx, std::uint32_t qy, std::uint32_t qz) {
    return spread_bits(qx) | (spread_bits(qy) << 1) | (spread_bits(qz) << 2);
}

// Lattice coordinate of x on [lo, lo + morton_max / scale], clamped.
XDM_HD inline std::uint32_t quantize(double x, double lo, double scale) {
    const double q = (x - lo) * scale;
    if (!(q > 0.0)) return 0;  // also catches NaN
    if (q >= static_cast<double>(morton_max)) return morton_max;
    return static_cast<std::uint32_t>(q);
}

// Cut sorted keys [0, n) into blocks of at most max_points, splitting each range at the
// highest bit on which its keys differ: every block is one cell of the implicit octree
// (or a run of one cell, when more than max_points points share a lattice cell). Blocks
// come out in key order and tile [0, n).
inline void split_sorted_keys(const std::uint64_t* keys, std::size_t n, std::size_t max_points,
                              std::vector<std::size_t>& start, std::vector<std::size_t>& count) {
    start.clear();
    count.clear();
    if (n == 0) return;
    if (max_points == 0) max_points = 1;
    struct Range {
        std::size_t lo, hi;
        int bit;
    };
    // Depth-first with the right half pushed first, so blocks are emitted left to right.
    std::vector<Range> stack{{0, n, 3 * morton_bits - 1}};
    while (!stack.empty()) {
        Range r = stack.back();
        stack.pop_back();
        if (r.hi - r.lo <= max_points) {
            start.push_back(r.lo);
            count.push_back(r.hi - r.lo);
            continue;
        }
        // Keys in a range agree above r.bit, so the ones with r.bit clear come first.
        std::size_t mid = r.lo;
        int bit = r.bit;
        for (; bit >= 0; bit--) {
            const std::uint64_t mask = std::uint64_t{1} << bit;
            mid = static_cast<std::size_t>(
                std::partition_point(keys + r.lo, keys + r.hi, [mask](std::uint64_t k) { return !(k & mask); }) -
                keys);
            if (mid != r.lo && mid != r.hi) break;
        }
        if (bit < 0) {
            for (std::size_t s = r.lo; s < r.hi; s += max_points) {
                start.push_back(s);
                count.push_back(std::min(max_points, r.hi - s));
            }
            continue;
        }
        stack.push_back({mid, r.hi, bit - 1});
        stack.push_back({r.lo, mid, bit - 1});
    }
}

}  // namespace blocking
}  // namespace xdm
}  // namespace psi

#undef XDM_HD
#endif
