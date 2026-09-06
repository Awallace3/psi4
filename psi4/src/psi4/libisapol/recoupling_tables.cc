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

/*
 * libisapol: ISA partitioning, distributed polarizabilities and dispersion
 * coefficients, ported with permission from CamCASP 6.0 by Alston J. Misquitta
 * and Anthony J. Stone (http://gitlab.com/anthonyjstone/camcasp).  Numerical
 * conventions follow the CamCASP source, referenced inline by file and line.
 *
 * See SPEC.md in this directory for the full specification.
 */

#include "recoupling_tables.h"

#include "psi4/libpsi4util/exception.h"

#include <algorithm>
#include <cmath>

namespace psi {
namespace isapol {

namespace {

/// One row of the block index.  Blocks are stored contiguously in
/// kRecouplingTerms, `first` terms in from the start.
struct RecouplingBlockRecord {
    int n, L1, L2, J;
    int first, nterm;
};

#include "recoupling_data.inc"

/// Packing that makes the block index sortable on a single integer.  J runs to
/// 2 * kMaxDispersionRank, hence the stride of 17.
constexpr int kRankStride = kMaxDispersionRank + 1;
constexpr int kJStride = 2 * kMaxDispersionRank + 1;

int block_key(int n, int L1, int L2, int J) {
    return ((n * kRankStride + L1) * kRankStride + L2) * kJStride + J;
}

}  // namespace

double RecouplingTerm::coefficient() const {
    // Written to match the generated Fortran term for term: CamCASP evaluates
    // `(p d0 / q d0) * sqrt(r d0 / s d0) * cpint(...)`, and r == s == 1 whenever
    // the term has no square-root factor, so sqrt(1.0) drops out exactly.
    return (static_cast<double>(p) / static_cast<double>(q)) *
           std::sqrt(static_cast<double>(r) / static_cast<double>(s));
}

const RecouplingTerm& RecouplingBlock::operator[](int i) const {
    if (i < 0 || i >= nterm_) throw PSIEXCEPTION("RecouplingBlock: term index out of range");
    return first_[i];
}

RecouplingBlock recoupling_block(int n, int L1, int L2, int J) {
    if (n < kMinDispersionOrder || n > kMaxDispersionOrder)
        throw PSIEXCEPTION("recoupling_block: dispersion order out of range; CamCASP tabulates C6 to C12");
    if (L1 < 0 || L2 < 0 || J < 0) throw PSIEXCEPTION("recoupling_block: negative rank");
    if (L1 > kMaxDispersionRank || L2 > kMaxDispersionRank || J > 2 * kMaxDispersionRank)
        return RecouplingBlock();

    const int key = block_key(n, L1, L2, J);
    const auto* found = std::lower_bound(
        kRecouplingBlocks, kRecouplingBlocks + kNumRecouplingBlocks, key,
        [](const RecouplingBlockRecord& b, int k) { return block_key(b.n, b.L1, b.L2, b.J) < k; });
    if (found == kRecouplingBlocks + kNumRecouplingBlocks || block_key(found->n, found->L1, found->L2, found->J) != key)
        return RecouplingBlock();
    return RecouplingBlock(kRecouplingTerms + found->first, found->nterm);
}

int num_recoupling_blocks() { return kNumRecouplingBlocks; }

int num_recoupling_terms() { return kNumRecouplingTerms; }

RecouplingBlock recoupling_block_at(int i, int* n, int* L1, int* L2, int* J) {
    if (i < 0 || i >= kNumRecouplingBlocks) throw PSIEXCEPTION("recoupling_block_at: index out of range");
    const RecouplingBlockRecord& b = kRecouplingBlocks[i];
    *n = b.n;
    *L1 = b.L1;
    *L2 = b.L2;
    *J = b.J;
    return RecouplingBlock(kRecouplingTerms + b.first, b.nterm);
}

int component_rank(int t) {
    if (t < 1 || t > kNumSphericalComponents) throw PSIEXCEPTION("component_rank: component index out of range");
    int L = 0;
    while ((L + 1) * (L + 1) < t) ++L;
    return L;
}

int component_first(int L) {
    if (L < 0 || L > kMaxDispersionRank) throw PSIEXCEPTION("component_first: rank out of range");
    return L * L + 1;
}

int component_last(int L) {
    if (L < 0 || L > kMaxDispersionRank) throw PSIEXCEPTION("component_last: rank out of range");
    return (L + 1) * (L + 1);
}

std::string component_label(int t) {
    const int L = component_rank(t);
    const int offset = t - component_first(L);
    std::string label = std::to_string(L);
    if (offset == 0) return label + "0";
    // Components come in cosine/sine pairs, K = 1 ... L, cosine first.
    label += std::to_string((offset + 1) / 2);
    label += (offset % 2 == 1) ? 'c' : 's';
    return label;
}

}  // namespace isapol
}  // namespace psi
