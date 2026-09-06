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

#ifndef PSI4_LIBISAPOL_RECOUPLING_TABLES_H
#define PSI4_LIBISAPOL_RECOUPLING_TABLES_H

#include <string>

namespace psi {
namespace isapol {

/// Lowest and highest dispersion order tabulated (CamCASP ships c6code.f90 ... c12code.f90).
constexpr int kMinDispersionOrder = 6;
constexpr int kMaxDispersionOrder = 12;

/// Highest multipole rank of a distributed polarizability that appears, `la` in
/// alpha^A_{LK(la la')}.  C_12 needs alpha(44).
constexpr int kMaxPolarizabilityRank = 4;

/// Highest rank L of a dispersion coefficient: two rank-4 polarizabilities couple
/// to at most rank 8.
constexpr int kMaxDispersionRank = 8;

/// Number of real spherical-tensor components up to rank kMaxDispersionRank, i.e.
/// the length of CamCASP's `label` array (casimir.f90:112-123).
constexpr int kNumSphericalComponents = (kMaxDispersionRank + 1) * (kMaxDispersionRank + 1);

/// One term of one dispersion coefficient.
///
/// CamCASP writes each anisotropic dispersion coefficient as a sum of
/// Casimir-Polder integrals over pairs of recoupled distributed polarizabilities
/// (casimir.f90:397-435, and SPEC.md 9.3):
///
///     C_n(t, u, J) = sum_terms  coefficient * i^ipow
///                              * (1/2pi) int alpha^A_{t(la la')}(iw)
///                                            alpha^B_{u(lb lb')}(iw) dw
///
/// where `t` is a real spherical-tensor component of rank L1 and `u` one of rank
/// L2.  The coefficient depends only on (n, L1, L2, J) and on which term this is;
/// it is the same for every component `t` of rank L1 and every `u` of rank L2,
/// which is why the tables are indexed by rank rather than by component.
///
/// The coefficient is an exact root-rational fraction, (p/q) sqrt(r/s), with the
/// sign carried in `p`.  It is stored that way rather than as a double so that
/// `coefficient()` can reproduce CamCASP's own floating-point expression exactly:
/// the generated Fortran evaluates `(p d0 / q d0) * sqrt(r d0 / s d0) * cpint(...)`
/// left to right, and so does this.
///
/// `ipow` is the power of i = sqrt(-1) to fold in, 0 or 1.  Individual terms may be
/// imaginary; the block sum is always real, and CamCASP warns if it is not.
struct RecouplingTerm {
    int p;     ///< numerator of the rational factor; carries the sign
    int q;     ///< denominator of the rational factor
    int r;     ///< numerator under the square root
    int s;     ///< denominator under the square root
    int la;    ///< first rank label of alpha^A
    int lap;   ///< second rank label of alpha^A
    int lb;    ///< first rank label of alpha^B
    int lbp;   ///< second rank label of alpha^B
    int ipow;  ///< power of i, 0 or 1

    /// (p/q) sqrt(r/s), in CamCASP's association.
    double coefficient() const;
};

/// The terms of one (n, L1, L2, J) block, as a non-owning range.
class RecouplingBlock {
   public:
    RecouplingBlock() = default;
    RecouplingBlock(const RecouplingTerm* first, int nterm) : first_(first), nterm_(nterm) {}

    int size() const { return nterm_; }
    bool empty() const { return nterm_ == 0; }
    const RecouplingTerm* begin() const { return first_; }
    const RecouplingTerm* end() const { return first_ + nterm_; }
    const RecouplingTerm& operator[](int i) const;

   private:
    const RecouplingTerm* first_ = nullptr;
    int nterm_ = 0;
};

/// Terms contributing to C_n(t, u, J) for any component `t` of rank L1 and `u` of
/// rank L2.  Most (L1, L2, J) combinations vanish -- of the 393 blocks CamCASP
/// tabulates, none is empty, but the great majority of triangles are -- so an
/// empty block is the normal answer, not an error.  Throws only if `n` is outside
/// [kMinDispersionOrder, kMaxDispersionOrder] or a rank is negative.
RecouplingBlock recoupling_block(int n, int L1, int L2, int J);

/// Total number of tabulated blocks and terms, for tests.
int num_recoupling_blocks();
int num_recoupling_terms();

/// The `i`th tabulated block, 0 <= i < num_recoupling_blocks(), in increasing
/// (n, L1, L2, J) order.  Writes the key through the out parameters.
RecouplingBlock recoupling_block_at(int i, int* n, int* L1, int* L2, int* J);

/// Rank L of real spherical-tensor component `t`, 1 <= t <= kNumSphericalComponents.
int component_rank(int t);

/// First and last component index of rank L: 1 and 1 for L = 0, 2 and 4 for L = 1,
/// 5 and 9 for L = 2, and so on.
int component_first(int L);
int component_last(int L);

/// CamCASP's label for component `t` -- "00", "10", "11c", "11s", "20", ...
/// (casimir.f90:112-123), without the trailing blank CamCASP pads to width 3.
std::string component_label(int t);

}  // namespace isapol
}  // namespace psi

#endif
