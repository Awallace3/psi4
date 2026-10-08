/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_HARMONIC_TRANSFORM_H
#define PSI4_LIBISAPOL_HARMONIC_TRANSFORM_H
#include <utility>
#include <vector>
namespace psi { namespace isapol {
/// One row per spherical component, ordered sin(l..1), m0, cos(1..l); l=1 is
/// x,y,z. Each entry pairs a Libint standard Cartesian component index with the
/// coefficient of the RAW monomial x^i y^j z^k, so callers must feed components
/// computed with ``CartesianShellNormalization::standard`` and must NOT apply the
/// GAMINT mixed-component factor to a transformed index. Same unnormalized
/// regular-harmonic recurrence, and same scale sqrt((2-d_m0)(l-m)!/(l+m)!), as
/// ``explicit_basis.cc``'s grid evaluation: integrals and samples of a spherical
/// shell therefore share one convention. No Condon--Shortley phase.
using IsaHarmonicTransform = std::vector<std::vector<std::pair<int, double>>>;
IsaHarmonicTransform isa_dalton_transform(int l);
} }
#endif
