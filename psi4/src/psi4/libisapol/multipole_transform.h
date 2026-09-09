/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_MULTIPOLE_TRANSFORM_H
#define PSI4_LIBISAPOL_MULTIPOLE_TRANSFORM_H
#include <array>
#include <memory>
namespace psi {
class Matrix;
namespace isapol {
/// Independent polynomial identities for real Racah components 00,10,11c,11s,...
/// T(d) R(x) = R(x+d). To move moments from site a to b use d=R_a-R_b (bohr).
/// Includes every rank from zero through rank, with rank in [0,4].
std::shared_ptr<Matrix> isa_multipole_translation(int rank, const std::array<double,3>& displacement);
/// D(F) R(x) = R(F x). F maps local Cartesian coordinates to global coordinates.
/// Require a finite proper orthogonal frame to absolute tolerance 1e-12.
/// Both functions return independent matrices and perform no tensor localization.
std::shared_ptr<Matrix> isa_multipole_rotation(int rank, const std::array<std::array<double,3>,3>& frame);
} }
#endif
