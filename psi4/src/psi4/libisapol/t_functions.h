/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_T_FUNCTIONS_H
#define PSI4_LIBISAPOL_T_FUNCTIONS_H
#include <array>
#include <vector>
namespace psi {
namespace isapol {
/// Irregular solid harmonics r^(-k-1) C_kq for every rank k in [0,rank], real
/// Racah normalization with no Condon-Shortley phase, ordered 00,10,11c,11s,
/// 20,21c,21s,22c,22s,... exactly as CamCASP's solidh leaves them: R(k,0) at
/// k^2, R(k,mc)/R(k,ms) at k^2+2m-1/k^2+2m (zero-based).  rank is in [0,4].
/// These are singular at the origin, so the displacement must be nonzero;
/// unlike the regular harmonics, rank 0 is 1/r and not 1.
std::vector<double> isa_irregular_solid_harmonics(int rank, const std::array<double,3>& displacement);
/// Tang-Toennies damping factor for the whole rank-k block at reduced distance
/// br = damping * r: 1 - exp(-br) * sum_{n=0}^{k+1} br^n / n!.  br must be
/// finite and nonnegative.  Not exercised by the reference protocol, which
/// leaves CamCASP's Damping keyword at zero.
double isa_t_function_damping(int rank, double br);
/// One row of CamCASP pfit's T matrix: the interaction functions between a unit
/// charge at point and the multipole components of site, in the site's LOCAL
/// axes.  frame maps local Cartesian coordinates to global ones (its columns
/// are the local axes in the global frame), the same convention as
/// isa_multipole_rotation; it must be proper orthogonal to 1e-12.  damping is
/// CamCASP's Damping keyword in bohr^-1, and zero disables it.  Component order
/// is that of isa_irregular_solid_harmonics.
std::vector<double> isa_t_functions(int rank, const std::array<double,3>& point_bohr,
                                    const std::array<double,3>& site_bohr,
                                    const std::array<std::array<double,3>,3>& frame,
                                    double damping = 0.0);
} }
#endif
