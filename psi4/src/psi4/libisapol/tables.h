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
 */

#ifndef PSI4_LIBISAPOL_TABLES_H
#define PSI4_LIBISAPOL_TABLES_H

#include <string>

namespace psi {
namespace isapol {

/// Highest atomic number carried by the CamCASP element table (src/atoms.f90).
constexpr int kMaxElementZ = 83;

/// Bohr radius used by CamCASP (src/parameters.f90:114, `a_o`).
///
/// Deliberately NOT Psi4's pc_bohr2angstroms (0.52917721067): the eighth-digit
/// difference moves grid points by ~1e-9 and breaks bit-parity with CamCASP.
constexpr double kCamcaspBohr = 0.529177249;

/// One row of CamCASP's `AtomProp` table (src/atoms.f90:38-121).
///
/// Radii are stored in the units atoms.f90 enters them in and converted by the
/// accessors below.
///
/// The fields are `float` on purpose: CamCASP's bare Fortran literals are default
/// (single-precision) real, widened only on assignment, so it integrates with
/// float32-rounded values (R_Slater(O) = 0.60000002384185791 A). Using double
/// would move grid radii by ~4e-8 relative.
struct ElementData {
    const char* symbol;
    float mass;            ///< amu (CRC Handbook, 72nd ed.)
    float rvdw_bondi;      ///< bohr; Bondi, J Phys Chem (1964) 68, 441
    float rslater_ang;     ///< angstrom; see slater_radius()
    float rvdw_grimme_ang; ///< angstrom; Grimme, J Comput Chem (2006) 27, 1787
    float c6_grimme_jnm6;  ///< J nm^6 / mol; Grimme, ibid.
    float covalent_ang;    ///< angstrom; WebElements
};

/// Row `Z` of the table.  Throws for Z outside [0, kMaxElementZ].
const ElementData& element_data(int Z);

/// Bragg-Slater radius in bohr, as CamCASP uses it: Slater, JCP (1964) 41, 3199,
/// with three documented departures (atoms.f90:26-30) --
///   * hydrogen's radius is *twice* Slater's value (0.50 A, not 0.25),
///   * the inert gases take the radius of the preceding halogen,
///   * the dummy site (Z = 0) is given a non-zero radius, 0.65 A.
/// This is the grid's radial scale `alpha`, so it must match CamCASP exactly; it
/// differs from Psi4's GetBSRadius() for H, He, Ne and (by rounding) Ti, Cr, Mn, Fe.
/// Throws for Z >= 55, where CamCASP tabulates zero.
double slater_radius(int Z);

/// Bondi van der Waals radius in bohr, from `AtomProp` (already in bohr there).
///
/// The float32-rounded copy. Use only where CamCASP reads `AtomProp(Z)%RvdwBondi`.
double vdw_radius_bondi(int Z);

/// Grimme van der Waals radius in bohr.
double vdw_radius_grimme(int Z);

/// Grimme atomic C6 in hartree bohr^6.
double c6_grimme(int Z);

/// Covalent radius in bohr.
double covalent_radius(int Z);

/// Element symbol, e.g. "O".  Z = 0 gives "DU".
std::string element_symbol(int Z);

}  // namespace isapol
}  // namespace psi

#endif
