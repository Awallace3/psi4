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

#ifndef PSI4_LIBISAPOL_TABLES_H
#define PSI4_LIBISAPOL_TABLES_H

#include <string>

namespace psi {
namespace isapol {

/// Highest atomic number carried by the CamCASP element table (src/atoms.f90).
constexpr int kMaxElementZ = 83;

/// Bohr radius used by CamCASP (src/parameters.f90:114, `a_o`).
///
/// This is deliberately NOT Psi4's pc_bohr2angstroms (0.52917721067).  The two
/// differ in the eighth digit, which is enough to shift integration-grid points
/// at the 1e-9 level and break bit-parity with CamCASP.  Do not "modernize" it.
constexpr double kCamcaspBohr = 0.529177249;

/// One row of CamCASP's `AtomProp` table (src/atoms.f90:38-121).
///
/// Radii are stored here in the units CamCASP *enters* them in, and converted on
/// access by the accessors below, so that the numbers can be diffed against
/// atoms.f90 by eye.
///
/// The numeric fields are `float`, not `double`, and that is not an oversight.
/// CamCASP writes the table as bare Fortran literals -- `element('O ',...,0.60,...)`
/// -- which are *default real*, i.e. single precision, and are only widened to
/// real(dp) on assignment into the real(dp) components.  CamCASP is built without
/// -fdefault-real-8, so the values it actually integrates with carry float32
/// rounding: R_Slater(O) is 0.60000002384185791 A, not 0.60.  That shifts grid
/// radii by ~4e-8 relative, which is 8 orders of magnitude above the parity
/// target.  Storing float here reproduces the rounding exactly, by construction.
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
/// This is the radial scale factor `alpha` of the integration grid (SPEC.md 3.5.3),
/// so it must agree with CamCASP exactly; it differs from Psi4's GetBSRadius() for
/// H, He, Ne and (by rounding) Ti, Cr, Mn, Fe.
///
/// CamCASP tabulates zero for Z >= 55, so this throws there rather than handing
/// back a radius of zero for the caller to divide by.
double slater_radius(int Z);

/// Bondi van der Waals radius in bohr, from `AtomProp` (already in bohr there).
///
/// NOT the same numbers as vdw_radius() below, despite both citing Bondi 1964.
/// CamCASP carries two copies of this table and this one, in `AtomProp`, is
/// written as default-real literals and so is rounded to float32.  Use this one
/// only where CamCASP reads `AtomProp(Z)%RvdwBondi`.
double vdw_radius_bondi(int Z);

/// Fallback van der Waals radius for elements MODULE radii does not tabulate
/// (atoms.f90:190, `vdwdef`), in bohr.
constexpr double kVdwRadiusDefault = 2.5;

/// Highest atomic number in MODULE radii's van der Waals table (atoms.f90:193).
constexpr int kMaxVdwRadiusZ = 82;

/// Bondi van der Waals radius in bohr, from `MODULE radii` (atoms.f90:191-211).
///
/// This is the *double precision* copy of the table -- CamCASP writes these
/// literals with an explicit `d0`, so unlike vdw_radius_bondi() above they carry
/// no float32 rounding, and the two differ by ~1e-8 relative.  Elements CamCASP
/// leaves out, and every Z outside [0, kMaxVdwRadiusZ], get kVdwRadiusDefault;
/// nothing throws, because CamCASP's table is indexed unconditionally.
///
/// This is the table the fit-point lattice uses (lattice.F90:6, `use radii, only
/// : vdw_radius`), so fit_points.cc must call this and not vdw_radius_bondi().
/// At 2000 points and a rejection threshold this shifts by ~1e-8, the difference
/// is enough to flip an accept/reject decision and break bit-parity.
double vdw_radius(int Z);

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
