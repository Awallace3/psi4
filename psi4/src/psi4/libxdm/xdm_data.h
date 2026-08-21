/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2024 The Psi4 Developers.
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

#ifndef PSI4_SRC_PSI4_LIBXDM_XDM_DATA_H
#define PSI4_SRC_PSI4_LIBXDM_XDM_DATA_H

#include <string>

namespace psi {
namespace xdm {

/// Maximum atomic number supported
static constexpr int MAX_Z = 103;

/// Number of radial grid points in the proatom density table per element
static constexpr int FTOT_NRAD = 1401;  // 0..1400

/// Number of elements in the proatom density table
static constexpr int FTOT_NELEM = 94;  // Z=1..94

/// Free atomic polarizabilities (atomic units), indexed 0..102.
/// From CRC Handbook of Chemistry and Physics, 88th Ed.
/// Original in AA^3, converted to a.u. by dividing by (0.52917720859)^3.
/// Index 0 is dummy (0.0).
extern const double frepol[MAX_Z];

/// DKH LSDA/UGBS free atomic volumes, indexed 0..103.
/// Used as fallback for elements beyond period-specific tables.
extern const double frevol0[MAX_Z + 1];

/// Free atomic volumes for specific functionals, indexed 0..36 (H through Kr).
/// For elements Z > 36 (or Z=19,20 for some), falls back to frevol0.
extern const double frevol_blyp[37];
extern const double frevol_b3lyp[37];
extern const double frevol_bhahlyp[37];
extern const double frevol_camb3lyp[37];
extern const double frevol_pbe[37];
extern const double frevol_pbe0[37];
extern const double frevol_lcwpbe[37];
extern const double frevol_pw86[37];
extern const double frevol_b971[37];

/// Free atomic volumes at 5 HF fractions (0%, 25%, 50%, 75%, 100%)
/// for periods 1 and 2 (Z=0..10). frevol1[hf_idx][z], hf_idx 0..4.
extern const double frevol1[5][11];

/// Get free-atom volume for element Z with a given functional.
/// @param Z              atomic number
/// @param functional     functional name (lowercase): "blyp", "b3lyp", "pbe", "pbe0",
///                       "bhahlyp", "camb3lyp", "lcwpbe", "pw86pbe", "b971"
/// @param hf_fraction    fraction of HF exchange (used for general hybrids and interpolation)
double get_free_volume(int Z, const std::string& functional, double hf_fraction = -1.0);

/// Get free-atom polarizability for element Z (atomic units).
double get_free_polarizability(int Z);

/// Number of radial data points for element Z in the proatom density table.
int ndata_for_z(int Z);

/// Proatom density table: ftot[j][Z-1] for j=0..1400, Z=1..94.
/// Stores r^2 * rho_atom on a transformed radial grid.
/// Declared in xdm_data_ftot.cc (separate file due to size).
extern const double ftot[FTOT_NRAD][FTOT_NELEM];

}  // namespace xdm
}  // namespace psi

#endif
