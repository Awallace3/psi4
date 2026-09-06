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

#include "tables.h"

#include "psi4/libpsi4util/exception.h"

#include <cmath>

namespace psi {
namespace isapol {

namespace {

/// CamCASP's hartree -> kJ/mol conversion (src/parameters.f90:97).
constexpr double kCamcaspAu2kJ = 2625.49962;

/// Transcribed verbatim from CamCASP src/atoms.f90:38-121.  `vdwdef` (2.5 bohr)
/// has been substituted for the elements that use it.  Columns, in order:
///   symbol, mass/amu, R_vdW(Bondi)/bohr, R_Slater/A, R_vdW(Grimme)/A,
///   C6(Grimme)/(J nm^6 mol^-1), R_covalent/A.
/// Held as float to reproduce Fortran's single-precision literals -- see the
/// ElementData comment in tables.h.
// clang-format off
const ElementData kElements[kMaxElementZ + 1] = {
    {"DU",        0.0,    0.0,  0.65,  0.000,   0.00,   0.0},  //  0 Dummy
    {"H",     1.00794,  2.268,  0.50,  1.001,   0.14,  0.37},  //  1 Hydrogen
    {"He",   4.002602,  2.646,  0.50,  1.012,   0.08,  0.32},  //  2 Helium
    {"Li",      6.941,  3.440,  1.45,  0.825,   1.61,  1.34},  //  3 Lithium
    {"Be",   9.012182,    2.5,  1.05,  1.408,   1.61,  0.90},  //  4 Beryllium
    {"B",      10.811,    2.5,  0.85,  1.485,   3.13,  0.82},  //  5 Boron
    {"C",     12.0107,  3.213,  0.70,  1.452,   1.75,  0.77},  //  6 Carbon
    {"N",     14.0067,  2.929,  0.65,  1.397,   1.23,  0.75},  //  7 Nitrogen
    {"O",     15.9994,  2.872,  0.60,  1.342,   0.70,  0.73},  //  8 Oxygen
    {"F",  18.9984032,  2.778,  0.50,  1.287,   0.75,  0.71},  //  9 Fluorine
    {"Ne",    20.1797,  2.910,  0.50,  1.243,   0.63,  0.69},  // 10 Neon
    {"Na",  22.989770,  4.290,  1.80,  1.144,   5.71,  1.54},  // 11 Sodium
    {"Mg",    24.3050,  3.270,  1.50,  1.364,   5.71,  1.30},  // 12 Magnesium
    {"Al",  26.981538,    2.5,  1.25,  1.639,   10.7,  1.18},  // 13 Aluminium
    {"Si",    28.0855,  3.968,  1.10,  1.716,   9.23,  1.11},  // 14 Silicon
    {"P",   30.973761,  3.402,  1.00,  1.705,   7.84,  1.06},  // 15 Phosphorus
    {"S",      32.065,  3.402,  1.00,  1.683,   5.57,  1.02},  // 16 Sulfur
    {"Cl",     35.453,  3.307,  1.00,  1.639,   5.07,  0.99},  // 17 Chlorine
    {"Ar",     39.948,  3.553,  1.00,  1.595,   4.61,  0.97},  // 18 Argon
    {"K",     39.0983,  5.197,  2.20,  1.485,  10.80,  1.96},  // 19 Potassium
    {"Ca",     40.078,    2.5,  1.80,  1.474,  10.80,  1.74},  // 20 Calcium
    {"Sc",  44.955910,    2.5,  1.60,  1.562,  10.80,  1.44},  // 21 Scandium
    {"Ti",     47.867,    2.5,  1.40,  1.562,  10.80,  1.36},  // 22 Titanium
    {"V",     50.9415,    2.5,  1.35,  1.562,  10.80,  1.25},  // 23 Vanadium
    {"Cr",    51.9961,    2.5,  1.40,  1.562,  10.80,  1.27},  // 24 Chromium
    {"Mn",  54.938049,    2.5,  1.40,  1.562,  10.80,  1.39},  // 25 Manganese
    {"Fe",     55.845,    2.5,  1.40,  1.562,  10.80,  1.25},  // 26 Iron
    {"Co",  58.933200,    2.5,  1.35,  1.562,  10.80,  1.26},  // 27 Cobalt
    {"Ni",    58.6934,  3.080,  1.35,  1.562,  10.80,  1.21},  // 28 Nickel
    {"Cu",     63.546,  2.646,  1.35,  1.562,  10.80,  1.38},  // 29 Copper
    {"Zn",     65.409,  2.627,  1.35,  1.562,  10.80,  1.31},  // 30 Zinc
    {"Ga",     69.723,  3.534,  1.30,  1.650,  16.99,  1.26},  // 31 Gallium
    {"Ge",      72.64,    2.5,  1.25,  1.727,  17.10,  1.22},  // 32 Germanium
    {"As",   74.92160,  3.496,  1.15,  1.760,  16.37,  1.19},  // 33 Arsenic
    {"Se",      78.96,  3.590,  1.15,  1.771,  12.64,  1.16},  // 34 Selenium
    {"Br",     79.904,  3.496,  1.15,  1.749,  12.47,  1.14},  // 35 Bromine
    {"Kr",     83.798,  3.817,  1.15,  1.727,  12.01,  1.10},  // 36 Krypton
    {"Rb",    85.4678,    2.5,  2.35,  1.628,  24.67,  2.11},  // 37 Rubidium
    {"Sr",      87.62,    2.5,  2.00,  1.606,  24.67,  1.92},  // 38 Strontium
    {"Y",    88.90585,    2.5,  1.80,  1.639,  24.67,  1.62},  // 39 Yttrium
    {"Zr",     91.224,    2.5,  1.55,  1.639,  24.67,  1.48},  // 40 Zirconium
    {"Nb",   92.90638,    2.5,  1.45,  1.639,  24.67,  1.37},  // 41 Niobium
    {"Mo",      95.94,    2.5,  1.45,  1.639,  24.67,  1.45},  // 42 Molybdenum
    {"Tc",       98.0,    2.5,  1.35,  1.639,  24.67,  1.56},  // 43 Technetium
    {"Ru",     101.07,    2.5,  1.30,  1.639,  24.67,  1.26},  // 44 Ruthenium
    {"Rh",  102.90550,    2.5,  1.35,  1.639,  24.67,  1.35},  // 45 Rhodium
    {"Pd",     106.42,  3.080,  1.40,  1.639,  24.67,  1.31},  // 46 Palladium
    {"Ag",   107.8682,  3.250,  1.60,  1.639,  24.67,  1.53},  // 47 Silver
    {"Cd",    112.411,  2.986,  1.55,  1.639,  24.67,  1.48},  // 48 Cadmium
    {"In",    114.818,  3.647,  1.55,  1.672,  37.32,  1.44},  // 49 Indium
    {"Sn",    118.710,  4.100,  1.45,  1.804,  38.71,  1.41},  // 50 Tin
    {"Sb",    121.760,    2.5,  1.45,  1.881,  38.44,  1.38},  // 51 Antimony
    {"Te",     127.60,  3.893,  1.40,  1.892,  31.74,  1.35},  // 52 Tellurium
    {"I",   126.90447,  3.742,  1.40,  1.892,  31.50,  1.33},  // 53 Iodine
    {"Xe",    131.293,  4.082,  1.40,  1.881,  29.99,  1.30},  // 54 Xenon
    {"Cs",  132.90545,    2.5,  0.00,  0.000,  00.00,  2.25},  // 55 Caesium
    {"Ba",    137.327,    2.5,  0.00,  0.000,  00.00,  1.98},  // 56 Barium
    {"La",   138.9055,    2.5,  0.00,  0.000,  00.00,  1.69},  // 57 Lanthanum
    {"Ce",    140.116,    2.5,  0.00,  0.000,  00.00,   0.0},  // 58 Cerium
    {"Pr",  140.90765,    2.5,  0.00,  0.000,  00.00,   0.0},  // 59 Praseodymium
    {"Nd",     144.24,    2.5,  0.00,  0.000,  00.00,   0.0},  // 60 Neodymium
    {"Pm",      145.0,    2.5,  0.00,  0.000,  00.00,   0.0},  // 61 Promethium
    {"Sm",     150.36,    2.5,  0.00,  0.000,  00.00,   0.0},  // 62 Samarium
    {"Eu",    151.964,    2.5,  0.00,  0.000,  00.00,   0.0},  // 63 Europium
    {"Gd",     157.25,    2.5,  0.00,  0.000,  00.00,   0.0},  // 64 Gadolinium
    {"Tb",  158.92534,    2.5,  0.00,  0.000,  00.00,   0.0},  // 65 Terbium
    {"Dy",    162.500,    2.5,  0.00,  0.000,  00.00,   0.0},  // 66 Dysprosium
    {"Ho",  164.93032,    2.5,  0.00,  0.000,  00.00,   0.0},  // 67 Holmium
    {"Er",    167.259,    2.5,  0.00,  0.000,  00.00,   0.0},  // 68 Erbium
    {"Tm",  168.93421,    2.5,  0.00,  0.000,  00.00,   0.0},  // 69 Thulium
    {"Yb",     173.04,    2.5,  0.00,  0.000,  00.00,   0.0},  // 70 Ytterbium
    {"Lu",    174.967,    2.5,  0.00,  0.000,  00.00,  1.60},  // 71 Lutetium
    {"Hf",     178.49,    2.5,  0.00,  0.000,  00.00,  1.50},  // 72 Hafnium
    {"Ta",   180.9479,    2.5,  0.00,  0.000,  00.00,  1.38},  // 73 Tantalum
    {"W",      183.84,    2.5,  0.00,  0.000,  00.00,  1.46},  // 74 Tungsten
    {"Re",    186.207,    2.5,  0.00,  0.000,  00.00,  1.59},  // 75 Rhenium
    {"Os",     190.23,    2.5,  0.00,  0.000,  00.00,  1.28},  // 76 Osmium
    {"Ir",    192.217,    2.5,  0.00,  0.000,  00.00,  1.37},  // 77 Iridium
    {"Pt",    195.078,  3.250,  0.00,  0.000,  00.00,  1.28},  // 78 Platinum
    {"Au",  196.96655,  3.137,  0.00,  0.000,  00.00,  1.44},  // 79 Gold
    {"Hg",     200.59,  2.929,  0.00,  0.000,  00.00,  1.49},  // 80 Mercury
    {"Tl",   204.3833,  3.704,  0.00,  0.000,  00.00,  1.48},  // 81 Thallium
    {"Pb",      207.2,  3.817,  0.00,  0.000,  00.00,  1.47},  // 82 Lead
    {"Bi",  208.98038,    2.5,  0.00,  0.000,  00.00,  1.46},  // 83 Bismuth
};
// clang-format on

/// MODULE radii's van der Waals radii in bohr (atoms.f90:191-211), transcribed
/// with CamCASP's own element comments so the two can be diffed by eye.  These
/// are genuine doubles in CamCASP -- note the explicit `d0` on every literal --
/// unlike the AtomProp copy, which is float32.
const double kVdwRadius[kMaxVdwRadiusZ + 1] = {
    0.000,                                  //  dummy site
    2.268, 2.646,                           //  H, He
    3.440, kVdwRadiusDefault, kVdwRadiusDefault, 3.213,     //  Li, Be, B, C
    2.929, 2.872, 2.778, 2.910,             //  N, O, F, Ne
    4.290, 3.270, kVdwRadiusDefault, 3.968,               //  Na, Mg, Al, Si
    3.402, 3.402, 3.307, 3.553,             //  P, S, Cl, Ar
    5.197, kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault,   //  K, Ca, Sc, Ti
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault,     //  V, Cr, Mn, Fe
    kVdwRadiusDefault, 3.080, 2.646, 2.627,               //  Co, Ni, Cu, Zn
    3.534, kVdwRadiusDefault, 3.496, 3.590,               //  Ga, Ge, As, Se
    3.496, 3.817, kVdwRadiusDefault, kVdwRadiusDefault,         //  Br, Kr, Rb, Sr
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault,     //  Y, Zr, Nb, Mo
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, 3.080,           //  Tc, Ru, Rh, Pd
    3.250, 2.986, 3.647, 4.100,             //  Ag, Cd, In, Sn
    kVdwRadiusDefault, 3.893, 3.742, 4.082,               //  Sb, Te, I, Xe
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault,     //  Cs, Ba, La, Ce
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault,     //  Pr, Nd, Pm, Sm
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault,     //  Eu, Gd, Tb, Dy
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault,     //  Ho, Er, Tm, Yb
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault,     //  Lu, Hf, Ta, W
    kVdwRadiusDefault, kVdwRadiusDefault, kVdwRadiusDefault, 3.250,           //  Re, Os, Ir, Pt
    3.137, 2.929, 3.704, 3.817              //  Au, Hg, Tl, Pb
};

const ElementData& row(int Z) {
    if (Z < 0 || Z > kMaxElementZ)
        throw PSIEXCEPTION("libisapol: no element data for Z = " + std::to_string(Z) +
                           "; CamCASP tabulates Z = 0 through " + std::to_string(kMaxElementZ) + ".");
    return kElements[Z];
}

}  // namespace

const ElementData& element_data(int Z) { return row(Z); }

double slater_radius(int Z) {
    const double r_ang = row(Z).rslater_ang;  // float -> double, exactly as Fortran widens it
    if (r_ang == 0.0)
        throw PSIEXCEPTION("libisapol: CamCASP tabulates no Bragg-Slater radius for " + element_symbol(Z) +
                           " (Z = " + std::to_string(Z) +
                           "), so the ISA integration grid cannot be built for it.  The table stops at Cs (Z = 55).");
    return r_ang / kCamcaspBohr;
}

double vdw_radius_bondi(int Z) { return row(Z).rvdw_bondi; }

double vdw_radius(int Z) {
    if (Z < 0 || Z > kMaxVdwRadiusZ) return kVdwRadiusDefault;
    return kVdwRadius[Z];
}

double vdw_radius_grimme(int Z) { return row(Z).rvdw_grimme_ang / kCamcaspBohr; }

double c6_grimme(int Z) {
    // atoms.f90:126-128: J nm^6/mol -> hartree bohr^6.
    const double conversion = std::pow(10.0 / kCamcaspBohr, 6) / (1000.0 * kCamcaspAu2kJ);
    return row(Z).c6_grimme_jnm6 * conversion;
}

double covalent_radius(int Z) { return row(Z).covalent_ang / kCamcaspBohr; }

std::string element_symbol(int Z) { return row(Z).symbol; }

}  // namespace isapol
}  // namespace psi
