/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2024 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */

#ifndef PSI4_SRC_PSI4_LIBXDM_HIRSHFELD_H
#define PSI4_SRC_PSI4_LIBXDM_HIRSHFELD_H

#include <vector>

namespace psi {
namespace xdm {

/// Proatom density evaluator using cubic spline interpolation
/// of tabulated free-atom densities on a transformed radial grid.
///
/// The radial coordinate transform is:
///   q = r / (r + rmid),  rmid = 1 / Z^(1/3)
///   r = rmid * q / (1 - q)
///
/// The table stores f(q) = r^2 * rho_atom(r) at uniformly-spaced q values.
/// The spline gives f(q), and rho_atom(r) = |f(q)| / r^2.
class ProatomDensity {
   public:
    ProatomDensity();

    /// Evaluate the free-atom density at distance r from a nucleus of element Z.
    /// Returns rho_atom(r) in atomic units.
    double evaluate(int Z, double r) const;

   private:
    /// Initialize spline coefficients for element Z (lazy, thread-safe via const + mutable).
    void initialize_spline(int Z) const;

    /// Whether spline has been initialized for each element
    mutable std::vector<bool> spline_initialized_;

    /// Spline coefficients: a[Z-1][j], b[Z-1][j], c[Z-1][j] for j=0..ndata
    mutable std::vector<std::vector<double>> spline_a_;
    mutable std::vector<std::vector<double>> spline_b_;
    mutable std::vector<std::vector<double>> spline_c_;
};

/// Compute Hirshfeld weights for a set of grid points given all atoms.
///
/// For each grid point, computes w_A(r) = rho_A^proatom(r) / sum_B rho_B^proatom(r)
///
/// @param proatom      Initialized ProatomDensity evaluator
/// @param natom        Number of atoms
/// @param atomic_nums  Atomic numbers [natom]
/// @param atom_coords  Atomic coordinates [natom][3] in bohr
/// @param npoints      Number of grid points
/// @param grid_x       Grid x-coordinates [npoints]
/// @param grid_y       Grid y-coordinates [npoints]
/// @param grid_z       Grid z-coordinates [npoints]
/// @param weights      Output: Hirshfeld weights [natom][npoints]
void compute_hirshfeld_weights(const ProatomDensity& proatom, int natom, const int* atomic_nums,
                               const double atom_coords[][3], int npoints, const double* grid_x,
                               const double* grid_y, const double* grid_z,
                               std::vector<std::vector<double>>& weights);

}  // namespace xdm
}  // namespace psi

#endif
