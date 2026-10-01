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

#ifndef PSI4_LIBISAPOL_ISA_GRID_H
#define PSI4_LIBISAPOL_ISA_GRID_H

#include <memory>
#include <string>
#include <vector>

namespace psi {

class Molecule;

namespace isapol {

/// Knobs of the ISA integration grid. Defaults are the CamCASP grid MODULE
/// defaults (80/590), not a method preset's grid (e.g. 100/400 requested,
/// 434 actual angular points). Select the preset explicitly when comparing it.
///
/// Defaults are src/parameters.f90:189-197. Only Lebedev
/// (par_integration_grid_type = 1) is implemented. These are declared model
/// parameters: another value is a different model, not a looser one.
struct IsaGridOptions {
    /// CamCASP's `n_r` (its `NumRadPoints`), *not* the number of shells: the
    /// Euler-MacLaurin map generates shells i = 1 ... n_r - 1, so n_r = 80 gives
    /// 79 shells.
    int radial_points = 80;

    /// Requested Lebedev order, rounded *up* to the next tabulated size as
    /// Lbdv() does (atom_grids.F90:509): `Angular 400` gives 434.
    int spherical_points = 590;

    /// Becke smoothing iterations, `k_mu` (atom_grids.F90:20).
    int becke_smoothing = 3;

    /// `rscale`, multiplying the tabulated Bragg-Slater radius
    /// (num_integration_grid.F90:433) that sets the radial scale.
    double radius_scaling = 1.0;
};

/// The molecular integration grid used by the ISA partition.
///
/// Re-implements CamCASP's make_grid (src/gdma/atom_grids.F90) instead of using
/// DFTGrid. The Lebedev tables, Euler-MacLaurin map and Becke partition agree;
/// DFTGrid differs in three ways it cannot disable:
///
///   1. DFTGrid *rotates* each atom's Lebedev sphere into a standard orientation
///      (cubature.cc, OrientationMgr::MoveIntoPosition).  CamCASP only translates.
///   2. DFTGrid clamps the Becke size-adjustment parameter `a` to [-1/2, 1/2].
///      CamCASP does not clamp.
///   3. DFTGrid discards points below DFT_WEIGHTS_TOLERANCE.  CamCASP keeps all
///      of them (its suppression block is commented out).
///
/// Each moves integrals at the 1e-8...1e-10 level this port is held to. The
/// Lebedev tables are Psi4's (lebedev_sphere()), bit-identical to CamCASP's on
/// the verified non-FMA build.
class IsaGrid {
   public:
    IsaGrid(std::shared_ptr<Molecule> molecule, const IsaGridOptions& options);

    int natom() const { return natom_; }
    int npoints() const { return static_cast<int>(x_.size()); }

    /// Number of Lebedev points actually used, after rounding up.
    int spherical_points() const { return spherical_points_; }
    int radial_points() const { return options_.radial_points; }
    const IsaGridOptions& options() const { return options_; }

    const double* x() const { return x_.data(); }
    const double* y() const { return y_.data(); }
    const double* z() const { return z_.data(); }
    const double* w() const { return w_.data(); }

    /// Index of the first point belonging to atom A.  Points are laid out
    /// atom-major, then radial shell, then angular point -- CamCASP's order.
    int atom_start(int A) const { return start_[A]; }
    int atom_npoints(int A) const { return start_[A + 1] - start_[A]; }

    /// Radial scale factor used for atom A (bohr), i.e. rscale * R_Slater(Z_A).
    double alpha(int A) const { return alpha_[A]; }

    void print_header() const;

   private:
    void build_radial(double alpha, std::vector<double>& r, std::vector<double>& wr) const;
    void apply_becke();

    std::shared_ptr<Molecule> molecule_;
    IsaGridOptions options_;
    int natom_;
    int spherical_points_;

    std::vector<double> x_, y_, z_, w_;
    std::vector<int> start_;   ///< natom_ + 1 entries
    std::vector<double> alpha_;
};

}  // namespace isapol
}  // namespace psi

#endif
