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

#ifndef PSI4_LIBISAPOL_FIT_POINTS_H
#define PSI4_LIBISAPOL_FIT_POINTS_H

#include <memory>
#include <vector>

namespace psi {

class Molecule;

namespace isapol {

/// Maclaren's (1992) additive lagged-Fibonacci generator, CamCASP's `sdprnd` /
/// `dprand` (src/random.f90).
///
/// Reimplemented here rather than linked because the fit-point cloud has to be
/// reproduced draw for draw: the point-response refinement of SPEC.md 12 fits to
/// the potential sampled at these points, so a single extra or missing draw
/// changes every subsequent point and every fitted coefficient.  The generator
/// itself is published --- N. M. Maclaren, "A Limited Portable Fortran-77 Random
/// Number Generator", University of Cambridge (1992) --- and CamCASP's copy
/// carries the notice reproduced below, whose condition this comment discharges:
///
///     Copyright (C) 1992  N. M. Maclaren
///     Copyright (C) 1992  The University of Cambridge
///
///     This software may be reproduced and used freely, provided that all users
///     of it agree that the copyright holders are not liable for any damage or
///     injury caused by use of this software and that this condition is passed
///     onto all subsequent recipients of the software, whether modified or not.
///
/// The state is 101 lags held as exact integers in `double`, plus a second,
/// independent multiplicative generator mixed in modulo 1.  Everything is done
/// in floating point on purpose: the recurrence is exact in double precision
/// (the modulus is under 2^30), and the published algorithm's rounding is part
/// of the definition.  Do not "improve" it into integer arithmetic.
class MaclarenRng {
   public:
    /// Seeds with `seed`, which CamCASP documents as an integer in [0, 9999];
    /// larger values are folded by `mod(abs(seed), 10000)` exactly as `sdprnd`
    /// does, so nothing is rejected.
    ///
    /// CamCASP's `sdprnd(0)` re-seeds only if the generator has never been
    /// seeded, using a module-wide `inital` flag.  Here the state is per object,
    /// so a freshly constructed MaclarenRng always seeds --- which is what a
    /// fresh CamCASP process does too.
    explicit MaclarenRng(int seed_value = 0);

    /// Re-seed in place.
    void seed(int seed_value);

    /// Next uniform deviate on (0, 1), CamCASP's `dprand()`.
    double next();

   private:
    double poly_[101];  ///< 1-based in CamCASP; poly_[i - 1] here
    double other_;
    double offset_;
    int index_;  ///< kept 1-based, as in CamCASP
};

/// Knobs of the fit-point cloud.  Defaults reproduce the reference protocol
/// `methods/isa-pol-from-isa-A`, i.e. CamCASP's `SET Lattice / Charge 1.0 /
/// LoLim 2.0 / HiLim 4.0 / Random 2000 / Seed 1 / END`.
struct FitPointsOptions {
    /// Number of points to *accept*, CamCASP's `nlat`.  Candidates are drawn
    /// until this many pass, so the cost is set by the acceptance rate, not by
    /// this number alone.
    int npoints = 2000;

    /// Inner cutoff in units of the van der Waals radius, CamCASP's `lowlim`.
    /// A point closer than lolim * R_vdW(k) to *any* atom k is rejected.
    double lolim = 2.0;

    /// Outer cutoff in units of the van der Waals radius, CamCASP's `hilim`.
    /// A point farther than hilim * R_vdW(k) from *every* atom k is rejected.
    double hilim = 4.0;

    /// Seed for the point generator.  CamCASP's `SEED n`; the protocols use 1.
    int seed = 1;
};

/// The cloud of points at which the point-response refinement samples the
/// electrostatic potential, CamCASP's "lattice"
/// (src/lattice.F90::generate_lattice, `LatticeType = "RANDOM"`).
///
/// Candidates are drawn uniformly from the cube of half-width `dmax` centred on
/// the unweighted mean of the nuclear positions, and kept if they lie in the
/// shell between lolim and hilim van der Waals radii --- see accepted().
///
/// Two details of the construction matter for reproducing CamCASP and are easy
/// to get wrong:
///
///   * `dmax` is *not* a radius.  It is the largest of |cm_j - x_j(i)| +
///     hilim * R_vdW(i) over every atom i and every Cartesian direction j, i.e.
///     a single half-width shared by all three axes, mixing a component-wise
///     displacement with a spherical envelope.  The sampled region is a cube.
///   * Rejected candidates still consume three deviates.  x, y and z are drawn
///     in that order, one call each, before the accept test runs.
///
/// The points carry a charge (CamCASP's `LatticeCharge`, +1 by default) used
/// when the potential is evaluated; that belongs to the refinement, not here.
class FitPoints {
   public:
    FitPoints(std::shared_ptr<Molecule> molecule, const FitPointsOptions& options);

    int npoints() const { return static_cast<int>(x_.size()); }
    const FitPointsOptions& options() const { return options_; }

    const double* x() const { return x_.data(); }
    const double* y() const { return y_.data(); }
    const double* z() const { return z_.data(); }

    /// Candidates drawn, accepted and rejected together; 3 * ncandidates()
    /// deviates were consumed.  Diagnostic only --- a low acceptance rate means
    /// a badly shaped molecule, not a bug.
    int ncandidates() const { return ncandidates_; }

    /// Cube half-width actually used (bohr); see the class comment.
    double dmax() const { return dmax_; }

    /// Centre of the sampling cube (bohr), the unweighted mean of the nuclear
    /// positions --- CamCASP calls it `cm` but does not mass-weight it.
    const double* centre() const { return centre_; }

    /// True if p lies in the accepted shell: no closer than lolim * R_vdW(k) to
    /// any atom k, and closer than hilim * R_vdW(k) to at least one.  This is
    /// CamCASP's `add` (lattice.F90:761) with its `s` offset fixed at zero, the
    /// only value generate_lattice passes.
    bool accepted(const double p[3]) const;

    void print_header() const;

   private:
    std::shared_ptr<Molecule> molecule_;
    FitPointsOptions options_;
    int natom_;
    int ncandidates_;
    double dmax_;
    double centre_[3];

    std::vector<double> minenv_;  ///< lolim * R_vdW per atom
    std::vector<double> maxenv_;  ///< hilim * R_vdW per atom
    std::vector<double> cx_, cy_, cz_;
    std::vector<double> x_, y_, z_;
};

}  // namespace isapol
}  // namespace psi

#endif
