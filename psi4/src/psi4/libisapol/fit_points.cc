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


#include "fit_points.h"

#include "tables.h"

#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace psi {
namespace isapol {

namespace {

// Moduli of the Maclaren generator (random.f90:24, 65-67).  kXmod is prime and
// under 2^30, so every intermediate below is exactly representable.
constexpr double kXmod = 1000009711.0;
constexpr double kYmod = 33554432.0;  // 2^25
constexpr double kXmod2 = 2000019422.0;
constexpr double kXmod4 = 4000038844.0;

// Nudge that keeps dprand() strictly positive (random.f90:67).
constexpr double kTiny = 1.0e-17;

}  // namespace

MaclarenRng::MaclarenRng(int s) { seed(s); }

void MaclarenRng::seed(int s) {
    // sdprnd (random.f90:21-58).  index must end up in [1, 101], poly in
    // [0, kXmod - 1] and not all zero, and other a non-negative proper fraction
    // over kYmod.  Maclaren uses Wichmann-Hill to get there.
    int ix = std::abs(s) % 10000 + 1;
    int iy = 2 * ix + 1;
    int iz = 3 * ix + 1;

    // x is read before it is written on the first pass in CamCASP too, but the
    // i >= 1 guard means no uninitialised value is ever stored: the loop starts
    // at -10 precisely to burn in the Wichmann-Hill state first.
    double x = 0.0;
    for (int i = -10; i <= 101; ++i) {
        if (i >= 1) poly_[i - 1] = std::trunc(kXmod * x);
        ix = 171 * ix % 30269;
        iy = 172 * iy % 30307;
        iz = 170 * iz % 30323;
        x = std::fmod(ix / 30269.0 + iy / 30307.0 + iz / 30323.0, 1.0);
    }
    other_ = std::trunc(kYmod * x) / kYmod;
    offset_ = 1.0 / kYmod;
    index_ = 1;
}

double MaclarenRng::next() {
    // dprand (random.f90:62-112).
    int n = index_ - 64;
    if (n <= 0) n += 101;

    double x = poly_[index_ - 1] + poly_[index_ - 1];
    x = kXmod4 - poly_[n - 1] - poly_[n - 1] - x - x - poly_[index_ - 1];

    // Reduce into [0, kXmod) by subtraction rather than fmod, as published: the
    // operands are exact integers and the branches are cheaper than a divide.
    if (x < 0.0) {
        if (x < -kXmod) x += kXmod2;
        if (x < 0.0) x += kXmod;
    } else {
        if (x >= kXmod2) {
            x -= kXmod2;
            if (x >= kXmod) x -= kXmod;
        }
        if (x >= kXmod) x -= kXmod;
    }

    poly_[index_ - 1] = x;
    index_ += 1;
    if (index_ > 101) index_ -= 101;

    // Mix in the second generator modulo 1 and force the result non-zero.
    double y;
    do {
        y = 37.0 * other_ + offset_;
        other_ = y - std::trunc(y);
    } while (other_ == 0.0);

    x = x / kXmod + other_;
    if (x >= 1.0) x -= 1.0;
    return x + kTiny;
}

FitPoints::FitPoints(std::shared_ptr<Molecule> molecule, const FitPointsOptions& options)
    : molecule_(molecule), options_(options), natom_(molecule->natom()), ncandidates_(0), dmax_(0.0) {
    if (natom_ < 1) throw PSIEXCEPTION("FitPoints: molecule has no atoms");
    if (options_.npoints < 1) throw PSIEXCEPTION("FitPoints: npoints must be positive");
    if (!(options_.hilim > options_.lolim)) {
        throw PSIEXCEPTION("FitPoints: hilim must exceed lolim, or no point can be accepted");
    }

    minenv_.resize(natom_);
    maxenv_.resize(natom_);
    cx_.resize(natom_);
    cy_.resize(natom_);
    cz_.resize(natom_);

    // lattice.F90:545-549.  The van der Waals radius comes from MODULE radii,
    // *not* from AtomProp -- see tables.h.  CamCASP rounds the nuclear charge to
    // the nearest integer, which for a ghost site gives Z = 0 and hence a radius
    // of zero: the site is then invisible to both cutoffs, exactly as in
    // CamCASP.
    for (int i = 0; i < natom_; ++i) {
        const int Z = static_cast<int>(std::lround(molecule_->Z(i)));
        const double r = vdw_radius(Z);
        minenv_[i] = options_.lolim * r;
        maxenv_[i] = options_.hilim * r;
        cx_[i] = molecule_->x(i);
        cy_[i] = molecule_->y(i);
        cz_[i] = molecule_->z(i);
    }

    // lattice.F90:551-563.  Note the accumulation order: CamCASP sums the first
    // atom in, then atoms 2...nat, then divides.  Summing in a different order
    // would move the centre by an ulp and, over 2000 candidates, flip an
    // accept/reject decision.
    centre_[0] = cx_[0];
    centre_[1] = cy_[0];
    centre_[2] = cz_[0];
    for (int i = 1; i < natom_; ++i) {
        centre_[0] += cx_[i];
        centre_[1] += cy_[i];
        centre_[2] += cz_[i];
    }
    for (int j = 0; j < 3; ++j) centre_[j] /= natom_;

    for (int i = 0; i < natom_; ++i) {
        const double c[3] = {cx_[i], cy_[i], cz_[i]};
        for (int j = 0; j < 3; ++j) {
            dmax_ = std::max(dmax_, std::fabs(centre_[j] - c[j]) + maxenv_[i]);
        }
    }

    // lattice.F90:570-580.  Three deviates per candidate, in the order x, y, z,
    // drawn before the accept test and consumed whether or not it passes.
    MaclarenRng rng(options_.seed);
    x_.reserve(options_.npoints);
    y_.reserve(options_.npoints);
    z_.reserve(options_.npoints);
    while (static_cast<int>(x_.size()) < options_.npoints) {
        double p[3];
        p[0] = centre_[0] + dmax_ * (2.0 * rng.next() - 1.0);
        p[1] = centre_[1] + dmax_ * (2.0 * rng.next() - 1.0);
        p[2] = centre_[2] + dmax_ * (2.0 * rng.next() - 1.0);
        ncandidates_ += 1;
        if (accepted(p)) {
            x_.push_back(p[0]);
            y_.push_back(p[1]);
            z_.push_back(p[2]);
        }
    }
}

bool FitPoints::accepted(const double p[3]) const {
    // lattice.F90:761-794, `add`, with s = 0.  The early exit on toolow is
    // CamCASP's; it leaves toohigh half-computed, which is harmless because
    // toolow alone decides the outcome.
    bool toohigh = true;
    for (int k = 0; k < natom_; ++k) {
        const double dx = p[0] - cx_[k];
        const double dy = p[1] - cy_[k];
        const double dz = p[2] - cz_[k];
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < minenv_[k]) return false;
        if (dist < maxenv_[k]) toohigh = false;
    }
    return !toohigh;
}

void FitPoints::print_header() const {
    outfile->Printf("  ==> ISA-Pol Fit Points <==\n\n");
    outfile->Printf("    Points            = %8d\n", npoints());
    outfile->Printf("    Candidates drawn  = %8d\n", ncandidates_);
    outfile->Printf("    Acceptance rate   = %8.2f %%\n",
                    100.0 * static_cast<double>(npoints()) / static_cast<double>(ncandidates_));
    outfile->Printf("    Inner cutoff      = %8.4f R_vdW\n", options_.lolim);
    outfile->Printf("    Outer cutoff      = %8.4f R_vdW\n", options_.hilim);
    outfile->Printf("    Cube half-width   = %8.4f bohr\n", dmax_);
    outfile->Printf("    Seed              = %8d\n\n", options_.seed);
}

}  // namespace isapol
}  // namespace psi
