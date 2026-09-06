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

#include "isa_grid.h"

#include "tables.h"

#include "psi4/libfock/cubature.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/psi4-dec.h"

#include <algorithm>
#include <cmath>

namespace psi {
namespace isapol {

namespace {

/// x^n for small non-negative integer n, by the same binary exponentiation a
/// Fortran compiler emits for `x**n`.  For the exponents used here (i^5, d^7 with
/// i, d <= a few hundred) every intermediate is an exactly representable integer,
/// so the result does not depend on the association anyway.
double ipow(double x, int n) {
    double result = 1.0;
    while (n > 0) {
        if (n & 1) result *= x;
        x *= x;
        n >>= 1;
    }
    return result;
}

}  // namespace

IsaGrid::IsaGrid(std::shared_ptr<Molecule> molecule, const IsaGridOptions& options)
    : molecule_(molecule), options_(options) {
    natom_ = molecule_->natom();
    if (natom_ < 1) throw PSIEXCEPTION("IsaGrid: molecule has no atoms.");
    if (options_.radial_points < 2)
        throw PSIEXCEPTION("IsaGrid: ISA_RADIAL_POINTS (CamCASP n_r) must be at least 2, got " +
                           std::to_string(options_.radial_points) + ".");
    if (options_.becke_smoothing < 0)
        throw PSIEXCEPTION("IsaGrid: ISA_BECKE_SMOOTHING must be non-negative, got " +
                           std::to_string(options_.becke_smoothing) + ".");
    if (!(options_.radius_scaling > 0.0))
        throw PSIEXCEPTION("IsaGrid: ISA_RADIUS_SCALING must be positive.");

    // Lbdv() (atom_grids.F90:509) rounds the requested order up to the next
    // tabulated Lebedev size and clamps at 5294.  Reproduce both.
    const int kCamcaspMaxLebedev = 5294;
    int requested = std::min(options_.spherical_points, kCamcaspMaxLebedev);
    spherical_points_ = lebedev_npoints_at_least(requested);
    if (spherical_points_ < 0 || spherical_points_ > kCamcaspMaxLebedev)
        throw PSIEXCEPTION("IsaGrid: no Lebedev grid available for ISA_SPHERICAL_POINTS = " +
                           std::to_string(options_.spherical_points) + ".");
    const MassPoint* ang = lebedev_sphere(spherical_points_);
    if (ang == nullptr)
        throw PSIEXCEPTION("IsaGrid: Lebedev grid of " + std::to_string(spherical_points_) +
                           " points is tabulated but unavailable.");

    const int n_shell = options_.radial_points - 1;
    const size_t npoints = static_cast<size_t>(natom_) * n_shell * spherical_points_;

    x_.reserve(npoints);
    y_.reserve(npoints);
    z_.reserve(npoints);
    w_.reserve(npoints);
    start_.resize(natom_ + 1);
    alpha_.resize(natom_);

    // make_grid, atom_grids.F90:302-316.  Atom-major, then radial shell, then
    // angular point; spheres are translated onto the nucleus but never rotated.
    std::vector<double> r, wr;
    for (int A = 0; A < natom_; ++A) {
        // CamCASP scales the radial grid by the Bragg-Slater radius of the site's
        // atomic number.  Psi4 ghost atoms keep their element identity here --
        // they carry basis functions and therefore take part in the partition.
        const int Z = static_cast<int>(molecule_->true_atomic_number(A));
        alpha_[A] = options_.radius_scaling * slater_radius(Z);
        build_radial(alpha_[A], r, wr);

        const double cx = molecule_->x(A), cy = molecule_->y(A), cz = molecule_->z(A);
        start_[A] = static_cast<int>(x_.size());
        for (int p = 0; p < n_shell; ++p) {
            for (int q = 0; q < spherical_points_; ++q) {
                x_.push_back(r[p] * ang[q].x + cx);
                y_.push_back(r[p] * ang[q].y + cy);
                z_.push_back(r[p] * ang[q].z + cz);
                // Psi4's Lebedev weights already carry the 4*pi that CamCASP
                // applies here explicitly, and sum to 4*pi rather than 1.
                w_.push_back(wr[p] * ang[q].w);
            }
        }
    }
    start_[natom_] = static_cast<int>(x_.size());

    apply_becke();
}

void IsaGrid::build_radial(double alpha, std::vector<double>& r, std::vector<double>& wr) const {
    // radial_grid, atom_grids.F90:411.  Murray-Handy-Laming ("Euler-MacLaurin")
    // map with m_r = 2:
    //
    //     r_i = alpha (i / (n_r - i))^m ,  i = 1 ... n_r - 1
    //     w_i = m n_r alpha^3 i^(3m-1) / (n_r - i)^(3m+1)
    //
    // n_r is CamCASP's radial count; only n_r - 1 shells are generated.
    const int m_r = 2;
    const int n_r = options_.radial_points;

    r.resize(n_r - 1);
    wr.resize(n_r - 1);

    const double f = double(m_r * n_r) * (alpha * alpha * alpha);
    for (int i = 1; i < n_r; ++i) {
        const double d = double(n_r - i);
        const double ratio = double(i) / d;
        r[i - 1] = alpha * ipow(ratio, m_r);
        wr[i - 1] = f * ipow(double(i), 3 * m_r - 1) / ipow(d, 3 * m_r + 1);
    }
}

void IsaGrid::apply_becke() {
    // make_grid, atom_grids.F90:326-383.  Skipped outright for a single atom,
    // where every weight is 1.
    if (natom_ < 2) return;

    const int n = natom_;
    std::vector<double> rr(n * n, 0.0), aa(n * n, 0.0), s(n * n, 0.0), pp(n), gr(n);

    for (int a = 0; a < n; ++a) {
        s[a * n + a] = 1.0;  // set once; the point loop only touches off-diagonals
        for (int b = 0; b < n; ++b) {
            if (b == a) continue;
            const double dx = molecule_->x(b) - molecule_->x(a);
            const double dy = molecule_->y(b) - molecule_->y(a);
            const double dz = molecule_->z(b) - molecule_->z(a);
            rr[a * n + b] = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (rr[a * n + b] == 0.0)
                throw PSIEXCEPTION("IsaGrid: atoms " + std::to_string(a) + " and " + std::to_string(b) +
                                   " are at the same position; the Becke partition is undefined.");
            // Becke's atomic-size adjustment.  This is algebraically
            // (1 - chi^2) / (4 chi), but is written in CamCASP's form so the
            // rounding matches, and is deliberately *not* clamped to [-1/2, 1/2]
            // the way Psi4's cubature.cc clamps it.
            const double chi = alpha_[a] / alpha_[b];
            const double u = (chi - 1.0) / (chi + 1.0);
            aa[a * n + b] = u / (u * u - 1.0);
        }
    }

    const int k_mu = options_.becke_smoothing;
    for (int a = 0; a < n; ++a) {
        const int lo = start_[a], hi = start_[a + 1];
        for (int g = lo; g < hi; ++g) {
            for (int m = 0; m < n; ++m) {
                const double dx = x_[g] - molecule_->x(m);
                const double dy = y_[g] - molecule_->y(m);
                const double dz = z_[g] - molecule_->z(m);
                gr[m] = std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            for (int ma = 0; ma < n; ++ma) {
                for (int mb = 0; mb < n; ++mb) {
                    if (ma == mb) continue;
                    const double mu_ab = (gr[ma] - gr[mb]) / rr[ma * n + mb];
                    const double nu_ab = mu_ab + aa[ma * n + mb] * (1.0 - mu_ab * mu_ab);
                    double f = nu_ab;
                    for (int i = 0; i < k_mu; ++i) f = f * (1.5 - 0.5 * f * f);
                    s[ma * n + mb] = 0.5 * (1.0 - f);
                }
                double prod = 1.0;
                for (int mb = 0; mb < n; ++mb) prod *= s[ma * n + mb];
                pp[ma] = prod;
            }
            double total = 0.0;
            for (int ma = 0; ma < n; ++ma) total += pp[ma];
            w_[g] *= pp[a] / total;
        }
    }
    // No weight cutoff: CamCASP's point-suppression block is commented out, so
    // every point survives, however small its weight.
}

void IsaGrid::print_header() const {
    outfile->Printf("  ==> ISA Integration Grid <==\n\n");
    outfile->Printf("    Radial scheme        =    Euler-MacLaurin (m = 2)\n");
    outfile->Printf("    Nuclear scheme       =    Becke (Bragg-Slater, unclamped)\n");
    outfile->Printf("    Becke smoothing      = %10d\n", options_.becke_smoothing);
    outfile->Printf("    Radial points (n_r)  = %10d\n", options_.radial_points);
    outfile->Printf("    Radial shells        = %10d\n", options_.radial_points - 1);
    outfile->Printf("    Spherical points     = %10d\n", spherical_points_);
    if (spherical_points_ != options_.spherical_points)
        outfile->Printf("      (requested %d, rounded up to the next Lebedev order)\n",
                        options_.spherical_points);
    outfile->Printf("    Radius scaling       = %10.4f\n", options_.radius_scaling);
    outfile->Printf("    Total points         = %10d\n\n", npoints());
}

}  // namespace isapol
}  // namespace psi
