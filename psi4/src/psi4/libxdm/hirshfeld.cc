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

// Hirshfeld partitioning for XDM dispersion.
// Translates postg's atomin subroutine (wfnmod.f90).

#include "hirshfeld.h"
#include "xdm_data.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace psi {
namespace xdm {

ProatomDensity::ProatomDensity()
    : spline_initialized_(FTOT_NELEM, false),
      spline_a_(FTOT_NELEM),
      spline_b_(FTOT_NELEM),
      spline_c_(FTOT_NELEM) {}

void ProatomDensity::initialize_spline(int Z) const {
    if (Z < 1 || Z > FTOT_NELEM) return;
    int idx = Z - 1;
    if (spline_initialized_[idx]) return;

    int ndata = ndata_for_z(Z);
    if (ndata == 0) return;

    double h = 1.0 / (ndata + 1);
    double hinv = 1.0 / h;
    double h2inv = hinv * hinv;

    // Read data from ftot table: f[j] = ftot[j][idx]
    std::vector<double> f(ndata + 1);
    f[0] = 0.0;
    for (int j = 1; j <= ndata; j++) {
        f[j] = ftot[j][idx];
    }

    // Natural cubic spline on uniform grid (from postg tools_math.f90 SPLINE routine)
    // Boundary condition: f -> 0 at r -> infinity (BCR = 0)
    double bcr = 0.0;

    // Set up tridiagonal system
    std::vector<double> d(ndata + 1), e(ndata + 1), bb(ndata + 1);
    for (int i = 1; i < ndata; i++) {
        d[i] = 4.0;
        e[i] = 1.0;
        bb[i] = 3.0 * h2inv * (f[i - 1] - 2.0 * f[i] + f[i + 1]);
    }
    d[ndata] = 4.0;
    e[ndata] = 0.0;
    bb[ndata] = 3.0 * h2inv * (f[ndata - 1] - 2.0 * f[ndata] + bcr);

    // Solve tridiagonal system (DPTSL from LINPACK)
    // Forward elimination
    int n = ndata;
    if (n >= 2) {
        int nm1 = n - 1;
        for (int k = 1; k < nm1; k++) {
            double t = e[k] / d[k];
            d[k + 1] -= t * e[k];
            bb[k + 1] -= t * bb[k];
        }
        bb[nm1] /= d[nm1];
        // Backward substitution
        for (int k = nm1 - 1; k >= 1; k--) {
            bb[k] = (bb[k] - e[k] * bb[k + 1]) / d[k];
        }
    } else if (n == 1) {
        bb[1] /= d[1];
    }

    // Compute spline coefficients
    spline_a_[idx].resize(ndata + 1);
    spline_b_[idx].resize(ndata + 1);
    spline_c_[idx].resize(ndata + 1);

    auto& a = spline_a_[idx];
    auto& b = spline_b_[idx];
    auto& c = spline_c_[idx];

    double third = 1.0 / 3.0;
    double two_third = 2.0 / 3.0;

    a[0] = hinv * (f[1] - f[0]) - third * h * bb[1];
    b[0] = 0.0;
    c[0] = third * hinv * bb[1];

    for (int i = 1; i < ndata; i++) {
        a[i] = hinv * (f[i + 1] - f[i]) - two_third * h * bb[i] - third * h * bb[i + 1];
        b[i] = bb[i];
        c[i] = third * hinv * (bb[i + 1] - bb[i]);
    }

    a[ndata] = hinv * (bcr - f[ndata]) - two_third * h * bb[ndata];
    b[ndata] = bb[ndata];
    c[ndata] = -third * hinv * bb[ndata];

    spline_initialized_[idx] = true;
}

double ProatomDensity::evaluate(int Z, double r) const {
    if (Z < 1 || Z > FTOT_NELEM) return 0.0;
    if (r < 1.0e-15) return 0.0;

    initialize_spline(Z);

    int idx = Z - 1;
    int ndata = ndata_for_z(Z);
    double h = 1.0 / (ndata + 1);
    double rmid = 1.0 / std::cbrt(static_cast<double>(Z));

    // Transform r -> q
    double q = r / (r + rmid);
    int intq = static_cast<int>((ndata + 1) * q);
    intq = std::min(intq, ndata);
    double dq = q - intq * h;

    const auto& a = spline_a_[idx];
    const auto& b = spline_b_[idx];
    const auto& c = spline_c_[idx];

    // Spline evaluation: f(q) = ftot[intq] + dq * (a[intq] + dq * (b[intq] + dq * c[intq]))
    double fval;
    if (intq == 0) {
        fval = 0.0 + dq * (a[0] + dq * (b[0] + dq * c[0]));
    } else {
        fval = ftot[intq][idx] + dq * (a[intq] + dq * (b[intq] + dq * c[intq]));
    }

    // rho_atom = |f(q)| / r^2
    return std::abs(fval) / (r * r);
}

void compute_hirshfeld_weights(const ProatomDensity& proatom, int natom, const int* atomic_nums,
                               const double atom_coords[][3], int npoints, const double* grid_x,
                               const double* grid_y, const double* grid_z,
                               std::vector<std::vector<double>>& weights) {
    weights.resize(natom);
    for (int a = 0; a < natom; a++) {
        weights[a].resize(npoints, 0.0);
    }

    // Compute promolecular density at each grid point
    std::vector<double> promol(npoints, 0.0);

    for (int a = 0; a < natom; a++) {
        if (atomic_nums[a] < 1) continue;
        for (int p = 0; p < npoints; p++) {
            double dx = grid_x[p] - atom_coords[a][0];
            double dy = grid_y[p] - atom_coords[a][1];
            double dz = grid_z[p] - atom_coords[a][2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            double arho = proatom.evaluate(atomic_nums[a], r);
            weights[a][p] = arho;
            promol[p] += arho;
        }
    }

    // Normalize: h_A(r) = rho_A(r) / promol(r)
    for (int a = 0; a < natom; a++) {
        if (atomic_nums[a] < 1) continue;
        for (int p = 0; p < npoints; p++) {
            weights[a][p] /= std::max(promol[p], 1.0e-40);
        }
    }
}

}  // namespace xdm
}  // namespace psi
