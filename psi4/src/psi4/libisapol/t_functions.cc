/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Irregular solid harmonics and pfit T functions, ported with permission from
 * CamCASP 6.0 by Alston J. Misquitta and Anthony J. Stone
 * (http://gitlab.com/anthonyjstone/camcasp).  The recursion, component order
 * and local-axis convention follow src/pfit/shift.f90::solidh and
 * src/pfit/process.F90::T_functions, referenced inline by line.  No CamCASP
 * source is linked into this module.
 *
 * See SPEC.md in this directory for the full specification.
 */
#include "t_functions.h"
#include <cmath>
#include <stdexcept>
#include <string>
namespace psi { namespace isapol {
namespace {
void t_require(bool ok, const std::string& message) {
    if (!ok) throw std::invalid_argument("isa_t_functions: " + message);
}
// shift.f90:26-30 tabulates rt(t)=sqrt(real(t)) once; sqrt is correctly rounded
// for every IEEE double, so recomputing it here gives the same bits.
double rt(int t) { return std::sqrt(static_cast<double>(t)); }
}  // namespace

std::vector<double> isa_irregular_solid_harmonics(int rank, const std::array<double,3>& displacement) {
    t_require(rank >= 0 && rank <= 4, "irregular solid harmonic rank must be in [0,4]");
    for (double v : displacement) t_require(std::isfinite(v), "displacement must be finite");
    const int l = rank;
    std::vector<double> r(static_cast<size_t>((l+1)*(l+1)), 0.0);
    const double x = displacement[0], y = displacement[1], z = displacement[2];
    // shift.f90:134-147.  Fortran index i maps to zero-based i-1 throughout, so
    // R(k,0) sits at k*k and R(k,mc)/R(k,ms) at k*k+2m-1 and k*k+2m.
    double rr = x*x+y*y+z*z;
    t_require(rr > 0.0, "irregular solid harmonics are singular at the origin");
    rr = 1.0/rr;
    r[0] = std::sqrt(rr);
    t_require(std::isfinite(r[0]), "displacement underflows the irregular harmonics");
    if (l == 0) return r;
    const double rfx = x*rr, rfy = y*rr, rfz = z*rr;
    r[1] = rfz*r[0];
    r[2] = rfx*r[0];
    r[3] = rfy*r[0];
    // shift.f90:161-204.  The regular branch differs only in its seeding, which
    // isa_regular_multipoles already provides by an independent construction.
    for (int k = 1; k < l; ++k) {
        const int n = k+1;
        int ln = n*n, lk = k*k, lp = (k-1)*(k-1);
        const double a2kp1 = k+k+1;
        r[ln] = (a2kp1*r[lk]*rfz-k*rr*r[lp])/(k+1);
        int m = 1;
        ++ln; ++lk; ++lp;
        while (m < k) {
            r[ln] = (a2kp1*r[lk]*rfz-rt(k+m)*rt(k-m)*rr*r[lp])/(rt(n+m)*rt(n-m));
            r[ln+1] = (a2kp1*r[lk+1]*rfz-rt(k+m)*rt(k-m)*rr*r[lp+1])/(rt(n+m)*rt(n-m));
            ++m; ln += 2; lk += 2; lp += 2;
        }
        r[ln] = rt(n+k)*r[lk]*rfz;
        r[ln+1] = rt(n+k)*r[lk+1]*rfz;
        ln += 2;
        const double s = rt(n+k)/rt(n+n);
        r[ln] = s*(rfx*r[lk]-rfy*r[lk+1]);
        r[ln+1] = s*(rfx*r[lk+1]+rfy*r[lk]);
    }
    for (double v : r) t_require(std::isfinite(v), "nonfinite irregular solid harmonic");
    return r;
}

double isa_t_function_damping(int rank, double br) {
    t_require(rank >= 0 && rank <= 4, "damping rank must be in [0,4]");
    t_require(std::isfinite(br) && br >= 0.0, "reduced distance must be finite and nonnegative");
    // process.F90:475-501 accumulates u once per rank block in ascending order.
    const double denominators[4] = {2.0, 6.0, 24.0, 120.0};
    double u = 1.0+br, power = br;
    for (int k = 1; k <= rank; ++k) {
        power *= br;
        u += power/denominators[k-1];
    }
    const double factor = 1.0-std::exp(-br)*u;
    t_require(std::isfinite(factor), "nonfinite damping factor");
    return factor;
}

std::vector<double> isa_t_functions(int rank, const std::array<double,3>& point_bohr,
                                    const std::array<double,3>& site_bohr,
                                    const std::array<std::array<double,3>,3>& frame,
                                    double damping) {
    for (double v : point_bohr) t_require(std::isfinite(v), "point must be finite");
    for (double v : site_bohr) t_require(std::isfinite(v), "site must be finite");
    for (const auto& row : frame) for (double v : row) t_require(std::isfinite(v), "frame must be finite");
    t_require(std::isfinite(damping) && damping >= 0.0, "damping must be finite and nonnegative");
    // Same frame contract as isa_multipole_rotation: columns are the local axes
    // in the global frame, which is CamCASP's sm(i,j,s) (sites.f90:22-23).
    for (int i = 0; i < 3; ++i) for (int j = i; j < 3; ++j) {
        long double dot = 0.L;
        for (int k = 0; k < 3; ++k) dot += static_cast<long double>(frame[k][i])*frame[k][j];
        t_require(std::abs(dot-(i == j ? 1.L : 0.L)) <= 1.e-12L, "frame must be orthogonal");
    }
    const long double determinant =
        static_cast<long double>(frame[0][0])*(frame[1][1]*frame[2][2]-frame[1][2]*frame[2][1])-
        static_cast<long double>(frame[0][1])*(frame[1][0]*frame[2][2]-frame[1][2]*frame[2][0])+
        static_cast<long double>(frame[0][2])*(frame[1][0]*frame[2][1]-frame[1][1]*frame[2][0]);
    t_require(std::abs(determinant-1.L) <= 1.e-12L, "frame must be a proper rotation");
    // process.F90:470-473: x = q - site, then x = matmul(x, sm), i.e. the local
    // component along axis j is the global displacement contracted with column j.
    std::array<double,3> local{};
    for (int j = 0; j < 3; ++j) {
        double sum = 0.0;
        for (int i = 0; i < 3; ++i) sum += (point_bohr[i]-site_bohr[i])*frame[i][j];
        local[j] = sum;
    }
    auto t = isa_irregular_solid_harmonics(rank, local);
    if (damping > 0.0) {
        const double r = std::sqrt(local[0]*local[0]+local[1]*local[1]+local[2]*local[2]);
        const double br = damping*r;
        for (int k = 0; k <= rank; ++k) {
            const double factor = isa_t_function_damping(k, br);
            for (int c = k*k; c < (k+1)*(k+1); ++c) t[c] = factor*t[c];
        }
    }
    return t;
}
} }
