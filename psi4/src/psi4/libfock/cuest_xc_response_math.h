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
 * Per-point algebra of the semilocal (LDA/GGA) XC response kernel on the cuEST grid.
 *
 * Shared verbatim by the host C++ route and the CUDA kernels, so both LibXC routes
 * contract the reference derivatives with the signed response density identically.
 * Every function here reproduces a native CPU expression in v.cc / superfunctional.cc;
 * the comments name the one it reproduces.
 *
 * Layouts. cuEST densities are point-major [rho, d_x rho, d_y rho, d_z rho] (GGA) or
 * [rho] (LDA), one spin at a time. LibXC derivative blocks keep LibXC's own per-point
 * layouts (unpolarized: one value; polarized: v2rho2 3, v2rhosigma 6, v2sigma2 6,
 * vsigma 3). The output is the cuEST grid potential [p0, p_x, p_y, p_z] whose AO
 * integral is  V_mn = sum_P p0 phi_m phi_n + p . grad(phi_m phi_n),  weights included.
 */

#ifndef PSI4_LIBFOCK_CUEST_XC_RESPONSE_MATH_H
#define PSI4_LIBFOCK_CUEST_XC_RESPONSE_MATH_H

#if defined(__CUDACC__)
#define PSI4_XC_RESPONSE_HD __host__ __device__ inline
#else
#define PSI4_XC_RESPONSE_HD inline
#endif

namespace psi {
namespace cuest_xc_response {

/// Spin-adapted RKS kernel (A, B, C, S) of one grid point. For a singlet this is the
/// unpolarized LibXC output at the total density; for a triplet, project_triplet below.
struct RKSKernel {
    double A;  ///< f_rho,rho
    double B;  ///< f_rho,gamma
    double C;  ///< f_gamma,gamma
    double S;  ///< f_gamma
};

/// SuperFunctional::compute_functional's triplet combinations (superfunctional.cc,
/// "magic triplet combinations"), from polarized LibXC derivatives evaluated at the
/// closed-shell reference rho_a = rho_b, sigma_aa = sigma_ab = sigma_bb.
PSI4_XC_RESPONSE_HD RKSKernel project_triplet(const double* v2rho2, const double* v2rhosigma,
                                              const double* v2sigma2, const double* vsigma, bool gga) {
    RKSKernel k{0.0, 0.0, 0.0, 0.0};
    k.A = 0.5 * v2rho2[0] - 0.5 * v2rho2[1];
    if (gga) {
        k.B = 0.25 * v2rhosigma[0] - 0.25 * v2rhosigma[2];
        k.C = 0.125 * v2sigma2[0] - 0.125 * v2sigma2[2];
        k.S = 0.5 * vsigma[0] - 0.25 * vsigma[1];
    }
    return k;
}

/// RV::compute_Vx_full for one point. ref_alpha is the cuEST one-spin reference density,
/// so the total density and gradient are twice it. resp is the (undoubled) signed response
/// density of Dx. The native loop skips a point when the total density is below the cutoff.
///   p0 = w [A u + B t]          t = 2 g.h
///   p  = 2 w [S h + (B u + C t) g]
/// This is the CPU intermediate T = 0.5 p0 phi + p.grad(phi) followed by phi^T T + T^T phi:
/// the half-alpha response normalization of compute_Vx_full, with the weights applied once.
PSI4_XC_RESPONSE_HD void pack_rks(bool gga, double w, double cutoff, const double* ref_alpha,
                                  const RKSKernel& k, const double* resp, double* out) {
    const double n = 2.0 * ref_alpha[0];
    if (!gga) {
        out[0] = (n < cutoff) ? 0.0 : w * k.A * resp[0];
        return;
    }
    if (n < cutoff) {
        out[0] = out[1] = out[2] = out[3] = 0.0;
        return;
    }
    const double gx = 2.0 * ref_alpha[1], gy = 2.0 * ref_alpha[2], gz = 2.0 * ref_alpha[3];
    const double u = resp[0];
    const double hx = resp[1], hy = resp[2], hz = resp[3];
    const double t = 2.0 * (gx * hx + gy * hy + gz * hz);
    out[0] = w * (k.A * u + k.B * t);
    const double scalar = k.B * u + k.C * t;
    out[1] = 2.0 * w * (k.S * hx + scalar * gx);
    out[2] = 2.0 * w * (k.S * hy + scalar * gy);
    out[3] = 2.0 * w * (k.S * hz + scalar * gz);
}

/// UV::compute_Vx for one point and one alpha/beta response pair. ref_a/ref_b are the
/// cuEST spin reference densities, resp_a/resp_b the signed spin response densities.
/// The derivative blocks are polarized LibXC layouts:
///   v2rho2 [a_a, a_b, b_b], v2rhosigma [a_aa, a_ab, a_bb, b_aa, b_ab, b_bb],
///   v2sigma2 [aa_aa, aa_ab, aa_bb, ab_ab, ab_bb, bb_bb], vsigma [aa, ab, bb].
/// The native LDA block is screened with rho_a + rho_b > cutoff, the GGA block with
/// rho_a + rho_b >= cutoff; at exact equality only the GGA terms survive.
PSI4_XC_RESPONSE_HD void pack_uks(bool gga, double w, double cutoff, const double* ref_a, const double* ref_b,
                                  const double* v2rho2, const double* v2rhosigma, const double* v2sigma2,
                                  const double* vsigma, const double* resp_a, const double* resp_b, double* out_a,
                                  double* out_b) {
    const int ncomponents = gga ? 4 : 1;
    for (int c = 0; c < ncomponents; ++c) out_a[c] = out_b[c] = 0.0;
    const double rho = ref_a[0] + ref_b[0];
    const double ua = resp_a[0], ub = resp_b[0];
    if (rho > cutoff) {
        out_a[0] = w * (v2rho2[0] * ua + v2rho2[1] * ub);
        out_b[0] = w * (v2rho2[2] * ub + v2rho2[1] * ua);
    }
    if (!gga || rho < cutoff) return;

    const double* ga = ref_a + 1;
    const double* gb = ref_b + 1;
    const double* ha = resp_a + 1;
    const double* hb = resp_b + 1;
    const double gamma_aak = 2.0 * (ha[0] * ga[0] + ha[1] * ga[1] + ha[2] * ga[2]);
    const double gamma_bbk = 2.0 * (hb[0] * gb[0] + hb[1] * gb[1] + hb[2] * gb[2]);
    const double gamma_abk = ha[0] * gb[0] + hb[0] * ga[0] + ha[1] * gb[1] + hb[1] * ga[1] + ha[2] * gb[2] +
                             hb[2] * ga[2];

    // "Term 2b": the rho-sigma block acting on the gradient perturbation.
    out_a[0] += w * (v2rhosigma[0] * gamma_aak + v2rhosigma[1] * gamma_abk + v2rhosigma[2] * gamma_bbk);
    out_b[0] += w * (v2rhosigma[3] * gamma_aak + v2rhosigma[4] * gamma_abk + v2rhosigma[5] * gamma_bbk);

    // The perturbed sigma derivatives, delta v_sigma = H_sigma,(rho,sigma) z.
    const double dv_aa = v2rhosigma[0] * ua + v2rhosigma[3] * ub + v2sigma2[0] * gamma_aak +
                         v2sigma2[1] * gamma_abk + v2sigma2[2] * gamma_bbk;
    const double dv_ab = v2rhosigma[1] * ua + v2rhosigma[4] * ub + v2sigma2[1] * gamma_aak +
                         v2sigma2[3] * gamma_abk + v2sigma2[4] * gamma_bbk;
    const double dv_bb = v2rhosigma[2] * ua + v2rhosigma[5] * ub + v2sigma2[2] * gamma_aak +
                         v2sigma2[4] * gamma_abk + v2sigma2[5] * gamma_bbk;

    for (int x = 0; x < 3; ++x) {
        out_a[x + 1] = w * (2.0 * vsigma[0] * ha[x] + vsigma[1] * hb[x] + 2.0 * dv_aa * ga[x] + dv_ab * gb[x]);
        out_b[x + 1] = w * (2.0 * vsigma[2] * hb[x] + vsigma[1] * ha[x] + 2.0 * dv_bb * gb[x] + dv_ab * ga[x]);
    }
}

}  // namespace cuest_xc_response
}  // namespace psi

#undef PSI4_XC_RESPONSE_HD

#endif
