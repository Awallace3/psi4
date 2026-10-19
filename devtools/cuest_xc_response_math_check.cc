/*
 * Host self-check of psi4/src/psi4/libfock/cuest_xc_response_math.h, the per-point
 * algebra shared by both cuEST XC response routes. Needs no GPU, cuEST or LibXC:
 *
 *   c++ -std=c++17 -I psi4/src devtools/cuest_xc_response_math_check.cc -o math_check && ./math_check
 *
 * Checks the cutoff masks at exact equality (native RKS skips rho < cutoff; native UKS
 * keeps the LDA block only for rho_a + rho_b > cutoff and the GGA block for >= cutoff),
 * and that the RKS triplet projection equals half the UKS alpha output for the
 * response pair (u, -u) at a closed-shell reference.
 */

#include <cmath>
#include <cstdio>
#include <random>

#include "psi4/libfock/cuest_xc_response_math.h"

using namespace psi::cuest_xc_response;

static int failures = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    auto r = [&]() { return uniform(rng); };

    const double cutoff = 1.e-10;
    const double w = 0.37;
    const RKSKernel k{r(), r(), r(), r()};
    const double resp[4] = {r(), r(), r(), r()};
    double out[4];

    // => RKS masks: the total density n = 2 rho_alpha <= //
    for (bool gga : {false, true}) {
        double ref[4] = {0.5 * cutoff, r(), r(), r()};
        pack_rks(gga, w, cutoff, ref, k, resp, out);
        expect(out[0] != 0.0, "RKS point at n == cutoff is kept");
        ref[0] = 0.5 * std::nextafter(cutoff, 0.0);
        pack_rks(gga, w, cutoff, ref, k, resp, out);
        bool zero = out[0] == 0.0;
        if (gga) zero = zero && out[1] == 0.0 && out[2] == 0.0 && out[3] == 0.0;
        expect(zero, "RKS point at n < cutoff is skipped");
    }

    // => UKS masks: rho_a + rho_b, separate LDA (>) and GGA (>=) blocks <= //
    double v2rho2[3], v2rhosigma[6], v2sigma2[6], vsigma[3];
    for (auto& x : v2rho2) x = r();
    for (auto& x : v2rhosigma) x = r();
    for (auto& x : v2sigma2) x = r();
    for (auto& x : vsigma) x = r();
    const double resp_b[4] = {r(), r(), r(), r()};
    double out_a[4], out_b[4], lda_a[4], lda_b[4];
    {
        double ref_a[4] = {0.25 * cutoff, r(), r(), r()};
        double ref_b[4] = {0.75 * cutoff, r(), r(), r()};
        expect(ref_a[0] + ref_b[0] == cutoff, "UKS test density sums exactly to the cutoff");
        pack_uks(false, w, cutoff, ref_a, ref_b, v2rho2, v2rhosigma, v2sigma2, vsigma, resp, resp_b, lda_a, lda_b);
        expect(lda_a[0] == 0.0 && lda_b[0] == 0.0, "UKS LDA block at rho == cutoff is skipped");
        pack_uks(true, w, cutoff, ref_a, ref_b, v2rho2, v2rhosigma, v2sigma2, vsigma, resp, resp_b, out_a, out_b);
        // Only the GGA terms survive at equality: compare with the full GGA output just above.
        double above_a[4], above_b[4];
        // (One ulp of rho_a is below the resolution of the sum, so step by a relative 1e-12.)
        double up_a[4] = {ref_a[0] * (1.0 + 1.e-12), ref_a[1], ref_a[2], ref_a[3]};
        expect(up_a[0] + ref_b[0] > cutoff, "UKS test density above the cutoff");
        pack_uks(true, w, cutoff, up_a, ref_b, v2rho2, v2rhosigma, v2sigma2, vsigma, resp, resp_b, above_a, above_b);
        const double lda_part_a = w * (v2rho2[0] * resp[0] + v2rho2[1] * resp_b[0]);
        const double lda_part_b = w * (v2rho2[2] * resp_b[0] + v2rho2[1] * resp[0]);
        expect(std::fabs(out_a[0] - (above_a[0] - lda_part_a)) < 1.e-14, "UKS alpha GGA-only scalar at equality");
        expect(std::fabs(out_b[0] - (above_b[0] - lda_part_b)) < 1.e-14, "UKS beta GGA-only scalar at equality");
        expect(out_a[1] != 0.0 && out_b[3] != 0.0, "UKS GGA vector terms kept at equality");
        double down_a[4] = {ref_a[0] * (1.0 - 1.e-12), ref_a[1], ref_a[2], ref_a[3]};
        expect(down_a[0] + ref_b[0] < cutoff, "UKS test density below the cutoff");
        pack_uks(true, w, cutoff, down_a, ref_b, v2rho2, v2rhosigma, v2sigma2, vsigma, resp, resp_b, out_a, out_b);
        bool zero = true;
        for (int c = 0; c < 4; ++c) zero = zero && out_a[c] == 0.0 && out_b[c] == 0.0;
        expect(zero, "UKS point below the cutoff is skipped");
    }

    // => Triplet: project_triplet + pack_rks == half the UKS alpha output for (u, -u) <= //
    for (bool gga : {false, true}) {
        // Closed-shell symmetric polarized derivatives.
        const double raa = r(), rab = r();
        const double s0 = r(), s1 = r(), s2 = r();  // a_aa = b_bb, a_ab = b_ab, a_bb = b_aa
        const double q0 = r(), q1 = r(), q2 = r(), q3 = r();  // aa_aa = bb_bb, aa_ab = ab_bb, aa_bb, ab_ab
        const double vaa = r(), vab = r();
        const double p_v2rho2[3] = {raa, rab, raa};
        const double p_v2rhosigma[6] = {s0, s1, s2, s2, s1, s0};
        const double p_v2sigma2[6] = {q0, q1, q2, q3, q1, q0};
        const double p_vsigma[3] = {vaa, vab, vaa};
        const RKSKernel t = project_triplet(p_v2rho2, p_v2rhosigma, p_v2sigma2, p_vsigma, gga);
        const double ref[4] = {0.3, r(), r(), r()};
        const double u[4] = {r(), r(), r(), r()};
        const double minus_u[4] = {-u[0], -u[1], -u[2], -u[3]};
        double rks[4] = {0, 0, 0, 0}, ua[4] = {0, 0, 0, 0}, ub[4] = {0, 0, 0, 0};
        pack_rks(gga, w, cutoff, ref, t, u, rks);
        pack_uks(gga, w, cutoff, ref, ref, p_v2rho2, p_v2rhosigma, p_v2sigma2, p_vsigma, u, minus_u, ua, ub);
        const int n = gga ? 4 : 1;
        double err = 0.0;
        for (int c = 0; c < n; ++c) err = std::fmax(err, std::fabs(rks[c] - 0.5 * ua[c]));
        std::printf("triplet %s: max |rks - uks_alpha/2| = %.3e\n", gga ? "GGA" : "LDA", err);
        expect(err < 1.e-14, "triplet projection equals half the UKS alpha output");
    }

    std::printf(failures ? "%d FAILURE(S)\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}
