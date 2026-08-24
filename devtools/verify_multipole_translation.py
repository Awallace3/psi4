"""Verify the rank-mixing multipole translation coefficient used at
atomic_polarizability.cc:4844-4846:

    coefficient = sqrt( binom(l - m, l' - m') * binom(l + m, l' + m') )

against the solid-harmonic addition theorem
    C_lm(a + b) = sum_{l'm'} coefficient * C_{l'm'}(a) * C_{l-l', m-m'}(b)
using an independent scipy Y_lm.
"""
import numpy as np
from math import comb
from scipy.special import sph_harm_y as SH

LMAX = 3


def C(l, m, v):
    """Complex Racah regular solid harmonic C_lm(v) = sqrt(4pi/(2l+1)) Y_lm(vhat) |v|^l."""
    if abs(m) > l:
        return 0.0 + 0.0j
    r = np.linalg.norm(v)
    if r == 0.0:
        return complex(1.0) if l == 0 else 0.0 + 0.0j
    th = np.arccos(v[2] / r)
    ph = np.arctan2(v[1], v[0])
    return np.sqrt(4 * np.pi / (2 * l + 1)) * SH(l, m, th, ph) * r**l


rng = np.random.default_rng(11)
worst = 0.0
worst_lbl = None
for _ in range(200):
    a = rng.normal(size=3) * 1.3          # P - S
    b = rng.normal(size=3) * 0.9          # S - T
    for l in range(LMAX + 1):
        for m in range(-l, l + 1):
            direct = C(l, m, a + b)
            acc = 0.0 + 0.0j
            for lp in range(l + 1):
                for mp in range(-lp, lp + 1):
                    dl, dm = l - lp, m - mp
                    if abs(dm) > dl:
                        continue
                    # exactly the code's coefficient
                    coef = np.sqrt(comb(l - m, lp - mp) * comb(l + m, lp + mp))
                    acc += coef * C(lp, mp, a) * C(dl, dm, b)
            err = abs(direct - acc)
            scale = max(abs(direct), 1.0)
            if err / scale > worst:
                worst, worst_lbl = err / scale, (l, m)

print(f"addition theorem, ranks 0-{LMAX}, 200 random (a,b) pairs")
print(f"worst relative deviation : {worst:.3e}   at (l,m) = {worst_lbl}")
print("VERDICT:", "coefficient CONFIRMED" if worst < 1e-12 else "<<< MISMATCH")

# comb(l-m, l'-m') is only nonzero for l'-m' in [0, l-m]; confirm the code's
# implicit selection rule matches the theorem's (no term dropped, none extra).
bad = [(l, m, lp, mp)
       for l in range(LMAX + 1) for m in range(-l, l + 1)
       for lp in range(l + 1) for mp in range(-lp, lp + 1)
       if abs(m - mp) <= l - lp
       and (comb(l - m, lp - mp) == 0) != (not (0 <= lp - mp <= l - m))]
print("selection-rule mismatches :", len(bad))
