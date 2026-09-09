"""Raw-B identities ONLY: not an oracle for production odd-J coefficients.

See ODD_J_VALIDATION.md. No Psi4, tables, archives, or home-tree reads.
CamCASP conventions: Misquitta / Stone, MIT notice in RECOUPLED_CAMCASP_LICENSE.
New implementation: Copyright (c) 2026 The Psi4 Developers, LGPL-3.0-only.
"""
from fractions import Fraction
from itertools import product
import math

import numpy as np
import pytest

from isapol_factorial_oracle import cg
from isapol_lower_j_oracle import (
    _exact_cg, channels, couple, factor, overlap, overlap_exact, reciprocal_raw_exact,
)


def value(rs):
    r, s = rs
    return float(r)*math.sqrt(float(s))


def same_exact(a, b, sign=1):
    r, s = a
    t, u = b
    assert r*r*s == t*t*u
    assert (r > 0)-(r < 0) == sign*((t > 0)-(t < 0))


def odd_classes():
    return sorted({(tuple(sorted(ch[:2])), tuple(sorted(ch[2:4])), *ch[4:])
                   for ch in channels() if sum(ch[4:]) % 2})


def raw_ordered(l, p, k, q, L, H, J):
    """Real coefficient of B, with NO normalization or CP phase conversion."""
    r, s = overlap_exact(l, p, k, q, L, H, J)
    x, y = _exact_cg(l+k, 0, p+q, 0, J, 0)
    return (r*x*(-1)**(k+q),
            s*y*math.comb(2*(l+k), 2*l)*math.comb(2*(p+q), 2*p))


def test_odd_raw_exact_denominator_and_site_exchange():
    classes = odd_classes()
    assert len(classes) == 394
    surviving = 0
    for a, b, L, H, J in classes:
        assert _exact_cg(L, 0, H, 0, J, 0)[0] == 0
        with pytest.raises(ValueError, match='normalization unresolved'):
            factor(*a, *b, L, H, J)
        # Exchange both molecular sites: stretched interaction CG exchanges
        # have phase +1; the outer response CG contributes (-1)^(L+H-J).
        # Including the radial sign gives (-1)^(L+H) on product parity.
        for x, y in product({a, a[::-1]}, {b, b[::-1]}):
            same_exact(raw_ordered(*x, *y, L, H, J),
                       raw_ordered(*y, *x, H, L, J), (-1)**(L+H))
        r = reciprocal_raw_exact(a, b, L, H, J)
        same_exact(r, reciprocal_raw_exact(b, a, H, L, J), (-1)**(L+H))
        surviving += bool(r[0])
    assert surviving == 182


@pytest.mark.parametrize('ranks,J', [((1, 2, 1, 3), 3),
                                    ((1, 2, 2, 3), 4),
                                    ((3, 2, 2, 3), 6)])
def test_odd_raw_pointwise_tree_resolution(ranks, J):
    # Stronger than norm-only completeness: resolve each magnetic component
    # of the first tree in the alternate basis, including the odd sector.
    l, p, k, q = ranks
    projections = [(L, H, overlap(*ranks, L, H, J))
                   for L in range(abs(l-p), l+p+1)
                   for H in range(abs(k-q), k+q+1) if abs(L-H) <= J <= L+H]
    direct, resolved, even_only = [], [], []
    for m, n, r in product(range(-l, l+1), range(-p, p+1), range(-k, k+1)):
        s = J-m-n-r
        if abs(s) > q:
            continue
        direct.append(cg(l, m, k, r, l+k, m+r)*cg(p, n, q, s, p+q, n+s)
                      *cg(l+k, m+r, p+q, n+s, J, J))
        terms, even = [], []
        for L, H, R in projections:
            if abs(m+n) > L or abs(r+s) > H:
                continue
            term = R*cg(l, m, p, n, L, m+n)*cg(k, r, q, s, H, r+s)
            term *= cg(L, m+n, H, r+s, J, J)
            terms.append(term)
            if (L+H+J) % 2 == 0:
                even.append(term)
        resolved.append(math.fsum(terms))
        even_only.append(math.fsum(even))
    np.testing.assert_allclose(resolved, direct, rtol=3e-13, atol=3e-12)
    assert np.max(np.abs(np.array(even_only)-direct)) > 1e-3
    # Exact odd-sector norm is positive: not a floating residual artifact.
    odd_norm = Fraction(0)
    for L, H, _ in projections:
        if (L+H+J) % 2:
            r, s = overlap_exact(*ranks, L, H, J)
            odd_norm += r*r*s
    assert 0 < odd_norm < 1


def synthetic(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, 15, 15))/math.sqrt(15)
    a = x@x.transpose(0, 2, 1)+np.eye(15)
    assert np.array_equal(a, a.transpose(0, 2, 1))
    assert np.linalg.eigvalsh(a).min() > 0
    return a


def test_odd_raw_reciprocal_synthetic_phase_and_compression():
    # Compare ordered raw sums to exact reciprocal-class sums, NOT to C_n.
    A, B = synthetic(811), synthetic(929)
    w = np.array([0., .17, .43])  # already CP weights; no further factor
    aa, bb = couple(A, (1, 2, 3)), couple(B, (1, 2, 3))
    compressed = {}
    offsets = {1: 0, 2: 3, 3: 8}
    for tag, arrays in (('A', A), ('B', B)):
        for pair in ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)):
            ranks = tuple(sorted(set(pair)))
            ix = [i for l in ranks for i in range(offsets[l], offsets[l]+2*l+1)]
            compressed[tag, pair] = couple(arrays[:, ix][:, :, ix], ranks)
    blocks, orders, witnesses = set(), set(), set()
    count = 0
    for a, b, L, H, J in odd_classes():
        exact = reciprocal_raw_exact(a, b, L, H, J)
        if not exact[0]:
            continue  # exact raw cancellation, never magnitude-based filtering
        x = aa[a][1:, L*L:(L+1)**2]
        y = bb[b][1:, H*H:(H+1)**2]
        cp = np.einsum('f,ft,fu->tu', w[1:], x, y)
        np.testing.assert_allclose(cp.real, 0., rtol=0., atol=3e-12)
        assert np.max(np.abs(cp.imag)) > 1e-8
        canonical = value(exact)*cp
        ordered = np.zeros_like(canonical)
        for pa, pb in product({a, a[::-1]}, {b, b[::-1]}):
            ca, cb = compressed['A', a][pa], compressed['B', b][pb]
            integral = np.einsum('f,ft,fu->tu', w[1:],
                                 ca[1:, L*L:(L+1)**2], cb[1:, H*H:(H+1)**2])
            ordered += value(raw_ordered(*pa, *pb, L, H, J))*integral
        np.testing.assert_allclose(ordered, canonical, rtol=3e-13, atol=3e-12)
        # i*CP is real, but that fact does NOT fix the coefficient convention:
        # -i*CP is equally real. Both fail if CP was discarded as real-only.
        np.testing.assert_allclose((1j*cp).imag, 0., rtol=0., atol=3e-12)
        if (sum(b)-H) % 2:
            wrong = np.einsum('f,ft,fu->tu', w[1:], x, y.conj())
            np.testing.assert_allclose(wrong, -cp, rtol=3e-13, atol=3e-12)
            witnesses.add('B imaginary')
        else:
            witnesses.add('A imaginary')
        blocks.add((sum(a+b)+2, L, H, J))
        orders.add((sum(a+b)+2) % 2)
        count += 1
    assert count == 182 and len(blocks) == 80
    assert orders == {0, 1} and witnesses == {'A imaginary', 'B imaginary'}


def test_odd_raw_standalone_import():
    import subprocess
    import sys
    from pathlib import Path
    script = ('import sys; sys.path.insert(0,'+repr(str(Path(__file__).parent))+'); '
              'from isapol_lower_j_oracle import reciprocal_raw_exact; '
              'assert reciprocal_raw_exact((1,2),(1,3),2,2,3)[0]; '
              'assert not any(k=="psi4" or k.startswith("psi4.") for k in sys.modules)')
    subprocess.run([sys.executable, '-I', '-c', script], check=True)
