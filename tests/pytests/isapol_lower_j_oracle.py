"""Table-independent even-(L+H+J) lower-J oracle; LOWER_J_VALIDATION.md.

Expected values use only factorial angular algebra and synthetic arrays.
CamCASP interface conventions: Alston J. Misquitta and Anthony J. Stone,
Copyright (c) 2019 Anthony Stone, MIT: see RECOUPLED_CAMCASP_LICENSE.
New implementation: Copyright (c) 2026 The Psi4 Developers, LGPL-3.0-only.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math

import numpy as np

from isapol_factorial_oracle import cg, independent_g


@lru_cache(maxsize=32768)
def _exact_cg(a, m, b, n, j, M):
    """Return r,s with CG=r*sqrt(s); factorial sum is exact rational."""
    assert 0<=a<=6 and 0<=b<=6 and 0<=j<=10
    if m+n != M or abs(m)>a or abs(n)>b or abs(M)>j or not abs(a-b)<=j<=a+b:
        return Fraction(0), Fraction(1)
    f = math.factorial
    s = Fraction((2*j+1)*f(a+b-j)*f(a-b+j)*f(-a+b+j), f(a+b+j+1))
    s *= math.prod(f(x) for x in (a+m,a-m,b+n,b-n,j+M,j-M))
    r = Fraction(0)
    for z in range(a+b+j+1):
        den = (z,a+b-j-z,a-m-z,b+n-z,j-b+m+z,j-a-n+z)
        if min(den)>=0:
            r += Fraction((-1)**z, math.prod(f(x) for x in den))
    return r, s


def _rational_sqrt(x):
    a, b = math.isqrt(x.numerator), math.isqrt(x.denominator)
    assert a*a == x.numerator and b*b == x.denominator
    return Fraction(a, b)


@lru_cache(maxsize=8192)
def overlap_exact(l, p, k, q, L, H, J, M=None):
    """Overlap of normalized [(lk)K(pq)P]J and [(lp)L(kq)H]J.

    A sparse magnetic sum, not a four-index projection tensor. Return r,s
    with overlap=r*sqrt(s). Exact cancellation decides structural zeros;
    numerical magnitude is NEVER used to infer support. M defaults to J.
    Bounds cover all rank<=3, C6..C12 channels and orthogonality checks.
    """
    assert all(type(x) is int and 1<=x<=3 for x in (l,p,k,q))
    assert 0<=L<=6 and 0<=H<=6 and 0<=J<=10
    M = J if M is None else M
    assert type(M) is int and abs(M)<=J
    K, P = l+k, p+q
    if not (abs(l-p)<=L<=l+p and abs(k-q)<=H<=k+q
            and abs(K-P)<=J<=K+P and abs(L-H)<=J<=L+H):
        return Fraction(0), Fraction(1)
    total, base = Fraction(0), None
    for m, n, r in product(range(-l,l+1), range(-p,p+1), range(-k,k+1)):
        s = M-m-n-r
        if abs(s)>q:
            continue
        args = ((l,m,k,r,K,m+r), (p,n,q,s,P,n+s), (K,m+r,P,n+s,J,M),
                (l,m,p,n,L,m+n), (k,r,q,s,H,r+s), (L,m+n,H,r+s,J,M))
        coeff, rad = Fraction(1), Fraction(1)
        for arg in args:
            x, y = _exact_cg(*arg)
            coeff *= x
            if not coeff:
                break
            rad *= y
        if not coeff:
            continue
        if base is None:
            base = rad
        # All magnetic terms share one radical up to a rational square.
        # Assert the algebraic property rather than approximating a square root.
        total += coeff*_rational_sqrt(rad/base)
    return total, Fraction(1) if base is None else base


def overlap(l, p, k, q, L, H, J, M=None):
    r, s = overlap_exact(l,p,k,q,L,H,J,M)
    return float(r)*math.sqrt(float(s))


def channels(ranks_a=(1,2,3), ranks_b=(1,2,3)):
    """Every triangle-allowed lower-J channel with nonzero harmonic product.

    Yield (l,p,k,q,L,H,J). Odd normalization is included for classification,
    never silently removed as a supposed structural zero.
    """
    for l,p,k,q in product(ranks_a,ranks_a,ranks_b,ranks_b):
        total = l+p+k+q
        if total>10:
            continue
        for L in range(abs(l-p),l+p+1):
            for H in range(abs(k-q),k+q+1):
                for J in range(abs(l+k-p-q), total, 2):
                    if abs(L-H)<=J<=L+H:
                        yield l,p,k,q,L,H,J


def classify(ch):
    l,p,k,q,L,H,J = ch
    if not overlap_exact(*ch)[0]:
        return 'overlap_zero'
    if (l==p and L%2) or (k==q and H%2):
        return 'reciprocal_zero'
    if (L+H+J)%2:
        return 'odd_normalization_unresolved'
    return 'supported'


def reciprocal_raw_exact(pair_a, pair_b, L, H, J):
    """Exact raw-B coefficient for one observable reciprocal rank-pair class.

    Coupled partners obey a(p,l)=(-1)^(l+p-L) a(l,p). Sum those
    partners BEFORE deciding cancellation. No Sbar normalization is used,
    so this also diagnoses genuinely surviving odd-parity channels.
    """
    assert tuple(sorted(pair_a))==tuple(pair_a) and tuple(sorted(pair_b))==tuple(pair_b)
    total, base = Fraction(0), None
    for (l,p),(k,q) in product(sorted({pair_a,pair_a[::-1]}), sorted({pair_b,pair_b[::-1]})):
        if (l==p and L%2) or (k==q and H%2):
            continue
        r,s = overlap_exact(l,p,k,q,L,H,J)
        x,y = _exact_cg(l+k,0,p+q,0,J,0)
        r *= x*(-1)**(k+q)
        if (l,p)!=pair_a:
            r *= (-1)**(l+p-L)
        if (k,q)!=pair_b:
            r *= (-1)**(k+q-H)
        if not r:
            continue
        s *= y*math.comb(2*(l+k),2*l)*math.comb(2*(p+q),2*p)
        if base is None:
            base = s
        total += r*_rational_sqrt(s/base)
    return total, Fraction(1) if base is None else base


def factor(l,p,k,q,L,H,J):
    """Radial sign * harmonic CG * overlap / Sbar phase * Sbar CG.

    This API deliberately rejects the singular odd-parity normalization.
    """
    assert all(type(x) is int and 1<=x<=3 for x in (l,p,k,q))
    assert 0<=L<=6 and 0<=H<=6 and 0<=J<=10
    if (L+H+J)%2:
        raise ValueError('odd L+H+J: Sbar zero-m normalization unresolved')
    denom = cg(L,0,H,0,J,0)
    assert denom != 0
    radial = math.sqrt(math.comb(2*(l+k),2*l)*math.comb(2*(p+q),2*p))
    phase = (-1)**((L-H-J)//2)
    return ((-1)**(k+q)*phase*radial*cg(l+k,0,p+q,0,J,0)
            * overlap(l,p,k,q,L,H,J)*denom)


@lru_cache(maxsize=9)
def angular_map(l,p):
    assert l in (1,2,3) and p in (1,2,3)
    g = independent_g(l,p)
    g.setflags(write=False)
    return g


def couple(matrices, ranks):
    """Bounded, compressed first stage computed independently from input."""
    assert ranks and tuple(sorted(set(ranks))) == tuple(ranks)
    assert set(ranks)<={1,2,3}
    a = np.asarray(matrices)
    d = sum(2*l+1 for l in ranks)
    assert a.ndim==3 and 1<=len(a)<=8 and a.shape[1:]==(d,d)
    assert np.isfinite(a).all() and np.isrealobj(a)
    assert np.array_equal(a,a.transpose(0,2,1))
    offsets, start = {}, 0
    for l in ranks:
        offsets[l] = slice(start,start+2*l+1)
        start += 2*l+1
    return {(l,p): np.einsum('tab,fab->ft', angular_map(l,p), a[:,offsets[l],offsets[p]])
            for l,p in product(ranks,repeat=2)}


def lower_j_coefficients(A,ranks_a,B,ranks_b,weights):
    """All even-parity triangle candidate rows, including exact zero blocks.

    Candidate rows can be absent from production when structurally zero.
    No production row enumeration or value is consumed by this calculation.
    """
    aa, bb = couple(A,ranks_a), couple(B,ranks_b)
    w = np.asarray(weights)
    assert w.shape==(len(A),) and len(A)==len(B)
    assert np.isfinite(w).all() and (w>=0).all() and (w>0).any()
    active = w>0
    blocks = {}
    for ch in channels(ranks_a,ranks_b):
        l,p,k,q,L,H,J = ch
        if (L+H+J)%2:
            continue
        key = (l+p+k+q+2,L,H,J)
        if key not in blocks:
            blocks[key] = np.zeros((2*L+1,2*H+1))
        if classify(ch) != 'supported':
            continue
        x = aa[l,p][active,L*L:(L+1)**2]
        y = bb[k,q][active,H*H:(H+1)**2]
        integral = np.einsum('f,ft,fu->tu',w[active],x,y)
        assert np.max(np.abs(integral.imag))<1e-12
        blocks[key] += factor(*ch)*integral.real
    return {(n,L*L+t+1,H*H+u+1,J): block[t,u]
            for (n,L,H,J),block in blocks.items() for t,u in np.ndindex(block.shape)}
