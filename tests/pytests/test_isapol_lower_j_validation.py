"""Portable lower-J mathematics and optional staged-core comparisons.

No Psi4 import in the algebra/expected path. Core tests import it only inside
actual_coefficients; they require the parent's staged API, with no binding skips.
Copyright (c) 2026 The Psi4 Developers. LGPL-3.0-only.
"""
from collections import Counter
from itertools import product
import math

import numpy as np
import pytest

from isapol_factorial_oracle import cg
from isapol_lower_j_oracle import (channels, classify, couple, factor, overlap,
                                   overlap_exact, lower_j_coefficients, reciprocal_raw_exact)

WEIGHTS = (0., .17, .43)


def inputs(ranks, seed):
    d = sum(2*l+1 for l in ranks)
    rng = np.random.default_rng(seed)
    arrays = []
    for f in range(3):
        x = rng.normal(size=(d,d))/math.sqrt(d)
        arrays.append(x@x.T+(1+f/4)*np.eye(d))
    a = np.array(arrays)
    assert np.array_equal(a,a.transpose(0,2,1))
    assert np.linalg.eigvalsh(a).min()>0
    return a


def actual_coefficients(A,ranks_a,B,ranks_b):
    import psi4  # Actual-under-test ONLY.
    c = psi4.core

    def model(a,ranks,label):
        s = c.IsaAnisotropicSite()
        s.label, s.ranks, s.origin, s.frame = label,list(ranks),[0.,0.,0.],np.eye(3)
        s.responses = [c.Matrix.from_array(x) for x in a]
        return c.IsaRecoupledModel(c.IsaAnisotropicModel([0.,.6,1.9],[s],
            'supplied_local_response','independent lower-J synthetic input'))

    pair = c.isa_recoupled_dispersion(model(A,ranks_a,'A'),model(B,ranks_b,'B'),
                                     list(WEIGHTS),12).pairs[0]
    rows = {(x.order,x.t,x.u,x.J):x.value for x in pair.coefficients}
    assert len(rows)==len(pair.coefficients)
    return rows


def selected(key):
    n,t,u,J = key
    return J<n-2 and (math.isqrt(t-1)+math.isqrt(u-1)+J)%2==0


def compare(actual,expected,required=()):
    assert set(required)<=actual.keys(), 'missing required structural row'
    view = {k:v for k,v in actual.items() if selected(k)}
    assert view.keys()<=expected.keys(), 'unexplained actual even-parity row'
    keys = sorted(expected)
    np.testing.assert_allclose([view.get(k,0.) for k in keys], [expected[k] for k in keys],
                               rtol=3e-13,atol=3e-12)


def test_lower_j_algebra_structural_partition():
    all_channels = list(channels())
    assert len(all_channels)==len(set(all_channels))==2367
    assert Counter(map(classify,all_channels)) == {
        'supported':1191, 'overlap_zero':136, 'reciprocal_zero':460,
        'odd_normalization_unresolved':580}
    # Independent brute force domain: no table metadata, no numeric cutoff.
    brute = {(*r,L,H,J) for r in product(range(1,4),repeat=4) if sum(r)<=10
             for L,H,J in product(range(7),range(7),range(11))
             if abs(r[0]-r[1])<=L<=r[0]+r[1] and abs(r[2]-r[3])<=H<=r[2]+r[3]
             and abs(r[0]+r[2]-r[1]-r[3])<=J<sum(r) and (sum(r)+J)%2==0
             and abs(L-H)<=J<=L+H}
    assert set(all_channels)==brute
    for ch in all_channels:
        if (sum(ch[4:]))%2:
            with pytest.raises(ValueError,match='normalization unresolved'):
                factor(*ch)
    # The odd set is genuinely nonempty AFTER both exact overlap zeros and
    # identical-rank reciprocal zeros have been removed.
    assert any(classify(ch)=='odd_normalization_unresolved' for ch in all_channels)


def test_lower_j_algebra_overlap_completeness_and_projection_invariance():
    # Complete resolution of the identity in the alternate (L,H) basis:
    # sum_LH |<tree1|tree2>|^2=1. Include odd channels here, too.
    # Exact rational squares make this a proof at every finite tested spin,
    # not a comparison with the same floating-point implementation.
    from fractions import Fraction
    for l,p,k,q in product(range(1,4),repeat=4):
        if l+p+k+q>10:
            continue
        for J in range(abs(l+k-p-q),min(l+p+k+q,10)+1):
            total = Fraction(0)
            for L in range(abs(l-p),l+p+1):
                for H in range(abs(k-q),k+q+1):
                    r,s = overlap_exact(l,p,k,q,L,H,J)
                    total += r*r*s
            assert total==1
    # Different M changes the actual summands, providing an independent
    # realization of each overlap for a representative nonstretched quartet.
    for l,p,k,q in ((1,2,2,3),(3,1,2,2),(1,1,1,1)):
        for L in range(abs(l-p),l+p+1):
            for H in range(abs(k-q),k+q+1):
                for J in range(max(abs(l+k-p-q),abs(L-H)),min(l+p+k+q,L+H)+1):
                    a,s = overlap_exact(l,p,k,q,L,H,J)
                    b,t = overlap_exact(l,p,k,q,L,H,J,0)
                    assert a*a*s==b*b*t and (a>0)-(a<0)==(b>0)-(b<0)


def test_lower_j_algebra_reciprocal_cancellation_vs_missing_normalization():
    classes = {(tuple(sorted(ch[:2])),tuple(sorted(ch[2:4])),*ch[4:]) for ch in channels()}
    counts = Counter()
    surviving_odd_blocks = set()
    cancelling_odd_classes = 0
    for a,b,L,H,J in sorted(classes):
        r,s = reciprocal_raw_exact(a,b,L,H,J)
        parity = 'odd' if (L+H+J)%2 else 'even'
        counts[parity,'survives' if r else 'zero'] += 1
        if r and parity=='odd':
            surviving_odd_blocks.add((sum(a+b)+2,L,H,J))
        if not r and any(classify((*x,*y,L,H,J))=='odd_normalization_unresolved'
                         for x in {a,a[::-1]} for y in {b,b[::-1]}):
            cancelling_odd_classes += 1
    assert len(classes)==1073
    assert counts=={('even','survives'):516, ('even','zero'):163,
                    ('odd','survives'):182, ('odd','zero'):212}
    assert len(surviving_odd_blocks)==80 and cancelling_odd_classes==16
    # Nonzero ordered overlaps can cancel only after reciprocal summation.
    assert classify((1,2,1,2,1,2,2))=='odd_normalization_unresolved'
    assert reciprocal_raw_exact((1,2),(1,2),1,2,2)[0]==0
    # Conversely these raw coefficients survive: undefined Sbar is NOT absence.
    assert reciprocal_raw_exact((1,2),(1,3),2,2,3)[0]!=0


def test_lower_j_algebra_standalone_import_and_direct_float_sum():
    import subprocess
    import sys
    from pathlib import Path
    script = ('import sys; sys.path.insert(0,'+repr(str(Path(__file__).parent))+'); '
              'import isapol_lower_j_oracle as o; '
              'assert not any(k=="psi4" or k.startswith("psi4.") for k in sys.modules); '
              'assert o.factor(1,1,1,1,0,0,0)>0')
    subprocess.run([sys.executable,'-I','-c',script],check=True)
    with pytest.raises(AssertionError):
        overlap(4,1,1,1,3,2,3)
    with pytest.raises(AssertionError):
        factor(1,1,1,1,0,0,11)
    with pytest.raises(AssertionError):
        couple(np.array([np.eye(3)]*9),(1,))
    # Separate floating factorial helper versus new exact-rational factorial
    # implementation. Orthogonality above is the non-formula-duplication check.
    for ch in ((1,1,1,1,0,0,0),(1,2,1,2,2,2,2),(1,2,1,3,2,2,3),
               (3,2,2,3,3,4,6),(3,3,1,2,6,3,5)):
        l,p,k,q,L,H,J = ch
        terms = []
        for m,n,r in product(range(-l,l+1),range(-p,p+1),range(-k,k+1)):
            s = J-m-n-r
            if abs(s)>q or abs(m+n)>L or abs(r+s)>H:
                continue
            terms.append(cg(l,m,k,r,l+k,m+r)*cg(p,n,q,s,p+q,n+s)
                         *cg(l+k,m+r,p+q,n+s,J,J)*cg(l,m,p,n,L,m+n)
                         *cg(k,r,q,s,H,r+s)*cg(L,m+n,H,r+s,J,J))
        assert math.fsum(terms)==pytest.approx(overlap(*ch),rel=3e-14,abs=3e-15)


def test_lower_j_algebra_scalar_isotropic_and_imaginary_channels():
    ranks = (1,2,3)
    sa,sb = (2.,3.,5.),(7.,11.,13.)
    A = np.array([np.diag(np.repeat(sa,(3,5,7)))]*3)
    B = np.array([np.diag(np.repeat(sb,(3,5,7)))]*3)
    out = lower_j_coefficients(A,ranks,B,ranks,WEIGHTS)
    for n in range(6,13):
        scalar = sum(math.comb(2*l+2*k,2*l)*sum(WEIGHTS)*sa[l-1]*sb[k-1]
                     for l,k in product(ranks,repeat=2) if 2*l+2*k+2==n)
        assert out.get((n,1,1,0),0.)==pytest.approx(scalar,rel=3e-14,abs=3e-12)
    np.testing.assert_allclose([v for (n,t,u,J),v in out.items() if (t,u,J)!=(1,1,0)],
                               0.,atol=3e-12,rtol=0.)
    aa = couple(inputs(ranks,101),ranks)
    # l+p-L odd mixed-rank channels need not vanish for symmetric full arrays.
    x = aa[1,2][:,4:9]
    assert np.max(np.abs(x.imag))>.1
    np.testing.assert_allclose(x.real,0.,atol=1e-15)
    np.testing.assert_allclose(aa[2,1][:,4:9],-x,atol=1e-15)
    assert classify((1,2,1,2,2,2,2))=='supported'
    assert factor(1,2,1,2,2,2,2)!=0.
    assert factor(1,1,1,1,0,0,0)==pytest.approx(2.,rel=2e-15)


def test_lower_j_core_full_spd_and_mutations():
    ranks = (1,2,3)
    A,B = inputs(ranks,101),inputs(ranks,202)
    expected = lower_j_coefficients(A,ranks,B,ranks,WEIGHTS)
    actual = actual_coefficients(A,ranks,B,ranks)
    compare(actual,expected)
    assert len(expected)==10256
    assert Counter(k[0] for k in expected)=={6:79,7:220,8:533,9:1018,10:1876,11:2748,12:3782}
    supported_blocks = {(sum(ch[:4])+2,*ch[4:]) for ch in channels() if classify(ch)=='supported'}
    required = {(n,L*L+t+1,H*H+u+1,J) for n,L,H,J in supported_blocks
                for t,u in product(range(2*L+1),range(2*H+1))}
    # Required structural rows cannot disappear even when a numerical entry is zero.
    assert len(supported_blocks)==224 and len(required)==10056
    compare(actual,expected,required)
    present = {k for k in actual if selected(k)}
    assert len(present)==10074
    missing = expected.keys()-present
    assert len(missing)==182
    assert {(n,math.isqrt(t-1),math.isqrt(u-1),J) for n,t,u,J in missing}=={
        (11,3,6,5),(11,6,3,5)}
    assert all(expected[k]==0. for k in missing)
    # The two additional production blocks are same-rank antisymmetric zeros.
    assert {(n,math.isqrt(t-1),math.isqrt(u-1),J) for n,t,u,J in present-required}=={
        (6,1,1,0),(6,1,1,2)}
    # Observe actual odd scope without inferring its normalization from values.
    assert sum(k[3]<k[0]-2 and not selected(k) for k in actual)==5341
    compare(actual_coefficients(2*A,ranks,3*B,ranks),{k:6*v for k,v in expected.items()})
    blocks = {(n,math.isqrt(t-1),math.isqrt(u-1),J) for n,t,u,J in expected}
    assert len(blocks)==228
    sensitive_blocks = set()
    for n,L,H,J in blocks:
        keys = [k for k in expected if (k[0],math.isqrt(k[1]-1),math.isqrt(k[2]-1),k[3])==(n,L,H,J)]
        if not any(abs(expected[k])>1e-8 for k in keys):
            continue  # Sensitivity criterion only; never removes oracle outputs.
        sensitive_blocks.add((n,L,H,J))
        for scale in (0.,-1.,1.01):
            mutant = actual.copy()
            for key in keys:
                mutant[key] = scale*actual.get(key,0.)
            with pytest.raises(AssertionError):
                compare(mutant,expected)
    assert sensitive_blocks==supported_blocks
    missing_required = actual.copy()
    del missing_required[next(iter(required))]
    with pytest.raises(AssertionError,match='missing required structural row'):
        compare(missing_required,expected,required)


# Every unordered local rank-pair class with C6..C12, including swapped A/B.
ISOLATIONS = tuple((a,b) for a in ((1,1),(1,2),(1,3),(2,2),(2,3),(3,3))
                   for b in ((1,1),(1,2),(1,3),(2,2),(2,3),(3,3)) if sum(a+b)<=10)


def test_lower_j_algebra_isolation_completeness():
    covered = {a+b for pa,pb in ISOLATIONS for a in {pa,pa[::-1]} for b in {pb,pb[::-1]}}
    required = {r for r in product(range(1,4),repeat=4) if sum(r)<=10}
    assert covered==required
    assert len(ISOLATIONS)==33 and len(covered)==76


@pytest.mark.parametrize('pair_a,pair_b',ISOLATIONS)
def test_lower_j_core_isolated_compressed(pair_a,pair_b):
    ra,rb = tuple(sorted(set(pair_a))),tuple(sorted(set(pair_b)))

    def isolate(pair,ranks,seed):
        a = inputs(ranks,seed)
        if pair[0]!=pair[1]:
            # Symmetric, not SPD: erase same-rank blocks to isolate ordered
            # mixed-pair partners exactly, rather than relying on rank sums.
            d = 2*ranks[0]+1
            a[:,:d,:d] = 0.
            a[:,d:,d:] = 0.
        return a

    A,B = isolate(pair_a,ra,317),isolate(pair_b,rb,419)
    expected = lower_j_coefficients(A,ra,B,rb,WEIGHTS)
    actual = actual_coefficients(A,ra,B,rb)
    supported = {(sum(ch[:4])+2,*ch[4:]) for ch in channels(ra,rb)
                 if classify(ch)=='supported'}
    required = {(n,L*L+t+1,H*H+u+1,J) for n,L,H,J in supported
                for t,u in product(range(2*L+1),range(2*H+1))}
    compare(actual,expected,required)
    assert any(abs(v)>1e-8 for k,v in expected.items() if k[0]==sum(pair_a+pair_b)+2)
    if pair_a==pair_b==(1,2):
        # With diagonal blocks erased, both first stages in L=H=2 are
        # imaginary. Conjugation on B reverses precisely this whole block.
        target = [k for k in expected if k[0]==8 and 5<=k[1]<=9 and 5<=k[2]<=9 and k[3]==2]
        assert any(abs(expected[k])>1e-8 for k in target)
        mutant = actual.copy()
        for key in target:
            mutant[key] *= -1
        with pytest.raises(AssertionError):
            compare(mutant,expected)
