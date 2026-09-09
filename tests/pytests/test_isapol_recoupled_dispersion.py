"""Recoupled tensor tests. No home-tree reads, generator imports, or binding skips.

Analytic inputs are independent of CamCASP files. Only the separately named
portable H2O-isagrid archive test uses MIT literal reference numerical records.
No native SCF/PFIT/GRAC protocol claim; no archived J9/10 numeric parity claim.
"""
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import psi4

from isapol_factorial_oracle import cg, real_transform, independent_g

c = psi4.core
ROOT = Path(__file__).parent/'data_isapol/recoupled_h2o_isagrid_l3'
RANK4 = Path(__file__).parent/'data_isapol/recoupled_rank4_casimir'


def local(matrices, ranks=(1,), frequencies=(1.,), label='analytic', origin=(0.,0.,0.), frame=None,
          provenance='independent analytic input'):
    site = c.IsaAnisotropicSite()
    site.label, site.ranks, site.origin = label, list(ranks), list(origin)
    site.frame = np.eye(3) if frame is None else frame
    site.responses = [c.Matrix.from_array(np.array(x, dtype=float)) for x in matrices]
    return c.IsaAnisotropicModel(list(frequencies), [site], 'supplied_local_response', provenance)


def coupled(matrices, **kw):
    return c.IsaRecoupledModel(local(matrices, **kw))


def calculate(a, b=None, weights=(1.,), order=12):
    return c.isa_recoupled_dispersion(a, a if b is None else b, list(weights), order)


DEFINED_PAIRS = [(l,p) for l,p in itertools.product(range(1,5), repeat=2) if l+p <= 6]
UNDEFINED_PAIRS = [(l,p) for l,p in itertools.product(range(1,5), repeat=2) if l+p > 6]


@pytest.mark.parametrize('l,p', DEFINED_PAIRS)
def test_all_thirteen_tables_factorial_phase_and_unitarity(l,p):
    expected = independent_g(l,p)
    actual = np.zeros_like(expected)
    records = c.isapol_realcg_terms(l,p)
    for x in records:
        actual[x.v,x.k,x.q] = x.value
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)
    square = actual[(l-p)**2:].reshape((2*l+1)*(2*p+1),-1)
    np.testing.assert_allclose(square@square.conj().T, np.eye(len(square)), rtol=0, atol=3e-15)
    for L in range(abs(l-p), l+p+1):
        phased = actual[L*L:(L+1)**2]*(1j if (l+p-L)%2 else 1.)
        np.testing.assert_array_equal(phased.imag, 0.)


def test_closed_dipole_and_sine_rank_one_symmetry():
    A = np.array([[2.,.3,-.4],[.3,3.,.8],[-.4,.8,5.]])
    rec = coupled([A])
    expected = {1:-np.trace(A)/math.sqrt(3), 5:(2*A[0,0]-A[1,1]-A[2,2])/math.sqrt(6),
                6:(A[0,1]+A[1,0])/math.sqrt(2), 7:(A[0,2]+A[2,0])/math.sqrt(2),
                8:(A[1,1]-A[2,2])/math.sqrt(2), 9:(A[1,2]+A[2,1])/math.sqrt(2)}
    for t,v in expected.items():
        assert rec.value(0,0,1,1,t) == pytest.approx(v, rel=2e-15, abs=2e-15)
    for t in (2,3,4): assert rec.value(0,0,1,1,t) == 0j
    assert rec.sites[0][0].components == ['00','10','11c','11s','20','21c','21s','22c','22s']


@pytest.mark.parametrize('ranks', [(1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)])
def test_compressed_rank_input_and_ownership(ranks):
    d = sum(2*l+1 for l in ranks)
    raw = np.arange(d*d,dtype=float).reshape(d,d)/29
    A = raw+raw.T
    inp = local([A], ranks=ranks, origin=(1.,2.,3.))
    rec = c.IsaRecoupledModel(inp)
    offset_l = 0
    for l in ranks:
        offset_p = 0
        for p in ranks:
            expected = np.einsum('vkq,kq->v', independent_g(l,p), A[offset_l:offset_l+2*l+1,offset_p:offset_p+2*p+1])
            for t in range((l-p)**2+1,(l+p+1)**2+1):
                assert rec.value(0,0,l,p,t) == pytest.approx(expected[t-1], rel=3e-14, abs=3e-14)
            offset_p += 2*p+1
        offset_l += 2*l+1
    before = rec.sites[0][0].values
    copy = rec.sites[0][0].values
    copy[0] = 100j
    rec.source.sites[0].responses[0].np[:] = 999.
    inp.sites[0].responses[0].np[:] = -999.
    assert rec.sites[0][0].values == before
    assert rec.source.provenance == 'independent analytic input'
    assert rec.source.sites[0].origin == [1.,2.,3.]


def test_frames_metadata_no_rotation_no_distance_and_ordered_pairs():
    A = np.diag([2.,3.,4.])
    rotation = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    a = coupled([A], frame=rotation)
    b = coupled([A])
    assert a.sites[0][0].values == b.sites[0][0].values
    result = calculate(a,b,order=6)  # coincident origins legal
    assert result.model_a.source.sites[0].frame == rotation
    sites = b.source.sites
    s2 = b.source.sites[0]; s2.label = 'second'; sites.append(s2)
    two = c.IsaRecoupledModel(c.IsaAnisotropicModel([1.],sites,'supplied_local_response','two sites'))
    assert [(p.site_a,p.site_b) for p in calculate(two,order=6).pairs] == [(0,0),(0,1),(1,0),(1,1)]
    assert result.pairs[0].coefficient(6,81,81,10) == 0.
    assert not hasattr(result,'truncated_energy')


@pytest.mark.parametrize('order', [6,8,10,12])
def test_isotropic_scalar_combinatorial_formula(order):
    # For isotropic diagonal rank blocks alpha_l I, C_(2l+2k+2) =
    # choose(2l+2k,2l)*CP(alpha_l alpha_k). No energy/R enters.
    scales_a, scales_b = [2.,3.,5.], [7.,11.,13.]
    A = np.diag(np.repeat(scales_a,[3,5,7]))
    B = np.diag(np.repeat(scales_b,[3,5,7]))
    result = calculate(coupled([A],ranks=(1,2,3)),coupled([B],ranks=(1,2,3)),weights=(.7,))
    expected = sum(math.comb(2*l+2*k,2*l)*.7*scales_a[l-1]*scales_b[k-1]
                   for l in range(1,4) for k in range(1,4) if 2*l+2*k+2==order)
    assert result.pairs[0].coefficient(order,1,1,0) == pytest.approx(expected,rel=3e-14)
    if order==12:
        assert not result.pairs[0].coverage[-1].unrestricted_complete
        assert not result.pairs[0].coverage[-1].table_complete


@pytest.mark.parametrize('order', [7,9,11])
def test_mixed_rank_odd_orders_no_conjugation(order):
    rng = np.random.default_rng(512)
    X,Y = rng.normal(size=(15,15)),rng.normal(size=(15,15))
    A,B = X+X.T,Y+Y.T
    def transform(M):
        offsets={1:0,2:3,3:8}
        return {(l,p): np.einsum('vkq,kq->v',independent_g(l,p),
            M[offsets[l]:offsets[l]+2*l+1,offsets[p]:offsets[p]+2*p+1])
            for l in range(1,4) for p in range(1,4)}
    aa,bb = transform(A),transform(B)
    assert any(np.max(np.abs(x.imag))>1 for x in aa.values())
    result = calculate(coupled([A],ranks=(1,2,3)),coupled([B],ranks=(1,2,3)))
    lookup = {(x.order,x.t,x.u,x.J):x.value for x in result.pairs[0].coefficients}
    sensitive = False
    for n,L1,L2,J,terms in c.isapol_recoupling_blocks():
        if n != order: continue
        t,u = L1*L1+1,L2*L2+1
        expected,wrong = 0.,0.
        for term in terms:
            if max(term.la,term.lap,term.lb,term.lbp)>3: continue
            x,y = aa[term.la,term.lap][t-1],bb[term.lb,term.lbp][u-1]
            expected += term.coefficient*(1j**term.ipow*x*y).real
            wrong += term.coefficient*(1j**term.ipow*x*y.conjugate()).real
        assert lookup.get((n,t,u,J),0.) == pytest.approx(expected,rel=3e-13,abs=3e-12)
        sensitive |= abs(expected-wrong)>1e-5
    assert sensitive, 'odd-order test must detect accidental complex conjugation'


def test_static_exclusion_weights_frequency_and_determinism():
    a = coupled([np.eye(3)*1e200,np.eye(3)],frequencies=(0.,1.))
    result = calculate(a,weights=(0.,1.))
    assert result.pairs[0].coefficient(6,1,1,0) == pytest.approx(6.,rel=2e-15)
    assert result.cp_weights == [0.,1.]
    detached_weights = result.cp_weights; detached_weights[1] = 999.
    assert result.cp_weights == [0.,1.]
    assert result.pairs[0].coverage[0].table_complete
    assert result.pairs[0].coverage[0].unrestricted_complete
    assert result.pairs[0].coverage[1].missing_table_rank_quadruples
    values = [(x.order,x.t,x.u,x.J,x.value) for x in result.pairs[0].coefficients]
    assert values == [(x.order,x.t,x.u,x.J,x.value) for x in calculate(a,weights=(0.,1.)).pairs[0].coefficients]
    for w in ((1.,1.),(0.,0.),(0.,-1.),(0.,float('inf')),(0.,float('nan')),(1.,)):
        with pytest.raises((ValueError,RuntimeError)): calculate(a,weights=w)
    b = coupled([np.eye(3),np.eye(3)],frequencies=(0.,np.nextafter(1.,2.)))
    with pytest.raises((ValueError,RuntimeError),match='frequency'): calculate(a,b,weights=(0.,1.))
    with pytest.raises((ValueError,RuntimeError),match='quadrature'):
        calculate(coupled([np.eye(3)],frequencies=(0.,)),weights=(0.,))


def test_upstream_undefined_rank_pairs_are_structurally_absent():
    # casimir.f90 read_cg and recouple both execute "if (j1+j2>6) cycle", so
    # realcg_3_4/4_3/4_4 are never read and alpha_c is never initialized there.
    assert [t for t in DEFINED_PAIRS if c.isapol_realcg_defined(*t)] == DEFINED_PAIRS
    assert not any(c.isapol_realcg_defined(*t) for t in UNDEFINED_PAIRS)
    assert len(DEFINED_PAIRS) == 13 and len(UNDEFINED_PAIRS) == 3
    for l,p in UNDEFINED_PAIRS:
        with pytest.raises((ValueError,RuntimeError),match='undefined upstream'):
            c.isapol_realcg_terms(l,p)
    for l in (0,5):
        assert not c.isapol_realcg_defined(l,1) and not c.isapol_realcg_defined(1,l)
        with pytest.raises((ValueError,RuntimeError),match='undefined upstream'):
            c.isapol_realcg_terms(l,1)
    # A declared rank-4-only site is accepted, builds no block at all, and is
    # reported as missing coverage; it is never summed as zero-valued data.
    rec = coupled([np.eye(9)],ranks=(4,))
    assert [(b.la,b.lap) for b in rec.sites[0]] == []
    assert rec.value(0,0,4,4,1) == 0
    with pytest.raises((ValueError,RuntimeError),match='invalid tensor index'):
        rec.value(0,0,5,1,1)
    cov = calculate(rec).pairs[0].coverage
    assert all(not x.table_complete and not x.included_rank_quadruples for x in cov)
    assert all(x.value == 0. for x in calculate(rec).pairs[0].coefficients)
    # Mixed declaration keeps every defined ordered pair and drops only (4,4).
    mixed = coupled([np.eye(3+9)],ranks=(1,4))
    assert [(b.la,b.lap) for b in mixed.sites[0]] == [(1,1),(1,4),(4,1)]
    cov = calculate(mixed).pairs[0].coverage
    missing = {tuple(q) for x in cov for q in x.missing_table_rank_quadruples}
    included = {tuple(q) for x in cov for q in x.included_rank_quadruples}
    assert any(q[:2]==(4,4) or q[2:]==(4,4) for q in missing)
    assert not any(q[:2]==(4,4) or q[2:]==(4,4) for q in included)
    assert (1,4,4,1) in included


def test_rank_four_overflow_and_budgets_rejected():
    with pytest.raises((ValueError,RuntimeError),match='overflow'):
        calculate(coupled([np.eye(3)*1e200]))
    with pytest.raises((ValueError,RuntimeError),match='overflow'):
        coupled([np.eye(3)*1.7e308])
    sites=[]
    for k in range(65):
        s=local([np.eye(3)],label=str(k)).sites[0]; sites.append(s)
    many=c.IsaRecoupledModel(c.IsaAnisotropicModel([1.],sites,'supplied_local_response','pair budget'))
    with pytest.raises((ValueError,RuntimeError),match='budget'): calculate(many)
    full=[]
    for k in range(20): full.append(local([np.eye(15)],ranks=(1,2,3),label=str(k)).sites[0])
    many=c.IsaRecoupledModel(c.IsaAnisotropicModel([1.],full,'supplied_local_response','record budget'))
    with pytest.raises((ValueError,RuntimeError),match='budget'): calculate(many)


def test_work_budget_checked_before_pair_evaluation():
    # 106203 active term-component operations per rank123 pair/node; 943 nodes
    # exceed 100M while the local and returned-record budgets remain small.
    rec = coupled([np.eye(15)]*943, ranks=(1,2,3), frequencies=list(range(1,944)))
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        calculate(rec,weights=[1.]*943)


def test_local_validation_is_not_silently_repaired():
    A=np.eye(3); A[0,1]=1e-15
    with pytest.raises((ValueError,RuntimeError),match='symmetric'): coupled([A])
    with pytest.raises((ValueError,RuntimeError)): coupled([np.eye(3)],frame=np.zeros((3,3)))
    with pytest.raises((ValueError,RuntimeError)): coupled([np.full((3,3),np.nan)])
    with pytest.raises((ValueError,RuntimeError)): coupled([np.eye(3)],frequencies=(-1.,))


def check_archive_field(actual, token):
    # The writer omits trailing below-threshold fields. Absence is not permission
    # for an extra visible coefficient on an otherwise correctly present row.
    if token is None:
        assert abs(actual) <= 1e-6
        return 'absent'
    expected = float(token)
    if expected == 0.:
        assert abs(actual) <= 1e-6
        return 'placeholder'
    assert actual == pytest.approx(expected, rel=1e-6, abs=0.)
    return 'nonzero'


def test_archive_comparison_detects_mutated_absent_field():
    assert check_archive_field(0., None) == 'absent'
    assert check_archive_field(1e-6, None) == 'absent'
    with pytest.raises(AssertionError):
        check_archive_field(2e-6, None)


def test_literal_h2o_isagrid_l3_archived_write_precision():
    manifest=json.loads((ROOT/'manifest.txt').read_text())
    blob=(ROOT/'literal_numeric.json.gz').read_bytes()
    assert len(blob)==manifest['compressed_bytes'] < 150000
    assert hashlib.sha256(blob).hexdigest()==manifest['fixture_sha256']=='750187e3ef331de120bc551d27ddda90c63e3e292535cdce9cd74d586b2a79bf'
    with gzip.open(ROOT/'literal_numeric.json.gz','rb') as handle:
        raw=handle.read(1000001)
    assert len(raw)==manifest['uncompressed_bytes'] < 1000000
    fixture=json.loads(raw)
    assert fixture['track']=='work/H2O-isagrid'
    assert fixture['source_sha256']==manifest['source_sha256']=={
        'H2O_ref_wt4_L3_casimir.data':'1d230aebab7a59c809028e28f98e2ecff47b806d771a125f1d223144adc2ea8b',
        'H2O_ref_wt4_L3_C12.pot':'04feceb378fc2e6865224c45e7cba00bdcef294246a8f2289e5c2d792c408393'}
    grid=c.CasimirGrid(10,.5)
    freq=[grid.omega(k) for k in range(1,11)]
    weights=[grid.cp_weight(k) for k in range(1,11)]
    assert -freq[0]**2==pytest.approx(-4.3686833258996777e-5,rel=3e-14)
    assert -freq[-1]**2==pytest.approx(-1430.6369983255513,rel=3e-14)
    models={}
    for site in fixture['sites']:
        A=np.zeros((10,15,15))
        for t,u,start,end,tokens in site['entries']:
            assert 1<=start<=end<=531 and len(tokens)==10
            A[:,t-2,u-2]=A[:,u-2,t-2]=list(map(float,tokens))
        models.setdefault(site['type'],coupled(A,ranks=(1,2,3),frequencies=freq,label=site['label'],
            provenance='work/H2O-isagrid exact L3 deck SHA256 '+fixture['source_sha256']['H2O_ref_wt4_L3_casimir.data']))
    row_count=nonzero=placeholders=high=0
    for archived in fixture['pairs']:
        a,b=archived['types']
        result=calculate(models[a],models[b],weights=weights)
        rows={}
        for x in result.pairs[0].coefficients:
            rows.setdefault((x.t,x.u,x.J),{})[x.order]=x.value
        visible={k for k,v in rows.items() if any(abs(x)>1e-6 for x in v.values())}
        expected={(t,u,J) for t,u,J,line,fields in archived['rows']}
        assert {k for k in visible if k[2]<=8}==expected  # exact rowset, 0 extras
        high+=sum(k[2]>8 for k in visible)
        for t,u,J,line,fields in archived['rows']:
            assert line>0 and len(fields)==7
            row_count+=1
            for n,token in enumerate(fields,6):
                actual=rows.get((t,u,J),{}).get(n,0.)
                kind = check_archive_field(actual, token)
                placeholders += kind == 'placeholder'
                nonzero += kind == 'nonzero'
        if a==b=='O':
            assert result.pairs[0].coefficient(6,1,1,0)==pytest.approx(26.48177,rel=1e-6)
    assert (row_count,nonzero,placeholders,high)==(6285,10457,30791,411)


def check_alpha_token(actual, token, imaginary):
    # casimir writes recoupled components with g14.6: six significant figures and
    # a trailing "i" for a pure imaginary value.  The comparison floor is the
    # write precision itself, never a loosened scientific tolerance.
    expected = float(token)
    assert abs(actual.real if imaginary else actual.imag) < 1e-12
    got = actual.imag if imaginary else actual.real
    if expected == 0.:
        assert abs(got) <= 1e-6
        return 'placeholder'
    assert abs(got-expected) <= .5*10**(math.floor(math.log10(abs(expected)))-5)
    return 'nonzero'


def test_rank_four_declaration_extends_shipped_table_coverage():
    """Included/missing shipped-table quadruples, by construction not tolerance."""
    def counts(ranks):
        rec = coupled([np.eye(sum(2*l+1 for l in ranks))], ranks=ranks)
        cov = calculate(rec).pairs[0].coverage
        return ({x.order:len(x.included_rank_quadruples) for x in cov},
                {x.order:len(x.missing_table_rank_quadruples) for x in cov})
    low, _ = counts((1,2,3))
    high, missing = counts((1,2,3,4))
    assert [low[n] for n in range(6,13)] == [1,4,10,16,19,16,10]
    assert [high[n] for n in range(6,13)] == [1,4,10,20,31,36,34]
    assert [missing[n] for n in range(6,13)] == [0,0,0,0,0,4,10]
    # Every remaining gap needs an ordered pair casimir never initializes.
    rec = coupled([np.eye(24)], ranks=(1,2,3,4))
    for x in calculate(rec).pairs[0].coverage:
        for q in x.missing_table_rank_quadruples:
            assert not (c.isapol_realcg_defined(q[0],q[1]) and c.isapol_realcg_defined(q[2],q[3]))
        assert x.table_complete == (x.order <= 10)


def test_alpha_comparator_detects_write_precision_mutation():
    assert check_alpha_token(complex(1.03877), '1.03877', False) == 'nonzero'
    assert check_alpha_token(complex(0.,-0.238944), '-0.238944', True) == 'nonzero'
    assert check_alpha_token(complex(1.0387748), '1.03877', False) == 'nonzero'
    assert check_alpha_token(0j, '0.00000', True) == 'placeholder'
    for bad in (complex(1.03878), complex(1.03876), complex(-1.03877)):
        with pytest.raises(AssertionError):
            check_alpha_token(bad, '1.03877', False)
    with pytest.raises(AssertionError):        # real leakage into an imaginary block
        check_alpha_token(complex(1e-11,-0.238944), '-0.238944', True)


def test_rank_four_casimir_write_precision_recoupling_and_dispersion():
    """Rank 4 taken from the upstream code: the 13 ordered pairs it defines.

    Reference values are literal casimir print tokens for a seeded synthetic
    mixed-rank 1..4 deck, not values recomputed by the fixture generator.
    """
    manifest = json.loads((RANK4/'manifest.txt').read_text())
    blob = (RANK4/'literal_numeric.json.gz').read_bytes()
    assert len(blob) == manifest['compressed_bytes'] < 250000
    assert hashlib.sha256(blob).hexdigest() == manifest['fixture_sha256'] == \
        'f5243b7c6e9228c57dc69a93b4b50b0f1d54003f59fb6a9b883ac16545108cb3'
    with gzip.open(RANK4/'literal_numeric.json.gz','rb') as handle:
        raw = handle.read(1000001)
    assert len(raw) == manifest['uncompressed_bytes'] < 1000000
    fixture = json.loads(raw)
    assert fixture['source_sha256'] == manifest['source_sha256'] == {
        'rank4_casimir.data':'3b31b33826f8e30e4ad8fa39d16e99310fc38aaabfc1bb8fcc426d770d3b75d2',
        'rank4_casimir.out':'49997c167a4ee904e36756726d23458972ab7defc4515df775aa6de2a389d681'}
    assert [tuple(x) for x in fixture['ordered_pairs']] == DEFINED_PAIRS
    assert fixture['ranks'] == [1,2,3,4] and fixture['frequencies'] == {
        'kind':'CasimirGrid','n':10,'omega0':0.5}

    grid = c.CasimirGrid(10,.5)
    freq = [grid.omega(k) for k in range(1,11)]
    weights = [grid.cp_weight(k) for k in range(1,11)]
    A = np.zeros((10,24,24))
    for t,u,line,tokens in fixture['deck_entries']:
        assert 1 <= line <= 400 and len(tokens) == 10 and 2 <= t <= u <= 25
        A[:,t-2,u-2] = A[:,u-2,t-2] = list(map(float,tokens))
    assert len(fixture['deck_entries']) == manifest['counts']['deck_entries'] == 300
    rec = coupled(A, ranks=(1,2,3,4), frequencies=freq, label='S1',
                  provenance='rank4 casimir deck SHA256 '+fixture['source_sha256']['rank4_casimir.data'])
    assert [(b.la,b.lap) for b in rec.sites[0]] == DEFINED_PAIRS

    checked = declared_zero = written_zero = 0
    for la,lap,label,t,line,tokens in fixture['alpha']:
        assert (la,lap) in DEFINED_PAIRS and line > 0
        first, last = (la-lap)**2+1, (la+lap+1)**2
        assert first <= t <= last
        for f in range(10):
            value = rec.value(0,f,la,lap,t)
            if tokens is None:                     # printed "<label>(<la><lap>) all zero"
                assert abs(value) <= 1e-6
                declared_zero += 1
                continue
            if check_alpha_token(value, *tokens[f]) == 'placeholder':
                written_zero += 1
            else:
                checked += 1
    assert (checked, written_zero, declared_zero) == (3473, 117, 100)
    assert len(fixture['alpha']) == manifest['counts']['alpha_components'] == 369

    result = calculate(rec, weights=weights)
    rows = {}
    for x in result.pairs[0].coefficients:
        rows.setdefault((x.t,x.u,x.J),{})[x.order] = x.value
    nonzero = placeholders = absent = 0
    for t,u,J,tlabel,ulabel,line,fields in fixture['cn_rows']:
        assert line > 0 and 0 <= J <= 8 and 1 <= len(fields) <= 7
        for n,token in enumerate(fields,6):
            kind = check_archive_field(rows.get((t,u,J),{}).get(n,0.), token)
            placeholders += kind == 'placeholder'
            nonzero += kind == 'nonzero'
        for n in range(6+len(fields),13):          # omitted trailing fields
            assert check_archive_field(rows.get((t,u,J),{}).get(n,0.), None) == 'absent'
            absent += 1
    visible = {k for k,v in rows.items() if any(abs(x)>1e-6 for x in v.values())}
    archived = {(t,u,J) for t,u,J,_,_,_,_ in fixture['cn_rows']}
    assert {k for k in visible if k[2] <= 8} == archived   # exact rowset, 0 extras
    # casimir's writer loops J=0..8 only, so the J9/10 rows are counted here and
    # certified elsewhere by the independent factorial oracle, not by this fixture.
    assert sum(k[2] > 8 for k in visible) == 735
    assert (len(fixture['cn_rows']), nonzero, placeholders, absent) == (10029,15877,50155,4171)
    assert len(fixture['cn_rows']) == manifest['counts']['cn_rows']
