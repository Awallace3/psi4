"""Numerical J9/C11 and J10/C12 tests, independent of both shipped tables.

See psi4/src/psi4/libisapol/HIGH_J_VALIDATION.md for the derivation and limits.
No home-tree or fixture reads; production calls occur only as actual-under-test.
"""
import math

import numpy as np
import pytest
import psi4

from isapol_factorial_oracle import cg
from isapol_high_j_oracle import high_j_coefficients, rank_quadruples, stretched_factor


FREQUENCIES = (0., .6, 1.9)
WEIGHTS = (0., .17, .43)


def spd_inputs(ranks, seed):
    """Nonidentical frequency-dependent SPD tensors, with mixed-rank blocks.

    Construct symmetric inputs, never repair supplied tensors. A fixed seed is
    only an input recipe: expected coefficients are calculated at test runtime.
    """
    d = sum(2*l+1 for l in ranks)
    rng = np.random.default_rng(seed)
    result = []
    for f in range(3):
        X = rng.normal(size=(d, d))/math.sqrt(d)
        A = X@X.T + (1.+f/4)*np.eye(d)
        np.testing.assert_array_equal(A, A.T)
        assert np.linalg.eigvalsh(A)[0] > 0
        result.append(A)
    return np.array(result)


def actual_high_j(A, ranks_a, B, ranks_b):
    c = psi4.core

    def model(matrices, ranks, label):
        site = c.IsaAnisotropicSite()
        site.label, site.ranks = label, list(ranks)
        site.origin, site.frame = [0., 0., 0.], np.eye(3)
        site.responses = [c.Matrix.from_array(x) for x in matrices]
        local = c.IsaAnisotropicModel(list(FREQUENCIES), [site],
            'supplied_local_response', 'independent stretched-J analytic test input')
        return c.IsaRecoupledModel(local)

    result = c.isa_recoupled_dispersion(model(A, ranks_a, 'A'), model(B, ranks_b, 'B'),
                                      list(WEIGHTS), 12)
    records = [x for x in result.pairs[0].coefficients if x.J >= 9]
    actual = {(x.order, x.t, x.u, x.J): x.value for x in records}
    assert len(actual) == len(records), 'duplicate high-J rows'
    return actual


def compare(actual, expected):
    assert actual.keys() == expected.keys(), 'missing or extra high-J rows'
    keys = sorted(expected)
    # Existing analytic recoupling tolerance, not the much looser archive gate.
    np.testing.assert_allclose([actual[k] for k in keys], [expected[k] for k in keys],
                               rtol=3e-13, atol=3e-12)


def test_all_high_j_blocks_spd_phase_scale_and_mutation_sensitivity():
    ranks = (1, 2, 3)
    A, B = spd_inputs(ranks, 90211), spd_inputs(ranks, 100212)
    expected = high_j_coefficients(A, ranks, B, ranks, WEIGHTS)
    actual = actual_high_j(A, ranks, B, ranks)
    compare(actual, expected)
    assert len(expected) == 735
    assert [len(rank_quadruples(ranks, ranks, J)) for J in (9, 10)] == [16, 10]
    blocks = {(n, math.isqrt(t-1), math.isqrt(u-1), J) for n, t, u, J in expected}
    assert blocks == {(11, 3, 6, 9), (11, 4, 5, 9), (11, 5, 4, 9), (11, 6, 3, 9),
                      (12, 4, 6, 10), (12, 5, 5, 10), (12, 6, 4, 10)}
    for n, L, H, J in sorted(blocks):
        keys = [k for k in expected if k[0] == n and math.isqrt(k[1]-1) == L
                and math.isqrt(k[2]-1) == H and k[3] == J]
        values = np.array([expected[k] for k in keys])
        assert np.count_nonzero(np.abs(values) > 1e-8) == (2*L+1)*(2*H+1)
        assert values.min() < -1e-5 and values.max() > 1e-5
        # Mutate the actual-under-test, one whole block at a time. Every block
        # must catch erased, sign-reversed, and misnormalized high-J output.
        for factor in (0., -1., 1.01):
            mutant = actual.copy()
            for k in keys:
                mutant[k] *= factor
            with pytest.raises(AssertionError):
                compare(mutant, expected)
    missing = actual.copy()
    del missing[next(iter(missing))]
    with pytest.raises(AssertionError):
        compare(missing, expected)
    # Separate scale/weight check with a fresh production evaluation.
    compare(actual_high_j(2*A, ranks, 3*B, ranks), {k: 6*v for k, v in expected.items()})


# Unordered rank pairs isolate every reciprocal equivalence class of the 26
# ordered quadruples. Minimal declared subsets exclude competing sum-4 pairs:
# (1,3)/(3,1) versus (2,2). All subsets use compressed offsets.
ISOLATED_PAIRS = [
    ((1, 2), (3, 3)), ((1, 3), (2, 3)), ((2, 2), (2, 3)),
    ((2, 3), (1, 3)), ((2, 3), (2, 2)), ((3, 3), (1, 2)),
    ((1, 3), (3, 3)), ((2, 2), (3, 3)), ((2, 3), (2, 3)),
    ((3, 3), (1, 3)), ((3, 3), (2, 2)),
]


def test_isolation_cases_cover_all_high_j_reciprocal_classes():
    assert len(ISOLATED_PAIRS) == len(set(ISOLATED_PAIRS)) == 11
    covered = {a+b for pair_a, pair_b in ISOLATED_PAIRS
               for a in {pair_a, pair_a[::-1]}
               for b in {pair_b, pair_b[::-1]}}
    required = {q for J in (9, 10)
                for q in rank_quadruples((1, 2, 3), (1, 2, 3), J)}
    assert covered == required
    assert len(covered) == 26


@pytest.mark.parametrize('pair_a,pair_b', ISOLATED_PAIRS)
def test_isolate_high_j_rank_quadruples_compressed(pair_a, pair_b):
    ranks_a, ranks_b = tuple(sorted(set(pair_a))), tuple(sorted(set(pair_b)))
    A, B = spd_inputs(ranks_a, 719), spd_inputs(ranks_b, 823)
    expected = high_j_coefficients(A, ranks_a, B, ranks_b, WEIGHTS)
    compare(actual_high_j(A, ranks_a, B, ranks_b), expected)
    L, H = sum(pair_a), sum(pair_b)
    J = L+H
    target = {k: v for k, v in expected.items()
              if k[0] == J+2 and math.isqrt(k[1]-1) == L and math.isqrt(k[2]-1) == H}
    assert len(target) == (2*L+1)*(2*H+1)
    assert min(target.values()) < -1e-5 and max(target.values()) > 1e-5
    contributing = {q for q in rank_quadruples(ranks_a, ranks_b, J)
                    if sum(q[:2]) == L and sum(q[2:]) == H}
    assert contributing == {a+b for a in {pair_a, pair_a[::-1]}
                            for b in {pair_b, pair_b[::-1]}}


def test_stretched_factor_legendre_projection_and_electrostatic_phase():
    # An independent polynomial-product check of the two zero-m CG factors.
    # Degree <=20: 11-point Gauss--Legendre integrates the projection exactly
    # in exact arithmetic. It is not a fit to any production coefficient.
    x, w = np.polynomial.legendre.leggauss(11)

    def P(l):
        return np.polynomial.legendre.legval(x, [0.]*l+[1.])

    for J in (9, 10):
        for l, p, k, q in rank_quadruples((1, 2, 3), (1, 2, 3), J):
            K, Q, L, H = l+k, p+q, l+p, k+q
            projections = []
            for a, b in ((K, Q), (L, H)):
                projection = (2*J+1)/2*np.dot(w, P(a)*P(b)*P(J))
                assert projection == pytest.approx(cg(a, 0, b, 0, J, 0)**2,
                                                    rel=3e-13, abs=3e-15)
                projections.append(projection)
            radial = math.sqrt(math.comb(2*K, 2*l)*math.comb(2*Q, 2*p))
            # i^(L-H-J) in Sbar, NOT an empirically selected odd-rank sign.
            angular_phase = (1j)**(L-H-J)
            assert angular_phase == (-1)**H
            from_projection = (-1)**H/angular_phase*radial*math.sqrt(math.prod(projections))
            assert stretched_factor(l, p, k, q) == pytest.approx(from_projection,
                                                               rel=3e-13, abs=3e-12)
