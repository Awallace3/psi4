"""Independent maximal-J dispersion oracle; see HIGH_J_VALIDATION.md.

Only factorial angular algebra and supplied arrays enter expected values.
No Psi4 import, table access, numerical reference files, or audit imports.
CamCASP conventions: Misquitta / Stone, MIT notice in
psi4/src/psi4/libisapol/RECOUPLED_CAMCASP_LICENSE (not a table transcode).
Psi4 additions: Copyright (c) 2026 The Psi4 Developers. LGPL-3.0-only.
"""
from functools import lru_cache
from itertools import product
import math

import numpy as np

from isapol_factorial_oracle import cg, independent_g


def rank_quadruples(ranks_a, ranks_b, J):
    """Ordered positive-rank quadruples, obtained without production metadata."""
    return tuple(q for q in product(ranks_a, ranks_a, ranks_b, ranks_b) if sum(q) == J)


def stretched_factor(l, p, k, q):
    """Coefficient multiplying alpha_A^(l+p) alpha_B^(k+q), J=l+p+k+q.

    The four-spin stretched recoupling overlap is +1. The electrostatic
    (-1)^(k+q) cancels i^(L-H-J)=(-1)^H in the normalized S function.
    No conjugation or extra CP prefactor is introduced.
    """
    K, P, L, H = l+k, p+q, l+p, k+q
    J = L+H
    return (math.sqrt(math.comb(2*K, 2*l)*math.comb(2*P, 2*p))
            * cg(K, 0, P, 0, J, 0) * cg(L, 0, H, 0, J, 0))


@lru_cache(maxsize=9)
def _stretched_g(l, p):
    g = independent_g(l, p)[(l+p)**2:]
    g.setflags(write=False)
    return g


def _couple(matrices, ranks):
    assert tuple(sorted(set(ranks))) == tuple(ranks)
    assert ranks and set(ranks) <= {1, 2, 3}
    d = sum(2*l+1 for l in ranks)
    matrices = np.asarray(matrices)
    assert matrices.ndim == 3 and matrices.shape[1:] == (d, d)
    offsets = {}
    offset = 0
    for l in ranks:
        offsets[l] = slice(offset, offset+2*l+1)
        offset += 2*l+1
    return {(l, p): np.einsum('tab,fab->ft', _stretched_g(l, p),
                              matrices[:, offsets[l], offsets[p]])
            for l, p in product(ranks, repeat=2)}


def high_j_coefficients(matrices_a, ranks_a, matrices_b, ranks_b, weights):
    """Return all structurally present (n,t,u,J) for J9/C11 and J10/C12.

    Every component is retained, including zeros. Storage offsets follow the
    supplied compressed ranks. Zero-weight nodes are excluded before CP multiplication.
    Intended only for bounded test inputs, not a production implementation.
    """
    aa, bb = _couple(matrices_a, ranks_a), _couple(matrices_b, ranks_b)
    weights = np.asarray(weights)
    assert len(weights) == len(matrices_a) == len(matrices_b)
    active = weights != 0
    result = {}
    for J in (9, 10):
        blocks = {}
        for l, p, k, q in rank_quadruples(ranks_a, ranks_b, J):
            L, H = l+p, k+q
            integral = np.einsum('f,ft,fu->tu', weights[active],
                                 aa[l, p][active], bb[k, q][active])
            assert np.max(np.abs(integral.imag)) < 1e-12
            term = stretched_factor(l, p, k, q)*integral.real
            if (L, H) not in blocks:
                blocks[L, H] = np.zeros_like(term)
            blocks[L, H] += term
        for (L, H), block in blocks.items():
            for t, u in np.ndindex(block.shape):
                result[J+2, L*L+t+1, H*H+u+1, J] = block[t, u]
    return result
