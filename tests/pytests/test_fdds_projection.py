"""Small independent projection references for future exact optimizations.

The cubic reference tensor is deliberately restricted to this tiny fixture.
No production tensor materialization or projection algorithm changes.
"""
import numpy as np
import pytest
import psi4

from test_cuest_fdds import fdds_factory

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("count", [0, 1, 2, 4])
@pytest.mark.parametrize("threads", [1, 2])
def test_projection_independent_reference(fdds_factory, count, threads):
    obj = fdds_factory()
    primary, auxiliary, _ = fdds_factory.inputs
    nbf, naux = primary.nbf(), auxiliary.nbf()
    assert naux <= 20, "Never build a production cubic reference tensor"
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(primary)
    eri = np.asarray(mints.ao_eri(auxiliary, zero, primary, primary)).reshape(naux, nbf, nbf)
    overlap = np.asarray(mints.ao_3coverlap(auxiliary, auxiliary, auxiliary)).reshape(naux, naux, naux)
    inverse = np.asarray(obj.metric_inv())
    rng = np.random.default_rng(391)
    # Include zero and signed nonsymmetric matrices: projection must perform
    # the full AO contraction rather than assume input symmetry.
    densities = [np.zeros((nbf, nbf))]
    densities += [rng.normal(size=(nbf, nbf)) for _ in range(max(0, count - 1))]
    densities = densities[:count]
    inputs = [psi4.core.Matrix.from_array(d) for d in densities]
    before = [np.array(d, copy=True) for d in inputs]
    psi4.set_num_threads(threads)
    try:
        for _ in range(2):
            actual = obj.project_densities(inputs)
            assert len(actual) == count
            for d, result in zip(densities, actual):
                aux_density = np.einsum("Rpq,pq->R", eri, d)
                coefficients = inverse @ aux_density
                expected = 2. * np.einsum("PQR,R->PQ", overlap, coefficients)
                np.testing.assert_allclose(np.asarray(result), expected, rtol=2.e-11, atol=2.e-11)
                np.testing.assert_array_equal(np.asarray(result), np.asarray(result).T)
        for d, original in zip(inputs, before):
            np.testing.assert_array_equal(np.asarray(d), original)
    finally:
        psi4.set_num_threads(1)
