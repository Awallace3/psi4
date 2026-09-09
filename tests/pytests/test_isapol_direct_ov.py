"""Direct-OV occupied-fast indexing oracle with multiple occupied and virtual MOs."""
import numpy as np
import psi4
from test_isapol_partitioned_response import basis, site, poly2


def test_multioccupied_multivirtual_occupied_fast_columns():
    centers = np.array([[0., 0., 0.], [.8, .1, -.3], [-.2, .7, .4],
                        [.3, -.9, .6], [-.5, -.2, -.8]])
    b = basis(tuple(map(tuple, centers)), psi4.core.IsaBasisRole.Orbital)
    s = site(neighbours=tuple(range(5)))
    coefficients = np.array([[.8, .2, .4, -.1, .5], [-.1, .7, -.2, .6, .3],
                             [.3, -.4, .9, .1, -.2], [.2, .5, -.6, .8, .4],
                             [-.7, .1, .3, -.5, .9]])
    out = psi4.core.IsaPartitionedMultipoles(
        b, [s], 'synthetic 2 occupied / 3 virtual', 1e-36,
        psi4.core.Matrix.from_array(coefficients), 2)
    points = np.array(s.samples.points)
    aos = 1.2*np.exp(-.7*np.sum((points[:, None, :] - centers)**2, axis=2))
    mos = aos @ coefficients
    # Explicit loops, not production flatten/reshape: column = i + nocc*a.
    expected = np.column_stack([poly2(points) @ (mos[:, i]*mos[:, 2+a])
                                for a in range(3) for i in range(2)])
    wrong_order = np.column_stack([poly2(points) @ (mos[:, i]*mos[:, 2+a])
                                   for i in range(2) for a in range(3)])
    assert not np.allclose(expected, wrong_order)
    np.testing.assert_allclose(out.values.np, expected, atol=2e-14, rtol=2e-14)
    assert out.representation == 'direct_ov'
    # Nonuniform response weights ensure a shared column permutation cannot hide.
    weights = np.arange(1., 7.)
    response = psi4.core.IsaDistributedResponse(
        out, [0.], [psi4.core.Matrix.from_array(-np.diag(weights))],
        'direct_ov', 'occupied-fast weighted response')
    np.testing.assert_allclose(response.at_index(0).np,
                               (expected*weights) @ expected.T, atol=2e-12, rtol=2e-14)
