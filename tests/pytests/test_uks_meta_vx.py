import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

_CH3 = """
0 2
C   0.000  0.000  0.000
H   1.080  0.000  0.000
H  -0.540  0.935  0.000
H  -0.540 -0.935  0.100
symmetry c1
"""


@pytest.mark.parametrize("func", ["pbe", "tpss", "m06-2x"])
def test_uks_vx_matches_finite_difference(func):
    """UV::compute_Vx against a central difference of compute_V, meta-GGA included."""
    psi4.geometry(_CH3)
    psi4.set_options({"basis": "cc-pvdz", "reference": "uks", "scf_type": "df", "d_convergence": 1e-8})
    _, wfn = psi4.energy(func, return_wfn=True)
    V = wfn.V_potential()
    nbf = wfn.nso()
    Da, Db = wfn.Da().np.copy(), wfn.Db().np.copy()
    rng = np.random.default_rng(7)

    def pert(C, n):
        M = C[:, :n] @ rng.standard_normal((n, nbf - n)) @ C[:, n:].T
        return 0.5 * (M + M.T)

    Dka, Dkb = pert(wfn.Ca().np, wfn.nalpha()), pert(wfn.Cb().np, wfn.nbeta())
    mat = psi4.core.Matrix.from_array

    def V_of(da, db):
        V.set_D([mat(da), mat(db)])
        ret = [psi4.core.Matrix(nbf, nbf), psi4.core.Matrix(nbf, nbf)]
        V.compute_V(ret)
        return [r.np.copy() for r in ret]

    h = 1e-4
    plus, minus = V_of(Da + h * Dka, Db + h * Dkb), V_of(Da - h * Dka, Db - h * Dkb)
    V.set_D([mat(Da), mat(Db)])
    ret = [psi4.core.Matrix(nbf, nbf), psi4.core.Matrix(nbf, nbf)]
    V.compute_Vx([mat(Dka), mat(Dkb)], ret)
    for vx, p, m in zip(ret, plus, minus):
        fd = (p - m) / (2 * h)
        assert np.abs(vx.np - fd).max() / np.abs(fd).max() < 1e-5

