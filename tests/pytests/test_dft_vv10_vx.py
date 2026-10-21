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

_H2O = """
O   0.000  0.000  0.117
H   0.000  0.757 -0.467
H   0.000 -0.757 -0.467
symmetry c1
"""


def _pert(rng, C, n, nbf):
    M = C[:, :n] @ rng.standard_normal((n, nbf - n)) @ C[:, n:].T
    return 0.5 * (M + M.T)


def _rel_err(vx, fd):
    return np.abs(vx - fd).max() / np.abs(fd).max()


@pytest.mark.parametrize("func", ["wb97x-v", "wb97m-v"])
def test_rks_vv10_vx_matches_finite_difference(func):
    """RV::compute_Vx (1/2 dV/dl) with the VV10 kernel against a central difference of compute_V."""
    psi4.geometry(_H2O)
    psi4.set_options({"basis": "cc-pvdz", "scf_type": "df", "d_convergence": 1e-8})
    _, wfn = psi4.energy(func, return_wfn=True)
    V = wfn.V_potential()
    nbf = wfn.nso()
    Da = wfn.Da().np.copy()
    mat = psi4.core.Matrix.from_array
    rng = np.random.default_rng(11)
    Dk = _pert(rng, wfn.Ca().np, wfn.nalpha(), nbf)

    def V_of(d):
        V.set_D([mat(d)])
        ret = [psi4.core.Matrix(nbf, nbf)]
        V.compute_V(ret)
        return ret[0].np.copy()

    h = 1e-4
    fd = 0.5 * (V_of(Da + h * Dk) - V_of(Da - h * Dk)) / (2 * h)
    V.set_D([mat(Da)])
    ret = [psi4.core.Matrix(nbf, nbf)]
    V.compute_Vx([mat(Dk)], ret)
    assert _rel_err(ret[0].np, fd) < 1e-5

    # the VV10 kernel is not negligible: without it the same comparison fails
    V.set_vx_neglect_vv10(True)
    V.compute_Vx([mat(Dk)], ret)
    V.set_vx_neglect_vv10(False)
    assert _rel_err(ret[0].np, fd) > 1e-4


@pytest.mark.parametrize("func", ["wb97x-v", "wb97m-v"])
def test_uks_vv10_vx_matches_finite_difference(func):
    """UV::compute_Vx (dV/dl per spin) with the VV10 kernel against a central difference of compute_V."""
    psi4.geometry(_CH3)
    psi4.set_options({"basis": "cc-pvdz", "reference": "uks", "scf_type": "df", "d_convergence": 1e-8})
    _, wfn = psi4.energy(func, return_wfn=True)
    V = wfn.V_potential()
    nbf = wfn.nso()
    Da, Db = wfn.Da().np.copy(), wfn.Db().np.copy()
    mat = psi4.core.Matrix.from_array
    rng = np.random.default_rng(7)
    Dka, Dkb = _pert(rng, wfn.Ca().np, wfn.nalpha(), nbf), _pert(rng, wfn.Cb().np, wfn.nbeta(), nbf)

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
        assert _rel_err(vx.np, (p - m) / (2 * h)) < 1e-5


@pytest.mark.parametrize("reference,mol", [("rks", _H2O), ("uks", _CH3)], ids=["rks", "uks"])
def test_soscf_vv10_reaches_diis_energy(reference, mol):
    """SOSCF with the full (VV10-including) orbital Hessian reaches the DIIS energy."""
    psi4.geometry(mol)
    psi4.set_options({"basis": "cc-pvdz", "reference": reference, "scf_type": "df", "e_convergence": 1e-9,
                      "d_convergence": 1e-8})
    e_diis = psi4.energy("wb97m-v")
    psi4.set_options({"soscf": True})
    e_soscf = psi4.energy("wb97m-v")
    assert psi4.compare_values(e_diis, e_soscf, 7, f"wB97M-V {reference} SOSCF vs DIIS energy")
