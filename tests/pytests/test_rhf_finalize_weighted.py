"""RHF.finalize's signed, occupied-only Lagrangian through the compiled interface."""
import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def _wfn(symmetry, hydrogen=False):
    psi4.core.clean()
    psi4.core.clean_options()
    geometry = "H 0 0 -0.7\nH 0 0 0.7" if hydrogen else "O 0 0 0\nH 0 1.4 1\nH 0 -1.4 1"
    mol = psi4.geometry(geometry + f"\nunits bohr\nsymmetry {symmetry}")
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk", "reference": "rhf",
                      "use_cuest": False, "save_jk": True,
                      "e_convergence": 10, "d_convergence": 10})
    return psi4.energy("hf", molecule=mol, return_wfn=True)


def _direct(wfn):
    result = []
    for h, (c, eps) in enumerate(zip(wfn.Ca().nph, wfn.epsilon_a().nph)):
        # Deliberately retain the scalar reference, not the new weighted GEMM.
        x = np.zeros((c.shape[0], c.shape[0]))
        for m in range(c.shape[0]):
            for n in range(c.shape[0]):
                for i in range(wfn.nalphapi()[h]):
                    x[m, n] += eps[i] * c[m, i] * c[n, i]
        result.append(x)
    return result


@pytest.mark.parametrize("symmetry,hydrogen", [("c1", False), ("c2v", False), ("d2h", True)])
@pytest.mark.parametrize("synthetic", [False, True])
def test_rhf_finalize_signed_occupied_lagrangian(symmetry, hydrogen, synthetic):
    energy, wfn = _wfn(symmetry, hydrogen)
    if synthetic:
        rng = np.random.default_rng(7701)
        for c, eps in zip(wfn.Ca().nph, wfn.epsilon_a().nph):
            c[:] = rng.normal(size=c.shape)
            eps[:] = np.resize([-1.7, 0.0, 0.4], eps.size)
        # Nonzero sentinel: empty-occupied blocks must be overwritten, not skipped.
        wfn.lagrangian().set(19.0)
        wfn.finalize()
    expected = _direct(wfn)
    for ref, got in zip(expected, wfn.lagrangian().nph):
        np.testing.assert_allclose(got, ref, atol=2.e-13, rtol=2.e-13)
        np.testing.assert_allclose(got, got.T, atol=2.e-13, rtol=2.e-13)
    assert wfn.energy() == energy
    if symmetry == "c2v":
        assert 0 in list(wfn.nalphapi())
        assert len(set(wfn.nalphapi())) > 1
    if hydrogen:
        assert 0 in list(wfn.nsopi())


def test_rhf_refinalize_preserves_response_and_energy():
    energy, wfn = _wfn("c1")
    rhs = psi4.core.Matrix.from_array(
        np.random.default_rng(907).normal(size=(wfn.nalpha(), wfn.nmo() - wfn.nalpha())) * 0.01)
    before = [m.clone() for m in wfn.cphf_Hx([rhs])]
    solution = wfn.cphf_solve([rhs], 1.e-10, 100, 0)
    assert wfn.cphf_converged()
    lagrangian = wfn.lagrangian().clone()
    wfn.finalize()
    np.testing.assert_allclose(wfn.lagrangian().np, lagrangian.np, atol=2.e-13, rtol=0)
    assert wfn.energy() == energy
    for ref, got in zip(before, wfn.cphf_Hx([rhs])):
        np.testing.assert_allclose(got.np, ref.np, atol=1.e-12, rtol=0)
    for got in wfn.cphf_Hx(solution):
        np.testing.assert_allclose(got.np, rhs.np, atol=1.e-10, rtol=0)
