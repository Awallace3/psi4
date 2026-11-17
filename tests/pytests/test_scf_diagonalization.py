import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("degenerate", [False, True])
def test_dsyevd_small_degenerate_and_triangle(degenerate):
    """Do not compare individual eigenvectors inside a degenerate eigenspace."""
    dimensions = psi4.core.Dimension([0, 1, 2, 7, 33])
    a = psi4.core.Matrix("A", dimensions, dimensions)
    rng = np.random.default_rng(894)
    expected = []
    for h, n in enumerate(dimensions.to_tuple()):
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        values = np.ones(n) if degenerate else np.arange(n, dtype=float)
        symmetric = (q * values) @ q.T
        # diagonalize() reads the row-major upper triangle, not the lower.
        a.nph[h][:] = np.triu(symmetric) + np.tril(np.full((n, n), 12345.0), -1)
        expected.append(symmetric)
    original = a.clone()
    c = psi4.core.Matrix("C", dimensions, dimensions)
    e = psi4.core.Vector("e", dimensions)
    a.diagonalize_dsyevd(c, e)
    for h, n in enumerate(dimensions.to_tuple()):
        if not n:
            continue
        assert np.array_equal(a.nph[h], original.nph[h])
        np.testing.assert_allclose(c.nph[h].T @ c.nph[h], np.eye(n), atol=2e-13)
        np.testing.assert_allclose(expected[h] @ c.nph[h], c.nph[h] * e.nph[h], atol=2e-12)


def test_rohf_dsyevd_level_shift():
    """Both ROHF diagonalization call sites reuse correctly sized workspaces."""
    psi4.geometry("1 2\nF 0 0 0\nH 0 0 0.92\n")
    psi4.set_options({
        "reference": "rohf", "basis": "cc-pvdz", "scf_type": "pk",
        "e_convergence": 1e-10, "d_convergence": 1e-10,
    })
    exact, ref = psi4.energy("hf", return_wfn=True)
    densities = [ref.Da().to_array(), ref.Db().to_array()]
    psi4.core.clean()
    psi4.set_options({"level_shift": 0.5, "level_shift_cutoff": 1e-2})
    energy, shifted = psi4.energy("hf", return_wfn=True)
    np.testing.assert_allclose(energy, exact, atol=1e-9, rtol=0)
    for expected, matrix in zip(densities, [shifted.Da(), shifted.Db()]):
        for a, b in zip(expected, matrix.to_array()):
            np.testing.assert_allclose(a, b, atol=2e-8, rtol=0)


def test_restricted_workspace_falls_back_to_dsyev(monkeypatch):
    psi4.geometry("O 0 0 0\nH 0 0 1\nH 1 0 0\nsymmetry c1")
    psi4.set_options({"basis": "cc-pvdz", "scf_type": "mem_df"})
    reserve = psi4.core.HF.reserve_diagonalization_workspace
    charges = []

    def select_workspace(wfn, spare):
        charge = reserve(wfn, spare if not charges else 0)
        charges.append(charge)
        return charge

    monkeypatch.setattr(psi4.core.HF, "reserve_diagonalization_workspace", select_workspace)
    first, ref = psi4.energy("hf", return_wfn=True)
    second, low = psi4.energy("hf", return_wfn=True)
    assert charges[0] > 0 and charges[1:] == [0]
    assert abs(first - second) < 1e-10
    np.testing.assert_allclose(ref.Da().np, low.Da().np, atol=1e-9)
