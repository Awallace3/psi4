"""Default, full-precision SCF speedups: no experimental options required."""
import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def _bases():
    mol = psi4.geometry("""
        O  0.000  0.000  0.117
        H  0.000  0.757 -0.467
        H  0.000 -0.757 -0.467
        symmetry c1
        no_reorient
        no_com
    """)
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")
    return primary, aux


def _jk(primary, aux, *, algorithm="mem_df", package="libint2", omega=None, threads=1, subtype="INCORE"):
    psi4.set_options({
        "scf_type": algorithm, "scf_subtype": subtype, "integral_package": package,
    })
    jk = psi4.core.JK.build_JK(primary, aux)
    jk.set_omp_nthread(threads)
    jk.set_do_wK(omega is not None)
    jk.set_wcombine(False)
    if omega is not None:
        jk.set_omega(omega)
    jk.set_memory(10_000_000)
    return jk


def _compute(jk, left, right=None):
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(left.copy()))
    if right is not None:
        jk.C_right_add(psi4.core.Matrix.from_array(right.copy()))
    jk.compute()
    return [m.np.copy() for m in
            list(jk.J()) + list(jk.K()) + list(jk.wK())]


def test_memdf_scratch_release_and_reuse():
    primary, aux = _bases()
    jk = _jk(primary, aux)
    jk.initialize()
    left = np.random.default_rng(402).normal(size=(primary.nbf(), 5))
    expected = _compute(jk, left)
    size = jk.dfh().JK_scratch_memory()
    assert size > 0
    repeated = _compute(jk, left)
    assert jk.dfh().JK_scratch_memory() == size
    jk.release_scratch()
    assert jk.dfh().JK_scratch_memory() == 0
    rebuilt = _compute(jk, left)
    for a, b, c in zip(expected, repeated, rebuilt):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a, c, atol=1e-12, rtol=0)
    jk.finalize()
    assert jk.dfh().JK_scratch_memory() == 0


@pytest.mark.parametrize("subtype", ["INCORE", "OUT_OF_CORE"])
def test_memdf_exact_tiny_basis_memory_floor(subtype):
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    jk = _jk(basis, basis, subtype=subtype)
    jk.set_memory(jk.memory_estimate())
    jk.initialize()
    assert jk.dfh().get_AO_core() == (subtype == "INCORE")
    _compute(jk, np.ones((1, 1)))
    assert jk.K()[0].get(0, 0) > 0
    jk.finalize()


@pytest.mark.parametrize("omega", [None, 0.3])
def test_memdf_held_metric_powers_are_independent(omega):
    """Holding J must not power the already-inverted matrix for a later wK power."""
    primary, aux = _bases()
    coefficients = np.random.default_rng(404).normal(size=(primary.nbf(), 5)) / primary.nbf()
    results = []
    for hold in (False, True):
        jk = _jk(primary, aux, omega=omega)
        jk.dfh().hold_met(hold)
        jk.initialize()
        results.append(_compute(jk, coefficients))
        jk.finalize()
    for expected, actual in zip(*results):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)


def test_memdf_out_of_core_metric_budget():
    """Charge both the original metric and its independent inverse power."""
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    jk = _jk(basis, basis, subtype="OUT_OF_CORE")
    jk.dfh().hold_met(True)
    jk.set_memory(100)
    jk.initialize()
    jk.dfh().set_memory(6)
    _compute(jk, np.ones((1, 1)))
    assert jk.dfh().JK_scratch_memory() == 3
    assert jk.K()[0].get(0, 0) > 0
    jk.dfh().set_memory(5)
    with pytest.raises(RuntimeError, match="not enough memory for JK blocking"):
        jk.compute()
    jk.finalize()


def test_memdf_narrowest_panels_fit_the_advertised_minimum():
    """A panel-width heuristic must not double the true minimum panel allowance."""
    mol = psi4.geometry("""
        Ar 0 0 0
        Ar 0 0 5
        Ar 0 0 15
        Ar 0 0 25
        Ar 0 0 35
        symmetry c1
    """)
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")
    jk = _jk(primary, aux)
    minimum = jk.memory_estimate()
    jk.set_memory(minimum)
    jk.initialize()
    assert jk.dfh().get_AO_core()
    coefficients = np.random.default_rng(405).normal(size=(primary.nbf(), 5)) / primary.nbf()
    _compute(jk, coefficients)
    assert jk.dfh().get_AO_size() + jk.dfh().JK_scratch_memory() <= jk.dfh().get_memory()
    jk.finalize()
