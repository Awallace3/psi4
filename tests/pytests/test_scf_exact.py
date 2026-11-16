"""Default, full-precision SCF speedups: no experimental options required."""
import numpy as np
import pytest

import psi4
from addons import uusing

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
    # Exercise the exact QI reference path on both tiers. Opt's AUTO can select
    # padded IQ on AMD; its different allocation floors have separate tests.
    if hasattr(psi4.core.DFHelper, "set_K_fp32"):
        psi4.set_options({"scf_df_k_layout": "QI"})
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


@pytest.mark.parametrize("omega", [None, 0.0, 0.3])
@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_memdf_exact_matrices_against_diskdf(omega, threads, symmetric):
    """Direct Libint/compact-index DF and wK agree with the independent backend."""
    primary, aux = _bases()
    rng = np.random.default_rng(401)
    left, right = rng.normal(size=(2, primary.nbf(), 16)) / primary.nbf()
    results = []
    for algorithm in ("disk_df", "mem_df"):
        jk = _jk(primary, aux, algorithm=algorithm, omega=omega, threads=threads)
        jk.initialize()
        results.append(_compute(jk, left, None if symmetric else right))
        jk.finalize()
    for expected, actual in zip(*results):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    if omega == 0.0:
        np.testing.assert_allclose(results[-1][-1], 0.0, atol=1e-14, rtol=0)


@pytest.mark.parametrize("threads", [1, 4])
def test_memdf_requested_vs_actual_team(threads):
    """Run also with OMP_THREAD_LIMIT=1 to exercise a reduced actual team."""
    primary, aux = _bases()
    coefficients = np.random.default_rng(402).normal(size=(primary.nbf(), 16)) / primary.nbf()
    jk = _jk(primary, aux, threads=threads)
    jk.initialize()
    # Explicit identical right coefficients take GEMM; omitted right takes the
    # exact symmetric SYRK contraction and mirrored-triangle reduction.
    general = _compute(jk, coefficients, coefficients)
    symmetric = _compute(jk, coefficients)
    for expected, actual in zip(general, symmetric):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    jk.finalize()


@pytest.mark.parametrize("omega", [None, 0.0, 0.3])
@pytest.mark.parametrize("threads", [1, 2])
@uusing("simint")
def test_memdf_exact_simint_fallback(omega, threads):
    primary, aux = _bases()
    # Use d-shell-or-lower auxiliary functions for minimally configured SIMINT.
    aux = primary
    coefficients = np.random.default_rng(403).normal(size=(primary.nbf(), 5)) / primary.nbf()
    results = []
    for package in ("libint2", "simint"):
        jk = _jk(primary, aux, package=package, omega=omega, threads=threads)
        jk.initialize()
        results.append(_compute(jk, coefficients))
        jk.finalize()
    for expected, actual in zip(*results):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)


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
