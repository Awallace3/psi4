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
