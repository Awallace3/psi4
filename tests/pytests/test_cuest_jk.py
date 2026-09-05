"""
cuEST J/K with C_left != C_right.

``JK::compute()`` supports two shapes of exchange matrix.  The symmetric one,
K[mu][nu] = sum_i (mu lambda|nu sigma) C[lambda][i] C[sigma][i], is all an SCF
ever asks for.  SAPT is different: nearly every exchange term contracts one set
of orbitals against a *different* one, and Psi4 expresses that by pushing
distinct matrices through ``C_left_add`` and ``C_right_add``.

cuEST has separate kernels for the two cases -- ``cuestDFSymmetricExchangeCompute``
and, since v0.2.0, ``cuestDFNonsymmetricExchangeCompute``.  Handing an asymmetric
request to the symmetric kernel is not an approximation with a small error: it
silently computes K(C_left, C_left) and returns it as K(C_left, C_right).  The
result is finite, smooth, and wrong, and in SAPT(DFT)-D4(I) it moved E_exch by
38% while leaving E_elst -- which needs only J -- exact to every printed digit.

These tests are deliberately at the J/K layer rather than at the SAPT layer,
because a failure here says "the exchange builder is wrong" instead of "one of
several hundred contractions disagrees".  The coefficient matrices are random:
K's correctness does not depend on the columns being orbitals, and random
columns make C_left and C_right maximally unlike each other.
"""

import numpy as np
import pytest

import psi4

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

_water_dimer = """
0 1
O   -0.702196054   -0.056060256   0.009942262
H   -1.022193224    0.846775782   -0.011488714
H    0.257521062    0.042121496    0.005218999
--
0 1
O    2.268880784    0.026340101    0.000508029
H    2.645502399   -0.412039965    0.766632411
H    2.641145101   -0.449872874   -0.744894473
units angstrom
symmetry c1
no_reorient
no_com
"""


def _basis_sets():
    psi4.core.clean()
    psi4.core.clean_options()
    mol = psi4.geometry(_water_dimer)
    mol.update_geometry()
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")
    return primary, aux


def _run_jk(use_cuest, primary, aux, pairs):
    """Build a J/K object of the requested flavour and evaluate every (C_left, C_right) pair."""
    psi4.set_options({
        "scf_type": "df",
        "USE_CUEST": use_cuest,
        # Compare the two builders in like precision; the emulated-GEMM path has
        # its own tolerance and is not what is under test here.
        "CUEST_MIXED_PRECISION": False,
    })
    jk = psi4.core.JK.build_JK(primary, aux)
    jk.set_do_J(True)
    jk.set_do_K(True)
    jk.initialize()

    jk.C_clear()
    for left, right in pairs:
        jk.C_left_add(psi4.core.Matrix.from_array(left))
        if right is not None:
            jk.C_right_add(psi4.core.Matrix.from_array(right))
    jk.compute()

    Js = [np.array(jk.J()[n]) for n in range(len(pairs))]
    Ks = [np.array(jk.K()[n]) for n in range(len(pairs))]
    name = jk.name()
    jk.finalize()
    return name, Js, Ks


def _random_coeffs(nbf, ncol, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nbf, ncol)) / np.sqrt(nbf)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_cuest_jk_asymmetric_exchange_matches_cpu():
    """K with C_left != C_right must match the CPU DF builder.

    This is the regression test for the cuEST exchange path taking the
    symmetric kernel unconditionally.  Before the fix it failed by O(10%) on
    every element of K while J agreed to machine precision.
    """
    primary, aux = _basis_sets()
    nbf = primary.nbf()
    pairs = [
        (_random_coeffs(nbf, 5, 11), _random_coeffs(nbf, 5, 22)),
        (_random_coeffs(nbf, 3, 33), _random_coeffs(nbf, 3, 44)),
    ]

    cpu_name, cpu_J, cpu_K = _run_jk(False, primary, aux, pairs)
    gpu_name, gpu_J, gpu_K = _run_jk(True, primary, aux, pairs)

    assert gpu_name == "cuESTJK", f"USE_CUEST did not select cuESTJK (got {gpu_name})"
    assert cpu_name != "cuESTJK"

    for n in range(len(pairs)):
        # An asymmetric K really is asymmetric; if it were not, the symmetric
        # kernel could stand in for the nonsymmetric one and this whole test
        # would be checking nothing.
        asymmetry = np.abs(cpu_K[n] - cpu_K[n].T).max()
        assert asymmetry > 1.0e-3, f"pair {n}: reference K is symmetric, test is vacuous"

        assert np.allclose(cpu_J[n], gpu_J[n], atol=1.0e-8), f"J mismatch on pair {n}"
        assert np.allclose(cpu_K[n], gpu_K[n], atol=1.0e-8), (
            f"K mismatch on pair {n}: max |diff| = {np.abs(cpu_K[n] - gpu_K[n]).max():.3e}"
        )


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_cuest_jk_symmetric_exchange_matches_cpu():
    """The symmetric path -- what every SCF uses -- must keep working.

    Passing no ``C_right`` at all is what sets ``lr_symmetric_``, which is the
    flag the new branch keys off, so this pins the other side of that branch.
    """
    primary, aux = _basis_sets()
    nbf = primary.nbf()
    pairs = [(_random_coeffs(nbf, 5, 55), None)]

    _, cpu_J, cpu_K = _run_jk(False, primary, aux, pairs)
    _, gpu_J, gpu_K = _run_jk(True, primary, aux, pairs)

    assert np.abs(cpu_K[0] - cpu_K[0].T).max() < 1.0e-10, "symmetric K should be symmetric"
    assert np.allclose(cpu_J[0], gpu_J[0], atol=1.0e-8)
    assert np.allclose(cpu_K[0], gpu_K[0], atol=1.0e-8)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_cuest_jk_asymmetric_is_not_the_symmetric_answer():
    """Pin the specific wrong answer, so a regression cannot hide behind a loose tolerance.

    K(C_left, C_right) and K(C_left, C_left) are both valid exchange matrices;
    what makes the old behaviour a bug is that it returned the second when asked
    for the first.  Here cuEST is asked for both and the two must differ by far
    more than the CPU/GPU agreement tolerance.
    """
    primary, aux = _basis_sets()
    nbf = primary.nbf()
    left = _random_coeffs(nbf, 5, 11)
    right = _random_coeffs(nbf, 5, 22)

    _, _, K_asym = _run_jk(True, primary, aux, [(left, right)])
    _, _, K_sym = _run_jk(True, primary, aux, [(left, None)])

    assert np.abs(K_asym[0] - K_sym[0]).max() > 1.0e-3
