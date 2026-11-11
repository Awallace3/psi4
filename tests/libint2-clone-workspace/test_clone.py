"""Native clone/sieve coverage without adding test-only production bindings."""
import importlib.util
import os
import numpy as np
import psi4
import pytest

if importlib.util.find_spec("_libint2_clone_test") is None:
    if os.environ.get("PSI4_REQUIRE_CLONE_TEST"):
        raise ImportError("CTest requires the built _libint2_clone_test module")
    pytest.skip("Build the standalone native clone test module first", allow_module_level=True)
from _libint2_clone_test import probe


@pytest.fixture
def bases():
    psi4.core.clean_options()
    psi4.set_num_threads(2)
    molecule = psi4.geometry("""
        0 1
        O
        H 1 1.0
        H 1 1.0 2 103.1
        symmetry c1
    """)
    primary = psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz")
    auxiliary = psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz-jkfit")
    return primary, auxiliary, psi4.core.BasisSet.zero_ao_basis_set()


@pytest.mark.parametrize("screening", ["SCHWARZ", "NONE", "CSAM", "DENSITY"])
@pytest.mark.parametrize("rank", [
    2, 3, 4,
    pytest.param("3rev", marks=pytest.mark.skip(reason="Existing xx_xs compute_shell path aborts on the baseline")),
])
@pytest.mark.parametrize("op,deriv", [
    ("coulomb", 0), ("coulomb", 1), ("coulomb", 2),
    ("erf", 0), ("erfc", 0),
    pytest.param("yukawa", 0, marks=pytest.mark.skip(
        reason="Baseline Yukawa constructor starts the same timer twice")),
    ("f12", 0), ("f12_squared", 0), ("f12g12", 0), ("f12dc", 0),
])
def test_clone_sieve(bases, screening, rank, op, deriv):
    primary, auxiliary, zero = bases
    psi4.set_options({"screening": screening})
    basis_tuple = {
        2: (auxiliary, zero, auxiliary, zero),
        3: (auxiliary, zero, primary, primary),
        "3rev": (primary, primary, auxiliary, zero),
        4: (primary, primary, primary, primary),
    }[rank]
    result = probe(psi4.core.IntegralFactory(*basis_tuple), op, deriv)
    if deriv:
        assert np.max(np.abs(result["normal"][0])) > 1e-10
    for values in result["normal"] + result["rebuilt"]:
        np.testing.assert_allclose(values, result["normal"][0], rtol=1e-11, atol=1e-11)
    for values in result["bounds"]:
        np.testing.assert_allclose(values, result["bounds"][0], rtol=1e-11, atol=1e-11)
