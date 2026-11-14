"""Lifecycle coverage for the C++-only DFHelper tensor release API."""
import importlib.util
import os

import numpy as np
import psi4
import pytest

if importlib.util.find_spec("_dfhelper_lifecycle_test") is None:
    if os.environ.get("PSI4_REQUIRE_DFHELPER_TEST"):
        raise ImportError("CTest requires the built _dfhelper_lifecycle_test module")
    pytest.skip("Build the standalone DFHelper lifecycle module first", allow_module_level=True)
from _dfhelper_lifecycle_test import release_tensor


@pytest.mark.parametrize("ao_core", [False, True])
@pytest.mark.parametrize("mo_core", [False, True])
@pytest.mark.parametrize("order", ["Qpq", "pQq", "pqQ"])
def test_transform_after_release(ao_core, mo_core, order, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    psi4.core.clean_options()
    psi4.core.clean_timers()
    psi4.set_num_threads(2)
    psi4.set_options({"scf_subtype": "INCORE" if ao_core else "OUT_OF_CORE"})
    molecule = psi4.geometry("He 0 0 0\nsymmetry c1")
    primary = psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz")
    auxiliary = psi4.core.BasisSet.build(molecule, "ORBITAL", "def2-universal-jkfit")
    helper = psi4.core.DFHelper(primary, auxiliary)
    helper.set_memory(1_000_000)
    helper.set_MO_core(mo_core)
    helper.initialize()
    assert helper.get_AO_core() == ao_core
    assert helper.get_MO_core() == mo_core
    rng = np.random.default_rng(43)
    helper.add_space("p", psi4.core.Matrix.from_array(rng.random((primary.nbf(), 2))))
    helper.add_space("q", psi4.core.Matrix.from_array(rng.random((primary.nbf(), 3))))
    helper.add_transformation("A", "p", "q", order)
    helper.add_transformation("B", "q", "p", order)
    helper.transform()
    expected = np.asarray(helper.get_tensor("B")).copy()

    release_tensor(helper, "A")
    release_tensor(helper, "A")  # Repeated release is harmless.
    with pytest.raises(RuntimeError, match="A not found"):
        helper.get_tensor_size("A")
    np.testing.assert_allclose(helper.get_tensor("B"), expected)
    helper.transform()
    np.testing.assert_allclose(helper.get_tensor("B"), expected)
    release_tensor(helper, "B")
    helper.add_transformation("C", "q", "p", order)
    helper.transform()
    np.testing.assert_allclose(helper.get_tensor("C"), expected)
    helper.clear_all()
    helper.release_AO()
