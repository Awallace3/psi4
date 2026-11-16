"""Exact DIIS storage optimizations: histories and stopping criteria unchanged."""
from types import SimpleNamespace
import numpy as np
import pytest
import psi4
from psi4.driver.procrouting.diis import DIIS, StoragePolicy
from psi4.driver.procrouting.scf_proc import scf_iterator, subclass_methods

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

@pytest.mark.parametrize("accelerator", ["NONE", "ADIIS", "EDIIS"])
@pytest.mark.parametrize("available", [1000, 100000])
def test_diis_reservation_precedes_jk_partition(monkeypatch, accelerator, available):
    psi4.set_options({"scf_initial_accelerator": accelerator, "diis_max_vecs": 8})
    get_global_option = psi4.core.get_global_option
    monkeypatch.setattr(psi4.core, "get_memory", lambda: available * 8)
    monkeypatch.setattr(
        psi4.core, "get_global_option",
        lambda key: 1.0 if key == "SCF_MEM_SAFETY_FACTOR" else get_global_option(key),
    )
    jk = SimpleNamespace(memory_estimate=lambda: 900)
    monkeypatch.setattr(scf_iterator, "_build_jk", lambda *args: jk)
    wfn = SimpleNamespace(
        diag_scratch_size=lambda: 0, nsopi=lambda: psi4.core.Dimension([1]),
        V_potential=lambda: None, jk=lambda: None, get_print=lambda: False,
        molecule=lambda: SimpleNamespace(), attempt_number_=2,
        form_D=lambda: None, compute_initial_E=lambda: 0.0, set_energies=lambda *args: None,
    )
    scf_iterator.scf_initialize(wfn)
    required = subclass_methods._scf_diis_memory_required(wfn, 8)
    assert wfn.memory_jk_ + wfn.memory_collocation_ + wfn._diis_memory_budget == available
    assert wfn._diis_memory_budget == (
        required if available == 100000 else subclass_methods._scf_diis_disk_scratch(wfn)
    )
    sample = [psi4.core.Matrix(1, 1)] * 6
    policy = subclass_methods._scf_diis_storage_policy(wfn, 8, sample)
    assert policy == (StoragePolicy.InCore if available == 100000 else StoragePolicy.OnDisk)



def test_diis_reservation_reselects_tight_memory_backend(monkeypatch):
    get_option = psi4.core.get_global_option
    monkeypatch.setattr(psi4.core, "get_memory", lambda: 910 * 8)
    monkeypatch.setattr(psi4.core, "get_global_option",
                        lambda key: 1.0 if key == "SCF_MEM_SAFETY_FACTOR" else get_option(key))
    calls = []
    def build(wfn, memory):
        calls.append(memory)
        return SimpleNamespace(memory_estimate=lambda: 900 if memory >= 900 else 500)
    monkeypatch.setattr(scf_iterator, "_build_jk", build)
    wfn = SimpleNamespace(
        diag_scratch_size=lambda: 0, nsopi=lambda: psi4.core.Dimension([1]),
        V_potential=lambda: None, jk=lambda: None, get_print=lambda: False,
        molecule=lambda: SimpleNamespace(), attempt_number_=2,
        form_D=lambda: None, compute_initial_E=lambda: 0.0, set_energies=lambda *args: None,
    )
    scf_iterator.scf_initialize(wfn)
    assert calls == [910, 894]
    assert wfn.memory_jk_ + wfn._diis_memory_budget == 910



def test_streamed_adiis_matches_incore():
    rng = np.random.default_rng(898)
    managers = [
        DIIS(5, "streamed ADIIS", storage_policy=storage, engines={"diis", "adiis"})
        for storage in (StoragePolicy.InCore, StoragePolicy.OnDisk)
    ]
    try:
        for _ in range(5):
            entry = {key: [psi4.core.Matrix.from_array(rng.normal(size=(3, 3)))]
                     for key in ("target", "error", "densities")}
            for manager in managers:
                manager.add_entry(entry)
        for manager in managers:
            manager.adiis_populate()
        np.testing.assert_allclose(managers[0].adiis_linear, managers[1].adiis_linear, atol=1e-13)
        np.testing.assert_allclose(managers[0].adiis_quadratic, managers[1].adiis_quadratic, atol=1e-13)
    finally:
        for manager in managers:
            manager.delete_diis_file()

