import importlib

import psi4
import pytest

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("held_mib", [0, 32])
def test_tiny_budget_reserve_with_retained_stores(monkeypatch, held_mib):
    """Only a standalone tiny budget may bypass the working reserve."""
    iterator = importlib.import_module("psi4.driver.procrouting.scf_proc.scf_iterator")
    words_per_mib = 1024**2 // 8
    monkeypatch.setattr(psi4.core, "memory_committed", lambda: held_mib * words_per_mib)
    monkeypatch.setattr(iterator, "_resident_doubles", lambda: None)
    monkeypatch.setattr(iterator, "_scf_memory_reserve", lambda wfn: {
        "base": 220 * words_per_mib, "matrices": 100 * words_per_mib,
        "grid": 0, "points": 0, "jk": 0,
    })
    previous_memory = psi4.core.get_memory()
    psi4.core.clean_options()
    psi4.set_num_threads(2)
    psi4.set_memory("256 MiB")
    psi4.set_options({"basis": "sto-3g", "scf_type": "df", "save_jk": True})
    molecule = psi4.geometry("""
        0 1
        O
        H 1 1.0
        H 1 1.0 2 103.1
        symmetry c1
    """)
    try:
        _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
        if held_mib:
            expected = int((256 - held_mib - 100) * words_per_mib * 0.95)
        else:
            expected = int(256 * words_per_mib * psi4.core.get_option("SCF", "SCF_MEM_SAFETY_FACTOR"))
        assert wfn.memory_jk_ == expected
    finally:
        psi4.set_memory(previous_memory)
