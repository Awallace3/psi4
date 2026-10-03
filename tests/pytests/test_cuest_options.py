"""CPU-only validation of unsupported cuEST/SCF option combinations."""

import importlib
from types import SimpleNamespace

import pytest

scf_iterator = importlib.import_module("psi4.driver.procrouting.scf_proc.scf_iterator")

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("optimizer", ["INTERNAL", "OOO", "OPENORBITALOPTIMIZER"])
@pytest.mark.parametrize("use_cuest", [False, True])
@pytest.mark.parametrize("cuest_xc", [False, True])
@pytest.mark.parametrize("needs_xc", [False, True])
def test_validate_ooo_cuest(optimizer, use_cuest, cuest_xc, needs_xc, monkeypatch):
    """Only OOO DFT with active GPU XC is rejected; HF and CPU XC remain allowed."""
    options = {
        "ORBITAL_OPTIMIZER_PACKAGE": optimizer,
        "USE_CUEST": use_cuest,
        "CUEST_XC": cuest_xc,
    }
    monkeypatch.setattr(scf_iterator.core, "get_option", lambda module, name: options[name])
    wfn = SimpleNamespace(functional=lambda: SimpleNamespace(needs_xc=lambda: needs_xc))

    if optimizer != "INTERNAL" and use_cuest and cuest_xc and needs_xc:
        with pytest.raises(scf_iterator.ValidationError, match="OpenOrbitalOptimizer with cuEST XC"):
            scf_iterator._validate_ooo_cuest(wfn)
    else:
        scf_iterator._validate_ooo_cuest(wfn)


@pytest.mark.parametrize("optimizer", ["OOO", "OPENORBITALOPTIMIZER"])
def test_ooo_cuest_rejected_before_iterations(optimizer, monkeypatch):
    """Raise before touching the SCF state or entering the native OOO callbacks."""
    options = {
        "ORBITAL_OPTIMIZER_PACKAGE": optimizer,
        "USE_CUEST": True,
        "CUEST_XC": True,
    }
    monkeypatch.setattr(scf_iterator.core, "get_option", lambda module, name: options[name])
    wfn = SimpleNamespace(functional=lambda: SimpleNamespace(needs_xc=lambda: True))
    with pytest.raises(scf_iterator.ValidationError, match="CUEST_XC false"):
        scf_iterator.scf_iterate(wfn)
