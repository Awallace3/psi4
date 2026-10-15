"""SCF lifecycle timers must nest correctly and survive failed SCFs."""

import importlib

import pytest

from psi4 import core
from psi4.driver import p4util

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def test_timer_balanced_on_exception(monkeypatch):
    events = []
    monkeypatch.setattr(core, "timer_on", lambda name: events.append(("on", name)))
    monkeypatch.setattr(core, "timer_off", lambda name: events.append(("off", name)))
    with pytest.raises(RuntimeError, match="probe"):
        with p4util.timer("outer"):
            with p4util.timer("inner"):
                raise RuntimeError("probe")
    assert events == [("on", "outer"), ("on", "inner"), ("off", "inner"), ("off", "outer")]


def test_timer_decorator_reusable(monkeypatch):
    events = []
    monkeypatch.setattr(core, "timer_on", lambda name: events.append(("on", name)))
    monkeypatch.setattr(core, "timer_off", lambda name: events.append(("off", name)))

    @p4util.timer("call")
    def square(value):
        return value * value

    assert square(2) == 4
    assert square(3) == 9
    assert square.__name__ == "square"
    assert events == [("on", "call"), ("off", "call")] * 2


def test_scf_lifecycle_timers(tmp_path, monkeypatch):
    """Drive a real CPU SCF, including properties and checkpoint serialization."""
    import psi4

    psi4.core.clean_options()
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk", "guess": "core"})
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    stack, seen = [], []
    original_on, original_off = core.timer_on, core.timer_off

    def start(name):
        original_on(name)
        if name.startswith("SCF:"):
            stack.append(name)
            seen.append(name)

    def stop(name):
        if name.startswith("SCF:"):
            assert stack.pop() == name
        original_off(name)

    monkeypatch.setattr(core, "timer_on", start)
    monkeypatch.setattr(core, "timer_off", stop)
    _, wfn = psi4.energy("hf", molecule=mol, return_wfn=True,
                         write_orbitals=str(tmp_path / "orbitals"))
    assert wfn.energy() < -2.0
    assert not stack
    assert {
        "SCF: Driver", "SCF: Wavefunction build", "SCF: Wfn factory",
        "SCF: Functional", "SCF: Native constructor", "SCF: Initialize",
        "SCF: JK initialize", "SCF: Iterations", "SCF: Finalize energy",
        "SCF: Native finalize", "SCF: Properties", "SCF: Checkpoint",
    } <= set(seen)
    assert (tmp_path / "orbitals.npy").exists()


def test_scf_initialize_timer_closes_on_failure(monkeypatch):
    iterator = importlib.import_module("psi4.driver.procrouting.scf_proc.scf_iterator")
    events = []
    monkeypatch.setattr(core, "timer_on", lambda name: events.append(("on", name)))
    monkeypatch.setattr(core, "timer_off", lambda name: events.append(("off", name)))
    # Missing wavefunction attributes fail immediately inside the decorated function.
    with pytest.raises(AttributeError):
        iterator.scf_initialize(object())
    assert events[0] == ("on", "SCF: Initialize")
    assert events[-1] == ("off", "SCF: Initialize")


def test_sad_basis_helper_forces_spherical_fit(monkeypatch):
    """The READ fallback uses the same spherical auxiliary contract as SAD."""
    from types import SimpleNamespace

    proc = importlib.import_module("psi4.driver.procrouting.proc")
    calls = []
    psi_options = {"PUREAM": False, "BASIS": "cc-pvdz"}

    def build(*args, **kwargs):
        calls.append((args[1], kwargs["puream"]))
        return []

    monkeypatch.setattr(core.BasisSet, "build", build)
    monkeypatch.setattr(core, "get_global_option", lambda key: psi_options[key])
    monkeypatch.setattr(core, "get_option", lambda mod, key: {"SAD_SCF_TYPE": "DF", "DF_BASIS_SAD": ""}[key])
    # Use real option save/restore; only mock the heavyweight basis builds.
    wfn = SimpleNamespace(
        molecule=lambda: None,
        basisset=lambda: SimpleNamespace(has_puream=lambda: False),
        set_sad_basissets=lambda value: None,
        set_sad_fitting_basissets=lambda value: None,
    )
    proc._set_sad_basissets(wfn)
    assert calls == [("ORBITAL", False), ("DF_BASIS_SAD", True)]


def test_sad_basis_failure_restores_options(monkeypatch):
    from types import SimpleNamespace

    proc = importlib.import_module("psi4.driver.procrouting.proc")
    core.set_global_option("PUREAM", False)
    changed = core.has_global_option_changed("PUREAM")

    def build(mol, key, *args, **kwargs):
        if key == "DF_BASIS_SAD":
            assert core.get_global_option("PUREAM")
            raise RuntimeError("fitting basis failed")
        return []

    monkeypatch.setattr(core.BasisSet, "build", build)
    wfn = SimpleNamespace(
        molecule=lambda: None,
        basisset=lambda: SimpleNamespace(has_puream=lambda: False),
        set_sad_basissets=lambda value: None,
    )
    with pytest.raises(RuntimeError, match="fitting basis failed"):
        proc._set_sad_basissets(wfn)
    assert not core.get_global_option("PUREAM")
    assert core.has_global_option_changed("PUREAM") == changed
