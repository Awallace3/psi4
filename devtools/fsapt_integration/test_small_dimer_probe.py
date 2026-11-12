from types import SimpleNamespace

import pytest

from small_dimer_probe import basis_receipt, first_dimer_only, trajectory


def test_workspace_identity_survives_private_worker_home(monkeypatch, tmp_path):
    from cachy_dimer_screen import dedicated_root
    expected = dedicated_root()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert dedicated_root() == expected
    assert not dedicated_root().is_relative_to(tmp_path)


def test_worker_alias_passes_real_sealed_loader_check(tmp_path):
    import importlib.util
    from pathlib import Path
    from cachy_dimer_screen import worker_alias
    source = Path(__file__).resolve().parents[1] / "disp0/phoenix_validation/gpu_runtime.py"
    spec = importlib.util.spec_from_file_location("checked_gpu_runtime", source)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    parent = tmp_path / "aliases"
    parent.mkdir(mode=0o700)
    config = dict(schema=2, identity_contract="process-owned-kernel-sealed-snapshots-v1",
                  sealed_alias_parent=str(parent), dependency_environment="/qualified/env",
                  startup_library_path="/qualified/env/lib")
    child = worker_alias(parent, "cpu-energy")
    assert runtime.sealed_startup_path(config, child) == "/qualified/env/lib"
    child.chmod(0o755)
    with pytest.raises(RuntimeError, match="Private per-worker"):
        runtime.sealed_startup_path(config, child)


def test_native_trajectory_and_second_scf_rejection():
    text = """
   @DF-RHF iter SAD: -1.75106606335984e3 -1.75107e+03 0.00000e+00
   @DF-RHF iter   1: -1752.79035103791716 -1.72429e+00 1.54790e-03 ADIIS/DIIS
"""
    rows = trajectory(text)
    assert rows[0]["iteration"] == 0
    assert rows[1]["energy_Eh"] == "-1752.79035103791716"
    assert rows[1]["tags"] == "ADIIS/DIIS"
    with pytest.raises(RuntimeError, match="second SCF"):
        trajectory(text + text)
    with pytest.raises(RuntimeError, match="No native"):
        trajectory("Startup failure")


def test_real_entrypoint_called_and_no_monomer_continuation():
    mol, wfn = object(), object()
    calls = []

    def helper(*args, **kwargs):
        calls.append("dimer")
        assert kwargs["molecule"] is mol
        return wfn

    module = SimpleNamespace(scf_helper=helper)

    def energy(method, *, molecule):
        assert method == "sapt(dft)" and molecule is mol
        calls.append("preamble")
        module.scf_helper("SCF", molecule=molecule)
        pytest.fail("Monomer continuation reached")

    assert first_dimer_only(SimpleNamespace(energy=energy), mol, module) is wfn
    assert calls == ["preamble", "dimer"]
    assert module.scf_helper is helper


def test_scf_failure_propagates_and_patch_restores():
    mol = object()

    def helper(*args, **kwargs):
        raise ValueError("scientific failure")

    module = SimpleNamespace(scf_helper=helper)
    psi4 = SimpleNamespace(energy=lambda *a, **k: module.scf_helper("SCF", molecule=mol))
    with pytest.raises(ValueError, match="scientific failure"):
        first_dimer_only(psi4, mol, module)
    assert module.scf_helper is helper


def test_basis_receipt_changes_with_normalized_coefficients():
    shell = SimpleNamespace(ncenter=0, am=0, nprimitive=1,
                            is_pure=lambda: True, coord=lambda k: 0.0,
                            exp=lambda k: 1.25, coef=lambda k: 0.5)
    basis = SimpleNamespace(nshell=lambda: 1, nbf=lambda: 1, shell=lambda i: shell)
    first = basis_receipt(basis)
    assert first["shells"][0]["coefficients"] == [float(0.5).hex()]
    assert basis_receipt(basis) == first
    shell.coef = lambda k: 0.5000000000000001
    assert basis_receipt(basis)["sha256"] != first["sha256"]
