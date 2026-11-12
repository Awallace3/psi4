from types import SimpleNamespace

import pytest

from small_dimer_probe import basis_receipt, first_dimer_only, trajectory


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
