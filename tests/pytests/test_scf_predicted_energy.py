"""Check the opt-in energy test with closed- and open-shell densities."""
import numpy as np
import psi4
import pytest


@pytest.mark.psi
@pytest.mark.api
@pytest.mark.quick
@pytest.mark.parametrize("method,reference,charge_mult", [
    ("hf", "rhf", "0 1"), ("pbe", "uks", "1 2"),
])
def test_predicted_energy_convergence(method, reference, charge_mult):
    mol = psi4.geometry(charge_mult + "\nO\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1")
    psi4.set_options({"basis": "def2-svp", "reference": reference, "scf_type": "df",
                      "e_convergence": 9, "d_convergence": 7})
    assert not psi4.core.get_option("SCF", "SCF_E_CONV_PREDICTED")
    energy, wfn = psi4.energy(method, molecule=mol, return_wfn=True)
    psi4.set_options({"scf_e_conv_predicted": True})
    predicted, other = psi4.energy(method, molecule=mol, return_wfn=True)
    assert abs(energy - predicted) < 1e-9
    for name in ("Da", "Db"):
        np.testing.assert_allclose(getattr(wfn, name)().np, getattr(other, name)().np,
                                   atol=5e-6, rtol=0)
