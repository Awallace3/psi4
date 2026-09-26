import numpy as np
import pytest

import psi4

from utils import compare_values

pytestmark = [pytest.mark.psi, pytest.mark.api]

# Reference data generated from Psi's dfmp2 module
data = {
    "df-mp2 ae": np.array([[ 0, 0,  9.62190509e-03],
         [0,  5.49835030e-03, -4.81095255e-03],
         [0, -5.49835030e-03, -4.81095255e-03]]),
    "df-mp2 fc": np.array([[ 0, 0,  1.02432654e-02],
         [0,  5.88581965e-03, -5.12163268e-03],
         [0, -5.88581965e-03, -5.12163268e-03]]),
    "df-mp2 fv": np.array([[ 0, 0,  1.22918833e-02],
         [0,  5.52107556e-03, -6.14594166e-03],
         [0, -5.52107556e-03, -6.14594166e-03]]),
    "df-mp2 fc/fv": np.array([[ 0, 0,  1.25984187e-02],
         [0,  5.71563223e-03, -6.29920936e-03],
         [0, -5.71563223e-03, -6.29920936e-03]]),
    "df-dct": np.array([[0, 0, 0.008477558394],
        [0,  0.005148825942, -0.004238779197],
        [0, -0.005148825942, -0.004238779197]]),
    "df-cc2": np.array([[0, 0, 0.011903811700],
        [0,  0.006730035450, -0.005951905850],
        [0, -0.006730035450, -0.005951905850]])
    }

@pytest.mark.findif
@pytest.mark.slow
# TODO: That "true" needs to be a string is silly. Convert it to a boolean when you can do that without incurring a NaN energy.
@pytest.mark.parametrize("inp", [
    pytest.param({'name': 'mp2', 'options': {'mp2_type': 'df'}, 'ref': data["df-mp2 ae"]}, id='df-mp2 ae'),
    pytest.param({'name': 'mp2', 'options': {'mp2_type': 'df', 'freeze_core': 'true'}, 'ref': data["df-mp2 fc"]}, id='df-mp2 fc'),
    pytest.param({'name': 'mp2', 'options': {'mp2_type': 'df', 'num_frozen_uocc': 4}, 'ref': data["df-mp2 fv"]}, id='df-mp2 fv'),
    pytest.param({'name': 'mp2', 'options': {'mp2_type': 'df', 'freeze_core': 'true', 'num_frozen_uocc': 4}, 'ref': data["df-mp2 fc/fv"]}, id='df-omp2 fc/fv'),
    pytest.param({'name': 'dct', 'options': {'dct_type': 'df'}, 'ref': data["df-dct"]}, id='df-rdct'),
    pytest.param({'name': 'dct', 'options': {'dct_type': 'df', 'reference': 'uhf'}, 'ref': data["df-dct"]}, id='df-udct'),
    pytest.param({'name': 'cc2', 'options': {'cc_type': 'df', 'qc_module': 'ccenergy'}, 'ref': data["df-cc2"]}, id='df-cc2')
    ]
)
def test_gradient(inp):
    h2o = psi4.geometry("""
        O
        H 1 0.958
        H 1 0.958 2 104.5
    """)

    psi4.set_options({'basis': 'aug-cc-pvdz', 'points': 5})
    psi4.set_options(inp['options'])
    
    analytic_gradient = psi4.gradient(inp['name'], dertype=1)
    print(analytic_gradient)
    findif_gradient = psi4.gradient(inp['name'], dertype=0)
    reference_gradient = inp["ref"]

    assert compare_values(findif_gradient, analytic_gradient, 5, "analytic vs. findif gradient")
    assert compare_values(reference_gradient, analytic_gradient.np, 5, "analytic vs. reference gradient")

@pytest.mark.findif
@pytest.mark.dft
@pytest.mark.gradient
@pytest.mark.parametrize("method", ["vv10", "wb97m-v"])
@pytest.mark.parametrize("inp", [
    pytest.param({"mol": "O\nH 1 0.958\nH 1 0.958 2 104.5\n", "reference": "rks"}, id="rks-h2o"),
    pytest.param({"mol": "0 2\nN\nH 1 1.024\nH 1 1.024 2 103.3\n", "reference": "uks"}, id="uks-nh2"),
])
def test_vv10_gradient_findif(inp, method):
    """GGA and hybrid meta-GGA VV10 gradients against finite differences of energies.

    The analytic gradients use fixed grids, so agreement with displaced, atom-centered
    energy grids is quadrature-limited. The meta-GGA cases also exercise the GGA-only
    VV10 contraction without reusing the local functional's tau potential.
    """
    psi4.geometry(inp["mol"])
    psi4.set_options({
        "basis": "cc-pvdz",
        "scf_type": "df",
        "reference": inp["reference"],
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
        "dft_radial_points": 99,
        "dft_spherical_points": 590,
        "dft_vv10_radial_points": 50,
        "dft_vv10_spherical_points": 146,
        "points": 3,
        "disp_size": 0.005,
    })

    analytic_gradient = psi4.gradient(method, dertype=1)
    findif_gradient = psi4.gradient(method, dertype=0)

    assert compare_values(findif_gradient, analytic_gradient, 5, f"{method} analytic vs. findif gradient")


@pytest.mark.dft
@pytest.mark.gradient
def test_vv10_postscf_gradient_rejected():
    """Post-SCF VV10 needs orbital response, even through the direct SCF gradient API."""
    psi4.geometry("H\nH 1 0.8")
    psi4.set_options({
        "basis": "sto-3g",
        "dft_vv10_postscf": True,
        "dft_radial_points": 30,
        "dft_spherical_points": 110,
        "dft_vv10_radial_points": 20,
        "dft_vv10_spherical_points": 50,
    })
    _, wfn = psi4.energy("vv10", return_wfn=True)
    with pytest.raises(RuntimeError, match="post-SCF VV10"):
        psi4.core.scfgrad(wfn)


def test_gradient_ref():
    h2o = psi4.geometry("""
        O
        H 1 0.958
        H 1 0.958 2 104.5
    """)

    psi4.set_options({"basis": "cc-pVDZ"})
    mp2_wfn = psi4.energy("mp2", return_wfn=True)[1]
    with pytest.raises(TypeError):
        psi4.gradient("scf", ref_wfn=mp2_wfn)
    scf_wfn = psi4.energy("scf", return_wfn=True)[1]
    psi4.gradient("scf", ref_wfn=scf_wfn)
