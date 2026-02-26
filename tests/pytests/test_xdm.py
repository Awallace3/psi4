import psi4
import pytest
import numpy as np

from pprint import pprint as pp
# pytestmark = [pytest.mark.psi, pytest.mark.api]


def test_water_xdm():
    """Test XDM on water dimer."""
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvtz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )
    e_reg, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
    e, wfn = psi4.energy("b3lyp-xdm", molecule=mol, return_wfn=True)
    print(e)
    qcvars = psi4.core.variables()
    pp(qcvars)
    # set np print options to have commas, no truncation and 12 decimal places
    print(qcvars["XDM C6 COEFFICIENTS"].np)
    wfn_vars = wfn.variables()
    print(wfn_vars["XDM C6 COEFFICIENTS"].np)
    print(f"Regular DFT energy: {e_reg}")
    print(f"XDM correction: {e - e_reg}")
    pp(wfn_vars)
    # check that "DISPERSION CORRECTION ENERGY" is in wfn variables and is equal to e - e_reg
    assert "DISPERSION CORRECTION ENERGY" in wfn_vars
    assert np.isclose(wfn_vars["DISPERSION CORRECTION ENERGY"], e - e_reg, atol=1e-6)
    assert np.isclose(
        wfn_vars["DFT TOTAL ENERGY"] - wfn_vars["DISPERSION CORRECTION ENERGY"],
        e_reg,
        atol=1e-6,
    )
    return


def test_water_water_xdm_IE():
    """Test XDM on water dimer."""
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
--
0 1
O    1.35062500   0.11146900   0.00000000
H    1.68039800  -0.37374100  -0.75856100
H    1.68039800  -0.37374100   0.75856100
units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvtz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )
    e, wfn = psi4.energy("b3lyp-xdm", molecule=mol, bsse_type="cp", return_wfn=True)
    print(e)
    qcvars = psi4.core.variables()
    pp(qcvars)
    pp(wfn.variables())
    return


def test_nh3_nh3_xdm_IE_energies():
    """Test XDM on water dimer."""
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
    # nh3_nh3: -2.102, -3.133, -2.953, error: 0.180

    dimer = psi4.geometry("""0 1
N -1.578718 -0.046611 0.000000
H -2.158621 0.136396 -0.809565
H -2.158621 0.136396 0.809565
H -0.849471 0.658193 0.000000
--
0 1
N 1.578718 0.046611 0.000000
H 2.158621 -0.136396 -0.809565
H 0.849471 -0.658193 0.000000
H 2.158621 -0.136396 0.809565

units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvtz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )
    e_dimer, wfn_dimer = psi4.energy(
        "b3lyp-xdm", molecule=dimer, return_wfn=True, bsse_type="cp"
    )
    # pp(wfn_dimer.variables())
    e_dimer, wfn_dimer = psi4.energy(
        "b3lyp-xdm", molecule=dimer, return_wfn=True, bsse_type="nocp"
    )
    return


def test_nh3_ghosts():
    """Test XDM on water dimer."""
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
    # nh3_nh3: -2.102, -3.133, -2.953, error: 0.180

    m = psi4.geometry("""0 1
Gh(N) -1.578718 -0.046611 0.000000
Gh(H) -2.158621 0.136396 -0.809565
Gh(H) -2.158621 0.136396 0.809565
Gh(H) -0.849471 0.658193 0.000000
--
0 1
N 1.578718 0.046611 0.000000
H 2.158621 -0.136396 -0.809565
H 0.849471 -0.658193 0.000000
H 2.158621 -0.136396 0.809565

units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )
    e_m, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    pp(wfn_m.variables())
    # shapes of XDM C6 COEFFICIENTS should be (4, 4)
    print(wfn_m.variables()["XDM C6 COEFFICIENTS"].np)
    print(wfn_m.variables()["XDM PAIRWISE ENERGY"].np)
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (4, 4)
    m = psi4.geometry("""0 1
N 1.578718 0.046611 0.000000
H 2.158621 -0.136396 -0.809565
H 0.849471 -0.658193 0.000000
H 2.158621 -0.136396 0.809565

units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )
    e_m, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    pp(wfn_m.variables())
    # shapes of XDM C6 COEFFICIENTS should be (4, 4)
    print(wfn_m.variables()["XDM C6 COEFFICIENTS"].np)
    print(wfn_m.variables()["XDM PAIRWISE ENERGY"].np)
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (4, 4)
    return


def test_nh3_nh3_xdm_IE():
    """Test XDM on water dimer."""
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
    monA = psi4.geometry("""0 1
N -1.578718 -0.046611 0.000000
H -2.158621 0.136396 -0.809565
H -2.158621 0.136396 0.809565
H -0.849471 0.658193 0.000000

units angstrom
    """)
    monB = psi4.geometry("""0 1
N 1.578718 0.046611 0.000000
H 2.158621 -0.136396 -0.809565
H 0.849471 -0.658193 0.000000
H 2.158621 -0.136396 0.809565
units angstrom
    """)

    dimer = psi4.geometry("""0 1
N -1.578718 -0.046611 0.000000
H -2.158621 0.136396 -0.809565
H -2.158621 0.136396 0.809565
H -0.849471 0.658193 0.000000
--
0 1
N 1.578718 0.046611 0.000000
H 2.158621 -0.136396 -0.809565
H 0.849471 -0.658193 0.000000
H 2.158621 -0.136396 0.809565

units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
            "XDM_DISPERSION_PARAMETERS": [0.0, 5.0],
        }
    )
    e_monA, wfn_A = psi4.energy("b3lyp-xdm", molecule=monA, return_wfn=True)
    e_monB, wfn_B = psi4.energy("b3lyp-xdm", molecule=monB, return_wfn=True)
    pp(wfn_A.variables())
    pp(wfn_B.variables())
    e_dimer, wfn_dimer = psi4.energy("b3lyp-xdm", molecule=dimer, return_wfn=True)
    pp(wfn_dimer.variables())
    e, wfn = psi4.energy("b3lyp", molecule=dimer, bsse_type="nocp", return_wfn=True)
    print(e)
    # in kcal/mol
    qcvars = psi4.core.variables()
    pp(qcvars)
    pp(wfn.variables())
    e, wfn = psi4.energy("b3lyp", molecule=dimer, bsse_type="cp", return_wfn=True)
    return


if __name__ == "__main__":
    # pytest.main([__file__, "-x", "-v"])
    # test_water_xdm()
    # test_water_water_xdm_IE()
    # test_nh3_nh3_xdm_IE_energies()
    test_nh3_ghosts()
