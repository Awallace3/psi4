import psi4
import pytest
import numpy as np
import re
from psi4 import compare_values

pytestmark = [pytest.mark.psi, pytest.mark.api]


@pytest.mark.xdm
def test_water_xdm():
    """Verify XDM energy bookkeeping for a single water molecule.

    Checks that the reported dispersion correction equals the difference between
    `b3lyp-xdm` and plain `b3lyp`, and that wavefunction variables are
    internally consistent.
    """
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
    psi4.set_options({"BASIS_GUESS": "sto-3g"})
    e, wfn = psi4.energy("b3lyp-xdm", molecule=mol, return_wfn=True)
    psi4.set_options({"BASIS_GUESS": False})
    wfn_vars = wfn.variables()
    assert "DISPERSION CORRECTION ENERGY" in wfn_vars
    assert np.isclose(wfn_vars["DISPERSION CORRECTION ENERGY"], e - e_reg, atol=1e-6)
    assert np.isclose(
        wfn_vars["DFT TOTAL ENERGY"] - wfn_vars["DISPERSION CORRECTION ENERGY"],
        e_reg,
        atol=1e-6,
    )

    psi4.set_options({"REFERENCE": "UKS"})
    e_uks = psi4.energy("b3lyp-xdm", molecule=mol)
    assert compare_values(e, e_uks, 8, "RKS and UKS XDM energies")
    psi4.set_options({"REFERENCE": "RKS"})


@pytest.mark.xdm
def test_h2o_ghosts():
    """Ensure XDM pairwise outputs size correctly with and without ghosts.

    Runs a ghost-containing fragment calculation and a normal water monomer, then
    confirms the XDM C6 matrix shape reflects only real atoms. Also confirms
    the XDM energy matches the reference
    """

    m = psi4.geometry("""
0 1
Gh(O)    -1.55100700  -0.11452000   0.00000000
Gh(H)    -1.93425900   0.76250300   0.00000000
Gh(H)    -0.59967700   0.04071200   0.00000000
--
0 1
O    1.35062500   0.11146900   0.00000000
H    1.68039800  -0.37374100  -0.75856100
H    1.68039800  -0.37374100   0.75856100
units angstrom
    """)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
            "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
        }
    )
    e_m, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    disp_corr = wfn_m.variables()["DISPERSION CORRECTION ENERGY"]
    ref_disp_corr = -0.00474203021866958
    assert np.isclose(disp_corr, ref_disp_corr, atol=1e-6), (
        f"Expected dispersion correction {ref_disp_corr}, got {disp_corr}"
    )
    # shapes of XDM C6 COEFFICIENTS should be (3, 3)
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (3, 3)
    m = psi4.geometry("""
0 1
O    1.35062500   0.11146900   0.00000000
H    1.68039800  -0.37374100  -0.75856100
H    1.68039800  -0.37374100   0.75856100

units angstrom
    """)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
            "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
        }
    )
    e_m, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (3, 3)
    disp_corr = wfn_m.variables()["DISPERSION CORRECTION ENERGY"]
    ref_disp_corr = -0.004982082759239244
    assert np.isclose(disp_corr, ref_disp_corr, atol=1e-6), (
        f"Expected dispersion correction {ref_disp_corr}, got {disp_corr}"
    )


@pytest.mark.xdm
def test_h2o_nh3_xdm_nocp():
    """Validate no-CP XDM interaction energies and reject CP."""

    dimer = psi4.geometry("""0 1
N -1.578718 -0.046611 0.000000
H -2.158621 0.136396 -0.809565
H -2.158621 0.136396 0.809565
H -0.849471 0.658193 0.000000
--
0 1
O    2.35062500   0.11146900   0.00000000
H    2.68039800  -0.37374100  -0.75856100
H    2.68039800  -0.37374100   0.75856100

units angstrom
    """)
    psi4.set_options({"basis": "aug-cc-pvdz"})

    for bsse_type in ["cp", "vmfc", ["nocp", "vmfc"]]:
        with pytest.raises(
            NotImplementedError,
            match="Counterpoise-based XDM energies are not implemented",
        ):
            psi4.energy("b3lyp-xdm", molecule=dimer, bsse_type=bsse_type)

    with pytest.raises(NotImplementedError, match="XDM derivatives are not implemented"):
        psi4.gradient("b3lyp-xdm", molecule=dimer)

    from psi4.driver.task_planner import task_planner

    with pytest.raises(NotImplementedError, match="Counterpoise-based XDM energies are not implemented"):
        task_planner("energy", "b3lyp-xdm/cc-pv[d,t]z", dimer, bsse_type="cp")
    with pytest.raises(NotImplementedError, match="XDM derivatives are not implemented"):
        task_planner("gradient", "b3lyp-xdm/cc-pv[d,t]z", dimer)
    with pytest.raises(NotImplementedError, match="XDM derivatives are not implemented"):
        task_planner("gradient", "b3lyp", dimer, bsse_type="nocp", levels={1: "b3lyp-xdm"})

    e_los_ii = psi4.energy("b3lyp-xdm(los-ii)", molecule=dimer, bsse_type="nocp")
    assert compare_values(e_los_ii, -0.0010595757, 8, "No-CP XDM(LoS-II) energy")

    psi4.set_options(
        {
            "basis": "sto-3g",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
            "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
        }
    )
    e_nocp = psi4.energy("b3lyp-xdm", molecule=dimer, bsse_type="nocp")
    assert compare_values(e_nocp, -0.0006344572, 8, "No-CP XDM energy")


@pytest.mark.xdm
def test_xdm_models_and_alias():
    """Check XDM model aliasing and explicit LOS-II selection.

    Verifies that ``-xdm`` and ``-xdm(kb49)`` are equivalent, and that
    ``-xdm(los-ii)`` selects a different parameterization.
    """

    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )

    e_alias = psi4.energy("b3lyp-xdm", molecule=mol)
    e_kb49 = psi4.energy("b3lyp-xdm(kb49)", molecule=mol)
    e_los_ii = psi4.energy("b3lyp-xdm(los-ii)", molecule=mol)

    assert compare_values(e_alias, e_kb49, 10, "-XDM alias equals -XDM(KB49)")
    assert not np.isclose(e_los_ii, e_kb49, rtol=0.0, atol=1.0e-8)


@pytest.mark.xdm
def test_xdm_unsupported_functional():
    """Ensure unsupported functionals fail unless parameters are provided."""

    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )
    err_msg = (
        "XDMDispersion: No fitted BJ parameters for hf/sto-3g with model kb49. "
        "Provide [a1, a2] through XDM_DISPERSION_PARAMETERS."
    )
    with pytest.raises(
        psi4.p4util.ValidationError,
        match=re.escape(err_msg),
    ):
        psi4.energy("hf-xdm", molecule=mol)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
            "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
        }
    )
    e_ref = -74.9690725681
    e = psi4.energy("hf-xdm", molecule=mol)
    assert compare_values(e, e_ref, 8, "HF-XDM energy with custom parameters")
