import psi4
import pytest
import numpy as np
from psi4 import compare_values
from pprint import pprint as pp

pytestmark = [pytest.mark.psi, pytest.mark.api]


@pytest.mark.xdm
def test_water_xdm():
    """Verify XDM energy bookkeeping for a single water molecule.

    Checks that the reported dispersion correction equals the difference between
    `b3lyp-xdm` and plain `b3lyp`, and that wavefunction variables are
    internally consistent.
    """
    # psi4.set_num_threads(12)
    # psi4.set_memory("32 GB")
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
    ref_disp_corr = -0.0036307541858282655
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
    ref_disp_corr = -0.0038699050025764892
    assert np.isclose(disp_corr, ref_disp_corr, atol=1e-6), (
        f"Expected dispersion correction {ref_disp_corr}, got {disp_corr}"
    )
    return


@pytest.mark.xdm
def test_h2o_nh3_xdm_IE_CP_NOCP():
    """
    Validate H2O-NH3 dimer XDM interaction energies for CP and NoCP workflows.

    Confirms counterpoise and non-counterpoise paths reproduce their reference
    energies, exercising distinct XDM damping-parameter selections.
    """
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
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
    ref_e_cp = -0.0006511131014690363
    ref_e_nocp = -0.0006791849756098145
    ref_e_nocp_losii = -0.0009743139328008965
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
        }
    )
    e_cp_losii, wfn_cp_losii = psi4.energy(
        "b3lyp-xdm(los-ii)", molecule=dimer, bsse_type="cp", return_wfn=True
    )
    print(e_cp_losii)
    assert compare_values(e_cp_losii, ref_e_nocp_losii, 8, "CP XDM(LoS-II) energy")
    psi4.set_options(
        {
            "basis": "sto-3g",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
            "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
        }
    )
    e_cp, wfn_cp = psi4.energy(
        "b3lyp-xdm", molecule=dimer, bsse_type="cp", return_wfn=True
    )
    print(e_cp)
    assert compare_values(e_cp, ref_e_cp, 8, "CP XDM energy")
    e_nocp, wfn_nocp = psi4.energy(
        "b3lyp-xdm", molecule=dimer, bsse_type="nocp", return_wfn=True
    )
    print(e_nocp)
    assert compare_values(e_nocp, ref_e_nocp, 8, "No CP XDM energy")
    return


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

    # TODO: Update the reference values once parameters are refined.
    assert compare_values(e_alias, e_kb49, 10, "-XDM alias equals -XDM(KB49)")
    assert not np.isclose(e_los_ii, e_kb49, rtol=0.0, atol=1.0e-8)
    return


if __name__ == "__main__":
    # pytest.main([__file__, "-x", "-v"])
    # test_xdm_models_and_alias()
    test_h2o_nh3_xdm_IE_CP_NOCP()
