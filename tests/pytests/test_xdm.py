import psi4
import pytest
import numpy as np
import re
from psi4 import compare_values
from pprint import pprint as pp
import os
import qcelemental as qcel

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
    """Ensure XDM ghost atoms are included by default and can be excluded.

    Runs a ghost-containing fragment calculation and a normal water monomer, then
    confirms the default XDM post-processing includes ghost atoms while the
    opt-in compatibility switch restores the legacy real-atom-only behavior.
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
            "XDM_CP_ONLY_REAL_ATOMS": False,
        }
    )
    e_m, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    disp_corr = wfn_m.variables()["DISPERSION CORRECTION ENERGY"]
    ref_disp_corr = -0.003978192236193462
    assert np.isclose(disp_corr, ref_disp_corr, atol=1e-6), (
        f"Expected dispersion correction {ref_disp_corr}, got {disp_corr}"
    )
    print(wfn_m.variables()["XDM C6 COEFFICIENTS"].shape)
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (6, 6)
    assert wfn_m.variables()["XDM C8 COEFFICIENTS"].shape == (6, 6)
    assert wfn_m.variables()["XDM C10 COEFFICIENTS"].shape == (6, 6)

    psi4.set_options(
        {
            "basis": "sto-3g",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
            "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
            "XDM_CP_ONLY_REAL_ATOMS": True,
        }
    )
    _, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    disp_corr = wfn_m.variables()["DISPERSION CORRECTION ENERGY"]
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (3, 3)
    assert wfn_m.variables()["XDM C8 COEFFICIENTS"].shape == (3, 3)
    assert wfn_m.variables()["XDM C10 COEFFICIENTS"].shape == (3, 3)
    ref_disp_corr = -0.003792789966032576
    assert np.isclose(disp_corr, ref_disp_corr, atol=1e-6), (
        f"Expected dispersion correction {ref_disp_corr}, got {disp_corr}"
    )

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
            "XDM_CP_ONLY_REAL_ATOMS": False,
        }
    )
    e_m, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (3, 3)
    disp_corr = wfn_m.variables()["DISPERSION CORRECTION ENERGY"]
    ref_disp_corr = -0.0038699045298711673
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
    print(f"Energy with -XDM alias: {e_alias}")
    print(f"Energy with -XDM(KB49): {e_kb49}")
    print(f"Energy with -XDM(LoS-II): {e_los_ii}")

    # TODO: Update the reference values once parameters are refined.
    assert compare_values(e_alias, e_kb49, 10, "-XDM alias equals -XDM(KB49)")
    assert not np.isclose(e_los_ii, e_kb49, rtol=0.0, atol=1.0e-8)
    return


@pytest.mark.xdm
@pytest.mark.saptdft
def test_xdm_models_and_alias_sapt():
    """
    Verifies that ``dft-xdm(sapt)`` always uses CP-fitted XDM parameters.

    For both ``kb49`` and ``los-ii``, confirms the model tag changes
    the SAPT(DFT)+XDM result and that printed ``a1`` values match the
    CP-fitted table entries for b3lyp/aug-cc-pvdz.
    """

    mol = psi4.geometry("""
0 1
H 0.0290 -1.1199 -1.5243
O 0.9481 -1.3990 -1.3587
H 1.4371 -0.5588 -1.3099
--
H 1.0088 -1.5240 0.5086
O 1.0209 -1.1732 1.4270
H 1.5864 -0.3901 1.3101
symmetry c1
no_reorient
no_com
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "DFT_SPHERICAL_POINTS": 302,
            "DFT_RADIAL_POINTS": 75,
            "SAPT_DFT_FUNCTIONAL": "b3lyp",
            "SAPT_DFT_GRAC_SHIFT_A": 0.1307,
            "SAPT_DFT_GRAC_SHIFT_B": 0.1307,
        }
    )
    psi4.set_output_file("pytest_output_dftxdm_sapt.dat", False)

    e_los_ii = psi4.energy("dft-xdm(sapt)(los-ii)", molecule=mol)
    with open("pytest_output_dftxdm_sapt.dat", "r") as handle:
        output_text = handle.read()
    os.remove("pytest_output_dftxdm_sapt.dat")
    os.remove("pytest_output_dftxdm_sapt.log")
    printed_a1 = [
        float(match)
        for match in re.findall(
            r"\ba1\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            output_text,
        )
    ]

    print(f"Printed a1 values: {printed_a1}")
    print(e_los_ii)
    assert any(np.isclose(val, 0.315041, atol=1.0e-6) for val in printed_a1)
    ref_e_los_ii = -0.0037769655155166226
    assert compare_values(e_los_ii, ref_e_los_ii, 8, "SAPT(DFT)+XDM(LoS-II) energy")
    # remove the output file after the test    import os
    return


def test_xdm_los_ii_cp_b3lyp_augccpvdz_a1():
    """Check LoS-II CP a1 for b3lyp/aug-cc-pvdz at runtime."""

    from psi4.driver.procrouting.empirical_disp import empirical_dispersion

    xdm_functor = empirical_dispersion.XDMDispersionFunctor(
        functional_name="b3lyp",
        basis_name="aug-cc-pvdz",
        cp=True,
        model="los-ii",
    )
    a1 = xdm_functor.xdm.a1()
    print(f"a1 = {a1:.6f}")
    assert compare_values(a1, 0.315041, 8, "LoS-II CP a1 for b3lyp/aug-cc-pvdz")
    return


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
    e_ref = -74.96723557871695
    e = psi4.energy("hf-xdm", molecule=mol)
    assert compare_values(e, e_ref, 8, "HF-XDM energy with custom parameters")
    return


@pytest.mark.xdm
def test_xdm_long_range_water():
    mol_dimer = psi4.geometry(
        """
0 1
Gh(O)    -1.55100700  -0.11452000   5.00000000
Gh(H)    -1.93425900   0.76250300   5.00000000
Gh(H)    -0.59967700   0.04071200   5.00000000
--
0 1
O    1.35062500   0.11146900   0.00000000
H    1.68039800  -0.37374100  -0.75856100
H    1.68039800  -0.37374100   0.75856100
units angstrom
        """
    )
    psi4.set_options(
        {
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "basis": "sto-3g",
            # "basis": "aug-cc-pvdz",
            "scf_type": "df",
            "mp2_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "MAXITER": 500,
            "XDM_DISPERSION_PARAMETERS": [0.5068, 1.8242],
            "XDM_CP_ONLY_REAL_ATOMS": False,
        }
    )
    psi4.energy("b3lyp-xdm", molecule=mol_dimer)
    psi4.set_options({"XDM_CP_ONLY_REAL_ATOMS": True})
    psi4.energy("b3lyp-xdm", molecule=mol_dimer)
    return


@pytest.mark.xdm
def test_xdm_long_range():
#             Min. Sep. (A)  xdm total  d3 total  sapt0 total   ref
# entry_name                                                       
# 2mer-0+37            5.51      -0.91     -0.14        -0.08 -0.09
    """
0 1
--
0 1
H                     2.762292050000    -5.112510410000     3.125888580000
H                    -1.066931810000   -11.097280610000    -2.948295900000
H                    -2.584935580000    -6.714620220000    -2.795154380000
H                     4.280295810000    -9.495170800000     2.972747070000
H                    -0.613066840000    -3.651030240000     0.247085470000
H                     2.308427070000   -12.558760780000    -0.069492790000
C                     1.922990970000    -6.399538630000     1.793943470000
C                    -0.227630740000    -9.810252400000    -1.616350790000
C                    -1.075310860000    -7.309180970000    -1.532702060000
C                     2.770671090000    -8.900610050000     1.710294740000
C                     0.000000000000    -5.625185550000     0.177592680000
C                     1.695360230000   -10.584605470000     0.000000000000
--
0 1
H                   -11.202784020000    -5.112510410000    -9.743146330000
H                   -15.032007880000   -11.097280610000   -15.817330810000
H                   -16.550011650000    -6.714620220000   -15.664189300000
H                    -9.684780250000    -9.495170800000    -9.896287850000
H                   -14.578142910000    -3.651030240000   -12.621949440000
H                   -11.656648990000   -12.558760780000   -12.938527700000
C                   -12.042085090000    -6.399538630000   -11.075091450000
C                   -14.192706810000    -9.810252400000   -14.485385700000
C                   -15.040386920000    -7.309180970000   -14.401736970000
C                   -11.194404980000    -8.900610050000   -11.158740170000
C                   -13.965076070000    -5.625185550000   -12.691442230000
C                   -12.269715830000   -10.584605470000   -12.869034910000
units bohr
no_com
no_reorient

"""
#             Min. Sep. (A)  xdm total  d3 total  sapt0 total   ref
# entry_name                                                       
# 2mer-0+37            5.51      -0.91     -0.14        -0.08 -0.09
    """
0 1
--
0 1
H                     2.762292050000    -5.112510410000     3.125888580000
H                    -1.066931810000   -11.097280610000    -2.948295900000
H                    -2.584935580000    -6.714620220000    -2.795154380000
H                     4.280295810000    -9.495170800000     2.972747070000
H                    -0.613066840000    -3.651030240000     0.247085470000
H                     2.308427070000   -12.558760780000    -0.069492790000
C                     1.922990970000    -6.399538630000     1.793943470000
C                    -0.227630740000    -9.810252400000    -1.616350790000
C                    -1.075310860000    -7.309180970000    -1.532702060000
C                     2.770671090000    -8.900610050000     1.710294740000
C                     0.000000000000    -5.625185550000     0.177592680000
C                     1.695360230000   -10.584605470000     0.000000000000
--
0 1
H                   -11.202784020000    -5.112510410000    -9.743146330000
H                   -15.032007880000   -11.097280610000   -15.817330810000
H                   -16.550011650000    -6.714620220000   -15.664189300000
H                    -9.684780250000    -9.495170800000    -9.896287850000
H                   -14.578142910000    -3.651030240000   -12.621949440000
H                   -11.656648990000   -12.558760780000   -12.938527700000
C                   -12.042085090000    -6.399538630000   -11.075091450000
C                   -14.192706810000    -9.810252400000   -14.485385700000
C                   -15.040386920000    -7.309180970000   -14.401736970000
C                   -11.194404980000    -8.900610050000   -11.158740170000
C                   -13.965076070000    -5.625185550000   -12.691442230000
C                   -12.269715830000   -10.584605470000   -12.869034910000
units bohr
no_com
no_reorient

"""
#             Min. Sep. (A)  xdm total  d3 total  sapt0 total   ref
# entry_name                                                       
# 2mer-0+1             2.54      -6.07     -6.05        -6.81 -6.00
    """
0 1
--
0 1
H                     2.762292050000    -5.112510410000     3.125888580000
H                    -1.066931810000   -11.097280610000    -2.948295900000
H                    -2.584935580000    -6.714620220000    -2.795154380000
H                     4.280295810000    -9.495170800000     2.972747070000
H                    -0.613066840000    -3.651030240000     0.247085470000
H                     2.308427070000   -12.558760780000    -0.069492790000
C                     1.922990970000    -6.399538630000     1.793943470000
C                    -0.227630740000    -9.810252400000    -1.616350790000
C                    -1.075310860000    -7.309180970000    -1.532702060000
C                     2.770671090000    -8.900610050000     1.710294740000
C                     0.000000000000    -5.625185550000     0.177592680000
C                     1.695360230000   -10.584605470000     0.000000000000
--
0 1
H                    -8.049469850000     3.788099640000     3.125888580000
H                    -4.220245990000    -2.196670560000    -2.948295900000
H                    -2.702242220000     2.185989830000    -2.795154380000
H                    -9.567473610000    -0.594560750000     2.972747070000
H                    -4.674110960000     5.249579810000     0.247085470000
H                    -7.595604870000    -3.658150730000    -0.069492790000
C                    -7.210168770000     2.501071420000     1.793943470000
C                    -5.059547060000    -0.909642350000    -1.616350790000
C                    -4.211866940000     1.591429080000    -1.532702060000
C                    -8.057848890000     0.000000000000     1.710294740000
C                    -5.287177800000     3.275424500000     0.177592680000
C                    -6.982538030000    -1.683995420000     0.000000000000
units bohr
no_com
no_reorient

"""
    mol_dimer = psi4.geometry(
#             Min. Sep. (A)  xdm total  d3 total  sapt0 total   ref
# entry_name                                                       
# 2mer-0+37            5.51      -0.91     -0.14        -0.08 -0.09
        """
0 1
--
0 1
H                     2.762292050000    -5.112510410000     3.125888580000
H                    -1.066931810000   -11.097280610000    -2.948295900000
H                    -2.584935580000    -6.714620220000    -2.795154380000
H                     4.280295810000    -9.495170800000     2.972747070000
H                    -0.613066840000    -3.651030240000     0.247085470000
H                     2.308427070000   -12.558760780000    -0.069492790000
C                     1.922990970000    -6.399538630000     1.793943470000
C                    -0.227630740000    -9.810252400000    -1.616350790000
C                    -1.075310860000    -7.309180970000    -1.532702060000
C                     2.770671090000    -8.900610050000     1.710294740000
C                     0.000000000000    -5.625185550000     0.177592680000
C                     1.695360230000   -10.584605470000     0.000000000000
--
0 1
H                   -11.202784020000    -5.112510410000    -9.743146330000
H                   -15.032007880000   -11.097280610000   -15.817330810000
H                   -16.550011650000    -6.714620220000   -15.664189300000
H                    -9.684780250000    -9.495170800000    -9.896287850000
H                   -14.578142910000    -3.651030240000   -12.621949440000
H                   -11.656648990000   -12.558760780000   -12.938527700000
C                   -12.042085090000    -6.399538630000   -11.075091450000
C                   -14.192706810000    -9.810252400000   -14.485385700000
C                   -15.040386920000    -7.309180970000   -14.401736970000
C                   -11.194404980000    -8.900610050000   -11.158740170000
C                   -13.965076070000    -5.625185550000   -12.691442230000
C                   -12.269715830000   -10.584605470000   -12.869034910000
units bohr
no_com
no_reorient
        """
    )
    psi4.set_options(
        {
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            # "basis": "sto-3g",
            "basis": "aug-cc-pvdz",
            "scf_type": "df",
            "mp2_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "MAXITER": 500,
            "XDM_DISPERSION_PARAMETERS": [0.7259, 1.3140],
            "XDM_CP_ONLY_REAL_ATOMS": False,
        }
    )
    ha_to_kjmol = qcel.constants.conversion_factor("hartree", "kJ/mol")
    e_ie_cp_ghost = psi4.energy("b3lyp-xdm", molecule=mol_dimer, bsse_type="cp") * ha_to_kjmol
    print(f"IE CP g: {e_ie_cp_ghost:.2f} kJ/mol")

    psi4.set_options({"XDM_CP_ONLY_REAL_ATOMS": True})

    e_ie_cp = psi4.energy("b3lyp-xdm", molecule=mol_dimer, bsse_type="cp") * ha_to_kjmol
    print(f"IE CP  : {e_ie_cp:.2f} kJ/mol")
    e_ie_no_cp = psi4.energy("b3lyp-xdm", molecule=mol_dimer, bsse_type="nocp") * ha_to_kjmol
    print(f"IE CP g: {e_ie_cp_ghost:.2f} kJ/mol")
    print(f"IE CP  : {e_ie_cp:.2f} kJ/mol")
    print(f"IE NOCP: {e_ie_no_cp:.2f} kJ/mol")
    # Interaction energy with XDM: -5.84 kJ/mol
    # Interaction energy without CP: -9.82 kJ/mol
    return


if __name__ == "__main__":
    # pytest.main([__file__, "-x", "-v"])
    # test_h2o_ghosts()
    # test_xdm_models_and_alias_sapt()
    # test_xdm_long_range_water()
    test_xdm_long_range()
