import pytest
import psi4
from qcelemental import constants
from psi4 import compare_values
from psi4 import core
import numpy as np
from pprint import pprint as pp
# from addons import uusing

hartree_to_kcalmol = constants.conversion_factor("hartree", "kcal/mol")
pytestmark = [pytest.mark.psi, pytest.mark.api]

_sapt_testing_mols = {
    "neutral_water_dimer": """
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
units angstrom
""",
    "hydroxide": """
-1 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
units angstrom
""",
}


@pytest.mark.skip(reason="Not completed fsapt einsums")
@pytest.mark.fsapt
@pytest.mark.saptdft
def test_fsaptdft():
    """
    built from sapt-dft1 ctest
    """
    Eref_nh = {
        # mEh
        "SAPT ELST ENERGY": -0.0033529619489769402,
        "SAPT EXCH ENERGY": 1.2025482154546578e-05,
        "SAPT IND ENERGY": -1.2227400973891604e-05,
        "SAPT DISP ENERGY": -0.005176878264916587,
        "CURRENT ENERGY": -0.008530042132712874,
    }  # TEST
    mol = psi4.geometry("""
0 1
C 0.00000000 0.00000000 0.00000000
H 1.09000000 0.00000000 0.00000000
H -0.36333333 0.83908239 0.59332085
H -0.36333333 0.09428973 -1.02332709
H -0.36333333 -0.93337212 0.43000624
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
no_com
""")
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "sapt_dft_grac_shift_a": 0.203293,
            "sapt_dft_grac_shift_b": 0.203293,
            "SAPT_DFT_DO_DHF": False,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_EXCH_DISP_SCALE_SCHEME": "None",
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )
    psi4.energy("fisapt0", molecule=mol)
    print("\n fisapt0 complete")
    psi4.energy("sapt(dft)", molecule=mol)
    for k, v in Eref_nh.items():  # TEST
        ref = v
        assert compare_values(
            ref, psi4.variable(k) * 1000, 8, "!hyb, xd=none, !dHF: " + k
        )


@pytest.mark.skip(reason="Not completed fsapt einsums")
@pytest.mark.fsapt
@pytest.mark.saptdft
@pytest.mark.saptdft
def test_fsaptdft_fsapt0():
    """
    built from sapt-dft1 ctest
    """
    Eref_nh = {
        # mEh
        "SAPT ELST ENERGY": -0.00233320,
        "SAPT EXCH ENERGY": 0.00001443,
        "SAPT IND ENERGY": -0.00001103,
        "SAPT DISP ENERGY": -0.00563062,
    }  # TEST
    mol = psi4.geometry("""
0 1
C 0.00000000 0.00000000 0.00000000
H 1.09000000 0.00000000 0.00000000
H -0.36333333 0.83908239 0.59332085
H -0.36333333 0.09428973 -1.02332709
H -0.36333333 -0.93337212 0.43000624
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
no_com
""")
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
        }
    )
    np.set_printoptions(precision=10, suppress=True)
    psi4.energy("sapt(dft)", molecule=mol)
    print("\n sapt(dft) complete")
    psi4.energy("fisapt0", molecule=mol)
    for k, v in Eref_nh.items():  # TEST
        ref = v
        assert compare_values(
            ref, psi4.variable(k) * 1000, 8, "!hyb, xd=none, !dHF: " + k
        )


@pytest.mark.skip(reason="Not completed fsapt einsums")
@pytest.mark.fsapt
@pytest.mark.saptdft
def test_fsapt0_fsaptdft():
    """
    built from sapt-dft1 ctest
    """
    Eref_nh = {
        # mEh
        "SAPT ELST ENERGY": -0.00233320,
        "SAPT EXCH ENERGY": 0.00001443,
        "SAPT IND ENERGY": -0.00001103,
        "SAPT DISP ENERGY": -0.00563062,
    }  # TEST
    mol = psi4.geometry("""
0 1
C 0.00000000 0.00000000 0.00000000
H 1.09000000 0.00000000 0.00000000
H -0.36333333 0.83908239 0.59332085
H -0.36333333 0.09428973 -1.02332709
H -0.36333333 -0.93337212 0.43000624
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
no_com
""")
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
        }
    )
    np.set_printoptions(precision=10, suppress=True)
    psi4.energy("fisapt0", molecule=mol)
    print("\n fisapt0 complete")
    psi4.energy("sapt(dft)", molecule=mol)
    for k, v in Eref_nh.items():  # TEST
        ref = v
        assert compare_values(
            ref, psi4.variable(k) * 1000, 8, "!hyb, xd=none, !dHF: " + k
        )


def test_fsaptdft_simple():
    """
    built from sapt-dft1 ctest
    """
    Eref_nh = {
        # mEh
        "SAPT ELST ENERGY": -0.00233320,
        "SAPT EXCH ENERGY": 0.00001443,
        "SAPT IND ENERGY": -0.00001103,
        "SAPT DISP ENERGY": -0.00563062,
    }  # TEST
    mol = psi4.geometry("""
0 1
C 0.00000000 0.00000000 0.00000000
H 1.09000000 0.00000000 0.00000000
H -0.36333333 0.83908239 0.59332085
H -0.36333333 0.09428973 -1.02332709
H -0.36333333 -0.93337212 0.43000624
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
no_com
""")
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "FISAPT",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )
    np.set_printoptions(precision=10, suppress=True)
    psi4.energy("sapt(dft)", molecule=mol)
    print('timer dict')
    pp(psi4.core.get_timer_dict())
    psi4.write_timer_csv('timer.csv')
    for k, v in Eref_nh.items():  # TEST
        ref = v
        assert compare_values(
            ref, psi4.variable(k) * 1000, 8, "!hyb, xd=none, !dHF: " + k
        )


@pytest.mark.skip(reason="Not completed fsapt einsums")
@pytest.mark.fsapt
@pytest.mark.saptdft
def test_fsaptdft_fsapt0_simple():
    """
    built from sapt-dft1 ctest
    """
    Eref_nh = {
        # mEh
        "SAPT ELST ENERGY": -0.00782717,
        "SAPT EXCH ENERGY": 0.05953516,
        "SAPT IND ENERGY": -0.00054743,
        "SAPT DISP ENERGY": -0.00012075,
    }  # TEST
    mol = psi4.geometry("""
0 1
He 3.00000000 0.00000000 0.00000000
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
no_com
""")
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
        }
    )
    np.set_printoptions(precision=10, suppress=True)
    psi4.energy("fisapt0", molecule=mol)
    print("\n fisapt0 complete")
    psi4.energy("sapt(dft)", molecule=mol)
    for k, v in Eref_nh.items():  # TEST
        ref = v
        assert compare_values(
            ref, psi4.variable(k) * 1000, 6, "!hyb, xd=none, !dHF: " + k
        )


@pytest.mark.saptdft
@pytest.mark.fsapt
# @uusing("pandas")
@pytest.mark.saptdft
def test_fsapthf_psivars():
    """
    fsapt-psivars: calling fsapt_analysis with psi4 variables after running an
    fisapt0 calcluation requires the user to pass the molecule object
    """
    import pandas as pd

    mol = psi4.geometry(
        """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
"""
    )
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            # "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
    from pprint import pprint as pp

    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.0007912165332922398,
        "Eelst": -0.0019765266134612602,
        "Eexch": 0.006335438658900877,
        "Eind": -0.0004635353246623952,
        "Enuc": 474.74808217020274,
        "Etot": 0.003104160187484982,
    }
    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    pp(Epsi)
    for key in keys:
        compare_values(Eref[key], Epsi[key], 5, key)
    data = psi4.fsapt_analysis(
        molecule=mol,
        fragments_a={
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        fragments_b={
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
    )
    df = pd.DataFrame(data)
    print("COMPUTED DF")
    print(df)
    # data_tmp = {k: v.tolist() for k, v in dict(df).items()}
    # pp(data_tmp)
    data = {
        "Disp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "EDisp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Elst": [
            -0.100513,
            -1.139774,
            -0.100513,
            -1.139774,
            0.511801,
            -1.752088,
            -0.100513,
            -1.139774,
            -1.240287,
        ],
        "Exch": [
            0.031751,
            3.943794,
            0.031751,
            3.943794,
            0.047332,
            3.928213,
            0.031751,
            3.943794,
            3.975545,
        ],
        "Frag1": [
            "Methyl1_A",
            "Methyl1_A",
            "Methyl2_A",
            "Methyl2_A",
            "Methyl1_A",
            "Methyl2_A",
            "All",
            "All",
            "All",
        ],
        "Frag2": [
            "Peptide_B",
            "T-Butyl_B",
            "Peptide_B",
            "T-Butyl_B",
            "All",
            "All",
            "Peptide_B",
            "T-Butyl_B",
            "All",
        ],
        "IndAB": [
            -0.033169,
            -0.190334,
            -0.033169,
            -0.190334,
            -0.022726,
            -0.200776,
            -0.033169,
            -0.190334,
            -0.223502,
        ],
        "IndBA": [
            -0.001398,
            -0.065972,
            -0.001398,
            -0.065972,
            0.015095,
            -0.082466,
            -0.001398,
            -0.065972,
            -0.067370,
        ],
        "Total": [
            -0.103328,
            2.547714,
            -0.103328,
            2.547714,
            0.551502,
            1.892883,
            -0.103328,
            2.547714,
            2.444386,
        ],
    }

    ref_df = pd.DataFrame(data)
    print("REF")
    print(ref_df)
    # difference df
    df_diff = ref_df.copy()
    df_diff.iloc[:, 2:] = ref_df.iloc[:, 2:] - df.iloc[:, 2:]
    print("DIFF")
    print(df_diff)
    print(df_diff[["Frag1", "Frag2", "IndAB"]])

    for col in ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]:
        for i in range(len(ref_df)):
            compare_values(
                ref_df[col].iloc[i],
                df[col].iloc[i],
                4,
                f"{ref_df['Frag1'].iloc[i]} {ref_df['Frag2'].iloc[i]} {col}",
            )


@pytest.mark.saptdft
@pytest.mark.fsapt
# @uusing("pandas")
@pytest.mark.saptdft
def test_fsaptdftd4_psivars():
    """
    fsapt-psivars: calling fsapt_analysis with psi4 variables after running an
    fisapt0 calcluation requires the user to pass the molecule object
    """
    import pandas as pd

    mol = psi4.geometry(
        """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
"""
    )
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": False,
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
    from pprint import pprint as pp

    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.002273897728190271,
        "Eelst": -0.0019765266134612602,
        "Eexch": 0.006335438658900877,
        "Eind": -0.0004635353246623952,
        "Enuc": 474.74808217020274,
        "Etot": 0.0016214790566494836,
    }
    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    pp(Epsi)
    pp(core.variables())
    for key in keys:
        compare_values(Eref[key], Epsi[key], 5, key)
    data = psi4.fsapt_analysis(
        molecule=mol,
        fragments_a={
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        fragments_b={
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
    )
    df = pd.DataFrame(data)
    print("COMPUTED DF")
    print(df)
    # data_tmp = {k: v.tolist() for k, v in dict(df).items()}
    # pp(data_tmp)
    data = {
        "Disp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "EDisp": [
            -0.030085905907278467,
            -1.3968064609333375,
            -0.030085905907278467,
            -1.3968064609333375,
            -0.13680656679856162,
            -1.2900858000420543,
            -0.030085905907278467,
            -1.3968064609333375,
            -1.426892366840616,
        ],
        "Elst": [
            -0.10051296364207474,
            -1.1397737426478898,
            -0.10051296364207474,
            -1.1397737426478898,
            0.511800833127964,
            -1.7520875394179285,
            -0.10051296364207474,
            -1.1397737426478898,
            -1.2402867062899645,
        ],
        "Exch": [
            0.03175138204831181,
            3.943793834216297,
            0.03175138204831181,
            3.943793834216297,
            0.04733227649800176,
            3.928212939766607,
            0.03175138204831181,
            3.943793834216297,
            3.9755452162646088,
        ],
        "Frag1": [
            "Methyl1_A",
            "Methyl1_A",
            "Methyl2_A",
            "Methyl2_A",
            "Methyl1_A",
            "Methyl2_A",
            "All",
            "All",
            "All",
        ],
        "Frag1_indices": [
            [1, 2, 7, 8],
            [1, 2, 7, 8],
            [3, 4, 5, 6],
            [3, 4, 5, 6],
            [1, 2, 7, 8],
            [3, 4, 5, 6],
            [1, 2, 7, 8, 3, 4, 5, 6],
            [1, 2, 7, 8, 3, 4, 5, 6],
            [1, 2, 7, 8, 3, 4, 5, 6],
        ],
        "Frag2": [
            "Peptide_B",
            "T-Butyl_B",
            "Peptide_B",
            "T-Butyl_B",
            "All",
            "All",
            "Peptide_B",
            "T-Butyl_B",
            "All",
        ],
        "Frag2_indices": [
            [9, 10, 11, 16, 26],
            [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26],
            [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26],
            [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        ],
        "IndAB": [
            -0.033168554918636954,
            -0.19033373433064685,
            -0.033168554918636954,
            -0.19033373433064685,
            -0.02272593046485042,
            -0.20077635878443337,
            -0.033168554918636954,
            -0.19033373433064685,
            -0.22350228924928378,
        ],
        "IndBA": [
            -0.0013980980367014283,
            -0.06597227880242341,
            -0.0013980980367014283,
            -0.06597227880242341,
            0.015095290099091916,
            -0.08246566693821676,
            -0.0013980980367014283,
            -0.06597227880242341,
            -0.06737037683912485,
        ],
        "Total": [
            -0.1334141404565592,
            1.1509076175030337,
            -0.1334141404565592,
            1.1509076175030337,
            0.4146959024639971,
            0.6027975745824774,
            -0.1334141404565592,
            1.1509076175030337,
            1.0174934770464745,
        ],
    }

    ref_df = pd.DataFrame(data)
    print("REF")
    print(ref_df)

    for col in ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]:
        for i in range(len(ref_df)):
            compare_values(
                ref_df[col].iloc[i],
                df[col].iloc[i],
                4,
                f"{ref_df['Frag1'].iloc[i]} {ref_df['Frag2'].iloc[i]} {col}",
            )


@pytest.mark.saptdft
@pytest.mark.fsapt
# @uusing("pandas")
@pytest.mark.saptdft
def test_fsaptdft_disp0_fisapt0_psivars():
    """
    fsapt-psivars: calling fsapt_analysis with psi4 variables after running an
    fisapt0 calcluation requires the user to pass the molecule object
    """
    import pandas as pd

    mol = psi4.geometry(
        """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
"""
    )
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_D4_IE": False,
            "SAPT_DFT_DO_DISP": True,
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.0007912165332931369,
        "Eelst": -0.0019765265492708295,
        "Eexch": 0.006335438658802855,
        "Eind": -0.0004635353239533062,
        "Enuc": 474.74808217020274,
        "Etot": 0.003104160252285582,
    }
    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    pp(Epsi)
    for key in keys:
        compare_values(Eref[key], Epsi[key], 5, key)
    data = psi4.fsapt_analysis(
        molecule=mol,
        fragments_a={
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        fragments_b={
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
    )
    df = pd.DataFrame(data)
    print("COMPUTED DF")
    print(df)
    # data_tmp = {k: v.tolist() for k, v in dict(df).items()}
    # pp(data_tmp)
    data = {
        "Disp": [
            -0.017541570508833145,
            -0.4789542999420328,
            -0.017541570508833145,
            -0.4789542999420328,
            -0.071407932691549,
            -0.42508793775931697,
            -0.017541570508833145,
            -0.4789542999420328,
            -0.49649587045086596,
        ],
        "EDisp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Elst": [
            -0.10051391028423495,
            -1.1397752245096058,
            -0.10051391028423495,
            -1.1397752245096058,
            0.5118009604139928,
            -1.7520900952078335,
            -0.10051391028423495,
            -1.1397752245096058,
            -1.2402891347938407,
        ],
        "Exch": [
            0.03175173588433509,
            3.9437960430509107,
            0.03175173588433509,
            3.9437960430509107,
            0.047332471641690285,
            3.9282153072935553,
            0.03175173588433509,
            3.9437960430509107,
            3.9755477789352454,
        ],
        "Frag1": [
            "Methyl1_A",
            "Methyl1_A",
            "Methyl2_A",
            "Methyl2_A",
            "Methyl1_A",
            "Methyl2_A",
            "All",
            "All",
            "All",
        ],
        "Frag1_indices": [
            [1, 2, 7, 8],
            [1, 2, 7, 8],
            [3, 4, 5, 6],
            [3, 4, 5, 6],
            [1, 2, 7, 8],
            [3, 4, 5, 6],
            [1, 2, 7, 8, 3, 4, 5, 6],
            [1, 2, 7, 8, 3, 4, 5, 6],
            [1, 2, 7, 8, 3, 4, 5, 6],
        ],
        "Frag2": [
            "Peptide_B",
            "T-Butyl_B",
            "Peptide_B",
            "T-Butyl_B",
            "All",
            "All",
            "Peptide_B",
            "T-Butyl_B",
            "All",
        ],
        "Frag2_indices": [
            [9, 10, 11, 16, 26],
            [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26],
            [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26],
            [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 16, 26, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        ],
        "IndAB": [
            -0.03316857695058072,
            -0.19033368814221727,
            -0.03316857695058072,
            -0.19033368814221727,
            -0.022725900092073127,
            -0.20077636500072488,
            -0.03316857695058072,
            -0.19033368814221727,
            -0.223502265092798,
        ],
        "IndBA": [
            -0.0013981022409371652,
            -0.06597243987421669,
            -0.0013981022409371652,
            -0.06597243987421669,
            0.01509541271545878,
            -0.08246595483061264,
            -0.0013981022409371652,
            -0.06597243987421669,
            -0.06737054211515386,
        ],
        "Total": [
            -0.12087042409913806,
            2.0687603905846856,
            -0.12087042409913806,
            2.0687603905846856,
            0.48009501199103966,
            1.467794954494508,
            -0.12087042409913806,
            2.0687603905846856,
            1.9478899664855476,
        ],
    }

    ref_df = pd.DataFrame(data)
    print("REF")
    print(ref_df)

    for col in ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]:
        for i in range(len(ref_df)):
            compare_values(
                ref_df[col].iloc[i],
                df[col].iloc[i],
                4,
                f"{ref_df['Frag1'].iloc[i]} {ref_df['Frag2'].iloc[i]} {col}",
            )


@pytest.mark.saptdft
@pytest.mark.fsapt
# @uusing("pandas")
@pytest.mark.saptdft
def test_fsaptdftd4_psivars_pbe0():
    import pandas as pd

    mol = psi4.geometry(
        """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
"""
    )
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_FUNCTIONAL": "PBE0",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": False,
            "SAPT_DFT_GRAC_SHIFT_A": 0.11652342,
            "SAPT_DFT_GRAC_SHIFT_B": 0.12724880,
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
    from pprint import pprint as pp

    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.0027693003947224628,
        "Eelst": -0.002059138272954897,
        "Eexch": 0.0065851135315064075,
        "Eind": -0.0004940302703933357,
        "Enuc": 474.74808217020274,
        "Etot": 0.0012626445934357123,
    }

    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    pp(Epsi)
    pp(core.variables())
    for key in keys:
        compare_values(Eref[key], Epsi[key], 5, key)
    data = psi4.fsapt_analysis(
        molecule=mol,
        fragments_a={
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        fragments_b={
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
    )
    df = pd.DataFrame(data)
    print("COMPUTED DF")
    print(
        df[
            [
                "Frag1",
                "Frag2",
                "Elst",
                "Exch",
                "IndAB",
                "IndBA",
                "Disp",
                "EDisp",
                "Total",
            ]
        ]
    )
    data = {
        "Frag1": [
            "Methyl1_A",
            "Methyl1_A",
            "Methyl2_A",
            "Methyl2_A",
            "Methyl1_A",
            "Methyl2_A",
            "All",
            "All",
            "All",
        ],
        "Frag2": [
            "Peptide_B",
            "T-Butyl_B",
            "Peptide_B",
            "T-Butyl_B",
            "All",
            "All",
            "Peptide_B",
            "T-Butyl_B",
            "All",
        ],
        "Elst": [
            -0.106663,
            -1.185465,
            -0.106663,
            -1.185465,
            0.554554,
            -1.846683,
            -0.106663,
            -1.185465,
            -1.292129,
        ],
        "Exch": [
            0.039172,
            4.093049,
            0.039172,
            4.093049,
            0.047454,
            4.084767,
            0.039172,
            4.093049,
            4.132221,
        ],
        "IndAB": [
            -0.038179,
            -0.193800,
            -0.038179,
            -0.193800,
            -0.023789,
            -0.208190,
            -0.038179,
            -0.193800,
            -0.231979,
        ],
        "IndBA": [
            -0.001862,
            -0.076168,
            -0.001862,
            -0.076168,
            0.020246,
            -0.098276,
            -0.001862,
            -0.076168,
            -0.078030,
        ],
        "Disp": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "EDisp": [
            -0.031540,
            -1.706223,
            -0.031540,
            -1.706223,
            -0.147529,
            -1.590233,
            -0.031540,
            -1.706223,
            -1.737762,
        ],
        "Total": [
            -0.139072,
            0.931393,
            -0.139072,
            0.931393,
            0.450936,
            0.341385,
            -0.139072,
            0.931393,
            0.792322,
        ],
    }

    ref_df = pd.DataFrame(data)
    print("REF")
    print(ref_df)
    # difference df
    df_diff = ref_df.copy()
    df_diff.iloc[:, 2:] = ref_df.iloc[:, 2:] - df.iloc[:, 2:]
    print("DIFF")
    print(df_diff)
    print(df_diff[["Frag1", "Frag2", "Disp", "EDisp"]])

    for col in ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]:
        for i in range(len(ref_df)):
            compare_values(
                ref_df[col].iloc[i],
                df[col].iloc[i],
                4,
                f"{ref_df['Frag1'].iloc[i]} {ref_df['Frag2'].iloc[i]} {col}",
            )


@pytest.mark.saptdft
@pytest.mark.fsapt
# @uusing("pandas")
@pytest.mark.saptdft
def test_fsaptdftd4_psivars_pbe0_frozen_core():
    import pandas as pd

    mol = psi4.geometry(
        """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
"""
    )
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_FUNCTIONAL": "PBE0",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": False,
            "SAPT_DFT_GRAC_SHIFT_A": 0.11652342,
            "SAPT_DFT_GRAC_SHIFT_B": 0.12724880,
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
    from pprint import pprint as pp

    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.0027693003947224628,
        "Eelst": -0.002059138272954897,
        "Eexch": 0.0065851135315064075,
        "Eind": -0.0004940302703933357,
        "Enuc": 474.74808217020274,
        "Etot": 0.0012626445934357123,
    }

    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    pp(Epsi)
    pp(core.variables())
    for key in keys:
        compare_values(Eref[key], Epsi[key], 5, key)
    data = psi4.fsapt_analysis(
        molecule=mol,
        fragments_a={
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        fragments_b={
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
    )
    df = pd.DataFrame(data)
    print("COMPUTED DF")
    print(df)
    data = {
        "Frag1": ["Methyl1_A", "Methyl2_A", "All", "All", "All"],
        "Frag2": ["All", "All", "Peptide_B", "T-Butyl_B", "All"],
        "Elst": [0.554554, -1.846683, -0.106663, -1.185465, -1.292129],
        "Exch": [0.047454, 4.084767, 0.039172, 4.093049, 4.132221],
        "IndAB": [-0.023789, -0.208190, -0.038179, -0.193800, -0.231979],
        "IndBA": [0.020246, -0.098276, -0.001862, -0.076168, -0.078030],
        "Disp": [0, 0, 0, 0, 0],
        "EDisp": [-0.147529, -1.590233, -0.031540, -1.706223, -1.737762],
        "Total": [0.450936, 0.341385, -0.139072, 0.931393, 0.792321],
    }

    ref_df = pd.DataFrame(data)
    print("REF")
    print(ref_df)
    # difference df
    df_diff = ref_df.copy()
    df_diff.iloc[:, 2:] = ref_df.iloc[:, 2:] - df.iloc[:, 2:]
    print("DIFF")
    print(df_diff)
    print(df_diff[["Frag1", "Frag2", "Disp", "EDisp"]])

    for col in ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]:
        for i in range(len(ref_df)):
            compare_values(
                ref_df[col].iloc[i],
                df[col].iloc[i],
                4,
                f"{ref_df['Frag1'].iloc[i]} {ref_df['Frag2'].iloc[i]} {col}",
            )


@pytest.mark.fsapt
@pytest.mark.saptdft
def test_fsaptdft_indices():
    # TODO: EDIT THIS TEST TO DROP SAVING DF
    import pandas as pd

    mol = psi4.geometry(
        """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
"""
    )
    print("FSAPT(PBE0)-D4(I)")
    functional = "HF"
    # functional = "PBE0"
    psi4.set_options(
        {
            # "basis": "aug-cc-pVDZ",
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            # "freeze_core": "false", # Frozen core not working with localization presently
            "freeze_core": "true",  # Frozen core not working with localization presently
            "FISAPT_FSAPT_FILEPATH": "none",
            # "SAPT_DFT_FUNCTIONAL": "PBE0",
            "SAPT_DFT_FUNCTIONAL": functional,
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": False,
            # "SAPT_DFT_D4_TYPE": "SUPERMOLECULAR",
            "SAPT_DFT_D4_TYPE": "INTERMOLECULAR",
            # "SAPT_DFT_GRAC_BASIS": "aug-cc-pVTZ",
            # "SAPT_DFT_GRAC_COMPUTE": "SINGLE",
            # If known...
            "SAPT_DFT_GRAC_SHIFT_A": 0.09605298,
            "SAPT_DFT_GRAC_SHIFT_B": 0.073504,
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
    data = psi4.fsapt_analysis(
        # NOTE: 1-indexed for fragments_a and fragments_b
        molecule=mol,
        fragments_a={
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": range(3, 7),
        },
        fragments_b={
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
    )
    df = pd.DataFrame(data)
    print(df)
    """
       Frag1      Frag2             Frag1_indices  ... Disp     EDisp     Total
0  Methyl1_A        All              [1, 2, 7, 8]  ...  0.0 -0.144284  0.407219
1  Methyl2_A        All              [3, 4, 5, 6]  ...  0.0 -1.358610  0.534274
2        All  Peptide_B  [1, 2, 7, 8, 3, 4, 5, 6]  ...  0.0 -0.031207 -0.134535
3        All  T-Butyl_B  [1, 2, 7, 8, 3, 4, 5, 6]  ...  0.0 -1.471686  1.076028
4        All        All  [1, 2, 7, 8, 3, 4, 5, 6]  ...  0.0 -1.502893  0.941492
    """
    mol_qcel_dict = mol.to_schema(dtype=2)
    frag1_indices = df["Frag1_indices"].tolist()
    frag2_indices = df["Frag2_indices"].tolist()
    # Using molecule object for all test to ensure right counts from each
    # fragment are achieved. Note +1 for 1-indexing in fsapt_analysis
    all_A = [i + 1 for i in mol_qcel_dict["fragments"][0]]
    expected_frag1_indices = [
        [1, 2, 7, 8],
        [1, 2, 7, 8],
        [3, 4, 5, 6],
        [3, 4, 5, 6],
        [1, 2, 7, 8],
        [3, 4, 5, 6],
        all_A,
        all_A,
        all_A,
    ]
    all_B = [j + 1 for j in mol_qcel_dict["fragments"][1]]
    expected_frag2_indices = [
        [9, 10, 11, 16, 26],
        [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        [9, 10, 11, 16, 26],
        [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        all_B,
        all_B,
        [9, 10, 11, 16, 26],
        [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        all_B,
    ]
    print(f"{all_A=}")
    print(f"{all_B=}")
    for i, indices in enumerate(frag1_indices):
        # Assert lists are identical
        e = expected_frag1_indices[i]
        sorted_frag = sorted(indices)
        assert sorted_frag == e, f"Frag1 indices do not match for fragment {
            i
        }: expected {e}, got {sorted_frag}"

    for i, indices in enumerate(frag2_indices):
        e = expected_frag2_indices[i]
        sorted_frag = sorted(indices)
        assert sorted_frag == e, f"Frag2 indices do not match for fragment {
            i
        }: expected {e}, got {sorted_frag}"
    df["F-Induction"] = df["IndAB"] + df["IndBA"]
    df.drop(columns=["IndAB", "IndBA"], inplace=True)
    df = df.rename(
        columns={
            "Elst": "F-Electrostatics",
            "Exch": "F-Exchange",
            "Disp": "F-Dispersion",
            "EDisp": "F-EDispersion",
            "Total": "F-Total",
        },
    )
    import qcelemental as qcel

    qcel_mol = qcel.models.Molecule.from_data(
        """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
"""
    )
    df["qcel_molecule"] = [qcel_mol] * len(df)
    # Save dataframe for future testing
    df.to_pickle(f"fsapt_{functional}_train_simple.pkl")


if __name__ == "__main__":
    psi4.set_memory("220 GB")
    psi4.set_num_threads(24)
    test_fsaptdft_simple()
    # test_fsaptdft_disp0_fisapt0_psivars()


    # test_fsaptdft()
    # test_fsaptdft_fsapt0_simple()
    # test_fsaptdftd4_psivars()
    # test_fsaptdft_fsapt0()
    # test_fsapt0_fsaptdft()
    # test_fsaptdft_psivars()
    # test_fsapthf_psivars()
    # test_fsaptdftd4_psivars()
    # test_fsaptdftd4_psivars_pbe0()
    # test_fsaptdft_indices()
