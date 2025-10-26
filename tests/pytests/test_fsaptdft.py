import pytest
import psi4
from qcelemental import constants
from psi4 import compare_values
from psi4 import core
import numpy as np
import qcelemental as qcel
from pprint import pprint as pp

# from addons import uusing
import pandas as pd

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
            "SAPT_DFT_DO_FSAPT": True,
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
            "SAPT_DFT_DO_FSAPT": True,
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
            "SAPT_DFT_DO_FSAPT": True,
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


@pytest.mark.skip(reason="Not completed fsapt einsums")
def test_fsaptdft_fsapt0_simple():
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
He 0.00000000 0.00000000 0.00000000
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
            "SAPT_DFT_DO_FSAPT": True,
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


@pytest.mark.saptdft
# @uusing("pandas")
def test_fsaptdft_psivars():
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
            "SAPT_DFT_DO_FSAPT": True,
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
    data = {
        "Frag1": ["Methyl1_A", "Methyl2_A", "All", "All", "All"],
        "Frag2": ["All", "All", "Peptide_B", "T-Butyl_B", "All"],
        "Elst": [0.511801, -1.752090, -0.100514, -1.139775, -1.240289],
        "Exch": [0.047332, 3.928215, 0.031752, 3.943796, 3.975548],
        "IndAB": [-0.022726, -0.200776, -0.033169, -0.190334, -0.223502],
        "IndBA": [0.015095, -0.082466, -0.001398, -0.065972, -0.067371],
        "Disp": [-0.071408, -0.425088, -0.017542, -0.478954, -0.496496],
        "EDisp": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Total": [0.480095, 1.467795, -0.120870, 2.068760, 1.947890],
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
# @uusing("pandas")
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
            "SAPT_DFT_DO_FSAPT": True,
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": False,
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
    from pprint import pprint as pp

    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.0027693003947224628,
        "Eelst": -0.0019765266134612602,
        "Eexch": 0.006335438658900877,
        "Eind": -0.0004635353246623952,
        "Enuc": 474.74808217020274,
        "Etot": 0.0011260761229532233,
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
        "Elst": [0.511801, -1.752090, -0.100514, -1.139775, -1.240289],
        "Exch": [0.047332, 3.928215, 0.031752, 3.943796, 3.975548],
        "IndAB": [-0.022726, -0.200776, -0.033169, -0.190334, -0.223502],
        "IndBA": [0.015095, -0.082466, -0.001398, -0.065972, -0.067371],
        "Disp": [0, 0, 0, 0, 0],
        "EDisp": [-0.147529, -1.590233, -0.031540, -1.706223, -1.737762],
        "Total": [0.403974, 0.302650, -0.134868, 0.841491, 0.706623],
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


if __name__ == "__main__":
    psi4.set_memory("64 GB")
    psi4.set_num_threads(12)

    # test_fsaptdft_fsapt0()
    # test_fsapt0_fsaptdft()
    # test_fsaptdft_psivars()
    test_fsaptdftd4_psivars()
