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


def test_fsaptdft_timer():
    """
    built from sapt-dft1 ctest
    """
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
    np.set_printoptions(precision=10, suppress=True)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "FISAPT_FSAPT_FILEPATH": "none",
            "FISAPT_DO_FSAPT": True,
            "FISAPT_DO_FSAPT_DISP": True,
        }
    )
    psi4.core.clean_timers()
    psi4.energy("fisapt0", molecule=mol)
    compute_time_fisapt0 = psi4.core.get_timer_dict()["FISAPT"]
    psi4.driver.p4util.write_timer_csv("fisapt0_timers.csv")
    psi4.core.clean()
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
            # OPTION distringuishing einsum vs fi
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_USE_EINSUMS": True,
        }
    )
    psi4.core.clean_timers()
    psi4.energy("sapt(dft)", molecule=mol)
    compute_time_fisapt = psi4.core.get_timer_dict()["SAPT(DFT) Energy"]
    psi4.driver.p4util.write_timer_csv("saptdft_useEin_timers.csv")
    psi4.core.clean()
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "FISAPT_FSAPT_FILEPATH": "none",
            # REALLY matters if SAPT_DFT_DO_FSAPT==FISAPT due to reuse of integrals
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
            # OPTION distringuishing einsum vs fi
            "SAPT_DFT_DO_FSAPT": "FISAPT",
            "SAPT_DFT_USE_EINSUMS": False,
        }
    )
    psi4.core.clean_timers()
    psi4.energy("sapt(dft)", molecule=mol)
    compute_time_saptdft = psi4.core.get_timer_dict()["SAPT(DFT) Energy"]
    psi4.driver.p4util.write_timer_csv("saptdft_fi_timers.csv")
    psi4.core.clean()
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "FISAPT_FSAPT_FILEPATH": "none",
            # REALLY matters if SAPT_DFT_DO_FSAPT==FISAPT due to reuse of integrals
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
            # OPTION distringuishing einsum vs fi
            "SAPT_DFT_DO_FSAPT": "FISAPT",
            "SAPT_DFT_USE_EINSUMS": True,
        }
    )
    psi4.core.clean_timers()
    psi4.energy("sapt(dft)", molecule=mol)
    compute_time_saptdft_fi_ein = psi4.core.get_timer_dict()["SAPT(DFT) Energy"]
    psi4.driver.p4util.write_timer_csv("saptdft_fi_useEin_timers.csv")
    print(
        f"compute time ein+fi: {compute_time_fisapt['wall_time']:.2f}s\n"
        f"compute_time_einsum: {compute_time_saptdft['wall_time']:.2f}s\n"
        f"compute_time_fi_ein: {compute_time_saptdft_fi_ein['wall_time']:.2f}s\n"
        f"time fisapt0: {compute_time_fisapt0['wall_time']:.2f}s\n"
    )
    return


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
        "SAPT DISP ENERGY": -0.0056304531,  # -0.00563062,
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
    print(
        df[
            [
                "Frag1",
                "Frag2",
                "ClosestContact",
                "Elst",
                "IndAB",
                "IndBA",
                "Disp",
                "EDisp",
                "Total",
            ]
        ]
    )
    data_tmp = {k: v.tolist() for k, v in dict(df).items()}
    pp(data_tmp)
    data = {
        "Disp": [
            -0.00399436152159229,
            -0.06741037189411032,
            -0.013546524596044364,
            -0.41148730370035314,
            -0.07140473341570261,
            -0.4250338282963975,
            -0.017540886117636656,
            -0.47889767559446345,
            -0.4964385617121001,
        ],
        "EDisp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Elst": [
            0.7173658712642776,
            -0.20556512118969295,
            -0.8178788514412361,
            -0.9342087754091182,
            0.5118007500745847,
            -1.7520876268503542,
            -0.10051298017695842,
            -1.1397738965988111,
            -1.2402868767757695,
        ],
        "Exch": [
            0.00013545439373786606,
            0.0471968221083766,
            0.03161592765409533,
            3.896597012155338,
            0.04733227650211447,
            3.928212939809433,
            0.0317513820478332,
            3.9437938342637144,
            3.9755452163115477,
        ],
        "IndAB": [
            -0.007097098270914404,
            -0.015628832205804594,
            -0.026071456661474337,
            -0.1747049022073196,
            -0.022725930476719,
            -0.20077635886879394,
            -0.03316855493238874,
            -0.1903337344131242,
            -0.22350228934551294,
        ],
        "IndBA": [
            0.0003539943194493893,
            0.014741295750793878,
            -0.0017520923551549333,
            -0.08071357456909534,
            0.015095290070243267,
            -0.08246566692425027,
            -0.001398098035705544,
            -0.06597227881830146,
            -0.067370376854007,
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
        "Total": [
            0.7067638601845871,
            -0.22666620743385835,
            -0.8276329973991707,
            2.2954824562673153,
            0.4800976527507288,
            1.4678494588681446,
            -0.12086913721458359,
            2.068816248833457,
            1.9479471116188734,
        ],
    }

    ref_df = pd.DataFrame(data)
    cols = [
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
    df = df[cols]
    print("REF")
    print(ref_df)

    for col in cols:
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
    # pp({k: v.tolist() for k, v in dict(df).items()})
    data = {
        "Disp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "EDisp": [
            -0.0063878415595735455,
            -0.1304187252389881,
            -0.02369806434770492,
            -1.2663877356943494,
            -0.13680656679856162,
            -1.2900858000420543,
            -0.030085905907278467,
            -1.3968064609333375,
            -1.426892366840616,
        ],
        "Elst": [
            0.7173658713369733,
            -0.20556512078896816,
            -0.8178788520169107,
            -0.9342087766410643,
            0.5118007505480051,
            -1.752087628657975,
            -0.10051298067993741,
            -1.1397738974300324,
            -1.2402868781099698,
        ],
        "Exch": [
            0.00013545439373716334,
            0.047196822108368085,
            0.031615927654092296,
            3.8965970121551456,
            0.047332276502105246,
            3.928212939809238,
            0.03175138204782946,
            3.9437938342635137,
            3.975545216311343,
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
            -0.007097098275574362,
            -0.01562883221584556,
            -0.026071456678509793,
            -0.17470490231988067,
            -0.02272593049141992,
            -0.20077635899839047,
            -0.033168554954084155,
            -0.19033373453572622,
            -0.22350228948981038,
        ],
        "IndBA": [
            0.0003539943196737901,
            0.014741295760311841,
            -0.001752092356286135,
            -0.08071357462105197,
            0.015095290079985632,
            -0.0824656669773381,
            -0.0013980980366123448,
            -0.06597227886074013,
            -0.06737037689735248,
        ],
        "Total": [
            0.7043703802150378,
            -0.2896745603772417,
            -0.8377845377450437,
            1.4405820228794561,
            0.41469581983779613,
            0.6027974851344124,
            -0.13341415753000585,
            1.1509074625022144,
            1.0174933049722086,
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
    data = {
        "Disp": [
            -0.003994502630897984,
            -0.0674134405493367,
            -0.013547073146612729,
            -0.41154085645677463,
            -0.07140794318023469,
            -0.42508792960338737,
            -0.017541575777510712,
            -0.47895429700611136,
            -0.49649587278362206,
        ],
        "EDisp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Elst": [
            0.7173658716748221,
            -0.2055651208882452,
            -0.8178788521844282,
            -0.9342087776241712,
            0.5118007507865769,
            -1.7520876298085994,
            -0.10051298050960611,
            -1.1397738985124164,
            -1.2402868790220225,
        ],
        "Exch": [
            0.000135454393737376,
            0.04719682210839049,
            0.03161592765410519,
            3.8965970121551616,
            0.04733227650212787,
            3.9282129398092667,
            0.031751382047842565,
            3.943793834263552,
            3.9755452163113945,
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
            -0.007097098257764316,
            -0.015628832176699036,
            -0.02607145661295123,
            -0.17470490188100032,
            -0.022725930434463353,
            -0.20077635849395156,
            -0.033168554870715544,
            -0.19033373405769935,
            -0.22350228892841492,
        ],
        "IndBA": [
            0.0003539943187897063,
            0.014741295723273411,
            -0.0017520923518852605,
            -0.0807135744181731,
            0.015095290042063118,
            -0.08246566677005836,
            -0.0013980980330955543,
            -0.06597227869489969,
            -0.06737037672799524,
        ],
        "Total": [
            0.706763719498241,
            -0.22666927578227103,
            -0.8276335466433693,
            2.2954289017738034,
            0.48009444371597,
            1.4677953551304341,
            -0.12086982714512828,
            2.0687596259915324,
            1.9478897988464041,
        ],
    }

    ref_df = pd.DataFrame(data)
    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.0007912165332931369,
        "Eelst": -0.0019765265492708295,
        "Eexch": 0.006335438658802855,
        "Eind": -0.0004635353239533062,
        "Enuc": 474.74808217020274,
        "Etot": 0.003104160252285582,
    }
    #     print("FISAPT0 (REF)")
    #     psi4.set_options(
    #         {
    #             "basis": "sto-3g",
    #             "scf_type": "df",
    #             "guess": "sad",
    #             "FISAPT_FSAPT_FILEPATH": "none",
    #             "SAPT_DFT_FUNCTIONAL": "HF",
    #             "SAPT_DFT_DO_DHF": True,
    #             "SAPT_DFT_DO_HYBRID": False,
    #             "SAPT_DFT_DO_FSAPT": "SAPTDFT",
    # # a1eb1c8f985f13b48a00c2751f7e751572f0a696
    #             "SAPT_DFT_D4_IE": False,
    #             "SAPT_DFT_DO_DISP": True,
    #             "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
    #             # Normally on
    #             "SAPT_DFT_USE_EINSUMS": False,
    #         }
    #     )
    #     psi4.energy("fisapt0", molecule=mol)
    #     Epsi = {
    #         "Enuc": mol.nuclear_repulsion_energy(),
    #         "Eelst": core.variable("SAPT ELST ENERGY"),
    #         "Eexch": core.variable("SAPT EXCH ENERGY"),
    #         "Eind": core.variable("SAPT IND ENERGY"),
    #         "Edisp": core.variable("SAPT DISP ENERGY"),
    #         "Etot": core.variable("SAPT TOTAL ENERGY"),
    #     }
    #     for key in keys:
    #         compare_values(Eref[key], Epsi[key], 5, key)
    #     data = psi4.fsapt_analysis(
    #         molecule=mol,
    #         fragments_a={
    #             "Methyl1_A": [1, 2, 7, 8],
    #             "Methyl2_A": [3, 4, 5, 6],
    #         },
    #         fragments_b={
    #             "Peptide_B": [9, 10, 11, 16, 26],
    #             "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    #         },
    #         links5050=True,
    #         print_output=False,
    #     )
    #     df = pd.DataFrame(data)
    #     print("COMPUTED DF")
    #     print(df[['Frag1', 'Elst', 'IndAB', 'IndBA', 'Disp', 'Total']])
    #     # data_tmp = {k: v.tolist() for k, v in dict(df).items()}
    #     # pp(data_tmp)
    #
    #     print("REF")
    #     print(ref_df[['Frag1', 'Elst', 'IndAB', 'IndBA', 'Disp', 'Total']])
    #
    #     for col in ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]:
    #         for i in range(len(ref_df)):
    #             compare_values(
    #                 ref_df[col].iloc[i],
    #                 df[col].iloc[i],
    #                 4,
    #                 f"{ref_df['Frag1'].iloc[i]} {ref_df['Frag2'].iloc[i]} {col}",
    #             )
    #     psi4.set_options(
    #         {
    #             "basis": "sto-3g",
    #             "scf_type": "df",
    #             "guess": "sad",
    #             "FISAPT_FSAPT_FILEPATH": "none",
    #             "SAPT_DFT_FUNCTIONAL": "HF",
    #             "SAPT_DFT_DO_DHF": True,
    #             "SAPT_DFT_DO_HYBRID": False,
    #             "SAPT_DFT_DO_FSAPT": "SAPTDFT",
    # # a1eb1c8f985f13b48a00c2751f7e751572f0a696
    #             "SAPT_DFT_D4_IE": False,
    #             "SAPT_DFT_DO_DISP": True,
    #             "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
    #             # Normally on
    #             "SAPT_DFT_USE_EINSUMS": True,
    #         }
    #     )
    #     psi4.energy("sapt(dft)", molecule=mol)
    #     Epsi = {
    #         "Enuc": mol.nuclear_repulsion_energy(),
    #         "Eelst": core.variable("SAPT ELST ENERGY"),
    #         "Eexch": core.variable("SAPT EXCH ENERGY"),
    #         "Eind": core.variable("SAPT IND ENERGY"),
    #         "Edisp": core.variable("SAPT DISP ENERGY"),
    #         "Etot": core.variable("SAPT TOTAL ENERGY"),
    #     }
    #     pp(Epsi)
    #     for key in keys:
    #         compare_values(Eref[key], Epsi[key], 5, key)
    #     data = psi4.fsapt_analysis(
    #         molecule=mol,
    #         fragments_a={
    #             "Methyl1_A": [1, 2, 7, 8],
    #             "Methyl2_A": [3, 4, 5, 6],
    #         },
    #         fragments_b={
    #             "Peptide_B": [9, 10, 11, 16, 26],
    #             "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    #         },
    #         links5050=True,
    #         print_output=False,
    #     )
    #     df = pd.DataFrame(data)
    #     print("COMPUTED DF")
    #     print(df[['Frag1', 'Elst', 'IndAB', 'IndBA', 'Disp', 'Total']])
    #     # data_tmp = {k: v.tolist() for k, v in dict(df).items()}
    #     # pp(data_tmp)
    #
    #     print("REF")
    #     print(ref_df[['Frag1', 'Elst', 'IndAB', 'IndBA', 'Disp', 'Total']])
    #
    #     for col in ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]:
    #         for i in range(len(ref_df)):
    #             compare_values(
    #                 ref_df[col].iloc[i],
    #                 df[col].iloc[i],
    #                 4,
    #                 f"{ref_df['Frag1'].iloc[i]} {ref_df['Frag2'].iloc[i]} {col}",
    #             )
    print("SAPT_DFT_DO_FSAPT = FISAPT0 now testing")
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "FISAPT",
            "SAPT_DFT_D4_IE": False,
            "SAPT_DFT_DO_DISP": True,
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
            # Normally on
            "SAPT_DFT_USE_EINSUMS": True,
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
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
    print(df[["Frag1", "Elst", "IndAB", "IndBA", "Disp", "Total"]])
    # pp({k: v.tolist() for k, v in dict(df).items()})

    print("REF")
    print(ref_df[["Frag1", "Elst", "IndAB", "IndBA", "Disp", "Total"]])
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
    # pp({k: v.tolist() for k, v in dict(df).items()})
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
    # pp({k: v.tolist() for k, v in dict(df).items()})
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
    # pp({k: v.tolist() for k, v in dict(df).items()})
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


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_fsaptdft_fisapt0():
    """
    Compare SAPT energies from standard FISAPT0 with SAPT(DFT) using
    FISAPT option (SAPT_DFT_DO_FSAPT: "FISAPT").

    This test validates that the C++ flocalize() integration in SAPT(DFT)
    produces results consistent with the standard FISAPT0 code path.
    """
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

    # Run standard FISAPT0
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )
    psi4.energy("fisapt0", molecule=mol)

    # Collect FISAPT0 energies
    fisapt0_energies = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    print("FISAPT0 energies:")
    pp(fisapt0_energies)

    # Clear variables for next calculation
    psi4.core.clean()
    psi4.core.clean_variables()

    # Run SAPT(DFT) with FISAPT option (HF functional to match SAPT0)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            # "SAPT_DFT_DO_FSAPT": "FISAPT",
            "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_D4_IE": False,
            "SAPT_DFT_DO_DISP": True,
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
            "SAPT_DFT_USE_EINSUMS": False,
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)

    # Collect SAPT(DFT) energies
    saptdft_energies = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    print("SAPT(DFT) with FISAPT energies:")
    pp(saptdft_energies)

    # Compare total energies (5 decimal places = ~0.01 kcal/mol precision)
    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    for key in keys:
        compare_values(
            fisapt0_energies[key],
            saptdft_energies[key],
            5,
            f"Total {key}",
        )


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_fsaptdft_fisapt0_d4():
    """
    Compare SAPT energies from standard FISAPT0 with SAPT(DFT) using
    FISAPT option (SAPT_DFT_DO_FSAPT: "FISAPT").

    This test validates that the C++ flocalize() integration in SAPT(DFT)
    produces results consistent with the standard FISAPT0 code path.
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

    # Run standard FISAPT0
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "tmp_fisapt",
        }
    )
    psi4.energy("fisapt0-d4", molecule=mol)
    with open("tmp_fisapt/fA.dat", "w") as fA:
        fA.write("MethylA 1 2 3 4 5")
    with open("tmp_fisapt/fB.dat", "w") as fB:
        fB.write("MethylB 6 7 8 9 10")
    with open("tmp_fisapt/fA.dat", "w") as fA:  # TEST
        fA.write("Methyl1_A 1 2 7 8\n")  # TEST
        fA.write("Methyl2_A 3 4 5 6")  # TEST
    with open("tmp_fisapt/fB.dat", "w") as fB:  # TEST
        fB.write("Peptide_B  9 10 11 16 26\n")  # TEST
        fB.write("T-Butyl_B  12 13 14 15 17 18 19 20 21 22 23 24 25")  # TEST
    # import subprocess, sys, os                                          #TEST
    # subprocess.run([sys.executable, os.path.join('..', 'fsapt.py')], check=True) #TEST
    data = psi4.fsapt_analysis(
        molecule=mol,
        fragments_a={
            # "Methyl1_A": [i for i in range(1, 6)],
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        fragments_b={
            # "Methyl1_B": [j for j in range(6, 11)],
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
        pdb_dir="tmp",
    )
    # remove_fisapt files
    import shutil

    shutil.rmtree("tmp_fisapt")
    df = pd.DataFrame(data)
    print("COMPUTED DF FISAPT0")
    print(
        df[
            [
                "Frag1",
                "Frag2",
                "ClosestContact",
                "Elst",
                "IndAB",
                "IndBA",
                "Disp",
                "EDisp",
                "Total",
            ]
        ]
    )

    # Collect FISAPT0 energies
    fisapt0_energies = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    print("FISAPT0 energies:")
    pp(fisapt0_energies)

    # Clear variables for next calculation
    psi4.core.clean()
    psi4.core.clean_variables()

    # Run SAPT(DFT) with FISAPT option (HF functional to match SAPT0)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "FISAPT",
            # "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_DO_DISP": False,
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_D4_TYPE": "intermolecular",
            "SAPT_DFT_USE_EINSUMS": True,
            "FISAPT_FSAPT_FILEPATH": "tmp",
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)

    # Collect SAPT(DFT) energies
    saptdft_energies = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": core.variable("SAPT ELST ENERGY"),
        "Eexch": core.variable("SAPT EXCH ENERGY"),
        "Eind": core.variable("SAPT IND ENERGY"),
        "Edisp": core.variable("SAPT DISP ENERGY"),
        "Etot": core.variable("SAPT TOTAL ENERGY"),
    }
    print("SAPT(DFT) with FISAPT energies:")
    pp(saptdft_energies)
    data = psi4.fsapt_analysis(
        molecule=mol,
        fragments_a={
            # "Methyl1_A": [i for i in range(1, 6)],
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        fragments_b={
            # "Methyl1_B": [j for j in range(6, 11)],
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
        pdb_dir="tmp",
    )

    df = pd.DataFrame(data)
    print("COMPUTED DF")
    print(
        df[
            [
                "Frag1",
                "Frag2",
                "ClosestContact",
                "Elst",
                "IndAB",
                "IndBA",
                "Disp",
                "EDisp",
                "Total",
            ]
        ]
    )
    saptdft_energies = {k: v.tolist() for k, v in dict(df).items()}
    pp(saptdft_energies)
    ref_data = {
        "Disp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "EDisp": [
            -0.0063878415595735455,
            -0.13041872523898806,
            -0.02369806434770492,
            -1.2663877356943494,
            -0.13680656679856162,
            -1.2900858000420543,
            -0.030085905907278467,
            -1.3968064609333375,
            -1.426892366840616,
        ],
        "Elst": [
            0.7150991989031183,
            -0.20424514540867733,
            -0.8155054641111548,
            -0.9356354507588378,
            0.510854053494441,
            -1.7511409148699926,
            -0.10040626520803642,
            -1.1398805961675151,
            -1.2402868613755516,
        ],
        "Exch": [
            0.00013680473229194022,
            0.053104025194981225,
            0.030944134785956742,
            3.8913602515981354,
            0.053240829927273164,
            3.9223043863840923,
            0.031080939518248682,
            3.9444642767931164,
            3.9755452163113656,
        ],
        "IndAB": [
            -0.007088768001216014,
            -0.015599384317671102,
            -0.02601507069216381,
            -0.1747990661073405,
            -0.022688152318887114,
            -0.2008141367995043,
            -0.03310383869337982,
            -0.1903984504250116,
            -0.22350228911839143,
        ],
        "IndBA": [
            0.0003529338034038042,
            0.01470702778121176,
            -0.0017519903139542258,
            -0.0806783480564871,
            0.015059961584615564,
            -0.08243033837044134,
            -0.0013990565105504217,
            -0.06597132027527534,
            -0.06737037678582578,
        ],
        "Total": [
            0.7021123278784873,
            -0.28245220199025173,
            -0.8360264546800376,
            1.433859650984156,
            0.4196601258882356,
            0.5978331963041184,
            -0.13391412680155024,
            1.1514074489939041,
            1.0174933221923539,
        ],
    }
    keys = ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]
    for key in keys:
        for i in range(len(ref_data[key])):
            compare_values(
                ref_data[key][i],
                saptdft_energies[key][i],
                6,
                f"{df['Frag1'].tolist()[i]} {df['Frag2'].tolist()[i]} {key}",
            )


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_fsaptdftd4i():
    """
    Compare SAPT energies from standard FISAPT0 with SAPT(DFT) using
    FISAPT option (SAPT_DFT_DO_FSAPT: "FISAPT").

    This test validates that the C++ flocalize() integration in SAPT(DFT)
    produces results consistent with the standard FISAPT0 code path.
    """
    mol = psi4.geometry(
        """
0 1
C           18.929386598711     9.703743654226    47.390551774231
C           16.164717277166    10.278220396365    47.883770292976
C           14.573567879531    10.954742349279    45.895778408994
C           12.026217062414    11.466858129278    46.322856513348
C           11.001985502416    11.289223873485    48.788949107070
C            8.424399067291    11.814567736362    49.301064887069
C           12.621480791933    10.563569041309    50.822294418063
C           15.206626131559    10.089247783819    50.327186173193
C           11.606697862562    10.346250536882    53.282717833409
C            9.063126497696    10.850807412379    53.719244568390
C            7.477646278437    11.595359505809    51.755819124039
S            4.244324877778    12.349360229867    52.432341076953
O            3.475206344717    10.775218367360    54.582849407724
O            2.842148092689    12.302117076731    50.043727254374
N            4.312355018295    15.482526145876    53.539720586471
C            5.138165335120    17.530989265872    51.829518442932
H           15.304891890083    11.088912904187    43.996603652909
H           10.822461520497    11.984643087654    44.760053007594
H            7.209305168622    12.404162287505    47.768496999323
H           16.438727565358     9.596029265075    51.876761596068
H           12.823681487357     9.794450508248    54.828513804033
H            8.329912761018    10.676952608837    55.616529598349
H            4.969979709954    15.660160401669    55.350078214660
H            5.977203734823    16.597464559896    50.196795070536
H           19.634254443507     8.261882620502    48.683124444045
H           19.207176339153     8.964860739172    45.487597565895
H           20.072670904613    11.393158810385    47.630546992165
H            6.566798285966    18.676163297900    52.772491779535
H            3.497883058222    18.621361240261    51.224806082785
--
0 1
C 0.00000000 0.00000000 0.00000000
H 1.09000000 0.00000000 0.00000000
H -0.36333333 0.83908239 0.59332085
H -0.36333333 0.09428973 -1.02332709
H -0.36333333 -0.93337212 0.43000624
units bohr

symmetry c1
no_reorient
no_com
"""
    )
    # Above test takes a while actually
    mol = psi4.geometry(_sapt_testing_mols['neutral_water_dimer'])
    psi4.set_options(
        {
            # "basis": "aug-cc-pv(d+d)z",
            # "mp2_type": "df",
            # "scf_type": "df",
            # "guess": "SAPGAU",
            # "freeze_core": "true",
            # # SAPT(DFT) OPTIONS
            # "SAPT_DFT_FUNCTIONAL": "PBE0",
            # # "SAPT_DFT_FUNCTIONAL": "HF",
            # "SAPT_DFT_DO_DHF": False,
            # "SAPT_DFT_DO_HYBRID": False,
            # "SAPT_DFT_DO_FSAPT": "FISAPT",
            # "SAPT_DFT_DO_DISP": False,
            # "SAPT_DFT_D4_IE": True,
            # "SAPT_DFT_D4_TYPE": "intermolecular",
            # "SAPT_DFT_USE_EINSUMS": True,
            # "FISAPT_FSAPT_FILEPATH": "none",
            # # ITERATIVE
            # "SAPT_DFT_GRAC_COMPUTE": "ITERATIVE",
            # # "SAPT_DFT_GRAC_SHIFT_A": 0.05299154,
            # # GRAC SHIFTS failing to converge in 100 iterations for these systems
            # "MAXITER": 500,
            "basis": "aug-cc-pv(d+d)z",
            "scf_type": "df",
            "mp2_type": "df",
            "guess": "SAPGAU",
            "freeze_core": "true",
            "SAPT_DFT_FUNCTIONAL": "PBE0",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "SAPT_DFT_DO_FSAPT": "FISAPT",
            "SAPT_DFT_DO_DISP": False,
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_D4_TYPE": "intermolecular",
            "SAPT_DFT_USE_EINSUMS": True,
            "FISAPT_FSAPT_FILEPATH": "none",
            # ITERATIVE
            "SAPT_DFT_GRAC_COMPUTE": "ITERATIVE",
            "MAXITER": 500,
        }
    )
    psi4.energy("sapt(dft)", molecule=mol)
    data = psi4.fsapt_analysis(
        molecule=mol,
        fragments_a={
            "water1": [1, 2, 3],
        },
        fragments_b={
            "water2": [4, 5, 6],
        },
        links5050=True,
        print_output=False,
    )


if __name__ == "__main__":
    psi4.set_memory("220 GB")
    # psi4.set_num_threads(24)
    psi4.set_num_threads(12)
    # test_fsaptdft_timer()
    # test_fsaptdft_simple()

    # test_fsaptdft_fisapt0()
    test_fsaptdftd4i()
    # test_fsaptdft_fisapt0_d4()
    # test_fsaptdft_fisapt0()
    # test_fsaptdft_fisapt0()
    # test_fsaptdft()
    # test_fsaptdft_fsapt0_simple()
    # test_fsaptdftd4_psivars()
    # test_fsaptdft_disp0_fisapt0_psivars()
    # test_fsaptdftd4_psivars_pbe0()
    # test_fsaptdftd4_psivars_pbe0_frozen_core()
    # test_fsaptdft_indices()
    # test_fsaptdft_fsapt0()
    # test_fsapt0_fsaptdft()
    # test_fsaptdft_psivars()
    # test_fsapthf_psivars()
