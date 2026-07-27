import json
import os
from pprint import pprint as pp
import subprocess
import sys
import textwrap
import types

import numpy as np
import psi4
import pytest
from addons import uusing
from psi4 import compare_values
from psi4 import core
from qcelemental import constants

hartree_to_kcalmol = constants.conversion_factor("hartree", "kcal/mol")
pytestmark = [pytest.mark.psi, pytest.mark.api]


@uusing("pandas")
def test_fsaptdft_timer():
    """Ensure SAPT(DFT) timer CSV output contains expected timing columns."""
    import pandas as pd

    mol = psi4.geometry(
        """
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
            """
    )
    np.set_printoptions(precision=10, suppress=True)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
            "SAPT_DFT_DO_FSAPT": "FISAPT",
            "SAPT_DFT_USE_EINSUMS": True,
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        }
    )
    psi4.core.clean_timers()
    _ = psi4.energy("sapt(dft)", molecule=mol)
    compute_time_saptdft_fi_ein = psi4.core.get_timer_dict()["SAPT(DFT) Energy"]
    psi4.driver.p4util.write_timer_csv("saptdft_fi_useEin_timers.csv")
    df = pd.read_csv("saptdft_fi_useEin_timers.csv")
    os.remove("saptdft_fi_useEin_timers.csv")
    print(f"compute_time_fi_ein: {compute_time_saptdft_fi_ein['wall_time']:.2f}s\n")
    print(df)
    timer_cols = ["timer_name", "wall_time", "user_time", "system_time", "n_calls"]
    for col in timer_cols:
        assert col in df.columns, f"Expected column '{col}' not found in timer CSV"
    return


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.saptdft
def test_fsapthf_disp0_fisapt0_psivars():
    """
    Validate HF SAPT(DFT)+FISAPT energies and fragment terms vs references.

    NOTE: test use this larger molecular system for differently sized monA and
    monB fragments to ensure code operates correctly.
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
    # Reference energies from FISAPT0/sto-3g
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

    ref_data = data
    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.0007912165332931369,
        "Eelst": -0.0019765265492708295,
        "Eexch": 0.006335438658802855,
        "Eind": -0.0004635353239533062,
        "Enuc": 474.74808217020274,
        "Etot": 0.003104160252285582,
    }
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
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        }
    )
    _, wfn = psi4.energy("sapt(dft)", molecule=mol, return_wfn=True)
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
    fsapt_data = psi4.fsapt_analysis(
        source=wfn,
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
    for label_key in ["Frag1", "Frag2"]:
        assert list(fsapt_data[label_key]) == ref_data[label_key]
    for col in ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]:
        for i in range(len(ref_data[col])):
            compare_values(
                ref_data[col][i],
                fsapt_data[col][i],
                4,
                f"{ref_data['Frag1'][i]} {ref_data['Frag2'][i]} {col}",
            )


@pytest.mark.saptdft
@pytest.mark.fsapt
@uusing("pandas")
@pytest.mark.saptdft
@uusing("dftd4")
@pytest.mark.dftd4
@uusing("dftd4")
def test_fsaptdftd4_psivars():
    """Validate SAPT(DFT)-D4(s) scalar variables and FSAPT terms vs references."""
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
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        }
    )
    _, wfn = psi4.energy("sapt(dft)-d4(s)", return_wfn=True)

    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Edisp": -0.004568534767691285,
        "Eelst": -0.0019765266134612602,
        "Eexch": 0.006335438658900877,
        "Eind": -0.0004635353246623952,
        "Enuc": 474.74808217020274,
        "Etot": -0.0006731581723675157,
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
        source=wfn,
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
    pp({k: v.tolist() for k, v in dict(df).items()})
    ref_data = {
        "Disp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "EDisp": [
            -0.01301450872859415,
            -0.27084948543172516,
            -0.04866416123143305,
            -2.4459445415531187,
            -0.2838639941603193,
            -2.494608702784552,
            -0.0616786699600272,
            -2.716794026984844,
            -2.778472696944871,
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
            0.6977436680167427,
            -0.4301052921347477,
            -0.8627506894690783,
            0.2610253266522511,
            0.267638375881995,
            -0.6017253628168272,
            -0.1650070214523356,
            -0.1690799654824966,
            -0.3340869869348322,
        ],
    }

    ref_df = pd.DataFrame(ref_data)
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


@pytest.mark.fsapt
@pytest.mark.saptdft
@pytest.mark.dftd4
@uusing("dftd4")
def test_fsaptdftd4_psivars_pbe0_frozen_core():
    """Check PBE0 SAPT(DFT)-D4(i) fragment indices and per-fragment energies."""

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
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "FISAPT_FSAPT_FILEPATH": "none",
            "SAPT_DFT_FUNCTIONAL": "PBE0",
            "SAPT_DFT_DO_DHF": True,
            "SAPT_DFT_DO_HYBRID": False,
            # "SAPT_DFT_DO_FSAPT": "SAPTDFT",
            "SAPT_DFT_DO_FSAPT": "FISAPT",
            "SAPT_DFT_GRAC_SHIFT_A": 0.11652342,
            "SAPT_DFT_GRAC_SHIFT_B": 0.12724880,
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
            # "SAPT_DFT_USE_EINSUMS": False,
            "SAPT_DFT_USE_EINSUMS": True,
            "e_convergence": 1e-10,
            "d_convergence": 1e-10,
        }
    )
    _, wfn = psi4.energy("sapt(dft)-d4(i)", molecule=mol, return_wfn=True)
    fsapt_data = psi4.fsapt_analysis(
        # NOTE: 1-indexed for fragments_a and fragments_b
        source=wfn,
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
    mol_qcel_dict = mol.to_schema(dtype=2)
    frag1_indices = fsapt_data["Frag1_indices"]
    frag2_indices = fsapt_data["Frag2_indices"]
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
        assert sorted_frag == e, f"Frag1 indices do not match for fragment {i}: expected {e}, got {sorted_frag}"

    for i, indices in enumerate(frag2_indices):
        e = expected_frag2_indices[i]
        sorted_frag = sorted(indices)
        assert sorted_frag == e, f"Frag2 indices do not match for fragment {i}: expected {e}, got {sorted_frag}"
    ref_data = {
        "Disp": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "EDisp": [
            -0.013635081934517781,
            -0.2986013898239719,
            -0.052012447785209845,
            -3.3086784527471322,
            -0.3122364717584897,
            -3.360690900532342,
            -0.06564752971972762,
            -3.607279842571104,
            -3.6729273722908315,
        ],
        "Elst": [
            0.6885821033339496,
            -0.13402813039414951,
            -0.7952455429916299,
            -1.0514371642138727,
            0.5545539729398001,
            -1.8466827072055025,
            -0.10666343965768021,
            -1.1854652946080222,
            -1.2921287342657024,
        ],
        "Exch": [
            0.00015884227824908658,
            0.04729543660136764,
            0.039013303481442406,
            4.045753530568525,
            0.04745427887961673,
            4.084766834049967,
            0.039172145759691496,
            4.093048967169892,
            4.132221112929583,
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
            -0.008959883619643375,
            -0.014829486318378118,
            -0.02921897289731889,
            -0.17897050531529546,
            -0.02378936993802149,
            -0.20818947821261435,
            -0.03817885651696226,
            -0.19379999163367356,
            -0.23197884815063585,
        ],
        "IndBA": [
            0.0006366757929361327,
            0.019609284394791815,
            -0.002498480171516983,
            -0.095777249060534,
            0.020245960187727948,
            -0.09827572923205098,
            -0.0018618043785808502,
            -0.07616796466574219,
            -0.07802976904432303,
        ],
        "Total": [
            0.6667826558499702,
            -0.3805542855363413,
            -0.8399621403660978,
            -0.5891098407707545,
            0.2862283703136289,
            -1.4290719811368522,
            -0.17317948451612764,
            -0.9696641263070958,
            -1.1428436108232232,
        ],
    }

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
    print("REF:")
    pp(ref_data)
    print("FSAPT DATA:")
    pp(fsapt_data)

    assert list(fsapt_data["Frag1"]) == ref_data["Frag1"]
    assert list(fsapt_data["Frag2"]) == ref_data["Frag2"]

    for col in cols[2:]:
        for i in range(len(ref_data[col])):
            compare_values(
                ref_data[col][i],
                fsapt_data[col][i],
                4,
                f"{ref_data['Frag1'][i]} {ref_data['Frag2'][i]} {col}",
            )


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_fsaptdft_fisapt0():
    """Confirm fisapt0 and HF SAPT(DFT) produce matching SAPT energy components."""
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
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
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
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        }
    )
    _, wfn = psi4.energy("sapt(dft)", molecule=mol, return_wfn=True)

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
@pytest.mark.dftd4
@uusing("dftd4")
def test_fsaptdft_fisapt0_d4():
    """Compare fisapt0-d4 and SAPT(DFT)-D4(i)/FISAPT fragment decompositions."""

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
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        }
    )
    _, wfn = psi4.energy("fisapt0-d4", molecule=mol, return_wfn=True)
    with open("tmp_fisapt/fA.dat", "w") as fA:  # TEST
        fA.write("Methyl1_A 1 2 7 8\n")  # TEST
    with open("tmp_fisapt/fB.dat", "w") as fB:  # TEST
        fB.write("Peptide_B  9 10 11 16 26\n")  # TEST
        fB.write("T-Butyl_B  12 13 14 15 17 18 19 20 21 22 23 24 25")  # TEST
    psi4.fsapt_analysis(
        source=wfn,
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
        pdb_dir="tmp_fisapt",
    )
    # remove_fisapt files
    import shutil

    shutil.rmtree("tmp_fisapt")
    # rm pdb
    pdb_files = [
        "Disp.pdb",
        "EDisp.pdb",
        "Elst.pdb",
        "Exch.pdb",
        "IndAB.pdb",
        "IndBA.pdb",
        "Total.pdb",
    ]
    for pdb_file in pdb_files:
        if os.path.exists(pdb_file):
            os.remove(pdb_file)
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
            "SAPT_DFT_USE_EINSUMS": True,
            "FISAPT_FSAPT_FILEPATH": "tmp",
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        }
    )
    _, wfn = psi4.energy("sapt(dft)-d4(i)", molecule=mol, return_wfn=True)

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
    saptdft_fsapt_data = psi4.fsapt_analysis(
        source=wfn,
        fragments_a={
            "Methyl1_A": [1, 2, 7, 8],
        },
        fragments_b={
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
        pdb_dir="tmp",
    )
    # remove_fisapt files
    shutil.rmtree("tmp")
    for pdb_file in pdb_files:
        if os.path.exists(pdb_file):
            os.remove(pdb_file)
    ref_data = {
        "Disp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "EDisp": [
            -0.01301450872859415,
            -0.27084948543172516,
            -0.04866416123143305,
            -2.4459445415531187,
            -0.2838639941603193,
            -2.494608702784552,
            -0.0616786699600272,
            -2.716794026984844,
            -2.778472696944871,
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
            0.6954856149378384,
            -0.4228829336993627,
            -0.8609926109811682,
            0.25430294407805487,
            0.27260268123847575,
            -0.6066896669031133,
            -0.16550699604332975,
            -0.16857998962130782,
            -0.33408698566463757,
        ],
    }
    keys = ["Elst", "Exch", "IndAB", "IndBA", "Disp", "EDisp", "Total"]
    for key in keys:
        for i in range(len(ref_data[key])):
            compare_values(
                ref_data[key][i],
                saptdft_fsapt_data[key][i],
                6,
                f"{saptdft_fsapt_data['Frag1'][i]} {saptdft_fsapt_data['Frag2'][i]} {key}",
            )


def _saptdft_checkpoint_module():
    from psi4.driver.procrouting.sapt import saptdft_checkpoint

    return saptdft_checkpoint


def _saptdft_checkpoint_molecule(distance=3.0):
    return psi4.geometry(
        f"""
0 1
Ne 0.0 0.0 0.0
--
0 1
Ne 0.0 0.0 {distance}
units angstrom
symmetry c1
no_reorient
no_com
"""
    )


def _configure_saptdft_checkpoint_identity_options():
    core.clean_options()
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "sapt_dft_functional": "hf",
            "sapt_dft_do_dhf": True,
            "sapt_dft_do_hybrid": False,
            "sapt_dft_do_disp": True,
            "sapt_dft_mp2_disp_alg": "fisapt",
            "sapt_dft_do_fsapt": "fisapt",
            "sapt_dft_use_einsums": True,
            "fisapt_fsapt_filepath": "none",
            "orbital_optimizer_package": "internal",
        }
    )


def _saptdft_checkpoint_identity_inputs(molecule, function_kwargs=None, method="sapt(dft)"):
    atomic_input = psi4.driver.p4util.state_to_atomicinput(
        dtype=2,
        driver="energy",
        method=method,
        molecule=molecule,
        function_kwargs=function_kwargs,
    )
    return atomic_input


@pytest.fixture
def saptdft_checkpoint_identity_fixture():
    _configure_saptdft_checkpoint_identity_options()
    molecule = _saptdft_checkpoint_molecule()
    function_kwargs = {
        "checkpoint_dir": "first-dir",
        "checkpoint_stop_after": "elst",
        "output": "first.out",
        "memory": "1 GiB",
        "threads": 1,
        "timer": False,
        "verbosity": 1,
    }
    atomic_input = _saptdft_checkpoint_identity_inputs(
        molecule, function_kwargs=function_kwargs
    )
    yield molecule, function_kwargs, atomic_input
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_identity_is_deterministic(
    saptdft_checkpoint_identity_fixture,
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture

    identity_from_atomic = checkpoint_mod.build_saptdft_job_identity(
        name="SAPT(DFT)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    identity_from_state = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=dict(function_kwargs),
    )
    repeat_identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=dict(function_kwargs),
        atomic_input=atomic_input,
    )

    assert identity_from_atomic == identity_from_state
    assert identity_from_atomic == repeat_identity
    assert len(identity_from_atomic["sha256"]) == 64


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_identity_excludes_runtime_controls(
    saptdft_checkpoint_identity_fixture,
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, _, _ = saptdft_checkpoint_identity_fixture

    first_kwargs = {
        "checkpoint_dir": "first-dir",
        "checkpoint_stop_after": "elst",
        "output": "first.out",
        "memory": "1 GiB",
        "threads": 1,
        "timer": False,
        "verbosity": 1,
    }
    second_kwargs = {
        "checkpoint_dir": "second-dir",
        "checkpoint_stop_after": "disp",
        "output": "second.out",
        "memory": "2 GiB",
        "threads": 8,
        "timer": True,
        "verbosity": 5,
    }

    first_identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=first_kwargs,
        atomic_input=_saptdft_checkpoint_identity_inputs(
            molecule, function_kwargs=first_kwargs
        ),
    )
    second_identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=second_kwargs,
        atomic_input=_saptdft_checkpoint_identity_inputs(
            molecule, function_kwargs=second_kwargs
        ),
    )

    assert first_identity == second_identity
    serialized = json.dumps(first_identity["canonical_input"], sort_keys=True)
    for forbidden in [
        "checkpoint_dir",
        "checkpoint_stop_after",
        "first.out",
        "second.out",
        "threads",
        "verbosity",
        "timer",
    ]:
        assert forbidden not in serialized


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_identity_selected_backend(
    saptdft_checkpoint_identity_fixture,
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, _ = saptdft_checkpoint_identity_fixture

    psi4.set_options({"sapt_dft_use_einsums": False})
    numpy_identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=_saptdft_checkpoint_identity_inputs(
            molecule, function_kwargs=function_kwargs
        ),
    )

    psi4.set_options({"sapt_dft_use_einsums": True})
    selected_identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=_saptdft_checkpoint_identity_inputs(
            molecule, function_kwargs=function_kwargs
        ),
    )

    expected_selected_backend = (
        "einsums" if checkpoint_mod._saptdft_einsums_bundle_available() else "numpy"
    )
    assert numpy_identity["execution_fingerprint"]["selected_backend"] == "numpy"
    assert selected_identity["execution_fingerprint"]["selected_backend"] == expected_selected_backend
    assert "dftd3_version" not in selected_identity["execution_fingerprint"]
    assert "dftd4_version" not in selected_identity["execution_fingerprint"]
    if expected_selected_backend == "einsums":
        assert "einsums_version" in selected_identity["execution_fingerprint"]
    else:
        assert "einsums_version" not in selected_identity["execution_fingerprint"]


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_identity_einsums_helper_unavailable_uses_numpy(
    monkeypatch, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, _ = saptdft_checkpoint_identity_fixture
    original_import_module = checkpoint_mod.importlib.import_module

    def fake_import_module(name):
        if name == "einsums":
            return types.SimpleNamespace(__version__="test-einsums")
        if name == "psi4.driver.procrouting.sapt.sapt_jk_terms_ein":
            raise ImportError("missing SAPT einsums helper")
        return original_import_module(name)

    monkeypatch.setattr(checkpoint_mod.importlib, "import_module", fake_import_module)
    psi4.set_options({"sapt_dft_use_einsums": True})

    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=_saptdft_checkpoint_identity_inputs(
            molecule, function_kwargs=function_kwargs
        ),
    )

    assert checkpoint_mod._saptdft_einsums_bundle_available() is False
    assert identity["execution_fingerprint"]["selected_backend"] == "numpy"
    assert "einsums_version" not in identity["execution_fingerprint"]


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_manifest_schema(
    tmp_path, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    checkpoint.commit_stage(
        "hf_dimer_scf",
        scalars={"SAPT ELST ENERGY": -0.125},
        arrays={"Elst_AB": np.arange(4.0).reshape(2, 2)},
    )
    np.testing.assert_allclose(
        checkpoint.restore_array("Elst_AB"), np.arange(4.0).reshape(2, 2)
    )
    assert checkpoint.restore_scalars(["SAPT ELST ENERGY"]) == {
        "SAPT ELST ENERGY": -0.125
    }
    checkpoint.close()

    manifest = json.loads((tmp_path / "saptdft_state.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["job_identity"]["sha256"] == identity["sha256"]
    assert manifest["completed_stages"]["hf_dimer_scf"]["artifacts"] == ["Elst_AB"]
    assert manifest["completed_stages"]["hf_dimer_scf"]["scalars"] == ["SAPT ELST ENERGY"]
    assert manifest["artifacts"]["Elst_AB"]["kind"] == "array"
    assert manifest["artifacts"]["Elst_AB"]["path"].endswith(".npy")
    assert manifest["artifacts"]["Elst_AB"]["size"] > 0


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_rejects_changed_geometry(
    tmp_path, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    checkpoint.commit_stage("hf_dimer_scf", scalars={"SAPT ELST ENERGY": -0.125})
    checkpoint.close()

    other_molecule = _saptdft_checkpoint_molecule(distance=3.4)
    other_identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=other_molecule,
        function_kwargs=function_kwargs,
        atomic_input=_saptdft_checkpoint_identity_inputs(
            other_molecule, function_kwargs=function_kwargs
        ),
    )

    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match="geometry"):
        checkpoint_mod.SAPTDFTCheckpoint(tmp_path, other_identity).open()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_rejects_checksum_failure(
    tmp_path, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    checkpoint.commit_stage(
        "hf_dimer_scf",
        arrays={"Elst_AB": np.arange(9.0).reshape(3, 3)},
    )
    checkpoint.close()

    manifest = json.loads((tmp_path / "saptdft_state.json").read_text())
    artifact_path = tmp_path / manifest["artifacts"]["Elst_AB"]["path"]
    artifact_path.write_bytes(b"corrupt")

    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match="checksum"):
        checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity).open()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_rejects_artifact_path_escape(
    tmp_path, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    checkpoint.commit_stage(
        "hf_dimer_scf",
        arrays={"Elst_AB": np.arange(4.0).reshape(2, 2)},
    )
    checkpoint.close()

    outside_path = tmp_path.parent / "outside.npy"
    with outside_path.open("wb") as handle:
        np.save(handle, np.arange(2.0), allow_pickle=False)
    sha256, size = checkpoint_mod._file_digest_and_size(outside_path)

    manifest_path = tmp_path / "saptdft_state.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["Elst_AB"]["path"] = "../outside.npy"
    manifest["artifacts"]["Elst_AB"]["sha256"] = sha256
    manifest["artifacts"]["Elst_AB"]["size"] = size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(
        psi4.driver.p4util.exceptions.ValidationError,
        match="checkpoint directory",
    ):
        checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity).open()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_rejects_unknown_stage(
    tmp_path, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match="Unknown stage"):
        checkpoint.commit_stage("not_a_stage", scalars={"VALUE": 1.0})
    checkpoint.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_artifact_first_interruption(
    tmp_path, monkeypatch, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()

    def boom(_manifest):
        raise RuntimeError("manifest boom")

    monkeypatch.setattr(checkpoint, "_write_manifest_atomic", boom)
    with pytest.raises(RuntimeError, match="manifest boom"):
        checkpoint.commit_stage("hf_dimer_scf", arrays={"Elst_AB": np.arange(4.0)})
    checkpoint.close()

    assert not (tmp_path / "saptdft_state.json").exists()
    assert any(path.suffix == ".npy" for path in tmp_path.iterdir())

    reopened = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    reopened.open()
    assert not reopened.is_complete("hf_dimer_scf")
    reopened.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_wavefunction_artifact_smoke(
    tmp_path, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )

    _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)

    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    checkpoint.commit_stage("hf_dimer_scf", wavefunctions={"dimer_wfn": wfn})
    checkpoint.close()

    reopened = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    reopened.open()
    assert reopened.is_complete("hf_dimer_scf")
    artifact = reopened._manifest["artifacts"]["dimer_wfn"]
    assert artifact["kind"] == "wavefunction"
    artifact_path = reopened._validate_artifact("dimer_wfn", artifact)
    restored_wfn = core.Wavefunction.from_file(str(artifact_path))
    compare_values(wfn.energy(), restored_wfn.energy(), 10, "checkpoint wavefunction energy")
    reopened.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_lock_contention(
    tmp_path, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )

    first = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    first.open()
    second = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match="lock"):
        second.open()
    first.close()

    second.open()
    second.close()


def _configure_scf_snapshot_options(reference):
    core.clean_options()
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "reference": reference.lower(),
        }
    )


def _scf_snapshot_molecule():
    return psi4.geometry(
        """
0 1
H
H 1 0.74
symmetry c1
no_reorient
no_com
"""
    )


def _build_scf_snapshot_case(method, reference):
    _configure_scf_snapshot_options(reference)
    molecule = _scf_snapshot_molecule()
    _, wfn = psi4.energy(method, molecule=molecule, return_wfn=True)
    return molecule, wfn


def _guard_scf_rehydrate(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("SCF convergence entry point called during checkpoint rehydration")

    for attr in ["compute_energy", "guess", "diis"]:
        monkeypatch.setattr(core.HF, attr, boom)


def _assert_rehydrated_matches_loaded(restored, loaded, wfn, snapshot, expected_variable, reference):
    assert type(restored) is type(wfn)
    assert restored.functional().name() == wfn.functional().name()
    assert restored.energy() == loaded.energy()
    assert restored.has_variable(expected_variable)

    for matrix_name in ["Ca", "Cb", "Da", "Db", "Fa", "Fb"]:
        np.testing.assert_array_equal(
            getattr(restored, matrix_name)().np, getattr(loaded, matrix_name)().np
        )

    for vector_name in ["epsilon_a", "epsilon_b"]:
        np.testing.assert_array_equal(
            getattr(restored, vector_name)().np, getattr(loaded, vector_name)().np
        )

    for dimension_name in [
        "doccpi",
        "frzcpi",
        "frzvpi",
        "nalphapi",
        "nbetapi",
        "nmopi",
        "nsopi",
        "soccpi",
    ]:
        assert getattr(restored, dimension_name)().to_tuple() == getattr(
            loaded, dimension_name
        )().to_tuple()

    for variable_name in snapshot["scf_snapshot"]["required_fields"]["floatvar"]:
        assert restored.variable(variable_name) == loaded.variable(variable_name)

    jk = core.JK.build(restored.basisset(), restored.get_basisset("DF_BASIS_SCF"))
    jk.initialize()
    restored.set_jk(jk)
    trial = core.Matrix("trial", restored.doccpi(), restored.nmopi() - restored.doccpi())
    hx = restored.cphf_Hx([trial])
    assert len(hx) == 1
    np.testing.assert_array_equal(hx[0].np, np.zeros_like(trial.np))


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    "method, reference, expected_variable",
    [("hf", "RHF", "HF TOTAL ENERGY"), ("svwn", "RKS", "DFT TOTAL ENERGY")],
)
def test_saptdft_checkpoint_rehydrate_roundtrip_rhf_and_rks(
    tmp_path, monkeypatch, method, reference, expected_variable
):
    checkpoint_mod = _saptdft_checkpoint_module()
    _, wfn = _build_scf_snapshot_case(method, reference)
    snapshot = checkpoint_mod.capture_scf_snapshot(wfn, reference=reference, method=method)
    snapshot_path = tmp_path / f"{reference.lower()}_snapshot.npy"
    np.save(snapshot_path, snapshot, allow_pickle=True)
    loaded = core.Wavefunction.from_file(snapshot_path)
    _guard_scf_rehydrate(monkeypatch)

    restored = checkpoint_mod.rehydrate_scf_wavefunction(
        snapshot_path, method=method, reference=reference
    )

    _assert_rehydrated_matches_loaded(
        restored, loaded, wfn, snapshot, expected_variable, reference
    )
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_scf_snapshot_roundtrip(
    tmp_path, monkeypatch, saptdft_checkpoint_identity_fixture
):
    checkpoint_mod = _saptdft_checkpoint_module()
    molecule, function_kwargs, atomic_input = saptdft_checkpoint_identity_fixture
    identity = checkpoint_mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )

    _, wfn = _build_scf_snapshot_case("hf", "RHF")
    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    checkpoint.commit_stage(
        "hf_dimer_scf",
        scf_snapshots={
            "dimer_scf": {
                "wavefunction": wfn,
                "reference": "RHF",
                "method": "hf",
            }
        },
    )
    checkpoint.close()

    reopened = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    reopened.open()
    artifact = reopened._manifest["artifacts"]["dimer_scf"]
    assert artifact["kind"] == "scf_snapshot"
    snapshot = reopened.restore_scf_snapshot("dimer_scf")
    artifact_path = reopened._validate_artifact("dimer_scf", artifact)
    loaded = core.Wavefunction.from_file(str(artifact_path))
    _guard_scf_rehydrate(monkeypatch)
    restored = checkpoint_mod.rehydrate_scf_wavefunction(
        snapshot, method="hf", reference="RHF"
    )

    _assert_rehydrated_matches_loaded(
        restored, loaded, wfn, snapshot, "HF TOTAL ENERGY", "RHF"
    )
    reopened.close()
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    "mutator, expected_message",
    [
        (lambda snapshot: snapshot["scf_snapshot"].__setitem__("version", 999), "version"),
        (
            lambda snapshot: snapshot["scf_snapshot"]["molecule"]["geom"].__setitem__(0, 9.99),
            "molecule",
        ),
        (
            lambda snapshot: snapshot["scf_snapshot"]["basis"].__setitem__("name", "cc-pvdz"),
            "basis",
        ),
        (lambda snapshot: snapshot["scf_snapshot"].__setitem__("reference", "RKS"), "reference"),
        (lambda snapshot: snapshot["scf_snapshot"].__setitem__("functional", "bogus"), "functional"),
        (
            lambda snapshot: snapshot["scf_snapshot"]["dimensions"].__setitem__("doccpi", [0]),
            "dimensions",
        ),
        (lambda snapshot: snapshot["matrix"].__setitem__("Ca", None), "required"),
    ],
)
def test_saptdft_checkpoint_rehydrate_validates_snapshot_metadata(
    mutator, expected_message
):
    checkpoint_mod = _saptdft_checkpoint_module()
    _, wfn = _build_scf_snapshot_case("hf", "RHF")
    snapshot = checkpoint_mod.capture_scf_snapshot(wfn, reference="RHF", method="hf")
    mutator(snapshot)

    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match=expected_message):
        checkpoint_mod.rehydrate_scf_wavefunction(snapshot, method="hf", reference="RHF")
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    "mutator, expected_message",
    [
        (lambda snapshot: snapshot.__delitem__("matrix"), "top-level section matrix"),
        (lambda snapshot: snapshot.__setitem__("matrix", []), "top-level section matrix"),
        (lambda snapshot: snapshot["string"].__delitem__("basisname"), "section string"),
        (lambda snapshot: snapshot["scf_snapshot"].__delitem__("reference"), "section scf_snapshot"),
    ],
)
def test_saptdft_checkpoint_rehydrate_prevalidates_malformed_structure(
    monkeypatch, mutator, expected_message
):
    checkpoint_mod = _saptdft_checkpoint_module()
    _, wfn = _build_scf_snapshot_case("hf", "RHF")
    snapshot = checkpoint_mod.capture_scf_snapshot(wfn, reference="RHF", method="hf")
    mutator(snapshot)

    def should_not_deserialize(_snapshot):
        raise AssertionError("from_file should not be reached for malformed structure")

    monkeypatch.setattr(core.Wavefunction, "from_file", should_not_deserialize)
    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match=expected_message):
        checkpoint_mod.rehydrate_scf_wavefunction(snapshot, method="hf", reference="RHF")
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_rehydrate_translates_deserialization_failure(monkeypatch):
    checkpoint_mod = _saptdft_checkpoint_module()
    _, wfn = _build_scf_snapshot_case("hf", "RHF")
    snapshot = checkpoint_mod.capture_scf_snapshot(wfn, reference="RHF", method="hf")

    def boom(_snapshot):
        raise RuntimeError("bad payload")

    monkeypatch.setattr(core.Wavefunction, "from_file", boom)
    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match="deserialization failed"):
        checkpoint_mod.rehydrate_scf_wavefunction(snapshot, method="hf", reference="RHF")
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_rehydrate_never_uses_unsafe_direct_wrap(tmp_path):
    geometry = """
0 1
H
H 1 0.74
symmetry c1
no_reorient
no_com
"""
    script = textwrap.dedent(
        f"""
        import numpy as np
        import psi4
        from psi4 import core
        from psi4.driver.procrouting import proc
        from psi4.driver.procrouting.sapt import saptdft_checkpoint as checkpoint_mod

        core.clean_options()
        psi4.set_options({{"basis": "sto-3g", "scf_type": "df", "reference": "rhf"}})
        molecule = psi4.geometry({geometry!r})
        _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
        snapshot = checkpoint_mod.capture_scf_snapshot(wfn, reference="RHF", method="hf")
        snapshot_path = r"{tmp_path / 'unsafe_regression.npy'}"
        np.save(snapshot_path, snapshot, allow_pickle=True)

        original_build = core.Wavefunction.build
        def tagged_build(*args, **kwargs):
            fresh = original_build(*args, **kwargs)
            fresh._rehydrate_fresh = True
            return fresh
        core.Wavefunction.build = tagged_build

        original_factory = proc.scf_wavefunction_factory
        def guarded_factory(name, ref_wfn, reference, **kwargs):
            if not getattr(ref_wfn, "_rehydrate_fresh", False):
                raise RuntimeError("unsafe direct wrap")
            return original_factory(name, ref_wfn, reference, **kwargs)
        proc.scf_wavefunction_factory = guarded_factory

        restored = checkpoint_mod.rehydrate_scf_wavefunction(
            snapshot_path, method="hf", reference="RHF"
        )
        assert restored.energy() == wfn.energy()
        print("safe")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "safe" in completed.stdout


def _saptdft_prepared_checkpoint_monomers():
    from psi4.driver.procrouting import proc_util

    dimer = psi4.geometry(
        """
0 1
Ne 0 0 0
--
0 1
Ne 0 0 3.0
units angstrom
symmetry c1
no_reorient
no_com
"""
    )
    return proc_util.prepare_sapt_molecule(dimer, "dimer")


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_rehydrate_preserves_prepared_molecule_subset_nre(monkeypatch):
    checkpoint_mod = _saptdft_checkpoint_module()
    _configure_scf_snapshot_options("RKS")
    _, monomerA, _ = _saptdft_prepared_checkpoint_monomers()
    _, wfn = psi4.energy("svwn", molecule=monomerA, return_wfn=True)
    snapshot = checkpoint_mod.capture_scf_snapshot(wfn, reference="RKS", method="svwn")
    loaded = core.Wavefunction.from_file(snapshot)
    _guard_scf_rehydrate(monkeypatch)

    restored = checkpoint_mod.rehydrate_scf_wavefunction(
        snapshot, method="svwn", reference="RKS", molecule=monomerA
    )

    assert loaded.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy() == 0.0
    compare_values(
        wfn.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy(),
        restored.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy(),
        12,
        "prepared molecule subset nuclear repulsion",
    )
    core.clean_options()



def _run_fsaptdft_checkpoint_worker(
    *,
    checkpoint_dir,
    mode,
    stop_after=None,
    name="sapt(dft)",
    scenario="default",
    guard_jk=False,
    forbid_banners=None,
):
    worker = os.path.join(os.path.dirname(__file__), "fsaptdft_checkpoint_worker.py")
    command = [sys.executable, worker, mode, str(checkpoint_dir), "--name", name, "--scenario", scenario]
    if stop_after is not None:
        command.extend(["--stop-after", stop_after])
    if guard_jk:
        command.append("--guard-jk")
    for banner in forbid_banners or []:
        command.extend(["--forbid-banner", banner])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(output_lines[-1])
    return completed, payload


_TASK3_DEFAULT_STOP_STAGE_SETS = {
    "hf_dimer_scf": ["hf_dimer_scf"],
    "hf_monomer_a_scf": ["hf_dimer_scf", "hf_monomer_a_scf"],
    "hf_monomer_b_scf": ["hf_dimer_scf", "hf_monomer_a_scf", "hf_monomer_b_scf"],
    "monomer_a_dft_scf": [
        "hf_dimer_scf",
        "hf_monomer_a_scf",
        "hf_monomer_b_scf",
        "hf_sapt_elst",
        "hf_sapt_exch",
        "hf_sapt_ind",
        "monomer_a_dft_scf",
    ],
    "monomer_b_dft_scf": [
        "hf_dimer_scf",
        "hf_monomer_a_scf",
        "hf_monomer_b_scf",
        "hf_sapt_elst",
        "hf_sapt_exch",
        "hf_sapt_ind",
        "monomer_a_dft_scf",
        "monomer_b_dft_scf",
    ],
    "delta_dft_dimer_scf": [
        "hf_dimer_scf",
        "hf_monomer_a_scf",
        "hf_monomer_b_scf",
        "hf_sapt_elst",
        "hf_sapt_exch",
        "hf_sapt_ind",
        "monomer_a_dft_scf",
        "monomer_b_dft_scf",
        "delta_dft_dimer_scf",
    ],
    "delta_dft_monomer_a_scf": [
        "hf_dimer_scf",
        "hf_monomer_a_scf",
        "hf_monomer_b_scf",
        "hf_sapt_elst",
        "hf_sapt_exch",
        "hf_sapt_ind",
        "monomer_a_dft_scf",
        "monomer_b_dft_scf",
        "delta_dft_dimer_scf",
        "delta_dft_monomer_a_scf",
    ],
    "delta_dft_monomer_b_scf": [
        "hf_dimer_scf",
        "hf_monomer_a_scf",
        "hf_monomer_b_scf",
        "hf_sapt_elst",
        "hf_sapt_exch",
        "hf_sapt_ind",
        "monomer_a_dft_scf",
        "monomer_b_dft_scf",
        "delta_dft_dimer_scf",
        "delta_dft_monomer_a_scf",
        "delta_dft_monomer_b_scf",
    ],
}


_TASK3_DEFAULT_FORBIDDEN_BANNERS = {
    "hf_dimer_scf": ["SAPT(DFT): delta HF Dimer"],
    "hf_monomer_a_scf": ["SAPT(DFT): delta HF Dimer", "SAPT(DFT): delta HF Monomer A"],
    "hf_monomer_b_scf": [
        "SAPT(DFT): delta HF Dimer",
        "SAPT(DFT): delta HF Monomer A",
        "SAPT(DFT): delta HF Monomer B",
    ],
    "monomer_a_dft_scf": [
        "SAPT(DFT): delta HF Dimer",
        "SAPT(DFT): delta HF Monomer A",
        "SAPT(DFT): delta HF Monomer B",
        "SAPT(DFT): DFT Monomer A",
    ],
    "monomer_b_dft_scf": [
        "SAPT(DFT): delta HF Dimer",
        "SAPT(DFT): delta HF Monomer A",
        "SAPT(DFT): delta HF Monomer B",
        "SAPT(DFT): DFT Monomer A",
        "SAPT(DFT): DFT Monomer B",
    ],
    "delta_dft_dimer_scf": [
        "SAPT(DFT): delta HF Dimer",
        "SAPT(DFT): delta HF Monomer A",
        "SAPT(DFT): delta HF Monomer B",
        "SAPT(DFT): DFT Monomer A",
        "SAPT(DFT): DFT Monomer B",
    ],
    "delta_dft_monomer_a_scf": [
        "SAPT(DFT): delta HF Dimer",
        "SAPT(DFT): delta HF Monomer A",
        "SAPT(DFT): delta HF Monomer B",
        "SAPT(DFT): DFT Monomer A",
        "SAPT(DFT): DFT Monomer B",
    ],
    "delta_dft_monomer_b_scf": [
        "SAPT(DFT): delta HF Dimer",
        "SAPT(DFT): delta HF Monomer A",
        "SAPT(DFT): delta HF Monomer B",
        "SAPT(DFT): DFT Monomer A",
        "SAPT(DFT): DFT Monomer B",
    ],
}


def _task3_test_options(overrides=None):
    options = {
        "basis": "sto-3g",
        "scf_type": "df",
        "guess": "sad",
        "freeze_core": False,
        "orbital_optimizer_package": "internal",
        "sapt_dft_functional": "svwn",
        "sapt_dft_do_dhf": True,
        "sapt_dft_do_ddft": True,
        "sapt_dft_do_disp": False,
        "sapt_dft_do_fsapt": "none",
        "sapt_dft_do_hybrid": False,
        "sapt_dft_grac_shift_a": 0.0,
        "sapt_dft_grac_shift_b": 0.0,
        "sapt_dft_use_einsums": False,
    }
    options.update(overrides or {})
    return options


def _build_task3_checkpoint_identity(checkpoint_mod, *, name="sapt(dft)", options=None):
    core.clean_options()
    psi4.set_options(_task3_test_options(options))
    molecule = _saptdft_checkpoint_molecule()
    function_kwargs = {"checkpoint_dir": "identity-dir", "checkpoint_stop_after": "final"}
    atomic_input = _saptdft_checkpoint_identity_inputs(
        molecule,
        function_kwargs=function_kwargs,
        method=name,
    )
    identity = checkpoint_mod.build_saptdft_job_identity(
        name=name,
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    return molecule, identity


def _assert_checkpoint_stop_result(stopped, *, expected_stages, stop_stage):
    checkpoint_mod = _saptdft_checkpoint_module()
    assert stopped["completed_stages"] == sorted(expected_stages)
    assert stopped["manifest"]["completed_stages"][stop_stage]["dependencies"] == list(
        checkpoint_mod.selected_stage_dependencies(stopped["manifest"]["job_identity"], stop_stage)
    )


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_stage_dependencies():
    checkpoint_mod = _saptdft_checkpoint_module()
    from psi4.driver.procrouting.sapt import sapt_proc as sapt_proc_mod

    assert not hasattr(sapt_proc_mod, "_SAPTDFT_CHECKPOINT_STAGES")
    for stage in ["elst", "exch", "ind", "disp", "delta_dft", "d3", "d4", "final"]:
        assert stage in checkpoint_mod.SAPTDFT_STAGE_DEFINITIONS
    assert "build_jk" not in checkpoint_mod.SAPTDFT_STAGE_DEFINITIONS
    assert "hf_sapt_jk" not in checkpoint_mod.SAPTDFT_STAGE_DEFINITIONS


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("name", "options", "expected_present", "expected_absent"),
    [
        pytest.param(
            "sapt(dft)",
            None,
            [
                "hf_dimer_scf",
                "hf_sapt_ind",
                "monomer_b_dft_scf",
                "delta_dft",
                "elst",
                "ind",
                "final",
            ],
            ["dimer_localization_scf", "disp", "d3", "d4", "fsapt_setup"],
            id="default",
        ),
        pytest.param(
            "sapt(dft)",
            {"sapt_dft_do_ddft": False},
            ["hf_dimer_scf", "hf_sapt_ind", "monomer_b_dft_scf", "elst", "ind", "final"],
            ["delta_dft_dimer_scf", "delta_dft_monomer_a_scf", "delta_dft_monomer_b_scf", "delta_dft"],
            id="no-ddft",
        ),
        pytest.param(
            "sapt(dft)",
            {"sapt_dft_do_dhf": False, "sapt_dft_do_ddft": False, "sapt_dft_do_fsapt": "fisapt"},
            [
                "dimer_localization_scf",
                "monomer_b_dft_scf",
                "elst",
                "ind",
                "fsapt_setup",
                "fsapt_ind",
                "fsapt_final",
                "final",
            ],
            ["hf_dimer_scf", "hf_sapt_elst", "delta_dft", "disp", "d3", "d4", "fsapt_disp"],
            id="localization",
        ),
        pytest.param(
            "sapt(dft)",
            {"sapt_dft_functional": "hf", "sapt_dft_do_ddft": False},
            ["hf_dimer_scf", "hf_monomer_b_scf", "elst", "ind", "final"],
            ["hf_sapt_elst", "monomer_a_dft_scf", "delta_dft", "disp", "fsapt_setup"],
            id="hf-do-dhf",
        ),
        pytest.param(
            "sapt(dft)",
            {"sapt_dft_do_ddft": False, "sapt_dft_do_disp": True},
            ["hf_dimer_scf", "monomer_b_dft_scf", "elst", "ind", "disp", "final"],
            ["delta_dft", "d3", "d4", "fsapt_setup"],
            id="disp",
        ),
        pytest.param(
            "sapt(dft)-d3(s)",
            {"sapt_dft_functional": "pbe0"},
            ["hf_dimer_scf", "monomer_b_dft_scf", "elst", "ind", "d3", "final"],
            ["delta_dft", "disp", "d4", "fsapt_setup"],
            id="d3-method-selected",
        ),
        pytest.param(
            "sapt(dft)-d4(s)",
            {"sapt_dft_functional": "pbe0"},
            ["hf_dimer_scf", "monomer_b_dft_scf", "elst", "ind", "d4", "final"],
            ["delta_dft", "disp", "d3", "fsapt_setup"],
            id="d4-method-selected",
        ),
        pytest.param(
            "sapt(dft)",
            {
                "sapt_dft_functional": "hf",
                "sapt_dft_do_dhf": False,
                "sapt_dft_do_ddft": False,
                "sapt_dft_do_disp": True,
                "sapt_dft_do_fsapt": "fisapt",
            },
            [
                "dimer_localization_scf",
                "monomer_b_dft_scf",
                "elst",
                "ind",
                "disp",
                "fsapt_setup",
                "fsapt_disp",
                "fsapt_final",
                "final",
            ],
            ["hf_dimer_scf", "hf_sapt_ind", "delta_dft", "d3", "d4"],
            id="fsapt-disp-conditional",
        ),
    ],
)
def test_saptdft_checkpoint_selected_stages(tmp_path, name, options, expected_present, expected_absent):
    checkpoint_mod = _saptdft_checkpoint_module()
    _, identity = _build_task3_checkpoint_identity(checkpoint_mod, name=name, options=options)
    selected = checkpoint_mod.selected_stages(identity)

    for stage in expected_present:
        assert stage in selected
    for stage in expected_absent:
        assert stage not in selected

    checkpoint_dir = tmp_path / name.replace("(", "_").replace(")", "_").replace("-", "_")
    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(checkpoint_dir, identity)
    checkpoint.open()
    for stage in selected:
        checkpoint.commit_stage(stage)
    checkpoint.close()

    reopened = checkpoint_mod.SAPTDFTCheckpoint(checkpoint_dir, identity)
    reopened.open()
    assert set(reopened._manifest["completed_stages"]) == set(selected)
    reopened.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("name", "options", "offpath_stages"),
    [
        pytest.param("sapt(dft)", None, ["dimer_localization_scf", "d3", "d4"], id="default-rejects-localization-and-d3d4"),
        pytest.param(
            "sapt(dft)",
            {"sapt_dft_do_ddft": False},
            ["delta_dft_dimer_scf", "delta_dft_monomer_a_scf", "delta_dft_monomer_b_scf", "delta_dft"],
            id="no-ddft-rejects-delta",
        ),
        pytest.param(
            "sapt(dft)",
            None,
            ["fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind", "fsapt_disp", "fsapt_final"],
            id="non-fsapt-rejects-fsapt",
        ),
    ],
)
def test_saptdft_checkpoint_rejects_offpath_stages(tmp_path, name, options, offpath_stages):
    checkpoint_mod = _saptdft_checkpoint_module()
    _, identity = _build_task3_checkpoint_identity(checkpoint_mod, name=name, options=options)

    commit_checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path / "commit", identity)
    commit_checkpoint.open()
    for stage in offpath_stages:
        with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match=stage):
            commit_checkpoint.commit_stage(stage)
    commit_checkpoint.close()

    for index, stage in enumerate(offpath_stages):
        checkpoint_dir = tmp_path / f"manifest-{index}"
        checkpoint_dir.mkdir()
        manifest = {
            "schema_version": checkpoint_mod.SAPTDFT_CHECKPOINT_SCHEMA_VERSION,
            "job_identity": identity,
            "completed_stages": {
                stage: {
                    "artifacts": [],
                    "dependencies": [],
                    "scalars": [],
                    "version": checkpoint_mod.SAPTDFT_STAGE_DEFINITION_VERSION,
                }
            },
            "scalars": {},
            "artifacts": {},
        }
        (checkpoint_dir / checkpoint_mod.SAPTDFT_MANIFEST_FILENAME).write_text(json.dumps(manifest))
        with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match=stage):
            checkpoint_mod.SAPTDFTCheckpoint(checkpoint_dir, identity).open()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("name", "options", "dispersion_stage"),
    [
        pytest.param("sapt(dft)-d3(s)", {"sapt_dft_functional": "pbe0"}, "d3", id="d3"),
        pytest.param("sapt(dft)-d4(s)", {"sapt_dft_functional": "pbe0"}, "d4", id="d4"),
    ],
)
def test_saptdft_checkpoint_stage_dependencies_selected_method_dispersion_path(tmp_path, name, options, dispersion_stage):
    checkpoint_mod = _saptdft_checkpoint_module()
    _, identity = _build_task3_checkpoint_identity(checkpoint_mod, name=name, options=options)
    assert dispersion_stage in checkpoint_mod.selected_stages(identity)
    assert checkpoint_mod.selected_stage_dependencies(identity, "final") == ("ind", dispersion_stage)

    checkpoint = checkpoint_mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    for stage in [
        "hf_dimer_scf",
        "hf_monomer_a_scf",
        "hf_monomer_b_scf",
        "hf_sapt_elst",
        "hf_sapt_exch",
        "hf_sapt_ind",
        "monomer_a_dft_scf",
        "monomer_b_dft_scf",
        "elst",
        "exch",
        "ind",
    ]:
        checkpoint.commit_stage(stage)
    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match=dispersion_stage):
        checkpoint.commit_stage("final")
    checkpoint.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize("stop_stage", list(_TASK3_DEFAULT_STOP_STAGE_SETS))
def test_saptdft_checkpoint_restart_skips_scf(tmp_path, stop_stage):
    _, reference = _run_fsaptdft_checkpoint_worker(checkpoint_dir="", mode="reference")
    assert reference["status"] == "ok"

    checkpoint_dir = tmp_path / stop_stage
    stopped_proc, stopped = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="stop",
        stop_after=stop_stage,
    )
    assert stopped_proc.returncode == 0, stopped_proc.stderr or stopped_proc.stdout
    assert stopped["status"] == "stopped"
    _assert_checkpoint_stop_result(
        stopped,
        expected_stages=_TASK3_DEFAULT_STOP_STAGE_SETS[stop_stage],
        stop_stage=stop_stage,
    )

    restarted_proc, restarted = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="restart_with_guards",
        forbid_banners=_TASK3_DEFAULT_FORBIDDEN_BANNERS[stop_stage],
    )
    assert restarted_proc.returncode == 0, restarted_proc.stderr or restarted_proc.stdout
    assert restarted["status"] == "ok"
    compare_values(reference["elst10_r"], restarted["elst10_r"], 8, f"checkpoint restart electrostatics {stop_stage}")
    compare_values(reference["sapt_total_energy"], restarted["sapt_total_energy"], 8, f"checkpoint restart energy {stop_stage}")


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_restart_skips_localization_scf(tmp_path):
    _, reference = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=tmp_path / "localization-reference",
        mode="stop",
        stop_after="monomer_a_dft_scf",
        scenario="localization",
    )
    assert reference["status"] == "stopped"

    checkpoint_dir = tmp_path / "dimer_localization_scf"
    stopped_proc, stopped = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="stop",
        stop_after="dimer_localization_scf",
        scenario="localization",
    )
    assert stopped_proc.returncode == 0, stopped_proc.stderr or stopped_proc.stdout
    assert stopped["status"] == "stopped"
    _assert_checkpoint_stop_result(
        stopped,
        expected_stages=["dimer_localization_scf"],
        stop_stage="dimer_localization_scf",
    )

    restarted_proc, restarted = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="restart_with_guards",
        scenario="localization",
        stop_after="monomer_a_dft_scf",
        forbid_banners=["SAPT(DFT): Dimer for Localization"],
    )
    assert restarted_proc.returncode == 0, restarted_proc.stderr or restarted_proc.stdout
    assert restarted["status"] == "stopped"
    _assert_checkpoint_stop_result(
        restarted,
        expected_stages=["dimer_localization_scf", "monomer_a_dft_scf"],
        stop_stage="monomer_a_dft_scf",
    )
    compare_values(reference["current_energy"], restarted["current_energy"], 8, "checkpoint restart localization monomer energy")


@pytest.mark.saptdft
@pytest.mark.fsapt
@uusing("dftd4")
@pytest.mark.dftd4
def test_saptdft_checkpoint_d4(tmp_path):
    expected_stages = [
        "hf_dimer_scf",
        "hf_monomer_a_scf",
        "hf_monomer_b_scf",
        "hf_sapt_elst",
        "hf_sapt_exch",
        "hf_sapt_ind",
        "monomer_a_dft_scf",
        "monomer_b_dft_scf",
        "d4",
    ]
    _, reference = _run_fsaptdft_checkpoint_worker(checkpoint_dir="", mode="reference", name="sapt(dft)-d4(s)")
    assert reference["status"] == "ok"

    checkpoint_dir = tmp_path / "d4"
    stopped_proc, stopped = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="stop",
        stop_after="d4",
        name="sapt(dft)-d4(s)",
    )
    assert stopped_proc.returncode == 0, stopped_proc.stderr or stopped_proc.stdout
    assert stopped["status"] == "stopped"
    _assert_checkpoint_stop_result(stopped, expected_stages=expected_stages, stop_stage="d4")

    restarted_proc, restarted = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="restart_with_guards",
        name="sapt(dft)-d4(s)",
        forbid_banners=[
            "SAPT(DFT): delta HF Dimer",
            "SAPT(DFT): delta HF Monomer A",
            "SAPT(DFT): delta HF Monomer B",
            "SAPT(DFT): DFT Monomer A",
            "SAPT(DFT): DFT Monomer B",
        ],
    )
    assert restarted_proc.returncode == 0, restarted_proc.stderr or restarted_proc.stdout
    assert restarted["status"] == "ok"
    compare_values(reference["sapt_total_energy"], restarted["sapt_total_energy"], 8, "checkpoint restart d4 energy")


@pytest.mark.saptdft
@pytest.mark.fsapt
@uusing("s-dftd3")
@pytest.mark.dftd3
def test_saptdft_checkpoint_d3(tmp_path):
    expected_stages = [
        "hf_dimer_scf",
        "hf_monomer_a_scf",
        "hf_monomer_b_scf",
        "hf_sapt_elst",
        "hf_sapt_exch",
        "hf_sapt_ind",
        "monomer_a_dft_scf",
        "monomer_b_dft_scf",
        "d3",
    ]
    _, reference = _run_fsaptdft_checkpoint_worker(checkpoint_dir="", mode="reference", name="sapt(dft)-d3(s)")
    assert reference["status"] == "ok"

    checkpoint_dir = tmp_path / "d3"
    stopped_proc, stopped = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="stop",
        stop_after="d3",
        name="sapt(dft)-d3(s)",
    )
    assert stopped_proc.returncode == 0, stopped_proc.stderr or stopped_proc.stdout
    assert stopped["status"] == "stopped"
    _assert_checkpoint_stop_result(stopped, expected_stages=expected_stages, stop_stage="d3")

    restarted_proc, restarted = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="restart_with_guards",
        name="sapt(dft)-d3(s)",
        forbid_banners=[
            "SAPT(DFT): delta HF Dimer",
            "SAPT(DFT): delta HF Monomer A",
            "SAPT(DFT): delta HF Monomer B",
            "SAPT(DFT): DFT Monomer A",
            "SAPT(DFT): DFT Monomer B",
        ],
    )
    assert restarted_proc.returncode == 0, restarted_proc.stderr or restarted_proc.stdout
    assert restarted["status"] == "ok"
    compare_values(reference["sapt_total_energy"], restarted["sapt_total_energy"], 8, "checkpoint restart d3 energy")


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_final_restart_returns_before_scf_and_jk(tmp_path):
    _, reference = _run_fsaptdft_checkpoint_worker(checkpoint_dir="", mode="reference")
    checkpoint_dir = tmp_path / "final"
    stopped_proc, stopped = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="stop",
        stop_after="final",
    )
    assert stopped_proc.returncode == 0, stopped_proc.stderr or stopped_proc.stdout
    assert stopped["status"] == "stopped"
    _assert_checkpoint_stop_result(
        stopped,
        expected_stages=[
            "hf_dimer_scf",
            "hf_monomer_a_scf",
            "hf_monomer_b_scf",
            "hf_sapt_elst",
            "hf_sapt_exch",
            "hf_sapt_ind",
            "monomer_a_dft_scf",
            "monomer_b_dft_scf",
            "delta_dft_dimer_scf",
            "delta_dft_monomer_a_scf",
            "delta_dft_monomer_b_scf",
            "delta_dft",
            "elst",
            "exch",
            "ind",
            "final",
        ],
        stop_stage="final",
    )

    restarted_proc, restarted = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir,
        mode="restart_with_guards",
        guard_jk=True,
    )
    assert restarted_proc.returncode == 0, restarted_proc.stderr or restarted_proc.stdout
    assert restarted["status"] == "ok"
    compare_values(reference["sapt_total_energy"], restarted["sapt_total_energy"], 8, "final restart energy")


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_unexpected_exception_closes_lock_same_process(tmp_path, monkeypatch):
    from psi4.driver.procrouting import proc
    from psi4.driver.procrouting.sapt import sapt_proc as sapt_proc_mod

    mol = _saptdft_checkpoint_molecule()
    psi4.set_options(_task3_test_options())

    def boom(*args, **kwargs):
        raise RuntimeError("checkpoint crash probe")

    with monkeypatch.context() as ctx:
        ctx.setattr(proc, "scf_helper", boom)
        ctx.setattr(sapt_proc_mod, "scf_helper", boom)
        with pytest.raises(RuntimeError, match="checkpoint crash probe"):
            psi4.energy("sapt(dft)", molecule=mol, checkpoint_dir=str(tmp_path))

    stopped_proc, stopped = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=tmp_path,
        mode="stop",
        stop_after="hf_dimer_scf",
    )
    assert stopped_proc.returncode == 0, stopped_proc.stderr or stopped_proc.stdout
    assert stopped["status"] == "stopped"


if __name__ == "__main__":
    psi4.set_memory("220 GB")
    # psi4.set_num_threads(24)
    psi4.set_num_threads(12)
    # test_fsaptdft_timer()
    # test_fsaptdftd4_psivars_pbe0_frozen_core()
    # test_fsapthf_disp0_fisapt0_psivars()
    pytest.main([
        __file__,
        "-v",
        "-s",
        "-k=test_saptdft_auto_grac",
        "--disable-warnings",
        # "--maxfail=1",
    ])
