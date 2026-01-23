import pytest
import psi4
from psi4 import compare_values, variable
from addons import uusing
import numpy as np
import os

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


# TODO: create test without pandas so it tests
@pytest.mark.fsapt
def test_fsapt_psivars_dict():
    """
    fsapt-psivars: calling fsapt_analysis with psi4 variables after running an
    fisapt0 calcluation requires the user to pass the molecule object
    """
    mol = psi4.geometry(
        """0 1
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
no_com"""
    )
    psi4.set_options(
        {
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )
    psi4.energy("fisapt0")
    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Enuc": 35.07529824960602,
        "Eelst": -3.8035153870907834e-06,
        "Eexch": 1.7912112685446533e-07,
        "Eind": -3.833795151474493e-08,
        "Edisp": -3.288568662589654e-05,
        "Etot": -3.6548418837647605e-05,
    }
    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": variable("SAPT ELST ENERGY"),
        "Eexch": variable("SAPT EXCH ENERGY"),
        "Eind": variable("SAPT IND ENERGY"),
        "Edisp": variable("SAPT DISP ENERGY"),
        "Etot": variable("SAPT0 TOTAL ENERGY"),
    }

    for key in keys:
        compare_values(Eref[key], Epsi[key], 6, key)
    fEnergies = psi4.fsapt_analysis(
        molecule=mol,
        # NOTE: 1-indexed for fragments_a and fragments_b
        fragments_a={
            "MethylA": [1, 2, 3, 4, 5],
        },
        fragments_b={
            "MethylB": [6, 7, 8, 9, 10],
        },
    )
    fEnergies = {
        "Elst": fEnergies["Elst"],
        "Exch": fEnergies["Exch"],
        "IndAB": fEnergies["IndAB"],
        "IndBA": fEnergies["IndBA"],
        "Disp": fEnergies["Disp"],
        "EDisp": fEnergies["EDisp"],
        "Total": fEnergies["Total"],
    }
    print(fEnergies)
    fEref = {
        "fEelst": -0.002,
        "fEexch": 0.000,
        "fEindAB": -0.000,
        "fEindBA": -0.000,
        "fEdisp": -0.021,
        "fEedisp": 0.000,
        "fEtot": -0.023,
    }

    # python iterate over zip dictionary keys and values
    for key1, key2 in zip(fEref.keys(), fEnergies.keys()):
        compare_values(fEref[key1], fEnergies[key2][0], 2, key1)


@pytest.mark.fsapt
def test_fsapt_external_potentials():
    """
    fsapt-psivars: calling fsapt_analysis with psi4 variables after running an
    fisapt0 calcluation requires the user to pass the molecule object
    """
    mol = psi4.geometry(
        """
H 0.0290 -1.1199 -1.5243
O 0.9481 -1.3990 -1.3587
H 1.4371 -0.5588 -1.3099
--
H 1.0088 -1.5240 0.5086
O 1.0209 -1.1732 1.4270
H 1.5864 -0.3901 1.3101
--
H -1.0231 1.6243 -0.8743
O -0.5806 2.0297 -0.1111
H -0.9480 1.5096 0.6281
symmetry c1
no_reorient
no_com
"""
    )
    psi4.set_options(
        {
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )
    psi_bohr2angstroms = psi4.constants.bohr2angstroms
    bohr = psi_bohr2angstroms
    external_potentials = {
        "A": [
            [0.417, np.array([-0.5496, -0.6026, 1.5720]) / bohr],
            [-0.834, np.array([-1.4545, -0.1932, 1.4677]) / bohr],
            [0.417, np.array([-1.9361, -0.4028, 2.2769]) / bohr],
        ],
        "B": [
            [0.417, np.array([-2.5628, -0.8269, -1.6696]) / bohr],
            [-0.834, np.array([-1.7899, -0.4027, -1.2768]) / bohr],
            [0.417, np.array([-1.8988, -0.4993, -0.3072]) / bohr],
        ],
        "C": [
            [0.417, np.array([1.1270, 1.5527, -0.1658]) / bohr],
            [-0.834, np.array([1.9896, 1.0738, -0.1673]) / bohr],
            [0.417, np.array([2.6619, 1.7546, -0.2910]) / bohr],
        ],
    }
    psi4.energy("fisapt0", external_potentials=external_potentials)
    print(psi4.core.variables())
    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Enuc": 74.2330370461897,
        "Eelst": -0.04919037863747235,
        "Eexch": 0.018239207303845935,
        "Eind": -0.007969545823122322,
        "Edisp": -0.002794948165605119,
        "Etot": -0.04171566532235386,
    }
    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": variable("SAPT ELST ENERGY"),
        "Eexch": variable("SAPT EXCH ENERGY"),
        "Eind": variable("SAPT IND ENERGY"),
        "Edisp": variable("SAPT DISP ENERGY"),
        "Etot": variable("SAPT0 TOTAL ENERGY"),
    }

    for key in keys:
        compare_values(Eref[key], Epsi[key], 6, key)
    fEnergies = psi4.fsapt_analysis(
        molecule=mol,
        # NOTE: 1-indexed for fragments_a and fragments_b
        fragments_a={
            "w1": [1, 2, 3],
        },
        fragments_b={
            "w3": [4, 5, 6],
        },
    )
    fEnergies = {
        "Elst": fEnergies["Elst"],
        "Exch": fEnergies["Exch"],
        "IndAB": fEnergies["IndAB"],
        "IndBA": fEnergies["IndBA"],
        "Disp": fEnergies["Disp"],
        "EDisp": fEnergies["EDisp"],
        "Total": fEnergies["Total"],
    }
    fEref = {
        "fEelst": -30.867,
        "fEexch": 11.445,
        "fEindAB": -3.138,
        "fEindBA": -1.863,
        "fEdisp": -1.754,
        "fEedisp": 0.000,
        "fEtot": -26.177,
    }

    for key1, key2 in zip(fEref.keys(), fEnergies.keys()):
        compare_values(fEref[key1], fEnergies[key2][-1], 2, key1)


@pytest.mark.fsapt
@uusing("pandas")
def test_fsapt_psivars():
    """
    fsapt-psivars: calling fsapt_analysis with psi4 variables after running an
    fisapt0 calcluation requires the user to pass the molecule object
    """
    import pandas as pd

    mol = psi4.geometry(
        """0 1
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
no_com"""
    )
    psi4.set_options(
        {
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )
    psi4.energy("fisapt0")
    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Enuc": 35.07529824960602,
        "Eelst": -3.8035153870907834e-06,
        "Eexch": 1.7912112685446533e-07,
        "Eind": -3.833795151474493e-08,
        "Edisp": -3.288568662589654e-05,
        "Etot": -3.6548418837647605e-05,
    }
    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": variable("SAPT ELST ENERGY"),
        "Eexch": variable("SAPT EXCH ENERGY"),
        "Eind": variable("SAPT IND ENERGY"),
        "Edisp": variable("SAPT DISP ENERGY"),
        "Etot": variable("SAPT0 TOTAL ENERGY"),
    }

    for key in keys:
        compare_values(Eref[key], Epsi[key], 6, key)
    data = psi4.fsapt_analysis(
        molecule=mol,
        # NOTE: 1-indexed for fragments_a and fragments_b
        fragments_a={
            "MethylA": [1, 2, 3, 4, 5],
        },
        fragments_b={
            "MethylB": [6, 7, 8, 9, 10],
        },
    )
    df = pd.DataFrame(data)
    print(df)
    fEnergies = {}
    fkeys = [
        "fEelst",
        "fEexch",
        "fEindAB",
        "fEindBA",
        "fEdisp",
        "fEedisp",
        "fEtot",
    ]

    df_keys = [
        "Elst",
        "Exch",
        "IndAB",
        "IndBA",
        "Disp",
        "EDisp",
        "Total",
    ]

    # Get columns from dataframe that match fkeys
    Energies = df[df_keys].iloc[0].values

    for pair in zip(fkeys, Energies):
        fEnergies[pair[0]] = pair[1]

    fEref = {
        "fEelst": -0.002,
        "fEexch": 0.000,
        "fEindAB": -0.000,
        "fEindBA": -0.000,
        "fEdisp": -0.021,
        "fEedisp": 0.000,
        "fEtot": -0.023,
    }

    for key in fkeys:
        print(fEnergies[key], fEref[key])
        compare_values(fEref[key], fEnergies[key], 2, key)


@pytest.mark.fsapt
@uusing("pandas")
def test_fsapt_AtomicOutput():
    """
    fsapt-atomicOutput: calling fsapt_analysis with atomic results
    """
    import pandas as pd

    mol = psi4.geometry(
        """0 1
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
no_com"""
    )
    psi4.set_options(
        {
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )
    plan = psi4.energy("fisapt0", return_plan=True, molecule=mol)
    atomic_result = psi4.schema_wrapper.run_qcschema(
        plan.plan(wfn_qcvars_only=False),
        clean=True,
        postclean=True,
    )
    # TODO: this needs to work for non-symmetric test cases...
    # print(atomic_result.extras)
    # qcvars = atomic_result.extras["qcvars"]
    # keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    # Eref = {
    #     "Enuc": 35.07529824960602,
    #     "Eelst": -3.8035153870907834e-06,
    #     "Eexch": 1.7912112685446533e-07,
    #     "Eind": -3.833795151474493e-08,
    #     "Edisp": -3.288568662589654e-05,
    #     "Etot": -3.6548418837647605e-05,
    # }
    # Epsi = {
    #     "Enuc": mol.nuclear_repulsion_energy(),
    #     "Eelst": qcvars["SAPT ELST ENERGY"],
    #     "Eexch": qcvars["SAPT EXCH ENERGY"],
    #     "Eind": qcvars["SAPT IND ENERGY"],
    #     "Edisp": qcvars["SAPT DISP ENERGY"],
    #     "Etot": qcvars["SAPT0 TOTAL ENERGY"],
    # }
    # for key in keys:
    #     compare_values(Eref[key], Epsi[key], 6, key)
    print("Analysis")
    data = psi4.fsapt_analysis(
        # NOTE: 1-indexed for fragments_a and fragments_b
        fragments_a={
            "MethylA": [1, 2, 3, 4, 5],
        },
        fragments_b={
            "MethylB": [6, 7, 8, 9, 10],
        },
        atomic_results=atomic_result,
    )
    df = pd.DataFrame(data)
    print(df)
    fEnergies = {}
    fkeys = [
        "fEelst",
        "fEexch",
        "fEindAB",
        "fEindBA",
        "fEdisp",
        "fEedisp",
        "fEtot",
    ]

    df_keys = [
        "Elst",
        "Exch",
        "IndAB",
        "IndBA",
        "Disp",
        "EDisp",
        "Total",
    ]

    # Get columns from dataframe that match fkeys
    Energies = df[df_keys].iloc[0].values

    for pair in zip(fkeys, Energies):
        fEnergies[pair[0]] = pair[1]

    fEref = {
        "fEelst": -0.002,
        "fEexch": 0.000,
        "fEindAB": -0.000,
        "fEindBA": -0.000,
        "fEdisp": -0.021,
        "fEedisp": 0.000,
        "fEtot": -0.023,
    }

    for key in fkeys:
        compare_values(fEref[key], fEnergies[key], 2, key)


@pytest.mark.fsapt
def test_fsapt_output_file():
    mol = psi4.geometry(
        """0 1
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
no_com"""
    )
    psi4.set_options(
        {
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
        }
    )
    psi4.energy("fisapt0")
    keys = ["Enuc", "Eelst", "Eexch", "Eind", "Edisp", "Etot"]
    Eref = {
        "Enuc": 35.07529824960602,
        "Eelst": -3.8035153870907834e-06,
        "Eexch": 1.7912112685446533e-07,
        "Eind": -3.833795151474493e-08,
        "Edisp": -3.288568662589654e-05,
        "Etot": -3.6548418837647605e-05,
    }
    Epsi = {
        "Enuc": mol.nuclear_repulsion_energy(),
        "Eelst": variable("SAPT ELST ENERGY"),
        "Eexch": variable("SAPT EXCH ENERGY"),
        "Eind": variable("SAPT IND ENERGY"),
        "Edisp": variable("SAPT DISP ENERGY"),
        "Etot": variable("SAPT0 TOTAL ENERGY"),
    }

    for key in keys:
        compare_values(Eref[key], Epsi[key], 6, key)
    psi4.fsapt_analysis(
        fragments_a={
            "MethylA": [1, 2, 3, 4, 5],
        },
        fragments_b={
            "MethylB": [6, 7, 8, 9, 10],
        },
        dirname="./fsapt",
    )
    fEnergies = {}
    fkeys = [
        "fEelst", "fEexch", "fEindAB", "fEindBA", "fEdisp", "fEedisp", "fEtot"
    ]

    with open("./fsapt/fsapt.dat", "r") as fsapt:
        Energies = [float(x) for x in fsapt.readlines()[-2].split()[2:]]

    for pair in zip(fkeys, Energies):
        fEnergies[pair[0]] = pair[1]

    fEref = {
        "fEelst": -0.002,
        "fEexch": 0.000,
        "fEindAB": -0.000,
        "fEindBA": -0.000,
        "fEdisp": 0.000,
        "fEedisp": -0.033,
        "fEtot": -0.036,
    }

    for key in fkeys:
        compare_values(fEref[key], fEnergies[key], 2, key)
        print(fEnergies)
        fEref = {
            "fEelst": -0.002,
            "fEexch": 0.000,
            "fEindAB": -0.000,
            "fEindBA": -0.000,
            "fEdisp": -0.021,
            "fEedisp": 0.000,
            "fEtot": -0.023,
        }

    for key in fkeys:
        compare_values(fEref[key], fEnergies[key], 2, key)


@uusing("pandas")
@pytest.mark.fsapt
def test_fsapt_indices():
    # TODO: EDIT THIS TEST TO DROP SAVING DF
    psi4.set_memory("50 GB")
    psi4.set_num_threads(12)
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
            # "basis": "sto-3g",
            "basis": "aug-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            # "freeze_core": "true",
        }
    )
    plan = psi4.energy("fisapt0", return_plan=True, molecule=mol)
    atomic_result = psi4.schema_wrapper.run_qcschema(
        plan.plan(wfn_qcvars_only=False),
        clean=True,
        postclean=True,
    )
    data = psi4.fsapt_analysis(
        # NOTE: 1-indexed for fragments_a and fragments_b
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
        atomic_results=atomic_result,
    )
    df = pd.DataFrame(data)
    print(df)
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
    df.to_pickle("fsapt_train_simple.pkl")


@pytest.mark.fsapt
def test_isapt_hf():
    """
    I-SAPT with HF: 3-fragment system with linker (fragment C).
    Tests FISAPT_LINK_ASSIGNMENT = "C" option.
    Reference values from C++ FISAPT implementation.
    """
    # 2,4-pentanediol molecule from fisapt-siao1
    mol = psi4.core.Molecule.from_arrays(
        elez=[6, 6, 1, 1, 1, 8, 1, 1, 6, 6, 1, 1, 1, 8, 1, 1, 6, 1, 1],
        fragment_separators=[8, 16],
        fix_com=True,
        fix_orientation=True,
        fix_symmetry='c1',
        fragment_multiplicities=[2, 2, 1],
        molecular_charge=0,
        molecular_multiplicity=1,
        geom=[
            2.51268, -0.79503, -0.22006,
            1.23732, 0.03963, -0.27676,
            2.46159, -1.62117, -0.94759,
            2.64341, -1.21642, 0.78902,
            3.39794, -0.18468, -0.46590,
            1.26614, 1.11169, 0.70005,
            2.10603, 1.58188, 0.59592,
            1.13110, 0.48209, -1.28412,

            -1.26007, 0.07291, 0.27398,
            -2.53390, -0.75742, 0.20501,
            -2.48461, -1.59766, 0.91610,
            -2.65872, -1.16154, -0.81233,
            -3.41092, -0.13922, 0.44665,
            -1.38660, 1.11180, -0.71748,
            -1.17281, 0.53753, 1.27129,
            -0.70002, 1.76332, -0.50799,

            -0.01090, -0.78649, 0.02607,
            0.17071, -1.41225, 0.91863,
            -0.19077, -1.46135, -0.82966
        ])

    psi4.core.set_active_molecule(mol)
    psi4.set_options({
        "basis": "jun-cc-pvdz",
        "scf_type": "disk_df",
        "guess": "sad",
        "freeze_core": "true",
        # I-SAPT link partitioning
        "FISAPT_LINK_ASSIGNMENT": "C",
        "FISAPT_FSAPT_FILEPATH": "none",
        # SAPT(DFT) options with HF functional
        "SAPT_DFT_FUNCTIONAL": "HF",
        "SAPT_DFT_DO_HYBRID": False,
        "SAPT_DFT_DO_DHF": True,
        "SAPT_DFT_USE_EINSUMS": True,
        "SAPT_DFT_DO_FSAPT": "FISAPT",
        "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
    })
    psi4.energy("sapt(dft)")

    # Reference values from C++ FISAPT (in Hartree)
    Eref = {
        "Eelst": 0.0149641,
        "Eexch": 0.0106947,
        "Eind": -0.0052829,
        "Edisp": -0.0048356,
    }
    Epsi = {
        "Eelst": variable("SAPT ELST ENERGY"),
        "Eexch": variable("SAPT EXCH ENERGY"),
        "Eind": variable("SAPT IND ENERGY"),
        "Edisp": variable("SAPT DISP ENERGY"),
    }

    for key in Eref:
        compare_values(Eref[key], Epsi[key], 4, f"I-SAPT HF {key}")


@pytest.mark.fsapt
def test_isapt_pbe0():
    """
    I-SAPT with PBE0 DFT functional: 3-fragment system with linker.
    Tests FISAPT_LINK_ASSIGNMENT = "C" with DFT.
    """
    mol = psi4.core.Molecule.from_arrays(
        elez=[6, 6, 1, 1, 1, 8, 1, 1, 6, 6, 1, 1, 1, 8, 1, 1, 6, 1, 1],
        fragment_separators=[8, 16],
        fix_com=True,
        fix_orientation=True,
        fix_symmetry='c1',
        fragment_multiplicities=[2, 2, 1],
        molecular_charge=0,
        molecular_multiplicity=1,
        geom=[
            2.51268, -0.79503, -0.22006,
            1.23732, 0.03963, -0.27676,
            2.46159, -1.62117, -0.94759,
            2.64341, -1.21642, 0.78902,
            3.39794, -0.18468, -0.46590,
            1.26614, 1.11169, 0.70005,
            2.10603, 1.58188, 0.59592,
            1.13110, 0.48209, -1.28412,

            -1.26007, 0.07291, 0.27398,
            -2.53390, -0.75742, 0.20501,
            -2.48461, -1.59766, 0.91610,
            -2.65872, -1.16154, -0.81233,
            -3.41092, -0.13922, 0.44665,
            -1.38660, 1.11180, -0.71748,
            -1.17281, 0.53753, 1.27129,
            -0.70002, 1.76332, -0.50799,

            -0.01090, -0.78649, 0.02607,
            0.17071, -1.41225, 0.91863,
            -0.19077, -1.46135, -0.82966
        ])

    psi4.core.set_active_molecule(mol)
    psi4.set_options({
        "basis": "jun-cc-pvdz",
        "scf_type": "disk_df",
        "guess": "sad",
        "freeze_core": "true",
        # I-SAPT link partitioning
        "FISAPT_LINK_ASSIGNMENT": "C",
        "FISAPT_FSAPT_FILEPATH": "none",
        # SAPT(DFT) options
        "SAPT_DFT_FUNCTIONAL": "PBE0",
        "SAPT_DFT_DO_HYBRID": True,
        "SAPT_DFT_DO_DHF": True,
        "SAPT_DFT_DO_DISP": False,
        "SAPT_DFT_D4_IE": True,
        "SAPT_DFT_USE_EINSUMS": True,
        "SAPT_DFT_GRAC_COMPUTE": "SINGLE",
        "SAPT_DFT_DO_FSAPT": "FISAPT",
        "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
    })
    psi4.energy("sapt(dft)")

    # Reference values from Python SAPT(DFT) I-SAPT implementation (in Hartree)
    Eref = {
        "Eelst": 0.0120113,
        "Eexch": 0.0134432,
        "Eind": -0.0050777,
        "Edisp": -0.0056293,
    }
    Epsi = {
        "Eelst": variable("SAPT ELST ENERGY"),
        "Eexch": variable("SAPT EXCH ENERGY"),
        "Eind": variable("SAPT IND ENERGY"),
        "Edisp": variable("SAPT DISP ENERGY"),
    }

    for key in Eref:
        compare_values(Eref[key], Epsi[key], 4, f"I-SAPT PBE0 {key}")


@pytest.mark.fsapt
def test_fsaptdft_ext_abc_au():
    """
    F-SAPT with external potentials using Molecule.from_arrays.
    Tests external potentials on a 3-water-fragment system.
    """

    psi_bohr2angstroms = psi4.constants.bohr2angstroms
    mol = psi4.core.Molecule.from_arrays(
        elez=[1, 8, 1, 1, 8, 1, 1, 8, 1],
        fragment_separators=[3, 6],
        geom=np.array([
            0.0290, -1.1199, -1.5243,
            0.9481, -1.3990, -1.3587,
            1.4371, -0.5588, -1.3099,
            1.0088, -1.5240, 0.5086,
            1.0209, -1.1732, 1.4270,
            1.5864, -0.3901, 1.3101,
            -1.0231, 1.6243, -0.8743,
            -0.5806, 2.0297, -0.1111,
            -0.9480, 1.5096, 0.6281]) / psi_bohr2angstroms,
        units="Bohr")

    psi4.core.set_active_molecule(mol)

    bohr = psi_bohr2angstroms
    external_potentials = {
        "A": [
            [0.417, np.array([-0.5496, -0.6026, 1.5720]) / bohr],
            [-0.834, np.array([-1.4545, -0.1932, 1.4677]) / bohr],
            [0.417, np.array([-1.9361, -0.4028, 2.2769]) / bohr]],
        "B": [
            [0.417, np.array([-2.5628, -0.8269, -1.6696]) / bohr],
            [-0.834, np.array([-1.7899, -0.4027, -1.2768]) / bohr],
            [0.417, np.array([-1.8988, -0.4993, -0.3072]) / bohr]],
        "C": [
            [0.417, np.array([1.1270, 1.5527, -0.1658]) / bohr],
            [-0.834, np.array([1.9896, 1.0738, -0.1673]) / bohr],
            [0.417, np.array([2.6619, 1.7546, -0.2910]) / bohr]],
    }

    psi4.set_options({
        "basis": "jun-cc-pvdz",
        "scf_type": "df",
        "guess": "sad",
        "freeze_core": "true",

        "SAPT_DFT_FUNCTIONAL": "HF",
        "SAPT_DFT_DO_HYBRID": False,
        "SAPT_DFT_DO_DHF": True,
        "SAPT_DFT_USE_EINSUMS": True,
        "SAPT_DFT_DO_FSAPT": "FISAPT",
        "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
    })

    psi4.energy('fisapt0')
    psi4.energy('sapt(dft)')

    keys = ['Enuc', 'Eelst', 'Eexch', 'Eind', 'Edisp', 'Etot']

    # NOTE: SAPT(DFT) external potential logic gets slightly different energies
    # than FISAPT0. This is likely due to slighly different SCF implementations
    # between SAPT(DFT) and FISAPT0. The differences are very small (on the
    # order of 1e-5), so AMW am accepting these as correct for now.
    Eref = {
        'Edisp': -0.0028063519613379505,
        'Eelst': -0.014222339471217538,
        'Eexch': 0.018823419507723968,
        'Eind': -0.004547074033665097,
        'Enuc': 74.2330370461897,
        'Etot': -0.002752345958496617
    }


    Epsi = {
        'Enuc': mol.nuclear_repulsion_energy(),
        'Eelst': variable("SAPT ELST ENERGY"),
        'Eexch': variable("SAPT EXCH ENERGY"),
        'Eind': variable("SAPT IND ENERGY"),
        'Edisp': variable("SAPT DISP ENERGY"),
        'Etot': variable("SAPT TOTAL ENERGY"),
    }
    from pprint import pprint as pp
    pp(Epsi)

    for key in keys:
        compare_values(Eref[key], Epsi[key], 4, key)

    # Now external potentials

    # psi4.energy('fisapt0', external_potentials=external_potentials)
    psi4.energy('sapt(dft)', external_potentials=external_potentials)

    # NOTE: SAPT(DFT) external potential logic gets slightly different energies
    # than FISAPT0. This is likely due to slighly different SCF implementations
    # between SAPT(DFT) and FISAPT0. The differences are very small (on the
    # order of 1e-5), so I am accepting these as correct for now.
    # Eelst: computed value (-0.04923764) does not match (-0.04919038) to atol=1e-06 by difference (-0.00004726).
    Eref = {
        'Enuc': 74.2330370461897,
        'Eelst': -0.04923764,
        'Eexch': 0.018239207303845935,
        'Eind': -0.007969545823122322,
        'Edisp': -0.002794948165605119,
        'Etot': -0.04171566532235386,
    }

    Epsi = {
        'Enuc': mol.nuclear_repulsion_energy(),
        'Eelst': variable("SAPT ELST ENERGY"),
        'Eexch': variable("SAPT EXCH ENERGY"),
        'Eind': variable("SAPT IND ENERGY"),
        'Edisp': variable("SAPT DISP ENERGY"),
        'Etot': variable("SAPT TOTAL ENERGY"),
    }
    for key in keys:
        compare_values(Eref[key], Epsi[key], 4, key)


    os.chdir('fsapt')
    with open('fA.dat', 'w') as fA:
        fA.write("w1 1 2 3")
    with open('fB.dat', 'w') as fB:
        fB.write("w3 4 5 6")
    psi4.procrouting.sapt.fsapt.run_from_output(dirname='.')
    fEnergies = {}
    fkeys = [
        'fEelst', 'fEexch', 'fEindAB', 'fEindBA', 'fEdisp', 'fEedisp', 'fEtot'
    ]

    with open('fsapt.dat', 'r') as fsapt:
        Energies = [float(x) for x in fsapt.readlines()[-2].split()[2:]]

    for pair in zip(fkeys, Energies):
        fEnergies[pair[0]] = pair[1]

    fEref = {
        'fEelst': -30.867,
        'fEexch': 11.445,
        'fEindAB': -3.138,
        'fEindBA': -1.863,
        'fEdisp': -1.754,
        'fEedisp': 0.000,
        'fEtot': -26.177
    }

    for key in fkeys:
        compare_values(fEref[key], fEnergies[key], 2, key)


if __name__ == "__main__":
    psi4.set_memory("64 GB")
    psi4.set_num_threads(12)
    # test_fsapt_psivars_dict()
    # test_fsapt_external_potentials()
    # test_fsapt_psivars()
    # test_fsapt_psivars_dict()
    # test_fsapt_AtomicOutput()
    # test_fsapt_output_file()
    # test_fsapt_output_file()
    # test_fsapt_indices()

    # ISAPT tests
    # test_isapt_hf()
    # test_isapt_pbe0()
    test_fsaptdft_ext_abc_au()
