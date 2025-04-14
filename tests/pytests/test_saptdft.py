import pytest
import psi4
from qcelemental import constants
from psi4 import compare_values
from psi4 import core
import numpy as np
import qcelemental as qcel
from pprint import pprint as pp

hartree_to_kcalmol = constants.conversion_factor("hartree", "kcal/mol")
pytestmark = [pytest.mark.psi, pytest.mark.api]


@pytest.mark.saptdft
def test_sapt_dft_compute_ddft_d4():
    """
    Test SAPT(DFT) for correct delta-DFT and -D4 IE terms
    """
    mol_dimer = psi4.geometry(
        """
  O -2.930978458   -0.216411437    0.000000000
  H -3.655219777    1.440921844    0.000000000
  H -1.133225297    0.076934530    0.000000000
   --
  O  2.552311356    0.210645882    0.000000000
  H  3.175492012   -0.706268134   -1.433472544
  H  3.175492012   -0.706268134    1.433472544
  units bohr
"""
    )
    dft_functional = "pbe0"
    psi4.set_options(
        {
            "basis": "STO-3G",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "sapt_dft_grac_shift_a": 0.136,
            "sapt_dft_grac_shift_b": 0.136,
            "SAPT_DFT_FUNCTIONAL": dft_functional,
            "SAPT_DFT_DO_DDFT": True,
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": True,
        }
    )
    dft_IE = (
        psi4.energy(dft_functional, bsse_type="CP", molecule=mol_dimer)
        * hartree_to_kcalmol
    )
    # many-body energies
    # dimer    :-150.5070040376297698
    # monomer A: -75.2534797048230359
    # monomer B: -75.2469427300573841

    psi4.energy("SAPT(DFT)")
    # sapt(dft) DFT energies
    # dimer    :-150.5070038696833024
    # monomer A: -75.2469415558837937
    # monomer B: -75.2534791716759486

    ELST = psi4.core.variable("SAPT ELST ENERGY")
    EXCH = psi4.core.variable("SAPT EXCH ENERGY")
    IND = psi4.core.variable("SAPT IND ENERGY")
    DFT_MONA = psi4.core.variable("DFT MONOMER A ENERGY")
    DFT_MONB = psi4.core.variable("DFT MONOMER B ENERGY")
    # DFT_MONA = psi4.core.variable("DFT MONOMERA")
    # DFT_MONB = psi4.core.variable("DFT MONOMERB")
    DFT_DIMER = psi4.core.variable("DFT DIMER ENERGY")
    # DFT_DIMER = psi4.core.variable("DFT DIMER")

    print(f"{DFT_DIMER = }\n{DFT_MONA = }\n{DFT_MONB = }")
    DFT_IE = (DFT_DIMER - DFT_MONA - DFT_MONB) * hartree_to_kcalmol
    print(f"bsse: {dft_IE = }")
    print(f"SAPT: {DFT_IE = }")
    assert compare_values(dft_IE, DFT_IE, 7, "DFT IE")
    assert compare_values(-4.130018077857232, DFT_IE, 7, "DFT IE")

    # get dft_IE from SAPT(DFT) Delta DFT term to back-calculate
    ELST = psi4.core.variable("SAPT ELST ENERGY")
    EXCH = psi4.core.variable("SAPT EXCH ENERGY")
    IND = psi4.core.variable("SAPT IND ENERGY")
    DELTA_HF = psi4.core.variable("SAPT(DFT) DELTA HF")
    DDFT = psi4.core.variable("SAPT(DFT) DELTA DFT")
    DFT_IE_from_dDFT = (DDFT + ELST + EXCH + IND - DELTA_HF) * hartree_to_kcalmol
    assert compare_values(DFT_IE_from_dDFT, DFT_IE, 7, "DFT IE")
    assert compare_values(-4.130018048599092, DFT_IE, 7, "DFT IE")
    print(f"dDFT: {DFT_IE_from_dDFT = }")

@pytest.mark.saptdft
def test_sapt_dft_compute_ddft_d4_diskdf():
    """
    Test SAPT(DFT) for correct delta-DFT and -D4 IE terms
    """
    mol_dimer = psi4.geometry(
        """
  O -2.930978458   -0.216411437    0.000000000
  H -3.655219777    1.440921844    0.000000000
  H -1.133225297    0.076934530    0.000000000
   --
  O  2.552311356    0.210645882    0.000000000
  H  3.175492012   -0.706268134   -1.433472544
  H  3.175492012   -0.706268134    1.433472544
  units bohr
"""
    )
    # psi4.set_memory("280 MB")
    psi4.set_memory("64 GB")
    dft_functional = "pbe0"
    psi4.set_options(
        {
            "basis": "STO-3G",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "scf_type": "disk_df",
            "sapt_dft_grac_shift_a": 0.136,
            "sapt_dft_grac_shift_b": 0.136,
            "SAPT_DFT_FUNCTIONAL": dft_functional,
            "SAPT_DFT_DO_DDFT": True,
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": True,
        }
    )
    dft_IE = (
        psi4.energy(dft_functional, bsse_type="CP", molecule=mol_dimer)
        * hartree_to_kcalmol
    )
    # many-body energies
    # dimer    :-150.5070040376297698
    # monomer A: -75.2534797048230359
    # monomer B: -75.2469427300573841

    psi4.energy("SAPT(DFT)")
    # sapt(dft) DFT energies
    # dimer    :-150.5070038696833024
    # monomer A: -75.2469415558837937
    # monomer B: -75.2534791716759486

    ELST = psi4.core.variable("SAPT ELST ENERGY")
    EXCH = psi4.core.variable("SAPT EXCH ENERGY")
    IND = psi4.core.variable("SAPT IND ENERGY")
    DFT_MONA = psi4.core.variable("DFT MONOMER A ENERGY")
    DFT_MONB = psi4.core.variable("DFT MONOMER B ENERGY")
    # DFT_MONA = psi4.core.variable("DFT MONOMERA")
    # DFT_MONB = psi4.core.variable("DFT MONOMERB")
    DFT_DIMER = psi4.core.variable("DFT DIMER ENERGY")
    # DFT_DIMER = psi4.core.variable("DFT DIMER")

    print(f"{DFT_DIMER = }\n{DFT_MONA = }\n{DFT_MONB = }")
    DFT_IE = (DFT_DIMER - DFT_MONA - DFT_MONB) * hartree_to_kcalmol
    print(f"bsse: {dft_IE = }")
    print(f"SAPT: {DFT_IE = }")
    assert compare_values(dft_IE, DFT_IE, 7, "DFT IE")
    assert compare_values(-4.130018077857232, DFT_IE, 7, "DFT IE")

    # get dft_IE from SAPT(DFT) Delta DFT term to back-calculate
    ELST = psi4.core.variable("SAPT ELST ENERGY")
    EXCH = psi4.core.variable("SAPT EXCH ENERGY")
    IND = psi4.core.variable("SAPT IND ENERGY")
    DELTA_HF = psi4.core.variable("SAPT(DFT) DELTA HF")
    DDFT = psi4.core.variable("SAPT(DFT) DELTA DFT")
    DFT_IE_from_dDFT = (DDFT + ELST + EXCH + IND - DELTA_HF) * hartree_to_kcalmol
    assert compare_values(DFT_IE_from_dDFT, DFT_IE, 7, "DFT IE")
    assert compare_values(-4.130018048599092, DFT_IE, 7, "DFT IE")
    print(f"dDFT: {DFT_IE_from_dDFT = }")


@pytest.mark.saptdft
def test_sapt_dft_compute_ddft_d4_auto_grac():
    """
    Test SAPT(DFT) for correct delta-DFT and -D4 IE terms
    """
    mol_dimer = psi4.geometry(
        """
  O -2.930978458   -0.216411437    0.000000000
  H -3.655219777    1.440921844    0.000000000
  H -1.133225297    0.076934530    0.000000000
   --
  O  2.552311356    0.210645882    0.000000000
  H  3.175492012   -0.706268134   -1.433472544
  H  3.175492012   -0.706268134    1.433472544
  units bohr
"""
    )
    dft_functional = "pbe0"
    psi4.set_options(
        {
            "basis": "STO-3G",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "SAPT_DFT_FUNCTIONAL": dft_functional,
            "SAPT_DFT_DO_DDFT": True,
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": True,
            "SAPT_DFT_GRAC_COMPUTE": "SINGLE",
        }
    )
    dft_IE = (
        psi4.energy(dft_functional, bsse_type="CP", molecule=mol_dimer)
        * hartree_to_kcalmol
    )
    psi4.energy("SAPT(DFT)")
    ELST = psi4.core.variable("SAPT ELST ENERGY")
    EXCH = psi4.core.variable("SAPT EXCH ENERGY")
    IND = psi4.core.variable("SAPT IND ENERGY")
    DFT_MONA = psi4.core.variable("DFT MONOMER A ENERGY")
    DFT_MONB = psi4.core.variable("DFT MONOMER B ENERGY")
    DFT_DIMER = psi4.core.variable("DFT DIMER ENERGY")

    print(f"{DFT_DIMER = }\n{DFT_MONA = }\n{DFT_MONB = }")
    DFT_IE = (DFT_DIMER - DFT_MONA - DFT_MONB) * hartree_to_kcalmol
    print(f"bsse: {dft_IE = }")
    print(f"SAPT: {DFT_IE = }")
    assert compare_values(dft_IE, DFT_IE, 7, "DFT IE")
    assert compare_values(-4.130018077857232, DFT_IE, 7, "DFT IE")

    # get dft_IE from SAPT(DFT) Delta DFT term to back-calculate
    ELST = psi4.core.variable("SAPT ELST ENERGY")
    EXCH = psi4.core.variable("SAPT EXCH ENERGY")
    IND = psi4.core.variable("SAPT IND ENERGY")
    DELTA_HF = psi4.core.variable("SAPT(DFT) DELTA HF")
    DDFT = psi4.core.variable("SAPT(DFT) DELTA DFT")
    DFT_IE_from_dDFT = (DDFT + ELST + EXCH + IND - DELTA_HF) * hartree_to_kcalmol
    assert compare_values(DFT_IE_from_dDFT, DFT_IE, 7, "DFT IE")
    assert compare_values(-4.130018048599092, DFT_IE, 7, "DFT IE")
    print(f"dDFT: {DFT_IE_from_dDFT = }")
    compare_values(
        #  STO-3G target
        0.1981702737,
        # aug-cc-pvdz target, 0.1307 (using experimental IP from CCCBDB)
        # 0.13053068183319516,
        psi4.core.variable("SAPT_DFT_GRAC_SHIFT_A"),
        8,
        "SAPT_DFT_GRAC_SHIFT_A",
    )
    compare_values(
        #  STO-3G target
        0.1983742234,
        # aug-cc-pvdz target, 0.1307 (using experimental IP from CCCBDB)
        # 0.13063798506967816,
        psi4.core.variable("SAPT_DFT_GRAC_SHIFT_B"),
        8,
        "SAPT_DFT_GRAC_SHIFT_B",
    )
    print(f"{psi4.core.variable('SAPT_DFT_GRAC_SHIFT_A') = }")
    print(f"{psi4.core.variable('SAPT_DFT_GRAC_SHIFT_B') = }")


@pytest.mark.saptdft
def test_dftd4():
    """
    Tests SAPT(DFT) module for computing DFT-D4 SAPT-decomposition of energy terms
    """
    mol_dimer = psi4.geometry(
        """
  O -2.930978458   -0.216411437    0.000000000
  H -3.655219777    1.440921844    0.000000000
  H -1.133225297    0.076934530    0.000000000
   --
  O  2.552311356    0.210645882    0.000000000
  H  3.175492012   -0.706268134   -1.433472544
  H  3.175492012   -0.706268134    1.433472544
  units bohr
"""
    )
    psi4.set_options(
        {
            "basis": "STO-3G",
            "sapt_dft_grac_shift_a": 0.136,
            "sapt_dft_grac_shift_b": 0.136,
            "SAPT_DFT_FUNCTIONAL": "pbe0",
            "SAPT_DFT_DO_DDFT": True,
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_DO_DISP": False,
        }
    )
    psi4.energy("SAPT(DFT)", molecule=mol_dimer)
    DISP = psi4.core.variable("SAPT DISP ENERGY")
    assert compare_values(-0.005731715146359108, DISP, 8, "DFT-D4 DISP")

@pytest.mark.saptdft
@pytest.mark.parametrize(
    "SAPT_DFT_GRAC_COMPUTE, refA, refB, gracA, gracB, geometry",
    [
        (
            "SINGLE",
            0.19807358,
            0.19830016,
            0.0,
            0.0,
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
""",
        ),
        (
            "SINGLE",
            0.1307,
            0.19830016,
            0.1307,
            0.0,
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
""",
        ),
        (
            "ITERATIVE",
            0.3258340368,
            0.19830016,
            0.0,
            0.0,
            """
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
        ),
    ],
)
def test_saptdft_auto_grac(SAPT_DFT_GRAC_COMPUTE, refA, refB, gracA, gracB, geometry):
    """
    For SAPT(DFT), one must compute a GRAC shift for each monomer. Ideally,
    this GRAC shift should be close to the experimental Ionization Potential
    (IP) of the monomer. Basis set incompleteness prevents this here.
    e.g., aug-DZ H2O has a shift of 0.1306, compared to 0.1307 experimental IP.
    """
    mol_dimer = psi4.geometry(geometry)
    psi4.set_options(
        {
            "basis": "STO-3G",
            "SAPT_DFT_FUNCTIONAL": "pbe0",
            "SAPT_DFT_GRAC_COMPUTE": SAPT_DFT_GRAC_COMPUTE,
        }
    )
    if gracA != 0.0:
        psi4.set_options({"SAPT_DFT_GRAC_SHIFT_A": gracA})
    if gracB != 0.0:
        psi4.set_options({"SAPT_DFT_GRAC_SHIFT_B": gracB})
    psi4.energy("SAPT(DFT)", molecule=mol_dimer)
    compare_values(
        refA,
        psi4.core.variable("SAPT_DFT_GRAC_SHIFT_A"),
        8,
        "SAPT_DFT_GRAC_SHIFT_A",
    )
    compare_values(
        refB,
        psi4.core.variable("SAPT_DFT_GRAC_SHIFT_B"),
        8,
        "SAPT_DFT_GRAC_SHIFT_B",
    )
    return


@pytest.mark.extern
@pytest.mark.saptdft
@pytest.mark.parametrize(
    "test_id, external_pot_keys, energy_method, expected_values, precision",
    [
        (
            "abc",
            ["A", "B", "C"],
            "sapt(dft)",
            {
                "Edisp": -0.002778631330469043,
                "Eelst": -0.04933182514082859,
                "Eexch": 0.01826072035756901,
                "Eind": -0.00783603487368914,
                "Enuc": 37.565065473343004,
                "Etot": -0.04168577098741776,
            },
            8,
        ),
        (
            "abc",
            ["A", "B", "C"],
            "fisapt0",
            {
                "Edisp": -0.002778631330469043,
                "Eelst": -0.04933182514082859,
                "Eexch": 0.01826072035756901,
                "Eind": -0.00783603487368914,
                "Enuc": 37.565065473343004,
                "Etot": -0.04168577098741776,
            },
            8,
        ),
        (
            "ab",
            ["A", "B"],
            "sapt(dft)",
            {
                "Eexch": 0.018716989207357836,
                "Edisp": -0.002806890580921549,
                "Eelst": -0.049844754081429986,
                "Eind": -0.007796394758818957,
                "Enuc": 37.565065473343004,
                "Etot": -0.041731050213812654,
            },
            7,
        ),
        (
            "a",
            ["A"],
            "sapt(dft)",
            {
                "Edisp": -0.002816214841039168,
                "Eelst": -0.029279397864449663,
                "Eexch": 0.01889146242465646,
                "Eind": -0.006356586601442907,
                "Enuc": 37.565065473343004,
                "Etot": -0.019560736882275276,
            },
            8,
        ),
        (
            "b",
            ["B"],
            "sapt(dft)",
            {
                "Edisp": -0.0027835628235023946,
                "Eelst": -0.024996838889304485,
                "Eexch": 0.018658252478543982,
                "Eind": -0.005940905006556488,
                "Enuc": 37.565065473343004,
                "Etot": -0.015063054240819385,
            },
            7,
        ),
        (
            "c",
            ["C"],
            "sapt(dft)",
            {
                "Edisp": -0.0027641749042896808,
                "Eelst": -0.013585309461394246,
                "Eexch": 0.01839802168509688,
                "Eind": -0.004537242936115817,
                "Enuc": 37.565065473343004,
                "Etot": -0.002488705616702865,
            },
            8,
        ),
    ],
)
def test_sapthf_external_potential(
    test_id, external_pot_keys, energy_method, expected_values, precision
):
    """
    Parameterized test for SAPT with external potentials.

    Parameters:
    -----------
    test_id : str
        Identifier for the test case
    external_pot_keys : list
        List of keys from the full external_potentials dict to include in this test
    energy_method : str
        Energy method to use (sapt(dft) or fisapt0)
    expected_values : dict
        Expected energy values to compare against
    precision : int
        Decimal precision for comparison
    """
    # Define the molecule geometry (common to all tests)
    mol = psi4.geometry(
        """
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
        """
    )

    # All potential definitions
    psi_bohr2angstroms = qcel.constants.bohr2angstroms
    all_external_potentials = {
        "A": [
            [0.417, np.array([-0.5496, -0.6026, 1.5720]) / psi_bohr2angstroms],
            [-0.834, np.array([-1.4545, -0.1932, 1.4677]) /
             psi_bohr2angstroms],
            [0.417, np.array([-1.9361, -0.4028, 2.2769]) / psi_bohr2angstroms],
        ],
        "B": [
            [0.417, np.array([-2.5628, -0.8269, -1.6696]) /
             psi_bohr2angstroms],
            [-0.834, np.array([-1.7899, -0.4027, -1.2768]) /
             psi_bohr2angstroms],
            [0.417, np.array([-1.8988, -0.4993, -0.3072]) /
             psi_bohr2angstroms],
        ],
        "C": [
            [0.417, np.array([1.1270, 1.5527, -0.1658]) / psi_bohr2angstroms],
            [-0.834, np.array([1.9896, 1.0738, -0.1673]) / psi_bohr2angstroms],
            [0.417, np.array([2.6619, 1.7546, -0.2910]) / psi_bohr2angstroms],
        ],
    }

    # Select only the potentials needed for this test
    external_potentials = {
        key: all_external_potentials[key] for key in external_pot_keys
    }

    # Set common options
    psi4.set_options(
        {
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "SAPT_DFT_FUNCTIONAL": "hf",
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
        }
    )

    # Run the energy calculation
    psi4.energy(
        energy_method,
        external_potentials=external_potentials,
        molecule=mol,
    )

    # Print reference values
    print(f"TEST ID: {test_id}")
    print("REF:")
    import pprint

    pprint.pprint(expected_values)

    # Collect calculated values
    key_labels = [
        ["Eexch", "SAPT EXCH ENERGY"],
        ["Edisp", "SAPT DISP ENERGY"],
        ["Eelst", "SAPT ELST ENERGY"],
        ["Eind", "SAPT IND ENERGY"],
        ["Etot", "SAPT TOTAL ENERGY"],
    ]
    calculated_energies = {k1: psi4.core.variable(k2) for k1, k2 in key_labels}
    calculated_energies["Enuc"] = mol.nuclear_repulsion_energy()

    # Print calculated values
    print("CALC:")
    pprint.pprint(calculated_energies)

    # Compare values
    for k1, k2 in key_labels:
        compare_values(
            expected_values[k1],
            calculated_energies[k1],
            precision,
            k1,
        )

    # Also check nuclear repulsion energy
    compare_values(
        expected_values["Enuc"],
        calculated_energies["Enuc"],
        precision,
        "Enuc",
    )

    return


@pytest.mark.extern
@pytest.mark.saptdft
@pytest.mark.parametrize(
    "test_id, external_pot_keys, energy_method, expected_values, precision",
    [
        (
            "abc",
            ["A", "B", "C"],
            "sapt(dft)",
            {
                "Edisp": -0.0030693649240918575,
                "Eelst": -0.04826201906029539,
                "Eexch": 0.021079055937265008,
                "Eind": -0.03707409031613845,
                "Enuc": 37.565065473343004,
                "Etot": -0.0673264183632607,
            },
            8,
        ),
        (
            "ab",
            ["A", "B"],
            "sapt(dft)",
            {
                "Edisp": -0.003104673709893406,
                "Eelst": -0.048898650090975625,
                "Eexch": 0.021726964968606006,
                "Eind": -0.036753394978655915,
                "Enuc": 37.565065473343004,
                "Etot": -0.06702975381091894,
            },
            8,
        ),
        (
            "a",
            ["A"],
            "sapt(dft)",
            {
                "Edisp": -0.0031075463855700478,
                "Eelst": -0.028802617979504674,
                "Eexch": 0.02197564074312773,
                "Eind": -0.02299208442241478,
                "Enuc": 37.565065473343004,
                "Etot": -0.032926608044361774,
            },
            8,
        ),
        (
            "b",
            ["B"],
            "sapt(dft)",
            {
                "Edisp": -0.003124390140185924,
                "Eelst": -0.027251596619528584,
                "Eexch": 0.02246610724478254,
                "Eind": -0.017826066755037367,
                "Enuc": 37.565065473343004,
                "Etot": -0.025735946269969334,
            },
            8,
        ),
        (
            "c",
            ["C"],
            "sapt(dft)",
            {
                "Edisp": -0.003035986247790965,
                "Eelst": -0.013872053118092253,
                "Eexch": 0.02117011636321129,
                "Eind": -0.004613609990417991,
                "Enuc": 37.565065473343004,
                "Etot": -0.0003515329930899192,
            },
            8,
        ),
    ],
)
def test_saptdft_external_potential(
    test_id, external_pot_keys, energy_method, expected_values, precision
):
    """
    Parameterized test for SAPT with external potentials.

    Parameters:
    -----------
    test_id : str
        Identifier for the test case
    external_pot_keys : list
        List of keys from the full external_potentials dict to include in this test
    energy_method : str
        Energy method to use (sapt(dft) or fisapt0)
    expected_values : dict
        Expected energy values to compare against
    precision : int
        Decimal precision for comparison
    """
    # Define the molecule geometry (common to all tests)
    mol = psi4.geometry(
        """
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
        """
    )

    # All potential definitions
    psi_bohr2angstroms = qcel.constants.bohr2angstroms
    all_external_potentials = {
        "A": [
            [0.417, np.array([-0.5496, -0.6026, 1.5720]) / psi_bohr2angstroms],
            [-0.834, np.array([-1.4545, -0.1932, 1.4677]) /
             psi_bohr2angstroms],
            [0.417, np.array([-1.9361, -0.4028, 2.2769]) / psi_bohr2angstroms],
        ],
        "B": [
            [0.417, np.array([-2.5628, -0.8269, -1.6696]) /
             psi_bohr2angstroms],
            [-0.834, np.array([-1.7899, -0.4027, -1.2768]) /
             psi_bohr2angstroms],
            [0.417, np.array([-1.8988, -0.4993, -0.3072]) /
             psi_bohr2angstroms],
        ],
        "C": [
            [0.417, np.array([1.1270, 1.5527, -0.1658]) / psi_bohr2angstroms],
            [-0.834, np.array([1.9896, 1.0738, -0.1673]) / psi_bohr2angstroms],
            [0.417, np.array([2.6619, 1.7546, -0.2910]) / psi_bohr2angstroms],
        ],
    }

    # Select only the potentials needed for this test
    external_potentials = {
        key: all_external_potentials[key] for key in external_pot_keys
    }

    # Set common options
    psi4.set_options(
        {
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "SAPT_DFT_FUNCTIONAL": "pbe0",
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
            "sapt_dft_grac_shift_a": 0.1307,
            "sapt_dft_grac_shift_b": 0.1307,
        }
    )

    # Run the energy calculation
    psi4.energy(
        energy_method,
        external_potentials=external_potentials,
        molecule=mol,
    )

    # Print reference values
    print(f"TEST ID: {test_id}")
    print("REF:")
    import pprint

    pprint.pprint(expected_values)

    # Collect calculated values
    key_labels = [
        ["Eexch", "SAPT EXCH ENERGY"],
        ["Edisp", "SAPT DISP ENERGY"],
        ["Eelst", "SAPT ELST ENERGY"],
        ["Eind", "SAPT IND ENERGY"],
        ["Etot", "SAPT TOTAL ENERGY"],
    ]
    calculated_energies = {k1: psi4.core.variable(k2) for k1, k2 in key_labels}
    calculated_energies["Enuc"] = mol.nuclear_repulsion_energy()

    # Print calculated values
    print("CALC:")
    pprint.pprint(calculated_energies)

    # Compare values
    for k1, k2 in key_labels:
        compare_values(
            expected_values[k1],
            calculated_energies[k1],
            precision,
            k1,
        )

    # Also check nuclear repulsion energy
    compare_values(
        expected_values["Enuc"],
        calculated_energies["Enuc"],
        precision,
        "Enuc",
    )

    return


@pytest.mark.extern
@pytest.mark.saptdft
def test_fisapt0_sapthf_external_potential():
    mol = psi4.geometry(
        """
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
        """
    )
    # All potential definitions
    psi_bohr2angstroms = qcel.constants.bohr2angstroms
    external_potentials = {
        "A": [
            [0.417, np.array([-0.5496, -0.6026, 1.5720]) / psi_bohr2angstroms],
            [-0.834, np.array([-1.4545, -0.1932, 1.4677]) /
             psi_bohr2angstroms],
            [0.417, np.array([-1.9361, -0.4028, 2.2769]) / psi_bohr2angstroms],
        ],
        "B": [
            [0.417, np.array([-2.5628, -0.8269, -1.6696]) /
             psi_bohr2angstroms],
            [-0.834, np.array([-1.7899, -0.4027, -1.2768]) /
             psi_bohr2angstroms],
            [0.417, np.array([-1.8988, -0.4993, -0.3072]) /
             psi_bohr2angstroms],
        ],
        "C": [
            [0.417, np.array([1.1270, 1.5527, -0.1658]) / psi_bohr2angstroms],
            [-0.834, np.array([1.9896, 1.0738, -0.1673]) / psi_bohr2angstroms],
            [0.417, np.array([2.6619, 1.7546, -0.2910]) / psi_bohr2angstroms],
        ],
    }

    # Set common options
    psi4.set_options(
        {
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "SAPT_DFT_FUNCTIONAL": "hf",
            "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
        }
    )

    # Run the FISAPT0 energy calculation
    psi4.energy(
        "fisapt0",
        external_potentials=external_potentials,
        molecule=mol,
    )

    # Collect calculated values
    key_labels = [
        ["Eexch", "SAPT EXCH ENERGY"],
        ["Edisp", "SAPT DISP ENERGY"],
        ["Eelst", "SAPT ELST ENERGY"],
        ["Eind", "SAPT IND ENERGY"],
        ["Etot", "SAPT TOTAL ENERGY"],
    ]
    calculated_fisapt0_energies = {k1: psi4.core.variable(k2) for k1, k2 in key_labels}
    calculated_fisapt0_energies["Enuc"] = mol.nuclear_repulsion_energy()

    # Run the SAPT(HF) energy calculation
    psi4.energy(
        "sapt(dft)",
        external_potentials=external_potentials,
        molecule=mol,
    )

    calculated_sapthf_energies = {k1: psi4.core.variable(k2) for k1, k2 in key_labels}
    calculated_sapthf_energies["Enuc"] = mol.nuclear_repulsion_energy()

    print("FISAPT0:")
    pp(calculated_fisapt0_energies)
    print("SAPT(HF):")
    pp(calculated_sapthf_energies)

    # Compare values
    for k1, k2 in key_labels:
        compare_values(
            calculated_fisapt0_energies[k1],
            calculated_sapthf_energies[k1],
            8,
            k1,
        )

    # Also check nuclear repulsion energy
    compare_values(
        calculated_fisapt0_energies["Enuc"],
        calculated_sapthf_energies["Enuc"],
        8,
        "Enuc",
    )


if __name__ == "__main__":
    test_saptdft_external_potential(
        "c",
        ["C"],
        "sapt(dft)",
        {
            "Edisp": -0.003035986247790965,
            "Eelst": -0.013872053118092253,
            "Eexch": 0.02117011636321129,
            "Eind": -0.004613609990417991,
            "Enuc": 37.565065473343004,
            "Etot": -0.0003515329930899192,
        },
        8,
    )
    test_fisapt0_sapthf_external_potential()
