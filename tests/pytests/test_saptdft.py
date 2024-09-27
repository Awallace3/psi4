import pytest
import psi4
from qcelemental import constants
from psi4 import compare_values

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


if __name__ == "__main__":
    test_sapt_dft_compute_ddft_d4()
