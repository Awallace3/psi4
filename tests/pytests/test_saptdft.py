import pytest
import psi4
from utils import *
from addons import using, uusing
from qcelemental import constants

hartree_to_kcalmol = constants.conversion_factor("hartree", "kcal/mol")
pytestmark = [pytest.mark.psi, pytest.mark.api]


def test_sapt_dft2():
    from psi4.driver.procrouting.proc_util import prepare_sapt_molecule
    from psi4.driver.procrouting.proc import run_scf
    mol_dimer = psi4.geometry("""
  O -2.930978458   -0.216411437    0.000000000
  H -3.655219777    1.440921844    0.000000000
  H -1.133225297    0.076934530    0.000000000
   --
  O  2.552311356    0.210645882    0.000000000
  H  3.175492012   -0.706268134   -1.433472544
  H  3.175492012   -0.706268134    1.433472544
  units bohr
""")
    dft_functional = "pbe0"
    psi4.set_memory("16 gb")
    psi4.set_num_threads(4)
    psi4.set_options({'basis': 'aug-cc-pvdz',
                     "e_convergence": 1e-8,
                     "d_convergence": 1e-8,
                     "sapt_dft_grac_shift_a": 0.136,
                     "sapt_dft_grac_shift_b": 0.136,
                     "SAPT_DFT_FUNCTIONAL": dft_functional,
                     "SAPT_DFT_DO_DDFT": True,
                     "SAPT_DFT_D4_IE": True,
      })
    # Seems like choosing dimer basis actually changes IE quite significantly for aDZ: DHF users dimer basis so making same choice here
    # d4_params = {"s8": 0.95948085, "a1": 0.38574991, "a2": 4.80688534, "s9": 1.0}
    mol_dimer, mol_monA, mol_monB = prepare_sapt_molecule(mol_dimer, 'dimer')
    dimer_d4, _ = mol_dimer.run_dftd4(dft_functional, 'd4bjeeqatm') # run_dftd4 returns energy and C6s
    monA_d4, _ = mol_monA.run_dftd4(dft_functional, 'd4bjeeqatm')
    monB_d4, _ = mol_monB.run_dftd4(dft_functional, 'd4bjeeqatm')
    d4_IE = dimer_d4 - monA_d4 - monB_d4
    print(f"{d4_IE = }")
    run_scf(dft_functional, molecule=mol_dimer)
    dimer = psi4.core.variable("CURRENT ENERGY")
    run_scf(dft_functional, molecule=mol_monA)
    monA = psi4.core.variable("CURRENT ENERGY")
    run_scf(dft_functional, molecule=mol_monB)
    monB = psi4.core.variable("CURRENT ENERGY")
    dft_IE = dimer - monA - monB
    print(f"{dft_IE = }")
    psi4.energy("SAPT(DFT)")
    ELST = psi4.core.variable("SAPT ELST ENERGY")
    EXCH = psi4.core.variable("SAPT EXCH ENERGY")
    IND = psi4.core.variable("SAPT IND ENERGY")
    DISP = psi4.core.variable("SAPT DISP ENERGY")
    DELTA_HF = psi4.core.variable("SAPT(DFT) DELTA HF")
    DDFT = psi4.core.variable("SAPT(DFT) DELTA DFT")
    DFT_MONA = psi4.core.variable("DFT MONOMER A ENERGY")
    DFT_MONB = psi4.core.variable("DFT MONOMER B ENERGY")
    # print(f"{monA = }")
    # print(f"{psi4.variable('DFT MONOMERA') = }")
    # assert compare_values(monA, psi4.variable("DFT MONOMERA"), 6, "DFT MONOMERA")
    # assert compare_values(monB, psi4.variable("DFT MONOMERB"), 6, "DFT MONOMERB")
    DFT_DIMER = psi4.core.variable("DFT DIMER ENERGY")
    D4_IE = psi4.core.variable("D4 IE")
    DFT_IE = DFT_DIMER - DFT_MONA - DFT_MONB
    print(f"Calc\n   {DFT_DIMER = }\n   {DFT_MONA = }\n   {DFT_MONB = }")
    print(f"Test\n   {dimer     = }\n   {monA     = }\n   {monB     = }")
    delta_DFT = dft_IE - ELST - EXCH - (IND - DELTA_HF)
    print(f"{delta_DFT = }")
    print(f"{DDFT = }")
    print(f"{DFT_IE = }")
    print(f"{dft_IE = }")
    print(f"{d4_IE = }")
    assert compare_values(d4_IE, psi4.variable("D4 IE"), 7, "D4 IE")
    assert compare_values(delta_DFT, psi4.variable("SAPT(DFT) DELTA DFT"), 7, "SAPT(DFT) delta DFT")

    d4_IE *= hartree_to_kcalmol
    delta_DFT *= hartree_to_kcalmol
    delta_DFT_d4_disp = d4_IE + delta_DFT
    print(f"{delta_DFT = }")
    print(f"{delta_DFT_d4_disp = }")


test_sapt_dft2()
