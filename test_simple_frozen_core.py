#!/usr/bin/env python3
"""Simple test for frozen core fix - using water dimer with minimal basis"""

import sys
import os

# Add the built psi4 to the path BEFORE importing
sys.path.insert(0, '/home/awallace43/gits/psi4/build_saptdft_ein/stage/lib')
sys.path.insert(0, '/home/awallace43/gits/einsums/install/lib')

import psi4

print("Testing frozen core with localization on water dimer...")

# Simple water dimer - much smaller than the original test
mol = psi4.geometry("""
0 1
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
--
0 1
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
units angstrom

symmetry c1
no_reorient
no_com
"""
)

print("Testing FSAPT(HF) with freeze_core=true on water dimer")
psi4.set_options(
    {
        "basis": "sto-3g",
        "scf_type": "df",
        "guess": "sad",
        "freeze_core": "true",  # This should work now!
        "FISAPT_FSAPT_FILEPATH": "none",
        "SAPT_DFT_FUNCTIONAL": "HF",
        "SAPT_DFT_DO_DHF": True,
        "SAPT_DFT_DO_FSAPT": True,
        "SAPT_DFT_D4_IE": False,  # Disable D4 for speed
        "SAPT_DFT_DO_DISP": False,
        "SAPT_DFT_D4_TYPE": "INTERMOLECULAR",
        "SAPT_DFT_GRAC_SHIFT_A": 0.09605298,
        "SAPT_DFT_GRAC_SHIFT_B": 0.073504,
    }
)

try:
    psi4.energy("sapt(dft)", molecule=mol)
    print("\n✓ SUCCESS! No segfault occurred.")
    print("✓ Frozen core with localization is now working!\n")
except Exception as e:
    print(f"\n✗ FAILED with error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
