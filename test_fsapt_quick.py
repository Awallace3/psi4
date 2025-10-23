#!/usr/bin/env python
"""Quick test to check F-SAPT functionality after AttributeError fixes"""

import sys
sys.path.insert(0, '/home/awallace43/gits/psi4/build_saptdft_ein/stage/lib')

import psi4
import numpy as np

print("=" * 80)
print("Quick F-SAPT Test - Checking for AttributeError fixes")
print("=" * 80)

# Simple test geometry from test_fsaptdft
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

print("\nRunning SAPT(DFT) with F-SAPT enabled...")
print("This will test if the AttributeError fixes work correctly.\n")

try:
    psi4.energy("sapt(dft)", molecule=mol)
    
    print("\n" + "=" * 80)
    print("SUCCESS: SAPT(DFT) calculation completed without errors!")
    print("=" * 80)
    
    # Get energy components
    elst = psi4.core.variable("SAPT ELST ENERGY")
    exch = psi4.core.variable("SAPT EXCH ENERGY")
    ind = psi4.core.variable("SAPT IND ENERGY")
    disp = psi4.core.variable("SAPT DISP ENERGY")
    total = psi4.core.variable("SAPT TOTAL ENERGY")
    
    print("\nEnergy Components (Hartree):")
    print(f"  Elst:  {elst:15.10f}")
    print(f"  Exch:  {exch:15.10f}")
    print(f"  Ind:   {ind:15.10f}")
    print(f"  Disp:  {disp:15.10f}")
    print(f"  Total: {total:15.10f}")
    
    # Check if exchange-dispersion was computed
    try:
        exch_disp = psi4.core.variable("SAPT EXCH-DISP20 ENERGY")
        print(f"\n  Exch-Disp20: {exch_disp:15.10f}")
        print("\n✓ Exchange-dispersion calculation successful!")
    except:
        print("\n⚠ Exchange-dispersion variable not found (may not be set yet)")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print("\n" + "=" * 80)
    print("ERROR: Calculation failed!")
    print("=" * 80)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
    print("\n" + "=" * 80)
    sys.exit(1)
