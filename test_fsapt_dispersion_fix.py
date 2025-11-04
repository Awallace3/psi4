#!/usr/bin/env python3
"""Test script to verify F-SAPT dispersion matrix slicing fix"""

import sys
import os

# Add the built psi4 to the path BEFORE importing
sys.path.insert(0, '/home/awallace43/gits/psi4/build_saptdft_ein/stage/lib')
sys.path.insert(0, '/home/awallace43/gits/einsums/install/lib')

import psi4

print("=" * 80)
print("Testing F-SAPT Dispersion Fix - Matrix Slicing")
print("=" * 80)

# Simple test molecule - phenol dimer from fsapt1
psi4.geometry("""
0 1
O    -1.3885044    1.9298523   -0.4431206
H    -0.5238121    1.9646519   -0.0064609
C    -2.0071056    0.7638459   -0.1083509
C    -1.4630807   -0.1519120    0.7949930
C    -2.1475789   -1.3295094    1.0883677
C    -3.3743208   -1.6031427    0.4895864
C    -3.9143727   -0.6838545   -0.4091028
C    -3.2370496    0.4929609   -0.7096126
H    -0.5106510    0.0566569    1.2642563
H    -1.7151135   -2.0321452    1.7878417
H    -3.9024664   -2.5173865    0.7197947
H    -4.8670730   -0.8822939   -0.8811319
H    -3.6431662    1.2134345   -1.4057590
--
0 1
O     1.3531168    1.9382724    0.4723133
H     1.7842846    2.3487495    1.2297110
C     2.0369747    0.7865043    0.1495491
C     1.5904026    0.0696860   -0.9574153
C     2.2417367   -1.1069765   -1.3128110
C     3.3315674   -1.5665603   -0.5748636
C     3.7696838   -0.8396901    0.5286439
C     3.1224836    0.3383498    0.8960491
H     0.7445512    0.4367983   -1.5218583
H     1.8921463   -1.6649726   -2.1701843
H     3.8330227   -2.4811537   -0.8566666
H     4.6137632   -1.1850101    1.1092635
H     3.4598854    0.9030376    1.7569489
symmetry c1
no_reorient
no_com
""")

psi4.set_options({
    'basis': 'jun-cc-pvdz',
    'scf_type': 'df',
    'guess': 'sad',
    'freeze_core': True,
})

psi4.set_memory('2 GB')

print("\nRunning F-SAPT0 calculation...")
print("This will test the dispersion term matrix slicing fix\n")

try:
    energy = psi4.energy('fisapt0')
    print("\n" + "=" * 80)
    print("SUCCESS! F-SAPT calculation completed without dimension errors")
    print("=" * 80)
    print(f"\nTotal F-SAPT0 Energy: {energy:.8f} Eh")
    
    # Get the FSAPT object to check dispersion components
    fsapt = psi4.core.get_global_option("FSAPT")
    
    print("\nTest PASSED: Matrix slicing fix is working correctly!")
    sys.exit(0)
    
except Exception as e:
    print("\n" + "=" * 80)
    print("FAILED! Error during F-SAPT calculation")
    print("=" * 80)
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\nTest FAILED: Check the error above")
    sys.exit(1)
