#!/usr/bin/env python3
"""
Test F-SAPT with frozen core enabled to verify flocalization() fix.
This should NOT segfault with the corrected frozen core handling.
"""

import psi4

# Set memory and output
psi4.set_memory('2 GB')
psi4.core.set_output_file('test_fsapt_frozen_core.log', False)

# Water dimer geometry
water_dimer = psi4.geometry("""
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
""")

# Set options for F-SAPT with frozen core
psi4.set_options({
    'basis': 'jun-cc-pvdz',
    'scf_type': 'df',
    'freeze_core': 'true',
    'fisapt_fsapt_filepath': 'fsapt_test_output.dat'
})

# Run F-SAPT (this calls flocalization internally)
print("Running F-SAPT with freeze_core=true...")
print("This tests the flocalization() function fix.")

try:
    energy = psi4.energy('fisapt0')
    print(f"\nF-SAPT Energy: {energy}")
    print("\n✓ SUCCESS: F-SAPT completed without segfault!")
    print("✓ The flocalization() frozen core fix is working correctly.")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    raise

psi4.core.clean()
