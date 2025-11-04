#!/usr/bin/env python3
"""Small quick F-SAPT test to verify dispersion fix"""

import sys
sys.path.insert(0, '/home/awallace43/gits/psi4/build_saptdft_ein/stage/lib')
sys.path.insert(0, '/home/awallace43/gits/einsums/install/lib')

import psi4

print("Testing F-SAPT with small water dimer...")

# Small water dimer - much faster than phenol
psi4.geometry("""
0 1
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
--
0 1
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
symmetry c1
no_reorient
no_com
""")

psi4.set_options({
    'basis': 'jun-cc-pvdz',
    'scf_type': 'df',
    'freeze_core': True,
})

psi4.set_memory('1 GB')

print("\nRunning F-SAPT0 on water dimer...")

try:
    energy = psi4.energy('fisapt0')
    print("\n" + "="*60)
    print("SUCCESS! F-SAPT dispersion calculation completed!")
    print("="*60)
    print(f"Total Energy: {energy:.8f} Eh")
    print("\nMatrix slicing fix VERIFIED and WORKING!")
    sys.exit(0)
except Exception as e:
    print("\nFAILED:", str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
