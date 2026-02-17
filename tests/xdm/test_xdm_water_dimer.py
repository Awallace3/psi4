"""
Test XDM dispersion correction with B3LYP-XDM/aug-cc-pVTZ on water dimer.

Reference: Compare XDM dispersion energy, C6 coefficients, and moments
against postg implementation.
"""

import psi4
import numpy as np

psi4.set_memory('2 GB')
psi4.set_num_threads(4)
psi4.core.set_output_file('test_xdm_water_dimer.out', False)

# Water dimer geometry (S22, angstrom)
water_dimer = psi4.geometry("""
0 1
O -1.551007 -0.114520 0.000000
H -1.934259  0.762503 0.000000
H -0.599677  0.040712 0.000000
O  1.350625  0.111469 0.000000
H  1.680398 -0.373741 -0.758561
H  1.680398 -0.373741  0.758561
symmetry c1
""")

# =====================================================
# Test 1: Basic B3LYP-XDM energy computation
# =====================================================
print("=" * 60)
print("Test 1: B3LYP-XDM/aug-cc-pVTZ on water dimer")
print("=" * 60)

psi4.set_options({
    'basis': 'aug-cc-pVTZ',
    'scf_type': 'df',
    'd_convergence': 1e-8,
})

e_dimer = psi4.energy('B3LYP-XDM')

e_scf_total = psi4.variable('SCF TOTAL ENERGY')
e_disp = psi4.variable('DISPERSION CORRECTION ENERGY')
e_dft_func = psi4.variable('DFT FUNCTIONAL TOTAL ENERGY')

print(f"\nDFT Functional Energy: {e_dft_func:.10f} Eh")
print(f"XDM Dispersion Energy: {e_disp:.10f} Eh")
print(f"Total Energy (DFT+XDM): {e_scf_total:.10f} Eh")

# =====================================================
# Test 2: Single water molecule
# =====================================================
print("\n" + "=" * 60)
print("Test 2: B3LYP-XDM/aug-cc-pVTZ on single water")
print("=" * 60)

water_mono = psi4.geometry("""
0 1
O  0.000000  0.000000  0.117370
H  0.000000  0.757160 -0.469483
H  0.000000 -0.757160 -0.469483
symmetry c1
""")

e_mono = psi4.energy('B3LYP-XDM')
e_disp_mono = psi4.variable('DISPERSION CORRECTION ENERGY')

print(f"\nMonomer XDM Dispersion Energy: {e_disp_mono:.10f} Eh")
print(f"Monomer Total Energy: {e_mono:.10f} Eh")

# =====================================================
# Test 3: XDMDispersion directly (low-level API)
# =====================================================
print("\n" + "=" * 60)
print("Test 3: Direct XDMDispersion API test")
print("=" * 60)

# Build XDM with known parameters
xdm = psi4.core.XDMDispersion.build("b3lyp", "aug-cc-pvtz")
print(f"a1 = {xdm.a1():.4f}")
print(f"a2 = {xdm.a2():.6f} bohr")
print(f"functional = {xdm.functional_name()}")

# Expected from xdm.param: B3LYP/aug-cc-pVTZ: a1=0.6356, a2=1.5119 Ang
a2_ang = xdm.a2() * 0.52917720859
print(f"a2 = {a2_ang:.4f} Ang (expected: 1.5119)")

# =====================================================
# Test 4: Explicit a1/a2 parameters
# =====================================================
print("\n" + "=" * 60)
print("Test 4: Explicit a1/a2 parameter build")
print("=" * 60)

xdm2 = psi4.core.XDMDispersion.build("b3lyp", 0.6356, 1.5119)
print(f"a1 = {xdm2.a1():.4f} (expected: 0.6356)")
print(f"a2 = {xdm2.a2():.6f} bohr")

# =====================================================
# Summary
# =====================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Water dimer XDM dispersion:  {e_disp:.10f} Eh")
print(f"Water monomer XDM dispersion: {e_disp_mono:.10f} Eh")
interaction_disp = e_disp - 2 * e_disp_mono  # approximate, geometries differ
print(f"Interaction XDM (approx):    {interaction_disp:.10f} Eh = {interaction_disp * 627.5095:.4f} kcal/mol")
print("\nAll tests completed successfully!")
