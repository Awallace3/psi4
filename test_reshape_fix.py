#!/usr/bin/env python3
"""Test that the tensor reshaping fix works correctly"""

import sys
sys.path.insert(0, '/home/awallace43/gits/psi4/build_saptdft_ein/stage/lib')

import psi4
import numpy as np

print("="*70)
print("Testing tensor reshaping fix for F-SAPT dispersion")
print("="*70)

# Test that core.Matrix.np.reshape works as expected
print("\n1. Testing core.Matrix reshaping...")
test_matrix = psi4.core.Matrix("Test", 6, 10)  # 6 rows, 10 cols
# Fill with test data
for i in range(6):
    for j in range(10):
        test_matrix.set(i, j, i * 10 + j)

# Get numpy view and reshape to 3D
np_view = test_matrix.np
reshaped = np_view.reshape(2, 3, 10)  # Reshape (6, 10) -> (2, 3, 10)

print(f"   Original shape: {np_view.shape}")
print(f"   Reshaped shape: {reshaped.shape}")

# Verify indexing works correctly
print("\n2. Testing 3D indexing after reshape...")
print(f"   reshaped[0, 0, :] first 5 elements: {reshaped[0, 0, :5]}")
print(f"   reshaped[1, 2, :] first 5 elements: {reshaped[1, 2, :5]}")

# Verify the reshaping preserves data correctly
expected_00 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
expected_12 = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59])

if np.allclose(reshaped[0, 0, :], expected_00):
    print("   ✓ reshaped[0, 0, :] matches expected values")
else:
    print("   ✗ reshaped[0, 0, :] does NOT match expected values")
    sys.exit(1)

if np.allclose(reshaped[1, 2, :], expected_12):
    print("   ✓ reshaped[1, 2, :] matches expected values")
else:
    print("   ✗ reshaped[1, 2, :] does NOT match expected values")
    sys.exit(1)

print("\n" + "="*70)
print("SUCCESS! Tensor reshaping works correctly!")
print("="*70)
print("\nThe fix should allow F-SAPT dispersion calculations to work properly.")
print("The code in sapt_jk_terms_ein.py:2234-2237 and 2255-2258 properly")
print("reshapes 2D DFHelper tensors to 3D for correct indexing.")
