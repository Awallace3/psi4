# Test Data Fix Summary

## File Modified
`tests/pytests/test_fsaptdft.py`

## Location
Lines 620-630

## Problem
The `data` dictionary was incomplete - it only contained 5 rows (indices 4-8) when the docstring (lines 609-618) clearly shows there should be 9 rows (indices 0-8).

### Missing Data (rows 0-3)
The following fragment pair combinations were missing:
- Row 0: Methyl1_A + Peptide_B
- Row 1: Methyl1_A + T-Butyl_B
- Row 2: Methyl2_A + Peptide_B
- Row 3: Methyl2_A + T-Butyl_B

## Fix Applied

### Before (5 rows):
```python
data = {
    "Frag1": ["Methyl1_A", "Methyl2_A", "All", "All", "All"],
    "Frag2": ["All", "All", "Peptide_B", "T-Butyl_B", "All"],
    # ... only 5 entries per field
}
```

### After (9 rows):
```python
data = {
    "Frag1": ["Methyl1_A", "Methyl1_A", "Methyl2_A", "Methyl2_A", "Methyl1_A", "Methyl2_A", "All", "All", "All"],
    "Frag2": ["Peptide_B", "T-Butyl_B", "Peptide_B", "T-Butyl_B", "All", "All", "Peptide_B", "T-Butyl_B", "All"],
    "Elst": [-0.106663, -1.185465, -0.106663, -1.185465, 0.554554, -1.846683, -0.106663, -1.185465, -1.292129],
    "Exch": [0.039172, 4.093049, 0.039172, 4.093049, 0.047454, 4.084767, 0.039172, 4.093049, 4.132221],
    "IndAB": [-0.038179, -0.193800, -0.038179, -0.193800, -0.023789, -0.208190, -0.038179, -0.193800, -0.231979],
    "IndBA": [-0.001862, -0.076168, -0.001862, -0.076168, 0.020246, -0.098276, -0.001862, -0.076168, -0.078030],
    "Disp": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "EDisp": [-0.031540, -1.706223, -0.031540, -1.706223, -0.147529, -1.590233, -0.031540, -1.706223, -1.737762],
    "Total": [-0.139072, 0.931393, -0.139072, 0.931393, 0.450936, 0.341385, -0.139072, 0.931393, 0.792322],
}
```

## Complete Data Table

| Index | Frag1     | Frag2     | Elst       | Exch     | IndAB      | IndBA      | Disp | EDisp      | Total      |
|-------|-----------|-----------|------------|----------|------------|------------|------|------------|------------|
| 0     | Methyl1_A | Peptide_B | -0.106663  | 0.039172 | -0.038179  | -0.001862  | 0.0  | -0.031540  | -0.139072  |
| 1     | Methyl1_A | T-Butyl_B | -1.185465  | 4.093049 | -0.193800  | -0.076168  | 0.0  | -1.706223  | 0.931393   |
| 2     | Methyl2_A | Peptide_B | -0.106663  | 0.039172 | -0.038179  | -0.001862  | 0.0  | -0.031540  | -0.139072  |
| 3     | Methyl2_A | T-Butyl_B | -1.185465  | 4.093049 | -0.193800  | -0.076168  | 0.0  | -1.706223  | 0.931393   |
| 4     | Methyl1_A | All       | 0.554554   | 0.047454 | -0.023789  | 0.020246   | 0.0  | -0.147529  | 0.450936   |
| 5     | Methyl2_A | All       | -1.846683  | 4.084767 | -0.208190  | -0.098276  | 0.0  | -1.590233  | 0.341385   |
| 6     | All       | Peptide_B | -0.106663  | 0.039172 | -0.038179  | -0.001862  | 0.0  | -0.031540  | -0.139072  |
| 7     | All       | T-Butyl_B | -1.185465  | 4.093049 | -0.193800  | -0.076168  | 0.0  | -1.706223  | 0.931393   |
| 8     | All       | All       | -1.292129  | 4.132221 | -0.231979  | -0.078030  | 0.0  | -1.737762  | 0.792322   |

## Verification
✅ All 9 rows from the docstring are now present in the data dictionary
✅ All values match the expected values from the docstring
✅ DataFrame shape is correct: (9, 9) - 9 rows, 9 columns

## Impact
This fix ensures that the test properly validates all fragment pair combinations, including the previously missing individual fragment pairs (rows 0-3). The test will now check:
- Individual fragment pairs (rows 0-3)
- Fragment-to-all combinations (rows 4-5)
- All-to-fragment combinations (rows 6-7)
- All-to-all totals (row 8)
