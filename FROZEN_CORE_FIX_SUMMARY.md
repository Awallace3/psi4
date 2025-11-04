# F-SAPT Frozen Core Fix - Implementation Summary

## Problem
The `flocalization()` function in `/home/awallace43/gits/psi4/psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py` incorrectly calculated the number of active orbitals when frozen core was enabled, leading to segmentation faults.

## Root Cause
The function incorrectly interpreted `cache["Cocc_A"].shape[1]` as the number of **active** orbitals, when it actually represents the **total** number of occupied orbitals (frozen + active).

### Original (Incorrect) Code - Monomer A (lines 117-121):
```python
nn = cache["Cocc_A"].shape[0]
nf = nfocc0A
na = cache["Cocc_A"].shape[1]  # WRONG: This is total, not active!
nm = nf + na                    # WRONG: Double counts frozen cores!
ranges = [0, nf, nm]
```

This caused `nm = nf + (nf + na_actual)` which double-counted frozen cores.

### Fixed Code - Monomer A (lines 117-121):
```python
nn = cache["Cocc_A"].shape[0]
nf = nfocc0A
nm = cache["Cocc_A"].shape[1]  # total occupied orbitals (frozen + active)
na = nm - nf                    # active occupied orbitals only
ranges = [0, nf, nm]
```

Now correctly computes: `na = total - frozen = active only`

## Changes Made

### File Modified
`/home/awallace43/gits/psi4/psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py`

### Monomer A Fix (lines 117-121)
- Changed order: compute `nm` first (total occupied)
- Then compute `na = nm - nf` (active = total - frozen)

### Monomer B Fix (lines 171-175)  
- Applied identical fix for monomer B
- Changed order: `nm` first, then `na = nm - nf`

## Testing

### Test Created
`test_fsapt_frozen_core.py` - Water dimer with `freeze_core=true`

### Test Results
```
✓ F-SAPT Energy computed: -0.008101989568376967
✓ IBO Localizer 2 converged successfully
✓ Number of frozen core orbitals: 1
✓ No segmentation fault
✓ All localization steps completed successfully
```

### Log Verification
The test log `/home/awallace43/gits/psi4/test_fsapt_frozen_core.log` shows:
- Frozen core recognized and set correctly
- F-SAPT Localization (IBO) completed
- IBO Localizer 2 converged for both monomers
- All F-SAPT components computed successfully

## Previous Session Fixes
This fix complements the earlier fix to the `localization()` function (lines 45-105 in the same file), which handles standard SAPT localization with frozen core.

## Technical Details

### Key Insight
- `cache["Cocc_X"]` contains ALL occupied orbitals
- `shape[1]` = total occupied = frozen + active
- Cannot use it directly as `na` (active only)
- Must compute: `na = total - frozen`

### Correct Ranges Parameter
- `ranges = [0, nf, nm]` tells localizer:
  - `[0, nf)`: frozen core orbitals (excluded from localization)
  - `[nf, nm)`: active orbitals (to be localized)
  
### Downstream Impact
The fix ensures:
- `Lfocc0A`, `Lfocc0B`: correctly sized frozen orbital matrices
- `Laocc0A`, `Laocc0B`: correctly sized active orbital matrices
- Slicing operations `[:, :nf]` and `[:, nf:nf+na]` work correctly

## Next Steps
1. Run full F-SAPT test suite (when einsums module is available)
2. Verify existing tests in `/home/awallace43/gits/psi4/tests/pytests/test_fisapt.py` pass
3. Consider adding regression tests specifically for this frozen core edge case

## Files Modified
1. `/home/awallace43/gits/psi4/psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py` (lines 117-121, 171-175)

## Files Created  
1. `/home/awallace43/gits/psi4/test_fsapt_frozen_core.py` - Standalone test for F-SAPT with frozen core
2. `/home/awallace43/gits/psi4/test_fsapt_frozen_core.log` - Test output log
