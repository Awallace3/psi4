# Frozen Core Fix Verification

## Code Changes Verified

### Monomer A (lines 117-121)
```python
nn = cache["Cocc_A"].shape[0]
nf = nfocc0A
nm = cache["Cocc_A"].shape[1]  # total occupied orbitals (frozen + active)
na = nm - nf  # active occupied orbitals only
ranges = [0, nf, nm]
```

### Monomer B (lines 171-175)
```python
nn = cache["Cocc_B"].shape[0]
nf = nfocc0B
nm = cache["Cocc_B"].shape[1]  # total occupied orbitals (frozen + active)
na = nm - nf  # active occupied orbitals only
ranges = [0, nf, nm]
```

## Test Execution Confirmed

### Command
```bash
./build_saptdft_ein/stage/bin/psi4 test_fsapt_frozen_core.py
```

### Result
- Exit code: 0 (SUCCESS)
- F-SAPT Energy: -0.008101989568376967
- No segmentation fault
- IBO Localizer converged for both monomers
- Frozen core orbitals: 1 (correctly identified)

## Technical Correctness

### Before Fix
- `na = cache["Cocc_A"].shape[1]` treated as active count
- `nm = nf + na` resulted in `nm = nf + (nf + actual_active)` = WRONG
- Double-counted frozen cores
- Incorrect array indexing → segfault

### After Fix  
- `nm = cache["Cocc_A"].shape[1]` is total occupied count
- `na = nm - nf` correctly computes active count
- `ranges = [0, nf, nm]` properly partitions frozen and active
- Correct array indexing → no segfault

## Files Modified
1. `/home/awallace43/gits/psi4/psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py`
   - Monomer A: lines 117-121
   - Monomer B: lines 171-175

## Files Created
1. `test_fsapt_frozen_core.py` - Standalone test
2. `test_fsapt_frozen_core.log` - Test output
3. `FROZEN_CORE_FIX_SUMMARY.md` - Detailed summary
4. `VERIFICATION.md` - This file

## Status
✅ FIX COMPLETE AND VERIFIED
