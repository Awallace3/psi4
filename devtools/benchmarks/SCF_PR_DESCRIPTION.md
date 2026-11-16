## Summary

This is the **exact, generally applicable SCF optimization tier**. It now includes
the exact portions of the newer integral, DF-JK, grid, XC, and VV10 contributions,
not just the original persistent-scratch/panel-GEMM changes.

```diff
 DF-JK initialization
- repeated wrapper/sieve setup, mask scans, scattered metric copies
+ parallel thread-local Libint engines; lazy shell-pair caches
+ compact significant-column indices; reused panels; row-partitioned scatter
 DF-JK iterations
- repeated large scratch allocations; general GEMM for symmetric K
+ budgeted reusable scratch; FP64 BLAS SYRK for symmetric K
 DFT grid and XC
- repeated nuclear-coordinate work; copied octree index vectors; separate dot loops
+ thread-local distance scratch; contiguous stable partitions; fused FP64 loops
+ row-tile locks without per-thread full Fock matrices
 VV10
- recreate the nonlocal grid; repeated map copies/divisions
+ retain the geometry-invariant grid/workers; tile the same FP64 pair kernel
 SCF diagonalization and DIIS
- repeated diagonalization workspace; avoidable history reads/copies
+ retained DSYEVD workspace; input-restoring DSYEV fallback
+ explicitly reserved in-core history when it fits; bounded disk ADIIS otherwise
```

The eigenvalue-truncated EIGEN fitting metric, final grids, screening thresholds,
and SCF stopping criteria are unchanged. “Exact” allows ordinary floating-point
reassociation, parallel accumulation order, and rotations within degenerate
eigenspaces; it does not promise bitwise equality.

**Not in this PR:** Cholesky fitting metrics (which omit eigenvalue truncation),
FP32 J/K/wK, CPU-selected IQ layouts, initial/final grid switching and DIIS
rebasing, VV10 deferral, predicted-energy stopping, approximate Becke denominator
pruning, or provider/NUMA-specific tuning. Those remain on `scf-dft-opt`.

The extraction includes exact portions of `bdced238c3`, `88e1ce672c`,
`71f2607745`, `4e73b7c14b`, and `399f5d8e9f`. Mixed-purpose commits were split
by code path rather than cherry-picked wholesale. `SCF_DF_METRIC`, previously
advertised here, has moved to the optional tier.

## Evidence

- **Before:** #41 ended at `d0b021feff`; the newer exact contributions existed only
  on the larger optimization branch.
  **After:** the exact paths are extracted into this branch without their
  optional precision/grid/convergence controls.
- **Before:** pre-split default-path observables were captured for RHF/UHF HF,
  LSDA, GGA, meta-GGA, range-separated exchange, and VV10.
  **After:** **861 tests passed**, plus **4** injected DSYEVD checks, **2**
  reduced-team checks, and both PSIO/IWL round trips. All **72** observable
  comparisons passed: maximum differences **8.53e-14 Eh** (energy),
  **1.00e-14** (density), **1.78e-14** (Fock), and
  **8.05e-10 Eh/Bohr** (gradient). Details:
  `devtools/benchmarks/SCF_BRANCH_SPLIT.md`.
- The corrected **pre-split narrow** six-lane native CI passed:
  https://github.com/Awallace3/psi4/actions/runs/43621893847 .
  This is historical evidence, not certification of the expanded tree.
- New regressions cover independent DiskDF J/K/wK comparisons, zero omega,
  symmetric versus general FP64 contraction, held-metric independence, tight
  memory/reduced teams, all five nuclear-weight schemes, deterministic exact-tier
  grid decisions, and DIIS storage equivalence.

Earlier Phoenix timings measured subsets of this work, not the final combined
tree. They are not presented as new combined-tier speedup measurements.

## Merge Danger

**Door:** two-way

No new acceleration keyword is required. `Matrix::diagonalize_dsyevd` and
scratch-accounting helpers are additive APIs. Reverting restores the old paths.

**Blast Radius:** SCF

DF integral screening/storage, grid blocking, XC/VV10 kernels, memory partitioning,
DIIS history storage, and diagonalization are affected. Floating-point reduction
order can change; raw degenerate eigenvectors need not match. Memory accounting
includes retained allocations and DIIS scratch rather than borrowing the same
allowance twice. Correctness fixes also cover PSIO cleanup without DPD,
character-table initialization, metric-power independence, and source-built
Gau2Grid's C11 aligned-allocation size requirement. Existing binary Gau2Grid
packages are not rewritten by the source patch.
