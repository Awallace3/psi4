# SCF branch contract

PR [#41](https://github.com/Awallace3/psi4/pull/41), published from
`scf-dft-pr`, is the **exact, generally applicable optimization tier**.
`scf-dft-opt` is the **optional/experimental tier**, retaining the exact tier's
improvements plus contributions needing a user choice or broader applicability
evidence.

“Exact” means the same mathematical operators, existing screening thresholds,
final quadrature, and convergence criteria, within ordinary FP64 roundoff. It
does not mean bitwise-identical parallel reductions or eigenvectors in degenerate
subspaces. The source worktree for #41 is named `scf-dft-pr` and its local branch
is `scf-dft-pr-v2`; publishing must explicitly update remote `scf-dft-pr`.

## Exact tier (#41)

- Persistent JK scratch and eigenvalue-solver workspace; consistent memory
  accounting, fallback/error handling, and cleanup.
- Panel metric GEMM using the existing EIGEN metric/eigenvalue cutoff.
- Parallel Libint sieving, lazy shell-pair caches, direct thread-local Coulomb
  and erf-Coulomb three-center builds, and the existing SIMINT fallback.
- Compact significant-column gathers, reused metric panels, destination-row
  scatter, and separate-wK buffer reuse. Zero omega remains distinct from no
  range separation.
- FP64 symmetric K via the configured BLAS's SYRK plus triangle mirroring.
- Parallel grid generation/flattening/block construction and contiguous stable
  octree partitions. Boundary decisions retain baseline-order center sums,
  division, and distance comparisons.
- Cached nuclear coordinates/thread-local distance scratch, specialized
  partition formulas, and a conservative bound that skips only points already
  below the existing weight cutoff. Every nonzero denominator product is
  completed.
- Fused LSDA/GGA/meta-GGA density, potential, and quadrature loops; row-tile
  locking; algebraically equivalent FP64 VV10 kernel and persistent VV10 grid.
  No VV10 pair cutoff or coarse-stage deferral.
- Memory-reserved in-core DIIS when it fits, avoiding read-only matrix clones,
  and bounded streamed disk ADIIS with unchanged histories/coefficients.
- Compiler-portable restrict hints and serial-build/threading safeguards.

Shared correctness fixes accompanying the extraction:

- Held metric powers clone the original Coulomb metric rather than modifying
  it in place; all resulting allocations are charged. The old separate-wK path
  was reproduced producing a J element of about 3442 instead of 0.166 when
  `hold_met(True)` was used.
- Symmetric AO offsets use the number of retained lower columns, not an assumed
  retained diagonal.
- Clone/coefficient initialization covers all requested logical worker slots,
  even if OpenMP temporarily creates a smaller team.
- A terminal auxiliary shell that cannot fit raises an error rather than
  emitting an invalid/empty transformation block.
- Minimum panel budgeting charges the panels plus the largest individual AO
  shell block. The half-of-spare width heuristic does not require a second full
  minimum panel allowance or force an otherwise admissible MemDF job to disk.

## Optional/experimental tier (`scf-dft-opt`)

- `SCF_DF_METRIC=CHOLESKY`, including the wK route. Unlike EIGEN, it does not
  apply `DF_FITTING_CONDITION` eigenvalue truncation.
- `SCF_K_FP32_CONVERGENCE`, including approximate J from an FP32 AO mirror.
- `SCF_DF_K_LAYOUT` and AMD CPU dispatch/padded IQ intermediates.
- Initial/final XC grids, DIIS rebasing/reset policies, and temporary VV10
  deferral.
- `SCF_E_CONV_PREDICTED` and its alternate stopping test.
- NUMA first-touch placement, fixed tile/K-slicing heuristics and provider-local
  threading control pending broader performance evidence.
- Relative `1e-20` pruning of unfinished Becke denominator products.
- Shared mutable spherical-grid objects, which change radial-shell aliasing.
- Parallel-reduction octree centers and squared-distance decision shortcuts
  whose floating-point boundary decisions may differ.

The C++ contributions originally landed in mixed-purpose commits. Classification
is by code path, not by whole commit: splitting a commit does not discard its
exact contributions or imply all its remaining changes are approximate.

## Validation status

The repartitioned exact tier passed **861 tests**, **4 injected DSYEVD workspace/
fallback checks**, **2 reduced-team checks**, and both standalone PSIO/IWL
round trips on the GCC/MKL/OpenMP build.

All **72** comparisons against the captured pre-split EIGEN/default-path
reference passed: RHF/UHF HF, LSDA, GGA, meta-GGA, LRC and VV10
energy/density/Fock/gradient observables. Maximum absolute differences were
`8.53e-14 Eh` (energy), `1.00e-14` (density), `1.78e-14` (Fock), and
`8.05e-10 Eh/Bohr` (gradient).

The exact regressions compare symmetric/nonsymmetric J/K/wK against DiskDFJK,
exercise zero omega, retained metrics, minimum memory and reduced OpenMP teams,
and check DIIS storage equivalence. All five nuclear-weight schemes preserve
exact-tier point/block decisions between one and four threads.

The optional tier passed **1,026 tests with 10 deliberate skips** under both
GCC/MKL/OpenMP and Clang/OpenBLAS/no OpenMP/no restrict, with identical case-ID
sets. The skipped cases require the exact tier's deterministic octree-center
decisions, which the optional tier deliberately overrides. Each numerical build
also passed **8 injected LAPACK checks** and both PSIO/IWL round trips; GCC passed
**4 reduced-team checks**. The Clang core's dependencies contain neither MKL nor
an OpenMP runtime.

The full optional ASan/UBSan core passed **221 focused tests**, using the private
corrected/instrumented Gau2Grid described in `SCF_REVIEW_RESULTS.md`. Python leak
detection was disabled; bounds, undefined-behavior and allocator diagnostics
were not suppressed.

The exact PR is published at `5bb6992778`; its new native matrix is
https://github.com/Awallace3/psi4/actions/runs/43634288925 . Native results for
the repartitioned trees remain pending. The historical section in
`SCF_REVIEW_RESULTS.md` describes the pre-split trees. Do not reuse old performance
numbers as measurements of this new combined tier.
