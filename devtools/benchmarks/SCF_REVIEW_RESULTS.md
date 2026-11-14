# SCF pre-PR validation results

Status: **in progress; not yet a complete PR-readiness sign-off**.

Source worktrees are `scf-dft-pr` (branch `scf-dft-pr-v2`) and `scf-dft-opt`.
Their original branch tips and pre-existing edits were preserved. Temporary
validation snapshots were pushed to:

- `review/scf-dft-portability-pr-20261113-v1`
- `review/scf-dft-portability-opt-20261113-v1`

## Verified so far

- The pre-fix insufficient-memory regression failed; the corrected budget checks
  passed on the rebuilt narrow branch.
- A vendor-workspace interposer made the old DSYEVD implementation request
  **290 work entries versus 81 reserved**. The corrected implementation passed
  minimum-workspace enforcement.
- Four injected LAPACK scenarios passed on the narrow build: larger vendor
  optimum, DSYEVD failure after input-copy corruption, failed Cholesky
  factorization, and low reciprocal condition number.
- The optimized switching/DIIS suite passed **30 tests**, including explicit
  memory reservation, disk-streamed ADIIS, historical-density residual rebasing,
  EDIIS reset, cancellation-sensitive predicted energies, OOO/FP32 fallback,
  precision/grid MAXITER safeguards, and isolated numerical equivalence.
- The final optimized review-regression suite passed **193 tests** covering the
  shared portability checks, DF/FP32 memory paths, and Python switching safeguards.
- The complete default optimized GCC/MKL/OpenMP portability selection passed
  **998 tests** in one run (four independent pytest workers).
- Clang 22/OpenBLAS/no-OpenMP/no-restrict passed the **same 998 cases**; the
  successful JUnit case-ID sets were compared exactly. Its 8 injected LAPACK
  tests also passed.
- The post-fix reduced-team regression passed **both FP64 and FP32 cases** under
  `OMP_THREAD_LIMIT=1` and `OMP_DYNAMIC=TRUE`.
- The updated optimized LAPACK injection passed **8 tests across 4 scenarios**,
  including Coulomb and range-separated exchange metric builds.
- The standalone typed-scratch regression passed GCC 13 and Clang 22 at `-O3`
  with strict aliasing, with restrict hints disabled, and under ASan/UBSan.
  The separate experimental TypeSanitizer run also passed after teaching its
  shadow tracking about trivial placement-new array lifetime transitions.
- Restrict-macro syntax checks passed Clang ARM64 Linux, Windows/MSVC, and macOS
  targets, both enabled and disabled. The AVX-512 VV10 translation unit compiled
  successfully; AVX-512 code has not been executed on this AVX2-only machine.
- The Clang 22 optimized OpenBLAS/no-OpenMP/no-restrict build succeeded; stale
  MKL linkage from the original build-cache configuration was removed before
  the provider comparison.
  `ldd` confirms OpenBLAS and no MKL/OpenMP dependency in the resulting extension.
- The narrow broad suite reached **807 passing tests with no failures** before
  its 30-minute job limit. The response-tail run passed 20 tests and the updated
  metric run passed 2. An exact case-ID audit establishes successful coverage of
  **all 825 cases in the current default selection** across those runs (removing
  duplicates and the obsolete unparameterized metric case). This is split-run
  coverage, not a claim that the timed-out process finished successfully.

## Additional reproduced defect

`OMP_THREAD_LIMIT=1` with a larger requested team reproduced a crash in the
optimized symmetric K contraction. It partitioned by the requested team and
reduced buffers for workers that did not exist. The code now allocates all
logical coefficient buffers and partitions/reduces using the actual team size.
Both post-fix reduced-team cases now pass.

## Remaining validation

- Repeat broad numerical runs after the sanitizer-discovered PSIO guard below.
- Full Psi4 ASan/UBSan build and focused runtime suite.
- Native macOS/Windows/ARM64 runtime matrix.

The full ASan/UBSan build completed. Its runtime suite exposed an existing null
DPD member call in `PSIO::close()` during disk-DIIS cleanup. The standalone
disk-DIIS test reproduced UBSan's `member call on null pointer of type DPD`.
Both branches now guard that cache-flush call when no DPD instance exists;
the focused sanitizer run and broad numerical runs are being repeated.
The PSIO and IWL standalone round-trip executables now use a real null DPD host
and abort on an accidental cache call, replacing their fake-class/non-null stub.
Both executables passed on GCC/MKL, Clang/OpenBLAS, the full ASan/UBSan build,
and the narrow build.
The subsequent UBSan pass found an uninitialized group enum in the bit-based
`CharacterTable` constructor when its result was copied. Both branches now
initialize every member consistently with the string constructor; direct C1/D2h
copy regressions were added. Final test counts will include those two new cases.

Native CI is **externally blocked**, not passed: the fork's permissions endpoint
reports Actions enabled, but explicit workflow dispatch returns HTTP 422,
“Actions has been disabled for this repository.” Enabling the already-active
workflow did not resolve this. No PR was opened.

The default portability runner excludes only the large single-threaded
methyl-oxirane TDDFT cases; `--extended` includes them. Earlier extended runs were
stopped rather than treating their partial coverage as complete evidence.

Test artifacts are under `/fastscratch/awallace43/scf-review-portability/` and
the corresponding `scf-dft-*-build` directories. Each portability run writes
JUnit XML and source/core provenance. Reproduction commands and the configured
matrix are documented in `SCF_PORTABILITY.md`.
