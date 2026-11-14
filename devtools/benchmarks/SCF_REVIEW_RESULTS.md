# SCF pre-PR validation results

**Local validation passed. Native cross-platform CI remains externally blocked.**

Source worktrees are `scf-dft-pr` (branch `scf-dft-pr-v2`) and `scf-dft-opt`.
Original branch tips, indexes, and pre-existing work were preserved. No PR was
opened. Temporary validation branches:

- `review/scf-dft-portability-pr-20261113-v1`
- `review/scf-dft-portability-opt-20261113-v1`

## Final results

| Build/check | Result |
|---|---:|
| Narrow branch, GCC 14 / MKL / OpenMP | 827 passed |
| Opt branch, GCC 14 / MKL / OpenMP / restrict enabled | 1,000 passed |
| Opt branch, Clang 22 / OpenBLAS / no OpenMP / no restrict | 1,000 passed |
| Full opt ASan/UBSan build, focused runtime suite | 183 passed, 12 deselected |
| Injected LAPACK workspace/failure tests, each numerical build | 8 passed |
| Reduced OpenMP team, symmetric FP64 and FP32 K | 2 passed |
| Standalone PSIO/IWL round trips | Both passed on all four local builds |

The two opt numerical builds ran identical JUnit case-ID sets. The OpenBLAS
extension was checked with `ldd`: it has no MKL/OpenMP dependency. Normal
numerical builds used the installed dependencies. The sanitizer dependency
qualification below is important.

Additional checks passed:

- Standalone typed scratch: GCC 13 and Clang 22 at `-O3` with strict aliasing,
  with restrict disabled, and under ASan/UBSan.
- Separate experimental TypeSanitizer test, with explicit shadow reset for
  trivial placement-new array lifetime transitions.
- ARM64 Linux, Windows/MSVC, and macOS restrict-macro syntax, enabled/disabled.
- AVX-512 VV10 translation-unit compilation. AVX-512 execution was not tested
  on this AVX2-only host. No QEMU execution was used.
- Python syntax, workflow YAML, and `git diff --check`.

## Reproduced failures and corrections

- Insufficient post-diagonalization memory reached allocation setup; it now
  raises an early descriptive error.
- An injected legal vendor workspace optimum made the old DSYEVD implementation
  request **290 work entries versus 81 reserved**. Budget and allocation now use
  the same minimum; forced DSYEVD failure restores the input and reuses scratch.
- Reducing an OpenMP team below its requested size crashed symmetric K. Buffer
  initialization and contraction/reduction now respect actual worker creation.
- UBSan found `PSIO::close()` calling a DPD member through a null pointer during
  disk-DIIS cleanup. Both branches guard that call. Standalone tests use the real
  class declaration and fail on any accidental cache call.
- UBSan found an uninitialized group enum copied from the bit-based
  `CharacterTable` constructor. Its members are now consistently initialized,
  with direct C1/D2h copy regressions.

Memory accounting, scratch lifecycle, Cholesky conditioning/fallback, diffuse
bases, matrix/gradient equivalence, ROHF level shifting, DIIS rebasing/reservation,
OOO fallback, precision/grid transitions, MAXITER boundaries, and predicted
convergence are covered by the successful suites. See `SCF_PORTABILITY.md` for
the implementation checklist and reproduction commands.

## Gau2Grid qualification

The installed Gau2Grid 2.0.8 GCC allocator calls `aligned_alloc(64, 24)` on small
grid buffers, violating C11's size-multiple requirement. ASan correctly rejected
it. Both branches now patch the GCC/PGI allocation templates when building the
pinned source fallback.

The successful sanitizer run used a **private, corrected and instrumented
Gau2Grid 2.0.8** (source commit
`786d445ab97744e137bb3d8c04f9c259ec88b5a2`, maximum angular momentum 2). Its loaded
library path was asserted. The installed environment was not modified and
allocator diagnostics were not suppressed. Existing binary packages require a
corrected build to pass this sanitizer check; the source-build patch cannot
rewrite an already installed library.

## Remaining external validation

The fork's permissions endpoint reports Actions enabled, but workflow dispatch
returns HTTP 422: **“Actions has been disabled for this repository.”** Enabling
the already-active workflow did not resolve it. Native Windows/macOS/ARM64
runtime results therefore remain pending, not passed.

The default runner excludes the 20 large methyl-oxirane TDDFT cases;
`--extended` includes them. Earlier stopped/timed-out runs are superseded by the
completed final runs above. Test timings are not performance benchmarks.

JUnit/provenance artifacts are in the corresponding
`/fastscratch/awallace43/scf-dft-*-build/` directories and
`/fastscratch/awallace43/scf-review-portability/`. The sanitizer result is
`scf-review-portability/sanitizers-final/sanitizers.xml`.
