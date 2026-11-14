# SCF pre-PR correctness and portability checks

The narrow branch is `scf-dft-pr-v2` (worktree `scf-dft-pr`). The larger
`scf-dft-opt` branch adds grid construction, fused XC/VV10 kernels, alternative K
layouts, and opt-in precision/convergence changes. Validate each branch's own
native extension; passing tests against an environment's release Psi4 is not
evidence for either branch.

## Corrections

- JK and separate wK share retained transform and coefficient scratch. Blocking
  charges high-water allocations and every resident AO/metric tensor on every
  block; it releases oversized scratch before replanning a reduced budget.
- DSYEVD allocates the same documented minimum workspace that SCF reserves.
  Growth releases old vectors first instead of retaining geometric capacity.
  A failed DSYEVD retries DSYEV from the original matrix in the same allocation.
  Integer workspace accounting uses `sizeof(int)`, not an assumed byte width.
- SCF rejects a nonpositive post-diagonalization memory allowance before
  constructing JK or collocation caches.
- OpenMP-disabled builds clamp explicitly requested JK/helper thread counts to
  one, so serial code does not reduce uninitialized per-thread buffers.
- PSIO close skips DPD-cache flushing when no DPD instance exists. Pure SCF and
  disk DIIS do not initialize DPD; calling its member through a null pointer was
  undefined behavior exposed by the full UBSan run.
- The bit-based character-table constructor initializes all metadata/pointers,
  as its string-based counterpart already did. Copying its uninitialized group
  enum was another UBSan failure reached during SCF setup.

Additional opt-branch safeguards:

- FP32 scratch uses byte-backed storage with explicit placement-new array
  lifetimes. Float accesses no longer reinterpret live double arrays. A precision
  switch reuses the allocation without copying, and pointers are reacquired
  before the next double contraction. Shared reduction buffers' type transitions
  happen before the parallel region.
- `PSI_RESTRICT` selects the compiler spelling, or no hint on unsupported
  compilers. Define `PSI_NO_RESTRICT` for optimized reference builds. This does
  not remove the need to audit buffer ownership: read-only VV10 arrays may
  overlap, while each modified XC/scratch output must be independent of its
  inputs.
- MKL thread control uses the configured provider's official service header,
  without weak-symbol assumptions on Windows/macOS. Each worker restores its
  previous thread-local setting, not a hard-coded global default.
- In-core DIIS requires an explicit reservation removed from the JK/grid
  partition, including accelerator/extrapolation copies. Large unreserved
  histories remain on disk, with a separately reserved bounded scratch allowance.
  Disk-backed ADIIS streams density/Fock differences instead of retaining a
  second complete subspace in RAM.
- Grid SHIFT computes residual corrections using each historical density.
  EDIIS resets because historical coarse-grid energies cannot be consistently
  shifted. Pure DIIS retains densities when SHIFT is requested.
- OOO falls back to INTERNAL when FP32 exchange is requested, so it cannot
  bypass FP64 reconvergence. Precision state is reset at iteration entry and
  exit, including exceptions. Deferred VV10 is restored after exceptions.
- Predicted energy changes contract `F` with `D_new - D_old`, not the
  difference of two extensive energy-like totals.
- Symmetric K partitions/reduces over the actual OpenMP team, not the requested
  team. Coefficient-buffer initialization covers all logical buffers even when
  OpenMP reduces a team.

## Native regression suite

From the build directory, with its staged extension on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/stage/lib" PYTHONSAFEPATH=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python /path/to/source/devtools/benchmarks/scf_portability_tests.py \
    --architecture x64 --workers 2
# For an OpenMP-disabled build, append --openmp off.
# For native ARM64, use --architecture arm64.
```

The runner verifies native architecture and staged module paths, records
provenance, and writes JUnit results. It includes memory thresholds/lifecycle,
singular and diffuse metric comparisons, gradients and a finite-difference
check, DSYEVD triangle/degeneracy checks, threaded XC matrices/gradients, and
existing JK, matrix, grid-blocking and TDDFT regressions. The opt-only switching
tests check DIIS policies/rebasing, cancellation-sensitive predicted energies,
OOO fallback, MAXITER safeguards, and isolated/full-stack observables.
The default response subset excludes the large single-threaded methyl-oxirane
TDDFT cases; pass `--extended` to include them.

## Deterministic LAPACK fault injection (Linux)

```bash
cc -shared -fPIC /path/to/source/devtools/benchmarks/scf_lapack_probe.c \
    -ldl -o "$PWD/lapack-probe.so"
python /path/to/source/devtools/benchmarks/scf_lapack_tests.py \
    --probe "$PWD/lapack-probe.so"
```

This separately simulates an optimum larger than the reserved workspace, a
DSYEVD failure that corrupts its input copy, a failed metric factorization, and a
low reciprocal condition estimate. The probe is test-only and must never be
linked into production.

## Compiler/architecture matrix

`.github/workflows/ecosystem.yml` runs staged regressions on GNU/Linux,
Clang/Intel macOS, native Apple ARM64, and Windows clang-cl. An additional Linux
Clang/OpenBLAS variant disables OpenMP and restrict hints. Source checkout is
pinned to the tested PR snapshot. These are configured checks, not proof that a
particular remote run passed.

The opt-only `scf-scratch.yml` also runs the standalone scratch test on Linux
x86/ARM64, macOS and Windows, including strict-aliasing and sanitizer variants:

```bash
c++ -std=c++17 -O3 -fstrict-aliasing \
    devtools/benchmarks/scf_scratch_portability.cc -o scratch-test
./scratch-test
c++ -std=c++17 -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer \
    devtools/benchmarks/scf_scratch_portability.cc -o scratch-sanitized
./scratch-sanitized
```

Experimental Clang TypeSanitizer can be tested separately with `-fsanitize=type`
and `TYSAN_OPTIONS=halt_on_error=1`. Trivial placement-array-new is not reliably
recognized by its shadow tracking; TypeSanitizer builds explicitly clear shadow
types through its documented memset interceptor when beginning a new array
lifetime. Normal builds do not zero or copy the storage.

For a full GCC ASan/UBSan core build, use a separate build directory and the same
dependency prefix/compiler as the optimized build:

```bash
cmake -S "$SOURCE" -B "$SANITIZER_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O0 -g0 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
    -DENABLE_XHOST=OFF -DENABLE_simint=OFF
cmake --build "$SANITIZER_BUILD" -j 6
export PYTHONPATH="$SANITIZER_BUILD/stage/lib" PYTHONSAFEPATH=1
export LD_PRELOAD="$(c++ -print-file-name=libasan.so)"
export ASAN_OPTIONS=detect_leaks=0:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q -s \
    "$SOURCE/tests/pytests/test_scf_optimization.py" \
    "$SOURCE/tests/pytests/test_scf_switching.py" \
    "$SOURCE/tests/pytests/test_scf_portability.py" \
    -k "not xc_threaded and not cholesky_diffuse and not grid_switch_isolated"
```

Use the matching GCC compiler for `-print-file-name=libasan.so`. The local
validation additionally enabled SIMINT against an already built library.
`-O0` keeps sanitizer build costs manageable; optimized correctness is checked
separately. Leak detection is disabled for the Python/dependency process, not
address/undefined-behavior diagnostics. Keep pytest capture disabled (`-s`):
a fail-fast sanitizer can otherwise exit before pytest exposes its captured
diagnostic.

Use `ENABLE_XHOST=OFF` for generic builds. AVX2 may be executed only on a capable
CPU; AVX-512 translation units can be compile-checked locally but require
appropriate hardware for execution. QEMU plus an ARM64 environment can supplement
native checks, but does not establish performance or prove weak-memory threading
correctness. ASan/UBSan and `-fno-strict-aliasing` do not check restrict contracts.

Do not interpret energy agreement alone as validation of an optimization:
compare densities/Focks/gradients and inspect precision/grid transition behavior.
