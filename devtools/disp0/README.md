# Dedicated Einsums2 CUDA build preparation

## Current blocker — do not reuse the historical build recipe

**Installed runtime blocked; not GPU-validated.** CPU builds14182584,
14193865 and14196330 failed; preserve their status/roots. The third compiled
both libraries but did not install E2 Python. Its whole-artifact recovery
passed byte/ELF checks (190,505,547bytes in59.5s), yet the new-prefix mapping
proof failed: E2 `_core` has **DT_RPATH** pointing first to the original
`14196330/einsums-build/lib`, overriding the requested new native prefix.
Both E2 binaries also retain site CUDA12.6 paths before explicit CUDA13.
The loader trace confirms CUDA12.6 cuSPARSE/nvJitLink alongside CUDA13;
the original cache selected those components after finding their unversioned
development links missing from the explicit CUDA13 prefix. No tiny
API pass, diagnostic compile, CUDA execution, GPU correctness or speedup
has been established for the deployed prefix.

At exact E2d17, `cmake/Einsums_FinalizePybind.cmake` builds the Python
package but contains no Python `install()` rules. Passing the historical
`EINSUMS_INSTALL_PYMODDIR=naive` did not install that package. The build
commands below are **historical evidence, not a corrected ready recipe**.
Do not patch RPATH, overlay packages, weaken map checks, retry builds, or
consume GPU allocations to repair this. A revised coherent installation/
loader contract needs explicit review/clearance; no additional CPU allocation
is authorized. See `../../disp0-progress.md` for exact evidence and approvals.
`runtime_correction_plan.md` records the minimal proposed revised contract;
it is planning only, not an executable recipe or new allocation clearance.

## Historical configuration and validation plan

* Einsums Git pin: `d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62`.
* Build native shared library and Python `_core` in the same configure/install.
  Soname is `libEinsums-abi1.so.1`; package names are not ABI proof.
* `environment.yml` is a fresh-prefix proposal matching essential local CPU
  dependency versions. It intentionally excludes conda Einsums1 and release
  Psi4. Solve/install success and the full explicit lock must be recorded.
  Catch2 3.16.0 is required even with E2 tests OFF: d17bb76e unconditionally
  probes its string-view symbol with `check_cxx_source_compiles`. The installed
  IMPORTED target works; the FetchContent source alias does not cross into
  that try_compile project. The recipe preflights this exact link and pins
  CMake 4.3.2 to the CPU baseline.
  Apiary1.0 also requires discoverable matching Clang builtin headers even
  when the native compiler is GNU. Install clangxx22.1.4 for AST resources,
  bind `CONDA_PREFIX` to the dedicated environment and run
  `apiary_config_probe` before building E2. Apiary can return0 and emit partial
  AST after fatal parser errors: reject diagnostics as well as missing output.
  The negative resource probe must fail; the matched-resource probe must pass.
* Use a fresh branch-specific native Psi4 build. Native `ENABLE_Einsums=OFF`
  is intentional, as in the CPU oracle: this route uses Python RuntimeTensors,
  not Psi4 C++ Einsums objects. Python `_core` must still link the one matching
  Einsums2 shared library. Inspect ELF maps and `einsums.sealed.world()`.
* The coordinator's cuEST SDK and Einsums1 stages are not dependencies of this
  initial dispersion route. Native SCF, DFHelper and other SAPT terms remain
  CPU. Only eligible Python Einsums2 contractions reach its existing cuBLAS
  endpoints. No whole-run GPU coverage or speedup is claimed.

Candidate CPU-only build allocation: 8 CPUs, 48G host, 2h, explicit embers,
gts-cs207-chemx. No GPU needed when arch 90 is explicitly configured.
Check the exact route with `sbatch --test-only`, send provenance/readiness to
the parent and obtain coordination before submission.
Job 14182584 already consumed the original one-job approval and failed at
Catch2 configuration. Separately approved corrected job14193865 consumed its
one-job approval and failed at Apiary resource discovery. No retry is
authorized by this recipe. Any corrected
build requires fresh approval, fresh source/environment/build roots and a
new immutable pin. Preserve the failed roots. Future jobs explicitly request
`--no-requeue`.

Candidate Einsums configuration (requires actual isolated paths):

```text
CMAKE_BUILD_TYPE=Release
BUILD_SHARED_LIBS=ON
CMAKE_INSTALL_LIBDIR=lib
EINSUMS_INSTALL_PYMODDIR=naive
EINSUMS_WITH_CXX_STANDARD=20
EINSUMS_WITH_CUDA=ON
EINSUMS_WITH_CUDA_ARCHITECTURES=90
EINSUMS_WITH_HIP=OFF
EINSUMS_WITH_MPS=OFF
EINSUMS_WITH_GPU_MOCK_DISCRETE=OFF
EINSUMS_WITH_MALLOC=system
EINSUMS_BUILD_PYTHON=ON
EINSUMS_PYBIND_NUM_TU=8
EINSUMS_WITH_PROFILER=ON
EINSUMS_WITH_TESTS=OFF
EINSUMS_WITH_EXAMPLES=OFF
EINSUMS_WITH_DOCUMENTATION=OFF
```

Bind the selected Python, Apiary, pybind11, HDF5, MKL LP64 BLAS/LAPACK,
GCC/G++14.3 compiler frontends and explicit CUDA13 prefix. After module
bootstrap clear the site's GCC_ROOT/include-path/library-path variables.
Validate cc1/cc1plus resolution, selected headers and compile/link ABI first.
CUDA13.3 is a candidate for E2, not already a validated combination.

Move all tracked source/recipes via Git. Never install into shared prefixes,
copy a new Python package over an old scientific core, or repoint queued pins.

Build acceptance:

1. Full source/config/environment lock and actual CMake compiler/toolkit pins.
2. Installed native/Python identity, binary hashes and single-world library map.
3. `disp0_einsums_api_probe.py` against the dedicated Psi4 stage, plus the
   signed CPU/default-graph tests and unchanged non-HF regression.
4. This proves build/CPU semantics, **not GPU shadow correctness**.

Then prepare the separately coordinated one-H200/4CPU/32G/30min embers
numerical gate. Keep the real cost model. Require a naturally eligible block
with actual CUDA/cuBLAS evidence, CPU-fallback edges, signed native-oracle
comparisons, changed physical inputs/repeated calls and retained discrepancies.
Record packing/transfers/CPU work in end-to-end time.

For a diagnostic run, Einsums profiler options include
`--einsums:profile:save=<session.json>` and `--einsums:profile:detailed`.
Use the equals form in Python: the import hook claims namespace-prefixed
tokens, not a separate following value.
The driver honors explicit profile CLI options instead of disabling profiling.
Actual `gpu_dispatch` annotations distinguish `gemm` from `cpu_fallback`;
planned targets alone do not establish execution. Correlate them with the
real CUDA backend/device UUID/library maps and a CUDA/cuBLAS trace.
