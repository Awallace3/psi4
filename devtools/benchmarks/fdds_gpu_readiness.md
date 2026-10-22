# FDDS 0.2.50 test readiness

This is a preparation recipe, **not a submitted allocation or device pass**.
Allocation14250361 is terminal. Historical accepted GPU timings used0.2.2.
Fresh resources, account/QoS, limits and remaining-time admission need explicit
authorization before executing an allocation-side build or test.
The current local host has no `nvcc` or CUDA runtime headers; host stubs do not
remove that prerequisite. No package installation or driver change is implied.

## What was learned from Disp0

Use its explicit ownership, queued-host testing and checked policy patterns.
FDDS currently uses FP64 cuBLAS/cuSOLVER adapters in `psi4/src/cuest_gemm.cc`,
not custom CUDA kernels. It does not need a new Python/NVRTC/CuPy dependency.
Disp0's DFMO factor contract and fitting cutoff are not substitutes for FDDS's
contact projection, metric layout, response equations or strict1e-13 pinv.

The new host-control tests compile actual adapter bodies against a queued CPU
model: capacities, nonsymmetric metrics, all four layouts, changing block sizes,
exchange/contraction tails, borrowed-storage drains and injected errors. These
are neither vendor-library tests nor parallel/device execution. In particular,
they do not certify CUDA OOM behavior, successful cleanup after a failed
`cudaFree`, or context/driver recovery.

`core._cuest_runtime_policy()` is a read-only live-handle receipt. It reports
native cuEST math mode, cuBLAS pointer/math modes and shared-stream equality;
it exposes no raw handles and performs no initialization or policy changes.
The end-to-end GPU test checks it after real JK/FDDS work. Native FP64 mode1
and cuBLAS default math mode0 are **different enum domains**. Existing
`cuESTJK::configure_math_mode()` already sets the requested policy before
computation; this change does not override `CUEST_MIXED_PRECISION`.

## Local preparation gate

Run from this worktree, using a CPU Python environment with pytest/NumPy:

```bash
PYTHONPATH="$PWD/devtools/benchmarks${PYTHONPATH:+:$PYTHONPATH}" \
python -m pytest \
  devtools/test_cuest_sdk_contract.py \
  devtools/test_cuest_cmake_admission.py \
  devtools/test_cuest_header_dependencies.py \
  devtools/test_cuest_run_admitted.py \
  devtools/test_cuest_native_version.py \
  devtools/test_cuest_runtime_lifecycle.py \
  devtools/test_cuest_runtime_policy.py \
  devtools/test_cuest_fdds_host_control.py \
  devtools/test_fdds_sdk_admission.py \
  devtools/test_cuest_grid_worker_admission.py \
  devtools/benchmarks/test_fdds_*.py \
  devtools/benchmarks/test_cuest_shared_step.py \
  devtools/benchmarks/test_cuest_shared_progress_watch.py -q \
  --junitxml=/absolute/new/evidence/local.xml
```

Freeze the resulting reviewed commit, then use Git to create a new private
source/build root. Existing validated sources/stages/outputs and shared conda
environments stay read-only. See `../cuest-sdk-0250.md` for canonical archive,
59-header, native CUDA13 library and closure hashes.

## Authorized allocation-side sequence

1. Check the actual authorized job identity/resources/remaining time, own
   duplicates, exclusive lease and cleanup before starting. No overlapping GPU
   work, builds or CPU timings. Use the established whole-step helper, deadlines,
   cleanup/quarantine rules and fresh output paths.
2. Build in the fresh checkout with the existing read-only CUDA13 dependency
   environment and separately staged canonical0.2.50 SDK. The existing driver
   unsets `GCC_ROOT` after module activation, probes absolute cc1/cc1plus,
   checks the selected nvcc/CUDA headers, and passes the exact SDK identity
   through both CMake configurations. Example *inside an authorized step*:

   ```bash
   export CUEST_SDK_ROOT=/absolute/read-only/canonical-cuest-0.2.50
   CUEST_FDDS_COMPILE_ONLY=1 bash "$SOURCE/devtools/benchmarks/fdds_validate.sh" \
       "$SOURCE" "$DEPENDENCY_ENV" "$SOURCE_PIN" "$NEW_BUILD_EVIDENCE"
   ```

   Compile-only success executes zero GPU tests and writes `BUILD_ONLY.txt`,
   never `VERIFIED.json`. Record the source, dependency/compiler/CUDA versions,
   original core/SDK closure manifest and actual consumed headers.
3. On an authorized H200 step, reuse only that new frozen binary, supplying its
   **original** `cuest-core-identity.json`, source pin and core hash. Use a separate
   frozen driver/output root; the driver's source diff guard rejects scientific
   differences from the binary pin. Set `CUEST_FDDS_READONLY_BUILD_ROOT`,
   `CUEST_FDDS_BINARY_PIN`, `CUEST_FDDS_EXPECTED_CORE_SHA256` and
   `CUEST_FDDS_BINARY_IDENTITY` explicitly. Run the driver with
   `CUEST_FDDS_FOCUSED=1 CUEST_FDDS_FULL_AFTER_FOCUSED=1`.
4. Require actual counts: focused63 plus one native-XC SAPT GPU test, then
   full112 (176 executions including repeats), zero failures/errors/skips.
   Existing command bounds are240s for focused SAPT and300s for full GDB.
   A GDB exit0 is not success: inspect native fault/breakpoint evidence, normal
   inferior exit, JUnit and same-process runtime receipts. A timeout or changed
   test count is a failure to audit, never an accepted skipped gate.
5. Require endpoint timers/counters, exact0.2.50 builtin version, post-JK
   policy receipt (native FP64, HOST pointers, shared streams), actual mapped
   core/vendor/CUDA identities and mandatory post-step cleanup. Preserve every
   failed attempt without overwriting/relabeling earlier evidence.
6. Only after correctness and cleanup pass, decide whether a complete matched
   CPU/GPU timing sequence fits the fresh remaining budget. Freeze cache,
   physical settings, storage backend, threads and application memory. Use
   protein83 and smaller; retain failures/all declared samples. Report FDDS
   separately from SCF and exchange-dispersion, with unchanged1e-5Eh energy
   gate and tighter observed errors. Do not infer a0.2.50 speedup from old0.2.2.
