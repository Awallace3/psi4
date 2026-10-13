# FDDS cuEST implementation handoff

## Branch and implementation

- Branch: `saptdft_cuest_disp`, based on response commit
  `4f47db835ad50ee4ed2b7b3005c99cb5cd10da35`. No merge.
- Implementation: `9fd3a2aa620ffe7bf96522dafde18b0999000d40`.
- Allocation compiler isolation: `738aa3efc9` (module GCC_ROOT unset, explicit
  conda activation, absolute cc1/cc1plus checks, C/C++ compile/link/run probes).
- Exception/partial-empty safety fixes: `5f63ef56c9`.
- Test-accessor fix: `be4d7bba6e` (`Matrix.nrow()` is not a Python API;
  use `Matrix.np.shape[0]`).
- New seam: `CuestFDDSContractions` in `psi4/src/cuest_gemm.{h,cc}`;
  used by `FDDS_Dispersion::form_unc_amplitude` and `form_aux_matrices`.
  Row-major `(ov,Q)` blocks, shared cuEST cuBLAS handle/stream, FP64.
  One/five output sums remain device-resident across all occupied blocks.
  Hybrid inputs upload once per block; only final sums download.
  Stream synchronization protects host scratch reuse. RAII cleans device
  allocations on errors; CUDA/cuBLAS errors propagate without CPU retries.
  Read-only safety review confirmed row-major mappings and capacity accounting;
  follow-up fixes drain queued transfers before exception rethrow and handle
  one-empty/one-nonempty monomers individually (including hybrid setup).
- Opt-in `USE_CUEST`, existing `CUEST_GEMM_MIN_DIM` threshold (default 256).
  CPU-only/disabled/small/empty/unsupported-layout paths remain native.
  Device preflight can shrink blocks or choose CPU before allocation.
  FDDS host memory budgets are not raised; new sizes use checked arithmetic.
- Actual-route evidence: `gpu_contraction_blocks()` plus timer
  `cuEST FDDS: contractions`. Timer includes host preparation and transfers,
  not just kernel time.
- Unchanged CPU work: integral/DFHelper preparation, density projection,
  ALDA kernel, hybrid QR and X/Y construction, Python coupled susceptibility
  products/pseudoinverses, quadrature and exchange-dispersion.
  No new MGGA/VV10/GPU-XC response support. Native HF/LDA/GGA response requires
  `CUEST_XC=false` before SCF.

## Validation status (pending; not a correctness pass)

- Static checks passed: `git diff --check`, `bash -n` for validation scripts,
  Python syntax compilation of `tests/pytests/test_cuest_fdds.py`.
- Own local CPU build: `build_fdds_cpu`, dependency environment
  `/home/awallace43/miniconda3/envs/p4_ci`, Release, Ninja, `ENABLE_cuEST=OFF`,
  `ENABLE_Einsums=OFF`, `BUILD_SHARED_LIBS=ON`, 12 build workers.
  Logs: `fdds-cpu-configure.log`, `fdds-cpu-build.log` (untracked).
  Configure and build succeeded. First native run: 13 passed, two test-only
  accessor failures, 27 GPU tests deselected; 15.00 s. Corrected native suite
  is rebuilding/rerunning. The final CPU suite includes thirteen FDDS cases
  (including integrated energies and full SAPT) plus ten native CPSCF cases.
  No scientific
  numerical assertion failed in that initial run.
- Local GPU validation is **not** attempted: known NVML driver/library mismatch.
- Own Phoenix checkout:
  `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-validation-20261012-v2`
  for the corrected retry; v1 checkout/pin is retained unchanged.
  Build: `build_fdds_validation`; dependency environment `p4cuest_sapt`
  (read-only), cuEST conda 0.2.2.2, site CUDA 12.6.1, conda GCC 14.3.
  Existing lifecycle stage supplies only read-only installed Einsums.
- Validation payload: `devtools/benchmarks/fdds_validate.sh`, batch script
  `fdds_validate.sbatch`. One H200, 8 CPUs, 64 GiB, two hours, explicitly free
  `embers`, account `gts-cs207-chemx`.
  Retry run root: `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261012-validation-02`.
  Initial job **13991223**, pin `5f63ef56c9bc8740f0f98107a42298b513bbff43`,
  was queued on free embers/gpu-h200 with Priority. User explicitly approved
  cancellation/replacement to avoid spending an allocation on the known test
  typo. It was cancelled without changing its frozen checkout.
  Retry submission is gated on corrected CPU tests passing; new job ID pending.
- Suite expectation: **20 executed tests, zero failures/errors/skips** in an
  H200 allocation; `VERIFIED.json` and `COMPLETED` required.
  Thirteen native tests cover formulas, missing keys, symmetric/asymmetric
  empty inputs, host-memory guards/recovery, full integrated FDDS and SAPT;
  seven GPU tests cover both monomers/all five hybrid contributions, 2+1
  occupied blocks, CPU/disabled/huge-threshold/forced controls, error preflight,
  default-quadrature integrated FDDS and full SVWN He-dimer SAPT(DFT).
  Matrix tolerance 2e-11, fixed-cache energy 1e-10 Eh, full SAPT energy
  target 1e-5 Eh; actual errors/times still pending.
- Integrated fixture timing excludes SCF and reports object/kernel/integration
  timers separately. Small deterministic fixtures are correctness checks,
  not representative throughput benchmarks. **No speedup claim.**
- Coordinator's response checkout/build/job and historical SCF packages were
  not modified, cancelled, rebuilt or retested.
  This includes v2 job 13980976 and v3 job 13991128/pin
  `ae8cedca770e1bc1a7a6091d313cfc6dec8887d9`.

## Next completion steps

Inspect automatic CPU build/compiler-staging notifications, run tests from
outside source root with asserted staged import path, submit the compiler-fixed
immutable FDDS pin after dry-run and live-slot checks, attach automatic SLURM
watcher, then record actual test counts, errors, times, source/core/library
provenance and job ID here. A skipped GPU suite is a validation failure.
