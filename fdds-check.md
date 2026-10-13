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
- Final CPU-tested source/tests and corrected GPU pin:
  `099706616a818041f56072c3b2471d6c04451cc8`. Also reshapes DFHelper's preserved
  `(occ,vir,Q)` NumPy metadata to `(ov,Q)` for the independent reference.
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

## Validation status (CPU passed; GPU pending, not a GPU correctness pass)

- Static checks passed: `git diff --check`, `bash -n` for validation scripts,
  Python syntax compilation of `tests/pytests/test_cuest_fdds.py`.
- Own local CPU build: `build_fdds_cpu`, dependency environment
  `/home/awallace43/miniconda3/envs/p4_ci`, Release, Ninja, `ENABLE_cuEST=OFF`,
  `ENABLE_Einsums=OFF`, `BUILD_SHARED_LIBS=ON`, 12 build workers.
  Logs: `fdds-cpu-configure.log`, `fdds-cpu-build.log` (untracked).
  Configure/build passed. Final command (from `build_fdds_cpu/fdds-tests-v3`):
  `python -m pytest <root>/tests/pytests/test_cuest_fdds.py <root>/tests/pytests/test_cuest_cphf.py -m 'not cuest' -q -s --junitxml=cpu.xml`.
  **23 passed, 0 failed/errors/skipped, 27 GPU tests deliberately deselected,
  17.36 s**: thirteen FDDS/SAPT cases plus ten native CPSCF cases.
  This is CPU compatibility evidence, not a skipped-GPU correctness pass.
  Earlier runs exposed two test-only mistakes (accessor and NumPy tensor
  shape): first 13 passed/2 failed (15.00 s), second 18 passed/2 failed
  (15.17 s); both fixed. Narrow contraction rerun: two passed (1.48 s).
- CPU numerical errors against independent NumPy contractions (absolute):
  nonhybrid amp `7.1054e-15`; hybrid amp `1.7764e-15`, K1LD `8.8818e-16`,
  K2LD `4.4409e-16`, K2L `1.7764e-15`, K21L `4.4409e-16`.
  Fixed-cache integrated FDDS elapsed (no SCF): 0.011980 s nonhybrid,
  0.021344 s hybrid. These tiny random-orbital integral fixtures are not
  representative throughput benchmarks.
  Full physical SVWN/cc-pVDZ He-dimer SAPT case (test time 2.061 s):
  `SAPT DISP ENERGY = -1.0907819605792185e-6 Eh`,
  `SAPT TOTAL ENERGY = -3.513103638665722e-6 Eh`.
- CPU import asserted at
  `build_fdds_cpu/stage/lib/psi4/__init__.py`, core
  `core.cpython-313-x86_64-linux-gnu.so`, version `1.12a1.dev721`.
  Core SHA256:
  `21d4115d948e7e35ad727ffbac1dc7ad5c51b023e80d0e2e9d6c1ba9fbc1b038`.
  Compiler: conda-forge GCC 14.3.0-18. Core links p4_ci's `libmkl_rt.so.2`,
  `libint2.so` (CMake 2.13.1), `libxc.so.15` (CMake 7.0.0), no cuEST/CUDA.
  Detailed artifacts: `build_fdds_cpu/fdds-tests-v3/{cpu.xml,cpu.log,provenance.txt}`.
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
  Corrected CPU gate passed. Retry **13991729** accepted on
  `gts-cs207-chemx / gpu-h200 / embers`, requested two-hour wall time.
  Frozen source pin: `099706616a818041f56072c3b2471d6c04451cc8`.
  At submission: **PENDING (Priority)**; no GPU build/tests executed yet.
  Dry-run prediction was 2026-10-29T19:47:44, not a reliable start ETA.
  Automatic watcher **b8a0442d4** runs the standard `watch_jobs.sh -i 300`
  for this job only (48-hour watcher deadline). If still queued at deadline,
  monitoring must be reattached; do not treat watcher timeout as job failure.
- Suite expectation: **20 executed tests, zero failures/errors/skips** in an
  H200 allocation; `VERIFIED.json` and `COMPLETED` required.
  Thirteen native tests cover formulas, missing keys, symmetric/asymmetric
  empty inputs, host-memory guards/recovery, full integrated FDDS and SAPT;
  seven GPU tests cover both monomers/all five hybrid contributions, 2+1
  occupied blocks, CPU/disabled/huge-threshold/forced controls, error preflight,
  default-quadrature integrated FDDS and full SVWN He-dimer SAPT(DFT).
  Matrix tolerance 2e-11, fixed-cache energy 1e-10 Eh, full SAPT energy
  target 1e-5 Eh; GPU errors/times still pending.
- Integrated fixture timing excludes SCF and reports object/kernel/integration
  timers separately. Small deterministic fixtures are correctness checks,
  not representative throughput benchmarks. **No speedup claim.**
- Coordinator's response checkout/build/job and historical SCF packages were
  not modified, cancelled, rebuilt or retested.
  This includes v2 job 13980976 and v3 job 13991128/pin
  `ae8cedca770e1bc1a7a6091d313cfc6dec8887d9`.

## Next completion steps

On the automatic watcher terminal notification, inspect the job's bounded
logs and `results-13991729/{compiler.txt,import.json,fdds.xml,fdds.log,VERIFIED.json,COMPLETED}`.
Record GPU executed counts, errors, absolute times and source/core/library
provenance here. A skipped GPU suite is a validation failure. Device-capacity
fallback and post-enqueue CUDA fault injection are not runtime-tested yet;
their preflight/lifetime handling has source-review evidence only.
