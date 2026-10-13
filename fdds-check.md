# FDDS cuEST implementation handoff

## v3 terminal audit: FAILED, not a GPU correctness pass

- **13992504** scientific pin `8810da5743cc7350e7620ecb708cdd3e47baa739`,
  harness `e60b32507b3069192032705b1d4161dd38886511`; actual prolog/epilog
  confirm authorized **inferno**, H200 on `atl1-1-03-019-14-0`.
  SLURM job/batch FAILED `143:0`, payload CANCELLED `0:15`, elapsed
  **01:57:57**, start/end Oct 13 09:08:24–11:06:21 Eastern.
- Build/configuration and compiler C/C++ probes succeeded. Exact staged import
  and both git pins were checked. Core `1.12a1.dev726` SHA256:
  `3fe5a446db657ecb82c6557973eab85f08c161729ab16d047df7c16fb9d61c73`.
  Actual resolved libraries: libcuest 0.2.2 (conda 0.2.2.2),
  CUDA runtime **13.3.29**, cuBLAS **13.6.0.2**, Libxc 7.0.0.
  Site nvcc was 12.6.1, but CMake detected CUDA toolkit major 13 and import
  ldd resolved the conda CUDA 13 libraries; do not label this CUDA-12 runtime.
  Device NVIDIA H200, driver 615.71.09.
- **51 completed progress outcomes: 50 passes, 1 failure, 0 reported errors
  or skips; 52 tests started, final case unfinished.** There is no finalized
  JUnit: `fdds.xml`, `VERIFIED.json`, `COMPLETED` are all absent.
  Counts are reconstructed only from pytest progress tokens, excluding the
  literal F in printed FDDS lines, by local `audit.py`; no synthetic JUnit.
  The failed completed case is unequal nonhybrid GPU contributions; traceback
  was not emitted before interruption, so the failing assertion/cause is not
  established. Final SAPT GPU case entered Psi4's debugger-wait fault handler
  (PID3340286); its first CPU control returned, but no GPU SAPT result appeared.
  A later TERM stopped the job near its pre-wall-limit signal; accounting says
  FAILED rather than TIMEOUT. No crash stack is available; do not attribute
  this to FDDS or cuEST lifecycle without diagnosis.
- Completed equal nonhybrid and equal/unequal hybrid GPU contribution samples
  print zero CPU/GPU errors. Completed integrated equal/unequal nonhybrid/hybrid
  samples print zero energy differences and actual cuEST FDDS timers.
  Integrated CPU/forced-GPU absolute elapsed seconds: equal nonhybrid
  0.011079/0.012633, equal hybrid 0.016547/0.021600, unequal nonhybrid
  0.010599/0.012759, unequal hybrid 0.016543/0.021534.
  These are tiny correctness fixtures, **not throughput/speedup evidence**.
  After-real-JK HOST pointer mode and full SAPT GPU correctness remain unproven.
- Preserved local evidence:
  `build_fdds_cpu/gpu-13992504/{accounting.txt,accounting-detailed.txt,remote-evidence.tar.gz,audit.py,audit.json,logs/,results-13992504/}`.
  Audit JSON includes hashes of every copied result/log. Source/test/harness
  checkouts are clean and retained unchanged; no retry/new allocation submitted.
  Atomic `fdds-validation-13992504-terminal.json` records this failed outcome.
  Existing pre-review13991729 and coordinator13991128 remain untouched.

## Authorized v3 GPU submission (pending, not validated)

- Subsequent explicit user approval selected **v3 only** for paid inferno.
  Same job **13992504** was updated to `QOS=inferno` with the unchanged
  `TimeLimit=02:00:00`, verified by scontrol; state was still PENDING.
  No duplicate submission, cancellation, source/payload change or environment
  mutation. Jobs 13991729 and 13991128 remain untouched. Original submission
  below was free embers; final audit must check the actual inferno prolog.
  Remote QoS-change evidence: run-root `inferno-switch-accounting.txt`.
  Atomic submission marker updated with effective and initial QoS; watcher
  bba95ab46 continues on the same job.
- Final focused read-only review reported no new correctness defect. Exactly
  one new allocation was submitted: **13992504**, `gts-cs207-chemx`,
  `gpu-h200`, explicit free `embers`, one H200/8 CPUs/64 GiB/two hours.
  SLURM accepted it as PENDING; no GPU execution evidence yet.
- Immutable scientific source/tests:
  `8810da5743cc7350e7620ecb708cdd3e47baa739`, isolated checkout
  `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-validation-20261012-v3`,
  own `build_fdds_validation`. Required **52 executed tests, zero skips,
  failures or errors** (41 native + 11 GPU), plus VERIFIED.json/COMPLETED.
- Separately git-pinned harness:
  `e60b32507b3069192032705b1d4161dd38886511`, checkout ending
  `psi4.fdds-validation-20261012-v3-harness`. This fixes only job activation:
  installed conda compiler hooks read unset HOST/CONDA_BACKUP_GXX under nounset.
  Nounset is relaxed only for conda activation, then restored; GCC_ROOT is
  unset after module load and activation. No shared environment was changed.
  The scientific checkout remains exactly at 8810da5743.
- Live submission headroom **37/50**; unique job-name/comment/run duplicate
  checks passed. Dry-run accepted gpu-h200; absolute cc1/cc1plus frontends
  resolved in p4cuest_sapt. Compile/link/run probes repeat inside allocation.
- Run directory:
  `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261012-validation-03`.
  Audit `results-13992504/{commit.txt,harness-commit.txt,compiler.txt,import.json,fdds.xml,fdds.log,VERIFIED.json,COMPLETED}`
  and configure/build logs after terminal notification, including actual QOS
  prolog, staged import/core/library hashes, executed GPU route/error/timings.
- Automatic watcher **bba95ab46**, standard watch_jobs.sh at 300 s, 48-hour
  watcher deadline. Submission marker `fdds-v3-submission.json` is atomic.
  No GPU numerical/speedup claim. Existing FDDS **13991729** and coordinator
  **13991128**, their frozen checkouts/builds/dependencies, remain untouched.

## CPU review fixes complete (not GPU validation)

- Fix commits: `c90fc65bcc` (checked one/five empty-output budgets and recovery),
  `efde54e03b` (exported helper null/tensor/shape/overflow/budget preflight),
  `8810da5743` (unequal nonempty monomers and checked cuBLAS HOST pointer mode).
  The DFHelper shape-query change is metadata-only: persisted/disk tensors are
  queried through the storage registry, even after clearing transformations.
- Latest CPU-tested source/tests:
  `8810da5743cc7350e7620ecb708cdd3e47baa739`.
  Own Release CPU stage rebuilt; **51 passed, 0 failures/errors/skips, 31 GPU
  cases deliberately deselected, 18.48 s** (41 FDDS/SAPT + 10 native CPSCF).
  Same outside-source-root command as below, with `--junitxml=full-cpu.xml`.
  Evidence: `build_fdds_cpu/review-fixes/{full-cpu.xml,full-cpu.log,full-provenance.txt,full-build.log}`.
  Version `1.12a1.dev726`; asserted staged import path unchanged. New core SHA256:
  `5c199ad6f2e92c489ba13ddaa334a6215d400eee397ecfa530781907ce657069`.
- Maximum independent native contraction error (both monomers, equal/unequal):
  **7.105427357601002e-15**. Unequal A=(2 occ,8 vir), B=(3 occ,7 vir).
  Native integrated FDDS samples: equal nonhybrid/hybrid 0.012004/0.038873 s;
  unequal 0.011409/0.020241 s. No SCF in those timings; no speedup conclusion.
  Physical SVWN He-dimer dispersion/total energies remain exactly as below.
- Review regressions: budget red 12 failed then green 12 passed (2.10 s);
  unguarded helper probe entered Psi4's debugger-wait fault handler and timed
  out, guarded helper/oracle subset 14 passed (2.10 s); unequal subset 4 passed
  (1.65 s). Earlier logs/XML/provenance are retained unchanged.
- HOST pointer mode is checked without mutating handle state; actual after-JK
  execution and new CUDA compilation remain **unverified** in this CPU-only build.
  Capacity-driven reduction/fallback and post-enqueue fault injection still
  lack runtime coverage. No user-facing fault-injection option was added.
- Existing GPU job **13991729** remains **PENDING (Priority)**, no allocation
  assigned at the latest status check, frozen at pre-review pin
  `099706616a818041f56072c3b2471d6c04451cc8`; its 20-test suite will **not**
  validate these review fixes. Job/checkout/dependencies are untouched.
  At CPU handoff, prospective v3/run-03 templates required 52 executed tests
  (41 native + 11 GPU) and were not yet submitted. The subsequent authorized
  isolated v3 submission is recorded above; no existing job was cancelled.
- `fdds-review-fixes-complete.json` is generated atomically only after final
  handoff commit and the green CPU gate; it certifies CPU review-fix completion
  only, not GPU correctness, performance, or merge approval.

## Branch and implementation

- Branch: `saptdft_cuest_disp`, based on response commit
  `4f47db835ad50ee4ed2b7b3005c99cb5cd10da35`. No merge.
- Implementation: `9fd3a2aa620ffe7bf96522dafde18b0999000d40`.
- Allocation compiler isolation: `738aa3efc9` (module GCC_ROOT unset, explicit
  conda activation, absolute cc1/cc1plus checks, C/C++ compile/link/run probes).
- Exception/partial-empty safety fixes: `5f63ef56c9`.
- Test-accessor fix: `be4d7bba6e` (`Matrix.nrow()` is not a Python API;
  use `Matrix.np.shape[0]`).
- Pre-review CPU-tested source/tests and existing GPU pin:
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

## Preserved pre-review evidence (GPU pending, not a GPU correctness pass)

- Static checks passed: `git diff --check`, `bash -n` for validation scripts,
  Python syntax compilation of `tests/pytests/test_cuest_fdds.py`.
- Own local CPU build: `build_fdds_cpu`, dependency environment
  `/home/awallace43/miniconda3/envs/p4_ci`, Release, Ninja, `ENABLE_cuEST=OFF`,
  `ENABLE_Einsums=OFF`, `BUILD_SHARED_LIBS=ON`, 12 build workers.
  Logs: `build_fdds_cpu/logs/fdds-cpu-{configure,build,rebuild*}.log` (untracked).
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
- Frozen v2 validation payload: `devtools/benchmarks/fdds_validate.sh`, batch
  script `fdds_validate.sbatch` at pin 099706616a (current branch templates
  prepare a separate v3 instead). One H200, 8 CPUs, 64 GiB, two hours, explicitly free
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
