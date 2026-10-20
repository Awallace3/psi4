# FDDS cuEST implementation handoff

## Current result: isolated GPU validation PASSED

- **Implemented and validated; no merge back.** Real FP64 occupied-block FDDS
  contractions use persistent device sums on the shared cuEST stream, with
  existing opt-in/threshold routing, native fallbacks, checked memory budgets,
  cleanup and runtime-error propagation. Equations and QCVariables unchanged.
- **Job14184910**, explicit free **embers**, H200/4 CPUs/16 GiB/15 minutes:
  job/batch/payload **COMPLETED 0:0**, elapsed **54 s** (payload51 s).
  **Full FDDS suite57 passed, zero failures/errors/skips, 18.596 s**:
  46 native/control cases plus **11 GPU cases**, all actually executed.
  Preceding focused checks:8 passed4.140 s plus real SAPT GPU test1 passed
  11.254 s. Thus66 executions including repeats, **57 unique tests**, not66.
  Both GDB inferiors exited normally without signal/breakpoint hits.
  Exact JUnit counts, `FOCUSED_VERIFIED.json`, `VERIFIED.json`, `COMPLETED`,
  actual timers/block assertions and pinned staged import were audited.
- Scientific binary/source:
  **`9d4b1cd4aab718dec1f7b831a53dd54d68f2a609`**.
  Test driver (budget/output-helper regressions):
  **`3b21adb662f47fc4b89f1cbb89a44592f7a1d10f`**.
  Replay harness:
  **`ce2ab6b38d157b3fb805708bf3da74a19e80c208`**.
  All three remote tracked trees verified clean and pinned after completion.
  No production-source/build/installed-Python difference between binary and
  driver pins; the integrated GPU stage was reused **read-only**, not rebuilt.
- GPU stage:
  `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-integrated-20261017-v1/build_fdds_validation/stage/lib/psi4`.
  Psi4 **1.12a1.dev802**, core SHA256
  `68e34452afa80002d4bb7fb1b56297e19389617fcb3ee3b926abc0b5fcead0b7`.
  Release GPU build from job14181925; successful compiler frontend probes
  after job-local GCC_ROOT isolation. Shared dependency environment unchanged.
  Actual runtime **cuEST0.2.2** (conda libcuest/libcuest-dev0.2.2.2),
  **CUDA13.3.29/cuBLAS13.6.0.2**, H200 driver615.71.09.
  `import.json` records exact resolved library paths and SHA256 hashes;
  the site nvcc12.6.1 is **not** the resolved runtime version.
- Largest full-suite contribution/reference absolute error:
  **7.105427357601002e-15**; all four integrated nonhybrid/hybrid,
  equal/unequal CPU/forced-GPU dispersion differences printed **0 Eh**.
  Real SVWN/cc-pVDZ He-dimer with native supported XC response:
  CPU dispersion **-1.090781960579051e-6 Eh**,
  forced-GPU **-1.0907819605790383e-6 Eh**, difference
  **1.2705494208814505e-20 Eh**.
  CPU total **-3.5131036459929232e-6 Eh**, forced-GPU total
  **-3.5131036441000883e-6 Eh**, difference **1.892834881076031e-15 Eh**.
  Across disabled/threshold/forced controls all six SAPT QCVariable errors
  are at most **6.6574573662142395e-15 Eh**.
  Forced real SAPT passed the20-call FDDS timer assertion after actual cuEST
  J/K builds, which also exercised the shared handle HOST-mode check.
- FDDS-only elapsed seconds, same source/H200 allocation, excludes SCF:

  | Fixture | Disabled CPU | Huge-threshold control | Forced GPU | GPU contraction timer |
  |---|---:|---:|---:|---:|
  | Equal nonhybrid | .011230 | .010759 | .012674 | .000834985 |
  | Equal hybrid | .016645 | .016663 | .021933 | .002517654 |
  | Unequal nonhybrid | .010739 | .010768 | .012748 | .000845762 |
  | Unequal hybrid | .016821 | .016780 | .021904 | .002529294 |

  Tiny transfer-dominated correctness fixtures, under GDB; **no representative
  throughput or speedup claim**. GPU timer includes transfers/preparation.
- Latest CPU-only gate **165 passed, zero failures/errors/skips,179.023 s**,
  including46 FDDS and67 CPSCF. Own Release/Ninja stage, cuEST/Einsums OFF,
  scientific9d4/test-driver3b21, outside-source import checked. CPU core SHA256
  `5b2825a1dd99a8f7e40c5a24ec2b14df2c1f7436b8d1197ea7de11c500a48358`.
  Evidence `build_fdds_cpu/integrated-9d4b1cd4aa/native-output-fixed{.xml,.log,-provenance.json}`.
- Local GPU evidence and reproducible audit:
  **`build_fdds_cpu/gpu-14184910/{audit.py,audit.json,accounting.txt,remote-evidence.tar.gz,results-14184910/}`**.
  Run `python3 build_fdds_cpu/gpu-14184910/audit.py` to recheck counts,
  provenance, native exits, timers and errors; audit records artifact hashes.
  Remote run
  `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261019-integrated-focused-02`.
  Watcher b07d6597e completed; no new allocation needed.
- Remaining limitations: device-capacity-driven reduction/rejection/fallback
  and post-enqueue CUDA/cuBLAS fault injection have **not** been runtime-tested.
  Unsupported layouts/symmetry, empty/small/native routes and host-budget
  exceptions/recovery are covered; runtime errors are not silently retried.
  Remaining CPU work: integrals/DF preparation, ALDA kernel, hybrid QR/X/Y,
  coupled susceptibility products/solves, quadrature and exchange-dispersion.
  This FDDS work adds **no MGGA/VV10/GPU-XC response kernels**. End-to-end
  tests deliberately use `CUEST_XC=false`; upstream LDA/GGA response support
  was integrated separately, not established by this FDDS GPU suite.
- Earlier failed jobs/stages/evidence and coordinator worktrees/jobs remain
  immutable. Fixture-budget and test-output-lifetime causes are confirmed and
  fixed, not attributed to FDDS equations or JK lifecycle. Old13991729 was
  not cancelled or modified; its current scheduler state was not re-queried
  in this terminal audit. No local GPU success, environment mutation, paid
  retry, auto-merge or throughput claim.

## Historical record (superseded status snapshots below)

The remaining sections preserve earlier evidence and decisions. Their
“pending”/“blocked” labels describe those earlier pins, not current14184910.

## Same allocation extended to gated full-suite validation

- User explicitly requested using the allocation for the full FDDS suite once
  focused checks pass, and authorized brief hold/release of **14184910** to
  update only its unstarted replay harness. That update succeeded; job ID,
  scientific binary9d4b1cd4aa, test driver3b21adb662, 15-minute limit and free
  embers/H200 route are unchanged. No new submission/cancellation.
- Replay harness advanced from e124e5cfba to
  **ce2ab6b38d157b3fb805708bf3da74a19e80c208** while held; clean tree and
  no scientific/test/build-code changes were verified before release.
  The previous harness commit remains available in git. Transition and
  before/after scheduler snapshots are recorded in run02.
- Execution plan: **9 focused passes first**, then **all57 FDDS tests** in
  the same allocation under bounded GDB (300 s, 15 s kill grace).
  Full success requires exact57 executed/no skips/failures/errors and normal
  GDB inferior exit. `FOCUSED_VERIFIED.json` is kept separately; only a verified
  full run writes final `VERIFIED.json` and allows `COMPLETED`.
  There are **66 total test executions including repeats**, not 66 unique tests.
  Six synthetic gate checks accepted normal completion and rejected partial
  counts, skipped/failed tests, native faults and abnormal exits; not GPU data.
- Effective post-release state PENDING, QOS embers. Existing watcher
  **b07d6597e** continues on the same job. No numerical/full-suite or speedup
  conclusion before terminal accounting/provenance/artifact audit.

## Diagnosis complete; corrected SAPT GPU confirmation still pending

- Final CPU gate with corrected test driver
  **3b21adb662f47fc4b89f1cbb89a44592f7a1d10f** and integrated scientific
  binary9d4b1cd4aa: **165 passed, zero failures/errors/skips, 179.023 s**
  (46 FDDS, 67 CPSCF, 6 SAD, 6 lifecycle, 8 basis-parser, 32 grid-switch).
  Evidence `build_fdds_cpu/integrated-9d4b1cd4aa/{native-output-fixed.xml,native-output-fixed.log,native-output-fixed-provenance.json}`.
  The unchanged CPU core SHA256 is explicitly rechecked in provenance.
- Integrated focused job **14181925** ran and failed on its own, no
  cancellation: free embers/H200, **6m46s**, exit1. Own integrated GPU build
  succeeded. **8 contribution/budget tests passed**, zero errors/failures/
  skips, **3.850 s**; all four equal/unequal nonhybrid/hybrid comparisons
  printed zero CPU/GPU matrix error and passed actual block/timer routing
  assertions. The final SAPT case then SIGSEGV'd in Printf with the old
  helper; no SAPT XML/VERIFIED/COMPLETED, so not a focused-suite pass.
  Preserved evidence `build_fdds_cpu/gpu-14181925/`.
- GPU binary pin9d4b1cd4aa, original focused harnessa6233fffac; staged core
  SHA256 `68e34452afa80002d4bb7fb1b56297e19389617fcb3ee3b926abc0b5fcead0b7`,
  version1.12a1.dev802, cuEST0.2.2 (conda0.2.2.2), CUDA runtime13.3.29,
  cuBLAS13.6.0.2. These are correctness fixtures, no speedup claim.
- Corrected replay uses existing integrated GPU stage **read-only**; scientific
  C++/installed Python/build configuration diff9d4b1cd4aa..3b21adb662 is empty.
  Test driver remains frozen3b21adb662; replay harness
  **e124e5cfba24a3a046aee4e5f055d1cdfc92dbfa** verifies the binary's exact
  source pin and core hash, records linked library hashes, and never rebuilds
  or overwrites it. New run02, H200/4 CPUs/16 GiB/15 minutes/free embers.
  Staging/dry-run passed; **14184910** accepted on explicit free embers/H200,
  4 CPUs/16 GiB/15 minutes. Live headroom30/50, duplicate/pin/clean-tree checks
  passed, effective QoS confirmed embers, initial state PENDING.
  Run `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261019-integrated-focused-02`;
  watcher **b07d6597e**, 300 s, 48-hour watcher deadline.
  Neither old job nor stage was cancelled or modified. Required nine actual
  passes/no skips plus normal GDB inferior exit, VERIFIED.json/COMPLETED and
  pinned import/core/library provenance; no full suite or speedup claim.
- Atomic `fdds-gpu-diagnosis-complete.json` reports diagnosis/CPU-fix completion
  with outcome **blocked**, not a full GPU pass. Remaining gates: corrected
  SAPT GPU test, full57-case FDDS suite, capacity-driven reduction/fallback and
  post-enqueue fault-injection runtime coverage. All prior evidence/pins remain
  immutable; no coordinator dirty work copied and no merge back.

## Native crash diagnosed: test helper closed the global output stream

- Recovered diagnostic **14006185** completed Oct16 on free embers in
  **27 s**. Contribution process exited1, exact fail-fast traceback confirms
  the unequal/nonhybrid CPU-reference memory exception; JUnit **1 failed**,
  2.956 s. Separate GDB process exited0 **after catching SIGSEGV** in
  `psi::PsiOutStream::Printf`; no SAPT JUnit exists, so this is not a pass.
  The archived stage core SHA256 exactly matches the failed13992504 build.
  Evidence `build_fdds_cpu/diagnostic-14006185/{accounting.txt,remote-evidence.tar.gz,logs/,results-14006185/}`.
- Minimal CPU-only reproduction on the integrated binary: open a native
  outfile, close it, call public `psi4.set_num_threads(1)`. GDB catches the
  same SIGSEGV/Printf frame. Keeping the stream open exits normally.
  No SCF, J/K, FDDS or GPU call is required. This confirms an invalid
  test-helper output lifetime, not a demonstrated backend scientific bug.
  Evidence `build_fdds_cpu/integrated-9d4b1cd4aa/{reproduce_closed_output.py,minimal-open-gdb.log,minimal-closed-gdb.log}`.
- `_run_saptdft_fdds` now opens output before any setup messages and restores
  the caller's previous native stream in `finally`. New repeated-control
  regression went red (**1 failed**, 3.54 s) against the old helper; fixed
  native SAPT + sequential-control subset **2 passed**, zero failures/errors/
  skips, **7.233 s**. Evidence `output-{red,green}.{log,xml}` in that directory.
  Full prospective FDDS suite is now **57**, focused GPU suite still **9**.
- User explicitly authorized replacing **14181925 only if still queued**,
  after the CPU gate passes; if running it must remain untouched. No
  cancellation has yet been issued. Prospective corrected focused payload
  uses a new v2 checkout/run02; original v1 scientific/harness pins and
  failed/diagnostic evidence remain frozen.

## User-authorized upstream integration and new CPU gate

- User authorized pulling committed `saptdft_cuest` work into this FDDS
  branch. Merge **9d4b1cd4aab718dec1f7b831a53dd54d68f2a609** integrates
  upstream **ca21ef8645a73263281480454b730085506adca5**, including committed
  SAD/lifecycle and LDA/GGA cuEST-grid XC response work. No dirty coordinator
  edits were copied, and no merge back into the coordinator branch occurred.
  FDDS scientific C++ and fixture corrections remain intact. One option-doc
  conflict was resolved retaining both routing descriptions.
- Own CPU-only Release stage rebuilt successfully, cuEST/Einsums OFF.
  **164 passed, zero failures/errors/skips, 209.808 s**:
  45 FDDS, 67 native CPSCF, 6 SAD, 6 lifecycle, 8 basis-parser, 32 grid-switch.
  Evidence `build_fdds_cpu/integrated-9d4b1cd4aa/{build.log,native.xml,native.log,provenance.json}`.
  Actual outside-root staged import asserted, version `1.12a1.dev802`,
  core SHA256 `5b2825a1dd99a8f7e40c5a24ec2b14df2c1f7436b8d1197ea7de11c500a48358`.
  No CUDA/cuEST linkage in this CPU gate.
- Focused GPU harness **a6233fffac7ce1311f8f3a0164adea55708f6c1f**
  prepares an entirely new source/build/run at scientific pin9d4b1cd4aa.
  Only 8 contribution/budget tests plus the final SAPT GPU test under bounded
  GDB (**9 targeted tests**, not the full 56-case suite). Separate processes
  preserve the crash stack even if contributions fail. Gate checks both
  subprocess statuses, exact JUnit counts/no skips, and normal GDB inferior
  exit; SIGSEGV/attachment-breakpoint output cannot become a false pass.
  Five synthetic artifact-gate checks passed (normal exit accepted; native
  signal, attachment breakpoint, skip and timeout rejected); these are
  harness checks only, not GPU correctness evidence.
- Staging and dry-run passed; exactly one new job **14181925** was accepted
  on explicit free `embers`/H200, 8 CPUs/64 GiB/one-hour build+focused-validation
  limit. Live headroom 32/50, unique job-name/comment/run checks passed;
  effective submission QoS confirmed embers, state PENDING.
  Source `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-integrated-20261017-v1`,
  separate harness checkout with `-harness` suffix. Run directory
  `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261017-integrated-focused-01`.
  Watcher **b945ab472**, standard 300 s interval, 48-hour watcher deadline.
  No full GPU rerun.
  Frozen previous stages, failed13992504 evidence, and diagnostic14006185
  scientific/driver pins remain immutable. Native crash is not yet established
  fixed by upstream integration; targeted GPU evidence is still required.

## Confirmed fixture correction (CPU evidence; GPU diagnosis pending)

- Local direct replay of the original unequal/nonhybrid test stopped in
  CPU reference construction with the FDDS host-memory exception, before
  any GPU comparison. This independently confirms the B-only-budget defect:
  262 available doubles versus A's required 276. Evidence
  `build_fdds_cpu/diagnosis-local/fixture-budget-red.log`.
- New real-interface regression first went red (**2 passed/1 failed**, fail-fast,
  1.65 s), then all four equal/unequal nonhybrid/hybrid cases passed
  (**4 passed, zero errors/failures/skips**, 1.581 s) after the fixture fix.
  Evidence `build_fdds_cpu/diagnosis-local/budget-{red,green}.{log,xml}`.
- Comparison budget now satisfies both monomer preflights and preserves B's
  two-block partial tail. Block assertions derive from monomer dimensions and
  the host occupied-block contract: unequal nonhybrid is **9** GPU blocks over
  three frequencies, not 12; other cases remain 12. Numerical tolerances
  unchanged. New four-case native budget regression also checks independent
  contraction references; prospective full GPU suite expands to **56**.
- Full expanded native FDDS+CPSCF gate at test-driver pin
  `2a6b3b63da2bee8b6718b81f18e6579c5e5e4346`: **55 passed, zero failures,
  errors or skips, 18.239 s** (45 native FDDS + 10 native CPSCF; GPU cases
  deliberately deselected). Evidence
  `build_fdds_cpu/diagnosis-local/{full-native.xml,full-native.log,full-native-provenance.json}`.
  Scientific C++ unchanged, so the existing CPU-only binary at scientific
  pin8810da5743 was used without rebuild; core SHA256 and native-only ldd
  provenance are recorded explicitly. This is not GPU validation.
- Frozen diagnostic job14006185 still uses original source/tests to obtain the
  requested GPU traceback/GDB stack; it is not repointed to these changes.
  Native crash cause remains unknown, no full GPU rerun submitted, and no
  diagnosis-complete marker or full-GPU pass claimed yet.

## Focused diagnostic allocation (pending)

- Authorized one short explicit free-embers job **14006185**, H200/4 CPUs/
  16 GiB/15 minutes, account gts-cs207-chemx. Dry-run accepted gpu-h200;
  live headroom was 49/50 and unique name/comment/run checks passed.
  Effective submission QoS verified `embers`, initial state PENDING.
- Scientific/build pin remains `8810da5743cc7350e7620ecb708cdd3e47baa739`;
  existing v3 package is used read-only without rebuild. Diagnostic driver
  pin `da356228a300f80d323524d29d97a3435888d4c1`, distinct from original
  build harness `e60b32507b3069192032705b1d4161dd38886511`.
- Driver checkout: `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-diagnosis-20261013-v1`;
  run: `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261013-diagnosis-01`.
  Separate processes select only unequal nonhybrid contributions (120 s)
  and the SAPT GPU case under batch GDB (240 s), each with 15 s kill grace.
  GDB stops SIGSEGV and requests the debugger-attachment breakpoint, then
  captures thread backtraces. GDB exit zero never certifies a test pass.
  Faulthandler enabled; scratch and pytest temp outputs persist in run.
- Core/import and actual linked-library hashes will be recorded inside
  allocation and checked against the preserved failed core SHA256.
  Watcher **bd8e1e8fa**, 300 s intervals, 48-hour watcher deadline.
  No test/scientific change yet; diagnosis-complete marker waits for evidence
  and subsequent confirmed fixes/CPU gates. No full suite or paid retry.

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
