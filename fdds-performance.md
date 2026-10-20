# FDDS performance campaign

User authorized inferno for this campaign. No changes to shared environments,
old frozen source/build/run directories or other agents' jobs.

## Benchmark contract

- At most89 atoms. Available timing-suite keys below this cap are **peptide
  (24 atoms), nanotube (42), protein83 (83)**. The supplied suite has no
  protein89 key; protein157 and pla15_3pe2 are excluded.
- Real PBE0 monomer SCFs in the common dimer-centered basis, fixed GRAC .136.
  Primary6-31+g**, JK fitting def2-universal-jkfit, dispersion fitting
  aug-cc-pVDZ-ri, pure spherical functions, FP64. Native XC75x302;
  E/D convergence1e-9/1e-8. Hybrid FDDS kernel enabled, unchanged ten-point
  quadrature/lambda .3/rho cutoff/pseudoinverse threshold.
- Prepare physical occupied/virtual coefficients, energies and densities once
  and save immutable caches with geometry/options/orbital/core hashes.
  SCF preparation is **not included** in FDDS timing.
- Replay the real `sapt_mp2_terms.df_fdds_dispersion()` interface, including
  object/integral/DF preparation, hybrid QR/X/Y, ALDA kernel and full coupled/
  uncoupled frequency integration. Record native timers and individual
  pseudoinverse walls. This is FDDS, not the subsequent MP2 exchange-dispersion
  correction and not total SAPT time.
- Fresh processes for native and GPU controls. CPU baseline must never reuse
  a CUDA-initialized process: some native eigensolvers select CUDA based on
  shared handle initialization rather than the current USE_CUEST value.
- Final speedup comparisons require the same physical cache/options and
  source, with CPU/GPU controls on one GPU hardware allocation; the initial
  CPU job is a separate profiling baseline, not a cross-node speedup claim.
  Keep thread/memory settings explicit. Require actual GPU route/timer proof,
  finite energies and no worse than1e-5 Eh agreement (tighter when measured).

## Initial baseline

- Harness `devtools/benchmarks/fdds_benchmark.py`; separate bounded CPU job
  `fdds_cpu_baseline.sbatch` prepares/replays the three systems sequentially.
  Own read-only validated scientific stage9d4b1cd4aa, exact core hash checked;
  USE_CUEST=false and fresh processes throughout the CPU baseline.
- Intended request:24 CPUs/160 GiB/4h/inferno, Psi4 memory96 GiB, per-system
  75-minute bound and30-second kill grace. No GPU requested.
- Small physical-water cache smoke test exposed a Python Matrix accessor
  mismatch before timing; fixed to use NumPy shape. Original failure/cache
  preserved. Corrected real hybrid replay passed:56 AO/236 auxiliary,
  occupied5/5, virtual51/51, **.689684 s FDDS**,22 pseudoinverses
  **.231785 s**, zero GPU calls. Coupled/uncoupled energies
  **-.0023791261270024056 / -.0031182443982617552 Eh**.
  This is harness validation, not a representative speedup measurement.
  No production scientific changes yet.
- Prospective matched GPU profiling job uses fresh-process **CPU/GPU/GPU/CPU**
  controls,8 threads/96 GiB on one H200 allocation; cached SCFs unchanged.
  Its completion gate checks exact cache/source/core/options/host/thread/memory
  identity, actual routing and1e-5 Eh errors. It is not submitted yet.
- Source-based candidate bottlenecks (not yet benchmark conclusions): hybrid
  X/Y pairwise DGEMMs, QR,22 dense pseudoinverses, and density/ALDA setup.
  Choose the next seam from measured profiles; preserve exact equations and
  1e-13 singular-value cutoff rather than replacing pseudoinverse with a solve.

## Submitted baseline and first measured bottleneck

- **14198087** accepted on **gts-cs207 / cpu-small / inferno**,24 CPUs/160 GiB/
  four hours. Effective QoS confirmed by scontrol. Worst-case credit$.79;
  monthly-account available balance$29.51 before submission. Payload pinned
  **1a5a95168f7cb4305c4a86a04d795d0274d859a9**, scientific binary9d4/hash68e344...
  Local staging/smoke, geometry checksum, dry-run and duplicate checks passed.
  Watcher **bd6b34709**. No job cancellation/modification.
  Remote run `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-performance-01`.
  Scheduler dry-run predicted an overnight delay; local profiles run in
  parallel, not substituted into any cross-node speedup ratio.
- Local physical peptide profile:250 AO/1042 auxiliary,20 occupied/230 virtual
  per monomer,8 threads/16 GiB, **38.216089 s FDDS**. Object16.948329 s,
  QR.641465 s, **X1.947119 s/Y6.633875 s**, ALDA3.620404 s,
  integration17.203429 s;22 pseudoinverses5.755491 s (inside integration).
  Coupled/uncoupled FDDS **-.006194210929276167 / -.009442035537422072 Eh**.
  Cache and profile at `build_fdds_cpu/perf-peptide-{cache,cpu}/`;
  native source9d4/core5b2825..., not an optimized GPU measurement.
- A local protein83 native profile (16 threads/64 GiB) removed queue latency
  from diagnosis. Its final atomic result was later recovered:
  **3591.575812 s FDDS**,22 pseudoinverses192.668927 s,
  coupled/uncoupled **-.007013972054413143 / -.011403077774448968 Eh**,
  old scientific9d4/core5b2825..., no GPU work. Background terminal metadata
  failed with ENOSPC after the result was written; that transport/status
  failure is not a failed scientific calculation. Local artifact
  `build_fdds_cpu/performance-baselines/protein83.json`.
- Local physical nanotube profile completed:42 atoms,548 AO/2240 auxiliary,
  occupied8/78 and virtual533/463 (SCF removes linear dependencies),
 8 threads/32 GiB, **807.463674 s FDDS**. Object427.663180 s,
  X108.110879 s, Y179.639530 s, QR8.744222 s, kernel37.240188 s,
  integration340.248031 s,22 pseudoinverses67.997986 s (inside integration).
  Coupled/uncoupled FDDS **-.02387909012126487 / -.04431040718821849 Eh**,
  zero GPU calls. This profiles old scientific9d4/core5b2825..., not the new
  candidate; no cross-node or cross-source speedup is inferred.
  Compact result `build_fdds_cpu/performance-baselines/nanotube.json`;
  original `/fastscratch/awallace43/fdds-performance-20261020/nanotube-cpu/`.

### Completed cluster CPU baseline

- Job **14198087** and batch completed0:0 in **35m40s**, all three physical
  cases actually finished. Audited VERIFIED/COMPLETED, source/core/import,
  fresh CPU controls with0 GPU calls, finite energies, exact metadata/orbital
  cache checksums and24-thread/96 GiB settings.
- Same CPU node `atl1-1-02-007-27-1.pace.gatech.edu`, scientific9d4,
  core68e344..., harness1a5a95168f. **FDDS-only** (SCF excluded):
  **peptide20.445979 s; nanotube240.152734 s; protein83 1592.475991 s**.
  Their22 pseudoinverses total3.754966/19.806817/87.188779 s, respectively.
- Protein83 object1099.025057 s, **X189.203824/Y489.876410 s**,
  QR27.420139 s, kernel86.499547 s, integration403.772981 s.
  Coupled/uncoupled **-.007013974490528032 / -.011403077901569983 Eh**.
  Thus the chosen X/Y seam accounts for679.080234 s of this actual baseline.
- Evidence `build_fdds_cpu/performance-cpu-14198087/` includes compact JSONs,
  full VERIFIED/provenance/completion, and audited counts. These use separate
  SCF caches from the local profiles; do **not** compare across nodes/caches/
  sources to claim a GPU speedup. The recovery still runs its own matched
  CPU/GPU controls on the same new scientific stage/allocation/cache.

## Candidate: persistent hybrid X/Y block contraction (not yet validated)

- `CuestFDDSExchange` uploads the outer integral block once and each inner
  integral/raw/orthogonal block once. Three exact FP64 cuBLAS products per
  occupied-pair/occupied-virtual pair; two output tensors remain resident
  across all inner blocks. Only completed outer outputs are downloaded.
  Existing DFHelper reads/writes, tensor order and equations are preserved.
- Shared cuEST stream/handle, actual HOST pointer-mode check, checked device
  capacity (80% free), shrinking caller-budgeted blocks only. Stream drains
  before host scratch can be reused and on every enqueued-error unwind;
  runtime failures propagate without native retry. Native/no-cuEST/small/
  unsupported-shape/capacity paths remain available; no new user option.
- Device doubles: `q*[bo*(k+2v)+bi*(v+2k)] + v*k`, with k=v for X,
  k=nocc for Y. Host memory budgets unchanged. Row limits and size arithmetic
  are checked. Y's former zero virtual block when nvir<nocc is corrected
  within the original six-o memory bound; dense-occupied regression added.
- New real-interface tensor tests compare X/QX/Y/QY against independent
  contractions of transformed integral tensors: equal/unequal/dense-occupied,
  whole/blocked host budgets, disabled/huge-threshold/forced GPU controls.
  Prospective full FDDS suite **69** (52 native/control,17 GPU cases);
  focused gate20+real SAPT1, then full69 under bounded GDB in one allocation.
  Collection count verified; runtime validation pending.
- Read-only safety review of the new backend is running. No new GPU numerical
  pass or performance speedup is claimed at this point.
- Red regression confirmed against the old9d4 CPU core: the dense-occupied
  whole-block case failed in1.800 s with DFHelper's axis0 bounds error
  (`rrR`, index18446744073709551615). This confirms the zero-sized virtual
  block problem rather than a timeout hypothesis. Failure XML is preserved at
  `build_fdds_cpu/perf-exchange-tests/dense-red.xml`.
- Independent equal/unequal X/Y reference checks passed against the old core:
  two executed tests, no failures/errors/skips,1.806 s. The new candidate's
  expanded171-test native gate remains separate from this reference check.
- Candidate CPU core was built at8dc714e4d30c272cc09bcbec5a348d3317518082,
  SHA256`8ee2d4357a56a7a1d73334a939ce88ff75999138b4f2e745d360a962717ff601`;
  no-cuEST release build in the dedicated `fdds-opt-cpu-20261020-v1` worktree.
  GPU optimization validation must build its own separate stage.
- Optimization recovery harness uses the verified H200 limit of8 CPU threads/one GPU,
 128 GiB allocation/96 GiB Psi4 memory/eight hours. It builds the frozen
  scientific source, requires21 focused plus69 full executed passes, then
  replays matched CPU/GPU/GPU/CPU controls on all three physical caches.
  The separate baseline CPU job is not a dependency of this allocation.
- CUDA build preflight explicitly pins the conda nvcc/toolkit and clears site
  include-path overrides. A compiled runtime/header/device-property probe must
  pass before configuration. The read-only path inspection found conda headers
  under `targets/x86_64-linux/include`, not `include`; its bounded-output
  pipeline ended141 (SIGPIPE), so it is path evidence, not a GPU execution pass.
- Static review `d06a70caa6ac7f823fa86e561e093f8d3` found no proven
  correctness defect in8dc's mappings, capacity arithmetic, host budgets or
  lifetime/error handling. GPU numerical/fault/capacity runtime evidence is
  still required; review is not a test pass.
- First expanded native run: **168 passed/3 failed, zero errors/skips,
  242.05 s**. All failures are new low-memory exchange fixtures rejecting
  forced in-core DFHelper initialization before X/Y. Evidence preserved under
  `/fastscratch/awallace43/fdds-performance-20261020/native-8dc714e4d3/`.
  The fixture now explicitly selects/restores SCF_SUBTYPE=AUTO to permit the
  budgeted out-of-core integral path. Targeted recovery at16 KiB passed all
  six native exchange cases in4.44 s, zero failures/errors/skips.
- Strengthened the final blocked fixture to12,800 bytes, with explicit
  dimension-derived assertions that both X occupied blocks and Y virtual
  blocks contain partial tails. Independent references/tolerances are
  unchanged. A fresh targeted plus complete171-test native gate is required
  for this final test-driver pin; the scientific C++ remains reviewed8dc.
- Final native gate at test-driver **fd2c09cd0cfbfce61a311508ba6c53039025179a**
  passed **171 tests, zero failures/errors/skips,240.780 s**;52 FDDS,
  67 CPSCF,6 SAD,6 lifecycle,8 basis-parser,32 grid-switch. Separate targeted
  six X/Y whole/blocked cases passed2.440 s. Maximum printed FDDS contribution
  error7.105427357601002e-15. Evidence and audited counts:
  `build_fdds_cpu/performance-cpu-fd2c09cd0c/`.
- Scheduler dry-runs found no near-term H200/H100/A100 slot, even with shorter
  wall/memory requests (estimates October28/November1). No optimization GPU
  job has been submitted yet. RTX6000's earlier October20 morning estimate
  is not a capability pass; its older Cascade Lake/RTX6000 node class may
  be below the required compute8 capability. No unsupported GPU or skipped
  suite will be called a correctness pass. Alternate account/QoS dry-runs
  are read-only checks, not authorization to silently change the charge route.

## Optimization GPU submission (not a pass)

- **14199355**, **gts-cs207 / gpu-h200 / inferno**, one H200,8 CPUs,
 128 GiB/four hours; effective route and PENDING/Priority state verified from
  its own scontrol record. $5.65 worst case; monthly balance$28.72 before
  submission. No duplicate optimization job; existing jobs were not changed.
- Scientific/test pin **fd2c09cd0cfbfce61a311508ba6c53039025179a** (production
  code identical to reviewed8dc714), own source/build
  `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-opt-validation-20261020-v2`.
  Frozen harness **f26b3b6d1bd997b9c0e135053e99187dda55ee99**, own
  `psi4.fdds-opt-harness-20261020-v1` checkout. Both clean pins checked;
  old8dc v1 checkout and all historic validation builds/runs remain immutable.
- Run `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-optimization-01`;
  inputs in `caches/{peptide,nanotube,protein83}` checksum-verified after
  transfer. Startup's CUDA header/runtime probe, compiler probes and import/
  library hashes must pass. Scientific build uses shared dependencies
  read-only; no installed package changes.
- Require21 focused plus69 full passes (90 executions/69 unique, no skips)
  before any timed replay. Then fresh-process CPU/GPU/GPU/CPU controls for
  each system,8 threads/96 GiB/same frozen stage/options/cache/allocation.
  GPU frequency work and four X/Y timer calls required; CPU route counts zero.
  Both coupled and uncoupled energies must agree within1e-5 Eh.
- Automatic watcher **b486ad2e3**,300-second state-change checks,14-day bound.
  Latest dry-run predicts October29; actual accepted job initially has
  StartTime=Unknown. No morning completion guarantee or speedup claim.

### Startup failure and harness recovery

- Actual job14199355 started within minutes, **not** the late dry-run date:
  H200/driver615.71.09, inferno, nodeatl1-1-03-019-2-0. It failed6 s after
  startup, before compiler probes/configuration/tests. **Zero test executions
  and benchmark processes**, no VERIFIED/COMPLETED artifacts.
- Separate own CPU compile-only job **14199512**, gts-cs207/cpu-small/inferno,
 8 CPUs/48 GiB/one hour ($.07 worst case), also failed7 s before compilation.
  Sourcefd2c09, harnessa178609; queued H200 checkout/build untouched.
- Exact read-only module+activation reproduction confirms the cause:
  site `/storage/pace-apps/.../cuda/12.6.1.../bin/nvcc` stays ahead of conda
  after activation. The strict equality guard correctly stopped the job but
  had no diagnostic message. `PATH="$env/bin:$PATH"; hash -r` after activation
  selects the pinned conda nvcc. Compiler/root/include isolation is unchanged.
  Guard now prints selected/required paths and missing-header diagnostics.
  This is harness recovery, not an FDDS equation/kernel change.
- Both frozen failed runs and local evidence are preserved at
  `build_fdds_cpu/performance-gpu-14199355/` and
  `build_fdds_cpu/performance-compile-14199512/`.
  Terminal markers honestly report0 tests. Corrected compile/link preflight
  is running before any new submission; no driver/package changes.
- Recovery will first compile/link/import the same fd2 scientific candidate
  with CUDA enabled on a fresh isolated CPU checkout/build. A later H200 job
  may reuse that own pinned stage read-only: binary pin, core SHA256 and
  production-source equivalence are enforced by the existing read-only
  validation contract. GPU mode is explicitly forced out of compile-only
  mode and still requires21+69 actual passes before matched benchmarks.
  CPU compilation alone never establishes GPU numerical correctness.
- Corrected environment/compiler/link probe passed: conda **nvcc13.3.73**,
  GCC14.3 absolute cc1/cc1plus, `CUDART_VERSION=13030`, conda
  `libcudart.so.13`. This probe did not execute a GPU call. Evidence
  `build_fdds_cpu/performance-gpu-14199355/corrected-toolkit-preflight.log`.
- The earlier partial protein83 snapshot (not a finalized timing result)
  showed922 AO/3820 auxiliary,78/68 occupied and844/854 virtual while the
  local16-thread replay was in object preparation. Recovery increases per-process timing bound
  to three hours and requests eight hours for the paired/repeated campaign,
  without changing physical settings or matched96 GiB host budgets.
- Recovery runs one CPU/GPU pair for **all systems before repeats**, retaining
  each system's overall ABBA order. Atomic per-system first-pair audits and
  the all-system FIRST_PAIR_VERIFIED.json distinguish one-repeat measurements
  from final two-repeat VERIFIED/COMPLETED. If a later process times out,
  completed earlier pairs survive but the overall campaign is not called
  complete. Auditor requires source/core/cache/options/host/job identity,
  finite times/two energies, actual frequency/X/Y GPU routes and1e-5 Eh.
  Its9 report-only unit tests passed; these are not scientific/GPU timings.
- Corrected CUDA build **14200049** accepted: **gts-cs207/cpu-small/inferno**,
 8 CPUs/48 GiB/one hour,$.07 worst case,$29.39 monthly available before submit.
  Scientificfd2c09, harnessebdb1950be; source
  `psi4.fdds-opt-cuda-compile-20261020-v2`, run
  `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-cuda-compile-02`.
  Clean pins, no duplicate compile job, dry-run and effective QoS verified.
  Automatic watcher **bb6a86037**; compilation/import outcome pending.
  GPU recovery harness **c289c502d55627eabf99acdc127a28334416704d** is
  staged separately; no second GPU job is submitted before the build audit.
- Corrected build14200049 completed0:0 in **14m57s**. Its provenance audit
  launch was interrupted by local ENOSPC (no launch receipt persisted).
  On resume `/home` has26 GiB available; no files were deleted by this agent.
  Audit was relaunched read-only as **b12bed014** before a GPU retry.
- Read-only audit **b12bed014** passed: clean fd2c09 source, Release,
  cuEST/Einsums/OpenMP enabled, XHOST disabled, correct staged import and
  core SHA256 **92e5bff6641dbb2f4790938e9886fc8038d6d03527eb26c6934ddeecabac5a7f**.
  BUILD_ONLY_COMPLETED identifies fd2c09; no GPU VERIFIED marker exists.
  Resolved cuEST0.2.2, cudart13.3.29 and cuBLAS/cuBLASLt13.6.0.2 match the
  recorded dependency provenance. **Zero GPU tests executed by this CPU build.**
  Evidence `build_fdds_cpu/performance-compile-14200049/remote-evidence/`.
  One read-only-stage H200 recovery was submitted under **b03a4f22f**.

### Verified-stage H200 recovery submission

- **14210379**, accepted **gts-cs207/gpu-h200/inferno**, one H200/8 CPUs/
  128GiB/eight hours. Effective QoS/account/resources verified in its own
  scontrol record; initial state PENDING/Priority, StartTime unknown.
  Fresh dry-run, live headroom/duplicate checks and stage/core/input checks
  passed. Monthly available$29.38; worst-case charge$11.31. No other jobs
  cancelled or modified. Dry-run estimate October31 is not an actual start.
- Source/test-driver **fd2c09cd0cfbfce61a311508ba6c53039025179a**,
  scientific8dc714, frozen harness
  **c289c502d55627eabf99acdc127a28334416704d**. Reuses audited compile-v2 stage
  **read-only**, expected core92e5bff6641dbb2f4790938e9886fc8038d6d03527eb26c6934ddeecabac5a7f.
  Bytecode and pytest cache writes disabled; shared environment unchanged.
- Run `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-optimization-02`.
  Automatic watcher **bcde45e0e**,300-second state-change checks,14-day bound.
  Require21 focused plus69 full passes (90 executions,69 unique,zero skips),
  actual X/Y/frequency GPU routes and pinned import/library provenance before
  timings. Then same-allocation8-thread96GiB CPU/GPU controls using identical
  physical caches; first pairs precede repeat pairs for all systems.
  **No new GPU numerical result or measured speedup yet.**

### User-approved shared interactive replacement

- The user explicitly selected replacing **only queued14210379** with one
  bounded interactive allocation, then asked cuEST agents to share it.
  A PENDING-state check preceded its authorized cancellation; old source,
  stage, submission and run evidence stay unchanged. No tests ran in that
  cancelled allocation. No other job was modified or cancelled.
- FDDS is sole allocation owner. Parent and PySCF/disp0 agents reported no
  usable interactive allocation and are holding duplicate requests. PySCF
  fixed its offline UUID-controller bug and requests an early bounded RKS
  preflight window. Disp0 is not ready and must not delay/hold the GPU idle.
  This allocation cannot satisfy disp0's hypothetical512GiB protein request.
- Replacement retains the approved ceiling **one H200/8CPU/128GiB/eight
  hours/inferno**. It uses `salloc --no-shell` plus serialized exclusive srun
  steps; final benchmarks have whole-allocation exclusivity. Common manifest
  publishes job/node/full UUID/account/resources and a shared flock path.
  Every process is bounded and cleanup must be proved before lock release;
  missing GPU cleanup quarantines the allocation, not silent reuse.
- Each agent keeps isolated pinned source/build/env/results. No shared
  environment or frozen stage mutation. FDDS reuses core92e5... read-only.
  Seven controller-only mock tests passed in.458s; these test locking/guards/
  exit-code/evidence behavior, **not** GPU numerical correctness.
  Replacement is not submitted until current account/cost/dry-run checks.
- First controller acquisition **bb154fdd3** stopped after the accepted
  scheduler dry-run, before creating its run directory or invoking salloc.
  Root-path guard used `test` literal equality instead of Bash pattern
  matching; positive-path rejection reproduced offline (exit1).
  Corrected guard now has accepted/rejected-path regression coverage and an
  ERR line diagnostic. **Eight controller mock tests passed in.479s**.
  This attempt consumed no allocation and ran no GPU tests. Preserve its
  old8e9 controller checkout; retry uses a new controller pin/worktree/run.
