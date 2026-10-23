# Phoenix full F-SAPT completion contract

## Latest CPU production comparison request

The user explicitly requested **inferno CPU comparisons** of the optimized
Einsums-v2 and C++ F-SAPT implementations, and clarified that production uses
scalar-rescaled induction for speed. The source defaults confirm:
`FISAPT_FSAPT_IND_RESPONSE=false`, `FISAPT_FSAPT_IND_SCALE=true`,
`FISAPT_FSAPT_EXCH_SCALE=true`; exact per-source response is documented
“not recommended”. Canonical coupled induction is still calculated once.

The new CPU batch therefore uses **production scalar rescaling**, not the new
exact per-source solver. Both arms use the common HF SAPT(DFT)/E2 canonical
driver, then `SAPTDFT` (v2) versus `FISAPT` (C++) fragment dispatch. Three fresh
repeats per arm per requested augmented-basis system; one paired CPU allocation
per system, counterbalanced arm order. Each allocation first checks a water
pair before the larger calls. No GPU requests or changes to14322613.
Peptide is submitted first. `submit_cpu.py` refuses nanotube/protein83 without
the exact helper/science/input-pinned successful peptide comparison, clean
job-status marker and COMPLETED0:0 job, batch and step accounting.

Planned resource caps are24 threads on exclusive Gold6226/core24/localNVMe
CPU-small nodes,64GiB/64GiB/160GiB SLURM memory,40GiB/40GiB/96GiB Psi4 memory,
and1h/2h/8h for peptide/nanotube/protein83 respectively. Exact-route test-only
accepted all three shapes. CPU-small billing304 yields a total worst-case
charge about$2.17 (11 node-hours); live account selection is rechecked before
submission. These are controlled18 fresh full-system timing calls, not an
automatically extending campaign.
Allocation-local admission verifies effective QoS/account/CPU count,
exclusivity, memory/wall limit, actual Gold6226/features and a real node-local
disk scratch mount/free space before either native arm starts.

`cpu_benchmark.py`, `compare_cpu.py`, `submit_cpu.py` and CPU job/payload
helpers implement this separate batch. No scientific driver/core overlay:
reuse the qualified whole stage1aed320cdb. Phase records are exclusive/additive,
SCF/canonical work is included in whole energy time, and arrays/checks are
processed after that timer. Disk F-SAPT output is disabled consistently.
Raw failures remain visible; only three validated repeats produce a final
median/range. Gauge permutations must preserve frozen/active subspaces.
Native localizer maxiter1000 versus Python50 is recorded; either one's
nonconvergence makes a timing unqualified.

These **production CPU times are not a valid denominator for the current
exact-fragment GPU mode**. A matching production GPU continuation and declared
CPU/GPU thread/resource policy must be qualified before any GPU speedup ratio.
The earlier exact-response qualification contract below remains diagnostic;
it is not the primary production CPU benchmark.

### CPU submission checkpoint

- **Nanotube14324682 FAILED1:0** after3m20s, not OOM/SCF/localization failure.
  Allocation admission and both water preflights passed. The first full
  v2 call completed its science in155.904s, then the old net-total guard
  rejected a0.64329819microEh fragment-sum residual;0 accepted samples.
  The log/source explain the known DF consistency: localized Elst is
  -28.77586628mEh versus JK canonical -28.776417821518407mEh
  (~0.55154microEh), with the remaining~0.09176microEh from scaled fitted
  induction. All three SCFs and all three IBO localizations converged.
  Do not promote this failed single call to a qualified timing median.
- A post-timing harness correction is being prepared: strict canonical
  component algebra plus unchanged per-component tolerances (including
  directional induction), raw residual reporting and unchanged v2/C++ matrix
  comparison. No matrix correction or scientific algorithm change.
  The nanotube cancellation counterexample reproduced the old failure in a
  fast regression; all31 corrected helper/input guards pass.
  Timed engine/resource/environment/shared-runtime fingerprints match005f.
  Existing successful peptide samples can be requalified without retiming
  only if those proof checks and new ledgers pass for every sample.
  Replay recomputes sums from hashed NPZ matrices, binds each sample's
  identity/options and repeats the paired/gauge checks. Fingerprints seal
  all other post-call qualification rules as well as the timed engine.
  Failed artifacts now preserve matrices before later guard failures.
  **No paid retry has been submitted or authorized automatically.**
- Corrected helper `c58a490670` was staged separately. Its first offline
  peptide replay stopped before numerical replay because Slurm's short
  node name differed from the worker's `.pace.gatech.edu` FQDN. All six
  saved worker records name the same allocated host. The follow-up accepts
  only this explicit Phoenix alias, not arbitrary domains or other hosts;
  no job/science execution is needed for this provenance-only replay.
- **Offline peptide requalification PASS** with helper
  `abd4c1bc468ce7f759988e3af2f8c985ff6c48bf`, staged clean at
  `/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-cpu-ledger-abd4c1bc46`.
  All six original sample identities, hashed matrices, component ledgers,
  paired comparisons and same-arm repeat comparisons passed. Canonical
  algebra residuals are at most1.74e-18Eh; maximum paired full-matrix error
  remains3.959922667551297e-12Eh. Timed work, other qualification rules,
  resources/environment and all eight shared helper files matched005f.
  Durable replay output: background task `b34e990d7`
  (`.pi/tasks/01a165d6-ab03-768d-83bd-c6172c2c1b1b-4065461/b34e990d7.output`).
  This reuses the original measured times; it is not six new calculations
  and incurred no new SLURM allocation. The original pilot is untouched.
  Failed nanotube14324682 cannot be requalified this way: its original
  helper did not save the full matrices before the guard failed.
- The user subsequently selected **Inferno retry** for exactly one corrected
  nanotube allocation:24CPUs64GiB2h, maximum$0.39. Fresh selector `b11f87311`
  reported gts-cs207 available-$0.07, selected gts-cs207-chemx for this one
  job. Submission `bc857a3ee` uses immutable helper `abd4c1bc46` and a fresh
  root `production-cpu-ledger-abd4c1bc46/nanotube`, without modifying the
  original nanotube failure or protein83 job14324700. No further retry or
  new GPU allocation is authorized by this approval.
- **14326813 submitted**, after exact CPU-small test-only and successful
  old-pilot requalification plus terminal accounting. The test-only probe
  printed14326812, which is not the submitted compute job.
  Fresh output:
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/production-cpu-ledger-abd4c1bc46/nanotube`.
  Durable watcher `b2ac040c0`; effective-route/prolog inspection `bebb0482a`.
  Controller readback confirmed gts-cs207-chemx/cpu-small/**inferno**,
  24CPUs64G2h and exact core24/gold6226/localNVMe features; PENDING(Priority).
  No prolog existed yet. Allocation-local admission still must pass at start.
- **14326813 COMPLETED0:0** (job/batch/step),16m27s; no TERM.
  `PASS_PRODUCTION_CPU_COMPARISON`,6 accepted CPU samples,0GPU speedups.
  Actual Gold6226 allocation, local XFS scratch and24CPU affinity admitted;
  both water controls and all three paired/repeat matrix checks passed.
  Nanotube whole-call medians/ranges (n3): C++148.126126s
  [146.540743,148.748254], v2 156.061176s [153.973121,156.988453].
  Thus v2 is5.36% slower. Maximum aligned matrix error6.83e-11Eh.
  All samples confirm retained rank740/virtual732,662; fitted raw net
  residuals0.6413–0.6425microEh with strict canonical algebra intact.
  Raw collection `b7b791acf`; phase/resource analysis in `CPU_RESULTS.md`.
  No further nanotube retry is needed or authorized.
- Nanotube's printed virtual extents are732/662 with742AOs and8/78 occupied:
  the retained orbital rank is740, not742. Future GPU qualification must not
  equate retained MO rank with the AO count in an augmented basis.
  The timing helpers now record retained rank/virtual counts after the timer.
  This does not change any timed work or old immutable job artifacts.
- Helper pin: `005f3b4c71c521fa5156585c2eea461b74c4c9c3`, staged clean at
  `/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-cpu-production-005f3b4c71`.
  All28 offline helper/input/CPU-comparison guards passed. Scientific source/
  whole stage remains1aed320cdb/job14321813.
- Frozen input JSON is staged and SHA-verified at
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/benchmark-inputs-f8c96058e938/inputs.json`.
- **14324190 COMPLETED0:0** (job/batch/step) in3m34s, effective cpu-small/
  inferno on gts-cs207-chemx confirmed by queue/accounting and job prolog.
  Peptide paired pilot,24CPU64GiB1h; worst-case charge$0.20.
  Campaign root:
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/production-cpu-005f3b4c71`.
  Its watcher returned terminal success.
- The pilot artifact is `PASS_PRODUCTION_CPU_COMPARISON`,6 accepted CPU
  samples,0 accepted GPU speedups. Whole-energy medians/ranges (n3 each):
  C++21.292634s [21.135252,21.539984]; v2 22.031666s
  [22.009369,22.899400]. Thus v2 is about3.5% slower on this point, not faster.
  The maximum aligned full-fragment matrix discrepancy is3.96e-12Eh.
- **Nanotube14324682 and protein83 14324700 submitted** on inferno,
  gts-cs207-chemx/cpu-small, after the exact-pinned pilot result and terminal
  accounting were enforced. Both use24CPUs, with64GiB/2h and160GiB/8h caps.
  The account selector was refreshed independently immediately before each;
  worst-case charges$0.39 and$1.58. Their roots are `<campaign>/nanotube`
  and `<campaign>/protein83`, with child artifacts under each `job-<id>`.
  Durable state-change watchers: `bf1357bc4` and `be8df994b`.
  **All three planned CPU allocations have now been submitted; no automatic
  paid retries or extra comparison jobs are authorized by this batch.**
- Inspect actual `allocation.json`, finalized child results and full matrix/
  canonical-scalar checks before interpreting any timings. Peptide and corrected
  nanotube14326813 are qualified; protein83 remains uncollected.
  Existing H200 gate14322613 failed
  separately, as recorded below.
- First allocation-native inspection passed: effective resources/CPU/features
  admitted, scratch is node-local XFS
  `/dev/mapper/sysvg-scratch[/14324190/.14324190/_tmp]`, free1.451TB.
  Both real native water preflights and all three peptide pairs completed.

User request: complete the missing GPU qualification, scalability, matched
CPU/GPU timings and analysis. Inputs are **peptide, nanotube, protein83** from
`~/data/timing_test_suite/geometries/systems.py` (the user confirmed protein83,
not protein87). Primary basis is spherical **aug-cc-pVDZ**. This is **full
fragment F-SAPT**, not scalar SAPT or dispersion-only timing.

## Ordered acceptance gates

1. **Pinned Phoenix runtime and small H200 gate**
   - Keep successful CPU build14309728/source3343e673c2 and all failed roots.
   - Pin optional cuda-bindings13.3.0 and resolved dependency artifacts in a
     new private dependency directory; do not install into shared environments.
   - Authenticate canonical cuEST0.2.50 archive/native/all59headers, complete
     CUDA13.3 closure, native/Python E2 world and the actual whole Psi4 stage.
   - Build a new whole stage when scientific Python files change. No new
     Python overlay on the successful old core.
   - Qualify real H200 UUID, NVRTC, FP64 cuBLAS including strided-batched
     exchange, native DFMO, signed terms/fragment matrices, changed/repeated
     inputs, complete cleanup and explicit no-CPU-factor/localizer fallback.
     CPU/mock/serial fixtures are not this evidence.

2. **Scale-ready response, overlaps and localization**
   - Replace reference O(Q O² V²) response exchange with exact blocked FP64
     GEMMs for BOTH distinct exchange contractions. Preserve full occupied
     response, physical amplitude sign, PCG curvature and true-residual checks.
   - Replace prototype P4096/union512/O64 policies with explicit simultaneous
     allocation/work budgets and native ABI limits. Never just disable guards.
   - Preserve original spherical/SPDF, metric, frozen/active, IBO pair ordering,
     convergence, owner lifetime and same-gauge JKFIT/RIFIT invariants.
   - All three actual augmented-basis systems must pass numerical/memory gates
     before accepting their timing measurements. OOM/nonconvergence are failed
     points, not permission for CPU fallback or relaxed tolerances.

3. **Matched measurements**
   - Three independent fresh-process repeats × CPU/GPU × three systems.
   - Match geometry, primary/auxiliary bases, threads, frozen core, coupled
     response, delta-HF, exchange/induction scaling and output conventions.
   - The CPU reference must implement the same coupled fragment science.
     Inspect the current CPU fragment path: canonical-scalar scaling of an
     uncoupled partition is not an exact coupled-response baseline. Use an
     independent matrix-free CPU oracle at large sizes, not a dense P×P solve.
   - Time whole energy calls and nonoverlapping phases, including first-use
     initialization, native factory, transfers, audits and cleanup. Separate
     diagnostic/oracle work. Record runtime/core/module/config/input identities,
     exact work, RSS/VRAM/scratch and failed/incomplete points.

4. **Analysis**
   - Require signed term/fragment correctness before reporting speed ratios.
   - Publish raw records, per-system three-repeat medians/ranges/sample counts,
     phase timing vectors, resource peaks and all failures.
   - Report slowdowns as slowdowns. No extrapolated protein results, historical
     PBE0-D4 baselines, kernel-only labels for whole-call timing, or fabricated
     confidence intervals.

## Science/resource decisions

Use matched **aug-cc-pVDZ-JKFIT** and **aug-cc-pVDZ-RI**. Their H/C/N/O shells
remain SPDF. `def2-universal-jkfit` contains G shells on C/N/O and is outside
the currently qualified native factory; do not silently substitute it to save
memory.

Atom-additive basis-file counts (confirm with the actual installed stage):

| System | Primary | JKFIT | RIFIT | Full occupied A/B |
|---|---:|---:|---:|---:|
| peptide | 356 | 1308 | 1042 | 20 / 20 |
| nanotube | 742 | 2748 | 2240 | 8 / 78 |
| protein83 | 1293 | 4762 | 3820 | 78 / 68 |

Prefer bounded free embers preparation jobs. Fresh inferno submissions require
the user-approved footprint and a live account selector; the earlier CPU-build
approval is not blanket approval for a new GPU campaign. Use one H200 and at
most8 CPUs unless exact-route validation establishes another approved shape.
No job-control operations or overwrites of old artifacts without approval.

## Current development evidence

- Baseline Phoenix CPU build: PASS14309728;244 CPU regressions,8 launch guards,
  5 E2 API checks; no GPU qualification.
- Response implementation now uses Q-blocked strided-batched native-FP64
  GEMMs, with resident-factor/PCG/workspace admission before DFMO.
- Focused CPU-shim tests pass for signed Hessians/solutions, nonsymmetric
  exchange indices, ragged Q blocks, BLAS transpose/broadcast descriptors,
  changed inputs and large-space admission without a dense Hessian.
- Overlap, native powers, IAO and ordered IBO memory/work admission changes
  are in progress; broad regression and actual device gates remain required.
- Cachy was inspected read-only for prior diagnostic conventions. Its GPU/build
  HOLD remains intact; no Cachy execution or environment change is authorized.
- Both matched public arms select `SAPT_DFT_DO_FSAPT=SAPTDFT` and
  `SAPT_DFT_INDUCTION_TYPE=CPKS`. At HF this is the HF response; explicit
  `CPHF+SAPTDFT` is rejected by the public driver. The opt-in CPU fragment
  solver explicitly constructs native **DF** JK with the same JKFIT basis.

The persistent goal's short label is “Complete these tasks”; this file records
the concrete scope and evidence required before claiming it complete.

## Native Phoenix checkpoint (2026-10-23)

- **H200 gate14322613 FAILED1:0** after10m32s. `gate-result.json` records
  `TimeoutExpired` for the first `gpu-unscaled` child at its600s limit.
  Its CPU-unscaled control completed, and GPU stdout shows native cuEST HF/
  canonical coupled induction completed followed by fragment basis loading.
  No completed GPU child result, fragment matrices or final cleanup proof
  exists; no GPU qualification or speedup is accepted. No GPU retry submitted.
  This failure does not block the separate production-scaled CPU batch.
  A read-only source hypothesis is excessive repeated full-DSO hashing in
  `identity.audit_maps()`/`provider.assert_identity()` inside the tiny-block
  continuation (block5); this is **not a measured diagnosis**. A follow-up
  should persist phase/stack checkpoints and test a still-ragged production
  block24, without weakening provenance or claiming the timeout fixed.
- **14321813 completed0:0** in9m49s on cpu-small/embers,8CPUs48GiB.
  Source pin `1aed320cdb8b031c281b4642001a750aa51c549f`.
  Root `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-fragments-1aed320cdb/job-14321813`.
- Its whole-stage CPU build result is PASS,266 CPU regressions; private runtime
  preparation is `PREPARED_NOT_GPU_VALIDATED`, with cuda-bindings13.3.0 and
  cuda-pathfinder1.8.4. Runtime-config SHA256:
  `4d255b795caae75a9df879b81b26fd84b10a79d0400f2831b5dbcdebad4afa6c`.
- Both actual native public CPU fragment checks completed, including the new
  exact coupled source solves and canonical directional sum checks:
  water dimer/spherical aug-cc-pVDZ, unscaled and delta-HF/exchange scaled.
  Energies are -0.007157905730603229 and -0.008584114664572127Eh respectively.
  These diagnostic calls are NOT accepted timings or GPU comparisons.
- Helper pin `127fc33590` is staged separately at
  `/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-gate-127fc33590`.
  Its17 offline launch/oracle guards pass. No scientific driver/core overlay.
- The user explicitly approved **one** inferno H200 correctness gate,4CPUs,
  32GiB,maximum1hour,$1.41 worst case on gts-cs207-chemx. This is NOT blanket
  approval for the subsequent large-system or three-repeat timing campaign.
  **14322613 submitted** after fresh account selection and exact gpu-h200
  route preflight, under fresh output root
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-h200-gate-127fc33590`.
  Scientific pin remains1aed320cdb; helper pin127fc33590. The state-change
  watcher returned terminal failure (see above). Preserve `gate-result.json`
  and every child result/cleanup; do not infer qualification from GPU progress.
- First effective-route check found14322613 **PENDING(Priority)**,
  QoS inferno; actual GPU/prolog proof remains pending until it runs.
- The three requested literal inputs are frozen locally in
  `.pi/inputs-augccpvdz-v1/inputs.json` (no execution/import of the data module,
  no off-limits systems selected). Input-manifest SHA256:
  `f8c96058e9387ab14179ce263a2d4896dbe9cc2e3bf1286654784825ff178e38`.
  Original systems.py SHA256:
  `72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb`.
  Actual source basis-file SPDF/spherical counts reproduce the table above.
  Frozen occupied counts are5/5,2/24,20/19 respectively.
  Resident response factors alone are1.170/1.170,11.152/10.212 and
  55.954/56.361GiB per direction. These are **not total peak-memory estimates**;
  native HF/JKFIT/RIFIT plans and all other live buffers must still be measured.
- Current offline helper/input contracts:20 passed. No accepted timing repeats.
