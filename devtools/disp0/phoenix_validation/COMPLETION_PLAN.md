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

- Helper pin: `005f3b4c71c521fa5156585c2eea461b74c4c9c3`, staged clean at
  `/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-cpu-production-005f3b4c71`.
  All28 offline helper/input/CPU-comparison guards passed. Scientific source/
  whole stage remains1aed320cdb/job14321813.
- Frozen input JSON is staged and SHA-verified at
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/benchmark-inputs-f8c96058e938/inputs.json`.
- **14324190 submitted and RUNNING**, effective cpu-small/inferno on
  gts-cs207-chemx confirmed by queue/accounting and job prolog. It is the
  peptide paired pilot,24CPU64GiB1h; worst-case charge$0.20.
  Campaign root:
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/production-cpu-005f3b4c71`.
  Watcher `b05716a78` is active and durable.
- **Nanotube/protein83 not yet submitted.** Once the pilot's
  `peptide/job-14324190/comparison-result.json` is PASS with6 accepted CPU
  samples and its job/batch/step accounting is COMPLETED0:0, use the same
  clean helper checkout and `submit_cpu.py`, with the same proof/input paths/
  SHAs, `--pilot-job-run <campaign>/peptide/job-14324190`, respective
  `--system nanotube|protein83` and fresh `--run-root <campaign>/<system>`.
  Recheck `inferno_account.sh` immediately before each submission with24CPUs
  and02:00:00/08:00:00 respectively. Current selector spills to chemx.
  This is the remaining two allocations in the user-requested three-job
  CPU batch, not permission for automatic paid retries or larger jobs.
- Inspect actual `allocation.json`, finalized child results and full matrix/
  canonical-scalar checks before interpreting any timings. No qualified full
  CPU timing summary exists yet. Existing H200 gate14322613 is unchanged.
- First allocation-native inspection passed: effective resources/CPU/features
  admitted, scratch is node-local XFS
  `/dev/mapper/sysvg-scratch[/14324190/.14324190/_tmp]`, free1.451TB.
  Both real native water preflights completed; first peptide v2/C++ pair also
  completed and the next repeat had started. These are finalized child
  artifacts, **not yet a three-repeat summary or terminal pilot approval**.

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
  watcher is active. Inspect `gate-result.json` and every child result/cleanup,
  not merely the scheduler exit code, before accepting this gate.
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
