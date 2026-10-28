# Current-stage split F-SAPT benchmark

## Terminal GPU checkpoint (2026-10-28)

Watcher `baf5af493` reached ALL-TERMINAL. H200 job **14481883 was PREEMPTED**
on embers after **1:00:16**. The runner records receipt of SIGTERM15, rather
than a spontaneous numerical exception; owned child groups were reaped and
scratch cleanup completed. The batch/step exit1 reflects this interruption.

All three peptide GPU calls finished with exit0 before preemption:
994.512498, 994.536646 and 993.473989 seconds; median **994.512498 s**.
The case passed its repeat checks and is recorded as
`PASS_REPEATS_AWAITING_OTHER_RESOURCE_CLASS`. Compared with the CPU median
295.103911 s, this is diagnostically **3.370 times slower**, not an accepted
scientifically qualified speedup: independent CPU fixed-GPU-gauge matrix
validation is still outstanding.

Nanotube's first GPU child was interrupted with no completed timing sample;
protein83 GPU was not reached. No full scaling curve is available.
The CPU outcomes below remain unchanged. No automatic retry or new allocation
was submitted. Earlier RUNNING/first-sample statements below are historical
checkpoints, superseded by this terminal record.

Measured peptide regression analysis (two completed GPU calls):
[`PEPTIDE_PERFORMANCE_DIAGNOSIS.md`](PEPTIDE_PERFORMANCE_DIAGNOSIS.md).
Both calls take about994.5s; about823s lies in fresh native factory/plan setup
windows before the fragment-term loop. This is not an accepted speedup or
a measured attribution specifically to JIT/audits; those subphases need timers.

## CPU batch terminal; first H200 sample (2026-10-28 checkpoint)

CPU-only watcher `b113822d4` reached ALL-TERMINAL. No CPU job was waiting
for GPU14481883.

| System | Job | Result |
|---|---|---|
| peptide | 14481886 | Three fresh CPU repeats, median **295.103911 s**, range 264.639815–296.875835 s |
| nanotube | 14481888 | Two completed calls, 5334.743746 / 5156.415041 s; repeat-localization check FAILED; no qualified median |
| protein83 | 14481890 | First child reached the **28800 s supervisor deadline** during fragment coupled induction; no completed sample |

Peptide result/sample/NPZ hashes, distinct process identities and cleanup were
verified. Its mode is exact per-source CPHF, not the cheaper historical
scalar-scaled uncoupled CPU partition. Nanotube's first failing field is the
signed/permuted `Locc_A` coefficient comparison (max violating difference
1.00277e-8 against unchanged atol/rtol1e-9). This does not establish that all
other fields pass, nor justify relaxing a guard. Both native calls exited0.
Protein83's allocation was24h for three calls; the per-child8h cap stopped
the first. The stderr SIGINT trace is the supervisor's interruption, not a
diagnosed segmentation fault or OOM. Its last log is in source-resolved
fragment induction after a DiskDFJK simple-constructor warning. The warning
is not by itself a measured attribution of all elapsed time.
Owned children were reaped and scratch cleanup completed for both failed jobs.
No paid retries were submitted.

After the QoS switch, H20014481883 is **RUNNING on embers**, node
`atl1-1-03-019-2-0`. The finalized water control confirms NVIDIA H200,
UUID `GPU-edf24d8b-43c2-1ce9-9967-7862cf713a99`, driver615.71.09,
143771MiB, compute90. Water energy differs from the current native CPU
control by -2.80618e-12Eh; this aggregate control is not the full matrix oracle.

The first peptide GPU call completed in **994.512498 s** with native cleanup;
the second was active at inspection. This first observation is slower than the
295.103911s CPU median, not evidence of acceleration. It is **one unqualified
GPU sample**, not a three-repeat GPU median or accepted CPU/GPU speedup.
Full repeat and independent CPU matrix qualification remain outstanding.
The running GPU job, source and inputs were not modified during inspection.

## Queued H200 switched to embers (latest)

User request: **switch queue qos=embers because of group billing limit**.
The only pending benchmark job was GPU14481883; peptide CPU was complete,
nanotube CPU had failed repeat qualification, and protein83 CPU was running.
No running or terminal CPU job was modified.

Task `bf207cd56` verified the pending GPU's identity, account, resource request,
8-hour cap and embers headroom, then switched the **same job** with:
`scontrol update JobId=14481883 QOS=embers TimeLimit=08:00:00`.
Controller readback confirms **embers**, the unchanged 8-hour cap, one H200,
8 host CPUs, 160 GiB and no dependency. The immediate post-update reason was
None (awaiting scheduler reevaluation), not a promise of an allocated node.
The job is now preemptible; no replacement submission or automatic retry.

The runtime guard checks the declared scheduler QoS, so only `qos` in the GPU
run manifest was changed to embers before the scheduler update. All scientific
fields, runtime/input hashes, source/helper pins and memory limits are unchanged.
Original manifest bytes are preserved as `gpu/manifest.before-embers.json`.
`gpu/qos-change-embers.json` records the exact command, before/after controller
fields and both manifest SHA256 values. The pinned source/stage was not edited.
The existing monitors follow the unchanged job ID.

## CPU scheduling independence (latest)

The user explicitly requested **CPU should not wait on GPU job**.
CPU14481886/14481888/14481890 must run independently of GPU14481883,
with their existing inferno account/resource/wall caps unchanged. Release only
the GPU scheduler dependency; do not cancel, resubmit or edit queued scientific
source/configuration. The GPU job is unchanged. The dependency declarations in
the historical submission records below document the original plan only.

The frozen worker already computes all three CPU reference calls and checks
their repeats before considering GPU artifacts. If GPU outputs are unavailable
or unqualified it preserves CPU timings and exits without idle waiting, without
running GPU work and without claiming a speedup. Same-gauge validation happens
only when eligible GPU outputs are available; otherwise it remains deferred.
No extra paid validation job is automatically authorized or submitted.
Future benchmark submissions omit the GPU scheduler dependency as well.

Dependency-change task: `b7b109778`; its per-job before/after controller
records and invariant resource checks are saved separately from original
immutable manifests in `cpu-independent-dependency-change.json` under the
current replacement run root.

**Verified:** all three updates returned 0 and controller readback reports
`Dependency=(null)` for CPU14481886/14481888/14481890. Account, inferno QoS,
CPU partition/features, 24 threads, 160 GiB and 2/24/24-hour caps are unchanged;
none reserves GPU TRES. They are independently eligible for scheduling, not
waiting for GPU14481883. No job was resubmitted and the GPU was not modified.
CPU-only completion monitor `b113822d4` returns when the CPU batch ends even if
the GPU is still queued, so CPU results can be collected without waiting for it.

## Replacement authorization (latest)

The original 16-hour inferno GPU job became blocked by
`AssocGrpBillingMinutes`; its three CPU consumers were pending dependencies.
The user approved replacing the queued four-job batch, then clarified
**all inferno actually**. The replacement therefore retains inferno for every
role, with an **8-hour GPU cap** and the same **2/24/24-hour CPU caps**.
No embers allocation was submitted. The shorter GPU cap reduces its billing
reservation; it does not guarantee that the account restriction is cleared.

Cancellation task `b4a58b249` checked that all four old jobs were still PENDING,
owned by the expected account and named for this benchmark, then received
successful cancellation acknowledgement for 14465903/14466146/14466149/14466153.
Old manifests, checkouts and outputs remain intact; no running work was stopped.
New job IDs and effective routes must be recorded after fresh submission.
The historical 16-hour plan and its IDs below are retained as provenance,
not current jobs to revive or automatically resubmit.

Only the allocation wall cap changes in the replacement helper. Scientific
stage, runtime/input seals, cache budget, exact CPHF method, three repeats,
numerical checks and device-memory guards remain unchanged.

Accounting verification (`bb4dbc934`) confirms **all four old jobs CANCELLED**
with zero elapsed time and no assigned nodes. No old benchmark call executed.
The replacement's 103 offline helper tests and shell/whitespace checks passed.
The 8-hour H200 inferno cap prices at $11.31; adding the unchanged CPU caps
gives approximately **$21.16** for the full replacement batch. Fresh account
selection still routes to gts-cs207-chemx ($0.89 monthly gts-cs207 available).

### Replacement submitted (all inferno)

Helper/source checkout:
`491ee751f4fde1a72901d600811063657836bd16`,
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-benchmark-8h-inferno-491ee751f4`.
The scientific/native trees are byte-identical in Git to the completed
`d2a98abe35` build; no native rebuild or installed science overlay was made.

| Job | Role | QoS | Wall cap |
|---|---|---|---|
| **14481883** | one H200 / 8 host CPUs / 160 GiB | inferno | 8 h |
| **14481886** | peptide / 24 CPUs / 160 GiB | inferno | 2 h |
| **14481888** | nanotube / 24 CPUs / 160 GiB | inferno | 24 h |
| **14481890** | protein83 / 24 CPUs / 160 GiB | inferno | 24 h |

All use gts-cs207-chemx after a fresh account query immediately before each
submission. CPU jobs reserve no GPUs and depend `afterany:14481883`.
The scheduler test-only numbers 14481882/14481885/14481887/14481889 are not
compute jobs. Submission task `bed8c593e` recorded zero exit status for every
role with durable per-role manifests/responses/job-ID files.

Fresh output root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-benchmark-8h-inferno-491ee751f4`.
The same three fresh repeats and independent CPU qualification requirements
apply. The 8-hour cap can leave partial cases; no automatic retries or invented
large-system speedup are allowed.

State-change monitor for all four replacements: `baf5af493` (retained across
same-process Pi reload). Routing readback: `beb1f5475`. No embers job exists
for this replacement. Old canceled IDs below are historical only.

Initial controller readback verified every role on inferno with the exact
account, partition, CPU/memory/wall cap and GPU TRES policy. GPU14481883 is
PENDING(Priority), not the former billing-limit reason at this snapshot.
CPU14481886/14481888/14481890 are PENDING(Dependency), each attached to the
new GPU14481883 and with no GPU TRES. This snapshot is not a guarantee against
a later billing restriction, nor evidence of allocation or completed science.

User authorization: one GPU job and multiple CPU inferno jobs. No CPU
reference or same-gauge CPU oracle runs in the GPU allocation. No automatic
retry, scheduler modification, guard removal, or tolerance relaxation.

## Science and timing

- Fresh coherent whole stage at `d2a98abe3529027256efa678be2af6ab89c8a415`;
  response-workspace reuse and opt-in packed-S dispersion reuse are included.
- Exact per-source **HF CPHF**, spherical aug-cc-pVDZ, augmented JKFIT/RI,
  frozen core, matched `INTS_TOLERANCE=1e-16`, exchange/delta-HF scaling.
  This is not the cheaper production-scaled uncoupled induction partition.
  The previous production CPU timings are not used as its denominator.
- Three fresh molecular energy calls per system/arm: peptide, nanotube,
  protein83 (the previously confirmed suite protein entry).
- GPU cache ceiling: additional 64 MiB per call, within unchanged total live
  device guards. Actual cache occupancy/hits remain in execution metadata.
  This budget may not fit a large slab; do not assume a cache hit or gain.
- Instrumented whole-energy timing includes normal F-SAPT output, scientific
  setup/SCF/canonical work, required native audits, and full continuation.
  Runtime admission/primitive controls precede it. Diagnostic HF/fragment
  copies, NPZ writes and post-call numerical validation follow it.
  Scientific runtime/native synchronization remains enabled.
- `--benchmark` retains the existing HF/output Matrix/Vector owners until
  post-call snapshots, rather than inserting large diagnostic copies into the
  timer. It does not supply saved orbitals/localizations to production calls.
- Same-arm repeat/gauge tests, signed separated components, canonical
  energy and independent native CPU fixed-GPU-gauge oracles remain required.
  A speedup ratio is recorded only after three qualified samples in each
  arm and all three independent oracles. No completed worker alone is enough.
- Report the resource policy explicitly: 24-core CPU node versus one H200
  with 8 host threads; not a matched-host-thread or GPU-kernel-only ratio.
  This measures current CPU/GPU relative latency, not a pre/post optimization
  comparison of two whole GPU implementations.

## Allocation plan

| Allocation | Resource | Cap |
|---|---|---|
| Fresh whole build/CPU preparation | 8 CPUs, 48 GiB, no GPU | 3 h |
| GPU producer, all three systems | 1 H200, 8 CPUs, 160 GiB | 16 h |
| Peptide CPU reference + oracles | 24 CPUs, 160 GiB, exclusive Gold6226 | 2 h |
| Nanotube CPU reference + oracles | same CPU shape | 24 h |
| Protein83 CPU reference + oracles | same CPU shape | 24 h |

CPU jobs use node-local writable scheduler scratch, and depend `afterany` on
the GPU producer. References execute even if the corresponding GPU case was
unqualified, so valid CPU baseline work is not thrown away. Oracles and speedup
qualification require a complete matching GPU case; no GPU resubmission is
triggered by missing or failed data.

The GPU allocation starts with a fresh scaled water GPU call compared with
the **current build's** completed native CPU water control. This is a new
experimental batch with CPU build evidence plus existing component-level
Cachy evidence, not a claim that old H200 release `14377643` qualifies the
new code. The complete new GPU/CPU case qualification occurs in the separate
CPU jobs. No fake `PASS_SMALL_H200_ONLY` release is generated.

## Known failures remain visible

Nanotube previously failed orbital-resolved localization-repeat equivalence;
protein83 previously failed native RIFIT creation-workspace admission.
Workspace/cache changes do not purport to fix either. Each remains an
experimental point: a failed repeat stops that point without retry, and a
safe completed child allows the other distinct approved systems to run.
CPU references and original failed artifacts are preserved. No failed point
may be promoted into an accepted performance median by selecting samples.

## Entry points

Use `benchmark_suite.py` to prepare fresh per-role manifests and submit one
explicit role at a time after fresh account selection. `--test-only` runs
before every actual scheduler call, and durable attempt/response/job-ID files
prevent ambiguous automatic resubmissions. Source/helper paths and installed
science pins are separate and recorded; no scientific core/Python overlays.

Fresh build submission: **14465563**, `gts-cs207-chemx/cpu-small/inferno`,
source worktree
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-benchmark-d2a98abe35`.
Build root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-benchmark-build-d2a98abe35`.
The account selector reported monthly `gts-cs207` $0.89 below its $1 floor;
the full build cap is about $0.20. Build submission is not build completion,
GPU submission, or benchmark success.

**14465563 COMPLETED 0:0**, job/batch/step, 9m59s, exit0/TERM0.
Whole-stage proof is PASS at the exact scientific pin above; 301 CPU tests,
89 build-helper tests (one explicit skip), native RHF/API proof and both exact
native CPU fragment preflights passed. Base runtime-config SHA256:
`251f4dbf0ee4c0019e2609ffe8c4a96c8727e29aaeb17d1e36ab4b6ceb58ec73`.

**GPU job 14465903 submitted** on gts-cs207-chemx/gpu-h200/inferno,
1 H200, 8 CPUs, 160 GiB, 16h cap, maximum $22.62. GPU helper
`b5aba145976d5b2a574eadcb9bc3dd7e34ef1bf8` and its source checkout
remain frozen. Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-benchmark-b5aba14597/gpu`.
The scheduler's test-only number 14465901 is not the compute job. No GPU
hardware/prolog, science completion or speedup is inferred before allocation.

The CPU-only consumer plan may use a later **qualification-only** helper
revision without changing the queued GPU checkout. It seals the existing GPU
submission manifest/helper pin and requires byte-identical timed worker,
runtime admission, process ownership and numerical/gauge comparison helpers.
It prepares no second GPU job, and submission of a GPU role through that plan
is forbidden. CPU consumers independently require completed GPU scratch cleanup
and no revoked case qualification before computing any accepted ratio; failed
GPU cleanup cannot be hidden by a previously recorded repeat-pass status.

## Submitted CPU consumers

CPU-only helper `0fea20b2c41854a238b0919dd855ffadb0f23f2e` preserves the
GPU checkout and seals its original manifest and byte-identical shared timed
engine. All 103 offline scheduling/snapshot/qualification tests passed.
Native scientific/runtime source remains the completed `d2a98abe35` stage.

| Job | System | Route | Cap |
|---|---|---|---|
| 14466146 | peptide | gts-cs207-chemx / cpu-small / inferno | 2 h |
| 14466149 | nanotube | same CPU route | 24 h |
| 14466153 | protein83 | same CPU route | 24 h |

Each requests 24 CPUs, 160 GiB, exclusive Gold6226/core24/localNVMe and **no
GPU**. Each depends `afterany:14465903` with invalid-dependency cleanup.
The three fresh account selections kept monthly gts-cs207 at $0.89, below
the $1 floor; upper bounds were $0.39/$4.73/$4.73. Together with the one
H200 cap, benchmark allocation upper bounds total about **$32.47**, excluding
the already completed build. No follow-on or automatic retry is authorized.

CPU output root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-benchmark-cpu-0fea20b2c4`.
Per-system manifests, scheduler attempts/responses and job IDs are retained.
The test-only IDs 14466145/14466148/14466152 are not compute jobs.

Durable state-change monitors:
- GPU14465903: background task `b9b9c7853`.
- CPU14466146/14466149/14466153: background task `b3e80c2fd`.

Submission is not execution or scientific qualification. Accepted new
benchmark results/speedups remain zero pending actual numerical outputs,
same-arm repeats, independent CPU oracles and final cleanup verification.

Controller readback (`b949359b7`) verified all four effective accounts,
inferno QoS, partition, CPU count, wall/memory caps and TRES. GPU14465903 is
PENDING(Priority), no dependency, exactly one H200 requested. All three CPU
jobs are PENDING(Dependency), `afterany:14465903(unfulfilled)`, with no GPU
TRES and the exact requested CPU features. Actual allocation hardware and
prolog QoS remain to be checked after scheduling.
