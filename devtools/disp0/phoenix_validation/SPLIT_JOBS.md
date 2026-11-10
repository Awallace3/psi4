# Separate GPU execution from CPU reference/oracle work

**Latest outcome:**14400888/14400901/14400904 all FAILED, with all outputs
preserved and no automatic retry. The CPU scratch launcher is corrected and
89 local tests pass; nanotube orbital-gauge repeat and protein native-plan
memory admission remain unresolved. Details:`GROUPED_FAILURES.md`.
Queued/pending readbacks below are historical submission checkpoints.

## User-requested regrouping (supersedes the unused eight-job plan)

The user then requested **one GPU job**, shared CPU allocation for small cases,
separate CPU allocation for protein-sized cases, and explicitly approved
inferno. The already-confirmed input here is protein83, not a new protein87
geometry. The grouped layout isolates protein83 as the protein-sized case:

| Allocation | Work | Cap |
|---|---|---|
| One H200 /8CPU /160GiB | nanotube then protein83, three GPU calls each |16h |
| Shared small CPU /24CPU /160GiB | remaining nanotube's three CPU references/oracles |24h |
| Separate protein83 CPU /24CPU /160GiB | protein83's three references/oracles |24h |

Peptide's completed results are preserved; no rerun. Total upper-bound H200
hours16 and CPU-node hours48 are identical to the unused2GPU×8h+6CPU×8h
proposal: **the same ~$32.07 additional maximum**, not a new larger budget.
Each individual scientific case/repeat remains bounded to8h. CPU jobs queue
after the one GPU job and reserve no GPU resources.

Grouped helper pin`8321063109dce76c9896d9ee73e10abc85c1e13a` is staged clean at
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-grouped-8321063109`.
Its prepared plan is
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-grouped-8321063109/plan.json`,
SHA256`c4ad252d66485da3e65a4cf21be9fb50b7039d7d7667dbc013147ae081f694e8`.
All88 offline tests and shell syntax checks passed; independent static review
found no concrete execution/cleanup/provenance/resource blocker.
The exact16h H200 and24h CPU-small inferno shapes passed test-only. Fresh
selectors chose gts-cs207-chemx, with$22.62 GPU and$4.73 per CPU allocation
worst cases. Actual job IDs and effective routing must be recorded separately.

**All three grouped jobs submitted on inferno**, each after its own refreshed
account selector and exact-route test-only, using gts-cs207-chemx:

- **14400888**,`gpu`:1H2008CPUs160GiB16h; all remaining GPU calculations.
- **14400901**,`cpu-small`:24CPUs160GiB24h, shared nanotube reference/oracle block.
- **14400904**,`cpu-protein83`:24CPUs160GiB24h, separate protein reference/oracle block.

Both CPU jobs have`afterany:14400888` plus kill-invalid-dependency and no GPU
GRES. Submission task`b206d04bc` returned0 for each actual call. Test-only
numbers14400887/14400900/14400903 are not compute jobs.
Per-job outputs are under the corresponding label in
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-grouped-8321063109`.
Watchers:`bfd80a9e0` GPU,`bcf7ed7f9` small CPU,`b61e639f6` protein CPU.
Effective-route inspection:`bdfba71b8`. Submission is not scientific success;
inspect actual prologs/hardware, case seals, full independent comparisons and
cleanup before promoting results. No retries or additional allocations are
automatically authorized.
Controller readback confirms GPU14400888 PENDING(Priority),gpu-h200/inferno,
8CPU160G16h and typed H200GRES; both CPU jobs are PENDING(Dependency),
cpu-small/inferno,24CPU160G24h,`afterany:14400888`, and their ReqTRES contains
**no GPU**. Prolog/actual device evidence remains pending until allocation.

`grouped_jobs.py` calls the existing separated case runner in the **same
orchestration process**, so it does not nest independently detached supervisor
groups. Scientific workers remain fresh processes with complete reaping.
Completed system-level GPU results have their own clean case marker and seals;
an unrelated system failure does not erase or invalidate those complete
prefix results. No failed system is automatically retried. CPU validation
accepts only its system's complete sealed three-repeat output. Cancellation
or unproved child cleanup stops further launches and prevents scratch deletion.
The CPU blocks also perform their three-reference gauge-repeat comparisons.

The old eight-job plan below was prepared only and is not to be submitted.
Its source and generated metadata remain intact for audit.

## Prepared checkpoint

**PREPARED_NOT_SUBMITTED**, helper
`23f55d5d361fee22cb5d51915417627561dbe1a6`; all81 offline tests and shell
syntax checks passed, including timeout/TERM scratch cleanup, early peer
configuration failure and explicit completed-producer CPU submission.
Scientific/CMake/external trees remain unchanged fromd471cd93fa.
Clean remote helper:
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-split-23f55d5d36`.
Plan:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-split-23f55d5d36/plan.json`,
SHA256`d5a73d535b5b500e042cc3d9636791ecf6d9659126402b61c4ac8c47fffe484e`.
Preparation task`bffa2edf3` completed0 and made no scheduler submissions.
The two GPU/six CPU jobs await an explicit QoS/resource decision.

The user rejected holding an H200 while independent CPU references ran and
explicitly approved **“Cancel and split”**. Mixed job14396956 was cancelled:
job accounting reports CANCELLED after4h50m05s, batch143:0, step1:0.
Its durable marker is exit143/TERM1; the final checkpoint records the expected
termination exception in the first nanotube CPU reference, not a scientific
failure or a finished nanotube result. Completed peptide remains
`PASS_FULL_SYSTEM_DIAGNOSTIC` with all three independently checked pairs.
Old outputs, source checkouts, configurations and processes are not rewritten.

## Prepared layout — no new paid submission implied

Only nanotube and protein83 remain:

- **Two GPU-only jobs**, one per system, three fresh GPU calculations each.
  One H200,8host threads,160GiB RAM,8h cap per job.
  The worker-command API rejects a CPU reference or `oracle_from` input in a
  GPU job. Normal host orchestration in the hybrid GPU calculation remains;
  no separate CPU reference, CPU oracle or CPU warmup is launched there.
- **Six CPU-only jobs**, one per system/repeat. Each runs one independent
  exact CPU reference and one fixed-gauge native CPU oracle.
  24threads,160GiB RAM,8h cap; exclusive Gold6226/core24/localNVMe CPU-small.
  No GPU GRES, hidden CUDA visibility, actual allocation checks reject a GPU.
  Scratch is node-local rather than the mixed job's shared Lustre DiskDFJK path.
- CPU jobs use scheduler `afterany` + kill-invalid-dependency and require the
  producer's clean terminal marker, complete successful GPU/repeat artifact,
  exact source/config/input identities and hashed raw/matrix files.
  **No GPU allocation waits for CPU validation.** No allocated CPU worker spins
  waiting for an incomplete GPU producer either; it fails closed immediately.
  The peer GPU runtime closure/thread-only policy is validated before the
  expensive CPU reference, not deferred until its later oracle.
- Preserve peptide's existing results; do not rerun it just to change scheduling.

The immutable whole science/stage remainsd471cd93fa/build14377089. A new clean
Git helper checkout is required; no native scientific change or installed
Python overlay is involved. The runtime configurations differ ONLY in declared
thread count24CPU versus8GPU-host. Both complete runtime closures are checked
before allowing that cross-role oracle comparison. Geometry, basis inventories,
screening1e-16, exact coupled source response, frozen counts, scaling, all23
fields and numerical tolerances remain unchanged.

This is a declared24-core CPU versus8-core/H200 hybrid resource policy, **not
a same-thread timing comparison**. Peptide's old exact CPU diagnostics used
8threads; its timings are not silently relabeled as24-thread measurements.
Production scalar-rescaled CPU benchmarks remain a different method. No
production speedup or accepted timing is inferred from these instrumented calls.
After all CPU jobs succeed, cross-repeat/aggregate qualification remains a
separate analysis step; a GPU execution-only result is not independent validation.

## Preparation and submission controls

`submit_split.py` without `--submit-one` only creates a fresh immutable plan,
derived runtime configurations and draft manifests; **it never calls sbatch**.
Each later `--submit-one LABEL` requires an explicit account/QoS, a clean
exact helper checkout and sealed prepared files. It writes a fresh final
manifest, validates the actual command with test-only, then makes at most one
scheduler call. Ambiguous responses/failed routes are not automatically retried.
GPU parent job IDs are captured before CPU dependencies are materialized.
Submit normal CPU dependencies while the producer remains registered.
For delayed CPU validation after SLURM purges that record, explicit
`--completed-producer` verifies the same complete artifact/config/cleanup seals
and submits CPU-only work without a stale dependency; it never resubmits a GPU.

Scratch is freshly created0700 by the job payload and bound to its original
device/inode. After every owned descendant group has been reaped, the worker
removes only that attempt's temporary root, including on timeout/TERM. It
refuses deletion for a replaced root or unreaped group. Diagnostic outputs and
old peptide artifacts are outside that root and retained. Uncatchable SIGKILL
cannot promise cleanup; a missing final result is never promoted to success.

Fresh inferno account pricing reports monthly gts-cs207$0.93, below its$1 floor;
all eight planned allocations would select gts-cs207-chemx. Maximums:
two H200 jobs at$11.31 each and six CPU jobs at about$1.58 each,
**about$32.07 total additional cap**, not an ETA or an automatic authorization.
Every job is also eligible for an explicit free embers choice, with the usual
preemption risk. Recheck the selector immediately before any paid submission.
