# Separate GPU execution from CPU reference/oracle work

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
