# Bounded H200 stage/scaling campaign

User requested actual stage timings and larger systems for more meaningful
CPU/GPU comparisons. Selected frozen timing-suite cases:

| Case | Atoms | Spherical aug-cc-pVDZ NBF | GPU allocation cap |
|---|---:|---:|---:|
| peptide | 24 | 356 | 1h |
| nanotube | 42 | 742 | 4h |
| protein83 | 83 | 1293 | 8h |

**Protein157 is excluded by user instruction.** No other systems, basis changes,
functional-group assignment files, or cheaper response partition are introduced.

## Execution identity

Scientific source and whole installation remain frozen at
`927283f8ed0a94d42179c35dd2a1a5bce674347e`. The separately pinned outer helper
`stage_scaling.py` reuses the authenticated original sealed bootstrap, runtime,
scientific workers and numerical checks from that frozen checkout. No scientific
overlay, stage rebuild, shared-environment installation or advanced source.

Method remains **scaled HF-reference F-SAPT/aug-cc-pVDZ**, spherical FP64,
exact coupled response, block24 and64MiB factor cache. GRAC shifts and DFT
monomer SCFs are not executed by this method.

## Timing design

Each GPU case first passes the existing full small H200 correctness gate.
It then runs three **unprofiled fresh-process** baseline samples, with the
unchanged strict repeat comparison, followed by one **separate synchronized
traced** sample that must pass the same repeat comparison.

The observer authenticates the installed workflow and source-shaped boundaries
before collecting localization, felst, fexch, find and fdisp intervals. GPU
localization means overlap/IAO metrics plus both monomers' IAO/IBO work; earlier
plans/uploads and later scaling/downloads/cleanup remain explicit residuals.
The traced process's stages close to **its own** full-fragment timer; they are
never rescaled into an unprofiled baseline. Raw `timer.dat`, `psi4.out`, matrix
and raw-HF artifacts and hashes are retained for every completed sample.

The sixteen requested report stages remain distinct. In this HF route `disp`
is dispatch/metadata, whereas `fdisp` contains actual fragment dispersion.
Cold-inclusive SCF clocks retain startup; native first/later JK logs are kept
without inventing a warmed dimer-SCF speedup.

## Resource and queue policy

- Explicit `gts-cs207-chemx / gpu-h200 / embers`, H200 constraint and one typed
  H200 GRES, eight host CPUs,160GiB host allocation, no requeue/automatic retry.
- Unchanged96GiB Psi4 host workspace,120GiB device ceiling and12GiB opaque-native
  accounting allowance. Actual memory/workspace guards remain active.
- Fresh aliases/cache homes and post-srun scratch; TERM-aware process groups,
  retirement receipts and reaping before owned scratch cleanup.
- Larger GPU submissions can use `afterany` on the peptide preflight, with
  kill-on-invalid-dependency and a runtime requirement for the completed
  successful artifact. No allocation waits for a parent or CPU reference.
- All results remain unaccepted until independent CPU qualification.

Nanotube historically failed a localized-orbital repeat check. Protein83
previously encountered simultaneous-plan workspace admission and its first
24-core CPU reference exceeded eight hours. Neither guard is relaxed: failures
remain visible, with no fabricated point on the scaling curve.

## CPU follow-up

Only after a GPU producer completes and cleans up, submit CPU-only jobs for
peptide and nanotube: one fresh24-core Gold6226 reference plus its independent
fixed-GPU-gauge CPU oracle per repeat. CPU config is derived from the matching
GPU config **by changing threads8→24 only**; use actual node-local scratch.
The three CPU references must also pass strict repeat qualification before
publishing a median ratio.

Protein83 GPU timings can be collected, but the known over-eight-hour24-core CPU
reference is not automatically resubmitted on embers. A separate scientifically
valid resource/chunking plan or explicit inferno approval is required for its
complete CPU speedup point.

## Launch interface

Run the new helper from a clean detached outer checkout on Phoenix:

```text
stage_scaling.py --kind gpu --system peptide|nanotube|protein83
  --base-config <original qualified GPU runtime-config.json>
  --inputs <frozen inputs.json> --inputs-sha256 <sha>
  --root <fresh absolute case root>
  [--prerequisite <peptide submission root>]
```

It runs `sbatch --test-only` first, saves scheduler responses and the exact
manifest, and records the numeric job ID without automatic retries. CPU mode
additionally requires `--gpu-job-root` and `--repeat 0|1|2`.

## Submitted 2026-11-06

Campaign root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/stage-scaling-20261106-01`.
Frozen outer checkout:
`/storage/project/r-cs207-0/awallace43/gits/psi4.stage-scaling-429826db9f`,
pin `429826db9f192b30e7e6b7509bae11ec3348b346`.

| System | Actual submitted job | Wall cap | Submission root | Prerequisite |
|---|---:|---:|---|---|
| peptide | 15233604 | 1h | `peptide-gpu/` | None |
| nanotube | 15233651 | 4h | `nanotube-gpu/` | `afterany:15233604` plus successful finalized artifact |
| protein83 | 15233653 | 8h | `protein83-gpu/` | `afterany:15233604` plus successful finalized artifact |

All three `--test-only` checks and actual submissions returned0. The test-only
request IDs15233603/15233650/15233652 are **not** the actual submitted jobs.
Peptide initially pends for Priority on embers; the others are gated behind it.
Scheduler test-only start predictions were around2026-11-15, not a promised ETA.
No inferno jobs, cancellations, requeues or shared-environment changes.

Hourly official `watch_jobs.sh` monitors:
peptide `b2d589ae4`, nanotube `b3a55478f`, protein83 `b5a9df59b`.
Their terminal notifications trigger artifact qualification and eligible
CPU-only reference/oracle submissions. No new timing or speedup is claimed at
submission. Existing scientific source remains clean/frozen at927283f8ed.

Prelaunch verification:18 local tests passed (5 campaign admission/CLI/producer
tests,7 observer tests,6 timer-parser/admission tests), shell syntax and
diff-whitespace checks passed. The frozen Phoenix workflow resolves the new
observed localization boundary at88, followed by95/97/99/103/112. Independent
inspect-only launch review found and prompted fixes to frozen `DISP0_PIN`,
writable CPU scratch selection, whole-coordinator TERM handling, and finalized
GPU producer/manifest/timing seals before CPU admission.
