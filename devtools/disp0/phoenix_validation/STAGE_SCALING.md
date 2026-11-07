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

## Peptide GPU passed; CPU qualification submitted 2026-11-07

GPU job15233604 completed0:0 without TERM in7min14s. Final artifact status:
`PASS_GPU_REPEATS_AND_TRACE_AWAITING_CPU`. The small H200 gate, three fresh
baseline repeat checks, separate trace repeat, retirement checks, owned group
reaping and owned scratch cleanup all passed.

- Unprofiled energy samples:37.137074203 /37.009711522 /36.967758893s.
- Median:37.009711522s; no new accepted CPU/GPU ratio yet.
- Separate traced energy:52.307596960s; traced fragment:43.109s.
- GPU localization0.767025920, felst5.206729961, fexch0.026774131,
  find9.069857403, fdisp3.908926633s.
- Additional traced workflow overhead11.844163563s and outer factory/
  cleanup12.285522389s remain residuals, not localization or dispersion.
- These five windows plus both residuals close to43.109s; none is rescaled
  into the37.010s unprofiled median. Instrumentation is intrusive.
- Final producer SHA256:
  `d4a8feb328aca041844143852f15e3a39661ac7822fd43d538d784f545cf3806`.
  Local downloaded bytes match the immutable hash sealed by CPU submission.
- Trace SHA256:
  `b300dc349d8b640e86ffaaa60772961757d5b107c37dace14d5d1a4f832618bb`.
- Traced worker result SHA256:
  `61a1e50ce34d842b3d3878807d61e89b448b20b29bca11ed4204701a3255f021`.
- Traced timer SHA256:
  `0926418c29c01ad1013697a3c576c29dbb6ce2d7ae46a17e073042021cc3e637`.

Three CPU-only reference+oracle jobs were submitted successfully, each with
one-hour cap,24Gold6226cores,160GiB,exclusive local-NVMe CPU-small and embers:

| Repeat | Actual job | Submission root |
|---|---:|---|
| r0 | 15278472 | `peptide-cpu-r0/` |
| r1 | 15278474 | `peptide-cpu-r1/` |
| r2 | 15278476 | `peptide-cpu-r2/` |

All three were confirmed RUNNING after submission; no GPU is reserved.
Test-only IDs15278471/15278473/15278475 are not actual submitted jobs.
Monitor `b4ea7cfb9` checks all three hourly. **These CPU jobs already exist;
do not resubmit them when the earlier GPU monitor delivers completion.**
After final artifacts pass, compare all three CPU references with the existing
strict repeat helper before publishing a ratio of the new per-arm medians.
Nanotube15233651 and Protein8315233653 were still pending at this checkpoint,
with the peptide prerequisite satisfied.

## Peptide qualification passed

CPU allocations15278472/15278474/15278476 completed0:0 with no TERM in
9min38s/9min35s/9min35s. Each produced
`PASS_CPU_REFERENCE_AND_ORACLE_NEEDS_THREE_REPEAT_AGGREGATION`, passed all23
independent fixed-GPU-gauge oracle fields and signed/whole-energy controls,
reported no retirement, reaped both workers and cleaned its owned scratch.

The separate `qualify_stage_scaling.py` consumer then rechecked immutable
GPU/CPU submission and producer seals, all worker/result/timer/stdout/NPZ/F-SAPT
file hashes, source/helper/config/input identities, independent oracle matrices
at the original tolerances, and **strict CPU cross-repeat agreement** using the
original frozen helper. It does not modify any scientific producer or consume
the extra traced timing as a speedup sample.

| Baseline energy timing | r0 (s) | r1 (s) | r2 (s) | Median (s) |
|---|---:|---:|---:|---:|
| CPU24 | 266.905526779 | 262.917661584 | 263.234152672 | 263.234152672 |
| H200+8 host CPUs | 37.137074203 | 37.009711522 | 36.967758893 | 37.009711522 |

**Qualified ratio of medians:7.112569697154473×.**
This fresh cohort reproduces the earlier approximately7× result; it is not a
new scientific/kernel optimization or a larger-system scaling point.
The separate52.307596960s traced GPU process is excluded from the ratio.

CPU cross-repeat maximum field differences:
9.924094879210088e-11 /8.722905731062269e-11.
Independent oracle largest major energy-matrix entry error:
1.0735120778426754e-10Eh (`IndBA_AB`, repeat1).

- Qualification recipe pin:`024957c4b3`.
- Output:`stage-scaling-20261106-01/peptide-qualification-01/qualification.json`.
- Status:`PASS_MATCHED_INSTRUMENTED_BENCHMARK`; accepted speedup count1.
- Qualification SHA256:
  `017095fc4e4cecc94f09ea28a1e0f6c4af4aeb31e7f5b36400ab01a1eb0e6db6`.
- CPU result-controller SHA256 values:
  `b87b6dad923d56dc82a196344f33d503f6c5d7070afba7a3fa30365e7f865bdc`,
  `139de65332050280db45a44761aa795f6c886fb6cee4133a09913aa21bc27f79`,
  `34a7e792f28b58c054dceb29c197b6b966dd9d600ffddc450bd364468562dafd`.

The immutable GPU producer still has its original awaiting-CPU status and
accepted count0. Qualification is the separate consumer above. Downloaded
qualification/CPU documents were matched to those sealed SHA256 values.
The HTML now defaults its stage view to the new CPU-baseline/H200-trace
comparison, with an explicit selector retaining the original matched timers.
Both localization and all four fragment stages have actual H200 lines; no
per-stage speedup is claimed from a3-sample CPU baseline versus one intrusive
GPU trace. The original7.004× campaign and its clocks remain separately labeled.
Nanotube15233651 and Protein8315233653 remain pending at this qualification
checkpoint; no larger-system ratio is inferred.
