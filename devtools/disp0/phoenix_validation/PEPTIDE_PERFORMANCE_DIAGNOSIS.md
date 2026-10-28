# Peptide H200 latency: measured setup bottleneck

## Scope

Read-only inspection of finalized GPU repeats0/1 from job14481883, not a new
allocation or modification of the running benchmark. Scientific stage
`d2a98abe35`, helper491ee751f4, physical H200 confirmed by the water control.
GPU8host threads versus CPU24Gold6226 threads; exact per-source CPHF in both.
The CPU median is295.103911s (n3). GPU repeat/matrix qualification is still
required: the times below are observations, not an accepted speedup.

Input/native dimensions:24atoms,356AOs,20occupied/5frozen/15active per monomer,
336virtuals per monomer. Snapshot copies and NPZ serialization are outside
the recorded whole-energy timer in both arms.

## Reproducible measurements

Seconds:

| Measured interval | GPU repeat0 | GPU repeat1 |
|---|---:|---:|
| Whole energy call | 994.512498 | 994.536646 |
| Full fragment continuation | 989.942060 | 989.982170 |
| Before factory → JKFIT constructed | 373.936899 | 374.218195 |
| JKFIT constructed → RIFIT/JKFIT ready | 448.854754 | 449.184192 |
| Both plans ready → before RIFIT cleanup | 162.870418 | 162.305221 |
| Blocked dispersion, included in preceding interval | 83.173784 | 83.237590 |
| DFMO producer, included in dispersion | 82.379087 | 82.418129 |

The two setup/observation windows consume822.79/823.40s, about83% of the
whole call. Factory teardown and other wrapper work account for the small
remaining gap. These are synchronized stage boundaries recorded by
`full_observe.observed_factory`, not CUDA-event kernel timings. Construction
can include real integral work, JIT, memory/workspace preparation and audits:
the saved data do **not** separate those subcomponents.

The broad conclusion is measured, not hypothetical: **fresh native factory
and plan setup dominates the end-to-end run**. The second fresh process
repeats the penalty; it is not just the first timed sample being an outlier.
Do not subtract this cost and advertise the remainder as a full-call speedup.

CPU fragment phases (n3):

| Phase | Repeat0 | Repeat1 | Repeat2 |
|---|---:|---:|---:|
| find | 241.426744 | 271.941046 | 267.434463 |
| fdisp0 | 3.128554 | 4.303933 | 4.407279 |

## Why the earlier microbench gains did not establish readiness

1. Response workspace measurements excluded session creation and native
   factor production. This peptide uses `q_batch=1308`, covering the entire
   JKFIT auxiliary dimension in **one Q slab**. The specific advantage of
   eliminating repeated per-Q-slab allocations/scopes is barely exercised.
   The resident response factors are reused: JKFIT producer reports only5
   DFMO batches overall, not one native DFMO batch per PCG iteration.
   There are88source solves and1297total PCG iterations in both GPU repeats.

2. The native blocked-dispersion cache microbenchmark began **after plan
   construction**. It therefore did not measure the823s whole-stage cost.
   Here the64MiB budget allocates60,076,800bytes and caches5 of14 equal
   S slabs. Its65hits reduce DFMO batches from the uncached schedule's210
   to145, but do not eliminate most factor production.

3. Dispersion itself remains expensive:82.38 of83.17s are inside the DFMO
   producer path, versus3.1–4.4s for CPU fdisp0. The producer timer includes
   fresh identity checks, packing, allocation/workspace queries, native
   compute and synchronization; it must **not** be called vendor-kernel time.
   Coefficient H2D bytes are0 and the raw HF upload is29,414,144bytes.
   In particular, the dominant setup windows precede raw-HF upload.

## Source-supported leads, not yet measured subcauses

- `sapt_cuest_plan_owner.OwnedDFPlan.build` creates an owned native context
  for each plan and explicitly configures `jit_compile_threads=1`.
  The sibling RIFIT provider shares the CUDA session but creates another
  owned native context/plan. Serialized/repeated native JIT initialization
  is a concrete lead; constructor-level timings are needed before assigning
  hundreds of seconds specifically to JIT rather than integral preparation.
- `sapt_cuest_dfmo_cuda.compute_device` performs an identity check,
  coefficient packing, workspace query/allocation, native compute, device
  synchronization and scoped cleanup for every DFMO batch.
- All14 equal S slabs would require168,215,040bytes (about160.4MiB), derived
  from the measured5-slab allocation and the observed336virtual/block24
  layout. If independently admitted by live/native memory guards, full reuse
  would reduce producer calls from145 to28. This is a work-count prediction,
  **not a demonstrated runtime gain** or permission to raise a running cap.

## Next discriminating measurements

Instrument the individual factory stages without changing math:
API/byte admission, native context creation, basis/pair construction,
DF-plan workspace query/create and teardown. Record wall and process CPU time
to distinguish compilation/CPU work from GPU/IO waiting. Measure identity
audit time separately from native DFMO compute and synchronization.

Then test one change at a time: bounded JIT parallelism/owned-context reuse,
and a fully admitted peptide S-cache budget. Preserve native ownership,
failure cleanup, original FP64 equations, output order and numerical checks.
Do not remove audits, borrow an unowned SCF handle or free native creation
workspace without a verified lifetime contract. No running source, guard,
job configuration or scientific output was changed for this diagnosis.

## Artifacts

GPU:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-benchmark-8h-inferno-491ee751f4/gpu/job-14481883/peptide-r{0,1}/result.json`.

CPU:
same run root, `peptide/job-14481886/peptide-r{0,1,2}/result.json`.
Read the fields `phases`, `fragment_execution`, `runtime.full_device_observations`
and `benchmark_whole_energy_seconds`; do not add nested timing intervals.
