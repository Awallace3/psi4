# Conditional HF-E2 timing campaign

Deadline assumption: **2026-10-20 08:00 America/New_York**. Not a promise.
Coherent installed stage and actual signed/device correctness must pass first.
No build, repair, automatic retry, or account fallback inside GPU allocations.

## Exact maximum footprint proposed

| Purpose | Jobs | GRES | CPUs | Host memory | Wall limit | Account/QoS | Charge bound |
| --- | ---: | --- | ---: | --- | --- | --- | ---: |
| Signed/device gate | 1 | gpu:h200:1 | 4 | 32G | 00:30:00 | gts-cs207/inferno | $0.71 |
| Matched HF timing | 1 | gpu:h200:1 | 8 | 512G | 06:00:00 maximum | gts-cs207/inferno | $8.48 |

Combined ceiling **$9.19**. No additional CPU timing allocation.
Live partition query reports `TRESBillingWeights=gres/GPU=52381`, with no
CPU/memory billing weights; rate is 2223529 billing-minutes per dollar.
Selector snapshot available balance $29.51; **refresh capacity immediately
before each submission** and stop if gts-cs207 cannot cover it.
Actual scheduling/routing still requires passing exact-resource test-only.
Both allocations use no-requeue and fresh isolated run/scratch roots.
CPU build14196330 failed its post-install Python import check on free embers;
its source, environment and artifacts remain unchanged. No GPU readiness
or additional CPU recovery allocation is implied.

The two GPU approvals are conditional and each permits exactly one job.
Neither has been consumed. Parent14185006 and all parent artifacts remain
untouched. This is distinct HF-E2 work, not a copy of the PBE0-D4 cohort.

## Inputs, science, and count

Read-only suite: `~/data/timing_test_suite/geometries/systems.py`, SHA256
`72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb`.
Phoenix execution must use the supplied immutable snapshot
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-cpks/scal631-fc851a8346-20261016/geometries/systems.py`
with that exact SHA (read-only). The current Phoenix home copy has a
different module hash and is not admitted. Verify snapshot bytes and all four
geometry-string hashes before launching; never modify the parent snapshot.
Routine systems only: peptide, nanotube, protein83, protein157. No PLA15.
Spherical `6-31+G**`, def2-universal-jkfit / aug-cc-pVDZ-RI; NBF
250/548/922/1786 respectively. Retain geometry hashes and actual basis counts.

HF SAPT through SAPT(DFT), opt-in canonical EINSUMS Disp20/Exch-Disp20,
native FP64, no D4/DFT response. Coupled HF induction/corrections and frozen
spaces must be identical across arms and explicitly recorded. Other terms,
SCF and DFHelper remain CPU: **GPU contractions with CPU elementwise and
reductions**, not whole-run GPU acceleration.

24 timed whole-energy calls: four systems × two arms × three independent
fresh processes. **No extra full-energy reconstruction calls.** Each process
performs one fresh bounded synthetic SAME-kernel probe after timing, plus
one separately logged native positive control: at most **24 timed full
calls +24 small synthetic probes +24 positive controls**.
No timed adapters/graphs/arrays/DF scratch are retained or reused; normal
production cleanup stays inside the measured boundaries. Synthetic probe
dimensions cap virtual edges at4, occupied widths at8 and RI rows at64
(matrix widths at most32); only scalar policy metadata selects its shape.
It uses the same local CPU-disabled-GPUPlacement or natural-GPU policy,
never a forced GPU target. The separately bounded naturally eligible positive
control has fixed1920×1920 outputs and approximately355MiB of host tensors
plus native device/workspace overhead. These diagnostics validate **diagnostic
policy/control/logger behavior only**, not actual timed dispatch. Measured
results are **policy-controlled 8-CPU versus hybrid timing**, with actual
timed GPU dispatch explicitly unverified unless independent nonperturbing
timed-call evidence becomes available. All diagnostics are excluded from
medians but included in deadline/six-hour admission; incomplete points remain
incomplete.
Both arms use the same node, eight host threads, stage,
memory, block policy and science. The CPU arm explicitly disables
**GPUPlacement** locally for the same E2 contraction graph and executes its
CPU kernels; it does not disable CPU placement. The GPU arm retains the
natural cost model. Record timed policy metadata and separate post-timing
diagnostic evidence; never present the latter as timed-call execution proof.
Small/CPU-fallback
blocks are legitimate, not forced onto GPU. Record any diagnostic calls
separately; they are not timed repeats and must fit the same six-hour limit.

## Memory and time caveats

Basis-only conservative protein157 bounds: dense DFJK 173.242GiB,
transformed DF-factor disk 155.782GiB, before other arrays, page cache and
temporary disk copies. These are estimates, not measured RSS or VRAM.
Initial per-process Psi4 budget proposed 128GiB within a 512G allocation;
record disk fallback, live RSS and scratch peaks. Preserve bounded tensor
blocks and do not claim the entire memory budget is a peak-RSS bound.
Persistent job-owned scratch must have sufficient free space and quota.
Check actual free RAM and persistent filesystem/quota before each protein
launch, not just the scheduler request. Preserve partial/failed points.
The 32G gate is not a container for these protein workloads.

No SAPT runtime estimate is established by the suite's RHF table.
Six hours is a ceiling, not evidence that all 24 calls will finish.
Allow at least 15min for final artifact validation before the deadline;
if readiness/queue/start times do not leave enough time, report partial or
blocked results rather than inventing ratios or discarding failed cases.

## Measurement and acceptance

Fresh process and fresh timer/output/scratch directory for every point.
No profiler, tracing interposer or oracle work in measured intervals.
Whole `energy()` wall is primary; synchronized dispersion end-to-end
includes construction, packing, transfer, CPU elementwise/reductions and
cache planning/replay. These are not kernel-only times.
Kernel-only time is unavailable unless independently measured.

Retain full hierarchical timer records and nonoverlapping phase cuts,
nested diagnostics separately, exact options, hardware, stage/library hashes,
source pins, signed components and total/sum rules. Do not sum timer.dat's
flat and tree sections or append multiple measurements to one file.
Use medians of three whole-call walls and the actual median-run phase vector;
report min/max and sample counts, no fitted confidence intervals.
CPU/GPU signed comparisons must pass declared physical tolerances; retain
all deltas and failures. Do not reuse PBE0-D4 energies or its CPU/GPU numbers
as an HF baseline. A ratio below one is a slowdown and must be reported.
Label this **8-CPU versus hybrid GPU**, not historical24-CPU/PBE0 speedup
or fully resident GPU.

## Superseded proposal history

An earlier proposal added24 full untimed reconstructions, for48 full calls.
Review showed those traces still would not prove actual timed dispatch.
Coordinator cleared the bounded synthetic replacement above; the48-full-call
proposal is superseded, not a second allocation or extra work authorization.
Static protein157 product work is approximately2.995PFLOP per full call;
a hypothetical1TFLOP/s illustration is work/risk accounting only, not a
measured throughput/runtime estimate. Even the24-call target may be partial.
