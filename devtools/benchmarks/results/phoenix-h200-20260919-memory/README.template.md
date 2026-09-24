# cuEST SAPT(DFT) after the memory merge — Phoenix H200, 2026-09-19

This directory repeats the [2026-09-10 paired campaign](../phoenix-h200-20260910/)
on a build that carries the process-memory work merged from
`saptdft_ein_fi_option_d4`, and adds the thing that campaign could not report at
all: **how much host and device memory each arm actually holds.**

The merge is three changes that only make sense together:

- a **process memory ledger** (`MemoryClaim`, `libpsi4util/memory_ledger.h`), a
  process-wide tally of the budget-sized resident buffers that are already live
  — every JK's in-core DF integrals, every V's collocation cache. It exists
  because `scf_initialize()` divides the SCF budget out of the global memory
  setting, which describes an *empty* process. SAPT(DFT) is not an empty
  process: it runs monomer SCFs while a dimer JK and a collocation cache are
  still resident, and each new SCF was being handed memory somebody else was
  already holding.
- **`release_freed_memory()`** (`malloc_trim(0)`), called when the collocation
  cache is dropped. glibc keeps freed arena pages mapped, so a buffer that is
  freed does not come off the resident set, and the next allocation sizes
  itself against a high-water mark nothing is using.
- a **cost-based collocation cache fill**, replacing a 1-in-N stride sieve that
  chose which blocks to cache without reference to what they cost.

Contents:

- **This file** — what ran, what changed, and what the change is attributable to.
- [`PROVENANCE.md`](PROVENANCE.md) — commits, hashes, jobs, settings, and the two
  jobs that failed before these three succeeded.
- `premerge-delta.md` — the controlled A/B: this build against its own first
  parent. **This is the only table here that isolates the merge.**
- `attribution-premerge.md` — the same phase decomposition on the control arm.
- `cuest-delta.md` — the second one-variable A/B: the same `core.so` against
  libcuest 0.2.1.2 and 0.2.2.2. `premerge-cuest-delta.md` is the same swap on
  the control build, and it is the table that retracts the first pass's GPU
  claim (see below).
- `protein157/summary.md`, `accuracy-protein157.md`,
  `attribution-protein157.md` — the 157-atom case, one CPU run and one GPU run.
- `paired/summary.md` — the paired timings and the memory tables.
- `attribution.md`, `dfk-tflops.md`, `grac-cost.md`, `accuracy.md`,
  `thread-scaling-*.md`, `host-speed.md` — generated the same way as in the
  2026-09-10 directory. `regenerate.sh` rebuilds all of them from the raw trees.

`protein157` (157 atoms, 1786 basis functions) is reported as **one CPU run
and one GPU run**, in its own tables, and is never pooled into the six-case
medians. Its 8-thread CPU arm takes 3.5 h. Embers preempted it in five of six
attempts, and SAPT(DFT) has no checkpoint to resume from. The CPU run that
finished is cpu-3 of P3 (job 13395715). Its partner is gpu-3 of P1 (job
13429862). That tree is marked FAILED only because its own CPU case was
preempted at 5342 s, after all three of its GPU cases had returned rc=0.
`regenerate.sh` checks that GPU case individually and relinks both runs as
repeat 1 in `protein157-single-pair/`. `SOURCES.txt` there records the original
names. This is the one speedup in the directory that crosses two allocations,
so the two hosts' canaries are quoted beside it.

## Status

All jobs ran on `--qos=embers`. No inferno job was submitted.

| Job | Arm | Build | libcuest | Partition | Elapsed | State | Used |
|---|---|---|---|---|---:|---|---|
| 13358747 | M1 — paired CPU/GPU, 8 threads | merge `ee6161a3b6` | 0.2.1.2 | gpu-h200 | 00:37:23 | COMPLETED | `cuest-delta.md` control |
| 13358750 | M2 — CPU-only, 8 vs 24 threads | merge `ee6161a3b6` | — | cpu-small | 00:41:38 | COMPLETED | thread scaling |
| 13367763 | M3 — paired CPU/GPU, 8 threads | pre-merge `d91b5f8e81` | 0.2.1.2 | gpu-h200 | 00:40:00 | COMPLETED | `premerge-cuest-delta.md` control |
| 13395711 | **M4 — paired CPU/GPU, 8 threads** | merge `ee6161a3b6` | 0.2.2.2 | gpu-h200 | 00:39:13 | COMPLETED | paired tables (A) |
| 13395712 | **M5 — paired CPU/GPU, 8 threads** | pre-merge `d91b5f8e81` | 0.2.2.2 | gpu-h200 | 00:37:53 | COMPLETED | `premerge-delta.md` control (B) |
| 13395715 | P3 — protein157 CPU | merge `ee6161a3b6` | 0.2.2.2 | gpu-h200 | 03:29:44 | COMPLETED | protein157 CPU |
| 13429862 | P1 — protein157 GPU ×3 + CPU | merge `ee6161a3b6` | 0.2.2.2 | gpu-h200 | 01:54:36 | PREEMPTED | protein157 GPU (gpu-3) |

Trees kept but not used:

- 13376151 (M4), 13376152 (M5), 13376153 (P1) were preempted on
  `atl1-1-02-012-23-0`, which read 24.5 GF/s per core against the 70 GF/s gate.
  They carry `metadata/DEGRADED-HOST`.
- 13395713 (P1) failed the host-speed gate at start-up (exit 75, 24.6 GF/s per
  core on `atl1-1-02-014-9-0`). Both nodes are now excluded in the P1/P2 scripts.
- 13395714, 13429863, 13480139 (P2) and 13480138 (P1) were preempted before
  their CPU case finished. 13480138's three GPU runs completed at 417.6–419.8 s
  and 13429862's at 419.3–507.3 s. They are consistent with the one reported here, but
  they are not reported because no CPU run pairs with them.

M4 and M5 verified themselves in-job: `cases=36 failed=0`, host memory on every
case, and device memory on every GPU case. M5 additionally verified that it is
the control build (`cases_with_a_ledger_reading=none (expected)`). Their
canaries read 84.9 and 84.2 GF/s per core, the same host class as M1/M3 (84.3 /
84.2).

## Read this before reading any speedup number here

**The controlled A/B is M4 against M5.** M5 runs `d91b5f8e81`, the merge
commit's own first parent. It was built from the same worktree with the same
compiler flags, measured by the byte-identical driver, on the same partition,
against the same libcuest 0.2.2.2, and it ran concurrently with M4. M4 minus M5
is the merge and nothing else.

**This replaces the first pass's M1-vs-M3 comparison, and it retracts that
comparison's headline.** The first pass reported that the merge made the GPU arm
17–29% faster. M4 vs M5 shows no such effect: GPU wall time moves by at most 1%.
`premerge-cuest-delta.md` shows where the gap went. On the control build, the
libcuest swap alone makes the GPU arm 16–28% faster (benzene cc-pVDZ
22.50→16.47 s, benzene aug-cc-pVDZ 26.46→19.04 s, nanotube 52.44→39.23 s,
peptide 28.77→24.13 s). On the merged build the same swap moves nothing
(`cuest-delta.md`). This directory cannot tell apart two explanations:

- M3's GPU arm was slow for a reason its CPU canary does not see. No GPU canary
  exists.
- The merge and libcuest 0.2.2.2 remove the same GPU cost, so it shows up only
  on the build/library combination that has neither.

Either way, on the library the branch now ships, the merge is a host-memory
change and not a GPU-speed change.

**The comparison against the [2026-09-10 campaign](../phoenix-h200-20260910/)
is not attributable.** It uses the same protocol, partition and six cases, and
every speedup is higher (2.55→2.99×, 5.85→6.43×, 2.23→2.71×, 7.56→8.26×). But
`e971d2957b` (the 2026-09-10 build) and `d91b5f8e81` are 55 commits apart. They
include the move of the XC quadrature onto the device (`52431fc2c0`,
`60322c4275`, `4a6164ef71`) and GRAC-protocol changes (`303f07ab2c`,
`dc4a74cb98`), and the libcuest version differs too. Read the cross-campaign
numbers as a level, not an effect.

## The controlled A/B: what the merge actually did

<!-- PREMERGE -->

The merge is a host-memory win and nothing else measurable:

- **CPU host peak falls 7–15%** on the four substantial cases. The nanotube
  drops 31688→27051 MiB, 4.5 GiB off the job that set the 112 GiB request.
  Benzene aug-cc-pVDZ drops 16521→14893 MiB, benzene cc-pVDZ 7964→7423, and
  peptide 7796→7297. This is the ledger doing what it was written for: the
  monomer SCFs stop being handed memory that the dimer JK and the collocation
  cache already hold.
- **GPU wall time does not move**: within scatter or +1% on every case.
- **CPU wall time reads 0–7% slower** on M4 (3–7% on the four cases that clear
  scatter). Same-build reruns on different nodes move 1–3% (M1→M4 in
  `cuest-delta.md`), so a CPU cost of a few percent from the merge cannot be
  ruled out. The merge does change what the CPU arm allocates.
- **GPU host peak is flat** at 0.9–1.9 GiB. The GPU arm was never the one
  over-committing the host.

The device column is the weakest evidence in this directory. It is a 0.5 s
sample, and its scatter across three identical repeats reaches ±1764 MiB on the
nanotube. Two cases, benzene cc-pVDZ and peptide at ~3140→2800 MiB, move further
than their own repeats. That is suggestive, not a measurement to size a device
against.

## The second A/B: the libcuest bump, 0.2.1.2 to 0.2.2.2

<!-- CUESTDELTA -->

This one is cleaner than any other comparison in this directory, because the
thing under test is the only thing that moved. libcuest 0.2.2.2 keeps the soname
and changes one header line, `CUEST_VER_PATCH`, so ninja had nothing to rebuild:
M4 and M5 loaded the **byte-identical `core.so`** that M1 and M3 loaded, against
a different shared object. The sha256 of the staged binary is pinned in
`PROVENANCE.md` and asserted in-job, and `common.inc` additionally checks that
the linked `libcuest.so` resolves inside the campaign environment — installed is
not linked, and a stale RPATH would have made this table a comparison of nothing.

The CPU arms are the control for the control: they never enter cuEST, so any
movement in a CPU column here is host or noise, not the library.

`cuest-delta.md` is generated **without** `--require-identical-numerics`. The
other delta table asserts the two builds agree to the last bit; this one cannot,
because changing the GPU library is exactly the kind of change that may reorder
a reduction. It did so once: water cc-pVDZ's CPU-vs-GPU difference moves
5.843224e-08→5.843170e-08 Eh, 5e-13 Eh against a scatter of ±4.9e-13. That is
reduction order, not a different calculation.

On the merged build, 0.2.2.2 changes nothing measurable. GPU wall moves 1–4%,
in the same direction as the CPU arm, which cuEST never touches. The nanotube
device sample reads 8914→5386 MiB, but that is inside its own scatter.

The same swap on the control build is a different story:

<!-- PREMERGECUESTDELTA -->

The CPU arm is flat here too (within 1% on all but the nanotube, which reads 5%
*faster*), so this is the GPU arm alone. It is the table behind the retraction
above. The `do not agree numerically` line is benzene aug-cc-pVDZ moving in the
fourteenth significant digit, 3e-14 against a scatter that small.

## Paired timings, automatic GRAC — post-merge build

Median of three fresh-process `energy()` calls per arm, CPU/GPU order
alternating by repeat, both arms in the same gpu-h200 allocation at eight
threads, 112 GiB, `SAPT_DFT_GRAC_COMPUTE=ITERATIVE`. Speedup is median CPU wall
/ median GPU wall.

<!-- PAIRED -->

Repeat spread is tight: under 1% on the nanotube arms, and under 2% everywhere
except the two water cases.

The memory table is new in this campaign. Two things about it are easy to
misread:

- **Host and device peaks are not measured the same way.** The host figure is
  the kernel's `VmHWM`, an exact high-water mark over the timed region, reset
  through `/proc/self/clear_refs` before the `energy()` call. The device figure
  is a 0.5 s poll of per-process NVML accounting, so a spike shorter than the
  interval is missed and a device peak is a lower bound in a way the host peak
  is not. The nanotube device range (5386–8914 MiB across three identical
  repeats) is that sampling, not three different calculations.
- **The CPU and GPU host columns are not two measurements of one number.** The
  GPU arm holds one to two GiB on the host for every case, including the
  nanotube, because the DF integrals and the collocation grid live on the
  device; the CPU arm holds 27 GiB on the same case. The interesting comparison
  is down a column, not across.

## protein157: one CPU run, one GPU run

<!-- PROTEIN157 -->

**29.8× end to end, n=1 per arm.** 12494 s on eight Xeon 8562Y+ cores against
419 s on one H200. The other five completed GPU runs of the same build and
input (jobs 13429862 and 13480138) took 417.6–507.3 s. Their median with this
one is 419.6 s, so the speedup is the same 29.8× taken against it. The two slow
runs, at 493 and 507 s, are gpu-1 and gpu-2 of 13429862. The CPU side has no
scatter to quote. The two hosts read 84.2 (P3) and 84.1
(P1 job 13429862) GF/s per core at start-up.

The memory columns matter more than the speedup here. The CPU arm peaks at
142 GiB of host memory, over the 112 GiB Psi4 was told it had. The GPU arm holds
5.3 GiB on the host and a sampled ~46–54 GiB on the device across the three GPU
runs of that job. A 157-atom SAPT(DFT) fits on one H200 with room to spare. The
CPU arm overshoots its `memory 112 GiB` setting by 30 GiB and survived only
because the SLURM allocation was 192 GiB. That overshoot is a memory-accounting
finding in its own right: something on the CPU path allocates outside the
budget. This campaign does not identify what.

<!-- PROTEIN157ACC -->

The largest component difference is 8.9e-07 Eh (exchange), and the GRAC shifts
agree to 1e-8. The three GPU runs of that job agree with this CPU run to the
same 8.93e-07.

<!-- PROTEIN157ATTR -->

At this size DF J/K becomes worth accelerating: it is 48% of the saving, against
5–16% for the six smaller cases, and a perfect DF-K alone would cap the speedup
at 1.88×. XC still supplies 43%, so neither phase alone explains the 29.8×.

## What automatic GRAC costs

<!-- GRACCOST -->

Automatic GRAC remains 34–52% of the CPU wall and 45–59% of the GPU wall. It is
a larger share of the GPU arm in five of six cases, because the device removes
everything else faster than it removes GRAC. Unchanged in substance from the
2026-09-10 campaign, which is the expected result: nothing in the memory merge
touches the GRAC protocol.

| System | Basis | Fixed-shift speedup (job 13024192) | ITERATIVE speedup (job 13395711) | GRAC % of CPU wall |
|---|---|---:|---:|---:|
<!-- FIXEDVSITER -->

The first two columns come from different builds as well as different jobs, so
the caveat from the 2026-09-10 directory now applies twice over. Only the last
column is measured within one job and is unconditional.

## Where the saving comes from: XC, not DF J/K

<!-- ATTRIBUTION -->

DF J/K supplies 5–16% of the saving on the four substantial cases. Driving J/K
to zero (an infinitely fast DF-K, everything else unchanged) caps the end-to-end
speedup at 1.04–1.18×. XC supplies 66–79% of the saving. protein157 is the
exception, above.

This is a within-job decomposition, so unlike the cross-campaign speedup it is
not confounded by the build difference: it says where *this* build's time goes.

The same decomposition on the control arm is the one that says what the merge
moved, phase by phase:

<!-- ATTRIBUTIONPRE -->

Neither device phase moves. XC speedups are within 0.2× of the control's on
every substantial case (nanotube 10.4→10.5×, benzene aug-cc-pVDZ 10.1→10.4×,
benzene cc-pVDZ 3.7→3.8×, peptide 3.1→3.3×). The DF-K ratios read 5–11% lower
on M4 (28.2→26.7×, 14.0→12.6×, 8.1→7.2×, 6.2→5.9×). That tracks M4's slightly
slower CPU denominator, not a slower device kernel. XC's share of the saving
is 68–81% before the merge and 66–79% after.

## DF-K in effective TFLOPS

<!-- TFLOPS -->

Computed exactly as in the 2026-09-10 directory: the dense rectangular-DGEMM
FLOP count implied by the DF-K formulation divided by the kernel's own wall
time. `JK: JK` is Psi4's timer around the whole builder; "K kernel" is the K
contraction alone, which is what a library figure quoting DF-K throughput
measures. The GPU arm runs `CUEST_MIXED_PRECISION=False`, so these are not
comparable with NVIDIA's emulated-FP64 figures.

## Backend accuracy

<!-- ACCURACY -->

**The merge did not change the arithmetic.** `premerge-delta.md` compares every
case's CPU-vs-GPU difference between M5 and M4 against that case's own
repeat-to-repeat scatter, and none of them clears it. Reaching further back,
every case also reproduces the 2026-09-10 figure to within 1e-10 Eh: 5.843e-08,
5.051e-08, 2.341e-06, 2.077e-06, 9.28e-08 and 4.72e-08. Fifty-five commits,
including a rewrite of where the XC quadrature is evaluated, and a libcuest
bump left the CPU-vs-GPU difference where it was.

**The gate here is 1e-5 Eh**, the scale at which a SAPT interaction energy would
be reported differently, and every case passes it. The two benzene cases are the
closest: ~2e-6 Eh on their largest component, which failed the 1e-6 gate the
[2026-09-10 report](../phoenix-h200-20260910/README.md#backend-accuracy)
predeclared and is published there as a FAIL. That report keeps its own
threshold; a past campaign's gate is a record of what it committed to, not a
parameter a later campaign gets to update, so `render_phoenix_report.py` pins
1e-6 in its own constant rather than inheriting this default.

Widening a gate normally means deleting a finding, and here it would have. Those
two benzene numbers are not noise: the benzene cation is Jahn–Teller degenerate,
the two arms reproducibly converge to different broken-symmetry solutions ~1e-4
Eh apart, and the ~2e-6 Eh component difference follows from the different GRAC
shift that implies. It is a property of the ITERATIVE protocol on a degenerate
cation, not a regression and not a cuEST defect — but it is also not something a
wider gate should be allowed to silence. `iterative_accuracy.py`'s `verdict()`
used to short-circuit on “within tolerance”; it now reports the solution split
alongside the pass and names which arm found the variationally lower cation, so
the Interpretation column above still says so.

## Thread scaling of the CPU baseline

From M2 (job 13358750), 8 against 24 threads of Xeon Gold 6226 on one cpu-small
node — the correction to apply before setting any of these ratios beside a
vendor figure quoted against a wide socket:

Total `energy()`:

<!-- THREADTOTAL -->

DF J/K alone:

<!-- THREADDFK -->

A 3× wider baseline buys 1.14–2.01×. A DF-K claim and an end-to-end claim need
different corrections, and neither is the core-count ratio.

## Host speed

<!-- HOSTSPEED -->

The aggregate verdict is `mismatched`, and it is expected. The table lists every
tree, including the four that ran on nodes reading 24.4–24.6 GF/s per core and
were dropped for it, and M2 on a cpu-small node (Xeon Gold 6226), whose
difference is the point of M2. Every tree a reported number comes from reads
84.1–84.9 GF/s per core. Each paired merge reports `host speed: matched`, and
every six-case speedup is computed within a single job's tree. Two comparisons
reach across trees: the A/B deltas, and protein157's single pair. Read their
hosts' rows together with them.

## Reproduce

```bash
# From a checkout at the commit in PROVENANCE.md, with the raw trees at $RAW:
devtools/benchmarks/results/phoenix-h200-20260919-memory/regenerate.sh "$RAW"

# The reporting tests need no Psi4:
python -m pytest devtools/benchmarks -q
```
