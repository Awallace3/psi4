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
- `paired/summary.md` — the paired timings and the memory tables.
- `attribution.md`, `dfk-tflops.md`, `grac-cost.md`, `accuracy.md`,
  `thread-scaling-*.md`, `host-speed.md` — generated the same way as in the
  2026-09-10 directory. `regenerate.sh` rebuilds all of them from the raw trees.

`protein157` is deliberately absent. It needs its own allocations and, on the
CPU side, an inferno approval that does not exist; it is held until the six
paired cases confirm the merged build.

## Status

Three jobs, all COMPLETED with exit code 0 on `--qos=embers`:

| Job | Arm | Build | Partition | Elapsed | Cases |
|---|---|---|---|---:|---:|
| 13358747 | M1 — paired CPU/GPU, 8 threads | merge `ee6161a3b6` | gpu-h200 | 00:37:23 | 36 |
| 13358750 | M2 — CPU-only, 8 vs 24 threads | merge `ee6161a3b6` | cpu-small | 00:41:38 | 22 |
| 13367763 | M3 — paired CPU/GPU, 8 threads | **pre-merge control `d91b5f8e81`** | gpu-h200 | 00:40:00 | 36 |

M1 verified itself in-job: `cases=36 failed=0`, `without_host_memory=none`,
`gpu_cases_without_device_memory=none`. Its host canary read 84.28 GF/s per core
at the start and 84.31 at the end, against 83.3 / 84.2 for the 2026-09-10
campaign — the same class of host, steady throughout.

M3 verified itself the same way and then verified that it is the *other* build:
`cases=36 failed=0`, `without_host_memory=none`,
`gpu_cases_without_device_memory=none`, and
`cases_with_a_ledger_reading=none (expected)`. That last line is the point of the
arm. The ledger reading comes from `psi4.core.memory_committed()`, which the
merge adds; a control binary that reported one would not be the control. Both
`merge_case_trees.py` passes print `host speed: matched`, so M1 and M3 are being
differenced across comparable hosts.

## Read this before reading any speedup number here

Two comparisons are available here and only one of them is attributable.

**The controlled A/B is M1 against M3**, and it is the next section. M3 runs
`d91b5f8e81` — the merge commit's own first parent, built from the same worktree
with the same compiler flags, measured by the byte-identical driver, on the same
partition, three days later. M1 minus M3 is the merge and nothing else.

**The comparison against the [2026-09-10 campaign](../phoenix-h200-20260910/) is
not**, even though it is the tempting one: same protocol, same partition, same
six cases, and every speedup higher (2.55→3.01×, 5.85→6.56×, 2.23→2.68×,
7.56→8.50×). Two things sit between the two directories. `e971d2957b`, the
2026-09-10 build, and `d91b5f8e81` are **55 commits apart**, and they include
both the work that moved the XC quadrature onto the device (`52431fc2c0`,
`60322c4275`, `4a6164ef71`) and changes to the GRAC protocol that the GPU arm
spends half its wall time in (`303f07ab2c`, `dc4a74cb98`). This campaign also
instruments what the earlier one did not: a `/proc/self/clear_refs` reset and a
0.5 s `nvidia-smi` poll alongside every timed region.

The control arm makes that tangle a measurement rather than a worry, and the
answer is not the simple one. On the GPU side M3 reads **slower** than the
2026-09-10 build on three of the four substantial cases — benzene cc-pVDZ 22.50 s
against 19.55, benzene aug-cc-pVDZ 26.46 against 22.15, nanotube 52.44 against
45.63 — while its CPU arms are 1–6% *faster* than them. So those 55 commits moved the
GPU arm in both directions at once, and their net is not something this directory
can decompose. Read the cross-campaign numbers as a level; read the next section
for an effect.

## The controlled A/B: what the merge actually did

<!-- PREMERGE -->

The merge is a large, one-sided win on the GPU arm and a smaller one on host
memory:

- **GPU wall time falls 17–29%** on the four substantial cases — nanotube
  52.44→38.54 s, benzene aug-cc-pVDZ 26.46→18.88 s, benzene cc-pVDZ
  22.50→16.34 s, peptide 28.77→23.78 s. Every one of those clears its scatter.
- **CPU wall time does not move**: 2–4% either way, and inside the scatter on
  the smallest case. The speedup column is therefore the GPU column, and it
  rises 2.14→3.01×, 2.18→2.68×, 4.60→6.56×, and 6.46→8.50×.
- **CPU host peak falls 6–15%** where there is enough of it to matter: the
  nanotube drops 31656→27044 MiB, 4.5 GiB off a job that had been the reason
  for the 112 GiB request, and benzene aug-cc-pVDZ 16466→14900 MiB. This is the
  ledger doing what it was written for — the monomer SCFs stop being handed
  memory the dimer JK and the collocation cache are already holding.
- **GPU host peak is flat**, 1–3% up on a 0.9–1.9 GiB base. The GPU arm was
  never the one over-committing the host.

The device column is the weakest evidence in this directory and should not be
read as a result. It is a 0.5 s sample, its scatter across three identical
repeats is ±166 to ±4767 MiB, and only one case — benzene aug-cc-pVDZ,
3734→2878 MiB — moves further than its own repeats do. In particular the
nanotube's 5386→8914 MiB is **not** a measured increase: the band on that case is
±4767 MiB, which is to say the poll never resolved it at all. Sizing a device
against these numbers needs a real allocator hook, not this.

## Paired timings, automatic GRAC — post-merge build

Median of three fresh-process `energy()` calls per arm, CPU/GPU order
alternating by repeat, both arms in the same gpu-h200 allocation at eight
threads, 112 GiB, `SAPT_DFT_GRAC_COMPUTE=ITERATIVE`. Speedup is median CPU wall
/ median GPU wall.

<!-- PAIRED -->

Repeat spread is tight: 0.1% on the nanotube arms, under 1% everywhere except
the smallest water case.

The memory table is new in this campaign. Two things about it are easy to
misread:

- **Host and device peaks are not measured the same way.** The host figure is
  the kernel's `VmHWM`, an exact high-water mark over the timed region, reset
  through `/proc/self/clear_refs` before the `energy()` call. The device figure
  is a 0.5 s poll of per-process NVML accounting, so a spike shorter than the
  interval is missed and a device peak is a lower bound in a way the host peak
  is not. The `benzene aug-cc-pVDZ` device range (2878–3734 MiB across three
  identical repeats) is that sampling, not three different calculations.
- **The CPU and GPU host columns are not two measurements of one number.** The
  GPU arm holds one to two GiB on the host for every case, including the
  nanotube, because the DF integrals and the collocation grid live on the
  device; the CPU arm holds 27 GiB on the same case. The interesting comparison
  is down a column, not across.

## What automatic GRAC costs

<!-- GRACCOST -->

Automatic GRAC remains 34–52% of the CPU wall and 45–59% of the GPU wall — a
larger share of the GPU arm in five of six cases, because the device removes
everything else faster than it removes GRAC. Unchanged in substance from the
2026-09-10 campaign, which is the expected result: nothing in the memory merge
touches the GRAC protocol.

| System | Basis | Fixed-shift speedup (job 13024192) | ITERATIVE speedup (job 13358747) | GRAC % of CPU wall |
|---|---|---:|---:|---:|
<!-- FIXEDVSITER -->

The first two columns come from different builds as well as different jobs, so
the caveat from the 2026-09-10 directory now applies twice over. Only the last
column is measured within one job and is unconditional.

## Where the saving comes from: XC, not DF J/K

<!-- ATTRIBUTION -->

DF J/K is 6–17% of ITERATIVE SAPT(DFT) wall time on the four substantial cases.
Driving J/K to zero — an infinitely fast DF-K, everything else unchanged — caps
the end-to-end speedup at 1.00–1.18×. XC supplies 65–78% of the saving.

This is a within-job decomposition, so unlike the cross-campaign speedup it is
not confounded by the build difference: it says where *this* build's time goes.

The same decomposition on the control arm is the one that says what the merge
moved, phase by phase:

<!-- ATTRIBUTIONPRE -->

Both device phases got faster and neither dominates the change. Against the
control the GPU XC phase speeds up on all four substantial cases — nanotube
7.3→10.6×, benzene aug-cc-pVDZ 7.1→10.3×, benzene cc-pVDZ 2.6→3.8×, peptide
2.5→3.2× — and DF-K on three of them (7.6→13.6×, 5.3→7.8×, 24.6→27.7×, against
peptide 7.8→6.4×). The CPU arms move by 2–4%, so each of those ratios is the
device phase getting faster rather than the baseline getting slower. The shares
barely shift: XC supplies 63–81% of the saving before the merge and 65–78%
after.

A change spread across both kernels like that is what an allocation-level change
looks like. None of the three merged commits touches the arithmetic of either
phase; they change how much memory the process believes is free and which
collocation blocks are worth caching, and every phase that allocates sees it.

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
case's CPU-vs-GPU disagreement between M3 and M1 against that case's own
repeat-to-repeat scatter, and none of them clears it: the two benzene cases are
bit-identical across the builds and the rest move in the twelfth or thirteenth
significant digit (2e-13 to 5e-11 Eh), which is the reduction order of a threaded
sum, not a different calculation. Reaching further back, every case also
reproduces the 2026-09-10 figure to the printed digits — 5.843e-08, 5.051e-08,
2.341e-06, 2.077e-06, 9.285e-08, with only peptide moving 4.767e-08 → 4.724e-08
— so fifty-five commits including a rewrite of where the XC quadrature is
evaluated left the CPU-vs-GPU disagreement where it was.

The two benzene FAILs are the same pre-existing cation-SCF solution difference
documented in the [2026-09-10 report](../phoenix-h200-20260910/README.md#backend-accuracy):
the benzene cation is Jahn–Teller degenerate, the two arms reproducibly converge
to different broken-symmetry solutions ~1e-4 Eh apart, and the ~2e-6 Eh
component difference follows from the different GRAC shift. It is a property of
the ITERATIVE protocol on a degenerate cation, not a regression and not a cuEST
defect.

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

The aggregate verdict is `mismatched`, and that is expected rather than a
problem: M1/M3 run on a gpu-h200 node (Xeon Platinum 8562Y+) and M2 on a
cpu-small node (Xeon Gold 6226), which is the whole point of M2. The 2.33× that
trips the tolerance is the live-clock column — 2800 MHz against 1200 MHz, the
latter against a 2700 MHz maximum, so the cpu-small node was reading idle-clocked
at canary time. The work-rate columns are much closer (1.10× on DGEMM, 1.55× on
triad, 1.92× on the scalar loop). **Nothing pools the two trees.** Every speedup
in this directory is computed within a single job's tree, and the paired merge
reports `host speed: matched`. The M1-vs-M3 comparison in `premerge-delta.md` is
the one place two trees are differenced, and the two hosts' canaries must be
read together with it.

## Reproduce

```bash
# From a checkout at the commit in PROVENANCE.md, with the raw trees at $RAW:
devtools/benchmarks/results/phoenix-h200-20260919-memory/regenerate.sh "$RAW"

# The reporting tests need no Psi4:
python -m pytest devtools/benchmarks -q
```
