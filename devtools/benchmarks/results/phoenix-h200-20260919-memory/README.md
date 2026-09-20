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

Control: `pre-merge d91b5f8e81` → Treatment: `merged ee6161a3b6`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 5.89 → 6.07 (~) | 6.83 → 6.67 (0.98×) | 0.86× → 0.91× | 1093.02 → 1114.36 (1.02×) | 890.06 → 917.25 (1.03×) | 786.00 → 774.00 (~) |
| water | aug-cc-pvdz | 82 | 7.43 → 7.72 (1.04×) | 7.68 → 6.72 (0.87×) | 0.97× → 1.15× | 1519.73 → 1512.98 (~) | 929.02 → 948.66 (1.02×) | 780.00 → 780.00 (~) |
| benzene | cc-pvdz | 228 | 48.15 → 49.18 (1.02×) | 22.50 → 16.34 (0.73×) | 2.14× → 3.01× | 7916.48 → 7429.83 (0.94×) | 1171.89 → 1192.84 (1.02×) | 3132.00 → 3132.00 (~) |
| peptide | 6-31+g** | 250 | 62.63 → 63.69 (1.02×) | 28.77 → 23.78 (0.83×) | 2.18× → 2.68× | 7746.24 → 7304.40 (0.94×) | 1139.37 → 1163.50 (1.02×) | 3150.00 → 2802.00 (~) |
| benzene | aug-cc-pvdz | 384 | 121.65 → 123.84 (1.02×) | 26.46 → 18.88 (0.71×) | 4.60× → 6.56× | 16466.48 → 14899.74 (0.90×) | 1280.23 → 1279.42 (~) | 3734.00 → 2878.00 (0.77×) |
| nanotube | 6-31+g** | 548 | 338.99 → 327.80 (0.97×) | 52.44 → 38.54 (0.73×) | 6.46× → 8.50× | 31655.85 → 27044.15 (0.85×) | 1911.08 → 1937.77 (1.01×) | 5386.00 → 8914.00 (~) |

Every case reproduced its CPU-vs-GPU energy difference to within that case's own repeat-to-repeat scatter, so the two builds agree numerically on this suite.

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

# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 6.07 [6.06–6.34] | 6.67 [6.62–6.71] | 0.91× | 5.843e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 7.72 [7.72–7.73] | 6.72 [6.66–6.77] | 1.15× | 5.051e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 49.18 [49.09–49.19] | 16.34 [16.34–16.36] | 3.01× | 2.341e-06 | FAIL |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 123.84 [123.13–123.94] | 18.88 [18.83–19.00] | 6.56× | 2.077e-06 | FAIL |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 63.69 [63.67–63.90] | 23.78 [23.77–23.82] | 2.68× | 4.724e-08 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 3/3 | 327.80 [327.56–327.85] | 38.54 [38.53–38.70] | 8.50× | 9.285e-08 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| water | cc-pvdz | 1114 [1110–1123] | 917 [914–918] | 774 [774–774] | per-process | 0.5 | timed region |
| water | aug-cc-pvdz | 1513 [1505–1515] | 949 [948–975] | 780 [764–780] | per-process | 0.5 | timed region |
| benzene | cc-pvdz | 7430 [7423–7432] | 1193 [1188–1195] | 3132 [2800–3132] | per-process | 0.5 | timed region |
| benzene | aug-cc-pvdz | 14900 [14884–14903] | 1279 [1276–1280] | 2878 [2878–3734] | per-process | 0.5 | timed region |
| peptide | 6-31+g** | 7304 [7299–7305] | 1164 [1163–1164] | 2802 [2802–3150] | per-process | 0.5 | timed region |
| nanotube | 6-31+g** | 27044 [27042–27045] | 1938 [1937–1938] | 8914 [4484–8914] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-06 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012689512780 | -0.012689530381 | 1.760e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010830298358 | 0.010830257528 | 4.083e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 6.837e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.008087035896 | -0.008087094328 | 5.843e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011314630000 | -0.011314617647 | 1.235e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010087438972 | 0.010087477131 | 3.816e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098794 | 6.248e-13 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007718047523 | -0.007717997013 | 5.051e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002987010440 | -0.002988513768 | 1.503e-06 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.011832611246 | 0.011834951820 | 2.341e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650878 | -0.001350650883 | 1.239e-11 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.004962451274 | -0.004961614032 | 8.372e-07 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.003435953786 | -0.003437433767 | 1.480e-06 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.012199781315 | 0.012201858754 | 2.077e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162543 | -0.001451161719 | 8.258e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.005144736216 | -0.005144137932 | 5.983e-07 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015531989909 | -0.015532010918 | 2.110e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.014757057567 | 0.014757031394 | 2.617e-08 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319097 | -0.004728319082 | 2.732e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013229519563 | -0.013229566760 | 4.724e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.024720720509 | -0.024720718936 | 2.044e-09 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.057773743032 | 0.057773835745 | 9.285e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463905706 | -0.006463924086 | 1.839e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.001308839815 | -0.001308763545 | 7.636e-08 |

## Failed or incomplete measurements

```json
[]
```

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

| System | Basis | Arm | Rep. | Total wall, s | GRAC A, s | GRAC B, s | GRAC total, s | % of wall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | cpu 8T | 3 | 123.84 | 25.2 | 25.0 | 50.23 | 40.6% |
| benzene | aug-cc-pvdz | gpu 8T | 3 | 18.88 | 4.7 | 3.7 | 8.41 | 44.7% |
| benzene | cc-pvdz | cpu 8T | 3 | 49.18 | 10.2 | 9.9 | 20.15 | 41.1% |
| benzene | cc-pvdz | gpu 8T | 3 | 16.34 | 4.4 | 3.3 | 7.72 | 47.2% |
| nanotube | 6-31+g** | cpu 8T | 3 | 327.80 | 2.6 | 166.4 | 169.03 | 51.6% |
| nanotube | 6-31+g** | gpu 8T | 3 | 38.54 | 2.6 | 16.2 | 18.80 | 48.6% |
| peptide | 6-31+g** | cpu 8T | 3 | 63.69 | 17.7 | 14.8 | 32.49 | 51.0% |
| peptide | 6-31+g** | gpu 8T | 3 | 23.78 | 8.1 | 5.9 | 13.97 | 58.7% |
| water | aug-cc-pvdz | cpu 8T | 3 | 7.72 | 1.5 | 1.1 | 2.64 | 34.2% |
| water | aug-cc-pvdz | gpu 8T | 3 | 6.72 | 2.0 | 1.0 | 3.02 | 45.2% |
| water | cc-pvdz | cpu 8T | 3 | 6.07 | 1.2 | 0.9 | 2.04 | 33.6% |
| water | cc-pvdz | gpu 8T | 3 | 6.67 | 2.1 | 1.1 | 3.13 | 47.0% |

| System | Basis | GRAC phase speedup | Whole-calculation speedup | GRAC % of CPU wall | GRAC % of GPU wall |
|---|---|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 5.97× | 6.56× | 40.6% | 44.7% |
| benzene | cc-pvdz | 2.61× | 3.01× | 41.1% | 47.2% |
| nanotube | 6-31+g** | 8.99× | 8.50× | 51.6% | 48.6% |
| peptide | 6-31+g** | 2.33× | 2.68× | 51.0% | 58.7% |
| water | aug-cc-pvdz | 0.87× | 1.15× | 34.2% | 45.2% |
| water | cc-pvdz | 0.65× | 0.91× | 33.6% | 47.0% |

Medians over repeats, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers. The second table pairs arms only at equal thread counts, so its ratios are accelerator speedups rather than baseline-width effects.

Automatic GRAC remains 34–52% of the CPU wall and 45–59% of the GPU wall — a
larger share of the GPU arm in five of six cases, because the device removes
everything else faster than it removes GRAC. Unchanged in substance from the
2026-09-10 campaign, which is the expected result: nothing in the memory merge
touches the GRAC protocol.

| System | Basis | Fixed-shift speedup (job 13024192) | ITERATIVE speedup (job 13358747) | GRAC % of CPU wall |
|---|---|---:|---:|---:|
| water | cc-pvdz | 0.97× | 0.91× | 34% |
| water | aug-cc-pvdz | 1.15× | 1.15× | 34% |
| benzene | cc-pvdz | 2.90× | 3.01× | 41% |
| benzene | aug-cc-pvdz | 6.09× | 6.56× | 41% |
| peptide | 6-31+g** | 2.79× | 2.68× | 51% |
| nanotube | 6-31+g** | 7.59× | 8.50× | 52% |

The first two columns come from different builds as well as different jobs, so
the caveat from the 2026-09-10 directory now applies twice over. Only the last
column is measured within one job and is unconditional.

## Where the saving comes from: XC, not DF J/K

| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 123.8 | 18.9 | 6.56× | 13.6× | 10.3× | 6% | 78% | 1.06× |
| benzene-cc-pvdz | 49.2 | 16.3 | 3.01× | 7.8× | 3.8× | 6% | 65% | 1.05× |
| nanotube-6-31+g** | 327.8 | 38.5 | 8.50× | 27.7× | 10.6× | 17% | 66% | 1.18× |
| peptide-6-31+g** | 63.7 | 23.8 | 2.68× | 6.4× | 3.2× | 6% | 74% | 1.05× |
| water-aug-cc-pvdz | 7.7 | 6.7 | 1.15× | 0.3× | 1.7× | -12% | 119% | 1.01× |
| water-cc-pvdz | 6.1 | 6.7 | 0.91× | 0.1× | 0.9× | 25% | 23% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.

DF J/K is 6–17% of ITERATIVE SAPT(DFT) wall time on the four substantial cases.
Driving J/K to zero — an infinitely fast DF-K, everything else unchanged — caps
the end-to-end speedup at 1.00–1.18×. XC supplies 65–78% of the saving.

This is a within-job decomposition, so unlike the cross-campaign speedup it is
not confounded by the build difference: it says where *this* build's time goes.

The same decomposition on the control arm is the one that says what the merge
moved, phase by phase:

| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 121.6 | 26.5 | 4.60× | 7.6× | 7.1× | 8% | 81% | 1.08× |
| benzene-cc-pvdz | 48.2 | 22.5 | 2.14× | 5.3× | 2.6× | 11% | 69% | 1.08× |
| nanotube-6-31+g** | 339.0 | 52.4 | 6.46× | 24.6× | 7.3× | 23% | 63% | 1.26× |
| peptide-6-31+g** | 62.6 | 28.8 | 2.18× | 7.8× | 2.5× | 10% | 76% | 1.06× |
| water-aug-cc-pvdz | 7.4 | 7.7 | 0.97× | 0.2× | 1.2× | 58% | -166% | 1.01× |
| water-cc-pvdz | 5.9 | 6.8 | 0.86× | 0.1× | 0.8× | 15% | 33% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.

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

| Case | Repeats | K GFLOP | `JK: JK` wall, s | `JK: JK` TF/s | K kernel TF/s |
|---|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz-cpu | 3 | 1343.4 | 6.95 | 0.20 | — |
| benzene-aug-cc-pvdz-gpu | 3 | 1413.6 | 0.51 | 2.87 | 8.48 |
| benzene-cc-pvdz-cpu | 3 | 373.3 | 2.31 | 0.17 | — |
| benzene-cc-pvdz-gpu | 3 | 392.8 | 0.30 | 1.38 | 7.16 |
| nanotube-6-31+g**-cpu | 3 | 16582.7 | 51.02 | 0.33 | — |
| peptide-6-31+g**-cpu | 3 | 496.7 | 2.76 | 0.19 | — |
| water-aug-cc-pvdz-cpu | 3 | 2.9 | 0.04 | 0.08 | — |
| water-aug-cc-pvdz-gpu | 3 | 3.1 | 0.17 | 0.02 | 0.35 |
| water-cc-pvdz-cpu | 3 | 0.8 | 0.02 | 0.05 | — |
| water-cc-pvdz-gpu | 3 | 0.8 | 0.17 | 0.01 | 0.10 |

Not modeled:

- `nanotube-6-31+g**-gpu-1`: ValueError: Invalid pattern: '**' can only be an entire path component
- `nanotube-6-31+g**-gpu-2`: ValueError: Invalid pattern: '**' can only be an entire path component
- `nanotube-6-31+g**-gpu-3`: ValueError: Invalid pattern: '**' can only be an entire path component
- `peptide-6-31+g**-gpu-1`: ValueError: Invalid pattern: '**' can only be an entire path component
- `peptide-6-31+g**-gpu-2`: ValueError: Invalid pattern: '**' can only be an entire path component
- `peptide-6-31+g**-gpu-3`: ValueError: Invalid pattern: '**' can only be an entire path component

Medians over repeats. `JK: JK` covers J, K, host-side setup and any transfer, so its rate is the whole builder's; the K kernel column exists only for the GPU arm, where cuEST prints per-call kernel milliseconds. A dash means the quantity is not available for that arm, not zero.

Computed exactly as in the 2026-09-10 directory: the dense rectangular-DGEMM
FLOP count implied by the DF-K formulation divided by the kernel's own wall
time. `JK: JK` is Psi4's timer around the whole builder; "K kernel" is the K
contraction alone, which is what a library figure quoting DF-K throughput
measures. The GPU arm runs `CUEST_MIXED_PRECISION=False`, so these are not
comparable with NVIDIA's emulated-FP64 figures.

## Backend accuracy

| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | Max neutral Δ, Eh | Max cation Δ, Eh | Within 1e-06 Eh | Interpretation |
|---|---:|---:|---:|---:|---:|:--:|---|
| benzene-aug-cc-pvdz | 2.08e-06 | 6.0e-11 | 1.17e-04 | 3.20e-07 | 1.17e-04 | no | exceeds tolerance because the arms converged to different cation SCF solutions; gpu found the lower one |
| benzene-cc-pvdz | 2.34e-06 | 8.5e-12 | 1.19e-04 | 2.50e-07 | 1.20e-04 | no | exceeds tolerance because the arms converged to different cation SCF solutions; gpu found the lower one |
| nanotube-6-31+g** | 9.27e-08 | 6.4e-10 | 2.00e-08 | 1.49e-06 | 1.44e-06 | yes | agrees within tolerance |
| peptide-6-31+g** | 4.72e-08 | 2.6e-10 | 1.90e-07 | 4.20e-07 | 4.30e-07 | yes | agrees within tolerance |
| water-aug-cc-pvdz | 5.05e-08 | 9.6e-13 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |
| water-cc-pvdz | 5.84e-08 | 4.8e-13 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |

The neutral and cation columns are the monomer SCF energies the GRAC shift is derived from. Where both agree to near machine precision, the arms solved the same problem the same way. Where the neutral agrees but the cation does not, the arms converged to different solutions of a near-degenerate open-shell SCF, and the component difference that follows is not a measure of GPU arithmetic error.

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

Scaling of **energy() wall time**.

| System | Basis | Narrow | Wide | Narrow median, s | Wide median, s | Measured speedup | Parallel eff. | Projected 56T, s | Projected speedup | Asymptote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 8T | 24T | 187.09 | 109.66 | 1.71× | 57% | 87.5 | 2.14× | 2.64× |
| nanotube | 6-31+g** | 8T | 24T | 507.28 | 252.97 | 2.01× | 67% | 180.3 | 2.81× | 4.03× |
| peptide | 6-31+g** | 8T | 24T | 102.01 | 65.19 | 1.56× | 52% | 54.7 | 1.87× | 2.18× |
| water | aug-cc-pvdz | 8T | 24T | 12.20 | 10.67 | 1.14× | 38% | 10.2 | 1.19× | 1.23× |

Projected columns are a two-point Amdahl fit evaluated at 56 threads. The fit passes exactly through both measurements, so it has no residual and its accuracy cannot be judged from these data. It assumes the serial fraction does not grow with width, which memory bandwidth contention makes optimistic, so read the projection as an upper bound on the baseline correction and the asymptote as the ceiling no thread count can beat.

DF J/K alone:

Scaling of **JK: JK**.

| System | Basis | Narrow | Wide | Narrow median, s | Wide median, s | Measured speedup | Parallel eff. | Projected 56T, s | Projected speedup | Asymptote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 8T | 24T | 9.31 | 5.94 | 1.57× | 52% | 5.0 | 1.87× | 2.19× |
| nanotube | 6-31+g** | 8T | 24T | 73.61 | 38.70 | 1.90× | 63% | 28.7 | 2.56× | 3.46× |
| peptide | 6-31+g** | 8T | 24T | 3.81 | 2.58 | 1.48× | 49% | 2.2 | 1.71× | 1.94× |
| water | aug-cc-pvdz | 8T | 24T | 0.08 | 0.06 | 1.25× | 42% | 0.1 | 1.35× | 1.44× |

Projected columns are a two-point Amdahl fit evaluated at 56 threads. The fit passes exactly through both measurements, so it has no residual and its accuracy cannot be judged from these data. It assumes the serial fraction does not grow with width, which memory bandwidth contention makes optimistic, so read the projection as an upper bound on the baseline correction and the asymptote as the ceiling no thread count can beat.

A 3× wider baseline buys 1.14–2.01×. A DF-K claim and an end-to-end claim need
different corrections, and neither is the core-count ratio.

## Host speed

| Tree | Node | Threads | DGEMM GF/s per core | Triad GB/s | Scalar Miter/s | Live MHz |
|---|---|---:|---:|---:|---:|---:|
| M1-core6-h200-job13358747 | atl1-1-02-014-9-0.pace.gatech.edu | 8 | 84.3 | 14.4 | 47.4 | 2800.0 |
| M2-cpu24-core6-job13358750 | atl1-1-02-008-2-2.pace.gatech.edu | 8 | 76.5 | 9.3 | 24.7 | 1200.0 |
| M3-premerge-core6-h200-job13367763 | atl1-1-02-012-9-0.pace.gatech.edu | 8 | 84.2 | 14.6 | 47.3 | 2800.0 |

Verdict: **mismatched** (worst pairwise ratio 2.33×, tolerance 1.25×).

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
