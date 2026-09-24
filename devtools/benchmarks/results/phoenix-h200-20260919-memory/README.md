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

Control: `pre-merge d91b5f8e81` → Treatment: `merged ee6161a3b6`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 5.95 → 6.31 (~) | 6.66 → 6.92 (~) | 0.89× → 0.91× | 1141.36 → 1112.52 (0.97×) | 915.41 → 911.02 (~) | 758.00 → 774.00 (~) |
| water | aug-cc-pvdz | 82 | 7.46 → 7.96 (1.07×) | 6.73 → 7.00 (1.04×) | 1.11× → 1.14× | 1576.21 → 1506.60 (0.96×) | 959.77 → 944.31 (0.98×) | 780.00 → 792.00 (~) |
| benzene | cc-pvdz | 228 | 47.64 → 49.70 (1.04×) | 16.47 → 16.64 (1.01×) | 2.89× → 2.99× | 7964.49 → 7422.93 (0.93×) | 1191.46 → 1191.74 (~) | 3132.00 → 2800.00 (0.89×) |
| peptide | 6-31+g** | 250 | 62.11 → 65.63 (1.06×) | 24.13 → 24.24 (~) | 2.57× → 2.71× | 7795.52 → 7297.37 (0.94×) | 1163.09 → 1162.11 (~) | 3150.00 → 2802.00 (0.89×) |
| benzene | aug-cc-pvdz | 384 | 120.26 → 123.81 (1.03×) | 19.04 → 19.26 (1.01×) | 6.32× → 6.43× | 16520.96 → 14893.33 (0.90×) | 1302.24 → 1279.71 (0.98×) | 3734.00 → 3218.00 (~) |
| nanotube | 6-31+g** | 548 | 322.09 → 322.91 (1.00×) | 39.23 → 39.07 (~) | 8.21× → 8.26× | 31688.42 → 27051.21 (0.85×) | 1935.22 → 1933.54 (~) | 5386.00 → 5386.00 (~) |

Every case reproduced its CPU-vs-GPU energy difference to within that case's own repeat-to-repeat scatter, so the two builds agree numerically on this suite.

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

Control: `libcuest 0.2.1.2` → Treatment: `libcuest 0.2.2.2`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 6.07 → 6.31 (~) | 6.67 → 6.92 (~) | 0.91× → 0.91× | 1114.36 → 1112.52 (~) | 917.25 → 911.02 (0.99×) | 774.00 → 774.00 (~) |
| water | aug-cc-pvdz | 82 | 7.72 → 7.96 (1.03×) | 6.72 → 7.00 (1.04×) | 1.15× → 1.14× | 1512.98 → 1506.60 (~) | 948.66 → 944.31 (~) | 780.00 → 792.00 (~) |
| benzene | cc-pvdz | 228 | 49.18 → 49.70 (1.01×) | 16.34 → 16.64 (1.02×) | 3.01× → 2.99× | 7429.83 → 7422.93 (~) | 1192.84 → 1191.74 (~) | 3132.00 → 2800.00 (~) |
| peptide | 6-31+g** | 250 | 63.69 → 65.63 (1.03×) | 23.78 → 24.24 (1.02×) | 2.68× → 2.71× | 7304.40 → 7297.37 (~) | 1163.50 → 1162.11 (~) | 2802.00 → 2802.00 (~) |
| benzene | aug-cc-pvdz | 384 | 123.84 → 123.81 (~) | 18.88 → 19.26 (1.02×) | 6.56× → 6.43× | 14899.74 → 14893.33 (~) | 1279.42 → 1279.71 (~) | 2878.00 → 3218.00 (~) |
| nanotube | 6-31+g** | 548 | 327.80 → 322.91 (0.99×) | 38.54 → 39.07 (1.01×) | 8.50× → 8.26× | 27044.15 → 27051.21 (~) | 1937.77 → 1933.54 (1.00×) | 8914.00 → 5386.00 (~) |

**The two builds do not agree numerically.** These cases moved by more than their own repeats do, which a change to memory accounting cannot explain:

- water/cc-pvdz: 5.843224e-08 → 5.843170e-08 Eh (scatter ±4.9e-13)

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

Control: `pre-merge, libcuest 0.2.1.2 (M3)` → Treatment: `pre-merge, libcuest 0.2.2.2 (M5)`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 5.89 → 5.95 (~) | 6.83 → 6.66 (~) | 0.86× → 0.89× | 1093.02 → 1141.36 (1.04×) | 890.06 → 915.41 (1.03×) | 786.00 → 758.00 (~) |
| water | aug-cc-pvdz | 82 | 7.43 → 7.46 (~) | 7.68 → 6.73 (0.88×) | 0.97× → 1.11× | 1519.73 → 1576.21 (1.04×) | 929.02 → 959.77 (1.03×) | 780.00 → 780.00 (~) |
| benzene | cc-pvdz | 228 | 48.15 → 47.64 (0.99×) | 22.50 → 16.47 (0.73×) | 2.14× → 2.89× | 7916.48 → 7964.49 (1.01×) | 1171.89 → 1191.46 (1.02×) | 3132.00 → 3132.00 (~) |
| peptide | 6-31+g** | 250 | 62.63 → 62.11 (0.99×) | 28.77 → 24.13 (0.84×) | 2.18× → 2.57× | 7746.24 → 7795.52 (1.01×) | 1139.37 → 1163.09 (1.02×) | 3150.00 → 3150.00 (~) |
| benzene | aug-cc-pvdz | 384 | 121.65 → 120.26 (0.99×) | 26.46 → 19.04 (0.72×) | 4.60× → 6.32× | 16466.48 → 16520.96 (1.00×) | 1280.23 → 1302.24 (1.02×) | 3734.00 → 3734.00 (~) |
| nanotube | 6-31+g** | 548 | 338.99 → 322.09 (0.95×) | 52.44 → 39.23 (0.75×) | 6.46× → 8.21× | 31655.85 → 31688.42 (~) | 1911.08 → 1935.22 (1.01×) | 5386.00 → 5386.00 (~) |

**The two builds do not agree numerically.** These cases moved by more than their own repeats do, which a change to memory accounting cannot explain:

- benzene/aug-cc-pvdz: 2.077440e-06 → 2.077440e-06 Eh (scatter ±3.4e-14)

The CPU arm is flat here too (within 1% on all but the nanotube, which reads 5%
*faster*), so this is the GPU arm alone. It is the table behind the retraction
above. The `do not agree numerically` line is benzene aug-cc-pVDZ moving in the
fourteenth significant digit, 3e-14 against a scatter that small.

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
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 6.31 [6.31–7.16] | 6.92 [6.90–7.34] | 0.91× | 5.843e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 7.96 [7.95–8.12] | 7.00 [6.86–7.11] | 1.14× | 5.051e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 49.70 [49.63–49.71] | 16.64 [16.63–16.68] | 2.99× | 2.341e-06 | PASS |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 123.81 [123.69–123.97] | 19.26 [19.06–19.28] | 6.43× | 2.077e-06 | PASS |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 65.63 [64.51–65.69] | 24.24 [24.13–24.48] | 2.71× | 4.727e-08 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 3/3 | 322.91 [322.49–323.38] | 39.07 [39.02–39.33] | 8.26× | 9.279e-08 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| water | cc-pvdz | 1113 [1101–1118] | 911 [909–917] | 774 [758–2648] | per-process | 0.5 | timed region |
| water | aug-cc-pvdz | 1507 [1499–1509] | 944 [941–944] | 792 [780–2642] | per-process | 0.5 | timed region |
| benzene | cc-pvdz | 7423 [7411–7427] | 1192 [1189–1198] | 2800 [2800–3132] | per-process | 0.5 | timed region |
| benzene | aug-cc-pvdz | 14893 [14888–14910] | 1280 [1280–1280] | 3218 [2878–3734] | per-process | 0.5 | timed region |
| peptide | 6-31+g** | 7297 [7295–7308] | 1162 [1159–1163] | 2802 [2802–3150] | per-process | 0.5 | timed region |
| nanotube | 6-31+g** | 27051 [27034–27059] | 1934 [1931–1935] | 5386 [5386–8914] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-05 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012689512780 | -0.012689530381 | 1.760e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010830298358 | 0.010830257528 | 4.083e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 1.686e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.008087035897 | -0.008087094328 | 5.843e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011314630000 | -0.011314617647 | 1.235e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010087438972 | 0.010087477131 | 3.816e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098793 | 1.215e-12 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007718047523 | -0.007717997012 | 5.051e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002987010440 | -0.002988513768 | 1.503e-06 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.011832611246 | 0.011834951820 | 2.341e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650878 | -0.001350650883 | 1.714e-11 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.004962451274 | -0.004961614032 | 8.373e-07 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.003435953786 | -0.003437433768 | 1.480e-06 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.012199781315 | 0.012201858754 | 2.077e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162542 | -0.001451161751 | 8.936e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.005144736215 | -0.005144137967 | 5.984e-07 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015531989910 | -0.015532010883 | 2.108e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.014757057567 | 0.014757031393 | 2.617e-08 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319096 | -0.004728319110 | 3.527e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013229519562 | -0.013229566735 | 4.727e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.024720720508 | -0.024720718808 | 2.468e-09 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.057773743031 | 0.057773835791 | 9.279e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463905712 | -0.006463924347 | 1.866e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.001308839817 | -0.001308763994 | 7.660e-08 |

## Failed or incomplete measurements

```json
[]
```

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

# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| protein157 | 6-31+g** | 1344 | 442 | 1786 | 1/1 | 12493.85 [12493.85–12493.85] | 419.30 [419.30–419.30] | 29.80× | 8.929e-07 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| protein157 | 6-31+g** | 145841 [145841–145841] | 5407 [5407–5407] | 46650 [46650–46650] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-05 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| protein157 / 6-31+g** | SAPT DISP ENERGY | -0.026973367165 | -0.026973367165 | 0.000e+00 |
| protein157 / 6-31+g** | SAPT ELST ENERGY | -0.007611520422 | -0.007611990745 | 4.703e-07 |
| protein157 / 6-31+g** | SAPT EXCH ENERGY | 0.021141587305 | 0.021142480181 | 8.929e-07 |
| protein157 / 6-31+g** | SAPT IND ENERGY | -0.003207670750 | -0.003207677851 | 7.102e-09 |
| protein157 / 6-31+g** | SAPT TOTAL ENERGY | -0.016650971031 | -0.016650555580 | 4.155e-07 |

## Failed or incomplete measurements

```json
[]
```

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

| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | Max neutral Δ, Eh | Max cation Δ, Eh | Within 1e-05 Eh | Interpretation |
|---|---:|---:|---:|---:|---:|:--:|---|
| protein157-6-31+g** | 8.93e-07 | 0.0e+00 | 1.00e-08 | 8.24e-06 | 9.01e-06 | yes | agrees within tolerance |

The neutral and cation columns are the monomer SCF energies the GRAC shift is derived from. Where both agree to near machine precision, the arms solved the same problem the same way. Where the neutral agrees but the cation does not, the arms converged to different solutions of a near-degenerate open-shell SCF, and the component difference that follows is not a measure of GPU arithmetic error.

The largest component difference is 8.9e-07 Eh (exchange), and the GRAC shifts
agree to 1e-8. The three GPU runs of that job agree with this CPU run to the
same 8.93e-07.

| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| protein157-6-31+g** | 12493.8 | 419.3 | 29.80× | 77.9× | 24.0× | 48% | 43% | 1.88× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.

At this size DF J/K becomes worth accelerating: it is 48% of the saving, against
5–16% for the six smaller cases, and a perfect DF-K alone would cap the speedup
at 1.88×. XC still supplies 43%, so neither phase alone explains the 29.8×.

## What automatic GRAC costs

| System | Basis | Arm | Rep. | Total wall, s | GRAC A, s | GRAC B, s | GRAC total, s | % of wall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | cpu 8T | 3 | 123.81 | 25.7 | 25.2 | 50.88 | 41.1% |
| benzene | aug-cc-pvdz | gpu 8T | 3 | 19.26 | 4.9 | 3.7 | 8.59 | 44.6% |
| benzene | cc-pvdz | cpu 8T | 3 | 49.70 | 10.4 | 10.0 | 20.43 | 41.2% |
| benzene | cc-pvdz | gpu 8T | 3 | 16.64 | 4.5 | 3.3 | 7.78 | 46.8% |
| nanotube | 6-31+g** | cpu 8T | 3 | 322.91 | 2.9 | 164.0 | 166.90 | 51.7% |
| nanotube | 6-31+g** | gpu 8T | 3 | 39.07 | 2.9 | 16.2 | 19.04 | 48.7% |
| peptide | 6-31+g** | cpu 8T | 3 | 65.63 | 18.4 | 15.2 | 33.60 | 51.2% |
| peptide | 6-31+g** | gpu 8T | 3 | 24.24 | 8.3 | 5.9 | 14.24 | 58.5% |
| water | aug-cc-pvdz | cpu 8T | 3 | 7.96 | 1.6 | 1.2 | 2.75 | 34.5% |
| water | aug-cc-pvdz | gpu 8T | 3 | 7.00 | 2.1 | 1.0 | 3.10 | 44.5% |
| water | cc-pvdz | cpu 8T | 3 | 6.31 | 1.3 | 0.9 | 2.19 | 34.7% |
| water | cc-pvdz | gpu 8T | 3 | 6.92 | 2.1 | 1.1 | 3.20 | 46.4% |

| System | Basis | GRAC phase speedup | Whole-calculation speedup | GRAC % of CPU wall | GRAC % of GPU wall |
|---|---|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 5.93× | 6.43× | 41.1% | 44.6% |
| benzene | cc-pvdz | 2.63× | 2.99× | 41.2% | 46.8% |
| nanotube | 6-31+g** | 8.77× | 8.26× | 51.7% | 48.7% |
| peptide | 6-31+g** | 2.36× | 2.71× | 51.2% | 58.5% |
| water | aug-cc-pvdz | 0.89× | 1.14× | 34.5% | 44.5% |
| water | cc-pvdz | 0.68× | 0.91× | 34.7% | 46.4% |

Medians over repeats, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers. The second table pairs arms only at equal thread counts, so its ratios are accelerator speedups rather than baseline-width effects.

Automatic GRAC remains 34–52% of the CPU wall and 45–59% of the GPU wall. It is
a larger share of the GPU arm in five of six cases, because the device removes
everything else faster than it removes GRAC. Unchanged in substance from the
2026-09-10 campaign, which is the expected result: nothing in the memory merge
touches the GRAC protocol.

| System | Basis | Fixed-shift speedup (job 13024192) | ITERATIVE speedup (job 13395711) | GRAC % of CPU wall |
|---|---|---:|---:|---:|
| water | cc-pvdz | 0.97× | 0.91× | 35% |
| water | aug-cc-pvdz | 1.15× | 1.14× | 34% |
| benzene | cc-pvdz | 2.90× | 2.99× | 41% |
| benzene | aug-cc-pvdz | 6.09× | 6.43× | 41% |
| peptide | 6-31+g** | 2.79× | 2.71× | 51% |
| nanotube | 6-31+g** | 7.59× | 8.26× | 52% |

The first two columns come from different builds as well as different jobs, so
the caveat from the 2026-09-10 directory now applies twice over. Only the last
column is measured within one job and is unconditional.

## Where the saving comes from: XC, not DF J/K

| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 123.8 | 19.3 | 6.43× | 12.6× | 10.4× | 6% | 79% | 1.05× |
| benzene-cc-pvdz | 49.7 | 16.6 | 2.99× | 7.2× | 3.8× | 6% | 66% | 1.05× |
| nanotube-6-31+g** | 322.9 | 39.1 | 8.26× | 26.7× | 10.5× | 16% | 66% | 1.18× |
| peptide-6-31+g** | 65.6 | 24.2 | 2.71× | 5.9× | 3.3× | 5% | 74% | 1.04× |
| water-aug-cc-pvdz | 8.0 | 7.0 | 1.14× | 0.3× | 1.6× | -13% | 122% | 1.01× |
| water-cc-pvdz | 6.3 | 6.9 | 0.91× | 0.1× | 0.9× | 25% | 23% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.

DF J/K supplies 5–16% of the saving on the four substantial cases. Driving J/K
to zero (an infinitely fast DF-K, everything else unchanged) caps the end-to-end
speedup at 1.04–1.18×. XC supplies 66–79% of the saving. protein157 is the
exception, above.

This is a within-job decomposition, so unlike the cross-campaign speedup it is
not confounded by the build difference: it says where *this* build's time goes.

The same decomposition on the control arm is the one that says what the merge
moved, phase by phase:

| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 120.3 | 19.0 | 6.32× | 14.0× | 10.1× | 6% | 81% | 1.06× |
| benzene-cc-pvdz | 47.6 | 16.5 | 2.89× | 8.1× | 3.7× | 6% | 69% | 1.05× |
| nanotube-6-31+g** | 322.1 | 39.2 | 8.21× | 28.2× | 10.4× | 18% | 68% | 1.19× |
| peptide-6-31+g** | 62.1 | 24.1 | 2.57× | 6.2× | 3.1× | 6% | 78% | 1.05× |
| water-aug-cc-pvdz | 7.5 | 6.7 | 1.11× | 0.2× | 1.6× | -18% | 161% | 1.01× |
| water-cc-pvdz | 6.0 | 6.7 | 0.89× | 0.1× | 0.9× | 22% | 28% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.

Neither device phase moves. XC speedups are within 0.2× of the control's on
every substantial case (nanotube 10.4→10.5×, benzene aug-cc-pVDZ 10.1→10.4×,
benzene cc-pVDZ 3.7→3.8×, peptide 3.1→3.3×). The DF-K ratios read 5–11% lower
on M4 (28.2→26.7×, 14.0→12.6×, 8.1→7.2×, 6.2→5.9×). That tracks M4's slightly
slower CPU denominator, not a slower device kernel. XC's share of the saving
is 68–81% before the merge and 66–79% after.

## DF-K in effective TFLOPS

| Case | Repeats | K GFLOP | `JK: JK` wall, s | `JK: JK` TF/s | K kernel TF/s |
|---|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz-cpu | 3 | 1343.4 | 6.41 | 0.22 | — |
| benzene-aug-cc-pvdz-gpu | 3 | 1413.6 | 0.51 | 2.90 | 8.48 |
| benzene-cc-pvdz-cpu | 3 | 373.3 | 2.14 | 0.18 | — |
| benzene-cc-pvdz-gpu | 3 | 392.8 | 0.30 | 1.37 | 7.10 |
| nanotube-6-31+g**-cpu | 3 | 16582.7 | 48.63 | 0.35 | — |
| nanotube-6-31+g**-gpu | 3 | 17421.4 | 1.82 | 9.70 | 16.00 |
| peptide-6-31+g**-cpu | 3 | 496.7 | 2.55 | 0.20 | — |
| peptide-6-31+g**-gpu | 3 | 516.9 | 0.43 | 1.24 | 6.45 |
| water-aug-cc-pvdz-cpu | 3 | 2.9 | 0.04 | 0.08 | — |
| water-aug-cc-pvdz-gpu | 3 | 3.1 | 0.17 | 0.02 | 0.33 |
| water-cc-pvdz-cpu | 3 | 0.8 | 0.02 | 0.05 | — |
| water-cc-pvdz-gpu | 3 | 0.8 | 0.17 | 0.01 | 0.09 |

Medians over repeats. `JK: JK` covers J, K, host-side setup and any transfer, so its rate is the whole builder's; the K kernel column exists only for the GPU arm, where cuEST prints per-call kernel milliseconds. A dash means the quantity is not available for that arm, not zero.

Computed exactly as in the 2026-09-10 directory: the dense rectangular-DGEMM
FLOP count implied by the DF-K formulation divided by the kernel's own wall
time. `JK: JK` is Psi4's timer around the whole builder; "K kernel" is the K
contraction alone, which is what a library figure quoting DF-K throughput
measures. The GPU arm runs `CUEST_MIXED_PRECISION=False`, so these are not
comparable with NVIDIA's emulated-FP64 figures.

## Backend accuracy

| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | Max neutral Δ, Eh | Max cation Δ, Eh | Within 1e-05 Eh | Interpretation |
|---|---:|---:|---:|---:|---:|:--:|---|
| benzene-aug-cc-pvdz | 2.08e-06 | 1.5e-10 | 1.17e-04 | 3.20e-07 | 1.17e-04 | yes | agrees within tolerance, but the arms converged to different cation SCF solutions; gpu found the lower one |
| benzene-cc-pvdz | 2.34e-06 | 2.6e-11 | 1.19e-04 | 2.50e-07 | 1.20e-04 | yes | agrees within tolerance, but the arms converged to different cation SCF solutions; gpu found the lower one |
| nanotube-6-31+g** | 9.28e-08 | 1.6e-09 | 2.00e-08 | 1.49e-06 | 1.44e-06 | yes | agrees within tolerance |
| peptide-6-31+g** | 4.72e-08 | 3.3e-10 | 1.90e-07 | 4.20e-07 | 4.30e-07 | yes | agrees within tolerance |
| water-aug-cc-pvdz | 5.05e-08 | 1.5e-12 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |
| water-cc-pvdz | 5.84e-08 | 2.8e-13 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |

The neutral and cation columns are the monomer SCF energies the GRAC shift is derived from. Where both agree to near machine precision, the arms solved the same problem the same way. Where the neutral agrees but the cation does not, the arms converged to different solutions of a near-degenerate open-shell SCF, and the component difference that follows is not a measure of GPU arithmetic error.

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
| M4-core6-h200-cuest022-job13376151 | atl1-1-02-012-23-0.pace.gatech.edu | 8 | 24.5 | 4.2 | 13.4 | 1800.0 |
| M4-core6-h200-cuest022-job13395711 | atl1-1-02-012-2-0.pace.gatech.edu | 8 | 84.9 | 14.5 | 47.4 | 2800.0 |
| M5-premerge-core6-h200-cuest022-job13376152 | atl1-1-02-012-23-0.pace.gatech.edu | 8 | 24.5 | 4.9 | 10.0 | 1800.0 |
| M5-premerge-core6-h200-cuest022-job13395712 | atl1-1-03-019-2-0.pace.gatech.edu | 8 | 84.2 | 12.1 | 47.0 | 2800.0 |
| P1-protein157-h200-job13376153 | atl1-1-02-012-23-0.pace.gatech.edu | 8 | 24.4 | 5.1 | 13.4 | 800.0 |
| P1-protein157-h200-job13395713 | atl1-1-02-014-9-0.pace.gatech.edu | 8 | 24.6 | 5.2 | 13.5 | 2800.0 |
| P1-protein157-h200-job13429862 | atl1-1-03-020-18-0.pace.gatech.edu | 8 | 84.1 | 12.1 | 35.1 | 2800.0 |
| P1-protein157-h200-job13480138 | atl1-1-03-020-11-0.pace.gatech.edu | 8 | 84.2 | 12.0 | 47.3 | 2800.0 |
| P2-protein157-h200-job13395714 | atl1-1-02-012-2-0.pace.gatech.edu | 8 | 83.3 | 14.6 | 47.3 | 2800.0 |
| P2-protein157-h200-job13429863 | atl1-1-03-018-14-0.pace.gatech.edu | 8 | 84.2 | 12.0 | 35.1 | 2797.2 |
| P2-protein157-h200-job13480139 | atl1-1-03-020-11-0.pace.gatech.edu | 8 | 84.3 | 11.8 | 47.5 | 2800.0 |
| P3-protein157-h200-job13395715 | atl1-1-03-019-2-0.pace.gatech.edu | 8 | 84.2 | 12.0 | 47.5 | 2800.0 |

Verdict: **mismatched** (worst pairwise ratio 4.76×, tolerance 1.25×).

Host speed changed between the start and end of the run, so this tree's own cases are not mutually comparable:

- `M4-core6-h200-cuest022-job13376151`: dgemm_gflops_per_core 1.00×, live_mhz 2.25×, scalar_miter_s 1.00×, stream_gb_s 1.21×
- `M4-core6-h200-cuest022-job13395711`: dgemm_gflops_per_core 1.00×, live_mhz 1.00×, scalar_miter_s 1.33×, stream_gb_s 1.00×
- `M5-premerge-core6-h200-cuest022-job13376152`: dgemm_gflops_per_core 1.00×, live_mhz 2.25×, scalar_miter_s 1.00×, stream_gb_s 1.00×

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
