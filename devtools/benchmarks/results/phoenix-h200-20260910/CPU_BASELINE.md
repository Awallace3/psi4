# Which CPU is the speedup measured against?

Every GPU speedup in this directory is a ratio, and the denominator is a CPU
run. That denominator is not one number: it depends on the CPU model, the core
count, and how well the kernel scales with cores. This file states what our
denominator actually is, how far it is from a fast CPU, and how far it is from
NVIDIA's 56-core baseline. It exists because the same GPU measurement supports
three quite different speedup claims depending on which of those you pick, and
only one of them is the same-host accelerator speedup.

## The measured denominator: 8 cores of Xeon Platinum 8562Y+

Job A ran both arms on one gpu-h200 node, so its speedups are the honest
same-host ratio: same host memory, same filesystem, same process, everything
except `USE_CUEST`. That is the right comparison for "what does adding cuEST to
this node do."

It is *not* the right comparison for "what does an H200 do relative to a CPU
node," because gpu-h200 enforces a maximum 8:1 CPU:GPU ratio. One GPU buys eight
cores, and eight cores of a 32-core socket is what the CPU arm gets.

## Job A's eight cores were degraded, by a factor of about three

This is the most consequential caveat in the directory, and it was not visible
in job A's own tree.

Job 13024192 (fixed shift) and job A / 13060539 (ITERATIVE) are both gpu-h200
allocations of the **same** Xeon Platinum 8562Y+, in the same partition, running
the **same** compiled `core.so` (sha256 `bc7b9620cd41…`, verified equal in both
trees) on byte-identical geometries. The differing git hashes in the two
provenance files are repo HEAD; the commits between them touched only benchmark
scripts and markdown.

Compare phases that do provably identical work under both protocols — the dimer
and monomer SCFs, one call each, with bit-identical iteration counts:

| Phase (CPU arm, 1 call) | job 13024192 CPU-s | job A CPU-s | Ratio |
|---|---:|---:|---:|
| benzene aug `Dimer SCF` | 53.98 | 171.93 | 3.19× |
| benzene aug `Monomer A SCF` | 19.58 | 58.15 | 2.97× |
| peptide `Dimer SCF` | 20.03 | 69.48 | 3.47× |
| peptide `Monomer A SCF` | 11.43 | 34.38 | 3.01× |

The deficit is systematic: every case, both arms, every phase, and it spans a
DGEMM-bound kernel (`JK: JK`) and grid-bound kernels (`RV`/`UV: Form V`) alike.

**The GPU device is not affected.** cuEST's own per-call kernel timings are the
same to within 1% across the two jobs — K median 2.05 vs 2.06 ms, K max 7.66 vs
7.73 ms — so this is the host, not the accelerator. Consistently, the GPU arm's
`Dimer SCF`, the most device-dominated phase, is the only one with a low ratio
(1.10–2.05×): the part cuEST does was unaffected and the host part was not.

Things checked and ruled out: NUMA placement (both jobs' masks lie wholly inside
one NUMA domain of the four — job 13024192 on node1, job A on node2, both
already packed), thread starvation (`timer.dat` user/wall matches to within 1%),
binary and workload differences (identical `core.so` hash, identical geometry
sha256, identical SCF iteration counts), and CPU oversubscription — job A's node
was in fact the *less* loaded of the two (CPULoad 1.14 vs 18.69, CPUAlloc 32 vs
56), which rules out contention for cores and points instead at clock or at
memory pressure from a co-tenant (job A's node had 2048000 of 2063000 MB
allocated). The one direct trace: `lscpu` reports the cores scaling at **68% of
max on job A's node and 100% on job 13024192's**.

An older, nominally slower CPU beats job A's allocation outright. Job C ran the
identical binary and protocol at eight threads on cpu-small (Xeon Gold 6226,
2.7 GHz Cascade Lake, which a healthy 8562Y+ should beat):

| Case | job A 8T, s | job C 8T, s | A / C |
|---|---:|---:|---:|
| benzene aug-cc-pVDZ | 389.1 | 186.6 | 2.08× |
| nanotube 6-31+G** | 1078.8 | 499.9 | 2.16× |
| peptide 6-31+G** | 202.3 | 99.7 | 2.03× |
| water aug-cc-pVDZ | 22.9 | 12.1 | 1.89× |

### What this does to the headline speedups

Job A's CPU arm is the denominator of every paired speedup in this report, so
every one of them is inflated. The size of the inflation is bounded but not
pinned, because **both** arms ran on the degraded host and they are not degraded
equally — the GPU arm offloads the work the slow host would otherwise do:

| Case | Same-host (job A, degraded) | vs job C 8 healthy cores | vs job C 24 healthy cores |
|---|---:|---:|---:|
| benzene aug-cc-pVDZ | 7.03× | 3.37× | 2.02× |
| benzene cc-pVDZ | 3.05× | — | 0.97× |
| nanotube 6-31+G** | 9.52× | 4.41× | 2.27× |
| peptide 6-31+G** | 2.67× | 1.31× | 0.85× |
| water aug-cc-pVDZ | 1.25× | 0.66× | 0.58× |
| water cc-pVDZ | 0.97× | — | 0.46× |

The same-host column is **too high**, and that direction is secure: the CPU arm
is degraded roughly three times and the GPU arm only about twice, so the ratio
absorbs the difference.

The middle column's direction is *not* secure, and an earlier version of this
file got it wrong. That version called it a lower bound, reasoning that it
"gives the CPU arm a healthy one." Two effects actually push it in opposite
directions:

- Its GPU numerator is still measured on the degraded host, which charges the
  GPU arm for a slow host and pushes the ratio **down**.
- Its CPU denominator is job C's Gold 6226, which the probes below show is
  healthy but a **slower model** than a healthy 8562Y+ — by 1.13× on DGEMM per
  core, 1.58× on triad, and 1.95× on a serial scalar loop. A healthy 8562Y+ CPU
  arm would finish sooner than job C did, which pushes the true ratio **down**
  relative to this column, i.e. this column is too **high** in that respect.

Which effect dominates depends on what each case is bound by, and this data
cannot say. So the honest statement is weaker than the one this file used to
make: **the same-host column is an upper bound, and the middle column is an
estimate of unknown sign.** For benzene aug-cc-pVDZ the true figure is below
7.0× and plausibly near 3.4×, but 3.4× is not established as a floor.

This is the kind of gap that arithmetic cannot close, and job 13080182 —
running now, on a host its own canary certifies healthy — closes it by
measurement.

The repair is measurement, not arithmetic: `common.inc` now runs `cpu_probe.py`
inside every allocation and writes `metadata/canary-<phase>-t<threads>.json`, so
each tree records the throughput of the cores it actually got. `host_speed.py`
reads those back, and `merge_case_trees.py` refuses to pool trees whose canaries
disagree or are missing. A rerun of the paired campaign on a canary-verified
host is what settles the range above.

## What a healthy core of each node type actually does

Both calibration probes have now run, so the two node types can be compared
directly rather than by reputation. Eight threads, same probe, same build:

| Probe | gpu-h200, Platinum 8562Y+ (job 13064568) | cpu-small, Gold 6226 (job 13064569) | Ratio |
|---|---:|---:|---:|
| DGEMM per core | 84.15 GF/s | 74.27 GF/s | 1.13× |
| DGEMM, 8 cores | 673.2 GF/s | 594.1 GF/s | 1.13× |
| STREAM triad | 14.52 GB/s | 9.19 GB/s | 1.58× |
| Serial scalar loop | 47.91 Miter/s | 24.61 Miter/s | 1.95× |
| Live clock | 2800 of 2800 MHz max | 2700 of 2700 MHz max | — |

Both nodes were at their full rated clock, so these are the healthy figures for
each type. The Gold 6226 reaches about 86% of its 86.4 GF/s AVX-512 peak at
2.7 GHz and scales near-linearly to eight cores (74.3 per core against 75.2 on
one).

The spread across the three probes is the useful part: a healthy 8562Y+ core
beats a healthy 6226 core by only 1.13× on dense DGEMM but by 1.95× on a serial
scalar loop. **So "how much faster is the gpu-h200 node" has no single answer —
it depends on what the phase is bound by**, and SAPT(DFT) with automatic GRAC
spans both extremes: DF-K is DGEMM-bound, while the XC grid and the GRAC
cation SCFs lean on bandwidth and serial work. That range, 1.13× to 1.95×, is
why the middle column of the bracket table above cannot be signed.

The probe on the gpu-h200 node ran while three of this campaign's jobs were
resident on it. That is visible in the numbers and it is small: job 13080182's
own start canary, taken on the same node under that load, reads 83.3 GF/s per
core (1% below the solo probe) and 13.77 GB/s triad (5% below). DGEMM is nearly
immune to the co-tenancy; bandwidth is mildly affected, as expected. Treat the
triad figures as slight underestimates of a quiet node.

For contrast, job A's degraded allocation was the same 8562Y+ model with
`lscpu` reporting its cores scaling at **68% of max**. Nothing about the model
was the problem.

## NVIDIA's denominator: 56 cores of Xeon Platinum 8570

NVIDIA's Figure 2 DF-K comparison is against PSI4 v1.9.1 on 56 cores. We cannot
allocate 56 cores: 32, 48, 56, 64, and 192 all pass `sbatch --test-only` and are
then rejected at real submit with "CPU count per node can not be satisfied."
24 is the widest single-node CPU shape this association actually gets.

So the 56-core baseline is extrapolated, from a measured 8→24 thread pair on one
node (job C), with a two-point Amdahl fit. That fit passes exactly through both
points and has no residual, so it cannot be validated by its own inputs; read it
as an upper bound on the correction and the asymptote as the ceiling.

Crucially, **a DF-K claim must be normalized by DF-K's own scaling, not by the
scaling of the method around it.** Those differ here. See
`thread-scaling-dfk.md` for the `JK: JK` timer and `thread-scaling-total.md` for
total `energy()` wall time; both project to 56 threads.

| Quantity | nanotube 8→24 measured | Parallel eff. | Projected 56T speedup vs 8T | Asymptote |
|---|---:|---:|---:|---:|
| `JK: JK` | 1.79× | 60% | 2.31× | 2.96× |
| total `energy()` | 1.94× | 65% | 2.66× | 3.68× |

Neither reaches three, let alone seven. Even the asymptote — infinite cores,
zero contention — is under 4×. A CPU baseline seven times wider is nowhere near
seven times faster on this workload.

## Three defensible statements, and one indefensible one

For any GPU number in this directory:

1. **Same-host, 8 cores, on the host job A actually got** — job A as measured.
   Defensible only as a statement about that allocation, which was degraded
   about threefold. It is *not* the number a user would see on a healthy node
   of the same type, and it is the one that must never be quoted bare.
2. **Same-host, 8 cores, on a healthy host** — not yet available. (1) is an
   upper bound on it; the second column of the bracket table is an estimate
   whose sign is unknown, for the reasons given with that table. Job 13080182
   measures it directly.
3. **Against a mainstream CPU node at the same width** — the second column of
   that table. Its two errors run in opposite directions, so read it as an
   estimate, not as a bound in either direction.
4. **Against a 56-core socket** — divide by the projected factor in
   `thread-scaling-dfk.md` (kernel claims) or `thread-scaling-total.md`
   (end-to-end claims), on top of (3). Defensible only as a bound, since the
   projection is an unvalidated two-point fit.

The indefensible one is quoting (1) as if it were (4), which is what a
side-by-side with NVIDIA's published multipliers would do if the baselines were
not stated.

## protein157 has no same-host baseline at all

The 24-core CPU measurement runs on cpu-small (job 13066284, Gold 6226) and the
GPU measurement on gpu-h200 (job 13060540, Platinum 8562Y+). **Different nodes,
different CPU models, different core counts.** That ratio is cross-node, is
labeled so in the results table, and must not be pooled with the same-host rows.

A same-host protein157 pair is possible, just not at 24 cores: job 13060540 also
runs one 8-thread CPU repeat in its own allocation, after its three GPU repeats.
If that arm survives the 8 h preemptible wall — it is the longest single case in
the campaign and it runs last — protein157 gains a same-host 8-core ratio
alongside the cross-node 24-core one. Until then the row carries only the
cross-node number.
