# Which CPU is the speedup measured against?

Every GPU speedup in this directory is a ratio, and the denominator is a CPU
run. That denominator is not one number: it depends on the CPU model, the core
count, and how well the kernel scales with cores. This file states what our
denominator actually is, how far it is from a fast CPU, and how far it is from
NVIDIA's 56-core baseline. It exists because the same GPU measurement supports
three quite different speedup claims depending on which of those you pick, and
only one of them is the same-host accelerator speedup.

## The measured denominator: 8 cores of Xeon Platinum 8562Y+

Job A3 (13080182) ran both arms on one gpu-h200 node, and that node certified
its own speed before and after the campaign: 83.3 then 84.2 GF/s per core,
13.8 then 14.7 GB/s triad, 47.6 Miter/s on a serial scalar loop both times,
every core at its full 2800 MHz. Those are the solo probe's healthy figures for
this node type to within 1%. So A3's speedups are the honest same-host ratio on
a host known to have been running at speed: same memory, same filesystem, same
process, everything except `USE_CUEST`. That is the right comparison for "what
does adding cuEST to this node do."

It is *not* the right comparison for "what does an H200 do relative to a CPU
node," because gpu-h200 enforces a maximum 8:1 CPU:GPU ratio. One GPU buys eight
cores, and eight cores of a 32-core socket is what the CPU arm gets.

## The three baselines, now all measured

Earlier versions of this file carried a bracket table of bounds and estimates,
because the paired campaign's host had been degraded and no arithmetic could
recover the healthy number from it. Job A3 replaces every estimate with a
measurement. The last column is the retired job A, kept so the size of the
error is on the record:

| Case | Same-host, 8 healthy 8562Y+ cores | vs 8 Gold 6226 cores | vs 24 Gold 6226 cores | Job A, degraded host (retired) |
|---|---:|---:|---:|---:|
| benzene aug-cc-pVDZ | **5.85×** | 8.42× | 5.03× | 7.03× |
| benzene cc-pVDZ | **2.55×** | — | 2.50× | 3.05× |
| nanotube 6-31+G** | **7.56×** | 10.96× | 5.64× | 9.52× |
| peptide 6-31+G** | **2.23×** | 3.50× | 2.26× | 2.67× |
| water aug-cc-pVDZ | **1.06×** | 1.63× | 1.43× | 1.25× |
| water cc-pVDZ | **0.86×** | — | 1.15× | 0.97× |

The first column is job A3 throughout. The two middle columns take A3's GPU arm
over job C's Gold 6226 CPU arm at each width; job C did not run the cc-pVDZ
cases at 8 threads, hence the dashes.

Three things are worth reading off this table.

**The degraded host inflated every speedup, by 1.12× to 1.26×.** Job A's column
divided by A3's gives 1.20, 1.20, 1.26, 1.20, 1.18, 1.12 — a tight, systematic
band, which is what a host-speed artifact should look like. The direction was
predicted correctly and the magnitude was not knowable in advance.

**The cross-node columns are now ordered, and the ordering is the CPU model.**
Against 8 Gold 6226 cores every case reads higher than the same-host figure,
because a healthy 8562Y+ core is faster than a Gold 6226 core on this workload
— measured below at 1.44× to 1.57×. Against 24 Gold 6226 cores the numbers fall
back to roughly the same-host figures, because 24 of those cores land within
0.75× to 1.35× of 8 healthy 8562Y+ cores on these four cases. That near-equality
is a coincidence of this hardware pair, not a rule.

**Nothing here is a bound any more.** The earlier text called the same-host
column an upper bound and the cross-node column "an estimate of unknown sign."
Both readings were correct about job A and both are now superseded: the sign
confusion came entirely from job A's GPU arm being degraded too, which pushed
its cross-node column down while the slower Gold 6226 pushed it up. With both
arms healthy the two effects no longer collide and the columns simply measure
different baselines.

## How the degraded host was caught, and what it cost

This section is history now, but it is the reason the canary exists and it is
the evidence that job A had to be retired rather than corrected.

Job 13024192 (fixed shift) and job A / 13060539 (ITERATIVE) were both gpu-h200
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

The deficit was systematic: every case, both arms, every phase, and it spanned a
DGEMM-bound kernel (`JK: JK`) and grid-bound kernels (`RV`/`UV: Form V`) alike.

**The GPU device was not affected.** cuEST's own per-call kernel timings are the
same to within 1% across the two jobs — K median 2.05 vs 2.06 ms, K max 7.66 vs
7.73 ms — so this was the host, not the accelerator. Consistently, the GPU arm's
`Dimer SCF`, the most device-dominated phase, was the only one with a low ratio
(1.10–2.05×): the part cuEST does was unaffected and the host part was not.

Things checked and ruled out: NUMA placement (both jobs' masks lay wholly inside
one NUMA domain of the four — job 13024192 on node1, job A on node2, both
already packed), thread starvation (`timer.dat` user/wall matched to within 1%),
binary and workload differences (identical `core.so` hash, identical geometry
sha256, identical SCF iteration counts), and CPU oversubscription — job A's node
was in fact the *less* loaded of the two (CPULoad 1.14 vs 18.69, CPUAlloc 32 vs
56), which rules out contention for cores and points instead at clock or at
memory pressure from a co-tenant (job A's node had 2048000 of 2063000 MB
allocated). The one direct trace: `lscpu` reported the cores scaling at **68% of
max on job A's node and 100% on job 13024192's**.

An older, nominally slower CPU beat job A's allocation outright. Job C ran the
identical binary and protocol at eight threads on cpu-small (Xeon Gold 6226,
2.7 GHz Cascade Lake, which a healthy 8562Y+ should beat), and job A3 is the
healthy 8562Y+ for comparison:

| Case | job A 8T, s | job C 8T, s | job A3 8T, s | A / A3 |
|---|---:|---:|---:|---:|
| benzene aug-cc-pVDZ | 389.1 | 186.6 | 129.5 | 3.00× |
| nanotube 6-31+G** | 1078.8 | 499.9 | 344.9 | 3.13× |
| peptide 6-31+G** | 202.3 | 99.7 | 63.5 | 3.19× |
| water aug-cc-pVDZ | 22.9 | 12.1 | 7.9 | 2.89× |

The last column is the deficit measured directly, 2.89–3.19×, against the
2.97–3.47× the phase ratios above inferred from a different job pair entirely.
The inference method was sound.

The repair was measurement, not arithmetic: `common.inc` now runs `cpu_probe.py`
inside every allocation and writes `metadata/canary-<phase>-t<threads>.json`, so
each tree records the throughput of the cores it actually got. `host_speed.py`
reads those back, and `merge_case_trees.py` refuses to pool trees whose canaries
disagree or are missing.

## Job A2: the first healthy measurement, and a check on A3

Job A2 (13065746) reran `nanotube-6-31+G**` on a canary-healthy gpu-h200 node
before A3 existed, as a controlled experiment on the host alone. Same binary,
same geometry, same settings, same node shape; the only variable is the host.
That it is the same problem and not a different one is checkable from the
energies: A2's CPU repeat reproduces job A's to 1e-15 Eh (−0.001308839817 both)
and its GPU repeat to 1e-12 Eh, with identical GRAC shifts (0.09676767 /
0.04612867 Eh).

| `nanotube-6-31+G**`, 8 threads | Job A, degraded | Job A2, healthy | Job A3, healthy |
|---|---:|---:|---:|
| CPU arm, s | 1078.82 (n=1) | 352.73 (n=2) | 344.86 (n=3) |
| GPU arm, s | 113.32 (n=2) | 46.51 (n=1) | 45.63 (n=3) |
| Same-host speedup | 9.52× | 7.58× | **7.56×** |

**A2 and A3 agree to 0.3% on the speedup**, from separate jobs run a day apart
with separate canaries. The healthy figure is reproducible, which is the claim
the whole rerun was for.

They also settle the question the previous version of this file left open. The
GPU arm was degraded 2.44× in job A against A2, and 2.48× against A3 — not
insulated by the accelerator, because most of its wall time is host-side work.
That is why job A's cross-node column read 4.41× for this case when the
same-host truth is 7.56×: its numerator was charged for a slow host. It was not
a conservative estimate, it was a wrong one, and anyone who had quoted it as a
floor would have understated the result by 1.7×.

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
cation SCFs lean on bandwidth and serial work.

The campaign now says where in that range the real workload lands. Job A3 and
job C ran the identical binary and protocol at eight threads on the two node
types, so the ratio is measured rather than inferred from probes:

| Case, 8 threads | Gold 6226 (job C), s | healthy 8562Y+ (job A3), s | Ratio |
|---|---:|---:|---:|
| benzene aug-cc-pVDZ | 186.63 | 129.51 | 1.44× |
| nanotube 6-31+G** | 499.93 | 344.86 | 1.45× |
| peptide 6-31+G** | 99.68 | 63.46 | 1.57× |
| water aug-cc-pVDZ | 12.13 | 7.92 | 1.53× |

**1.44× to 1.57× end to end** — nearer the bandwidth ratio than the DGEMM one,
which is what a calculation dominated by XC grid work should do. Job A2's
nanotube figure of 1.42× falls in the same place.

The `JK: JK` timer alone goes the other way and is the sharper warning. On
nanotube it is 72.89 s on the Gold 6226 against 72.17 s on the healthy 8562Y+,
a ratio of 1.01×; on benzene aug-cc-pVDZ, 9.55 s against 9.61 s, or 0.99×. **The
two CPUs build DF-K at the same speed while differing 1.44× on the whole
calculation.** So a DF-K claim and an end-to-end claim do not even share a CPU
baseline correction on this hardware pair, let alone a value.

The probe on the gpu-h200 node ran while three of this campaign's jobs were
resident on it. That is visible in the numbers and it is small: job 13080182's
own start canary, taken on the same node under that load, reads 83.3 GF/s per
core (1% below the solo probe) and 13.77 GB/s triad (5% below). DGEMM is nearly
immune to the co-tenancy; bandwidth is mildly affected, as expected. Treat the
triad figures as slight underestimates of a quiet node.

For contrast, job A's degraded allocation was the same 8562Y+ model with
`lscpu` reporting its cores scaling at **68% of max**. Nothing about the model
was the problem.

## The canary found a second, smaller effect: serial throughput varies by allocation

Three jobs ran on the same physical node, `atl1-1-02-012-9-0`, at the same
clock, on the same probe binary:

| Job | Cores it got | DGEMM/core | Triad | Serial scalar loop |
|---|---|---:|---:|---:|
| A2 (13065746) | 0, 4, 8 … 28 | 83.5 GF/s | 13.8 GB/s | 47.6 Miter/s |
| A3 (13080182) | 0, 4, 8 … 28 | 83.3 GF/s | 13.8 GB/s | 47.7 Miter/s |
| B (13060540) | 17, 21, 25 … 45 | **84.3 GF/s** | **14.5 GB/s** | **35.0 Miter/s** |

Job B has the *best* DGEMM and the *best* bandwidth on the node and is 1.36×
slower on a dependent-chain scalar loop. The figure is not noise: B measures
34.95 at one thread and 35.01 at eight, A2 measures 47.67 and 47.63. It is
reproducible within a job and different between jobs, so it is a property of
the core set the allocation got — B's cores come from a different NUMA domain —
rather than of the node or the moment. The mechanism is not determined here.

`host_speed.py` rates the campaign `uncertified` at a worst pairwise ratio of
1.36×, and that entire spread is this one probe. That is the guard working as
intended: it is reporting a real difference between allocations that DGEMM
alone would have called identical.

It has a signed consequence for one number. protein157's GPU arm spends 60% of
its wall inside GRAC, which carries serial host-side work, and that arm ran on
job B's cores. Slower serial throughput makes that wall longer, so the
cross-node ratio of 17.19× against job 13066284 is an **underestimate**. By how
much is not known; the 1.36× applies to the serial fraction of the 60%, not to
the whole run.

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

Carrying the projection through gives the H200 speedup against a 56-core Gold
6226 socket, which is the closest thing here to NVIDIA's baseline shape. The
GPU arm is job A3's; the CPU arm is job C's 24-thread measurement pushed to 56
threads by the Amdahl fit:

| Case | vs 24T measured | vs 56T projected | vs 56T asymptote |
|---|---:|---:|---:|
| benzene aug-cc-pVDZ | 5.03× | **4.07×** | 3.34× |
| nanotube 6-31+G** | 5.64× | **4.12×** | 2.98× |
| peptide 6-31+G** | 2.26× | **1.91×** | 1.64× |
| water aug-cc-pVDZ | 1.43× | **1.38×** | 1.34× |

The middle column is the one to quote against a 56-core claim, and it is an
*upper bound on the speedup* precisely because the projection is an upper bound
on the baseline's scaling: a real 56-core run would likely be slower than the
fit says, which would raise these numbers, but the fit's optimism about the
serial fraction cuts the other way. The asymptote column is where these land if
a CPU socket could be made infinitely wide, and it is the floor: **no CPU core
count removes the H200's advantage on nanotube or benzene aug-cc-pVDZ, and none
of them is needed to remove it on water.**

Three caveats travel with this table. It is a different CPU model from NVIDIA's
Platinum 8570. It is a two-point fit with no residual. And per the `JK: JK`
comparison above, a DF-K-specific claim needs `thread-scaling-dfk.md`'s
projection instead, which is smaller.

## Three baselines, and the one thing that must not be said

For any GPU number in this directory, name which of these it is:

1. **Same-host, 8 healthy cores of Platinum 8562Y+** — job A3, first column of
   the bracket table. 0.86× to 7.56×. This is the accelerator speedup: what
   adding cuEST to this node does, with nothing else changed.
2. **Against a mainstream CPU node at the same 8-core width** — second column,
   1.63× to 10.96×. Higher than (1) because the Gold 6226 is a slower core by
   1.44–1.57× on this workload. Defensible, but it is a statement about two
   CPU models as much as about the GPU.
3. **Against 24 Gold 6226 cores** — third column, 1.15× to 5.64×. This is the
   "H200 versus a CPU node you would actually be given" number.
4. **Against a 56-core socket** — the projected column in the NVIDIA-denominator
   section, 1.38× to 4.12×, or the DF-K-specific projection in
   `thread-scaling-dfk.md` for a kernel claim. Defensible as a bound only: the
   fit is an unvalidated two-point Amdahl extrapolation.

Job A's numbers (7.03×, 9.52×, and the rest) are retired and appear here only
as the last column of the bracket table, labeled. They were measured on a host
running threefold slow and must not be quoted at all.

The indefensible statement is quoting (1) as if it were (4). A side-by-side
with NVIDIA's published multipliers does exactly that unless both baselines are
printed next to both numbers.

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

That job has been preempted and requeued once already, on 2026-09-11 at
07:58, and it came back to the same node and the same eight cores (17, 21 …
45), so its canary reads identically and the restart changes nothing about the
host. It re-ran the three GPU repeats — 489, 490, 492 s against the first
attempt's 486.24, 486.57, 488.21, a 0.6% spread that is just re-measurement —
and is now in the CPU arm. The requeue is also the argument for why this arm
may never land: it is a ~3 h single case with no interior checkpoint inside a
preemptible request that has already been interrupted once.
