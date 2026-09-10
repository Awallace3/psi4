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

## Those eight cores are about half as fast as a mainstream Psi4 CPU node

Job C ran the identical binary, driver, geometries, and settings at eight
threads on cpu-small (Xeon Gold 6226). Call counts are bit-identical. Job A's
CPU arm burns roughly twice the CPU-seconds:

| Case | Total wall, A / C | `JK: JK` CPU-s | `RV: Form V` CPU-s | `UV: Form V` CPU-s |
|---|---:|---:|---:|---:|
| benzene aug-cc-pVDZ | 2.08× | 2.38× | 2.02× | 2.13× |
| nanotube 6-31+G** | 2.16× | 2.09× | 2.01× | 2.05× |
| peptide 6-31+G** | 2.03× | 2.49× | 2.06× | 1.90× |
| water aug-cc-pVDZ | 1.89× | 1.63× | 1.90× | 1.44× |

(Repeat 1 of each; ratios are A over C, so above one means the GPU node's host
cores are slower. Call counts match exactly in every row.)

The penalty is close to uniform across a DGEMM-bound kernel (`JK: JK`) and two
grid-bound kernels (`RV`/`UV: Form V`), which rules out an effect specific to
cache or memory bandwidth. Parallel utilization is the same in both jobs
(`timer.dat` user/wall ≈ 13.2 at eight threads in each), so it is not thread
starvation. The nominal clock favors the GPU node's CPU (2800 MHz max vs 2700),
and both parts have AVX-512.

Things checked and ruled out: NUMA placement (job A's mask `17,21,25,29,33,37,41,45`
lies entirely inside one NUMA domain on a sub-NUMA-clustered node, i.e. already
packed), binary and workload differences (same commit, same `core.so` sha256,
same call counts), and co-tenancy heavy enough to explain a factor of two
(24 of 64 CPUs allocated, CPULoad 3.16–6.24).

For calibration, a single-core microbenchmark on a cpu-small node (job 13064569,
Gold 6226) gives 75.2 GF/s DGEMM per core — about 87% of the 86.4 GF/s AVX-512
peak at 2.7 GHz — 594 GF/s across eight cores (74.3 per core, so near-linear),
24.6 Miter/s on a serial scalar loop, and 9.18 GB/s on a memory-bound triad. The
matching probe on a gpu-h200 node (job 13064568) has not run; without it the
*cause* of the deficit is uncharacterized, though its size and uniformity are
not in doubt.

**What this means for the numbers.** Job A's speedups are correct as same-host
ratios and should be read that way. A user replacing a CPU cluster node with an
H200 node would see roughly half of them, because their CPU baseline would be
about twice as fast per core as the eight cores attached to this GPU.

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

1. **Same-host, 8 cores** — job A as measured. Defensible, and labeled as such.
2. **Against a mainstream CPU node at the same width** — divide by ~2 using the
   job A / job C ratio above. Defensible with the caveat that the cause of the
   per-core deficit is not yet established.
3. **Against a 56-core socket** — divide by the projected factor in
   `thread-scaling-dfk.md` (kernel claims) or `thread-scaling-total.md`
   (end-to-end claims), on top of (2). Defensible only as a bound, since the
   projection is an unvalidated two-point fit.

The indefensible one is quoting (1) as if it were (3), which is what a
side-by-side with NVIDIA's published multipliers would do if the baselines were
not stated.

## protein157 has no same-host baseline at all

The protein157 CPU measurement runs on cpu-small (job 13066284, 24 cores of Gold
6226) and the GPU measurement on gpu-h200 (job 13060540, 8 cores of Platinum
8562Y+). **Different nodes, different CPU models, different core counts.** Any
protein157 speedup is a cross-node ratio and is labeled so in the results table.
It is not an isolated accelerator speedup and must not be pooled with the
same-host rows.
