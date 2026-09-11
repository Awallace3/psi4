# What NVIDIA measured, and why our numbers are not the same quantity

Source: <https://developer.nvidia.com/cuda/cuda-x-libraries/cuest>, read 2026-09-10.
Quoted text below is from that page. Nothing here disputes NVIDIA's numbers; the
point is that three different quantities are all being called "speedup", and only
one of them is comparable to a SAPT(DFT) wall-time ratio.

## The four published quantities

| Where | What is timed | CPU baseline | Comparable to our number? |
|---|---|---|---|
| Headline | "50X speedups over traditional CPU-based quantum chemistry methods" — unqualified | not stated | No: no method, system, or baseline is attached |
| Figure 1 | "End-to-End cuEST Speedup", measured "using cuEST library calls driven by a lightweight example SCF procedure" | "state-of-the-art tensor-compressed CPU code" | No: an SCF, in a driver written for the benchmark, not a production correlated method |
| Figure 2 | DF-K speedup, "Both codes run 20 RHF iterations" | **PSI4 v1.9.1**, "56x cores of Intel Xeon Platinum 8570" | Yes, against our DF-K timer — same kernel, same reference code |
| Figures 3-4 | "DF-K performance in effective TFLOPS"; "Emulated results use cuEST v0.1 with dynamic Ozaki threshold scheme" | none (absolute rate) | Yes, against our effective DF-K TFLOPS in `dfk-tflops.md` — 16.1 TF/s at the K kernel on our largest paired case, native FP64 |

Figure 1's "end-to-end" is end-to-end *of an SCF*. Our end-to-end is a
SAPT(DFT)-D4(I) interaction energy: two monomer SCFs in the dimer basis, four more
SCFs for the ITERATIVE GRAC shifts, the exchange and dispersion machinery, and
delta-HF induction. Those share the DF J/K kernel and nothing else. A ratio taken
over the second is not the ratio Figure 1 reports, and quoting it beside the "50X"
headline would be comparing a whole method to one of its kernels.

The molecules also differ in kind. Figure 1 uses "systematic globular cutouts of
benzene crystal" — one dense, compact, growing system, which is the best case for a
dense DF-K kernel. Our set spans a water dimer through a 1786-function protein
fragment, including a nanotube whose two fragments are grossly unequal (56 and 492
own-basis functions).

## The baseline is 56 cores, and ours is not

Figure 2's baseline is Psi4 on 56 Xeon Platinum 8570 cores. This is the one
comparison where our numbers and NVIDIA's measure the same thing, so the core count
matters directly.

This association cannot allocate 56 cores on one Phoenix node: 32, 48, 56, 64, and
192 all pass `sbatch --test-only` and are then rejected at real submission with
"CPU count per node can not be satisfied". 24 is the largest shape that actually
runs. Rather than assume a linear correction, the campaign measured 8 -> 24 scaling
on one node and fits Amdahl's law to bound what 56 cores would give. See
`thread-scaling-total.md` and `thread-scaling-dfk.md`. The projection is an upper bound on the correction, not a
measurement: a two-point fit has no residual, and it assumes the serial fraction
does not grow with width, which bandwidth contention makes optimistic.

There was a second baseline problem on our side, and it has been fixed rather
than caveated. The eight cores the first paired campaign got were degraded about
threefold relative to another allocation of the same CPU model in the same
partition, which inflated every speedup in it by 1.12-1.26×. That tree is
retired. The paired numbers here come from job 13080182, whose allocation
measured its own throughput before and after the campaign and read healthy both
times. A too-slow baseline is exactly the criticism one would level at a vendor
figure, so it had to be removed from ours before any of these numbers could sit
beside one; [`CPU_BASELINE.md`](CPU_BASELINE.md) has the detection and the cost.

Normalized to NVIDIA's width, the end-to-end SAPT(DFT) ratios become:

| Case | Same-host, 8 cores | vs 24 cores measured | vs 56 cores projected |
|---|---:|---:|---:|
| benzene aug-cc-pVDZ | 5.85× | 5.03× | **4.07×** |
| nanotube 6-31+G** | 7.56× | 5.64× | **4.12×** |
| peptide 6-31+G** | 2.23× | 2.26× | **1.91×** |
| water aug-cc-pVDZ | 1.06× | 1.43× | **1.38×** |

Those are still not Figure 2's quantity — they are whole-method wall times, not
DF-K — but they are the honest form of the comparison NVIDIA's core count
invites.

One finding here cuts directly against doing that correction uniformly. The
Gold 6226 of our wider baseline and the healthy Platinum 8562Y+ of the paired
node differ by 1.44-1.57× on total SAPT(DFT) wall at the same eight threads,
but build DF-K at the *same* speed: 72.89 s against 72.17 s on nanotube (1.01×)
and 9.55 s against 9.61 s on benzene aug-cc-pVDZ (0.99×). So a DF-K claim and an
end-to-end claim do not share a CPU-baseline correction even before the core
count changes, and a single scalar cannot convert between our baseline and
NVIDIA's for both quantities at once.

## Precision

Figures 3-4 state "Emulated results use cuEST v0.1 with dynamic Ozaki threshold
scheme" and "PSI4 speedups use emulated cuEST results" — that is, FP64 emulated
from lower-precision tensor-core products, not native FP64. Our GPU arm runs
`CUEST_MIXED_PRECISION=False`. If NVIDIA's DF-K speedups rest on emulated FP64 and
ours do not, the two are not measuring the same arithmetic, and ours is the
conservative side of that difference.
