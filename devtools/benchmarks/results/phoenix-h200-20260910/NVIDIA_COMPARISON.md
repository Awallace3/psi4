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
| Figures 3-4 | "DF-K performance in effective TFLOPS"; "Emulated results use cuEST v0.1 with dynamic Ozaki threshold scheme" | none (absolute rate) | Yes, against our effective DF-K TFLOPS |

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
`thread-scaling.md`. The projection is an upper bound on the correction, not a
measurement: a two-point fit has no residual, and it assumes the serial fraction
does not grow with width, which bandwidth contention makes optimistic.

## Precision

Figures 3-4 state "Emulated results use cuEST v0.1 with dynamic Ozaki threshold
scheme" and "PSI4 speedups use emulated cuEST results" — that is, FP64 emulated
from lower-precision tensor-core products, not native FP64. Our GPU arm runs
`CUEST_MIXED_PRECISION=False`. If NVIDIA's DF-K speedups rest on emulated FP64 and
ours do not, the two are not measuring the same arithmetic, and ours is the
conservative side of that difference.
