# cuEST SAPT(DFT) on an H200, with automatic GRAC — Phoenix, 2026-09-10

Every measurement here uses `SAPT_DFT_GRAC_COMPUTE=ITERATIVE`: both monomers'
GRAC shifts are determined from neutral and doublet-cation SCFs *inside* the
timed `energy()` call. That is what a user running SAPT(DFT) on an unknown dimer
actually pays. The earlier fixed-shift rows assumed the shift was already known
and so timed a calculation nobody can run without first doing the work they
skipped; they are superseded here, not merely supplemented.

Contents:

- **This file** — what ran, the paired timings, and where the speedup comes from.
- [`CPU_BASELINE.md`](CPU_BASELINE.md) — which CPU the ratios are against, and
  how to convert them to other baselines. Read this before quoting any number.
- [`NVIDIA_COMPARISON.md`](NVIDIA_COMPARISON.md) — which of NVIDIA's published
  cuEST quantities these are and are not comparable to.
- [`PROVENANCE.md`](PROVENANCE.md) — build, hashes, jobs, settings, failure modes.
- `paired/summary.md`, `attribution.md`, `dfk-tflops.md`, `accuracy.md`,
  `thread-scaling-total.md`, `thread-scaling-dfk.md` — generated tables.
  `regenerate.sh` rebuilds all of them from the raw case trees.

## Status

Six paired cases (water, benzene, peptide, nanotube; two bases each for
water and benzene), three repeats per arm, CPU and GPU on the same node.
protein157 runs separately in its own allocations and is reported at the bottom.

Two things in this report are *not* clean measurements and are labeled where
they appear:

1. Any row the generated tables mark with fewer than 3/3 repeats, or that
   appears in the "Failed or incomplete measurements" block of
   `paired/summary.md`, is short a run. The nanotube CPU arm in particular is
   prone to it: Einsums parks a failed process on a debugger-attach prompt
   instead of exiting, so the case burns its timeout rather than failing fast
   (see [`PROVENANCE.md`](PROVENANCE.md)).
2. Two benzene cases fail the 1e-6 Eh accuracy gate. The cause is a cation-SCF
   solution difference, not an arithmetic disagreement; see "Backend accuracy".

Everything else — the attribution, the TFLOPS table, the thread scaling — is
derived from the same case trees by `regenerate.sh`, so nothing here is
transcribed by hand.

## Paired timings, automatic GRAC

Median of three fresh-process `energy()` calls per arm, CPU/GPU order alternating
by repeat, both arms on the same gpu-h200 node (job 13060539) at eight threads.
Speedup is median CPU wall / median GPU wall.

<!-- PAIRED -->

## Automatic GRAC changes the speedups, in both directions

The GRAC SCFs are not a fixed overhead added to both arms: they are themselves
DFT work, and cuEST accelerates them. So switching from a fixed shift to
ITERATIVE moves the speedup, and which way depends on how well that extra work
maps to the device.

| System | Basis | Fixed-shift speedup | ITERATIVE speedup | CPU work multiplier |
|---|---|---:|---:|---:|
<!-- FIXEDVSITER -->

The fixed-shift rows are from job 13024192 on a gpu-h200 node with the same CPU
model and the same eight-thread shape, so the comparison is like-for-like. The
"CPU work multiplier" is ITERATIVE CPU wall over fixed-shift CPU wall — how much
more work automatic GRAC actually is.

The direction is not uniform. Benzene and nanotube gain, because the extra work
is more monomer DFT and that is what the GPU is good at here. Peptide loses
slightly. Water is unchanged at or below parity: it is too small for the device
to matter, and adding more small work does not change that.

## Where the saving comes from: XC, not DF J/K

This is the single most important result for reading NVIDIA's DF-K claims
against a real SAPT(DFT) calculation.

<!-- ATTRIBUTION -->

DF J/K is **0.3–19.2% of ITERATIVE SAPT(DFT) wall time** on these systems. Even
an infinitely fast DF-K — J/K wall driven to zero, everything else unchanged —
caps the end-to-end speedup at the "max from DF-K alone" column: 1.00× to 1.24×.
The observed 2.7–9.5× comes overwhelmingly from XC, which is 62–98% of the
saving on every case where there is a saving.

A DF-K kernel speedup is therefore a claim about a minority of this workload.
That does not make it wrong; it makes it not an end-to-end claim, and the two
must not be printed in the same column.

## DF-K in effective TFLOPS

NVIDIA's Figures 3–4 report "DF-K performance in effective TFLOPS." We compute
the same quantity the same way: the dense rectangular-DGEMM FLOP count implied
by the DF-K formulation (K = 4·naux·nbf²·n_occ per call, doubled for
unrestricted) divided by the kernel's own wall time. It is *effective* because
the algorithm need not perform those FLOPs densely; it is the throughput a dense
implementation would have needed to finish in that time.

<!-- TFLOPS -->

Two columns, because there are two defensible denominators. `JK: JK` is Psi4's
timer around the whole J/K builder, including the J half, host-side setup, and
any transfer. "K kernel" is the K contraction alone, which is what a library
figure quoting DF-K throughput is measuring. Reporting only the second would
flatter the GPU; reporting only the first would understate the kernel.

For scale: NVIDIA's spec sheet gives the H200 SXM 34 TFLOP/s of FP64 on the
vector units and 67 TFLOP/s with FP64 tensor cores. We did not measure device
peak, so treat those as the vendor's numbers, not ours. The nanotube K kernel at
15.9 TF/s is the only case that gets within striking distance of the vector
figure; everything smaller is dominated by per-call overhead, and the two water
cases run the kernel *slower* than the CPU does because a 0.4 s launch cost
cannot be amortized over 3 GFLOP of work.

Note also that our GPU arm runs `CUEST_MIXED_PRECISION=False`. NVIDIA's
effective-TFLOPS figures are annotated "Emulated results use cuEST v0.1 with
dynamic Ozaki threshold scheme," i.e. FP64 emulated on lower-precision tensor
cores. Those are different precision modes and the throughput numbers are not
interchangeable.

## Normalizing to a wider CPU baseline

Our CPU arm is eight cores. NVIDIA's is 56. The correction between them is not
7× — see [`CPU_BASELINE.md`](CPU_BASELINE.md) for the measured 8→24 scaling, the
projection to 56, and the separate finding that the eight cores attached to this
GPU are themselves about half as fast per core as a mainstream Psi4 CPU node.

The short version: on nanotube, DF-K scales 1.79× from 8 to 24 threads (60%
parallel efficiency) and projects to 2.31× at 56, with a 2.96× asymptote. Total
`energy()` scales 1.94× measured and projects to 2.66×, asymptote 3.68×. A
seven-times-wider baseline is nowhere near seven times faster on this workload.

## Backend accuracy

<!-- ACCURACY -->

Two benzene cases exceed the predeclared 1e-6 Eh per-component threshold. The
cause is not arithmetic: **the two arms converged to different cation SCF
solutions.**

The benzene cation is Jahn–Teller degenerate — a D6h ring with a degenerate e1g
HOMO — so the doublet UKS problem has several broken-symmetry solutions
separated by ~1e-4 Eh, distinguished by where the hole localizes (visible in the
core 1s orbital energies). The GPU arm found the lower solution in both bases
(cc-pVDZ: CPU −231.64267427 vs GPU −231.64279399 Eh; aug-cc-pVDZ: CPU
−231.64983338 vs GPU −231.64995033 Eh). A different cation energy is a different
GRAC shift is a different monomer potential, and the ~2e-6 Eh component
difference follows.

This is a property of the ITERATIVE protocol on a degenerate cation, not a cuEST
defect, and it is not thread-dependent: CPU at 8 and 24 threads gives the
identical shift 0.07513675 Eh. Only the backend flips it. Run-to-run scatter
within an arm is 4e-13 to 6e-10 Eh, so the arms are each individually
reproducible to far below the threshold — they are reproducibly converging to
two different states.

`iterative_accuracy.py` exits zero here because both misses are explained by a
cation-SCF disagreement it can identify. An unexplained miss would fail.

## protein157

TODO-PROTEIN157

## Reproduce

```bash
# From a checkout at the commit in PROVENANCE.md, with the raw case trees at $RAW:
devtools/benchmarks/results/phoenix-h200-20260910/regenerate.sh "$RAW"

# The reporting tests need no Psi4:
python -m pytest devtools/benchmarks -q
```
