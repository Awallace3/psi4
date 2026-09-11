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
  `grac-cost.md`, `thread-scaling-total.md`, `thread-scaling-dfk.md` —
  generated tables. `regenerate.sh` rebuilds all of them from the raw case
  trees.
- `host-speed.md` — the throughput each tree's own allocation measured, and
  whether those trees may be pooled. Every tree in this report predates the
  canary and so reads `uncertified`; that is the finding, not a gap.

## Status

Six paired cases (water, benzene, peptide, nanotube; two bases each for
water and benzene), three repeats per arm, CPU and GPU on the same node.
protein157 runs separately in its own allocations and is reported at the bottom.

Three things in this report are *not* clean measurements and are labeled where
they appear:

0. **The paired campaign's host was degraded about threefold.** Job 13060539's
   gpu-h200 allocation ran its CPU work at roughly a third the speed of another
   allocation of the same CPU model in the same partition, on the same binary
   and geometries — while the cuEST kernels were unaffected. The CPU arm is the
   denominator of every paired speedup below, so **every speedup in this report
   is inflated**, and the true same-node figure is not recoverable from this
   tree by arithmetic. Read [`CPU_BASELINE.md`](CPU_BASELINE.md) before quoting
   any of them. A canary-verified rerun is what closes this; `common.inc` now
   measures host throughput inside every allocation so it cannot recur
   silently, and the two calibration probes there now give the healthy
   throughput of both node types for comparison.
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

**These are same-host ratios on a degraded host and are upper bounds, not
results** (see item 0 above). The bracket table in
[`CPU_BASELINE.md`](CPU_BASELINE.md) gives what can and cannot be said about
each case; note that only the upper bound is secure.

<!-- PAIRED -->

## What automatic GRAC costs

The GRAC SCFs are not a fixed overhead bolted onto both arms: they are
themselves DFT work — a neutral RKS and a doublet-cation UKS per monomer — and
cuEST accelerates them like any other DFT work. Their cost is measured *inside*
each job, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers, so it
needs no comparison against a separate fixed-shift run:

<!-- GRACCOST -->

Automatic GRAC is **36-51% of wall time on every case**, and a *larger* share of
the GPU arm than of the CPU arm in every one, because the GPU removes the rest
of the calculation faster than it removes GRAC. The GRAC phase speedup is below
the whole-calculation speedup in all six paired cases. So the fixed-shift
protocol did not merely omit a preliminary step — it omitted the part of the
calculation the device handles *least* well, which flatters the GPU.

| System | Basis | Fixed-shift speedup (job 13024192) | ITERATIVE speedup (job 13060539) | GRAC % of CPU wall |
|---|---|---:|---:|---:|
<!-- FIXEDVSITER -->

**The two speedup columns are not comparable to each other.** Each is a valid
same-host ratio within its own job, but the two jobs did not run at the same
host speed: job 13060539's allocation was 3.2-3.5× slower per CPU-second than
job 13024192's, on the same CPU model, same partition, same `core.so`, and
byte-identical geometries (see [`CPU_BASELINE.md`](CPU_BASELINE.md)). Differencing
the columns mixes the protocol change with a factor-of-three hardware change, so
no statement of the form "benzene gains, peptide loses" is supportable from
them. Only the last column, measured within one job, is.

## Where the saving comes from: XC, not DF J/K

This is the single most important result for reading NVIDIA's DF-K claims
against a real SAPT(DFT) calculation.

<!-- ATTRIBUTION -->

DF J/K is **0.3–19.2% of ITERATIVE SAPT(DFT) wall time** on these systems. Even
an infinitely fast DF-K — J/K wall driven to zero, everything else unchanged —
caps the end-to-end speedup at the "max from DF-K alone" column: 1.00× to 1.24×.
The rest comes overwhelmingly from XC, which is 62–98% of the saving on every
case where there is a saving. This attribution is a *within-job* decomposition
of where one calculation's time goes, so unlike the speedups it is unaffected by
the host deficit: a slow host inflates numerator and denominator together.

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

The largest case in the campaign: 157 atoms, 1786 basis functions in 6-31+G**
(monomer A 1344, monomer B 442). It is here because the paired cases top out at
a size a workstation can already run, and a DF-K claim is only interesting where
DF-K dominates.

protein157 is measured two ways, and only one of them is an accelerator
speedup:

- **Same-host, 8 threads.** Job 13060540 (queued) runs three GPU repeats and
  then one 8-thread CPU repeat in the same gpu-h200 allocation. That pair is a
  genuine same-host ratio. It runs the CPU arm last on purpose: the GPU repeats
  are cheap and guarantee data if the long CPU arm is cut short, which is a real
  possibility — a 6 h per-case timeout inside an 8 h preemptible request, at the
  8-core width gpu-h200's 8:1 CPU:GPU ratio imposes.
- **Cross-node, 24 threads.** Job 13066284, 24 cores of Xeon Gold 6226 on
  cpu-small, is the wider CPU baseline. Against the gpu-h200 GPU arm it is a
  different node, a different CPU model, and a different core count, so that
  ratio is labeled cross-node in the results table and is not pooled with the
  same-host rows. What gpu-h200 cannot give is a same-host *24-core* pair.

So the 24-thread number answers "how does an H200 compare to a mainstream CPU
node" and the 8-thread number answers "what does adding cuEST to this node do."
Neither substitutes for the other.

What the CPU arm already establishes stands on its own, because it is a
within-job decomposition:

| Arm | Wall, s | GRAC A, s | GRAC B, s | GRAC total, s | GRAC % of wall |
|---|---:|---:|---:|---:|---:|
| cpu 24T (job 13066284) | 8365.25 | 3403.2 | 107.5 | 3510.72 | 42.0% |

Automatic GRAC is 42% of the run — the largest absolute GRAC cost measured
anywhere in this campaign, and consistent with the 36-51% seen on the small
cases. The A/B asymmetry (3403 s against 107 s) is the monomer size ratio
showing through: GRAC runs a neutral RKS and a doublet-cation UKS SCF per
monomer, so it scales with each monomer separately, and monomer A carries 1344
of the 1786 functions.

The interaction energy is -0.01665095 Eh (-10.45 kcal/mol).

## Reproduce

```bash
# From a checkout at the commit in PROVENANCE.md, with the raw case trees at $RAW:
devtools/benchmarks/results/phoenix-h200-20260910/regenerate.sh "$RAW"

# The reporting tests need no Psi4:
python -m pytest devtools/benchmarks -q
```
