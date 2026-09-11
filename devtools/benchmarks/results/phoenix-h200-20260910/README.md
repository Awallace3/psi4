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

   One case has since been re-measured on a certified-healthy host (job
   13065746, `nanotube-6-31+G**`): the CPU arm was degraded 3.06×, the GPU arm
   2.44×, and the same-host speedup falls from **9.52× to 7.58×**. The
   inflation is real but smaller than the CPU deficit alone suggests, because
   the GPU arm was degraded too. That is one case; the other five are still
   only bounded.
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
each case; note that only the upper bound is secure. On the one case
re-measured on a healthy host the bound was 1.26× high — and the cross-node
estimate, which looks like the conservative choice, was 1.72× **low**.

The nanotube row below is job 13060539's degraded measurement, not job
13065746's healthy one. The two are deliberately not pooled: they are repeats
of the same case on different-speed hosts, and a median across them is a number
neither machine produced.

# Phoenix cuEST GRAC timing and accuracy

Status: partial.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 17.87 [17.80–17.89] | 18.45 [18.41–18.55] | 0.97× | 5.843e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 22.89 [22.87–22.98] | 18.29 [18.16–18.32] | 1.25× | 5.051e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 154.50 [153.45–154.66] | 50.67 [50.42–51.04] | 3.05× | 2.341e-06 | FAIL |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 389.06 [388.81–389.29] | 55.32 [54.99–55.78] | 7.03× | 2.077e-06 | FAIL |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 202.34 [201.07–202.48] | 75.84 [75.78–76.21] | 2.67× | 4.754e-08 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 1/2 | 1078.82 [1078.82–1078.82] | 113.32 [113.19–113.45] | 9.52× | 9.268e-08 | PASS |

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-06 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012689512780 | -0.012689530381 | 1.760e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010830298358 | 0.010830257528 | 4.083e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 3.607e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.008087035897 | -0.008087094328 | 5.843e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011314630000 | -0.011314617647 | 1.235e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010087438972 | 0.010087477131 | 3.816e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098793 | 1.429e-12 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007718047523 | -0.007717997012 | 5.051e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002987010440 | -0.002988513767 | 1.503e-06 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.011832611246 | 0.011834951820 | 2.341e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650879 | -0.001350650899 | 2.224e-11 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.004962451275 | -0.004961614048 | 8.372e-07 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.003435953786 | -0.003437433767 | 1.480e-06 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.012199781315 | 0.012201858754 | 2.077e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162542 | -0.001451161722 | 8.433e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.005144736215 | -0.005144137936 | 5.983e-07 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015531989909 | -0.015532010979 | 2.134e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.014757057567 | 0.014757031393 | 2.617e-08 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319097 | -0.004728319113 | 1.744e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013229519563 | -0.013229566801 | 4.754e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.024720720508 | -0.024720719056 | 1.272e-09 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.057773743032 | 0.057773835831 | 9.268e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463905712 | -0.006463924514 | 1.882e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.001308839817 | -0.001308764368 | 7.513e-08 |

## Failed or incomplete measurements

```json
[
  {
    "error": "missing result.json",
    "name": "nanotube-6-31+g**-cpu-2",
    "process_wall_s": 0.0,
    "returncode": 1
  },
  {
    "error": "missing result.json",
    "name": "nanotube-6-31+g**-cpu-3",
    "process_wall_s": 0.0,
    "returncode": 1
  },
  {
    "error": "missing result.json",
    "name": "nanotube-6-31+g**-gpu-3",
    "process_wall_s": 0.0,
    "returncode": 1
  }
]
```

## What automatic GRAC costs

The GRAC SCFs are not a fixed overhead bolted onto both arms: they are
themselves DFT work — a neutral RKS and a doublet-cation UKS per monomer — and
cuEST accelerates them like any other DFT work. Their cost is measured *inside*
each job, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers, so it
needs no comparison against a separate fixed-shift run:

| System | Basis | Arm | Rep. | Total wall, s | GRAC A, s | GRAC B, s | GRAC total, s | % of wall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | cpu 8T | 3 | 389.06 | 78.5 | 76.6 | 154.76 | 39.8% |
| benzene | aug-cc-pvdz | gpu 8T | 3 | 55.32 | 14.0 | 11.6 | 25.64 | 46.4% |
| benzene | cc-pvdz | cpu 8T | 3 | 154.50 | 32.4 | 30.7 | 63.08 | 40.8% |
| benzene | cc-pvdz | gpu 8T | 3 | 50.67 | 13.4 | 11.0 | 24.37 | 48.2% |
| nanotube | 6-31+g** | cpu 8T | 1 | 1078.82 | 8.2 | 540.1 | 548.30 | 50.8% |
| nanotube | 6-31+g** | gpu 8T | 2 | 113.32 | 7.2 | 50.8 | 57.98 | 51.2% |
| peptide | 6-31+g** | cpu 8T | 3 | 202.34 | 56.8 | 46.8 | 103.70 | 51.2% |
| peptide | 6-31+g** | gpu 8T | 3 | 75.84 | 26.1 | 19.7 | 45.87 | 60.5% |
| water | aug-cc-pvdz | cpu 8T | 3 | 22.89 | 4.8 | 3.5 | 8.35 | 36.4% |
| water | aug-cc-pvdz | gpu 8T | 3 | 18.29 | 5.3 | 2.9 | 8.22 | 44.9% |
| water | cc-pvdz | cpu 8T | 3 | 17.87 | 3.8 | 2.7 | 6.51 | 36.4% |
| water | cc-pvdz | gpu 8T | 3 | 18.45 | 5.5 | 3.3 | 8.86 | 47.8% |

| System | Basis | GRAC phase speedup | Whole-calculation speedup | GRAC % of CPU wall | GRAC % of GPU wall |
|---|---|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 6.04× | 7.03× | 39.8% | 46.4% |
| benzene | cc-pvdz | 2.59× | 3.05× | 40.8% | 48.2% |
| nanotube | 6-31+g** | 9.46× | 9.52× | 50.8% | 51.2% |
| peptide | 6-31+g** | 2.26× | 2.67× | 51.2% | 60.5% |
| water | aug-cc-pvdz | 1.02× | 1.25× | 36.4% | 44.9% |
| water | cc-pvdz | 0.73× | 0.97× | 36.4% | 47.8% |

Medians over repeats, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers. The second table pairs arms only at equal thread counts, so its ratios are accelerator speedups rather than baseline-width effects.

Automatic GRAC is **36-51% of wall time on every case**, and a *larger* share of
the GPU arm than of the CPU arm in every one, because the GPU removes the rest
of the calculation faster than it removes GRAC. The GRAC phase speedup is below
the whole-calculation speedup in all six paired cases. So the fixed-shift
protocol did not merely omit a preliminary step — it omitted the part of the
calculation the device handles *least* well, which flatters the GPU.

| System | Basis | Fixed-shift speedup (job 13024192) | ITERATIVE speedup (job 13060539) | GRAC % of CPU wall |
|---|---|---:|---:|---:|
| water | cc-pvdz | 0.97× | 0.97× | 36% |
| water | aug-cc-pvdz | 1.15× | 1.25× | 36% |
| benzene | cc-pvdz | 2.90× | 3.05× | 41% |
| benzene | aug-cc-pvdz | 6.09× | 7.03× | 40% |
| peptide | 6-31+g** | 2.79× | 2.67× | 51% |
| nanotube | 6-31+g** | 7.59× | 9.52× | 51% |

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

| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 389.1 | 55.3 | 7.03× | 24.7× | 11.1× | 7% | 80% | 1.07× |
| benzene-cc-pvdz | 154.5 | 50.7 | 3.05× | 15.0× | 3.8× | 8% | 68% | 1.06× |
| nanotube-6-31+g** | 1078.8 | 113.3 | 9.52× | 72.2× | 11.3× | 21% | 64% | 1.24× |
| peptide-6-31+g** | 202.3 | 75.8 | 2.67× | 11.3× | 3.1× | 7% | 77% | 1.05× |
| water-aug-cc-pvdz | 22.9 | 18.3 | 1.25× | 0.3× | 1.8× | -7% | 98% | 1.01× |
| water-cc-pvdz | 17.9 | 18.4 | 0.97× | 0.1× | 1.0× | 66% | 22% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.

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

| Case | Repeats | K GFLOP | `JK: JK` wall, s | `JK: JK` TF/s | K kernel TF/s |
|---|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz-cpu | 3 | 1343.4 | 23.95 | 0.06 | — |
| benzene-aug-cc-pvdz-gpu | 3 | 1413.6 | 0.97 | 1.52 | 8.28 |
| benzene-cc-pvdz-cpu | 3 | 373.3 | 8.81 | 0.04 | — |
| benzene-cc-pvdz-gpu | 3 | 392.8 | 0.59 | 0.69 | 6.19 |
| nanotube-6-31+g**-cpu | 1 | 16582.7 | 207.13 | 0.08 | — |
| nanotube-6-31+g**-gpu | 2 | 17421.4 | 2.87 | 6.16 | 15.90 |
| peptide-6-31+g**-cpu | 3 | 496.7 | 9.93 | 0.05 | — |
| peptide-6-31+g**-gpu | 3 | 516.9 | 0.88 | 0.61 | 5.58 |
| water-aug-cc-pvdz-cpu | 3 | 2.9 | 0.12 | 0.03 | — |
| water-aug-cc-pvdz-gpu | 3 | 3.1 | 0.44 | 0.01 | 0.12 |
| water-cc-pvdz-cpu | 3 | 0.8 | 0.05 | 0.02 | — |
| water-cc-pvdz-gpu | 3 | 0.8 | 0.43 | 0.00 | 0.03 |

Medians over repeats. `JK: JK` covers J, K, host-side setup and any transfer, so its rate is the whole builder's; the K kernel column exists only for the GPU arm, where cuEST prints per-call kernel milliseconds. A dash means the quantity is not available for that arm, not zero.

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

| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | Max neutral Δ, Eh | Max cation Δ, Eh | Within 1e-06 Eh | Interpretation |
|---|---:|---:|---:|---:|---:|:--:|---|
| benzene-aug-cc-pvdz | 2.08e-06 | 1.2e-10 | 1.17e-04 | 3.20e-07 | 1.17e-04 | no | exceeds tolerance because the arms converged to different cation SCF solutions; gpu found the lower one |
| benzene-cc-pvdz | 2.34e-06 | 8.9e-12 | 1.19e-04 | 2.50e-07 | 1.20e-04 | no | exceeds tolerance because the arms converged to different cation SCF solutions; gpu found the lower one |
| nanotube-6-31+g** | 9.28e-08 | 6.3e-10 | 2.00e-08 | 1.49e-06 | 1.45e-06 | yes | agrees within tolerance |
| peptide-6-31+g** | 4.72e-08 | 3.7e-10 | 1.90e-07 | 4.20e-07 | 4.30e-07 | yes | agrees within tolerance |
| water-aug-cc-pvdz | 5.05e-08 | 1.9e-12 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |
| water-cc-pvdz | 5.84e-08 | 3.9e-13 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |

The neutral and cation columns are the monomer SCF energies the GRAC shift is derived from. Where both agree to near machine precision, the arms solved the same problem the same way. Where the neutral agrees but the cation does not, the arms converged to different solutions of a near-degenerate open-shell SCF, and the component difference that follows is not a measure of GPU arithmetic error.

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
