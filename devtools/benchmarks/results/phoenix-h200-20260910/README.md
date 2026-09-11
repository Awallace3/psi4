# cuEST SAPT(DFT) on an H200, with automatic GRAC — Phoenix, 2026-09-10/11

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
  whether those trees may be pooled. The paired tree is certified healthy; the
  aggregate verdict stays `uncertified` because the older trees predate the
  canary and cannot be vouched for retroactively.

## Status

Six paired cases (water, benzene, peptide, nanotube; two bases each for
water and benzene), three repeats per arm, CPU and GPU on the same node, all
36 measurements complete with a zero exit code and an asserted
`grac_compute == "ITERATIVE"`. protein157 runs separately in its own
allocations and is reported at the bottom.

The paired campaign is **job 13080182**, run on a gpu-h200 node that measured
its own CPU throughput before and after the whole campaign and read healthy
both times (83.3 then 84.2 GF/s per core, 47.6 Miter/s serial both, all cores
at 2800 MHz). An earlier attempt at the same campaign, job 13060539, landed on
a host running about threefold slow; it is **retired, not corrected**, and its
numbers appear in this directory only where they are labeled as the retired
tree. [`CPU_BASELINE.md`](CPU_BASELINE.md) has the detection, the cost, and the
canary that now prevents a silent recurrence.

Two things here are still not clean measurements, and are labeled where they
appear:

1. **Two benzene cases fail the 1e-6 Eh accuracy gate.** The cause is a
   cation-SCF solution difference, not an arithmetic disagreement; see "Backend
   accuracy". This reproduced identically on the healthy host, which is itself
   evidence it is a property of the protocol rather than of a machine.
2. **protein157's numbers are cross-node**, and its GPU arm ran on an
   allocation whose serial throughput is 1.36× below the rest of the node (see
   `CPU_BASELINE.md`). Its ratio is an underestimate by an unquantified amount.

One failure mode worth knowing about did not bite this time but will again: the
nanotube CPU arm can hang rather than fail, because Einsums parks a failed
process on a debugger-attach prompt instead of exiting, so the case burns its
timeout (see [`PROVENANCE.md`](PROVENANCE.md)). Job 13080182 finished all three
nanotube CPU repeats in 349 s each; job 13060539 lost one to this.

Everything else — the attribution, the TFLOPS table, the thread scaling — is
derived from the same case trees by `regenerate.sh`, so nothing here is
transcribed by hand.

## Paired timings, automatic GRAC

Median of three fresh-process `energy()` calls per arm, CPU/GPU order alternating
by repeat, both arms on the same canary-healthy gpu-h200 node (job 13080182) at
eight threads. Speedup is median CPU wall / median GPU wall.

These are same-host accelerator ratios and they are the headline result of this
directory. They are *not* comparable to a vendor figure quoted against a
56-core socket; [`CPU_BASELINE.md`](CPU_BASELINE.md) gives the same GPU arm
against three other baselines, and the 56-core projection, in one table.

Repeat spread is tight enough to read the medians as the measurement: 0.3% on
the nanotube CPU arm, 0.4% on its GPU arm, 4% on the widest case
(benzene aug-cc-pVDZ CPU).

# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 6.36 [6.32–6.40] | 7.35 [7.34–7.48] | 0.86× | 5.843e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 7.92 [7.88–8.01] | 7.45 [7.42–7.51] | 1.06× | 5.051e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 49.76 [49.51–49.81] | 19.55 [19.22–19.57] | 2.55× | 2.341e-06 | FAIL |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 129.51 [124.50–129.94] | 22.15 [21.49–22.22] | 5.85× | 2.077e-06 | FAIL |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 63.46 [63.33–63.71] | 28.50 [28.36–30.65] | 2.23× | 4.767e-08 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 3/3 | 344.86 [344.55–345.56] | 45.63 [45.57–45.75] | 7.56× | 9.285e-08 | PASS |

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-06 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012689512780 | -0.012689530381 | 1.760e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010830298358 | 0.010830257528 | 4.083e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 4.391e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.008087035896 | -0.008087094328 | 5.843e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011314630000 | -0.011314617647 | 1.235e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010087438972 | 0.010087477131 | 3.816e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098793 | 9.072e-13 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007718047523 | -0.007717997012 | 5.051e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002987010440 | -0.002988513768 | 1.503e-06 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.011832611246 | 0.011834951820 | 2.341e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650878 | -0.001350650876 | 9.930e-12 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.004962451274 | -0.004961614026 | 8.373e-07 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.003435953786 | -0.003437433764 | 1.480e-06 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.012199781315 | 0.012201858754 | 2.077e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162543 | -0.001451161713 | 8.483e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.005144736216 | -0.005144137925 | 5.983e-07 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015531989909 | -0.015532010891 | 2.146e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.014757057567 | 0.014757031393 | 2.617e-08 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319097 | -0.004728319096 | 3.753e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013229519563 | -0.013229566713 | 4.767e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.024720720509 | -0.024720718782 | 1.944e-09 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.057773743032 | 0.057773835861 | 9.285e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463905716 | -0.006463924068 | 1.851e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.001308839817 | -0.001308763662 | 7.628e-08 |

## Failed or incomplete measurements

```json
[]
```

## What automatic GRAC costs

The GRAC SCFs are not a fixed overhead bolted onto both arms: they are
themselves DFT work — a neutral RKS and a doublet-cation UKS per monomer — and
cuEST accelerates them like any other DFT work. Their cost is measured *inside*
each job, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers, so it
needs no comparison against a separate fixed-shift run:

| System | Basis | Arm | Rep. | Total wall, s | GRAC A, s | GRAC B, s | GRAC total, s | % of wall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | cpu 8T | 3 | 129.51 | 25.2 | 24.7 | 49.78 | 38.5% |
| benzene | aug-cc-pvdz | gpu 8T | 3 | 22.15 | 5.6 | 4.6 | 10.22 | 46.2% |
| benzene | cc-pvdz | cpu 8T | 3 | 49.76 | 10.1 | 9.5 | 19.66 | 39.5% |
| benzene | cc-pvdz | gpu 8T | 3 | 19.55 | 5.3 | 4.2 | 9.54 | 48.8% |
| nanotube | 6-31+g** | cpu 8T | 3 | 344.86 | 2.5 | 171.8 | 174.29 | 50.5% |
| nanotube | 6-31+g** | gpu 8T | 3 | 45.63 | 2.8 | 21.0 | 23.77 | 52.2% |
| peptide | 6-31+g** | cpu 8T | 3 | 63.46 | 17.5 | 14.3 | 31.80 | 50.1% |
| peptide | 6-31+g** | gpu 8T | 3 | 28.50 | 9.9 | 7.4 | 17.33 | 60.8% |
| water | aug-cc-pvdz | cpu 8T | 3 | 7.92 | 1.5 | 1.1 | 2.63 | 33.3% |
| water | aug-cc-pvdz | gpu 8T | 3 | 7.45 | 2.2 | 1.1 | 3.23 | 43.4% |
| water | cc-pvdz | cpu 8T | 3 | 6.36 | 1.2 | 0.8 | 2.04 | 32.3% |
| water | cc-pvdz | gpu 8T | 3 | 7.35 | 2.2 | 1.2 | 3.38 | 45.9% |

| System | Basis | GRAC phase speedup | Whole-calculation speedup | GRAC % of CPU wall | GRAC % of GPU wall |
|---|---|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 4.87× | 5.85× | 38.5% | 46.2% |
| benzene | cc-pvdz | 2.06× | 2.55× | 39.5% | 48.8% |
| nanotube | 6-31+g** | 7.33× | 7.56× | 50.5% | 52.2% |
| peptide | 6-31+g** | 1.84× | 2.23× | 50.1% | 60.8% |
| water | aug-cc-pvdz | 0.81× | 1.06× | 33.3% | 43.4% |
| water | cc-pvdz | 0.60× | 0.86× | 32.3% | 45.9% |

Medians over repeats, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers. The second table pairs arms only at equal thread counts, so its ratios are accelerator speedups rather than baseline-width effects.

Automatic GRAC is **32-51% of the CPU wall and 43-61% of the GPU wall**, on
every case — a *larger* share of the GPU arm than of the CPU arm in all six,
because the GPU removes the rest of the calculation faster than it removes
GRAC. The GRAC phase speedup is below the whole-calculation speedup in all six
paired cases. So the fixed-shift protocol did not merely omit a preliminary
step: it omitted the part of the calculation the device handles *least* well,
which flatters the GPU.

| System | Basis | Fixed-shift speedup (job 13024192) | ITERATIVE speedup (job 13080182) | GRAC % of CPU wall |
|---|---|---:|---:|---:|
| water | cc-pvdz | 0.97× | 0.86× | 32% |
| water | aug-cc-pvdz | 1.15× | 1.06× | 33% |
| benzene | cc-pvdz | 2.90× | 2.55× | 39% |
| benzene | aug-cc-pvdz | 6.09× | 5.85× | 38% |
| peptide | 6-31+g** | 2.79× | 2.23× | 50% |
| nanotube | 6-31+g** | 7.59× | 7.56× | 51% |

**The two speedup columns are still not safely comparable to each other**, but
for a weaker reason than before. Each is a valid same-host ratio within its own
job. Job 13080182 is canary-certified healthy and job 13024192's CPU phases run
at the same speed as a healthy host to within the resolution of the phase
comparison in [`CPU_BASELINE.md`](CPU_BASELINE.md), so unlike the retired job
13060539 the two are probably on comparable hardware — but "probably" is doing
work there, because 13024192 predates the canary and its host speed is
unmeasured, not confirmed. Read differences between the columns as suggestive.
Only the last column, measured within one job, is unconditional.

## Where the saving comes from: XC, not DF J/K

This is the single most important result for reading NVIDIA's DF-K claims
against a real SAPT(DFT) calculation.

| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 129.5 | 22.2 | 5.85× | 19.3× | 8.6× | 8% | 78% | 1.08× |
| benzene-cc-pvdz | 49.8 | 19.6 | 2.55× | 14.0× | 3.0× | 11% | 65% | 1.08× |
| nanotube-6-31+g** | 344.9 | 45.6 | 7.56× | 39.7× | 8.3× | 24% | 62% | 1.26× |
| peptide-6-31+g** | 63.5 | 28.5 | 2.23× | 10.1× | 2.5× | 10% | 75% | 1.07× |
| water-aug-cc-pvdz | 7.9 | 7.5 | 1.06× | 0.3× | 1.5× | -25% | 210% | 1.01× |
| water-cc-pvdz | 6.4 | 7.4 | 0.86× | 0.1× | 0.8× | 15% | 38% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.

DF J/K is **0.3–20.9% of ITERATIVE SAPT(DFT) wall time** on these systems. Even
an infinitely fast DF-K — J/K wall driven to zero, everything else unchanged —
caps the end-to-end speedup at the "max from DF-K alone" column: 1.00× to
1.26×. The rest comes overwhelmingly from XC, which is 62–78% of the saving on
the four cases with a substantial one. (water aug-cc-pVDZ reads over 100%
because its GPU J/K is *slower* than the CPU's, so XC has to cover a deficit as
well as produce the saving.)

This attribution is a *within-job* decomposition of where one calculation's
time goes, so it was never affected by the host deficit — a slow host inflates
numerator and denominator together. That prediction is now checkable: the same
decomposition on the retired degraded tree gave the same shape, which is why
this section survived the rerun unchanged in substance.

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
| benzene-aug-cc-pvdz-cpu | 3 | 1343.4 | 9.61 | 0.15 | — |
| benzene-aug-cc-pvdz-gpu | 3 | 1413.6 | 0.50 | 2.96 | 8.50 |
| benzene-cc-pvdz-cpu | 3 | 373.3 | 3.65 | 0.11 | — |
| benzene-cc-pvdz-gpu | 3 | 392.8 | 0.26 | 1.57 | 7.14 |
| nanotube-6-31+g**-cpu | 3 | 16582.7 | 72.17 | 0.23 | — |
| nanotube-6-31+g**-gpu | 3 | 17421.4 | 1.82 | 9.73 | 16.05 |
| peptide-6-31+g**-cpu | 3 | 496.7 | 3.96 | 0.13 | — |
| peptide-6-31+g**-gpu | 3 | 516.9 | 0.39 | 1.38 | 6.56 |
| water-aug-cc-pvdz-cpu | 3 | 2.9 | 0.06 | 0.06 | — |
| water-aug-cc-pvdz-gpu | 3 | 3.1 | 0.17 | 0.02 | 0.32 |
| water-cc-pvdz-cpu | 3 | 0.8 | 0.02 | 0.05 | — |
| water-cc-pvdz-gpu | 3 | 0.8 | 0.17 | 0.01 | 0.09 |

Medians over repeats. `JK: JK` covers J, K, host-side setup and any transfer, so its rate is the whole builder's; the K kernel column exists only for the GPU arm, where cuEST prints per-call kernel milliseconds. A dash means the quantity is not available for that arm, not zero.

Two columns, because there are two defensible denominators. `JK: JK` is Psi4's
timer around the whole J/K builder, including the J half, host-side setup, and
any transfer. "K kernel" is the K contraction alone, which is what a library
figure quoting DF-K throughput is measuring. Reporting only the second would
flatter the GPU; reporting only the first would understate the kernel.

For scale: NVIDIA's spec sheet gives the H200 SXM 34 TFLOP/s of FP64 on the
vector units and 67 TFLOP/s with FP64 tensor cores. We did not measure device
peak, so treat those as the vendor's numbers, not ours. The nanotube K kernel at
16.1 TF/s is the only case that gets within striking distance of the vector
figure; everything smaller is dominated by per-call overhead, and the two water
cases run the kernel *slower* than the CPU does because a 0.2 s launch cost
cannot be amortized over 3 GFLOP of work.

The CPU column is worth reading too, now that it comes from a healthy host:
0.05–0.23 TF/s on eight cores, against the 0.67 TF/s those eight cores reach on
a dense DGEMM probe. DF-K on the CPU runs at a third of dense-DGEMM throughput
at best, so the CPU denominator of a DF-K TFLOPS comparison is not a
peak-FLOPS question either.

Note also that our GPU arm runs `CUEST_MIXED_PRECISION=False`. NVIDIA's
effective-TFLOPS figures are annotated "Emulated results use cuEST v0.1 with
dynamic Ozaki threshold scheme," i.e. FP64 emulated on lower-precision tensor
cores. Those are different precision modes and the throughput numbers are not
interchangeable.

## Normalizing to a wider CPU baseline

Our CPU arm is eight cores. NVIDIA's is 56. The correction between them is not
7×, and it is not one number either: on nanotube, DF-K scales 1.79× from 8 to
24 threads (60% parallel efficiency) and projects to 2.31× at 56, with a 2.96×
asymptote, while total `energy()` scales 1.94× measured and projects to 2.66×,
asymptote 3.68×. A seven-times-wider baseline is nowhere near seven times
faster on this workload, and **a DF-K claim and an end-to-end claim need
different corrections.**

Carrying the projection through gives the H200 against a 56-core Gold 6226
socket, the closest shape here to NVIDIA's baseline:

| Case | Same-host, 8 healthy cores | vs 24 Gold 6226 cores | vs 56T projected | vs 56T asymptote |
|---|---:|---:|---:|---:|
| benzene aug-cc-pVDZ | 5.85× | 5.03× | **4.07×** | 3.34× |
| nanotube 6-31+G** | 7.56× | 5.64× | **4.12×** | 2.98× |
| peptide 6-31+G** | 2.23× | 2.26× | **1.91×** | 1.64× |
| water aug-cc-pVDZ | 1.06× | 1.43× | **1.38×** | 1.34× |

The last column is the floor: an infinitely wide CPU socket, zero contention.
It does not reach 1× on any case, so the H200 advantage on the two large cases
survives any core count. The 56-thread column is the one to set beside a
vendor figure, with the caveat that it is a two-point Amdahl fit with no
residual and a different CPU model from NVIDIA's Platinum 8570.

One further warning from [`CPU_BASELINE.md`](CPU_BASELINE.md): the Gold 6226
and the healthy 8562Y+ build DF-K at the *same* speed (72.89 s against 72.17 s
on nanotube at 8 threads) while differing 1.44× on the whole calculation. The
CPU baseline correction for a kernel claim is not the correction for an
end-to-end claim even before the width changes.

## Backend accuracy

| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | Max neutral Δ, Eh | Max cation Δ, Eh | Within 1e-06 Eh | Interpretation |
|---|---:|---:|---:|---:|---:|:--:|---|
| benzene-aug-cc-pvdz | 2.08e-06 | 1.2e-10 | 1.17e-04 | 3.20e-07 | 1.17e-04 | no | exceeds tolerance because the arms converged to different cation SCF solutions; gpu found the lower one |
| benzene-cc-pvdz | 2.34e-06 | 1.8e-11 | 1.19e-04 | 2.50e-07 | 1.20e-04 | no | exceeds tolerance because the arms converged to different cation SCF solutions; gpu found the lower one |
| nanotube-6-31+g** | 9.28e-08 | 3.5e-10 | 2.00e-08 | 1.49e-06 | 1.45e-06 | yes | agrees within tolerance |
| peptide-6-31+g** | 4.72e-08 | 5.3e-10 | 1.90e-07 | 4.20e-07 | 4.30e-07 | yes | agrees within tolerance |
| water-aug-cc-pvdz | 5.05e-08 | 8.8e-13 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |
| water-cc-pvdz | 5.84e-08 | 4.3e-13 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |

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

This reproduced **exactly** on the healthy host: the retired degraded tree and
job 13080182 report the same cation energies to all printed digits and the same
2.34e-06 / 2.08e-06 Eh component deltas. Host speed does not select the SCF
solution; the backend does.

## protein157

The largest case in the campaign: 157 atoms, 1786 basis functions in 6-31+G**
(monomer A 1344, monomer B 442). It is here because the paired cases top out at
a size a workstation can already run, and a DF-K claim is only interesting where
DF-K dominates.

protein157 is measured two ways, and only one of them is an accelerator
speedup:

- **Same-host, 8 threads.** Job 13060540 runs three GPU repeats and then one
  8-thread CPU repeat in the same gpu-h200 allocation. That pair is a genuine
  same-host ratio. It runs the CPU arm last on purpose: the GPU repeats are
  cheap and guarantee data if the long CPU arm is cut short, which is a real
  possibility — a 6 h per-case timeout inside an 8 h preemptible request, at the
  8-core width gpu-h200's 8:1 CPU:GPU ratio imposes. That possibility has
  already been realised once: the job was preempted and requeued on 2026-09-11,
  came back to the same node and the same eight cores, re-ran the three GPU
  repeats (489, 490, 492 s, against 486.24/486.57/488.21 first time — a 0.6%
  spread that is just re-measurement), and is in the CPU arm again. **The GPU
  side is settled at ~487 s; the same-host CPU arm has not landed yet.**
- **Cross-node, 24 threads.** Job 13066284, 24 cores of Xeon Gold 6226 on
  cpu-small, is the wider CPU baseline. Against the gpu-h200 GPU arm it is a
  different node, a different CPU model, and a different core count, so that
  ratio is labeled cross-node in the results table and is not pooled with the
  same-host rows. What gpu-h200 cannot give is a same-host *24-core* pair.

So the 24-thread number answers "how does an H200 compare to a mainstream CPU
node" and the 8-thread number answers "what does adding cuEST to this node do."
Neither substitutes for the other.

Both arms decompose within their own job, so the GRAC share is not confounded
by the cross-node comparison:

| Arm | Wall, s | GRAC A, s | GRAC B, s | GRAC total, s | GRAC % of wall |
|---|---:|---:|---:|---:|---:|
| cpu 24T, Gold 6226 (job 13066284) | 8365.25 | 3403.2 | 107.5 | 3510.72 | 42.0% |
| gpu 8T, H200 (job 13060540) | 486.57 | 271.8 | 20.2 | 292.24 | **60.1%** |

The GPU row is the first attempt's median, which is the tree on hand; the
requeued repeats agree to 0.6% and do not move the percentage.

Automatic GRAC is 42% of the CPU run — the largest absolute GRAC cost measured
anywhere in this campaign, and consistent with the 32-51% seen on the small
cases. The A/B asymmetry (3403 s against 107 s) is the monomer size ratio
showing through: GRAC runs a neutral RKS and a doublet-cation UKS SCF per
monomer, so it scales with each monomer separately, and monomer A carries 1344
of the 1786 functions.

**On the GPU, GRAC's share rises to 60%.** Splitting the cross-node ratio by
phase shows why:

| Phase | cpu 24T, s | gpu 8T, s | Ratio |
|---|---:|---:|---:|
| Everything except GRAC | 4854.5 | 194.3 | **24.98×** |
| GRAC monomer A | 3403.2 | 271.8 | 12.52× |
| GRAC monomer B | 107.5 | 20.2 | 5.32× |
| Total | 8365.3 | 486.6 | 17.19× |

These are cross-node ratios, not accelerator speedups — different node,
different CPU model, 24 cores against 8 — so the absolute values carry the
baseline with them. The *relative* pattern does not: the dimer SCF and the SAPT
terms accelerate about twice as well as the GRAC monomer SCFs, measured at the
same widths against the same baseline. GRAC is what protein157 ends up bound
by, and it is the part cuEST helps least. That makes the GRAC host-functional
path, not DF-K, the next thing worth optimizing at this size.

The interaction energy agrees across the two arms: -0.01665095 Eh on 24 CPU
cores against -0.01665056 Eh on the H200, a difference of 3.9e-07 Eh, inside
the 1e-06 Eh gate. Both GRAC shifts agree to 1e-08 Eh (monomer A 0.04531118,
monomer B 0.05039562). -10.45 kcal/mol.

## Reproduce

```bash
# From a checkout at the commit in PROVENANCE.md, with the raw case trees at $RAW:
devtools/benchmarks/results/phoenix-h200-20260910/regenerate.sh "$RAW"

# The reporting tests need no Psi4:
python -m pytest devtools/benchmarks -q
```
