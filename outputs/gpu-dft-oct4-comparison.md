# Reconciliation with the October 4 cuEST breakdown

The 2026-10-13 GPU4PySCF comparison is not an apples-to-apples timing
comparison with the October 4 SAPT(DFT) report. Both use seconds.

| Boundary or setting | October 4 Protein83 GPU route | October 13 Psi4 comparison |
|---|---|---|
| Timed workload | Complete SAPT(DFT)-D4(I) energy call | One restricted, supermolecular DFT energy calculation |
| Electronic solves | Normally nine: four GRAC, three HF, two corrected DFT | One converged DFT SCF |
| Functional | PBE0, with iterative GRAC for final monomer DFT | SVWN, PBE, M06-L, B3LYP, or ωB97X; no GRAC |
| Orbital basis | Spherical 6-31+G**, 922 dimer AOs | Spherical cc-pVDZ, 766 dimer AOs |
| SCF auxiliary basis | def2-universal-jkfit | def2-universal-jkfit |
| LibXC functional evaluation | Host, explicitly in every old report arm | CUDA LibXC |
| Host resources / GPU | Eight threads, H200, native FP64 | Eight threads, H200, native FP64 |
| Density/gradient stopping target | Psi4 D_CONVERGENCE=8 | Psi4 residual target 1e-7; PySCF uses a different norm |
| Scientific build | 1.12a1.dev716 | Historical CUDA-LibXC staged build, 1.12a1.dev660 |

## Numerical accounting of the old 140.10 seconds

The October 4 updated/GPU-SAD Protein83 representative run takes
140.097329 s, and its additive stage groups are:

| Region | Seconds |
|---|---:|
| Iterative GRAC | 90.467553 |
| Three HF solves | 10.948777 |
| Two corrected DFT solves | 35.170817 |
| HF interaction terms | 0.595619 |
| D4 | 1.408944 |
| Interaction build | 0.888909 |
| Remaining interaction terms | 0.296723 |
| Outer remainder | 0.319987 |

The two final DFT stages alone are 17.4986 s and 17.6722 s.
They are different monomer/GRAC-corrected calculations, not the new
uncorrected dimer calculation. They should not be divided by new timings
to infer a controlled implementation speedup.

The old run's inclusive XC total across its SCFs is 96.715011 s:
63.576803 s host functional evaluation plus 33.138207 s other XC work.
These are an orthogonal partition, already contained in the stage totals;
do not add them to the table above.

For comparison, new Protein83 median **calculation** times are 9.108 s
for PBE and 8.986 s for B3LYP. PBE whole-process medians including imports,
validation and teardown are 12.037 s (Psi4) and 17.544 s (GPU4PySCF).
The Python `time.perf_counter()` measurements use seconds; independent Unix
start/end timestamps agree with whole-process times. Thus a minute/second
conversion error does not explain the apparent discrepancy.

**Interpretation:** the large apparent difference is not a measured
14–16× acceleration of the same workload. Workload scope, basis,
functional, XC implementation, stopping target and scientific build all differ.
The valid new ratios compare the two new engines under their documented
matched nominal settings, not either engine against the old SAPT total.
A controlled bridge would require a separate matched functional/basis
experiment; no additional cluster work has been submitted.

## Sources

- [October 4 report](https://cos-4a16505.tail7d58ac.ts.net/docs/saptdft/cuest/breakdown-2026-10-04.html):
  “The end-to-end computational graph,” “Scope & scientific comparability,”
  and the Protein83 updated GPU-route headline.
- [Report data](https://cos-4a16505.tail7d58ac.ts.net/docs/saptdft/cuest/breakdown-data-2026-10-04.json):
  `campaigns[id=updated].cases["protein83 / 6-31+g**"].gpu`;
  its `options`, `nbf`, `groups`, `bins`, and `stages`.
- New raw campaign:
  `.pi/gpu_compare/archive/20261013-inferno/production/campaign/`
  (`config.json`, `records.json`, `summary.json`, and `manifest.json`).
- New timing implementation:
  `devtools/benchmarks/gpu_compare/compare.py`, `run_psi4`,
  `run_pyscf`, and controller worker timestamps.
