| System | Basis | Arm | Rep. | Total wall, s | GRAC A, s | GRAC B, s | GRAC total, s | % of wall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | cpu 8T | 3 | 123.81 | 25.7 | 25.2 | 50.88 | 41.1% |
| benzene | aug-cc-pvdz | gpu 8T | 3 | 19.26 | 4.9 | 3.7 | 8.59 | 44.6% |
| benzene | cc-pvdz | cpu 8T | 3 | 49.70 | 10.4 | 10.0 | 20.43 | 41.2% |
| benzene | cc-pvdz | gpu 8T | 3 | 16.64 | 4.5 | 3.3 | 7.78 | 46.8% |
| nanotube | 6-31+g** | cpu 8T | 3 | 322.91 | 2.9 | 164.0 | 166.90 | 51.7% |
| nanotube | 6-31+g** | gpu 8T | 3 | 39.07 | 2.9 | 16.2 | 19.04 | 48.7% |
| peptide | 6-31+g** | cpu 8T | 3 | 65.63 | 18.4 | 15.2 | 33.60 | 51.2% |
| peptide | 6-31+g** | gpu 8T | 3 | 24.24 | 8.3 | 5.9 | 14.24 | 58.5% |
| water | aug-cc-pvdz | cpu 8T | 3 | 7.96 | 1.6 | 1.2 | 2.75 | 34.5% |
| water | aug-cc-pvdz | gpu 8T | 3 | 7.00 | 2.1 | 1.0 | 3.10 | 44.5% |
| water | cc-pvdz | cpu 8T | 3 | 6.31 | 1.3 | 0.9 | 2.19 | 34.7% |
| water | cc-pvdz | gpu 8T | 3 | 6.92 | 2.1 | 1.1 | 3.20 | 46.4% |

| System | Basis | GRAC phase speedup | Whole-calculation speedup | GRAC % of CPU wall | GRAC % of GPU wall |
|---|---|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 5.93× | 6.43× | 41.1% | 44.6% |
| benzene | cc-pvdz | 2.63× | 2.99× | 41.2% | 46.8% |
| nanotube | 6-31+g** | 8.77× | 8.26× | 51.7% | 48.7% |
| peptide | 6-31+g** | 2.36× | 2.71× | 51.2% | 58.5% |
| water | aug-cc-pvdz | 0.89× | 1.14× | 34.5% | 44.5% |
| water | cc-pvdz | 0.68× | 0.91× | 34.7% | 46.4% |

Medians over repeats, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers. The second table pairs arms only at equal thread counts, so its ratios are accelerator speedups rather than baseline-width effects.
