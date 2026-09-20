| System | Basis | Arm | Rep. | Total wall, s | GRAC A, s | GRAC B, s | GRAC total, s | % of wall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | cpu 8T | 3 | 123.84 | 25.2 | 25.0 | 50.23 | 40.6% |
| benzene | aug-cc-pvdz | gpu 8T | 3 | 18.88 | 4.7 | 3.7 | 8.41 | 44.7% |
| benzene | cc-pvdz | cpu 8T | 3 | 49.18 | 10.2 | 9.9 | 20.15 | 41.1% |
| benzene | cc-pvdz | gpu 8T | 3 | 16.34 | 4.4 | 3.3 | 7.72 | 47.2% |
| nanotube | 6-31+g** | cpu 8T | 3 | 327.80 | 2.6 | 166.4 | 169.03 | 51.6% |
| nanotube | 6-31+g** | gpu 8T | 3 | 38.54 | 2.6 | 16.2 | 18.80 | 48.6% |
| peptide | 6-31+g** | cpu 8T | 3 | 63.69 | 17.7 | 14.8 | 32.49 | 51.0% |
| peptide | 6-31+g** | gpu 8T | 3 | 23.78 | 8.1 | 5.9 | 13.97 | 58.7% |
| water | aug-cc-pvdz | cpu 8T | 3 | 7.72 | 1.5 | 1.1 | 2.64 | 34.2% |
| water | aug-cc-pvdz | gpu 8T | 3 | 6.72 | 2.0 | 1.0 | 3.02 | 45.2% |
| water | cc-pvdz | cpu 8T | 3 | 6.07 | 1.2 | 0.9 | 2.04 | 33.6% |
| water | cc-pvdz | gpu 8T | 3 | 6.67 | 2.1 | 1.1 | 3.13 | 47.0% |

| System | Basis | GRAC phase speedup | Whole-calculation speedup | GRAC % of CPU wall | GRAC % of GPU wall |
|---|---|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 5.97× | 6.56× | 40.6% | 44.7% |
| benzene | cc-pvdz | 2.61× | 3.01× | 41.1% | 47.2% |
| nanotube | 6-31+g** | 8.99× | 8.50× | 51.6% | 48.6% |
| peptide | 6-31+g** | 2.33× | 2.68× | 51.0% | 58.7% |
| water | aug-cc-pvdz | 0.87× | 1.15× | 34.2% | 45.2% |
| water | cc-pvdz | 0.65× | 0.91× | 33.6% | 47.0% |

Medians over repeats, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase timers. The second table pairs arms only at equal thread counts, so its ratios are accelerator speedups rather than baseline-width effects.
