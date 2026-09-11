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
