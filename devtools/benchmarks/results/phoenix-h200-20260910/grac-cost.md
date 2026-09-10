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
