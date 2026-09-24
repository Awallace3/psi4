| Case | Repeats | K GFLOP | `JK: JK` wall, s | `JK: JK` TF/s | K kernel TF/s |
|---|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz-cpu | 3 | 1343.4 | 6.41 | 0.22 | — |
| benzene-aug-cc-pvdz-gpu | 3 | 1413.6 | 0.51 | 2.90 | 8.48 |
| benzene-cc-pvdz-cpu | 3 | 373.3 | 2.14 | 0.18 | — |
| benzene-cc-pvdz-gpu | 3 | 392.8 | 0.30 | 1.37 | 7.10 |
| nanotube-6-31+g**-cpu | 3 | 16582.7 | 48.63 | 0.35 | — |
| nanotube-6-31+g**-gpu | 3 | 17421.4 | 1.82 | 9.70 | 16.00 |
| peptide-6-31+g**-cpu | 3 | 496.7 | 2.55 | 0.20 | — |
| peptide-6-31+g**-gpu | 3 | 516.9 | 0.43 | 1.24 | 6.45 |
| water-aug-cc-pvdz-cpu | 3 | 2.9 | 0.04 | 0.08 | — |
| water-aug-cc-pvdz-gpu | 3 | 3.1 | 0.17 | 0.02 | 0.33 |
| water-cc-pvdz-cpu | 3 | 0.8 | 0.02 | 0.05 | — |
| water-cc-pvdz-gpu | 3 | 0.8 | 0.17 | 0.01 | 0.09 |

Medians over repeats. `JK: JK` covers J, K, host-side setup and any transfer, so its rate is the whole builder's; the K kernel column exists only for the GPU arm, where cuEST prints per-call kernel milliseconds. A dash means the quantity is not available for that arm, not zero.
