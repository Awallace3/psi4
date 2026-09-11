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
