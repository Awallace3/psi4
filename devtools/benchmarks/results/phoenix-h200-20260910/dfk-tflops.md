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
