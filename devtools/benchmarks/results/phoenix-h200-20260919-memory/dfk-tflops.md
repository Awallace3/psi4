| Case | Repeats | K GFLOP | `JK: JK` wall, s | `JK: JK` TF/s | K kernel TF/s |
|---|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz-cpu | 3 | 1343.4 | 6.95 | 0.20 | — |
| benzene-aug-cc-pvdz-gpu | 3 | 1413.6 | 0.51 | 2.87 | 8.48 |
| benzene-cc-pvdz-cpu | 3 | 373.3 | 2.31 | 0.17 | — |
| benzene-cc-pvdz-gpu | 3 | 392.8 | 0.30 | 1.38 | 7.16 |
| nanotube-6-31+g**-cpu | 3 | 16582.7 | 51.02 | 0.33 | — |
| peptide-6-31+g**-cpu | 3 | 496.7 | 2.76 | 0.19 | — |
| water-aug-cc-pvdz-cpu | 3 | 2.9 | 0.04 | 0.08 | — |
| water-aug-cc-pvdz-gpu | 3 | 3.1 | 0.17 | 0.02 | 0.35 |
| water-cc-pvdz-cpu | 3 | 0.8 | 0.02 | 0.05 | — |
| water-cc-pvdz-gpu | 3 | 0.8 | 0.17 | 0.01 | 0.10 |

Not modeled:

- `nanotube-6-31+g**-gpu-1`: ValueError: Invalid pattern: '**' can only be an entire path component
- `nanotube-6-31+g**-gpu-2`: ValueError: Invalid pattern: '**' can only be an entire path component
- `nanotube-6-31+g**-gpu-3`: ValueError: Invalid pattern: '**' can only be an entire path component
- `peptide-6-31+g**-gpu-1`: ValueError: Invalid pattern: '**' can only be an entire path component
- `peptide-6-31+g**-gpu-2`: ValueError: Invalid pattern: '**' can only be an entire path component
- `peptide-6-31+g**-gpu-3`: ValueError: Invalid pattern: '**' can only be an entire path component

Medians over repeats. `JK: JK` covers J, K, host-side setup and any transfer, so its rate is the whole builder's; the K kernel column exists only for the GPU arm, where cuEST prints per-call kernel milliseconds. A dash means the quantity is not available for that arm, not zero.
