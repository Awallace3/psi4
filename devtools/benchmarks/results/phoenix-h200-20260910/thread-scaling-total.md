Scaling of **energy() wall time**.

| System | Basis | Narrow | Wide | Narrow median, s | Wide median, s | Measured speedup | Parallel eff. | Projected 56T, s | Projected speedup | Asymptote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 8T | 24T | 186.63 | 111.53 | 1.67× | 56% | 90.1 | 2.07× | 2.52× |
| nanotube | 6-31+g** | 8T | 24T | 499.93 | 257.15 | 1.94× | 65% | 187.8 | 2.66× | 3.68× |
| peptide | 6-31+g** | 8T | 24T | 99.68 | 64.47 | 1.55× | 52% | 54.4 | 1.83× | 2.13× |
| water | aug-cc-pvdz | 8T | 24T | 12.13 | 10.68 | 1.14× | 38% | 10.3 | 1.18× | 1.22× |

Projected columns are a two-point Amdahl fit evaluated at 56 threads. The fit passes exactly through both measurements, so it has no residual and its accuracy cannot be judged from these data. It assumes the serial fraction does not grow with width, which memory bandwidth contention makes optimistic, so read the projection as an upper bound on the baseline correction and the asymptote as the ceiling no thread count can beat.
