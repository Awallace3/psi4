Scaling of **JK: JK**.

| System | Basis | Narrow | Wide | Narrow median, s | Wide median, s | Measured speedup | Parallel eff. | Projected 56T, s | Projected speedup | Asymptote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 8T | 24T | 9.55 | 6.44 | 1.48× | 49% | 5.5 | 1.72× | 1.96× |
| nanotube | 6-31+g** | 8T | 24T | 72.89 | 40.72 | 1.79× | 60% | 31.5 | 2.31× | 2.96× |
| peptide | 6-31+g** | 8T | 24T | 3.80 | 2.72 | 1.40× | 47% | 2.4 | 1.58× | 1.75× |
| water | aug-cc-pvdz | 8T | 24T | 0.07 | 0.06 | 1.27× | 42% | 0.1 | 1.38× | 1.47× |

Projected columns are a two-point Amdahl fit evaluated at 56 threads. The fit passes exactly through both measurements, so it has no residual and its accuracy cannot be judged from these data. It assumes the serial fraction does not grow with width, which memory bandwidth contention makes optimistic, so read the projection as an upper bound on the baseline correction and the asymptote as the ceiling no thread count can beat.
