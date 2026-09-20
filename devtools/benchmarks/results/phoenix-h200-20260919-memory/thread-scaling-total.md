Scaling of **energy() wall time**.

| System | Basis | Narrow | Wide | Narrow median, s | Wide median, s | Measured speedup | Parallel eff. | Projected 56T, s | Projected speedup | Asymptote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 8T | 24T | 187.09 | 109.66 | 1.71× | 57% | 87.5 | 2.14× | 2.64× |
| nanotube | 6-31+g** | 8T | 24T | 507.28 | 252.97 | 2.01× | 67% | 180.3 | 2.81× | 4.03× |
| peptide | 6-31+g** | 8T | 24T | 102.01 | 65.19 | 1.56× | 52% | 54.7 | 1.87× | 2.18× |
| water | aug-cc-pvdz | 8T | 24T | 12.20 | 10.67 | 1.14× | 38% | 10.2 | 1.19× | 1.23× |

Projected columns are a two-point Amdahl fit evaluated at 56 threads. The fit passes exactly through both measurements, so it has no residual and its accuracy cannot be judged from these data. It assumes the serial fraction does not grow with width, which memory bandwidth contention makes optimistic, so read the projection as an upper bound on the baseline correction and the asymptote as the ceiling no thread count can beat.
