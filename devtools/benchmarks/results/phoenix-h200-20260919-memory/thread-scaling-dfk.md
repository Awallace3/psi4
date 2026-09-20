Scaling of **JK: JK**.

| System | Basis | Narrow | Wide | Narrow median, s | Wide median, s | Measured speedup | Parallel eff. | Projected 56T, s | Projected speedup | Asymptote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene | aug-cc-pvdz | 8T | 24T | 9.31 | 5.94 | 1.57× | 52% | 5.0 | 1.87× | 2.19× |
| nanotube | 6-31+g** | 8T | 24T | 73.61 | 38.70 | 1.90× | 63% | 28.7 | 2.56× | 3.46× |
| peptide | 6-31+g** | 8T | 24T | 3.81 | 2.58 | 1.48× | 49% | 2.2 | 1.71× | 1.94× |
| water | aug-cc-pvdz | 8T | 24T | 0.08 | 0.06 | 1.25× | 42% | 0.1 | 1.35× | 1.44× |

Projected columns are a two-point Amdahl fit evaluated at 56 threads. The fit passes exactly through both measurements, so it has no residual and its accuracy cannot be judged from these data. It assumes the serial fraction does not grow with width, which memory bandwidth contention makes optimistic, so read the projection as an upper bound on the baseline correction and the asymptote as the ceiling no thread count can beat.
