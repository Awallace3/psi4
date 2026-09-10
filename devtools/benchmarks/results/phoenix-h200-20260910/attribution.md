| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 389.1 | 55.3 | 7.03× | 24.7× | 11.1× | 7% | 80% | 1.07× |
| benzene-cc-pvdz | 154.5 | 50.7 | 3.05× | 15.0× | 3.8× | 8% | 68% | 1.06× |
| nanotube-6-31+g** | 1078.8 | 113.3 | 9.52× | 72.2× | 11.3× | 21% | 64% | 1.24× |
| peptide-6-31+g** | 202.3 | 75.8 | 2.67× | 11.3× | 3.1× | 7% | 77% | 1.05× |
| water-aug-cc-pvdz | 22.9 | 18.3 | 1.25× | 0.3× | 1.8× | -7% | 98% | 1.01× |
| water-cc-pvdz | 17.9 | 18.4 | 0.97× | 0.1× | 1.0× | 66% | 22% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.
