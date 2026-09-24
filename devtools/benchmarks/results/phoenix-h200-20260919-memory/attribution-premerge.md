| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 120.3 | 19.0 | 6.32× | 14.0× | 10.1× | 6% | 81% | 1.06× |
| benzene-cc-pvdz | 47.6 | 16.5 | 2.89× | 8.1× | 3.7× | 6% | 69% | 1.05× |
| nanotube-6-31+g** | 322.1 | 39.2 | 8.21× | 28.2× | 10.4× | 18% | 68% | 1.19× |
| peptide-6-31+g** | 62.1 | 24.1 | 2.57× | 6.2× | 3.1× | 6% | 78% | 1.05× |
| water-aug-cc-pvdz | 7.5 | 6.7 | 1.11× | 0.2× | 1.6× | -18% | 161% | 1.01× |
| water-cc-pvdz | 6.0 | 6.7 | 0.89× | 0.1× | 0.9× | 22% | 28% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.
