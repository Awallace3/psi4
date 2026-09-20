| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 121.6 | 26.5 | 4.60× | 7.6× | 7.1× | 8% | 81% | 1.08× |
| benzene-cc-pvdz | 48.2 | 22.5 | 2.14× | 5.3× | 2.6× | 11% | 69% | 1.08× |
| nanotube-6-31+g** | 339.0 | 52.4 | 6.46× | 24.6× | 7.3× | 23% | 63% | 1.26× |
| peptide-6-31+g** | 62.6 | 28.8 | 2.18× | 7.8× | 2.5× | 10% | 76% | 1.06× |
| water-aug-cc-pvdz | 7.4 | 7.7 | 0.97× | 0.2× | 1.2× | 58% | -166% | 1.01× |
| water-cc-pvdz | 5.9 | 6.8 | 0.86× | 0.1× | 0.8× | 15% | 33% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.
