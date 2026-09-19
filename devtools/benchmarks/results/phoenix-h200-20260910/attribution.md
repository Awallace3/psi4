| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 129.5 | 22.2 | 5.85× | 19.3× | 8.6× | 8% | 78% | 1.08× |
| benzene-cc-pvdz | 49.8 | 19.6 | 2.55× | 14.0× | 3.0× | 11% | 65% | 1.08× |
| nanotube-6-31+g** | 344.9 | 45.6 | 7.56× | 39.7× | 8.3× | 24% | 62% | 1.26× |
| peptide-6-31+g** | 63.5 | 28.5 | 2.23× | 10.1× | 2.5× | 10% | 75% | 1.07× |
| water-aug-cc-pvdz | 7.9 | 7.5 | 1.06× | 0.3× | 1.5× | -25% | 210% | 1.01× |
| water-cc-pvdz | 6.4 | 7.4 | 0.86× | 0.1× | 0.8× | 15% | 38% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.
