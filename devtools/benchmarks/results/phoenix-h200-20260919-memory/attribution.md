| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 123.8 | 18.9 | 6.56× | 13.6× | 10.3× | 6% | 78% | 1.06× |
| benzene-cc-pvdz | 49.2 | 16.3 | 3.01× | 7.8× | 3.8× | 6% | 65% | 1.05× |
| nanotube-6-31+g** | 327.8 | 38.5 | 8.50× | 27.7× | 10.6× | 17% | 66% | 1.18× |
| peptide-6-31+g** | 63.7 | 23.8 | 2.68× | 6.4× | 3.2× | 6% | 74% | 1.05× |
| water-aug-cc-pvdz | 7.7 | 6.7 | 1.15× | 0.3× | 1.7× | -12% | 119% | 1.01× |
| water-cc-pvdz | 6.1 | 6.7 | 0.91× | 0.1× | 0.9× | 25% | 23% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.
