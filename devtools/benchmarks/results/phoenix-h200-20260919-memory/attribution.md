| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene-aug-cc-pvdz | 123.8 | 19.3 | 6.43× | 12.6× | 10.4× | 6% | 79% | 1.05× |
| benzene-cc-pvdz | 49.7 | 16.6 | 2.99× | 7.2× | 3.8× | 6% | 66% | 1.05× |
| nanotube-6-31+g** | 322.9 | 39.1 | 8.26× | 26.7× | 10.5× | 16% | 66% | 1.18× |
| peptide-6-31+g** | 65.6 | 24.2 | 2.71× | 5.9× | 3.3× | 5% | 74% | 1.04× |
| water-aug-cc-pvdz | 8.0 | 7.0 | 1.14× | 0.3× | 1.6× | -13% | 122% | 1.01× |
| water-cc-pvdz | 6.3 | 6.9 | 0.91× | 0.1× | 0.9× | 25% | 23% | 1.00× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.
