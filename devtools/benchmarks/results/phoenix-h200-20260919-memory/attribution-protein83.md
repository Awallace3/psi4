| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | XC share of saving | Max speedup from DF-K alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| protein83-6-31+g** | 1034.1 | 122.6 | 8.43× | 32.4× | 7.7× | 26% | 56% | 1.31× |
| protein83-aug-cc-pvdz | 4823.7 | 374.2 | 12.89× | 33.6× | 14.3× | 16% | 77% | 1.18× |

The last column is Amdahl's bound: the end-to-end speedup that would result if DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot produce more than this on this workload, whatever its magnitude.
