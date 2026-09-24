Control: `pre-merge, libcuest 0.2.1.2 (M3)` → Treatment: `pre-merge, libcuest 0.2.2.2 (M5)`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 5.89 → 5.95 (~) | 6.83 → 6.66 (~) | 0.86× → 0.89× | 1093.02 → 1141.36 (1.04×) | 890.06 → 915.41 (1.03×) | 786.00 → 758.00 (~) |
| water | aug-cc-pvdz | 82 | 7.43 → 7.46 (~) | 7.68 → 6.73 (0.88×) | 0.97× → 1.11× | 1519.73 → 1576.21 (1.04×) | 929.02 → 959.77 (1.03×) | 780.00 → 780.00 (~) |
| benzene | cc-pvdz | 228 | 48.15 → 47.64 (0.99×) | 22.50 → 16.47 (0.73×) | 2.14× → 2.89× | 7916.48 → 7964.49 (1.01×) | 1171.89 → 1191.46 (1.02×) | 3132.00 → 3132.00 (~) |
| peptide | 6-31+g** | 250 | 62.63 → 62.11 (0.99×) | 28.77 → 24.13 (0.84×) | 2.18× → 2.57× | 7746.24 → 7795.52 (1.01×) | 1139.37 → 1163.09 (1.02×) | 3150.00 → 3150.00 (~) |
| benzene | aug-cc-pvdz | 384 | 121.65 → 120.26 (0.99×) | 26.46 → 19.04 (0.72×) | 4.60× → 6.32× | 16466.48 → 16520.96 (1.00×) | 1280.23 → 1302.24 (1.02×) | 3734.00 → 3734.00 (~) |
| nanotube | 6-31+g** | 548 | 338.99 → 322.09 (0.95×) | 52.44 → 39.23 (0.75×) | 6.46× → 8.21× | 31655.85 → 31688.42 (~) | 1911.08 → 1935.22 (1.01×) | 5386.00 → 5386.00 (~) |

**The two builds do not agree numerically.** These cases moved by more than their own repeats do, which a change to memory accounting cannot explain:

- benzene/aug-cc-pvdz: 2.077440e-06 → 2.077440e-06 Eh (scatter ±3.4e-14)
