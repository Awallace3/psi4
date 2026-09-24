Control: `pre-merge d91b5f8e81` → Treatment: `merged ee6161a3b6`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 5.95 → 6.31 (~) | 6.66 → 6.92 (~) | 0.89× → 0.91× | 1141.36 → 1112.52 (0.97×) | 915.41 → 911.02 (~) | 758.00 → 774.00 (~) |
| water | aug-cc-pvdz | 82 | 7.46 → 7.96 (1.07×) | 6.73 → 7.00 (1.04×) | 1.11× → 1.14× | 1576.21 → 1506.60 (0.96×) | 959.77 → 944.31 (0.98×) | 780.00 → 792.00 (~) |
| benzene | cc-pvdz | 228 | 47.64 → 49.70 (1.04×) | 16.47 → 16.64 (1.01×) | 2.89× → 2.99× | 7964.49 → 7422.93 (0.93×) | 1191.46 → 1191.74 (~) | 3132.00 → 2800.00 (0.89×) |
| peptide | 6-31+g** | 250 | 62.11 → 65.63 (1.06×) | 24.13 → 24.24 (~) | 2.57× → 2.71× | 7795.52 → 7297.37 (0.94×) | 1163.09 → 1162.11 (~) | 3150.00 → 2802.00 (0.89×) |
| benzene | aug-cc-pvdz | 384 | 120.26 → 123.81 (1.03×) | 19.04 → 19.26 (1.01×) | 6.32× → 6.43× | 16520.96 → 14893.33 (0.90×) | 1302.24 → 1279.71 (0.98×) | 3734.00 → 3218.00 (~) |
| nanotube | 6-31+g** | 548 | 322.09 → 322.91 (1.00×) | 39.23 → 39.07 (~) | 8.21× → 8.26× | 31688.42 → 27051.21 (0.85×) | 1935.22 → 1933.54 (~) | 5386.00 → 5386.00 (~) |

Every case reproduced its CPU-vs-GPU energy difference to within that case's own repeat-to-repeat scatter, so the two builds agree numerically on this suite.
