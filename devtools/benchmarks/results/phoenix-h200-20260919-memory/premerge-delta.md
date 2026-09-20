Control: `pre-merge d91b5f8e81` → Treatment: `merged ee6161a3b6`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 5.89 → 6.07 (~) | 6.83 → 6.67 (0.98×) | 0.86× → 0.91× | 1093.02 → 1114.36 (1.02×) | 890.06 → 917.25 (1.03×) | 786.00 → 774.00 (~) |
| water | aug-cc-pvdz | 82 | 7.43 → 7.72 (1.04×) | 7.68 → 6.72 (0.87×) | 0.97× → 1.15× | 1519.73 → 1512.98 (~) | 929.02 → 948.66 (1.02×) | 780.00 → 780.00 (~) |
| benzene | cc-pvdz | 228 | 48.15 → 49.18 (1.02×) | 22.50 → 16.34 (0.73×) | 2.14× → 3.01× | 7916.48 → 7429.83 (0.94×) | 1171.89 → 1192.84 (1.02×) | 3132.00 → 3132.00 (~) |
| peptide | 6-31+g** | 250 | 62.63 → 63.69 (1.02×) | 28.77 → 23.78 (0.83×) | 2.18× → 2.68× | 7746.24 → 7304.40 (0.94×) | 1139.37 → 1163.50 (1.02×) | 3150.00 → 2802.00 (~) |
| benzene | aug-cc-pvdz | 384 | 121.65 → 123.84 (1.02×) | 26.46 → 18.88 (0.71×) | 4.60× → 6.56× | 16466.48 → 14899.74 (0.90×) | 1280.23 → 1279.42 (~) | 3734.00 → 2878.00 (0.77×) |
| nanotube | 6-31+g** | 548 | 338.99 → 327.80 (0.97×) | 52.44 → 38.54 (0.73×) | 6.46× → 8.50× | 31655.85 → 27044.15 (0.85×) | 1911.08 → 1937.77 (1.01×) | 5386.00 → 8914.00 (~) |

Every case reproduced its CPU-vs-GPU energy difference to within that case's own repeat-to-repeat scatter, so the two builds agree numerically on this suite.
