Control: `M4 (3e107987f7 source)` → Treatment: `head 9317f406b2 (H1)`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 6.31 → 6.22 (~) | 6.92 → 6.76 (~) | 0.91× → 0.92× | 1112.52 → 1076.11 (0.97×) | 911.02 → 912.72 (~) | 774.00 → 774.00 (~) |
| water | aug-cc-pvdz | 82 | 7.96 → 7.79 (0.98×) | 7.00 → 6.82 (~) | 1.14× → 1.14× | 1506.60 → 1456.85 (0.97×) | 944.31 → 946.97 (~) | 792.00 → 780.00 (~) |
| benzene | cc-pvdz | 228 | 49.70 → 49.76 (~) | 16.64 → 16.43 (~) | 2.99× → 3.03× | 7422.93 → 7276.42 (0.98×) | 1191.74 → 1189.66 (~) | — |
| peptide | 6-31+g** | 250 | 65.63 → 65.75 (~) | 24.24 → 26.28 (1.08×) | 2.71× → 2.50× | 7297.37 → 7137.77 (0.98×) | 1162.11 → 1161.61 (~) | 2802.00 → 3150.00 (1.12×) |
| benzene | aug-cc-pvdz | 384 | 123.81 → 125.06 (1.01×) | 19.26 → 19.29 (~) | 6.43× → 6.48× | 14893.33 → 14764.10 (0.99×) | 1279.71 → 1278.77 (~) | 3218.00 → 3608.00 (~) |
| nanotube | 6-31+g** | 548 | 322.91 → 337.21 (1.04×) | 39.07 → 39.49 (~) | 8.26× → 8.54× | 27051.21 → 26762.16 (0.99×) | 1933.54 → 1933.71 (~) | 5386.00 → 8914.00 (~) |

**The two builds do not agree numerically.** These cases moved by more than their own repeats do, which a change to memory accounting cannot explain:

- nanotube/6-31+g**: 9.278977e-08 → 9.286653e-08 Eh (scatter ±7.1e-11)
