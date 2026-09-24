Control: `libcuest 0.2.1.2` → Treatment: `libcuest 0.2.2.2`. A `~` marks a difference inside the combined run-to-run scatter of the two jobs, which is not a measured change.

| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |
|---|---|---:|---|---|---|---|---|---|
| water | cc-pvdz | 48 | 6.07 → 6.31 (~) | 6.67 → 6.92 (~) | 0.91× → 0.91× | 1114.36 → 1112.52 (~) | 917.25 → 911.02 (0.99×) | 774.00 → 774.00 (~) |
| water | aug-cc-pvdz | 82 | 7.72 → 7.96 (1.03×) | 6.72 → 7.00 (1.04×) | 1.15× → 1.14× | 1512.98 → 1506.60 (~) | 948.66 → 944.31 (~) | 780.00 → 792.00 (~) |
| benzene | cc-pvdz | 228 | 49.18 → 49.70 (1.01×) | 16.34 → 16.64 (1.02×) | 3.01× → 2.99× | 7429.83 → 7422.93 (~) | 1192.84 → 1191.74 (~) | 3132.00 → 2800.00 (~) |
| peptide | 6-31+g** | 250 | 63.69 → 65.63 (1.03×) | 23.78 → 24.24 (1.02×) | 2.68× → 2.71× | 7304.40 → 7297.37 (~) | 1163.50 → 1162.11 (~) | 2802.00 → 2802.00 (~) |
| benzene | aug-cc-pvdz | 384 | 123.84 → 123.81 (~) | 18.88 → 19.26 (1.02×) | 6.56× → 6.43× | 14899.74 → 14893.33 (~) | 1279.42 → 1279.71 (~) | 2878.00 → 3218.00 (~) |
| nanotube | 6-31+g** | 548 | 327.80 → 322.91 (0.99×) | 38.54 → 39.07 (1.01×) | 8.50× → 8.26× | 27044.15 → 27051.21 (~) | 1937.77 → 1933.54 (1.00×) | 8914.00 → 5386.00 (~) |

**The two builds do not agree numerically.** These cases moved by more than their own repeats do, which a change to memory accounting cannot explain:

- water/cc-pvdz: 5.843224e-08 → 5.843170e-08 Eh (scatter ±4.9e-13)
