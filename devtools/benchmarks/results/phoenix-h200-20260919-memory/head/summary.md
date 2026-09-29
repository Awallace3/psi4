# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 6.22 [6.12–6.47] | 6.76 [6.65–6.78] | 0.92× | 5.843e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 7.79 [7.77–7.80] | 6.82 [6.79–7.48] | 1.14× | 5.051e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 49.76 [49.55–50.01] | 16.43 [16.24–16.64] | 3.03× | 2.341e-06 | PASS |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 125.06 [124.98–125.20] | 19.29 [19.17–19.44] | 6.48× | 2.077e-06 | PASS |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 65.75 [65.71–65.85] | 26.28 [25.78–26.38] | 2.50× | 4.725e-08 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 3/3 | 337.21 [336.27–339.13] | 39.49 [39.37–41.10] | 8.54× | 9.287e-08 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| water | cc-pvdz | 1076 [1073–1080] | 913 [912–919] | 774 [756–774] | per-process | 0.5 | timed region |
| water | aug-cc-pvdz | 1457 [1453–1462] | 947 [944–947] | 780 [764–780] | per-process | 0.5 | timed region |
| benzene | cc-pvdz | 7276 [7239–7279] | 1190 [1188–1193] | — | per-process | 0.5 | timed region |
| benzene | aug-cc-pvdz | 14764 [14764–14790] | 1279 [1278–1282] | 3608 [3222–3734] | per-process | 0.5 | timed region |
| peptide | 6-31+g** | 7138 [7129–7139] | 1162 [1161–1162] | 3150 [3028–3150] | per-process | 0.5 | timed region |
| nanotube | 6-31+g** | 26762 [26757–26763] | 1934 [1933–1936] | 8914 [3402–9974] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-05 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012689512780 | -0.012689530381 | 1.760e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010830298358 | 0.010830257528 | 4.083e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 2.648e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.008087035897 | -0.008087094328 | 5.843e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011314630000 | -0.011314617647 | 1.235e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010087438972 | 0.010087477131 | 3.816e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098793 | 7.008e-13 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007718047523 | -0.007717997012 | 5.051e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002987010440 | -0.002988513768 | 1.503e-06 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.011832611246 | 0.011834951820 | 2.341e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650879 | -0.001350650887 | 1.192e-11 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.004962451275 | -0.004961614036 | 8.372e-07 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.003435953786 | -0.003437433769 | 1.480e-06 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.012199781315 | 0.012201858754 | 2.077e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162543 | -0.001451161707 | 8.396e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.005144736216 | -0.005144137923 | 5.983e-07 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015531989909 | -0.015532010943 | 2.107e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.014757057567 | 0.014757031394 | 2.617e-08 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319097 | -0.004728319106 | 1.103e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013229519563 | -0.013229566781 | 4.725e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.024720720510 | -0.024720718830 | 1.868e-09 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.057773743032 | 0.057773835842 | 9.287e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463905715 | -0.006463924191 | 1.867e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.001308839823 | -0.001308763630 | 7.634e-08 |

## Failed or incomplete measurements

```json
[]
```
