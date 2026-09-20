# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 5.89 [5.84–6.10] | 6.83 [6.79–6.92] | 0.86× | 5.843e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 7.43 [7.41–7.51] | 7.68 [7.59–7.97] | 0.97× | 5.051e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 48.15 [48.12–48.22] | 22.50 [22.12–23.32] | 2.14× | 2.341e-06 | FAIL |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 121.65 [121.59–122.17] | 26.46 [26.20–26.48] | 4.60× | 2.077e-06 | FAIL |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 62.63 [62.11–62.72] | 28.77 [28.17–31.51] | 2.18× | 4.720e-08 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 3/3 | 338.99 [336.27–340.09] | 52.44 [51.11–52.46] | 6.46× | 9.285e-08 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| water | cc-pvdz | 1093 [1085–1099] | 890 [890–890] | 786 [774–2648] | per-process | 0.5 | timed region |
| water | aug-cc-pvdz | 1520 [1499–1525] | 929 [928–931] | 780 [780–2668] | per-process | 0.5 | timed region |
| benzene | cc-pvdz | 7916 [7916–7926] | 1172 [1168–1173] | 3132 [3132–3132] | per-process | 0.5 | timed region |
| benzene | aug-cc-pvdz | 16466 [16453–16480] | 1280 [1279–1283] | 3734 [3734–3734] | per-process | 0.5 | timed region |
| peptide | 6-31+g** | 7746 [7744–7762] | 1139 [1139–1141] | 3150 [2802–3150] | per-process | 0.5 | timed region |
| nanotube | 6-31+g** | 31656 [31573–31663] | 1911 [1910–1911] | 5386 [4484–9588] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-06 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012689512780 | -0.012689530381 | 1.760e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010830298358 | 0.010830257528 | 4.083e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 4.429e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.008087035897 | -0.008087094328 | 5.843e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011314630000 | -0.011314617647 | 1.235e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010087438972 | 0.010087477131 | 3.816e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098793 | 9.997e-13 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007718047523 | -0.007717997012 | 5.051e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002987010440 | -0.002988513768 | 1.503e-06 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.011832611246 | 0.011834951820 | 2.341e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650878 | -0.001350650888 | 1.138e-11 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.004962451274 | -0.004961614038 | 8.372e-07 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.003435953786 | -0.003437433767 | 1.480e-06 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.012199781315 | 0.012201858754 | 2.077e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162542 | -0.001451161813 | 7.897e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.005144736215 | -0.005144138028 | 5.982e-07 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015531989909 | -0.015532010908 | 2.103e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.014757057567 | 0.014757031393 | 2.617e-08 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319096 | -0.004728319083 | 2.810e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013229519562 | -0.013229566754 | 4.720e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.024720720508 | -0.024720718672 | 1.911e-09 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.057773743032 | 0.057773835814 | 9.285e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463905706 | -0.006463924103 | 1.858e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.001308839814 | -0.001308763451 | 7.639e-08 |

## Failed or incomplete measurements

```json
[]
```
