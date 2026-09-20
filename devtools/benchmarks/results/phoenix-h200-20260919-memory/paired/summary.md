# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 6.07 [6.06–6.34] | 6.67 [6.62–6.71] | 0.91× | 5.843e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 7.72 [7.72–7.73] | 6.72 [6.66–6.77] | 1.15× | 5.051e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 49.18 [49.09–49.19] | 16.34 [16.34–16.36] | 3.01× | 2.341e-06 | FAIL |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 123.84 [123.13–123.94] | 18.88 [18.83–19.00] | 6.56× | 2.077e-06 | FAIL |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 63.69 [63.67–63.90] | 23.78 [23.77–23.82] | 2.68× | 4.724e-08 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 3/3 | 327.80 [327.56–327.85] | 38.54 [38.53–38.70] | 8.50× | 9.285e-08 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| water | cc-pvdz | 1114 [1110–1123] | 917 [914–918] | 774 [774–774] | per-process | 0.5 | timed region |
| water | aug-cc-pvdz | 1513 [1505–1515] | 949 [948–975] | 780 [764–780] | per-process | 0.5 | timed region |
| benzene | cc-pvdz | 7430 [7423–7432] | 1193 [1188–1195] | 3132 [2800–3132] | per-process | 0.5 | timed region |
| benzene | aug-cc-pvdz | 14900 [14884–14903] | 1279 [1276–1280] | 2878 [2878–3734] | per-process | 0.5 | timed region |
| peptide | 6-31+g** | 7304 [7299–7305] | 1164 [1163–1164] | 2802 [2802–3150] | per-process | 0.5 | timed region |
| nanotube | 6-31+g** | 27044 [27042–27045] | 1938 [1937–1938] | 8914 [4484–8914] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-06 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012689512780 | -0.012689530381 | 1.760e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010830298358 | 0.010830257528 | 4.083e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 6.837e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.008087035896 | -0.008087094328 | 5.843e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011314630000 | -0.011314617647 | 1.235e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010087438972 | 0.010087477131 | 3.816e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098794 | 6.248e-13 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007718047523 | -0.007717997013 | 5.051e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002987010440 | -0.002988513768 | 1.503e-06 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.011832611246 | 0.011834951820 | 2.341e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650878 | -0.001350650883 | 1.239e-11 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.004962451274 | -0.004961614032 | 8.372e-07 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.003435953786 | -0.003437433767 | 1.480e-06 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.012199781315 | 0.012201858754 | 2.077e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162543 | -0.001451161719 | 8.258e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.005144736216 | -0.005144137932 | 5.983e-07 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015531989909 | -0.015532010918 | 2.110e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.014757057567 | 0.014757031394 | 2.617e-08 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319097 | -0.004728319082 | 2.732e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013229519563 | -0.013229566760 | 4.724e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.024720720509 | -0.024720718936 | 2.044e-09 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.057773743032 | 0.057773835745 | 9.285e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463905706 | -0.006463924086 | 1.839e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.001308839815 | -0.001308763545 | 7.636e-08 |

## Failed or incomplete measurements

```json
[]
```
