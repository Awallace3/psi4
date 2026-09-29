# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| protein83 | aug-cc-pvdz | 676 | 617 | 1293 | 2/2 | 4823.68 [4798.93–4848.44] | 374.23 [373.57–374.89] | 12.89× | 2.424e-07 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| protein83 | aug-cc-pvdz | 107472 [107468–107477] | 3138 [3137–3139] | 24034 [24034–24034] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-05 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| protein83 / aug-cc-pvdz | SAPT DISP ENERGY | -0.008907212504 | -0.008907212504 | 0.000e+00 |
| protein83 / aug-cc-pvdz | SAPT ELST ENERGY | -0.003081713896 | -0.003081753156 | 3.935e-08 |
| protein83 / aug-cc-pvdz | SAPT EXCH ENERGY | 0.006849220184 | 0.006849017141 | 2.030e-07 |
| protein83 / aug-cc-pvdz | SAPT IND ENERGY | -0.001395878982 | -0.001395878826 | 2.697e-10 |
| protein83 / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.006535585198 | -0.006535827345 | 2.424e-07 |

## Failed or incomplete measurements

```json
[]
```
