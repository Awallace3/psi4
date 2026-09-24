# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| protein157 | 6-31+g** | 1344 | 442 | 1786 | 1/1 | 12493.85 [12493.85–12493.85] | 419.30 [419.30–419.30] | 29.80× | 8.929e-07 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| protein157 | 6-31+g** | 145841 [145841–145841] | 5407 [5407–5407] | 46650 [46650–46650] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-05 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| protein157 / 6-31+g** | SAPT DISP ENERGY | -0.026973367165 | -0.026973367165 | 0.000e+00 |
| protein157 / 6-31+g** | SAPT ELST ENERGY | -0.007611520422 | -0.007611990745 | 4.703e-07 |
| protein157 / 6-31+g** | SAPT EXCH ENERGY | 0.021141587305 | 0.021142480181 | 8.929e-07 |
| protein157 / 6-31+g** | SAPT IND ENERGY | -0.003207670750 | -0.003207677851 | 7.102e-09 |
| protein157 / 6-31+g** | SAPT TOTAL ENERGY | -0.016650971031 | -0.016650555580 | 4.155e-07 |

## Failed or incomplete measurements

```json
[]
```
