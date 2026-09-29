# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| protein83 | 6-31+g** | 480 | 442 | 922 | 3/3 | 1034.08 [1031.20–1034.22] | 122.63 [121.99–124.13] | 8.43× | 1.577e-07 | PASS |

## Memory

Host memory is the kernel's `VmHWM` high-water mark over the timed region, so it is exact rather than sampled. Device memory is polled and is a **sampled** peak: a spike shorter than the interval is missed.
A blank cell means that quantity was not recorded for every repeat of that arm.

| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |
|---|---|---:|---:|---:|---|---:|---|
| protein83 | 6-31+g** | 58694 [58677–58708] | 2413 [2410–2414] | 15964 [15208–15964] | per-process | 0.5 | timed region |

`whole process` means the kernel did not honor the high-water-mark reset, so that figure also covers Python imports and basis construction and is not comparable with a `timed region` one.

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-05 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| protein83 / 6-31+g** | SAPT DISP ENERGY | -0.008907212504 | -0.008907212504 | 0.000e+00 |
| protein83 / 6-31+g** | SAPT ELST ENERGY | -0.003161457193 | -0.003161374415 | 8.533e-08 |
| protein83 / 6-31+g** | SAPT EXCH ENERGY | 0.006854774451 | 0.006854616806 | 1.577e-07 |
| protein83 / 6-31+g** | SAPT IND ENERGY | -0.001337604635 | -0.001337604017 | 6.262e-10 |
| protein83 / 6-31+g** | SAPT TOTAL ENERGY | -0.006551499881 | -0.006551574123 | 7.621e-08 |

## Failed or incomplete measurements

```json
[]
```
