# Why the A100 speedup is below 10×

## Hardware and utilization: measured, not assumed

The CUDA-visible device was resolved via its PCI bus ID and NVIDIA UUID to **NVIDIA A100-SXM4-80GB**, UUID `GPU-7027904a-5a89-6206-de14-6fceedb9846c`—the same device recorded for the original benchmarks. **It was not an L40S.** Driver 595.71.05; the environment uses CUDA 13.3 packages / libcuEST 0.2.1.2.

Allocation **13020320**, step **12**, completed `0:0` in 2m40s. These are new, single-run diagnostics, not replacements for the published three-run timing medians. GPU and process CPU activity were sampled at 200 ms intervals for each whole fresh child process, including imports/setup.

| Profile | energy() wall, s | Mean GPU activity | Samples at 0% | Samples ≥90% | Mean CPU cores used |
|---|---:|---:|---:|---:|---:|
| nanotube-native | 57.55 | 34.1% | 53.5% | 21.4% | 2.65 / 8 |
| nanotube-mixed | 57.26 | 33.7% | 51.5% | 24.3% | 2.60 / 8 |
| benzene-native | 27.68 | 26.6% | 50.6% | 10.3% | 3.15 / 8 |

**The GPU was not continuously busy.** NVML GPU-utilization windows may overlap; these percentages measure sampled device activity, **not SM occupancy or fraction of peak FP64 FLOP/s**. They are not retrospective utilization measurements of the earlier runs. A Nsight executable is available, but this collection did not capture a kernel/CPU-stack trace.

## The measured phase bottleneck

Flat `timer.dat` sections were parsed once per process; their duplicated call-tree sections were excluded. Timers are inclusive/nested and must not all be added together. Values below are medians of the three original matched runs, not telemetry-run timings.

| System / basis | CPU J/K, s | GPU J/K, s | J/K speedup | CPU XC, s | GPU XC, s | GPU total, s | XC / GPU total |
|---|---:|---:|---:|---:|---:|---:|---:|
| benzene / aug-cc-pvdz | 15.86 | 0.88 | 18.0× | 54.41 | 19.29 | 28.13 | 68.6% |
| nanotube / 6-31+g** | 68.29 | 3.01 | 22.7× | 138.99 | 41.38 | 57.34 | 72.2% |

`JK: JK` excludes DF setup: the quoted 18–23× factors are **J/K-call speedups**, not full density-fitting or total-method speedups. They show that the J/K acceleration itself is substantial.

Protein157's GPU run spent **269.02 s** in XC potential formation and **113.95 s** in J/K, out of 486.21 s total. Its CPU comparison is separate.

For nanotube, making its remaining ~3.01 s of GPU J/K calls free would improve total speedup only from 4.41× to about **4.65×** (holding everything else fixed). To reach 10× with the other phases unchanged, the ~41.38 s XC phase would have to fall below roughly **9.31 s**. A universal 10× expectation is particularly inappropriate for small systems with initialization and ~2 s of D4-call overhead.

## Why XC/GRAC still stalls on the host

The source confirms that `CUEST_XC=True` does **not** mean all functional work is GPU-native:

1. `RV::compute_V` obtains grid densities on the GPU, then performs blocking device-to-host copies of weights and densities, plus a host transpose.
2. [`evaluate_cuest_grac`](../../../../psi4/src/psi4/libfock/v.cc#L203-L251) builds one Psi4 functional worker and evaluates **256-point blocks serially on the CPU**. Its point loop includes map lookups, density/gradient conversion, and LibXC/GRAC functional evaluation.
3. The non-GRAC LDA/GGA/meta-GGA branches have OpenMP regions, but the GRAC branch calls this serial helper instead. Thus eight requested threads do not parallelize this helper.
4. The resulting potential is copied back to the GPU for AO integration, followed by another device-to-host AO-matrix copy. Buffers/weights/workspaces are repeatedly allocated or rebuilt.

`RV: Form V` contains **all** these host and GPU phases, so its entire time cannot be assigned to the serial helper alone. The source plus device-idle samples identify a strong bottleneck candidate; exact helper, copy, allocation, and kernel costs still require finer timers or Nsight tracing.

## Native FP64, mixed precision, and CUDA 13

Native FP64: **57.55 s** total, **3.004 s** J/K. Mixed precision allowed: **57.26 s** total, **3.001 s** J/K. Maximum component difference: **1.063e-09 Eh**.

There is no material end-to-end difference in this **single pair**. The baseline explicitly requests `CUEST_MIXED_PRECISION=False`; `cuESTJK::preiterations` calls `cuestSetMathMode(..., CUEST_NATIVE_FP64_MATH_MODE)` before iterative J/K work. The enabled flag permits mixed precision; it does **not prove** which internal cuEST kernel/precision was selected on the A100. A kernel trace would be needed for that claim. No L40S measurements or CUDA-12 control were performed, so this is **not evidence for or against a general CUDA-13 regression**.

## Optimization priorities

1. Add subphase timers/NVTX around GRAC host evaluation, density/potential copies, GPU density and AO integration, and allocation/setup; collect a Nsight Systems trace.
2. Parallelize GRAC blocks with one independent functional worker/input set per thread; cache workers and hoist repeated map lookups. Preserve the tested spin factors and nonzero-shift controls.
3. Reuse fixed-grid weights and work buffers; avoid repeated blocking transfers where possible.
4. Longer term, evaluate GRAC on device to eliminate the host round trip. Benchmark and retest energies, densities, and orbital shifts before claiming a 10× total speedup.

No performance-sensitive Psi4 implementation code was changed for this investigation.

## Reproduce

```bash
python devtools/benchmarks/analyze_gpu_profile.py \
  devtools/benchmarks/results/phoenix-a100-20260909/profile \
  --output devtools/benchmarks/results/phoenix-a100-20260909/profile/summary.json \
  --report devtools/benchmarks/results/phoenix-a100-20260909/GPU_PROFILE.md
```

Raw sampled CSVs, per-case JSON and timers, original comparison timers, launcher, runner snapshots, device mapping, and file hashes are preserved in `profile/`. The first telemetry attempt failed before sampling because the launcher used an unavailable runtime API symbol; basis counts from its completed count stage were retained, and step 12 used PCI-bus mapping successfully.
