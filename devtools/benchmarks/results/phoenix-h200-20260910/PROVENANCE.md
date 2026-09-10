# Provenance

Everything in this directory derives from four Phoenix jobs run on 2026-09-10
against one build. The raw per-case trees are not committed; `regenerate.sh`
rebuilds every table here from them.

## Build

| | |
|---|---|
| Source commit | `e971d2957b` ("Include iterative GRAC shifts in benchmark timings"), branch `saptdft_cuest` |
| Staged extension | `build_saptdft_cuest/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so` |
| sha256 | `bc7b9620cd41f69e0d90ac83939a7ce9570c291cebdc574da2e771bae1e4227c` |
| Psi4 | `1.12a1.dev636` |
| libcuEST | 0.2.1.2 (conda-forge) |
| Einsums / pyeinsums | 1.1.2 |
| MKL | 2025.3.0; numpy 2.4.2; libblas 3.11.0 |
| DFT-D4 | 3.7.0 |
| Python | 3.13.11 |
| Driver script | `saptdft_cuest_grac.py`, sha256 `1ae3257474559fe658f94b4e4a2c9d55fd281b554bfc19ec8b25cc4d09f3663d` |
| Geometries | `saptdft_suite_geometries.json`, sha256 `17640f5394055437a1286546d83224614acee6237deb7ed2329f0f14546e45cd` |

Each job verified the commit and the extension's sha256 before running anything,
and verified that the staged binary contains the GRAC host-functional symbol.
Every case asserts `grac_compute == "ITERATIVE"` in its own `result.json`.

## Jobs

| Job | SLURM ID | Partition / QoS | Hardware | Purpose |
|---|---|---|---|---|
| A | 13060539 | gpu-h200 / embers | 1x NVIDIA H200 143771 MiB, driver 595.71.05; 8 cores of Intel Xeon Platinum 8562Y+ | Paired CPU/GPU, 6 cases x 3 repeats, alternating order |
| C | 13061073 | cpu-small / embers | 24 cores of Intel Xeon Gold 6226 | CPU-only 8 vs 24 thread scaling, same node |
| D | 13063342 | cpu-small / embers | 24 cores of Intel Xeon Gold 6226 | protein157 CPU baseline |
| B | 13060540 | gpu-h200 / embers | 1x NVIDIA H200 | protein157 GPU |

Jobs A and C ran the same six systems with the same driver, settings, and binary;
they differ only in hardware and thread width. **They are different nodes with
different CPUs**, which matters for how their numbers may be combined — see
`CPU_BASELINE.md`.

## Settings, identical in both arms

`SAPT_DFT_GRAC_COMPUTE=ITERATIVE` for every case, with no exceptions: both GRAC
shifts are determined by neutral and doublet-cation SCFs inside the timed
`energy()` call. PBE0, DF SCF, 99/590 grid, SCF convergence 1e-9/1e-8, 112 GiB
Psi4 memory, 8 threads (job A) or 8 and 24 (job C),
`SAPT_DFT_INDUCTION_TYPE=NONE` with `SAPT_DFT_DO_DHF=True`,
`CUEST_MIXED_PRECISION=False`. The arms differ only in `USE_CUEST`.

## Observed failure mode: a hung case is not a slow case

`nanotube-6-31+g**-cpu-2` in job A stopped producing output after 40 s and stayed
alive. Its log ends with

```
PID: 2116552 on atl1-1-02-012-23-0 ready for attaching debugger. Once attached set i = 1 and continue
```

That string is in `libEinsums.so.1.1.2`, whose `--einsums:no-attach-debugger`
option is documented as "do not provide mechanism to attach debugger on detected
errors". Einsums installs signal handlers by default and parks the process on a
detected error instead of aborting, so an intermittent fault presents as an
indefinite hang with a zero exit path, no traceback, and no partial result. Only
the per-case `timeout` recovered it, at the cost of the full 7200 s budget. The
same case succeeded on repeat 1 and repeat 3 with the same binary and input.

Anyone extending this campaign should either pass `--einsums:no-attach-debugger`
or have the launcher kill a case as soon as that marker appears in its log. Do
not raise the per-case timeout to accommodate it: that converts a fast failure
into a slow one.
