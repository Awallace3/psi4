# CPU and GPU memory across both GPU DFT campaigns

All unprofiled attempts, including failures. CPU column is the largest recorded Linux peak RSS across those workers; GPU column is the largest 1-second whole-device observation within their process intervals. Neither is a sum of repeat peaks.

| Basis | System | Engine | Passed / attempts | Peak RSS / GiB | RSS coverage | Sampled GPU device peak / GiB |
|---|---|---|---:|---:|---:|---:|
| aug-cc-pvtz | peptide | psi4 | 1/1 | 1.965 | 1/1 | 3.392 |
| aug-cc-pvtz | peptide | pyscf | 1/1 | 1.176 | 1/1 | 2.630 |
| aug-cc-pvtz | protein157 | psi4 | 0/1 | N/A | 0/1 | 1.802 |
| aug-cc-pvtz | protein157 | pyscf | 1/1 | 1.273 | 1/1 | 8.130 |
| aug-cc-pvtz | protein83 | psi4 | 5/5 | 4.826 | 5/5 | 75.415 |
| aug-cc-pvtz | protein83 | pyscf | 4/5 | 1.692 | 5/5 | 101.933 |
| cc-pvdz | nanotube | psi4 | 15/15 | 1.316 | 15/15 | 7.114 |
| cc-pvdz | nanotube | pyscf | 15/15 | 1.685 | 15/15 | 7.644 |
| cc-pvdz | peptide | psi4 | 15/15 | 1.210 | 15/15 | 3.226 |
| cc-pvdz | peptide | pyscf | 15/15 | 1.682 | 15/15 | 2.456 |
| cc-pvdz | protein83 | psi4 | 15/15 | 1.513 | 15/15 | 8.040 |
| cc-pvdz | protein83 | pyscf | 15/15 | 1.690 | 15/15 | 16.317 |

## Interpretation and collection limitations

- RSS is resident host memory, not virtual memory, and does not include GPU memory. Linux KiB is converted to GiB by dividing by 1024²; NVML MiB by 1024.
- cc-pVDZ is completed production with three timings per functional. aug-cc-pVTZ contains only single-run preflights, not production medians.
- Historical RSS is captured inside workers. Protein157/Psi4 exited in native GPU allocation code before writing a result; its peak RSS is unavailable, **not zero**. Aggregate coverage exposes this gap. GPU traces and the OOM log are retained.
- Historical GPU memory is whole-device sampling restricted to the campaign's UUID and correlated to worker Unix start/end times. Samples may miss short peaks; they cannot recover exact per-process GPU allocation peaks retrospectively.
- Protein157/Psi4 requested a 364.065 GiB device buffer with 138.043 GiB free. The failed request is not an allocated/observed peak and is not added to the GPU peak.
- Profiles are retained as separately marked rows in JSON/CSV, excluded from this table.
- New runs add a fresh supervisor's Linux child peak RSS even on nonzero/signal exits, plus controller-side 1-second process-tree RSS and NVML PID-filtered GPU memory. A whole-job cgroup kill can still prevent final usage collection; saved samples remain lower bounds. CPU tree sums can double-count shared pages. Unsupported queries remain N/A.
- Observation overhead is external to chemistry timing boundaries but can perturb execution. Process clocks include supervisor startup/exit, not memory-monitor teardown. Use the same instrumentation for both engines; historical timings used a different observer.

## Audit files

`memory-workers.csv` and `memory-report.json` preserve every worker, status, profile flag, RSS source, GPU scope, sample counts, UUID, and source hashes.
