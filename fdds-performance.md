# FDDS performance campaign

User authorized inferno for this campaign. No changes to shared environments,
old frozen source/build/run directories or other agents' jobs.

## Benchmark contract

- At most89 atoms. Available timing-suite keys below this cap are **peptide
  (24 atoms), nanotube (42), protein83 (83)**. The supplied suite has no
  protein89 key; protein157 and pla15_3pe2 are excluded.
- Real PBE0 monomer SCFs in the common dimer-centered basis, fixed GRAC .136.
  Primary6-31+g**, JK fitting def2-universal-jkfit, dispersion fitting
  aug-cc-pVDZ-ri, pure spherical functions, FP64. Native XC75x302;
  E/D convergence1e-9/1e-8. Hybrid FDDS kernel enabled, unchanged ten-point
  quadrature/lambda .3/rho cutoff/pseudoinverse threshold.
- Prepare physical occupied/virtual coefficients, energies and densities once
  and save immutable caches with geometry/options/orbital/core hashes.
  SCF preparation is **not included** in FDDS timing.
- Replay the real `sapt_mp2_terms.df_fdds_dispersion()` interface, including
  object/integral/DF preparation, hybrid QR/X/Y, ALDA kernel and full coupled/
  uncoupled frequency integration. Record native timers and individual
  pseudoinverse walls. This is FDDS, not the subsequent MP2 exchange-dispersion
  correction and not total SAPT time.
- Fresh processes for native and GPU controls. CPU baseline must never reuse
  a CUDA-initialized process: some native eigensolvers select CUDA based on
  shared handle initialization rather than the current USE_CUEST value.
- Final speedup comparisons require the same physical cache/options and
  source, with CPU/GPU controls on one GPU hardware allocation; the initial
  CPU job is a separate profiling baseline, not a cross-node speedup claim.
  Keep thread/memory settings explicit. Require actual GPU route/timer proof,
  finite energies and no worse than1e-5 Eh agreement (tighter when measured).

## Initial baseline

- Harness `devtools/benchmarks/fdds_benchmark.py`; separate bounded CPU job
  `fdds_cpu_baseline.sbatch` prepares/replays the three systems sequentially.
  Own read-only validated scientific stage9d4b1cd4aa, exact core hash checked;
  USE_CUEST=false and fresh processes throughout the CPU baseline.
- Intended request:24 CPUs/160 GiB/4h/inferno, Psi4 memory96 GiB, per-system
  75-minute bound and30-second kill grace. No GPU requested.
- Small physical-water cache smoke test exposed a Python Matrix accessor
  mismatch before timing; fixed to use NumPy shape. Original failure/cache
  preserved. Corrected real hybrid replay passed:56 AO/236 auxiliary,
  occupied5/5, virtual51/51, **.689684 s FDDS**,22 pseudoinverses
  **.231785 s**, zero GPU calls. Coupled/uncoupled energies
  **-.0023791261270024056 / -.0031182443982617552 Eh**.
  This is harness validation, not a representative speedup measurement.
  No production scientific changes yet.
- Prospective matched GPU profiling job uses fresh-process **CPU/GPU/GPU/CPU**
  controls,8 threads/96 GiB on one H200 allocation; cached SCFs unchanged.
  Its completion gate checks exact cache/source/core/options/host/thread/memory
  identity, actual routing and1e-5 Eh errors. It is not submitted yet.
- Source-based candidate bottlenecks (not yet benchmark conclusions): hybrid
  X/Y pairwise DGEMMs, QR,22 dense pseudoinverses, and density/ALDA setup.
  Choose the next seam from measured profiles; preserve exact equations and
  1e-13 singular-value cutoff rather than replacing pseudoinverse with a solve.
