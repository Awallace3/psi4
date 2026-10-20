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

## Submitted baseline and first measured bottleneck

- **14198087** accepted on **gts-cs207 / cpu-small / inferno**,24 CPUs/160 GiB/
  four hours. Effective QoS confirmed by scontrol. Worst-case credit$.79;
  monthly-account available balance$29.51 before submission. Payload pinned
  **1a5a95168f7cb4305c4a86a04d795d0274d859a9**, scientific binary9d4/hash68e344...
  Local staging/smoke, geometry checksum, dry-run and duplicate checks passed.
  Watcher **bd6b34709**. No job cancellation/modification.
  Remote run `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-performance-01`.
  Scheduler dry-run predicted an overnight delay; local profiles run in
  parallel, not substituted into any cross-node speedup ratio.
- Local physical peptide profile:250 AO/1042 auxiliary,20 occupied/230 virtual
  per monomer,8 threads/16 GiB, **38.216089 s FDDS**. Object16.948329 s,
  QR.641465 s, **X1.947119 s/Y6.633875 s**, ALDA3.620404 s,
  integration17.203429 s;22 pseudoinverses5.755491 s (inside integration).
  Coupled/uncoupled FDDS **-.006194210929276167 / -.009442035537422072 Eh**.
  Cache and profile at `build_fdds_cpu/perf-peptide-{cache,cpu}/`;
  native source9d4/core5b2825..., not an optimized GPU measurement.
- Protein83 native profile is running locally (16 threads/64 GiB) to remove
  queue latency from diagnosis; its final measurement is not yet available.

## Candidate: persistent hybrid X/Y block contraction (not yet validated)

- `CuestFDDSExchange` uploads the outer integral block once and each inner
  integral/raw/orthogonal block once. Three exact FP64 cuBLAS products per
  occupied-pair/occupied-virtual pair; two output tensors remain resident
  across all inner blocks. Only completed outer outputs are downloaded.
  Existing DFHelper reads/writes, tensor order and equations are preserved.
- Shared cuEST stream/handle, actual HOST pointer-mode check, checked device
  capacity (80% free), shrinking caller-budgeted blocks only. Stream drains
  before host scratch can be reused and on every enqueued-error unwind;
  runtime failures propagate without native retry. Native/no-cuEST/small/
  unsupported-shape/capacity paths remain available; no new user option.
- Device doubles: `q*[bo*(k+2v)+bi*(v+2k)] + v*k`, with k=v for X,
  k=nocc for Y. Host memory budgets unchanged. Row limits and size arithmetic
  are checked. Y's former zero virtual block when nvir<nocc is corrected
  within the original six-o memory bound; dense-occupied regression added.
- New real-interface tensor tests compare X/QX/Y/QY against independent
  contractions of transformed integral tensors: equal/unequal/dense-occupied,
  whole/blocked host budgets, disabled/huge-threshold/forced GPU controls.
  Prospective full FDDS suite **69** (52 native/control,17 GPU cases);
  focused gate20+real SAPT1, then full69 under bounded GDB in one allocation.
  Collection count verified; runtime validation pending.
- Read-only safety review of the new backend is running. No new GPU numerical
  pass or performance speedup is claimed at this point.
- Red regression confirmed against the old9d4 CPU core: the dense-occupied
  whole-block case failed in1.800 s with DFHelper's axis0 bounds error
  (`rrR`, index18446744073709551615). This confirms the zero-sized virtual
  block problem rather than a timeout hypothesis. Failure XML is preserved at
  `build_fdds_cpu/perf-exchange-tests/dense-red.xml`.
- Independent equal/unequal X/Y reference checks passed against the old core:
  two executed tests, no failures/errors/skips,1.806 s. The new candidate's
  expanded171-test native gate remains separate from this reference check.
- Candidate CPU core was built at8dc714e4d30c272cc09bcbec5a348d3317518082,
  SHA256`8ee2d4357a56a7a1d73334a939ce88ff75999138b4f2e745d360a962717ff601`;
  no-cuEST release build in the dedicated `fdds-opt-cpu-20261020-v1` worktree.
  GPU optimization validation must build its own separate stage.
- Optimization harness uses the verified H200 limit of8 CPU threads/one GPU,
 128 GiB allocation/96 GiB Psi4 memory/four hours. It builds the frozen
  scientific source, requires21 focused plus69 full executed passes, then
  replays matched CPU/GPU/GPU/CPU controls on all three physical caches.
  The separate baseline CPU job is not a dependency of this allocation.
- CUDA build preflight explicitly pins the conda nvcc/toolkit and clears site
  include-path overrides. A compiled runtime/header/device-property probe must
  pass before configuration. The read-only path inspection found conda headers
  under `targets/x86_64-linux/include`, not `include`; its bounded-output
  pipeline ended141 (SIGPIPE), so it is path evidence, not a GPU execution pass.
- Static review `d06a70caa6ac7f823fa86e561e093f8d3` found no proven
  correctness defect in8dc's mappings, capacity arithmetic, host budgets or
  lifetime/error handling. GPU numerical/fault/capacity runtime evidence is
  still required; review is not a test pass.
- First expanded native run: **168 passed/3 failed, zero errors/skips,
  242.05 s**. All failures are new low-memory exchange fixtures rejecting
  forced in-core DFHelper initialization before X/Y. Evidence preserved under
  `/fastscratch/awallace43/fdds-performance-20261020/native-8dc714e4d3/`.
  The fixture now explicitly selects/restores SCF_SUBTYPE=AUTO to permit the
  budgeted out-of-core integral path. Targeted recovery at16 KiB passed all
  six native exchange cases in4.44 s, zero failures/errors/skips.
- Strengthened the final blocked fixture to12,800 bytes, with explicit
  dimension-derived assertions that both X occupied blocks and Y virtual
  blocks contain partial tails. Independent references/tolerances are
  unchanged. A fresh targeted plus complete171-test native gate is required
  for this final test-driver pin; the scientific C++ remains reviewed8dc.
- Final native gate at test-driver **fd2c09cd0cfbfce61a311508ba6c53039025179a**
  passed **171 tests, zero failures/errors/skips,240.780 s**;52 FDDS,
  67 CPSCF,6 SAD,6 lifecycle,8 basis-parser,32 grid-switch. Separate targeted
  six X/Y whole/blocked cases passed2.440 s. Maximum printed FDDS contribution
  error7.105427357601002e-15. Evidence and audited counts:
  `build_fdds_cpu/performance-cpu-fd2c09cd0c/`.
- Scheduler dry-runs found no near-term H200/H100/A100 slot, even with shorter
  wall/memory requests (estimates October28/November1). No optimization GPU
  job has been submitted yet. RTX6000's earlier October20 morning estimate
  is not a capability pass; its older Cascade Lake/RTX6000 node class may
  be below the required compute8 capability. No unsupported GPU or skipped
  suite will be called a correctness pass. Alternate account/QoS dry-runs
  are read-only checks, not authorization to silently change the charge route.
