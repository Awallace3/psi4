# GPU4PySCF versus Psi4 cuEST + CUDA LibXC

Comparison branch: `saptdft_cuest_pyscf`. This is restricted supermolecular
DFT/DF-SCF, **not** SAPT, counterpoise, gradients or a repetition of the earlier
interaction-energy certification.

## Protocol

- Fixed timing-suite dimers: peptide (24), nanotube (42), Protein83, Protein157.
  No PLA15/317-atom case. Input XYZ files are copied, staged and SHA256 pinned.
- Spherical cc-pVDZ: respectively 210, 444, 766, 1478 AOs.
- Full density-fitted J **and K**, def2-universal-jkfit in both engines.
- SVWN/VWN-RPA, PBE, M06-L, B3LYP/VWN-RPA, ωB97X. Explicit
  `HYB_GGA_XC_B3LYP` and `LDA_X,LDA_C_VWN_RPA` match Psi4's actual components;
  do not mix these with the distinct VWN5 variants.
  Record actual exchange parameters and require agreement for timing ratios.
- 99 × 590, unpruned; nominal Becke weights. These are **not identical grids**:
  radial construction, orientation, weight/point retention and implementation
  differ. Earlier cuEST numerical checks demonstrated quadrature discrepancies.
- E tolerance 1e-9 Eh; orbital gradient tolerance 1e-7; 200 iterations.
  Psi4 uses an RMS commutator criterion, PySCF an orbital-gradient norm.
  SAD/atom guesses and DIIS implementations also differ. This is a matched
  accuracy-target/end-to-end implementation comparison, not identical iterations.
- Psi4 native FP64 cuEST (mixed precision **off**) versus GPU4PySCF's standard
  FP64 route. Baseline is not a best-tuned mixed-precision cuEST speed claim.
- 8 CPU threads, 96 GiB internal budget, one H200, sequential independent
  processes on the same allocation. Reverse engine order every other repeat.
  Do not substitute GPUs across arms or silently fall back to CPU/cuTENSOR-less.
- Three fresh-process timing repeats, then one separate instrumented repeat.
  No hidden warm SCF/reuse. CUDA initialization/setup inside the energy call
  remains part of that call; caches/OS effects remain in observed repeat spread.

**Headline `calculation_wall_s`** spans post-import runtime configuration,
device initialization, molecule/basis construction and converged energy in
both engines. It ends before result validation/provenance work.
`scf_wall_s` additionally surrounds `psi4.energy` or `mf.kernel` and includes their own
SCF setup/initialization. PySCF molecule/object creation is separately timed;
Psi4's driver does more setup inside its energy call. These are honest API-call
boundaries, **not exactly identical pure iterative SCF boundaries**.
`process_wall_s` includes imports, validation/provenance and process teardown;
it is a harness end-to-end metric, not standalone chemistry execution cost.
SCF iterations and min/median/max accompany medians.

## Profiling and accuracy interpretation

Psi4 saves complete C++ hierarchical timer records, including cuEST setup,
J/K, XC assembly and CUDA LibXC. The actual JK header and CUDA functional timer
are mandatory; a host-LibXC build cannot masquerade as this GPU arm.
GPU4PySCF's extra diagnostic wraps DF setup, J/K, grid, XC and diagonalization,
with CUDA synchronization at each boundary. It **adds overhead** and is excluded
from headline medians. Timers are inclusive/nested: **never sum them blindly**.
Native PySCF textual timing diagnostics are also preserved in `pyscf.out`.
Actual LibXC component names/IDs and weights are validated against the same
explicit configuration in both workers, not just exact-exchange fractions.

The controller and workers validate exactly one visible CUDA GPU and the same
selected UUID; NVML monitoring is restricted to that UUID.
`gpu-samples.csv` provides 1-second allocation-level utilization, GPU-memory and
power observations, **not exact per-worker peak device memory**. Worker Unix
timestamps permit correlation. Host `maxrss_kib` is per fresh worker.
CuPy's final reserved pool is not reported as peak memory.

`summary.json`/CSV retain failed/missing repeats. Process nonzero exits invalidate
results even after a successful-looking JSON write. Ratios require complete
unprofiled repeats and equal AO/auxiliary dimensions and exchange parameters.
`psi4_over_pyscf_calculation_time > 1` means GPU4PySCF was faster on the common
calculation boundary. The additional SCF/API ratio has asymmetric setup scope
and must not be presented as equivalent pure iterative throughput.
`energy_review` flags |Δ total energy| > 1e-4 Eh as a diagnostic, **not**
certification. Never apply the former 1e-5 Eh IE tolerance to these total energies.
Different quadratures/DF implementations and LibXC versions can contribute.
Raw final energies and actual backend parameters remain available for review.

## Phoenix setup

Reuse the explicitly identified historical CUDA-LibXC staged build from the
prior rung campaign; do not pretend its compiled revision is this harness HEAD.
The manifest pins staged core, Python drivers, basis data and LibXC, verifies
them unchanged, and workers record actual mapped shared libraries.
This comparison worktree owns the harness, not a rebuilt Psi4 binary.
Each campaign archives/hashes the harness, captures both installed pip locks,
and pins PySCF/GPU4PySCF source, basis and extension files. Actual mapped library
paths are recorded. Transitive CUDA/CuPy/cuTENSOR binaries are version-locked,
not exhaustively hashed; this is not a hermetic container.

The new dedicated `gpu4pyscf_compare` conda environment uses Python 3.11,
GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.4.1/cuTENSOR 2.2.0, CUDA 12.6.1 module.
Psi4 workers retain their separate `p4cuest_sapt` environment and CUDA-LibXC
libraries. Do **not** inherit Psi4's CUDA13 libraries into PySCF's CUDA12 process.
The setup allocation creates the environment (refuses an existing path), captures
pip/conda locks, and runs one peptide calculation per engine per functional.
Preflight must pass before production timing submissions.

Sync code by committing/pushing this branch and fetching/creating its Phoenix
worktree. Stage only the four untracked XYZ inputs with rsync into each unique
run's `inputs/geometries/`. Create `logs/` **before** sbatch.

Example submission (resolve absolute values first):

```bash
sbatch --test-only --account=gts-cs207-chemx --partition=gpu-h200 \
  --qos=embers --gres=gpu:h200:1 --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=128G --time=02:00:00 --signal=TERM@120 \
  --output="$RUN/logs/%x-%j.out" --error="$RUN/logs/%x-%j.err" \
  "$REPO/devtools/benchmarks/gpu_compare/job.sh" "$REPO" "$COMMIT" "$RUN" setup
# Repeat without --test-only to submit once the route is validated.
# Later, mode "campaign peptide" or "campaign protein157" runs three repeats
# plus separate diagnostic profiles; one system per allocation bounds recovery.
```

Free preemptible **embers only**; no inferno without fresh explicit approval.
TERM is forwarded to srun; the controller terminates active worker process groups.
Each result is atomic. An interrupted campaign is not complete; use a new run
directory for a recovery, preserving previous failures. Never advance a queued
job's commit-pinned checkout. `COMPLETE.json` and process records, not SLURM's
COMPLETED alone, determine scientific completion. `energy_review` remains distinct.

Local harness checks (no installed chemistry packages required):

```bash
python3 -m unittest discover -s devtools/benchmarks/gpu_compare -p 'test_*.py'
bash -n devtools/benchmarks/gpu_compare/job.sh
bash -n devtools/benchmarks/gpu_compare/payload.sh
```

Installation reference: https://github.com/pyscf/gpu4pyscf

## Pilot startup recovery

Setup job `13981566` ran on H200/embers, installed the dedicated GPU environment
and passed `pip check`, then failed before any SCF because `--psiapi` emits an
expansion of `$PYTHONPATH` and the wrapper had unset it under `set -u`.
The corrected wrapper exports an empty value instead, with a shell regression
test that reproduces the historical output. No numerical results exist for
that job. Recovery uses `campaign peptide` with the existing installed
environment, a fresh run directory and one timing repeat plus a separate
instrumented repeat. This is still a preflight, not the full timing matrix.

## Completed consolidated comparison (2026-10-13)

At the user's request, production stopped at Protein83 (not Protein157) and
used one explicitly authorized inferno H200 allocation, `14000107`.
All 120 workers succeeded: three systems × five functionals × two engines ×
three independent timings plus one diagnostic. All 15 comparisons are eligible,
and staged build hashes remained unchanged. The paid allocation was released
after scientific completion with user approval; monitoring is disabled.

Psi4's common calculation-time advantage was 1.34–2.85× overall and
1.36–2.31× on Protein83. Maximum absolute total-energy difference was
3.0011e-5 Eh (Protein83/ωB97X), below the 1e-4 Eh review threshold.
GPU4PySCF's Protein83 profiles are XC-dominated and take more iterations.
Neither time-to-convergence nor inclusive diagnostic clocks establish an
identical-grid, identical-iteration, kernel-only speedup.

Immutable campaign:
`/storage/project/r-cs207-0/awallace43/runs/gpu-dft-compare/20261013T113848Z-interactive-inferno/production/campaign`.
Production harness commit: `9595a76753`. Compact results and prior failed
diagnostics are separately archived locally under
`.pi/gpu_compare/archive/20261013-inferno/`.

Render the audited fixed three-system campaign (requires an unused output directory):

```bash
python devtools/benchmarks/gpu_compare/report.py \
  --campaign .pi/gpu_compare/archive/20261013-inferno/production/campaign \
  --output .lavish/gpu-dft-comparison-20261013
```

The renderer checks completion, worker identities/exit codes, GPU UUID and
config hashes, then independently replays the summary before writing JSON,
Markdown and self-contained HTML. Summary comparison ignores presentation order
because sorted JSON config keys lose the original functional insertion order.
The report preserves timing ranges, host RSS and whole-device memory samples.
Its palette/type follows the existing cuEST report style, without external assets.

### Not comparable to the October 4 SAPT total

The October 4 report's Protein83 **140.10 s** is a complete
SAPT(DFT)-D4(I)/PBE0 calculation with iterative GRAC, normally nine SCF solves,
6-31+G** (922 dimer AOs), and **host** LibXC in every arm.
Its GRAC stages alone take 90.47 s; the two final corrected DFT stages
take 35.17 s combined. The new comparison instead times one uncorrected
supermolecular DFT solve, cc-pVDZ (766 AOs), and CUDA LibXC.
Both reports use seconds, but neither their workload boundaries nor their
scientific builds match. Do not interpret their headline ratio as a controlled
before/after acceleration. See `outputs/gpu-dft-oct4-comparison.md` for
source-backed stage accounting and the remaining comparability differences.

## Augmented triple-zeta follow-up

The user explicitly authorized one **new inferno interactive H200 allocation**
for preflight and production, including Protein157, with five-minute progress
checks. `config-aug-cc-pvtz.json` changes the orbital basis to spherical
aug-cc-pVTZ and the internal host budget to 160 GiB (256G requested host memory),
retaining the same def2-universal-jkfit auxiliary basis, functional definitions,
99×590 nominal grids and convergence targets. Retaining the auxiliary basis
isolates the primary-basis change; this is not a certification of auxiliary
completeness for augmented triple-zeta.

| System | Primary AOs | Auxiliary functions | Unscreened packed DF tensor upper bound |
|---|---:|---:|---:|
| peptide | 782 | 1010 | 2.3 GiB |
| nanotube | 1564 | 2238 | 20.4 GiB |
| Protein83 | 2806 | 3735 | 109.6 GiB |
| Protein157 | 5382 | 7265 | 784.1 GiB |

The last column is `8 * Naux * Nao * (Nao+1) / 2`, **not** a forecast of
allocated GPU memory. Screening, blocking, direct J paths and host storage
can change actual memory substantially. Largest-system hybrid and
range-separated paths require feasibility checks; do not silently change basis,
precision, DF parameters or GPU route to obtain a successful timing.

`interactive_step.sh` checks the live reservation's account/QoS/state,
the pinned clean checkout, and unique phase output paths before running
the isolated payload using absolute `/usr/bin/env` and `/bin/bash`.
`COMPARE_CONFIG`, `COMPARE_FUNCTIONALS`, `COMPARE_REPEATS`,
`COMPARE_PROFILES=0`, and `COMPARE_FAIL_FAST=1` support focused preflight.
Every selected functional and configuration is copied/hash-pinned.
Production failures remain explicit; fail-fast avoids spending extra repeats
on an already failing large-basis path. No unsupported-basis counts are accepted.
