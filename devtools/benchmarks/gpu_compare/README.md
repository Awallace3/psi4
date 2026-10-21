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

## CPU and GPU memory accounting

Both engines now run under a fresh, stdlib-only resource supervisor
(`memory.py`). After reaping its child, it writes Linux
`RUSAGE_CHILDREN.ru_maxrss` in **KiB**, even when the worker exits nonzero or
is killed by a signal. This is the largest **individual child process** RSS
high-water mark, not summed process-tree memory; the worker's existing
`RUSAGE_SELF` field is also preserved. CPU RSS does not include device memory.
A whole-job/cgroup kill can prevent the supervisor's final write; missing
RSS is explicitly unavailable, never zero.

An independent controller thread records a flushed per-worker `*.memory.csv`
at a nominal **1-second** interval:

- Sum of `/proc` RSS for the worker and descendants (may double-count shared
  pages); before the child identity file exists, the supervisor tree is used.
- NVML compute-process GPU memory filtered by worker-tree PIDs and the
  validated GPU UUID, including cached allocator pools.
- Whole-device used memory on that UUID, separately labeled.

`*.usage.json` and the record's `memory` object retain RSS source, CPU/GPU
observed peaks, sample counts and collection errors. Unsupported GPU process
queries/namespace restrictions must be reviewed; sampled peaks are lower
bounds and are not exact allocator high-water marks. `memory` summaries include
failed workers and coverage counts, independently of timing eligibility.
CPU and GPU sampling use independent threads, so a slow NVML query cannot
block RSS observations. Device and process queries retain independent results;
per-metric successful sample counts and historical-fallback sources are reported.
Interrupted attempts are persisted before the controller re-raises its signal
exception. The supervisor terminates the worker's separate process group,
including lingering descendants, before another attempt is allowed to start.
Sample queries can perturb execution; both arms use the same observer.
Common chemistry clocks remain unchanged; process clocks additionally include
supervisor startup/exit, but exclude sampler shutdown latency.
Previous runs used only worker RSS and allocation-level NVML samples, so their
exact failed-worker RSS and process-attributed GPU peaks cannot be recovered.

Historical data from both bases are reported without modifying raw results:

```bash
python3 devtools/benchmarks/gpu_compare/memory_report.py \
  --campaign .pi/gpu_compare/archive/20261013-inferno/production/campaign \
  --campaign .pi/gpu_compare/archive/20261013-aug-cc-pvtz/peptide_pbe/campaign \
  --campaign .pi/gpu_compare/archive/20261013-aug-cc-pvtz/protein83_all/campaign \
  --campaign .pi/gpu_compare/archive/20261013-aug-cc-pvtz/protein157_pbe/campaign \
  --output outputs/gpu-dft-memory-20261014
```

That Markdown report displays **peak RSS** and observed GPU device peaks with
RSS coverage; JSON/CSV retain all 134 workers including profiles and failures.
The main cc-pVDZ renderer also displays per-functional CPU/GPU memory.
Protein157/Psi4's 364.065-GiB failed allocation request is not an observed
allocated GPU peak.

## Tailscale results page

Reusable dashboard:
https://cos-4a16505.tail7d58ac.ts.net/docs/plots/gpu-dft-comparison/gpu-dft-results/results.html

It consolidates both bases' per-functional timings, worker peak RSS and
sampled GPU memory, plus failed arms and numerical review flags. Only derived
share-safe values are published; raw geometries, environment manifests, machine
paths and GPU UUIDs are omitted. It uses the existing cuEST report style.
The page checks for **republished snapshots** every 15 seconds; it does not
query SLURM and is not a live cluster monitor. The current header accurately
records that the augmented-basis allocation expired without production.
The page defaults to dark mode with larger type and shows NBF and the recorded
PySCF grid storage length. Actual cuEST device-grid counts are currently
unavailable, not zero; PySCF storage counts may include zero-weight padding.
Every historical timing ratio is explicitly provisional because the actual
quadratures have not been matched.

The page now includes interactive measured SVG plots for calculation walltime
(with min–max repeat whiskers), peak CPU RSS and sampled whole-device GPU memory.
Filters apply to both plots and the NBF/grid table; the initial view selects PBE
across both bases. Every mark has an exact-value tooltip and each plot has an
expandable data table. Missing timings/RSS are N/A, not zero; dashed GPU-memory
bars mark failed attempts, not complete-SCF capacity. These remain **historical,
unmatched-grid observations**, not controlled speedup or memory-efficiency claims.
The figure-first shell follows the docs-html template, uses larger 20px body
type and preserves dark/light themes and a mobile contents menu without CDN
dependencies. The new page checks only republished snapshots, not cluster state.

### Required next grid-controlled protocol

Exporter implementation is in progress on this branch. The historical staged
binary has **not** acquired these interfaces, and no native-default grid audit
has been run on a GPU yet. A separate instrumented build must be validated and
identified as a new compiled implementation; do not replace the historical
stage or mix its timings with a rebuild's timings.

`VBase.cuest_grid_points()` queries the actual molecular device grid;
`VBase.export_cuest_grid()` exports owned `(N, 3)` coordinates and `(N, 1)`
**total** weights from the live XC plan, before `finalize()` destroys it.
`test_cuest_grid_export.py` checks count/shape, translated-atom Gaussian
integration (units and layout), ownership, repeatability and lifetime.
This test still needs a compiled cuEST build and allocated CUDA device.

The controller's `--native-default-grids` removes its quadrature overrides
from both engines, including the GPU4PySCF density cutoff override.
`--export-grids --profiles` exports only the separate diagnostic repeat:
Psi4 before SCF iterations release the plan, GPU4PySCF at the first numerical
integration entry **after** native initialization/density pruning and again
after SCF. Subsequent XC entries must match the first mesh's ordered fingerprint;
the final capture additionally checks endpoint equality. A requested export that
never executes is a failed audit, even if the SCF energy succeeds.
Export overhead and host/device staging memory must not enter
unprofiled timing or memory comparisons. The original 99×590 mode remains
the default for replay; native-default runs must be named separately.

Raw immutable `*.npz` meshes and adjacent fingerprint JSON belong only in
private campaign artifacts, **not** the served gallery. Compare them with:

```bash
python3 devtools/benchmarks/gpu_compare/grids.py \
  /private/psi4/xc-grid.npz /private/pyscf/xc-entry-grid.npz
```

The comparison validates content against recorded fingerprints and required
geometry/basis/functional identity, with grid mode, config/build fingerprints
and effective settings recorded in each artifact. A smaller **native-default**
grid is only identified when both exports declare native-default mode.
Its exact nonzero coordinate/weight
multiset hash ignores order and zero-weight padding but preserves duplicates
and negative weights. Stored, zero-weight and nonzero-weight counts remain
separate; a nonzero-weight count is not automatically a generated physical-point
count. A mesh audit does not alone certify a completed SCF run.

Installed-interface review on Phoenix found only radial-shell/atom-grid
construction in cuEST's public molecular-grid creation interface; no documented
arbitrary molecular point/total-weight importer was found. Molecular-grid
parameter attributes are reserved, so Psi4's nominal nuclear-scheme option does
not establish the partition used by cuEST. GPU4PySCF's installed initialization
checks for existing coordinates, and its AO screening derives from those
coordinates, making fixed arrays a feasible candidate there; it still needs
an allocated-GPU consumption test. **Both engines on the smaller native grid
are not implemented or verified yet.** If GPU4PySCF's native grid is smaller,
an unsupported cuEST import is a blocker, not permission to substitute host XC
or claim that lower matched settings constitute an identical grid.

Before further performance interpretation, export **both native-default**
quadratures from the actual GPU XC paths (not an empty Psi4 host grid):
coordinates in Bohr, total integration weights, grid options/element-level
radial and angular rules, pruning/partition/radii conventions, stored-point
count and nonzero-weight count. Pin raw exports and geometry with hashes.
Compare coordinates and weights independently of point ordering, distinguishing
zero-weight alignment padding from the effective quadrature.

Then identify the smaller default by measured nonzero-weight point count and
run both engines on that **same exported quadrature**, verifying its consumed
coordinates/weights after any sorting or padding. Native-default comparisons
and common-grid comparisons must remain separate, with the same basis, DF,
precision and convergence targets. If the cuEST SDK cannot ingest an arbitrary
external grid, matching nominal options is not an equivalent substitute.
Report that limitation explicitly or implement/validate the required adapter.
An export hook, if it requires a new Psi4 build, changes the scientific build
identity and must be pinned and revalidated before a timing campaign.

The installed Phoenix SDK exposes `cuestXCIntegrationGridCompute` (GPU point
coordinates), `cuestXCIntegrationWeightCompute` (total angular × radial × Becke
weights), and `CUEST_MOLECULARGRID_NUM_POINT`. The historical Python worker
did not expose its opaque cuEST plan; reconstructing a different host grid would
not be a valid audit. Export/import feasibility investigation is in progress;
no grid-controlled campaign has been run.

### Triple-zeta augmentation control

`tz_job.sh` and `tz_control.py` prepare a new, separate control: **cc-pVTZ
versus aug-cc-pVTZ**, peptide/PBE preflight followed by Protein83 across all
five rungs. Their AO counts are 496/782 and 1786/2806, respectively.
This isolates augmentation without also changing double- to triple-zeta.
The internal host budget is 160 GiB, with 256G requested and one H200.

For each system/functional, a diagnostic cc-pVTZ Psi4 solve exports its actual
native-default cuEST coordinates and total weights. Both engines also export
native defaults for **each basis**; the private native-default comparisons record
both measured point counts and identify the smaller active mesh. Missing native
exports prevent the campaign from being marked complete; the native meshes
themselves need not be equal.
The control freezes the **cuEST source mesh**, not necessarily the smaller
native default. GPU4PySCF imports its final weights directly, disables
density pruning and permits zero-weight alignment padding only.

Four successful diagnostic SCFs (two bases × two engines) and exact nonzero
coordinate/total-weight multiset comparisons are mandatory before any timing
repeat for that functional. Equal counts alone fail this gate. The subsequent
fresh timing workers do not export meshes: Psi4 queries the actual device point
count, while GPU4PySCF verifies and installs the same hash-pinned raw mesh.
Full-array verification is therefore diagnostic-arm evidence, not a claim
that every unprofiled cuEST XC entry was copied and hashed.
The controller preserves every failure, including a range-separated DF
batching assertion; a failed gate leaves timings explicitly skipped, not zero.

The chemistry clock includes fixed-grid import/verification for GPU4PySCF;
`grid_import_wall_s` records that setup cost separately. Neither this asymmetry,
different guesses/convergence norms, nor unchanged DF implementation differences
justify calling these clocks identical kernel-throughput measurements.
Peak RSS and independent sampled process/device GPU memory accompany workers,
including failures. Profile memory/export overhead is excluded from timing-arm
headlines. Numerical energy differences still need scientific review.

The GPU wrapper verifies the new compiled build artifact and runs **both**
allocated-GPU exporter tests (skips fail the gate), then the peptide control
before the Protein83 panel. Use a new detached harness checkout rather than
advancing the source checkout pinned by exporter build job14043163.
Submit with explicit `--qos=embers`, `--signal=TERM@120`, unique logs/results,
and five-minute state monitoring by default. On 2026-10-14 the user explicitly
approved cancelling/replacing embers build14043163 with one new inferno CPU
build (2h) and one inferno H200 control job (4h), charged to `gts-cs207`.
This approval does not extend to additional recovery jobs. The build wrapper's
fourth argument records the authorized QoS; defaults remain embers.
The control wrapper's sixth argument likewise records the intended QoS and its
seventh independently pins the build job ID. The revised wrapper can reuse a
validated build for harness-only changes, but requires unchanged `psi4/`,
`cmake/`, `external/`, root CMake and exporter-test inputs between the compiled
and harness commits. It preserves the compiled manifest/core hash and records
both commit identities separately; it never rebuilds the historical stage.

#### Revised stored-DF/full-process protocol

After the user requested an explicitly stored-DF comparison and imports-inclusive
timing, the revised controller runs **two separately labeled DF cohorts**:
`native-strategy` and `stored-df`. Both use the shared actual cuEST quadrature
and matched orbital-overlap policy; the native-strategy label refers only to
DF storage/evaluation, not to leaving every program option at its default.

In GPU4PySCF's stored cohort, `with_df.use_gpu_memory=True` and an explicit
`with_df.build()` construct the FP64 CDERI tensor **inside the timed worker**,
before `kernel()`. It must actually reside on the selected GPU. Direct DF-J
contractions are forbidden; stored-tensor contractions are counted and required.
Lazy RSH DF tensors are explicitly built with the contraction's requested omega,
inside its range-Coulomb context, before consumption; their residency is inspected
too. Peptide PBE **and WB97X** preflights gate the larger panel.
No host spill, CPU/exact-JK fallback,
or smaller auxiliary basis silently rescues a failed allocation. The default
cohort does not prebuild; its observed direct/stored call counts are preserved.
Psi4 retains its cuEST persistent-plan path in both cohorts.

Both engines explicitly use canonical orthogonalization with an absolute
overlap eigenvalue cutoff of **1e-7**. Each worker records actual retained NMO,
not just AO count; diagnostic workers export the full overlap eigenvalue
spectrum. Per-basis cross-engine audits require the same cutoff/rank and
matching spectra before timing repeats. This checks the observed 1-vs-17
orbital deletion discrepancy; it is not a full AO-projector/density certificate.
Timed configs pin the basis-specific audited rank and hashes/paths of diagnostic
proofs and spectra. Every timed worker must reproduce that rank; agreement only
between the timed engines is insufficient. Unprofiled Psi4 mesh identity still
relies on pinned deterministic generation plus a live device-point count, not
independent full-array export during each timed worker.
The separate 1e-12 auxiliary-metric settings remain separate.

Both workers now receive the same explicit **Bohr coordinates**, converted once
from the input XYZ using pinned `geometry_angstrom_per_bohr=0.52917721067`
(Psi4's CODATA2014 convention). Installed PySCF uses0.52917721092 when converting
Angstrom input. Those different constants caused the first shared-grid preflight
to fail its spectrum audit despite matching retained ranks. A CPU-only PySCF2.14
replay changing only that conversion matches the saved Psi4 spectra within
2.5e-14, with the original spectrum tolerance unchanged. Live molecule coordinates
must equal the common Bohr input exactly; normalized coordinate hashes are
recorded and compared, including in overlap artifacts. An identical Angstrom
file hash alone is insufficient. Historical configs without this policy retain
their original unit conversion for replay.

**New headline: `full_process_wall_s`.** A fresh external supervisor measures
worker launch through exit/reaping: Python startup, all imports, initialization,
basis/grid construction/import, DF tensor construction, converged SCF, worker
validation, JSON writing and interpreter/CUDA teardown. It excludes supervisor
startup and the external memory observer's shutdown. Missing measurements fail
closed: no substitution of a post-import chemistry clock. Existing
`calculation_wall_s`, `scf_wall_s`, `imports_wall_s` and `df_prebuild_wall_s` remain
diagnostics, not new headline timings. Fresh processes do not imply cold OS
filesystem caches. Both full-lifetime peak RSS and independent GPU-memory
samples remain active during imports and teardown.
Completion additionally requires a valid supervisor peak RSS and at least one
whole-device GPU-memory observation for every successful timing worker. Missing
measurements never become zero. GPU process attribution is reported separately:
namespace/query unavailability does not invalidate an available whole-device
observation, which is explicitly not worker-attributed. Chemistry/timing success
is retained separately when memory coverage prevents benchmark completion.

Historical reports still use their explicitly labeled post-import clocks and
unmatched grids; do not relabel old results as this revised protocol.
The preceding jobs14056196/14056197 failed at the LibXC import/artifact gate;
no shared-grid chemistry results were produced. The user authorized one
additional inferno CPU build and one H200 job for this revision after validation.
If queuing behind the build, use `afterany` with artifact verification, never
an `afterok` chain. Prepared scripts and passing local unit tests do not mean
that the GPU exporter or fixed-grid integration has been validated.
Subsequently, build14071936 and both exporter tests in H20014071937 passed.
The shared peptide mesh was exactly equal across both engines, but the
overlap-conversion discrepancy and a list-versus-dict stored-tensor verifier
bug stopped preflight: no clean timing repeats or Protein83 panel ran.
See `outputs/gpu-dft-preflight-20261015.md` and the local CPU-only replay helper
`diagnose_overlap_units.py`. A new GPU run is still required to accept the
corrected combined protocol; no additional paid job is implicitly authorized.

Regenerate and publish through the existing tailnet-only `/docs/` mount:

```bash
python3 devtools/benchmarks/gpu_compare/dashboard.py \
  --campaign .pi/gpu_compare/archive/20261013-inferno/production/campaign \
  --campaign .pi/gpu_compare/archive/20261013-aug-cc-pvtz/peptide_pbe/campaign \
  --campaign .pi/gpu_compare/archive/20261013-aug-cc-pvtz/protein83_all/campaign \
  --campaign .pi/gpu_compare/archive/20261013-aug-cc-pvtz/protein157_pbe/campaign \
  --output .lavish/gpu-dft-results/results.html
python3 /home/awallace43/.pi/agent/skills/python-plot-tailscale/scripts/plot_gallery.py \
  --name gpu-dft-comparison --src .lavish/gpu-dft-results --glob '*.html' \
  --title 'GPU DFT comparison: timing and memory'
```

When adding a new campaign, update the dashboard's status/count header too;
the current header describes these four fixed archived campaign directories.
Do not change tailscale serving/Funnel settings to publish updates.

### Matched-grid recovery 14076480 (October 16)

The corrected combined protocol has now run on H200. Peptide preflight passed
for PBE/cc-pVTZ, PBE/aug-cc-pVTZ, WB97X/cc-pVTZ and WB97X/aug-cc-pVTZ in both
native and stored-DF strategies. Protein83 completed 16 of 20 strategy/basis
comparisons: SVWN, PBE, M06-L and B3LYP with both bases in both strategies.
The WB97X/aug-cc-pVTZ GPU4PySCF arm failed in the range-separated three-center
integral batching assertion in `df/int3c2e_bdiv.py:199`. The four-arm audit gate
therefore correctly withheld timing repeats for WB97X in **both** bases;
job14076480 exited1 rather than declaring the partial panel complete.

`control_report.py` renders this protocol separately from the historical
unmatched-grid dashboard, using **full_process_wall_s**, exact worker peak RSS,
sampled GPU memory, actual NBF/MO/grid counts and explicit failed/missing arms.
It publishes only derived share-safe data, embedded in the page:

```bash
python3 devtools/benchmarks/gpu_compare/control_report.py \
  --control .pi/gpu_compare/bohr-14076480-evidence \
  --template /home/awallace43/.pi/agent/skills/docs-html/template.html \
  --revision "$(git rev-parse --short HEAD)" \
  --output /home/awallace43/docs/saptdft/cuest/matched-df-2026-10-16.html
```

Stored-DF and native strategies are separate filters; the latter does not imply
equal storage or integral algorithms. This is a shared **cuEST-source** mesh,
not the smaller-native-default experiment. Native storage lengths can differ
because of zero-weight padding; eligibility requires the actual nonzero
coordinate/total-weight multiset and overlap spectrum/rank proofs.
The separately compiled cuEST0.2.50 **four-center direct** adapter and its H100
debug tests are not mixed into these H200 DF timings.

### Future-only strict cuEST 0.2.50 admission

`cuest-sdk-0.2.50.json` independently pins canonical SDK header and CUDA13
library bytes, plus exported four-center direct, DFMO and version capabilities.
It also requires the complete 59-header tree: sorted include-relative POSIX
path, NUL, lowercase file SHA256, newline; aggregate
`3d26a7ef607d02ee294f66f11f1374f3badbd83430f5be28ed0a05e6c8d73d18`.
The separately reconciled full-archive-path JSON digest uses a different
serialization and is not substituted for this tree digest.
`cuest_sdk.py` statically checks those files without loading cuEST. New builds
require an explicit SDK (`lib/13` only, no environment or `lib` fallback), capture
header hashes before and after compilation, and pin the resolved CUDA
user-library closure. A self-consistent old build manifest is not sufficient.

Future Psi4 workers require the FDDS-owned builtin
`psi4.core._cuest_runtime_version()` from the pinned staged extension. The early
check verifies its **presence only**; it must not create/query a handle. Only
after an actual energy/JK route has created a live handle does the worker call
the bridge, requiring exactly `tuple(int, int, int) == (0, 2, 50)`. The bridge
must check all native getter statuses, raise on absent/non-live handles or a
version mismatch, and never initialize implicitly. The vendor getters take
`(cuestHandle_t, uint32_t*)`; no guessed no-argument vendor call, NULL handle,
mangled global handle, or hidden diagnostic handle is permitted.

Each Psi4 worker additionally checks actual mapped cuEST inode/device and file
hash, mapped executable addresses for the required API symbols, and the actual
mapped CUDA user-library closure against the build receipt. The driver is not
a toolkit library and remains part of separate GPU provenance. GPU4PySCF stays
in its separate CUDA12 process and rejects mapped cuEST/CUDA13; SDK paths are
prepended only to the Psi4 subprocess environment. Provenance checks remain
inside `full_process_wall_s`, not silently subtracted from headline timing.

The CUDA12 arm additionally requires an independently reviewed inventory file
and its independent hash in `GPU_COMPARE_CUDA12_RECEIPT` (absolute path) and
`GPU_COMPARE_CUDA12_RECEIPT_SHA256`. Its schema is
`{"schema": 1, "cuda_major": 12, "python_prefix": "...", "libraries": {...}}`;
`libraries` maps each CUDA user-library SONAME to its absolute resolved `path`
and `sha256`. It must include `libcudart.so.12` and every CUDA user-library that
the actual run maps, including cuBLAS/cuSOLVER/cuTENSOR where used. Both
environment identity and ABI families are checked; CUDA12 cudart alone cannot
hide a CUDA13 cuBLAS mapping. Unknown/unpinned mappings or a missing inventory
fail closed. No inventory has been captured/reviewed for this new contract yet;
the guard does not silently generate a self-consistent inventory from whatever
happens to be loaded. Such capture/review is a separate future setup task.
Preliminary device discovery and eigensolver subprocesses get the same explicit
sanitized CUDA12 environment as the workers, and check mapped isolation before
device initialization and again afterward. These changes do not rewrite the
old queued launcher's environment.

These are **local future guards**, not a retroactive certification. The new
native bridge has no reviewed final provider pin/device execution receipt yet;
missing/old providers fail closed. Offline mock tests do not establish native
version execution, DFMO numerical correctness or a completed direct/direct
comparison. The frozen `86feda59ec84e82eb8c90685c2d9d2c8fbc09237` checkout,
existing `2e542f9` build, environments, and queued job14254253 are unchanged.
That old payload pins library bytes but does not implement this independent
exact-version/header/provider admission. Any future run needs its own reviewed
build and authorization; this guard work submits nothing. Scientific settings,
including the RKS absolute `1e-7` overlap cutoff, are unchanged.
