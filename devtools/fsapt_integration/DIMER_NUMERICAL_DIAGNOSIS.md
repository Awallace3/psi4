# Protein83 dimer-only numerical diagnosis

## Observation, not yet attribution

Original frozen native build: `b641cd4254751bfd42a461d0d0fe1c5647765b58`.
CPU jobs 15474909/15474910 both converged the dimer in 25 iterations:
`-1753.99033624680806` and `-1753.99033624681192` Eh.
H200 job 15474911 failed after 100 iterations. Its final RMS density criterion
was `5.71616e-13` (below D12), but its energy change was `1.57090e-08` Eh
(above E10). The queried native header reported **Native FP64**; mixed
precision was disabled. See `H200_PROTEIN83_TIMINGS.md`.

This demonstrates a convergence discrepancy, not yet fixed-input
nondeterminism or an NVIDIA SDK defect. SCF changes its input every iteration.
The Psi4 adapter, density fitting/conditioning, one-electron terms, reduction
order, and SDK calculations must be distinguished.

## First isolation: one public energy call

`dimer_energy_probe.py` performs only:

```python
psi4.set_options({
    "basis": "jun-cc-pvdz", "df_basis_scf": "aug-cc-pvdz-jkfit",
    "reference": "rhf", "scf_type": "df", "puream": True,
    "use_cuest": backend == "gpu", "cuest_mixed_precision": False,
    "cuest_gemm_min_dim": 0, "ints_tolerance": 1e-16,
    "e_convergence": 10, "d_convergence": 12, "maxiter": 100,
    "guess": "sad", "save_jk": True, "df_ints_io": "save",
})
energy, wfn = psi4.energy("hf", molecule=dimer, return_wfn=True)
```

No SAPT, monomer SCFs, localization, fragment terms, CP correction, Python
source tracing, extra J/K computations, threshold relaxation, or retry.
The geometry and basis bytes are checked against the frozen timing-suite
manifest SHA `988d3b5c6b80b611c6fecc5b6292d5fde51d8f1be89b0a1bb7a017561f4e337a`.
Each allocation runs one fresh process. The original CPU result directories
contain no saved dimer checkpoint; a converged run saves `Cocc`, `D`, `S`,
and `H` for follow-up controls.

Launch helpers: `submit_dimer.py`, `phoenix_dimer.py`. They reuse the original
sealed native build, with separately pinned helper code; no source overlay or
new native kernels. CPU: 48 threads on CPU-only Granite Rapids, 360 GiB
allocation, 1 h cap. GPU: one H200 with 8 host threads, 160 GiB allocation,
30 min cap. Both use 96 GiB Psi4 memory and explicit `embers`.
These are numerical controls, not equal-resource speedup measurements.
The existing full-runtime sanitizer loader limitation remains outstanding.

### Submitted standalone controls

Helper pin: `79d2873a461a6ce4cb0be3aa6a03023546e28994`.
Native science pin remains `b641cd4254751bfd42a461d0d0fe1c5647765b58`.

- CPU: **15482205**, CPU-only `cpu-gnr`, 48 threads, 1 hour, embers.
- GPU: **15482206**, `gpu-h200`, one H200/8 host threads, 30 minutes, embers.
- Account: `gts-cs207-chemx`.
- Run root:
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/dimer-energy-79d2873a46-01`.
- Per-role `manifest.json` and `submissions.json` record launch provenance;
  `job-<id>/energy/{psi4.out,result.json,dimer-wavefunction.npz}` hold scientific
  outputs (the snapshot exists only after convergence).

The initial submitter invocation stopped before creating a run directory or
calling SLURM because system Python lacked NumPy. Reinvoking with the pinned
dependency environment's Python submitted these two jobs. `--test-only`
reported pseudo-IDs 15482203/15482204; those are **not** the submitted jobs.
Submission alone is not evidence of convergence or successful reproduction.

### CPU standalone result: 15482205

Verified completed allocation/batch/step, exit 0, no TERM; owned worker reaped,
no runtime retirement, owned scratch cleanup complete. The scientific result
is `CONVERGED_DIAGNOSTIC_ONLY`, with a native **MemDFJK** header, fitting
condition `1E-10`, and no cuEST header.

- 922 AO functions, **25 iterations**, unchanged E10/D12.
- Final energy **-1753.99033624681078 Eh**, within `3e-12 Eh` of both original
  CPU FSAPT dimer energies.
- Final energy change `4.54747e-13 Eh`, density criterion `7.65071e-13`.
- Diagnostic wall (including snapshot serialization): **64.7032 s**.
  Entire allocation: **86 s**. Neither is an accepted CPU/GPU speedup.
- Saved full-dimer snapshot: **21,479,876 bytes**.
  Producer-reported SHA256:
  `880326ef47784f289111adb88e230a5c4c64c306c8f052373a952d8c73099c8e`.
- Result SHA256:
  `366b4579d7fb00d6db287fb0e15caee33d0c8fd08f01da4214ed65b2d7c70878`.

The isolated CPU calculation reproduces the original CPU convergence.

### H200 standalone result: 15482206

**The same failure reproduces without SAPT.** `psi4.energy("hf")` raised
`SCFConvergenceError` after 100 iterations, with confirmed native cuEST/Native
FP64 headers. Actual hardware: H200/sm90, driver `615.71.09`, node
`atl1-1-02-012-2-0`. Prolog confirms embers. Allocation/batch/step failed with
exit 1; no TERM or runtime retirement, owned workers reaped and scratch
cleanup complete. This is not an OOM, timeout, or loader rejection.

- Iteration 100: energy `-1753.99033625102334 Eh`,
  delta `-3.02521e-09 Eh`, density criterion `6.24252e-13`.
- D12 passes; E10 fails. Late energy changes reach roughly `1.8e-08 Eh`
  while density criteria remain below `1e-12`.
- Failed scientific diagnostic wall: **24.4957 s**; allocation: **43 s**.
  These are failure durations, not a converged GPU timing or speedup.
- Result SHA256:
  `ebeb91d3e3f23b67b0e2294af0d5d2420998ad97914dd7a56e8471a2016dea80`.

SAPT/fragment algorithms are therefore not required to trigger the discrepancy.
SCF still changes density/coefficients each iteration, so this result alone
does not establish nondeterminism at identical SDK inputs or blame the SDK
rather than its Psi4 adapter.
Terminology: the printed `Dnorm` is the orbital-gradient/DIIS residual from
`compute_orbital_gradient`, not a direct norm of the change in density between
iterations. Passing D12 therefore does not prove that the SDK saw identical
density or coefficient arrays.

## Next discriminating controls (not yet run)

`dimer_fixed_jk.py` implements the first control, using the existing native
`scf_repeat_probe.sample_plan`: four computations per plan, three plans in
one process, separately on CPU-only and H200 allocations. It checks the CPU
snapshot hash, exact geometry/bases, occupied orthonormality, and `D = CCᵀ`.
It records actual reconstructed-density differences, J/K matrices, symmetry
defects, and direct difference contractions accumulated in host extended
precision. Completion means the diagnostic executed, not that parity passed.

Submitted fixed-input controls (native build unchanged):

- Helper pin: `1f5d1cfe2e277c5eae14bf3b00bb0c88262a6cdd`.
- CPU-only **15485611**, H200 **15485612**; both embers, with the same
  CPU48/1h and GPU-host8/30min resource limits as the energy controls.
- Run root:
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/dimer-fixed-jk-1f5d1cfe2e-01`.
- Both bind CPU job 15482205's result/snapshot hashes. The shared launcher
  retains the directory name `energy/`, but this mode runs **no SCF**.
- Both fixed-input controls completed; results below.

CPU fixed-input job **15485611** completed (allocation/batch/step exit 0, no
TERM/retirement, owned scratch cleaned). All **12 J and K matrices were
bitwise identical**, including across the three plan constructions; actual
reconstructed D also matched both its first value and saved CPU D exactly.
Symmetry defects and direct difference energy contractions were zero.
Diagnostic wall: 86.861 s; allocation: 110 s.

**Backend caveat:** this reused helper selected **DiskDFJK**, not the original
SCF's MemDFJK. It calls `JK.build(..., jk_type="DF")` without a factory memory
argument, then grants memory with `set_memory` *after* backend selection.
The SCF factory instead passes `memory=` into `JK.build`. This result is a
valid DiskDFJK repeatability control, **not an exact original-CPU-backend
comparison**. Use an explicit factory memory grant and assert `MemDFJK` before
attributing a cross-backend bias to cuEST.

- CPU result SHA256:
  `c4f05dff7ac2b7bf3317bdee50f557274044fe2d45d9b38e775c5378ff7f5d3e`.
- CPU matrix artifact producer SHA256:
  `145cc1a0cc6669560016b45c663866247700e66f9a2486fe0385faa82cacc725`.

H200 fixed-input job **15485612** completed (exit 0, no TERM/retirement,
owned scratch cleaned). Actual H200/sm90, driver `615.71.09`, node
`atl1-1-02-014-2-0`; cuESTJK and Native FP64 headers confirmed.

- **Within each plan:** J is bitwise identical. K varies by at most
  `2.554e-15` elementwise; absolute direct JK energy-contraction change is
  below `9.814e-16 Eh` relative to each plan's first call.
- **Across new plans:** maximum absolute energy offset from plan 0 is
  `2.118511e-10 Eh`, composed of `2.134952e-10 Eh` from J and
  `-1.644160e-12 Eh` from K. Plan 2's offset is `2.503068e-11 Eh`.
  Maximum J/K element changes from plan 0 are `1.717382e-11` and
  `7.651657e-13`, respectively.
- Actual reconstructed D is identical across all calls and exactly matches
  the saved CPU D. J symmetry defect is about `1.7755e-13`; K is symmetric.
- Diagnostic wall 7.2049 s, allocation 37 s; not accepted speedup evidence.
- Result SHA256:
  `0e68b4838c95af1fdd4b3df7c53a9a878584297e7caf16dce06f74af79c45118`.
- Matrix artifact producer SHA256:
  `0932da015e49bbe333da1b9387139d8ff19114abca444f50858af772a729b339`.

**Interpretation:** repeated J/K evaluation at this identical input does not
reproduce the `~1e-8 Eh` SCF fluctuations. The plan-construction variability
is real and exceeds E10 in one comparison, but SCF normally retains its
plan; it does not by itself explain iteration-to-iteration changes.
This control does not rule out sensitivity to changing inputs, different
SCF buffer histories, or other GPU paths.

`USE_CUEST` also dispatches `HF::diagonalize_F` transforms and
`HF::form_FDSmSDF` to cuBLAS in `libscf_solver/hf.cc`. Those files match the
frozen b641 native source. Thus the remaining investigation must not equate
the entire GPU SCF path with the cuEST J/K SDK.

Next prepared control: `trace-energy` snapshots the native `HF.compute_E`
boundary while still invoking exactly one public `energy("hf")`. It preserves
native calls/returns and options, performs no extra J/K evaluations, records
extended-precision energy-component changes, actual D/JK-D and electron-count
changes, and saves selected-iteration C/D/S/H/J/K matrices. CPU tracing uses
the original SCF factory (and must confirm MemDFJK), avoiding the DiskDFJK
control caveat. Traced durations are not timings.

Trace controls submitted with helper pin
`a83b4d5f81994d149cf50e743d7fbef39256104a`:

- CPU-only **15503334**, H200 **15503335**, both embers.
- Run root:
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/dimer-trace-a83b4d5f81-01`.
- Same original native build, CPU48/1h and GPU-host8/30min limits.
- `energy/energy-trace.json` records the observed native energy boundaries;
  selected `iteration-*.npz` snapshots are hashed in that trace.
- 14 host-side helper tests passed before submission.

CPU trace **15503334** completed cleanly (exit 0, no TERM/retirement, cleanup
complete), with **MemDFJK confirmed at all 25 recorded energy boundaries**.
It converged in 25 iterations at `-1753.990336246810 Eh`, consistent with the
untraced CPU result. Traced wall 82.6898 s; allocation 104 s, not performance
measurements.

- Native returned energy minus extended-precision reconstruction:
  maximum absolute `3.33356e-12 Eh` over the trace.
- JK's reconstructed D equals the wavefunction D exactly throughout.
- At iteration 25, electron count is `292.0`, occupied orthonormality defect
  `6.28386e-14`, and `D = Cocc Coccᵀ` exactly at the array level.
- The late CPU energy components do change appreciably while cancelling:
  iteration 25's one-electron/Coulomb/exchange changes are
  `8.93479e-10`, `-1.06996e-9`, `1.75404e-10 Eh`, summing to
  `-1.08000e-12 Eh` using direct difference contractions.
  Its maximum elementwise D change is `4.35021e-8`, reinforcing that a small
  DIIS residual is not a literal small successive-density norm.
- Trace SHA256:
  `3ee3597ca81f7d201829c53ea5efed1d177cd97329b1870d0ed5756b4b3d2023`.
- Result SHA256:
  `d8fdffcacd82f836bfb60255d89033baa713f1ff8cf3254177b456f3d530278a`.
- Selected snapshots at iterations 20 and 25 were written; their hashes are
  in `energy-trace.json`. Native method restoration is confirmed.

H200 trace **15503335** completed cleanly and met the stopping criterion in
**46 iterations**, at `-1753.990336252874 Eh`, about `6.064e-9 Eh` below the
CPU trace result. This is **not evidence that instrumentation fixed the
problem**: the preceding iterations still show the same large energy changes.
Hardware H200/sm90, driver `615.71.09`, node `atl1-1-02-014-9-0`; embers,
cuESTJK and Native FP64 confirmed. No TERM/retirement; cleanup complete.

- At iteration 40, direct energy-component changes (Eh) were:
  - Coulomb: `+6.1651277e-9`
  - one-electron: `-2.0658976e-10`
  - exchange: `-1.5829445e-11`
  - total: `+5.9427085e-9`
- At iteration 44, Coulomb changed by `-1.0461792e-8 Eh`, dominating the
  total change `-9.7597880e-9 Eh`.
- The iteration-46 direct total change was `-4.07121e-11 Eh`: it passes E10
  despite larger fluctuations immediately beforehand. Do not claim a
  deterministic 100-iteration failure rate from these results.
- Maximum native returned energy minus extended-precision reconstruction:
  `3.88056e-12 Eh`. The large fluctuations are present in the matrix energy
  components, not introduced by the final scalar energy contraction/sum.
- JK D equals SCF D exactly throughout. H and S do not change. Electron
  count remains approximately `292.0000000000003–292.0000000000004` in the
  late iterations; occupied orthonormality defects at sampled iterations
  20/25/40 are `6.27e-14`, `4.68e-14`, `5.86e-14`. This provides no evidence
  for loss of orbital normalization as the explanation.
- Selected snapshots at iterations 20, 25, 40 preserve actual evolving
  C/D/S/H/J/K inputs and outputs. No extra JK calls; method restored.
- Trace SHA256:
  `e99ce19d60d93f7b448eae04f7e3497adac096785e806f28e8238d25a8285b84`.
- Result SHA256:
  `5be023f965eba32edec9fe2816fb3d4ec1f95e5c884ceb3bb3a22db63de9c5f6`.
- Traced diagnostic wall 25.0158 s, allocation 48 s; not a speedup result.

### Current inference and remaining discriminant

The symptom is now localized to **predominantly Coulomb energy changes along
the evolving GPU SCF trajectory**. It is not reproduced by repeatedly
evaluating the one tested fixed CPU density, nor explained by final host
energy summation, mismatched SCF/JK density arrays, or observed normalization
defects. This does not yet establish whether the error originates in the
adapter, DF conditioning/inversion, input-dependent finite precision in the
cuEST Coulomb calculation, or the SCF trajectory.

The next decisive experiment is to replay the **same saved GPU-trajectory
Cocc/D snapshots** through cuEST and explicitly selected **MemDFJK**, with a
fixed plan and matching numerical policy. Compare J/K and direct Coulomb
energy differences across snapshots; test Coulomb linearity/self-adjointness
if the mismatch follows J. Do not substitute the existing DiskDFJK control
for that exact-backend comparison. No fix or SDK-only attribution is claimed.

### Requested replay and SAPT interpretation

The user authorized the trajectory replay. `dimer_trace_replay.py` uses GPU
trace snapshots **20, 25, 40**, unchanged coefficient bytes in both arms,
and two new plans: forward order repeated twice, then reverse order repeated
twice. This tests input-history and plan sensitivity without SCF iteration.
CPU factory selection receives an explicit **72 GiB grant before building**
and must return `MemDFJK`; GPU must return `cuESTJK`. Snapshot hashes and
geometry/basis identities are checked; actual reconstructed densities,
matrix deltas versus original trace, same-input repeats, direct energy
contractions, and Coulomb reciprocity defects are recorded. No native build
changes or source overlays.

Replay submitted with helper pin `aabb1266b6043a98c8bcd502998d4faec8555b81`:
CPU-only **15521564** (MemDFJK required) and H200 **15521565** (cuESTJK
required), both `gts-cs207-chemx`/embers. Run root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/dimer-replay-aabb1266b6-01`.
CPU48/1h and GPU-host8/30min limits; no SCF or SAPT calls. All 18 host helper
tests passed. Both manifests bind the GPU trace/result hashes above.
CPU **15521564** failed before any JK computation: the intended `memory=`
argument was ignored by `_core_jk_build` because its overload dispatch
requires **both `do_wK` and `memory`**. The explicit MemDFJK guard rejected
DiskDFJK, so no CPU replay numbers were accepted. This is a helper defect,
not scientific evidence. The regression now executes the actual Python
factory wrapper extracted from `python_helpers.py`; it reproduced the wrong
two-argument native call before adding `do_wK=False`. A CPU-only resubmission
will use a new helper pin/run directory. GPU **15521565** is left untouched:
the old factory selected cuESTJK and the helper sets its memory before
initialization; no CPU-backend substitution can occur in that arm.

Corrected CPU-only replay **15521609** submitted on `cpu-gnr`/embers,
48 threads/1h, helper pin `9904dfcc89fd6aa842b14480941baa5aa09e77a8`.
New run root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/dimer-replay-9904dfcc89-cpu-01`.
The real-wrapper regression failed before the fix and all 19 host tests
passed afterward. Original GPU **15521565** retains its frozen helper/run;
the relevant helper difference is the explicit factory `do_wK=False` needed
for CPU MemDF selection, not a change to native numerical code or inputs.

Corrected CPU replay **15521609** completed cleanly (exit 0, no TERM or
retirement, scratch cleaned), with **MemDFJK** in all 12 calls and the
72 GiB factory grant confirmed. Every reconstructed density matches its
original GPU snapshot exactly. Same-input energy-contraction repeat drift,
including reversed input order and a rebuilt plan, is at most
`4.40972e-13 Eh`; not all matrices are bitwise identical.

For plan 0's first pass, **CPU replay minus original GPU trace**, at identical
Cocc/D:

| GPU iteration | Coulomb correction (Eh) | Exchange correction (Eh) | Total JK correction (Eh) |
|---|---:|---:|---:|
| 20 | +1.3617492254e-8 | -5.0593756212e-10 | +1.3111554692e-8 |
| 25 | -1.1757401329e-8 | -5.0600831633e-10 | -1.2263409645e-8 |
| 40 | +5.7176903362e-10 | -5.0594817431e-10 | +6.5820859305e-11 |

J matrix max differences are about `6.77–7.54e-10`; K differences about
`4.21e-11`. The CPU Coulomb reciprocity defect is at most `1.390e-12 Eh`
over the selected pairs/plans.

Adding each direct JK correction to the original GPU trace's reconstructed
total, while retaining that snapshot's H and nuclear term, gives:

| GPU iteration | Energy with CPU-replayed JK (Eh, approximate) |
|---|---:|
| 20 | -1753.99033624681025 |
| 25 | -1753.99033624681311 |
| 40 | -1753.99033624681178 |

Their span is approximately **2.864e-12 Eh** (derived from stored scalar
summaries, not a new SCF). Thus the large original fluctuations disappear
when the *same GPU trajectory states* are evaluated with CPU J/K. The
variable discrepancy is predominantly Coulomb, while the exchange offset
is nearly constant. This is stronger evidence than comparing independently
converged SCFs; the fresh cuEST replay remains necessary to separate
input-dependent error from original-plan/history effects.

- CPU replay result SHA256:
  `c232263784a213e7d1483a8ac03deab4fa44b0b98ae289112f10a17ea5b5b780`.
- Matrix artifact producer SHA256:
  `9980e82da3600b110f01f4daf46684bdc0c83f079a8230fc7f3b9e4c7cd5ce5f`.
- Diagnostic wall 53.898 s, allocation 67 s; not speedup evidence.

Fresh cuEST replay **15521565** also completed cleanly, with Native FP64,
H200/sm90 and driver `615.71.09` confirmed on `atl1-1-02-012-2-0`.
All reconstructed densities exactly match the GPU snapshots.

- Same-input repeat drift, after interleaving other snapshots:
  at most **1.81890e-15 Eh**.
- Across the two freshly constructed plans:
  at most **7.08045e-11 Eh**.
- Coulomb reciprocity defect:
  at most **1.46549e-13 Eh**.
- Result SHA256:
  `b903cb2fc35e4d51ced16c2a3e6ed533bdd7abcffccc2d892ed1797d0f7c7bca`.
- Matrix artifact producer SHA256:
  `7c27a217d28b56f5f15bb12c5b465d7d8d94f989dfaf5a1794ff95845ca46b3c`.
- Diagnostic wall 7.1370 s, allocation 26 s; not speedup evidence.

Using the two arms' direct corrections to the *same original GPU snapshots*,
plan 0 yields:

| GPU iteration | Fresh cuEST minus MemDFJK JK energy (Eh) | Total with fresh cuEST JK (Eh, approximate) |
|---|---:|---:|
| 20 | -2.0271873684e-9 | -1753.99033624883743 |
| 25 | -2.0246385072e-9 | -1753.99033624883775 |
| 40 | -2.0259024521e-9 | -1753.99033624883768 |

Fresh cuEST's corrected totals span only **3.155e-13 Eh** in these stored
summaries. Its offset from MemDFJK is nearly constant (span **2.549e-12 Eh**).
**Therefore fresh cuEST replay does not reproduce the noisy SCF response.**
This revises the earlier inference: the large variable error is associated
with the original SCF plan/configuration or execution context, not merely
with these densities passed to an arbitrary fresh cuEST plan. The original
SCF snapshots remain discrepant, and standalone SCF reproduces the symptom,
so SAPT is not required. Compare SCF versus replay plan settings/state before
claiming a general SDK Coulomb accuracy defect.

Log comparison found a concrete configuration clue: the original SCF reports
**10.265 GiB DF-plan storage**, versus **10.096 GiB** for both fresh replay
plans. Both print MATRIXPOWER/relative `1e-10` and pair threshold `1e-16`.
Do not yet assume these are otherwise identical plans.

Header-order caveat: the replay helper calls `print_header()` *before*
`initialize()`. Its first header reports the handle's initial default math
mode; the second reports Native FP64. Source inspection confirms
`preiterations()` calls `configure_math_mode()` before plan construction and
`compute_JK()` reapplies it before every computation, with the recorded
`CUEST_MIXED_PRECISION=False`. Thus the first pre-initialization header is not
evidence of mixed-precision evaluation. Future helpers should print/query
the header after initialization to avoid this ambiguity.

The observed 46-iteration convergence does not establish a SAPT effect:
standalone untraced HF already reproduced the 100-iteration failure, and
the traced run met a **successive-energy-change** criterion while still
showing large fluctuations. The runs used different plan initializations,
nodes, and execution schedules, so tracing has not been causally isolated.
SAPT startup/reuse effects are possible secondary influences, not necessary
conditions for this symptom. Preserve both failed and converged outcomes;
do not retry until a favorable stopping event is obtained.

### Initialization review: remaining discriminating measurements

Initial source review inferred that JK defaults `do_wK=False`, `omega=0`,
`alpha=1`, and `beta=0` matched HF. **That inference was wrong for beta**:
the subsequent live peptide SCF receipt below records **beta=1**, assigned
from the functional by SCF initialization. Earlier replay explicitly set
beta=0. A direct unused-beta A/B is now required; do not treat the earlier
SCF/fresh-replay comparisons as configuration-matched.

The next comparison must capture the **actual original SCF auxiliary basis**,
not only its requested name: function/shell counts, ordered centers and shells,
angular momenta/purity, exponents, normalized coefficients, and exact binary
coordinates. Both paths use the same `cuESTJK::build_cuest_basis` conversion;
the original SCF auxiliary fingerprint was not saved. Also record unrounded
workspace bytes and effective DF/pair parameters after initialization.
The workspace query does not take the host JK memory grant, so that grant
alone does not explain the observed DF-plan storage difference.

An additional execution-context hypothesis is the shared CUDA library state:
`cuest_runtime.cc` supplies cuEST with cuBLAS/cuSOLVER handles and a stream
also used by other Psi4 GPU operations. SCF exercises those operations whereas
the standalone replay does not. Reapplying the cuEST math mode does not by
itself establish that every shared cuBLAS setting agrees. If basis and plan
parameters match, capture cuBLAS math/atomics/pointer modes and stream before
plan construction and Coulomb evaluation, together with plan-creation counts.
**No handle-state corruption has been demonstrated.** These are proposed
measurements, not completed controls or a root-cause attribution.

### SAPT entrypoint versus standalone: source audit

Both routes reach the common `scf_helper` and native RHF iteration machinery.
The SAPT dimer runs before monomer SCFs and before fragment calculations;
`SAVE_JK` retains its newly constructed JK, rather than supplying a reused
monomer JK to the dimer. The protein83 benchmark launches `fragment_gate.py`
through the sealed bootstrap, not the Python-tracing `timing_worker.py`.

Real preamble differences remain:

- SAPT prepares C1/fixed-frame dimer and ghosted monomer objects before SCF.
- Standalone explicitly sets SAD/MAXITER=100; the SAPT worker leaves these
  to defaults/resolution. Global/local option values and changed flags need
  comparison at the actual SCF boundary.
- The fragment worker sets the Einsums GPU-disable policy before importing
  Psi4 and executes `batched_probe(runtime)` GPU controls before `energy()`.
  The standalone scientific probe does not perform those primitive controls.
  This does not establish shared-handle corruption or causality.
- The energy observer performs matrix copies, contractions, allocations and
  file writes between iterations. Preserving the returned energy and adding
  zero JK calls does not make it an unperturbed execution-context repeat.

Local `git diff b641cd4254 HEAD` confirmed unchanged `proc.py`, `proc_util.py`,
`sapt_proc.py`, `scf_iterator.py`, `gpu_runtime.py` and `protein83_timing.py`
for this audit. `fragment_gate.py` has subsequent additional optimization
controls; its diff confirms the pre-energy primitive probe and entrypoint
structure already existed in the frozen benchmark source.

The DF storage discrepancy above is **SCF versus fresh JK replay**, not an
established **SAPT versus standalone SCF** difference. Neither that discrepancy
nor one converged traced run establishes an entrypoint-specific bug.
Compare historical iteration tables first; subsequent controlled repeats
should separate entrypoint, primitive-probe preamble, and trace instrumentation
while keeping scientific options, native runtime and hardware fixed.

### Historical trajectory comparison: measured, not inferred from stopping

Parsed the first RHF iteration block (including SAD as iteration 0) from the
six original `psi4.out` files; stop at the next iteration reset to exclude
monomer SCFs. Decimal arithmetic on printed energies avoids introducing
additional subtraction rounding. Raw extracted rows and comparisons:
`.pi/tasks/01a1c82b-a0ba-7771-b262-9bef9e925c8f-3373270/b4d5c1f60.output`.
These comparisons use finite-precision printed energies/residuals, not full
wavefunction equality.

The three original GPU SCF headers were also compared directly
(`b30026ceb.output` in the same task directory). SAPT, standalone and traced
standalone all report identical:

- 444 primary shells, 922 spherical / 961 Cartesian functions and the same
  element-to-basis-file assignments, including aug-cc-pVDZ-JKFIT;
- SAD guess, enabled DIIS, E10/D12 and integral threshold `1e-16`;
- Native FP64, MATRIXPOWER relative cutoff `1e-10`, pair threshold `1e-16`;
- DF plan **10.265 GiB**, explicit JK plans **10.269 GiB**, J/K scratch
  **7.449 GiB**, and free/total device memory **128.877/139.844 GiB**;
- overlap reciprocal condition number **5.7063048214e-8**.

Host JK grants differ slightly: SAPT **89.475 GiB**, standalone **89.578 GiB**,
traced standalone **89.579 GiB**, reflecting different resident host memory.
The reported device plan sizes do not differ. This strengthens the distinction:
the smaller **10.096 GiB** plan belongs to fresh fixed-input JK replay, not
to a different standalone SCF plan. Formatted headers still do not substitute
for exact basis/plan/state fingerprints.

| Comparison | Common iterations | Maximum absolute energy difference (Eh) |
|---|---:|---:|
| CPU SAPT 15474909 vs standalone 15482205 | SAD–25 | 8.64e-12 |
| CPU SAPT repeat 15474909 vs 15474910 | SAD–25 | 9.78e-12 |
| GPU SAPT 15474911 vs standalone 15482206 | SAD–46 | 1.936996e-8 |
| GPU standalone 15482206 vs traced standalone 15503335 | SAD–46 | 2.393017e-8 |
| GPU SAPT 15474911 vs traced standalone 15503335 | SAD–46 | 1.720400e-8 |

GPU SAPT versus untraced standalone differs by `2.20416e-9 Eh` already at
SAD and `1.061040e-8 Eh` at iteration 1; its maximum through iteration 100
is `2.049274e-8 Eh`. Untraced versus traced standalone differs by
`2.9149e-10 Eh` at SAD and `-1.508534e-8 Eh` at iteration 1.
The observer skips SAD and first copies matrices only after the native
iteration-1 energy is computed: later per-iteration copying/I/O cannot alone
explain a discrepancy already present at these first energies.

**All six runs first cross the printed orbital-gradient threshold `1e-12`
at iteration 25.** The GPU residual trajectories closely track each other:

| Iteration | SAPT GPU | Standalone GPU | Traced standalone GPU |
|---|---:|---:|---:|
| 1 | 1.54790e-3 | 1.54790e-3 | 1.54790e-3 |
| 10 | 1.32576e-7 | 1.32576e-7 | 1.32576e-7 |
| 20 | 2.81102e-11 | 2.80881e-11 | 2.80560e-11 |
| 25 | 7.87925e-13 | 8.46628e-13 | 8.65695e-13 |

Thus the available logs do not show grossly different SCF descent or a
SAPT-only failure. They show near-identical residual descent with small
absolute, but E10-significant, energy differences already early in the GPU
runs. Standalone-versus-standalone differences are comparable to entrypoint
differences. Different nodes, fresh plans, preambles and tracing remain
confounded; controlled repeats are still needed to assign a cause. These
data do not invalidate the 46-iteration endpoint merely because it followed
fluctuations.

Source-log SHA256 values:

- SAPT GPU: `7a3e568fd3bb52fc7db714c2f4f2efa8ef5273510da95afb1b9f0d845abcd603`
- SAPT CPU 0: `4a181e97c4087f05effa4fde0d4eab2cdf976c26b344ca67acefb8ad35b3adef`
- SAPT CPU 1: `b81219db498294b09d84bd8a257f903c5085eb8be4d81f3322f69649c163548d`
- Standalone GPU: `0ed65c5c56e58148c4b0c7b818e77714cfc76a6896f4fc4fcb111709547a9a8c`
- Standalone CPU: `7d5601d64e852ac0cc07e05f3585440b40eb0ac2bf996b0a9b092987c07a7d5b`
- Traced GPU: `ee855e531dbe3f4141826cd6c78e71f447afd5a543ad812980b97ad441cfd9ac`

### Small-system cachy screen

User approved small GPU tests on cachy. `small_dimer_probe.py` and
`cachy_dimer_screen.py` implement a bounded 16-process screen: HCl dimer and
timing-suite peptide × CPU/cuEST × public HF/SAPT dimer entrypoints × two
fresh-process repetitions. Entry order reverses on the second repetition.
The SAPT arm runs its real preamble, intercepts the first `scf_helper` return,
and stops before any monomer/fragment work. Both arms have matched explicit
SAD/MAXITER=100, E10/D12, FP64 and basis settings; neither has primitive GPU
warmups or an iteration observer. This tests the entrypoint, not the original
benchmark's complete GPU preamble.

The preserved `install-owned-ba9251b785` installation is read-only. Its SCF
driver/native sources match the b641 Phoenix baseline in the inspected
`libscf_solver`, `cuESTJK.cc`, `scf_proc`, and `proc.py` paths. This is still a
different build, architecture and driver, so a negative result is not an H200
clearance. Existing qualification-helper hashes were unchanged at preflight.
The full authenticated runtime checks must also pass at launch.

Dedicated root: `~/gits/psi4-small-dimer-cachy`, with runs beneath the new
helper worktree's `source/tmp`. No edits/rebuilds to the manual workspace or
shared environment. The launcher holds the common GPU lease, honors HOLD,
enforces four CPUs/12 GiB/no swap/128 tasks via an outer cgroup, and bounds
each owned process and the whole campaign. SCF nonconvergence is retained as
a diagnostic outcome; infrastructure errors stop the campaign.

Exact primary/auxiliary shell fingerprints and terminal C/D/S/H are recorded
after SCF. Terminal arrays are not claimed to be a synchronized energy-boundary
snapshot. A subsequent fixed-input replay must reconstruct D from the saved
occupied orbitals and distinguish it from the terminal D.

Completed screen: `~/gits/psi4-small-dimer-cachy/source/tmp/screen-04`.
Helper pin `1e9723549a3b382c69d89598386c0e37c0fd0d71`, preserved science pin
`ba9251b785043df16dfa60acf64a7ddcbfca0ea9`. All 16 workers converged, with
CPU MemDFJK / GPU Native FP64 cuEST execution confirmed. Master result SHA256:
`9e20460302ed487d8ff7a0d2ddcd17f532e2fe2100f107377077f48453150e4e`.
Analysis verified each worker-result/log hash and matching helper/science/config
pins. Host contract suite: 15 passed. No fragment provider was executed.

Attempts 01–03 remain preserved: 01 rejected nonprivate alias-directory
permissions; 02 rejected a workspace check that incorrectly depended on
sandboxed HOME; 03 completed four HCl CPU cases and one HCl GPU case, then
rejected the GPU SAPT preamble's missing audited provider before its SCF.
Those are harness failures, not SCF failures. The corrected screen supplies
the real provider without invoking it, matches the full relevant SAPT option
set, and preserves original errors rather than masking them with missing-log
parsing errors.

| System / arm | Standalone iterations (two processes) | SAPT dimer iterations (two processes) |
|---|---:|---:|
| HCl CPU | 15, 15 | 15, 15 |
| HCl GPU | 15, 15 | 15, 15 |
| Peptide CPU | 22, 22 | 22, 22 |
| Peptide GPU | 26, 25 | 23, 22 |

At aligned iterations:

| Comparison | HCl max absolute energy difference (Eh) | Peptide max absolute energy difference (Eh) |
|---|---:|---:|
| CPU standalone repeat | 1.82e-12 | 1.30e-12 |
| CPU SAPT repeat | 1.26e-12 | 1.08e-12 |
| GPU standalone repeat | 1.808e-11 | 6.1470e-10 |
| GPU SAPT repeat | 1.875e-11 | 5.2200e-10 |
| GPU standalone vs SAPT, first repeats | 1.922e-11 | 5.3484e-10 |

Peptide GPU terminal energies are about **1.12–1.22e-9 Eh above CPU**.
At iteration 22 all four peptide GPU residuals are approximately `6e-13`;
some energy changes still exceed E10, resulting in different stopping
iterations. This is a small-system numerical symptom, not proof of false
convergence or a common root cause with protein83.

Primary and auxiliary basis fingerprints (including exact coordinates and
normalized coefficients) are identical across all eight arms/repeats for
each system:

- HCl: 54 AO / 320 auxiliary; primary
  `37a84c7b3921d967314e92c747f0efde1b488d53fae66632b2aa10c22cf171e2`,
  auxiliary `89f6b42a9f37fed2adb575a51e89f72af999739dc31b8c8a30f801a73795acd6`.
- Peptide: 250 AO / 1308 auxiliary; primary
  `f0c7d919f9ed1cfee34bdc9bff753ee7e1fa21f62f3f1c16100b12ebc8bc16b7`,
  auxiliary `6baa176244fdddf40804b36b915273465f881f8fe97da176a62fe6799f7a55a5`.

The peptide is therefore a useful smaller diagnostic case. Its GPU
repeat-to-repeat differences are comparable to entrypoint differences, while
CPU controls agree much more closely. Next isolate the SCF-owned versus fresh
JK plan at identical occupied-orbital bytes, using these actual SCF basis
objects; do not infer an SDK defect from free-running SCF trajectories alone.

### Retained peptide plan: live beta mismatch identified

The GPU producer in `source/tmp/plans-02/retained` completed its SCF and
16 post-SCF fixed-orbital JK calls. The campaign subsequently stopped in
the independent CPU process because a host `C @ C.T` reconstruction was
incorrectly required to be bitwise identical. Preserve the producer; resume
only CPU/GPU replay, recording host roundoff rather than relaxing energy
criteria. No native code was changed.

Producer result SHA256:
`3bbf56aa5734541d050c6e76610377b5ef282d71920cca344442b791e220fcbc`.
Fixed-input SHA256:
`32812f827acb8bd27fd7bca9386aea97af8b5864efef586322e2c05e51942c5b`.

- Actual SCF JK settings, both after initialization and after energy return:
  **alpha=1, beta=1, omega=0**, wcombine=false, wK disabled in the native
  header; memory=694521315 doubles.
- Effective/global options and changed flags matched across those two
  boundaries, with module SCF, CSAM screening, fitting cutoff1e-10,
  integral tolerance1e-16, Native FP64, slices5/moduli8, no DirectJ/GridK.
- No retained-plan recreation was logged during the comparisons.
- Retained and fresh same-basis-object plans all reported **0.316 GiB**.
- Identical-C repeated calls showed Coulomb-dominated variation: retained
  first block max `8.73252e-11 Eh` from its first call; fresh plan0 max
  `2.54507e-10 Eh`; K contraction differences were only about `1e-16 Eh`
  within a plan. Actual JK D versus the common host D differed by at most
  `5.55112e-17` in every recorded call.
- Rebuilt final-orbital commutator RMS was approximately `1.1–1.24e-13`.
  The final orbitals' D differs from the last evaluated SCF D by
  `1.26800e-11`; those are saved separately, not conflated.

The old protein83 replay's explicit beta=0 is therefore a concrete
configuration difference to test, despite omega=0 and wK disabled.
Independent GPU replay now interleaves actual-beta/zero-beta/actual-beta/
zero-beta plans on the exact saved C bytes, changing no other plan setter.
This is an investigation, not an established SDK diagnosis or production fix.

### Controlled zero-omega beta replay

Completed continuation: `source/tmp/plans-02-replay-01`, under the dedicated
cachy root above. Helper pin `6ab32e92cc847cfe7d8107e4fe73e8eb2d44a932`;
science install unchanged. Master result SHA256:
`5c19e7caf776f45ff9c7e41c241e4a113457c715257c08127470c5b423c11cee`.
The CPU and GPU workers use the completed retained producer's identical
occupied-orbital bytes and exact primary/auxiliary basis fingerprints.
GPU plans interleave beta1/beta0/beta1/beta0, with four calls per plan.
All other exposed plan setters are held fixed: alpha1, omega0, wK disabled,
Native FP64. This is a diagnostic, not a production change or timing result.

| Plan | Max absolute repeated JK energy change from its first call (Eh) | J bitwise stable within plan |
|---|---:|---|
| CPU MemDFJK, plan0 | 1.40360e-14 | Not used as a bitwise gate |
| CPU MemDFJK, plan1 | 3.56246e-14 | Not used as a bitwise gate |
| GPU beta1, plan0 | 1.89171e-10 | No |
| GPU beta0, plan1 | 1.61903e-16 | Yes |
| GPU beta1, plan2 | 2.74322e-10 | No |
| GPU beta0, plan3 | 2.15603e-16 | Yes |

The reported variation is the direct contraction
`2 Tr(D delta_J) - Tr(D delta_K)` on the common saved D, rather than
subtracting two large total energies. The beta1 variation is Coulomb
dominated. Beta0 removes the observed within-plan J jitter, but **does not
establish CPU parity**: its stable offset is approximately
**1.17848–1.18404e-9 Eh above CPU**, with a roughly 5.55e-12 difference
between its two freshly constructed plans. Native plan memory changes from
0.316 GiB to 0.303 GiB.

At this beta1-SCF endpoint, the independently reconstructed commutator RMS
is about 1.1–1.2e-13 for beta1, 4.51–4.54e-12 for beta0, and 4.91e-12 for
CPU. Therefore the next control is **full SCF with beta chosen before native
initialization**, allowing each variant to relax its own density. It must
not be inferred from a fixed endpoint that the convergence issue is solved.
The diagnostic leaves the production adapter unchanged.

CPU result/matrix SHA256:
`f769c1873521cd827228591fef120dcfa5cc5b638082b8e2127b829ca1d552d6` /
`b783e5e608c65e8da7a2e1f97207986d474a0d90cc975d5abbfe66ef5f37e257`.
GPU result/matrix SHA256:
`f634b269161a3a2cdfab5f9e432e2f9b3bd5f1afd054315bac0938209b918714` /
`5975b6efe5422d33dbb451bdce37ac204aa4f72b0e7f395e1f59d291c19baa24`.
Hash-verified per-call native D arrays are **bitwise identical across all
24 CPU/GPU calls**, including both beta settings and fresh plans. Their
maximum within-role and cross-role difference is exactly zero. Each native
D differs from the common host-constructed saved D by at most 5.55112e-17;
that fixed roundoff difference does not explain the changing J outputs.

### Full standalone SCF beta control

Completed `source/tmp/beta-scf-01`, helper
`47e8a3c834d11d3548bd92e84173f9daafac124f`, same preserved science install.
Master SHA256:
`9b0c681f063020df6432802e377c7c4d10c74bccba19d28327c8427216fc40b6`.
Four fresh processes ran in beta1/beta0/beta0/beta1 order, unchanged E10/D12.
The diagnostic intercepts JK initialization, verifies alpha1/omega0, and sets
only beta immediately before native initialization. There is no per-iteration
observer. Actual memory grants (automatically estimated by SCF) are recorded;
the earlier fixed-input replay, unlike this fresh-process SCF comparison,
held the exact JK memory grant fixed.

| Beta | Iterations, two runs | Max aligned repeat energy difference (Eh) | Max absolute SCF energy change at iterations >=15 (Eh) |
|---|---|---:|---:|
| 1 | 26, 23 | 6.46651e-10 | 3.75849e-10 |
| 0 | 22, 22 | 2.33058e-12 | 1.36424e-12 |

All four legitimately met the original convergence criteria. Beta0 matches
the earlier CPU iteration count, and its final energies are
`-494.07407982240863` and `-494.07407982240755` Eh. Its independently rebuilt
last-evaluated commutator RMS is about 5.94e-13; final-orbital RMS is about
2.54e-13. Repeated post-SCF J is bitwise stable for beta0 but not beta1,
including fresh plans and the revisited original plan. No retained plan
recreation was logged. **The roughly 1.18e-9 Eh offset above CPU remains.**

Result SHA256s (run order):
`a5df8d84a038b2a234167ea9a4ed0e1ecaa0d56e1f5031a4be2515667f3d09aa`,
`c05fb1bc3d8b56b5ad13698dcf2a075c19a41311e389d4c8c3348daa7e3eb547`,
`43e6fbab423d9b768aa4213882017393cd5007157696e3d0a5ebf76c2e25d1e5`,
`3654b2282b559d67b5c0bf3f556195b918228d30c42ce27a76f169426cc71d60`.

These observations support a zero-omega plan-selection issue, not a
SAPT-only failure. Matched standalone/SAPT-dimer beta controls are the next
test, followed by H200/protein83 validation before claiming the large-system
failure is resolved. No native production fix or sanitizer pass is claimed.

### Further attribution and vendor report requirements

1. If standalone SCF reproduces the discrepancy, use identical saved dimer
   coefficient bytes for repeated native J/K calls within a plan, across
   rebuilt plans, and across fresh processes.
2. Compute CPU J/K on a CPU-only allocation with those exact coefficients.
   Report matrix max/RMS differences and closed-shell energy contractions
   `2 Tr(D ΔJ) - Tr(D ΔK)`, separately for J and K.
3. Compare saved density, overlap and core Hamiltonian across backends.
   Record conditioning policy and actual native math mode, not only requested
   options. Separate deterministic bias from repeat variation.
4. If necessary, reduce to a direct SDK reproducer to distinguish adapter
   errors from SDK behavior.

A vendor package should contain geometry/basis identities, a minimal script,
exact library/driver/build provenance, complete SCF logs, and fixed-input
measurements. Do not report an SDK defect as established before those controls,
or send any package externally without the user's direction.
