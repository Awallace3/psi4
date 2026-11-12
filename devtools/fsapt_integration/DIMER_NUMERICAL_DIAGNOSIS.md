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
