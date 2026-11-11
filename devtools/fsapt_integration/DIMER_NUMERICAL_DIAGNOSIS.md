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
