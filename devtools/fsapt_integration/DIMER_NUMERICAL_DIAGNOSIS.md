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
