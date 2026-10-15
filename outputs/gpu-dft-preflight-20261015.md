# Stored/native DF comparison: peptide preflight

Build14071936 succeeded, including a real staged import and CUDA LibXC linkage.
H200 job14071937 failed1:0 after6m32s during peptide preflight. Both allocated-GPU
exporter tests passed; runtime pins remained unchanged.

**No clean timing repeats or Protein83 production ran.** Diagnostic SCFs are
not performance benchmarks and must not be pooled into historical timing tables.

## Actual grids

For successful native-strategy PBE and WB97X diagnostics in both cc-pVTZ and
aug-cc-pVTZ, cross-engine shared quadratures have the exact same nonzero
coordinate/total-weight multiset: **477,977 points**.

- Psi4/cuEST stores543,600 points, including65,623 zero-weight points.
- GPU4PySCF stores543,744 shared points:144 additional zero-weight padding points.
- GPU4PySCF native level3 grid has277,310 nonzero points in278,016 stored points.
- Thus the implemented shared cuEST-source control is **not** both engines on
  the smaller native default. Importing that smaller PySCF quadrature into
  cuEST remains unsupported by this harness.

## Why preflight stopped

1. The stored-DF residency verifier assumed `_cderi` is a dictionary. Installed
   GPU4PySCF `_cholesky_eri` returns the list produced by `multi_gpu.run`, even
   on one GPU. The verifier therefore incorrectly rejected a built tensor before
   SCF. A local API-shaped regression reproduces the exact exception before the
   fix, then passes after accepting per-device sequences. FP64/device-residency
   and no-host-spill checks remain. **105 local tests pass; real GPU acceptance
   of this fix is pending.**
2. Native-strategy SCFs converged and retained dimensions agree, but the
   full-spectrum audit's `atol=1e-9, rtol=1e-10` rejects small overlap differences:

| Basis | NBF / retained NMO, both engines | Max absolute spectrum difference |
|---|---:|---:|
| cc-pVTZ |496 /496|5.36676e-9|
| aug-cc-pVTZ |782 /782|1.22039e-8|

   The largest differences are at eigenvalues approximately7.54 and19.00.
   Minimum eigenvalues are approximately5.3981e-4 and7.0701e-6 respectively,
   both above the prescribed1e-7 cutoff. The origin of the differences remains
   unproven; no tolerance has been loosened merely to make the gate pass.

## Preliminary diagnostic energy differences

GPU4PySCF minus Psi4, total energies, not interaction energies:

| Functional | cc-pVTZ (microhartree) | aug-cc-pVTZ (microhartree) |
|---|---:|---:|
| PBE |-0.000113573|+0.000112323|
| WB97X |+8.97985|+6.22712|

These are single diagnostic solves from the native-DF-strategy cohort on the
shared cuEST mesh. They do not establish stored-DF equivalence or resolve the
remaining overlap audit. Per-worker RSS and sampled process/device GPU memory
were captured, but profile/export overhead makes these unsuitable as clean
benchmark memory comparisons.

The failed immutable run and raw evidence are preserved. Additional paid
recovery jobs require fresh approval; none has been submitted.
