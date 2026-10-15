# GPU DFT comparability audit — 2026-10-14

## Conclusion

The orbital basis names, spherical representation, geometry hashes, AO counts,
auxiliary basis names/counts and functional definitions match. This does **not**
establish numerically identical calculations. Historical quadratures are not
verified equal, and an additional, measured orbital-space mismatch exists for
Protein83/aug-cc-pVTZ.

Both engines begin with 2806 AOs and 3735 auxiliary functions. Psi4 logs
`Overall, 1 of 2806 possible MOs eliminated.` GPU4PySCF logs
`17 small eigenvectors of overlap matrix removed`. Thus the retained dimensions
are 2805 versus 2789. This appears for all four successful augmented Protein83
functional pairs, not just PBE.

Installed GPU4PySCF `scf/hf.py:40–41,637–653,764–773` uses an absolute overlap
eigenvalue cutoff of 1e-6 by default. Psi4's source default S_TOLERANCE is 1e-7;
the archived log's minimum overlap eigenvalue is 6.0317358456e-8.
Protein157/GPU4PySCF also reports eliminating 88 directions.
The historical harness did not explicitly align these cutoffs. The 1e-12
DF_FITTING_CONDITION/LINEAR_DEP_THR settings concern the **auxiliary fitting
metric**, not orbital-overlap truncation.

More aggressive orbital truncation is a plausible contributor to the consistently
higher augmented GPU4PySCF energies; it has not been isolated experimentally.
Grid differences, fitting algorithms, XC implementation/version and distinct
SCF convergence norms remain additional contributors. Do not attribute the
entire energy discrepancy to any one of them.

## Recorded basis and grid checks

- GPU4PySCF uses `gto.M(...basis=config["basis"], cart=False, charge=0, spin=0)`.
  Psi4 uses the same basis name, PUREAM true, charge/multiplicity 0/1, and disables
  reorientation/translation. Geometry SHA256 values agree in paired workers.
- Protein83 counts: cc-pVDZ 766 AOs; aug-cc-pVTZ 2806 AOs; Naux 3735 in both.
- The requested new cc-pVTZ control expects 1786 AOs, not 766.
- Both use def2-universal-jkfit and density fitting; hybrid exact-exchange
  fractions/range parameters and explicit LibXC components are checked.
- Historical nominal grid settings were 99 radial ×590 angular, unpruned,
  GPU4PySCF small_rho_cutoff=0. Its stored Protein83 mesh has 4,848,128 points.
  Psi4's historical actual cuEST count remains unknown; host-grid zeros are
  placeholders, not measurements.
- Equal AO counts do not prove equal retained spaces or identical primitive/
  contraction data; no cross-engine normalized basis/overlap-matrix fingerprint
  was archived.
- The new queued campaign exports actual cuEST coordinates/total weights,
  audits both native defaults for both bases and verifies four shared-grid arms
  before accepting timing repeats. It **does not yet align overlap truncation**.
  Those results remain diagnostic rather than fully apples-to-apples numerical
  certification until retained orbital spaces are also aligned and checked.

## Protein83 total-energy differences

Delta = E(GPU4PySCF) − E(Psi4), microhartree (1 microEh = 1e-6 Eh).
cc-pVDZ uses medians of three successful unprofiled repeats per engine.
Augmented results are single successful preflights, not production medians.

| Functional | cc-pVDZ / microEh | aug-cc-pVTZ / microEh | Augmented / kcal mol^-1 |
|---|---:|---:|---:|
| SVWN | -2.86972 | +73.63041 | +0.046204 |
| PBE | -2.69266 | +113.66076 | +0.071323 |
| M06-L | -4.51657 | +288.61780 | +0.181110 |
| B3LYP | -1.45765 | +100.25549 | +0.062911 |
| wB97X | -30.01083 | unavailable | unavailable |

The maximum absolute cc-pVDZ discrepancy across all 15 successful system/
functional pairs is 30.01083 microEh (0.018832 kcal/mol).
Peptide augmented PBE differs by only +0.26409 microEh.
Augmented wB97X/Protein83 failed in GPU4PySCF's DF batching, so it has no paired
energy. Protein157/PBE failed in Psi4 GPU allocation, so it has no paired energy.
These are **total energies**, not interaction energies; do not assume cancellation.

## Why GPU memory differs

The large GPU4PySCF advantage is specific to pure functionals, not universal:

| Protein83/aug-cc-pVTZ | Psi4 GPU / GiB | GPU4PySCF GPU / GiB |
|---|---:|---:|
| PBE | 75.415 | 4.612 |
| B3LYP | 75.310 | 101.931 |

These are historical 1-second **whole-device observed peaks**, not exact
per-process allocator high-water marks. Profiles are excluded.

Installed GPU4PySCF `dft/rks.py:119–125` obtains J and requests K only for hybrids.
`df/df.py:110–117` selects the J-only engine when K is unnecessary and CDERI
has not been built. `df/df_jk.py:582–618` calls
`build(build_cderi=False)`, contracts the density into an auxiliary vector,
solves the auxiliary metric system, and contracts that vector back to J.
It avoids retaining the full auxiliary-by-AO-pair CDERI tensor for pure DFT.
Setting `only_dfj=False` allows density-fitted K where needed; it does not force
construction of an unnecessary exchange tensor for a pure functional.

Psi4's cuEST integration instead queries/allocates a persistent DF integral
plan in `cuESTJK::preiterations`. Actual archived Protein83 logs report
56.678 GiB of DF plan device storage for SVWN/PBE/M06-L and 56.574 GiB for
B3LYP. The opaque SDK's precise internal arrays are not exposed, but the
large persistent allocation is measured, not inferred from a grid estimate.
Protein157/PBE fails on a 364.065 GiB allocation request; GPU4PySCF's J-only
route converges with an observed 8.130 GiB whole-device peak.

Removing 17 instead of 1 orbital cannot by itself explain an approximately
16-fold GPU-memory difference. Direct J versus persistent DF-plan storage
is the primary source-supported algorithmic distinction.

CPU RSS is separate: augmented Protein83 PBE peaks are approximately 4.360 GiB
(Psi4) and 1.202 GiB (GPU4PySCF). These must not be confused with GPU peaks.

## Provenance and reproduction

Energy values were recomputed directly from successful records in:

- `.pi/gpu_compare/archive/20261013-inferno/production/campaign/records.json`
- `.pi/gpu_compare/archive/20261013-aug-cc-pvtz/peptide_pbe/campaign/records.json`
- `.pi/gpu_compare/archive/20261013-aug-cc-pvtz/protein83_all/campaign/records.json`
- `.pi/gpu_compare/archive/20261013-aug-cc-pvtz/protein157_pbe/campaign/records.json`

Filter `profile=False`, `ok=True`, `process_returncode=0`, group by system/
functional/engine and subtract medians. Multiply Eh by 1e6 for microEh or
627.509474 for kcal/mol.

Memory comes from `outputs/gpu-dft-memory-20261014/memory-workers.csv`, excluding
profiles, selecting matching system/basis/functional/engine and taking the
maximum observed field. RSS KiB /1024^2; GPU MiB /1024.
Truncation and plan sizes come from the corresponding immutable archived
`results/*/psi4.out` and `pyscf.out` logs.
Algorithm snippets were read without importing/executing the GPU stack from
the installed GPU4PySCF 1.8.1 environment on Phoenix.

At the status check, build14056196 was PENDING/Priority and control14056197
PENDING/Dependency. No new shared-grid results exist. Neither queued checkout
nor installed environment was modified during this audit.
