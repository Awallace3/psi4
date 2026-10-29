# Standalone native ISA acceptance

The installed public route is:

1. perform an ordinary live restricted PBE0/aug-cc-pVTZ SCF;
2. declare an ISA recipe and response/Q grids;
3. call `psi4.oeprop(..., atomic_backend="BOUNDED_DF", distribution="isa", ...)`;
4. obtain the owned result with `psi4.atomic_property_result(wfn)`.

It needs no CamCASP executable, ORIENT/PFIT subprocess, reference orbitals,
saved partition or precomputed molecular response. `reference_h2h1` names a
fresh native-input response formulation, not a replay source. Native FDDS is
a separate opt-in response and must retain its own refusal gates.

## Integration provenance

This integration starts from cleanup `8f1533ea05d9ffc5c8b5ae205ead1b9dfaf1f187`
and ports the coherent native modules from full-stack candidate
`6f147f4829ffc83c868b9e78618ebe7c4f17fbdb`. It is not an ordinary merge or a
claim that either parent's numerical results validate the resulting tree.

The source port includes the ISA distribution/response/streamed-PFIT closure,
matching native ISA/FDDS/DFHelper/Libint2 interfaces, MBIS producer snapshots,
and the explicit finite `large` resource tier/72-variable allowance.
It retains the cleanup declared-asymptotic-correction producer and options,
shared SCF seals, compatibility basis imports, and existing golden fixture
contents. The bounded chain adopts the candidate's explicitly documented
static penalty inventory while direct expert refinement keeps its legacy
current-node/explicit-subset conventions.

## Independent installed run

The package ships `psi4.driver.procrouting.isapol_recipes.native_water_recipe`,
which constructs an explicit **regression** recipe from native Psi4 basis
data. Its absolute 1.5-bohr tails are not a Bragg-Slater multiplier or a
CamCASP-matched model. No test module is needed to construct it.

The optional exporter can run from an empty working directory:

```console
python -I -B -m psi4.driver.procrouting.isapol_acceptance \
  --build-manifest /new-build/build-identity.json \
  --output /new-run/water-smoke --profile smoke \
  --forbid-read-prefix /path/to/source-tree \
  --forbid-read-prefix /path/to/historical-research-products
```

The build manifest binds the exact clean source commit/tree and installed
Python/core bytes. It is provenance metadata, not a molecular input.
The `smoke` profile uses a small three-node rank-1 C6 calculation; `rank2`
uses eleven nodes, rank-2 refinement, 2000 fitting points and optional native
recoupled coefficients. These are different declared models.

The exporter refuses different installed bytes, child executables,
forbidden Python file reads, incomplete node coverage, unsolved PFIT, bad
response/localization residuals or leftover private response checkpoints.
Record OS-level file/process tracing separately: Python's audit hook does
not observe native-library `fopen` calls.

No completion or accuracy claim is implied by this documentation. Publish
results only after the exact integrated tree is built and the selected run
passes, with its loaded file hashes and scientific declarations retained.
Neither a smoke pass nor historical runs establish full C10, grid/quadrature
convergence, benzene execution, or CamCASP/SAPT parity.

## Phoenix-only benzene declaration

`isapol_recipes.native_benzene_recipe` and the installed
`psi4.driver.procrouting.isapol_benzene_acceptance` entry point use the existing
native engine, not the historical research driver or its molecular products.
Only a newly attributed neutral benzene geometry is an external scientific
input. The entry point refuses local/login-node calculation before scientific
work or output creation.

The geometry JSON schema is `native-benzene-geometry/1`: explicit bohr
coordinates in cyclic C1..C6,H1..H6 order, neutral charge/singlet multiplicity,
source description and source SHA-256. Hash-pin the exact extracted file at
invocation; never load a prior dataframe fixture, wavefunction or response.
Retain the original source-system/fragment and atom-order extraction record.

The declared PBE0/aug-cc-pVTZ model uses414 spherical MAIN functions,1146
Cartesian aug-cc-pVTZ-RI AUX functions, native ISA100/200 and Q200/974 grids,
75/302 SCF grid, unshifted orbitals and all11 Casimir nodes. It explicitly
declares72 noncharge rank1/2 upper-triangle variables on C1/H1, with unchanged
COPY frames/static penalty selection,2000 fitting points and full streamed
pair passes. Tails use1.5 times Bragg-Slater radii, unlike water's absolute
1.5-bohr regression cutoffs.

Numeric resources are25GiB,6.4e13 work and64GiB checkpoint I/O in the explicit
large tier. These are native numerical admission limits, not scheduler-memory,
RSS or runtime guarantees. Every result must cover432 isotropic pair/order
records and144 recoupled ordered pairs, including declared-model coverage;
rank2 C10 remains partial.

```console
python -I -B -X pycache_prefix=/new-empty-run/cache \
  -m psi4.driver.procrouting.isapol_benzene_acceptance \
  --build-manifest /new-build/build-identity.json \
  --geometry /new-input/benzene-geometry.json \
  --geometry-sha256 ACTUAL_SHA256 \
  --output /new-run/benzene-results \
  --forbid-read-prefix /path/to/source-tree \
  --forbid-read-prefix /path/to/historical-molecular-products
```

`phoenix_build.py` configures a matching clean source/core in a new private
dependency-only environment inside a compute allocation; optional adcc/Einsums
are disabled. It does not submit jobs or reuse old build outputs. Use an
explicit free `embers` job (never QoS defaults), independent decoded-file/
process tracing, fresh output/scratch directories and a TERM-safe wrapper.
No paid allocation, retry or historical runtime composition is authorized by
these files. A new convenience entry point creates a new source snapshot:
prior water results must not be relabeled as runs of that descendant commit.
