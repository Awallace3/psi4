# Traced H2O properties reference: basis and resource contract

## This is not an ISA preset

`h2o_props_psi4_777f904_manifest()` in
`psi4.driver.procrouting.isapol_reference_basis` returns the immutable named
protocol `tracedH2O_props_psi4_777f904`. It does not register an
`ATOMIC_PROPERTY_RECIPE`, implement `PartitionRecipe`, run SCF, or promise a
matched property calculation. Passing it to `native_properties` is a type error.
The existing generated ordinary-A recipe and its defaults are unchanged.

The gate CLT requests `properties`, aVTZ, Psi4, and `Options Tests`, with the
explicit bohr geometry and fixed GRAC reference value recorded in the manifest.
The permitted generator trace selects **ALDA+CHF constrained NN**, spherical
MAIN, ordinary **Cartesian RI AUX**, and no separately emitted AtomAux.
`molecule_parser.F90` supplies the **same AUX as the AtomAux fallback**. No ISA
basis, shape, initialization or controller was requested: `isa_algorithm=None`,
`shape='shape_not_applicable'`. The response route uses eta=0 then .0005 and
lambda=1000; the final reference uses **LW plus PFIT**, not unrefined native LW.

Reference-file revision `777f90498868d33847de525612628b2dc8448523` is distinct
from inspected CamCASP revision `63b16a22b9bae597fe81ecdb8b8d91c21868c814`.
CLT, potential and O/H data bytes agree across these revisions. The generator
changes only three Psi4 charge/multiplicity/comment lines. Neither fact proves
which executable/environment generated the reference. The 2016 DALTON archive
has different grids/refinement; current ISA-A and ISA-A+DF templates request
other basis roles, representations and algorithms. They are not gate-9 history.

## Exact expected data, not a historical export

The manifest stores ordered immutable `ReferenceShell(l, exponents,
source_coefficients)` records for O and H only. MAIN is the expected spherical
aug-cc-pVTZ: O has 14 shells/36 primitive entries/46 functions, each H has
9/13/23, for **32 shells, 62 primitive entries, 92 functions**. Repeated
exponents in distinct contractions are retained, including signed O coefficients.

**New verification distinction:** native Psi4's bundled aug-cc-pVTZ has the
same orbital span but not literally the same O contractions. Both contracted
s shells omit terms at 1.752/.2384 and the contracted p shell omits .7156/.214;
these exponents are separately present unit shells. Subtracting those unit-shell
terms gives the native records exactly. H agrees as a contraction multiset.
Native therefore has 56 primitive entries versus the expected full 62. This
span equivalence is not proof of historical coefficients, orbital ordering or
orthogonalization. **Actual wavefunction MAIN remains `adapt_main(wfn)`**;
the manifest never builds or substitutes an orbital basis.

AUX is ordinary aVTZ RI: O has 28 primitive shells/136 Cartesian functions;
each H has 14/55, giving **56 shells/246 functions**. Source order is preserved:
a final diffuse S/P/D/F/(G for O) sequence follows tighter shells of *all*
ranks. Sorting by l changes this signature. The 198-function spherical
alternative is not the traced protocol. No set2 replacement is applied.

Source coefficients are 1.0, **not native effective coefficients**.
`ReferenceShell.effective_coefficients` first applies

```
c_pre[p] = c[p] * 2**l * (2/pi)**.75 * a[p]**(l/2+.75) / sqrt((2*l-1)!!)
K = sum_pq c_pre[p]*c_pre[q] * (2*l-1)!! * pi**1.5
           / (2**l * (a[p]+a[q])**(l+1.5))
c_eff = c_pre / sqrt(K)
```

`build_expected_reference_aux()` builds ONLY the expected fixed-geometry
`IsaExplicitBasis(MolecularAux, Cartesian, ...)`. The existing API supplies
GAMINT component scaling `sqrt((2*l-1)!! / product((2*i-1)!!))` and component
order. Do not apply that factor twice or renormalize these coefficients again.
This helper builds no MAIN, AtomAux, shapes, partition, NN response or PFIT.

`historical_scf_export_verified` is a read-only property returning **False**,
not an init/replace parameter. Missing artifacts include the generating CKS,
`H2O-A.basis`, SCF input/output/fchk/version, included snapshots, response-grid
and propagator metadata, unlocalized frequency responses, point-response p2p,
pdef and per-frequency PFIT/refined data. Inspecting signatures cannot upgrade
that status. No runtime reads of CamCASP, home directories, tests or options
are made; records and provenance are embedded in the production module.

## Pure resource preflight

`isapol_response_preflight.estimate_response_work(nbf, nmo, nocc, grid_rows,
max_nov=512)` uses explicit integer dimensions and actual supplied row counts.
Booleans, floats, negatives, values outside native-int range and inconsistent
`0 < nocc < nmo <= nbf` are rejected. Zero rows explicitly denotes gridless
`no_local`, not a valid ALDA estimate. Products use exact bounded Python integer
arithmetic without native overflow/wrap. Failure order mirrors
`native_response.cc`: OV, direct-JK, then ALDA. The diagnostic includes later
failures even though C++ short-circuits on the first.

The mirrored dimensional guards are nOV=nocc*(nmo-nocc) <= min(max_nov,512),
nbf<=256, nOV*nbf^4<=64,000,000,000, and (for ALDA) grid_rows<=1,000,000 with
grid_rows*nOV^2<=2,000,000,000. Equality passes. `require_pass()` raises
`ValueError` with the native `NativeResponseProvider:` prefix and first guard
message. This is **not a resource reservation or general validity certificate**:
shell l/primitive caps, workspace bytes, restricted-state/density/overlap checks,
grid values and scientific integration accuracy remain unchecked by this helper.
All C++ guards, error authority and limits remain unchanged.

The response wrapper checks before its grid snapshot/provider construction.
The public oeprop factory also checks actual `response_grid.shape[0]` before
partition/response construction. It still constructs its declared IsaGrid;
there is no allocation-free configured-grid estimator or automatic pruning.
No response rows are inferred from SCF DFTGrid options or a historical alias.

For the labeled **source-derived modern** 3-atom IsaGrid 99/590 case, there are
3*(99-1)*590 = **173,460 rows**, not 3*99*590 and not a pruned SCF grid. With
expected nbf=nmo=92, nocc=5, nOV=435, AO work is 31,163,093,760 (passes), but
ALDA work is **32,822,968,500** (fails). Maximum rows are **10,569**; 10,570
fails. This modern case is not the historical angular200/radial100 request,
a measured historical export, or permission to alter quadrature accuracy.

## Worked inspection (no SCF or integrals)

```python
from psi4.driver.procrouting.isapol_reference_basis import h2o_props_psi4_777f904_manifest
from psi4.driver.procrouting.isapol_response_preflight import estimate_response_work

m = h2o_props_psi4_777f904_manifest()
assert (m.expected_main_nfunction, m.expected_aux_nfunction) == (92, 246)
assert m.isa_algorithm is None and not m.historical_scf_export_verified
oxygen_main_signature = tuple((s.l, s.exponents, s.source_coefficients)
                              for s in m.elements[0].main_shells)
oxygen_aux_effective = tuple(s.effective_coefficients for s in m.elements[0].aux_shells)
check = estimate_response_work(92, 92, 5, 173460)  # expected dimensions, not measured
assert check.nov == 435 and check.max_grid_rows == 10569
assert check.failures == ('ALDA work resource limit',)
# check.require_pass() raises ValueError; do not lower the grid merely to pass.
```

## Source anchors, licenses and tests

Exact CLT/potential/current-and-historical-generator/source SHA-256 hashes are
in `source_sha256` and the independent fixture manifest under
`tests/pytests/data_isapol/h2o_props_psi4_basis/`. Anchors in the inspected
permitted CamCASP source: `cluster_file_interface.F90` defaults245–280,
properties406–416, ISA-only1123–1175, basis2539–2654, commands2657–2785,
NN5505–5571; `molecule_parser.F90` normalization299–336/fallback418–454;
`basis_operations.F90` normalization1944–2092. Native guard authority is
`native_response.cc` constructor; modern grid source is `isa_grid.cc`.

No whole third-party basis text was copied. Both O/H numeric datasets were
independently verified in the **BSD-3-Clause MolSSI Basis Set Exchange**:
aug-cc-pVTZ v0 agrees as a full contraction multiset and aug-cc-pVTZ-RIFIT v1
agrees in exact order. BSE's harmonic metadata is not adopted. Source basis
files were inspected for attribution/restrictions; their presence in CamCASP
alone was not treated as redistribution permission. See production
`isapol_reference_basis.NOTICE` and fixture `NOTICE` for full BSD and applicable
CamCASP MIT notices, Dunning/Kendall/Weigend–Köhn–Hättig and BSE attribution.
Retain these notices when distributing the module/data; not all CamCASP bundled
components or basis databases are asserted to be MIT. No ORIENT/RRF source is
used. Runtime needs neither the fixture nor its notice path to execute.

Fast tests independently hash literal records, check all coefficients/order,
analytic GAMINT component self-overlaps, immutable/non-ISA provenance, and
preflight bounds. Explicit native-core tests exercise the tiny sampling helper,
recipe refusal and genuine factory orchestration with bounded supplied rows;
a real small response test checks early rejection and unchanged default output.
Pure tests do not replace the existing basis/coulomb/native-partition/response
core suites. Parent integrated validation passed **1895 tests in36.79s**;
default water outputs remain bitwise baseline-identical in29.6536s/594120KiB.
The public fixed-GRAC strict-LW/9-pair endpoint also passes30.0985s/588436KiB.
Evidence: `.pi/audit/reference-basis-regressions-v1.log`,
`native-water-post-preflight-comparison.json`, and
`native-fixed-grac-post-preflight-water.json`. No large aVTZ response was run.

Review found no numerical defect. Parent added scoped fixture JSON ignore
exceptions and an explicit CMake notice-install rule (including the Cythonized
branch). The normal installed notice was byte-compared to source; Cythonized
installation itself was not exercised. Full fixture records are committed with
notices, not dependent on ignored local JSON.

Remaining blockers are constrained NN/distributed fitted-response reproduction,
historical SCF/grid/target artifacts, PFIT targets/model conventions and
refinement, and the modern grid's ALDA work bound. This patch supplies no large
matched endpoint, numerical parity, ISA reinterpretation or scientific-grid
convergence claim.
