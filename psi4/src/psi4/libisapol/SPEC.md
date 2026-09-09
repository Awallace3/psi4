# libisapol: ISA-family properties and staged CamCASP parity

## 1. Objectives and current status

### Independent maximal-J validation after checkpoint f996942ad6

All735 J9/C11 and J10/C12 coefficient rows now have table-independent
factorial/electrostatic synthetic-input validation across7 blocks,26 ordered
quadruples and11 reciprocal classes. Parent1755 ISA/FDDS tests pass24.71s,
including14 new high-J cases; independent review found no scientific must-fix.
No production algorithm changes. See HIGH_J_VALIDATION.md for Sbar normalization,
external mathematical source and precise certification limits. This does not
numerically compare the archive's411 visible high-J rows or derive a general
lower-J second-stage oracle. Rank4 and native-SCF/PFIT/GRAC parity remain open.

### Accepted OpenMP / bounded-memory followup

Final staged performance build passes1710 ISA/FDDS tests in24.06s. Relative to
newly profiled360.6726s /780104KiB RSS baseline, controlled fresh-water runs at
1/2/4/8 property threads take29.4297/22.5083/17.9785/15.9646s (12.26–22.59x).
Peak RSS586864/584392/586736/598196KiB is23.32–25.09% lower. SCF is fixed at
one thread for this scaling comparison; same input hash, energy and31 iterations.
All scalars, local/global tensors and Cn pass the unchanged1e-9 scaled-error gate:
1thread bitwise identical, maximum across other threads9.83e-16. No statistical
speedup claim or full no-OpenMP core validation (helper-only compile/run passed).

Independent-output OpenMP keeps per-output primitive/point sums ordered and selects
the lowest-index worker exception. ISA caches immutable preparation within checked
numerical-payload bounds (not an RSS cap); a cacheless owned snapshot releases it
before final-Q/response work. AUX-Q uses bounded collocation batches without changing
running point-order sums; direct-MO GEMM remains unchanged. The initially faster
but15%-larger-RSS build was rejected. A snapshot unique/shared holder mismatch was
caught and fixed before acceptance, with ownership/deletion regression coverage.

Initial2/4thread drift was isolated to thread-sensitive MKL SVD arithmetic in the
small AO convention conversion. `_isa_serial_blas_call` scopes an exception-safe
**thread-local** MKL override to that conversion, restoring prior state on success,
exception and nesting. Other callers/OpenMP settings are untouched. Non-MKL builds
retain their backend behavior; no process-wide fallback is used. Tests exercise
actual-water adaptation and cross-Python-thread isolation. No scientific tolerance,
charge, symmetry or rank repair was introduced.
Evidence: `.pi/audit/native-water-final-comparison.json` and
`native-water-performance-tests-v4.log`. Final molecular/SAPT reruns PASS:
3 molecular tests in97.35s and4 SAPT tests in113.48s. Normal8thread SCF+property
input also passes14.32s/743280KiB RSS; do not apply the controlled memory reduction
to this different SCF thread policy. Recoupled rank<=3 parity is now built/staged and accepted at archived write
precision:31 focused tests and1741 integrated regressions24.74s. Exact6285 rows,
10457 nonzero values rtol1e-6/atol0,30791 written placeholders abs<=1e-6; trailing
omitted fields also bounded, mutation-tested.411 J9/10 rows counted but not
numerically oracle-validated. Rank4 rejected; C12 partial. New shipped1101 real-CG
records plus existing393blocks/4673terms implement Cn(t,u,J), independently of the
orientation-resolved scalar engine. Fixtures are hash-pinned H2O-isagrid (O-O C6
26.48177), not historical H2O. See RECOUPLED_CONTRACT.md. No native-SCF/PFIT/GRAC
full-protocol parity claim. Final integrated core fresh-water outputs remain
bitwise baseline at1thread:29.8982s/589516KiB RSS. Parent review fixed an absent-field
oracle gap and removed internal tensor clones before preflight; public ownership
is unchanged. Evidence `recoupled-parity-regressions-v2.log` and
`native-water-post-recoupling-comparison.json`.

### Native ordinary oeprop integration (current work)

The wavefunction-first `ATOMIC_PARTITION`, `ATOMIC_POLARIZABILITIES` and
`ATOMIC_DISPERSION` requests are implemented, with unchanged None return and
`psi4.atomic_property_result(wfn)` owned result access. `PARTITION_SCHEME` is
registered separately from `ATOMIC_RESPONSE_LOCALIZATION`. The automatic H/O
`GENERATED_JKFIT_ISA_A` recipe is explicitly NOT the modern CamCASP preset.
It uses runtime shipped Gaussian data and generated grids, native Drho-C ISA-A,
and separately labeled direct-OV response moments rather than a finite-penalty
transition fit. Existing fitted-AUX defaults remain unchanged. No charge or
asymmetry repair, hidden SCF, archived electronic input, PFIT or LW waiver is
introduced. See `NATIVE_OEPROP.md` for policy and representation details.
Configured build/staging and1686 installed ISA/FDDS regression tests pass. Independent
review fixes require actual wavefunction-local SCF stopping evidence, unchanged
state and effective canonical PBE0, reject unknown atomic requests and invalidate
stale result attachments. Initial fresh-water partition passed, but response hit its
ALDA workload bound. ISA and response grids are now separately configured (160/590
and99/590 defaults); the original workload bound and strict LW1e-6 remain unchanged.
The actual uncommitted `tmp/psi4_camcasp.py` now completes a fresh PBE0/cc-pVDZ
water SCF through all11 response nodes and nine site-pair Cn sets in341.305s.
Static O/H1/H2 dipole trace alphas are3.54925461809/.875697468695/.875697468444
bohr^3; all9 independent saved-array C6 contractions agree within1.78e-15.
Four unchanged SAPT regressions pass. This establishes the explicitly labeled H/O
native demonstration endpoint, not modern CamCASP parity or PFIT refinement.
All3 final fresh-water molecular assertion tests pass in1346.09s, including the
independent analytic MO-dipole/H1 check, request isolation and result ownership.
Evidence and commands:
`.pi/audit/native-oeprop-water-handoff.md`.
Older checkpoint paragraphs below retain their historical scope; they do not
override this latest status or establish full CamCASP parity.

The objective is numerically validated **properties**, not merely reproducing tables:
ISA-family atomic densities/shapes and multipoles, distributed and localized
frequency-dependent atomic polarizabilities, and isotropic and anisotropic dispersion.
The implementation is primarily C++; Python drives calculations, selects explicit
protocols, and exposes inspectable stage results.

Agreed implementation sequence:

1. **ISA-A first**, within an extensible ISA-family architecture.
2. Compare **matched intermediate inputs** between implementations before comparing
   independently generated wavefunctions. Water is the first molecular target.
3. Make partitioning, response, localization, fitting, and dispersion independently
   callable. `oeprop` is a convenience adapter, not the owner of pipeline state.
4. Target isotropic and anisotropic dispersion, with explicit **rank coverage**;
   never advertise a truncated model coefficient as a complete coefficient.

Existing code implements element tables, integration/frequency grids, fit-point
sampling and dispersion coefficient **tables**. Existing tests check these building
blocks. Neither ISA convergence, atomic polarizabilities nor a dispersion engine is
implemented by that groundwork. Claims of full anisotropic parity are premature.

The first new numerical increment is a **frozen ISA-A fitting update** operating on
explicit sampled density/shape/basis values and an explicitly supplied weighted
atomic overlap. This is a developer-stage API, not a converged ISA property task.
A water checkpoint test can establish arithmetic parity at this boundary without
claiming that the input density is the reference Drho-C density, or that basis
construction, activation scheduling, tails, and the fixed point have been validated.

### Latest supplied-external property workflow

At the user's direction, internal localization/refinement is deferred by importing
actual ORIENT local/refined outputs through a hash-verified manifest. The Python
`isapol_supplied` driver now emits imported atomic tensors, Psi4-derived trace
polarizabilities/global dipoles, isotropic C_n and optional explicitly placed
orientation-resolved anisotropic C_n. Numerical disagreement remains reported
without suppressing valid output; structural invalidity remains an error.

Both reviewed bridge boundary issues are fixed and independently cleared;
54 focused and1152 combined tests pass. The generated water artifact preserves
frequency-header conflicts, indefinite tensors and missing ranks. All12 bounded
archived isotropic coefficient comparisons pass at printed precision. See
`SUPPLIED_PROPERTIES.md` for runnable commands and
`tests/pytests/data_isapol/psi4_orient_bridge_evidence.json` for measured evidence.

Native-integral/supplied-orbital C++ OV fitting also passes provisional1e-3 and
matches its same-input NumPy control coefficient values bitwise in common layout.
This does not close native SCF/Hessian/kernel or wavefunction-first property gates.
Imported ORIENT tensors are not attributed to a new Psi4 wavefunction prediction.
The LW replacement now localizes supplied nonlocal tensors through C++ and the
expert `isapol_lw.supplied_nonlocal_properties` driver. It returns owned raw
local/global tensors, trace scalars, global dipoles and diagnostics; a separate
explicit-weight isotropic-dispersion adapter reuses the existing C++ engine.
Static-only requests require neither PFIT nor quadrature. Parent installed-module
water generation and1335 combined tests pass; independent driver review found no
must-fix for factory-produced results. The exact historical snapshot requires the
user-authorized identity-guarded diagnostic policy and retains failed production
postconditions (section6). An explicit-placement anisotropic driver adapter now
reuses the C++ engine with unchanged raw-global tensors and exact reciprocity;
it never repairs asymmetry or roundtrips through local frames. Independent review
found no must-fix; final parent installed-import tests passed52 and combined
ISA/FDDS tests passed1387 in14.71s, including the portability-only getter-access
regression. Bounded evidence: `psi4_lw_anisotropic_driver_evidence.json` under
`tests/pytests/data_isapol/`.
Results preserve source models, placements, quadrature, energies and ordered-rank
coverage (unrestricted completeness through C8 for ranks1–3). This is synthetic
supplied-input integration, not dynamic molecular acceptance. Dynamic molecular
LW-to-C_n, native upstream generation and stricter numerical parity remain TODOs.

The ten historical positive-frequency NONLOCAL snapshots and independent unrefined
L3 references are now imported with63000 literal tokens and20 pinned source hashes.
Parent strict measurement rejects ALL10 nodes at1e-6; exact worst input charge
magnitudes range7.011e-4 to2.990e-6. Zero of6750 candidate local entries are exposed,
and dispersion is null. Passing1458 combined tests validates import and expected
rejection, not molecular localization. Independent review found no must-fix for
this bounded checkpoint. Chosen CasimirGrid(10,0.5) is not certified exact producer
quadrature; four L3 printed-header failures remain. No dynamic extension of the
static-only historical exception is authorized. See
`tests/pytests/data_isapol/psi4_lw_dynamic_evidence.json`.

### Latest independent property and observer gates

The approved supplied-local anisotropic direct-contraction stage is implemented,
canonically built/staged, and independently reviewed with no must-fix findings.
It returns orientation-resolved scalar C6–C12 (including odd orders), truncated
energies and explicit ordered-rank coverage, not CamCASP recoupled coefficients.
120 focused tests and 1002 combined ISA/FDDS tests pass. See
`ANISOTROPIC_CONTRACT.md`, `SUPPLIED_PROPERTIES.md`, and
`tests/pytests/data_isapol/psi4_anisotropic_dispersion_evidence.json` for formulas,
provenance and qualified oracle/resource limits. No native response, localization
or historical-reference anisotropic parity is established by this increment.

The supplied-input PFIT stage now implements the explicit single-frequency objective
with exact within-physical-batch triangular pairs, correlated matrix/LC penalties,
fixed elimination, DSYSV and streaming QR. The reviewed LC scaled-row diagnostic
fix has six reproduced regression cases; 102 focused PFIT tests pass. Canonical
installation matches the built core, and the combined ISA/FDDS suite passes
882 tests. API and limitations are in `SUPPLIED_PROPERTIES.md`; bounded evidence
is `tests/pytests/data_isapol/psi4_pfit_evidence.json`. This is **not** native target
generation, localization, historical-reference PFIT parity or driver acceptance.

The specific fresh schema8 attempt6 reference-observer gate is independently
verified: 4,855 events, zero disabled events, 35 byte-identical traced/untraced
artifacts, frozen metric/tensor consumer generations, exact H1/H2 reconstruction
and eleven-frequency strict replay at unchanged tolerance. Independent source
review found no must-fix in its bounded audit. The worker timed out during reporting;
parent verification recovered the evidence, rather than treating timeout as success.
See `tests/pytests/data_isapol/camcasp_response_cache_v9_evidence.json` and
`.pi/audit/cache-observer-v9-handoff.md`. Acceptance is limited to the adapted
fixed-geometry molecule-A water path; B mapping/rebuild coverage remains synthetic
or source-based. The reader's `observer_validated=False` remains intentional.
Native Drho-C/OV fitting, tails, localization and end-to-end gates remain open.

### Temporary user-authorized numerical profile

For struggling forward numerical comparisons, the user authorizes an explicit
provisional1e-3 profile to unblock integration, with stage-specific tightening
TODOs. Original strict metrics and results remain recorded; existing passing
tests retain their tolerances. Structural/safety/provenance requirements and
missing scientific/native implementations are not waived. See
`PROVISIONAL_ACCEPTANCE.md` for the scope and TODO9–12. Numerical values above1e-3
remain failures. Earlier unchanged-tolerance statements describe historical runs,
not a prohibition against this new separately labeled provisional profile.

### Evidence discipline

Separate these claims in tests and documentation:

* analytic/invariant checks;
* same-input numerical-kernel parity;
* fixture/parser consistency;
* full independently executed stage parity;
* wavefunction-to-property parity.

Every molecular fixture needs a schema version, geometry in bohr, atom labels/order,
input representations, normalization, units, effective settings, source paths/hashes,
generator commands, and limitations. Record measured absolute and scaled errors.
Relative errors alone are not useful for symmetry-zero components. Printed output
precision bounds achievable comparison accuracy. Bitwise compatibility is appropriate
for selected constants/RNG/grid fixtures, not a blanket requirement on BLAS results.

The pre-audit draft was preserved locally at `.pi/audit/SPEC.pre-source-audit.md`.
This document supersedes its equations, scope exclusions, causal claims and status.

## 2. Reference tracks and protocols

Source paths below are relative to the local CamCASP checkout
(`/home/awallace43/gits/CamCASP`). References identify executable code, not just comments.

### 2.1 New ISA-A reference

Use **both** `methods/isa-pol-from-isa-A` and
`methods/isa-pol-from-isa-A.clt_tmpl`; retain the expanded generated input. The method
includes constrained/unconstrained density fits, response kernel settings and
regularization that were absent from the previous specification's excerpt.

The template specifies:

* OBS aVTZ;
* molecular AUX aVTZ, spherical, with ISA-basis modifications;
* AtomAux aVQZ, spherical, with ISA-basis modifications;
* ISA basis set2; minimum H s exponent 0.2.

These are **not** established aliases for decontracted Psi4 JKFIT. Preserve separate
OBS, molecular AUX, atomic AUX, and ISA augmentation roles. Audit and export actual
primitive lists, contraction factors, harmonic order and s-shell mapping before
building native equivalents. Explicit supplied bases are supported architecture,
not an exception to hide. Hybrid ISA algorithms impose stronger basis constraints.

Selected method settings include:

* Gauss-Legendre quadrature, beta **0.5**, ten positive imaginary frequencies;
* ISA grid radial 100, angular request 400 (434 actual Lebedev points);
* lattice charge +1, inner/outer vdW factors 2/4, 2000 accepted points, seed 1;
* molecular DF type NN, eta=gamma=0, fits with lambda=1000 and lambda=0;
* ISA density **Drho-C**, ONE-GTO initialization with alpha0=1, algorithm A, LU;
* W convergence 1e-9, EPS-Q 1e-4 (not an additional W stopping condition), max 120;
* W damping/mixing zero, skip 20, W-Eps 0.17, s-block-only, coupled activation 1e-5;
* Positive-W lambda 0.001, auto, max exponent 0.2;
* Func-1 tails, fit type 3, Slater radius multipliers 1.5/2.5;
* distributed response ISA-GRID, spherical rank 4.

The `Tail-Iterations 30` line does **not** request thirty postconvergence iterations
for ordinary A without SELF-CONSISTENT-TAIL; see section 4.

Response/kernel parameters must be copied from the expanded method input, including
NEW-PROP, C-DF, kernel-integral controls and propagator DF settings. Do not infer them
solely from the SCF functional name. Record any asymptotic correction/IP explicitly.

Use water O=(0,0,0), H1=(-1.45365196,0,-1.12168732),
H2=(+1.45365196,0,-1.12168732), in bohr, C1, no COM shift/reorientation. Pin SCF
thresholds and grid separately from the ISA grid; run parity references single-threaded.
For subsequent native-wavefunction comparison, start with the same exported
coefficients, energies, occupations and auxiliary representations in both codes.

### 2.2 Historical downstream water regression

`examples/properties/H2O/output_1/H2O_aTZ.cks` and its output describe a historical
**cDF**, not ISA-A/ISA-GRID, calculation (output reports CamCASP 5.6.10). The actual
input has Cartesian AUX, angular 100, radial 60, and **500** random points despite
`p2000` filenames. Point-response and distributed-response DF settings differ
(eta=0 versus eta=0.0005, lambda=1000). The `output_2` localized/refined tensors and
parameter definitions are useful supplied-input downstream fixtures only.

The archived Casimir input requests `Dispersion 10`; its output contains C6 through
C10, **not C12**. Its reported penalties suggest strength 1e-5, not the modern
script default 1e-3. Reconstruct actual settings rather than assuming modern defaults.
A complete chain of custody for all archived artifacts has not been established.

The prior implementation at `/home/awallace43/gits/camcasp_psi4` is a migration and
reference-data source, not automatically correct or automatically failed. In particular,
`.camcasp-reference/work/H2O-isagrid/` contains water orbitals, expanded input,
serialized ISA shapes and response files. These require their own provenance audit.
Its input differs from the modern preset (Cartesian molecular AUX, aVTZ AtomAux,
angular 200 and 500 fit points). Inspection of `H2O.cks` additionally finds
**Doo-C density and BVLS**, not Drho-C/LU. Its constrained solved coefficients are
not an unconditional reference for the frozen LU API. Do not relabel it as the
modern preset. The opt-in `prepare_isa_water_run.py` creates a separately identified
Drho-C/LU adaptation with explicit NN fits; archived files remain unchanged.

### 2.3 Upstream in-tree psi4-driven end-to-end reference

CamCASP's own test suite commits a complete water properties reference computed **with
psi4 as the SCF code**: `tests/H2O_props/psi4/{H2O-avtz.clt, H2O.axes}` ->
`tests/H2O_props/psi4/check/L2H1/H2O_ref_wt3_L2_Cn.pot` (26,305 B, 402 lines, C6-C10),
driven by `tests/test_H2O_props.py --scfcode psi4` (`runcamcasp.py` then
`localize.py H2O --limit 2 --hlimit 1 --subdir L2H1`). This is the end-to-end target for
section 9's final gate. It is MIT CamCASP test data, so it may be committed as a fixture
with the MIT notice and Misquitta/Stone attribution; at 26 KB it is small enough to commit,
which the 697 KB L3 `H2O_ref_wt4_L3_C12.pot` is not. Upstream commits only the *output* - there is
no `casimir.data` deck in `check/L2H1/` - so it cannot substitute for the hermetic
input/output pair used by section 7's dispersion gate.

The `.pot` header records the full model/localization definition (`Limit 2`, `WSM-Limit 2`,
`H-Limit 1`, `Isotropic? False`, `Pol Cutoff 1e-4`, `Loc algorithm LW`, `Weight 3`,
`Weight coeff 1e-3`, `SVD threshold 0.0`, `NoRefine? False`); the remainder is
`bin/camcasp.py`'s defaults (`DFT`, `PBE0`, `ALDA+CHF`) plus aVTZ.

**The parallel `nwchem/` and `dalton/` references are not interchangeable with it.** The
three `.clt` decks are byte-identical apart from `SCFcode` and `HOMO`, but `ac_type` is
hard-wired by SCF code (`camcasp.py:1237-1244`: psi4 -> GRAC, nwchem -> CS00,
dalton -> LB94+TANH) and the shift is fixed at `delta_ac = ip + homo`
(`camcasp.py:1251-1253`), giving `0.064900 / 0.065800 / 0.131930` Eh. Measured over the 380
common rows, isotropic O-O C6 is `19.27258 / 18.76416 / 18.26039` (5.25% below the psi4
value) and C10 is `4106.707 / 3891.527 / 3672.440` (10.57%); psi4-vs-nwchem
`max |dev| 2.1518e+02`, `max rel 1.8958`, with 223 of 526 nonzero values differing by more
than 10% and 499 by more than 1% (psi4-vs-dalton: 332 and 519). The largest relative
deviations are sign flips on near-cancelling H-H anisotropic terms of order `1e-05`. So the
AC scheme and shift are part of the gate definition rather than an implementation detail,
end-to-end comparison must use the psi4 reference alone, and a uniform relative tolerance is
not meaningful for the small anisotropic coefficients. Measured evidence:
`.pi/audit/gate9-endtoend-reference-objective.json`.

*Not an ORIENT-free route.* This chain still calls `localize.py`, whose `Loc algorithm: LW`
step is ORIENT, so it sits downstream of the leg A reimplementation in section 6 and
inherits its `~1e-07` floor. Separately, `src/not_used/localize.f90` (3,590 lines, in no
build file) is **not** a leg A candidate despite its name: it localizes the FDDS in an
auxiliary basis, solving for `B^s_{kl,rs}` projection coefficients from T- and U-type
integrals over auxiliary-function pairs with seven interchangeable solvers, and mentions
LW, LS, multipoles and redistribution nowhere. It is an abandoned alternative physical
route to distributed polarizabilities, not the LW redistribution.

## 3. Modular API and result contracts

### 3.1 C++ owns numerics; Python owns orchestration

Planned stages (names are design targets, not currently available functions):

```
DensitySource + basis recipe + grid -> ISA strategy -> PartitionResult
Wavefunction/transition data + kernel policy       -> ResponseModel
PartitionResult + molecular AUX                    -> PartitionedMultipoles
PartitionedMultipoles + ResponseModel + frequencies -> DistributedResponse
DistributedResponse + graph + frames                -> LocalizedResponse
PointResponse + parameter model + anchors           -> RefinedResponse
Response model A + response model B + quadrature    -> DispersionResult
```

Use separate typed options/data/results per stage. A result must identify units,
site coordinates/order, frame and component convention, ranks, frequency list,
provenance, convergence and diagnostics. Do not encode the primary result in a padded
QCVariable matrix or an undocumented dictionary. Do not retain mutable global options
or stale wavefunction-dependent caches inside numerical kernels.

Allow injected density, partition, response tensors and parameter models so developers
can replace one stage without rerunning unrelated stages. Copies/views and ownership
must be documented at the binding boundary. Validate dimensions, symmetry blocks,
finite numbers, ranks and singular solves with descriptive exceptions. Unsupported
methods must fail explicitly rather than silently selecting A.

The first `IsaAFitData`/`IsaAFitOptions`/`IsaAFitResult` API is a frozen, atom-local
fit boundary: sampled quadrature weights, density, selected shape, shape sum, distance
squared and basis values; supplied **already W-Eps-weighted** overlap; previous
coefficients, per-function angular momenta and primitive exponents. This initial
boundary requires an uncontracted atomic basis and a positive exponent for every
function, including non-s functions; callers must decontract before sampling.
It does not build bases or decide activation. `isa_a_fit_step` returns the modified metric, RHS,
coefficients, integrated partition population and a normalized linear residual.
The supplied overlap must correspond to the active W-Eps/s-block setting. The separate
`isa_overlap_change` diagnostic uses an **unweighted** overlap, as CamCASP does.

### Explicit exported-input sampling provider

`IsaGaussianShell`, `IsaExplicitBasis` and `IsaFixedDensity` now provide a typed
C++ sampling boundary for supplied Gaussian descriptors. Basis construction takes
an owned immutable snapshot of centres (bohr), shells, positive primitive exponents
(bohr^-2) and **effective** contraction coefficients, without renormalizing them.
Shell centre indices are zero-based; shell/function order is preserved. Supported
angular ranks are S–G, Cartesian GAMINT and spherical DALTON (p=x,y,z), evaluated
using a solid-harmonic recurrence independent of the Python polynomial oracle.
Contracted signed radial sums are supported. Results are fresh point-by-function
Matrices; mutations do not affect subsequent evaluations.

Explicit roles distinguish molecular AUX, AtomAux and s-only shape bases.
`IsaFixedDensity` accepts only molecular AUX and a finite, dimension-matched
coefficient vector, copied on construction. It evaluates a supplied expansion;
Drho-C identity/provenance belongs to the caller, not inferred from coefficients.
Density sampling requires unique zero-based active neighbour sites, **without
Fortran padding**. Empty means no active sites, not all sites. Descriptor adapters
must explicitly convert the positive one-based prefix and discard zero padding.
Signed density is retained, without clipping or charge rescaling. Inputs and
nonfinite computed samples fail descriptively. Screening is batch-wide, so callers
with distinct neighbour lists must evaluate separate batches.

`IsaExplicitBasis.overlap(w_eps, s_block_only)` constructs an analytic, co-centred
AtomAux/Shape metric before damping/ridge. Molecular AUX and distinct used centres
are rejected (different site indices at exactly equal coordinates are allowed).
Cartesian blocks use Gaussian monomial moments; spherical blocks use angular
orthogonality and the Racah radial normalization identity, not the Python polynomial
product algorithm. Effective coefficients remain untouched. W-Eps is finite and
nonnegative, shifts primitive exponent sums only in the s/s block by default, and
has **no grid exponent cap**. Every primitive pair must have finite positive shifted
exponent sum, even if its angular integral or coefficient is zero. Nonfinite results
fail; a finite returned metric is not a certificate of linear independence or positive
definiteness. The all-block option implements the stated mathematical metric; it
is **not** certified as parity with the known anomalous upstream all-block path.

`IsaShapeMap(atomic, shape, shell_map)` validates a unique zero-based target shell
for each shape shell, without padding. Target s shells must exactly match centre
coordinates, ordered primitive exponents and effective contraction coefficients.
An explicit subset/permutation is allowed; radial equivalence under reordered or
rescaled contractions is not inferred. Cartesian/spherical s shells are equivalent
at this boundary. The map owns function-column indices and projects a finite,
dimension-matched full AtomAux coefficient vector into a fresh raw shape vector,
without clipping, normalization, mixing/DIIS or tail replacement.

This is exported-input reconstruction in C++, **not** native basis/Drho-C generation,
JKFIT substitution or AO density. The analytic metric can be supplied to the existing
frozen-fit API. `IsaAFitProvider` now owns copies of primitive co-centred AtomAux
and fixed molecular-AUX density; `assemble(IsaAFitSamples, options)` derives atomic
samples, density, raw squared distances, weighted overlap and per-function primitive
metadata. Samples explicitly supply weights, screened/tail-processed shape and shape
sum, previous full coefficients and a batch-wide density neighbour list. No shape
is reconstructed or clipped. Returned fit data is independent and inspectable;
a later solve must use the same weighting settings. `fit(samples, options)` couples
assembly and solve with identical options. This is one frozen update, not a sweep.
The constructor rejects contracted AtomAux and wrong roles; assembly validates
finite dimensions/options and delegates density screening to the explicit provider.
No shape-tail policy, controller, or end-to-end property capability is implied.
Contracted sampling and overlap do not relax the frozen fitter's primitive-only
requirement. The four-file staged suite passes 261 tests including synthetic assembly
and replay checks; all three full reconstructed production replays passed against
the current pinned reference. The requested Libint2 reference transition is deferred
by the user, not a prerequisite for this evidence (see `plan.md`). Psi4's staged general integral backend is
Libint2 2.13.1, but the current libisapol explicit Gaussian sampling and co-centred
weighted metric remain independent analytic arithmetic, not Libint2 calls.

### Native explicit-recipe Drho-C / ISA-A partition adapter

`isapol_native_partition.native_partition` now accepts a real restricted C1
wavefunction and explicit named Cartesian molecular-AUX/primitive atomic/shape
basis recipe. Validated shell collocation transforms actual Psi4 orbitals into
DALTON coordinates, checked by held-out points and independent multicentre overlap.
Native lambda1000 Drho-C, absolute-nearest unit ONE-GTO initialization, existing
ordinary-A controller and final stored-tail-aware full-grid Q are connected.
No archived orbitals/density/shape coefficients, AO-density substitution, Doo-C,
hidden SCF or automatic modern spherical-AUX recipe is used.

A fresh RHF/aug-cc-pVTZ water calculation converged in38 iterations and produced
Q(75,246). Independent review found no must-fix in this expert path; canonical
prerequisite installation/byte checks and1595 combined tests pass. Severe metric
conditioning, Q quadrature discrepancies and a separately failed small H2 recipe
remain recorded. Drho comparison without reference is not marked passing merely
by the1e-2 allowance. This establishes partition-through-Q, not returned atomic
polarizabilities/Cn; their native pipeline integration is currently running.

### 3.2 User surface

The ordinary user workflow is **wavefunction-first and property-request-driven**:
run `energy(level_of_theory, return_wfn=True)`, then pass that wavefunction to an
`oeprop`-style property request, as users already do for MBIS. ISA is not an energy
method and must not be registered as `energy('isa')` or `energy('isapol')`. Examples
should start with PBE0/aug-cc-pVDZ; that usability example is not the separately
pinned CamCASP parity protocol. No hidden SCF is performed by the property call.

Keep the normal input short: the user supplies a converged wavefunction, selects
partition/localization policies through Psi4 options, and requests properties.
Use the method-neutral option name `PARTITION_SCHEME` (now registered),
with choices such as `ISA_A` and, after adapter validation, `MBIS`; do not use
`ISAPOL_PARTITION`. Scope it to atomic density partitioning, not orbital or tensor
localization, which require separate options. The name selects a strategy rather
than coupling the public interface to the first implemented algorithm.
The adapter constructs the required density/basis/grid providers internally. Users
must not assemble sampled arrays or call every internal stage. A separate expert
API retains explicit prior-stage injection for parity testing and interoperability;
this does not remove the wavefunction requirement from the ordinary ISA entry point.

Use `oeprop(wfn, ...)` as the initial usage model. Whether response and dispersion
ultimately live behind the same function or a sibling such as `teprop(wfn, ...)`
remains a naming decision; no `teprop` API or new property task names are established
by this text. `PARTITION_SCHEME` is now available for the bounded native adapter; unsupported strategies fail explicitly. Existing `oeprop` returns None, so any structured-result access or
sibling return contract must be explicit and backward-compatible. Publish small
conventional QCVariables while retaining large labeled tensors in typed results.

Selectable partition/localization is a project goal, not just an ISA-only parity
path. ISA-A remains the reference strategy; adapters should also reuse suitable
existing Psi4 methods, with MBIS as a density-partition candidate. Distinguish
atomic density partitioning (ISA/MBIS), orbital localization (e.g. Boys/Pipek–Mezey),
and distributed-tensor localization (LW/LS): these are different transformations,
not interchangeable enum values. Audit which existing methods expose the weights,
shapes, moments or transformations each downstream stage actually needs. Charges
alone do not define a continuous partition. Unsupported combinations must fail
explicitly, never silently fall back to ISA. Validate and report how changing the
selected policy affects downstream properties, including conservation, translations,
frames and rank coverage. Snapshot effective options and policy identity in results.

The property adapter calls the same independently callable C++ stages as the expert
API. Provide separate partition, atomic-response and dispersion requests. A static
polarizability request must not implicitly fit a 2000-point cloud or compute every
C_n. Requesting partition and response together must reuse a validated partition
for the same wavefunction and effective policies. Do not build `DFTGrid` in oeprop;
the dedicated ISA grid is owned by the ISA partition strategy. Other strategies may
own different grids. Unsupported tasks are not registered until their complete
prerequisites and strategy adapters are implemented and tested.

Keep raw stockholder populations, fitted-shape populations and any charge-rescaled
legacy values distinct. Volumes mean explicit radial moments, e.g. integral of
r_a^3 rho_a, not a volume ratio unless a free-atom reference is also supplied.
Site-pair dispersion must support different models A/B, not just a monomer paired
with itself. Large/anisotropic tensors belong in object results with labeled axes.

### 3.3 ISA-family roadmap

A is first. A+DF, DF+ISA, B1, B2 and a real-space strategy remain planned capabilities,
not permanent exclusions. `stockholder.F90:700–815` distinguishes these basis-space
algorithms and requires equal spherical molecular/atomic AUX for the hybrid paths.
The source module defaults to A+DF; the selected preset overrides it to A.
GISA is also a basis-family label in the input parser, not evidence of another
implemented solver. Transition-density ISA is unfinished upstream and not promised.
Neutral closed-shell water is the first acceptance case, not an architectural ban on
ions, external partitions or disconnected systems. Response spin support and graph
localization support must be validated independently of density partitioning.

## 4. ISA-A mathematical and operational contract

### 4.1 Fixed-density fitting

The native density gate remains pending. `IsaAuxCoulomb` now implements the first
integral component: Cartesian molecular-AUX S-G q and J from explicit effective
coefficients. J uses Libint2 raw Cartesian shells (normalization embedding disabled),
standard Cartesian normalization, configuration-aware indices, explicit GAMINT
component factors, true unit shells, BraKet xs_xs and precision0. Analytic q includes
all even Cartesian components. Production q/J generation now agrees with exported
reference to max scaled 3.47e-16 / 1.60e-14 (largest angular-block error 4.94e-14).

The subsequent `three_center` API uses Cartesian-only xs_xx engines and explicit
DALTON spherical MAIN polynomial transforms, returning rows AUX and column
`mu*nmain+nu` (nu fastest). `closed_shell_rhs` forms `2 sum_occ C_i^T B_k C_i` before
solving, assuming all supplied occupied spatial orbitals have occupation 2. It adds
no charge penalty and does not sample an AO density as a substitute for Drho-C.
The explicit Orbital basis role supports spherical S-G only; Cartesian MAIN is
rejected pending its distinct angular-normalization adapter. Native occupied-trace
RHS now agrees with reconstructed reference to scaled 4.60e-13 (raw) and 4.27e-16
(with penalty). Spherical AUX remains unsupported.

`fit_drho_c` and `IsaDrhoCResult` pass analytic tests but FAIL production density
parity: coefficient max absolute error 0.04459, sampled-density max absolute 1.425e-4
and pointwise scaled 4.664e-6 at unchanged 1e-9. Matrix/RHS hybrid diagnostics show
sensitivity to both inputs; C++/NumPy native solutions agree exactly. Portable failure
evidence is `camcasp_native_density_failure_evidence.json`. The implemented model is: positive
finite charge penalty, native q/J/B, occupied diagonal penalty before the trace,
explicit column-major LU (no refinement), original/constrained matrices and unrescaled
coefficients. No symmetrization or regularization is applied. The reported residual
is infinity-norm backward error, not a certificate of coefficient accuracy. Fitted
charge and sampled-density comparisons must be reported separately under the observed
severe conditioning. This still takes explicit basis descriptors and occupied C;
it does not generate the SCF or basis recipe. Psi4 owns
Libint2 global initialization and ordering. Source audit clarifies that DF `TYPE NN`
selects MO pair space, **not** the fitting norm. For the current Coulomb-norm,
closed-shell eta=gamma=0 track, let J be the two-centre molecular AUX Coulomb
metric, B the AUX–MAIN–MAIN three-centre Coulomb integrals, q the integral of each
AUX function, and C the occupied spatial orbitals. Drho-C solves
`(J + lambda*q*q^T) d = 2*sum_i(C_i^T B C_i) + lambda*(2*nocc)*q`.
The pinned DF-only export now verifies 92 spherical MAIN functions, 246 Cartesian
molecular AUX functions and 5 occupied orbitals. Exported-input NumPy solve exactly
reproduces Drho-C (relative residual 6.48e-18); this is **not native generation**.
The constrained metric has condition number about 6.03e15 and a roundoff-level
asymmetry (scaled 1.39e-16). Do not silently symmetrize or treat raw coefficient
sensitivity as interchangeable with represented-density error. Evidence is recorded
in `camcasp_native_df_export_evidence.json`.

Lambda=1000 is a finite quadratic charge penalty, not an exact constraint; do not
rescale coefficients afterward. Doo-C instead fits occupied orbital pairs before
tracing and remains a separately named source product. Cartesian q includes all
even-power components, not only s functions. A density matrix may contract B but
must not be sampled directly as a substitute for this fitted density.

Source anchors: `df_Smat.F90:34–68,489–547`, `df_Tmat.F90:815–891`,
`df_monomer.F90:482–507,574–660`. Native integral work should use Libint2 where
applicable, with verified component transforms and effective-coefficient handling.
The old GAMINT reduced-centre path uses an exponent-1e-18 dummy; Libint2's true
unit-shell limit is not a bitwise-identical reference. Matching libraries alone
does not establish matching coefficients, normalization, screening or DF equations.

`stockholder.F90:1451–1545` constructs a FuncExpansion from Doo, Doo_c, Drho or Drho_c.
The selected preset uses **Drho_c**, not the direct OBS density. A sampled AO density
is useful for isolated tests but is a different density-source choice.

For atom a, with old shapes w0 and atomic basis chi, define

```
f_a(r) = rho(r) w0_a(r) / sum_b w0_b(r), if abs(sum_w0) > density_cutoff
         0, otherwise
E_a(r) = exp(min(w_eps * |r-R_a|^2, 230)), if w_eps > 0; otherwise 1
```

The RHS for s functions is the integral of
`chi_k * (f_a + eta*w0_a) * E_a`. For other functions it is
`chi_k * f_a`, additionally multiplied by E_a unless s-block-only is selected.
See `num_integrals.F90:622–870`, **executable** accumulation at 774–843.
Default source denominator cutoff is 1e-36 (`parameters.f90:133`). Density and active
tailed shape samples may be signed; do not silently clip them in this fitting kernel.
Neighbor/grid screening is part of the upstream sampling provider and must be recorded.

The input fitting metric is
`S_kl = integral chi_k chi_l exp(w_eps*r_a^2)`, with weighting only in the s/s block
when requested. See `overlap_integrals.F90:33–56`. The analytic metric has no grid
exponent cap and requires integrable primitive exponent sums. Then:

* multiply s/s entries by `(1+eta)`;
* add Positive-W lambda to eligible diffuse s diagonals, exponent <= max_alpha,
  only when the preceding D_k<0 if auto is enabled;
* solve `S_tilde D_new = RHS` by LU, retaining residual and failure diagnostics;
* extract new shape coefficients using the explicit s-shell map.

See `stockholder.F90:2985–3125,3318–3419,3845–3923`. Positive-W is a soft ridge,
**not** a positivity-constrained fit. No coefficient normalization is implied.

### 4.2 Controller and convergence (explicit-input implementation; production parity pending)

`IsaASweep` is an explicit synchronous sweep boundary. Its `run()` is Gaussian/no-tail
and retains the compatibility name `IsaNoTailSweep`; `run_with_tails()` takes explicit
per-site Func-1 parameters and apply flags. It does not schedule their activation. It owns ordered AtomAux/shape providers and maps, and accepts
`IsaSweepState` plus per-atom `IsaNoTailGrid` objects. All atoms read **only old**
shape expansions during a call. Shape values are clipped at zero according to the
no-tail branch; fitted and projected coefficients are not clipped or renormalized.
Shape-neighbour lists index the ordered sweep atoms (must include the selected
atom), whereas density-neighbour lists index molecular AUX centres. Both are
unique, zero-based and unpadded. Density sites may be empty to screen all density.
Output is a new raw state plus frozen fit diagnostics and selected-old-shape
clipping counts. Failure returns no partial state and mutates no inputs. Input
atomic and shape expansions may be independently supplied for initialization.
This boundary does not provide ONE-GTO initialization, activation, mixing, DIIS,
convergence decisions or any active exponential-tail policy; active-tail states
must not be passed off as Gaussian no-tail reconstruction.

`IsaAController` now wraps these explicit-input sweeps with ordinary-A W convergence
and inspectable initialize/step/run state. Options pin configured fit controls,
thresholds, mixing, limits and explicit per-site tail radii. Optional per-site masks
control tail eligibility and inclusion in MaxDelta/global convergence (default all);
they do not infer nuclear identity or dummy-site semantics. All atom fits and
per-atom convergence tests still run, including excluded sites.
Positive activation thresholds disable the corresponding controls initially;
threshold zero starts them active. Only strict W convergence is implemented, not
Q/RHO selection, DIIS, symmetry, decoupled subiterations or self-consistent tails.
No ONE-GTO/native initialization recipe is implied. The caller supplies initial
atomic/shape coefficients and explicit grids/density. A restart cursor is valid only
with the identical controller bases, density, grids and options; it is not a portable
versioned restart format. All returned states/results are independent snapshots.

Each step fits all atoms synchronously, measures delta and analytic raw shape charge
before mixing, mixes only unconverged shapes when iteration>skip, computes next
active controls and fits tails from the **old** Gaussian shapes, then returns the
committed next state. Full fitted D is not mixed. Saved shape charges deliberately
retain pre-mixing bookkeeping. Active tails can be recomputed even when replacement
is currently off. With tails disabled and no cutoff vector, tail analysis is omitted
explicitly. Strict iteration>tail_iteration_limit activates replacement for the next
sweep; threshold activation is non-latched. A converged sweep stops without forcing
an additional sweep under newly activated controls. Undefined overlap norms fail;
max-iteration nonconvergence is returned distinctly. Production transition capture
and whole-controller parity remain pending, not inferred from analytic unit tests.

All atoms use the old w0 during a sweep; copy w and D to old state only after the
whole sweep (`stockholder.F90:1290–1312`). Do not skip converged atoms. Mixing applies
to unconverged atoms after the configured skip count.

For W convergence:

`delta = abs(1 - abs(w_new^T S w_old) / sqrt((w_new^T S w_new)(w_old^T S w_old)))`

Here S is the **ordinary**, unweighted s-basis overlap. RHO uses the full fitted
atomic density instead. Q is a separate selected test of population change.
`EPS-Q` is not an extra W stopping condition (`3971–4094`). The angle metric cannot
detect pure amplitude rescaling; report population changes independently.

Activation is stateful (`1049–1068,1246–1287`): W-Eps/Positive-W can turn on or off
according to MaxDelta; tail replacement activates at its threshold or after iteration
20. Tail analysis runs during the main iteration. A skips the postconvergence tail
loop unless self-consistent tails are requested (`1572–1600`). Applicable tail loops
freeze fitted tails; A+DF performs coupled DF+ISA updates. Test state transitions,
not only a converged charge.

### 4.3 Tail policies

Func 1 is `A exp(-b r)`; Fit-Type is a separate choice (value/gradient, two values,
or gradient plus tail-charge conservation). Func 2 is another planned policy.

`IsaGaussianShape` now provides an explicit owned co-centred Shape-basis radial
expansion and analytic exterior charge using the effective s-shell contractions.
`fit_tail(r1, previous)` implements only Func-1/Fit-3, returning typed parameters,
status, fallback flag, Gaussian tail charge and IP. It does not choose Slater cutoffs
or schedule activation. Cutoff must be finite and at least the fixed finite-difference
step 1e-8; previous defined Fit-3 exponents must satisfy strict 1<b<4. Invalid slope
without fallback returns a canonical undefined tail and IP=0. A zero exponential
integral also returns undefined instead of pretending an underflowed fit is valid.
This deterministic handling intentionally does not reproduce the reference's stale
saved cross-call A sign gate or its undefined IP assignment. A valid charge-conserving amplitude may be signed.

`sample(points, tail, apply_tail)` replaces values only at r>cutoff when the tail
is active and defined, retaining signed Gaussian interior values. Otherwise it
uses the no-tail max(w,0) branch. It does not implement dummy-site activation rules;
callers select `apply_tail` explicitly. Tail and sample inputs are validated.
These analytic kernels are not yet a native production tail/controller parity gate.

Type 3 uses `b=-w0'(r1)/w0(r1)`, strict `1<b<4`, falling back to a previous valid b.
Amplitude is the ratio of analytic tail integrals Q_w0/Q_exp beyond r1. This conserves
tail charge, **not** continuity at r1 (`stockholder.F90:4660–4717`). IP=b^2/8.
The reference derivative is centered finite difference with step **1e-8**, not the
analytic GTO derivative (`4983–5050`, `parameters.f90:80`). Its three radial points
are translated along the centre's Cartesian z axis, then sampled through contracted
shells before multiplying/summing expansion coefficients. Preserve this order:
ordinary Gaussian sampling must likewise sum each shell's primitive samples before
applying its expansion coefficient. Flattening `(d*c)*exp` or skipping coordinate
round trips changes a cancellation-prone
1e-8 difference (measured iteration-21 tail error 1.54e-8). Offer any improved derivative
as an explicit policy, not a silent parity-path change.

The source has legacy saved-local A/undefined-IP behavior. Do not reproduce undefined
memory semantics; document deterministic handling of invalid fits. Printed A/b/IP
values have only five decimals and cannot support a 1e-8 oracle.

`num_integrals.F90:1394–1458` does not clamp interior negative w in the active-tail
branch; the no-tail branch does clamp. Tail replacement alone does not guarantee
positive finite ratios. Preserve branch semantics for parity and report negative
samples/denominator exclusions. A robust alternate policy must be separately named.
CamCASP's analytic shape population, integrated stockholder population and rescaled
`ISAcharge` differ (`stockholder.F90:4432–4487`); test each under its own name.

## 5. Response and distributed polarizability

### Native wavefunction response provider — private-runtime validated

`NativeResponseProvider` now constructs owned occupied-fast full-OV Coulomb,
exchange and explicit local-kernel operators from a restricted C1 wavefunction.
`isapol_native_response.native_response_from_wavefunction` reuses the existing
shared full-OV frequency solver; it does not run SCF or infer a GRAC protocol.
Convergence is explicitly caller-declared, with density/orthonormality validation,
not a verified SCF residual seal. Kernel policies are explicit no-local,
Slater, Slater/PW92 or Slater/VWN, with explicit exchange/local scales and supplied
numerical grid for ALDA. Direct-OV identity coordinates are not fitted AUX;
optional fitted legs must match the exact native orbital context. No reference
kernel parity follows from a functional name.

Configured build and private-runtime validation pass1571 ISA/FDDS tests, excluding
only the unfinished partition worker's new test. Reviewed fixes address ALDA
component cutoffs, ambient overlap construction and Gaussian/Libint shell safety.
An independent test-oracle correction establishes this Gaussian build's pure-P
z,x,y versus Cartesian x,y,z order; no numerical tolerance changed. Canonical
staging and partition-to-property integration remain pending. This is a native
response prerequisite, not returned atomic alpha/Cn or full-pipeline acceptance.

### Reuse existing Psi4 FDDS machinery first

Parity is the first validation milestone, not a mandate to reimplement existing
Psi4 numerics. The response stage must first adapt or factor reusable machinery
from SAPT(DFT), rather than develop a parallel FDDS stack inside libisapol:

* `psi4/src/psi4/libsapt_solver/fdds_disp.{h,cc}`: existing
  `psi::sapt::FDDS_Dispersion`, DFHelper-based infrastructure, auxiliary metrics,
  density projection, uncoupled amplitudes, and hybrid auxiliary matrices.
* `psi4/driver/procrouting/sapt/sapt_mp2_terms.py::df_fdds_dispersion`:
  existing ALDA-kernel orchestration, coupled frequency-dependent response,
  hybrid/nonhybrid branches and mapped Gauss–Legendre integration.

The first extraction now provides `sapt.fdds_response.solve_fdds_response` for
one system/frequency from supplied J, J-inverse, W and **signed negative** auxiliary
uncoupled response, with optional hybrid K intermediates and R pseudoinverse-transpose.
The SAPT pair loop calls this shared helper twice; its arithmetic, rcond=1e-13,
R sanitation, symmetrization and quadrature/prefactor policy are preserved. Returned
raw and symmetrized arrays are owned copies, labeled `fdds_coulomb_auxiliary`, not
CamCASP C_DF. For ordinary unregularized Coulomb fitting, coefficient response is
J-inverse * auxiliary-response * J-inverse; constrained fits and differing kernels
require a separate mapping audit. The first solve extraction passed 32 focused tests
and unchanged sapt-dft1/dft2/dft-api/dft-lrc inputs. That evidence predates the next
construction/ALDA extraction below and does not certify those newer changes.

The subsequent construction/ALDA extraction passed 445 combined tests, all four
unchanged SAPT input regressions and four GRAC tests. Portable evidence is
`psi4_fdds_monomer_evidence.json`. It adds `core.FDDS_Monomer` through shared one-/
two-system C++ construction: only A spaces/tensors are built for one system; the old
pair constructor and numerical bodies remain shared. Its explicit orbital/energy
inputs are copied, getters return copies, and C1/finite/dimension/positive-gap checks
precede integral generation. Hybrid nov<naux fails explicitly; no alternate rank
policy is substituted. `FDDSMonomerResponse` builds the existing gridless ALDA W once
and exposes `at_frequency`, using the same coupling helper and moved `_compute_fxc`
as SAPT. This constructs Psi4 FDDS intermediates from supplied orbitals/bases, NOT
native SCF, CamCASP constrained-DF response or end-to-end parity. The current pair
interface takes A/B orbital/energy caches and its Python driver integrates a
pair dispersion energy. Factor a reusable single-system, per-frequency response
provider before the A/B energy contraction; do not require a dummy partner or a
full SAPT calculation for an atomic-property request. Preserve the existing SAPT
entry point and its results. Shared numerical functionality belongs in a reusable
layer, with libisapol supplying partitioned multipoles and property contractions.

Audit the representation boundary before identifying an FDDS response with the
CamCASP `C_DF` below: auxiliary metric factors, signs, spin/frequency factors,
normalization, fitting regularization, kernel/exact-exchange policy, asymptotic
correction and orbital data must be explicit. The existing SAPT route is not
assumed numerically identical to CamCASP. Add only the missing reference-policy
pieces justified by controlled matched-input comparisons; document any remaining
approximation difference instead of silently substituting one method for another.

Reuse quadrature infrastructure where compatible while preserving explicit
nodes/weights for reference replay. SAPT's current Python loop uses `leggauss`,
`omega=lambda*(1-t)/(1+t)` and its mapping Jacobian, with default lambda=0.3;
the modern ISA-Pol reference uses beta=0.5. Account for the integral prefactor
exactly once. Standalone CasimirGrid fixtures remain compatibility evidence, not
a requirement to replace SAPT quadrature or duplicate its response solver.

Acceptance must include unchanged SAPT(DFT) regression results (hybrid and
nonhybrid), monomer/per-frequency adapter tests, matched-input response comparisons,
residual/reciprocity and frequency-limit checks, then partitioned-property tests
for each supported strategy. No full FDDS parity claim follows from code reuse
alone. The equations below define the reference contract to map and validate.

Pinned response-capture dispatch: `SET NEW-PROP` only configures kernel options;
the new driver requires `NEW-PROP` inside the polarizability block. The archived
blocks therefore select old/internal Hessians. Record the old driver's actual
`kernel_alda`, CxKernel and CxFunctional separately: a NEW-PROP kernel setting
must not be taken as proof of zero effective exact exchange in the old driver.
For the archived point-response settings now used in the adapted total-only case,
the old driver resolves kernel_alda=false and CxKernel=CxFunctional=0.25. It scales
the internal local kernel by 4*(1-0.25) and retains hybrid two-electron terms.
The reference's fitted-pair normalization warnings test D*Iint, not true MO C^T S C. Do not substitute the new driver
for easier capture. The numerical kernel selects Slater/PW92 for its own ALDA
selector, and Slater exchange-only for ALDAX. Record that resolved selector rather
than inferring it from old kernel_alda. The shared Psi4 helper preserves its
gridless LDA/VWN policy. Capture actual fitted D,
auxiliary kernel and OVOV Coulomb before asserting a mapping through D J D^T.
OV indexing is occupied-fast; old-driver omega2=-xi^2 on the imaginary axis, and
Quad10 includes a static evaluation plus ten imaginary nodes.

Schema3 read-only capture records NN-derived OV rows at their existing file-write
producer, kernel matrices before producer release, and fit/filename metadata at
projection without forcing operand residency. Preserve complete producer epochs:
the kernel-projection D snapshot must not be silently replaced by a later D used
for response. Full C/E and actual diagonal-energy uses are recorded separately.
Fresh direct OV solves remain unsupported by this observer.

The exported-input oracle `replay_response.py` checks the unsymmetrized equations
`(H2 H1 - omega2 I) Y = -4 H2 D` and `C_DF = D^T Y`. The measured 11-frequency v4
case passes the unchanged scaled 1e-9 gate (maximum response error 5.6443e-19),
with identical kernel/response D snapshots. This certifies supplied-input equation
replay only, not native transition fitting or shared Psi4 FDDS kernel equivalence.
Schema5 subsequently exports the consumed DF OVOV/VVOO tensors; quadrature weights
and full OO/VV/cache-generation provenance remain unexported.

A separate lambda1/eta0 OV diagnostic uses native explicit-basis q/J/B, supplied
full C, occupied-fast `Cocc^T B[k] Cvir` without spin factors, and a NumPy solve of
`(J + lambda q q^T) D^T = T_OV^T`. It fails forward parity: coefficient scaled error
2.3020e-5 and pointwise represented-transition-density error2.2143e-6, despite a
1.5410e-17 backward residual and condition(A)8.0861e12. This does not establish a
production transition-fitting API or native response. Keep the unchanged1e-9 gates;
raw reference A/RHS capture is needed to isolate their contributions. Finite lambda
must not be replaced by exact neutrality or post-fit charge rescaling.

Schema4 adds a bounded pre-LU NN-fit observer (v5 traced/untraced artifact identity
passed; raw-fit replay and controlled substitution diagnostics completed). Its saved
context contains only scalar/string metadata from the current fit parameters;
the naturally read A and valid AUX-by-packed-pair RHS block are observed before
LU mutation. Unused RHS allocation columns may be undefined and must not be read.
Full NN packing uses nmos*(nmos+1)/2; off-diagonal RHS is C_i^T B_k C_j with no
2/sqrt(2) factor, while only diagonal pairs receive lambda*q. Actual incoming
operands—not A reconstructed from a recipe or RHS reconstructed from A*D—are
required for contribution-isolation evidence. The v5 full NN reference-input
solve reproduces outgoing OV coefficients exactly, with backward residual1.4376e-17;
this validates the captured-operand control, not native-integral forward parity.
The matched OV-only RHS-batch reference control also passes (coefficient scaled
2.4718e-11, pointwise density2.2581e-12), but is not bit-identical to the full NN batch.
Controlled substitutions fail with either native A alone (pointwise density9.8210e-7)
or native RHS alone (2.2888e-6). Both inputs contribute; coefficient and pointwise
error dominance differ. These diagnostics do not justify regularization, rescaling,
a new reference, or changing the1e-9 forward gates.

Executable DF and conventional branches in `densfit_prop.F90:1490–1789` give

```
H1 = Delta_e + 4(ar|bs) - CxKernel * [(ab|rs) + (as|br)] + K_local
H2 = Delta_e           - CxKernel * [(ab|rs) - (as|br)]
```

The old spec copied an incorrect nearby comment. **The H1 exchange sum is correct.**
Do not change it to a difference or attribute prototype errors to the sum.
CxKernel and CxFunctional need not be identical (`1060–1127`). The internal hybrid
branch has `K_local=4(1-CxKernel)K_ALDA`, with a separate kernel_alda branch.
`prop_utilities.F90:329–441` projects the auxiliary kernel with Dov or Dov_c. Pin the
DF coefficients/metric, regularization and kernel policy; a direct MO-grid kernel
is not automatically the same approximation.

The archived point case has separate fitting policies: response/kernel projection
uses constrained λ1 Dov_c, whereas ordinary λ0 Dov/Doo/Dvv supply the DF-produced
Coulomb/exchange tensors and kernel density. The latter density is
`2 sum_i Doo[u(i,i),:]`, not Drho-C or an AO-density substitution. Actual Coulomb
construction is `Dov0*(J*Dov0^T)`; VVOO is `Dvv0*(J*Doo0^T)`, without robust-DF
corrections. Do not demand equality to the λ1 response-D construction.

Schema5 consumer capture records OVOV, VVOO, raw projected kernel and numerical
kernel selector. V6 traced/untraced artifacts are identical; H1/H2 reconstruct
exactly, AUX projection differs8.4134e-15, and all11 reconstructed-Hessian CDFs pass
at maximum scaled error5.6443e-19. The observed selector is ALDA (Slater/PW92),
cutoff1e-8, batch1000 and local-kernel multiplier3. With occupied-fast p(a,r), the actual
exchange accesses are `OVOV[p(b,r),p(a,s)]` and `VVOO[u(r,s),u(a,b)]`, using
virtual-relative r,s and packed OO/VV indices, with no multiplicity factors.
Independent replay must preserve these orientations without assuming tensor
symmetry. OO/VV coefficient and integral-cache generation provenance remain
incomplete; numerical D J D^T agreement is a separate check, not a substitute.

Pending schema6/v7 revision adds naturally resident OO/VV rows and their existing
setter metadata, plus the actual AUX1 kernel-density vector immediately after
`Rho_Doo2FuncExpansion`'s ordered diagonal sum and factor2, before its existing
close. A scalar-only context brackets this producer; KERNEL_SOURCE records the
exact density event serial. The strict reader commits subset generations only at
successful setters, checks full-NN VV offsets, freezes density-source associations
and reports ordered-sum coefficient error without repairing it. These are subset/
density links only: full parent-solve and integral-cache producer generations are
still incomplete and explicitly unaccepted. No observer reference-object I/O,
retention or numerical policy change is introduced. Schema6 safety/reader tests
pass51 tests in1.49s. Fresh v7 build and both runs pass:4793 events and all35
reference artifacts byte-identical traced/untraced. Strict density, raw-fit,
Hessian and11-frequency reconstruction initially failed on a metadata assumption:
cached OV subsets are retagged NN(type1)->OV(type3) without new coefficient rows.
Pinned compare_type_df_parameters compares df_type to itself, while the caller's
final setter runs even after skipping fitting. The reader now records only this
narrow observed retag separately, preserving original generation metadata; other
unexplained changes still fail closed. The correction passes52 tests in1.47s and
strict v7 replay: five subset generations, two retags and exact ordered kernel-density
reconstruction. Raw NN fitting and H1/H2 remain exact; all11 CDFs pass at maximum
scaled5.6443e-19. `camcasp_subset_density_evidence.json` retains this supplied-control
evidence. Full parent-solve and integral-cache generation provenance remains open.

Pending schema7/v8 adds scalar-only begin/success events around actual fresh NN
solves at lambda0 andlambda1. Subset generations must follow a successful matching
parent solve; its original identity, parameters and solve ID remain frozen across
later cache retags/overwrites. This does not add raw lambda0 A/RHS capture or change
solver arithmetic. Parent/safety/reader tests pass61 tests in1.51s. V8 build and
strict replay pass:4797 events, all35 artifacts byte-identical traced/untraced,
two successful NN solves binding five subsets. Density, raw NN fit and H1/H2
reconstruct exactly; all11 CDFs pass at maximum scaled5.6443e-19. Portable evidence:
`camcasp_parent_solve_evidence.json`. Integral-cache construction generations remain
a separate uncompleted gate; this does not establish native acceptance.

Shared FDDS hybrid equivalence requires independent supplied-operator tests, not
just monomer/pair agreement. Audit identified separate frequency/exchange ordering
and auxiliary-closure conditions; full column rank alone does not prove full-OV
equivalence. Do not modify inherited SAPT arithmetic or rank policies based on this
audit without independent tests and explicit review.

An effective-coordinate diagnostic can isolate the existing coupling helper:
retain actual V, X, Y and H2, define H10=Delta+4V-a(X+Y), and form
`Ueff=D1^T solve(H2*H10+xi^2 I,-4 H2 D1)`. Supply artificial identity metric/inverse,
`W=(1-a)*Kaux`, and no additional hybrid correction to `solve_fdds_response`.
The actual exchange a=.25 is already in H10/H2; the helper's x_alpha=0 does NOT
mean zero-exchange physics. Ueff is not ordinary uncoupled FDDS. These diagnostic
coefficient coordinates must be stated separately from the helper's fixed
Coulomb-auxiliary representation label. Compare raw outputs and record retained
pseudoinverse ranks; no native construction or production-representation claim
follows. The effective bridge passes all11 frequencies (maximum scaled error
5.7038e-13); all246 modes survive the inherited1e-13 cutoff, with maximum denominator
condition980. The84-test suite passes, including supplied-operator noncommuting and
leaking counterexamples and commuting/closed controls. This does not measure native
provider equivalence. A separate captured-operator full-OV identity-coordinate
hybrid diagnostic now fails full-OV parity: maximum coefficient-response scaled
error2.1918e-3, versus passing reference control2.0572e-14. All435 modes are retained
and maximum denominator condition is3.9692; literal fixed-point residual9.8865e-4
contrasts with physical-identity residual1.1653e-15. This captured-operator result
is not a native-provider test, but projection or truncation cannot explain it.
Legacy SAPT arithmetic remains unchanged. The user explicitly approved a separate
shared full-OV route; it must not be silently substituted for the inherited route
or presented as resolving native-fit gates.

`FDDSFullOVResponse` in the shared `fdds_response.py` now accepts owned, finite
`h1_baseline`, `h2`, `transition_legs` D and remaining `coupling` matrices, with a
required coordinate representation. It implements H1=H1_baseline+4D coupling D^T,
using an unregularized full-OV baseline solve and the existing auxiliary coupling
helper. Singular baselines raise without fallback; the coupling step preserves
the inherited1e-13 pseudoinverse policy. Nonsingular/untruncated coupling is required
for exact full-coupled equivalence. Neither symmetry nor an OV ordering is inferred.

`at_frequency(omega)` returns `FDDSFullOVFrequencyResponse` with owned raw baseline
and coupled arrays, explicit representation and method `full_ov_effective_baseline`.
The baseline is exchange-containing/effective, not ordinary uncoupled FDDS. No
symmetrized result is used or relabeled. Caller-selected representations are
`fitted_density_coefficients`, `fdds_coulomb_auxiliary` or
`supplied_transition_leg_coordinates`; declarations do not perform transformations.
No SCF, integral/fit/kernel generation or partner system is introduced. Existing
SAPT callers continue to use the unchanged legacy route. The combined515-test suite
and11-frequency supplied-operator provider comparison pass (maximum scaled error
5.7038e-13, full246-mode coupling rank). Advisory review found no must-fix algebra
or ownership bug. Four unchanged SAPT fixtures and4 GRAC tests pass, as do81
focused tests including added overflow/noncommuting-coupling cases. Final validation
passes520 combined tests and all11 provider frequencies at maximum scaled error
5.7038e-13, explicitly requiring full246-mode coupling rank. Evidence is in
`psi4_full_ov_provider_evidence.json`. This accepts the supplied-operator route,
not native construction or the unchanged inherited-hybrid method's full-OV parity.
NumPy/BLAS validation failures may surface as ValueError,
FloatingPointError or LinAlgError; none triggers an automatic fallback.

At imaginary frequency i*xi:

```
(H2 H1 + xi^2 I) X = -4 H2 D
C_DF = D^T X
```

D has shape `(nov,naux)`; solve nov equations with naux RHS. Reuse H2H1 and -4H2D
(`densfit_prop.F90:492–630,940–977`). The reduced system is a source-compatible choice,
not a proof of superior conditioning. Validate solve residuals, reciprocity and
zero/large-frequency limits; retain a robust fallback with explicit diagnostics.

Partitioned multipoles use the **molecular AUX**, not AtomAux:

`Q[a,k,t] = integral R_t(r-R_a) * w_a/sum(w) * chi_k(r) dr`

`alpha[a,t,b,u](i*xi) = - Q[a,k,t] C_DF[k,l] Q[b,l,u]`.

See `polarizability.F90:1350–1580`. Store labeled spherical tensors, component order
`00,10,11c,11s,20,...`; rank L has (L+1)^2 components. Record origins/frames explicitly.
Check transpose symmetry, charge-flow sum rules and molecular multipole recovery
with translations, not just sums of local dipole blocks. Convert conventions once
at documented boundaries. Support explicit frequencies as well as a quadrature.

### Supplied-partition multipole and response contraction boundary

`IsaPartitionedMultipoles` now integrates Q in C++ from an explicit molecular AUX
basis and per-site `IsaMultipoleSite`/`IsaMultipoleSamples`. Each site supplies a
unique label, origin in bohr, rank 0–4, points, quadrature weights, already
screened/tail-processed shape and shape sum, and a batch-wide AUX neighbour list.
Neighbours are unique zero-based indices without padding; empty screens all AUX.
Signed weights and partition ratios are retained without clipping or rescaling.
Denominators with absolute value <= the explicit cutoff (default 1e-36) are excluded;
per-site exclusion and negative-ratio counts are returned. Separate per-site grids
are allowed; the caller remains responsible for consistent partitions/quadratures.

Rows concatenate sites and Racah regular real components `00,10,11c,11s,...`,
columns retain molecular-AUX function order. The harmonic evaluation reuses the
existing independent solid-harmonic recurrence with an explicit conversion from
DALTON basis order. The result labels offsets, ranks, components, site order,
origins, global Cartesian frame, atomic units and required caller provenance.
No local-frame rotation, native partition construction or convergence is inferred.

`IsaDistributedResponse` accepts explicit nonnegative imaginary frequencies and
owned coefficient-response inputs only under the required representation declaration
`fitted_density_coefficients`. It computes raw `-Q C_DF Q^T` in C++, retaining both
site/component axes, frequencies, required response provenance and measured scaled
reciprocity defects. No metric/sign/spin transformation, symmetrization or response
solve occurs. Frequencies preserve supplied order; no quadrature is inferred.
Getters return independent snapshots, and nonfinite computations fail descriptively.

The staged build and final expanded regression suite pass **589 tests in 4.08 s**,
including 45 new supplied-partition tests: independent rank-2 polynomials, rank-3/4
Legendre oracles, rank addition identities, Gaussian integrated moments, rigid
translation/rotation, charge-flow translation and sum rules, raw nonsymmetric
contractions, ownership, screening and malformed-input checks. These establish
analytic/invariant and supplied-operator software checks only. Production Q/alpha
capture/replay, native providers, arbitrary local frames, and wavefunction-to-property
acceptance remain unvalidated; no public property task is registered by this addition.
Independent advisory review found no must-fix implementation defect. Its requested
finite-input overflow, null/symmetry, partial-screening and Cartesian-p AUX cases
are included in the final passing suite. This advisory result does not establish
production Q/alpha or end-to-end acceptance.

## 6. Localization and point-response refinement

### Independent multipole translation/rotation kernels

`isa_multipole_translation(L,d)` and `isa_multipole_rotation(L,F)` now return fresh
real Racah matrices through rank L=0–4 in `00,10,11c,11s,...` order. Translation is
defined by `R(x+d)=T(d)R(x)`, so moving moments from source a to target b uses
`d=R_a-R_b`. Rotation is `R(F*x)=D(F)R(x)`, with a finite proper orthogonal
local-to-global Cartesian frame. Coordinates/displacements are in bohr. These
functions do not localize or discard tensor components.

The implementation generates Cartesian harmonic coefficients from Legendre
polynomials, substitutes affine coordinate polynomials, and decomposes homogeneous
terms in the harmonic basis with an explicitly checked polynomial residual. It
uses no ORIENT source or external translation table. All rank pairs through (4,4)
remain representable. Invalid ranks/frames and nonfinite computed transformations
fail rather than truncate or repair inputs.

Build/stage/byte comparison and **665 tests in 4.29 s** pass, including 26 new
independent polynomial/group/rotation-covariance, axial-binomial, full-rank transfer
nullspace and invalid/overflow tests. Advisory review found no must-fix defect;
noncommuting rotations, direct Gaussian-evaluator checks, large representable
translations and frame-boundary tests were added. Portable evidence is
`tests/pytests/data_isapol/psi4_multipole_transform_evidence.json`. This establishes mathematical transforms,
not ORIENT localization parity. A proposed LS draft was rejected and moved outside
the source tree: its transfer equations conserve the molecular tensor, but the
source audit withdrew an unsupported LS attribution and high-rank oracle. No
localization capability is registered or accepted from it. The paper blocker is
resolved as **not required—wrong method identified**: the targeted oracle uses LW,
not LS. User guidance `orient_replacement.md` identifies our existing LGPLv3
implementation at `camcasp_psi4` commit5449bd1a01c73f45c307b36b006e264c1e43b994.
That code, not ORIENT source or the rejected LS transfer, is the port source.
Local ORIENT source is GPLv3, must not be accessed for this port, and must not be
transliterated into LGPL Psi4 code. This prohibition is **specific to ORIENT** and
does not extend to CamCASP, which is MIT and transcodable; see the CamCASP licensing
subsection in section 7.

The supplied-input LW port now provides `isa_localize_lw` and owned
`IsaLocalizedResponse` results. It uses an explicit undirected bond graph, negative
graph Laplacian and componentwise Moore–Penrose inverse, rank0–3 (16component)
working blocks, symmetric bond-flow transfers, and rank1–3 (15component) local
output. Eigenvalue cutoff1e-4, transfer omission threshold1e-7, all four
Moore–Penrose checks and five postcondition residuals are retained. Results expose
frequency, positions, local tensors, transfers, omissions and `refined_pairs`:
the latter is the LW workspace, NOT PFIT-refined output. Python getters return
owned copies. Bounds are256sites,1million retained/pending transfers and a
conservative768MiB native workspace; caller inputs/getter copies are additional.
No rank4 working-space extension or automatic bond derivation is included.

Graph tests passed24 cases before the translation guard: the new rank3 kernel
matches the permitted old LGPL kernel across13 displacements/all16x16 entries at
maxabs4.440892098500626e-16, with unchanged literal fixtures. Transfer validation
then passed72 graph/translation/transfer cases; independent review found no
must-fix defect. The combined ISA/FDDS suite passed1224 tests. The previously
disputed3/8=0.375 value does not justify reviving the rejected LS draft.

The default production postcondition remains1e-6, distinct from a hermetic
NL4→L3 output comparison at1e-11 over675 entries; neither is iterative convergence.
The supplied historical water input ALREADY violates charge sums by7.011e-4 in
retained components, so the default correctly rejects it. User explicitly
authorized a separate historical-fixture diagnostic at1e-3, NOT a new production
default or a forward-profile waiver. After exact top-left25→16 truncation and H1
rotation using `isa_multipole_rotation`, all675 raw local-output entries match the
external unrefined L3 fixture: per-site maxabs O6.252776074688882e-13,
H1 5.60440582830779e-13, H2 5.089262344881718e-13. The27 fixture/diagnostic tests
include strict-default rejection and the negative unrotated-H1 control. This
establishes historical supplied-input tensor agreement with an explicit failed
production postcondition, not native-wavefunction or PFIT acceptance. Fixture
numbers are immutable extracted external output, not generated from this port;
independent fixture/provenance review found no numerical must-fix; its stale
README finding was corrected. Bounded measured evidence is
`tests/pytests/data_isapol/psi4_lw_localization_evidence.json`.

ORIENT, not CamCASP, implements the localization invoked by `bin/localize.py`.
LW and LS are distinct policies. Prototype dipole agreement is useful evidence but
does not validate rank-4 translations, general graphs or higher multipole sum rules.
Re-test migrated code on full tensors with exact graph, axes, site order and ORIENT
inputs. Do not confuse a sum-rule test tolerance with an iterative convergence setting.
Equal elements alone do not establish equivalent sites: COPY requires compatible
site types and local frames. Permit explicit graphs, frames and parameter models.

PFIT target v_ij(i*xi) must be defined as a point-charge response, including sign,
units, charge and generation from the fitted propagator. It must not silently become
a response reconstructed from a truncated multipole model.

`tools/process_data.F90:2024–2149` selects free static tensor entries by absolute
component cutoff, per-site model ranks and site types. Existing .pdef files are not
automatically overwritten. Keep explicit user models and automatic generation distinct.

`pfit/process.F90:118–270` forms the Gram matrix and RHS over **within-batch pairs
j<=i**, including diagonals, no extra off-diagonal multiplicity. Add the penalty
matrix P and P*anchor, then DSYSV. Fixed parameters are eliminated consistently.
Batching must preserve the intended pair set. A 2000-point cloud has 2,001,000 pairs;
20 free parameters require about 420 million Gram products per frequency, not
microseconds. Memory includes fields/targets as well as the small Gram matrix.
Normal equations square the design condition number; retain an alternative solver
for the same objective and report residual/condition diagnostics.

### Localization/PFIT gate: three separable legs, with measured floors

CamCASP's `bin/localize.py` chain splits into three legs that must be gated separately,
because only one of them touches ORIENT:

* **Leg A - localization** (`NL4 .pol -> L3 .pol`). Performed by ORIENT, whose source is
  GPLv3 and may not be read. **Now implemented and gated** - see "Leg A is closed" below.
* **Leg B - `process` + `pfit`** (deck + `.p2p` -> `H2O_ref_wt4_L3_{tag}.pol`). MIT CamCASP,
  hermetic, no ORIENT involvement, and **already reproduced**: all 104 parameters at all 11
  frequencies to `2.52e-08` absolute / `6.03e-09` relative, against a gate of `3e-08` /
  `1e-08`. Deck anchors match `H2O_L3_0f10.pol` to the 8-significant-digit deck write
  (`4.91e-06` / `4.69e-08`).
* **Leg C - `process` + `casimir`**, i.e. the dispersion table; see section 7.

*Precision floor on leg A.* `localize.py` defaults to `--format NEW/B`, so the reference
localization consumed the `(3x,<ncols>(e15.7,1x))` 7-significant-digit view, **not** the
17-digit format-A file. The amplification is visible in the recorded output: elements that
symmetry forces to zero sit at `~1e-07` (O `10-11c` = `-1.13753e-07`). Leg A is therefore
floored near `1e-07`, for a different structural reason than the format-A `3.24e-13` floor
that anchors section 5's frequency-dependent comparison. Do not state a `1e-9` gate on it,
and do not "fix" it by re-running localization off format A - that would no longer be the
reference chain that produced `H2O_L3_0f10.pol` and the dispersion input deck.

#### Leg A is closed: recorded file-in/file-out agreement at all eleven indices

`isa_localize_lw` reproduces the reference localization for every recorded index, driven
only from committed fixture literals (`tests/pytests/test_isapol_lw_leg_a.py`, 40 tests).
No ORIENT source was read, quoted or transliterated; the recipe was taken from CamCASP's
own MIT-licensed input template `H2O.ornt` (read -> `Limit all rank 3` -> `Sum-rule test
1e-7` -> `Localise LW test 1e-7 Limit 3` -> `Edit ... #include H2O.axes` -> write), and the
bond graph `O-H1`, `O-H2` and the site frames (O, H2 identity; H1 180 degrees about z) from
the committed `H2O.axes` and manifest.

675 entries per index, 7,425 total, in each site's local frame via
`isa_multipole_rotation(3, frame)[1:16,1:16]` (`local_to_global_columns`):

| index | `omega` | worst \|delta\| vs recorded L3 | supplied sum-rule defect | `off_site` |
| --- | --- | --- | --- | --- |
| 0 | 0 | `6.25e-13` | `7.011e-04` | `2.20e-11` |
| 1 | 0.0066096 | `8.24e-13` | `7.011e-04` | `2.20e-11` |
| 2 | 0.0361748 | `5.68e-13` | `7.001e-04` | `2.20e-11` |
| 3 | 0.0954474 | `6.25e-13` | `6.940e-04` | `2.18e-11` |
| 4 | 0.197644 | `7.39e-13` | `6.727e-04` | `2.13e-11` |
| 5 | 0.370417 | `5.20e-13` | `6.185e-04` | `1.95e-11` |
| 6 | 0.674915 | `6.54e-13` | `5.104e-04` | `1.54e-11` |
| 7 | 1.2649 | `5.19e-13` | `3.4365e-04` | `9.05e-12` |
| 8 | 2.61924 | `5.23e-13` | `1.7116e-04` | `4.11e-12` |
| 9 | 6.91089 | `4.99e-13` | `6.008e-05` | `9.83e-13` |
| 10 | 37.8238 | `4.99e-13` | `2.990e-06` | `3.93e-14` |

Gate: `atol = 1e-11, rtol = 0`; worst observed `8.24e-13`. This sits far below the `1e-07`
format-B floor because both sides of the comparison are the *same* 7-digit view - the
reference tool and `isa_localize_lw` read the identical `H2O_NL4_{tag}.pol` bytes, so the
printing loss is common-mode and cancels. The `1e-07` floor still governs the moment leg A
is fed a natively produced NL4 file, which is where it must be quoted; it is not a licence
to quote `1e-9` there.

*What the acceptance criterion is, and is not.* It is an output comparison against the
recorded L3 blocks. It is **not** a sum-rule postcondition on the input, for a reason
established from the reference tool's own printed *log* (`H2O_L3_000.out`, an output
artifact, not source): under `Sum-rule test 1e-7` it printed per-site failures up to
`1.215e-4` and then localized the data anyway. `Sum-rule test` is a diagnostic report, not
an admission gate. Our own charge-flow defect table has the same symmetry-allowed component
set as that printed table (O: `00,10,20,22c,30,32c`; H1/H2 additionally `11c,21c,31c,33c`)
but different magnitudes, so the two are differently normalized statements of the same rule;
they are not interchangeable and the printed `1.215e-4` must not be quoted as ours.

*Consequently `IsaLocalizationResiduals` now separates two things it used to conflate.*
Algorithm-controlled, always gated at `residual_tolerance` (default `1e-6`): `off_site`,
`reciprocity`, `molecular_sum`, and the new `charge_sum_transport`. Supplied-input quality,
gated separately by the new `input_sum_rule_tolerance`: `input_sum_rule` (the supplied
data's own defect), and `charge_sum`/`local_charge`, which merely reproduce it. LW's bond
transfers are antisymmetric in the two site slots, so the charge-flow sums
`sum_b alpha(a,b;t,00)` and `sum_b alpha(b,a;00,t)` are exact invariants: measured
`<= 2.2e-16` across all eleven indices. The old single gate was therefore rejecting every
recorded reference input on a property of its *producer* while the four residuals LW
actually owns passed by five orders of margin.

`input_sum_rule_tolerance` is negative by default (inherit `residual_tolerance`, i.e.
bit-identical legacy behaviour); positive finite sets an explicit separate threshold;
infinity measures and reports without gating. The driver exposes this as
`residual_policy='reported_input_sum_rule'`, which holds the algorithm at the unmodified
`1e-6`, warns with the measured defect, and records
`production_postcondition_passed=False` alongside a new
`algorithm_postcondition_passed=True`. **No comparison tolerance was relaxed** to close
leg A, and one consequence is worth recording: the authorized static-only `1e-3`
`historical_water_diagnostic` waiver is no longer needed for the reference input - index 0
now passes under the stricter `1e-6` algorithm gate. The waiver is retained only for the
existing pinned diagnostic and was **not** extended to the ten dynamic nodes.

*Still open on leg A.* The gate consumes recorded NL4 literals, so it certifies the
localization transform, not native production of its input. The four recorded L3 header
`FREQSQ` disagreements at nodes 7-10 remain as recorded and are untouched.

*Why leg B cannot be gated on high-rank parameters alone.* The design matrix over the
recorded 500-point cloud has `cond2(A) = 7.644e+04`, and the normal-equations operator
actually solved has `cond2(A^T A + strength) = 5.213e+09`; the reference's own high-rank
numbers are whatever `DSYSV` returned at that conditioning. Rank-resolved parameter
uncertainties at the static node span `8.48e-03` (1-1) to `4.34e+01` (3-3), collapsing to
`4.71e-06 .. 5.36e-02` at the highest frequency `w2 = -1430.637`, so the ill-conditioning is
a static/low-frequency phenomenon. Gate the objective and the identifiable blocks, not the
individual high-rank parameters.

Weight scheme 4 applies to rank<=1 pairs with coefficient/(1+xi^2), zero otherwise
(`process_data.F90:1810–1876`). Modern default 1e-3 is not established for the old
water archive. Soft anchors do not guarantee positivity or sum rules.
MATRIX penalties exist alongside scalar, named and LC penalties. Nonzero LC anchors
have a source inconsistency; do not claim general support without a tested policy.
For a true `s(t^T p-a)^2` penalty use `s*t*t^T` and `s*a*t`.

## 7. Dispersion coverage and quadrature

`CasimirGrid` reproduces the standalone tabulated frequency rule (even orders 2–10),
whose default beta is 0.3. The **ISA-Pol preset beta is 0.5**. Response quadrature
`quadrature.f90::makequad` Newton-iterates roots and differs from the tabulated rule
by ~1e-13; record actual nodes/weights instead of silently replacing them.
Index 0 is static and has zero integration weight. `cp_weight` includes 1/(2*pi),
so isotropic C6 is `6*sum(cp_weight*alpha_A*alpha_B)`.

For scalar rank-(l,l) local responses, complete isotropic terms require:

* C6: (1,1);
* C8: (1,2),(2,1);
* C10: (1,3),(2,2),(3,1);
* C12: (1,4),(2,3),(3,2),(4,1).

Thus uniform L2 gives partial C10 and no local isotropic C12; L3 omits part of C12.
O=L2/H=L1 is a deliberately truncated water model. Rank-4 distributed input does
not restore terms discarded during model construction. Report included/missing rank
pairs and distinguish **zero within model** from **missing physical contributions**.

General anisotropic terms satisfy `n=la+la'+lb+lb'+2`. With ranks>=1, unrestricted
C10/C12 can require individual ranks 5/7; the CamCASP rank-4 coefficient tables do
not cover that unrestricted expansion. Initial anisotropic support therefore means
CamCASP-compatible **within table/model coverage**, not complete general C12.

The 393 blocks/4673 terms in `recoupling_tables` are coefficient data, not a complete
engine. The shared parser generating both fixture and compiled data tests consistency;
it is not an independent anisotropic oracle. Add independent CG/rotation, exchange,
isotropic-limit and tensor-to-C_n tests before property parity claims.

Important upstream issue: `read_cg:309` and `recouple:501` both skip recoupling rank
pairs with j1+j2>6, although c12code requires (4,4),(4,3),(3,4). Corresponding storage
(`alpha_c`, casimir.f90) has no initializer, so those blocks carry static-storage zero.
Refinement from the gate 8 audit: the recorded L3 water reference **cannot discriminate
zero from undefined** for those blocks, because its rank-4 input alpha is itself zero, so
every dependent c11code/c12code term vanishes either way. The caution therefore stands
unchanged for a rank-4 model with nonzero rank-4 response: do not reproduce undefined
values or treat this executable as an unqualified complete-C12 oracle. Use independently
derived isotropic rank-4 identities and a documented corrected reference for that coverage.

### Reference C_n table: measured reproduction and the remaining gap

The reference dispersion step is a hermetic file-in/file-out oracle needing no CamCASP
build and no ORIENT: `H2O_ref_wt4_L3_casimir.data` -> (casimir) -> `H2O_ref_wt4_L3_C12.pot`.
`.pi/audit/gate8-casimir-reference-objective.py` transcodes `frequencies`
(casimir.f90:439-463), `read_cg` (:262-330), `recouple` (:486-531), `cpint` (:397-435) and
`Cn` (:535-649) plus `c6code..c12code`, and reproduces **every printed coefficient** of the
reference `.pot` - all three site-type pairs, C6 through C12, 10,457 values - at
`4.998e-07` relative, which is the half-ulp of the reference's own `g15.7` write. 30,791
structural `0.0` placeholders are all satisfied, with 0 missing rows and 0 unexplained
extras, and the imaginary residue asserted real by `cpint` is exactly `0.0` over the whole
table. Two structural properties of the reference follow and constrain any comparison:

* **The reference truncates J at 8.** `C` is dimensioned `C(6:12,81,81,0:10)` and
  c11code/c12code do populate J=9,10, but the print loop is `do k=0,8` (casimir.f90:591).
  411 rows above the reference's own `1e-6` threshold (67/122/222 by pair, up to `3.1e+03`)
  are computed and never written. Emit them if wanted, gate them only on internal
  consistency, and never suppress them silently to make a row-set comparison pass.
* **The write ceiling is `1e-6`.** `g15.7` with a `>1d-6` magnitude threshold means nothing
  below `1e-6` absolute is recorded at all. `1e-6` relative is the right and final tolerance
  here; do not restate it in absolute terms, since these coefficients span `1e-06` to
  `1.5e+05` in one table.

Both shipped kernels this needs are now verified against CamCASP source rather than against
a previous run of our own code. An independent fresh parse of `c6code.f90..c12code.f90`
matches the committed `recoupling_tables.{h,cc}` + `recoupling_data.inc` through
`isapol_recoupling_blocks` on **393 blocks both sides, key sets equal, 4673 terms compared,
0 rank-label mismatches, worst relative coefficient deviation 0.0** - confirming the
"393 blocks/4673 terms" figure above from the source side. `CasimirGrid(10, 0.5)` matches
the independent transcode with `max |domega| = 0.0` and `max rel dev` on `cp_weight` of
`0.0`, and reproduces the recorded `FREQ2` header list to its 7-digit write (worst
`|dev| 4.62e-06` on `-47.76034`). The grid is **generated, not captured**; no captured
frequency list is needed and none should be introduced.

The remaining gap is exactly two items, neither of them a reference problem. First, there is
no `alpha_u -> alpha_c` recoupling kernel and no realcg/Clebsch-Gordan data in the compiled
sources: apart from this document, nothing under `psi4/src` mentions either. The other tree
hits are prose (`doc/sphinxman/source/sapt.rst`), a test that hard-codes the single
`-1/sqrt(3)` value inline rather than reading a table
(`test_recoupling_reproduces_the_isotropic_c6`), and the committed deck fixture. The
contract to implement is
`recouple`: `alpha_c(:,j1,j2,v,m) = sum_{t,u} cg(j1,j2)%p(t,u,v) * alpha_u(:, j1^2+t, j2^2+u, m)`
for `v` in `[|j1-j2|^2+1, (j1+j2+1)^2]`, with j1+j2>6 left at zero as above; `realcg_j1_j2`
records `l1 l2 L i1 i2 i3 i4` with 0-based indices, value `(i1/i2)sqrt(i3/i4)`, pure
imaginary when `i3<0`. `recoupling_tables` is the *second* stage (`alpha_c -> C_n`); this
first stage has no counterpart in the port. Second, no shipped entry point computes the
reference quantity: `anisotropic_dispersion.h` documents `isa_anisotropic_dispersion` as
"Undamped, nonretarded orientation-resolved scalar coefficients, NOT C_n(t,u,J)", a
different observable from the `.pot` table, so it cannot be gated against
`H2O_ref_wt4_L3_C12.pot` as written however accurate it is on its own terms. Until the
recoupling kernel and a `C_n(t,u,J)` driver land, the reproduction above is a property of
the audit script only and must not be written up as a claim about the shipped library.

### Supplied local isotropic dispersion boundary

`IsaIsotropicModel` now owns explicit local scalar responses in C++ with labeled
sites, origins, strictly increasing frequencies, explicit rank subsets 1–4 and
required provenance. Scalar inputs mean `trace(alpha_ll)/(2l+1)` for a local Racah
spherical block; the constructor does not localize or isotropize a distributed
response. Signed finite values are retained. Matrix getters return independent
snapshots.

`isa_isotropic_dispersion(model_a, model_b, cp_weights, max_order)` computes all
A/B site pairs through the selected even order 6–12. A/B frequency grids must match
exactly; weights are explicitly supplied with `1/(2*pi)` already included. Static
nodes require zero weight. The rank-pair factor is
`binomial(2*la+2*lb,2*la)` at `n=2*la+2*lb+2`, yielding factors 6; 15/15;
28/70/28; and 45/210/210/45 for C6/C8/C10/C12. Results label both site lists,
origins, frequencies/weights, provenance, units, included/missing rank pairs and
per-coefficient completeness. A within-model zero is distinct from missing terms.
This supports complete **local isotropic** C12 with rank4, not anisotropic C12.

Build/stage/byte comparison and the final expanded ISA/FDDS suite pass **639 tests
in 4.16 s**. New checks include independent analytic Lorentz integrals, explicit
rank factors, L1/L2/L3 coverage, scalar trace normalization, general 2x2 A/B pair
exchange, ownership, static exclusion and separate product/sum/coefficient overflow.
A composed synthetic one-site Q/response/scalar/C6 check passes an analytic C6;
it injects a unity partition and Lorentz coefficient operators and is not native
end-to-end parity. Independent advisory reviews found no must-fix defect in the
supplied-partition and isotropic boundaries. All four unchanged SAPT fixtures pass for this new extension, as do four GRAC
tests (50.31 s). Separate bounded evidence with source/core/output hashes is in
`tests/pytests/data_isapol/psi4_supplied_properties_evidence.json`.
See `SUPPLIED_PROPERTIES.md` for the expert contracts. Production Q/alpha,
localization, PFIT, anisotropic dispersion and the ordinary driver remain open.

## 8. Existing compatibility infrastructure

Keep the dedicated ISA grid: Psi4 Lebedev accessor, radial map
`r_i=alpha*(i/(n_r-i))^2`, weights `2*n_r*alpha^3*i^5/(n_r-i)^7`, i=1..n_r-1,
unclamped Becke size adjustment, no orientation rotation/pruning/cutoff. Module grid
defaults 80/590 differ from the method preset 100/434. Distinguish requested and
actual angular order. Shell ordering can differ; compare with a documented permutation.

CamCASP AtomProp literals are float32 widened to double and use historical
angstrom/bohr constant 0.529177249. Keep this confined to compatibility tables rather
than changing Psi4 physical constants globally. Lattice radii use a different,
double-precision MODULE radii table. Unsupported zero Slater radii must fail clearly.
Maclaren RNG and acceptance order are fixture-tested; changing rejected-draw consumption
changes the whole cloud. Keep deterministic sampling separate from fitting policy.

Existing no-FMA settings support selected strict grid fixtures; they are not evidence
that every later kernel should sacrifice performance for bit parity. Measure kernels
and tolerances independently. Do not weaken existing fixtures to hide regressions.

### CamCASP licensing: MIT, confirmed, transcoding permitted

**CamCASP is MIT-licensed** (`MIT License, Copyright (c) 2019 Anthony Stone`),
confirmed by the user. MIT is compatible with Psi4's LGPL-3.0, so CamCASP Fortran at
`~/gits/CamCASP` (v6.0 patchlevel 051, 225 files, ~195k lines in `src/`) **may be read
and transcoded into `libisapol`**. Do not re-derive this question: earlier increments
wrongly treated CamCASP as unlicensed because the `LICENSE` file was **deleted from the
tree** on 2019-10-30 in commit `b40ae4f` ("Recompiled; redundant files removed") while
`INSTALL.md:21-23` still refers to it. Recover the text with
`git -C ~/gits/CamCASP show b40ae4f^:LICENSE`.

Transcoding obligations, which are not optional:
- Retain the MIT copyright notice and permission text in every derived file, and add a
  CamCASP entry to Psi4's licensing documentation. MIT requires the notice in "all
  copies or substantial portions".
- Preserve attribution to Misquitta and Stone in `libisapol` headers and SPEC prose.
- Prefer citing the CamCASP source file and line for each ported kernel, so later
  agents can re-derive the provenance.

Three prohibitions are unchanged and are **not** relaxed by the above:
- **ORIENT remains forbidden.** `~/gits/camcasp_psi4/orient/` is GPLv3; see the
  localization section. That restriction is specific to ORIENT and must not be
  generalized to CamCASP.
- Do not transliterate/link the separately licensed **GDMA grid** into Psi4.
- Do not **ship** extracted CamCASP source or binaries. Local source-extraction
  oracles are development tools; ship only our harness and permitted numerical
  fixtures. This constrains distribution, not developer reading.

The rule that production code and pytest never *invoke* CamCASP, ORIENT, PFIT,
CASIMIR or `.camcasp-reference/` at runtime is a **dependency** constraint on the
shipped product. It has been misread as a ban on reading CamCASP source during
development. It is not.

## 9. Acceptance gates and build discipline

Run in `p4_ci`, build with `bash build.sh`, then explicitly select the staged Psi4
for pytest. The current build script is not fail-fast and its final cleanup can
return failure independently of compilation; inspect the actual build result and
verify the imported extension path. Never infer successful compilation solely from
its final status. Keep existing uncommitted work intact.

For this worktree, avoid source-package shadowing and an empty trailing PYTHONPATH entry:

```bash
bash build.sh
export PYTHONPATH="$PWD/build_camcasp_psi4_joint/stage/lib"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -P -m pytest \
  tests/pytests/test_isapol.py tests/pytests/test_isapol_fit.py -q
```

`python -c`/`python -m` from the repository root can otherwise import the source
`psi4/` without its compiled extension. `--psiapi` appends the existing PYTHONPATH;
when empty, that trailing entry also exposes the source directory.

Dependency-ordered gates:

1. Existing constants/RNG/cloud/frequency/grid/coefficient-table tests.
2. Frozen ISA-A update: metric modifications, W-Eps RHS, damping, Positive-W,
   s/non-s selection, denominator cutoff, solve and overlap-change diagnostic.
   Compare same water samples with an executable-source oracle; include independent
   analytic tests and malformed/singular inputs. Explicitly label sampled-density
   tests that are not Drho-C reference runs.
3. Basis/normalization and fixed Drho-C density export, actual water intermediate
   metric/RHS/D/w checkpoints, including activated states. This is not achieved by
   gate 2 alone.
4. Full ISA-A controller/tails and final water shapes, separate population definitions,
   multipoles, convergence/nonconvergence, restart, rotation and grid refinement.
5. Native DF density and response Hessians/transition coefficients; compare fixed-input
   and independently generated-wavefunction runs separately.
6. Partitioned Q and distributed alpha at each frequency; sum rules and translations.
7. Full-rank localization and frames, then PFIT model/target/objective and held-out
   point-response residuals. The localization leg is gated against the **recorded
   format-B `.pol`** (`~1e-07` provenance floor), not against ORIENT: ORIENT source is
   GPLv3 and must not be read or transliterated, so this leg must be implemented without
   it. PFIT itself is MIT CamCASP and may be transcoded directly.
8. Dispersion: independent isotropic identities, anisotropic coverage/invariants,
   historical water C6–C10 regressions and corrected rank-4 C12 - plus the hermetic
   reference gate now measured in section 7, i.e. reproduce every printed coefficient of
   `H2O_ref_wt4_L3_C12.pot` from `H2O_ref_wt4_L3_casimir.data` to `<= 1e-06` relative with
   an exact match on the printed row set and on placeholder placement. Do not claim
   CamCASP parity for the J=9,10 coefficients the reference never wrote.
9. Public Python and oeprop adapters, water end-to-end ISA-A properties and docs,
   gated against the upstream in-tree **psi4** reference of section 2.3
   (`H2O_ref_wt3_L2_Cn.pot`). Upstream's own criterion is byte equality for a fixed SCF
   code. State the gate on the isotropic and identifiable coefficients with an absolute
   floor on the small anisotropic ones, record the GRAC shift (`0.064900` Eh) and its
   provenance in the result, and never compare against the nwchem or dalton reference.

Fast numerical tests live in `tests/pytests/`; committed fixtures in `data_isapol/`.
Source-dependent generation is opt-in development tooling, not a runtime dependency.
Mark genuinely expensive molecular tests appropriately; do not mark absent stages
passing via unconditional skips. No full parity percentage or root-cause attribution
until controlled stage substitutions and precision-aware error reports support it.

### Current measured checkpoint

The frozen fitting stage and its Python bindings are implemented in `isa_fit.{h,cc}`
and `export_isapol.cc`. In `p4_ci`, `bash build.sh` compiled/installed the reviewed
code; the installed extension was byte-compared with the build output. The script
still returns 1 for missing optional `stubgen` and its unmatched cleanup glob;
these are not compiler failures and the script itself was not changed.

The explicit staged-package command above passed **126 tests** (existing
infrastructure plus frozen-fit tests). Python compilation and `git diff --check`
also passed. On this environment, all nine source-extracted water update cases
had zero observed absolute difference in metric, RHS, fitted coefficients and
population. Maximum normalized linear residual was **1.3612e-17**. This is an
observation for matched samples and this LAPACK/compiler environment, not a promise
of bitwise agreement on other platforms. The tests retain precision-aware tolerances.
Four further source-extracted synthetic cases cover signed samples, cutoff/damping,
ridge selection and nonzero capped contributions.

**Still unvalidated/unimplemented:** native AtomAux construction and native Drho-C
fitting; ISA fixed-point/activation/tails; partitioned multipoles;
FDDS, localization, PFIT and tensor-to-dispersion stages; full water ISA-family
properties and high-level driver/oeprop integration. Supplied-metric symmetry and
LU residuals are checked, but this frozen API does not establish that a supplied
metric came from a valid basis or prove its positive definiteness. The next gate
is an actual production-water fixed-density checkpoint, not more table parity.

Opt-in production capture/replay tooling now lives in
`tests/pytests/data_isapol/oracle/{capture_isa_checkpoint,prepare_isa_water_run,replay_isa_checkpoint}.py`.
See `oracle/PRODUCTION_CHECKPOINT.md` for the isolated-source workflow and root
`plan.md` for validation results and handoff state. The strict stream reader's
synthetic tests establish format/replay plumbing only.

Two actual **adapted-archive Drho-C/LU oxygen checkpoints** have now passed C++
replay: 68,310 points and 109 atomic functions each. At call 1, metric/RHS/D errors
were zero observed, population error 8.4377e-14 and normalized solve residual
5.3526e-17. At call 157 (iteration 53), active W-Eps=0.17 and ridge=0.001,
maximum absolute errors were 0 (metric), 1.7764e-15 (RHS), 2.8910e-12 (D), and
4.0856e-14 (population); normalized residual was 2.2062e-17. The activated D
error scaled by max(1,max(abs(reference D))) was 4.20393e-13. These are measured
same-production-sample results, not native basis/DF construction parity.

The reference converged in 53 iterations. Seven final serialized shape/tail files
were byte-identical across first-update-traced, activated-traced and untraced
reference runs. Final tooling/infrastructure tests: **149 passed**; Python
compilation and diff checks passed. No Psi4 C++ implementation changed in this
increment. Bounded evidence and provenance hashes are recorded in
`tests/pytests/data_isapol/camcasp_isa_production_evidence.json`; large sampled
streams remain local development artifacts, not portable regression fixtures.

The adaptation retains archived orbitals/bases, not the modern preset. It omits
unsupported newer `SET Num-Int-Pars` controls and uses inspected source defaults;
angular request 200 resolves to 230 actual points. It also omits unsupported
explicit `FIX = ON` syntax, since this source defaults tails ON. These differences
are recorded in the run manifest. Reference convergence does not mean a Psi4 ISA
controller has been implemented.

### Descriptor export and independent reconstruction checkpoint

The v2 capture now exports actual molecular Drho-C coefficients, complete runtime
molecular/atomic/shape basis descriptors, density neighbour storage, old shape,
explicit shape-to-AtomAux s-map and raw new shape **before DIIS/mixing**. The reader
still accepts v1. Allocated neighbour/map arrays have zero-padded storage; active
sizes must not be inferred from allocation lengths.

`oracle/reconstruct_isa_basis.py` independently derives real regular harmonics from
Legendre polynomials and checks contracted radial evaluation, GAMINT Cartesian
factors, DALTON spherical order, analytic co-centred overlap, shell normalization
and raw shape projection. Runtime contraction coefficients are already normalized
and must **not** be normalized again. This is a development oracle, not a native
Psi4 basis/DF provider; supported ranks are S–G.

Three actual v2 checkpoints passed full C++ frozen replay and selected-point basis
reconstruction: first oxygen, activated oxygen and activated H1. Each full stream
has 68,310 points; atomic dimensions are 109/109/49, and the common molecular
Drho-C basis has 246 functions. On **97 deterministic points per checkpoint**,
maximum absolute reconstruction errors were 7.10543e-15 (atomic samples),
1.13687e-13 (density), 5.32907e-15 (complete atomic overlap) and 4.44089e-16
(shell normalization). Raw s-projection and shape-basis mapping had zero observed
error. Active exponential-tail samples are explicitly **not** reconstructed from
bare Gaussian descriptors. New H1 full fitting replay had maximum D error
2.21993e-14 and normalized residual 7.98003e-17. Seven final shape/tail artifacts
remain byte-identical across v2 traced runs and the earlier untraced reference.
Final four-file regression suite: **185 tests passed**, including three portable
production descriptor cases; Python compilation and diff checks passed.

Bounded portable `camcasp_isa_basis_{first,activated,hydrogen}.json` fixtures and
`camcasp_isa_basis_evidence.json` are in `tests/pytests/data_isapol/`. They validate
selected descriptor samples and complete atomic metrics, **not** full frozen-fit
replay without the local streams. See `oracle/BASIS_RECONSTRUCTION.md` and root
`plan.md` for tests, exact provenance, limitations and the next implementation
milestone. Gate 3's exported-representation boundary is now numerically supported
for this adapted protocol; native C++ basis/density generation, intermediate
activation/phase checkpoints and modern-preset equivalence remain separate gates.

The subsequent typed C++ exported-input sampling provider in
`explicit_basis.{h,cc}` now passes the same three portable descriptor cases
(97 points each). Maximum absolute errors are **7.10543e-15** for atomic samples
and **5.68434e-14** for fixed density; maximum scaled density error is
**2.44725e-16**. Complete molecular sample columns had zero observed difference
from the independent Python polynomial oracle. Contracted S–G synthetic tests
cover both representations; production fixtures remain the adapted-archive protocol.
The staged `p4_ci` build compiled/installed successfully and the four-file suite
passed **209 tests in 1.93 s**. Python compilation and diff checks passed.
A unity-build helper overload collision found by the initial regression run was
corrected without changing inherited fitter tests or behavior. See `plan.md` and
`.pi/audit/explicit-basis-{retests.log,errors.json}` for exact evidence.
This validates exported-input sampling only; complete atomic metrics still use
supplied inputs/the independent audit, not a new C++ overlap builder. Native basis
recipes, Drho-C fitting, tails/controller and end-to-end parity remain unimplemented.

The next increment adds C++ co-centred analytic overlap and validated raw shape
projection. All three **complete** portable atomic metrics (109/109/49 functions)
pass: maximum absolute error **1.77636e-15**, maximum globally scaled error
**6.66134e-16**. Raw shape projection and metric symmetry differences are zero
observed. Maximum difference versus the independent Python polynomial metric is
**3.55271e-15**; unweighted diagonal normalization error is at most **6.66134e-16**.
The staged `p4_ci` build compiled/installed; **243 tests passed in 2.14 s**, including
34 new production/algebraic/validation cases. Python compilation and diff checks
passed. Exact commands and limitations are in `plan.md`; measured evidence with
fixture/source hashes is `.pi/audit/atomic-overlap-errors.json`. This advances the
exported-input metric/projection boundary only: provider-to-fit assembly, full-grid
reconstruction, native basis/DF generation, active tails/controller and end-to-end
parity remain separate gates.

Typed provider-to-fit assembly now passes full exported-input reconstruction and
frozen replay for first O, activated O and activated H1, **68,310 points each**.
Maximum errors over all three are **7.10543e-15** (basis), **5.68434e-14** (density),
**1.77636e-15** (weighted overlap), **3.55271e-15** (RHS), **6.42908e-12**
(fitted coefficients/raw shape), and **8.34888e-14** (population). The largest
coefficient error scaled by max(1,max(abs(reference D))) is **9.34873e-13**;
normalized solve residuals are at most **3.99002e-17**. Derived primitive metadata
and raw squared distances have zero observed difference. The original scaled
replay tolerance 1e-9 was unchanged. Final staged suite: **261 passed in 2.13 s**;
Python compilation and diff checks passed. Bounded evidence and hashes are in
`tests/pytests/data_isapol/camcasp_isa_provider_evidence.json`.

Unlike the earlier 97-point checks, this covers all recorded basis/density samples
and the reconstructed metric through the solved frozen update. It still injects
reference quadrature, old coefficients, density neighbours and screened/tail-processed
shape samples. It does not implement native Drho-C/basis generation, tail replacement,
synchronous sweeps, activation/mixing, a fixed point, or end-to-end properties. The
future CamCASP Libint2 branch will require its own provenance and controlled comparison;
these legacy reference artifacts remain unchanged.
