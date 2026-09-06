# libisapol: ISA-family properties and staged CamCASP parity

## 1. Objectives and current status

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

### 3.2 User surface

The ordinary user workflow is **wavefunction-first and property-request-driven**:
run `energy(level_of_theory, return_wfn=True)`, then pass that wavefunction to an
`oeprop`-style property request, as users already do for MBIS. ISA is not an energy
method and must not be registered as `energy('isa')` or `energy('isapol')`. Examples
should start with PBE0/aug-cc-pVDZ; that usability example is not the separately
pinned CamCASP parity protocol. No hidden SCF is performed by the property call.

Keep the normal input short: the user supplies a converged wavefunction, selects
partition/localization policies through Psi4 options, and requests properties.
Use the method-neutral option name `PARTITION_SCHEME` (planned, not yet registered),
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
by this text. `PARTITION_SCHEME` is the agreed design name, not an available option. Existing `oeprop` returns None, so any structured-result access or
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
require a separate mapping audit. Initial extraction tests/regressions are pending.

This is supplied-intermediate reuse, NOT native monomer construction or response
parity. C++ still takes A/B caches, and its hybrid QR assumes nov>=naux. A genuinely
single-system construction facade remains to be factored without duplicating
numerics or adding a dummy partner. The current
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

## 6. Localization and point-response refinement

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

Important upstream issue: `casimir.f90:501–514` skips recoupling rank pairs with
j1+j2>6, although c12code requires (4,4),(4,3),(3,4). Corresponding storage is not
explicitly initialized. Do not reproduce undefined values or treat this executable
as an unqualified complete-C12 oracle. Use independently derived isotropic rank-4
identities and a documented corrected reference for that coverage.

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

CamCASP author permission is recorded in the prior draft but the authorization text
is not in this repository. Preserve attribution; confirm redistribution terms before
shipping source-derived material. Do not transliterate/link the separately licensed
GDMA grid into Psi4. Local source-extraction oracles are development tools; ship only
our harness and permitted numerical fixtures, not extracted CamCASP source/binaries.

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
7. Full-rank localization and frames against ORIENT, then PFIT model/target/objective
   and held-out point-response residuals.
8. Dispersion from supplied models: independent isotropic identities, anisotropic
   coverage/invariants, historical water C6–C10 regressions and corrected rank-4 C12.
9. Public Python and oeprop adapters, water end-to-end ISA-A properties and docs.

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
