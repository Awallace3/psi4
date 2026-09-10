# libisapol — current scientific and API specification

This is the authoritative compact specification at accepted code checkpoint
`1a097ec9f6a033428053354b294c5524e98b6137`.
Execution state, exact build/test commands and next-agent steps live in
[plan.md](../../../../plan.md). Stage-specific details are linked below.
The pre-compaction SPEC is preserved at `git show 1a097ec9f6:psi4/src/psi4/libisapol/SPEC.md`.
Its historical status paragraphs are superseded here; its detailed source-audit
chronology remains available. No numerical code changed during compaction.

## 1. Accepted capability is not full protocol parity

- Fresh restricted C1 PBE0/cc-pVDZ H/O demonstration: native Drho-C/ordinary ISA-A,
  direct-OV ALDA, strict LW, atomic polarizabilities and nine ordered Cn pairs.
- Explicit fixed-GRAC SCF-input admission; default NONE remains unchanged.
- Owned supplied-input response/localization/PFIT APIs, isotropic and
  orientation-resolved dispersion, and independently callable recoupled
  `Cn(t,u,J)`. These are distinct representations and acceptance boundaries.
- Provenance-qualified expected reference basis manifest and early response
  dimensional preflight; **not** a registered matched property/ISA recipe.
- Latest integrated suite: **1,895 passes / 36.79 s**. Latest default water:
  **29.6536 s / 594,120 KiB**, all saved outputs bitwise baseline-identical.
  Fixed-GRAC public strict-LW endpoint also passes. Exact evidence is in plan.

Do not equate analytic invariants, parser consistency, same-input kernel replay,
independently generated stage parity and full wavefunction-to-property parity.
No percentage-complete or root-cause claim without scoped evidence. Preserve raw
errors, signs, exclusions, rank coverage, conditioning and reference-print limits.

### Reference tracks must remain separate

1. **Default demonstration:** generated H/O JKFIT/even-tempered recipe, cc-pVDZ,
   ordinary ISA-A, direct-OV response, no PFIT. Not a modern CamCASP preset.
2. **Historical gate 9:** the Psi4 `H2O_props` CLT and traced `777f904` generator
   select **non-ISA constrained NN → distributed response → LW → PFIT**.
   NN means MO-pair space, not fitting norm. Expected spherical aug-cc-pVTZ MAIN92;
   ordinary Cartesian RI AUX246; same-AUX AtomAux fallback; no ISA shapes/controller.
   Historical actual SCF export, grid/propagator and PFIT run artifacts are missing.
   Native MAIN is span-equivalent, not literally contraction-identical.
3. **Modern ISA-A / ISA-A+DF templates:** distinct explicit protocols, not implied
   by the gate-9 CLT, basis alias or final potential. A+DF is not ordinary A with
   a different initialization. Do not silently switch the chosen target.
4. **Recoupled archive oracle:** `H2O-isagrid`, not historical `H2O/orient_local`.
   O–O C6 anchors differ (26.48177 versus 17.25559); do not mix their fixtures.

[NATIVE_REFERENCE_BASIS.md](NATIVE_REFERENCE_BASIS.md) owns exact expected data,
source hashes, literal/span distinctions, licensing and missing-artifact lists.
`historical_scf_export_verified` remains false; use actual adapted orbitals.

## 2. Public lifecycle, provenance and fixed-GRAC admission

`psi4.oeprop(wfn, 'ATOMIC_PARTITION', 'ATOMIC_POLARIZABILITIES',
'ATOMIC_DISPERSION')` returns **None**. Retrieve the owned structured result via
`psi4.atomic_property_result(wfn)`. Native request entry invalidates the previous
attachment, including rejected/mixed requests; unknown `ATOMIC_*` requests fail.
Ordinary properties retain their existing dispatch. Do not expose ISA as an SCF
energy method or add automatic PFIT/dispersion/cloud generation to static requests.

C++ owns numerics; Python orchestrates. Constructors/getters have explicit snapshot
ownership; some returned records are mutable copies, not live internal state.
Results identify units, frequency axes, site/component order, ranks, origins/frames,
representations, provenance, convergence and diagnostics. Reuse requires matching
actual wavefunction state and effective policies. Reject stale state, nonfinite or
malformed inputs, unsupported combinations and missing declarations before work.
Density partitioning (`PARTITION_SCHEME`) and tensor localization are separate;
charges alone do not define a continuous partition. Preserve expert stage injection.

No hidden SCF, reference electronic-data input, post-SCF energy/orbital shifting,
charge rescaling, silent symmetrization, strategy fallback or inferred convergence.
The public native boundary requires actual SCF stopping evidence and unchanged state.

Fixed-GRAC opt-in requires `ATOMIC_SCF_ASYMPTOTIC_CORRECTION=FIXED_GRAC` plus a
positive finite **exactly matching** `ATOMIC_SCF_EXPECTED_GRAC_SHIFT`. Validate
underlying canonical full-LibXC PBE0 and actual unpolarized LB×0.75/VWN×1,
alpha=0.5, beta=40, no component tweaks. Never infer actual state from ambient
options. SCF seals/context snapshots include GRAC attachments and underlying LibXC
cutoffs; canonical functional identity remains distinct from numerical cutoffs.
Owned correction provenance enters context reuse. NONE rejects undeclared GRAC.
This is **ALDA using fixed-GRAC SCF orbitals**, not a GRAC response derivative.

Details: [NATIVE_OEPROP.md](NATIVE_OEPROP.md),
[NATIVE_FIXED_GRAC.md](NATIVE_FIXED_GRAC.md),
[SUPPLIED_PROPERTIES.md](SUPPLIED_PROPERTIES.md).

## 3. Explicit bases, grids and bounded execution

- Explicit basis coefficients are **effective**; copy them without renormalizing.
  Literal reference coefficients must first undergo their declared primitive and
  contraction normalization. Preserve shell order, including diffuse placement.
- Current basis evaluator supports S–G, Cartesian GAMINT and spherical DALTON.
  Basis p order is x,y,z; observable Racah multipoles use a different ordering.
  Actual MAIN adaptation verifies collocation/overlap/orbitals; do not presume
  Psi4's component order or substitute reference contractions for actual orbitals.
- Current molecular-AUX Coulomb/three-centre path is **Cartesian-only**; Orbital
  role is spherical. Sampling support alone does not establish spherical AUX
  Coulomb support. Any future transform must cover charges, metric and 3-centre rows.
- Frozen ISA/native SiteRecipe uses primitive atomic functions, co-centred s shapes
  and exact explicit shape maps. Distinct AUX/AtomAux is required on the current
  native ordinary-A path. Identical-basis initialization would require separately
  sourced unconstrained-density slices, not constrained Drho-C substitution.
- Neighbours are unique zero-based/unpadded indices. Empty screens everything.
  Shape-neighbour and molecular-AUX-centre indices are different spaces.
- ISA radial map: `r_i=alpha*(i/(n_r-i))^2`, radial weights
  `2*n_r*alpha^3*i^5/(n_r-i)^7`, i=1..n_r−1; supported Lebedev resolution,
  unclamped Becke size adjustment, no implicit rotation/pruning/cutoff.
  Record requested **and actual** grid orders. Compatibility AtomProp float32
  literals and legacy bohr conversion stay local; do not change Psi4 constants.
  Preserve fixture-tested RNG/rejected-draw consumption and lattice acceptance.

Preflight uses actual nbf/nmo/nocc and supplied grid rows, not basis aliases or
pruned SCF-grid estimates. It mirrors existing dimensional guards; C++ shell,
state, workspace and grid checks remain authoritative. Defaults: nbf≤256,
nov≤512, `nov*nbf^4≤64e9`, `grid_rows*nov^2≤2e9`; keep explicit existing
configuration semantics and rejection order. Passing preflight is not convergence,
complete memory admission or scientific integration accuracy.

At nbf=nmo=92, nocc=5 and the current public 3-atom IsaGrid(99,590): nov435,
rows=173460, ALDA work=32822968500, so it **fails**. The 10569-row resource ceiling
is not permission to reduce the grid or a matched-reference integration policy.
Public preflight precedes partition/provider work, not all allocations/validation.

Parallelize independent outputs without changing primitive/point sum order; select
lowest-index worker failures deterministically. Preparation bounds numerical payload,
not RSS. Release immutable preparation before final Q/response via independent owned
snapshots. Stream AUX-Q while preserving point-order sums; direct-MO GEMM is distinct.
MKL isolation for tiny AO conversion is **thread-local**, restores on nesting/errors,
and has no process-global/non-MKL fallback. Measure runtime/RSS and equivalence;
do not infer speedups. Full non-OpenMP core validation remains unperformed.

## 4. ISA-A equations and ordering invariants

### Fixed molecular density

Let J be the AUX Coulomb metric, q integrated AUX functions, B AUX–MAIN–MAIN
three-centre integrals and C occupied spatial orbitals of occupation two:

```text
(J + lambda*q*q^T) d = 2*sum_i(C_i^T B C_i) + lambda*(2*nocc)*q
```

Drho-C lambda=1000 is a **finite charge penalty**, not exact conservation. Do not
rescale d, symmetrize J or silently regularize. Drho-C, Doo-C (pair-fit then trace)
and sampled AO density are different products. Cartesian q includes all even-power
components. Report coefficient, represented-density, charge and backward residual
errors separately; a small backward residual does not imply forward accuracy.

### Frozen atomic fit

With old shapes w0, atomic functions chi and cutoff c (default 1e−36):

```text
f_a = rho*w0_a/sum_b(w0_b) when abs(sum_w0)>c; otherwise 0
E_a = exp(min(w_eps*|r-R_a|^2,230)) when w_eps>0; otherwise 1
s RHS:     integral chi_k*(f_a + eta*w0_a)*E_a
non-s RHS: integral chi_k*f_a*[E_a unless s-block-only]
```

The supplied metric is already W-Eps weighted; its analytic weighting has **no
exponent cap** and requires integrable primitive sums. Multiply s/s entries by
(1+eta). Add Positive-W ridge only on eligible diffuse-s diagonals (exponent≤max_alpha),
and under auto only if preceding D_k<0. LU solve, retain diagnostics, project via
explicit s-map. Positive-W is a soft ridge, not positivity-constrained optimization.
Do not clip signed density/active-tailed samples or normalize fitted coefficients.

### Synchronous controller and tails

All atoms, including converged ones, fit from the **old** shapes. Measure delta and
raw shape charge before mixing. Mix only unconverged shapes when iteration>skip;
never mix full fitted D. Compute next controls and fit tails from old Gaussian shapes,
then commit. Saved charges retain pre-mixing bookkeeping. W convergence is:

```text
delta = abs(1 - abs(w_new^T S w_old)
                  / sqrt((w_new^T S w_new)*(w_old^T S w_old)))
```

S is **unweighted** s overlap. Pure amplitude changes can be invisible; report
populations separately. EPS-Q is not an added W stopping condition. Positive activation
thresholds start inactive; zero starts active; threshold activation is non-latched.
Strict iteration>tail_limit activates next-sweep replacement. Convergence does not
force another sweep under newly activated controls. Ordinary A does not imply a
postconvergence self-consistent tail loop, A+DF, DIIS, Q/RHO stopping or symmetry.
Restart requires identical bases/density/grids/options, not merely matching shapes.

Func-1/Fit-3: `A*exp(-b*r)`, `b=-w0'(r1)/w0(r1)`, strict **1<b<4** or previous
valid-b fallback; `A=Q_w0/Q_exp`, `IP=b^2/8`. It conserves exterior charge, not
continuity. Preserve centred finite difference **1e−8**, translated Cartesian-z
samples and primitive shell sum **before** expansion-coefficient multiplication.
Do not silently replace this cancellation-sensitive path with an analytic derivative.
Active defined tails replace only r>cutoff and retain signed Gaussian interior;
no-tail sampling clamps max(w,0). Signed valid A is allowed. Invalid/underflowed fits
return deterministic undefined results, not legacy uninitialized/saved-local behavior.
Printed five-decimal tail parameters cannot certify a 1e−8 oracle. Analytic shape,
stockholder and rescaled ISA populations are different quantities.

Detailed legacy source anchors/evidence: pre-compaction SPEC §4; executable contracts
also live in `isa_fit.h`, `isa_sweep.h`, `isa_controller.h`, `isa_shape.h` and tests.

## 5. Response and multipole representation boundary

For occupied a,b and virtual r,s, preserve the reference Hessian convention:

```text
H1 = Delta + 4(ar|bs) - CxKernel*[(ab|rs) + (as|br)] + K_local
H2 = Delta            - CxKernel*[(ab|rs) - (as|br)]
(H2*H1 + xi^2*I) X = -4*H2*D
C_DF = D^T*X
```

**H1 exchange uses the sum.** D is nov×naux with occupied-fast OV ordering; no
invented 2/sqrt(2) pair multiplicity. NN off-diagonal fit RHS has no spin multiplier;
only diagonal pairs get lambda*q. Internal hybrid reference convention uses
`K_local=4*(1-CxKernel)*K_ALDA`; record actual kernel selector/scales separately
from functional identity, GRAC and density source. “NEW-PROP” configuration alone
is not proof that a particular executable response branch ran.

Archived tracks may use lambda1 response/kernel D and lambda0 interaction/kernel-
density fits. Preserve actual incoming operands and complete producer epochs;
do not reconstruct historical input as A*D or read unused padded allocation columns.
Matched equation replay does not certify native kernel/integral/DF construction.

For molecular-AUX function chi_k and already screened/tail-processed ratios:

```text
Q_(a,t),k = integral R_t(r-R_a) * w_a/sum_b(w_b) * chi_k(r) dr
alpha = -Q*C_DF*Q^T
```

Q uses **molecular AUX**, not AtomAux. Observable real Racah order is
`00,10,11c,11s,20,...`; preserve site/component axes, global origins, bohr/atomic
units and supplied frequency order. Retain signed weights/ratios, raw reciprocity
defects and denominator exclusions (abs(sum_w)≤cutoff). No extra metric/sign/spin
transformation or symmetrization in this supplied-coefficient contraction.
Molecular recovery needs translations, not just sums of dipole blocks.

Direct-OV identity coordinates/moments are not finite-penalty fitted AUX. SAPT
Coulomb-auxiliary FDDS is not automatically C_DF: ordinary unregularized mapping
uses `J^-1 U J^-1`; constrained cases need explicit mapping validation. Reuse the
shared FDDS solver and preserve its legacy SAPT behavior. Full-OV effective baseline
is not ordinary uncoupled response; singular baselines fail without fallback.
Existing coupling pseudoinverse rcond=1e−13 remains; exact equivalence requires
nonsingular, untruncated coupling. Kernel/source provenance is not inferred by name.

Details: [SUPPLIED_PROPERTIES.md](SUPPLIED_PROPERTIES.md), native contracts,
`native_response.h`, `partitioned_response.h`, `ov_fit.h`, shared `sapt/fdds_response.py`.

## 6. Localization, recorded-input exception and PFIT

Translations: `R(x+d)=T(d)R(x)`; source→target d=Ra−Rb. Rotations:
`R(F*x)=D(F)R(x)`, finite proper local-to-global F. They do not localize/truncate.

`isa_irregular_solid_harmonics(rank, d)` gives `r^(-k-1) C_kq` for every k≤rank in
the same Racah order `00,10,11c,11s,...`; rank0 is `1/r`, **not** 1, and d=0 is a
rejected singularity, not zero. `isa_t_functions(rank, point, site, F, damping)` is
one pfit T row: the displacement point−site rotated into the site's LOCAL axes by
F (columns are the local axes, same contract as `isa_multipole_rotation`), then the
irregular harmonics, then optional Tang–Toennies `1-e^(-br)*sum_{n≤k+1} br^n/n!`
applied per whole rank block. Equivalently `t = D(F)^T R_irr(point−site)` (verified
to 1.7e−18). The reference protocol leaves CamCASP's `Damping` at zero, so the damped
branch carries algebra from source but **no reference artifact**. Bitwise agreement
with CamCASP's own compiled `solidh` holds for 22751 components over 2045 points and
for the reference water axes' T rows; it is a claim about non-contracting arithmetic
only (§3.5.5) — Fortran built `-O2 -march=native` moves by ≤1.3e−13 relative, and for
general (non signed-permutation) frames gfortran's `matmul` reduction order costs
≤1.53e−12 relative on the frame contraction, not in the recursion.

LW uses explicit graphs, negative graph Laplacian and componentwise pseudoinverse;
rank0–3 workspace and rank1–3 local output, eigen cutoff1e−4, transfer omission1e−7.
Keep Moore–Penrose and postcondition diagnostics. Equal elements do not establish
site equivalence without compatible types/frames. `refined_pairs` is LW workspace,
**not PFIT-refined output**. Do not resurrect the rejected LS attribution/draft.

Production algorithm postconditions remain **1e−6**, including off-site,
reciprocity, molecular sum and charge transport. Recorded-input leg A matches all
11 indices / 7425 local entries at **atol1e−11, rtol0**, but the supplied input has
its own sum-rule defect (~7.011e−4) and four frequency-header conflicts. The explicit
input-defect policy reports that defect separately; `reported_input_sum_rule` is
not passing the original production postcondition. Historical static-only 1e−3
exception is identity-pinned, not a dynamic/default waiver. Native generated water
passes strict checks without that exception. Format-B native comparisons retain
their ~1e−7 print floor. See `test_isapol_lw_leg_a.py` and LW result policy fields.

PFIT targets must declare actual point-charge response `-d(phi_induced)/dq`, charge,
units, and native-direct versus fitted-propagator origin. Never silently substitute
a truncated multipole reconstruction, bare electrostatics or energy factor1/2.
Use fields/model ranks/frames, equivalences, fixed flags and anchors from the same
localized response. Explicit model input and automatic selection are different APIs.

Objective pairs are **within each physical batch, j≤i including diagonals**, with
no extra off-diagonal multiplicity. Eliminate fixed parameters consistently; add
P and P*anchor. DSYSV and streaming QR must represent the same objective. Preserve
linear-constraint policy: `s*(t^T p-a)^2` gives `s*t*t^T` and `s*a*t`; any source
nonzero-LC inconsistency requires a named tested policy. Weight4 uses coefficient/
(1+xi²) for rank≤1 pairs, zero otherwise. Track conditioning/identifiable observables,
not unstable high-rank parameters alone. Targets/fields dominate work too; batching
must not change the pair set. Native targets come only from the bounded direct-OV
prerequisite below; no other solver API generates them.

### Native direct-OV point-charge target prerequisite

`IsaPointChargeOperators` (`point_response.h`) builds eagerly and immutably the
restricted C1 point-charge OV coupling

```text
W(t,p) = + integral phi_i(r) phi_a(r) / |r - R_p| dr,   t = a*nocc + i
```

with the **positive** Coulomb kernel, i.e. exactly minus the charge-inclusive
electron ESP operator oeprop obtains from `ElectrostaticInt::compute(result, C)`.
Both conventions are admissible for a response leg because W enters the
contraction twice; mixing them within one leg is a sign error, so the single
convention is fixed and published in `convention()`. The class performs no
response solve, frequency, fitted auxiliary metric, charge/multipole model,
energy 1/2 or bare electrostatics, and holds no nuclear term. Geometry
diagnostics (nearest nucleus, closest source pair, largest element, planned
bytes) are recorded and never used to screen, repair or condition; exactly
coincident source rows are rejected as duplicate **input**, not as a
conditioning heuristic. Admission, shell/Libint validation and resource
accounting follow `native_response.cc` in this module — restricted closed-shell
C1, nov≤512, nbf≤256, max_am≤4, max_nprimitive≤64, npoint≤512, explicit byte
envelope — and the operator itself is new native work with no CamCASP or ORIENT
source consulted for it.

`isapol_native_point_response.native_point_charge_response` reuses one existing
native response's owned H1/H2 in a **second** shared full-OV solver whose
transition legs are W, leaving the caller's response unmutated, and returns

```text
v(i*xi)_pq = -(W^T C(i*xi) W)_pq
```

in atomic units Eh/e², equal to `-d(phi_induced at R_p)/d(q at R_q)`.
Frequencies are imaginary-axis magnitudes in hartree, xi≥0, xi=0 static, no
quadrature weight applied. The npoint right-hand sides avoid only the nov×nov
right-hand-side and solution blocks: the dominant cost is unchanged — one
O(nov³) factorization of the same nov×nov operator plus the shared solver's own
O(nov³) H2·H1 product, recomputed rather than reused, and O(nov²) dense
workspace — and the saving vanishes at npoint = nov. What holds
unconditionally is that this does **not** relax, bypass or re-tune the response
work guard the supplied response already passed; the bound on the numpy side is
the nov cap, and `max_bytes` covers the C++ dense envelope only. Context reuse
is exact-equality only on occupation, dimension, orbitals and orbital energies —
no tolerance can make a re-converged wavefunction the same physical state.
`correction_provenance`, `caller_converged` and `convergence_evidence` are
forwarded from the supplied response and hashed into the context digest; they
are **not** re-verified here, and hashing a declaration is not validating it.
The declared-`ov_order` equality check is hygiene between fixed literals — the
`t = a*nocc+i` packing is verified only by the independent MO transform in the
tests.

Packed targets take the **computed** lower triangle at `i*(i+1)/2+j`, every j≤i
once; the residual asymmetry is reported in `reciprocity_defects` and never
averaged away, and `v_pp>0` (required by alpha positive definite) is recorded,
not enforced. The origin enum is `NativeDirectActualPointResponse`, which
`pfit.cc` accepts only with representation `native_point_charge_ov_operators`
and **no** declared auxiliary basis, since there is no auxiliary fit to name. It
must never be relabelled `SuppliedActualPointResponse` or
`SuppliedFittedPropagatorPointResponse`. The model stays with the caller:
`batch()` consumes the caller's design matrix and infers no channel, site,
parameter count or convention, and nothing is derived from a final Cn or another
track's parameter count.

This is a **prerequisite, not the historical target**, which is a
constrained-NN/distributed fitted-propagator quantity on a different point
lattice, with refinement, anchoring, frame/point conventions and the large
aVTZ response resource blocker all still unreproduced.

Details: `pfit.h`, `lw_localization.h`, [SUPPLIED_PROPERTIES.md](SUPPLIED_PROPERTIES.md)
and pre-compaction SPEC §6. Legacy PFIT leg-B gate is 3e−8 absolute/1e−8 relative;
do not transfer it to other targets or a new native endpoint.

### Distributed-model refinement driver

`psi4/driver/procrouting/isapol_refine.py` builds the `IsaPfitProblem` that
stands between raw per-frequency polarizabilities and refined distributed
tensors. It owns only the *model and penalty construction*; the arithmetic stays
in the certified `isa_pfit_solve`, and the interaction functions are the
bitwise-certified `isa_t_functions` (§6, `.pi/audit/t-functions/`).

Construction is transcribed from CamCASP's `write_pfit_local_symm`
(`src/tools/process_data.F90:2100-2230`, MIT, attribution in the module
docstring): one set of variables per unique **site type** in order of first
appearance, read off that type's **first** site (`indices(1)`), over the upper
triangle of `(lim+1)²` components; `lim == 0` contributes nothing; a component
pair survives iff the *reference site's* anchor exceeds `cutoff` in magnitude,
so a large value at an equivalent site does not rescue it; remaining sites of
the type become `COPY` and share the variable in their own local axes. Penalty
strengths come from `weights` (all seven types, including the mis-documented
`10.0e-3`/`10.0e-2` literals and the `/(1+ω²)` frequency scaling) and enter as
`strengths[k]·(z[k]−anchors[k])²` with the anchor as initial guess, matching
`read_penalties` (`src/pfit/process.F90:520-659`).

Frames are local-to-global **by column**, the same contract as `isa_t_functions`
and as CamCASP's `Axes` direction cosines, and are required proper orthogonal to
1e-12. Bounds are declared, not adjustable: `MAX_RANK 4`, `MAX_SITES 64`,
`MAX_POINTS 512`, `MAX_PARAMETERS 4096`. `refine()` requires an explicit
`target_origin`, `source_id` and `generation_record`, and enforces the same
origin/representation pairing as `pfit.cc`: a `NativeDirectActualPointResponse`
target must be `native_point_charge_ov_operators` with no auxiliary basis named.
A non-`Solved` status is returned, never repaired.

**Numeric parity with CamCASP `pfit`.** Three formatted-`Lattice` inputs
carrying the same sites, axes, `.pdef` `COPY` model, point cloud, point-to-point
responses and penalty anchors/strengths were run through upstream's own
`pfit`: the reference L2H1 water shape (55 parameters, 17 channels, 40 points),
a rank-4 oxygen model whose cutoff excludes 234 of 325 component pairs (101
parameters, 33 channels, 30 points), and a Tang–Toennies damped case (b=1.5,
weight type 5). Every fitted parameter agrees to the last printed digit —
max |Δ| 4.998e-09 / 4.961e-09 / 4.969e-09, relative 1.6e-09 / 5.6e-10 /
1.7e-09 — as do `R.m.s.` and max |residual| to all printed digits. `pfit`
prints with `f15.8`, so **5e-09 absolute is the oracle's resolution, not the
algebra's error**; `Print Polarizabilities` (`g16.8`) is no better and raising
it would require modifying the reference tree. Recorded in
`tests/pytests/test_isapol_refine.py`, whose inputs are dyadic rationals and
integer directions so no NumPy `Generator` stream stability is assumed. The
damped case independently certifies `isa_t_function_damping` against
`T_functions`' Tang–Toennies staging; the rank-4 case exercises rank-3/4 T rows
and the cutoff-exclusion branch.

**Measured cost.** `pfit.cc::data_rows` is an O(np·nc²) dense triple loop per
data row: 15,895 `finite`-guarded operations per point pair for the L2H1 model,
a marginal 1.441 ms/pair, linear in pair count. `parameter_tensors[k]` holds one
or two nonzeros, so ~99.7% of those operations multiply exact zeros; the kernel
is certified as-is and was not rewritten to make a demo cheaper. A 500-point
cloud projects to 180 s per sweep, 1000 points to 721 s, and `MAX_POINTS 512`
caps one refinement at 131,328 pairs (~190 s) — a CamCASP-scale 2000-point
lattice (~2,883 s) is refused by the driver rather than silently attempted.

This is the refinement *stage*, not end-to-end parity: the historical target
additionally needs the constrained-NN distributed response on the reference
point lattice (plan §5 items 1 and 6).

**The stage runs on the intended protocol.** `.pi/audit/avtz-grac-refinement-demo.py`
drives PBE0/aug-cc-pVTZ with the reference fixed-GRAC shift 0.06490004527520865 Eh,
`scf_type pk`, DFT 99/590, the full unpruned `IsaGrid(99,590)` response quadrature
admitted by `SHARED_SWEEP`, ISA-A with strict LW, then refines the LW local
tensors against this script's own native direct-OV point-charge targets
(`.pi/audit/avtz-grac-refinement-demo.json`). E = −76.37966827740793, SCF 2.58 s,
properties 25.53 s, peak RSS 1,492,476 KiB, nbf 92, nOV 435. The model is
**caller-declared** — ranks 2/1/1 on O/H, 17 channels, 17 parameters
(21 non-symmetric), cutoff 1e-4, weight type 3, coefficient 1e-3, no damping,
anchor SHA256 `f15a69d2…d117ee67` — because the historical `.pdef`, point
lattice and per-frequency `pfit` inputs are missing artifacts (plan §4); nothing
is read from, or inferred from, the reference `Cn` potential, and no parameter
count is borrowed from the dispersion track.

Two 150-point golden-angle lattices are declared and **both** are reported,
because a refined model is a property of the lattice it was refined on and a
single choice must not be presented as canonical:

| shells (bohr) | target min diag | target max\|·\| | reciprocity | rank | data rms | max residual | anchor shift max\|Δ\| (rel) | refined isotropic α (O, H, H) |
|---|---|---|---|---|---|---|---|---|
| 4.5/6.0/7.5 | 2.794e-03 | 4.947e-02 | 1.0e-16 | 17/17 | 4.047e-04 | 3.741e-03 | 2.852 (0.962) | 7.21270, 1.07847, 1.07847 |
| 7.5/9.0/10.5 | 7.385e-04 | 4.576e-03 | 1.7e-17 | 17/17 | 8.749e-05 | 6.844e-04 | 0.246 (0.355) | 7.10176, 1.38061, 1.38061 |

against anchors 7.10885 / 1.38106 / 1.38106. Both solve at full numerical rank.
The far lattice barely moves the anchors (O isotropic 0.1%, H 0.03%); the near
lattice, whose inner 4.5-bohr shell sits inside the density, moves H isotropic
by −22% — the refinement is doing what it is asked to do, and what it is asked
to do depends entirely on where the caller puts the probes.

`tests/pytests/test_isapol_native_point_response.py` certifies the same wiring
at the cheap sto-3g fixture: refining a test-declared isotropic rank-1 model
against real `native_point_charge_response` targets solves at full rank, beats
its own anchors in the packed-target rms (which reproduces the solver's
`data_rms`), preserves `COPY` equivalence and symmetry, holds the parameters at
the anchors under a large penalty coefficient, and refuses a mislabelled origin,
representation or auxiliary-basis claim.

### Error-bounded ALDA quadrature row screening

`IsaAldaGridScreen` (`native_response.h`) reports, per caller quadrature row, the
**exact** norm of that row's contribution to the ALDA local primitive. Row p adds
`factor(p)·tr_p tr_pᵀ` with `factor(p)=w(p)·fxc(p)` and `tr_p(t)=phi_i(p)phi_a(p)`;
since `tr_p` is an occupied×virtual outer product that contribution is exactly
rank one, so `‖factor(p) tr_p tr_pᵀ‖_F = |factor(p)|·o(p)·u(p)` with
`o(p)=Σ_i phi_i(p)²`, `u(p)=Σ_a phi_a(p)²`, and every element obeys the same
bound. `omitted_bound` therefore bounds the full-vs-pruned deviation of L in
**both** maxabs and Frobenius norm. Rows the primitive already skips (`rho` below
cutoff, or zero weight) have value exactly 0, so threshold 0 is **lossless**: the
pruned primitive is bitwise the same doubles, and the bound is exactly 0.

Retained rows are the input rows verbatim — same coordinates, same weights, same
order, no renormalization, no radial/angular reduction, no AO screening. The
class carries its **own** gate on `grid_rows·nbf·nmo` collocation work and
deliberately no nOV² term, which is the entire reason it may examine a grid the
primitive cannot yet afford. It grants no waiver: whatever row subset is finally
passed still faces `estimate_response_work`'s unchanged nOV² ALDA limit in full.

This is a screen, not a parity result. Measured for PBE0/aug-cc-pVTZ water on the
public `IsaGrid(99,590)` grid, lossless screening still needs 12.8× the permitted
ALDA work, and the 10,569 rows that limit does admit omit 53.6% of the total
contribution norm — 6.94% of the isotropic dipole polarizability at cc-pVDZ,
where the full primitive is affordable. Screening does **not** make that demo
affordable; see plan §4.

### Two named native response algorithms

`NativeResponseProvider` takes an explicit `algorithm` string and accepts exactly
two values, `ordered_pairwise` (default) and `shared_sweep`. Nothing is inferred:
an unknown, empty or abbreviated name is rejected in the constructor, in
`estimate_response_work`, in `native_response_from_wavefunction`, and by the
public `ATOMIC_RESPONSE_ALGORITHM` option's own choice list. The name is part of
`isapol_native._policy`'s hash, so a context built under one arrangement is never
reused under the other.

Both arrangements evaluate the *same* ordered nbf⁴ shell-quartet sweep and the
*same* ordered ALDA quadrature. They differ only in how that work is arranged:

- `ordered_pairwise` re-runs the quartet sweep once per `(b,j)` transition pair
  and accumulates the local primitive with the hand triple loop. This is the
  shipped, calibrated arrangement and is unchanged.
- `shared_sweep` visits the quartet sweep **once**, updating every `(b,j)`
  transition inside it, parallel over the outer shell `s0` only — shell `s0` owns
  rows `mu` of both J and K, so workers write disjoint output rows and there is no
  reduction and no reordering. Each worker builds its own `libint2::Engine` at
  `set_precision(1.e-15)`. Its ALDA local primitive is one blocked DGEMM per
  128-row block with the identical left-factor scaling.

**Agreement.** `coulomb()`, `exchange_direct()` and `exchange_transpose()` are
**bitwise identical** between the two, because every addend arrives in the same
`s0,s1,s2,s3` then `m,n,r,s` order as the same left-associated
`eri*co(rho,j)*cv(sigma,b)`. `local_primitive()` agrees **to rounding, not
bitwise**: the DGEMM reorders summation inside a 128-row block. Measured, cc-pVDZ
(nbf=24, nOV=95, 4000 rows): V/X/Y bitwise, L max|d| 4.235e-22 (rel 2.730e-16),
H1 rel 2.398e-26. aug-cc-pVDZ (nbf=41, nOV=180): V/X/Y bitwise, L rel 2.016e-16,
H1 rel 1.946e-26. `test_shared_sweep_reproduces_ordered_pairwise` asserts the
bitwise operators, checks L against the independent analytic/LibXC oracle, and
checks L/H1/H2 against the accumulator at 1e-13 — never claiming bitwise for L.

**Separate gates, neither authorizing the other.** Both keep the same
`nOV·nbf⁴ ≤ 6.4e10` AO limit, the same `nbf ≤ 256`, `nOV ≤ 512` and
`grid_rows ≤ 1e6`. The ALDA `grid_rows·nOV²` limit is per algorithm,
`ALDA_WORK_LIMITS = {"ordered_pairwise": 2e9, "shared_sweep": 6.4e10}`, because
the *measured rate* per unit of gated work differs: 9.08 GFLOP/s for the
accumulator versus 104 GFLOP/s for the blocked update, i.e. both limits are
≈0.5 s of accumulation at the same wall-clock budget. Measured end-to-end,
4000 rows: cc-pVDZ 5.05 s → 0.16 s (31.8×), aug-cc-pVDZ 33.78 s → 1.93 s
(17.5×). No caller argument raises either limit; asking for the faster primitive
is asking for a different primitive, and `estimate_response_work` is still called
on the real dimensions on both paths. `shared_sweep` declares its extra
`2·nOV·nbf²` doubles of J/K workspace in the same gated `planned_bytes_`
envelope, asserted exactly by the test.

**What this unblocks, measured.** At the reference protocol's dimensions
(PBE0/aug-cc-pVTZ water, fixed GRAC 0.06490004527520865, `scf_type pk`, DFT
99/590, full unpruned `IsaGrid(99,590)` = 173,460 rows, nbf=nmo=92, nOV=435,
16 threads): `ao_work` 31,163,093,760 and `alda_work` 32,822,968,500 are the same
for both, but `ordered_pairwise` fails its ALDA limit (`max_grid_rows` 10,569)
while `shared_sweep` passes (`max_grid_rows` 338,221). The direct-API response
then constructs in **15.29 s** with `planned_bytes` 113,886,352, giving
α_iso **9.870961449318902** bohr³ (diag 10.389483143568143, 9.414895197052905,
9.808506007335655; off-diagonals ≤1.07e-13) and L_maxabs 0.14909129325125306.
The public `oeprop` endpoint at the same protocol completes with **zero stage
failures** (strict LW 1e-6 passed, 37 ISA iterations) in 2.61 s SCF + **24.49 s**
properties at **1,488,060 KiB** peak RSS, E=-76.37966827740807, atomic dipole
α = 7.108845614906964, 1.3810579153027442, 1.381057910633834 bohr³ (summing to
the molecular α_iso).

Those two aVTZ costs are quoted separately on purpose. The ALDA accumulation the
new gate covers is ≈0.6 s of the demo; the `ordered_pairwise` `(b,j)` quartet
loop that `shared_sweep` replaces was separately measured at **953.9 s** with
`no_local`. `shared_sweep` addresses that loop as well as the primitive, which is
why the whole response is now seconds — but the two are never one number.

This is a resource result and a demo at the reference protocol's basis and
SCF-input policy. It is **not** a parity claim: the run above is still SPEC track
1's generated H/O JKFIT/even-tempered recipe with ordinary ISA-A and direct-OV
response, not the trace's constrained NN → distributed response → LW → PFIT path
(plan §4, §5 items 1 and 6).

## 7. Dispersion coverage and oracle scope

Scalar rank-l alpha is trace(alpha_ll)/(2l+1). Explicit CP weights already contain
**1/(2*pi)**; static weight is zero and skipped before CP multiplication, not
necessarily before standalone tensor construction. Exact matching grids, finite
positive dynamic weights and nonempty positive quadrature are required.

```text
C6 = 6*sum_f(w_f*alpha_A(f)*alpha_B(f))
isotropic rank factor = binomial(2*lA+2*lB, 2*lA)
isotropic order n = 2*lA+2*lB+2
general anisotropic order n = la+lap+lb+lbp+2
```

Complete isotropic C12 needs (1,4),(2,3),(3,2),(4,1); rank3 C12 is partial.
Declared ranks run 1..4 over the **13 ordered pairs upstream defines** (`la+lap<=6`,
casimir.f90 read_cg/recouple `if (j1+j2>6) cycle`); (3,4),(4,3),(4,4) are never read
upstream and `alpha_c` is uninitialized there, so they stay structurally absent and
their C11/C12 quadruples are reported missing, never summed as zero. Shipped-table
included quadruples per order n=6..12 go 1,4,10,16,19,16,10 (rank1-3) to
1,4,10,20,31,36,34 (rank1-4); the 4+10 remaining gaps at n=11,12 are exactly those
needing an uninitialized pair, so `table_complete` is False there by construction.
General anisotropic unrestricted C10/C12 can need individual ranks5/7. Preserve
missing-rank diagnostics; supplied zero blocks differ from absent ranks. The
orientation-resolved scalar/energy engine is not recoupled `Cn(t,u,J)`.

Recoupled engine uses supplied **local-axis** tensors, ordered rank pairs and a
bilinear CP product: no conjugation, extra spin/off-diagonal factors or prefactors.
Check imaginary residue per phased integral at strict <1e−8; retain structural zeros
and every ordered site pair. No R/damping/coincident-site restriction is implied.
Exact CP and table-term order is part of the contract.

- Rank-4 casimir oracle (CamCASP 6.0.051, seeded synthetic mixed-rank 1..4 deck):
  **369 recoupled components over 13 ordered pairs at g14.6 write precision**
  (3473 nonzero / 117 written zeros / 10 `all zero`) and **10029 Cn rows, 15877
  nonzero at rtol1e−6/atol0, 50155 written zeros, 4171 omitted trailing fields**,
  exact J≤8 rowset. Literal casimir print tokens, hash-pinned; the generator
  computes no expected value. 735 visible J9/10 rows counted, not certified.
- Archive: **6285 exact J≤8 rows, 10457 nonzero values at rtol1e−6/atol0,
  30791 written zeros at abs≤1e−6**, plus threshold checks for omitted trailing
  fields. Hash-pinned literal expected output, not generated by the new engine.
- Independent maximal J9/C11 and J10/C12: **735 rows / 7 blocks**, 26 ordered
  quadruples / 11 observable reciprocal classes, factorial/electrostatic oracle.
- Independent strict lower-J, **even L+H+J**: 224 signal blocks / 10056 required
  rows, 76 ordered quadruples / 33 reciprocal rank-pair classes. Includes structural
  zeros and required-row checks for full and compressed ranks.
- **5341 odd strict-lower-J rows remain uncertified**. Zero-m angular normalization
  is unresolved; raw identities prove surviving channels, not coefficient values.
  User deferred further RRF investigation. The archive's 411 visible high-J rows
  still lack a literal high-J oracle; synthetic validation is a separate boundary.
- General maximal channels of lower orders and full arbitrary-rank theory are
  outside those dedicated numerical oracles. Reciprocal inputs cannot identify
  compensating errors inside unobservable ordered-term classes.

Use explicit quadrature nodes/weights. Standalone/SAPT beta0.3, ISA reference beta0.5
and actual root-generated response quadrature are not automatically interchangeable.

Details: [ANISOTROPIC_CONTRACT.md](ANISOTROPIC_CONTRACT.md),
[RECOUPLED_CONTRACT.md](RECOUPLED_CONTRACT.md),
[HIGH_J_VALIDATION.md](HIGH_J_VALIDATION.md),
[LOWER_J_VALIDATION.md](LOWER_J_VALIDATION.md),
[ODD_J_VALIDATION.md](ODD_J_VALIDATION.md).

## 8. Tolerance and evidence policy

Existing passing tests keep their tolerances. No profile waives dimensions,
finiteness, reciprocity/PSD policy, rank/conditioning, ownership, provenance,
cache generations, schema/safety or missing algorithms. Do not replace failed
coefficient checks with nicer downstream observables or alter error normalization.

Previously authorized **explicit** profiles remain narrowly available:
- `provisional-1e-3`: struggling forward comparisons only; retain original strict
  result, same metric/units, source/input provenance and tightening TODO.
- `provisional-drho-1e-2`: **Drho-C forward-error allowlist only**. Metric/RHS,
  electron counts, residual and structural checks stay strict. Other stages cannot
  select it. The original 1e−3 profile retains its meaning even for Drho-C.
- Custom legacy trajectory tolerance is labelled uncertified, not a named profile.

Historical raw tails ~2.35888e−8 and fitted-OV coefficients ~2.30198e−5 fail the
strict scaled1e−9 gate though they passed their named1e−3 comparison. Native Drho-C
~0.00128021885 fails1e−3 and strict1e−9; it passed only the separately selected
Drho-C1e−2 profile. These are prior scoped measurements, not fresh native protocol
acceptance. Per-stage allowances imply no end-to-end bound. Strict response replay
scaled1e−9 and default LW1e−6 remain separate from these forward profiles.

The native point-charge prerequisite carries its own gates, transferable to
nothing else: an analytic AO ESP oracle (`math.erf` only, no scipy; Boys F0–F2
by series below 1 and upward recursion above) over **every** shell of the
fixture basis, s and p, in both pure and Cartesian orderings, at
**atol1e−13**; the owned W against an independent MO transform at
**atol1e−13**; the bounded npoint-RHS solve against both the full nov-RHS
contraction and explicit `np.linalg.solve` at **rtol2e−9/atol2e−12**; and a
far-field `(R_p^T alpha R_q)/(R_p^3 R_q^3)` limit required to *halve* per
doubling of R rather than meet a fixed threshold, with alpha built in-test from
independent dipole OV legs. Reciprocity defect is asserted **<1e−12** and
reported, never symmetrized.

The right-hand-side factor 4 has its own absolute gate, because none of the
above can see it. Configured at `exact_exchange=1`, `kernel='no_local'`, H1/H2
are the closed-shell (A+B)/(A−B) matrices and the ω=0 native solve is
coupled-perturbed Hartree-Fock, so `alpha = -D^T C D` from dipole OV legs must
equal (i) the 3×3 tensor from Psi4's own iterative `Wavefunction.cphf_solve`
via `psi4.properties`, at **atol1e−9** on the full tensor — a different solver
carrying its own independently written restricted prefactor — and (ii) the
curvature of *perturbed SCF total energies*,
`alpha_kk = −d²E/dλ_k²`, at **rtol1e−6/atol1e−8** after one Richardson step
over h=8e−3 and 4e−3, with the h-halving error ratio itself asserted
**=4 ±5%**. Gate (ii) involves no response theory, no orbital Hessian and no
prefactor, and a central second difference is even in λ, so it is also
independent of Psi4's `perturb_dipole` sign convention. Measured: 2.1e−14 on
(i), 5.0e−9 on (ii). Verified discriminating by mutation: replacing the factor
with 2.0, 8.0 or 4.5 fails both gates.

The `a` and `b` kernel scalings have their own absolute gate, away from that
corner. The pre-existing ALDA gates re-derive the written
`H1=Δ+4V−a(X+Y)+4bL` and use only complementary `(a,1−a)` pairs, so they can
neither see a wrong overall factor on `L` nor separate the two scalings from
their sum. The gate instead builds a **matched** custom functional —
`x_hf` scaled by `a`, `LDA_X` and its LDA correlation partner by `b`, LibXC
names unprefixed — whose CPKS kernel *is* the native operators at `(a,b)`, and
compares (i) `sqrt(eig(H2·H1))` against Psi4's Davidson `tdscf_excitations`
(`scf_products.py`: `twoel_Hx_full`, `onel_Hx`, `compute_Vx`; requires
`save_jk`), at **atol1e−11**, measured 7e−14…1.8e−13 over
`(0.25,0.75,pw92)`, `(0.5,0.5,vwn)`, `(0.3,0.9,slater)` and `(0,1,pw92)`;
and (ii) perturbed matched-RKS total-energy curvature, at
**rtol1e−5/atol1e−9** with the same asserted h-halving ratio **=4 ±5%**,
measured 1.9e−9 (rel 4.1e−8, ratio 4.0000) at `(0.25,0.75,pw92)` and 9.4e−9
(rel 2.9e−7, ratio 4.0009) at `(0.3,0.9,slater)`. Gate (i) is the only one
anywhere in this track that reaches **H2**: at ω=0 the solve
`(H2·H1+ω²)X = −4·H2·D` collapses to `−4·H1⁻¹D` and H2 cancels identically, so
no static polarizability can see it. `(0.3,0.9)` is deliberately
non-complementary. Gate (ii) is absolute — no response theory, no orbital
Hessian, no prefactor, no field sign convention — and is *not* degraded by the
coarse (50,25) grid: the second difference of the grid-discretized `E_xc` is
the grid-discretized `f_xc`, and the native `L` is built on the SCF's own grid,
so the quadrature error cancels between the two sides rather than entering as
an error; checked against (590,99)/168,883 rows, which agrees no better.
Grid size is not free, though: (74,35) makes Psi4's Becke pruning emit 832
negative weights (min −92.15), which the provider's grid guard rejects,
correctly. Verified discriminating by mutation of the staged provider call:
`local_scale*1.01` or `exact_exchange+0.001` fails all six layer-6 tests, and
in-test controls resolve `a` to 0.001 (spectrum moves 6e−4), `b` to 1%
(1e−4), and `a` **in H2 alone** to 0.001 (3e−4), against a 1e−13 baseline.

What those gates still do not cover must not be described as if they did.
`core.ExternalPotential.computePotentialMatrix` reaches the same
`libint2::Operator::nuclear` integrals the C++ drives, so the MO-transform gate
certifies the AO→MO transform and OV packing, **not** the kernel; only the
analytic oracle pins the kernel. No shell above p is exercised, because the
fixture basis has none, and neither the factor-4 nor the `a`/`b` gate touches
that. The `a`/`b` gate covers the three ALDA kernel names at four `(a,b)`
points including the shipped `isapol_oeprop` default `(0.25,0.75,pw92)`, but
`b` is anchored absolutely against an *energy* only at `(0.25,0.75,pw92)` and
`(0.3,0.9,slater)`; the remaining two points rest on the excitation-energy
gate alone. Both gates are ω=0 or excitation-energy statements about the
operators, not about frequency-dependent propagator conventions. These certify the
prerequisite only; they are not matched-protocol acceptance and must not be
conflated with the legacy PFIT leg-B gate.

### Property-anchored precision budget

A per-stage error becomes a scientific statement only once it is tied to a
property tolerance. `psi4.driver.procrouting.isapol_budget` answers, for each
named intermediate X and each property group Y, how precisely X must be known:
it perturbs X, rebuilds the **whole** downstream through the shipped objects
under the unrelaxed production LW policy, and reports the amplification
`A = defect(Y)/defect(X)`; the required precision is `tolerance(Y)/A`. Seven
intermediates are named: `drho_c_coefficients`, `shape_coefficients`,
`raw_tail_parameters`, `partition_shape_samples`, `ov_transition_legs`,
`coefficient_responses`, `distributed_site_tensors`. Two are probed as raw
parameters rather than as sampled arrays, each because that is the form in
which its error was recorded. `raw_tail_parameters` is the per-site joint
`(amplitude, exponent)` of the Func-1/Fit-3 exponential tail; its `cutoff` is
supplied configuration, held fixed exactly as the trajectory comparator holds
it fixed, and outside the probed array, and a non-positive perturbed exponent
is rejected rather than sampled. `shape_coefficients` is the concatenated
per-site ISA-A coefficient vector W from which the final shapes are sampled;
across that probe the shipped tails are **held fixed**, which is the
algorithm's own boundary and not a convenience — `IsaAController::step` fits
iteration n+1's tails from iteration n's coefficients (the documented source
lag) and the final sampling re-uses the stored tails without refitting, so
refitting a tail from a perturbed *final* W would model a different algorithm.
Both rebuilds build a *surrogate* controller state, so the shipped state is
never mutated.

Three disciplines make the number mean something. First, the unperturbed
rebuild must reproduce every shipped property group at **exactly 0.0**, so the
rebuild path *is* the shipped computation and not a re-implementation of it;
this is asserted on both fixtures and every available stage. Second, a probe
direction must preserve every structural invariant the strict LW gate enforces
— charge-neutral OV legs, symmetric charge-null coefficient responses,
reciprocal charge-sum-free pair tensors — so a measured defect is physics and
not the gate refusing malformed input; a direction that cannot be projected
onto its manifold raises rather than being probed. Third, each amplification is
measured at `eps` and `eps/2` and is **withheld** unless the two agree to
`linearity_tolerance`; on the water chain 25 of 112 rows are correctly withheld.
Every quoted A is a *first-order directional lower bound*: meeting the
requirement is necessary, not sufficient. The whole facility is labelled
`diagnostic_only_first_order_directional_lower_bound_not_a_gate` and is not a
gate.

Two probe geometries exist because two error models do. The `absolute`
geometry uses the recorded-error metric `max|actual−reference|/max(1,max|reference|)`,
which is the right question only for an intermediate whose elements share a
scale. The shape samples span many decades, and a max-scaled probe adds
`eps·max(shape)` uniformly, swamping the exponentially small tail where
`w_a/Σw` actually lives: measured on water, the absolute amplification for
`alpha_iso_rank1` **rises** 5.6e4 → 2.4e5 → 6.3e5 as eps falls 1e−6 → 1e−8 →
1e−10 (and 1.8e6 → 1.4e7 → 6.6e7 for `alpha_iso_rank3`), with a linearity
defect never below ~0.1. That probe is not measuring a derivative and no
absolute requirement for that stage is well posed. The `relative` geometry
therefore probes `max(|actual−reference|/|reference|)` over the reference
support, leaving exact zeros exactly zero, and converges: 8.9e−4 → 7.8e−4 and
9.8e−3 → 8.5e−3 over the same eps range. It is offered **only** for the two
stages whose downstream is regenerated from scratch; the restricted stages
carry linear invariants an elementwise multiplicative probe does not preserve,
and asking for it there raises. The two metrics are not commensurable, and
`precision_budget` **refuses** to compare a recorded error measured in one
against a requirement derived in the other rather than performing the
comparison silently.

Two intermediates are additionally reported in a **second input form**, because
their reference comparison recorded two errors and not one. Drho-C records the
error on its auxiliary coefficients (1.28021885e−3 scaled) *and* on the density
those coefficients sample to (6.13011642e−7); the fitted-OV route records its
coefficients (2.30197983e−5) *and* the sampled transition density
(5.03034372e−8). Such a stage is probed **once** and reported **twice**: one
property defect divided by the probed array's own defect (`as_probed`) and by
the defect that same perturbation induces in the sampled field
(`sampled_density`, `sampled_transition_density`), computed with the sampler and
the metric the comparator itself used — `IsaFixedDensity` on the shipped ISA
grid, and the screened auxiliary samples contracted with the fitted legs, both
streamed in blocks so the point count sets no array size here. A bare recorded
number names the probed array alone: `precision_budget` refuses to compare it
against a sampled-form requirement, refuses a sampled-form recorded error
outside the absolute metric, and offers no sampled form in the relative
geometry.

The two recorded numbers differ by three orders of magnitude because the
recorded coefficient error lies close to the null space of its own expansion
map. That is a property of the recorded error, not of the map: an amplification
is a property of the *image* direction, a generic coefficient probe reaches a
generic image direction, and the measured sampled-form amplification is
therefore not depressed by the same cancellation. It is a measured directional
lower bound over the **range** of that stage's own expansion map — the subspace
any coefficient error inhabits — and not a bound over arbitrary sampled fields.
`IsaFixedDensity` has no tabulated constructor, so a sampled-density *stage*
cannot be perturbed freely, and the induced defect of the coefficient probe is
the honest construction rather than a chosen one.

Linearity of the expansion is measured here, not assumed: the sampled/probed
defect ratio is eps-independent to 6–7 digits across eps = 1e−6, 1e−8 and 1e−10
in every direction (water Drho-C 4.414114, 1.202447, 1.159737, 3.229728; He
fitted-OV 22.238682, 13.048567, 14.731959, 14.403853), so a sampled-form
amplification inherits the probed-form linearity defect exactly and the two
forms are quoted or withheld together — including the one He direction whose C8
row exceeds the tolerance in both forms at once. Adding the second form did not
disturb the first: the re-measured probed-form water rows reproduce the earlier
run to 4.4e−6 relative, and the He rows reproduce it bit-for-bit.

Measured requirements at a 1e−6 property tolerance, evidence in
`.pi/audit/property-anchored-budget.json`:

| stage | chain | metric | binding A | required | recorded | meets |
|---|---|---|---|---|---|---|
| `drho_c_coefficients` | water direct-OV | absolute | 3.60e2 (α₃) | 2.78e−9 | 1.28021885e−3 | **no**, by ~6 orders |
| `drho_c_coefficients` → `sampled_density` | water direct-OV | absolute | 1.11e2 (α₃) | 8.98e−9 | 6.13011642e−7 | **no**, by 68× |
| `drho_c_coefficients` | water direct-OV | relative | 3.64 (α₃) | 2.75e−7 | — | not measured |
| `partition_shape_samples` | water direct-OV | relative | 9.75e−3 (α₃) | 1.03e−4 | — | not measured |
| `raw_tail_parameters` | water direct-OV | absolute | 1.52e1 (α₃) | 6.59e−8 | 2.35888047e−8 | **yes**, 2.8× margin |
| `shape_coefficients` | water direct-OV | absolute | 3.11e1 (C12) | 3.22e−8 | 3.01435416e−11 | **yes**, 1.07e3× margin |
| `coefficient_responses` | water direct-OV | absolute | 2.50e1 (α₃) | 4.00e−8 | — | not measured |
| `distributed_site_tensors` | water direct-OV | absolute | 1.52e2 (C12) | 6.58e−9 | — | not measured |
| `ov_transition_legs` | He fitted | absolute | 1.61e3 (C10) | 6.21e−10 | 2.30197983e−5 | **no**, by ~4.5 orders |
| `ov_transition_legs` → `sampled_transition_density` | He fitted | absolute | 7.24e1 (C10) | 1.38e−8 | 5.03034372e−8 | **no**, by 3.6×; 4 of 7 groups **yes** |
| `coefficient_responses` | He fitted | absolute | 2.28e5 (α₃) | 4.39e−12 | — | not measured |
| `distributed_site_tensors` | He fitted | absolute | 3.78e1 (C12) | 2.65e−8 | — | not measured |

Two chains are needed and neither substitutes for the other. Water
(PBE0/cc-pVDZ, generated recipe, direct-OV, three sites) is the only
non-degenerate partition measurement, but the **fitted-auxiliary route on water
is rejected outright** by the strict production LW gate — charge-sum ≈4.1e−4 at
every frequency — so the fitted-OV intermediate cannot be anchored there and is
anchored on He instead. He is monatomic, so `w_a/Σw ≡ 1` and the two partition
stages are *exactly* degenerate on it (A = 0); their "satisfied" rows on He are
an artifact of that degeneracy and are labelled as structurally insensitive,
not offered as evidence. The rejection was not worked around: relaxing the
residual policy to admit it is forbidden and was not done.

The recorded 2.35888047e−8 is a joint-tail **parameter** error, so it is
compared against a parameter stage and not against the sample array it was
previously read against. That comparison is apples-to-apples by an *identity*,
not an approximation: the comparator scales each site's joint
`(amplitude, exponent)` error by that site's own largest parameter, this module
scales one array by its single largest, and on the shipped reference the site
carrying the largest error also carries the largest parameter — so
`max(1.63572023e−7)/max(1,6.93430743) = 2.35888047e−8` bit-for-bit. The
identity is re-derived from `tests/pytests/data_isapol/psi4_provisional_acceptance_evidence.json`
in the test suite rather than asserted. Because tail parameters are O(1)
numbers sharing one scale, the absolute geometry *is* their property-relevant
error model, and the amplification is a converged derivative: A(α₃) = 1.5176e1
at eps = 1e−6, 1e−8 and 1e−10 alike (five figures), with every row quoted at
every probe size. At a 1e−6 property tolerance the recorded tail error
**meets** its requirement with a 2.8× margin; it would first fail at a property
tolerance of 3.6e−7.

The shape samples are anchored by the same move, one step upstream. The
comparator records a per-site error for the ISA-A coefficients W, and there
*two* of the three water sites have a clamped denominator, so their
coefficients cannot exceed one and the concatenated array's single denominator
is exactly the unclamped site's: `max(2.07295715e−10)/max(1,6.87695288) =
3.01435416e−11`, again an identity re-derived from the shipped evidence file in
the test suite rather than asserted. Note this is *not* the per-site maximum
(2.07295715e−10): unlike the tails, the largest error and the largest
coefficient sit on different sites, so quoting the per-site number for the
concatenated array would be wrong by 6.9×. The coefficients are O(1) and share
one scale, so the absolute geometry is again their error model, and the
amplification converges: A(C12) = 3.108e1 at eps = 1e−6, 1e−8 and 1e−10 alike
(agreeing to 8e−5 relative), A(α₃) = 2.848e1, all 28 rows quoted at every probe
size, self-consistency exactly 0.0. At a 1e−6 property tolerance the recorded
coefficient error **meets** its requirement with a 1.07e3× margin, and would
first fail at a property tolerance of 9.4e−10. The shape-sample *array* itself
remains uncompared in both metrics — absolutely because no absolute requirement
for that stage is well posed, relatively because no error was ever recorded in
that metric — and that gap is a missing measurement, not a mismatched
association; what is now measured is the coefficient input that generates the
array, not the array.

What the second input form buys is four to five orders of the apparent gap, and
not the stage. On He fitted-OV it **closes four of the seven property groups**
at a 1e−6 tolerance (α₁ at 0.60× of its requirement, α₂ 0.14×, C6 0.21×, C8
0.31×) and misses the other three by 1.93×–3.64×, against 3.7e4× for the probed
form; the binding row is C10, whose coverage is itself structurally partial, so
the α₃ row (A = 3.978e1, requiring 2.51e−8, failing by 2.00×) is quoted beside
it. On water Drho-C it closes **none**: the miss runs from 14.3× (α₁) to 68.3×
(α₃), against 3.6e5× for the probed form, and the recorded sampled-density error
supports Drho-C-induced property defects of 1.43e−5 to 6.83e−5 in the max-scaled
metric — not 1e−6. Both sampled-form binding rows therefore remain
**unsatisfied**, and the comparison is still made on the PBE0/cc-pVDZ demo water
and the compact He model rather than on the comparator's own input, because the
exported reference carries no orbitals and a same-input property chain is not
constructible.

Of the six recorded errors the budget can compare, two (raw tails, shape
coefficients) meet their requirements outright; the two coefficient arrays —
Drho-C and fitted-OV — miss theirs by roughly six and four and a half orders of
magnitude in the form they were recorded in as coefficients, and the two sampled
fields those coefficients expand to miss by 68× and 3.6×. Neither stage carries
a 1e−6 property guarantee in either form, and the per-stage provisional profiles
above must not be read as implying one.

[PROVISIONAL_ACCEPTANCE.md](PROVISIONAL_ACCEPTANCE.md) owns exact opt-in policy;
its historical milestone statuses do not supersede current capability above.

## 9. Licensing, portability and acceptance discipline

CamCASP MIT source is permitted; retain Anthony Stone's copyright/permission and
Misquitta/Stone attribution with source anchors. Its historical license is available
via `git -C ~/gits/CamCASP show b40ae4f^:LICENSE`; do not reopen the resolved blanket
licensing question. This does not license every external bundled component/data set.
Retain the reference-basis MIT/BSD notices in source **and installed packages**.

Do not inspect/transcode ORIENT executable source or transliterate/link the separate
GDMA grid. RRF source/convention investigation is deferred; no automatic permission
follows from CamCASP's license. Do not ship extracted reference executables/source;
local extraction tools are development-only. Shipped production/tests must not
invoke external CamCASP/ORIENT/PFIT/CASIMIR or require home-directory reference data.
Portable fixtures need identity/hash/unit/representation provenance and notices.

Validate in dependency order: algebra/invariants → fixed-input stage → actual
native producer → stage integration → public endpoint → matched protocol. Preserve
both failed and successful evidence. Full-source read/replay is not native generation;
a final potential cannot supply missing intermediate provenance. Independently
review scientific boundaries; reviews do not replace builds/tests.

Use the nested configured core build and matching staged Python. Run pytest outside
the source package, filter explicit slow-test filenames before invocation, retain
SAPT regressions after shared response/SCF changes, and measure default output
compatibility/performance after affecting its path. See plan for exact commands.
Documentation-only compaction does not imply a new test/build run.
