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
  orientation-resolved dispersion, and independently callable rank≤3 recoupled
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
General anisotropic unrestricted C10/C12 can need individual ranks5/7. Preserve
missing-rank diagnostics; supplied zero blocks differ from absent ranks. The
orientation-resolved scalar/energy engine is not recoupled `Cn(t,u,J)`.

Recoupled engine uses supplied **local-axis** tensors, ordered rank pairs and a
bilinear CP product: no conjugation, extra spin/off-diagonal factors or prefactors.
Check imaginary residue per phased integral at strict <1e−8; retain structural zeros
and every ordered site pair. No R/damping/coincident-site restriction is implied.
Rank4 is explicitly rejected. Exact CP and table-term order is part of the contract.

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
