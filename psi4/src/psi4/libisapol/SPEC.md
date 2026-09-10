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

**Caller-declared OV charge penalty.** `fit_ov` solves `A D^T = T^T` with
`A = J + lambda*q q^T`, a rank-1 penalty on the *total* fitted charge of the
transition density. An OV transition density has exactly zero charge by MO
orthonormality, so the penalty converges a constraint the exact answer already
satisfies; it is **not** a tolerance and raising it relaxes no gate.
`run_native_properties(..., ov_charge_penalty=...)` makes the value an explicit
caller declaration (default 1.0; `direct_ov` forms no transition fit and accepts
only the default), records it in fit/result provenance and in the model string,
and deliberately keeps it **out** of the response policy hash — H1/H2 come from
the orbitals, so one native context serves every lambda.

Measured on water (PBE0/cc-pVDZ, generated recipe, 99/590 response grid,
`shared_sweep`, strict production LW at every point,
`.pi/audit/ov-charge-penalty-water.json`): the residual fitted transition charge
falls as exactly 1/lambda (`lambda*charge` ≈ 1.5e−4 for lambda≥1e2; lambda1's
1.2366e−4 is not yet asymptotic), and the LW `input-sum-rule` residual is
proportional to that charge (≈3.35×) until it reaches a lambda-independent
1.6e−8–3.9e−8 floor. That floor **equals the fit-free `direct_ov` route's own
2.283e−8 on the identical grid**, so it is a property of the response quadrature
and not of the constraint. Strict production LW therefore rejects lambda1 on 7/7
nodes (4.14394e−4 … 3.03216e−6) and lambda1e2 on 5/7 (4.99918e−6 … 3.03369e−6),
but **accepts** the water fitted-auxiliary chain at every declared lambda≥1e3.
Two accepted lambdas are reported separately rather than as one "converged"
claim, because they differ in *margin*, not in physics. **lambda1000 is the
traced route's own value** — the exported constrained-NN state carries penalty
`(1000, 0, 0)` and the replay oracle refuses anything else — and it passes with
`input-sum-rule` 4.6538e−7, only 2.1× under the gate and still
penalty-dominated (3.09× the fitted charge, 20× the quadrature floor).
lambda1e4 reaches that floor: 2.3313e−8, 43× under the gate and 1.02× the
fit-free route's own 2.2834e−8. Between them the model barely moves — raw
tensors 1.345e−7 in the absolute metric, C6 total 18.199203613 vs
18.199203034 — so the choice is a margin declaration. Intermediate decades
interpolate monotonically (lambda2e3: 2.1298e−7; lambda5e3: 6.1518e−8). The
algorithm-controlled residuals sat at machine level at every lambda (off-site
≤5.7e−15, reciprocity ≤2.9e−14): only the supplied-input class ever failed, which
is why this fixes the **producer** — `reported_input_sum_rule` was never used.
Conditioning of `A` degrades linearly in lambda: lambda1e4 and lambda1e6 agree to
5.7e−7 in the raw tensors and 4.7e−7 in the C6 total, while lambda1e8 deviates by
7.3e−5; on He, where the constraint already holds to 4.86e−15 at lambda1, raising
lambda only injects noise (atomic-scalar movement 7.6e−14 → 8.4e−12 → 1.18e−9 for
1e2 → 1e4 → 1e6). Declared water values: **1000** to reproduce the traced
route, **1e4** where the constraint is to be at the quadrature floor.

A chain at any lambda other than the recorded one is a *differently declared
model*. The archived recorded-input D and the `ov_transition_legs` error
2.3019798321950356e−5 below are lambda1 numbers, so the lambda1e4 chain must
never be quoted against them, and must not be used to close the water fitted-OV
budget row — that row stays unmeasurable at lambda1, now for a precisely
understood reason. See `test_isapol_native_charge_penalty.py`.

**The declared molecular AUX, not the penalty, is what decides acceptance at a
larger MAIN.** The generated recipe's molecular AUX also carries the Drho-C/ISA-A
density fit, so it is now a declared argument, `generated_recipe(..., aux_basis=)`
and option `ATOMIC_PROPERTY_AUXILIARY_BASIS` (default `cc-pVDZ-JKFIT`,
deliberately *not* MAIN-matched, never inferred from `BASIS`/`DF_BASIS_SCF`), and
its name is carried in `NativeProperties.model` as `Drho-C ISA-A[<name>]` so two
partitions cannot be confused. Measured at PBE0/aug-cc-pVTZ water with the
reference GRAC shift .06490004527520865, nbf 92, E=−76.3796682774079, 99/590
response grid, 173460 rows, `shared_sweep`, 11 Casimir nodes, ISA 37 iterations in
every case, strict production LW throughout (`.pi/audit/avtz-aux-match.json`):

| AUX (shells) | route | fitted charge | input-sum-rule | LW fail | pair defect vs `direct_ov` | C6 total |
| --- | --- | --- | --- | --- | --- | --- |
| cc-pVDZ-JKFIT (42) | `direct_ov` | 0 | 6.2779e−9 | 0 | 0 | 46.897125 |
| cc-pVDZ-JKFIT | lambda1e3 | 1.8601e−6 | **6.6463e−6** | **9/11** | 0.687237 | — |
| cc-pVDZ-JKFIT | lambda1e4 | 1.8605e−7 | 5.4480e−7 | 0 | 0.687237 | 42.129547 |
| aug-cc-pVTZ-JKFIT (58) | `direct_ov` | 0 | 6.2986e−9 | 0 | 0 | 46.897125 |
| aug-cc-pVTZ-JKFIT | **lambda1e3 (traced)** | 1.4438e−7 | **2.7921e−7** | **0** | 0.0791458 | 46.768291 |
| aug-cc-pVTZ-JKFIT | lambda1e4 | 1.4440e−8 | 4.7555e−7 | 0 | 0.0791458 | 46.768425 |

So at the larger MAIN the traced lambda1000 is rejected with the default AUX and
accepted with the MAIN-matched one, and the naive reading — "the penalty is not
converged" — is measurably wrong: the defect against the fit-free route is
identical to five digits at 1e3 and 1e4 for each AUX, and 8.7× smaller for the
matched AUX at the traced lambda itself. Choosing the AUX is therefore declaring
a different model with its own partition, never a relaxation: `residual_policy`
stays `production`, tolerance 1e−6, and no grid or penalty moves. The two AUX
partitions are not comparable to each other except to report the distance:
molecular isotropic C6 is partition-invariant for `direct_ov` (46.8971254018
bit-identical under both), C8/C10 totals are not (825.9105/15805.9658 vs
827.0344/15864.8357). The residual defect the accepted matched chain still
carries against `direct_ov` is concentrated in the rank-3 column (O 177.75 vs
165.23; H 9.2370 vs 2.9586) — a component the reference's `H-Limit 1` model does
not carry at all. See `test_isapol_matched_auxiliary.py`.

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
1e-12. Because a `COPY` variable is shared in each site's own local axes, the
model also reports `copy_anchor_discrepancy`, the largest gap between a
variable's anchor and the same component at an equivalent site; it is zero
exactly when the caller's frames really do make the type's sites equivalent (see
below). Bounds are declared, not adjustable: `MAX_RANK 4`, `MAX_SITES 64`,
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

The refinement stage now also runs on the reference case's **own** point
lattice, driven by the constrained-NN chain of plan §5 item 1; the end-to-end
`Cn` comparison that consumes it is in §7. The two demos immediately below stay
what they are — staged runs on caller-declared lattices — and are not restated
as parity.

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

**Refinement on the constrained-NN chain, and the frame its `COPY` needs.**
The demo above anchors on ISA-A/oeprop, which is not the trace's path.
`.pi/audit/avtz-nn-refinement.py` instead refines the **constrained-NN** chain's
own LW local tensors — same protocol, plus the MAIN-matched `aug-cc-pVTZ-JKFIT`
AUX and the traced `lambda=1000` — with the fit-free `direct_ov` row refined
alongside it off the *same* native context, so only the anchor and penalty
centre differ. E = −76.37966827740806, SCF 2.67 s, `direct_ov` 25.48 s,
constrained NN a further 7.82 s, peak RSS 1,494,280 KiB, nbf 92, nOV 435,
`input-sum-rule` 6.299e-09 / 2.792e-07. Targets are formed once per lattice and
shared, and are declared for what they are: this script's own
`NativeDirectActualPointResponse` point-charge quantities. Feeding
constrained-NN anchors does not turn them into the reference's
`SuppliedFittedPropagatorPointResponse` target.

| anchors from | shells (bohr) | rank | data rms | max residual | anchor shift max\|Δ\| (rel) | refined isotropic α (O, H, H) |
|---|---|---|---|---|---|---|
| `direct_ov` | 4.5/6.0/7.5 | 17/17 | 3.593e-04 | 3.273e-03 | 2.278 (0.539) | 7.27406, 1.04740, 1.04740 |
| `direct_ov` | 7.5/9.0/10.5 | 17/17 | 3.303e-05 | 3.798e-04 | 0.029 (0.038) | 7.12793, 1.36796, 1.36796 |
| `lambda=1000` | 4.5/6.0/7.5 | 17/17 | 3.589e-04 | 3.267e-03 | 2.185 (0.538) | 7.27529, 1.04662, 1.04662 |
| `lambda=1000` | 7.5/9.0/10.5 | 17/17 | 3.329e-05 | 3.817e-04 | 0.053 (0.038) | 7.12452, 1.36549, 1.36549 |

against anchors 7.12410 / 1.37343 / 1.37343 (`direct_ov`) and 7.10024 /
1.36972 / 1.36972 (`lambda=1000`). The **measured ordering** is that refinement
dominates the response basis at the site level: the constrained-NN fit moves the
anchors by 0.024 (O) and 0.0037 (H) in isotropic α, while refining on the near
lattice moves them by ~0.15 (O) and ~0.33 (H). Where the two rows differ is not
the dipole: sorted by the ranks each variable couples, the pure dipole variables
move by at most 2.5e-03 of the largest anchor while the variables touching rank
2 move by 2.0e-02 of it, eight times as much (forty times at PBE0/cc-pVDZ). The
constrained-NN fit's site-level footprint is therefore a **quadrupole** effect,
and a dipole-level agreement between the rows is not agreement of the localized
model.

Frames had to be declared to get here, and that is a structural finding, not a
detail. A `COPY` equivalence is expressed in each site's **own local axes**, so
declaring one commits the caller to sites whose local tensors coincide — and
under global-identity frames (`frames=None`, LW's explicit default) water's two
hydrogens are mirror images, with the in-plane `10,11c` dipole coupling carrying
opposite signs (±0.703 at PBE0/cc-pVDZ). `refinement_model` reads the
*reference* site's value and `refine` writes it to every site of the type with
the same sign, exactly as CamCASP does, so in that frame the second hydrogen's
own anchors are unreachable: a hard-pinned fit misses them by twice the
coupling. `RefinementModel.copy_anchor_discrepancy` now **measures** that — the
largest distance between a variable's anchor and the same component at any
equivalent site — and it is reported, not repaired and not refused, because
imposing the reference site's value is the transcribed behaviour. Under the
frames used above it is 8.089e-09 / 7.497e-09, i.e. grid noise.

The frames themselves come from the reference case's `H2O.axes` — `H1  z global
Z x from H2 to H1`, `H2  z global Z x from H1 to H2` — which is an **input**
artifact, already committed verbatim as the `axes` field of
`tests/pytests/data_isapol/camcasp_cn_pot_h2o_l2h1.json`, and is rebuilt from
the molecule's own geometry rather than imported as numbers; nothing is read
from the reference `Cn` output, so plan §5 item 2's boundary holds. For this
water that declaration is `diag(-1,-1,1)` on the first hydrogen and the identity
elsewhere. Getting it right is worth a factor in fit quality, because the model
can then actually represent both hydrogens: against the identity-frame run the
data rms falls from 4.047e-04 to 3.593e-04 (near) and 8.749e-05 to 3.303e-05
(far), and the anchor distortion the fit needs falls from 2.892 (0.962 relative)
to 2.278 (0.539) and from 0.246 (0.354) to 0.029 (0.038).

`tests/pytests/test_isapol_nn_refinement.py` carries this at PBE0/cc-pVDZ, whose
default `cc-pVDZ-JKFIT` AUX is already MAIN-matched, for a fraction of the cost:
it asserts that the anchors are the accepted constrained-NN chain's own and are
distinguished by their digest, that both rows refine against one shared target
set (re-deriving the targets from the NN context and comparing bitwise), that
each refinement beats its *own* anchors in the packed-target rms, that `COPY`
equivalence and symmetry survive, the rank ordering above, and that a
`1e12` penalty coefficient holds the constrained-NN anchors to within
`copy_anchor_discrepancy`. The frame failure itself is certified without an SCF
by `test_isapol_refine.py::test_a_copy_equivalence_reports_how_far_its_sites_disagree`,
which mirrors one hydrogen's rank-1 anchors, checks the reported discrepancy is
exactly twice the flipped coupling, and shows a pinned fit missing the
equivalent site's anchors by precisely that much.

`tests/pytests/test_isapol_native_point_response.py` certifies the same wiring
at the cheap sto-3g fixture: refining a test-declared isotropic rank-1 model
against real `native_point_charge_response` targets solves at full rank, beats
its own anchors in the packed-target rms (which reproduces the solver's
`data_rms`), preserves `COPY` equivalence and symmetry, holds the parameters at
the anchors under a large penalty coefficient, and refuses a mislabelled origin,
representation or auxiliary-basis claim.

#### Refinement on the reference case's own 500-point lattice

`.pi/audit/avtz-end-to-end-cn.py` → `.pi/audit/avtz-end-to-end-cn.json` refines
all eleven Casimir nodes of both accepted aVTZ chains against native
point-to-point response on CamCASP's own `Random 500 / Seed 1 / LoLim 2.0 /
HiLim 4.0` cloud. That cloud is an **input** declaration read from CamCASP
source — `cluster_file_interface.F90::write_camcasp_1` line 2438 emits
`npts_grid = 500` for a `properties` run whose `.clt` declares `Options Tests`,
2000 otherwise — and is never inferred from the `Cn` output (plan §5 item 2).
It is reproduced **bitwise by three independent routes**: the in-tree oracle
`latticedump` that `oracle/make_lattice_oracle.sh` builds out of CamCASP's own
`generate_lattice`/`random.f90`, a standalone driver linked directly against the
built CamCASP objects, and Psi4's own `core.FitPoints`
(`libisapol/fit_points.cc`) — all three agreeing to
`sha256 693d2c092b36f85171da0acd08702e6c1fd24deb95c7b1146a0d83487a2c80aa`,
`dmax 12.235791546666666`, `centre (0, 0, −0.7477915466666666)`,
`ncandidates 1327`. Locked by
`test_isapol.py::test_fit_points_reference_case_cloud`; **no Python
transcription of the lattice enters the repo** — the transcription survives only
as one of the three certification routes. 500 < `MAXIMUM_POINTS = 512` is what
makes this comparison possible at all, and the 2000-point production lattice
stays **refused**, not accommodated. The targets remain this script's own
`NativeDirectActualPointResponse` point-charge quantities and are declared as
such; feeding them the reference's lattice does not turn them into the
reference's `SuppliedFittedPropagatorPointResponse` target.

Measured, per chain: 17 channels and 17 parameters at every node except
ω = 37.82 au, where one component falls below the 1e-4 anchor cutoff
(16 parameters, numerical rank 16); `IsaPfitStatus.Solved` at all eleven nodes;
data rms 2.94e-05 → 7.27e-09 and max |residual| 8.74e-04 → 4.18e-07 from the
static node to the last; anchor shift max|Δ| 0.36 at ω = 0 falling to 2.9e-04;
49.4–52.9 s per node, 575 s per chain, `copy_anchor_discrepancy` 5.07e-13 /
4.88e-11, `input-sum-rule` 6.299e-09 / 2.792e-07. All eleven nodes share **one**
geometry-only `channel_fields` matrix, because `channel_fields` depends on the
points, site origins, frames and `rank_limit` only and never on frequency — that
is what makes 22 refinements at 500 points practical rather than projected.

Static isotropic α per site moves from anchors 7.1241 / 1.3734 / 1.3734 to
refined 7.1824 / 1.3390 / 1.3390 (`direct_ov`), and 7.1002 / 1.3697 / 1.3697 →
7.1849 / 1.3374 / 1.3374 (`lambda=1000`). The two chains' **refined** models
agree with each other far more closely than their anchors do, because the fit is
driven by a point-to-point response the two chains nearly share.

Two driver additions were needed for the Casimir–Polder consumer, and both are
committed with tests. `isapol_refine.isotropic_scalars(result)` reduces a
`RefinementResult` to `α_l = trace(α_ll)/(2l+1)` per site for l = 1…limit — rank
0 is deliberately absent, so a nonzero charge-flow row cannot leak into a
dispersion coefficient — encoded once instead of once per caller
(`test_isapol_refine.py`, 2 tests, one of which recomputes every value straight
from `refined_tensors`). `isapol_lw.isotropic_dispersion` accepts
`site_ranks_a=` / `site_ranks_b=`, so a caller can **declare** the reference's
per-site L2/H1 limit instead of the hardcoded uniform `[1,2,3]`; a limited
site's dropped rank pairs are reported through `missing_rank_pairs` and
`complete` rather than silently completed, and rank 4 is **refused** there
rather than zero-filled, because `supplied_nonlocal_properties` localizes no
rank-4 tensor (`test_isapol_lw_driver.py`, 3 tests). A declared limit is a model
choice, not a post-hoc scaling of an unlimited result, and the tests pin the
limited coefficients against `core.isa_isotropic_dispersion` called directly on
the sliced scalars, plus the closed form
`C_n = Σ binom(2l_a+2l_b, 2l_a) α_{l_a} α_{l_b} Σ_f w_f g_f²`.

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

### End-to-end comparison against the reference localized `Cn` potential

`data_isapol/camcasp_cn_pot_h2o_l2h1.json` decodes the reference water `Cn.pot` in
all three shipped SCF back-end rows (`oracle/read_cn_pot.py`; printed output text
only, nothing compiled, no source read, reference tree never written). The
localization header is byte-identical across the rows (`Limit 2`, `WSM-Limit 2`,
`H-Limit 1`, `LW`, `Weight 3`/`0.001`, `SVD threshold 0.0`, `Pol Cutoff 1e-4`,
`NoRefine? False`) and so is `H2O.axes`, but the `.clt` inputs differ in **two**
declarations, `SCFcode` *and* `HOMO`: against the shared `I.P. 12.62063 eV` and
CamCASP's own **27.21136** eV/Eh divisor those are three different GRAC shifts
(0.13193 dalton / 0.06580 nwchem / **0.06490** psi4). The family spread is
therefore not back-end noise; only the psi4 row declares our shift, DALTON
declares twice it, and the defensible yardstick is the nwchem/psi4 pair, 1.9% on
molecular C6.

Printed isotropic orders are exactly the admissible `n = 2(l_a+l_b+1)`: O-O
{6,8,10}, H-O {6,8}, H-H {6}, with the odd orders printed and identically zero on
the isotropic row only. The dispersion side of our chain can now **declare** that
per-site limit — `isotropic_dispersion(site_ranks_a=, site_ranks_b=)` and
`RefinementSite.rank_limit`, §6 — but the **localization** cannot: LW's workspace
is uniform, so `native_properties` still rejects `{O:2, H:1}` with
`LW pipeline requires uniform explicit rank3 or rank4` before any compute
(`isapol_native.py:217`). We therefore localize at uniform rank 3 and truncate
afterwards. That is a **different model** from localizing under the restriction,
is labelled as such everywhere below, and must never be quoted as agreement.

The first comparison, kept only as the **superseded uniform-rank-3 baseline it
is**, ran at PBE0/aug-cc-pVTZ with the reference GRAC shift and the matched
Cartesian AUX (§6) against the psi4 row, per ordered site pair, with our side
admitting {6,8,10,12} on every pair:

| quantity | `direct_ov` | traced lambda1e3 NN | reference |
| --- | --- | --- | --- |
| molecular isotropic C6 | 46.89713 (+0.600%) | 46.76829 (+0.324%) | 46.61741 |
| O-O C6 / C8 / C10 | 25.60926 / 503.0882 / 10666.265 | 25.52363 / 508.2636 / 11406.429 | 19.27258 / 410.2453 / 4106.707 |
| H-O C6 / C8 | 4.51717 / 71.45023 | 4.50724 / 72.39140 | 5.338895 / 57.3419 |
| H-H C6 | 0.80479 | 0.80392 | 1.497312 |

The molecular isotropic C6 total is the one partition-invariant number in that
table, and both routes land inside the family's own nwchem/psi4 spread, the
constrained-NN route closer. Every site split was far outside it, consistently:
O-O too large and worsening with order (+159.7%/+177.8% at C10), H-H ~46% low.
That localized the remaining gap to the two stages that chain had **not** applied
— PFIT refinement and the rank-limited model — not to a tolerance, and no
tolerance against the reference is asserted anywhere.

**Both of those stages are now applied**, on the reference case's own lattice.
`.pi/audit/avtz-end-to-end-cn.py` → `.pi/audit/avtz-end-to-end-cn.json` runs
PBE0/aug-cc-pVTZ with the psi4 row's own fixed GRAC 0.06490004527520865 Eh →
distributed response (both `direct_ov` and the constrained-NN λ=1000 route) → LW
at all eleven Casimir nodes → PFIT refinement per node on the bitwise-reproduced
`Random 500 / Seed 1` cloud (§6) → `Cn` at the reference's own rank limit,
`O.ranks = [1,2]`, `H.ranks = [1]`. Each type pair is compared only over the
orders the reference itself prints, which under those limits is exactly
`n ≤ 2(l_a^max + l_b^max + 1)`. The kernel's `unrestricted_complete` flag is
**not** the comparison gate — it reports which rank pairs a rank-4 model would
have added, and the reference, being the restricted model, has none of them
either.

| quantity | `direct_ov` unrefined | `direct_ov` refined | `lambda1000` unrefined | `lambda1000` refined | reference (psi4 row) |
| --- | --- | --- | --- | --- | --- |
| molecular C6 (sum rule) | 46.8971 (**+0.600%**) | 46.8935 (**+0.592%**) | 46.7683 (**+0.324%**) | 46.8875 (**+0.579%**) | 46.61741 |
| O-O C6 | 25.6093 (+32.88%) | 26.1627 (+35.75%) | 25.5236 (+32.43%) | 26.1606 (+35.74%) | 19.27258 |
| O-O C8 | 503.088 (+22.63%) | 509.359 (+24.16%) | 508.264 (+23.89%) | 515.391 (+25.63%) | 410.2453 |
| O-O C10 | 4641.37 (+13.02%) | 4656.24 (+13.38%) | 4758.35 (+15.87%) | 4773.23 (+16.23%) | 4106.707 |
| H-O C6 | 4.51717 (−15.39%) | 4.42571 (−17.10%) | 4.50724 (−15.58%) | 4.42493 (−17.12%) | 5.338895 |
| H-O C8 | 44.0216 (−23.23%) | 42.7349 (−25.47%) | 44.4947 (−22.40%) | 43.2014 (−24.66%) | 57.3419 |
| H-H C6 | 0.804794 (−46.25%) | 0.757004 (−49.44%) | 0.803922 (−46.31%) | 0.756790 (−49.46%) | 1.497312 |

Two results follow, and they point in opposite directions. **(i) The rank limit
was most of the high-order discrepancy.** O-O C10 goes from +159.7% — where the
uniform-rank-3 kernel adds the (1,3)/(3,1) terms the reference never had — to
**+13.0%** once both sides run the same model, and H-O C8 changes sign,
+24.6% → −23.2%. Comparing at the reference's model instead of across models was
worth an order of magnitude on C10. The same run's uniform-rank-3 row reproduces
the superseded table above to every printed digit (25.60926 / 503.0882 /
10666.26 / 4.517173 / 71.45023 / 0.8047943), which is an independent consistency
check on the whole chain. **(ii) Refinement is not the missing piece, and that is
now measured rather than assumed.** It leaves the partition-invariant sum rule
alone (+0.600% → +0.592%) and moves every site-resolved row *slightly further*
from the reference. The fit converges properly — `Solved` at all eleven nodes,
data rms 2.9e-05, max |residual| 8.7e-04 over 125,250 packed targets — so this is
not a failed refinement. It is what a fit does when its point-to-point response
is nearly invariant to the O/H split: the split is set by the weight-3 penalty
anchors, and the anchors are what disagree. Both chains' refined models nearly
coincide (O-O C6 26.1627 vs 26.1606) although their anchors differ.

**The whole site-resolved gap is one scalar.** Both models are separable to ~1%
— `C6_ab ≈ K α_a α_b` with one common frequency shape per site, residual
`C6_HO²/(C6_HH·C6_OO) − 1` = −1.22% for the reference against −0.99%/−1.10% for
our unrefined/refined models — so the six site rows carry a single degree of
freedom, `ρ = α_H/α_O = sqrt(C6_HH/C6_OO)`: reference **0.278732**, ours
**0.177274** unrefined and **0.170102** refined, 0.636× and 0.610× the reference
(`lambda1000`: 0.177474 / 0.170084). Our chain gives hydrogen too little
polarizability and oxygen too much, by a factor ~1.6 in the ratio, while
conserving the total to 0.6%. The multiplicity convention behind the sum rule is
corroborated rather than assumed: all ordered pairs within a type are equal for
the isotropic `Cn` (spread 0 to 7.9e-08), so per-ordered-pair and summed readings
of the reference blocks differ by exactly 4×, and it is the +0.6% agreement under
`{O-O: 1, H-O: 4, H-H: 4}` that selects the per-ordered-pair reading.

**The rank-limited localization was the leading candidate for ρ. It is now
implemented, and it is ELIMINATED — by proof and by measurement, not bounded.**
`isa_localize_lw` takes a declared `rank_limit` in 1..3 (default 3, the
historical behaviour, reproduced bitwise), threaded through
`supplied_nonlocal_properties(localization_rank_limit=...)` and
`native_properties`. A declared `L` truncates the supplied blocks to the leading
`(L+1)²` real Racah components in both index slots — reporting the discarded
magnitude as `truncated_input_maxabs`, a report on the caller's declaration and
deliberately not a gate — and then runs the component-pair loop, the translated
transfer application, the molecular-sum conservation check and the local output
inside that space, with every algorithm-controlled residual on the *unchanged*
`1e-6` gate.

Two prior claims here were wrong and are corrected. First, the limit reaching
the localization is a **single uniform integer**, not the per-site
`WSM-Limit`/`H-Limit`: `cluster_file_interface.F90::write_orient_file_localize`
emits only `Limit all rank {LIMIT}`, `Localise {LOC} test 1e-7 Limit {LIMIT}` and
`Write/Print all local ranks 1 to {LIMIT}`; the per-site
`Limit rank to {HLIMIT} for sites +++` lines belong to
`write_process_file_for_pfit` and `write_process_file_for_casimir`, and
`{WSMLIMIT}` to the energy and display writers. `bin/localize.py` corroborates:
only `LIMIT` and `LOC` reach the ORIENT localization. Second, localizing under
the restriction is **not** a different least-squares problem from localizing at
rank 3 and reading the low components:

> For any declared `L`, the localized blocks equal the rank-3 localized blocks
> restricted to the leading `(L+1)²` components, bitwise.

The pair loop is ordered `first_component ≤ second_component`; a transfer for
`(t,u)` writes only into slot `u`, with target weight
`δ(target,t) + T(±d)[target][t]`, and multipole translation is rank-raising, so
that weight vanishes for `rank(target) < rank(t)`. A pair whose `t` lies above
the declared space therefore writes only above it, and `t ≤ u` puts `u` above it
too, while the screening decisions for the lower pairs read only lower
components. Measured at exactly **0.0** over the eleven recorded water
frequencies and over random reciprocal input on a 4-site path and a 6-site
branched graph, and at the `Cn` level: every coefficient that declared limits 3
and 2 both report *complete* (all nine site pairs at `n=6` and `n=8`, so rank 2
is exercised, not just rank 1) agrees to **0.000e+00**, and ρ itself is flat to
**0.000e+00** across declared limits 1, 2 and 3. A declared limit cannot change
any rank ≤ L observable, and ρ is one, so this candidate is closed by
elimination — the number is untouched. What the limit does buy is an honest,
cheaper model whose high components are absent *by declaration* rather than
dropped after the fact, and a limit a consumer can check: `isotropic_dispersion`
now refuses a declared rank above the model's own localization limit, and its
`site_ranks=None` default is read off the model rather than fixed at `(1,2,3)`,
so a limited model can never be scored `complete` on terms it never had.

**ρ therefore remains open, and the candidate list moves upstream of LW.** One
measurement relocates it: running our LW localization and `Cn` kernel on the
*reference's own recorded distributed* pair response (`H2O_NL4_000.pol`, the same
`wt4_L3` case, on the manifest's declared `CasimirGrid(10,0.5)` grid) gives
`α_H/α_O = 0.28216` and `sqrt(C6_HH/C6_OO) = 0.31294`, i.e. **1.12×** the
reference's 0.278732 rather than our native chain's 0.636×. That is not a parity
claim and is not quotable as one — it compares our *unrefined* localization of
the reference's distributed input against the reference's *refined* L3 value, and
the fixture's own README records that unrefined indices 7..10 fail its literal
seven-decimal comparison. It does say that neither the localization nor the
isotropic `Cn` kernel is where the factor ~1.6 in ρ is created: fed the
reference's partition, our chain reproduces the reference's partition ratio to
~12%. The surviving candidates for ρ are therefore all in the **distributed
response/partition step that produces those blocks** — the constrained-NN
partition and its site weights, the response basis, and the asymptotic-correction
form — and each needs its own gate. Candidate 2 (the response-step AC form,
+1.36% on the H2O molecular polarizability, §6) is absorbed nowhere and still
cannot explain ρ by itself, which is a partition ratio at fixed total.

The 377 nonzero recoupled reference rows (O-O 258, H-O 86, H-H 33) are an
explicitly **uncompared** track, because the native anisotropic product is
`orientation_resolved_scalars_not_recoupled_components`. Unlike the isotropic
row they do not vanish at odd orders (O-O 16/46/95/118/103 at n=6..10, H-O
23/33/53 at n=6..8, H-H 33 at n=6), so the fixture carries a census of them and
withholds their values by design. See
`tests/pytests/test_isapol_reference_dispersion.py` (4 quick + 3 long).

### Closed Casimir-Polder oracle on a second reference case

`data_isapol/camcasp_local_pol_h2o_atz_wt4.json` decodes CamCASP's shipped
`examples/properties/H2O` (`oracle/read_local_pol.py`; printed output text and the
case's own input declarations only, nothing compiled, no source read, reference
tree never written). It is **not** the `tests/H2O_props` case above and the two
are never conflated: weight type **4**, prefix `H2O_aTZ`, `Scf-code DALTON`,
`XC-func PBE0`/aug-cc-pVTZ, CKS propagator with `Hessians Internal` and `DF with
constraints`, `DF-TYPE-MONOMER NN`, and -- decisively for any local tensor --
bond axes, `H1  z from O to H1   x from H2 to H1`, against the other case's
`z global Z`. The `.cks` declares **two** DF/Polarizability stages, `Eta = 0.0`/
`Rank 2` for total polarizabilities and `Eta = 0.0005`, `Lambda = 1000`/`Rank 4`
for the distributed stage, so the fixture keeps the protocol as an ordered list
rather than a dict that would lose one of each.

What the case adds is a **closed loop**: it prints refined local polarizabilities
and, from those same numbers, dispersion coefficients. Reducing the printed local
tensors to per-rank isotropic scalars `alpha_bar_l = tr(alpha_ll)/(2l+1)` and
feeding them to our own `isa_isotropic_dispersion` on the declared grid
reproduces every reference row it prints, worst **2.03e-7** relative:

| pair | order | ours | reference | relative |
| --- | --- | --- | --- | --- |
| O-O | C6 / C8 / C10 | 21.594626 / 422.27894 / 3894.9955 | 21.59463 / 422.2789 / 3894.995 | 2.03e-7 / 1.03e-7 / 1.28e-7 |
| H-O | C6 / C8 | 4.590924 / 44.659514 | 4.590924 / 44.65951 | 9.23e-9 / 9.52e-8 |
| H-H | C6 | 0.98298589 | 0.9829859 | 9.90e-9 |

Both sides of that comparison are the reference's, so what is under test is ours
alone -- CP weights, the `(-1)^l sqrt(2l+1)` recoupling convention behind the
isotropic reduction, the `n = 2(l_a+l_b+1)` bookkeeping and the pair assembly.
It holds to the reference's printed precision and no SCF runs.

That recoupling convention is itself measured rather than assumed, because the
same `_casimir.out` prints its own `00(l l)` recoupled rows. Rebuilding them from
the `.pdef` variables reproduces all 40 printed entries (O ranks 1 and 2, rank 1
on each hydrogen, 10 dynamic nodes each) with worst relative **3.14e-6**, and the
sign is part of the test: oxygen's `00(11)` is negative where `alpha_bar_1` is
positive, which is the entire content of the `(-1)^l`. Those rows carry the
dynamic nodes only -- `Quad 10` prints 10 columns, not 11 -- so the static point
is not covered by that check; it is covered instead by the exact model
reconstruction and the molecular-polarizability results below. Only the isotropic
`00(l l)` diagonal is extracted; the anisotropic recoupled components stay the
uncompared track they are for the other case.

The declared quadrature is `Quad 10`, `Beta 0.5`, which the reference itself
names `f11` in its pol-file: **11** nodes. `core.CasimirGrid(10, 0.5)` is exactly
that grid -- `n_freq()` is the Gauss-Legendre *order* and the object carries
`n_freq + 1` frequencies with index 0 the static point
(`casimir_grid.h:120` and the `n_freq` parameter doc above it), the dynamic nodes
coming in reciprocal pairs `omega(k)*omega(n_freq-k+1) = omega0^2`. There is no
coverage gap against this reference grid; `kMaxCasimirFrequencies = 10` bounds the
order, not the node count -- its name and the `n_freq` parameter comment both
said "frequencies" and are corrected to say so, because that wording is what
misread the grid as capped at static + 9 in the first place.

Model, read from the input side. `H2O_aTZ.pdef` declares **17** free numbers per
frequency: 13 on O (rank 2), 4 on H1 (rank 1), and `H2  H2  COPY  H1  H1`. The
fixture carries those 17 variables per frequency rather than 2673 tensor
elements, and the decoder asserts the rebuild reproduces the printed 9x9 blocks
**exactly** (`reconstruction_error == 0`), which is simultaneously a lossless
representation and a check that the declared model is the model the printed
tensors were fitted under. The rank limits appear in the reference's own output as
exact zeros -- the `00` row and column of every block, and everything above rank 1
on hydrogen -- matching the `anchor[1:,1:]` construction of the refinement driver.

The frame requirement of a `COPY` is independently confirmed by the reference's
own output: the `H1 H1` and `H2 H2` printed blocks are **bit-identical** in local
axes, while their globalized dipole blocks are related by the molecular plane's
x mirror and differ by `2|alpha_xz| = 0.46794962`. A refinement pinning the two
hydrogens on *global* anchors is left with exactly that as an irreducible
residual, which is what `RefinementModel.copy_anchor_discrepancy` measures.

Rank-4 reference quantity. `H2O_aTZ_NL4_static.pol` is a 75x75 = 3 sites x 25
components static distributed polarizability at full double precision, the only
rank-4 reference quantity available. Its symmetry defect is 5.81e-11 and
`sum_ab alpha^ab_{00,00} = -3.11e-12`, but its charge-flow sum rules
`sum_a alpha^ab_{00,u}` and `sum_b alpha^ab_{t,00}` close only to **7.68e-7** --
a direct measurement of the reference's own `Eta = 0.0005, Lambda = 1000`
constrained-NN fit quality, and the level any comparison against it is limited
to. Translating it to the molecular polarizability,
`alpha_ij = sum_ab [alpha^ab_{t_i t_j} + r_i(a) alpha^ab_{00,t_j} +
r_j(b) alpha^ab_{t_i,00} + r_i(a) r_j(b) alpha^ab_{00,00}]`, the relative sign is
**measured** rather than assumed: `+` leaves the C2v-forbidden xz element at
7.03e-8 and `-` leaves 2.07e-6, so `+` is the convention. It gives
diag(9.848796, 8.633546, 9.259731), isotropic **9.247357**.

The reference pipeline is internally inconsistent at 0.26% and that bounds
everything. The molecular polarizability is partition-invariant, so translating
the distributed tensor and rotating-and-summing the refined local tensors by the
declared axes must agree. They do not: the refined route gives
diag(9.916906, 8.632315, 9.265532), isotropic **9.271584**, i.e. **0.024227
(0.26%)** away from 9.247357. That is the reference refinement's own distortion,
the same order as the 0.024 O-site anchor movement our constrained-NN refinement
makes, and it is compounded by the refinement lattice being 500 `Random`/`Seed 1`
points between `LoLim 2.0` and `HiLim 4.0` -- not reproducible without CamCASP's
RNG, so its refined tensors cannot be matched exactly by construction. Reference
`alpha_iso(i*omega)` from the refined local model, indices 0..10: 9.271584,
9.270256, 9.232090, 9.008918, 8.299214, 6.804279, 4.616113, 2.381110, 0.834219,
0.154917, 0.006117.

Our own rank-4 chain at the matched protocol. `SiteRecipe.rank` is an `int` in
`range(5)`, so rank 4 is reached by replacing every site's rank on the generated
recipe; `o.generated_recipe` hardcodes 3. At PBE0/aug-cc-pVTZ, the reference GRAC
shift 0.06490004527520865 Eh, IsaGrid(99,590), `aug-cc-pVTZ-JKFIT`,
`shared_sweep`, static frequency, identity frames (E = -76.37966827740804,
HOMO = -0.3989569916800326 -- matching the other case's declared `HOMO -0.3989`
to every printed digit -- nbf 92), LW accepts rank 4 with no failures:

| row | molecular alpha diag | isotropic | charge sum rule |
| --- | --- | --- | --- |
| `direct_ov` | 10.389483 / 9.414895 / 9.808506 | 9.870961 (+6.75%) | 1.05e-8 |
| traced lambda1e3 NN | 10.395560 / 9.314226 / 9.809239 | 9.839675 (+6.41%) | 1.98e-6 |
| reference (distributed) | 9.848796 / 8.633546 / 9.259731 | 9.247357 | 7.68e-7 |

Because the translated molecular polarizability is partition-invariant, that
6.4-6.8% excess is located in the **response step**, not in the partition and not
in the refinement. Note also that the rows above are run at the *other* case's
declared GRAC shift 0.06490004527520865 and `HOMO -0.3989`, which belong to
`tests/H2O_props`; the reference row's own case (`examples/properties/H2O`,
weight type 4) converges to `E_HOMO = -0.330596560344` with DALTON's printed
`v_xc(inf) = 0.133203439656`. The two must not be quoted as one protocol.

**Response step: closed on identical orbitals.** The excess is now assigned. The
obstacle to assigning it was that no shipped case both exports usable MO vectors
and prints a polarizability at admissible size, so the reference propagator was
rebuilt from CamCASP's own MIT sources (`NAME=camcasp`, which alone pulls
`prop_utilities.F90 densfit_prop.F90`) and certified by reproducing the shipped
`examples/energies/He2/aTZ_MC/check/OUT/He2.out` **digit for digit on all 80
numeric lines** (`E^{2}_{disp} = -13.108210 CM-1`, `E^{2}_{ind} = -0.26851267E-02
CM-1`, ...). Build provenance: CamCASP 6.0.051, gfortran 14.3.0, serial `make`
(the makefile is not parallel-safe -- separate `%.o`/`%.mod` rules compile the
same source twice), `-fallow-argument-mismatch` prepended to `FFLAGS/FFLAGS2/FFLAGS3`
to demote legacy-F77 rank mismatches in `gamint.F` to warnings. That flag is a
compiler diagnostic setting, not a scientific tolerance, and the digit-for-digit
certification above is what licenses it.

Both codes were then run on the **same** orbitals: the shipped 15-digit DALTON
set `examples/energies/He2/aTZ_MC/check/He2-A-asc.movecs` (23x23, PBE0), decoded
into Psi4 at `max |C^T S_psi C - I| = 1.132e-14` using `PMAP = [2,0,1]` and
`DMAP = [2,3,1,4,0]` from `basis_trans_mats.F90::init_basis_trans_mats(option=2)`.
Decoding requires a `.gbs` reproducing DALTON's **6-primitive** first-S aug-cc-pVTZ
contraction in DALTON's shell order: Psi4's shipped He block spans the same space
(SCF energies agree to 1e-10) with a 4-primitive contraction, so its MO
coefficients are not interchangeable. CamCASP's own `basis/gamess_us/aug-cc-pVTZ/He`
is the 6-primitive form. This removes the asymptotic-correction form from the
comparison entirely, and is an **external-orbital comparison track** -- it must be
labelled as such wherever quoted. No gate is relaxed to run it:
`validate_correction` calls `require_scf_seal` only on the `FIXED_GRAC` branch,
and the `NONE` branch's assertion that the wavefunction carries no asymptotic
correction is factually true here (the Psi4 functional object is plain PBE0; the
AC lives entirely in the injected orbitals).

At the reference's declared protocol (`Type CKS`, `Hessians Internal`, `DF with
constraints`, `DF-TYPE-MONOMER NN`, `Eta = 0.0`, `Lambda = 1000.0`, `Quad 10`,
`Beta 0.5`, AUX Cartesian `aug-cc-pVTZ-RI`), with CamCASP's own gates passing
(orthonormality max difference `.00000`, `\int \rho(r) dr = 2.00000`):

The reference's grid is refined row by row; our column is the single
grid-invariant value our propagator returns over its own independent IsaGrid
ladder (`(60,110)`, `(75,302)`, `(99,302)`, `(99,590)`, `(150,590)`, `(200,974)`
= 6490 to 193826 points), and the rows are *not* pairwise grid-matched:

| CamCASP response grid | atom points | CamCASP isotropic | ours (fitted lambda=1000) |
| --- | --- | --- | --- |
| `Angular 100 / Radial 60` (shipped) | 6490 | 1.416255 | 1.41637208 |
| `Angular 302 / Radial 120` | 35938 | 1.416323 | 1.41637208 |
| `Angular 590 / Radial 200` | 117410 | 1.416370 | 1.41637208 |
| `Angular 974 / Radial 300` | 291226 | 1.416358 | 1.41637208 |
| `Angular 1454 / Radial 400` | 580146 | 1.416352 | 1.41637208 |

The shipped-grid difference is +1.171e-04 absolute / +8.27e-05 relative. It is
**the reference's own quadrature error**: refining the reference's grid moves its
answer 9.7e-05 toward ours, after which its last three rows oscillate within
+/-9e-06 of 1.41636. Our value is grid-converged -- identical to nine digits from
6490 to 193826 points, and the falsification test confirms the grid is live, not
a dead parameter (IsaGrid(10,6)/54 points gives 1.42171160, (15,26)/364 gives
1.41538714, (30,50)/1450 gives 1.41636887). Against the reference's converged
plateau the residual is **+2.0e-05 absolute / +1.4e-05 relative**, i.e. the size
of the reference's own remaining grid noise, and ~4x the E13.7 print resolution
(+/-5e-08). Both anisotropies are numerically zero (CamCASP `0.5669751E-14`).

The residual is not the DF penalty: lambda in {1e2, 1e3, 1e4, 1e5, 1e6, 1e8} all
give exactly 1.41637208, so the quadratic-penalty-versus-exact-Lagrange-constraint
difference is not a source. `direct_ov` on the same orbitals gives 1.41129611
(-3.50e-03 relative), so the reference's constrained-NN DF space is the correct
comparison space, as declared.

Because this is a same-input comparison it bounds candidates 1, 3, 4, 5, 6, 7 and
the previously unmeasured 8 **in aggregate** at 1.4e-05 -- a stronger statement
than eliminating them individually, and in particular the first measurement of
candidate 8 (CamCASP builds `Ker_kk' = \int chi_k dv_xc/drho chi_k' dr` between
*auxiliary* functions and forms `KerOVOV = Dov_c Ker Dov_c^T`,
`prop_utilities.F90:329-441`, so its kernel never sees exact orbital products).
What it does **not** bound is candidate 2, the asymptotic-correction form, which
by elimination carries the whole +1.36% (H2O) / +1.67% (He) gap seen with Psi4
GRAC orbitals. That gap is channel-resolved for He: alpha for a 1s^2 atom is
carried entirely by the 1s->np channel, and at the declared shift our p-channel
excitation is 0.58% low while the (irrelevant) lowest-s excitation is
over-corrected by +0.146 au. The three reference AC forms are distinct and must
not be interchanged -- `He_aTZ.dal` plain `.DFTAC 0.9036 3.0 4.0`; `H2O_aTZ_A.dal`
`.DFTAC / MULTPOLE / TANH / 0.46380 0.46380 3.0 4.0`; `He2_A.dal`
`.DFTAC / MULTPOLE / TANH / VARSHIFT / 0.90360 0.90360 3.5 4.7` -- against Psi4's
LB94-based GRAC with alpha=0.5, beta=40. DALTON's relation is
`v_xc(inf) = IP_declared - |E_HOMO|`, verified at every GRAC iteration.

Two operational constraints on reproducing this. CamCASP's propagator reads the
`.cks` on **stdin** (`bin/camcasp.py:1760`), and `src/precision.f90` sets
`lchar = 80` while `free_format_reader.F90`'s `reada` truncates at 64 characters,
so a long scratch path must be reached through a short symlink driven by the
`CAMCASP` environment variable (`initialize.F90:22` uses
`get_environment_variable`, bypassing `reada`). `src/tests` is on `Makefile_body`'s
vpath, so `src/tests/test_routines.F90` is required to link.

Artifacts: `$SP/he2_response.py`, `$SP/he2_gridscan.py`, `$SP/he_common.py`,
`$SP/dalton_avtz_he.gbs`, `$SP/camcasp_run/he_same_orb.cks`. The Formamide and
Benzene same-orbital routes are fully decoded (`max |C^T S_psi C - I| = 7.805e-13`
for Formamide) but **blocked** by the declared hard caps (nov 1044 > 512,
ao_work ~1.0e11 > 6.4e10); they are not quoted as agreement and the caps are not
raised. The `examples/energy-scan/water2-B` cross-check is abandoned as
precision-limited (~8 significant digits; signature assignment cost 8.728e-01).

Coverage and spread. Under `L2` with rank 1 on hydrogen the admissible orders are
O-O {6,8,10}, H-O {6,8}, H-H {6}, so the only **complete** molecular isotropic
total is `n = 6`; C8 and C10 summed over pairs are structurally partial and are
labelled as such in the fixture (`molecular_isotropic_complete_orders == [6]`).
That total is **43.890270** here against **46.617408** for the other case's psi4
row -- the same property from the same code, 2.727138 apart across declaration
choices, 6.21% of this case's total and 5.85% of the other's. The reference
family's own spread, not ours, is therefore the honest bound on what agreement
with "the" reference number can mean. See `tests/pytests/test_isapol_camcasp_local_pol_oracle.py`
(9 tests, no SCF).

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
is rejected at the recorded lambda1** by the strict production LW gate —
charge-sum ≈4.1e−4 at every frequency — so the fitted-OV intermediate cannot be
anchored there and is anchored on He instead. Declaring a converged charge
penalty (lambda≥1e3, §6) does make strict LW accept a water fitted chain, but
that is a differently declared model and cannot be quoted against the lambda1
numbers in this table. He is monatomic, so `w_a/Σw ≡ 1` and the two partition
stages are *exactly* degenerate on it (A = 0); their "satisfied" rows on He are
an artifact of that degeneracy and are labelled as structurally insensitive,
not offered as evidence. The rejection was not worked around: relaxing the
residual policy to admit it is forbidden and was not done; the accepting
lambda converges the constraint in the producer instead, and leaves every gate
at production strength.

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
