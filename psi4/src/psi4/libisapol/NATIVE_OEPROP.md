# Native wavefunction-first atomic properties

**Fixed-GRAC opt-in:** the default demonstration below remains no-GRAC. An
explicit `ATOMIC_SCF_ASYMPTOTIC_CORRECTION=FIXED_GRAC` plus positive, exactly
matching `ATOMIC_SCF_EXPECTED_GRAC_SHIFT` now admits a supported already-converged
fixed-GRAC PBE0 state. See NATIVE_FIXED_GRAC.md for actual-component validation,
SCF sealing and immutable correction provenance. This never configures/runs SCF
or supplies GRAC response derivatives. The real public strict-LW endpoint passed;
it is not matched aVTZ/PFIT protocol parity.

`psi4.oeprop(wfn, 'ATOMIC_PARTITION', 'ATOMIC_POLARIZABILITIES',
'ATOMIC_DISPERSION')` returns **None**, as before. Access the owned result with
`psi4.atomic_property_result(wfn)`. Each request replaces the wavefunction's
latest result, without mutating results previously returned to the caller.
The attachment is invalidated on native request entry, including rejected
requests. Unknown `ATOMIC_*` names fail before either ordinary or native
computation, including mixed requests. Ordinary non-atomic properties retain
their existing dispatch behavior.
Core getters return copies; some core records are caller-mutable snapshots.

The tasks are independently requestable:

- `ATOMIC_PARTITION`: Drho-C / ordinary ISA-A, final shapes and AUX multipoles;
  no response, localization, frequency grid, dispersion or PFIT.
- `ATOMIC_POLARIZABILITIES`: partition plus static native response and strict LW;
  no dispersion quadrature or PFIT.
- `ATOMIC_DISPERSION`: partition plus static and ten imaginary-frequency responses,
  strict LW and isotropic self-pair C6–C12. C6/C8/C10 are complete **within the
  rank-3 local isotropic expansion**; C12 lacks (1,4)/(4,1). This is not a
  recoupled CamCASP anisotropic coefficient table. The expert native API also
  supports an explicitly supplied different partner.

Use a fresh converged restricted C1 PBE0 wavefunction, e.g. `energy('pbe0',
return_wfn=True)` with basis `cc-pvdz`. The property call does not run SCF.
The ordinary adapter requires wavefunction-local evidence from a successful
internal SCF stopping test: iteration, actual energy change, orbital-gradient
norm, the effective energy/density thresholds, and RMS-versus-max norm policy.
Evidence is attached after finalization only for restricted C1 wavefunctions;
nonfatal MAXITER exits, external optimizer exits and fixed-count post-screening
exits do not receive this evidence. Arbitrary external/deserialized wavefunctions
are not assigned inferred convergence. A snapshot digest of the finalized
energy, occupations, geometry, functional metadata, overlap/Hamiltonian,
alpha/beta densities, Focks, orbitals and orbital energies (plus basis identity)
rejects stale state. Reinitializing/reiterating clears prior evidence.

Finite nonzero energy and maxabs(FDS-SDF) <= 2e-7 remain additional consistency
checks, **not convergence proof**: diagonalization can form a commuting density
before an unsuccessful stopping test. Underlying providers also validate density,
occupations and orbital overlap. None of these proves a global SCF minimum.

Response requires the effective unmodified full-LibXC PBE0 definition, not just
its name: 25% exact exchange, canonical component identities and internal/outer
scales, and matching range-separation, correlation, VV10 and GRAC parameters.
An independent unmodified LibXC factory supplies the comparison without reading
or mutating ambient DFT options. Custom/name-spoofed definitions, DFT_ALPHA
overrides changing the fraction, LibXC PBEH mixing tweaks and attached empirical
dispersion are rejected. Equivalent custom decompositions are conservatively
unsupported. Density-screening thresholds are not functional identity checks.

## Explicit policies, not implicit CamCASP aliases

- `PARTITION_SCHEME=ISA_A` controls **density partitioning**. MBIS is recognized
  but rejected here until a continuous-weight adapter is validated.
- `ATOMIC_RESPONSE_LOCALIZATION=LW` controls **distributed tensor localization**.
  LS is rejected. This option does not select an orbital localization method.
- `ATOMIC_PROPERTY_RECIPE=GENERATED_JKFIT_ISA_A` is a self-contained **H/O
  demonstration recipe**, not the modern CamCASP spherical AUX/AtomAux preset.
  Molecular AUX comes from a *declared* shipped Cartesian JKFIT set, named by
  `ATOMIC_PROPERTY_AUXILIARY_BASIS` (default `cc-pVDZ-JKFIT`, deliberately not
  MAIN-matched) and never inferred from `BASIS` or `DF_BASIS_SCF`; that same AUX
  also carries the Drho-C/ISA-A density fit, so naming it selects a partition and
  a different name is a different declared model, not a tuned one. AtomAux and
  Shape are separate even-tempered radial s roles. Their normalized primitive
  exponents are .1*2^k, k=0..16 for O and .2*2^k, k=0..10 for H. Effective
  coefficients, centres, all grids, and controller options are in `result.partition.recipe`.
- The ordinary response policy is explicitly native direct-OV Slater/PW92 ALDA
  with exact_exchange=.25 and local_scale=.75 (C++ multiplies the local primitive
  by 4). **By default no GRAC/asymptotic correction; no reference-kernel parity or PFIT.**
The explicit fixed-GRAC opt-in only changes admissible SCF input; ALDA remains
a separately declared response model, not the derivative of the GRAC correction.
  This does not infer or reproduce a CamCASP protocol from the SCF method name.
- ISA uses dedicated runtime IsaGrid quadrature, default 160 radial / 590 angular,
  configurable with `ATOMIC_PROPERTY_RADIAL_POINTS` and
  `ATOMIC_PROPERTY_SPHERICAL_POINTS`. ALDA response has its own runtime IsaGrid,
  default 99 radial / 590 angular, selected with `ATOMIC_RESPONSE_RADIAL_POINTS`
  and `ATOMIC_RESPONSE_SPHERICAL_POINTS`. SCF grid options are separate from both.
  The denser ISA default exceeds the current ALDA work bound for cc-pVDZ water;
  the response grid is a separately recorded model policy, not an increased bound.
- The automatic H/O bond graph uses a distance less than 1.3 times the sum of
  covalent radii (H .31, O .66 angstrom), global frames. Other elements and
  unsupported native resource/state policies fail explicitly.

## Direct OV versus fitted transitions

The density partition remains the native lambda1000 **Drho-C** fit, never AO
sampled density or Doo-C. Existing ISA-A controller, weighting, soft Positive-W
and Func-1/Fit-3 tail policies remain unchanged. No charge rescaling is applied.

The new explicit `response_basis='direct_ov'` option integrates the *actual*
occupied–virtual orbital products against those converged partition weights,
using the C++ Gaussian and real Racah harmonic kernels. Columns are occupied-fast
`a*nocc+i`. The shared full-OV solver acts in identity OV coordinates. This avoids
a transition-density fit, rather than repairing a fitted transition's charge.
Raw quadrature charge errors and reciprocity are measured; no row is overwritten
and no tensor is symmetrized. The default expert `fitted_auxiliary` route remains
unchanged and retains finite lambda1 fit diagnostics/failures. Its charge
penalty is now a caller declaration, `ov_charge_penalty` (default 1.0;
`direct_ov` accepts only the default), which converges that defect in the
producer rather than repairing a fitted charge after the fact: the residual
fitted charge falls as 1/lambda, and strict production LW accepts the water
fitted chain at every declared lambda >= 1e3 — including lambda1000, the traced
constrained-NN route's own penalty (`input-sum-rule` 4.65e-7), and lambda1e4,
where that residual reaches the quadrature floor `direct_ov` itself reports
(2.33e-8 vs 2.28e-8). Any lambda other than the recorded one is a differently
declared model; SPEC §6 has the measured table and the binding caveat.

That lambda acceptance is **basis-dependent, and the dependence is in the
declared AUX rather than in the penalty**. At PBE0/aug-cc-pVTZ with the reference
GRAC shift the default cc-pVDZ-JKFIT AUX is far smaller than MAIN, and there the
traced lambda1000 chain supplies `input-sum-rule` 6.65e-6 against the same strict
1e-6 gate, rejected at 9 of 11 Casimir nodes; the MAIN-matched aug-cc-pVTZ-JKFIT
AUX reports 2.79e-7 at that same traced lambda and passes every node. Raising
lambda a decade does *not* close the gap it leaves against the fit-free route --
the pair-tensor defect is identical to five digits at 1e3 and 1e4 (0.687 for the
default AUX, 0.0791 for the matched one) -- so the fix is the declared AUX, never
a tolerance, a grid or a penalty. The two AUX choices are two different declared
models with different partitions and different dispersion: molecular isotropic C6
is partition-invariant (46.8971254018 for `direct_ov` under both), while C8/C10
totals are not. `tests/pytests/test_isapol_matched_auxiliary.py` measures all of
this under the untouched production policy.

Prior parent water attempts 1/2 failed strict LW even after grid refinement:
the fitted OV analytic charge defect remained 6.0372e-5; the second grid's AUX
charge integration error fell to 7.3221e-7. Attempt 3 has only a recipe, not a
completed JSON/NPZ endpoint. They are not accepted molecular results.

LW remains **production 1e-6**, without reported-input or historical waivers.
Failures raise from ordinary oeprop, but completed stages and raw failures remain
inspectable through the accessor. Partial LW success never becomes a dispersion
model. `result.atomic_scalars` has axes (frequency, site, rank1..3), with scalar
`trace(alpha_ll)/(2*l+1)` in atomic units. Full tensors, axes, units, frequencies,
raw charge diagnostics and provenance are in `result.properties`.

This integration uses existing permitted native stages. No new CamCASP source was
transcoded here. Existing CamCASP-derived kernels retain their MIT notices and
Misquitta/Stone attribution; see SPEC sections 4 and 8 for source anchors. No
ORIENT source, static CamCASP descriptors, archived electronic states or tensors
are used by the example or the runtime adapter.

Measured fresh-run status and exact commands are recorded in
`.pi/audit/native-oeprop-water-handoff.md`; neither software tests nor this recipe
establishes modern CamCASP parity or basis-limit physical accuracy.
