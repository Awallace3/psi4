# Native CamCASP-Style Atomic Polarizability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a native Psi4 C++/Python atomic-polarizability pipeline that clean-room reproduces the reviewed CamCASP H2O L3 model, without calling or copying CamCASP, ORIENT, PFIT, or CASIMIR in production.

**Architecture:** Build the pipeline around explicit native intermediate representations: frequency-dependent site-pair spherical response, LW-localized site tensors, PFIT-style constrained local models, and real-spherical dispersion recoupling. Each stage has conservation/reconstruction tests before it is connected to an SCF response provider. CamCASP remains an ignored development oracle for fixed literals only.

**Tech Stack:** Psi4 C++17/libmints/libscf, Python pytest, Eigen/BLAS/LAPACK facilities already used by Psi4, generated local reference JSON only during literal extraction.

## Global Constraints

- Production code and pytest must not invoke, clone, access, or read CamCASP, ORIENT, PFIT, CASIMIR, or `.camcasp-reference/`.
- Do not copy ORIENT GPLv3 source, comments, structure, or control flow; implement from published equations and independently written specifications.
- Preserve atom order `O, H1, H2`; global Cartesian packed order is `xx, xy, xz, yy, yz, zz`.
- Dynamic outputs contain static plus ten increasing imaginary frequencies with frequency-major atom blocks.
- Public variables are `ATOMIC POLARIZABILITIES`, `ATOMIC DYNAMIC POLARIZABILITIES`, `ATOMIC POLARIZABILITY FREQUENCIES`, and `ATOMIC C6`, `ATOMIC C8`, `ATOMIC C10`, `ATOMIC C12`.
- Polarizability tensor/Cn comparisons use `rtol=1e-4, atol=1e-5`; frequency comparisons use `rtol=1e-10, atol=1e-12`.
- Use strict TDD: each behavior must fail before its implementation is written; production pytest uses checked-in literals, never generated JSON.
- Treat C8/C10/C12 as reviewed CamCASP L3-model parity, not a claim of rank-complete physical dispersion.
- Preserve full L3 tensors even when a reviewed PFIT model reports non-positive-definite higher-rank blocks; expose a diagnostic rather than silently altering the model.

---

## File structure

- `psi4/src/psi4/libmints/atomic_polarizability.{h,cc}`: native intermediate types, frequency grid, spherical/cartesian maps, LW localization, fitting, dispersion, and wavefunction-variable publication.
- `psi4/src/psi4/libmints/oeprop.{h,cc}`: OEProp entry point only.
- `psi4/src/psi4/libmints/CMakeLists.txt`: build registration.
- `psi4/src/read_options.cc`: opt-in native-pipeline options and fail-closed prerequisites.
- `psi4/driver/p4util/python_helpers.py`: OEProp method allowlist.
- `tests/pytests/test_atomic_polarizabilities.py`: fixed H2O parity tests with hard-coded literals only.
- `tests/pytests/test_atomic_polarizability_math.py`: cheap native-math fixtures/invariants independent of SCF and external tools.

## Implementation status (2026-08-17)

Checkbox state in the task sections below was never maintained during implementation. This
section is the authoritative record; it is evidence-backed and supersedes the checkboxes.

| Task | State | Evidence |
| ---- | ----- | -------- |
| 1. Plumbing | **done** | `valid_methods` accepts `ATOMIC_POLARIZABILITIES` (`python_helpers.py:744`); OEProp dispatch at `oeprop.cc:805`; calculator skeleton fails closed. |
| 2. Frequency/tensor algebra | **done, oracle-verified** | Grid matches reviewed frequencies to `7.1e-15`; `(10,11c,11s) -> (z,x,y)` and `R alpha R^T` reproduce the reviewed tensors exactly (`0.0`). |
| 3. LW localization | **done** | Conserves the molecular sum on real SCF data to `4.5e-8` (dipole) / `5.7e-7` (full L3) at all eleven frequencies. |
| 4. ISA-Pol response | **done, magnitude verified** | Summing the site-pair blocks with rank 0 translated to a common origin gives `(9.9106, 8.3580, 8.9598)`, isotropic `9.0761`, i.e. `0.970` of Psi4's own molecular `DIPOLE POLARIZABILITY` at the identical functional/basis/grid. The residual 3% is the deliberate `25% CHF + 75% ALDA` kernel difference. |
| 5. WSM refinement | **conserves; anchor scope corrected** | Was losing 36% of the molecular polarizability by fitting inside the charge density; fixed 2026-08-17, now conserves to `1.5e-2` of the response it is derived from. The rank-1 anchor scope was also corrected the same day (the reviewed protocol penalizes the whole dipole block, not just its diagonal), restoring a positive-definite published hydrogen dipole block. |
| 6. Dispersion recoupling | **done, oracle-verified** | All four coefficients within `2.5e-7` relative of the reviewed CASIMIR values; see below. |
| 7. End-to-end publication | **done** | All seven variables publish from one `OEProp` call on the SCF triple; verified end to end on PBE0/aug-cc-pVDZ. See the Task 7 record below. |
| 8. Oracle acceptance | **partial** | Frequencies and C6–C12 accepted in isolation. The full aug-cc-pVTZ/GRAC protocol has never been run; the six reviewed-literal comparisons exist but are skipped by default. |

Test suite for this feature: **381 passing** under `-m mints`
(`--ignore=tests/pytests/test_camcasp_reference.py`, which is pre-existing-uncollectable),
0 failing, 6 skipped (the reviewed-literal parity comparisons).

### Task 7 record

The chain is `FrozenResponseContext::create` -> `compute_isa_weights` ->
`ISAPolResponseProvider::compute_isapol_response` -> `derive_bond_graph` -> `localize_lw`
-> `generate_wsm_fit_points` + `evaluate_point_response` -> `derive_pdef_constraints` ->
`refine_wsm` -> `compute_dispersion` -> pack -> publish, with a gate between every pair of
stages. `AtomicPolarizabilityCalculator::run()` either returns all seven arrays or throws
`AtomicPolarizabilityPrerequisiteError`; `compute()` publishes only after `run()` returns.
The driver entry point is
`psi4.driver.procrouting.atomic_polarizability.atomic_polarizabilities`.

Measured facts that constrain any future change:

- **Memory.** The default 329-point fit grid needs a WSM peak of `304,876,088` bytes, and
  the stage gate reserves half of configured memory, so it requires at least ~0.58 GiB
  configured. Psi4's 500 MB default fails closed. The driver sets 4 GiB explicitly
  (`PIPELINE_MEMORY_BYTES`) and restores the previous value afterwards. (Before the fit
  shells were moved outside the charge density the default grid was 407 points and needed
  `454,828,904` bytes.)
- **Grid quality is basis dependent.** With aug-cc-pVDZ the LW charge-sum residual sticks
  at `1.2e-05` on a `302/50` DFT grid regardless of ISA density (tested to `150/24/32`);
  only `590/99` brings it inside `1e-6`. The DFT grid, not the ISA grid, is binding. The
  wiring spec's grid table was measured without diffuse functions and does not transfer.
- **PDef mask under declared C1.** For the reviewed geometry with `symmetry c1`, `no_com`,
  `no_reorient`, `derive_pdef_constraints` reports `C2v(Z)` with site groups
  `C2v(Z)/Cs(Y)/Cs(Y)`, `38/66/66` active pairs, 170 active, 66 equality rows, and 104
  independent variables, exactly as the PDef spec predicts. This is now pinned by a test.
- **Frame.** `derive_pdef_constraints` is called with empty `site_axes` and the packing
  rotation is the identity, because `refine_wsm`'s harmonics are molecular-frame.

Known non-parity residual at aug-cc-pVDZ, left for Task 8: the site-by-site distribution of
the localized response differs from the reviewed model, so `alpha_yy` and `alpha_zz` are
mis-split between O and H even though the total conserves to `0.955`. See remaining work
item 1. The hydrogen `xz` drift reported here earlier (`+4.29`) was a separate defect in the
anchor scope and is fixed. Hydrogen C10/C12 were negative at this basis before the
conservation fix and are now positive (`47.3`, `269`).

### Task 6 acceptance record

`compute_dispersion` was fed the reviewed L3 models directly, isolating it from the
response/localization/refinement stages, and compared against the reviewed CASIMIR
coefficients under the plan tolerances (`rtol=1e-4, atol=1e-5`):

| coefficient | max abs dev | max rel dev | pair-symmetric |
| ----------- | ----------- | ----------- | -------------- |
| `C6`  | `4.2e-06` | `2.5e-07` | exact |
| `C8`  | `4.1e-05` | `2.0e-07` | exact |
| `C10` | `2.7e-04` | `1.1e-07` | exact |
| `C12` | `9.0e-03` | `2.0e-07` | exact |

Residuals are consistent with the six/seven-figure rounding of the reviewed literals. See
[the recoupling spec](../specs/2026-08-17-isotropic-dispersion-recoupling.md).

### Completed since: constraints, bond graph, fit points

- **PDef active-variable constraints** — done ([spec](../specs/2026-08-17-pdef-constraint-derivation.md)).
  Reproduces the reviewed definition exactly (O 38/38, H1 66/66, 104 independent variables)
  and, independently, the reviewed *output*: every symmetry-forbidden tensor entry in the
  reviewed model is exactly `0.000e+00` (1804 checked for O, 1188 for H1, all frequencies).
  This eliminates the invented hydrogen `alpha_yz = +/-5.34` (eigenvalues
  `[-4.51, -0.011, +6.19]`) that the unconstrained fit produced. Verified to detect
  `C2v(Z)` geometrically even under the reviewed `symmetry c1 / no_com / no_reorient` flags.
- **Bond-graph derivation** — done. Covalent-radius based, scale `1.3`, reusing the
  versioned Bragg–Slater table promoted out of `isa_weights.cc`. Fails closed on
  disconnected graphs (so non-covalent complexes are rejected; accepted, monomer-only).
- **WSM fit-point generation** — done. Nested equidistant Lebedev surfaces, exact `O_h`
  orbit structure, `407` points by default with measured symmetry deviation exactly `0.0`
  and fit recovery stable to `~1e-12` across `129/189/249` points.

- **GRAC three-SCF orchestration, stage chaining, publication** — done in Task 7; see the
  Task 7 record above. The driver runs the three SCFs and the calculator receives them, per
  [the wiring spec](../specs/2026-08-17-end-to-end-wiring.md).

### Remaining work

1. **Site misdistribution of the localized response — the sole remaining parity gap.**
   The hydrogen `xz` under-determination recorded here earlier was **resolved 2026-08-17**:
   the reviewed PFIT protocol anchors the whole rank-1 dipole block, not just its diagonal
   (its log penalizes exactly seven parameters, including `H1_10_11c`, which *is* the `xz`
   component). Anchoring the full block moved hydrogen `xz` from `+4.29` to `-0.679` and
   restored a positive-definite published hydrogen dipole block, eigenvalues
   `(0.654, 0.654, 2.079)`.

   What remains is that our **LW-localized values differ from the reviewed anchors**, and
   since the penalty holds the dipole block near its anchor, the final answer inherits that
   error. At PBE0/aug-cc-pVDZ against the reviewed aug-cc-pVTZ anchors:

   | | `alpha_xx` | `alpha_yy` | `alpha_zz` |
   | - | ---------- | ---------- | ---------- |
   | `O` ours | `6.692` | `6.854` | `6.494` |
   | `O` reviewed | `7.035` | `5.764` | `5.583` |
   | `H` ours | `1.583` | `0.655` | `1.150` |
   | `H` reviewed | `1.557` | `1.621` | `2.009` |

   The totals nearly agree (conservation `0.955`), so this is a **misdistribution between
   sites, not a magnitude error**: our oxygen is far too isotropic and absorbs out-of-plane
   response the reviewed model assigns to the hydrogens. `alpha_xx` agrees well on both
   sites while `yy` and `zz` are badly split — a directional signature. This lives in the
   ISA partition (Task 4) or the LW localization (Task 3). Bisect against the reviewed
   anchor table, which is a per-component oracle for Task 3's output; see
   [the debugging map](../specs/2026-08-17-parity-debugging-map.md).

   Note the reviewed model's *full* L3 hydrogen array is itself not positive definite (its
   log reports a `-0.754777` eigenvalue), which the plan's Global Constraints anticipate.
   The published *dipole* block is a separate matter and is now positive definite.

2. **Molecular-polarizability conservation deficit — FIXED 2026-08-17.** Root cause: the
   WSM fit points were generated 2.0–4.0 bohr from the nuclei, i.e. *inside* the molecular
   charge density, where a rank-3 distributed multipole model cannot represent the
   point-to-point response at all. See the record below.
3. **Task 8 full-protocol parity run** — the aug-cc-pVTZ/GRAC protocol has never been run,
   and the six reviewed-literal comparisons are skipped by default behind
   `PSI4_ATOMIC_POLARIZABILITY_PARITY=1`. They must be reported as skipped, never as passed,
   until they are actually exercised.

### Conservation-deficit record (2026-08-17)

**Stage localization.** The site-pair response was summed with every site's rank-0 block
translated to a common origin, which is algebraically the molecular dipole operator for any
partition of unity. Static, PBE0/aug-cc-pVDZ, DFT `590/99`, ISA `60/18/24`:

| stage | `xx` | `yy` | `zz` | isotropic | vs. own response |
| ----- | ---- | ---- | ---- | --------- | ---------------- |
| ISA-Pol site-pair response (Task 4) | `9.9106` | `8.3580` | `8.9598` | `9.0761` | — |
| LW `local[site]` rank-1 sum (Task 3) | `9.9106` | `8.3580` | `8.9598` | `9.0761` | `1.0000` |
| refined L3 rank-1 sum (Task 5, published) | `7.7729` | `4.4369` | `5.8910` | `6.0336` | **`0.6648`** |
| Psi4 molecular `DIPOLE POLARIZABILITY` | `10.1035` | `8.7373` | `9.2378` | `9.3595` | — |

So Tasks 3 and 4 conserve exactly and the entire loss was inside `refine_wsm`. The 3%
between the response stage and Psi4's molecular value is the deliberate kernel difference
(`25% CHF + 75% ALDA` versus PBE0's own kernel) and is *not* a defect. That number also
refutes the kernel hypothesis outright: the translated site-pair sum **is**
`4 mu^T (H1 + omega^2 H2^-1)^-1 mu` with the reviewed kernel, so a kernel over-screening
by 37% is arithmetically impossible given a measured ratio of `0.970`.

**Root cause.** The fit points sat 2.0–4.0 bohr from the nuclei, inside the valence
density. Feeding the conserving LW model through the same multipole formula `refine_wsm`
fits reveals it cannot represent the ab initio point response there:

| nearest-nucleus distance | `Pi_obs` | `Pi_model(LW)` | ratio |
| ------------------------ | -------- | -------------- | ----- |
| `2.0` bohr | `2.504e-01` | `1.406e+00` | `5.61` |
| `3.0` bohr | `4.614e-02` | `5.332e-02` | `1.16` |
| `4.0` bohr | `1.606e-02` | `1.627e-02` | `1.013` |
| `6.0` bohr | `3.543e-03` | `3.549e-03` | `1.002` |
| `8.0` bohr | `1.328e-03` | `1.334e-03` | `1.004` |

Charge penetration damps the true response by a factor of `5.6` at 2 bohr. Least squares
against data the model provably cannot fit (461% relative model error) drives the fitted
polarizabilities down. The reviewed CamCASP point grid, read from the ignored development
oracle, places its 500 points **4.63 to 11.46 bohr** from the nearest nucleus (mean
`8.48`) — two to three times further out than ours.

Why the deficit ramped with frequency (`0.63` at `omega=0` to `~0.90` in the tail) and
looked like over-screening: at higher imaginary frequency the response is shorter ranged,
so the unrepresentable penetration region contributes less of the total and the fit error
shrinks. The ramp is generated entirely inside `refine_wsm` — the site-pair response
conserves at every frequency.

**Enabling defect.** `solve_constrained_least_squares` compares the `1e-4` policy cutoff
against the *absolute* weighted column 2-norm. Since the irregular harmonics fall off as
`r^-(2l+1)`, that makes the retained rank a function of the shell radii: at the reviewed
radii the rank-3 columns are pruned and the constraint elimination then fails closed with
"constraints are ambiguous (linearly dependent)". The absolute reading therefore *cannot
express the reviewed protocol at all*. `refine_wsm` now scales the policy cutoff by the
largest weighted column norm, making it the rank threshold it was always meant to be.
On the corrected default grid the smallest rank-3 column is `2.36e-05` absolute (pruned
under the absolute reading) but `2.78e-04` relative (retained).

**After the fix**, defaults `4.5`–`11.5` bohr, 329 points:

| | `xx` | `yy` | `zz` | isotropic |
| - | ---- | ---- | ---- | --------- |
| published atomic sum (static) | `9.8572` | `8.1627` | `8.7956` | `8.9385` |
| ratio to the site-pair response | `0.9946` | `0.9766` | `0.9817` | `0.9848` |
| ratio to Psi4 molecular | `0.9756` | `0.9342` | `0.9521` | `0.9550` |

At `omega = 0.370417` the ratio to the site-pair response is `0.9902`. Against the reviewed
literals the isotropic sum now runs `0.931` at `omega=0` rising through `0.998` at
`omega=1.26` (residual is the aug-cc-pVDZ/aug-cc-pVTZ basis difference and the kernel); the
pre-fix curve ran `0.629` to `0.915`. The apparent "second, frequency-independent 10%
residual" was the same single cause.

The regression tests are
`test_atomic_polarizabilities.py::test_published_atomic_sum_conserves_the_site_pair_response`
(static and `omega=0.370417`) and
`::test_published_atomic_sum_conserves_psi4s_molecular_dipole_polarizability`. Both fail on
the pre-fix configuration by 19–23%.

### Constraints discovered during implementation

- **Fit points must lie outside the charge density.** Superseded claim, kept as a warning:
  an earlier revision read the `1e-4` WSM cutoff as an absolute weighted column norm and
  concluded from it that the shell limits had to be `2.0`–`4.0` bohr rather than van der
  Waals multiples. That inference was invalid and cost 36% of the molecular polarizability;
  see the conservation record above. The cutoff is relative and cannot select a radial
  convention.
- **The reviewed 2000-point grid is architecturally infeasible** with the current dense
  pair-row design: 2000 points implies ~2.0e6 rows x 360 columns, about `5.8 GB` for the
  design matrix alone, and `kWSMMaximumPoints` is `500`. Use the converged 329-point grid.
- **Memory.** Psi4's `500 MB` default supports only ~125 fit points; 249 points needs
  ~`510 MB` and the 329-point default ~`0.9 GB`. End-to-end wiring must raise process memory
  explicitly or `refine_wsm` fails closed on its own default grid.
- **Grid quality must be pinned.** The existing SCF test fixtures
  (`dft_spherical_points 50 / radial 12`, ISA `30/10/12`) are far too coarse for the
  `1e-4` gate — `localize_lw` rejects them above `1e-2`. Production quality measured at
  `302/50` with ISA `60/18/24`. Do not mistake the fixtures for the parity protocol.
- **Pre-existing defect, worked around not fixed:** `PointGroup::operator=` in
  `psi4/src/psi4/libmints/pointgrp.h` does not copy `bits_`, so a copied `PointGroup` has an
  indeterminate character table. Also `SymmetryOperation::rotation(2)` leaves
  `sin(pi) ~ 1.2e-16` off-diagonal, so `C2(z)` is not exactly diagonal; comparisons must use
  a tolerance, not exact zero.

### Task 1: Native representation and public plumbing

**Files:** create atomic header/source and math test; modify CMake, OEProp header/source, options, Python allowlist.

**Interfaces:** `AtomicPolarizabilityCalculator(std::shared_ptr<Wavefunction>)`; `void OEProp::compute_atomic_polarizabilities()`; `compute()` publishes the seven public array variables.

- [ ] Write failing Python API tests that `core.OEProp.valid_methods` accepts `ATOMIC_POLARIZABILITIES`, that requesting it creates no external process, and that missing required response data raises a named `PsiException` rather than publishing partial arrays.
- [ ] Run the focused test and record the missing-method failure.
- [ ] Add the CMake source entry, OEProp task dispatch, allowlist entries, `ATOMIC_POLARIZABILITY` options, and a calculator skeleton which rejects unsupported wavefunctions before allocating output.
- [ ] Re-run API tests; verify accepted dispatch and fail-closed unsupported behavior.
- [ ] Commit `feat: add native atomic polarizability plumbing`.

### Task 2: Frequency and tensor algebra kernel

**Files:** modify atomic header/source; create math test.

**Interfaces:** `FrequencyGrid make_casimir_grid(unsigned int nonzero_count, double scale)`; `Matrix local_spherical_dipole_to_cartesian(const L3Matrix&)`; `Matrix rotate_tensor(const Matrix&, const Matrix&)`; `std::array<double, 6> pack_symmetric_tensor(const Matrix&)`.

- [ ] Write failing unit tests for the eleven reviewed frequency values, `(10,11c,11s) -> (z,x,y)` mapping, right-handed rotation, packed ordering, non-orthonormal frame rejection, and tensor symmetry rejection.
- [ ] Run only math tests and record failures caused by absent functions.
- [ ] Implement these pure functions with no SCF, file, JSON, or process dependency; require finite values and `det(R)>0`.
- [ ] Re-run math tests and commit `feat: add atomic tensor and quadrature algebra`.

### Task 3: Clean-room LW localization kernel

**Files:** modify atomic header/source and math test.

**Interfaces:** `LocalizedResponse localize_lw(const SitePairResponse&, const BondGraph&, double residual_tolerance)`.

- [ ] Write failing synthetic three-site tests: graph Laplacian symmetry/null mode, molecular sum preservation, reciprocity preservation, no transfer over absent bonds, and residual rejection above tolerance.
- [ ] Run those tests and record red output.
- [ ] Independently implement the published graph-Laplacian/pseudoinverse bond-flow equations and real-spherical translation operations; do not consult/copy ORIENT source while coding.
- [ ] Re-run tests and commit `feat: add clean-room LW response localization`.

### Task 4: Native point-response and ISA partition provider

**Files:** modify atomic header/source; add SCF-marked tests.

**Interfaces:** `SitePairResponse compute_isapol_response(const FrequencyGrid&, const ISAWeights&, ResponseKernel)`.

- [ ] Write failing small-basis H2O tests for partition unity, molecular response sum recovery, finite imaginary-frequency response, and reciprocity. Tests must construct a Psi4 wavefunction and must not contain CamCASP literals.
- [ ] Confirm red due to unsupported provider.
- [ ] Implement native point-potential/multipole perturbation assembly, frequency-dependent coupled response, and stockholder-weighted site-pair contraction. Implement the reference response kernel explicitly as 25% CHF + 75% ALDA rather than inheriting PBE0’s kernel.
- [ ] Make GRAC ground-state prerequisites explicit and fail closed if unavailable; preserve a test-only small-basis provider separate from the protocol parity provider.
- [ ] Run focused SCF tests and commit `feat: add native ISA-Pol response partition`.

### Task 5: PFIT-style L3 constrained refinement

**Files:** modify atomic header/source and math/SCF tests.

**Interfaces:** `RefinedL3Model refine_wsm(const LocalizedResponse&, const PointResponseData&, const PDefConstraints&, RefinementOptions)`.

- [ ] Write failing synthetic least-squares tests for exact reconstruction, dipole-only diagonal penalties, H2 copy symmetry, cutoff pruning, finite-condition-number rejection, and residual diagnostics.
- [ ] Run and record red output.
- [ ] Implement design-matrix construction, constraints, QR/SVD solve, diagonal anchor penalty, L3 spherical packing, and diagnostics. Use documented weight 4/coefficient `0.001`/cutoff `1e-4`; do not solve normal equations by explicit inversion.
- [ ] Re-run tests and commit `feat: add constrained L3 WSM refinement`.

### Task 6: Native real-spherical dispersion engine

**Files:** modify atomic header/source and math test.

**Interfaces:** `DispersionMatrices compute_dispersion(const std::vector<RefinedL3Model>&, const FrequencyGrid&)` returning C6/C8/C10/C12 matrices.

- [ ] Write failing analytic tests for C6 isotropic `3/pi` integration, pair symmetry, quadrature convergence, and recoupling-table validation; add fixture tests for each permitted rank-pair contribution to C8/C10/C12.
- [ ] Run and record red output.
- [ ] Implement independently specified real-CG table loader/validator, rank recoupling, frequency integration, and isotropic `00 00 0` extraction. Require all L3 inputs; reject missing terms instead of manufacturing C8–C12 from dipoles.
- [ ] Re-run tests and commit `feat: add native L3 dispersion recoupling`.

### Task 7: OEProp end-to-end publication and fixed reference tests

**Files:** modify atomic/OEProp code; create `tests/pytests/test_atomic_polarizabilities.py`; remove obsolete misspelled test if present.

**Interfaces:** one OEProp call publishes static `(3,6)`, dynamic `(33,6)`, frequencies `(11,1)`, and four `(3,3)` Cn matrices.

- [ ] Extract reviewed literals once from the already approved JSON into tracked Python constants, then verify the test module does not import/read JSON or call external commands.
- [ ] Write the six failing property tests required by the design, plus shapes, finiteness, global symmetry, H2O C2 relation, and diagnostic trace checks.
- [ ] Run each test against the native pipeline and retain its red output until its corresponding stage is implemented.
- [ ] Connect Tasks 2–6 through OEProp; publish only complete arrays after all validation gates pass.
- [ ] Run focused parity tests and commit `feat: publish native atomic polarizability parity outputs`.

### Task 8: Oracle comparison and scientific acceptance

**Files:** modify tests only as needed for fixed literals; update developer documentation.

- [ ] Run the native fixed-protocol H2O calculation once and compare all 11 global tensors and C6/C8/C10/C12 against the checked-in literals with the global tolerances.
- [ ] Investigate every mismatch by stage invariant; do not loosen tolerances, replace literals, or invoke external software from pytest.
- [ ] Run the focused module, relevant Psi4 regression suite, build/test targets, and static checks.
- [ ] Obtain independent code and scientific reviews; record residual C12/L3-model and full-L3-positive-definiteness caveats.
- [ ] Commit documentation/test changes only after all property tests pass.

## Plan self-review

- Coverage: Tasks 1–7 cover every required public variable, coordinate mapping, frequency grid, native ISA response, LW localization, PFIT refinement, and C6–C12; Task 8 covers oracle acceptance.
- Constraints: every task prohibits production external dependencies and preserves TDD/fail-closed behavior.
- Known prerequisite: Task 6 requires an independently specified/validated real-CG contraction table and rank-pair formulas; it may not infer prefactors solely from four final scalar values.
