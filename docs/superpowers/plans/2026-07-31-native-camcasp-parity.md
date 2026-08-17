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
| 4. ISA-Pol response | **runs; magnitude unverified** | Full composed chain verified in `test_native_atomic_polarizability_source_guard.py:96-126`. The molecular-sum check confirmed only *diagonality* (`7e-15`) and monotonic decay, **not magnitude against an independent molecular polarizability** — see the conservation deficit below. |
| 5. WSM refinement | **done, but under-determined without constraints** | Runs, conserves the isotropic sum to `3e-4`. Anisotropy is NOT pinned down: see the gaps below. |
| 6. Dispersion recoupling | **done, oracle-verified** | All four coefficients within `2.5e-7` relative of the reviewed CASIMIR values; see below. |
| 7. End-to-end publication | **done** | All seven variables publish from one `OEProp` call on the SCF triple; verified end to end on PBE0/aug-cc-pVDZ. See the Task 7 record below. |
| 8. Oracle acceptance | **partial** | Frequencies and C6–C12 accepted in isolation. The full aug-cc-pVTZ/GRAC protocol has never been run; the six reviewed-literal comparisons exist but are skipped by default. |

Test suite for this feature: **375 passing** under `-m mints`, 0 failing, 6 skipped
(the reviewed-literal parity comparisons).

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

- **Memory.** The default 407-point fit grid needs a WSM peak of `454,828,904` bytes, and
  the stage gate reserves half of configured memory, so it requires at least ~0.87 GiB
  configured. Psi4's 500 MB default fails closed. The driver sets 4 GiB explicitly
  (`PIPELINE_MEMORY_BYTES`) and restores the previous value afterwards.
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

Known non-parity residual at aug-cc-pVDZ, left for Task 8: the published hydrogen `xz`
component is about `-0.91` against a reviewed `+0.0058`. It is symmetry-allowed (`10` and
`11c` share `A'` at a `Cs` site), so the mask is not at fault, and the debugging map rules
out the publication and dispersion math, which points at Task 4 or Task 5. C10/C12 also
come out negative for hydrogen at this basis, consistent with a non-positive isotropic
rank-2/rank-3 trace in a small basis.

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

1. **Investigate the hydrogen `xz` discrepancy.** Published `-0.91` at aug-cc-pVDZ against a
   reviewed `+0.0058`. It is symmetry-*allowed*, so the mask is not responsible, and the
   frequency grid, spherical/Cartesian mapping, and dispersion recoupling are all
   oracle-exact — so the cause lies in Task 4 (ISA-Pol response) or Task 5 (WSM refinement).
   **This is the top open item and the main obstacle to polarizability parity.** Bisect using
   the stage oracles listed in
   [the debugging map](../specs/2026-08-17-parity-debugging-map.md).
2. **Molecular-polarizability conservation deficit — confirmed defect, not a basis effect.**
   Measured 2026-08-17 at PBE0/aug-cc-pVDZ, DFT `590/99`, ISA `60/18/24`, comparing the
   published atomic sum against Psi4's own `DIPOLE POLARIZABILITY` at the identical
   functional, basis, and grid:

   | component | published atomic sum | Psi4 molecular | ratio |
   | --------- | -------------------- | -------------- | ----- |
   | `xx` | `7.7729` | `10.1035` | `0.77` |
   | `yy` | `4.4369` | `8.7373` | `0.51` |
   | `zz` | `5.8910` | `9.2378` | `0.64` |
   | isotropic | `6.0336` | `9.3595` | `0.64` |

   The deficit is **anisotropic and worst out of plane** (`yy`, perpendicular to the
   molecular plane). For contrast the reviewed model sums to `(10.191, 8.997, 9.603)`,
   isotropic `9.597`, conserving to 1–2%. So a correct distributed model *does* reproduce the
   molecular polarizability, and ours loses about 36% of it.

   Scope note: the earlier stage checks do not localize this. LW localization and WSM
   refinement were each shown to conserve *relative to their own input*, and the Task 4 check
   verified only diagonality and decay, never magnitude — so a uniform deficit originating in
   the response stage would have passed every existing test. Confirm by summing the ISA-Pol
   site-pair response directly and comparing to `9.3595` before looking downstream.

   Leading hypotheses, in order: (a) the `25% CHF + 75% ALDA` reference kernel is assembled
   incorrectly (e.g. blended as a weighted average of kernels rather than the intended
   functional derivative); (b) the site-pair contraction drops contributions, e.g. charge-flow
   (rank 0) terms that the reviewed L3 model has absorbed into local dipole terms — note the
   reviewed model carries no rank 0, yet still conserves, which is precisely what WSM
   refinement is supposed to achieve; (c) transition-multipole projection is truncated or
   misnormalized.
3. **Task 8 full-protocol parity run** — the aug-cc-pVTZ/GRAC protocol has never been run,
   and the six reviewed-literal comparisons are skipped by default behind
   `PSI4_ATOMIC_POLARIZABILITY_PARITY=1`. They must be reported as skipped, never as passed,
   until they are actually exercised.

### Constraints discovered during implementation

- **Shell limits are absolute bohr, not vdW multiples.** Decided by measurement: under the
  vdW reading the minimum weighted design-column norm is `6.6e-06`, 15x *below* the frozen
  WSM column cutoff of `1e-4`, which would prune the entire rank-3 block and contradict
  `wsm_rank=3`. Under the bohr reading it is `1.6e-02`, clearing the cutoff by 163x.
- **The reviewed 2000-point grid is architecturally infeasible** with the current dense
  pair-row design: 2000 points implies ~2.0e6 rows x 360 columns, about `5.8 GB` for the
  design matrix alone, and `kWSMMaximumPoints` is `500`. Use the converged 407-point grid.
- **Memory.** Psi4's `500 MB` default supports only ~125 fit points; 249 points needs
  ~`510 MB` and the 407-point default ~`1.3 GB`. End-to-end wiring must raise process memory
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
