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
