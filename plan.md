# libisapol implementation handoff

**Start here:** read `psi4/src/psi4/libisapol/SPEC.md`, then this plan. The spec owns scientific/API contracts; this file owns execution state. Preserve existing uncommitted work. No commits, resets, pushes or reference-tree edits were performed.

## Ongoing objective — continue until end-to-end acceptance

User explicitly requests continuing implementation until end-to-end passes. Do not
stop at supplied-input kernels or relabel them as full parity. Remaining dependency
order: synchronous sweep, controller/activation/tails, native density and response,
partitioned multipoles/alpha, localization/PFIT/dispersion, driver/end-to-end water.
The Libint2 reference switch is independently deferred, not a blocker on current work.
Read-only native Drho-C/Libint2 interface audit `d3f0e11afa906572641f48dee539c9c2f` is
completed independently while whole-sweep reference jobs execute (verified artifacts
under `.pi/delegate/01a07228-a554-7f85-8b2e-e4c5136ec050-3719637/d3f0e11afa906572641f48dee539c9c2f`).
No native-density implementation or parity is claimed from that audit.
Key contracts: NN is pair space, not integral norm; current Drho-C is Coulomb-norm
`(J+lambda*q*q^T)d = 2*sum_occ(t_ii)+lambda*N*q`, with finite charge penalty, no
post-rescaling. Doo-C is a separate pair-fit/trace product. Native integrals require
actual MAIN and molecular AUX, not JKFIT; AO density can only contract the integral
RHS, not replace the fitted ISA density. Psi4 BasisSet constructors can re-embed
normalization even through ShellInfo Normalized; prefer a narrowly validated direct
Libint2 Shell path with `embed_normalization_into_coefficients=false` and explicit
Cartesian/spherical transforms. Local Libint2 2.13.1 headers confirm that path exists.
A separate MAIN/MO/J/B/q/RHS DF export is required before claiming native DF parity.

Native-capture hook delegate `dee626233d8f76af7249cb5483a87634f` completed (verified
answer SHA256 `6965622c30924bcdad2d4404198f891bbe4a86e95048901b4c6b2f6134d3c73c`).
Implementation-ready source anchors, not yet implemented or built:
- `df_Smat.F90`: observe resident J before line 343 write/release, and constrained A
  before line 555 close/release. Metrics may be built BEFORE entering the density
  routine, so records must be molecule-keyed and independent of density entry.
- `df_monomer.F90:609`: before writing Trho, export resident `mol%main`, `mol%aux`,
  `mol%C%matrix(1:ndim,1:nocc)`, `Iint%int%vector`, and original Trho RHS. At line 633,
  after the existing assignment, export solved Drho. Do NOT open/release matrices
  just to observe: the LU solver mutates/resizes/releases both S and Trho.
- Occupations are the routine's assumed 2, not an independently read occupation
  vector. Require nelectrons==2*nocc, retain electronic-state input provenance.
- MAIN DALTON Cartesian needs both angular scaling and permutation, not the current
  GAMINT Cartesian adapter. Preserve SCFcode and actual global g_scf_code. MAIN/AUX
  effective contraction coefficients are already normalized in place.
- Raw B is absent from do_DFrho_monomer; optional `make_T_AO_mono` hook before the
  line-542 block write sees rows AUX, column mu+ndim*(nu-1), before MO/penalty terms.
  First native gate can deliberately export J/A/q/RHS/C/bases/Drho, with B absence
  explicit; this tests metric and occupied trace but cannot localize B discrepancies.
- Use observer-only resident-array reads, strict fresh/Coulomb/LU/NN/DALTON/eta=gamma=0
  guards, duplicate/selector rejection, new isolated builds and trace-on/off hashes.
  Existing generic patch helper must support indented subroutine declarations.
No native capture or native Libint2 parity is claimed by these read-only findings.

### Completed — explicit synchronous no-tail sweep

New `isa_sweep.{h,cc}` and bindings own per-atom primitive providers, shape bases and
maps. Every atom samples the same immutable old shape state; Gaussian/no-tail values
are clipped at zero, while raw projected coefficients stay signed. Shape neighbour
indices are sweep-atom indices; density neighbours remain independent molecular AUX
centre indices. Per-atom quadrature is explicit. Results return fresh next state,
frozen fits and selected-shape clipping counts. No convergence/activation/mixing or
active-tail fallback is claimed. Synthetic tests in `test_isapol_sweep.py` check
independent scalar Gaussian equations, a deliberately different asynchronous result,
permutation, screening, clipping/zero denominators and failure atomicity.
Build task `b8328d721` compiled/installed successfully (same inherited optional
stubgen/cleanup exit 1; `.pi/audit/no-tail-sweep-build.log`). Staged import and
byte comparison passed. Five-file regression task `b69cb178f` passed **277 tests
in 2.17 s**; `.pi/audit/no-tail-sweep-tests.log`. Python compilation and diff checks
passed. Command extends the established staged pytest invocation with
`tests/pytests/test_isapol_sweep.py`.

### In progress — explicit Gaussian shape and Func-1/Fit-3 tail kernel

Implement an owned radial shape expansion from effective s-shell coefficients,
analytic exterior Gaussian charge, finite-difference slope (1e-8), strict 1<b<4
with previous-valid-exponent fallback, and tail-charge-conserving amplitude. Sampling
must retain interior signed values in the active-tail branch and clip only in the
no-tail branch. Deterministic undefined-fit handling replaces upstream uninitialized
saved cross-call A gate and undefined-IP behavior; no undefined-memory parity claim. Native controller and
activation scheduling remain subsequent work. Tail build `bfb582184` compiled and
installed successfully (same inherited script-only exit 1); six-file test task
`be75952e9` passed **300 tests in 2.29 s**, log `.pi/audit/shape-tail-tests.log`.
Staged extension verification, Python compilation and diff checks passed. This
validates the standalone Gaussian/Func-1-Fit-3 kernels, not production tail state.
Controller/supplied-tail sweep additions have their own pending rebuild/tests.

### In progress — ordinary-A explicit-input controller

`IsaASweep` generalizes the no-tail sweep with explicit per-site tail policies;
`IsaNoTailSweep` remains a compatibility alias and `run()` still means no-tail.
New `isa_controller.{h,cc}` provides initialize/step/run and inspectable restart
cursors for identical inputs/settings. It implements W convergence before mixing,
next-sweep non-latched activation, strict iteration tail threshold, tail analysis
from OLD shapes, saved pre-mixing charges and no forced extra converged sweep.
Full atomic D remains unmixed. Undefined norms fail rather than claim convergence.
Only ordinary A/W is supported: no DIIS, symmetry, initialization recipe, decoupled
subiterations or self-consistent postconvergence tail loop. Explicit per-site tail
cutoffs and allowed/MaxDelta masks replace hidden element guesses. `run()` reports
nonconvergence at max iterations. Source-driven synthetic ordering/restart/activation
tests are in `test_isapol_controller.py`; production controller parity is not yet
established. Build task `b39d8dfdc` compiled/installed successfully with the inherited
script-only exit 1; `.pi/audit/isa-controller-build.log`. Seven-file regression task
`b0395989f` passed **323 tests in 2.51 s** (`.pi/audit/isa-controller-tests.log`),
with staged import/byte comparison, Python compilation and diff checks passing.
This predates the added exclusion-mask regression. Source inspection at
stockholder.F90:1290–1307 corrected the audit summary: excluded dummy sites are
omitted from BOTH MaxDelta and global convergence, though still fitted/tested.
Updated the C++ mask reduction and added a two-site exclusion regression; rebuild
`bb1703b0f` compiled/installed successfully (same script-only exit 1;
`.pi/audit/isa-controller-mask-build.log`). Eight-file staged regression task
`ba717ef6a` passed **341 tests in 2.49 s**, including the exclusion mask and
new sweep-format tests; `.pi/audit/controller-capture-tests.log`. Staged import/
byte comparison, Python compilation and diff checks passed. Whole-sweep production
capture/replay remains pending; this does not close the controller parity gate.
Read-only delegate `d9dc569776477d55b33eb1fa55c31c227` is locating fail-closed whole-sweep
capture hooks for production transition evidence; it completed with verified
artifacts in the same delegate root. Its initial dispatch recommendation was WRONG for this source and was corrected
by an actual run plus dispatcher inspection (stockholder.F90:823): fresh calculations
also use `Iterative_Stockholder_Atoms_restart`; the unsuffixed routine is historical.

### In progress — whole-sweep production instrumentation

New `oracle/capture_isa_sweep.py` keeps per-atom v2 streams unchanged and adds a
separate PRE/POST controller state sidecar selected by ordinary iteration. It
requires the three-site, no-DIIS/no-symmetry ordinary A/W/Func1-Fit3 source path.
The initial two preparation attempts failed closed before creating the destination:
unnamed `cp_begin` end marker and main routine's contained `implicit none` anchors
were corrected with unique scopes. Fresh isolated source is
`.pi/audit/production-camcasp-sweep-v1`; original sources/archives are untouched.
Reference serial build task `be6dea86e` completed successfully
(`.pi/audit/sweep-reference-build.log`), preserving prior reference builds.
Fresh adapted input directories `.pi/audit/sweep-water-{first,transition,final,untraced}`
are prepared for iteration 1/21/53 captures and a tracing-disabled comparison.
`oracle/replay_isa_sweep.py` strictly reads the sidecar, cross-checks all three v2
streams and reconstructs one C++ controller transition. Synthetic format tests
are in `test_isapol_sweep_checkpoint.py`; **17 passed in 1.41 s** under task
`b1ed3ea99` (`.pi/audit/sweep-format-tests.log`), with Python compilation and diff
checks passing. These are format checks, not production evidence.
Reference capture task `bfe0cb17a` is running via `.pi/audit/run-sweep-reference.py`;
log `.pi/audit/sweep-reference-runs.log`. It retains compiler/library/executable/source
provenance per new run, strictly parses all outputs, and compares seven final
shape/tail artifacts across trace-on/off runs. Task `bfe0cb17a` failed correctly:
the reference converged in 53 iterations but no sidecar was produced because the
unsuffixed routine was never executed. This exposed the incorrect audit dispatch
assumption. The v1 source/build/run remain intact as failure evidence.

Corrected hooks target the live restart-capable routine while explicitly rejecting
actual restart inputs. Fresh `.pi/audit/production-camcasp-sweep-v2` compiled successfully
under `b1ba9db5b` (`.pi/audit/sweep-v2-reference-build.log`).
Live-path capture task `b6a730f9b` is now running (`.pi/audit/sweep-v2-reference-runs.log`).
A new harness dispatch guard
and regression reject the stale unsuffixed target or ambiguous calls. Focused task
`b34bceb57`: **18 passed in 1.27 s** (`.pi/audit/sweep-dispatch-tests.log`), Python
compilation and diff checks passed. New run inputs
are `.pi/audit/sweep-v2-water-{first,transition,final,untraced}`; runner
`.pi/audit/run-sweep-v2-reference.py` is prepared. Live-path task `b6a730f9b`
produced complete first-sweep outputs but the parser rejected initial legacy state:
`ISAcharge` is NaN from pre-fit 0/0 rescaling, and undefined tail index is -1.
Reader now explicitly retains the nonfinite third charge field as a diagnostic token
(None numeric value), while saved shape charges, coefficients and active tail data
remain strictly finite. Undefined -1 is allowed only with the tail undefined.
A regression covers these exact initialization semantics. The capture files were
not modified or rerun. Resume task `bebc3898e` validates the existing first case and
continues fresh remaining runs (`.pi/audit/sweep-v2-reference-resume.log`). Concurrent
first transition replay task `b82e2a85f` writes new controller-replay.json/log in the
first run directory. Task `b82e2a85f` PASSED the full first controller transition:
all three atoms have **68,310 points**; flags and next controls match exactly;
max D/W error **5.711431327881655e-12**, max raw shape-charge error
**5.027089855502709e-13**, max tail-parameter error **1.4210854715202004e-14**,
max delta error **1.1102230246251565e-16**, and max residual **4.2007969131848756e-17**.
The scaled 1e-9 threshold was unchanged. Resume task `bebc3898e` completed all
three selected captures plus the untraced run; seven final shape/tail artifacts are
byte-identical across all four. Evidence: `.pi/audit/sweep-v2-trace-comparison.json`.
Transition/final one-step replay task `b1d7ecb3a` FAILED at iteration 21:
all fitted/state/control checks passed except atom-2 tail, max absolute
**1.5411108833518483e-8**, scaled **5.142246590493385e-9**. Failed evidence retained.
Source `shape_function_eval_grad` evaluates translated Cartesian z-axis points and
ordered contracted-shell values before expansion multiplication. The previous
flattened `(d*c)*exp` radial path changed the 1e-8 finite difference. Reconstructing
that order through the existing explicit basis gives exponent **2.996960290082049**,
exactly the reference, versus previous **2.9969602999286837**.
Pre-fix regression task `b0ab32b25` passed the complete eight-file suite:
**343 passed in 2.50 s**, plus Python compilation and `git diff --check`
(`.pi/audit/isa-regression-current.log`). This covers the diagnostic-reader change,
NOT the subsequent C++ tail-order correction.
The C++ fit now preserves those points/order; three new regression cases cover translated
contracted shapes. Compile-only task `bcd3d15ff` PASSED with two jobs
(`.pi/audit/ordered-tail-compile.log`). With the pre-fix trajectory now finished,
staging/regression task `b1a0d8b3b` runs explicit `cmake --install` to the existing
stage prefix, byte-compares built/staged extensions, then runs all eight test files,
Python compilation and diff checks. Task `b1a0d8b3b` PASSED: **346 tests in 2.48 s**,
built/staged byte comparison, Python compilation and diff checks all succeeded.
Logs: `ordered-tail-install.log`, `ordered-tail-tests.log`.
Fresh production replay task `b8ed3d53d` PASSED at iterations 1/21/53:
max scaled errors **8.794295397261792e-13 / 1.1437309008791764e-12 /
9.29965468968671e-13**; max tail abs errors **1.4210854715202004e-14 /
1.7763568394002505e-15 / 8.881784197001252e-16**. All flags/controls match.
Reports: `ordered-controller-replay.{json,log}` in each capture directory.
Portable measured evidence with producer provenance, source/report/extension hashes
and trace comparison: `tests/pytests/data_isapol/camcasp_isa_controller_evidence.json`.
This certifies three injected-PRE transitions only, not the full trajectory.
Read-only delegate `dee626233d8f76af7249cb5483a87634f` is identifying exact native
Drho-C capture hooks (MAIN/MO/AUX/J/q/RHS/Drho) while the full trajectory executes. Corrected full
trajectory plus final-reference comparison is running as `b85449b91`, writing
`.pi/audit/cpp-controller-ordered-trajectory.{json,log}` and
`.pi/audit/cpp-controller-ordered-final-comparison.{json,log}`. Prior evidence is unchanged.
Corrected full task `b85449b91` converged in **53 iterations / 240.192337453 s**,
but final comparison FAILED at unchanged scaled 1e-9: maximum tail absolute error
**1.6838058414236912e-7** (scaled **2.428224964352429e-8**). All D/W, saved charges
and controls passed. Three injected-input transitions remain passed; full trajectory
parameter gate is explicitly still open. Task `be031955a` repeats the unchanged
arithmetic while retaining iteration-52 tail-input shapes in a fresh
`cpp-controller-lagged-trajectory.{json,log}`. Prepared diagnostic
`.pi/audit/diagnose-tail-conditioning.py` will verify exact trajectory reproducibility,
then compare actual/reference lagged inputs using C++ finite differences, 80-digit
finite differences at the same translated points, and analytic Gaussian slopes.
This measures conditioning; it does not change tolerances or certify acceptance.
Lagged-input task `be031955a` completed: converged in **53 iterations / 241.356130824 s**.
Diagnostic task `b9bf110ad` now runs the prepared 80-digit comparison, writing fresh
`.pi/audit/tail-conditioning.{json,log}`; its assertions also check rerun state identity
and reproduction of both recorded final exponents from the actual lagged inputs.
Task `b9bf110ad` failed an overly strict bitwise-equality assertion comparing the
recomputed C++ exponent with the reference (production replay certifies numerical,
not bitwise, agreement). Preserved v1 diagnostic script/log. Revised task `bf3c59ea6`
records both reconstruction errors and a four-epsilon diagnostic flag instead of
assuming equality; no production acceptance threshold changes. Output paths are
fresh `tail-conditioning-v2.{json,log}`. Task `bf3c59ea6` completed: full rerun state
is exactly reproducible, and both sets of lagged-input exponents reconstruct within
roundoff (only H1 reference differs by -4.440892098500626e-16).
O slope difference: double FD **-1.0798995475624906e-8**, 80-digit FD/analytic
**-6.347384839955339e-10**. H1: **-9.407373990910628e-9** versus
**-7.134879353998258e-10**. H2: **6.292259602247441e-9** versus
**-3.223395772522508e-9**. Maximum represented tail-density difference on 10,001
radial samples per atom is **4.137397646708507e-10**. These diagnostics establish
finite-difference amplification, NOT a passed parameter gate.

Ordinary Gaussian value/sample still flattened `(d*c)*exp`, unlike the reference's
ordered shell contraction. Corrected `value_squared` to sum primitive samples within
each shell before multiplying its expansion coefficient; analytic exterior-charge
amplitudes remain unchanged. The translated-contraction regression now requires
bitwise sampling agreement with the explicit basis expansion as well as Fit-3 order.
Task `bef55b022` PASSED compile/install, built/staged byte comparison,
**346 tests in 2.49 s**, Python compilation and diff checks (`ordered-sampling-*.log`).
Production replay task `b27d1c157` PASSED iterations 1/21/53 into fresh
`ordered-sampling-replay.{json,log}` files. Max scaled errors remain
**8.794295397261792e-13 / 1.1437309008791764e-12 / 9.29965468968671e-13**;
all flags match, and max tail errors remain **1.4210854715202004e-14 /
1.7763568394002505e-15 / 8.881784197001252e-16**. Full trajectory/comparison task
`b9d909242` writes fresh `.pi/audit/cpp-controller-ordered-sampling-trajectory.{json,log}`
and `cpp-controller-ordered-sampling-final-comparison.{json,log}`.
Full parameter tolerance remains unchanged; the gate is not yet passed.
Task `b9d909242` converged in **53 iterations / 239.986885482 s**, but FAILED final
raw-tail comparison: O abs **7.034917537396268e-9** (scaled **1.0145090346059646e-9**),
H1 abs **1.3679203103578175e-8** (scaled **5.097030976936278e-9**), H2 abs
**2.8819071573593646e-9** (scaled **1.073832272481616e-9**). D/W, charges and controls pass.
Source inspection identifies another explicit arithmetic difference: exterior charge
reference uses ordered double accumulation and `exp((-alpha*r)*r)`; C++ used long-double
accumulation and `exp(-alpha*(r*r))`. This is now source-ordered, with three exact
primitive-order regressions alongside independent quadrature tests. Build/stage/test
task `bfdeac0b6` PASSED compilation, staging, built/staged byte comparison,
**349 tests in 2.51 s**, and diff checks (`ordered-charge-*.log`). This does NOT
establish full raw-tail acceptance; no tolerance changes. Fresh transition replay
task `b53438b3b` PASSED all three cases (`ordered-charge-replay.{json,log}`).
Max scaled errors: **8.794295397261792e-13 / 1.1437309008791764e-12 /
9.29965468968671e-13**; all flags match. Max tail abs errors:
**1.4210854715202004e-14 / 1.7763568394002505e-15 / 4.440892098500626e-16**.
Full trajectory/comparison task `bcc4c6815` writes
`cpp-controller-ordered-charge-trajectory.{json,log}` and
`cpp-controller-ordered-charge-final-comparison.{json,log}` under `.pi/audit`.
Task `bcc4c6815` converged in **53 iterations / 242.174385517 s**, but FAILED raw-tail
comparison: O abs **1.6357202348160627e-7**, scaled **2.3588804665973028e-8**.
H tails, all D/W, charges and controls pass the existing metric. No further algebraic
rearrangements will be selected just for lower final error. Strict new comparator
task `be8610e35` completed with FAILED acceptance (not a reader/runtime error), in
`cpp-controller-ordered-charge-strict-comparison.{json,log}`. All added checks pass:
iteration count, history consistency, per-atom flags, deltas/MaxDelta, cutoff endpoint
invariance and configuration. The sole failing joint-vector check remains O tail:
scaled **2.3588804665973028e-8**. Separately reported O exponent scaled error is
**3.9251197217008126e-9**; H1 amplitude alone is **2.4427939115945208e-9**, although
its historical joint A/b check passes. Individual diagnostics do not replace the
existing gate. These omissions were real validator weaknesses but do not explain
or remove the current tail mismatch.
Task #14 remains open/pending and #11 remains blocked; independent native capture
now advances as task #15 without claiming the raw-tail or end-to-end gate passed.

### Native Drho-C capture implementation (in progress)

Added `oracle/capture_native_df.py`: an observer module emitted before `df_Smat`,
explicit build dependency for the density consumer, resident J/A hooks before
release, MAIN/AUX/C/q/RHS before the solver and Drho after assignment. Records use
fresh files; strict selected molecule, NN/Coulomb/LU/DALTON/closed-shell/penalty guards;
no reference-object open/close/write/release calls in observers. Serialization reuses
only the existing basis writer, not the independent polynomial oracle. Raw B is
explicitly absent; occupations are labeled as assumed by the closed-shell routine.
`test_isapol_native_capture.py` checks anchors, residency-safe observer design,
source preservation and fresh-destination protection. Task `bf6cd30f1` runs harness
tests/compilation/diff checks (`native-df-harness-tests.log`): task `bf6cd30f1`
PASSED **7 tests in 1.22 s**, Python compilation and diff checks. This pre-v2 result
does not cover the later real-source anchor correction; that regression is included
in the separate v2 build task.
Task `bb249c224` prepares fresh `.pi/audit/production-camcasp-native-df-v1` and runs
serial reference make with the prior pinned compiler/link flags; logs
`native-df-prepare-v1.log`, `native-df-build-v1.log`. Native capture build/run and
trace-on/off validation remain pending; no native integral/DF parity claimed.
Preparation task `bb249c224` failed BEFORE copying/building: `use precision` occurred
twice in real df_Smat, so the exactly-one-anchor guard correctly rejected it. Changed
the import insertion anchor to the unique `module df_Smat` declaration and added a
regression with routine-local precision imports. Failed v1 log is unchanged.
Task `baba5db02` runs updated harness tests, prepares fresh
`.pi/audit/production-camcasp-native-df-v2`, then serial-builds it; new logs
`native-df-harness-v2-tests.log`, `native-df-prepare-v2.log`, `native-df-build-v2.log`.
Task `baba5db02` stopped at tests: substring `module df_Smat` also matched END MODULE.
The import anchor now uses a line-anchored module-declaration regex with exact-count
validation. **8 tests passed in 1.20 s**, and real-source preparation succeeded
(`native-df-harness-v3-tests.log`, `native-df-prepare-v3.log`). Task `b7fe3fc1f` is
serial-compiling fresh `.pi/audit/production-camcasp-native-df-v3`
(`native-df-build-v3.log`). Both earlier failures occurred before a Fortran build.

Input inspection found the archived controller job has TWO DF blocks: lambda=1000
then lambda=0. The native observer intentionally supports a single constrained fit,
so its first fixture must be a clearly labeled DF-only prefix ending after the first
block, not an altered claim about the complete archived controller job. Preserve all
basis/MO/geometry settings; omit the later unconstrained fit and ISA only in fresh
DF-only inputs. Compare captured Drho-C against the existing full-job export separately.
Original full-job inputs and captures remain unchanged.
Native reference compile task `b7fe3fc1f` PASSED. Task `b52c76e5b` now runs traced
and untraced DF-only prefixes in fresh `.pi/audit/native-df-v3-water-{traced,untraced}`.
Runner `.pi/audit/run-native-df-reference.py` records executable/compiler/library/input
provenance, requires all three capture footers, compares every TMP artifact across
trace modes, and checks TMP_DrhoCa byte identity against the full-job capture.
Logs/report: `native-df-v3-reference-runs.log`, `native-df-v3-trace-comparison.json`.

Added strict `oracle/replay_native_df.py`: reads effective MAIN/AUX descriptors, C,
assumed occupations, q/RHS/Drho and raw/constrained metrics; validates dimensions,
route/solver/penalty, A=J+lambda*q*qT, symmetry and solve residual; separately measures
NumPy coefficient replay at scaled1e-9 and condition number. Reports finite-penalty
electron error without rescaling or pretending an exact charge constraint. This is
exported-input replay, not native Libint2 integral generation. Synthetic valid/malformed
and equation-mismatch task `bd91b71b2` PASSED **15 tests in 1.22 s**, Python
compilation and diff checks (`native-df-reader-tests.log`). This is the earlier
single-record/schema1 validation result, superseded by the separately measured
18-test multi-record/schema2 suite below.

Task `b52c76e5b` stopped with observer exit91: the real reference legitimately
constructs J twice before density entry. V3 failed-run evidence is preserved. The
observer now records every occurrence in exclusively created numbered metric files;
state schema2 records J/A counts. Reader requires a complete sequence and exact
agreement of repeated arrays/controls, rather than overwriting or choosing one.
Added complete/changed/missing-repeat tests: **18 passed in 1.27 s**
(`native-df-v4-tests.log`); fresh v4 preparation succeeded (`native-df-prepare-v4.log`).
Task `ba245070c` PASSED the serial build of `.pi/audit/production-camcasp-native-df-v4`
(`native-df-build-v4.log`). Updated fresh-run script
`.pi/audit/run-native-df-reference-v4.py` is now running as `b13b1e57e`
(`native-df-v4-reference-runs.log`). It validates numbered records, records its own
and reader hashes, and compares traced/untraced TMP artifacts plus Drho-C against
the preserved full-job output. Task `b13b1e57e` PASSED: all **17 TMP artifacts**
are byte-identical between trace modes, and captured-job TMP_DrhoCa is byte-identical
to the preserved full-job Drho-C. Evidence: `native-df-v4-trace-comparison.json`.
Task `bbd9f2392` now validates repeated metrics and replays the exported DF equations
with NumPy, writing `native-df-v4-water-traced/df-equation-replay.{json,log}`.
Task `bbd9f2392` PASSED equation/NumPy replay: **92 spherical MAIN, 246 Cartesian AUX,
5 occupied**; both J records identical; NumPy coefficients exactly equal reference;
solve residual **6.481634804853118e-18**. A symmetry and A-(J+lambda*q*qT) scaled error
**1.3871023723249596e-16** (absolute **5.820766091346741e-11**). Condition(A) is
**6.026362161121845e15**; fitted electron count **10.000000015583097**, consistent with
finite-penalty rather than exact normalization. Screening cutoff **1e-12**; reference
dummy-s exponent **1e-18**. Do not silently symmetrize the slightly asymmetric A.
Portable evidence: `tests/pytests/data_isapol/camcasp_native_df_export_evidence.json`.
Task #15 capture gate is complete; task #16 begins explicit Libint2 q/J/three-centre
integrals, keeping raw-tail/native/end-to-end claims separate. Read-only API audit
`da978353852c0ea780f9daf15ba55bd77` checks exact local Libint2 raw-shell/engine/order
interfaces. Full ten-file regression task `b7e34447d` PASSED **379 tests in 2.60 s**,
Python compilation and diff checks (`native-df-complete-tests.log`). This baseline
predates the new native AUX C++ adapter and does not validate that pending build.

### Native explicit AUX Coulomb integrals (in progress)

API audit `da978353852c0ea780f9daf15ba55bd77` completed (verified SHA256
`8ca493b772bae75b563a1c4efe897145771d1205e52e8d9496ef22313b06b490`). Verified local
Libint2 raw-coefficient Shell fourth argument=false; standard Cartesian normalization
has no mixed-component factors; `libint2::INT_CARTINDEX` is a configuration-aware
inline function (not the stale-documented macro). Construct BraKet xs_xs/xs_xx at
engine creation, use true Shell::unit(), precision0, and do not own global init or SH
ordering. Psi4 runtime SH order is Gaussian despite Libint build default Standard;
Cartesian-only engines avoid that ambiguity.

Implemented `aux_coulomb.{h,cc}` / `IsaAuxCoulomb`: initial Cartesian molecular AUX
S-G scope, analytic charges including even higher Cartesian components, and native
Libint2 two-centre J. Uses existing canonical GAMINT power table, configured native
indices and one explicit angular factor per component. Shell coefficients are not
renormalized; no screening beyond engine precision0 and no fake finite-exponent unit.
Separate non-unity TU and explicit Libint2::cxx dependency; Python binding added.
Task `bfdc47953` PASSED compilation, staging and built/staged byte comparison
(`native-aux-{compile,install}.log`). Task `bacfe181e` now runs the expanded eleven-file
suite, then production q/J comparison and diff checks (`native-aux-tests.log`,
`native-aux-metric-comparison.{json,log}`). Task `bacfe181e` PASSED **388 tests in
2.69 s**, diff checks and production native q/J checks. q max abs **7.105427357601002e-15**,
scaled **3.469064650290015e-16**; J max abs **1.7905676941154525e-12**, scaled
**1.597559351988609e-14**; largest angular-block scaled error **4.932962093530418e-14**.
These are actual native Libint2 AUX integrals from supplied explicit basis inputs.
Added `test_isapol_coulomb.py` with analytic contracted/displaced s-s integrals,
d-s mapping from independent Boys quadrature, mixed-component scaling, S-G Hermite
charge quadrature, translation/ownership checks and unsupported-role guards.
Prepared `.pi/audit/measure-native-aux.py` compares q/J and angular blocks against
production export after build/tests, with passing measurements recorded above.
Three-centre/Drho-C generation acceptance was not included in that q/J result.

Extended the provider with `three_center` and `closed_shell_rhs` in
`orbital_coulomb.cc`. Explicit new Orbital role is DALTON spherical only (Cartesian
MAIN fails closed). Cartesian-only Libint2 xs_xx engines use a symbolic independent
regular-harmonic polynomial transform for MAIN; no global SH ordering changes or
extra mixed-component normalization. B rows are AUX, columns mu*nmain+nu; occupied
trace is 2 sum C_i^T B_k C_i before any solve, with no penalty or AO-density sampling.
Atomic overlap rejects Orbital role. Added analytic contracted s-s-s and trace tests,
plus independent Coulomb-potential Hermite quadrature for P-G MAIN with S cross blocks
to catch normalization, order and phase errors. Task `be28ef155` PASSED compilation,
staging and built/staged byte comparison (`native-three-center-{compile,install}.log`).
Task `b8897127c` now runs all eleven test files, production RHS comparison and diff
checks (`native-three-center-tests.log`, `native-three-center-rhs-comparison.{json,log}`). Prepared production
`.pi/audit/measure-native-rhs.py` compares native raw and constrained RHS against the
export (raw reference RHS must be reconstructed by subtracting its penalty).
Task `b8897127c` PASSED **394 tests in 2.82 s**, production RHS comparison and diff
checks. Raw occupied-trace RHS versus reconstructed reference: max abs
**3.33244543071487e-11**, scaled **4.599730600496091e-13**. Constrained RHS max abs
**8.731149137020111e-11**, scaled **4.261279366058036e-16**. Evaluation took 0.075 s.
Portable q/J/RHS evidence: `tests/pytests/data_isapol/camcasp_native_integral_evidence.json`.
Task #16 integral generation is complete; native density solve task #17 is in progress.

Added `fit_drho_c` / `IsaDrhoCResult`: native q/J/B and supplied occupied C, positive
finite charge penalty, per-occupied diagonal penalty before tracing, explicit
column-major LU without refinement, no symmetrization/regularization/rescaling.
Result exposes original J, constrained A, q/raw/RHS/coefficients, fitted electrons and
infinity-norm backward residual (small residual does not establish forward accuracy).
Task `b1b7df838` PASSED compilation, staging and built/staged byte comparison
(`native-drho-{compile,install}.log`). Task `ba737d849` now runs all eleven test files,
then production native-density comparison and diff checks (`native-drho-tests.log`,
`native-drho-comparison.{json,log}`). Task `ba737d849` PASSED **402 tests in 2.95 s**,
but FAILED production native-density parity (the chained diff check did not run).
A scaled error **6.935511861624798e-16**, RHS **4.261279366058036e-16**, yet coefficient
max abs **0.04458958227804288** / scaled **0.0012802188514176112**. Native condition(A)
**6.331740426980903e15**; backward residual **3.4261594590164743e-18**. Charge error
relative to reference **8.840927989695047e-12**. Over 68,310 points density max abs
**1.4250481785893498e-4**, global scaled **6.130116422102165e-7**, pointwise scaled
**4.664438903259456e-6**, relative abs-weighted L2 **7.569795013515608e-7**. Both
coefficient and density gates FAIL at unchanged 1e-9. Saved native coefficients:
`native-drho-coefficients.json`; no rescaling, regularization or symmetrization.
New blocker #18 tracks this separately from legacy tail blocker #14. Diagnostic task
`b416a0bda` performs reference/native A/RHS hybrid solves with a common basis sampler,
checks exact native rerun and C++/NumPy solve identity, and compares condition(J)
versus condition(A). Outputs `native-drho-conditioning.{json,log}`; diagnostic only.
Separate post-failure `git diff --check` and Python compilation of both measurement/
diagnostic scripts passed.

Task `b416a0bda` completed the hybrid diagnostic. Native rerun is bitwise identical;
C++ and NumPy native solves agree exactly; reference-input NumPy solve agrees exactly
with reference coefficients. Condition(J) **1.9351470110863735e12**, condition(A)
**6.026362161121845e15**. Native/reference A Frobenius difference **1.629527278089118e-9**,
RHS L2 difference **1.9460101090620136e-10**. Holding reference RHS but substituting
native A gives density max abs **2.1498961028628555e-4**, local scaled **5.92430892997544e-6**;
holding reference A but substituting native RHS gives **7.254867412191147e-5** / local
scaled **1.2703308643032332e-6**. Both substitutions recover the measured native result.
Thus both input differences matter, predominantly the metric; this is not a C++ versus
NumPy solve-wrapper disagreement. No changed solver/integral/tolerance was tested.
Source recheck `df_Smat.F90:544–547` confirms the implemented left-associated
lambda*q(row)*q(column) ordering; no transposition defect found there.
Native-density and legacy raw-tail parity remain open. Any stabilized solver/reference
track must be explicitly authorized/labeled and cannot retroactively pass these gates.
The user explicitly chose **Continue pinned response work**: preserve the method,
tolerances and both blockers; no stabilized comparison track or reference switch
was authorized. Task #17 is pending/blocked by #18; independent task #19 is active.
Read-only delegate `db278d7f70bb5123dc314c0d9dd9cdc95` audits reuse/extraction of
`FDDS_Dispersion` and `df_fdds_dispersion` for single-system/per-frequency response,
representation mapping and hybrid/nonhybrid regression coverage. No parallel FDDS
stack, dummy partner or full-SAPT prerequisite is acceptable (SPEC section5).

Independent archive input inspection for response provenance: original H2O-isagrid
`H2O.cks` specifies SCF-code PSI4/PBE0, Kernel ALDA/C-DF, CKS internal constrained
DF Hessians, kernel FD controls rho-eps1e-8/fmax600/delta0.01/alpha1/cutoff1e-8;
point response NN eta0/lambda1, subsequent distributed response NN eta0.0005/lambda1,
then an OO eta0/lambda100 Doo-C/BVLS ISA block. Quad10 beta0.5, rank2 total point
response and rank4 ISA-GRID distributed response; random500 despite p2000 names.
These are NOT the adapted Drho-C/LU/lambda1000 density fixture's policies. Preserve
separate explicit provenance rather than silently reuse that density fit as the
response transition fit. The archive is not certified as a current pinned-source run.

Delegate `db278d7f70bb5123dc314c0d9dd9cdc95` retrieved and verified (answer SHA256
c2d98db259c97e343ad68d2f0ae3998f01856227d3cb6600749dc9fea2ad7cdb). Implemented the
smallest shared extraction in `psi4/driver/procrouting/sapt/fdds_response.py`: supplied-
intermediate, single-frequency auxiliary response, explicit hybrid intermediates,
owned raw/symmetrized outputs. Moved duplicated A/B solves from `sapt_mp2_terms.py`
into this helper; preserved dot order, rcond1e-13, R sanitation and pair quadrature.
Old driver saved as `.pi/audit/fdds-pre-extraction-sapt_mp2_terms.py`. No native monomer
construction or CamCASP response parity claim. C++ remains paired and its QR requires
nov>=naux; subsequent factoring must diagnose this rather than invent rank policy.

Task `b6bcb5088` installs/byte-compares Python modules, runs `test_fdds_response.py`,
then unchanged sapt-dft1/dft2/dft-api/dft-lrc input fixtures in fresh retained directories
`.pi/audit/fdds-regressions-v1/`. Launcher/output logs and input hashes are retained;
no fixture values/tolerances modified. Focused tests include independent small MO-space
nonhybrid solves, scalar signs/frequency limits, noncommuting hybrid matrices,
raw reciprocity defects, ownership and inherited pseudoinverse policy. Validation is
pending; see `fdds-extraction-{install,tests,regressions}-v1.log`.
Also prepared `test_fdds_pair.py` (not in that running command): synthetic hybrid/
nonhybrid wiring checks ensure both monomers call the shared signed-response helper
at every quadrature node and retain the pair energy prefactor. Run this with the
full libisapol regression suite after the current validation finishes.
Analytic finite-penalty tests include signed AUX coefficients, fitted-density sampling
and invalid penalties. Prepared `.pi/audit/measure-native-drho.py` separately reports
coefficient, charge, 68,310-point density, local scaled density and weighted-L2 errors,
with no automatic acceptance from a small solve residual. Native SCF/basis recipe,
native-density ISA trajectory and end-to-end properties remain absent. Independent critique delegate
`d4654b669a8563c4de3ab54db1dc3183d` completed its independent critique (verified
SHA256 `37e0a17b317a125c98f139c1a13abdd3269de2a225bf03bd7651fea4ffdcf465`). No
controller ordering defect was identified; strict raw-tail agreement is not demonstrated
robust at the legacy 1e-8 FD step, but impossibility is not established. Do not keep
rearranging arithmetic to chase a lucky passing trajectory. Complete current measurement
once, refresh conditioning on latest inputs, and proceed with independent native DF
work while preserving the raw-tail failure.
The review found real validator omissions: iteration match was not in `passed`, and
final per-atom flags/deltas/MaxDelta/cutoffs were not checked. New separate public
`oracle/compare_isa_trajectory.py` adds these, history consistency, fixed-cutoff endpoint
and config checks, plus individually reported A/b errors without replacing the legacy
joint-vector denominator. Existing in-flight comparator and old reports are unchanged.
`test_isapol_trajectory.py` adds negative synthetic tests; task `bb9114789` PASSED
**361 tests in 2.50 s** across the expanded nine-file suite, plus Python compilation
and diff checks (`strict-trajectory-tests.log`). It deliberately does NOT install over the staged
extension while the pre-fix full trajectory is running (build/stage have distinct
inodes). Preserve that baseline trajectory before installing/retesting the fix. Final trajectory
comparison helper `.pi/audit/compare-controller-trajectory.py` is ready to compare
against the captured iteration-53 committed state once the C++ trajectory finishes.

New `oracle/run_isa_controller.py` runs the full C++ trajectory from the initial
exported bases/density/grids, with no intermediate shape/tail injection after entry.
Pre-fix task `be49109dd` COMPLETED: converged in **53 iterations**, matching reference,
in **237.504540937 s**. Evidence `.pi/audit/cpp-controller-trajectory.{json,log}`.
Final comparison `.pi/audit/cpp-controller-final-comparison.{json,log}` FAILED only
tail parameters at unchanged scaled 1e-9 (max abs O tail **1.6219057652477886e-7**).
All final D/W/shape charges and next controls passed; max D/W abs **3.4880942578752183e-10**,
max shape-charge abs **1.6755503651211257e-9** (scaled **1.9023635459080004e-10**).
Post-run, pre-replacement hashes are saved in `cpp-controller-baseline-provenance.json`.
Future runner reports now hash the entry extension/runner/replay/sidecar before
computation and refuse to overwrite an existing report; all reruns must use fresh paths.
It reports convergence and history separately from final-reference agreement; no
native basis/SCF/DF or end-to-end property claim. The replay helper was refactored
to share only input assembly (`prepare_replay`) between one-step and full runs. Read-only controller source audit delegate `d5d4c7f3518538154b14537847cbd2dad`
completed against the preserved legacy source. Verified result artifacts are under
`.pi/delegate/01a07228-a554-7f85-8b2e-e4c5136ec050-3719637/d5d4c7f3518538154b14537847cbd2dad`.
Key controller contracts: fit/project/DIIS/convergence BEFORE mixing; update next
activation with <= thresholds; tails switch on when iteration>20 (next sweep), not
>=20; analyze tails from OLD w0 before committing all mixed w and D; convergence
strictly delta<eps with no forced extra active sweep. DIIS is absent in the target
preset and will fail explicitly until separately implemented. Legacy A is initialized
and saved across calls, not uninitialized; its stale sign gates new tail slopes,
while undefined IP really is uninitialized. Deterministic C++ tails omit the stale-A
gate and report undefined fits. Whole-sweep capture is required for production
controller parity; selected-atom v2 cannot establish it.

## Deferred reference transition — CamCASP Libint2 branch

User requests eventually switching the reference code to the CamCASP branch using
Libint2 and ensuring Libint2 is used where applicable. **User is finding that branch
independently and explicitly authorized continuing the current implementation.** Preserve
old master checkout, archives, captures and evidence as separately identified legacy
references. Prefer a new pinned worktree, not an in-place reference/archive rewrite.
Local `/home/awallace43/gits/CamCASP` is clean on master `63b16a2`. None of its
currently cached remote branches contains a Libint integration (only two branches
contain the word in saved Psi4 output). Remote head discovery task `b31d09ac8` is
complete; QMUL origin advertises only the same cached branch heads and no Libint2
branch. Web discovery identified the separate public distribution
`https://gitlab.com/anthonyjs/camcasp.git`; branch listing task `be203ae92` completed.
Public heads are master plus CamCASP-6.0/6.1-linux/macos; none is named for Libint2.
Public branch contents have not been audited. User clarified the branch should
be under `~/gits/CamCASP`. Verified HOME resolves to `/home/awallace43`; that exact
repo has only `refs/heads/master`, no other local refs, no Libint commit-message
matches, and only initial-clone reflog entries. Nearby `camcasp_psi4/camcasp-bin`
also has only master (binary distribution); `.worktrees/camcasp-code` is a Psi4
worktree, not the requested CamCASP branch. The intended fork/branch is absent
from the inspected checkout/advertised origin; exact branch name, commit or fetch
URL is needed before a truthful switch. No branch was fabricated or substituted. Listings:
`.pi/audit/camcasp-{remote,public}-heads.txt`. No reference checkout was switched.

Verified Psi4: staged core links `/home/awallace43/miniconda3/envs/p4_ci/lib/libint2.so`,
CMake `Libint2_DIR` points to that environment, installed version **2.13.1**, and
runtime `INTEGRAL_PACKAGE=LIBINT2`. The current libisapol explicit-basis sampler
and co-centred weighted analytic overlap are independent arithmetic, not Libint2
calls; do not mislabel them. Audit the new reference branch's scope before deciding
which integral boundaries require a Libint2 adapter (especially normalized component
conventions and W-Eps weighting).

## Active increment — typed provider-to-frozen-fit assembly and full-stream replay

Completed: immutable typed fit provider owns primitive co-centred AtomAux and
fixed molecular-AUX density; assembly derives per-function metadata, raw distances,
analytic overlap and basis/density samples in C++. Caller supplies quadrature,
old full coefficients, screened/tail-processed shape and shape-sum samples and an
explicit active density-neighbour list. A direct fit method uses the same options
for assembly and solve. No native density generation or tail reconstruction.
Extend opt-in v2 replay with a separately labelled provider reconstruction mode;
run all three existing full streams, keeping original streams/reports untouched.
Implementation and synthetic tests are written; staged build `b3ca9019f` compiled
and installed successfully (log `.pi/audit/fit-provider-build.log`), with the same
unrelated optional `stubgen`/cleanup exit 1. Regression task `b518f0d61` passed
**260 tests in 2.19 s** (17 added over 243; log `.pi/audit/fit-provider-tests.log`).
Staged extension import/byte comparison, runtime LIBINT2 selection, Python
compilation and diff checks passed. Full provider replay resumed under task
`b5b0f5bde` for first O, activated O and activated H1, writing new
`.pi/audit/basis-water-{first,activated,hydrogen}/provider-replay.{json,log}`.
Old captures/replay reports are untouched. Full replay task `b5b0f5bde` passed
all three cases, each **68,310 points**, atomic sizes **109/109/49**:
- first O: max D/raw-shape error **5.889178034124143e-12**, scaled
  **9.067984591939813e-13**, normalized residual **3.568388448276264e-17**;
- activated O: max D/raw-shape error **6.4290794909993565e-12**, scaled
  **9.348732792347371e-13**, residual **2.2062003729832137e-17**;
- activated H1: max D/raw-shape error **9.216932772559971e-14**, residual
  **3.9900174571495284e-17**.
Across all points/cases: max basis error **7.105427357601002e-15**, density
**5.684341886080802e-14**, weighted overlap **1.7763568394002505e-15**, RHS
**3.552713678800501e-15**, population **8.348877145181177e-14**. Primitive
metadata and raw squared distances had zero observed discrepancy.
The larger solved-D differences versus sample-only replay follow reconstruction
rounding at the metric/RHS boundary; no tolerance was relaxed (CLI scaled 1e-9).
This validates full exported-input reconstruction with supplied shapes, not native
basis/density/tails/controller generation.

Bounded portable evidence: `tests/pytests/data_isapol/camcasp_isa_provider_evidence.json`
contains full per-case errors, stream/report/source/extension hashes and reference
track limitations. `.gitignore` explicitly preserves its visibility. Added one
analytic v2 replay-success test after the 260-test run; final regression task
`b11f9fadd` passed **261 tests in 2.13 s**
(`.pi/audit/fit-provider-final-tests.log`). Python compilation and diff checks passed.
No unresolved implementation/test blocker. This bounded increment is complete;
the Libint2 reference transition remains deferred by user request.

Exact current commands (p4_ci PATH, staged PYTHONPATH, all three thread variables=1):
```bash
bash build.sh > .pi/audit/fit-provider-build.log 2>&1
python -P -m pytest tests/pytests/test_isapol.py tests/pytests/test_isapol_fit.py tests/pytests/test_isapol_checkpoint.py tests/pytests/test_isapol_basis.py -q
for case in first activated hydrogen; do
  python -P tests/pytests/data_isapol/oracle/replay_isa_checkpoint.py \
    .pi/audit/basis-water-$case/isapol-checkpoint.dat --reconstruct-providers \
    --report .pi/audit/basis-water-$case/provider-replay.json
done
python -m py_compile tests/pytests/test_isapol_basis.py tests/pytests/test_isapol_checkpoint.py tests/pytests/data_isapol/oracle/replay_isa_checkpoint.py
git diff --check
```
Immediate next task: explicit no-tail synchronous sweep state/assembly using typed
providers; keep activation, mixing and active-tail policies separately testable.
Production intermediate activation/phase captures and the user-provided future
Libint2 reference branch remain separate reference work.

## Previous increment — analytic atomic overlap and explicit shape projection

Completed: extended the C++ exported-input basis with co-centred analytic overlap
(W-Eps, s-block-only or mathematical all-block), and a validated explicit zero-based
shape-shell map projecting raw full AtomAux coefficients without mixing or tails.
Use all three complete portable atomic metrics and raw shape vectors, independent
polynomial overlap tests and malformed/integrability cases. No native generation,
full-grid reconstruction or controller is claimed. Preserve existing fixtures and
reference archives.

Implementation is in `explicit_basis.{h,cc}` and `export_isapol.cc`:
- `overlap()` rejects molecular AUX, distinct used centres, invalid W-Eps and
  nonintegrable primitive pairs; Cartesian Gaussian moments and spherical angular
  orthogonality use exported effective coefficients without normalization.
- `IsaShapeMap` checks role/dimension, unique target s shells and exact centre,
  ordered exponents/effective coefficients. It owns function-column indices and
  projects raw vectors without DIIS/mixing/clipping/tails; subsets/permutations
  and noncontiguous atomic s blocks are explicit.
- Existing independent Python oracle and JSON fixtures are unchanged.

Build task `bd08b0829` compiled/linked/installed successfully in p4_ci;
`.pi/audit/atomic-overlap-build.log`. The script exit 1 is only the inherited
missing `stubgen` and unmatched cleanup glob. Regression/measurement task
`b40142b89` completed: **243 passed in 2.14 s** (34 new tests over 209).
Staged extension import and byte comparison with build output passed. Python
compilation and `git diff --check` passed; no unresolved test/implementation blocker.

Measured complete metric errors (absolute / globally scaled):
- first O, 109 functions: **6.661338147750939e-16 / 6.661338147750933e-16**;
- activated O, 109 functions: **1.7763568394002505e-15 / 3.2155493553843693e-16**;
- activated H1, 49 functions: **8.007583902037866e-16 / 4.293558663801019e-16**.
Raw shape projection and metric symmetry errors were **zero observed** for all
three cases. Largest metric difference versus the independent polynomial oracle
was **3.552713678800501e-15**; ordinary diagonal normalization error was at most
**6.661338147750939e-16**. These are complete exported-input atomic metric checks,
not a new full-grid replay or native density/basis generation. Synthetic contracted
S–G cases validate the mathematical all-block option, not the anomalous CamCASP
all-block source implementation. The frozen fitter remains primitive-only.

Exact validation commands (from worktree root):
```bash
export PATH=/home/awallace43/miniconda3/envs/p4_ci/bin:$PATH
bash build.sh > .pi/audit/atomic-overlap-build.log 2>&1
export PYTHONPATH="$PWD/build_camcasp_psi4_joint/stage/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -P -c 'import psi4; print(psi4.core.__file__)'
cmp build_camcasp_psi4_joint/psi4-core-prefix/src/psi4-core-build/src/core.cpython-313-x86_64-linux-gnu.so build_camcasp_psi4_joint/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so
python -P -m pytest tests/pytests/test_isapol.py tests/pytests/test_isapol_fit.py tests/pytests/test_isapol_checkpoint.py tests/pytests/test_isapol_basis.py -q
python -P .pi/audit/measure-atomic-overlap.py
python -m py_compile tests/pytests/test_isapol_basis.py .pi/audit/measure-atomic-overlap.py
git diff --check
```
Test log: `.pi/audit/atomic-overlap-tests.log`; measurement report:
`.pi/audit/atomic-overlap-errors.json` (fixture/source hashes and per-case errors).
Immediate next task: assemble frozen-fit data from typed
providers, preserving primitive-only fitter metadata and explicit screened/tail
samples; use full local streams for production replay, not the 97-point subsets.

## Previous increment — typed C++ exported-input sampling provider

Implemented `explicit_basis.{h,cc}` and bindings: immutable owned typed contracted
basis (S–G, GAMINT/DALTON), distinct molecular AUX/AtomAux/s-only shape roles,
and signed fixed molecular-AUX density evaluation. Effective coefficients are not
renormalized. Spherical evaluation uses an independent solid-harmonic recurrence;
the Python polynomial oracle and production JSON fixtures remain unchanged.
Neighbour lists are explicit unique zero-based active sites (empty screens all);
the test adapter converts the fixture's one-based positive entries, discarding padding.
Tests cover production samples, contracted S–G shells in both representations,
screening, translation, roles, ownership and malformed inputs.

Build succeeded through linking/install; `bash build.sh` returned 1 only for inherited
missing optional `stubgen` and unmatched `psi.*` cleanup. Build log:
`.pi/audit/explicit-basis-build.log` (task `b7e4562d8`). Staged core was byte-compared
with the built extension and imported from the staged path. Regression task
`b3853b19e`: **21 failed, 188 passed in 2.15 s**. All failures were inherited
fitter exception types: in the unity build the new `require(bool,const char*)`
overload intercepted fitter literal messages that previously selected its
`require(bool,const std::string&)`. Renamed the new helper `basis_require` to
avoid altering existing behavior; existing tests are unchanged. Rebuild `bc3bba67b`
compiled/installed successfully (same unrelated script exit 1). Final staged run
`bdace5feb`: **209 passed in 1.93 s**, up from the 185-test baseline; log
`.pi/audit/explicit-basis-retests.log`. Byte comparison and staged import checks
passed again. Measurements rerun unchanged; Python compilation and `git diff --check`
passed. No unresolved implementation/test blocker; build-script optional tooling
failures remain unchanged. This increment is complete.

Measured on 97 points for each of first O, activated O and activated H1:
maximum atomic-sample absolute error **7.105427357601002e-15**, density error
**5.684341886080802e-14**; maximum globally scaled density error
**2.4472522064008934e-16**. All molecular columns versus the independent polynomial
oracle had zero observed difference. Evidence: `.pi/audit/explicit-basis-errors.json`;
reproduction script `.pi/audit/measure-explicit-basis.py`.

This bounded increment does not include analytic overlap, native basis/DF generation,
active tails or controller/end-to-end parity. Fixed-density identity/provenance is
caller-supplied; the class cannot certify arbitrary coefficients as Drho-C. Sampling
supports contractions but the frozen fitter remains primitive-only. Evaluation
materializes the sampled basis; batching and analytic metric construction remain
future work. Immediate next task: typed co-centred weighted atomic overlap plus
explicit shape-map/frozen-fit assembly validated against the full portable metrics.

### Exact current validation commands

```bash
export PATH=/home/awallace43/miniconda3/envs/p4_ci/bin:$PATH
bash build.sh > .pi/audit/explicit-basis-build.log 2>&1
# After unity-helper fix:
bash build.sh > .pi/audit/explicit-basis-rebuild.log 2>&1
export PYTHONPATH="$PWD/build_camcasp_psi4_joint/stage/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -P -c 'import psi4; print(psi4.core.__file__)'
cmp build_camcasp_psi4_joint/psi4-core-prefix/src/psi4-core-build/src/core.cpython-313-x86_64-linux-gnu.so build_camcasp_psi4_joint/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so
python -P -m pytest tests/pytests/test_isapol.py tests/pytests/test_isapol_fit.py tests/pytests/test_isapol_checkpoint.py tests/pytests/test_isapol_basis.py -q
python -P .pi/audit/measure-explicit-basis.py
python -m py_compile tests/pytests/test_isapol_basis.py
git diff --check
```

## Previous increment — descriptor export and reconstruction

**This previous increment is complete.** Production audits passed and final fixture suite `baccb83da` passed **185 tests in 1.85 s**. Python compilation and `git diff --check` also passed. All current background tasks are finished.

- Capture format **v2** exports complete atomic/molecular/shape basis descriptors, actual Drho-C coefficients and density neighbour storage, old shape coefficients, explicit s-shell map, and raw new shape coefficients **before DIIS/mixing**. Reader remains v1-compatible. New complete-stream marker follows shape extraction.
- `oracle/reconstruct_isa_basis.py` independently derives real harmonics from differentiated Legendre polynomials, handles GAMINT Cartesian factors/order and contracted radial sums, reconstructs sampled molecular density with neighbour screening, analytic co-centred overlap, radial normalization and raw shape projection. **No renormalization of exported effective coefficients.** Scope S–G; active exponential-tail samples are explicitly excluded.
- Source audit confirmed DALTON spherical p=x,y,z, no Condon–Shortley phase, and that the live density evaluation opens without releasing Rho%D; capture reads immediately afterward without extra evaluation.
- Portable fixtures: `tests/pytests/data_isapol/camcasp_isa_basis_{first,activated,hydrogen}.json` (about 1.25 MB total), companion `camcasp_isa_basis_evidence.json`, and `test_isapol_basis.py` production regressions. Fixtures contain complete descriptors/atomic metric plus **97 selected points each**, not full frozen-fit replay inputs. Generation selects samples only after a successful audit. `.gitignore` exceptions keep all four JSON files visible.
- Full v2 replay and independent audit passed for oxygen call 1, activated oxygen call 157, activated H1 call 158. Full captures contain 68,310 points; atomic dimensions **109 (O), 49 (H1)**; molecular density basis **246 functions / 56 primitive shells**.
- Maximum reconstruction absolute errors over the three 97-point audits: basis samples **7.10543e-15**, density **1.13687e-13**, atomic metric **5.32907e-15**, shell normalization **4.44089e-16**. Shape-map basis consistency and raw s-projection errors were **zero observed**. Maximum scaled reconstruction error was below **1e-15**.
- New H1 full C++ replay: max RHS error 5.55112e-17, D error 2.21993e-14, population error 1.07692e-14, residual **7.98003e-17**. Oxygen replay metrics match the previous increment's observations.
- Seven final shape/tail files remain byte-identical across all three v2 runs and the previous untraced run. Evidence: `.pi/audit/basis-trace-comparison.json`.
- Pre-fixture suite passed **182 tests in 1.76 s**. Final suite adds three portable production cases: **185 passed in 1.85 s**, with Python compilation and diff check clean. Evidence is finalized in `camcasp_isa_basis_evidence.json`.

### Current artifact / reproduction pointers

- Build `.pi/audit/production-camcasp-v2` succeeded (`bbf1b67b6`); old successful build is untouched.
- Fresh capture directories: `.pi/audit/basis-water-{first,activated,hydrogen}`. Each has complete v2 stream, `replay.json`, `basis-audit.json`, expanded input, producer/protocol provenance and result files.
- Successful capture task `be2805c1f`; successful three-case replay/audit task `ba8c97b57`.
- Read `oracle/BASIS_RECONSTRUCTION.md` for formulas, v2 layout and fixture regeneration; producer/build workflow remains in `oracle/PRODUCTION_CHECKPOINT.md`.
- Resolved format pitfalls: density neighbour list is allocated length **400**, positive active prefix followed by zeros; shape map is allocated length **30**, with active count from shape basis size (12 for O). Reader preserves raw storage, checks padding and rejects malformed indices. Do not treat allocation length as active dimension.
- No new Psi4 C++ implementation in this increment: the reconstruction tool is an **independent development oracle**, not a native basis/DF provider.

## Previous increment — validated production frozen-update tooling

The previous increment is complete. **Gate 3 is advanced, not fully closed.** Actual CamCASP-produced Drho-C/AtomAux samples now replay through the existing C++ `isa_a_fit_step`. Native Psi4 basis/DF construction and the ISA controller are still absent.

### Changes in this increment

- `tests/pytests/data_isapol/oracle/capture_isa_checkpoint.py`: opt-in instrumentation of a **new scratch source copy**. Captures actual screened density/shape/basis samples, primitive shell coefficients/order, active settings, old D, weighted overlap before ridge/damping, modified metric, pre-solve RHS, population and solved D. Requires unperturbed Drho-C, A, LU and uncontracted atomic shells. All source anchors fail closed. No upstream numerical equations are replaced.
- `.../prepare_isa_water_run.py`: preserves the archive and produces an explicitly adapted, fully expanded Drho-C/LU water input with geometry, settings and input/basis/orbital hashes.
- `.../replay_isa_checkpoint.py`: strict versioned stream reader, staged-C++ replay, precision-aware error report and nonzero exit on failed tolerance. Incomplete/malformed streams are errors.
- `tests/pytests/test_isapol_checkpoint.py`: **23 synthetic format/plumbing tests**, including malformed streams, replay, source-copy isolation and input adaptation. They are not molecular parity tests.
- `.../oracle/PRODUCTION_CHECKPOINT.md`: reproducible development workflow and limitations.
- `tests/pytests/data_isapol/camcasp_isa_production_evidence.json`: bounded measured evidence, source/tool/executable/input hashes and tracing comparison. Local `.gitignore` exception keeps this JSON visible despite the repository-wide ignore.
- Updated `SPEC.md` with archive corrections and measured production evidence.
- No Psi4 C++ code changed in this increment; tests used the inherited staged extension. The separately instrumented CamCASP reference was compiled successfully.

### Measured evidence

Protocol: adapted archive, **not modern ISA-Pol preset**. Oxygen, **68,310 points / 109 functions** per checkpoint.

| Checkpoint | Active W-Eps / ridge | Max absolute metric / RHS / D error | Population error | Normalized solve residual |
|---|---|---|---|---|
| Call 1, iteration 1 | 0 / 0 | 0 / 0 / 0 | 8.4377e-14 | 5.3526e-17 |
| Call 157, iteration 53 | 0.17 / 0.001 | 0 / 1.7764e-15 / 2.8910e-12 | 4.0856e-14 | 2.2062e-17 |

Activated coefficient error scaled by max(1, max(abs(reference D))) is **4.20393e-13**. Both replays passed the explicit CLI default scaled tolerance 1e-9; do not describe that default as an independently established scientific tolerance. Population summation grouping differs between site-wise CamCASP accumulation and flattened C++ input.

CamCASP converged in **53 iterations** in each run. Seven final shape/tail files (serialized `H2O_atoms.ISA` plus three `fn1-CONV` and three `exp-analysis` files) are **byte-identical** across first-update-traced, activated-traced and tracing-disabled runs. Untraced run has no checkpoint. This checks diagnostic perturbation of these reference outputs, not native Psi4 controller parity.

Final validation: **149 tests passed in 1.67 s**, Python compilation and `git diff --check` passed. Existing inherited tests remain intact.

## Next agent: immediate next milestone

1. Implement an **explicit no-tail synchronous sweep boundary** using typed providers, old shape coefficients and inspectable per-atom results. Provider-to-frozen-fit assembly and all three full exported-input replays now pass (see active increment above). Preserve old-state synchrony and distinguish no-tail clipping from active-tail policy. Native basis recipe/DF generation remains separate; retain molecular AUX versus AtomAux and neighbour screening. Do not substitute JKFIT/AO density or claim a controller from a single sweep.
2. Add intermediate activation-transition captures and explicit iteration/phase selection; current selectors are call numbers, and captured new shape is raw pre-DIIS/mixing. Extend to H2 or a rotation check. The sampled descriptor boundary is now validated for the adapted archive, but this is not the modern preset or independently generated native density parity.
3. Implement synchronous ISA-A iteration/activation/tails with inspectable state and restart tests. Active exponential-tail reconstruction is still absent from the independent audit; do not infer it from successful Gaussian basis checks or CamCASP convergence.
4. Continue dependency order: native DF/response, partitioned Q/alpha, localization/frames, PFIT, dispersion with rank coverage, then public driver/oeprop. Do not register unsupported public properties.

## Readable evidence and local artifacts

- Small portable evidence: `tests/pytests/data_isapol/camcasp_isa_production_evidence.json`.
- Instrumented reference: `.pi/audit/production-camcasp/`; executable `bin/camcasp`; `capture-provenance.json`.
- Successful first capture: `.pi/audit/production-water-v3/`.
- Successful activated capture: `.pi/audit/production-water-activated/`.
- Tracing disabled: `.pi/audit/production-water-untraced/`.
- Each capture directory contains `H2O-expanded.cks`, `run-provenance.json`, `producer-provenance.json`, `reference.log`, `isapol-checkpoint.dat`, `replay.json` and reference result files. Untraced directory intentionally has no checkpoint/replay.
- `.pi/audit/production-trace-comparison.json` records all compared file hashes.
- Large text streams and upstream source/binaries are **local development artifacts, not distributed fixtures**. Regenerate using `oracle/PRODUCTION_CHECKPOINT.md` if absent. Do not redistribute extracted upstream source/binaries.
- Previous-increment background tasks are finished. Useful logs: build `b49b880de`, first run `b24bef213`, first replay `b6bad492a`, activated/untraced runs `b77dbc19e`, activated replay `b91db070d`, final tests `b47ed98b2` under `.pi/tasks/session-3719637-3719637/`.

## Reference provenance / resolved pitfalls

- Source: `/home/awallace43/gits/CamCASP` (no prebuilt executable; source version metadata differs from archived runtime).
- Archive: `/home/awallace43/gits/camcasp_psi4/.camcasp-reference/work/H2O-isagrid`.
- Archived runtime: `/home/awallace43/gits/camcasp_psi4/.camcasp-reference/tools/camcasp-runtime` (binary distribution, not interchangeable source provenance).
- Archive requests **Doo-C and BVLS**, not Drho-C/LU. It has Cartesian molecular aVTZ AUX, spherical aVTZ AtomAux+ISA set2, radial 100/angular request 200, and 500 fit points. It is not the modern preset. New run explicitly uses Drho-C/LU and NN lambda 1000/0, retaining archived orbitals/bases and removing response/GDMA work.
- Older source rejects `SET Num-Int-Pars`; adapted input omits it and records source-default numerical controls. Angular request 200 resolves to **230 actual** points.
- Older source rejects explicit `W-TAILS / FIX = ON`; its defaults are already ON (`stockholder.F90:85,890`). The preparer removes that ON directive and records why.
- Build **serially**: parallel Makefile races on duplicate compilation of `precision.f90`/`.mod`. Preparer copies required `bin/version.py`. Use `GIT_CEILING_DIRECTORIES` to avoid attributing the parent Psi4 revision to CamCASP. Initial build banners may contain that ambient revision; source/executable hashes are authoritative.
- Actual capture hooks: `make_Stilde_stockholder_A`, `Chi_rho_w_overlap`, `update_D_one_atom`. LU overwrites RHS, so capture before solve. Do not use weighted/capped `rA2` as raw squared distance.
- Source inspection found all-block W-Eps overlap behavior differs from intended mathematics; first production protocol is **s-block-only**. Do not certify the all-block source path from this evidence.

## Validation commands

From this worktree root:

```bash
export PATH=/home/awallace43/miniconda3/envs/p4_ci/bin:$PATH
export PYTHONPATH="$PWD/build_camcasp_psi4_joint/stage/lib"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -P -m pytest \
  tests/pytests/test_isapol.py tests/pytests/test_isapol_fit.py \
  tests/pytests/test_isapol_checkpoint.py tests/pytests/test_isapol_basis.py -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -P \
  tests/pytests/data_isapol/oracle/replay_isa_checkpoint.py \
  .pi/audit/production-water-activated/isapol-checkpoint.dat \
  --report .pi/audit/production-water-activated/replay.json
```

Do not append an empty PYTHONPATH entry. `python -P` avoids source-package shadowing.
For future C++ changes run `bash build.sh` in `p4_ci`, inspect compilation/install
and imported extension separately: inherited script still has unrelated optional
stubgen/cleanup failures. No changes to that script were made here.

## Inherited work preserved

Previous increment supplied grids/tables/fit points, C++ frozen ISA-A fitting and
bindings, 126 tests, diagnostic AO-density water/edge oracles. It did **not**
establish production Drho-C/AtomAux parity. Existing uncommitted CMake/core/cubature,
libisapol and tests were preserved; no global cleanup or staging was performed.
