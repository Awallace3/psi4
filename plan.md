# libisapol implementation handoff

**Start here:** read `psi4/src/psi4/libisapol/SPEC.md`, then this plan. The spec owns scientific/API contracts; this file owns execution state. Preserve unrelated uncommitted work. User authorized committing and continuing; checkpoint `f996942ad6` contains the validated native/performance/recoupled implementation. No pushes, resets or reference-tree edits.

## Accepted independent high-J followup

Parent1755 regressions pass24.71s (`.pi/audit/high-j-parent-regressions-v2.log`).
14 new tests independently validate all735 J9/C11 and J10/C12 rows,7 blocks,
26 ordered rank quadruples/11 reciprocal classes using factorial CG/electrostatics
and documented Sbar normalization, not production table expectations. Source review
found no scientific must-fix; parent checked cited dissertation eqs4.3–4.6 directly.
Added isolation-case completeness assertion and corrected static-weight wording.
No production algorithm changes or rebuild required. See HIGH_J_VALIDATION.md.
Literal411 high-J archive rows remain unverified; general lower-J second-stage
oracle, rank4 support and matched native basis/GRAC/PFIT remain open. No active
background tasks. tmp/ and orient_replacement.md remain untouched/uncommitted.

## Accepted performance and rank<=3 recoupled parity (prior checkpoint)

FINAL: worker completed; parent built/staged and independently reviewed the engine.
31 focused tests passed1.94s, including exact6285 archive rows/10457nonzero values/
30791written placeholders plus bounded omitted fields. Parent fixed oracle null-field
hole (mutation test) and removed internal electronic-tensor clones before preflight.
Final1741 ISA/FDDS regressions passed24.74s; finalcore water1thread remains bitwise
baseline,29.8982s/589516KiB. Evidence `.pi/audit/recoupled-parity-regressions-v2.log`,
`native-water-post-recoupling-comparison.json`; contract RECOUPLED_CONTRACT.md.
Accepted boundary is shipped rank<=3 Cn(t,u,J) at H2O-isagrid L3 write precision,
NOT full native-SCF/PFIT/GRAC protocol parity.411 J9/10 rows counted, values not
archive-certified; rank4 rejected, C12 partial. No active background tasks remain.
Next scientific work: matched native basis/GRAC/PFIT pipeline, independent high-J
oracle and justified rank4 coverage. Performance and this parity increment complete.

### Execution history for this increment

User requests significant OpenMP/memory/runtime improvement of the working water
input, then continued CamCASP source-based parity. Parent baseline task b5f6b5b2a
completed:360.6726s total,350.9239s native partition,5.4104s native response,
2.0617s SCF,780104KiB peak RSS,31 ISA iterations. Artifacts:
`.pi/audit/native-water-baseline-t1.{json,npz,profile.txt,log}`. Explicit wall-time
wrappers are authoritative; cProfile's captured Python totals omit most native time.
Worker b7b8952cd completed immutable prepared ISA work, streaming density and
independent-output OpenMP. Its source-only handoff is
`.pi/audit/native-water-performance-handoff.md`; parent subsequently fixed memory,
ownership and AO-adaptation reproducibility, built/staged and validated them.
`.pi/audit/profile-native-water.py` now fixes SCF at1thread while varying property
threads for controlled scaling; baseline already used1thread. Comparator
`.pi/audit/compare-native-performance.py` checks all outputs/iteration counts and
reports measured speedups/RSS, never infers gains from code inspection.

After performance acceptance, parent launched task baf02e557 using
`.pi/audit/implement-recoupled-parity-prompt.md`: shipped rank<=3 local recoupling
and Cn(t,u,J), new realcg data/code and typed result. Read-only audit identified
CamCASP src/casimir/casimir.f90 source anchors and exact audit reference track
H2O-isagrid (NOT historical H2O). Correct audit count:10457 nonzero coefficient
VALUES across6285 rows,30791 placeholders; no native-SCF/PFIT/GRAC parity claim.
Performance source build and first1700tests passed. Initial cached version cut
runtime to30.55s1thread/16.34s8threads but increased RSS15%; not accepted as the
memory endpoint. Parent fixed cache-construction accounting (nf+12), added readonly
cache admission, independent cacheless controller snapshots released before finalQ,
and streamed AUX-Q collocation in4096point/8MiB blocks. Review caught a real pybind
unique/shared holder mismatch in the new snapshot API; private-runtime crash was
reproduced and fixed with a shared holder plus deletion/lifetime regression.
Final private build passes1707tests in23.05s. New measured memory runs PASS:
29.9609s1thread (12.04x,590144KiB RSS,24.35% lower,all outputs bitwise baseline),
16.0515s8threads (22.47x,598380KiB,23.29% lower,max tensor abs2.14e-14).
Artifacts `.pi/audit/native-water-memory-comparison.{json,log}`. Initial2/4thread
comparison failed due to MKL SVD AO conversion; the environment-only BLAS domain
attempt did NOT fix it. A thread-local serial MKL callback guard around adapt_main
DOES preserve baseline arithmetic, with no process-global setters. Non-MKL builds
are untouched. Tests cover restoration, nesting, exceptions and cross-thread scope.

FINAL PERFORMANCE ACCEPTED: build/install-v5 completed, staged core byte-compared;
1710tests passed24.06s (`native-water-performance-tests-v4.log`). Full1/2/4/8 runs
PASS unchanged1e-9 gate, worstscaled9.83e-16, one-thread allbitwise. Runtime
29.4297/22.5083/17.9785/15.9646s; RSS586864/584392/586736/598196KiB (23–25% lower),
12.26–22.59x speedup. Final evidence `native-water-final-comparison.{json,log}`.
Standalone no-OpenMP helper compile/run passed; no full serial-core build.

Completed parity worker baf02e557 was SOURCE-ONLY: no build/install/stage changes,
no commits, no long tests, no edits to performance implementations or existing
Python modules. Handoff `.pi/audit/recoupled-parity-handoff.md`; worker logs
`recoupled-parity-worker.{jsonl,stderr}`. Parent completed build/tests after handoff.
Independent final molecular/SAPT endpoint task bb1354c09 PASSED against frozen
final staged core:3 molecular tests97.35s,4 SAPT tests113.48s. Logs
`.pi/audit/native-water-final-{molecular,sapt}-tests.log`. Normal8thread SCF+property task
b931fd28d PASSED without SCF override:14.3199s,743280KiB RSS,31iterations,
energy-76.3387589007227. Its memory is separate from controlled SCF1thread runs;
do not reuse the23–25% reduction for this different thread policy. Report HTML and SPEC updated with performance and accepted rank<=3 archived-table parity.
ORIENT source remains forbidden; permitted MIT CamCASP source retains attribution.

## Current native oeprop integration

Direct implementation (no background agents): native task dispatch, generated H/O
JKFIT/even-tempered recipe, owned result accessor, and explicit direct-OV C++
partitioned moments are implemented. Density remains Drho-C ordinary ISA-A;
response transitions are integrated from actual MOs, not repaired lambda1 fits.
Production LW stays1e-6. Existing uncommitted changes and parent attempts retained.
Parent water attempts1/2 failed; attempt3 has only a recipe, no completed endpoint.
Configured core build/install passed (`native-oeprop-{build,install}.log`), built
and staged core byte-compared; public Python modules staged. Focused74 tests pass
in2.97s (`native-oeprop-focused-v1.log`). Fresh `tmp/psi4_camcasp.py` and full
molecular execution is running; full ISA/FDDS regressions passed1664 in28.67s
(`native-oeprop-regressions-v1.log`), including all previously existing tests.
No new water endpoint acceptance yet. Worker exited without its requested handoff;
its background water run did not survive. Parent recovered installed fresh molecular
tests in `native-oeprop-water-parent-tests-v2.log` (v1 was source-import shadowing).
Independent review found SCF-convergence evidence, effective-PBE0 validation and
unknown ATOMIC_* dispatch defects. These are fixed in source, with34 focused tests
passing in3.17s in a private runtime (`native-oeprop-review-parent-tests-v4.log`);
canonical stage remains unchanged while the older numerical test runs. New fixtures'
option-state and LibXC setup errors were corrected, with failed logs preserved.
The installed water tests measured partition PASS and two response resource-limit
failures (1009.39s): reusing the160/590 ISA grid exceeded the native ALDA work bound.
Added independently configurable ATOMIC_RESPONSE_RADIAL_POINTS=99 and
ATOMIC_RESPONSE_SPHERICAL_POINTS=590; no work-bound or numerical tolerance change.
Canonical core rebuild/install and source/core byte identity pass. Final installed
ISA/FDDS regressions pass1686 in31.09s (`native-oeprop-parent-regressions-v3.log`).
Fresh actual tmp input execution PASSED in341.305s: all11 response nodes and9
site-pair Cn sets, static O/H alphas3.54925461809/.875697468695/.875697468444.
Saved-array independent C6 check max error1.78e-15; current input/core hashes match.
Four actual SAPT input regressions passed112.40s. Final slow water assertions
PASSED3 tests in1346.09s (`native-oeprop-water-parent-tests-v3.log`), including
partition/static/dispersion isolation, analytic Mints dipole/H1 consistency, all9
independent C6 checks and result ownership. No numerical threshold was weakened.
Completed endpoint artifacts,
commands, implementation scope and preserved failures are recorded in
`.pi/audit/native-oeprop-water-handoff.md`.
A mistaken CTest launch found no tests and is NOT evidence; actual SAPT pytest
runners replaced it. An earlier regression was stopped after explicit-file arguments
bypassed --ignore and included the known-failing slow water test; final invocation
filters filenames before collection. See `psi4/src/psi4/libisapol/NATIVE_OEPROP.md` and
`.pi/audit/fix-native-oeprop-review-handoff.md`. No commit/push; tmp remains uncommitted.

## Latest directive — port toward returned atomic alpha and Cn

User explicitly requests continued code porting until atomic polarizabilities and
Cn coefficients are returned, and reaffirms Drho1e-2. Named `~/gits/psi4_camcasp`
is absent; the established source `/home/awallace43/gits/camcasp_psi4` exists on
branchcamcasp at the previously permitted LGPL commit
5449bd1a01c73f45c307b36b006e264c1e43b994. Parent found LGPL-headered
`psi4/driver/procrouting/atomic_polarizability.py`, an existing native orchestration
entry point; its multi-SCF/GRAC and Doo-C policies require reconciliation, not
blind substitution for agreed no-hidden-SCF/Drho contracts.
Read-only audit `d0f763eb72b4cf6d8878c1f372ef5772f` inspects only permitted old
Psi4 source/test subtrees and current contracts to select concrete pipeline port
scope. Forbidden old `orient/` is not accessed/searched. Native-response worker
`b09058979` remains independent in its new comparator/test scope. Drho1e-2 does
not imply blanket downstream tolerance changes; all10 historical LW failures
remain blocked unless explicitly superseded. Goal is actual newly computed
atomic alpha and Cn, not relabeled imported refined properties.
Full-pipeline audit retrieved (answer SHA256
`d52a104f4ed75da92d60b7cfadee842096b2a8eb28cc799df8d1c7fc755fef3a`).
Port native providers/orchestration, not monolithic old calculator or its forbidden
CDF LW tolerance max(requested,100*residual). Old basis-space branch uses Doo-C,
not approved Drho-C ISA-A. A separately named real-space AO-density ISA path exists
but must not silently replace ISA-A. Audit incorrectly suggested incomplete local
rank3 isotropic C10; current engine/spec correctly mark C10 complete and C12 partial.
Worker `bf61c5667` now ports common native response operator prerequisite to new
`native_response.{h,cc}`, optional new `isapol_native_response.py`, new tests and
narrow additive CMake/bindings, from LGPL native builders. Prompt
`.pi/audit/port-native-response-provider-prompt.md`; handoff
`.pi/audit/native-response-provider-handoff.md`. No worker build/stage. Preserve
owned scope until completion. Reuse existing shared frequency solve, explicit
kernel/representation, no hidden SCF or unverified provenance seals. Parent still
owes partition recipe/final-state Q adapters and returned alpha/Cn integration.
User explicitly chose **Keep Drho-C ISA-A** for the first full pipeline; do NOT
switch to native real-space AO-density ISA for convenience. Read-only partition
adapter audit `d9675382a748b88b0695c4c315e13377e` specifies source-grounded
basis conversion, ONE-GTO/current controller setup and final tail-aware Q in
parallel with response source worker, avoiding its owned scope. Preserve Drho1e-2
and existing explicit Cartesian adapted track labeling; no invented basis aliases.
Native-response provider worker completed source handoff: owned wfn-derived
occupied-fast fullOV V/X/Y/H1/H2, explicitly named no-local/Slater/PW92/VWN
kernel policies, native shell-integral construction, explicit grid, and shared
FDDSFullOVResponse returning factory. No runtime acceptance yet. Parent
`bea4ed00f` builds configured core (`native-response-provider-build-v1.log`),
review `d1d28f515c2688442f52ff137bfd5a408` inspects frozen new source/tests.
This still needs canonical staging, actual molecule tests and partition integration;
source presence is not returned atomic alpha/Cn.
Build `bea4ed00f` FAILED: native_response.h uses SharedMatrix/SharedVector
without their declaring header (psi4-dec.h does not provide them); downstream
constructor/getter mismatch errors follow. No core was staged. Fix after active
review releases frozen scope, then rebuild; failed v1 log retained.
Partition audit retrieved (answer SHA256
`74616a99f1e098410f13c2e9cb31cbd5657cde72fc356fa63fcffa4fc4aaa46a`):
current kernels connect Drho->controller->Q, but exact automatic native basis
recipe and Psi4-to-DALTON C conversion remain unestablished. Do not substitute
old Doo-C/log-nearest initialization. Worker `ba9609bbe` now implements new
`isapol_native_partition.py`/tests only: explicit fully declared basis recipe,
validated wfn AO conversion, fresh native Drho, absolute-nearest ONE-GTO alpha1,
current ordinaryA controller and stored-tail-aware final Q. Explicit recipe
support must not be mislabeled automatic modern preset or reference-orbital input.
Prompt `.pi/audit/implement-native-isa-partition-prompt.md`; expected
`.pi/audit/native-isa-partition-handoff.md`. No existing C++/build/staging edits.
Requires actual SCF-based smoke and honest convergence; missing prerequisites
must remain blockers rather than placeholders.
Response review retrieved (answer SHA256
`35d724b976f4364502902321135f27f81dfded32b55102ad16b17703e1115859`):
three must-fix findings: SuperFunctional overwrites component cutoff, overlap
constructor still reads ambient options/threads, mixed Gaussian/Libint shell
representation can cause buffer dimension mismatch. J/K permutations/assembly
otherwise sound. Worker `b970ed67f` fixes these plus missing header declarations,
adds cutoff/ambient-state/highAM coverage; prompt
`.pi/audit/fix-native-response-provider-prompt.md`, expected
`.pi/audit/native-response-provider-fix-handoff.md`. Only response source/tests;
no build/stage, no partition scope edits. Native response remains unaccepted.
Fix worker completed source changes and regressions; parent `bd9b29980` rebuilds
v2. Prepared `.pi/audit/test-native-response-provider-private.py` for post-build
runtime using private copy of prior staged package plus finished response module/
built core and staged dependency fallback, verifying shared core unchanged. Do
not canonically install the unfinished partition module while its worker runs.
V2 configured core build `bd9b29980` PASSED. Private molecular/provider tests
`ba1b89c4b` now run; follow-up review `db4c03cedaf78829cf9c4a4faf5154463`
checks fixes in frozen response source/test scope. No runtime acceptance yet.
Private-v1 failed before tests: missing private share/psi4 data path. Parent fixes
runner only with explicit PSIDATADIR=canonical stage/share/psi4 and fresh private-v2
prefix/logs; `be1cae5bb` reruns. No numerical failure was measured by import error.
Private-v2 imported correct built core and ran actual tests:72PASS/10FAIL in2.59s.
All failures are ALDA local-primitive comparisons (nine policy/scale cases plus
single-point cutoff); native integral/Hessian no-local controls pass. This is a
real collocation/local-kernel-oracle mismatch, cause not yet established; no
threshold change. Shared core unchanged. Keep response scope frozen until active
follow-up review retrieval, then isolate AO ordering/normalization before fixes.
Follow-up review retrieved (answer SHA256
`f7548b86a3191a2bc40e2a389f567b16fe29d785433492f2fa6ad160fcb5415e`):
prior safety fixes resolved; runtime mismatch remains independent. Worker
`bc455637b` investigates ALDA vs test collocation using private-v2 core only,
from `.pi/audit/diagnose-native-alda-prompt.md`, handoff
`.pi/audit/native-alda-handoff.md`. Prove spherical/Cartesian AO conventions and
normalization before changing oracle or producer; no tolerance changes/staging.
Worker proved oracle error: configured Gaussian pure P order is z,x,y, Cartesian
is x,y,z. Independent asymmetric-point polynomials and cross-center analytic
S/P overlap establish mapping (overlap errors<8e-16; old ordering error0.552).
Corrected only test oracle, added pure/Cartesian and block-boundary regressions;
private85 tests PASS2.72s. C++ unchanged, no thresholds changed. Handoff
`.pi/audit/native-alda-handoff.md`. Parent `b403a4fee` now runs all ISA/FDDS
against private-v2 core, explicitly excluding only unfinished partition-worker
new test. Parent `b403a4fee` PASSED1571 tests in15.61s and diff check.
Canonical staged core still unchanged; partition integration pending. SPEC
records private-runtime native provider scope, explicit policies and no endpoint
claim. Partition worker subsequently completed new factory/tests: actual fresh
RHF/aug-cc-pVTZ water converged38iter and Q75x246 from explicit adapted basis
recipe, no archived C/D/shapes. Q quadrature error1.98e-4 is not accepted parity;
small H2 recipe max120 failure preserved. Handoff
`.pi/audit/native-isa-partition-handoff.md`; partition review
`d6ec49ad4ee3b80cbeffb3c22958dbcee` active, freeze module/test scope.
Parent `babd209a2` canonically installs all finished prerequisites, checks core/
module byte identity and full tests. Prepared returning native pipeline prompt
`.pi/audit/implement-returning-native-pipeline-prompt.md` for launch after staging
finishes: validated full-C transform/native OV->shared full response->Q/LW->alpha/Cn,
no historical LW exception; legitimate recorded grid refinement if strict gate fails.
Canonical prerequisite stage `babd209a2` PASSED core/module byte identity and1595
combined tests in26.36s. Worker `b51dc8235` now implements new `isapol_native.py`,
new pipeline tests and actual fresh-water run under the prepared prompt; expected
`.pi/audit/native-pipeline-handoff.md`. Owns only new pipeline files/audit outputs,
not frozen partition source. No endpoint acceptance until actual values and parent
validation. Native response operator generation may be expensive at435OV; report
resource/runtime blockers truthfully and preserve all failed attempts.
Partition review retrieved (answer SHA256
`68d2ff48b27325fc9156ee3e8877f7849b205cb296a6bf4aed7f0c2e1566abdf`):
no must-fix for expert native Drho->ordinaryA->Q path. Optional direct helper
CartesianD+ rejection, broader boundary tests and aggregate memory preflight remain
followups, not bounded water acceptance defects. Preserve source/staged identities
while pipeline worker runs; source review freeze released, measurement freeze stays.
Pipeline worker `b51dc8235` exited0 WITHOUT handoff while its water subtask was
unfinished; no matching water process remained at parent inspection. Treat as
PARTIAL, not completed molecular acceptance. Source/new17 tests and real He
smoke exist: He alpha1 .21704560691327368, C6 .037262714184643324 in an explicitly
small demonstration recipe, not accurate water/property parity. Parent recovers
fresh all11-node water run `bd1924d2c` from copied runner with new
`native-pipeline-water-parent-v1-*` outputs, preserving incomplete old logs.
Review `dbff6c2f5bffcddd262dff614d0d9b374` checks frozen new driver/test scope.
Parent `bc6e15586` stages only finished pipeline module with byte identity and
runs full suite. No stopped/incomplete child run is accepted as water output.
Parent full suite PASSED1612 tests in29.05s. `b0679e874` independently regenerates
He output through normal installed pipeline import with explicit path assertion,
to fresh `native-pipeline-he-parent-v1.{json,npz,log}`. Water run and review remain
pending; don't mutate fingerprinted pipeline/prerequisite sources.
Installed He regeneration `b0679e874` PASSED: compact-model static dipole trace
alpha=.21704560691327368bohr^3; He-He C6=.037262714184643324,
C8=.08395135528327327,C10=2.2003210700152813,C12partial=8.737728993162474
(atomic units). Independent C6 recomputation differs6.94e-18. All11 nodes passed
strict LW in this small demonstration recipe, but these are not basis-limit helium
values or water acceptance. Parent JSON/NPZ retains tensors/recipe/diagnostics.
Pipeline review retrieved (answer SHA256
`80f50cefcea0136c8a99841a98ece31b1dcc71f758bca2d5aa356246ad889f18`):
no must-fix production orchestration defect; standalone artifact verifier was
fail-open (empty glob, unasserted hashes). Parent changed only audit verifier:
requires explicit nonempty completed report paths, asserted source/core/NPZ hashes,
separate failure-evidence vs endpoint mode and endpoint arrays/all11LW/9pair-C6
checks. Negative absent/empty/hash-mismatch controls pass without emitting evidence.
Do not use old verifier success as acceptance. Water run still active; no live
source/numerical mutation. Pipeline result ownership/failure-retention limits
remain as documented by review, not an all-node upstream-failure collector.

## Resolved — CamCASP is MIT; source is a permitted porting input

User confirmed **CamCASP is MIT-licensed and not blocking**
(`MIT License, Copyright (c) 2019 Anthony Stone`). MIT is compatible with Psi4's
LGPL-3.0, so `~/gits/CamCASP/src` (v6.0 patchlevel 051, 225 Fortran files, ~195k
lines) **may be read and transcoded** into `libisapol`. Recover the license text with
`git -C ~/gits/CamCASP show b40ae4f^:LICENSE`; the file was deleted from HEAD on
2019-10-30 in commit `b40ae4f` ("Recompiled; redundant files removed"), which is why
earlier increments kept concluding "unlicensed" and reverse-engineered stages from
captured numerical output instead. Do not re-litigate this.

Obligations: retain the MIT notice and permission text in derived files, add a
CamCASP entry to Psi4 licensing docs, preserve Misquitta/Stone attribution, and cite
the CamCASP file and line per ported kernel. Unchanged prohibitions: **ORIENT stays
GPLv3 and forbidden**, no GDMA grid transliteration, and do not *ship* extracted
CamCASP source/binaries as fixtures. The rule that production code and pytest never
*invoke* CamCASP at runtime is a dependency constraint, not a reading ban.

### Blocker #18 — df_Smat.F90 read against our metric

Direct source comparison of `~/gits/CamCASP/src/df_Smat.F90` against
`orbital_coulomb.cc::fit_drho_c`. Three results, no code change yet.

**(a) Our constraint algebra is correct — confirmed, not inferred.**
`make_s_matrix_constraints` (df_Smat.F90:415-564) builds
`(Sc)_ij = factor1*<chi_j|chi_i> + factor2*I_i*I_j`, i on site m and j on site n, with
site-block-dependent factors: ConstraintType 1 gives `factor1 = 1` on-site and
`1 - eta` off-site; ConstraintType 2 gives `1 + eta` on-site and `1` off-site; both
give `factor2 = lambda + gamma` on-site (gamma gated to `Z==1` sites when
`gamma_H_only`) and `lambda` off-site. The reference method files
`methods/isa-A+DF:25-27` and `methods/isa-A:27-29` set **Eta = 0.0, Gamma = 0.0,
Lambda = 1000.0**, and `ConstraintType = 1` is the default (df_parameters.F90:50).
At eta=gamma=0 the formula collapses to `factor1 = 1`, `factor2 = lambda` uniformly,
which is exactly our `J_kj + lambda*q_k*q_j`. The RHS also matches: df_Tmat.F90:587
gives `(TMO)_ij,k = <phi_i phi_j|chi_k> + lambda*delta_ij*I_k`, and our
`rhs[k] += 2*(diagonal + penalty_q)` accumulated over occupied orbitals yields
`raw_rhs + lambda*q_k*N` for closed shell. **No missing constraint term; the metric
difference is not structural.**

**(b) New confirmed difference: CamCASP screens the metric, we do not.**
`make_s_matrix_coulomb` applies a Cauchy-Schwarz shell-pair screen
(df_Smat.F90:219-222):
`upper_bound = sqrt(Gaux(ishell)*Gaux(jshell)); if (upper_bound < integral_cutoff) skip`,
leaving those entries as **exact zeros** and counting them in `number_skipped`
(reported at df_Smat.F90:348). The threshold is
`par_integral_cutoff = 1.0e-12` (parameters.f90:122), runtime-overridable via `CUTOFF`
(df_main.F90:165-166) but **not overridden in the ISA-A/ISA-A+DF method files**. Our
`IsaAuxCoulomb::metric()` builds every entry densely. So reference `A` carries exact
zeros where ours carries small nonzeros — a genuine structural difference in the
metric, and one of the candidates named in the conditioning critique.
*Refuted as a cause — see (d).* The screen exists, but measurement shows it moves
nothing: the reference `CUTOFF` was the 1e-12 default and the screened entries are
below 1e-14 in our dense build. The earlier `~2.5e-10 ~ 15%` estimate in this section
was a loose upper bound; the measured share is `9.4e-10` of `||dA||_F^2`. Do not
spend further effort reproducing the screen for parity reasons.

**(c) The conditioning is self-inflicted by lambda = 1000.**
The predicted mechanism is confirmed: `cond(J) = 1.9351470110863735e12` versus
`cond(A) = 6.026362161121845e15` is a factor of **3108**, and the augmentation is a
rank-1 `lambda*q q^T` with **lambda = 1000.0** from the method file. CamCASP tolerates
this because it solves by LU and never compares coefficients against another code; we
inherit the conditioning and then gate on coefficients. CamCASP itself exposes the
alternative: `densfit_prop.F90` provides `PropSolver` with an SVD path plus
`SVDcondition = par_condition_number` (`1.0e-15`, parameters.f90:81) and
`use_constraints`. An exact KKT/Lagrange or nullspace-projection treatment of the
charge constraint, or the SVD path, should recover ~1e12 conditioning.

**(d) RESOLVED — the reference `CUTOFF` is 1e-12, screening is irrelevant, and the
raw-coefficient gate is unidentifiable in double precision.**
Diagnostic `.pi/audit/native-drho-identifiability.py`, evidence
`.pi/audit/native-drho-identifiability.json`. Run with
`python -P .pi/audit/native-drho-identifiability.py`.

*Reference `CUTOFF` found.* The instrumented capture writes `integral_cutoff` into
every metric record (`capture_native_df.py` `ndc_metric`, line 4 of each `.dat`).
Both `isapol-native-df-J-*.dat` and `isapol-native-df-A-000001.dat` in
`.pi/audit/native-df-v4-water-traced` record **`1.0e-12`**, i.e. the
`par_integral_cutoff` default; the run's own deck `H2O-native-df.cks` has no `CUTOFF`
directive (only `Eta = 0.0`, `Lambda = 1000.0`, `Gamma = 0.0` under
`SET DF-INTEGRALS`). The state header records `norm=1, solver=0 (LU), iterations=0,
constraint=1`. So the screen was active at the method default and nothing was looser.

*Screening contributes nothing.* Reference `A` has 37,672 exact zeros (62.25% of
246^2); 764 of them are nonzero in our dense build, with
`max|A_ours| = 6.27e-15` there. Restricted to that pattern
`||dA||_F = 4.99e-14`, i.e. `9.4e-10` of `||dA||_F^2`. Forcing our metric to the
reference zero pattern moves the coefficient error from `1.280219e-03` to
`1.280220e-03`. Most of those zeros are structural (angular selection rules), not
screened.

*The whole of `||dA||_F` is the lambda = 1000 amplification of last-bit charges.*
`||dJ||_F = 1.908e-11` but `||dA||_F = 1.630e-09`, and
`lambda*||d(q q^T)||_F = 1.581e-09` — **97%** of it. The charges themselves agree at
`||dq||_2 = 1.72e-14`, relative `2.89e-16` = **1.30 machine epsilon**. There is no
integral defect to find: `q` and `J` are both at roundoff, and `lambda = 1000` then
multiplies the rank-1 term's roundoff by ~85x relative to `J`.

*The reference solve is bit-exactly reproducible.* LAPACK LU on the reference `A` and
reference RHS returns the reference `Drho` with `max|delta| = 0.0` exactly. The solver
path is therefore not a divergence either.

*The gate is provably unidentifiable.* `cond_2(A) = 6.026e15`, so
`cond*eps = 1.338` and the coefficient vector has **zero guaranteed correct digits** in
double precision. A *random* `dA` of the measured Frobenius norm yields scaled
coefficient error `8.36e-03` — 6.5x worse than our observed `1.28e-03`. Our fit is
better than a generic perturbation of the same size, not worse. Attribution:
our-A/ref-b `1.565e-03`, ref-A/our-b `3.181e-04` (non-additive, as the conditioning
critique predicted).

*What does agree.* Gauge-invariant errors on the same `dc`:
`||dc||_A / ||d||_A = 5.947e-09`, `||dc||_J / ||d||_J = 1.946e-07`, electron-count
drift `q.dc = -8.87e-12`, metric `max_scaled = 6.94e-16`, fitted electrons
`8.84e-13`.

Next on #18: the remaining choice is a **gate-definition** question, not a numerics
bug. Options, in preference order: (i) gate the A-norm (energy-norm) coefficient error
and the electron count, which are the quantities the Dunlap-robust functional actually
sees, and record `5.95e-09` as the current value against a stated threshold;
(ii) ask CamCASP for the well-posed solve — `densfit_prop.F90` exposes `PropSolver`
with an SVD path and `SVDcondition = par_condition_number` (`1.0e-15`,
parameters.f90:81) — and recapture the reference through it, which makes both sides
compare a pseudoinverse solution that is stable rather than an LU solution that is not.
Do not rescale, regularize or symmetrize the existing fit, and do not present option
(i) as parity at 1e-9 on raw coefficients: that number is unreachable, and the reason is
now measured rather than argued.

**(e) RESOLVED — gate 5's native-OV leg has the same shape, and its identifiability
floor is set by CamCASP's own documented solver choice.**

The second reference `CUTOFF` question is also answered: the response run's deck
`.pi/audit/response-v5-water-traced/H2O-response.cks` fixes
`KERNEL-INTEGRAL-CUTOFF = 0.10E-07` (:317), `SOLVER LU` (:319 and :331, with the
comment `Options are LU and GELSS`), `Eta = 0.0` (:336) and **`Lambda = 1.0`** (:337).
So the OV leg carries no `lambda=1000` amplification - unlike the Drho-C leg, the
rank-1 penalty is unit weight and `A = J + q q^T` with `||J||_F = 1.159e+03` against
`||q q^T||_F = 3.531e+03`.

*The solver path, read from CamCASP.* `solve_df_equations_lu` (`src/df_utilities.F90`)
solves `S*D^t = T^t` via `lineq_solver_lu(S,T,D,'N','T',iterations,...)`
(`src/matrix_operations_types.F90:301`), which blocks over **all** columns of `T`
(`determine_rhs_block_size`, :483/:601) using **one** factorization: the first block
factors and writes the LU factors to `lu_file` (:527), later blocks re-enter with
`A_lu_file=lu_file` (:539). Both calls reach `lineq_lu_iter(...,iter=iterations,...)`
(`src/matrix.F90:1698`), where `iter == 0` sets `l_iterate = .false.`, i.e. plain
DGETRF/DGETRS with no iterative improvement. This closes the earlier open question
about the reference-inputs control: solving the recorded **full** RHS (4278 columns)
with one factorization and selecting the 435 OV columns afterwards reproduces the
exported OV coefficients at **exactly 0.0**, while solving the 435 OV columns alone
lands at `2.471767645468478e-11`. The `2.47e-11` was never a solver-path difference -
it is DGETRS panel blocking over a different column count. Both numbers stay recorded
separately; they are not conflated as bit-identical.

*The identifiability floor, measured.* On the identical reference inputs, the deck's
own documented alternative solver disagrees with LU by the size of our native error:
GELSS min-norm at `rcond = 1e-15` gives scaled `2.309291e-05` (stable across
`rcond` in `{1e-15, 1e-14, 1e-13}`) against our both-native `2.301980e-05`.
`cond_2(A) = 8.085609e+12`, `cond*eps = 1.795e-03`, two guaranteed correct digits;
`cond_2(J)` alone is `1.935e+12`, so the conditioning here is intrinsic to the aux
Coulomb metric, not manufactured by the penalty. Our inputs are at roundoff - metric
`max_scaled 3.849410e-15`, `||dA||_F = 1.963e-11`, `||dRHS||_F = 1.415e-11` - and a
*random* `dA` of the same Frobenius norm yields `1.047898e-04`, 4.6x worse than ours
(a random `dRHS` of the same norm yields `3.415992e-06`). What does agree is
gauge-invariant: the energy-norm relative error over the 435 transitions is median
`7.498502e-10` / max `1.265225e-07`, and the LU-vs-GELSS spread in the same norm is
median `7.635825e-10` / max `2.111125e-07` - our disagreement and the solver-choice
disagreement are the same size in the norm the Dunlap-robust functional actually sees.
Charge-constraint drift is `max |q.dc| = 4.643138e-10`.

Reproduce with `python -P .pi/audit/native-ov-identifiability.py [--json OUT]`;
evidence retained as `.pi/audit/native-ov-identifiability.json`.

*Consequence for the gate.* A raw-coefficient gate below `~2.3e-05` on the native OV
leg is not a parity statement about our integrals: it is a statement about which of two
solvers CamCASP was configured with, and the deck offers both. Gate the energy norm and
the charge constraint (`7.50e-10` median / `1.27e-07` max, `4.64e-10`), recorded against
stated thresholds, and keep the raw-coefficient number as a reported diagnostic with the
GELSS spread printed beside it. The same prohibition as (d) applies: do not rescale,
regularize or symmetrize the fit, and do not recalibrate the tolerance to whatever number
the current build happens to produce.

## Active continuation — LW anisotropic adapter and dynamic input gate

Parent resumed the accepted supplied-nonlocal checkpoint without modifying existing
numerical policies or uncommitted work. Source worker `b1ab635b7` implements only
`isapol_lw.py` and new `test_isapol_lw_anisotropic_driver.py`, using existing C++
anisotropic contraction with explicit whole-model placements, exact reciprocity,
owned typed results and no tensor repair. Prompt
`.pi/audit/implement-lw-anisotropic-continuation-prompt.md`; expected handoff
`.pi/audit/lw-anisotropic-continuation-handoff.md`. No build/install/staging by
worker. Keep its source/test scope frozen until completion; installed validation
and independent review remain pending. Synthetic dynamic tests do not establish
actual molecular-frequency or native acceptance.

Worker `b1ab635b7` completed the handoff: typed explicit-placement adapter and
new synthetic tests. Selected source-loaded checks passed255 with one stale
metadata assertion deselected, not full acceptance. Parent updated only that
existing assertion to the new truthful adapter-available/not-computed status.
Parent `b89df4ae7` stages only the finished Python module, asserts installed import
and source/staged byte identity, runs a private installed-import copy of all new
adapter tests and the full ISA/FDDS suite without deselection. Runner
`.pi/audit/validate-lw-anisotropic-parent.py`; logs
`lw-anisotropic-parent-{validation,installed,combined}-v1.log`. Review
`dff5cb1a64ea4dc7438354d299be96552` is active; preserve adapter/new-test source
scope until retrieval. Numerical kernels and historical exception are unchanged.
Review `dff5cb1a64ea4dc7438354d299be96552` retrieved (answer SHA256
`3a2a499c3d9ab5382999b80babcfe038ea17730d7c687dad4aa7d9a290807184`): no
must-fix within the documented trusted-factory boundary. Optional follow-ups:
remove backend-dependent requirement that frame rotation creates roundoff asymmetry,
add independent rotated-placement/A-B exchange and actual nonlocal-transfer chain
tests. Parent validation `b89df4ae7` PASSED52 installed-import tests and1387
combined tests in14.64s, with source/staged module SHA256
`3205333cbd2e300b63c41f7707db9adcf5daddea742afacee6e53dbf6dccdefc`.
After completion parent replaced the backend-dependent asymmetry assertion with a
deterministic getter-access guard: conversion must never read either raw_local.
Task `baa747e4c` reruns installed and combined tests into fresh v2 logs. No runtime
source or numerical policy changed. Final rerun `baa747e4c` PASSED52 installed
adapter tests and1387 combined tests in14.71s; diff check passed. Bounded evidence
`tests/pytests/data_isapol/psi4_lw_anisotropic_driver_evidence.json` SHA256
`0323315127b22a3a7064a75c54e81f6a6cda35ffa6dd7ca98792bb661b29f614`.
Anisotropic adapter increment accepted at synthetic supplied-input scope; dynamic
molecular import/strict validation is next, not already accepted.

Dynamic-input audit `db7f3d967ee1947ba16b6a60b2f1145af` retrieved (verified answer
SHA256 `2c3367fa0eb3c7abca56972b2f29d863aad66778057d14a5687b76894f2a6807`).
Portable NONLOCAL input is static only. Existing local audit inventory
`.pi/audit/orient-bridge-input-hashes.json` identifies historical numerical
`H2O_NL4_{001..010}.pol` and matching unrefined `H2O_L3_{001..010}.pol` in the
old reference work/H2O directory; payloads have not been imported or freshly
verified. Dynamic refined local tensors are not substitutes. The existing
CasimirGrid(10,0.5) manifest governs refined sections, not automatically the
NONLOCAL files. Next molecular gate requires separate bounded literal-preserving
extraction, original source hashes, explicit grid authority and raw-header
comparison statuses. Preserve four known unrefined L3 header failures. Each
dynamic input must pass production1e-6 before unchanged1e-11 tensor comparisons
and actual-weight dispersion. Never extend the exact-static-only1e-3 exception,
manufacture frequencies, repair tensors or omit failing dynamic nodes.

Source worker `be988aa5a` now performs the separately bounded dynamic numeric
import and strict all10-node measurement. Prompt
`.pi/audit/import-lw-dynamic-continuation-prompt.md`; expected handoff
`.pi/audit/lw-dynamic-continuation-handoff.md`. It owns only NEW dynamic fixture,
extractor/test and audit artifacts; no existing driver/test/docs edits or staging.
Allowed external reads are the20 specifically named numerical NL4/L3 outputs and
adjacent hashes only, verified against existing inventory; no ORIENT source or
recursive external exploration. Chosen CasimirGrid rule must be distinguished
from exact producer quadrature. Any production1e-6 node failure blocks molecular
dispersion, with unavailable local outputs explicit and no tolerance retry.

Worker `be988aa5a` completed import of all63000 literal tokens across20
hash-verified numerical files; new fixture SHA256
`132283408a5906e523231df9f99b1dcec2b88a29eb773b0c867e541a7eeced20`.
Worker focused/portable checks passed71 each, but ALL10 dynamic nodes reject the
production1e-6 postcondition. Exact worst input charge magnitudes range from
7.011e-4 (node1) to2.990e-6 (node10). Zero of6750 candidate output entries are
available; no Cn artifact was emitted. Four L3 header failures remain recorded;
chosen CasimirGrid authority does not certify exact producer quadrature or missing
historical shell commands. Handoff `.pi/audit/lw-dynamic-continuation-handoff.md`.
Parent `bb2f577cc` reruns strict measurement, portable audit and full ISA/FDDS
suite to fresh parent-v1 reports. Review `d468bf5c8ac42af0f40aa9306516b5cdf`
checks frozen fixture/parser/test/measurement scope. Added explicit .gitignore
exceptions for new dynamic fixture and bounded dynamic/anisotropic evidence;
no git staging. Molecular gate remains BLOCKED, not accepted by rejection tests.
Parent `bb2f577cc` independently reproduced all10 strict rejections and zero6750
measured output entries; dispersion remains null. Portable audit passed71 tests
with zero prohibited Python I/O/process attempts; full suite passed1458 tests
in14.92s. Review is still pending; numerical rejection is not a test-suite failure
or successful molecular localization. No dynamic historical exception authorized.
Review `d468bf5c8ac42af0f40aa9306516b5cdf` retrieved (answer SHA256
`0406163cc9d9bd770775fad2799ae2f7efc533359c84862efca0ef26fc45bc60`): no
must-fix for bounded import/strict rejection. Fixture scope released. Bounded
`tests/pytests/data_isapol/psi4_lw_dynamic_evidence.json` SHA256
`6b7dbaba8a4eb4750d1273570299bb8f4db2cb395ed100f4d38f6bd00ac87298`
records all10 exceptions, exact offending input terms and null outputs. Import
checkpoint is complete; molecular numerical gate remains blocked. A separate
user decision is needed before any broader historical diagnostic tolerance;
otherwise continue strict upstream investigation without modifying literal inputs.

User selected **Continue independent native work** instead of either historical
diagnostic tolerance expansion or strict historical upstream investigation.
Dynamic molecular gate stays BLOCKED, no new1e-3 permission. Read-only audit
`df6199fa5acc855ae9cbfc66c08acca64` now selects a dependency-ready native partition/
response increment, including feasibility of pending native-transition-leg
propagation through captured operators versus native provider prerequisites.
Reuse existing FDDS machinery; no fabricated basis aliases/kernel/SCF policy or
premature wavefunction-first claim. Preserve inspected native/FDDS source scope
until audit retrieval.
Audit `df6199fa5acc855ae9cbfc66c08acca64` retrieved (answer SHA256
`1e808dd579d4f25e890b8d01d76bf7665963d993cbb198abd96761db52768ebd`).
Dependency-ready task18 is fresh native C++ OV legs propagated through supplied
captured operators using existing FDDSFullOVResponse, not native kernel generation.
Worker `b09058979` implements only new `compare_native_response_legs.py`,
`test_isapol_native_response_legs.py` and audit artifacts, from
`.pi/audit/implement-native-response-legs-prompt.md`; expected handoff
`.pi/audit/native-response-legs-handoff.md`. No existing numerics/staging edits.
Require one internally matched capture and fresh CPP fit, not mixed v4/v6 arrays.
Separate frozen-Hessian leg substitution from leg+kernel-projection substitution,
with captured-D controls/direct fullOV checks at every captured frequency.
Native OV provisional1e-3 does not waive downstream strict1e-9 forward gates;
retain failures and complete diagnostics. Native partition recipe/initialization/
screening remain separate prerequisites, and all10 historical LW nodes stay blocked.
Worker `b09058979` completed61 new tests/210 focused passes and fresh CPP fitting
from internally matched retained v6 capture. Both experiments completed all11
frequencies: captured-D/direct controls pass, but native forward errors remain
FAIL1e-9 (maxscaled4.57189948e-5 frozen /4.57190794e-5 reprojection). All246
coupling modes retained; no fallback/waiver. Handoff
`.pi/audit/native-response-legs-handoff.md`; use bounded
`native-response-legs-v6-v2-summary.json`, NOT332MB fullreport in context.
Parent `b0fa46256` verifies hashes/operands and runs full suite; review
`d183e38ca8d522b8291c8ae80041eb801` active, preserve new helper/test scope.
This numerical bridge is not the requested fullpipeline endpoint; old LGPL
pipeline port audit continues independently under latest user directive.
Parent `b0fa46256` PASSED1519 combined tests in15.50s and verified949
source/capture/core hashes,26 common numeric operands and132 finite raw matrices.
Strict native forward failure remains unchanged. Review
`d183e38ca8d522b8291c8ae80041eb801` retrieved (answer SHA256
`e1827eae962c86ad24e6868f6c85dc11392cb3a624511a66db7267567cff1bcf`): one
must-fix recursive ownership gap for MappingProxyType identity inputs; measured
ordinary-dict artifacts unaffected. Parent extends frozen() recursion to proxies
and adds nested mutation regression; `bff3f19f8` runs fullsuite v2. Original
measurement/source hashes remain historical snapshots, not silently rewritten.
Ownership fix rerun `bff3f19f8` PASSED1520 tests in15.56s; diff check passed.
Native-response diagnostic implementation validated with explicit forward failure;
full-pipeline port remains the active objective.

## Resumed continuation — recover LW transfer checkpoint

The prior transfer worker left `lw_localization.{h,cc}`, bindings and
`test_isapol_lw_localization.py`, but no transfer handoff. On resumption no
background task or matching worker/build process remained; source presence is not
acceptance. Parent task `b340147b5` now runs configured core build, canonical
install, built/staged byte comparison and graph/translation/transfer tests, logging
`lw-transfer-{build,install,tests}-resume-v1.log`. Independent read-only review
`d04a7f8c374e400dd25e1ef2f96446c41` inspects the transfer port and boundary/tests;
keep that source scope frozen until retrieval. No hermetic fixture import begins
before transfer runtime green. Existing graph and translation acceptance, literal
fixtures, explicit graph policy and separate 1e-6 postcondition/1e-11 hermetic
comparison remain unchanged. Native end-to-end acceptance remains open.

Resume validation `b340147b5` built/installed successfully and verified core byte
identity, then FAILED at 71 passes/1 failure (1.71s). The overflow negative receives
`ValueError: Nonfinite multipole transform` from the reused translation kernel;
the ported test expects RuntimeError. This is an exception-contract mismatch, not
accepted transfer green or a demonstrated numerical failure. Preserve the failed
log; resolve after the independent review releases the frozen source/test scope.

Independent transfer review retrieved (answer SHA256
ca1591275f6c056101adfe0bf37447f609374d8c50d00a4955e79fe9b377dbf2): no
must-fix static findings; transfer arithmetic/ownership/guards preserved. Scope
released. Parent adapted only the overflow negative to accept the shared transform's
ValueError alongside LW RuntimeError, retaining required finite/overflow diagnostic
and all numerical thresholds/literals. Renamed the supplemental .375 test to avoid
historical-oracle ambiguity. Task `b83db49e3` reruns all72 graph/translation/transfer
cases into `lw-transfer-tests-resume-v2.log`: PASSED 72tests in1.65s. Transfer
checkpoint accepted with independent review; no core arithmetic change was needed.
Only then parent launched hermetic worker `ba9335911` from
`.pi/audit/implement-lw-hermetic-resume-prompt.md`, scoped to new bounded numeric
fixtures, new hermetic test/extraction tool and fixture README. No C++/driver/build
edits by worker. Expected `.pi/audit/lw-hermetic-resume-handoff.md`. Parent combined
ISA/FDDS regressions `bdf6be602` PASSED 1224tests in13.76s
(`lw-transfer-combined-resume-v1.log`), excluding the worker-owned new test. Preserve hermetic worker scope until completion; no675-entry acceptance yet.

### Hermetic import recovered — historical input postcondition blocker

Worker `ba9335911` completed fixture/parser/test implementation but the checkpoint
is RED: 25passes/1failure in1.39s. Handoff
`.pi/audit/lw-hermetic-resume-handoff.md`; fixture SHA256
b86d411e5fd81fc20358370ee211ba5b9bd997c57f8918c32ce0334c93ad7f49.
All6300 source tokens retained exactly (5625distributed input +675local expected).
Default1e-6 LW postcondition rejects charge_sum/local_charge7.011e-4; offsite
2.20145e-11, reciprocity5.77138e-12, molecular_sum3.97904e-13. Candidate local
output and675entry errors are unavailable because the API throws, not zeros.
ExactDecimal input terms already give H1/H2 retained32c charge sum
0.3426003-0.5534848+0.2101834=-0.0007011. No cause beyond this measured input
violation is established. The guide's approximately7e-4 historical warning and
its1e-6 production target are incompatible for this fixture's absolute residual.
No threshold/fixture repair/skip/xfail performed. Portability audit reproduced
25passes/1failure with zero prohibited reference-tree I/O. Native integration
remains blocked; seek explicit historical-diagnostic policy rather than treating
forward provisional profiles as permission to relax postconditions.

User explicitly selected **Separate diagnostic run**: authorize1e-3 postcondition
ONLY for this historical fixture, retain production1e-6 failure, and compare all675
output entries at unchanged1e-11. Parent split the test into a strict-default
rejection regression (including exactDecimal input sum) and an explicitly named
historical diagnostic. No default/kernel/fixture numbers changed. Measurement
`baa4ba6ed` PASSED27tests in1.36s (one JUnit record_property format warning).
All675 entries match at1e-11: maxabs O6.252776074688882e-13,
H1 5.60440582830779e-13,H2 5.089262344881718e-13. Production postcondition
remainsFAIL7.011e-4; no native/end-to-end acceptance. Review
`d6a2892de049b4569e383978d80b9e617` inspects fixture/parser/test scope; freeze
that scope until retrieval. Parent updates SPEC only. Source worker `b3fbac3fb`
from `.pi/audit/implement-lw-property-driver-prompt.md` now implements NEW
`isapol_lw.py` and `test_isapol_lw_driver.py` supplied-nonlocal orchestration,
static atomic tensors/scalars and explicit-weight isotropicdispersion. No C++,
existingbridge or staging edits byworker. Expected `.pi/audit/lw-driver-handoff.md`.
Historical1e-3 policy must be exact-input-identity scoped, never broaddefault.

Hermetic review retrieved (answer SHA256
a34b797a2307bb742864316f2828690c0704ca0dbd8c01e47fd8d97acd1c8197): no
numerical/test must-fix; stale README still claimed output unmeasured. Parent
corrected it to retain strict failure and separately record authorized diagnostic
675entry success. Fixture scope released. Bounded evidence
`tests/pytests/data_isapol/psi4_lw_localization_evidence.json` SHA256
54675af5533957f6e1de75d43092a14acb33f700be529e7d71689c8a346739d9 records
source/core/log identities, metrics and explicit failed productionpostcondition.
Driver worker `b3fbac3fb` completed `.pi/audit/lw-driver-handoff.md`:84newtests,
241bounded combinedpasses in6.97s, source-loaded staticwaterartifact with675entry
agreement. Production stillrejects; historical policy guards completeinput/geometry/
frame/frequency/sourcehash identity. Optionalisotropic usesexistingC++; noanisotropic
adapter/PFIT/oeprop/nativeclaim. Parent `b2b61fb57` stagesonlynewPythonmodule,
checks source/stagedbyteidentity, generatesfresh installed-module waterartifact
and runs allISA/FDDS tests. Independent driverreview
`d8f76fc430111b8235bf0e059d35e404f` active; freeze newmodule/testscope untilretrieval.
Source-loaded worker results are not installed/fullsuite acceptance.
Driver review retrieved (answer SHA256
4f73ad2c970809d82a228b406b11abb403d0ff7a89a7c91a88d4641dfb7e6ad8): no
must-fix for factory-produced workflow. Suggested follow-ups are finite-overflow
regressions, nonzero high-rank dispersion tests and trusted-result-constructor
contract documentation. Preserve source while parent installedmeasurement runs;
review alone is not runtime/fullpipeline acceptance.

Parent installed validation `b2b61fb57` PASSED source/stagedmodulebyteidentity,
fresh installed-module watergeneration and1335combinedISA/FDDS tests in14.40s.
Actual artifact `.pi/audit/lw-driver-water-static-parent-v1.json` retains original
nonlocalinput, computedLW local/globaltensors, scalars,dipoles, all5residuals and
productionFAIL. Staticdipoletraces O6.127427953741659,H1 1.7289320737992027,
H2 1.7289324071325358 bohr^3; unrefined, distinctfromimportedPFIT bridgevalues.
All675localentries stillpass1e-11. Bounded driver evidence
`tests/pytests/data_isapol/psi4_lw_driver_evidence.json` SHA256
f37eb26703a90d6421ef3245b54efa0bea295376be2da042b1c5e609783da810.
SPEC/APIguide updated withownership, trustfactoryboundary, policies andlimitations.
LW suppliednonlocal increment complete atstatedscope. Stillopen: dynamic molecular
nonlocalLW-to-Cn withactualquadrature, anisotropicdriveradapter, nativepartition/
responseproviders/PFIT/oeprop andfullwavefunction-firstwateracceptance. Do not
extrapolate staticwater intoimaginaryfrequencies or broadenhistoricalidentityguard.

## Current continuation — validated LGPL LW port

User directs implementation from `orient_replacement.md`. Verified old permitted
source repo branchcamcasp HEAD5449bd1a01c73f45c307b36b006e264c1e43b994 and clean
specified source/test files; LGPLv3 header confirmed. Task4 paper/method blocker
closed as **not required—wrong method identified**. Target is LW, not rejectedLS;
3/8=0.375 dispute resolved. RejectedLSdraft stays outside production source.
Forbidden `camcasp_psi4/orient/` is not accessed, searched or copied.

Task19 implements sequential green checkpoints: graph/pseudoinverse; translation
compatibility; transfer math; hermetic675entryNL4→L3 comparison; evidence/docs.
Only first graph checkpoint worker `b24f5f7f5` is launched, scoped to new
lw_localization graph code/tests plus narrow CMake/export wiring. Prompt
`.pi/audit/implement-lw-graph-checkpoint-prompt.md`; expected handoff
`.pi/audit/lw-graph-checkpoint-handoff.md`. No build/install by worker. Preserve
owned scope until completion. Parent must validate graph before translationguard,
and translation before transfer, without changing literal fixtures. Production
postcondition1e-6 and hermeticcomparison1e-11 are separate from general provisional
forward profiles and must not be conflated. Existing hybrid remains available.

First graph worker completed `.pi/audit/lw-graph-checkpoint-handoff.md`: graph-only
port, explicit resource cap256sites, original eigencutoff/algebra checks and all
four Moore–Penrose identities retained,24 intended runtime cases. No localizer or
translation seam/stub added. Parent `bc424748f` runs configured build/install/core
cmp and graph tests (`lw-graph-{build,install,tests}-v1.log`). Independent graph
review `d6ba0417dc0c851078e692c5358b38a7f` active; keep graph source/tests/binding
scope frozen until retrieval. Graph is not green merely from source/static checks;
Graph runtime checkpoint `bc424748f` PASSED configured build/install/corecmp and
**24tests in1.25s**. Independent graphreview retrieved (answer SHA256
3f4887a504d9b9544ff2c761f3e2e28881ebeedd5fa1e118a8cc24d832c883ee): no must-fix
port regressions. Graph checkpoint accepted. Review's static test-count estimate
is not execution evidence; actual recorded pytest run collected/passed24. Optional
cap-success/BFS-order/diagnostic fault-injection coverage remains nonblocking. Only after that runtime
green, parent started translationguard `b94fca794`: extract permitted old LGPL
translation functions into a development-only C++ harness, compare all16x16 entries
at13 fixed/seeded displacements with current rank3kernel at5e-12, then run verbatim
scalar/densefixtures unchanged. Numeric expectedmatrices become portable fixtures;
pytest never reads oldrepo or invokesreferenceprograms. Runner
`.pi/audit/check-lw-translation-equivalence.py`; result pending. No transfer/localizer
implementation begins before this checkpoint passes.
Translationguard `b94fca794` PASSED: oldLGPL C++vsnewrank3matrix maxabs
4.440892098500626e-16 across13displacements/all16x16entries; both unchanged literal
tests pass (2tests in1.19s). Numericfixture `lw_translation_reference.json` is
portable; oldharness remains development-only. No convention reconciliation needed.

Only aftergraph+translationgreen, transfercheckpoint worker `b7cb0d22b` now ports
localize_lw/residual helpers/PODresults and remainingmath tests, reusingexisting
rank3translation. Prompt `.pi/audit/implement-lw-transfer-checkpoint-prompt.md`;
handoff `.pi/audit/lw-transfer-checkpoint-handoff.md`. No build/install byworker,
no hermeticfixture imports yet. PreserveownedLWsource/binding/newtestscope until
completion. Parentmustverifytransfermath before675entryNL4→L3 gate.

## Completed supplied ORIENT localization bridge

User explicitly directs deferring internal localization by consuming ORIENT outputs,
then producing atomic polarizabilities and C_n even when numerical comparison
problems remain. Valid results must be emitted with warnings/TODOs, not suppressed
by strict parity failures; structural invalidity remains an error. Imported atomic
tensors must never be called fresh native Psi4 wavefunction predictions.

Audit `de743d847272406fc4868297ead20f84c` retrieved (answer SHA256
039776a9848a98ef5592993976c4a7f7046d21c5d8fc161b4cd70b05c5a4219e) found actual
localized/refined11frequency H2O outputs in the existing archive. Important traps:
refined headers falsely mark all frequencies static, requiring explicit hashed
manifest authority; H1 has nonidentity local frame; rank3C12 is partial;
unrefined tensors can be slightly asymmetric. No localized inference from
nonlocal diagonal blocks is allowed.

Task16 source worker `bfb31407b` implements new Python supplied driver/parser,
bounded fixture/manifest/tests and CLI in nonoverlapping scope. Prompt
`.pi/audit/implement-orient-property-bridge-prompt.md`; handoff
`.pi/audit/orient-property-bridge-handoff.md`. NativeOV worker remains separate;
neither worker may build/install/stage. Parent will integrate and emit actual
alpha/C_n artifacts after tests. Task4 is deferred internal-localization replacement,
not a papers prerequisite. Task2 now depends on bridge16 rather than task4.

Bridge worker `bfb31407b` completed: typed immutable supplied-response driver,
strict NEW parser, explicit manifest/frames, portable archived tensors/independent
casimir data and CLI. Its40 pure tests passed; productioncore tests were not run.
All33 refined tensors reciprocal;22 hydrogen tensors indefinite; four rounded
unrefined frequency headers fail displayed precision and remain warnings/comparison
failures. Independent review `d21909782c0282baacc7729880fb6501b` active; freeze bridge
source/tests/fixture scope. Parent `b15f14136` now canonically installs builtOVcore
and Python bridge, verifies both binary/source copy identities, runs focused/full
tests, then emits `.pi/audit/orient-bridge-supplied-properties.json` from actual
archive with explicit exampleB translation(0,0,10)bohr. This output is hybrid
imported-alpha/Psi4-computed-dispersion, not referencegeometry or wavefunction-first
native property acceptance. Numerical agreement failure must not suppress output.

Parent `b15f14136` PASSED canonical core/Python-copy identity, focused bridge tests,
and **1146 combined tests in11.69s**; produced the requested properties JSON.
Both isotropic and anisotropic9pair outputs are available. All12 bounded archived
isotropic potential-row comparisons pass at printed precision; aggregate numerical
agreement remains false solely for four unrefined frequency-header comparisons.
Static derived alpha1: O6.129740498218001, H1/H2 1.7335580596856666 bohr^3.
O-O isotropic C6=17.25558575556027,C8=346.4239586717264,
C10=7484.441267944163,C12(partial)=127231.00898599371. Preserve imported-tensor/
computed-dispersion labels. Independent bridge review remains pending.

Fresh production OV measurements now run on the canonical core with matchingv4
lineage: CPP `b488a3285` and NumPycontrol `ba1041626`; outputs
`.pi/audit/native-ov-{cpp,numpy}-v4-v1.{json,log}` plus coefficient NPYs. CPP task `b488a3285` PASSED provisional1e-3, strict remains FAIL:
coefficient scaled2.3019798321950356e-5, sampled density5.0303437185125854e-8;
C_DGESV INFO0 and C++ backward residual1.0803202379140819e-17. This is actual
native-integral/supplied-orbital C++ fitting, not the former NumPy-only diagnostic.
NumPy control `ba1041626` also PASSED provisional1e-3 (strict FAIL). Parent
confirmed exact matching compared_input_lineage and bitwise-identical435x246
CPP/NumPy coefficient arrays (maximum difference0). Both fingerprinted measurements
have completed; source scopes released for bridge boundary fixes.

Bridge review `d21909782c0282baacc7729880fb6501b` retrieved (answer SHA256
d34297dcb08e5984dba1597c72ca711f0c42350e1978d54d6935522e320b8fd3): two must-fix
boundary issues, not demonstrated arithmetic errors in unchanged water artifact.
Optional archived type-H comparison aborts output if H1/H2 traces differ; change
it to explicit comparison-unavailable status without selecting a representative.
NEW parser must enforce required ENDFILE and reject leading/repeated/premature
termination/data after termination except a new enclosing INDEX section.
Task17 blocks bridge16; fixes/tests will begin after active native measurement
fingerprints finish, avoiding mid-run source mutations. Prior artifact/results
remain retained and not silently rewritten.

Task17 now implements explicit section termination state and optional comparison
availability. ENDFILE is required; leading/repeated/missing/premature termination
and post-termination data without newINDEX fail. Invalid H1/H2 type equivalence
returns unavailable comparison with null error/agreement, zero checked points and
no representative selection; malformed reference syntax still fails explicitly.
Added parser+manifest negatives, valid unsectioned case, exact22indefinite-tensor
assertion, and actual perturbed-H CLI regression preserving computed/raw outputs.
Parent `b6d9dfc53` stages only the updated Python module, runs focused/full suites
and attempts fresh `orient-bridge-supplied-properties-v2.json`. V2 stopped at
52passes/2testfailures: new mutation tests assumed single-space headers whereas
archive headers use repeated spaces, so neither constructed its intended input.
Corrected tests to locate H1 headers by tokens; production fixes/tolerances unchanged.
Rerun `ba1fe9679` PASSED **54focused tests in6.55s**, **1152combined tests
in13.67s**, and emitted `orient-bridge-supplied-properties-v3.json` (SHA256
308b73d9bc2f86374ba8cbe8a250efe2972de1c0a22909b3ef27db53b8d011c3).
Parent verified all atomic-scalar/isotropic/anisotropic numeric values unchanged
from v1. Exported `orient-bridge-{atomic-scalars,isotropic-cn,anisotropic-cn}-v3.csv`
for convenient inspection; fullJSON retains tensors/provenance/warnings.
Follow-up review `d02294397a01ff3ce4db8e4dad7d482fa` checks both boundary fixes;
review retrieved (answer SHA25694c88fde7541eae29d840390fd2a181f5890768b8bc4ce82d0b778bb5ce70117):
both must-fix findings resolved, no new boundary defect. Task17 and bridge16
complete. Public supplied API guide/SPEC updated with runnable hybrid workflow.
Portable evidence `tests/pytests/data_isapol/psi4_orient_bridge_evidence.json`
SHA256a73695529db14fd8dac61e171a7745a66bbc8bb5bb0ed214e90a7d461ff4c0e0.
Initial evidence writer incorrectly compared NPY file hashes; corrected to compare
same-dtype/shape contiguous value bytes (equal); distinct storage-layout file
hashes remain recorded, not claimed equal. NativeOVAPI task15 complete at
provisional scope; downstream captured-operator transition-leg diagnostic tracked
separately in task18. Native wavefunction/fullpipeline agreement remains unclaimed.
Failed v2 log retained; no v2 production artifact was emitted.

## Concurrent continuation — native C++ transition fitting

User requested continued end-to-end implementation after provisional Drho-C
acceptance. Task15 now scopes a production C++ OV-fit API replacing the validated
native-integral/supplied-C NumPy diagnostic, followed by native-transition-leg
error propagation through explicitly supplied response operators. Read-only
integration audit `d46c7a0d62c80aee97a8acd33b8ba8457` is active; preserve inspected
libisapol/FDDS sources, SPEC and oracle comparator/replay scope until retrieval.
No published localization substitute or premature native-response/driver claim.
Strictness restoration tasks9–12 remain deferred, not forgotten.

Native OV audit retrieved (answer SHA256632f934d302edd52ff7502f3c1a5c1e318e4b6c4af5384bec12dfae1f97bebda).
Source worker `b03b075a6` implements owned general-LU OV fit preserving occupied-fast
left-associated contractions, and an explicit CPP comparator producer mode while
retaining NumPy control. Prompt `.pi/audit/implement-native-ov-prompt.md`; handoff
`.pi/audit/native-ov-implementation-handoff.md`. No build/install by worker; parent
owns integration. Do not mutate worker-owned sources/tests until completion.

NativeOV worker `b03b075a6` completed source-only handoff. Owned IsaOvFitResult,
DGESV fit_ov and explicit CPP comparator mode/tests are written; only syntax and
pure selector tests were run by worker. Parent compile-only `bf2305610` writes
`native-ov-cpp-build-v1.log`. Independent review
`de20992a3e3cf3e197dc3a791f0104a7c` is active; preserve OV source/tests/comparator
scope until retrieval. Do not stage incomplete ORIENT bridge Python files while
its worker is active. Actual native CPP measurements and combined integration
remain pending; original NumPy provisional result is not CPP acceptance.
Compile-only `bf2305610` PASSED. Parent task `b892c41cf` now tests a private copy
of the prior staged Python package plus the built OV core, with staged dependencies
second on PYTHONPATH and explicit imported-core/unchanged-shared-core checks.
Runner `.pi/audit/test-native-ov-isolated-v1.py`; logs
`native-ov-isolated-{runtime,tests}-v1.*`. This avoids installing the unfinished
ORIENT bridge or changing the shared runtime; molecular CPP comparison still pending.
Native OV review `de20992a3e3cf3e197dc3a791f0104a7c` retrieved (answer SHA256
21f85008850dc9030003370c2ee9fb341fe71cb2fe91b62699dae3fa276b09bc): no must-fix
source/API/ownership defect. Source preserves two GEMMs, occupied-fast packing,
original operands, single general-LU solve and truthful CPP measurement scope.
Optional follow-ups: adversarial association and mixed-angular tests; earlier
Drho-profile CLI rejection; comparator zero-residual denominator handling before
non-water generalization. Existing water scope remains pending actual CPP tests
and reference-forward measurement, not accepted by review alone.
Isolated runtime `b892c41cf` PASSED **72 tests in2.90s** including real OV C++
algorithm/ownership/resource tests and producer/profile plumbing. Built/private
core SHA256cf2af5507e01cbc3583bf92b17e9a29072961b6063883622a7401f0362bf72de;
shared core verified unchanged. Evidence `native-ov-isolated-tests-v1.{json,log}`.
Canonical install/full suite and fresh molecular comparison await the ORIENT
bridge worker's release of source/Python scope (comparison fingerprints include
all oracle helpers, so concurrent bridge writes must not invalidate provenance).

Separate localization retry found alpha still unavailable and HTTP403 on all three
primary DOI full-text fetches (LS1994,LW2007,LW2008); no defining equations acquired.
Evidence `.pi/audit/localization-retrieval-retry-v2.md`. Task4 remains pending, and
parent focus returns to task15. No unsupported localization policy introduced.

## Latest exception — Drho-C-only provisional 1e-2

User explicitly permits Drho-C1e-2. Task14 adds named
`provisional-drho-1e-2`, restricted to native-drho's existing forward-error
allowlist; all non-forward/structural checks remain strict. Existing1e-3 profile,
other stages and historical reports remain unchanged. Boundary/isolation and
full ISA/FDDS tests run in `bd89189cf`; fresh native Drho measurement under the
new profile runs in `bfc91f9b2`, writing fresh
`.pi/audit/provisional-parent-drho-1e-2-v1.{json,log}` and coefficient output.
Exception validation `bd89189cf` PASSED **1064 tests in6.39s**, including boundary,
stage-isolation and retained strict checks (`provisional-drho-exception-tests-v1.log`).
Fresh Drho measurement `bfc91f9b2` PASSED the1e-2 provisional profile:
coefficient scaled0.0012802188514176112, strict FAIL unchanged. Fresh JSON retains
both thresholds/statuses and input/source/core hashes. Task14 complete; TODO10
retains eventual strict parity. This exception does not
supply missing native APIs/localization or certify end-to-end properties.

## Earlier user authorization — provisional numerical gates

User now authorizes temporary **1e-3** tolerance for struggling stages, with eventual
strictness TODOs. This supersedes older blanket no-relaxation instructions below,
not historical results. Policy: `libisapol/PROVISIONAL_ACCEPTANCE.md`. Preserve strict
metrics/results, opt in separately, and keep all structural/provenance/solver checks
unchanged. Tasks9–12 track raw tails, native Drho-C, native OV and downstream
strictness restoration. Historical Drho coefficient scaled0.0012802188514176112 is
still above1e-3; no rounding or alternative metric may turn it into acceptance.
Missing algorithms/localization definitions remain separate from numerical error.

Scope audit `d24356f728635ed742b0358de5f388a14` retrieved (verified answer
SHA2569e1671ab280c31339b11c6b8ab5f1ca9b227d54bb07d727e40e9b9e845d02984).
Task8 source worker `bda89f6fb` now implements fixed strict/provisional profiles,
allowlisted forward comparisons, callable fresh native Drho/OV measurements and
strict-preserving trajectory integration/tests. Owned files named in
`.pi/audit/implement-provisional-acceptance-prompt.md`; keep them untouched until
completion. Parent performs fresh native measurements/full validation; no report
is provisionally accepted yet. Expected handoff:
`.pi/audit/provisional-acceptance-implementation-handoff.md`.

Worker completed implementation; its64 pure tests passed but no native comparisons
were run. Parent review `d4f1413e3a8a206150e8c346852116155` is active; preserve
profile/comparator/test sources until retrieval. Parent jobs now run independently:
`b5020dcbf` full ISA/FDDS integration tests PASSED **1054 tests in6.31s**
(`provisional-parent-combined-tests-v1.log`); `b956d78cb` retained raw-tail trajectory
recomparison PASSED the provisional profile (strict remains FAIL): maximum
joint-tail scaled2.3588804665973028e-8; other numerical/structural gates retain
strict requirements. This is retained-trajectory recomparison, not fresh controller
execution. TODO9 remains for strict raw-tail parity. `bfa534c28` fresh native C++ Drho
measurement finished with numerical acceptance FAILURE, not a runtime/import error.
Coefficient scaled0.0012802188514176112 exceeds1e-3; both strict/provisional flags
are false. Metric/RHS remain ~1e-16, sampled density scaled6.130116422102165e-7.
Fresh report and coefficient outputs retained; TODO10 updated. No threshold change; `b847452d8` fresh
native-integral/supplied-C NumPy OV measurement PASSED provisional1e-3, strict FAIL:
coefficient scaled2.3019798321950356e-5, sampled density5.0303437185125854e-8.
TODO11 retains strictness work. This is not production C++ OV/native response
acceptance; missing implementation status remains explicit. All output to fresh
`.pi/audit/provisional-parent-*-v1.{json,log}` paths; expected Drho failure must not
suppress OV execution. No provisional acceptance claimed until reports inspected.

Profile review `d4f1413e3a8a206150e8c346852116155` retrieved (answer
SHA25613ca2c234cc7922fc1bec64807d4d6023dc6dc0c546f2da9f96e6c9dae0d0879): one
must-fix legacy custom-tolerance compatibility regression, otherwise bounded
profiles/metrics/provenance sound. Task13 blocks task8. Parent restores custom
callable comparisons as explicitly uncertified legacy reports (no named profile
flags); custom+provisional remains rejected. Added stricter/looser custom,
structural-failure and invalid-tolerance tests. Task `b19d0ac28` now runs combined
suite and fresh v2 retained-tail recomparison: PASSED **1061 tests in6.30s** and
unchanged provisional-tail PASS/strict FAIL (max scaled2.3588804665973028e-8).
Task13 resolved. Bounded dual-status evidence recorded in
`tests/pytests/data_isapol/psi4_provisional_acceptance_evidence.json` (SHA256
38c5bf6f7e81a77930e7efedf8b4ca76286d6cd87266a0ef15a765232fb00307).
Task8 profile implementation is complete; numerical restoration tasks9–12 remain.
Native measurements
already complete: tail/OV provisional pass, Drho provisional fail. No broadened
acceptance or threshold beyond1e-3 is authorized.

## Historical independent continuation — response-cache provenance

User explicitly selected **Continue independent gates** while localization papers
remain unavailable. Task1 is active; task2's remaining property stages and task4's
published localization definition gate remain pending. No experimental LS policy,
new reference track, stabilized native fit or tolerance change was authorized.

Write-capable background worker `b65f09ddd` now implements the audited response
metric/tensor lifecycle in the isolated observer/reader scope. Authoritative prompt:
`.pi/audit/implement-cache-provenance-prompt.md`; requested report:
`.pi/audit/cache-provenance-v9-handoff.md`. Owned source scope is response capture/
replay helper scripts and their tests, plus fresh v9 cache/reference artifacts and
bounded evidence. It must not edit C++/bindings/CMake/runtime/SPEC/plan or original
reference trees, commit/reset/push, or delegate further. Parent must not mutate its
owned files before completion. It must keep schema3–7 compatibility, preserve v8,
observe both ordinary/constrained metric producer routes, and require actual
successful original metric/subset/tensor generations at consumers. Read-only
cache/metric audits referenced above are inputs, not completed implementation.
Full acceptance requires synthetic safety/state tests, a fresh serial Fortran build,
traced/untraced artifact identity and unchanged strict replay gates.

Worker `b65f09ddd` finished PARTIAL. It added experimental `response_cache.py`,
reader integration, synthetic cache tests and read-only source-anchor tests;
`capture_response.py` remains unchanged schema7. No new observer, source preparation,
Fortran build, v9 traced/untraced run or portable acceptance evidence exists. Report
`.pi/audit/cache-provenance-v9-handoff.md` explicitly records this incompleteness.
Its **145 focused tests in 1.88 s** pass; all 4,797 retained schema7/v8 events replay
with unchanged eleven-frequency/raw-fit/Hessian/symmetry results. These are legacy
compatibility and synthetic reader checks, not a fresh producer gate. Task1 is
pending behind new task6 (actual observer/build/runs). Read-only review
`d8f4c812e5d51debe5f40ee9af920ba19` retrieved (SHA256
`355da8220fe80347614e286b861d46d86468c9b7183c8e3960305444b9485008`) found two fail-open
cases: full-parent writes could alias committed subset storage, and DF_SC could
seed metric history from asserted S.done without observed ordinary-wrapper success.
Parent now separates coefficient parent/subset storage roles and requires the prior
ordinary lifecycle. Added in-memory and serialized rejection tests, explicit
`observer_validated=False`/structural labels, and an integrated rebuilt/different-H2
consumer test with a wrong H1-era negative reference. Task `ba9c8d970` PASSED
**152 focused tests in 2.04 s** and retained-v8 replay at unchanged maximum CDF
scaled5.644276459242761e-19. Legacy cache completeness and observer validation
remain false. Logs/reports: `response-v9-reader-hardening-tests-v1.log` and
`response-v9-reader-hardening-v8.{json,log}`.

Focused observer worker `bfc0645b2` is now active under task6, from
`.pi/audit/implement-cache-observer-prompt.md`; requested report
`.pi/audit/cache-observer-v9-handoff.md`. It owns response observer/reader/tests
and fresh isolated reference artifacts only, not C++/CMake/exports/staged runtime
or PFIT source. Parent must not mutate either worker's owned scope until its
completion. PFIT task5 is awaiting source worker completion/integration while
current orchestration focus is task6. No actual new emitter acceptance yet.

Independent PFIT API audit `d83b901c01cd00b112682dc5e6076f274` retrieved (SHA256
`8f7fab71ebabb5218e1cfff742315c7f7f44e2174fee64084b5ae104ffa3feed`). It supplies a
complete single-frequency explicit-target objective: within-batch j<=i rows,
full symmetric channel tensors, fixed elimination including P_FC*c, matrix PSD
prior and separately retained nonzero LC residuals. Two same-objective solvers:
column-major DSYSV/DSYCON and chunked GEQRF/ORMQR/TRTRS, with PSD square-root rows,
rank/condition diagnostics and no hidden ridge/pseudoinverse. Canonical target is
negative induced-potential derivative per unit source charge (Eh/e^2), not energy
with factor1/2; provenance must distinguish actual/fitted-propagator/synthetic
sources and must not reconstruct targets from truncated model tensors. Native
AUX point potentials via nuclear shell/unit are plausible but explicitly unverified.
Task5 is now active. Scoped write-capable worker `b8369e077` implements the complete
supplied-input PFIT C++ source and tests from that audit. Prompt:
`.pi/audit/implement-pfit-prompt.md`; requested report:
`.pi/audit/pfit-implementation-handoff.md`. Its owned edits are new pfit.h/.cc,
pfit-only tests and narrow CMake/export additions; it must not touch existing
numerics, cache-review scope, plan/SPEC/docs or staged runtime. Source worker
`b8369e077` finished and wrote `.pi/audit/pfit-implementation-handoff.md`: source and
tests complete, only syntax/AST checks performed; no build/runtime acceptance.
It implements typed value-owned single-frequency PFIT, exact physical-batch pair
sets, fixed/matrix/LC penalties, DSYSV and bounded streaming Givens QR, strict PSD,
explicit rank/condition/resource diagnostics and failed-result parameter guards.

Parent compile-only task `b6ca03346` PASSED full core linking (`pfit-compile-v1.log`). Independent
read-only review `de9c2b7d53c823457267461e623a16638` retrieved (verified SHA256
`75ab69234d38e6cd4a29732ddfbc7b8e29d436908fe2ad2a8b6a276c887d4e1d`). It found one
must-fix: LC diagnostic products applied strength before coefficients/target,
underflowing differently from the solvers' sqrt-strength rows and potentially
reporting an asymmetric matrix. Task7 now blocks task5. Added separate
`test_isapol_pfit_review.py` regressions for RHS underflow, matrix asymmetry and
unscaled residual overflow. Baseline attempt `ba60a3b3b` failed during import,
not numerical execution. Corrected baseline `be18ccd42` reproduced all SIX
expected failures in 1.45s on the pre-fix private core: zero vs1e-250 RHS,
asymmetric0/1e-200 cross entries and unscaled-dot overflow, for both solvers.
Evidence: `.pi/audit/pfit-review-regressions-red-v2.log`.
Parent uses one scaled-row helper for LC matrix/RHS, solver input and objective.
Build/private-runtime validation task `b6a972de4` writes `pfit-compile-v2.log` and
`pfit-isolated-{runtime,tests}-v2.*`. Review's additional diagnostic/condition/
ownership/resource test suggestions remain follow-up, not proven acceptance. Existing generated CMake installs contain
absolute stage destinations, so --prefix alone is NOT safe isolation. Prepared
`.pi/audit/test-pfit-isolated-runtime.py` will copy the prior Python package to a
fresh private prefix and copy only the completed new core for focused PFIT tests,
verifying the shared staged extension remains unchanged while the cache observer
worker is active. After compile success, task `bca53588a` now runs it, logging to
`pfit-isolated-runtime-v1.log`: FAILED before tests, because private-only PYTHONPATH
selected environment QCEElemental0.30.1, lacking models.v2. Canonical stage/lib
also contains QCEElemental0.50.4/QCEngine0.50.0 dependencies. The v2 runner has the
same isolation defect; preserve its attempt. Prepared v3 runner places private/lib
FIRST and stage/lib SECOND, with explicit imported-core path assertion. Launch v3
only after v2 build completion. V2 built successfully but repeated the known
import failure. Corrected fixed-core task `bb1ff08ea` PASSED **98 tests in1.41s**,
including all six previously failing numerical regressions. Private-core SHA256
`1854a6b5cc29b2b5d50042c28a0ceb60b7af281a23787d805096534020f66f8d` matches built core;
shared staged core verified unchanged. Task7 resolved; task5 active. Added full
dense H/b and separate/combined penalty diagnostic oracle, exact symmetry and PSD
correction reporting checks, analytic condition estimates/forced condition failure,
rank-failure result contracts, and exact/one-byte-below workspace boundary checks.
Expanded focused task `bdb3f7f19` PASSED **102 tests in1.45s**
(`pfit-expanded-tests-v1.log`). Follow-up independent review
`dcb7cf6b3162a9b39a6d3f31239c36c0b` retrieved (verified SHA256
`fa8516824777756d60d80ce771b4e477a4fcd9585627cfdb067c3242e792b74b`): prior LC
must-fix resolved, no new must-fix in bounded review. Optional zero-strength/extreme
LC validation tests remain suggestions, not known code defects. PFIT scope is now
released. Observer worker `bfc0645b2` timed out after2400s while writing its
handoff (missing), NOT a successful task completion. Recovered attempt6 evidence
claims4855 schema8 events,35 identical artifacts,zero disabled events,strict replay;
parent has not yet accepted those claims. Point-in-time process inspection found
no matching orphan reference build/run jobs. Parent verifier `b1223c5c4` reruns
hash/source/trace/replay checks into fresh parent-only reports: PASSED. Parent
verified current source/build/run hashes,4855 schema8 events,zero disabled events,
35 byte-identical artifacts,two metric/tensor generations,four actual consumers,
exact H1/H2 reconstruction,all11 CDFs (max scaled5.644276459242761e-19), and unchanged
legacy schema7 replay. Evidence SHA256
`12da28db95f459cf3207abc6b34fdbdd83aeb39195765c835e62a14c2ddcbee8` at
`.pi/audit/response-v9-observer-parent-evidence.json`. This recovers specific run
validation despite worker timeout; final observer source review remains pending. Independent observer
review `db0e9f438cce96c4d8d975deeb853db0a` is active: keep observer/reader/test scope
frozen. With worker processes gone, canonical install+binary cmp+full ISA/FDDS
regression task `b38b77e0b` PASSED canonical installation, built/staged binary cmp,
and **882 tests in5.69s** (`pfit-canonical-install-v1.log`,
`pfit-observer-combined-tests-v1.log`). Supplied PFIT task5 complete; API/objective,
scaled penalties, ownership and numerical/resource limitations documented in
`SUPPLIED_PROPERTIES.md`. Task6 now focuses on recovered observer review/evidence.
Native/reference PFIT acceptance is no longer absent: see the gate 7 resolution
block below, which reproduces the reference WSM refinement (all 104 parameters,
all 11 frequencies) at 2.52e-08 absolute / 6.03e-09 relative from the on-disk
deck + `.p2p` reference with no ORIENT involvement.

Final observer review `db0e9f438cce96c4d8d975deeb853db0a` retrieved: no must-fix in
bounded audit (answer SHA2568ba2b8f731bc7058ad1c0efa001192f59673a3ad12e4f286b35c11dbc7e0a287).
Together with parent verification this completes tasks6/1 at their stated bounded
reference-provenance scope. Parent recovery handoff:
`.pi/audit/cache-observer-v9-handoff.md`. Portable evidence recorded in
`tests/pytests/data_isapol/{camcasp_response_cache_v9_evidence,psi4_pfit_evidence}.json`;
SPEC updated. No native/LS/end-to-end gate has been relabeled.

Task2 now scopes the next independent anisotropic-dispersion gate. Read-only delegate
`d8368ffebbf06f8fe90549fcb0c04a2df` investigates the in-repo scientific/API contract,
conventions and missing definitions before any implementation. Its inspected
libisapol source/tests/docs remain frozen until retrieval.

Scope delegate retrieved (answer SHA25669188036c59c6965104354d34e03fca09a5e8adcd314872c2201db73603a1bb9).
User explicitly selected **Direct contraction**, not recoupled-coefficient research.
Approved boundary documented in `libisapol/ANISOTROPIC_CONTRACT.md`: supplied local
rank1–4 tensors, exact input symmetry, Coulomb-derivative interaction, orientation-
resolved scalar C6–C12 including odd orders, explicit unrestricted missing ranks,
no localization/native/recoupled output claim. Source worker `b04476d23` implements
new anisotropic files/tests and narrow additive CMake/export entries only; no
build/install or docs/cache modifications. Prompt `.pi/audit/implement-anisotropic-contraction-prompt.md`;
expected `.pi/audit/anisotropic-implementation-handoff.md`. Preserve its owned scope
until completion; parent will compile, test and independently review.

Anisotropic worker `b04476d23` completed source-only handoff at the expected path.
It added analytic degree8 Coulomb jets, supplied-local tensor models, scalar C6–C12
(including odd orders), explicit rank coverage and29 parametrized test functions.
Only syntax/AST checks were run by the worker; no numerical acceptance yet.
Parent configured build `be567dd6f` writes `anisotropic-build-v1.log`. Independent
read-only review `dbcebf536bace31b46e715180bec30cba` checks mathematics, API,
resource/ownership and independent tests. Keep new anisotropic source/tests/bindings
and contract frozen until retrieval. Configured build `be567dd6f` passed;
canonical installation/built-staged binary comparison and runtime task `ba78cbf7d`
also passed: **120 anisotropic tests in1.57s**, **1002 combined ISA/FDDS tests in6.11s**.
Logs: `.pi/audit/anisotropic-{focused,combined}-tests-v1.log`. Independent review
retrieved: no must-fix numerical/API/ownership/memory-bound defect in source
inspection (answer SHA256835cefde174dc998a03cab6dbdbb0aee0271085322fb96eacc60c156ddfbfa36).
Together with successful runtime tests, the approved supplied-local direct-contraction
increment is accepted. Canonical core SHA256
`0ab8c722bf541c709e160bc4ea86be34d7e54e9a03e9e28066fbff021c38cf82`;
bounded evidence `tests/pytests/data_isapol/psi4_anisotropic_dispersion_evidence.json`
SHA256366061bbdda5e37dfe94c4478d5e56351756b3337c5e5967ebe3e42aeb5cb5f3.
SPEC, supplied API guide and contract updated with indirect high-rank off-axis
oracle coverage, state-specific B averaging, storage-vs-time limits and conservative
arithmetic rejection. Recoupled-coefficient acceptance is no longer absent: see the
gate 8 resolution block below, which reproduces the reference `C6..C12` table (all
three site-type pairs, 10,457 printed coefficients plus 30,791 structural zeros) from
the on-disk `casimir.data` deck at `4.998e-07` relative - the `g15.7` write precision
of the reference `.pot` - with no CamCASP build and no ORIENT involvement. Native
localization (leg A) acceptance remains open; see gate 7c.
Task2 remains incomplete for published localization-dependent stages; independent
supplied stages are now validated.
Baseline `be18ccd42` uses the correct dependency path with unchanged pre-fix
private-v1 core and reproduced all six numerical failures as described above. Canonical
bash build.sh/staging and combined tests remain pending; no PFIT acceptance yet.

## Previous increment — polynomial frames; localization rejected

Localization audit `df1b55771ad105f7961798ec465bf688f` retrieved (SHA256
`6c2c6f7b0e4a0f678d3b36b192632ef3e900095d72770f5549753602f0e5cc64`). Local ORIENT
is GPLv3: do not translate its source into LGPL Psi4. That applies to ORIENT only,
not to CamCASP (MIT; see the licensing section at the top of this file). The next kernels are derived
from mathematical Legendre polynomials and translation-nullspace identities, not
source-code migration. No ORIENT parity or permission clearance is inferred.

`multipole_transform.{h,cc}` implements independent real Racah rank0–4 translation
and proper-frame rotation from Cartesian polynomial substitution/decomposition.
Tests cover independent angular evaluation, group/composition/inverse/covariance,
axial binomial translations, all rank-pair conservation and invalid/overflow inputs.
Compile-only task `b33343131` PASSED (`multipole-transform-compile-v1.log`).
Build/stage task `b0db0e1cf` PASSED build, built/staged byte comparison, staged import,
**661 tests in 4.27 s**, Python compilation and diff checks (`multipole-transform-tests-v1.log`).
The build script again returned1 only for optional stubgen/cleanup, separately
confirmed in `multipole-transform-build-v1.log`. Transform review
`d08b00275ec5891853fc2662740d84ee5` retrieved (SHA256
`da08e64a7a07f36a821b0bdb22a9d043b6ab65b3a70d4b4a70507d2cc4a25dac`) found no must-fix
defect. Added noncommuting rotations, direct Gaussian-harmonic evaluation, large
representable translations/ownership and frame-boundary checks. Final task
`b71046099` PASSED **665 tests in 4.29 s**, Python compilation and diff checks.
Portable source/core/log evidence: `tests/pytests/data_isapol/psi4_multipole_transform_evidence.json`. It also adds the review-requested scalar trace convention to the
isotropic model's Python docstring. Earlier supplied-stage evidence remains an
immutable snapshot of the previously built source/extension, not this new build.

The attempted `localized_response.{h,cc}` draft was never bound or added to CMake
and is now moved out of the source tree to `.pi/audit/rejected-ls-draft/`. It
implements sequential balanced-endpoint nullspace transfers, but calling this LS
was unsupported; no localization acceptance is claimed. A direct axial
algebra check found the audit's proposed two-site high-rank charge oracle inconsistent
with its own sequential formulas: dipole .5 matches at d=kappa=1, but local (10,20)
is .375 rather than .5, despite zero offsite response and exact full molecular
conservation. This is an unresolved scientific-policy/oracle issue, not permission
to weaken tests or silently name a variant LS. Follow-up audit
`dedb802a841338f6e474393b964675f2f` retrieved (SHA256
`f8ebe0c061bd973a9986b981080b48d8e5e646f9a7a9aa73241ef0b9f82b2fe8`) independently
derived local rank1/2 block [[1/2,3/8],[3/8,1/4]], confirming the implemented
arithmetic, but withdrew both its earlier oracle and confident LS attribution.
The general mathematical prose does not uniquely specify this published policy.
The draft is retained only as rejected evidence; do not restore/export it as LS
or an unapproved experimental substitute. Task4 now tracks obtaining defining
published equations or independently validated reference outputs. Literature search
`b4e58f83e` FAILED because `alpha` is not installed (`localization-paper-search-v1.log`).
Fallback web search located LS DOI10.1080/00268979400101261 and author-hosted LW
`https://wheatley.chem.nottingham.ac.uk/publications/papers/localpols.pdf`
(DOI10.1021/jp073151y), plus follow-up `localdisp.pdf`. Direct tool retrieval of the
author PDFs and CiteSeer mirror failed; no defining full text has been verified.
Task `bdda6c9c1` FAILED direct curl with certificate-chain verification error60
(`localization-pdf-fetch-v1.log`). Task `bb31b5409` also FAILED with error60 using
`/usr/bin/curl` and `/etc/ssl/certs/ca-certificates.crt`, keeping TLS verification
enabled (`localization-pdf-fetch-v2.log`). No original paper full text was obtained.
Never commit publisher/author PDFs as source. Need accessible original LS/LW PDFs
or defining equations; independent cache/PFIT work can continue without relabeling
localization, native fitting, or end-to-end gates.
The transform advisory review is now complete as recorded above. Preserve accepted
supplied-property and transformation code independently.

## Current continuation — supplied-partition properties and isotropic dispersion

A new supplied-partition C++ boundary is written in
`partitioned_response.{h,cc}` with bindings in `export_isapol.cc` and independent
`test_isapol_partitioned_response.py` tests. `IsaPartitionedMultipoles` integrates
molecular-AUX Q with explicit per-site screened/tail-processed shape samples,
quadrature and neighbour lists. It retains signed ratios, reports denominator
exclusions/negative ratios and labels site/component axes, origins, ranks and
provenance. Regular Racah harmonics reuse the existing Gaussian recurrence through
an explicit DALTON-to-00,10,11c,11s,... component conversion. `IsaDistributedResponse`
contracts raw `-Q C_DF Q^T` only after an explicit fitted-density-coefficient
representation declaration; no hidden metric conversion or symmetrization.
Getters return owned snapshots. Global Cartesian frames and ranks 0–4 only.
This is not native partition/response construction or a registered property task.

Compile-only task `b70e61102` PASSED (`partitioned-response-compile-v1.log`).
Baseline task `b1def0a42` PASSED **544 tests in 3.92 s** against the prior staged
extension. Subsequent task `bf753b44f` PASSED build/staging, built/staged byte
comparison, staged import verification and **580 tests in 4.04 s**, including all
36 new tests (`partitioned-response-tests-v1.log`). `bash build.sh` compiled/installed
successfully; its status1 is the inherited missing optional stubgen and unmatched
cleanup glob, separately inspected in `partitioned-response-build-v1.log`.
Python compilation and diff checks passed. Independent review
`da755282db78700ad696c8d1716c0619a` retrieved (SHA256
`e786e738c735feab4a34a815da5dbeb44c270ba3087e27c479f061cef2ee1217`) found no must-fix
implementation defect. Added recommended finite-overflow, null/symmetry/provenance,
partial-neighbour and Cartesian-p AUX tests; task `be6ccebf8` PASSED **589 tests
in 4.08 s**, Python compilation and diff checks (`partitioned-response-tests-v2.log`). No production Q/alpha reference
comparison is yet established. **Superseded below:** the full-precision reference
(`H2O_ISA-GRID_f11_NL4_fmtA.pol`) is already on disk and requires no capture; see
"Gate 6 RESOLVED as a locating problem, not a capture problem".

Cache-provenance audit `dec2516958fe0b87ba5ed1a2c2931c1b1` retrieved and verified
(SHA256 `e98aa331a25dd883528756a4f6b59a6739cbd3200d0f1b4ff5d252975edcc0e1`).
It supplies metadata-only make_D_S_D request before the done-return and completion
after done=true, frozen original subset/parent generations across cached uses,
and exact consumer associations. Crucially, using the first equal J is insufficient:
metric-wrapper successful lifecycle and dimer-subset rewrites must also be tracked
or rejected. Follow-up read-only metric audit `d29a01fd58394f6e916e2ecd00ebdfbf3`
retrieved (SHA256 `68f015b1444748b1244ac3e1cf70fcba6a9c4ec54afce3417d97df020e7f0619`).
It identifies the second unconstrained-S rewrite through make_s_matrix_constraints,
requiring common producer lifecycle plus both ordinary/constrained wrapper contexts;
dimer-subset and geometry/rotation rewrites must be tracked or rejected. Exact hooks
and state-machine details are in the verified delegate artifact. Capture/reader and
reference code remain unchanged; complete cache provenance remains open.

Independent `isotropic_dispersion.{h,cc}` and bindings are now written with typed
supplied scalar local models, all A/B site pairs and C6–C12 included/missing rank-pair
coverage. Explicit cp_weights include 1/(2*pi); static nodes require zero weight.
No localization, anisotropic engine, native generation or public task is implied.
Tests use independent rank factors, analytic Lorentz-oscillator integrals, truncated
rank models, A/B exchange, ownership and invalid/overflow inputs. Compile-only task
`b45117923` PASSED. Build/stage task `b254eedbf` PASSED build, built/staged byte
comparison, staged import and **630 tests in 4.10 s**. Build script status1 again
comes only from inherited optional stubgen and cleanup failures; actual core build
and install completed. Review `d7ecde44b8083fb0a4bf376df6f21c44c` retrieved (SHA256
`dfe2b8d0025ea0e712fd8137456c4aea5ee486fdcc51fcfedd2aed5729b28d9a`) found no must-fix
bug. Its requested sum/coefficient overflow, general 2x2 pairs, metadata lifetime,
wrong columns/infinity/static-only and max-order tests were added. Final task
`bdfb17cf3` PASSED **639 tests in 4.16 s**, including the new synthetic composed
Q/response/scalar/analytic-C6 chain, Python compilation and diff checks.
`SUPPLIED_PROPERTIES.md` documents the expert contracts and explicit non-native scope.

Task `b2aadadc7` PASSED four unchanged SAPT fixtures in fresh
`sapt-regressions-supplied-properties-v1/` (18.75/74.55/9.91/6.54 s), followed by
**4 GRAC tests in 50.31 s**. Logs `supplied-properties-sapt-v1.log` and
`supplied-properties-grac-v1.log` retain results. Portable bounded evidence with
source/core/output hashes is `tests/pytests/data_isapol/psi4_supplied_properties_evidence.json`.
No current test process holds this staged extension; the next build may stage only
after its own compile success. Supplied-stage validation is complete, not the broader
localization/native/end-to-end objective. Read-only localization audit
`df1b55771ad105f7961798ec465bf688f` checks full-rank LW/LS equations, graphs,
translations and available ORIENT references for the next implementation boundary.
End-to-end, native-fit and raw-tail gates remain open at unchanged tolerances.

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

**#14 RESOLVED by source reading — the failing O amplitude is the O exponent error,
propagated through Fit-3's own formula with no free parameters.**

CamCASP's Func-1/Fit-3 tail (`~/gits/CamCASP/src/stockholder.F90`,
`shape_function_tail_fit1`, `ShapeFuncFitType` `case(3)`) is a two-step construction:

    r1 = R_Slater(Z) * w_tail_r1_multiplier        (:4592; multiplier 1.5 at :54 and :893)
    b  = - w'(r1) / w(r1)                          (single-point logarithmic derivative)
    A  = Qw(r1) / QwL(r1,b)                        (charge conservation beyond r1)
    QwL(r1,b) = 4*pi*b^-3*(b^2 r1^2 + 2 b r1 + 2)*exp(-b*r1)      (`integrate_wL1`)

with `R_Slater` from `src/atoms.f90` (:40 H = 0.50 Ang, :47 O = 0.60 Ang, converted to
Bohr at :125). Our converged `r1` reproduces that recipe exactly: O
`1.7007534573028988` against `1.5 * 0.60/a_o = 1.7007535205291264`, H
`1.4172944914341925` against `1.4172946004409388`.

`Qw` does not depend on `b`, so `dlnA/db = -dln(QwL)/db` in closed form. At O's
converged `b = 2.7027018668938383` and `r1 = 1.7007534573028988` that sensitivity is
`2.2217797`, and the observed exponent error `db = 1.0608428e-08` predicts

    dA = A * 2.2217797 * db = 1.634388e-07   versus observed  1.635720e-07

**agreeing to 0.08% with nothing fitted.** The amplitude gate therefore carries no
information the exponent gate does not already carry; the two H atoms bracket the
prediction (`observed/predicted` 1.231 and 0.712) as expected at the 1e-9/1e-10 level
where independent roundoff in `Qw` also contributes.

The exponent is itself an amplification, not a defect. `b = -w'(r1)/w(r1)` is evaluated
at a single point 1.70 a0 out in the tail, so it divides a shape-function error by a
small `w(r1)`: O's shape coefficients agree at relative `2.812408e-11` while
`db/b = 3.925120e-09`, a **139.6x** amplification, and the amplitude compounds it to
**838.7x**. A raw `(A,b)` gate at scaled 1e-9 on the O tail is therefore a demand for
`~1.2e-12` relative agreement in the ISA-A coefficients after 53 nonlinear iterations —
i.e. beyond what the converged fixed point determines.

Reproduce with `python -P .pi/audit/tail-amplitude-sensitivity.py [--json OUT]`;
evidence retained as `.pi/audit/tail-amplitude-sensitivity.json`. It reads only the
existing `cpp-controller-ordered-charge-{trajectory,strict-comparison}.json`, so it
re-derives the attribution without another 4-minute controller run.

*Consequence for the gate.* Gate the primitives and the represented function, not the
derived pair: the shape coefficients/W (already passing at `2.81e-11`), the charges
(`3.63e-11`), and the represented tail density on the 10,001-point radial grid (already
measured at `-3.223395772522508e-9`). Keep raw `(A,b)` as a reported diagnostic with the
`dlnA/db` sensitivity printed beside it so the number is interpretable. No algebraic
rearrangement of the tail kernel should be selected to chase a lower `(A,b)` error — the
previous ordered-sampling and ordered-charge rounds already established that ordering
changes move this number without changing the represented tail, and the standing
instruction that "no further algebraic rearrangements will be selected just for lower
final error" still holds.

**Gate 6 RESOLVED as a locating problem, not a capture problem — the full-precision
CamCASP distributed-alpha reference is already on disk, and its own noise floor is
three orders of magnitude below a 1e-9 gate.**

plan.md previously recorded "No production Q/alpha reference comparison is yet
established." That was read as "the reference must be captured, which needs source
instrumentation." Reading `~/gits/CamCASP/src/polarizability.F90` shows otherwise: stock
CamCASP writes the distributed polarizability matrix at full double precision on **every**
distributed run, with no instrumentation and no keyword.

    :609   write_pols_format_A = write_pols_format_A .or. dist_pol   (unconditional)
    :224   PolFileSuffix_format_A = '_fmtA.pol'
    :225   PolFileSuffix_format_B = '_fmtB.pol'
    :602   DEFAULT prefix  <mol>_<DistPolAlgorithm>_f<num_freq>_NL<rank>
    :1939  write_pols_to_file_format_A, fmt1 = '(1p,4e24.16)'  -> 17 significant digits,
           row-major over `tot_dist_comp`, 4 per line, one `# INDEX nnn  w2 = ...` block
           per frequency
    :1974  write_pols_to_file_format_B, fmt '(3x,<ncols>(e15.7,1x))' -> 7 digits only

So **format A is the gate-6 reference and format B is not**: measured against format A,
the format-B blocks agree only to `3.658012e-07` (ISA) / `3.933494e-07` (DF) relative,
exactly the e15.7 truncation. Format B stays useful for a different reason — its header
line is self-describing (`POL SITE-LABELS s1 s2 SITE-INDICES i1 i2 RANK l1min : l1max BY
l2min : l2max FREQ2 w2 CARTSPHER <val>`), which is how format A's row/column ordering is
pinned down below without guessing.

*The right run.* `dist_polarizabilities` dispatches on `DistPolAlgorithm` (:1271) to
`dist_polarizabilities_DF` (the default, :219) or `dist_polarizabilities_ISA` (:1289).
Our `IsaDistributedResponse` implements the ISA contract, so the correct reference is the
ISA-GRID run, **not** the Drho-C production run that anchors gates 4 and 5:

    ISA:  .camcasp-reference/work/H2O-isagrid/OUT/H2O_ISA-GRID_f11_NL4_fmtA.pol
    DF:   .camcasp-reference/work/H2O/OUT/H2O_0.0005_1000_f11_NL4_fmtA.pol

Both are 11 frequencies x 75 x 75, `CARTSPHER S`, `RANK 0 : 4`, i.e.
3 sites x `tot_comp = lmax(lmax+2)+1 = 25` — confirming our spherical component count
against the reference's own header rather than against our derivation of it.

*A bit-level discriminator confirms which routine produced which file.*
`dist_polarizabilities_ISA` loops only `site_a >= site_b` and fills the mirror with
`Pol%matrix(col_block,row_block) = transpose(alpha)`, so its off-diagonal site-pair blocks
must be **bit-exact** transposes; the DF routine computes both orderings independently and
cannot be. Measured: ISA off-diagonal transpose residual `0.0` at every one of the 11
frequencies, DF `7.430856e-11`. The source reading is therefore confirmed at the bit level,
and with it the standing conclusion that our `IsaDistributedResponse::reciprocity_errors()`
is **strictly stronger than anything the ISA reference can validate** — the reference
imposes the symmetry and so cannot disagree with itself.

*The reference's own floors, measured.* The diagonal site blocks `alpha(a,a){t,u}` are
computed directly and never symmetrised (the routine header states "NOTE: this makes t and
u inequivalent!"), so their asymmetry is genuine reference noise: `3.237149e-13` relative,
maximum over all frequencies. Separately, water is C2v, so the two hydrogens must agree on
rotationally invariant site quantities; they differ by `3.637547e-08` (`alpha_00`) and
`2.447790e-07` (isotropic dipole). **This is the first gate whose floor lies well below its
tolerance:** unlike the raw-tail (gate 4) and Drho-C/OV coefficient (gate 5) legs, a 1e-9
relative site-by-site gate against format A is not obstructed by the reference — the
`3.24e-13` asymmetry is the only floor intrinsic to a site-by-site comparison. The
`2.45e-07` H/H figure is a different bound and must not be conflated with it: it bounds how
well any *independently gridded* ISA solution can reproduce this reference, so it is the
level to expect once our own grid, not the reference's, generates the weights.

*Conventions validated against an independent number.* The reference output documents the
component ordering explicitly (`H2O-isagrid/OUT/H2O.out`: "l=0: 00 / l=1: 10 11c 11s /
l=2: 20 21c 21s 22c 22s / ..."), which is exactly the ordering `partitioned_response.h`
claims. To check the ordering, the Racah normalisation and the charge-flow content of the
distribution at once, translate the distributed rank-0/rank-1 blocks to the printed origin
— for `mu = sum_a (mu_a + q_a R_a)` this needs no Racah translation matrices:

    alpha_ij(0) = sum_{a,b} [ alpha^{ab}_{mu_i mu_j} + R_{a,i} alpha^{ab}_{q mu_j}
                              + alpha^{ab}_{mu_i q} R_{b,j} + R_{a,i} R_{b,j} alpha^{ab}_{qq} ]

Against the separately printed total (`Order: 1 by 1`, origin `(0,0,0)` Bohr, `xx
10.23728  yy 9.001031  zz 9.625734`, isotropic `9.621348`), the ISA file translates to
`xx 10.237244  yy 9.001218  zz 9.625729`, isotropic `9.6213970` — **relative errors
`3.51e-06`, `2.08e-05`, `5.21e-07`, isotropic `5.06e-06`**, with the largest off-diagonal
`1.17e-07`. The residual is not a floor on gate 6: the total polarizability comes from a
*separate* `Polarizability` block of the same deck (`Rank 2`, `Quad 10`, `Invert No`,
static only), so its propagator settings differ from the distributed block's. It is a
convention check, and an ordering or normalisation error would show up here as O(1), not
O(1e-5). Two collateral facts fall out. First, the naive sum of the dipole-dipole blocks is
**not** the molecular polarizability when the distribution carries local charges: ISA gives
`7.086237` and DF `9.206003` against the true `9.621348`, and only the charge-flow terms
close the gap. Any future sum-rule check must translate. Second, the DF distribution does
not satisfy the same sum rule nearly as well (`3.76e-03` isotropic) — another reason the DF
file is not our reference.

Reproduce with `python -P .pi/audit/gate6-reference-layout.py [--json OUT]`; evidence
retained as `.pi/audit/gate6-reference-layout.json`. It reads only the two reference
`.pol` files, so it needs no CamCASP rerun.

*Consequence for the gate.* Gate 6's remaining work is a comparison harness, not a
capture: parse format A (`# INDEX` blocks, `(1p,4e24.16)`, row-major over site x Racah
component), and compare our `IsaDistributedResponse` output site-block by site-block at
each of the 11 recorded `w^2`. Gate the `3.24e-13`-floored quantities at a stated
tolerance rather than screening; report the H/H `2.45e-07` alongside so a grid-limited
disagreement is not misread as an algorithm defect. The prohibitions from (d), (e) and #14
carry over unchanged: no symmetrisation, no rescaling, and no recalibrating the tolerance
to whatever the current build produces. Note also that the ISA-GRID reference deck is a
*different deck* from the Drho-C production run — do not compare an ISA alpha against a
Drho-C propagator and attribute the difference to the contraction.

**Gate 7 RESOLVED the same way as gate 6 — a locating problem, not a capture problem —
and the reference refinement is now reproduced from first principles at `2.52e-08`
absolute / `6.03e-09` relative on all 104 parameters at all 11 frequencies, with no
ORIENT involvement whatsoever.**

plan.md previously recorded "No native/reference PFIT acceptance." Reading
`~/gits/CamCASP/bin/localize.py` (694 lines) shows the gate-7 pipeline is three
*separately gateable* legs, and that only the first of them is off limits:

    leg A  localization (LW or LS)   ORIENT      GPLv3, must not be read
    leg B  WSM refinement            process + pfit   MIT, readable
    leg C  dispersion integration    process + casimir MIT, readable  (gate 8)

`localize.py` hard-requires ORIENT (`subprocess.check_output("type orient")`, exit 1
otherwise) and drives it through the `{name}.ornt` template once per frequency. That
makes leg A's *algorithm* unreadable — but not its *result*: the concatenated
localization output is on disk as
`.camcasp-reference/work/H2O-isagrid/H2O_L3_0f10.pol` (151,602 B, 539 lines,
`ALPHA H2O SITE-NAMES O O RANK 1 TO 3 INDEX n FREQSQ w2`, 15x15 blocks at 12
decimals, sites O/H1/H2). Leg A therefore has a complete file-in/file-out oracle
(`H2O_NL4_{000..010}.pol` -> `H2O_L3_0f10.pol`) that needs neither the ORIENT binary
nor a single line of ORIENT source. **Leg A is now implemented and gated against that
oracle at all eleven indices** (worst `|delta| = 8.24e-13` over 7,425 entries against a
`1e-11` gate; see gate 7c below and SPEC "Leg A is closed").

*Negative result, recorded so it is not re-attempted:*
`~/gits/CamCASP/src/not_used/localize.f90` (3,590 lines, in no build file) is **not** a
leg-A route. Despite the name it localizes the frequency-dependent density susceptibility
in an auxiliary basis (`B^s_{kl,rs}` projection coefficients, seven solvers, two sum
rules) and contains zero mentions of LW, LS, multipole redistribution or bond transfers.
An earlier revision of this file described it as "a candidate ORIENT-free route for leg
A"; that was wrong.

*Leg B is gateable today, independently of leg A*, because the localization result
enters `pfit` only through the deck's anchor column. The whole hermetic chain is on
disk: `H2O_ref_wt4_L3_{000..010}.data` (the decks, 6,662 B each) plus
`H2O_{000..010}.p2p` (the point-to-point response, 2,020,012 B each) in, and
`H2O_ref_wt4_L3_{000..010}.pol` + `.out` out.

*The objective, from source.* `src/pfit/process.F90:118-201 setup` minimises

    S(z) = sum_batches sum_{i=1..nq} sum_{j=1..i} [ v(i,j) - sum_k z_k a_ij(k) ]^2
           + sum_{k,l} (z_k - a_k) strength(k,l) (z_l - a_l)

    a_ij(k) = sum_{entries p of parameter k} [ T(i,t_p,a_p) c_p T(j,u_p,b_p)
              + (if (t_p,a_p) /= (u_p,b_p)) T(i,u_p,b_p) c_p T(j,t_p,a_p) ]

with normal equations `c = A^T A + strength`, `rhs = A^T v + strength * anchor`,
solved by `dsysv("L")` (`:233 solve`, `NB=72`). Only lower-triangle point pairs are
used and the diagonal `v(i,i)` is counted once — which is **exactly** the pairing our
`IsaPfitBatch::targets` already documents (`i*(i+1)/2+j`, every `j<=i` once), so
`pfit.h`'s data layout, its `IsaPfitMatrixPenalty{matrix, anchor}` and its
`NormalEquationsDSYSV` solver are a faithful structural match to the reference and
need no change. The knobs `localize.py` exposes that our `IsaPfitOptions` lacks
(`Loc algorithm`, `Weight`, `Weight coeff`, `Pol Cutoff`, `Limit`/`WSM-Limit`/
`H-Limit`, `SVD threshold`, `NoRefine?`) belong to `process`, the *deck generator*,
not to the solver; their absence from the solver's options is correct.

*The `.p2p` binary format, decoded and byte-verified.* Fortran sequential unformatted
(`src/pfit/points.f90:read_points/read_resp`): record 1 is one integer `n` (=500);
then `n` records of 3 doubles (point coordinates); then, per frequency block,
`n(n+1)/2` records **each holding a single scalar** `v(i,j)` for `i=1..n, j=1..i`.
Predicted and actual sizes agree exactly for both shapes —
`12 + 500*32 + 125250*16 = 2,020,012` B (split) and
`12 + 500*32 + 11*125250*16 = 22,060,012` B (the 11-frequency
`H2O_lim2.0_4.0_p2000_f11.p2p`). The "p2000" in the filename is the point-generation
*request*; `MAXQ` is only an allocation bound. `read_resp`'s own comment fixes the
sign convention — "v(i,j,b) is the response at point i to unit charge at point j" —
fitted as `v = T alpha T` with positive alpha, i.e. our
`IsaPfitTargetConvention::NegativeInducedPotentialPerUnitSourceChargeAtomicUnits`.

*Every remaining knob pinned to code, then checked against the reference's own files.*

  - Parameter set (`src/tools/process_data.F90:1894-2391 write_pfit_local_symm`): one
    reference site per unique *type* with `COPY` for equivalents (`H2 H2 COPY H1 H1`
    at `H2O.pdef:107`), `ncomp=(lim+1)^2`, upper triangle `col=row..ncomp`, included
    iff `abs(pol(1,1,RefSite,row,col)) > cutoff`. The screening frequency index is
    **hard-coded 1**, not the deck's FREQ index, so the parameter set is identical
    across all 11 decks — confirmed by the 11 decks all being 6,662 B, and by our
    reproduction reusing one frequency-independent design matrix for all 11.
  - Weights (`process_data.F90:1810-1875 weights`): scheme 4 gives `wt_coeff` iff both
    ranks <= 1 and 0 otherwise, then `wt = wt/(1+freq^2)` for `freq /= 0`. The decks
    carry exactly 7 nonzero strengths (O 10-10, 11c-11c, 11s-11s; H1 10-10, 10-11c,
    11c-11c, 11s-11s) — precisely `case(4)`'s rank-1-only prediction — with a single
    distinct value per deck: `1.000000e-03` at `w2 = 0` and `8.7934591e-04` at
    `w2 = -0.1372089` (`1e-3/(1+0.1372089)`), matching the law to 8 digits across all
    11 frequencies and simultaneously confirming that the frequency ordering is
    ascending in `|w2|` and that `freq` is the real magnitude `w`.
  - Penalty realization (`process.F90:520-659 read_penalties`): the deck's
    `name anchor strength` form does `strength(p,p) += s`, `anchor(freq)=a` and
    `z(freq)=a` (initial guess), so the realized penalty is `s*(z-a)^2`. The module
    header at `process.F90:1-60` documents it as `[(p-a)*s]^2`. **This is a genuine
    doc/code discrepancy inside the reference; the port follows the code**, and our
    reproduction is bit-consistent with the code reading, not the comment.
  - Local axes (`src/pfit/axes.f90`): `C(i,j)` is the direction cosine between global
    axis `i` and local axis `j`; the first axis is normalised as given, the second is
    Gram-Schmidt-orthogonalised against it, the third comes from a *signed* cross
    product (`i3=6-i1-i2`, `s=+1` iff `mod(i2,3)==mod(i1+1,3)`). For this reference
    `H2O.axes` yields `sm(O)=sm(H2)=I` and `sm(H1)=diag(-1,-1,1)` — the C2v-related
    frame that is what makes `COPY H1 H1` legitimate, and the reason H1's parameter
    list admits `10-11c` (local mirror plane `xz`) but not `10-11s`.
  - T functions (`process.F90:451 T_functions`): `x = q_i - s_a`, rotated by
    `matmul(x, sm(:,:,s))`, then irregular solid harmonics `r^(-k-1) C_kq` from
    `solidh(x,y,z,-l(tmax),...)` with `l = (0, 3x1, 5x2, 7x3, 9x4)` and Racah real
    ordering `00,10,11c,11s,20,21c,21s,22c,22s,30,...` (`src/pfit/alphas.f90:25`).
    Tang-Toennies damping is applied per rank block only when `damping > 0`; it is 0
    in this reference. `tmax = 16` for rank 3, so `solidh` is called with `J = -3`.
  - Anchors are the *localized* values truncated to 8 significant digits by `process`:
    across all 104 parameters and all 11 frequencies, deck anchor vs
    `H2O_L3_0f10.pol` agrees to `4.91e-06` absolute / `4.69e-08` relative, exactly the
    8-digit write. The `parameters 170` in the deck's `Allocate` block is `MAXP`, not
    `np`; `np = 104` (38 O + 66 H1, with H2 supplied by `COPY`).

*The reproduction.* `.pi/audit/gate7-pfit-reference-objective.py` transcodes `solidh`,
the `axes.f90` direction-cosine convention, the `.pdef` grammar with `COPY`, the
`.p2p` reader and `setup`'s normal equations into Python, and solves each of the 11
decks against its own `.p2p`. Against the reference `H2O_ref_wt4_L3_{tag}.pol`:

    worst over 11 frequencies, 104 parameters   max abs      max rel
      normal equations (dsysv equivalent)       2.519e-08    6.031e-09
      augmented QR on [A; sqrt(strength)]       1.090e-08    -

Static case: `SSR 1.098e-06` fitted against `3.705e-06` at the anchors, penalty
`1.422e-07`, `rms(v) 1.089e-03`, `rms(residual) 2.961e-06`. Reproduce with
`python -P .pi/audit/gate7-pfit-reference-objective.py`; evidence in
`.pi/audit/gate7-pfit-reference-objective.json`. It reads only reference data files —
no CamCASP build, no ORIENT, no rerun.

*The conditioning, measured — this is the finding that sets the gate.* `pfit` ships a
`condition` routine (`process.F90:205`, DPOTRF + DPOCON) but **its call site is
commented out** (`! if (freq .eq. 0) call condition`), so the reference never printed
an rcond and the identifiability analysis had to be done independently, exactly as for
gate 5. Measured on the static deck: `sigma_max(A) = 5.316e-02`,
`sigma_min(A) = 6.955e-07`, `cond2(A) = 7.644e+04`, hence
`cond2(A^T A + strength) = 5.213e+09` (range `3.854e+09 .. 5.213e+09` over the 11
frequencies). In guaranteed-digit terms that is **10.8 digits on a QR path against
5.9 on the normal-equations path** — a real 4.9-digit advantage for our
`IsaPfitSolver::StreamingQR` default over the reference's `dsysv`, and it shows up in
the table above as the QR reproduction being 2.3x closer to the reference than the
normal-equations reproduction despite the reference itself having been produced by
`dsysv`.

*Consequence for the gate: the invariant is the fit, not the parameters.* The
refinement moves the penalized rank-1 parameters only slightly from their anchors but
moves the unpenalized high-rank parameters enormously — static deck, `max|z-anchor|`
by rank pair:

    1-1  8.48e-03      1-2  2.96e+00      1-3  1.03e+01
    2-2  9.77e+00      2-3  1.76e+01      3-3  4.34e+01

(at the highest frequency, `w2 = -1430.637`, the same spread collapses to
`4.71e-06 .. 5.36e-02`, i.e. the ill-conditioning is a static/low-frequency
phenomenon). So a gate stated on high-rank *parameter* values against a
localization-derived anchor is meaningless — those parameters are not individually
identifiable from 500 points, and the reference's own numbers for them are whatever
`dsysv` returned at `cond2 = 5.2e+09`. Gate 7 must therefore be stated as:

    7a  leg B solver acceptance (available now, hermetic, no ORIENT):
        given the reference deck + .p2p, reproduce H2O_ref_wt4_L3_{tag}.pol to
        <= 3e-08 absolute / 1e-08 relative on all 104 parameters at all 11
        frequencies.  MEASURED: 2.52e-08 / 6.03e-09.
    7b  anchor/localization linkage: deck anchors equal H2O_L3_0f10.pol to the
        8-significant-digit deck write.  MEASURED: 4.91e-06 / 4.69e-08.
    7c  leg A localization: file-in/file-out against H2O_NL4_{tag}.pol ->
        H2O_L3_{tag}.pol at all eleven indices, 675 entries each, in each site's
        local frame.  Gate 1e-11 absolute.  MEASURED: 8.24e-13 worst over 7,425
        entries (tests/pytests/test_isapol_lw_leg_a.py, 40 tests).  Implemented in
        psi4/src/psi4/libisapol/lw_localization.cc without reading ORIENT source;
        recipe taken from CamCASP's MIT input template H2O.ornt, and the
        report-not-gate reading of `Sum-rule test 1e-7` from ORIENT's printed
        output log H2O_L3_000.out (an output artifact, not source).
        The gate consumes recorded NL4 literals, so it certifies the transform and
        not native production of its input; the fit-quality invariants (SSR against
        v, the rank-1 block) belong to leg B and are unaffected.
        Structural change this required: IsaLocalizationResiduals now separates the
        algorithm-controlled residuals (off_site, reciprocity, molecular_sum, and the
        new charge_sum_transport, all still gated at 1e-6) from the supplied input's
        own charge-flow sum-rule defect (input_sum_rule, plus charge_sum/local_charge
        which only reproduce it), gated separately by the new
        input_sum_rule_tolerance.  LW's bond transfers are antisymmetric in the site
        slots, so those sums are exact invariants: charge_sum_transport <= 2.2e-16 at
        all eleven indices.  The old single gate was rejecting every recorded
        reference input on a property of its producer.  No comparison tolerance was
        relaxed; the static-only 1e-3 historical waiver is no longer needed for the
        reference input (index 0 now passes at the stricter 1e-6 algorithm gate) and
        was not extended to the ten dynamic nodes.

*The precision ceiling on 7c, and why it differs from gate 6's.* `localize.py`
defaults to `--format NEW/B`, so the localization consumed
`H2O_ISA-GRID_f11_NL4_fmtB.pol` — the `(3x,<ncols>(e15.7,1x))` 7-significant-digit
view — **not** the 17-digit format-A file that anchors gate 6. The amplification is
visible in the localized output: symmetry-forbidden elements that should be exactly
zero sit at `~1e-07` (O 10-11c = `-1.13753e-07`). Gate 7c is therefore floored near
`1e-07` for the same structural reason as gates 4 and 5, and for a *different* reason
than gate 6, whose format-A floor was `3.24e-13`. Do not state a 1e-9 gate on 7c, and
do not "fix" it by re-running localization off format A — that would no longer be the
reference chain that produced `H2O_L3_0f10.pol` and the gate-8 dispersion files.
Gates 4/5's prohibitions carry over unchanged: no rescaling, no symmetrisation, and no
recalibrating a tolerance to whatever the current build produces.

### Gate 8 (dispersion) resolution

**Gate 8 resolves the same way gates 6 and 7 did — a locating problem, not a capture
problem — and it is now the strongest reproduction on the branch: the entire reference
dispersion table (`C6` through `C12`, all three site-type pairs, 10,457 printed
coefficients) is reproduced from the reference deck at `<= 5.00e-07` relative, which is
exactly the `g15.7` half-ulp of the reference write. There is nothing left to capture and
nothing left to instrument for gate 8.**

Artifacts: `.pi/audit/gate8-casimir-reference-objective.py` (+ `.json`). Development
oracle only: it reads the MIT CamCASP tree and the on-disk reference run, is not shipped,
and no libisapol/pytest code depends on it.

The oracle is again file-in/file-out and hermetic — no CamCASP build, no ORIENT:

    in   H2O_ref_wt4_L3_casimir.data   (529 lines: Frequencies 0.5 10 / Skip 0 /
                                       Print nonzero / 3 SITE blocks of local
                                       polarizabilities / CGdir / Dispersion 12 H2O)
    out  H2O_ref_wt4_L3_C12.pot        (697,680 B; 3 type-pair blocks: O O @17,
                                       H O @998, H H @2840)
    aux  ~/gits/CamCASP/data/realcg/realcg_{j1}_{j2}   (MIT, in-tree, 1,887 coefficients)

**Transcoded kernels** (MIT License, Copyright (c) 2019 Anthony Stone; cite per ported
kernel):

| CamCASP | what |
| --- | --- |
| `src/casimir/casimir.f90:41-75` | `rlow`/`wlow`: squared Gauss-Legendre roots and weights for orders 2,4,...,18 |
| `src/casimir/casimir.f90:439-463` | `frequencies`: the `omega = omega0 (1+t)/(1-t)` transform, `halfn = n_freq/2`, `base = halfn(halfn-1)/2` |
| `src/casimir/casimir.f90:262-330` | `read_cg`: `realcg_j1_j2` parse, value `(i1/i2) sqrt(i3/i4)`, `i3<0` => pure imaginary |
| `src/casimir/casimir.f90:486-531` | `recouple`: `alpha_c` from `alpha_u` |
| `src/casimir/casimir.f90:397-435` | `cpint`: the Casimir-Polder quadrature |
| `src/casimir/casimir.f90:535-649` | `Cn`: type-pair loops, `>1d-6` print threshold, `g15.7` fields |
| `src/casimir/c{6..12}code.f90` | the generated `Cn` coefficient tables (8,210 lines) |
| `src/casimir/casimir.f90:104-124` | `label(81)` component ordering (`00,10,11c,11s,20,...,88s`) |
| `data/realcg/realcg_notes` | coupling-coefficient file format |

*The quadrature grid is closed-form, not reference-derived.* This is the single most
useful fact for the port: the 11 frequencies used by the whole ISA-Pol chain are not data
to be captured, they are `omega0 = 0.5` with a 10-point transformed Gauss-Legendre rule,
plus the static point. `frequencies` gives

    omega(i)         = omega0 (1-t)/(1+t),   tm1sq(i)       = (1+t)^2
    omega(n-i+1)     = omega0 (1+t)/(1-t),   tm1sq(n-i+1)   = (1-t)^2
    weight(i) = weight(n-i+1) = wlow(base+i),  t = sqrt(rlow(base+i))

and reproduces every `FREQ2` recorded in `H2O_ISA-GRID_f11_NL4_fmtB.pol` to the
7-significant-digit header write:

    reference FREQ2      computed -omega^2      |dev|
    0.0000000E+00        0                      0
    -0.4368683E-04       -4.36868333e-05        3.26e-12
    -0.1308617E-02       -1.30861702e-03        2.31e-11
    -0.9110199E-02       -9.11019924e-03        2.35e-10
    -0.3906323E-01       -3.90632345e-02        4.48e-09
    -0.1372089E+00       -1.37208912e-01        1.15e-08
    -0.4555098E+00       -4.55509772e-01        2.81e-08
    -0.1599970E+01       -1.59996992e+00        8.36e-08
    -0.6860443E+01       -6.86044272e+00        2.83e-07
    -0.4776034E+02       -4.77603446e+01        4.62e-06
    -0.1430637E+04       -1.43063700e+03        1.67e-06

(An earlier apparent `1.3e-05` discrepancy on the second frequency was an artifact of
comparing against `H2O_L3_0f10.pol`, which prints `FREQSQ` at 7 *decimals* -
`-0.0013086`, only 5 significant digits - not 7 significant digits.)

*The Casimir-Polder kernel.* `cpint(a,t,j1a,j2a, b,u,j1b,j2b, ip)` is

    s = sum_k weight(k) * (omega0/(pi*tm1sq(k))) * alpha_c(k,j1a,j2a,t,a)
                                                 * alpha_c(k,j1b,j2b,u,b)

i.e. the `1/2pi` Casimir-Polder factor folded with the `2*omega0/(1-t)^2` Jacobian of the
frequency transform; `ip=1` multiplies by `i` and the result is then asserted real
(`|Im| < 1e-8`). Measured over the whole reference table, the imaginary residue is
**exactly 0.0** for all three pairs — every recoupled product came out real, so a real
implementation of the kernel is sufficient and the `ip` power is a pure sign/phase device.

*Recoupling.* `alpha_c(:,j1,j2,v,m) = sum_{t=1..2j1+1} sum_{u=1..2j2+1}
cg(j1,j2)%p(t,u,v) * alpha_u(:, j1^2+t, j2^2+u, m)`, with `v` running
`|j1-j2|^2+1 .. (j1+j2+1)^2`. The `realcg_j1_j2` line format is `l1 l2 L i1 i2 i3 i4`
(all three indices 0-based; `l1`,`l2` index components of `j1`,`j2` as
`0, 1c, 1s, 2c, 2s, ...`, `L` indexes the coupled function as `00, 10, 11c, 11s, 20, ...`)
with value `(i1/i2) sqrt(i3/i4)`, pure imaginary if `i3<0`. Example: `realcg_1_1`'s
`0 0 0 -1 1 1 3`, `1 1 0 -1 1 1 3`, `2 2 0 -1 1 1 3` give
`alpha_c(1,1,v=00) = -Tr(alpha_dipole)/sqrt(3)`, from which the isotropic
`C6(00,00,0) = 2 * sum_k W_k * [-Tr(alpha_a)/sqrt(3)] * [-Tr(alpha_b)/sqrt(3)]`.

**A rank ceiling baked into the reference, not into the algebra.** `read_cg:311` and
`recouple:495` both `cycle` when `j1+j2>6`, so `alpha_c(:,4,3,..)`, `(3,4)`, `(4,4)` are
never filled and stay at their static-storage zero even though `realcg_4_3` and
`realcg_4_4` exist in the data directory. Every `c11code`/`c12code` term that needs those
blocks therefore vanishes silently in the reference. For this L3 run it makes no
difference (the input polarizabilities stop at rank 3), but a port must not "improve" on
it without saying so, or its `C11`/`C12` will not match any CamCASP output.

**Reproduction (measured).** All three site-type pairs, `C6..C12`, `J = 0..8`:

| types | sites | reference rows | nonzero values compared | zero placeholders | max abs dev | max rel dev |
| --- | --- | --- | --- | --- | --- | --- |
| `O O`  | O  O  |   979 | 1,671 |  ~5.7k | 2.715e-02 | **4.711e-07** |
| `H O`  | H1 O  | 1,840 | 3,083 | ~10.3k | 3.732e-03 | **4.873e-07** |
| `H H`  | H1 H1 | 3,466 | 5,703 | ~14.8k | 3.210e-04 | **4.998e-07** |
| total  |       | 6,285 | **10,457** | **30,791** | 2.715e-02 | **4.998e-07** |

`missing_rows = 0` (every reference row is reproduced) and
`zero_placeholder_violations = 0` (every `"     0.0"` field the reference printed is below
its own `1e-6` threshold in our reconstruction, so the structural-zero pattern matches
too). The `max abs dev` column is not the meaningful one: it sits on large coefficients
(worst is `O O 42c 30 J=7 C11 = 103.2808485` vs the reference's `103.2808`), and every one
of those round-trips exactly to the printed digits. `4.998e-07` *is* the `g15.7` half-ulp:
the reproduction is exact to every digit the reference file contains. Sanity anchors:
`C6(00,00,0)` = 26.4817671 / 4.1423169 / 0.6514697 vs 26.48177 / 4.142317 / 0.6514697.

**The reference truncates `J` at 8 and the port should not silently follow.** `C` is
dimensioned `C(6:12,81,81,0:10)` and `c11code`/`c12code` do populate `J = 9, 10`, but the
print loop at `casimir.f90:591` is `do k=0,8`. Our reconstruction produces 411 rows with
`J = 9` or `J = 10` above the `1e-6` threshold (67 / 122 / 222 by pair) with magnitudes up
to `3.1e+03` (O O) - real coefficients that the reference `.pot` simply never wrote. So:

    8a  Cn acceptance (available now, hermetic, no ORIENT, no CamCASP build):
        given H2O_ref_wt4_L3_casimir.data + the realcg data, reproduce every
        printed coefficient of H2O_ref_wt4_L3_C12.pot to <= 1e-06 relative
        (the g15.7 write precision), with an exact match on the set of rows
        above the reference's own 1e-6 print threshold and on the placement of
        its "0.0" placeholders.  MEASURED: 4.998e-07 relative, 10,457 values,
        30,791 placeholders, 0 missing rows, 0 placeholder violations.
    8b  quadrature acceptance: the frequency grid is generated, not captured -
        omega0 = 0.5, 10-point transformed Gauss-Legendre - and must reproduce
        the recorded FREQ2 list to the 7-digit header write.  MEASURED: yes,
        max |dev| 4.62e-06 on -47.76034.
    8c  J = 9, 10 coefficients have no reference to compare against, because the
        reference never printed them.  Emit them if wanted, but gate them only
        on internal consistency (e.g. rotational invariants), never claim
        CamCASP parity for them, and never suppress them silently to make a
        row-set comparison pass.

*Precision ceiling.* Gate 8's ceiling is the `g15.7` write in `Cn`'s print statements:
7 significant digits, `1e-6` relative, with an absolute `1e-6` cutoff below which the
reference records nothing at all. This sits on top of gate 7c's `~1e-07` format-B
localization floor, so `1e-6` relative is the right and final tolerance for 8a - a tighter
gate is not merely unreachable, it is unmeasurable from this reference. Do not restate 8a
in absolute terms: the coefficients span `1e-06` to `1.5e+05` in this one table.

*What gate 8 does not cover.* The `.pot` is a table of site-site dispersion coefficients;
turning it into a dispersion energy (damping, the `S`/`S-bar` angular functions selected by
the parity of `L1+L2+J`, and the sum over site pairs) is downstream of gate 8 and belongs
to gate 9. Nothing in gate 8 validates that step.

*Two of the three kernels gate 8 needs are already shipped, and both are now verified
against CamCASP source rather than against a previous run of our own code.*

**Recoupling tables: bit-exact.** `.pi/audit/gate8-casimir-reference-objective.py` parses
`c6code.f90 .. c12code.f90` from scratch (join `&` continuations, split the RHS into signed
`cpint(...)` terms with a no-text-lost assertion, evaluate `(p/q) sqrt(r/s)` in CamCASP's
own left-to-right association) and compares the result term-by-term against the committed
`recoupling_tables.{h,cc}` + `recoupling_data.inc` through the `isapol_recoupling_blocks`
binding, mapping each `(n, t-range, u-range, J)` statement to `(n, L1, L2, J)` via
`isapol_component_rank/first/last`:

    n_blocks_parsed_from_camcasp   393
    n_blocks_shipped               393
    key_sets_equal                 True
    n_terms_compared              4673
    n_rank_label_mismatches          0
    worst_rel_coefficient_dev      0.0

So SPEC section 7's "393 blocks / 4673 terms" is confirmed from the source side, exactly, and
gate 8a's coefficient input is a *shipped* artifact, not a re-derivation.

**Quadrature: bit-exact.** `CasimirGrid(10, 0.5)` (bound as `psi4.core.CasimirGrid`,
`casimir_grid.{h,cc}`, a transliteration of `SUBROUTINE frequencies`, casimir.f90:437-462)
reproduces the audit's independent transcode with `max |omega_k - omega_k^py| = 0.0` and
`max rel dev` on `cp_weight(k) = weight(k) * omega0 / (pi * tm1sq(k))` of `0.0` over all ten
points, with `omega(0) = weight(0) = cp_weight(0) = 0` for the static node. Gate 8b is
therefore a statement about shipped code: no captured frequency list is needed, and none
should be introduced.

**The remaining product gap is exactly two items.** Neither is a reference problem:

1.  *No `alpha_u -> alpha_c` recoupling kernel and no realcg data in `psi4/src`.*
    Nothing under `psi4/src` mentions realcg or Clebsch-Gordan except SPEC.md itself; the
    other tree hits are prose in `doc/sphinxman/source/sapt.rst`, the committed deck
    fixture, and `test_recoupling_reproduces_the_isotropic_c6`, which hard-codes the single
    `-1/sqrt(3)` value inline rather than reading a table. The contract to
    implement is `recouple` (casimir.f90:486-531):
    `alpha_c(:,j1,j2,v,m) = sum_{t,u} cg(j1,j2)%p(t,u,v) * alpha_u(:, j1^2+t, j2^2+u, m)`
    for `v` in `[|j1-j2|^2+1, (j1+j2+1)^2]`, with the `j1+j2 > 6` blocks left at zero
    (read_cg:309 and recouple:501 both `cycle`). `recoupling_tables` is the *second*
    stage (`alpha_c -> C_n`); this first stage has no counterpart in the port yet.
2.  *No shipped entry point computes the reference quantity.* `anisotropic_dispersion.h`
    documents `isa_anisotropic_dispersion` as "Undamped, nonretarded orientation-resolved
    scalar coefficients, NOT C_n(t,u,J)". That is a different observable from the `.pot`
    table, so the existing driver cannot be gated against `H2O_ref_wt4_L3_C12.pot` as
    written, however accurate it is on its own terms. Gate 8a needs a `C_n(t,u,J)` driver
    that consumes `alpha_c` and the shipped blocks and emits the 81x81x(0..8) table.

Until item 1 lands, gate 8a is demonstrated only by the audit script. It is not a claim
about the shipped library, and it must not be written up as one.

### Gate 9 resolution: CamCASP ships its own psi4-driven end-to-end reference

Gate 9 ("public Python and oeprop adapters, water end-to-end ISA-A properties and docs")
was the last gate with no reference identified. It resolves the same way the others did -
by locating what upstream already committed. CamCASP's own test suite contains a complete,
small, **MIT-licensed, in-tree** end-to-end water properties reference *computed with psi4
as the SCF code*:

    input      ~/gits/CamCASP/tests/H2O_props/psi4/H2O-avtz.clt   (geometry, I.P., HOMO)
               ~/gits/CamCASP/tests/H2O_props/psi4/H2O.axes       (local frames)
    driver     tests/test_H2O_props.py --scfcode psi4
                 runcamcasp.py H2O --clt H2O-avtz.clt
                 localize.py H2O --limit 2 --hlimit 1 --subdir L2H1
    reference  tests/H2O_props/psi4/check/L2H1/H2O_ref_wt3_L2_Cn.pot   (26,305 B, 402 lines)

There are parallel `nwchem/` and `dalton/` trees with their own references. **They are not
interchangeable, and the psi4 one is the only valid target for this port** - see the
measured spread below.

*The reference records its own run definition.* The `.pot` header carries the entire
localization/model parameter set, so none of it has to be inferred:

    Axes file: H2O.axes      Pol file format: NEW     Limit: 2
    WSM-Limit: 2             H-Limit: 1               Isotropic?: False
    Model file: H2O.pdef     Pol Cutoff: 0.0001       Loc algorithm: LW
    Weight: 3                Weight coeff: 0.001      SVD threshold: 0.0
    NoRefine?: False

The rest comes from `bin/camcasp.py`: `method = DFT`, `func = PBE0`, `kernel = ALDA+CHF`
(defaults at :66-68 and :902-911), basis aVTZ, and the asymptotic correction resolved at
:1237-1262. That last one is the load-bearing detail:

    ac_type is chosen by SCF code (camcasp.py:1237-1244) -
        psi4 -> GRAC,  nwchem -> CS00,  dalton -> LB94 + TANH join
    and with both I.P. and HOMO supplied, the shift is fixed (not variable):
        delta_ac = ip + homo                                    (camcasp.py:1251-1253)
    psi4 deck:  ip = 12.62063 eV = 0.463800 Eh, homo = -0.3989  ->  delta_ac = 0.064900

*Measured: this is not a tolerance-scale difference between the three references, it is a
different calculation.* The three `.clt` decks are byte-identical apart from `SCFcode` and
the `HOMO` line (`-0.3989` psi4 / `-0.3980` nwchem / `-0.33187` dalton, giving
`delta_ac = 0.0649 / 0.0658 / 0.13193`), and the AC *scheme* also changes with the SCF code.
Comparing the three committed `.pot` files over their 380 common rows:

    isotropic (00,00,0)      psi4        nwchem      dalton     spread vs psi4
      C6   O-O              19.27258    18.76416    18.26039       5.25%
      C8   O-O             410.2453    393.8576    377.1047        8.08%
      C10  O-O            4106.707    3891.527    3672.440        10.57%
      C6   H-O               5.338895    5.253435    5.200611      2.59%
      C6   H-H               1.497312    1.488211    1.496865      0.61%

    380 common rows / 526 nonzero values in every pairing; no row-set differences.
    psi4 vs nwchem : max |dev| 2.1518e+02, max rel 1.8958, 223 values > 10%, 499 > 1%
    psi4 vs dalton : max |dev| 4.3427e+02, max rel 1.8699, 332 values > 10%, 519 > 1%
    The max-rel cases are sign flips on near-cancelling H-H anisotropic terms of
    order 1e-05 (e.g. (22c,20,J=2): -1.18029e-05 vs +1.05726e-05).

Reproducible in `.pi/audit/gate9-endtoend-reference-objective.py`; measured output in
`.pi/audit/gate9-endtoend-reference-objective.json`.

Three consequences for how gate 9 must be stated:

    9a  end-to-end acceptance is against the psi4 reference only.  Upstream's own
        criterion is byte equality (`cmp`) with a max-coefficient-difference fallback
        report (test_H2O_props.py:96-135), i.e. upstream expects an exact match for a
        fixed SCF code.  Do not borrow a tolerance from the cross-code spread, and do
        not compare against the nwchem or dalton reference at any tolerance.
    9b  the AC scheme and shift are part of the gate, not an implementation detail.
        GRAC with a FIXED delta_ac = I.P. + eps_HOMO = 0.064900 Eh, PBE0, ALDA+CHF,
        aVTZ.  Scale of the sensitivity: substituting the nwchem deck's AC (CS00,
        0.0658 Eh) moves isotropic O-O C10 by 5.2%, and dalton's (LB94+TANH,
        0.1319 Eh) by 10.6%.  Those two changes move scheme and shift together, so
        this bounds their combined effect rather than isolating either; it is enough
        to show the AC cannot be treated as a free parameter.  Record the shift, the
        scheme and their provenance in the result object.
    9c  gate 9 must be stated on the isotropic and identifiable coefficients, with an
        absolute floor on the small anisotropic ones.  A relative tolerance applied
        uniformly is meaningless here: the smallest retained terms are ~1e-05 and
        sign-unstable under a change of AC scheme.

*What this reference does and does not give us.* It is complete C6-C10 for an L2/H-L1
model at weight 3 - a different model from the L3/weight-4 reference that anchors gates 6-8
- and at 26 KB it is small enough to commit, which the 697 KB `H2O_ref_wt4_L3_C12.pot` is
not. But upstream commits only the *output*: there is no `casimir.data` deck in
`check/L2H1/`, so it cannot substitute for the hermetic gate 8a input/output pair. Its role
is gate 9 (whole pipeline from geometry + axes) and, secondarily, a redistributable
non-isotropic `.pot` fixture. Because it is MIT CamCASP test data, committing it is
permitted, and requires the MIT notice and Misquitta/Stone attribution alongside it.

*Leg A is still leg A.* This chain calls `localize.py`, whose `Loc algorithm: LW` step is
ORIENT. So gate 9 sits downstream of the leg A reimplementation and inherits its `~1e-07`
format-B floor; it does not provide a way around it.

### Not the leg A route: `src/not_used/localize.f90`

Recorded so it is not chased again. This 3,590-line module was the outstanding candidate
for an ORIENT-free localization, and it is not one. It localizes the **frequency-dependent
density susceptibility in an auxiliary basis** - it builds T- and U-type integrals over
auxiliary-function pairs and solves for `B^s_{kl,rs}` projection coefficients per site
(`make_u_matrix`, `make_t_matrix`, `solve_block`, with seven interchangeable solvers:
iterative, SVD, GELSD, GELSY, GELS, GELSS, LU+iteration), then validates with two
sum rules on the I- and L-type integrals. It contains **zero** references to LW, LS,
multipoles, or redistribution, and it is not in any build file. It is an abandoned
alternative *physical* route to distributed polarizabilities - localize the propagator
directly rather than redistribute converged multipole polarizabilities - not an
implementation of the LW redistribution that `localize.py` invokes ORIENT for.

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

Task `b6bcb5088` PASSED install/byte comparison, **32 focused tests in 5.06 s**, all
four unchanged input regressions (dft1 19.21s, dft2 75.35s, api10.00s, lrc6.70s) and
`git diff --check`. Snapshot evidence with staged-module/core and output hashes:
`.pi/audit/fdds-extraction-v1-evidence.json`. This evidence PRECEDES subsequent source
changes; do not relabel it as monomer/ALDA extraction validation. Task `bb74db810`
ran the combined 13-file ISA/FDDS/pair-wiring suite against that unchanged staged build:
**436 tests PASSED in 3.13 s** (`fdds-combined-tests-v1.log`). The old staged extension
is no longer in use by that test task; staging the new build may proceed only after
its own successful completion notification.

Implemented the next source step: `core.FDDS_Monomer` privately reuses the existing
FDDS implementation via a one-system constructor; no B spaces, transformations,
QR or exchange tensors are built. Legacy pair path retains its constructor and
operation order. Monomer preflights C1/finite/dimensions/positive gaps and explicitly
rejects hybrid nov<naux. Coefficients/energies are copied and metric/overlap/R getters
return copies. Added explicit-frequency/density validation and corrected stale R /
Y-X API comments. Build-only task `b6b5d4b28` FAILED (`fdds-monomer-compile-v1.log`):
`Vector::clone()` returns a value, unlike Matrix's shared-pointer clone. Corrected
energy-cache ownership with `std::make_shared<Vector>(*source)` using Vector's deep
copy constructor. Task `b5708f7b3` PASSED rebuilding, installation and byte comparison
of the extension and both Python modules (`fdds-monomer-{compile,install}-v2.log`).
Build blocker #20 is resolved; response task #19 is back in progress. The combined staged test task has
completed, so this installation does not overwrite an extension used by that task.
Also moved `_compute_fxc` into the shared Python module and added
`FDDSMonomerResponse` with explicit bases/orbitals/D_alpha/kernel settings and
`at_frequency`. No default basis/SCF/dummy partner. Python and C++ construction
changes require staging and fresh monomer plus pair regressions; tests not yet run.
Prepared `test_fdds_monomer.py`: explicit HF/cc-pVDZ water with an explicitly chosen
cc-pVDZ-RI auxiliary basis, one-system vs existing pair metrics/projections/uncoupled
and hybrid intermediates, shared-kernel frequency responses, ownership/getter checks,
and preflight errors (dimensions, nonfinite inputs, gaps, hybrid QR size). These are
Psi4 FDDS construction tests, not the legacy CamCASP basis or native SCF parity.
Task `b16f75055` now runs all fourteen ISA/FDDS test files, then unchanged SAPT input
regressions in fresh `.pi/audit/fdds-regressions-v2/` directories, then
`test_saptdft.py` GRAC coverage and diff checks. Logs:
`fdds-monomer-{tests,regressions}-v2.log`, `fdds-grac-tests-v2.log`.
Task `b16f75055` FAILED with **444 passes / 1 failure in 3.79 s**: malformed-row
Python test called unbound `Matrix.ncol()`. Monomer/intermediate comparisons passed,
but the chained SAPT and GRAC runs did not execute. Corrected the test to
`occ.np.shape[1]` (no production/numerical change); blocker #21 tracks verification.
Task `b9439403a` repeats the full fourteen-file suite, all four retained SAPT inputs,
GRAC tests and diff checks with fresh v3 paths (`fdds-monomer-{tests,regressions}-v3.log`,
`fdds-grac-tests-v3.log`, `fdds-regressions-v3/`). Response task #19 is temporarily
pending behind #21. Task `b9439403a` subsequently PASSED **445 tests in 3.58 s**,
all four unchanged SAPT inputs (18.99/75.17/10.12/6.62 s), **4 GRAC tests in 51.49 s**,
and diff checks. Blocker #21 is resolved and task #19 is complete for the Psi4 FDDS
single-system/shared-kernel scope. Portable evidence with staged/output hashes:
`tests/pytests/data_isapol/psi4_fdds_monomer_evidence.json`. This is NOT CamCASP
response or native SCF/end-to-end acceptance; task #22 remains active.
Independent read-only delegate `d3f728a9252da1c972495637450533656` audits pinned
CamCASP response-only capture hooks for resident Hessians, fitted transitions,
processed orbital data, kernels and frequency-dependent C_DF. This plans matched-
input validation without ISA prerequisites; no reference policy changes authorized.
Delegate retrieved and verified: answer SHA256
110bc63ac6393e66cae1177ae6558d62cada57f3d7c9236313db878a0dd67ade. Task #22 now prepares
the old/internal response observer; #21 waits for running v3 validation (not resolved).

Critical response findings: SET NEW-PROP configures kernel options but does NOT select
the new driver; only NEW-PROP inside a Polarizability block does. Preserve the old
`make_prop -> densfit_prop` route. Capture H1/H2 before their final close/release
(actual inspected anchors at densfit_prop.F90:1150/1266), selected D and Ker after
successful natural projection in PropUtilities, processed energies/full C at adddiag
if resident, and C_DF after the caller naturally opens DFprop. No observer residency
operations. OV index is occupied-fast a+nocc*(r-1). Actual omega2 is negative for
imaginary frequency; Quad10 gives static plus ten imaginary evaluations.
Pinned CamCASP ALDA is numerical Slater/PW92, not the shared Psi4 gridless LDA/VWN
kernel. Also verify actual OVOV Coulomb equals D J D^T rather than assuming the DF
integral and response fits coincide. These are explicit mapping gates, not permission
to replace the old driver. Parent rechecked reference HEAD
`63b16a22b9bae597fe81ecdb8b8d91c21868c814` and zero tracked changes; delegate line
numbers are advisory and every hook will require an exact routine-local match.

Implemented initial `oracle/capture_response.py`: new numbered
`ISAPOL_RESPONSE_EVENT 1` files for J, H1/H2, processed C/E plus explicit bases,
selected kernel-projection D/K and stored fit parameters, resolved old-response
policy, and caller-resident C_DF with actual signed omega2. Observation uses resident
arrays only, `status=new`, finite checks and exact routine-local patch anchors. It
requires a clean pinned source and uses an isolated copy, not modifications to the
reference. Policy is observed after init_prop so effective exchange is resolved.

Task `b1cd647c8` prepares/builds fresh `.pi/audit/production-camcasp-response-v1`
with the established serial/FP flags; logs `response-{prepare,build}-v1.log`.
Task `b1cd647c8` PASSED preparation and serial build. The observer has not yet
passed runtime validation. Strict reader, response-only input
provenance and traced/untraced identity are pending. It deliberately does not yet
export raw OVOV Coulomb or quadrature weights, and kernel-D versus response-D
selection must still be verified before interpreting a shared-kernel comparison.

Parent verified source-supported PSI4 import: global_data.f90:275 maps PSI4 to
`g_scf_code='dalton'`, symmetry order1; molecule_parser.F90:222 explicitly says Psi4
orbitals have been transformed into Dalton form and maps that alias to `dalton`.
Thus no input relabeling is needed; this corrects the earlier unresolved support
caution, without certifying independent SCF/basis conversion accuracy.

Task `b8201dd1d` runs `.pi/audit/run-response-reference-v1.py`: fresh traced/untraced
`.pi/audit/response-v1-water-*` directories; expanded MAIN/AUX/MO/grid retained,
AtomAux/ISA removed, archived ALDA/CKS/NN eta0/lambda1 controls restored, rank2
Quad10 total-only response. No in-block NEW-PROP, lattice or perturbation task.
Inputs/executable/provenance are hashed; outputs retained and compared without
observer files/logs. Event checks require eleven CDF records plus all prerequisite
labels, but do not replace the still-pending strict reader/replay. Log:
`response-reference-v1.log`; planned identity report `response-v1-trace-comparison.json`.
Task `b8201dd1d` FAILED: traced run stopped92 at `matrix not naturally resident`,
after two J events and before any H1/H2/ORBITALS/PROJECTION/CDF event. Untraced run
and identity comparison did not execute. All v1 files retained. Do not reopen or
retain reference objects to bypass the observer guard. Blocker #23 is active;
parent capture task #22 is pending behind it.

Task `b61173ff4` prepares/builds/runs a fresh v2 observer with operand names, caller
contexts and residency flags on guard failure. Event schema2 additionally records
actual CxKernel/CxFunctional, kernel_alda, internal-Hessian code and constraints,
instead of prematurely requiring CxKernel=0. This changes only observation, not
reference options or numerics. Paths: production-camcasp-response-v2,
response-v2-water-*, response-{prepare,build,reference}-v2.log.
Read-only delegate `d4a7879f9f28a846e62e401f5d4706c92` audits true matrix-multiply
residency semantics and safe Dov/kernel producer hooks. It also checks old-driver
exchange policy and whether the .00951 'orthonormality' warning concerns fitted
pair normalization rather than actual MO overlap; no input-defect claim is made.
Task `b61173ff4` built v2 successfully but traced execution again stopped92. It
identified **PROJECTION D**, descriptor `OV part of D  Full DF sol for H2O`, with
in_memory=F, allocated=F, shape435x246. Thus release=false on the projection does
not guarantee full input residency; the observer assumption, not a response solve,
failed. Guard retained; no reference objects reopened. Producer-hook audit remains
pending and its scope is not being modified.
Task `b4cbfdb25` runs the same v2 executable with all observer selectors disabled,
in fresh `response-v2-baseline-untraced/`, using byte-identical input/MO/basis files.
This checks baseline execution independently of the observer; it cannot establish
noninterference or response parity while the traced run remains incomplete.

Producer audit `d4a7879f9f28a846e62e401f5d4706c92` retrieved and verified (answer SHA256
645e09323852b3d13bb015132a733de9378a4812ad44411cb7d39bb331413a40). matmult_types restores
operand entry residency, confirming the measured v2 failure. Source resolves OLD
kernel_alda=false, CxKernel=CxFunctional=.25, internal CKS, constrained DF: NEW-PROP
has separate settings. The .00951 warning tests fitted-pair normalization D*Iint,
not true MO C^T S C; it is not an orbital-input defect diagnosis.

Implemented schema3 producer observation: capture NN->OV outgoing rows directly
from `Afull%matrix(1,:)` before the existing fill_Aov write; preserve destination and
parent file identities plus ar/packed indices. Associate stored fit parameters at
existing setters, including cached-fit metadata updates. Fresh direct OV solves
fail closed pending a dedicated LU-block hook. Kernel is observed in its enclosing
producer before natural release, with Doo density-source identity and constraint
selector; projection records metadata only. C/E are observed at ASCII-2 production,
while adddiag separately records actual energies/fraction/basis data without C
residency assumptions. Observer module moved to low-level df_data.o to avoid cycles.
No reference object I/O or numerical policy was added.
Task `bdafa7a18` prepares/builds/runs fresh response-v3 trees and directories;
`response-{prepare,build,reference}-v3.log` retain results. Task `bae6708be` runs
`test_isapol_response_capture.py` safety/patching tests (`response-observer-tests-v3.log`).
Task `bdafa7a18` FAILED during compilation, before any v3 response run:
`molecular_orbitals.mod` was built via a .mod pattern rule, bypassing the .o-only
observer dependency. V3 preparation succeeded and all build evidence is retained.
Blocker #24 now tracks explicit observer-provider plus both .o/.mod consumer edges;
#23 waits behind it. Task `bd4ba4be8` prepares/builds/runs fresh response-v4 paths,
with schema3 observation unchanged and established compiler/numerical flags.
Observer safety task `bae6708be` PASSED **7 tests in 1.19 s** against the schema3
producer-observer source before the dependency-only fix. Checks cover absence of
reference-object I/O, metadata-only projection, separate orbital/diagonal hooks,
row serialization, unique nested-routine patching and destination protection.
These tests do not establish Fortran build success, runtime noninterference or
response parity. V4 build/run and strict temporal producer association/replay remain
pending.

Task `bd4ba4be8` PASSED v4 preparation/build, traced/untraced runs and **35-file
byte identity**. Schema3 produced 921 events: ORBITALS1, J2, FIT_METADATA19,
OV_ROW870, KERNEL_SOURCE1, KERNEL1, PROJECTION1, DIAGONAL_ENERGIES2, H1/H2 one each,
POLICY11 and CDF11. Build blocker #24 and residency blocker #23 are resolved;
capture/replay task #22 is active. Event data remain supplied reference inputs,
not native Psi4/CamCASP response parity.

Implemented `oracle/replay_response.py`: strict event counts/identities/shapes,
finite parsing, occupied-fast/packed row indices, complete chronological row
producers, stored fit metadata, kernel source and per-frequency policy association.
Kernel-projection D and later response D are retained separately. The NumPy test
oracle solves the recorded H2H1-omega2*I system; no production solver is added.
Task `b0638aed2` runs real-case replay (`response-v4-equation-replay.{json,log}`).
Task `be9c35d36` runs observer plus synthetic strict-reader tests
(`response-reader-tests-v1.log`), including evolving D producer snapshots and corrupt
records. Task `be9c35d36` PASSED **19 tests in 1.30 s**. Together with the v4
noninterference and11-frequency equation replay, this completes supplied-input
capture/replay task #22. It does not complete native response or end-to-end gates.

Task `b0638aed2` PASSED exported equation replay at all11 frequencies, dimensions
92 MAIN /246 AUX /5 occupied /87 virtual. Maximum CDF absolute/scaled errors:
1.5543122344752192e-15 /5.644276459242761e-19; maximum relative backward residual
9.72267065989958e-17. Reference reciprocity scaled error 3.971830944308396e-13;
H1/H2 asymmetry scaled 3.371803877324214e-13 /2.1956315585383325e-14. No matrices
were symmetrized. Kernel/response D are identical; all responses use OV rows from
events5–439. Actual fit metadata: norm1, option0, df_type3, done1; lambda1,
eta/gamma/gamma_DeltaZ0; gamma_H_only0, stored constrained flag0, while the actual
response use_constraints selector is1. Do not reinterpret that stored flag as
overriding the runtime selection. Orbital-producer/diagonal energies are identical.

Task #25 and background `bdb2cb6a5` independently measure native q/J/B plus supplied
full-C transition fitting at lambda1, eta0, using a NumPy solve—not a new production
response stack. `.pi/audit/measure-native-ov-fit.py` fixes the contraction order
before measurement and compares coefficients plus435 transition densities on the
existing68,310 points. `native-ov-fit-comparison.{json,log}` and coefficients retain
results; unchanged1e-9 gates and native-density/tail blockers remain open. #22 is
complete after reader tests; this native diagnostic is a separate gate.

Task `bdb2cb6a5` FAILED native OV forward parity, with report and coefficients
retained. Native J scaled error remains1.597559351988609e-14. OV coefficient maximum
absolute/scaled errors:9.040851995152366e-4 /2.3019798321950356e-5. Across435
transitions ×68,310 points, represented-density maximum absolute/global-scaled:
5.895789788880419e-6 /5.0303437185125854e-8; pointwise-scaled2.2143328081905677e-6;
relative abs-weighted L2=6.665561522827979e-8. Native A condition8.086127239307867e12,
backward residual1.541031618034847e-17. Charge difference maximum4.6431383000575255e-10;
finite-penalty transition charges are not forced to zero. This is a NumPy fit with
native integrals and supplied C, not a production transition-fit API or native SCF.
#25 is pending behind new blocker #26. Conditioning is relevant but does not yet
separate A and RHS contributions; no direct reference OV RHS has been captured.
Read-only delegate `deb2de09f1cb05dd354a73b9542dee60c` audits naturally resident
pre-LU NN A/RHS blocks and indexing/gating, with no reference-object I/O allowed.

Portable, separate evidence files:
`tests/pytests/data_isapol/camcasp_response_replay_evidence.json` (supplied replay PASS)
and `camcasp_native_ov_failure_evidence.json` (native-integral OV diagnostic FAIL).
Native Drho and legacy-tail blockers remain unchanged; end-to-end is not accepted.

Retrieved raw-operand audit `deb2de09f1cb05dd354a73b9542dee60c`, verified answer SHA256
`e70f587dee6acd2a2ce5bc44f36bf2538ce9a8a52ca5b451dc51d772251fa7d2`.
Implemented schema4 FIT_BEGIN/FIT_A/FIT_RHS_BLOCK/FIT_END observation. Arm only around
selected fresh NN lambda1/eta0/gamma0 unrefined LU; use current local par, not stale
Dpar. Generic `lineq_solver_lu` hook follows successful existing A/B reads, before
first-call factorization or later-block solve. Export original A for each block
and only valid RHS columns, excluding uninitialized allocation tails. Observer
context owns scalar/string metadata only; no molecule/matrix pointers or copies,
no additional reference-object I/O. Track fit IDs, filenames, original dimensions,
packed-pair ranges and terminal status. Both matrix_operations_types .o/.mod targets
now depend on the low-level observer module.

Task `b709b29c9` prepares/builds/runs fresh response-v5 paths with unchanged numerical
flags; `response-{prepare,build,reference}-v5.log` preserve results. Task `bfc27ef07`
runs safety and legacy schema3 reader tests (`response-fit-observer-tests-v1.log`).
Task `b709b29c9` PASSED fresh v5 build, both runs and reference artifact identity.
It emitted925 events, including one FIT_BEGIN/A/RHS_BLOCK/END each and all11 CDFs;
the other event counts match v4. Observer safety-test result remains pending.

Implemented backward-compatible schema3/4 parsing and raw-fit validation: exact
NN dimensions/policy, contiguous block coverage, repeated original-A equality,
complete successful terminal records and outgoing OV association by full-D parent.
Task `b58afce20` runs full reference NN solve/OV and response equation replay
(`response-v5-equation-replay.{json,log}`). Task `b68f0e16f` runs synthetic multiblock,
unused-allocation-width, malformed-epoch and legacy tests (`response-raw-reader-tests-v1.log`).
Task `b6c622d13` runs `.pi/audit/diagnose-native-ov-inputs.py`: actual reference-input
control, native A-only, native RHS-only and both-native substitutions, with the
same435 transitions/68,310 sample points and unchanged gates. Its reference-input
control must pass before substitutions are interpreted. Reports/logs are retained
as `native-ov-input-conditioning.{json,log}`. All three results remain pending;
previous v4/schema3 success and native forward-parity failures are unchanged.

Task `bfc27ef07` FAILED one static assertion (19 passes in1.38s): the regex matched
`call resident(B)` inside the explanatory comment forbidding that call. No actual
whole-allocation RHS read was found. Blocker #27 tracks the test-only fix: anchor
to executable call lines and test both a real-call positive control and a comment
negative control. No observer or numerical code changed. #26 waits behind #27;
its already-running numerical diagnostics remain independent. Task `b344106ca`
runs fresh observer/legacy/raw-reader tests (`response-raw-reader-tests-v2.log`).
Earlier failed logs remain retained; rerun results are pending.

Task `b58afce20` PASSED schema4 replay over925 hashed events. The actual captured
pre-LU A and full NN RHS reproduce outgoing OV coefficients exactly (maximum
absolute/scaled error0), with relative backward residual1.4376372774147477e-17.
All11 CDF replays still pass at maximum scaled error5.644276459242761e-19.
Thus the raw reference-input solve control is established without reconstructing
RHS from solved coefficients. Native A-only/RHS-only substitution results and the
corrected safety-test rerun remain pending; no native forward gate is closed.

Task `b68f0e16f` retained the same comment-regex failure (27 passes/1 failure in1.42s).
Corrected task `b344106ca` PASSED **28 tests in1.34s**; blocker #27 is resolved.
Task `b6c622d13` completed the controlled OV substitutions with a passing reference
control. Full NN solve remains exactly equal to exported OV; the OV-only RHS batch
control has coefficient scaled2.471767645468478e-11 and pointwise density2.2580666451381324e-12.
Both controls are retained rather than conflated as bit-identical.

Native A/reference A scaled error3.849410316150209e-15; native/direct-reference OV
RHS scaled error1.2463086118685851e-12. Reference/native condition(A):
8.085609034730648e12 /8.086127239307867e12. A-only coefficient scaled2.2702984609916423e-5,
pointwise density9.821019172029908e-7, weighted L2=4.8095202154329714e-8. RHS-only
coefficient scaled8.71172768389058e-6, pointwise density2.288776079310732e-6,
weighted L2=4.0423591295398635e-8. Both affect forward accuracy; relative importance
is observable-dependent. Both-native reproduces the prior native diagnostic.
#26/#25 remain pending numerical blockers, not accepted fits. No tolerance change.
Portable `camcasp_response_raw_fit_evidence.json` and `camcasp_native_ov_conditioning_evidence.json`
under tests/pytests/data_isapol retain separate supplied-control success/native failure.

Following user's continue instruction: #28 independently audits raw actual OVOV
Coulomb/exchange and projected-kernel producers for explicit H1/H2 reconstruction.
Read-only delegate `d8077c6f7e13d8c67de7e3b6073da7823` is pending; do not mutate its
capture/reference scope before retrieval. Combined sixteen-file isapol/FDDS tests
task `bbfc50127` PASSED **473 tests in3.94s** (`response-combined-tests-v1.log`).
The raw-Hessian producer audit remains pending. Native Drho, legacy-tail, native OV
and end-to-end gates remain open; passing software regressions does not close them.

Retrieved Hessian audit `d8077c6f7e13d8c67de7e3b6073da7823`; verified answer SHA256
`7afa0872c475927fe2c19d395d9d5d4c5d5f14584da77cdf3cd313257c38839b`.
Source and existing v5 metadata distinguish λ1 response/projection Dov_c from
λ0 ordinary Dov/Doo/Dvv for Coulomb/exchange and kernel density. Actual two-electron
construction is Da*(J*Db^T), not robust DF. Numerical-kernel selector belongs to
PropParameters (ALDA versus ALDAX), separately from old CKS/kernel_alda settings.

Implemented schema5 bounded consumer capture: actual OVOV and packed VVOO tensors
at existing open Hessian consumers, raw projected kernel after natural open and
before scaling, and numerical-kernel branch/cutoff/batch at its numerical producer.
No object opens/releases/retention added. Task `bd6fe31f3` prepares/builds/runs fresh
response-v6 paths (`response-{prepare,build,reference}-v6.log`) PASSED. Traced and
untraced artifacts are identical. Captured931 events: previous925 plus numerical
kernel policy1, raw projected kernel1 and Hessian tensors4; all11 CDFs completed.
Task `bd749f0f8` now runs real-case Hessian/projection/construction/CDF replay
(`response-v6-hessian-replay.{json,log}`), pending. This capture success alone does
not establish reconstructed-Hessian or native-response parity.

Reader now reconstructs H1/H2 with literal occupied-fast exchange permutations,
source addition order and observed kernel scaling, then replays CDF with those
reconstructed Hessians. It independently compares AUX kernel projection and raw
OVOV against the captured ordinary λ0 D J D^T; λ1 response-D comparison is an
explicit diagnostic, NOT an equality premise. OO/VV coefficient rows and cache
producer-generation metadata are not yet exported; numerical agreement alone
is not full cache provenance. New nonsymmetric/permutation and malformed-association
fixtures are in `test_isapol_hessian_replay.py`. Task `b5424d0bc` runs the three
observer/reader suites (`response-hessian-tests-v1.log`) PASSED **37 tests in1.41s**,
including nonsymmetric exchange permutations and malformed tensor/fit associations.
Real-case reconstructed-Hessian replay remains pending. The473-test result
applies before this schema5 expansion, not to the new changes.

Task `bd749f0f8` PASSED actual schema5 Hessian reconstruction over931 events:
H1/H2 maximum absolute/scaled errors are exactly0. AUX-kernel projection error
8.413408858487514e-15. Observed numerical selector ALDA (Slater/PW92), cutoff1e-8,
batch1000; local multiplier3. Ordinary λ0 D J D^T error1.2120721093467068e-12.
The λ1 response-D construction differs3.6258576176229074e-8, retained as a policy
comparison, not an equality gate. All11 reconstructed-Hessian CDF errors remain
at maximum scaled5.644276459242761e-19. Full NN reference fit still reproduces OV
exactly. Portable `camcasp_hessian_replay_evidence.json` records the supplied-input
scope and limitations. #28 is complete; native forward gates remain failed/open.

#29 now covers complete construction provenance and matched-input shared-FDDS
mapping. Delegate `dd263aa72e6687d10c99f68e032ad65ed` audits the existing shared Python/
C++ FDDS equations, auxiliary hybrid projection, explicit fitting-policy mapping
and required algebra tests. Its source/SPEC scope must remain unchanged until
retrieval. SPEC validation-status updates are deferred until that read-only audit
returns. Task `bef8a738e` runs the expanded seventeen-file regression suite
(`response-combined-tests-v2.log`): **482 tests PASSED in3.80s**. Shared-FDDS mapping
audit remains pending; this software-regression result does not close native
forward-parity or end-to-end gates.

Parent verified follow-up provenance anchors: make_D_S_D in df_integrals.F90 has
Da/Db real_matrix and S twoe2indx. Metadata-only request capture belongs before
`if (I%done) return`; completion after `I%done=.true.`. No arrays there are safely
assumed resident. Explicitly record optional transpose_S, cached versus constructed
generation and stored Dov/Doo/Dvv parameters (df_integrals currently imports only
coefficient objects from df_data). Existing fill_Aoo/fill_Avv row producers remain
preferred for OO/VV and kernel-density association; no new reference-object I/O.

Retrieved mapping audit `dd263aa72e6687d10c99f68e032ad65ed`, verified answer SHA256
`07936e25a4db687eb0a3f7b0a4470bf9ea215a7ad1b768097aca41032b52e197`.
It separates inherited hybrid gap/exchange ordering from auxiliary-space closure;
no equivalence or native failure should be inferred merely from rectangular B.
Added `test_fdds_hybrid_equivalence.py`: commuting control, noncommuting full-rank
first-order characterization, rectangular closed/leaking actions and effective
baseline coupling identity. Task `bc236242e` runs these plus existing FDDS/response
suites (`fdds-equivalence-tests-v1.log`): **84 tests PASSED in2.20s**. Synthetic
supplied-matrix tests confirm the noncommuting ordering difference and rectangular
leakage case, while commuting/closed controls and effective coupling pass. These
are not measurements of the water provider's hybrid equivalence. Production FDDS
code is unchanged.

Task `b4440fe6a` runs `.pi/audit/measure-effective-fdds-bridge.py` against captured
V/X/Y/H2/D1/K: Ueff contains actual Coulomb and25% exchange, then the existing helper
adds .75*K with artificial identity metric. No extra hybrid correction is applied;
this is NOT ordinary uncoupled FDDS or zero-exchange physics. Report explicitly
separates effective coefficient coordinates from the helper's fixed representation
label, compares raw outputs, logs1e-13 pseudoinverse retained ranks/residuals and
checks staged helper bytes against source. Artifacts `effective-fdds-bridge-v1.{json,log}`
are pending. No native basis/kernel/SCF construction or production solver extension
is introduced. SPEC status/limitations now reflect v6 results and this diagnostic.
OO/VV/cache-generation provenance and all failed native/tail gates remain open.

Task `b4440fe6a` PASSED effective-coordinate coupling at all11 frequencies. Maximum
absolute/scaled error1.570697349961847e-9 /5.703776809025682e-13; baseline backward
residual7.060358956574724e-17; reference fixed-point residual1.1600694987670985e-14.
All246 modes retained at inherited rcond1e-13; maximum denominator condition980.0000210013706.
Portable `psi4_effective_fdds_bridge_evidence.json` keeps artificial-coordinate and
exchange-in-baseline caveats explicit. This is not inherited-hybrid or native
provider equivalence.

Task `b659cb81d` now measures the inherited five-intermediate hybrid formulas on
captured S=X+Y, E=Y-X, gaps and full Coulomb/local coupling with B=Q=R=J=I in the
435-dimensional OV space. Thus no auxiliary projection is present. It reports
reference full-OV/CDF control, raw hybrid deviations/reciprocity, retained ranks,
commutator and literal versus independently expanded physical fixed-point residuals.
No production code is modified. `.pi/audit/measure-full-ov-hybrid-ordering.py` and
`full-ov-hybrid-ordering-v1.{json,log}` retain this diagnostic.

Task `b659cb81d` completed with passing reference control but FAILED inherited
hybrid/full-OV equivalence on captured operators: maximum CDF scaled2.1917554073716218e-3,
raw operator scaled2.730911805796228e-2 and raw CDF reciprocity2.4656933821703216e-4.
Reference full-OV/CDF control2.0571744441538628e-14; physical identity residual
1.1652983873218173e-15 versus literal inherited fixed-point residual9.886526934317025e-4.
All435 modes retained; maximum denominator condition3.969227437166417. Gap/exchange
commutator Frobenius norm15.658940146534606. Thus this supplied-operator mismatch
is not auxiliary projection or rank truncation. It is NOT a native-provider test.
Portable `psi4_full_ov_hybrid_ordering_evidence.json` preserves scope and failure.

New #30 tracks the explicit method decision; #29 waits behind it. An exact-full-OV
route would belong in the shared response layer and reuse auxiliary coupling,
with separate policy/result labels and unchanged legacy SAPT defaults. No such
production extension or legacy arithmetic change has been made. User decision
requested before implementing it; provenance-only continuation remains an option.
Native density/OV fitting, tails and end-to-end gates remain failed/open.

User selected **Add shared exact-OV route**. Implemented explicitly selected
`FDDSFullOVResponse` and owned `FDDSFullOVFrequencyResponse` in shared fdds_response.py.
Inputs are complete supplied baseline H1/H2, transition legs and remaining coupling,
with required coordinate label. Baseline uses unregularized NumPy solve; singular
inputs raise without fallback. Auxiliary coupling reuses the unchanged helper and
1e-13 cutoff. Return effective/raw baseline and raw coupled arrays, no implicit
symmetry or ordinary-uncoupled/Coulomb-label substitution. Legacy defaults are intact.
Old staged helper retained as `.pi/audit/fdds-response-before-full-ov.py`; updated
Python module staged byte-identically, no C++ rebuild/install. All7 legacy function/
class source definitions are unchanged (`full-ov-legacy-source-comparison.json`).

New `test_fdds_full_ov.py` covers independent coupled solves, ownership, nonsymmetric
operators, overcomplete legs, malformed inputs/frequencies, singular baseline and
inherited coupling policy. Task `bec9d54b2` runs the combined19-file suite
(`full-ov-combined-tests-v1.log`). Task `b2111efb1` invokes the new provider on captured
operators and compares all11 CDFs (`shared-full-ov-provider-v1.{json,log}`). Task
`bbf847325` runs four unchanged SAPT fixtures in `fdds-regressions-full-ov-v1/`, then
GRAC tests (`full-ov-{sapt-regressions,grac-tests}-v1.log`). Read-only advisory review
`db637db67b2e2d011e9e6da271589f9e0` covers new API/tests/measurement. Combined task `bec9d54b2` PASSED **515 tests in3.85s**. Captured-case comparison,
SAPT/GRAC and advisory review remain pending; do not mutate reviewed/runtime module
scope before completion. #30 remains active.
This addition does not implement native operator generation or close prior failures.

Task `b2111efb1` PASSED the new provider's captured-case comparison at all11
frequencies: maximum absolute/scaled CDF error1.570697349961847e-9 /
5.703776809025682e-13. Provider baseline equals the independent baseline calculation
exactly; all246 coupling modes retained. Baseline backward residual7.060358956574724e-17;
reference fixed-point residual1.1600694987670985e-14. Returned representation is
explicitly `fitted_density_coefficients`. These supplied-operator results do not
establish native generation. SAPT/GRAC and advisory review remain pending.

Retrieved advisory review `db637db67b2e2d011e9e6da271589f9e0`, verified SHA256
`d622dd69883a9aae06588e81c6b4e2fcd5bebebca7b96b6d5c01e8afb220e181`: no must-fix
algebra/ownership bug. Review independently confirms Woodbury sign/order, owned
arrays, no singular-baseline fallback, inherited coupling policy and untouched
legacy dispatch. This is advisory, not runtime validation.

Added review-requested nov>ncoordinates/nonzero nonsymmetric coupling and finite-
input overflow tests, without changing production code. `b658b59f8` runs focused
review tests (`full-ov-review-tests-v1.log`), pending. Supplement
`shared-full-ov-provider-v1-rank-acceptance.json` explicitly requires both measured
numerical agreement and full retained coupling rank; it passes and preserves the
original report unchanged. Exceptions on invalid/overflowing computations are
not normalized into fallback values. Module-level documentation clarification is
deferred until remaining SAPT/runtime tasks complete; #30 remains active.

Task `bbf847325` PASSED all four unchanged SAPT fixtures (dft1/dft2/api/lrc:
19.01/75.07/10.12/6.64s) and **4 GRAC tests in51.10s**. Input hashes and outputs
are retained under `fdds-regressions-full-ov-v1/`; logs are
`full-ov-sapt-regressions-v1.log` and `full-ov-grac-tests-v1.log`. Additional
review-edge tests remain pending; no native forward gate is closed.

Review-edge task `b658b59f8` PASSED **81 focused tests in2.09s**, including all added
finite-input overflow and noncommuting rectangular-coupling cases. Clarified the
module docstring's explicit-coordinate exception only after runtime tasks finished;
all function/class source definitions remain identical (AST source-segment check).
Pre-doc staged module retained as `.pi/audit/fdds-full-ov-before-doc.py` and updated
source staged. Task `bcf1d3f0a` performs final combined tests plus v2 provider replay
(`full-ov-combined-tests-v2.log`, `shared-full-ov-provider-v2.{json,log}`). V2 makes
full retained coupling rank an explicit acceptance condition. Final result pending;
#30 remains active until it completes.

Final task `bcf1d3f0a` PASSED **520 tests in3.93s** and v2 provider replay at all11
frequencies (maximum scaled5.703776809025682e-13), with explicit246/246 rank acceptance.
Targeted git diff --check passes. Portable `psi4_full_ov_provider_evidence.json`
records final supplied-operator acceptance, unchanged legacy definitions, SAPT/GRAC
and advisory evidence. #30 is complete: this resolves explicit method selection,
NOT the retained inherited-hybrid mismatch or native generation failures.
#29 resumes construction provenance. Read-only implementation audit
`d2decb4e278c3f8e2c83744ea9a5a0723` targets OO/VV producer indices/metadata,
make_D_S_D request/cache/completion generations and actual Rho_Doo2FuncExpansion
density coefficients. Its capture/reader/reference scope remains unchanged until
retrieval. Native density/OV fitting, tails, native response and end-to-end remain
unaccepted.

Retrieved construction audit `d2decb4e278c3f8e2c83744ea9a5a0723`, verified SHA256
`860d0bc720579b1cc05aafee758dec47d187fad3e42df8b4daa72fd550cae0b7`.
Implementing in bounded stages: schema6 first adds OO/VV producer rows, subsequent
setter metadata and the actual AUX1 kernel-density coefficients at the natural
Rho_Doo2FuncExpansion producer. KERNEL_SOURCE references that exact density event.
Reader requires complete ordered subset streams plus setter commitment; VV parent
indices include the occupied offset. Density reconstruction uses source-ordered
2*sum_i Doo[u(i,i),:] with unchanged1e-9 gate, no repair/rescaling. A later OO overwrite
does not rewrite the density's frozen source association. Full successful parent-
solve and original integral-cache generations remain the NEXT separate stage;
`complete_parent_solve_and_cache_provenance` is explicitly false. #29 is not complete.

Schema6 safety/reader tests (including `test_isapol_subset_density.py`) run as
`b02de5ac0`, log `response-subset-tests-v1.log`. Fresh isolated preparation succeeded
at `production-camcasp-response-v7/` (`response-prepare-v7.json`). Source/hash manifest
includes function_expansion_operations.F90 with both.o/.mod dependencies on the
low-level observer. Task `b9725e57c` builds and runs traced/untraced v7 reference:
`response-build-v7.log`, `response-reference-v7.log`, `response-v7-water-{traced,untraced}`.
Test task `b02de5ac0` PASSED **51 tests in1.49s**. Reference build/traced-untraced
task `b9725e57c` remains pending. All earlier builds, failures and measured v6 evidence retained;
no source/native runtime install or production SAPT arithmetic change.

Reference task `b9725e57c` PASSED build and both v7 runs:4793 events, including
30 OO rows,3828 VV rows,870 OV rows,22 fit metadata and1 actual kernel-density
vector. All35 reference artifacts are byte-identical traced versus untraced
(`response-v7-trace-comparison.json`). This establishes observer noninterference,
not reconstruction accuracy. Task `bab4210ab` now runs strict schema6 association,
ordered density reconstruction and existing raw-fit/Hessian/11-frequency replay;
report/log `response-v7-subset-density-replay.{json,log}` pending.

Replay `bab4210ab` FAILED on `Changed subset metadata without producer`; original
failure log retained. #31 tracks this blocker under active #29. The observed changes
are only NN(type1)->OV(type3) retags: constrained Dov metadata460->461 and ordinary
Dov4742->4764; shapes and every other captured parameter remain identical.
Source `df_monomer.F90:425–461` skips fitting when parameters compare equal but
always executes the final setter. Pinned `df_parameters.F90:236` literally compares
`A%df_type/=A%df_type`, so type differences do not invalidate the cached fit.
No reference fix is made. Reader now records this narrow OV-only1->3 retag separately,
preserving original producer metadata and generation IDs; changed fit values/shapes
still fail closed. Fresh direct OV solves remain rejected by the observer guard.
New test verifies preservation and rejection of simultaneous lambda changes.
Task `b79c16c50` PASSED **52 tests in1.47s** and strict v7 replay. Five subset
generations and two metadata retags are associated without rewriting originals.
Kernel-density event4744 reconstructs exactly from OO generation starting463
(maximum absolute/scaled error0). Raw NN fit and H1/H2 reconstruct exactly;
all11 CDFs pass at maximum scaled5.644276459242761e-19. Portable evidence:
`camcasp_subset_density_evidence.json`, with immutable report/hash and failed-log
provenance. #31 is resolved. #29 remains active: full parent-solve/integral-cache
provenance is explicitly false, and native generation/end-to-end remain unaccepted.

Continuing #29 with schema7/v8 parent-solve provenance. Metadata-only SOLVE_BEGIN/
SOLVE_END bracket actual fresh NN solves at lambda0 andlambda1; no additional raw
A/RHS arrays are exported. Scalar IDs, original identities, shapes and actual fit/
solver controls are captured. The reader invalidates overwritten parent identities,
requires successful completion before subset rows, freezes each subset's parent
solve ID and checks original setter parameters against that parent. Cached metadata
retags do not mint a solve. Integral-cache generation provenance remains explicitly
false and separate. No numerical/reference-object I/O changes are made.

Task `b74270b48` runs parent/safety/reader tests (`response-parent-tests-v1.log`).
Fresh v8 preparation succeeded (`response-prepare-v8.json`). Task `b940e8936` builds,
runs traced/untraced and strictly replays the new capture; outputs
`response-build-v8.log`, `response-reference-v8.log`, `response-v8-parent-replay.{json,log}`.
Parent/safety/reader task `b74270b48` PASSED **61 tests in1.51s**. Fresh reference
build/run/replay task `b940e8936` subsequently PASSED. V8 emits4797 events and
all35 traced/untraced reference artifacts are byte-identical. Two successful NN
parent solves (lambda1 ID1, lambda0 ID2) bind five subsets with parent IDs1/1/2/2/2.
Density event4748 reconstructs exactly from OO generation467. Raw NN fit and H1/H2
remain exact; all11 CDFs pass at maximum scaled5.644276459242761e-19. Portable
`camcasp_parent_solve_evidence.json` records report/hash and noninterference evidence.
Parent-solve provenance is accepted for this supplied capture; integral-cache
provenance remains false. #29 stays active for original cached tensor construction
identities/generations. Native fitting, tails and end-to-end gates remain open.
Retain all earlier source versions, reports and failures.

Cross-build v2-disabled versus v4 comparison: 34 files identical; only
`data-summary.data` differs in its compile timestamp line. The raw equality-false
report is preserved in `response-v4-v2-baseline-comparison.json`, with the exact
single-line explanation in `response-v4-v2-baseline-interpretation.json`. No
numerical differences or tolerance changes were masked. Within v4, all35 files
remain byte-identical traced/untraced.

Disabled-observer baseline task `b4cbfdb25` PASSED execution (exit0), with zero
observer files and eleven logged frequency evaluations: static plus ten negative
omega2 values through -1430.6369983254772. The normal ending and polarization format-A
output are retained in `response-v2-baseline-untraced/`; `baseline-report.json`
records 40 file hashes including inputs/logs. Thus the same v2 executable completes
this input when observation is disabled. This does NOT establish traced/untraced
identity or response replay/parity; the producer-relocated v3 run remains pending.
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
