# Next-agent handoff — libisapol / CamCASP

## 1. Read first

1. [SPEC.md](psi4/src/psi4/libisapol/SPEC.md): current scientific/API contract.
2. [NATIVE_REFERENCE_BASIS.md](psi4/src/psi4/libisapol/NATIVE_REFERENCE_BASIS.md):
   the corrected historical target and present resource blocker.
3. [NATIVE_FIXED_GRAC.md](psi4/src/psi4/libisapol/NATIVE_FIXED_GRAC.md) and
   [NATIVE_OEPROP.md](psi4/src/psi4/libisapol/NATIVE_OEPROP.md): working public API.
4. Read the stage-specific contracts linked from SPEC before changing that stage.

**Accepted code checkpoint:** `86b548c492`, plus `c07dafd37d`, which
adds the bounded native direct-OV point-charge response prerequisite (section 3),
plus `8d4841ce68`, which closes the response right-hand-side factor 4 against
Psi4's own CPHF dipole polarizability and against perturbed-SCF energy
curvature, plus `061fb83e8c`, which closes the `a` and `b` kernel scalings against
Psi4's Davidson TDSCF and against matched-functional perturbed-SCF energy
curvature.
All prior execution history remains in Git (section 8).
No implementation/build/test background tasks are pending. Old task IDs and
“in progress” paragraphs in historical documents are not current instructions.

## 2. User direction and boundaries

- Continue toward a matched **native** protocol. The user explicitly chose this
  instead of further investigating the odd-angular-normalization/RRF gap.
- CamCASP MIT source may be inspected/transcoded with attribution and notices.
  **ORIENT executable source is forbidden.** Reference input/output data is a
  distinct category; do not mistake `orient_local` fixtures for permission to
  inspect ORIENT source. RRF executable source was not inspected; its separate
  license/convention question is deferred, not implicitly authorized.
- Preserve unrelated untracked `orient_replacement.md` and `tmp/`. In particular,
  `tmp/psi4_camcasp.py` stays self-contained and uncommitted. Its SHA256 is
  `8ac079774c0f89e22d416ee9d6942b03a4b36bd935d42140e5342563f86532a5`.
- The user authorized commits; accepted increments were committed, nothing
  pushed. Do not reset, discard other work, or modify the CamCASP reference tree.
- No hidden SCF, post-SCF orbital/energy repair, charge/symmetry repair, reduced
  scientific grids, relaxed tolerances or increased resource limits merely to
  make a reference run pass. Label incomplete coverage and comparison tracks.

## 3. What works now

- Fresh restricted C1 PBE0/cc-pVDZ H/O water → native Drho-C/ordinary ISA-A →
  direct-OV ALDA → strict LW → atomic polarizabilities and isotropic Cn.
  `oeprop` still returns `None`; `psi4.atomic_property_result(wfn)` owns the result.
- Explicit fixed-GRAC SCF-input admission, actual-component/state validation,
  owned correction provenance and correction-aware context reuse. This uses
  **ALDA on fixed-GRAC orbitals**, not a GRAC kernel derivative. NONE is default.
- Bounded ISA preparation, streamed density/AUX-Q work and deterministic
  independent-output OpenMP. Default one-thread results remain bitwise baseline.
- Independently callable supplied-input localization/PFIT, isotropic,
  orientation-resolved and rank≤3 **recoupled** dispersion APIs. They are distinct
  representations, not interchangeable public end-to-end claims.
- Hash/provenance-qualified expected reference basis manifest plus early
  dimensional response preflight. It is deliberately **not a PartitionRecipe**.
- Bounded native **direct-OV point-charge response prerequisite** for PFIT:
  `IsaPointChargeOperators` (C++) builds the positive-kernel point-charge OV
  coupling `W(t,p)`, and `native_point_charge_response` reuses one native
  response's owned H1/H2 in a second full-OV solver with W as legs to return
  signed `v = -W^T C W = -d(phi_induced)/dq` in Eh/e² at caller-owned points and
  imaginary frequencies. The npoint right-hand sides save only the nov×nov
  RHS/solution blocks — the O(nov³) factorization remains, and no work guard is
  relaxed. The owned PFIT solver accepts the packed lower-triangle
  targets under the separate `NativeDirectActualPointResponse` origin. It is
  **not** the historical constrained-NN/fitted-propagator target, and it infers
  no charge/multipole model, channel set or parameter count.

### Latest verified evidence (local paths relative to worktree)

- **1,923 ISA/FDDS tests passed in 46.28 s** (1,895 prior + 28 new point-response
  tests): `.pi/audit/native-point-response-regressions.log`. The 28 are the 19
  originally written, plus the second analytic-oracle test added when the review
  forced the s-only oracle to be generalized to p shells (the single s-block test
  became one full-basis pure test and one Cartesian test that assumes no pure
  ordering), plus the 2 layer-5 tests that close the right-hand-side factor 4,
  plus the 6 layer-6 tests that close the `a`/`b` kernel scalings (4 spectrum
  points and 2 finite-field points). Run from `/tmp` against the staged tree
  with `OMP_NUM_THREADS=1`
  over `tests/pytests/test_isapol*.py` + `test_fdds*.py` minus the slow
  `test_isapol_oeprop_water.py`, which is run separately below. The new file
  alone is 28 passed in 10.57 s; it was 22 passed in 3.96 s before layer 6 and
  20 passed in 1.68 s before layer 5. Each layer costs SCFs, not solves: layer 6
  runs 4 DFT SCFs plus 24 perturbed ones. Previous 1,917-test/39.10 s,
  1,915-test/36.93 s and 1,895-test/36.79 s runs are superseded in that log and
  in `.pi/audit/reference-basis-regressions-v1.log`.
- Point-response test boundary is mutation-checked, not just green: on the staged
  module, flipping the target sign, symmetrizing the packed triangle, packing the
  upper triangle instead, dropping the orbital context check and leaving the
  returned arrays writable each fail the new file. Dropping `provider.kernel`
  from the digest text list does **not** fail it, because kernel/model identity
  already enters the digest through the hashed H1/H2; the test name says
  "response model", not "kernel", for that reason.
- An independent review of the scientific boundary confirmed the sign, units,
  absence of any 1/2 or bare/nuclear term, the `t=a*nocc+i` ordering and the
  libint2 charge convention, and found no defect in immutability, packing or
  reciprocity handling. It also found three claim defects, all now corrected in
  code/SPEC/plan: the boundedness wording overstated what npoint right-hand
  sides save (the O(nov³) factorization remains);
  `core.ExternalPotential.computePotentialMatrix` is the **same** libint2
  `nuclear` engine, so that gate certifies the AO→MO transform and packing, not
  the kernel; and "no screening" needed the libint2 precision-zero qualifier.
  Acted on as well: the analytic oracle now covers every shell of the fixture
  basis (s and p, pure and Cartesian) rather than the s block only; forwarded
  provenance/convergence strings are hashed into the context digest and
  documented as unverified; and the new PFIT origin is appended to
  `IsaPfitTargetOrigin` so existing enumerator values are unchanged.
  Of the two normalizations that review left open, **the factor 4 is now
  closed** (see the next bullet); no shell above p is exercised, and that stays
  labelled in SPEC §8.
- **Factor-4 normalization closed absolutely** (test layer 5, 2 new tests). At
  `exact_exchange=1`/`kernel='no_local'` the native H1/H2 are the closed-shell
  (A+B)/(A−B) matrices, so the ω=0 solve *is* coupled-perturbed Hartree-Fock and
  `alpha = -D^T C D` from dipole OV legs acquires an absolute scale. Water/STO-3G
  agrees with Psi4's own iterative `Wavefunction.cphf_solve` (reached through
  `psi4.properties`, a different solver with its own independently written
  restricted prefactor) to **2.1e−14** on the full 3×3 tensor, and with the
  curvature of *perturbed SCF total energies* to **5.0e−9** (rel 1.2e−7) after
  one Richardson step over h=8e−3 and 4e−3, the h-halving error ratio measuring
  **4.001**. The energy-curvature oracle is the stronger of the two: it uses no
  response theory, no orbital Hessian and no prefactor at all, and because a
  central second difference is even in λ it does not inherit Psi4's
  `perturb_dipole` sign convention either. Confirmed discriminating by mutating
  the staged solver's `-4.0 * h2.dot(legs)` to −2.0, −8.0 and −4.5: all three
  fail both tests, so even a 12.5% error is caught, not just a factor of two.
  The staged file was restored and its SHA256 re-verified after each mutation.
  Cross-check at cc-pVDZ (nbf=24, nov=95) also matched CPHF to 2.1e−13; STO-3G
  is what the committed test uses, because the native construction is 0.02 s
  there against 5.08 s at cc-pVDZ. This does **not** certify the separate `a`
  and `b` kernel scalings away from that configuration; the bullet below does.
- **`a`/`b` kernel scalings closed absolutely** (test layer 6, 6 new tests).
  The pre-existing ALDA gates re-derive the written `H1=Δ+4V−a(X+Y)+4bL` and
  use only complementary `(a,1−a)` pairs, so they can neither see a wrong
  overall factor on `L` nor separate the two scalings from their sum. The new
  gate builds a **matched** custom functional (`x_hf` by `a`, `LDA_X` and its
  LDA correlation partner by `b`; LibXC names unprefixed — the builder adds
  `XC_`) whose CPKS kernel *is* the native operators at `(a,b)`, then closes
  both scalings twice:
  - `sqrt(eig(H2·H1))` — the eigenvalues of `(A−B)(A+B)` are Ω² — against
    Psi4's independent Davidson `tdscf_excitations`: **7.2e−14 … 1.8e−13**
    over `(0.25,0.75,pw92)`, `(0.5,0.5,vwn)`, `(0.3,0.9,slater)` and
    `(0,1,pw92)`, max imaginary part exactly 0. This is the **only** oracle in
    the track that reaches `H2`: at ω=0 the solve collapses to `−4·H1⁻¹D` and
    `H2` cancels identically, so layer 5 and every polarizability gate are
    blind to it. `(0.3,0.9)` is deliberately non-complementary and breaks the
    `b=1−a` degeneracy.
  - Perturbed matched-RKS **total-energy** curvature: **1.9e−9** (rel 4.1e−8,
    h-halving ratio 4.0000) at `(0.25,0.75,pw92)` and **9.4e−9** (rel 2.9e−7,
    ratio 4.0009) at `(0.3,0.9,slater)`. Absolute: no response theory, no
    orbital Hessian, no prefactor, no field sign convention.
  The coarse (50,25) grid costs nothing here, which is what makes this cheap
  enough to commit: the second difference of the grid-discretized `E_xc` is the
  grid-discretized `f_xc`, and the native `L` uses the SCF's own grid, so the
  quadrature error cancels between the two sides. Confirmed against
  (590,99)/168,883 rows, which agrees no better (1.3e−8/1.1e−8). Grid size is
  not free, though: (74,35) makes Psi4's Becke pruning emit 832 negative
  weights (min −92.15), which the provider's grid guard rejects, correctly.
  Confirmed discriminating by mutating the staged provider call to
  `local_scale*1.01` and to `exact_exchange+0.001`: each fails all six layer-6
  tests. In-test controls resolve `a` to 0.001 (spectrum moves 6e−4), `b` to
  1% (1e−4), `b` halved (6e−3…1.2e−2), and `a` **in `H2` alone** to 0.001
  (3e−4), against a 1e−13 baseline. The staged file was restored and its
  SHA256 re-verified after the mutations. Whole layer costs 7.9 s.
- Default water demo after the point-response commit: **29.5248 s / 590,992 KiB**,
  31 ISA iterations, energy `-76.33875890072267 Eh`, and every saved array
  (`scalars`, `local`, `global_tensors`, `coefficients`) **bitwise equal** to
  `native-water-baseline-t1` (max absolute error 0.0) under the unchanged
  `<=1e-9` global-scaled equivalence policy:
  `.pi/audit/native-point-response-water-comparison.json`. Preceding
  post-preflight run for comparison: **29.6536 s / 594,120 KiB**, also bitwise
  baseline: `.pi/audit/native-water-post-preflight-comparison.json`.
- Integrated runs at this commit: `test_isapol_oeprop_water.py` **3 passed /
  97.22 s** (wall 1:38.13, peak RSS 664,832 KiB) and the four SAPT-DFT
  regressions **4 passed / 112.72 s**:
  `.pi/audit/native-point-response-integrated.log`. The uncommitted
  `tmp/psi4_camcasp.py` demo still exits 0 and prints the same static atomic
  dipole trace polarizabilities `[3.5492546180905062, 0.8756974686948751,
  0.8756974684436827]` bohr³ and 3×3 site-pair Cn table.
- Final fixed-GRAC public strict-LW/9-pair endpoint: **30.0985 s / 588,436 KiB**,
  30 ISA iterations, energy `-76.33871950327045 Eh`:
  `.pi/audit/native-fixed-grac-post-preflight-water.json`.
- Separate molecular/SAPT regressions at the fixed-GRAC checkpoint:
  **3 passed / 99.05 s**, **4 passed / 113.29 s**:
  `.pi/audit/native-fixed-grac-{molecular,sapt}-regressions.log`.
  These are prior-checkpoint results, not a later rerun of all seven tests.
- Controlled performance: baseline **360.6726 s / 780,104 KiB**; property threads
  1/2/4/8 gave **29.4297 / 22.5083 / 17.9785 / 15.9646 s** (12.26–22.59×),
  23–25% lower RSS. SCF held at one thread; maximum scaled output error
  `9.83e-16` under the unchanged `1e-9` comparison gate:
  `.pi/audit/native-water-final-comparison.json`. Not a statistical scaling study.
- Non-OpenMP helper compiled/ran; **no full no-OpenMP core build**. Installed
  reference-basis NOTICE was byte-verified; Cythonized installation not exercised.

`.pi/audit/` is local/ignored evidence, not a runtime or portable-test dependency.
Portable fixtures, source manifests, notices and tests are committed.

## 4. Critical correction: do not implement a fictitious gate-9 ISA preset

Target: `~/gits/CamCASP/tests/H2O_props/psi4/H2O-avtz.clt` and
`check/L2H1/H2O_ref_wt3_L2_Cn.pot`. The current and historical `777f904` generator
trace selects **constrained NN → distributed response → LW → PFIT**, not ISA.
Expected MAIN is spherical aug-cc-pVTZ (92 functions); ordinary Cartesian RI AUX
has 246 functions and is the AtomAux fallback. No ISA shapes/controller requested.
The actual historical Psi4 MAIN export is missing. Native contractions differ
literally (56 vs 62 primitive entries) but span the expected basis; continue
adapting actual wavefunction orbitals, never substitute manifest contractions.

The manifest `h2o_props_psi4_777f904_manifest()` keeps
`historical_scf_export_verified=False`, records missing artifacts, and supplies
only expected data/buildable AUX. ISA-A and ISA-A+DF templates are different
protocols. Do not silently switch the target to one of them.

Resource blocker: for expected nbf=nmo=92, nocc=5, nOV=435. The current public
3-atom IsaGrid(99,590) has **3×98×590=173,460 rows**, no pruning. ALDA work is
**32,822,968,500 > 2,000,000,000**. The 10,569-row ceiling is not a proposed grid.
The preflight mirrors existing guards; C++ checks remain authoritative.

Missing historical artifacts: generated CKS, actual SCF/basis export and versions,
response grids/propagators, pre-refinement frequency tensors, actual point lattice
and `.p2p` responses, `.pdef`, per-frequency PFIT data and refined tensors.
The final potential and a basis alias do not reconstruct these.

## 5. Immediate next implementation direction

The bounded point-charge prerequisite of the previous direction is **done** and
committed (section 3, SPEC §6 "Native direct-OV point-charge target
prerequisite"). It closes only the "native targets exist at all" gap. What
remains between it and the `777f904` target, in dependency order:

1. **Reproduce constrained NN → distributed response**, which is the trace's
   actual partition/response path. The committed prerequisite deliberately
   feeds direct-OV response; it does not become the target by relabelling.
   Do not substitute ISA-A for constrained NN (section 4).
2. **Pin the point/model/frame/anchor conventions from actual artifacts.** The
   point lattice, `.pdef` model definition and per-frequency PFIT inputs are
   among the *missing* artifacts (section 4). Until they exist, the caller still
   owns points and model; do not infer either from the final `Cn` potential or
   borrow a parameter count from the dispersion track.
3. **Add the refinement stage** that stands between raw per-frequency PFIT
   output and the reference refined tensors. There is currently no refinement.
4. **Resolve the large-response resource blocker honestly** (section 4:
   nOV=435 × 173,460 grid rows ⇒ ALDA work 3.28e10 vs the 2e9 limit). The
   npoint-RHS solve in the prerequisite bounds only the *new* work; it does not
   make the aVTZ response affordable. A justified answer means real grid
   pruning or a different response algorithm with its own gate — **not** raising
   `max_bytes`/work limits, coarsening the scientific grid, or bypassing
   `estimate_response_work`.

Each step needs its own independent oracle before it is wired to the next, and
each must stay separately labelled in provenance; see the SPEC §8 note that the
point-response gates are explicitly non-transferable.

Useful local investigation: `.pi/audit/matched-basis-trace-handoff.md` (full exact
trace, hashes and separate ISA candidates). Its portable conclusions are in
`NATIVE_REFERENCE_BASIS.md`; do not depend on the ignored handoff for runtime.

## 6. Still-open numerical/coverage gaps

- Odd `L+H+J`, strict lower-J normalization: **5,341 coefficient rows uncertified**.
  Six raw-identity tests do not close this; 182 reciprocal classes in 80 blocks
  survive, so the channels are not absent/zero. User deferred this investigation.
- General maximal-J channels of lower orders are outside the dedicated high-J
  certification; rank4 remains rejected by recoupled API; C12 is partial.
- Matched native ISA reference comparisons, raw tails/Drho/fitted-OV conditioning,
  fitted versus direct response and strict recorded-input LW defects retain
  distinct gates. See SPEC/PROVISIONAL_ACCEPTANCE.md; no blanket tolerance waiver.
- Full native SCF/PFIT/GRAC matched protocol and modern ISA preset are not closed.
- Point-response coverage: no shell above p is exercised by the analytic ESP
  oracle (the fixture basis has none), and neither the factor-4 nor the `a`/`b`
  gate touches that. The right-hand-side factor 4 and the `a`/`b` kernel
  scalings are both closed absolutely (section 3). What remains open there is
  narrower: `b` is anchored against an *energy* at only two of the four `(a,b)`
  points, the other two resting on the excitation-energy gate; and both gates
  are ω=0 or excitation-energy statements about the operators, so no
  frequency-dependent propagator convention is certified by them.

## 7. Build and test commands

```bash
ROOT="$HOME/gits/psi4__worktrees/camcasp-psi4-joint"
PY="$HOME/miniconda3/envs/p4_ci/bin/python"
BUILD="$ROOT/build_camcasp_psi4_joint/psi4-core-prefix/src/psi4-core-build"
export PYTHONPATH="$ROOT/build_camcasp_psi4_joint/stage/lib"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/p4_ci/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PSIDATADIR="$ROOT/build_camcasp_psi4_joint/stage/share/psi4"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Only when source/build rules changed; use background execution for long jobs.
cmake --build "$BUILD" --target core -j 2
cmake --install "$BUILD"
cmp "$BUILD/src/core.cpython-313-x86_64-linux-gnu.so" \
  "$ROOT/build_camcasp_psi4_joint/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so"

# Run outside the source package to prevent import shadowing.
cd /tmp
files=()
for f in "$ROOT"/tests/pytests/test_isapol*.py "$ROOT"/tests/pytests/test_fdds*.py; do
  [[ "$f" == */test_isapol_oeprop_water.py ]] || files+=("$f")
done
"$PY" -m pytest "${files[@]}" -q --tb=short
"$PY" -m pytest "$ROOT/tests/pytests/test_isapol_oeprop_water.py" -q
"$PY" -m pytest "$ROOT/tests/sapt-dft1/test_input.py" \
  "$ROOT/tests/sapt-dft2/test_input.py" "$ROOT/tests/sapt-dft-api/test_input.py" \
  "$ROOT/tests/sapt-dft-lrc/test_input.py" -q
"$PY" "$ROOT/tmp/psi4_camcasp.py"
```

Inspect the configured install destination before using another build tree.
Do not install over a runtime used by active calculations. New Python requiring
new bindings must be staged with its matching core. `--ignore` does not exclude
files explicitly passed to pytest: filter the list first, as above.
Background terminal notifications are authoritative; do not poll to wait.

## 8. Checkpoints and archived history

- `f996942ad6`: native properties, performance and rank≤3 recoupled engine.
- `e0e9fcfbbe`: independent maximal-J9/10 oracle.
- `636c6db6e0`: independent even-parity lower-J oracle.
- `d52fb16a19`: explicit fixed-GRAC admission; deferred odd raw identities.
- `1a097ec9f6`: corrected non-ISA basis manifest, packaging and preflight.
- `86b548c492`: compacted SPEC/handoff only; no scientific code changed.
- `c07dafd37d`: bounded native direct-OV point-charge response
  prerequisite — `point_response.{h,cc}`, `isapol_native_point_response.py`,
  the `NativeDirectActualPointResponse` PFIT origin, 20 new tests in
  `tests/pytests/test_isapol_native_point_response.py`, and SPEC §6/§8.
- `8d909a53b0`: recorded that commit's SHA in this handoff; no code changed.
- `8d4841ce68`: closed the shared right-hand-side factor 4 — test layer 5
  (2 tests) against Psi4's `Wavefunction.cphf_solve` and against the curvature
  of perturbed SCF total energies, plus the SPEC §8 absolute-gate paragraph and
  the section 3/6 records here. No scientific code path changed; the factor was
  mutated in the staged copy only, and restored.
- `061fb83e8c`: closed the `a` and `b` kernel scalings — test layer 6 (6 tests)
  against Psi4's Davidson `tdscf_excitations` via `sqrt(eig(H2·H1))` and
  against matched-functional perturbed-SCF total-energy curvature, plus the
  SPEC §8 paragraph and the section 3/6 records here. No scientific code path
  changed; the provider call was mutated in the staged copy only, and restored.

The pre-compaction 3,505-line plan and 1,724-line SPEC are preserved exactly:

```bash
git show 1a097ec9f6:plan.md
git show 1a097ec9f6:psi4/src/psi4/libisapol/SPEC.md
```

Their historical status/in-progress text is superseded by this handoff and the
current SPEC. Use them only for detailed derivations or old audit evidence.
The external summary report is `~/docs/camcasp-2026-09-08.html`; it is outside
this Git repository. Do not treat report prose as stronger than scoped tests.
