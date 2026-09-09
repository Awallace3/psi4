# Next-agent handoff — libisapol / CamCASP

## 1. Read first

1. [SPEC.md](psi4/src/psi4/libisapol/SPEC.md): current scientific/API contract.
2. [NATIVE_REFERENCE_BASIS.md](psi4/src/psi4/libisapol/NATIVE_REFERENCE_BASIS.md):
   the corrected historical target and present resource blocker.
3. [NATIVE_FIXED_GRAC.md](psi4/src/psi4/libisapol/NATIVE_FIXED_GRAC.md) and
   [NATIVE_OEPROP.md](psi4/src/psi4/libisapol/NATIVE_OEPROP.md): working public API.
4. Read the stage-specific contracts linked from SPEC before changing that stage.

**Accepted code checkpoint:** `1a097ec9f6a033428053354b294c5524e98b6137`.
This handoff compacts the state at that checkpoint; no scientific code changed
as part of compaction. All prior execution history remains in Git (section 8).
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

### Latest verified evidence (local paths relative to worktree)

- **1,895 ISA/FDDS tests passed in 36.79 s**:
  `.pi/audit/reference-basis-regressions-v1.log`.
- Final default water: **29.6536 s / 594,120 KiB**, all saved outputs bitwise
  baseline: `.pi/audit/native-water-post-preflight-comparison.json`.
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

**Implement a bounded native point-response/PFIT prerequisite, not another basis
alias or a large aVTZ run that bypasses the guard.** Before choosing the patch:

1. Inspect `pfit.h`, `native_response.{h,cc}`, `partitioned_response.{h,cc}` and
   `isapol_native_response.py`, plus the target trace and PFIT provenance enums.
2. Determine an explicit native point-charge coupling/response API from actual
   wavefunction/response state. Preserve the target sign `-d(phi_induced)/dq`,
   atomic units, no energy `1/2` or bare electrostatics. Direct-OV and the target's
   constrained-NN/fitted-propagator origins must remain separately labelled.
3. Define a small deterministic analytic/independent test boundary, owned inputs,
   context invalidation, frequency conventions, work limits and target provenance
   before wiring it into the existing PFIT solver. Do not infer targets/model
   parameters from final Cn or reuse another track's parameter count.
4. Keep the current demo operational. Parent-run focused + integrated tests and
   default numerical comparison; independently review the scientific boundary.
5. Only claim the implemented prerequisite. Full target still needs constrained
   NN/distributed-response reproduction, point/model/frame/anchor conventions,
   refinement and a justified solution to the large-response resource blocker.

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

The pre-compaction 3,505-line plan and 1,724-line SPEC are preserved exactly:

```bash
git show 1a097ec9f6:plan.md
git show 1a097ec9f6:psi4/src/psi4/libisapol/SPEC.md
```

Their historical status/in-progress text is superseded by this handoff and the
current SPEC. Use them only for detailed derivations or old audit evidence.
The external summary report is `~/docs/camcasp-2026-09-08.html`; it is outside
this Git repository. Do not treat report prose as stronger than scoped tests.
