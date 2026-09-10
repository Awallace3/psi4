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
  orientation-resolved and **recoupled** dispersion APIs (declared ranks 1..4 over
  the 13 ordered pairs upstream defines). They are distinct
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
- Two explicitly named native response algorithms, `ordered_pairwise` (default,
  unchanged) and `shared_sweep`, selected by `ATOMIC_RESPONSE_ALGORITHM` on the
  public path and by `algorithm=`/`response_algorithm=` on the driver APIs, and
  folded into the native-context policy hash. Same ordered sweep and quadrature;
  V/X/Y bitwise identical, L equal to rounding. Separately calibrated ALDA work
  limits (2e9 vs 6.4e10 on `grid_rows*nOV**2`); neither authorizes the other and
  no caller argument raises either. This is what makes the full unpruned
  PBE0/aug-cc-pVTZ IsaGrid(99,590) response affordable (section 4).

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

Measured, on this branch (PBE0, that grid, `dft_radial_points` 99 /
`dft_spherical_points` 590, `scf_type pk`): the blocker is now quantified and
**row screening alone does not clear it**. `IsaAldaGridScreen` bounds each
quadrature row's exact contribution (see SPEC §6); at aug-cc-pVTZ the actual SCF
gives nbf=nmo=92, nOV=435 as expected, 37,958 of the 173,460 rows are exactly
zero, and 135,502 survive lossless screening — still **12.8×** the permitted
ALDA work. Keeping only the 10,569 rows the limit admits omits **53.6%** of the
total contribution norm (bound 2.854 of total 5.326). A 1e-8-quality screen
needs 113,834 rows, 10.8× over. At cc-pVDZ, where the full 173,460-row
primitive is affordable (work 1.565e9 < 2e9), the same 10,569-row budget shifts
the isotropic dipole polarizability by **6.94%** (5.585 → 5.198). So the
remaining sanctioned route for the aVTZ demo is the other one: **a different
response algorithm with its own gate**, not a coarser or screened grid.

Measured throughput behind that statement: the shipped accumulator (hand
triple loop, parallel over t) runs at **9.08 GFLOP/s** at nov=95, i.e. the 2e9
limit is ≈**0.44 s** of accumulation. The identical np·nOV² contraction as a
blocked BLAS3 update runs at **104 GFLOP/s** at nov=435, so the *full* unpruned
aVTZ accumulation is ≈**0.63 s**. The flop count is the same; only the rate
differs. A BLAS3 primitive may therefore carry its own, higher, *measured* work
limit at the same wall-clock budget — that is a new algorithm with a new gate,
not a raise of this one, which stays in force for the accumulator it was
calibrated against.

The *other* aVTZ cost is now measured too, and it is larger: with `no_local`
(no ALDA rows at all, so the gate above is irrelevant) the direct-JK quartet
loop at nOV=435 takes **953.9 s**, after a 5.3 s PBE0/aVTZ SCF
(E=-76.3770293137793). It passes every existing gate. So a BLAS3 ALDA primitive
would remove ~0.6 s of a ~16-minute demo: closing item 4 needs the (b,j)-pair
recomputation of the ordered nbf⁴ shell loop addressed as well, and the two
costs must never be quoted as one.

**Resolved, by that sanctioned route only.** `shared_sweep` is now the second
explicitly named native response algorithm (SPEC §6 "Two named native response
algorithms"): the ordered nbf⁴ quartet sweep visited **once** for all `(b,j)`
transitions, parallel over `s0` only, plus a blocked-BLAS3 ALDA primitive. It
addresses *both* aVTZ costs above, which are still quoted separately: the ≈0.6 s
of accumulation the new gate covers, and the 953.9 s `(b,j)` quartet loop the
single sweep removes. V/X/Y are bitwise identical to `ordered_pairwise`; L agrees
to rounding (rel 2.7e-16 / 2.0e-16) and against the independent analytic oracle.
It carries its **own measured** ALDA limit (`ALDA_WORK_LIMITS`, 6.4e10 vs the
accumulator's 2e9, both ≈0.5 s at the measured 104 vs 9.08 GFLOP/s); every other
limit is byte-identical, no grid was pruned or coarsened, no tolerance relaxed,
no `max_bytes` raised, and `estimate_response_work` is still called on the real
dimensions on both the direct and the public path. The accumulator's 2e9 stays in
force for the accumulator.

So the aVTZ blocker is cleared and the demo is **measured**: full unpruned
173,460-row `IsaGrid(99,590)` at nbf=nmo=92, nOV=435, `ordered_pairwise`
preflight still failing (`max_grid_rows` 10,569) and `shared_sweep` passing
(338,221); direct-API response **15.29 s**, `planned_bytes` 113,886,352,
α_iso **9.870961449318902** bohr³ (diag 10.389483143568143, 9.414895197052905,
9.808506007335655). Through the public `oeprop` endpoint with
`ATOMIC_RESPONSE_ALGORITHM SHARED_SWEEP` and the reference GRAC shift
0.06490004527520865: E=-76.37966827740807, 2.61 s SCF + **24.49 s** properties,
peak RSS **1,488,060 KiB**, 37 ISA iterations, **zero stage failures** (strict LW
1e-6 passed), atomic dipole α = 7.108845614906964, 1.3810579153027442,
1.381057910633834 bohr³. That is track 1's generated recipe and ISA-A at track
2's basis and SCF-input policy — a resource and demo result, **not** parity;
items 1 and 6 of section 5 still own the constrained-NN response path.

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
   *Status: the constrained-NN chain now runs and is accepted by strict
   production LW on the intended protocol* — PBE0/aug-cc-pVTZ water with the
   reference GRAC shift 0.06490004527520865, `shared_sweep`, 99/590 response
   grid, 11 Casimir nodes, at the trace's own penalty lambda=1000. What was
   actually blocking it was **not** the penalty but the recipe's declared
   molecular AUX, which also carries the Drho-C/ISA-A density fit and was
   hardcoded to Cartesian cc-pVDZ-JKFIT regardless of MAIN. At that AUX the
   traced lambda supplies `input-sum-rule` 6.6463e-6 and LW rejects 9 of 11
   nodes; with the MAIN-matched aug-cc-pVTZ-JKFIT it is 2.7921e-7 and every
   node passes. Raising lambda a decade leaves the defect against the fit-free
   route identical to five digits (0.687237 default AUX, 0.0791458 matched, at
   both 1e3 and 1e4), so the AUX is the cause and the penalty is not. The AUX is
   therefore now a **declared** argument — `generated_recipe(..., aux_basis=)`
   and option `ATOMIC_PROPERTY_AUXILIARY_BASIS`, default unchanged and
   deliberately not MAIN-matched, never inferred from `BASIS`/`DF_BASIS_SCF` —
   and its name is carried in `NativeProperties.model` as `Drho-C ISA-A[<name>]`.
   Naming it selects a partition, so the two AUX choices are two different
   declared models and nothing is compared across them except the distance:
   molecular isotropic C6 is partition-invariant for `direct_ov`
   (46.8971254018 bit-identical under both), C8/C10 totals are not. The full
   table is in SPEC §6 and is measured by
   `tests/pytests/test_isapol_matched_auxiliary.py` under the untouched
   production policy — no tolerance, grid or penalty moved.
   *Part (a), the refinement step on this chain, is now done.*
   `.pi/audit/avtz-nn-refinement.py` refines the accepted constrained-NN chain's
   **own** LW local tensors on that protocol, with the fit-free `direct_ov` row
   refined alongside off the same native context, so the two refinements differ
   in the anchor and penalty centre alone. Targets stay this script's own
   `NativeDirectActualPointResponse` point-charge quantities and are declared as
   such; the lattice and model stay caller-declared, per item 2. Two measured
   orderings came out of it. First, **refinement dominates the response basis**
   at the site level: the constrained-NN fit moves the anchors by 0.024 (O) /
   0.0037 (H) in isotropic α, while refining on the near 4.5/6.0/7.5 lattice
   moves them by ~0.15 (O) / ~0.33 (H) — consistent with item 6's finding that
   the site-resolved dispersion splits, not the molecular C6, are what disagree.
   Second, the two rows differ in the **quadrupole**, not the dipole: the pure
   dipole variables move by ≤2.5e-3 of the largest anchor and the variables
   touching rank 2 by 2.0e-2 of it (a factor 8 at aVTZ, 40 at cc-pVDZ), so a
   dipole-level agreement between the rows must not be quoted as agreement of
   the localized model.
   *And a structural finding that had to be fixed first.* A `COPY` equivalence
   is expressed in each site's own local axes, so under LW's explicit
   `frames=None` identity default water's two hydrogens — mirror images, with
   the in-plane `10,11c` coupling at ±0.703 — cannot share one variable set at
   all: `refine` writes the reference site's value to both with the same sign,
   as CamCASP does, and misses the second hydrogen's anchors by twice the
   coupling. `RefinementModel.copy_anchor_discrepancy` now measures this and is
   reported rather than repaired. The frames used are the reference case's
   `H2O.axes` declaration, an **input** artifact already committed verbatim as
   the `axes` field of `tests/pytests/data_isapol/camcasp_cn_pot_h2o_l2h1.json`
   and rebuilt from the molecule's geometry, not read from the reference `Cn`
   output — item 2's boundary is intact. Declaring them lets the model actually
   represent both hydrogens: data rms 4.047e-4 → 3.593e-4 (near), 8.749e-5 →
   3.303e-5 (far), and the anchor distortion the fit needs 2.892 (0.962 rel) →
   2.278 (0.539) and 0.246 (0.354) → 0.029 (0.038). Committed coverage:
   `tests/pytests/test_isapol_nn_refinement.py` at PBE0/cc-pVDZ, plus the
   SCF-free `test_isapol_refine.py::test_a_copy_equivalence_reports_how_far_its_sites_disagree`.
   *What is still open in this item:* (b) the accepted matched
   chain's remaining defect against `direct_ov` is concentrated in the **rank-3
   column** (O static scalar 177.75 vs 165.23; H 9.2370 vs 2.9586), a component
   the reference's `H-Limit 1` model does not carry at all, so it cannot be
   closed against the reference and must not be quoted as agreement.
2. **Pin the point/model/frame/anchor conventions from actual artifacts.** The
   point lattice, `.pdef` model definition and per-frequency PFIT inputs are
   among the *missing* artifacts (section 4). Until they exist, the caller still
   owns points and model; do not infer either from the final `Cn` potential or
   borrow a parameter count from the dispersion track.
3. **Add the refinement stage** that stands between raw per-frequency PFIT
   output and the reference refined tensors. There is currently no refinement.
   *Oracle (decoded, MIT source, attribution recorded):* `bin/localize.py`
   drives, per frequency tag 000..010, `process` → `<refine><tag>.data` →
   `pfit` → `.pol`, concatenated to `<name>_ref_wt<W>_L<WSM>_0f10.pol`. The
   reference `check/L2H1/H2O_ref_wt3_L2_Cn.pot` decodes to weight-type 3
   (`coeff/(α²+1)`, then `/(1+ω²)` for ω≠0), `weight_coeff` 1e-3, `cutoff`
   1e-4, SVD off, WSMLIMIT 2 (O) / HLIMIT 1 (H), symmetry ON
   (`write_pfit_local_symm`). Model variables are one per unique *site type*
   taken from that type's first site, over upper-triangle component pairs of
   `(lim+1)²`, kept iff `|α_static| > cutoff`, with `COPY` for the type's
   remaining sites. Per-parameter penalties enter as `s*(z−α)²`, i.e.
   `c(k,k)+=s`, `rhs(k)+=s*α`, solved by DSYSV — which is what psi4's existing
   `isa_pfit_solve`/`data_rows` (`row[k]=f_i^T M_k f_j`) already computes.
   *Status:* **the refinement stage now exists and is matched numerically
   against CamCASP `pfit`.** `psi4/driver/procrouting/isapol_refine.py` builds
   the `IsaPfitProblem` from sites, types/COPY equivalences, local frames, rank
   limits, anchors and weights, transcribing `write_pfit_local_symm`
   (variable order = first appearance of each type, reference site
   `indices(1)`, `lim==0` contributes nothing, cutoff tested on the *reference
   site alone*, upper triangle of `(lim+1)²`, name `<label>_<row>_<col>_A`),
   `weights` (all seven types, the mis-documented `10.0e-3`/`10.0e-2`
   literals, the `/(1+ω²)` scaling) and `read_penalties`
   (`s*(z−a)²`, anchor as initial guess). The T functions are the
   bitwise-certified `isa_t_functions` (SPEC §6, `.pi/audit/t-functions/`).
   *Numeric oracle:* three formatted-`Lattice` `pfit` inputs — the reference
   L2H1 shape (55 parameters, 17 channels, 40 points), a rank-4 oxygen model
   whose cutoff excludes 234 of 325 component pairs (101 parameters, 33
   channels, 30 points), and a Tang–Toennies damped case (b=1.5, weight type
   5) — were run through `.pi/camcasp-build/x86-64/gfortran/pfit`. Every
   fitted parameter agrees with `refine(...)` to the last printed digit:
   max |Δ| 4.998e-09 / 4.961e-09 / 4.969e-09 against `f15.8` print rounding
   (5e-09), relative 1.6e-09 / 5.6e-10 / 1.7e-09; `R.m.s.` and max |residual|
   match every printed digit. `pfit`'s `f15.8` — not the algebra — sets that
   floor, and `Print Polarizabilities` (`g16.8`) is no better; raising it would
   mean modifying the reference tree, which is out of bounds. Committed as
   `tests/pytests/test_isapol_refine.py` (27 tests, 9.6 s), with inputs built
   from dyadic rationals and integer directions so no NumPy `Generator` stream
   stability is assumed. The damped case independently certifies
   `isa_t_function_damping` against CamCASP's `T_functions` staging, and the
   rank-4 case exercises rank-3/4 T rows and the cutoff-exclusion branch.
   *Measured cost, not worked around:* `pfit.cc::data_rows` is an
   O(np·nc²) dense triple loop per data row — for the L2H1 model
   (np=55, nc=17) that is 15,895 `finite`-guarded operations per point pair,
   measured at a marginal **1.441 ms/pair** and linear in pair count
   (0.332 s/210 pairs → 4.699 s/3240 pairs). `parameter_tensors[k]` carries
   one or two nonzeros, so ~99.7% of the inner loops multiply exact zeros; the
   kernel is certified as-is and was not rewritten. A 500-point cloud projects
   to 180 s per sweep and a 1000-point cloud to 721 s; `MAX_POINTS = 512`
   caps a single refinement at 131,328 pairs (~190 s), so a CamCASP-scale
   2000-point lattice (2,001,000 pairs, ~2,883 s projected) is **refused by the
   driver rather than silently attempted**. What remains for this item is the
   end-to-end comparison against the reference `Cn` potential (item 6), which
   additionally needs the constrained-NN response of item 1.
   *Run on the intended protocol:* the refinement stage now runs at the end of
   the PBE0/aug-cc-pVTZ + reference-GRAC (0.06490004527520865 Eh) demo,
   `.pi/audit/avtz-grac-refinement-demo.py` →
   `.pi/audit/avtz-grac-refinement-demo.json` (E = −76.37966827740793,
   properties 25.53 s, peak RSS 1,492,476 KiB, nbf 92, nOV 435). The sites,
   ranks (2/1/1), cutoff, weights and point lattices are **declared by the
   script**, per item 2 — the historical `.pdef` and lattice are missing, so
   nothing is inferred from the reference `Cn` potential and no parameter count
   is borrowed from the dispersion track. Two 150-point golden-angle lattices
   are reported rather than one, because a refined model is a property of its
   lattice: 4.5/6.0/7.5 bohr gives rank 17/17, data rms 4.047e-04, anchor shift
   max|Δ| 2.852 (0.962 relative), refined isotropic α 7.21270/1.07847/1.07847;
   7.5/9.0/10.5 bohr gives rank 17/17, data rms 8.749e-05, shift 0.246 (0.355),
   refined isotropic α 7.10176/1.38061/1.38061, against anchors
   7.10885/1.38106/1.38106. This is a **staged demo, not a parity claim**: the
   historical target still needs item 1's constrained-NN distributed response
   on the reference lattice. Committed coverage of the same wiring at the cheap
   sto-3g fixture (real native direct-OV targets, full-rank solve, residual
   beating the anchors and reproducing the solver's `data_rms`, COPY/symmetry
   preservation, penalty holding the anchors, and refusal of a mislabelled
   origin/representation/auxiliary claim) is in
   `tests/pytests/test_isapol_native_point_response.py`.
4. **Resolve the large-response resource blocker honestly** (section 4:
   nOV=435 × 173,460 grid rows ⇒ ALDA work 3.28e10 vs the 2e9 limit). The
   npoint-RHS solve in the prerequisite bounds only the *new* work; it does not
   make the aVTZ response affordable. A justified answer means real grid
   pruning or a different response algorithm with its own gate — **not** raising
   `max_bytes`/work limits, coarsening the scientific grid, or bypassing
   `estimate_response_work`.
   *Status:* the grid-pruning half is now implemented, certified and
   **measured insufficient** (section 4): lossless screening still leaves
   12.8× the permitted work, and the admissible 10,569 rows cost 6.94% of the
   cc-pVDZ isotropic polarizability. Remaining route: a BLAS3 local primitive
   with its own measured gate. Two separate aVTZ costs must be reported, not
   conflated — the ALDA accumulation and the direct JK quartet loop, which
   recomputes the ordered nbf⁴ shell loop once per (b,j) pair (≈3.1e10
   integral values, inside the unchanged 6.4e10 gate but not free). That second
   cost is now **measured**: the aVTZ `no_local` build at nOV=435 takes
   **953.9 s** of quartet loop after a 5.3 s PBE0 SCF, so it passes every gate
   and is still the dominant wall clock — a BLAS3 ALDA primitive alone would not
   make the demo interactive.
   *Status: closed.* `shared_sweep` (SPEC §6 "Two named native response
   algorithms", section 4 "Resolved") is a second explicitly named algorithm with
   its own **measured** ALDA gate that also visits the quartet sweep once for all
   `(b,j)`, so it addresses both costs while still reporting them separately.
   V/X/Y bitwise identical, L to rounding and against the analytic oracle.
   Full unpruned 173,460-row aVTZ response: **15.29 s** direct API, α_iso
   9.870961449318902 bohr³; **24.49 s** / 1,488,060 KiB through public `oeprop`
   with zero stage failures. No limit raised, no grid pruned, no tolerance
   relaxed. The accumulator's 2e9 limit stays in force for the accumulator.

5. **Match intermediates to a precision that depends on the property of
   interest.** *Status: the budget exists and is measured; the intermediates do
   not meet it.* `psi4/driver/procrouting/isapol_budget.py` perturbs a named
   intermediate and rebuilds the entire downstream through the shipped objects
   under the **unrelaxed** production LW policy, reporting
   `A = defect(property)/defect(intermediate)` and hence the required precision
   `tolerance/A`. Three disciplines keep it honest: the unperturbed rebuild
   reproduces every shipped property group at exactly 0.0 (the rebuild path *is*
   the shipped computation); every probe direction is projected onto the
   manifold the strict LW gate enforces, and a direction that cannot be
   projected raises; and every amplification is measured at `eps` and `eps/2`
   and withheld unless the two agree (25 of 112 water rows correctly withheld).
   It is a first-order directional **lower bound**, labelled
   `..._not_a_gate`, and meeting a requirement is necessary, not sufficient.
   *Two geometries, because two error models.* The recorded-error metric
   `max|Δ|/max(1,max|ref|)` is well posed only where elements share a scale.
   For the shape samples, which span many decades, it is not: the absolute
   amplification **rises** 5.6e4 → 2.4e5 → 6.3e5 as eps falls 1e-6 → 1e-10
   (1.8e6 → 1.4e7 → 6.6e7 for α₃), linearity defect never under ~0.1. The
   elementwise-relative geometry converges there (8.9e-4 → 7.8e-4;
   9.8e-3 → 8.5e-3) and is offered only for the two stages whose downstream is
   regenerated from scratch. The two metrics are **not** comparable and the
   driver refuses to compare across them rather than doing it quietly.
   *Two chains, neither substitutable.* Water PBE0/cc-pVDZ direct-OV is the only
   non-degenerate partition measurement; the **fitted-auxiliary route on water
   is rejected at the recorded lambda1** by strict LW (charge-sum ≈4.1e-4 at
   every frequency), which was **not** worked around — relaxing
   `residual_policy` is forbidden — so fitted-OV is anchored on He, which is
   monatomic and therefore *exactly* degenerate in the two partition stages
   (A=0, labelled structurally insensitive, not offered as evidence). A
   declared `ov_charge_penalty` >= 1e3 does earn strict-LW acceptance on water
   (SPEC §6), but is a differently declared model: it does not close this row,
   whose reference numbers are lambda1.
   *Result at a 1e-6 property tolerance* (`.pi/audit/property-anchored-budget.json`,
   SPEC §8 "Property-anchored precision budget"): Drho-C needs 2.78e-9 and
   records 1.28e-3 (**~6 orders short**); fitted OV needs 6.21e-10 and records
   2.30e-5 (**~4.5 orders short**); relative requirements are 2.75e-7 for
   Drho-C and 1.03e-4 for the shape samples; `coefficient_responses` needs
   4.00e-8 (water) / 4.39e-12 (He fitted) and `distributed_site_tensors`
   6.58e-9 (water) / 2.65e-8 (He). The raw-tail 2.36e-8 **meets** its
   requirement of 6.59e-8 (A=1.5176e1, α₃, water direct-OV) with a 2.8× margin
   and would first fail at a property tolerance of 3.6e-7 — the first recorded
   error to pass. That comparison is against a separate stage,
   `raw_tail_parameters` (per-site joint amplitude/exponent, cutoff held fixed
   as supplied configuration), because the recorded number is a tail
   *parameter* error; it is apples-to-apples by an **identity**, since the
   comparator's per-site denominator and the budget's single denominator agree
   bit-for-bit on that reference (1.63572023e-7 / max(1, 6.93430743)), which the
   test suite re-derives from the evidence file. Tail parameters are O(1) and
   share a scale, so A is a converged derivative: 1.5176e1 at eps 1e-6, 1e-8 and
   1e-10 alike, all rows quoted.
   *The shape samples are anchored one step upstream by the same move.* A
   seventh stage, `shape_coefficients`, probes the concatenated per-site ISA-A
   coefficient vector W, holding the shipped tails fixed — which is the
   algorithm's own boundary, since `IsaAController::step` fits iteration n+1's
   tails from iteration n's coefficients and the final sampling never refits.
   Here the regrouping is an *exact reconstruction* rather than a coincidence:
   two of the three water sites have a clamped denominator, so their
   coefficients cannot exceed one and the concatenated denominator is exactly
   the unclamped site's, giving 2.07295715e-10 / max(1, 6.87695288) =
   **3.0143541609579276e-11** — deliberately not the per-site maximum
   2.5755e-10, which would overstate the array error 6.9×, because unlike the
   tails the largest error and the largest coefficient sit on different sites.
   That recorded error **meets** its requirement of 3.22e-8 (A=3.108e1, C12;
   3.51e-8 at A=2.848e1 for α₃) with a 1.07e3× margin and would first fail at a
   property tolerance of 9.4e-10; A converges to 8e-5 relative across eps 1e-6 →
   1e-10, all 28 rows quoted, self-consistency exactly 0.0.
   What stays open is the shape-sample *array itself*, uncompared in both
   metrics: absolutely because no absolute requirement for it is well posed,
   relatively because no error for it was ever recorded in that metric. That is
   a missing measurement against a same-input reference, not a mismatched
   association, and the coefficient stage measures the input that generates the
   array, not the array. None of this reopens TODO9's strict trajectory parity,
   which is still FAIL (the shape-coefficient rows belong to that same
   comparison), and the amplifications were measured on the demo water rather
   than on the comparator's own input.
   *Two input forms, where two errors were recorded.* Drho-C and fitted-OV each
   recorded a coefficient error **and** an error on the field those coefficients
   expand to (sampled density 6.13011642e-7; sampled transition density
   5.03034372e-8). Those stages are now probed **once** and reported **twice**,
   the second row dividing the same property defect by the defect the same
   perturbation induces in the sampled field, using the comparator's own sampler
   and metric (`IsaFixedDensity` on the shipped ISA grid; screened auxiliary
   samples contracted with the fitted legs; both streamed in blocks, so the point
   count sets no array size). A bare recorded number names the probed array alone
   and is refused against a sampled-form requirement, as is any sampled-form
   error outside the absolute metric. Linearity is measured, not assumed: the
   sampled/probed defect ratio is eps-independent to 6-7 digits over eps 1e-6,
   1e-8 and 1e-10 in every direction (water 4.414114 / 1.202447 / 1.159737 /
   3.229728; He 22.238682 / 13.048567 / 14.731959 / 14.403853), so the sampled
   row inherits the probed row's linearity defect and the two are quoted or
   withheld together.
   *What the sampled form closes, and what it does not.* On He fitted-OV it
   satisfies **four of the seven** property groups at 1e-6 (α₁ at 0.60× of its
   requirement, α₂ 0.14×, C6 0.21×, C8 0.31×) and misses the other three by
   1.93×–3.64× (binding C10 needs 1.38e-8 at A=7.24e1; α₃ needs 2.51e-8 at
   A=3.978e1), against ~3.7e4× for the coefficient form. On water Drho-C it
   closes **none**: 14.3× (α₁) to 68.3× (α₃) short of 8.98e-9 at A=1.11e2,
   against ~3.6e5× for the coefficient form. The second form therefore removes
   four to five orders of the apparent gap and leaves one to two; both binding
   rows stay **unsatisfied**, and the recorded sampled-density error supports
   Drho-C-induced property defects of 1.43e-5 to 6.83e-5 in the max-scaled
   metric, not 1e-6. It remains a lower bound over the **range** of each stage's
   own expansion map — `IsaFixedDensity` has no tabulated constructor, so a
   sampled-density *stage* cannot be perturbed freely — and the two recorded
   numbers differ by three orders because the recorded coefficient error sits
   near that map's null space, which is a property of the recorded error and not
   of the map.
   Tests: `tests/pytests/test_isapol_budget.py` (29 quick + 1 long).
6. **End-to-end match against the reference `Cn` potential.** *Status: the
   reference structure is decoded and a first comparison exists; the models do
   not yet coincide, and the gap is localized rather than closed.*
   The reference family is
   `.pi/camcasp-build/tests/H2O_props/{dalton,nwchem,psi4}/check/L2H1/H2O_ref_wt3_L2_Cn.pot`.
   Their localization headers are byte-identical (`Limit: 2`, `WSM-Limit: 2`,
   `H-Limit: 1`, `Loc algorithm: LW`, `Weight: 3`, `Weight coeff: 0.001`,
   `SVD threshold: 0.0`, `Pol Cutoff: 0.0001`, `NoRefine?: False`,
   `Model file: H2O.pdef`, `Axes file: H2O.axes`), and the `H2O.axes` files are
   identical, so the family's spread is a **yardstick that has to be read
   carefully**: molecular isotropic C6 45.0503 / 45.7307 / 46.6174 (3.5% wide),
   O-O site C6 18.26039 / 18.76416 / 19.27258 (5.5%), O-O C10 3672.440 /
   3891.527 / 4106.707 (11.8%); the separate aVTZ `wt4` example gives 43.8903.
   The three `H2O-avtz.clt` inputs differ in **two** things, not one: `SCFcode`
   *and* `HOMO` (−0.33187 / −0.3980 / −0.3989 against a shared
   `I.P. 12.62063 eV`), so with CamCASP's own 27.21136 eV/Eh they declare GRAC
   shifts 0.13193004527520863 / 0.06580004527520861 / **0.06490004527520865**.
   Only the Psi4 row's shift is the one our chain declares, and the DALTON row
   carries twice that shift — a different asymptotic correction, not just a
   different SCF code. The defensible yardstick is therefore the nwchem/psi4
   pair (1.9% on molecular C6), with DALTON reported but not counted as
   back-end noise.
   Per-pair order truncation in the reference is exactly the admissible
   `n = 2(l_a+l_b+1)` at `Limit 2`/`H-Limit 1`: O-O {6,8,10}, H-O {6,8},
   H-H {6}. Comparison of our accepted aVTZ chains (matched AUX, uniform rank 3)
   against the Psi4-back-end reference, per **ordered site pair** as printed:

   | quantity | `direct_ov` | `lambda1000` (traced NN) | reference (psi4 row) |
   | --- | --- | --- | --- |
   | molecular isotropic C6 | 46.89713 (**+0.600%**) | 46.76829 (**+0.324%**) | 46.61741 |
   | O-O C6 | 25.60926 (+32.88%) | 25.52363 (+32.43%) | 19.27258 |
   | O-O C8 | 503.0882 (+22.63%) | 508.2636 (+23.89%) | 410.2453 |
   | O-O C10 | 10666.265 (+159.73%) | 11406.429 (+177.75%) | 4106.707 |
   | H-O C6 | 4.51717 (−15.39%) | 4.50724 (−15.58%) | 5.338895 |
   | H-O C8 | 71.45023 (+24.60%) | 72.39140 (+26.25%) | 57.3419 |
   | H-H C6 | 0.80479 (−46.25%) | 0.80392 (−46.31%) | 1.497312 |

   The one **partition-invariant** number, the molecular isotropic C6 total, is
   inside the reference family's own nwchem/psi4 spread for both routes, and the
   constrained-NN route is the closer of the two. Every site-resolved split is
   far outside it, in a consistent pattern — O-O too large and worsening with
   order, H-H ~46% too small — which localizes the disagreement to the two stages
   our chain does not yet apply: PFIT refinement (item 1 part B now drives it
   from the constrained-NN chain's own anchors, but against native direct-OV
   point-charge targets on a caller-declared lattice, not the reference's) and
   the rank-limited `.pdef` model.
   *A measured structural obstacle to the second of those.* Declaring the
   reference's per-site rank limits directly is currently **impossible**, not
   merely unimplemented: `native_properties` rejects non-uniform ranks with
   `LW pipeline requires uniform explicit rank3 or rank4` (`isapol_native.py:217`),
   because LW's workspace is uniform rank0–3. So `{O: 2, H: 1}` fails before any
   compute (`.pi/audit/avtz-rank-limited.json`), and our uniform rank 3 admits
   `{6,8,10,12}` on every pair where the reference admits fewer — the order
   totals are therefore **structurally different quantities** and were not
   quoted against each other. Closing this needs either a rank-limited LW path
   with its own gate, or an explicitly labelled post-hoc truncation of the
   reported components (which is still a different model from fitting under the
   restriction, and must be labelled as such — not as agreement).
   The recoupled anisotropic track stays separate: 377 nonzero recoupled rows in
   the L2H1 reference (O-O 258, H-O 86, H-H 33) have no counterpart, because
   `AnisotropicDispersion.kind == 'orientation_resolved_scalars_not_recoupled_components'`
   is a *different representation*, not a coarser one. Note that only the
   **isotropic** row vanishes at the odd orders; the recoupled rows do not
   (nonzero components per order: O-O 16/46/95/118/103 at n=6…10, H-O 23/33/53
   at n=6…8, H-H 33 at n=6), so this track cannot be dismissed as zeros. The
   committed fixture therefore carries a census of it and deliberately not its
   values, so that no later test can start quoting a recoupled component against
   an orientation-resolved scalar.
   *Committed as of this item:* `tests/pytests/data_isapol/oracle/read_cn_pot.py`
   (decodes printed `.pot`/`.clt`/`.axes` output only; compiles nothing, reads no
   CamCASP source, never writes the reference tree),
   `tests/pytests/data_isapol/camcasp_cn_pot_h2o_l2h1.json` (all three back-end
   rows, with the MIT notice travelling inside it) and
   `tests/pytests/test_isapol_reference_dispersion.py` (4 quick + 3 long). No
   tolerance is asserted against the reference anywhere in that test: what is
   asserted is where our totals sit relative to the family's own internal spread,
   which the model mismatch cannot explain away.
   *The polarizability-level follow-on is now done as well*, on a **second**
   reference case that must never be conflated with the L2H1 one above:
   `.pi/camcasp-build/examples/properties/H2O` (weight type 4, prefix `H2O_aTZ`,
   `Scf-code DALTON`, PBE0/aug-cc-pVTZ, CKS with `Hessians Internal`,
   `DF-TYPE-MONOMER NN`, and **bond** axes `H1  z from O to H1   x from H2 to H1`
   against the other case's `z global Z`). Decoded by
   `tests/pytests/data_isapol/oracle/read_local_pol.py` into
   `camcasp_local_pol_h2o_atz_wt4.json` and measured by
   `tests/pytests/test_isapol_camcasp_local_pol_oracle.py` (9 tests, no SCF,
   1.3 s). Full write-up in SPEC §7 "Closed Casimir-Polder oracle on a second
   reference case"; the load-bearing results:
   - **The Casimir-Polder step is closed against the reference.** Its printed
     refined local tensors in, reduced to per-rank isotropic scalars, and our own
     `isa_isotropic_dispersion` reproduces every `C_n` it prints — worst relative
     **2.03e-7**, at printed precision. Both inputs are the reference's, so what
     is under test is ours alone. The `(-1)^l sqrt(2l+1)` recoupling convention
     that reduction rests on is separately measured against the same file's own
     `00(l l)` rows, 40 entries, worst **3.14e-6**, sign included.
   - **Retract any note claiming a Casimir grid coverage gap.** `Quad 10` /
     `Beta 0.5` is 11 frequencies (the reference names its own file `f11`), and
     `core.CasimirGrid(10, .5)` *is* that grid: `n_freq()` is the Gauss-Legendre
     order and the object holds `n_freq + 1` nodes, index 0 static
     (`casimir_grid.h:120` and the `n_freq` parameter doc above it, whose wording
     is tightened in this commit for exactly this reason). An earlier
     "static + 9 dynamic" reading was an off-by-one in a probe loop, not a cap.
   - **The reference is internally inconsistent by 0.26% on a
     partition-invariant number**: molecular isotropic alpha is 9.247357 from its
     rank-4 distributed tensor translated to the origin, 9.271584 from its
     refined local tensors rotated by the declared axes and summed. That is its
     own refinement's distortion — same order as the 0.024 anchor movement our
     constrained-NN refinement makes — and it is compounded by the refinement
     lattice being 500 `Random`/`Seed 1` points in `LoLim 2.0`..`HiLim 4.0`,
     which is not reproducible without CamCASP's RNG. Its refined tensors
     therefore cannot be matched exactly by construction, and this must be stated
     wherever the comparison appears.
   - **Our rank-4 chain's residual excess is in the response step**, not the
     partition and not the refinement, because the translated molecular alpha is
     partition-invariant: 9.870961 (`direct_ov`, +6.75%) and 9.839675 (traced
     lambda=1000 NN, +6.41%) against its 9.247357, at the matched protocol
     (E = -76.37966827740804, HOMO = -0.3989569916800326 reproducing its declared
     `HOMO -0.3989` to every printed digit, nbf 92). Candidates to be labelled
     and not absorbed: our `alda_slater_pw92` with `exact_exchange=.25,
     local_scale=.75` against CKS/`Hessians Internal`; GRAC form; 99/590 against
     `Angular 100 / Radial 60`; and their `Eta = 0.0005`, which we do not apply.
   - The rank-4 reference quantity `H2O_aTZ_NL4_static.pol` (75 = 3x25, full
     double precision) is symmetric to 5.81e-11 with
     `sum_ab alpha^ab_{00,00} = -3.11e-12`, but its charge-flow sum rules close
     only to **7.68e-7** — the level any comparison against it is limited to. The
     translation sign is measured, not assumed (`+` leaves the C2v-forbidden xz
     element at 7.03e-8, `-` at 2.07e-6).
   - The 17 `.pdef` variables reproduce all 33 printed 9x9 blocks **exactly**
     (`reconstruction_error == 0`), so the declared model is provably the fitted
     model and 187 numbers replace 2673 losslessly. `H2 H2 COPY H1 H1` holds
     bit-identically in local axes while the globalized dipole blocks differ by
     `2|alpha_xz| = 0.46794962` — the reference's own confirmation of the
     frame/COPY finding that `copy_anchor_discrepancy` measures.
   - Only `n = 6` is a **complete** molecular isotropic total under `L2`/rank-1 H;
     C8/C10 pair sums are structurally partial and labelled so. That total is
     43.890270 here against 46.617408 for the other case's psi4 row — the same
     property from the same code, **2.727138 apart** across declaration choices
     (6.21% of this case's total, 5.85% of the other's), so the reference
     family's own spread bounds what agreement with "the" reference number can
     mean.

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
  certification; C12 is partial. Rank 4 is now accepted over the 13 ordered pairs
  upstream defines (`la+lap<=6`) and certified against built casimir at its write
  precision; (3,4),(4,3),(4,4) are uninitialized upstream and therefore remain
  structurally absent, so the C9..C12 quadruples needing them are reported missing
  coverage rather than zeroed.
- Matched native ISA reference comparisons, raw tails/Drho/fitted-OV conditioning,
  fitted versus direct response and strict recorded-input LW defects retain
  distinct gates. See SPEC/PROVISIONAL_ACCEPTANCE.md; no blanket tolerance waiver.
  These are now *property-anchored* (section 5 item 5): Drho-C misses its 1e-6
  requirement by ~6 orders and fitted-OV by ~4.5, while the raw-tail parameter
  error meets its 6.59e-8 requirement with a 2.8× margin and the ISA-A
  shape-coefficient error meets its 3.22e-8 requirement with a 1.07e3× margin.
  In the second form those two comparisons also recorded — the sampled density
  and the sampled transition density, which is what every downstream stage
  actually reads — the misses shrink to 68× (water Drho-C, **no** property group
  closing) and 3.6× (He fitted-OV, four of seven groups closing), narrowing the
  gap by four to five orders without closing either stage.
  The shape-sample array stays uncompared in both metrics pending an
  elementwise-relative measurement against a same-input reference; only the
  coefficients that generate it are anchored. Meeting a requirement is *not* a
  certification: the amplifications are directional lower bounds, so they bound
  nothing from above, and no stage is certified to a property tolerance.
- The fitted-auxiliary response route is **rejected by strict production LW on
  water at the recorded lambda1** (charge-sum ~4.1e-4 at every frequency), so it
  has no accepted multi-atom chain *at that declared model*; every lambda1
  fitted-route statement here rests on the monatomic He chain, where the
  partition stages are exactly degenerate. The cause is now measured and is a
  producer defect, not a gate: the penalty `A = J + lambda*q q^T` converges the
  (already exactly zero) transition charge as 1/lambda, and strict production LW
  **accepts** the water fitted chain at every declared lambda >= 1e3 — including
  **lambda1000, which is the traced constrained-NN route's own exported penalty**
  (`input-sum-rule` 4.65e-7, 2.1x under the gate); at lambda1e4 that residual
  reaches the fit-free `direct_ov` route's own quadrature floor (2.33e-8 vs
  2.28e-8), and the two accepted models differ by only 1.3e-7 in the raw
  tensors — see SPEC §6 and
  `tests/pytests/test_isapol_native_charge_penalty.py`. That chain is a
  differently declared model and must never be compared against a recorded
  lambda1 number.
- Full native SCF/PFIT/GRAC matched protocol and modern ISA preset are not closed.
- The reference's refined local tensors cannot be matched exactly **by
  construction**: `examples/properties/H2O` refines on 500 `Random` points with
  `Seed 1` between `LoLim 2.0` and `HiLim 4.0`, and that lattice is not
  reproducible without CamCASP's RNG. State this wherever a refined-tensor
  comparison appears. What *is* closed against that case is the Casimir-Polder
  step (worst 2.03e-7) and the recoupling convention (worst 3.14e-6); what is
  bounded rather than closed is the polarizability, by the reference's own 0.26%
  internal inconsistency and the family's 2.727138 (6.21%/5.85%) cross-protocol
  C6 spread (section 5 item 6, SPEC §7).
- The residual molecular-alpha excess of our rank-4 chain against that case,
  +6.75% (`direct_ov`) / +6.41% (traced NN), is localized to the **response
  step**, and the response step is now **closed on identical orbitals**: the
  excess belongs to the asymptotic-correction form (candidate 2), not to the
  propagator. Evidence, all external-orbital track and to be labelled as such:
  - The reference propagator was rebuilt from CamCASP's MIT sources
    (`NAME=camcasp`; serial `make` -- the makefile is not parallel-safe;
    `-fallow-argument-mismatch` for `gamint.F`'s legacy-F77 rank mismatches;
    `src/tests` is on the vpath) and **certified digit-for-digit on all 80
    numeric lines** of the shipped `examples/energies/He2/aTZ_MC/check/OUT/He2.out`.
    This was necessary because no shipped case both exports usable MO vectors and
    prints a polarizability at admissible size.
  - The shipped 15-digit DALTON orbitals `He2-A-asc.movecs` decode into Psi4 at
    `max |C^T S_psi C - I| = 1.132e-14` (`PMAP = [2,0,1]`, `DMAP = [2,3,1,4,0]`),
    once a `.gbs` reproducing DALTON's 6-primitive first-S aug-cc-pVTZ
    contraction in DALTON's shell order is supplied. Psi4's shipped He block
    spans the same space (1e-10 in energy) but its coefficients are not
    interchangeable.
  - On those same orbitals at the declared protocol (`CKS` / `Hessians Internal`
    / `DF with constraints` / `NN` / `Eta = 0.0` / `Lambda = 1000`), CamCASP's
    shipped grid gives 1.416255 and ours 1.41637208 (+8.27e-05 relative). That
    difference is the **reference's own quadrature error**: refining its
    `Angular`/`Radial` grid moves it 9.7e-05 toward ours (1.416255 -> 1.416323 ->
    1.416370 -> 1.416358 -> 1.416352 over 6490 -> 580146 atom points), after which
    it oscillates within +/-9e-06. Ours is grid-converged to nine digits from 6490
    to 193826 points, with the grid confirmed live by falsification (54 points
    gives 1.42171160, 1450 gives 1.41636887). Residual against the reference's
    converged plateau: **+2.0e-05 absolute / +1.4e-05 relative**, the size of the
    reference's own remaining grid noise.
  - Not the DF penalty: lambda in {1e2 ... 1e8} all give exactly 1.41637208.
    `direct_ov` on the same orbitals gives 1.41129611, so constrained-NN is the
    correct comparison space, as declared.
  - Being a same-input comparison, this bounds candidates 1, 3, 4, 5, 6, 7 and
    the previously unmeasured 8 (auxiliary-space `KerOVOV = Dov_c Ker Dov_c^T`,
    `prop_utilities.F90:329-441`) **in aggregate** at 1.4e-05. It does not bound
    candidate 2, which by elimination carries the whole +1.36%/+1.67% gap seen
    with Psi4 GRAC orbitals, and which is channel-resolved for He (the p channel
    is 0.58% low at the declared shift). None of these is absorbed into a
    tolerance; the aggregate bound is stated as a bound.
- What remains open in the response step is therefore candidate 2 alone: Psi4's
  LB94-based GRAC (alpha=0.5, beta=40) against DALTON's three distinct declared
  Tozer-Handy forms (plain `.DFTAC`, `MULTPOLE TANH`, and `MULTPOLE TANH
  VARSHIFT`), related by `v_xc(inf) = IP_declared - |E_HOMO|`. The AC-form
  sensitivity band alone spans 1.85% of the admissible isotropic value, so the
  observed gap sits inside it; that is a bound, not an explanation, and the H2O
  same-orbital test needed to close it requires a 15-digit water movecs
  (`examples/energy-scan/water2-B` carries only ~8 and is abandoned).
- Per-site rank limits are still not declarable in the LW pipeline
  (`isapol_native.py:217` requires uniform explicit rank 3 or rank 4), so the
  reference's `L2`/rank-1-H model cannot be *fitted* natively. Closing that needs
  either a rank-limited LW path with its own gate or an explicitly labelled
  post-hoc truncation of reported components — a different model from fitting
  under the restriction, and never quotable as agreement.
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

### Rebuilding the reference CamCASP propagator (external-orbital track)

The shipped `.pi/camcasp-build` tree has no `camcasp` binary, and
`densfit_prop.F90`/`prop_utilities.F90` are only pulled by `NAME=camcasp`
(`src/SRCS.list`). Build it *outside* the reference tree -- the reference tree
must not be modified -- and drive it through a short path alias, because
`src/precision.f90` sets `lchar = 80` while `free_format_reader.F90`'s `reada`
truncates at 64 characters:

```bash
CAM="$ROOT/.pi/camcasp-build"
CC="$SP/ccbuild"                       # $SP = session scratchpad
rsync -a --exclude .git "$CAM/" "$CC/" # keep src/tests: it is on the vpath
sed -i 's/^FFLAGS\(2\|3\)\? :=/&  -fallow-argument-mismatch/' \
  "$CC/x86-64/gfortran/exe/Flags"      # gamint.F legacy-F77 rank mismatches
ln -sfn "$CC" /tmp/ccb                 # name only; all real files stay in $SP
cd "$CC" && PATH="$HOME/miniconda3/envs/p4_ci/bin:$PATH" make -j 1 NAME=camcasp \
  LIBDIRS="-L/usr/lib/x86_64-linux-gnu" \
  LIBS="-llapack -lblas -lpthread -lm -ldl" LDFLAGS=""
# The propagator reads the .cks on STDIN (bin/camcasp.py:1760):
CAMCASP=/tmp/ccb /tmp/ccb/bin/camcasp < job.cks > job.out
```

`make -j 1` is required: separate `%.o` and `%.mod` rules compile the same
source twice. Certify any rebuild by reproducing
`examples/energies/He2/aTZ_MC/check/OUT/He2.out` digit-for-digit before quoting
a number from it.
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
