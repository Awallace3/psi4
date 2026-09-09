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
