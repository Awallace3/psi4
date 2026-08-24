# Slicing the `camcasp` branch into upstreamable PRs

Written 2026-08-24. Records which parts of the branch are at parity, which are structurally
separable, and what the first PR should contain.

The full CamCASP implementation is **not** viable for the C_n coefficients — C8/C10/C12 run
20–46 % low against the matching ISA-GRID oracle. This document is about what remains true and
shippable once that is set aside.

---

## Contents

1. [Upstream sync check](#1-upstream-sync-check)
2. [Scoping correction: the diff base](#2-scoping-correction-the-diff-base)
3. [What is actually at parity](#3-what-is-actually-at-parity)
4. [PR 1 scope](#4-pr-1-scope)
5. [Does PR 1 include atomic polarizabilities?](#5-does-pr-1-include-atomic-polarizabilities)
6. [Extraction mechanics](#6-extraction-mechanics)
7. [Excluded, with reasons](#7-excluded-with-reasons)
8. [Open risks](#8-open-risks)

---

## 1. Upstream sync check

Performed 2026-08-24 against `upstream/master` = `da94da4ae` ("fix CI on Windows (#3502)",
2026-08-23).

| check | result |
| ----- | ------ |
| New upstream commits since the last tip merged into `camcasp` (`6a39a6ace`) | **6** |
| Files upstream changed | **13** |
| Files the CamCASP delta changes | **83** |
| Intersection of the two sets | **empty** |
| `git merge-tree --write-tree camcasp upstream/master` | **clean**, exit 0, no conflicts |

The six commits are `cceom/diagSS`, `cceom/form_diagonal`, `cctriples/count_ijk`, `libtrans`
Dimension deprecation, an `mpi4py` import guard, and a Windows CI fix. **Nothing lands in
`libmints`, `libfock`, `libfunctional`, `libscf_solver`, `libqt`, `read_options.cc`,
`export_*.cc`, or any driver file the branch touches.**

Two changed files were checked for semantic rather than textual impact and are irrelevant to us:

- `codedeps.yaml` — win-64 conda build dependencies plus a QCFractal constraint note. It is the
  conda dependency matrix, **not** a source-dependency manifest, so the new `libmints` sources
  need no registration there (they are already added in `libmints/CMakeLists.txt`).
- `psi4/driver/task_base.py` — a QCPortal `>= 0.70.0` version gate.

**Conclusion: no major upstream change impacts this work.** The merge was *verified* clean but
deliberately **not performed** — the working checkout has uncommitted modifications, and the
merge is the branch owner's call. To take it:

```bash
git checkout camcasp && git merge upstream/master
```

Re-run this whole section before cutting the PR if more than a few days pass.

---

## 2. Scoping correction: the diff base

**Do not diff against local `master`.**

```
git diff --stat master...camcasp     ->  2982 files
git diff --stat 6a39a6ace..camcasp   ->    83 files   <-- the real delta
```

Local `master` is ~7 months behind upstream and `camcasp` has merged upstream six times. One of
those merges carries a tree-wide copyright-year bump that touches nearly every source file,
which makes every CamCASP-modified file appear to "overlap" upstream work. The 2982 figure makes
the branch look unslittable when it is not.

The correct base is the second parent of the most recent upstream merge — as of 2026-08-24,
`27e1b15f4^2` = `6a39a6ace`, verified a strict ancestor of `camcasp`. **Recompute it each time;
it advances with every upstream merge.**

The 83-file delta breaks down as:

| category | files | notes |
| -------- | ----- | ----- |
| (a) Modified pre-existing Psi4 files | 35 | +3060 / −140 |
| (b) New CamCASP-only source | 5 | +13,304 |
| (c) Tests | 17 | +15,849, ~616 test functions |
| (d) Docs / specs / reports | 15 | +11,020 |
| (e) Devtools / dev scripts | 11 | +7,701 |

---

## 3. What is actually at parity

The cut line is **structural, not just numerical**. Every stage that agrees with its oracle is
one that takes plain data in and returns plain data out; every stage that disagrees is wired to
the SCF/response/partition chain.

| piece | oracle | agreement |
| ----- | ------ | --------- |
| `make_casimir_grid` | reviewed CASIMIR frequencies | `7.1e-15` absolute |
| `localize_lw`, `lw_graph_pseudoinverse`, `translate_l3_multipoles` | ORIENT, hermetic (reviewed `NL4_000.pol` in, reviewed `L3_000.pol` out) | `5.5e-13` max abs over **all 675 tensor entries** |
| `compute_dispersion` (isotropic C6–C12 recoupling) | CASIMIR, fed the reviewed L3 models directly | `2.5e-7` max relative |
| Spherical → local Cartesian; local → global rotation | reviewed tensors | exact `0.0` |
| PDef symmetry mask | reviewed model definition **and** output | exact both directions; forbidden entries exactly `0.000e+00` |

Verified mechanically rather than taken from the docs: lines **4348–5620** of
`atomic_polarizability.cc` (PDef mask + bond graph + LW + isotropic dispersion) and **59–520**
(tensor-algebra helpers) contain **zero** occurrences of `FrozenResponseContext`, `ISAWeights`,
`ISAPolResponseProvider`, `Wavefunction`, `scf::HF`, `BasisSet`, or `DirectJK`.

The public API at `atomic_polarizability.h:1690-1996` takes only POD types — `RefinedL3Model`,
`SitePairResponse`, `BondGraph`, `FrequencyGrid`, `L3Matrix`, `SitePosition`, `SiteAxes`.
Python bindings already exist (`export_oeprop.cc:123, 1554, 1729`) and take plain matrices.

### Why this slice is credible

Two independent reasons, beyond the tolerances above:

1. **It is already proven innocent of the C_n failure.** §4.5 of `camcasp_psi4_todo.md` recouples
   *both* CamCASP models with our own dispersion engine, so the recoupling is **bit-identical
   across all three columns** of the C8–C12 comparison and only the L3 models differ. §4.6 then
   localizes the deficit entirely to the WSM refinement's *inputs*. The transform being sliced
   out is the part already shown not to be at fault.
2. **Its tests are ungated and analytic.** `test_atomic_polarizability_math.py` (1,177 lines)
   holds 5 Casimir-grid, 12 LW, and 16 dispersion tests validated against **closed-form**
   oracles — `3/pi` integration, single-pole quadrature convergence, the binomial prefactor
   table — not CamCASP literals. They need neither `PSI4_ATOMIC_POLARIZABILITY_PARITY=1` nor the
   reference tree. `test_atomic_polarizability_symmetry.py` (553 lines, 23 tests) covers the
   PDef mask and bond graph with no SCF and no oracle, re-deriving characters numerically so the
   production integer parity table is never mirrored.

---

## 4. PR 1 scope

Agreed scope: **core enablement + the isotropic machinery.**

### 4a. Core enablement — no CamCASP semantics at all

These need no CamCASP context to review, and one is a genuine upstream bug fix. Listed in
descending order of "merge today" confidence.

| change | files | size | note |
| ------ | ----- | ---- | ---- |
| **`SuperFunctional::build_worker()` drops `density_tolerance_`** | `superfunctional.cc` | 1 line | **Real defect.** `build_worker()` copies `deriv_`, `max_points_`, `libxc_xc_func_`, the VV10 block and the GRAC block, but never `density_tolerance_` — so a user-set `DFT_DENSITY_TOLERANCE` is silently lost across the worker boundary. **Lead the PR with this.** |
| `C_DGESVD` libqt wrapper | `lapack_intfc.cc`, `qt.h` | +11 | Pure new symbol beside the existing `C_DGESDD`. Zero behavior change. |
| Lebedev grid accessors | `cubature.h/.cc` | +55 | `lebedev_spherical_grid()`, `..._sizes()`, `..._order()`. Read-only accessors on tables Psi4 already ships; throws with a listing of supported sizes rather than silently falling back. |
| `IntegralFactory` isolated-`Options` seam | `integral.h/.cc`, `twobody.cc` | +56/−9 | Decouples libmints from the global option registry; default path byte-identical. Flag to reviewers that `twobody.cc` now reads options through the factory rather than `Process::environment` — a subtle lifetime/ordering change. |
| Const-correctness batch | `molecule.h`, `functional.h`, `superfunctional.h`, `LibXCfunctional.h`, `export_functional.cc` | ~20 | `clone() const`, `parameters() const`, const getter overloads, `vv10_beta()`. `export_functional.cc` must ship with these (`py::overload_cast`). |
| LibXC provenance getters | `LibXCfunctional.cc/.h` | +26 | `libxc_canonical_name()`, `effective_parameter_map()`. Depends on the const batch. |

**Not a bug fix — do not pitch it as one.** The `DirectJK` change from
`int nthread = df_ints_num_threads_` to `static_cast<int>(ints.size())` was checked against both
call sites: the `ints` vector is filled by a loop bounded by `df_ints_num_threads_`, so the two
are always equal today. It is defensive hardening. Ship it quietly or hold it.

### 4b. Isotropic machinery

The at-parity math from §3, extracted into its own translation unit:

- `make_casimir_grid` and `FrequencyGrid`
- the tensor algebra: spherical↔Cartesian, local↔global rotation, `translate_l3_multipoles`,
  real solid harmonics
- the bond graph and the PDef symmetry mask
- `localize_lw` and `lw_graph_pseudoinverse`
- `compute_dispersion` — isotropic C6/C8/C10/C12 recoupling
- the two ungated test modules from §3
- the corresponding Python bindings, moved out of `export_oeprop.cc` (see §6)

### 4c. Recommendation: stack these as two PRs, not one

§4a has **zero** dependency on §4b. Bundling them makes the trivially-mergeable half wait on
review of the half that needs a physics argument, for no gain. Suggested stack:

- **PR 1** — §4a core enablement. Lead with the `density_tolerance_` fix.
- **PR 2** — §4b isotropic machinery, on top of PR 1.

Recorded as a recommendation only; the agreed scope above is a single PR and either shape works.

---

## 5. Does PR 1 include atomic polarizabilities?

**No — and this is the most important thing in this document.**

The distinction is between two things that share a filename:

### The published `ATOMIC POLARIZABILITIES` variable — excluded

The `(natom, 6)` static distributed dipole polarizability tensors, and their dynamic
`(11·natom, 6)` counterpart, are **not** in this PR and are **not** near parity:

| site | comp | ratio to ISA-GRID |
| ---- | ---- | ----------------- |
| `H1` | `yy` | **`0.847`** ← worst |
| `H1` | `zz` | `0.934` |
| `H1` | `xx` | `0.970` |
| `O` | `zz` | `0.976` |
| `O` | `xx` | `0.983` |
| `O` | `yy` | `0.992` |
| `H1` | `xz` | `1.012` |

Six of seven components sit within 2–5 %, but the worst is 15 % out, and the residual is a known
physics gap: CamCASP's ISA-GRID takes its shape functions from the basis-space ISA-A functional
and applies them pointwise on the grid, while ours is real-space throughout.

More decisively, **producing** them requires exactly the coupled chain this slice excludes: the
SCF triple → `FrozenResponseContext` → `ISAWeights` → `PointResponseData` → `refine_wsm` → LW.
That chain is where the C_n deficit originates.

### The `atomic_polarizability.{cc,h}` files — partially included, by extraction

The at-parity code physically **lives inside** those files. So PR 1 does touch them, but only to
move ~1,800 of the 9,387 lines out. See §6.

### What PR 1 therefore delivers

A verified transform: **given** a distributed dynamic polarizability model, produce localized
site models and isotropic C6–C12. It does **not** produce the model.

⚠️ **Consequence to decide before cutting the PR.** With the polarizability pipeline excluded,
`compute_dispersion` takes a `RefinedL3Model` that a user has no in-Psi4 route to construct.
That leaves the PR without an end-to-end user story, which upstream reviewers will push on.
Three options:

1. **Ship as internal machinery + tests**, framed as foundation work. Honest, but a reviewer may
   ask why it is in the tree with no caller.
2. **Add a documented public Python entry point** accepting a user-supplied distributed model
   (from CamCASP, MBIS, anywhere) and returning C6–C12. Gives a real user story — *"Psi4 can now
   recouple any distributed dynamic polarizability model into dispersion coefficients"* — and the
   binding already exists as `_atomic_polarizability_compute_dispersion`; it needs a public name,
   input validation and docs. **Recommended.**
3. **Drop §4b from PR 1** and ship core enablement alone.

### A second caveat on wording

Do not describe PR 1 as shipping "isotropic dispersion coefficients at parity." The **transform**
is at parity (`2.5e-7` against CASIMIR, fed reviewed models). The **coefficients this branch
currently produces end-to-end** are not: C6 `0.967`, C8 `0.789`, C10 `0.717`, C12 `0.628` of the
ISA-GRID totals. PR 1 ships the former and none of the latter.

---

## 6. Extraction mechanics

This cannot be done by cherry-picking commits. It is a file split.

**Regions to move** out of `atomic_polarizability.cc` (9,387 lines):

| lines | content |
| ----- | ------- |
| 59–520 | tensor-algebra helpers, spherical-harmonic indexing, graph components |
| 4348–5620 | D2h operations, local axis signs, bond graph, PDef mask, LW localization, isotropic dispersion |

**Shared helpers.** The moved regions call exactly **three** functions defined in the coupled part
of the file, all trivial pure arithmetic:

| helper | line | body |
| ------ | ---- | ---- |
| `checked_c1_product` | 855 | 4-line overflow-checked multiply |
| `checked_c1_sum` | 861 | 4-line overflow-checked add |
| `wsm_upper_index` | 3880 | 1-line upper-triangle index |

Move them to a small shared internal header, or duplicate them. Nothing else crosses.

**Boundary types** to bring along, all POD: `FrequencyGrid`, `L3Matrix`, `L3WorkingVector`,
`SitePosition`/`SiteAxes` (`std::array` aliases), `SitePairResponse`, `BondGraph`, `BondTransfer`,
`LocalizationResiduals`, `LocalizedResponse`, `SiteSymmetry`, `PDefDerivation`,
`BondGraphDerivation`, `DispersionRankPair`, `DispersionPlan`, `DispersionDiagnostics`,
`DispersionMatrices`, and `RefinedL3Model`.

`RefinedL3Model` carries a `RefinementDiagnostics` member, which in turn carries a
`WSMRefinementPlan`. Both were inspected and are pure POD — scalars, `std::vector`, `std::string`,
no behavior. They come along as data types with no dependency on the refinement code.

**The anisotropic block is one-directional.** The isotropic region references nothing in the
anisotropic region; the anisotropic region calls `dispersion_rank_pairs` (twice) from the
isotropic one. So isotropic ships cleanly without anisotropic, but not the reverse.

**Prerequisite refactor.** `export_oeprop.cc` grew by 2,298 lines, roughly 90 `m.def` entries of
which all but three are test seams. It should move to a dedicated
`export_atomic_polarizability.cc` **before** any slicing, or every PR fights the same
merge-conflict magnet in a shared file.

---

## 7. Excluded, with reasons

| excluded | reason |
| -------- | ------ |
| **Anisotropic C_n** | **Not near parity.** 10,457 coefficients compared against a partition-matched CASIMIR run, **0 inside `rtol=1e-4`**, with an unresolved `1/\|<l1 0; l2 0 \| j 0>\|` convention difference on the sigma-even sector that the spec states cannot be decided internally. Internal invariants pass; that is not oracle agreement. 222 nonzero `j >= 9` labels have no external oracle at any rank, since `casimir` hard-caps at C12 and `j <= 8`. |
| **Published `ATOMIC POLARIZABILITIES` / dynamic block** | Worst component 15 % out; requires the full coupled chain. §5. |
| **Published `ATOMIC C6`…`C12`** | 0.967 / 0.789 / 0.717 / 0.628 of the ISA-GRID totals. §5. |
| **WSM refinement (`refine_wsm`)** | References `PointResponseData` in 6 places — the contaminated input. The refinement *mathematics* is correct (reproduces PFIT's 104-parameter ledger to `a_l` ratios `0.9999 / 1.0003 / 1.0223` when fed the reference's own inputs), but production feeds it a fit-point cloud that interacts badly with our response. Not a parity slice. |
| **SCF response-provenance seal** | `hf.{h,cc}` +433 plus `rhf`/`uhf`/`rohf`/`cuhf`, wrapping `form_D()` and `finalize()` in `try/catch` on every reference. Deliberate anti-tamper architecture with no meaning outside CamCASP. Needs a design discussion, not a slice. The `converged_ = false` initialization in `common_init()` may be an independent defect fix worth splitting out — **not yet verified**. |
| **`BasisSet::structural_snapshot`** | +145, self-contained and read-only, but its only consumer is the provenance seal. Upstream will ask who uses it. Hold, or bundle with the seal. |
| **ISA weights** | Partition-of-unity is exact (`1.0000`), but populations differ from CamCASP's ISA-GRID by `0.018 e` (O) and `0.006 e` (H) — two different ISA variants. Also structurally bound to a sealed `FrozenResponseContext`, with `create_test_only` the sole constructor. |
| **C-DF partition** | Not reproducible in Psi4; retained only as strict xfails. |

---

## 8. Open risks

1. **No end-to-end user story for §4b.** The blocking decision in §5. Resolve before cutting.
2. **`export_oeprop.cc` must be split first** (§6), or PR 1 carries a 2,298-line diff in a shared
   file.
3. **The extraction is untested as of this writing.** The line ranges, the three shared helpers,
   and the POD-ness of the boundary types were verified by inspection. **The branch was not built
   and the test suite was not run** for this document. Build and run
   `-m mints` before opening the PR.
4. **Naming leaks CamCASP vocabulary into general classes.** `VBase::response_grid()`,
   `response_functional_workers()`, `SuperFunctional::build_response_copy()`. Rename before
   upstreaming.
5. **Recompute the diff base** (§2) at PR time.
6. Independent code and scientific review of the branch remains open (plan Task 8).
