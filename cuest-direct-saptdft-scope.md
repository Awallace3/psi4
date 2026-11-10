# cuEST direct J/K for SAPT(DFT): status and scope

Investigated 2026-10-16; independent call-site audit incorporated.
**Scoping complete; implementation remains gated.** No implementation, builds, chemistry,
job submissions, or changes to frozen DF experiments were made here.

## 1. Current answer

**Not hooked up in this SAPT(DFT) worktree.** Current `jk.cc` selects
`cuESTJK` only for `MEM_DF`/`DISK_DF` with `USE_CUEST`. Its `DIRECT` branch
constructs CPU `DirectJK`. GPU XC and cuBLAS do not make this a GPU direct-JK
implementation.

`SCF_TYPE=CUEST` appearing in the accepted option strings is not a working
alternative: this parent factory has no corresponding branch. The implemented
cuEST route here is stored DF.

**A separate direct-SCF prototype already exists; reuse it rather than starting
another implementation.**

| Item | Verified state |
|---|---|
| Compiled source | `7896a4bd12905b050f0ba2f70a4c40d787b509b1` |
| Adapter | `psi4/src/psi4/libfock/cuESTDirectJK.{h,cc}` in that commit |
| Selection in prototype | `SCF_TYPE=DIRECT`, `USE_CUEST=True` |
| Dependency | Private cuEST0.2.50 CUDA13 SDK, direct compile/link capability gate |
| Build | Phoenix14080574, COMPLETED0:0,9m38s |
| Core SHA256 | `311f1be6fd80cfad502f1605e48d8b7ecf66f966e9e7a535efd6f3f0630f39b3` |
| SDK SHA256 | `65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c` |
| Later validation harness | `f4289f3debe49fe78f24cfd00d0c815b6413850a` |
| GPU numerical validation | Not established: manifest says `gpu_test_run=false`; later investigation record says not run |
| Supported contraction | Four-center J and symmetric-density K, including configured full-/range-separated exchange |
| SAPT blocker | Nonsymmetric K is explicitly rejected |

The prototype is a **compiled, capability-gated adapter**, not demonstrated
GPU correctness or a complete SAPT(DFT) backend.

Remote build manifest:
`/storage/project/r-cs207-0/awallace43/runs/cuest-direct/20261015-7896a4b-r1/build-stage/build-manifest.json`.
Prototype rationale and tests are recorded in that commit's
`outputs/cuest-direct-investigation.md`,
`devtools/benchmarks/cuest_direct/README.md`, and
`tests/pytests/test_cuest_direct_jk.py`.

## 2. Hard blocker: general exchange, not just dispatch

The installed SDK was inspected directly, including exported symbols.
It provides direct-plan creation/destruction, direct Coulomb, symmetric
exchange, and derivative variants. **No direct nonsymmetric-exchange entry
point was found.**

Its `direct_exchange_compute_api.h:65–76` defines

\[
K_{pr}(D)=\sum_{qs}(pq|rs)D_{qs}
\]

and explicitly requires symmetric density matrices; nonsymmetric input yields
undefined results. The prototype therefore throws when K is requested with
`lr_symmetric_ == false` (`cuESTDirectJK.cc:222–224`).

SAPT exchange and response routinely pass **distinct** `C_left`/`C_right`,
forming \(D=L R^T\), not necessarily symmetric or positive semidefinite.
The existing cuEST **DF** adapter deliberately uses a separate nonsymmetric
exchange entry point for those calls.

**Do not silently replace D with (D+Dᵀ)/2.** That preserves J but generally
loses part of K. With an eightfold-symmetric rank-one ERI tensor
\((pq|rs)=B_{pq}B_{rs}\), symmetric \(B=\mathrm{diag}(1,2)\), and
antisymmetric \(D=\left[\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\right]\),
J(D)=0 while K(D)=BDBᵀ is nonzero. A call on the symmetrized density does not
recover that missing antisymmetric contribution. Any more elaborate
reconstruction would be a separate algorithm requiring correctness and
performance validation, not this substitution.

Some response callers ultimately use K+Kᵀ; a proved, explicitly specialized
symmetric-response formulation may be possible there. It is **not** a general
JK replacement and does not establish correctness for SAPT exchange.

## 3. Recommended module and interface

Use the existing **JK seam** and a separate `cuESTDirectJK` adapter.
Keep plan/workspace/density transfer/operator details inside that deep module;
SAPT callers should continue supplying factors and requesting J/K through JK.

Future opt-in selection should reuse the prototype's ordinary options:

```text
SCF_TYPE = DIRECT
USE_CUEST = TRUE
```

This is a proposed integrated interface, **not a working instruction for the
current parent branch**. Existing DF options remain unchanged. Missing SDK
capability or an unsupported contraction must raise a targeted error, never
quietly use CPU DIRECT or DF.

Defaults remain DF; the two explicit selections above are the opt-in. Before
expensive work, a SAPT request against a symmetric-only direct adapter should
fail with a workflow-specific capability error, not complete several SCFs and
then fail at its first nonsymmetric exchange. A small capability predicate at
the JK seam is reasonable if needed for that early check; do not infer it
from the backend name or `USE_CUEST`.

Keep `returns_combined_exchange()` truthful. The prototype returns weighted
`alpha*K + beta*K_erf(omega)` in K and no independent wK. Its older header has
no override for the parent's newer capability method; **porting must add the
correct override and audit consumers**, not copy the class unchanged.

Current parent response code uses the backend capability, but RHF SCF/Fock/
energy paths still inspect global `USE_CUEST` to infer weighted K. Factory
dispatch to CPU `DirectJK` does not justify those assumptions. Audit RHF/UHF
Fock, energy, response and stability consumers together; use the actual
backend's exchange convention rather than the global option.

For the first PBE0 integration, preserve the prototype's actual weighted-K
contract and advertise it correctly. A bare-K/separate-wK adapter would be a
clean alternative, but requires explicitly configured operator plans or other
SDK support; it is not obtained by changing the capability boolean. Supporting
RSH SCF does not by itself remove SAPT's separate-operator restrictions.

## 4. Deliverable phases

### A. Reuse and validate the direct-SCF prototype

1. Port only the direct adapter, factory/CMake capability gates, options and
   tests into an isolated integration worktree based on the chosen SAPT branch.
2. Validate declaration **and linkage** capability; old DF-only SDK builds
   continue working. Pin actual runtime cuEST/CUDA/LibXC libraries: old and
   new SDKs share the same SONAME.
3. Correct backend exchange-convention dispatch before enabling the route.
4. Run allocated-GPU numerical tests against **CPU DirectJK**, not DF, first
   with GPU XC disabled to isolate the integral backend.
5. Keep strict FP64 and `CUEST_DIRECT_JIT=OFF` for the first validation cohort.
   AUTO/ON/JIT performance are later, separately labelled experiments.

Start the SAPT milestone with **HF/PBE0 and no range separation**.
The parent's `sapt_proc.py:1455–1462` already rejects range-separated
functionals under cuEST; retain that guard until backend-specific operator
capabilities and dedicated tests justify a change.

**Exit gate:** symmetric RHF/UHF/hybrid/RSH matrices and energies correct,
actual direct backend proven, lifecycle/parameter changes safe. This gate
does not declare SAPT exchange or CPKS fully supported.

### B. Close the nonsymmetric-K contract

Preferred: obtain a vendor-supported direct nonsymmetric exchange interface,
with workspace queries, signed arbitrary factors/densities, batches, operator
semantics and reference tests.

Alternative: a separately scoped direct four-center contraction implementation
that retains the missing antisymmetric part. First establish whether the SDK
exposes suitable integral tiles; do not assume its symmetric exchange routine
can be repurposed safely. This is materially larger work.

Until that contract is available, direct SCF can be evaluated separately, or an
explicitly named **direct-SCF / DF-SAPT** experiment can be proposed.
Such a mixed route is **not full direct SAPT**, must retain DF memory accounting,
and is not an implicit fallback or a solution proven to remove the memory cap.

### C. Enable opt-in SAPT(DFT) and measure it

Only after B:

- Route neutral/cation GRAC, delta-HF dimer/monomers, DFT monomers, SAPT
  exchange, and coupled induction through the intended backend.
- Preserve explicit user selection through driver option stashing/reuse.
- Reconfigure reused DFT builders for **bare** SAPT exchange correctly; do not
  retain a PBE0 fractional-exchange operator or multiply fractions twice.
- Keep GPU XC/grid switching/CPKS final-grid guards independent of direct J/K.
- On this parent, initially validate coupled response with `CUEST_XC=False`
  (native XC). GPU XC response is implemented only on the unmerged CPKS sibling;
  enabling the direct JK adapter does not bring that implementation with it.
  Fully GPU CPKS needs an explicit subsequent integration of both changes.
- Validate complete SAPT components and solver residuals before performance.
- Measure direct as a **new cohort**, with CPU DirectJK and DF/cuEST-DF controls
  separately identified; do not overwrite any stored-DF result.

## 5. Required contraction and lifecycle coverage

- J-only, K-only, J+K; multiple densities with preserved batch ordering.
- Preserve each pair's output independently: SAPT unpacks three J/K matrices.
  UHF response submits all alpha densities then all beta densities internally;
  do not combine spins/pairs or silently reorder them.
- RKS/RHF and UKS/UHF alpha/beta densities.
- General signed, nonsymmetric and symmetric-indefinite inputs, not just
  occupied-orbital positive densities. Include L→−L, transpose and linearity.
- Explicit equal left/right lists are still a nonsymmetric-contract request:
  the base class sets `lr_symmetric_` from whether the right list is absent,
  not by comparing its contents (`jk.cc:644–650`). Cover this case as well as
  changing occupations, zero-column factors and shrinking response batches.
- Correct distinction: J ignores antisymmetric D; K generally does not.
- Full, long-range and short-range operators with valid fractions/omega.
  Distinguish separate wK from weighted combined K explicitly.
- Parameter changes after initialize: fractions, omega, cutoff and task flags;
  transactional rebuilds, clear errors, failed call followed by valid reuse.
- Interleaved objects sharing the cuEST handle; restore requested math mode
  per computation, not whatever another object left on the handle.
- Initialize/finalize/reinitialize, exception cleanup and owned versus borrowed
  basis/plan/workspace lifetimes; no cross-grid or cross-molecule stale state.
- AO/USO transformations and symmetry where claimed; initial SAPT validation
  can be C1, but broader adapter claims require corresponding tests.
- If the SDK fails partway through a batch, do not treat earlier/stale outputs
  as a valid completed computation.

## 6. SAD, preliminary guesses and auxiliary bases

For an initially strict direct-only SCF test, disable `DF_SCF_GUESS` and use an
explicitly identified CPU SAD route. Otherwise a DF preliminary calculation or
the current GPU-SAD implementation can allocate a different JK backend.

Atomic SAD **bypasses the main factory**: `sad.cc:272–351` constructs its own
builder, selecting DF cuESTJK for fitting SAD or CPU DirectJK for nonfitting
SAD. The molecular direct-cuEST factory branch therefore does not accelerate
SAD automatically. Respect `SAD_SCF_TYPE`, atomic bare-HF exchange and fractional
occupations; direct GPU SAD is a separate deliverable.

Decide and record atomic SAD policy separately before claiming whole-call
GPU-direct coverage. A small atomic DF/SAD subcalculation is not the same as
a stored dimer DF tensor, but it must not be hidden in a “direct only” label.

The direct-JK adapter itself must not require a DF auxiliary basis or build a
stored three-index tensor. SAPT's other method stages may still need auxiliary
bases; especially do not claim that DIRECT solves FDDS/DFHelper memory use.
The first complete workflow target should remain PBE0 SAPT(DFT)-D4(I), not all
SAPT variants, F-SAPT, dispersion response or analytic derivatives.

Existing no-auxiliary seams are already present:
`proc.py:1457–1474` and `python_helpers.py:490–505` supply a zero auxiliary basis
for ordinary DIRECT; `scf_iterator.py:77–97` is the explicit optional DF-guess
detour. Avoid introducing a redundant SAPT-only constructor.

## 7. Validation matrix and acceptance gates

1. **Build/routing:** new SDK, old SDK, missing declaration/symbol and conflicting
   runtime SONAME; direct request errors honestly when unavailable; existing
   DF and CPU DIRECT behavior preserved.
2. **Low-level numerical oracle:** CPU DirectJK at matched basis/cutoff;
   J/K matrices, operator fractions, batches, signed/nonsymmetric inputs,
   transpose/linearity, screening and symmetry. Reject invalid use before
   calling the symmetric-only SDK.
3. **SCF:** RHF/UHF, PBE0 and selected range-separated functionals, CPU XC first;
   then GPU XC with quadrature differences independently qualified.
4. **Response:** same-orbital Hx/residual checks, RKS singlet/triplet and UKS
   where supported, shrinking/zero RHS and repeated solves. Compare complete
   CPHF/CPKS solves, not only one matrix action.
5. **SAPT:** water dimer, peptide, nanotube, Protein83; exact component
   decomposition, delta-HF and fixed/iterative GRAC variants. Validate native
   CPU DirectJK parity before comparing to a DF calculation with fitting error.
6. **Capacity/performance:** Protein157 and full aug-cc-pVDZ only after earlier
   gates. Record persistent plan/pair-list storage, maximum scratch, other live
   builders, sampled peak and complete energy() wall in fresh processes.

Direct avoids the stored auxiliary×AO-pair DF tensor, but still uses pair lists,
plans and temporary workspace and recomputes integrals. **Neither fitting on
H200 nor being faster than cuEST DF is established.** Include construction,
all SCFs/response actions and cleanup; do not compare only kernel timers.

## 8. Non-goals and experiment protection

- No silent DF→DIRECT switch for OOMs or incomplete timings.
- No changes to the frozen scaling cohort `fc851a8346`, CPU14103914, old
  Protein157 runs, builds or queued jobs.
- No new inferno approval, allocation or GPU work follows from this scope.
- No wholesale cherry-pick of the divergent prototype branch.
- No mGGA/VV10 XC-response, FDDS, analytic Hessian or general derivative work.
- No report assertion that full direct SAPT is supported before every required
  contraction and numerical gate passes.

## 9. Repository evidence and completed call-site audit

Parent inspection:

- `psi4/src/psi4/libfock/jk.cc:140–153,193ff` — DF-only cuEST selection / CPU DIRECT.
- `psi4/src/psi4/libfock/cuESTJK.h` — separate DF nonsymmetric-K parameters,
  operator setters and combined-exchange capability.
- `psi4/src/psi4/libfock/jk.h:479–501` — actual-backend exchange convention and setters.
- `psi4/src/psi4/libscf_solver/rhf.cc:208–239,334–373,551ff,1158ff` —
  mixed global-option versus backend-capability consumers needing audit.
- `psi4/driver/procrouting/sapt/sapt_proc.py:142–144` — DF is the default,
  not a mandate overriding every explicit SCF_TYPE choice.
- `psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py` — distinct factor
  exchange/induction calls; existing chain-GEMM GPU acceleration is not direct JK.

Prototype files can be inspected with `git show 7896a4b:<path>`. Later
validation-only pins do not establish GPU validation without final artifacts.

The independent read-only SAPT contract audit
`d0f961534bbc32e90ce1533ad6d04b02e` completed and was retrieved. Important
additional sites:

- `jk.h:70–110`, `jk.cc:330–377,415–731`: documented factor/density contract,
  construction, AO/USO adaptation and central compute pipeline.
- Native SAPT `sapt_jk_terms.py:355–371,439–453` and einsums equivalents
  `sapt_jk_terms_ein.py:2967–2974,3109–3122,3383–3390`: multiple distinct factor
  pairs requiring actual nonsymmetric K.
- `rhf.cc:483–511,570–579`, `uhf.cc:574–636,702–717`: signed response factors
  and subsequent use of K and Kᵀ.
- `sapt_proc.py:517–540,584–585,815–816` and
  `sapt_jk_terms.py:826–870`: builders saved/shared across stages and attached
  to both response wavefunctions.
- `sapt_proc.py:1426–1447,1466–1471`: independent builder construction and
  reset of inherited hybrid exchange fractions to bare SAPT exchange.
- `libmints/basisset.cc:1312–1382`: cached basis-owned cuEST AO handle; direct
  plan must retain its shared owner, not destroy the borrowed handle.
- `DirectJK.cc:88–103,142–197,364–435`: CPU density-screening/incremental-Fock
  state is not automatically provided by a GPU adapter. Reject unimplemented
  settings rather than advertising CPU feature parity.

No tests were run as part of this documentation-only scope. The numerical and
performance work above is an acceptance plan, not completed evidence.
