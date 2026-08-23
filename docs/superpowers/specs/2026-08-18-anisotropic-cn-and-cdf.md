# Anisotropic Dispersion Coefficients and C-DF Partitioning

Specification for the two remaining CamCASP capabilities that Psi4 does not have.
Written 2026-08-18. Companion to
[`plans/2026-07-31-native-camcasp-parity.md`](../plans/2026-07-31-native-camcasp-parity.md),
[the ISA partition spec](2026-08-18-isa-partition-oeprop.md),
[the ISA-GRID oracle](2026-08-18-isa-grid-oracle.md) and
[the isotropic recoupling spec](2026-08-17-isotropic-dispersion-recoupling.md).

Status: **Part A implemented** (2026-08-19; see the task list in A.8 and the corrections
recorded under it). Part B remains proposed.

---

## 0. Scope, and why these two are specified together

Two capabilities are missing, and they are the last two that separate this pipeline from the
CamCASP properties workflow:

| Part | Capability | Plan reference |
| ---- | ---------- | -------------- |
| **A** | **C-DF partitioning** — partition the FDDS by *constrained density fitting* onto atom-centred auxiliary functions, as an alternative to the real-space ISA partition | plan Task G, currently deferred |
| **B** | **Anisotropic `Cn`** — publish the full set of orientation-dependent dispersion coefficients rather than only the isotropic `00 00 0` component | design-spec non-goal; isotropic spec names the missing prerequisite |

They are **independent workstreams** — either can be built without the other — and they are
specified in one document only because they share the clean-room constraints, the oracle
tooling and the validation philosophy. They also sit at opposite ends of the pipeline:

```
   response  ->  [ PARTITION ]  ->  LW  ->  WSM  ->  [ RECOUPLING ]  ->  Cn
                      ^                                    ^
                   Part A                                Part B
```

Neither touches the other's stage. Build them in either order, or in parallel.

### 0.1 Inherited constraints (non-negotiable)

Carried unchanged from the plan's Global Constraints:

- Production code and pytest **must not** invoke, clone, access or read CamCASP, ORIENT, PFIT,
  CASIMIR or `.camcasp-reference/`.
- **Do not** copy ORIENT or CamCASP source, comments, structure or control flow. Implement from
  published equations and independently written specifications.
- Strict TDD: each behaviour fails before its implementation is written. Production pytest uses
  checked-in literals, never generated JSON.
- Mismatches are investigated **by stage invariant**. Tolerances are not loosened and literals
  are not rewritten to make anything pass.
- Preserve full L3 tensors; expose a diagnostic rather than silently altering a model.

### 0.2 A warning that applies to both parts

The single most expensive mistake on this branch was **comparing against a reference that was a
different physical model** and spending weeks debugging a non-defect. Before either part is
gated against an oracle, establish and write down *which* CamCASP model the oracle is, and add
a test that fails loudly if the two are conflated. §A.6 and §B.6 each specify one.

---

# Part A — C-DF partitioning of the FDDS

## A.1 Objective

Reproduce CamCASP's `ALGORITHM: DF` distributed-polarizability partition, so that:

1. the six retained `DF_*` comparisons in `tests/pytests/test_atomic_polarizabilities.py`
   become **live tests at the plan's `rtol=1e-4, atol=1e-5` gate**, not strict xfails;
2. both CamCASP partition schemes are reproducible in Psi4, making the partition an
   explicit, switchable input rather than an implicit property of the implementation;
3. the partition A/B experiment becomes possible with *both* arms native.

Point 1 is the deliverable that matters. The `DF_*` literals record a real CamCASP calculation
at the reviewed protocol, and they are currently the only oracle on this branch that has been
through full scientific review. Reproducing them at `1e-4` would be the strongest parity
statement the project can make.

## A.2 The key structural fact: only one function changes

The reviewed CamCASP run and this pipeline share **everything** except how the
frequency-dependent density susceptibility is distributed over sites. In our code that
difference is confined to the construction of the projection matrix `B`:

```
                     G(i omega)      [identical: same wavefunction, same 25/75 kernel]
                          |
        +-----------------+------------------+
        |                                    |
   ISA route (implemented)             C-DF route (this spec)

   B[A,t,(ia)] = sum_p w_p             B[A,t,(ia)] = sum_{k in A} Q_t^A[k] d_k^{ia}
                 * w_A(r_p)
                 * R_t(r_p - R_A)
                 * phi_i(r_p) phi_a(r_p)
        |                                    |
        +-----------------+------------------+
                          |
             alpha_AB(t,u) = 4 B[A,t,:] G B[B,u,:]^T     [identical algebra]
                          |
              LW  ->  WSM  ->  Casimir-Polder            [identical]
```

`contract_site_pair_response(site_count, projection, response)` is already a **pure** evaluator
over a caller-supplied projection matrix, and `project_transition_multipoles` is already a pure
evaluator over caller-supplied `tau[p,k]`. So C-DF is a **new producer of `B` with the same
layout**, not a new pipeline. Everything from `contract_site_pair_response` onward — including
LW localization, WSM refinement and the dispersion engine, all of which are already at or near
parity — is reused unmodified.

`B` layout to match exactly: **site-major**, 16 components per site in CamCASP order
`00; 10 11c 11s; 20 21c 21s 22c 22s; 30 31c 31s 32c 32s 33c 33s`, columns in the canonical
occupied-major transition order. Rank 0 is required and must be populated (LW localization
consumes ranks 0–3).

## A.3 Equations

### A.3.1 Constrained density fitting of the transition densities

Each transition density is expanded in an atom-centred auxiliary basis `{chi_k}`:

```
rho_ia(r)      = phi_i(r) phi_a(r)
rho_ia~(r)     = sum_k d_k^{ia} chi_k(r)
```

The fit minimises the **Coulomb self-repulsion of the fit error**, which is the standard
robust functional and the one that makes the resulting distributed polarizabilities
well-conditioned:

```
Delta = ( rho_ia - rho_ia~ || rho_ia - rho_ia~ )          (|| denotes 1/|r - r'|)
```

subject to a set of linear constraints `C d = n`. In the baseline (charge-constrained) form
there is a single constraint per transition:

```
sum_k d_k^{ia} q_k = N_ia ,   q_k = integral chi_k(r) dr ,   N_ia = <i|a> = 0  for i != a
```

Define the Coulomb metric and the right-hand side:

```
J_kl   = ( chi_k || chi_l )
b_k^{ia} = ( chi_k || rho_ia )
```

The unconstrained solution is `d = J^-1 b`. With one linear constraint the Lagrange
elimination is closed form:

```
lambda   = ( q^T J^-1 b - N_ia ) / ( q^T J^-1 q )
d^{ia}   = J^-1 b^{ia} - lambda J^-1 q
```

and with a general constraint matrix `C` (rows = constraints):

```
Lambda   = ( C J^-1 C^T )^-1 ( C J^-1 b - n )
d^{ia}   = J^-1 ( b^{ia} - C^T Lambda )
```

**Implement the general form.** The exact constraint set CamCASP applies under
`DF with constraints` / `DF-TYPE-MONOMER NN` is an open question (§A.7), and the general form
makes the answer a data change rather than a code change.

`J` is symmetric positive definite but frequently ill-conditioned for a large auxiliary basis.
**Do not** form `J^-1` explicitly. Use a pivoted Cholesky or an SVD with an explicit,
diagnosed rank cutoff, following the precedent already set in
`solve_constrained_least_squares` — and note the lesson recorded there: **a column/eigenvalue
cutoff must be relative to the largest retained scale, never an absolute magnitude**, or the
retained rank silently becomes a function of the basis exponents.

### A.3.2 Site projection

Each auxiliary function `chi_k` is centred on exactly one atom `A(k)`. Its real-spherical
multipole moments about its own centre are analytic:

```
Q_t^A[k] = integral chi_k(r) R_t(r - R_A) dr        for t in ranks 0..3, A = A(k)
```

where `R_t` is the **regular** real solid harmonic (the same one already used by
`project_transition_multipoles`). Note `Q_00[k] = q_k`, so the charge constraint above and the
rank-0 projection use the same quantity.

The site-resolved projection is then

```
B_DF[A,t,(ia)] = sum_{k : A(k) = A} Q_t^A[k] d_k^{ia}
```

and the site-pair response is unchanged:

```
alpha_AB(t,u; i omega) = 4 * B_DF[A,t,:] * G(i omega) * B_DF[B,u,:]^T
```

### A.3.3 Why this is a partition

`sum_A sum_{k in A} Q_00[k] d_k^{ia} = sum_k q_k d_k^{ia} = N_ia`, which is exactly the
constraint. So the constrained fit makes the *auxiliary-basis* partition reproduce the same
total that the ISA partition reproduces by partition-of-unity. This is the algebraic reason
the two schemes agree on the molecular total (`0.971` of Psi4's own molecular value, measured)
and disagree on the per-site split. The same conservation invariant from §3.6 of the TODO
document — sum the rank-1 blocks with rank 0 translated to a common origin — therefore applies
unchanged and is the first thing to assert.

## A.4 Psi4 infrastructure available

| Need | Existing Psi4 facility |
| ---- | ---------------------- |
| Coulomb metric `J_kl` | `FittingMetric` (`lib3index/dftensor.h`), constructed from an auxiliary `BasisSet` |
| Three-index integrals `(chi_k \|\| mu nu)` | `IntegralFactory::eri` with `BasisSet::zero_ao_basis_set()` on the fourth centre — the standard DF pattern; `DFHelper` (`lib3index/dfhelper.h`) if a managed path is wanted |
| Auxiliary basis construction | `MintsHelper::get_basisset(label)`; a `DF_BASIS_*`-style keyword |
| Multipole moments `Q_t[k]` | analytic for Gaussians; or `IntegralFactory::ao_multipoles(order)` against `zero_ao_basis_set()`, then the Cartesian-to-real-solid-harmonic transform in `libmints/solidharmonics.cc` |
| Transition densities | already available: the canonical occupied-major `(i,a)` ordering and `C` from `FrozenResponseContext` |
| Downstream | `contract_site_pair_response`, `localize_lw`, `refine_wsm`, `compute_dispersion` — all reused unmodified |

Note the auxiliary basis must be attached to the **frozen** context, not read from process
options at use time, and must be captured in a `BasisSetStructuralSnapshot` exactly as the
orbital basis already is. The provenance seal is not optional decoration here: the whole point
of the comparison is that only the partition differs, and an unattested auxiliary basis makes
that unprovable.

## A.5 Proposed interfaces

All new symbols in `psi4/src/psi4/libmints/atomic_polarizability.h`, in the `detail`
namespace where they are pure evaluators, mirroring the existing ISA path.

```cpp
/** Auxiliary-basis-space partition policy. */
struct PSI_API CDFOptions {
    /** Auxiliary basis label; must resolve through MintsHelper::get_basisset. */
    std::string auxiliary_basis;
    /** Relative eigenvalue cutoff for the Coulomb metric inverse. Never absolute. */
    double metric_relative_cutoff{1.0e-10};
    /** Largest metric condition number accepted before failing closed. */
    double maximum_condition_number{1.0e12};
    /** Constraint policy; see the open question in A.7. */
    CDFConstraintPolicy constraints{CDFConstraintPolicy::Charge};
};

/** Up-front storage/work accounting, gated before any dense allocation. */
struct PSI_API CDFPartitionPlan { /* naux, nbf, nocc, nvir, transition_count,
                                     metric_bytes, three_index_bytes, coefficient_bytes,
                                     moment_bytes, projection_bytes, estimated_bytes,
                                     configured_memory_bytes, reserved_memory_bytes,
                                     work_terms, max_work_terms, algorithm,
                                     memory_semantics */ };

CDFPartitionPlan plan_cdf_partition(std::size_t nbf, std::size_t naux, std::size_t nocc,
                                    std::size_t nvir, std::size_t memory_bytes);

/** Pure evaluator: analytic real-solid-harmonic moments of the auxiliary functions. */
Matrix auxiliary_multipole_moments(const BasisSet& auxiliary,
                                   const std::vector<SitePosition>& sites,
                                   const std::vector<std::size_t>& function_to_site);

/** Pure evaluator: constrained fit coefficients d[k, (ia)]. */
Matrix solve_constrained_density_fit(const Matrix& metric, const Matrix& rhs,
                                     const Matrix& constraints,
                                     const std::vector<double>& constraint_targets,
                                     const CDFOptions& options,
                                     CDFDiagnostics* diagnostics);

/** Site-major B[A,t,(ia)] with the same layout as project_transition_multipoles. */
TransitionMultipoleProjection project_transition_multipoles_cdf(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const CDFOptions& options);
```

The provider gains a partition selector rather than a second class:

```cpp
enum class PSI_API ResponsePartition { RealSpaceISA, ConstrainedDF };
```

`ISAPolResponseProvider::compute_isapol_response` dispatches on it and is otherwise unchanged.
A new keyword `ATOMIC_POLARIZABILITY_PARTITION` (`ISA` | `CDF`, default `ISA`) selects it, and
`ATOMIC_POLARIZABILITY_CDF_AUX_BASIS` names the auxiliary basis. **Default behaviour does not
change.**

## A.6 Validation plan

In priority order. Every check before the last needs no external reference.

1. **Partition conservation (no oracle needed).** Sum every site's rank-1 block with rank 0
   translated to a common origin. This is algebraically exact for any partition that
   reproduces the correct total, so it must recover the molecular dipole polarizability to the
   same `~1.0000` the ISA route achieves. If it does not, the constraint is wrong, and no
   downstream comparison is meaningful.
2. **Fit quality.** The Coulomb-metric residual `Delta` per transition, and the reconstructed
   `sum_k q_k d_k^{ia}` against `<i|a>`, both reported as diagnostics and gated.
3. **Reciprocity.** `alpha_AB(t,u) = alpha_BA(u,t)`, verified against a bound derived from the
   fit residual — exactly as the ISA route derives its bound from the solver's `FERR`.
4. **Basis-limit agreement between partitions.** As the auxiliary basis is saturated, the C-DF
   and ISA per-site splits do **not** converge to each other — they are different definitions.
   Assert instead that the *totals* agree, and record the split difference as a measured
   quantity. This is the test that would have prevented the two-oracle confusion.
5. **The `DF_*` literals at `rtol=1e-4, atol=1e-5`.** These already exist, already went through
   review, and are currently strict xfails. This is the acceptance oracle.
6. **The `ISA_GRID_*` literals must continue to pass** under `ATOMIC_POLARIZABILITY_PARTITION
   = ISA`. A C-DF implementation that perturbs the ISA path is a regression.

### The anti-conflation test (required)

Add one test that pins the *discriminating set*: the components where the two partitions
genuinely separate (`O yy/zz`, `H xz/yy/zz`), and assert that

- under `PARTITION=CDF` the output is closer to `DF_*` on all of them, and
- under `PARTITION=ISA` the output is closer to `ISA_GRID_*` on all of them.

The existing `test_parity_output_is_closer_to_the_isa_oracle_than_to_the_df_oracle` is exactly
half of this test; generalise it rather than duplicating it.

## A.7 Open questions to resolve before implementation

1. **What exactly does `DF with constraints` constrain?** The reviewed `H2O.cks` selects
   `C-DF`, `DF with constraints` and `DF-TYPE-MONOMER NN`. Charge conservation per transition
   density is the baseline reading and is what §A.3.1 specifies. Whether CamCASP additionally
   constrains dipole (or higher) moments, and what `NN` selects, must be determined from the
   CamCASP documentation before the constraint matrix is fixed. **Mitigation:** implement the
   general `C d = n` form so the answer is a data change.
2. **Auxiliary basis reproduction.** The reviewed run used a **246-function Cartesian `MC`-type**
   auxiliary basis. Reproducing the `DF_*` literals at `1e-4` almost certainly requires the same
   auxiliary basis, function for function. Determine whether it is expressible as a Psi4 basis
   file. If it is not, the `1e-4` gate is unreachable and the deliverable degrades to a
   *measured band* comparison — which is still worth having, but must be stated up front
   rather than discovered at the end.
3. **Cartesian vs spherical auxiliary functions.** The reviewed aux basis is Cartesian. Psi4's
   DF machinery is routinely spherical. Confirm the moment and metric paths handle Cartesian
   auxiliaries, or convert explicitly and attest the conversion.
4. **Cost.** `J` is `naux x naux` and `b` is `naux x n_ov`. At the reviewed protocol
   `naux = 246` and `n_ov` is a few thousand, so this is comfortably smaller than the existing
   dense `n_ov x n_ov` response solve. C-DF is **not** expected to be the bottleneck.

## A.8 Task breakdown

- [x] **A1.** Failing tests for `auxiliary_multipole_moments`: analytic moments of a single
      s/p/d Gaussian against closed form; rank-0 moment equals `q_k`; frame covariance under
      rotation.
- [x] **A2.** Implement `auxiliary_multipole_moments`. Commit `feat: add auxiliary multipole moments`.
- [x] **A3.** Failing tests for `solve_constrained_density_fit`: exact recovery when the target
      is in the auxiliary span; constraint satisfied to machine precision; relative (not
      absolute) metric cutoff; ill-conditioned metric fails closed; general multi-row `C`.
- [x] **A4.** Implement it, without forming `J^-1`. Commit `feat: add constrained density fitting core`.
- [x] **A5.** Failing tests for `project_transition_multipoles_cdf`: layout identical to the ISA
      producer; rank-0 row sums to `<i|a>`; resource plan gates before allocation. Note the
      rank-0 row sums to `<i|a>` only *within the penalty model's tolerance* — see the correction
      below.
- [x] **A6.** Implement it and attach the auxiliary basis snapshot to the frozen context.
      Commit `feat: partition the FDDS by constrained density fitting`.
- [x] **A7.** Wire `ResponsePartition` through the provider and the options. Assert the ISA
      path is bit-identical to before. Commit `feat: select the response partition scheme`.
      Verified by running the pinned wiring protocol on both sides of the change: all seven
      published arrays bit-identical under `np.array_equal`.
- [x] **A8.** Run the reviewed protocol under `PARTITION=CDF`. **All six comparisons FAIL at
      `rtol=1e-4, atol=1e-5`**; the markers stay and the gate was not widened. Measured
      per-component deviations and the stage-invariant diagnosis are in the plan's Task G record.
- [x] **A9.** Add the anti-conflation test (§A.6), generalised two-sided over both partitions.
      Plan Task G entry and partition spec resolved decision 1 updated.

### Corrections to Part A, established during implementation

1. **§A.3.1 is the wrong model** and §A.3.3 is wrong in consequence. The reference applies the
   charge condition as a *finite quadratic penalty* of weight `1.0`, not a hard constraint
   `C d = n`, so `sum_k q_k d_k^{ia}` is small but **nonzero** and §A.6 check 1 cannot be
   asserted at ISA precision. See `2026-08-18-cdf-open-questions-resolved.md`, which supersedes
   this section. Measured at the reviewed protocol: `max_ia |q^T d^{ia}| = 4.57e-05`.
2. **§A.5's numerical defaults reject the calculation they exist to reproduce.**
   `maximum_condition_number{1.0e12}` fails closed on a normal matrix measured at `7.7966e+12`,
   and `metric_relative_cutoff{1.0e-10}` would discard roughly thirty retained spectral
   directions. The shipped defaults are `1.0e14` and `1.0e-14`.
3. **§A.4's "`ao_multipoles` against `zero_ao_basis_set()`" route is unusable** — that path
   segfaults from Python in this build, and `MultipoleInt` produces Cartesian monomial moments
   from rank 1 upward with an electronic sign convention, so rank 0 is unavailable from it at
   all. The moments are analytic.
4. **§A.4's promise that only the partition differs needs one more seal than the spec lists.**
   `BasisSetStructuralSnapshot` must record `has_puream()` — it already does — because the
   Cartesian and spherical forms of the same shell list are 246 and 198 functions and are
   different auxiliary spaces. The auxiliary snapshot is checked inside
   `FrozenResponseContext::verify_basis_unchanged()` so every existing call site inherits it.
5. **The localisation form's sign is settled, empirically.** The published prose and equation
   disagree. The assembled normal matrix is `(1 - eta) J + eta K_self`, positive semidefinite
   only for `0 <= eta <= 1`; at `eta = -5.0e-4` it is indefinite and the solver fails closed. So
   `eta = +5.0e-4` in `J - eta K_inter` is the only usable reading. This closes §8 item 1 of the
   research document without needing the paywalled paper.
6. **One stage gate downstream of the partition had to be re-derived.** `localize_lw`'s
   charge-sum postcondition is exactly linear in the fit's charge residual, so under the
   auxiliary partition it measures the penalty rather than grid convergence. See the plan's
   Task G record for the measured amplification.
7. **§A.6 check 4's prediction is confirmed and is the most valuable output of Part A.** The two
   partitions agree on the molecular total (`0.11` percent) and disagree on the split (18
   percent), and the residual against the reviewed literals is partition-independent.

---

# Part B — Anisotropic dispersion coefficients

## B.1 Objective

Publish the full orientation-dependent set of atom-pair dispersion coefficients, replacing the
current isotropic-only output. CamCASP/CASIMIR labels these `Cn[l1 k1, l2 k2, j]`; the single
coefficient published today is `00 00 0`.

The design spec lists this as an explicit non-goal ("This work does not add anisotropic
spherical Cn components to the public Psi4 API"), and the isotropic recoupling spec names the
blocker precisely:

> A real Clebsch–Gordan contraction table is therefore **not** required for the published
> coefficients. Any rank-mixing table remains a prerequisite only if anisotropic components
> are added later.

This spec is that "later". The prerequisite is the whole job.

## B.2 What is already sufficient, and what is missing

**Already sufficient.** The refined L3 model already contains everything anisotropic
coefficients need: the full `15 x 15` real-spherical polarizability matrix per site per
frequency, including every off-diagonal `(l, l')` block. The isotropic output *discards* this
information by taking traces. Nothing upstream of `compute_dispersion` needs to change.

**Missing.** Two things:

1. the generalisation of the Casimir–Polder integral from traces to full block products —
   trivial;
2. the recoupling table that contracts those products into the S-function basis — the actual
   work.

## B.3 Equations

### B.3.1 The exact second-order dispersion energy

For two sites `A` and `B` separated by `R_AB`, in the distributed multipole representation, the
second-order dispersion energy is exactly

```
E_disp = - (1 / 2 pi) * integral_0^inf d(omega)
             sum_{t,t',u,u'}  T_{tu}(R_AB) T_{t'u'}(R_AB)
                              alpha^A_{t t'}(i omega)  alpha^B_{u u'}(i omega)
```

where `T_{tu}` is the multipole interaction function between component `t` on `A` and
component `u` on `B`, and `t, t', u, u'` run over the real-spherical components of the L3
model.

This expression is the foundation of the whole part, and it has two properties that make it
unusually tractable:

- it requires **no** Clebsch–Gordan table — only the interaction tensor `T`, which is a
  standard, independently testable object;
- it is **directly computable** for any given geometry and orientation. That makes it a
  numerical oracle for the recoupling table (§B.5), internal to the repository.

### B.3.2 The frequency integral generalises trivially

Define the full Casimir–Polder block product on the existing ten-point grid:

```
M^{AB}_{(t t')(u u')} = (1 / 2 pi) * sum_k w_k
                          alpha^A_{t t'}(i omega_k) * alpha^B_{u u'}(i omega_k)
```

with `(omega_k, w_k)` exactly as produced by `make_casimir_grid(10, 0.5)`, and with the static
`omega = 0` point excluded because it carries no quadrature weight. This is a direct
generalisation of the existing code, which computes the same thing after tracing each diagonal
rank block.

### B.3.3 The recoupling table

Expanding `T_{tu} T_{t'u'}` in Stone's S-function basis and integrating over the relative
orientation produces the coefficient set

```
C_n[ l1 k1, l2 k2, j ]  =  sum over (t,t',u,u') of
                              W^{n, l1 k1, l2 k2, j}_{(t t')(u u')}  *  M^{AB}_{(t t')(u u')}
```

with `n = 2 (l_a + l_b + 1)` fixed by the ranks involved. `W` is the **real recoupling table**:
the product of two real Clebsch–Gordan coefficients and the geometric factors from the
interaction tensor. Constructing, validating and versioning `W` *is* the deliverable.

### B.3.4 The isotropic limit is a hard consistency check — and it already passes

Take the diagonal-block, rank-`l` isotropic case `alpha^A_{t t'} = alphabar_l^A delta_{t t'}`.
Then the double sum collapses to

```
E_disp = - (1/2pi) integral d(omega) alphabar_{la}^A alphabar_{lb}^B
                    sum_{t,u} | T^{(la,lb)}_{tu} |^2
```

and the known norm identity for the interaction tensor,

```
sum_{t,u} | T^{(la,lb)}_{tu} |^2  =  binom(2 la + 2 lb, 2 la) / R^{2 (la + lb + 1)}
```

reproduces **exactly** the prefactors already implemented and already validated to `2.5e-7`
against CASIMIR:

| `(la, lb)` | `binom(2la+2lb, 2la)` | implemented `K = binom/(2 pi)` |
| ---------- | --------------------- | ------------------------------ |
| `(1,1)` | `binom(4,2) = 6` | `6/(2 pi)` — C6 |
| `(1,2)`, `(2,1)` | `binom(6,2) = 15` | `15/(2 pi)` — C8 |
| `(1,3)`, `(3,1)` | `binom(8,2) = 28` | `28/(2 pi)` — C10 |
| `(2,2)` | `binom(8,4) = 70` | `70/(2 pi)` — C10 |
| `(2,3)`, `(3,2)` | `binom(10,4) = 210` | `210/(2 pi)` — C12 |

Worked check for `(1,1)`: with `T_ab = (delta_ab - 3 n_a n_b)/R^3`,
`sum_ab (delta_ab - 3 n_a n_b)^2 = 3 - 6 + 9 = 6`, giving `C6 = (6/2pi) integral ... = (3/pi) integral ...`,
which is exactly the analytic C6 check already asserted in
`test_atomic_polarizability_math.py`.

**Therefore the general expression in §B.3.1 is not a new theory to be trusted on faith — it is
the generalisation of an expression already verified to `2.5e-7` against CASIMIR.** Any
candidate table `W` that fails to reduce to this table in the isotropic limit is wrong, and
that check is cheap, exact, and available on day one.

## B.4 Output contract

The published surface must be decided before implementation. Three options:

| Option | Shape | Notes |
| ------ | ----- | ----- |
| **(a)** Keep the four isotropic matrices; add one array variable per anisotropic label | many variables | matches CASIMIR's own output layout; verbose |
| **(b)** One `ATOMIC DISPERSION COEFFICIENTS` array, `(npair, nlabel)`, plus an `ATOMIC DISPERSION LABELS` companion | 2 variables | compact, self-describing, extensible |
| **(c)** Publish the frequency-resolved block products `M` and let callers recouple | 1 large array | maximally general, pushes the physics to the caller |

**DECIDED 2026-08-19: (b), truncated to `n <= 12`, with the existing four isotropic matrices
retained unchanged.**

The recommendation above was written on the assumption that the label set numbered "dozens".
The independent derivation
([`2026-08-18-anisotropic-recoupling-derivation.md`](2026-08-18-anisotropic-recoupling-derivation.md))
measured it at **29 762** labels spanning orders `n = 6..14`, so the sizing argument that
motivated (b) no longer holds unmodified. The decision taken is (b) restricted to
orders **6 through 12**, aligning the published surface with the existing
`ATOMIC C6`..`ATOMIC C12`.

Consequences that must be honoured, not worked around:

- **Orders 13 and 14 are computed and validated but not published.** They fall out of an L3
  model naturally and the derivation proves them correct. Truncation is a *publication filter*
  applied at the very end, never a restriction on the internal table.
- **Therefore the §B.5 check 2 direct-energy reconstruction MUST run against the full
  internal label set, not the published subset.** Reconstructing `E_disp` from an `n <= 12`
  truncation is *not* exact and cannot be, because the discarded orders carry real energy. A
  test that reconstructs from the published array and expects machine precision is wrong and
  will send the next reader hunting a non-defect. Assert machine precision on the full set;
  assert a *measured, recorded* truncation residual on the published set.
- The acceptance criterion "Direct energy reconstruction agrees to machine precision at every
  tested orientation" refers to the full set.
- `n = 2 (l_a + l_b + 1)` as stated in §B.3.3 is **wrong**; the correct relation is
  `n = l_a + l_a' + l_b + l_b' + 2`, which is why odd orders appear at all.

Whichever is chosen, the isotropic `00 00 0` entries in the new array **must equal**
`ATOMIC C6`…`ATOMIC C12` to machine precision. That is a required test.

## B.5 Validation plan

The decisive point: **the recoupling table can be validated to machine precision without any
external reference**, because §B.3.1 is directly computable.

1. **Isotropic reduction (day one).** `W` restricted to diagonal blocks and traced must
   reproduce `binom(2la+2lb, 2la)/(2 pi)` exactly for all five entries in §B.3.4. Cheap and
   completely diagnostic.
2. **Direct energy reconstruction (decisive).** Pick a site pair, a separation `R` and a set of
   relative orientations. Compute `E_disp` two ways:
   - directly from the double sum in §B.3.1 using the interaction tensor, no table involved;
   - from the anisotropic coefficients summed against their S-functions.

   These must agree to machine precision at **every** orientation. This single test validates
   the entire table, and it requires nothing but our own L3 models.
3. **Orientational average.** Averaging the anisotropic expansion over relative orientations
   must return the isotropic coefficient exactly.
4. **Pair symmetry and permutation.** `C_n[A][B]` under `(l1 k1) <-> (l2 k2)` exchange must map
   onto `C_n[B][A]`, exactly, as the isotropic engine already guarantees by summing both
   orderings.
5. **Rank completeness.** Missing rank 2 or 3 is an error, not a zero contribution — inherited
   from the isotropic engine.
6. **CASIMIR anisotropic output** as the final external oracle, at `rtol=1e-4, atol=1e-5`,
   extracted through `devtools/camcasp_reference.py` as fixed literals.

Note the ordering: checks 1–5 are internal and should all pass **before** any CASIMIR
comparison is attempted. That inverts the sequence that caused the two-oracle problem.

## B.6 The anti-conflation requirement

Anisotropic coefficients are far more sensitive to the partition than the isotropic ones —
they measure exactly the off-diagonal and out-of-plane structure that §4 of the TODO document
shows is where ISA and DF disagree by factors up to 113.

Therefore: **do not gate anisotropic Cn against an oracle whose partition differs from the run
being tested.** The oracle must be regenerated with `DIST-ALG ISA-GRID` (as Task F did) if the
run under test uses the ISA partition, or Part A must land first. State the oracle's partition
in the literal block, and add a test that fails if the two are mixed.

Expect the anisotropic coefficients to inherit the **rank deficit** documented in §4.6 of the
TODO — likely worse, since anisotropy lives disproportionately in the higher-rank blocks that
are currently 20–46 % low. **The rank deficit should be closed before anisotropic Cn is gated
against anything**, or the comparison will measure a known upstream defect and teach nothing.

## B.7 Task breakdown

- [x] **B1.** Decide the output contract (§B.4). Record the decision in this spec. **Done: (b) truncated to `n <= 12`; see §B.4.**
- [x] **B2.** Failing tests for the interaction tensor `T_{tu}` in the real-spherical L3 basis:
      analytic dipole–dipole form, rank covariance under rotation, the norm identity of §B.3.4
      for all five `(la, lb)` pairs.
- [x] **B3.** Implement `T_{tu}`. Commit `feat: add real-spherical multipole interaction tensor`.
- [x] **B4.** Failing tests for the full block-product Casimir–Polder integral `M`, including
      that tracing its diagonal rank blocks reproduces the existing isotropic path bit-for-bit.
- [x] **B5.** Implement `M`. Commit `feat: generalize Casimir-Polder integration to full blocks`.
- [x] **B6.** Failing tests for the recoupling table `W`: isotropic reduction (check 1),
      orientational average (check 3), permutation symmetry (check 4).
- [x] **B7.** Independently derive, implement and **version** `W`, with a loader/validator that
      refuses a table failing any structural invariant — following the precedent of
      `validate_dispersion_rank_pairs`, which already refuses any prefactor that is not
      `binom(2la+2lb, 2la)/(2 pi)`. Commit `feat: add real recoupling table for anisotropic dispersion`.
- [x] **B8.** Implement the decisive direct-energy reconstruction test (check 2) at several
      orientations. This is the acceptance gate for `W`.
- [x] **B9.** Publish per the chosen contract; assert the isotropic entries equal the existing
      four matrices to machine precision. Commit `feat: publish anisotropic dispersion coefficients`.
- [x] **B10.** Extend `devtools/camcasp_reference.py` to parse CASIMIR's anisotropic output;
      extract literals from a **partition-matched** run; compare at the plan gate.
      **Done 2026-08-20. Complete as a measurement, negative as a gate, and it located a new
      defect.** A partition-matched ISA-GRID CASIMIR run now exists and validates end to end
      (its isotropic `00 00 0` C6 reproduces the `ISA_GRID_C6` literals to all seven printed
      figures). 10457 shared coefficients compared, **0 inside `rtol=1e-4`**, two strict xfails
      retained with measured reasons; the gate was not widened. Three findings:
      (a) our published label set **strictly contains** CamCASP's, with 0 labels it emits that
      we omit, and all 5703 of its nonzero coefficients landing on labels we also make nonzero;
      (b) `casimir` hard-caps at `C12` and at coupled rank `j <= 8`, so our 222 nonzero `j >= 9`
      labels have **no external oracle at any rank** — and that sector is not marginal, one such
      label reaches `247.2` against CASIMIR's largest printed `|C11|` of `299.1`;
      (c) the `j` dependence of our recoupling prefactor differs from CASIMIR's by exactly
      `1/|<l1 0; l2 0 | j 0>|` on the `sigma`-even sector at C6, to `6.45e-07`. Whether that is
      the S-function normalisation convention §9.1 flagged as unverified, or a defect on one
      side, **cannot be decided internally** — see the B10 section of
      [the ISA-GRID oracle spec](2026-08-18-isa-grid-oracle.md). Until it is, do not "fix"
      either implementation, and do not gate anything anisotropic label-by-label.

---

## Acceptance criteria

### Part A — C-DF

- [ ] `solve_constrained_density_fit` exists, never forms `J^-1`, uses a **relative** cutoff, and
      fails closed on an ill-conditioned metric.
- [ ] `project_transition_multipoles_cdf` produces a `B` with layout identical to the ISA
      producer, and its rank-0 row reproduces `<i|a>` per transition.
- [ ] The molecular-sum conservation invariant holds for the C-DF partition to the same
      precision as for ISA.
- [ ] `ATOMIC_POLARIZABILITY_PARTITION` selects the scheme; the ISA path is **bit-identical** to
      its pre-change output.
- [ ] The six `DF_*` comparisons pass at `rtol=1e-4, atol=1e-5` with their `xfail(strict=True)`
      markers removed — **or** the measured deviation is recorded per component with a
      stage-invariant explanation, and the markers stay.
- [ ] The `ISA_GRID_*` comparisons still pass under `PARTITION=ISA`.
- [ ] The anti-conflation test passes in both directions.
- [ ] Full `-m mints` suite green.

### Part B — anisotropic Cn

- [ ] The interaction tensor satisfies the norm identity for all five `(la, lb)` pairs.
- [ ] The block-product integral reproduces the existing isotropic path bit-for-bit when traced.
- [ ] The recoupling table reduces to `binom(2la+2lb, 2la)/(2 pi)` in the isotropic limit,
      exactly.
- [ ] **Direct energy reconstruction agrees to machine precision at every tested orientation.**
- [ ] Orientational averaging returns the isotropic coefficients exactly.
- [ ] Published isotropic entries equal `ATOMIC C6`…`ATOMIC C12` to machine precision.
- [ ] The table is versioned and its loader refuses a structurally invalid table.
- [ ] CASIMIR comparison performed against a **partition-matched** oracle, with the partition
      stated in the literal block.
- [ ] Full `-m mints` suite green.

---

## Sequencing recommendation

```
   close the rank deficit  ──►  Part B (anisotropic Cn)
   (TODO P0)                     needs healthy rank-2/3 blocks to be gateable

   Part A (C-DF)  ──────────►  independent; can start immediately
                                converts 6 strict xfails into live 1e-4 tests
```

**Part A first** if the goal is the strongest defensible parity claim, because it targets the
only fully reviewed oracle on the branch and it is a contained, well-understood change with a
clear architectural seam.

**The rank deficit before Part B**, because anisotropic coefficients amplify exactly the
higher-rank blocks that are currently 20–46 % low. Gating them first would measure a known
upstream defect.

## Prerequisite: literature verification

Both parts must be implemented from published equations, and the following references should be
located and verified **before** coding rather than cited from memory:

- **C-DF:** the constrained density-fitting distributed-polarizability algorithm — believed to
  be Misquitta & Stone, *J. Chem. Phys.* **124**, 024111 (2006). Verify the citation, and in
  particular verify what the constraint set actually is (§A.7 question 1).
- **Anisotropic dispersion / S-functions:** Stone, *The Theory of Intermolecular Forces*,
  2nd ed., §4.3 (dispersion) and the S-function definitions in §3.3. The isotropic spec already
  cites §4.3 for the Casimir–Polder integration and that citation held up.

Record the verified citations in this document before Task A1 / B1 begins.
