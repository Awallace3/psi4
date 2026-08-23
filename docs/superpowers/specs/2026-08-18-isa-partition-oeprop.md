# ISA Partition as a Standalone `oeprop` Method

Status: **accepted 2026-08-18**; all four scoping decisions resolved, see
[Resolved decisions](#resolved-decisions). Task F is **done** and refines the rest — see
[the ISA-GRID oracle](2026-08-18-isa-grid-oracle.md).
Companion to `plans/2026-07-31-native-camcasp-parity.md` and
[the parity debugging map](2026-08-17-parity-debugging-map.md).

## Objective as stated

> Implement the ISA partition as another `oeprop` method in Psi4, to then pass into the
> atomic-polarizability/C6 code, so we can isolate the localization discrepancy and reach
> nearly 100% agreement on both polarizabilities and C6 coefficients.

Two measurements taken while scoping this spec change what "isolate the localization
discrepancy" should mean. Both are recorded below because they redirect the work.

## Finding 1: the LW localization is already exact — it is not the discrepancy

The reviewed CamCASP tree contains **both ends of the localization step as files**:

| file | role |
| ---- | ---- |
| `work/H2O/H2O_NL4_000.pol` | the nonlocal site-pair response, ranks 0–4, 25x25, all 9 ordered pairs — the *input* to localization |
| `work/H2O/H2O.temp000` | the ORIENT script: `Limit all rank 3`, then `Localise LW test 1e-7 Limit 3` |
| `work/H2O/H2O_L3_000.pol` | the localized L3 model, 15x15 per site — the *output* of localization |

That makes Task 3 testable **hermetically**: feed the reviewed nonlocal model into our own
`localize_lw` and compare against the reviewed localized model. No SCF, no basis, no grid,
no partition enters the comparison, so the result is an unambiguous verdict on Task 3 alone.
The existing `_atomic_polarizability_localize_lw` binding already accepts arbitrary
16x16 rank-0-through-3 blocks plus positions and bonds, so no new code was needed.

Measured 2026-08-18, truncating rank 4 exactly as ORIENT's `Limit all rank 3` does:

| site | max abs difference over all 15x15 entries |
| ---- | ----------------------------------------- |
| `O`  | `7.4e-13` |
| `H1` | `5.5e-13` (after the local-frame sign below) |
| `H2` | `5.1e-13` |

`H1` initially showed `1.5e+01`. That is not an error: the ORIENT log states H1's local axes
are the parent axes **rotated 180 degrees about z** (`p_x = -1, p_y = -1, p_z = +1`), applied
after localization by `Edit H2O / #include H2O.axes`, while `O` and `H2` local axes coincide
with the parent. Under that rotation a real spherical harmonic `lm{c,s}` picks up `(-1)^m`, so
`alpha_(t,u) -> (-1)^(m_t + m_u) alpha_(t,u)`. Applying exactly that sign pattern takes H1
from `1.5472e+01` to `5.4623e-13` — confirming both the localization and the frame convention.

**Conclusion: Task 3 reproduces ORIENT's Lillestolen–Wheatley localization to `~1e-12` on
every one of the 675 tensor entries.** It cannot be the source of the site misdistribution,
and the localization does not need to be isolated — it is already settled.

This also *retires* the interpretation in the debugging map that the gap "lives in the ISA
partition (Task 4) or the LW localization (Task 3)". It lives in Task 4, exclusively.

## Finding 2: the reviewed oracle does not use ISA at all

`work/H2O/OUT/H2O.out` line 411 states the distributed-polarizability algorithm outright:

```
 Distributed polarizability calculation
 ALGORITHM: DF : density-fitting-based partitioning of the FDDS
```

and `work/H2O/H2O.cks` selects `C-DF` ("For unconstrained DF use: DF"), `DF with
constraints`, `DF-TYPE-MONOMER NN`, over a 246-function Cartesian `MC`-type auxiliary basis.
The reviewed control file `work/H2O/H2O.clt` contains **no ISA directive**; it is a plain
`Run-type properties` job. (`Initialized stockholder atoms = T` on line 93 refers to the
ground-state/DMA machinery, not to the response partition.)

CamCASP selects the partition with a single `DIST-ALG` directive inside its
`Polarizability` block. `examples/properties/H2O-wxhole/output/H2O_DZ.out` line 1372 shows the
alternative we want:

```
BEGIN Polarizability
  DIST-ALG ISA-GRID   ( this sets the ISA-GRID-based partitioning )
```

`ISA-GRID` is CamCASP's **real-space, grid-based** iterated-stockholder partition — the same
family as our implementation — and it writes `*_ISA-GRID_f11_NL4_fmtA.pol` in place of
`*_DF_...`/`*_NL4_...`. So a matching oracle is one directive away from the reviewed control
file, not a new method. (The wxhole example additionally enables a WXhole model, which we do
not want; only `DIST-ALG` should change relative to the reviewed protocol.)

For contrast, CamCASP's basis-space ISA path — present in the runtime as
`methods/isa-pol-from-isa-A` and exercised by
`examples/properties/H2O-wxhole` — requires an `ISA-Basis`, an `AtomAux-Basis`, a converged
ISA-A step, and emits an `H2O_atoms.ISA` shape-function file. None of those exist in the
reviewed tree.

So the reviewed oracle partitions the FDDS by **constrained density fitting onto atom-centred
auxiliary functions**, an auxiliary-basis-space partition, while our Task 4 partitions it by
**real-space iterated-stockholder weights**. These are two different, both legitimate,
distributions of the same molecular response. They agree on the total — which is exactly what
we observe, conservation `0.955` — and disagree on how it is split between sites.

The observed signature matches this mechanism rather than a bug: hydrogen's diffuse auxiliary
functions own out-of-plane response that a real-space stockholder weight assigns to oxygen,
which is why `alpha_xx` agrees on both sites while `yy` and `zz` are misallocated.

**Consequence: improving or exposing the ISA partition cannot by itself reach agreement with
this oracle, because this oracle is not an ISA calculation.** Any plan that measures ISA
against `H2O_ref_wt4_L3_*.pol` is comparing two different physical models.

## What this means for the objective

The objective decomposes into two separable goals that were previously conflated:

1. **Inspectability** — expose the ISA partition as a first-class, standalone Psi4 property so
   it can be computed, published, printed and validated without running the whole response
   pipeline, and so it can be handed to the polarizability/C6 code as an explicit input.
   This is worth doing regardless of Finding 2 and is what the request literally asks for.
2. **Parity** — reach ~100% agreement on polarizabilities and C6. This now requires the
   *partition schemes to match*, which Finding 2 says they currently do not. See the options
   in [Open decisions](#open-decisions).

Goal 1 is a prerequisite for diagnosing goal 2 either way: with ISA published as its own
property we can compare partition observables (populations, charges, shape functions,
ISA-DMA multipoles) against a matching oracle instead of inferring partition error from a
polarizability four stages downstream.

## Existing state

The ISA partition is already implemented and converging; it is simply not addressable.

- `psi4/src/psi4/libmints/isa_weights.cc` (1119 lines) — real-space iterated stockholder:
  Gauss–Legendre mapped radial quadrature, product spherical angular grid, PCHIP log-profile
  interpolation, exponential tail fitting, log-sum-exp promolecule, Bragg–Slater radii.
- `compute_isa_weights(context, ISAOptions)` returns `ISAWeights`, whose `partition_weights_`
  are **private** and reachable only by `friend class ISAPolResponseProvider`.
- `ISADiagnostics` already carries `atomic_populations`, `radial_nodes`, `log_profiles`,
  `tail_join_radii`, `tail_alphas`, convergence residuals and a `context_digest`.
- Options exist: `ATOMIC_POLARIZABILITY_ISA_{RADIAL_POINTS, ANGULAR_POLAR_POINTS,
  ANGULAR_AZIMUTHAL_POINTS, MAX_ITERATIONS, CONVERGENCE}`.
- Test-only bindings exist (`_atomic_polarizability_compute_isa_weights`,
  `_atomic_polarizability_test_isa*`), but nothing publishes ISA results as Psi4 variables and
  there is no `oeprop` task.

The blocking structural fact: `ISAWeights` is **bound to a sealed `FrozenResponseContext`**
(its ordered response grid, sites and frozen AO density). A standalone `oeprop` method must
therefore either construct such a context from a plain wavefunction, or the partition must be
refactored to accept a grid + density directly. That is the main design decision below.

## Proposed design (Goal 1)

### Task A — decouple the partition from the response context

Introduce a narrow input struct so the stockholder solver does not require a sealed response
context:

```cpp
struct PSI_API ISAPartitionInput {
    std::vector<SitePosition> sites;
    std::vector<int> atomic_numbers;
    std::vector<SitePosition> points;     // ordered output grid
    std::vector<double> weights;          // integration weights, same order
    std::vector<double> density;          // molecular density at points
    double formal_electron_count{};
};
```

`compute_isa_partition(const ISAPartitionInput&, const ISAOptions&) -> ISAPartition`
becomes the primitive. `compute_isa_weights(context, options)` is reimplemented as a thin
adapter over it, preserving the existing seal and the `friend` access path so Task 4 output is
bit-identical. The `solve()` core in `isa_weights.cc` already takes exactly these arguments,
so this is a signature extraction rather than new numerics.

**Invariant to assert:** for the wiring protocol, the partition weights obtained through the
new primitive equal those obtained through the existing sealed path to `0.0` exactly.

### Task B — the `oeprop` task

Register `ISA_CHARGES` (name pending the decision below) in `OEProp::compute()`, dispatching
to `OEProp::compute_isa_partition()`, following `compute_mbis_multipoles` as the structural
model: build a `MolecularGrid` from `ISA_*_POINTS` options, evaluate the density on it, call
the primitive, publish, print.

Published `wfn_` variables (final list pending the decision below):

| variable | shape | content |
| -------- | ----- | ------- |
| `ISA CHARGES` | natom x 1 | `Z_a - N_a`, the ISA atomic charges |
| `ISA POPULATIONS` | natom x 1 | integrated `N_a` |
| `ISA SHAPE FUNCTION NODES` | ragged | per-atom radial nodes |
| `ISA SHAPE FUNCTION LOG VALUES` | ragged | per-atom `log w_a(r)` |
| `ISA TAIL PARAMETERS` | natom x 2 | join radius, exponent |
| `ISA ITERATIONS` (scalar) | — | fixed-point iteration count |
| `ISA MAX OVERLAP RESIDUAL` (scalar) | — | convergence residual |

Ragged arrays do not fit `set_array_variable`; the shape functions are therefore proposed as
`natom` separate variables (`ISA SHAPE FUNCTION 0`, ...) or as a single padded matrix. This is
a decision point.

**Fail-closed behaviour, matching the rest of this pipeline:** if the fixed point does not
converge, if pointwise partition unity fails, or if the integrated population does not
conserve, `compute_isa_partition` throws before publishing anything. No partially populated
variable is ever visible.

### Task C — accept an externally supplied partition

Add a documented, non-test entry point so a partition computed by Task B (or supplied by the
user) can be handed to the response provider, replacing the current
`ISAWeights::create_test_only`. This is what "pass into this atomic-polarizability/C6 code"
requires, and it is also what makes a **partition A/B test** possible: run the entire
downstream pipeline twice, changing only the partition, with every other stage fixed.

### Task D — Python driver surface

Extend `psi4/driver/procrouting/atomic_polarizability.py` with an `isa_partition(molecule,
...)` entry, and allow `atomic_polarizabilities(..., partition=...)` to consume one.

## Validation plan

Because the reviewed oracle is not an ISA calculation, ISA needs oracles of its own. In
priority order:

1. **Analytic.** A promolecule of two spherical Gaussians has a closed-form stockholder
   partition; ISA must reproduce it. The existing `test_isa_gaussian_fixed_point` already
   covers the fixed point and can be lifted to the public path.
2. **Partition-of-unity and conservation.** `sum_a w_a(r) = 1` pointwise; `sum_a N_a = N`;
   `sum_a q_a = Q`. These need no external reference and are already enforced internally.
3. **Grid convergence.** ISA charges stable to `1e-5` under refinement of both the DFT grid
   and the ISA grid. Note the recorded prerequisite: the **DFT grid, not the ISA grid, was
   binding** in earlier tests — `302/50` sat at `1.2e-5` regardless of ISA density, and only
   `590/99` came inside `1e-6`.
4. **Literature.** Published ISA charges for H2O (and the `CO2_ISA.mom` /
   `formamide_ISA.mom` reference multipoles in the runtime examples) as fixed checked-in
   literals.
5. **A matching CamCASP oracle**, if the decision below calls for one.

Constraints inherited unchanged from the plan's Global Constraints: production code and pytest
must not invoke, clone, access or read CamCASP, ORIENT, PFIT, CASIMIR or `.camcasp-reference/`;
no ORIENT GPLv3 source, comments, structure or control flow may be copied; mismatches are
investigated by stage invariant and tolerances are not loosened.

## Acceptance criteria

- [ ] `compute_isa_partition` primitive exists; sealed-path weights unchanged (exact `0.0`).
- [ ] `oeprop` task computes and publishes the ISA partition from a plain wavefunction, and
      fails closed on non-convergence.
- [ ] The response provider accepts an externally supplied partition through a non-test API.
- [ ] Partition-of-unity, population and charge conservation asserted in pytest.
- [ ] Grid-convergence test at the documented `590/99` prerequisite.
- [ ] A partition A/B harness measures the per-site `alpha` split as a function of partition
      alone, with all other stages fixed.
- [ ] Full `-m mints` suite still green (currently 381 passed, 6 skipped).

## Resolved decisions

Decided by the user 2026-08-18.

1. **Parity target: both, ISA-GRID first.** Regenerate an ISA-GRID oracle now to close parity
   against the partition we already implement, and keep C-DF as a follow-on so both CamCASP
   partition schemes are reproducible. The existing DF reference is retained as a second data
   point rather than discarded.

   **Discharged 2026-08-19.** Both halves are done: the ISA-GRID oracle is the acceptance
   oracle inside a measured band, and C-DF is implemented as the second selectable partition
   with the DF reference retained and still gated at the plan tolerance as a strict xfail.
   Neither literal set was rewritten and neither gate was loosened. Two findings from having
   both arms native, neither of which was reachable with one:

   - the C-DF arm is what proves the two references are two *partitions* of one response
     rather than two answers: the two oracles agree on the site-summed isotropic dipole
     polarizability to `0.11` percent while their O/H split differs by 18 percent, and
     switching one keyword swaps which oracle our output lands on, by factors of 8.8-49 (C-DF
     arm) and 7.9-82 (real-space arm) on the discriminating components, with the whole rest of
     the pipeline held fixed;
   - the residual that remains is *not* the partition. It splits into a uniform `0.971`
     molecular-total deficit that is identical on both arms against both oracles, and a
     rank-2/rank-3 site-block deficit that the partition barely moves. Both are now recorded
     as measured quantities rather than suspected causes.
2. **Published surface: all four groups** — charges and populations, convergence diagnostics,
   radial shape functions, and ISA-DMA multipoles.
3. **Plumbing: computed by default, injectable by argument.** The polarizability entry point
   keeps recomputing the partition internally; an explicit `partition=` overrides it. This is
   what makes a partition A/B test possible.
4. **Naming: `ISA_CHARGES` task with an `ISA_*` option prefix.** The existing
   `ATOMIC_POLARIZABILITY_ISA_*` keywords are kept as the polarizability pipeline's own
   overrides so no current behaviour changes.

## Work breakdown

### Task A — decouple the partition from the response context

As designed above: extract `compute_isa_partition(const ISAPartitionInput&, const ISAOptions&)`
from the existing `solve()` core, and reimplement `compute_isa_weights(context, options)` as a
thin adapter over it, preserving the seal and the `friend` access path.

**Gate:** partition weights through the new primitive equal the sealed path exactly (`0.0`).

### Task B — the `ISA_CHARGES` oeprop task

Register `ISA_CHARGES` in `OEProp::compute()` dispatching to `OEProp::compute_isa_partition()`,
structurally following `compute_mbis_multipoles`: build a `MolecularGrid` from the `ISA_*`
options, evaluate the density, call the primitive, publish, print. New options
`ISA_RADIAL_POINTS`, `ISA_ANGULAR_POLAR_POINTS`, `ISA_ANGULAR_AZIMUTHAL_POINTS`,
`ISA_MAX_ITERATIONS`, `ISA_CONVERGENCE`, `ISA_SPHERICAL_POINTS`, `ISA_GRID_RADIAL_POINTS`,
defaulting to the same values as the `ATOMIC_POLARIZABILITY_ISA_*` set.

Published variables:

| variable | shape | content |
| -------- | ----- | ------- |
| `ISA CHARGES` | natom x 1 | `Z_a - N_a` |
| `ISA POPULATIONS` | natom x 1 | integrated `N_a` |
| `ISA ITERATIONS` | scalar | fixed-point iteration count |
| `ISA MAX OVERLAP RESIDUAL` | scalar | shape-function convergence residual |
| `ISA MAX UNITY RESIDUAL` | scalar | pointwise `sum_a w_a(r) - 1` |
| `ISA POPULATION ERROR` | scalar | `sum_a N_a - N` |
| `ISA SHAPE FUNCTION <a>` | n_r x 2 | per-atom radial node, `log w_a(r)` |
| `ISA TAIL PARAMETERS` | natom x 2 | join radius, exponent |

Radial shape functions are ragged, so they are published as one `n_r x 2` matrix per atom
rather than padded into a single array; the per-atom row counts are recoverable from the
matrices themselves.

**Fail-closed:** non-convergence, pointwise unity failure or population non-conservation throws
before any variable is published, matching the rest of this pipeline.

### Task C — externally supplied partitions

Replace `ISAWeights::create_test_only` with a documented non-test constructor that validates
the supplied partition against the sealed context (site count, grid cardinality, pointwise
unity, non-negativity) before accepting it.

### Task D — Python driver surface

Add `isa_partition(molecule, ...)` to `psi4/driver/procrouting/atomic_polarizability.py`, and
accept `atomic_polarizabilities(..., partition=None)`; `None` keeps today's internal path.

### Task E — ISA-DMA multipoles

Distributed multipoles to rank 4 using the ISA weights, published as `ISA MULTIPOLES`
(natom x 25, spherical, CamCASP component order). Comparable to CamCASP's `*_ISA.mom` files —
`CO2_ISA.mom` and `formamide_ISA.mom` exist in the runtime examples and give two independent
molecules to check against.

### Task F — regenerate the ISA-GRID oracle — **DONE 2026-08-18**

See [the ISA-GRID oracle](2026-08-18-isa-grid-oracle.md) for the method, the discovered
prerequisites and the full result tables. Outcome: the partition scheme accounts for the entire
site misdistribution, and our implementation agrees with the matching oracle to 2–5% on six of
seven per-site dipole components and to 1–10% per-pair on C6.

Acceptance literals it produced, for use by Tasks A–E:

| quantity | ISA-GRID oracle value |
| -------- | --------------------- |
| `ISA POPULATIONS` `O` | `8.81587` |
| `ISA POPULATIONS` `H` | `0.588661` |
| `O` static `(xx, yy, zz)` | `7.0420, 7.4738, 7.1290` |
| `H` static `(xx, yy, zz)` | `1.5870, 0.7609, 1.2408` |
| `H` static `zx` | `-0.64527` |
| `C6` `(O-O, O-H, H-H)` | `26.48177, 4.14232, 0.65147` |

### Task G — C-DF partitioning — **built 2026-08-19**

Superseded. Task F showed ISA-GRID closes most of the dipole-block gap, so C-DF was not required
for parity — but it *was* wanted as a deliverable in its own right, because it is the only way to
turn the partition from an implicit property of the implementation into an explicit input and so
make the partition A/B experiment native on both arms. It is now implemented and switchable
(`ATOMIC_POLARIZABILITY_PARTITION = ISA | CDF`, default `ISA`, ISA path bit-identical), and the
reviewed DF reference is reproduced to `0.0368` worst-case on the static dipole block and
`0.0399` worst-pair on C6 — not to the plan's `rtol=1e-4`, which is missed by four orders of
magnitude, so the six `DF_*` `xfail(strict=True)` markers stay. The experiment's real payoff is
that it localises the remaining residual away from the partition entirely: see the Task G record
in [the plan](../plans/2026-07-31-native-camcasp-parity.md).

## Validation plan

Because the current reviewed oracle is not an ISA calculation, ISA needs oracles of its own.
In priority order:

1. **Analytic.** A promolecule of two spherical Gaussians has a closed-form stockholder
   partition. `test_isa_gaussian_fixed_point` already covers this and can be lifted to the
   public path.
2. **Partition-of-unity and conservation.** `sum_a w_a(r) = 1` pointwise; `sum_a N_a = N`;
   `sum_a q_a = Q`. No external reference needed; already enforced internally.
3. **Grid convergence.** ISA charges stable to `1e-5` under refinement of both grids. Recorded
   prerequisite: the **DFT grid, not the ISA grid, is binding** — `302/50` sat at `1.2e-5`
   regardless of ISA density, and only `590/99` came inside `1e-6`.
4. **Literature and `*_ISA.mom` references** as fixed checked-in literals.
5. **The regenerated ISA-GRID oracle** from Task F. Note it cannot be gated at
   `rtol=1e-4, atol=1e-5`: our real-space ISA and CamCASP's ISA-GRID are *different ISA
   variants* (CamCASP takes its shape functions from the basis-space ISA-A functional), so the
   measured agreement is 2–5% on most components. Use it as a **regression and
   direction-of-effect oracle** with an explicitly measured band, and reserve the `1e-4` gate
   for quantities that must agree exactly — the frequency grid, the recoupling prefactors, the
   LW localization, and partition-of-unity.

Two prerequisites Task F uncovered on our side must be fixed for any of this to run at the
reviewed basis:

- the response provenance seal rejects the marginally converged aug-cc-pVTZ cation SCF, so
  protocols need explicit `e_convergence`/`d_convergence` (and the seal's re-derivation of
  convergence should be reconciled with Psi4's own verdict);
- `PARITY_PROTOCOL`'s `1e-8` localization tolerance is below the measured `5.39e-07` LW
  charge-sum residual at `590/99`, so it must be set from measurement.

Constraints inherited unchanged from the plan's Global Constraints: production code and pytest
must not invoke, clone, access or read CamCASP, ORIENT, PFIT, CASIMIR or `.camcasp-reference/`;
no ORIENT GPLv3 source, comments, structure or control flow may be copied; mismatches are
investigated by stage invariant and tolerances are not loosened.

## Acceptance criteria

- [ ] `compute_isa_partition` primitive exists; sealed-path weights unchanged (exact `0.0`).
- [ ] `ISA_CHARGES` oeprop task publishes all eight variables from a plain wavefunction, and
      fails closed on non-convergence.
- [ ] The response provider accepts an externally supplied partition through a non-test API,
      validated against the sealed context.
- [ ] `atomic_polarizabilities(..., partition=...)` round-trips a partition from
      `isa_partition(...)` and reproduces the internal path bit-for-bit when given its own
      output.
- [ ] Partition-of-unity, population and charge conservation asserted in pytest.
- [ ] Grid-convergence test at the documented `590/99` prerequisite.
- [ ] ISA-DMA multipoles checked against fixed literals for at least two molecules.
- [x] Task F oracle regenerated and the residual recorded (2026-08-18).
- [x] Per-site `alpha` and per-pair `C6` regression-tested against the Task F literals within
      the measured band, not the `1e-4` gate (2026-08-18; C8/C10/C12 added as well, each with
      its own band because the deviation grows with rank).
- [ ] A partition A/B harness measures the per-site `alpha` split as a function of partition
      alone, with all other stages fixed.
- [ ] Full `-m mints` suite still green (currently 381 passed, 6 skipped).
