# Task F: The Regenerated ISA-GRID Oracle

Executed 2026-08-18. Resolves the parity-target question in
[the ISA partition spec](2026-08-18-isa-partition-oeprop.md) and supersedes the
partition-related conclusions in [the debugging map](2026-08-17-parity-debugging-map.md).

**Result: our ISA implementation was correct and was being measured against the wrong oracle.**
Switching the reference's partition from constrained density fitting to grid ISA, with the
wavefunction, auxiliary basis and fit points all held byte-identical, reproduces our
discrepancy in both magnitude and direction — including the hydrogen dipole off-diagonal that
looked like a bug.

## Method

The run reuses the reviewed calculation's own converged orbitals, so the comparison is a clean
single-variable experiment. Development-only: nothing here is a production or pytest dependency.

1. **Control.** The `camcasp` binary reads its command file on stdin. Copying the reviewed
   `H2O.cks`, `H2O-A-asc.movecs` and `H2O-A.basis` into a fresh directory and running
   `camcasp < H2O.cks` reproduces the reviewed `*_NL4_fmtA.pol`, `*_NL4_fmtB.pol` and
   `*.p2p` **bit-for-bit identically**. The harness is therefore faithful and deterministic,
   and no SCF re-run is needed.
2. **Learning the syntax.** `runcamcasp.py --setup` on a control file carrying
   `AtomAux-Basis aVTZ Type MC Spherical Use-ISA-Basis` and `ISA-Basis set2 Min-S-exp-H = 0.2`
   emits a `Basis Atom-Aux` block. The generated Psi4 SCF input `H2O_A.in` is **byte-identical**
   to the reviewed one, confirming the orbitals do not depend on the auxiliary bases.
3. **The ISA-GRID run.** The reviewed `H2O.cks` was modified in exactly three ways:
   - the cluster-generated `Basis Atom-Aux` block (spherical, with ISA `set2` functions) was
     grafted in. `Basis Aux` was **left exactly as reviewed** — Cartesian, no ISA functions —
     so the density-fitted FDDS and the propagator are unchanged;
   - an `Edit / NEIGHBOURS`, an overlap-metric `BEGIN DF Type OO`, a `Set Num-Int-Pars` and a
     `Begin ISA` block were inserted before the NL4 polarizability block, following the
     runtime's own `methods/wxhole-pols` template, which is the only in-tree example of
     ISA-GRID;
   - `DIST-ALG ISA-GRID` was added to the NL4 `BEGIN Polarizability` block.
4. **Downstream.** `localize.py H2O --loc LW --limit 3 --wsmlimit 3 --hlimit 3 --weight 4
   --weightcoeff 0.001 --cutoff 0.0001 --force loc refine` with the reviewed protocol
   constants, giving regenerated `H2O_L3_*.pol` (LW) and `H2O_ref_wt4_L3_0f10.pol` (PFIT WSM).

Two facts make this a genuine single-variable experiment:

- the generated `H2O_A.in` is byte-identical, so the wavefunction is the reviewed one;
- the emitted `*.p2p` is byte-identical to the reviewed perturbation matrix, so the WSM
  refinement solves the **identical fit problem** at the identical fit points.

`ALGORITHM: ISA-GRID : ISA partitioning using a numerical grid` confirms the switch took effect.

### Prerequisites discovered

- **`DIST-ALG ISA-GRID` is not a one-line change.** It needs a preceding `Begin ISA` block; the
  spec's earlier "one directive away" claim was too optimistic and is corrected here.
- **ISA algorithm A requires a spherical Atom-Aux basis.** With the reviewed Cartesian aux basis
  and no Atom-Aux block, CamCASP fails with `Basis-based ISA schemes B1, B2 & A need AtomAux
  basis sets with spherical GTOs`. Adding a spherical Atom-Aux block clears it without
  perturbing the response basis.
- CamCASP's ISA-GRID is a hybrid: shape functions from the basis-space ISA-A functional,
  applied pointwise on the integration grid. Ours is real-space throughout.
- CamCASP's dispersion step (`casimir`) failed on a hardcoded relative runtime path from the
  reviewed run's directory depth. Not pursued: our own recoupling is verified to `2.5e-7`.

## ISA convergence and charges

Converged in 40 iterations, all three sites flagged `Y`.

| site | population | charge |
| ---- | ---------- | ------ |
| `O`  | `8.81587`  | `-0.81587` |
| `H1` | `0.588661` | `+0.411339` |
| `H2` | `0.588661` | `+0.411339` |

Residual charge `0.0068`; `KL(BASIS) = 0.0706`, `KL(GRID) = 0.0449`. These are a direct oracle
for the `ISA POPULATIONS` / `ISA CHARGES` that Task B will publish.

## The partition is the whole discrepancy

Static rank-1 blocks in the site-local frame. Reviewed = C-DF partition; ISA-GRID = regenerated.
Everything except the partition is identical between the two columns.

**After LW localization** (`H2O_L3_000.pol`):

| site | component | DF | ISA-GRID | delta |
| ---- | --------- | -- | -------- | ----- |
| `O`  | `alpha_zz` | `5.5832` | `7.1320` | `+1.5488` |
| `O`  | `alpha_xx` | `7.0353` | `7.0462` | `+0.0109` |
| `O`  | `alpha_yy` | `5.7637` | `7.4749` | `+1.7112` |
| `H1` | `alpha_zz` | `2.0087` | `1.2469` | `-0.7618` |
| `H1` | `alpha_xx` | `1.5574` | `1.5955` | `+0.0381` |
| `H1` | `alpha_yy` | `1.6208` | `0.7632` | `-0.8576` |
| `H1` | `alpha_zx` | `-0.0058` | `-0.6453` | `-0.6395` |

Molecular sum from the localized sites: DF `9.5853`, ISA-GRID `9.6214` isotropic. Both conserve;
the partition **redistributes without changing the total**, exactly as it must.

**After PFIT WSM refinement** (`H2O_ref_wt4_L3_0f10.pol`, static):

| site | DF `(zz, xx, yy)` | ISA-GRID `(zz, xx, yy)` | DF `zx` | ISA-GRID `zx` |
| ---- | ----------------- | ----------------------- | ------- | ------------- |
| `O`  | `5.5837 7.0435 5.7621` | `7.1290 7.0420 7.4738` | `+0.00000` | `+0.00000` |
| `H1` | `2.0096 1.5737 1.6174` | `1.2408 1.5870 0.7609` | `-0.00576` | `-0.64527` |

Site sums: DF isotropic `9.5969`, ISA-GRID `9.6074`.

## Basis-matched comparison against our pipeline

Our pipeline run at `PARITY_PROTOCOL` (PBE0/aug-cc-pVTZ, GRAC, DFT `590/99`, ISA `100/24/32`),
so basis, geometry and functional match both CamCASP columns. Static per-site dipole
polarizabilities in the site-local frame:

| site | comp | ours | CamCASP DF | ratio | CamCASP ISA-GRID | ratio |
| ---- | ---- | ---- | ---------- | ----- | ---------------- | ----- |
| `O` | `xx` | `6.9203` | `7.0435` | `0.9825` | `7.0420` | `0.9827` |
| `O` | `yy` | `7.4144` | `5.7621` | `1.2868` | `7.4738` | `0.9921` |
| `O` | `zz` | `6.9551` | `5.5837` | `1.2456` | `7.1290` | `0.9756` |
| `H` | `xx` | `1.5388` | `1.5737` | `0.9778` | `1.5870` | `0.9696` |
| `H` | `yy` | `0.6446` | `1.6174` | `0.3985` | `0.7609` | `0.8471` |
| `H` | `zz` | `1.1591` | `2.0096` | `0.5768` | `1.2408` | `0.9341` |
| `H` | `zx` | `-0.6531` | `-0.0058` | `113.35` | `-0.6453` | `1.0122` |

Isotropic: `O` ours `7.0966` (DF `1.158`, ISA-GRID `0.984`); `H` ours `1.1142`
(DF `0.643`, ISA-GRID `0.931`). Molecular sum ours `9.3249`, `0.971` of both references —
the totals were never the problem.

**Against the matching oracle six of seven components agree to 2–5% and the worst,
`H alpha_yy`, to 15%; against DF the same numbers are wrong by up to a factor of 113.**

## C6 dispersion coefficients

Both CamCASP refined models were recoupled with **our own** dispersion engine (verified to
`2.5e-7`), so this comparison isolates the partition and not the recoupling.

| pair | ours | CamCASP DF | ratio | CamCASP ISA-GRID | ratio |
| ---- | ---- | ---------- | ----- | ---------------- | ----- |
| `O-O` | `26.17151` | `17.25559` | `1.5167` | `26.48177` | `0.9883` |
| `O-H` | `3.90955`  | `5.38233`  | `0.7264` | `4.14232`  | `0.9438` |
| `H-H` | `0.58678`  | `1.69868`  | `0.3454` | `0.65147`  | `0.9007` |
| total | `44.15682` | `45.57962` | `0.9688` | `45.65691` | `0.9671` |

Per-pair C6 goes from wrong by up to a factor of three to within 1–10%. Note the total was
already within 3% of the DF reference — a pairwise-blind check would have missed the error
entirely, which is why the per-pair oracle matters.

## ISA populations

Ours at the reviewed basis against CamCASP's ISA-GRID shape-function charges. Our fixed point
converged in 32 iterations with max overlap residual `7.0e-10`, max unity residual `2.2e-16`
and electron count `10.00000002`.

| site | ours | CamCASP | delta | ours `q` | CamCASP `q` |
| ---- | ---- | ------- | ----- | -------- | ----------- |
| `O`  | `8.83434` | `8.81587` | `+0.01847` | `-0.8343` | `-0.8159` |
| `H1` | `0.58283` | `0.58866` | `-0.00583` | `+0.4172` | `+0.4113` |
| `H2` | `0.58283` | `0.58866` | `-0.00583` | `+0.4172` | `+0.4113` |

Agreement to `0.018 e` on oxygen and `0.006 e` on hydrogen, consistent with the two being
different ISA variants (ours real-space throughout; CamCASP's shape functions come from the
basis-space ISA-A functional) and with CamCASP's own `0.0068` residual charge. These are the
Task B acceptance literals.

## Two blockers found while running our side

Both are real and neither is caused by the partition.

1. **The response provenance seal refuses a converged SCF.** At aug-cc-pVTZ the cation UKS
   converges by Psi4's own criteria — `Energy and wave function converged`, final
   `Delta E = -3.08e-07` and `RMS |[F,P]| = 8.92e-07` against `1e-6` thresholds — yet
   `FrozenResponseContext` rejects it with `cation SCF state has no finalized provenance seal`.
   Convergence is marginal, and `HF::capture_response_provenance_if_converged` re-derives the
   criteria from its own last *observed* iteration rather than trusting the SCF's verdict.
   Tightening to `e_convergence 1e-10, d_convergence 1e-9` clears it. **This makes the reviewed
   parity protocol unrunnable as written and should be fixed before Task 8**; the protocol
   dictionary needs explicit convergence keys either way.
2. **`PARITY_PROTOCOL`'s localization tolerance is unattainable.** It sets
   `atomic_polarizability_localization_tolerance = 1e-8`, but the measured LW charge-sum
   residual at aug-cc-pVTZ with DFT `590/99` and ISA `100/24/32` is `5.39e-07`, so
   `localize_lw` fails its postcondition. For reference the reviewed ORIENT run used
   `Sum-rule test 1e-7` and tolerated its own rank-truncation residuals up to `7e-4`. The
   numbers above were taken at `1e-6`, a **diagnostic** relaxation which is irrelevant at the
   1% level this comparison probes; the parity gate itself (`rtol=1e-4, atol=1e-5`) was not
   touched. Task 8 must either justify `1e-8` with a converged grid study or set the tolerance
   from measurement.

## Consequences

1. **Every feature of our "misdistribution" is reproduced by the partition change alone.** ISA
   makes oxygen nearly isotropic (`7.13 / 7.04 / 7.47`) where DF is strongly anisotropic
   (`5.58 / 7.04 / 5.76`), and moves out-of-plane response off the hydrogens
   (`alpha_yy` `1.62 -> 0.76`). `alpha_xx` barely moves on either site. That is precisely the
   directional signature recorded in the debugging map — "`alpha_xx` agrees well on both sites
   while `yy` and `zz` are badly split" — and it is a property of the partition, not a bug.
2. **The hydrogen dipole off-diagonal was never wrong.** The reviewed DF model has
   `H1 alpha_zx = -0.00576`. Under ISA the correct value is `-0.645`, and our pipeline publishes
   `-0.679`. The rank-1 anchor fix stands — the reviewed PFIT log's seven penalized parameters
   settled the anchor *scope* as a matter of fact, and an unanchored `+4.29` was still wrong —
   but the target it was being compared against was the wrong model's.
3. **Task G (C-DF partitioning) is not needed for parity** and should stay deferred. The
   existing DF reference remains valid; it simply answers a different question.
4. Tasks A–E are unchanged in scope but now have a matching oracle to be validated against,
   including per-site `alpha` at `rtol=1e-4, atol=1e-5` and the ISA charges above.

## Reproduction

Development-only, run outside the repository:

```
/fastscratch/awallace43/task-f-isagrid/
  control/   reviewed cks, reused MOs -> bit-identical reproduction of the reviewed artifacts
  setupA/    runcamcasp.py --setup, used only to learn the Basis Atom-Aux syntax
  exp1/      the ISA-GRID run and its localized/refined models
```

Per the plan's Global Constraints, only fixed literals from this run may be checked in; nothing
in the repository may read `.camcasp-reference/` or invoke CamCASP, ORIENT, PFIT or CASIMIR.
