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
