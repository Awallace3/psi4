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

Three of these were found only when the run was reproduced on 2026-08-20. The recipe above
is **not sufficient on its own**; all three are required.

- **`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` are mandatory for bit-reproducibility.** The
  statically linked `camcasp` binary carries an OpenBLAS built with OpenMP loops (it prints
  `OpenBLAS Warning : Detect OpenMP Loop`), so on a many-core host the default thread count
  changes the reduction order. Without them the "Control" step above **fails**: every product
  drifts at ~`1e-8` relative (`3.5222874976599733E-02` against `3.5222875406388465E-02`)
  while remaining self-consistent and self-reproducible, so the run looks healthy and the
  bit-for-bit claim silently does not hold. With them the control is exactly byte-identical
  across `*_NL4_fmtA.pol`, `*_NL4_fmtB.pol`, `*.p2p` and `*_DMA2_L4.mom`.
- **`orient` must be on `PATH`** or `localize.py` aborts with "Can't find the Orient program".
  The working executable is `.camcasp-reference/tools/orient/bin/orient`, ORIENT 5.0.10, the
  same version the reviewed run used.
- **The job directory must sit exactly one level under `.camcasp-reference/work/`**, for the
  `CGdir` reason recorded above.

- **`DIST-ALG ISA-GRID` is not a one-line change.** It needs a preceding `Begin ISA` block; the
  spec's earlier "one directive away" claim was too optimistic and is corrected here.
- **ISA algorithm A requires a spherical Atom-Aux basis.** With the reviewed Cartesian aux basis
  and no Atom-Aux block, CamCASP fails with `Basis-based ISA schemes B1, B2 & A need AtomAux
  basis sets with spherical GTOs`. Adding a spherical Atom-Aux block clears it without
  perturbing the response basis.
- CamCASP's ISA-GRID is a hybrid: shape functions from the basis-space ISA-A functional,
  applied pointwise on the integration grid. Ours is real-space throughout.
- CamCASP's dispersion step (`casimir`) failed on a hardcoded relative runtime path from the
  reviewed run's directory depth. Not pursued at the time: our own recoupling is verified to
  `2.5e-7`.

  > **Resolved 2026-08-20 (Task B10).** The path is the `.data` file's
  > `CGdir ../../tools/camcasp-runtime/data/realcg`, read by `casimir` itself and not by
  > `process`. It resolves correctly from any job directory exactly one level under
  > `.camcasp-reference/work/`; the original attempt ran three levels deep under
  > `/fastscratch`, where it resolved to a nonexistent path. **No path rewrite is needed** —
  > only the right directory depth. The ISA-GRID casimir run is now complete; see
  > "## Anisotropic dispersion coefficients" below.

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

Both are real and neither is caused by the partition. **Both are now fixed in
`PARITY_PROTOCOL`,** which was verified to run end to end and publish all seven variables
after the change.

1. **The response provenance seal refuses a converged SCF.** At aug-cc-pVTZ the cation UKS
   converges by Psi4's own criteria — `Energy and wave function converged`, final
   `Delta E = -3.08e-07` and `RMS |[F,P]| = 8.92e-07` against `1e-6` thresholds — yet
   `FrozenResponseContext` rejects it with `cation SCF state has no finalized provenance seal`.
   Convergence is marginal, and `HF::capture_response_provenance_if_converged` re-derives the
   criteria from its own last *observed* iteration rather than trusting the SCF's verdict.
   Tightening to `e_convergence 1e-10, d_convergence 1e-9` clears it, and those keys are now
   pinned in `PARITY_PROTOCOL`. The underlying strictness remains: the seal will refuse any
   marginally converged state, so a protocol that inherits Psi4's `1e-6` defaults at a diffuse
   basis can still fail closed. Worth reconciling with Psi4's own verdict, but no longer a
   blocker.
2. **`PARITY_PROTOCOL`'s localization tolerance is unattainable.** It sets
   `atomic_polarizability_localization_tolerance = 1e-8`, but the measured LW charge-sum
   residual at aug-cc-pVTZ with DFT `590/99` and ISA `100/24/32` is `5.39e-07`, so
   `localize_lw` fails its postcondition. For reference the reviewed ORIENT run used
   `Sum-rule test 1e-7` and tolerated its own rank-truncation residuals up to `7e-4`. The
   numbers above were taken at `1e-6`, which is also what the wiring protocol uses and is now
   what `PARITY_PROTOCOL` sets. The parity gate itself (`rtol=1e-4, atol=1e-5`) was not
   touched.

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

The `ISA_GRID_*` polarizability literals are reproducible from the tracked extractor rather than
from an ad-hoc script. `devtools/camcasp_reference.py` already carries the whole path — using
`parse_refined_polarizabilities(refined, ACCEPTED_ATOM_LABELS, limit=3)`, `build_local_frames`
against the run's `H2O.axes`, then `dipole_local_cartesian` and `rotate_tensor` per site, packed
as `xx, xy, xz, yy, yz, zz` — and it reproduces the checked-in values to exactly `0.0`. Its
`build` subcommand cannot be used end to end here only because it requires a CASIMIR `.pot`
file, which this run has none of; the Cn literals were recoupled with our own engine instead.

## Resolution: two oracles, two gates (2026-08-18)

With `PARITY_PROTOCOL` fixed, the six reviewed-literal comparisons execute rather than skip.
They cannot pass, because those literals record the **DF-partitioned** calculation. The open
decision named in the earlier revision of this section is now closed, along the line the
partition spec's resolved decision 1 already set out ("both, ISA-GRID first; the existing DF
reference is retained as a second data point"):

- The ISA-GRID model above was extracted into `ISA_GRID_*` literals in
  `tests/pytests/test_atomic_polarizabilities.py` — all eleven frequencies, per site, plus
  C6–C12 recoupled with our own engine — and is the **acceptance oracle**, compared inside an
  explicitly measured band.
- The `DF_*` literals are **retained unchanged** and their comparisons stay at the plan's
  `rtol=1e-4, atol=1e-5` gate as `xfail(strict=True)`. They record what a C-DF partition
  (Task G) would have to satisfy; `strict=True` means implementing one produces a loud failure
  demanding the marker be removed rather than a quiet pass.

Neither the gate nor any literal was altered.

**The extraction is validated, not asserted.** Run against the reviewed DF model, the same
extractor reproduces the previously reviewed 33x6 literal table to exactly `0.0`, and the same
recoupling harness reproduces the checked-in `DF_C6`–`DF_C12` to their printed precision. Only
then was it applied to the ISA-GRID model. The frame handling matters: the `.pol` files report
each site in its own local axes, and H1's are the molecular axes rotated 180 degrees about z,
so a real harmonic `lm{c,s}` picks up `(-1)^m` and `alpha_(t,u) -> (-1)^(m_t+m_u) alpha_(t,u)`.

**A geometry defect surfaced in the process.** The test module's `REVIEWED_GEOMETRY` placed
`H1` at *positive* x while the reviewed `H2O_A.in` places it at negative x — a mirror image.
That is invisible in every x-even component and flips the sign of `xz` and `yz`. It is fixed;
`H1 alpha_xz` now reads `+0.653` against the oracle's `+0.645` instead of `-0.653`.

## Measured band, and what it does not cover

Ours at `PARITY_PROTOCOL` against the ISA-GRID oracle:

| quantity | worst relative deviation | where |
| -------- | ------------------------ | ----- |
| static dipole block | `0.1529` | `H alpha_yy` (ours `0.6446`, oracle `0.7609`) |
| dynamic dipole block | `0.1529` at `omega=0`, falling monotonically to `0.0443` at `omega=37.8` | same component |
| per-pair `C6` | `0.0993` | `H-H` |
| per-pair `C8` | `0.2552` | `H-H` |
| per-pair `C10` | `0.3593` | `H-H` |
| per-pair `C12` | `0.4555` | `H-H` |

Components that are zero by symmetry are exactly `0.0` in both, so the absolute floor is `1e-5`.

**Direction of effect, independent of any band.** On each of the eight components where the two
oracles separate by more than 5 percent — `O yy/zz` and `H xz/yy/zz` — the published value is
closer to ISA-GRID by factors of 7.9 to 83:

| component | ours | `d` to ISA-GRID | `d` to DF | ratio |
| --------- | ---- | --------------- | --------- | ----- |
| `O yy`  | `7.4144` | `0.0594` | `1.6523` | `27.8` |
| `O zz`  | `6.9551` | `0.1739` | `1.3714` | `7.9`  |
| `H xz`  | `0.6531` | `0.0078` | `0.6474` | `83.0` |
| `H yy`  | `0.6446` | `0.1163` | `0.9728` | `8.4`  |
| `H zz`  | `1.1591` | `0.0817` | `0.8505` | `10.4` |

This is asserted as a test, because no tolerance can satisfy it. The `xx` components are
deliberately excluded: the two oracles agree there to better than one percent (`O` `0.0002`,
`H` `0.0084` separation), so which is nearer is noise — and DF is in fact marginally nearer on
`H xx`, `0.035` against `0.048`. The discriminating set is pinned in the test so it cannot
quietly shrink to whichever components happen to agree.

**Finding: the partition does not explain the higher-rank deficit.** Once the oracle matches,
C6 lands within 10 percent per pair — but the deficit grows monotonically with rank, and it
grows against *both* CamCASP models even though the recoupling is bit-identical across the
comparison. Per-pair ratios to ISA-GRID:

| coefficient | `O-O` | `O-H` | `H-H` | total |
| ----------- | ----- | ----- | ----- | ----- |
| `C6`  | `0.988` | `0.944` | `0.901` | `0.967` |
| `C8`  | `0.802` | `0.771` | `0.745` | `0.789` |
| `C10` | `0.737` | `0.690` | `0.641` | `0.717` |
| `C12` | `0.653` | `0.589` | `0.545` | `0.628` |

So our rank-2 and rank-3 site blocks are systematically smaller than CamCASP's. That is a Task
5 (`refine_wsm`) or rank-truncation question, not a partition one, and it is now the leading
parity gap. Untested candidates: the 329-point fit grid against the reviewed 2000, the relative
rank cutoff pruning rank-3 columns, and the anchor penalty constraining only rank 1.

## Anisotropic dispersion coefficients (added 2026-08-20, Task B10)

The ISA-GRID `casimir` run that this document originally abandoned is now complete, giving a
**partition-matched external oracle for the full anisotropic `Cn[l1 k1, l2 k2, j]` table**
rather than only the isotropic `00 00 0` row.

Product: `.camcasp-reference/work/H2O-isagrid/H2O_ref_wt4_L3_casimir.out`, sha256
`a405b6b07904beecdb7d7fb7a527cc64d0b92355212261a8c5b349a09c296999`.

### The run is the same one this document already reported

Every intermediate is byte-identical to the run whose numbers appear in the tables above, so
the anisotropic table and the existing `ISA_GRID_*` literals come from one calculation:

| artifact | result |
| -------- | ------ |
| `H2O_atoms.ISA`, `H2O_ISA-GRID_f11_NL4_fmtB.pol` | byte-identical |
| `H2O.pdef` (regenerated, not copied) | byte-identical |
| `H2O_L3_0f10.pol`, `H2O_ref_wt4_L3_0f10.pol` | byte-identical |
| `H2O_casimir.temp`, `H2O_ref_wt4_L3_casimir.data` | byte-identical structure |
| emitted `*.p2p` | byte-identical to the **reviewed DF** run |

ISA convergence reproduced exactly: 40 iterations, all sites `Y`, residual charge
`0.006810839`, `KL(BASIS) = 0.070602963`, `KL(GRID) = 0.044905972`, populations
`O 8.8158670` / `H 0.58866109`.

The `.data` diff against the reviewed DF casimir input is the cleanest possible statement of
the experiment: **529 lines each, all 189 structural lines identical byte-for-byte at
identical positions, all 340 value lines different.** Every protocol constant — `Maximum
rank 3`, `Limit rank to 3`, `Limit rank to 3 for H1 H2`, `Frequencies STATIC + 10`, `cutoff
0.0001`, `Print nonzero`, `Skip 0`, `Dispersion 12` — is provably unchanged. Only the
partition moved.

### End-to-end validation

CamCASP's own recoupling, run on its own ISA-GRID refined model, reproduces the isotropic C6
that **our** engine produced from the same model:

| pair | CASIMIR `00 00 0` C6 | reviewed `ISA_GRID_C6` literal |
| ---- | -------------------- | ------------------------------ |
| O-O  | `26.48177`  | `26.48176709`  |
| H-O  | `4.142317`  | `4.142316899`  |
| H-H  | `0.6514697` | `0.6514696683` |

Agreement to all seven printed significant figures, each printed value being the correct
7-figure rounding of the literal. The same extraction applied to the reviewed DF output
returns `17.25559 / 5.382332 / 1.698678`, matching the DF column exactly — so the extraction
is sound and the difference between the two oracles really is the partition.

Higher orders, O-O: C8 `490.4584`, C10 `9673.248`, C12 `150417.4`.

### Table shape

Three site-**type** pair blocks in the order `O O`, `H O`, `H H` (note `H O`, not `O H`), with
979 / 1840 / 3466 nonzero rows, 6285 total — **identical row counts to the DF output**, so the
two share a sparsity pattern and differ only in values.

### On-disk format, and the one way to get it silently wrong

Rows are `l1k1  l2k2  j` followed by values for C6..C12 in order. `Print nonzero` omits
all-zero rows entirely, so an absent label means an exact zero and the row count is not fixed.

**Values are left-aligned from C6, and only *trailing* zeros are truncated.** Leading and
interior zeros are printed. The decisive evidence is that `10 10 0` carries seven fields of
which the first two are explicit `0.0`:

```
    10   10   0     0.0            0.0         -1.840416         0.0         -3.878088         0.0         -5.086604
```

Under a "leading zeros omitted, columns right-shifted" reading that row could only have five
fields. A parser that left-pads instead of right-pads shifts every coefficient by one order
and still produces plausible, correctly-signed, correctly-magnituded output — the exact
silent-wrong-answer failure mode of §0.2.

The invariant that catches it, derived and then confirmed against every sampled row:

> **`n === j (mod 2)`** — a label's nonzero orders all share the parity of its coupled
> rank `j`.

> **Corrected 2026-08-20 (task B10).** This invariant was originally written as
> `n === l1 + l2 (mod 2)`, generalising from a sample in which every row happened to have
> `l1 + l2 === j`. Measured over the whole file: `n === j (mod 2)` holds on **0 of 10457**
> nonzero coefficients as a violation, while `n === l1 + l2 (mod 2)` is violated by **2968**
> of them — the first counterexample being O-O `20 22s 1`, nonzero at C7 (-0.009141865), C9 and
> C11 with `l1 + l2 = 4` against odd `n`. The `j` form is the one the derivation predicts: step 5 of
> [`2026-08-18-anisotropic-recoupling-derivation.md`](2026-08-18-anisotropic-recoupling-derivation.md)
> forces `L1 + L2 + j` even with `L1 + L2 = n - 2`, and there is no parity constraint on
> `l1 + l2` at all. The alignment argument below is unaffected, because `00 10 1` has
> `l1 + l2 = j = 1`. Parsers must enforce the `j` form; enforcing the `l1 + l2` form would
> reject the file.

Confirmed on `00 00 0`, `00 10 1`, `10 10 0`, `10 10 2`, `00 20 2`, `00 22c 2`, `00 30 3`,
`10 20 1`, `20 20 0`. Under the shifted reading, `00 10 1` would place its nonzeros at even
orders against `l1 + l2 = 1`, so **the misalignment violates this invariant on the first
odd-`j` row it touches.** Parsers must enforce it. There is also a per-label minimum order:
`10 10 *` starts at C8 because `l1 = 1` forces `la != la'` on both sites, hence
`la + la' >= 3` and `n >= 8`; `00 10 1` starts at C7.

Fortran `E` exponents appear as `0.9276924E-01`.

### C13 and C14 have no oracle at any rank

`casimir` **hard-caps at C12**. Setting `Dispersion 13`, `14` or `16` aborts immediately with

```
Dispersion coefficients only up to C12
```

exit code 9 plus a `casimir.error` marker; the string is compiled into the binary, so it is
not a keyword or protocol setting. `process` likewise emits `Dispersion 12` for an L3 model,
and `localize.py` hardcodes wsmlimit 3 -> 12.

An L3 model does reach `n = 14` by the rank algebra, so this is CamCASP declining to compute
orders it could define. **Consequence: the decision to publish only `n <= 12` while computing
6..14 internally discards nothing that could ever have been validated externally.** The
missing C13/C14 is not an extraction gap to close. Evidence at
`.camcasp-reference/work/H2O-c1314-probe/`.

### The anisotropic parity comparison, and the convention it did *not* resolve (2026-08-20, B10)

Measured at `PARITY_PROTOCOL` on `REVIEWED_GEOMETRY` under `PARTITION=ISA`, comparing
`ATOMIC DISPERSION COEFFICIENTS` against the table above. Test:
`tests/pytests/test_anisotropic_dispersion_parity.py`.

**The run is the right one.** The isotropic `00 00 0` entries of the anisotropic array
reproduce the already-recorded `ISA_GRID_*` bands to the digit: C6 `0.0117 / 0.0562 / 0.0993`
and C12 `0.347 / 0.411 / 0.456` on O-O / H-O / H-H. So everything below is a property of the
anisotropic sector, not of the run or the extraction.

**Frames.** Our array is indexed by *ordered site pairs*, row `A * 3 + B`, with every site
frame the molecular frame. CASIMIR prints one block per site *type* in each site's own local
axes, and `H2O.axes` says `H1 z global Z x from H2 to H1`, `H2 z global Z x from H1 to H2`.
With H1 at negative x, **H2's local axes are the molecular axes and H1's are those rotated by
pi about z**, so the comparable rows are `(O,O)`, `(H2,O)`, `(H2,H2)` and no rotation is
needed. Using H1 multiplies every coefficient by `(-1)^(|m1| + |m2|)`. Verified rather than
assumed: the `(H1,·)` rows equal the `(H2,·)` rows under exactly that factor, 0 violations out
of 5921 / 3195 / 3195 entries above 1e-6 of the per-order scale, worst relative deviation
`1.65e-11`.

#### The exchange conventions really do differ

| table | law under `(l1 k1) <-> (l2 k2)` at a fixed diagonal site pair | violations |
| ----- | ------------------------------------------------------------- | ---------- |
| ours | `C[l2 k2, l1 k1, j] = (-1)^(l1+l2) C[l1 k1, l2 k2, j]` | 0 / 1706 (O,O), 0 / 5921 (H2,H2) |
| CASIMIR | `C[l2 k2, l1 k1, j] = C[l1 k1, l2 k2, j]`, identical digits | 0 / 1671 (O-O), 0 / 5703 (H-H) |

Ours is the law the derivation proves (§B.5 check 4, residual `1.01e-16`, tied to the physical
statement `E(A,B;R) = E(B,A;-R)` at `6.66e-16`). The counts for our side are taken above a
noise floor of `1e-6` of the per-order scale; with no floor there are 68 / 568 apparent
violations out of 16985, all of them cancellation-dominated entries below that floor.

**Correction to an earlier analysis.** `(-1)^(l1+l2)` cannot reconcile the two, because it is
*symmetric* under the label exchange and therefore cannot change any table's exchange
symmetry. `(-1)^l1` and `(-1)^l2` both can — if `A(2,1) = (-1)^(l1+l2) A(1,2)` then
`B = (-1)^(l2) A` satisfies `B(2,1) = (-1)^(l1) A(2,1) = (-1)^(2 l1 + l2) A(1,2) = B(1,2)` —
and they differ from each other exactly on the `l1 + l2` odd labels, so at most one of them
can be right. Verified numerically on a synthetic table obeying the law.

#### No sign map reconciles the tables

Over the 10457 shared nonzero coefficients (CASIMIR's `j <= 8`, `n <= 12`, restricted to
labels it prints as nonzero; **0** of its nonzero entries falls outside our published label
set):

| map applied to ours | sign disagreements | median \|rel\| | p90 | worst | inside 1e-4 |
| ------------------- | -----------------: | -------------: | --: | ----: | ----------: |
| identity | 5343 | 1.1760 | 7.57 | 3088.8 | 0 |
| `(-1)^l1` | 5158 | 1.1576 | 7.61 | 3090.8 | 0 |
| `(-1)^l2` | 5134 | 1.1369 | 7.71 | 3090.8 | 0 |
| `(-1)^(l1+l2)` | 5295 | 1.1663 | 7.54 | 3088.8 | 0 |
| `(-1)^j` | 5263 | 1.1652 | 7.55 | 3088.8 | 0 |
| `(-1)^(l1+l2+j)` | 5313 | 1.1731 | 7.61 | 3088.8 | 0 |
| `i^(l1-l2-j) / Ncal` of §4.4 | 5189 | 1.1443 | 7.67 | 3090.8 | 0 |
| `(-1)^floor((l1-l2-j)/2)` | 5271 | 1.1524 | 7.67 | 3090.8 | 0 |

**No sign function explains 100 % of the flips, and none comes near.** The best of twelve
tried explains 51.1 % of the observed signs, against an identity baseline of 48.9 %. Every map leaves a residual near or above 1, i.e.
near a pure sign, and none lands in the 1-10 %..46 % band of the known property deficits.

More decisively, **the required sign is not a function of the label at all**, so no
S-function phase convention — which by construction is one real `±1` per `(l1, l2, j)`
(§9.1 of the derivation) — can be the explanation:

| grouping | populated groups | groups requiring *both* signs |
| -------- | ---------------: | ----------------------------: |
| `(l1, l2, j)` | 128 | 114 |
| `(n, l1, l2, j)` | 273 | 238 |

This is not cancellation noise. Restricted to entries above 1 % of the per-order scale:

| order | entries | sign disagreements | `(l1,l2,j)` groups | mixed-sign groups | median \|rel\| |
| ----- | ------: | -----------------: | -----------------: | ----------------: | -------------: |
| C6 | 27 | **0** | 6 | **0** | 0.467 |
| C7 | 113 | 57 | 16 | 14 | 1.749 |
| C8 | 176 | 96 | 23 | 18 | 1.220 |
| C9 | 434 | 215 | 34 | 30 | 1.069 |
| C10 | 413 | 213 | 40 | 32 | 1.053 |
| C11 | 717 | 350 | 53 | 47 | 1.001 |
| C12 | 593 | 300 | 51 | 43 | 1.003 |

C6 is clean and everything above it is not.

#### There *is* a convention difference, and it is a magnitude, not a sign

C6 is the only published order at which exactly one site-block quadruple contributes —
`(la, la', lb, lb') = (1,1,1,1)` is the only solution of `la+la'+lb+lb' = 4` with every rank
in 1..3 — so at fixed `(l1, l2, k1, k2)` the `j` dependence of the ratio ours/CASIMIR measures
the recoupling prefactor and nothing else. Measured:

> **ratio(j) x `|<l1 0; l2 0 | j 0>|` is independent of `j`, to `6.45e-07`** — the printed
> precision of the reference — for every one of the `4 + 6 + 9 = 19` component pairs
> `(k1, k2)` of the three blocks, at `(l1, l2) = (2, 2)`, `j = 0, 2, 4`.

`1 / |<2 0; 2 0 | j 0>|` is `2.23607 / 1.87083 / 1.39443` for `j = 0 / 2 / 4`; the raw ratios
on H-H are `2.40573 / 2.01275 / 1.50044`, and the worst per-component spread after the
correction is `6.45e-07`. Example, H-H `(k1,k2) = (3,3)` (`22c 22c`):
`1.1217966 / 1.1217966 / 1.1217966`.

The rival reading — that the factor is `<L1 0; L2 0 | j 0>` with `L1 = L2 = 2`, which the
`j` scan alone cannot distinguish because `(l1, l2) = (L1, L2)` at C6 — is excluded by the
rank cross-product identity. Writing `rho` for the per-rank physical residual, a correct
factor must leave `rho_0^A rho_2^B . rho_2^A rho_0^B = rho_0^A rho_0^B . rho_2^A rho_2^B`:

| factor | H-H | H-O | O-O |
| ------ | --: | --: | --: |
| `<l1 0; l2 0 \| j 0>` | `0.969017` vs `0.969232`, **2.2e-04** | `1.684319` vs `1.682947`, **8.2e-04** | `2.937` vs `2.474`, 1.9e-01 |
| `<L1 0; L2 0 \| j 0>` | fails by **36 %** | — | — |

(The O-O column fails under both for a physical reason, below.)

**This factor does not extend to the whole table.** `<l1 0; l2 0 | j 0>` vanishes identically
whenever `l1 + l2 + j` is odd, and **2968 of the 10457** shared nonzero entries (28.4 %) are
of that kind, with both tables nonzero there — the largest being O-O C12 `22s 30 4`, CASIMIR
`-2194.16` against ours `-276.144`.

> **Corrected 2026-08-20 on review.** An earlier draft of this section concluded from the
> paragraph above that "a convention factor that annihilates 28 % of the reference is not a
> convention", and therefore that one of the two `j` dependences must be wrong. **That
> inference does not hold, and the stronger claim is withdrawn.**
>
> It assumes the reconciling factor must be *literally* `<l1 0; l2 0 | j 0>` on every label.
> It need not be. Those 2968 entries are **exactly** the sine-carrying labels: measured over
> the whole file, `l1 + l2 + j + sigma` is even on **6285 of 6285** rows, so
> `l1 + l2 + j` odd is identical to `sigma` odd. The real S functions on that sector are built
> from sine components, whose normalisation involves Clebsch-Gordan coefficients with
> **nonzero** `m`, not the `m = 0` coefficient fitted here. An `m = 0` CG is only the right
> object for the cosine-only sector, so its vanishing on the sine sector is expected and
> carries no information about whether either table is wrong. A normalisation equal to
> `1/|<l1 0; l2 0 | j 0>|` on the `sigma`-even sector and finite on the `sigma`-odd sector is
> entirely admissible.
>
> The evidence in fact points the other way: if our `j` dependence were genuinely wrong, a
> *convention-shaped* correction would not be expected to land H-H inside the independently
> recorded 1-10 % deficit band — and it does, factorising per component to 0.07-0.25 %.

**So the conclusion of B10 is negative, specific, and bounded:**

1. On the `sigma`-even sector at C6 the two tables differ by exactly `1/|<l1 0; l2 0 | j 0>|`,
   to `6.45e-07` — the oracle's printed precision — across all 19 component pairs of all three
   blocks. That is an exact, reproducible localisation of a real discrepancy in the `j`
   dependence of the recoupling prefactor.
2. After removing it, the residual on H-H **is** the already-known property deficit and nothing
   more, and the O blocks are fully accounted for by measured cancellation amplification
   (below). So on that sector the factor is very likely the correct reconciliation.
3. **Which side matches Stone's convention cannot be decided internally**, and the `sigma`-odd
   sector is not covered by the factor at all. Both remain open.

§9.1 of the derivation already named this as the open item: the
`<L1 0; L2 0 | j 0> * Lambda_j` split inside `g^r` is invariant under `C -> C.kappa`,
`S -> S/kappa`, so the derivation's own §B.5 check 2 — direct energy reconstruction, exact to
machine precision — cannot and does not pin it. **B10 has now measured the discrepancy and
localised it exactly; it has not established which side is right.** Resolving that needs the
published Stone S-function definition or a third independent table, and **no internal check
can substitute.** Until then, do not "fix" either implementation.

#### What is left after the C6 factor, and where it lives

Residual `|`ours/CASIMIR`|` at C6 after multiplying by `|<l1 0; l2 0 | j 0>|`:

Columns are `(l1, l2)`, and `l1` belongs to the block's **first** printed type — so on the
`H O` block `(2, 0)` is rank 2 on H and `(0, 2)` is rank 2 on O.

| block | `(0,0)` | `(0,2)` | `(2,0)` | `(2,2)` |
| ----- | ------: | ------: | ------: | ------: |
| H-H | 0.9007 | 0.9844 | 0.9844 | 1.0761 |
| H-O | 0.9438 | 1.6324 | 1.0318 | 1.7831 |
| O-O | 0.9883 | 1.7139 | 1.7139 | 2.5031 |

On **H-H** that is `9.9 % / 1.6 % / 7.6 %` — inside the recorded 1-10 % ISA-GRID C6 band, and
it factorises into a per-component product to `0.07-0.25 %` (`rho_20 = 1.0310`,
`rho_21c = 1.0229`, `rho_22c = 1.0592` against `rho_0 = 0.9490`). So on H-H the anisotropic
residual *is* the already-known property deficit and nothing more.

On every block containing **O** it is 63-150 %, and the excess sits entirely on labels
carrying a rank-2 coupled index on the O site — `(0,2)` and `(2,2)` in the table above, never
`(0,0)` or the H-side `(2,0)`. The reason is cancellation, not a new defect, and it is
directly measurable in the dipole block that feeds C6. Static `alpha` against the same
oracle:

| site | per-component | isotropic mean | coupled rank-2 `q20 = (2 a_zz - a_xx - a_yy)/2` |
| ---- | ------------- | -------------- | ----------------------------------------------- |
| O | `xx 6.920310/7.041967` (1.7 %), `yy 7.414385/7.473775` (0.8 %), `zz 6.955088/7.128955` (2.4 %) | 1.6 % | `-0.212260` vs `-0.128916`, **64.7 %** |
| H | `yy 0.644611/0.760938` (15.3 %) worst | 6.9 % | `+0.067367` vs `+0.066802`, **0.85 %** |

O's dipole polarizability is nearly isotropic, so `q20` is a small difference of large
numbers and a 2 % component error becomes a **65 %** error in the traceless part — which is
precisely the `1.714` ratio the C6 `(0,2)` labels show, to 4 %. H's dipole block is strongly
anisotropic, so the same arithmetic does not amplify: H is 15 % out per component and 0.85 %
out in `q20`. **The O-O anisotropic excess is the O dipole deficit amplified about 40-fold by
cancellation, not a separate defect.** Consistent with that, the per-component factorisation
that works to `2.2e-04` on H fails at `1.9e-01` on O-O.

Median relative deviation at C6 falls from `1.2275` to `0.1218` once the factor is applied;
over the whole shared set it barely moves, `1.1760` to `1.0677`, with 0 entries inside `1e-4`
either way.

#### The B10 parity table (identity map, as published)

| pair | n | compared | inside 1e-4 | median \|rel\| | p90 | worst \|rel\| | worst label |
| ---- | -: | -------: | ----------: | -------------: | --: | ------------: | ----------- |
| O-O | 6 | 17 | 0 | 2.469 | 5.21 | 5.88 | `22c 20 0`, ours 6.497e-04 vs 9.443e-05 |
| O-O | 7 | 46 | 0 | 5.287 | 76.4 | 105.0 | `32c 20 1`, ours 8.054e-03 vs -7.745e-05 |
| O-O | 8 | 104 | 0 | 1.436 | 61.7 | 317.8 | `22s 32c 2`, ours 0.291903 vs -9.213e-04 |
| O-O | 9 | 194 | 0 | 1.499 | 31.6 | 183.8 | `32c 40 3`, ours 0.138674 vs -7.586e-04 |
| O-O | 10 | 334 | 0 | 1.276 | 8.54 | 3088.8 | `32c 52c 4`, ours 8.126e-02 vs 2.630e-05 |
| O-O | 11 | 412 | 0 | 1.046 | 2.98 | 57.7 | `42c 30 5`, ours -4.86195 vs 8.581e-02 |
| O-O | 12 | 564 | 0 | 1.047 | 3.03 | 32.7 | `60 42c 8`, ours 7.74861 vs -0.244681 |
| H-O | 6 | 24 | 0 | 1.991 | 3.02 | 3.82 | `22c 22c 0`, ours -2.638e-03 vs -5.476e-04 |
| H-O | 7 | 79 | 0 | 2.887 | 42.7 | 160.4 | `32c 22c 1`, ours -7.014e-03 vs 4.401e-05 |
| H-O | 8 | 181 | 0 | 2.071 | 25.6 | 141.4 | `30 32c 0`, ours 5.370e-02 vs 3.771e-04 |
| H-O | 9 | 356 | 0 | 1.530 | 19.3 | 1354.2 | `43c 32c 7`, ours -20.2878 vs 1.499e-02 |
| H-O | 10 | 612 | 0 | 1.229 | 6.62 | 631.4 | `52c 32c 4`, ours -2.515e-02 vs -3.976e-05 |
| H-O | 11 | 781 | 0 | 1.048 | 5.30 | 145.9 | `32s 22s 1`, ours 11.7693 vs -8.122e-02 |
| H-O | 12 | 1050 | 0 | 1.058 | 3.56 | 240.3 | `43c 22c 2`, ours 92.5743 vs 0.383661 |
| H-H | 6 | 34 | 0 | 0.963 | 1.42 | 1.51 | `22c 22c 0`, ours 1.407e-02 vs 5.610e-03 |
| H-H | 7 | 132 | 0 | 2.367 | 12.0 | 21.3 | `31c 20 1`, ours -4.553e-04 vs 2.239e-05 |
| H-H | 8 | 318 | 0 | 1.867 | 13.2 | 40.6 | `30 32c 0`, ours -4.837e-02 vs -1.163e-03 |
| H-H | 9 | 648 | 0 | 1.557 | 13.0 | 514.5 | `43c 31c 5`, ours -5.980e-02 vs 1.165e-04 |
| H-H | 10 | 1132 | 0 | 1.427 | 9.77 | 1721.5 | `22c 43c 4`, ours -3.67574 vs -2.134e-03 |
| H-H | 11 | 1476 | 0 | 1.097 | 7.54 | 346.7 | `10 22c 1`, ours -23.0982 vs 6.681e-02 |
| H-H | 12 | 1963 | 0 | 1.070 | 5.85 | 222.0 | `22c 22c 2`, ours -52.3156 vs -0.234602 |

Whole shared set: **10457 comparisons, 0 inside `rtol = 1e-4`, median `1.1760`, worst
`3088.8`.** The plan's gate is missed by four orders of magnitude and the test is recorded as
a strict xfail, exactly as the six `DF_*` comparisons are. Nothing was tuned and no tolerance
was loosened.

Note that the worst offenders are *not* the leading coefficients — they are small entries
sitting near a cancellation, where a 30 % error in a large term becomes a factor of 1000.
Selecting instead the single largest `|Cn|` per block per order gives 18 comparisons, median
`0.50`, worst `1.55`; **six of those 18 have the opposite sign, and all six are odd-order
labels with `l1 + l2` odd and `j = 1`** — precisely where the exchange conventions clash —
while the nine even-order ones are all the isotropic `00 00 0` coefficient and reproduce the
recorded ISA-GRID bands exactly.

#### The label sets, and the 735 coefficients that have no oracle

Our published label set **strictly contains** CASIMIR's: 0 of its 10457 nonzero coefficients
lands on a label we do not publish. Of the labels we publish and it omits:

| block | omitted `(label, order)` entries with `j <= 8` | of those, nonzero in ours | entries with `j >= 9` | of those, nonzero |
| ----- | --: | --: | --: | --: |
| O-O | 14579 | **0** | 735 | 67 |
| H-O | 13167 | **0** | 735 | 122 |
| H-H | 10547 | **0** | 735 | 222 |

So every omission inside CASIMIR's `j <= 8` cap is a zero in our table too, and almost all of
them are *bit*-exact: 14523 / 12996 / 10144 of the entries above are literally `0.0`, and the
largest residual among the remainder is `5.68e-14` (`7.11e-15` on H-H) — floating-point
summation noise, against per-order scales of order `10^1` to `10^5`. **The sparsity patterns
agree completely.** The 735 entries above the cap exist only at `n = 11`
(380) and `n = 12` (355), because `j <= 8` cannot be exceeded at lower order, and they are
**not** a negligible tail: the single largest `|C11|` of the entire H-H pair is one of them,
label `43c 54c 9` at `247.242`. `casimir` prints nothing above `j = 8` and refuses
`Dispersion 13` outright, so these coefficients cannot be validated externally at all. They
are a recorded gap, like C13/C14 above, not an extraction gap to close.

#### Consequences for the plan

1. **B10 is complete as a measurement and negative as a gate.** The comparison is landed at
   `rtol = 1e-4` as a strict xfail with the numbers above in its reason string.
2. **The `j` dependence of the recoupling prefactor is now a known open defect**, not merely
   an unverified convention. It is localised to a single identified Clebsch-Gordan factor and
   is measured at C6 to `6.45e-07`. Closing it requires the published `[S13]` §3.3 / `[S78]`
   S-function definition, or an independent third table — not more of our own internal
   checks, all of which are invariant under exactly this ambiguity.
3. **§B.6's sequencing advice is confirmed, twice over.** Even on the block where the
   convention question is settled (H-H at C6) the residual is the rank deficit; and on every
   block containing O the anisotropic comparison is amplified by cancellation to 70-150 %
   from a dipole block that agrees to 2.5 %. Gating anisotropic `Cn` against anything is
   premature until both the rank deficit and the `j`-dependence factor are closed.
