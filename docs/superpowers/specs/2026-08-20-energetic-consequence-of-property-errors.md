# What our distributed-property errors cost in energy

Measured 2026-08-20. Answers the question the parity work could not: the parity results are
in *property* space, and a 46% C12 deficit is alarming there while possibly being irrelevant
in energy. This puts kcal/mol on it.

Driver: `water-pes-aim.py` (development-only, outside the Psi4 tree, to be dropped before any
upstream PR). Findings here are durable; the script is not.

## Method

Rigid water-water surface built from the S22 hydrogen-bonded dimer, scanning `R(O-O)`. Both
monomers are byte-identical copies of the reviewed geometry, Kabsch-superimposed onto the S22
fragments, so the transferred properties belong to the molecule actually being placed.
Properties transform by rank: charges invariant, dipoles rotate once, `alpha` twice.

**Every arm shares the same geometries, the same MBIS permanent multipoles (PBE0/aug-cc-pVTZ,
`O -0.9123`, `H +0.4561`), the same damping parameters and the same energy kernels.** The only
thing that changes between arms is the numbers in the `alpha` / `Cn` tables, so any energy
difference is attributable to the properties and nothing else.

Kernels follow `~/gits/qcmlforge` conventions and were cross-checked against it: the BJ
dispersion sum agrees bit-exactly (`1.1e-16`), and the direct-solve induction agrees with
qcmlforge's Jacobi+SOR iteration to `3.3e-8` kcal/mol, its own convergence threshold. Neither
calls qcmlforge, because its induction model accepts only isotropic scalar `alpha` and
`qcml_dftd3.d3` exposes no seam for user-supplied coefficients.

Damping: Thole "mutual", `a = 0.39`, for induction. Becke-Johnson with D3(I)/SAPT-PBE0
`s6 = 1.0, s8 = 0.8614, a1 = 0.7171, a2 = 0.5375` for dispersion.

## Result 1: dispersion is the problem, induction is not

At the S22 equilibrium `R(O-O) = 2.912 A`, against the partition-matched ISA-GRID oracle:

| term | ours | CamCASP ISA-GRID | delta | delta % |
| ---- | ---- | ---------------- | ----- | ------- |
| induction | `-1.2462` | `-1.2645` | `+0.0182` | `1.4%` |
| dispersion (C6+C8) | `-2.2109` | `-2.5131` | `+0.3022` | `12.0%` |

**The 2.9% molecular polarizability deficit costs only 1.4% of the induction energy**, and the
induction error is nearly flat in `R` — `1.26%` to `1.49%` across 2.8 to 8.0 A — as a linear
property should be. The dispersion error is an order of magnitude larger in kcal/mol and
strongly `R`-dependent, falling from 12.8% at 2.6 A to 4.5% at 8.0 A.

One anomaly worth noting rather than smoothing over: at 2.6 A the induction error collapses to
`0.19%`, an order of magnitude below its plateau. At that separation the mutual-polarization
response is large enough that our per-site anisotropy error and our total-magnitude deficit
work in opposite directions and very nearly cancel. It is a coincidence of this geometry, not
a property of the model, and it is a good reminder that a single-point agreement at short range
is not evidence of a correct polarizability.

By order, at equilibrium:

| order | ours | ISA-GRID | delta |
| ----- | ---- | -------- | ----- |
| C6  | `-1.3561` | `-1.4149` | `0.0589` |
| C8  | `-0.8548` | `-1.0982` | `0.2434` |
| C10 | `-1.0836` | `-1.5580` | `0.4745` |

**The C8 error alone is four times the C6 error in energy terms.** The rank-growing deficit
that looked like a higher-order curiosity in property space is the dominant energetic error.
This localises the remaining parity work: closing C6 further buys little; the rank-2/rank-3
site blocks are where the energy is.

## Result 2: most of the apparent dispersion error is a damping-radius artifact

Stock D3 sets `R0 = a1*sqrt(3 r4r2_A r4r2_B) + a2`, where the `r4r2` ratio *approximates*
`sqrt(C8/C6)`. We have genuine distributed `C8` and `C6`, so Becke and Johnson's original
`R0 = a1*sqrt(C8/C6) + a2` is directly available. Recomputing with it:

| `R(O-O)` | delta, `r4r2` radius | delta, `sqrt(C8/C6)` radius |
| -------- | -------------------- | --------------------------- |
| `2.600` | `+0.6150` (`-12.8%`) | `-0.3800` (`+7.9%`) |
| `2.912` | `+0.3022` (`-12.0%`) | `+0.0557` (`-2.2%`) |
| `3.500` | `+0.0657` (`-9.7%`)  | `+0.0517` (`-7.6%`) |
| `8.000` | `+0.0001` (`-4.5%`)  | `+0.0001` (`-4.5%`) |

**At equilibrium the discrepancy falls from 12.0% to 2.2%, and at 2.6 A it changes sign.**
Our `C8/C6` ratio is smaller than CamCASP's, so a coefficient-derived `R0` is smaller, damping
less, which partly cancels the coefficient deficit.

Read carefully: this does **not** mean the coefficients are better than Result 1 says. It means
the *energetic* consequence of the rank deficit depends strongly on how the damping is
parametrized, and that a large part of the deficit behaves like a uniform scaling that a
refitted damping radius absorbs. Caveat in the other direction: `a1`/`a2` were fitted against
`r4r2` radii, so the second column uses them outside their fit domain. Both columns are
honest; neither alone is the answer.

Actionable consequence: for a force field fitted on our own coefficients, the rank deficit is
substantially absorbable. For a drop-in replacement of coefficients into an existing
`r4r2`-parametrized D3, it is not.

## Result 3: alpha anisotropy is a quarter of the induction energy

Full-tensor `alpha` against the same `alpha` isotropised to `trace/3`, same arm:

| `R(O-O)` | ours, anisotropic | ours, isotropic | anisotropy |
| -------- | ----------------- | --------------- | ---------- |
| `2.600` | `-3.4146` | `-2.4884` | `-0.9262` |
| `2.912` | `-1.2462` | `-0.9510` | `-0.2952` |
| `3.500` | `-0.2625` | `-0.2102` | `-0.0523` |

**24% of the induction energy at equilibrium**, and always attractive. Our anisotropy is
right: we get `-0.2952` where ISA-GRID gives `-0.2834`, a 4% agreement, consistent with the
2-5% per-component agreement the ISA-GRID experiment measured.

This is the concrete argument for publishing the anisotropic tensors rather than the rank-1
trace. Any consumer restricted to isotropic `alpha` -- which includes every induction model in
`qcmlforge` -- discards a quarter of the induction energy of water.

## Result 4: the wrong oracle inverts the sign of the dispersion verdict

Same kernels, C-DF properties instead of ISA-GRID, at equilibrium:

| term | delta vs ISA-GRID |
| ---- | ----------------- |
| induction | `+0.3086` |
| dispersion | `-0.3609` |

Against DF, our dispersion looks **14% too attractive**; against the matching oracle it is
**12% too weak**. Not a magnitude discrepancy -- a sign inversion of the conclusion. This is
the energetic restatement of the ISA-GRID finding, and it is worth keeping because a
plausible-looking sign error in a fitted force field is far harder to detect than a magnitude
error.

## Caveat: a single BJ radius is inadequate for C10

At equilibrium `C6 + C8 + C10 = -3.29` kcal/mol. No SAPT reference was computed here, so
that number is not being compared against one; the diagnostic is internal.
The `C10` term alone (`-1.08`) is nearly as large as `C6` (`-1.36`), i.e. the pairwise
multipole expansion is barely converging at 2.9 A under a single order-independent damping
radius. `C10` is therefore reported as its own column and excluded from any total claiming to
be D3-comparable. A Tang-Toennies-style damping with order-dependent parameters, which is what
CamCASP's own downstream models use, is the right treatment and is not implemented here.

## Orientation dependence

Rotating the acceptor about the `O-O` axis at fixed `R` moves induction by 2.3% over 0-120 deg
and dispersion by 0.4%. The dispersion delta is constant to `0.0016` kcal/mol across all
rotations, as it must be for isotropic `Cn` -- confirming the scan is not accidentally
sampling orientation dependence that the isotropic model cannot represent. Exercising the
anisotropic `Cn` table against orientation is the natural next step and is not done here.

## Where the deficit actually lives: it is rank-resolved, not diffuse

Measured 2026-08-20, immediately after both prerequisites landed on the same day: the new
`ATOMIC ANISOTROPIC POLARIZABILITIES` publication, and the ISA-GRID refined `.pol` oracle from
the regenerated partition-matched run. Neither existed before, which is why this had not been
measured. Comparison is in the molecular frame, using
`devtools/camcasp_reference.l3_local_to_molecular`; the oracle's rank-1 diagonal reproduces the
reviewed `ISA_GRID_STATIC_POLARIZABILITIES` literals exactly, and our rank-1 diagonal
reproduces the "ours" column of this document's basis-matched table exactly, so both sides are
anchored to already-reviewed numbers before anything new is claimed.

Site-summed rank invariants `a_l = Tr(alpha^{ll})/(2l+1)`, static:

| `l` | ours | CamCASP ISA-GRID | ratio | deficit |
| --- | ---- | ---------------- | ----- | ------- |
| 1 | `9.324909`   | `9.607417`   | `0.9706` | **`2.94%`** |
| 2 | `21.459264`  | `31.511307`  | `0.6810` | **`31.90%`** |
| 3 | `136.969050` | `195.571120` | `0.7004` | **`29.96%`** |

**The "uniform 2.9% upstream deficit" recorded in the A8 diagnosis is the rank-1 deficit, and
there is a second, ten-times-larger deficit of about 30% sitting in ranks 2 and 3.** Per site
the ratios are `O` `0.9836 / 0.6867 / 0.7063` and each `H` `0.9314 / 0.6578 / 0.6760`.

### This explains the dispersion result quantitatively

`C6` is built only from rank-1 x rank-1, so it inherits roughly `0.97^2`. `C8` picks up
rank-1 x rank-2, roughly `0.97 x 0.68`. `C10` and `C12` are dominated by rank-2 x rank-2,
rank-2 x rank-3 and rank-3 x rank-3, roughly `0.68^2` to `0.70^2`. That ordering reproduces the
measured per-order Cn deficits against this same oracle (about 1-10% at C6 growing to
~25/36/46% at C8/C10/C12) without any further assumption, and it is the same ordering that
makes the C8 energy error four times the C6 energy error in Result 1.

So the property-space and energy-space pictures now agree on a single cause, and the target is
specific: **the rank-2 and rank-3 site-diagonal blocks, at about 30%.** Closing C6 further buys
almost nothing.

### Leading hypothesis: the missing rank 4

Two features point away from diffuse numerical error and toward a systematic cause. The deficit
is nearly the same at `l = 2` and `l = 3` (`31.9%` vs `30.0%`) and nearly the same on oxygen and
hydrogen, which a diffuse grid or convergence error would not respect. And it is essentially
absent at `l = 1`.

The most plausible mechanism is **rank truncation**. CamCASP computes non-local distributed
polarizabilities to rank 4 (`H2O_NL4_*.pol` is 25x25, ranks 0:4) and *then* localizes to L3, so
its L3 ranks 2 and 3 absorb rank-4 content during localization. Our pipeline is L3 throughout
and never has rank-4 content to fold down. That predicts a deficit concentrated in the highest
retained ranks and negligible at rank 1 — which is what is measured. It is consistent with the
independent observation that our model carries no rank 0 (charge-flow) or rank 4 term at all.

Two things keep this a hypothesis rather than a conclusion:

* An alternative reading is a **normalisation convention** on the rank-2/3 real solid
  harmonics. `sqrt(0.681) = 0.825` and `sqrt(0.700) = 0.837` are suspiciously close to each
  other, which is what a per-rank factor `c_l` would produce. Against it: the *cross-rank*
  blocks do not follow `c_1 c_2`. Oxygen's 1-2 block ratio is `0.7553` where `c_1 c_2` predicts
  `0.813`, and each hydrogen's 1-2 and 1-3 blocks come out at `1.2744` and `1.3298` — **ours is
  30% too large there**, in the opposite direction to everything else. A pure normalisation
  cannot change sign of the discrepancy, so it cannot be the whole story.
* The Frobenius-norm ratios in the per-block table mix components and signs and are a cruder
  statistic than the traces. The hydrogen cross-block inversion should be re-measured
  component-by-component before it is built on.

Neither the truncation test nor the component-level hydrogen measurement was run here. They are
the two experiments that would settle it, and they are now cheap, because both the publication
and the oracle exist.
