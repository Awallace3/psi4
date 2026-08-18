# Parity Debugging Map

Companion to `plans/2026-07-31-native-camcasp-parity.md`. Records which parts of the
pipeline are now *analytically settled* versus which still carry numerical risk, so
Task 8 mismatch investigation can start at the right stage instead of bisecting blindly.

## Settled exactly (no remaining numerical risk)

Each of these was checked against the reviewed oracle and reproduces it to the stated
precision. They are not sources of parity error.

| Stage | Check | Agreement |
| ----- | ----- | --------- |
| Frequency grid (Task 2) | `omega_k = 0.5 (1-t_k)/(1+t_k)`, 10-pt Gauss–Legendre | `7.1e-15` abs |
| Spherical → local Cartesian (Task 2) | `(10, 11c, 11s) -> (z, x, y)` permutation | exact (`0.0`) |
| Local → global rotation (Task 2) | `R alpha R^T`, `det R = 1`, `R R^T = I` | exact (`0.0`) |
| Dispersion recoupling (Task 6) | closed-form binomial prefactors, see [isotropic dispersion spec](2026-08-17-isotropic-dispersion-recoupling.md) | `<= 2.5e-7` rel |
| ISA-Pol response magnitude (Task 4) | site-pair blocks summed with rank 0 translated to a common origin, against Psi4's own molecular `DIPOLE POLARIZABILITY` at the identical functional/basis/grid | `0.970` (the deliberate `25% CHF + 75% ALDA` kernel difference) |
| LW localization magnitude (Task 3) | same sum on `local[site]` rank-1 blocks, against the site-pair response it localizes | `1.0000` |

Consequence: **all parity error in the published outputs must originate in Task 5 (PFIT WSM
refinement)**. The publication, dispersion, response and localization stages cannot be the
cause. The response and localization rows above were added 2026-08-17 when the 36%
conservation deficit was localized; they are cheap to re-measure and should be the first
thing checked if a future change moves the published magnitudes.

Reusable measurement: summing every site's rank-1 block together with rank 0 times the site
position is *algebraically exact* for any partition of unity, so it recovers the molecular
dipole polarizability from any site-pair or localized model at any frequency, independently
of the ISA partition, the rank truncation and the basis. That makes it the cheapest
magnitude oracle in the pipeline and it needs no external reference. It is now asserted by
`tests/pytests/test_atomic_polarizabilities.py::test_published_atomic_sum_conserves_the_site_pair_response`.

## Reviewed-model structural facts

Confirmed from the oracle; usable as cheap invariants in tests:

- Every per-atom global Cartesian tensor is exactly symmetric (`max|M - M^T| = 0`) at every
  one of the eleven frequencies.
- Geometry places the C2 axis along `z` with the molecule in the `xz` plane
  (`O` at origin, `H` at `(+/-1.45365196, 0, -1.12168732)` bohr).
- The C2 relation is exact: `alpha_H2 = S_x alpha_H1 S_x` with `S_x = diag(-1, 1, 1)`.
  This holds at every frequency, so it is a valid gate on dynamic output too.
- The L3 spherical block is `15 x 15` covering ranks 1–3 only (no rank 0 / charge flow),
  in CamCASP order `10 11c 11s 20 21c 21s 22c 22s 30 31c 31s 32c 32s 33c 33s`.
- Static isotropic dipole polarizabilities: `O = 6.129740`, `H = 1.733558` bohr^3.

## Development-only stage oracles

The ignored `.camcasp-reference/` tree contains intermediate CamCASP/PFIT artifacts that
bracket the risky stages. Per the plan's Global Constraints these are a **development
oracle only**: production code and pytest must never read them, and only fixed reviewed
literals may be checked in.

| Artifact | Brackets | Use |
| -------- | -------- | --- |
| `work/H2O/OUT/*_NL4_fmtB.pol` | output of Task 4 | nonlocal NL4 response before localization |
| `work/H2O/OUT/*_f11.p2p` | inside Task 4 | point-to-point response grid |
| `work/H2O/H2O_ref_wt4_L3_0f10.pol` | output of Task 5 | refined L3 model, i.e. the direct input to Task 6 |
| `work/H2O/H2O.pdef` | Task 5 input | site/constraint definitions |
| `work/H2O/OUT/H2O_A.out` | Task 4 prerequisite | ground-state/GRAC provenance |

Recommended investigation order for any Task 8 mismatch, given the settled rows above:

1. Compare the native refined L3 model against `refined_pol`. If it matches, the mismatch
   is a publication/packing bug, which the settled rows make unlikely.
2. If it does not, compare the native localized response against `nonlocal_pol` pushed
   through LW localization, isolating Task 5 from Tasks 3–4.
3. If that also disagrees, compare the native site-pair response against `nonlocal_pol`
   directly, isolating Task 4.

Do not loosen the `rtol=1e-4, atol=1e-5` gate at any step.

## Charge flow is not the explanation for a conservation deficit

Measured 2026-08-17 from the reviewed nonlocal NL4 oracle, which is `RANK 0 : 4` and so
carries the rank-0 charge-flow blocks explicitly. Reconstructing the molecular dipole
polarizability from that model, including and excluding rank 0:

| | `xx` | `yy` | `zz` | isotropic |
| - | ---- | ---- | ---- | --------- |
| with rank-0 charge flow | `10.1499` | `9.0052` | `9.6003` | `9.5852` |
| rank-0 discarded | `9.1793` | `9.0052` | `9.4334` | `9.2060` |
| ratio | `0.904` | `1.000` | `0.983` | `0.960` |

Two consequences:

- Charge flow is worth only about **4% isotropically**. Discarding it entirely still leaves
  96% of the molecular polarizability, so it cannot explain a large deficit.
- Charge flow contributes **exactly nothing out of plane** (`yy` ratio `1.000`); it only
  moves the in-plane components. Any deficit that is *worst* out of plane therefore has a
  different mechanism.

Also useful as targets: the reviewed *nonlocal* model reconstructs to isotropic `9.5852`, and
the reviewed *refined* L3 model sums to `(10.191, 8.997, 9.603)`, isotropic `9.597`. Both
reviewed stages conserve, and the refinement absorbs charge flow essentially losslessly —
which is the behaviour a correct WSM implementation must reproduce.

## The conservation deficit is frequency-dependent

Measured 2026-08-17: our published isotropic atomic sum against the reviewed one at all
eleven frequencies. The bases differ (aug-cc-pVDZ vs the reviewed aug-cc-pVTZ), so the
absolute ratio carries basis error — the *shape* is the signal.

| `omega` | ours | reviewed | ratio |
| ------- | ---- | -------- | ----- |
| `0.000000` | `6.03357` | `9.59686` | `0.629` |
| `0.095447` | `5.91086` | `9.29983` | `0.636` |
| `0.370417` | `4.78189` | `6.91797` | `0.691` |
| `1.264899` | `1.98139` | `2.38764` | `0.830` |
| `6.910886` | `0.14174` | `0.15490` | `0.915` |
| `37.823762` | `0.00533` | `0.00612` | `0.871` |

This separates the defect into two independent effects:

1. **A frequency-dependent part, about 37% at `omega = 0`, closing as `omega` grows.** As
   `omega` rises the bare response `chi0` shrinks and the kernel correction to
   `alpha = chi0 (1 - K chi0)^-1` matters less. A deficit that vanishes in exactly that limit
   is the signature of a **kernel that screens too strongly**. Check the Hartree term for
   double counting, the sign and prefactor on the ALDA `fxc`, and whether the 25% exchange
   fraction is applied twice (once via the superfunctional, again in the blend).
2. **A residual frequency-independent part of roughly 10%,** visible where kernel effects
   have died away (the plateau near `0.90`, not `1.00`). This cannot be the kernel and points
   at the uncoupled response or transition-multipole normalization. Treat it as a second,
   independent bug and re-measure the plateau after fixing the kernel.

Basis differences do not produce a monotone `0.63 -> 0.90` ramp in `omega`, so the trend is
not a basis artifact. For a basis-clean single point, compare the `omega = 0` sum against
Psi4's own `DIPOLE POLARIZABILITY` at the same basis and grid (`9.3595` at PBE0/aug-cc-pVDZ,
DFT `590/99`).

## The reviewed PFIT anchor values are a direct Task 3 oracle

Discovered 2026-08-17. The reviewed PFIT log tabulates, for every independent variable,
both the fitted value and its **anchor** — and the anchor *is* the LW-localized value that
`refine_wsm` receives as `LocalizedResponse.local`. That makes it a precise, per-component
oracle for the output of Task 3, without needing to parse the localized model itself.

Only seven parameters carry nonzero penalty, which settles the penalty scope as a matter of
fact rather than judgement: the reviewed protocol anchors the **whole rank-1 dipole block**,
not just its diagonal.

| parameter | reviewed anchor | meaning (local frame) |
| --------- | --------------- | --------------------- |
| `O_10_10` | `5.58320` | `alpha_zz` |
| `O_11c_11c` | `7.03535` | `alpha_xx` |
| `O_11s_11s` | `5.76374` | `alpha_yy` |
| `H1_10_10` | `2.00865` | `alpha_zz` |
| `H1_10_11c` | `-0.00576` | `alpha_zx` |
| `H1_11c_11c` | `1.55739` | `alpha_xx` |
| `H1_11s_11s` | `1.62075` | `alpha_yy` |

On a site whose only symmetry is a mirror plane the dipole off-diagonal is symmetry-allowed,
and it is the component the point response constrains least — so it must be anchored or it
drifts. Anchoring only the diagonal let it reach `+4.29` against a reviewed `+0.0058` while
still fitting the response and conserving the molecular sum, and left the published hydrogen
dipole block indefinite. Fixed by anchoring the full rank-1 block.

### Remaining discrepancy: our localized values differ from the reviewed anchors

Because the penalty holds the dipole block near its anchor, our final answer inherits any
error in our LW-localized model. Measured at PBE0/aug-cc-pVDZ against the reviewed
aug-cc-pVTZ anchors above (so some basis error is expected, but not this much):

| | `alpha_xx` | `alpha_yy` | `alpha_zz` |
| - | ---------- | ---------- | ---------- |
| `O` ours | `6.692` | `6.854` | `6.494` |
| `O` reviewed | `7.035` | `5.764` | `5.583` |
| `H` ours | `1.583` | `0.655` | `1.150` |
| `H` reviewed | `1.557` | `1.621` | `2.009` |

The totals nearly agree (conservation is `0.955`), so this is a **misdistribution**, not a
magnitude error: our oxygen comes out far too isotropic and absorbs out-of-plane response
that the reviewed model assigns to the hydrogens. Note `alpha_xx` agrees well on both sites
while `yy` and `zz` are badly split — a directional signature, not a uniform scale.

This was bisected on 2026-08-18 and **the localization is not the cause**. Feeding the
reviewed nonlocal `H2O_NL4_000.pol` through our own `localize_lw` — truncated to rank 3 exactly
as ORIENT's `Limit all rank 3` does — reproduces the reviewed `H2O_L3_000.pol` to `7.4e-13` on
every one of the 675 tensor entries, for all three sites. `H1` requires the documented
local-frame rotation (180 degrees about `z`, so `alpha_(t,u) -> (-1)^(m_t + m_u) alpha_(t,u)`),
which takes it from `1.5472e+01` to `5.4623e-13`. That test is hermetic: no SCF, basis, grid or
partition enters it, so it is an unambiguous verdict on Task 3 alone.

The cause is instead a **partition-scheme mismatch**, not a bug. `work/H2O/OUT/H2O.out` line
411 states the reviewed algorithm as `ALGORITHM: DF : density-fitting-based partitioning of the
FDDS`, and `H2O.cks` selects `C-DF` over a 246-function auxiliary basis; the reviewed control
file has no ISA directive at all. Our Task 4 partitions by real-space stockholder weights. Two
different distributions of the same molecular response agree on the total and disagree on the
split — which is exactly the observed signature. See
[the ISA partition spec](2026-08-18-isa-partition-oeprop.md) for the resolution, including the
one-directive `DIST-ALG ISA-GRID` route to a matching oracle.
