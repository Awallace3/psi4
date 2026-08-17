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

Consequence: **all parity error in the published outputs must originate upstream of the
L3 refined model** — i.e. in Task 4 (ISA-Pol response) or Task 5 (PFIT WSM refinement),
or in Task 3 (LW localization) between them. The publication and dispersion math cannot
be the cause.

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
