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
