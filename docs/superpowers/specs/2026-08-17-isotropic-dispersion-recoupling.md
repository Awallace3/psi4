# Isotropic L3 Dispersion Recoupling Specification

Independent specification for Task 6 of `plans/2026-07-31-native-camcasp-parity.md`.
Derived from published distributed-dispersion theory (Stone, *Theory of Intermolecular
Forces*, §4.3; Casimir–Polder frequency integration). No CASIMIR/ORIENT source was read
or transcribed while writing this document.

## Scope

This specifies the **isotropic `00 00 0` component only**, which is the sole component
published in `ATOMIC C6`, `ATOMIC C8`, `ATOMIC C10`, and `ATOMIC C12`.

Consequence that simplifies the implementation: the isotropic component depends only on
the **trace of each diagonal rank block** of the L3 polarizability. Off-diagonal rank
blocks (`l != l'`) and all anisotropic components drop out of the `00 00 0` term. A real
Clebsch–Gordan contraction table is therefore **not** required for the published
coefficients. Any rank-mixing table remains a prerequisite only if anisotropic components
are added later.

## Inputs

Per atom `A` and per frequency `i`, the refined L3 model supplies the real-spherical
polarizability matrix in CamCASP component order:

```text
10 11c 11s | 20 21c 21s 22c 22s | 30 31c 31s 32c 32s 33c 33s
```

Row/column blocks: rank 1 = indices `[0,3)`, rank 2 = `[3,8)`, rank 3 = `[8,15)`.

Define the isotropic rank-`l` polarizability

```text
alpha_bar_l^A(i omega) = Tr( alpha^{ll}_A(i omega) ) / (2l + 1)
```

## Frequency quadrature

Ten-point Gauss–Legendre on the Casimir–Polder half line, mapped with base frequency
`omega0 = 0.5 a.u.`:

```text
omega_k = omega0 * (1 - t_k) / (1 + t_k)
w_k     = wgl_k * 2 * omega0 / (1 + t_k)^2
```

where `(t_k, wgl_k)` are the standard 10-point Gauss–Legendre nodes/weights on `[-1, 1]`.

Verified: this reproduces the eleven reviewed grid frequencies (the static point plus the
ten mapped nodes, ascending) to `7.1e-15` absolute. The static `omega = 0` point carries
**no** quadrature weight and must be excluded from the dispersion sum; it is published
only in `ATOMIC POLARIZABILITY FREQUENCIES` and the static tensor.

## Recoupling formula

For an ordered rank pair `(la, lb)`, the contribution to `C_n` with
`n = 2 * (la + lb + 1)` is

```text
K(la, lb) = binom(2*la + 2*lb, 2*la) / (2*pi)

C_n[A][B] = sum over permitted (la, lb) of
              K(la, lb) * sum_k w_k * alpha_bar_la^A(i omega_k) * alpha_bar_lb^B(i omega_k)
```

Permitted ordered rank pairs under the L3 model (`la, lb` in `1..3`):

| coefficient | ordered pairs        | prefactors `K`                          |
| ----------- | -------------------- | --------------------------------------- |
| `C6`        | `(1,1)`              | `6/(2pi)`                               |
| `C8`        | `(1,2)`, `(2,1)`     | `15/(2pi)` each                         |
| `C10`       | `(1,3)`, `(3,1)`, `(2,2)` | `28/(2pi)`, `28/(2pi)`, `70/(2pi)` |
| `C12`       | `(2,3)`, `(3,2)`     | `210/(2pi)` each                        |

`(1,4)` and `(4,1)` also satisfy `n = 12` but are excluded: the model is L3, so rank 4 is
absent. This is the documented reason `C12` is reviewed-model parity rather than a
rank-complete physical coefficient (plan Global Constraints). The implementation must
reject a model missing any of ranks 1–3 rather than manufacturing higher coefficients
from dipole terms.

## Validation status

Evaluated against the reviewed oracle L3 models and CASIMIR coefficients. Agreement on
every distinct matrix entry, relative deviation:

| coefficient | `O-O`     | `O-H`     | `H-H`     | max rel. dev. |
| ----------- | --------- | --------- | --------- | ------------- |
| `C6`        | 17.25559  | 5.382332  | 1.698678  | `2.5e-7`      |
| `C8`        | 346.424   | 83.90759  | 18.32833  | `2.0e-7`      |
| `C10`       | 7484.441  | 1523.525  | 291.4843  | `1.1e-7`      |
| `C12`       | 127231.0  | 20293.77  | 3216.541  | `2.0e-7`      |

Residual deviation is consistent with the six/seven-significant-figure rounding of the
reviewed literals, not with a formula error. This is comfortably inside the plan's
`rtol=1e-4, atol=1e-5` gate.

## Required properties

An implementation of `compute_dispersion` must satisfy:

- **Pair symmetry.** `C_n[A][B] == C_n[B][A]`, exactly, for every `n`. Note this holds
  even though individual `(la, lb)` terms are not symmetric: the permitted pair sets are
  closed under exchange, and `K(la, lb) == K(lb, la)` for `la + lb` fixed only when
  `la == lb`, so symmetry follows from summing both orderings, not from a single term.
- **Analytic C6 check.** Two sites with constant `alpha_bar_1 = a` across the grid and no
  higher ranks give `C6 = (3/pi) * a^2 * sum_k w_k`.
- **Quadrature convergence.** Refining the node count leaves `C6` stable for a smooth
  model.
- **Rank completeness.** Missing rank 2 or rank 3 is an error, not a zero contribution.
- **Frequency-grid agreement.** The grid handed in must match `make_casimir_grid`;
  mismatched length or values is an error rather than a silent truncation.
