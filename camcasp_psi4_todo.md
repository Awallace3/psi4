# CamCASP Parity in Psi4 — State of the Branch and Outstanding Work

Branch `camcasp` against `master`. Written 2026-08-18.

This document records what the branch implements, the equations it implements, how well each
published property agrees with CamCASP, and every route that does **not** currently have full
Psi4 support. It is a status-and-TODO record, not a plan; the executable plan is
[`docs/superpowers/plans/2026-07-31-native-camcasp-parity.md`](docs/superpowers/plans/2026-07-31-native-camcasp-parity.md).

**Headline.** The pipeline is at parity on the frequency grid, the LW localization, and the
dispersion recoupling; it is *accepted with a measured band* on the dipole polarizability
block and C6; and it is **not** at parity on C8/C10/C12, which run 20–46 % low against the
matching oracle. The dipole-block residual was proven to be the *partition scheme*, not a
defect. The higher-rank residual is a genuine open defect and is the leading parity gap.

---

## Contents

1. [Scope of the branch](#1-scope-of-the-branch)
2. [Architecture](#2-architecture)
3. [Equations, stage by stage](#3-equations-stage-by-stage)
4. [Agreement with CamCASP](#4-agreement-with-camcasp)
5. [Routes without full Psi4 support](#5-routes-without-full-psi4-support)
6. [Verified test status](#6-verified-test-status)
7. [Prioritised TODO](#7-prioritised-todo)

---

## 1. Scope of the branch

`master..camcasp` is **≈33,000 added lines across 65 files**, all CamCASP-related
(32,998 insertions, 134 deletions). Three layers:

| Layer | Files | ~Lines |
| ----- | ----- | ------ |
| **Native pipeline (production)** | `libmints/atomic_polarizability.{h,cc}` (7,328), `libmints/isa_weights.cc` (1,119), `libmints/wsm_fit_points.cc` (496), `libmints/oeprop.{h,cc}`, `src/read_options.cc`, `src/export_oeprop.cc` (1,622), `driver/procrouting/atomic_polarizability.py` (218) | ~9,900 |
| **Psi4 core enablement** — narrow seams the pipeline needs | `libscf_solver/{hf,rhf,uhf,rohf,cuhf}.{cc,h}` response-provenance seal; `libfock/{cubature,jk,v}.h`, `libfock/DirectJK.cc`; `libfunctional/{LibXCfunctional,superfunctional,functional}`; `libmints/{basisset,integral,twobody,molecule}`; `libqt` (`C_DGESVD`) | ~800 |
| **Dev-only oracle tooling and tests** | `devtools/camcasp_reference.py` (2,799), `devtools/regenerate-camcasp.sh` (1,518), 14 pytest modules (~12,200), 9 spec/plan documents (4,662) | ~22,300 |

**Clean-room discipline.** Production code and pytest never invoke, clone, access or read
CamCASP, ORIENT, PFIT, CASIMIR or `.camcasp-reference/`. Those paths are gitignored
(`.camcasp-reference/`, `orient/`, `camcasp-bin/`), the reference values in pytest are
hard-coded literals, and two source-guard modules
(`test_native_atomic_polarizability_source_guard.py`,
`test_response_provenance_source_guard.py`) assert the absence of any runtime dependency.
No ORIENT GPLv3 source, comments, structure or control flow was copied.

### 1.1 Public API surface added

Seven wavefunction array variables, published atomically:

| Variable | Shape | Content |
| -------- | ----- | ------- |
| `ATOMIC POLARIZABILITIES` | `(natom, 6)` | static, global Cartesian, packed `xx, xy, xz, yy, yz, zz` |
| `ATOMIC DYNAMIC POLARIZABILITIES` | `(11·natom, 6)` | frequency-major atom blocks |
| `ATOMIC POLARIZABILITY FREQUENCIES` | `(11, 1)` | static point plus ten mapped Gauss–Legendre nodes |
| `ATOMIC C6` / `C8` / `C10` / `C12` | `(natom, natom)` | isotropic `00 00 0`, in `hartree·bohr^n` |

Entry point: `psi4.driver.procrouting.atomic_polarizability.atomic_polarizabilities()`.
`OEProp` remains the publication point; `ATOMIC_POLARIZABILITIES` was added to
`core.OEProp.valid_methods`.

Fifteen new `ATOMIC_POLARIZABILITY_*` keywords in `read_options.cc` cover the frequency grid,
fit-point generation, the ISA quadrature, the localization tolerance, the condition-number
ceiling and the covalent-bond scale.

---

## 2. Architecture

### 2.1 Full pipeline — leaves are the computed properties

```
psi4.driver.procrouting.atomic_polarizability.atomic_polarizabilities(molecule, "pbe0")
│
├─[SCF triple, run by the driver]─────────────────────────────────────────┐
│   RKS neutral precursor ──► epsilon_HOMO, E_0                           │  GRAC shift
│   UKS cation (vertical) ──► E_+                                         │  D = E_+ - E_0
│   RKS + DFT_GRAC_SHIFT  ──► reference wavefunction                      │      + eps_HOMO
└──────────────────────────────────────────────────────────────────────────┘
        │  each SCF seals a ResponseSCFProvenance: C, epsilon, occ, D,
        │  functional state, basis structural snapshot, DFT grid + blocks
        ▼
  FrozenResponseContext::create(grac, precursor, cation)        [fail-closed seal]
        │
        ├───────────────────────────────► compute_isa_weights      (Task 4a)
        │                                   │  real-space ISA fixed point
        │                                   │  w_A(r), populations N_A
        ▼                                   │
  Response primitives (restricted singlet, C1)
   ├ construct_restricted_c1_primitives   J, K_direct, K_transpose  (DirectJK)
   ├ construct_restricted_alda_kernel     full XC f_xc on the frozen grid
   └ assemble_restricted_singlet_hessian  ──►  H1 = A + B ,  H2 = A - B
        │
        ▼
  make_casimir_grid(10, 0.5)  ──►  omega = {0, w_1 ... w_10}
        │
        ├───────────────────────────────────────────► ( ) ATOMIC POLARIZABILITY
        │                                                 FREQUENCIES   (11,1)
        ▼
  solve_dense_restricted_response(H1, H2, omega, rhs)  ──►  G(i omega)
        │
        ├─► project_transition_multipoles     B[A,t,(ia)]     ISA-weighted   (Task 4b)
        └─► contract_site_pair_response       alpha_AB(t,u)   16x16/pair, ranks 0-3
                │
                ├─► derive_bond_graph(molecule)     Bragg-Slater x 1.3, must be connected
                ▼
          localize_lw(alpha_AB, graph, tol)   ──►  alpha_A(t,u)  15x15, ranks 1-3  (Task 3)
                │
                ├─► generate_wsm_fit_points(molecule)   nested Lebedev, exact O_h
                ├─► evaluate_point_response             Pi_obs(g,h; i omega)
                ├─► derive_pdef_constraints(molecule)   active-variable symmetry mask
                ▼
          refine_wsm(...)  ──►  RefinedL3Model per frequency   (Task 5, PFIT-style WSM)
                │
                ├──► pack + rotate ──► ( ) ATOMIC POLARIZABILITIES           (3,6)
                │                      ( ) ATOMIC DYNAMIC POLARIZABILITIES   (33,6)
                │
                └──► compute_dispersion(models, grid)                  (Task 6)
                            ├──► ( ) ATOMIC C6    (3,3)
                            ├──► ( ) ATOMIC C8    (3,3)
                            ├──► ( ) ATOMIC C10   (3,3)
                            └──► ( ) ATOMIC C12   (3,3)
```

`AtomicPolarizabilityCalculator::run()` either returns all seven arrays or throws
`AtomicPolarizabilityPrerequisiteError`; `compute()` publishes only after `run()` returns.
No partial output is ever visible.

### 2.2 Where the two partition schemes diverge

Only one box changes between our route and the reviewed CamCASP route. Everything downstream
of the site-pair response is shared.

```
                     G(i omega)   [identical: same wavefunction, same kernel]
                          |
        +-----------------+------------------+
        |                                    |
   ISA route (ours)                    C-DF route (CamCASP reviewed)
   B[A,t,ia] = sum_p w_p               B[A,t,ia] = sum_{k in A} Q_t[k] d^ia_k
               * w_A(r_p)                        (constrained density fitting onto
               * R_t(r_p - R_A)                   atom-centred auxiliary functions)
               * phi_i phi_a
        |                                    |
        +-----------------+------------------+
                          |
              alpha_AB(t,u) = 4 B_A G B_B^T     [identical algebra]
                          |
              LW -> WSM -> Casimir-Polder       [identical]
```

This is why the two oracles agree on the molecular total to `0.971` and disagree on the
per-site split by up to a factor of 113. It is also why implementing C-DF (§5, item 1) is a
much smaller change than it looks — see the companion spec.

---

## 3. Equations, stage by stage

Notation: `i,j` occupied, `a,b` virtual, `A,B` sites, `t,u` real-spherical component indices
in CamCASP order `00; 10 11c 11s; 20 21c 21s 22c 22s; 30 31c 31s 32c 32s 33c 33s`.

### 3.1 GRAC asymptotic correction (driver)

```
Delta = E(cation, vertical) - E(neutral) + epsilon_HOMO(neutral)
```

Three SCFs at one geometry in one basis. `DFT_GRAC_SHIFT` is restored to its incoming value
before returning, so a caller's option state is not silently mutated.

### 3.2 Frequency grid — `make_casimir_grid(10, 0.5)`

Ten-point Gauss–Legendre on the Casimir–Polder half line, base frequency `omega_0 = 0.5` a.u.:

```
omega_k = omega_0 (1 - t_k) / (1 + t_k)
w_k     = wgl_k * 2 omega_0 / (1 + t_k)^2
```

where `(t_k, wgl_k)` are the standard 10-point Gauss–Legendre nodes and weights on `[-1, 1]`.
The static `omega = 0` point carries **no** quadrature weight and is excluded from every
dispersion sum; it is published only in the frequency array and the static tensor.

### 3.3 Response kernel and Hessians — `assemble_restricted_singlet_hessian`

The reviewed kernel is fixed at **`a = 0.25` CHF exchange plus `b = 0.75` ALDA**, independent
of the ground-state functional (which is PBE0):

```
H1[ia,jb] = delta_{ia,jb} * dEps_ia + 4 (ia|jb) - a [ (ij|ab) + (aj|bi) ] + 4 b f_xc[ia,jb]
H2[ia,jb] = delta_{ia,jb} * dEps_ia               - a   (ij|ab) + a (aj|bi)

f_xc[ia,jb] = integral  w(r) f_xc(rho(r)) phi_i(r) phi_a(r) phi_j(r) phi_b(r)  dr
```

with `H1 = A + B` and `H2 = A - B`. Primitive indexing follows
`driver/procrouting/response/scf_products.py`: `J[ia,jb] = (ia|jb)`,
`K_direct[ia,jb] = (ij|ab)`, `K_transpose[ia,jb] = (aj|bi)` (the alternate native exchange
contraction, *not* the matrix transpose of `K_direct`). Every primitive is independently
required to be finite and symmetric; none is silently symmetrised.

The ALDA normalisation is derived directly from `RV::compute_Vx_full` rather than assumed:
for the nonsymmetric transition density `D_ia = C_i C_a^T`, `compute_Vx_full` forms
`rho_k = 0.5 phi (D_ia + D_ia^T) phi = phi_i phi_a`; its LDA intermediate carries a `0.5`
multiplier and the explicit adjoint doubles it, so the projected primitive is exactly the
integral above with no internal spin factor. The `4b` in `H1` supplies the rest.

### 3.4 Dense response solve — `solve_dense_restricted_response`

Native convention, positive right-hand side:

```
[  H1     omega*I ] [ P ]   [ rhs ]
[ -omega*I   H2   ] [ Q ] = [  0  ]
```

At exactly zero frequency only `H1 P = rhs` is solved and `Q` is identically zero. Solved by
LAPACK `DGESVX` with `FACT='N'` (no equilibration), under a hard scientific-quality budget:

| diagnostic | gate |
| ---------- | ---- |
| `RCOND` | `>= 1e-12` |
| reciprocal pivot growth | `>= 1e-12` |
| `FERR` (forward error) | `<= 1e-8` |
| `BERR` (backward error) | `<= 1e-11` |
| independently recomputed scale-aware residual | `<= 1e-11` |

The forward-error limit leaves four decimal orders beneath the downstream `1e-4` parity gate;
the backward/residual limits leave seven.

### 3.5 ISA partition — `isa_weights.cc`

Real-space iterated stockholder solved as a fixed point on the frozen AO density. No MBIS,
nearest-centre or uniform fallback exists.

```
w_A(r)          = rho0_A(|r - R_A|) / sum_B rho0_B(|r - R_B|)

rho0_A^(n+1)(r) = < rho(R_A + r u) * w_A^(n)(R_A + r u) >_u        (spherical average over u)
```

Implementation details that matter numerically:

- **log-sum-exp promolecule.** Shape functions are stored and interpolated as `log w_A(r)`
  with a floor at `log(DBL_MIN) + 32`, and the denominator is formed as
  `exp(log_A - max) / sum_B exp(log_B - max)` so no site underflows to a hard zero.
- **PCHIP** monotone cubic interpolation of the log profile between radial nodes.
- **Exponential tail.** Beyond a fitted join radius `r_0`, `log w_A(r) = logAmp - alpha r`,
  with `alpha` solved so the tail carries the remaining charge. Overlap integrals over the
  tail are then analytic:
  `4 pi exp(logAmp - beta r_0) [ r_0^2/beta + 2 r_0/beta^2 + 2/beta^3 ]`.
- **Kahan summation** on every angular average and every charge integral.
- Radial quadrature is Gauss–Legendre mapped to the half line; angular is a product
  polar × azimuthal grid.

Convergence is measured by the normalised overlap of successive shape functions, to
`ATOMIC_POLARIZABILITY_ISA_CONVERGENCE` (default `1e-9`). Partition-of-unity
(`sum_A w_A(r) = 1`, pointwise) and population conservation (`sum_A N_A = N`) are enforced,
not assumed.

### 3.6 ISA-weighted transition multipole projection and site-pair response

```
B[A,t,(ia)]           = sum_p  w_p * w_A(r_p) * R_t(r_p - R_A) * phi_i(r_p) phi_a(r_p)

alpha_AB(t,u; i omega) = 4 * B[A,t,:] * G(i omega) * B[B,u,:]^T
```

`R_t` is the **regular** real solid harmonic. `B` is stored site-major with 16 components
(ranks 0–3) per site. Reciprocity `alpha_AB(t,u) = alpha_BA(u,t)` is verified against a bound
derived from the solver's own maximum `FERR`, then enforced; it is never assumed.

**Magnitude invariant.** Summing every site's rank-1 block together with rank 0 translated to
a common origin is *algebraically exact* for any partition of unity, so it recovers the
molecular dipole polarizability from any site-pair or localized model at any frequency,
independently of the partition, the rank truncation and the basis. This is the cheapest
oracle in the pipeline and needs no external reference.

### 3.7 LW localization — `localize_lw`

Lillestølen–Wheatley bond-flow localization on the graph Laplacian. For a bond graph with
`L_ab = 1` on each bond and `L_aa = -deg(a)`:

```
L phi   = u            u = balanced off-site response for one component pair, sum(u) = 0
flow_ab = 0.5 * ( phi_b - phi_a )
```

`L` is singular (constant null mode per connected component), so `phi` is obtained from the
Moore–Penrose pseudoinverse built per component from a verified symmetric eigendecomposition;
the range residual `|L phi - u|` is gated against the caller's tolerance.

Each flow is applied through the real-spherical **multipole translation operator** `T(R)`, so
moving response along a bond redistributes it correctly across ranks 0–3 rather than only at
the transferred component. Five residuals are gated before the result is returned: off-site,
charge sum, reciprocity, molecular sum, local charge.

Output is a `15 x 15` per-site model spanning ranks 1–3 only. **There is no rank 0**, i.e. no
charge-flow term, in the published L3 model.

### 3.8 WSM refinement — `refine_wsm` (PFIT-style constrained fit)

Model of the point-to-point response using **irregular** solid harmonics
`T_t(r) = R_t(r) / r^(2l+1)`:

```
Pi_model(g,h) = sum_A sum_{t <= u} alpha_A(t,u)
                  [ T_t(g-A) T_u(h-A) + (1 - delta_tu) T_u(g-A) T_t(h-A) ]

Pi_obs(g,h; omega) = 4 sum_ia v(g,ia) P(ia,h)
```

solved as a constrained, anchored, weighted least-squares problem:

```
min || W (A x - b) ||^2  +  lambda || D (x - x0) ||^2      subject to   C x = d
```

| symbol | value / meaning |
| ------ | --------------- |
| `W` | row weights: `1` on diagonal point pairs, `sqrt(2)` on off-diagonal pairs |
| `lambda` | `0.001` (the reviewed PFIT weight coefficient) |
| `D` | `1` on the **whole rank-1 (dipole–dipole) block**, `0` elsewhere |
| `x0` | the LW-localized model — the anchor reference |
| `C, d` | PDef symmetry mask plus the `alpha_H2 = S_x alpha_H1 S_x` equality copy |
| column cutoff | `1e-4`, **relative** to the largest weighted column norm |
| condition ceiling | `1e12` |

No normal equations are formed: equality elimination and the reduced fit both use direct
SVDs (`C_DGESVD`).

**Two corrections recorded in the code, both of which were real defects:**

1. *The cutoff is relative, not absolute.* Irregular harmonics fall off as `r^-(2l+1)`, so
   every design column norm shrinks as the fit points move outward. Under an absolute reading
   the retained rank becomes a function of the shell radii, and at the reviewed protocol's own
   grid (4.63–11.46 bohr) the rank-3 columns are pruned and the constraint elimination then
   fails closed with "constraints are ambiguous (linearly dependent)" — i.e. the absolute
   reading *cannot express the reviewed protocol at all*. On the corrected default grid the
   smallest rank-3 column is `2.36e-05` absolute (pruned) but `2.78e-04` relative (retained).

2. *The anchor covers the whole rank-1 block, not just its diagonal.* On a site with
   mirror-only symmetry the allowed dipole off-diagonal is exactly the Cartesian component the
   point response constrains least, so leaving it free lets it drift far from any physical
   value while still fitting the response and conserving the molecular sum. Unanchored, the
   hydrogen `alpha_xz` came out at `+4.29`.

### 3.9 PDef active-variable constraints — `derive_pdef_constraints`

A component pair `(t, u)`, `t <= u`, is an **active fit variable** if and only if the real
spherical harmonics `t` and `u` transform as the **same irreducible representation** of the
site's local point group. Everything else is frozen at zero by *omitting the column from the
design matrix* — not by fitting and then zeroing, which would change the residual and defeat
the conditioning benefit.

Characters are computed exactly, with integer signs and no floating-point tolerance test:

| operation | action | character |
| --------- | ------ | --------- |
| `C2(z)` | `phi -> phi + pi`, so `cos(m phi)` and `sin(m phi)` both pick up `(-1)^m` | `(-1)^m` |
| `sigma(xz)` | `y -> -y`, so `sin(m phi) -> -sin(m phi)` | `-1` for `s`-type, `+1` for `0`/`c`-type |

For H2O with the C2 axis along `z` and the molecule in the `xz` plane:

| site | group | classes | active pairs |
| ---- | ----- | ------- | ------------ |
| `O` | `C2v` | `A1={10,20,22c,30,32c}`, `B1={11c,21c,31c,33c}`, `B2={11s,21s,31s,33s}`, `A2={22s,32s}` | `15+10+10+3` = **38** |
| `H1` | `Cs` | `A'` (9 components, 45 pairs), `A''` (6 components, 21 pairs) | **66** |
| `H2` | — | symmetry copy of `H1`, imposed as an equality constraint | **0 independent** |

Total: **170 active, 66 equality rows, 104 independent variables**, down from 360.

This is what forces `alpha_yz = alpha_xy = 0` exactly on the hydrogens — the `+/-5.34` element
the unconstrained fit was inventing, against diagonals of `~1.17`/`~0.50`, giving eigenvalues
`[-4.51, -0.011, +6.19]`.

The mask is derived **geometrically**, not from the declared point group: the reviewed
protocol runs `symmetry c1`, `no_com`, `no_reorient`, and `derive_pdef_constraints` still
detects `C2v(Z)` and returns the same 170/104. Keying it off the declared group would silently
disable all constraints, so this is pinned by a test.

### 3.10 WSM fit points — `generate_wsm_fit_points`

Union over shells and atoms of Lebedev-sampled surfaces, keeping only nodes no closer to any
other nucleus, then merging coincident points. Defaults: 50 Lebedev nodes per atom per shell,
5 nested shells spanning **4.5 to 11.5 bohr**, giving **329 points**.

Every symmetry operation must be orthogonal, must map the nuclear framework onto itself, and
must be a signed coordinate permutation in the angular frame (an element of `O_h` there).
Those three conditions make the node set exactly invariant, which is then verified as a
postcondition — measured symmetry deviation exactly `0.0`. An arbitrary point set would inject
symmetry-violating residuals directly into the fitted anisotropy, so anything else fails
closed rather than fitting.

**Why the radii matter (this cost 36 % of the molecular polarizability once).** Fit points
must lie *outside* the molecular charge density; a rank-3 distributed multipole model cannot
represent the point-to-point response where the point penetrates the density:

| nearest-nucleus distance | `Pi_obs` | `Pi_model(LW)` | ratio |
| ------------------------ | -------- | -------------- | ----- |
| `2.0` bohr | `2.504e-01` | `1.406e+00` | **`5.61`** |
| `3.0` bohr | `4.614e-02` | `5.332e-02` | `1.16` |
| `4.0` bohr | `1.606e-02` | `1.627e-02` | `1.013` |
| `6.0` bohr | `3.543e-03` | `3.549e-03` | `1.002` |
| `8.0` bohr | `1.328e-03` | `1.334e-03` | `1.004` |

Charge penetration damps the true response by a factor of `5.6` at 2 bohr. Least squares
against data the model provably cannot fit (461 % relative model error) drives the fitted
polarizabilities down. The reviewed CamCASP point grid places its 500 points 4.63–11.46 bohr
from the nearest nucleus (mean `8.48`); the `4.5`–`11.5` default brackets it.

### 3.11 Dispersion recoupling — `compute_dispersion`

Published output is the isotropic `00 00 0` component only, which depends **solely on the
trace of each diagonal rank block**. Off-diagonal rank blocks (`l != l'`) and all anisotropic
components drop out, so no real Clebsch–Gordan table is required for the current outputs.

```
alphabar_l^A(i omega) = Tr( alpha^{ll}_A(i omega) ) / (2l + 1)

K(la, lb)  = binom(2 la + 2 lb, 2 la) / (2 pi)

C_n[A][B]  = sum over permitted ordered (la, lb) of
                K(la, lb) * sum_k w_k * alphabar_la^A(i omega_k) * alphabar_lb^B(i omega_k)

with n = 2 (la + lb + 1)
```

| coefficient | ordered pairs | prefactors `K` |
| ----------- | ------------- | -------------- |
| `C6` | `(1,1)` | `6/(2 pi)` |
| `C8` | `(1,2)`, `(2,1)` | `15/(2 pi)` each |
| `C10` | `(1,3)`, `(3,1)`, `(2,2)` | `28/(2 pi)`, `28/(2 pi)`, `70/(2 pi)` |
| `C12` | `(2,3)`, `(3,2)` | `210/(2 pi)` each |

`(1,4)` and `(4,1)` also satisfy `n = 12` but are **excluded**: the model is L3, so rank 4 is
absent. This is the documented reason C12 is *reviewed-model parity* rather than a
rank-complete physical coefficient. The implementation rejects a model missing any of ranks
1–3 rather than manufacturing higher coefficients from dipole terms.

Pair symmetry `C_n[A][B] == C_n[B][A]` is exact, and non-trivially so: individual `(la, lb)`
terms are not symmetric, but the permitted pair sets are closed under exchange, so symmetry
follows from summing both orderings. The isotropic product is explicitly parenthesised so that
exchanging the ordered pair reproduces the same value bit for bit.

Analytic check built into the tests: two sites with constant `alphabar_1 = a` across the grid
and no higher ranks give `C6 = (3/pi) a^2 sum_k w_k`.

### 3.12 Frame and packing

```
(10, 11c, 11s)  ->  (z, x, y)                     real-spherical to local Cartesian
alpha_global    =  R alpha_local R^T              R R^T = I and det R = +1 enforced
packed order    =  xx, xy, xz, yy, yz, zz
```

`refine_wsm`'s harmonics are molecular-frame, so `derive_pdef_constraints` is called with
empty `site_axes` and the packing rotation is the identity — but the rotation is kept explicit
so `rotate_tensor` still enforces orthonormality and `det(R) = 1` on the frame actually used.

**Frame hazard.** If a caller supplies non-identity local axes, the PDef mask indexes variables
in *those* frames and must not be handed to `refine_wsm` unchanged. A silent mismatch produces
plausible-looking wrong anisotropy. Asserted at the call site.

---

## 4. Agreement with CamCASP

### 4.1 The two-oracle problem

There are two CamCASP oracles on this branch and **they are not interchangeable**. Both were
run at the reviewed protocol (PBE0/aug-cc-pVTZ, GRAC, ALDA+CHF, LW→L3, PFIT WSM L3/L3/L3,
weight 4, coefficient 0.001, cutoff 1e-4) and differ in exactly one respect — how the
frequency-dependent density susceptibility is partitioned between sites.

| oracle | partition | relationship to us |
| ------ | --------- | ------------------ |
| `DF_*` | constrained density fitting of the FDDS onto atom-centred auxiliary functions (`ALGORITHM: DF`) — the **originally reviewed** model | a different physical model; not implemented here (Task G) |
| `ISA_GRID_*` | real-space grid iterated stockholder (`DIST-ALG ISA-GRID`) — **regenerated 2026-08-18** | the same family as our Task 4; this is the **acceptance oracle** |

The regeneration was a genuine single-variable experiment: the reviewed `H2O.cks`,
`H2O-A-asc.movecs` and `H2O-A.basis` were reused, the generated Psi4 SCF input `H2O_A.in` came
out **byte-identical** to the reviewed one, and the emitted `*.p2p` perturbation matrix was
**byte-identical**, so the WSM refinement solves the identical fit problem at identical fit
points. Only `DIST-ALG` changed.

```
Static per-site dipole polarizability, a.u.

          0                  2                  4                  6   a.u.
          |                  |                  |                  |
O  a_xx                                                                     o*
O  a_yy                                                          D              oI
O  a_zz                                                        D            o I
H  a_xx                  *
H  a_yy         oI       D
H  a_zz              oI      D
H  a_xz   D     *

   o = psi4 (ours)    I = CamCASP ISA-GRID    D = CamCASP DF    * = coincident
```

Read that figure as: wherever `I` and `D` sit on top of each other (`O xx`, `H xx`), the
partition does not matter and we agree with both. Wherever they separate (`O yy`, `O zz`,
`H yy`, `H zz`, `H xz`), we track `I` and not `D`.

### 4.2 Stage-level agreement (each isolated from downstream error)

| Stage | Oracle and method | Agreement | Verdict |
| ----- | ----------------- | --------- | ------- |
| Frequency grid | reviewed CASIMIR frequencies | `7.1e-15` absolute | **at parity** (`rtol=1e-10`) |
| Spherical → local Cartesian | reviewed tensors | exact `0.0` | **at parity** |
| Local → global rotation | reviewed tensors | exact `0.0` | **at parity** |
| **LW localization** | ORIENT, hermetic: reviewed `H2O_NL4_000.pol` in, reviewed `H2O_L3_000.pol` out, rank 4 truncated exactly as `Limit all rank 3` does | `O 7.4e-13`, `H1 5.5e-13`, `H2 5.1e-13` max abs over **all 675 tensor entries** | **at parity** |
| **Dispersion recoupling** | CASIMIR, fed the reviewed L3 models directly | `C6 2.5e-7`, `C8 2.0e-7`, `C10 1.1e-7`, `C12 2.0e-7` max relative | **at parity** (passes `rtol=1e-4`) |
| PDef mask vs reviewed model *definition* | pair-by-pair membership | O 38/38, H1 66/66, **no set difference either direction** | **at parity** |
| PDef mask vs reviewed model *output* | every forbidden entry across all 11 frequency blocks | max abs on a forbidden entry **exactly `0.000e+00`** (1804 checked for O, 1188 for H1); max on an allowed entry `1.895e+02` / `4.204e+01` | **at parity** |
| ISA populations | CamCASP ISA-GRID shape-function charges | O `8.83434` vs `8.81587` (Δ `0.018 e`); H `0.58283` vs `0.58866` (Δ `0.006 e`) | consistent with two different ISA variants; CamCASP's own residual charge is `0.0068` |
| Response magnitude | Psi4's own molecular `DIPOLE POLARIZABILITY`, identical functional/basis/grid | `0.970` | **correct** — this *is* the deliberate 25 % CHF + 75 % ALDA kernel difference |
| LW conservation | the site-pair response it localizes | `1.0000` | **exact** |
| Published sum conservation | the site-pair response it derives from | `0.9848` static, `0.9902` at `omega = 0.370` | accepted |

Note the LW localization result: because both ends of the localization step exist as files in
the reviewed tree, Task 3 is testable *hermetically* — no SCF, no basis, no grid and no
partition enters the comparison. It is an unambiguous verdict, and the answer is that our
localization is exact.

One subtlety that had to be resolved to get there: `H1` initially showed a `1.5e+01`
discrepancy. That was not an error. The ORIENT log states H1's local axes are the parent axes
rotated 180° about `z` (`p_x = -1, p_y = -1, p_z = +1`), applied after localization. Under that
rotation a real spherical harmonic `lm{c,s}` picks up `(-1)^m`, so
`alpha_(t,u) -> (-1)^(m_t + m_u) alpha_(t,u)`. Applying exactly that sign pattern took H1 from
`1.5472e+01` to `5.4623e-13`.

### 4.3 Static dipole block against both oracles

Ours at `PARITY_PROTOCOL` (PBE0/aug-cc-pVTZ, GRAC, DFT `590/99`, ISA `100/24/32`):

| site | comp | ours | ISA-GRID | ratio | DF | ratio |
| ---- | ---- | ---- | -------- | ----- | -- | ----- |
| `O` | `xx` | `6.9203` | `7.0420` | `0.9827` | `7.0435` | `0.9825` |
| `O` | `yy` | `7.4144` | `7.4738` | **`0.9921`** | `5.7621` | `1.287` |
| `O` | `zz` | `6.9551` | `7.1290` | `0.9756` | `5.5837` | `1.246` |
| `H1` | `xx` | `1.5388` | `1.5870` | `0.9696` | `1.5737` | `0.9778` |
| `H1` | `yy` | `0.6446` | `0.7609` | **`0.8471`** ← worst | `1.6174` | `0.3985` |
| `H1` | `zz` | `1.1591` | `1.2408` | `0.9341` | `2.0096` | `0.5768` |
| `H1` | `xz` | `0.6531` | `0.6453` | **`1.0122`** | `0.0058` | **`113.4`** |

Isotropic: `O` `7.0966` (ISA-GRID `0.984`, DF `1.158`); `H` `1.1142` (ISA-GRID `0.931`, DF
`0.643`); **molecular sum `9.3249`, which is `0.971` of *both* references** — the totals were
never the problem.

**Direction of effect, asserted without reference to any band.** On each of the eight
components where the two oracles separate by more than 5 %, the published value is closer to
ISA-GRID by factors of 7.9 to 83:

| component | ours | `d` to ISA-GRID | `d` to DF | ratio |
| --------- | ---- | --------------- | --------- | ----- |
| `O yy` | `7.4144` | `0.0594` | `1.6523` | `27.8` |
| `O zz` | `6.9551` | `0.1739` | `1.3714` | `7.9` |
| `H xz` | `0.6531` | `0.0078` | `0.6474` | **`83.0`** |
| `H yy` | `0.6446` | `0.1163` | `0.9728` | `8.4` |
| `H zz` | `1.1591` | `0.0817` | `0.8505` | `10.4` |

This is a test in its own right, because no tolerance can express it. A band can always be
widened; this cannot. The `xx` components are deliberately excluded from the discriminating
set — the two oracles agree there to better than one percent (`O` `0.0002`, `H` `0.0084`
separation), so which is nearer is noise, and DF is in fact marginally nearer on `H xx`
(`0.035` against `0.048`). The set is pinned in the test so it cannot quietly shrink to
whichever components happen to agree.

**A geometry defect surfaced in the process.** `REVIEWED_GEOMETRY` placed `H1` at *positive*
x while the reviewed `H2O_A.in` places it at negative x — a mirror image. That is invisible in
every x-even component and flips the sign of `xz` and `yz` on both hydrogens. Before the fix
`H1 alpha_xz` read `-0.653` against an oracle `+0.645`; after it, `+0.653`. Every symmetry,
conservation and fail-closed test passed either way, which is why it survived so long — only
a per-site comparison against a signed literal can see it.

### 4.4 Dynamic block

The same component (`H alpha_yy`) is worst at every frequency and the deviation falls
monotonically with frequency, so the static band bounds all eleven:

| `omega` | worst relative deviation |
| ------- | ------------------------ |
| `0.0` | `0.1529` |
| `0.370417` | `0.1104` |
| `37.82376` | `0.0443` |

Components that are zero by symmetry (`xy`, `yz` on every site; `xz` on O) are exactly `0.0`
in both, so the absolute floor only has to absorb representation noise.

### 4.5 Per-pair dispersion

Both CamCASP models were recoupled with **our own** dispersion engine (verified to `2.5e-7`),
so the recoupling is bit-identical across all three columns and only the L3 models differ.

| coefficient | ours | ISA-GRID | ratio | DF | ratio |
| ----------- | ---- | -------- | ----- | -- | ----- |
| `C6` `O-O` / `O-H` / `H-H` | `26.172` / `3.9095` / `0.5868` | `26.482` / `4.1423` / `0.6515` | `0.988` / `0.944` / `0.901` | `17.256` / `5.3823` / `1.6987` | `1.517` / `0.726` / `0.345` |
| `C8` | `393.48` / `50.163` / `6.3031` | `490.46` / `65.083` / `8.4633` | `0.802` / `0.771` / `0.745` | `346.42` / `83.908` / `18.328` | `1.136` / `0.598` / `0.344` |
| `C10` | `7129.9` / `870.87` / `107.76` | `9673.2` / `1262.3` / `168.19` | `0.737` / `0.690` / `0.641` | `7484.4` / `1523.5` / `291.48` | `0.953` / `0.572` / `0.370` |
| `C12` | `98233` / `11048` / `1240.7` | `1.504e5` / `18759` / `2278.8` | `0.653` / `0.589` / `0.545` | `1.272e5` / `20294` / `3216.5` | `0.772` / `0.544` / `0.386` |

Totals against ISA-GRID: C6 `0.967`, C8 `0.789`, C10 `0.717`, C12 `0.628`.

**The C6 total was already within 3 % of the DF reference even though `O-O` was wrong by a
factor of 1.5 and `H-H` by 3.** A pairwise-blind check would have missed the partition error
entirely. This is why the per-pair oracle matters and why the tests compare per-pair.

### 4.6 The rank deficit — the leading open gap

```
ratio to the matching (ISA-GRID) oracle

              0.50                    0.75                     1.00 = parity
              |                        |                        |
dipole  O yy  -------------------------------------------------@ 0.992
dipole  O xx  ------------------------------------------------@ 0.983
dipole  O zz  ------------------------------------------------@ 0.976
dipole  H xz  ---------------------------------------------------@ 1.012
dipole  H xx  -----------------------------------------------@ 0.970
dipole  H zz  -------------------------------------------@ 0.934
dipole  H yy  -----------------------------------@ 0.847
              .                        .                        .
C6      O-O   -------------------------------------------------@ 0.988
C6      O-H   --------------------------------------------@ 0.944
C6      H-H   ----------------------------------------@ 0.901
              .                        .                        .
C8      O-O   ------------------------------@ 0.802
C8      O-H   ---------------------------@ 0.771
C8      H-H   ------------------------@ 0.745
              .                        .                        .
C10     O-O   ------------------------@ 0.737
C10     O-H   -------------------@ 0.690
C10     H-H   --------------@ 0.641
              .                        .                        .
C12     O-O   ---------------@ 0.653
C12     O-H   ---------@ 0.589
C12     H-H   -----@ 0.545
```

**The partition does not explain this.** Once the oracle matches, the dipole block and C6 land
inside 10 %, but the deficit grows monotonically with rank and it grows against *both* CamCASP
models even though the recoupling is bit-identical across the comparison. Therefore our rank-2
and rank-3 site blocks are systematically smaller than CamCASP's. That is a Task 5
(`refine_wsm`) or rank-truncation question, not a partition one.

Untested candidate causes, in the order they should be checked:

1. **Fit-grid density.** 329 points against the reviewed 2000. The reviewed grid is
   architecturally infeasible in the current dense pair-row design (2000 points implies
   ~2.0e6 rows × 360 columns, about `5.8 GB` for the design matrix alone) and
   `kWSMMaximumPoints` is `500`.
2. **The relative rank cutoff pruning rank-3 columns.** The cutoff is now relative, which is
   correct, but its *value* (`1e-4` of the largest weighted column norm) has not been swept.
3. **The anchor penalty constrains only rank 1**, so the higher ranks are determined by the
   fit alone and are free to shrink toward whatever the point-response data weakly supports.

Candidate 3 is the most suspicious given the signature: an anchored block agrees, unanchored
blocks are uniformly *small*, and the deficit worsens with rank exactly as fit information
per variable decreases.

### 4.7 Verdict per published property

| Variable | Gate applied | State |
| -------- | ------------ | ----- |
| `ATOMIC POLARIZABILITY FREQUENCIES` | `rtol=1e-10, atol=1e-12` | **at parity** |
| `ATOMIC POLARIZABILITIES` | measured band `0.16`, `atol=1e-5` | **accepted** — worst `0.153` (`H alpha_yy`), 6 of 7 within 2–5 % |
| `ATOMIC DYNAMIC POLARIZABILITIES` | measured band `0.16`, `atol=1e-5` | **accepted** — same residual, improves with frequency |
| `ATOMIC C6` | band `0.11` | **accepted** — 1–10 % per pair |
| `ATOMIC C8` | band `0.27` | **not at parity** — 20–26 % low |
| `ATOMIC C10` | band `0.37` | **not at parity** — 26–36 % low |
| `ATOMIC C12` | band `0.47` | **not at parity** — 35–46 % low |

Two honest caveats about how this is gated:

- **The design spec's `rtol=1e-4` gate is not met** by any of the six polarizability/Cn
  comparisons against ISA-GRID. It was not loosened; instead the ISA-GRID comparisons use
  explicitly measured bands recorded at the point of use, and the `1e-4` gate is reserved for
  quantities that must agree exactly (frequency grid, recoupling prefactors, LW localization,
  partition-of-unity). That is a defensible choice — no available oracle uses our exact ISA
  variant — but design-spec acceptance criterion 8 ("all six property tests pass without
  `xfail`, broad numerical ranges, or skipped scientific stages") is **not literally
  satisfied**, and should not be reported as satisfied.
- The bands are **per-coefficient** rather than one number, precisely so the C6 comparison
  keeps testing something. Collapsing them to the C12 value would make C6 vacuous.

The six `DF_*` comparisons are retained at the `rtol=1e-4` gate as `xfail(strict=True)`. They
currently and correctly fail. `strict=True` means implementing a C-DF partition turns them
into a loud failure demanding the marker be removed, rather than letting them quietly start
passing.

### 4.8 Residual physics gap in the dipole block

The remaining `H alpha_yy` gap of `0.153` against the matching oracle is that the two ISA
variants differ: **CamCASP's ISA-GRID takes its shape functions from the basis-space ISA-A
functional and applies them pointwise on the integration grid; ours is real-space throughout.**
Closing it means implementing that hybrid variant. It is not currently planned.

Note also that the reviewed model's *full* L3 hydrogen array is itself not positive definite
(its log reports a `-0.754777` eigenvalue), which the plan's Global Constraints anticipate:
preserve full L3 tensors and expose a diagnostic rather than silently altering the model. The
published *dipole* block is a separate matter and is positive definite.

---

## 5. Routes without full Psi4 support

### 5.1 Not implemented at all

| # | Route | Status |
| - | ----- | ------ |
| 1 | **C-DF partitioning of the FDDS** (plan Task G) | The reviewed CamCASP default (`ALGORITHM: DF`, `C-DF` with constraints, `DF-TYPE-MONOMER NN`, 246-function Cartesian `MC` auxiliary basis) is **not reproducible in Psi4**. The `DF_*` literals exist only as strict xfails. Explicitly deferred after Task F showed ISA-GRID closes the parity gap. See the companion spec. |
| 2 | **ISA as a standalone `oeprop` property** (spec Tasks A–E) | Spec is *accepted* with all four scoping decisions resolved, but there is **zero code**: `grep` finds no `ISA_CHARGES`, no `compute_isa_partition`, no `ISA POPULATIONS`, and no `ISA_*` options. There is no way to obtain ISA charges, populations, convergence diagnostics, radial shape functions or ISA-DMA multipoles from Psi4. `ISAWeights::partition_weights_` is private with `friend class ISAPolResponseProvider` as the only access path, and `ISAWeights` is structurally bound to a sealed `FrozenResponseContext`. |
| 3 | **Externally supplied partitions and the A/B harness** (Tasks C, D) | No non-test entry point accepts a partition; `ISAWeights::create_test_only` is the only constructor. `isa_partition(...)` does not exist on the Python driver and `atomic_polarizabilities(..., partition=...)` is not accepted. So the decisive experiment — run the whole pipeline twice changing *only* the partition — cannot be run from Psi4. |
| 4 | **CamCASP's hybrid ISA-GRID variant** | Shape functions from the basis-space ISA-A functional applied pointwise on the grid. Ours is real-space throughout. This difference *is* the residual `H alpha_yy` 15 % gap. Requires an `ISA-Basis`, an `AtomAux-Basis` and a converged ISA-A step, none of which exist here. Not planned. |
| 5 | **Anisotropic Cn components** | Only the isotropic `00 00 0` coefficient is exposed. A real Clebsch–Gordan / rank-mixing contraction table is a hard prerequisite and does not exist. See the companion spec. |
| 6 | **Rank 4 anywhere** | The model is L3. `(1,4)` and `(4,1)` are excluded from C12, so **C12 is reviewed-model parity, not rank-complete physics**. The L3 block also carries no rank 0, i.e. no charge-flow term. |
| 7 | **Independent code and scientific review** | Task 8's last two checkboxes are open. The `ISA_GRID_*` literals in particular are marked "pending scientific review". |

### 5.2 Implemented but narrowly scoped — fails closed outside the reviewed protocol

| # | Restriction | Detail |
| - | ----------- | ------ |
| 8 | **Not reachable from a bare `OEProp` call** | `FrozenResponseContext` needs the GRAC/precursor/cation triple. Only the driver entry point drives it; `OEProp(wfn).add("ATOMIC_POLARIZABILITIES")` on a single wavefunction fails closed by design. This deliberately narrows the plan's original "one OEProp call publishes …" interface. |
| 9 | **Closed-shell neutral targets only** | Charge ≠ 0 or multiplicity ≠ 1 is rejected in the driver. The response layer is restricted-singlet C1 and requires `Ca == Cb`, `epsilon_a == epsilon_b`, and integer 0/1 occupations. The cation is UKS but exists only to fix the vertical IP. |
| 10 | **Covalent monomers only** | `derive_bond_graph` fails closed on a disconnected graph, so water dimers and all non-covalent complexes are rejected. LW localization over a disconnected graph is not well defined, and the reviewed model is a monomer. Accepted, but it means cluster/dimer C6 is out of reach. |
| 11 | **The WSM policy is frozen** | `validate_wsm_policy` throws unless rank = 3/3, weight type = 4, weight coefficient = 0.001 and cutoff = 1e-4 *exactly*. No other PFIT protocol is expressible. |
| 12 | **Fit grid capped at 500 points** | `kWSMMaximumPoints = 500`; the reviewed protocol used 2000. The dense pair-row design makes 2000 infeasible (~5.8 GB design matrix). Default is 329. |
| 13 | **Dense O(n_ov²) response** | Explicit Hessians solved by `DGESVX` on a doubled `2 n_ov` system, ERIs from `DirectJK` with BrianQC bypassed (`set_standard_integral_backend_only`), `SCREENING=NONE`, `INCFOCK` off, batching and integral OpenMP both pinned to one for determinism. This does not scale past small molecules. Hard caps: **16 sites** for constraints, **64** for response planning, **256** for dispersion. |
| 14 | **Provenance-seal strictness** | `HF::capture_response_provenance_if_converged` re-derives convergence from its own last *observed* iteration rather than trusting the SCF's verdict, so it refuses states Psi4 itself reports as converged. At aug-cc-pVTZ the cation UKS converges by Psi4's criteria (final `Delta E = -3.08e-07`, `RMS |[F,P]| = 8.92e-07` against `1e-6`) yet is rejected. `PARITY_PROTOCOL` works around it with `e_convergence 1e-10` / `d_convergence 1e-9`. Any protocol inheriting Psi4's `1e-6` defaults at a diffuse basis can still fail closed. |
| 15 | **Resource prerequisites are hard gates** | Psi4's 500 MB default memory fails closed on the default fit grid, which needs a WSM peak of `304,876,088` bytes and — because the stage gate reserves half of configured memory — at least ~0.58 GiB configured. The driver sets 4 GiB (`PIPELINE_MEMORY_BYTES`) and restores the previous value. |
| 16 | **Grid quality is basis dependent and must be pinned** | With aug-cc-pVDZ the LW charge-sum residual sticks at `1.2e-05` on a `302/50` DFT grid regardless of ISA density (tested to `150/24/32`); only `590/99` brings it inside `1e-6`. **The DFT grid, not the ISA grid, is binding once diffuse functions are present** — densifying ISA past `60/18/24` moved C6 by only `4e-05` relative. The wiring spec's grid table was measured without diffuse functions and does not transfer. |
| 17 | **Localization tolerance must be set from measurement** | `1e-8` is unattainable at aug-cc-pVTZ/`590/99`, where the measured LW charge-sum residual is `5.39e-07`. `PARITY_PROTOCOL` now uses `1e-6`. For reference the reviewed ORIENT run used `Sum-rule test 1e-7` and tolerated rank-truncation residuals up to `7e-4`. |
| 18 | **Parity tests are skipped by default** | The six oracle comparisons need `PSI4_ATOMIC_POLARIZABILITY_PARITY=1`. A routine test run exercises **none** of them; they report as skipped rather than passed precisely so an unexercised comparison can never be mistaken for a satisfied one. |

### 5.3 Pre-existing Psi4 defects worked around, not fixed

- `PointGroup::operator=` in `psi4/src/psi4/libmints/pointgrp.h` does not copy `bits_`, so a
  copied `PointGroup` has an indeterminate character table.
- `SymmetryOperation::rotation(2)` leaves `sin(pi) ~ 1.2e-16` off-diagonal, so `C2(z)` is not
  exactly diagonal; comparisons must use a tolerance, not exact zero.

---

## 6. Verified test status

Run locally on 2026-08-18 against `build_camcasp/stage` (Psi4 `1.11a1.dev152`):

```
$ python -m pytest tests/pytests -m mints -q --ignore=tests/pytests/test_camcasp_reference.py
389 passed, 13 skipped, 7540 deselected in 142.78s
```

```
$ PYTHONPATH=<repo root> python -m pytest tests/pytests/test_camcasp_reference.py -q
382 passed in 18.28s
```

| Suite | Result |
| ----- | ------ |
| `-m mints` (pipeline, math, symmetry, prerequisites, guards) | **389 passed, 13 skipped** |
| `test_camcasp_reference.py` (devtools oracle parser/validator, 182 test functions) | **382 passed** |
| Parity comparisons against `ISA_GRID_*` / `DF_*` literals | **not run** — gated behind `PSI4_ATOMIC_POLARIZABILITY_PARITY=1` |

The 13 skips are exactly the reviewed-literal parity comparisons. With the environment variable
set, the plan records 7 passing and 6 xfailed (the retained DF comparisons).

**Correction to the plan's record:** the plan calls `test_camcasp_reference.py`
"pre-existing-uncollectable". It is not — it collects and passes cleanly once the repository
root is on `PYTHONPATH` for `import devtools.camcasp_reference`. That note should be amended.

Test module inventory:

| Module | Test functions |
| ------ | -------------- |
| `test_camcasp_reference.py` | 182 |
| `test_atomic_polarizability_math.py` | 58 |
| `test_atomic_polarizabilities.py` | 35 |
| `test_wsm_fit_points.py` | 30 |
| `test_native_isa_weights.py` | 29 |
| `test_atomic_polarizability_prerequisites.py` | 24 |
| `test_atomic_polarizability_symmetry.py` | 23 |
| `test_native_constrained_least_squares.py` | 19 |
| `test_site_pair_response_contraction.py` | 19 |
| `test_point_response_evaluator.py` | 11 |
| `test_wsm_refinement.py` | 10 |
| `test_transition_multipole_projection.py` | 8 |
| `test_response_provenance_source_guard.py` | 8 |
| `test_native_atomic_polarizability_source_guard.py` | 2 |

---

## 7. Prioritised TODO

### P0 — the leading parity gap

- [ ] **Close the higher-rank deficit** (§4.6). C8/C10/C12 are 20–46 % low against the matching
      oracle with the recoupling held bit-identical. Sweep, in order: the anchor scope (extend
      the penalty to ranks 2–3 with an appropriately scaled weight and measure), the relative
      column cutoff, and the fit-grid density. Each is a single-variable experiment against the
      `ISA_GRID_*` literals already checked in.
- [ ] If it cannot be closed, **accept it as a documented caveat** with the mechanism named —
      not as a tolerance widening.

### P1 — deliverables with accepted specs and no code

- [ ] **ISA as a standalone `oeprop` property** (Tasks A–E). Decouple the partition from the
      response context, register `ISA_CHARGES`, publish the eight variables, add the
      non-test injection path and the Python driver surface, add ISA-DMA multipoles. Acceptance
      literals already exist from Task F (`ISA POPULATIONS`: O `8.81587`, H `0.588661`).
- [ ] **Partition A/B harness.** Once injection exists, run the pipeline twice changing only the
      partition. This is the experiment that would have found the two-oracle problem in a day.
- [ ] **Anisotropic Cn** — see the companion spec.
- [ ] **C-DF partitioning** — see the companion spec. Note this also converts six strict xfails
      into live tests at the `rtol=1e-4` gate.

### P2 — sharp edges that will bite the next user

- [ ] **Reconcile the provenance seal with Psi4's own convergence verdict** (§5.2 item 14). As
      it stands, a protocol inheriting Psi4's defaults at a diffuse basis fails closed on a
      calculation Psi4 reports as converged.
- [ ] **Document the grid/memory prerequisites in user-facing docs**, not only in specs. The
      failure mode (fail-closed at 500 MB, LW rejection at a coarse DFT grid) is opaque.
- [ ] **Amend the plan's claim** that `test_camcasp_reference.py` is uncollectable (§6).
- [ ] **Obtain independent code and scientific review** (Task 8, open).

### P3 — scope extensions

- [ ] Cluster/dimer support (currently rejected by the connected-bond-graph gate).
- [ ] Open-shell and charged targets.
- [ ] A sparse or blocked WSM design so the reviewed 2000-point fit grid becomes feasible.
- [ ] Rank 4, which would make C12 rank-complete physics rather than model parity.

---

## Cross-references

| Document | Covers |
| -------- | ------ |
| [`plans/2026-07-31-native-camcasp-parity.md`](docs/superpowers/plans/2026-07-31-native-camcasp-parity.md) | the executable plan, Tasks 1–8, implementation status, conservation-deficit record |
| [`plans/2026-07-29-camcasp-reference-provenance.md`](docs/superpowers/plans/2026-07-29-camcasp-reference-provenance.md) | the dev-only reference regeneration workflow |
| [`specs/2026-07-29-camcasp-atomic-polarizability-parity-design.md`](docs/superpowers/specs/2026-07-29-camcasp-atomic-polarizability-parity-design.md) | the original design, canonical protocol, output contract, tolerances |
| [`specs/2026-08-17-isotropic-dispersion-recoupling.md`](docs/superpowers/specs/2026-08-17-isotropic-dispersion-recoupling.md) | the `00 00 0` recoupling formula and its validation |
| [`specs/2026-08-17-pdef-constraint-derivation.md`](docs/superpowers/specs/2026-08-17-pdef-constraint-derivation.md) | the symmetry mask derivation and its two-way validation |
| [`specs/2026-08-17-end-to-end-wiring.md`](docs/superpowers/specs/2026-08-17-end-to-end-wiring.md) | stage chain, frame hazard, publication contract, grid and memory |
| [`specs/2026-08-17-parity-debugging-map.md`](docs/superpowers/specs/2026-08-17-parity-debugging-map.md) | which stages are settled exactly (partly superseded) |
| [`specs/2026-08-18-isa-partition-oeprop.md`](docs/superpowers/specs/2026-08-18-isa-partition-oeprop.md) | ISA as a first-class property; Tasks A–G |
| [`specs/2026-08-18-isa-grid-oracle.md`](docs/superpowers/specs/2026-08-18-isa-grid-oracle.md) | the Task F regeneration, the measured band, the rank-deficit finding |
| `docs/superpowers/specs/2026-08-18-anisotropic-cn-and-cdf.md` | **new** — the companion spec for anisotropic Cn and C-DF |
