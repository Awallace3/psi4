# Anisotropic Dispersion Recoupling — Derivation and Numerical Proof

Companion to [`2026-08-18-anisotropic-cn-and-cdf.md`](2026-08-18-anisotropic-cn-and-cdf.md),
**Part B only**. Written 2026-08-18.

Status: **derivation complete and numerically proven**; no C++ has been written and no
production file has been touched. This document is the input to tasks B2–B8.

The executable reference that generates and validates everything below is
`partB_reference.py` (scratchpad). It depends only on `numpy`, runs in ~25 s, prints one
PASS/FAIL line per check and exits non-zero on any failure. `--dump <path>` writes the
versioned table as JSON.

---

## 1. Verified citations

Every section number and page below was checked against the publisher's own front matter
(the OUP/ETH table-of-contents PDF for the book) rather than recalled.

| Key | Reference | What was verified |
| --- | --------- | ----------------- |
| **[S13]** | A. J. Stone, *The Theory of Intermolecular Forces*, 2nd ed., Oxford University Press (2013). ISBN 978-0-19-967239-4 | **§3.3 "Spherical tensor formulation", p. 47** ✔; **§4.3 "The dispersion energy", p. 64** ✔; §9.5 "Distributed dispersion interactions", p. 179 ✔; Appendix B "Spherical Tensors", p. 271 (B.1 Spherical harmonics 271, B.2 Rotations of the coordinate system 273, B.3 Spherical tensors 274, B.4 Coupling of wavefunctions and spherical tensors 275) ✔; Appendix E "Cartesian–Spherical Conversion Tables", p. 285 ✔; **Appendix F "Interaction Functions", p. 291** ✔ |
| **[S13-corr]** | Stone, *Corrections and notes on The Theory of Intermolecular Forces, 2nd edition*, last updated 30 May 2016, `https://www-stone.ch.cam.ac.uk/timf_info/corrections_2e.pdf` | The published errata list corrections at p. 5, p. 17, p. 60, p. 164 (eqn 9.1.17) and p. 165 (eqn 9.1.22) **only**. **No correction touches §3.3, §4.3 or Appendix F**, so the sections this derivation leans on stand as printed. ✔ |
| **[S78]** | A. J. Stone, "The description of bimolecular potentials, forces and torques: the S and V function expansions", *Mol. Phys.* **36**, 241 (1978). DOI `10.1080/00268977800101541` | Journal, volume and DOI verified; this is the **primary definition of the S functions**. |
| **[ST84]** | A. J. Stone and R. J. A. Tough, "Spherical tensor theory of long-range intermolecular forces", *Chem. Phys. Lett.* **110**, 123 (1984). DOI `10.1016/0009-2614(84)80160-8` | Verified. Gives electrostatic, induction **and dispersion** energies in terms of spherical-tensor multipoles/polarizabilities with the orientation dependence carried by `S_{l1 l2 j}^{k1 k2}`. |
| **[PSA84]** | S. L. Price, A. J. Stone and M. Alderton, "Explicit formulae for the electrostatic energy, forces and torques between a pair of molecules of arbitrary symmetry", *Mol. Phys.* **52**, 987–1001 (1984) | Verified via ADS. |
| **[WS03]** | G. J. Williams and A. J. Stone, "Distributed dispersion: A new approach", *J. Chem. Phys.* **119**, 4620–4628 (2003). DOI `10.1063/1.1595636` | Verified. This is the method reference cited by CamCASP's own published description of the `Casimir` recoupling step. |
| **[MS18]** | A. J. Misquitta and A. J. Stone, "ISA-Pol: distributed polarizabilities and dispersion models from a basis-space implementation of the iterated stockholder atoms procedure", *Theor. Chem. Acc.* **137**, 153 (2018). DOI `10.1007/s00214-018-2371-4`; arXiv:1806.06737 | Verified, and read. It states that the dispersion models are "recombined using methods [WS03], [S13] (**§4.3.4**) implemented in the Casimir module", and that both isotropic **and anisotropic** models are produced with `C6 … C12` (and, in the text, `C7`), which independently corroborates that odd-order coefficients are part of the anisotropic set. |
| **[BS68]** | D. M. Brink and G. R. Satchler, *Angular Momentum*, 2nd ed., Oxford University Press (1968) | Standard source for Racah's Clebsch–Gordan formula, the Racah-normalised harmonic product rule and the rotation law of coupled tensors. |

Part A's citation (Misquitta & Stone, *J. Chem. Phys.* **124**, 024111 (2006), constrained
density fitting) was confirmed as reference 17 of [MS18], but Part A is out of scope here.

**What is cited for what.** The spec's starting point — "[S13] §3.3 for S-functions and §4.3
for dispersion" — is *almost* right and needs one correction: **§3.3 is where the spherical
tensor formulation and the interaction functions `T` live; the S functions themselves are
defined in [S78] and used in [S13] §3.3/Appendix F.** §4.3 is indeed the Casimir–Polder /
dispersion section, and [MS18] pins the anisotropic recoupling to §4.3.4 specifically.

---

## 2. Conventions, stated explicitly

### 2.1 Harmonics

Racah-normalised complex spherical harmonics ([S13] Appendix B):

```
C_{l m}(rhat) = sqrt(4 pi / (2l+1)) Y_{l m}(rhat)
R_{l m}(r)    = r^l C_{l m}(rhat)          (regular solid harmonic)
I_{l m}(r)    = C_{l m}(rhat) / r^{l+1}    (irregular solid harmonic)
```

### 2.2 Real components and their ordering — matching the C++ exactly

Read off `psi4/src/psi4/libmints/atomic_polarizability.cc`
(`regular_harmonics`, `real_to_complex`, `complex_to_real`, `real_cosine_index`,
`real_sine_index`, the `kL3MonomialParity` comment at line 4256, and
`local_spherical_dipole_to_cartesian`):

```
C_{l,+m} = (-1)^m ( R_{l m c} + i R_{l m s} ) / sqrt(2)      (m > 0)
C_{l,-m} =         ( R_{l m c} - i R_{l m s} ) / sqrt(2)
C_{l, 0} =           R_{l 0}
```

Inverting gives the unitary matrix `U^{(l)}` used throughout, `R_t = sum_m U^{(l)}[t,m] C_{l m}`:

```
R_{l 0}   = C_{l 0}
R_{l m c} = [ (-1)^m C_{l,+m} + C_{l,-m} ] / sqrt(2)
R_{l m s} = [ (-1)^m C_{l,+m} - C_{l,-m} ] / (i sqrt(2))
```

**The 15-component L3 ordering is fixed and is not negotiable:**

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
| ----- | - | - | - | - | - | - | - | - | - | - | -- | -- | -- | -- | -- |
| label | `10` | `11c` | `11s` | `20` | `21c` | `21s` | `22c` | `22s` | `30` | `31c` | `31s` | `32c` | `32s` | `33c` | `33s` |

Rank block offsets are `dispersion_rank_offset(l) = l*l - 1`, i.e. rank 1 → `[0,3)`,
rank 2 → `[3,8)`, rank 3 → `[8,15)`. Rank 0 is **absent** from the 15-component dispersion
tensor (it lives in the 16-component `L3WorkingVector` used upstream). The dipole ordering
`(10, 11c, 11s) = (z, x, y)` is confirmed by
`local_spherical_dipole_to_cartesian`'s permutation `{1, 2, 0}`.

The general-`l` real components (needed for `l1, l2` up to 6) follow the same pattern:
`l0, l1c, l1s, l2c, l2s, ..., llc, lls`, i.e. index `0` for `m = 0` and `2m-1 / 2m` for the
cosine/sine partners of order `m`.

### 2.3 Rotations

`W^{(l)}[O]` is the real rank-`l` rotation defined by `R_t(O r) = sum_s W^{(l)}[t,s] R_s(r)`;
`D^{l}_{m k}(O)` is defined by `C_{l m}(O rhat) = sum_k D^{l}_{m k}(O) C_{l k}(rhat)`, and
`D^{(l)} = U^{(l)†} W^{(l)} U^{(l)}`. A polarizability given in a site's local frame is taken
to the global frame by `alpha^glob = W alpha^loc W^T` on each `(la, la')` block.

---

## 3. Derivation of the interaction tensor `T`

### 3.1 From published elementary results

**(a) Racah addition theorem** ([S13] App. B):

```
P_l(cos gamma) = sum_{m=-l}^{l} (-1)^m C_{l m}(1) C_{l,-m}(2)
```

Substituting the real transform of §2.2 collapses the phases and gives the *real* form

```
r1^l r2^l P_l(cos gamma) = sum_{t in rank l} R_t(r1) R_t(r2)                     (3.1)
```

so the real basis of §2.2 is orthonormal in exactly the sense the multipole expansion needs.

**(b) Laplace expansion.** For `r1 < r2`, `1/|r1 - r2| = sum_l (r1^l / r2^{l+1}) P_l(cos gamma)`,
so with (3.1)

```
1/|r1 - r2| = sum_{l,t} R_t(r1) I_t(r2),      I_t(r) = R_t(r) / r^{2l+1}          (3.2)
```

Hence the electrostatic potential of a unit real-spherical multipole `t` sitting at `R_A` is
exactly `I_t(r - R_A)`.

**(c) Definition of `T`.** With `R = R_B - R_A` and `s = r - R_B`,

```
E = sum_{t,u} Q^A_t T_{tu}(R) Q^B_u ,
```

where `T_{tu}(R)` is the coefficient of `R_u(s)` in the expansion of `I_t(s + R)` in *regular*
solid harmonics of `s`. That expansion exists and is unique because `I_t(s + R)` is harmonic
in `s`, and — crucially — each rank appears with exactly one power of `|s|`. The coefficient
can therefore be extracted *exactly* by a surface projection:

```
T_{tu}(R) = (2 l_u + 1) rho^{-l_u} * < I_t(rho shat + R) Rhat_u(shat) >_{shat}     (3.3)
```

with `< . >` the unit-sphere average. (3.3) is implemented in the reference as
`interaction_tensor_numeric` and is a fully independent construction of `T` — no
Clebsch–Gordan algebra, no closed form, nothing recalled.

### 3.2 The closed form

Transforming (3.3) into the complex basis, `T^c = U^{(la)T} T U^{(lb)}`, and comparing against
Clebsch–Gordan structure **determines** (it was not assumed):

```
                                              ______________
T^c_{la ma, lb mb}(R) = (-1)^{lb} * sqrt( binom(2L, 2 la) )
                        * <la ma; lb mb | L M>
                        * conj( C_{L M}(Rhat) ) / R^{L+1}                          (3.4)

     L = la + lb ,   M = ma + mb ,   R = R_B - R_A
```

Equivalently `conj(C_{L M}) = (-1)^M C_{L,-M}`. Both the amplitude and the phase `(-1)^{lb}`
were fitted numerically against (3.3) over all nine `(la, lb)` in `{1,2,3}^2` and reproduce
`T` to the quadrature accuracy of (3.3); the *amplitude* is additionally fixed independently
by the norm identity, since

```
sum_{ma mb} |CG|^2 |C_{L M}(Rhat)|^2  =  sum_M |C_{L M}(Rhat)|^2  =  1
```

so `sum_{t,u} |T^{(la,lb)}_{tu}|^2 = binom(2L, 2la) / R^{2(L+1)}` follows *automatically*
from (3.4). The spec's §B.3.4 norm identity is therefore not an extra assumption — it is a
corollary of the CG structure of `T`.

The real-basis tensor is `T = U^{(la)*} T^c U^{(lb)†}`, and the full 15×15 tensor is the
block assembly over `la, lb in {1,2,3}`.

---

## 4. Derivation of the recoupling table `W`

Start from §B.3.1, with `t = (la, ma)`, `t' = (la', ma')` on A and `u = (lb, mb)`,
`u' = (lb', mb')` on B:

```
E_disp = -(1/2 pi) integral dw sum_{t t' u u'} T_{tu} T_{t'u'} alpha^A_{t t'} alpha^B_{u u'}
       = - sum_{(la, la', lb, lb')} Term(la, la', lb, lb')
```

**Step 1 — complex basis.** With `T^c = U^{(la)T} T U^{(lb)}` and

```
alpha^c_{ma ma'} = sum_{t t'} conj(U^{(la)}[t, ma]) conj(U^{(la')}[t', ma']) alpha_{t t'}
```

the contraction is unchanged, so the whole double sum can be evaluated in the complex basis.
(The two transforms differ — one uses `U`, the other `U*` — which is exactly what makes the
bilinear form invariant.)

**Step 2 — insert (3.4).** `T^c_{la ma, lb mb}` carries `L1 = la + lb`, `M1 = ma + mb`;
`T^c_{la' ma', lb' mb'}` carries `L2 = la' + lb'`, `M2 = ma' + mb'`. The total radial power is

```
n = L1 + L2 + 2 = la + la' + lb + lb' + 2
```

**Step 3 — couple the index pairs.** By CG completeness,

```
alpha^{A,c}_{la ma, la' ma'} = sum_{l1 mu1} <la ma; la' ma' | l1 mu1> Acal_{l1 mu1}
alpha^{B,c}_{lb mb, lb' mb'} = sum_{l2 mu2} <lb mb; lb' mb' | l2 mu2> Bcal_{l2 mu2}
```

and, under a rotation `Omega_A` of site A, the coupled tensor rotates as a single rank-`l1`
object ([BS68], the standard "rotation of a coupled product" identity, verified numerically):

```
Acal^{glob}_{l1 mu1} = sum_{k1} D^{l1}_{mu1 k1}(Omega_A) a^{loc}_{l1 k1}
```

**Step 4 — the four-CG geometric sum.** Define

```
G_{M1 M2, mu1 mu2} = sum_{ma mb ma' mb'}
        <la ma ; lb mb  | L1 M1> <la' ma'; lb' mb' | L2 M2>
        <la ma ; la' ma'| l1 mu1> <lb mb ; lb' mb'  | l2 mu2>
```

Contracting `G` against `<L1 M1; L2 M2 | j M>` collapses onto a single CG (this is a 9j symbol
in disguise):

```
sum_{M1 M2} G_{M1 M2, mu1 mu2} <L1 M1; L2 M2 | j M> = Lambda_j * <l1 mu1; l2 mu2 | j mu>   (4.1)
```

with `Lambda_j` independent of `(mu1, mu2)`. **(4.1) is proven numerically, not assumed:**
`Lambda_j` is obtained by least squares over all `(mu1, mu2)` and the worst residual over all
5497 `(block, l1, l2, j)` combinations is `4.4e-16`.

**Step 5 — product of the two `Rhat` harmonics** ([BS68]):

```
C_{L1 M1} C_{L2 M2} = sum_j <L1 M1; L2 M2 | j M> <L1 0; L2 0 | j 0> C_{j M}
```

The `<L1 0; L2 0 | j 0>` factor forces **`L1 + L2 + j` even**, which is the master selection
rule of the whole table.

**Step 6 — assemble.**

```
Term = R^{-n} sum_{l1 l2 j} g * sum_{k1 k2} a^A_{l1 k1} a^B_{l2 k2}
                                  Stilde^{l1 l2 j}_{k1 k2}(Omega_A, Omega_B, Rhat)

g = (-1)^{lb + lb'} sqrt( binom(2 L1, 2 la) binom(2 L2, 2 la') )
    * <L1 0; L2 0 | j 0> * Lambda_j                                                (4.2)

Stilde^{l1 l2 j}_{k1 k2} = sum_{mu1 mu2} <l1 mu1; l2 mu2 | j mu>
        D^{l1}_{mu1 k1}(Omega_A) D^{l2}_{mu2 k2}(Omega_B) conj(C_{j mu}(Rhat)),
        mu = mu1 + mu2                                                              (4.3)
```

(4.3) is the S function, and its structure — CG coupling of two Wigner-D factors against a
harmonic of the intermolecular vector — is exactly the [S78]/[ST84] form.

**Step 7 — real form and the reality phase.** Transforming `k1, k2` to the real ordering of
§2.2 and inserting one phase per side:

```
S^{l1 l2 j}_{k1 k2}   = Ncal * sum_{kappa1 kappa2} conj(U^{(l1)}[k1, kappa1])
                                                   conj(U^{(l2)}[k2, kappa2])
                                                   Stilde^{l1 l2 j}_{kappa1 kappa2}

Pcheck^{(la, la', l1)}_{k1, (t t')} = eta * sum_{kappa1 ka ka'}
        U^{(l1)}[k1, kappa1] <la ka; la' ka' | l1 kappa1>
        conj(U^{(la)}[t, ka]) conj(U^{(la')}[t', ka'])

eta  = 1 if (la + la' + l1) is even, else -i
Ncal = 1 if (l1 + l2 + j)   is even, else -i                                        (4.4)
```

`Pcheck` is then real and `S` is then real. Because `L1 + L2 + j` is even and
`L1 + L2 = n - 2 = la + la' + lb + lb'`, the three parities
`(la+la'+l1)`, `(lb+lb'+l2)`, `(l1+l2+j)` always sum to an even number, so
`eta_A eta_B Ncal = ±1` and `g^r = g / (eta_A eta_B Ncal)` is **real**. This is a structural
consistency check, not a convention — if the phases had not conspired, the derivation would
be wrong.

*Relation to Stone's phase.* The phase that makes `S` real is `i^{-(l1 - l2 - j)}` up to a
real sign; that is the origin of the `i^{l1 - l2 - j}` factor that appears in Stone's own
definition of the S functions. The convention adopted here (`Ncal` as in (4.4)) differs from
Stone's by a real `±1` per `(l1, l2, j)` triple and possibly by an overall `(2j+1)^{1/2}`-type
normalisation; see §9.

### 4.1 The table

```
W^{n, l1 k1, l2 k2, j}_{(t t')(u u')}
    = sum over blocks (la, la', lb, lb') with la+la'+lb+lb'+2 = n  of
          g^r_{(la la' lb lb'), l1 l2 j}
        * Pcheck^{(la, la', l1)}_{k1, (t t')}
        * Pcheck^{(lb, lb', l2)}_{k2, (u u')}                                       (4.5)

C_n[l1 k1, l2 k2, j] = sum_{(t t')(u u')} W^{n, ...}_{(t t')(u u')} M^{AB}_{(t t')(u u')}

E_disp = - sum_n R^{-n} sum_{labels} C_n[label] * S_label(Omega_A, Omega_B, Rhat)   (4.6)
```

with `M^{AB}_{(t t')(u u')} = (1/2 pi) sum_k w_k alpha^A_{t t'}(i w_k) alpha^B_{u u'}(i w_k)`
exactly as §B.3.2 defines it.

**The table is stored factorised**, because (4.5) is a rank-one product per block:

* `g^r` — **2723 nonzero scalars**, indexed by `(la, la', lb, lb', l1, l2, j)`. This is the
  physics.
* `Pcheck` — universal CG/real-transform coupling matrices, indexed by `(la, la', l1)`
  (**37** arrays of shape `(2l1+1, 2la+1, 2la'+1)`). This is pure angular-momentum algebra and
  can equally well be regenerated at load time.

  > **Corrected 2026-08-19.** This section originally said 30 arrays. The count is 37: `l1`
  > runs over the triangle `|la - la'| <= l1 <= la + la'`, giving `2*min(la, la') + 1` values
  > per `(la, la')` pair, so `3+3+3+3+5+5+3+5+7 = 37` over `la, la' in {1,2,3}`. The C++
  > implementation stores 37 and its loader asserts the count.

Materialising `W` densely for one label costs `15^4 = 50625` doubles; there are 29762 labels,
so a dense table would be 12 GB. **The factorised form must be the storage form.** The
reference provides `dense_W(label)` for testing and proves (over 40 random labels) that the
factorised contraction reproduces the dense `W · M` contraction to `1.5e-17`.

---

## 5. The label set for an L3 model

Enumeration rules (all forced by the derivation):

```
la, la', lb, lb' in {1, 2, 3}                       (L3 model: ranks 1..3)
n  = la + la' + lb + lb' + 2                        in 6 .. 14
L1 = la + lb ,  L2 = la' + lb'
l1 in |la - la'| .. la + la'                        in 0 .. 6
l2 in |lb - lb'| .. lb + lb'                        in 0 .. 6
j  in max(|L1-L2|, |l1-l2|) .. min(L1+L2, l1+l2)    in 0 .. 12
(L1 + L2 + j) even                                  [<L1 0; L2 0|j 0> selection rule]
<L1 0; L2 0|j 0> != 0  and  Lambda_j != 0
k1 in 0 .. 2 l1 ,  k2 in 0 .. 2 l2                  (real components of §2.2)
```

Measured, from `partB_reference.py`:

| Quantity | Value |
| -------- | ----- |
| **Total labels `(n, l1, k1, l2, k2, j)`** | **29 762** |
| distinct `(n, l1, l2, j)` quadruples | 530 |
| nonzero `g^r` block entries | 2 723 |
| labels per order | C6 104, C7 391, C8 896, C9 1748, C10 3063, C11 4486, C12 6297, C13 7457, C14 5320 |
| `l1`, `l2` range | 0 … 6 |
| `j` range | 0 … 12 |
| **Labels that survive a *symmetric* L3 model** | **26 104** (3 658 identically zero) |
| live labels per order | C6 86, C7 340, C8 812, C9 1631, C10 2979, C11 4369, C12 6213, C13 6126, C14 3548 |

**Ordering convention (published order).** Labels are sorted lexicographically by
`(n, l1, k1, l2, k2, j)`, with `k1`/`k2` running over the real components in the §2.2 order
(`l0, l1c, l1s, …, llc, lls`). The isotropic label is `(n, 0, 0, 0, 0, 0)`, printed as
`Cn [00 00 0]`.

### 5.1 The symmetry-null sub-set

Psi4's refined L3 tensors are symmetric **by construction**, and this was checked in the
source, not assumed: `refine_wsm` solves for only the `kWSMVariablesPerSite = 120 = 15*16/2`
upper-triangle variables per site (`wsm_upper_index`), and assigns both triangles from the
same value at `atomic_polarizability.cc` lines 4057–4058, recording
`max_output_asymmetry` as a diagnostic. It additionally rejects a non-symmetric localized
reference (line ≈3885) and a non-symmetric point response (line ≈3932). The 3658
symmetry-null labels are therefore genuinely prunable for this pipeline.

For a symmetric `alpha`, the `(la, la')` and
`(la', la)` blocks are transposes, and the CG exchange rule
`<la' ka'; la ka | l1 kappa> = (-1)^{la + la' - l1} <la ka; la' ka' | l1 kappa>` gives

```
Acal(la', la) = (-1)^{la + la' - l1} Acal(la, la')
```

so a label is identically zero unless some unordered pair `({la,la'}, {lb,lb'})` has

```
sum over the (<=4) ordered variants of  sigma_A sigma_B g^r  !=  0
sigma_A = (-1)^{la + la' - l1}  (and the whole side vanishes when la == la' and l1 is odd)
```

This analytic criterion was checked against a numerically determined null set (intersection
over six independent random symmetric model pairs): **3658 analytic, 3658 numerical, 0
disagreements**.

The C6 consequence is the familiar textbook result and is a strong external sanity check:
the surviving C6 `(l1, l2, j)` triples are exactly

```
(0,0,0)  (0,2,2)  (2,0,2)  (2,2,0)  (2,2,2)  (2,2,4)
```

i.e. the dipole–dipole dispersion anisotropy carries only the rank-0 and rank-2 parts of each
site's polarizability, with `j` from the `(l1, l2)` triangle — which is the standard
anisotropic-`C6` label set.

### 5.2 Worked `g^r` values for `n = 6`

The single `n = 6` block is `(la, la', lb, lb') = (1,1,1,1)`:

| `(l1, l2, j)` | `g^r` | closed form |
| ------------- | ----- | ----------- |
| `(0, 0, 0)` | `2.000000000000` | `2` |
| `(0, 2, 2)` | `-1.414213562373` | `-sqrt(2)` |
| `(1, 1, 0)` | `1.732050807569` | `sqrt(3)` — symmetry-null |
| `(1, 1, 2)` | `2.449489742783` | `sqrt(6)` — symmetry-null |
| `(2, 0, 2)` | `-1.414213562373` | `-sqrt(2)` |
| `(2, 2, 0)` | `0.447213595500` | `1/sqrt(5)` |
| `(2, 2, 2)` | `0.534522483825` | `sqrt(2/7)` |
| `(2, 2, 4)` | `4.302822993604` | `18 sqrt(2/35)` |

These were **generated by code**, not typed; the closed forms are annotations only.

---

## 6. Validation results (measured)

All from one run of `partB_reference.py` (numpy 1.26.4, Python 3, 25 s, exit code 0).
**22 checks, 0 failures.**

### Interaction tensor

| Check | Residual |
| ----- | -------- |
| `T(1,1)` equals `(delta_ab - 3 n_a n_b)/R^3` in Cartesians | max dev `4.16e-17` |
| closed form (3.4) matches the independent surface projection (3.3) | max rel dev `1.27e-12` (limited by the projection quadrature, not by (3.4)) |
| `T` covariant under rotation of the frame | max rel dev `6.22e-15` |
| **norm identity** `sum|T|^2 = binom(2la+2lb,2la)/R^{2(la+lb+1)}` over all 9 rank pairs (the 8 in `dispersion_rank_pairs()` plus `(3,3)`) | max rel dev `8.88e-16` |

### Block-product Casimir–Polder integral

| Check | Residual |
| ----- | -------- |
| tracing the diagonal rank blocks of `M` reproduces `compute_dispersion_impl` (re-implemented in numpy) on synthetic L3 models, C6…C12 | max rel dev `2.22e-16` |

### Recoupling algebra

| Check | Residual |
| ----- | -------- |
| the four-CG sum (4.1) collapses onto a single CG, over 5497 `(block, l1, l2, j)` | max residual `4.44e-16` |
| factorised contraction equals the dense `W · M`, 40 random labels | max rel dev `1.45e-17` |

### §B.5 check 1 — isotropic reduction

`W[n, 00, 00, 0]` traced over the diagonal blocks, **max rel dev `9.47e-16`**:

| `(la, lb)` | traced | expected `binom(2la+2lb, 2la)` |
| ---------- | ------ | ------------------------------ |
| `(1,1)` | `6.000000000000` | `6` |
| `(1,2)` | `15.000000000000` | `15` |
| `(2,1)` | `15.000000000000` | `15` |
| `(1,3)` | `28.000000000000` | `28` |
| `(3,1)` | `28.000000000000` | `28` |
| `(2,2)` | `70.000000000000` | `70` |
| `(2,3)` | `210.000000000000` | `210` |
| `(3,2)` | `210.000000000000` | `210` |

and end-to-end, `C_n[00 00 0]` against the existing isotropic engine, **max rel dev `7.77e-16`**:

```
C6   2.5302571995e+02  vs  2.5302571995e+02
C8   1.6761549954e+03  vs  1.6761549954e+03
C10  8.2205330981e+03  vs  8.2205330981e+03
C12  3.0250596550e+04  vs  3.0250596550e+04
```

### §B.5 check 2 — direct energy reconstruction (the decisive one)

`R = 6.5 a0`, one synthetic 15×15 symmetric L3 model per grid point per site on the protocol
`make_casimir_grid(10, 0.5)` grid. `E_disp` computed (a) from the §B.3.1 double sum with the
interaction tensor and no table, (b) from `C_n[label]` contracted with the S functions:

| orientation | `E_direct` | rel dev |
| ----------- | ---------- | ------- |
| axial, both frames identity | `-6.7349635577200889e-03` | `1.03e-15` |
| axial, A rotated about z | `-6.1976400272176187e-03` | `5.60e-15` |
| axial, B tilted by pi/2 | `-7.6639091251600906e-03` | `1.13e-15` |
| axial along x | `-8.3328675414398468e-03` | `6.66e-15` |
| both frames identity, R along (1,1,1) | `-8.6049022549641099e-03` | `8.67e-15` |
| A = B = pi/2 about y, R = z | `-8.2717353153022866e-03` | `8.81e-15` |
| degenerate: A rotated by pi about y | `-5.3664068189616205e-03` | `4.85e-16` |
| generic 1 | `-9.2193178457897814e-03` | `9.60e-15` |
| generic 2 | `-4.6633818666984477e-03` | `0.00e+00` |
| generic 3 | `-9.9347488904789893e-03` | `1.54e-14` |
| generic 4 | `-6.8746791204637447e-03` | `6.18e-15` |

**Max rel dev `1.54e-14` over 11 orientations** (target was `~1e-13`). The S functions were
simultaneously confirmed real, `max |Im S| = 1.73e-16`.

### §B.5 check 3 — orientational average

| Check | Residual |
| ----- | -------- |
| explicit `SO(3) × SO(3) × S^2` quadrature of every S function equals the analytic selection rule `<S> = delta_{label,(0,0,0,0,0)}` (Euler-angle Gauss quadrature exact to `l = 6`, product-Gauss sphere grid exact to `j = 12`), 29762 labels | max dev `2.95e-14` |
| orientationally averaged expansion returns `C_n[00 00 0]` | max rel dev `2.96e-14` |

The analytic route is `<D^l_{mu k}>_{SO(3)} = delta_{l0}` and `<C_{j mu}>_{S^2} = delta_{j0}`,
which also gives an independent *analytic* proof of the isotropic limit: averaging the direct
energy over both molecular orientations uses
`<W^{(l)}_{ts} W^{(l')}_{t's'}> = delta_{ll'} delta_{tt'} delta_{ss'} / (2l+1)` and collapses
§B.3.1 straight onto `sum_{la lb} alphabar_{la} alphabar_{lb} binom(2la+2lb,2la)/R^{2(la+lb+1)}`.

### §B.5 check 4 — permutation symmetry

The exchange rule follows from `<l2 mu2; l1 mu1|j mu> = (-1)^{l1+l2-j} <l1 mu1; l2 mu2|j mu>`
together with `C_{j mu}(-Rhat) = (-1)^j C_{j mu}(Rhat)`:

```
W^{n, l2 k2, l1 k1, j}_{(u u')(t t')} = (-1)^{l1 + l2} W^{n, l1 k1, l2 k2, j}_{(t t')(u u')}
C_n[B][A][l2 k2, l1 k1, j]            = (-1)^{l1 + l2} C_n[A][B][l1 k1, l2 k2, j]
```

| Check | Residual |
| ----- | -------- |
| label set closed under `(l1 k1) <-> (l2 k2)` | 0 unmatched labels |
| `W` exchange rule, 20 random labels | max rel dev `8.02e-16` |
| `C_n` exchange rule over all 29762 labels | max rel dev `1.01e-16` |
| the physical statement it encodes, `E_disp(A,B; R) = E_disp(B,A; -R)` | rel dev `6.66e-16` |

**Note the sign is `(-1)^{l1+l2}`, not `(-1)^j`.** Since `L1 + L2 + j` is even and
`L1 + L2 = n - 2`, `j` has the same parity as `n`, so `(-1)^j` is *not* the right factor —
that was tested and fails by a full sign on part of the table.

### Symmetry-null criterion and loader invariants

| Check | Result |
| ----- | ------ |
| analytic symmetry-null criterion vs numerical null set | 3658 / 3658, 0 disagreements |
| every published label satisfies the structural invariants of §7 | 0 violations |
| no published label has an identically vanishing `W` | 0 empty labels |
| orders present | `n = 6 … 14`, all nine |
| isotropic `00 00 0` labels present | `n = 6, 8, 10, 12, 14` |

---

## 7. Structural invariants the C++ loader must enforce

Following the precedent of `validate_dispersion_rank_pairs`, the loader must **fail closed**
on any of these. All are checked in the reference (`check_structural_invariants`).

**Per table (file-level):**

1. A `version` string is present and matches the compiled-in expected version
   (`partB-recoupling-1`). A table without a version, or with an unrecognised one, is rejected.
2. The declared component ordering equals the compiled-in 15-component L3 ordering, string for
   string (`10, 11c, 11s, 20, …, 33s`).
3. The declared convention block matches the compiled-in conventions (Racah normalisation, the
   real/complex transform, the `T` closed form, the reality phase, the sign of `E_disp`, and
   the fact that `M` carries the `1/(2 pi)`).
4. Every `g^r` is finite.

**Per `g^r` entry `(la, la', lb, lb', l1, l2, j)`:**

5. `la, la', lb, lb'` all in `{1, 2, 3}` (rank completeness — a missing rank is an error, not a
   zero contribution, inherited from the isotropic engine).
6. `n = la + la' + lb + lb' + 2`, and `6 <= n <= 14`.
7. `|la - la'| <= l1 <= la + la'` and `|lb - lb'| <= l2 <= lb + lb'`.
8. `|L1 - L2| <= j <= L1 + L2` **and** `|l1 - l2| <= j <= l1 + l2`, with `L1 = la + lb`,
   `L2 = la' + lb'`.
9. `(L1 + L2 + j)` is even.
10. `g^r != 0` (entries below the generation tolerance must be absent, not stored as zeros).
11. Entries are unique on `(la, la', lb, lb', l1, l2, j)`.

**Per label `(n, l1, k1, l2, k2, j)`:**

12. `0 <= k1 <= 2 l1` and `0 <= k2 <= 2 l2`.
13. `|l1 - l2| <= j <= l1 + l2`.
14. Every label is backed by at least one `g^r` entry whose `Pcheck` rows for `k1` and `k2` are
    non-zero — i.e. no label with an identically vanishing `W`.
15. Labels are sorted by `(n, l1, k1, l2, k2, j)` and unique.
16. The label set is **closed under `(l1 k1) <-> (l2 k2)` exchange**, and the loader verifies
    the exchange relation `g^r`-wise.

**Closure invariants the loader must *prove*, not merely assume (cheap, run at load):**

17. For each of the eight ordered rank pairs in `dispersion_rank_pairs()`, tracing
    `W[2(la+lb+1), 00, 00, 0]` over the diagonal blocks returns `binom(2la+2lb, 2la)` to
    `1e-12` relative. This is the isotropic-reduction check and costs microseconds.
18. `sum_{t,u} T^{(la,lb)}_{tu}(R)^2 * R^{2(la+lb+1)} == binom(2la+2lb, 2la)` for one canonical
    `R`, all nine rank pairs.
19. `S_{(0,0,0,0,0)} == 1` identically.

**Versioning.** The table is generated, never hand-edited. Regeneration must be reproducible
from the generator plus the version string alone; a table whose entries do not reproduce from
the recorded generator version is rejected rather than trusted.

---

## 8. Where this derivation disagrees with the spec

These are substantive and the spec should be amended before B1.

1. **`n` is not `2(la + lb + 1)` in general.** §B.3.3 says "with `n = 2 (l_a + l_b + 1)` fixed
   by the ranks involved". That is only true for the *diagonal* terms `la = la'`, `lb = lb'`.
   The general anisotropic expansion has

   ```
   n = la + la' + lb + lb' + 2
   ```

   because `T_{tu}` and `T_{t'u'}` may involve different ranks. For an L3 model this produces
   **`n = 6, 7, 8, 9, 10, 11, 12, 13, 14`** — including the **odd orders C7, C9, C11, C13**,
   which vanish on orientational averaging but are genuine anisotropic coefficients, and
   **C14**. [MS18] explicitly mentions CamCASP/Casimir producing `C7`, which corroborates this.
   The published output contract (§B.4) must therefore carry nine orders, not four.

2. **The `1/(2 pi)` is double-counted in §B.5 check 1.** §B.3.2 puts the `1/(2 pi)` inside `M`;
   §B.5 check 1 then asks `W` traced to equal `binom/(2 pi)`. With `M` as defined, `W` traced
   must equal **`binom`**, not `binom/(2 pi)` — otherwise `C_n` comes out `2 pi` times too
   small. This derivation uses `M` with the `1/(2 pi)` and `W` traced `= binom`, and the
   end-to-end agreement with `compute_dispersion` (`7.8e-16`) confirms that bookkeeping.

3. **The permutation sign is `(-1)^{l1 + l2}`, not a plain exchange.** §B.5 check 4 says
   "`C_n[A][B]` under `(l1 k1) <-> (l2 k2)` exchange must map onto `C_n[B][A]`, exactly". It
   does, but with the factor `(-1)^{l1 + l2}`. Writing the test without that factor makes it
   fail by a full sign on part of the table.

4. **§B.3.4 counts "five" `(la, lb)` pairs but lists eight.** The implemented table
   (`dispersion_rank_pairs()`) has eight *ordered* pairs over five distinct `binom` values.
   The norm identity was verified for all eight plus `(3,3)`.

5. **The output-contract sizing in §B.4 is optimistic.** For a full L3 model the label set is
   **29 762** (26 104 after pruning the symmetry-null entries), not "dozens". Option (b) — one
   `(npair, nlabel)` array — is still the right shape, but for an `N`-site molecule it is
   `N^2 x 26104` doubles (≈1.9 MB for three sites, ≈17 MB for nine). A resource plan gate in
   the style of `plan_dispersion` is required, and pruning to the symmetry-live labels (or to a
   caller-selected `n_max` / `l_max`) should be a first-class option rather than an afterthought.

6. **`(3,3)` and hence isotropic C14 exist but are not published today.** The C++ rank-pair
   table deliberately stops at `n <= 12`. The anisotropic table naturally contains `n = 14`,
   whose isotropic entry `C14[00 00 0]` is a legitimate, rank-complete L3 quantity. Whether to
   publish it is an output-contract decision, but it should be a decision rather than an
   accident.

7. **The blocker was overstated in one direction and understated in another.** §B.2 says the
   missing pieces are "the trivial generalisation of the Casimir–Polder integral" and "the
   recoupling table". Correct — but the recoupling table cannot be stored densely (12 GB); the
   deliverable is the *factorised* `(g^r, Pcheck)` pair, and the C++ interface should be shaped
   around that from the start.

---

## 9. What could not be verified

Stated plainly, as required.

1. **Stone's exact S-function normalisation and phase.** [S13] §3.3 and [S78] are behind
   paywalls; the publisher's table of contents, the errata list and the reference metadata were
   all obtained and checked, but the *equations* were not. The derivation therefore fixes its
   own convention, (4.3)+(4.4), and states it explicitly. Two consequences:
   * The `i^{l1 - l2 - j}` factor in Stone's definition is *reproduced* by this derivation as
     the phase required to make `S` real — that is a strong structural corroboration, but the
     residual real sign `±1` per `(l1, l2, j)`, and any `(2j+1)^{1/2}` normalisation Stone may
     carry, are **not** pinned down.
   * **Therefore the numerical values of `C_n[l1 k1, l2 k2, j]` produced here are not yet
     guaranteed to be comparable term-by-term with CamCASP/CASIMIR output.** Before task B10,
     the convention must be reconciled against the published definition (obtain [S13] §3.3 /
     [S78] / [ST84]), or fixed empirically by matching a small number of CASIMIR values whose
     sign and magnitude are unambiguous. **Nothing in §6 depends on this** — every check is
     convention-internal — but the external comparison does.
2. **The label ordering CamCASP itself publishes.** The ordering chosen here
   (`(n, l1, k1, l2, k2, j)`, `k` over `l0, l1c, l1s, …`) is a choice, stated in §5. It was not
   verified against CamCASP's output ordering (that would require reading CamCASP material,
   which the clean-room constraint forbids for source and which was avoided for documentation
   too).
3. **[S13] §4.3.4 specifically.** [MS18] pins the Casimir recoupling to §4.3.4, and §4.3 "The
   dispersion energy" (p. 64) was confirmed to exist, but the subsection breakdown of §4.3 is
   not in the published table of contents and could not be checked.
4. **[S78] page range.** Volume 36 and the first page 241 are confirmed via two independent
   listings; the closing page was not independently confirmed.
5. **The rank deficit** documented in §4.6 of the TODO is untouched here. This derivation is
   exact given an L3 model; it does not and cannot compensate for higher-rank blocks that are
   20–46 % low. §B.6's sequencing advice stands.

---

## 10. Reproducing this

```
python partB_reference.py                    # 22 checks, exits non-zero on failure
python partB_reference.py --dump table.json  # versioned table: 2723 g^r entries, 29762 labels
```

`partB_reference.py` requires only `numpy`. It contains no CamCASP, ORIENT, PFIT or CASIMIR
material; every equation is derived in this document from the published sources of §1, and the
two that were *fitted* rather than quoted — the `(-1)^{lb}` phase and `sqrt(binom(2L,2la))`
amplitude of (3.4) — were fitted against the independent construction (3.3), which itself uses
nothing beyond the Laplace expansion and the Racah addition theorem.
