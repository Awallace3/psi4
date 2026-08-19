# C-DF open questions — resolved

Research findings for **Part A §A.7 "Open questions"** and the **"Prerequisite: literature
verification"** section (C-DF only) of
[`2026-08-18-anisotropic-cn-and-cdf.md`](2026-08-18-anisotropic-cn-and-cdf.md).

Written 2026-08-18. Research document only — no production code, no tests, no
`.camcasp-reference/` access from anything that ships.

**Provenance rules observed.** Facts below come from three kinds of source and each is
labelled: (i) *our own recorded run record* under `.camcasp-reference/inputs/`,
`.camcasp-reference/logs/` and `.camcasp-reference/work/` — the same material
`devtools/camcasp_reference.py` already reads; (ii) *published literature* with DOIs; (iii)
*this repository's own source and a live Psi4 probe run*. No CamCASP or ORIENT source code
was read, and nothing here is derived from CamCASP's implementation — the algorithm is taken
from the published CamCASP User's Guide equations and from the Misquitta–Stone paper's
citation record.

---

## 1. Verdicts

| # | Question | Verdict |
| - | -------- | ------- |
| **1** | What does `DF with constraints` / `DF-TYPE-MONOMER NN` constrain? | **Not a hard linear constraint at all.** It is a *quadratic penalty* on the fitted charge, weight `λ`, plus a *localisation quadratic form* with weight `η` that modifies the metric. The reviewed run used `λ = 1.0`, `η = 0.0005`, inter-site type, `γ = 0`. `NN` = `FULL` = fit **all** MO pairs; it is a superset selector and does not change the coefficients for the occupied–virtual pairs the polarizability needs. The normal matrix is `J − η K_inter + λ q qᵀ`; the RHS for every `i ≠ a` pair is just `b`. |
| **2** | Is the `1e-4` gate reachable — is the 246-function Cartesian aux basis reproducible? | **(a) EXACTLY REPRODUCIBLE.** The reviewed aux basis *is* `aug-cc-pVTZ-RI` (Weigend/Köhn/Hättig), which Psi4 already ships as `psi4/share/psi4/basis/aug-cc-pvtz-ri.gbs`, token-for-token identical to CamCASP's file. The only difference is the `.gbs` header word: CamCASP used it **Cartesian**, Psi4's file says `spherical`. Changing line 1 to `cartesian` gives **nbf = 246, nshell = 56, puream = false** — verified live against this build. §A.8 task A8 does **not** degrade to a measured band on aux-basis grounds. |
| **3** | Cartesian vs spherical auxiliaries in Psi4 | **Supported, and verified live.** A Cartesian aux basis coexists with a spherical orbital basis; `FittingMetric` and the 3-index `(P\|μν)` ERI path are both puream-agnostic and were run successfully at 246×246 / 246×92×92. **Do not** use `IntegralFactory::ao_multipoles` against `zero_ao_basis_set()` — the one-body `IntegralFactory` path segfaults from Python in this build even for ordinary bases. Compute `Q_t[k]` **analytically** (§5) with an explicit Cartesian→real-solid-harmonic contraction; no `SolidHarmonic` reuse, because `solidharmonics.cc` is unit-normalised, not Racah. |
| **4** | Literature citations | **All verified with DOIs.** Misquitta & Stone, *J. Chem. Phys.* **124**(2), 024111 (2006), DOI `10.1063/1.2150828` — title, journal, volume, issue, article number and year all confirmed. Dunlap/Connolly/Sabin (1979) DOI `10.1063/1.438728`; Dunlap (2000) DOI `10.1039/B000027M`; Weigend/Köhn/Hättig (2002) DOI `10.1063/1.1445115`. The *equations* come from the CamCASP 6.0 User's Guide §9.4, quoted verbatim in §4.2. |
| **5** | Analytic multipole moments `Q_t[k]` | **Derived, closed form given, numerically validated to 1.6e-12** (target was 1e-8) across all 16 components for s/p/d/f/g Cartesian primitives, pure solid-harmonic Gaussians, and the general non-centred case. **The spec's "most vanish by parity" claim is wrong for a Cartesian aux basis:** 67 of the 246 reviewed aux functions carry non-zero rank-0 charge, not 19. d and g Cartesian components carry rank-0 and rank-2; f components carry rank-1 and rank-3. |

### The one consequence that dominates everything else

The spec's §A.3.1 — *hard* constraint `C d = n`, closed-form Lagrange elimination, "`Σ_k q_k
d_k^{ia} = N_ia`, which is exactly the constraint" — **does not describe the reviewed
calculation.** CamCASP imposes a *finite-weight penalty*, and the reviewed run used the
weakest possible setting, `λ = 1.0` (the CamCASP default is `1000.0`). Our own log records the
consequence directly: the fitted transition densities violate orthonormality by up to
`0.01065` in the block that produced the `DF_*` literals (`H2O.out:328`; the `0.00951` at
`:181` belongs to the *earlier* eta = 0.0 total-only block and is not the relevant number).
A hard-constrained implementation would produce a *different partition* and would
not reach `1e-4`, and §A.6 check 1 (molecular-sum conservation) would be asserted at a
precision the reference itself does not have.

---

## 2. Question 1 — what `DF with constraints` / `DF-TYPE-MONOMER NN` actually does

### 2.1 What the reviewed run selected, verbatim

Source: our own run record, `.camcasp-reference/work/H2O/H2O.cks` (sha256 recorded alongside
as `H2O.cks.sha256`; the identical file is echoed into
`.camcasp-reference/work/H2O/OUT/H2O.out`).

`H2O.cks` lines 63–99 — the settings in force for the **first** (rank-2, total-only)
polarizability block:

```
SET NEW-PROP
  Kernel ALDA
  C-DF  ( For unconstrained DF use: DF )
  ...
  SOLVER LU ( Options: GELSS )
END
SET PROPAGATOR
  Type CKS
  Hessians Internal
  DF with constraints
  DF-integrals
END

SET DF-INTEGRALS
  DF-TYPE-MONOMER NN
END
SET DF
  Solver LU (Options are LU and GELSS )
END

BEGIN DF
  Molecule H2O
  Type NN
  Eta    = 0.0
  Lambda = 1.0
  Print only normalization constraints
END
```

`H2O.cks` lines 113–153 — the settings in force for the **second** block, which is the one
that produced the distributed `DF_*` literals:

```
BEGIN DF
  Molecule H2O
  Type NN
  Eta    = 0.0005
  Lambda = 1.0
  Print only normalization constraints
END
SET NEW-PROP
  Kernel ALDA
  C-DF  ( For unconstrained DF use: DF )
  ...
END
SET PROPAGATOR
  Type CKS
  Hessians Internal
  DF with constraints
  DF-integrals
END
BEGIN Polarizability
  Molecule H2O
  Invert No
  Quad   10
  Spherical
  Rank  4
  Calculate only total and distributed polarizabilities
  Print only static total and distributed pols upto rank 1
  Print pols for Orient
  Pol-file H2O_0.0005_1000_f11_NL4
END
```

Note there is **no `CONSTRAINT` line** anywhere in `H2O.cks` and **no `GAMMA`**. Both take
their documented defaults: `CONSTRAINT` → `INTER-SITE`, `γ` → `0.0`.

The output filename `H2O_0.0005_1000_f11_NL4` encodes `η = 0.0005` and — misleadingly — a
`1000`, but the *actual* `LAMBDA` in the input is `1.0`. The filename token comes from the
`camcasp.py` driver's naming template, not from the value in force. **Use the `.cks` value.**

Recorded consequence, `.camcasp-reference/work/H2O/OUT/H2O.out`:

```
179: Tests of DF for H2O
180:  Summary of orthonormality failures:
181: Off-diagonal elements: maximum difference from 0: .00951      <- eta = 0.0
...
326: Tests of DF for H2O
328: Off-diagonal elements: maximum difference from 0: .01065      <- eta = 0.0005
```

This is decisive. If the charge condition were a hard Lagrange constraint, every off-diagonal
`∫ρ̃_ij dr` would be zero to machine precision. It is `~1e-2`. The condition is soft.

### 2.2 What the documentation says the functional is

Source: **CamCASP 6.0 User's Guide**, A. J. Misquitta and A. J. Stone,
<https://www-stone.ch.cam.ac.uk/documentation/camcasp/users_guide.pdf>, §9.4
"Density-fitting". Quoted verbatim (transcribed from the PDF text layer; `∬` rendered as the
double integral, subscripts flattened):

> Significant computational gains are achieved by expanding orbital products ρ_ij = φ_i φ_j as
> a single index expansion:
>
>     ρ_ij(r) ≈ ρ̃_ij(r) = Σ_k d_ij,k χ_k(r).
>
> This is usually achieved by the minimization of the functional:
>
>     Δ_ij = ∬ [ρ_ij(r1) − ρ̃_ij(r1)] (1/r12) [ρ_ij(r2) − ρ̃_ij(r2)] dr1 dr2
>
> In CamCASP we additionally include constraints to impose the orthonormality of ρ_ij :
>
>     ∫ ρ_ij(r) dr = δ_ij ,
>
> and localization constraints that are used to obtain distributed polarizabilities [Misquitta
> and Stone, 2006]. The latter are optional and should not be used in interaction energy
> calculations. This gives us two types of functional that we minimize:
>
> • In the first we include a term that minimizes the inter-site repulsion (the default):
>
>       Ξ^A_ij = Δ_ij − η Σ_{a,b≠a} E^ab_ij + λ ( ∫ ρ̃_ij(r) dr − δ_ij )²
>
> • In the second we include a term that maximizes the site self-repulsion:
>
>       Ξ^B_ij = Δ_ij + η Σ_a E^aa_ij + λ ( ∫ ρ̃_ij(r) dr − δ_ij )²
>
> where
>
>       E^ab_ij = ∬ ρ̃^a_ij(r1) ρ̃^b_ij(r2) (1/r12) dr1 dr2
>
> and ρ̃^a_ij(r) is the (transition-)density associated with site a which is defined as
>
>       ρ̃^a_ij(r) = Σ_{k ∈ a} d_ij,k χ_k(r).                                     (6)
>
> For details and numerical results see Misquitta & Stone (2006).
> In addition to the above, the following constraint can be optionally included:
>
>       C_ij = γ Σ_a [ ∫ ρ^a_ij(r) dr ]²                                          (7)
>
> This constraint forces the (transition-)density 'charge' on each site to be close to zero.

and the parameter definitions from the same section:

> **ETA [=] η** — η is the coefficient in front of the self-repulsion constraint term. Default
> = 0.0, i.e., no constraint. For distributed polarizabilities use η = +0.0005 with the
> INTER-SITE constraint. …
> **LAMBDA [=] λ** — λ applies to the (ortho)normality Lagrange multiplier. Default value 1000.0.
> **GAMMA [=] γ** — γ applies to the SITE self-repulsion constraint only. … Default value 0.0.
> **CONSTRAINT [type] [SITE | INTER-SITE] [repulsion | self-repulsion] [SRLO]** — Implement one
> of the two localization constraints described above. **The default is the INTER-SITE
> constraint.**

and, for `TYPE`:

> **TYPE [OO] [OV | VO] [FULL | NN] [RHO] [RHO-W]** — Specifies the type of density-fitting to
> perform. OO applies density-fitting only to occupied–occupied orbital pairs, needed for the
> density. OV does the occupied–virtual pairs, which are needed for all second-order terms.
> **FULL does all pairs, and is the default.**

and the propagator keyword:

> **[{CONSTRAINED-DF | C-DF} | DF]** — Choose the type of density-fitting used in calculating
> the propagator. **C-DF sets the density-fitting solution with constraints and DF uses the
> density-fitting solution without constraints.**

**So `NN` ≡ `FULL`** — fit every MO pair, not a special constraint mode. Because `Δ_ij`,
`E^ab_ij` and the normalisation penalty are all **per-pair**, the linear system for each
`(i,j)` is independent, and `NN` therefore yields *bit-identical* coefficients for the `(i,a)`
occupied–virtual pairs as `OV` would. **`NN` costs more and changes nothing we consume.**
Fit only the `(i,a)` pairs.

### 2.3 The resulting linear algebra (the deliverable)

Per transition pair `(i,j)`, with `naux` auxiliary functions:

| Symbol | Definition | Shape |
| ------ | ---------- | ----- |
| `d` | fit coefficients `d_ij,k` | `naux` |
| `J_kl` | `(χ_k ‖ χ_l)`, Coulomb metric | `naux × naux` |
| `b_k` | `(χ_k ‖ ρ_ij) = Σ_μν C_μi C_νj (χ_k \| μν)` | `naux` |
| `q_k` | `∫ χ_k dr = Q_00[k]` | `naux` |
| `A(k)` | the atom `χ_k` is centred on | — |
| `K^inter_kl` | `J_kl` if `A(k) ≠ A(l)`, else `0` | `naux × naux` |
| `K^self_kl` | `J_kl` if `A(k) = A(l)`, else `0` | `naux × naux` |
| `v_a` | `(v_a)_k = q_k · [A(k) = a]` | `naux` per site |

Note `J = K^self + K^inter` exactly. Then

```
Δ_ij(d)                = (ρ_ij‖ρ_ij) − 2 dᵀ b + dᵀ J d
Σ_{a,b≠a} E^ab_ij(d)   = dᵀ K^inter d
Σ_a E^aa_ij(d)         = dᵀ K^self  d
λ (∫ρ̃ − δ_ij)²         = λ (qᵀd − δ_ij)²
C_ij(d)                = γ Σ_a (v_aᵀ d)²
```

Stationarity of `Ξ^A_ij` gives the **inter-site (default, and what the reviewed run used)**
normal equations

```
    [ J − η K^inter + λ q qᵀ + γ Σ_a v_a v_aᵀ ] d^{ij}  =  b^{ij} + λ δ_ij q
```

and of `Ξ^B_ij` the **site self-repulsion** variant

```
    [ J + η K^self  + λ q qᵀ + γ Σ_a v_a v_aᵀ ] d^{ij}  =  b^{ij} + λ δ_ij q
```

**For the reviewed protocol, specialise to:** `Ξ^A`, `η = 0.0005`, `λ = 1.0`, `γ = 0`, and
`i ≠ j` (every pair we consume is occupied–virtual, so `δ_ij = 0`):

```
    [ J − 0.0005 · K^inter + 1.0 · q qᵀ ]  d^{ia}  =  b^{ia}
```

One matrix, factorised once, applied to all `n_ov` right-hand sides. CamCASP solved it with
`SOLVER LU` (`H2O.cks:90`), i.e. a plain LU factorisation with no rank truncation.

**Relation to the spec's `C d = n` form.** The hard-constraint form in §A.3.1 is the `λ → ∞`
limit of the single-row penalty. Implementing the *penalty* form therefore subsumes the spec's
form; implementing only the spec's form cannot represent the reviewed run. Recommend the
penalty form, with `λ = ∞` available as a documented special case that dispatches to the KKT
solve.

**Representation invariance (reassuring).** Every term above — `Δ`, `E^ab`, the normalisation
penalty, the site-charge penalty — is a functional of the *density* `ρ̃`, not of the
coefficient representation. Rescaling `χ_k → s χ_k` rescales `d_k → d_k/s` and `Q_t[k] → s
Q_t[k]`, leaving `B_DF` invariant. **CamCASP's and Psi4's Gaussian normalisation conventions
therefore do not need to agree.** Only the *span* must, and it does (§3).

### 2.4 Sign caveat on `η`

The Guide's prose says `Ξ^A` "minimizes the inter-site repulsion", but the displayed equation
has `Δ − η Σ E^ab` with the recommended `η = +0.0005`, which *rewards* inter-site repulsion.
Either the prose or the sign is loose. `Ξ^B`'s prose ("maximizes the site self-repulsion") and
sign (`+η Σ E^aa`) are mutually consistent, and to first order in `η` the two produce the same
solution up to scale (since `J = K^self + K^inter`), so the *intent* is unambiguous: push
density weight onto its own site. **Recorded as a known unknown (§8, item 1); resolve by
trying both signs — `η = 5e-4` is small enough that this is a ~0.05 % effect, and the sign
that reproduces `DF_*` is the answer. Attest whichever is used.**

---

## 3. Question 2 — auxiliary basis reproduction. **Verdict (a): exactly reproducible.**

### 3.1 What the reviewed run used

`.camcasp-reference/work/H2O/H2O.cks` lines 26–42:

```
   Basis Aux
        CARTESIAN
        Units Bohr
        Format TURBOMOLE
      O          8.0        0.00000000       0.00000000       0.00000000  TYPE O
        Limit G
        #include-camcasp basis/auxiliary/aug-cc-pVTZ/O
      ---
      H1         1.0       -1.45365196       0.00000000      -1.12168732  TYPE H
        Limit G
        #include-camcasp basis/auxiliary/aug-cc-pVTZ/H
      ---
      H2         1.0        1.45365196       0.00000000      -1.12168732  TYPE H
        Limit G
        #include-camcasp basis/auxiliary/aug-cc-pVTZ/H
      ---
   End
```

`.camcasp-reference/work/H2O/OUT/H2O.out` lines 81–92:

```
81:  Main basis: Size   =    92
82:              Shells =    32
83:              Type   = MC
87:              GTOs   = Spherical
88:  Aux  basis: Size   =   246
89:              Shells =    56
90:              Type   = MC
92:              GTOs   = Cartesian
```

**"MC" is not a basis family.** It is CamCASP's basis *placement* type — monomer-centred, as
opposed to `DC` (dimer-centred) or `MC+` (monomer plus mid-bond). The main basis is also `MC`.
`.camcasp-reference/logs/camcasp.log` records `main basis = aug-cc-pvtz, type = None`, i.e.
the placement type was defaulted. The spec's phrase "246-function Cartesian `MC`-type
auxiliary basis" should be read as "246-function Cartesian atom-centred auxiliary basis".

### 3.2 What that file actually is

`.camcasp-reference/tools/camcasp-runtime/basis/auxiliary/aug-cc-pVTZ/O`, header lines 1–11:

```
! o aug-cc-pVTZ
! o (9s7p6d4f2g)
...
! Ref.: Weigend, Köhn, Hättig, JCP 116 (2002) 3175.
```

and `.../aug-cc-pVTZ/H`:

```
! h aug-cc-pVTZ
! h     (5s4p3d2f)
...
! Ref.: Weigend, Köhn, Hättig, JCP 116 (2002) 3175.
```

That is the **RI-MP2 `aug-cc-pVTZ-RI` auxiliary basis**, which Psi4 ships as
`psi4/share/psi4/basis/aug-cc-pvtz-ri.gbs` and whose own header cites the same paper.

**Verified token-for-token identical.** A parse of both files into `(shell-am)` /
`(exponent, coefficient)` token streams (comments stripped) gives:

```
O len 56 56 IDENTICAL
H len 28 28 IDENTICAL
```

Same exponents, same digits, same shell ordering, same count. `Limit G` truncates nothing:
oxygen's highest shell is `g`, hydrogen's is `f`.

### 3.3 The function count arithmetic

| | shells | Cartesian nbf | spherical nbf |
| - | ------ | ------------- | ------------- |
| O `9s7p6d4f2g` | 28 | `9·1 + 7·3 + 6·6 + 4·10 + 2·15` = **136** | 106 |
| H `5s4p3d2f` | 14 | `5·1 + 4·3 + 3·6 + 2·10` = **55** | 46 |
| **H₂O total** | **56** | **246** | 198 |

`56` shells and `246` functions — exactly the reviewed run's `Shells = 56`, `Size = 246`.
The spherical form would have given 198. **The Cartesian reading is not incidental; it is a
32-function difference and it changes which auxiliary functions carry charge (§5.4).**

### 3.4 Verified live against this Psi4 build

The only change needed is line 1 of the `.gbs`, `spherical` → `cartesian`. With that file on
the basis path and no other change:

```
orbital basis: nbf=92  nshell=32 puream=True     <- CamCASP: Size 92,  Shells 32, Spherical
aux     basis: nbf=246 nshell=56 puream=False    <- CamCASP: Size 246, Shells 56, Cartesian
  aux centre 0 am counts {0: 9, 1: 7, 2: 6, 3: 4, 4: 2}
  aux centre 1 am counts {0: 5, 1: 4, 2: 3, 3: 2}
  aux centre 2 am counts {0: 5, 1: 4, 2: 3, 3: 2}
stock aug-cc-pvtz-ri (spherical): nbf=198 nshell=56 puream=True
```

(Run with `PYTHONPATH=build_camcasp/stage/lib`, `miniconda3/envs/p4_camcasp/bin/python`,
geometry exactly as `H2O.clt`, `no_reorient no_com symmetry c1`.)

**Both the orbital-basis and auxiliary-basis structural fingerprints of the reviewed run are
reproduced exactly.** §A.8 task A8 keeps its `rtol=1e-4, atol=1e-5` gate on aux-basis grounds.

### 3.5 What to ship

Add `psi4/share/psi4/basis/aug-cc-pvtz-ri-cart.gbs` (name to taste; see below), byte-identical
to `aug-cc-pvtz-ri.gbs` except that line 1 reads `cartesian`, with a provenance comment. Do
**not** use the global `PUREAM` option to achieve this: `export_mints.cc:98–105` gives
`PUREAM` precedence over both the per-file setting and the `puream=` build argument, and it is
a *global* keyword — setting it would flip the orbital basis to Cartesian too. Use either the
`cartesian`-header file or `BasisSet.build(mol, "DF_BASIS_...", name, puream=0)`.

Because §A.5 already requires the aux basis to be captured in a
`BasisSetStructuralSnapshot`, the snapshot must record `has_puream()` — otherwise the seal
cannot distinguish the 246-function run from the 198-function one, which is precisely the
distinction that matters.

---

## 4. Question 4 — literature verification

### 4.1 Verified citations

| Purpose | Citation | DOI | Verified how |
| ------- | -------- | --- | ------------ |
| **The C-DF distributed-polarizability algorithm** | A. J. Misquitta and A. J. Stone, "Distributed polarizabilities obtained using a constrained density-fitting algorithm", *J. Chem. Phys.* **124**(2), 024111 (2006) | `10.1063/1.2150828` | Title, authors, journal, volume **124**, issue **2**, article number **024111**, year **2006**, DOI and PubMed ID **16422575** all confirmed. The spec's guess was correct in every field. |
| **Robust/variational Coulomb-metric density fitting** | B. I. Dunlap, J. W. D. Connolly and J. R. Sabin, "On some approximations in applications of Xα theory", *J. Chem. Phys.* **71**(8), 3396–3402 (1979) | `10.1063/1.438728` | Confirmed: journal, volume 71, pages 3396–3402, year 1979. |
| **Robust fitting, modern statement** | B. I. Dunlap, "Robust and variational fitting", *Phys. Chem. Chem. Phys.* **2**(10), 2113–2116 (2000) | `10.1039/B000027M` | Confirmed: journal, volume 2, issue 10, pages 2113–2116, year 2000. |
| **The auxiliary basis** | F. Weigend, A. Köhn and C. Hättig, "Efficient use of the correlation consistent basis sets in resolution of the identity MP2 calculations", *J. Chem. Phys.* **116**(8), 3175–3183 (2002) | `10.1063/1.1445115` | Confirmed. Cited in the header of *both* `.camcasp-reference/tools/camcasp-runtime/basis/auxiliary/aug-cc-pVTZ/{O,H}` and `psi4/share/psi4/basis/aug-cc-pvtz-ri.gbs`. |
| **The equations** | A. J. Misquitta and A. J. Stone, *CamCASP 6.0 User's Guide*, §9.4 "Density-fitting"; §9.5 "Propagator settings"; §9.20 "Integrals" | <https://www-stone.ch.cam.ac.uk/documentation/camcasp/users_guide.pdf> | Quoted verbatim in §2.2. |
| **Corroborating description** | A. J. Misquitta and A. J. Stone, "Ab initio atom–atom potentials using CamCASP", arXiv:1512.06150 | — | Corroborates: "the charge density susceptibility is partitioned between atoms" via "a constrained density-fitting-based approach [48]" where [48] is the 2006 paper; and "The **DC+** form of the **RI-MP2 aug-cc-pVTZ auxiliary basis** [66, 67] **with Cartesian GTOs** was used for density-fitting". |
| **`SRLO` variant (not used here)** | F. Rob and K. Szalewicz (2013), self-repulsion + local orthogonality; selected in CamCASP by `CONSTRAINT SITE self-repulsion SRLO` with `γ = 1.0` | — | **Not** selected in `H2O.cks`. Listed only so the implementer does not confuse it with the default. |

The 2006 paper's full text is paywalled (AIP returns HTTP 403 to unauthenticated fetches); the
citation metadata was verified independently, and the **equations handed to the implementer
come from the User's Guide**, which is public and which the arXiv paper explicitly points to
for constraint details ("see the CamCASP User's Guide for details").

### 4.2 The equations to hand to the implementer, in plain text

**Robust Coulomb-metric density fitting (Dunlap et al. 1979; Dunlap 2000).** Expand the pair
density `ρ_ij = φ_i φ_j` in an atom-centred auxiliary set `{χ_k}`:

```
    ρ̃_ij(r) = Σ_k d_ij,k χ_k(r)
```

and minimise the Coulomb self-repulsion of the fit error

```
    Δ_ij = ( ρ_ij − ρ̃_ij ‖ ρ_ij − ρ̃_ij ),        (f‖g) = ∫∫ f(r1) g(r2) / |r1 − r2| dr1 dr2
```

whose stationary point is `J d = b`, with `J_kl = (χ_k‖χ_l)` and `b_k = (χ_k‖ρ_ij)`. This is
the metric that makes the resulting distributed polarizabilities well-conditioned; it is
*not* the overlap metric.

**CamCASP's constrained variant (Misquitta & Stone 2006; User's Guide §9.4).** Add a
localisation quadratic form and a normalisation penalty; minimise

```
    Ξ^A_ij = Δ_ij − η Σ_{a≠b} E^ab_ij + λ ( ∫ ρ̃_ij dr − δ_ij )²         [INTER-SITE, default]
    Ξ^B_ij = Δ_ij + η Σ_a     E^aa_ij + λ ( ∫ ρ̃_ij dr − δ_ij )²         [SITE self-repulsion]
    E^ab_ij = ( ρ̃^a_ij ‖ ρ̃^b_ij ),   ρ̃^a_ij(r) = Σ_{k ∈ a} d_ij,k χ_k(r)
```

optionally plus `C_ij = γ Σ_a ( ∫ ρ̃^a_ij dr )²`. The stationarity conditions are the two
normal-equation systems written out in §2.3.

**Site projection (unchanged from the spec).**

```
    Q_t^A[k] = ∫ χ_k(r) R_t(r − R_A) dr ,      t ∈ ranks 0..3,   A = A(k)
    B_DF[A, t, (ia)] = Σ_{k : A(k) = A} Q_t^A[k] d_k^{ia}
```

with `R_t` the **regular real solid harmonic in Racah normalisation** — the same convention as
`regular_harmonics` at `psi4/src/psi4/libmints/atomic_polarizability.cc:206–222`.

---

## 5. Question 5 — analytic multipole moments of a Gaussian auxiliary function

### 5.1 Convention

`R_t` is the regular real solid harmonic, Racah-normalised, in CamCASP component order
`00; 10 11c 11s; 20 21c 21s 22c 22s; 30 31c 31s 32c 32s 33c 33s`:

```
  R_00  = 1
  R_10  = z                  R_11c = x                   R_11s = y
  R_20  = (3z² − r²)/2       R_21c = √3 · xz             R_21s = √3 · yz
  R_22c = (√3/2)(x² − y²)    R_22s = √3 · xy
  R_30  = (5z³ − 3z r²)/2
  R_31c = √(3/8) · x(5z² − r²)          R_31s = √(3/8) · y(5z² − r²)
  R_32c = (√15/2) · z(x² − y²)          R_32s = √15 · xyz
  R_33c = (√10/4) · x(x² − 3y²)         R_33s = (√10/4) · y(3x² − y²)
```

This is **exactly** the table already in this repository at
`psi4/src/psi4/libmints/atomic_polarizability.cc:206–222` (`regular_harmonics`). Verified
term-by-term: e.g. the repo's `√10/4` for `33c` equals `√(5/8)` in the alternate spelling.
**Do not** reuse `psi4/src/psi4/libmints/solidharmonics.cc`'s `solidharmonic(l, Matrix&)` — it
generates *unit-normalised* real spherical harmonics, not Racah, so the coefficients differ by
an `l`-dependent factor.

### 5.2 The one-dimensional building block

```
    M(n; α) = ∫_{−∞}^{∞} x^n e^{−α x²} dx
            = 0                                    if n is odd
            = Γ((n+1)/2) · α^{−(n+1)/2}
            = (n−1)!! · (2α)^{−n/2} · √(π/α)       if n is even
```

### 5.3 Case A — Cartesian Gaussian centred **on** the site (the case that matters)

Let `χ_k(r) = c · x^a y^b z^c · e^{−α r²}` with the origin at `R_A` (so `R_k = R_A`), and
write `R_t = Σ_j γ_j x^{p_j} y^{q_j} z^{s_j}` using the table in §5.1. Then

```
    Q_t[k] = c · Σ_j γ_j · M(a + p_j; α) · M(b + q_j; α) · M(c + s_j; α)
```

**Non-zero iff `a+p_j`, `b+q_j`, `c+s_j` are all even for at least one term `j`.**

For a contracted shell, sum the same expression over primitives with their contraction
coefficients. Psi4's normalisation is folded in exactly by using `GaussianShell::coef(j)` for
`c` — see §5.6.

### 5.4 Which moments survive — corrected

The spec asserts "for a Gaussian centred exactly at `R_A`, most of these vanish by
parity/orthogonality". **That is true only for a *pure* (solid-harmonic) auxiliary basis.**
For the reviewed **Cartesian** basis it is substantially false, and the difference is the
whole reason CamCASP chose Cartesian.

*Pure solid-harmonic Gaussian* `χ_lm(r) = R_lm(r) e^{−α r²}` centred on `R_A`: **only the
single component `t = (l,m)` survives**, by orthogonality of the real spherical harmonics, with

```
    Q_lm = (2π/(2l+1)) · Γ(l + 3/2) · α^{−(l+3/2)}
         = π^{3/2} · (2l−1)!! / 2^l · α^{−(l+3/2)}          [identical, double-factorial form]
```

(check `l = 0`: `π^{3/2} α^{−3/2} = (π/α)^{3/2}` ✓.) All other 15 components vanish.

*Cartesian Gaussian* of total degree `L = a+b+c` centred on `R_A`: non-zero moments exist for
**every rank `l = L, L−2, L−4, …, ≥ 0`**, restricted within a rank to the components whose
monomials share the parity pattern of `(a,b,c)`. Measured directly on the reviewed
246-function basis:

| shell | distinct non-zero `Q_t` sets across the shell's Cartesian components |
| ----- | ------------------------------------------------------------------- |
| `s` (1 fn) | `(00)` |
| `p` (3 fns) | `(10)`, `(11c)`, `(11s)` |
| `d` (6 fns) | `(00, 20, 22c)`, `(00, 20)`, `(21c)`, `(21s)`, `(22s)` |
| `f` (10 fns) | `(10, 30, 32c)`, `(11c, 31c, 33c)`, `(11s, 31s, 33s)`, `(10, 30)`, `(11c, 31c)`, `(11s, 31s)`, `(32s)` |
| `g` (15 fns) | `(00, 20, 22c)`, `(00, 20)`, `(21c)`, `(21s)`, `(22s)` — rank 4 exists but is outside the L3 model |

**Rank-0 charge count for the reviewed basis: 67 of 246 auxiliary functions carry non-zero
`q_k`, not 19.** Breakdown: 19 `s` functions, 36 `d` components (`xx, yy, zz` in each of
12 `d` shells), 12 `g` components (`x⁴, y⁴, z⁴, x²y², x²z², y²z²` in each of oxygen's 2 `g`
shells). This is exactly the "Cartesian contaminant" content — a Cartesian `d` shell is
5 spherical `d` + 1 `s`, a Cartesian `g` shell is 9 `g` + 5 `d` + 1 `s`.

**Implementer consequence:** the charge-penalty vector `q` is *dense over 67 components*, so
the rank-1 update `λ q qᵀ` is not nearly diagonal, and the rank-0 projection `B_DF[A, 00, :]`
draws from `d`- and `g`-type coefficients as well as `s`. Any implementation that assumes
"only `s` functions carry charge" is wrong for this basis.

### 5.5 Case B — general non-centred Gaussian

For `χ_k` centred at `R_k ≠ R_A`, write `D = R_k − R_A` and expand `R_t` about `R_k` by the
binomial theorem:

```
    Q_t[k] = c · Σ_j γ_j  Σ_{n_x=0}^{p_j} Σ_{n_y=0}^{q_j} Σ_{n_z=0}^{s_j}
                  C(p_j, n_x) C(q_j, n_y) C(s_j, n_z)
                  · D_x^{p_j−n_x} D_y^{q_j−n_y} D_z^{s_j−n_z}
                  · M(a + n_x; α) M(b + n_y; α) M(c + n_z; α)
```

with `C(·,·)` the binomial coefficient. This reduces to §5.3 when `D = 0`. It is not needed
for the C-DF partition itself (every `χ_k` is projected onto its own atom) but *is* needed for
the §A.6 check-1 conservation invariant, which translates every site's moments to a common
origin.

### 5.6 Psi4's Cartesian shell normalisation — measured, not assumed

Psi4 stores one contraction coefficient per primitive per *shell*, shared by all Cartesian
components:

```
    χ_k(r) = coef(j) · (x−X)^a (y−Y)^b (z−Z)^c · e^{−α_j |r−R|²}
```

with `coef(j) = GaussianShell::coef(j)` normalised so that the `x^l`-type component has unit
self-overlap. Consequence: `S_kk = (2a−1)!!(2b−1)!!(2c−1)!! / (2l−1)!!`, so a Cartesian `g`
shell's `xxyz` component has `S_kk = 3/105 = 0.028571`. Measured on the reviewed basis:

```
aux overlap: (246, 246) diag min 0.028571 max 1.000000
shell  0 centre 0 am 0 exp 366.54531503 coef 59.7043324598  Sdiag=[1.]
shell 16 centre 0 am 2 exp   2.95526239 coef 10.9635837534  Sdiag=[1. 0.333333 0.333333 1. 0.333333 1.]
shell 22 centre 0 am 4 exp   2.32709649 coef 11.3546826256  Sdiag=[1. 0.142857 0.142857 0.085714 0.028571 ... 1.]
```

**Independent confirmation:** reconstructing the full `246 × 246` auxiliary overlap matrix from
that model with an Obara–Saika/Gaussian-product-theorem 1-D recursion and comparing against
Psi4's own `MintsHelper(aux).ao_overlap()`:

```
max |S_analytic − S_psi4| = 1.335e-15
```

So the closed forms in §5.3–5.5, evaluated with `GaussianShell::coef(j)` and
`GaussianShell::exp(j)`, are exactly consistent with Psi4's internal basis representation.

### 5.7 Numerical validation of the closed forms

Script: `scratchpad/moments.py` + driver (throwaway; not committed). Method: independent
product Gauss–Legendre quadrature on `[−9/√α, 9/√α]³` at 160–170 nodes per dimension —
genuinely numerical, not a re-derivation of the same algebra.

**Test 0 — the `R_t` polynomial table against the closed-form Stone definitions:** max
deviation `1.8e-15` over all 16 components at 7 random points.

**Test 1 — Cartesian primitives centred on `R_A`,** 12 cases (`s`, 2×`p`, 3×`d`, 3×`f`,
3×`g`) × all 16 components:

```
  chi = x^0 y^0 z^0 exp(-0.7 r^2)   [s]
      Q_00   analytic = +9.507749897101e+00   numeric = +9.507749897101e+00   |diff| = 9.24e-14
  chi = x^2 y^0 z^0 exp(-0.55 r^2)  [d_xx]
      Q_00   analytic = +1.241046601525e+01   numeric = +1.241046601525e+01   |diff| = 1.23e-13
      Q_20   analytic = -1.128224183205e+01   numeric = -1.128224183205e+01   |diff| = 1.14e-13
      Q_22c  analytic = +1.954141607639e+01   numeric = +1.954141607639e+01   |diff| = 1.92e-13
  chi = x^3 y^0 z^0 exp(-0.6 r^2)   [f_xxx]
      Q_11c  analytic = +2.496069629392e+01   numeric = +2.496069629392e+01   |diff| = 2.56e-13
      Q_31c  analytic = -2.547540397695e+01   numeric = -2.547540397695e+01   |diff| = 2.20e-13
      Q_33c  analytic = +3.288860511355e+01   numeric = +3.288860511355e+01   |diff| = 2.70e-13
  chi = x^4 y^0 z^0 exp(-0.5 r^2)   [g_xxxx]
      Q_00   analytic = +4.724882983717e+01   numeric = +4.724882983717e+01   |diff| = 4.76e-13
      Q_20   analytic = -9.449765967433e+01   numeric = -9.449765967433e+01   |diff| = 9.95e-13
      Q_22c  analytic = +1.636747477523e+02   numeric = +1.636747477523e+02   |diff| = 1.59e-12
  chi = x^2 y^1 z^1 exp(-0.95 r^2)  [g_xxyz]
      Q_21s  analytic = +1.518585419043e+00   numeric = +1.518585419043e+00   |diff| = 1.67e-14

  worst |analytic - numeric| over ALL 16 components x 12 cases = 1.592e-12
```

**Test 2 — pure solid-harmonic Gaussians `R_lm e^{−αr²}`,** `l = 0..3`, all 16 `m`:

```
  l=0 m=00   alpha=0.7 : Q_00  analytic=+9.507749897101e+00 numeric=+9.507749897102e+00 |d|=1.83e-13
  l=1 m=11c  alpha=1.2 : Q_11c analytic=+1.764987761257e+00 numeric=+1.764987761257e+00 |d|=3.46e-14
  l=2 m=22s  alpha=0.6 : Q_22s analytic=+2.496069629392e+01 numeric=+2.496069629392e+01 |d|=5.12e-13
  l=3 m=33s  alpha=0.85: Q_33s analytic=+2.169406391753e+01 numeric=+2.169406391753e+01 |d|=4.41e-13

  worst |diff| on the SURVIVING (t == lm) component      = 5.258e-13
  worst |numeric| on all VANISHING (t != lm) components  = 7.272e-15   <- confirms orthogonality
```

Both closed forms for `Q_lm` (`Γ` and `(2l−1)!!`) agree with each other to `1e-12` relative.

**Test 3 — general non-centred formula,** Gaussian on the reviewed H1 position
`(−1.45365196, 0, −1.12168732)`, moments about the O site at the origin, 4 shell types ×
16 components: worst `|analytic − numeric| = 2.174e-12`.

**Test 4 — reduction:** the non-centred formula with `R_k = R_A` reproduces the centred
formula to `1.4e-14`.

**Test 5 — the rank-0 identity `Q_00[k] = q_k = ∫χ_k dr`** used by the charge penalty: exact
agreement (`diff = 0.0`) for `s`, `d_xx`, `d_zz`, `g_xxxx`.

All well inside the requested `~1e-8`.

---

## 6. Question 3 — Psi4 code paths for a Cartesian auxiliary basis

Findings from reading `psi4/src/psi4/libmints/basisset.{h,cc}`, `gshell.h`, `integral.{h,cc}`,
`eribase.cc`, `onebody.cc`, `multipoles.{h,cc}`, `solidharmonics.cc`, `transform.cc`,
`lib3index/dftensor.h`, `lib3index/fittingmetric.cc`, `lib3index/dfhelper.{h,cc}`,
`export_mints.cc`, `driver/qcdb/libmintsbasissetparser.py`, `p4util/python_helpers.py`;
plus a live probe against `build_camcasp/stage`.

### 6.1 Per-basis `puream`

- `.gbs` line 1 is parsed by `Gaussian94BasisSetParser`
  (`psi4/driver/qcdb/libmintsbasissetparser.py:111–112, 170–183`); a bare `cartesian` /
  `spherical` line is file-global and sticky.
- Flows to C++ via `bsdict['puream']` (`libmintsbasisset.py:616, 744, 760`) →
  `construct_basisset_from_pydict` (`psi4/src/export_mints.cc:89–133`).
- **Precedence:** global `PUREAM` (if `has_changed()`) > the `puream=` build argument > the
  `.gbs` header (`export_mints.cc:98–105`). `PUREAM` is global — do not use it here.
- `BasisSet::puream_` is a per-object member (`basisset.h:176`), `has_puream()` at
  `basisset.h:295`. `MintsHelper::get_basisset(label)` (`mintshelper.h:167`,
  `mintshelper.cc:237–247`) is a plain map lookup with no puream consistency check.
- **A Cartesian aux and a spherical orbital basis coexist without issue** — confirmed live
  (§3.4).

### 6.2 Three-index ERIs — works, verified live

The spherical transform is applied **inside libint2, per shell**, driven by each shell's own
`pure` flag (`eribase.cc:220–246`, `basisset.cc:969`). Psi4 does no post-processing;
`TwoBodyAOInt::pure_transform` (`twobody.cc:850`) is correct per-shell but is dead code with
no callers. Zero-basis detection is structural (`eribase.cc:95–113`, `eri.cc:55–70`), keying
off `libint2::Shell::unit()`.

Live probe with the Cartesian aux and the spherical `aug-cc-pVTZ` orbital basis:

```
(P|mn) shape (246, 92, 92)  finite: True  max 7.724605448752139
```

Use `MintsHelper::ao_eri(aux, zero, orb, orb)` or the equivalent
`IntegralFactory(aux, zero, orb, orb).eri()` shell loop. **No Cartesian→spherical fixup is
needed anywhere in the ERI path.**

### 6.3 `FittingMetric` — works, and reveals a conditioning problem

`FittingMetric` is declared in `psi4/src/psi4/lib3index/dftensor.h:45–117` (there is no
`fittingmetric.h`), implemented in `lib3index/fittingmetric.cc`. Public API:

```cpp
FittingMetric(std::shared_ptr<BasisSet> aux, bool force_C1 = false);   // dftensor.h:75
void form_fitting_metric();          // :104  raw (A|B) — this is the one to use
void form_eig_inverse(double tol);   // :110  J^{-1/2} via Matrix::power(-0.5, tol)
void form_full_eig_inverse(double);  // :114  J^{-1}
void form_cholesky_inverse();        // :106
SharedMatrix get_metric() const;     // :92
std::string get_algorithm() const;   // :85
bool is_inverted() const;            // :89
```

It is entirely `nfunction()` / `function_index()`-based and **puream-agnostic**
(`fittingmetric.cc:105–152`). `DFHelper` (`lib3index/dfhelper.h:53`) is likewise
puream-agnostic (`naux_ = aux_->nbf()`), but is more machinery than this job needs.

**Live measurement on the reviewed basis** (`FittingMetric(aux, /*force_C1=*/true)` +
`form_fitting_metric()`):

```
J (bare Coulomb metric, Cartesian aux)  n=246  lam_min=+5.4657e-10 lam_max=1.0393e+03 cond=1.902e+12  (neg:0)
spherical aug-cc-pvtz-ri for contrast   n=198  lam_min=+5.3377e-05 lam_max=4.5222e+02 cond=8.472e+06
```

**The Cartesian metric is ~10⁵ worse conditioned than the spherical one** — the extra 48
contaminant functions are nearly linearly dependent on the rest. And with the reviewed
penalty terms applied:

```
J - 0.0000*K_inter +    1.0*q qT   cond=8.071e+12
J - 0.0005*K_inter +    1.0*q qT   cond=7.798e+12       <- THE REVIEWED PROTOCOL
J - 0.0000*K_inter + 1000.0*q qT   cond=5.977e+15
J - 0.0005*K_inter + 1000.0*q qT   cond=5.937e+15
J + 0.0005*K_self  +    1.0*q qT   cond=7.794e+12       (Xi^B variant, for contrast)
```

**Direct consequence for §A.5:** the proposed default `maximum_condition_number{1.0e12}` would
**fail closed on the exact calculation we are trying to reproduce** (`7.8e12`). And
`metric_relative_cutoff{1.0e-10}` relative to `λ_max ≈ 4.4e3` is a cutoff at `4.4e-7`, which
would discard the entire low end of the spectrum (`λ_min ≈ 5.7e-10`) — 30-odd directions.
CamCASP used `SOLVER LU`, i.e. **no truncation at all**. To reproduce the `DF_*` literals at
`1e-4`, start with an LU/Cholesky solve of the full system and treat truncation as a
diagnostic, not a default. Raise `maximum_condition_number` to `1e14` and set
`metric_relative_cutoff` to something like `1e-14`, and record both in the provenance seal.

The lesson recorded in `solve_constrained_least_squares` — that the cutoff must be *relative*
— still holds and is reinforced: an absolute cutoff of `1e-10` here would be at the very
bottom of the spectrum for the Cartesian basis and nowhere near it for the spherical one.

### 6.4 Multipole moments — **do not use `ao_multipoles` against the zero basis**

`IntegralFactory::ao_multipoles(int order, int deriv)` (`integral.h:488`, `integral.cc:211`)
builds a `MultipoleInt` (`multipoles.h:42–76`). It computes **Cartesian monomial moments**
`x^a y^b z^c`, not solid harmonics, in CCA lexicographic order **starting at rank 1** (rank 0
is excluded; `multipoles.cc:48–52` throws for `order == 0`), with an **electronic sign
convention** `prefac = −1.0 · ca · cb` (`multipoles.cc:183`). Chunk order for `order = 3`:

```
X, Y, Z | XX, XY, XZ, YY, YZ, ZZ | XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ
```

The per-shell transform (`multipoles.cc:204` → `onebody.cc:186–195`) is correctly keyed off
each shell's own `pure` flag, so mixed Cartesian/spherical is fine *in principle*.

**In practice this path is unusable from the Python layer in this build, and untested in C++.**
A live probe segfaults at `IntegralFactory(bs1, bs2, bs3, bs4).ao_overlap()` — *before* any
`compute_shell` call, and *for an ordinary spherical orbital basis with no zero basis
involved*:

```
A: bases built
B: factory
<segmentation fault>            # f.ao_overlap() never returned
```

This reproduces for `ao_overlap` and `ao_multipoles`, with `(orb,orb,orb,orb)`,
`(aux,aux,aux,aux)` and `(aux,zero,zero,zero)`. It is a pre-existing binding defect unrelated
to Cartesian auxiliaries, but it means the "`ao_multipoles` against `zero_ao_basis_set()`"
route sketched in §A.4 of the spec has **never been exercised** and cannot be validated
cheaply. No shipped Psi4 code computes a one-body integral against the zero basis.

**Recommendation: implement `auxiliary_multipole_moments` analytically** from §5.3/§5.5,
reading `GaussianShell::exp(j)`, `GaussianShell::coef(j)`, `am()`, `ncenter()`,
`function_index()`, `nfunction()`. This is:

- exact, with the closed form already validated to `1.6e-12` (§5.7);
- consistent with Psi4's normalisation, validated to `1.3e-15` against `ao_overlap` (§5.6);
- free of the segfaulting code path;
- and cheap — 246 × 16 numbers.

### 6.5 Solid-harmonic transform

`psi4/src/psi4/libmints/solidharmonics.cc` provides `solidharmonic(int l, Matrix& coefmat)`
(`:224`/`:226`, dispatching on the `psi4_SHGSHELL_ORDERING` macro; this build is
`gaussian` = 2, i.e. within-shell order `m = 0, +1, −1, +2, −2, …`, which coincidentally
matches the CamCASP `l0, l1c, l1s, l2c, l2s, …` shape). **But its normalisation is
unit-normalised real spherical harmonics, not Racah** (`solidharmonics.cc:152–163`), so it is
*not* directly reusable for `R_t`. Use the explicit 16-term table already in
`atomic_polarizability.cc:206–222`, extended into the coefficient list of §5.1 — i.e. an
explicit Cartesian→real-solid-harmonic contraction, hard-coded and unit-tested. **Yes, an
explicit transform is needed; no, it should not come from `solidharmonics.cc`.**

---

## 7. Handoff to the implementer

### 7.1 Constraint / penalty specification — exact

Replace §A.3.1's `C d = n` with the following. Per occupied–virtual pair `(i,a)`:

```
    M  =  J  −  η · K^inter  +  λ · q qᵀ  +  γ · Σ_A v_A v_Aᵀ
    M · d^{ia}  =  b^{ia}                       (RHS has no λ term, because δ_ia = 0)
```

| Object | How to build | Reviewed value |
| ------ | ------------ | -------------- |
| `J` | `FittingMetric(aux, true).form_fitting_metric(); get_metric()` | 246×246 |
| `b^{ia}` | `Σ_μν C_μi C_νa (χ_k\|μν)`, from `MintsHelper::ao_eri(aux, zero, orb, orb)` | 246 × n_ov |
| `q` | `Q_00[:]` from `auxiliary_multipole_moments` (§7.3) | 67 non-zero of 246 |
| `K^inter` | `J` masked to `A(k) ≠ A(l)` | — |
| `K^self` | `J` masked to `A(k) = A(l)`; `J = K^self + K^inter` | — |
| `v_A` | `(v_A)_k = q_k · [A(k) = A]` | unused (`γ = 0`) |
| `η` | `CDFOptions::eta` | **`0.0005`** |
| `λ` | `CDFOptions::lambda` | **`1.0`** (not the CamCASP default `1000.0`) |
| `γ` | `CDFOptions::gamma` | **`0.0`** |
| variant | inter-site (`Ξ^A`) vs site self-repulsion (`Ξ^B`) | **inter-site** (CamCASP default; no `CONSTRAINT` line in `H2O.cks`) |
| solver | LU/Cholesky of the full `M`, no truncation | CamCASP `SOLVER LU` |

Keep §A.5's general `C d = n` interface as the `λ → ∞` special case, so the two forms live in
one function. Do **not** make the hard-constraint form the default; it is not what the oracle
did.

Diagnostics to report and gate (revising §A.6 check 2): per pair, the Coulomb residual `Δ`,
**and** `qᵀ d^{ia}`, which should be *small but non-zero* — the reference's own maximum is
`0.0095` (`H2O.out:181`) / `0.01065` (`:328`). Gating it at machine zero would be gating
against the wrong model.

Revise §A.6 check 1 accordingly: molecular-sum conservation for C-DF is exact only in the
`λ → ∞` limit. At `λ = 1.0` it holds to roughly the same `~1e-2` scale. **Assert the invariant
with a tolerance derived from the measured `max_ia |qᵀd^{ia}|`, not at ISA precision.**

### 7.2 Auxiliary basis — exact

- Ship `psi4/share/psi4/basis/aug-cc-pvtz-ri-cart.gbs`: byte-identical to the existing
  `aug-cc-pvtz-ri.gbs` except line 1 = `cartesian`, plus a provenance comment naming
  Weigend/Köhn/Hättig DOI `10.1063/1.1445115` and recording that CamCASP's
  `basis/auxiliary/aug-cc-pVTZ/{O,H}` files are token-identical.
- Select it with `ATOMIC_POLARIZABILITY_CDF_AUX_BASIS = aug-cc-pvtz-ri-cart`.
- Build with `BasisSet::build(mol, "DF_BASIS_...", label)`; **never** touch the global
  `PUREAM` option.
- Expected structural fingerprint, to assert in a test:
  `nbf() == 246`, `nshell() == 56`, `has_puream() == false`; per-centre shell counts
  `O {s:9, p:7, d:6, f:4, g:2}`, `H {s:5, p:4, d:3, f:2}`.
- Extend `BasisSetStructuralSnapshot` to record `has_puream()`. Without it the seal cannot
  distinguish 246 from 198.

### 7.3 Moment formulas — exact, with their validated checks

`auxiliary_multipole_moments(const BasisSet& aux, sites, function_to_site)` returns
`naux × 16` in CamCASP component order. For each shell `(centre A, am L, primitives
{α_j, c_j})` and each Cartesian component `(a,b,c)` with `a+b+c = L`:

```
    Q_t[k] = Σ_j c_j · Σ_m γ^{(t)}_m · M(a + p_m; α_j) · M(b + q_m; α_j) · M(c + s_m; α_j)

    M(n; α) = 0                              if n odd
            = (n−1)!! · (2α)^{−n/2} · √(π/α)  if n even
```

with `{(γ_m, p_m, q_m, s_m)}` the monomial expansion of `R_t` from §5.1. Use
`GaussianShell::coef(j)` for `c_j` (it carries Psi4's normalisation) and `exp(j)` for `α_j`.
The general non-centred form is §5.5, needed only for the conservation invariant.

Required unit tests, with the validated reference numbers from §5.7 (targets given at `1e-10`,
achieved `1.6e-12`):

1. `α = 0.7`, `s`: `Q_00 = 9.507749897101e+00`, all other 15 components zero.
2. `α = 0.55`, `d_xx`: `Q_00 = 1.241046601525e+01`, `Q_20 = −1.128224183205e+01`,
   `Q_22c = 1.954141607639e+01`, rest zero. **This is the Cartesian-contaminant test — it
   fails for a pure basis and is the one that catches a spherical/Cartesian mix-up.**
3. `α = 0.6`, `f_xxx`: `Q_11c = 2.496069629392e+01`, `Q_31c = −2.547540397695e+01`,
   `Q_33c = 3.288860511355e+01`, rest zero.
4. `α = 0.5`, `g_xxxx`: `Q_00 = 4.724882983717e+01`, `Q_20 = −9.449765967433e+01`,
   `Q_22c = 1.636747477523e+02`, rest zero (within ranks 0–3).
5. `α = 0.95`, `g_xxyz`: only `Q_21s = 1.518585419043e+00` non-zero.
6. Pure solid-harmonic Gaussian, all `l = 0..3`: only the matching `(l,m)` survives, value
   `π^{3/2}(2l−1)!!/2^l · α^{−(l+3/2)}`. (`l=0, α=0.7` → `9.507749897101e+00`.)
7. `Q_00[k] == ∫χ_k dr` for every `k`. On the reviewed 246-function basis, exactly **67** of
   246 must be non-zero — a strong, cheap structural assertion.
8. Frame covariance under rotation, as §A.8 A1 already requires.

### 7.4 Psi4 API calls — exact

```cpp
// aux basis (Cartesian), attached to the frozen context and snapshotted
auto aux  = mints->get_basisset("DF_BASIS_ATOMIC_POLARIZABILITY");   // has_puream() == false
auto zero = BasisSet::zero_ao_basis_set();

// Coulomb metric J (naux x naux) -- puream-agnostic, force C1
FittingMetric metric(aux, /*force_C1=*/true);
metric.form_fitting_metric();
SharedMatrix J = metric.get_metric();          // lib3index/dftensor.h:75,104,92

// three-index (P|mu nu) -- Cartesian aux, spherical orbital: handled inside libint2
auto Pmn = mints->ao_eri(aux, zero, orb, orb); // or IntegralFactory(aux,zero,orb,orb).eri()

// moments Q_t[k]: ANALYTIC (do NOT use IntegralFactory::ao_multipoles against zero basis)
Matrix Q = detail::auxiliary_multipole_moments(*aux, sites, function_to_site);

// solve: LU/Cholesky of M = J - eta*K_inter + lambda*q q^T, no truncation by default
```

Do **not** call `IntegralFactory::ao_multipoles(order)` / `ao_overlap()` on a
multi-basis `IntegralFactory` — it segfaults in this build (§6.4). Do **not** reuse
`solidharmonic()` from `solidharmonics.cc` — wrong normalisation (§6.5).

`CDFOptions` should become:

```cpp
struct PSI_API CDFOptions {
    std::string auxiliary_basis;                 // "aug-cc-pvtz-ri-cart"
    CDFLocalisation localisation{CDFLocalisation::InterSite};  // Xi^A (CamCASP default)
    double eta{0.0005};                          // localisation weight
    double lambda{1.0};                          // normalisation PENALTY weight (not a multiplier)
    double gamma{0.0};                           // optional site-charge penalty
    double metric_relative_cutoff{1.0e-14};      // relative; 1e-10 discards ~30 directions here
    double maximum_condition_number{1.0e14};     // 1e12 rejects the reviewed protocol (7.8e12)
    bool   hard_charge_constraint{false};        // lambda -> infinity special case
};
```

---

## 8. What remains unknown

1. **The sign of the `η` term.** The User's Guide's `Ξ^A` prose and equation disagree (§2.4).
   The 2006 paper is paywalled and was not obtained. `η = 5e-4` makes this a sub-0.1 % effect,
   but it is a real ambiguity. **Resolve empirically against `DF_*` and attest the choice.**
   Getting the paper (institutional access, or an author preprint) would settle it outright,
   and is worth ~15 minutes.

2. **The exact meaning of `LAMBDA` inside CamCASP.** The Guide's prose calls it "the
   (ortho)normality **Lagrange multiplier**" while the displayed equation uses it as a
   quadratic **penalty weight** `λ(∫ρ̃ − δ)²`. These are different objects. The recorded
   orthonormality violation of `0.01065` proves it behaves as a finite penalty (a true Lagrange
   multiplier would give exact satisfaction), so the equation is the operative reading — but
   whether CamCASP's internal `λ` is applied as written, or scaled, is not established from
   public sources. **Mitigation: `λ` is one scalar. If `1.0` does not reproduce `DF_*`, sweep
   it and report the value that does, as a measured quantity.**

3. **CamCASP density-fitted the response matrix itself.** `H2O.cks:83` sets `DF-integrals` in
   `SET PROPAGATOR`, and User's Guide §9.20 states the integral routine "uses the
   density-fitting solution **without** any constraints" by default — no `DF-CONSTRAINTS` line
   appears in the reviewed `SET DF-INTEGRALS` block. So the two-electron integrals entering
   CamCASP's CKS Hessians are themselves DF-approximated in the 246-function Cartesian basis,
   whereas Psi4 computes `G(iω)` from exact integrals. **The spec's §A.2 claim that the two
   routes "share everything except how the FDDS is distributed over sites" is not strictly
   true.** This is an unquantified difference upstream of the partition. It may well be below
   `1e-4`; it has not been measured. Measure it before attributing any residual to the
   partition — e.g. by computing the molecular (undistributed) polarizability both ways, which
   `H2O.out` already reports (`1 by 1` block, isotropic `9.621348`, at `η = 0`; `9.585151` at
   `η = 5e-4`).

4. **`ALDA` vs `ALDA+CHF`.** `H2O.clt:21` requests `Kernel ALDA+CHF`, but the generated
   `H2O.cks:64,121` says `Kernel ALDA` inside `SET NEW-PROP`, and User's Guide §9.5 notes that
   `NEW-PROP` is for non-hybrid kernels while the older `PROPAGATOR` module handles
   `ALDA+CHF` — and both `SET NEW-PROP` and `SET PROPAGATOR ... Kernel` blocks are present.
   Which kernel was actually in force for the distributed run is **not established** from the
   log. This is a pre-existing question about the reviewed protocol, not new to C-DF, but it
   bears on any `1e-4` claim. (Our pipeline uses a 25/75 kernel; if CamCASP used pure ALDA the
   comparison is not like-for-like.)

5. **The one-body zero-basis integral path is untested and currently crashes from Python**
   (§6.4). The analytic route sidesteps it, so this is not a blocker — but if anyone later
   wants `ao_multipoles` against `zero_ao_basis_set()`, that segfault must be fixed first, and
   it reproduces for ordinary spherical bases too, so it is an independent latent defect worth
   filing.

6. **Whether `1e-4` is actually reachable** — this document establishes that the *aux basis*
   is not the obstacle, and that the constraint algebra is now fully specified. It does **not**
   establish that the reviewed `DF_*` literals will be reproduced. Items 1–4 are each capable
   of costing more than `1e-4`. Per §A.8 task A8: if the comparison misses, record the
   measured deviation per component and open a stage-invariant investigation. **Do not widen
   the gate.**

7. **`solidharmonics.cc` normalisation was read, not measured.** The claim that it is
   unit-normalised rather than Racah rests on reading `solidharmonics.cc:152–163`. Since the
   recommendation is not to use it, this is low-risk, but if anyone does use it, verify
   numerically first.

---

## Appendix — evidence index

| Claim | Evidence |
| ----- | -------- |
| Reviewed DF settings `η=0.0005, λ=1.0, γ=0, Type NN, inter-site` | `.camcasp-reference/work/H2O/H2O.cks:86–99, 113–119` |
| `C-DF` and `DF with constraints` selected | `.camcasp-reference/work/H2O/H2O.cks:65, 82, 122, 139` |
| `SOLVER LU` for the DF equations | `.camcasp-reference/work/H2O/H2O.cks:90` |
| Constraints are soft (`0.0095` / `0.0107` violation) | `.camcasp-reference/work/H2O/OUT/H2O.out:181, 328` |
| Aux basis = `aug-cc-pVTZ` RI file, Cartesian, `Limit G` | `.camcasp-reference/work/H2O/H2O.cks:26–42` |
| `Size = 246`, `Shells = 56`, `Cartesian`; main `92 / 32 / Spherical` | `.camcasp-reference/work/H2O/OUT/H2O.out:81–92` |
| Aux basis is Weigend/Köhn/Hättig | `.camcasp-reference/tools/camcasp-runtime/basis/auxiliary/aug-cc-pVTZ/{O,H}:1–11` |
| Token-identical to Psi4's `aug-cc-pvtz-ri.gbs` | programmatic token diff: `O 56/56 IDENTICAL`, `H 28/28 IDENTICAL` |
| `nbf = 246`, `nshell = 56`, `puream = false` in Psi4 | live probe, `build_camcasp/stage` |
| `J` condition number `1.9e12` bare, `7.8e12` with penalties | live `FittingMetric` probe |
| `(P\|μν)` works with Cartesian aux + spherical orbital | live probe, shape `(246, 92, 92)`, finite |
| `IntegralFactory::ao_overlap()` segfaults | live probe, 4 configurations |
| Psi4 shell normalisation model exact to `1.3e-15` | analytic `S` vs `MintsHelper(aux).ao_overlap()` |
| Moment closed forms exact to `1.6e-12` | Gauss–Legendre quadrature, 5 test batteries |
| 67 of 246 aux functions carry charge | analytic `Q_00` over the reviewed basis |
| CamCASP functional equations | CamCASP 6.0 User's Guide §9.4 |
| Project's Racah `R_t` convention | `psi4/src/psi4/libmints/atomic_polarizability.cc:206–222` |
| `solidharmonic()` is unit-normalised, not Racah | `psi4/src/psi4/libmints/solidharmonics.cc:152–163, 224` |
| `PUREAM` precedence | `psi4/src/export_mints.cc:98–105` |
| `FittingMetric` API | `psi4/src/psi4/lib3index/dftensor.h:45–117` |
| `MultipoleInt` ordering / sign | `psi4/src/psi4/libmints/multipoles.cc:48–52, 126–129, 183`; `mcmurchiedavidson.cc:34–43` |

---

## Appendix Z — two verification notes added on review

1. **The `1000` in the pol-file name is a red herring.** The reviewed distributed block writes
   `Pol-file H2O_0.0005_1000_f11_NL4` (`H2O.cks:151`), and the `0.0005` in that name does match
   `Eta`. The `1000` does **not** match `Lambda`. CamCASP's own echo of the parsed input
   (`OUT/H2O.out:321-322`) records `Eta = 0.0005` / `Lambda = 1.0`, so `lambda = 1.0` is what
   ran and the `1000` is a stale user-chosen label, most likely carried over from CamCASP's
   default `lambda = 1000.0`. Do not "correct" the implementation to `lambda = 1000` on the
   strength of the filename.

2. **Which orthonormality number to gate against.** There are two `Tests of DF` blocks in
   `OUT/H2O.out`. The first (`:181`, max off-diagonal `0.00951`) belongs to the `Eta = 0.0`,
   rank-2, total-only polarizability block. The second (`:328`, max off-diagonal `0.01065`) is
   the `Eta = 0.0005` block that produced the distributed `DF_*` literals. **`0.01065` is the
   number that bounds §A.6 check 1.**
