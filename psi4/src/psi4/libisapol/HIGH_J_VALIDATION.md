# Independent maximal-J validation: J9/C11 and J10/C12

## Status and precise claim

This source-only increment after `f996942ad6` adds a factorial/electrostatic
oracle for **all seven structurally present high-J blocks** in the rank<=3,
C6..C12 API: four J9/C11 blocks and three J10/C12 blocks, 735 component rows.
Expected numbers do not come from either shipped table, the archive, or an
execution of the production recoupling engine.

Parent integrated acceptance PASSED: **1755 ISA/FDDS tests in24.71s**, including
14 new high-J tests and31 existing recoupled tests, against the accepted staged
core. This is a test-only increment, not a fresh-build or performance claim.
Evidence: `.pi/audit/high-j-parent-regressions-v2.log`. Independent source review
found no scientific must-fix; parent verified the cited dissertation's angular
normalization directly. A completeness regression ensures the isolation cases
cover all26 ordered quadruples and11 distinct reciprocal classes. Zero-weight
nodes are excluded before CP multiplication, not before first-stage coupling.
Earlier development counts (13 tests/1.48s and9 factorial tests/1.28s) are
superseded by the parent's full regression result.

The literal water archive's 411 visible high-J rows remain **not individually
archive-validated**. This new check is independent numerical validation on
synthetic nonidentical, anisotropic, mixed-rank SPD inputs, not a change to the
archive's accepted tolerances or a claim about native SCF/PFIT/GRAC parity.

## Sources, conventions, and attribution

1. Allowed CamCASP `docs/casimir_manual.tex`, theory section (lines 85–150),
   describes the real coupled response, unitary CG transform, CP integral,
   and second coupling; it refers to Stone, *The Theory of Intermolecular
   Forces*, section 4.3.4. `src/casimir/casimir.f90:395–435` specifies the CP
   weight and phase conventions. These describe the interface but do not
   give a sufficient explicit angular normalization on their own.
2. Richard Hatz, *Computational Studies of Dispersion Interactions in Coinage
   and Volatile Metal Clusters*, University of Helsinki dissertation (2016),
   ISBN 978-951-51-1909-4, printed pp. 23–24, equations (4.3)–(4.6), gives the
   normalized angular functions, **including** `i^(L-H-J)` and division by
   the zero-m 3j symbol. Public source:
   <https://helda.helsinki.fi/server/api/core/bitstreams/473f3918-0d2e-4005-9918-ae5ac9ad2279/content>.
   This is a mathematical reference, not executable code or numerical data
   copied into the tests. The dissertation is not attributed an MIT license.
3. `ANISOTROPIC_CONTRACT.md` defines the independent electrostatic starting
   point: real Racah harmonics, `(-1)^l H_l(grad) H_k(grad)/(d_l d_k R)`,
   axial `tau_l0,k0=(-1)^k binomial(l+k,l)`, and the ordered response
   contraction with weights already including `1/(2*pi)`.
4. The existing independent factorial Wigner/real-complex helpers were moved
   **unchanged** to `tests/pytests/isapol_factorial_oracle.py`. AST identity
   with the three original functions was checked. The old test module imports
   them; its existing tests and tolerances are otherwise unchanged.

CamCASP conventions are due to **Alston J. Misquitta and Anthony J. Stone**,
Copyright (c) 2019 Anthony Stone, MIT; the complete permission and warranty
notice remains in [RECOUPLED_CAMCASP_LICENSE](RECOUPLED_CAMCASP_LICENSE).
The new test implementation/derivation is not a numerical table transcode.
Psi4 additions are LGPL-3.0-only. No ORIENT source was consulted. A web search
returned an ORIENT manual link; it was not opened or used.

## Derivation

Use ordered multipole ranks `(l,p,k,q)=(la,lap,lb,lbp)` and define

```
K = l+k,  P = p+q,  L = l+p,  H = k+q,  n = l+p+k+q+2.
```

All complex CG coefficients below use the Condon–Shortley convention. For
real components use the existing no-CS Racah transform `U`, with rank1
ordered **z,x,y**. The coupled response is

```
aA[L,t](l,p) = sum_ab (U_L CG(l,p;L) U_l* U_p*)[t,a,b] A[a,b]
```

and similarly for B. This is a **bilinear** product of responses, not a
Hermitian product. The input offsets follow the explicitly declared rank
subset. No component is copied from an `IsaRecoupledModel` into the oracle.

### 1. Electrostatic radial factor

Rotational covariance and the harmonic derivative in the electrostatic
contract put a rank-l/rank-k interaction in the single irreducible rank
`K=l+k`. Its reduced factor is

```
F(l,k) = sqrt((2K)! / ((2l)! (2k)!)) = sqrt(binomial(2K,2l)).
```

The signed interaction is `(-1)^k F(l,k)` times the scalar contraction
of `[Q_l x Q_k]^K` with `C_K(Rhat)`. The sign and scale are fixed without
using a dispersion table: on positive z the coefficient of `Q_l0 Q_k0` is

```
(-1)^k F(l,k) <l0,k0|K0> = (-1)^k binomial(K,l),
```

exactly the axial derivative. The identity follows from the stretched CG
formula

```
<a m,b s|a+b,m+s>
  = sqrt(binomial(2a,a+m) binomial(2b,b+s)
         / binomial(2a+2b,a+b+m+s)).
```

Thus the ordered product of two interactions carries the signed radial
factor `(-1)^(k+q) F(l,k) F(p,q)`.

### 2. Product harmonics and the stretched recoupling overlap

The product of Racah harmonics of ranks K and P has a rank-J component with
factor `<K0,P0|J0>`. This is the usual CG product theorem, including its
normalization. For maximal `J=K+P=n-2`, triangle inequalities force

```
L=l+p, H=k+q, J=L+H.
```

There are no non-stretched local L/H channels at this J. The recoupling
overlap between

```
[(l k)K (p q)P]J   and   [(l p)L (k q)H]J
```

is **+1**. A direct proof avoids a tabulated 9j: at `M=J` both normalized
states are the same product of the four highest-weight states and every
CG is +1. Rotational invariance makes the overlap independent of M.
Equivalently the usual `sqrt((2K+1)(2P+1)(2L+1)(2H+1))` times the associated
9j is one. These are distinguishable tensor slots, not fermions; swapping
the middle slots introduces no exchange sign.

Before choosing the normalized angular basis, the rank-J contraction thus
has factor

```
(-1)^H F(l,k) F(p,q) <K0,P0|J0>.
```

### 3. Angular reconstruction: phase and normalization matter

For complex local components t,u, write the raw rotational scalar as

```
B[Lt,Hu,J] = sum_MN <LM,HN|J,M+N> (-1)^(M+N)
                    D^L_{M,t}(Omega_A)* D^H_{N,u}(Omega_B)*
                    C_{J,-M-N}(Rhat).
```

Transform local columns to the same real basis for real t,u. Dividing the
3j version of the angular scalar by its zero-m 3j (Hatz eqs. 4.4–4.5)
gives, in this CG notation,

```
Sbar[Lt,Hu,J] = i^(L-H-J) B[Lt,Hu,J] / <L0,H0|J0>.
```

The identity between 3j and CG supplies the displayed `(-1)^(M+N)`; it is
not another empirical phase. In particular Sbar at aligned axes with
`t=u=0` has value `i^(L-H-J)`, **not always +1**. For our stretched J,
`i^(L-H-J)=(-1)^H`. The denominator is strictly positive and nonzero because
`J=L+H`, even when L or H is odd. No extension to the zero-denominator,
odd-`L+H+J` cases is needed.

The physical orientation-resolved coefficient is reconstructed as
`sum_tu C_n(t,u,J) Sbar[Lt,Hu,J]` (and summed over all other J/L/H channels
when present). Therefore division by the angular prefactor cancels the
electrostatic `(-1)^H` and multiplies by `<L0,H0|J0>`:

```
D(l,p,k,q) = F(l,k) F(p,q) <K0,P0|J0> <L0,H0|J0>  > 0,

C_(J+2)(t,u,J) = sum_(l,p,k,q: sum=J; l+p=L; k+q=H)
                  D(l,p,k,q) sum_f w_f aA[f;L,t](l,p) aB[f;H,u](k,q).
```

This is the oracle. All factors are factorial CGs or binomial coefficients.
For stretched local ranks the real-basis first-stage coefficients are real;
the CP product is already real and needs no `i` phase or conjugation. There
is no extra off-diagonal multiplicity: the ordered-rank loops already
include reciprocal partners once each. Weights contain the sole CP prefactor.

A preliminary development calculation compared the raw electrostatic
coefficient before the Sbar phase conversion: odd H had the opposite sign.
The published `i^(L-H-J)` resolves that distinction. The final formula does
not fit phases or normalization constants to high-J production values.

### 4. Independent angular projection check

The helper's zero-m factors also have a bounded polynomial check:

```
<K0,P0|J0>^2 = (2J+1)/2 integral_-1^1 P_K(x) P_P(x) P_J(x) dx.
```

For `J=K+P<=10`, the polynomial degree is <=20, so 11-point Gauss–Legendre
quadrature is exact in exact arithmetic. The same check is made for L,H.
The positive stretched-CG sign fixes the square-root branch. This verifies
the angular factors against Legendre projection, rather than reconstructing
an energy using the same production tables. No numerical orientation fitting
or comparison of unlike representations is used.

## Tests and independence boundary

`tests/pytests/isapol_high_j_oracle.py` imports only NumPy, standard-library
math/iteration/cache facilities and the pure factorial helper. Importing it
was checked not to load `psi4`. Its expected-value path has **no**:

- production `realcg_data`, recoupling tables, copied table numbers, metadata
  enumeration, recoupled tensor getter, or `isa_recoupled_dispersion` call;
- audit transcode or generator import;
- CamCASP/home-tree/archive/runtime reference file read;
- external service, SymPy/SciPy dependency, charge or symmetry repair.

Only `actual_high_j` in the new test module constructs production models
and calls `isa_recoupled_dispersion`. Structural expected keys are derived
from integer triangles/rank sums, not production coverage metadata.

Tests cover:

- Full ranks123 A/B, nonidentical mixed-rank SPD matrices at two positive
  frequencies plus a zero-weight static node. All 735 rows have a numerical
  comparison, with nonzero signals and both signs in each block, including
  cosine/sine components. Seeds generate inputs, never stored expected values.
- All 16 ordered C11 and 10 ordered C12 rank quadruples. Eleven minimal
  compressed-rank cases isolate each class modulo the required reciprocity.
  In particular ranks13 distinguish `(1,3)/(3,1)` from `(2,2)` despite the
  same coupled rank L=4. Both A/B assignments are covered where distinct.
- Per-block mutation sensitivity: zeroed, sign-reversed and 1%-scaled actual
  high-J values must fail. Missing-row mutation fails too. A separate actual
  evaluation with A scaled by 2 and B by 3 must give six times the oracle.
- Factorial zero-m factors checked by the Legendre projection above, together
  with the analytic electrostatic/Sbar phase cancellation.

The new comparison uses the **existing analytic** `rtol=3e-13, atol=3e-12`.
The old archive gate is untouched. Arrays are constructed as `X X^T + c I`,
checked exactly symmetric and SPD, never symmetrized after construction.

## What is not certified

- Individual ordered terms indistinguishable under reciprocal inputs cannot
  be certified separately: only their observable sum can be isolated. For
  stretched coupling `a(l,p)=a(p,l)` after reciprocal transposition. A
  compensating redistribution within such a class is unobservable to this API.
- No independent general second-stage 6j/9j oracle for lower J is claimed;
  the maximal-J overlap proof does not extend to non-stretched channels.
- Rank4 is still rejected. General/unrestricted C12 remains partial. High-J
  triangles plus product parity imply the only possible J>=9 outputs for
  positive ranks<=3 and n<=12 are precisely J9/C11 and J10/C12; this is not
  an assertion that all unrestricted multipole quadruples are present.
- No water-deck high-J literal values, arbitrary nonreciprocal responses,
  imaginary lower-J CP phases, damping, global-axis production reconstruction,
  native electronic-response protocol, or performance code is certified here.
- This is a finite deterministic numerical regression plus a mathematical
  derivation, not exhaustive testing of all tensors or floating-point scales.
