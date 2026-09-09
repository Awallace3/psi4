# Independent lower-J validation: nonsingular even-parity subset

## Claim and status

Test-only increment after `f996942ad6` and `e0e9fcfbbe`. Parent integrated
acceptance PASSED: **1795 ISA/FDDS tests in34.18s**, including40 new cases, against
the unchanged staged core. No production/performance algorithms or tolerances
changed; no rebuild needed. Independent review found no must-fix issue for this
subset. Parent additionally requires independently derived structural rows in
all compressed-rank isolation comparisons, even for numerical zeros. Evidence:
`.pi/audit/lower-j-parent-regressions-v2.log`.

The new standalone `tests/pytests/isapol_lower_j_oracle.py` evaluates the GENERAL
four-angular-momentum overlap, not the maximal-J shortcut. It independently
validates **all even-(L+H+J), strictly lower-J channels** for positive ranks <=3,
C6..C12, including odd dispersion orders and imaginary first-stage components.
Here "lower" means **J < n-2**, not merely J<=8. Maximal-J lower-order channels
are outside this new numerical comparison, as are J9/C11 and J10/C12.

This is a justified restricted oracle, **not full lower-J closure**. The
zero-m Sbar normalization is singular for odd L+H+J. Those channels are not
absent: after exact reciprocal summation, 182 angular/rank-pair classes survive
in 80 odd-parity blocks. Their normalization in the production coefficient API
is not established here. No factor was fitted to production values.

Development checks: **40 new tests pass in 10.86 s** against the unchanged staged
core (normal pytest fixtures); **6 standalone algebra tests pass in 3.52 s**, with an empty
PYTHONPATH and no staged-core requirement. These development checks are now
supplemented by the parent integrated acceptance above. Commands and failures are in
`.pi/audit/lower-j-validation-handoff.md`.

## Exact sources and independence boundary

Public mathematics consulted (not executable source, numeric reference tables,
or a claim that the publications have an MIT license):

1. Richard Hatz, *Computational Studies of Dispersion Interactions in Coinage
   and Volatile Metal Clusters*, Helsinki dissertation (2016), ISBN
   978-951-51-1909-4, printed pp. 23–24, equations **4.3–4.5**:
   <https://helda.helsinki.fi/server/api/core/bitstreams/473f3918-0d2e-4005-9918-ae5ac9ad2279/content>.
   Downloaded the public PDF and read the extracted equations and surrounding
   text. Eq. 4.4 establishes the unnormalized S phase `i^(L-H-J)` and the 3j
   contraction; eq. 4.5 divides by the zero-m 3j. Eq. 4.3 and its following
   paragraph explicitly say that the normalized functions are **undefined for
   odd rank sum**, which the dissertation's high-symmetry examples do not need.
   That symmetry restriction cannot be transferred to arbitrary mixed-rank
   reciprocal tensors. The unnormalized eq. 4.4 remains available.
2. NIST DLMF **34.1.1**, <https://dlmf.nist.gov/34.1#E1>, specifically the equation
   image <https://dlmf.nist.gov/34.1.E1.png>: CG/3j conversion and its phase.
   **34.2.1–4**, <https://dlmf.nist.gov/34.2>, including
   <https://dlmf.nist.gov/34.2.E4.png>: triangles, magnetic selection, and the
   finite factorial sum. These establish the new exact-rational CG helper.
3. NIST DLMF **34.3.8–10, 34.3.16–17, 34.3.20**,
   <https://dlmf.nist.gov/34.3>: permutation/sign-reversal symmetry, CG
   orthogonality/completeness, and the spherical-harmonic product theorem.
   Converting `Y_lm` to Racah `C_lm=sqrt(4*pi/(2l+1))*Y_lm` gives the product
   coefficient used below. Sign reversal also proves the odd zero-m zero.
4. NIST DLMF **34.7.2, 34.7.4**, <https://dlmf.nist.gov/34.7>, including
   <https://dlmf.nist.gov/34.7.E2.png>: 9j orthogonality and its magnetic sum rule.
   These corroborate the interpretation of the direct tree overlap and the
   exact sum-of-squared-overlaps check. No tabulated 9j numbers are consumed.

Convention discovery, distinguished from independent mathematical validation:
`ANISOTROPIC_CONTRACT.md`, `RECOUPLED_CONTRACT.md`, `HIGH_J_VALIDATION.md`,
`anisotropic_dispersion.cc`, `recoupled_dispersion.cc`, and the four requested
existing Python helper/test files were read. They establish the target's real
no-CS component ordering, electrostatic sign, bilinear CP weights, offsets, and
result interface. Neither shipped table, `realcg_data`, an audit transcode,
a generator, an archive, nor a production coupled tensor enters expected values.
No ORIENT source (including source generated from it) was consulted or copied.
No new CamCASP source extraction was needed.

CamCASP interface conventions are due to **Alston J. Misquitta and Anthony J.
Stone**, Copyright (c) 2019 Anthony Stone, MIT. Full permission and warranty
notice: [RECOUPLED_CAMCASP_LICENSE](RECOUPLED_CAMCASP_LICENSE). The new code and
mathematical implementation are Psi4 additions, LGPL-3.0-only; this is not a
numerical table transcode. Only Python standard library and existing NumPy are
used by the expected helper. Pytest is used by tests. A fresh isolated subprocess
checks that importing and evaluating the helper does not import Psi4.

## Derivation

### 1. Electrostatics and product harmonics

Use ordered slots `(l,p,k,q)=(la,lap,lb,lbp)`, `K=l+k`, `P=p+q`, and
`n=l+p+k+q+2`. Local response ranks range over
`|l-p|<=L<=l+p`, `|k-q|<=H<=k+q`; they are **not** restricted to their maxima.

For complex Condon–Shortley components, the interaction is the irreducible
rank-K scalar contraction, with signed reduced coefficient

```
(-1)^k F(l,k),     F(l,k) = sqrt(binomial(2(l+k),2l)).
```

This follows from covariance of the regular-harmonic derivative: its only
surviving traceless rank is K. Its scale and sign are fixed by the positive-z
axis, without a dispersion table:

```
(-1)^k F(l,k) <l0,k0|K0> = (-1)^k binomial(l+k,l).
```

The stretched factorial CG identity proves this equality. The right side is
also the direct axial Coulomb derivative in the anisotropic contract. The
ordered product of two interactions therefore has radial sign `(-1)^(k+q)`
and magnitude `F(l,k)F(p,q)`.

The Racah product theorem is

```
C_Ka C_Pb = sum_J <K0,P0|J0> <Ka,Pb|J,a+b> C_J,a+b.
```

Consequently only `|K-P|<=J<=K+P` and **K+P+J even** contribute. Conjugate
harmonic indices in the two scalar contractions introduce no hidden exchange
phase: `CG(K,-a,P,-b|J,-M)=(-1)^(K+P-J) CG(Ka,Pb|JM)`, and that phase is +1
on precisely these allowed product channels.

### 2. General overlap of two normalized trees

For fixed total `J,M`, use the common ordered product basis
`|l m, p n, k r, q s>`, with `m+n+r+s=M`. Define

```
T1(m,n,r,s) = CG(lm,kr|K,m+r) CG(pn,qs|P,n+s)
             CG(K,m+r,P,n+s|JM),
T2(m,n,r,s) = CG(lm,pn|L,m+n) CG(kr,qs|H,r+s)
             CG(L,m+n,H,r+s|JM),
R(l,p,k,q;L,H,J) = sum_(m,n,r,s) T1 T2.
```

Every CG is real in this complex CS basis. These are distinguishable slots:
reordering the middle slots to express the two trees in a common basis does not
introduce a fermion exchange sign. CG orthogonality normalizes both trees, and
rotational invariance makes R independent of M. In standard 9j notation this is

```
R = sqrt((2K+1)(2P+1)(2L+1)(2H+1)) { l k K
                                    p q P
                                    L H J }.
```

The oracle evaluates the six-CG finite sum directly at M=J. It does not assume
R=1. At maximal J this expression reduces to +1 by the highest-weight argument,
but the lower-J sum has nontrivial magnitude, sign, and exact zeros.

Each new CG is represented as `r*sqrt(s)` with exact rational r,s: a rational
factorial alternating sum and a rational factorial prefactor squared. For each
fixed overlap, magnetic terms have the same radical up to a rational square.
The implementation checks that square using integer square roots of numerator
and denominator, then sums **exact Fractions**. A failed square property raises;
it is never approximated. Thus zero classification uses exact cancellation,
not a floating cutoff. Conversion to double occurs only after the angular sum.

Independent checks include exact rational
`sum_(L,H) R^2=1` for every rank<=3 quadruple with sum<=10 and every allowed J
(including both product parities), and equality of sign and exact squared
magnitude at M=J versus M=0 for three representative quadruples. A separate
floating direct sum uses the pre-existing factorial CG helper on five examples,
including an odd-parity channel and a zero. This is not a second 6j engine;
orthogonality and changed-M summands are the complementary checks.

### 3. Sbar phase and normalization, explicitly

Before normalizing the angular scalar, write

```
B[Lt,Hu,J] = sum_MN CG(LM,HN|J,M+N) (-1)^(M+N)
                    D^L_(M,t)(Omega_A)* D^H_(N,u)(Omega_B)* C_J,-M-N(Rhat).
```

Transform the local columns with the same real Racah U used for the input
components. From DLMF 34.1.1,

```
3j(L,H,J; M,N,-M-N)
  = (-1)^(L-H+M+N) CG(LM,HN|J,M+N)/sqrt(2J+1).
```

Hatz eq. 4.4 consequently gives
`S = i^(L-H-J) (-1)^(L-H) B/sqrt(2J+1)`. Dividing by the zero-m 3j in eq. 4.5
cancels the last sign and square root, yielding exactly

```
Sbar = i^(L-H-J) B / CG(L0,H0|J0).
```

For integer triangle-allowed **even L+H+J**, this denominator is nonzero and
the phase is real, `(-1)^((L-H-J)/2)`. The coefficient multiplying Sbar is
therefore

```
f(l,p,k,q;L,H,J) = (-1)^(k+q) F(l,k)F(p,q) CG(K0,P0|J0)
                   R(l,p,k,q;L,H,J) CG(L0,H0|J0) / i^(L-H-J),

C_n(t,u,J) = sum_ordered_(l,p,k,q) f * sum_f w_f aA[f;lp;Lt] aB[f;kq;Hu].
```

No conjugation, fitted phase, extra off-diagonal multiplicity, or additional CP
prefactor occurs. Ordered rank pairs already count reciprocal partners once.
The input weights contain `1/(2*pi)`. This formula was implemented before the
first staged-core comparison; that comparison passed without a phase adjustment.

The real-basis first stage is calculated only from synthetic arrays:

```
aA[Lt](l,p) = sum_ab (U_L CG(l,p;L) U_l* U_p*)[t,a,b] A[a,b].
```

For real inputs, conjugating these coefficients gives
`aA* = (-1)^(l+p-L) aA`. Thus local odd-parity channels are purely imaginary,
not automatically zero. On the supported product/even-Sbar domain,
`l+p+k+q-L-H` is even, so the **bilinear** CP product is real: it is either
real*real or imaginary*imaginary. An imaginary residue assertion guards this
property; it is not used to threshold the underlying outputs.

### 4. Structural zeros, reciprocity, and missing normalization

Keep these distinct:

- Triangle failure and odd K+P+J: zero harmonic product, excluded analytically
  before lower-J candidate enumeration.
- Exact R=0 within the triangle/product domain: an angular overlap zero.
- For reciprocal inputs, `a(p,l;L)=(-1)^(l+p-L) a(l,p;L)`. In particular,
  identical ranks and odd L are exactly zero because symmetric input contracts
  an antisymmetric CG map. This is a response-reciprocity zero, not a missing
  first-stage table or missing second-stage normalization.
- Different ordered partners can cancel even when their overlaps are nonzero.
  `reciprocal_raw_exact` sums the raw-B radial/overlap coefficients with these
  exchange signs for each canonical unordered local-pair class. This exact
  calculation does **not** use a Sbar denominator.
- Odd L+H+J gives `CG(L0,H0|J0)=0` by 3j sign-reversal symmetry. It is not
  permissible to divide by it. The normalized oracle raises `ValueError` for
  any such request, including channels whose physical coefficient is zero.

A concrete cancellation: `(pairA,pairB,L,H,J)=((1,2),(1,2),1,2,2)` has
nonzero ordered overlap but exactly zero reciprocal raw coefficient. A concrete
survivor: `((1,2),(1,3),2,2,3)` has nonzero exact reciprocal raw coefficient.
For the latter one local first stage is imaginary, the other real; neither is
forbidden by symmetric full mixed-rank input. These examples prevent a blanket
claim that odd channels vanish.

A mathematically permitted nonsingular alternative is the **unnormalized S**
of Hatz eq. 4.4, or equivalently `i^(L-H-J) B`. But choosing either convention
would change coefficient magnitudes relative to any other odd-channel extension.
The public reference does not establish which extension the production API's
odd coefficients use. No such choice is silently identified with that API.
Production agreement alone cannot supply the missing derivation.

## Exact finite coverage and tests

There are 76 ordered rank quadruples with positive ranks<=3 and sum<=10,
forming 33 unordered-local-pair classes with A/B kept ordered. Every class is
isolated by one test: same-rank inputs use a compressed single rank; mixed-pair
inputs use precisely the two declared ranks with diagonal rank blocks erased.
Those isolated arrays remain exactly symmetric but need not be SPD. Separate
full-ranks inputs are nonidentical frequency-dependent SPD `X X^T + c I`, with
mixed blocks and a zero-weight static node. No input is repaired.

Integer triangles and product parity generate **2367** strictly lower-J
ordered angular channels. The mutually exclusive priority classification is:

- 136 exact overlap zeros;
- 460 identical-rank reciprocal zeros among the remaining channels;
- 1191 supported nonsingular even-parity ordered channels;
- 580 odd-parity ordered channels still requiring normalization investigation.

Canonical reciprocal grouping gives **1073** angular classes: 516 surviving
even classes, 163 zero even classes, 182 surviving odd classes, 212 zero odd
classes. Sixteen classes with otherwise nonzero, non-identical-rank odd terms
cancel only after summing their reciprocal partners. The surviving odd classes
span 80 blocks. This is a stronger scope statement than the ordered-channel
classification alone, and is checked exactly rather than inferred from a seed.

The even-parity oracle retains **10256 candidate component rows in 228 blocks**.
Counts by n=6..12 are **79, 220, 533, 1018, 1876, 2748, 3782**. They include
zero blocks to test structural absence rather than thresholding it away.

The unchanged staged core returns **10074** of these rows:

- **10056 rows in 224 blocks** have a supported angular contribution and are
  required to be present, irrespective of their numerical value.
- 18 additional returned rows, `(n,L,H,J)=(6,1,1,0)` and `(6,1,1,2)`, are
  provable same-rank reciprocal zeros.
- The other 182 candidate rows, `(11,3,6,5)` and `(11,6,3,5)`, are absent
  from actual output and all four contributing ordered overlaps are exactly
  zero in the independent oracle. This is explained structural absence, not missing support for a nonzero channel.

No unexplained extra even-parity row is allowed. Numeric comparisons retain the
existing analytic **rtol=3e-13, atol=3e-12**. No archive tolerance was changed.
Every supported block has a nonzero signal on the full SPD seed and catches
zero, sign, and 1%-scale mutations. Fresh actual evaluation with A scaled by 2
and B by 3 checks sixfold scaling. Required row presence is checked explicitly.
The isolated `(1,2)/(1,2), L=H=2,J=2` block detects conjugation of B because
both coupled responses there are imaginary. Scalar anchors give
`f(1,1,1,1;0,0,0)=2` and the isotropic combinatorial C6..C12 formula, with all
other isotropic angular coefficients zero within the existing analytic tolerance.

Actual output also contains **5341 strictly lower-J odd-parity rows**. The test
records this scope without matching their magnitudes to an invented normalized
oracle. It does not certify the values of those rows.

## Bounds and limitations

- Expected evaluation consumes only constants, factorial algebra, declared ranks,
  synthetic matrices and weights. Actual-under-test construction is isolated in
  `actual_coefficients`; production models are never passed to the oracle.
- Rank domain 1..3; coupled L,H<=6; J<=10; at most eight input frequency nodes.
  Each sparse magnetic sum loops at most 7^3=343 triples, with the fourth
  projection fixed by M. No giant four-angular projection tensor is allocated.
  Cache ceilings: 9 first-stage maps, 8192 overlap results, 32768 exact CGs.
  First-stage maps have at most 49*7*7 complex entries. Cache sizes are explicit
  finite maxima, not assertions that every slot is populated.
- Zero-weight nodes are excluded before CP multiplication. Coupled arrays still
  include the static input. Algebraic zeros are removed by proofs, never by an
  output magnitude cutoff; 1e-8 is used only to assert mutation-test signal.
- No second independent 6j implementation was added. Exact identity resolution,
  changed-M sums, independent existing floating CGs and analytic scalar anchors
  supplement the finite-sum oracle; they do not constitute general arbitrary-rank
  symbolic verification or exhaustive floating-point testing.
- Reciprocal API inputs cannot identify compensating changes within an
  unobservable ordered-rank class. Isolation certifies observable sums only.
- Odd-parity coefficient normalization, maximal-J lower-order coefficients,
  unrestricted multipole ranks, rank4, full native SCF/PFIT/GRAC parity, water
  archive values, global-axis reconstruction, damping and performance are not
  certified. Rank4 remains rejected. Parent owns SPEC/report and integrated
  acceptance. No production defect was demonstrated or silently corrected.
