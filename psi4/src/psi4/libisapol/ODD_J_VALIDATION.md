# Odd-parity lower-J investigation: convention remains unresolved

## Status — raw identities, not production coefficient certification

Source-only investigation after `636c6db6e0`. Here **odd** means
`L+H+J` odd, with **J<n-2**, not simply J odd or dispersion order n odd.
The exact production angular normalization/phase has **not** been established
from the allowed sources inspected. No production-ratio fit was attempted.
The **5341 production odd rows remain numerically uncertified by an independent
oracle**. Their reported existence comes from the prior lower-J work and its
unchanged structural test, not from assuming a normalization in this increment.

New deliverable `tests/pytests/test_isapol_odd_j_raw.py` adds six bounded tests of
raw coupling identities, with no new helper or production edits. It reuses the
existing exact factorial CG/tree helpers. Standalone: **6 passed in 0.91 s**;
focused normal-fixture run with the existing lower-J module: **46 passed in
11.06 s**, using the unchanged staged core. Parent integrated validation now
passes **1801 tests in34.85s** (`.pi/audit/odd-j-parent-regressions.log`). This
accepts the raw-identity tests only, not the5341 production coefficient magnitudes.

Parent inspected public RRF repository metadata and README only: default branch
`public`, <https://gitlab.com/anthonyjs/rrf/-/blob/public/README.txt>. The README
states GPL but also restrictive redistribution language; no executable source was
inspected and no MIT permission inferred. **User chose to defer this gap and move
to matched native protocol work**, rather than inspect RRF source. The missing
odd-channel angular definition remains a recorded blocker.

## Source investigation and precise boundary

### Allowed CamCASP documentation and analytical interface

Read `~/gits/CamCASP/docs/casimir_manual.tex` in full, especially programming
notes, lines 85–150. It establishes the unitary real-component first coupling,
possible pure-imaginary coupled responses, the bilinear CP integral and that
`C6coeffs` in Stone's **separate RRF package** generated the explicit second-stage
formulae. The reference is Stone, *The Theory of Intermolecular Forces*, section
4.3.4. It does **not** state an explicit odd-channel angular basis/normalizer.
Its description of a second coupling into a single tensor label alone is not
sufficient to infer the output coefficient's scalar angular normalization.

Read `src/casimir/casimir.f90:330–499`, including `cpint:395–435`. The integral
multiplies the two coupled responses **without conjugation**. Its `ip` argument
requests a power of i so that the result is real; the implemented `ip==1` branch
multiplies by +i. This establishes an integral interface, **not** the block's
angular normalization or the sign/scale of the multiplying analytical factor.
No numerical `cterm` or generated `c6code`…`c12code` entry was read for this work.

Bounded local discovery:

- Listed the CamCASP root, `src`, `src/casimir`, and `docs` directories.
- Searched filenames for `C6coeff`, `RRF`, `realcg`, `recoup`, `sfunc`, `angular`,
  excluding `.git`, data, examples, basis, x86-64 and ORIENT-named directories.
  No matching generator file was found in that search. This is **not** proof
  that no copy exists in an archive, excluded directory, other checkout or history.
- Searched `.tex`, `.md`, `.txt` documentation in `docs`, `design`, `dev` for
  generator/recoupling/S-function terms, excluding ORIENT-named paths. The relevant
  generator description was the casimir manual; users-guide hits are usage text,
  not a definition of the missing angular normalization.
- Did not unpack `CamCASP_src.tgz`, inspect numeric realcg/recoupling tables,
  inspect audit transcodes, or open ORIENT executable or generated source.
  The initial fuzzy filename search returned some example/output path names;
  their contents were not read or used.

CamCASP checkout HEAD at inspection:
`63b16a22b9bae597fe81ecdb8b8d91c21868c814`. SHA256 of actual inspected files:

```
casimir_manual.tex  1c4c6a71254f8a109cdaf2efadcd1e72387f2ec3f6d565cbcbe6baf38b80fd88
casimir.f90         3f6d4bc31c046d2440f5e0e7489be18d6a4c5aac09afa3bab25074c3062637ec
```

CamCASP convention attribution: **Alston J. Misquitta and Anthony J. Stone**,
Copyright (c) 2019 Anthony Stone, MIT. Full permission and warranty notice remains
in [RECOUPLED_CAMCASP_LICENSE](RECOUPLED_CAMCASP_LICENSE), read in this increment.
No source implementation or numeric table was transcribed into the new tests.
The new mathematical test implementation is a Psi4 LGPL-3.0-only addition.

### Public mathematical and author documentation

1. Anthony Stone, **RRF manual**, printed p.4:
   <https://www-stone.ch.cam.ac.uk/documentation/rrf/manual.pdf>.
   Downloaded PDF and read the generator paragraph in its extracted text. It
   explicitly identifies `realcg.F90` and `C6coeffs.F90`, the latter generating
   Fortran dispersion code through C12 from local distributed polarizabilities.
   It supplies no odd-channel angular formula. SHA256 of downloaded PDF:
   `4541d3fc045008d593eaa9987ae05a0a74ece993111f93ae7b5ed9c1bf83b6ab`.
2. Author's **RRF routines documentation**:
   <https://www-stone.ch.cam.ac.uk/documentation/rrf/routines.html>.
   Read the documentation, not the linked repository. It describes exact radical
   arithmetic, 3j/6j/9j evaluation, and conversion to `A*i**I` with I=0 or 1.
   Its link to `gitlab.com/anthonyjs/rrf` identifies a possible missing-source
   location. **That separate repository was not fetched or inspected**, and its
   source license/provenance is not established here. CamCASP's MIT permission
   must not silently be assumed to cover arbitrary separately hosted source.
3. Richard Hatz (2016), *Computational Studies of Dispersion Interactions in
   Coinage and Volatile Metal Clusters*, ISBN 978-951-51-1909-4, printed pp.23–24,
   equations **4.3–4.5**:
   <https://helda.helsinki.fi/server/api/core/bitstreams/473f3918-0d2e-4005-9918-ae5ac9ad2279/content>.
   Re-downloaded and read equations and limiting prose in extracted PDF text.
   Eq.4.4 gives unnormalized S with phase `i^(L-H-J)` and a 3j contraction.
   Eq.4.5 divides by the zero-m 3j; eq.4.3's surrounding prose explicitly notes
   undefined odd-rank-sum normalization and the symmetry restriction of the
   examples. This is a published mathematical definition, but **not evidence
   that CamCASP's odd coefficients multiply that unnormalized S**. PDF SHA256:
   `3dd4e4a72f2ddc264e599b9acc2c3be729515583c6f738368db84ae95219932b`.
4. NIST DLMF <https://dlmf.nist.gov/34.3>, particularly **34.3.9–10** (column
   exchange and magnetic sign reversal) and **34.3.16–17** (orthogonality).
   Read the page; these justify the new exact exchange and basis-resolution
   checks. CG conversion/factorial formula and raw electrostatic derivation are
   inherited unchanged from [LOWER_J_VALIDATION.md](LOWER_J_VALIDATION.md),
   which cites DLMF 34.1.1, 34.2.4 and 34.3.20.
5. Primary-paper lead: A. J. Stone (1978), *The description of bimolecular
   potentials, forces and torques: the S and V function expansions*, Molecular
   Physics 36, DOI **10.1080/00268977800101541**. Publisher fetch returned
   **HTTP 403**. No equation from this inaccessible paper is claimed as read
   or used. Search-result summaries are discovery leads, not convention evidence.
   The Stone book information page was also fetched but establishes no formula.

These public documents are cited as mathematics/author documentation, **not**
as MIT-licensed CamCASP software. No executable ORIENT source was retrieved.
The web search's generic recommendations to use unnormalized S are not adopted
as the definition of this API.

## What is established mathematically

Use ordered `(l,p,k,q)`, K=l+k, P=p+q, n=l+p+k+q+2 and the two normalized trees
from LOWER_J_VALIDATION. The coefficient of the raw scalar B is exactly

```
f_raw = (-1)^(k+q) sqrt[binom(2K,2l) binom(2P,2p)]
        * CG(K0,P0|J0) * <[(lp)L(kq)H]J | [(lk)K(pq)P]J>.
```

The new test-local `raw_ordered` retains this as `r*sqrt(s)` using Fractions.
It has **no zero-m Sbar factor, denominator fallback, or CP phase conversion**.
It is not exposed as a production coefficient oracle.

### Exact site exchange

Swapping molecular sites changes each stretched interaction CG by +1, and the
outer response CG by `(-1)^(L+H-J)`. The radial factor changes by
`(-1)^(l+p-k-q)`. Since the harmonic product requires `l+p+k+q+J` even,

```
f_raw(k,q,l,p;H,L,J) = (-1)^(L+H) f_raw(l,p,k,q;L,H,J).
```

The same holds after local reciprocal sums. Tests compare exact rational squared
magnitudes **and signs**, not rounded numbers. All 394 odd reciprocal classes
are exercised, including the 212 exactly vanishing classes and 182 survivors.
Each also has exactly zero `CG(L0,H0|J0)` and is rejected by the existing
normalized helper. Site exchange is for the raw B coefficient; translating it
to a differently normalized production angular function requires its definition.

### Pointwise tree resolution, not only unit norm

For each magnetic product state at M=J, completeness gives

```
T1(m,n,r,s) = sum_LH R(L,H,J) T2_LH(m,n,r,s).
```

Three nonstretched cases `(l,p,k,q;J)=(1,2,1,3;3), (1,2,2,3;4), (3,2,2,3;6)`
are checked at every allowed magnetic state using the separate existing floating
factorial CG implementation for the pointwise vectors and exact-overlap helper
for R. Omitting the odd sector fails with a signal >1e-3. Its exact rational
projected norm is strictly between zero and one in each case. These are finite
identity checks, not a second independent 9j implementation or an API test.

### Imaginary CP and compressed synthetic responses

Real reciprocal input obeys `a(lp;L)*=(-1)^(l+p-L) a(lp;L)` and
`a(pl;L)=(-1)^(l+p-L) a(lp;L)`. For odd L+H+J on the product domain,
`l+p+k+q-L-H` is odd. Thus **one first stage is imaginary and the other real**;
their bilinear CP is imaginary and need not vanish.

For **all 182 surviving reciprocal classes / 80 blocks**, test inputs are two
nonidentical three-frequency full rank123 SPD arrays `X X^T + I`. Minimal
compressed local rank-pair submatrices independently exercise offsets, including
ranks13. Canonical full-input raw CP times the exact reciprocal coefficient is
compared to the sum of ordered raw CP terms calculated from compressed inputs.
Both even and odd dispersion orders occur, and both choices of imaginary site
are witnessed. Conjugating B when B is imaginary reverses CP. Multiplying by +i
makes CP real, but **-i does too**; reality alone cannot choose the convention.
A zero-weight static node is excluded before the bilinear multiplication;
weights already contain the CP prefactor. No extra factor or conjugation occurs.

These checks use only declared ranks, exact algebra and synthetic arrays. No
production tensor, output, table, archive, or source file enters expected values.
Tolerances are the unchanged analytic `rtol=3e-13, atol=3e-12`. Signal thresholds
only assert witness strength, never determine support. The inherited bounds are
rank<=3, J<=10, at most eight nodes, bounded 32768-CG/8192-overlap caches and nine
first-stage maps. The new tests use three nodes and add no persistent cache.

## Exact missing definition and next admissible step

To certify production numbers one needs a sourced formula specifying, for odd
L+H+J, the angular basis multiplied by `C_n(t,u,J)`: e.g. an explicit

```
S_API[Lt,Hu,J] = N[L,H,J] * B[Lt,Hu,J]
```

in the same real Racah local-column convention, including **N's sign, magnitude
and complex phase**, and a provenance link from that definition to CamCASP's
second-stage generator. A more general component transformation must be specified
if the convention is not simply a block scalar. The casimir manual's named
`C6coeffs.F90` analytical generator is the most specific lead: an authorized,
license-checked, non-ORIENT-derived copy, or an author/public mathematical statement
of its normalization branch, is needed. Numerical generated entries are not a
substitute. No claim is made that that file necessarily contains the answer.

If `S_API=N B`, the raw coefficient fixes `C_API=raw_CP/N`; electrostatics fixes
the product, not N. Arbitrary nonzero block rescalings of S and inverse rescalings
of C preserve the interaction. Even exchange symmetry can be preserved by choosing
symmetric rescalings. Neither raw overlap identities, CP reality, nor agreement
at an aligned zero of the odd function determines the missing normalization.
Hatz's unnormalized S is a legitimate published basis; equating it to S_API
without a source would be a **discovery hypothesis**, not a verified convention.
No such hypothesis was calibrated against production here.

Consequently this increment does not supply required production row sets,
5341-row numeric comparisons, production scale/sign/zero/missing-row mutation
certification, or a new normalized odd oracle. Those requested closure tests
remain contingent on the definition above. Existing lower-J tests retain their
full/compressed even required-row checks unchanged. No maximal-J lower-order
extension was pursued instead of the main question. No production defect was
demonstrated or silently corrected; rank4 rejection, partial C12 and lack of full
native SCF/PFIT/GRAC parity remain unchanged. Parent owns plan/SPEC/report and
integrated acceptance. Commands and handoff: `.pi/audit/odd-j-validation-handoff.md`.
