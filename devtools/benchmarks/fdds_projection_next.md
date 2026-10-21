# Next projection seam — preparation, not an implemented speedup

Read-only review `d0fddd7a1b35b7147ec6279b7c261aeab` examined the measured
80–86s auxiliary-overlap-and-collapse phase. It includes Libint delta
integrals, copying, and contractions; no individual cost is established.

Current code computes each triangular shell pair `(P,Q)` against every R.
The two density matrices already share integral evaluation. At3820 auxiliary
functions this moves roughly223GB of cumulative scratch writes, not a live
cubic tensor. It then performs separate DDOTs for both densities.

## Bounded first experiment

Keep integral generation, normalization, precision0, shell ordering, scratch
layout and output ownership unchanged. Attribute build/copy versus
contraction/scatter using thread-local accumulated durations (not global
hot-loop timers, and not elapsed-wall totals). Only if contractions matter,
try one tiny-output DGEMM over all densities instead of repeated DDOTs.
One-density native fallback and explicit zero-density behavior remain.
Retain factor2 and the existing surviving lower-triangle orientation on
diagonal shell blocks. Account for packed density/result scratch in the
object's host budget. Nested BLAS threading may make this slower; measure.

This is CPU preparation optimization, not a justified large GPU offload.

## Larger possible integral reduction

The contact overlap tensor is symmetric in all three indices. Canonical
shell triples `P>=Q>=R` could avoid repeated ordered integral calls.
Contribute to PQ always, PR only when Q!=R, QR only when P!=Q.
Do not introduce a generic factor6. Different triples update common output
blocks, so current race-free pair ownership cannot simply be retained.

Private output reductions need explicit process-memory accounting, equal-shell
and higher-angular-momentum tests, deterministic ownership, and a separate
validation window. No dense production O(naux^3) tensor is acceptable.
This algorithm is NOT implemented or GPU validated in the current branch.

## Preparation and next gate

`tests/pytests/test_fdds_projection.py` adds a tiny independent Mints reference
for density counts0/1/2/4, signed nonsymmetric inputs, two thread counts,
repeat calls, symmetry and input preservation. It deliberately caps naux20
before constructing the cubic reference tensor. This is separate from the
accepted SVD183-native/112-GPU counts.
Initial native-only reference run: eight passes in2.124s, zero
failures/errors/skips, existing scientific1d1 CPU stage/core1e731,
local `p4_ci` oncos-4a16505. Evidence:
`build_fdds_cpu/cycle-20261021/projection-reference-preparation/native.xml`.
This is not a new production build or GPU projection validation.

Additional production-change gates still needed: heterogeneous higher-L
pure/cartesian and uncontracted shells; changed-input freshness; complete
scratch/output memory failure/recovery; native and actual GPU suites; final
physical FDDS energies and matched whole-workflow controls. No new allocation
or scientific relaxation is authorized by this note.
