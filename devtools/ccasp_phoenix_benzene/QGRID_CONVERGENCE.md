# Controlled four-grid study — preparation only

Authority: external `ccasp-benzene-grid-convergence-grant.md`. No publication,
remote/native execution, scheduler test or submission by the author. Coordinator
independent review/acceptance precedes those actions. Exactly ONE free-embers
job, no retry, extra probe, SCF, ISA iteration, fit or response.

## Fixed recipe and reproducibility

`convergence_plan.json` explicitly declares, in order:

| Case | Requested radial / angular | Returned shells / angular | Points | AUX values |
|---|---|---|---|---|
| baseline | 200 / 590 | 199 / 590 | 1,408,920 | 1,614,622,320 |
| radial | 300 / 590 | 299 / 590 | 2,116,920 | 2,425,990,320 |
| angular | 200 / 974 | 199 / 974 | 2,325,912 | 2,665,495,152 |
| joint | 300 / 974 | 299 / 974 | 3,494,712 | 4,004,939,952 |

`isa_grid.cc` constructs `radial_points-1` shells in atom/radial/angular order;
974 is a tabulated Lebedev size (`libfock/cubature.cc`). Check returned radial
and angular counts, all13 atom starts and all12 per-atom counts, rejecting
round-up or layout changes. All12 centres, original absolute geometry/order,
exact1146Cartesian AUX columns/effective coefficients, Becke3, radius1,
unfiltered weights, numerical threads1 and1024-point tile reduction association
remain fixed. No production file or old launcher/config is changed.

Canonical grid bytes are contiguous little-endian binary64 `(x,y,z,w)` rows,
not sorted or filtered. Baseline must reproduce
`a01a17a4ab55dab608883cf5a184bfc13a42a15fae15efecf0256a9df383ab29`
**before collocation**. Variant hashes are unknown measured identities, not
invented expected values; reject equality with any earlier distinct grid.
Every case has its own declared recipe/count/output identity.

The plan pins immutable probe14308581 completion, source/runtime/derivative,
before/after closure, exact basis, grid identity, diagnostic and all four
original vector bytes, independently of capture14291484 inputs. The old
`qgrid_job.verify_capture` retains its original capture-driver base identity
`c31ad7fb…`; it is not relabelled as the new branch base `e24f97e9…`.
The original MAIN/AUX/probe/native/dependency admission is unchanged.

Declared baseline remeasurement condition: all four original vectors
(`U`, `U_compensated`, `absolute_sum`, `roundoff_bound`) **byte-exact** to
14308581. Retain every signed difference, both original/new hexadecimal values
and vector hashes in `baseline/remeasurement.json` and case diagnostics
**before** refusing any mismatch. No tolerance or fallback. This exact condition
is deliberately stronger than a small-norm comparison; an actual failure is
a blocker, not permission to retry or silently relax it.

Two actual sample addresses per atom are regenerated from each returned layout:
shell indices `floor(100*shells/199)` and `floor(150*shells/199)`, angular
indices `floor(137*angular/590)` and `floor(389*angular/590)`. They reproduce
the original baseline addresses but differ for each variant; grids are not
nested and physical sample points need not match. Save actual indices,
atom/radial/angular addresses, coordinates,90 selected columns and2160
independent native/oracle comparisons per case. Reuse the independent literal
GAMINT/effective-coefficient oracle, not the native producer twice.

## Bounded work and failure semantics

Total9,346,464points / **10,711,047,744AUX values** /9,129tiles. One public
`IsaExplicitBasis.evaluate_screened` traversal per case; NumPy tile reductions,
ordinary ascending tile sums, vector Kahan over those same sums, absolute
weighted sums and conditional summation bounds are unchanged from14308581.
No whole-grid-by-AUX matrix, spill or scalar-billion-element Python loop.

Maximum canonical grid111,830,784bytes (~112MB). Grid construction holds native
coordinate arrays and canonical storage simultaneously; getters may create
Python component lists. Copy **one component at a time**, reserving512MiB for
grid assembly, not claiming112MB total process peak. Native tile and weighted
products total18,776,064bytes, well below the unchanged8G allocation alongside
the4GiB Psi4 setting/dependencies. `one_case` owns and releases grid/native/
tile/report arrays before returning only two compact vectors to the next case.
No accumulating collection of grids.

Lossless vectors: five per case (`U`, `U_compensated`, `absolute_sum`,
`roundoff_bound`, `U_minus_q`),183,360bytes total. At most4MiB JSON per case
(2160 checks, site/shell signed summaries and identity), plus4MiB aggregate
metadata/comparisons/basis/vectors:20MiB planned payload output, hard bound32MiB.
Metadata writer independently caps each JSON at4MiB. Payload file hard limit
16MiB; scratch64MiB with256MiB free admission; run free admission128MiB;
96artifacts maximum. Stdout/stderr32MiB each; output+logs+scratch96MiB total.
Periodic aggregate checks (at most1s apart between bounded operations), each
case's reserved write check, final checks and inherited OS file limits fail
closed. Native calls/hash/filesystem operations can overrun until returning;
the30min scheduler limit is the hard outer bound, not a timing guarantee.

One cumulative1500s payload deadline starts before admission/loader and includes
all four cases, comparisons and final runtime closure. It is never reset at
case boundaries. A forced unthrottled final scan includes closure artifacts
and reserves64KiB/two files for final receipts. The86.24s measured baseline suggests ~9.5min proportional
collocation work, **not** a guarantee. No resource widening, adaptation or retry.

Run leaves all exclusive partial case files on any failure. Completion requires
all four sequential successful cases, final retained-handle runtime closure,
final deadline/storage checks and unsignalled zero exit. Batch TERM handling
forwards to child, waits, exits nonzero and never records complete after TERM.
`science_pass` and `properties_accepted` always remain false.

## Saved comparisons and interpretation

Each case records signed U−q minima/maxima/peaks/L1/L2/signed sums with column
identities and per-site/per-shell breakdowns, ordinary and compensated vectors,
timings, exact source/runtime/grid/basis/capture identities and selected-point
checks. Final comparisons retain signed vectors and norms for radial effect
at590 and974, angular effect at200 and300, and
`(U_joint-U_angular)-(U_radial-U_baseline)`. No monotonicity assumption.

Old S belongs ONLY to baseline. Baseline S−U is a net partition/exclusion/
accumulation diagnostic. For every variant, `S_baseline_minus_U_cross_grid`
is explicitly labelled **cross-grid only, NOT a new-grid stockholder/exclusion
error**. No new Q, masks, partition, response or accepted property is inferred.
Sampled oracle agreement is not all-point certification. The roundoff bound
only concerns computed-product ordinary summation, not quadrature/collocation/
exp/product/underflow error or a Kahan certificate.

## Coordinator-only future commands

After independent acceptance, publish only the new branch and create exact
immutable remote source. `convergence_job.py prepare --commit COMMIT --tree TREE`
in isolated accepted Python is metadata-only: verifies private inputs,
creates exclusive paths, composes configured Python/new source with accepted
native15cc/coreb283901d/Einsums1.1.3, creates separate runtime and derivative
manifests and prints exact future scheduler argv. No raw-template overlay,
build, install, download, old-stage mutation or native import in preparation.
Configured runtime composition is separately bounded to32MiB copied Python
plus two <=4MiB manifests (accepted core/data are links), outside the96MiB
payload/run/scratch bound. Coordinator storage review must cover both.

Generated manifest hashes remain **unknown** until that future preparation.
Coordinator reviews actual hashes/argv, fresh capacity/storage/log conditions
and exact `sbatch --test-only` on explicit account/partition/embers. Only then
submit ONE `convergence.sbatch` with actual commit/tree/runtime/derivative pins.
No extra baseline/probe/replay allocation.

## Not production-grid admission

Before ANY larger production Q grid, recompute full-chain resource plan against
unchanged24GiB numeric /6.4e13work /64GiB I/O, caller/shared reservations included.
Q alone is3200bytes/point for12sites: baseline4,508,544,000bytes versus
joint11,183,078,400bytes. A cheap tiled study neither widens those gates nor
promises a corrected full response passes.

Engine-free validation:

```bash
python3 -B -m unittest discover -s devtools/ccasp_phoenix_benzene -p 'test_*.py' -v
bash -n devtools/ccasp_phoenix_benzene/convergence.sbatch
git diff e24f97e9b7fbafcc25231be50feb1c39a38b5862 HEAD --check
```

Tests use synthetic arrays/fake APIs. Actual new native hashes, numerical
reproducibility, timing, source/probe composition and allocation remain pending.
