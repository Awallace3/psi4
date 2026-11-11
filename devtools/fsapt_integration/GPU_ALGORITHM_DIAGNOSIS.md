# Peptide GPU algorithm diagnosis and protein83 follow-up

Evidence: H200 job 15472290, unchanged science `08984a56e6`, timing helper
`ade1014462`. The timing-suite peptide has 250 AOs, 20 occupied orbitals per
monomer (5 frozen + 15 active), 64 minimal functions and 24 atoms.

## Dimer SCF: setup matters

Untraced `timer.dat` medians over all three fresh workers, seconds:

| Dimer scope | CPU8 | H200/host8 |
|---|---:|---:|
| Complete SCF | 4.643 | 4.960 |
| Initialize | 1.425 | 2.291 |
| Form core H (inside initialize) | 0.014 | 2.131 |
| Iterations | 2.862 | 2.177 |
| JK (inside iterations) | 2.151 | 1.503 |
| DIIS (inside iterations) | 0.498 | 0.541 |

These rows are nested, not additive. CPU uses 23 dimer G builds in each sample;
GPU uses 23/26/23. The slow GPU sample (8.841 s complete SCF) remains included.
One-time first-use one-electron work is a strong lead; its exact subdivision
between native startup, kernel preparation and integral computation is not yet
measured. Do not call this a cuEST JK regression or subtract startup from the
workflow baseline. The dimer runs first; both monomers reuse its JK object.

## Localization: ordered tiny launches, not packed GEMMs

`sapt_fisapt_localization_cuda.py::jacobi_localize_device` launches one
128-thread `ibo_pair` block per ordered pair. For this case:

- 115 pairs per sweep: choose(5,2) + choose(15,2).
- 11 converged sweeps per monomer: **2,530 pair launches** total.
- Only 24 threads contribute atom-wise angle statistics over 64 labels.
- Pair updates share orbitals and must observe preceding updates. Naively
  batching overlapping pairs changes the localization trajectory/gauge.
- Sweep controls download three statistics and synchronize: 22 sweep-control
  downloads across both monomers. There is not a host synchronization per pair.
- IAO construction additionally uses six small symmetric powers, repeated
  allocation/scopes, and explicit dense AO operators; it is not a batched-GEMM
  localization implementation.

First candidate: fuse **one ordered sweep** in a resident kernel, preserving
pair order and reduction/rotation arithmetic. The peptide coordinate and
rotation arrays total only 13,440 bytes per monomer. Shared-memory residency
is plausible, not yet implemented or measured. Then reduce redundant solver
workspace allocation/validation and hoist common IAO products.

## Electrostatics: avoidable calls and repeated host work

`sapt_fisapt_elst_cuda.py::felst_device` evaluates both charge-assignment sides
for every atom:

- **48 point-potential calls, including 24 zero-charge contributions**.
- Each contracts `(250,250) @ (250,20)`, then `(20,250) @ (250,20)`, retaining
  only the diagonal of the second result.
- 97 explicit GEMMs including the electron–electron contraction. Nuclear–
  electron contractions contain about 129.6 million FP64 operations total.
- `CuESTPointPotential.compute_device` repeatedly calls `owner.assert_live`;
  that validates both complete basis snapshots. The present call chain
  rebuilds **288 basis snapshots** within these 48 potential calls.
- The loop has 120 explicit synchronizations (native completion and scope
  exits) and 192 device allocations, excluding surrounding setup.
- The electron–electron factors retain only 40Q of 800Q returned doubles:
  95% of the returned DFMO entries are discarded. This is output-volume waste,
  not evidence of a corresponding 20-fold runtime opportunity.

Thus skinny GEMMs are a factor, but cannot alone explain seconds of elapsed
time. Priorities: skip exactly zero contributions; directly reduce column
inner products instead of forming occupied off-diagonals; reuse bounded
workspaces; establish a sound immutable validation epoch; then batch active
atom contractions. Do not simply disable drift/ownership checks.

The potential plan is created once, not once per atom. Multiple charges in
one native potential call sum into one AO matrix; they are not independently
resolved atom outputs and cannot replace the fragment decomposition.

## Measurement limitation

The Python source tracer biases host-heavy GPU work. In repeat zero:

| Scope | Baseline (s) | Traced (s) |
|---|---:|---:|
| Whole energy | 32.9505 | 52.8477 |
| Fragment entry including factory/cleanup | 25.9020 | 44.0693 |
| Inner fragment workflow | 17.3122 | 31.4662 |

The 6.061 s electrostatics and 0.843 s localization medians in the step report
are **traced**. Their untraced term times remain unmeasured. The structural
counts above are source/runtime evidence, not measured time attribution.

## User-selected next experiment

- **protein83**, not a separate protein89 geometry.
- Spherical jun-cc-pVDZ, E10/D12, 922 AOs, JKFIT 4762, RIFIT 3820.
- **Two fresh timed runs per arm**. No Python source tracing.
- Two separate **48-thread CPU-only cpu-gnr** jobs, 360 GiB allocations and
  node-local scratch; one timed reference per 8-hour embers job.
- H200 job with eight host threads, 160 GiB allocation and a 4-hour limit.
- Small scaled/unscaled GPU water controls compare against the CPU build's
  saved references. No CPU reference or oracle runs inside the H200 allocation.
- Separate untimed CPU48 same-gauge oracle/comparison job after producers.
  Its failed/missing prerequisites cause a fail-fast exit, not a stranded
  `afterok` dependency or a fabricated speedup.
- Same 96 GiB Psi4 host budget in both arms; CPU/GPU hardware, threading and
  scratch differences must accompany any ratio.
- Two-run statistics retain both samples and their range. Timeouts/preemptions
  remain failed/incomplete evidence, with no automatic inferno escalation.
- A fresh CPU build requalifies the explicit CPU48/GPU8 oracle helper contract.
  Production scientific code remains unchanged.
- Existing native timer trees and execution counters are retained. New
  low-overhead term profiling must be separate from these clean baselines.

Larger work may amortize launch overhead, but does not remove the ordered-pair
launch loop or repeated basis reconstruction. Algorithm improvements still
need fixed-gauge correctness checks and untraced before/after measurements.
