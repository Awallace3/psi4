# FDDS cuEST implementation handoff

## New cycle authorized — acquisition started, no new grant established yet

On 2026-10-21 the user explicitly approved **one H200 / 8 CPU / 128 GiB /
four-hour inferno interactive allocation**, account gts-cs207-chemx.
This replaces the historical "no standing approval" state below, not the
terminal state of old job14214205. Frozen harness/controller
`089c4a63163257ee339ba48877a845c11cc84c9a` is staged pristine at
`/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-cycle-089c4a6316-20261021-v1`.
Twenty-four offline controller/storage/timer/observer guards passed. Narrow
reviewd952 identified a late reference-validation guard; it now runs before
any paid validation/replay and has an ordering regression. First reviewdbf9
hit a tool limit and is not treated as a pass.

Fresh native timer-only source6c133 built and passed **176 tests, zero
failures/errors/skips, 178.672s**, workflowb7f00ba8a.
Core SHA256 `147e56bdb3068476fa545c90e5e9ee8f8b3e979f9b103d72eead758d19e120e0`.
Local package:
`/fastscratch/awallace43/fdds-performance-20261021/native-density-6c133b4e5c-v1/build/stage/lib/psi4`.
Raw XML/logs/provenance:
`build_fdds_cpu/cycle-20261021/native-density-6c133b4e5c-v1/`.

Git stagingba02d3335 passed. Initial readonly preflightb8b80efdb passed at
11:16:44Z: no related queued/running FDDS request; project filesystem1.9TB
free; dependencies, accepted stage and protein cache exist. Test-only14250291
is simulated, NOT an accepted job. Scheduler estimateNovember4 is not a
guaranteed start.

Single acquisition workflowbb6dbd345 was started after these gates, with fresh
account/duplicate/test-only admission before dispatch, explicit04:00:00 and
durable request intent. Intended new run:
`/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-shared/20261021-fdds-4h-01`.
No accepted ID or grant has yet been audited; do not replay an ambiguous
request. No new GPU execution or speedup is established by this note.
Scope/gates: `devtools/benchmarks/fdds_cycle_20261021.md`.

## Previous allocation closed — no live GPU resource

Established watcherbda4c38b4 confirmed14214205 **TIMEOUT**, exit0:0,
elapsed08:00:08, at2026-10-21T03:53:00Z (`ALL-TERMINAL`).
The response-product and ALDA steps had already completed0:0 with verified
empty cleanup. Allocation expiry does not invalidate their accepted results.

The explicitly approved early-release attemptb5579598e ran at
08:49:40–08:49:42Z, after allocation expiry. Its fresh identity guard
received `squeue` exit1, `slurm_load_jobs error: Invalid job id specified`,
and stopped **before the `scancel` invocation**. No cancellation command,
other job change or blind replay occurred. Do not report an early-release
success: the actual allocation ended by timeout. No replacement resource,
participant resume or further compute was launched.
Local raw accounting/release evidence:
`build_fdds_cpu/allocation-14214205-terminal/`.
Root summary: `fdds-allocation-14214205-terminal.json`.

Latest numerically validated scientific source remains **dcc8ac9713/corec457**.
The later6c133 timer-only source is reviewed/staged but uncompiled and
resource-deferred in that cycle, not a successor GPU-validation pass.
The newly authorized cycle above must establish its own build/test evidence.

## Response-product iteration — independent GPU/timing audit passed

Scientific/test `dcc8ac971310530688920efd0b2ba587a6e5cfd7`,
harness `b776f35d59a6d517b9bd92b649b5ebffbea5a55c`, CUDA core
`c4579a401c5c7fb81d009694c569124636f9d7205668e3a6da20597bb649293f`.
Native176 passes/177.642s. Actual H20014214205 GPU **93 unique tests/
138 executions passed**, zero errors/failures/skips: focused44/5.747s,
SAPT1/11.065s, full93/21.288s. Both GDB inferiors exited normally.
Independent auditb9e468376 checked exact pins/imports/runtime libraries,
all15 planned samples, actual GPU routes,22 unchanged pseudoinverse calls,
finite energies and mandatory empty pre/post GPU process tables.
Helper exit0; cleanup verified2026-10-21T02:37:50Z, no quarantine.

| System | Native CPU s (one run) | Candidate GPU s (two runs) | CPU/GPU median | Prior metric GPU/new GPU median |
|---|---:|---:|---:|---:|
| Peptide24 |16.181435|13.438290 /13.409532|1.205419x|1.015969x|
| Nanotube42 |222.444475|109.921805 /109.361113|2.028835x|1.034405x|
| Protein83 |1664.982366|560.758781 /560.797561|2.969057x|1.025895x|

Same physical caches/science, eight threads/96GiB, fresh processes,
same allocation/node and verified private local-XFS backend. All five
controls/system retained, including two prior metric GPU controls.
Maximum candidate coupled/uncoupled errors versus native:
peptide5.956963985e-10/2.103676608e-12Eh,
nanotube7.718928004e-8/7.791047668e-11Eh,
protein1.050541662e-9/1.956602588e-11Eh, below unchanged1e-5Eh.
GPU60 response chains/12 metric/20 frequency/4 hybrid calls; native zero.
These are FDDS-only, excluding SCF and subsequent exchange-dispersion,
not whole-SAPT speedups.

Protein first matched GPU control: integration219.320→202.685s,
new response endpoint4.033s/60 calls, object246.263→245.456s.
Remaining inclusive frequency-contraction region95.403s, native22-pinv
84.173s and XC107.783s; do not sum nested timers or call the inclusive
region pure BLAS. Candidate peakRSS≈61.32GiB.
Actual runtime remains cuEST0.2.2 (conda0.2.2.2), CUDA13.3.29/
cuBLAS13.6.0.2. No solver, cutoff, quadrature or fitting-policy changes.

Evidence root:
`build_fdds_cpu/response-products-dcc8ac9713/gpu-14214205/{run,step,compile,compile-step}/`.
Remote run `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-response-products-01`.
Owner snapshot2026-10-21T02:38:54Z: RUNNING1:10:47 remaining; the
exclusive step has finished. No new allocation authority is inferred.
The next preparation/ALDA investigation led to the bounded diagnostic
below; compute requires fresh cleanup/budget gates within this grant.

### ALDA attribution diagnostic — independently accepted, not an optimization

Read-only investigationd0a3b711 found no justified next optimization from
the inclusive contraction/XC timers. The selected next step is a single
protein83 ALDA attribution replay on unchanged scientificdcc8/corec457.
Harness `c22bd043ae38cf91454e8b00d173f768f10da221` instruments private AST
copies only: seven driver sites plus six `_compute_fxc` sites, exactly
19 successful calls. It measures density projection, both overlap powers,
inclusive ALDA A/B, leveling, density eigensolver, LibXC exchange/correlation,
back rotation, undo leveling and potential conversions. Inclusive ALDA
records contain nested records and must not be summed with them.
No pinv, operands, multiplication association, cache, cutoff or core changes.

Seven stdlib AST/order/freshness/exception/namespace/publication guards pass.
Real water on the exact dcc8 native stage observed13 sites/19 calls/22pinv,
zero GPU routes; same-source uninstrumented control differs by
7.107304762e-12Eh coupled/0 uncoupled. This is an instrumentation gate,
not a speedup comparison. Native evidence:
`build_fdds_cpu/alda-profile/native-water-01/`.
Reviewdfe55 found no scientific-semantic defect and identified one marker
ordering bug. Fixed before GPU launch: audit remains pending until all
final source/harness pin/pristine checks pass, then VERIFIED is renamed
atomically. Runner-failure restoration/invalid-profile regression added.
SIGTERM/SIGKILL may leave no profile; absence is never success.

Workflowb9a12804e stages a fresh c22 harness (old6f818 stage stays unused),
requires all prior cleanup, free common lock, accepted93/138 validation,
exact pins and fresh RUNNING allocation time≥1080s, then uses only the
existing shared helper `fdds-alda-profile-c22b-v1`:900s outer bound,
840s process plus30s kill grace. It does not build or submit a new job.
One diagnostic replay, no CPU ratio, no extrapolated protein speedup and
no silently omitted controls. Exact reference libraries/import/core/cache/
science/8threads/96GiB/localXFS and GPU60/12/20/4 routes remain required.
Remote output:
`/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261021-alda-profile-01`.
Workflowb9a12804e/helper finished0. Independent auditbd111c6d3 confirmed
all13 sites/19 successful calls, exact source/core/driver/AST/harness hashes,
all reference libraries/options/cache/imports and60/12/20/4 GPU routes,
22 original pseudoinverses and finite energies. Coupled difference from
the accepted dcc8 GPU reference1.265785506e-9Eh; uncoupled0. Empty pre/post
GPU process tables and cleanup verified2026-10-21T03:01:13Z, no quarantine.

Measured projection **97.170512s** dominates the105.762707s XC envelope.
Overlap powers2.526359/2.531075s; inclusive ALDA A/B1.694546/1.680321s.
Across both monomers, nested leveling1.602044s, density diagonalization
.476956s, LibXC exchange/correlation.000422/.002192s, back rotation
.388410s and undo leveling.825529s. These nested records are not additive
with inclusive ALDA. Bookkeeping measured≥.000382555s, excluding added
dispatch/argument evaluation; do not correct the559.695448s diagnostic
whole wall into a new speedup benchmark.
Preservation: `build_fdds_cpu/alda-profile/gpu-14214205/{run,step}/`.
Snapshot03:02:03Z RUNNING47:39 remaining, not a later budget guarantee.
This rules out dense ALDA/LibXC as a large next offload target.

### Timer-only projection split — reviewed/staged, resource-gated DEFER

Source/harness `6c133b4e5c63548d6d2228075058969464a6a0b5` adds only three
coarse RAII timers outside complete OpenMP loops: AO integrals/collapse,
inverse metric and auxiliary three-center overlap/collapse. Expected one
call per region for the physical replay. One offline byte-reconstruction
test passes: removing the timer class/three starts/three stops reproduces
the exact accepted dcc8 C++ bytes. Bash syntax/diff checks pass. Initial
reviewd3d553 failed its child turn limit and produced no committed answer,
not a review pass. Replacement reviewd8a9e9 returned no confirmed bug:
timers surround complete loops outside OpenMP, labels/lifetimes/stops are
safe and all external native/GPU/budget/cleanup gates remain mandatory.
Git-only stageb611 passed exact6c133/pristine/byte-reconstruction proof at
`/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-density-profile-6c133b4e5c-v1`.
No numerical/compilation result for6c133.
Accepted scientific dcc8/corec457 and prior stages remain immutable.

ALDA preservation and static/source gates passed, but the immediately
fresh owner snapshot2026-10-21T03:15:26Z showed RUNNING34:16 remaining:
**2056s < required2340s** (284s short). Thus **DEFER: zero build, native/GPU
test or diagnostic executions for6c133**. All prior steps had verified
empty cleanup, no quarantine and a free common lease. No allocation-side
compute, expiry extension, new allocation, participant resume or job-control
action was taken. The required window includes build≤900s, native176/
GPU93 gates, one≤840s protein diagnostic, cleanup and expiry grace; no
gate was silently skipped to squeeze work into the expiry.
This is not another15-control campaign and cannot establish a new ratio.
Resource decision for later: explicitly approved sufficient allocation time
after readiness gates; no standing new-job authority. Historical early
release was later explicitly approved, but its guarded attempt found the
allocation already expired; see the final accounting section above.

### Historical preparation and scope

Frozen scientific/test source `dcc8ac971310530688920efd0b2ba587a6e5cfd7`.
Separate fresh native source/build at
`/fastscratch/awallace43/fdds-performance-20261020/native-products-dcc8ac9713/`;
native build/test workflowb4c2aef33 **passed176 tests**, zero failures/errors/
skips,177.642s, CPU core
`61c19a574493d7399129faf2f1d7a2703f509260b8d63d1f8c4161b2ef311530`.
Evidence: `build_fdds_cpu/response-products-dcc8ac9713/native/`.
Prior ada/core6f829 and fd2/core92e5
stages/evidence stay immutable.

The accepted diagnostic at harness25bf9d87cfe6657d7e27e4f53f03af26fcefc019
observed8 product sites/80 calls (all3820-square), **29.021014s** total.
Every group took3.619–3.636s across10 calls; original association/operands,
22 general pseudoinverses, cutoff/science were unchanged. Strict library/
core/cache/options identity and energy checks passed. Instrumentation
bookkeeping measured≥.000655s, excluding added Python dispatch; the whole
diagnostic wall is not a benchmark. Both diagnostic attempts are preserved
in `build_fdds_cpu/product-profile-14214205/`.
Attempt01 remains FAILED: importing Psi4 before NumPy selected GNU instead
of Intel MKL threading. The fixed NumPy-first wrapper retained exact library
checking, with a red→green regression; no provenance relaxation.

The next candidate borrows only exact C-contiguous FP64 square NumPy
buffers, keeps each two-GEMM intermediate on device and returns one array.
No host packing, input mutation, generic DFHelper changes or solver changes.
Noncontiguous/empty/unsupported/threshold/capacity cases fall back before
enqueue; GPU runtime errors propagate after stream draining.
Hybrid `Rtinv` transpose views deliberately remain native, so only6 sites/
60 chains across ten quadrature points execute on GPU, targeting≈21.78s of the measured
29.02s, not the inclusive95.66s contraction timer. Even eliminating this
target entirely would provide only≈1.04x additional whole-FDDS speedup.
The accepted measured result is reported above.

Added5 native and7 GPU layout/freshness/alias/preflight cases; integrated
tests additionally check unchanged pseudoinverse inputs/counts/rcond and
actual60-chain routing. Full GPU gate93 unique/138 executions and native176
now pass. The separate harness retained all15 matched
baseGPU/candidateGPU/nativeCPU controls, with no dropped samples.
Matched timings require prior cleanup, native/GPU gates, static review and
a fresh remaining-budget check; no new allocation. An isolated compile-only
step may run while the separate workstation builds its native compatibility
stage; neither is an active timing sample or GPU correctness execution.

Read-only reviewdff5b462dee167a895c10de234562da80 found no confirmed
correctness/safety defect in dcc8. Owners/stream ordering/three-buffer reuse/
exception draining and counts were checked statically. Capacity exhaustion,
overflow and post-enqueue faults remain un-injected. NumPy's pinv output
layout is not an API guarantee: the pinned local probe confirmed C output/
F transpose; actual GPU60-chain gates and physical replays now establish
this for the pinned NumPy/runtime.
The full-SAPT test checks frequency routing; integrated FDDS and physical
benchmarks separately require the new response endpoint.

## Latest audited performance status — supersedes historical progress below

Scientific8dc714/test-driverfd2c09/core92e5:171 native tests pass and69 unique
GPU tests pass (90 executions including focused repeats), zero skips/errors/
failures. Actual shared H20014214205,8threads/96GiB, same source/core/cache/
options/libraries for each fresh-process CPU/GPU control; SCF is excluded.

**Local-XFS baseline repeats complete:** two runs per arm/system, all12
samples independently audited and preserved, cleanup verified.
Peptide CPU16.144716/16.167149s vsGPU13.763155/13.792320s: **1.172611x**.
Nanotube CPU222.794186/222.049114s vsGPU121.374003/121.205100s:
**1.833807x**. Protein83 CPU1669.110843/1667.379830s vsGPU644.636978/
647.064827s: **2.583019x**. Ratios use per-arm medians.
Maximum coupled errors across all CPU/GPU sample pairs:
3.456794130e-10/8.787432640e-8/9.467751930e-10Eh, respectively;
uncoupled errors7.365963742e-12/2.150354825e-10/2.332058678e-11Eh.
Actual20 frequency and4 hybrid GPU routes versus zero CPU routes.
Protein peakRSS≈61.04GiB.
Preserved locally at
`build_fdds_cpu/performance-gpu-14214205/var-local-v2/`.
This full case succeeds on verified private localXFS after the preserved
network-scratch CPU failure; the underlying networkEIO cause remains unknown.

Earlier network-scratch first pairs: peptide1.190898x and nanotube2.034857x
(errors≤6.83e-11/9.62e-8Eh). These are separate backend measurements, not
mixed with the local-XFS protein row into a final benchmark aggregate.
Those network-backend ratios remain preliminary one-pair measurements;
the local-XFS figures above now have two-per-arm repeat evidence.
Five-minute monitoring and the official allocation watcher remain active.
No merge, new allocation, overlapping timing, settings relaxation or shared
environment mutation.

## Metric-fold candidate — GPU and matched timing audit passed

Scientific/test pin `ada1a1d2128d6d6f3f5b89952b08f6ffd4ca5917`,
harness `60ee73db408c58be208367abef7bc9a894f6d860`, CUDA core
`6f8294594134277ac55ae1a3b87d9003367a451b4f8a506a16e1a2b8a823de68`.
Shared H20014214205: **81 unique GPU tests /114 executions passed**,
zero failures/errors/skips (32 focused6.155s,1 SAPT10.977s,81 full21.143s).
Both GDB inferiors exited normally. Exact import/library provenance and
empty pre/post GPU process tables were independently audited; terminal0,
cleanup verified at2026-10-21T00:21:28Z.

All15 fresh-process matched samples were retained: baseline GPU0,
candidate GPU0, candidate CPU0, candidate GPU1, baseline GPU1 per system.
Same physical cache, runner, science,8threads/96GiB, allocation/node and
verified private local-XFS backend; only baseline/candidate source/core differ.

| System | Candidate CPU s (one run) | Candidate GPU s (two runs) | CPU/GPU median | Prior GPU/new GPU median |
|---|---:|---:|---:|---:|
| Peptide24 |16.252794|13.632431 /13.665514|1.190770x|1.010803x|
| Nanotube42 |221.751859|113.101500 /112.476784|1.966075x|1.072547x|
| Protein83 |1665.228173|579.193648 /577.174427|2.880101x|1.117253x|

Maximum candidate coupled/uncoupled deviations from its same-allocation
native CPU control: peptide3.288835107e-10/2.387883294e-12Eh,
nanotube4.948029657e-8/1.248341437e-9Eh,
protein1.036713920e-9/3.861587612e-11Eh. All meet unchanged1e-5Eh.
Candidate GPU executed12 metric folds,20 frequency calls and4 hybrid calls;
native controls executed zero GPU calls. These remain **FDDS-only**, not
SCF, subsequent exchange-dispersion or whole-SAPT speedups.

Protein first matched GPU sample: object316.216→247.041s;
DF transform207.215→138.573s, including metric phase133.653→63.991s
and its nested multiplication84.129→14.208s. Do not sum nested timers.
Remaining measured phases: frequency220.488s (includes95.663s FDDS
contractions and83.912s native general-SVD pseudoinverses), XC106.866s.
No attribution yet for the frequency remainder or a next-candidate speedup.
Preserved complete logs/results:
`build_fdds_cpu/metric-fold-ada1a1d212/gpu-14214205/{run,step}/`.
Remote root: `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-metric-fold-01`.
Owner's post-terminal snapshot: RUNNING3:27:39 remaining; a fresh budget
gate is still required before another allocation-side step.

### Historical candidate preparation

Frozen scientific/testpin`ada1a1d2128d6d6f3f5b89952b08f6ffd4ca5917`,
controller`60ee73db408c58be208367abef7bc9a894f6d860`. Static review found
no confirmed new defect. CPU-only build and **171 tests passed**, zero
failures/errors/skips,175.704s, core
`c08b6341ce7b29099225fc83fde6dec336c6bc3e38caba0ca5840b7c6722d23f`.
Local XML/provenance: `build_fdds_cpu/metric-fold-ada1a1d212/native/`.
Actual candidate GPU compilation,81-test numerical/route gate and timing
were run in gated workflowbb600889a (9000s exclusive bound); it required
a fresh≥9180s remaining-time check and preceding cleanup. The prior audited
snapshot wasRUNNING5:03:26 remaining. Old fd2/core92e5 baseline stays frozen.

The new seam is FDDS-only metric-fold FP64 multiplication with one resident
prepared metric per transform and reusable device blocks. Generic DFHelper
defaults, host budgets, powered metrics, threshold/rank/pinv policies and
X/Y algorithms are unchanged. Twelve added GPU regressions cover real
FDDS endpoints/controls and four nonsymmetric/changing-block layouts.
Runtime fault injection, forced capacity exhaustion and proof of actual
production partial-tail routing remain gaps; static checks are not tests.

## Performance iteration in progress

The validated result below applies to scientific9d4/test-driver3b21, **not**
the new hybrid X/Y optimization candidate. See `fdds-performance.md` for
profiling, the new isolated CPU/GPU gates and measured speedup status.
CPU baseline job14198087 completed and audited all three systems on24 CPUs:
**20.446/240.153/1592.476 s FDDS** for peptide24/nanotube42/protein83.
This is old scientific9d4 on a CPU node, not a matched GPU speedup comparison.
Reviewed X/Y candidate scientific8dc714/test-driverfd2c09 passes the
expanded native gate: **171 passed, no failures/errors/skips,240.780 s**.
**New GPU gate now audited on shared H20014214205:**69 unique tests passed,
zero failures/errors/skips, full22.790s; preceding focused20 passed7.937s
and real SAPT1 passed11.425s (**90 actual executions**, not90 unique).
Both GDB inferiors exited normally; import/core92e5/library provenance,
VERIFIED.json and empty pre/post GPU process tables were audited. New X/Y
whole/blocked/equal/unequal/dense-occupied routes actually ran.
**First matched physical timing (preliminary, one pair):** peptide24 CPU
17.749984s/GPU14.904702s, **1.190898x**, same source/cache/allocation8threads/
96GiB, actual20 frequency/4 X/Y GPU calls, energy error≤6.83e-11Eh.
**Nanotube42 first pair:** CPU285.588827s/GPU140.348366s, **2.034857x**,
same matched controls; coupled-energy error9.61322e-8Eh<1e-5.
Both ratios are preliminary one-pair measurements; protein83 and ABBA
repeats remain pending. No cross-node baseline ratio.
Protein83 first CPU/GPU pair is running exclusively asbd4b572c6 after owner
verification of the preceding participant's timeout/cleanup; no overlapping
GPU or CPU timing. Further participant diagnostics are offline/on HOLD.
Latest user-requested snapshotbf25d9b90 timed out (Phoenix110s, task125s):
current protein progress UNKNOWN. Prior client log proves launch with7:21:52
allocation remaining/10800s bound. Leave existing client/watch untouched;
all new steps HOLD until actual completion/state/results/cleanup proof.
Actual protein terminal then arrived: **CPU control failed** with reported
DFHelper network-scratch writeEIO/errno5; GPU never started, both results
absent. Cleanup independently verified empty/no quarantine. Preserve failed
attempt; preparing verified allocation-local scratch and a fresh matched
root (same science/core/memory/tolerances) before any recovery measurement.
Preserved artifacts `build_fdds_cpu/performance-gpu-14214205/`:
printed integrated coupled/uncoupled FDDS errors≤1.1102230246251565e-16/0Eh;
real native-XC SAPT dispersion error≤1.2705494208814505e-20Eh.
Optimization job **14199355** started promptly on inferno/H200 but failed
startup after6 s, **zero tests/benchmarks executed**. Site CUDA12 nvcc kept
PATH precedence after activation, triggering the strict pinned-toolkit guard.
Explicit post-activation env/bin rebinding fixes the reproduced guard failure;
fresh preflight/recovery is in progress. Scientific/test pinfd2c09cd0c and
frozen harnessf26b3b6d1b/sourcev2 remain unchanged. CPU compile-only14199512
also failed the same guard before compilation. No candidate GPU correctness
or speedup result exists; see terminal markers and `fdds-performance.md`.
Corrected CUDA-enabled build job **14200049** completed0:0 in14m57s on explicit
inferno/cpu-small/8CPU/48GiB/1h ($.07 maximum), sourcefd2c09/harnessebdb1950be,
watcherbb6a86037, fresh own compile-v2 checkout. Read-only auditb12bed014
verified the pinned import/core92e5bff6641d..., actual CUDA/cuEST libraries
and BUILD_ONLY completion. This establishes compilation/import, **not** GPU
numerical validation. Recovery harnessc289c502 uses that stage read-only
with early per-system pair checkpoints. **14210379** accepted on
**gts-cs207/gpu-h200/inferno**, one H200/8CPU/128GiB/eight hours, effective
QoS verified. Watcher **bcde45e0e**. It requires21 focused plus69 full
executions before matched timings; no GPU result for this candidate yet.
Source/testfd2c09, harnessc289c502, read-only core92e5bff6641d...
Run `.../runs/psi4-fdds/20261020-optimization-02`.
**Superseding instruction:** user explicitly approved replacing queued14210379
with one bounded shared interactive allocation. It was cancelled only after
confirming PENDING, with no test executions; no other jobs changed. New shared
allocation is being prepared (FDDS sole owner, serial locked/exclusive steps,
same1H200/8CPU128GiB8h inferno ceiling). PySCF has an early bounded ready
preflight; parent/disp0 are not ready and will not hold it idle.
Actual replacement request **14214205** appears in the corrected salloc log
as queued/waiting, controller9a026ca149, run
`.../runs/psi4-cuest-shared/20261020-interactive-02`. No grant/full UUID/
allocated manifest yet. Disp0 reported separate SSH access timeouts; this
does not prove this request failed or was released. One bounded owner-side
state query **bb1721fec timed out after110s (exit124)**. Current allocation
state is **UNKNOWN; all GPU steps paused**, no duplicate or further SSH retry.
Existing acquisition client is untouched; wait for its automatic notification,
then re-establish actual manifest/state/cleanup proof before any launch.
**Recovery snapshot13:27:16Z:** bounded owner reconciliationb4cf26e3a passed;
14214205 is **PENDING/Priority**, gts-cs207-chemx/inferno, zero elapsed/no node,
and the pinned manifest says pending. State is known again, but all GPU
steps remain HOLD until grant/full UUID/cleanup/GO. Wait the existing client;
no repeated queries or duplicate requests.
**Actual grant:** acquisitionb022d3628 completed0, H200 job14214205 on
`atl1-1-03-019-14-0.pace.gatech.edu`, UUID
`GPU-5fdf3697-20d4-8564-a920-317ecda4e1a9`, driver615.71.09.
Manifest now allocated; initial FDDS21+69 gate **b57a73af9** is running
under the common exclusive lock (1200-second bound), before PySCF's assigned
window or timing. Lifetime watcherbda4c38b4. No GPU pass/speedup claim yet.

## Historical pre-optimization result: isolated GPU validation PASSED

- **Implemented and validated; no merge back.** Real FP64 occupied-block FDDS
  contractions use persistent device sums on the shared cuEST stream, with
  existing opt-in/threshold routing, native fallbacks, checked memory budgets,
  cleanup and runtime-error propagation. Equations and QCVariables unchanged.
- **Job14184910**, explicit free **embers**, H200/4 CPUs/16 GiB/15 minutes:
  job/batch/payload **COMPLETED 0:0**, elapsed **54 s** (payload51 s).
  **Full FDDS suite57 passed, zero failures/errors/skips, 18.596 s**:
  46 native/control cases plus **11 GPU cases**, all actually executed.
  Preceding focused checks:8 passed4.140 s plus real SAPT GPU test1 passed
  11.254 s. Thus66 executions including repeats, **57 unique tests**, not66.
  Both GDB inferiors exited normally without signal/breakpoint hits.
  Exact JUnit counts, `FOCUSED_VERIFIED.json`, `VERIFIED.json`, `COMPLETED`,
  actual timers/block assertions and pinned staged import were audited.
- Scientific binary/source:
  **`9d4b1cd4aab718dec1f7b831a53dd54d68f2a609`**.
  Test driver (budget/output-helper regressions):
  **`3b21adb662f47fc4b89f1cbb89a44592f7a1d10f`**.
  Replay harness:
  **`ce2ab6b38d157b3fb805708bf3da74a19e80c208`**.
  All three remote tracked trees verified clean and pinned after completion.
  No production-source/build/installed-Python difference between binary and
  driver pins; the integrated GPU stage was reused **read-only**, not rebuilt.
- GPU stage:
  `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-integrated-20261017-v1/build_fdds_validation/stage/lib/psi4`.
  Psi4 **1.12a1.dev802**, core SHA256
  `68e34452afa80002d4bb7fb1b56297e19389617fcb3ee3b926abc0b5fcead0b7`.
  Release GPU build from job14181925; successful compiler frontend probes
  after job-local GCC_ROOT isolation. Shared dependency environment unchanged.
  Actual runtime **cuEST0.2.2** (conda libcuest/libcuest-dev0.2.2.2),
  **CUDA13.3.29/cuBLAS13.6.0.2**, H200 driver615.71.09.
  `import.json` records exact resolved library paths and SHA256 hashes;
  the site nvcc12.6.1 is **not** the resolved runtime version.
- Largest full-suite contribution/reference absolute error:
  **7.105427357601002e-15**; all four integrated nonhybrid/hybrid,
  equal/unequal CPU/forced-GPU dispersion differences printed **0 Eh**.
  Real SVWN/cc-pVDZ He-dimer with native supported XC response:
  CPU dispersion **-1.090781960579051e-6 Eh**,
  forced-GPU **-1.0907819605790383e-6 Eh**, difference
  **1.2705494208814505e-20 Eh**.
  CPU total **-3.5131036459929232e-6 Eh**, forced-GPU total
  **-3.5131036441000883e-6 Eh**, difference **1.892834881076031e-15 Eh**.
  Across disabled/threshold/forced controls all six SAPT QCVariable errors
  are at most **6.6574573662142395e-15 Eh**.
  Forced real SAPT passed the20-call FDDS timer assertion after actual cuEST
  J/K builds, which also exercised the shared handle HOST-mode check.
- FDDS-only elapsed seconds, same source/H200 allocation, excludes SCF:

  | Fixture | Disabled CPU | Huge-threshold control | Forced GPU | GPU contraction timer |
  |---|---:|---:|---:|---:|
  | Equal nonhybrid | .011230 | .010759 | .012674 | .000834985 |
  | Equal hybrid | .016645 | .016663 | .021933 | .002517654 |
  | Unequal nonhybrid | .010739 | .010768 | .012748 | .000845762 |
  | Unequal hybrid | .016821 | .016780 | .021904 | .002529294 |

  Tiny transfer-dominated correctness fixtures, under GDB; **no representative
  throughput or speedup claim**. GPU timer includes transfers/preparation.
- Latest CPU-only gate **165 passed, zero failures/errors/skips,179.023 s**,
  including46 FDDS and67 CPSCF. Own Release/Ninja stage, cuEST/Einsums OFF,
  scientific9d4/test-driver3b21, outside-source import checked. CPU core SHA256
  `5b2825a1dd99a8f7e40c5a24ec2b14df2c1f7436b8d1197ea7de11c500a48358`.
  Evidence `build_fdds_cpu/integrated-9d4b1cd4aa/native-output-fixed{.xml,.log,-provenance.json}`.
- Local GPU evidence and reproducible audit:
  **`build_fdds_cpu/gpu-14184910/{audit.py,audit.json,accounting.txt,remote-evidence.tar.gz,results-14184910/}`**.
  Run `python3 build_fdds_cpu/gpu-14184910/audit.py` to recheck counts,
  provenance, native exits, timers and errors; audit records artifact hashes.
  Remote run
  `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261019-integrated-focused-02`.
  Watcher b07d6597e completed; no new allocation needed.
- Remaining limitations: device-capacity-driven reduction/rejection/fallback
  and post-enqueue CUDA/cuBLAS fault injection have **not** been runtime-tested.
  Unsupported layouts/symmetry, empty/small/native routes and host-budget
  exceptions/recovery are covered; runtime errors are not silently retried.
  Remaining CPU work: integrals/DF preparation, ALDA kernel, hybrid QR/X/Y,
  coupled susceptibility products/solves, quadrature and exchange-dispersion.
  This FDDS work adds **no MGGA/VV10/GPU-XC response kernels**. End-to-end
  tests deliberately use `CUEST_XC=false`; upstream LDA/GGA response support
  was integrated separately, not established by this FDDS GPU suite.
- Earlier failed jobs/stages/evidence and coordinator worktrees/jobs remain
  immutable. Fixture-budget and test-output-lifetime causes are confirmed and
  fixed, not attributed to FDDS equations or JK lifecycle. Old13991729 was
  not cancelled or modified; its current scheduler state was not re-queried
  in this terminal audit. No local GPU success, environment mutation, paid
  retry, auto-merge or throughput claim.

## Historical record (superseded status snapshots below)

The remaining sections preserve earlier evidence and decisions. Their
“pending”/“blocked” labels describe those earlier pins, not current14184910.

## Same allocation extended to gated full-suite validation

- User explicitly requested using the allocation for the full FDDS suite once
  focused checks pass, and authorized brief hold/release of **14184910** to
  update only its unstarted replay harness. That update succeeded; job ID,
  scientific binary9d4b1cd4aa, test driver3b21adb662, 15-minute limit and free
  embers/H200 route are unchanged. No new submission/cancellation.
- Replay harness advanced from e124e5cfba to
  **ce2ab6b38d157b3fb805708bf3da74a19e80c208** while held; clean tree and
  no scientific/test/build-code changes were verified before release.
  The previous harness commit remains available in git. Transition and
  before/after scheduler snapshots are recorded in run02.
- Execution plan: **9 focused passes first**, then **all57 FDDS tests** in
  the same allocation under bounded GDB (300 s, 15 s kill grace).
  Full success requires exact57 executed/no skips/failures/errors and normal
  GDB inferior exit. `FOCUSED_VERIFIED.json` is kept separately; only a verified
  full run writes final `VERIFIED.json` and allows `COMPLETED`.
  There are **66 total test executions including repeats**, not 66 unique tests.
  Six synthetic gate checks accepted normal completion and rejected partial
  counts, skipped/failed tests, native faults and abnormal exits; not GPU data.
- Effective post-release state PENDING, QOS embers. Existing watcher
  **b07d6597e** continues on the same job. No numerical/full-suite or speedup
  conclusion before terminal accounting/provenance/artifact audit.

## Diagnosis complete; corrected SAPT GPU confirmation still pending

- Final CPU gate with corrected test driver
  **3b21adb662f47fc4b89f1cbb89a44592f7a1d10f** and integrated scientific
  binary9d4b1cd4aa: **165 passed, zero failures/errors/skips, 179.023 s**
  (46 FDDS, 67 CPSCF, 6 SAD, 6 lifecycle, 8 basis-parser, 32 grid-switch).
  Evidence `build_fdds_cpu/integrated-9d4b1cd4aa/{native-output-fixed.xml,native-output-fixed.log,native-output-fixed-provenance.json}`.
  The unchanged CPU core SHA256 is explicitly rechecked in provenance.
- Integrated focused job **14181925** ran and failed on its own, no
  cancellation: free embers/H200, **6m46s**, exit1. Own integrated GPU build
  succeeded. **8 contribution/budget tests passed**, zero errors/failures/
  skips, **3.850 s**; all four equal/unequal nonhybrid/hybrid comparisons
  printed zero CPU/GPU matrix error and passed actual block/timer routing
  assertions. The final SAPT case then SIGSEGV'd in Printf with the old
  helper; no SAPT XML/VERIFIED/COMPLETED, so not a focused-suite pass.
  Preserved evidence `build_fdds_cpu/gpu-14181925/`.
- GPU binary pin9d4b1cd4aa, original focused harnessa6233fffac; staged core
  SHA256 `68e34452afa80002d4bb7fb1b56297e19389617fcb3ee3b926abc0b5fcead0b7`,
  version1.12a1.dev802, cuEST0.2.2 (conda0.2.2.2), CUDA runtime13.3.29,
  cuBLAS13.6.0.2. These are correctness fixtures, no speedup claim.
- Corrected replay uses existing integrated GPU stage **read-only**; scientific
  C++/installed Python/build configuration diff9d4b1cd4aa..3b21adb662 is empty.
  Test driver remains frozen3b21adb662; replay harness
  **e124e5cfba24a3a046aee4e5f055d1cdfc92dbfa** verifies the binary's exact
  source pin and core hash, records linked library hashes, and never rebuilds
  or overwrites it. New run02, H200/4 CPUs/16 GiB/15 minutes/free embers.
  Staging/dry-run passed; **14184910** accepted on explicit free embers/H200,
  4 CPUs/16 GiB/15 minutes. Live headroom30/50, duplicate/pin/clean-tree checks
  passed, effective QoS confirmed embers, initial state PENDING.
  Run `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261019-integrated-focused-02`;
  watcher **b07d6597e**, 300 s, 48-hour watcher deadline.
  Neither old job nor stage was cancelled or modified. Required nine actual
  passes/no skips plus normal GDB inferior exit, VERIFIED.json/COMPLETED and
  pinned import/core/library provenance; no full suite or speedup claim.
- Atomic `fdds-gpu-diagnosis-complete.json` reports diagnosis/CPU-fix completion
  with outcome **blocked**, not a full GPU pass. Remaining gates: corrected
  SAPT GPU test, full57-case FDDS suite, capacity-driven reduction/fallback and
  post-enqueue fault-injection runtime coverage. All prior evidence/pins remain
  immutable; no coordinator dirty work copied and no merge back.

## Native crash diagnosed: test helper closed the global output stream

- Recovered diagnostic **14006185** completed Oct16 on free embers in
  **27 s**. Contribution process exited1, exact fail-fast traceback confirms
  the unequal/nonhybrid CPU-reference memory exception; JUnit **1 failed**,
  2.956 s. Separate GDB process exited0 **after catching SIGSEGV** in
  `psi::PsiOutStream::Printf`; no SAPT JUnit exists, so this is not a pass.
  The archived stage core SHA256 exactly matches the failed13992504 build.
  Evidence `build_fdds_cpu/diagnostic-14006185/{accounting.txt,remote-evidence.tar.gz,logs/,results-14006185/}`.
- Minimal CPU-only reproduction on the integrated binary: open a native
  outfile, close it, call public `psi4.set_num_threads(1)`. GDB catches the
  same SIGSEGV/Printf frame. Keeping the stream open exits normally.
  No SCF, J/K, FDDS or GPU call is required. This confirms an invalid
  test-helper output lifetime, not a demonstrated backend scientific bug.
  Evidence `build_fdds_cpu/integrated-9d4b1cd4aa/{reproduce_closed_output.py,minimal-open-gdb.log,minimal-closed-gdb.log}`.
- `_run_saptdft_fdds` now opens output before any setup messages and restores
  the caller's previous native stream in `finally`. New repeated-control
  regression went red (**1 failed**, 3.54 s) against the old helper; fixed
  native SAPT + sequential-control subset **2 passed**, zero failures/errors/
  skips, **7.233 s**. Evidence `output-{red,green}.{log,xml}` in that directory.
  Full prospective FDDS suite is now **57**, focused GPU suite still **9**.
- User explicitly authorized replacing **14181925 only if still queued**,
  after the CPU gate passes; if running it must remain untouched. No
  cancellation has yet been issued. Prospective corrected focused payload
  uses a new v2 checkout/run02; original v1 scientific/harness pins and
  failed/diagnostic evidence remain frozen.

## User-authorized upstream integration and new CPU gate

- User authorized pulling committed `saptdft_cuest` work into this FDDS
  branch. Merge **9d4b1cd4aab718dec1f7b831a53dd54d68f2a609** integrates
  upstream **ca21ef8645a73263281480454b730085506adca5**, including committed
  SAD/lifecycle and LDA/GGA cuEST-grid XC response work. No dirty coordinator
  edits were copied, and no merge back into the coordinator branch occurred.
  FDDS scientific C++ and fixture corrections remain intact. One option-doc
  conflict was resolved retaining both routing descriptions.
- Own CPU-only Release stage rebuilt successfully, cuEST/Einsums OFF.
  **164 passed, zero failures/errors/skips, 209.808 s**:
  45 FDDS, 67 native CPSCF, 6 SAD, 6 lifecycle, 8 basis-parser, 32 grid-switch.
  Evidence `build_fdds_cpu/integrated-9d4b1cd4aa/{build.log,native.xml,native.log,provenance.json}`.
  Actual outside-root staged import asserted, version `1.12a1.dev802`,
  core SHA256 `5b2825a1dd99a8f7e40c5a24ec2b14df2c1f7436b8d1197ea7de11c500a48358`.
  No CUDA/cuEST linkage in this CPU gate.
- Focused GPU harness **a6233fffac7ce1311f8f3a0164adea55708f6c1f**
  prepares an entirely new source/build/run at scientific pin9d4b1cd4aa.
  Only 8 contribution/budget tests plus the final SAPT GPU test under bounded
  GDB (**9 targeted tests**, not the full 56-case suite). Separate processes
  preserve the crash stack even if contributions fail. Gate checks both
  subprocess statuses, exact JUnit counts/no skips, and normal GDB inferior
  exit; SIGSEGV/attachment-breakpoint output cannot become a false pass.
  Five synthetic artifact-gate checks passed (normal exit accepted; native
  signal, attachment breakpoint, skip and timeout rejected); these are
  harness checks only, not GPU correctness evidence.
- Staging and dry-run passed; exactly one new job **14181925** was accepted
  on explicit free `embers`/H200, 8 CPUs/64 GiB/one-hour build+focused-validation
  limit. Live headroom 32/50, unique job-name/comment/run checks passed;
  effective submission QoS confirmed embers, state PENDING.
  Source `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-integrated-20261017-v1`,
  separate harness checkout with `-harness` suffix. Run directory
  `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261017-integrated-focused-01`.
  Watcher **b945ab472**, standard 300 s interval, 48-hour watcher deadline.
  No full GPU rerun.
  Frozen previous stages, failed13992504 evidence, and diagnostic14006185
  scientific/driver pins remain immutable. Native crash is not yet established
  fixed by upstream integration; targeted GPU evidence is still required.

## Confirmed fixture correction (CPU evidence; GPU diagnosis pending)

- Local direct replay of the original unequal/nonhybrid test stopped in
  CPU reference construction with the FDDS host-memory exception, before
  any GPU comparison. This independently confirms the B-only-budget defect:
  262 available doubles versus A's required 276. Evidence
  `build_fdds_cpu/diagnosis-local/fixture-budget-red.log`.
- New real-interface regression first went red (**2 passed/1 failed**, fail-fast,
  1.65 s), then all four equal/unequal nonhybrid/hybrid cases passed
  (**4 passed, zero errors/failures/skips**, 1.581 s) after the fixture fix.
  Evidence `build_fdds_cpu/diagnosis-local/budget-{red,green}.{log,xml}`.
- Comparison budget now satisfies both monomer preflights and preserves B's
  two-block partial tail. Block assertions derive from monomer dimensions and
  the host occupied-block contract: unequal nonhybrid is **9** GPU blocks over
  three frequencies, not 12; other cases remain 12. Numerical tolerances
  unchanged. New four-case native budget regression also checks independent
  contraction references; prospective full GPU suite expands to **56**.
- Full expanded native FDDS+CPSCF gate at test-driver pin
  `2a6b3b63da2bee8b6718b81f18e6579c5e5e4346`: **55 passed, zero failures,
  errors or skips, 18.239 s** (45 native FDDS + 10 native CPSCF; GPU cases
  deliberately deselected). Evidence
  `build_fdds_cpu/diagnosis-local/{full-native.xml,full-native.log,full-native-provenance.json}`.
  Scientific C++ unchanged, so the existing CPU-only binary at scientific
  pin8810da5743 was used without rebuild; core SHA256 and native-only ldd
  provenance are recorded explicitly. This is not GPU validation.
- Frozen diagnostic job14006185 still uses original source/tests to obtain the
  requested GPU traceback/GDB stack; it is not repointed to these changes.
  Native crash cause remains unknown, no full GPU rerun submitted, and no
  diagnosis-complete marker or full-GPU pass claimed yet.

## Focused diagnostic allocation (pending)

- Authorized one short explicit free-embers job **14006185**, H200/4 CPUs/
  16 GiB/15 minutes, account gts-cs207-chemx. Dry-run accepted gpu-h200;
  live headroom was 49/50 and unique name/comment/run checks passed.
  Effective submission QoS verified `embers`, initial state PENDING.
- Scientific/build pin remains `8810da5743cc7350e7620ecb708cdd3e47baa739`;
  existing v3 package is used read-only without rebuild. Diagnostic driver
  pin `da356228a300f80d323524d29d97a3435888d4c1`, distinct from original
  build harness `e60b32507b3069192032705b1d4161dd38886511`.
- Driver checkout: `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-diagnosis-20261013-v1`;
  run: `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261013-diagnosis-01`.
  Separate processes select only unequal nonhybrid contributions (120 s)
  and the SAPT GPU case under batch GDB (240 s), each with 15 s kill grace.
  GDB stops SIGSEGV and requests the debugger-attachment breakpoint, then
  captures thread backtraces. GDB exit zero never certifies a test pass.
  Faulthandler enabled; scratch and pytest temp outputs persist in run.
- Core/import and actual linked-library hashes will be recorded inside
  allocation and checked against the preserved failed core SHA256.
  Watcher **bd8e1e8fa**, 300 s intervals, 48-hour watcher deadline.
  No test/scientific change yet; diagnosis-complete marker waits for evidence
  and subsequent confirmed fixes/CPU gates. No full suite or paid retry.

## v3 terminal audit: FAILED, not a GPU correctness pass

- **13992504** scientific pin `8810da5743cc7350e7620ecb708cdd3e47baa739`,
  harness `e60b32507b3069192032705b1d4161dd38886511`; actual prolog/epilog
  confirm authorized **inferno**, H200 on `atl1-1-03-019-14-0`.
  SLURM job/batch FAILED `143:0`, payload CANCELLED `0:15`, elapsed
  **01:57:57**, start/end Oct 13 09:08:24–11:06:21 Eastern.
- Build/configuration and compiler C/C++ probes succeeded. Exact staged import
  and both git pins were checked. Core `1.12a1.dev726` SHA256:
  `3fe5a446db657ecb82c6557973eab85f08c161729ab16d047df7c16fb9d61c73`.
  Actual resolved libraries: libcuest 0.2.2 (conda 0.2.2.2),
  CUDA runtime **13.3.29**, cuBLAS **13.6.0.2**, Libxc 7.0.0.
  Site nvcc was 12.6.1, but CMake detected CUDA toolkit major 13 and import
  ldd resolved the conda CUDA 13 libraries; do not label this CUDA-12 runtime.
  Device NVIDIA H200, driver 615.71.09.
- **51 completed progress outcomes: 50 passes, 1 failure, 0 reported errors
  or skips; 52 tests started, final case unfinished.** There is no finalized
  JUnit: `fdds.xml`, `VERIFIED.json`, `COMPLETED` are all absent.
  Counts are reconstructed only from pytest progress tokens, excluding the
  literal F in printed FDDS lines, by local `audit.py`; no synthetic JUnit.
  The failed completed case is unequal nonhybrid GPU contributions; traceback
  was not emitted before interruption, so the failing assertion/cause is not
  established. Final SAPT GPU case entered Psi4's debugger-wait fault handler
  (PID3340286); its first CPU control returned, but no GPU SAPT result appeared.
  A later TERM stopped the job near its pre-wall-limit signal; accounting says
  FAILED rather than TIMEOUT. No crash stack is available; do not attribute
  this to FDDS or cuEST lifecycle without diagnosis.
- Completed equal nonhybrid and equal/unequal hybrid GPU contribution samples
  print zero CPU/GPU errors. Completed integrated equal/unequal nonhybrid/hybrid
  samples print zero energy differences and actual cuEST FDDS timers.
  Integrated CPU/forced-GPU absolute elapsed seconds: equal nonhybrid
  0.011079/0.012633, equal hybrid 0.016547/0.021600, unequal nonhybrid
  0.010599/0.012759, unequal hybrid 0.016543/0.021534.
  These are tiny correctness fixtures, **not throughput/speedup evidence**.
  After-real-JK HOST pointer mode and full SAPT GPU correctness remain unproven.
- Preserved local evidence:
  `build_fdds_cpu/gpu-13992504/{accounting.txt,accounting-detailed.txt,remote-evidence.tar.gz,audit.py,audit.json,logs/,results-13992504/}`.
  Audit JSON includes hashes of every copied result/log. Source/test/harness
  checkouts are clean and retained unchanged; no retry/new allocation submitted.
  Atomic `fdds-validation-13992504-terminal.json` records this failed outcome.
  Existing pre-review13991729 and coordinator13991128 remain untouched.

## Authorized v3 GPU submission (pending, not validated)

- Subsequent explicit user approval selected **v3 only** for paid inferno.
  Same job **13992504** was updated to `QOS=inferno` with the unchanged
  `TimeLimit=02:00:00`, verified by scontrol; state was still PENDING.
  No duplicate submission, cancellation, source/payload change or environment
  mutation. Jobs 13991729 and 13991128 remain untouched. Original submission
  below was free embers; final audit must check the actual inferno prolog.
  Remote QoS-change evidence: run-root `inferno-switch-accounting.txt`.
  Atomic submission marker updated with effective and initial QoS; watcher
  bba95ab46 continues on the same job.
- Final focused read-only review reported no new correctness defect. Exactly
  one new allocation was submitted: **13992504**, `gts-cs207-chemx`,
  `gpu-h200`, explicit free `embers`, one H200/8 CPUs/64 GiB/two hours.
  SLURM accepted it as PENDING; no GPU execution evidence yet.
- Immutable scientific source/tests:
  `8810da5743cc7350e7620ecb708cdd3e47baa739`, isolated checkout
  `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-validation-20261012-v3`,
  own `build_fdds_validation`. Required **52 executed tests, zero skips,
  failures or errors** (41 native + 11 GPU), plus VERIFIED.json/COMPLETED.
- Separately git-pinned harness:
  `e60b32507b3069192032705b1d4161dd38886511`, checkout ending
  `psi4.fdds-validation-20261012-v3-harness`. This fixes only job activation:
  installed conda compiler hooks read unset HOST/CONDA_BACKUP_GXX under nounset.
  Nounset is relaxed only for conda activation, then restored; GCC_ROOT is
  unset after module load and activation. No shared environment was changed.
  The scientific checkout remains exactly at 8810da5743.
- Live submission headroom **37/50**; unique job-name/comment/run duplicate
  checks passed. Dry-run accepted gpu-h200; absolute cc1/cc1plus frontends
  resolved in p4cuest_sapt. Compile/link/run probes repeat inside allocation.
- Run directory:
  `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261012-validation-03`.
  Audit `results-13992504/{commit.txt,harness-commit.txt,compiler.txt,import.json,fdds.xml,fdds.log,VERIFIED.json,COMPLETED}`
  and configure/build logs after terminal notification, including actual QOS
  prolog, staged import/core/library hashes, executed GPU route/error/timings.
- Automatic watcher **bba95ab46**, standard watch_jobs.sh at 300 s, 48-hour
  watcher deadline. Submission marker `fdds-v3-submission.json` is atomic.
  No GPU numerical/speedup claim. Existing FDDS **13991729** and coordinator
  **13991128**, their frozen checkouts/builds/dependencies, remain untouched.

## CPU review fixes complete (not GPU validation)

- Fix commits: `c90fc65bcc` (checked one/five empty-output budgets and recovery),
  `efde54e03b` (exported helper null/tensor/shape/overflow/budget preflight),
  `8810da5743` (unequal nonempty monomers and checked cuBLAS HOST pointer mode).
  The DFHelper shape-query change is metadata-only: persisted/disk tensors are
  queried through the storage registry, even after clearing transformations.
- Latest CPU-tested source/tests:
  `8810da5743cc7350e7620ecb708cdd3e47baa739`.
  Own Release CPU stage rebuilt; **51 passed, 0 failures/errors/skips, 31 GPU
  cases deliberately deselected, 18.48 s** (41 FDDS/SAPT + 10 native CPSCF).
  Same outside-source-root command as below, with `--junitxml=full-cpu.xml`.
  Evidence: `build_fdds_cpu/review-fixes/{full-cpu.xml,full-cpu.log,full-provenance.txt,full-build.log}`.
  Version `1.12a1.dev726`; asserted staged import path unchanged. New core SHA256:
  `5c199ad6f2e92c489ba13ddaa334a6215d400eee397ecfa530781907ce657069`.
- Maximum independent native contraction error (both monomers, equal/unequal):
  **7.105427357601002e-15**. Unequal A=(2 occ,8 vir), B=(3 occ,7 vir).
  Native integrated FDDS samples: equal nonhybrid/hybrid 0.012004/0.038873 s;
  unequal 0.011409/0.020241 s. No SCF in those timings; no speedup conclusion.
  Physical SVWN He-dimer dispersion/total energies remain exactly as below.
- Review regressions: budget red 12 failed then green 12 passed (2.10 s);
  unguarded helper probe entered Psi4's debugger-wait fault handler and timed
  out, guarded helper/oracle subset 14 passed (2.10 s); unequal subset 4 passed
  (1.65 s). Earlier logs/XML/provenance are retained unchanged.
- HOST pointer mode is checked without mutating handle state; actual after-JK
  execution and new CUDA compilation remain **unverified** in this CPU-only build.
  Capacity-driven reduction/fallback and post-enqueue fault injection still
  lack runtime coverage. No user-facing fault-injection option was added.
- Existing GPU job **13991729** remains **PENDING (Priority)**, no allocation
  assigned at the latest status check, frozen at pre-review pin
  `099706616a818041f56072c3b2471d6c04451cc8`; its 20-test suite will **not**
  validate these review fixes. Job/checkout/dependencies are untouched.
  At CPU handoff, prospective v3/run-03 templates required 52 executed tests
  (41 native + 11 GPU) and were not yet submitted. The subsequent authorized
  isolated v3 submission is recorded above; no existing job was cancelled.
- `fdds-review-fixes-complete.json` is generated atomically only after final
  handoff commit and the green CPU gate; it certifies CPU review-fix completion
  only, not GPU correctness, performance, or merge approval.

## Branch and implementation

- Branch: `saptdft_cuest_disp`, based on response commit
  `4f47db835ad50ee4ed2b7b3005c99cb5cd10da35`. No merge.
- Implementation: `9fd3a2aa620ffe7bf96522dafde18b0999000d40`.
- Allocation compiler isolation: `738aa3efc9` (module GCC_ROOT unset, explicit
  conda activation, absolute cc1/cc1plus checks, C/C++ compile/link/run probes).
- Exception/partial-empty safety fixes: `5f63ef56c9`.
- Test-accessor fix: `be4d7bba6e` (`Matrix.nrow()` is not a Python API;
  use `Matrix.np.shape[0]`).
- Pre-review CPU-tested source/tests and existing GPU pin:
  `099706616a818041f56072c3b2471d6c04451cc8`. Also reshapes DFHelper's preserved
  `(occ,vir,Q)` NumPy metadata to `(ov,Q)` for the independent reference.
- New seam: `CuestFDDSContractions` in `psi4/src/cuest_gemm.{h,cc}`;
  used by `FDDS_Dispersion::form_unc_amplitude` and `form_aux_matrices`.
  Row-major `(ov,Q)` blocks, shared cuEST cuBLAS handle/stream, FP64.
  One/five output sums remain device-resident across all occupied blocks.
  Hybrid inputs upload once per block; only final sums download.
  Stream synchronization protects host scratch reuse. RAII cleans device
  allocations on errors; CUDA/cuBLAS errors propagate without CPU retries.
  Read-only safety review confirmed row-major mappings and capacity accounting;
  follow-up fixes drain queued transfers before exception rethrow and handle
  one-empty/one-nonempty monomers individually (including hybrid setup).
- Opt-in `USE_CUEST`, existing `CUEST_GEMM_MIN_DIM` threshold (default 256).
  CPU-only/disabled/small/empty/unsupported-layout paths remain native.
  Device preflight can shrink blocks or choose CPU before allocation.
  FDDS host memory budgets are not raised; new sizes use checked arithmetic.
- Actual-route evidence: `gpu_contraction_blocks()` plus timer
  `cuEST FDDS: contractions`. Timer includes host preparation and transfers,
  not just kernel time.
- Unchanged CPU work: integral/DFHelper preparation, density projection,
  ALDA kernel, hybrid QR and X/Y construction, Python coupled susceptibility
  products/pseudoinverses, quadrature and exchange-dispersion.
  No new MGGA/VV10/GPU-XC response support. Native HF/LDA/GGA response requires
  `CUEST_XC=false` before SCF.

## Preserved pre-review evidence (GPU pending, not a GPU correctness pass)

- Static checks passed: `git diff --check`, `bash -n` for validation scripts,
  Python syntax compilation of `tests/pytests/test_cuest_fdds.py`.
- Own local CPU build: `build_fdds_cpu`, dependency environment
  `/home/awallace43/miniconda3/envs/p4_ci`, Release, Ninja, `ENABLE_cuEST=OFF`,
  `ENABLE_Einsums=OFF`, `BUILD_SHARED_LIBS=ON`, 12 build workers.
  Logs: `build_fdds_cpu/logs/fdds-cpu-{configure,build,rebuild*}.log` (untracked).
  Configure/build passed. Final command (from `build_fdds_cpu/fdds-tests-v3`):
  `python -m pytest <root>/tests/pytests/test_cuest_fdds.py <root>/tests/pytests/test_cuest_cphf.py -m 'not cuest' -q -s --junitxml=cpu.xml`.
  **23 passed, 0 failed/errors/skipped, 27 GPU tests deliberately deselected,
  17.36 s**: thirteen FDDS/SAPT cases plus ten native CPSCF cases.
  This is CPU compatibility evidence, not a skipped-GPU correctness pass.
  Earlier runs exposed two test-only mistakes (accessor and NumPy tensor
  shape): first 13 passed/2 failed (15.00 s), second 18 passed/2 failed
  (15.17 s); both fixed. Narrow contraction rerun: two passed (1.48 s).
- CPU numerical errors against independent NumPy contractions (absolute):
  nonhybrid amp `7.1054e-15`; hybrid amp `1.7764e-15`, K1LD `8.8818e-16`,
  K2LD `4.4409e-16`, K2L `1.7764e-15`, K21L `4.4409e-16`.
  Fixed-cache integrated FDDS elapsed (no SCF): 0.011980 s nonhybrid,
  0.021344 s hybrid. These tiny random-orbital integral fixtures are not
  representative throughput benchmarks.
  Full physical SVWN/cc-pVDZ He-dimer SAPT case (test time 2.061 s):
  `SAPT DISP ENERGY = -1.0907819605792185e-6 Eh`,
  `SAPT TOTAL ENERGY = -3.513103638665722e-6 Eh`.
- CPU import asserted at
  `build_fdds_cpu/stage/lib/psi4/__init__.py`, core
  `core.cpython-313-x86_64-linux-gnu.so`, version `1.12a1.dev721`.
  Core SHA256:
  `21d4115d948e7e35ad727ffbac1dc7ad5c51b023e80d0e2e9d6c1ba9fbc1b038`.
  Compiler: conda-forge GCC 14.3.0-18. Core links p4_ci's `libmkl_rt.so.2`,
  `libint2.so` (CMake 2.13.1), `libxc.so.15` (CMake 7.0.0), no cuEST/CUDA.
  Detailed artifacts: `build_fdds_cpu/fdds-tests-v3/{cpu.xml,cpu.log,provenance.txt}`.
- Local GPU validation is **not** attempted: known NVML driver/library mismatch.
- Own Phoenix checkout:
  `/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-validation-20261012-v2`
  for the corrected retry; v1 checkout/pin is retained unchanged.
  Build: `build_fdds_validation`; dependency environment `p4cuest_sapt`
  (read-only), cuEST conda 0.2.2.2, site CUDA 12.6.1, conda GCC 14.3.
  Existing lifecycle stage supplies only read-only installed Einsums.
- Frozen v2 validation payload: `devtools/benchmarks/fdds_validate.sh`, batch
  script `fdds_validate.sbatch` at pin 099706616a (current branch templates
  prepare a separate v3 instead). One H200, 8 CPUs, 64 GiB, two hours, explicitly free
  `embers`, account `gts-cs207-chemx`.
  Retry run root: `/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261012-validation-02`.
  Initial job **13991223**, pin `5f63ef56c9bc8740f0f98107a42298b513bbff43`,
  was queued on free embers/gpu-h200 with Priority. User explicitly approved
  cancellation/replacement to avoid spending an allocation on the known test
  typo. It was cancelled without changing its frozen checkout.
  Corrected CPU gate passed. Retry **13991729** accepted on
  `gts-cs207-chemx / gpu-h200 / embers`, requested two-hour wall time.
  Frozen source pin: `099706616a818041f56072c3b2471d6c04451cc8`.
  At submission: **PENDING (Priority)**; no GPU build/tests executed yet.
  Dry-run prediction was 2026-10-29T19:47:44, not a reliable start ETA.
  Automatic watcher **b8a0442d4** runs the standard `watch_jobs.sh -i 300`
  for this job only (48-hour watcher deadline). If still queued at deadline,
  monitoring must be reattached; do not treat watcher timeout as job failure.
- Suite expectation: **20 executed tests, zero failures/errors/skips** in an
  H200 allocation; `VERIFIED.json` and `COMPLETED` required.
  Thirteen native tests cover formulas, missing keys, symmetric/asymmetric
  empty inputs, host-memory guards/recovery, full integrated FDDS and SAPT;
  seven GPU tests cover both monomers/all five hybrid contributions, 2+1
  occupied blocks, CPU/disabled/huge-threshold/forced controls, error preflight,
  default-quadrature integrated FDDS and full SVWN He-dimer SAPT(DFT).
  Matrix tolerance 2e-11, fixed-cache energy 1e-10 Eh, full SAPT energy
  target 1e-5 Eh; GPU errors/times still pending.
- Integrated fixture timing excludes SCF and reports object/kernel/integration
  timers separately. Small deterministic fixtures are correctness checks,
  not representative throughput benchmarks. **No speedup claim.**
- Coordinator's response checkout/build/job and historical SCF packages were
  not modified, cancelled, rebuilt or retested.
  This includes v2 job 13980976 and v3 job 13991128/pin
  `ae8cedca770e1bc1a7a6091d313cfc6dec8887d9`.

## Next completion steps

On the automatic watcher terminal notification, inspect the job's bounded
logs and `results-13991729/{compiler.txt,import.json,fdds.xml,fdds.log,VERIFIED.json,COMPLETED}`.
Record GPU executed counts, errors, absolute times and source/core/library
provenance here. A skipped GPU suite is a validation failure. Device-capacity
fallback and post-enqueue CUDA fault injection are not runtime-tested yet;
their preflight/lifetime handling has source-review evidence only.
