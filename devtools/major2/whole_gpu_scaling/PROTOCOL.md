# Whole-energy GPU scaling — frozen observational protocol

This is a NEW cohort, not the failed primary whole A/B protocol and not the
repeated-initialize microbenchmark. It will collect 24 fresh processes on one
H200: three full coupled SAPT(DFT)-D4(I) energy calls per build/system. No new
scientific build, CPU run, smoke/warmup energy, sampler, density input, orbitals,
JK monkeypatch, matrix observer, or lifecycle adjustment is permitted.

`config.py` freezes the exact deterministic adjacent pair sequence inherited
from `policy.sequence()` (labels have a new prefix), all four suite geometries,
the unchanged pair/gates, six-hour embers route and admission budgets.
The inherited original compiled UUID helper and `arm.sh --psiapi` establish
the selected staged build and actual H200 UUID; each child checks its own PID
holds that UUID exclusively before and after its one endpoint.

## Timing and verbosity source delta

`provenance.verify_runner()` proves `runner.py` is an exact owned copy of the
old SHA256-pinned successful whole runner plus only:

* symmetric normal `PRINT=3`, `SCF__PRINT=3`, `SAPT__PRINT=3` before options;
* a record of the effective global/SCF/SAPT print values before energy.

The original `SCIENCE`, final-grid HF iteration proof adapter, coupled-solver
timing wrappers, geometry/basis preparation, `energy()` timing window,
backend/residual checks, finalization and output behavior remain byte-identical.
No timing of individual JK calls is added. Synchronous CUDA status checks
before/after the runner are outside the primary timing window. The primary
clock remains immediately around the actual single `psi4.energy()` call.

Normal SCF print reaches HF then `initialize_jk.set_print`; the native JK
factory reads changed PRINT for new post-SCF SAPT builders. Shared objects
retain their verbosity. SAD hardcodes print0/1 internally; no unsupported
option/binding is invented to override this.

## Individual validity versus cross-record findings

Every record must pass the unchanged `science.validate_record` checks:
backend/options/input/build identity, finite component/total consistency,
all five converged SCFs, terminal final-grid chronology (at least two rows),
both coupled solvers and residuals below 1e-8, full native timer/printed-tree
reconciliation and all required output files. Caps, signal, import, runtime,
nonfinite, nonconvergence or missing output remain fatal, preserving receipts.
The only adaptation of that validator's returned metadata is replacing its
old “plan counts unavailable at unchanged PRINT” note with explicit coverage.

`science.compare` and inherited `policy.GATES` are NEVER modified. In this new
cohort, finite component differences above 1e-8 or differing SCF/solver counts
do not discard observations or stop collection. After all 24 individually valid
records, report each of the nine cross-arm repeat combinations per system and
all within-arm combinations, every component delta, largest component, total
delta, five SCF iteration counts and both solver counts. Unmatched cases carry
“numerical/iteration matched-work acceptance unresolved”. They are not treated
as accepted matched-work cache speedups. Even matched cases are combined-build
observations; weighted RHF finalize and other regions also differ.

Report actual NBF (250,548,922,1786), raw three walls/min/max/median, absolute
seconds saved, percent reduction, median BASE/CAND ratio and three balanced
adjacent pair ratios. Descriptive n=3 only; no CI or significance.
Native initialize and finalize path/call counts and actual per-run cuts stay
separate, preserving nested RHF subphase provenance; never sum inclusive
ancestors or independently medianed component vectors.

## Trace evidence and limitations

`trace.py` counts only anchored native `created #N` / `reused #N` messages for
CAND. BASE has no native explicit hit messages; count its DF-plan memory
banners. Require candidate banner/event agreement. Per-object counters can
repeat and must never be promoted to global IDs. Preserve exact raw lines,
line numbers, hashes, whole log hash and last printed stage/banner context.
Ordinary driver passed-JK messages are separate, not cache hit events.
Native initialize timer calls do not enumerate every compute-triggered rebuild.
SAD plans are not fully observable; silent same-object compute reuse is not
an explicit initialize cache hit. Zero explicit hits is a valid outcome.
`trace-fixtures.json` is line-level evidence from completed job14179309, NOT
new whole-energy counts or performance. All workload/statistics tests use
synthetic records and are not scientific results.

## Resource/custody boundaries

One `gpu-h200`, constraint H200, GRES gpu:h200:1, eight host threads, 128G
scheduler / 100GiB Psi4, six-hour explicitly free embers allocation.
Per-call caps180/300/600/1500s; total15480s. Prolog2400s, validation24×60s,
analysis1200s, teardown300s:20820s within21600s. Prolog, each record and final
verification have fatal deadline alarms. Admission reserves all remaining
caps; no partial cohort or dropped record is a valid final report.
Budget inputs are prior published whole GPU scaling medians8.387831/17.027823/
41.793996/199.087910s, not failed-primary A/B timing values.

Bounded login preparation only stats old roots and hashes small metadata;
full source/stage/ELF/config/ABI/SASS/driver/library/compiler/header checks
are deferred to the allocation and reuse exact inherited implementations.
CUDA_ROOT12.6 is an allowed site locator; CUDA_HOME/PATH13 and pinned
headers/compiler/ELF/runtime remain authoritative. No new dependency/build.

Preserve all17 actual old roots (13 FAILED,4 COMPLETED including14179309),
separately inherited unsubmitted stages, scientific builds, finalizers and
their receipts. New run is pristine, readonly generated bundle only. Code
moves only by Git. `custody.audit` rechecks preservation, accounting, queue
headroom and no own active H200 holder and issues only `sbatch --test-only`.

`parent-finalize.sh` MUST remain UNEXECUTED during preparation. A parent review
sets the exact manifest SHA approval. Permanent local and remote attempt
directories prevent retries after any ambiguity. The accepted numeric job ID
is fsynced before any controller query. The TERM-safe supervisor preserves
partials, requires the exact 24-call completion schema plus report, and writes
exclusive fsynced immutable exit/terminal markers. A zero process exit without
endpoint completion cannot create a success marker.

Run `bash devtools/major2/whole_gpu_scaling/run-tests.sh` locally with its
prepared absolute cwd/env. All295 inherited host checks run unchanged before
the new synthetic/fixture/custody tests. Do not publish HTML or infer new GPU
performance from these preparation checks.
