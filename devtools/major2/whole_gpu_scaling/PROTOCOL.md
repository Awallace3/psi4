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

Preserve all18 actual old roots (14 FAILED,4 COMPLETED including14179309),
separately inherited unsubmitted stages, scientific builds, finalizers and
their receipts. The rejected `major2-whole-gpu-scaling-20261019` root remains
unchanged unsubmitted staging, separately snapshotted and small-metadata
hashed alongside the two inherited unsubmitted stages. It is NOT a jobroot.
Failed14182594 at `major2-whole-gpu-scaling-20261019-v2` remains FAILED1:0,
including its exit-zero first-child COMPLETE custody and all saved output hashes.
Its original source stays pinned at6ae8; no old finalizer or lock is reused.
The fresh replacement run is `major2-whole-gpu-scaling-20261019-v3`, from
`psi4.major2-whole-gpu-print3-repair-20261019` with a dedicated scratch parent.
Only this new namespace has no actual attempt/job ID during preparation.
New run is pristine, readonly generated bundle only. Code
moves only by Git. `custody.audit` rechecks preservation, accounting, queue
headroom and campaign-specific duplicate admission and issues only
`sbatch --test-only`.

### Campaign admission is not allocation exclusivity

Only this custody module binds `admission.queue`; the inherited
`native_plan/handoff.py` global protocol is unchanged. A fresh, user-filtered,
expanded-task `squeue` snapshot includes owner/name/command/workdir/route/state.
Both effective login name and numeric UID must agree with every row. Explicit
header-only output is valid empty evidence; missing/truncated/malformed output,
unknown metadata, foreign owners, command failure, stderr, timeout, duplicates
and exhausted embers headroom are fatal. Queries have 35s/2MiB/4096-row bounds.
There is no cluster-wide dump, unbounded controller fan-out, wait or scheduler write.

Literal `Command=(null)` is not an available command and is never invented.
Only otherwise valid own, nonduplicate PENDING/RUNNING rows on a known single
CPU partition can request targeted `scontrol show job -o JOBID` evidence.
The whole snapshot, duplicates, headroom and maximum eight candidates are
validated before querying. Each controller read has a 5-second/64KiB cap
(at most 40 seconds additional), exact job/owner/name/WD/partition/state/
account/QoS/Command agreement, CPU/memory/node ReqTRES and running AllocTRES.
Only bounded CPU TRES are recognized; GPU/unknown GRES/TresPer requests fail.
Missing fields are recorded honestly; BatchFlag=1 is valid and is NOT called
interactive. The raw bytes/hash and selected controller fields accompany
`command_available=false`, `unrelatedCPUproven-no-GPU`. Without proof, parsing
still rejects null commands. Query failure/disappearance/ambiguity stops for
a new audited snapshot; no retry/wait or claim of device custody is permitted.

Duplicate identity is the unique `major2-whole-gpu-scaling` stem in JobName,
Command or WorkDir (including dated/versioned/prior-partial source/run
namespaces), or the owned `/devtools/major2/whole_gpu_scaling` helper path.
This deliberately catches direct helper commands from differently named
checkouts too. Generic `major2`, partition, GPU type, account or QoS alone do
not identify this campaign. Every queued state is checked, not just RUNNING.

Unrelated pending/running H200 jobs are permitted, including XDM14179965/66
in `/runs/xdm-gpu/20261019`. Their inferno route is neither ours nor permission
to charge inferno; they are never controlled, attached to or used as holders.
The receipt preserves their metadata as `unrelated`, with fresh per-user
embers counts and explicitly `allocation_exclusivity_proven=false`.
SLURM must grant our separate six-hour embers allocation. Only the unchanged
in-allocation route/environment, compiled CUDA logical0→physical UUID,
before-child empty-holder, and actual own-PID exclusive-UUID checks establish
device custody. Queue admission NEVER establishes GPU exclusivity.

`parent-finalize.sh` MUST remain UNEXECUTED during preparation. A parent review
sets the exact manifest SHA approval. Permanent local and remote attempt
directories prevent retries after any ambiguity. The accepted numeric job ID
is fsynced before any controller query. The TERM-safe supervisor preserves
partials, requires the exact 24-call completion schema plus report, and writes
exclusive fsynced immutable exit/terminal markers. A zero process exit without
endpoint completion cannot create a success marker.

### Interpreter exit and completed native artifacts

The scientific runner, clock and minimal proof adapters are unchanged.
`execute.py` retains early environment/import, actual PID and allocated UUID
exclusivity, PRINT3/one-call endpoint return, CUDA status and runtime/maps
proof **before interpreter exit**. It writes `ENDPOINT-PRE-EXIT.json`,
explicitly labeled pre-exit endpoint evidence, not completed native artifact
verification. There are no pre-exit native file hashes or child COMPLETE.
Ordinary Psi4 atexit owns native timer/profile finalization: an early
`timer.dat` may exist but is incomplete; the profile may not yet exist.
No explicit core.finalize, atexit removal, double finalize or observer is added.

Only after `run_command` has waited for exit zero, the parent `complete_case`
requires the actual PID/unsignalled exit receipt and matching endpoint
PID/UUID/job/manifest evidence. It checks directory/file paths (no symlinks),
required completed files, ordinary single-link files and output/record caps.
It seals `result.json`, `psi4.out`, `timer.dat`, `profile.txt` readonly,
fsyncs, SHA256-hashes their final post-exit bytes, checks stable identity/bytes
and fsyncs the output directory. The parent's exclusive readonly COMPLETE
receipt binds the actual child exit, exact endpoint evidence+SHA and all four
post-exit hashes. Only then does unchanged individual science validation run.
Nonzero/signal/timeout/missing output cannot complete; partials remain.

`test_exit_lifecycle.py` launches genuine fresh Python subprocesses through
the production child wrapper and parent wait/custody path, replacing only the
scientific endpoint/device dependencies with stdlib fixtures. Actual atexit
appends an early timer and creates the profile for the first time. Tests prove
final byte hashes and fsync ordering, no child completion, readonly outputs,
nonzero after endpoint (even with atexit files), actual signal, missing file,
wrong PID, caps and file/directory symlinks. These are real temporary files,
not a mocked filesystem; they provide no native scientific/GPU evidence.

Run `bash devtools/major2/whole_gpu_scaling/run-tests.sh` locally with its
prepared absolute cwd/env. All295 inherited host checks run unchanged before
the whole/admission/lifecycle host checks. The prior362 tests remain unchanged;
no old fixture expectations or timing/source/numerical criteria are weakened.
Do not publish HTML or infer new GPU
performance from these preparation checks.
