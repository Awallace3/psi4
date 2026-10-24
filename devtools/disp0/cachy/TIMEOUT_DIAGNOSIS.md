# Cachy diagnosis of the Phoenix fragment timeout

## Scope and evidence before reproduction

User directed debugging onto Cachy rather than another H200 allocation.
No Phoenix changes/submissions, FDDS inspection, existing install overlay,
identity-check bypass, or desktop/game process changes are authorized.

Phoenix gate14322613/source1aed320cdb ended with `TimeoutExpired` at600s,
not a recorded SIGSEGV. It printed completed native HF/canonical induction
then fragment basis loading; no completed GPU fragment or cleanup result exists.

Read-only Cachy inspection (2026-10-24) found:

- Common `/home/amwalla3/.cache/psi4-cachy/gpu0.lock`; no `gpu0.HOLD` at that
  snapshot. No process for the exact persisted Disp0 owner session and no
  Disp0 tmux pane were found. These observations do not acquire a lease.
- `source` remains clean at a7204feec6. Separate `factory-source` remains
  clean at f735f5c194. Neither is edited by this diagnosis.
- Coherent installed `install/psi4-cuest-10` is scientific source9cc6784336,
  authenticated by `runs/psi4-cuest-import-13/receipt.json`. This predates
  Phoenix1aed's blocked-response changes; it is a representative common-path
  reproducer, not byte-identical H200 execution.
- Its earlier cc-pVDZ frozen/scaled water-water/block5 whole GPU diagnostic
  completed in364.552s; its independent CPU oracle07 passed. Dispersion
  reported81compute blocks/90DFMO batches and72.523s inside blocked execution,
  of which provider calls reported72.175s. These include audit overhead, not
  integral kernel-only time.
- Earlier scalar-driver cProfile `runs/automatic-cuest-profile-01` recorded
  111.363s total:60.456s exclusive HASH.update and14.624s exclusive file reads.
  This is measured historical Cachy overhead, not a measured explanation
  of the later H200 failure.

Hypothesis: repeated strict whole-DSO hashing in `audit_maps`/provider identity
checks dominates tiny blocks. Alternative locations (native DFMO, localization,
response, startup) must be separated by actual phase/stack/profile evidence.
No hash cache or weakened admission is proposed by the diagnostic.

## Bounded reproducer

`trace_fragment_workflow.py` authenticates and imports the untouched whole
Cachy10 stage through its existing `metadata/stage05_native.py` loader.
It uses the H200 water geometry, aug-cc-pVDZ/aug-cc-pVDZ-JKFIT/aug-cc-pVDZ-RI,
full exact GPU fragment mode and block5. No CPU factors/localizer/FISAPT
fallback is allowed. All runtime patches are observing wrappers only.

Record JSONL phase entry/exit timings, periodic15s Python/native-call-site
stack snapshots and cProfile data. The internal deadline is180s, with an
outer210s SIGINT timeout and30s kill grace. Stop if common HOLD appears.
Any intentionally stopped attempt remains stopped/failed, not qualification.

Launch only with nonblocking common whole-process flock, live absent-HOLD
checks, a fresh output directory and an explicit bounded resource scope:
4CPU,12GiB host RAM; Psi4 budget2GiB, E2 budget1GB, existing native explicit
device budget1GiB plus512MiB declared persistent. Native internal VRAM is not
claimed equal to that explicit budget. No heavy build is required.
Keep the lock through interpreter/native cleanup; never wait-spin on a busy
GPU or alter another owner's HOLD.

Phase totals are inclusive/nested: do not add `audit_maps` to its child
`mapped_file_digest`, or either to enclosing DFMO time. This is diagnostic
timing with instrumentation, not an accepted CPU/GPU speedup measurement.

## Reproduced result: audit-bound, not a recorded segfault

First launch `b135e88da` stopped in shell preflight on an unset PYTHONPATH
expansion, before the traced Python process or its attempt directory existed.
The corrected fresh run was `timeout-trace-aug-b5-02`, task `bcbd1eeae`.
Its user scope read back CPUQuota4s/second, MemoryMax12884901888, no swap,
TasksMax128. Common nonblocking whole-process flock and absent HOLD were checked.

The actual traced process PID2285739 stopped on its180s Python diagnostic
deadline, with final elapsed181.067s including recorded unwind:

| Observed region | Calls | Inclusive seconds |
|---|---:|---:|
| `audit_maps` | 166 | 146.029 |
| `mapped_file_digest` | 1494 | 128.910 |
| `fisapt_cuda` | 1 | 165.762 |

These rows are nested, not additive. cProfile separately attributes104.081s
exclusive to HASH.update and32.979s to buffered file reads.
`fragment_terms` was not entered until178.202s; the deadline fired inside
`CuESTPointPotential._bind → api.assert_identity → audit_maps → mapped_file_digest`.
Repeated15s stacks show the same hashing path during per-shell native plan
construction. No numerical fragment kernel stall or SIGSEGV was observed.
The native plan's cleanup report is complete with no retained/ambiguous
resources; no completed full-fragment/session-qualification result is claimed.

Library identity/provider/blocked-dispersion source files are byte-identical
between Cachy9cc6784336 and Phoenix1aed320cdb (Git comparison). This establishes
a shared, measured overhead problem, not a complete causal trace of H20014322613.
Block size alone cannot remove the already-large factory-construction cost.

Raw evidence: Cachy run `runs/timeout-trace-aug-b5-02/`
(`diagnostic-result.json`, `phases.jsonl`, `stacks.txt`, `profile.pstats`).
Selected evidence was collected by task `b0cb38ca4`.

Candidate narrow optimization: bounded concurrent fresh CUDA file hashes,
retaining serial canonical cuEST admission before symbol binding and retaining
all same-FD/inode/device/byte/race checks. No hash cache, verification-frequency
reduction, scientific change, installed overlay or solved-H200 claim.

## Candidate CPU evidence

The serial implementation failed the new concurrency regression in2.015s:
`Fresh CUDA byte proofs must run concurrently` (task `b318a3b83`).
After the bounded-worker change,63cuEST and205CUDA CPU/mock/serial guards
passed (`bdad5a2b7`), including fresh hashes on both repeated audits, maximum
four overlapping workers, native-first admission, unlisted-path rejection,
CUDA inode/device/replacement/open errors, deterministic error ordering,
executor submission failure and worker `KeyboardInterrupt`/shutdown.
Independent inspect-only review `d7799c3c05ac12596261fff9226d6eb5c` supported
this constrained design; it is not a device or H200 validation result.

The helper's separate `--audit-only --candidate-source <Git worktree>` mode
compares old/new byte-audit receipts against the same loaded native world
without replacing any installed Psi4 module. It does not request an energy,
integral or kernel workload. Whole-stage installation and full numerical
requalification are still required before using the changed guard in production.

First real-byte paired audit `timeout-audit-paired-01` passed with six identical
receipts, no installed overlay (tasks `bd705dcd3`, `b6179328b`). Recorded medians
were serial0.879755s and concurrent0.499718s. Because cProfile was attached only
to the main thread, a follow-up explicitly disables profiling for BOTH timed
implementations before interpreting the latency improvement.

The **unprofiled follow-up passed**: `timeout-audit-paired-02`, helper4e3ce70c1d,
tasks `b39fe0ee4` / `b11c22f0f`. Same common nonblocking lease and4CPU/12GiB
scope; no installed module was changed. Six byte-admission receipts are identical.

| Fresh byte audit | n | Median seconds | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Original serial | 3 | 0.824477 | 0.818215 | 0.828119 |
| Bounded concurrent | 3 | 0.457401 | 0.438444 | 0.458649 |

The median **audit-only latency ratio is1.803**, not a GPU energy speedup.
Timed sections exclude cProfile on both arms; initialization remains outside
the individual audit timers. Original/patched mapped-byte checks returned the
same proofs with no cache. Candidate identity-module SHA256:
`7c1be48c8329fc80580531335da97a0033ff82a6f8bdc32b247133ca8ac8df28`.

The candidate scientific guard change is commit16cd0d5f18. It still requires
a new coherent whole-stage install and complete fragment/numerical/cleanup
rerun before claiming this fixes the full timeout or requesting another H200
gate. Do not overlay it onto Cachy10 or the qualified Phoenix stage.
The bounded diagnostic processes/leases have ended; no Phoenix allocation,
FDDS source/environment/process change or desktop/game modification occurred.

## Coherent guard-fixed build and first gate attempt

At the user's conditional request (“if Cachy good, inferno Phoenix”), the normal
whole build/install of source `dda47ee73d0d21cef62e137a22b4e5ae84979e65`
passed its CPU native smoke (task `b5f05efd4`). The previous whole build was
verified and preserved read-only before advancing only the owned
`factory-source`; `source`, old installs and the dependency environment remain
unchanged. External limits read back 4 CPU/12 GiB/no swap/128 tasks; build
parallelism was one. New artifacts:

- `install/psi4-cuest-11`
- `runs/psi4-cuest-10-snapshot-01`
- `runs/psi4-cuest-11-config`
- `runs/psi4-cuest-build-11`
- `runs/psi4-cuest-import-14`

This is **CPU-build validation only**, not GPU qualification.

The first shared-gate attempt, `runs/guard-fixed-aug-water-gate-11-01`
(task `bc616d830`), stopped in the CPU arm's bootstrap before its energy call or
any GPU child. The new helper assumed all qualified dependency modules lived
under the conda prefix. The actual whole CMake installation contains QCElemental
under the new Psi4 prefix, which the CPU proof correctly inventories. Bootstrap
therefore rejected `qcelemental.datum` despite its authenticated provenance.

The correction binds **each package's actual CPU-qualified root**, restricted
to the selected whole stage/private dependency installations, and continues to
check every loaded module against the exact stored byte inventory. A regression
first reproduces the erroneous rejection with stage-installed QCElemental, then
also checks that changing its bytes is still rejected. This is a helper-only
fix: no numerical options/tolerances, installed science/core or dependencies
are changed. Preserve attempt01 and use a fresh helper pin/output directory.

The new complete augmented-water GPU/CPU matrix gate must still pass before
the condition authorizing a new bounded H200/inferno correctness gate is met.
Neither the audit-only ratio nor CPU build completion meets that condition.

### Attempt02: full unscaled GPU execution, oracle still blocked

Helper `9746fe1b72` passed45 admission/launch regressions locally and on Cachy.
The fresh attempt `runs/guard-fixed-aug-water-gate-11-02` completed its CPU
unscaled energy call in3.287490s and its GPU unscaled call in354.701242s. Both
children returned0 and saved `EXECUTED_REQUIRES_INDEPENDENT_COMPARISON`; these
are diagnostic elapsed times, not accepted benchmark samples or a speedup.
The GPU child passed its required native/device-work/output/cleanup checks.
No timeout or segfault occurred in that child.

The independent same-gauge CPU oracle then failed the existing
`rtol=1e-10, atol=1e-10` one-electron `V_A` comparison:8 of6724 entries differed,
including an absolute difference2.360213447676e-10. The complete gate remains
FAILED; scaled/displaced/repeat cases were not run and Phoenix is not released.
Do not discard the failure or enlarge the oracle tolerance to obtain a pass.

Source inspection shows native `MintsHelper` creates its cuEST AO pair list
with the current global `INTS_TOLERANCE` (default1e-12), including for nuclear
attraction. A tiny separately bounded parameter probe compares unchanged saved
GPU inputs and independent CPU Mints matrices at1e-12/1e-14/1e-16. It preserves
the oracle threshold and records every difference/violation. Screening is a
hypothesis until that physical probe runs; no production accuracy change or
full qualification follows from the probe alone.

The physical probe completed in9.542s with complete session cleanup,
zero live explicit bytes/native owners, and the actual leased RTX3090 UUID
(`runs/one-electron-screening-11-01/probe/probe-result.json`,
tasks `bd3e7dac2` / `b5e0b3f2e`). At the original cutoff it reproduces the saved
GPU potentials to9.24e-14 and7.76e-14, respectively, including the original
oracle failures. Only the input screening cutoff changes:

| AO pair cutoff | Max \|GPU−CPU\|, V_A | Max \|GPU−CPU\|, V_B | Oracle violations A/B |
|---|---:|---:|---:|
| 1e-12 | 2.36021345e-10 | 1.92093415e-10 | 8 / 2 |
| 1e-14 | 7.78703400e-12 | 6.33767434e-12 | 0 / 0 |
| 1e-16 | 1.42108547e-13 | 8.93729535e-14 | 0 / 0 |

The diagnostic gate now requests `INTS_TOLERANCE=1e-16` in **both** CPU and GPU
arms, with unchanged `1e-10` one-electron comparison and unchanged downstream
matrix thresholds. A regression asserts that both arms use this tighter input
accuracy and the original2.36e-10 error is still rejected. This is an explicit
diagnostic input change, not permission to reuse prior timings/inputs or relabel
attempt02. The whole installed scientific package/core remains unchanged.
Full unscaled/scaled/displaced/repeat and independent matrix qualification must
be rerun in a fresh attempt before Phoenix release.

### Attempt03: every physical child/oracle passed; raw-label repeat rejected

`runs/guard-fixed-aug-water-gate-11-03`, helper02fa343786, completed all10 fresh
processes with return0. All three independent native CPU same-gauge oracles
passed all23 required fields, unchanged thresholds and standard file checks:

| Input | Diagnostic GPU energy-call seconds | Largest oracle energy-matrix absolute error (Eh) |
|---|---:|---:|
| Unscaled | 351.004129 | 5.35571587e-13 |
| Scaled | 354.038818 | 6.91276203e-12 |
| B shifted0.08Å, scaled | 354.585258 | 5.60063801e-12 |
| Fresh scaled GPU repeat | 356.121806 | Not a fourth CPU-oracle case |

The parent still records **FAILED**: its final naive elementwise repeat check
compared orbital labels directly. Actual occupied-overlap analysis shows
monomer A's labels unchanged and monomer B's equivalent localized orbitals1/2
swapped (`[0,2,1,3,4]`), with positive signs. Off-permutation overlaps are at
most1.823e-12. Applying only those orbital permutations gives maximum energy
matrix difference1.695e-11Eh, localized-coefficient difference1.146e-12,
population difference4.436e-14 and same-canonical-frame U difference1.823e-12.
The raw failed artifact remains unchanged.

The corrected repeat comparator requires a unique unit-overlap permutation
and orbital signs, forbids continuous localized rotations and frozen/active
mixing, verifies Cocc×U=Locc and canonical-frame/energy consistency, and compares
all23 original fields at the unchanged1e-9 repeat threshold. Energy arrays
receive **only row/column permutations**, never orbital signs or dense U
rotations; atomic/external positions remain fixed. Both signed active dispersion
pair arrays are checked separately. Tests reject real field errors, improper
energy rotations/signs, nondegenerate canonical mixing and frozen/active swaps.

A fresh complete gate, rather than relabeling attempt03, must establish final
PASS under this label-invariant comparator. The production CPU benchmark also
explicitly excludes the diagnostic gate's tightened screening setting, retaining
its original native default; published CPU production timings are not replaced
or presented as matching this exact-response GPU diagnostic.

Independent mathematical inspection (`d703197b0fce38aae1a17853dc180e563`)
confirmed the mapping direction and absence of energy-rotation/sign scaling.
Its concrete safeguards are incorporated before another physical gate:
normalize only each child's verified own F-SAPT output-directory option;
require distinct resolved repeat directories/PIDs; capture dimensions from
native wavefunctions and require the fixed six-atom/82-AO/five-occupied,
one-frozen/four-active-per-monomer water contract. Pair-array dimensions cannot
redefine the frozen space. Wrapper tests now cover these boundaries and artifact
tampering, in addition to unequal A/B dimensions, a non-self-inverse cycle,
nonidentity AO metric/rectangular coefficients and unchanged numerical checks.
Inspection/tests alone do not release Phoenix.
