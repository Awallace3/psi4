# Grouped H200 follow-up: failed qualification, preserved data

## Attributable runs

Helper8321063109dce76c9896d9ee73e10abc85c1e13a; immutable whole science/stage
d471cd93fa/build14377089. Frozen input manifest remainsf8c96058e938…
(see`COMPLETION_PLAN.md` for the complete hash).

Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-grouped-8321063109`.

- **GPU14400888 FAILED1:0**,2h11m33s; no TERM.
- **CPU14400901 and14400904 FAILED1:0**,4seconds each; no TERM, before science.
- All are inferno on their declared GPU/CPU partitions. No paid retry has been
  submitted. Peptide's earlier independently checked three pairs are unchanged.
- There are **zero accepted new large-system timing/speedup samples**.

## Nanotube: two executed calls, not repeat-qualified

`gpu/job-14400888/nanotube/r0-gpu` and`r1-gpu` both completed their full GPU
call and cleanup, returning`EXECUTED_REQUIRES_INDEPENDENT_COMPARISON`.
Diagnostic energy-call times2474.143562s and2476.374571s are **not production
speedup measurements**. Energies are−0.0032600607023847844 and
−0.003260060224943802Eh, difference4.77440983e−10Eh. Actual retained ranks
are740/740 for742AOs, with virtual732/662, as independently anticipated.
No large same-gauge CPU oracle has executed.

The original repeat comparator first stopped at raw`Locc_A` coefficients:
maximum difference4.69681025e−9 against unchanged1e−9 tolerance.
**That first small failure is not the whole issue.**

Read-only replay of the saved, SHA-verified arrays collected every original
numerical check, explicitly labeled`DIAGNOSTIC_ONLY_NOT_REQUALIFICATION`
(task`bebe75844`). Original guard verdict remainsFAIL:

| Quantity after candidate orbital sign/permutation matching | Difference |
|---|---:|
| A raw localized coefficients |4.69681e−9 |
| A physical S-metric orbital difference |6.78119e−11 |
| A populations /same-frame rotation |7.228e−12 /6.615e−11 |
| B matched overlap magnitude error |0.37416 |
| B off-permutation overlap |0.48979 |
| B physical S-metric orbital difference |0.86505 |
| B populations /same-frame rotation |0.31960 /0.46664 |
| Full orbital-resolved Elst matrix |0.41434Eh |

Other orbital-resolved energy matrices also fail. **These cell-wise differences
are not errors of0.414Eh in the total interaction energy**; the localized
orbital partitions are not equivalent under the allowed labels/signs. Close
canonical energies alone cannot validate the full decomposition.

Both device localizers report convergence:9sweeps forA and22forB in each run.
B's frozen Q4 sums are23.340784786138414 /23.34078478613503;
active Q4 sums6.10604127257924 /6.1060412728631. Thus very similar
localization objectives coexist with different B-orbital partitions.
Degenerate/alternative localized solutions or canonical-start sensitivity is
a hypothesis, **not an independently validated diagnosis**. No native CPU
stationarity/full-matrix oracle exists yet for these large calls.
Do not loosen tolerances or rotate energy entries to manufacture repeat success.

## Protein83: controlled memory admission failure

`gpu/job-14400888/protein83/r0-gpu/result.json` records
`MemoryError: Explicit CUDA live-buffer estimate exceeded`.
It occurs creating the RIFIT sibling DF plan's **creation-temporary workspace**,
before the full fragment continuation's numerical BLAS/kernels. Upstream
native HF/canonical work did execute; the full F-SAPT call did not complete.
This is not SIGSEGV or an observed cgroup OOM kill.

Actual GPU: H200UUID`GPU-49116497-5f1d-d130-c01b-e2e19509e8d6`,
143771MiB, driver615.71.09, compute90. Session limit120GiB inclusive of a
12GiB opaque-native reservation. The failed attempt reached tracked explicit
allocations92,448,450,560bytes; this is **not total/peak VRAM**.

Synchronized stage observations:

- Before fragment factory, upstream JK still live: NVMLused23163MiB/free120038MiB.
- After JKFIT plan construction: tracked66,217,481,728bytes; native queried
  persistent32,209,721,600bytes plus retained creation temporary34,007,760,128bytes.
  CUDAfree59,411,988,480bytes; NVMLfree56660MiB.
- RIFIT construction then reaches the admission failure. The driver has not
  proved a safe full-system simultaneous workspace/resident-factor budget.

Native/session cleanup completed with live0bytes,0native owners; owned
temporary scratch was removed. No finished protein energy/matrices or timings
exist. Candidate work is to verify SDK creation-temporary lifetime guarantees
and reduce simultaneous plan/stage residency, **not blindly raise/remove guards**.

## CPU launch failure and focused correction

Both CPU stderr logs show:

```
mkdir: cannot create directory '/scratch/JOBID/disp0-split-owned': Permission denied
```

The payload incorrectly assumed that the site's top-level job scratch parent
is user-writable. The corrected payload creates a private subdirectory beneath
the scheduler-provided writable`TMPDIR` (or`SLURM_TMPDIR` fallback), checks the
base exists/is writable, canonicalizes only its own new path, and retains
node-local filesystem/free-space/ownership/cleanup admission. No shared/GPU
fallback is introduced.

Regression`test_payload_scratch.py` runs the real payload with only Git/OS
permission/job-exec boundaries stubbed: it reproduced the exact unowned-parent
failure in0.012s before the fix, then passed in0.018s. The full89-test helper
suite and shell syntax pass (`bbfe103f6`). This is a **local regression**, not
proof of a new successful Phoenix CPU allocation; no retry was submitted.

Old results/configurations/source checkouts remain intact. Fixing this launcher
alone would still not qualify the failed GPU cases, so rerunning the same paid
batch unchanged is not appropriate.

## Bounded source investigations — not implemented fixes

Read-only investigations`d8de680b17e3b55522d186ccb80ec7271` (memory) and
`d7046aa471086e52815eb362403929a23` (localization) found these seams:

- `sapt_proc.py` currently finalizes upstream JK after GPU fragments; cached
  raw JK matrices are cloned earlier. Any earlier finalize must preserve
  `cleanup_jk=False`, external ownership/aliases and exactly-once cleanup.
  This improves actual free VRAM but does not reduce Python tracked bytes.
- `sapt_fisapt_workflow_cuda.py` creates RIFIT before localization/JKFIT terms.
  Phase serialization can preserve one device IBO gauge while avoiding
  unnecessary plan overlap, but native construction cannot simply be moved
  inside an allocation scope. Carry buffers need explicit device ownership;
  no downloaded localization may be reused as production input.
- Plan creation temporaries are deliberately retained and existing mocks
  enforce that lifetime. Early retirement requires the exact admitted SDK
  borrowing/completion contract and synchronized-success/failure tests.
  Do not free everything labeled`temporary`: reusable compute scratch shares
  that accounting flag. Peak live/native/request/limit/free values should be
  captured at rejection, because the current exception combines two predicates.
- Ordered IBO sweeps initialize directly from each SCF's canonical gauge.
  SCF phase fixing does not orient degenerate occupied subspaces. Equal final
  Q4 objectives and reported convergence do not establish a unique optimum or
  independent final-state stationarity. Initial-gauge covariance, final CPU
  stationarity/curvature and IAO spectral-cutoff invariants are the discriminating
  tests. A projected-IAO deterministic initial basis is a candidate scientific
  policy change, not an accepted fix.

These inspections executed no native jobs and changed no numerical algorithm.
They do not justify increasing memory budgets, relaxing comparisons, rotating
energy matrices, mixing source pins in repeats or launching another paid batch.
