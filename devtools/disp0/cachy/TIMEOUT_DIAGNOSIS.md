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
