# Oracle first, then private offline build

Only these new ops files are added after frozen Q patch5ae597cd93:
`ops.py`, `oracle.sbatch`, `build.sbatch`, `test_ops.py`, this report.
The five SCI/oracle/planning files reviewed at5ae remain byte-identical; the
835 profiler and campaign operations are untouched.

**No local native execution/build, quantum calculation, network or cluster
action.** Parent stages through Git and submits only after independent review.

## First job: original-qualified32-case oracle

- Existing qualified science:
  `/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-build001`
- Original BASE:
  `/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt`
- Exact frozen file: `tests/pytests/test_isapol_q_parallel.py`.
- Runs **only32 frozen-operand native API cases**, no SCF. Cases internally
  exercise1/2/4/8 against the original serial arithmetic oracle.
- **1node/1task/8CPUs/32GiB/02:00:00**, `cpu-small`,
  `gts-cs207-chemx`, explicit free `embers`, **no requeue**.
- Full original manifest-listed runtime/dependency/scientific-file hashes
  checked before/after; original core/Python identity asserted.
- Fresh owned CWD before all import-only/native-test children, including core
  finalization. Bytecode/cache/output/scratch stay outside source/install.
- Root contains JUnit, complete command/stdout/stderr logs, imported identity,
  allocation, `result.json`, and wrapper-owned `COMPLETED`/`FAILED`.
- **READY_ORACLE_NOT_ACCEPTANCE** is a kernel test result, not solver/scientific
  certification, deployment or a copied qualification claim.

## Exact parent CLI (roots supplied by parent)

`SOURCE` must be the clean Git-staged **ops child commit/tree** reported in the
handoff, containing exactly the frozen5ae science/test bytes. `ORACLE`, `OSCR`,
`CAND`, `BSCR` are fresh nonexistent absolute disjoint roots, never under source,
OLD, BASE or old receipts. LOG must be a separate new existing writable parent
for SLURM output; do not precreate ORACLE/OSCR.

```bash
# Parent supplies SOURCE, COMMIT, TREE, ORACLE, OSCR, LOG.
/opt/slurm/current/bin/sbatch --test-only \
  --chdir="$LOG" --output="$LOG/oracle-%j.out" --error="$LOG/oracle-%j.err" \
  "$SOURCE/devtools/isapol_q_omp/oracle.sbatch" \
  --source "$SOURCE" --commit "$COMMIT" --tree "$TREE" \
  --root "$ORACLE" --scratch-root "$OSCR"
```

Only parent, after review/route checks, may replace `--test-only` with
`--parsable`. No submission, dependency chain, automatic retry or requeue exists.
Check actual allocation and32 passed/no skipped cases plus `COMPLETED` before
considering a build.

## Second job: private clone, original CMake replay, candidate native gates

**Only after oracle passes**, parent records SHA256 of its immutable
`$ORACLE/result.json` as `ORACLE_RECEIPT_SHA`. The build helper refuses a missing/
wrong digest, source identity, old runtime identity or wrapper completion.

```bash
/opt/slurm/current/bin/sbatch --test-only \
  --chdir="$LOG" --output="$LOG/build-%j.out" --error="$LOG/build-%j.err" \
  "$SOURCE/devtools/isapol_q_omp/build.sbatch" \
  --source "$SOURCE" --commit "$COMMIT" --tree "$TREE" \
  --root "$CAND" --scratch-root "$BSCR" \
  --oracle-report "$ORACLE/result.json" --oracle-sha256 "$ORACLE_RECEIPT_SHA"
```

- **1node/1task/8CPUs/32GiB/08:00:00 embers**, same account/partition, no requeue.
  These are requested finite limits, **not measured completion estimates**.
- Copies OLD/env offline into **new CAND/env**, excludes old bytecode. Nothing
  writes OLD/env. Links back into OLD/env are retargeted only in the clone.
  External BASE file links are materialized as identical private bytes to
  prevent install-through writes; unexpected external directory links/`.pth`
  old-env injection fail closed for parent review.
- Replays OLD/build-record.json configure arguments, replacing only source,
  object-dir and private-prefix/Python paths. Original Release/OpenMP/x86-64-v3/
  MKL/Libint2/Libxc/gau2grid/Python stack flags remain. Requires all original
  insist-find flags; FetchContent download/update is disabled and pip index use
  disabled. No dependency-package installation/download or new dependency stack.
- Clears GCC_ROOT, site includes/compiler flags, preload/Python contamination.
  Direct clone Python, explicit private/BASE PATH and LD_LIBRARY_PATH.
- Normal CMake install: `BASE/bin/cmake --build CAND/build --target install -j2`.
  Two heavy Libint TUs can each use~4.3GiB;32GiB/-j2 is conservative, not a
  promise of measured peakRSS/runtime.
- Original non-science dependency bytes checked in clone before/after install.
  Candidate installed Python compared with committed source; imported package/
  core must resolve under CAND/env, new core/Python digests must differ from OLD.
- Candidate32-case oracle plus existing `test_isapol_native_performance.py` and
  `test_isapol_partitioned_response.py`, no SCF. All must pass/no skips.
- Records candidate installed `.py`/`.so` hashes and source commit/tree, original
  configure digest/command, old manifest digest, oracle digest and test identity.
  **READY_BUILD_NOT_QUALIFIED**, never relabels old receipts as candidate proof.

Build time has not been measured. If the finite free-embers job fails/preempts/
times out, retain partial evidence and stop; parent chooses any later new root.
No inferno or automatic recovery.

## Status and TERM behavior

Batch traps forward TERM to active srun; helper forwards to its newly created
child process group, waits and escalates after20s if necessary. Interruption or
nonzero child status cannot produce a successful result. Helper deadlines leave
cleanup margin (6900s oracle,28200s build); SLURM supplies final2h/8h hard caps.

Metadata publishes atomically with no-clobber links. Wrapper alone creates
COMPLETED after zero unsignalled exit and final source/owner/result checks.
An unpredictable per-launch token prevents writing FAILED/COMPLETED into a
preexisting or foreign root. SIGKILL/node loss may leave partial evidence and
no final marker; a marker is not a scientific acceptance certificate.

No QCA, held-job, receipt mutation, paid QoS, source optimization extension or
PFIT change is performed. Full candidate profiling is separate after native
checks, with a derived benchmark-only identity/input—not weakened835 guards.

## Pure tests only

```bash
python3 -I -B devtools/isapol_q_omp/test_ops.py
python3 -I -B devtools/isapol_q_omp/test_workspace.py
bash -n devtools/isapol_q_omp/oracle.sbatch
bash -n devtools/isapol_q_omp/build.sbatch
```

Tests cover frozen bytes, wrappers/resources, paths, atomic no-clobber status,
offline symlink copying, original configure replay, exact32-case/no-skip gates,
parent-bound oracle digest/completion, failure/foreign-root finalization and a
plain stdlib child's TERM handling. They invoke no real Slurm/compiler/Psi4.

Local result: **11 ops metadata tests +5 pure workspace tests passed**, both
batch files passed `bash -n`. The five reviewed5ae files were hash-checked
unchanged. No32-case native test, configure, clone of a real science environment,
build or cluster operation has run locally.
