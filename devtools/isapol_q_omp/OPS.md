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
- Pins OLD/build-record.json to the parent-reviewed **4356 bytes**, SHA256
  `896949860e3eb012e07b2ec45556ab4580d87e36de094ca5567d800149803380`,
  before cloning/configuring and checks it again after build and at finalization.
  Replays its exact configure arguments, replacing only source,
  object-dir and private-prefix/Python paths. Original Release/OpenMP/x86-64-v3/
  MKL/Libint2/Libxc/gau2grid/Python stack flags remain, including the intentional
  `PYMOD_INSTALL_LIBDIR=/python3.13/site-packages` suffix. Exact argv matching
  rejects duplicate overrides, `-C`/preload/toolchain injection and changed
  write-bearing paths; no additional configure flags are appended.
  Original insist-find dependency flags and disabled pip index remain.
  No dependency-package installation/download step or new stack is introduced;
  missing original dependencies require stopping, not fetching replacements.
- Clears DESTDIR, CMAKE_* (including CMAKE_INSTALL_MODE/toolchain overrides),
  GCC_ROOT, site includes/compiler/make flags, shell-startup/preload/Python contamination.
  Direct clone Python, explicit private/BASE PATH and LD_LIBRARY_PATH.
- Normal CMake install: `BASE/bin/cmake --build CAND/build --target install -j2`.
  Two heavy Libint TUs can each use~4.3GiB;32GiB/-j2 is conservative, not a
  promise of measured peakRSS/runtime.
- Original non-science dependency bytes checked in clone before/after install.
  Candidate installed Python compared with committed source, **except** the
  two CMake templates `__init__.py` and `extras.py`: these compare strictly with
  `CAND/build/psi4-core-prefix/src/psi4-core-build/{__init__,extras}.py`,
  the known configured outputs, not raw source templates. Missing configured
  outputs fail; actual isapol source byte checks remain strict. Imported package/
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

TERM/INT handlers remain active **through finalization**, not ignored or
inherited as SIG_IGN. Success finalization runs as a tracked background child;
an interrupt is forwarded and forces exit143/failure finalization. If success
was already published while the finalizer was still alive, its record is
retained as `COMPLETED_INTERRUPTED` (or `COMPLETED_INVALIDATED` for another
failure), not left as `COMPLETED`; `FAILED` is monotonic/idempotent. The short
non-success cleanup is not killed again, but further signals are still detected
and keep the wrapper nonzero.

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

Previous7eb local result: **13 ops metadata tests +5 pure workspace tests passed**, both
batch files passed `bash -n`. The five reviewed5ae files were hash-checked
unchanged. No32-case native test, configure, clone of a real science environment,
build or cluster operation has run locally.

The new deterministic regression signals both wrappers **during FINALIZATION**,
before and after success publication, with TERM and INT (eight scenarios).
It also runs the original b0 wrapper bytes from Git in temporary test copies and
reproduces their ignored-TERM/COMPLETED0 bug. No old checkout or receipt is edited.

The ops-only build fix adds exact approved-argv/raw-record rejection tests,
inherited install/toolchain/site-environment poisoning tests, and configured-
template versus strict isapol source tests. Original oracle002's independently
verified32/32 pass on job15279995 belongs only to its immutable7eb source.
Every new ops commit requires a **fresh same-source original oracle** before
building; oracle002 is not adopted or rewritten. Parent must verify actual
allocation, batch and step exit0 as well as JUnit/markers. Candidate native tests,
build, scientific validation and performance measurement remain pending.

Build-fix local verification: **17 ops metadata +5 pure workspace tests passed**,
both batch files passed `bash -n`, frozen5ae file diff and `git diff --check`
passed. A stdlib-only read of the confirmed transferred record passed the raw
pin and exact argv gate (51 original and replay arguments; only indices
2/4/8/35/39/43 changed). No local native execution or build.
