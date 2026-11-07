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
  three generated files `__init__.py`, `extras.py`, `metadata.py`: these compare strictly with
  `CAND/build/psi4-core-prefix/src/psi4-core-build/{__init__,extras,metadata}.py`,
  the known generated outputs, not raw source templates/stub. Missing generated
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

## Separate read-only diagnostic of existing build001

**Do not rerun build/install merely to correct the metadata validator.**
Job15280334 remains historically FAILED1:0 at the post-install metadata check.
`metadata.py` is generated by `update_version`/`versioner.py` (psi4/CMakeLists.txt
303–312), not installed from the raw fallback stub. Parent verified generated,
staged and installed metadata all have SHA256
`31a3fec8334714b53fed8b21fb8e9c15c65620f58d7b8b253c523cbe2a07e6a8`.

New `validate.py`/`validate.sbatch` perform **no configure/build/install/clone**.
They read the existing build001, run candidate-native tests, and write only a
new nonce-owned diagnostic root and fresh scratch/CWD/cache. Every file and
namespace entry in the entire old candidate tree (including private install,
objects, logs and FAILED) is hashed before/after; qualified OLD is independently
sealed before/after, even on child failure. Nothing relabels the failed build,
qualifies the candidate or adopts it into production.

Compiled provenance is pinned to SOURCE003 commit
`5f83eb146986cb0e6a7113d9a9374d8ca3e5e610`, tree
`8a8a18d326342d38214a1fde488a95abfe013955`, frozen5ae science/tests.
Validation operations come from a **new independently staged source004 ops
child**, with its own commit/tree. Both sources must remain clean/unchanged.
Tests execute the compiled-source frozen32 cases plus existing native
performance/partitioned-response regressions, no SCF.

### Parent-bound historical evidence required

Parent supplies a new private JSON file plus its verified SHA via
`--evidence`/`--evidence-sha256`; do not write it inside build001 or source.
It must contain the following fields (paths are absolute unless relative is
explicit in the key):

```text
schema: "q-omp-compiled-evidence/1"
candidate_root: existing .../runs/ccasp-native-q-omp-build001
compiled_source:
  path: absolute SOURCE003
  commit: 5f83eb146986cb0e6a7113d9a9374d8ca3e5e610
  tree: 8a8a18d326342d38214a1fde488a95abfe013955
  frozen_patch: 5ae597cd9305588d14681d0da435834ef761c264
historical_job: {id: "15280334", status: "FAILED", exit_code: "1:0"}
command_completion_basis:
  "source_control_flow_after_runner_checks_not_stored_exit_codes"
artifact_sha256: map of relative name -> parent-verified SHA256 for EXACTLY:
  result.json, FAILED, configure-command.json, configure.stdout,
  configure.stderr, build-install-command.json, build-install.stdout,
  build-install.stderr
traceback: {path: absolute retained batch/step error log, sha256: verified SHA}
core:
  env_relative: "lib/python3.13/site-packages/psi4/core.<actual ABI>.so"
  inner_build_relative: actual relative path under
    build/psi4-core-prefix/src/psi4-core-build (e.g. src/core.<actual ABI>.so)
  sha256: identical compiled/staged/installed core digest
```

Four known historical artifact hashes and metadata hash are pinned in code.
The evidence gate verifies exact configure/install argv lists, parent-approved
original build record, pinned failure/traceback location and source bytes.
**Neither historical command JSON stores return codes.** Pinned5f83 reaches the
specific installed metadata exception only after both Runner.run calls return,
each enforcing exit0. Result records zeros as **inferred command exit codes**
with that explicit source-control-flow basis, never stored exit receipts or a
successful historical build. Fresh native test exit0 is independently enforced
by the new Runner and JUnit; no source exception is treated as native success.

Candidate checks compare **all installed Python, including installed tests**,
with explicit CMake source projections or the three generated outputs;
generated/staged/installed files must match.
Compiled/staged/installed core bytes must equal the parent-bound digest.
Outer/inner CMake caches must bind compiled source/prefix/compiler/OpenMP and
x86-64-v3 (XHOST OFF checked in outer cache only). Imported Python implementation
digest must equal the source-only expected digest, and imported core must equal
the bound candidate core. Original dependency hashes remain strict.

### Parent CLI (review and test-only first)

Parent supplies new SOURCE004/COMMIT/TREE, fresh nonexistent VALIDATION/VSCR,
separate fresh existing VLOG, and verified private EVIDENCE/EVIDENCE_SHA.

```bash
/opt/slurm/current/bin/sbatch --test-only \
  --chdir="$VLOG" --output="$VLOG/validate-%j.out" --error="$VLOG/validate-%j.err" \
  "$SOURCE004/devtools/isapol_q_omp/validate.sbatch" \
  --source "$SOURCE004" --commit "$COMMIT" --tree "$TREE" \
  --root "$VALIDATION" --scratch-root "$VSCR" \
  --compiled-source /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-q-omp003 \
  --candidate-root /storage/project/r-cs207-0/awallace43/runs/ccasp-native-q-omp-build001 \
  --evidence "$EVIDENCE" --evidence-sha256 "$EVIDENCE_SHA"
```

Only parent replaces `--test-only` after approval. Fixed8CPU/32GiB/2h free
embers/cpu-small/gts-cs207-chemx, no requeue/retry, same TERM-safe finalization
as7eb wrappers. Hashing overhead/native duration unmeasured;2h is a finite limit.
New output status is `READY_DIAGNOSTIC_NOT_ACCEPTANCE`, `acceptance=false`,
`adoption=false`; wrapper COMPLETED means only this diagnostic succeeded.
Parent must verify actual new allocation/batch/.0 exit0 and all JUnit cases.
Old FAILED/result/install are not rewritten. Candidate profiling remains later.

Additional pure command (stdlib fake files/children only):

```bash
python3 -I -B devtools/isapol_q_omp/test_validate.py
```

Local final verification: **17 ops +9 diagnostic +5 workspace pure tests PASS**
(taskbff1d9214). All three wrappers pass `bash -n`, Python AST syntax checks
pass, `git diff --check` passes, frozen5ae files and both existing5f83 wrappers
have zero diff. Profiler835 remains clean. Diagnostic tests use only fake
files/stdlib children, including native-child/import failure, candidate mutation,
pre-import CWD protection and actual finalization TERM/INT scenarios.
No real candidate native test, rebuild, cluster or quantum action ran locally.

## Source005 narrow installed-test mapping correction

Validation001 job15281287 failed before native import/tests at
`tests/isapol_factorial_oracle.py`; it is not a kernel-test failure.
Both that diagnostic and build001 remain FAILED with original receipts intact.
No rebuild/install or scientific/test/tolerance change is required.

The mapper follows **all** Python install classes in psi4/CMakeLists.txt:

| Package-relative path | Expected source | CMake lines |
|---|---|---|
| `driver/**.py` | `SOURCE/psi4/driver/**.py` (pytest directories excluded) | 330–335 |
| `header.py` | `SOURCE/psi4/header.py` | 338–339 |
| `__init__.py`, `extras.py` | known inner-build configured outputs | 341–345 |
| `metadata.py` | known inner-build versioner output | 303–312 |
| `tests/{pytest-file}` | `SOURCE/tests/pytests/{pytest-file}` | 356–373 |
| `tests/data_isapol/**`, four writer/QCSchema directories | matching `SOURCE/tests/pytests/**` | 374–385 |
| `tests/{sample}/test_input.py`, `input.py`, selected Psithon helpers | `SOURCE/tests/{sample}/...` | 387–413 |

Trailing slash on `../tests/pytests/` flattens that root into package/tests.
The subsequent raw-tests install keeps sample subdirectories. An unexpected
ambiguity between selected source projections fails closed—never choose by
precedence or whichever source hash happens to match. Raw pytests/CFour
exclusions are retained.

Every installed `.py` is inventoried independently of OLD's manifest and
must map to an existing contained source/generated file and match its bytes.
Missing/tampered tests fail, including isapol helpers. Unknown classes fail;
there is no generic missing-source skip or installed-tests exemption.
Native tests still execute unchanged compiled SOURCE003 test bytes.

Local source/CMake projection audit: **1000 paths**,3 generated,169 runtime
source,180 pytest-source,648 raw sample-test-source; zero missing sources.
Parent's full actual installed audit b9c2fd6a8 independently reports1000 files
with the same3/169/180/648 split: **zero unknown, zero hash differences, zero
ambiguous source paths**. This is a byte/mapping audit, not native execution.
The actual isapol factorial helper matches source SHA256
`0be372d66094dac68707b87fb0cabc1e511c23d491a442261adae3b21974981c`.
Sample `tests/aediis-1/test_input.py` maps to SOURCE/tests/aediis-1/test_input.py,
SHA256 `37c7f7b3d644cf241b0eceb8290ce44d9b0384e5b358de88c6676c0dff3643dc`.

New SOURCE005 must use new validation002 output/scratch/log roots, **same**
parent-bound evidence SHA
`44582da052154970c01f309d0f623299896559f1a1ce58dad1d8b42ba7b1e2f9`,
existing CAND build001/core676ea1..., and immutable compiled SOURCE0035f83.
Do not edit SOURCE004, validation001, build001, OLD,835 or frozen5ae tests.
No benchmark implementation/work is authorized by this mapping correction.
