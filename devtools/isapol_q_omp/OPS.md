# Low-thread SCI731: original oracle, then one fresh normal build

Only five reused operations files are added to the independently reviewed SCI
commit **731fb3d52f3c301b28b22fae469e0deb3f7e805c**:
`ops.py`, `oracle.sbatch`, `build.sbatch`, `test_ops.py`, this document.
They derive from corrected a1e ops, not the original5ae-era operations.
The four SCI731 files and unchanged Python planner are hash-bound in FROZEN.
No SCI/test/planner edits, benchmark adaptation or recovery workflow is included.
No local native/quantum/build, network, cluster, push, merge or deployment.

## Deliberate rebinds, unchanged protections

- Source gate requires clean exact CLI commit/tree/root, ancestry of SCI731,
  all five frozen file hashes and **only these five ops paths** differing from
  SCI731. No validate/recovery/benchmark file is allowed.
- Original oracle runs **34 Q cases only** on original qualified1aaa.
  Candidate gate requires **exact34 Q +18 native-performance +32 partition-
  response =84 cases**, no failures/errors/skips/unknown modules.
- Both jobs bind the same new ops source path/commit/tree plus frozen_patch731.
  Older source/oracle32 receipts are rejected, not adopted.
- Reviewed a1e clone/configure/environment/mapping/ownership/TERM mechanisms
  retained. Wrappers export GIT_OPTIONAL_LOCKS=0 before helper/Git calls;
  no new OMP_PROC_BIND/OMP_PLACES/NUMA or BLAS backend policy.
- OLD fixed:
  `/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-build001`
  with manifest SHA d2b4049c2e5818e462ad3c6ac36f93d6a9623a6a205c701b885887ec2a9f3bfd.
  BASE fixed:
  `/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt`.
  Their manifest-listed runtime/science/dependency bytes checked before/after.
- Original configure record remains **4356 bytes**, SHA256
  `896949860e3eb012e07b2ec45556ab4580d87e36de094ca5567d800149803380`.
  Exact51-argument replay only, source/build/prefix/clone-env substitutions.
  Includes intentional PYMOD_INSTALL_LIBDIR=/python3.13/site-packages suffix.
  Reject extra/duplicate/preload/toolchain/write-bearing overrides; no appended
  flags, changed stack, dependency install/download step or fresh self-approval.
  Missing original dependencies require stopping, not fetching replacements.
- Clears inherited DESTDIR/CMAKE_INSTALL_MODE/CMAKE_*/toolchain/site/compiler/
  preload/Python/make contamination, sets direct prefix/BASE PATH and libraries.
  Offline OLD/env clone; links into OLD/env retargeted in clone, external BASE
  file links materialized as identical private bytes, external directory/.pth
  injection rejected. Original dependencies checked in clone before/after.
- Full installed Python inventory, not just old manifest: exactly three
  generated files __init__/extras/metadata compare with NEW configured inner-
  build outputs. Runtime driver/header map to SOURCE/psi4; flat pytest helpers
  and selected test directories to SOURCE/tests/pytests; raw sample tests to
  SOURCE/tests. Missing/unknown/ambiguous/outside/mismatched files fail closed.
- Fresh root/scratch/CWD before ANY Psi4 child import, including finalizers.
  -I/-B and fresh cache/test/temp destinations; source/install/fixtures/receipts
  never used as working/scratch directories.
- TERM/INT detection remains active through finalization. Forward/wait and
  bounded escalation, monotonic FAILED, superseded COMPLETED preserved under
  interrupted/invalidated name, no false exit0, retry, requeue or chain.

## Parent-only exact command shape

Concrete proposed dedicated paths; parent verifies all are new/disjoint and
outside every old source, prefix, fixture, log/receipt tree and BASE. Parent
stages SOURCE through Git; commit/tree supplied in the handoff.
Only OLOG/BLOG are precreated (fresh separate directories). ORACLE/OSCR/
BUILD/BSCR must be nonexistent at launch.

```bash
SOURCE=/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-q-lowthreads001
ORACLE=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-q-lowthreads-oracle001
OSCR=/storage/scratch1/3/awallace43/ccasp-native-q-lowthreads-oracle001
OLOG=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-q-lowthreads-oracle001-logs
BUILD=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-q-lowthreads-build001
BSCR=/storage/scratch1/3/awallace43/ccasp-native-q-lowthreads-build001
BLOG=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-q-lowthreads-build001-logs
: "${COMMIT:?parent stages reviewed ops commit}"
: "${TREE:?parent verifies staged tree}"
export GIT_OPTIONAL_LOCKS=0

/opt/slurm/current/bin/sbatch --test-only \
  --chdir="$OLOG" --output="$OLOG/oracle-%j.out" --error="$OLOG/oracle-%j.err" \
  "$SOURCE/devtools/isapol_q_omp/oracle.sbatch" \
  --source "$SOURCE" --commit "$COMMIT" --tree "$TREE" \
  --root "$ORACLE" --scratch-root "$OSCR"
```

Oracle finite contract:1node/1task/8CPU/32GiB/2h free embers,
cpu-small/gts-cs207-chemx, no requeue. **No SCF**.
Only parent, after review/test-only/preflight, replaces --test-only with
--parsable. Parent verifies actual allocation/.batch/.0 COMPLETED0 and literal
result digest, all34 JUnit cases/no skips, OLD identity/hash seals, source pins,
helper READY_ORACLE_NOT_ACCEPTANCE, wrapper COMPLETED and no FAILED.
Marker alone is not Slurm accounting or scientific acceptance.

Only after those checks, bind the new immutable result SHA as ORACLE_SHA:

```bash
: "${ORACLE_SHA:?parent verifies fresh same-source34-case oracle result digest}"
/opt/slurm/current/bin/sbatch --test-only \
  --chdir="$BLOG" --output="$BLOG/build-%j.out" --error="$BLOG/build-%j.err" \
  "$SOURCE/devtools/isapol_q_omp/build.sbatch" \
  --source "$SOURCE" --commit "$COMMIT" --tree "$TREE" \
  --root "$BUILD" --scratch-root "$BSCR" \
  --oracle-report "$ORACLE/result.json" --oracle-sha256 "$ORACLE_SHA"
```

One normal fresh build/install:8CPU/32GiB/8h free embers, same account/partition,
no requeue/retry/inferno. `BASE/bin/cmake --build BUILD/build --target install -j2`.
Finite limits are not promised completion/peakRSS estimates. Failure preserves
partial evidence and stops; no historical validate.py recovery or failed-prefix
adoption. No old prefix/source/receipt/qualification claim is overwritten.

## Expected evidence

Each root: owner nonce, actual allocation snapshot, source commit/tree/frozen731,
full command/stdout/stderr logs, imported identity-before/after, junit.xml,
result.json and wrapper COMPLETED/FAILED. Build additionally: clone link changes,
approved configure SHA/replayed command, oracle result SHA, all installed Python
checks and actual installed .py/.so hashes. Commands are argv JSON LISTS, not
stored per-command return-code receipts; helper requires actual command exit0
before proceeding. Parent separately verifies actual job/batch/.0 accounting.

Build success: READY_BUILD_NOT_QUALIFIED, acceptance=false, actual NEW installed
core/Python identity pair, equal identity before/after tests, exact84 cases,
OLD/runtime/dependency seals true. This is native correctness evidence only.
Do not guess/reuse a candidate identity, call this qualification or infer a
speedup. All frozen earlier success/failure records remain earlier evidence.

## Pure verification

```bash
python3 -I -B devtools/isapol_q_omp/test_ops.py
python3 -I -B devtools/isapol_q_omp/test_workspace.py
bash -n devtools/isapol_q_omp/oracle.sbatch
bash -n devtools/isapol_q_omp/build.sbatch
```

Tests use stdlib/fake processes only. New guards bind source ancestry/allowlist,
count34 deliberately from native-test AST without importing it, enforce exact
84 module split and reject wrong source/old32 oracle receipts. Existing mapping,
record/argv/environment/clone/budget/ownership and FINALIZATION TERM/INT
regressions retained. Historical negative-control test reads immutable b0f9
wrapper Git objects; pure source guards also require1aaa/5ae Git objects.
None is a native compile/test/performance substitute.

Local verification: **24 ops +8 workspace pure tests PASS**, taskb1a7e8a83;
both wrappers bash-n and diff-check PASS. SCI731 files have zero diff.
Actual copied approved4356-byte record passed the stdlib gate/replay check:
51 arguments, only indices2/4/8/35/39/43 substituted, no Psi4 import.
Wrapper bodies match a1e exactly apart from GIT_OPTIONAL_LOCKS export.

## Existing benchmark seams — document only, wait for real build

After the new build/accounting/native84 gates pass, a separate reviewed minimal
benchmark rebind can reuse5b's runner structure and frozen835 telemetry:

1. Bind this ops/SCI source commit/tree, successful build root/result digest,
   parent-verified oracle/accounting/JUnit and **actual** installed identity.
2. Replace old failed-build recovery-specific validation_gate/seal dependency
   with the fresh normal READY_BUILD_NOT_QUALIFIED evidence, installed Python/
   dependency/science hashes and OLD/new-prefix pre/post checks. No recovery
   receipt, old676 identity, guessed hash or borrowed qualification.
3. Keep exact two implementation-hash leaf derivation into a NEW benchmark
   copy of accepted input, original fixture inventory and835 main pins intact.
4. Reuse matched1/2/4/8 frozen-operand timing (constructor call INCLUDING pybind
   marshalling), fixed-bit/diagnostic checks and separate full-water repeats.
   No benchmark code, new framework, deployment or acceptance is added here.
