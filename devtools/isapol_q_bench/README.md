# Existing benchmark rebound to the fresh successful low-thread build

Only four reused5b benchmark files are added to frozen ops8573. SCI731,
ops8573, old benchmark5b, profiler835, all prefixes/receipts remain unchanged.
No build, recovery workflow, framework, production adoption or pin override.
Local tests use stdlib/fake files/processes only; parent stages/submits.

## Closed normal-build contract

- Compiled source is fixed:
  `/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-q-lowthreads001`,
  commit8573ded3233001b50d99c26bf9a577e13b68c956,
  treeea812747f4b2c7fbb0967b1c591f0b07b5c26bad, reviewed SCI731fb3d52f...
- Candidate root is fixed:
  `/storage/project/r-cs207-0/awallace43/runs/ccasp-native-q-lowthreads-build001`.
  Result is exactly root/result.json, job15284185. No CLI source/prefix/result-
  path/identity override, recovery mode or skip-policy option.
- **Required `--build-result-sha256` is an explicit parent-checked64-lowercase-
  hex literal**, with no default/auto-discovery. Parent supplied real digest:
  `ab8a01fef0bb715bf31e084ae7f9c56251b9cae1471f3d1b4389a795ebd041cf`.
  Parent must independently verify accounting before supplying it.
- Digest check precedes reading identities. Requires normal
  READY_BUILD_NOT_QUALIFIED/modebuild/acceptancefalse, correct source/root/job,
  original manifest/configure pins, fresh34-oracle digest2af43012... and
  old_runtime_unchanged; no exception/integrity error.
- Requires build COMPLETED0/no FAILED; actual JUnit exactly34Q+18 performance+
  32 partition response=84, no failures/errors/skips/unknown modules, matching
  result JUnit SHA. Old32/82, failed and recovery receipts fail closed.
- Before/after stdout parsed using the existing **QOMP_IDENTITY= line protocol**,
  not raw json.loads(stdout). Both records must equal result.identity.
- Candidate identity is read only from the digest-bound result's direct
  implementation pair—**no hardcoded new identity**. Actual compiled/staged/
  installed core hashes, package/prefix paths and source/installed Python
  implementation digest must match it.
- Every entrypoint also checks qualified OLD/dependencies, approved original
  4356-byte configure record/normal command argv, complete installed Python
  mapping and installed .py/.so hash inventory. New configured __init__/extras/
  metadata outputs and staged bytes are checked; all flat/native and sample
  tests mapped strictly. Missing/unknown/ambiguous/tampered data fails.
- Master seals the entire existing candidate tree before/after; OLD/source/
  dependency checks repeat. No existing source/prefix/result/marker is written.

`receipt_gate(copy_directory, digest)` is a stdlib copied-data sanity helper:
it checks report/JUnit/markers/recorded identities, **not installed real files**.
It cannot authorize computation alone. Execution always uses `build_gate` on
the fixed candidate root plus all real-file/source checks above.

## Unchanged measurements

1. **Synthetic Q constructor-call proxy**, OLD/CAND x1/2/4/8, same8CPU affinity.
   N65,543,3 rank4 sites, AUX246, Q75x246, seed4103/site seeds41/97/173.
   The frozen builder is unchanged:246 spherical all-s shells on1 AUX centre,
   not real water's mixed shells/grid/ISA shapes. Label remains
   SYNTHETIC_ALL_S_WIDTH_MATCHED_NOT_MOLECULAR_Q.
   One untimed warm-up+5 timed calls/cell; no dropped replicates.
   **Constructor-call wall time includes pybind/STL argument copies/marshalling**
   plus native collocation/harmonics/screens/accumulation—not pure accumulation.
   Explicit Python setup/export outside timer. No full-grid AUX matrix/cache.
   All repeated values/diagnostics/offsets and all eight saved Q arrays require
   exact bits matching OLD1; complete small inputs/basis recipe/Q are retained.
2. **Full water**,3 fresh eight-thread runs/arm, same interleaving
   OLD,CAND,CAND,OLD,OLD,CAND, no state/orbital reuse. PBE0/aug-cc-pVTZ,
   partition400/2030 vs response100/200, H2H1, Drho32, thresholds/caps/coverage/
   protocols unchanged,20GiB science. OMP/MKL requestedN, OpenBLAS1.

Original fixture and835 profiler paths are fixed to existing profile001 and
native-profile-fixtures001/monomer-0001, with exact835 commit/tree and original
fixture hash checks. Their main/qualification/production pins are never patched.
OLD uses exact accepted claim; CAND uses a NEW deep copy with exactly two leaf
replacements at keywords.function_kwargs.native_monomer.implementation_sha256.
The actual verified pair supplies these values; no accepted fixture edit.
Normal installed admission still enforces implementation identity.

Warm checks precede measurements. Each child has a fresh owned CWD/cache/scratch
before any Psi4 import and requires -I/-B/-X owned-cache. Actual installed paths/
version/pair, CPU/affinity/pools/RSS are recorded. Parent code uses frozen835
telemetry/hooks read-only; no new affinity/NUMA/backend knob.

All independent outputs retained: argv/stdout/stderr/process times, inputs/exact
derivation, events/exclusive spans, full QCSchema/envelope/NPZ/Cn, proxy inputs/Q,
per-child and independent controller scratch sampling and candidate snapshots.
Added integrity work is outside constructor/schema timing but included in
process wall time; don't compare process times across different harnesses.
Scratch256GiB sampled-byte ceiling, no inode reservation. Samples miss possible
transients; byte/namespace/link-target snapshots are not a filesystem sandbox.

## Parent CLI — review/preflight first

Stage new benchmark source through Git; its commit/tree are in the handoff.
ROOT/SCRATCH must be nonexistent, absolute and disjoint from all old inputs/
source/prefix/receipts; LOG is a separate fresh existing directory.

```bash
BUILD_SHA=ab8a01fef0bb715bf31e084ae7f9c56251b9cae1471f3d1b4389a795ebd041cf
/opt/slurm/current/bin/sbatch --test-only \
  --chdir="$LOG" --output="$LOG/bench-%j.out" --error="$LOG/bench-%j.err" \
  "$SOURCE/devtools/isapol_q_bench/benchmark.sbatch" \
  --source "$SOURCE" --commit "$COMMIT" --tree "$TREE" \
  --root "$ROOT" --scratch-root "$SCRATCH" --build-result-sha256 "$BUILD_SHA"
```

Only parent replaces --test-only after approval. Fixed1node/1task/8CPU/32GiB/
2h free embers/cpu-small/gts-cs207-chemx, no requeue/retry/build/recovery.
2h is a finite cap, not a runtime promise. GIT_OPTIONAL_LOCKS=0; no new
OMP_PROC_BIND/OMP_PLACES/NUMA policy. Reviewed TERM/INT behavior through
finalization retained; no false COMPLETED. Benchmark acceptance/adoption false.

Parent checks actual benchmark allocation/batch/.0 completion and all artifacts,
then analyzes all constructor replicates and three full runs/arm, exclusive
stages/RSS/scratch, every tensor and27 C6/C8/partial-C10 entries. Full-recompute
variability is not the fixed-input bitwise oracle. No scientific tolerance
relaxation, solver certification or speedup claim before measurement.

## Pure checks

```bash
python3 -I -B devtools/isapol_q_bench/test_bench.py
python3 -I -B devtools/isapol_q_omp/test_ops.py
python3 -I -B devtools/isapol_q_omp/test_workspace.py
bash -n devtools/isapol_q_bench/benchmark.sbatch
```

Fake-file tests exercise real hashes, three generated outputs, flat/native/
sample test mapping and dependency checks; reject stale/failure receipts,
wrong source/root/job/identity, missing/tampered files and unknown mapping.
Exact derivation/matrix/order, failure post-seal, pre-import CWD and
TERM/INT-before/after-publication regressions retained. No native imports/calls.
