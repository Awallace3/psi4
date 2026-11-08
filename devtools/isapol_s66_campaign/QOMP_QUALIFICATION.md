# Closed fb11 genuine build / installed-check mode

## 152: exact three-entry BASE metadata compatibility; current ops004

Historical verification15322764 failed BEFORE native import/collection in
`existing_snapshot(BASE)`: strict resolution of the malformed compiler
`TOOLS=addr2line` link found a missing target. This is diagnostic metadata
failure, not science or Q mismatch. Preserve failedcollection003/ops003 and
all compiled002/earlier records unchanged. No automatic retry or BASE repair.

Parent's complete sealed-link census found exactly THREE known missing-target
links, all with raw text matching original protected snapshot99558a98:

|BASE-relative link|Exact raw text|Lexical target / current absent anchor|
|---|---|---|
|libexec/gcc/x86_64-conda-linux-gnu/14.3.0/TOOLS=addr2line|../../../../bin/x86_64-conda-linux-gnu-TOOLS=addr2line|bin/x86_64-conda-linux-gnu-TOOLS=addr2line (terminal)|
|share/cmake/Einsums/tests/cxx23_static_call_operator.cu|../../cmake/tests/cxx23_static_call_operator.cpp|share/cmake/cmake/tests/cxx23_static_call_operator.cpp / share/cmake/cmake|
|share/cmake/Einsums/tests/cxx23_static_call_operator.hip|../../cmake/tests/cxx23_static_call_operator.cpp|share/cmake/cmake/tests/cxx23_static_call_operator.cpp / share/cmake/cmake|

This is a literal three-entry table, not permission to ignore broken links in
the baseline. It applies ONLY when the full snapshot explicitly selects the
actual BASE root, never compiled prefix/OLD/required executables/imports.
Original protected-before raw size18257504/SHA99558a98 is checked before JSON
use. Require exact historical kind/raw row, unchanged actual symlink/raw text,
canonical non-symlink sealed directories for link and anchor parents, exact
lexical target, and specific `lstat` ENOENT at the table's anchor. All other
errors/missing ancestors fail. Creating/repairing an anchor (file/dir/link),
retargeting, permissions, loops, namespace changes or an unknown broken link
fail. Presence of an anchor/target row in original sealed metadata also fails;
missing inventory entries are NOT historical filesystem/content attestations.
Snapshot rows explicitly record CURRENT absence and historical raw metadata
only, with no target hash. Recheck at start, row capture and end, as well as
whole-operation pre/post integrity. Generic valid-link resolved/content
bindings and original raw BASE comparison remain unchanged.

The malformed compiler name is not normal`bin/addr2line`; Einsums is OFF in
the approved configure and collection/checks do not compile packaged CMake
tests. Normal addr2line still must resolve to the sealed regular
`bin/x86_64-conda-linux-gnu-addr2line`, SHA
`c1f9abaad54a61ae6cb621566b4388f5265853ffb4de7b26c004dbe34d66966f`.
No `resolve(strict=False)`, shared-environment modification, native/source/
test change, count discount, cap/policy change or observed-hash approval.

Current closed paths (all others, including TEST003, remain unchanged):

```
OPS=$P/gits/psi4.ccasp-native-s66-qomp-ops004
COLLECT=$P/runs/ccasp-native-s66-qomp-fb11-collection004
CHECK=$P/runs/ccasp-native-s66-qomp-fb11-installed004
# corresponding fresh scratch basenames and sibling -logs
```

After reviewed Git staging, from an already clean approved parent environment:

```bash
/usr/bin/env -i PATH=/usr/bin:/bin USER=awallace43 LOGNAME=awallace43 HOME="$APPROVED_HOME" \
 /usr/bin/python3 -I -S -B - "$OPS" "$OPS_COMMIT" "$OPS_TREE" <<'PY'
import pathlib,sys,types
p=pathlib.Path(sys.argv[1])/'devtools/isapol_s66_campaign/phoenix_installed_checks.py'
ns={'__name__':'_metadata_only','__file__':str(p)}
exec(compile(p.read_bytes(),str(p),'exec'),ns)
engine=ns['qomp_engine'](); engine.qomp_enter()
args=types.SimpleNamespace(ops_commit=sys.argv[2],ops_tree=sys.argv[3])
print(ns['existing_source_gate'](engine,args))
engine.qomp_library()  # verify frozen8573 Git bytes
print(ns['existing_base_metadata_preflight'](engine))
assert 'psi4' not in sys.modules
print('three exact BASE metadata cases + normal addr2line PASS; no native')
PY
```

This hashes ONE pinned historical JSON and ONE normal executable as data;
it stats only the declared links/parents/anchors. No environment traversal,
compiler/executable call, native import, collection, config/receipt write,
network or full-prefix verification. Earlier loader/site execution cannot be
undone by `env -i`; parent still supplies explicitly clean approved exports.
After separate review/approval, use the151 two-mode CLI below with these004
paths and NEW reviewed ops commit/tree, both2h/8CPU/32GiB embers, never003.
Successful collection still needs independently observed literal new hashes
and actual allocation/batch/.0 COMPLETED0 before separate556 qualification.
All acceptance/adoption flags remain false; no actual native pass is claimed.

## Archived151 interface: use152's004 paths, not the003 examples

## 151: separate read-only verification of failed build002 (ops003)

**Not a successful normal build or production adoption.** Job15320675 and its
batch/step FAILED1:0. Configure/install/identity component records returned0;
collection returned2 because pytest reserves the parametrization name
`request`. It collected547 nodes plus an error instead of the fixed556.
The subsequent Runner stop-latch message is not evidence of external TERM.
All historical bytes/statuses remain unchanged. The normal build/check gates
below are unchanged and STILL REFUSE this failed build.

Three explicit source roles:

* compiled source/runtime and installed-Python mapping: exactfb11/tree3a192,
  existing`BUILD002/source` and unchangedscience001;
* executed tests: detached`311d17bf9177a972f3212c048e2f2177408b95d7`,
  tree`57ff961824c0da6d76cf48e4da63fc83979f4f91`, in
  `$P/gits/psi4.ccasp-native-s66-qomp-tests003`. Only five identifier/string
  substitutions in the grid test; whole-AST alpha-rename and nine cases are
  checked without imports. Literal-paramname lint covers the entire fixed
  union before any native import. No corrected file is overlaid/installed.
* operations: new clean child ofd8b in
  `$P/gits/psi4.ccasp-native-s66-qomp-ops003`, explicit commit/tree and the same
  exact five-file allowlist. Neither historical module dispatch nor source
  pins are loosened.

`collect-existing` and `checks-existing` share the sanitized wrapper, pinned
Git/executables, guarded `-I -S -B` imports, verified8573 library, finite Runner
and TERM/finalization behavior. They do NOT clone, configure, compile,
materialize, install or create an environment. Their only outputs are NEW
`$P/runs/ccasp-native-s66-qomp-fb11-{collection003,installed003}` and matching
`/storage/scratch1/3/awallace43/<basename>`, plus parent-owned sibling logs.
The wrapper's default remains8h for the original build mode; parent MUST use
`--time=02:00:00` for either new mode (allocation gate checks it).

Before ANY native import, verify all literal parent-observed historical
artifact hashes, actual0 component records/approved configure/-j2 argv,
FAILED15320675 accounting, anchored CMake caches/install manifests and
absence of ordinary success receipts. Outer manifest has6348 entries and
6344 unique paths: retain raw bytes/hash, explicitly deduplicate only its
inventory view. Inner manifest has4985 unique paths. Full current prefix
files must be attributed to original sealed clone bytes, strict compiled
Python mapping, parent-observed core, or explicit source/configured install
projections. Unexplained files/empty directories/new native dependencies fail.
Literal `.cmake` install origins are read, never executed.

Full BUILD002 snapshot covers prefix, objects, compiled source and EVERY
retained metadata/log/marker, not just a selected receipt list. Link text AND
resolved target/type/content binding are included; escaping files require
verified dependency inventory entries; external directory links are refused.
Internal directory targets are bound through the full namespace. OLD/BASE
seals plus current resolved bindings and all source roles are checked before
and after success/failure/TERM. Attribution gaps STOP, not a fallback or repair.
This is a **new current attributed inventory**, not reconstructed historical
full-seal evidence, a permissions sandbox or proof against transient writes.
Cooperative TERM follows the integrity finally; SIGKILL/node loss can prevent
postchecks. Missing postcheck equality proof is never replaced with a success claim.

Parent must stage/approve, then independently verify actual job/batch/.0
COMPLETED0, zero collection errors/skips, exact ordered556 and literal
`verification.json`, `collection.json`, `result.json` digests before checks.

```bash
# Parent supplies new approved ops Git IDs; no launch is authorized by this doc.
P=/storage/project/r-cs207-0/awallace43
OPS=$P/gits/psi4.ccasp-native-s66-qomp-ops003
# submit with explicitly clean approved scheduler environment (not --export=ALL
# from a contaminated shell), absolute fresh sibling log paths and --chdir=$P:
sbatch --time=02:00:00 --output="$COLLECTION_LOG/%j.out" \
  --error="$COLLECTION_LOG/%j.err" "$OPS/devtools/isapol_native_acceptance/qomp.sbatch" \
  collect-existing "$OPS" "$OPS_COMMIT" "$OPS_TREE"
# ONLY after parent accounting/artifact review and literal hashes:
sbatch --time=02:00:00 --output="$CHECK_LOG/%j.out" --error="$CHECK_LOG/%j.err" \
  "$OPS/devtools/isapol_native_acceptance/qomp.sbatch" \
  checks-existing "$OPS" "$OPS_COMMIT" "$OPS_TREE" \
  "$VERIFICATION_SHA" "$COLLECTION_SHA" "$COLLECTION_RESULT_SHA"
```

The collection receipt schema is`existing-failed-build-install-verification/1`;
qualification uses`existing-install-qualification/1`. Neither is a normal
READY/fresh-native-build receipt or a measured production profile. All556
tests run from corrected311d17 with ONLY hash-verified compiledfb11 installed
helpers; the bad installed grid module is checked as data and not collected.
Captured identity must equal parent-observed coref4a1/Python331b/versiondev212.
Later profile/producer adoption must explicitly recognize this historical
failure + new verification + new qualification chain, outside this phase.
Two-hour resource budgets are not runtime promises.

## Historical normal build/check interface (unchanged)

Operations only, child of scientificfb11. No measured profile, descriptor/
worker/cohort dispatch, receipt adoption or production acceptance. Historical
producer/policy/FILES modules and old builder/check functions stay unchanged.

## Two source roles, one new prefix

Scientific **and test** source is fixed detached
`fb11b78d1f44658d3a9d62365b77f3d217f53129`, tree
`3a1922c885498131b3b51b7214dcab429cac073e`.
Ops entrypoint commit/tree are explicit, separate, clean, descended fromfb11;
only these five ops paths may differ:

* `devtools/isapol_native_acceptance/{phoenix_build.py,qomp.sbatch}`
* `devtools/isapol_s66_campaign/{phoenix_installed_checks.py,test_qomp_production_ops.py,QOMP_QUALIFICATION.md}`

Closed remote paths, no CLI overrides:

```
P=/storage/project/r-cs207-0/awallace43
SCI_SOURCE=$P/gits/psi4.ccasp-native-s66-qomp-fb11-science001
OPS_SOURCE=$P/gits/psi4.ccasp-native-s66-qomp-ops002
BUILD=$P/runs/ccasp-native-s66-qomp-fb11-build002
CHECK=$P/runs/ccasp-native-s66-qomp-fb11-installed002
BASE=$P/miniconda/envs/p4cuest_sapt
```

Both run roots and `/storage/scratch1/3/awallace43/<run-basename>` must be new.
Separate sibling `-logs` directories hold SLURM logs; parent creates only those
before submission. Never mutate frozen old/deployed sources,1aaa/8558, failed
history, accepted fixtures, old25/dataset357 or any held job.

## Build (requires separate parent launch approval)

The helper library is loaded from **verified Git bytes**, not a mutable module/
cache/path search:
`8573ded3233001b50d99c26bf9a577e13b68c956:devtools/isapol_q_omp/ops.py`,
SHA256 `f14a0e7be6cc73544de07b0427dd4dba6a5d8b5b137d0adc7586dd5e4bd269e3`.
Parent must stage that Git object as well as SCI and ops. Its731 source gate/
oracle/84-case dispatch is **never called**.

The new mode uses its offline clone/link, dependency, configure, environment,
mapping, seal, identity and finite Runner helpers. It materializes a NEW
`BUILD/source` detachedfb11, offline-copies the qualified1aaa dependency env
into NEW`BUILD/env`, rejects install-through/external directory links and
preserves dependency bytes. It never copies8558 or invokes pip/downloads.

Original approved record is exactly4356bytes/SHA
`896949860e3eb012e07b2ec45556ab4580d87e36de094ca5567d800149803380`.
Only source/build/prefix/OLD-env-rooted paths are substituted; no flags,
toolchain/preload/duplicate override, thresholds, grids or stack change.
Actual CMake configure/build/install uses-j2. GCC_ROOT/site includes/compiler/
install/CMake/Python/plugin contamination is sanitized. No inherited
OMP_PROC_BIND/OMP_PLACES is accepted; no new affinity policy is added.

The private CWD exists before imports. RecordedRunner reuses the reviewed
Runner but retains each actual Popen returncode, including failures/signals,
in `*-result.json`; argv-only command records are not claimed to attest exits.
Its `interrupted` field is the inherited stop latch, also set on command errors.

Full installed Python mapping precedes import: generated3 come from NEW
configured build; source/driver/header, flat unit-test helpers and nested
sample tests all match strict source bytes. Unknown/missing/ambiguous/tampered
files fail. Prefix/source/dependency/OLD checks surround collection; complete
OLD/env and BASE file/namespace/link snapshots are checked again on failure.
External old runtime dependencies are separately verified. These are ordinary
host-trust byte/link seals, **not permissions/transient-write/network sandboxes**.
Snapshots traverse only these fixed env prefixes, never user-wide scratch.

After actual compile/install, collection only runs in the build allocation:
private Psi4/core asserted before pytest/helper imports; only verified
installed`package/tests` is added for bare test helper imports. No SCI/ops root
on sys.path. Tests execute from exactfb11 when qualification is later authorized.
Collection imports test modules/parametrizations, not fixtures or test bodies.

Expected unique union **556**: full461 legacy grid400 cases +34Q +18performance
+32partition response +9grid contracts +2 molecular nodes:

```
test_isapol_distribution.py::test_isa_nonconvergence_and_admission
test_isapol_distribution.py::test_isa_plan_charges_drho_refinement_once_and_to_the_cap
```

Thus95 focused/native+molecular and the full461, no overlap. Actual ordered
nodes/group/module counts, hashes of all test files, source role, command/
actual returncode digests and raw collection are saved. Any count/import/
collection skip/error stops; no count/filter override or skip workaround.

On success the builder creates **genuine new**
`fresh-native-build-identity/1`, a complete build record, actual unknown identity
before/after and an explicitly unqualified result. No old receipt is renamed.
Parent verifies allocation/.batch/.0COMPLETED0, manifest/record/collection and
all source/runtime proofs before providing literal digests for checks.

## Exact parent CLI (not executed locally)

Use actual reviewed ops commit/tree, not SCI as an ops alias:

```bash
# Parent first checks all freshness/protected paths and sbatch --test-only.
# No script here submits, retries, resumes or chains work.
/opt/slurm/current/bin/sbatch --parsable --export=USER,LOGNAME,HOME,PATH,LANG,LC_ALL \
 --account=gts-cs207-chemx --partition=cpu-small --qos=embers \
 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=08:00:00 --no-requeue \
 --chdir="$BUILD-logs" --output="$BUILD-logs/job-%j.out" --error="$BUILD-logs/job-%j.err" \
 "$OPS_SOURCE/devtools/isapol_native_acceptance/qomp.sbatch" \
 build "$OPS_SOURCE" "$OPS_COMMIT" "$OPS_TREE"

# ONLY AFTER parent independently verifies actual new build+collection data:
/opt/slurm/current/bin/sbatch --parsable --export=USER,LOGNAME,HOME,PATH,LANG,LC_ALL \
 --account=gts-cs207-chemx --partition=cpu-small --qos=embers \
 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=02:00:00 --no-requeue \
 --chdir="$CHECK-logs" --output="$CHECK-logs/job-%j.out" --error="$CHECK-logs/job-%j.err" \
 "$OPS_SOURCE/devtools/isapol_native_acceptance/qomp.sbatch" \
 checks "$OPS_SOURCE" "$OPS_COMMIT" "$OPS_TREE" \
 "$ACTUAL_BUILD_IDENTITY_SHA" "$ACTUAL_BUILD_RECORD_SHA" "$ACTUAL_COLLECTION_SHA"
```

These are parent-supplied observed literals, not self-approved current hashes.
Checks require genuine fresh manifest/record/COMPLETED/noFAILED, actual0 build
commands, source roles, full mapping/dependencies and job/batch/step accounting.
Old READY/32/82/failure receipts, foreign roots/identities, stale digests fail.
One whole556-node installed-only pytest run uses the same collection/bootstrap;
actual pass nodes and JUnit names must equal the reviewed ordered collection.
No skips/errors/failures, missing/substituted/duplicate cases or late prefix/
source/dependency changes can publish success. Molecular fixture does real SCF
on compute; no local execution/collection is allowed.

TERM/INT remains detected through finalization, including after COMPLETED
publication. Failure invalidates that marker while preserving evidence.
No requeue/retry/recovery/adoption. Qualified interface report remains
acceptance/adoptionfalse. Only actual returned evidence can justify a later
closed measured profile; this code has no such profile or activation.

## Pure local checks

```bash
python3 -I -B devtools/isapol_s66_campaign/test_qomp_production_ops.py
bash -n devtools/isapol_native_acceptance/qomp.sbatch
git diff --check
```

Tests use stdlib fakes, fake receipts/ELF bytes, temp roots and a fake srun;
no native import/compiler/cluster/server call. Signal tests target both active
payload and FINALIZATION after fake completion publication for both modes.
Run documented campaign pure suites and frozen8573 pure ops regressions too.
No new performance, full solver/scientific acceptance or water3 attribution
claim. Inode policy147/cohort148 remain separate.

##149 control and import confinement correction (child of40ce)

The submitted command above is permitted **only from a known clean parent
submission context**: USER/LOGNAME=awallace43, the approved HOME,
PATH=/usr/bin:/bin, LANG/LC_ALL=C.UTF-8. Export only those explicitly approved
fields, not the inherited environment/ALL. Parent verifies site submission
does not repopulate an unreviewed login environment. No LD_*/DYLD_*, Git repo/
config selectors, BASH_ENV/startup injection or exported functions are
approved at scheduler launch.

The wrapper pins `/bin/bash`, and uses shell builtins to discard other
inherited environment variables BEFORE its first external command, including
token Python, srun, payload and finalizers. Control Python runs-I-S-B.
The Python control environment independently uses a small explicit scheduler/
identity allowlist and clean PATH; inherited GIT_* selectors/count/key/value,
loader and Python injection are absent. Its owned Git settings disable
system/global config, prompts and attribute files. All qomp verification,
library loading, detachment checks and materialization pin `/usr/bin/git`,
disable hooks/fsmonitor/pagers and transports. Local/worktree includes and
executable hooks/filter/diff/alias config are rejected, not executed.
The historical materializer/8573 library remain unchanged; qomp uses a small
pinned-Git materialization adapter. Model child environment starts from these
clean controls, then adds only the reviewed prefix/BASE loader paths.

**Boundary:** neither cleanup can undo code already executed by the scheduler's
initial shell loader or site startup. System binaries, scheduler and ordinary
host/repository trust still apply. This is not a hostile-host or network sandbox.

Guarded identity/collection/test Python also uses-I-S-B, avoiding automatic
site/.pth execution. Before importing Psi4 it validates every resolved sys.path
entry: only BASE's reviewed3.13 stdlib/zip/dynload and private site-packages;
then adds only the already sealed private site directory. It checks again
after Psi4 (which may append addon paths). SCI, ops, build-source trees,
foreign site-packages, empty/relative/unknown and symlink-escaping entries
fail. Before pytest/helper imports, every installed test Python file is checked
against the complete source-verified inventory; ONLY then may the resolved
private package/tests directory be added. Paths are rechecked before/after
pytest import. No foreign addon path gets an automatic allowance. Actual
installed compatibility remains a compute gate, not a local-native claim.

The unchanged trusted fb11 test bodies may launch their own subprocesses;
this top-level bootstrap is not a test-code/process sandbox. Existing complete
runtime/dependency before/after and actual node/JUnit gates still apply.
Regression tests use stdlib fakes, temporary Git repositories, inert marker
hooks/fakegit and loader-variable assertions before any exec; no hostile
shared object or real Psi4 is loaded.

##150 optional worktree-config correction

Actual startup15320435 failed in9s, before BUILD creation, compilation or
native collection. Parent's read-only capture be8f922ae found **Git2.52.0**,
extensions.worktreeConfig=true and an absent selected administrative
`config.worktree`; explicit `git config --worktree ... --list` returned128.
This is normal optional-file absence, **not an older-Git-specific problem**,
scientific failure or an invitation to swallow128.

When enabled, the audit now obtains BOTH the absolute administrative
directory and selected config path from pinned clean Git. The exact
`admin/config.worktree` may be skipped only when lstat proves it absent
with the administrative directory still present/unchanged. Present files
must be regular, nonsymlink and actually readable via no-follow open/fstat.
An escaping/dangling link, unreadable file, malformed content, unsafe keys,
administrative mismatch or any Git command/read error still fails closed.
Disabled extensions remain disabled; their unselected optional file is not read.
No remote config is edited or fabricated. These checks remain ordinary-host
observations, not a hostile concurrent-filesystem sandbox.

All three new closed operations/build/check names are002; scratch inherits
those basenames. Reuse the immutable detached fb11 **science001** source.
Preserve failed ops001, all001 logs/intents and the absent001 run names;
never retry/overwrite them. Parent stages/submits only after independent
review/approval. The same approved configure/helper/science556/cap/seal/TERM
contracts remain; no actual new identity/qualification exists yet.
