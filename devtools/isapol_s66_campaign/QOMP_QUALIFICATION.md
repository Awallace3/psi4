# Closed fb11 genuine build / installed-check mode

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
