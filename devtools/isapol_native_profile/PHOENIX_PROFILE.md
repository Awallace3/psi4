# Water-only bounded wrapper handoff

## Scope

`phoenix_profile.sbatch` is the sole new runtime wrapper. No scientific kernels,
thresholds, bases, grids, methods, resource caps, receipts or datasets change.
First diagnostic is **accepted0001 water only**, fresh SCF/native properties,
one node/task, **8 CPUs fixed**, **32 GiB job**, **20 GiB science**, **2 h**,
`gts-cs207-chemx`, `cpu-small`, explicit free `embers`, **no requeue**.

The ordered plan is:

```
check8 --check-only, fresh output/scratch
stages1 --run --threads1
stages2 --run --threads2
stages4 --run --threads4
stages8 --run --threads8
off8    --run --threads8 --instrumentation off
```

Every srun step still requests eight CPUs and binds to cores. No new
`OMP_PROC_BIND` or `OMP_PLACES` values are introduced. Inherited values and actual
task affinity are recorded. The off control still samples scratch and captures
output; it is not pristine production execution.

## Fixed scientific paths

```
BUILD=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-build001
QUAL=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-installed001/payload/installed-validation.json
BASE=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
QCE=/storage/project/r-cs207-0/awallace43/gits/qcengine/qcengine
```

The wrapper accepts no overrides for those paths or for `/opt/slurm/current/bin/srun`.
Driver verifies the matching qualified installation; no install/build takes place.
The copied fixtures remain read-only. Partition/Q400/2030, response100/200, ISA
recipe100/200, Drho32 and H2H1 remain exact.

The accepted0001 worker preflight supplied by parent (`b347129b8`) records:

```
harness=/storage/project/r-cs207-0/awallace43/gits/qcengine/qcengine/programs/psi4.py
sha256=31c59282cafe0405468ea320e58039cb633acd873264d16ae38dfeac9fd60781
QCE=v0.34.1+3.gd56a0598
```

This is a previously recorded worker hash, **not newly self-approved**. Wrapper
requires it, and the driver reads/hashes actual QCE source for wall-time scope.

## Parent-supplied Git/input/run paths

```
SOURCE=/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-profile001
FIXTURE=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-profile-fixtures001/monomer-0001
ROOT=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-profile-water001
SCRATCHROOT=/storage/scratch1/3/awallace43/ccasp-native-profile-water001
```

Source must be a **clean committed Git root**, pinned to the full handoff commit
reported separately. The wrapper checks HEAD and cleanliness before, between,
and after steps, including failure/TERM exit. It records the source tree.
Source must be staged through Git, not copied as tracked files.

`ROOT` and `SCRATCHROOT` must **not exist at launch**. They must be disjoint from
each other and from source, fixtures, build, receipts and QCE. The wrapper creates
new per-step children, never overwrites an existing run, never cleans evidence.

SLURM log directories must exist **outside** those two new roots before sbatch.
Suggested separate fresh log root:

```
LOGROOT=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-profile-water001-logs
```

Use it as initial batch CWD via `--chdir`; do not launch from source/fixtures.

## Exact parent CLI

Parent obtains `COMMIT` from the final handoff and first verifies Git staging,
independent review completion, fixture hashes and that ROOT/SCRATCHROOT are absent.
The following is a command template for parent execution, **not an executed
submission**:

```bash
SOURCE=/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-profile001
FIXTURE=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-profile-fixtures001/monomer-0001
ROOT=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-profile-water001
SCRATCHROOT=/storage/scratch1/3/awallace43/ccasp-native-profile-water001
LOGROOT=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-profile-water001-logs
COMMIT=FULL_HANDOFF_COMMIT
HARNESSSHA=31c59282cafe0405468ea320e58039cb633acd873264d16ae38dfeac9fd60781

# Parent creates LOGROOT only; keep ROOT and SCRATCHROOT nonexistent.
mkdir -m 700 "$LOGROOT"

/opt/slurm/current/bin/sbatch --test-only \
  --chdir="$LOGROOT" --output="$LOGROOT/job-%j.out" --error="$LOGROOT/job-%j.err" \
  "$SOURCE/devtools/isapol_native_profile/phoenix_profile.sbatch" \
  --source "$SOURCE" --commit "$COMMIT" --run-root "$ROOT" \
  --scratch-root "$SCRATCHROOT" --fixtures "$FIXTURE" --harness-sha256 "$HARNESSSHA"

# Only parent, after review/preflight/route confirmation, may replace --test-only
# with --parsable and submit the identical finite job. No automated retry.
```

`scontrol show job` is read-only and validates actual account/QoS/partition,
node/task/CPU/memory/time/requeue/restart values before any profiling step.
No held jobs, QCA writes, manager or server actions are used.

## TERM and finite-runtime semantics

- Each active srun runs in background so the batch-shell trap is not deferred.
- Batch TERM/INT forwards TERM to the active srun. srun forwards to its task.
- The task starts `/usr/bin/time` plus Python in a new process group and forwards
  TERM to **that group**, not just the immediate time process.
- Both levels wait for real termination; interruption forces failure even if a
  child subsequently exits zero. No subsequent matrix step starts after TERM.
- A monotonic per-step monitor shares one workflow deadline of **6900 s**
  (five-minute cleanup margin). It sends TERM on deadline; SLURM's2h allocation
  is the final hard bound. Noninterruptible native work may wait for SLURM kill.
- `COMPLETED` is written only after all six zero exits, required output/summary
  checks, no monitor failure/interruption, and the final source identity check.
  It explicitly says **NOT ACCEPTANCE**. After the plan starts, other orderly
  terminal paths write `FAILED`; admission failures before that may have only
  the allocation/log evidence.
- SIGKILL/node loss cannot guarantee final markers; partial files remain evidence.

## CWD isolation (review blocker fixed)

The wrapper changes into a new owned `ROOT/wrapper-cwd`; each srun step uses
`SCRATCHROOT/launch-LABEL` as an explicit `--chdir` and task CWD. Environment
scratch variables initially point there.

Before **any Psi4 import**, including check-only, the driver then creates,
writability-probes and enters `SCRATCHROOT/LABEL/cwd`. It **never restores the
caller's CWD** before interpreter shutdown. This is necessary because core exit
finalization can append `timer.dat` and write `psi.PID.clean`, even for check-only
or partially failed imports. Those files and a small ownership marker stay in
owned scratch; neither source nor fixtures/install/old receipts are CWD targets.

Driver now resolves input paths before changing CWD. No science values change.

## Scratch monitoring and outputs

The wrapper monitor scans **only** the new scratch root (all step children),
without following symlinks, every0.25s. It records logical/allocated bytes and
actual file/directory counts. Observed logical **or** allocated bytes over
**256 GiB** fail and TERM the active step. There is **no arbitrary inode cap or
reservation**. Samples can miss transient or deleted-open files, so this is a
sampled guard, not a filesystem quota or guaranteed peak.

Wrapper evidence under ROOT:

```
allocation.txt
wrapper-status.txt
COMPLETED or FAILED (when orderly termination permits)
LABEL-affinity.json
LABEL-time.txt
LABEL-scratch-samples.jsonl
LABEL-scratch-monitor.json
LABEL-monitor-stop
check8/ stages1/ stages2/ stages4/ stages8/ off8/
```

Each science child retains the complete native envelope/result/tensors and
QCSchema result, stage timings, scratch/RSS samples, BLAS/thread metadata, output
sidecars and identity checks described in README.md. Parent independently
validates and compares all full tensors/Cn across threads and accepted baseline;
wrapper success does not claim numerical or scientific acceptance.

## Pure verification commands

```bash
PROFILE_FIXTURE_ROOT=/scratch/awallace43/ccasp-s66-native-campaign-20261030/native-perf-inputs001 \
  python3 -I -B devtools/isapol_native_profile/test_profile.py
python3 -I -B devtools/isapol_native_profile/test_phoenix_profile.py
bash -n devtools/isapol_native_profile/phoenix_profile.sbatch
```

Profiler regressions use **fake Psi4 modules** at the real pre-import path in
separate subprocesses; they verify protected sentinels remain untouched for
check-only/run, partial-import failures and runtime exceptions, including
deferred interpreter-exit writes. They perform no quantum calculations.

Wrapper tests use only a temporary Git repo, fake srun and a stdlib fake driver:
exact matrix/fixed allocation, source drift, nonzero child, zero exit with
missing evidence, TERM process-group forwarding, deadline, sparse scratch-byte
breach (no large physical allocation), CWD isolation and allocation rejection.
They invoke no real SLURM executable, network or science package.

Final local verification: **13 profiler tests +9 wrapper tests passed** (all
three immutable fixture sets enabled), plus `bash -n` and staged whitespace
checks. The CWD test includes five subprocess success/failure scenarios and
deferred finalizer writes. Failure-path wrapper tests caught and verified the
fix for Bash unwinding `main` locals before its EXIT trap; pinned exit identity
is now retained separately. No scientific timings or numerical agreement are
claimed by these pure tests.
