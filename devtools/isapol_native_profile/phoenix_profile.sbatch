#!/usr/bin/env bash
#SBATCH --job-name=ccasp-native-profile-0001
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=TERM@120
#
# Parent supplies ABSOLUTE --output/--error to sbatch; their parent must exist.
# Source/run/fixture/commit arguments below are mandatory. No install or build.
# This file may be sourced ONLY by pure wrapper tests; main uses fixed science.

set -euo pipefail
umask 077

die() { printf 'ERROR: %s\n' "$*" >&2; return 1; }

verify_source() {
    local observed top changes
    [[ $commit =~ ^[0-9a-f]{40}$ ]] || { die "full source commit required"; return 1; }
    observed=$(git -C "$source" rev-parse HEAD) || return
    [[ $observed == "$commit" ]] || { die "source commit differs"; return 1; }
    top=$(git -C "$source" rev-parse --show-toplevel) || return
    [[ $top == "$source" ]] || { die "source must be Git root"; return 1; }
    changes=$(git -C "$source" status --porcelain --untracked-files=all) || return
    [[ -z $changes ]] || { die "source is not clean"; return 1; }
    local name
    for name in .gitignore README.md fixtures.json profile.py telemetry.py test_profile.py phoenix_profile.sbatch; do
        git -C "$source" ls-files --error-unmatch "devtools/isapol_native_profile/$name" >/dev/null || return
    done
}

check_allocation() {
    /usr/bin/python3 -I -B - "$1" <<'PY'
import sys
row = dict(part.split("=", 1) for part in open(sys.argv[1]).read().split() if "=" in part)
required = dict(Account="gts-cs207-chemx", QOS="embers", Partition="cpu-small",
                NumNodes="1", NumTasks="1", NumCPUs="8", **{"CPUs/Task": "8"},
                MinMemoryNode="32G", TimeLimit="02:00:00", Requeue="0", Restarts="0")
bad = {k: (row.get(k), v) for k, v in required.items() if row.get(k) != v}
if bad:
    raise SystemExit("allocation differs: " + repr(bad))
PY
}

# This runs as the single srun task. The payload has its own process group, so
# TERM reaches Python as well as /usr/bin/time, not merely its immediate parent.
task_payload() {
    local timing=$1 affinity=$2 owned_cwd=$3
    shift 3
    cd -- "$owned_cwd"
    set +m
    local payload= interrupted=0 status=0
    forward_payload() {
        interrupted=1
        if [[ -n $payload ]]; then kill -TERM -- "-$payload" 2>/dev/null || true; fi
    }
    trap forward_payload TERM INT
    /usr/bin/python3 -I -B - "$affinity" <<'PY'
import json, os, sys
with open(sys.argv[1], "x") as f:
    json.dump(dict(pid=os.getpid(), cwd=os.getcwd(), affinity=sorted(os.sched_getaffinity(0)),
                   allocated_cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
                   omp_proc_bind=os.environ.get("OMP_PROC_BIND"),
                   omp_places=os.environ.get("OMP_PLACES")), f, indent=2)
PY
    (( interrupted == 0 )) || return 143
    /usr/bin/setsid /usr/bin/time -v -o "$timing" -- "$@" &
    payload=$!
    if (( interrupted )); then forward_payload; fi
    wait "$payload" || status=$?
    if (( interrupted )); then
        # TERM can interrupt wait before the payload actually terminates.
        while kill -0 "$payload" 2>/dev/null; do wait "$payload" || true; done
        status=143
    fi
    trap - TERM INT
    return "$status"
}

monitor_step() {
    # Only inspect the newly owned scratch root. The parent signals this monitor
    # to finish after wait; any scan failure, byte breach or deadline is failure.
    /usr/bin/python3 -I -B - "$scratch_root" "$run_root" "$1" "$2" "$3" <<'PY'
import json, os, pathlib, signal, sys, time
scratch, output = map(pathlib.Path, sys.argv[1:3])
child, remaining, label = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
stop_file = output / (label + "-monitor-stop")
cap = 256 * 1024**3
deadline = time.monotonic() + max(0, remaining)
stopped = False
def stop(*unused):
    global stopped
    stopped = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
peaks = dict(logical_bytes=0, allocated_bytes=0, files=0, directories=0, symlinks=0)
failure = None
samples = 0
def scan():
    current = dict.fromkeys(peaks, 0)
    for directory, dirs, files in os.walk(scratch, followlinks=False,
                                         onerror=lambda error: (_ for _ in ()).throw(error)):
        for name in dirs + files:
            path = pathlib.Path(directory) / name
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            if path.is_symlink():
                current["symlinks"] += 1
            elif name in dirs:
                current["directories"] += 1
            else:
                current["files"] += 1
                current["logical_bytes"] += stat.st_size
                current["allocated_bytes"] += stat.st_blocks * 512
    return current
try:
    with (output / (label + "-scratch-samples.jsonl")).open("x") as stream:
        while True:
            stopped = stopped or stop_file.exists()
            current = scan()
            for key, value in current.items():
                peaks[key] = max(peaks[key], value)
            samples += 1
            stream.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(), **current)) + "\n")
            stream.flush()
            if max(current["logical_bytes"], current["allocated_bytes"]) > cap:
                failure = "scratch_byte_cap"
            elif not stopped and time.monotonic() >= deadline:
                failure = "workflow_deadline"
            if failure or stopped:
                break
            time.sleep(0.25)
except BaseException as error:
    failure = "scratch_monitor_error: " + repr(error)
finally:
    if failure and not stopped:
        try:
            os.kill(child, signal.SIGTERM)  # srun forwards to task_payload.
        except ProcessLookupError:
            pass
    with (output / (label + "-scratch-monitor.json")).open("x") as stream:
        json.dump(dict(status="FAILED" if failure else "STOPPED_AFTER_STEP",
                       failure=failure, cap_bytes=cap, samples=samples, sampled_maxima=peaks,
                       limitation="Sampled lower bounds; short-lived/deleted-open files may be missed. "
                                  "No inode reservation/cap; metric peaks need not coincide."),
                  stream, indent=2)
sys.exit(42 if failure else 0)
PY
}

validate_step() {
    /usr/bin/python3 -I -B - "$run_root/$1" "$2" "$3" <<'PY'
import json, pathlib, sys
root, threads, mode = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
def read(name):
    return json.loads((root / name).read_text())
manifest = read("manifest.json")
assert manifest["threads"] == threads and manifest["science_memory_gib"] == 20
assert manifest["case_metadata"]["monomer_id"] == "monomer-0001"
assert manifest["declarations"] == dict(partition_grid=[400, 2030], response_grid=[100, 200],
                                       drho_refinement_iterations=32, response="reference_h2h1")
assert manifest["protocols_unchanged"] is True
assert manifest["instrumentation"] == ("off" if mode == "off" else "stages")
if mode == "check":
    assert manifest["status"] == "ADMITTED_NOT_EXECUTED"
else:
    result = read("summary.json")
    assert result["status"] == "COMPUTED_NOT_ACCEPTED" and result["acceptance"] is False
    assert result["installed_science_unchanged"] is True
    assert result["telemetry"]["errors"] == []
    for name in ("server-native-envelope.json", "server-tensors.npz", "native-result.json",
                 "qcschema-result.json", "events.jsonl", "samples.jsonl", "hooks.json"):
        assert (root / name).stat().st_size > 0, "missing full export: " + name
assert read("wall-time-source.json")["recognized_qcengine_wrapper_timer"] is True
PY
}

on_term() {
    interrupted=1
    printf 'TERM/INT: forwarding to active srun step\n' >&2
    if [[ -n ${child:-} ]]; then kill -TERM "$child" 2>/dev/null || true; fi
}

on_exit() {
    local status=$?
    # Bash unwinds caller locals on errexit before EXIT. Keep immutable copies
    # outside main's dynamic scope, including failure and signal paths.
    local source=$profile_exit_source commit=$profile_exit_commit run_root=$profile_exit_run_root
    trap - EXIT
    if [[ -n ${monitor:-} ]]; then
        : > "$run_root/$current_label-monitor-stop"
        wait "$monitor" || status=42
    fi
    if (( interrupted )); then status=143; fi
    # Recheck even after failure/TERM. Never mask a source drift with COMPLETED.
    if ! verify_source; then status=43; fi
    if (( status == 0 && matrix_finished == 0 )); then status=44; fi
    printf 'exit_code=%s\ninterrupted=%s\nsource_commit=%s\nsource_tree=%s\n' \
        "$status" "$interrupted" "$commit" "$(git -C "$source" rev-parse 'HEAD^{tree}')" \
        >> "$run_root/wrapper-status.txt"
    if (( status == 0 )); then
        printf 'COMPLETED: all six steps returned and artifacts verified; NOT ACCEPTANCE.\n' \
            > "$run_root/COMPLETED"
    else
        printf 'FAILED: exit=%s; partial evidence retained; no retry.\n' "$status" > "$run_root/FAILED"
    fi
    exit "$status"
}

run_step() {
    local label=$1 threads=$2 mode=$3 status=0 monitor_status=0 remaining
    (( interrupted == 0 )) || return 143
    verify_source
    (( interrupted == 0 )) || return 143
    remaining=$((deadline - SECONDS))
    (( remaining > 0 )) || die "finite workflow deadline exhausted"
    local flag=--run instrumentation=stages
    [[ $mode != check ]] || flag=--check-only
    [[ $mode != off ]] || instrumentation=off
    printf 'START %s threads=%s fixed_allocation_cpus=8\n' "$label" "$threads" \
        >> "$run_root/wrapper-status.txt"
    mkdir -- "$scratch_root/launch-$label"
    "$srun" --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
        --chdir="$scratch_root/launch-$label" \
        --kill-on-bad-exit=1 --wait=30 -- \
        /usr/bin/bash "$source/devtools/isapol_native_profile/phoenix_profile.sbatch" --payload \
        "$run_root/$label-time.txt" "$run_root/$label-affinity.json" "$scratch_root/launch-$label" \
        /usr/bin/env -u PYTHONPATH -u PYTHONHOME -u PSIPATH -u PSIDATADIR -u LD_PRELOAD \
        PATH="$build/env/bin:$base/bin:/usr/bin:/bin" \
        LD_LIBRARY_PATH="$build/env/lib:$base/lib" \
        TMPDIR="$scratch_root/launch-$label" SCRATCH="$scratch_root/launch-$label" \
        PSI_SCRATCH="$scratch_root/launch-$label" \
        OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS=1 \
        "$science_python" -I -B -X "pycache_prefix=$scratch_root/$label/cache" \
        "$source/devtools/isapol_native_profile/profile.py" "$flag" \
        --fixtures "$fixtures" --build-manifest "$build/build-identity.json" \
        --qualification "$qualification" --qcengine-package "$qce" \
        --qcengine-harness-sha256 "$harness_sha" --threads "$threads" \
        --instrumentation "$instrumentation" --sample-seconds 1 \
        --scratch "$scratch_root/$label" --output "$run_root/$label" &
    child=$!
    if (( interrupted )); then on_term; fi
    current_label=$label
    monitor_step "$child" "$remaining" "$label" &
    monitor=$!
    wait "$child" || status=$?
    if (( interrupted )); then
        while kill -0 "$child" 2>/dev/null; do wait "$child" || true; done
        status=143
    fi
    child=
    : > "$run_root/$label-monitor-stop"
    wait "$monitor" || monitor_status=$?
    monitor=
    (( monitor_status == 0 )) || return "$monitor_status"
    (( status == 0 )) || return "$status"
    (( interrupted == 0 )) || return 143
    validate_step "$label" "$threads" "$mode"
    printf 'VERIFIED %s\n' "$label" >> "$run_root/wrapper-status.txt"
}

# Test seam: tests supply fake srun/driver paths, NEVER scientific installations.
# Normal main below does not accept overrides for fixed science or SLURM tools.
execute_plan() {
    profile_exit_source=$source
    profile_exit_commit=$commit
    profile_exit_run_root=$run_root
    child= monitor= interrupted=0 matrix_finished=0
    trap on_term TERM INT
    trap on_exit EXIT
    run_step check8 8 check
    run_step stages1 1 stages
    run_step stages2 2 stages
    run_step stages4 4 stages
    run_step stages8 8 stages
    run_step off8 8 off
    verify_source
    matrix_finished=1
}

usage() {
    printf '%s\n' \
      'phoenix_profile.sbatch --source ABS_GIT_ROOT --commit FULL_SHA --run-root NEW_ABS_ROOT \' \
      '  --scratch-root NEW_ABS_ROOT --fixtures ABS_MONOMER_0001 --harness-sha256 WORKER_RECEIPT_SHA'
}

main() {
    local started=$SECONDS
    local source= commit= run_root= scratch_root= fixtures= harness_sha=
    while (( $# )); do
        case "$1" in
            --source|--commit|--run-root|--scratch-root|--fixtures|--harness-sha256)
                (( $# >= 2 )) || die "missing argument: $1"
                case "$1" in
                    --source) source=$2;; --commit) commit=$2;; --run-root) run_root=$2;;
                    --scratch-root) scratch_root=$2;; --fixtures) fixtures=$2;;
                    --harness-sha256) harness_sha=$2;;
                esac
                shift 2;;
            --help) usage; return 0;;
            *) die "unknown argument: $1";;
        esac
    done
    [[ -n ${SLURM_JOB_ID:-} && ${SLURM_CPUS_PER_TASK:-} == 8 ]] || die "fresh eight-CPU allocation required"
    [[ ${SLURM_RESTART_COUNT:-0} == 0 ]] || die "no requeue/restarts"
    [[ $commit =~ ^[0-9a-f]{40}$ && $harness_sha =~ ^[0-9a-f]{64}$ ]] || die "full commit and worker harness SHA required"
    [[ $harness_sha == 31c59282cafe0405468ea320e58039cb633acd873264d16ae38dfeac9fd60781 ]] \
        || die "must use previously recorded accepted0001 worker harness SHA"
    local path
    for path in "$source" "$run_root" "$scratch_root" "$fixtures"; do
        [[ $path == /* && $path != / ]] || die "absolute dedicated paths required"
    done
    source=$(/usr/bin/realpath -e "$source")
    fixtures=$(/usr/bin/realpath -e "$fixtures")
    run_root=$(/usr/bin/realpath -m "$run_root")
    scratch_root=$(/usr/bin/realpath -m "$scratch_root")
    [[ ${fixtures##*/} == monomer-0001 ]] || die "first diagnostic is fixed accepted0001"
    [[ ! -e $run_root && ! -e $scratch_root ]] || die "new nonexistent run/scratch roots required"
    local build=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-build001
    local qualification=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-installed001/payload/installed-validation.json
    local base=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
    local qce=/storage/project/r-cs207-0/awallace43/gits/qcengine/qcengine
    local srun=/opt/slurm/current/bin/srun
    local science_python=$build/env/bin/python
    for path in "$source" "$fixtures" "$build" "${qualification%/*}" "$qce"; do
        [[ $run_root != "$path" && $run_root != "$path/"* &&
           $scratch_root != "$path" && $scratch_root != "$path/"* ]] || die "protected path overlap"
    done
    [[ $run_root != "$scratch_root/"* && $scratch_root != "$run_root/"* &&
       $run_root != "$scratch_root" ]] || die "run/scratch roots must be disjoint"
    [[ -x $srun && -x $science_python ]] || die "fixed installed tools missing"
    verify_source
    mkdir -p -- "${run_root%/*}" "${scratch_root%/*}"
    mkdir -- "$run_root" "$scratch_root"
    mkdir -- "$run_root/wrapper-cwd"
    cd -- "$run_root/wrapper-cwd"
    /opt/slurm/current/bin/scontrol show job -o "$SLURM_JOB_ID" > "$run_root/allocation.txt"
    check_allocation "$run_root/allocation.txt"
    printf 'source=%s\ncommit=%s\nfixture=%s\njob=%s\naccepted_qce=v0.34.1+3.gd56a0598\nharness_sha=%s\n' \
        "$source" "$commit" "$fixtures" "$SLURM_JOB_ID" "$harness_sha" > "$run_root/wrapper-status.txt"
    # Five minutes reserved for TERM/native cleanup before the hard two-hour cap.
    local deadline=$((started + 6900))
    execute_plan
    exit 0  # Keep main's fixed path locals alive for the EXIT identity check.
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    if [[ ${1:-} == --payload ]]; then
        shift
        task_payload "$@"
    else
        main "$@"
    fi
fi
