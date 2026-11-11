#!/usr/bin/env bash
# Editable repo-root workflow; never rebuild a sibling qualified prefix.
set -euo pipefail
SOURCE=$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)
ROOT=$SOURCE/tmp
MODE=${1:-all}
if (($#)); then shift; fi
case "$MODE" in
    paths)
        printf 'SOURCE=%s\nOBJECTS=%s/obj\nINSTALL=%s/install\nRUNS=%s/runs\n' "$SOURCE" "$ROOT" "$ROOT" "$ROOT"
        exit 0;;
    all|build|run|compare|reconfigure) ;;
    *) echo "Usage: ./build.sh [all|build|run|compare|reconfigure|paths] [peptide_scf.py arguments]" >&2; exit 2;;
esac
needs_configure() {
    [[ "$MODE" == reconfigure || ! -f "$ROOT/obj/CMakeCache.txt" ||
       ! -f "$ROOT/obj/build.ninja" ]]
}
JOBS=${FSAPT_BUILD_JOBS:-2}
case "$JOBS" in 1|2|3|4) ;; *) echo "FSAPT_BUILD_JOBS must be 1–4" >&2; exit 2;; esac
ENV=$HOME/gits/psi4-disp0-cachy/environment-v2
export PATH="$ENV/bin:/usr/bin:/bin" CONDA_PREFIX="$ENV" MATH_ROOT="$ENV"
export CUDA_HOME="$ENV" CUDA_PATH="$ENV" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export LD_LIBRARY_PATH="$ROOT/runtime/lib:$ENV/lib" PYTHONNOUSERSITE=1 PYTHONPATH=""
export OMP_NUM_THREADS=${SCF_THREADS:-4} MKL_NUM_THREADS=${SCF_THREADS:-4}
export OPENBLAS_NUM_THREADS=${SCF_THREADS:-4} OMP_DYNAMIC=FALSE
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_PRELOAD
mkdir -p "$ROOT/runs"
# Runtime and frozen input are non-source assets. Keep private copies inside
# tmp; all tracked source changes continue to move through git.
if [[ ! -d "$ROOT/runtime" ]]; then
    RUNTIME_SOURCE=${SCF_RUNTIME_SOURCE:-$(dirname "$SOURCE")/runtime}
    test -d "$RUNTIME_SOURCE/cuest" || {
        echo "Set SCF_RUNTIME_SOURCE to the existing cuEST 0.2.50 runtime directory" >&2; exit 2;
    }
    cp -a --reflink=auto "$RUNTIME_SOURCE" "$ROOT/runtime"
fi
if [[ ! -f "$ROOT/inputs.json" ]]; then
    INPUT_SOURCE=${SCF_INPUTS:-$HOME/gits/psi4-fsapt-cpks-integration-cachy/runs/fsapt-jun-inputs-01/inputs.json}
    cp "$INPUT_SOURCE" "$ROOT/inputs.json"
fi
# The outer superbuild's compilation database is not the native core's.
# Leave an existing user-owned database/link alone unless it is ours.
COMPILE_DB=tmp/obj/psi4-core-prefix/src/psi4-core-build/compile_commands.json
if [[ -e "$SOURCE/compile_commands.json" || -L "$SOURCE/compile_commands.json" ]]; then
    [[ -L "$SOURCE/compile_commands.json" &&
       "$(readlink "$SOURCE/compile_commands.json")" == "$COMPILE_DB" ]] || {
        echo "Existing compile_commands.json differs; refusing to overwrite it" >&2; exit 2;
    }
else
    ln -s "$COMPILE_DB" "$SOURCE/compile_commands.json"
fi
if [[ "$MODE" == all || "$MODE" == build || "$MODE" == reconfigure ]]; then
    RP="$ROOT/runtime/lib;$ENV/lib"
    if needs_configure; then
    cmake -S "$SOURCE" -B "$ROOT/obj" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$ROOT/install" -DCMAKE_INSTALL_LIBDIR=lib \
        -DPYMOD_INSTALL_LIBDIR=/python3.13/site-packages \
        -DCMAKE_C_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-cc" \
        -DCMAKE_CXX_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-c++" \
        -DCMAKE_PREFIX_PATH="$ENV" -DPython_EXECUTABLE="$ENV/bin/python" \
        -DCUDAToolkit_ROOT="$ENV" -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_CUDA_COMPILER="$ENV/bin/nvcc" \
        -DCMAKE_CUDA_HOST_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-c++" \
        -DBLAS_TYPE=MKL -DLAPACK_TYPE=MKL -DCMAKE_INSIST_FIND_PACKAGE_Libxc=ON \
        -DLibxc_DIR="$ENV/lib/cmake/Libxc" -DENABLE_cuEST=ON \
        -DcuEST_ROOT="$ROOT/runtime/cuest" -DcuEST_INCLUDE_DIR="$ROOT/runtime/cuest/include" \
        -DcuEST_LIBRARY="$ROOT/runtime/cuest/lib/13/libcuest.so" \
        -DCMAKE_BUILD_RPATH="$RP" -DCMAKE_INSTALL_RPATH="$RP" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Dpsi4_CMAKE_UNITY_BUILD_BATCH_SIZE=4
    else
        echo "Reusing tmp/obj configuration (./build.sh reconfigure to refresh)"
    fi
    cmake --build "$ROOT/obj" --parallel "$JOBS"
    test -s "$SOURCE/$COMPILE_DB"
    cmake --install "$ROOT/obj"
    python - "$SOURCE" "$ROOT/install/manual-build.json" <<'PY'
import datetime, hashlib, json, subprocess, sys
from pathlib import Path
source,output=map(Path,sys.argv[1:])
git=lambda *args:subprocess.check_output(["git","-C",str(source),*args])
diff=git("diff","HEAD","--binary")
output.write_text(json.dumps({"head":git("rev-parse","HEAD").decode().strip(),
    "status":git("status","--porcelain").decode(),"diff_sha256":hashlib.sha256(diff).hexdigest(),
    "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope":"MANUAL_MUTABLE_BUILD_NOT_QUALIFICATION"},indent=2)+"\n")
output.with_suffix(".diff").write_bytes(diff)
PY
fi
[[ "$MODE" != build && "$MODE" != reconfigure ]] || exit 0
test -f "$ROOT/install/manual-build.json" || { echo "Run ./build.sh build first" >&2; exit 2; }
LOCK=$HOME/.cache/psi4-cachy
mkdir -p "$LOCK"
test ! -e "$LOCK/gpu0.HOLD" || { echo "GPU HOLD is active" >&2; exit 2; }
exec 9>>"$LOCK/gpu0.lock"
flock -n 9 || { echo "GPU is leased by another session; no test started" >&2; exit 2; }
test ! -e "$LOCK/gpu0.HOLD" || exit 2
eval "$("$ROOT/install/bin/psi4" --psiapi)"
export PSI4_CUEST_SCF_TIMERS=${PSI4_CUEST_SCF_TIMERS:-1}
RUN=$(mktemp -d "$ROOT/runs/scf-$(date +%Y%m%d-%H%M%S)-XXXXXX")
cp "$ROOT/install/manual-build.json" "$RUN/build.json"
echo "Outputs: $RUN"
run_one() {
    local backend=$1
    shift
    mkdir -p "$RUN/$backend/scratch"
    (
        cd "$RUN/$backend"
        export TMPDIR="$PWD/scratch" SCRATCH="$PWD/scratch" PSI_SCRATCH="$PWD/scratch"
        if [[ "$backend" == cpu ]]; then export CUDA_VISIBLE_DEVICES="";
        else export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}; fi
        python -u -B "$SOURCE/devtools/fsapt_integration/peptide_scf.py" \
            --prefix "$ROOT/install" --inputs "$ROOT/inputs.json" \
            --backend "$backend" --threads "$OMP_NUM_THREADS" "$@"
    ) 2>&1 | tee "$RUN/$backend/console.log"
}
if [[ "$MODE" == compare ]]; then
    run_one cpu "$@"
    run_one gpu "$@"
else
    run_one "${SCF_BACKEND:-gpu}" "$@"
fi
