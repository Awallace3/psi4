#!/usr/bin/env bash
# Mutable manual-debug root only; never rebuild the qualified FSAPT prefix.
set -euo pipefail
SOURCE=$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)
ROOT=$(dirname "$SOURCE")
test "$ROOT" = "$HOME/gits/psi4-scf-dimer-manual" || {
    echo "This helper is restricted to ~/gits/psi4-scf-dimer-manual/source" >&2; exit 2;
}
MODE=${1:-all}
if (($#)); then shift; fi
case "$MODE" in all|build|run|compare) ;; *)
    echo "Usage: ./build.sh [all|build|run|compare] [peptide_scf.py arguments]" >&2; exit 2;;
esac
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
if [[ "$MODE" == all || "$MODE" == build ]]; then
    RP="$ROOT/runtime/lib;$ENV/lib"
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
    cmake --build "$ROOT/obj" --parallel "$JOBS"
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
[[ "$MODE" != build ]] || exit 0
test -f "$ROOT/install/manual-build.json" || { echo "Run ./build.sh build first" >&2; exit 2; }
LOCK=$HOME/.cache/psi4-cachy
mkdir -p "$LOCK"
test ! -e "$LOCK/gpu0.HOLD" || { echo "GPU HOLD is active" >&2; exit 2; }
exec 9>"$LOCK/gpu0.lock"
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
        python -B "$SOURCE/devtools/fsapt_integration/peptide_scf.py" \
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
