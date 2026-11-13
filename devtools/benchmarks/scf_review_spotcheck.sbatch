#!/usr/bin/env bash
# Supply absolute --output/--error paths at submission.
#SBATCH --job-name=scf-review-spotcheck
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --constraint=gold6226
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@120

set -euo pipefail
mode=$1
root=$2
after=$3
before=$4
commit=$5
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4es
script="$after/devtools/benchmarks/scf_review_spotcheck.py"
batch="$after/devtools/benchmarks/scf_review_spotcheck.sbatch"

if [[ "$mode" == payload ]]; then
    test "$(git -C "$after" rev-parse HEAD)" = "$commit"
    test "$(git -C "$before" rev-parse HEAD)" = f21cb6880866c61fc80900af6a0a75642ada12c9
    test -z "$(git -C "$after" status --porcelain --untracked-files=no)"
    test -z "$(git -C "$before" status --porcelain --untracked-files=no)"
    unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
    unset OMP_PROC_BIND OMP_PLACES PYTHONPATH
    export PATH="$env/bin:/opt/slurm/current/bin:/usr/bin:/bin"
    export LD_LIBRARY_PATH="$env/lib"
    export PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24
    export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
    mkdir -p "$root/runtime-scratch"
    export TMPDIR="$root/runtime-scratch" SCRATCH="$root/runtime-scratch" PSI_SCRATCH="$root/runtime-scratch"
    "$env/bin/python" --version
    /storage/project/r-cs207-0/awallace43/miniconda/bin/conda list -p "$env" --explicit > "$root/environment.txt"
    for arm in before after; do
        source="$before"
        [[ "$arm" == after ]] && source="$after"
        cmake -S "$source" -B "$root/build-$arm" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DENABLE_XHOST=ON -DENABLE_simint=OFF \
            -DCMAKE_C_COMPILER="$env/bin/x86_64-conda-linux-gnu-cc" \
            -DCMAKE_CXX_COMPILER="$env/bin/x86_64-conda-linux-gnu-c++" \
            -DCMAKE_PREFIX_PATH="$env" -DPython_EXECUTABLE="$env/bin/python" \
            -DBLAS_TYPE=MKL -DLAPACK_TYPE=MKL > "$root/configure-$arm.log" 2>&1
        cmake --build "$root/build-$arm" -j 24 > "$root/build-$arm.log" 2>&1
        # Verify each staged executable's Python setup, not an environment release.
        eval "$("$root/build-$arm/stage/bin/psi4" --psiapi)"
        "$env/bin/python" -c 'import psi4; print(psi4.__file__, psi4.__version__)'
    done
    mkdir -p "$root/tests"
    (
        cd "$root/tests"
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        "$env/bin/python" -c 'import psi4,pytest,sys; psi4.set_num_threads(1); raise SystemExit(pytest.main([sys.argv[1], "-q", "-k", "grid_switch", "--junitxml=grid-switch.xml"]))' \
            "$after/tests/pytests/test_scf_optimization.py"
    )
    "$env/bin/python" "$script" prepare "$root/timings" \
        --before "$root/build-before/stage/lib" --after "$root/build-after/stage/lib" \
        --threads 24 --interleave --commit "$commit" \
        --inputs "$after/devtools/benchmarks/scf_review_spotcheck.json"
    "$env/bin/python" "$script" run "$root/timings"
    exit 0
fi

test "$mode" = launch
mkdir -p "$root/metadata"
printf 'job_id=%s\nhost=%s\ncommit=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$commit" \
    > "$root/metadata/runtime.txt"
term_received=0
child=
on_term() {
    term_received=1
    [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true
}
on_exit() {
    status=$?
    (( term_received )) && status=143
    printf 'exit_code=%s\nterm_received=%s\n' "$status" "$term_received" >> "$root/metadata/runtime.txt"
    if (( status == 0 )); then
        touch "$root/metadata/COMPLETED"
    else
        touch "$root/metadata/FAILED"
    fi
}
trap on_term TERM
trap on_exit EXIT
srun --cpu-bind=none bash "$batch" payload "$root" "$after" "$before" "$commit" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
