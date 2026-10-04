#!/usr/bin/env bash
# Submit with explicit account/partition/resources and absolute --output/--error.
# Arguments: build|smoke|campaign SOURCE COMMIT RUN_ROOT
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=TERM@120
set -euo pipefail
umask 027
mode=$1
source_tree=$2
commit=$3
run_root=$4
root=/storage/project/r-cs207-0/awallace43
build="$source_tree/build_lifecycle"
old_source="$root/gits/psi4.saptdft_cuest_head"
old_commit=9317f406b2b66f33b65866cbc44a4189f913339e
test "$(git -C "$source_tree" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$source_tree" diff HEAD -- psi4 tests devtools)"
mkdir -p "$run_root/metadata"
metadata="$run_root/metadata/$mode-$SLURM_JOB_ID"
{
    printf 'mode=%s\ncommit=%s\njob_id=%s\nhost=%s\nstarted=%s\n' \
        "$mode" "$commit" "$SLURM_JOB_ID" "$(hostname -f)" "$(date -Iseconds)"
    scontrol show job "$SLURM_JOB_ID" -o
} > "$metadata.env"
# Conda compiler hooks legitimately read unset variables such as HOST.
set +u
source "$root/miniconda/etc/profile.d/conda.sh"
conda activate p4cuest_sapt
set -u
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
    export TMPDIR="/scratch/$USER/lifecycle-$SLURM_JOB_ID"
    mkdir -p "$TMPDIR"
fi
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
cd "$source_tree"

payload() {
    if [[ "$mode" == build ]]; then
        unset GCC_ROOT GCCROOT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH
        export CFLAGS="${CFLAGS:-} -march=x86-64-v3"
        export CXXFLAGS="${CXXFLAGS:-} -march=x86-64-v3"
        export FFLAGS="${FFLAGS:-} -march=x86-64-v3"
        export CMAKE_BUILD_PARALLEL_LEVEL="$SLURM_CPUS_PER_TASK"
        cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_Einsums=ON -DENABLE_XHOST=OFF \
            -DEinsums_DIR="$CONDA_PREFIX/share/cmake/Einsums" \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_cuEST=ON \
            -DCMAKE_INSIST_FIND_PACKAGE_cuEST=ON -DcuEST_ROOT="$CONDA_PREFIX" \
            -S "$source_tree" -B "$build" -G Ninja || return $?
        cmake --build "$build" -j "$SLURM_CPUS_PER_TASK" || return $?
        test -x "$build/stage/bin/psi4" || return $?
        printf '%s\n' "$commit" > "$run_root/metadata/BUILD_COMMIT"
        return
    fi
    # Python -c/-m searches cwd before PYTHONPATH. Never run from the source
    # root, where the unbuilt psi4/ package shadows stage/lib/psi4.
    cd "$run_root"
    built_commit=$(head -n 1 "$run_root/metadata/BUILD_COMMIT") || return 2
    eval "$("$build/stage/bin/psi4" --psiapi)" || return $?
    export PYTHONPATH="$build/stage/lib"
    "$CONDA_PREFIX/bin/python" -c \
        'import pathlib,psi4,sys; print(psi4.__file__,psi4.__version__); assert pathlib.Path(psi4.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())' \
        "$build/stage/lib/psi4" || return $?
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv || return $?
    nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H200 || return 2
    if [[ "$mode" == smoke ]]; then
        "$CONDA_PREFIX/bin/python" "$source_tree/devtools/benchmarks/build_receipt.py" \
            --source "$source_tree" --package "$build/stage/lib/psi4" \
            --built-commit "$built_commit" --output "$run_root/metadata/new-build.json" || return $?
        "$CONDA_PREFIX/bin/python" "$source_tree/devtools/benchmarks/build_receipt.py" \
            --source "$old_source" --package "$old_source/build_saptdft_cuest_head/stage/lib/psi4" \
            --built-commit "$old_commit" --output "$run_root/metadata/old-build.json" || return $?
        "$CONDA_PREFIX/bin/python" -m pytest -q \
            "$source_tree/tests/pytests/test_cuest_sad.py" "$source_tree/tests/pytests/test_cuest_jk.py" \
            "$source_tree/tests/pytests/test_scf_lifecycle.py" "$source_tree/tests/pytests/test_basis_parse_reuse.py" \
            --junitxml="$run_root/smoke-$SLURM_JOB_ID.xml" || return $?
        "$CONDA_PREFIX/bin/python" -c \
            'import sys,xml.etree.ElementTree as E; t=E.parse(sys.argv[1]); cases=[c for c in t.iter("testcase") if "test_sad_gpu_density_matches_cpu" in c.get("name","")]; assert len(cases)==4 and all(len(c)==0 for c in cases), "GPU SAD coverage missing or skipped"' \
            "$run_root/smoke-$SLURM_JOB_ID.xml" || return $?
        printf '%s\n' "$commit" > "$run_root/metadata/SMOKE_COMMIT"
    elif [[ "$mode" == campaign ]]; then
        test "$(head -n 1 "$run_root/metadata/SMOKE_COMMIT")" = "$commit" || return 2
        "$CONDA_PREFIX/bin/python" "$source_tree/devtools/benchmarks/lifecycle_campaign.py" \
            --output "$run_root/results-$SLURM_JOB_ID" \
            --old-source "$old_source" --old-commit "$old_commit" \
            --old-package "$old_source/build_saptdft_cuest_head/stage/lib/psi4" \
            --new-source "$source_tree" --new-commit "$commit" \
            --old-receipt "$run_root/metadata/old-build.json" \
            --new-receipt "$run_root/metadata/new-build.json" \
            --new-package "$build/stage/lib/psi4" --repeats 3 --threads 8 --memory "112 GiB"
    else
        echo "Unknown mode: $mode" >&2
        return 2
    fi
}
term_received=0
child=
on_term() {
    term_received=1
    [[ -n $child ]] && kill -TERM "$child" 2>/dev/null || true
}
on_exit() {
    status=$?
    (( term_received )) && status=143
    printf 'exit_code=%s\nterm_received=%s\nfinished=%s\n' \
        "$status" "$term_received" "$(date -Iseconds)" >> "$metadata.env"
    if (( status == 0 )); then touch "$metadata.COMPLETED"; else touch "$metadata.FAILED"; fi
}
trap on_term TERM
trap on_exit EXIT
payload &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
(( term_received )) && exit 143
exit "$status"
