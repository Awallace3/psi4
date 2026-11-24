#!/bin/bash
#SBATCH -A gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --constraint=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -J cdk2-fsapt
set -euo pipefail

# sbatch cdk2_h200.sbatch INPUT_DIR OUTPUT_ROOT HF|PBE0
input=$(realpath -e "${1:?prepared ligand directory}")
output_root=${2:?absolute results directory}
functional=${3:?HF or PBE0}
[[ $output_root == /* && ( $functional == HF || $functional == PBE0 ) ]]
project=/storage/project/r-cs207-0/awallace43
campaign=$project/runs/fsapt-d4i-0301-20261117
stage=$campaign/build-15825650/obj/stage
cuda=$campaign/build-15797561/cuda135
xc=$project/software/libxc-7.1.2-cuda-sm80-89-90-120/lib64/libxc.so.15
envp=$project/miniconda/envs/p4cuest_sapt
source_tree=$project/gits/psi4.fsapt-reuse-decab0ab58
revision=decab0ab58acf5bea9b30cd5af7719cc3f79959d
[[ $(cat "$campaign/build-15825650/BUILT_FROM") == "$revision" ]]
[[ $(git -C "$source_tree" rev-parse HEAD) == "$revision" ]]
model=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $model == *H200* ]]
run=$output_root/$(basename "$input")/$functional/$SLURM_JOB_ID
mkdir -p "$(dirname "$run")"
mkdir "$run"
cd "$run"
sha256sum "$input"/{dimer.geometry,field.json,groups.json,meta.json} > inputs.sha256
local_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
[[ "$local_base" != /storage/* && -d "$local_base" ]] || local_base=/tmp
export TMPDIR="$local_base/cdk2-fsapt-$SLURM_JOB_ID"
mkdir -m 700 "$TMPDIR"
[[ "$(stat -f -c %T "$TMPDIR")" =~ ^(xfs|ext2/ext3|ext4|tmpfs|btrfs)$ ]]
export PSI_SCRATCH="$TMPDIR" SCRATCH="$TMPDIR"
export PATH="$envp/bin:$cuda/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$cuda/lib:$cuda/targets/x86_64-linux/lib:$(dirname "$xc"):$envp/lib"
export LD_PRELOAD="$xc"
for lib in libcuest.so.0 libcudart.so.13 libnvrtc.so.13 libcublas.so.13 libcublasLt.so.13 libcusparse.so.12 libcusolver.so.12 libnvJitLink.so.13; do
    LD_PRELOAD+=" $cuda/lib/$lib"
done
export PYTHONPATH="$stage/lib:$envp/lib/python3.13/site-packages" PSIDATADIR="$stage/share/psi4" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=1 PSI4_CUEST_SCF_TIMERS=1
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total --format=csv -lms 200 > "$run/memory.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
cd "$TMPDIR"
"$envp/bin/python" -P "$source_tree/devtools/fsapt_integration/d4i_timing.py" \
    --stage "$stage" --runtime "$cuda" --libxc "$xc" --arm gpu --threads "$SLURM_CPUS_PER_TASK" --memory 55GB \
    --functional "$functional" --fsapt --fragment-integrals AUTO --geometry "$input/dimer.geometry" \
    --field "$input/field.json" --groups "$input/groups.json" --output "$run/energy"
cd "$run"
"$envp/bin/python" - <<'PY'
import json
import math
from pathlib import Path

energy = Path("energy")
r = json.loads((energy / "result.json").read_text())
s = r["fragment_sums_hartree"]
sums = dict(ELST=s["ELST_AB"], EXCH=s["EXCH_AB"], IND=s["INDAB_AB"] + s["INDBA_AB"], DISP=s["EMPIRICAL_DISP"])
assert r["reused_scalar_plan"] is True and r["fragment_plan_creations"] == 0
for term, value in sums.items():
    assert math.isclose(value, r["scalar_hartree"][term], rel_tol=0, abs_tol=1e-6), term
assert (energy / "fragments.npz").stat().st_size > 0
groups = json.loads((energy / "groups-kcal-mol.json").read_text())
grand_total, = [value for a, b, value in zip(groups["Frag1"], groups["Frag2"], groups["Total"], strict=True)
                if a == "All" and b == "All"]
assert math.isclose(grand_total / 627.5094740631, r["energy_hartree"], rel_tol=0, abs_tol=1e-6)
PY
# On success Psi4 has closed its scratch files. Preserve failed scratch for diagnosis.
cd "$run"
rm -rf -- "$TMPDIR"
touch COMPLETED
