#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${DISP0_INPUT_SOURCE:?}" "${DISP0_INPUT_PIN:?}" "${DISP0_INPUT_RUN:?}" "${SLURM_JOB_ID:?}"
P=/storage/project/r-cs207-0/awallace43
PY="$P/miniconda/bin/python"
[[ "$(git -C "$DISP0_INPUT_SOURCE" rev-parse HEAD)" == "$DISP0_INPUT_PIN" ]]
[[ -z "$(git -C "$DISP0_INPUT_SOURCE" status --porcelain)" ]]
[[ "$DISP0_INPUT_RUN" == "$P/runs/saptdft-cuest-disp0/input-prep-$DISP0_INPUT_PIN-$SLURM_JOB_ID" ]]
mkdir -m 0700 "$DISP0_INPUT_RUN"
export TMPDIR="/storage/scratch1/3/awallace43/disp0-input-$SLURM_JOB_ID"
mkdir -m 0700 "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$P/miniconda/lib"
unset PYTHONPATH PYTHONHOME
cd "$DISP0_INPUT_SOURCE"
printf 'sourcepin=%s jobid=%s host=%s approval=%s\n' \
    "$DISP0_INPUT_PIN" "$SLURM_JOB_ID" "$(hostname)" "${DISP0_INPUT_APPROVAL:?}"
# The helper verifies ACTUAL remote0.19 streaming-frame behavior before the
# first package download. No unbounded decoder or version fallback.
timeout --signal=TERM --kill-after=5s 110s "$PY" -B \
    devtools/disp0/provision_sdk_inputs.py \
    --source "$P/software/toolchains/cuda13" \
    --destination "$DISP0_INPUT_RUN/private-inputs" \
    --vendor-environment "$P/miniconda/envs/p4cuest_sapt" \
    --lock "$P/runs/saptdft-cuest-disp0/build-587e172a33afaadb00eb7f574a5089533f7fc755-14196330/environment-explicit.txt"
sha256sum "$DISP0_INPUT_RUN/private-inputs/sdk-manifest.json" \
    "$DISP0_INPUT_RUN/private-inputs/provision-receipt.json"
du -sb "$DISP0_INPUT_RUN"
