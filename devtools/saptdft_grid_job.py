#!/usr/bin/env python3
"""Generate an immutable, TERM-safe Phoenix validation job and manifest."""

import argparse
import json
from pathlib import Path
import re
import shlex

PROJECT = "/storage/project/r-cs207-0/awallace43"
ENVIRONMENT = PROJECT + "/miniconda/envs/p4cuest_sapt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--local-output", required=True)
    parser.add_argument("--remote-output", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.commit):
        raise SystemExit("Expected a complete SHA-1 commit pin")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.branch):
        raise SystemExit("Use a filesystem-safe checkpoint branch")
    if not args.remote_output.startswith(PROJECT + "/"):
        raise SystemExit("Output must be an absolute path in the project allocation")
    local = Path(args.local_output).resolve()
    local.mkdir(parents=True, exist_ok=True)
    for file in ["grid.sbatch", "manifest.json"]:
        if (local / file).exists():
            raise SystemExit("Refusing to overwrite an immutable job artifact")
    root = PROJECT + "/gits/psi4." + args.branch
    quote = shlex.quote
    batch = f"""#!/usr/bin/env bash
#SBATCH --job-name=xc-grid-validation
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --signal=TERM@120
#SBATCH --output={args.remote_output}/logs/%x-%j.out
#SBATCH --error={args.remote_output}/logs/%x-%j.err
set -euo pipefail
umask 027
ROOT={quote(root)}
PIN={quote(args.commit)}
ENVIRONMENT={quote(ENVIRONMENT)}
RUN_DIR={quote(args.remote_output)}/"$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"/{{metadata,host,cuda}}
cd "$ROOT"
test "$(git rev-parse HEAD)" = "$PIN"
printf 'job_id=%s\\nhost=%s\\ncommit=%s\\nstarted_at=%s\\n' \\
    "$SLURM_JOB_ID" "$(hostname -f)" "$PIN" "$(date --iso-8601=seconds)" \\
    > "$RUN_DIR/metadata/runtime.env"
term_received=0
child=
on_term() {{
    term_received=1
    [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true
}}
on_exit() {{
    status=$?
    (( term_received )) && status=143
    printf 'exit_code=%s\\nterm_received=%s\\nfinished_at=%s\\n' \\
        "$status" "$term_received" "$(date --iso-8601=seconds)" >> "$RUN_DIR/metadata/runtime.env"
    if (( status == 0 )); then touch "$RUN_DIR/metadata/COMPLETED"
    else touch "$RUN_DIR/metadata/FAILED"; fi
}}
trap on_term TERM
trap on_exit EXIT
overall=0
for variant in host cuda; do
    srun bash "$ROOT/devtools/saptdft_grid_gpu_validate.sh" \\
        "$ROOT" "$ENVIRONMENT" "$PIN" "$RUN_DIR/$variant" "$variant" &
    child=$!
    set +e
    wait "$child"; status=$?
    while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
    set -e
    printf '%s=%s\\n' "$variant" "$status" >> "$RUN_DIR/metadata/variants.env"
    if (( term_received )); then exit 143; fi
    (( status == 0 )) || overall=1
    child=
done
exit "$overall"
"""
    (local / "grid.sbatch").write_text(batch)
    (local / "manifest.json").write_text(json.dumps(dict(
        cluster="phoenix", account="gts-cs207-chemx", partition="gpu-h200",
        qos="embers", walltime="04:00:00", cpus=8, memory="64G", gpu="h200:1",
        source=root, branch=args.branch, commit=args.commit,
        dependency_environment=ENVIRONMENT,
        read_only_dependencies=[
            PROJECT + "/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage",
            PROJECT + "/software/libxc-7.1.2-cuda",
            PROJECT + "/software/toolchains",
        ],
        fresh_builds=["build_grid_host", "build_grid_cuda"], output=args.remote_output,
        command="devtools/saptdft_grid_gpu_validate.sh (host, then cuda)",
        acceptance=[
            "Exact clean commit pin; staged package and linked dependencies verified",
            "34 regressions per variant, no skips/errors/failures",
            "XC microtimer artifact includes cuEST and LibXC API calls",
            "Nine fixed-shift and nine iterative-GRAC water cases per variant",
            "Four full-SAPT water cases per variant include CPKS induction and dispersion",
            "Switching error separated from final-grid quadrature error; tiny-grid failures retained",
        ],
        timing_caveat="Diagnostic synchronized profiling is separate from unprofiled numerical sweeps",
        source_sync="git push/fetch only; dependency environments/builds are never modified",
        preemption="TERM forwarded to srun; no COMPLETED marker on signalled or nonzero exit",
    ), indent=2) + "\n")
    print(local / "grid.sbatch")


if __name__ == "__main__":
    main()
