#!/usr/bin/env bash
# Focused correctness check, not a CPU scaling campaign.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
env="${DISP0_ENV:-$HOME/miniconda3/envs/p4_einv2}"
build="${DISP0_BUILD_DIR:-$root/build_disp0_cpu}"
run="$(mktemp -d "$build/reference-check-XXXXXXXX")"
export PATH="$env/bin:$PATH"
export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${DISP0_TEST_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TMPDIR="$run/scratch" SCRATCH="$run/scratch" PSI_SCRATCH="$run/scratch"
mkdir -p "$PSI_SCRATCH"
# --psiapi appends $PYTHONPATH literally; define it under nounset even when
# the launching shell has no Python overlay (the normal isolated case).
export PYTHONPATH="${PYTHONPATH:-}"
eval "$("$build/stage/bin/psi4" --psiapi)"
export DISP0_ROOT="$root"
export DISP0_EXPECTED_STAGE="$build/stage"
printf 'source=%s\nrun=%s\n' "$(git rev-parse HEAD)" "$run"
# Python stdin puts cwd before PYTHONPATH. Do not let the source-tree psi4/
# package shadow the dedicated native stage selected above.
cd "$run"
"$env/bin/python" - <<'PY'
import os
from pathlib import Path
import psi4
print("psi4:", psi4.__file__, psi4.__version__)
print("core:", psi4.core.__file__)
assert Path(psi4.__file__).resolve().is_relative_to(Path(os.environ["DISP0_EXPECTED_STAGE"]).resolve())
assert Path(psi4.core.__file__).resolve().is_relative_to(Path(os.environ["DISP0_EXPECTED_STAGE"]).resolve())
assert psi4.core.get_global_option("FISAPT_FDISP_ALGORITHM") == "BLOCKED"
PY
if [[ $# == 0 ]]; then
    set -- "$root/tests/pytests/test_fisapt.py" -k test_fsapt_fdisp_blocked_matches_pair
fi
"$env/bin/python" -m pytest "$@" -v \
    -o junit_family=legacy \
    --basetemp="$run/pytest" --junitxml="$run/results.xml"
