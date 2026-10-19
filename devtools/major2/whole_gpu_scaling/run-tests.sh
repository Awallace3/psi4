#!/usr/bin/env bash
# Foreground host tests only. Never imports Psi4 or invokes a device/scheduler.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
H=$(dirname "$here")
A=/home/awallace43/gits/psi4__worktrees/saptdft-cuest/.pi/major2-gpu-optimization-20261017
export MAJOR2_TEST_TMP="$A/whole-gpu-scaling/test-tmp"
export MAJOR2_PAIR_JSON="$A/fitting-test-repair/artifact-pair.json"
export MAJOR2_REPEATABILITY_COLLECTION="$A/p157-repeatability/collection.json"
export MAJOR2_FIXED_MANIFEST="$A/workspace-control/bundle/manifest.json"
export PYTHONDONTWRITEBYTECODE=1
PY=/home/awallace43/miniconda3/envs/p4qca/bin/python
mkdir -p "$MAJOR2_TEST_TMP"
test -r "$MAJOR2_PAIR_JSON"
test -r "$MAJOR2_REPEATABILITY_COLLECTION"
test -r "$MAJOR2_FIXED_MANIFEST"
cd "$H"
"$PY" -B -m unittest -v \
  test_harness test_jkexception test_fitcase test_constructor_native test_nonc1 \
  test_repeatability test_fixed_jk test_fixed_jk_warning workspace_checks
cd "$H/native_plan"
"$PY" -B -m unittest -v test_native_plan test_repair test_env_repair
cd "$H/planreuse_speed"
"$PY" -B -m unittest -v test_speed
cd "$here"
"$PY" -B -m unittest -v test_whole
"$PY" -B - <<'PY'
import ast
from pathlib import Path
from provenance import verify_runner
import config as s
for path in sorted(Path(".").glob("*.py")):
    ast.parse(path.read_text(), filename=str(path))
print("Exact runner delta:",verify_runner(s.ROOT)["owned_sha256"])
PY
bash -n supervise.sh parent-finalize.sh run-tests.sh
git diff --check
echo "295 inherited checks unchanged plus new host checks; NO new GPU results"
