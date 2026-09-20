#!/bin/bash
# Rebuild every derived table in this directory from the raw campaign trees.
#
# The raw trees are the per-case directories written by
# `saptdft_cuest_grac.py --case`, one per measurement, each with its own
# result.json, psi4.out, and timer.dat. They are not in the repository: they run
# to hundreds of megabytes. RAW must point at a directory holding the job trees
# named below; see PROVENANCE.md for where they came from and how to re-run them.
#
# This directory repeats the 2026-09-10 campaign on a build that carries the
# process memory ledger (MemoryClaim), the cost-based collocation cache fill,
# and the malloc_trim release. The protocol is otherwise identical to job A3, so
# the only variable between the two directories is the build. protein157 is
# absent here on purpose -- it was deferred until the memory changes are
# confirmed, and it never ran in A3 either.
set -euo pipefail
RAW=${1:?usage: regenerate.sh /path/to/memory-campaign-20260919}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$HERE/../..

# The paired table is job M1 and nothing else. One job, one host, 6 cases x 3
# repeats, canary certified at both ends -- so the merge has a single input and
# needs no override.
python "$TOOLS/merge_case_trees.py" "$RAW/M1-core6-h200-job13358747/results" \
  --output "$RAW/merged-paired"

# M3 is the control arm: the merge's own first parent, d91b5f8e81, built from
# the same worktree and measured by a byte-identical driver. The job id is not
# hardcoded because the control was built after the treatment; there must be
# exactly one such tree, and a missing one is an error rather than a skipped
# table, since the README's central claim is the M1-vs-M3 difference.
shopt -s nullglob
M3=("$RAW"/M3-premerge-core6-h200-job*/results)
shopt -u nullglob
if [ "${#M3[@]}" -ne 1 ]; then
  echo "expected exactly one M3 control tree under $RAW, found ${#M3[@]}" >&2
  exit 1
fi
python "$TOOLS/merge_case_trees.py" "${M3[0]}" --output "$RAW/merged-premerge"

A=$RAW/merged-paired                           # paired CPU/GPU, one H200 node, 8 threads
B=$RAW/merged-premerge                         # same, on the pre-merge control build
C=$RAW/M2-cpu24-core6-job13358750/results      # CPU-only thread scaling, 8 vs 24, one node

# Paired timings, host/device memory, and backend accuracy. The manifest is
# rebuilt from the case directories because an external launcher drove the
# campaign, so no campaign.json was written; --expect makes a case that never ran
# a failure rather than a silent absence.
python "$TOOLS/case_dirs_to_campaign.py" "$A" --force \
  --expect water:cc-pvdz:3 --expect water:aug-cc-pvdz:3 \
  --expect benzene:cc-pvdz:3 --expect benzene:aug-cc-pvdz:3 \
  --expect peptide:6-31+g**:3 --expect nanotube:6-31+g**:3
python "$TOOLS/summarize_saptdft_cuest.py" "$A" --output "$HERE/paired" || true

# The control arm, reduced by the same summarizer so the two are comparable term
# by term, then differenced. --require-identical-numerics is deliberate: a memory
# change that moved a CPU-vs-GPU delta would not be a memory change, and this is
# the cheapest place to find that out.
python "$TOOLS/case_dirs_to_campaign.py" "$B" --force \
  --expect water:cc-pvdz:3 --expect water:aug-cc-pvdz:3 \
  --expect benzene:cc-pvdz:3 --expect benzene:aug-cc-pvdz:3 \
  --expect peptide:6-31+g**:3 --expect nanotube:6-31+g**:3
python "$TOOLS/summarize_saptdft_cuest.py" "$B" --output "$HERE/premerge" || true
python "$TOOLS/build_delta.py" "$HERE/premerge/summary.json" "$HERE/paired/summary.json" \
  --control-label "pre-merge d91b5f8e81" --treatment-label "merged ee6161a3b6" \
  --require-identical-numerics \
  --output "$HERE/premerge-delta.json" > "$HERE/premerge-delta.md"

# Where the GPU saving actually comes from, and how much of it DF-K could ever
# explain on its own.
python "$TOOLS/speedup_attribution.py" "$A" --output "$HERE/attribution.json" > "$HERE/attribution.md"

# The same decomposition on the control arm. Without it the only way to ask what
# the merge did to each phase is to reach across campaigns to 2026-09-10, which
# is a different build and a different driver; with it the question is answered
# inside the A/B.
python "$TOOLS/speedup_attribution.py" "$B" --output "$HERE/attribution-premerge.json" \
  > "$HERE/attribution-premerge.md"

# DF-K in effective TFLOPS, computed the way NVIDIA's figures 3-4 are: dense
# rectangular-DGEMM FLOPs for the kernel divided by the kernel's own wall time.
python "$TOOLS/dfk_effective_tflops.py" "$A" --output "$HERE/dfk-tflops.json" > "$HERE/dfk-tflops.md"

# What automatic GRAC costs, measured inside each job from its own phase timers.
python "$TOOLS/grac_cost.py" "$A" --output "$HERE/grac-cost.json" > "$HERE/grac-cost.md"

# Which host each tree actually ran on. Both trees here carry a start and end
# canary, so unlike the 2026-09-10 directory this verdict is certified.
python "$TOOLS/host_speed.py" "$RAW"/*/results --output "$HERE/host-speed.json" \
  > "$HERE/host-speed.md" || true

# Whether an out-of-tolerance component is arithmetic or a different SCF solution.
python "$TOOLS/iterative_accuracy.py" "$A" --output "$HERE/accuracy.json" > "$HERE/accuracy.md"

# CPU baseline normalization toward NVIDIA's 56 cores: total wall, and DF-K on its
# own, because a DF-K claim must be normalized by DF-K's scaling, not the method's.
python "$TOOLS/thread_scaling.py" "$C" --output "$HERE/thread-scaling-total.json" \
  > "$HERE/thread-scaling-total.md"
python "$TOOLS/thread_scaling.py" "$C" --timer 'JK: JK' \
  --output "$HERE/thread-scaling-dfk.json" > "$HERE/thread-scaling-dfk.md"

# The report itself. README.md is generated: the prose lives in
# README.template.md and the tables are substituted, so a re-run cannot leave a
# stale number in the narrative while the generated tables move.
python "$HERE/fixed_vs_iterative.py" "$HERE/paired/summary.json" \
  --grac-cost "$HERE/grac-cost.json" > "$HERE/fixed-vs-iterative.md"
python "$TOOLS/splice.py" "$HERE/README.template.md" "$HERE/README.md" \
  --block "PAIRED=$HERE/paired/summary.md" \
  --block "GRACCOST=$HERE/grac-cost.md" \
  --block "FIXEDVSITER=$HERE/fixed-vs-iterative.md" \
  --block "ATTRIBUTION=$HERE/attribution.md" \
  --block "ATTRIBUTIONPRE=$HERE/attribution-premerge.md" \
  --block "TFLOPS=$HERE/dfk-tflops.md" \
  --block "ACCURACY=$HERE/accuracy.md" \
  --block "PREMERGE=$HERE/premerge-delta.md" \
  --block "THREADTOTAL=$HERE/thread-scaling-total.md" \
  --block "THREADDFK=$HERE/thread-scaling-dfk.md" \
  --block "HOSTSPEED=$HERE/host-speed.md"
