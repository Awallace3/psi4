#!/bin/bash
# Rebuild every derived table in this directory from the raw campaign trees.
#
# The raw trees are the per-case directories written by
# `saptdft_cuest_grac.py --case`, one per measurement, each with its own
# result.json, psi4.out, and timer.dat. They are not in the repository: they run
# to hundreds of megabytes. RAW must point at a directory holding the job trees
# named below; see PROVENANCE.md for where they came from and how to re-run them.
set -euo pipefail
RAW=${1:?usage: regenerate.sh /path/to/iterative-campaign}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$HERE/../..

# Job A was preempted with three nanotube cases outstanding; job A2 reran exactly
# those. They are two job trees holding one campaign, so present them as one
# directory of symlinks rather than copying either. The merge refuses if both
# trees claim a completed copy of the same case.
python "$TOOLS/merge_case_trees.py" \
  "$RAW/A-core6-h200-job13060539/results" \
  "$RAW/A2-nanotube-h200-job13065746/results" \
  --output "$RAW/merged-paired"

A=$RAW/merged-paired                      # paired CPU/GPU, one H200 node, 8 threads
C=$RAW/C-cpu24-core6-job13061073/results  # CPU-only thread scaling, 8 vs 24, one node

# Paired timings and backend accuracy. The manifest is rebuilt from the case
# directories because an external launcher drove the campaign, so no campaign.json
# was written; --expect makes a case that never ran a failure rather than a silent
# absence.
python "$TOOLS/case_dirs_to_campaign.py" "$A" --force \
  --expect water:cc-pvdz:3 --expect water:aug-cc-pvdz:3 \
  --expect benzene:cc-pvdz:3 --expect benzene:aug-cc-pvdz:3 \
  --expect peptide:6-31+g**:3 --expect nanotube:6-31+g**:3
python "$TOOLS/summarize_saptdft_cuest.py" "$A" --output "$HERE/paired" || true

# Where the GPU saving actually comes from, and how much of it DF-K could ever
# explain on its own.
python "$TOOLS/speedup_attribution.py" "$A" --output "$HERE/attribution.json" > "$HERE/attribution.md"

# DF-K in effective TFLOPS, computed the way NVIDIA's figures 3-4 are: dense
# rectangular-DGEMM FLOPs for the kernel divided by the kernel's own wall time.
python "$TOOLS/dfk_effective_tflops.py" "$A" --output "$HERE/dfk-tflops.json" > "$HERE/dfk-tflops.md"

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
python "$HERE/fixed_vs_iterative.py" "$HERE/paired/summary.json" > "$HERE/fixed-vs-iterative.md"
python "$TOOLS/splice.py" "$HERE/README.template.md" "$HERE/README.md" \
  --block "PAIRED=$HERE/paired/summary.md" \
  --block "FIXEDVSITER=$HERE/fixed-vs-iterative.md" \
  --block "ATTRIBUTION=$HERE/attribution.md" \
  --block "TFLOPS=$HERE/dfk-tflops.md" \
  --block "ACCURACY=$HERE/accuracy.md"
