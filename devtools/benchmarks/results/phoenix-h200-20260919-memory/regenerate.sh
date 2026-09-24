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
# the only variable between the paired and control tables is the build.
#
# Two things changed after the first pass, and both left their earlier trees in
# place rather than replacing them:
#
#   * libcuest moved 0.2.1.2 -> 0.2.2.2. The bump needs no recompilation -- same
#     soname, and the only header change is CUEST_VER_PATCH -- so M4 and M5 ran
#     the *byte-identical* core.so that M1 and M3 ran, against a different
#     shared object. That makes M1 -> M4 a one-variable measurement of the
#     library itself, which is what cuest-delta.md reports.
#   * protein157 now runs, as one CPU run and one GPU run. It is the largest case
#     in the suite, and its CPU arm takes ~3.5h, which embers preempted in five
#     of six attempts. SAPT(DFT) has no checkpoint, so the run cannot be split.
#     The one CPU run that finished is P3 job 13395715 (cpu-3). Its GPU partner
#     is gpu-3 from P1 job 13429862. That tree is marked FAILED only because its
#     own CPU case was preempted; all three of its GPU cases returned rc=0. The
#     pair therefore crosses two allocations. host-speed.md prints both canaries.
#     protein157 is reported in its own tables (PROTEIN157*), never pooled into
#     the six-case medians, and is absent from premerge-delta.md because no
#     control-build arm was run for it.
set -euo pipefail
RAW=${1:?usage: regenerate.sh /path/to/memory-campaign-20260919}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$HERE/../..

# Exactly one *usable* tree per job prefix. Globbing rather than hardcoding job
# ids keeps a resubmitted slice from silently going unused, and a count that is
# not one is an error rather than a skipped table: every table below is a claim
# about a specific set of measurements.
#
# A prefix can have more than one tree, because a slice that is preempted or
# lands on a bad host gets resubmitted and the earlier tree is kept rather than
# deleted. Two markers written by the job itself decide which trees are
# candidates, so the choice is made from what the run recorded and not from the
# job id or the mtime:
#
#   metadata/COMPLETED     every case in the tree returned zero. Absent on a
#                          tree that was killed partway -- including job
#                          13376151, which was preempted with its scratch
#                          directory unmounted under it, failed its last six
#                          cases in under a second each, and still ran its loop
#                          to the end.
#   metadata/DEGRADED-HOST the host-speed gate measured this allocation below
#                          the campaign's floor. Job 13376151 again: its cores
#                          sat at the 800 MHz floor of a 2800 MHz part, so its
#                          wall times are ~3.4x slow and comparable with
#                          nothing. Energies and VmHWM would have been fine;
#                          this directory is mostly about wall times, so the
#                          whole tree is dropped rather than half-used.
#
# Dropping is announced. A tree that cost hours of allocation should not vanish
# from the analysis without saying so on the way past.
one_tree() {
  local prefix=$1 d found=()
  shopt -s nullglob
  for d in "$RAW"/"$prefix"-job*/; do
    d=${d%/}
    if [ -e "$d/metadata/DEGRADED-HOST" ]; then
      echo "note: skipping $(basename "$d"): host-speed gate marked it degraded" >&2
      sed -n 's/^/      /p' "$d/metadata/DEGRADED-HOST" >&2
      continue
    fi
    if [ ! -e "$d/metadata/COMPLETED" ]; then
      echo "note: skipping $(basename "$d"): no metadata/COMPLETED (run did not finish clean)" >&2
      continue
    fi
    found+=("$d/results")
  done
  shopt -u nullglob
  if [ "${#found[@]}" -ne 1 ]; then
    echo "expected exactly one usable $prefix tree under $RAW, found ${#found[@]}" >&2
    exit 1
  fi
  printf '%s' "${found[0]}"
}

M4=$(one_tree M4-core6-h200-cuest022)
M5=$(one_tree M5-premerge-core6-h200-cuest022)
P3=$(one_tree P3-protein157-h200)

# Within every case in the paired tree, both arms ran in one allocation.
python "$TOOLS/merge_case_trees.py" "$M4" --output "$RAW/merged-paired"

# protein157: one CPU run and one GPU run, curated by hand instead of by
# one_tree, because the GPU source tree is FAILED at tree level. The GPU case is
# checked individually instead: its case-status line must say rc=0 and its
# result.json must say ok. Both are relinked as repeat 1, because the summarizer
# pairs by repeat index and assumes one repeat count per campaign. SOURCES.txt
# records the original names.
P157_GPU_TREE=$RAW/P1-protein157-h200-job13429862
P157_GPU_CASE='protein157-6-31+g**-gpu-3'
P157_CPU_CASE='protein157-6-31+g**-cpu-3'
[ -e "$P157_GPU_TREE/metadata/DEGRADED-HOST" ] && { echo "protein157 GPU tree is degraded" >&2; exit 1; }
grep -qxF "$P157_GPU_CASE rc=0" <(awk '{print $1, $2}' "$P157_GPU_TREE/metadata/case-status.txt") \
  || { echo "protein157 GPU case did not return rc=0" >&2; exit 1; }
python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["ok"] else 1)' \
  "$P157_GPU_TREE/results/$P157_GPU_CASE/result.json" \
  || { echo "protein157 GPU case reports ok=false" >&2; exit 1; }
P157=$RAW/protein157-single-pair
mkdir -p "$P157"
ln -sfn "$P3/$P157_CPU_CASE" "$P157/protein157-6-31+g**-cpu-1"
ln -sfn "$P157_GPU_TREE/results/$P157_GPU_CASE" "$P157/protein157-6-31+g**-gpu-1"
cat > "$P157/SOURCES.txt" <<SRC
protein157 single pair: one CPU run, one GPU run, from two allocations.
cpu-1 <- $P3/$P157_CPU_CASE  (job 13395715, tree COMPLETED)
gpu-1 <- $P157_GPU_TREE/results/$P157_GPU_CASE  (job 13429862, tree FAILED: its cpu-1 was preempted; the gpu case rc=0)
SRC

# M5 is the control arm: the merge's own first parent, d91b5f8e81, built from
# the same worktree and measured by a byte-identical driver.
python "$TOOLS/merge_case_trees.py" "$M5" --output "$RAW/merged-premerge"

# The same six cases on the same two binaries before the cuEST bump.
python "$TOOLS/merge_case_trees.py" "$(one_tree M1-core6-h200)" \
  --output "$RAW/merged-paired-cuest0212"
python "$TOOLS/merge_case_trees.py" "$(one_tree M3-premerge-core6-h200)" \
  --output "$RAW/merged-premerge-cuest0212"

A=$RAW/merged-paired                           # paired CPU/GPU, H200, 8 threads, libcuest 0.2.2.2
B=$RAW/merged-premerge                         # same, on the pre-merge control build
D=$RAW/merged-paired-cuest0212                 # same build as A, libcuest 0.2.1.2
E=$RAW/merged-premerge-cuest0212               # same build as B, libcuest 0.2.1.2
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

# protein157, n=1 per arm: timings, memory, accuracy and phase attribution.
python "$TOOLS/case_dirs_to_campaign.py" "$P157" --force --expect protein157:6-31+g**:1
python "$TOOLS/summarize_saptdft_cuest.py" "$P157" --output "$HERE/protein157" || true
python "$TOOLS/iterative_accuracy.py" "$P157" --output "$HERE/accuracy-protein157.json" \
  > "$HERE/accuracy-protein157.md"
python "$TOOLS/speedup_attribution.py" "$P157" --output "$HERE/attribution-protein157.json" \
  > "$HERE/attribution-protein157.md"

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

# What the cuEST upgrade alone did. Same core.so on both sides, same driver,
# same partition; the only difference is which libcuest.so.0 the loader found.
# No --require-identical-numerics here, and not as an oversight: a GPU library
# is entitled to change its arithmetic between releases, and whether it did is
# the question this table is asked to answer rather than a precondition it
# should abort on.
python "$TOOLS/case_dirs_to_campaign.py" "$D" --force \
  --expect water:cc-pvdz:3 --expect water:aug-cc-pvdz:3 \
  --expect benzene:cc-pvdz:3 --expect benzene:aug-cc-pvdz:3 \
  --expect peptide:6-31+g**:3 --expect nanotube:6-31+g**:3
python "$TOOLS/summarize_saptdft_cuest.py" "$D" --output "$HERE/paired-cuest0212" || true
python "$TOOLS/build_delta.py" "$HERE/paired-cuest0212/summary.json" "$HERE/paired/summary.json" \
  --control-label "libcuest 0.2.1.2" --treatment-label "libcuest 0.2.2.2" \
  --output "$HERE/cuest-delta.json" > "$HERE/cuest-delta.md"

# The same library swap on the control build, M3 -> M5. This is the table that
# shows the first pass's M1-vs-M3 GPU gap was M3, not the merge: the control
# build's GPU arm moves by ~25% here while the treatment's (cuest-delta.md)
# does not move at all, and nothing in the library bump distinguishes the two.
python "$TOOLS/case_dirs_to_campaign.py" "$E" --force \
  --expect water:cc-pvdz:3 --expect water:aug-cc-pvdz:3 \
  --expect benzene:cc-pvdz:3 --expect benzene:aug-cc-pvdz:3 \
  --expect peptide:6-31+g**:3 --expect nanotube:6-31+g**:3
python "$TOOLS/summarize_saptdft_cuest.py" "$E" --output "$HERE/premerge-cuest0212" || true
python "$TOOLS/build_delta.py" "$HERE/premerge-cuest0212/summary.json" "$HERE/premerge/summary.json" \
  --control-label "pre-merge, libcuest 0.2.1.2 (M3)" --treatment-label "pre-merge, libcuest 0.2.2.2 (M5)" \
  --output "$HERE/premerge-cuest-delta.json" > "$HERE/premerge-cuest-delta.md"

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

# Which host each tree actually ran on. Every tree here carries a start and end
# canary, so unlike the 2026-09-10 directory this verdict is certified -- and it
# is what licenses pairing protein157 across two allocations above.
python "$TOOLS/host_speed.py" "$RAW"/*/results --output "$HERE/host-speed.json" \
  > "$HERE/host-speed.md" || true

# Whether an out-of-tolerance component is arithmetic or a different SCF solution.
python "$TOOLS/iterative_accuracy.py" "$A" --output "$HERE/accuracy.json" > "$HERE/accuracy.md"

# CPU baseline normalization toward NVIDIA's 56 cores: total wall, and DF-K on its
# own, because a DF-K claim must be normalized by DF-K's scaling, not the method's.
# This tree is from the first pass and predates the cuEST bump; it is CPU-only,
# so the GPU library it was linked against cannot have moved a number in it.
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
  --block "CUESTDELTA=$HERE/cuest-delta.md" \
  --block "PREMERGECUESTDELTA=$HERE/premerge-cuest-delta.md" \
  --block "THREADTOTAL=$HERE/thread-scaling-total.md" \
  --block "THREADDFK=$HERE/thread-scaling-dfk.md" \
  --block "HOSTSPEED=$HERE/host-speed.md" \
  --block "PROTEIN157=$HERE/protein157/summary.md" \
  --block "PROTEIN157ACC=$HERE/accuracy-protein157.md" \
  --block "PROTEIN157ATTR=$HERE/attribution-protein157.md"
