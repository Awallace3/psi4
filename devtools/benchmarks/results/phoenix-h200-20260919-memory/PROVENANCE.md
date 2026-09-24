# Provenance — Phoenix H200 memory campaign, 2026-09-19

Everything needed to decide whether a number in [`README.md`](README.md) is
attributable, and everything needed to run the campaign again.

## What question the campaign was built to answer

Whether merging the process-memory work from `saptdft_ein_fi_option_d4` into
`saptdft_cuest` changes SAPT(DFT) wall time or memory, and by how much.

That question has exactly one honest control, and it is *not* the previous
campaign. It is the merge commit's own first parent, so that the two arms differ
in the Psi4 binary and in nothing else — same worktree layout, same compiler and
flags, same driver file by sha256, same geometries by sha256, same partition,
same thread count, same GPU model, same run directory structure. The reported
arms are M4 (treatment) and M5 (control), both against libcuest 0.2.2.2. M1 and
M3 are the same two builds against libcuest 0.2.1.2. They are kept as the
controls of the two library-swap tables. M2 is a separate CPU-only measurement
that exists only to normalize the CPU baseline for thread count and is never
differenced against anything.

## libcuest

M1/M3 ran against libcuest 0.2.1.2. M4/M5/P* ran against **0.2.2.2** (`cuest
0.2.2.2` in `metadata/conda-packages.json`). The bump keeps the soname and
changes only `CUEST_VER_PATCH` in the headers, so nothing was rebuilt: M4/M5
load the byte-identical `core*.so` pinned below. `common.inc` asserts the
version and checks that the linked `libcuest.so` resolves inside the campaign
environment.

## Builds

| Arm | Branch | Commit | Build directory | `core*.so` sha256 |
|---|---|---|---|---|
| Treatment | `saptdft_cuest_benchmarks` | `ee6161a3b6` | `psi4.saptdft_cuest_benchmarks/build_saptdft_cuest_benchmarks` | `f219602e8cf337b9094fa062544b9072da612757ac7fbaa5e404226e3915b862` |
| Control | `premerge_d91b5f8e81` | `d91b5f8e81` | `psi4.premerge_d91b5f8e81/build_premerge_d91b5f8e81` | `b90ac1366e7bd819358838fba879b1c518827e65388496abbaca1ad5fe69f6e1` |

`d91b5f8e81` is `git rev-parse ee6161a3b6^` through the merge — the first parent
of `3e107987f7`, which `ee6161a3b6` carries. `ee6161a3b6` adds to it the merge of
`saptdft_ein_fi_option_d4` plus the benchmark harness; nothing else.

Both trees resolve `Einsums_DIR` to their own `build_*/stage/share/cmake/Einsums`,
and `git diff d91b5f8e81 ee6161a3b6 -- external/upstream/einsums CMakeLists.txt`
is empty, so the two builds carry the same Einsums.

**How each arm proves it is the arm it claims to be.** The merge adds the Python
bindings `psi4.core.memory_committed` and `psi4.core.release_freed_memory`. Each
job asserts on them before running anything, in opposite directions: the
treatment requires both to exist, the control requires both to be absent. The
build job additionally greps the shared object for those symbols and for the
cuEST GRAC marker, and `pin_core_sha_premerge.sh` refuses to pin a control
hash that either exports `memory_committed` or equals the treatment's hash. A
binary mix-up would have to defeat all three.

The harness is deliberately *not* taken from each arm's own checkout — the
control predates `devtools/benchmarks` and has no harness to take. Both arms copy
`saptdft_cuest_grac.py`, `process_memory.py`, and
`saptdft_suite_geometries.json` from the one benchmarks worktree, and the driver
runs unmodified on the pre-merge binary because it reads the ledger through
`getattr(psi4.core, ..., None)`. On the control those columns are simply `null`,
which the M3 job checks for explicitly. The `peak_rss_mib` column is the kernel's
`VmHWM` and does not depend on the ledger at all, so it is comparable across arms.

## Jobs

All on `--qos=embers`, account `gts-cs207-chemx`. No inferno job was submitted
and no inferno approval exists.

| Job | Arm | Partition | Node | Elapsed | Exit | Cases |
|---|---|---|---|---:|---:|---:|
| 13356149 | harness build (treatment) | cpu-small | — | — | 0 | — |
| 13358747 | M1 paired CPU/GPU, 8 threads | gpu-h200 | `atl1-1-02-014-9-0` | 00:37:23 | 0 | 36 |
| 13358750 | M2 CPU-only, 8 vs 24 threads | cpu-small | `atl1-1-02-008-2-2` | 00:41:38 | 0 | 22 |
| 13367034 | control build | cpu-small | `atl1-1-02-006-14-1` | 00:16:30 | 0 | — |
| 13367763 | M3 paired CPU/GPU, 8 threads | gpu-h200 | `atl1-1-02-012-9-0` | 00:40:00 | 0 | 36 |
| 13395711 | M4 paired CPU/GPU, 8 threads, libcuest 0.2.2.2 | gpu-h200 | `atl1-1-02-012-2-0` | 00:39:13 | 0 | 36 |
| 13395712 | M5 control paired CPU/GPU, libcuest 0.2.2.2 | gpu-h200 | `atl1-1-03-019-2-0` | 00:37:53 | 0 | 36 |
| 13395715 | P3 protein157 CPU | gpu-h200 | `atl1-1-03-019-2-0` | 03:29:44 | 0 | 1 |
| 13429862 | P1 protein157 GPU ×3 + CPU | gpu-h200 | `atl1-1-03-020-18-0` | 01:54:36 | preempted | 3 of 4 (GPU) |

Kept but not used: 13376151/13376152/13376153 (preempted on
`atl1-1-02-012-23-0`, 24.5 GF/s per core, marked `metadata/DEGRADED-HOST`;
13376152's marker was written after the fact on 2026-09-23 with the same fields
as 13376151's). 13395713 (host gate, exit 75, `atl1-1-02-014-9-0`).
13395714, 13429863, 13480138, 13480139 (protein157 slices preempted before their
CPU case finished).

Two earlier attempts are kept rather than deleted, under
`memory-campaign-20260919/failed/`:

- `M1-job13357281-BUILD-var-clobber`
- `M2-job13357306-BUILD-var-clobber`

Both died in two seconds with
`sha256sum: 'x86_64-conda-linux-gnu/stage/lib/psi4/core*.so': No such file`.
The cause is that the scripts held the build directory in a variable named
`BUILD`, and `conda activate` overwrites `BUILD` with the conda build triplet.
The fix is in `common.inc`: the variable is now `P4BUILD`, and the script
asserts the directory exists *after* activation rather than before. Every script
added afterwards was checked with `grep -nE '^(BUILD|HOST)='` before submission.

## Measurement settings

Identical in M1, M3, M4, M5 and the P* jobs:

- `SAPT_DFT_GRAC_COMPUTE=ITERATIVE`, automatic GRAC shifts for both monomers.
- 8 OpenMP threads, 112 GiB Psi4 memory, 128 GiB SLURM allocation.
- Three repeats per case per arm, each in a **fresh process**, with CPU/GPU
  order alternating by repeat so neither arm always runs into a cold cache.
- GPU arm: `USE_CUEST=True`, `CUEST_XC=True`, `CUEST_MIXED_PRECISION=False`.
- Reported statistic is the median of the three repeats; `[min–max]` is printed
  beside it so the reader can see the scatter rather than trust the median.

M2 differs only in having no GPU arm and in running 8 against 24 threads.

## How memory is measured, and what each number can support

- **Host peak** — the kernel's `VmHWM` for the Psi4 process, an exact
  high-water mark rather than a sample. `/proc/self/clear_refs` is written with
  `"5"` immediately before the timed `energy()` call, so the peak covers the
  timed region only and not interpreter start-up or basis-set parsing. This
  number is exact and is directly comparable between arms.
- **Device peak** — the maximum over a 0.5 s poll of per-process NVML compute-app
  accounting, filtered to this PID. NVML exposes no high-water mark, so this is
  a **lower bound**: an allocation shorter than the sampling interval is invisible
  to it. Repeat-to-repeat spread in the device column (for example benzene
  aug-cc-pVDZ at 2878–3734 MiB across three identical repeats) is the sampler,
  not three different calculations. Do not difference two device numbers whose
  gap is smaller than that spread.
- **Ledger** — `psi4.core.memory_committed()` before and after the call, in MiB.
  Present only on the treatment arm; `null` throughout the control arm, which is
  how the control proves it is one.

## Attribution rules used in the report

1. A difference between M4 and M5 is attributable to the merge. A difference
   between M1 and M4, or M3 and M5, is attributable to libcuest. The first pass
   read M1 vs M3 as the merge; the M3→M5 table shows that reading does not hold
   on 0.2.2.2 (see the README).
2. A difference between this campaign and `phoenix-h200-20260910` is **not**
   attributable to the merge. `e971d2957b`, that campaign's build, and
   `d91b5f8e81` are 55 commits apart, including `52431fc2c0`, `60322c4275`, and
   `4a6164ef71`, which move the XC quadrature onto the device — and XC supplies
   66–79% of the GPU saving on every substantial six-case entry. The README reports that
   difference as a level and names the confound.
3. A median difference is reported as a change only when it exceeds the combined
   half-range of both arms' repeats. `build_delta.py` applies this mechanically
   and prints `~` otherwise; it does not have a tunable threshold.
4. Speedups, attribution shares, and GRAC shares are computed within a single
   job's tree, so they are unaffected by (2).

## Canaries

Each job runs a DGEMM / triad / scalar-loop probe at the start and at the end and
stores both in `metadata/`. M1 read 84.28 GF/s per core at the start and 84.31 at
the end, against 83.3 / 84.2 for the 2026-09-10 campaign — the same class of host,
steady across the job. `host_speed.py` reduces every tree's canaries to one table
and prints a verdict; see the README's host-speed section for why the aggregate
verdict is `mismatched` and why that is harmless.

## Re-running

```bash
# On Phoenix, from the campaign directory:
CAMPAIGN=/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-timing/memory-campaign-20260919
sbatch "$CAMPAIGN/jobM1-core6-h200.sbatch"      # treatment, gpu-h200, ~40 min
sbatch "$CAMPAIGN/jobM2-cpu24-core6.sbatch"     # thread scaling, cpu-small, ~45 min
sbatch "$CAMPAIGN/jobM3-premerge-core6-h200.sbatch"  # control, gpu-h200, ~40 min
sbatch "$CAMPAIGN/jobM4-core6-h200-cuest022.sbatch"          # treatment, libcuest 0.2.2.2
sbatch "$CAMPAIGN/jobM5-premerge-core6-h200-cuest022.sbatch" # control, libcuest 0.2.2.2
sbatch "$CAMPAIGN/jobP1-protein157-h200.sbatch"   # protein157 GPU x3 + CPU; CPU ~3.5 h, embers may preempt
sbatch "$CAMPAIGN/jobP3-protein157-h200.sbatch"   # protein157 CPU

# Then, from a checkout of the benchmarks branch with the trees at $RAW:
devtools/benchmarks/results/phoenix-h200-20260919-memory/regenerate.sh "$RAW"
```

`regenerate.sh` fails rather than skipping a table if any tree it uses is missing,
degraded, or not COMPLETED. The one exception is protein157's GPU run, which
comes from a tree that is FAILED only because its CPU case was preempted.
`regenerate.sh` checks that case's own `rc=0` and `ok` instead.
