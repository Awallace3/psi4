# GPU CPKS implementation handoff

This file records coordinator context for the independent Claude implementation
agent. Read `cpks-gpu-scope.md` for the scientific design and acceptance tests.

## Source baseline and ownership

- Parent worktree:
  `/home/awallace43/gits/psi4__worktrees/saptdft-cuest`
- Parent branch/head before handoff:
  `saptdft_cuest` / `4f47db835ad50ee4ed2b7b3005c99cb5cd10da35`.
- The parent has significant uncommitted work. The target is a NEW checkpoint
  branch made with `devtools/saptdft_grid_freeze.py` using an alternate index.
  Its manifest records the exact new commit and unchanged source state.
- Target branch/handle: `saptdft-cuest-cpks`.
- Target local worktree:
  `/home/awallace43/gits/psi4__worktrees/saptdft-cuest-cpks`.
- Existing CPHF J/K/GEMM implementation and tests are already part of the
  ancestry. `cphf-check.md` records their history; older results are not
  validation of your new GPU XC response implementation.
- The snapshot is not a promise that every dirty-tree declaration currently
  compiles. Establish a baseline build and reconcile any inherited
  declaration/definition mismatches transparently.

## Phoenix access and build

Use the installed Phoenix skill; resolve its files from the skill directory.
Remote commands go through its local `scripts/phx` with
`PHX_ALIAS=phoenix2`. The default alias has hung in this campaign.

Project root:
`/storage/project/r-cs207-0/awallace43`.
Create a distinct remote development worktree, branch-specific build directories
and run root for this task, e.g.:

```text
gits/psi4.saptdft-cuest-cpks
runs/saptdft-cuest-cpks/<immutable-checkpoint-or-job>
```

Existing readonly dependencies:

```text
Conda environment:
  /storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
cuEST:
  <env>/include/cuest.h
  <env>/include/cuest/exchange_correlation/xc_potential_compute_api.h
  <env>/lib/libcuest.so.0.2.2
CUDA toolkit:
  <env>/targets/x86_64-linux/include
  <env>/targets/x86_64-linux/lib
CUDA LibXC 7.1.2:
  /storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda
  library lib64/libxc.so.15
```

The selected CUDA headers report `CUDART_VERSION=13030`; CUDART resolves to
the matching CUDA 13 toolkit. The conda C++ compiler is
`<env>/bin/x86_64-conda-linux-gnu-c++`. CUDA LibXC's installed header and
shared-library exports include `xc_lda_fxc`, `xc_gga_fxc`,
`xc_lda_vxc_fxc` and `xc_gga_vxc_fxc`; their device behavior remains to test.

After loading site modules, isolate the explicit toolchain by clearing
`GCC_ROOT`, `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`,
`OBJC_INCLUDE_PATH` and `LIBRARY_PATH` where applicable, and bind the selected
CUDA toolkit and runtime libraries consistently. Follow the existing ABI
probe/gate machinery (`devtools/saptdft_cuda_abi_probe.py`) and the skill's
same-environment preprocessing/compile/link checks.

Use existing build scripts/manifests as references, but build in your own
directories. Reuse dependencies readonly; clone/create a dedicated environment
if dependency changes are actually needed. Never rebuild a frozen benchmark
binary or install into the shared environment.

Working historical scientific checkpoint:
`c28950041bbe8bb10617264a7b6ddc56a436cda1` in remote worktree
`gits/psi4.saptdft-cuest-grid-correctness-v1`, with `build_grid_host` and
`build_grid_cuda`. Its CUDA core SHA256 is
`91d6018627e830cb7a088bf44b6e715c9a9694aaebb86907b78186f03735b8d7`.
This is reference material, not an editable build for this task.

## Submission and monitoring policy

- New development/test jobs: **`gts-cs207-chemx` / suitable GPU partition /
  explicit `--qos=embers`**, no more than eight hours per job.
- Prefer one appropriately small GPU allocation for SDK and response smoke
  tests before expanding validation. H200 is a known route in this campaign;
  select the smallest credible resources and validate with `sbatch --test-only`.
- Check shared per-user `embers` headroom before submitting. Paid `inferno`
  jobs need fresh user approval for this task.
- Use TERM-safe scripts, immutable source and input pins, fresh run directories,
  truthful completion markers, `.batch` exit codes and real output validation.
- Every staged import must assert that Psi4 resolves into your new build, not
  the conda release or the repository's Python package directory. Run Python
  outside the source root or use safe-path setup; source-root shadowing caused
  earlier misleading import failures.
- Follow the skill's PsiAPI/scratch setup. Long jobs should use a job-specific
  writable Lustre scratch directory because private `/tmp` has vanished
  mid-allocation in this campaign.
- Watch with the skill's `scripts/watch_jobs.sh -i 1800 phoenix2 <jobids>`.
  Use a durable, harness-native background task and report its identity.
  Do not replace this with a rapid SSH polling loop.
- Never cancel, modify or replace any existing job without user authorization.

Protected previous jobs/checkouts:

- Protein157 benchmark **14074070** uses a frozen scientific build and harness
  `006bf18f75509befb103d35134418d156589aecc`. Its results belong to the
  coordinator's separate grid-savings campaign, not CPKS.
- Earlier CPHF validation **13991128** pins
  `ae8cedca770e1bc1a7a6091d313cfc6dec8887d9`. Do not change its source,
  build or configuration, and do not treat its eventual result as validation
  of this new implementation.
- Remote Protein157 harness worktree
  `gits/psi4.saptdft-cuest-protein157-grids-v2-sparse` and existing frozen
  correctness/campaign worktrees must stay unchanged.

## Local CPU environment

```text
/home/awallace43/miniconda3/envs/p4_ci/bin/python
```

The parent's staged CPU package is under
`build_saptdft_cuest/stage/lib/psi4`, built without cuEST. It can be a
reference but is not your newly modified binary. Build/test independently.
Local NVML previously reported a driver/library mismatch; local CPU checks
are not evidence of GPU execution.

## Evidence boundaries

The existing report and PR cover grid switching and older response J/K work:

```text
https://cos-4a16505.tail7d58ac.ts.net/docs/saptdft/cuest/breakdown-2026-10-04.html
https://github.com/Awallace3/psi4/pull/12
```

Keep new CPKS results in your branch/progress document first. This task does
not authorize merging or automatically rewriting the parent PR/report.
Energy agreement of `1e-5 Eh` for sparse grids is separate from the much
tighter fixed-orbital response and solver-residual validation described in
the scope. `induction=NONE` runs are not CPKS benchmarks.
