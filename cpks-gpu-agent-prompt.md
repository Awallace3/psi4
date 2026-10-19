# Implement GPU XC response on saptdft-cuest-cpks

Implement and test the reviewed design in `cpks-gpu-scope.md`, not just another
scope document. The user explicitly requested a Claude workmux agent on branch
`saptdft-cuest-cpks` and Phoenix GPU validation.

## Start

1. Read `cpks-gpu-scope.md` completely, then `cpks-gpu-runtime.md` and the
   repository's applicable instructions. Treat scope code snippets as
   illustrative: reconcile actual declarations, call sites, build guards and
   SDK contracts before writing code.
2. Confirm your branch/worktree. This branch starts from an isolated snapshot
   of the coordinator's dirty tree, including existing grid switching and
   cuEST response J/K/GEMM changes. Keep edits and commits on your branch.
   Preserve other branches, worktrees, queued-job pins and shared environments.
3. Use the Phoenix skill at
   `../../../.pi/agent/skills/phoenix-slurm-experiments/SKILL.md`
   and its Phoenix/Psi4 references for remote work. Runtime details in
   `cpks-gpu-runtime.md` refine the generic environment defaults.

## Deliver

- Working cuEST-grid `compute_Vx` for native-supported LDA/GGA RKS singlet,
  RKS triplet and UKS, with host LibXC and CUDA LibXC routes.
- Existing dense API preserved; optimized reuse of existing CPKS left/right
  J/K factors; signed/nonsymmetric inputs, correct spin scaling and ordering,
  symmetry, reference screening, bounded memory and safe object lifetime.
- Preserve native RKS-singlet GRAC response conventions and explicit native
  exclusions. mGGA, VV10, direct J/K, analytic nuclear Hessians and the
  separate FDDS work remain outside this task.
- Regression tests with actual GPU-grid V objects, fixed-orbital CPU/GPU
  comparisons, signed/transpose/zero/cross-spin cases, dense/factored parity,
  nonzero perturbation irreps, repeated reference/grid changes, solver
  residuals and a small SAPT induction calculation.
- Phoenix GPU evidence for the SDK normalization/signed-density contract and
  required CUDA LibXC derivative calls before claiming correctness. Then
  measure cold setup, repeated kernel actions and complete response solves.
  Record failures or missing cases honestly rather than weakening tests.

Resolve the source-reviewed edge cases explicitly: separate UKS cutoff masks,
RKS half-alpha normalization, triplet polarized derivative combinations,
native UKS output-pair indexing, GRAC-triplet rejection, and derivative
capabilities. Header/symbol availability is not a device numerical test.

## Cluster authorization and isolation

This request authorizes appropriately sized free `embers` development/test
jobs. It does not grant new paid `inferno` approval; earlier paid approvals
were for a different Protein157 campaign. Use the route and limits in the
runtime handoff and skill. Show a concise job plan before a multi-job batch.

Keep separate source/build/run directories and immutable job checkpoints;
move source through git, never copied tracked files. Use small systems first.
Monitor cluster jobs with the official watcher at **1800-second intervals**.
Keep output paths, job IDs, effective account/QoS, source commit, binary hash
and real backend provenance in the progress document.

## Finish

Maintain `cpks-gpu-progress.md` with implementation milestones, exact commands,
tests, submissions, results, blockers and resume instructions. Commit coherent
changes on `saptdft-cuest-cpks`; do not merge into the parent branch, rewrite
other branches, update the benchmark HTML/PR with unverified results, or
claim completion while required tests fail.

Return a concise implementation summary with commit IDs, local test results,
Phoenix job IDs and artifact locations, CPU/GPU numerical errors, timings,
remaining limitations and any still-running monitor. If blocked, leave the
branch and progress document ready to resume and identify the precise blocker.
