# Transferring the XDM/cuEST performance lessons to Disp0/F-SAPT

## What the XDM result actually establishes

Source: <https://cos-4a16505.tail7d58ac.ts.net/docs/psi4/xdm/cuest.html>,
especially sections 8–11. Read the locally served HTML and inspected
`xdm-gpu-kernels` commit `dd12e4c89aaf95de2c69a6a64a097ff7ee9c35de`.
The XDM worktrees were not modified or merged.

At 120 atoms, the reported XDM step falls from about 2.41 to 0.87 seconds
after broadening the collocation launch and making the point dimension
contiguous. The 369× comparison is against host XDM on the psi4 grid, not
against an entire F-SAPT calculation. It cannot be transferred as a speedup
prediction. The later `3661ec3da6` deliberately reverts smaller GEMM-waste
chunks: reducing arithmetic made those runs slower by reducing occupancy.

The reusable lessons are:

1. Profile stages on identical numerical inputs before selecting a kernel.
2. Keep inputs resident and reuse an explicitly bounded workspace.
3. Queue dependent work on one stream instead of synchronizing each chunk.
4. Change launch geometry/layout together when independent work underfills
   the GPU; do not confuse fewer operations with less elapsed time.
5. Check same-input parity separately from differences between independently
   converged wavefunctions. Record GPU clocks/power/throttling as well as memory.

## Implemented here: stream-ordered response workspace reuse

`HFResponseOperator.apply` previously made four temporary batched-GEMM
allocations inside every Q-slab scope. Leaving each scope invokes device-wide
synchronization and frees those buffers. This happens for every operator
application in every PCG solve, including its true-residual check, for every
induction source.

The new implementation allocates four buffers once **per apply**, sized by
the existing `q_batch` and `max_workspace_bytes` contract. Each slab writes
views of those buffers; the short final slab uses only its prefix. The
accumulation kernel and all four GEMMs remain on the session's single stream,
so the next slab cannot overwrite data before its preceding accumulation.
Only the enclosing apply scope synchronizes and frees workspace.

Changes:

- `sapt_cuda_runtime.py`: optional `gemm_batched(..., out=...)`, with exact
  extent/type/layout/session/lifetime checks and rejection of overlapping input
  storage. Native FP64, transpose descriptors, strides and beta=0 are unchanged.
- `sapt_fisapt_response_cuda.py`: reuse T1/K1/T2/K2, with no per-Q allocations
  or cleanup scopes. Scratch is not retained across applies or source vectors.
- `sapt_cuda_runtime.py`: memory rejection now preserves request, tracked live
  bytes, opaque reserve, limit, CUDA free bytes and the rejecting predicate(s).
  This adds no CUDA query, retry or relaxed budget. Previously protein83's
  exception could not distinguish declared-budget from CUDA-free rejection.

The two exchange contractions, ascending Q accumulation order, canonical
denominators, PCG tolerances/curvature checks and independent true-residual
check are unchanged. No localization policy, SDK allocation ownership or
native-plan lifetime is changed.

### Structural savings, not measured GPU speedups

For `L` Q slabs, apply allocations change from `4 + 4L` to **8**, and cleanup
scopes from `1 + L` to **1**. The four BLAS calls per slab are unchanged.
Using the recorded dimensions and the default 256 MiB response workspace:

| Case | O / V / Q | q_batch / slabs | Allocations before → after | Cleanup scopes before → after |
|---|---|---|---|---|
| Nanotube B | 78 / 662 / 2748 | 207 / 14 | 60 → 8 | 15 → 1 |
| Protein83 A | 78 / 1215 / 4762 | 114 / 42 | 172 → 8 | 43 → 1 |
| Protein83 B | 68 / 1225 / 4762 | 131 / 37 | 152 → 8 | 38 → 1 |

These are calculated application-level work counts, **not executions of those
large cases**. Eliminating host allocation/synchronization overhead does not
accelerate the remaining GEMM arithmetic by the same ratios.

### Verification

The regression exercising the real response orchestration failed before the
change with **24 allocations instead of 8** at Q=5, q_batch=1. It checks
q_batch=1/2/5, the short tail, repeated changed RHS inputs, output independence,
one cleanup scope, and workspace peak/live accounting.

The descriptor tests execute the real cuBLAS descriptor builder against a CPU
BLAS spy, including both operand transposes, rank-two broadcast, nonzero
offsets, poisoned output padding, no output allocation, invalid ownership,
ended lifetimes and aliases. The response tests compare nonsymmetric factors
against independent four-index Hessians and solve changed RHS vectors.
Memory-rejection tests distinguish both predicates, their conjunction, negative
requests and exact-bound acceptance without allocating/retrying.

Run from the repository root:

```bash
PYTHONPATH=devtools/disp0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m unittest discover -s devtools/disp0 -p 'test_cuda*.py' -v
```

These are CPU equation/descriptor/ownership tests. They do not simulate CUDA
parallel scheduling and do not establish actual GPU parity, throughput or a
production speedup. No GPU/cluster allocation was launched for this change.
The expanded command above passed **230 tests in 9.440 seconds** on the local
CPU, including the new workspace/descriptor and memory-rejection regressions.

## Next bottlenecks, deliberately not changed blindly

- **Protein83 plan-creation peak:** two native DF plans retain large creation
  temporaries before fragment numerics start. Response scratch reuse does not
  reach or fix this earlier failure. Verify the exact SDK borrowing/completion
  contract, then shorten proven-safe lifetimes or serialize stages. Do not
  simply increase the budget or free every buffer marked temporary.
- **Nanotube localized gauge:** nearly equal energies/objectives coexist with
  different B localized partitions. Broadening ordered Jacobi pair launches
  would change dependencies; a different gauge must not be passed off as an
  optimization or accepted by loosening tolerances.
- **Repeated S-side DFMO production:** `blocked_device_dispersion` computes
  identical S-side factors inside every R-tile iteration. Its factor-batch
  count is `nR + nR*nS`; a budgeted cache of packed S-side slabs could approach
  `nR + nS` if they fit. This is the strongest remaining algorithmic reuse
  candidate, not an implemented speedup. Preserve R/S accumulation order,
  provider/session/gauge identity, crossed orientation and changed-input
  invalidation. Packed-cache residency must fit together with native plans and
  response factors; do not retain raw and packed duplicates indefinitely.
- **Duplicate input staging:** `fragment_terms` already uploads raw HF inputs
  before passing their host copies to `prepare_hf`, which uploads them again.
  A validated device-input preparation seam can reuse the original buffers.
  This is narrower and safer than a zero-roundtrip native-SCF handoff.
- **One-block reductions / strided factor loads:** candidate profiling targets,
  not demonstrated bottlenecks. A two-dimensional launch needs independent work,
  bounded partial sums and explicit parity checks. XDM's independent point/shell
  launch cannot be copied onto dependent orbital rotations.

The next numerical validation should use fixed factors/RHS for old/new
response applies and solves, then the full same-input oracle, under the existing
bounded Cachy GPU lease protocol. A subsequent Phoenix comparison should use
GPU-only allocations, event/whole-call timing, clocks/throttle samples, separate
CPU references and the existing accuracy/cleanup gates. No massive end-to-end
speed claim is warranted until those measurements and the large-case blockers
are resolved.
