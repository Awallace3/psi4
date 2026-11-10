# Cachy response workspace: real CUDA parity and diagnostic timings

**PASS_RESPONSE_CUDA_MICROBENCH_ONLY**, 2026-10-26.
This is not a whole Psi4/cuEST DFMO, F-SAPT, protein-admission or H200 pass.

## Attributable execution

- Source/helper: `2eb23026dd25c881d8746c806f23b24c558d7b04`.
- Optimization: `39ddf576eabf93ab02ba977210cac2ce10cedbd7`.
- Old response implementation: Git
  `7ebf33179d4b8d4c4a443c8a46e5fcfeb2d3274f`.
- Both arms use the same current CUDA transport/kernel implementation, actual
  CUDA session, device, fixed factors, energies and trial vectors. Thus the
  comparison isolates response workspace orchestration, not two whole installs.
- RTX3090, compute86, UUID
  `GPU-d36fd194-bdd6-3d53-96c9-6a67e3225eac`.
- Existing CUDA13.3/runtime13030, cuda-bindings13.3.0,
  cuda-pathfinder1.8.3, NumPy2.4.3. Exact library bytes and live mapped
  inode/device identities verified before and after numerical execution.
- Remote root:
  `/home/amwalla3/gits/psi4-disp0-cachy/runs/response-reuse-2eb23026dd-01`.
- Raw artifacts: `result.json`, `launch.json`, `cuda-restage.json`,
  `cuda-manifest.json`, per-case NPZ files, and child stdout/stderr.
- Verified `result.json` SHA256:
  `3ec481228bba273d4811707e028ab5742f315a01e9d52fc695020ff138bd22ce`.

Baseline response SHA256:
`27b0abb91b2216c427c58a10f56be3380276edb7d8b91973368afe7935861081`.
Candidate:
`45548e100472ff34408ac1f84d1eded147576d4a05107816347dce51f5329817`.
Runtime:
`19c22c061e7b181097bac44eaa17ee8d2814d7128471e869b3724dc9f90edc93`.
Compiled PTX:
`364a5b357adc29d9aba93e2b5ea5a0720e7486d59d1d13962deea58a2a61c6bc`.

## Measured operator-application latency

One untimed warmup per arm/case, then three timed calls per arm, alternating
arm order. Values are medians of all three, not selected best times.
Timing includes `apply()` allocations/contractions/internal cleanup and an
explicit completion synchronization. Factor generation/upload, NVRTC/session
creation, final reporting download and caller-owned output disposal are outside
the timed interval in both arms. Desktop co-tenancy is retained.

| Occupied / virtual / auxiliary | Q batch | Old median | Reuse median | Old / reuse |
|---|---:|---:|---:|---:|
| 2 / 3 / 5 | 2 | 2.701 ms | 1.387 ms | 1.95× |
| 8 / 32 / 65 | 7 | 6.715 ms | 2.274 ms | 2.95× |
| 16 / 96 / 129 | 7 | 15.169 ms | 3.652 ms | 4.15× |

All samples, in repeat order (milliseconds):

- Small: old `[2.849345, 2.700839, 2.414043]`;
  reuse `[1.409079, 1.387027, 1.380124]`.
- Medium: old `[6.711531, 6.759801, 6.715087]`;
  reuse `[2.274282, 2.209642, 3.694222]`.
- Larger: old `[15.169229, 14.945461, 15.260269]`;
  reuse `[3.554150, 3.870261, 3.652053]`.

These short synthetic cases deliberately exercise repeated Q slabs; they do not
establish a 1.95–4.15× whole F-SAPT or H200 speedup. Only three samples were
collected, and the medium reuse arm includes an outlier. No confidence interval
or production throughput claim is inferred.

## Numerical and lifetime checks

- Every warmup/timed output agrees with independent NumPy Hessian-action
  equations at unchanged `rtol=1e-11`, `atol=1e-12`. Worst observed absolute
  difference across all timed cases: **1.77636e-15** (synthetic response units,
  not an interaction-energy error in Eh).
- A separate CPU replay checked the saved NPZ hashes and recomputed each
  Hessian action using batched matrix products rather than the probe's
  `einsum` reference. Saved old/new outputs are **bitwise identical** for
  all three cases; the largest replay-reference difference is8.88178e-16.
- Small PCG solves include the original RHS, a changed RHS and zero RHS.
  Both arms take three iterations for the nonzero sources and zero for the
  zero source. Maximum solution error against an independent dense solve:
  **5.77316e-15**; maximum relative true residual **5.03488e-15**.
- Actual execution recorded 1,216 strided-batched cuBLAS GEMMs, 80 ordinary
  cuBLAS GEMMs and the CUDA response/update/reduction kernels, including warmups,
  timed applies and solves. This is not CPU serial-shim evidence.
- Peak explicitly tracked allocation: **23,144,328 bytes**. This is not total
  or peak VRAM. Boundary NVML samples showed 1,368 MiB in use, including desktop
  and runtime contexts; they are not peak measurements.
- Session cleanup completed with **0 live bytes and 0 native owners**.
  A post-process nonblocking lock check confirmed that the common lease had
  been released. No context reset, foreign process inspection/signalling,
  desktop shutdown or FDDS modification occurred.
- Actual cgroup readback: 4 CPU, 12 GiB RAM, swap0, tasks128. The common
  nonblocking lock and live HOLD checks covered staging, child work and cleanup.

## Runtime-locator repair, not a numerical relaxation

The first probe (`response-reuse-39ddf576ea-01`) stopped before CUDA because
the old temporary deployment no longer existed. Its artifacts were preserved.
The successful attempt staged **1,668,363,408 bytes** from the unchanged
canonical private CUDA environment into a new private executable tmpfs root,
retaining the original strict same-descriptor copy, RAM reserve, SHA256 and
mapped inode/device checks. SONAME aliases are local to that new root.

Canonical source-manifest SHA256:
`585820b1dab98790b71340de97a356ae473b260941bf8380df11a3fd4aa4869c`.
New manifest file SHA256:
`f82e7e3475a73abf1d22c772590df598a2222834787033062e310d7127cb2398`.

The installed Psi4 stages, dependency environment, old source checkouts and
original runtime locators were not modified. No cuEST SDK, native DFMO plan,
whole F-SAPT calculation or Phoenix job was started by this probe.

## Still outstanding

This closes actual CUDA validation of the bounded response-workspace reuse on
fixed synthetic factors, not the full production qualification. A coherent
whole-stage install and same-input F-SAPT oracle remain necessary before
claiming an end-to-end gain. Protein83's earlier DF-plan peak and nanotube's
localized-gauge repeat failure remain unresolved and must not be bypassed.
