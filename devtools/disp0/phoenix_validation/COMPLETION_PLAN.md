# Phoenix full F-SAPT completion contract

User request: complete the missing GPU qualification, scalability, matched
CPU/GPU timings and analysis. Inputs are **peptide, nanotube, protein83** from
`~/data/timing_test_suite/geometries/systems.py` (the user confirmed protein83,
not protein87). Primary basis is spherical **aug-cc-pVDZ**. This is **full
fragment F-SAPT**, not scalar SAPT or dispersion-only timing.

## Ordered acceptance gates

1. **Pinned Phoenix runtime and small H200 gate**
   - Keep successful CPU build14309728/source3343e673c2 and all failed roots.
   - Pin optional cuda-bindings13.3.0 and resolved dependency artifacts in a
     new private dependency directory; do not install into shared environments.
   - Authenticate canonical cuEST0.2.50 archive/native/all59headers, complete
     CUDA13.3 closure, native/Python E2 world and the actual whole Psi4 stage.
   - Build a new whole stage when scientific Python files change. No new
     Python overlay on the successful old core.
   - Qualify real H200 UUID, NVRTC, FP64 cuBLAS including strided-batched
     exchange, native DFMO, signed terms/fragment matrices, changed/repeated
     inputs, complete cleanup and explicit no-CPU-factor/localizer fallback.
     CPU/mock/serial fixtures are not this evidence.

2. **Scale-ready response, overlaps and localization**
   - Replace reference O(Q O² V²) response exchange with exact blocked FP64
     GEMMs for BOTH distinct exchange contractions. Preserve full occupied
     response, physical amplitude sign, PCG curvature and true-residual checks.
   - Replace prototype P4096/union512/O64 policies with explicit simultaneous
     allocation/work budgets and native ABI limits. Never just disable guards.
   - Preserve original spherical/SPDF, metric, frozen/active, IBO pair ordering,
     convergence, owner lifetime and same-gauge JKFIT/RIFIT invariants.
   - All three actual augmented-basis systems must pass numerical/memory gates
     before accepting their timing measurements. OOM/nonconvergence are failed
     points, not permission for CPU fallback or relaxed tolerances.

3. **Matched measurements**
   - Three independent fresh-process repeats × CPU/GPU × three systems.
   - Match geometry, primary/auxiliary bases, threads, frozen core, coupled
     response, delta-HF, exchange/induction scaling and output conventions.
   - The CPU reference must implement the same coupled fragment science.
     Inspect the current CPU fragment path: canonical-scalar scaling of an
     uncoupled partition is not an exact coupled-response baseline. Use an
     independent matrix-free CPU oracle at large sizes, not a dense P×P solve.
   - Time whole energy calls and nonoverlapping phases, including first-use
     initialization, native factory, transfers, audits and cleanup. Separate
     diagnostic/oracle work. Record runtime/core/module/config/input identities,
     exact work, RSS/VRAM/scratch and failed/incomplete points.

4. **Analysis**
   - Require signed term/fragment correctness before reporting speed ratios.
   - Publish raw records, per-system three-repeat medians/ranges/sample counts,
     phase timing vectors, resource peaks and all failures.
   - Report slowdowns as slowdowns. No extrapolated protein results, historical
     PBE0-D4 baselines, kernel-only labels for whole-call timing, or fabricated
     confidence intervals.

## Science/resource decisions

Use matched **aug-cc-pVDZ-JKFIT** and **aug-cc-pVDZ-RI**. Their H/C/N/O shells
remain SPDF. `def2-universal-jkfit` contains G shells on C/N/O and is outside
the currently qualified native factory; do not silently substitute it to save
memory.

Atom-additive basis-file counts (confirm with the actual installed stage):

| System | Primary | JKFIT | RIFIT | Full occupied A/B |
|---|---:|---:|---:|---:|
| peptide | 356 | 1308 | 1042 | 20 / 20 |
| nanotube | 742 | 2748 | 2240 | 8 / 78 |
| protein83 | 1293 | 4762 | 3820 | 78 / 68 |

Prefer bounded free embers preparation jobs. Fresh inferno submissions require
the user-approved footprint and a live account selector; the earlier CPU-build
approval is not blanket approval for a new GPU campaign. Use one H200 and at
most8 CPUs unless exact-route validation establishes another approved shape.
No job-control operations or overwrites of old artifacts without approval.

## Current development evidence

- Baseline Phoenix CPU build: PASS14309728;244 CPU regressions,8 launch guards,
  5 E2 API checks; no GPU qualification.
- Response implementation now uses Q-blocked strided-batched native-FP64
  GEMMs, with resident-factor/PCG/workspace admission before DFMO.
- Focused CPU-shim tests pass for signed Hessians/solutions, nonsymmetric
  exchange indices, ragged Q blocks, BLAS transpose/broadcast descriptors,
  changed inputs and large-space admission without a dense Hessian.
- Overlap, native powers, IAO and ordered IBO memory/work admission changes
  are in progress; broad regression and actual device gates remain required.
- Cachy was inspected read-only for prior diagnostic conventions. Its GPU/build
  HOLD remains intact; no Cachy execution or environment change is authorized.
- Both matched public arms select `SAPT_DFT_DO_FSAPT=SAPTDFT` and
  `SAPT_DFT_INDUCTION_TYPE=CPKS`. At HF this is the HF response; explicit
  `CPHF+SAPTDFT` is rejected by the public driver. The opt-in CPU fragment
  solver explicitly constructs native **DF** JK with the same JKFIT basis.

The persistent goal's short label is “Complete these tasks”; this file records
the concrete scope and evidence required before claiming it complete.
