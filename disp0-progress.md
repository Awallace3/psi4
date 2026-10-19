# Python/Einsums GPU SAPT0 dispersion progress

## Ownership and immutable launch pins

Dedicated worktree: `saptdft-cuest-disp0`; branch: `saptdft_cuest_disp0`.
No changes to sibling worktrees, coordinator builds, environments or run roots.

* Base `satpdft_einv2`: `1592851675cbd8ce6898eca969976a6e7becdb24`
  (verified HEAD and ref at start).
* Required reference `fisapt_fdisp_blocked`:
  `907f6d75690c071dd9b452e3f0a82abb7ba317ea` (verified).
* Optional plumbing `saptdft_cuest` and `saptdft-cuest-cpks`:
  both `ca21ef8645a73263281480454b730085506adca5` at inspection.
* Shared merge base with the two source histories:
  `288fe50fc673747944a77b08af250fb0ce557c8f`.

No tracked AGENTS.md/CLAUDE.md or ancestor guidance found at initial inspection.
`workmux send saptdft-cuest` succeeded with ownership/readiness notice.
No GPU allocation requested/submitted yet.

Parent runtime handoff received: Phoenix prefix
`P=/storage/project/r-cs207-0/awallace43`, env
`P/miniconda/envs/p4cuest_sapt`; cuEST 0.2.50
`P/software/cuest-0.2.50-a06e8148/lib/13`, explicit cuda13 toolkit
(13.3.73, runtime 13030), conda GCC/G++ 14.3 wrappers and CUDA LibXC 7.1.2.
Recipe pin `8af4a8bf034c16c0c01821b8442673db0f54514f`; scientific cuEST
pin `ca21ef8645` remains reference-only, not an ABI claim for this branch.
The parent's coherent dependency stage has native/Python Einsums 1.1.3.
This launch base actually calls `RuntimeTensorD`, `reshape_view`,
`MemoryPool`, `einsums.graph`, `outer_sum` and `direct_division`; check those
APIs explicitly rather than infer compatibility from the path's name.
Parent clarification: that 1.1.3 stage is **not applicable** to this branch.
Do not downgrade or mix native/Python major versions. Investigate a dedicated
coherent CUDA-enabled Einsums2 build, preferably at the CPU-oracle revision
`d17bb76e`, and test used operations, views, pools, graph replay, layout and
division semantics. Toolchain recipes are candidates, not validated Einsums2
compatibility. Readiness coordination remains mandatory before any GPU job.

Allocation constraint: **do not submit yet**. Proposed correctness gate is
one H200 / 4 CPUs / 32G host / 30min / embers / gts-cs207-chemx, subject to
another readiness checkpoint (CPU outcomes, dedicated GPU build/import/ABI
proof, comparisons/tolerances, memory/runtime). No parent holder is available.
A separate bounded build job, if needed, must first be proposed. Do not
repair workstation drivers or change other allocations.

## Plan and gates

1. Inspect actual HF dispatch, einsums_v2 APIs, blocked FISAPT equations,
   tests and available isolated runtimes.
2. Review and import the optimized blocked reference **before GPU work**.
   Prefer minimal faithful port/cherry-pick over broad divergent histories.
   Preserve pair blocking and the einsums_v2 memory order; record exact commits.
   Validate reference and commit this as an explicit prerequisite milestone.
3. Port only necessary committed GPU endpoints, then implement Python-level
   FP64 blocked SAPT0 dispersion and exchange-dispersion for
   SAPT_DFT_FUNCTIONAL=HF. Preserve non-HF and CPU paths.
4. Validate signed components, sums, dispatch, block/edge cases and multiple
   dimers/bases against CPU oracles; perform actual GPU checks with endpoint
   evidence and bounded memory. Record source/build/runtime pins and commands.
5. Only after that numerical gate passes, inventory all F-SAPT terms and
   implement/validate Python Einsums GPU coverage and fragment sum rules.
   If GPU validation is unavailable or fails, stop before Phase 2.

Initial tolerance policy (to be confirmed against oracle precision before
interpreting results): synthetic FP64 kernel comparisons rtol=1e-11,
atol=1e-12 Eh; physical signed energy components atol=1e-8 Eh and rtol=1e-8,
with discrepancies retained individually. Compare unrounded values where
available; printed reference precision cannot justify tighter assertions.
No tolerance relaxation to hide failures.

## Current status

* Lineage verified, worktree initially clean.
* Reference imported as `fb430094cc` via
  `git cherry-pick -x 907f6d75690c071dd9b452e3f0a82abb7ba317ea`.
  This exact, clean three-file cherry-pick imports the native CPU oracle,
  BLOCKED/PAIR options and frozen-core regression test. No new native GPU
  algorithm is being introduced.
* Reviewed equations: `V=Aar.Abs`; exchange `Aar.Fbs+Far.Abs`, plus
  permuted `Bas.Bbr+Cas.Cbr` and four rank-one overlap/potential terms.
  Denominator `ea+eb-er-es`, factors +4 (dispersion) and -2 (exchange).
  Pair fallback retained for parallel/perpendicular link-orbital terms.
* Deliberately excluded `3c06680fea` and `e8478e2064`: these touch the
  divergent v1 Python module. The launch base already has the v2 Q-major
  packed factors, bounded staging, localized factors and block reductions.
  Replacing that module would regress the required route.
* **Reference numerical gate PASSED**: 2 physical tests, frozen-core off/on,
  same localized ethane-water/cc-pVDZ wavefunction for PAIR vs BLOCKED edges
  1/5/500. Both signed Disp20 and Exch-Disp20, combined Disp_AB and partition
  sums agree at rtol=1e-11/atol=1e-12 Eh. No tolerance changes.
* Phase 1 implementation checkpoint (not yet numerically validated):
  opt-in `SAPT_DFT_MP2_DISP_ALG=EINSUMS`, with HF-only validation and
  `SAPT_DFT_EINSUMS_GPU=false` default. Existing SAPT/FISAPT defaults and
  non-HF paths remain selected as before. Canonical mode shares fdisp0's
  equations with active occupied slices, without localization caches.
* Kernel-local `sapt_dispersion_gpu.BlockContractions` uses three packed
  rank-2 einsum descriptors in default graph replay, cached per edge shape.
  No global executor changes and no new native algorithms. It reports
  placement as **planned**, never device proof, and accounts for packing,
  replay including transfers, other host work and total elapsed time.
* GPU implementation has not executed on CUDA. No GPU correctness claim.
  Fragment-GPU and S-infinity dispersion are explicitly rejected in this
  initial scalar checkpoint rather than silently taking a CPU route.
  Fragment dispersion/sum-rule coverage is still required before claiming
  the full Phase 1 gate; no broader F-SAPT terms have been implemented.
* Phase 2: gated, not started.

Parent scope agreement: Phase 1 is explicitly **GPU contractions with CPU
elementwise/reductions**, not fully GPU-resident dispersion. Use default
`Graph.execute` only in the new local kernel route; do not globally change
existing executor semantics. No new C++ dispersion or elementwise algorithms.
Test packed/strided layouts, pool/view lifetimes, repeated replay and shadow
freshness against signed components. Account for host work and transfers in
end-to-end timings. After the real GPU numerical gate, inventory remaining
F-SAPT GPU API gaps and propose a Python-level approach before expanding.

## Dispatch and GPU capability review

* Existing `energy("sapt(dft)")` + `SAPT_DFT_FUNCTIONAL=HF` correctly skips
  FDDS and DFT exchange-dispersion scaling. Plain scalar dispersion still
  calls native `FISAPT.disp` or `core.sapt`, even from
  `sapt_mp2_terms_ein.py`. Only the SAPTDFT F-SAPT branch calls Python
  `sapt_jk_terms_ein.fdisp0`. There is no `einsums_v2` option.
* A scalar Python entrypoint needs active canonical occupied coefficients,
  separate frozen-core counts and identity occupied rotations (or an explicit
  canonical mode), not accidental dependence on fragment localization caches.
  Non-HF FDDS and explicit D3/D4 aliases must retain existing dispatch.
* Local Einsums source pin `d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62`
  (tracked files clean; untracked install/). Its current local build has
  CUDA/HIP/GPU support OFF; Python imports report 2.0.0.
* Reviewed GPU graph dispatcher: packed column-major rank-2
  `ein.einsum` descriptors and strided-batched GEMM can dispatch to FP64
  cuBLAS. Merely calling `ein.linalg.gemm` is **not** evidence: its
  GemmDescriptor is excluded by this GPUPlacement implementation.
  The current driver's custom SequentialExecutor also bypasses the default
  device-shadow replay. These must not be labelled GPU execution unchanged.
* Python exposes host RuntimeTensors, not explicit device arrays. Default graph
  replay can retain shadows across consecutive supported contractions, but
  final replay flushes to host. Denominator/division, permutations and general
  reductions lack GPU dispatch in the inspected library; an initial
  GPU-contraction/CPU-elementwise hybrid must be reported as such.
  Inspect cuBLAS trace/profiler `gpu_dispatch` rather than placement alone:
  replay can legitimately decline to CPU.
* Committed Psi4 generic `core.cuest_chain_gemm` is another candidate, from
  `f5ed28020cc5fb65e86a9391460bd827d00c59e1`, reviewed at `ca21ef8645`.
  It accepts host `core.Matrix` operands, uploads factors, retains chain
  intermediates and downloads requested results; it is not a public persistent
  device tensor API. It depends on the committed cuEST runtime/handle plumbing
  absent from this base. No broad history has been merged to obtain it.
* Einsums2 CUDA build audit at exact `d17bb76e`: CMake >=3.25.2, C++20,
  native CUDA toolkit; build shared native library and Python `_core`
  together. Dependencies include Apiary 1.x code generator, fmt 12,
  mimalloc >=3.3,<3.6 (excluding 3.4.0/3.4.1), spdlog 1.17 source,
  HDF5, OpenMP, BLAS/LAPACK and a Python-3.13-capable pybind11.
  `EINSUMS_BUILD_PYTHON=ON`, `EINSUMS_WITH_CUDA=ON`, explicit arch 90,
  HIP/MPS/mock OFF, split binding TUs for bounded memory are candidates.
  No source rule excludes CUDA13.3/GCC14.3, but compatibility is untested.
  No upstream source change is yet justified merely to import/build.
* Native E2 soname is `libEinsums-abi1.so.1`, not inferred from major version.
  `einsums.sealed.world()` after importing the dedicated Psi4 stage reports
  2.0.0-trunk, exact `d17bb76e`, compiler GNU14 and a single mapped library
  `$HOME/gits/Einsums/build/lib/libEinsums-abi1.so.1`. No 1.x library mapped.
  This validates CPU runtime coherence, not CUDA execution.
* Stock Python does not expose GPUPlacement's force/threshold parameters.
  `pm.enable("GPUPlacement")` does not force cost-model-rejected small
  blocks onto the device. Tests must retain and label CPU edge-block
  fallbacks and require actual eligible contractions plus cuBLAS trace.
* Pool `pinned=True` is host resident-page policy, **not CUDA-pinned memory**.
  `outer_sum` assumes contiguous vector elements in its inspected loop;
  energy vectors used by the kernel must be packed/stride-one. Do not
  generalize its semantics to arbitrary strided vectors or broadcasting.

## Execution log

* `git status --short`, `git branch --show-current`,
  `git rev-parse --verify` for HEAD and all named source refs: PASS.
* Read-only source-history diff/log inspection: prerequisite tip changes
  FISAPT native reference plus its tests; divergent earlier histories also
  contain einsums changes, so a blind broad merge is inappropriate.
* Runtime preflight: system `python` absent; system python3 has no Psi4.
  `$HOME/miniconda3/envs/p4_einv2/bin/python` imports release Psi4 1.10 and
  editable Einsums v2 from `$HOME/gits/Einsums/build/lib/einsums`.
  Release Psi4 lacks `FISAPT_FDISP_ALGORITHM`; it cannot validate the import.
* `bash devtools/disp0_build_cpu.sh`: background isolated build started in
  this worktree's `build_disp0_cpu`, reading p4_einv2 dependencies without
  installing into or modifying that environment. Native ENABLE_Einsums=OFF
  follows the base build (Python Einsums remains available independently).
* Local `nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader`:
  FAIL, NVML driver/library mismatch (library 580.178). No local GPU
  correctness evidence; Phoenix allocation coordination will be required.
* Isolated CPU build: PASS (background task b748623db, exit 0), native
  scientific source at `fb430094cc`; stage
  `build_disp0_cpu/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so`,
  SHA256 `cbc0ff81eadbab331193c4ad9fb35c0ba70b70e8edef0606713c87dd0188e72a`.
  Einsums2 Python native module SHA256
  `94a7f03f3ee0402de4b2ba43f16c3093dfd869ee7fc0184dfe8969638e6f6c02`.
* `bash devtools/disp0_test_reference.sh`: launched after successful build;
  first attempt FAIL before numerical execution (task b75d4fd27):
  generated `--psiapi` appends `$PYTHONPATH`, which was unset under nounset.
  Fixed runner to define an empty PYTHONPATH before evaluating --psiapi.
  Second attempt ba57c4f18 failed before numerical execution because source
  cwd shadowed the staged package (ImportError: no source-tree core).
  Runner now changes to its isolated run directory before Python/pytest.
  Both are harness failures, not numerical pass/fail. Full rerun launched as
  b2f6b3d0a: **PASS**, exit 0, `2 passed, 9 deselected in 9.08s`.
  Results: `build_disp0_cpu/reference-check-9PFhJqF7/results.xml`.
  Source scientific pin `fb430094cc48d55c3479dae0040276c6c6dc6dd1`,
  dedicated core/import proof printed before pytest, 2 CPU threads.
  This is the prerequisite CPU reference gate, **not the GPU gate**.
* `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1
  $HOME/miniconda3/envs/p4_einv2/bin/python devtools/disp0_einsums_api_probe.py`:
  PASS, exit 0 (task b693753c0). Tested FP64 packed contractions,
  occupied-fastest rank-4 denominator construction/division/reduction at
  shapes 1x1, 3x4 and 2x1; default graph replay with a strided submatrix
  and mutated input; view keepalive after dropping its pool-carve owner;
  rejection of reset with a live view and release after view destruction.
  This initial dependency probe imports installed release Psi4 first; it is
  not a scientific source-pinned calculation. Repeat with the dedicated
  scientific stage: PASS (b05a7e7ad), identical five checks, Psi4
  1.12a1.dev615 imported from `build_disp0_cpu/stage/lib/psi4`, Einsums2
  from the exact CPU-oracle dependency build.
* Additional runner isolation fix: run Python stdin/pytest from the new
  run directory, not the repository root. A direct audit command from the
  source root reproduced source `psi4/` shadowing the staged native package.
  The successful stage API probe and sealed-world probe use a non-source
  working directory. No scientific mismatch is attributed to this harness issue.
* Canonical checkpoint incremental dedicated build: PASS (bf9409911).
  Source is currently the uncommitted Phase 1 checkpoint above; no claim of
  a finalized source pin until its tests pass and it is committed.
* `bash devtools/disp0_test_reference.sh
  "$PWD/tests/pytests/test_saptdft_disp0.py"
  "$PWD/tests/pytests/test_saptdft.py::test_einsum_terms"
  "$PWD/tests/pytests/test_fisapt.py::test_fsapt_fdisp_blocked_matches_pair"`:
  b55c1dc04: **PASS, 15 passed in 18.50s**, 3 JUnit-schema warnings.
  Results: `build_disp0_cpu/reference-check-IvAuRbOB/results.xml`.
  XML parsing confirmed all 36 signed discrepancy records were retained.
  Runner now selects legacy JUnit schema to avoid record_property warnings.
  This run used the uncommitted checkpoint over `69e8299046`; it is not
  represented as a finalized source pin. Physical cases: water-Ne at two
  distinct geometries/cc-pVDZ spherical/unfrozen, ethane-water/6-31G*
  Cartesian/frozen. Each signed component is compared to native FISAPT for
  edges 1/5/500, eager and independent default CPU graph execution. Every
  discrepancy is retained in JUnit properties. Additional tests poison and
  mutate graph inputs across cache hits, pack strided views, verify pool
  reset fails with live replay borrows and succeeds after release.
  Maximum absolute discrepancies (Eh), Disp20 / Exch-Disp20:
  water-Ne geometry 1: 3.66e-19 / 4.40e-20;
  geometry 2: 1.42e-19 / 1.06e-20;
  ethane-water: 7.81e-18 / 9.13e-18.
  Both the non-HF regression and imported physical blocked-reference tests
  passed in this run. No GPU numerical/device-shadow test has run.

## Dedicated CUDA2 build proposal (not submitted)

Use exact upstream Git pin `d17bb76e` (present in local origin/main remote
tracking and pinned fork refs), a new environment without the conda Einsums1
package, separate build/install directories, and a fresh dedicated Psi4 stage.
The Python-only dispersion route needs no native cuEST algorithm or older
Einsums stage. SCF/DFHelper and non-dispersion work remain CPU in this first
implementation; do not describe it as a whole-run GPU implementation.

A CPU-only embers build job can configure CUDA arch 90 explicitly, compile
the CUDA-enabled E2 library/Python together, rebuild this branch's Psi4, and
run CPU/API probes without borrowing a GPU. Candidate request: 8 CPUs, 48G,
2h, gts-cs207-chemx, explicit embers; exact-resource test-only and parent
coordination still required. Then request the separately coordinated
one-H200/4CPU/32G/30min numerical gate only after build/import/ABI proof.
The known CUDA13.3/GCC14.3 prefix is a toolchain candidate, not a passing E2
configuration. No job or shared environment has been changed.

Parent approval received for **ONE CPU-only dependency/build job**, after
canonical CPU regressions are green and the source is committed/pinned:
8 CPUs / 48G / 2h / embers / gts-cs207-chemx, cpu-small subject to actual
test-only routing. This is not a benchmark rerun. Use isolated immutable
source/build/install/environment roots, explicit sm90/CUDA/GNU pins and
persistent job-owned scratch; TERM-safe honest outputs, no automatic retry,
inferno or shared mutation. Validate test-only/headroom, record the actual
QoS/account/job ID and send the parent a receipt. GPU-enabled import failure
on a CPU node must not be masked. CPU/API/header/ELF/import checks establish
preparation only. H200 GPU numerical gate still requires separate readiness.

Prepared `devtools/disp0/{environment.yml,condarc,cpu_build.sbatch,
cpu_build_payload.sh,cuda_header_abi.cc,runtime_provenance.py}`.
Syntax/AST/YAML/diff checks PASS; environment solve/build not yet attempted.
The recipe uses job-owned package/cache/env/build/install roots, no conda
Einsums1 or release Psi4 package, explicit native CUDA13 header/runtime check
(13030) and GCC14.3 frontends, E2 shared/Python build together, strict import
pairing and CPU-only tests. No native E2 unit-test suite is built in this
bounded recipe; used Python operations and physical signed comparisons are
the focused checks. TERM handling propagates failure and cannot write a
CPU preparation success marker after a signal. No retry or GPU submission.

## Read-only review follow-up before source pin

Review d0a5ab9b8243c1321ec536956c8a16bca found no signed-equation or
frozen-energy slicing defect, but identified three P2 gaps in the green
checkpoint. Retained here rather than treating the earlier tests as exhaustive:

1. Non-aligned DF/compute edges could retain nine shapes, exceeding a four-shape
   charge. DF extents now align to compute extents; a pure planner test includes
   the exact 93/47/24 reproducer. Both host work and graph packs charge all four
   possible main/final-edge shapes.
2. Final Far/Fbs construction matrices remained live with packed factors.
   Construction now uses bounded staging slabs, deletes its temporaries before
   packing, charges all eight stage arrays and includes the rank-one rows in
   packed-factor memory costs.
3. Direct exported `sapt_dft()` bypassed unsupported-mode validation. Both
   entrypoints now share validation using effective do_dft/do_disp/backend
   arguments, with direct-entrypoint fail-fast tests.

Expanded tests add main/edge/main reuse in one graph cache, forced non-aligned
DF caps, asymmetric frozen counts, zero-active/virtual spaces and A/B-reversed
geometry. Prior 15-test CPU pass remains valid for its checkpoint; revised
source must pass again before commit/pin and the single approved CPU job.
Incremental revised stage build launched as ba1079bc6. No submission yet.

Revised stage build ba1079bc6: PASS. Expanded CPU gate b1f7d8f24:
**26 passed in 21.17s, no warnings or skips**. Same focused command as above.
Results `build_disp0_cpu/reference-check-8JYWWupu/results.xml` retain 64 signed
comparison records across four physical cases, eager/default replay,
forced non-aligned DF cap and asymmetric freezing. Max |delta|:
Disp20 7.81e-18 Eh; Exch-Disp20 1.01e-17 Eh. A/B reversed native references
differ by 1.18e-17 / -8.05e-18 Eh. Also passed zero-active/virtual, one-cache
main/edge/main replay, lifetime/reset, dispatch guards, unchanged non-HF and
imported fragment reference tests. A separate deterministic planner property
check passed 1000 extent/cap cases with <=4 cached shapes.

Follow-up read-only review d8ebbed82e78c854667d91df47c1cc14c retrieved:
no source blockers in the four scoped files; all three prior findings appear
resolved. Advisory review is not a build/test or GPU correctness substitute.
No CUDA/device-shadow numerical evidence exists yet. CPU build submission
still waits for finalized committed source plus exact-route/headroom checks.

Finalized implementation/build-recipe source pin:
`1e25b5e89a0ec29e17e20c791d9f3d23f7ef3e86`, committed and pushed on
`saptdft_cuest_disp0` (be962a2cf, exit 0). No merge into a parent branch.
This pin represents the initial canonical scalar hybrid checkpoint, not a
passed GPU gate or F-SAPT GPU coverage. The implementation task is complete
only in that explicitly limited sense.

One-shot headroom/path check ba6538b83 and new detached-source/test-only
preparation b76687edd started. Planned immutable remote source:
`P/gits/psi4.saptdft-cuest-disp0-build-1e25b5e89a`; source moved by Git only,
without updating sibling checkouts or queued pins. No accepted job ID yet.
Live preflight ba6538b83: PASS, embers **29/50** slots used, main repository
resolves to `P/gits/psi4`, conda 24.11.3 available, new own source path free,
explicit CUDA13 nvcc present. Exact-route test-only still pending.

Exact-resource test-only b76687edd: PASS, route **cpu-small**, 8 processors,
2h/48G/embers/gts-cs207-chemx script. The test-only forecast number 14182567
is **not an accepted job receipt**. Immutable detached remote source verified
at `1e25b5e89a0ec29e17e20c791d9f3d23f7ef3e86`; only this new worktree created.
The one approved actual submission is now in flight as b15fcd7f7.
No GPU submission and no automatic retry.

### Accepted CPU build receipt

* **Job 14182584**, accepted by b15fcd7f7 (exit 0), submitted
  2026-10-19 12:35:23 Phoenix local time.
* Actual `scontrol`: account **gts-cs207-chemx**, partition **cpu-small**,
  QoS **embers**, 8 CPUs / 48G / 02:00:00, **no GPU TRES**.
  Initial state PENDING (Priority); no start-time estimate claimed.
* Immutable source:
  `P/gits/psi4.saptdft-cuest-disp0-build-1e25b5e89a`,
  exact commit `1e25b5e89a0ec29e17e20c791d9f3d23f7ef3e86`.
* Run root:
  `P/runs/saptdft-cuest-disp0/build-1e25b5e89a0ec29e17e20c791d9f3d23f7ef3e86-14182584`.
  Environment, package/cache, E2 source/build/install and Psi4 build all
  live under that new job-owned root.
* Persistent scratch:
  `/storage/scratch1/3/awallace43/saptdft-cuest-disp0/14182584`.
* Logs:
  `P/runs/saptdft-cuest-disp0/logs/disp0-e2-build-14182584.{out,err}`;
  terminal status file `logs/14182584.status`;
  accepted receipt `logs/submission-1e25b5e89a.receipt`.
* Parent receipt sent successfully with workmux. Exactly one actual
  submission; test-only forecast 14182567 is not a second allocation.
* Scheduler caveat: effective Requeue=1 default was observed. Payload never
  resubmits and refuses reuse of a partial build root. Asked parent permission
  to set Requeue=0 on this job only; **no job modification made yet**.
  Future recipes should explicitly use --no-requeue. Queued source is frozen.

This is CPU build/API/ABI preparation only. It does not establish actual CUDA
device, GPU shadow or cuBLAS numerical correctness. H200 gate and Phase 2 remain
unapproved/unpassed and no GPU job has been submitted.

### CPU build terminal failure (no retry)

Official state-change monitor b68b89e8d reports job **14182584 FAILED**,
job/.batch/.0 all exit **1:0**, elapsed **00:06:31** (step 00:06:26).
It waited pending until the 2026-10-19 23:04:45Z snapshot showed RUNNING;
23:09:48Z snapshot showed FAILED/ALL-TERMINAL. These are observation times,
not exact scheduler start/end timestamps. No requeue or job modification was
made by this agent, and no second build or GPU job has been submitted.

Bounded stdout/stderr/status inspection launched as bceb45947; exact cause
pending artifacts. The single build authorization is consumed; preserve
failed immutable source/environment/run roots and coordinate any next build
separately. CPU/API/ABI preparation has **not passed** merely because its
submission succeeded. GPU numerical/device-shadow gate and Phase 2 remain
blocked.

Bounded artifact inspection bceb45947: confirmed FAILED_OR_SIGNALLED with
`exit_code=1`, `term_received=0`; epilog confirms actual cpu-small / embers /
gts-cs207-chemx. Exact failure occurs in Einsums d17bb76e configure:
`cmake/Einsums_SetupCatch.cmake:37` calls `check_cxx_source_compiles`; its
try_compile cannot resolve `Catch2::Catch2`. This setup is unconditional even
when native tests are OFF. The fresh env omitted installed Catch2, selecting
the FetchContent source-backed alias. Local CPU baseline has installed
Catch2 3.16.0 and imported targets. The solve also selected CMake 4.4 rather
than local 4.3.2; this is recorded drift, not the sole assumed cause.

Local diagnostic:
`devtools/disp0/catch_config_probe` with GNU14.3/CMake4.3.2 reproduced the
same missing-target error for a source alias (exit 1) and linked the actual
installed Catch2 string-view symbol with imported target (exit 0).
This red/green result establishes the build-system issue independently of
CUDA. No scientific contraction or native dispersion algorithm was changed.

Prepared recipe-only correction: Catch2=3.16.0, explicit Catch2_DIR and
preflight link probe, CMake=4.3.2 for baseline comparability, and
`#SBATCH --no-requeue`. These changes affect only this local branch, **not**
failed immutable source/env/build/install roots. No new job submitted.
Any corrected dependency build requires fresh parent approval.

Additional failed-run provenance bafb39732 (read-only): header/link probe
reported `CUDART_VERSION=13030`, runtime version 13030, status 0; nvcc
13.3.73. Configure selected explicit CUDA13 compiler/toolkit, sm90 and
Python build ON. These are **CPU header/link preparation checks only**,
not CUDA device/runtime or numerical correctness.

Environment lock exists. Actual packages included Apiary 1.0.0,
pybind11 3.0.2, mimalloc 3.3.2, CMake **4.4.4**, GCC/G++ implementations
14.3.0 **build 20**; the local CPU oracle uses GCC/G++14.3.0 build18.
The version-only GCC pin did not freeze package build numbers; this drift is
retained rather than claiming identical toolchains. `Catch2_DIR` pointed to
CMake's `pkgRedirects`, confirming source fallback, not the installed imported
target used by the CPU oracle. No E2 native/Python install or fresh Psi4 stage
was completed.

Recipe-only correction pin `5245210a100b73c3c5b275d889ad2de3c51ec736`
was sent to parent with a fresh one-job CPU build approval request. No new
approval or allocation is assumed.
* Read-only Phoenix access preflight: PASS, exit 0 (bfefc7153).
  No remote branch worktree/build has yet been created and no job submitted.
* Read-only Phoenix dependency discovery: PASS, exit 0 (b99607f10).
  Confirmed `P/miniconda/envs/p4cuest_sapt` (Python 3.13.11) and cuda13
  toolkit exist. No Einsums checkout found at either checked conventional
  `P/gits/Einsums` or `$HOME/gits/Einsums`, and no `P/miniconda/envs/p4_einv2`.
  This does not exclude custom locations; do not treat the parent's old
  1.x measurement stage as a compatible ready build. Dedicated source,
  environment and native/Python install preparation are still needed.
