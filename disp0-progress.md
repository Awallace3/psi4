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
* Initial scalar Phase 1 implementation checkpoint `1e25b5e89a` is
  **CPU-validated: 26 passed**, 64 signed physical comparisons retained.
  CUDA numerical/device-shadow correctness remains unvalidated.
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
* GPU test is now user-authorized but build-blocked. CPU job14182584 failed
  Catch2 configuration; corrected job14193865 passed that point and linked
  the native E2 library, but Python codegen failed on Apiary/Clang missing
  `stddef.h`. No installed coherent E2 Python/Psi4 stage is ready and **no GPU
  job has been submitted**. Failed roots are preserved; no further CPU retry
  is implicitly authorized.

Parent scope agreement: Phase 1 is explicitly **GPU contractions with CPU
elementwise/reductions**, not fully GPU-resident dispersion. Use default
`Graph.execute` only in the new local kernel route; do not globally change
existing executor semantics. No new C++ dispersion or elementwise algorithms.
Test packed/strided layouts, pool/view lifetimes, repeated replay and shadow
freshness against signed components. Account for host work and transfers in
end-to-end timings. After the real GPU numerical gate, inventory remaining
F-SAPT GPU API gaps and propose a Python-level approach before expanding.

## Launch-base dispatch and GPU capability review

* At launch pin1592851675, `energy("sapt(dft)")` + `SAPT_DFT_FUNCTIONAL=HF` correctly skips
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

## Corrected CPU build authorization and preflight

Coordinator explicitly relayed fresh approval for **EXACTLY ONE** corrected
CPU-only build at `5245210a100b73c3c5b275d889ad2de3c51ec736`, 8 CPUs / 48G /
2h / free embers / gts-cs207-chemx. This is one authorization, not permission
for an extra submission if already acted upon. At receipt, this agent had
submitted only failed job14182584; **no corrected job had been submitted**.

Require fresh passing exact-route test-only, live headroom, explicit
--no-requeue and new detached source/env/build/run roots. Preserve all
14182584/1e25b5e89a evidence, including actual CMake4.4.4 / GNU14.3 build20
versus CPU-oracle build18 drift. No automatic retry, inferno or GPU submission.

Started corrected-branch publication b4eae7c65 and fresh headroom/duplicate
guard be1c8d608. Intended new source
`P/gits/psi4.saptdft-cuest-disp0-build-5245210a10`; receipt must not already
exist at `logs/submission-5245210a10.receipt` before any actual submission.
No corrected accepted job ID yet.

Corrected preflight be1c8d608: PASS, live embers **26/50** slots used.
Corrected source path/receipt were absent before staging, no active
`disp0-e2-build` job was reported, and the original failed receipt is present.
Branch publication b4eae7c65: PASS. Exact corrected test-only b2fe53fb3 pending.

Corrected exact-route test-only b2fe53fb3: PASS, **cpu-small**, 8 processors.
Detached source verified at the exact approved 5245210a full SHA and committed
--no-requeue verified. Forecast number 14193846 is test-only, **not an actual
allocation**. One corrected submission is now in flight as bb8c805ec, with
fresh receipt guard and a second active-build duplicate check immediately
before sbatch. Explicit --no-requeue also passed on the actual command line.

### Accepted corrected CPU build receipt

* **Job 14193865**, accepted by bb8c805ec (exit 0), submit timestamp
  2026-10-19 19:18:04 Phoenix local time.
* Exactly approved source pin:
  `5245210a100b73c3c5b275d889ad2de3c51ec736`,
  detached at `P/gits/psi4.saptdft-cuest-disp0-build-5245210a10`.
* Effective `scontrol`: **gts-cs207-chemx / cpu-small / embers**, 8 CPUs,
  48G, 02:00:00, **Requeue=0**, Restarts=0 and **no GPU TRES**.
  Initial PENDING (Priority); no start-time estimate claimed.
* New job-owned root:
  `P/runs/saptdft-cuest-disp0/build-5245210a100b73c3c5b275d889ad2de3c51ec736-14193865`.
  Source/env/build/install/cache roots are distinct from failed job14182584.
* Persistent scratch:
  `/storage/scratch1/3/awallace43/saptdft-cuest-disp0/14193865`.
* Logs `P/runs/saptdft-cuest-disp0/logs/disp0-e2-build-14193865.{out,err}`;
  terminal artifact `logs/14193865.status`;
  accepted receipt `logs/submission-5245210a10.receipt`.
* Parent receipt delivered. Exactly one corrected actual submission.
  Official state-change monitor b130435b3 started; no automatic retry.
  Failed14182584/1e25 roots and evidence remain untouched.

Acceptance is not build success or CUDA correctness. No H200/GPU numerical
job or broader F-SAPT work is authorized/started by this receipt.

## GPU-test authorization

User subsequently explicitly authorized the GPU test. Apply the previously
agreed bounded shape: one H200 / 4 CPUs / 32G / 30min, free embers /
gts-cs207-chemx. This does not waive corrected-build readiness, exact-route
test-only/headroom, matching native/Python import/ABI checks, actual device
identity or signed numerical/trace evidence. Do not finish an unready build
inside this allocation, borrow a holder, use inferno or start broader F-SAPT.
First inspect corrected job14193865 and its strict preparation artifacts.

Readiness inspection b56d9f49b: corrected build14193865 is **RUNNING** on
atl1-1-02-005-24-2, elapsed 00:05:35 at snapshot; job/.batch/.0 still running.
No terminal status, runtime-provenance.json, api-probe.json or staged Psi4
executable existed yet. The GPU gate is authorized but **not build-ready**;
do not allocate a GPU merely to wait or finish this build. Official watcher
b130435b3 remains responsible for terminal notification.

While the corrected build runs, prepared standalone GPU gate files
`devtools/disp0/{gpu_gate.py,cublas_trace.cc,prepare_gpu_gate.sh,
gpu_gate_payload.sh,gpu_gate.sbatch}` and parser-only fixtures
`tests/pytests/test_disp0_gate_evidence.py`. Scientific `psi4/` files remain
byte-for-byte unchanged from build pin5245210a; the harness must import only
that prepared stage, never a source overlay.

Gate plan: native FISAPT plus eager/default CPU replay vs CUDA requests at
edges1/5/24/500, forced DFcap7, asymmetric freezing, changed/reversed/repeated
geometries, spherical/Cartesian bases and a nontrivial benzene dimer to obtain
naturally eligible blocks without changing the cost model. Explicit larger
production block for that case remains subject to the existing memory planner.
Real GPU replay tests strided input packs, main/edge/main reuse and live pool
reset rejection/release. Native CPU fragment partition sum is an oracle, not
GPU fragment coverage; unsupported fragment/S-inf requests must still reject.
Unchanged non-HF regression is repeated with the real CUDA runtime present.

Evidence plan: exact prepared binary hashes/single-world import, header-ABI
identity function compiled against CUDA13, logical CUDA UUID matched to NVML
UUID and this process/exclusive context (no physical-index-zero assumption),
default graph's observed `gpu_dispatch` annotations AND forwarded native
`cublasDgemm_v2` API/status records within physical/production phases.
The diagnostic interposer implements no mathematics and forwards arguments
unchanged. It must be compiled/ELF-checked **before** the GPU allocation.
Packing, transfers, CPU work and diagnostic/oracle time are labelled; no
speedup/full-GPU/F-SAPT coverage claim.

Gate scripts pass AST/shell/diff checks. Parser-only CPU fixture tests launched
as b45d86fc5; review d61de3e3d264a3ad3977373062ef1950d pending. The tracer has
not yet been compiled and the gate has not run on CUDA. No GPU submitted.
* Read-only Phoenix access preflight: PASS, exit 0 (bfefc7153).
  No remote branch worktree/build has yet been created and no job submitted.
* Read-only Phoenix dependency discovery: PASS, exit 0 (b99607f10).
  Confirmed `P/miniconda/envs/p4cuest_sapt` (Python 3.13.11) and cuda13
  toolkit exist. No Einsums checkout found at either checked conventional
  `P/gits/Einsums` or `$HOME/gits/Einsums`, and no `P/miniconda/envs/p4_einv2`.
  This does not exclude custom locations; do not treat the parent's old
  1.x measurement stage as a compatible ready build. Dedicated source,
  environment and native/Python install preparation are still needed.

### Corrected CPU build failed; authorized GPU gate remains blocked

Official watcher b130435b3 reports corrected job **14193865 FAILED**, job,
.batch and .0 all exit **1:0**, elapsed **00:17:51** (step00:17:42).
The terminal observation was 2026-10-19T23:38:47Z. No retry or job mutation
was made. Both immutable failed roots remain preserved.

Bounded stdout/stderr/status inspection launched as b6ef30efe for the exact
new blocker. The user's GPU-test authorization is retained, but is not a
waiver of coherent E2/Psi4 build/import/API readiness. **No GPU submitted**;
do not repair/finish a full dependency build inside the 30-minute H200 gate.
No third CPU allocation is authorized by either consumed one-job approval.

Parser-only CPU fixture task b45d86fc5: **11 passed in 1.33s**, no warnings
or skips. Results:
`build_disp0_cpu/reference-check-lOGVr7MK/results.xml`.
It verifies acceptance/rejection logic for native API records and observed
profile annotations, including rejection of placement-only/CPU-fallback,
wrong-PID, failed-status and nonphysical fixtures. These are deliberately
synthetic parser fixtures, **not CUDA execution or GPU numerical evidence**.

GPU harness review d61de3e3d264a3ad3977373062ef1950d found seven proof/control
gaps; addressing before any allocation: explicit native cuBLAS loader/link
preflight, attesting currently imported/mapped binaries and scientific modules
instead of only prepared filenames, actual cuBLAS on each repeated synthetic
main replay, dropping independent pool-output references before ownership
checks, observed CPU edge execution (not placement alone), published driver
signed/sum QCVariables, and EXIT status handling before fallible preflight.

Native forwarding now explicitly links cuBLAS and exposes a resolve-only
identity probe, still no numerical algorithm. Larger bounded synthetic main
matrices exercise naturally eligible CUDA replays without forcing the cost
model; small edges remain separate. Physical CPU-edge proof is scoped to
actual `ComputeGraph::execute(sapt0-dispersion-products)` profile events.
Additional negative parser fixtures and fake-scheduler parameter/controller/
QoS/signal tests launched as b2210c4b4; CPU QCVariable wiring is also checked.
These tests do not substitute for native compilation, loader preflight or GPU
numerics. Underlying E2/Psi4 build readiness is still blocked by job14193865.

Corrected-build artifact inspection b6ef30efe confirms the new exact blocker:
Apiary/Clang cannot find `stddef.h` while parsing Options.hpp through GNU14.3
`<complex>/<cmath>` and the conda sysroot's stdlib.h. `generate_rc` then fails
"no option descriptors found". The native E2 shared library linked at build
step378/520, but installed Python `_core`, Psi4 stage, runtime proof and API
proof are all absent. Status records `exit_code=1`, `term_received=0`,
FAILED_OR_SIGNALLED; epilog confirms cpu-small/embers/gts-cs207-chemx.
This is not a device runtime failure that a GPU allocation can solve.

Source audit: Apiary's CMake helper only locates its Clang builtin resource
directory through a Clang native compiler or `CONDA_PREFIX/bin/clang++`.
Native compiler must remain GNU14.3. CPU oracle has clangxx/Clang22.1.4 and
`lib/clang/22/include/stddef.h`; original build recipe did not explicitly
install clangxx or bind CONDA_PREFIX to its new env. Read-only failed-cache/
package inspection b32b6a119 is pending before confirming this environment
discovery hypothesis.

Prepared a stricter isolated Apiary standard-library/annotated-AST preflight.
Initial negative resource probes unexpectedly exited0 and retained the
annotated declaration; direct diagnostics revealed **Apiary exits0 and emits
partial AST even after a fatal missing-stddef error**. These failed diagnostic
assumptions are retained. The strengthened preflight rejects fatal parser
stderr as well as exit failures/missing AST, removes prior output and excludes
ambient host include rescue. It now gives local RED (missing resource,
exit1 with fatal stddef) / GREEN (matched Clang resources, native GNU14.3,
nonempty AST, exit0).

Prospective recipe adds clangxx22.1.4 for AST resources only, job-local
CONDA_PREFIX binding, this fail-fast preflight and fatal-parser diagnostics
checking on the full build log. Native GNU/CUDA pins and scientific source
are unchanged. No failed root has been modified and no third CPU allocation
or GPU allocation has been submitted.

Strengthened CPU gate-control task b2210c4b4 completed: **22 passed in
2.11s**, no warnings/skips. Command:
`bash devtools/disp0_test_reference.sh "$PWD/tests/pytests/test_disp0_gate_evidence.py" "$PWD/tests/pytests/test_disp0_gate_batch.py"`.
JUnit:
`build_disp0_cpu/reference-check-iqa4RZ30/results.xml`.
This covers synthetic trace/profile acceptance and rejection, actual CPU
driver QCVariable names/component sum wiring, and fake-scheduler early
failure/QoS/controller/TERM handling. It is separate from the 26-pass physical
CPU checkpoint and provides **no CUDA execution or GPU numerical proof**.
Follow-up read-only review da307e05df0e1659daaa9052fa847294b is pending.

Read-only failed-environment inspection b32b6a119 confirms the Apiary resource
discovery failure, rather than merely suggesting it:
`APIARY_RESOURCE_DIR` is empty; the dedicated env has no `bin/clang++` and no
Clang builtin `stddef.h`; `APIARY_EXTRA_ISYSTEM` incorrectly names the base
`P/miniconda/include`, not this job's environment. Native C++ is correctly
the job-owned GNU14.3 compiler. Failed job14193865 has Apiary1.0.0,
Catch2 3.16.0, **CMake4.3.2** and libllvm22 22.1.8. This distinguishes its
corrected CMake pin from failed job14182584's actual CMake4.4.4. The only
partial native result found is
`einsums-build/lib/libEinsums-abi1.so.2.0.0`; it is not an installed coherent
native/Python runtime and will not be overlaid with an old Python extension.
All inspection was read-only. Proposed Clang resources/CONDA_PREFIX/preflight
fix still needs review and a new committed pin before requesting any fresh
CPU build approval.

Follow-up review da307e05df0e1659daaa9052fa847294b retrieved: no further
concrete GPU harness blockers; one P2 in the CPU launcher: required parameter,
controller and QoS checks preceded its EXIT trap. Moved trap installation
ahead of those checks, added unset-safe status fields and status-write failure
handling. The fake-scheduler early-failure/TERM suite now exercises both
CPU and GPU launchers. Final CPU control rerun bfd547bbb and fresh Apiary
red/green preflight b1743f6f0 are pending. Static bash/diff checks pass;
`psi4/` source remains byte-identical to5245210a. Native CUDA tracer compilation,
loader preflight and actual GPU numerical execution remain untested.

Final CPU gate-control rerun bfd547bbb: **26 passed in2.10s**, no warnings or
skips, including all eight CPU/GPU fake-scheduler early-error and signal cases.
Command is the same two-file invocation above; JUnit:
`build_disp0_cpu/reference-check-9hsDyCjK/results.xml`.
These 26 control tests are **distinct from** the earlier 26 physical
dispersion/API tests and64 signed comparisons; do not combine the counts into
a GPU pass claim.

Final fresh Apiary preflight b1743f6f0: **PASS**. Deliberately missing
resource headers give RED exit1 with the expected fatal missing-stddef
diagnostic; matched Clang22.1.4 resources with native GNU14.3 give GREEN
exit0 and the required nonempty annotated AST. Artifacts:
`build_disp0_cpu/apiary-final-review/{red,green,red.log,green.log}`.
No failed remote root/environment was touched. Review findings are addressed
and the build fix plus gate controls are ready to commit/pin. This is local
parser preparation only, not proof that a fresh Phoenix solve/build or CUDA
runtime will pass. Another CPU build requires fresh explicit approval;
the already authorized GPU test stays blocked until matching preparation
passes. No automatic retry and no broader F-SAPT transition.

## Approved single additional CPU preparation build

User explicitly approved exactly one additional CPU-only build at
`587e172a33afaadb00eb7f574a5089533f7fc755`, 8 CPUs/48G/2h,
free embers/gts-cs207-chemx, cpu-small subject fresh exact-route test-only and
live headroom, explicit no-requeue, fresh isolated source/environment/build/
run/persistent scratch. Failed14182584/14193865 remain untouched.
Push b70e9aa83 completed successfully to origin/saptdft_cuest_disp0.
Source staging b333efc2f uses a new independent filtered Git clone and exact
detached clean identity; it does not advance a shared checkout or queued pin.
Live headroom inspection b03e6706f launched. Approval has not yet been
consumed by an actual submission. This is CPU preparation only; GPU
authorization remains conditional on matching stage/runtime readiness.

Source staging b333efc2f completed: new independent
`P/gits/psi4.saptdft-cuest-disp0-build-587e172a33`, exact587e172a33 full pin,
detached and clean; no shared checkout advanced. Headroom b03e6706f:
27/50 embers slots. Fresh exact-route/no-duplicate inspection ba11331a6
is pending; no actual submission yet. Clang/Apiary resource and linked LLVM/
libclang provenance must be retained from the newly solved job environment,
not inferred from the local red/green probe.

Exact-route preflight ba11331a6 passed at2026-10-20T00:14:37Z: source
identity/cleanliness verified, 27/50 embers slots and zero active dispersion
build duplicates. `sbatch --test-only` pseudo-ID14196325 selected
**cpu-small**, 8 processors; its predicted start is not an accepted job
or scheduling guarantee. Invocation explicitly requests account
gts-cs207-chemx, QoSembers, 8CPU/48G/2h and no-requeue, identical to the
actual single-attempt submission launched as b7a547b83.
Actual accepted ID/effective route are pending that receipt. The receipt
is created exclusively before sbatch; an ambiguous failure must be inspected,
not automatically resubmitted.

### Accepted additional CPU build receipt

* Exactly one actual submission accepted: **14196330** (b7a547b83, exit0).
* Source pin: **587e172a33afaadb00eb7f574a5089533f7fc755**, detached clean
  independent clone `P/gits/psi4.saptdft-cuest-disp0-build-587e172a33`.
* Effective controller route: **gts-cs207-chemx / cpu-small / embers**,
  8CPUs/48G/02:00:00, **Requeue=0**, Restarts=0, no GPU TRES.
  Initially PENDING(Priority); no actual start-time guarantee.
* Fresh job-owned environment/source/build/install/cache run root:
  `P/runs/saptdft-cuest-disp0/build-587e172a33afaadb00eb7f574a5089533f7fc755-14196330`.
  Persistent scratch:
  `/storage/scratch1/3/awallace43/saptdft-cuest-disp0/14196330`.
* Logs: `P/runs/saptdft-cuest-disp0/logs/disp0-e2-build-587e-14196330.{out,err}`.
  Terminal artifact `logs/14196330.status`; submission receipt
  `logs/submission-587e172a33.receipt`; preflight receipt
  `logs/preflight-587e172a33.txt`.
* Official state-change monitor b7e91b661 is running. This approval is
  consumed; no retry or fourth CPU submission is authorized.
  Failed14182584/14193865 environments and evidence remain unchanged.

Acceptance is not successful preparation. At terminal state, require actual
clang/libclang/LLVM/resource paths and versions, full explicit environment,
header/ELF/import/API/physical checks and a coherent matching stage before
the separately authorized GPU gate. No GPU submission or F-SAPT expansion.
