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
* GPU dispersion: not implemented or validated.
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
* Read-only Phoenix access preflight: PASS, exit 0 (bfefc7153).
  No remote branch worktree/build has yet been created and no job submitted.
* Read-only Phoenix dependency discovery: PASS, exit 0 (b99607f10).
  Confirmed `P/miniconda/envs/p4cuest_sapt` (Python 3.13.11) and cuda13
  toolkit exist. No Einsums checkout found at either checked conventional
  `P/gits/Einsums` or `$HOME/gits/Einsums`, and no `P/miniconda/envs/p4_einv2`.
  This does not exclude custom locations; do not treat the parent's old
  1.x measurement stage as a compatible ready build. Dedicated source,
  environment and native/Python install preparation are still needed.
