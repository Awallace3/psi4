# Optimized Einsums2 branch — LOCAL source/provenance review

**Planning only. Real input submission UNKNOWN; approval RESERVED; all remote
queries/writes/builds/submissions/GPU actions remain STOPPED.**
No new execution authority follows from finding the correct dependency.

## Correct target and preserved lineage

The comparison agent identifies the special working CPU branch as:

* Fork:`https://github.com/Awallace3/Einsums`
  (`git@github.com:awallace3/Einsums.git`).
* Fork branch:`perf/fdisp0-hotpaths`.
* Exact target:`06404fa69ce9346443d7d4f19a250f6db9f0f62c`.
* Upstream base:`3835ef3b3777ec221ee59c68da8acb0417d7c29d`.
* Stock old base:`d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62`.
* Preserved old-base optimization branch:`perf/fdisp0-hotpaths-d17bb76e`,
  `10be5e391afbb66344b658116e5eddf76946d2f1`.

LOCAL Git object/ref inspection independently confirmed these four commits.
Stockd17 is NOT the special branch. Upstreamd17→3835 has330commits and1122
changed paths;3835→064 has6commits/12paths. This is not a small substitution
inside the old d17 build contract.

Important local alias trap: local`refs/heads/perf/fdisp0-hotpaths` is10be,
while`refs/remotes/fork/perf/fdisp0-hotpaths` and local
`perf/fdisp0-hotpaths-rebase` are064. Pin fork URL plus full commit, not a
local branch name. Existing source checkout
`/home/awallace43/gits/Einsums__worktrees/fdisp0-rebase` is HEAD064 with no
tracked working-tree modifications; it was read, not edited.

Six fork commits:

1. 952d3876: direct_division beta0 must not read destination.
2. 49b85ad3: outer_sum vectorization/threading.
3. 2bb18da7: fused batch-dot.
4. 03ba1e72: GeneralRuntimeTensor move constructor.
5. c82cd818: gemm_batch promotion.
6. 06404fa6: promotion gate/`einsums:packed-gemm:batch-promotion`.

Historical d17 CPU26tests/64signed comparisons remain d17 evidence. They do
not validate a newly built064 library or CUDA execution.

## Comparison evidence and corrected binary provenance

Agent reports all required Python symbols present in four existing local
builds (d17/10be/3835/064-like). All four have CUDA and HIP OFF.
Attribute presence is not layout/lifetime/device correctness.

The local064-like build is **not a clean commit-attested064 binary**:
`fdisp0-rebase/build` GitInfo recordsc82cd818a7d8947329298856edbf753973968e22,
DIRTY1; `_core` predates064. A later saved working diff resembles the gate
commit, but intervening edits cannot be excluded. Local upstream3835's
GitInfo also recordsDIRTY1, cause unknown. These cannot supply clean-build
or CUDA evidence, nor cp314 binaries for a cp313 Psi4 stage.

Agent's primary clean-source assertion is Phoenix build13922267 and81-run
CPU `_up` campaign. LOCAL copy of its script was read:
`/scratch/awallace43/ein_backend_bench/phoenix/scripts/build_up.sbatch`.
It asserts HEAD064 before configuring, with fresh build directories and
cp313/p4einv2. Actual Phoenix GitInfo was not inspected; no SSH added.
That campaign used C++ Psi4 e8478e2064 with driver overlays356be33609 or
1592851675, NOT our whole scientific587 stage. Do not silently transfer the
pairing/cleanliness/numerical claims.

## Build/packaging findings from committed064 source

Source-root prefix below:
`/home/awallace43/gits/Einsums__worktrees/fdisp0-rebase/`.

* `cmake/Einsums_SetupApiary.cmake`: annotation headers are unconditional;
  Python codegen needs the executable. Installed Apiary range1.2..<2;
  fallback1.2.0 archive SHA
  `336e6299cba2da8c42074a3ea3519f3f296f9e66c480714aa6ff3f9f64094c8a`.
  Source explicitly supports LLVM22/23; the comparison's working recipe
  uses separate build-only Clang/LLVM23.1.2. Old Apiary1.0 lock is not the
  new dependency solution.
* `cmake/Einsums_SetupWaggle.cmake`: Waggle pin
  `62a9fc5ffc9a2f3f5efb2aa6964bc2583dc932e6`; Python companion enabled with
  Einsums Python. Test flag comes from`EINSUMS_WITH_TESTS_UNIT`.
* `cmake/Einsums_SetupStripes.cmake`: Stripes pin
  `bc02978f33a902e233b2333c80ab1d6724ad9608`; portable SIMD dispatch and
  native-arch controls exist. Installed-package discovery may supersede
  source fallback, so record actual selected origins.
* `cmake/Einsums_SetupSpdlog.cmake`: source-pinned spdlog1.17.0, static,
  external fmt. Do not assume the old dependency closure.
* `cmake/Einsums_SetupAllocator.cmake`: accepted mimalloc header range>=3.3,
  <3.6 excluding3.4.0/3.4.1; source comments describe an arena-size fix.
  Agent's existing3.5.x pool-cap observation remains a reason to retain
  fallback/probe coverage, not declare behavior fixed from comments.
* `cmake/Einsums_FinalizePybind.cmake:436–465` now has **native whole-package
  install rules**, target installation of PyEinsums and a relative
  `$ORIGIN/...` path to native Einsums. Includes generated rc/stubs and
  package assets. This differs materially from d17's missing Python install:
  do not blindly reuse the d17 manual packaging hook.
* `CMakeLists.txt:973–996`: actual`EINSUMS_INSTALL_PYMODDIR` option;
  `naive` installs under`CMAKE_INSTALL_LIBDIR`; `prefix` selects interpreter
  layout within install prefix. Declare the chosen layout explicitly.
* `cmake/Einsums_SetFullRPATH.cmake`: default full-RPATH mode enables
  `CMAKE_INSTALL_RPATH_USE_LINK_PATH`. A new isolated install must control
  this deliberately and audit actual ELF RUNPATH/closure; relative Python
  RPATH alone is not proof against old-build/site12 dependency selection.
  Proposed configuration should disable`EINSUMS_WITH_FULL_RPATH` and set
  controlled per-target installed paths; setting only the cache variable
  for USE_LINK_PATH is insufficient if the module then sets it ON.
  Include the matching Waggle native/Python companion in package/map proofs.

The comparison script uses build-tree PYTHONPATH, no install, permits a
Waggle C-API test link failure with Ninja`-k0`, and creates driver overlays.
Those are historical CPU campaign mechanics, **not our acceptance policy**.
New required native/Python/stub/companion targets must have successful exits;
optional excluded targets need explicit inventory, not blanket ignored errors.
All new installs/environments would be fresh and private, with no shared
modification, old-core overlay or old failed-root reuse.

## CUDA configuration findings — source only

`CMakeLists.txt:349` defaults CUDA OFF.
`cmake/Einsums_SetupCUDA.cmake` enables native CUDA and CUDAToolkit targets.
It skips setup if`CUDA::cudart` already exists: do NOT precreate that target
and accidentally bypass backend/config defines.

Set the actual`EINSUMS_WITH_CUDA_ARCHITECTURES=90` option explicitly for a
future H200 build. Otherwise its default detection calls`nvidia-smi -L`;
setting only CMAKE_CUDA_ARCHITECTURES is not the same gate. No such probe,
configuration or compilation ran in this review.

CUDA13.3/GNU14.3/cuEST candidate conventions remain candidate toolchain inputs,
not proof of064 CUDA compatibility. Need complete matched development
headers/libraries, no site12 fallback, compiler/include/ELF provenance,
matching native+Python ABI and one whole matching Psi4 scientific587 stage.
Neither a cp314 local binary nor an E1 stage is a compatible shortcut.

## Runtime and validation requirements for a later proposal

* Keep Python orchestration/contractions on the Einsums2 route; no new native
  scientific dispersion algorithm.
* Set`EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1` before any Psi4/Einsums import
  if that is the reviewed policy. Post-init rc writes may be silently ignored.
  Both matched CPU/GPU arms must use the same reviewed option policy.
* Preserve default Graph.execute semantics and explicit placement evidence;
  CPU elementwise/reductions stay labeled as such unless actual APIs/device
  tests justify a later change.
* Repeat API/layout/view/reshape/pool/replay/freshness tests on the actual new
  installed064 build; retain pool fallbacks until proved safe.
* Rerun signed native/eager/default-CPU references and physical component/
  block/fragment sum checks on the coherent new stage before GPU acceptance.
* Actual eligible CUDA/cuBLAS execution and signed GPU comparison remain an
  unpassed separate gate. No phase2/all-F-SAPT claim or speedup inference.

Current input submission remains UNKNOWN/RESERVED. Old conditional CPU89
approval targets d17 and does not authorize this changed dependency build.
Only an offline revised proposal is contemplated, not a job/build request.
## Completed GPU surface source audit

Read-only delegate`d332a20366321b36567d38a9293525a2e` inspected the clean064
checkout and our helper. It ran no probes/builds/device tests.
Conclusion: source supports the existing packed Python hybrid seam; no new
C++ scientific dispersion algorithm is indicated. CUDA correctness is unproved.

* `libs/Einsums/ComputeGraph/src/Graph/Execute.cpp:103–117,230–506`:
  default replay owns GPU dispatch/transfers/fallback/host synchronization.
  `src/Executor.cpp:38–47,70–79,95–97`: SequentialExecutor runs closures
  directly and bypasses that GPU replay. Preserve our dedicated helper's
  default execution; do not globally change the general driver graph block.
* `src/Passes/GPUPlacement.cpp:42–161` and
  `src/Graph/GpuDispatch.cpp:570–588`: supported descriptor/layout subset,
  not arbitrary einsum. Our three packed rank2 contractions are appropriate.
  Ordinary captured linalg.gemm is not this descriptor path; pointer-array
  BatchedGemm is CPU-only. Preserve local protection against that batching
  hiding the GPU-eligible contractions.
* `libs/Einsums/GPU/src/BLAS.cpp:84,203`: committed double-precision vendor
  endpoints exist. Planned placement is not evidence of actual calls.
* `ComputeGraph/src/Optimizer.cpp:877–878,1008–1019,1053–1059` and
  `src/Passes/GPUPlacement.cpp:215–236,292–350`: natural placement includes
  device availability, transfer/launch cost and memory budget. Small CPU
  fallback is legitimate; require at least one naturally eligible actual
  CUDA/cuBLAS contraction later, without force-GPU claims.
* `src/Graph/GpuDispatch.cpp:60–111,209–214`: live dimensions and column-major
  strides`[1,rows]` matter. Owning packed host RuntimeTensorD/ViewD tensors
  are shadowed/transferred; device-tensor substitution is not required.
* `Tensor/RuntimeTensor.hpp:1184–1193,2112–2120` and
  `TensorImpl/TensorImpl.hpp:1498–1525`: reshape_view has stride-aware
  validation and rejects reshapes requiring a copy. The old blanket
  trailing-axis-only limitation must not be blindly asserted for064;
  neither does arbitrary reshape become GPU-packed.
* `BufferAllocator/MemoryPool.hpp:124–152`: pool is host mimalloc storage;
  its `pinned` meaning is retained host pages, NOT CUDA-pinned/device memory.
* `Graph/Execute.cpp:186–195,315–398,418–506`: per-replay shadow authority,
  host uploads, CPU-consumer synchronization and final flush are implemented.
  Repacking before each replay is appropriate; overlapping-view safety still
  requires actual tests, not a source-only guarantee.

Denominator outer_sum/division, occupied localization and rank4 reductions
remain CPU regions in our hybrid. `ComputeGraph/src/StridedBatchedEinsum.hpp:
99–145` requires equal operand ranks/common outer batch axes/canonical dense
layouts, so our rank3×rank2 localization is not GPU coverage.

All this is source inspection, not a CUDA build, runtime/device proof,
component numerical pass or speed claim. Source APIs and native package
installation are promising changes; the actual new coherent build, exact
dependency locks, CPU comparisons and real GPU gate remain outstanding.
Old d17-only build scripts/hooks/approval must not be silently relabeled064.
