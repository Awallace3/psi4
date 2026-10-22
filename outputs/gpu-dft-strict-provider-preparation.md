# Strict direct/direct preparation: source is not execution

## Reconciled state

- Comparison base: `81fe27657750039346aceca94ee7cc3df55a6ff7`.
- Reviewed provider source: `a323afa886b6753370d798aa9ae5410b83f4aa06`
  (handoff `9d6d3d40cddb2eded816da66f6c27855cea835b2`).
- Job14254253 was checked once on resume: PENDING/Priority, H200/embers,
  elapsed zero, no start/end, no result directory and empty logs.
- Its clean frozen checkout is still `86feda59ec84e82eb8c90685c2d9d2c8fbc09237`;
  submission receipt and peptide/Protein83 geometry hashes match `progress.md`.
- User chose **keep that job; prepare the strict build**. No cancellation,
  queued-checkout mutation, allocation reuse or duplicate GPU submission.

## Focused source integration

Port native version provider, checked stream setters and parameter RAII from
the reviewed source. It checks statuses for the verified
`status(handle, uint32_t*)` getter ABI and exact `(0,2,50)` with a live
Psi4-owned handle. Early presence checks do not initialize or query CUDA.
Initialization version failures unwind handles and permit retry.

Port exact CMake SDK admission, whole-header/native hashes, selected CUDA
closure, outer/inner identity forwarding, and post-build consumed-header audit.
Preserve this branch's compile/link direct-JK capability detection, exported
direct definitions, grid exporter and cuEST/CUDA-LibXC RPATH precedence.
No FDDS/DFMO numerical algorithms or scientific policy changes are integrated.

Canonical hashes and the required API symbol lists remain in
`devtools/benchmarks/gpu_compare/cuest-sdk-0.2.50.json`. CMake's SDK identity
schema is distinct from the comparison SDK identity schema; do not substitute
one receipt for the other. Adding CMake's Python contract does not wire its
runtime admission into unrelated FDDS/grid drivers.

## Fresh CPU build preparation — not submitted

The user subsequently authorized executing setup and then submitting GPU
testing. The implementation now includes `gpu_compare/strict_setup_job.sh`:
one free `embers` CPU job captures static CUDA12 candidate evidence and builds
the strict CUDA13 core in a separate run directory. Resource preflight accepted
`cpu-small`, 8 CPUs, 64G, four hours. This is a requested limit, not an ETA.
Submission/job receipts will be recorded in the actual run directory.

Capture uses `python -S -B` to avoid executable site hooks and shared bytecode
writes. Its candidate records resolved files/ELF SONAMEs, hashes, competing
providers, local wheel/conda ownership and installed loader sources; it is
explicitly **not an approved receipt**. Independent review remains mandatory.
Setup TERM tests cover prelaunch, inter-step, launch-assignment and late-receipt
cancellation; interrupted evidence is preserved without a success marker.
Phoenix SLURM executables use the explicit `/opt/slurm/current/bin` directory,
since the isolated worker PATH excludes it.
The hardened setup/gate workflow passed **292 offline tests** (212 comparison,
79 provider, one direct harness), shell syntax and diff checks; evidence is
`.pi/tasks/01a162c2-c287-766d-a381-64f22ecfda3a-2808822/b87ceed64.output`.

1. Finish local offline checks and review; commit a new comparison source pin.
   Push/fetch by git into a **new** Phoenix checkout pinned at that commit.
   Do not advance the checkout used by job14254253.
2. Pick a new unique run directory beneath
   `/storage/project/r-cs207-0/awallace43/runs/gpu-dft-direct/`, with `logs/`
   created before submission. Record source pin, command and input identifiers.
3. Validate one CPU build route using `sbatch --test-only`, explicit account
   `gts-cs207-chemx`, QoS `embers`, <=8h, and appropriately bounded CPU/memory.
   The job must forward TERM and run its child with `wait`; never build on a
   login node or in an old cache. No paid inferno authorization is implied.
4. Run the following launcher **inside the authorized build job**, supplying
   absolute pinned repo/new-run paths and the actual commit:

   ```text
   /bin/bash devtools/benchmarks/gpu_compare/build_grid_export.sh \
     ABS_REPO COMMIT ABS_NEW_BUILD_RUN embers \
     /storage/project/r-cs207-0/awallace43/software/cuest-0.2.50-a06e8148 \
     /storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
   ```

The sixth argument is the **resolved CUDA runtime prefix**. A lightweight
read-only path inspection found the CUDA13 shim's cudart/cuBLAS/cuSOLVER files
resolve into `p4cuest_sapt/targets/x86_64-linux/lib`, not under the shim.
This is only path evidence; the build must verify every resolved CUDA component,
canonical SDK headers/library before and after compilation, actual compiler
header dependencies, compiled direct-JK capability and staged core hash.
No environment/toolkit edits are required or authorized by this plan.

## Independent CUDA12 inventory preparation — not captured/reviewed

Keep this separate from both Psi4/CUDA13 and benchmark acceptance:

1. In a new setup-output directory and bounded authorized compute job, use
   `gpu4pyscf_compare/bin/python`, `PYTHONNOUSERSITE=1`, empty `PYTHONPATH`,
   no `LD_PRELOAD`, and the same sanitized CUDA12 loader setup as
   `cuest_direct/recovery_job.sh`. Do not import chemistry/GPU packages on a
   login node and do not inject cuEST/CUDA13 paths.
2. Record Python prefix, package locks/versions and static ELF dependencies
   for CuPy/GPU4PySCF/cuTENSOR. Resolve actual library paths, SONAMEs, package
   origins and SHA256 hashes. Module `cuda/12.6.1` alone is not identity proof;
   previous wheel-runtime observations reported CUDA12.8.
3. Form a **candidate**, not automatically approved, schema-1 inventory:

   ```json
   {"schema": 1, "cuda_major": 12, "python_prefix": "/absolute/environment",
    "libraries": {"libcudart.so.12":
      {"path": "/absolute/resolved/library", "sha256": "64-lowercase-hex"}}}
   ```

   Include every CUDA user-library the actual workers may map, including lazy
   dependencies. Keys must be actual SONAMEs and agree with resolved file ABI
   families: CUDA12 uses cuSOLVER11, cuBLAS12 and cuSPARSE12. Do not relabel
   CUDA13 binaries via symlinks. Site driver `libcuda.so.1` is separate.
4. Independently review candidates against ELF/installation evidence and the
   ABI rules in `gpu_compare/cuest_sdk.py`; reject mixed/unknown origins and
   ambiguous duplicate providers. Save review evidence and an independent
   SHA256 of the immutable approved JSON. Do not regenerate an approved
   receipt from whatever libraries happen to map in a benchmark worker.
5. Pass its absolute path and reviewed digest via
   `GPU_COMPARE_CUDA12_RECEIPT` and `GPU_COMPARE_CUDA12_RECEIPT_SHA256`.
   Missing/unpinned mapped components must fail closed, never extend the
   allowlist automatically.

No approved inventory, native strict core or GPU success is claimed here.
Only after these prerequisites and a separate resource decision may the new
serialized eigensolver/RKS/grid gates and direct/direct comparison proceed.
Keep FP64, direct J/K, consumed-grid equality and strict absolute `1e-7`
overlap policy unchanged. Protein157 remains a later feasibility/authorization
decision; historical DF timings are not direct/direct results.

## Local verification

Baseline reproduced 191 offline tests before integration. The selective
integration's 79 provider/admission tests passed and a separate read-only
source audit reported no actionable bugs. Python AST checks, launcher syntax
and `git diff --check` passed.

The complete serial offline gate **passed 272 tests**: 79 provider/admission,
192 comparison, and one direct-harness test. Launcher syntax and diff checks
also passed. Gate evidence:
`.pi/tasks/01a162c2-c287-766d-a381-64f22ecfda3a-2808822/ba67fb6b7.output`.

Two earlier attempts encountered compiler/CMake and/or unchanged containment
timeouts while the 48-thread host's load average was about137. Those failures
are retained in `bf30e153c.output` and `bd30dd5e6.output` under the same task
directory. All nine affected checks passed focused serial reruns
(`b8301739b.output`), then the full gate passed without relaxing thresholds.
A final process snapshot found no surviving gate/compiler commands. Evidence
supports transient host contention rather than a deterministic code regression;
the earlier failures are not erased or accepted as passing runs.

No native build or GPU execution has occurred.

## First GPU allocation

After the real build and inventory review, use `recovery_job.sh`'s optional
ninth argument `gates` (default remains `panel` for existing callers).
Gate-only mode performs the real peptide eigenprobe and RKS/grid/direct
preflight, then writes `GATES_COMPLETE.json` with `benchmarks_run: false` and
returns before the timing panel. Planned shape: one H200, eight CPUs, 128G,
two hours, free `embers`, with fresh hardware/driver/QoS receipts.
If old job14254253 is still active, serialize with `afterany:14254253` and
`--kill-on-invalid-dep=yes`; do not cancel or modify it. Check immutable
artifacts, not a successful scheduler exit alone.
