# Cachy agent: Python/cuEST SAPT0 dispersion

## Ownership and objective

Continue this branch on **Cachy**, using its GPU for testing instead of waiting
for Phoenix. Own this checkout and its dedicated sibling setup/build/run roots.
Preserve existing checkouts, environments, user desktop processes and data.
Use Git commits for code transport and fast-forward pushes; no automatic merge
into master or another agent's branch. Phoenix jobs/allocations are out of scope.

Goal: correct GPU SAPT0 Disp20 and Exch-Disp20 through Python orchestration,
cuEST DFMO, cuBLAS and small CUDA kernels; only after the signed GPU numerical
gate, expand to all F-SAPT terms. Greater-than10x speedup is a goal, not evidence.
The user explicitly replaced the earlier CuPy/E2-only proposal with custom
CUDA kernels. **CuPy is not a dependency.** Existing CPU/E2/default/non-HF
dispatch must remain correct.

## Start here, in order

1. Read sibling `../metadata/SETUP_STATUS.md`, `../metadata/source-pin.txt` and
   `../metadata/setup.env` when present. They are the setup receipt, not a
   claim of numerical readiness. Compare the recorded source pin with HEAD;
   inspect Git status before changing files. Keep generated artifacts outside
   the checkout so source-version checks remain clean.
2. Read `devtools/disp0/cuda_backend.md` and the CURRENT section at the top of
   `disp0-progress.md`. Older sections are preserved history. Read
   `devtools/disp0/dfmo_device_plan.md` only for factor/gauge/layout reasoning:
   its original E2/CuPy proposals are superseded.
3. Verify the actual machine/dependencies from the setup receipts. Finish any
   explicitly incomplete private dependency/build steps, recording commands,
   source/package hashes, compilers, architectures, library maps and failures.
4. Implement/audit the missing native handle/basis/plan factory and runtime
   admission before claiming the GPU path can run. Run the small correctness
   gate below before performance work or F-SAPT expansion.

### Completion criteria

Report these milestones separately: dependency setup; clean staged Psi4 import;
used API/ABI checks; real GPU kernel/DFMO execution; signed physical gate;
term/fragment coverage; matched timings. A passed CPU fixture is not GPU proof.
Update `disp0-progress.md` and the sibling setup receipt with actual outcomes.
Retain failed artifacts and exact discrepancies.

## Known source state

Source checkpoint before this handoff: `dd0aa72485d8f0deb5f0d97d44340302617742cd`.
The handoff/setup commit may be newer; the sibling source-pin receipt is exact.

* Base Python route: `satpdft_einv2` at
  `1592851675cbd8ce6898eca969976a6e7becdb24`.
* Optimized blocked reference imported FIRST, cherry-pick-x of
  `907f6d75690c071dd9b452e3f0a82abb7ba317ea`, milestone
  `fb430094cc48d55c3479dae0040276c6c6dc6dd1`.
* Current explicit experimental entrypoint:
  `sapt_jk_terms_ein.dispersion0_cuda(..., provider_factory=...)`.
  It reuses canonical E2 setup, bypasses DFHelper, and shares component placement.
  There is **no** automatic `SAPT_DFT_MP2_DISP_ALG=CUDA` option.
* CUDA/E2 selection conflict, HF/ordinary dispersion, C1, frozen-space,
  fragment/S-inf and zero-size cases have explicit guards. Keep unsupported
  cases fail-closed.
* Files: `sapt_cuda_runtime.py`, `sapt_cuest250_identity.py`,
  `sapt_cuest_dfmo_cuda.py`, `sapt_dispersion_device_common.py`,
  `sapt_dispersion_cuda.py`, `sapt_dispersion_cuda_kernels.py`
  under `psi4/driver/procrouting/sapt/`.

Latest local evidence:16 CPU serial-kernel-body/transport/setup tests,11
synthetic identity tests,58 build guards and96 legacy runtime guards passed.
Read-only scoped review cleared the fixes. The kernel bodies were compiled
through a **CPU C++ qualifier shim with one serial thread**, NOT NVRTC/CUDA.
No actual cuEST DFMO/native version getter/device/signed physical gate has run.
Historical native CPU signed checks (26 tests/64 comparisons, near1e-17Eh)
used an older d17 CPU runtime; rerun with the actual new installed stage.

## Dependencies: pins and portability

Use the dedicated roots recorded in setup.env; install/build privately there.
The user's Cachy GPU was identified as RTX3090,24GiB (expected capability8.6),
driver615.78.08; host has32GiB RAM/16 logical CPUs. Reverify actual CUDA device
UUID against NVML before testing. Desktop compositor/application contexts are
expected: leave them intact and record co-tenancy. Start with small dimers,
bounded blocks and modest memory. Prefer2 build jobs; use at most4 on this
32GiB desktop unless measured memory permits more. No protein campaign.

* **cuEST0.2.50 ONLY**, CUDA13 native library. Canonical archive206185073 bytes:
  `a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d`.
  Native `lib/13/libcuest.so.0.2.50`:
  `65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c`.
  All59 headers must match; include-relative NUL-stream aggregate:
  `3d26a7ef607d02ee294f66f11f1374f3badbd83430f5be28ed0a05e6c8d73d18`.
  Exact header/native/ABI/map checks live in `sapt_cuest250_identity.py`.
* CUDA13.3/runtime13030/NVRTC13.3 and GNU14.3 are the current prototype's
  explicit ABI target. They are candidate build/runtime combinations, not a
  new-machine validation claim. Capture real versions/ELF/maps. Any deliberate
  target change requires reviewed source/config changes plus fresh evidence,
  not disabling checks or silently using CUDA12.
* **Einsums2 fork** `https://github.com/Awallace3/Einsums`, exact
  `06404fa69ce9346443d7d4f19a250f6db9f0f62c` (`perf/fdisp0-hotpaths`).
  Stock d17 is NOT this special branch. Keep native/Python versions, Python
  ABI and build prefix coherent. Build CUDA for **86**, not Phoenix's90.
  Apiary1.2 source generation needs matching Clang/LLVM23 build tools.
  Use native whole Python installation, including matching Waggle, not overlays.
  Set `EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1` BEFORE imports.
* GPU LibXC7.1.2: verify CUDA-enabled build/source and actual architectures;
  a conda CPU libxc package is not proof of GPU support. Keep CPU oracle and
  GPU artifacts distinguishable. HF dispersion does not exercise LibXC.
* `cuda-bindings` is the optional Python CUDA runtime/driver/NVRTC dependency.
  Pin its actual package/artifacts in the new environment lock. Direct cuEST
  C ABI avoids a separate unverified `cuest.bindings` package dependency.

Always import matching staged Psi4 core before E2. Use that build's `--psiapi`.
Never mix an old core with a new Python overlay or downgrade E2 to1.x.
Ensure complete Git history and the real annotatedv1.11 tag BEFORE configuring:
tag object75fa0e14aa16a868a00eced50eaaa2f8658aa806, peeled commit
f16f4ea2ff2351cc2ab877d9b3887fcf81c2a670. The previous Phoenix install
compiled successfully but generated `undefined` metadata from shallow/no-tag
source and correctly refused import. Use `source_version_guard.py` before
dependency work. Do not copy the Phoenix-hardcoded payload onto Cachy.

## Native factory and correctness gate

The prototype requires an **explicit audited same-process live handle/plan**,
exact primary/RI basis identities, persistent owners, total declared native
allocation bytes, output layout and frozen-hash CUDA closure manifest. No
hidden handle is created merely to query version. Getters have signature
`status(handle,uint32_t*)`; verify actual0/2/50 and mapped symbol ownership.

Factory configuration: full Coulomb fraction1/range-separated fraction+omega0,
**MATRIXPOWER1**, relative conditioning1, cutoff1e-12. SDK defaultQR is not
the intended policy. Prove relative rank/negative-eigenvalue behavior against
Psi4. Keep one metric gauge and auxiliary ordering across all10 factors/slabs.
Do not whiten already metric-transformed outputs twice. Preserve independent
RI pure/cartesian conventions, corrections, occupied-fastest packing, crossed
W permutation and signed prefactors4/−2. No denominator clamp.

Shared future contract/provider references (available through Git, not merged):
`a323afa886b6753370d798aa9ae5410b83f4aa06`, handoff
`9d6d3d40cddb2eded816da66f6c27855cea835b2` on `saptdft_cuest_disp`.
Inspect exact commits before any minimal port. They have offline tests only,
not vendor/CUDA/device certification; this branch's native Psi4 build did not
enable cuEST merely because its CUDA/E2 dependency proof passed.

Gate: independent native FISAPT, eager/default CPU and GPU signed Disp20 and
Exch-Disp20, plus occupied-pair/global sums. Several dimers/geometries/bases,
unequal spaces, frozen core, blocks1/5/500/edge cases, changed-input/repeated
calls, owner lifetimes and supported symmetry cases. Existing declared kernel
tolerances rtol1e-11/atol1e-12; physical components rtol1e-8/atol1e-8Eh.
Retain disagreements; no tolerance relaxation to cover environment failures.
Record actual successful cuEST/CUDA/cuBLAS execution, UUID/PID, peak memory,
source/native/Python/library pins, commands and pass/fail/skip.

Only after that gate: enumerate all F-SAPT terms and fragment conventions,
then implement missing GPU coverage. Separate diagnostic/kernel timing from
fresh-process whole-call wall including preparation/transfers/CPU work/cleanup.
Compare identical HF science and work. Desktop3090 results are not Phoenix
H200 speedups and parent historical PBE0-D4 timings are not this HF baseline.
