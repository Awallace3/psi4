# Direct cuEST J/K implementation and validation record

## Verified SDK contents

User-supplied archive:
`/projects/cos-lab-cs207/ds/cs207/cuest_v0.2.50.tar.gz`

SHA256:
`a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d`

The archive provides both CUDA 12 and CUDA 13 shared/static libraries, headers
and direct-JK examples. Headers/examples/libraries remain private, outside git.
Both shared variants export:

- `cuestDirectIntPlanCreate`, `cuestDirectIntPlanCreateWorkspaceQuery`,
  `cuestDirectIntPlanDestroy`;
- `cuestDirectCoulombCompute`, `cuestDirectCoulombComputeWorkspaceQuery`;
- `cuestDirectSymmetricExchangeCompute`,
  `cuestDirectSymmetricExchangeComputeWorkspaceQuery`.

These are conventional four-center direct integrals, not a smaller DF tensor.
The direct plan accepts the exchange operator
`alpha/r + beta*erf(omega*r)/r`, including short-range exchange through negative
beta when alpha and alpha+beta remain nonnegative. J stays full-range.
The exchange API requires symmetric densities. No nonsymmetric direct K was
found in the supplied interface.

The SDK version is 0.2.50 even though these headers label the APIs “since
v0.3.0”; build capability is tested through declarations **and linkage** rather
than version comparisons. New/old releases share `libcuest.so.0`, so merely
compiling against newer headers is insufficient: runtime loading must select
the same SDK. The CUDA 13 variant also needs the matching CUDA solver, BLAS,
sparse, NVRTC and nvJitLink libraries.

## Implementation

- Separate `cuESTDirectJK`, selected by `SCF_TYPE=DIRECT` with `USE_CUEST=True`.
- Borrows Psi4's primary AO basis; owns plan/pair list and their persistent
  storage, compute parameters and temporary GPU/scratch buffers.
- Implements JK's existing AO/USO adaptation and cuEST weighted-K convention.
- Rebuilds captured operator parameters after changes; cleanup is idempotent.
- Checks CUDA/SDK failures; rejects unsupported nonsymmetric exchange.
- Preserves existing DF selection and CPU DIRECT when cuEST is disabled.
- Compile/link feature probe keeps old DF-only SDKs usable; unavailable direct
  requests raise rather than silently falling back.
- `CUEST_DIRECT_JIT=AUTO|ON|OFF`; strict FP64 tests use OFF.
- Explicit linked-SDK RPATH precedence prevents an old conda SDK winning by
  shared SONAME.

See `devtools/benchmarks/cuest_direct/README.md` for usage, test instructions and
the required NVIDIA dependency notice.

## Evidence so far

Local, chemistry-independent harness: **110 tests passed**, including actual
CMake compile/link capability fixtures (new SDK, missing declarations, missing
symbol, switching SDKs in the same build cache) and runtime-loader regressions
for LibXC and cuEST with conflicting SONAMEs.

New GPU-test Python and shell entry points pass syntax checks.
Phoenix CPU build **14080574** (`gts-cs207-chemx/cpu-small/embers`, 8 cores,
48 GiB, 2-hour limit) completed with exit 0 in **9m38s**. The staged Psi4 imports,
the direct API compile/link probe is true, and runtime linkage resolves the
isolated cuEST 0.2.50 CUDA 13 library plus CUDA-enabled LibXC.

- Compiled source: `7896a4bd12905b050f0ba2f70a4c40d787b509b1`.
- Core SHA256: `311f1be6fd80cfad502f1605e48d8b7ecf66f966e9e7a535efd6f3f0630f39b3`.
- Linked cuEST SHA256: `65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c`.
- Run: `/storage/project/r-cs207-0/awallace43/runs/cuest-direct/20261015-7896a4b-r1`.
- Manifest: `build-stage/build-manifest.json` beneath that run.

**Allocated-GPU numerical validation passed on 2026-10-16.** Free H100 job
**14081007**, `gts-cs207-chemx/gpu-h100/embers`, completed 0:0 in **2m06s**:
**23 tests passed, zero failures, zero skips**. `validation.json` reports
success and the JUnit file independently confirms all 23 cases. The actual
device was an NVIDIA H100 80GB HBM3. The 30-minute allocation requested one GPU,
four CPU cores and 32 GiB host RAM. No paid GPU job was submitted.

The GPU harness source is pinned independently to
`f4289f3debe49fe78f24cfd00d0c815b6413850a`; the compiled source/core remains the
immutable build above. Test output is `validation-h100-r1` beneath the
same run, and logs are `logs/test-h100-14081007.{out,err}`. H100 timings must not
be mixed into H200 performance comparisons.

The six ordinary DFT/cc-pVDZ energy tests (SVWN, PBE, M06-L, B3LYP, WB97X and LC-WPBE)
passed the CPU DIRECT comparison threshold of 5e-9 Eh with host XC isolating
JK. Cartesian/spherical, UKS, density-batch and weighted-exchange, J-only/K-only,
zero-density, operator-reset and nonsymmetric-density checks also passed.
The separate GPU-XC smoke explicitly required a CUDA LibXC timer. Five existing
DF-JK regressions passed. This is small-system correctness validation, not a
matched-grid CPU/GPU XC comparison or large-system direct performance test.

GNU time reports **2:00.89** for the whole Python validation process and
**1,336,976 KiB** maximum RSS (~1.275 GiB). The SLURM allocation duration and
pytest-only duration (79.04s) are different clocks, not per-energy timings.
No sampled GPU memory high-water series was recorded by this correctness
runner; do not substitute the device's capacity for measured GPU usage.

The finalized Cedar handoff's original pending-status snapshot remains intact.
New artifacts and a separately checksummed addendum are under:
`/projects/cos-lab-cs207/ds/awallace43/share/cuest-pyscf/updates/direct-h100-14081007`.

The Phoenix validation runner records exact hashes, runtime
linkage, GPU identity, per-test chemistry logs, JUnit and `validation.json`.

Existing stored-DF comparison build/job outputs remain immutable and are not
reused as evidence for this new C++ implementation.
