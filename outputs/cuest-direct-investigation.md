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
**Actual Psi4 C++ compilation and allocated-GPU numerical validation are not
yet complete.** The adapter must not be treated as a validated direct GPU route
until those pass. The Phoenix validation runner records exact hashes, runtime
linkage, GPU identity, per-test chemistry logs, JUnit and `validation.json`.

Existing stored-DF comparison build/job outputs remain immutable and are not
reused as evidence for this new C++ implementation.
