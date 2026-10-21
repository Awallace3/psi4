# Experimental Python/cuEST CUDA dispersion — CuPy-free

User explicitly superseded the earlier no-custom-CUDA restriction for this
backend, selecting small CUDA kernels patterned after NVIDIA's MP2 example.
The CuPy prototype is preserved in Git history at5b161a5736, not retained as
an unchecked alternate execution path. No CuPy dependency/import is used.

## Mandatory production factor provenance

Production dispersion must use **cuEST0.2.50 DFMO → device-resident,
metric-transformed MO factors → cuBLAS/custom CUDA**, orchestrated from Python.
No DFHelper integral generation, transforms, packing or storage may feed this
route, including indirectly through cached factors or disk tensors. No NumPy
factor producer or implicit CPU/DFHelper fallback is permitted.

The actual explicit call chain is `dispersion0_cuda` → canonical setup callback
in `fdisp0` (returns BEFORE the DFHelper branch) → strict `CuESTDFMOProvider`
→ `cuestDFMOIntegralsCompute` into session-owned device output → device packing,
three cuBLAS products and signed CUDA reduction. Host arrays at this boundary
are orbital coefficients, energies and rank-one corrections, not DF factors.
Final occupied-pair components are downloaded for common placement/sum rules.
The factor consumer rejects host arrays or buffers from a different session.

Factory/native failure must propagate; the missing live native factory means
**BLOCKED**, never permission to substitute a host producer. Existing default
CPU/non-HF routes remain intact. An explicitly separate CPU/native FISAPT oracle
may use DFHelper only for correctness reference; neither its factors nor its
caches may feed production. Keep reference and production timings distinct,
and include host setup/transfers/cleanup in whole-call production timing.

CPU orchestration negatives poison cached DFHelper/host-factor entries and
exercise the public CUDA entrypoint. They also require factory/native failures
and wrong providers to reject without reaching the forbidden DFHelper sentinel.
These guards are source/CPU evidence only, not native factor provenance proof.

## Architecture / dependencies

* Python retains canonical SAPT orchestration and all setup equations.
* Existing cuEST0.2.50 DFMO produces bounded transformed factor slabs.
* Four generic packing launches combine Far/Fbs and rank-one corrections.
* Three FP64 cuBLAS GEMMs form V/X/W for a compute block.
* One CUDA kernel computes exact signed HF denominators and occupied-pair
  dispersion/exchange-dispersion reductions. No clamp, fast-math or atomic sum.
* CUDA source is embedded in an installed Python module, compiled with NVRTC
  only when an explicit session is constructed. Imports never initialize CUDA.

Optional Python dependency: **cuda-bindings**, not CuPy. NumPy and ctypes
remain host-side tools. The cuEST C ABI is called directly with ctypes, so
this backend does not install/import `cuest.bindings` or `cuest_scf`.
Native CUDA13.3 runtime/NVRTC/driver, cuBLAS and canonical cuEST0.2.50 are
still required. No solve/install or matching new dependency lock exists yet.
We do not replace this with nvmath or another implicit package dependency.

`dispersion0_cuda(...,provider_factory=...)` is an explicit experimental Python
entrypoint, not a new automatic SAPT option. It shares the existing E2 orbital/
correction setup and original component placement, but bypasses host DFHelper.
Default/non-HF/CPU/E2 dispatch is unchanged. HF, ordinary dispersion, C1,
frozen-space, fragment/S-inf and conflicting-backend guards remain explicit.

## Strict independent cuEST admission

`sapt_cuest250_identity.py` is distinct from CUDA/E2 preparation evidence.
It requires canonical archive bytes206185073 and SHA256
`a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d`,
the whole matching header tree, and CUDA13 native library SHA256
`65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c`.
No0.2.2 admission, paths/SONAME-only inference or implicit package upgrade.

Binding mode is the checked-in0.2.50 ctypes C ABI, tied to exact context,
parameter, workspace, DFMO and plan headers. Unadmitted Python cuEST bindings
in the same process fail rather than bypass this contract. Version getter
signatures are **status(handle,uint32_t*)**, independently read from the actual
0.2.50 context header. A caller must provide an already-valid, owned, live
same-process handle and plan; we never create a hidden handle or reuse one
serialized from another PID.

The admitted handle must already use native FP64 math mode1; cuEST's default
mode0 may use mixed precision even with FP64 input/output buffers. Its owner
must explicitly set the mode. The provider reads it back after mapped-symbol
admission and before every DFMO batch, rejecting/invalidation on disagreement
or getter failure rather than silently changing a borrowed handle's policy.

Admission verifies the native getter result0/2/50, required DFMO symbols,
actual function-address ownership in mapped canonical cuEST bytes, and every
recognized loaded CUDA toolkit library against an explicitly rooted,
hash-pinned closure manifest. Closure schema:

```
cuda_root: absolute SDK locator
cuda_runtime_version: 13030
nvrtc_version: [13,3]
components: [{path: absolute path under that locator, sha256: exact hash}, ...]
```

The caller supplies the immutable manifest's expected canonical-JSON SHA.
Logical SDK-locator symlinks may resolve to explicitly pinned package files;
actual mapped targets must match. Driver libcuda is a separate device/runtime
gate, not a toolkit component to copy or modify. Header/file/ABI/map proofs
are capabilities/provenance, **not** DFMO numerical/rank proof.

Map hashing compares each mapping's inode/device against an opened file's
`fstat`, hashes that same descriptor, and verifies before/after fd/path
fingerprints. Mapped ownership is admitted before version getter calls.
Known CUDA13 toolkit families reject a self-consistent CUDA12 BLAS entry even
when CUDART reports13. Failed final cleanup retains failed resources and
permanently disables execution; an explicit cleanup-only retry is possible,
never an automatic computation/allocation/resource-job retry.

## Plan and resource prerequisites

An audited AO/RI basis/plan factory is still missing. It must preserve exact
Psi4 AO and independent RI spherical/Cartesian conventions, normalization,
auxiliary ordering and common factor gauge, and explicitly configure:

* full Coulomb exchange fraction1, range-separated fraction/omega0;
* fitting algorithm **MATRIXPOWER1**, not defaultQR0;
* relative conditioning1, fitting cutoff1e-12 and metric−1/2 factors.
* native FP64 math mode1 (set by the factory, read back by the provider).

`sapt_cuest_basis.py` implements structural-only basis preflight without
initializing CUDA or selecting a native coefficient convention. It retains
all three exported Psi4 coefficient variants and detects unsupported basis
structures and subsequent changes. This is NOT native normalization,
component-order, coordinate-unit or output-layout admission, and it cannot
yield a production provider. A fake-shell CPU pass is not native basis proof.

Negative eigenvalue treatment, relative rank truncation and contracted-factor/
signed-energy parity require actual comparisons against native Psi4.
Those Disp0 policies must NOT be imposed on PySCF/RKS's separate absolute1e-7
overlap/strict-retention baseline. Existing historical and queued jobs remain
unchanged. Header `@since0.2.0` is not proof of0.2.2 capability or permission
to upgrade an old runtime silently.

The session owns its stream, BLAS handle, JIT module and bounded allocations.
Views retain owners and reject ended lifetimes. Native work is conservatively
device-synchronized before temporary release; no global context reset,
allocation retry or CPU fallback. Live-byte accounting includes every session
allocation plus a declared aggregate of all native persistent owners, but
does not claim total-VRAM certification. Actual peak/context/library/internal
allocation accounting remains part of the GPU gate.

## Validation status

CPU-only tests execute the actual kernel bodies through a C++ qualifier shim
with one serial thread and NumPy GEMM/DFMO stand-ins. This checks indexing,
signs, C/F layouts, edges, ownership and sums; **it does not compile with NVRTC,
exercise parallel synchronization or execute CUDA/cuEST/cuBLAS**.
Identity negatives use explicit synthetic files/callbacks/maps, not loaded SDK
evidence. Actual local archive inspection is streaming read-only, no extraction.

Runtime prerequisites still include the corrected complete Psi4 installation,
exact optional dependency lock, audited native basis/plan factory and loaded
closure/valid-handle provider. Actual GPU numerical and placement gates remain
unpassed. Allocation14250361 is terminal; no new window, remote probe, resource
job, build/install, retry or speedup claim follows from this source checkpoint.
