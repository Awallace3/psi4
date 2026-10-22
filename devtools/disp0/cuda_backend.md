# Experimental Python/cuEST CUDA dispersion — CuPy-free

## Automatic public HF selection with USE_CUEST

As requested, ordinary HF dispersion now selects the custom CUDA/cuEST route
automatically when `USE_CUEST=true`, independently of the old MP2 algorithm
default. The caller must inject `cuda_provider_factory(primary, auxiliary)`,
a context manager yielding the strictly audited `CuESTDFMOProvider`.
There are no hardcoded SDK paths or implicit native-handle/factor fallbacks.

```python
psi4.set_options({
    "USE_CUEST": True,
    "CUEST_MIXED_PRECISION": False,
    "SAPT_DFT_FUNCTIONAL": "HF",
    "SAPT_DFT_USE_EINSUMS": True,
})
energy, wave = psi4.energy(
    "sapt(dft)", molecule=dimer, return_wfn=True,
    cuda_provider_factory=audited_factory,
    cuda_dispersion_block_size=24,
)
execution = wave._sapt_cuda_dispersion_execution
```

Missing factory configuration is rejected before SCF. Automatic HF selection
requires native FP64, ordinary dispersion, spherical/C1 inputs and the current
Einsums2 SAPT cache. Fragment-driver wiring, external potentials and conflicting
E2 GPU-dispersion selections fail closed; no CPU dispersion retry follows native
failure. Structured diagnostics are not inserted into scalar QCVariables.
`USE_CUEST=false` and existing non-HF/empirical routes remain unchanged.

Public dispatch guards have CPU-only regression coverage. The coherent install
and combined real GPU SCF -> automatic dispersion gate remain pending here;
separate native GPU HF and numerical dispersion gates already pass. This is not
yet complete/default FISAPT, certified upstream cache placement or speedup proof.

## Current target and uncompleted GPU scope

User requires spherical primary and spherical RI bases only, functional=HF,
both signed Disp20 and Exch-Disp20, with all production dispersion numerical
work on GPU. Existing GPU SCF/elst/exch/induction paths are outside current
reimplementation. All FISAPT terms follow only after the HF dispersion gate.
No Cartesian calculation case is a supported target; historical diagnostics
and negative Cartesian-input rejection tests are not production support.

The explicit raw-HF route uploads upstream matrices and performs dispersion
setup, coefficient packing, native DFMO, contractions and signed pair/global
reductions on GPU. Both Disp20 and Exch-Disp20 pass bounded physical gates,
including unequal/frozen spaces and block-size changes. It no longer uses the
legacy host-setup callback or host coefficient packing/reduction route.

Native GPU overlap, metric powers, IAO assembly and full ordered IBO localization
also pass bounded physical gates. All core unscaled two-fragment term families
have separate physical GPU evidence; downstream tests using the original GPU
IBO buffers are in progress. Current raw HF/upstream inputs remain explicitly
CPU fixtures, not a resident SCF/full-workflow claim. Automatic driver dispatch,
scaled/link/external cases, resident upstream handoff and matched performance
remain open. CPU scalar control and separate oracles are allowed; numerical
CPU fallback is not. See `disp0-progress.md` for current receipts.

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

The explicit chain is `dispersion0_cuda` → `prepare_hf` (cuBLAS/CUDA setup) →
`blocked_device_dispersion` → strict `CuESTDFMOProvider.compute_device` →
`cuestDFMOIntegralsCompute` into session-owned device output → device packing,
three cuBLAS products and signed CUDA pair/global reductions. Host inputs are
raw upstream HF matrices and energies, not prepared dispersion corrections or
DF factors. Final outputs are downloaded after GPU reductions. The factor
consumer rejects host arrays or buffers from a different session.

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
still required. Cachy's private coherent install and exact corrected dependency
lock are recorded in the sibling setup receipt; this is not portable-machine
or production numerical certification.
We do not replace this with nvmath or another implicit package dependency.

`dispersion0_cuda(...,provider_factory=...)` remains the explicit low-level
entrypoint used by automatic `USE_CUEST` HF dispatch. It implements the canonical
orbital/correction equations with owned GPU operations and bypasses DFHelper.
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
fingerprints. Btrfs maps/mountinfo superblock devices may differ from subvolume
`st_dev`; these mappings remain fail-closed. A proposed fdinfo/mountinfo/path-link
exception was rejected because stable cross-subvolume overmounts can make
different mapped/opened files present identical observations. No such fallback
is installed: descriptor pathname evidence must not replace backing-file proof.
Mapped ownership is admitted before version getter calls.
An explicit byte-identical private tmpfs locator deployment now passes real
fresh-process toolkit/cuEST loading, original strict mapped-file proof and
actual resolver-pointer checks on Cachy. Dependency loading occurs after the
matching native core but before first E2 request; all owners remain retained.
Original environment/SDK/whole installed stage are unchanged. See
`../runs/native-loader-device-01/` from the repository root. This loader-only
probe issued no cuEST construction/version getter, DFMO or custom-kernel API;
ELF initializers may execute. Core pathname/file-hash evidence assumes a trusted
frozen install; system-driver file hash plus mapped pointer path is separately
scoped, not claimed as strict mapped-byte proof.
Known CUDA13 toolkit families reject a self-consistent CUDA12 BLAS entry even
when CUDART reports13. Failed final cleanup retains failed resources and
permanently disables execution; an explicit cleanup-only retry is possible,
never an automatic computation/allocation/resource-job retry.

## Plan and resource prerequisites

The explicit owned factory remains diagnostic-scoped, not general production
admission. Its spherical normalization/order/unit and bounded signed-factor/
physical gates now pass. It preserves exact primary/RI identities and a common
auxiliary gauge, and explicitly configures:

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

`sapt_cuest_plan_owner.py` implements diagnostic resource construction and
dependency-ordered cleanup, but requires an explicitly unvalidated coefficient
candidate and cannot yield a production factor provider. It retains queried
persistent/creation workspaces, rejects dimension disagreement and ambiguous
native outcomes, and separates accounted workspace bytes from unmeasured opaque
internals. Native construction and bounded numerical conversion now have
separate device receipts; CPU fake-handle/lifecycle tests alone are not that
evidence.

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

Explicit native-owner leases prevent session teardown from freeing persistent
workspaces or destroying borrowed stream/BLAS/module resources after failed or
incomplete native destruction. Native owners must unregister only after cleanup
completion; registration inside temporary allocation scopes is forbidden.
This retention mechanism does not itself implement a native plan owner graph.

## Validation status

CPU-only tests execute the actual kernel bodies through a C++ qualifier shim
with one serial thread and NumPy GEMM/DFMO stand-ins. This checks indexing,
signs, C/F layouts, edges, ownership and sums; **it does not compile with NVRTC,
exercise parallel synchronization or execute CUDA/cuEST/cuBLAS**.
Identity negatives use explicit synthetic files/callbacks/maps, not loaded SDK
evidence. Actual local archive inspection is streaming read-only, no extraction.

Runtime prerequisites include the coherent whole Psi4 installation, exact
dependency lock, owned native basis/plan factory and loaded-closure/live-handle
provider. These and bounded physical GPU gates pass on Cachy's local3090;
see the CURRENT progress section and sibling receipts. This does not certify
another machine, general/default/full-FISAPT coverage or a speedup. Historical
Phoenix allocation14250361 is terminal and outside this local workflow.
