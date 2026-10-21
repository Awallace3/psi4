# cuEST DFMO → Python/Einsums2 dispersion seam

Status: **local investigation only; not implemented or runtime-ready**.
No CUDA execution, remote query, allocation, source merge or speed claim.
Psi4's undefined installed version metadata remains a separate import blocker.

## Evidence boundaries

The owner reports cuEST0.2.50's
`include/cuest/density_fitted_integrals/density_fitted_mo_integrals_compute_api.h`:
`cuestDFMOIntegralsCompute` accepts batches of independent device CL/CR
coefficients and produces device `[naux,nleft,nright]` **fitting-metric
transformed** factors, with stored-plan or `THREE_INDEX_INTEGRAL_DIRECT=1`
execution. This is an attributed SDK finding, pending direct inspection of
the local archive/header and its hash here. Availability in a header is not
proof of Python exposure, device-buffer interoperability or numerical parity.
The reported API is not a contact/triple-overlap integral endpoint. FDDS's
ALDA operator must not be substituted for Coulomb SAPT dispersion.

The inspected local E2 checkout is exactly
`06404fa69ce9346443d7d4f19a250f6db9f0f62c`; tracked files are clean
(untracked historical logs are not build provenance).

## Current factor requirements

Source: `psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py`, `fdisp0`.
All ten current transformations use the same DFHelper, auxiliary basis,
default metric power **−1/2**, and full-Coulomb operator.
`psi4/src/psi4/lib3index/dfhelper.h` sets `mpower_=-0.5`,
`condition_=1e-12`, `cutoff_=1e-12`. Actual effective settings still need
recording in a numerical comparison.

Coefficient pairs, in the current `(left,right)` transformation order:

* `Aar`: `(Cvir_A,Cocc_A)`; `Abs`: `(Cvir_B,Cocc_B)`.
* `Bas`: `(Cs1,Cocc_A)`; `Bbr`: `(Cr1,Cocc_B)`.
* `Cas`: `(Cvir_B,Ca2)`; `Cbr`: `(Cvir_A,Cb2)`.
* `Dar`: `(Cr3,Cocc_A)`; `Dbs`: `(Cs3,Cocc_B)`.
* `Ear`: `(Cvir_A,Ca4)`; `Ebs`: `(Cvir_B,Cb4)`.

The kernel constructs `Far=Dar+Ear`, `Fbs=Dbs+Ebs`, then packs:

```
AFar = [Aar | Far | Qar  | SBar]
FAbs = [Fbs | Abs | SAbs | Qbs ]
BCas = [Bas | Cas | Sas  | Qas ]
BCbr = [Bbr | Cbr | Qbr  | Sbr ]
```

The first two fields are auxiliary-length vectors; the final two are
rank-one correction rows, **not additional DF auxiliary functions**.
Those corrections and the setup that computes them must survive unchanged.
The contractions produce V, X and W, with the existing occupied-fastest
index convention and W's crossed orbital layout. Denominators, signs,
spin factors, frozen spaces and exchange-dispersion permutation stay fixed.

For this canonical full-Coulomb dispersion seam, the consumed DF factors
are metric-transformed; raw AO three-center tensors are not directly consumed
by the signed energy kernel. Do not apply the fitting metric twice. Raw
integrals/metric may still be needed as *diagnostic oracle inputs* to establish
the SDK's precise whitening convention and threshold behavior.

Different valid whitening factorizations can rotate the auxiliary gauge.
Elementwise cuEST/DFHelper equality must therefore not be assumed. Establish
the common metric convention or an explicit gauge transformation first; also
compare contracted Gram/cross products and every signed physical component.
All ten factors and every block must use one compatible gauge/ordering,
not independently whitened subblocks. Document shell normalization, spherical
ordering, auxiliary rank truncation and screening.

## Proposed Python-level producer boundary

### NVIDIA Python example: caller-owned output is the useful direction

User-linked NVIDIA `CUDALibrarySamples` inspected at commit
`1c1c103ca6141c0a868dcfdb4c4c213b4588d872`, without imports, installation
or execution:

* `cuEST/python_examples/helpers/cuda_utils.py`, SHA256
  `2016e51665a2642396da9fd10135749451b2a4a664eed5b8d186beaceba72fd8`,
  uses `cuda.bindings.runtime.cudaMalloc/cudaFree`, explicit stream helpers
  and synchronized host/device copies.
* `cuEST/python_examples/3_density_fitting/df_mo_integrals/run.py`, SHA256
  `208776d8a5aa8885f858bea9e5ae46cad49b8d8aed7d1e53073ffb4b07eb8f11`,
  allocates coefficient and output device buffers in the caller, wraps their
  addresses in `ce.Pointer`, and passes them to workspace query and
  `cuestDFMOIntegralsCompute(...,outTensors=...)`. It finally copies outputs
  to host and explicitly frees the device buffers.

This demonstrates Python orchestration and **caller-managed allocation**,
not an obligation to import a cuEST-owned allocation. Python ownership itself
is not the fundamental obstacle. Prefer one owner:

1. A Python-visible E2 device allocation, or a retained external allocation
   owner, supplies storage.
2. cuEST borrows its address for output; `ce.Pointer` alone is not assumed to
   retain that owner or synchronize asynchronous consumers.
3. A generic E2 binding registers the same buffer/layout and its device-current
   epoch, waits on producer completion and retains the owner through graph
   execution. No redundant host upload or premature free.

This can avoid ownership transfer entirely. The remaining missing part is
E2's exposed device-storage/graph handoff contract, not a new dispersion
algorithm. A CuPy-owned buffer is another possible owner, but CuPy allocation
does **not** add a missing E2 DLPack importer. Replacing E2 contractions with
CuPy would change the requested backend and must not happen silently.

The example itself performs D2H copying, not an E2 zero-copy handoff. Its
random coefficients are not a physical SAPT validation. The fetched main
branch and Python package dependencies also are not automatically compatible
with the pinned cuEST0.2.50 native SDK; verify that exact binding/native pair
before use and do not install the latest package as an implicit upgrade.

Replace the DFHelper transformation/staging producer, not the SAPT equations:

1. Build an audited full-Coulomb cuEST integral plan for the identical AO/RI
   basis, metric/threshold policy and active occupied/virtual spaces.
2. Supply coefficient **slabs** to batched DFMO calls, obtaining bounded
   transformed device factors. Independent left/right spaces are necessary
   for all exchange factors, not only `Aar` and `Abs`.
3. Combine Far/Fbs and pack their auxiliary and rank-one fields using
   already-supported Python-level device operations, if those operations
   genuinely exist. No new C++ dispersion or elementwise algorithm.
4. Feed the same blocked V/X/W equations via default E2 graph execution.
   Preserve r-side reuse, edge blocks and bounded per-shape caches.
5. Initially label the result GPU producer/contractions with remaining CPU
   setup/elementwise/reductions wherever applicable. Do not claim whole-call
   device residency, full F-SAPT coverage or a speedup from a device producer.

A legal buffer contract is prerequisite to steps3–4. A NumPy view of a host
RuntimeTensor plus a GPU shadow is **not** an external-device import API.
Required evidence includes:

* Python-visible import/export and ownership API, or an explicitly identified
  missing generic binding; no guessed pointer constructor.
* Device/UUID, dtype FP64, allocation size, extent, strides, offset/alignment,
  and cuEST row/column layout versus E2 occupied-fastest mapping.
* Parent ownership across graph capture/replay and slices; pool/reset rules;
  release only after asynchronous consumers finish.
* Producer/consumer stream synchronization and error propagation.
* External writes invalidate stale host/device shadows without an accidental
  H2D overwrite; subsequent CPU reads synchronize the actual device result.
* No hidden D2H→host-pack→H2D round trip advertised as zero-copy.

If generic import/residency support is missing, report it explicitly and
propose the narrow upstream plumbing. A host-staged bridge may serve only as
an explicitly transfer-inclusive numerical prototype, not as this GPU-resident
optimization or evidence of its achievable speed.

## Exact064 E2 source audit: external-device seam is missing

Read-only delegate `dd2a4fd55fec87804bfc193d4393aefb1`, verified report SHA256
`d4592168e3f3b686d5e86cd715a0e6b7bb3869e2bc05eddc8ce8aaba29916152`,
plus direct inspection of the cited storage/executor files:

* `libs/Einsums/Tensor/include/Einsums/Tensor/TensorForward.hpp:49–94`:
  `RuntimeTensor<T>` uses `std::allocator`; native `RuntimeGPUTensor` uses
  `gpu::DeviceAllocator` and is explicitly **not exposed to Python**.
* `.../RuntimeTensor.hpp:88–100,1444–1450` and
  `libs/Einsums/Python/include/Einsums/Python/Protocol.hpp:50–80`:
  Python bindings/buffer protocol represent host storage, not a CUDA import.
  No committed DLPack or `__cuda_array_interface__` integration was found.
* `.../RuntimeTensor.hpp:1250–1277,1316–1334`: native host-only
  `materialize_into` can retain an owner; native `alias_to/set_data` do not
  provide a Python-visible device ownership/synchronization contract.
  View storage keepalive does not acquire ownership of an arbitrary pointer.
* `libs/Einsums/ComputeGraph/include/Einsums/ComputeGraph/DeviceShadowMap.hpp:20–90`:
  shadows own `device_malloc` allocations and free them with `device_free`;
  there is no external-buffer adoption/registration method.
* `libs/Einsums/ComputeGraph/src/Graph/GpuDispatch.cpp:79–130,209–216`:
  discrete-GPU execution resolves shadow pointers; ordinary GEMM expects
  packed column-major rank2 buffers and can decline unsupported layouts.
* `.../Graph/Execute.cpp:190–506`: device validity starts empty per replay;
  host data can be uploaded, CPU fallback restores host pointers/downloads
  current values, and the replay ends with a D2H flush of all remaining
  device-current tensors. Persistent shadow *allocations* are not persistent
  external-producer authority.
* `libs/Einsums/GPU/include/Einsums/GPU/Stream.hpp:50–80` provides native
  streams/events but no Python external-producer handoff;
  `GPU/Runtime.hpp:56–59` exposes coarse device synchronization. Assigned
  graph stream IDs alone do not wait for a cuEST producer's event.

**Conclusion:** existing Python-only pointer aliasing cannot safely implement
the proposed zero-copy producer. It could cause host dereference of device
memory or overwrite fresh cuEST data with stale host bytes. The current
host-packed phase1 implementation is not upgraded by these findings.

Minimum proposed *generic E2 interoperability*, not a dispersion algorithm:

1. An owner-retaining Python device-buffer import with validated allocation
   span/dtype/layout/device/context and explicit borrowed/owned policy.
2. Graph external-device bindings distinct from host tensors and owned shadows,
   with external-write epochs, producer/consumer completion and no borrowed
   pointer passed to `device_free`.
3. Explicit per-buffer/per-call host-mirror policy for CPU fallback, or an
   opt-in fail-fast supported-GPU-only contract. Never change the existing
   default executor's semantics globally to suppress fallback or transfers.
4. Validated Python-accessible generic layout/copy/concatenate operations, or
   a documented supported layout that avoids them. Native device tensor
   bindings alone do not prove the full graph import/consumer seam.
5. Regression tests for owner destruction, pool reset, external mutation and
   repeated replay, asynchronous producer events, unsupported layout/fallback,
   multiple device/context rejection and borrowed-pointer release.

Any native work would be narrowly scoped generic E2 binding/storage/runtime
plumbing on a separately reviewed pin. SAPT factor construction and scientific
contractions stay Python-level. No such change has been implemented or merged.

## Memory and execution accounting

SDK scratch (including any advertised2GiB setting) is not total VRAM.
Budget persistent plan/AO data, metric/factorization, coefficient slabs,
all live output factors, Far/Fbs combination temporaries, packed rows,
E2 graph shadows/workspaces, V/X/W/T/D blocks and CUDA context/library use.
Measure peak allocation and allocator reserve, not just a configured limit.

For slab extents R/S, Q auxiliary functions and occupied counts a/b, the four
packed inputs alone cost `8*(2*Q+2)*(a+b)*(R+S)` bytes. Compute blocks
M=a*r, N=b*s also need product/amplitude/denominator/cross-layout workspaces;
do not materialize the complete occupied/virtual four-index tensor.
Stored-plan and direct/recompute modes have distinct memory/work tradeoffs
and must be reported separately. Avoid reconstructing identical s factors
for every r block unless required by the explicit memory budget.

## Gates before any allocation window

* Resolve Psi4 version metadata through correct upstream build/install from
  verified release history, not an import bypass or installed-file overlay.
  Prove the complete installed E2/Psi4/library world and used CPU APIs.
* Confirm the SDK prototype, layouts, plan lifetime, full-Coulomb policy and
  E2 device-buffer seam in pinned source; enumerate unsupported operations.
* Only then request an owner-assigned bounded window after the owner verifies
  its actual remaining allocation and exclusive cleanup/lock state.
* Inside that window, rerun independent native/eager/default-CPU signed
  references with the actual installed stage, then compare device DF products,
  Disp20 and Exch-Disp20 under the existing predeclared tolerances. Include
  several dimers/bases, frozen spaces, block edges and changed-input replay.
* Prove actual successful cuEST/CUDA/cuBLAS work, mapping CUDA/NVML UUID and
  own PID. Natural CPU fallback on small contraction blocks is legitimate.
* Keep diagnostics separate from exclusive uninstrumented matched timing.
  Include coefficient preparation, integral plan, producer/recomputation,
  packing, synchronization/transfers, CPU work and cleanup in whole wall.

Greater-than10x is a target only. Establish an actual matched CPU/HF baseline,
profile the end-to-end fractions and use those fractions to bound optimization
headroom; a faster contraction cannot erase unaffected host/setup time.
No reuse of parent PBE0-D4 timings as an HF-E2 speedup baseline.
