# Experimental optional CuPy/cuEST dispersion

This is an explicitly user-authorized alternative backend, not a claim that
Einsums owns or consumes external CUDA buffers. Existing default/CPU/E2
dispatch remains unchanged. There is no automatic new SAPT option or native
plan construction hidden behind it.

## Dependency answer

**Yes: CuPy is an additional optional GPU-environment dependency.**

Options, not executed installation instructions:

* Conda-forge `cupy-core` supplies the array package without automatically
  replacing the private CUDA libraries. Full `cupy` also solves CUDA runtime
  dependencies and may change the selected CUDA stack.
* Alternatively, the CUDA13 wheel family is `cupy-cuda13x`, installable within
  a dedicated conda environment. Do not install it alongside conda `cupy` or
  another CuPy wheel family.
* The native DFMO adapter also requires `cuest.bindings` matching the selected
  native cuEST0.2.50 library and its Python/CUDA dependencies (NVIDIA's example
  uses `cuda-bindings`). Never blindly install the sample's latest extras or
  replace a pinned native SDK.
* No cuTENSOR, PyTorch, new compiler or source-built dispersion library is
  required by these Python equations.

CuPy's inspected stable installation docs list Python3.13 and CUDA13.x wheel
support, but the listed tested toolkits stop at13.2. Our private13.3 runtime
is therefore a compatibility candidate, not a tested dependency combination.
Exact package versions/artifact hashes and a new explicit environment lock
must be established before execution. Existing shared/oracle/failed environments
remain unchanged; no package installation or solve has occurred.

## Implemented building blocks

* `sapt_dispersion_cupy.py`: lazy CuPy import; device factor packing, three
  FP64 matrix products, negative HF denominators, signed Disp20/Exch-Disp20
  occupied-pair reductions, bounded r/s blocks and explicit final D2H.
* `sapt_cuest_dfmo.py`: caller-owned CuPy coefficient/output/workspace buffers
  passed through the existing `cuest.bindings` DFMO API. Batched coefficient
  slabs, queried workspace caps, retained native owners and conservative
  device synchronization before/after native execution. Native errors fail,
  rather than falling back to DFHelper or NumPy.
* `test_cupy_backend.py`: CPU-only independent scalar-equation and transport
  fixtures; fake pointer/workspace API and NumPy stand-ins are explicitly
  tests, not GPU evidence.
* Explicit `sapt_jk_terms_ein.dispersion0_cupy(...,provider_factory=...)`
  reuses the canonical setup, bypasses DFHelper entirely, and shares original
  occupied-pair matrix placement/sum rules. No automatic driver option is
  wired; the factory must provide the complete audited native plan context.

The caller must supply an independently audited **full-Coulomb** native plan,
the exact AO/RI basis objects and persistent-workspace ownership. This code
does not yet construct/validate a native cuEST basis/plan from Psi4. SDK output
order is an explicit parameter, not guessed from logical shape. Matching
metric−1/2 conventions, gauge/auxiliary ordering, screening, coefficient
layout and numerical thresholds remain physical-gate requirements.

Device-wide synchronization is intentionally conservative until the exact
native stream contract is proved. It is included in producer timing. Only
coefficient slabs are uploaded; factors/products are not downloaded.
No custom RawKernel, C++ dispersion algorithm or new CUDA elementwise kernel
is introduced; the scientific expressions use existing CuPy operations.

Memory admission is a **live-buffer estimate**, not a total-VRAM guarantee.
`native_persistent_bytes` must aggregate all retained native AO/RI basis,
pair-list, integral-plan and persistent-workspace allocations. CuPy used and
reserved pool bytes are reported separately; cached blocks are considered
reusable, though fragmentation can still cause an allocation failure.
CUDA context/library/internal/native allocations and whole-process peak VRAM
still require independent measurement. No allocation retry is implemented.

`blocked_execution_seconds` includes the factor-production/block loop and
final output transfer, but excludes upstream orbital setup and borrowed-plan
creation/destruction. The integrated wrapper separately reports
`python_dispersion_call_seconds`, including its callback/factory context and
cleanup; this is still not a complete fresh-process HF/SAPT wall measurement.

Cleanup failures never replace the original native/body error: they are
attached as exception notes. With no earlier failure, cleanup failures raise
directly. A native or synchronization failure invalidates the adapter, so it
cannot silently retry on a possibly unhealthy CUDA/native context.

## Limits / not ready

No actual CuPy/CUDA execution, cuEST native-call verification or physical
signed-component gate has run. Reports deliberately keep independent device
and numerical verification false. Native plan creation and complete runtime
admission are still missing, not silently substituted with CPU work.

The current installed Psi4 stage still rejects undefined version metadata.
Resolve this through a proper complete build/install from verified release
history, not a Python overlay or an import-check bypass.

No GPU allocation/window, remote query or dependency installation follows
from this implementation checkpoint. The FDDS owner must first grant a live
serialized window after actual runtime readiness. No >10x speedup or broader
F-SAPT coverage is established.
