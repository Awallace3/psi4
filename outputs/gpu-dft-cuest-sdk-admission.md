# Future strict cuEST admission: local implementation status

**Local guards implemented; native/GPU admission unrun. The direct/direct
comparison is not newly completed by this work.**

## Immutable existing experiment

No job, remote checkout, environment, SDK installation or staged binary was
modified. In particular:

- Queued experiment: **14254253**; no fresh scheduler query was made here.
- Frozen remote harness: `86feda59ec84e82eb8c90685c2d9d2c8fbc09237`.
- Existing compiled source: `2e542f9803be8d9d446df5bcc4e88360756a142d`.
- Build job: 14191123.
- Existing core SHA256:
  `3e9aff54e9501647305f2eb122a21b1f2da6f9f287de3d2e33f6eb95c4a9d21d`.

Read-only audit `ba1c8463b` matched the selected existing library to the canonical
CUDA13 bytes and found the direct/DFMO/version exports and current canonical
header selection. A prior Psi4 worker recorded that library's actual mapped
path and CUDA13 runtime. However, the old build did not attest build-time header
hashes, and the old worker did not query native version getters or attest
mapped inode/hash in that process. The old queued payload's self-consistent
manifest checks are **not** the new independent strict admission contract.

## Independent canonical contract

`devtools/benchmarks/gpu_compare/cuest-sdk-0.2.50.json` pins:

- Native release: **0.2.50**, CUDA13.
- Archive: 206185073 bytes, SHA256
  `a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d`.
- Shared library: 88186400 bytes, SHA256
  `65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c`.
- Entire 59-header tree, SHA256
  `3d26a7ef607d02ee294f66f11f1374f3badbd83430f5be28ed0a05e6c8d73d18`.
  Serialization: sorted include-relative POSIX path, NUL, lowercase file hash,
  newline. FDDS/Disp0 independently reconciled this against the full-archive
  JSON inventory; the JSON digest is a different serialization, not a mismatch.
- Additional selected API-header hashes and required direct-JK, DFMO and native
  version symbols.

Static verification does not dlopen cuEST or create a handle. Future builds
require the explicit private SDK and `lib/13`, check the header tree before and
after compilation, and save library/header/CUDA linkage receipts.

## Future process admission

Psi4 requires a pinned staged native core with the FDDS-owned builtin
`_cuest_runtime_version()`. The early check verifies presence without calling
it. After an actual JK/energy route creates a live handle, the bridge must
return exactly `tuple(int,int,int) == (0,2,50)`. It must check all native getter
statuses and raise on absent/non-live handles or version mismatch. No hidden
initialization, guessed no-argument vendor getter, NULL-handle call, or mangled
global handle is used.

Workers verify actual mapped cuEST inode/device/hash, required symbol addresses
inside the canonical executable mapping, and pinned mapped CUDA13 user-library
closure. GPU4PySCF stays in a sanitized separate CUDA12 process. Its preliminary
device/eigensolver subprocesses also receive that environment; admission checks
run before device use and afterward.

The CUDA12 side requires an independently reviewed inventory file and independent
SHA256 via `GPU_COMPARE_CUDA12_RECEIPT` and
`GPU_COMPARE_CUDA12_RECEIPT_SHA256`. It checks actual Python prefix, real resolved
ABI filenames, mapped paths/inodes/hashes and library closure membership.
CUDA12 cudart cannot conceal CUDA13 BLAS, including a mislabeled receipt key or
symlink. Missing inventory fails closed; no inventory is manufactured from
whatever libraries happen to load.

Scientific settings are unchanged, including FP64 conventional direct JK and
the RKS absolute overlap cutoff `1e-7` with strict `>` retention. DFMO symbol
admission is not a DFMO numerical test. Provenance overhead stays inside the
full-process headline timing.

## Offline verification

Final local task `b20315388` exited0:

- 190 `gpu_compare` units, including **30 SDK-admission tests**.
- 1 `cuest_direct` unit.
- Four changed launcher scripts passed `bash -n`.
- Staged and unstaged `git diff --check` passed.

Static review found preliminary environment, cudart-only closure and
receipt-key/real-ABI gaps; these were fixed with regression tests. Rechecks
`d76d319ded81877a8012a4d0254daad14` and
`daa14cf1503c07a9add8e5934dc609ad7` resolved those specific findings.
The later whole-header-tree guard has four additional passing offline tests.
These reviews and synthetic tests are not native/device execution evidence.

## Remaining prerequisites

1. FDDS-reviewed final native provider source/core build and its receipts.
2. Independently captured/reviewed CUDA12 inventory.
3. Separately authorized allocated-GPU validation of the new admission contract.
4. Completed correctness-gated direct/direct benchmark artifacts, with actual
   consumed grids, full-process timing, peak RSS and GPU-memory observations.

No new resource request, queued-payload update, remote staging, cancellation,
environment change, merge or auto-retry is authorized by this implementation.
