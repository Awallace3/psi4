# Active cuEST admission: exact 0.2.50, not SONAME-based

Future builds require a separately staged, read-only vendor SDK. These changes
do not extract/install it, upgrade a conda environment, rebuild old stages or
modify queued payloads. Historical FDDS benchmarks used **0.2.2**, not 0.2.50.

Canonical archive: `cuest_v0.2.50.tar.gz`, 206185073 bytes,
SHA256 `a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d`.
Archive prefix: `cuest_PID`. Native member `lib/13/libcuest.so.0.2.50`:
`65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c`.
CUDA12 has a different canonical member/hash; selection must match the toolkit,
never fall back between `lib/13`, `lib/12`, generic `lib` or system paths.

All 59 include files are pinned together: sorted relative POSIX path, NUL,
lowercase SHA256 of file bytes, newline; SHA256 of that stream:
`3d26a7ef607d02ee294f66f11f1374f3badbd83430f5be28ed0a05e6c8d73d18`.
Independently reconciled Disp0's alternate serialization: full archive member
path -> `{"bytes": member.size, "sha256": lowercase_hex}`, compact sorted-key
JSON without trailing newline gives
`644fb4d1b64862e91d9278153aaeb78273a38717209e9ff91d10f4f9f1cecc85`.
The differing aggregate values reflect serialization, not different header bytes.
`cuest.h`: `1220c8209af57c914dc976244f00a81b6f1922ee2feb4b31d5c594439c22b4aa`.
DFMO header `cuest/density_fitted_integrals/density_fitted_mo_integrals_compute_api.h`:
`c93868fcc871fd1fafea374661ad9bc188a3b40646e54eb6b906b5353f06775b`.
Parameter header `cuest/types/cuest_parameter_types.h`:
`ce1b4313eab84d3027d0ad5202ca17e49baf41573f9c700c9089c6535051d882`.

## Configure and runtime contract

- CPU-only builds do not seek or require cuEST/CUDA.
- With cuEST enabled, set `cuEST_ROOT` and `cuEST_CUDA_RUNTIME_ROOT` explicitly.
  Shared CMake discovery accepts only exact canonical header/native bytes,
  matches CUDA::cudart and transitive CUDA dependencies to the selected runtime
  prefix, rejects mixed CUDA components, and checks exported DFMO capabilities.
- CUDA13 uses cuSOLVER **SONAME12** and cuSPARSE12; these are not mismatches.
  CUDA12 uses cuSOLVER11. Do not equate every component major with toolkit major.
- Outer configure forwards selected include/library/runtime-root and the full
  verified identity via a GLOBAL property to inner configure (ordinary variables
  in the discovery subdirectory do not propagate). A superbuild inner configure
  without this identity fails. An SDK/closure change in a reused build is an
  error, not an implicit cache repair. Foreign pre-existing imported targets
  cannot bypass admission with configuration-specific library locations.
- Canonical includes precede dependency headers and are not demoted behind an
  old environment header as system includes. A post-link dependency audit checks
  *actual consumed* cuEST header paths/hashes using Ninja/GCC dependency records.
  Unsupported generators fail explicitly rather than claim header provenance.
- Active CPKS, FDDS and grid build drivers use `devtools/cuest_sdk_env.sh`.
  Default standalone SDK is
  `/storage/project/r-cs207-0/awallace43/software/cuest-0.2.50-a06e8148`.
  Override with `CUEST_SDK_ROOT` (legacy CPKS_CUEST_SDK alias is supported),
  but the same canonical bytes are mandatory.
- Build admission records the staged core hash and exact CUDA ELF closure.
  Runtime admission, inside the importing process before validation, verifies
  that identity again and compares `/proc/self/maps` against the expected
  native/CUDA paths, inodes/devices and hashes, and an executable mapping of the
  pinned core itself. Scientific processes (including
  GDB inferiors and grid-sweep child workers) run through
  `devtools/cuest_run_admitted.py`, with pre/post map checks and a native-version
  receipt after actual cuEST use. A disposable preflight alone is insufficient.
  Grid workers reject a nonzero child exit even if it wrote a successful science
  result before failing post-admission; receipts must bind the original identity.
  Lazy nvrtc-builtins/nvfatbin components have an explicit build-time hash
  allowlist; arbitrary later mappings are not accepted. Driver `libcuda.so.1`
  is site-owned, not part of the toolkit SDK closure.
- Read-only binary reuse requires its **original** core/closure manifest via
  `CUEST_FDDS_BINARY_IDENTITY`; it cannot generate new build-time proof from
  whatever dependencies happen to exist today.
- Verified version API: `cuestStatus_t cuestGetMajorVersion(cuestHandle_t,
  uint32_t*)`, likewise Minor/Patch. Context header SHA256:
  `eea9ba0be5a9f5739ed947f8b97d4819d7a4f019d52629a33832028d14c6a863`.
  Initialization checks successful status and exact native (0,2,50), unwinding
  handles on mismatch/error. `_cuest_runtime_version()` queries an already live
  Psi4-owned handle; it does not initialize CUDA or expose a raw handle.
  The worker checks provider presence before GPU work and queries only after
  its real route. An old core without the provider fails closed.
- Static DFMO symbol admission is **not** a device numerical test. Actual
  0.2.50 FDDS compilation/device/scientific validation is still required.

## Scientific scope is unchanged

DFMO returns fitting-metric-transformed GPU factors and inherits the integral
plan's exchange/LRC settings. It cannot be substituted blindly for FDDS's raw
AO/MO factors or the **contact/delta** three-center auxiliary projection.
Native fitting metric conditioning (1e-12 where used) is distinct from FDDS
response pseudoinverse's strict relative **1e-13** cutoff. No policy, formula,
quadrature, default CPU route, or historical evidence changes here.
