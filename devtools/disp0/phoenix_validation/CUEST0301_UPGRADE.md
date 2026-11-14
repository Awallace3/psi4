# cuEST 0.3.0.1 campaign upgrade

The user requested conda-forge **0.3.0.1 on both Phoenix and Cachy**. This
supersedes another performance build using the old 0.2.50 SDK. Historical
builds, queued jobs and benchmark evidence must remain unchanged.

## Package staging

`devtools/disp0/install_cuest0301.py` creates a **new** dedicated prefix,
refuses an existing prefix, and writes a package-only receipt plus an explicit
conda lock. It does not load cuEST or execute GPU code.

Pinned linux-64 CUDA13 packages:

| Package | Version / build | Published artifact SHA256 |
|---|---|---|
| libcuest | 0.3.0.1 / h345135a_0 | b766bf75603dc25162907b3ed8d29d72ebaebded067ee1a6afd8eb174628d1be |
| libcuest-dev | 0.3.0.1 / h345135a_0 | 0159f8dcf95b2207946a9f5bfd2892b2b9d8200685032e1ae8c29d1ad2eb4d85 |

Source: `https://api.anaconda.org/package/conda-forge/{libcuest,libcuest-dev}`.
The CUDA12 build `hb7e823c_0` is deliberately excluded.
The installer pins CUDA 13.3 and NVRTC/nvJitLink 13.3.33. The first dry solve
showed that `cuda-version=13.3` alone permits nvJitLink 13.5, so its minor
version is pinned separately. Full transitive package versions and hashes
are recorded in each installation's `explicit.txt`.

Intended owned prefixes:

- Cachy: `/home/amwalla3/gits/psi4-cuest0301-cachy/sdk`
- Phoenix: `/storage/project/r-cs207-0/awallace43/software/cuest-0.3.0.1-conda-cuda13`

Installer source is frozen at `cade71cc40`. Cachy installation was launched
with receipt directory `~/gits/psi4-cuest0301-cachy/runs/package-install-01`.
Phoenix installer source is staged at
`/storage/project/r-cs207-0/awallace43/gits/psi4.cuest0301-cade71cc40`;
its output root is
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/cuest0301-install-01`.
Installation completion must be established from `receipt.json`, not these
intended paths. `qualified_for_sapt0: false` is intentional.

Cachy package installation **passed**. Its receipt verifies both pinned
packages and the installed file inventory: **60 headers** and native
`lib/libcuest.so.0.3.0`, SHA256
`cb810a15dff2d0c9c0c83cb56336bf80d77a74ace53f6e4d30921726679fc221`.
`readelf -d` reports SONAME `libcuest.so.0`; dependencies include
cuSOLVER/cuSPARSE major 12 and cuBLAS/NVRTC/nvJitLink major 13.
These component major numbers are not all equal to the toolkit major.
The compact receipts and headers were retrieved to
`/home/awallace43/data/cuest0301-cachy-sdk/` for ABI comparison.
No Psi4/GPU qualification has run against the new SDK yet.

The compile-only `devtools/disp0/cuest0301_abi_probe.cc` **passed locally**
against these copied headers. It covers all 25 bound function calling
signatures, enum/scalar widths, workspace sizes/offsets, basis query constants,
and DF plan/DFMO policy constants. Pointer pointee types and constness are
deliberately erased to match the ctypes boundary; this is not a layout or
numerical test of arrays. Direct inspection of the remaining constructor
headers found compatible declarations, caller-normalized shell coefficients,
and the expected persistent workspace/parent-handle lifetime requirements.

Version 0.3 adds explicitly selectable DFMO operator weighting and DF-plan
fixed-point emulation. For the existing full-Coulomb policy, configure/read
back cache weighting FULL (plan attribute 7/value 0), DFMO weighting FULL
(DFMO attribute 0/value 0), and disable fixed-point fitting emulation
(plan attribute 8/value 1) for the native-FP64 baseline. Do not silently
benchmark a different arithmetic policy just because the SDK defaults changed.

`devtools/disp0/cachy/build_cuest0301.sh` prepares a separate **native-only**
compile/link stage against the new SDK with two compiler workers, sm86,
portable CPU flags and explicit new-library paths. It does not enable the
new SDK in the strict Python admission route and does not constitute a
canonical campaign build (in particular, it does not enable Einsums).
Its purpose is to expose native API/build incompatibilities before migration.

The first native attempt (`d401d17499`) failed at `cuESTJK.cc`: the optional
preview Direct-J threshold enum no longer exists, and Grid-K now takes an
OE integral plan plus a variable workspace descriptor instead of the old
basis/pair-list argument group. These interfaces are outside the 25-function
DFMO probe. The remote wrapper also hid the real build result by assigning
zsh's read-only `status`; inspect the saved build log, and use Bash explicitly
for subsequent wrappers.

The compatibility fix probes the **actual optional API** in CMake and exports
the resulting feature definitions to both Psi4 modules and plugin consumers.
It retains the old optional implementations on compatible SDKs. On 0.3.0.1,
Grid-K and Direct-J are explicitly unavailable until ported and numerically
qualified; selecting them must throw, not fall back. The requested SAPT0
campaign already disables both and uses DF-JK. This is **not** a port or
qualification of the new optional methods.
New native build attempts use pin-specific directories, retain the failed
attempt, and set `CMAKE_BUILD_PARALLEL_LEVEL=2` so the nested superbuild's
compiler count is also bounded.

Phoenix setup belongs in the small **embers** CPU job
`install_cuest0301.sbatch`, not a long login-node solve/install. All 50 embers
slots were occupied at initial staging. No new inferno use is authorized.
Cachy's short GPU correctness check was explicitly authorized after the
rebuild; the desktop has only approximately 19 GiB free at initial setup,
so check space before allocating a full new build.

## Remaining integration gates — not complete

Do not substitute the new library into the old build/proof or disable its
version checks. The current production path intentionally admits only exact
0.2.50 bytes.

1. Inspect the installed package's headers, native SONAME/dependencies and
   native version API. A four-component conda version does **not** establish
   the native three-component version tuple.
2. Add an explicit authenticated SDK profile while retaining the old profile.
   Compare context, parameter, basis, pair-list, DF-plan and DFMO signatures,
   numeric enum values, workspace layouts, and required exported symbols.
   Verify only SDK-owned headers, not unrelated headers in a conda prefix.
3. Migrate identity/build/bootstrap assumptions together:
   - `sapt_cuest250_identity.py`, factory ABI and DFMO consumers;
   - Cachy `bootstrap_gate.py` (also AST-discovers the required symbols);
   - Phoenix `submit.py`, `payload.sh`, `validate.py`,
     `prepare_gpu_runtime.py`, `sealed_h200.py`, `sealed_bootstrap.py`,
     `gpu_runtime.py`;
   - `psi4_runtime_support/sealed.py` authenticated SONAME alias creation;
   - Cachy construction/stage qualification and explicit CUDA closure.
4. Build fresh Psi4 stages with matching new headers/library; preserve
   installed-source provenance, sealed mapping/symbol checks and fail-closed
   retirement. Do not infer success from a working generic `find_package`.
5. Run CPU admission and the approved small Cachy GPU parity gate, followed
   by a Phoenix architecture-specific parity gate. Keep the accepted 1e-7 Eh
   CPU/GPU tolerance and stricter within-arm repeatability criteria.
6. Re-run term timings on Phoenix only after these gates pass. Neither the
   34.6x audit-overhead improvement on 0.2.50 nor a successful package install
   qualifies 0.3.0.1 performance.
