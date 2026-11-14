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
