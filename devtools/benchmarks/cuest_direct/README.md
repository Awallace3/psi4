# cuEST four-center direct J/K

Select the route through ordinary PsiAPI:

```python
import psi4
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1")
psi4.set_options({
    "basis": "cc-pvdz",
    "scf_type": "direct",
    "use_cuest": True,
    "df_scf_guess": False,  # prevent a DF preliminary guess calculation
    "cuest_xc": False,      # isolate direct JK first; True enables GPU quadrature
    "cuest_mixed_precision": False,
    "cuest_direct_jit": "off",  # AUTO follows SDK/toolkit defaults
})
energy = psi4.energy("b3lyp", molecule=mol)
```

This selects `cuESTDirectJK`, not `cuESTJK` (stored density fitting), and needs no
auxiliary basis. `USE_CUEST=False` keeps CPU `DirectJK`. No silent CPU fallback
is permitted if the linked SDK lacks direct integrals.

The SDK is a separately licensed dependency, **not vendored**. Configure
`cuEST_ROOT` and, when changing cached SDKs, explicit `cuEST_INCLUDE_DIR` and
`cuEST_LIBRARY`. Both declarations and seven linked direct API symbols are
probed; old DF-only SDKs remain supported. The supplied 0.2.50 SDK exports these
interfaces despite their header comments saying “since v0.3.0”.
Its CUDA 12 and 13 libraries are distinct builds with the same SONAME; use the
matching CUDA toolkit and ensure the selected SDK precedes old conda cuEST in
runtime linkage. Psi4's core RPATH now prioritizes the linked SDK.

## Semantics and limitations

- Conventional four-center J and symmetric-density K, including full-, short-
  and long-range exchange. J remains the full Coulomb operator.
- Like the existing cuEST DF adapter, returned `K` already includes
  `alpha*K + beta*K_erf(omega)`. The cuEST-aware SCF code must not scale it twice.
- Nonsymmetric exchange is rejected: the supplied SDK only exposes symmetric
  exchange. Nonsymmetric J uses the symmetric part of the density, which gives
  the same Coulomb contraction. This is not a general replacement for response
  or SAPT exchange contractions with distinct left/right orbitals.
- `CUEST_DIRECT_JIT=AUTO` leaves the SDK default; `ON`/`OFF` explicitly select
  kernel compilation. SDK JIT requires CUDA 13+. FP64 validation uses OFF.
- Plan/pair-list persistent workspace and maximum reusable J/K scratch are
  reported separately. No stored three-index DF tensor is allocated.
- Energy-reference tests disable GPU XC to isolate JK errors. A separate
  GPU-XC smoke checks the existing cuEST+CUDA-LibXC path; it does **not** imply
  native CPU and GPU grids are identical.

## Reproducible Phoenix validation

Keep the existing comparison build and queued jobs unchanged. Use a fresh
detached worktree and fresh build/output directories. A CPU `embers` job runs:

```bash
/bin/bash devtools/benchmarks/gpu_compare/build_grid_export.sh \
  ABS_REPO COMMIT ABS_BUILD embers ABS_PRIVATE_SDK
```

A separate one-H200 `embers` job, gated by that build manifest, runs:

```bash
/bin/bash ABS_REPO/devtools/benchmarks/cuest_direct/run_validation.sh \
  ABS_REPO COMMIT ABS_BUILD ABS_NEW_GPU_OUTPUT
```

`validate.py` verifies source, core and SDK hashes, actual H200 and runtime
linkage; runs ordinary DFT `energy()` comparisons across LDA/GGA/mGGA/hybrid/RSH,
RHF/UHF, Cartesian/spherical/symmetry cases, low-level matrix checks and existing
DF regressions; rejects skips/failures and retains each chemistry output plus
JUnit and `validation.json`. GNU time reports whole-suite peak RSS. This is a
correctness/debug suite, **not** a matched-grid performance benchmark.

## Required NVIDIA dependency notice

NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THESE LICENSED
DELIVERABLES FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE
LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. NOTWITHSTANDING ANY
TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE AGREEMENT, IN NO EVENT SHALL
NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THESE LICENSED
DELIVERABLES.

U.S. Government End Users. These Licensed Deliverables are a "commercial item"
as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting of "commercial
computer software" and "commercial computer software documentation" as such
terms are used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S.
Government only as a commercial end item. Consistent with 48 C.F.R.12.212 and
48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all U.S. Government End
Users acquire the Licensed Deliverables with only those rights set forth
herein.
