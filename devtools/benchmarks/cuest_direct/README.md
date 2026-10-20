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

The runner defaults to `embers` and verifies the allocation QoS. Only after
explicit approval for that particular paid job may a submitter pass `inferno`
as the fifth argument. An optional sixth argument pins a separately recorded
compiled-source commit when a newer validation-only harness uses an unchanged
core; both commits and the core/SDK hashes appear in the report.

The optional seventh argument selects `H200` (default) or `H100`. Both are
Hopper/sm90 devices, compatible with this build's CUDA XC code. H100 is suitable
for focused numerical debugging if the H200 queue is blocked; do not combine
its timings with an H200 performance panel.

`validate.py` verifies source, core and SDK hashes, actual requested GPU and runtime
linkage; runs ordinary DFT `energy()` comparisons across LDA/GGA/mGGA/hybrid/RSH,
RHF/UHF, Cartesian/spherical/symmetry cases, low-level matrix checks and existing
DF regressions; rejects skips/failures and retains each chemistry output plus
JUnit and `validation.json`. GNU time reports whole-suite peak RSS. This is a
correctness/debug suite, **not** a matched-grid performance benchmark.

## Matched cuEST / GPU4PySCF four-center comparison

The comparison branch merges **committed** `saptdft_cuest@ca21ef8645` at
`2e542f9803`, retaining both direct JK and live-grid exports. The direct adapter
advertises the parent's `returns_combined_exchange()` contract. No unpublished
changes from the other worktree are included. The old validated core is not a
validation of this merged implementation; build and test it separately.

`compare_job.sh` runs a fresh **RKS-only** direct validation suite and the two GPU
grid-export tests before a peptide PBE/WB97X preflight in cc-pVTZ and
aug-cc-pVTZ:

The RKS gate selects exactly nine cases: six closed-shell functional energies,
two Cartesian/symmetry checks and one CUDA-XC smoke. UKS and unrelated DF
regressions are not prerequisites for this closed-shell comparison. The gate
has a ten-minute timeout and rejects missing cases, failures and skips.
`run_validation.sh` still defaults to its broader historical `full` suite;
pass its eighth argument `rks` for this focused gate. Reports record the suite.

```bash
bash ABS_REPO/devtools/benchmarks/cuest_direct/compare_job.sh \
  ABS_REPO HARNESS_COMMIT ABS_BUILD ABS_NEW_RUN ABS_DATA BUILD_JOB \
  H100 preflight embers
```

The optional `protein83` panel first repeats the preflight, then all five
functionals with three timed repeats per engine and basis. It needs a
separately sized/authorized submission; a preflight does not automatically
launch it. Default is free `embers`. An inferno submission still needs fresh
approval for that job.

This uses `jk_mode=conventional-direct`, **not** either historical `df_mode`:

- Psi4: `SCF_TYPE=DIRECT`, `USE_CUEST=True`, `DF_SCF_GUESS=False`,
  core guess, strict FP64, JIT OFF, no auxiliary basis. Each worker verifies
  the actual `cuESTDirectJK` object, direct compute timer and header, and absence
  of a DF-JK header.
- GPU4PySCF: undecorated `gpu4pyscf.dft.rks.RKS`, **no `.density_fit()`**,
  1-electron/core guess, no stored ERIs or DF object. The pinned implementation
  uses `scf.j_engine.get_j` and `scf.jk.get_k`; consumed GPU densities/matrices,
  successful `MD_build_j`/`RYS_build_k` calls and nonzero-omega K for RSH are
  audited. CPU four-center fallback and GPU DF build/contraction entries raise.
  No changes are made to the installed package.
- Auxiliary dimensions are null/not applicable. Both engines must provide
  direct-route proofs before a speed ratio is eligible; DF and direct
  records cannot mix.

The controller reuses the shared exported cuEST native mesh, exact nonzero
point/weight proofs, explicit common Bohr geometry, canonical-overlap
cutoff/spectrum/rank audit, full fresh-process walltime including imports,
worker peak RSS and sampled GPU/process memory. Grid exports/profiling run
separately from timed repeats. **This still does not implement the
smaller-PySCF-native-grid-in-cuEST control.**

CPU atomic/one-electron setup, screening implementations and SCF convergence
norms can differ. This is conventional four-center direct/direct, not a claim
that both programs execute identical kernels or convergence histories.
Successful small-system correctness does not imply protein-scale feasibility.

### Opt-in PySCF stall diagnostic (not a benchmark)

`gpu_compare/diagnose_pyscf.py` invokes the existing profiled PySCF worker with
the supplied immutable config and geometry, without changing the scientific
settings. It is a separate entry point; normal campaigns are unaffected.
It requires a committed, pristine harness and a new output directory.

The parent bounds one child process group (180 s by default, at most 300 s),
requests a final Python stack on timeout, then TERM/KILL/reaps it with at most
five seconds cleanup grace. Group cleanup includes surviving descendants even
if the leader exited; post-KILL reaping never blocks indefinitely. Absence of
the entire group plus leader reaping is required for cleanup verification;
otherwise exit 125 explicitly requires owner containment, not an assumed
release. A second TERM is ignored only during bounded cleanup.
TERM/INT during child creation are deferred, then delivered inside the
cleanup-protected region; they are not silently discarded.
The child emits all-thread Python stacks every 30 s by default and
flushes paired phase markers around initial guess, core/overlap, eigensolver,
J/K, effective potential, grid build and XC calls. Phase wrappers are installed
only **after** the normal direct-dispatch policy has validated original methods.
Input/source hashes and diagnostic-only terminal metadata are retained; the
diagnostic's own bytes must be tracked and identical to its claimed HEAD.
all worker records have `profile=True`, not eligible for headline timings.

Use only inside a separately authorized owner-helper window with pinned
environments and verified allocation cleanup. This is diagnostic preparation,
not a GPU test pass, SCF fix, route certification or permission to launch.
Python stacks identify Python call boundaries, not native CUDA kernel stacks.
No automatic retry, grid export, production panel or environment provisioning
is performed. Full RKS/grid/route gates remain necessary for later benchmarks.

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
