# GPU XC response (cuEST-grid `compute_Vx`) — progress

Branch `saptdft-cuest-cpks`, local worktree
`/home/awallace43/gits/psi4__worktrees/saptdft-cuest-cpks`. Design: `cpks-gpu-scope.md`;
runtime/cluster rules: `cpks-gpu-runtime.md`. Nothing here is merged into the
parent branch, report or PR.

## Status (2026-10-15)

- Implementation, tests and Phoenix payloads are committed and pushed.
- Local CPU build/tests pass. **No GPU evidence yet**: Phoenix jobs are queued
  (inferno since 2026-10-15). Correctness on the cuEST grid is **not claimed** until
  the GPU jobs below pass.

## Commits

| Commit | Content |
|---|---|
| `74f62655af` | Implementation, tests, `devtools/cuest_xc_response_{math_check.cc,probe.py}` |
| `3cc9d44ebb` | `devtools/cpks_gpu_build.sh`, `devtools/cpks_gpu_validate.sh` |
| `c477c22060` | Host-LibXC GPU link may run on A100 (**pin of the queued jobs**) |

## What was implemented

- `psi4/src/psi4/libfock/cuest_xc_response.{h,cc}`: `VBase::compute_Vx_cuest`
  for RKS singlet, RKS triplet and UKS, LDA/GGA, on the potential's own cuEST
  grid. Stages (all timed with device sync): `cuEST Vx: setup / weights /
  reference density / derivatives / response density / contraction / AO
  integration / output symmetry`, under `RV|UV: Form Vx (cuEST)`.
  - Response densities: `cuestXCNonsymmetricDensityCompute` (signed, nonsymmetric
    factors). Factored entry points (`compute_Vx_factored`,
    `RV::compute_Vx_full_factored`) take CPKS's existing left/right factors
    (rhf.cc/uhf.cc use them when `uses_cuest_xc()`); dense `compute_Vx` callers keep
    the API via identity-tile factors (tile ≤ 256 MiB for both factors).
    Rank-0 factors are zeroed without an SDK call.
  - Reference density: ordinary SDK density from the `set_Cocc` factors only when
    `‖CCᵀ − D‖ ≤ 1e-12·max(1,‖D‖)` (cached until `set_D`/`set_Cocc`), else the dense
    `D_AO_` through the nonsymmetric path. `D` is authoritative.
  - Derivatives: host LibXC (SuperFunctional workers at deriv 2, OpenMP, exceptions
    captured) or CUDA LibXC (`xc_lda_fxc`, `xc_gga_vxc_fxc` per component; polarized
    clone for triplet/UKS). Option `CUEST_XC_RESPONSE_HOST_LIBXC` forces host LibXC
    in a CUDA-LibXC build. Nonfinite derivatives raise.
  - Per-point algebra `cuest_xc_response_math.h`, shared by both routes: RKS
    half-alpha normalization (n = 2ρα, weights applied once), triplet combinations
    A=(f_aa−f_ab)/2, B=(f_a,saa−f_a,sbb)/4, C=(f_saa,saa−f_saa,sbb)/8, S=f_saa/2−f_sab/4,
    native masks (RKS skip n<cutoff; UKS LDA block ρa+ρb>cutoff, GGA block ≥cutoff).
  - Output C1 or symmetry-blocked (any perturbation irrep) via AO2USO.
  - Workspace: grow-only device arrays (throwing allocation, non-throwing free),
    reusable SDK temp workspace; budget {0, 2 GB}; released in `finalize()` before
    the plan/grid. One RHS (UKS: one α/β pair) at a time bounds memory.
  - Rejections with native messages: VV10, MGGA, GRAC for triplet/UKS, initial
    SCF grid, potentials built on a CPU grid then toggled to `CUEST_XC`.
  - Builds whose cuEST lacks the nonsymmetric density (CMake
    `PSI4_CUEST_HAS_NONSYMMETRIC_XC`) keep an explicit unsupported error.
- Native fixes found on the way (v.cc):
  - UKS `compute_Vx` output indexing (`Dx[2i+s]`).
  - RKS triplet with GRAC threw inside the OpenMP loop → `std::terminate`/hang; now
    rejected before the loop.
  - Validation errors in `RV::compute_Vx_full`/`UV::compute_Vx` leaked the
    `Form Vx` timers (next call: "Timer … is already on"); timer starts after validation.
  - Known, left as is: native `compute_Vx` does not check output dimensions
    (out-of-bounds write with wrong-size `ret`); the cuEST path does check.
- Python bindings: `VBase.set_Cocc`, `uses_cuest_xc`, `compute_Vx_factored`,
  `compute_Vx_full(Dx, ret, singlet)`, `compute_Vx_full_factored`.

## Oracles (why the tests look like this)

cuEST has no grid-coordinate query, so the host and cuEST grids are different
quadratures. Same-grid oracles: central differences of the grid's own
`compute_V` (verified on the native kernel first: RKS `Vx[dDα] = ½·dVα`, FD error
~2e-10; UKS `Vx = dV`; triplet `½·dVα` with Cα=C+εU, Cβ=C−εU on the polarized
functional, error ~1e-10), algebraic identities, dense/factored/occupied/LibXC-route
equalities. Host-vs-cuEST grid comparisons are quadrature-level only (loose gate
1e-4·scale for Vx, polarizability atol 1e-3/rtol 1e-4, SAPT 1e-6 Eh).

Empty-β (H atom) GGA: FD of V_β is ill-conditioned where ρβ=0 (error grows as the
step shrinks: 3.6e-4 → 2.7e-2 for ε 1e-3 → 1e-5, native kernel), so the GGA β
block is only checked for finiteness; SVWN checks both spins.

## Tests (`tests/pytests/test_cuest_cphf.py`)

Parametrized over `grid ∈ {host, cuest}` (cuest needs cuEST + CC≥8):
`test_vx_finite_difference` (svwn, blyp, pbe0, cam-b3lyp, lrc-wpbe × singlet/triplet/uks,
dense and factored), `test_vx_algebra` (sign, transpose, antisymmetric→0, zero,
linearity, adjoint, batch order, UKS cross-spin, zero-rank factors),
`test_vx_empty_beta_spin`, `test_vx_symmetry_irreps` (c2v, every irrep, SO in/out,
symmetry-blocked factors), `test_vx_reference_changes` (repeat, other reference,
rotated/stale Cocc, independent V objects interleaved, Initial grid rejection and
Final rebuild, failed call then reuse, finalize/initialize), `test_vx_grac_conventions`
(singlet GRAC response = uncorrected kernel, triplet+GRAC raises),
`test_vx_unsupported_on_grid` (MGGA, VV10).
GPU only: `test_vx_cuest_routes_and_host_grid` (occupied vs dense reference,
CUDA vs host LibXC, verbose-print route evidence, host-grid quadrature gate),
`test_vx_cuest_cpks_solve` (GPU SCF, solver residual, polarizability vs host grid,
`Form Vx (cuEST)` timer reached), `test_vx_cuest_symmetry_cpks` (c2v mixed AO/SO
batches, zero RHS shrinking the batch), `test_vx_cuest_saptdft_cpks_induction`
(SAPT(DFT)-D4(I) CPKS induction, CUEST_XC on vs off, 6 decimals).
`test_cpscf_gpu_xc_guard` now expects "not initialized on the cuEST grid".

`devtools/cuest_xc_response_math_check.cc`: exact cutoff-equality masks and
triplet projection = ½·UKS α output for (u, −u) (error ≤ 5.6e-17).

### Local results (CPU build, no GPU; not GPU evidence)

Build: `build_cpks_cpu` (p4_ci env, Einsums from the parent's stage, unity build).

```
cd <scratch>; PYTHONPATH=$W/build_cpks_cpu/stage/lib:$W/tests/pytests \
  python -m pytest -q -p no:cacheprovider $W/tests/pytests/test_cuest_cphf.py -n 8
→ 43 passed, 74 skipped (GPU)
pytest test_tdscf_products.py test_tdscf_excitations.py test_cuest_options.py
  test_dft_grid_switch.py test_grac_cpu.py test_cuest.py test_cuest_jk.py
  test_cuest_gemm.py -m "quick or not long" -n 12
→ 144 passed, 55 skipped
c++ -std=c++17 -I psi4/src devtools/cuest_xc_response_math_check.cc && ./a.out → all checks passed
PROBE_CPU_SMOKE=1 python -P devtools/cuest_xc_response_probe.py OUT contract timing quick → ok
```

## Phoenix

- Remote worktree `/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft-cuest-cpks`
  (branch `saptdft-cuest-cpks`, at `c477c22060`, clean). **Frozen while jobs are
  queued** — do not advance it.
- Builds go to `build_cpks_host` / `build_cpks_cuda` inside that worktree.
- Run root `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-cpks/c477c22060-20261015/`
  (`JOBS.txt`, per-link `job.sbatch`, `logs/`, `metadata/{runtime.env,COMPLETED|FAILED}`, `out/`).
  (`…/3cc9d44ebb-20261015/` holds only a `--test-only` log; unused.)
- **Route change 2026-10-15:** the user approved inferno for this task ("use inferno
  for this task"). In-place `scontrol update QOS=inferno` was rejected ("Requested
  time limit is invalid"), so the embers jobs 14081580–14081583 were cancelled
  before starting (record: `JOBS-embers-cancelled.txt`) and the same `job.sbatch`
  files, edited to `--account=gts-cs207 --qos=inferno`, were resubmitted.
  `inferno_account.sh` split: all on `gts-cs207` (worst case $0.79 per build,
  $5.69 per H100 link). Effective QoS/partition is in each link's
  `metadata/runtime.env`.

| Job | Link | Route | Wall | Status |
|---|---|---|---|---|
| 14085247 | build-host | gts-cs207 / cpu-small / inferno, 24 c, 96 G | 4 h | queued |
| 14085248 | build-cuda | gts-cs207 / cpu-small / inferno, 24 c, 96 G | 4 h | queued |
| 14085249 | gpu-host: tests + probe `contract timing sapt quick` | gts-cs207 / A100\|H100\|H200 / inferno, 8 c, 64 G | 6 h | afterany:14085247 |
| 14085250 | gpu-cuda: tests + probe `contract timing sapt` | gts-cs207 / H100\|H200 / inferno, 8 c, 64 G | 6 h | afterany:14085248 |

`--test-only` estimates at resubmission (conservative): builds 2026-10-16 01:10;
gpu-host 2026-10-18 (A100); gpu-cuda 2026-10-27 (H100).

Watcher: `scripts/watch_jobs.sh -i 1800 phoenix2 14085247 14085248 14085249 14085250`,
local harness background task, log `~/docs/saptdft/cuest/cpks/watch-c477c22060-inferno.log`.

Each GPU link writes `out/{gpu.txt,import.json (package path, core SHA256, linked
libxc/cuest/cudart), cphf.xml/log, regressions.xml/log, tests_summary.json,
probe/{contract,timing,sapt}.json, status.txt}` and `VERIFIED` only if tests and
probe all exit 0. Build links write `out/{configure,build}.log`, `abi-after-build`,
`math_check.log`, `core.sha256`, `BUILD_OK`.

## GPU run 1 (pin `c477c22060`, 2026-10-15) — results and fixes

- Builds 14085247 (host LibXC, core SHA256 `8c626d98…7258`) and 14085248 (CUDA LibXC,
  `039a68fa…0487`) COMPLETED 0:0 in ~6 min; nonsymmetric-SDK gate, ABI probe and
  algebra self-check passed.
- 14085249 (A100-SXM4-80GB, host LibXC, `gts-cs207`/inferno): `test_cuest_cphf.py`
  **112 passed, 5 failed**. Every new cuEST-grid test passed on this route. Host vs
  cuEST grid Vx: 1.6e-7–9.4e-7 (quadrature-level). Failures:
  - `test_cuest_cpscf_fixed_orbitals[cam-b3lyp|lrc-wpbe, RKS|UKS]` (4): pre-existing
    cuEST J/K vs CPU J/K test with CPU XC, wK gap 5e-6–1.2e-5 — the known cuEST wK
    density-fit difference, unrelated to the XC response. Left unweakened.
  - `test_vx_cuest_saptdft_cpks_induction`: the SAPT(DFT) runs completed, then the test
    read `SAPT IND20,R ENERGY`, which SAPT(DFT) never sets. Fixed in `a085b035eb`
    (`SAPT IND ENERGY`; same fix in the probe).
- 14085249 probe: **segfault** in the contract section, then Einsums parked the process
  in its debugger-wait loop (GPU idle ~1.3 h until I killed PID 640115 inside the
  allocation; job then ended FAILED). Localized in-allocation with `srun --overlap`:
  building a host-grid V sets `CUEST_XC=false` globally; the still-alive cuEST-grid V then
  dispatched `compute_Vx` (and `set_Cocc`) from the live options, ran the native kernel
  over a grid with no CPU blocks and crashed. **Real bug, fixed in `391dc95521`**:
  dispatch follows how the V was initialized (`uses_cuest_xc()` is true when its cuEST
  plan exists); options requesting cuEST for a host-grid V still raise. Regression test
  `test_vx_cuest_grid_survives_option_changes`. (Pre-existing, unchanged: `compute_V`
  and other XC entry points still read the live options.)
- Probe now passes `--einsums:no-attach-debugger` so crashes exit (`c0f23b8d9d`).
- 14085250 (CUDA LibXC, pending) is pinned to the buggy `c477c22060`; its probe would hit
  the same crash. Needs cancelling (awaiting user approval) or it will be superseded.

## Speedup requirement (user, 2026-10-15)

Success includes demonstrating speedup over **CPU CPHF/CPKS**. Probe `c0f23b8d9d` times
three arms per system: `cuest` (cuEST J/K + cuEST-grid XC), `hostxc` (cuEST J/K + native
XC) and `cpu` (CPU DF J/K + native XC, the all-CPU baseline), for per-action `cphf_Hx`,
complete dipole-polarizability solves and SAPT(DFT)-D4(I) CPKS induction; systems water,
benzene dimer, coronene/aug-cc-pVDZ (660 bf). The CPU baseline must also be run on a full
CPU node (24 threads), not only on the GPU allocation's 8 cores.

## Interactive debugging (user, 2026-10-15)

"Until you are confident in your code, get an interactive inferno node at 2hr time outs."
Holders (`runs/saptdft-cuest-cpks/interactive/hold*.sbatch`, `sleep` for 2 h; work via
`srun --jobid=<id> --overlap`): 14087771 (H100|H200, 8 c — max 8:1), 14087772 (A100, 16 c).
Dev worktree `gits/psi4.saptdft-cuest-cpks-dev` (detached, separate from the pinned one),
dev build job 14087810 (cpu-small inferno) → `runs/saptdft-cuest-cpks/interactive/build-c0f23b8d9d/`.

## Resume

1. `scripts/job_status.sh phoenix2 14085247 14085248 14085249 14085250`.
2. For each finished link: check `metadata/COMPLETED`, `out/VERIFIED` (GPU) or
   `out/BUILD_OK` (build); read `out/tests_summary.json` and `out/probe/contract.json`.
   Confirm `import.json` package path is the run's build and the variant's libxc.
3. Acceptance to check in `contract.json` (per functional/kind): `finite_difference`
   ≲ 1e-8 absolute; `*_occupied_vs_dense_reference`, `*_L_eq_R_vs_dense`,
   `*_factored_vs_dense`, `*_transpose`, `*_sign`, `cuda_vs_host_libxc` at ~1e-11·scale;
   `cuda_libxc_scf_timer` true only in the cuda build; `negative_reference`
   behavior recorded. Then timing (cold first factored call vs warm, per-stage
   times, cphf_Hx per action, complete solves, SAPT CPKS seconds, host-grid XC vs
   cuEST-grid XC).
4. If a build fails, fix on the local branch, push, ff-merge the remote worktree
   (only once no job is pending against it), and resubmit to a new run dir.

## Not yet covered / open

- All GPU evidence (SDK contract, CUDA LibXC derivative calls, timings) — pending jobs.
- Not exercisable by a test: workspace allocation failure path, nonfinite-derivative
  error path (the probe records what a negative reference density does).
- Functional parameter changes on a live V (e.g. omega) are not tested; independent
  V objects and reference/grid changes are.
- Outside scope (unchanged): mGGA/VV10 response, direct J/K, analytic Hessians, FDDS.
