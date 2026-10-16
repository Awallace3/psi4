# GPU XC response (cuEST-grid `compute_Vx`) — progress

Branch `saptdft-cuest-cpks`, local worktree
`/home/awallace43/gits/psi4__worktrees/saptdft-cuest-cpks`. Design: `cpks-gpu-scope.md`;
runtime/cluster rules: `cpks-gpu-runtime.md`. Nothing here is merged into the
parent branch, report or PR.

## Status (2026-10-15)

- Implementation, tests and Phoenix payloads are committed and pushed.
- **Host-LibXC route: GPU-verified on A100** (interactive job 14087772, dev build of
  `c0f23b8d9d`): 116 tests pass, 4 fail (pre-existing cuEST wK DF gap, unrelated);
  18/18 contract cases clean; 13–27x speedup of complete CPKS solves over all-CPU CPKS
  at 16 threads for 384–660 bf (see "A100 interactive results").
- **CUDA-LibXC route: GPU-verified on H200** (14087771, multi-arch build `f2d8ec7a93`, cuEST
  0.2.50): contract 18/18, CUDA vs host LibXC ≤ 1.7e-17, 25–34x CPKS solve speedup over the
  24-core CPU node for 384–660 bf, SAPT CPKS 26x. Empty-beta oracle fixed (`afa5ae424b`).
  Earlier note: Holders: 14087771
  (inferno, est. start 2026-10-17 00:10) and 14096296 (embers, gts-cs207-chemx, submitted
  2026-10-16 04:13 at user request). `interactive/h100-chain.sh` (tests, then probe with
  `PROBE_ARMS=cuest,hostxc`) is launched on whichever starts first; sessions
  `h100-tests-1`, `h100-probe-1`.
- 14085250 (CUDA batch job pinned to the buggy `c477c22060`) cancelled at user request
  2026-10-16 04:13 before it started.
- Full-node (24-core Xeon Gold 6226) CPU baseline, job 14091690: GPU solves still 4.0x
  (228 bf), 9.8–10.1x (384 bf), 12.9–13.1x (660 bf) faster; SAPT benzene CPKS 11.1x.

## Pins (explicit)

| Pin | Role | Remote worktree | Status |
|---|---|---|---|
| `c477c22060` | **Original batch pin, buggy** (cuEST-grid dispatch crash) | `gits/psi4.saptdft-cuest-cpks` (frozen, clean, unchanged) | Evidence preserved in `runs/saptdft-cuest-cpks/c477c22060-20261015/` |
| `c0f23b8d9d` | **Corrected pin** (dispatch fix `391dc95521` + probe fixes) — all GPU results below | `gits/psi4.saptdft-cuest-cpks-dev` (detached, clean) | Host build verified on A100; CUDA build awaiting H100/H200 |

Original-run evidence, kept as is:
- 14085247/14085248 builds COMPLETED (host core sha256 8c626d98…7258, cuda 039a68fa…0487).
- 14085249 gpu-host: accounting COMPLETED, but `out/status.txt` = `cphf=1 regressions=0
  probe=143` (5 test failures: 4 pre-existing cuEST wK + 1 SAPT variable name, fixed in
  `a085b035eb`; probe segfault from the dispatch bug then Einsums debugger hang, PID killed).
  **Its `out/VERIFIED` file is spurious**: `cpks_gpu_validate.sh` ended in an `a && b && c`
  test list, which `set -e` does not abort on, so it wrote VERIFIED and exited 0. Fixed in
  the validation script (if-guard); the evidence directory was left untouched.
- 14085250 gpu-cuda (pinned to `c477c22060`) cancelled before start, 2026-10-16 04:13,
  at user request (also approved via the coordinator questionnaire). No CUDA-route result
  exists from the original pin.

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

## A100 interactive results (job 14087772, 2026-10-15)

Node atl1-1-01-002-12-0, NVIDIA A100 80GB PCIe, 16 CPU threads, account gts-cs207 /
QoS inferno. Dev worktree `gits/psi4.saptdft-cuest-cpks-dev` at `c0f23b8d9d`, host-LibXC
build `build_cpks_host`, core sha256 `5092d24c651ac05f272daa9897c233c6169e5222679b50ba850bc22fcfe13d4b`.
Outputs: `runs/saptdft-cuest-cpks/interactive/sessions/{a100-tests-1,a100-probe-1}/`
(`meta.txt`, `log.txt`, `probe/{contract,timing,sapt}.json`).

**Tests** (`a100-tests-1`): 116 passed, 4 failed — only
`test_cuest_cpscf_fixed_orbitals[cam-b3lyp|lrc-wpbe x RKS|UKS]`, the known cuEST wK DF
fit gap (5e-6–1.2e-5; upstream tolerates it at 5e-4), not XC response.
The SAPT CPKS induction test and `test_vx_cuest_grid_survives_option_changes` pass.

**Contract** (`probe contract`, water, 6 functionals x singlet/triplet/UKS, every V on the
cuEST grid): factored-vs-dense, transpose, sign, antisymmetric, L=R and
occupied-vs-dense reference identities <= 3.3e-16; finite difference of the native-convention
potential 7e-11–8.5e-10 on a 4e-2–1.3e-1 scale; host-grid vs cuEST-grid Vx 2.7e-7–1.5e-6
(grid difference); a negative reference density gives finite output. The SDK nonsymmetric
density normalization/sign contract holds on device for the host-LibXC route.

**Timing** (`probe timing`, s; Hx8 = one `cphf_Hx` with 8 vectors; solve = 3 dipole RHS to
1e-8; arms: cuest = cuEST J/K + cuEST-grid XC response, hostxc = cuEST J/K + native XC,
cpu = DF J/K + native XC; all 16 threads on the same node):

| System (nbf) | XC | Hx8 cuest / hostxc / cpu | solve cuest / hostxc / cpu | solve speedup vs cpu |
|---|---|---|---|---|
| benzene dimer cc-pVDZ (228) | PBE | 0.43 / 1.58 / 1.77 | 2.22 / 8.39 / 9.29 | 4.2x |
| benzene dimer cc-pVDZ (228) | PBE0 | 0.43 / 1.60 / 2.40 | 2.38 / 8.74 / 12.27 | 5.2x |
| benzene dimer aug-cc-pVDZ (384) | PBE | 0.65 / 7.68 / 8.19 | 3.33 / 40.09 / 44.45 | 13.3x |
| benzene dimer aug-cc-pVDZ (384) | PBE0 | 0.71 / 7.74 / 10.70 | 3.57 / 39.70 / 52.76 | 14.8x |
| coronene aug-cc-pVDZ (660) | PBE | 1.50 / 31.0 / 29.5 | 9.29 / 214.4 / 176.1 | 19.0x |
| coronene aug-cc-pVDZ (660) | PBE0 | 1.94 / 37.6 / 55.1 | 11.58 / 201.1 / 309.2 | 26.7x |

Water (24–92 bf, RKS and UKS cation): no speedup (GPU 0.15–0.71 s vs CPU 0.11–1.12 s);
launch/transfer overhead dominates. Cold first factored call vs warm: water 19/12 ms,
benzene cc-pVDZ 90/85 ms. Per-action cuEST stage split (coronene PBE, Hx8): response
density 0.61 s, AO integration 0.58 s, reference density 0.07 s, derivatives 0.03 s,
contraction 0.04 s — the per-vector SDK density/potential calls dominate. All solves
converged (residuals 1e-9–8e-9) in every arm.

**SAPT(DFT)-D4(I) CPKS** (`probe sapt`, PBE0, GRAC 0.136, `_sapt_cpscf_solve` wall time):

| System | CPKS s cuest / hostxc / cpu | total s cuest / cpu | IND kcal/mol cuest / cpu |
|---|---|---|---|
| water dimer cc-pVDZ | 0.29 / 0.41 / 0.40 | 3.7 / 2.6 | -1.660964 / -1.660953 |
| water dimer aug-cc-pVDZ | 0.49 / 0.81 / 0.84 | 2.3 / 3.5 | -1.810282 / -1.810262 |
| benzene dimer aug-cc-pVDZ | 3.25 / 34.4 / 49.9 (15.3x) | 11.4 / 93.0 | -0.319269 / -0.318822 |

**Full-node CPU baseline** (job 14091690, cpu-small, gts-cs207/inferno, node
atl1-1-02-018-5-2, 24 x Xeon Gold 6226, same build/hash, `PROBE_ARMS=cpu`, 17:45 wall;
output `interactive/sessions/cpu24-baseline-1/`):

| System (nbf) | XC | cpu24 Hx8 / solve | A100 cuest solve | speedup |
|---|---|---|---|---|
| benzene dimer cc-pVDZ (228) | PBE / PBE0 | 1.39 / 8.78 ; 1.53 / 9.47 | 2.22 ; 2.38 | 4.0x ; 4.0x |
| benzene dimer aug-cc-pVDZ (384) | PBE / PBE0 | 5.66 / 32.68 ; 6.45 / 36.07 | 3.33 ; 3.57 | 9.8x ; 10.1x |
| coronene aug-cc-pVDZ (660) | PBE / PBE0 | 18.66 / 121.74 ; 23.41 / 149.28 | 9.29 ; 11.58 | 13.1x ; 12.9x |
| SAPT benzene dimer aug-cc-pVDZ CPKS | PBE0 | 35.97 | 3.25 | 11.1x |

SAPT energies of cpu24 equal the 16-thread cpu arm (IND -0.318822 kcal/mol).

The cuest arm runs SCF and response on the cuEST grid, so its energies differ from the
host grid by grid error (benzene IND 4.5e-4, ELST 1.4e-3 kcal/mol); hostxc and cpu agree.

## Multi-architecture CUDA-LibXC route with cuEST 0.2.50 (user, 2026-10-16)

User: "find the cuest 0.2.50 and then build the more general cuda libxc route".

- **cuEST 0.2.50 SDK**: `software/cuest-0.2.50-a06e8148` (`lib/13/libcuest.so.0.2.50`, sha256
  65c9e559…963c), already linked by the cuest-direct build 14080574 (pin 7896a4b, embers).
  SASS sm_80/90/100/120 (PTX sm_120): A100, H100/H200 and RTX Pro Blackwell; **not L40S
  (sm_89)**. cuEST 0.2.2.2 (env) has sm_80/86/89/90/100/120.
- XC API 0.2.2 → 0.2.50: `exchange_correlation/` headers identical apart from comments;
  `cuestXCNonsymmetricDensityCompute` unchanged. 0.2.50 adds direct-integral, grid-K and
  Hessian APIs only. Same soname `libcuest.so.0` as the env's 0.2.2 → the SDK `lib/13` must
  precede `$env/lib` (builds use RUNPATH, so LD_LIBRARY_PATH decides); `cpks_gpu_build.sh`
  now writes `runtime.env` and checks `ldd` resolves the SDK.
- Existing CUDA LibXC `software/libxc-7.1.2-cuda` is sm_90 only (built 2026-09-11, 26 min).
- Scripts (`d6e30f93f2`): `devtools/cuda_libxc_build.sh` (new prefix only, verifies SASS per
  architecture), `cpks_gpu_build.sh` overrides `CPKS_CUDA_ARCHS`, `CPKS_CUDA_LIBXC`,
  `CPKS_CUEST_SDK`, `CPKS_BUILD_DIR`; `cpks_gpu_validate.sh` sources `runtime.env` and
  admits a GPU whose compute capability is in the build's list.
- Pin `d6e30f93f2` (C++ identical to `c0f23b8d9d`), remote worktree
  `gits/psi4.saptdft-cuest-cpks-multi` (detached), run dir
  `runs/saptdft-cuest-cpks/multi-d6e30f93f2-20261016/`, architectures `80;89;90;120`:

| Job | What | QoS / account | Dependency |
|---|---|---|---|
| 14096518 | CUDA LibXC → `software/libxc-7.1.2-cuda-sm80-89-90-120` (build dir `software/build/libxc-7.1.2-cuda-sm80-89-90-120`), cpu-small 24 c, 6 h | embers / gts-cs207-chemx | — |
| 14096519 | Psi4 cuda variant, cuEST 0.2.50, `build_cpks_cuda_multi` → `build-cuda/`, 3 h | embers / gts-cs207-chemx | afterok:14096518 |
| 14096526 | A100 (16 c, 3 h): `chain.sh` = tests then probe (`PROBE_ARMS=cuest,hostxc`) → `sessions/a100-{tests,probe}-1` | embers / gts-cs207-chemx | afterok:14096519 |

Results (2026-10-16):
- 14096518 COMPLETED 4:31 (build 252 s, 24 c): `libxc-7.1.2-cuda-sm80-89-90-120`, SASS
  sm_80/89/90/120 verified, `libxc.so.15` sha256 b8ba0110…cc98.
- 14096519 COMPLETED 5:50: BUILD_OK `d6e30f93f2 cuda`, core sha256 cfd8adb2…ba40, `ldd`
  resolves cuEST 0.2.50 (`lib/13`) and the new LibXC; math check passes. **But core.so
  holds sm_80 SASS only**: the superbuild forwards `CMAKE_CUDA_ARCHITECTURES` to
  psi4-core without a LIST_SEPARATOR, so `80;89;90;120` arrived as `80` (inner CMakeCache).
  Valid for A100 (14096526 left queued); would fail on H100/H200/L40S.
- Fix `f2d8ec7a93`: for a list, `cpks_gpu_build.sh` passes `-gencode=arch=compute_N,code=sm_N`
  in `CMAKE_CUDA_FLAGS` with `CMAKE_CUDA_ARCHITECTURES=OFF`, and checks `core.so` SASS for
  every architecture. Worktree `gits/psi4.saptdft-cuest-cpks-multi2` (detached at
  `f2d8ec7a93`), run dir `runs/saptdft-cuest-cpks/multi-f2d8ec7a93-20261016/`, build job
  **14096965** (embers, gts-cs207-chemx, reusing the 14096518 LibXC).
- 14096965 COMPLETED 5:53: BUILD_OK `f2d8ec7a93 cuda`, core sha256 ec656ab1…d1d3, core SASS
  sm_80/89/90/120 (verified by the new check), math check passes.
- `interactive/h100-both.sh`: on the first H100/H200 holder, the `f2d8ec7a93` multi build
  chain (waits for its BUILD_OK), then the original sm_90 / cuEST 0.2.2 chain.

The H100/H200 holders 14087771 (inferno) / 14096296 (embers) still run the original sm_90,
cuEST 0.2.2 CUDA build of `c0f23b8d9d` (`interactive/h100-chain.sh`) when one starts; the
multi build can also be run there with `multi-…/chain.sh`.

## H200 results: CUDA-LibXC route, multi-arch build `f2d8ec7a93` (2026-10-16)

Holder 14087771 (inferno, gts-cs207, gpu-h200, 8 c) started 06:08 on atl1-1-02-014-2-0,
NVIDIA H200 (cc 9.0, driver 615.71.09). `interactive/h100-both.sh` ran
`multi-f2d8ec7a93-20261016/chain.sh` (core sha256 ec656ab1…d1d3; `ldd`: cuEST 0.2.50
`lib/13`, LibXC `libxc-7.1.2-cuda-sm80-89-90-120`). Sessions in
`runs/saptdft-cuest-cpks/multi-f2d8ec7a93-20261016/sessions/`.

**Tests** (`h200-tests-1`: test_cuest_cphf + test_cuest_grac + test_dft_grid_switch):
148 passed, 6 failed — the 4 known cuEST wK LRC fixed-orbital cases, and
`test_vx_empty_beta_spin[svwn-host|svwn-cuest]` (V_beta off by 1.5e-5, 0.3 %).

Diagnosis of the empty-beta failure (all on the same H200):
- The host-LibXC build (LibXC 7.0.0, cuEST 0.2.2) passes the same test there → not hardware.
- Bisection (`h200-bisect-1..5`): fails with every preceding test group → not test-order state.
- Failing array is the beta block. At rho_b = 0 (zeta = 1) LibXC's VWN vrho_b is
  ill-conditioned: FD of vrho_b along rho_a vs its own analytic v2rho2_ab differs by 1.1 %
  (7.0.0, rho_a = 0.1) and 0.26–2.6 % (7.1.2, rho_a = 1e-3…1e-6); v2rho2_ab agrees between
  the versions to 3e-5; alpha-block fxc vs FD agree to 1e-9 in both; host vs device LibXC
  7.1.2 agree to ≤1e-14 (scripts `xczeta.py`, `xcab.py`, `xcdev.py` run with ctypes).
  The svwn beta-block FD oracle passed under 7.0.0 by accident.
- Fix (`afa5ae424b`, tolerance `8e0c645e4d`): alpha block keeps the FD check at 1e-7; beta
  block is checked for finiteness and, on the cuEST grid, against the native host-grid
  V_beta at 2e-6·max(1, scale). H200 results: CUDA build svwn 1.7e-7, pbe 2.7–2.8e-6
  (scale 4.4); host build svwn 4e-11, pbe 1.6–1.9e-6. All 4 cases pass with both builds.

**Contract** (`h200-probe-1`, CUDA LibXC device derivatives): 18/18 clean — identities
≤ 3.3e-16, FD 7e-11–8.5e-10 on 4e-2–1.3e-1, host grid vs cuEST grid ≤ 1.5e-6, negative
reference finite; **CUDA vs host LibXC response ≤ 1.73e-17** for all 6 functionals × 3 kinds.

**Timing** (`h200-probe-1`, 8 threads, arms cuest (CUDA LibXC device derivatives,
`cuda_libxc_scf_timer` true) and hostxc; speedup vs the 24-core CPU baseline 14091690):

| System (nbf) | XC | Hx8 cuest / hostxc | solve cuest / hostxc | cpu24 solve | speedup vs cpu24 |
|---|---|---|---|---|---|
| benzene dimer cc-pVDZ (228) | PBE | 0.167 / 1.72 | 0.79 / 9.60 | 8.78 | 11.1x |
| benzene dimer cc-pVDZ (228) | PBE0 | 0.174 / 1.72 | 0.89 / 9.83 | 9.47 | 10.6x |
| benzene dimer aug-cc-pVDZ (384) | PBE | 0.264 / 7.63 | 1.28 / 39.99 | 32.68 | 25.5x |
| benzene dimer aug-cc-pVDZ (384) | PBE0 | 0.286 / 7.66 | 1.43 / 40.82 | 36.07 | 25.2x |
| coronene aug-cc-pVDZ (660) | PBE | 0.619 / 28.1 | 3.63 / 155.9 | 121.74 | 33.5x |
| coronene aug-cc-pVDZ (660) | PBE0 | 0.772 / 25.1 | 4.88 / 161.2 | 149.28 | 30.6x |

Water (24–92 bf): GPU 0.05–0.32 s vs hostxc 0.15–1.33 s (now faster than the host
route at every size). Per-action stage split (coronene PBE, Hx8): response density 0.234 s,
AO integration 0.267 s, reference density 0.029 s, derivatives 0.0008 s, contraction
0.0004 s (vs 0.03/0.04 s on the A100 host-LibXC route).

**SAPT(DFT)-D4(I) CPKS** (PBE0, GRAC 0.136): water dimer cc-pVDZ 0.17 s (hostxc 0.98),
aug-cc-pVDZ 0.20 s (0.93); **benzene dimer aug-cc-pVDZ 1.36 s vs cpu24 35.97 s (26.4x)**,
total 5.6 s vs 70.8 s. Energies equal the A100 cuest arm (IND -0.319269, TOTAL -2.218377
kcal/mol); the cuEST-grid vs host-grid difference is grid error as before.

**Original CUDA build on the same H200** (`c0f23b8d9d`, sm_90, cuEST 0.2.2, LibXC
`libxc-7.1.2-cuda`, `interactive/sessions/h100-{tests,probe}-1`): tests 148 passed / 6 failed
— the same 4 wK LRC cases and the 2 svwn empty-beta cases (old oracle, as diagnosed above);
probe exit 0, contract clean, timings equal to the cuEST 0.2.50 build (coronene PBE solve
3.64 vs 3.63 s, PBE0 4.62 vs 4.88 s; benzene aDZ 1.28/1.44 s; SAPT benzene CPKS 1.37 s).
The H200-vs-A100 gain is therefore the device (H200 + CUDA LibXC derivatives), not cuEST 0.2.50.

**GPU memory** (`h200-vxmem-1`, multi build, cuest arm; `cudaMemGetInfo` used MiB after SCF →
after the first `cphf_Hx` (8 vectors) → after a dipole solve → after `V.finalize()`):

| System (nbf) | XC | after SCF | after 1st Hx | added | solve growth | after finalize |
|---|---|---|---|---|---|---|
| water+ UKS aug-cc-pVTZ (92) | PBE | 1123 | 3085 | +1962 | 0 | 1115 |
| water+ UKS aug-cc-pVTZ (92) | PBE0 | 1101 | 3051 | +1950 | -16 | 1065 |
| benzene dimer aug-cc-pVDZ (384) | PBE | 1991 | 4029 | +2038 | 0 | 1943 |
| benzene dimer aug-cc-pVDZ (384) | PBE0 | 1927 | 3943 | +2016 | +72 | 1929 |
| coronene aug-cc-pVDZ (660) | PBE | 5099 | 7191 | +2092 | 0 | 5029 |
| coronene aug-cc-pVDZ (660) | PBE0 | 5023 | 7079 | +2056 | +28 | 4945 |

The ~2 GB is nearly size-independent (only ~130 MB scales from 92 to 660 bf) and matches the
2e9-byte cuEST variable-buffer budget the response passes (`cuestWorkspaceDescriptor_t{0,
2000000000}`): the cached SDK temporary workspace (`sdk_device`) dominates, not the grid
fields. It is bounded across a solve (≤ +72 MiB), released by `V.finalize()`, held per live
V (SAPT(DFT) monomers: up to ~2x), and not counted in any Psi4 GPU budget. Open option:
make the budget configurable / release the SDK workspace after each call, with a timing sweep.

## Response workspace budget (`58d061979b`, H200 14087771, 2026-10-16)

`CUEST_XC_RESPONSE_BUDGET_MB` (default 2000, MB = 1e6 bytes, must be > 0) sizes the cuEST
temporary workspace the response passes as `cuestWorkspaceDescriptor_t{0, budget*1e6}`.
Build job 14098104 (embers cpu-small, worktree `gits/psi4.saptdft-cuest-cpks-budget`, core
sha256 15363319…ed07d, cuEST 0.2.50, CUDA LibXC multi). Run dir
`runs/saptdft-cuest-cpks/budget-58d061979b-20261016/`.

- `sessions/h200-budget-tests`: 10 passed (budget 256/16 MB vs 2000 MB: max |dVx| ≤ 1.4e-17
  for singlet/triplet/UKS; budget 0 raises; empty-beta and option-survival tests pass).
- `sessions/h200-coronene-budget` (same orbitals, coronene aDZ PBE, 8 vectors): 250 and 16 MB
  vs 2000 MB max |dVx| 9.7e-17 / 8.0e-17 (scale 3.3e-2).
- `sessions/h200-budget-sweep` (`budgetsweep.py`; **a fresh SCF per budget**, so its
  `hx_vs_first_budget` column mixes in orbital gauge: 1e-12 water, 7e-7 benzene dimer, 7e-3
  coronene from degenerate-orbital rotations — the same-orbital check above shows the budget
  itself changes nothing):

| System (nbf) | budget MB | added GPU MiB | Hx8 s | solve s |
|---|---|---|---|---|
| water+ UKS aTZ (92) | 2000 / 500 / 125 / 16 | 1962 / 532 / 174 / 70 | 0.077 / 0.078 / 0.078 / 0.086 | 0.33 / 0.33 / 0.33 / 0.37 |
| benzene dimer aDZ PBE0 (384) | 2000 / 500 / 125 / 16 | 2014 / 710 / 226 / 122 | 0.288 / 0.285 / 0.312 / 0.503 | 1.47 / 1.45 / 1.59 / 2.49 |
| coronene aDZ PBE (660) | 2000 / 500 / 250 / 125 / 16 | 2092 / 662 / 422 / 304 / 200 | 0.618 / 0.623 / 0.654 / 0.711 / 1.370 | 3.68 / 3.72 / 3.94 / 4.21 / 8.10 |

Recommendation: 500 MB costs ≤ 1% time at these sizes and saves ~1.4 GB per live V; 250 MB
costs ≤ 7%. The default stays 2000 until the timing-suite sizes (up to 1293 bf) are measured.

## CPKS after initial-to-final XC grid switching (user request, 2026-10-16)

Candidates from CPU job 14097678 (SCF/SAPT induction NONE only): final 75 x 302, initial
25x110, 30x146, 20x110, 35x194, `DFT_GRID_SWITCH_FACTOR=10`. Tests added in
`48e88bbc23`/`aa44bb5080` (`test_cuest_cphf.py`, `-k grid_switch`):

- `test_vx_grid_switch_invalidates_response` (25x110, 30x146 × singlet/triplet/UKS × host/cuEST,
  cuEST with CUDA- and host-LibXC derivatives): a V with populated response caches on the
  final grid goes Initial → Final twice; back on the final grid it must equal a V that was
  only ever final (≤ 1e-11·scale), dense and factored. On the initial grid the cuEST kernel
  **refuses** (`cuest_xc_response.cc` guard: "must be built on the final DFT grid"); the
  native host kernel computes on the coarse quadrature (2.4e-4 off fixed75 at 25x110). The
  first version of the test (`48e88bbc23`) assumed both compute and failed 6 cuEST cases
  on the H200 for that reason; fixed in `aa44bb5080` by asserting the per-route behavior.
- `test_vx_grid_switched_scf_cpks` (all 4 initial grids × svwn/pbe0 × RKS/UKS × host/cuEST):
  switched SCF ends on the final grid (host: same npoints as fixed75), ≥ 10 initial-grid
  iterations; the switched SCF's live V equals a fresh final-grid V on the same orbitals
  (≤ 1.7e-17); CPKS converges with |Hx − b| ≤ 1e-9; static polarizability vs fixed75 SCF.
- `test_vx_grid_switched_saptdft_cpks_induction` (25x110, 30x146 × host/cuEST): SAPT(DFT)-D4(I)
  CPKS, GRAC monomers, components vs fixed75 < 1e-7.

Results. Local CPU build: 24 passed (cuEST skipped). H200 (holder 14087771, build `58d061979b`
+ tests `48e88bbc23`, `budget-58d061979b-20261016/sessions/h200-switch-tests`): 42 passed, 6
failed (the refusal above). Rerun with `aa44bb5080`: `suite-aa44bb5080-20261016/sessions/
h200-switch-tests` (see below).

| Check | native CPU | cuEST grid, CUDA LibXC |
|---|---|---|
| live switched V vs fresh final V | ≤ 1.7e-17 | ≤ 6.9e-18 |
| alpha switched vs fixed75 (4 grids × svwn/pbe0 × RKS/UKS = 16 per route) | ≤ 4.1e-10 | ≤ 7.5e-10 |
| SAPT IND switched − fixed75 (25x110 / 30x146), Eh | 1.1e-10 / 7.8e-11 | 6.0e-11 / 2.5e-10 |
| SAPT ELST/EXCH/TOTAL worst, Eh | 1.9e-9 | 2.0e-9 |

## Timing-suite scale-up, aug-cc-pVDZ (user request, 2026-10-16)

Suite `~/data/timing_test_suite/geometries/systems.py` (local copy, md5 73ec495b…; the
Phoenix `~/data` copy lacks protein83 — uploaded to the run dir, shared geometries identical);
`pla15_3pe2` excluded. Run dir `runs/saptdft-cuest-cpks/suite-aa44bb5080-20261016/`, probe and
tests from worktree `gits/psi4.saptdft-cuest-cpks-suite` (`aa44bb5080`), GPU binary = budget
build `58d061979b` (C++ identical; `run.sh` asserts no psi4/src or driver diff), CPU binary
= `c0f23b8d9d` host build (CPU code unchanged since; same as the cpu24 baseline 14091690).
PBE0, fixed 75 x 302 unless noted.

- GPU: `gpu-chain.sh` on H200 holder 14087771 (ends 08:05): switch tests; timing
  peptide+nanotube; SAPT peptide; timing peptide+nanotube switched 25x110; peptide with
  host-LibXC derivatives; protein83; SAPT nanotube. Whatever does not finish needs a new GPU
  allocation (embers holder 14096296 still pending).
- CPU (embers, gts-cs207-chemx, cpu-small 24c, Xeon Gold 6226): 14098243 peptide (timing,
  timing switched 25x110, SAPT), 14098244 nanotube (timing, SAPT), 14098245 protein83 (timing).
  14098238–40 were cancelled seconds after start: an escaping bug left `PROBE_MEMORY` empty.

GPU results (H200 14087771, CUDA LibXC, all exit 0 unless noted):

| System, basis (nbf) | SCF s | Hx1 / Hx8 s | CPKS solve s | SAPT total / CPKS s |
|---|---|---|---|---|
| peptide aDZ (356) | 2.42 | 0.051 / 0.258 | 2.00 | 9.7 / 1.78 |
| peptide aDZ, host-LibXC derivatives | 2.60 | 0.107 / 0.335 | 3.05 | — |
| peptide aDZ, switched 25x110 | 2.47 (15 initial it.) | 0.051 / 0.257 | 1.97 | — |
| nanotube aDZ (742) | 4.58 | 0.196 / 1.116 | 8.06 | 24.7 / 7.09 |
| nanotube aDZ, switched 25x110 | 4.52 (14 initial it.) | 0.193 / 1.109 | 8.07 | — |
| protein83 aDZ (1293) | 14.0 | 0.677 / 4.453 | 35.2 | 79.0 / 30.9 |
| protein157 aDZ (2491) | **OOM in SCF setup** | — | — | — |
| protein157 heavy-aDZ (2171) | 62.0 | 3.60 / 25.96 | 210.1 | 348.4 / 142.0 |

protein157 aug-cc-pVDZ (`gpu-timing-protein157`, 110 GB, and `-mem50`, 50 GB): "Failed to
allocate device buffer: requested 83.748 GiB; device has 12.185 GiB free of 139.844 GiB"
during cuEST JK setup, before any response call; the request does not follow PROBE_MEMORY.
An SCF/JK device-memory limit, outside this task. heavy-aug-cc-pVDZ (user request) runs:
`gpu-{timing,sapt}-protein157-haDZ`. Protein157 heavy-aDZ solve: 22 Vx calls 53.2 s of 210 s
(AO integration 24.3, response density 20.9, reference density 6.9, derivatives 0.54 s) —
J/K dominates the solve. SAPT values (Eh): ELST -0.0064245, EXCH 0.0175524, IND -0.0030426,
TOTAL -0.0188881. No CPU baseline for protein157 (hours on 24 cores; not submitted).

CPU (24c Gold 6226) so far: peptide SCF 14.67 s, Hx1/Hx8 1.080/4.251 s, solve 36.14 s,
SAPT total 68.6 s / CPKS 34.41 s; switched 25x110 SCF 17.17 s (slower), solve 36.37 s.
**Peptide speedups: CPKS solve 18.1x, SAPT CPKS 19.3x, SAPT total 7.1x; host-LibXC
derivatives route 11.8x.** Polarizability switched vs fixed: CPU ≤ 1e-6, GPU ≤ 1e-5 au
(nanotube), i.e. relative ≤ 3e-8.

CPU jobs all COMPLETED 0:0 (14098243 3:44, 14098244 13:33, 14098245 25:55). Matched
speedups, H200 CUDA-LibXC vs 24-core CPU, aug-cc-pVDZ PBE0, fixed 75 x 302:

| System (nbf) | SCF CPU / GPU s | Hx8 CPU / GPU s | **CPKS solve CPU / GPU s** | SAPT CPKS CPU / GPU s | SAPT total CPU / GPU s |
|---|---|---|---|---|---|
| peptide (356) | 14.67 / 2.42 (6.1x) | 4.25 / 0.258 (16.5x) | **36.14 / 2.00 (18.1x)** | 34.41 / 1.78 (19.3x) | 68.6 / 9.7 (7.1x) |
| nanotube (742) | 77.3 / 4.58 (16.9x) | 31.40 / 1.116 (28.1x) | **236.4 / 8.06 (29.3x)** | 188.3 / 7.09 (26.6x) | 355.7 / 24.7 (14.4x) |
| protein83 (1293) | 261.4 / 14.0 (18.6x) | 103.5 / 4.45 (23.3x) | **865.0 / 35.2 (24.6x)** | — / 30.9 | — / 79.0 |

Protein83 SAPT CPU baseline was not run. CPU vs GPU energies differ by grid only (cuEST vs
host grid): nanotube IND −3.78901 vs −3.78874 kcal/mol, TOTAL −1.17762 vs −1.17692; peptide
IND −2.93960 vs −2.93962.

## Protein157 6-31+g** GPU run, full-threshold / minimum-2 grid policy (user, 2026-10-16)

Request: GPU counterpart of CPU job 13868775 (`docs/saptdft/cuest/phoenix-2026-10-04/protein157-inferno/
results-13868775/protein157-6-31+g**-new-cpu-1/result.json`, sha256 2e2a2921…fc94c; geometry sha256
a1e9c894…6b60; nbf 1786; 8 threads; 10358.56 s; fixed 99x590): SAPT(DFT)-D4(I) PBE0, induction NONE,
iterative unseeded GRAC, δHF, cuEST DF-JK (no DIRECT), GPU SAD, FP64; grid 35x194 → 75x302,
DFT_GRID_SWITCH_FACTOR=1 and **at least 2 final-grid iterations**.

- Not run on 14087771: 9.5 min left at 07:56, guard unported, input not yet located (reported).
- Guard: `scf_iterate(..., xc_final_min_iterations=N)` ported in `ff2ebc4c4f` (same diff as sibling
  `saptdft-cuest-grid-policy-v1` f4e4c68970; my base iterator blob 0547c274 is identical). Local CPU:
  `test_dft_grid_switch.py` 32 passed (10 new: RKS/UKS energy parity at factor 1/min 2; exact final count
  1/2/3 for converged He; MAXITER cannot bypass; invalid minima rejected). Runner CPU dry run
  (`--cpu-smoke`, water dimer): 6 switched SCFs, all converged, final 5 ≥ 2, final grid after.
- Runner `devtools/protein157_grid_policy_run.py`: options = reference options + USE_CUEST/CUEST_XC/
  CUEST_SAD true, mixed precision false, DFT 75x302 final, 35x194 initial, factor 1; the iterator is
  overlaid from the run worktree (pinned GPU build `58d061979b` untouched); no memory sampler.
  Checks: cuESTJK seen, no MemDFJK/DiskDFJK/DirectJK, GPU SAD seen, every switched SCF converged with
  final ≥ 2 and ended on the final grid. Records options changed vs reference, nbf/naux, components,
  GRAC shifts, energy() wall, timer records, per-SCF proofs, core/libcuest/libxc sha256.
- Phoenix: worktree `gits/psi4.saptdft-cuest-cpks-p157` (ff2ebc4c4f), run dir
  `runs/saptdft-cuest-cpks/p157-full2-ff2ebc4c4f-20261016/` (`run.sh`, `chain.sh`: GPU water-dimer
  smoke gates `protein157-631gss-full2`). Launch on embers holder 14096296 (H100|H200, 8c, 128G, 2h)
  via `srun --overlap` when it starts; auto-launcher armed. No new submissions.

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
