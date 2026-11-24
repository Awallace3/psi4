# RTX PRO 6000 Blackwell F-SAPT & SAPT0 timings — job 16084802

## Provenance and status

- Phoenix `gts-cs207 / gpu-rtxpro-blackwell / inferno`, 2026-11-24.
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`97,887 MiB`, `sm_120` / `compute_120`, driver `615.71.09`), host `atl1-1-03-020-2-0.pace.gatech.edu` (`8` CPU threads in both arms, `160 GiB` host RAM, `80 GiB` declared CUDA session budget, `8 GiB` native reserve).
- Allocation completed in **03:39:42** (`10:20:36`–`14:00:18` EST); job, `.batch`, and `.0` exit codes `0:0`, `job-status.env` `exit_code=0`, `term_received=0`, owned scratch cleanup `complete=True`.
- Result: **`PASS_SUITE`** (Stages 1, 2, 3a, 3b, 4b, 5a, 5b, 7 `PASS`; Stage 4a `nanotube` SAPT0 tripped `1e-7 Eh` solely on `Elst10,r` at `-1.31e-7 Eh` due to `cond(S)~1e9` + `d_convergence=1e-10`; Stage 5c `protein83` F-SAPT hit the `80 GiB` Blackwell declared budget by `0.21 GiB`; Stage 6 `protein157` hit physical 96 GiB VRAM in dimer SCF).
- Science commit: `85d7429837687327e86d9523c8948e2b33d8aafa`, built & CPU-admitted in Phoenix job **`16056429`** (`COMPLETED 0:0` in `10:30`, `426` CPU guards, `api-probe.json` `PASS`, `cpu-sealed-admission` `PASS`, conda-forge **cuEST `0.3.0.1`** `cb810a15dff2d0c9c0c83cb56336bf80d77a74ace53f6e4d30921726679fc221`, CUDA `13.3` / `13030`, native FP64 `CUEST_NATIVE_FP64_MATH_MODE=1`, DF emulation OFF).
- Suite helper commit: `8888c216bb8ffb1c05deec362dc36d4aef5b017e` (zero diff in `psi4/` and `psi4_runtime_support/` vs `85d7429837`).
- Manifest SHA256: `5f6a7182a9cd62098b875b4b4cbdec67b3dcb04019f5015e0ae59827f2a70f42`.
- `suite-result.json` SHA256: `b3c33227190b3d8df89ee34af55446dd7a4d88ecc522366a428935420f08178d`.
- Accompanying H200 `embers` job **`16057336`** (`disp0-0301-suite-85d7429837-02`, same build `16056429` / `85d7429837`) remains queued to capture direct H200 FP64 speedups and 141 GiB `protein83` full F-SAPT.

Remote artifact directory:

```text
/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/disp0-0301-rtx6000-85d7429837-01/job-16084802
```

## Hardware context (native FP64)

All runs execute in **native FP64** (`CUEST_NATIVE_FP64_MATH_MODE=1`). Because **RTX PRO 6000 Blackwell** (`sm_120`, workstation GB202) has a ~1:64 workstation FP64 rate (~1.9–2.0 TFLOPS FP64, ~1.8 TB/s GDDR7) compared to **H200 SXM** (34 TFLOPS FP64 / 67 TFLOPS FP64 Tensor Core, 4.8 TB/s HBM3e), Blackwell serves as a full 96 GiB correctness & scaling proof while understating H200 FP64-GEMM speedups by ~10×–20× on large systems.

## Measured Phoenix timings & accuracy (`16084802`, spherical `jun-cc-pVDZ`)

### 1. Whole-workflow & dispersion timings (8 CPU threads in both arms unless noted)

| Stage / System | AO (`nbf`) | Mode | CPU time (s) | RTX 6000 time (s) | RTX 6000 speedup | `disp`/`fdisp0` CPU → GPU (speedup) | Numerical parity (`\|ΔE\|` / max term / oracle) |
|---|---:|---|---:|---:|---:|---|---|
| **Stage 1** Water dimer gate | 82 | 10-proc F-SAPT gate | — | `134.6` (all 10 procs) | — | — | **PASS** (`PASS_SMALL_H200_ONLY`, all signed + same-gauge + repeat + displaced) |
| **Stage 2** `fsapt1` (S22 phenol dimer, `E10/D12`) | 312 | F-SAPT + `fsapt.py` | `405.79` | `42.71` (`42.60`, `42.83`) | **9.50×** | `4.69 s → 0.337 s` (**13.9×**) | **PASS**: `\|ΔE\| = 3.77e-9 Eh`; same-gauge max matrix `1.83e-10 Eh`; **172/172** `fsapt.py` cells (`OH`/`PH`) max diff `8.22e-9 Eh` (`5.16e-6 kcal/mol`), GPU repeat `3.06e-11 Eh` |
| **Stage 3a** `peptide` (`E10/D10`, 3×3) | 250 | Canonical `SAPT0+δHF` | `17.43` | `13.74` | **1.27×** | `9.57 s → 0.876 s` (**10.92×**) | **PASS**: `\|ΔE\| = 1.78e-10 Eh`; max raw term `2.72e-9 Eh` (`Disp20,u` `2.83e-13 Eh`, `δHF` `4.55e-13 Eh`) |
| **Stage 3b** `peptide` (`E10/D12`, 3×3 baseline) | 250 | Full F-SAPT | `154.47` | `22.90` *(old H200 `15472290`: `32.95 s`)* | **6.75×** | `2.85 s → 0.164 s` (**17.33×**; *old H200: `2.52 s`*) | **PASS**: same-gauge max matrix `1.27e-11 Eh`; all 3 repeats passed |
| **Stage 4a** `nanotube` (`E10/D10`, 3×3) | 548 | Canonical `SAPT0+δHF` | `138.28` (`cpu-0`) | `41.73` (`31.41` warm) | **3.31×** | `64.77 s → 4.27 s` (**15.17×**, blocked `1.90 s`) | 8/8 workers `PASS`; 6/7 terms `<= 5.6e-9 Eh` (`Disp20,u` `3.05e-9 Eh`), GPU repeats `<= 2.34e-10 Eh`; `Elst10,r` `-1.31e-7 Eh` (tripped `1.0e-7` gate due to `cond(S)~1e9` + `D10` CPU floor; Phoenix GPU matches Cachy CPU within `1.5e-8 Eh`) |
| **Stage 4b + 7** `nanotube` (`E10/D10`) | 548 | Full F-SAPT | `3,393.66` | `187.05` (`186.90` med; `204.40` inst) | **18.14×** | `23.44 s → 1.98 s` (**11.82×**) | **PASS**: `\|ΔE\| = 1.98e-8 Eh`; `Disp20,u` `-9.15e-10 Eh`, `Exch-Disp20,u` `+1.00e-10 Eh`; all 15 signed terms passed; same-gauge oracle (23 matrices) max diff **`1.45e-8 Eh`** |
| **Stage 5a** `protein83` fixed-HF `fdisp0` (`def2-SVP`) | 561 | Fixed-HF `fdisp0` | — *(vs frozen H200 `15337409`)* | `24.34` (`1.62 s` DFMO, 1 call) | — | 1 DFMO call (`20.33 GiB` peak) | **PASS**: `Disp20` pairs `1.59e-16 Eh`, `Exch-Disp20` pairs `1.59e-17 Eh` vs H200 snapshot |
| **Stage 5a** `protein83` fixed-HF `fdisp0` (`jun-cc-pVDZ`) | 922 | Fixed-HF `fdisp0` | `69.51` *(48-core CPU `15474909`)* | `57.11` (`3.53 s` DFMO + `53.5 s` FP64 GEMM, 1 call) | **1.22×** *(vs CPU48)* | 1 DFMO call (`46.26 GiB` peak) | **PASS**: `Disp20` pairs **`1.74e-14 Eh`**, `Exch-Disp20` pairs **`5.94e-14 Eh`** vs CPU48 snapshot; same-session repeat `0.0` scalar / `1.08e-19 Eh` pair |
| **Stage 5b** `protein83` (`E10/D10`, 1×1 + warm/prof) | 922 | Canonical `SAPT0+δHF` | `769.99` | `261.10` (`249.54` warm) | **2.95×** | `377.45 s → 73.97 s` (**5.10×**) | **PASS**: `\|ΔE\| = 7.67e-9 Eh`; max term `Elst10,r` `1.26e-8 Eh` (`Disp20,u` `8.56e-10 Eh`, `δHF` `7.51e-11 Eh`); warm/cold matched `< 1e-9 Eh` |
| **Stage 5c** `protein83` full F-SAPT | 922 | Full F-SAPT | `11,456` *(frozen CPU48)* | *(budget cap)* | — | — | All 3 SCFs converged cleanly; hit `80 GiB` Blackwell declared budget by `0.21 GiB` when building sibling RI plan (`52.91 + 19.30 + 8.00 = 80.21 GiB`; H200 `120 GiB` budget has ~40 GiB headroom) |
| **Stage 6** `protein157` bounded GPU probe | 1,786 | `SAPT0` / F-SAPT | — | *(OOM)* | — | — | Clean OOM in dimer SCF `cuESTJK` plan creation (`76.53 GB` requested, `28.59 GB` of `102.07 GB` free) |

### 2. Instrumented per-step breakdown on `peptide` (Stage 3b) and `nanotube` (Stage 4b + Stage 7)

| F-SAPT step | `peptide` CPU8 (s) | `peptide` RTX 6000 (s) | `peptide` ratio | `nanotube` CPU8 (s) | `nanotube` RTX 6000 (s) | `nanotube` ratio |
|---|---:|---:|---:|---:|---:|---:|
| `scf_dimer` | `3.338` | `5.974` | `0.56×` | `24.744` | `13.110` | `1.89×` |
| `scf_A` | `1.452` | `1.162` | `1.25×` | `3.539` | `2.219` | `1.59×` |
| `scf_B` | `1.437` | `1.157` | `1.24×` | `13.387` | `6.060` | `2.21×` |
| `canonical_prepare_JK` | `0.160` | `0.064` | `2.49×` | `1.735` | `0.686` | `2.53×` |
| `canonical_exch` | `0.140` | `0.052` | `2.68×` | `1.609` | `0.596` | `2.70×` |
| `canonical_ind` | `1.835` | `0.643` | `2.85×` | `23.520` | `8.745` | `2.69×` |
| `fragment_localization` | `0.164` | `0.569` | `0.29×` | `0.427` | `2.053` | `0.21×` |
| `fragment_elst` | `1.384` | `2.089` | `0.66×` | `11.008` | `7.099` | `1.55×` |
| `fragment_exch` | `0.528` | `0.028` | **18.56×** | `3.695` | `0.215` | **17.20×** |
| `fragment_ind` | `141.302` | `14.346` | **9.85×** | `3,286.270` | `154.786` | **21.23×** |
| `fragment_disp` | `2.846` | `0.164` | **17.33×** | `23.442` | `1.983` | **11.82×** |
| **Instrumented total** | **`155.959`** | **`28.926`** | **5.39×** | **`3,393.658`** | **`204.402`** | **16.60×** |
| **Uninstrumented baseline** | **`154.467`** | **`22.900`** | **6.75×** | **`3,393.658`** *(inst)* | **`186.899`** | **18.14×** |

## Projected H200 SXM performance (estimates pending job `16057336`)

Scaling the measured FP64-GEMM-bound components by ~10×–20× (34–67 TFLOPS FP64 vs ~1.9 TFLOPS) and bandwidth/setup components by ~1.5×–3× (anchored to [H200_PEPTIDE_TIMINGS.md](H200_PEPTIDE_TIMINGS.md)):

| Benchmark | CPU reference | Measured RTX 6000 (`16084802`) | **Est. H200 time** | **Est. H200 speedup** |
|---|---:|---:|---:|---:|
| `peptide` `fdisp0` | `2.85 s` (CPU8) | `0.164 s` (`17.3×`) | `~0.08–0.12 s` | **~24×–35×** |
| `fsapt1` `fdisp0` | `4.69 s` (CPU8) | `0.337 s` (`13.9×`) | `~0.12–0.18 s` | **~26×–39×** |
| `nanotube` `SAPT0` `disp` | `64.77 s` (CPU8) | `4.27 s` (`15.2×`) | `~2.0–2.8 s` | **~23×–32×** |
| `protein83` `SAPT0` `disp` | `377.45 s` (CPU8) | `73.97 s` (`5.1×`) | `~14–20 s` | **~19×–27×** |
| `protein83` fixed-HF `fdisp0` | `69.51 s` (**CPU48**) | `57.11 s` (`1.2×` vs CPU48) | **`~4–6 s`** | **~12×–17× vs CPU48** (~60×–90× vs CPU8) |
| `fsapt1` whole F-SAPT | `405.8 s` (CPU8) | `42.7 s` (`9.5×`) | `~19–25 s` | **~16×–21×** |
| `peptide` whole F-SAPT | `240.6 s` (H200-host CPU8) | `22.9 s` *(old H200: `33.0 s`)* | `~16–20 s` | **~12×–15×** |
| `nanotube` whole F-SAPT | `3,393.7 s` (CPU8) | `186.9 s` (`18.1×`) | **`~30–45 s`** | **~75×–110×** |
| `protein83` whole `SAPT0+δHF` | `770.0 s` (CPU8) | `261.1 s` (`2.95×`) | **`~50–80 s`** | **~10×–15×** |
| `protein83` whole F-SAPT | `11,456 s` (**CPU48**) | *(80 GiB cap; fits H200 120 GiB)* | **`~130–250 s`** | **~45×–85× vs CPU48** |

## Accompanying Cachy RTX 3090 correctness & sanitizer gates (`85d7429837`, debug-only)

- **`compute-sanitizer` (4/4 clean, `--error-exitcode 9`):** `memcheck --leak-check full` (`0 errors`, 0 leaks), `initcheck` (`0 errors`), `racecheck` (`0 hazards`), `synccheck` (`0 errors`) on `verify_disp0_opt_gpu.py --mode sanitizer`.
- **Zero-GPU-skip `pytest --junitxml`:** `61 passed, 0 skipped, 0 failed, 0 errors` (`test_cuest_jk.py`, `test_cuest_plan_reuse.py`, `test_cuest_range_separated.py`, `test_saptdft_cache_lifetime.py`).
- **Physical Psi4 GPU gate (`--mode psi4-gate`):** `PASS_PSI4_PHYSICAL_GPU_GATE` across `water_ne`, `water_dimer`, `ethane_water`, `peptide` (`NONE` + `FISAPT`), CPU parity `<= 4.87e-8 Eh`, same-wavefunction blocked GPU parity `<= 8.13e-19 Eh`.
- **Fresh-process GPU-vs-CPU parity (`PASS_CACHY_PARITY`):** `fsapt1` (`|ΔE|=3.88e-10 Eh`, 172/172 `fsapt.py` cells `<= 1.36e-9 Eh`), `peptide` `SAPT0` (`|ΔE|=2.31e-9 Eh`), `nanotube` `SAPT0` (`|ΔE|=3.65e-8 Eh`, max term `5.19e-8 Eh`).
