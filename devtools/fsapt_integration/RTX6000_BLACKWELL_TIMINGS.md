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

### 1a. Same-host 8-thread CPU (`CPU8`) vs. GPU (`8` host threads)

| Stage / System | AO (`nbf`) | Mode | CPU8 (s) | RTX 6000 (s) | RTX 6000 vs CPU8 | `disp`/`fdisp0` CPU8 → RTX 6000 | Numerical parity (`\|ΔE\|` / max term / oracle) |
|---|---:|---|---:|---:|---:|---|---|
| **Stage 1** Water dimer gate | 82 | 10-proc F-SAPT gate | — | `134.6` (all 10 procs) | — | — | **PASS** (`PASS_SMALL_H200_ONLY`, all signed + same-gauge + repeat + displaced) |
| **Stage 2** `fsapt1` (S22 phenol dimer, `E10/D12`) | 312 | F-SAPT + `fsapt.py` | `405.79` | `42.71` (`42.60`, `42.83`) | **9.50×** | `4.69 s → 0.337 s` (**13.9×**) | **PASS**: `\|ΔE\| = 3.77e-9 Eh`; same-gauge max matrix `1.83e-10 Eh`; **172/172** `fsapt.py` cells (`OH`/`PH`) max diff `8.22e-9 Eh` (`5.16e-6 kcal/mol`), GPU repeat `3.06e-11 Eh` |
| **Stage 3a** `peptide` (`E10/D10`, 3×3) | 250 | Canonical `SAPT0+δHF` | `17.43` | `13.74` | **1.27×** | `9.57 s → 0.876 s` (**10.92×**) | **PASS**: `\|ΔE\| = 1.78e-10 Eh`; max raw term `2.72e-9 Eh` (`Disp20,u` `2.83e-13 Eh`, `δHF` `4.55e-13 Eh`) |
| **Stage 3b** `peptide` (`E10/D12`, 3×3 baseline) | 250 | Full F-SAPT | `154.47` | `22.90` *(old H200 `15472290`: `32.95 s`)* | **6.75×** | `2.85 s → 0.164 s` (**17.33×**; *old H200: `2.52 s`*) | **PASS**: same-gauge max matrix `1.27e-11 Eh`; all 3 repeats passed |
| **Stage 4a** `nanotube` (`E10/D10`, 3×3) | 548 | Canonical `SAPT0+δHF` | `138.28` (`cpu-0`) | `41.73` (`31.41` warm) | **3.31×** | `64.77 s → 4.27 s` (**15.17×**, blocked `1.90 s`) | 8/8 workers `PASS`; 6/7 terms `<= 5.6e-9 Eh` (`Disp20,u` `3.05e-9 Eh`), GPU repeats `<= 2.34e-10 Eh`; `Elst10,r` `-1.31e-7 Eh` (`cond(S)~1e9` + `D10` CPU floor; Phoenix GPU matches Cachy CPU within `1.5e-8 Eh`) |
| **Stage 4b + 7** `nanotube` (`E10/D10`) | 548 | Full F-SAPT | `3,393.66` | `187.05` (`186.90` med; `204.40` inst) | **18.14×** | `23.44 s → 1.98 s` (**11.82×**) | **PASS**: `\|ΔE\| = 1.98e-8 Eh`; `Disp20,u` `-9.15e-10 Eh`, `Exch-Disp20,u` `+1.00e-10 Eh`; all 15 signed terms passed; same-gauge oracle (23 matrices) max diff **`1.45e-8 Eh`** |
| **Stage 5b** `protein83` (`E10/D10`, 1×1 + warm/prof) | 922 | Canonical `SAPT0+δHF` | `769.99` | `261.10` (`249.54` warm) | **2.95×** | `377.45 s → 73.97 s` (**5.10×**) | **PASS**: `\|ΔE\| = 7.67e-9 Eh`; max term `Elst10,r` `1.26e-8 Eh` (`Disp20,u` `8.56e-10 Eh`, `δHF` `7.51e-11 Eh`); warm/cold matched `< 1e-9 Eh` |

### 1b. Frozen 48-thread `cpu-gnr` CPU (`CPU48`, jobs `15474909`/`15474910`) vs. GPU (`8` host threads)

| Stage / System | AO (`nbf`) | Mode | CPU48 (s) | RTX 6000 (s) | RTX 6000 vs CPU48 | Notes / Numerical parity |
|---|---:|---|---:|---:|---:|---|
| **Stage 5a** `protein83` fixed-HF `fdisp0` (`jun-cc-pVDZ`) | 922 | Fixed-HF `fdisp0` | `69.51` | `57.11` (`3.53 s` DFMO + `53.5 s` FP64 GEMM, 1 call, `46.26 GiB` peak) | **1.22×** | **PASS**: `Disp20` pairs **`1.74e-14 Eh`**, `Exch-Disp20` pairs **`5.94e-14 Eh`** vs CPU48 snapshot; same-session repeat `0.0` scalar / `1.08e-19 Eh` pair |
| **Stage 5c** `protein83` full F-SAPT (`jun-cc-pVDZ`) | 922 | Full F-SAPT | `11,456` | *(80 GiB budget cap)* | — | All 3 SCFs converged cleanly; hit `80 GiB` Blackwell declared budget by `0.21 GiB` when building sibling RI plan (`52.91 + 19.30 + 8.00 = 80.21 GiB`; H200 `120 GiB` budget has ~40 GiB headroom) |

### 1c. Additional non-CPU-ratio probes

- **Stage 5a** `protein83` fixed-HF `fdisp0` (`def2-SVP`, 561 AO, vs frozen H200 snapshot `15337409`): **PASS** (`24.34 s`, `1.62 s` 1-call DFMO, `20.33 GiB` peak; `Disp20` pairs `1.59e-16 Eh`, `Exch-Disp20` pairs `1.59e-17 Eh`).
- **Stage 6** `protein157` bounded GPU probe (1,786 AO): Clean OOM in dimer SCF `cuESTJK` plan creation (`76.53 GB` requested, `28.59 GB` of `102.07 GB` free on 96 GiB Blackwell).

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

### Projected vs. Same-Host 8-Thread CPU (`CPU8`)

| Benchmark | CPU8 (s) | Measured RTX 6000 (`16084802`) | **Est. H200 time** | **Est. H200 vs CPU8** |
|---|---:|---:|---:|---:|
| `peptide` `fdisp0` | `2.85` | `0.164 s` (`17.3×`) | `~0.08–0.12 s` | **~24×–35×** |
| `fsapt1` `fdisp0` | `4.69` | `0.337 s` (`13.9×`) | `~0.12–0.18 s` | **~26×–39×** |
| `nanotube` `SAPT0` `disp` | `64.77` | `4.27 s` (`15.2×`) | `~2.0–2.8 s` | **~23×–32×** |
| `protein83` `SAPT0` `disp` | `377.45` | `73.97 s` (`5.1×`) | `~14–20 s` | **~19×–27×** |
| `fsapt1` whole F-SAPT | `405.8` | `42.7 s` (`9.5×`) | `~19–25 s` | **~16×–21×** |
| `peptide` whole F-SAPT | `240.6` *(H200-host CPU8)* | `22.9 s` *(old H200: `33.0 s`)* | `~16–20 s` | **~12×–15×** |
| `nanotube` whole F-SAPT | `3,393.7` | `186.9 s` (`18.1×`) | **`~30–45 s`** | **~75×–110×** |
| `protein83` whole `SAPT0+δHF` | `770.0` | `261.1 s` (`2.95×`) | **`~50–80 s`** | **~10×–15×** |

### Projected vs. 48-Thread `cpu-gnr` CPU (`CPU48`)

| Benchmark | CPU48 (s) | Measured RTX 6000 (`16084802`) | **Est. H200 time** | **Est. H200 vs CPU48** |
|---|---:|---:|---:|---:|
| `protein83` fixed-HF `fdisp0` | `69.51` | `57.11 s` (`1.22×`) | **`~4–6 s`** | **~12×–17×** |
| `protein83` whole F-SAPT | `11,456` | *(80 GiB cap; fits H200 120 GiB)* | **`~130–250 s`** | **~45×–85×** |

## Accompanying Cachy RTX 3090 correctness & sanitizer gates (debug-only)

- **Pre-merge (`85d7429837`) & post-merge (`8ee9cbe8d2`, with `origin/saptdft_cuest` `2cd959b9ae` merged in):**
  - **`compute-sanitizer` (4/4 clean, `--error-exitcode 9`):** `memcheck --leak-check full` (`0 errors`, 0 leaks), `initcheck` (`0 errors`), `racecheck` (`0 hazards`), `synccheck` (`0 errors`) on `verify_disp0_opt_gpu.py --mode sanitizer`.
  - **Zero-GPU-skip core `pytest --junitxml`:** `65 passed, 0 skipped, 0 failed, 0 errors` (`test_cuest_jk.py` incl. 4 integral-direct DF tests, `test_cuest_plan_reuse.py`, `test_cuest_range_separated.py`, `test_saptdft_cache_lifetime.py`).
  - **Merged `saptdft_cuest` `test_fsaptdft.py`:** `21 passed, 0 skipped` (all 12 GPU/CPU `DF`/`AUTO`/`CUEST_JK`/`CUEST_DFMO`/`CUEST_DFMO_DIRECT`/`CUEST_DFMO_UNSCREENED` `sapt(dft)-d4(i)` cases with `FISAPT.reuse_sapt_jk` and `0` fragment DF plan creations verified); `test_saptdft.py -k sapt_hf` (`5/5` pass); `devtools/fsapt_integration/test_*_probe.py` (`21/21` pass).
  - **Physical Psi4 GPU gate (`--mode psi4-gate`):** `PASS_PSI4_PHYSICAL_GPU_GATE` across `water_ne`, `water_dimer`, `ethane_water`, `peptide` (`NONE` + `FISAPT`), CPU parity `<= 4.87e-8 Eh`, same-wavefunction blocked GPU parity `<= 8.13e-19 Eh`.
  - **Fresh-process GPU-vs-CPU parity (`PASS_CACHY_PARITY`, `85d7429837`):** `fsapt1` (`|ΔE|=3.88e-10 Eh`, 172/172 `fsapt.py` cells `<= 1.36e-9 Eh`), `peptide` `SAPT0` (`|ΔE|=2.31e-9 Eh`), `nanotube` `SAPT0` (`|ΔE|=3.65e-8 Eh`, max term `5.19e-8 Eh`).

---

## Merged-branch Phoenix RTX PRO 6000 Blackwell verification (`a6ce45a6bc`, 2026-11-24)

After merging `origin/saptdft_cuest` (`2cd959b9ae`) into `saptdft_cuest_disp0` (`8ee9cbe8d2`, helper commit `a6ce45a6bc`), two separate Phoenix CPU builds compiled cleanly ahead of GPU execution:
- **Build A (`16096764`, `disp0-current-build`, `COMPLETED 0:0` in `11:45`):** `result.json` `PASS` (`426` CPU guards), `cpu-sealed-admission` `PASS`, cuEST `0.3.0.1` + CUDA `13.3`.
- **Build B (`16096766`, `d4i-0301-build`):** compiled `[34/34]` with cuEST `0.3.0.1`, CUDA `13.5`, and CUDA LibXC `libxc-7.1.2-cuda-sm80-89-90-120` (`-DUSING_Libxc_CUDA -DPSI4_CUEST_NONSYMMETRIC_XC`).

### A. Merged ab initio `disp0`/`fdisp0` + coupled F-SAPT0 suite — job **`16099563`** (`PASS_SUITE`, `0:0`, `02:47:50`)

- Node `atl1-1-02-014-23-0`, `gts-cs207 / gpu-rtxpro-blackwell / inferno`, `budget_gib=86`, `stage6_protein157_gpu_probe` skipped by manifest.
- Manifest SHA256: `1eb1c78ec19bc113c3b5fa7cd2f4475f4e87ee5cb001155f404609b15a355cfa`.
- `suite-result.json` SHA256: `e59787d9a05eb4928ab56fe999e42c6197934531dbc4d5ffa6b83a436f1b828c`.
- Remote artifact directory: `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/disp0-0301-rtx6000-a6ce45a6bc-01/job-16099563`.

#### A1. Same-host `CPU8` vs. RTX 6000 (`16099563`, spherical `jun-cc-pVDZ`, native FP64)

| Stage / System | AO | Mode | CPU8 (s) | RTX 6000 (s) | RTX 6000 vs CPU8 | `disp`/`fdisp0` CPU8 → RTX 6000 | Numerical parity |
|---|---:|---|---:|---:|---:|---|---|
| **Stage 1** Water gate | 82 | 10-proc F-SAPT | — | `130.6` | — | — | **PASS** |
| **Stage 2** `fsapt1` (`E10/D12`) | 312 | F-SAPT + `fsapt.py` | `397.28` | `39.82` (`39.90`, `39.74`) | **9.98×** | `3.94 s → 0.338 s` (**11.6×**) | **PASS**: `\|ΔE\| 3.78e-9`, oracle `1.85e-10`, 172/172 cells `<= 8.13e-9 Eh` |
| **Stage 3a** `peptide` (`E10/D10`, 3×3) | 250 | Canonical `SAPT0+δHF` | `30.30`\* | `32.65`\* (`14.04` min) | `0.93×`\* | `23.20 s → 0.861 s` (**26.9×**) | **PASS**: `\|ΔE\| 1.80e-10`, max term `2.72e-9 Eh` |
| **Stage 3b** `peptide` (`E10/D12`, 3×3) | 250 | Full F-SAPT | `163.59` | `22.34` | **7.32×** | `2.63 s → 0.166 s` (**15.79×**) | **PASS**: oracle `<= 1.3e-11 Eh`, all 3 repeats pass |
| **Stage 4b + 7** `nanotube` (`E10/D10`) | 548 | Full F-SAPT | `3,299.31` | `187.53` (`187.63` med) | **17.59×** | `23.63 s → 1.99 s` (**11.87×**) | **PASS**: `\|ΔE\| 1.98e-8 Eh`, all 15 signed terms pass |
| **Stage 5b** `protein83` (`E10/D10`) | 922 | Canonical `SAPT0+δHF` | `815.98` | `263.89` | **3.09×** | `403.33 s → 74.10 s` (**5.44×**) | **PASS**: `\|ΔE\| 7.56e-9`, max term `1.25e-8 Eh` |

*\*During `stage3a`, Lustre `/storage/scratch1` checkpoint contention from concurrent `16098082` on the same node added ~15–29 s to `delta_hf` (min GPU sample `14.04 s` matches single-tenant `16084802`'s `13.74 s`). Fixed for future runs by enforcing node-local NVMe scratch in commit `939bf919f6`.*

#### A2. Frozen `CPU48` (`15474909`/`15474910`) vs. RTX 6000 (`16099563`, spherical `jun-cc-pVDZ`, native FP64)

| Stage / System | AO | Mode | CPU48 (s) | RTX 6000 (s) | RTX 6000 vs CPU48 | Notes |
|---|---:|---|---:|---:|---:|---|
| **Stage 5a** `protein83` fixed-HF `fdisp0` | 922 | Fixed-HF `fdisp0` | `69.51` | `59.49` (`1` DFMO call, `46.26 GiB` peak) | **1.17×** | **PASS** (`def2-SVP` also **PASS**: `25.44 s`, `1` DFMO call, `20.33 GiB` peak) |
| **Stage 5c** `protein83` full F-SAPT | 922 | Full F-SAPT (`86 GiB` cap) | `11,456` | *(H200-limited)* | — | Passed all 3 SCFs, sibling RI plan, IBO, `felst`, `fexch`; hit `86 GiB` cap inside `coupled_find_device` (`72.37 GiB` live + `9.17 GiB` req + `8.00 GiB` reserve = `89.54 GiB`, `11.31 GiB` physically free because JKFIT & RIFIT plans stay live concurrently) |

---

### B. Merged `saptdft_cuest` `F-SAPT(HF/PBE0)-D4(I)/cc-pVDZ` suite — job **`16098082`** (`COMPLETED 0:0`, `49:24`, `20/20` PASS)

- Node `atl1-1-02-014-23-0`, `gts-cs207 / gpu-rtxpro-blackwell / inferno`, Build B `16096766` (cuEST `0.3.0.1`, CUDA 13.5, CUDA LibXC `sm_120`, `cuest_mixed_precision=True`, 8 host threads).
- Remote artifact directory: `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/d4i-0301-rtx6000-a6ce45a6bc-01/job-16098082`.
- All 10 fragment runs confirmed `reused_scalar_plan=True`, **`0` fragment DF plan creations**, component closure `<= 4.79e-10 Eh`, plain-vs-fragment scalar parity `<= 5.70e-8 Eh`.
- `cdk2_c2` QM/MM residue-group parity vs. Phoenix CPU16: HF max Total error **`1.58e-5 kcal/mol`** (`< 0.001`), PBE0 max Total error **`2.91e-3 kcal/mol`** (`< 0.005`, matching H200 `15825734`'s `2.88e-3 kcal/mol`).

#### B1. `F-SAPT(HF/PBE0)-D4(I)/cc-pVDZ` vs. Phoenix `CPU8` (job `15799432`, [D4I_TERM_TIMES.csv](D4I_TERM_TIMES.csv))

| System (`nbf`) | Method | CPU8 (s) | RTX 6000 (s) | RTX 6000 vs CPU8 | Peak VRAM (GiB) | Fragment timers (`felst` / `fexch` / `find`, s) |
|---|---|---:|---:|---:|---:|---|
| `water` (`48`, embedded) | HF F-SAPT | `4.37` | `3.56` | **1.23×** | `0.85` | `0.008` / `0.001` / `0.014` |
| `water` (`48`, embedded) | PBE0 F-SAPT | `10.32` | `6.73` | **1.53×** | `5.44` | `0.006` / `<0.001` / `0.036` |
| `peptide` (`210`) | HF F-SAPT | `12.28` | `5.42` | **2.27×** | `1.69` | `0.044` / `0.008` / `0.107` |
| `peptide` (`210`) | PBE0 F-SAPT | `75.47` | `21.01` | **3.59×** | `6.50` | `0.049` / `0.008` / `0.451` |
| `protein83` (`766`) | HF F-SAPT | `347.40` | `36.62` | **9.49×** | `12.59` | `0.842` / `0.666` / `1.918` |
| `protein83` (`766`) | PBE0 plain | `1,069.31` | `139.88` | **7.64×** | `18.27` | *(plain; CPU8 PBE0 frag timed out at 1h)* |

#### B2. `F-SAPT(HF/PBE0)-D4(I)/cc-pVDZ` vs. Phoenix `CPU16` (`15801202` / `15801531` / `15826017`) and measured H200 (`15825734`)

| System (`nbf`) | Method | CPU16 (s) | RTX 6000 (s) | RTX 6000 vs CPU16 | H200 `15825734` (s) | H200 vs CPU16 | Peak VRAM RTX / H200 (GiB) |
|---|---|---:|---:|---:|---:|---:|---|
| `protein83` (`766`) | PBE0 F-SAPT | `602.99` | `119.24` | **5.06×** | — | — | `18.28` / — |
| `cdk2_c2` (`995`, +8,977 MM) | HF F-SAPT | `677.47` | `74.48` | **9.10×** | `31.92` | **21.22×** | `18.11` / `16.78` |
| `cdk2_c2` (`995`, +8,977 MM) | PBE0 F-SAPT | `1,638.12` | `217.83` | **7.52×** | `107.46`\* | **15.24×**\* | `23.20` / `22.20` |

*\*H200 `15825734` `cdk2_c2` PBE0 fragment hit a `36.84 s` Lustre `Post-SCF` checkpoint stall in `GRAC Monomer A` cation (`compute_energy` was `4.42 s`, identical to `plain`'s `4.65 s`; without the Lustre stall, clean H200 PBE0 F-SAPT is `~70.7 s` = `~23.2×` vs CPU16).*

#### B3. Large-system GPU-only `F-SAPT-D4(I)/cc-pVDZ` (`protein157`, `1,478` AO)

| System (`nbf`) | Method | Plain RTX 6000 (s) | Fragment RTX 6000 (s) | Fragment H200 `15825734` (s) | Peak VRAM RTX / H200 (GiB) | RTX 6000 `felst` / `fexch` / `find` (s) |
|---|---|---:|---:|---:|---|---|
| `protein157` (`1,478`) | HF | `278.58`\* | `374.64`\* | `69.10` | `44.39` / `41.29` | `4.19` / `11.86` / `128.2`\* |
| `protein157` (`1,478`) | PBE0 | `627.17`\* | `627.52` | `188.03` | `45.73` / `42.23` | `4.26` / `11.88` / `15.91` |

*\*Lustre `/storage/scratch1` `Post-SCF` checkpoint stalls (`94.63 s` in `protein157-HF-plain`, `41.22 s` + `7.67 s` in `protein157-HF-fragment`, `22.51 s` + `10.84 s` in `protein157-PBE0-plain`) and disk-backed `find` contention (`128.2 s` vs `15.91 s` in PBE0) motivated enforcing node-local NVMe scratch across all Phoenix runners in commit `939bf919f6`.*

---

## H200 SXM (`141 GiB`, node-local NVMe `xfs`) & Unified `CPU16` Measured Timings (`436ed46555` / `a6ce45a6bc`, 2026-11-27–28)

### Provenance

- **Unbounded physical VRAM + ~56 GiB VRAM reclaim (`5e5acaacc3`, `c47ecd4f1f`, `436ed46555`):**
  - Removed artificial upfront VRAM budget blocking (`CUDASession(memory_limit_bytes=None, native_persistent_bytes=0)` runs directly to physical `cudaMalloc`, raising `MemoryError` only on real `cudaErrorMemoryAllocation`).
  - Released cuEST creation-temporary workspaces immediately after `cuest*Create` + `synchronize_device()` in `DiagnosticPlanOwner` (~56 GiB reclaimed on `protein83`).
  - Deferred `RIFIT` `sibling_provider` construction in `sapt_fisapt_workflow_cuda.py` until after `coupled_find_device` releases `HFResponseOperator` (~28 GiB), cutting `protein83` full coupled F-SAPT0 peak VRAM from `~109 GiB` → **`53.59 GiB`** (fits comfortably on both 96 GiB RTX PRO 6000 Blackwell and 141 GiB H200 SXM).
- **Phoenix CPU builds (`COMPLETED 0:0`, cuEST `0.3.0.1`):**
  - Build `16152230` (`c47ecd4f1f`, `cpu-small / inferno`, `15:38`, `cpu-sealed-admission` `PASS`).
  - Build **`16270782`** (`436ed46555`, `cpu-small / embers`, `10:03`, `441` CPU guards, `result.json` `PASS`, `cpu-sealed-admission` `PASS` with `device_memory_limit_bytes=None`, `native_persistent_bytes=0`).
- **Node-local NVMe `CPU16` baselines (`COMPLETED 0:0`, `FSTYPE=xfs`):**
  - Ab initio `CPU16` **`16260393`** (`cpu-small / embers`, `atl1-1-01-003-12-2`, `01:22:23`, Build `16152230`, `879 GB` free `/tmp`).
  - `D4(I)` `CPU16` **`16112589`** (`cpu-small / embers`, `atl1-1-02-009-7-1`, `09:10`, Build B `16096766`; combined with `15801202` `protein83` PBE0, `15801531` `cdk2_c2` HF, `15826017` `cdk2_c2` PBE0).
- **Node-local NVMe H200 SXM (`143,771 MiB`) GPU runs (`FSTYPE=xfs`):**
  - **Ab initio `disp0`/`fdisp0` + coupled F-SAPT0 suite `16272610`** (`gpu-h200 / embers`, `atl1-1-03-020-18-0`, Build **`16270782`** `436ed46555`, **`PASS_SUITE`**, `exit_code=0`, `scratch_cleanup.complete=True`; `stage1`, `stage2`, `stage3a`, `stage3b`, `stage4b`, `stage5a`, `stage5b`, `stage5c` all **`PASS`**; `embers` preempted at `01:59:33` during `stage7` CPU8 nanotube, superseded by `CPU16` `16260393`).
    - Manifest SHA256: `69744670998f5523f04856a73cfe56cb19a177aa1a2a691bc58a66bb1e988b3a`
    - `suite-result.json` SHA256: `e56968ac4375796600f345cd2111618b94876c55b2104a332f2d2d9aa5439e0a`
    - Remote artifact directory: `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/disp0-0301-suite-436ed46555-01/job-16272610`
  - **Merged `F-SAPT(HF/PBE0)-D4(I)/cc-pVDZ` suite `16112597`** (`gpu-h200 / embers`, `atl1-1-02-014-16-0`, Build B `16096766`, **`COMPLETED 0:0`** in `14:57`, **`20/20 PASS`**, `reused_scalar_plan=True`, `0` fragment DF plans, closure `<= 4.79e-10 Eh`, plain-vs-fragment `<= 4.72e-8 Eh`, `cdk2_c2` residue parity vs CPU16: HF `1.67e-5 kcal/mol`, PBE0 `2.89e-3 kcal/mol`).

### C1. Ab Initio `SAPT0+δHF` & Coupled `F-SAPT0` — Consistent `CPU16` (`16260393`, NVMe) vs. RTX 6000 & H200 (`16272610`, NVMe, spherical `jun-cc-pVDZ`, native FP64)

| System (`nbf`) | Mode | CPU16 `16260393` (s) | RTX 6000 (s) | RTX 6000 vs CPU16 | **H200 `16272610` (s)** | **H200 vs CPU16** | `disp`/`fdisp0` CPU16 → H200 | H200 Peak VRAM (GiB) | Numerical parity (`16272610`) |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| `fsapt1` (`312`, `E10/D12`) | Full F-SAPT0 + `fsapt.py` | `404.96` | `39.82` (`16099563`) | `10.17×` | **`21.25`** (`22.25`, `20.25`) | **`19.06×`** | `4.33 s → 0.0392 s` (**`110.5×`**) | `2.94` | **PASS**: `\|ΔE\| 6.34e-9`, oracle `1.78e-10`, **172/172** cells `<= 9.58e-10 Eh` (`6.01e-7 kcal/mol`), repeat `7.67e-12 Eh` |
| `peptide` (`250`, `E10/D10`) | `SAPT0+δHF` | `16.23` | `13.74` (`16084802`) | `1.18×` | **`6.14`** (`7.42`, `6.14`, `5.09`) | **`2.64×`** | `6.22 s → 0.957 s` (**`6.50×`**) | — | **PASS**: `\|ΔE\| 2.57e-10 Eh`, `Disp20,u` `3.91e-13 Eh`, `δHF` `9.47e-13 Eh` |
| `peptide` (`250`, `E10/D12`) | Full F-SAPT0 | `195.31` | `22.34` (`16099563`) | `8.74×` | **`15.48`** (`14.85`, `15.48`, `16.46`) | **`12.61×`** | `2.39 s → 0.0439 s` (**`54.4×`**) | `2.00` | **PASS**: same-gauge oracle **`1.44e-11 Eh`**, all 3 repeats pass |
| `nanotube` (`548`, `E10/D10`) | `SAPT0+δHF` | `126.03` | `41.73` (`16084802`) | `3.02×` | **`10.33`** (`8.06` warm)\* | **`12.20×`** (`15.64×` warm) | `51.26 s → 1.25 s` (**`41.1×`**, `56.7×` warm) | — | 8/8 workers `PASS`; 6/7 terms `<= 6.2e-9 Eh`; known `cond(S)~1e9` `D10` `Elst10,r` floor `1.18e-7 Eh` |
| `nanotube` (`548`, `E10/D10`) | Full F-SAPT0 | `3,366.92` | `187.53` (`16099563`) | `17.95×` | **`37.09`** (`37.15`, `37.04`) | **`90.77×`** | `20.86 s → 0.150 s` (**`139.0×`**) | `11.51` | **PASS** (`1` DFMO call; `fragment_ind` `3,256.33 s → 35.69 s` = **`91.23×`**) |
| `protein83` (`922`, `E10/D10`) | `SAPT0+δHF` | `724.14` | `263.89` (`16099563`) | `2.74×` | **`28.85`** | **`25.10×`** | `321.46 s → 5.45 s` (**`58.95×`**) | — | **PASS**: `\|ΔE\| 7.99e-8 Eh`, `Disp20,u` `4.89e-10 Eh`, `δHF` `1.56e-11 Eh` |

*\*`fdisp0` values for F-SAPT0 rows report pure `blocked_execution_seconds` (`1` DFMO call); instrumented `fragment_disp` windows including deferred RIFIT plan creation/teardown are `0.746 s` (`peptide`), `1.54 s` (`nanotube`), and `5.52 s` (`protein83`). Same-host H200 `CPU8` ratios inside `16272610`: `fsapt1` `36.14×` (`767.79 → 21.25 s`), `peptide` F-SAPT `14.04×` (`217.39 → 15.48 s`), `protein83` `SAPT0+δHF` `33.44×` (`964.85 → 28.85 s`, `disp` `66.50×`).*

**`protein83` `SAPT0+δHF` per-term H200 (`16272610`) vs. `CPU16` (`16260393`):** `disp` `321.46 → 5.45 s` (**`58.95×`**), `ind` `166.44 → 6.65 s` (**`25.03×`**), `exch` `8.47 → 0.420 s` (**`20.17×`**), `delta_hf` `123.13 → 7.66 s` (**`16.07×`**), `shared_setup_other` `104.63 → 8.66 s` (**`12.08×`**).

### C2. Separate `CPU48` Baseline Table (`15474909`/`15474910`, `cpu-gnr`, `jun-cc-pVDZ`) vs. RTX 6000 & H200 (`16272610`)

| System (`nbf`) | Mode | CPU48 (s) | RTX 6000 `16099563` (s) | **H200 `16272610` (s)** | **H200 vs CPU48** | H200 Peak VRAM (GiB) | Numerical parity & Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `protein83` (`922`) | Fixed-HF `fdisp0` | `69.51` | `59.49` (`1.17×`, `46.26 GiB`) | **`2.55`** (`2.54` blocked, `0.352 s` DFMO) | **`27.21×`** | **`31.61`** | **PASS**: `Disp20` pairs **`1.53e-14 Eh`**, `Exch-Disp20` **`4.89e-14 Eh`**, repeat **`3.79e-19 Eh`** (`def2-SVP` 561 AO: `1.11 s`, `14.46 GiB`, err `2.76e-16 Eh`) |
| `protein83` (`922`, `E10/D12`) | **Full Coupled F-SAPT0** | **`11,456.28`** (`3h 11m`) | *(pre-fix 86 GiB cap; now fits)* | **`277.99`** (`277.74`, `278.24`; `4m 38s`) | **`41.21×`** | **`53.59`** | **PASS** vs both frozen CPU48 refs: `\|ΔE\| 5.01e-9 Eh`, `Disp20,u` `8.03e-11 Eh`, `Exch_AB` `2.11e-12 Eh`, `IndAB_AB` `2.23e-11 Eh`, `IndBA_AB` `1.94e-10 Eh`, `Elst_AB` `-1.44e-7 Eh` |

- `protein83` full F-SAPT0 H200 instrumented breakdown (`352.26 s` traced): `scf_dimer` `9.72 s`, `scf_A` `6.02 s`, `scf_B` `6.16 s`, `canonical_ind` `5.50 s`, `fragment_localization` `2.64 s`, `fragment_elst` `33.40 s`, `fragment_exch` `0.425 s`, `fragment_ind` `274.89 s`, `fragment_disp` `5.52 s` (`2.56 s` blocked, `0.353 s` 1-call DFMO).
- `stage6` `protein157` (`1,786` AO `jun-cc-pVDZ`) unbounded physical VRAM probe: allocated `88.54 GiB` persistent `RIFIT` DF plan directly on H200 and raised clean `MemoryError` (`cudaErrorMemoryAllocation`) only when cuEST `RIFIT` plan creation requested `137.45 GiB` creation-temp with `50.22 GiB` physically free (`226 GiB > 140 GiB` physical H200 VRAM), confirming zero artificial software blocking.

### C3. Unified `CPU16` vs. RTX 6000 (`16098082`) & H200 (`16112597`, NVMe `xfs`) `F-SAPT(HF/PBE0)-D4(I)/cc-pVDZ`

*Consistent `CPU16` baseline: `water`, `peptide`, `protein83` HF from `16112589` (NVMe `xfs`); `protein83` PBE0 from `15801202`; `cdk2_c2` HF from `15801531`; `cdk2_c2` PBE0 from `15826017`.*

| System (`nbf`) | Method | CPU16 (s) | RTX 6000 `16098082` (s) | RTX 6000 vs CPU16 | **H200 `16112597` (s)** | **H200 vs CPU16** |
|---|---|---:|---:|---:|---:|---:|
| `water` (`48`, embedded) | HF F-SAPT | `3.85` | `3.56` | `1.08×` | **`3.66`** | **`1.05×`** |
| `water` (`48`, embedded) | PBE0 F-SAPT | `7.78` | `6.73` | `1.16×` | **`5.87`** | **`1.33×`** |
| `peptide` (`210`) | HF F-SAPT | `10.59` | `5.42` | `1.95×` | **`4.86`** | **`2.18×`** |
| `peptide` (`210`) | PBE0 F-SAPT | `50.73` | `21.01` | `2.41×` | **`11.90`** | **`4.26×`** |
| `protein83` (`766`) | HF F-SAPT | `196.38` | `36.62` | `5.36×` | **`16.02`** | **`12.26×`** |
| `protein83` (`766`) | PBE0 F-SAPT | `602.99` | `119.24` | `5.06×` | **`44.99`** | **`13.40×`** |
| `cdk2_c2` (`995`, +8,977 MM) | HF F-SAPT | `677.47` | `74.48` | `9.10×` | **`28.27`** | **`23.96×`** |
| `cdk2_c2` (`995`, +8,977 MM) | PBE0 F-SAPT | `1,638.12` | `217.83` | `7.52×` | **`72.55`** | **`22.58×`** |

**GPU-only `protein157` (`1,478` AO, `cc-pVDZ`, `F-SAPT-D4(I)`):**
- HF F-SAPT: RTX 6000 `16098082` `374.64 s`\* vs. **H200 `16112597` `63.42 s`** (`5.91×` faster than RTX 6000).
- PBE0 F-SAPT: RTX 6000 `16098082` `627.52 s` vs. **H200 `16112597` `184.77 s`** (`3.40×` faster than RTX 6000).

**`cdk2_c2` (`995` AO + `8,977` MM) per-term H200 (`16112597`, NVMe) vs. `CPU16`:**
- **HF (`27.94 s` vs `676.97 s`, `24.23×`):** Dimer SCF `5.91 s` (`29.85×`), MonA/B SCF `3.71/3.54 s` (`13.96×`/`25.51×`), Build JK `0.72 s` (`71.52×`), `exch` `0.35 s` (`28.85×`), `ind` `5.82 s` (`33.30×`), F-SAPT Elst/Exch/Ind `1.29/0.14/2.96 s` (`39.24×`/`56.31×`/`13.19×`), D4 `2.12 s` (`1.37×`).
- **PBE0 (`72.23 s` vs `1,637.62 s`, `22.67×`):** GRAC MonA `8.60 s` (`9.36×`, down from `45.20 s` on Lustre `15825734`), GRAC MonB `11.01 s` (`11.75×`), Dimer SCF `5.57 s` (`32.10×`), MonA/B SCF `3.61/4.29 s` (`15.04×`/`21.42×`), SAPT(HF):`exch`/`ind` `0.35/5.76 s` (`28.90×`/`34.33×`), MonA/B DFT `6.70/6.12 s` (`21.40×`/`23.92×`), Build JK `1.56 s` (`33.60×`), `exch`/`ind` `0.35/9.79 s` (`29.82×`/`39.42×`), F-SAPT Elst/Exch/Ind `1.28/0.14/2.95 s` (`40.11×`/`55.49×`/`13.48×`), D4 `1.95 s` (`1.24×`).
