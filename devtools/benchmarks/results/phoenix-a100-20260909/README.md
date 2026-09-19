> **Superseded.** These rows use a fixed GRAC shift
> (`SAPT_DFT_GRAC_COMPUTE=NONE --shift 0.136`), so they time a calculation that
> presumes the shift is already known — work a user must actually do before the
> SAPT runs. Automatic GRAC changes the speedups in both directions. Quote
> [`../phoenix-h200-20260910/`](../phoenix-h200-20260910/) instead. Kept because
> it is the A100 data point and the fixed-shift comparison in that report is
> against it.
>
> Everything below is `summarize_saptdft_cuest.py` output, unedited.

# Phoenix cuEST GRAC timing and accuracy

Status: complete.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 4.31 [4.28–4.55] | 7.62 [7.61–7.66] | 0.57× | 5.327e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 5.32 [5.31–5.44] | 7.90 [7.88–8.00] | 0.67× | 3.555e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 32.39 [32.39–32.74] | 24.38 [24.29–24.58] | 1.33× | 1.597e-06 | FAIL |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 87.76 [87.63–87.93] | 28.13 [28.08–28.45] | 3.12× | 1.757e-06 | FAIL |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 37.22 [36.91–37.38] | 30.19 [30.08–30.52] | 1.23× | 1.388e-07 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 3/3 | 252.73 [252.46–253.78] | 57.34 [57.22–58.08] | 4.41× | 1.386e-07 | PASS |

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-06 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012722579836 | -0.012722597532 | 1.770e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010974216077 | 0.010974269352 | 5.327e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 4.134e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.007976185234 | -0.007976149656 | 3.558e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011298172482 | -0.011298156380 | 1.610e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010023727571 | 0.010023747022 | 1.945e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098792 | 1.130e-12 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007765301406 | -0.007765265853 | 3.555e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002262878698 | -0.002262746832 | 1.319e-07 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.010130080788 | 0.010128484214 | 1.597e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650878 | -0.001350650873 | 2.220e-11 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.005940849989 | -0.005942314692 | 1.465e-06 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.002622296893 | -0.002622473759 | 1.769e-07 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010384412066 | 0.010382831505 | 1.581e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162561 | -0.001451161830 | 8.896e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.006146448589 | -0.006148205289 | 1.757e-06 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015249545270 | -0.015249498015 | 4.734e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.013812498080 | 0.013812359302 | 1.388e-07 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319096 | -0.004728319105 | 1.372e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013891634409 | -0.013891725945 | 9.155e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.023327143678 | -0.023327281337 | 1.386e-07 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.055137117463 | 0.055137135334 | 1.789e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463908866 | -0.006463924412 | 1.556e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.002551891711 | -0.002552026958 | 1.363e-07 |


See [GPU utilization and bottleneck analysis](GPU_PROFILE.md) for sampled A100 activity, native/mixed-precision controls, and the XC/GRAC host bottleneck.

## Protocol and hardware

Measured on Phoenix, 2026-09-09, allocation **13020320**: one NVIDIA A100-SXM4-80GB and eight AMD EPYC 7543 cores (affinity 8–15). The node was shared, not exclusively reserved.

Three fresh-process measurements per backend and system/basis; identical eight-thread settings for OpenMP/MKL/OpenBLAS. No discarded warm-up. Python imports and the preliminary nbf check are outside the timer; `energy()` includes its internal setup and GPU initialization. Do not use the historical timing suite's RHF/24-thread numbers as the CPU baseline.

SAPT(DFT)-D4(I), PBE0, induction from delta-HF (`INDUCTION_TYPE=NONE`, `DO_DHF=True`), fixed GRAC shifts 0.136 Eh on both monomers, no automatic IP/GRAC calculation; 99×590 grid, SCF convergence 1e-9/1e-8, native double precision. The fixed shifts exercise GRAC but are not claimed to be physically determined for these systems. Response-based induction, GRAC gradients, and cuEST XC response are outside this evidence.

Water/benzene use their default fitting bases. Peptide and nanotube use **spherical** 6-31+G** (`PUREAM=True`), def2-universal-jkfit and aug-cc-pVDZ-RI fitting bases in **both** arms. Their nbf values (250/548) intentionally differ from the suite's Cartesian 260/574. Benzene is idealized, not the published S22 geometry. Full geometries/options are in each raw JSON.

Build: source-equivalent commit `c7927d3ec9f992ef3906f4227404c99fe9413ae2`, Psi4 `1.12a1.dev631`, Release `-O3 -DNDEBUG`; libcuEST 0.2.1.2, Einsums 1.1.2, DFT-D4 3.7.0, LibXC 7.0.0, MKL 2025.3.0, CUDA packages 13.3, driver 595.71.05, Python 3.13.11. The recorded snapshot base plus patch matches all eight files changed through the source commit. See `initial-build-provenance.json`, `hardware.json`, and protocol snapshots.

## GRAC accuracy controls

**Do not interpret all original-grid rows as passing 1e-6 Eh.** Both benzene rows miss that predeclared component threshold; the largest difference is about 0.0011 kcal/mol. The following single-run aug-cc-pVDZ controls isolate quadrature sensitivity without changing the original timing table or loosening its threshold.

| Control | Max component difference, Eh | Interpretation |
|---|---:|---|
| GPU J/K + CPU XC versus CPU | 7.360e-09 | Passes; isolates XC/grid effects |
| GPU XC versus CPU, 250×974 grid | 2.222e-07 | Passes; denser-grid agreement |
| GPU nonzero GRAC versus zero shift | 1.496e-03 | Correction demonstrably changes the result |

The denser-grid single CPU/GPU timings were 316.81/89.37 s (3.54×). This is a diagnostic **single pair**, not a three-run median.

Step **13020320.9** completed `0:0`. Separately, **51 focused regression tests** passed (zero failures/errors/skips) on the same source-equivalent build in step **13020320.5**, including density/orbital GRAC checks, CPU GRAC, SAPT-D4(I), J/K, and GEMM tests. See `pytest-fixed-4.xml` and `validation-final.json`; this is not the full Psi4 suite.

## Protein157: separate batch measurements

GPU job **13021499** completed `0:0` (batch elapsed 8m27s). Its single `energy()` measurement was **486.21 s (8.10 min)**, with **1786 basis functions**. Same spherical 6-31+G** / fitting bases and fixed GRAC protocol; eight threads, 112 GiB Psi4 memory, 128 GiB allocated host memory, one A100 80 GB. Four cuEST builders were observed with no CPU DF J/K fallback.

Largest reported allocations: DF integral plan **43.339 GiB**, J/K scratch **10.528 GiB**, AO pair list **0.007 GiB**. These are cuEST workspace reports, not an instrumented GPU peak. SLURM sampled step MaxRSS was **5,570,940 KiB** (~5.31 GiB); requested host memory is not usage.

| Component | GPU energy, Eh |
|---|---:|
| SAPT DISP ENERGY | -0.026973367165 |
| SAPT ELST ENERGY | -0.006601244105 |
| SAPT EXCH ENERGY | 0.017510675158 |
| SAPT IND ENERGY | -0.003207674698 |
| SAPT TOTAL ENERGY | -0.019271610810 |

| System | MonA own nbf | MonB own nbf | Dimer / ghosted-monomer SCF nbf |
|---|---:|---:|---:|
| Protein157 | 1344 | 442 | 1786 |


A matched CPU baseline was submitted separately as **13021636** (eight CPUs, 128 GiB, 8-hour `embers` cap). The saved script requests `cpu-small`; Phoenix's submission policy automatically remapped it to **`cpu-medium`**, with no command-line partition override. **Its result is not yet included here, so no protein157 speedup or CPU-agreement claim is made.** The CPU partition may use different host hardware; any eventual timing ratio must disclose that.

## Failures and completion accounting

- Initial launcher step 13020320.6 failed before calculations because conda overwrote `BUILD`; the launcher now uses `PSI4_BUILD`.
- Step 13020320.7 contains successful water/benzene child processes, each with zero return code and finalized JSON, but the overall step failed on peptide's generated Cartesian auxiliary basis. The 25th successful calculation was an **unpaired Cartesian peptide CPU attempt**, excluded from timing comparisons.
- Step 13020320.8 again failed on peptide: naming a spherical fitting basis alone still inherited the primary Cartesian setting. This unpaired CPU attempt is also excluded. Both errors remain in `raw/`.
- Step **13020320.10** completed `0:0`, including all 12 corrected spherical peptide/nanotube calculations. The benchmark does not quietly change only the GPU basis to obtain agreement.
- Original-grid benzene accuracy misses remain visible above; they are not discarded runs.

## Reproduce the summaries

```bash
python devtools/benchmarks/summarize_saptdft_cuest.py \
  devtools/benchmarks/results/phoenix-a100-20260909/raw/retry1/results --output /tmp/initial-summary
# Expected nonzero exit: initial campaign failed later, and benzene misses the original-grid threshold.
python devtools/benchmarks/summarize_saptdft_cuest.py \
  devtools/benchmarks/results/phoenix-a100-20260909/raw/suite2/results --output /tmp/suite-summary
# Rebuild paired-summary.json directly from both raw campaigns, validate all six groups,
# and regenerate this report (retaining original accuracy failures):
python devtools/benchmarks/render_phoenix_report.py
```

Exact run-time runner snapshots are in `protocols/*.py.txt`; the maintained runner is two directories above this evidence directory. Raw Psi4 outputs are retained under `/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-timing/` on Phoenix and locally in `tmp/benchmark-results/`; `output-sha256.json` records their hashes. All five component energies, geometries, settings, timings, and run/step IDs are committed as raw JSON.
