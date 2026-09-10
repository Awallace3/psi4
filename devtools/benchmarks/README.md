# cuEST SAPT(DFT)-D4(I) GRAC benchmarks

Run `saptdft_cuest_grac.py` with a cuEST-enabled Psi4 development build and
DFT-D4 installed, inside a GPU allocation. Initialize PsiAPI with that build's
staged executable, not a release Psi4 from another environment.

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
# Set TMPDIR, SCRATCH, and PSI_SCRATCH to a private, writable scratch directory.
eval "$(/absolute/build/stage/bin/psi4 --psiapi)"
python saptdft_cuest_grac.py --output /absolute/new-run/results --repeats 3 --threads 8
python summarize_saptdft_cuest.py /absolute/new-run/results --output /absolute/new-run/report
```

The campaign output directory must not exist. Each calculation runs in a fresh
Python process and its own working directory, preserving a separate `timer.dat`,
Psi4 output, console log, and atomic JSON result. The campaign stops on execution
failure rather than repeatedly spending resources on the same failure. Completed
measurements survive interruption. `COMPLETE.json` indicates execution completion;
**only the summarizer checks accuracy**. A nonzero summarizer exit means incomplete
results, failures, or an accuracy deviation exceeding the declared tolerance.

## Matched calculation settings

- SAPT(DFT)-D4(I), PBE0, DF SCF, internal orbital optimizer.
- `SAPT_DFT_INDUCTION_TYPE=NONE`, `SAPT_DFT_DO_DHF=True`: induction is assigned
  from delta-HF. This is not a response-based CPKS induction benchmark.
- `SAPT_DFT_GRAC_COMPUTE=ITERATIVE` is the default. Neutral and cation SCFs for
  both monomers determine the GRAC shifts inside each timed `energy()` call.
  `--grac-compute NONE --shift 0.136` is retained only for explicit fixed-shift
  diagnostics and must be labeled as excluding automatic GRAC work.
- 99 radial / 590 spherical DFT grid settings; SCF energy/density convergence
  1e-9 / 1e-8; 24 GiB Psi4 memory; equal thread counts.
- Double precision (`CUEST_MIXED_PRECISION=False`) in the GPU arm.
- CPU and GPU differ only in `USE_CUEST`. `CUEST_XC=True` is effective only in
  the GPU arm. The GRAC functional is evaluated through Psi4 on cuEST grid
  densities, with AO integration handled by cuEST; this is not a claim that
  every operation runs on the device.
- `SAPT_DFT_USE_EINSUMS=True` requests Einsums; record the installed dependency
  and inspect output because the driver can fall back when it is unavailable.
  Default cuBLAS multiplication thresholds are retained.

Wall time covers the complete `psi4.energy()` call, including backend
initialization and internal setup. It excludes Python imports and the preliminary
molecule/basis construction used to check nbf. There is no discarded warm-up.
CPU/GPU order alternates between repeats. Report medians **and ranges**; speedup
is median CPU wall time divided by median GPU wall time. A value below one is a
GPU slowdown and must not be omitted.

Accuracy is backend agreement, not accuracy against experiment or a higher-level
method. Compare electrostatics, exchange, induction, dispersion, and total energy
for each matched repeat; the threshold is 1e-6 Eh per component. Use the focused
GRAC regression tests separately to establish a nonzero correction effect and
agreement of density/orbital energies, rather than inferring those from total
interaction-energy agreement alone.

## Systems

- Water dimer: the geometry in `tests/pytests/test_saptdft_cuest.py`.
- Benzene dimer: idealized parallel-displaced rings (3.4 Å separation, 1.6 Å
  lateral offset); **not the published S22 geometry**.
- Peptide (timing-set 144) and ethene/nanotube (154): frozen two-fragment inputs
  copied from `~/data/timing_test_suite/geometries/systems.py` into
  `saptdft_suite_geometries.json`. The spherical 6-31+G** basis counts are checked
  (250 and 548), not the original Cartesian counts (260 and 574). They are timing
  geometries, not reoptimized structures.
- Protein157 is excluded from the paired interactive campaign. Run it as an
  explicit case in a separate allocation. A CPU baseline additionally requires
  `--allow-protein157-cpu` to avoid accidentally starting a large CPU job.
  Spherical 6-31+G** has 1786 functions (the historical Cartesian basis has 1863).
  A GPU-only result has no speedup or backend accuracy comparison until a matched
  CPU measurement completes.

Water and benzene use cc-pVDZ and aug-cc-pVDZ. Peptide and nanotube use **spherical
6-31+G\*\*** (`PUREAM=True`) with explicit **def2-universal-jkfit** SCF and
**aug-cc-pVDZ-RI** correlation fitting bases in both arms. Their default generated
Cartesian auxiliary basis is unsupported by cuEST; even a named fitting basis
inherits the orbital basis's Cartesian setting on this path. Changing only the
GPU basis would invalidate the comparison, so rerun both arms with these settings.

Use `--systems peptide nanotube` to run just the timing-suite cases. Diagnostic
controls can use `--case --cpu-xc` (GPU J/K with CPU XC), `--shift 0.0`, or
`--radial-points 250 --spherical-points 974`. Keep diagnostic results separate
from the fixed-protocol timing medians.

For a separately allocated protein157 run, request enough host/device memory and
set Psi4's memory explicitly, for example:

```bash
python saptdft_cuest_grac.py --case --system protein157 --basis '6-31+g**' \
  --mode gpu --threads 8 --memory '112 GiB' --output /absolute/new-run/protein157-gpu-1
```

The Phoenix batch configuration requests one A100 80 GB, 128 GiB host memory,
and a 4-hour `embers` cap, runs a water GPU/GRAC preflight, and then measures one
protein157 calculation. That resource request is not a guarantee that cuEST's
queried device workspace will fit. A separately authorized CPU baseline uses
`--mode cpu --allow-protein157-cpu` with the same inputs, eight threads, and
112 GiB Psi4 memory, with a CPU-partition allocation of 128 GiB and an 8-hour
`embers` cap. Record the CPU model: a ratio across CPU and GPU nodes is not an
isolated same-host accelerator speedup.

Do not compare these SAPT timings to the timing suite's historical RHF timings
as if they measured the same method or hardware.

## Provenance to retain beside a run

Record the source commit and any patch, staged binary path and hash, build flags,
Psi4/cuEST/DFT-D4/Einsums/BLAS versions, GPU/driver and CPU model, process affinity,
thread environment, scheduler job/step IDs and terminal exit status. Retain the
launcher, geometry file/hash, script/hash, raw results and logs. A still-running
parent interactive allocation is expected: verify the benchmark **step** finished
successfully, not that the parent allocation has ended.

The reporting unit tests do not require Psi4:

```bash
python -m unittest discover -s devtools/benchmarks -p 'test_*.py' -v
```

GRAC analytic gradients and cuEST XC response remain outside the supported scope.
