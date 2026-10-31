# Sealed-runtime H200 peptide pilot

User requested an **embers H200 speedup test** after the restricted-water
Cachy gate at `0433ec0452`. This pilot keeps that scientific/runtime source
unchanged and ports its early bootstrap into the Phoenix launchers.

## Allocation and acceptance

1. Fresh whole sm_90 build and CPU qualification: 8 CPUs, 48 GiB, 3 hours,
   `gts-cs207-chemx / cpu-small / embers`, no GPU.
2. One H200, 8 host CPUs, 160 GiB, 8 hours, explicit `--constraint=H200`,
   `gts-cs207-chemx / gpu-h200 / embers`.
   Uses `afterany` plus artifact checks, not `afterok`. A failed or interrupted
   CPU build stops the GPU job before scientific work. No automatic retry.
3. The full small ten-process CPU/GPU/fixed-gauge-oracle/displacement/repeat
   gate runs first on H200 at unchanged tolerances. Only on PASS do three
   fresh peptide GPU benchmark calls run. Nanotube/protein83 are not submitted.

The whole prefix includes authenticated `psi4_runtime_support`. Sealed workers
prepare executable snapshots before normal Psi4/core/Einsums imports, verify
the native capability owners before/after imports, use private cache homes,
and retain fresh mapping/symbol checks. The readonly environment is the
existing Phoenix CUDA13.3/GNU14/Einsums064 dependency root, never Cachy's build
or shared environment. The CPU proof seals source/staged support inventories
and launch-helper provenance in addition to the scientific/core inventories.
Kernel lease/ELF test temporaries use private executable node-local `/var/tmp`
directories, not the Lustre scientific scratch directory. Offline validation
passed115 Phoenix helper tests and33 Cachy compatibility/bootstrap tests.

RHEL's kernel version alone is not a capability test. A tiny login-node
preflight on `5.14.0-687.49.1.el9_8.x86_64` accepted executable memfd flags
`0x13` and returned seal mask63. This is evidence of backports, **not compute
node or GPU qualification**. Actual sealed loading and numerical gates must
still pass in the allocation; there is no older-kernel/fresh-byte fallback.

The producer retains the existing exact per-source HF CPHF, aug-cc-pVDZ,
scaling, 64-MiB factor-cache ceiling, memory guards, normal F-SAPT output,
repeat checks and native/scratch cleanup. Both whole energy-call and inclusive
worker wall times are recorded. No large CPU reference/oracle runs while
holding the H200.

This is a bounded first performance point, **not an accepted CPU/GPU speedup
by itself**. Three qualified CPU references plus independent fixed-GPU-gauge
CPU oracles are still required before publishing a CPU/GPU ratio. Historical
peptide timings (GPU994.5s, CPU295.1s) remain diagnostic context, not an
automatically accepted denominator for a different whole-stage pin.

## Entry points

- CPU build: `submit.py --qos embers --prepare-gpu-runtime` with explicit source,
  exact pin and fresh run root.
- H200 pilot: `sealed_h200.py` with explicit source/pin, CPU build job/root,
  frozen input manifest/SHA and fresh root.
- Early scientific worker: `sealed_bootstrap.py`.
- Output: `pilot-result.json`, `small-gate/gate-result.json`,
  `benchmark/execution/benchmark-result.json`, per-process arrays/results,
  scheduler/allocation/submission receipts and TERM-aware `job-status.env`.

Submission IDs and absolute output roots are recorded after scheduler acceptance.
