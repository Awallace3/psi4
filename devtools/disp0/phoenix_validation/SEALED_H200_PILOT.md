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

## Submitted 2026-10-31

Frozen science/helper checkout:
`c850f57a32085b46d961402c3cffb9a38d75b212`,
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-sealed-h200-c850f57a32`.
The `psi4/` and `psi4_runtime_support/` Git trees are unchanged from the
Cachy-gated `0433ec0452`; only experiment helpers/proofs/docs differ. The new
whole build still binds generated metadata and binaries to its actual full pin.
This checkout must not be advanced while either submitted job uses it.

| Job | Role | Route | Cap |
|---|---|---|---|
| **14779212** | Whole build/CPU prerequisite | gts-cs207-chemx / cpu-small / embers | 3h |
| **14779252** | Small H200 gate then three peptide GPU repeats | gts-cs207-chemx / gpu-h200 / embers | 8h |

The GPU job depends `afterany:14779212` and checks successful CPU artifacts
itself; a failed build does not become a scientific fallback. Scheduler
test-only IDs14779211/14779251 are not compute jobs.

Output roots under
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/`:

- `phoenix-sealed-build-c850f57a32` (build and CPU evidence).
- `phoenix-sealed-h200-c850f57a32` (H200 pilot).

Each retains immutable manifests, test-only responses, submission attempts,
responses and job IDs. Frozen input manifest SHA256:
`f8c96058e9387ab14179ce263a2d4896dbe9cc2e3bf1286654784825ff178e38`.
State-change watcher:`bd1a9c8e7`.
Initial controller readback confirms both effective QoS values are **embers**,
with the requested account/resources. CPU14779212 is RUNNING; its prolog also
reports embers. H20014779252 is PENDING(Dependency), one requested H200,
8 CPUs/160GiB/8h and `Features=H200`. Actual H200 hardware and its prolog
remain to be verified after allocation.
Submission is not execution, numerical qualification or an accepted speedup.

## CPU prerequisite completed

Job14779212 completed in9m48s with job/batch/step exit0:0; the payload records
exit0 and no TERM. Whole-build proof is PASS at exact pin`c850f57a32`.
All376 CPU regressions passed without skips. The115 helper tests completed
with one explicit skip (`test_frozen_full_inputs_bind_actual_independent_dimensions`:
the optional literal input suite is not installed in this checkout). The
submitted frozen peptide input manifest is separately hash/dimension-checked.
Native CPU RHF is`-1.1167143250625702 Eh`; fresh
unscaled/scaled water controls completed at`-0.00715790573059643` and
`-0.008584114664799455 Eh`. These controls are not GPU qualification or timings.

Under`phoenix-sealed-build-c850f57a32/job-14779212/`:

| Artifact | SHA256 |
|---|---|
| `result.json` | `c7be92ad57ec0f434ddef30e37827f9c4de5cf718843ed14ea7eabb8fdfb3846` |
| `gpu-runtime/preparation-result.json` | `218a1de6895ab30cc61dd61f60e434a15d84565bf70b4be4669ee872e2b33125` |
| `gpu-runtime/runtime-config.json` | `26f9ea00c5be93f7ba87e14cc93fa74b9a02a502b046544838411a245fcbbc9f` |
| `cpu-fragment-unscaled/result.json` | `6d459b293e7d73beb430320eaec80caefc8fe6fc8ed8f922cd8d3b58357365c7` |
| `cpu-fragment-scaled/result.json` | `3f0169f88964dc8ba70699c992a1dd05d07eb87456c233f602274c9c4a8357c8` |

H20014779252 is now PENDING(Priority), no longer waiting on the CPU dependency.
No H200 computation, small-gate PASS or accepted speedup is inferred yet.
The frozen remote checkout/configuration and queued job remain unchanged.
