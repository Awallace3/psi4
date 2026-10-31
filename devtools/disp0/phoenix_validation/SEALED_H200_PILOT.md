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

## H200 attempt14779252 failed before scientific work

Job/batch/step report FAILED1:0 after9s (the Python step ran2s).
The prolog confirms embers on`atl1-1-03-019-14-0`, with one H200 requested,
8 CPUs and160GiB. The exception is from the first scratch-ownership check:
`Scratch must be this payload's own fresh external directory`.
It occurs before configuration preparation, CUDA admission, the small gate,
or any peptide sample. There is no accepted timing, numerical result or
GPU device-identity evidence from this attempt; no OOM is reported.

The batch wrapper created/bound scratch before`srun`, but its step entered
with an inconsistent scratch environment. The inherited values were not
captured in the failed process. Moving creation and binding into the Python
step removes dependence on`srun` preserving batch TMPDIR: the step creates
its own job-named0700 directory, validates ownership, then sets all four
scratch variables together. Existing directories reject; scheduler-provided
temporary directories are neither adopted nor deleted. The owned-directory
guard itself is unchanged.

The regression starts with conflicting scheduler/batch scratch variables,
checks all four variables bind the newly created directory, rejects reuse,
and verifies cleanup leaves scheduler scratch untouched. Failure before
scientific preflight still records FAILED and cleans the new owned directory.
The corrected offline suites passed116 Phoenix helper tests and33 Cachy
compatibility/bootstrap tests; shell syntax and whitespace checks passed.

Original logs/status, successful CPU build and frozen checkout are preserved.
The failed batch-created scratch directory was not removed by a successful
ownership proof; complete cleanup of that original attempt is not claimed.
No automatic replacement job was submitted. A corrected attempt requires a
fresh coherent source/build proof and new output roots, not editing the frozen
whole stage or reviving its helper proof.

## Explicitly approved corrected submission

After reviewing the scratch-launch failure and149 passing offline tests, the
user explicitly selected **Resubmit on embers**. New frozen source/helper:
`6c6cc8885d04bf943c807805387565559efe9ec1`, checkout
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-sealed-h200-6c6cc8885d`.
The prior source, whole build, manifests and failed job remain untouched.

| Job | Role | Route | Cap |
|---|---|---|---|
| **14784757** | New whole build/CPU prerequisite | gts-cs207-chemx / cpu-small / embers | 3h |
| **14784867** | Corrected H200 pilot | gts-cs207-chemx / gpu-h200 / embers | 8h |

H200 resources remain1 GPU/8 host CPUs/160GiB, with the full small gate before
three fresh peptide calls. The CPU prerequisite remains8 CPUs/48GiB/no GPU.
Dependency:`afterany:14784757`, plus successful exact-pin CPU artifact checks.
Test-only scheduler IDs14784756/14784865 are not compute jobs.

Fresh roots under
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/`:

- `phoenix-sealed-build-6c6cc8885d`
- `phoenix-sealed-h200-6c6cc8885d`

State-change watchers:`bc14f6c6e` (both jobs),`bd0ce55a9` (CPU prerequisite).
Controller readback confirms both effective QoS values are embers with the
declared account/resources. CPU14784757 is RUNNING (prolog: embers);
H20014784867 is PENDING(Dependency), with H200 constraint and8-hour cap.
No inferno, automatic retry, old-proof reactivation or scientific-code overlay.
This remains a submitted pilot, not a completed gate or accepted speedup.

### Corrected CPU prerequisite passed

Job14784757 completed in10m09s with job/batch/step exit0:0 and no TERM.
Whole-build proof is PASS at exact pin`6c6cc8885d`. All376 CPU regressions
passed without skips;116 helper tests completed with the same single optional
literal-input-suite skip documented above. Native CPU RHF remains
`-1.1167143250625702 Eh`. Fresh unscaled/scaled water CPU controls completed
at`-0.007157905730603095` and`-0.008584114664472444 Eh`.

Under`phoenix-sealed-build-6c6cc8885d/job-14784757/`:

| Artifact | SHA256 |
|---|---|
| `result.json` | `cc65a8e184a2d9e0361662256391ac53f70ac8157726f1359a6d71fb60ba961d` |
| `gpu-runtime/preparation-result.json` | `cac5129e79f00bec80e615df7f3e4513bb1255b4dc22b0ee449173dbebc1bdc9` |
| `gpu-runtime/runtime-config.json` | `ff6ef5c63b60f224f5446c843a6eb6ec80d087239784065188f6f3e1d956d778` |
| `cpu-fragment-unscaled/result.json` | `48cf5beb69c108cc73c49d8e0044713a88f146438c922948140c1019d46f2c47` |
| `cpu-fragment-scaled/result.json` | `d996aeb29b14b25471e857aad5c5fcfd767fa949696eb2d37e1718e8288e54d3` |

H20014784867 is now PENDING(Priority), no longer waiting on the CPU build.
The frozen remote checkout and queued job are unchanged. No H200 computation,
small-gate PASS or accepted timing/speedup is inferred from CPU completion.
