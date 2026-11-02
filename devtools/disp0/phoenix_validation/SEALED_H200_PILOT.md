# Sealed-runtime H200 peptide pilot

**Latest:** H20014946636 and CPU14991749 completed successfully.
The peptide case is **`PASS_MATCHED_INSTRUMENTED_BENCHMARK`**, with an
accepted **7.004× CPU/GPU energy-call speedup**: CPU median259.483s on24 cores
versus37.048s on one H200 with8 host cores. Three fresh references, three GPU
repeats and three independent fixed-GPU-gauge CPU oracles passed.
Inclusive worker medians are277.044s versus50.784s (**5.455×**).
This qualifies this restricted peptide point, not a kernel-only speedup or
general production release. Earlier failed attempts are preserved below.

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

## Corrected H200 attempt14784867: guard retirement before peptide

Job/batch/step report FAILED1:0 after34s (Python step29s), not preemption.
The prolog confirms embers on`atl1-1-03-020-18-0`, one H200/8 CPUs/160GiB.
The corrected private scratch setup passed, and the fresh sealed CPU-unscaled
water worker returned0. The first GPU worker returned86 before producing a
result; its Psi4 log contains only the memory/thread settings. The exact
retirement reason was not captured by the intentionally silent guard.
No peptide sample, physical primitive-control PASS, full small-gate PASS or
accepted speedup exists from this attempt. OOM is not reported.

Under`phoenix-sealed-h200-6c6cc8885d/job-14784867/`:

| Artifact | SHA256 |
|---|---|
| `pilot-result.json` | `8fa251067a160ed9942feb0ce992b254756cbdc9412b5632bb7f247946b5f2e9` |
| `runtime-config.json` | `0f655f936c1a49b2981db229eb3de176159cd69e6ba5943d2c9d75e953d7d903` |
| `small-gate/gate-result.json` | `b3d9f2f503a4a2838a62a6438e85c04facbd7b7ed3941533f87c2eac49e096ae` |
| `small-gate/cpu-unscaled/result.json` | `055180aa66852046de1cfec6dae36afd63005bedc6d1edcdfae377d9221ca795` |

The pilot records zero accepted speedups and successful removal of its owned
top-level scratch directory after child-group reaping. This is not a claim of
graceful native GPU teardown inside the fail-stopped worker.

### Reproduced cold-binding mismatch

Read-only inspection of the frozen configuration shows both cuBLAS/cuSOLVER
manifest locators are private SDK symlinks, while their already-preloaded
sealed snapshot keys are canonical files under
`miniconda/envs/p4cuest_sapt/targets/x86_64-linux/lib/`.
For example the manifest's`gpu-runtime/cuda13/lib64/libcublas.so.13.6.0.2`
is not that canonical source string.

The shared CUDA `library()` adapter previously forwarded the unresolved
locator to the sealed owner, which intentionally accepts exact admitted
source keys. A tiny real sealed-ELF/SDK-locator fixture reproduced exit86
instead of loading the retained image. This is consistent with the H200
failure boundary, but the old worker's missing reason is not retroactively
invented.

The candidate resolves filesystem locators at the **cold library-binding
boundary**, brackets that resolution with owner validity checks, and still
loads only the admitted retained image. The sealed owner's direct snapshot
API and hot execution checks perform no new filesystem resolution. SHA,
mapped inode/device, symbol, root and native/device/FP64 checks are not removed.
Foreign targets still retire; a locator retargeted after resolution cannot
change the loaded bytes. No mapping/loader-state cache is introduced.

### Diagnostic and CPU-first gate

The missing-reason fixture was run red before adding an optional private Unix
datagram receipt. Retirement sends at most512 bytes with per-call
`MSG_DONTWAIT|MSG_NOSIGNAL` and Python socket timeout0, then exits86 regardless
of send success. No stderr/file/pipe write is added to invalidation. A full
blocking-mode socket, closed receiver and default SIGPIPE cannot delay/change
retirement; dropped diagnostics never authorize continued execution.
Review found a diagnostic-setup rollback leak if socket validation/setup
raises. Ownership is now registered immediately after construction, before
either operation; injected interruptions verify closure of every duplicate.

Fresh CPU preparation now bootstraps/adopts the actual sealed GPU adapter,
binds cuBLAS/cuSOLVER through the same shared loader, and verifies native
symbol ownership **without creating a CUDASession**. It hides/allocates no
GPUs and makes no correctness/timing claim. A finalized exact-pin PASS with
reaped children is required before H200 execution. This narrows the feedback
loop before spending another H200 allocation; later GPU-specific failures
remain possible and still require the full physical gate.

Both failed attempts, successful CPU builds, frozen sources and configurations
remain unchanged. No further CPU/H200 job was automatically submitted.

Candidate host validation passed**383+121+33=537 tests**, without skips,
plus shell/whitespace checks. The real-ELF locator regression and missing-reason
regression were each run red before their fixes. Additional checks cover
foreign/retargeted locators, full/closed diagnostic receivers, forced blocking
descriptor flags, interrupted socket setup/rollback, actual descriptor passage
through the PsiAPI/Bash/exec chain, and absence of CUDA-session creation in the
CPU admission mode. These are host evidence, not GPU numerical qualification.

## User-approved CPU-only validation14942939

The user explicitly selected **CPU-only validation**, not another H200
submission. Frozen source/helper:
`927283f8ed0a94d42179c35dd2a1a5bce674347e`, checkout
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-sealed-admission-927283f8ed`.

Job**14942939** uses`gts-cs207-chemx / cpu-small / embers`,8 CPUs,48GiB,
3-hour cap and **no GPU request**. It makes a fresh coherent whole stage,
runs CPU regressions and native water controls, then exercises the new
CPU-only sealed adapter and cuBLAS/cuSOLVER binding/symbol checks.
Scheduler test-only ID14942938 is not a compute job.

Output root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-sealed-cpu-admission-927283f8ed`.
Important final artifacts are`result.json`,
`cpu-sealed-admission/preflight-result.json` and
`cpu-sealed-admission/runtime-admission.json` inside the job directory.
Watcher:`bbf417ea0`.
Initial controller readback and prolog confirm RUNNING on embers with the
declared account/CPU/memory/wall cap; neither requested nor allocated TRES
contains a GPU.
Submission is not admission PASS, GPU qualification or a speedup result.

### CPU validation14942939 passed

Job/batch/step completed0:0 in10m56s; payload exit0/no TERM.
The whole-build proof is PASS at exact pin`927283f8ed`;383 CPU regressions
passed without skips and121 helper tests completed with the same single
optional literal-input-suite skip. Native CPU RHF remains
`-1.1167143250625702 Eh`.

The new real CPU runtime admission returned0 with no retirement reason and
all owned child groups reaped. Both private manifest symlink locators bound
the preloaded canonical snapshots:

- cuBLAS`libcublas.so.13.6.0.2`, SHA256
  `c13e3af56f07357b3419f01f183013577274709637b8a35aec008c3fe6ab6102`,
  with`cublasCreate_v2`/`cublasDestroy_v2` owner checks.
- cuSOLVER`libcusolver.so.12.2.6.9`, SHA256
  `b0825c65a735cc2eba019194103043b7db05e4e54ef79b788071c20b5582864f`,
  with`cusolverDnCreate`/`cusolverDnDestroy` owner checks.

The proof explicitly records`cuda_session_created=false` and
`gpu_correctness_verified=false`. This validates the binding fix on the real
Phoenix dependencies, not a CUDA calculation or accepted performance sample.
The actual Phoenix closure contains10 sealed files/20 source-plus-destination
hashes, all with seal mask63, retaining1,353,495,253 snapshot bytes. It is not
silently described as Cachy's different17-image dependency copy.

Under`phoenix-sealed-cpu-admission-927283f8ed/job-14942939/`:

| Artifact | SHA256 |
|---|---|
| `result.json` | `a8915b40305761542426494fbe2a373727fc3de0a38b6d90a3f82eb3a06a6c42` |
| `gpu-runtime/preparation-result.json` | `eb4eff4c84d6eedb2eba0cf0002686b9826d03bd3c7113acdb7f313d530c788b` |
| `gpu-runtime/runtime-config.json` | `9539fdedd71d30fd719b5e35e50f8754aeca9fb9df9bad64389efbf4024c8b7f` |
| `cpu-sealed-admission/preflight-result.json` | `c1f1d57993c807f82a3a93ff63447fd24d7da89cb179ad1dd0f7be350f4c42fe` |
| `cpu-sealed-admission/runtime-admission.json` | `2c7caab8deb0a90442992fe622636f1cf4c66194eb275b5cddfa6f1338dd485a` |

The qualified whole prefix and remote source stay frozen. A new H200 attempt
still requires its physical small gate before three peptide repeats; none
has been submitted under this CPU-only authorization.

### Reusing the completed prerequisite

The user subsequently approved the H200 pilot using this validated build,
without another build. The submission-only helper accepts finalized exit0/
no-TERM CPU artifacts, checks exact source/build-proof hashes and successful
sealed admission, and omits a scheduler dependency on an already-completed
job ID (which SLURM may have purged). Pending prerequisites still use
`afterany` with the original runtime artifact checks. Failed, interrupted,
changed or wrong-pin completed proofs reject before submission.

Submission recipe and execution helper pins are recorded separately.
The actual job, bootstrap, installed science and complete execution-helper
checkout remain frozen at`927283f8ed`; no worker code is copied or overlaid.
The actual H200 worker still reauthenticates the whole prefix and CPU admission
before the physical gate. Removing an obsolete scheduling dependency does
not waive any scientific/runtime gate or authorize automatic retries.

## H20014946636 submitted using the validated build

The user explicitly approved **Submit H200 on embers** after the real CPU
admission PASS. One job was accepted: **14946636**,
`gts-cs207-chemx / gpu-h200 / embers`, one H200,8 host CPUs,160GiB,8-hour cap.
It must pass the full physical small gate before three fresh peptide calls.
No rebuild, additional CPU allocation, large CPU oracle on the GPU node or
automatic retry was submitted.

Execution source/helper and whole stage remain exactly
`927283f8ed0a94d42179c35dd2a1a5bce674347e`, using completed CPU job14942939.
Submission-only recipe:
`04db1a673925b1709cd7bf97bca0bd3708385e36`, from
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-sealed-submit-04db1a6739`.
The fresh manifest records both pins, recipe SHA and the completed CPU proof/
admission hashes. Its scheduler parent is null because the finalized CPU
artifacts are already valid; the runtime checks remain mandatory.
Test-only scheduler ID14946635 is not a compute job.

Output root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-sealed-h200-927283f8ed`.
Watcher:`b95fa6655`.
Controller readback confirms PENDING(Priority), no dependency, embers,
one requested H200/8 CPUs/160GiB/8h and`Features=H200`. The manifest's
completed-proof hashes match the CPU artifacts above. The submission-only
helper suite passed124 tests; no scientific or execution-helper code changed.
Submission is not a small-gate PASS or an accepted speedup.

## H20014946636 completed: small gate PASS, peptide median37.048s

Job/batch completed0:0 in8m05s; the Python step completed0:0 in6m59s.
Payload exit0/no TERM. Pilot status:
`PASS_GPU_PILOT_AWAITING_INDEPENDENT_CPU_QUALIFICATION`.
The physical ten-process gate records`PASS_SMALL_H200_ONLY` with actual
small-case correctness verified. Three fresh peptide GPU calls then passed
the existing23-field repeat checks at unchanged1e-9 tolerances.

| Peptide repeat | Instrumented energy-call wall (s) | Inclusive worker wall (s) |
|---|---:|---:|
| 0 | 37.226450 | 50.784330 |
| 1 | 36.789883 | 50.071116 |
| 2 | 37.047639 | 50.843936 |
| **Median** | **37.047639** | **50.784330** |

These are instrumented whole-energy calls, not isolated kernels or
uninstrumented production. Normal scientific setup/SCF, F-SAPT output and
native continuation/cleanup are inside the energy timer; snapshot/bootstrap,
primitive controls and diagnostic output also appear in the separate worker
wall. Do not replace the energy-call timer with a subphase sum.

All three result/NPZ/output-file hashes were checked, with distinct PIDs
1986677/1986889/1987130. Each performed real native/device work and complete
native/session cleanup, zero live tracked bytes/owners afterward, and reaped
its owned process group. The benchmark removed its owned scratch root; the
outer pilot confirms it is already absent.

Actual hardware: NVIDIA H200,143771MiB, driver615.71.09,`compute_90`,
UUID`GPU-90510df2-57d7-8d2a-2e07-fc0c5dc97cf5`. PTX SHA256:
`0f3341120728784af9804df3ee44137dd9f31989b40b8a98ec07c2075c83c118`.
The case is24 atoms/356 AOs/20 occupied per fragment,5 frozen and15 active.

### Historical GPU latency comparison, separate from CPU/GPU qualification

The previous H200 peptide GPU diagnostic14481883 had median994.512498s
(993.473989–994.536646s). The new recorded median is**26.844 times lower**.
Geometry, numerical options, dimensions, block size and cache ceiling match;
both use the same driver, PTX and CUDA kernel-source SHA. Work/memory ledgers
also match:

- tracked explicit device peak:**4,477,018,672 bytes** in every new repeat,
  identical to the historical sample; opaque native internals are not this
  tracked peak;
- 64MiB cache ceiling,60,076,800 bytes allocated,5 cached slabs/65 hits;
- 145 RIFIT producer calls and196 dispersion compute blocks;
- unchanged12GiB declared opaque allowance.

This is a historical host-overhead diagnostic comparison across different
H200 UUIDs/build pins, **not** a cache/host-matched controlled GPU-kernel
speedup. No new memory allowance, cache increase or reduced work count buys
the observed latency reduction.

At GPU completion the peptide status was
`PASS_REPEATS_AWAITING_OTHER_RESOURCE_CLASS`;`accepted_speedup_count=0`
remains in both immutable producer/pilot receipts. The final CPU consumer
records qualification separately below. The historical295.104s CPU median
is not reused as the new denominator. No nanotube/protein83 speedup or broad
large-system admission follows from this pilot.

Under`phoenix-sealed-h200-927283f8ed/job-14946636/`:

| Artifact | SHA256 |
|---|---|
| `pilot-result.json` | `31e4ea89d7215c63b93fa51951096133e41970f79e1c836615886f5ce3916c1c` |
| `runtime-config.json` | `6dc206026d947623862659206bce8b0a1fe6f7ca070814073c06c6283c0e25ad` |
| `small-gate/gate-result.json` | `536e5c062a4bf33137e164844802b6b9067fd0164cd8dd384dbfbb4ef42913de` |
| `benchmark/execution/benchmark-result.json` | `76fd14b90fab00d986a9f2a16bb2f285417708f54bcfdb5e76f5c4822a00f0e7` |
| `benchmark/execution/peptide-r0/result.json` | `4299c4963884cd7975974910ab9f77f21786ad11cb891c112f75ea4baabce76c` |
| `benchmark/execution/peptide-r1/result.json` | `a7a20c033f8124880f591f6d687dc355e46f17981d4aea6f06d32321b26c0a97` |
| `benchmark/execution/peptide-r2/result.json` | `dccc8b788ca62faa0a100992269c83f01eba4fc0aa1a747ef97b309371b2ef07` |

## User-approved final CPU qualification

After the H200 result, the user selected **Qualify CPU/GPU ratio**:
one CPU-only embers allocation,24 cores,160GiB,2-hour cap, three fresh
peptide references and three independent fixed-GPU-gauge CPU oracles.
No new GPU allocation, rebuild or automatic retry is authorized.

`submit_sealed_cpu_qualification.py` is a submission-only recipe. It requires
the finalized exit0/no-TERM H200 pilot, successful physical gate, reaped/cleaned
producer, three distinct sealed samples and a timing summary equal to their
actual energy timers. It records the final GPU document hashes for rechecking
after CPU completion; the CPU worker also verifies raw/matrix artifact hashes
before each oracle. All scientific workers, benchmark/comparison helpers and
the whole stage execute unchanged from`927283f8ed`.

CPU configuration is an exact copy of the GPU configuration except threads24
versus8. The existing CPU benchmark driver enforces node-local scratch,
unchanged methods/options, three same-arm repeats, retained spaces and all
independent signed component/matrix checks before accepting a ratio.
Per-CPU-worker diagnostic aliases are fresh children of the already-declared
private diagnostic namespace; existing GPU aliases/data are not repointed.
The recipe requests no GPU TRES or scheduler dependency. It passed the
130-test helper suite, including frozen-execution-source, role/resource,
changed-proof, duplicate-PID and fabricated-timer rejection checks.

### CPU qualification14991749 submitted

One job was accepted: **14991749**,
`gts-cs207-chemx / cpu-small / embers`,24 CPUs/160GiB/2h,
exclusive`core24&gold6226&localNVMe`, no dependency or requested GPU.
Controller readback and prolog confirm RUNNING on embers with those resources;
allocated TRES contains no GPU. Test-only ID14991748 is not a compute job.

Execution source/helper/whole stage remain
`927283f8ed0a94d42179c35dd2a1a5bce674347e`; submission-only recipe is
`832ce79b3c2b316ea4d32599f3326daa023c7c47`. The recipe's checkout is
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-sealed-cpu-qual-submit-832ce79b3c`.
The completed GPU producer and its output files are unchanged.

Output root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-sealed-cpu-qualification-927283f8ed`.
The final benchmark/qualification receipt will be
`job-14991749/benchmark-result.json`, with per-process reference/oracle outputs.
Watcher:`ba307febf`.
No qualified CPU/GPU ratio is inferred until those independent checks and
cleanup pass and the recorded final GPU hashes are rechecked.

## CPU14991749 completed: accepted7.004× peptide speedup

Job/batch/step completed0:0 in28m25s; payload exit0/no TERM.
The parent receipt records`accepted_speedup_count=1` and peptide status
**`PASS_MATCHED_INSTRUMENTED_BENCHMARK`**, with
`cpu_over_gpu_ratio=7.004036032786262`. It used the same frozen scientific/
execution pin`927283f8ed`; CPU/GPU runtime configurations differ only in
thread count24 versus8.

| Repeat | CPU energy-call wall (s) | H200 energy-call wall (s) |
|---|---:|---:|
| 0 | 263.152222 | 37.226450 |
| 1 | 259.482995 | 36.789883 |
| 2 | 258.832354 | 37.047639 |
| **Median** | **259.482995** | **37.047639** |

This is24 CPU cores versus one H200 plus8 host cores, not a matched-host-thread
or isolated-kernel comparison. The instrumented energy timer includes normal
scientific setup/SCF, F-SAPT output and native cleanup. Inclusive worker wall,
also including snapshot/bootstrap, primitive controls and diagnostic overhead,
has CPU/GPU medians277.044203/50.784330s: a separate**5.455309× latency ratio**.
Neither timer is uninstrumented production throughput.

All three fresh CPU references passed repeat checks. Three additional
independent CPU processes recomputed the fixed-GPU-gauge oracle for the three
GPU samples; all23 fields passed at unchanged tolerances. Maximum absolute
energy-matrix entry errors were6.47438e-12,7.27670e-11 and3.47301e-10Eh.
Aggregate GPU-minus-CPU energy differences were respectively
-1.68550e-10,-1.81466e-10 and-2.38138e-10Eh.
All six CPU results, NPZs and standard F-SAPT output hashes were verified;
process identities were distinct, CPU-only execution was checked, all owned
groups were reaped, and owned scratch cleanup completed.

Final GPU pilot/config/gate/manifest/benchmark hashes were rechecked against
the CPU submission manifest and are unchanged. The producer's awaiting-CPU
status is deliberately not rewritten; the accepted result lives in the
separate CPU consumer receipt.

Under`phoenix-sealed-cpu-qualification-927283f8ed/`:

| Artifact | SHA256 |
|---|---|
| `manifest.json` | `58ff123109c1f1188bf57b8bf0ac93b834ceb2faba658fbf38060b70232bcd2a` |
| `runtime-config.json` | `e187f2ac0a826a136787f9ba143682523ecd19922ac24ffc818e6cdf7877e859` |
| `job-14991749/benchmark-result.json` | `6ee50f8a3c51f74dca67893b2d15282278150050604f8031d480ddbd1b458d95` |

Scope:24-atom/356-AO spherical aug-cc-pVDZ peptide, exact per-source HF CPHF
F-SAPT with5 frozen/15 active occupied orbitals per fragment and unchanged
64MiB factor-cache ceiling. This is not the cheaper scalar-scaled uncoupled
CPU partition. No nanotube/protein83 scaling curve, unrestricted/Cartesian
qualification, or broad provider/production release is established.
No further jobs or automatic retries were submitted.
