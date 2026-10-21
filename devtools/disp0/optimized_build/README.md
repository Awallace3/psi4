# Optimized Einsums2 fresh CPU build

## Current authorization: one interactive CPU allocation

After failed job14249189, user approved one corrected build, then explicitly
requested an **interactive CPU node for build/debug instead of sbatching each
attempt**. Current execution is ONE `salloc --no-shell`, 8CPU/48GiB/2h,
freeembers/chemx, no GPU; the corrected batch proposal is superseded, not an
additional job. Queue wait is bounded by `--immediate=600`; no allocation retry.
Resource/account/route/quota/headroom/duplicate checks remain mandatory.
`salloc` has no `--no-requeue` option; effective `Requeue=0`, `Restarts=0`,
CPU-only RUNNING and exact resources are checked before any step is allowed.
No automatic cancellation/release/job modification is performed by acquisition.

Final source namespace is `P/gits/psi4.disp0-opt064-20261021-03/source`.
The staged6d1a0e9f/-02 controller was not used for acquisition: final review
required proof of BOTH requested and allocated TRES. The corrected controller
requires present/parseable CPU8/node1/mem48GiB in both, rejects GPU resources
in both, and normalizes equivalent Slurm memory units. Stage-02 stays immutable;
its fresh hash-bound quota evidence may still support the same filesystems.
The failed20261021 root and every earlier input/source/build are preserved.
Acquisition uses permanent `interactive-attempt`, line-fsynced raw responses,
durable accepted ID before controller query, and a grant manifest. Pending
receipts are never grant proof. One owner; serial exclusive8CPU steps, with
source/pin/job/step-qualified run/scratch and honest signal-aware status.
Within-allocation debug does not authorize another allocation or scientific
source/setting drift. Freeze final source/build provenance before correctness.

The corrected input entrypoint `prepare_inputs.py` starts a **separate bounded
CPU loader-trace process before** lowering the decoder's AS hard limit.
Decoder remains1GiB/110s/oneCPU/128MiBwindow. A 1MiB evidence reserve tightens
decoder writes to511MiB; aggregate bound remains512MiB. Loader worker:
8GiB virtual AS,70sCPU/110swall, six serial canonical major-bound CUDA
component requests,10sCPU/10s wall perldd, no respawn/retry/device constructors.
Actual stdout/stderr is durable before each reply. Mandatory original closure
and complete trace-count checks still precede BOTH final input manifests.
No generic legacy-CLI admission is weakened; its original limits stay intact.
An actual local fork/IPC test checks1GiB decoder/8GiB tracer inheritance without
loading any library. This is not a GPU or realCUDA-positive proof.

## Superseded first build proposal (history)

User instruction on 2026-10-21: **start the build**. This is one new isolated
8-CPU / 48-GiB / 2-hour, free `embers/gts-cs207-chemx` preparation job (16 CPUh
maximum), subject to fresh source, exact CPU route, quota/headroom/duplicates.
No GPU allocation, automatic retry or paid work. This is not the old conditional
89b616/d17 recipe, not replay of the UNKNOWN input submission, and not acceptance
of the rejected b3/8f/9e binary tuple.

## Immutable inputs and intended changes

- Einsums fork `https://github.com/Awallace3/Einsums.git`,
  `06404fa69ce9346443d7d4f19a250f6db9f0f62c`. Never infer its pin from the
  misleading local `perf/fdisp0-hotpaths` alias.
- Psi4 scientific source identical to
  `587e172a33afaadb00eb7f574a5089533f7fc755` under `psi4/`; fresh matching
  whole stage, no older native/Python overlay.
- Read-only historical explicit runtime dependency lock SHA256
  `3f9a508fd601306f548a9088f5cd304367a5fde3988e8bbd4fe00d9e6efb0205`:
  Python3.13/GNU14.3/MKL. It is not represented as a complete new E2 lock.
- Separate fresh **build-only** LLVM/Clang23.1.2 environment, complete solved
  explicit lock retained. Apiary1.2 source fallback and source-pinned Waggle /
  Stripes are selected by the exact optimized E2 source. No login conda solves.
- Native E2 whole Python install rules retained; no d17 manual packaging hook.
  Every built shared/module target gets a deliberate installed RUNPATH.
- Complete genuine CUDA13.3 input inventory and two exact developer archives
  are prepared independently **inside this build job**, using the reviewed
  virtual-link/terminal-hash/SONAME/decoder guards. Their old UNKNOWN input
  attempt stays reserved and untouched. No adoption of any failed prefix.
- Explicit sm90, CUDA enabled/HIP disabled, native FP64, mimalloc/MKL FFT,
  SIMD dispatch on/native-architecture specialization off; profiler policy
  preserved. Batch promotion environment set **before first import**.
- Original nvcc wrapper's implicit headers are not assumed correct. Imported
  CUDA targets are admitted by complete file/hash inventory, rebound to the
  new private locators/include tree, and actual compiler dependencies audited.
  The added CUDA compiler flag is only `-I<private>/include`, not a numerical
  optimization. Raw configuration/build/Ninja/link/header evidence retained.

## Submission transport and evidence

`submit.py` is one self-contained invocation. It permanently creates a fresh
attempt, fsyncs the immutable intent/file+parent, records all preflight stdout
and stderr, then fsyncs `dispatch-intent.json` **before the sole actual sbatch
RPC**. Raw response and accepted real ID are durable **before** controller
queries. Existing attempts always reject. Reply loss/timeout remains UNKNOWN,
never authorizes replay. Simulated test-only IDs are never accepted IDs.

Fresh quota evidence must be human-reviewed, timestamped, hash-bind retained
raw project and scratch reports, and admit >=64GiB/50k inodes for each. Actual
filesystem available bytes/inodes and queue/headroom/CPU route are checked
again within the submission invocation. No open SSH session across model
review; no STDIN second phase.

First (failed14249189) source namespace:
`P/gits/psi4.disp0-opt064-20261021/source`.
Fresh run, env, source, CUDA locator, install, cache and job-owned persistent
scratch paths are job/pin-qualified. Original roots/envs/locks are read only.

## Acceptance, not assumed success

Clean job and batch exit, full native/Python install, source/ABI/ELF/maps,
single exact064 GNU14 world, no E1/old build/site12/empty loader path, actual
header audit and tiny one-thread CPU API/layout/pool/replay checks are required.
Do not mask GPU-enabled import failure on a CPU node. LLVM build-only libraries
unexpectedly required by the installed runtime cause admission failure.

This job does **not** accept signed physical SAPT or GPU correctness. Historical
26/64 d17 comparisons remain historical. Actual optimized-stage signed native/
eager/default-graph references and actual CUDA/cuBLAS correctness must pass in
a separately coordinated GPU window before timings/Phase2.

## Local guards

`python3 -B -m unittest discover -s devtools/disp0/optimized_build -p 'test_*.py' -v`

Initial 13 mock/metadata guards passed in 1.203s; both Bash syntax checks passed.
They cover late/reply-loss no replay, fsync-before-RPC, real-ID-before-query,
wrong route, duplicate, stale quota, wrong E2 world, unsafe loader paths,
resource/pin/packaging policy. Mock IDs are fixtures, not actual job receipts.
No remote operation or build is implied by these local results.
