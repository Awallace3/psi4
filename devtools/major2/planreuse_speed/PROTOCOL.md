# Actual-wrapper fixed-work GPU plan-reuse speed protocol

Preparation is not submission. Only the parent may execute `parent-finalize.sh`.
No CPU benchmark, SCF/energy call, RHF finalization, CPHF/CPKS, backend fallback,
Psi4/SDK build, compiler invocation, historical timing adoption, or auto-followup.
This measures repeated initialization of one live JK object, not whole SAPT.

## Feasibility and inherited gates

Both exact pinned sources export `.def("D", &JK::D, ...)` in
`psi4/src/export_fock.cc`; native `compute_D` reconstructs actual density from C.
Do not confuse original frozen D.npy with that actual wrapper density.
The actual BASE banner is `Device memory, DF int plan:`. CAND additionally emits
`cuESTJK plan created #1`, then five `cuESTJK plan reused #1` messages at print3.
`fixed_jk.traces` is reused unchanged; no invented counter binding.

BASE c56708fc6f6e8fcf2801868f0d3cb49bdb1bb0a2:
`psi4.saptdft-cuest-major2-jkexception-baseline-20261018/build_major2_cuda_jkexception/stage`,
core 67c1e9021cc2fcaaf3b4ad3f98f6750ee39049c1aafe8cdfbe75b46c8335f224.
CAND 21a7b4d9c922f293d6220cafd8143d70f7d85e18:
`psi4.saptdft-cuest-major2-nonc1-candidate-20261019/build_major2_cuda_nonc1/stage`,
core 839e7bc2e52fe9d435689225fa718946710bb72d8147357ae3a8578fb8bdbf58.
Pair digest c9d4ec615a37a6a50d493fcfa6c01cf567bea9b616f298ef1fd49fcea0e87a33.

`repeatability_campaign.Engine.prolog` retains the full same-allocation
source, stage, core, runtime, ABI13030, CUDA13.3, SDK0.2.50, native FP64, SASS,
143-driver, config/compile-database, original122-positive-gate and import checks.
It adopts existing metadata and never builds or reruns those chemistry tests.
Its existing compiled CUDA logical-device UUID helper selects the physical
NVML UUID; never equate CUDA logical0 to NVML index0.
Each process additionally verifies checked cudart version/count/current device,
maps after CDLL, maps after work, and exclusive own PID at every output.

`native_plan.common.check_env` is reused unchanged, before scientific imports.
Site CUDA_ROOT12.6 and pinned CUDA_HOME/PATH13 are locator evidence, not numerical
controls. No numerical overrides are removed/added to obtain a passing run.
The original `arm.sh` alone binds each actual stage's `--psiapi` and runtime.

## Fixed work and timing

One 4h gts-cs207-chemx/embers H200 GPU, 8CPUs, 128G, one node.
Psi4 host memory100GiB. Job-specific scratch under
`/storage/scratch1/3/awallace43/cpks-major2-planreuse-speed/$SLURM_JOB_ID`, never /tmp.

Six sequential fresh processes: BASE1,CAND1,CAND2,BASE2,BASE3,CAND3.
Each keeps one JK and its coefficient Matrix alive. Exact original frozen
C.npy1786x465 bytes are added once; no right C. No factor generation.
Protein157, nbf1786/naux7265, 6-31+g** puream, def2-universal-jkfit;
alpha1/beta0/omega0; J/K true, wK false; threshold1e-12, slices5/moduli8.
All `fixed_jk.OPTIONS` are checked rather than merely requested.

Warmup initialize once, add C once, compute once and validate. Exclude warmup
from primary. Five rounds: initialize then compute, then another same-plan
compute. Finalize only once after all five. Per-process exact calls:
initialize6, compute11, finalize1, C_left_add1, C_right_add0.
BASE creates6; CAND creates1/reuses5. Total66 computes.

`Timer.initialize_compute` sequence:

    checked sync; t0; initialize; checked sync; t1; compute; checked sync; t2

Store initialize=t1-t0, compute-after-initialize=t2-t1, combined=t2-t0.
The intermediate checked sync and clock overhead are deliberately included,
identically in both arms. Control: checked sync; t0; compute; checked sync; t1.
`time.perf_counter_ns` only; strictly positive integer samples.
ctypes loads the exact already-required pinned libcudart realpath/hash, with
explicit argtypes/restype and checked status; maps checked after loading too.
No CUDA events, new library, SDK compile, or inferred asynchronous timing.

Trace/array/density/C validation, host copies, hashes, scalars, differences,
snapshot IO, nvidia-smi/PID checks, environment/probe analysis all occur AFTER
the end clock, never between t0,t1,t2. No sampler/per-loop NVML inside windows.
Setup/full process elapsed and warmup are separate from primary.

## Numerical findings and storage

After every compute capture actual `jk.D()` and J/K host copies. Every actualD
hash must match across all66 calls/processes, or stop acceptance and preserve
the exact identities/difference metadata. Density is never written back.
Scalar is `2Tr(actualD J)-Tr(actualD K)`, not original frozenD contraction.
Every J/K compared against its own process's warmup: hashes, maxabs, relative
Frobenius, scalar differences and symmetry findings retained for all66.

Original maxabs1e-8/relF5e-12/scalar1e-8 bounds are findings, not outlier filters.
Baseline finite breaches remain visible and are not dropped. Candidate
unchanged-initialize outputs must remain inside those prior roundoff bounds;
otherwise retain findings but withhold performance acceptance. Nonfinite,
wrong input/backend/runtime/count/timing or signal is fatal. Original full-SAPT
gate remains unchanged and unresolved; numerical variation is not newly fixed.

Save only warmup and round5 last-control J/K, exclusive fsynced readonly NPY:
four arrays per process,24 arrays =612440832 payload bytes plus NPY headers.
All intervening hashes/scalars/diffs are JSON metadata only. A reference and
current host copy are transient. Process reservation110MiB and total2GiB cap.
Original FACTOR.json0600 is accepted only at original path with exact SHA;
original arrays0444 remain unchanged. NEVER chmod historical files.

## Statistics, custody and interpretation

Primary = median of three BASE process medians / median of three CAND process
medians for combined initialize+compute. Display all three process medians and
each process's raw five samples/range. Balanced adjacent pair ratios secondary.
Also show initialization, compute-after-init, same-plan compute-control, actual
creation/reuse counts, setup and warmup. No dropped samples/tuning/re-warmup.
Descriptive n3 only: no invented CI/significance, no SCF-iteration confounding,
no whole-SAPT/RHF-finalize/general-GPU-speedup claim.

Failure writes SPEED-FAILED, never partial accepted summary. Full success needs
SPEED-REPORT, SPEED-COMPLETE and unsignalled supervisor completion. Markers are
written/fsynced by explicit Python, not optimistic shell exit0.

Before freeze/stage/parent submission, verify controller/headroom and no active
own major2/H200 holder. Preserve all16 actual old roots (13FAILED+3COMPLETED,
including native14178506) by stats/bounded small hashes only; never read failed
full-SAPT timing reports or old large arrays/core bytes on login.
Original native-v1 unsubmitted stage is preserved separately. Exact-resource
sbatch --test-only only in preparation.

Tracked helpers move by Git commit/push/fetch into a new checkout. Generated
manifest/batch/receipt alone move as data. New run must remain pristine.
Parent approval must equal exact manifest SHA in MAJOR2_SPEED_PARENT_REVIEW.
Local AND remote permanent exclusive attempt directories are fsynced before
any real submit request. Numeric accepted job ID is fsynced before controller
query. Rejection, lost SSH, timeout or ambiguity NEVER allows automatic retry.
Parent finalizer is deliberately not invoked by tests/preparation/job payload.
