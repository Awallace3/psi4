# Frozen reused-factor Protein157 endpoint protocol v2

This is not an accepted performance experiment, molecular energy, full SCF,
guess, response calculation, or substitute for the failed primary comparison.
The unchanged original SAPT component gate remains failed. No primary helper
or scientific API is modified.

## Source review and identity

Parent harness f52cc6e9f704f89f2a45303106d2b3807b7777e1 has148 checks.
All prior test functions remain byte-identical. Only the owned fixed-JK
helpers/protocol and new regression tests/data change.
Compiled BASE c56708fc6f6e8fcf2801868f0d3cb49bdb1bb0a2
and CAND21a7b4d9c922f293d6220cafd8143d70f7d85e18 are reused, never rebuilt;
origins14165998/14170562 both have compiled_in_this_job=false.
Exact pair digest c9d4ec615a37a6a50d493fcfa6c01cf567bea9b616f298ef1fd49fcea0e87a33.
Full pair/source/stage/config/compileDB/ABI/SASS/import/runtime/143-driver
verification is repeated on the allocated H200 host. Existing identity helper
is pinned and reused; expiry is fatal, never an implicit rebuild.

Original B1 job14173808 stage0 NPZ is never overwritten. Archive SHA
dbe9d5ccbab8ebdb8299c6a8ba6463997ba3aca24fb828f124edf1c0f7b0c32e,
Da1786×1786 little-endian FP64 SHA
4a17f42cba7778bad576751b2cad0a066111c9231dd4641c347fb79c75b6a410.
Original COMPLETE.json bytes are frozen too. All12 failed roots (including
job14174849), both earlier staging bundles and failed fixed-v1 stay unchanged.
New run/harness/cache/bundle namespace is major2-fixed-jk-20261019-v2.

`sad.cc:377–380` constructs Ca by
`Da_->partial_cholesky_factorize(options_.get_double("SAD_CHOL_TOLERANCE"))`.
`read_options.cc:1971` defaults that threshold to1e-7. `sad.cc:1341ff`
copies these factors and temporarily sets the occupied dimensions to their
rank. RHF::form_G passes Ca_subset(SO,OCC), not Da, to JK.
Job14174849 ALREADY generated rank465 with that exact native BASE core.
v2 MUST NOT factor again: the executable protocol has only reuse-factor/jk
modes. The original factor_once helper remains solely for prior local tests.
The allocation verifies and exclusively copies the exact original readonly
C.npy/D.npy and byte-identical FACTOR.json into NEW raw/factor, rechecks the
source and destination, and writes a distinct REUSED_FACTOR_ORIGIN.json
(original job14174849, new_factor_calls=0). The old receipt is never relabeled.

Pinned FACTOR.json b6777acd5ec6098c0a81bcf08af74591b8c17edb511c4bd04c5aee9be3c9121e;
C.npy 76b2aa312d8b6de1568224a793f33c634630b8d7f418fbd4a2b10522220a92a0;
D.npy 76cef9607770669ffd08b5f5599b385bb2f1e42ed57254261c5209dfb3025f92.
Complete array hashes/shape/dtype/rank/tolerance/original Da/NPZ/BASE core
are frozen in FROZEN_FACTOR and the manifest; all file hashes and origin are
checked throughout every child, before/after copying and at parent boundaries.
Original receipt reconstruction maxabs2.220446049250313e-16 and relative
Frobenius1.7586966334785784e-16 are retained as original measurements.
Every JK process reads those exact already-generated C bytes; no refactoring.

`rhf.cc:323ff`: coulomb_E=2*Da.vector_dot(J), exchange_E=-alpha*Da.vector_dot(K).
Bare HF alpha1,beta0,omega0 gives 2Tr(DJ)-Tr(DK) for common D=C@C.T.
This scalar excludes the one-electron, nuclear, XC and SAPT terms; it is not
a molecular energy. Source-semantic sentinels and exact file hashes are frozen.

## Dispatch and trace gate

Six fresh processes: B1,C1,C2,B2,B3,C3. Geometry is the frozen Protein157
suite string.6-31+g**/def2-universal-jkfit, PUREAM true;157 atoms,1786 AO,
7265 JKFIT functions. DFJK resolves to cuESTJK, never CPU/direct fallback.
FP64 math mode; J,K true,wK false; no right factor;8threads/100GiB.
Original SCF INTS_TOLERANCE1e-12, DF_FITTING_CONDITION1e-10,
Ozaki slices5/moduli8 and native8,000,000,000-byte variable device workspace
are unchanged; no deterministic/CUBLAS/affinity knobs.
PRINT3 is for lifecycle evidence, explicitly NOT a performance workload.

One C_left_add receives the exact native Matrix copied from C.npy. Its actual
array hash, shape and465 columns are checked before and after every compute.
No writes to native coefficient/J/K arrays. Each process performs exactly:

- A: initialize fresh, compute, snapshot.
- B: compute again without initialize, snapshot.
- C: unchanged initialize, compute, snapshot.
- D: finalize, initialize, compute, snapshot; final cleanup finalize.

Exactly3 initialize,4 compute,2 finalize,1 left-add,0 right-add calls per
process. BASE cumulative creations after A/B/C/D are1/1/2/3, with no reuse.
CAND cumulative creations1/1/1/2 and reuse0/0/1/1, created counters1,2.
Counts are checked both before and after each compute, immediately after
initialize, after explicit finalize, and after final cleanup.
BASE lacks new counters: use its native "Device memory, DF int plan:" banner
which follows successful creation, just as the unchanged focused regression.
CAND counters must agree with those banners. No optional extra object is used.
Actual PID/physical GPU UUID is checked at each snapshot; all6 PIDs must differ.
The factory and dispatch initial jk.name()=="cuESTJK" checks stay mandatory,
with additional live name guards before/after each state and before compute.
Each state and COMPLETE records actual_backend, independently verified.

The exact three-line simple-DF constructor warning at jk.cc:120–122 is
allowed ONCE at the beginning of a trace, ignoring LF-only leading blank
lines. It precedes USE_CUEST selection and is not fallback evidence.
No global DiskDFJK whitelist: altered, duplicate or moved warnings, actual
fallback/SDK text and wrong live backends stay fatal. Raw plan.out bytes
are never changed. Captured 329-byte job14174849-plan.out is SHA256
d7cd3bba1ae1977f788df11b53077c76eae30d10fa4961dae52560c8e8488777.
The pinned old parser must reject; v2 parses zero initial creations, then
native-like suffixes prove exact A/B/C/D counts (not actual GPU proof).

## Documented workspace-control cohort (preparation only)

The new namespace is `major2-workspace-20261019`, parent harness
`b9abe3e38fd194524185648d9e15807a8d66d4c9`. Exactly six real JK children
receive `CUBLAS_WORKSPACE_CONFIG=:4096:8` in a fresh Popen environment copy.
MAIN/controller, CUDA identity, import/probe-only and factor-copy environments
remain unset. Unexpected inherited overrides cause STOP, never silent removal.
`child-arm.sh` verifies the whitelist before the unchanged `arm.sh` executes
any native `--psiapi`; the runner writes/fsyncs a readonly raw ENV receipt
before importing numpy, policy, probe or Psi4. Every output/COMPLETE and the
final report carry observed ENV provenance; missing/wrong/extra overrides,
receipt edits or identity mismatch are fatal.

Source: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
(current docs13.4; installed CUDA13.3/GCC14/cuEST0.2.50 unchanged).
NVIDIA documents this debug ENV as a workaround for cuBLAS concurrent-stream
reproducibility (~24MiB extra workspace). It is not a guarantee for cuEST,
custom kernels or cuSOLVER; no mode/precision/cutoff/stream/handle change.

Completed UNSET job14175243 is preserved separately, requiring
`COMPLETED 0:0` job and batch accounting, never added to FAILED_JOBS.
Its full file-set/stat proof and small JSON/trace hashes are frozen; old
readonly arrays are not rehashed on login. Full new allocated source/stage,
ABI/SASS/runtime/config/143-driver and factor/input checks remain unchanged.
New analysis retains all24 outputs/all276 comparisons and reports three
explicit anchored pairs per arm for A/B (sameplan), A/C (unchanged initialize),
A/D (finalize/newplan) and fresh A/A. It compares those numerical summaries
and unchanged finding bounds to the preserved old report, without old-new
matrix crossproducts, timing fields, SAPT acceptance or causal guarantees.

Host/fake tests use `workspace_checks` in addition to the unchanged prior161.
No GPU result exists until parent review and the one-shot free-embers run.

## Persistence and numerical findings

All24 outputs (48 J/K arrays) are saved as compressed NPZ, with file and array
hashes, byte sizes, shapes, dtypes, stage/state/provenance receipts and readback.
Each process has a300MiB cap; total NPZ+factor NPY storage has a hard2GiB cap,
checked at every process boundary and analysis.48 full1786² FP64 arrays use
1,224,881,664 bytes; conservative ZIP overhead is reserved before each write.
Actual stored bytes, not just arithmetic, are gated. Existing archives untouched.

Analysis on the allocated GPU host reports all24 scalars/hashes/symmetry
residuals and all276 pairs, classified same-plan, unchanged-reinitialize,
finalize-new-plan, fresh-process-within-arm or fresh-process-across-arm.
Differences include bitwise equality,maxabs,relative Frobenius and scalar delta.
Frozen findings bounds: matrix maxabs1e-8,relF5e-12,scalar1e-8Eh,
symmetry maxabs1e-10. **Finite bounds violations are findings, not stops**:
retain the complete cohort. Nonfinite, SDK, fallback, mutation, identity,
count, source/runtime or storage failures are fatal. No tolerance/physics fix,
scientific equivalence or performance verdict is sought or produced.
Endpoint report uses a bounded schema and rejects measured-duration/ratio
fields. Native logs are retained but never summarized for performance.

## One-shot preparation and handoff

Only one4h gts-cs207-chemx/embers/gpu-h200 H200×1/8CPU/128G/no-requeue
allocation is eligible. Reserve6×1800 child budgets+600 verified copy+1200 analysis
+300 teardown after prolog; stop if admission fails. No external warmups.
No real factor, JK, cuEST or energy is run on a login node.
Fake/host tests establish protocol behavior only, not GPU endpoint correctness.

`fixed_jk_handoff.py` collects bounded preservation/input proofs, stages only
the three explicit generated bundle files, audits source pins and performs
actual sbatch --test-only. Code moves by Git. Only parent-submit with the
explicit review token calls the unchanged guarded submission helper:
permanent exclusive remote lock, durable attempt, fsynced numericID before
controller queries, ambiguity consumes the attempt. Local finalizer adds its
own permanent attempt directory. Neither lock may be deleted or retried.
The preparation worker never executes that finalizer or an actual submission.
