# Preregistered minimal native J diagnostic

Preparation only; the parent must review and execute the separate finalizer.
Exactly one free embers H200, 8 CPUs, 128 GiB, 4 hours, no requeue. There is no
automatic submission, retry, extra cohort, CPU fallback, Psi4/SDK rebuild or
shared-environment mutation.

## Feasibility and scope

Installed cuEST 0.2.50 public headers were inspected before implementation.
DFIntPlan exposes Create, CreateWorkspaceQuery, Destroy, **no metric or fitted
three-center getter**. The installed Create documentation describes building
both. No opaque buffer reading or SDK reverse engineering is authorized.
Lifetime contracts allow the exact A/B/C schedule below. Archive
`cuest_v0.2.50.tar.gz` contains `examples/c_api/common/helper_workspaces.h`
and `examples/c_api/3_density_fitting/core_df_jk/main.c`; no extracted examples
directory is installed. Workspace allocator example does not initialize buffers;
J API documents that it overwrites its output. Retain both policies consistently.

The stored wrapper `condition_` is not passed to the SDK. Installed parameter
documentation gives SDK defaults: fitting cutoff 1e-12, relative conditioning,
QR inversion, stored three-index integrals (not integral-direct). Leave these
defaults unchanged; do not attribute them to a passed Psi4 option.

## Exact input seam

An allocation-only BASE Psi4 setup process imports the exact staged core
(`67c1e9021cc2fcaaf3b4ad3f98f6750ee39049c1aafe8cdfbe75b46c8335f224`),
exports public GaussianShell exp/coef/coord, global shell index, center,
function offset, angular momentum, pure flag, primitive count, and molecular
Bohr coordinates. It performs **zero SCF, energy, or JK operations**.
Coef is verbatim normalized `coef()`, never original_coef/erd_coef. Missing API
is fatal; no coefficient reconstruction. The exact suite path/hash comes from
the previous frozen fixed-JK manifest, not a guessed geometry file.
Dimensions 157/1786/7265, primary 6-31+g**, aux def2-universal-jkfit, puream true,
C1, original factor rank 465.

Density bytes are extracted from the original read-only D.npy (job14174849),
hash checked as both NPY file and array, then reloaded from read-only D.bin and
compared as uint64 bit patterns. No factor regeneration, C multiplication or
substituted density. **Original D is not claimed byte-identical to the old
JK::compute_D reconstructed density.** No old/new scalar crossproduct is done.

`NDFPLAN1` wire format: 8-byte magic; little-endian uint64 natom, primary nbf,
aux nbf; 3*natom IEEE754 binary64 molecular xyz; for each basis uint64 nshell,
natom counts, then each shell's uint64 index/center/function-offset/pure/am/nprim,
3 binary64 shell xyz, nprim exponents, nprim coefficients. No padding. Exact EOF
required. Explicit limits: 128 MiB input, 512 centers, 16384 basis functions,
32768 shells/basis, 512 shells/center, 256 primitives/shell, 2 million primitives.
Production enforces exact 157/1786/7265. Shell coordinates are retained as
manifest evidence; like the wrapper, pairlist uses final molecule coordinates.
Do not replace those with shell coordinates. JSON shell snapshots additionally
provide readable, IEEE-double-roundtripping provenance.

## Native schedule

Compile only this owned diagnostic inside the future allocation using the
previously verified conda GCC14 wrapper/resolved frontend and CUDA13 headers.
Record source hashes, full compile argv, ELF hash/dynamic section and mapped
runtime hashes; native measurement process cannot link/load libpsi4, Python or
Einsums. Reuse the prior compiled CUDA13 identity helper without rebuilding it.
Use the same explicit stream/cuBLAS/cuSOLVER construction pattern as BASE,
but check every status and synchronize at construction/compute boundaries.

Three fresh sequential native processes, one handle each:

1. Create primary/aux bases and pairlist once. Create plan A, J twice.
2. Destroy only plan A and its persistent storage; retain bases/pairlist/handle.
   Create plan B, J twice.
3. Destroy plan B before upstream destruction. Recreate bases/pairlist, then
   plan C, J twice. Destroy plan before upstream objects.

Total: 3 handles, 6 upstream builds, 9 plan creations, **18 J arrays**.
Never overlap giant plans. Fixed native FP64, operator alpha1/beta0/omega0,
pair threshold1e-12, SDK default J parameters. No J zeroing, workspace poisoning,
math/stream/handle/workspace-control variations. Reject inherited numerical
controls before imports and in every native process, never silently unset them.
Cache/scratch path and thread settings follow the original UNSET runtime.

Actual H200 UUID comes from CUDA logical-device mapping, not scheduler numeric
index assumptions. Match CUDA/NVML UUID; reject wrong hardware. Require empty
GPU before each process and exactly the native PID on that UUID at each
construction/compute boundary. Shared binaries and data are read-only.

## Numerical report and high-signal verdicts

All 18 readonly FP64 J arrays and **153 raw all-pair differences** retained.
Report SHA256, maxabs, relative Frobenius norm using max(normA,normB,tiny),
symmetry maxabs, and fixed-density **Tr(DJ)**. This is not prior 2TrDJ-TrDK.
Group counts: 9 same-plan, 12 within-process DF-only, 24 within-process upstream
rebuild, 108 cross-process pairs. Exact integer-bit equality includes signed zero.

Verdict precedence, descriptive only:

* SAME_PLAN_VARIATION: contraction/repeated-compute variation is present.
* DF_BOUNDARY_VARIATION: same-plan exact, A/B differs with upstream retained.
  Implicates the plan boundary as a unit, not metric vs tensor or a unique cause;
  cannot exclude plan-selected contraction order or lazy initialization.
* UPSTREAM_BOUNDARY_VARIATION: A/B exact but C differs.
* FRESH_PROCESS_ONLY_VARIATION: within-process exact but processes differ.
* MINIMIZATION_FAILED_NO_VARIATION: all exact. Stop; **does not clear wrapper**.

Any finite difference/symmetry discrepancy is a finding, not a fatal threshold.
Input/backend/identity/count/nonfinite/SDK/CUDA/failure/signal/missing artifacts
are fatal and cannot produce COMPLETED. No performance/timing summaries, ratios,
timing unblinding, full-SAPT acceptance, publication, or automatic followups.
2 GiB total output cap; native arrays use 459,330,624 bytes. Inputs ~26 MiB,
bounded metadata/logs and diagnostic ELF leave ample reserve. Compilation and
analysis have bounded budgets, signals propagate, unsignalled success alone
permits the fsynced completion marker.

## Submission custody

Stage tracked helpers only by committed Git push/fetch/worktree. Generated
bundle data may cross stdin. Verify exact commit, clean tracked state, every
owned helper SHA, pristine run, all 14 old roots (12 failed, 2 completed),
scientific build stat sentinels, old staging, queue/headroom and exact-resource
sbatch --test-only. No broad matrix rehash on login.

Finalizer is parent-only, with permanent exclusive local and remote attempt
directories. After any attempted sbatch, ambiguity is terminal: no retry.
Numeric accepted ID is written and fsynced before controller query. A timeout,
nonzero exit, malformed response or query failure needs human reconciliation.
