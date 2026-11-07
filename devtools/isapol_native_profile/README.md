# Exact-input native stage profiler (Phase B)

**Profiling only; not acceptance. No scientific source/kernel changes.**
Parent owns fresh remote free-embers execution. Nothing here submits jobs,
contacts QCArchive, changes held jobs, builds/reinstalls Psi4, or updates receipts.
The driver refuses local `--run` without a SLURM allocation.

## Files to stage through Git

```
devtools/isapol_native_profile/.gitignore
devtools/isapol_native_profile/profile.py
devtools/isapol_native_profile/telemetry.py
devtools/isapol_native_profile/fixtures.json
devtools/isapol_native_profile/test_profile.py
devtools/isapol_native_profile/README.md
```

Do not alter the sealed scientific source or environment. `fixtures.json` binds
the six exact private copies for each accepted 0001/0002/0025. Do not commit
fixtures/tensors or regenerate them from production. Source staging is through
Git only; no push/commit/deploy is performed by this harness.

## Required paths and identity

| Variable | Required existing path |
|---|---|
| `DRIVER` | Absolute Git-staged `devtools/isapol_native_profile/profile.py` |
| `FIXTURE` | Private copied `monomer-0001` (or 0002/0025) directory containing `case.json`, `expected-runtime.json`, `claimed-input.json`, `pilot-outcome.json`, `server-native-envelope.json`, `server-tensors.npz` |
| `BUILD` | Qualified science1aaa build root, containing `build-identity.json` and `env/` |
| `BASE` | Original ordinary dependency base `/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt` (same reviewed dependency route as production science CLI) |
| `QUALIFICATION` | Original `installed-validation.json`; sibling `imported-identity.json` and `junit.xml` required |
| `QCE_PACKAGE` | Actual deployed manager QCEngine package directory, e.g. `/storage/project/r-cs207-0/awallace43/gits/qcengine/qcengine` |
| `QCE_HARNESS_SHA` | `harness_sha256` from the actual accepted worker preflight receipt, independently checked by parent; do not self-approve by hashing an arbitrary replacement |
| `OUT`, `SCR` | New, nonexistent, disjoint profile-output and owned scratch paths. Never production paths. `SCR/cache` is the fresh interpreter bytecode lookup prefix. |

The local immutable fixture root supplied by parent is:

```
/scratch/awallace43/ccasp-s66-native-campaign-20261030/native-perf-inputs001
```

The driver verifies the four original grid400 receipt hashes, scientific
commit/tree, every manifest `installed_files` hash before/after, actual imported
package/core paths, and installed `implementation_identity`:

```
commit 1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2
core   ef69cddb2dd7b675688a733e4c5dd68b78fc78ac35b77b363114d638ac9dae44
Python e1476e2d0b8e395c40b48219ca8bc8def9e69035cab3ab970f6b88940bebc74b
```

It requires the qualified build's own interpreter (`sys.prefix == BUILD/env`).
Do not point PYTHONPATH at this worktree or install this worktree as science.
Fresh `-I -B -X pycache_prefix=SCR/cache` avoids reading old bytecode beside the
sealed sources and prevents writes into the installation.

Before **any** Psi4 import (also check-only), the driver enters a newly created,
owned/writability-probed `SCR/cwd`. It deliberately does not restore the caller's
CWD on return or exception: Psi4 finalization can still append `timer.dat` and
write `psi.PID.clean` at interpreter exit. The zero-impact ownership marker and
those relative artifacts stay in owned scratch and count in its actual samples.

## Pure tests: no installed Psi4 needed

```bash
PROFILE_FIXTURE_ROOT=/scratch/awallace43/ccasp-s66-native-campaign-20261030/native-perf-inputs001 \
  python3 -I -B devtools/isapol_native_profile/test_profile.py
```

Without `PROFILE_FIXTURE_ROOT`, the optional readonly fixture test skips.
All other tests use stdlib/fakes/temporary files, not quantum calculations.
The static line-probe test reads the scientific source without importing it.

## Admission-only preflight

Run on parent's compute allocation, not the login node (imports/hashing still
consume resources). Set all variables above. Use a new `OUT`/`SCR` for preflight;
the subsequent run must use different new paths.

```bash
N=8
env PATH="$BUILD/env/bin:$BASE/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$BUILD/env/lib:$BASE/lib" \
  OMP_NUM_THREADS="$N" MKL_NUM_THREADS="$N" OPENBLAS_NUM_THREADS=1 \
  "$BUILD/env/bin/python" -I -B -X "pycache_prefix=$SCR/cache" \
  "$DRIVER" --check-only \
  --fixtures "$FIXTURE" \
  --build-manifest "$BUILD/build-identity.json" \
  --qualification "$QUALIFICATION" \
  --qcengine-package "$QCE_PACKAGE" \
  --qcengine-harness-sha256 "$QCE_HARNESS_SHA" \
  --threads "$N" --scratch "$SCR" --output "$OUT"
```

Preflight hashes immutable inputs and identity, checks actual QCEngine source,
imports the installed science package, sets memory/threads/owned scratch, and
performs descriptor/request admission only. **No SCF, grid or response runs.**
It writes `manifest.json` with `ADMITTED_NOT_EXECUTED`, input views and
`wall-time-source.json`. Inspect those before running science.

## Exact-input profiling invocation

Use a **fresh process per N**, explicitly N=1,2,4,8. Do not change science memory
between thread counts. The driver's memory comes from the hash-bound accepted
case: 20 GiB for 0001/0002, 60 GiB for 0025.

```bash
N=1  # repeat fresh invocations at 2,4,8; choose new OUT and SCR each time
env PATH="$BUILD/env/bin:$BASE/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$BUILD/env/lib:$BASE/lib" \
  OMP_NUM_THREADS="$N" MKL_NUM_THREADS="$N" OPENBLAS_NUM_THREADS=1 \
  /usr/bin/time -v -o "$TIME_LOG" \
  "$BUILD/env/bin/python" -I -B -X "pycache_prefix=$SCR/cache" \
  "$DRIVER" --run \
  --fixtures "$FIXTURE" \
  --build-manifest "$BUILD/build-identity.json" \
  --qualification "$QUALIFICATION" \
  --qcengine-package "$QCE_PACKAGE" \
  --qcengine-harness-sha256 "$QCE_HARNESS_SHA" \
  --threads "$N" --instrumentation stages --sample-seconds 1 \
  --scratch "$SCR" --output "$OUT"
```

`TIME_LOG` must have an existing fresh parent outside `OUT` and `SCR`, since
`time` opens it before the driver creates those directories. Parent should use
consistent physical-core binding (e.g. its reviewed `srun --cpu-bind=cores`
wrapper) and record actual CPU model/affinity. Use a TERM-safe parent wrapper
which propagates the child status; the driver records SIGTERM/SIGINT as failures
and never marks interrupted calculations complete. SIGKILL cannot guarantee
final summary delivery; flushed begin/end events remain partial evidence.

An 8-thread `--instrumentation off` control has the same exact input, scratch
sampler and output sidecar capture, but no detailed stage/function/line hooks.
It is a **low-instrumentation control**, not a completely pristine production
route. Compare its numbers and wall time with the stage-instrumented 8-thread
run before interpreting small speed differences.

## What remains exactly scientific

- Execute the **unchanged** `claimed-input.json:function_kwargs.input_data` via
  installed `schema_wrapper.run_qcschema`, including original input protocols.
- Original properties call performs fresh SCF; only then does the installed
  schema call the native producer. No saved orbitals/partitions/responses.
- PBE0/aug-cc-pVTZ, original AUX/shape bases, strict SCF stopping criteria,
  smoothing, fitting/lattice/quadrature settings, caps and thresholds unchanged.
- **Partition/Q 400/2030**, **response 100/200**, ISA iteration recipe 100/200,
  **Drho32**, `reference_h2h1` remain distinct and unchanged.
- Initial admission also compares science fields in expected runtime with the
  pinned actual claimed v1 input. QCElemental originally added molecule defaults
  and stripped the `fix_symmetry` field; the exact claimed input is executed, and
  the native request still enforces computational C1. Nothing is “fixed” in it.
- No BLAS pool limit manipulation. Production-style OMP/MKL=N, OpenBLAS=1,
  explicit `core.set_num_threads(N)`. Loaded pools are inspected, not reconfigured.
  Alongside optional `threadpoolctl`, read-only exported thread-count getters are
  queried through `RTLD_NOLOAD` handles of already mapped BLAS/OpenMP libraries.
  Unsupported symbols are reported absent; no new backend is loaded.

## Profiling/output-only differences

- Direct installed schema execution omits manager, QCEngine CLI launch/transport,
  production OS strace and queue/audit/acquisition overhead. It is not a replay
  of the accepted `provenance.wall_time` scope.
- Python function wrappers and StageLog windows use monotonic clocks. Hooks
  call originals with unchanged arguments/returns and restore bindings.
- Two function-local line-boundary probes split Drho-C, density sampling,
  controller construction/cache, initialization, ISA iterations/final tails,
  cache release, and P construction versus contractions. They observe statement
  boundaries, never numerical arrays. Python tracing dispatch adds overhead;
  quantify using the off control. They refuse an existing debugger trace hook.
  They also report the actual controller cache admission and scalar Drho
  diagnostics, and P's shape/storage, without copying numerical operands.
- Original `_read_output` text is additionally copied to separate sidecar files.
  Its return and input verbosity/stdout protocol remain unchanged, so empty
  persisted result stdout is no longer the sole source of timing evidence.
- A sampler thread scans only owned scratch (no symlink traversal), plus process
  `/proc` status. Scratch maxima are **sampled lower bounds**, not reservations
  or guaranteed peaks; short-lived files can be missed. Samples may perturb I/O.
- No rebuild or scientific module edits. The process is nevertheless
  **instrumented, COMPUTED_NOT_ACCEPTED**, never promoted to ordinary acceptance.

## Outputs and interpretation

| File | Meaning |
|---|---|
| `manifest.json` | Input/build/harness hashes, fixed declarations, threads/pools/affinity, allocation env, profiling-only qualification |
| `used-qcschema-input.json` | Actual unmodified claimed input passed to installed schema |
| `normalized-input.json` | Separate expected-runtime view for admission, not the executed input |
| `wall-time-source.json` | Actual deployed QCEngine source paths/hashes/functions, timer-scope recognition |
| `events.jsonl` | Flushed monotonic span begin/end and StageLog windows |
| `samples.jsonl` | Owned scratch counters and `/proc` RSS/thread samples |
| `psi4-output-*.txt` | Separate full Psi4 output before schema protocol suppression |
| `qcschema-result.json` | Complete raw JSON result, including failure result if returned |
| `server-native-envelope.json` | Complete native envelope, input-compatible name for independent validator |
| `native-result.json` | Complete exported numerical result metadata |
| `server-tensors.npz` | Unmodified exported local/refined tensors |
| `hooks.json` | Exact runtime-only hook inventory |
| `summary.json` | Terminal status, process peakRSS (Linux bytes), sampled scratch maxima/errors, final pools and identity checks |

`summary.status == COMPUTED_NOT_ACCEPTED`, zero exit, no telemetry errors and
`installed_science_unchanged == true` mean a completed profile, **not acceptance**.
Use a fresh independent numeric check:

```bash
"$VALIDATOR_PYTHON" \
  "$SOURCE/devtools/isapol_s66_campaign/validate_numeric.py" \
  "$OUT" --output "$OUT/independent-numeric-validation.json"
```

Parent should compare every local/refined tensor entry and every C6/C8/partial-C10
pair across N and against the copied accepted tensors, using existing approved
gates and disclosing max-absolute/scale-aware differences. Do not relax
residuals/ISA convergence/PFIT/native verification/coverage. No numeric
agreement is claimed by the harness itself.

Span durations are inclusive; StageLog windows overlap function spans.
**Do not sum overlapping durations.** H2H1 operator construction is separately
hooked because it is otherwise charged to the preceding StageLog window.
The Drho/ISA line intervals are nested inside native_partition.

## Source-checking accepted provenance wall_time

The driver reads (does not import/execute) actual deployed QCEngine:

```
QCE_PACKAGE/util.py                compute_wrapper, handle_output_metadata
QCE_PACKAGE/compute.py             compute
QCE_PACKAGE/programs/psi4.py        Psi4Harness.compute
```

It records full selected function source, line numbers and SHA256; requires
the harness hash supplied from the accepted worker preflight; recognizes the
known `comp_time`/`metadata.wall_time`/provenance transfer pattern, or refuses
unresolved attribution. Parent must review captured context for retries and
launch/transport boundaries. The old util.py/compute.py bytes were not separately
sealed in the supplied accepted fixtures: current source plus harness equality
alone is **not proof of historical util.py/compute.py identity**.

## Minimum credible finite resource request

These are bounded planning recommendations, not measured minima or runtime
guarantees. Parent verifies free-embers/account/partition and CPU class.

- **First job:** accepted 0001 only; 1 node, 1 task, **8 CPUs, 32 GiB job memory,
  science20 GiB, 02:00:00, explicit embers**. Sequential fresh-process N=1/2/4/8
  plus one off control at8. Reuse a fixed 8-core affinity/cpuset, varying actual
  scientific threads, rather than changing node types between N.
- The accepted small job already used 32 GiB job/20 GiB science memory. Until
  peakRSS exists, do not shrink memory from that shape. Owned scratch capacity
  bounded by parent at the accepted **256 GiB** ceiling; sampler does not reserve
  the old arbitrary inode ceiling.
- Optional 0002: same shape, separate bounded job after 0001 establishes utility.
- **0025:** do not place an unmeasured full1/2/4/8 sweep in the first job. Its
  accepted16-thread wall-time metadata is ~4429.6 s, not a bound at1 thread.
  Start a separately approved **8-thread**, science60 GiB/**96 GiB job memory**,
  **04:00:00 embers** diagnostic if parent wants the large-case contrast. Parent
  must validate memory-per-core routing; no claim it fits the original cpu-small
  route with fewer cores. Choose later counts only after its stage profile.
- No automatic resubmit/retry or paid QoS. If any run exhausts the finite budget,
  preserve partial evidence and stop; parent decides next steps.
