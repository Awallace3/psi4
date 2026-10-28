# cuEST construction performance: first discriminating experiment

Starting point: [independent review](../phoenix_validation/CLAUDE_PERFORMANCE_REVIEW.md).
The measured H200 whole-call regression remains unresolved. None of the source
changes below establishes a speedup or releases another Phoenix allocation.

## Changes

- `DiagnosticPlanOwner(..., jit_compile_threads=...)` explicitly sets and reads
  back the requested count. The compatibility default stays **1** pending
  measurement. Sibling RIFIT plans inherit the parent's explicit setting.
  Booleans, nonintegers and values outside the native API's 1–256 bound reject
  before ownership/native construction. Experiments use at most **4**.
- Owner reports include inclusive host wall and process-CPU timing for admission,
  snapshots, context configuration/create/sync, primary/auxiliary bases, pair
  lists, workspace queries/allocations, DF-plan creation and teardown.
  Shell intervals aggregate by primary/auxiliary role; their call counts remain
  visible. FP64 context-mode readback is recorded explicitly.
- Device DFMO reports separate identity/ABI audits, coefficient packing,
  output/workspace allocation, workspace query, native compute enqueue and
  synchronization. `producer_seconds` retains its original meaning.
- `construction_probe.py` compares fresh, serialized 1/4-thread processes on a
  qualified whole local installation. It creates real spherical Psi4 basis
  metadata but no SCF, provider compute or energy call. Independent JKFIT and
  RIFIT contexts coexist on the same session, retaining current ownership and
  reverse teardown. Live CUDA free-memory observations are **samples, not peaks**.

Every interval is **inclusive**. Nested/aggregate intervals must not be summed.
Process CPU includes native worker threads and may exceed wall time. Enqueue
time and following synchronization are not independent CUDA-event kernel
measurements. No additional synchronization was added to production code.

Identity verification, fresh byte audits, ABI rebinding, native lifetimes,
creation-temporary retention and memory-admission rules are unchanged.
Cache/reserve budgets and production numerical policy are unchanged.

## Local verification

CPU/fake-native/serial-shim tests exercise thread roundtrip and sibling
inheritance, exact-clock nested timing, failure timing, pending-input retention,
ambiguous-outcome quarantine, teardown, unchanged signed tensor equations,
paired-input rejection and a second-plan failure unwinding both contexts.
These tests are **not GPU numerical or performance evidence**.
The initial local run passed **307** root-suite tests and **4** construction
probe tests, with no skips. A separate **24** local-runtime/full-contract tests
also passed.

```bash
python3 -B -m unittest discover -s devtools/disp0 -p 'test_*.py' -v
python3 -B -m unittest discover -s devtools/disp0/cachy -p 'test_construction_probe.py' -v
```

## Cachy execution contract

User explicitly approved bounded serialized Cachy diagnostics. Use a new private
root `~/gits/psi4-disp0-perf-cachy`; the shared dependency environment and existing
roots are read-only. Stage source by Git, install the entire stage normally,
then generate a matching schema-2 whole-build/runtime proof. An older installed
stage plus a source-module overlay is forbidden. The probe deliberately refuses
an H200 configuration or an unqualified/changed installation.

Run within a user scope bounded to 4 CPUs, 12 GiB RAM, no swap and 128 tasks.
The probe acquires the common nonblocking GPU flock and inherits it into each
supervised child, honoring HOLD. It uses the authenticated import bootstrap.
Supply a frozen geometry file and digest; start with water/HCl before peptide.

```bash
# Invoke with the selected dependency environment's Python.
python -B source/devtools/disp0/cachy/construction_probe.py \
  --root "$ROOT" --attempt construction-water-01 \
  --config "$CONFIG" --config-sha256 "$CONFIG_SHA256" \
  --geometry "$GEOMETRY" --geometry-sha256 "$GEOMETRY_SHA256" \
  --seconds 600 --order 1 4
```

Default bases are aug-cc-pVDZ, aug-cc-pVDZ-JKFIT and aug-cc-pVDZ-RI.
Each child has a fresh CUDA cache directory. The vendor's internal/global cache
state is unknown: **do not label this a fully cold-cache comparison**.
`--order 1 4 4 1` permits a bounded order-reversed follow-up. Failed/time-limited
arms remain failed; their launch logs and partial JSON are retained.

The comparator requires identical geometry, basis fingerprints/dimensions,
plan policy, native version, FP64 mode and complete native/session cleanup.
The only intended policy difference is the explicit JIT thread count. Stream
addresses and timing/memory samples may differ.

This construction-only experiment intentionally has **no upstream SCF context**.
Its coexistence samples cannot establish whole-workflow peak-memory admission.
Actual native numerical gates, zero-skip GPU tests, before/after parity and full
workflow admission remain required before selecting a faster default or moving
to Phoenix. Cache-budget and context-sharing changes are separate experiments.

## Private build attempts

Scientific candidate: `a13849f8b066752e8f1df999321047136ae9f8da`.
The detached private source is clean; cuEST library and archive hashes match the
canonical 0.2.50 constants. The reference roots/environment remain read-only.

1. `psi4-disp0-perf-build-a13849f8b0.service` configured successfully, then
   **failed with `Result=oom-kill`** at the 12 GiB cgroup limit during C++
   compilation. Its log is preserved at
   `~/gits/psi4-disp0-perf-cachy/build-attempt-01-oom.log`. No installed-stage
   success or GPU test resulted. Top-level `-j 4` alone does not establish a
   bound on nested external-project Ninja builds. The generated
   `build/build.ninja` confirms that the inner step runs only `cmake --build .`,
   with no explicit job limit; the failed unit's environment did not set
   `CMAKE_BUILD_PARALLEL_LEVEL`. Its recorded `MemoryPeak` equals the exact
   12,884,901,888-byte limit. CPU quota throttles execution, not compiler-process
   concurrency or aggregate memory. This was a build-launch mistake, not water
   dimer or GPU memory usage.
2. Retry unit `psi4-disp0-perf-build-a13849f8b0-serial.service` explicitly sets
   both `JOBS=1` and `CMAKE_BUILD_PARALLEL_LEVEL=1`, plus one OpenMP/BLAS thread.
   The limits remain 4 CPU quota, 12 GiB RAM, zero swap and 128 tasks, with a
   90-minute deadline. Output is `build-attempt-02.log`. The completion monitor
   requires both `BUILD_OK` and the exact `install/BUILT_FROM` pin above.
   A subsequent point-in-time inspection showed 6,102,331,392 bytes peak
   (about 5.7 GiB) and 8 current tasks, with the serial environment verified.
   This is an intermediate build observation, not a completed-build result.
   The serial retry subsequently **completed successfully**: the monitor verified
   `BUILD_OK` and exact `install/BUILT_FROM`. Its native core is
   `install/lib/python3.13/site-packages/psi4/core.cpython-313-x86_64-linux-gnu.so`.

Neither a started retry nor a successful build is construction-performance or
GPU correctness evidence. The qualified-runtime proof and actual paired probe
must still complete.

`qualify_construction_root.py` supplies that private-root CPU qualification:
it compares the complete installed/staged Python inventories, checks
source-matching scientific files and the staged/installed native core. The
generated `__init__.py`, `extras.py` and `metadata.py` must match their configured
build outputs, not checked-in templates; generated version metadata must also
identify the clean seven-digit source pin. The qualifier admits
read-only dependencies against the existing frozen proof, and matches every
private CUDA copy by its original hash. It creates aliases only in the private
run directory; obsolete reference tmpfs paths are never revived or repointed.
It imports the exact new whole core before Einsums and runs a separate
**CPU-only H2 RHF/STO-3G build smoke with cuEST disabled**, then freezes a
schema-2 runtime proof. This build smoke is not part of the timed water-dimer
construction test and is not GPU numerical qualification.

The first private qualification attempt (`private-cpu-qual-01`) correctly stopped
before importing the core or running a GPU test, but on an overly strict helper
comparison: it compared configured `__init__.py` to its source template.
The corrected helper checks all three generated files against actual configured
outputs and retains installed/staged equality and version-pin checks. A separate
preload-contract regression also requires the canonical, actually mapped cuEST
SDK DSO in the binary proof even though it is outside the install/environment
prefixes; otherwise the unchanged authenticated bootstrap rejects it.
Keep the failed attempt unchanged and qualify with a fresh output directory.

`private-cpu-qual-02` **passed the whole private CPU RHF smoke and proof freeze**.
Before launching GPU construction, a separate file-only executable-mapping
probe reproduced the existing strict guard's Btrfs limitation:
`Mapped inode/device differs from the hashed file` for the private cuEST DSO.
No native cuEST call or GPU calculation was made by that mapping probe.
The durable install/build remains valid, but this persistent-file runtime is
not suitable for the unchanged mapped-identity gate.

The next qualification makes a fresh **per-experiment executable copy** on
tmpfs, using the already audited same-descriptor/inode/race-checking copier.
All authoritative dependencies and build paths remain on persistent storage.
The staging operation requires matching frozen bytes, a new private directory,
at most 2 GiB copied and at least 1 GiB remaining free, and a successful original
strict executable-mapping probe. CUDA aliases are private to the run. The new
runtime proof binds the copied SDK, mapped binaries and copy-helper provenance;
no identity check is bypassed. Executable copies are temporary test artifacts,
not new build dependencies.
