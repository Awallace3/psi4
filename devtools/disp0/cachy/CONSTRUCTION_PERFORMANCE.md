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
