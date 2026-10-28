# Second performance step: remove nested admission work

**Candidate; no new Cachy GPU result yet.** The first owned-runtime gate remains
the last accepted small-water result. Its 155.5–158.6-second whole calls are not
considered a satisfactory performance fix.

## Measured target

The [first-step CPU profile](OWNED_RUNTIME_RESULTS.md) found approximately
232,000 complete live checks per energy call, at 361 us/check on Cachy.
Factory mapping audits cost96 ms and parse the same mapping text twice.
Native construction/JIT is not the demonstrated bottleneck.

## Changes and preserved obligations

- A complete live snapshot checks caller/process/thread/signal ownership at
  entry and exit. **Every file is still checked every time** with `stat`,
  `fstat`, `F_GETLEASE`, `F_GETOWN` and `F_GETSIG`. Use the pinned path string
  directly rather than constructing a `Path` for each syscall. Individually
  queried file proofs retain their own caller brackets.
- Owned mapping audits resolve each repeated pathname only once within one
  freshly read snapshot, retaining every address/inode/device segment. The
  dictionary is discarded before the next admission. Native and factory
  symbol ownership are checked in the same parsed snapshot before returning
  the owner's validated receipt. No rows/proofs survive for reuse.
- Owned kernel/GEMM argument packing performs only address/lifetime/session
  validation, not recursive runtime admission for every operand. Complete
  proof/current-device validation occurs immediately before native compute.
  Public `DeviceArray.pointer` and the legacy adapter retain their checks.
- Launch argument iteration, scalar conversion and descriptor conversion occur
  before the final native admission. Buffer lifetimes and packed addresses are
  rechecked afterward. This also closes a reproduced gap: a scalar's
  `__index__` could close a leased descriptor after the old last admission,
  yet the old code still enqueued. The new real-lease regression requires
  exit86 before any native-call spy.

No arithmetic, GEMM descriptors, kernel source, precision, cache budgets,
workspace ownership or opaque reserve changes are included. Revocation remains
fail-stop; cleanup cannot enter a revoked library.

## Current verification

The full CPU suites passed **350 +28 +109 =487 tests**, with no skips.
Additional coverage includes argument ABI/holder lifetimes, late identity/device
revocation, freed/changed buffers, wrong-session inputs, fresh mapping resolution
after alias replacement, and every requested factory symbol's mapped ownership.
The shader/native-call spies are CPU doubles, not GPU evidence.

Safety review found an exception-path gap in the first candidate: a converter
could revoke a proof and then throw before the successful-packing check.
New real-lease scalar/iterator/descriptor regressions reproduced caller cleanup
unwinding with an invalid proof. Owned launch now checks identity on preparation
and enqueue exceptions before unwinding. Healthy conversion errors retain the
original exception object and do not invalidate an otherwise healthy session.

An isolated 17-file real-kernel-lease microdiagnostic compared the gated base
against the working checker in separate CPU-only processes on the local host:
base0.585–0.591 ms/check; candidate0.225–0.234 ms/check. Both retained17 full
hashes. These fixture observations do not predict a whole-call or H200 gain.

Before accepting this candidate: complete safety review, normally rebuild a
separate whole Cachy prefix, freeze a fresh CPU qualification/executable copy,
then run the same bounded construction and ten-process numerical gate.
All first-step runtime configurations are expired; no scientific overlay or
revival of old temporary dependencies is permitted.
