# Second performance step: remove nested admission work

**Small-water construction, numerical and strict before/after gates passed.**
Whole-call diagnostics now take42.5–44.4 seconds instead of155.5–158.6.
This is RTX3090 host-overhead debugging, not an H200 speedup measurement.
Performance is still not considered solved.

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

## Completed coherent validation

Science/helper pin: `3ea43ee66ebb2920138c9dfc89c1951d6fbb3673`.
Whole prefix: `install-owned-3ea43ee66e` in the private performance root.
The target build passed33 real lease tests and9 native-packing CPU-spy tests;
CPU RHF qualification retained `-1.1167593073781577 Eh`.
The original baseline and previous whole candidate prefixes remain intact.

The first combined launcher had a shell-quoting error and stopped **before**
construction or any GPU calculation. Its log is retained. A separate Python
orchestration input launched the frozen helpers with fresh attempt names; it
did not replace or overlay any scientific module.

`construction-hot-water-05` passed both JIT1/4 processes:
construction+cleanup9.695/9.906 s, versus33.393/34.316 s in the first owned
candidate. Whole children took16.524/16.647 s. Tracked device peaks remain
33,757,184 bytes, with zero live bytes/native owners after complete teardown.

`hot-full-water-gate-02` passed all ten fresh CPU/GPU/oracle/repeat processes.
All four GPU cases executed real native/device work and completed cleanup.

| Diagnostic whole energy (s) | First owned candidate | Hot-path candidate |
|---|---:|---:|
| Unscaled | 156.864 | 43.728 |
| Scaled | 158.600 | 42.472 |
| Displaced | 157.259 | 44.433 |
| Fresh scaled repeat | 155.490 | 43.482 |

All retain17 hashes and6,236 mapped-byte receipts. Live checks decrease from
about232,000 to163,000; every remaining full snapshot still checks every file.
Tracked allocation peaks are unchanged:88,794,624 bytes for unscaled/scaled/
repeat and88,793,600 for displaced geometry. The opaque reserve did not change.

Independent same-gauge CPU energy-matrix/pair maximum errors were
5.198e-13,5.773e-12 and1.095e-11 Eh for unscaled/scaled/displaced geometry.
A separate strict before/after comparison checked all23 fields in each of
the four GPU cases, with the original1e-9 tolerances and orbital-label/sign
matching only. Maximum energy errors:

| Before/after case | Maximum absolute error (Eh) |
|---|---:|
| Unscaled | 3.986e-14 |
| Scaled | 3.571e-12 |
| Displaced | 1.571e-11 |
| Repeat | 8.233e-12 |

The comparison also validates whole installed Python inventories, core hashes
against their respective sealed proofs, numerical options/dimensions, native
fingerprints/policy, CUDA library hashes, device/driver/PTX, output artifact
hashes, actual GPU work and cleanup. Only the four intended scientific Python
files and generated metadata differ. Historical configurations are not revived
or treated as reusable runtime proofs.

Under `~/gits/psi4-disp0-perf-cachy/runs/`:

| Artifact | SHA256 |
|---|---|
| `hot-owned-cpu-01/runtime-config.json` | `4f3fbda55b461b75550094466d03392570270afc57ee620f483a915a29ea8ef1` |
| `construction-hot-water-05/comparison.json` | `5c87e90e2f50305f174947924fe218b62b1a059e2772562ccbadc82efeab671b` |
| `hot-full-water-gate-02/gate-result.json` | `70e542195d8ac9fd89fe8344b3b7437ef91ec5b1212b26611ccffce1c9598e5c` |
| `hot-before-after-01.json` | `f3c9284a484a796832c81942ed3e5b24705ef611317efecf1fc6c61e7d851383` |

These remain small restricted-water results, with zero accepted H200 timing
samples and no Phoenix job. The remaining163,000 checks warrant caller-level
profiling before another optimization.

## Qualified caller profile and next experiment

`hot-profile-water-01` profiles one qualified scaled GPU case without changing
or overlaying scientific code. Its output passed the existing strict fresh-repeat
comparison against `hot-full-water-gate-02/gpu-scaled`. Profiled durations are
**not** accepted timing samples and must not be added together:

| Profiled scope | Calls | Inclusive seconds |
|---|---:|---:|
| `check_live` | 163,164 | 22.985 |
| `mapped_rows` | 734 | 20.593 |
| `CUDASession.check` | 64,540 | 18.487 |
| `DeviceMath.require` | 15,963 | 4.569 |

This confirms that simply trimming buffer validation cannot solve the remaining
cost. The profile also reports11,808 memory checks,23,574 allocation-free guards
and216,968 pathname resolutions. Native/GPU wait time is not separately isolated
by this Python profile.

The user selected a **CPU-first kernel-sealed snapshot prototype** to test a
stronger immutability contract that need not poll filesystem metadata on every
kernel operation. Primary source is kept separately on
`prototype/cuest-sealed-runtime` (CPU probe code`689388fd63`), under
`devtools/disp0/cachy/prototypes/`. The question is actual Linux/ELF loader
compatibility, not an HTML/state simulation. The toy `$ORIGIN`/seal probe and
the pinned vendor-library/CPU-only NVRTC probe both passed on Cachy. All17 vendor
images have irreversible seal mask63, all nine preloads succeeded, and25 cuEST
native/factory symbols map to the correct sealed SDK inode/device. NVRTC13.3
compiled PTX with its cache disabled, without loading/executing a GPU kernel.
Result:`runs/sealed-prototype-cpu-02/result.json`, SHA256
`0ddca4f8e23a3cbf69aa5936e393b9b9602612befe6f94cc9f37ea9c185d1923`.
At that CPU-only stage, Python CUDA-binding lazy loading and real cuEST
context/JIT were still untested. No production or GPU qualification was implied.

The separately approved bounded GPU compatibility probe subsequently passed
at prototype code`57c6af5366`. Existing CUDA13.3 bindings lazily loaded the
sealed toolkit, a one-element FP64 kernel returned3.0 from2.0, and one native
cuEST0.2.50 FP64 context was created/destroyed with one compiler thread and a
private cache HOME. All explicit resource cleanup completed; no second/unsealed
toolkit appeared at the recorded mapping checkpoints.
Artifact:`runs/sealed-prototype-gpu-01/result.json`, SHA256
`0c20aa4b9aa07d0189683a8edaa1552b07b1be1e39d9458da1f220cb91e1d42b`.
This is **only loader compatibility**, not a production owner, native basis/DFMO
JIT gate, whole-Psi4 numerical/performance result or zero-cache-memory claim.
The qualified implementation remains unchanged.

Independent probe inspection found two defects, subsequently fixed on the
prototype branch: include cuDNN/CUPTI in foreign-toolkit rejection, and resolve
relative outputs before changing the worker directory. CPU negatives reject
real unsealed mappings for both families. Corrected code`6997a49701` passed
`sealed-prototype-gpu-02`, including a deliberately relative output path.
Its current result SHA256 is
`cced9283e58983183b1628a6e8a3ebcb5a922534a078a74ef08be11afc7085d3`;
this supersedes attempt01's narrower classifier. Kernel/context/readbacks and
all explicit resource cleanup passed again. Production integration and the
whole-water numerical/performance gate remain the next steps.

The [sealed-runtime integration](SEALED_RUNTIME.md) now has a separate adapter
and normally installed pre-native support package under CPU validation. Review
reproduced a constructor-time hole in the first alias-based candidate; its
substituted-dependency regression was run red and then green after switching
first loads and pinned dependencies to sealed descriptors/explicit SONAMEs.
Diagnostic aliases are excluded from the loader search path. A separate Cachy
CPU probe passed all nine exact descriptor-only vendor preloads and NVRTC
compilation. Neither the candidate nor a further whole-water speedup is
GPU-qualified yet; the earlier alias-prototype PASS is not treated as a
production first-load admission proof.
