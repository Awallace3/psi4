---
status: proposed
---
# Own byte admission for an immutable runtime lifetime

The paired water control spends over 98.46% of plan-build time in per-object
identity admission, not demonstrated native/JIT work. The user selected an
explicit owned immutable runtime instead of retaining whole-library hashing
for every object. Keep the legacy fresh-byte path available; enable the new
contract only explicitly and after its numerical/cleanup gates pass.

The proposed module admits the complete cuEST/CUDA file set through same-FD
SHA256 checks under Linux kernel read leases. Cheap execution checks retain the
exact descriptor/path/inode/device, process/thread, mapping/symbol, ABI, version,
device and FP64 obligations. All library descriptors and SIGIO ownership last
until process exit because upstream C++ owners and loaded ELF objects may
outlive Python CUDA sessions. There is no proof close, renewal or transfer.
Explicit library loads use `/proc/self/fd/<leased-fd>`, not a second open of a
potentially retargeted pathname. Existing mapping/root/symbol checks still apply.
The complete pinned ELF closure must already be mapped by the authenticated
canonical-path bootstrap. Descriptor loading is reuse only: a first load changes
`$ORIGIN` dependency resolution, so missing mappings revoke the proof before
`dlopen`, rather than discovering dependencies through `/proc/self/fd`.
`RTLD_NOLOAD` enforces dynamic-linker residence; arbitrary read-only ELF mmaps
are not sufficient evidence of a previous load.

A read lease is **revocable**, not an unbreakable seal: attempted writes or
truncation notify SIGIO, and `F_GETLEASE` reports the pending target unlock even
while the writer is blocked. Either observation permanently invalidates the
proof. Adapted sessions check before/after synchronization; unadapted native
entrypoints require a synchronous publication guard. Published-proof revocation
**terminates the dedicated worker**, without Python/C++ finalizers: failed live
checks use exit86, and kernel SIGIO invokes libc's async-signal-safe `_exit(int)`
directly (exit29 on the supported Linux x86-64 ABI).
Raising a catchable exception is insufficient: discarded results, tracebacks or
an inner `finally` could otherwise enter a revoked native library. The caller
must explicitly acknowledge `dedicated_worker=True`; ordinary admission/input
failures before publication still roll back and raise normally. Unresolved or
interrupted rollback must attempt all remaining cleanup and terminate the worker
if descriptor/signal retirement cannot be proved.
Guard-detected retirement writes no stderr diagnostic: a full logging pipe must
not delay `_exit`. The dedicated worker's exit code is the revocation diagnostic.

Never publish a result after revocation. Do not respond by rehashing, retrying,
restoring the signal handler, unwinding native cleanup, or silently reverting to
the legacy path. A kernel-forced break during arbitrary in-flight native code
can still cause a process failure; the guarantee is admission/publication and
fail-closed worker retirement, not an unbreakable seal on arbitrary native code.

Python signal callbacks are also insufficient: a real regression demonstrated
an inner `finally` running before Python dispatched its SIGIO callback, even
when that callback called `os._exit`. The native `_exit(int)` function matches
the one-argument signal-handler signature and never enters Python or the GIL.
The actual kernel disposition is queried with libc `sigaction`, not merely
Python's cached `getsignal()` result, both before admission and during checks.

This adapter requires Linux x86-64, a dedicated main-thread worker,
unclaimed/unblocked SIGIO, current-user
owned regular files, and a filesystem supporting leases. Existing writers,
shared writable mappings, fork/thread misuse, lost descriptors, path
replacement, signal-handler replacement and unsupported leases must fail closed.
Both the kernel notification PID (`F_GETOWN`) and signal selection (`F_GETSIG`)
must remain as admitted. Terminal receipt checks catch revocation during
metadata/mapping traversal. A legacy session cannot be upgraded in place.
File leases are associated with open file descriptions: a fork must not issue
`F_UNLCK` on an inherited descriptor. Descriptors are close-on-exec.
Session selection and admission-in-progress publication share a lock; no legacy
session may begin while admission releases the GIL for hashing or filesystem IO.
The child-side fork hook resets only the Python state lock, so an inherited
locked mutex cannot hide the PID guard; it never unlocks a leased descriptor.

Rejected: filename/mtime hash caches (not an immutability proof), changing the
hash algorithm/pins, deleting identity checks, or merely moving repeated hashes
without defining ownership. This is an opt-in guard optimization, not permission
to change numerical policy, native ownership/workspace lifetimes or budgets.

Reference: [Linux F_SETLEASE/F_GETLEASE](https://man7.org/linux/man-pages/man2/F_SETLEASE.2const.html).
CPU-only probes confirmed read-lease acquisition, blocked write-open, SIGIO
notification and pending-unlock reporting on Cachy's tmpfs; no CUDA library was
loaded by those probes. The full CPU suites now pass 473 tests without skips;
real forced-break, first-load/mmap-only and cleanup regressions are included.
GPU gates remain pending; no performance gain is established.
