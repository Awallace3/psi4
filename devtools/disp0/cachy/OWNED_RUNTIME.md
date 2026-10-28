# Opt-in owned-runtime construction candidate

**Not GPU-qualified.** The measured baseline and its timing remain unchanged.
This candidate addresses repeated identity-admission work, not numerical kernels,
JIT policy, cache budgets, context sharing or opaque memory reserves.

The default remains `fresh-bytes-v1`: every existing mapped-byte audit hashes
the actual files afresh. Select `process-owned-kernel-read-leases-v1` explicitly
when qualifying a **dedicated worker**. The contract and failure semantics are
described in [the proposed ADR](../docs/adr/0001-owned-runtime-byte-admission.md).

The owned adapter hashes each pinned cuEST/CUDA library once under a retained
Linux read lease. Explicit loads use the retained descriptor. Subsequent audits
still verify live leases, notification ownership, process/thread, file/path
identity, actual mappings, symbol ownership and ABI. It is not an mtime cache.
Revocation terminates that worker without native/Python finalizers: exit29 for
kernel SIGIO, exit86 for a failed live check. There is no retry, downgrade,
proof renewal or in-process transition from an existing legacy session.
The authenticated bootstrap must already map the complete pinned closure
through canonical paths; descriptor loading is reuse only, not first-load
dependency discovery (`$ORIGIN` would otherwise change). `RTLD_NOLOAD` also
requires actual dynamic-linker residence: read-only file mappings alone are
insufficient. Revocation writes
nothing to stderr, so even a full logging pipe cannot prevent retirement.

## CPU verification

The current implementation passed **336** root tests, **28** Cachy helper tests
and **109** validation-helper tests, with no skips:

```bash
python3 -B -m unittest discover -s devtools/disp0 -p 'test_*.py'
python3 -B -m unittest discover -s devtools/disp0/cachy -p 'test_*.py'
python3 -B -m unittest discover -s devtools/disp0/phoenix_validation -p 'test_*.py'
```

These include real read leases, an existing writable mapping, a blocking writer,
an actual forced lease break under the existing kernel timeout, admission
rollback faults, descriptor/signal tampering, fork/thread misuse, actual ELF
loading through a leased descriptor despite a retargeted alias, and suppression
of native-style cleanup after revocation. Review follow-ups additionally cover
`$ORIGIN` first loads and fake mmap-only residence, the actual cuSOLVER loading
call site, cleanup after a consumed break notification, a full stderr pipe,
session selection during hashing and a fork while the state lock is held.
No GPU library/device is needed.
The tests never modify the system's lease timeout.
Signal-suppression fixtures explicitly use one OpenMP/BLAS thread: masking the
main thread alone cannot suppress process-directed SIGIO in native workers.

An earlier red regression demonstrated why a Python signal handler was unsafe:
an inner `finally` ran before Python dispatched SIGIO. The corrected native
handler invokes libc's async-signal-safe `_exit` directly; the regression now
passes. These tests do not establish GPU correctness or a performance gain.

## Remaining qualification

Preserve the measured `install/` baseline. `build_owned_candidate.sh ROOT PIN`
uses the private configured build serially and normally installs the **whole**
candidate into `install-owned-<pin>`. Run it only inside the established
4-CPU/12-GiB/no-swap/128-task scope. No module overlays are permitted.

Use a clean, exact candidate pin and a fresh qualifier output:

```bash
python -B source/devtools/disp0/cachy/qualify_construction_root.py --help
```

The qualifier accepts `--stage` for that whole candidate prefix and an explicit
`--identity-contract`. It must freeze a fresh CPU build/runtime proof and
temporary executable copies; the earlier baseline configurations have expired.
Then run the same water construction schedule in fresh-byte and owned modes,
retain failures and complete cleanup evidence, and run numerical/before-after
GPU gates before changing a default or moving to Phoenix.
