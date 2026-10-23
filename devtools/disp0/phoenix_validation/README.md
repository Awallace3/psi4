# Fresh Phoenix native build and CPU validation

This is a new build of the current scientific source, not recovery or
relabeling of any previous failed Psi4 installation. It does not submit
protein-size F-SAPT or certify GPU correctness.

The dependency environment and installed Einsums2/Waggle pair from allocation
14254131 are read-only build/runtime inputs. That allocation's original Psi4
installation and failed status remain unchanged. The validator requires the
special Einsums commit `06404fa69ce9346443d7d4f19a250f6db9f0f62c`, one native
world, and a newly built, versioned, cuEST-enabled Psi4 stage. It never imports
the old Psi4 core or overlays Python scientific files.

The CPU job:

1. Verifies clean exact-pin source with complete history and annotated v1.11.
2. Inventories CUDA13.3 and creates a fresh private developer-link locator.
3. Pins canonical cuEST native bytes and the complete 59-header aggregate.
4. Configures and builds a whole new Psi4 stage with native cuEST enabled,
   native Einsums disabled (Python E2 is independent), CPU LibXC, and portable
   x86-64-v3 CPU code. HF validation does not need GPU LibXC.
5. Runs the standalone CPU/mock/serial regressions, then imports the new stage
   before E2, checks identities/maps, and executes a native CPU RHF smoke test.

No environment/package installation, GPU allocation/context, old-root changes,
automatic retry, or GPU/scientific fallback is performed. CUDA_VISIBLE_DEVICES
is empty throughout. A successful result is **CPU_BUILD_VALIDATION_ONLY**.
Missing dependencies or incompatible ELF maps fail closed.

`submit.py` supplies every resource option explicitly, tests the route first,
records submission provenance, and permits only one submission per fresh run
root. The job uses 8 CPUs, 48 GiB, 3 hours, and free embers. Build parallelism
is six; the extra CPUs accommodate orchestration/linking. Watch the captured
job ID with the user-level Phoenix skill's `watch_jobs.sh`.

The two existing uncommitted fixture edits in the development checkout are
not part of this pinned build and are neither committed nor discarded.
