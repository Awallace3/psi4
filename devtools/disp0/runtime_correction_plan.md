# Runtime correction handoff — planning only, no new allocation approved

## Established blockers

Original job14196330 remains FAILED1:0. Artifact recipe14e copied the complete
same-job Python/native E2 and Psi4 stage unchanged into a new private prefix:
190,505,547bytes in59.5s. That byte/ELF success is **not runtime readiness**.

The new-prefix metadata proof failed before tiny API tests:

1. E2 Python `_core` has DT_RPATH pointing first to the old build-tree native
   library. This takes precedence over the intended LD_LIBRARY_PATH.
2. The original E2 CMake cache selected siteCUDA12.6 cuSPARSE and nvJitLink.
   Its declared CUDA13 compiler/root did not prevent component fallback.
   A read-only loader trace resolves those CUDA12.6 libraries alongside CUDA13.
3. The configured explicit CUDA13 prefix lacks **unversioned development
   links** `libcusparse.so` and `libnvJitLink.so`. This explains why accepting
   only nvcc/CUDART/cuBLAS proof missed the incomplete toolkit contract.
4. Exact E2d17 builds a Python package but its binding-finalization CMake file
   has no Python install rules; the old `EINSUMS_INSTALL_PYMODDIR=naive`
   argument did not install it.

Evidence tasks: b9ec91c8f (failed mapping), b13607b43 (ELF), bf472f992
(loader closure), b2b62ea4e (original cache and toolkit links). No CUDA
numerical execution has occurred.

## Minimum revised contract to prepare and review

* Keep E2 scientific/API revisiond17 and Psi4 scalar scientific source equal
  to587. No algorithm rewrite, E1 downgrade, or new C++ dispersion code.
* Use a NEW complete private CUDA13 development prefix/environment, with
  verified headers, real SONAMEs and required development link names. Reuse
  installed read-only inputs only when their identities are proven coherent.
  Do not modify the shared CUDA13 prefix or coordinator environment.
* Explicitly inspect every used CUDA imported target at configure time;
  fail if it falls back to siteCUDA12.6 or is missing. Audit actual compiler
  include/link commands and the resulting ELF closure, not CUDA_ROOT names.
  Never alias a CUDA12 SONAME to a CUDA13 binary to silence missing dependencies.
* Provide reviewed tracked CMake packaging/install rules for the COMPLETE
  E2 Python package plus matching native library, with deliberate installed
  RPATH/RUNPATH semantics. A configuration-only deferred CMake hook may avoid
  changing E2 algorithm source, but its actual target/install compatibility
  must be verified before it becomes an executable recipe.
* Fresh source/configure/build/install roots and objects; no relink/repair
  of the failed job, binary/RPATH patch, old-core overlay, or preload workaround.
  Install the whole matching Psi4 stage, not a Python source overlay.
* Replace the now-ineligible fixed b3/8f/9e admission only through an explicitly
  reviewed **new build/artifact contract**. Record new hashes rather than
  pretending repaired binaries retain the approved tuple.
* Before GPU readiness, require all CUDA component paths/SONAMEs/hashes,
  exactly one installed E2 world, actual module paths, CPU layout/view/pool/
  replay tests and small signed physical checks on the new stage. Historical
  local26/64 comparisons remain historical support only.

## Resource/approval boundary

No executable corrected recipe or new immutable build pin is ready yet.
First prepare/review the configuration and packaging changes above. Only
then propose ONE fresh bounded CPU-only dependency/build job with an exact
source pin, resource/time/account/cost and route/headroom checks. A candidate
shape is8CPU/48G/up-to2h/freeembers/gts-cs207-chemx; **this is not approval
or a submission request**, and no paid CPU use is inferred from GPU approval.
No CPU scaling or benchmark campaign is included.

After successful matching-stage preparation and a new readiness checkpoint,
the existing unconsumed one-H200/4CPU/32G/30min gate may run its independent
native/eager/defaultCPU signed references plus actual eligible GPU work.
Only a passing gate can admit the separately approved timing allocation.
No build/repair in either GPU allocation; no automatic retry.

The2026-10-20 08:00 America/New_York speedup target is **not credible while
these runtime prerequisites remain blocked**. Do not consume a GPU allocation
to manufacture a result or compare against the parent's different PBE0-D4 work.
