# Guard-fixed whole-stage Cachy qualification

The standalone paired audit result is **not** a GPU correctness pass. The next
attempt normally builds and installs the complete current source into
`install/psi4-cuest-11`; it does not overlay installed Python on an older core.

## Bounded sequence

1. `coherent_stage11.py --root ROOT --pin EXACT_GIT_PIN` holds the common
   nonblocking `gpu0.lock`, rejects/monitors `gpu0.HOLD`, verifies the completed
   stage10 provenance, and preserves the whole old build as a verified read-only
   reflink copy. Only the literal owned `factory-source` checkout is advanced.
   Original `source`, installed stages, environment and receipts remain intact.
2. Replay the approved configuration into the same working cache, changing the
   new installation/proof destinations. Run normal **whole** CMake build and
   install, one build job, with fresh source/version/nested-install evidence.
   The generic existing validator verifies the whole install and native CPU H2
   PK/direct smoke. This publishes CPU-build qualification only.
3. `qualify_stage11.py --root ROOT --attempt FRESH_NAME` takes a new common
   lease, freezes actual completed build/dependency/helper identities, and runs
   the shared augmented-water gate. Each scientific child uses authenticated
   absolute native preloads, matching core first, then the real E2 loader with
   GPU-disabled upstream policy before lazy symbol lookup.
4. The numerical gate remains the same: unscaled, scaled, displaced geometry,
   fresh-process GPU repeat, independent CPU calculations and same-gauge
   matrix/IAO/JK oracles. All required fields, signed reductions, eleven output
   files, native/device kernel work and cleanup must pass. Block size remains5.
   Only `PASS_SMALL_CACHY_ONLY` qualifies this small local case, never timing or
   large-system admission.
5. **Only after step4 passes:** prepare a coherent guard-fixed Phoenix install
   and a bounded H200/inferno correctness gate, using fresh account selection and
   route/resource verification. Do not submit from an audit-only or CPU-build
   pass. No automatic paid retry or large timing campaign.

## Resource and lifetime policy

Both launches require an external user systemd scope: CPUQuota400%,
MemoryMax12GiB, MemorySwapMax0, TasksMax128, low priority. Build commands have
explicit phase deadlines (snapshot900s, configuration600s each, build/install
1800s each, smoke300s); the outer build deadline is4200s. The full small gate has
a3300s scientific deadline and600s per child; outer qualification is bounded.

The common flock descriptor is inherited through the runner and every child.
HOLD and termination signals abort owned work; supervision stops/reaps the
owned process group even if its original leader exits first. No foreign PID,
FDDS work, desktop context or games are enumerated or signalled. RTX identity
comes from the actual CUDA context UUID and query-gpu hardware, not a desktop
compute-process exclusion policy. H200 retains its separate allocation policy.

All destinations must be fresh. Failure artifacts are retained, not overwritten
or silently retried. A failed new attempt cannot publish a successful receipt.
