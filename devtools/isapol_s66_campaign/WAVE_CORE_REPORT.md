# Wave core implementation status

Interface: `95f5c69784399c5167b6b8384b7bb362f5fbc0ad`, unchanged.

Initial integration commit includes pure whole-wave reduction, measured closed
campaign loading, wave-only record-body identity snapshots, separate one-shot
registration and read-only transports, shared serial/wave intent collision,
bounded held/released controller, and new metadata-only ops wrapper/tests.
Production activation remains disabled in `bank_wave_controller.py`.

Initial local verification:

* `/home/awallace43/miniconda3/envs/p4qca/bin/python -m pytest -q
  devtools/isapol_s66_campaign/test_bank_wave.py`: **57 passed**.
* Explicit p4qca `py_compile` for new wave Python modules: passed.
* `bash -n devtools/isapol_s66_campaign/bank_wave_ops.sbatch`: passed.

At this integration checkpoint, real runtime integration, SDK mockHTTP
failure-path coverage, full wave publication tests and full regression are
still pending. This commit is for concurrent integration/review, **not**
deployment approval or a scientific claim. No credentials, network, Phoenix,
SLURM jobs or quantum calculations have been accessed/executed.

Runtime agreed extra helper:
`bank_wave_transport.snapshot_dataset(client, *, binding)` enriches declared
wave records only (and previously wave-accepted records) with
`wave_identity_sha256`; serial prefix representations stay unchanged.
The stable full-record identity excludes the explicit result/lifecycle field
set in `LIFECYCLE_FIELDS`, retaining unknown/user/immutable fields.
`validate_capture` returns `snapshot_sha256`, `preserved_records`,
`peer_statuses`; `require_complete=True` is whole-wave-finalizer only.

## Integration / review followups

* Core initial implementation: `36fd620517808f9974ee957a9ff2c98008936c34`.
* Runtime code `0fcebb7f998efc3f65e0f72cf0b4869072dcbaa5` was explicitly
  authorized for integration and cherry-picked as `5332de1bfc`.
* Alias correction: `41cd3434d4bb2b64d23f3b1061a1b1193cc693ce`.
* Preserve nonnative extras: `9b106dd2e5a3e32be0324e9ac028851617f80580`.

The reviewed full-record lifecycle set uses actual `by_alias=True` keys.
Only the exact `extras['psi4:native_monomer']` result slot may vary; other
extras/user metadata and sibling-prefix keys remain bound. Registration
scope deep-copies caller metadata before typed-model payload construction.

Additional review fixes bind the accepted checkpoint's actual latest wave
link hashes to this controller's plan/registration/permit/barrier/predecessor,
and admit eager metadata-job startup using the fsynced **pre-submission**
launch intent. Ops writes its actual allocation identity; controller still
requires equality with the eventual sbatch response before release permit or
acceptance. **No additional ops hold/release actions were introduced.**

Expanded tests cover actual capture chains and failed/lost registration,
shared serial/wave reservations, SDK one-shot writes/login/refresh,
preflight-before-registration and permit/intent revalidation, all-member
accounting, complete audited reduction (+8 once), double-live/proof-before-
checkpoint ordering, failed finalizer/live disagreement, two independently
valid successor cross-pair rejection, and eager ops start before sbatch return.
The transitive source inventory includes `native_controller.py`.

Intermediate results (before final frozen-suite rerun):

* Combined initial wave + compatible serial/SDK focus: **286 passed**.
* Expanded pure reduction/publication: **83 passed**.
* Review-blocker focused rerun (cross-pair, eager startup, finalizer acceptance,
  typed-payload immutability, actual Portal model lifecycle/extras):
  **13 passed, 111 deselected**.

Full final suites and runtime's expanded actual-SDK integration are pending
at this followup checkpoint. Earlier failed runs exposed fixture geometry/
permissions and a caller/payload aliasing bug; these were fixed rather than
relaxing guards. No production or scientific claim.
