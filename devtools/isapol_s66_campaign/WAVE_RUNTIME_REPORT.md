# Registered-wave runtime handoff

## Scope

Implements the explicit `eligible-wave` / `registered-wave/1` runtime branch
against interface commit `95f5c69784399c5167b6b8384b7bb362f5fbc0ad`
(cherry-picked here as `1bd18df871`).

Owned changes: worker/science/auditor Python and sbatch wrappers,
`grid400_bank_execution.py`, `single_record_cb63305.py`,
`bank_wave_runtime.py`, and three runtime test modules. No runtime-authored core/controller,
serial campaign/audit/transport/store, scientific/build, resource forecast,
`runtime_audit.py`, `validate_numeric.py`, or `trace_gate.py` edits.

## Behavior

- Workers reconstruct cases before registration and emit the distinct wave
  preflight status and plan/context/predecessor bindings.
- SCI uses the core's permit-validated read-only dataset client and consumes
  its actual registered job/tag/record. It cannot enter serial append/submit
  branches. Existing manager, scientific setup and delivery gates remain.
- Wave snapshots use `bank_wave_transport.snapshot_dataset(client,
  binding=binding)`, including the core's stable full-record identity evidence.
  Delivery requires its own completed record, while peers may still compute.
  Registration observations are not copied into SCI payloads.
- Member audits require the verified whole-SCI barrier; preserve numeric,
  decoded-FD, finite CLI role, JSON identity, source/archive and exact delivery
  gates; emit `PASS_SELECTED_WAVE_CASE_SINGLETON_EVIDENCE`. Both singleton
  advancement flags are rejected before operational reads.
- Exact closed nested wave case/run paths and controller wrapper argument
  grammar are explicit. Actual job-ID namespaces and private scratch remain.
- Wave JSON payload publication uses exclusive no-follow/fsynced bank-store
  writes. Source inventories include every new helper/wrapper.

## Local verification before core integration

All commands use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-B -m pytest`,
`--noconftest -c /dev/null -p no:cacheprovider -q`.

- `/home/awallace43/miniconda3/envs/p4_ci/bin/python`:
  existing focused runtime suites: **236 passed**.
- Same interpreter, two new contract/routing modules: **83 passed** at the
  initial implementation checkpoint.
- Same interpreter, `devtools/isapol_s66_campaign/test_*.py`:
  **1515 passed** (1432 baseline + 83 new), before final narrowly added
  no-follow publication/own-member checks.
- `/home/awallace43/miniconda3/envs/p4qca/bin/python` (QCPortal 0.63),
  `devtools/isapol_s66_campaign/*_tests.py` plus the two new modules:
  **282 passed** before those final checks.
- Final owned revision, both new modules: **85 passed**. Focused legacy
  plus new modules immediately before the last own-member test addition:
  **320 passed**. All three sbatch wrappers passed `bash -n`.
- `git diff --check` passed; the three protected numeric/trace-policy files
  are byte-unchanged against the original base.

Contract tests deliberately replace unavailable core callbacks at explicit
seams. These are not evidence that the real wave core integrates.

## Real integrated verification and review followup

Integrated dependencies (authored by core, no runtime edits):

- Core `36fd620517808f9974ee957a9ff2c98008936c34`, cherry-picked here as
  `24989592dc`.
- Core alias fix `41cd3434d4bb2b64d23f3b1061a1b1193cc693ce`,
  cherry-picked here as `0feb80fd40`.
- Core exact-extras fix `9b106dd2e5a3e32be0324e9ac028851617f80580`,
  cherry-picked here as `27a4eb48b8`.
- Core publication-link/startup-intent/guard review fixes
  `11235cb47c5dc84619ed570206b4af3d963a1da0`,
  cherry-picked here as `3c5d32496b`.

Read-only review correction: explicitly union
`bank_campaign.ADDITIONAL_SOURCES` with wave-specific source names rather
than relying on core returning a superset. The regression provides minimal
wave-only campaign names and checks the complete common closure, including
`grid400_bank_execution.py`. Also explicitly commit `native_controller.py`,
which the core's positive-accounting verifier imports.

`wave_runtime_integration_tests.py` runs with pinned p4qca and the actual
core/runtime/transport implementations, actual `SinglepointRecord` bodies,
the real read-only factory, and mock HTTP. Local synthetic fixture authority
replaces production-only path/pool reads, while repeatedly validating the
entire synthetic plan, serial predecessor/proofs, registration, permit,
captures and barrier using the real pure reducers.

Results:

- Actual SDK/body integration: **16 passed**. Covers registered own record,
  own completion with waiting peers, completed peer before own claim, exact
  stable-body/spec/history/prefix preservation, wrong actual job/tag/record/
  member/worker bytes/permit, actual whole-wave barrier, independent
  preservation and failed-peer accounting, no serial report acceptance,
  and session allocation/runtime/protocol binding.
- The real-model tests exposed the core's underscore-versus-alias lifecycle
  hash bug; core corrected it in `41cd3434`, and actual lifecycle transitions
  now pass without weakening immutable identity checks.
- Combined p4_ci campaign suite before the final native-controller inventory
  addition: **1575 passed**.
- Repetition including that inventory addition: **1575 passed** with p4_ci,
  **358 passed** with pinned p4qca (all explicit `*_tests.py`, core wave,
  runtime and routing modules).
- Expanded real-model gates exposed the whole-extras mask: **17 passed,
  3 failed** on unrelated metadata addition/change/removal before the core
  exact-slot fix. A sibling-prefix negative case was then added as well.
  The SDK waiting/running/complete test verifies three distinct observed
  body hashes and one stable identity, not synthetic identity constants.
  Preexisting user extras must survive unchanged; only the exact
  `psi4:native_monomer` result slot may change.
- After `9b106dd2`: expanded actual-model/runtime tests **21 passed**,
  including preexisting metadata, waiting/running/complete transitions,
  unrelated extras add/change/remove and native sibling-prefix rejection.

Repetition with the actual extras fix and all expanded runtime tests, before
the remaining core review fixes:

- p4_ci `test_*.py`: **1575 passed** (54.32s).
- p4qca `*_tests.py` plus `test_bank_wave.py`,
  `test_bank_wave_runtime.py`, `test_bank_wave_routing.py`:
  **363 passed** (28.22s), including all **21** real SDK/runtime integration
  cases.
- All three owned wrappers pass `bash -n`; `git diff --check` passes; the
  protected numerical/trace files remain unchanged against the original base.

Final review dependency verification through `11235cb4`:

- Pinned p4qca: **443 passed** (97.40s), running every `*_tests.py` plus
  `test_bank_wave.py`, `test_bank_wave_campaign.py`,
  `test_bank_wave_runtime.py`, `test_bank_wave_routing.py`. This includes the
  real **21** SDK/runtime cases, expanded core reduction/publication,
  eager-start authority, real HTTP/auth, and source-union regressions.
- p4_ci full `test_*.py` campaign repetition: **1622 passed**, five
  dependency deprecation warnings (97.23s).

All requested runtime/source-inventory integration checks passed against
the imported review fixes. The owned runtime followup is frozen for
coordinator integration. The controller remains production-disabled;
coordinator activation review remains a separate gate. No production
readiness or scientific acceptance is claimed.

No network, credentials, dependency installation, quantum calculations,
cluster/job actions, merge, or push was performed.
