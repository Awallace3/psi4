# Registered-wave runtime handoff

## Scope

Implements the explicit `eligible-wave` / `registered-wave/1` runtime branch
against interface commit `95f5c69784399c5167b6b8384b7bb362f5fbc0ad`
(cherry-picked here as `1bd18df871`).

Owned changes: worker/science/auditor Python and sbatch wrappers,
`grid400_bank_execution.py`, `single_record_cb63305.py`,
`bank_wave_runtime.py`, and two runtime test modules. No core/controller,
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

## Required remaining gate

Await the coordinator's core implementation commit, integrate it, add/run
real combined metadata/mockHTTP coverage, and repeat focused/full compatible
suites. In particular, real immutable-peer mutation/lifecycle tests and
read-only HTTP/auth guards must exercise the actual core, not callbacks.
No production readiness or deployment claim is made before that gate.

No network, credentials, dependency installation, quantum calculations,
cluster/job actions, merge, or push was performed.
