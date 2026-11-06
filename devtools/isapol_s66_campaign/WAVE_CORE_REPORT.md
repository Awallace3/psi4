# Registered-wave core handoff

## Frozen implementation and commits

Interface `95f5c69784399c5167b6b8384b7bb362f5fbc0ad` is unchanged.
Final integrated **code** revision:
`d87c6c8562b6282d68b906614e5510b8cb4df75d`.

Core commits:

* `36fd620517808f9974ee957a9ff2c98008936c34`: initial implementation.
* `41cd3434d4bb2b64d23f3b1061a1b1193cc693ce`: actual SDK lifecycle aliases.
* `9b106dd2e5a3e32be0324e9ac028851617f80580`: preserve nonnative extras.
* `11235cb47c5dc84619ed570206b4af3d963a1da0`: review fixes and expanded tests.

Authorized runtime integrations: `0fcebb7f998efc3f65e0f72cf0b4869072dcbaa5`
as `5332de1bfc`, and `493b4420e145d148b0923bca01e653b6dc744234` as
`d87c6c8562`. Runtime-owned files were not independently edited.

Owned paths under `devtools/isapol_s66_campaign/`: `bank_wave.py`,
`bank_wave_campaign.py`, `bank_wave_transport.py`, `bank_wave_controller.py`,
`bank_wave_ops.sbatch`, private additions in `bank_store.py`,
`test_bank_wave.py`, `test_bank_wave_campaign.py`,
`bank_wave_transport_tests.py`, interface and this report.
Old public serial constructors/guards remain unchanged.

## Behavior / reviewed fixes

Bounded next-eligible waves, concurrent preflight barrier, HELD SCI submission,
single-writer sequential registration in the serial intent namespace,
exclusive plan-bound permit, one release attempt per new SCI job, complete
SCI accounting barrier, independent audits, and one whole-wave finalizer.
Generation advances once by the wave size, with actual serial anchor/proofs
and subsequent wave links. No invented singleton evidence or observations.

Actual full-record identities use serialized SDK aliases. Only the exact
`extras['psi4:native_monomer']` result slot may vary; preexisting/unknown
extras and sibling-prefix keys remain bound. Payload/scoped authority is
deep-copied. Acceptance checks actual latest-link hashes against the current
plan/registration/permit/barrier/predecessor, including a regression using two
independently valid successors. Ops startup validates the fsynced prewritten
launch intent, with actual response-ID equality required afterward before
permit/acceptance. **No additional ops hold/release actions were introduced.**

The extra agreed helper is
`bank_wave_transport.snapshot_dataset(client, *, binding)`.
Transitive operational source inventory includes `native_controller.py`.

## Final verification

Commands executed locally on frozen integrated code; metadata, mockHTTP and
pure arithmetic only:

```text
/home/awallace43/miniconda3/envs/p4_ci/bin/python -m pytest -q devtools/isapol_s66_campaign/test_*.py
1622 passed, 5 dependency deprecation warnings (90.61s)

/home/awallace43/miniconda3/envs/p4qca/bin/python -m pytest -q devtools/isapol_s66_campaign/*_tests.py devtools/isapol_s66_campaign/test_bank_wave*.py
443 passed (89.90s), including 21 actual-SDK/runtime integration tests.
```

All four wrappers pass `bash -n`; `git diff --check` passes.
`runtime_audit.py`, `validate_numeric.py`, and the decoded-FD `trace_gate.py`
are byte-unchanged against `cf834cacf0157b206eede0df1db049db954e667c`.
Earlier failing development tests exposed and corrected SDK aliasing,
whole-extras masking, typed-payload aliasing, and fixture mistakes; guards
were not relaxed. A final read-only safety check found no concrete remaining
blocker in core reduction/controller ordering; this is not deployment proof.

## Coordinator-authorized activation

After final recheck `dc7260c2` closed all four findings and the source-inventory
issue, the coordinator explicitly authorized **only**
`PRODUCTION_ENABLED=True`, necessary disabled-gate test-fixture adjustments,
this report, tests, and a safe local fast-forward into `ccasp-cleanup`.
The barrier test now explicitly injects the disabled flag; a new regression
verifies disabled CLI startup stops before operational helpers are called.
No other execution logic, scientific source, cap, or policy changed.

Post-activation repetition of the two commands above passed:
**1623 pure/regression tests** (95.38s; five dependency deprecation warnings)
and **444 pinned SDK/mockHTTP/wave tests** (93.01s), including the actual
SDK/runtime authority cases and the explicit disabled-CLI regression.
All four wrappers still pass `bash -n`; `git diff --check` passes.
No known runtime integration blocker remains. Operator `initialize` must
still positively validate original proofs/accounting and live dataset357;
the controller's subsequent stage headroom/preservation guards remain intact.
No scientific accuracy, actual production acceptance, or completed launch is
claimed. No network, credentials, Phoenix/SLURM actions, quantum calculations
or push were performed.

## First-wave operator command / paths

After deploying the exact activation commit into the clean remote checkout
below, run locally in a persistent session (`COMMIT` is the activation SHA):

```bash
SOURCE=/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-s66-grid400-wave001
COMMIT=$(git -C /home/awallace43/gits/psi4__worktrees/ccasp-wave-core rev-parse HEAD)
STATE=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-bank001/states/025.json
/home/awallace43/miniconda3/envs/p4qca/bin/python -I -B \
  /home/awallace43/gits/psi4__worktrees/ccasp-wave-core/devtools/isapol_s66_campaign/bank_wave_controller.py run \
  --source "$SOURCE" --commit "$COMMIT" --predecessor-state "$STATE"
```

Root: `bank_wave_campaign.WAVE_ROOT/wave-0025-0033`. It contains `plan.json`,
`cases/MID.json`, `members/MID/{worker,science,audit}/`, `registration.json`,
actual `registration/MID/` captures, `release-permit.json`,
`science-barrier.json`, `finalizer/payload/bank-checkpoint.json`,
`finalizer/payload/wave-advance.json`, and finally `accepted.json` (generation33).
Each stage run retains `intent.json`, `submission-response.json`,
`submission.json`, `logs/job-JOB.{out,err}`, `accounting.txt`, `verified.json`.
SCI also retains release-attempt/response receipts. The driver prints watcher
progress and final accepted-state JSON, not every submission receipt.
Any ambiguity stops; do not blindly rerun/clear a partial wave.
