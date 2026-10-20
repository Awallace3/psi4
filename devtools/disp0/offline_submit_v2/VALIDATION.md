# Offline model validation record

**Real submission: UNKNOWN. Real approval: RESERVED. All remote action STOPPED.**
This record is not a zero-delivery finding, live transport fix, deployment,
replacement approval, SDK admission or GPU result.

Only files in this NEW local directory were authored. Tests use newly created
local temporary directories and an in-memory scheduler. No original script,
attempt, receipt, remote source, Slurm command or network adapter was executed
or modified. No package/environment/build work occurred.

Command:

```bash
python3 -B -m unittest discover \
  -s devtools/disp0/offline_submit_v2 -p 'test_*.py' -v
```

## Red/green history

* Initial suite `bad21f556`:12 tests passed in5.580s. This did not cover
  reviewed-intent namespace binding or a different matching fake cluster.
* Additional local negative control demonstrated two fake RPCs under the
  same reviewed digest when only the external namespace argument changed.
  This was a model defect, not an invocation of any original attempt.
* Expanded red suite `b84550ce3`:17 tests in5.387s, exactly3 expected failures:
  different artifact parent, different attempt name, different cluster with
  matching fake reply. Direct fsync-syscall and query-proof failure tests
  already passed. The failed output is retained.
* Corrected suite `b2fa06935`:**17 tests passed in6.532s**, no skips.
  Canonical reviewed intent now binds parent/name; fixed OFFLINE cluster
  and exact numeric/boolean resource types fail closed.

The successful fixture trace contains49 checkpoints,8 file-fsync events,
8 record-directory-fsync events and the namespace-parent fsync. The crash
regression iterates every one of those49 checkpoints and refuses reentry.
AST checks passed; `git diff --name-only` was empty for existing tracked
files, with only this NEW directory untracked.

Durable logs are under local
`.pi/tasks/01a154d8-a459-768d-83bd-c57c6a5dc42c-4065461/TASK.output`.

## Covered properties and limitations

* Real local file and parent-directory fsync ordering before simulated RPC.
* Direct mocked `os.fsync` failures at namespace-parent and job-ID-directory
  sync, not only checkpoint exceptions.
* At every successful trace checkpoint, exception-injected interruption and
  refusal of any existing-namespace reentry without altering saved records.
* Failed preflights, reviewed-byte mismatch/tampering, namespace/path drift.
* Simulated acceptance with lost reply, nonzero/malformed/mismatched replies.
* Raw receipt and ID sync before simulated query; query/proof persistence
  failure preserves a durable positive fixture identity without retries.
* No interactive input, CLI or live execution adapter.

This does not prove actual power-loss behavior, remote filesystem semantics,
SSH byte delivery, real timeout bounds, authority validity, live quota/Slurm
compatibility or scientific correctness.

## Source review

Initial read-only review `dc09806ba6dd37eb544016ca4e2896f5d` confirmed the core
ordering by inspection and identified the cluster, namespace and syscall-test
gaps. These findings drove the red/green changes above.

Scoped follow-up `d4f1247e15d44d022723d3a8b755f1878` completed: no concrete
remaining scoped ordering blocker; all three earlier findings addressed.
Reviewer read the three source/design/test files and executed nothing; the
reported test results were supplied, not independently rerun by the review.
No remote query or actual attempt follows from any model/test/review result.

Frozen local source fingerprints:

* `protocol_model.py`:
  `5eb1a82052665ddec654ac3e10cff163ffda6d85462cc0b8eb092f34068b5f51`.
* `test_protocol_model.py`:
  `c5d30c38391352477c0365ea998fe549d0e2c0740b1103df6e5bd87e09badc9e`.
* `DESIGN.md`:
  `08a6d589938a2f7d39f364b5add392f5f6ad47188b70d850056c610b0bba22a1`.

AST checks passed; no existing tracked file differed from HEAD. Exactly four
new local files form this artifact. No live adapter, deployment, remote
publication or actual submission was performed.
