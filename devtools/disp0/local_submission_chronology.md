# Local-only audit: exec session77157

**Status remains UNKNOWN; approval RESERVED; NEVER REPLAY.**
No remote command, original script or submission was executed in this audit.
The findings below describe preserved local records, not new cluster proof.

## Original evidence

Journal:
`/home/awallace43/.pi/agent/sessions/--home-awallace43-gits-psi4__worktrees-saptdft-cuest-disp0--/2026-10-19T15-47-26-425Z_01a154d8-a459-768d-83bd-c57c6a5dc42c.jsonl`.
Line numbers refer to that existing append-only journal.

| Journal UTC,2026-10-20 | Event / line | Observation |
|---|---|---|
|13:41:56.699|abb96a05 /4185|`exec_command` requests the bounded interactive connection.|
|13:41:57.810|d831848a /4186|Tool returns running session77157; measured tool wait1.000335480s.|
|13:42:19.851|45db584a /4188|First `write_stdin`:2816 requested UTF-8 bytes of preflight commands.|
|13:42:20.954|7bb632bd /4189|PTY echo plus actual remote preflight/source-binding output.|
|13:42:57.311|2200b723 /4195|Tool returns route/quota output and `PREFLIGHT_GATE_DATA_READY_NO_SUBMISSION`, session still running.|
|13:43:47.321|3bc84940 /4197|Second `write_stdin`:2618 requested UTF-8 bytes,35 script lines.|
|13:43:47.323|7f6a799a /4198|Tool returns exit124 in0.000128127992s, only `phx: timed out after100s...`; no echoed second phase or submission acknowledgement.|

The second native function-call item's `create_time` is
13:43:47.146772Z. This is function-call metadata, not a PTY/SSH write
acknowledgement. The assistant message also has an earlier
`message.timestamp` of13:42:57.573Z; that is not a byte-delivery event and
must not be substituted for the journaled tool-call time.

## Bounds and their limitations

The original command requests:

```text
timeout --signal=TERM --kill-after=2s 108s
  env PHX_TIMEOUT=100 phx
    'timeout --signal=TERM --kill-after=3s 95s bash -s'
```

Journaled exec-call → second-write-call interval: **110.622s**.
Using the exec-call journal time as a nominal start:

* 100s →13:43:36.699Z; second call is10.622s later.
* 108s →13:43:44.699Z; second call is2.622s later.
* 110s including outer kill grace →13:43:46.699Z; second call is0.622s later.

These are configured/nominal bounds, not an independently captured kernel
process-start/timeout-fire trace. The immediate exited124 response is
consistent with a late call to an already-finished session. The inspected
tool-result details contain only `chunkId`, `exitCode`, `originalTokenCount`,
and `wallTimeSeconds`: **no bytes-written count or transport acknowledgement**.
Thus the record does not establish that the second phase reached the remote
shell, nor independently certify zero bytes delivered. No pre-deadline
submission-phase dispatch is recorded.

## Exact requested payloads

These hashes identify argument strings, NOT confirmed transport deliveries:

* Preflight2816bytes:
`cf8a57092ff70dcb300aedd076c6fecc1015368f4b6d31b794c8d3bf29550ad3`.
* Submission phase2618bytes:
`33ed577ce46b2034979c2533a77fde79427fbef92572336a8a956e2f48dee4bf`.

The first payload has only `sbatch --test-only` (line35), not an actual
submission. In the second payload, inherited `set -e` gives this ordering:

1. Line1 writes `preflight-review.txt`.
2. Line2 creates `SUBMISSION_INVOKED`.
3. Line3 temporarily disables exit-on-error.
4. Line4 invokes actual `sbatch --parsable`, opening
   `sbatch.stdout`/`sbatch.stderr` before executing it.
5. Lines12/14/17–20 write/fsync raw responses, submission JSON and, only after
   exit0 plus valid ID parsing, `job-id.txt` and the receipt directory.
6. Only after successful receipt code and reading that ID does line26 issue
   the single own-job controller query.

No original command removes these records. However, the pre-submission
`mkdir`/review write have **no explicit pre-RPC fsync barrier** in this script.
Their subsequent absence is therefore supporting evidence, not a certified
durable negative proof of invocation.

The already-authorized remote reconciliation at13:46:40Z found all
submission-phase markers/receipts absent; own queue/accounting returned
exit0/empty. Those existing observations do not conclusively prove absence,
and this local audit does not relax UNKNOWN or authorize another query/job.

## Raw journal-line fingerprints

SHA256 over each original JSONL line including its newline:

* 4185:`47a1bdbe8eca765cd69b203395448654a8c10cab4bb998ce5e9b59599668112e`
* 4188:`1088a5a3d70925777451a51f1b99c682d94cd48c911308004faf24fbc041fdb1`
* 4197:`c4b00f866ac3304c00ebf188aa4edbf9bbdccac9cda8b91ec60f90194a787ad2`
* 4198:`2a13cbf59748057c5141c57ee480769028bbedce5f58034c63bb747f6b77072a`
