# Offline-only noninteractive submission protocol model

**The real attempt remains UNKNOWN, its approval RESERVED, and every remote
action STOPPED. Nothing here authorizes a replacement, proves zero delivery,
modifies the original attempt, or implements a live SSH/Slurm adapter.**

This NEW directory is an offline design/regression artifact. The module only
accepts an in-memory fake scheduler, OFFLINE fixture decision references and
the literal `OFFLINE_FAKE_RPC` command token. There is no CLI, network client,
shell/subprocess execution or deployment integration.

## Diagnosis and scope

The old two-phase interaction left a100s SSH session open while a model
reviewed preflight output and composed the next input. The journal records
the second request after the nominal deadline; it has no byte-delivery ACK.
Its invocation marker lacked an explicit pre-RPC directory-fsync barrier.
Original records and conclusions are unchanged.

Three red-capable offline checks target that pattern:

1. No human input or second transport phase: `run_model_once` performs the
   entire sequence synchronously with immutable already-reviewed inputs.
2. Omit/fail a pre-RPC file or directory fsync: simulated RPC must not begin.
3. Simulate accepted RPC with lost reply: preserve a durable dispatch record,
   stop unresolved, never query or replay automatically.

These are protocol-model checks, not a reproduction of actual SSH byte
delivery, remote NFS crash durability or a new scientific/runtime test.

## Future invocation contract — not execution authority

Before any future actual attempt, a human must explicitly handle the existing
ambiguity and make a NEW decision. An assigned owner turn and resource
authorization are separate required inputs; a string field is not authority.
The model tests their presence/digest binding using clearly fake references.

Review and freeze the complete intent BEFORE opening the one bounded
noninteractive connection. A future separately reviewed live implementation
must bind controller/helper version, exact source/batch/payload/policy hashes,
argv/resource/account/QoS/no-requeue settings, source/install paths and new
attempt namespace, fresh decision/ambiguity-handling reference and owner turn.
No floating branch, mutable stdin phase, later prompt or model review belongs
inside that invocation. Preflight policy must be deterministic/fail-closed
and completely reviewed first; this model does not invent a live quota parser.

The model binds the canonical artifact parent and exact attempt name into
the reviewed bytes and rejects either invocation argument changing. A second
namespace therefore cannot use the same reviewed digest. Constructing a
different digest does NOT grant a new approval; real authorization and global
approval accounting remain external and unimplemented here.

The future transport deadline is separate from the scheduler job wall limit.
All preflights, sync barriers, RPC and receipt capture must fit one outer
budget without model interaction. If a positive ID is durable but insufficient
time remains for a query, return the receipt and stop; do not stretch the
connection or open a second one. This offline model does not measure or prove
live transport deadlines.

## Durable ordering

Within one invocation:

1. Exclusively create a NEW private attempt directory. Fsync its existing
   parent so the attempt namespace itself is durable. Never adopt/reset an
   old namespace or delete a failed attempt.
2. Write exact reviewed canonical intent bytes with exclusive/no-follow
   creation. Make the file read-only, fsync the file, then fsync its parent
   directory. Reject digest mismatch, storage error or unsupported fsync.
3. Perform bounded noninteractive preflights; durably retain their proof.
   Revalidate stored intent against the reviewed digest.
4. Exclusively write a dispatch-intent record, **fsync the file AND its parent
   directory before the submission RPC**. This is a conservative potentially-
   dispatched seam: even a crash between this barrier and the actual RPC must
   never cause automatic replay.
5. Make at most one submission RPC. Persist raw stdout/stderr and return
   status/identity bindings with file+directory fsync. An exception, nonzero
   return or malformed/mismatched reply remains unresolved, never an automatic
   absence/retry finding.
6. For a validated positive reply, exclusively write job-ID metadata bound to
   the reviewed intent and raw response. Fsync file and directory. Only after
   these return successfully may one own-job controller query occur.
7. Query or query-proof persistence failure preserves the durable positive
   identity; it does not trigger another submit/query or job change.

Existing namespace always means **refuse**, not resume. Reconciliation is a
separate read-only protocol requiring its own authority. Empty accounting and
missing markers do not release the current real approval. There is no
"exactly-once scheduler commit" claim: the enforceable property is at most
one RPC attempt per immutable namespace, with conservative ambiguity after
the dispatch-intent barrier.

## Regression surface

Run LOCAL only, without plugin discovery or bytecode writes:

```bash
python3 -B -m unittest discover \
  -s devtools/disp0/offline_submit_v2 -p 'test_*.py' -v
```

Tests use new temporary local directories, actual local `os.fsync` calls,
ordered trace assertions and an in-memory scheduler. Injected crashes are
Python exceptions, not process kills/power cuts. Coverage includes:

* Legacy missing-directory-sync negative control.
* Reviewed-intent mismatch/tampering and policy drift.
* Parent/name binding and rejection of a different OFFLINE cluster.
* Failed preflights and pre/post-RPC fsync failures.
* Crash at every successful trace checkpoint; refusal of every reentry.
* Simulated acceptance with lost reply, malformed/nonzero/wrong-cluster replies.
* Durable-ID-before-query and query-failure identity preservation.
* Existing/path-escaping namespaces and absence of a live execution adapter.
* Actual mocked `os.fsync` failures at namespace-parent and ID-directory
  sync, plus query-proof persistence failure preserving a durable identity.

Passing these checks does not establish remote filesystem guarantees, actual
tool byte-delivery semantics, transport deadlines, live quota/controller
compatibility, SDK/E2 readiness or GPU/scientific correctness. Deployment,
new live adapters and actual attempts require separate review, explicit new
decisions after ambiguity handling and an owner-assigned turn.
