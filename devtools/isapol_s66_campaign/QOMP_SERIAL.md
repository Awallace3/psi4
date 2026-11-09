# QOMP audited serial advance (not activation)

This is the explicit `bank_controller.py --producer qomp` branch. It reuses
the existing sequential `run_chunk`, one-record transport, ordinary worker,
scientific launcher and independent auditor. **It is not a parallel wave.**
Default grid400 dispatch and its historical evidence remain separate.

## Authority and immutable roles

The initial accepted QOMP member is the external source6033 pilot: SCI
15371663 and independent audit15372459. Original25 results remain a separate
carryforward with original provenance, not QOMP records. The actual seed
report digest is
`43f42a12737f7b13af3ba7b4bd62b4b95820a4f56c443e8e5d12f117dae2270f`.
The dataset/record/specification and remaining members are **derived and
validated**, not selected through dataset357 globals or a new cohort.

Compiled science fb11, installed tests311, qualifying operations970, and the
historical descriptor-producing cba19 inventory are unchanged. The new
operational source must be recorded truthfully in all new worker/SCI/audit
receipts. Source6033 and all pilot receipts stay immutable. Historical build
15320675 remains FAILED.

Missing or unpinned actual pilot evidence is a hard refusal. No bootstrap
intent is reset and no completed records are fabricated. A new per-case
private intent prevents duplicate submission even under another job/context.
Every subsequent selection replays the accepted proof chain and requires its
immediate predecessor, live dataset agreement and a successful audit job.

## Parent-only setup

After review and Git staging of the frozen new HEAD:

```bash
P=/storage/project/r-cs207-0/awallace43
SOURCE="$P/gits/psi4.ccasp-native-qomp-production148-v3"
PREP="$P/runs/ccasp-native-s66-qomp-production148-003"
BANK="$P/runs/ccasp-native-s66-qomp148-bank003"
BOOT="$BANK/inputs/bootstrap"
```

Do not modify/reuse sourcev2/PREP002/worker002/science002/audit002.
Prepare PREP003 with the unchanged QOMP preparation CLI and the same pinned
raw4, measured bank/all131 reports/resources, original25 anchors, runtime view
and14 receipts. There is no arbitrary production path override.

Before controller setup, create only the fresh owned mode0700 BANK/inputs/
bootstrap directories and stage the following **actual**, parent-verified
inputs there:

```text
authority.json
worker-preflight.json
science-accounting.txt
audit-accounting.txt
audit-report.json
qomp-outcome.json
case.json
server-native-envelope.json
bank-initial-snapshot.json
server-tensors.npz
```

`authority.json` has the exact fields in `qomp_expansion.MANIFEST_FIELDS`,
schema `qomp-ordinary-bootstrap-authority/1`. It binds compiled/test/
qualifying/descriptor/SCI/audit revisions, SCI/worker/audit jobs, worker hash,
dataset and record IDs, the four raw seed receipt hashes, normalized dataset
snapshot hash, tensor hash and raw SCI/audit accounting hashes. Supply the
literal SHA of this parent-sealed manifest as `BOOTSTRAP_SHA`; it is **not**
the audit-report SHA. No unverified current hash becomes historical authority.
The raw accounting captures must contain exactly the successful allocation,
`.batch`, `.0` rows (optional successful `.extern`), without headers.

The controller additionally reads original25 anchors from PREP003 inputs.
It reconstructs the whole131 measured pool and106 missing drafts. Bootstrap
stores actual source6033 as generation1, leaving105 eligible successors.
It refuses partial/replayed setup; do not clean/reset a failed transaction.

From the operator host using the existing sealed SSH/watcher adapters:

```bash
/path/to/operator/python -I -B "$OPERATOR_SOURCE/devtools/isapol_s66_campaign/bank_controller.py" setup \
  --producer qomp --source "$SOURCE" --commit "$FROZEN_HEAD" \
  --bootstrap-sha256 "$BOOTSTRAP_SHA"
/path/to/operator/python -I -B "$OPERATOR_SOURCE/devtools/isapol_s66_campaign/bank_controller.py" run \
  --producer qomp --source "$SOURCE" --commit "$FROZEN_HEAD" \
  --bootstrap-sha256 "$BOOTSTRAP_SHA" --maximum 1
```

`OPERATOR_SOURCE` is the local checkout of the exact same frozen HEAD;
`SOURCE` remains the fixed remote path above. Use the operator's existing
Python interpreter. Alternatively use the parent-controlled `remote`
subcommands below with the actual fixed remote path. `native_controller.Remote`
uses `~/.pi/agent/skills/phoenix-slurm-experiments/scripts/{phx,watch_jobs.sh}`
and seals both bytes; it does not require a new adapter.
No operator command is executed by source preparation or pure tests.

Exact individual remote operations, through the approved clean manager
environment, are:

```text
bank_controller.py remote --producer qomp --action initialize ...
bank_controller.py remote --producer qomp --action select ...
bank_controller.py remote --producer qomp --action submit --monomer MID --phase worker ...
bank_controller.py remote --producer qomp --action verify --monomer MID --phase worker ...
bank_controller.py remote --producer qomp --action submit --monomer MID --phase science ...
bank_controller.py remote --producer qomp --action verify --monomer MID --phase science ...
bank_controller.py remote --producer qomp --action submit --monomer MID --phase audit ...
bank_controller.py remote --producer qomp --action verify --monomer MID --phase audit ...
```

Every `...` is the same explicit `--source SOURCE --commit FROZEN_HEAD
--bootstrap-sha256 BOOTSTRAP_SHA --adapter-seal ACTUAL_ADAPTER_SHA`.
Wait for actual accounting before each `verify`. `phase_command` emits the
exact finite sbatch arguments; it is not a generic free-form command builder.
Only the parent may submit, cancel or release jobs.

Selected member cases are `BANK/cases/MID.json`; role allocations are
`BANK/runs/MID/{worker,science,audit}`. Each SCI consumes its positively
verified member worker SHA. Its one-record append repeats live original357
preservation, manager-tag isolation, owned cohort/previous snapshot checks
and fresh raw filesystem/user/effective-group admission immediately before
release. Controller admission repeats before allocation. The same147 member
and sum-plus-largest predicates,65536 engineering file estimate,256GiB bytes,
five resource profiles and all scientific caps remain unchanged.

The actual science command/environment/CWD still comes from the reviewed
ordinary CLI launcher inside decoded-FD tracing. Literal metadata discovery
retains its6033 fast path. Full scientific seals and audit final seals precede
admissible success publication. Signal/child/postseal failure stops and keeps
partials/intents; there is no retry, repair or continuation through failure.

## Limits / next gate

`--maximum 1..8` is a bounded **sequential** chunk. Do not call it parallel
eight. This milestone intentionally does not expose QOMP through the old
parallel-wave controller or relax active-manager namespace isolation. That
port and its authorized member-tag isolation need separate review.

Ordinary pilot evidence is numeric/export/transport/runtime evidence, not
independent solver/model accuracy certification. The historical numerical
outlier remains unresolved. Genuine bounded scientific failure-lifecycle
evidence (not a metadata discovery failure), fresh quota and parent approval
remain gates before widening. No original held jobs or QCA357 records are
cancelled, reset, released or altered by this implementation.
