# Closed QOMP registered waves (offline implementation; NOT activated)

This extends the **existing** `bank_wave*` controller, registration transport,
all-SCI barrier and whole-wave reducer. It does not replace them with another
orchestrator. Legacy grid400 remains its explicit default/dataset357 policy.
QOMP uses `--producer qomp`, scope `qomp-wave` and the reviewed147 execution
policy. Scientific code, methods/grids/ISA120/Drho32, numerical/IO/work caps,
five resource shapes,256GiB and65536 engineering file estimate are unchanged.
That estimate is NOT a file-creation cap.

**`qomp_wave_policy.PRODUCTION_ENABLED=False`.** Genuine bounded scientific
failure-lifecycle evidence and parent review are still widening gates.
Earlier metadata discovery failure is not such evidence. No force/skip flag
or automatic activation is provided. Do not deploy/bind plans to a dormant
HEAD and then advance that checkout. Parent must explicitly approve the final
activation revision BEFORE creating its first immutable wave plan.
While inactive, ALL controller actions, including `remote initialize` and
the public initializer, refuse before plan publication/control/server access.
Inactive metadata validation is only `qomp_wave_policy.load_adoption`, which
does not create a plan or activate anything.

## Fixed authority, distinct revisions

- Compiled science fb11, installed tests311, qualifying ops970, actual core
  f4a1/Python331b/dev212/build002,14 receipts/view remain unchanged.
- Measured all131/cba19 producing inventory remains historicalcba.
- Original25 remain original-provenance results in357, NOT new QOMP records.
- External0026 SCI/audit remain source6033; an adopted0027 or later ordinary
  serial successor remains source37aa. Neither is relabelled as current ops.
- Authority raw1350B SHA
  `091f87bbea8562787b476122df602991d6b2e9fbec23e1190876b6cbe14ae11a`;
  unchanged context SHA
  `5b5b9037bfaf93da2312e5f050f51610455ed8f84a654203f282d39896cd5576`.
  Dataset/specification are derived from this verified authority/context,
  not a monkeypatched357 or a second cohort.
- New source: `P/gits/psi4.ccasp-native-qomp-production148-v4`.
- New root: `P/runs/ccasp-native-s66-qomp148-waves004`.
- New scratch:
  `/storage/scratch1/3/awallace43/ccasp-native-qomp148-waves004`.

Here `P=/storage/project/r-cs207-0/awallace43`. No deployedv3/PREP003/BANK003
path is modified. No old held jobs are controlled. All eight historical held
IDs are explicitly forbidden to the QOMP held-job validator.

## Actual cutover metadata, no fabricated anchor

Parent stages private immutable copies under new root `inputs/`:

```text
bootstrap/{authority.json,worker-preflight.json,science-accounting.txt,audit-accounting.txt}
original25/<same three pinned relative paths as qomp_campaign.ORIGINAL_PATHS>
adopted/{adoption.json,state.json,bank-context.json,bank-checkpoint.json,
         science-accounting.txt,audit-accounting.txt,verified.json}
adopted/proofs/<SHA>.json
```

`verified.json` is mandatory for a serial generation>1; generation1 instead
retains explicit external6033 provenance and a null verification commitment.
Proof keys are exactly seed4 plus every actual serial ledger receipt hash;
never invent completed prefixes or projected intermediate checkpoints.
Bootstrap original anchors/worker/accounting/authority are also carried into
wave proofs. Subsequent wave proofs use a separate private QOMP wave archive;
the deployed serial proof archive stays untouched.

`adoption.json` is a parent-sealed `qomp-wave-serial-cutover/1` object, exactly:

```text
schema
producing_operations = 37aa33fd215a87832e382c483ab1ec8adb499708
bootstrap_authority_sha256 = 091f...
state_sha256
context_sha256 = 5b5b...
checkpoint_sha256
science_accounting_sha256
audit_accounting_sha256
verification_sha256
```

All are actual byte commitments, not normalized projections or current-hash
approval. Parent supplies its literal SHA via `--adoption-sha256`.
Raw terminal captures use `sacct -n -P -j JOB -o JobIDRaw,State,ExitCode` and
must have unique successful allocation/batch/.0 rows; optional successful
extern is permitted. For generation>1, preserve the actual audit-payload
context/checkpoint origin paths and matching controller `verified.json`/
state/audit/source/job. Do not substitute the original seed context PATH
just because its context BYTES are unchanged.

Missing/unobserved actual cutover evidence refuses. Parent may wait for the
serial0027 audit, then copy its genuinely accepted state/proofs; no agent
network action or placeholder receipt is required/allowed.

## Existing controller route, bounded1..8

After actual lifecycle evidence, parent review and final activation revision:

```text
bank_wave_controller.py run --producer qomp
 --source SOURCE_V4 --commit FROZEN_ACTIVATION_HEAD
 --predecessor-state WAVE_ROOT/inputs/adopted/state.json
 --adoption-sha256 PARENT_LITERAL_CUTOVER_SHA --member-limit 1
```

Use the existing sealed local `native_controller.Remote` SSH/watcher adapters,
same as serial. Start with1; widening to8 needs parent approval/evidence.
The command remains DISABLED in this offline revision. No environment,
rebuild, receipt normalization or new generic campaign configuration.
Next waves use their previous own `accepted.json`, same initial adoption pin,
and exact next eligible slice. Accepted counts derive from the proof chain;
do not hardcode0028 or105 when the actual adopted generation may differ.

The reused finite barriers are:

1. Check actual adopted prefix/proofs/accounting, current live dataset/spec,
   old357 preservation and manager namespace isolation. Reconstruct all
   measured cases; select exact next1..8 eligible members, never numeric gaps.
2. Fresh raw filesystem/user/effective-group admission uses reviewed147
   per-member and sum-plus-largest predicates. Submit workers on finite
   embers resources; wait/verify ALL workers and actual receipt SHAs.
3. Submit only these SCI jobs HELD. Bind each exact verified member worker
   SHA in its command and registration slot. Check own source `Command`,
   actual held job/user/resources, and old-job exclusion.
4. Separate4CPU/24GiB/45min registration allocation performs sequential,
   typed one-entry/one-record writes to the existing cohort/spec. No create,
   spec insertion, reset or retries. Master plus every canonical per-case
   reservation is acquired BEFORE HTTP; serial/wave races collide.
5. Verify registration/accounting/held slots. Fresh aggregate quota before
   permit and AGAIN before EACH individual release; recheck only that still-
   held member after earlier members start. No retry/cancel/repair on partial.
6. ALL SCI must finish/deliver successfully before the immutable barrier.
   Compute/audit dataset clients are factory-owned read-only clients.
   Only exact validated active peer tags are permitted; wildcard, own duplicate
   or unrelated same-namespace manager tags refuse. One owned record/claim/
   ACK/delivery/executor, zero retries; previous/peer record IDs protected.
7. Run independent member audits; they publish evidence only, NOT serial
   checkpoints. Numeric/export/envelope/decoded-FD gates are unchanged.
8. Separate whole-wave finalizer validates every audit/accounting/barrier/
   capture and double-live snapshot. Full source/runtime postseals precede
   proof/checkpoint/advance publication. Controller accepts only after actual
   finalizer allocation/batch/.0 success, full proof replay and live agreement.

Fixed nested member roots: `WAVE_ROOT/wave-G-H/members/MID/{worker,science,audit}`.
The existing `qomp148.sbatch` is v4-wave-only14 slots:

```text
SOURCE COMMIT PHASE RUN CASE WORKER_OR_- WORKER_SHA_OR_- SCIENCE_OR_-
PLAN PLAN_SHA REGISTRATION_OR_- REG_SHA_OR_- BARRIER_OR_- BARRIER_SHA_OR_-
```

Worker uses `-` for all future authority. HELD SCI binds registration PATH,
never a not-yet-existing registration hash. Audit binds actual registration
and full all-SCI barrier hashes. Ops reuse `bank_wave_ops.sbatch` with its
original six args plus `qomp ADOPTION_SHA`. No role/root/argv override.
Clean env, fresh owned HOME/cache/CWD before imports, cooperative TERM plus
child forwarding/wait/reap, existing metadata discovery fast path and full
ordinary scientific runtime seals remain in force.

## Limits

This is ordinary-OS attributable runtime/export/transport audit evidence, not
hostile-host attestation, solver/model accuracy certification or whole-S66
scientific acceptance. The historical numerical variation remains unresolved.
No failure-lifecycle trace is fabricated. No old357 result/intent, historical
FAILED build, old held-job state or immutable deployed source is rewritten.
No automatic retry, cleanup, resume, cohort creation or cap raising.
