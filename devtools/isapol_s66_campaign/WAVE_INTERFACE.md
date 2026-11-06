# Eligible-wave integration contract

Interface revision: `registered-wave/1`. This document is the first-stage
handoff, based on `cf834cacf0157b206eede0df1db049db954e667c`.
It specifies interfaces, not implemented functionality or production approval.
Coordinate changes before changing this contract.

## Ownership and safety

Core agent: `bank_wave.py`, `bank_wave_campaign.py`,
`bank_wave_transport.py`, `bank_wave_controller.py`, `bank_wave_ops.sbatch`,
their tests, and necessary **private** seams in `bank_store.py` /
`bank_transport.py`. Runtime agent: `bank_wave_runtime.py` and existing
worker / science / auditor / session integration and wrappers.

The runtime must dispatch explicitly on `campaign_scope == 'eligible-wave'`.
Never route wave cases through serial `audited_bank`, `bank_append`,
`bank_audit.publish`, `bank_transport.RequestScope`, or serial checkpoint
constructors. Preserve exact-type serial client guards and all default modes.
An individual member audit is evidence, **not** checkpoint publication.

Reuse scientific runtime, finite profiles, raw molecular preparation, numeric,
envelope, FD/CLI, one-shot claim/delivery, and scratch checks. No scientific
source, caps, environment, QCEngine, or executor concurrency changes.
Each member remains `maximum_records=1`, `find_existing=False`, no retries,
one executor, `max_workers=1`, concurrency 1.

No deployment, network access, credentials, or SLURM actions are authorized
for development of this contract. Production activation remains disabled
pending coordinator review.

## Encoding, types, and authority

Unless stated otherwise, arguments / returns are ordinary JSON dictionaries.
`*_raw` is bounded `bytes`; `*_sha256` hashes those **actual bytes**, never a
parse/re-encode. Pure dictionary commitments use `audited_bank.encode` /
`audited_bank.digest`; snapshot commitments use `dataset_guard.digest`.
Do not interchange those hashes. `Path` means `pathlib.Path`.
Malformed, missing, duplicate, ambiguous, or changed evidence raises and stops;
there is no repair, retry, resume, fallback to serial, or inferred success.

The original `bank_campaign` constants and measured pool authority are reused:

* SCI `1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2`;
  descriptors job `15153484`, OPS
  `f7d4bb4314baa7372ed47d5876e9d34475ec9c19`,
  allocation SHA
  `f3c8cf04297444ea67a79352f4aae8cef7b645262a48176b9b04e89832b88050`.
* Dataset **357**, 116 eligible / 15 refused. Bootstrap context SHA
  `a92028bc139edcffe5c6f03cc8dd1945c30183149a235b94852882dc2678fec9`.
* Serial anchor generation **25**, checkpoint SHA
  `d4ace10ad0823800f2b0a678feaa88539a30907c8a6a869e9363f7a91ff7ced8`,
  at `bank_campaign.CAMPAIGN_ROOT /
  'ccasp-native-s66-grid400-monomer-0025-audit/payload/bank-checkpoint.json'`.
  Science `15216900`, audit `15220617`: require original proofs, verified
  receipts, and positive allocation / batch / `.0` exit0 plus fresh live
  agreement before mutation. These pins do not replace those checks.

New closed root:
`bank_campaign.RUNS / 'ccasp-native-s66-grid400-1aaa9e0a30-waves001'`.
Do not write the old serial root, state, plan, or checkout.
Wave directory: `WAVE_ROOT / wave_id`, where
`wave_id = 'wave-%04d-%04d' % (generation_before, generation_after)`.
No caller-selected alternate production roots or symlink aliases.

## Case and CLI contract

Reconstruct the complete raw prepared inventory with the existing verified
compiled pool and `grid400_bank_case.prepare_case`. Select exactly
`eligible_ids[g:g + min(8, len(eligible_ids)-g)]`, not a numeric ID range.
Reject duplicate IDs, entries, record IDs, jobs, tags, or geometry identities,
including collisions with the accepted prefix. Preserve all 15 refusals.

The bound wave case is a deep copy of its raw prepared case, with ONLY these
top-level changes / additions:

```python
{
    "campaign_scope": "eligible-wave",
    "dispatch_protocol": "registered-wave/1",
    "dataset_mode": "append",
    "dataset_id": 357,
    "wave_id": wave_id,
    "bootstrap_context_sha256": context_sha256,
    "checkpoint_sha256": predecessor_sha256,
    "predecessor_snapshot_sha256": predecessor["snapshot_sha256"],
    "prepared_case_sha256": sha256(encode(original_prepared_case)),
}
```

Schema stays `native-selected-molecular-pilot/1`. All raw `entry`, molecule,
reference attributes, native request, runtime hash, profile and queue fields
remain unchanged. No plan hash, actual job ID, registration hash, or record ID
is added to the case: this avoids plan/case circular hashes and permits worker
preflight before registration.

All three Python CLIs accept:

```text
--producer grid400 --campaign-scope eligible-wave --remaining-case
--wave-plan PATH --wave-plan-sha256 SHA
```

Science and audit additionally require `--wave-registration PATH`.
Audit may carry `--wave-registration-sha256 SHA`; science must work without
that flag. The already-HELD science command cannot bind a hash that does not
yet exist. No historical owned-dataset acceptance flags or independent
`--bank-context` / `--bank-checkpoint` arguments are accepted in wave mode:
the plan binds those paths and bytes.

New wrapper selector is
`--grid400-bank-wave PLAN_PATH PLAN_SHA256 REGISTRATION_PATH`.
It expands to the Python flags above, with fixed producer/scope/append mode.
For worker only, `REGISTRATION_PATH` is the literal `-`, and no registration
flag is forwarded. Audit can additionally accept
`--wave-registration-sha256 SHA` after the selector. Existing positional
wrapper arguments and all old selectors remain unchanged.

### Exact controller wrapper argv

These are argument arrays, not shell command strings. Let `D` be the closed
wave directory, `M` the MID, `S` the source checkout, `C` its operational
commit, `P = D/'plan.json'`, `H = sha256(actual_plan_bytes)`,
`R = D/'registration.json'`, `RH = sha256(actual_registration_bytes)`,
`B = D/'science-barrier.json'`, `BH = sha256(actual_barrier_bytes)`,
`K = D/'cases'/ (M+'.json')`, and
`F = D/'members'/M/'worker/payload/worker-preflight.json'`.

After the controller's explicit sbatch resource/routing/output options:

```python
# Worker: 8 arguments after script name.
[S/'devtools/isapol_s66_campaign/phoenix_worker_cb63305.sbatch',
 S, C, K, D/'members'/M/'worker',
 '--grid400-bank-wave', P, H, '-']

# Science: 9 arguments after script name; sbatch options MUST include --hold.
[S/'devtools/isapol_s66_campaign/phoenix_pilot_cb63305.sbatch',
 S, C, K, F, D/'members'/M/'science',
 '--grid400-bank-wave', P, H, R]

# Audit: 16 arguments after script name; emitted only after whole-SCI barrier.
[S/'devtools/isapol_s66_campaign/runtime_audit_cb63305.sbatch',
 S, C, K, F, D/'members'/M/'science', D/'members'/M/'audit',
 '--grid400-bank-wave', P, H, R,
 '--wave-registration-sha256', RH,
 '--wave-science-barrier', B, '--wave-science-barrier-sha256', BH]
```

All Paths are converted to absolute strings before execution. The audit
wrapper requires the barrier path/hash pair; the registration hash pair is
optional in the wrapper grammar but **always emitted by this controller**.
No `--advance-bank-checkpoint` / `--advance-checkpoint` is forwarded in wave
mode. Wrapper grammar rejects extra/duplicate options and mixed old modes.
Both SCI and audit forward `--wave-registration`; audit also forwards the
barrier flags and optional registration SHA verbatim. The wrappers select
the existing `grid400-1aaa9e0a30-bank` producer paths (unchanged preparation,
build, qualification and descriptors), not a new scientific installation.
Their wave-only path checks admit precisely these nested case/run paths;
leave historical path guards intact.

Tags stay `ccasp-native-grid400-bank-MONOMER-ACTUAL_SCI_JOB_ID`.
Science scratch stays under the existing private scratch root with that tag;
worker scratch retains `ccasp-native-worker-grid400-bank-ACTUAL_WORKER_JOB_ID`.
Never introduce shared scientific scratch.

## Immutable artifacts

Paths below are relative to the closed wave directory. Each is exclusive,
durable metadata written once using the bank store's bounded-FD /
`O_EXCL` / no-follow / fsync primitives. A partial artifact blocks progress.

### `plan.json` — `native-eligible-wave-plan/1`

Required fields:

```python
{
    "schema": "native-eligible-wave-plan/1",
    "campaign_scope": "eligible-wave",
    "dispatch_protocol": "registered-wave/1",
    "wave_id": wave_id, "dataset_id": 357,
    "source": absolute_source, "operational_commit": commit,
    "bootstrap_context_path": absolute_context_path,
    "bootstrap_context_sha256": context_sha256,
    "predecessor_path": absolute_checkpoint_path,
    "predecessor_sha256": predecessor_sha256,
    "predecessor_snapshot_sha256": predecessor_snapshot_sha256,
    "generation_before": g, "generation_after": g + len(member_ids),
    "eligible_ids": eligible_ids, "refused_ids": refused_ids,
    "eligibility_sha256": eligibility_sha256,
    "member_ids": member_ids,  # ordered eligible slice, length 1..8
    "members": {
        mid: {
            "case_path": absolute_wave_case_path,  # cases/MID.json
            "prepared_case_sha256": original_prepared_case_sha256,
            "case_sha256": bound_case_sha256,
            "execution_profile": unchanged_profile,
        } for mid in member_ids
    },
}
```

The plan has no science job IDs or future registration hashes. Controller
submission intents separately bind exact commands, source, plan SHA, actual
job responses and verified routes. Worker proof paths are
`members/MID/worker/payload/worker-preflight.json`; science and audit run roots
are `members/MID/science` and `members/MID/audit`.

### `registration.json` — `native-eligible-wave-registration/1`

Required fields: `schema`, `campaign_scope`, `dispatch_protocol`, `wave_id`,
`plan_sha256`, `dataset_id`, `bootstrap_context_sha256`,
`predecessor_sha256`, `predecessor_snapshot_sha256`,
`registration_role`, `dispatch_intent_sha256`, `member_ids`, `members`,
`registered_snapshot_path`, `registered_snapshot_sha256`.

`registration_role` is `{"kind": "allocation", "job_id": actual_ops_job_id,
"allocation": actual_allocation}`. It binds the real registration allocation,
never a borrowed science job. Members map each MID to exactly:

```python
{
    "case_sha256": case_sha256,
    "worker_preflight_sha256": worker_preflight_sha256,
    "science_job_id": actual_science_job_id,
    "compute_tag": actual_tag,
    "entry_name": entry_name, "specification_name": specification_name,
    "record_id": record_id, "molecule_id": molecule_id,
    "captures": {
        role: {"path": absolute_capture_path, "snapshot_sha256": snapshot_sha256}
        for role in ("before", "entry", "record")
    },
}
```

Captures are actual per-insert observations, not projections:
`registration/MID/owned-before-entry.json`, `owned-after-entry.json`,
`owned-after-record.json`. Final all-waiting observation:
`registration/registered-snapshot.json`. Validate each sequential one-entry
delta with `dataset_guard.assert_append_delta`; every insert and submit is
exactly 1 new / 0 existing, no retries. All records are waiting, no histories,
with exact actual tags and effective specs before registration is sealed.

Reserve **one wave intent** before the first dataset mutation in the SAME
`bank_store.INTENTS / str(dataset_id) / predecessor_snapshot_sha256` directory
used by serial dispatch. New schema `native-bank-wave-dispatch-intent/1`
binds plan, predecessor, real registration role, and all held job/tag/case/
preflight identities. Do not encode fake singleton or science authority.
The reservation is never removed, retried, or cleared.

### `release-permit.json` — `native-eligible-wave-release-permit/1`

Required fields: `schema`, `dispatch_protocol`, `wave_id`, `plan_sha256`,
`registration_sha256`, `dataset_id`, `predecessor_sha256`,
`dispatch_intent_sha256`, and `members`. `members` maps every MID to exactly
`science_job_id`, `compute_tag`, `record_id`, `case_sha256`.
The fixed permit path is **derived from the verified plan path**, not a CLI
path. Registration path must be that same directory's `registration.json`.

Only after registration and all HELD-job identities/routes are verified and
aggregate scratch/headroom admission passes may the controller exclusively
write this permit. Then durably record each release-attempt intent before
issuing that job's single `scontrol release`. Failure or lost reply stops;
never retry, clear, cancel, or release preexisting jobs. The permit authorizes
consumption; it is not a claim that every release succeeded.
SCI independently checks its actual allocation JobId against its permit slot.

### `science-barrier.json` — `native-eligible-wave-science-barrier/1`

Only the controller writes this, exclusively, after **every** declared SCI
member has positively completed and delivered. It is required input to every
member audit and to finalization; successful accounting of just the audited
member is insufficient. Required structure:

```python
{
    "schema": "native-eligible-wave-science-barrier/1",
    "status": "ALL_SCIENCE_POSITIVELY_VERIFIED",
    "wave_id": wave_id, "plan_sha256": H,
    "registration_sha256": RH, "release_permit_sha256": permit_sha256,
    "member_ids": member_ids,
    "members": {
        mid: {
            "science_job_id": job_id,
            "record_id": record_id, "compute_tag": tag,
            "case_sha256": case_sha256,
            "outcome_path": str(D/'members'/mid/'science/payload/pilot-outcome.json'),
            "outcome_sha256": outcome_sha256,
            "delivery_snapshot_sha256": actual_delivery_snapshot_sha256,
            "accounting_path": str(D/'members'/mid/'science/accounting.txt'),
            "accounting_sha256": accounting_raw_sha256,
        } for mid in member_ids
    },
}
```

`accounting.txt` is the saved actual, nonempty `sacct -n -P -j JOB -o
JobIDRaw,State,ExitCode` response, not an invented boolean or normalized
projection. Require each unique `JOB`, `JOB.batch`, `JOB.0` row to be
`COMPLETED|0:0` using the existing positive-accounting verifier; unknown,
missing, contradictory or unsuccessful required rows stop the barrier.
Bind every outcome to plan/case/preflight/job/tag/record, successful one-shot
delivery and actual saved delivery snapshot. Audit loading verifies all
barrier member hashes and accounting, not merely the barrier status string.

## Core-agent callable interfaces

All functions here are supplied by the core agent. Signature names below are
stable; additional internal helpers are not part of the handoff.

### `bank_wave.py` (pure)

```python
validate_plan(plan_raw, plan_sha256, *, prepared_cases,
              checkpoint_raw, checkpoint_sha256,
              context_raw, context_sha256, proof_blobs) -> dict  # parsed plan
bind_member(prepared_case, *, plan, monomer_id) -> dict  # reconstructed case
validate_registration(plan, registration_raw, registration_sha256, *,
                      predecessor_snapshot, cases, captures,
                      registered_snapshot) -> dict  # parsed registration
validate_release_permit(plan, plan_sha256, registration,
                        registration_sha256, permit_raw) -> dict  # parsed permit
validate_registered_slot(plan, registration, release_permit, draft, *,
                         allocation=None) -> dict  # registration members[MID]
validate_capture(plan, registration, *, predecessor_snapshot,
                 registered_snapshot, snapshot,
                 member_id=None, require_complete=False) -> dict
```

`prepared_cases` is the complete original inventory keyed by eligible MID;
`cases` contains exactly this wave's reconstructed cases.
`captures` is `{mid: {role: snapshot for role in ('before','entry','record')}}`.
Allocation, if supplied, must match the actual science job; audit uses `None`
and independently verifies the science allocation / outcome / accounting.
The runtime SCI entry point MUST supply its allocation.

`validate_capture` validates the entire observed dataset, not just one record.
It returns a preservation summary, with snapshot SHA, preserved prefix count,
and declared peer statuses. Metadata, specs, entries, associations, inventories
and accepted-prefix records must match exactly. Only declared wave records
may transition waiting -> running -> complete, with no error/retry history.
Their id, molecule, effective specification and any remaining task identity/
tag must agree with registration. Completion requires one attributable history
and consistent native envelope/tensor identities; complete peers need not yet
have audit receipts. No unknown/missing member or changed immutable field is
permitted. Full final acceptance checks those peer identities against their
own audited receipts; interim peer completion is not acceptance.

`dataset_guard.snapshot_dataset` remains unchanged. It records an opaque
whole-body hash plus explicit record identity fields: do not simply ignore a
changed body hash or add a broad "peer differences allowed" exception.
Where additional record-body identity evidence is necessary, collect and bind
it through new wave-only read helpers/captures and validate an explicit
closed set of lifecycle changes. Never manufacture serial observations.

The core also supplies whole-wave checkpoint validation / reduction internally;
the runtime agent does not call serial `advance` or any member publication.

### `bank_wave_campaign.py` (closed reads and reconstruction)

```python
pins() -> dict
source_names() -> tuple[str, ...]
verify_prepared_from_paths(args, manifest_path) -> dict
load_binding(args, *, require_registration=False) -> dict
revalidate_binding(binding) -> None
```

`pins()` extends `bank_campaign.pins()` with scope/protocol (same namespaces,
dataset/spec, producer and protected objects). `source_names()` supplies
wave-specific operational dependency names; the runtime combines them with
the existing common source inventory, deduplicated, rather than replacing it.

`verify_prepared_from_paths` checks measured authority and entire raw inputs,
context, predecessor proofs, plan hash, exact case path/content, and selection.
It works before registration and does not read registration or permit.

`load_binding` validates those same authorities and returns these exact keys:

```python
{
    "plan_path": Path, "plan_raw": bytes, "plan_sha256": str, "plan": dict,
    "context_raw": bytes, "context_sha256": str,
    "checkpoint_raw": bytes, "checkpoint_sha256": str,
    "proof_blobs": dict,  # complete predecessor chain, content-addressed bytes
    "monomer_id": str,   # selected from args.pilot, verified against plan
    "registration_path": Path_or_None,
    "registration_raw": bytes_or_None, "registration_sha256": str_or_None,
    "registration": dict_or_None,
    "release_permit_path": Path_or_None,
    "release_permit_raw": bytes_or_None, "release_permit_sha256": str_or_None,
    "release_permit": dict_or_None,
    "captures": dict_or_None, "registered_snapshot": dict_or_None,
    "science_barrier_path": Path_or_None, "science_barrier_raw": bytes_or_None,
    "science_barrier_sha256": str_or_None, "science_barrier": dict_or_None,
}
```

With `require_registration=False`, registration/permit/capture fields are
`None`: worker mode cannot accidentally depend on future files.
With `True`, every field is required, registration captures are loaded and
verified, registration SHA comes from the exclusive verified permit, and any
explicit audit SHA must also agree. No fabricated `/1` serial checkpoint.
Subsequent waves carry the unchanged actual serial anchor plus wave links.

Audit arguments additionally require `args.wave_science_barrier: Path` and
`args.wave_science_barrier_sha256: str`; those populate the four barrier keys
after complete barrier validation. They are `None` for worker/SCI.
Runtime `load_binding(args)` identifies audit by non-None `args.case` (the
existing audit CLI science-run argument), and rejects missing barrier
arguments for that role. Do not accidentally reuse `args.case` as a case JSON
path: `args.pilot` is the case JSON path in all three CLIs.

`release_permit_path` is always
`plan_path.parent/'release-permit.json'`. `registration_path` must equal
`plan_path.parent/'registration.json'`; no alternate aliases are accepted.
`revalidate_binding` reopens bounded no-follow plan/context/predecessor/proofs,
registration/captures/permit (and barrier when present), compares actual
bytes/commitments and repeats authority checks against the loaded binding.
Runtime calls it on entry, immediately before starting the owned manager,
after delivery before writing outcome, and in audit before report publication.
`prepare_draft` repeated at exit remains necessary but is not a replacement
for permit/registration revalidation.

### `bank_wave_transport.py`

```python
transport_binding() -> dict
readonly_client_class(binding, *, monomer_id) -> type
require_readonly_client(client, binding, *, monomer_id) -> None
```

`readonly_client_class` requires a fully registered/permit-validated binding.
The returned client exposes existing PortalClient metadata reads but rejects
every dataset mutation (including create/spec/entry/submit/reset/retag/delete).
It preserves one-shot constructor login and refresh behavior, no refresh-to-
login fallback, no mutation authority inherited from a serial client.
There is no client inside `binding`. Callers create one explicitly:

```python
wave = helper('bank_wave_runtime')  # caller's cached helper module
binding = wave.load_binding(args)
transport = wave.helper('bank_wave_transport')  # public cached helper(name)
Client = transport.readonly_client_class(
    binding, monomer_id=binding['monomer_id'])
client = Client(server_uri, username=username, password=password,
                verify=verify, show_motd=False)
# server config remains in memory; use existing server-config validator.
```

The runtime supplies `helper(name)` for its closed cached dependency set;
`bank_wave_transport` and `bank_wave_campaign` are permitted names.
The transport caches the exact generated class by
`(plan_sha256, registration_sha256, release_permit_sha256, monomer_id)`.
`require_readonly_client` demands that exact factory-owned class and matching
authority, not duck typing/subclassing; runtime uses the same cached module
for factory and validation. Each client independently checks the approved
server identity before use. Audit uses the same readonly factory, never the
serial `bank_audit.approved_client`. Manager activation/claim/result delivery
continues through the separately reviewed one-shot manager client, not this
dataset-read client; session validation binds its sole registered record/tag.

The separate registration-only `WaveScope` and typed one-entry write client
are controller internals. Their allowlist authorizes one entry POST and one
submit POST per declared slot, no other dataset mutations, consuming attempts
before HTTP even for lost replies/auth errors.
`transport_binding()` is stable before registration so worker preflight can
bind the reviewed wave transport source/policy without job IDs.

## Runtime-agent callable interfaces

The runtime agent supplies these exact `bank_wave_runtime.py` functions:

```python
prepare_draft(args, manifest_path) -> dict
load_binding(args) -> dict
registered_record(client, binding, draft, allocation, preflight, save) -> dict
capture_delivery(client, binding, draft, *, record_id, manager,
                 envelope, tensor_raw, save) -> dict
audit_preservation(data, outcome, draft, binding, specification) -> dict
pins() -> dict
source_names() -> tuple[str, ...]
started_fields(binding) -> dict
```

* `prepare_draft` delegates to campaign reconstruction; works in worker before
  registration. No scientific imports or dataset mutation.
* `load_binding` calls campaign `load_binding(..., require_registration=True)`;
  SCI and audit call this, workers do not. The loaded binding is immutable
  input; do not stash clients, record objects, or runtime outcomes inside it.
* `registered_record` requires the reviewed read-only client, validates actual
  allocation/tag/record slot, exact preflight hash/content/profile, refreshes
  dataset/record, and validates the observed wave inventory including peers.
  Its **own** record must still be waiting with exact registered task/tag.
  Return `{"dataset": dataset_object, "record": record_object,
  "created": {"dataset_id": int, "record_id": int, "molecule_id": int,
  "compute_tag": str, "n_inserted": 1, "n_existing": 0,
  "dispatch_intent_sha256": str}}`. Counts describe the verified registration,
  not a SCI-side insert. Save `record-created.json` with those fields.
* `capture_delivery` uses fresh read-only snapshots and pure peer-aware
  validation, binds its completed record to manager/envelope/tensor bytes,
  saves `owned-after-delivery.json`, and returns a preservation summary.
  `save(name, value)` throughout is an exclusive durable JSON callback.
  `tensor_raw` is bounded bytes; `envelope` is the actual decoded JSON.
  This name is deliberately SCI-local:
  `D/members/MID/science/payload/owned-after-delivery.json`.
  It does **not** save `owned-before-entry.json`, `owned-after-entry.json`,
  or `owned-after-record.json` under the SCI payload: those were observed by
  registration, live only under `D/registration/MID/`, and are loaded through
  `binding['captures']`. Never copy/projection-label them as SCI observations.
* `audit_preservation` independently reads saved SCI delivery, registration
  captures/permit, server-envelope and tensor bytes under `data: Path`;
  verifies outcome and draft bindings plus the normalized `specification`.
  It returns `status='PASS_CAPTURED_WAVE_DATASET_PRESERVATION'`, `wave_id`,
  `plan_sha256`, `registration_sha256`, `release_permit_sha256`,
  `checkpoint_sha256`, `dispatch_intent_sha256`, `before_sha256` (actual
  all-waiting registration snapshot), `after_sha256` (actual delivery),
  `audited_native_envelope_sha256`, `audited_tensor_sha256`,
  `preserved_records`, `peer_statuses`. No checkpoint publication/live write.
  Audit additionally requires the validated all-SCI barrier in its binding.
* `pins` / `source_names` delegate to campaign helpers. Runtime source names
  additionally include `bank_wave_runtime.py`; callers retain their existing
  worker/SCI/audit common dependency and wrapper source checks.
* `started_fields` returns fields for worker/preflight, SCI started/outcome
  and audit report binding: `campaign_scope`, `dispatch_protocol`, `wave_id`,
  `plan_sha256`, `bootstrap_context_sha256`, `checkpoint_sha256`,
  `predecessor_snapshot_sha256`, `registration_sha256`,
  `release_permit_sha256`, `dispatch_intent_sha256`.
  All are mandatory for SCI/audit; workers set only scope/protocol/wave/plan/
  context/predecessor fields directly from their verified plan (no fake
  registration fields). Existing monomer, actual allocation, pilot hash,
  preflight hash, dataset/spec/record/tag and source fields remain mandatory.

Worker success status: `PASS_SELECTED_WAVE_CASE_WORKER_PREFLIGHT_ONLY`.
SCI outcome retains
`DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION`, explicitly wave
bound. Member audit success:
`PASS_SELECTED_WAVE_CASE_SINGLETON_EVIDENCE`, explicitly wave bound.
These wave statuses/scope are deliberately unacceptable to serial reducers.
Do not weaken original numeric, CLI/FD, witness, source, manager, or delivery
checks to produce those statuses.

### Required member-report fields

Worker `worker-preflight.json`, SCI `started.json` / `pilot-outcome.json`, and
audit `audit-report.json` retain their existing schemas and role-specific
scientific/runtime evidence, but carry the explicit statuses above. All carry
`producer_profile='grid400'`, `dataset_mode='append'`, `monomer_id`,
`campaign`, `execution_profile`, and the available `started_fields` bindings.
Workers carry `pilot_sha256`, actual worker `allocation`,
`operational_commit`, source/transport/runtime hashes and unchanged preflight
evidence; no registration/permit is fabricated for a worker.

SCI additionally retains actual SCI `allocation`, `compute_tag`,
`dataset_id`, `record_id`, `worker_preflight_sha256`, `pilot_sha256`,
`operational_commit`, `scientific_build_commit`, `manager`, `task_id`,
`runtime_input_sha256`, `runtime_unchanged=True`, `maximum_records=1`,
`find_existing=False`, `qcengine_internal_retries=0`, `tensors_sha256`,
`tensor_bytes`, and all original trace/source/delivery fields.

Every audit report requires:

* actual audit `allocation`, `case_job` (registered SCI job), `compute_tag`,
  `dataset_id`, `record_id`, `monomer_id`, `pilot_sha256`,
  `worker_preflight_sha256`, `case_outcome_sha256`;
* `scientific_build_commit`, `case_operational_commit`, `audit_commit`,
  `runtime_unchanged=True`, `execution_profile`;
* all non-None `started_fields(binding)` bindings plus
  `science_barrier_sha256` (the complete verified whole-SCI barrier);
* original `file_gate_sha256`, `numeric`, `envelope_identity`, `roles`,
  `traces`, and `captured_preservation` returned by `audit_preservation`.

Auditor checks its own allocation shape and SCI/outcome job independently.
The finalizer checks each member report's actual audit job/submission and
positive audit accounting, all original evidence/proof hashes, and exact
barrier SHA equality across all reports. A report cannot bless itself or a
failed/missing peer by setting success fields.

## Controller / finalizer barrier contract

One bounded wave, stop at first failure:

1. Verify cutover anchor, full original proofs and accounting/live agreement.
   Create separate immutable controller metadata, plan and reconstructed cases.
2. Admit the complete stage against **all user's embers jobs**, cap 50 with
   five slots reserved: `observed_used + new_stage_jobs <= 45`. HELD and
   unrelated jobs count. Never hardcode last-observed usage of two.
3. Submit / wait / positively verify all workers concurrently. Only then
   submit exactly the declared SCI jobs with `--hold`, durable one-shot
   submission intents/responses and actual route/held-state verification.
4. A separate 4CPU/24GiB/45min embers registration allocation performs the
   single-writer sequential inserts while SCI jobs remain HELD. Verify its
   allocation/batch/`.0` success and actual captures/registration before
   permit creation or any release.
5. Aggregate scratch admission retains every per-job check, sums conservative
   declared peak needs, and checks shared capacity/quota with headroom before
   release. Production adapters alone read capacity; tests inject observations.
   Recheck shared embers headroom / held identities. Exclusively write permit,
   durably record one release attempt per job, then release each once.
6. Wait / positively verify **all** SCI deliveries and accounting. No ninth
   concurrent SCI, no next wave, no record repair. Then run independent member
   audits concurrently; no member auditor publishes a checkpoint.
7. After **all** member audits/accounting succeed, submit one read-only
   finalizer (4CPU/24GiB/45min embers). It validates complete receipts,
   captures, actual serial anchor and subsequent wave links, accepted-prefix
   exact preservation, all audited completed members, and double-live
   agreement. Store all content-addressed proofs **before** checkpoint bytes.
8. Controller accepts only after finalizer allocation/batch/`.0` are positively
   completed0, proofs/publication validate, and its own live read agrees.
   Failed/ambiguous finalizer or live disagreement cannot advance state.

New checkpoint schema: `native-audited-bank-wave-checkpoint/1`; status
`READ_ONLY_AUDITED_WAVE_ADVANCE`. It binds unchanged bootstrap context,
`serial_anchor_sha256` plus the actual anchor bytes/proofs, exact predecessor,
ordered whole-wave links, actual final snapshot/hash, and audited-record order.
Generation changes by `len(member_ids)` **once** (normally +8), never per
member. No projected serial captures/checkpoints or invented intermediate
generations. Finalizer result `PASS_AUDITED_BANK_WAVE_ADVANCE` is not itself
controller acceptance.

## Validation and handoff

First-stage validation is document inspection / whitespace only; no production
or scientific claim. Implementation-stage local tests use explicit existing
home `p4_ci` / `p4qca` interpreters, metadata/mockHTTP/pure arithmetic only.
Required coverage includes eligible gaps/refusals/duplicates, predecessor/proof
tampering, actual job/tag/record binding, serial/wave reservation collisions,
one-shot HTTP/auth lost replies, real capture chains, held/registration/permit/
release ordering, shared cap/headroom, immutable peer preservation, all-audit
barrier, +8-only acceptance, failed finalizer/live disagreement and legacy
regression. Coordinator handles production activation and deployment.
