# One explicitly paid nonpreemptible complete-pdef execution

**Local engine-free implementation only.** Parent owns independent review/tests,
Git publication, isolated staging, fresh price/route/storage preflight and at most
**ONE** submission. No scheduler call is made by these adapters. No fallback to
another account, QoS, driver or batch; no chains, automatic retry, watchdog,
extra native probe/smoke allocation, cleanup, overwrite or old-path reuse.

Authority is the separate user grant
`/home/awallace43/docs/camcasp/ccasp-benzene-pdef-inferno-grant.md`:
one gts-cs207-chemx/cpu-small/inferno allocation, 8CPU/48G/6h, same native-build
node `atl1-1-02-007-23-1`, up to **USD0.39**. Parent must recheck the actual
price/account balance/route immediately before submission and record approval
externally. Do not switch to gts-cs207 or materially exceed the cap; stop for
renewed authority instead. No account-selector fallback is implemented.

Two free corrected jobs14439181/14456809 were preempted inside ISA/Q, not at
a scientific refusal. Their sources, runs, stages, scratch and receipts remain
frozen. No resumable full-chain data exists. Native PFIT smoke14417619's
parent-audited eleven-node synthetic pass is historical evidence for the
declaration/solver mechanism only, not molecular LW/PFIT or this runtime.

## Separate execution, identical science

Explicit opt-in files: `pdef_inferno_plan.json`, `pdef_inferno_profile.py`,
`pdef_inferno_driver.py`, `pdef_inferno_job.py`, `pdef_inferno_full.sbatch`.
All prior tracked files, including production `psi4/`, are immutable.
`integrated_runtime.for_plan` creates a separate namespace; no global PLAN,
default, function replacement or production monkeypatch.

The old full-pdef profile both rejects inferno and hashes its own old plan.
The new profile therefore derives cfg from the immutable full-pdef plan using
a separate `SimpleNamespace` parent view. It verifies the frozen parent's
identity and science hash, then independently checks **the entire actual inferno
disk JSON and namespace PLAN** against the parent plus the closed approved
operational changes. PLAN_PATH must name the inferno plan. Unknown fields,
different paths, grant, account, QoS, resources, numerical settings, fixture,
native/probe/fingerprint pins or scientific overrides refuse. No arbitrary
operational injection interface.

* Exact canonical scientific cfg SHA:
  `088290e61602c39d4b51a805f24d30ccc6e1ef12f3d18c522fd3223e0cfce4fb`.
* Frozen full-pdef parent:
  `f7f6d2400088352cec7fe7c8f8c7878e675d3eb25cf5a90dce07fa6d3869be54`.
* Historical free retry:
  `f83618a3ba469f459747483da815a196e06e0fe3a682357f300ba1872c0459b1`.

Neither historical identity is the new execution identity. The latter hashes
the **actual inferno JSON bytes**, new adapter/batch bytes, closed operational
paths, explicit inferno allocation, unique one-paid-grant identity and original
parent scientific identity. Source binding additionally pins actual commit/tree,
all tools and routing. Whitespace-only plan byte changes alter execution identity
even if closed semantic validation still succeeds.

The new driver does NOT call the old hardcoded capture wrapper. It loads its
own profile and calls unchanged `integrated_driver.capture_science` with
`configuration=cfg`, `profile_identity=new_identity` and existing
`pdef_driver.validate_complete`. Tests compare actual wrapper configuration kwargs
exactly with **both** full-pdef and free-retry wrappers, and exercise the actual
shared driver through source-only synthetic oeprop/constructor boundaries.

Unchanged science: all72 noncharge declared rank1/2 upper-triangle C1/H1 variables,
COPY ownership and frames, static selection/current anchors/penalties (unselected
extras remain fitted **FREE**, not fixed-zero), PBE0/noGRAC, original private
geometry and basis files, MAIN414sph/AUX1146cart, SCF75/302, ISA iteration100/200,
Q200/974, response100/200, numerical thread1 and25GiB for both Psi4 and
BoundedResources,11nodes, PFIT2000points and both packed all-pair passes,
strict physical LW1e-6, native Solved/residual/reciprocity/property gates,
isotropic and recoupled output, work6.4e13, I/O64GiB. No projection,
renormalization, normalization substitution, gate relaxation or model fallback.
Rank2 C10 is still not unrestricted-complete C10.

Existing resource tests recalculate ISA26570915696bytes under26843545600
(headroom272629904), actual72PFIT326608384bytes/94490544000work,
fullchain source-work bound62337903183272 and I/O6968466000bytes. These are
unchanged numerical/source plans, **not measured RSS, timing, six-hour
completion or physical-accuracy guarantees**. Scheduler48G includes dirty cache
and is separate from numerical25GiB. Keep400GiB free scratch admission.

## Parent-owned future preparation

Not performed by the implementation worker. Publish through Git only; stage
the reviewed clean commit/tree in the exclusive new source:

```text
/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-benzene-pdef-inferno-20261026
```

Root:
`/storage/project/r-cs207-0/awallace43/runs/ccasp-benzene-pdef-inferno-20261026`;
children `runtime-001`, `full-001`; scratch:
`/storage/scratch1/3/awallace43/ccasp-benzene-pdef-inferno-20261026-full001`.
All must be new, including root. Retain partial paths and stop on any failure;
no automatic preparation recovery, overwrite, cleanup or reuse.

Use actual reviewed C/T from the external handover:

```text
/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt/bin/python3.13
  -I -S -B
  /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-benzene-pdef-inferno-20261026/devtools/ccasp_phoenix_benzene/pdef_inferno_job.py
  prepare --commit C --tree T
```

Preparation reuses exact source/core-role, original entire private probe,
fixture and configured runtime composition/verification admissions. It
materializes bounded small configured Python/launcher files, links immutable
accepted core/data, writes **actual** runtime and derivative hashes and creates
absolute log parents. No native import, build, numerical calculation, core
hash measurement, scheduler call or price query. Source/core/GBS trees and original
probe measurement are retained separately from new derivative provenance.

Inspect `full-001/metadata/preparation.json`. Run only its exact concrete
`test_only_argv`; `future_submit_argv_not_executed` differs by `--parsable`
instead of `--test-only`. Both contain the actual C/T/runtime/derivative pins:

```text
/opt/slurm/current/bin/sbatch --test-only --export=NONE --chdir=NEW_FULL_RUN
  NEW_SOURCE/devtools/ccasp_phoenix_benzene/pdef_inferno_full.sbatch
  C T GENERATED_RUNTIME_SHA GENERATED_DERIVATIVE_SHA
```

Generated remote hashes are unavailable locally; do not invent them. Check
exact paid route, balance/price/cap, native/source/private evidence, writable
storage and400GiB free scratch. No new allocation is needed for test-only.
Before **one** submission, parent records fresh price/route approval and
consumes the grant in its external submission record. Ambiguous scheduler
responses are NOT authority to retry.

## Required external price approval record

Prepare deliberately does **not** write
`full-001/metadata/paid-price-approval.json`. Parent must exclusively create
that external record **before submitting**. No static quote, timestamp or job ID
is native evidence. The exact required schema has these fields (notation below
describes values; it is not a ready approval or live-price observation):

| Field | Required value |
|---|---|
| schema | `ccasp-benzene-pdef-inferno-price-approval/1` |
| grant_identity | `ccasp-benzene-pdef-inferno-20261026-one-paid-001` |
| approved_by | `parent` |
| approved_for_single_submission | JSON `true` |
| automatic_further_retry | JSON `false` |
| commit / tree | actual reviewed source pins from preparation binding |
| execution_identity_sha256 | preparation binding's effective profile identity |
| runtime_manifest_sha256 / derivative_sha256 | actual preparation pins |
| allocation | exact full inferno plan allocation object, including numeric settings |
| quoted_maximum_usd | fresh worst-case USD quote as decimal string, `0` through `0.39` |
| checked_at_utc | actual parent's fresh check, `YYYY-MM-DDTHH:MM:SSZ` |
| price_route_evidence | nonempty reference to parent's actual price/balance/route evidence |

Unknown fields/wrong route/pins/grant/excess price refuse. The job uses bounded
regular no-follow/nonblocking reads, records the actual approval hash and
checks it again at final closure. This is payload admission, **not a scheduler
spending lock**: a missing record discovered inside an allocation can already
incur charges. The parent must enforce freshness and the one-allocation cap
externally before submission; formatting validation cannot attest the truth or
freshness of a quote. No code queries cost, switches accounts or submits jobs.

## Runtime, failure and acceptance limits

Inherited exact isolated p4cuest_sapt interpreter/dependencies, retained Einsums
CDLL, core/data/package roles, original probe and complete measured normalized
shell comparisons before SCF, post-SCF MAIN and final loader/runtime/source closure
remain mandatory. No change to admitted native stage or dependency prefix.
Read `INTEGRATED_CAPTURE.md`, `FULL_PDEF.md`, `PDEF_RETRY.md`,
`ANGULAR_FULL.md`, `BASIS_ADMISSION.md`, `SCIENCE_JOB.md` and `BUILD_SMOKE.md`
for their specific contracts; older activation restrictions are historical and
do not grant another job. This separate paid grant supersedes no science gate.

The batch is the old TERM-safe background-child/wait boilerplate with only
explicit QoS, names/paths/job selector and paid authority comment changed.
`--no-requeue`, honest nonzero exceptions, primary traceback retention, separate
closure errors, live handle through final closure and payload/batch terminal
receipts remain. Existing captures are **pre-LW**, not guaranteed post-LW anchors,
point targets or resumable PFIT state. Termination may prevent payload/final
closure receipts; absence is incomplete evidence, never success.

No local native/current-runtime/molecular acceptance exists. Complete acceptance
needs attributable all11physical nodes, native Solved/full2000point replay,
strict original physical gates, accepted exports and final closure plus
unsignalled zero payload/child/batch receipts. Scheduler COMPLETED, synthetic
tests, captures or resource plans alone are insufficient.

## Engine-free checks

```bash
export PYTHONDONTWRITEBYTECODE=1
P=/home/awallace43/miniconda3/envs/p4_ci/bin/python
"$P" -B -m unittest discover -s devtools/ccasp_phoenix_benzene -p 'test_*.py' -v
PYTHONPATH=tests/pytests "$P" -B -m unittest \
  test_isapol_bounded_campaign_source test_isapol_selection_source \
  test_isapol_scheme_source test_isapol_property_water_source -v
for script in devtools/ccasp_phoenix_benzene/*.sbatch; do bash -n "$script"; done
git diff --check
git diff --cached --check
```

These use fake core/source-only boundaries, never real Psi4/core/adcc/Einsums.
After normal local commit and external handover the implementation worker exits;
no idle agents, native probe, lifecycle experiment or automatic continuation.
