# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Closed operational retry identity; the frozen complete-pdef science is unchanged."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

CONFIG_SHA = "088290e61602c39d4b51a805f24d30ccc6e1ef12f3d18c522fd3223e0cfce4fb"
PDEF_IDENTITY = "f7f6d2400088352cec7fe7c8f8c7878e675d3eb25cf5a90dce07fa6d3869be54"
PDEF_PLAN_SHA = "1684b7d8aed604d5b464fd527cbd6c572ac145d97cf2ed3e4ac3fce1d6b1c04c"
SOURCE = "/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-benzene-pdef-retry-20261026"
ROOT = "/storage/project/r-cs207-0/awallace43/runs/ccasp-benzene-pdef-retry-20261026"
SCRATCH = "/storage/scratch1/3/awallace43/ccasp-benzene-pdef-retry-20261026-full001"
AUTHORITY = "ONE fresh free full retry after preemption14439181; parent only; no automatic further retry"
GRANT = dict(schema="ccasp-benzene-pdef-retry-grant/1", prior_job="14439181",
    prior_outcome="preempted during native ISA/Q; own SCF passed; no response/PFIT nodes or observed scientific failure",
    fresh_allocations=1, automatic_further_retry=False, resumable_data=False,
    approval="user selected One free retry; unchanged full72 model and resource limits")


def configuration(runtime):
    """Validate the entire actual plan, not an arbitrary operational override."""
    here = runtime.HERE
    parent = runtime.module("_ccasp_retry_parent_profile", "pdef_profile.py")
    if runtime.sha(here/"pdef_plan.json") != PDEF_PLAN_SHA:
        raise ValueError("frozen parent pdef plan differs")
    parent_plan = json.loads((here/"pdef_plan.json").read_text())
    # Do not call the old profile on the retry runtime: it hashes its own plan.
    frozen = SimpleNamespace(PLAN=parent_plan, HERE=here, OLD=runtime.OLD, sha=runtime.sha)
    cfg, parent_identity = parent.configuration(frozen)
    if (parent_identity["identity_sha256"] != PDEF_IDENTITY
            or parent.canonical_sha(cfg) != CONFIG_SHA):
        raise ValueError("unchanged full pdef scientific configuration/parent identity required")
    expected = copy.deepcopy(parent_plan)
    expected.update(schema="ccasp-benzene-pdef-retry-plan/1",
        activation_authority=AUTHORITY, source=SOURCE, root=ROOT,
        runtime=ROOT+"/runtime-001", run=ROOT+"/full-001", scratch=SCRATCH,
        effective_profile_module="pdef_retry_profile.py",
        parent_pdef_identity_sha256=PDEF_IDENTITY, effective_config_sha256=CONFIG_SHA,
        retry_grant=GRANT)
    path = here/"pdef_retry_plan.json"
    actual = json.loads(path.read_text())
    if (Path(runtime.PLAN_PATH) != path
            or parent.canonical_sha(runtime.PLAN) != parent.canonical_sha(expected)
            or parent.canonical_sha(actual) != parent.canonical_sha(expected)):
        raise ValueError("exact closed approved retry plan/path/allocation required")
    files = ("pdef_retry_profile.py", "pdef_retry_driver.py", "pdef_retry_job.py",
        "pdef_retry_full.sbatch", "pdef_retry_plan.json", "pdef_profile.py",
        "pdef_plan.json", "pdef_driver.py", "pdef_job.py", "pdef_full.sbatch",
        "integrated_driver.py", "integrated_runtime.py", "angular_driver.py")
    identity = dict(schema="ccasp-benzene-pdef-retry-effective-profile/1",
        semantics="path-only fresh execution; unchanged complete72 scientific configuration",
        parent_pdef_identity_sha256=PDEF_IDENTITY, parent_pdef_profile=parent_identity,
        parent_profile=parent_identity["parent_profile"],
        effective_config=cfg, effective_config_sha256=CONFIG_SHA,
        plan=dict(path=path.name, sha256=runtime.sha(path)),
        plan_sha256=runtime.sha(path), operational_paths={k: expected[k] for k in
            ("source", "root", "runtime", "run", "scratch")},
        retry_grant=copy.deepcopy(GRANT),
        files_sha256={name: runtime.sha(here/name) for name in files})
    identity["identity_sha256"] = parent.canonical_sha(identity)
    return cfg, identity
