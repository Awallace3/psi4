# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Closed single paid execution identity; frozen complete-pdef science."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

CONFIG_SHA = "088290e61602c39d4b51a805f24d30ccc6e1ef12f3d18c522fd3223e0cfce4fb"
PDEF_IDENTITY = "f7f6d2400088352cec7fe7c8f8c7878e675d3eb25cf5a90dce07fa6d3869be54"
RETRY_IDENTITY = "f83618a3ba469f459747483da815a196e06e0fe3a682357f300ba1872c0459b1"
PDEF_PLAN_SHA = "1684b7d8aed604d5b464fd527cbd6c572ac145d97cf2ed3e4ac3fce1d6b1c04c"
SOURCE = "/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-benzene-pdef-inferno-20261026"
ROOT = "/storage/project/r-cs207-0/awallace43/runs/ccasp-benzene-pdef-inferno-20261026"
SCRATCH = "/storage/scratch1/3/awallace43/ccasp-benzene-pdef-inferno-20261026-full001"
AUTHORITY = "ONE paid nonpreemptible inferno allocation up to USD0.39; parent price/route recheck required; no automatic further retry"
GRANT = dict(schema="ccasp-benzene-pdef-inferno-grant/1",
    identity="ccasp-benzene-pdef-inferno-20261026-one-paid-001",
    prior_jobs=["14439181", "14456809"],
    prior_outcome="both preempted inside ISA/Q; no observed scientific failure; frozen",
    fresh_allocations=1, automatic_further_retry=False, resumable_data=False,
    maximum_usd="0.39", parent_price_route_approval_required=True,
    approval="user selected Yes, one inferno run; unchanged full72 science/resources")


def configuration(runtime):
    """Check the entire real JSON and namespace against one approved closed plan."""
    here = runtime.HERE
    parent = runtime.module("_ccasp_inferno_parent_profile", "pdef_profile.py")
    if runtime.sha(here/"pdef_plan.json") != PDEF_PLAN_SHA:
        raise ValueError("frozen parent pdef plan differs")
    parent_plan = json.loads((here/"pdef_plan.json").read_text())
    # The original profile rejects inferno and hashes its own old plan.
    # Derive science on the immutable parent view, never on the inferno PLAN.
    frozen = SimpleNamespace(PLAN=parent_plan, HERE=here, OLD=runtime.OLD, sha=runtime.sha)
    cfg, parent_identity = parent.configuration(frozen)
    if (parent_identity["identity_sha256"] != PDEF_IDENTITY
            or parent.canonical_sha(cfg) != CONFIG_SHA):
        raise ValueError("unchanged full pdef scientific configuration/parent identity required")
    expected = copy.deepcopy(parent_plan)
    expected.update(schema="ccasp-benzene-pdef-inferno-plan/1",
        activation_authority=AUTHORITY, source=SOURCE, root=ROOT,
        runtime=ROOT+"/runtime-001", run=ROOT+"/full-001", scratch=SCRATCH,
        effective_profile_module="pdef_inferno_profile.py",
        parent_pdef_identity_sha256=PDEF_IDENTITY, historical_retry_identity_sha256=RETRY_IDENTITY,
        effective_config_sha256=CONFIG_SHA, paid_grant=GRANT)
    expected["allocation"]["qos"] = "inferno"
    path = here/"pdef_inferno_plan.json"
    actual = json.loads(path.read_text())
    if (Path(runtime.PLAN_PATH) != path
            or parent.canonical_sha(runtime.PLAN) != parent.canonical_sha(expected)
            or parent.canonical_sha(actual) != parent.canonical_sha(expected)):
        raise ValueError("exact closed approved inferno plan/path/allocation required")
    files = ("pdef_inferno_profile.py", "pdef_inferno_driver.py", "pdef_inferno_job.py",
        "pdef_inferno_full.sbatch", "pdef_inferno_plan.json", "pdef_profile.py",
        "pdef_plan.json", "pdef_driver.py", "pdef_job.py", "pdef_full.sbatch",
        "integrated_driver.py", "integrated_runtime.py", "angular_driver.py")
    identity = dict(schema="ccasp-benzene-pdef-inferno-effective-profile/1",
        semantics="one paid execution; QoS/path/grant-only change; unchanged complete72 science",
        parent_pdef_identity_sha256=PDEF_IDENTITY, parent_pdef_profile=parent_identity,
        historical_retry_identity_sha256=RETRY_IDENTITY,
        parent_profile=parent_identity["parent_profile"],
        effective_config=cfg, effective_config_sha256=CONFIG_SHA,
        plan=dict(path=path.name, sha256=runtime.sha(path)), plan_sha256=runtime.sha(path),
        operational_paths={k: expected[k] for k in ("source", "root", "runtime", "run", "scratch")},
        allocation=copy.deepcopy(expected["allocation"]), paid_grant=copy.deepcopy(GRANT),
        files_sha256={name: runtime.sha(here/name) for name in files})
    identity["identity_sha256"] = parent.canonical_sha(identity)
    return cfg, identity
