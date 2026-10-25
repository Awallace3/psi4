# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Exact two-field effective profile; original pilot/probe identities stay intact."""
import copy
import hashlib
import json

PARENT_SHA = "19170c907137c7883dfbceaa4a23e4b919ec2d97dac38343dc4559b8bffb0f0b"
OVERRIDES = {"isa.q_grid": [200, 974], "resources.psi4_memory_bytes": 26843545600}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode()).hexdigest()


def configuration(runtime):
    plan, here = runtime.PLAN, runtime.HERE
    old = runtime.OLD
    native = dict(source=str(old.SOURCE),commit=old.BUILD_COMMIT,tree=old.BUILD_TREE,
                  stage=str(old.STAGE),core_sha256=old.CORE_SHA,python=str(old.ENV/"bin/python3.13"))
    allocation = dict(account="gts-cs207-chemx",partition="cpu-small",qos="embers",
        nodes=1,tasks=1,cpus=8,memory="48G",time="06:00:00",node="atl1-1-02-007-23-1",
        numerical_threads=1,psi4_memory_bytes=26843545600)
    if (plan["schema"] != "ccasp-benzene-angular-full-plan/1"
            or plan["parent_profile_sha256"] != PARENT_SHA
            or runtime.sha(here/"pilot.json") != PARENT_SHA
            or plan["overrides"] != OVERRIDES
            or plan["native"] != native or plan["allocation"] != allocation):
        raise ValueError("exact explicit angular full profile required")
    parent = json.loads((here/"pilot.json").read_text())
    cfg = copy.deepcopy(parent)
    cfg["isa"]["q_grid"] = list(OVERRIDES["isa.q_grid"])
    cfg["resources"]["psi4_memory_bytes"] = OVERRIDES["resources.psi4_memory_bytes"]
    files = ("angular_profile.py","angular_driver.py","angular_job.py","angular_full.sbatch",
             "integrated_driver.py","integrated_runtime.py")
    identity = dict(schema="ccasp-benzene-angular-effective-profile/1",
        parent_profile=dict(path="pilot.json",sha256=PARENT_SHA),
        overrides=copy.deepcopy(OVERRIDES),effective_config=cfg,
        effective_config_sha256=canonical_sha(cfg),
        plan_sha256=runtime.sha(here/"angular_plan.json"),
        files_sha256={name:runtime.sha(here/name) for name in files})
    identity["identity_sha256"] = canonical_sha(identity)
    return cfg,identity
