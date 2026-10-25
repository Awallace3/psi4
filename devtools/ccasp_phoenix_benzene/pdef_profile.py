# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Complete rank-2 pdef: a new fitted model, not angular-model parity."""
import copy
import hashlib
import json

PARENT_SHA = "19170c907137c7883dfbceaa4a23e4b919ec2d97dac38343dc4559b8bffb0f0b"
# Frozen real Racah rank 0..2 order; source equality is tested engine-free.
COMPONENT_NAMES = ("00", "10", "11c", "11s", "20", "21c", "21s", "22c", "22s")


def declared_variables():
    """36 noncharge upper-triangle variables per reference; never data-derived."""
    return tuple(f"{ref}_{COMPONENT_NAMES[i]}_{COMPONENT_NAMES[j]}_A"
                 for ref in ("C1", "H1") for i in range(1, 9) for j in range(i, 9))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def configuration(runtime):
    plan, here, old = runtime.PLAN, runtime.HERE, runtime.OLD
    native = dict(source=str(old.SOURCE), commit=old.BUILD_COMMIT, tree=old.BUILD_TREE,
                  stage=str(old.STAGE), core_sha256=old.CORE_SHA, python=str(old.ENV/"bin/python3.13"))
    allocation = dict(account="gts-cs207-chemx", partition="cpu-small", qos="embers",
        nodes=1, tasks=1, cpus=8, memory="48G", time="06:00:00", node="atl1-1-02-007-23-1",
        numerical_threads=1, psi4_memory_bytes=26843545600)
    overrides = {"isa.q_grid": [200, 974], "resources.psi4_memory_bytes": 26843545600,
                 "refinement.declared_variables": list(declared_variables())}
    if (plan["schema"] != "ccasp-benzene-full-pdef-plan/1"
            or plan["parent_profile_sha256"] != PARENT_SHA
            or runtime.sha(here/"pilot.json") != PARENT_SHA
            or plan["overrides"] != overrides
            or plan["native"] != native or plan["allocation"] != allocation
            or plan["activation_authority"] != "NONE; new explicit user approval required"):
        raise ValueError("exact explicit full pdef profile required")
    cfg = copy.deepcopy(json.loads((here/"pilot.json").read_text()))
    cfg["isa"]["q_grid"] = [200, 974]
    cfg["resources"]["psi4_memory_bytes"] = 26843545600
    cfg["refinement"]["declared_variables"] = list(declared_variables())
    files = ("pdef_profile.py", "pdef_driver.py", "pdef_job.py", "pdef_full.sbatch",
             "integrated_driver.py", "integrated_runtime.py", "angular_driver.py")
    identity = dict(schema="ccasp-benzene-full-pdef-effective-profile/1",
        semantics="new effective scientific model; static-unselected declarations fitted FREE, not fixed zero",
        parent_profile=dict(path="pilot.json", sha256=PARENT_SHA),
        declaration=dict(generator="pdef_profile.declared_variables/1",
            variables=list(declared_variables()), sha256=canonical_sha(declared_variables())),
        overrides=overrides, effective_config=cfg, effective_config_sha256=canonical_sha(cfg),
        plan_sha256=runtime.sha(here/"pdef_plan.json"),
        files_sha256={name: runtime.sha(here/name) for name in files})
    identity["identity_sha256"] = canonical_sha(identity)
    return cfg, identity
