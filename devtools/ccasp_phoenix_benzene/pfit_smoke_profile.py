# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Synthetic-only identity; the complete molecular pdef profile is NOT activated."""
import json
from pathlib import Path

import importlib.util

BASE = "d1b23d019d59d30b6ca9cff9018177cac285d341"
DECLARATION_SHA = "f3ed7ee6ab75cdfc090c7f246c48bf4d4f00a22b3f1aa30c1dc5abeec5fcd28e"
PDEF_PROFILE_SHA = "027e5a69d295e881a025a74e029c30a322e90c974c5a2baef9572cab2850f273"
PDEF_PLAN_SHA = "1684b7d8aed604d5b464fd527cbd6c572ac145d97cf2ed3e4ac3fce1d6b1c04c"


def declaration(here):
    spec = importlib.util.spec_from_file_location("_smoke_complete_pdef", Path(here)/"pdef_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = module.declared_variables()
    frozen = json.loads((Path(here)/"pdef_plan.json").read_text())["overrides"]["refinement.declared_variables"]
    if list(labels) != frozen or module.canonical_sha(labels) != DECLARATION_SHA:
        raise ValueError("original complete declaration differs")
    return labels


def configuration(runtime):
    here, plan = runtime.HERE, runtime.PLAN
    # Explicit separate operational plan. All values are closed, not overrides
    # of the molecular allocation or a mutable shared PLAN.
    prefix = "/storage/project/r-cs207-0/awallace43"
    source = prefix+"/gits/psi4.ccasp-pdef-native-smoke-20261025"
    root = prefix+"/runs/ccasp-pdef-native-smoke-20261025"
    allocation = dict(account="gts-cs207-chemx", partition="cpu-small", qos="embers",
        nodes=1, tasks=1, cpus=2, memory="8G", time="00:30:00",
        node="atl1-1-02-007-23-1", numerical_threads=1, psi4_memory_bytes=4*1024**3)
    procedure = dict(identity="synthetic-complete72-static-mask-default-pfit/1",
        points=32, channels=108, nodes=11, max_parameters=72, block_rows=37, seed=20261025,
        max_bytes=4*1024**3, max_work=48_000_000_000, max_io_bytes=32*1024**2,
        max_output_files=48, max_output_bytes=16*1024**2, max_snapshot_bytes=256*1024,
        max_design_condition=100., max_normal_condition=10000.,
        parameter_atol=1e-10, parameter_rtol=1e-9,
        objective_atol=1e-12, objective_rtol=1e-8)
    old = json.loads((here/"pdef_plan.json").read_text())
    expected = dict(schema="ccasp-synthetic-pfit-smoke-plan/1", source=source,
        runtime=root+"/runtime-001", run=root+"/smoke-001",
        scratch="/storage/scratch1/3/awallace43/ccasp-pdef-native-smoke-20261025-smoke001",
        effective_profile_module="pfit_smoke_profile.py", allocation=allocation,
        procedure=procedure, prior_driver_files=old["prior_driver_files"], routing=old["routing"])
    if plan != expected:
        raise ValueError("exact synthetic smoke plan required")
    if runtime.sha(here/"pdef_profile.py") != PDEF_PROFILE_SHA or runtime.sha(here/"pdef_plan.json") != PDEF_PLAN_SHA:
        raise ValueError("original pdef provenance differs")
    labels = declaration(here)
    native = dict(source=str(runtime.OLD.SOURCE), commit=runtime.OLD.BUILD_COMMIT,
        tree=runtime.OLD.BUILD_TREE, stage=str(runtime.OLD.STAGE),
        core_sha256=runtime.OLD.CORE_SHA, python=str(runtime.OLD.ENV/"bin/python3.13"))
    if native != old["native"]:
        raise ValueError("accepted native identity differs")
    identity = dict(schema="ccasp-synthetic-pfit-smoke-profile/1", base=BASE,
        procedure=procedure, allocation=allocation, native=native,
        declaration_provenance=dict(commit=BASE, generator="pdef_profile.declared_variables/1",
            profile_sha256=PDEF_PROFILE_SHA, plan_sha256=PDEF_PLAN_SHA,
            declaration_sha256=DECLARATION_SHA, variables=list(labels)),
        untested=["molecular basis/frame admission", "SCF", "ISA", "H2H1", "LW",
                  "real benzene", "full cloud 2000", "all11 physical response nodes"],
        plan_sha256=runtime.sha(here/"pfit_smoke_plan.json"),
        files_sha256={name: runtime.sha(here/name) for name in (
            "pfit_smoke_profile.py", "pfit_smoke_driver.py", "pfit_smoke_job.py",
            "pfit_smoke.sbatch", "integrated_runtime.py")})
    # Canonical hashing reused without importing a scientific package.
    spec = importlib.util.spec_from_file_location("_smoke_pdef_hash", here/"pdef_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity["identity_sha256"] = module.canonical_sha(identity)
    return procedure, identity
