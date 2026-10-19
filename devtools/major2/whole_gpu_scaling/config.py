"""Preregistered whole-energy observational cohort; stdlib only."""
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))


def inherited(directory, name):
    spec = importlib.util.spec_from_file_location(
        "whole_inherited_"+directory+"_"+name, HERE.parent/directory/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n = inherited("native_plan", "common")
import policy as p

require, sha, durable, load = n.require, n.sha, n.durable, n.load
P, E = n.P, n.E
PARENT = "8af4a8bf034c16c0c01821b8442673db0f54514f"
SOURCE = P+"/gits/psi4.major2-whole-gpu-print3-repair-20261019"
REJECTED_RUN = P+"/runs/saptdft-cuest-cpks/major2-whole-gpu-scaling-20261019"
FAILED_RUN = REJECTED_RUN+"-v2"
FAILED_SOURCE = P+"/gits/psi4.major2-whole-gpu-scaling-20261019"
FAILED_PIN = "6ae8a733279f6b830775fc0ceb7061a7ec3a6f9a"
RUN = P+"/runs/saptdft-cuest-cpks/major2-whole-gpu-scaling-20261019-v3"
SPEED_RUN = P+"/runs/saptdft-cuest-cpks/major2-planreuse-speed-20261019"
SCRATCH = "/storage/scratch1/3/awallace43/cpks-major2-whole-gpu-print3-repair-20261019-v3"
SEQUENCE = [dict(v, label=v["label"].replace("major2-", "whole-")) for v in p.sequence()]
CAPS = dict(peptide=180, nanotube=300, protein83=600, protein157=1500)
ROUTE = dict(p.ROUTE, seconds=21600)
POLICY = dict(
    scope="new whole-energy observational cohort; not replacement acceptance for failed primary",
    calls=24, fresh_processes=24, calls_per_process=1, n_per_arm_system=3,
    sequence=SEQUENCE, method="sapt(dft)-d4(i)", warmups=0, builds=0,
    cpu_runs=0, sampler=False, filtering="none",
    timing="unchanged runner: perf_counter immediately around energy(), synchronous return",
    print_options={"PRINT": 3, "SCF__PRINT": 3, "SAPT__PRINT": 3},
    inherited_gates=p.GATES, cross_arm_action="collect all individually valid then classify all 9",
    failed_gate_qualification="numerical/iteration matched-work acceptance unresolved",
    forced_cache_reuse=False, lifecycle_changes=False, matrix_observer=False,
    child_caps=CAPS, prolog_seconds=2400, validation_seconds=60,
    analysis_seconds=1200, teardown_seconds=300, allocation_seconds=21600,
    inference="descriptive n=3; no CI/significance or pure cache causality",
    output_cap_bytes=64*1024**2, record_cap_bytes=16*1024**2)


def allocated():
    n.allocation_only()
    require(os.environ.get("SLURM_JOB_QOS") == "embers" and
            os.environ.get("SLURM_CPUS_PER_TASK") == "8" and
            bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "exact GPU allocation only")


def remaining_budget(index):
    """Reserve all unstarted children, validations, analysis and teardown."""
    require(0 <= index <= 24, "schedule index")
    return (sum(CAPS[v["system"]]+POLICY["validation_seconds"] for v in SEQUENCE[index:])
            + POLICY["analysis_seconds"]+POLICY["teardown_seconds"])


def admit(seconds, index, prolog=False):
    import math
    need = remaining_budget(index)+(POLICY["prolog_seconds"] if prolog else 0)
    require(math.isfinite(seconds) and seconds >= need, "insufficient whole-cohort budget")


def source_check(m, root=SOURCE):
    require(p.git(root, "rev-parse", "HEAD") == m["pin"], "source pin")
    require(not p.git(root, "status", "--porcelain", "--untracked-files=no"), "dirty source")
    changes = p.git(root, "diff", "--name-status", PARENT, m["pin"]).splitlines()
    require(changes and all(v.startswith("A\tdevtools/major2/whole_gpu_scaling/")
                            for v in changes), "owned additions only")
    require(sorted(v.split("\t")[1] for v in changes) == sorted(m["helpers"]), "owned set")
    for rel, digest in m["helpers"].items():
        require(sha(Path(root)/rel) == digest, "owned source bytes: "+rel)


def manifest(path, digest):
    require(sha(path) == digest, "manifest bytes")
    m = load(path)
    import repeatability as r
    r.validate_pair(m["diagnostic_pair"])
    require(m["policy"] == POLICY and m["parent"] == PARENT and
            m["source"] == SOURCE and m["run_root"] == RUN and
            m["scratch_parent"] == SCRATCH and m["route"] == ROUTE and
            m["sequence"] == SEQUENCE and m["systems"] == p.SYSTEMS and
            m["suite"] == p.H0+"/geometries" and m["suite_sha256"] == p.SUITE_SHA and
            m["gates"] == p.GATES and m["identity_helper"] == r.IDENTITY and
            m["build_budget_seconds"] == [0, 0], "whole policy identity")
    require(m["harness"]["pin"] == m["pin"] and m["harness"]["path"] == SOURCE and
            p.digest(m["harness"]["files"]) == m["harness"]["files_digest"], "harness identity")
    require(set(m["arms"]) == set(r.ALLOWLIST), "exact arms")
    for arm, v in m["arms"].items():
        require(all(v[k] == r.ALLOWLIST[arm][k] for k in ("source", "pin", "build")) and
                p.digest(v["files"]) == m["diagnostic_pair"]["arms"][arm]["source_files_digest"],
                "exact original scientific pair")
    require(m["compiler"]["actual"] == n.CXX_REAL and
            m["compiler"]["actual_sha256"] == n.CXX_REAL_SHA and
            m["compiler"]["frontend"] == n.CC1PLUS and
            m["compiler"]["frontend_sha256"] == n.CC1PLUS_SHA and
            m["native_libraries"].get(n.SDK+"/lib/13/libcuest.so.0.2.50") == n.SDK_SHA and
            m["native_libraries"].get(p.RUNTIME_LIBS["libcudart.so"]["realpath"]) ==
                p.RUNTIME_LIBS["libcudart.so"]["sha256"], "compiler/native pins")
    require(len(m["preservation"]["roots"]) == 18, "eighteen actual old roots")
    from provenance import verify_runner
    require(m["runner_provenance"] == verify_runner(ROOT), "frozen exact runner delta")
    return m
