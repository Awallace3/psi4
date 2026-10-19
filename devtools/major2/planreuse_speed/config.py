"""Preregistered fixed-work policy. Stdlib only; never executes a device call."""
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = "f9a40cf26a6408b346508bef26b2d25cd7ed3c41"
sys.path.insert(0, str(HERE.parent))


def native_module(name):
    spec = importlib.util.spec_from_file_location("speed_native_"+name,
                                                 HERE.parent/"native_plan"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    if name == "handoff":
        # The inherited handoff uses its own stdlib common, not major2/handoff.
        sys.modules.setdefault("common", n)
    spec.loader.exec_module(module)
    return module


n = native_module("common")
require, sha, durable, load = n.require, n.sha, n.durable, n.load
P, E = n.P, n.E
SOURCE = P+"/gits/psi4.major2-planreuse-speed-20261019"
RUN = P+"/runs/saptdft-cuest-cpks/major2-planreuse-speed-20261019"
SCRATCH = "/storage/scratch1/3/awallace43/cpks-major2-planreuse-speed"
NATIVE_RUN = n.RUN
SEQUENCE = [dict(arm=a, repeat=i, label=f"speed-{a}-{i}") for a, i in (
    ("baseline", 1), ("candidate", 1), ("candidate", 2),
    ("baseline", 2), ("baseline", 3), ("candidate", 3))]
CALLS = dict(initialize=6, compute=11, finalize=1, C_left_add=1, C_right_add=0)
CAP = 2*1024**3
PROCESS_CAP = 110*1024**2
POLICY = dict(
    scope="actual-wrapper fixed-C J/K plan reuse, NOT RHF finalize or whole SAPT",
    sequence=SEQUENCE, rounds=5, warmup_computes=1, computes=66,
    calls_per_process=CALLS, baseline_creations=6, candidate_creations=1,
    candidate_reuses=5, synchronization="checked cudaDeviceSynchronize",
    clock="time.perf_counter_ns", warmup_in_primary=False,
    primary="median of three process medians, initialize+compute, BASE/CAND",
    secondary="three balanced adjacent pair BASE/CAND ratios; same-plan compute control",
    filtering="none", inference="descriptive n=3 per arm, no CI/significance/generalization",
    matrix_cap=CAP, stored_snapshots=["warmup", "round5-control"],
    factor_generation=0, cpu_reference=False, builds=0, numerical_overrides={},
    findings_bounds=dict(matrix_maxabs=1e-8, matrix_relative_frobenius=5e-12,
                         scalar_abs_eh=1e-8, symmetry_maxabs=1e-10),
    actual_D_must_match=True,
    candidate_stability="original findings bounds; breach withholds performance acceptance",
    original_full_SAPT_gate="unchanged, unresolved; this does not replace it",
    child_budget_seconds=1800, analysis_budget_seconds=1200, teardown_seconds=300)


def source_check(m, root=SOURCE):
    require(n.output("git", "-C", root, "rev-parse", "HEAD") == m["pin"], "source pin")
    require(not n.output("git", "-C", root, "status", "--porcelain",
                         "--untracked-files=no"), "dirty source")
    names = n.output("git", "-C", root, "diff", "--name-status", PARENT, m["pin"]).splitlines()
    require(names and all(v.startswith("A\tdevtools/major2/planreuse_speed/") for v in names),
            "only owned additions")
    require(sorted(v.split("\t")[1] for v in names) == sorted(m["helpers"]), "owned helper set")
    for rel, h in m["helpers"].items():
        require(sha(Path(root)/rel) == h, "helper bytes: "+rel)


def manifest(path, digest):
    require(sha(path) == digest, "manifest hash")
    m = load(path)
    require(m["policy"] == POLICY and m["parent"] == PARENT and
            m["source"] == SOURCE and m["run_root"] == RUN and
            m["scratch_parent"] == SCRATCH, "speed policy identity")
    import fixed_jk as f
    import policy as p
    import repeatability as r
    r.validate_pair(m["diagnostic_pair"])
    require(m["settings"] == dict(f.SETTINGS, computes_per_process=11) and
            m["frozen_factor"] == f.FROZEN_FACTOR and
            m["sequence"] == SEQUENCE and m["route"] == p.ROUTE and
            m["gates"] == p.GATES and m["suite"] == p.H0+"/geometries" and
            m["suite_sha256"] == p.SUITE_SHA and
            m["identity_helper"] == r.IDENTITY and
            m["harness"]["pin"] == m["pin"] and m["harness"]["path"] == SOURCE,
            "inherited exact pair/settings/runtime identity")
    require(set(m["arms"]) == set(r.ALLOWLIST), "exact arms")
    for arm, item in m["arms"].items():
        require(all(item[k] == r.ALLOWLIST[arm][k] for k in ("source", "pin", "build")) and
                p.digest(item["files"]) == m["diagnostic_pair"]["arms"][arm]["source_files_digest"],
                "exact scientific source")
    require(p.digest(m["harness"]["files"]) == m["harness"]["files_digest"], "harness file digest")
    require(len(m["preservation"]["roots"]) == 16, "preserve sixteen actual old roots")
    return m


def allocated():
    n.allocation_only()
    require(os.environ.get("SLURM_JOB_QOS") == "embers" and
            os.environ.get("SLURM_CPUS_PER_TASK") == "8" and
            bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "exact GPU allocation only")
