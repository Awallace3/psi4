"""Separate diagnostic policy. Never call or weaken science.compare/summary."""
import itertools
from pathlib import Path
import re

import artifacts as a
import nonc1 as n
import policy as p
import science

MODE = "p157-numerical-repeatability-only"
REVIEW_ID = "parent-blinded-p157-repeatability-review-required"
RUN = p.P+"/runs/saptdft-cuest-cpks/major2-p157-repeatability-20261019-v2"
HARNESS = p.P+"/gits/psi4.saptdft-cuest-major2-p157-repeatability-20261019"
PARENT_PIN = "7046ed48cb7284cb1eaa71ec23bbac699dd68085"
PAIR_DIGEST = "c9d4ec615a37a6a50d493fcfa6c01cf567bea9b616f298ef1fd49fcea0e87a33"
COMPILED_IN_NEW_JOB = False
REPLACEMENT_FOR = "none-diagnostic-not-primary-replacement"
SUBMISSION_BUILD_ORIGINS = {
    "baseline": dict(origin_job="14165998", compiled_in_this_job=False),
    "candidate": dict(origin_job="14170562", compiled_in_this_job=False),
}
ALLOWLIST = {
    "baseline": dict(source=n.SOURCES["baseline"], pin=n.BASE, build=n.BUILDS["baseline"],
        metadata=p.P+"/runs/saptdft-cuest-cpks/major2-jkexception-20261018/build-baseline",
        origin_job="14165998", core_sha256="67c1e9021cc2fcaaf3b4ad3f98f6750ee39049c1aafe8cdfbe75b46c8335f224"),
    "candidate": dict(source=n.SOURCES["candidate"], pin=n.CANDIDATE, build=n.BUILDS["candidate"],
        metadata=n.RUN+"/build-candidate", origin_job="14170562",
        core_sha256="839e7bc2e52fe9d435689225fa718946710bb72d8147357ae3a8578fb8bdbf58"),
}
# 90 dense 1786-square FP64 arrays occupy 2.139 GiB before compression.
# Explicit total cap 2.25 GiB, including .npz files (no Db duplicate for RHF).
MATRIX_CAP = 2304 * 1024**2
PROCESS_CAP = MATRIX_CAP // 6
IDENTITY = dict(path="/storage/scratch1/3/awallace43/cpks-major2/14170562/gpu-identity",
                sha256="640c758d52c5c73a5a6df61ec061f6a0c382403639d6a18cd489cf807845f1c4")


def sequence():
    return [dict(arm=arm, label=f"p157-{arm}-r{rep}", repeat=rep,
                 system="protein157", backend="gpu", smoke=False)
            for arm, rep in [("baseline", 1), ("candidate", 1), ("candidate", 2),
                             ("baseline", 2), ("baseline", 3), ("candidate", 3)]]


def validate_pair(pair):
    p.require(pair["mode"] == "p157-repeatability-exact-pair", "explicit diagnostic pair")
    p.require(set(pair["arms"]) == set(ALLOWLIST), "exact two arms")
    for arm, allowed in ALLOWLIST.items():
        p.require(all(pair["arms"][arm].get(k) == v for k, v in allowed.items()), "wrong exact pair: "+arm)
    p.require(p.digest(pair) == PAIR_DIGEST, "immutable diagnostic pair digest")


def verify_shared(pair):
    validate_pair(pair)
    for group in ("shared_inputs", "origin_proofs"):
        for path, sha in pair[group].items():
            p.require(p.sha(path) == sha, "origin/shared evidence changed")
    core = pair["pyeinsums_core"]
    p.require(p.sha(core["path"]) == core["sha256"], "PyEinsums changed")
    for path, sha in p.LIBS.items():
        p.require(p.sha(path) == sha, "dependency changed")
    p.check_runtime_aliases()
    p.einsums_coherence()


def verify_artifact(pair, arm, files):
    """Full same-allocation rehash, no arbitrary directory adoption."""
    validate_pair(pair)
    item = pair["arms"][arm]
    p.check_source(item["source"], item["pin"], files)
    p.require(p.digest(files) == item["source_files_digest"], "whole source digest")
    build, meta = Path(item["build"]), Path(item["metadata"])
    p.require(build.is_dir() and not build.is_symlink(), "build retarget")
    p.require(a.hashed_tree(meta) == item["metadata_files"], "original metadata changed")
    p.require((meta/"BUILD_OK").read_text().strip() == item["pin"]+" cuda", "BUILD_OK")
    for rel, expected in item["build_files"].items():
        p.require(p.sha(build/rel) == expected, "original compiler input changed")
    p.require(a.hashed_tree(build/"stage") == item["stage_files"], "complete original stage changed")
    p.require(p.sha(build/"stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so") == item["core_sha256"], "core")
    p.require(len(p.check_installed_drivers(build/"stage", item["source"], files)) == 143, "143 drivers")
    p.require(p.digest(a.normalized_configs(item)) == pair["normalized_configs_digest"], "complete configs")
    p.require(p.digest(a.normalized_compile_commands(item)) == pair["normalized_commands_digest"], "compile database")


def validate(m):
    p.require(m["mode"] == MODE and m["approval"] == REVIEW_ID, "separate diagnostic mode")
    validate_pair(m["diagnostic_pair"])
    p.require(m["run_root"] == RUN and m["route"] == p.ROUTE and m["sequence"] == sequence(), "route/sequence")
    p.require(m["suite"] == p.H0+"/geometries" and m["suite_sha256"] == p.SUITE_SHA, "suite identity")
    p.require(m["gates"] == p.GATES, "fatal scientific gates unchanged")
    p.require(m["matrix_cap_bytes"] == MATRIX_CAP and m["per_process_timeout_seconds"] == 1800, "capture/budget")
    p.require(m["build_budget_seconds"] == [0, 0] and not m["accepted_speedup_data"], "no rebuild/benchmark")
    p.require(m["harness"]["path"] == HARNESS, "new harness only")
    p.require(m["identity_helper"] == IDENTITY and
              m["scratch_parent"] == "/storage/scratch1/3/awallace43/cpks-major2-p157", "identity/scratch policy")
    p.full_pin(m["harness"]["pin"])
    p.require(set(m["arms"]) == set(ALLOWLIST), "arm set")
    for arm, item in m["arms"].items():
        p.require(all(item[k] == ALLOWLIST[arm][k] for k in ("pin", "source", "build")), "arm pin/path")
        p.require(p.digest(item["files"]) == m["diagnostic_pair"]["arms"][arm]["source_files_digest"], "arm files")
    p.require(p.digest(m["harness"]["files"]) == m["harness"]["files_digest"], "harness manifest")


def trajectory(text):
    science.sections(text)  # unchanged chronology/grid gate
    result = []
    for stage, chunk in zip(science.STAGES, re.split(r"@DF-R(?:HF|KS) Final Energy:", text)[:5]):
        rows = []
        for line in chunk.splitlines():
            match = re.match(r"^\s*@DF-R(?:HF|KS) iter\s+(SAD|\d+):\s+(\S+)\s+(\S+)\s+(\S+)", line)
            if match:
                row, energy, delta, density = match.groups()
                values = list(map(float, (energy, delta, density)))
                p.require(all(science.finite(x) for x in values), "nonfinite trajectory")
                rows.append(dict(row=row, energy_eh=values[0], delta_eh=values[1], density_residual=values[2]))
        result.append(dict(stage=stage, rows=rows))
        p.require(rows and abs(rows[-1]["delta_eh"]) < 1e-9 and
                  abs(rows[-1]["density_residual"]) < 1e-8, "fatal final SCF convergence")
    return result


def first_trajectory_difference(a, b):
    p.require([x["stage"] for x in a] == [x["stage"] for x in b] == science.STAGES, "trajectory stages")
    for sa, sb in zip(a, b):
        for index, (ra, rb) in enumerate(itertools.zip_longest(sa["rows"], sb["rows"])):
            if ra != rb:
                fields = ["row_count"] if ra is None or rb is None else [k for k in ra if ra[k] != rb[k]]
                return dict(stage=sa["stage"], ordinal=index, fields=fields)
    return None


def numerical_report(records):
    """Explicit whitelist; original timer/result bytes never copied into this report."""
    p.require([x["label"] for x in records] == [x["label"] for x in sequence()], "all six, no cherry-picking")
    rows = []
    for row in records:
        d = row["data"]
        rows.append(dict(label=row["label"], arm=row["arm"], repeat=row["repeat"],
            scf_iterations=[x["iterations"] for x in d["proofs"]],
            solver_iterations=[x["iterations"] for x in d["cpks_solvers"]],
            final_residuals=[x["residuals"][-1] for x in d["cpks_solvers"]],
            components_hartree=d["components_hartree"]))
    pairs = []
    for arow, brow in itertools.combinations(rows, 2):
        delta = {k: brow["components_hartree"][k]-arow["components_hartree"][k] for k in science.COMPONENTS}
        p.require(all(abs(v) <= p.GATES["component_abs_eh"] for v in delta.values()), "fatal diagnostic energy delta")
        pairs.append(dict(a=arow["label"], b=brow["label"], within_arm=arow["arm"] == brow["arm"],
            component_delta_eh=delta,
            scf_iteration_drift=arow["scf_iterations"] != brow["scf_iterations"],
            solver_iteration_drift=arow["solver_iterations"] != brow["solver_iterations"]))
    p.require(all(max(v) < p.GATES["residual_max"] for row in rows for v in row["final_residuals"]), "fatal residual")
    return dict(mode=MODE, accepted_speedup_data=False, timings_blinded=True, records=rows, comparisons=pairs)
