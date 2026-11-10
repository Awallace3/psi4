#!/usr/bin/env python3
"""Publish the verified fixed-GRAC coupled suite; never run chemistry.

Archive digest pins the complete 119-file independently verified snapshot, including
excluded smoke records. Semantic checks remain separate so future snapshots cannot
silently inherit an induction-NONE adapter. Assertions are intentional fail-closed
publication guards: run without python -O.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

# Evidence trees (including snapshot01) must never acquire import bytecode.
sys.dont_write_bytecode = True
if not __debug__:
    raise RuntimeError("Publication requires assertions; never run with -O")
KEY = "coupled_scaling_suite"
SOURCE = "scal631-20261016-snapshot-01"
CONTINUATION_SOURCE = "scal631-20261017-snapshot-02"
CONTINUATION_SHA = "08b0f2bc652ede6380991ebbab8ca714e8e9926c8fecfcf8cbd65100668b269c"
CONTINUATION_FILES = {
    "inventory.json": "d969515b8bf5993e0cdb41e1adeee64f717e305b9a179b3d3593002c8df76686",
    "raw-sha256.txt": "a4e15127328d566f9dbd4c672afffcbabd13494722305aa3fe4ae0a481a767ac",
    "verification-receipt.md": "454185bcf69e848aea0d3447540e4861ffbad068ac7cf94a63845ceb0be75aa3",
    "recovery-report.md": "3e5c7d274e3971b79f5bf0ef861288881e1d24c4df55da02f05891cea41ce18d",
    "validate.py": "89075b96e96b91a744cd97243563f33530ede5f6cce9afb0c6b7392344eec490",
}
CONTINUATION_REPEATS = {"protein83": (1, 2), "protein157": (1,)}
CONTINUATION_JOB = "14120335"
FINAL_SOURCE = "scal631-20261017-snapshot-03"
FINAL_SHA = "e900d26bbc18ffab5f856292f2dedd9776d89e17e32402521a2b0400d92af3ea"
FINAL_FILES = {
    "inventory.json": "af8d9f346ec744b82ddd2685ebc9c1f288eb571e36f13dbfa3bbb0a68a13a146",
    "raw-sha256.txt": "611b53f1fa2a08fefc9ccc22ef2a4f6bf608f198890e36bfb2fcddb5e4356b77",
    "verification-receipt.md": "b47e6942c18c3590ae0e0f0a8e26e8645c44ec238ae49d72a5917fcfc6bdcd41",
    "completion-report.md": "928cc889c67a95038ac9acecdc21b525105ed54c3dfb5bd7c0a90a17bf3d11f6",
    "audit-receipt.md": "7073aa48020155dfc68a7ccae9060a7cda9993d96efc4686d0d7d2822404c84a",
    "validate.py": "45f9c2cc7d3bdee6ee514bc75cfa970af73b6d97497d2d793b8ce8920d10417d",
    "reparser.py": "f90ea73b4c060a9f92524340c2c7dc3b0912afbb279673c18b4fc3d780198aa8",
    "analysis.json": "0b0b819cdec16a2aa834e42660d141e12e6e525e15eee85cecde1d43d33c01c2",
    "accounting-detail.txt": "8b43e62094c63a6c159db4fd72b0aa4876cd31c0fd4a5a597414b13a5789ebe6",
    "remote-state-and-sha256.json": "357901e32f1aad18df0fc00ec7d04a8cefd24b0187e2445a1a62730edd4acf66",
}
FINAL_CASES = {
    "cpu-protein157-r2": ("14132958", "atl1-1-01-004-24-1"),
    "cpu-protein157-r3": ("14132959", "atl1-1-02-008-26-2"),
    **{f"cpu-{system}-r{repeat}": ("14132960", "atl1-1-03-003-24-2")
       for system, repeats in (("peptide", (2, 3)), ("nanotube", (2, 3)), ("protein83", (3,)))
       for repeat in repeats},
}
ARCHIVE_SHA = "a776c1e2529938368241fa27c69a9e9257795f16c8f9d912355124f2161dfd32"
PIN = "5f71b8ca43c331d6d070f3c0cf33f25ba267ed95"
RUNNER = "06f7b9d47ed5723eb52f54c9142a40d9311cfe0a85313d38a449a1d6baf20a16"
SCIENCE = "e8954d0c9a6267a3b1593ffd7bde745b82293c8f68629caa7945535adbb39346"
SUITE = "72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb"
ITERATOR = "0cb954e4f1cbf3b1c4b95c6b9652a51da08d2467df62041c978a6d229b0051d7"
DRIVER = "7b8dd57183c6b9fb0e2b9448e26c4f48eeb8fc0547f18b076f9c0e1b55a63376"
SDK = "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c"
UUID = "GPU-cef53330-5214-b247-8ee2-ea62cc683988"
SYSTEMS = {
    "peptide": (250, 1010, "14b4041b272a66bea7daf3d010ffd695a6340fd554906b0d7309de4c80b2acab"),
    "nanotube": (548, 2238, "5010ea2807d2f43d06290cf41b99e4890911d810cf818678616dffce98cb4f94"),
    "protein83": (922, 3735, "c4bc7429189d820d4fd8990317ad7e930383576a638adfc2eb70561e9805ab48"),
    "protein157": (1786, 7265, "a1e9c8949bb5950fdb17c07ab718cead826835f042a6b6e94c5cb7dcf75b6b60"),
}
HARDWARE = {
    "cpu": dict(threads=24, job="14108848", node="atl1-1-01-004-19-1",
        memory="200 GiB",
        model="Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz", build="fc851a8346",
        core="90ac4f8292c4c3748b0b4ada5c9ee4d1e6ed933fd72d78f01780fb3cceced1a5",
        libxc="c2c7e453e74c88f6aa8a4470d77a64f368f3efbb61009d45243a3b7e5003378d"),
    "gpu": dict(threads=8, job="14104298", node="atl1-1-03-019-2-0",
        memory="100 GiB",
        model="INTEL(R) XEON(R) PLATINUM 8562Y+", build="b9032fdf25",
        core="8846b91c8a2a8c9224caaf662b3579767904f4a9738a712204d780d94acbe1b8",
        libxc="b8ba011064d688748fca2676801602d6fd8a51712f13235ba2b4511d2976cc98"),
}
CONTINUATION_CPU = dict(HARDWARE["cpu"], job=CONTINUATION_JOB, node="atl1-1-02-003-21-1")
OPTIONS = dict(CUEST_MIXED_PRECISION=False, DFT_GRID_SWITCH_FACTOR=1.0,
    DFT_RADIAL_POINTS=75, DFT_RADIAL_POINTS_INITIAL=35, DFT_SPHERICAL_POINTS=302,
    DFT_SPHERICAL_POINTS_INITIAL=194, DF_BASIS_MP2="aug-cc-pvdz-ri",
    DF_BASIS_SCF="def2-universal-jkfit", D_CONVERGENCE=8, E_CONVERGENCE=9,
    MAXITER=150, ORBITAL_OPTIMIZER_PACKAGE="INTERNAL", PUREAM=True, SAD_PRINT=1,
    SAPT_DFT_DO_DHF=True, SAPT_DFT_FUNCTIONAL="pbe0", SAPT_DFT_GRAC_SHIFT_A=0.136,
    SAPT_DFT_GRAC_SHIFT_B=0.136, SAPT_DFT_INDUCTION_TYPE="CPKS",
    SAPT_DFT_USE_EINSUMS=True, SCF_SUBTYPE="AUTO", BASIS="6-31+g**",
    REFERENCE="rhf", SCF_TYPE="df")
POLICY = dict(final="75x302", initial="35x194", minimum_final_iterations=2, switch_factor=1.0)
SCF = ("SAPT(DFT):Dimer SCF", "SAPT(DFT):Monomer A SCF", "SAPT(DFT):Monomer B SCF",
       "SAPT(DFT):Monomer A DFT", "SAPT(DFT):Monomer B DFT")
# Same 17 legacy hues/roles, with inclusive response Vx explicitly added.
OPERATIONS = [
    ("potential", "XC potential V (inclusive)", "#59d4e8"),
    ("jk", "J/K compute + setup · all phases", "#72d7aa"),
    ("diis", "DIIS", "#ffca67"), ("orbitals", "Orbitals / Form C", "#bba5ff"),
    ("grid", "Grid / collocation construction", "#ec879e"),
    ("sapt", "SAPT algebra / response solver overhead", "#f59d56"),
    ("d4", "D4 dispersion", "#d3ce6b"), ("basis", "Basis / wavefunction", "#819de0"),
    ("housekeeping", "Properties / checkpoint / cleanup", "#be93b0"),
    ("core", "Core H / orthogonalization", "#83b7bd"),
    ("guess", "Orbital guess / SAD remainder", "#b5cb89"),
    ("density", "Density / AO transforms", "#c0a077"),
    ("fd", "Fock / density updates", "#8cb18b"),
    ("formG", "Other work inside Form G", "#c67e63"),
    ("measured", "Other measured SCF work", "#a6a6ba"),
    ("scf", "Unassigned SCF exclusive time", "#647f8d"),
    ("outer", "Driver / wrapper remainder", "#879097"),
    ("response", "XC response Vx (inclusive)", "#3697bf"),
]
INDUCTION_CUTS = ("SAPT(DFT) Energy;SAPT(HF):ind", "SAPT(DFT) Energy;SAPT(DFT):ind")
CLOSURE_TOL_S = 1e-7


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def verify_archive(root):
    hashes = inventory(root)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len(hashes) == 119 and digest == ARCHIVE_SHA, "Archive identity/hash mismatch"
    return hashes


def verify_continuation_archive(root):
    """Pin the increment independently; never merge it into snapshot01."""
    hashes = inventory(root)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len(hashes) == 40 and digest == CONTINUATION_SHA, "continuation archive identity/hash mismatch"
    for name, expected in CONTINUATION_FILES.items():
        assert sha(root.parent / name) == expected, f"continuation evidence identity: {name}"
    records = json.loads((root.parent / "inventory.json").read_text())
    assert len(records) == 40
    assert {r["path"] for r in records} == {"raw/" + p for p in hashes}
    for r in records:
        p = root.parent / r["path"]
        assert p.stat().st_size == r["bytes"]
        assert sha(p) == r["local_sha256"] == r["remote_sha256"]
    return hashes


def continuation_validator(root):
    """Reuse the pinned independent *read-only* reparser, never its writing main."""
    spec = importlib.util.spec_from_file_location("scal631_snapshot02_validator", root.parent / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_data


def final_validator(root):
    """Load pinned read-only functions, NOT main/verify_integrity's historical walk."""
    spec = importlib.util.spec_from_file_location("scal631_snapshot03_validator", root.parent / "validate.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def verify_final_archive(root):
    hashes = inventory(root)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len(hashes) == 115 and digest == FINAL_SHA, "final archive identity/hash mismatch"
    for name, expected in FINAL_FILES.items():
        assert sha(root.parent / name) == expected, f"final evidence identity: {name}"
    records = json.loads((root.parent / "inventory.json").read_text())
    assert len(records) == len({r["path"] for r in records}) == 115
    assert {r["path"] for r in records} == {"raw/" + p for p in hashes}
    for r in records:
        assert r["path"].startswith("raw/") and ".." not in Path(r["path"]).parts
        p = root.parent / r["path"]
        assert p.stat().st_size == r["bytes"]
        assert sha(p) == r["local_sha256"] == r["remote_sha256"]
    assert (root.parent / "raw-sha256.txt").read_text() == "".join(
        f'{r["local_sha256"]}  {r["path"]}\n' for r in records)
    return hashes


def validate(row, meta, output, arm, system, *, job=None, label=None):
    """Validate identities, scientific controls, completion and hardware cohort."""
    gpu = arm == "gpu"
    hw = HARDWARE[arm]
    job = hw["job"] if job is None else job
    if job in {v[0] for v in FINAL_CASES.values()}:
        assert label in FINAL_CASES and label.startswith(f"{arm}-{system}-r"), "Not an allowlisted final label"
        expected_job, node = FINAL_CASES[label]
        assert arm == "cpu" and job == expected_job, "Not an allowlisted final job"
        hw = dict(HARDWARE["cpu"], job=job, node=node)
    elif job == CONTINUATION_JOB:
        assert arm == "cpu" and system in CONTINUATION_REPEATS, "Not an allowlisted continuation case"
        hw = CONTINUATION_CPU
    else:
        assert job == hw["job"], "Not an allowlisted job"
    assert row["ok"] is True and row["arm"] == arm
    assert row["system"] == row["suite_system"] == system
    nbf, naux, geometry = SYSTEMS[system]
    assert (row["nbf"], row["naux_jkfit"], row["geometry_sha256"]) == (nbf, naux, geometry)
    assert row["source_commit"] == PIN and row["runner_sha256"] == RUNNER
    assert row["science_options_sha256"] == SCIENCE and row["suite_systems_sha256"] == SUITE
    assert row["iterator"]["sha256"] == ITERATOR
    assert row["method"] == "sapt(dft)-d4(i)" and row["basis"] == "6-31+g**"
    expected = dict(OPTIONS, USE_CUEST=gpu, CUEST_XC=gpu, CUEST_SAD=gpu)
    # Exact dictionary also disallows iterative GRAC and hidden extra options.
    assert {k.upper(): v for k, v in row["options"].items()} == expected, "Scientific options mismatch"
    assert row["grid_policy"] == POLICY and row["grac_shifts_hartree"] == dict(A=.136, B=.136)
    assert row["threads"] == hw["threads"] and row["slurm_job_id"] == hw["job"]
    assert row["memory"] == hw["memory"]
    assert row["hostname"] == hw["node"] + ".pace.gatech.edu"
    for pin in (f"pin={PIN} arm={arm}", "build_pin=" + hw["build"], "job=" + hw["job"],
                "step=" + row["slurm_step_id"], "host=" + row["hostname"], hw["model"],
                "QOS=embers", "NumCPUs=" + str(hw["threads"]), DRIVER, RUNNER, SUITE, hw["core"]):
        assert pin in meta, f"Missing metadata: {pin}"
    assert re.findall(r"^exit=(\d+) end=\S+$", meta, re.M) == ["0"], "Missing unique successful completion"
    assert ("Partition=gpu-h200" if gpu else "Partition=cpu-medium") in meta
    if gpu:
        assert "NVIDIA H200, " + UUID in meta
    libs = row["libraries_sha256"]
    for name, value in (("/psi4/core.cpython-", hw["core"]), ("libxc.so", hw["libxc"]),
                        ("libcuest.so.0.2.50", SDK)):
        matches = [v for k, v in libs.items() if name in k]
        assert matches == [value], f"Loaded library mismatch: {name}"
    assert row["gpu_builder_seen"] is gpu and row["sad_gpu_builder_seen"] is gpu
    assert row["cpu_builder_seen"] is (not gpu)
    assert not row["direct_seen"] and not row["disk_df_builder_seen"]
    assert "DirectJK" not in output and "DiskDFJK" not in output
    if gpu:
        assert "cuESTJK: GPU-Accelerated" in output and "SAD J/K backend: cuESTJK" in output
        assert "NVIDIA H200" in output
    else:
        assert "MemDFJK" in output and "cuESTJK" not in output
    assert row["policy_checks"] == dict(problems=[], switched_scfs=2)
    proofs = row["proofs"]
    assert len(proofs) == 5 and all(p["converged"] and p["nbf"] == nbf for p in proofs)
    assert [p["functional"] for p in proofs] == ["HF"] * 3 + ["PBE0"] * 2
    assert all(not p["switched"] for p in proofs[:3])
    assert all(p["switched"] and p["final_grid_after"] and p["final"] >= 2 for p in proofs[3:])
    assert output.count("Energy and wave function converged.") == 5
    assert output.count("Near convergence on initial XC grid; continuing on final XC grid.") == 2
    assert output.count("GRAC shift supplied by the user (not computed).") == 2
    solvers = row["cpks_solvers"]
    assert len(solvers) == len(row["cpks_solve_calls_s"]) == 2
    blocks = output.split("SAPT Coupled Induction Solver")[1:]
    assert len(blocks) == 2
    for solver, block in zip(solvers, blocks):
        assert solver["converged"] and solver["convergence"] == 1e-8
        assert len(solver["residuals"]) == solver["iterations"] + 1
        assert all(math.isfinite(v) and 0 <= v < 1e-8 for v in solver["residuals"][-1])
        # Independently re-read the printed numeric iteration rows (not guesses).
        printed = re.findall(r"^\s+(\d+)\s+([\d.eE+-]+)\*?\s+([\d.eE+-]+)\*?\s+\d+\s*$", block, re.M)
        assert len(printed) >= solver["iterations"]
        last = printed[solver["iterations"] - 1]
        assert int(last[0]) == solver["iterations"]
        assert [float(v) for v in last[1:]] == solver["residuals"][-1]
    assert all(math.isfinite(t) and t > 0 for t in row["cpks_solve_calls_s"])
    assert math.isclose(sum(row["cpks_solve_calls_s"]), row["cpks_solve_s"], abs_tol=1e-9)
    assert math.isfinite(row["wall_s"]) and 0 < row["cpks_solve_s"] < row["wall_s"]


def coupled_subtree_partition(root, inherited="outer"):
    """Shared 18-bin cut, including terminal inclusive V/Vx/JK owners.

    Native Functional/Properties derivative children can accumulate thread wall
    and exceed inclusive V/Vx; stop at those inclusive owners, never sum them.
    JK compute/setup is discovered throughout SCF, SAD, HF and DFT induction.
    """
    bins = dict.fromkeys((k for k, _, _ in OPERATIONS), 0.0)
    terminal = {
        "JK:JK": "jk", "SCF:JK factory": "jk", "SCF:JK estimate": "jk",
        "SCF:JK initialize": "jk", "SCF:JK native initialize": "jk",
        "RV:Form V": "potential", "UV:Form V": "potential",
        "RV:Form Vx": "response", "UV:Form Vx": "response",
        "RV:Form Vx (cuEST)": "response", "UV:Form Vx (cuEST)": "response",
        "HF:DIIS": "diis", "HF:Form C": "orbitals", "V:Grid": "grid",
        "SCF:Build collocation": "grid", "HF:Form S/X": "core", "HF:Form core H": "core",
        "JK:D": "density", "JK:AO2USO": "density", "JK:USO2AO": "density",
        "HF:Form F": "fd", "HF:Form D": "fd", "SAPT(DFT):D4 Interaction Energy": "d4",
        "SCF:Auxiliary basis": "basis", "SCF:SAD bases": "basis",
        "SCF:Wavefunction build": "basis",
    }
    def visit(node, inherited="outer"):
        name, wall, children = node["name"], node["wall"], node["children"]
        assert math.isfinite(wall) and wall >= 0
        category = terminal.get(name)
        if name.startswith("DFH:"):
            category = "jk"
        if category:
            bins[category] += wall
            return
        if name in SCF:
            inherited = "scf"
        elif name == "SAPT(DFT):Build JK":
            # Setup owner outside SCF; still traverse its measured children.
            inherited = "jk"
        elif name.startswith(("SAPT(HF):", "SAPT(DFT):")):
            inherited = "sapt"
        elif name == "HF:Form G":
            inherited = "formG"
        elif name in ("HF:Guess", "SAD Guess"):
            inherited = "guess"
        elif name in ("SCF:Finalize energy", "SCF:Properties", "SCF:Checkpoint",
                      "SCF:Publish variables", "SCF:Native finalize", "SCF:Reclaim memory",
                      "SCF:Release collocation"):
            inherited = "housekeeping"
        elif not children and inherited == "scf":
            inherited = "measured"
        exclusive = wall - sum(c["wall"] for c in children)
        assert exclusive >= -1e-7, f"Non-additive timer owner: {name}: {exclusive}"
        bins[inherited] += exclusive
        for child in children:
            visit(child, inherited)
    visit(root, inherited)
    values = [bins[k] for k, _, _ in OPERATIONS]
    assert all(math.isfinite(v) and v >= 0 for v in values)
    assert abs(sum(values) - root["wall"]) < CLOSURE_TOL_S, "Subtree closure"
    return values


def coupled_tree(row, report):
    records = {k: dict(v, timer_path=k.split(";")) for k, v in row["timer_records"].items()}
    assert all(math.isfinite(v["wall_time"]) and v["wall_time"] >= 0 for v in records.values())
    root = report.tree_from_records(records)
    tops = {n["name"]: n for n in root["children"]}
    assert set(SCF) <= tops.keys(), "Missing required fixed-GRAC SCF stage"
    assert not any("GRAC Shift Monomer" in n for n in tops), "Unexpected iterative GRAC"
    return root


def coupled_partition(row, report):
    """Original measured-vector contract, including its bounded wrapper rounding."""
    root = coupled_tree(row, report)
    bins = dict(zip((k for k, _, _ in OPERATIONS), coupled_subtree_partition(root)))
    wrapper = row["wall_s"] - root["wall"]
    assert wrapper >= 0
    bins["outer"] += wrapper
    assert min(bins.values()) >= 0 and abs(sum(bins.values()) - row["wall_s"]) < 1e-7
    # Correct final floating-point rounding only; no balancing of real errors.
    bins["outer"] += row["wall_s"] - sum(bins.values())
    return [bins[k] for k, _, _ in OPERATIONS]


def induction_projection(row, report, *, full_values=None, cut_paths=INDUCTION_CUTS):
    """Subtract two entire disjoint induction owners, NEVER nested solve durations.

    The parent exclusive times stay unchanged: deleting subtrees and repartitioning
    the original inclusive energy root would incorrectly put the cut back there.
    No clamping, balancing or proportional redistribution is performed.
    """
    assert len(cut_paths) == len(set(cut_paths)) == 2, "Exactly two unique cuts required"
    paths = [tuple(p.split(";")) for p in cut_paths]
    assert not any(a == b[:len(a)] for a in paths for b in paths if a != b), "Overlapping cuts"
    assert tuple(cut_paths) == INDUCTION_CUTS, "Wrong induction cut allowlist/order"
    assert all(len(p) == 2 and p[0] == "SAPT(DFT) Energy" for p in paths), "Cuts must be direct siblings"
    records = row["timer_records"]
    assert all(p in records for p in cut_paths), "Missing exact induction cut path"
    assert all(records[p]["n_calls"] == 1 for p in cut_paths), "Induction cut call count"
    root = coupled_tree(row, report)  # Also rejects normalized duplicate timer paths.
    owners = {n["name"]: n for n in root["children"]}
    calls = row["cpks_solve_calls_s"]
    assert len(calls) == 2 and all(math.isfinite(t) and t > 0 for t in calls)
    assert math.isclose(sum(calls), row["cpks_solve_s"], rel_tol=0, abs_tol=1e-9), "Solver sum"
    phases = []
    for path, solve in zip(cut_paths, calls):
        owner = owners[path.split(";")[-1]]
        assert owner["calls"] == 1
        assert 0 < solve <= owner["wall"], "Solver does not fit induction owner"
        vector = coupled_subtree_partition(owner, inherited="sapt")
        phases.append(dict(path=path, wall_s=owner["wall"], values=vector))
    removed = [sum(p["values"][i] for p in phases) for i in range(len(OPERATIONS))]
    removed_wall = sum(p["wall_s"] for p in phases)
    assert abs(sum(removed) - removed_wall) < CLOSURE_TOL_S, "Removed closure"
    full = coupled_partition(row, report) if full_values is None else full_values
    assert len(full) == len(removed)
    projected = [f-r for f, r in zip(full, removed)]
    assert all(math.isfinite(v) and v >= 0 for v in projected), "Negative projected component"
    wall = row["wall_s"] - removed_wall
    assert math.isfinite(wall) and 0 < wall < row["wall_s"], "Invalid projected wall"
    assert abs(sum(projected) - wall) < CLOSURE_TOL_S, "Projected closure"
    assert all(abs(f-(p+r)) < CLOSURE_TOL_S for f, p, r in zip(full, projected, removed))
    return dict(wall_s=wall, values=projected, removed_values=removed, phases=phases)


def summarize(runs):
    """Additive median-wall display; n2 is explicitly not a measured run."""
    assert 1 <= len(runs) <= 3 and len({r["repeat"] for r in runs}) == len(runs)
    assert len({len(r["values"]) for r in runs}) == 1
    assert all(math.isfinite(r["wall_s"]) and r["wall_s"] > 0 and
               all(math.isfinite(v) and v >= 0 for v in r["values"]) and
               abs(sum(r["values"]) - r["wall_s"]) < 1e-7 for r in runs)
    samples = [r["wall_s"] for r in runs]
    median = statistics.median(samples)
    ordered = sorted(runs, key=lambda r: (r["wall_s"], r["repeat"]))
    if len(runs) == 2:
        representative = None
        display = dict(wall_s=median, values=[(a+b)/2 for a, b in zip(
            ordered[0]["values"], ordered[1]["values"])],
            repeats=sorted(r["repeat"] for r in ordered), kind="averaged-two-run",
            label="averaged two-run vector")
    else:
        representative = ordered[len(runs)//2]
        assert representative["wall_s"] == median
        display = dict(wall_s=median, values=representative["values"],
            repeats=[representative["repeat"]], kind="measured-run",
            label="actual single-run vector" if len(runs) == 1 else "actual median-wall-run vector")
    assert abs(sum(display["values"]) - median) < 1e-7
    return dict(n=len(runs), planned=3, median_s=median, min_s=min(samples), max_s=max(samples),
                representative=representative, display_vector=display, runs=runs)


def summarize_projection(runs):
    # Transform each run FIRST, then choose the new median and observation range.
    summary = summarize([dict(repeat=r["repeat"], wall_s=r["induction_projection"]["wall_s"],
        values=r["induction_projection"]["values"]) for r in runs])
    vector = summary["display_vector"]
    if summary["n"] != 2:
        vector["kind"] = "projected-run"
        vector["label"] = "projected single-run vector" if summary["n"] == 1 else "projected median-wall-run vector"
    return summary


def verify(root, docs, *, continuation=None, final_cpu=None):
    assert not final_cpu or continuation, "Final CPU requires independent snapshot02"
    hashes = verify_archive(root)
    extra_hashes = verify_continuation_archive(continuation) if continuation else None
    reparse = continuation_validator(continuation) if continuation else None
    final_hashes = verify_final_archive(final_cpu) if final_cpu else None
    final_proof = None
    if final_cpu:
        validator = final_validator(final_cpu)
        assert set(validator.NEW) == set(FINAL_CASES)
        for label, (job, node) in FINAL_CASES.items():
            assert validator.NEW[label] == (job, node + ".pace.gatech.edu")
        outcomes = validator.verify_operations()
        rejected = validator.negative_controls()
        assert len(rejected) == 20
        # All 24 actual outputs, full timer.dat trees, profiles and process logs.
        checked = [validator.verify_record(f"{arm}-{system}-r{repeat}")[1]
                   for system in SYSTEMS for arm in ("cpu", "gpu") for repeat in (1, 2, 3)]
        final_proof = dict(jobs=outcomes, negative_controls_rejected=rejected,
            reconciled_timer_paths=sum(r["printed_hierarchical_timers_reconciled"] for r in checked),
            historical_checkpoint="Parent bebf3565f exit0 before publication edits: 2619 historical files. "
                "Not rerun after authorized checker updates; scoped raw/receipt/operational guards rechecked.")
    sys.path.insert(0, str(docs.resolve()))
    import build_breakdown_2026_10_04 as report
    cases = []
    for system, (nbf, naux, geometry) in SYSTEMS.items():
        arms = {}
        for arm in ("cpu", "gpu"):
            rows = []
            extra = arm == "cpu" and system in CONTINUATION_REPEATS and continuation is not None
            expected = (CONTINUATION_REPEATS[system] if extra else
                ((1,) if arm == "cpu" and system in ("peptide", "nanotube") else
                 (() if arm == "cpu" else (1, 2, 3))))
            for repeat in ((1, 2, 3) if final_cpu else expected):
                label = f"{arm}-{system}-r{repeat}"
                final = final_cpu is not None and label in FINAL_CASES
                archive, source = ((final_cpu, FINAL_SOURCE + "/raw") if final else
                    ((continuation, CONTINUATION_SOURCE) if extra else (root, SOURCE)))
                stem = f"sessions/{arm}-{system}-r{repeat}"
                row = json.loads((archive / stem / "result.json").read_text())
                meta = (archive / (stem + ".meta.txt")).read_text()
                output = (archive / stem / "psi4.out").read_text()
                validate(row, meta, output, arm, system,
                         job=FINAL_CASES[label][0] if final else
                             (CONTINUATION_JOB if extra else HARDWARE[arm]["job"]), label=label)
                if extra and not final:
                    reparse(row, output, meta, arm, system, CONTINUATION_JOB)
                values = coupled_partition(row, report)
                rows.append(dict(repeat=repeat, wall_s=row["wall_s"], values=values,
                    induction_projection=induction_projection(row, report, full_values=values),
                    combined_coupled_s=row["cpks_solve_s"], solver_calls_s=row["cpks_solve_calls_s"],
                    proofs=row["proofs"], solvers=[dict(iterations=s["iterations"],
                        final_residuals=s["residuals"][-1]) for s in row["cpks_solvers"]],
                    source=f"{source}/{stem}/result.json", output=f"{source}/{stem}/psi4.out",
                    metadata=f"{source}/{stem}.meta.txt", components_eh=row["components_hartree"],
                    job=row["slurm_job_id"], node=row["hostname"],
                    result_sha256=sha(archive / stem / "result.json")))
            arms[arm] = summarize(rows) if rows else None
        cpu, gpu = arms["cpu"], arms["gpu"]
        ratio = cpu["median_s"] / gpu["median_s"] if cpu else None
        delta = max((abs(c["components_eh"][k] - g["components_eh"][k])
                     for c in cpu["runs"] for g in gpu["runs"] for k in c["components_eh"]),
                    default=None) if cpu else None
        projected_arms = {a: summarize_projection(arms[a]["runs"]) if arms[a] else None for a in arms}
        pc, pg = projected_arms["cpu"], projected_arms["gpu"]
        cases.append(dict(system=system, nbf=nbf, naux=naux, geometry_sha256=geometry,
            max_component_delta_eh=delta,
            timing_views=dict(induction_none_like=dict(**projected_arms,
                ratio=pc["median_s"]/pg["median_s"] if pc else None,
                ratio_range=[pc["min_s"]/pg["max_s"], pc["max_s"]/pg["min_s"]] if pc else None)),
            **arms, ratio=ratio, ratio_range=([cpu["min_s"]/gpu["max_s"],
                cpu["max_s"]/gpu["min_s"]] if cpu else None)))
    identities = [r["source"] for c in cases for a in ("cpu", "gpu") if c[a] for r in c[a]["runs"]]
    assert len(identities) == len(set(identities)) == (24 if final_cpu else 17 if continuation else 14)
    return dict(id=FINAL_SOURCE if final_cpu else CONTINUATION_SOURCE if continuation else SOURCE,
        schema="fixed-grac-coupled-scaling-v1", partial=not bool(final_cpu),
        planned_repeats_per_arm=3, measured_records=len(identities), planned_records=24,
        method="PBE0 SAPT(DFT)-D4(I)/6-31+g**", policy=POLICY, options=OPTIONS,
        hardware=HARDWARE, gpu_uuid=UUID, source_commit=PIN, runner_sha256=RUNNER,
        science_options_sha256=SCIENCE, suite_sha256=SUITE, iterator_sha256=ITERATOR,
        driver_tree_sha256=DRIVER, sdk_sha256=SDK, archive_sha256=ARCHIVE_SHA,
        evidence_sha256=hashes, operations=OPERATIONS, cases=cases,
        timing_views=dict(full=dict(label="Full coupled · measured", status="measured"),
            induction_none_like=dict(label="Induction NONE-like · projected", status="derived",
                cut_paths=INDUCTION_CUTS, closure_tolerance_s=CLOSURE_TOL_S,
                formula="projected = full - inclusive HF induction - inclusive DFT induction",
                caveat="Not independently measured NONE. Removes both solves and surrounding induction/ExchInd algebra; retained setup, correction bookkeeping and runtimes may differ. Coupled energies unchanged.")),
        continuation=(dict(id=CONTINUATION_SOURCE, job=CONTINUATION_JOB,
            hardware=CONTINUATION_CPU, archive_sha256=CONTINUATION_SHA,
            evidence_sha256=extra_hashes, receipts_sha256=CONTINUATION_FILES,
            status="PREEMPTED; .batch FAILED; three complete records; seven never started")
            if continuation else None),
        **(dict(final_cpu=dict(id=FINAL_SOURCE, archive_sha256=FINAL_SHA,
            evidence_sha256=final_hashes, receipts_sha256=FINAL_FILES,
            allowlist=FINAL_CASES, verification=final_proof)) if final_cpu else {}),
        excluded=["pla15_3pe2: expensive opt-in, unmeasured", "all smoke runs, including failed GPU SAD-print smoke"],
        caveat="Nominal grid policy matches; native/cuEST quadratures and host/CUDA LibXC builds differ, not pointwise identical grids.")


def publish(root, docs, *, render=False, continuation=None, final_cpu=None):
    campaign = verify(root, docs, continuation=continuation, final_cpu=final_cpu)
    path = docs / "grid-evidence.json"
    before = json.loads(path.read_text())
    assert not (before.get(KEY, {}).get("final_cpu") and final_cpu is None), (
        "Refusing final24 downgrade: supply the verified final CPU archive")
    assert not (before.get(KEY, {}).get("continuation") and continuation is None), (
        "Refusing stale snapshot01 downgrade: supply the verified continuation")
    after = copy.deepcopy(before)
    after[KEY] = campaign
    assert {k: v for k, v in after.items() if k != KEY} == {k: v for k, v in before.items() if k != KEY}
    artifacts = dict(campaign["evidence_sha256"])
    receipt = root.parent / "verification-receipt.md"
    assert receipt.is_file(), "Missing independent verification receipt"
    artifacts["verification-receipt.md"] = sha(receipt)
    copies = []
    for relative, digest in artifacts.items():
        copies.append((receipt if relative == "verification-receipt.md" else root / relative,
                       docs / SOURCE / relative, digest))
    if continuation:
        for relative, digest in campaign["continuation"]["evidence_sha256"].items():
            copies.append((continuation / relative, docs / CONTINUATION_SOURCE / relative, digest))
        for relative, digest in CONTINUATION_FILES.items():
            copies.append((continuation.parent / relative, docs / CONTINUATION_SOURCE / relative, digest))
    if final_cpu:
        for relative, digest in campaign["final_cpu"]["evidence_sha256"].items():
            copies.append((final_cpu / relative, docs / FINAL_SOURCE / "raw" / relative, digest))
        for relative, digest in FINAL_FILES.items():
            copies.append((final_cpu.parent / relative, docs / FINAL_SOURCE / relative, digest))
        target_root = docs / FINAL_SOURCE
        allowed = {str(t) for _, t, _ in copies if target_root in t.parents}
        assert all(str(p) in allowed for p in target_root.rglob("*") if p.is_file()), "Unexpected final artifact"
    # Preflight BOTH snapshots before any copy or data replacement.
    for source, target, digest in copies:
        assert sha(source) == digest, f"Changed source artifact: {source}"
        assert not target.exists() or sha(target) == digest, f"Refusing mismatched artifact: {target}"
    for source, target, _ in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
    path.write_text(json.dumps(after, indent=2, allow_nan=False) + "\n")
    if render:
        from publish_grid_evidence import render_preserving_history
        render_preserving_history(docs)
    return campaign


def publish_cpu_disclosure(root, docs):
    """Separate full-aug CPU-only disclosure; never enters the scaling cohort."""
    result = root / "raw/sessions/protein157-adz-cpks-full2/result.json"
    assert sha(result) == "9622b56517d2b0e95e1c9650dfdf86a7cb4fe5a075c93e42721474a746346d63"
    hashes = inventory(root)
    targets = [(root / name, docs / "protein157-cpu-14103914-complete" / name, digest)
               for name, digest in hashes.items()]
    for _, target, digest in targets:
        assert not target.exists() or sha(target) == digest, f"Refusing mismatched artifact: {target}"
    for source, target, _ in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--continuation", type=Path, help="Verified snapshot02 incremental raw directory")
    parser.add_argument("--final-cpu", type=Path, help="Verified snapshot03 independent 115-file raw archive")
    parser.add_argument("--cpu-disclosure", type=Path)
    args = parser.parse_args()
    result = publish(args.root, args.docs, render=args.render, continuation=args.continuation, final_cpu=args.final_cpu)
    if args.cpu_disclosure:
        publish_cpu_disclosure(args.cpu_disclosure, args.docs)
    print(f"Published {result['measured_records']}/24 records; partial={result['partial']}; old evidence keys preserved.")
