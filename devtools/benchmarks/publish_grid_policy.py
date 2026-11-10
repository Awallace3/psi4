#!/usr/bin/env python3
"""Append the matched policy cohort without pooling it with the earlier pilot."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import publish_grid_evidence as renderer
import publish_tiny_grid_pilot as base

JOB = "14098797"
PIN = "f4e4c689700b5e8eeea618a714b45bae17e2e466"
POLICIES = dict(early1=(10., 1), full2=(1., 2), full1=(1., 1), early2=(10., 2))
CONFIGS = dict(base.CONFIGS)
for arm, config in base.CONFIGS.items():
    if arm not in ("fixed75", "fixed99"):
        for policy in POLICIES:
            CONFIGS[arm+"_"+policy] = config


def validate_policy(row, overlay_sha):
    """Fail closed on missing/wrong policy or inadequate final-grid evaluations."""
    if not row.get("ok"):
        return
    name = row["arm"].rsplit("_", 1)[1] if "_" in row["arm"] else "early1"
    factor, minimum = POLICIES[name]
    expected = dict(name=name, switch_factor=factor, minimum_final_iterations=minimum)
    assert row["grid_policy"] == expected
    assert row["options"]["DFT_GRID_SWITCH_FACTOR"] == factor
    assert row["iterator_overlay_sha256"] == overlay_sha
    proofs = row["per_scf_grid_iteration_proofs"]
    switched = row["options"]["DFT_RADIAL_POINTS_INITIAL"] > 0
    if switched:
        assert proofs, "No per-SCF iteration proof for a switched calculation"
        assert all(p["minimum_final_iterations"] == minimum for p in proofs)
        assert all(p["final"] >= minimum for p in proofs if p["converged"])
        assert sum(p["initial"] for p in proofs) == row["initial_iterations"]
        assert sum(p["final"] for p in proofs) == row["final_iterations"]
    else:
        assert not proofs


def publish(root, docs, frozen, render=False):
    driver = subprocess.check_output(["git", "-C", str(frozen), "show",
        f"{PIN}:psi4/driver/procrouting/scf_proc/scf_iterator.py"])
    sha = hashlib.sha256(driver).hexdigest()
    manifest, rows = base.load_snapshot(root, expected_job=JOB,
        expected_harness=PIN, configs=CONFIGS, case_timeout=360)
    probe = json.loads((root / "probe/result.json").read_text())
    assert probe["ok"]
    assert {c["overlay_sha256"] for c in probe["cases"] if "overlay_sha256" in c} == {sha}
    for row in rows:
        validate_policy(row, sha)
    data = base.prepare(manifest, rows, configs=CONFIGS)
    data["harness_commit"] = PIN
    data["driver_overlay_sha256"] = sha
    data["policy_label"] = "Matched policy comparison on one allocation: select threshold/final-iteration policy below."
    expected = len(manifest["systems"])*len(manifest["arms"])
    success = sum(r.get("ok", False) for r in rows)
    finished = any((root / "metadata" / name).exists() for name in ("COMPLETED", "FAILED"))
    data["state"] = (("Finished" if finished else "Running snapshot")+
                     f": {success} converged; {len(rows)}/{expected} cases attempted")
    target = docs / f"grid-policy-{JOB}"
    target.mkdir(exist_ok=True)
    archive = root.parent / "artifacts.tar.gz"
    shutil.copyfile(archive, target / "artifacts.tar.gz")
    data["source"] = f"grid-policy-{JOB}/artifacts.tar.gz"
    data["artifact_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    source_by_key = {(r["system"], r["arm"]): r for r in rows}
    for group in data["groups"]:
        for item in group["rows"]:
            row = source_by_key[group["system"], item["arm"]]
            name = item["arm"].rsplit("_", 1)[1] if "_" in item["arm"] else "early1"
            factor, minimum = POLICIES[name]
            item["policy"] = name
            item["base_arm"] = item["arm"].rsplit("_", 1)[0] if "_" in item["arm"] else item["arm"]
            item["policy_label"] = f"{factor:g}× threshold / ≥{minimum} final iterations"
            if item["nominal_fraction"] is not None:
                item["label"] += " · "+item["policy_label"]
            item["per_scf_proofs"] = row.get("per_scf_grid_iteration_proofs", [])
        group.update(job_id=JOB, harness_commit=PIN, scientific_commit=base.SCIENCE,
                     driver_overlay_sha256=sha, threads=manifest["threads"], state=data["state"],
                     title=f'{group["system"]} · policy job {JOB}', source=data["source"],
                     policy_label=data["policy_label"])
    (target / "evidence.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    path = docs / "grid-evidence.json"
    evidence = json.loads(path.read_text())
    before = copy.deepcopy(evidence)
    old = evidence.get("tiny_grid_pilot")
    if old:
        # Add source annotations only; preserve all original numeric rows.
        groups = []
        for original in old["groups"]:
            if original.get("job_id", old["job_id"]) == JOB:
                continue
            group = copy.deepcopy(original)
            for field in ("job_id", "harness_commit", "scientific_commit", "threads",
                          "state", "source", "policy_label"):
                group.setdefault(field, old.get(field))
            group.setdefault("title", f'{group["system"]} · initial job {group["job_id"]}')
            groups.append(group)
        data["groups"] = groups+data["groups"]
    evidence["tiny_grid_pilot"] = data
    for key in before:
        if key != "tiny_grid_pilot":
            assert evidence[key] == before[key], f"Unrelated evidence changed: {key}"
    path.write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")
    if render:
        renderer.render_preserving_history(docs)
    print(json.dumps(dict(job_id=JOB, cases=len(rows), converged=success,
                          groups=len(data["groups"]), state=data["state"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    publish(args.root, args.docs, args.frozen, args.render)
