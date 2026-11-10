#!/usr/bin/env python3
"""Publish finalized pilot cases and exact printed iteration-row accounting."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys

SCIENCE = "c28950041bbe8bb10617264a7b6ddc56a436cda1"
HARNESS = "4a21d9e7a167ca02322676695305689d509a08ec"
CORE = "5d3041f50b82a5fc5ea634ae6a0ee1cb4d2f420a8a53086a83cf1aef27ce1121"
STAGES = OrderedDict([
    ("gracA", ("GRAC A · all neutral/cation solves", "SAPT(DFT):GRAC Shift Monomer A")),
    ("gracB", ("GRAC B · all neutral/cation solves", "SAPT(DFT):GRAC Shift Monomer B")),
    ("hfD", ("δHF · dimer HF", "SAPT(DFT):Dimer SCF")),
    ("hfA", ("δHF · monomer A HF", "SAPT(DFT):Monomer A SCF")),
    ("hfB", ("δHF · monomer B HF", "SAPT(DFT):Monomer B SCF")),
    ("dftA", ("DFT monomer A", "SAPT(DFT):Monomer A DFT")),
    ("dftB", ("DFT monomer B", "SAPT(DFT):Monomer B DFT")),
])
COMPONENTS = [f"SAPT {name} ENERGY" for name in ("ELST", "EXCH", "IND", "DISP", "TOTAL")]
CONFIGS = dict(fixed75=((75, 302), (0, 0)), fixed99=((99, 590), (0, 0)),
               tiny10=((75, 302), (20, 110)), tiny12=((75, 302), (25, 110)),
               tiny19=((75, 302), (30, 146)), tiny26=((75, 302), (30, 194)),
               coarse75=((75, 302), (35, 194)), tiny08=((75, 302), (20, 86)),
               tiny05=((75, 302), (15, 74)))


def iteration_sections(text):
    """Count the same matches as the worker, never iteration numbers or calls."""
    sections = OrderedDict((key, dict(name=label, initial=0, final=0, untagged=0, total=0))
                           for key, (label, _) in STAGES.items())
    current = None
    unknown = []
    for number, line in enumerate(text.splitlines(), 1):
        if match := re.match(r"^[ \t]*Computing GRAC shift for Monomer ([AB]) using ", line):
            current = "grac"+match[1]
        elif re.match(r"^[ \t]*GRAC shift Monomer [AB]:", line):
            current = None
        elif match := re.match(r"^[ \t]*SAPT\(DFT\):[ \t]*(.*?)[ \t]*$", line):
            banner = match[1]
            if match := re.fullmatch(r"(?:delta HF|SAPT0 induction) (Dimer|Monomer [AB])", banner):
                current = "hf"+("D" if match[1] == "Dimer" else match[1][-1])
            elif match := re.fullmatch(r"DFT Monomer ([AB])", banner):
                current = "dft"+match[1]
            else:
                current = None
        matches = list(re.finditer(r"@\S+ iter ", line))
        if not matches:
            continue
        if current is None:
            unknown.append(number)
            continue
        assert not ("XC-INITIAL" in line and "XC-FINAL" in line)
        category = "initial" if "XC-INITIAL" in line else "final" if "XC-FINAL" in line else "untagged"
        sections[current][category] += len(matches)
        sections[current]["total"] += len(matches)
    assert not unknown, f"Unassigned SCF iteration rows at lines {unknown[:10]}"
    assert sum(s["total"] for s in sections.values()) == len(re.findall(r"@\S+ iter ", text))
    return sections


def normalize(value):
    return re.sub(r":\s*", ":", value.strip())


def attach_times(sections, timers):
    for key, section in sections.items():
        target = normalize(STAGES[key][1])
        matching = [(path.split(";"), value) for path, value in timers.items()
                    if len(path.split(";")) > 1 and normalize(path.split(";")[1]) == target]
        wall = [value["wall_time"] for path, value in matching if len(path) == 2]
        section["wall_s"] = wall[0] if wall else None
        section["xc_s"] = sum(value["wall_time"] for path, value in matching
                              if normalize(path[-1]) in ("RV:Form V", "UV:Form V"))


def load_snapshot(root, *, expected_job="14097678", expected_harness=HARNESS,
                  configs=None, case_timeout=1500):
    configs = CONFIGS if configs is None else configs
    manifest = json.loads((root / "results/campaign.json").read_text())
    assert manifest["job_id"] == expected_job
    assert manifest["scientific_commit"] == SCIENCE and manifest["harness_commit"] == expected_harness
    assert manifest["basis"] == "aug-cc-pvdz" and manifest["variant"] == "cpu"
    assert not manifest["sampling"] and not manifest["microprofiling"]
    rows = []
    for record in manifest["records"]:
        directory = root / "results" / Path(record["path"]).parent.name
        path = directory / "result.json"
        row = json.loads(path.read_text()) if path.exists() else dict(
            system=record["system"], arm=record["arm"], ok=False,
            error=f"No complete worker result; exit {record['returncode']}")
        if record["returncode"]:
            row["ok"] = False
            if record["returncode"] == 124:
                row["error"] = f"Exceeded the {case_timeout} s per-case limit; retained iteration rows are partial."
        assert row["system"] == record["system"] and row["arm"] == record["arm"]
        text = (directory / "psi4.out").read_text() if (directory / "psi4.out").exists() else ""
        sections = iteration_sections(text)
        if row.get("ok"):
            assert record["returncode"] == 0 and row["core_sha256"] == CORE
            assert row["scientific_commit"] == SCIENCE and row["backend"] == "cpu"
            assert row["cuda_visible_devices"] == "" and row["options"]["BASIS"] == "aug-cc-pvdz"
            assert all(row["options"][k] is False for k in
                       ("USE_CUEST", "CUEST_XC", "CUEST_SAD", "CUEST_MIXED_PRECISION", "CUEST_XC_TIMING"))
            assert row["options"]["SAPT_DFT_INDUCTION_TYPE"] == "NONE"
            final, initial = configs[row["arm"]]
            assert (row["options"]["DFT_RADIAL_POINTS"],
                    row["options"]["DFT_SPHERICAL_POINTS"]) == final
            assert (row["options"]["DFT_RADIAL_POINTS_INITIAL"],
                    row["options"]["DFT_SPHERICAL_POINTS_INITIAL"]) == initial
            for tag, field in (("total", "all_iterations"), ("initial", "initial_iterations"),
                               ("final", "final_iterations")):
                assert sum(s[tag] for s in sections.values()) == row[field]
            attach_times(sections, row["timers"])
            assert abs(sum(s["xc_s"] for s in sections.values())-row["xc_s"]) < 1.e-7
        row.update(sections=list(sections.values()), returncode=record["returncode"],
                   iterations_observed=(directory / "psi4.out").exists())
        rows.append(row)
    return manifest, rows


def prepare(manifest, rows, *, configs=None):
    configs = CONFIGS if configs is None else configs
    groups = []
    for system in dict.fromkeys(r["system"] for r in rows):
        cases = [r for r in rows if r["system"] == system]
        controls = {a: [r for r in cases if r["arm"] == a and r.get("ok")]
                    for a in ("fixed75", "fixed99")}
        reduced = []
        for r in cases:
            opts = r.get("options", {})
            arm = r["arm"]
            planned_final, planned_initial = configs[arm]
            initial = (opts.get("DFT_RADIAL_POINTS_INITIAL", planned_initial[0]),
                       opts.get("DFT_SPHERICAL_POINTS_INITIAL", planned_initial[1]))
            fraction = initial[0]*initial[1]/(75*302) if all(initial) else None
            final = (opts.get("DFT_RADIAL_POINTS", planned_final[0]),
                     opts.get("DFT_SPHERICAL_POINTS", planned_final[1]))
            short = f"{initial[0]}×{initial[1]}" if fraction else f"{final[0]}×{final[1]}"
            label = f"{short} → 75×302" if fraction else f"Fixed {short}"
            item = dict(arm=arm, short_label=short, label=label, nominal_fraction=fraction,
                        sections=r["sections"], acceptable=False, status="FAILED",
                        iterations={k: sum(s[k] for s in r["sections"])
                                    for k in ("initial", "final", "untagged")})
            item["total_iterations"] = sum(item["iterations"].values())
            item["partial_iterations"] = not r.get("ok")
            if not r.get("iterations_observed", True):
                item.update(total_iterations=None, iterations={}, sections=[])
            if r.get("ok"):
                item.update(status="Converged · controls pending", wall_s=r["wall_s"])
                if controls["fixed75"]:
                    item["speedup"] = statistics.median(x["wall_s"] for x in controls["fixed75"])/r["wall_s"]
                for number, control in (("75", controls["fixed75"]), ("99", controls["fixed99"])):
                    if control:
                        item["max_error_ref"+number] = max(
                            abs(r["components"][key]-c["components"][key])
                            for c in control for key in COMPONENTS)
                if all(controls.values()):
                    item["acceptable"] = max(item["max_error_ref75"], item["max_error_ref99"]) <= 1.e-5
                    item["status"] = "Converged · accuracy PASS" if item["acceptable"] else "Converged · accuracy FAIL"
            else:
                item["error"] = r.get("error", "Worker failed")
                if r.get("returncode") == 124:
                    item["status"] = "Timed out · partial rows only"
            reduced.append(item)
        reduced.sort(key=lambda r: (r["nominal_fraction"] is not None,
                                   r["nominal_fraction"] or (0 if r["arm"] == "fixed75" else 1)))
        groups.append(dict(system=system, basis="aug-cc-pvdz", rows=reduced))
    expected = len(manifest["systems"])*len(manifest["arms"])
    return dict(job_id=manifest["job_id"], scientific_commit=SCIENCE, harness_commit=HARNESS,
                backend="CPU · all accelerator routes disabled", threads=manifest["threads"],
                policy_label="Initial study: switch at 10× effective SCF thresholds; minimum 1 final-grid iteration.",
                updated_at=datetime.now(timezone.utc).isoformat(), groups=groups,
                state=f"Partial snapshot: {len(rows)}/{expected} attempted cases" if len(rows) != expected
                      else f"Initial matrix: {len(rows)}/{expected} attempted cases",
                repeats_label="one fresh process per arm; exploratory, not a robust speedup claim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    manifest, rows = load_snapshot(args.root)
    data = prepare(manifest, rows)
    if (args.root / "metadata/FAILED").exists():
        data["state"] = (f"Stopped early: {sum(r.get('ok', False) for r in rows)} completed; "
                         f"{len(rows)}/{len(manifest['systems'])*len(manifest['arms'])} attempted cases. "
                         "Protein83 fixed75 exceeded its 1500 s case limit")
    destination = args.docs / "tiny-grid-pilot-14097678"
    destination.mkdir(exist_ok=True)
    archive = args.root.parent / "artifacts.tar.gz"
    shutil.copyfile(archive, destination / "artifacts.tar.gz")
    data["source"] = "tiny-grid-pilot-14097678/artifacts.tar.gz"
    data["artifact_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (destination / "evidence.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    target = args.docs / "grid-evidence.json"
    evidence = json.loads(target.read_text())
    evidence["tiny_grid_pilot"] = data
    target.write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")
    if args.render:
        import publish_grid_evidence
        publish_grid_evidence.render_preserving_history(args.docs)
    print(json.dumps(dict(cases=len(rows), systems=[g["system"] for g in data["groups"]],
                         summed_iterations={g["system"]: [r["total_iterations"] for r in g["rows"]]
                                            for g in data["groups"]}), indent=2))
