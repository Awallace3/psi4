#!/usr/bin/env python3
"""Bounded, read-only progress snapshot; missing files never mean completion."""
import argparse
import json
from pathlib import Path
import time


def snapshot(root):
    results = []
    for phase in sorted(p for p in root.iterdir() if p.is_dir()):
        base = phase / "campaign"
        if not base.is_dir():
            continue
        item = {"phase": phase.name, "state": "INITIALIZING"}
        try:
            manifest = json.loads((base / "manifest.json").read_text())
            config = manifest["config"]
            planned = len(manifest["systems"]) * len(config["functionals"]) * 2 * (
                config["repeats"] + int("--profiles" in manifest["command"]))
            records_path = base / "records.json"
            records = json.loads(records_path.read_text()) if records_path.exists() else []
            good = sum(bool(r.get("ok")) and r.get("process_returncode") == 0 for r in records)
            item.update(state="IN_PROGRESS", planned_workers=planned,
                        finished_workers=len(records), successful_workers=good,
                        failed_workers=len(records) - good)
            if records:
                last = records[-1]
                item.update(last_worker={k: last.get(k) for k in
                            ("system", "functional", "engine", "repeat", "profile",
                             "process_returncode")},
                            seconds_since_last_worker=round(time.time() - last["ended_unix_s"], 1))
                if last.get("error"):
                    item["last_error"] = last["error"][:500]
            complete = base / "COMPLETE.json"
            if complete.exists():
                marker = json.loads(complete.read_text())
                finished = (marker["complete"] and marker["final_build_hashes_unchanged"] and
                            len(records) == planned and good == planned)
                item["state"] = "COMPLETE" if finished else "INCOMPLETE"
                item["build_hashes_unchanged"] = marker["final_build_hashes_unchanged"]
            elif (base / "FAILED.json").exists():
                item["state"] = "FAILED"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            item.update(state="UNKNOWN", error=f"{type(exc).__name__}: {exc}"[:500])
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    args = parser.parse_args()
    items = snapshot(args.campaign_root)
    print(json.dumps(items, indent=2))
    return 0 if items else 3


if __name__ == "__main__":
    raise SystemExit(main())
