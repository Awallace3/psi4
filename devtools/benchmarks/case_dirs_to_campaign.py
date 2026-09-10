#!/usr/bin/env python3
"""Synthesize a campaign manifest from per-case result directories.

`saptdft_cuest_grac.py --case` writes one self-describing directory per
measurement, but `campaign()` refuses to continue past the first failure, so a
long campaign is normally driven by an external launcher issuing one `--case`
per measurement. That leaves no `campaign.json`, which
`summarize_saptdft_cuest.py` requires. This rebuilds the manifest from the case
directories so the existing summarizer runs unmodified.

Completeness is judged against what was *requested*, given as `--expect`, not
against what happens to be on disk: a campaign whose cases silently never ran
must not summarize as complete. A case directory with no `result.json`, or one
whose `ok` is false, is recorded with a nonzero return code so the summarizer
reports it as a failure instead of dropping it.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

NAME = re.compile(r"^(?P<stem>.+)-(?P<mode>cpu|gpu)-(?P<repeat>\d+)$")


def parse_expect(text):
    """Parse `system:basis:repeats`, e.g. `benzene:aug-cc-pvdz:3`."""
    fields = text.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(f"expected system:basis:repeats, got {text!r}")
    system, basis, repeats = fields
    if not system or not basis:
        raise argparse.ArgumentTypeError(f"empty system or basis in {text!r}")
    try:
        count = int(repeats)
    except ValueError:
        raise argparse.ArgumentTypeError(f"non-integer repeats in {text!r}") from None
    if count < 1:
        raise argparse.ArgumentTypeError(f"repeats must be positive in {text!r}")
    return system, basis, count


def build(results, expected, modes=("cpu", "gpu")):
    results = Path(results)
    records, missing = [], []
    for system, basis, repeats in expected:
        for repeat in range(1, repeats + 1):
            for mode in modes:
                name = f"{system}-{basis}-{mode}-{repeat}"
                path = results / name / "result.json"
                if not path.exists():
                    missing.append(name)
                    records.append({"name": name, "returncode": 1, "process_wall_s": 0.0,
                                    "error": "missing result.json"})
                    continue
                record = json.loads(path.read_text())
                ok = bool(record.get("ok"))
                entry = {"name": name, "returncode": 0 if ok else 1,
                         "process_wall_s": float(record.get("wall_s") or 0.0)}
                if not ok:
                    entry["error"] = record.get("error", "run reported ok=false")
                records.append(entry)
    cases = defaultdict(list)
    for system, basis, _ in expected:
        cases[system].append(basis)
    repeat_counts = {count for _, _, count in expected}
    if len(repeat_counts) != 1:
        raise ValueError(f"summarizer assumes one repeat count per campaign, got {sorted(repeat_counts)}")
    manifest = {
        "command": ["case_dirs_to_campaign.py", str(results)],
        "synthesized_from": "per-case --case directories written by an external launcher",
        "repeats": repeat_counts.pop(),
        "cases": [[system, bases] for system, bases in cases.items()],
        "timing": "fresh-process energy() wall time, including backend initialization",
        "accuracy_tolerance_hartree": 1e-6,
        "records": records,
    }
    failed = [r for r in records if r["returncode"] != 0]
    complete = {"ok": not failed, "count": len(records)}
    return manifest, complete, missing


def discover(results, modes=("cpu", "gpu")):
    """Report the (system, basis, max repeat) triples present, for --expect drafting."""
    seen = defaultdict(set)
    for path in sorted(Path(results).glob("*/result.json")):
        match = NAME.match(path.parent.name)
        if not match or match.group("mode") not in modes:
            continue
        record = json.loads(path.read_text())
        seen[(record["system"], record["basis"])].add(int(match.group("repeat")))
    return {key: max(values) for key, values in seen.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--expect", type=parse_expect, action="append", default=[],
                        metavar="SYSTEM:BASIS:REPEATS",
                        help="a requested case; repeat for each. Omit to use --discover.")
    parser.add_argument("--discover", action="store_true",
                        help="derive the expected set from directories present. This cannot "
                             "detect a case that never ran, so the result is not evidence of "
                             "completeness.")
    parser.add_argument("--force", action="store_true", help="overwrite an existing campaign.json")
    args = parser.parse_args()
    if bool(args.expect) == bool(args.discover):
        parser.error("give either --expect (repeatable) or --discover")
    expected = args.expect or [(system, basis, repeats)
                               for (system, basis), repeats in sorted(discover(args.results).items())]
    if not expected:
        parser.error(f"no case directories found under {args.results}")
    manifest, complete, missing = build(args.results, expected)
    target = args.results / "campaign.json"
    if target.exists() and not args.force:
        parser.error(f"{target} already exists; pass --force to overwrite")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.results / "COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(f"records={len(manifest['records'])} ok={complete['ok']} missing={len(missing)}")
    for name in missing:
        print(f"  missing {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
