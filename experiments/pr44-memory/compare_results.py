"""Read-only dual-reference assessment of full protein157 memory-isolation runs.

python compare_results.py RESULT_DIRECTORY [RESULT_DIRECTORY ...]
Preserves original 1e-8 assertions while reporting user-approved 1e-7 acceptance.
Never interprets incomplete runs as successful endpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references import compare


def assess(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    expected_geometry = hashlib.sha256(
        (HERE.parent / "pr44-protein157/protein157.xyz").read_bytes()).hexdigest()
    assert metadata["geometry_sha256"] == expected_geometry
    options = json.loads((HERE.parent / "pr44-protein157/config.json").read_text())["options"]
    assert all(metadata["options"][k] == v for k, v in options.items())
    assert metadata.get("mode", "full") == "full", "FDDS replay is not a full SAPT result"
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    row = {
        "directory": str(directory), "source_commit": metadata["source_commit"],
        "core_sha256": metadata["core_sha256"], "harness_commit": metadata["harness_commit"],
        "namespace_workaround": metadata.get("namespace_workaround"),
        "memory_bytes": metadata["config"]["memory_bytes"],
        "completed": result.get("completed", False),
        "original_harness_parity_passed": result.get("parity_passed"),
        "process_peak_rss_bytes": result.get("process_peak_rss_bytes"),
        "final_cgroup": result.get("final_cgroup"),
        "exception": result.get("exception"),
    }
    if row["completed"]:
        row["comparison"] = compare(result["variables"])
        row["sampler_exitcode"] = result.get("sampler_exitcode")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([assess(p) for p in args.directories], indent=2))
