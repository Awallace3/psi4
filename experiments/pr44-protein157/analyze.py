"""Summarize completed paired jobs without confusing RSS with page cache."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("campaign", type=Path)
args = parser.parse_args()
root = args.campaign
manifest = json.loads((root / "manifest.json").read_text())
receipts = json.loads((root / "submissions.json").read_text())
assert len(receipts) == 2
summary = {"manifest": str(root / "manifest.json"), "arms": {}}
for arm, receipt in zip(["base", "head"], receipts):
    assert receipt["returncode"] == 0
    path = root / "results" / arm / receipt["job_id"]
    assert (path / "COMPLETED").is_file() and not (path / "FAILED").exists(), path
    assert (path / "status").read_text() == "exit_code=0\nterm_received=0\n"
    result = json.loads((path / "result.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    assert result["completed"] and result["sampler_exitcode"] == 0, result
    assert result["source_commit"] == metadata["source_commit"] == manifest["config"][arm]
    assert metadata["config"] == manifest["config"]
    assert metadata["harness_commit"] == manifest["harness_commit"]
    assert metadata["slurm"]["SLURM_JOB_ID"] == receipt["job_id"]
    samples = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
    assert samples
    for field in ["rss_bytes", "scratch_logical_bytes", "scratch_allocated_bytes"]:
        assert any(field in sample for sample in samples), field
    cgroup_values = []
    events = []
    for sample in [*samples, result["final_cgroup"]]:
        for field in ["memory.peak", "memory.max_usage_in_bytes", "memory.current", "memory.usage_in_bytes"]:
            if field in sample:
                cgroup_values.append(int(sample[field]))
        if "memory.events" in sample:
            events.append(dict((key, int(value)) for key, value in
                               (line.split() for line in sample["memory.events"].splitlines())))
    assert cgroup_values, "No cgroup samples"
    observed_events = {key: max(e.get(key, 0) for e in events)
                       for key in {key for e in events for key in e}}
    row = {
        "job_id": receipt["job_id"], "path": str(path), "host": metadata["host"],
        "source_commit": result["source_commit"], "seconds": result["seconds"],
        "process_peak_rss_gib": result["process_peak_rss_bytes"] / 1024**3,
        "cgroup_peak_gib": max(cgroup_values) / 1024**3,
        "scratch_logical_peak_gib": max(s.get("scratch_logical_bytes", 0) for s in samples) / 1024**3,
        "scratch_allocated_peak_gib": max(s.get("scratch_allocated_bytes", 0) for s in samples) / 1024**3,
        "sample_count": len(samples), "sampling_errors": sum("sampling_error" in s for s in samples),
        "cgroup_events_max": observed_events, "variables_Eh": result["variables"],
    }
    assert not observed_events.get("oom", 0) and not observed_events.get("oom_kill", 0), row
    summary["arms"][arm] = row
base, head = summary["arms"]["base"], summary["arms"]["head"]
summary["reductions"] = {}
for field in ["process_peak_rss_gib", "cgroup_peak_gib", "scratch_logical_peak_gib", "scratch_allocated_peak_gib"]:
    delta = base[field] - head[field]
    summary["reductions"][field] = {"saved_gib": delta, "saved_percent": 100 * delta / base[field]}
summary["energy_delta_Eh"] = {key: head["variables_Eh"][key] - value
                              for key, value in base["variables_Eh"].items()}
summary["energy_tolerance_Eh"] = manifest["config"]["energy_tolerance_Eh"]
summary["energy_parity_passed"] = all(abs(v) <= summary["energy_tolerance_Eh"]
                                      for v in summary["energy_delta_Eh"].values())
print(json.dumps(summary, indent=2))
# Save the resource evidence even when numerical parity fails.
(root / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
raise SystemExit(0 if summary["energy_parity_passed"] else 2)
