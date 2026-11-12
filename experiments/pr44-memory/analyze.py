"""Join native phase/plan probes to process RSS samples by wall-clock time."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("stderr", type=Path)
args = parser.parse_args()
events = []
with args.stderr.open() as handle:
    for line in handle:
        if line.startswith("PR44_MEM "):
            events.append(json.loads(line[len("PR44_MEM "):]))
assert events, "No native probes captured"
events.sort(key=lambda event: event["time"])
stack = []
index = 0
phases = {}
peak = None
count = errors = 0
with (args.output / "samples.jsonl").open() as handle:
    for line in handle:
        sample = json.loads(line)
        count += 1
        errors += "sampling_error" in sample
        while index < len(events) and events[index]["time"] <= sample["time"]:
            event = events[index]
            if event["event"] == "begin":
                stack.append(event["phase"])
            elif event["event"] == "end":
                assert stack and stack[-1] == event["phase"], (stack, event)
                stack.pop()
            index += 1
        phase = " > ".join(stack) if stack else "outside-native-probes"
        entry = phases.setdefault(phase, {"samples": 0, "peak_rss_gib": 0, "peak_scratch_gib": 0})
        entry["samples"] += 1
        rss = sample.get("rss_bytes", 0) / 2**30
        scratch = sample.get("scratch_logical_bytes", 0) / 2**30
        entry["peak_rss_gib"] = max(entry["peak_rss_gib"], rss)
        entry["peak_scratch_gib"] = max(entry["peak_scratch_gib"], scratch)
        if peak is None or rss > peak["rss_gib"]:
            peak = {"time": sample["time"], "rss_gib": rss, "phase": phase,
                    "scratch_gib": scratch, "memory_stat": sample.get("memory.stat")}
plans = [e for e in events if e["event"] == "plan"]
summary = {
    "output": str(args.output), "stderr": str(args.stderr), "sample_count": count,
    "sampling_errors": errors, "sampled_rss_peak": peak,
    "phase_peaks": phases, "allocation_plans": plans,
    "result": json.loads((args.output / "result.json").read_text()) if (args.output / "result.json").exists() else None,
}
(args.output / "phase-analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
