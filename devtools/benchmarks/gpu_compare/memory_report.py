#!/usr/bin/env python3
"""Report CPU/GPU memory from successful AND failed campaign records."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def whole_device_samples(base, uuid):
    rows = []
    path = base / "gpu-samples.csv"
    if not path.exists():
        return rows
    with path.open() as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            if row["uuid"] != uuid:
                raise ValueError("Archived GPU sample UUID mismatch")
            stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
            unix = stamp.replace(tzinfo=ZoneInfo("America/New_York")).timestamp()
            rows.append((unix, float(row["memory.used [MiB]"].split()[0])))
    return rows


def campaign_rows(base):
    manifest = json.loads((base / "manifest.json").read_text())
    records = json.loads((base / "records.json").read_text())
    uuid = manifest["gpu_identity"]["uuid"]
    samples = whole_device_samples(base, uuid)
    result = []
    for record in records:
        mem = record.get("memory", {})
        peak = mem.get("peak_rss_kib")
        source = mem.get("peak_rss_source", "")
        if peak is None and "maxrss_kib" in record:
            peak, source = record["maxrss_kib"], "Worker Linux RUSAGE_SELF.ru_maxrss"
        device_peak = mem.get("sampled_gpu_device_peak_mib")
        counts = mem.get("metric_sample_counts", {})
        device_count = counts.get("gpu_device_mib")
        device_source = "Controller UUID-restricted GPU sampler"
        observed = [(t, m) for t, m in samples
                    if record["started_unix_s"] <= t <= record["ended_unix_s"]]
        if device_peak is None:
            device_peak = max((m for _, m in observed), default=None)
            device_count = len(observed)
            device_source = "Archived UUID-restricted whole-device trace"
        result.append(dict(
            campaign=str(base), basis=manifest["config"]["basis"],
            system=record["system"], functional=record["functional"], engine=record["engine"],
            repeat=record["repeat"], profile=record["profile"],
            ok=bool(record["ok"] and record.get("process_returncode") == 0),
            process_returncode=record.get("process_returncode"),
            peak_rss_kib=peak, peak_rss_gib=peak / 1024**2 if peak is not None else None,
            peak_rss_source=source or "Unavailable: worker exited before collecting RSS",
            sampled_tree_rss_peak_kib=mem.get("sampled_tree_rss_peak_kib"),
            sampled_gpu_process_peak_mib=mem.get("sampled_gpu_process_peak_mib"),
            sampled_gpu_device_peak_mib=device_peak,
            gpu_sample_count=device_count, gpu_device_sample_source=device_source,
            gpu_process_sample_count=counts.get("gpu_process_mib"),
            cpu_tree_sample_count=counts.get("tree_rss_kib"),
            peak_rss_error=mem.get("peak_rss_error"),
            gpu_uuid=uuid,
            gpu_memory_scope="Whole-device observed peak, not worker allocation high-water mark",
            error=record.get("error", ""), memory_errors=mem.get("monitor_error"),
            gpu_query_errors=mem.get("gpu_query_errors")))
    hashes = {str(base / name): hashlib.sha256((base / name).read_bytes()).hexdigest()
              for name in ("records.json", "manifest.json", "gpu-samples.csv")
              if (base / name).exists()}
    return result, hashes


def aggregate(rows):
    keys = sorted({(r["basis"], r["system"], r["engine"]) for r in rows if not r["profile"]})
    result = []
    for basis, system, engine in keys:
        group = [r for r in rows if not r["profile"] and
                 (r["basis"], r["system"], r["engine"]) == (basis, system, engine)]
        cpu = [r["peak_rss_gib"] for r in group if r["peak_rss_gib"] is not None]
        gpu = [r["sampled_gpu_device_peak_mib"] / 1024 for r in group
               if r["sampled_gpu_device_peak_mib"] is not None]
        result.append(dict(basis=basis, system=system, engine=engine, attempts=len(group),
                           successes=sum(r["ok"] for r in group), rss_observed=len(cpu),
                           rss_peak_gib=max(cpu) if cpu else None,
                           gpu_observed=len(gpu), sampled_device_peak_gib=max(gpu) if gpu else None))
    return result


def markdown(data):
    out = ["# CPU and GPU memory across both GPU DFT campaigns", "",
           "All unprofiled attempts, including failures. CPU column is the largest recorded "
           "Linux peak RSS across those workers; GPU column is the largest 1-second "
           "whole-device observation within their process intervals. Neither is a sum of repeat peaks.",
           "",
           "| Basis | System | Engine | Passed / attempts | Peak RSS / GiB | RSS coverage | Sampled GPU device peak / GiB |",
           "|---|---|---|---:|---:|---:|---:|"]
    for r in data["summary"]:
        def fmt(value):
            return "N/A" if value is None else f"{value:.3f}"
        out.append(f"| {r['basis']} | {r['system']} | {r['engine']} | "
                   f"{r['successes']}/{r['attempts']} | {fmt(r['rss_peak_gib'])} | "
                   f"{r['rss_observed']}/{r['attempts']} | {fmt(r['sampled_device_peak_gib'])} |")
    out += ["", "## Interpretation and collection limitations", "",
            "- RSS is resident host memory, not virtual memory, and does not include GPU memory. "
            "Linux KiB is converted to GiB by dividing by 1024²; NVML MiB by 1024.",
            "- cc-pVDZ is completed production with three timings per functional. "
            "aug-cc-pVTZ contains only single-run preflights, not production medians.",
            "- Historical RSS is captured inside workers. Protein157/Psi4 exited in native "
            "GPU allocation code before writing a result; its peak RSS is unavailable, **not zero**. "
            "Aggregate coverage exposes this gap. GPU traces and the OOM log are retained.",
            "- Historical GPU memory is whole-device sampling restricted to the campaign's UUID "
            "and correlated to worker Unix start/end times. Samples may miss short peaks; "
            "they cannot recover exact per-process GPU allocation peaks retrospectively.",
            "- Protein157/Psi4 requested a 364.065 GiB device buffer with 138.043 GiB free. "
            "The failed request is not an allocated/observed peak and is not added to the GPU peak.",
            "- Profiles are retained as separately marked rows in JSON/CSV, excluded from this table.",
            "- New runs add a fresh supervisor's Linux child peak RSS even on nonzero/signal exits, "
            "plus controller-side 1-second process-tree RSS and NVML PID-filtered GPU memory. "
            "A whole-job cgroup kill can still prevent final usage collection; saved samples remain "
            "lower bounds. CPU tree sums can double-count shared pages. Unsupported queries remain N/A.",
            "- Observation overhead is external to chemistry timing boundaries but can perturb execution. "
            "Process clocks include supervisor startup/exit, not memory-monitor teardown. "
            "Use the same instrumentation for both engines; historical timings used a different observer.",
            "", "## Audit files", "",
            "`memory-workers.csv` and `memory-report.json` preserve every worker, status, profile flag, "
            "RSS source, GPU scope, sample counts, UUID, and source hashes.", ""]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows, hashes = [], {}
    for base in a.campaign:
        items, sha = campaign_rows(base.resolve())
        rows.extend(items)
        hashes.update(sha)
    data = dict(workers=rows, summary=aggregate(rows), source_sha256=hashes)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "memory-report.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (a.output / "memory-report.md").write_text(markdown(data))
    with (a.output / "memory-workers.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reported {len(rows)} workers; outputs: {a.output}")


if __name__ == "__main__":
    main()
