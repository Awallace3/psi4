#!/usr/bin/env python3
"""Build a share-safe, self-contained timing and memory dashboard."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import statistics

from memory_report import campaign_rows
from dashboard_view import render_page


def grid_count(record, base):
    if record.get("grid_points") is not None:
        return record["grid_points"], "worker record"
    if record["engine"] == "pyscf" and record.get("output"):
        # Archive paths are local; never read the recorded remote absolute path.
        path = base / "results" / Path(record["output"]).name / "pyscf.out"
        if path.exists():
            counts = re.findall(r"\btot grids\s*=\s*(\d+)", path.read_text())
            if counts:
                return int(counts[-1]), "constructed grid (log)"
    return None, "not exported"


def data_for(campaigns):
    comparisons = []
    for base in campaigns:
        records = json.loads((base / "records.json").read_text())
        summaries = json.loads((base / "summary.json").read_text())
        memory, _ = campaign_rows(base)
        config = json.loads((base / "config.json").read_text())
        for summary in summaries:
            item = dict(basis=config["basis"], system=summary["system"],
                        functional=summary["functional"], rung=summary["rung"],
                        energy_delta_eh=summary.get("energy_delta_pyscf_minus_psi4_eh"),
                        energy_review=summary.get("energy_review", False))
            counts = sorted({r["nbf"] for r in records if r.get("nbf") is not None and
                             (r["system"], r["functional"]) == (item["system"], item["functional"])})
            item["nbf"] = counts[0] if len(counts) == 1 else None
            for engine in ("psi4", "pyscf"):
                group = [r for r in records if not r["profile"] and
                         (r["system"], r["functional"], r["engine"]) ==
                         (item["system"], item["functional"], engine)]
                good = [r for r in group if r["ok"] and r.get("process_returncode") == 0]
                observations = [r for r in memory if not r["profile"] and
                                (r["system"], r["functional"], r["engine"]) ==
                                (item["system"], item["functional"], engine)]
                def peak(key):
                    return max((r[key] for r in observations if r[key] is not None), default=None)
                times = [r["calculation_wall_s"] for r in good]
                grid_info = [grid_count(r, base) for r in group]
                grids = sorted({n for n, _ in grid_info if n is not None})
                # No paths, geometry, environment manifests or private identifiers are published.
                item[engine] = dict(passed=len(good), attempted=len(group),
                    calculation_s=statistics.median(times) if times else None,
                    time_min=min(times) if times else None, time_max=max(times) if times else None,
                    peak_rss_gib=peak("peak_rss_gib"),
                    rss_coverage=sum(r["peak_rss_gib"] is not None for r in observations),
                    grid_storage_points=grids[0] if len(grids) == 1 else None,
                    grid_count_source=("constructed grid (log)" if any(
                        src == "constructed grid (log)" for _, src in grid_info)
                        else "worker record" if grids else "not exported"),
                    sampled_gpu_device_peak_gib=(peak("sampled_gpu_device_peak_mib") / 1024
                                                if peak("sampled_gpu_device_peak_mib") is not None else None))
            comparisons.append(item)
    return comparisons


def render(rows):
    def num(value, digits=2):
        return "N/A" if value is None else f"{value:.{digits}f}"
    table = []
    for r in rows:
        cells = [html.escape(r["basis"]), html.escape(r["system"]),
                 html.escape(r["functional"]), num(r.get("nbf"), 0)]
        p, g = r["psi4"], r["pyscf"]
        for e in (p, g):
            label = num(e.get("grid_storage_points"), 0)
            if e.get("grid_count_source") == "constructed grid (log)":
                label += "<small>constructed; log recovery</small>"
            cells.append(label)
        status = "unmatched grids"
        if p["passed"] != p["attempted"] or g["passed"] != g["attempted"]:
            status = "failed arm"
        elif r["energy_review"]:
            status = "energy review"
        for e in (p, g):
            cells.append(f"{num(e['calculation_s'])}<small>{e['passed']}/{e['attempted']} passed</small>")
        advantage = None
        if status != "failed arm" and p["calculation_s"] and g["calculation_s"]:
            advantage = g["calculation_s"] / p["calculation_s"]
        cells.append(num(advantage) + ("×" if advantage else ""))
        cells += [num(p["peak_rss_gib"], 3), num(g["peak_rss_gib"], 3),
                  num(p["sampled_gpu_device_peak_gib"]), num(g["sampled_gpu_device_peak_gib"])]
        cells.append("N/A" if r["energy_delta_eh"] is None else f"{r['energy_delta_eh']:+.2e}")
        cells.append(f"<span class='warn'>{status}</span>")
        table.append(f"<tr data-basis='{html.escape(r['basis'])}' data-system='{html.escape(r['system'])}' "
                     f"data-functional='{html.escape(r['functional'])}'>"
                     + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return render_page(rows, table, stamp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = data_for(a.campaign)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(render(rows) + "\n")
    print(f"Rendered {len(rows)} comparisons: {a.output}")


if __name__ == "__main__":
    main()
