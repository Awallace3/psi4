"""All six processes and all five rounds, no filtering or timing-data adoption."""
import statistics
from config import SEQUENCE, require

FIELDS = ("initialize_ns", "compute_ns", "combined_ns", "control_ns")


def summarize(values):
    require([v["label"] for v in values] == [v["label"] for v in SEQUENCE], "balanced six order")
    require([v["arm"] for v in values] == [v["arm"] for v in SEQUENCE], "balanced six arms")
    processes = []
    for value in values:
        records = value["records"]
        require([(r["phase"], r["round"]) for r in records] ==
                [("warmup", 0)]+[(p, i) for i in range(1, 6)
                                 for p in ("initialize-compute", "control")], "all eleven snapshots")
        samples = {}
        for field in FIELDS:
            phase = "control" if field == "control_ns" else "initialize-compute"
            sample = [r["timing"][field] for r in records if r["phase"] == phase]
            require(len(sample) == 5 and all(type(x) is int and x > 0 for x in sample),
                    "all five positive raw samples required")
            samples[field] = dict(raw=sample, median=statistics.median(sample),
                                  minimum=min(sample), maximum=max(sample))
        require(all(r["timing"]["combined_ns"] == r["timing"]["initialize_ns"]+
                    r["timing"]["compute_ns"] for r in records
                    if r["phase"] == "initialize-compute"), "timing decomposition")
        processes.append(dict(label=value["label"], arm=value["arm"], samples=samples,
                              warmup=records[0]["timing"]))
    arms = {}
    for arm in ("baseline", "candidate"):
        rows = [v for v in processes if v["arm"] == arm]
        require(len(rows) == 3, "three process repetitions")
        arms[arm] = {k:dict(process_medians=[v["samples"][k]["median"] for v in rows],
                            median_of_process_medians=statistics.median(
                                v["samples"][k]["median"] for v in rows)) for k in FIELDS}
    ratios = {k:arms["baseline"][k]["median_of_process_medians"]/
                arms["candidate"][k]["median_of_process_medians"] for k in FIELDS}
    pairs = []
    for i in (0, 2, 4):
        pair = processes[i:i+2]
        b = next(v for v in pair if v["arm"] == "baseline")
        c = next(v for v in pair if v["arm"] == "candidate")
        pairs.append(dict(labels=[v["label"] for v in pair],
                          ratios={k:b["samples"][k]["median"]/c["samples"][k]["median"]
                                  for k in FIELDS}))
    return dict(processes=processes, arms=arms, ratios_BASE_over_CAND=ratios,
                primary=ratios["combined_ns"], balanced_adjacent_pairs=pairs,
                filtering="none; warmup excluded a priori",
                inference="descriptive n=3; no CI, significance or whole-SAPT claim")
