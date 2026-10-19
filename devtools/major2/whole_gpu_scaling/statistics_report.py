"""Retain all valid observations; numerical/trajectory gates are findings, not filters."""
import itertools
import statistics
import config as s
import science
from trace import timer_regions


def classify(a, b):
    deltas = {key:abs(a["components_hartree"][key]-b["components_hartree"][key])
              for key in science.COMPONENTS}
    s.require(all(science.finite(x) for x in deltas.values()), "nonfinite comparison")
    largest = max(deltas, key=deltas.get)
    counts = lambda d:dict(scfs=[v["iterations"] for v in d["proofs"]],
                          solvers=[v["iterations"] for v in d["cpks_solvers"]])
    ca, cb = counts(a), counts(b)
    s.require(len(ca["scfs"]) == len(cb["scfs"]) == 5 and
              len(ca["solvers"]) == len(cb["solvers"]) == 2, "count coverage")
    for value in (ca, cb):
        s.require(all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                      for xs in value.values() for x in xs), "finite integer counts")
    energy_pass = max(deltas.values()) <= s.p.GATES["component_abs_eh"]
    matched = ca == cb
    return dict(component_abs_deltas_eh=deltas, max_component=largest,
        max_component_abs_delta_eh=deltas[largest],
        total_abs_delta_eh=deltas[science.COMPONENTS[-1]],
        total_signed_b_minus_a_eh=b["returned_energy_hartree"]-a["returned_energy_hartree"],
        a_counts=ca, b_counts=cb, component_gate="pass" if energy_pass else "fail",
        iteration_gate="matched" if matched else "different",
        matched_work_acceptance=energy_pass and matched)


def summary(rows):
    s.require(len(rows) == 24 and len({v["label"] for v in rows}) == 24, "all 24 observations")
    s.require([{k:v[k] for k in s.SEQUENCE[0]} for v in rows] == s.SEQUENCE,
              "frozen sequence; no deleted/reordered samples")
    result = {}
    for system in s.p.SYSTEMS:
        arms = {}
        selected = {arm:[r for r in rows if r["system"] == system and r["arm"] == arm]
                    for arm in ("baseline", "candidate")}
        for arm, values in selected.items():
            s.require([v["repeat"] for v in sorted(values, key=lambda v:v["repeat"])] == [1,2,3],
                      "three unique repeats")
            walls = [v["data"]["wall_s"] for v in values]
            s.require(all(science.finite(v) and v > 0 for v in walls), "invalid wall")
            s.require(all(v["data"]["ok"] is True for v in values), "invalid observation")
            s.require(all(v["data"]["nbf"] == s.p.SYSTEMS[system][1] for v in values), "actual NBF")
            representative = sorted(values, key=lambda v:v["data"]["wall_s"])[1]
            arms[arm] = dict(samples_s=walls, min_s=min(walls), max_s=max(walls),
                median_s=statistics.median(walls), representative_label=representative["label"],
                representative_timer_cut=representative["validation"]["timer_cut"],
                timer_cut_by_run={v["label"]:v["validation"]["timer_cut"] for v in values},
                native_regions_by_run={v["label"]:timer_regions(v["data"]) for v in values},
                trace_by_run={v["label"]:v["trace"] for v in values})
        cross = [dict(a=a["label"], b=b["label"], **classify(a["data"], b["data"]))
                 for a,b in itertools.product(selected["baseline"], selected["candidate"])]
        within = {arm:[dict(a=a["label"], b=b["label"], **classify(a["data"], b["data"]))
                       for a,b in itertools.combinations(values, 2)]
                  for arm,values in selected.items()}
        pairs = []
        for repeat in (1,2,3):
            a,b = [next(v for v in selected[arm] if v["repeat"] == repeat)
                   for arm in ("baseline", "candidate")]
            pairs.append(dict(repeat=repeat, first_arm=next(v["arm"] for v in rows
                if v["system"] == system and v["repeat"] == repeat),
                baseline_s=a["data"]["wall_s"], candidate_s=b["data"]["wall_s"],
                baseline_over_candidate=a["data"]["wall_s"]/b["data"]["wall_s"]))
        base, cand = [arms[a]["median_s"] for a in ("baseline", "candidate")]
        matched = all(v["matched_work_acceptance"] for v in cross) and all(
            v["matched_work_acceptance"] for values in within.values() for v in values)
        result[system] = dict(nbf=s.p.SYSTEMS[system][1], **arms,
            seconds_saved=base-cand, percent_reduction=100*(1-cand/base),
            baseline_over_candidate=base/cand, balanced_adjacent_pairs=pairs,
            all_nine_cross_arm_comparisons=cross, within_arm_comparisons=within,
            qualification=("all reported numerical/count gates matched; combined-build observational "
                           "difference, not isolated cache causality" if matched else
                           s.POLICY["failed_gate_qualification"]))
    return dict(primary="fresh whole energy() wall", systems=result,
        n_per_arm_system=3, confidence_intervals=None, significance=None, dropped_samples=0,
        count_definitions=dict(scf_stages=science.STAGES,
            scf_iterations="proof iteration index, excludes initial SAD row",
            solvers=["HF coupled induction","DFT coupled induction"]),
        old_failed_primary="unchanged and failed; this new cohort does not replace it",
        causality="never assign whole wall difference solely to cache; SCF/solver trajectories and RHF differ")
