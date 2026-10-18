"""Independent new-cohort checks. No historical core/driver expectations or data writes."""
import math
from pathlib import Path
import re
import statistics

from policy import GATES, SYSTEMS, require

STAGES = ["delta HF Dimer", "delta HF Monomer A", "delta HF Monomer B",
          "DFT Monomer A", "DFT Monomer B"]
COMPONENTS = ["SAPT " + x + " ENERGY" for x in ["ELST", "EXCH", "IND", "DISP", "TOTAL"]]


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def sections(text):
    """Chronological five SCFs, canonical row ordinals, terminal final-grid rows."""
    chunks = re.split(r"@DF-R(?:HF|KS) Final Energy:", text)
    require(len(chunks) == 6, "not exactly five final SCF energies")
    require(re.findall(r"@DF-(RHF|RKS) Final Energy:", text) == ["RHF"] * 3 + ["RKS"] * 2, "SCF types")
    positions, counts = [], []
    for index, (stage, chunk) in enumerate(zip(STAGES, chunks[:5])):
        banner = "SAPT(DFT): " + stage
        hits = list(re.finditer(r"^\s*" + re.escape(banner) + r"\s*$", text, re.M))
        require(len(hits) == 1 and banner in chunk, "missing/duplicate/reordered stage banner")
        positions.append(hits[0].start())
        rows = list(re.finditer(r"^\s*@(\S+) iter\s+(SAD|\d+):.*$", chunk, re.M))
        require(rows and [0 if x[2] == "SAD" else int(x[2]) for x in rows] == list(range(len(rows))), "SCF row chronology")
        require(all(x[1] == ("DF-RHF" if index < 3 else "DF-RKS") for x in rows), "SCF row type")
        tags = [re.findall(r"XC-(INITIAL|FINAL)", x[0]) for x in rows]
        initial = sum(x == ["INITIAL"] for x in tags)
        final = sum(x == ["FINAL"] for x in tags)
        if index >= 3:
            require(initial > 0 and final >= 2 and tags == [["INITIAL"]] * initial + [["FINAL"]] * final,
                    "nonterminal/missing final grid")
            switch = list(re.finditer(re.escape("Near convergence on initial XC grid; continuing on final XC grid."), chunk))
            require(len(switch) == 1 and rows[initial-1].end() < switch[0].start() < rows[initial].start(), "switch chronology")
        else:
            require(initial == final == 0, "HF grid tags")
        counts.append(dict(total=len(rows), initial=initial, final=final))
    require(positions == sorted(positions), "SCF stage chronology")
    require(sum(c["total"] for c in counts) == len(re.findall(r"^\s*@\S+ iter\s+", text, re.M)), "extra SCF rows")
    return counts


def solvers(text):
    parsed = []
    for block in text.split("SAPT Coupled Induction Solver")[1:]:
        match = re.search(r"Convergence\s*=\s*(\S+)", block)
        require(match and "Time [s]" in block, "missing solver header")
        threshold = float(match[1])
        rows = []
        for line in block.split("Time [s]", 1)[1].splitlines():
            fields = line.split()
            if fields and (fields[0] == "Guess" or fields[0].isdigit()):
                require(len(fields) == 4 and fields[0] == ("Guess" if not rows else str(len(rows))), "solver chronology")
                a, b = [float(x.rstrip("*")) for x in fields[1:3]]
                require(finite(a) and finite(b) and min(a, b) >= 0, "nonfinite residual")
                rows.append([a, b])
            elif rows and line.strip().startswith("---"):
                break
        require(threshold == GATES["residual_max"] and len(rows) >= 2 and max(rows[-1]) < threshold, "solver convergence")
        parsed.append(dict(convergence=threshold, residuals=rows, iterations=len(rows)-1, converged=True))
    require(len(parsed) == 2, "exactly two coupled solvers required")
    pos = [m.start() for m in re.finditer("SAPT Coupled Induction Solver", text)]
    require(pos[0] < text.index("SAPT(DFT): DFT Monomer A") < text.index("SAPT(DFT): DFT Monomer B") < pos[1], "solver stage binding")
    return parsed


def geometry(text, value):
    expected = []
    for line in value.splitlines():
        f = line.split()
        if len(f) == 4 and re.fullmatch("[A-Z][a-z]?", f[0]):
            expected.append((f[0], list(map(float, f[1:]))))
    require("Geometry (in Angstrom), charge = 0, multiplicity = 1:" in text, "printed geometry header")
    observed = []
    for line in text.split("Geometry (in Angstrom)", 1)[1].split("Running in", 1)[0].splitlines():
        f = line.split()
        if len(f) == 5 and re.fullmatch("[A-Z][a-z]?", f[0]):
            observed.append((f[0], list(map(float, f[1:4]))))
    require(expected and len(expected) == len(observed), "geometry atom count")
    for (symbol, xyz), (other, coords) in zip(expected, observed):
        require(symbol == other and all(abs(a-b) <= 1e-10 for a, b in zip(xyz, coords)), "geometry coordinates")


def timers(d, printed, candidate):
    records = d["timer_records"]
    require(records and all(finite(v["wall_time"]) and v["wall_time"] >= 0 for v in records.values()), "timer finiteness")
    root = "SAPT(DFT) Energy"
    require(root in records and 0 <= d["wall_s"] - records[root]["wall_time"] < 1, "energy timer coverage")
    for key, v in records.items():
        path = key.split(";")
        require(v["timer_path"] == path and path[0] == root and v["level"] == len(path)-1 and
                v["timer_name"] == path[-1] and v["parent_id"] == (";".join(path[:-1]) or None) and
                (v["parent_id"] is None or v["parent_id"] in records) and v["n_calls"] > 0, "timer identity/orphan")
    for stage in ["Dimer SCF", "Monomer A SCF", "Monomer B SCF", " Monomer A DFT", " Monomer B DFT"]:
        require(root + ";SAPT(DFT):" + stage in records, "missing SCF inclusive timer")
    require(any(k.endswith(";SCF: JK native initialize") for k in records), "missing JK native initialize timer")
    require(sum(k.endswith(";SCF: Native finalize") for k in records) == 5, "missing native finalize timer")
    if candidate:
        for name in ["RHF: Lagrangian", "RHF: Finalize cleanup"]:
            require(sum(k.endswith(";" + name) for k in records) >= 5, "missing RHF subphase " + name)
    require(any("cuEST XC: CUDA LibXC Functional" in k for k in records), "actual CUDA XC timer absent")
    roots = [i for i, line in enumerate(printed.splitlines()) if line.startswith(root)]
    require(len(roots) == 2, "flat/hierarchical timer roots")
    number = r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?"
    pattern = re.compile(r"^((?:\| )*)(.*?)\s*:\s+(?:(" + number + r")u\s+(" +
                         number + r")s\s+(" + number + r")w|(" + number + r")p)\s+(\d+) calls\s*$")
    stack, parsed = [], {}
    for line in printed.splitlines()[roots[-1]:]:
        if not line.strip() or line.startswith("*"):
            continue
        m = pattern.fullmatch(line)
        require(m, "unparsed timer line: " + line)
        depth = m[1].count("|")
        require(depth <= len(stack), "timer depth")
        stack = stack[:depth] + [m[2].strip()]
        path = ";".join(stack)
        require(path not in parsed, "duplicate timer")
        parsed[path] = dict(wall_time=float(m[6] or m[5]), n_calls=int(m[7]))
        if m[6] is None:
            parsed[path].update(user_time=float(m[3]), system_time=float(m[4]))
    require(set(parsed) == set(records), "printed/JSON timer path coverage")
    for path, row in parsed.items():
        require(row["n_calls"] == records[path]["n_calls"], "printed timer calls")
        require(all(abs(row[k] - records[path][k]) < .00051 for k in row if k != "n_calls"), "printed timer values")
    # Root immediate children form a disjoint top-level cut, not a sum of inclusive descendants.
    cut = {k: v["wall_time"] for k, v in records.items() if k.count(";") == 1}
    remainder = records[root]["wall_time"] - sum(cut.values())
    require(remainder >= -1e-6, "overlapping top-level timer cut")
    return dict(top_level_cut_s=cut, energy_unattributed_s=max(0, remainder),
                outside_energy_s=d["wall_s"]-records[root]["wall_time"])


def profile(text):
    require(text.splitlines() and text.splitlines()[0] == "Timing information:", "missing Einsums profile")
    names = []
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        m = re.fullmatch(r"\s*(\d+)ms\s*:\s*(\d+) calls\s*:\s*(\d+)ms per call\s*:\s*(.+)", line)
        require(m and int(m[2]) > 0, "profile record")
        names.append(m[4])
    require(names and len(set(names)) == len(names), "empty/duplicate profile")


def validate_record(d, text, timer_text, profile_text, expected, science, geometry_value, candidate=False):
    """expected derives only from frozen manifest/build probe/current allocation."""
    for key, value in expected.items():
        require(d.get(key) == value, "record identity/options: " + key)
    require(d["options"] == science, "scientific options changed")
    require(d["ok"] is True and d["arm"] == "gpu" and d["threads"] == 8 and d["memory"] == "100 GiB", "runtime policy")
    require(d["gpu_builder_seen"] and d["sad_gpu_builder_seen"] and not any(d[x] for x in
            ["cpu_builder_seen", "direct_seen", "disk_df_builder_seen"]), "GPU backend record")
    for token in ["cuESTJK: GPU-Accelerated", "NVIDIA H200", "Threads set to 8 by Python driver.",
                  "Memory set to 100.000 GiB by Python driver.", "Spherical Harmonics?: true",
                  "Basis Set: 6-31+G**", "Name: DEF2-UNIVERSAL-JKFIT"]:
        require(token in text, "missing backend/options print: " + token)
    require(not any(x in text for x in ["MemDFJK: Density-Fitted", "DiskDFJK", "DirectJK"]), "CPU fallback")
    require(set(re.findall(r"SAD J/K backend: (\S+)", text)) == {"cuESTJK"}, "actual SAD backend")
    require(not re.search(r"Traceback \(most recent call last\)|PsiException|SCFConvergenceError|out of memory", text, re.I), "fatal output")
    require(text.count("Energy and wave function converged.") == 5 and text.count("==> DFT Potential <==") == 2, "SCF/DFT count")
    require(text.count("GRAC shift supplied by the user (not computed).") == 2 and d["grac_shifts_hartree"] == {"A": .136, "B": .136}, "GRAC")
    for letter in "AB":
        require(re.search(rf"Monomer {letter} GRAC Shift\s+0\.136000\b", text), "printed GRAC")
    geometry(text, geometry_value)
    require(d["nbf_monomer_a"] + d["nbf_monomer_b"] == d["nbf"], "monomer basis sizes")
    counts = sections(text)
    require(len(d["proofs"]) == 5, "proof count")
    for i, (count, proof) in enumerate(zip(counts, d["proofs"])):
        require(proof["converged"] is True and proof["nbf"] == d["nbf"] and
                count["total"] == proof["iterations"] + 1 and proof["switched"] == (i >= 3) and
                proof["functional"] == ("PBE0" if i >= 3 else "HF"), "SCF proof binding")
        if i >= 3:
            require(proof["final_grid_after"] and proof["initial"] == count["initial"] and
                    proof["final"] == count["final"], "grid proof differs from chronological output")
    reparsed = solvers(text)
    require(d["cpks_solvers"] == reparsed, "solver histories disagree")
    require(finite(d["wall_s"]) and d["wall_s"] > 0 and finite(d["ru_maxrss_gib"]) and
            0 < d["ru_maxrss_gib"] < 128, "wall/RSS")
    calls = d["cpks_solve_calls_s"]
    require(len(calls) == 2 and all(finite(x) and x > 0 for x in calls) and
            abs(sum(calls)-d["cpks_solve_s"]) < 1e-7 and sum(calls) < d["wall_s"], "solver timing coverage")
    c = d["components_hartree"]
    require(set(c) == set(COMPONENTS) and all(finite(v) for v in c.values()), "energy components finite")
    require(abs(c[COMPONENTS[-1]] - d["returned_energy_hartree"]) <= 1e-12 and
            abs(sum(c[k] for k in COMPONENTS[:-1]) - c[COMPONENTS[-1]]) <= 1e-12, "energy sum")
    for label, key in zip(["Electrostatics", "Exchange", "Induction", "Dispersion", r"Total SAPT\(DFT\)"], COMPONENTS):
        rows = re.findall(r"^\s+" + label + r"\s+([-0-9.]+)\s+\[mEh\]", text, re.M)
        require(rows and abs(float(rows[-1])/1000 - c[key]) < 5.1e-12, "printed components")
    cut = timers(d, timer_text, candidate)
    profile(profile_text)
    return dict(scfs=counts, solvers=reparsed, timer_cut=cut,
                plan_counts="UNAVAILABLE at unchanged production PRINT; focused PRINT=3 tests establish reuse")


def compare(a, b):
    """Never relax science after timings. Also used across all repeats and smokes."""
    for key in COMPONENTS:
        require(abs(a["components_hartree"][key] - b["components_hartree"][key]) <= GATES["component_abs_eh"],
                "paired/repeat energy difference: " + key)
    require([x["iterations"] for x in a["proofs"]] == [x["iterations"] for x in b["proofs"]] and
            [x["iterations"] for x in a["cpks_solvers"]] == [x["iterations"] for x in b["cpks_solvers"]],
            "SCF/solver iteration differences: STOP_FOR_REVIEW; no speedup claim")


def summary(records):
    require(len(records) == 24 and len({x["label"] for x in records}) == 24, "need all 24 unique records")
    output = {}
    for system in SYSTEMS:
        arms = {}
        for arm in ("baseline", "candidate"):
            rows = [x for x in records if x["system"] == system and x["arm"] == arm]
            require(sorted(x["repeat"] for x in rows) == [1, 2, 3], "missing/duplicate repeats")
            walls = [x["data"]["wall_s"] for x in rows]
            arms[arm] = dict(median_s=statistics.median(walls), min_s=min(walls), max_s=max(walls))
        pairs = []
        for repeat in (1, 2, 3):
            a, b = [next(x["data"] for x in records if x["system"] == system and
                         x["repeat"] == repeat and x["arm"] == arm) for arm in ("baseline", "candidate")]
            compare(a, b)
            pairs.append(dict(repeat=repeat, candidate_over_baseline=b["wall_s"]/a["wall_s"],
                              baseline_over_candidate=a["wall_s"]/b["wall_s"]))
        ref = next(x["data"] for x in records if x["system"] == system)
        for row in [x for x in records if x["system"] == system]:
            compare(ref, row["data"])
        output[system] = dict(**arms, adjacent_pair_ratios=pairs,
                              improvement_percent=100*(1-arms["candidate"]["median_s"]/arms["baseline"]["median_s"]))
    return dict(primary="fresh-process energy() wall", systems=output, confidence_intervals=None,
                cpu_gpu_ratio=None, n_per_arm=3, plan_counts="UNAVAILABLE at unchanged PRINT")
