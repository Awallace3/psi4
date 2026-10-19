"""Owned strict PRINT3 cohort validator; shared legacy validation is unchanged."""
import re

from science import (COMPONENTS, STAGES, finite, geometry, profile, require,
                     sections, solvers, timers)


def print3_basis(text, expected, science):
    """Bind five primary metadata blocks to SCFs, not to loading banners.

    AO is Cartesian and may differ from nbf; SO is the spherical dimension.
    Loading records establish role/keyword identity only, never orbital size/AM.
    This grammar is deliberately specific to this PRINT3 cohort.
    """
    headers = list(re.finditer(r"^.*Primary Basis.*$", text, re.M))
    require(len(headers) == 5, "PRINT3: exactly five Primary Basis blocks")
    finals = list(re.finditer(r"^[ \t]*@DF-(RHF|RKS) Final Energy:.*$", text, re.M))
    require([m[1] for m in finals] == ["RHF"] * 3 + ["RKS"] * 2, "PRINT3: SCF finals")
    previous = -1
    labels = ["Name", "Blend", "Total number of shells", "Number of primitives",
              "Number of AO", "Number of SO", "Maximum AM", "Spherical Harmonics"]
    basis = expected["basis"].upper()
    for i, (stage, header, final) in enumerate(zip(STAGES, headers, finals)):
        require(re.fullmatch(r"[ \t]*==> Primary Basis <==[ \t]*", header[0]),
                "PRINT3: malformed Primary Basis header")
        banners = list(re.finditer(r"^[ \t]*SAPT\(DFT\): " + re.escape(stage) + r"[ \t]*$", text, re.M))
        require(len(banners) == 1, "PRINT3: missing/duplicate stage")
        start = banners[0].start()
        # The final-energy interval is independent of movable stage banners.
        interval = max(0, previous)
        rows = list(re.finditer(r"^[ \t]*@\S+ iter[ \t]+.*$", text[interval:final.start()], re.M))
        require(previous < start < header.start() < final.start() and rows,
                "PRINT3: displaced/reordered primary block")
        end = re.search(r"^[ \t]*-Contraction Scheme:[ \t]*$", text[header.end():], re.M)
        require(end is not None, "PRINT3: missing primary metadata boundary")
        boundary = header.end() + end.start()
        require(boundary < interval + rows[0].start(), "PRINT3: late primary block")
        lines = [line.strip() for line in text[header.end():boundary].splitlines() if line.strip()]
        require(lines and lines[0] == "-AO BASIS SET INFORMATION:" and len(lines) == 9,
                "PRINT3: malformed/duplicate primary fields")
        fields = [re.fullmatch(r"([A-Za-z][A-Za-z ]*?)[ \t]*=[ \t]*(\S+)", line) for line in lines[1:]]
        require(all(fields) and [m[1] for m in fields] == labels,
                "PRINT3: missing/malformed/reordered primary fields")
        values = {m[1]: m[2] for m in fields}
        require(values["Name"] == values["Blend"] == basis, "PRINT3: primary basis name")
        require(values["Spherical Harmonics"] == "TRUE", "PRINT3: primary spherical TRUE required")
        for label in labels[2:7]:
            require(re.fullmatch(r"0|[1-9]\d*", values[label]) is not None,
                    "PRINT3: malformed primary integer: " + label)
            require(int(values[label]) >= (0 if label == "Maximum AM" else 1),
                    "PRINT3: invalid primary integer: " + label)
        require(int(values["Number of SO"]) == expected["nbf"], "PRINT3: primary SO count")
        require(int(values["Number of AO"]) >= int(values["Number of SO"]), "PRINT3: primary AO/SO")
        previous = final.end()

    identities = {
        "ORBITAL": ("BASIS", basis),
        "JKFIT": ("DF_BASIS_SCF", science["DF_BASIS_SCF"].upper()),
        "RIFIT": ("DF_BASIS_MP2", science["DF_BASIS_MP2"].upper()),
    }
    seen = set()
    loaders = list(re.finditer(r"^.*Loading Basis Set.*$", text, re.M))
    require(loaders, "PRINT3: missing role-bound basis loaders")
    for header in loaders:
        require(re.fullmatch(r"[ \t]*=> Loading Basis Set <=[ \t]*", header[0]),
                "PRINT3: malformed loading header")
        # One uninterrupted Name/Role/Keyword record followed by atom source rows.
        record = re.match(
            r"\n(?:[ \t]*\n)*[ \t]+Name: ([^\s]+)[ \t]*\n"
            r"[ \t]+Role: ([^\s]+)[ \t]*\n[ \t]+Keyword: ([^\s]+)[ \t]*\n"
            r"(?:[ \t]+atoms [^\n]+\n)+(?=[ \t]*\n|\Z)", text[header.end():])
        require(record is not None, "PRINT3: malformed/duplicate loading fields")
        # A blank line must not hide conflicting identity fields after a valid
        # prefix. Inspect the remainder up to the next structural section/banner.
        remainder = text[header.end()+record.end():]
        next_section = re.search(r"^[ \t]*(?:=>|==>|SAPT\(DFT\):).*$", remainder, re.M)
        if next_section:
            remainder = remainder[:next_section.start()]
        require(not re.search(r"^[ \t]*(?:Name|Role|Keyword)\b.*$", remainder, re.M),
                "PRINT3: duplicate/conflicting loading fields")
        name, role, keyword = record.group(1, 2, 3)
        require(role in identities and (keyword, name) == identities[role],
                "PRINT3: basis loader role/keyword/name identity")
        seen.add(role)
    require(seen == set(identities), "PRINT3: missing orbital/JKFIT/RIFIT role identity")


def validate_record(d, text, timer_text, profile_text, expected, science, geometry_value, candidate=False):
    """expected derives only from frozen manifest/build probe/current allocation."""
    for key, value in expected.items():
        require(d.get(key) == value, "record identity/options: " + key)
    require(d["options"] == science, "scientific options changed")
    require(d["ok"] is True and d["arm"] == "gpu" and d["threads"] == 8 and d["memory"] == "100 GiB", "runtime policy")
    require(d["gpu_builder_seen"] and d["sad_gpu_builder_seen"] and not any(d[x] for x in
            ["cpu_builder_seen", "direct_seen", "disk_df_builder_seen"]), "GPU backend record")
    for token in ["cuESTJK: GPU-Accelerated", "NVIDIA H200", "Threads set to 8 by Python driver.",
                  "Memory set to 100.000 GiB by Python driver.", "Name: DEF2-UNIVERSAL-JKFIT"]:
        require(token in text, "missing backend/options print: " + token)
    print3_basis(text, expected, science)
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
