#!/usr/bin/env python3
"""Verify step 14096296.0 independently; never overwrite legacy campaign keys.

One historical, basis-matched whole-call pair, NOT a controlled GPU speedup,
switching-only result or CPKS measurement. No chemistry or remote access.
"""
import argparse
import copy
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys

import publish_grid_evidence as publication
from publish_tiny_grid_pilot import iteration_sections, attach_times

KEY = "cpu_gpu_basis_matched_campaigns"
SOURCE = "protein157-basis-matched-14096296"
GPU = "sessions/protein157-631gss-full2"
SCIENCE = "ff2ebc4c4f4acd2f6c3d21d32a4b1f2e90530f5e"
OVERLAY = "0cb954e4f1cbf3b1c4b95c6b9652a51da08d2467df62041c978a6d229b0051d7"
GEOMETRY = "a1e9c8949bb5950fdb17c07ab718cead826835f042a6b6e94c5cb7dcf75b6b60"
HASHES = {
    GPU + "/result.json": "e4f6f93d7fd0eb8150512213122eb09291053ca3bf80c6e7d786543c52ca0631",
    GPU + "/psi4.out": "0f66e68ba5da5ff2e1e691628b0d6081ec38445eb32b41e911b2d35fbda65a3f",
    GPU + ".meta.txt": "2af98ece8299726a846ccec02c83c016abede64a627a89f6a912ebcc780d6132",
    "sessions/gpu-samples.csv": "263b398378822e74a80315528cebfc50990b07fb7143e1666137090da9c55386",
    "sessions/gpu-samples.meta.txt": "127b8f5f1b872c19e26ac6f4a03114ab240713583d7840aaa774ad31fcfcbb67",
    "reference-13868775.json": "2e2a2921d043f2e275af8d2bf878b0b9169628c3d228c1d221b9bc1db70fc94c",
}
NOTE = (
    "Basis-matched historical whole-call ratio, not a controlled GPU speedup. "
    "SAPT(DFT)-D4(I)/6-31+g**, NBF 1786, Naux 7265, eight threads, induction NONE, "
    "iterative unseeded GRAC, delta-HF, FP64. CPU job 13868775: fixed 99×590; "
    "GPU step 14096296.0: 35×194 → 75×302, factor 1/min 2, cuEST DFJK + GPU SAD, "
    "CUDA LibXC. Geometry and reference hashes match; grids, builds and hardware differ. "
    "Not switching-only gain and not CPKS evidence. No verified fixed75 control for this exact cohort."
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_policy(row, cpu):
    assert row["ok"] and cpu["ok"]
    assert row["slurm_job_id"] == "14096296" and row["slurm_step_id"] == "0"
    assert cpu["slurm_job_id"] == "13868775" and row["reference_job"] == "13868775"
    assert row["source_commit"] == SCIENCE
    assert row["iterator_overlay"]["sha256"] == OVERLAY
    assert row["geometry_sha256"] == hashlib.sha256(cpu["geometry"].encode()).hexdigest() == GEOMETRY
    assert row["reference_result_sha256"] == HASHES["reference-13868775.json"]
    assert row["basis"] == cpu["basis"] == "6-31+g**"
    assert row["nbf"] == cpu["nbf"] == 1786 and row["naux_jkfit"] == 7265
    assert row["threads"] == cpu["threads"] == 8
    assert row["gpu_builder_seen"] and row["sad_gpu_builder_seen"]
    assert not row["cpu_builder_seen"] and not row["direct_seen"]
    assert not cpu["gpu_builder_seen"] and not cpu["disk_df_builder_seen"]
    opts = {k.upper(): v for k, v in row["options"].items()}
    ref = {k.upper(): v for k, v in cpu["options"].items()}
    expected = dict(BASIS="6-31+g**", SCF_TYPE="df", SAPT_DFT_INDUCTION_TYPE="NONE",
        SAPT_DFT_GRAC_COMPUTE="ITERATIVE", SAPT_DFT_GRAC_SEED_CATION=False,
        SAPT_DFT_DO_DHF=True, CUEST_MIXED_PRECISION=False, USE_CUEST=True,
        CUEST_XC=True, CUEST_SAD=True, DFT_RADIAL_POINTS=75, DFT_SPHERICAL_POINTS=302,
        DFT_RADIAL_POINTS_INITIAL=35, DFT_SPHERICAL_POINTS_INITIAL=194,
        DFT_GRID_SWITCH_FACTOR=1.0, DF_BASIS_SCF="def2-universal-jkfit",
        DF_BASIS_MP2="aug-cc-pvdz-ri")
    assert all(opts[k] == v for k, v in expected.items())
    assert opts["SAPT_DFT_FUNCTIONAL"].lower() == "pbe0"
    assert not ref["USE_CUEST"] and (ref["DFT_RADIAL_POINTS"], ref["DFT_SPHERICAL_POINTS"]) == (99, 590)
    allowed = {"CUEST_SAD", "USE_CUEST", "DFT_GRID_SWITCH_FACTOR", "DFT_RADIAL_POINTS",
               "DFT_SPHERICAL_POINTS", "DFT_RADIAL_POINTS_INITIAL", "DFT_SPHERICAL_POINTS_INITIAL"}
    # Independently derive differences, rather than trusting the recorded diff.
    assert {k for k in set(opts) | set(ref) if opts.get(k) != ref.get(k)} <= allowed
    assert row["grid_policy"] == dict(final="75x302", initial="35x194",
                                    minimum_final_iterations=2, switch_factor=1.0)
    proofs = [p for p in row["proofs"] if p["switched"]]
    assert len(proofs) == 6 and not row["policy_checks"]["problems"]
    assert [p["final"] for p in proofs] == [6, 10, 6, 7, 10, 7]
    assert all(p["converged"] and p["final_grid_after"] and p["final"] >= 2 for p in proofs)
    return proofs


def csv_memory(text):
    """nvidia-smi CSV is device-wide; strip both header and value whitespace."""
    rows = [{k.strip(): v.strip() for k, v in r.items()}
            for r in csv.DictReader(io.StringIO(text))]
    assert len(rows) >= 2
    uuids = {r["uuid"] for r in rows}
    assert len(uuids) == 1
    stamp = lambda r: datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
    times = [stamp(r) for r in rows]
    gaps = [(b-a).total_seconds() for a, b in zip(times, times[1:])]
    assert min(gaps) > 0
    start, end = datetime(2026, 10, 16, 11, 24, 24), datetime(2026, 10, 16, 11, 28, 47)
    selected = [r for r, t in zip(rows, times) if start <= t <= end]
    def mib(r, key):
        value, unit = r[key].split()
        assert unit == "MiB"
        return int(value)
    peak = max(mib(r, "memory.used [MiB]") for r in selected)
    assert {mib(r, "memory.total [MiB]") for r in rows} == {143771}
    return dict(peak_mib=peak, peak_gib=peak/1024, samples=len(rows),
        interval_samples=len(selected), max_gap_s=max(gaps), uuid=uuids.pop(),
        scope="Sampled device-wide residency, not exact or process-isolated allocation.")


def partition(row, report, operations, path):
    """Adapt raw timer_records to the existing recursive, additive partition."""
    timers = {k: dict(v, timer_path=k.split(";")) for k, v in row["timer_records"].items()}
    xc = sum(v["wall_time"] for k, v in timers.items()
             if report.normalize(k.split(";")[-1]) in ("RV:Form V", "UV:Form V"))
    functional = sum(v["wall_time"] for k, v in timers.items()
                     if report.normalize(k.split(";")[-1]) in
                     ("cuEST XC:CUDA LibXC Functional", "cuEST XC:Host Functional"))
    adapted = dict(timers=timers, wall_s=row["wall_s"], functional_s=functional,
                   xc_s=xc, result_path=path)
    run = publication.operation_partition(adapted, report, operations, return_run=True)
    operations.operation_values(run)
    return run


def verify(root, docs):
    for relative, expected in HASHES.items():
        assert sha(root / relative) == expected, f"Evidence hash mismatch: {relative}"
    row = publication.load(root / GPU / "result.json")
    cpu = publication.load(root / "reference-13868775.json")
    proofs = validate_policy(row, cpu)
    meta = (root / (GPU + ".meta.txt")).read_text()
    for pin in (SCIENCE, "build_pin=58d061979b", OVERLAY,
                "15363319529b82564ac19a75b4caace4a86e055e87130556e34981c7338ed07d",
                "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c",
                "b8ba011064d688748fca2676801602d6fd8a51712f13235ba2b4511d2976cc98",
                "exit=0 end=2026-10-16T11:28:47-04:00"):
        assert pin in meta
    sections = iteration_sections((root / GPU / "psi4.out").read_text())
    attach_times(sections, row["timer_records"])
    counts = {k: sum(s[k] for s in sections.values()) for k in ("initial", "final", "untagged", "total")}
    assert counts == dict(initial=174, final=46, untagged=50, total=270)
    assert abs(sections["gracA"]["wall_s"] - 123.053579420) < 1e-8
    memory = csv_memory((root / "sessions/gpu-samples.csv").read_text())
    assert (memory["peak_mib"], memory["samples"], memory["interval_samples"]) == (78649, 294, 263)
    assert abs(memory["max_gap_s"] - 1.068) < 1e-8
    sys.path.insert(0, str(docs.resolve()))
    import build_breakdown_2026_10_04 as report
    import operation_pairs as operations
    runs = {}
    for route, raw, path, grid in (
        ("cpu", cpu, "reference-13868775.json", "Fixed 99×590"),
        ("gpu", row, GPU + "/result.json", "35×194 → 75×302 · factor 1/min 2")):
        run = partition(raw, report, operations, f"{SOURCE}/{path}")
        run.update(label=f"{route.upper()} · SAPT(DFT)-D4(I)/6-31+g** · eight threads · "
                         + ("job 13868775" if route == "cpu" else "H200 · step 14096296.0"),
                   grid_title=grid)
        runs[route] = run
    deltas = {k: row["components_hartree"][k]-cpu["components_hartree"][k]
              for k in row["components_hartree"]}
    assert abs(deltas["SAPT TOTAL ENERGY"] + 9.177723487158174e-7) < 1e-15
    assert max(abs(v) for v in deltas.values()) < 1.508e-6
    return dict(id="protein157-basis-matched-14096296", schema="basis-matched-whole-call-v1",
        comparison_cohort="protein157-basis-matched-14096296",
        comparison_type="basis_matched_historical", verified_basis_matched=True, estimated=False,
        title="Protein157 · basis-matched historical whole-call ratio · 6-31+g**",
        grid_arm="coarse75", grid_title="CPU fixed99 / GPU 35→75 · factor 1/min 2",
        comparison_note=NOTE, source_commit=SCIENCE, build_pin="58d061979b",
        iterator_sha256=OVERLAY, cpu_source_commit=cpu["source_commit"], evidence_sha256=HASHES,
        geometry_sha256=GEOMETRY, memory=memory, sections=list(sections.values()),
        iterations=counts, proofs=proofs, component_deltas_eh=deltas,
        ratio=cpu["wall_s"]/row["wall_s"],
        cases={"protein157 / 6-31+g**": dict(**runs, comparison_note=NOTE)})


def publish(root, docs, *, render=False):
    campaign = verify(root, docs)
    path = docs / "grid-evidence.json"
    before = publication.load(path)
    data = copy.deepcopy(before)
    data[KEY] = [campaign]
    assert {k: v for k, v in data.items() if k != KEY} == {k: v for k, v in before.items() if k != KEY}
    # Copy only verified files; reject accidental overwrite of any older evidence.
    for relative in HASHES:
        target = docs / SOURCE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == HASHES[relative]
        else:
            target.write_bytes((root / relative).read_bytes())
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if render:
        publication.render_preserving_history(docs)
    print(f"Verified/published {KEY}; ratio {campaign['ratio']:.9f}×; legacy keys preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    publish(args.root, args.docs, render=args.render)
