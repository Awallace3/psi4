"""Observed native lines only; no inferred identities or missed opportunities."""
import hashlib
import re
import config as s

EVENT = re.compile(r"^\s*cuESTJK plan (created|reused) #([1-9]\d*)\s*$")
STAGE = re.compile(r"^\s*SAPT\(DFT\): (delta HF Dimer|delta HF Monomer A|"
                   r"delta HF Monomer B|DFT Monomer A|DFT Monomer B)\s*$")
COVERAGE = dict(
    scope="observed PRINT3 molecular JK events only; not all whole-call plans",
    sad="internal SAD JK uses PRINT0/1; counts unavailable; SAD_PRINT1 backend proof only",
    silent_compute="same-object ordinary compute reuse is silent, not explicit initialize hits",
    counters="native counters reset per JK object; no global plan IDs",
    context="last printed stage or interaction banner; not object or causal attribution")


def parse(text, arm):
    s.require(arm in ("baseline", "candidate"), "trace arm")
    evidence, created, reused, banners, passed = [], 0, 0, 0, 0
    context = "before first SCF/unknown"
    for number, line in enumerate(text.splitlines(), 1):
        stage = STAGE.fullmatch(line)
        if stage:
            context = stage[1]
        if "SAPT(DFT): Intermolecular Interaction Segment" in line:
            context = "intermolecular interaction (last banner)"
        event = EVENT.fullmatch(line)
        kind, counter = None, None
        if event:
            kind, counter = event[1], int(event[2])
            created += kind == "created"
            reused += kind == "reused"
        elif "cuESTJK plan created" in line or "cuESTJK plan reused" in line:
            raise RuntimeError("malformed native plan trace")
        elif "Device memory, DF int plan:" in line:
            kind = "DF-plan-memory-banner"
            banners += 1
        elif line.strip() == "Re-using passed JK object instead of rebuilding":
            kind = "driver-passed-object"
            passed += 1
        if kind:
            evidence.append(dict(line_number=number, raw=line,
                sha256=hashlib.sha256(line.encode()).hexdigest(), context=context,
                kind=kind, local_counter=counter))
    if arm == "candidate":
        s.require(created == banners, "candidate trace/banner coverage mismatch")
    else:
        s.require(created == reused == 0, "BASE must not emit CAND native events")
        created = banners
    return dict(observed_created=created, observed_explicit_reused=reused,
                df_plan_banners=banners, driver_passed_object_messages=passed,
                coverage=COVERAGE, evidence=evidence,
                raw_log_sha256=hashlib.sha256(text.encode()).hexdigest())


def timer_regions(data):
    """Disjoint requested native regions; do not sum inclusive ancestors."""
    records = data["timer_records"]
    suffixes = ("SCF: JK native initialize", "SCF: Native finalize")
    out = {}
    for suffix in suffixes:
        selected = {k: dict(v) for k, v in records.items() if k.endswith(";"+suffix)}
        paths = list(selected)
        s.require(all(not a.startswith(b+";") for a in paths for b in paths if a != b),
                  "nested native region selection")
        out[suffix] = dict(records=selected,
            wall_s=sum(v["wall_time"] for v in selected.values()),
            n_calls=sum(v["n_calls"] for v in selected.values()))
    out["RHF subphases"] = {k:dict(v) for k,v in records.items()
                           if k.endswith((";RHF: Lagrangian", ";RHF: Finalize cleanup"))}
    # RHF subphases are already inside finalize; never add them a second time.
    return out
