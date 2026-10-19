"""Bounded login staging only; parent authorization required for permanent one-shot submit."""
import json
import os
from pathlib import Path
import subprocess
import sys

import fixed_jk as f
import fixed_jk_campaign as c
import fitcase_handoff as fh
import handoff as h
import observe as o
import policy as p
import repeatability as r
import repeatability_handoff as rh
import workspace_control as w

CACHE = Path.home()/".cache/psi4-major2/workspace-20261019"
FAILED_JOBS = rh.FAILED_JOBS+["14173808", "14174849"]
COMPLETED_JOB = "14175243"


def original_input(m=None):
    file = Path(f.INPUT)
    complete = file.parent/"COMPLETE.json"
    value = p.read_json(complete)
    stage = value["stages"][0]
    p.require(value["pid"] == 1443792 and stage["pid"] == 1443792 and
              stage["sad"] and stage["nalpha"] == stage["nbeta"] == f.RANK and
              stage["stage"] == "delta HF Dimer" and stage["file"] == file.name and
              stage["file_sha256"] == f.INPUT_SHA and stage["arrays"]["Da"] == f.DA, "original SAD receipt")
    # The inherited failed-root stat proof includes this original archive.
    # Full archive/factor hashes remain mandatory on the allocated host.
    p.require(file.is_file(), "original frozen NPZ exists")
    if m is not None:
        p.require(p.sha(complete) == m["input_complete_sha256"], "original COMPLETE unchanged")
    return p.sha(complete)


def old_staging():
    result = {}
    for name, root in (("v1",r.RUN.removesuffix("-v2")), ("v2",r.RUN)):
        root = Path(root)
        result[name] = dict(root=str(root), manifest_sha256=p.sha(root/"manifest.json"),
                            batch_sha256=p.sha(root/"campaign.sbatch"))
    p.require(not (Path(result["v1"]["root"])/"STARTED.json").exists(), "retain unsubmitted v1")
    p.require(p.read_json(Path(r.RUN)/"STARTED.json")["job"] == "14173808", "old failed cohort")
    root = Path(f.FIXED_V1)
    result["fixed-v1"] = dict(root=str(root), manifest_sha256=p.sha(root/"manifest.json"),
                             batch_sha256=p.sha(root/"campaign.sbatch"))
    p.require(p.read_json(root/"STARTED.json")["job"] == "14174849", "original factor job")
    return result


def frozen_origin():
    factor = Path(f.FROZEN_FACTOR["root"])
    p.require(p.sha(factor/"FACTOR.json") == f.FROZEN_FACTOR["files"]["FACTOR.json"],
              "frozen factor receipt bytes")
    receipt = p.read_json(factor/"FACTOR.json")
    for name in ("C.npy", "D.npy"):
        p.require((factor/name).is_file() and not (factor/name).stat().st_mode & 0o222,
                  "frozen factor readonly stat proof")
    root = Path(f.FIXED_V1)
    outputs = list((root/"raw").glob("fixed-*/*.npz"))
    complete = list((root/"raw").glob("fixed-*/COMPLETE.json"))
    p.require(not outputs and not complete and not (root/"ENDPOINT-COMPLETE.json").exists(),
              "v1 has no JK outputs or complete processes")
    p.require(p.read_json(root/"raw/factor-prepare.exit.json") == dict(exit=0, interrupted=False) and
              p.read_json(root/"raw/fixed-baseline-r1.exit.json") == dict(exit=1, interrupted=False),
              "original factor success and startup failure")
    trace = root/"raw/fixed-baseline-r1/plan.out"
    p.require(p.sha(trace) == "d7cd3bba1ae1977f788df11b53077c76eae30d10fa4961dae52560c8e8488777" and
              f.traces(trace.read_bytes().decode("utf-8"), "baseline") ==
              dict(created=0, reused=0, counters=[]), "captured startup trace")
    return dict(frozen_factor=f.FROZEN_FACTOR, factor_receipt=receipt,
                jk_outputs=[], completed_jk_processes=[], native_factor_exit=0, first_base_exit=1)


def completed_control():
    """Bounded stats + small metadata hashes ONLY; never old array rehash/JK."""
    root = Path(w.OLD_RUN)
    files = sorted(x for x in root.rglob("*") if x.is_file())
    p.require(len(files) < 1000 and all(not x.is_symlink() for x in files), "bounded completed root")
    stats = {str(x.relative_to(root)):fh.stat(x) for x in files}
    metadata = [x for x in files if x.suffix == ".json" or x.name in
                ("campaign.sbatch", "plan.out", "job-id.txt")]
    p.require(all(x.stat().st_size < 16*1024**2 for x in metadata) and
              sum(x.stat().st_size for x in metadata) < 32*1024**2, "small completed metadata")
    hashes = {str(x.relative_to(root)):p.sha(x) for x in metadata}
    m = p.read_json(root/"manifest.json")
    p.require(m["harness"]["pin"] == w.PARENT and
              "numerical_control" not in m and m["frozen_factor"] == f.FROZEN_FACTOR,
              "original completed UNSET manifest")
    p.require(p.read_json(root/"STARTED.json")["job"] == COMPLETED_JOB and
              p.read_json(root/"ENDPOINT-COMPLETE.json") ==
              dict(outputs=24, computes=24, accepted_speedup_data=False), "completed endpoint identity")
    failures = [x for x in files if "FAILED" in x.name or "AMBIGUOUS" in x.name]
    p.require(not failures, "old completed failure markers")
    report = p.read_json(root/"ENDPOINT-DIAGNOSTIC.json")
    w.summary(report)
    pids = set()
    for row in f.sequence():
        name = row["label"]
        p.require(p.read_json(root/"raw"/(name+".exit.json")) ==
                  dict(exit=0, interrupted=False), "old complete process exit")
        value = p.read_json(root/"raw"/name/"COMPLETE.json")
        pids.add(value["pid"])
        p.require(value["job"] == COMPLETED_JOB and value["label"] == name and
                  value["arm"] == row["arm"] and value["actual_backend"] == "cuESTJK" and
                  value["factor_receipt_sha256"] == f.FROZEN_FACTOR["files"]["FACTOR.json"] and
                  value["trace"] == f.expected_trace(row["arm"], "D"), "old completed provenance")
        p.require([v["state"] for v in value["records"]] == list(f.STATES), "old four states")
        for v in value["records"]:
            array = root/"raw"/name/v["file"]
            p.require(array.stat().st_size == v["file_bytes"] and not array.stat().st_mode & 0o222,
                      "old readonly output stats, no rehash")
    p.require(len(pids) == 6 and len(list((root/"raw").glob("fixed-*/*.npz"))) == 24,
              "old six processes/24 readonly outputs")
    return dict(root=str(root), job=COMPLETED_JOB, numerical_control={},
                stats=stats, hashes=hashes, report=report)


def preserve(m):
    p.require(len(m["preservation"]["failed_root_stats"]) == 12, "twelve failed roots")
    fh.preservation(m["preservation"])
    p.require(old_staging() == m["prior_staging"], "both previous staging manifests/batches unchanged")
    original_input(m)
    frozen_origin()
    p.require(completed_control() == m["completed_control"], "completed root/report/metadata unchanged")


def collect():
    old = p.read_json(Path(f.FIXED_V1)/"manifest.json")
    p.require(old["harness"]["pin"] == f.PARENT_PIN and
              old["input_sha256"] == f.INPUT_SHA, "original fixed harness/input pin")
    r.validate_pair(old["diagnostic_pair"])
    fh.preservation(old["preservation"])
    stats = dict(old["preservation"]["failed_root_stats"])
    root = Path(f.FIXED_V1)
    files = [x for x in root.rglob("*") if x.is_file()]
    p.require(len(files) < 1000, "bounded failed cohort snapshot")
    stats[str(root)] = {str(x.relative_to(root)):fh.stat(x) for x in files}
    p.require(len(stats) == 12, "twelve retained cohorts")
    p.require(p.sha(f.IDENTITY["path"]) == f.IDENTITY["sha256"], "expired identity helper: STOP, no rebuild")
    result = dict(pair=old["diagnostic_pair"], pair_digest=r.PAIR_DIGEST,
        preservation=dict(failed_root_stats=stats,build_sentinel_stats=old["preservation"]["build_sentinel_stats"]),
        prior_staging=old_staging(), input_complete_sha256=original_input(), no_submission=True,
        original_primary_gate_unchanged=True, frozen_origin=frozen_origin(),
        completed_control=completed_control())
    print(json.dumps(result,sort_keys=True,indent=2))


def accounting():
    p.require(COMPLETED_JOB not in FAILED_JOBS, "never classify completed cohort as failed")
    rows = [line.split("|") for line in h.output("sacct","-n","-P","-j",",".join(FAILED_JOBS+[COMPLETED_JOB]),
            "--format=JobIDRaw,State,ExitCode").splitlines()]
    for job in FAILED_JOBS:
        for ident in (job,job+".batch"):
            p.require(any(v[:3] == [ident,"FAILED","1:0"] for v in rows), "old failed accounting: "+ident)
    for ident in (COMPLETED_JOB, COMPLETED_JOB+".batch"):
        p.require(any(v[:3] == [ident,"COMPLETED","0:0"] for v in rows), "old completed accounting: "+ident)
    return rows


def sources(m, full=False):
    for item in [*m["arms"].values(),dict(source=f.HARNESS, **m["harness"])]:
        p.check_source(item["source"],item["pin"],item["files"] if full else None)
        if not full:
            for rel, expected in item["files"].items():
                if rel.startswith("devtools/major2/") or rel in p.OWNED_DIFF:
                    p.require(p.sha(Path(item["source"])/rel) == expected["sha256"], "source sentinel")


def preflight(pin):
    m = json.load(sys.stdin)
    f.validate(m)
    p.require(not Path(f.RUN).exists() and not CACHE.exists() and m["harness"]["pin"] == pin, "fresh namespace")
    sources(m)
    preserve(m)
    r.verify_shared(m["diagnostic_pair"])
    p.require(p.sha(f.IDENTITY["path"]) == f.IDENTITY["sha256"], "identity helper expired: STOP, no rebuild")
    value = dict(no_submission=True, pair_digest=r.PAIR_DIGEST, preservation=m["preservation"],
                 prior_staging=m["prior_staging"], input_complete_sha256=m["input_complete_sha256"],
                 no_import=True, no_device=True, queue=h.queue(), accounting=accounting())
    CACHE.mkdir()
    o.atomic(CACHE/"preflight.json",value)
    print(json.dumps(dict(no_submission=True, preserved_roots=12, preflight_sha256=p.sha(CACHE/"preflight.json"))))


def verify_bundle(pin, mh, bh, full_sources=False):
    root = Path(f.RUN)
    m = p.read_json(root/"manifest.json")
    f.validate(m)
    receipt = p.read_json(root/"bundle-receipt.json")
    p.require(m["harness"]["pin"] == receipt["harness_pin"] == pin, "new harness pin")
    p.require(p.sha(root/"manifest.json") == receipt["manifest_sha256"] == mh and
              p.sha(root/"campaign.sbatch") == receipt["batch_sha256"] == bh, "bundle bytes")
    p.require((root/"campaign.sbatch").read_text() == c.batch_text(mh), "exact batch")
    sources(m,full_sources)
    for name in ("STARTED.json","raw","build-baseline","build-candidate"):
        p.require(not (root/name).exists(), "no resume/retry")
    pre = p.read_json(root/"logs/preflight.json")
    p.require(pre == p.read_json(CACHE/"preflight.json") and pre["preservation"] == m["preservation"] and
              pre["pair_digest"] == r.PAIR_DIGEST, "frozen preflight bytes")
    preserve(m)
    return m,pre


def stage(pin):
    payload = json.load(sys.stdin)
    p.require(set(payload) == {"manifest.json","campaign.sbatch","bundle-receipt.json"}, "generated data only")
    m = json.loads(payload["manifest.json"])
    f.validate(m)
    p.require(m["harness"]["pin"] == pin, "stage harness pin")
    root = Path(f.RUN)
    root.mkdir()
    (root/"logs").mkdir()
    for name, text in payload.items():
        h.durable(root/name,text)
    o.atomic(root/"logs/preflight.json",p.read_json(CACHE/"preflight.json"))
    receipt = json.loads(payload["bundle-receipt.json"])
    verify_bundle(pin,receipt["manifest_sha256"],receipt["batch_sha256"],True)
    print(json.dumps({**receipt,"staged":True,"no_submission":True}))


def audit(pin,mh,bh):
    p.require(not (Path(f.RUN)/"logs/submission.lock").exists(), "permanent attempt consumed")
    m, _ = verify_bundle(pin,mh,bh)
    r.verify_shared(m["diagnostic_pair"])
    p.require(p.sha(f.IDENTITY["path"]) == f.IDENTITY["sha256"], "identity helper expired; STOP")
    accounting()
    rows = h.queue()
    subprocess.run(["bash","-n",str(Path(f.RUN)/"campaign.sbatch")],check=True,timeout=10)
    result = subprocess.run(["sbatch","--test-only",str(Path(f.RUN)/"campaign.sbatch")],
                            capture_output=True,text=True,timeout=35)
    p.require(result.returncode == 0 and "gpu-h200" in result.stdout+result.stderr, "actual H200 test-only")
    return dict(harness_pin=pin,manifest_sha256=mh,batch_sha256=bh,
        preflight_sha256=p.sha(CACHE/"preflight.json"), pair_digest=r.PAIR_DIGEST,
        pristine_staging=True,old_run_unchanged=True,old_build_sentinels_unchanged=True,
        no_submission=True,preserved_roots=12,frozen_origin=frozen_origin(),
        test_only_stdout=result.stdout,test_only_stderr=result.stderr,queue=rows)


def main():
    w.check_env(os.environ)
    p.require(os.environ["USER"] == "awallace43", "owner")
    mode, pin, *hashes = sys.argv[1:]
    p.require(p.git(f.HARNESS,"rev-parse","HEAD") == pin, "handoff harness pin")
    if mode == "collect":
        collect()
    elif mode == "preflight":
        preflight(pin)
    elif mode == "stage":
        stage(pin)
    elif mode == "audit":
        print(json.dumps(audit(pin,*hashes),indent=2))
    elif mode == "parent-submit":
        p.require(os.environ.get("MAJOR2_PARENT_REVIEW") == f.REVIEW_ID, "explicit parent review ONLY")
        gates = audit(pin,*hashes)
        o.atomic(Path(f.RUN)/"logs/pre-submit-gates.json",gates)
        h.submit(pin,*hashes,cohort=f,verifier=verify_bundle,cache=CACHE,failed_jobs=FAILED_JOBS)
    else:
        raise RuntimeError("unknown fixed handoff mode")


if __name__ == "__main__":
    main()
