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

CACHE = Path.home()/".cache/psi4-major2/fixed-jk-20261019"
FAILED_JOBS = rh.FAILED_JOBS+["14173808"]


def original_input(m=None):
    file = Path(f.INPUT)
    complete = file.parent/"COMPLETE.json"
    value = p.read_json(complete)
    stage = value["stages"][0]
    p.require(value["pid"] == 1443792 and stage["pid"] == 1443792 and
              stage["sad"] and stage["nalpha"] == stage["nbeta"] == f.RANK and
              stage["stage"] == "delta HF Dimer" and stage["file"] == file.name and
              stage["file_sha256"] == f.INPUT_SHA and stage["arrays"]["Da"] == f.DA, "original SAD receipt")
    p.require(p.sha(file) == f.INPUT_SHA, "original frozen NPZ")
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
    return result


def preserve(m):
    p.require(len(m["preservation"]["failed_root_stats"]) == 11, "eleven failed roots")
    fh.preservation(m["preservation"])
    p.require(old_staging() == m["prior_staging"], "both previous staging manifests/batches unchanged")
    original_input(m)


def collect():
    old = p.read_json(Path(r.RUN)/"manifest.json")
    r.validate(old)
    fh.preservation(old["preservation"])
    p.require(p.read_json(Path(r.RUN)/"ENGINE_FAILED.json")["finalized_diagnostic_records"] == 2,
              "retain failed two-record diagnostic")
    stats = dict(old["preservation"]["failed_root_stats"])
    root = Path(r.RUN)
    files = [x for x in root.rglob("*") if x.is_file()]
    p.require(len(files) < 1000, "bounded failed cohort snapshot")
    stats[str(root)] = {str(x.relative_to(root)):fh.stat(x) for x in files}
    p.require(len(stats) == 11, "eleven retained cohorts")
    p.require(p.sha(f.IDENTITY["path"]) == f.IDENTITY["sha256"], "expired identity helper: STOP, no rebuild")
    result = dict(pair=old["diagnostic_pair"], pair_digest=r.PAIR_DIGEST,
        preservation=dict(failed_root_stats=stats,build_sentinel_stats=old["preservation"]["build_sentinel_stats"]),
        prior_staging=old_staging(), input_complete_sha256=original_input(), no_submission=True,
        original_primary_gate_unchanged=True)
    print(json.dumps(result,sort_keys=True,indent=2))


def accounting():
    rows = [line.split("|") for line in h.output("sacct","-n","-P","-j",",".join(FAILED_JOBS),
            "--format=JobIDRaw,State,ExitCode").splitlines()]
    for job in FAILED_JOBS:
        for ident in (job,job+".batch"):
            p.require(any(v[:3] == [ident,"FAILED","1:0"] for v in rows), "old failed accounting: "+ident)
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
    print(json.dumps(dict(no_submission=True, preserved_roots=11, preflight_sha256=p.sha(CACHE/"preflight.json"))))


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
        no_submission=True,preserved_roots=11,test_only_stdout=result.stdout,test_only_stderr=result.stderr,queue=rows)


def main():
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
