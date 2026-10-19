"""Foreground staging/audit; reviewed parent ONLY may consume the permanent submission attempt."""
import json
import os
from pathlib import Path
import subprocess
import sys

import fitcase_handoff as f
import handoff as h
import nonc1_handoff as n
import observe as o
import policy as p
import repeatability as r
import repeatability_campaign as c

CACHE = Path.home()/".cache/psi4-major2/p157-repeatability-20261019"
FAILED_JOBS = n.FAILED_JOBS+["14170562"]


def preservation(value):
    p.require(len(value["failed_root_stats"]) == 10, "preserve ten failed roots including eight finalized records")
    f.preservation(value)


def accounting():
    rows = [line.split("|") for line in h.output("sacct", "-n", "-P", "-j", ",".join(FAILED_JOBS),
            "--format=JobIDRaw,State,ExitCode").splitlines()]
    for job in FAILED_JOBS:
        for ident in (job, job+".batch"):
            p.require(any(row[0] == ident and row[1:3] == ["FAILED", "1:0"] for row in rows), "old failed accounting")
    # Deliberately no Elapsed field.
    return rows


def sources(m, full=False):
    for item in [*m["arms"].values(), dict(source=r.HARNESS, **m["harness"])]:
        p.check_source(item["source"], item["pin"], item["files"] if full else None)
        if not full:
            for rel, expected in item["files"].items():
                if rel in p.OWNED_DIFF or rel.startswith("devtools/major2/") or rel in (
                        "psi4/src/export_wavefunction.cc", "psi4/src/export_fock.cc",
                        "psi4/driver/procrouting/proc.py", "psi4/driver/procrouting/scf_proc/scf_iterator.py"):
                    p.require(p.sha(Path(item["source"])/rel) == expected["sha256"], "source sentinel")


def preflight(pin):
    m = json.load(sys.stdin)
    r.validate(m)
    p.require(not Path(r.RUN).exists() and not CACHE.exists() and m["harness"]["pin"] == pin, "fresh namespace")
    sources(m)
    preservation(m["preservation"])
    r.verify_shared(m["diagnostic_pair"])
    p.require(p.sha(c.IDENTITY["path"]) == c.IDENTITY["sha256"], "identity helper changed; no rebuild")
    value = dict(pair_digest=r.PAIR_DIGEST, preservation=m["preservation"],
                 no_submission=True, no_import=True, no_device=True, queue=h.queue(), accounting=accounting())
    CACHE.mkdir()
    o.atomic(CACHE/"preflight.json", value)
    print(json.dumps(dict(no_submission=True, preserved_roots=10, preflight_sha256=p.sha(CACHE/"preflight.json"))))


def verify_bundle(pin, manifest_sha, batch_sha, full_sources=False):
    root = Path(r.RUN)
    m = p.read_json(root/"manifest.json")
    r.validate(m)
    receipt = p.read_json(root/"bundle-receipt.json")
    p.require(m["harness"]["pin"] == receipt["harness_pin"] == pin, "diagnostic harness pin")
    p.require(p.sha(root/"manifest.json") == receipt["manifest_sha256"] == manifest_sha, "manifest hash")
    p.require(p.sha(root/"campaign.sbatch") == receipt["batch_sha256"] == batch_sha, "batch hash")
    p.require((root/"campaign.sbatch").read_text() == c.batch_text(manifest_sha), "batch exact bytes")
    sources(m, full_sources)
    for name in ("STARTED.json", "raw", "build-baseline", "build-candidate"):
        p.require(not (root/name).exists(), "no diagnostic resume/retry")
    pre = p.read_json(root/"logs/preflight.json")
    p.require(pre == p.read_json(CACHE/"preflight.json") and pre["pair_digest"] == r.PAIR_DIGEST, "preflight")
    p.require(pre["preservation"] == m["preservation"], "frozen preservation")
    preservation(pre["preservation"])
    return m, pre


def stage(pin):
    payload = json.load(sys.stdin)
    p.require(set(payload) == {"manifest.json", "campaign.sbatch", "bundle-receipt.json"}, "generated data only")
    m = json.loads(payload["manifest.json"])
    r.validate(m)
    p.require(m["harness"]["pin"] == pin, "stage pin")
    root = Path(r.RUN)
    root.mkdir()
    (root/"logs").mkdir()
    for name, text in payload.items():
        h.durable(root/name, text)
    o.atomic(root/"logs/preflight.json", p.read_json(CACHE/"preflight.json"))
    receipt = json.loads(payload["bundle-receipt.json"])
    verify_bundle(pin, receipt["manifest_sha256"], receipt["batch_sha256"], full_sources=True)
    print(json.dumps(dict(staged=True, no_submission=True, **receipt)))


def audit(pin, mh, bh):
    p.require(not (Path(r.RUN)/"logs/submission.lock").exists(), "permanent attempt exists; never retry")
    m, pre = verify_bundle(pin, mh, bh)
    r.verify_shared(m["diagnostic_pair"])
    p.require(p.sha(c.IDENTITY["path"]) == c.IDENTITY["sha256"], "identity helper changed")
    accounting()
    rows = h.queue()
    subprocess.run(["bash", "-n", str(Path(r.RUN)/"campaign.sbatch")], check=True, timeout=10)
    test = subprocess.run(["sbatch", "--test-only", str(Path(r.RUN)/"campaign.sbatch")],
                          text=True, capture_output=True, timeout=35)
    p.require(test.returncode == 0 and "gpu-h200" in test.stdout+test.stderr, "actual H200 test-only")
    return dict(harness_pin=pin, manifest_sha256=mh, batch_sha256=bh, pair_digest=r.PAIR_DIGEST,
                preflight_sha256=p.sha(CACHE/"preflight.json"), pristine_staging=True,
                old_run_unchanged=True, old_build_sentinels_unchanged=True,
                no_submission=True, test_only_stdout=test.stdout, test_only_stderr=test.stderr, queue=rows)


def main():
    p.require(os.environ["USER"] == "awallace43", "owner")
    mode, pin, *hashes = sys.argv[1:]
    p.require(p.git(r.HARNESS, "rev-parse", "HEAD") == pin, "handoff harness pin")
    if mode == "preflight":
        preflight(pin)
    elif mode == "stage":
        stage(pin)
    elif mode == "audit":
        print(json.dumps(audit(pin, *hashes), indent=2))
    elif mode == "parent-submit":
        # No generic 'submit' alias. Review token is never implied by preparation.
        p.require(os.environ.get("MAJOR2_PARENT_REVIEW") == r.REVIEW_ID, "explicit parent review required")
        gates = audit(pin, *hashes)
        o.atomic(Path(r.RUN)/"logs/pre-submit-gates.json", gates)
        h.submit(pin, *hashes, cohort=r, verifier=verify_bundle, cache=CACHE, failed_jobs=FAILED_JOBS)
    else:
        raise RuntimeError("unknown handoff mode")


if __name__ == "__main__":
    main()
