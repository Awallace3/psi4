"""New namespace, bounded foreground provenance; parent-only one-shot submit."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import artifacts as a
import campaign
import fitcase as f
import fitcase_handoff as old
import handoff as h
import nonc1 as n
import policy as p

CACHE = Path.home()/".cache/psi4-major2/nonc1-20261019"
FAILED_JOBS = old.FAILED_JOBS+["14169328"]


def fresh_candidate():
    build = Path(n.BUILDS["candidate"])
    p.require(not build.exists() and not build.is_symlink(), "candidate must be fresh, never imported")


def preflight():
    p.require(not Path(n.RUN).exists() and not CACHE.exists(), "new nonc1 namespace already exists")
    origin = json.load(sys.stdin)
    n.verify_shared(origin)
    fresh_candidate()
    scientific = n.proof(n.SOURCES)
    test = n.test_proof(n.TEST_SOURCE)
    # Reuse captured old hashes + verify stats. No rehash of nine old run roots,
    # and absolutely no walks of old build trees.
    prior = p.read_json(Path(f.RUN)/"logs/preflight.json")
    old.preservation(prior)
    stats = dict(prior["failed_root_stats"])
    root = Path(f.RUN)
    names = a.hashed_tree(root)
    p.require(len(names) < 1000, "unbounded last failed root")
    p.require(p.read_json(root/"STARTED.json")["job"] == "14169328" and
              p.read_json(root/"ENGINE_FAILED.json")["production_records"] == 0, "last failed origin")
    stats[str(root)] = {name: old.stat(root/name) for name in names}
    p.require(len(stats) == 9, "preserve all nine old failed roots")
    # Dedicated old source checkouts remain pinned as well.
    p.check_source(f.SOURCES["candidate"], f.CANDIDATE)
    p.check_source(f.TEST_SOURCE, f.TEST_PIN)
    p.check_source(f.HARNESS, "11e852dc4a41835f5c27bc198b01b8fb644d03ac")
    rows = accounting()
    value = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 no_chemistry=True, no_device_calls=True, no_new_import=True,
                 baseline_origin_digest=n.ORIGIN_DIGEST, constructor_proof=scientific,
                 test_source_proof=test, failed_root_stats=stats,
                 build_sentinel_stats=prior["build_sentinel_stats"],
                 last_failed_root_sha256=names,
                 previous_preflight_sha256=p.sha(root/"logs/preflight.json"),
                 queue=h.queue(), accounting=rows)
    CACHE.mkdir()
    h.durable(CACHE/"preflight.json", value)
    print(json.dumps(dict(no_submission=True, cache=str(CACHE), failed_roots=9,
                         preflight_sha256=p.sha(CACHE/"preflight.json"))))


def accounting():
    rows = [line.split("|") for line in h.output("sacct", "-n", "-P", "-j", ",".join(FAILED_JOBS),
                    "--format=JobIDRaw,State,ExitCode,Elapsed").splitlines()]
    for job in FAILED_JOBS:
        for ident in (job, job+".batch"):
            p.require(any(row[0] == ident and row[1:3] == ["FAILED", "1:0"] for row in rows),
                      "old accounting: "+ident)
    return rows


def verify_bundle(pin, manifest_sha, batch_sha, *, full_sources=False):
    root = Path(n.RUN)
    m = p.read_json(root/"manifest.json")
    p.validate_manifest(m)
    p.require(m["build_mode"] == n.MODE and m["harness"]["pin"] == pin, "nonc1 harness pin")
    receipt = p.read_json(root/"bundle-receipt.json")
    p.require(p.sha(root/"manifest.json") == receipt["manifest_sha256"] == manifest_sha, "manifest hash")
    p.require(p.sha(root/"campaign.sbatch") == receipt["batch_sha256"] == batch_sha, "batch hash")
    p.require((root/"campaign.sbatch").read_text() == campaign.batch_text(manifest_sha, hybrid=True), "batch bytes")
    for item in [*m["arms"].values(), m["test_source"], dict(source=n.HARNESS, **m["harness"])]:
        p.check_source(item["source"], item["pin"], item["files"] if full_sources else None)
        if not full_sources:
            for name in item["files"]:
                if name in p.OWNED_DIFF or name in ("psi4/src/psi4/libfock/jk.cc", "tests/pytests/test_cuest_jk.py",
                      "tests/pytests/conftest.py", "tests/pytests/addons.py") or name.startswith("devtools/major2/"):
                    p.require(p.sha(Path(item["source"])/name) == item["files"][name]["sha256"], "source sentinel")
    p.require(n.proof(n.SOURCES) == m["constructor_proof"], "scientific source proof")
    p.require(n.test_proof(n.TEST_SOURCE) == m["test_source"]["proof"], "new test proof")
    fresh_candidate()
    for name in ("STARTED.json", "build-baseline", "build-candidate", "raw"):
        p.require(not (root/name).exists(), "bundle already started")
    pre = p.read_json(root/"logs/preflight.json")
    p.require(pre == p.read_json(CACHE/"preflight.json") and pre["baseline_origin_digest"] == n.ORIGIN_DIGEST,
              "preflight identity")
    old.preservation(pre)
    return m, pre


def stage(pin):
    payload = json.load(sys.stdin)
    p.require(set(payload) == {"manifest.json", "campaign.sbatch", "bundle-receipt.json"}, "generated data only")
    root = Path(n.RUN)
    p.require(not root.exists(), "new nonc1 root already exists")
    m = json.loads(payload["manifest.json"])
    p.validate_manifest(m)
    p.require(m["harness"]["pin"] == pin, "stage harness pin")
    root.mkdir()
    (root/"logs").mkdir()
    for name, text in payload.items():
        h.durable(root/name, text)
    h.durable(root/"logs/preflight.json", p.read_json(CACHE/"preflight.json"))
    receipt = json.loads(payload["bundle-receipt.json"])
    verify_bundle(pin, receipt["manifest_sha256"], receipt["batch_sha256"], full_sources=True)
    print(json.dumps(dict(staged=True, no_submission=True, **receipt)))


def audit(pin, manifest_sha, batch_sha):
    p.require(not (Path(n.RUN)/"logs/submission.lock").exists(), "permanent submission lock exists")
    m, pre = verify_bundle(pin, manifest_sha, batch_sha)
    n.verify_shared(m["baseline_origin"])
    accounting()
    subprocess.run(["bash", "-n", str(Path(n.RUN)/"campaign.sbatch")], check=True, timeout=10)
    rows = h.queue()
    result = subprocess.run(["sbatch", "--test-only", str(Path(n.RUN)/"campaign.sbatch")],
                            capture_output=True, text=True, timeout=35)
    p.require(result.returncode == 0 and "gpu-h200" in result.stdout+result.stderr, "actual H200 test-only")
    print(json.dumps(dict(harness_pin=pin, manifest_sha256=manifest_sha, batch_sha256=batch_sha,
                         preflight_sha256=p.sha(CACHE/"preflight.json"), baseline_origin_digest=n.ORIGIN_DIGEST,
                         pristine_staging=True, old_run_unchanged=True, old_build_sentinels_unchanged=True,
                         queue=rows, test_only_stdout=result.stdout, test_only_stderr=result.stderr,
                         no_submission=True), indent=2))


def store_gates(pin, manifest_sha, batch_sha):
    value = json.load(sys.stdin)
    verify_bundle(pin, manifest_sha, batch_sha)
    p.require(value["harness_pin"] == pin and value["manifest_sha256"] == manifest_sha and
              value["batch_sha256"] == batch_sha and value["baseline_origin_digest"] == n.ORIGIN_DIGEST and
              value["preflight_sha256"] == p.sha(CACHE/"preflight.json") and
              value["pristine_staging"] and value["old_run_unchanged"] and value["old_build_sentinels_unchanged"] and
              value["no_submission"] and "gpu-h200" in value["test_only_stdout"]+value["test_only_stderr"],
              "parent gates identity")
    h.durable(Path(n.RUN)/"logs/pre-submit-gates.json", value)
    print(json.dumps(dict(gates_stored=True, no_submission=True)))


def main():
    p.require(os.environ["USER"] == "awallace43", "remote owner")
    mode, pin, *hashes = sys.argv[1:]
    p.require(p.git(n.HARNESS, "rev-parse", "HEAD") == pin, "nonc1 handoff pin")
    if mode == "preflight":
        preflight()
    elif mode == "stage":
        stage(pin)
    elif mode == "audit":
        audit(pin, *hashes)
    elif mode == "store-gates":
        store_gates(pin, *hashes)
    elif mode == "submit":
        h.submit(pin, *hashes, cohort=n, verifier=verify_bundle, cache=CACHE, failed_jobs=FAILED_JOBS)
    else:
        raise RuntimeError("unknown handoff mode")


if __name__ == "__main__":
    main()
