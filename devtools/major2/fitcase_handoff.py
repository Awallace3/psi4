"""Foreground-only staging/audit; actual submit is exclusively parent authorized."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import artifacts as a
import campaign
import commonfix as c
import fitcase as f
import handoff as h
import jkexception as j
import policy as p

CACHE = Path.home()/".cache/psi4-major2/fitcase-20261019"
FAILED_JOBS = h.FAILED_JOBS+["14165998"]


def preflight():
    p.require(not Path(f.RUN).exists() and not CACHE.exists(), "new fitcase namespace already exists")
    pair = json.load(sys.stdin)
    c.verify_shared(pair)  # receipt/toolchain hashes, NOT heavy imports or stage hashwalk
    shared = j.proof(j.SOURCES)
    test = f.test_proof(f.TEST_SOURCE)
    origin = p.read_json(Path(c.ORIGIN)/"logs/preflight.json")
    # Capture each old run once. Never touch original build trees.
    old_runs = origin["preservation"]["old_runs"]
    stats = {}
    for job, entry in old_runs.items():
        root = Path(entry["root"])
        p.require(a.hashed_tree(root) == entry["files"], "original failed root changed: "+job)
        stats[str(root)] = {name: stat(root/name) for name in entry["files"]}
    root = Path(c.ORIGIN)
    files = a.hashed_tree(root)
    p.require(len(files) < 1000, "unbounded origin run expansion")
    stats[str(root)] = {name: stat(root/name) for name in files}
    builds = dict(origin["preservation"]["build_sentinel_stats"])
    for item in pair["arms"].values():
        build = Path(item["build"])
        names = list(item["build_files"])+["stage/bin/psi4", "stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so"]
        builds[str(build)] = {name: stat(build/name) for name in names}
    rows = accounting()
    value = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 no_chemistry=True, no_device_calls=True, no_new_import=True,
                 origin_job="14165998", artifact_pair_digest=c.PAIR_DIGEST, shared_fix=shared,
                 test_source_proof=test, failed_root_stats=stats, build_sentinel_stats=builds,
                 origin_preflight_sha256=p.sha(Path(c.ORIGIN)/"logs/preflight.json"),
                 queue=h.queue(), accounting=rows)
    CACHE.mkdir()
    h.durable(CACHE/"preflight.json", value)
    print(json.dumps(dict(no_submission=True, cache=str(CACHE), preflight_sha256=p.sha(CACHE/"preflight.json"),
                          failed_roots=len(stats), build_roots=len(builds))))


def stat(path):
    st = Path(path).stat()
    return [st.st_size, st.st_mtime_ns, st.st_mode]


def accounting():
    rows = [line.split("|") for line in h.output("sacct", "-n", "-P", "-j", ",".join(FAILED_JOBS),
                    "--format=JobIDRaw,State,ExitCode,Elapsed").splitlines()]
    for job in FAILED_JOBS:
        for ident in (job, job+".batch"):
            p.require(any(row[0] == ident and row[1:3] == ["FAILED", "1:0"] for row in rows),
                      "old job accounting changed: "+ident)
    return rows


def preservation(pre):
    for key in ("failed_root_stats", "build_sentinel_stats"):
        for root, files in pre[key].items():
            if key == "failed_root_stats":
                p.require({str(x.relative_to(root)) for x in Path(root).rglob("*") if x.is_file()} == set(files),
                          "old failed root file set changed")
            for name, expected in files.items():
                p.require(stat(Path(root)/name) == expected, "old sentinel changed: "+root+"/"+name)
    # Stat checks are bounded preservation evidence, not claims of fresh binary hashes.


def verify_bundle(pin, manifest_sha, batch_sha, *, full_sources=False):
    root = Path(f.RUN)
    m = p.read_json(root/"manifest.json")
    p.validate_manifest(m)
    p.require(m["build_mode"] == f.MODE and m["harness"]["pin"] == pin, "fitcase harness pin")
    receipt = p.read_json(root/"bundle-receipt.json")
    p.require(p.sha(root/"manifest.json") == receipt["manifest_sha256"] == manifest_sha, "manifest hash")
    p.require(p.sha(root/"campaign.sbatch") == receipt["batch_sha256"] == batch_sha, "batch hash")
    p.require((root/"campaign.sbatch").read_text() ==
              campaign.batch_text(manifest_sha, commonfix=True), "batch bytes")
    for item in [*m["arms"].values(), m["test_source"], dict(source=f.HARNESS, **m["harness"])]:
        # Full remote source bytes checked ONCE at staging; later host audits
        # check Git identity/cleanliness and selected byte sentinels. Allocation
        # Engine.pins still checks ALL bytes, including every test/harness file.
        p.check_source(item["source"], item["pin"], item["files"] if full_sources else None)
        if not full_sources:
            selected = [name for name in item["files"] if name in p.OWNED_DIFF or name in
                        (j.JK, j.RANGE_TEST, "tests/pytests/conftest.py", "tests/pytests/addons.py") or
                        name.startswith("devtools/major2/")]
            for name in selected:
                p.require(p.sha(Path(item["source"])/name) == item["files"][name]["sha256"], "source sentinel")
    p.require(j.proof(j.SOURCES) == m["shared_fix"], "frozen commonfix proof")
    p.require(f.test_proof(f.TEST_SOURCE) == m["test_source"]["proof"], "corrected test proof")
    for name in ("STARTED.json", "build-baseline", "build-candidate", "raw"):
        p.require(not (root/name).exists(), "bundle already started")
    pre = p.read_json(root/"logs/preflight.json")
    p.require(pre == p.read_json(CACHE/"preflight.json") and pre["artifact_pair_digest"] == c.PAIR_DIGEST,
              "preflight identity")
    preservation(pre)
    return m, pre


def stage(pin):
    payload = json.load(sys.stdin)
    p.require(set(payload) == {"manifest.json", "campaign.sbatch", "bundle-receipt.json"}, "generated payload")
    root = Path(f.RUN)
    p.require(not root.exists(), "new fitcase run already exists")
    p.validate_manifest(json.loads(payload["manifest.json"]))
    root.mkdir()
    (root/"logs").mkdir()
    for name, text in payload.items():
        h.durable(root/name, text)
    h.durable(root/"logs/preflight.json", p.read_json(CACHE/"preflight.json"))
    receipt = json.loads(payload["bundle-receipt.json"])
    verify_bundle(pin, receipt["manifest_sha256"], receipt["batch_sha256"], full_sources=True)
    print(json.dumps(dict(staged=True, no_submission=True, **receipt)))


def store_gates(pin, manifest_sha, batch_sha):
    """Parent-only data transfer; exclusive, durable and tied to this exact bundle."""
    value = json.load(sys.stdin)
    verify_bundle(pin, manifest_sha, batch_sha)
    p.require(value["harness_pin"] == pin and value["manifest_sha256"] == manifest_sha and
              value["batch_sha256"] == batch_sha and value["artifact_pair_digest"] == c.PAIR_DIGEST and
              value["pristine_staging"] and value["old_run_unchanged"] and value["old_build_sentinels_unchanged"] and
              value["no_submission"] and "gpu-h200" in value["test_only_stdout"]+value["test_only_stderr"],
              "parent gates identity")
    h.durable(Path(f.RUN)/"logs/pre-submit-gates.json", value)
    print(json.dumps(dict(gates_stored=True, no_submission=True)))


def audit(pin, manifest_sha, batch_sha):
    p.require(not (Path(f.RUN)/"logs/submission.lock").exists(), "permanent submission lock exists")
    m, pre = verify_bundle(pin, manifest_sha, batch_sha)
    c.verify_shared(m["artifact_pair"])
    subprocess.run(["bash", "-n", str(Path(f.RUN)/"campaign.sbatch")], check=True, timeout=10)
    rows = h.queue()
    result = subprocess.run(["sbatch", "--test-only", str(Path(f.RUN)/"campaign.sbatch")],
                            capture_output=True, text=True, timeout=35)
    p.require(result.returncode == 0 and "gpu-h200" in result.stdout+result.stderr, "actual H200 test-only")
    print(json.dumps(dict(harness_pin=pin, manifest_sha256=manifest_sha, batch_sha256=batch_sha,
                         preflight_sha256=p.sha(CACHE/"preflight.json"), artifact_pair_digest=c.PAIR_DIGEST,
                         pristine_staging=True, old_run_unchanged=True, old_build_sentinels_unchanged=True,
                         queue=rows, test_only_stdout=result.stdout, test_only_stderr=result.stderr,
                         no_submission=True), indent=2))


def main():
    p.require(os.environ["USER"] == "awallace43", "remote owner")
    mode, pin, *hashes = sys.argv[1:]
    p.require(p.git(f.HARNESS, "rev-parse", "HEAD") == pin, "fitcase handoff harness pin")
    if mode == "preflight":
        preflight()
    elif mode == "stage":
        stage(pin)
    elif mode == "audit":
        audit(pin, *hashes)
    elif mode == "store-gates":
        store_gates(pin, *hashes)
    elif mode == "submit":
        h.submit(pin, *hashes, cohort=f, verifier=verify_bundle, cache=CACHE, failed_jobs=FAILED_JOBS)
    else:
        raise RuntimeError("unknown handoff mode")


if __name__ == "__main__":
    main()
