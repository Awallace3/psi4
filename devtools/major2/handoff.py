"""Foreground preparation and PARENT-ONLY one-shot handoff. Never auto-submit."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import artifacts as a
import campaign
import jkexception as j
import policy as p

OLD_RUN = Path(p.RUN)
CACHE = Path.home()/".cache/psi4-major2/jkexception-20261018"
FAILED_JOBS = ["14158602", "14159755", "14160001", "14160573", "14160948", "14161975", "14164895"]


def output(*args):
    return subprocess.check_output(list(map(str, args)), text=True, timeout=35).strip()


def durable(path, value):
    path = Path(path)
    with path.open("x") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def queue():
    rows = []
    for line in output("squeue", "-h", "-r", "-u", "awallace43", "-o", "%i|%q|%j|%P").splitlines():
        ident, qos, name, partition = line.split("|")
        fields = dict(re.findall(r"(\w+)=([^\s]+)", output("scontrol", "show", "job", "-o", ident)))
        p.require(fields["UserId"].startswith("awallace43("), "queue user")
        p.require(not any("major2" in fields.get(k, "").lower() for k in ["JobName", "Command", "WorkDir"]),
                  "active major2 job; do not duplicate")
        p.require("gpu-h200" not in fields.get("Partition", ""), "own H200 holder")
        rows.append(dict(id=ident, qos=qos, name=name, partition=partition))
    p.require(sum(r["qos"] == "embers" for r in rows) < 50, "embers headroom")
    return rows


def historical():
    """Reuse full captured binary hashes; compare only bounded immutable sentinels."""
    before_path = OLD_RUN/"logs/old-state.json"
    before = p.read_json(before_path)
    a.validate_pair(before["pair"])  # OLD exact artifact API, never repointed.
    old = dict(before["old_runs"])
    if "14164895" not in old:
        old["14164895"] = dict(root=str(OLD_RUN))
    runs = {}
    for job, item in old.items():
        root = Path(item["root"])
        files = a.hashed_tree(root)
        p.require(len(files) < 1000, "unexpected old root expansion")
        if job != "14164895":
            p.require(files == item["files"], "old failure bytes changed: "+job)
        failure = p.read_json(root/"ENGINE_FAILED.json")
        p.require(failure["production_records"] == 0, "unexpected old production")
        runs[job] = dict(root=str(root), files=files)
    # Existing immutable snapshot includes all 22k build file stats and full
    # stage hashes. Do not rehash binaries or walk enormous trees on login.
    sentinels = {}
    for name, stats in before["build_stats"].items():
        selected = {rel: expected for rel, expected in stats.items()
                    if rel in ["CMakeCache.txt", "cpks-commit.txt",
                               "psi4-core-prefix/src/psi4-core-build/CMakeCache.txt",
                               "psi4-core-prefix/src/psi4-core-build/build.ninja",
                               "psi4-core-prefix/src/psi4-core-build/compile_commands.json",
                               "stage/bin/psi4"] or
                    (rel.startswith("stage/lib/psi4/core") and rel.endswith(".so"))}
        p.require(len(selected) >= 6, "old build snapshot missing sentinels")
        for rel, expected in selected.items():
            st = (Path(name)/rel).stat()
            p.require([st.st_size, st.st_mtime_ns, st.st_mode] == expected, "old build sentinel stat changed")
        sentinels[name] = selected
    for arm, item in a.ALLOWLIST.items():
        p.check_source(item["source"], item["pin"])
    rows = [line.split("|") for line in output("sacct", "-n", "-P", "-j", ",".join(FAILED_JOBS),
                                            "--format=JobIDRaw,State,ExitCode,Elapsed").splitlines()]
    for job in FAILED_JOBS:
        for ident in (job, job+".batch"):
            p.require(any(row[0] == ident and row[1:3] == ["FAILED", "1:0"] for row in rows),
                      "unexpected old accounting: "+ident)
    return dict(captured_full_snapshot=str(before_path), captured_full_snapshot_sha256=p.sha(before_path),
                old_runs=runs, build_sentinel_stats=sentinels, old_accounting=rows,
                binary_preservation="captured full hashes + unchanged selected size/mtime/mode; no new binary hash claim")


def preflight():
    p.require(not Path(j.RUN).exists() and not CACHE.exists(), "fresh run/cache namespace already exists")
    for arm, source in j.SOURCES.items():
        p.check_source(source, j.PINS[arm])
        p.require(not (Path(source)/j.BUILD).exists(), "fresh build exists")
    common = j.proof(j.SOURCES)
    # No Psi4 import or device/chemistry execution. New binaries do not exist yet.
    a.verify_shared(p.read_json(OLD_RUN/"logs/old-state.json")["pair"])
    oldhost_path = OLD_RUN/"logs/host-preflight.json"
    host = p.read_json(oldhost_path)
    p.require(host["no_chemistry"] and host["no_device_calls"] and
              all(item["PASS"] and item["import_exit"] == 0 and item["installed_drivers"] == 143
                  for item in host["pass_by_arm"].values()), "historical host import evidence")
    tools = {}
    for compiler, frontend in [("conda-gcc", "cc1"), ("conda-g++", "cc1plus")]:
        binary = p.P+"/software/toolchains/"+compiler
        path = output("bash", str(campaign.HERE/"build.sh"), binary, "-print-prog-name="+frontend)
        p.require(Path(path).is_file() and os.access(path, os.X_OK), "selected compiler frontend")
        tools[compiler] = dict(frontend=path,
            version=output("bash", str(campaign.HERE/"build.sh"), binary, "--version"))
    tools["nvcc"] = output(p.CUDA+"/bin/nvcc", "--version")
    p.require("V13." in tools["nvcc"] and all("14." in tools[k]["version"] for k in ["conda-gcc", "conda-g++"]),
              "CUDA13/GCC14 toolchain")
    preserved = historical()
    rows = queue()
    CACHE.mkdir()
    receipt = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   no_chemistry=True, no_device_calls=True, no_new_import=True,
                   historical_import_receipt=str(oldhost_path), historical_import_sha256=p.sha(oldhost_path),
                   shared_fix=common, toolchain=tools, queue=rows, preservation=preserved)
    durable(CACHE/"preflight.json", receipt)
    print(json.dumps(dict(cache=str(CACHE), preflight_sha256=p.sha(CACHE/"preflight.json"),
                          old_jobs=FAILED_JOBS, no_submission=True)))


def verify_bundle(pin, manifest_sha, batch_sha, sources=True):
    root = Path(j.RUN)
    m = p.read_json(root/"manifest.json")
    p.validate_manifest(m)
    p.require(m["build_mode"] == j.MODE and m["harness"]["pin"] == pin, "fresh harness pin")
    receipt = p.read_json(root/"bundle-receipt.json")
    p.require(p.sha(root/"manifest.json") == receipt["manifest_sha256"] == manifest_sha, "manifest hash")
    p.require(p.sha(root/"campaign.sbatch") == receipt["batch_sha256"] == batch_sha, "batch hash")
    p.require((root/"campaign.sbatch").read_text() == campaign.batch_text(manifest_sha, fresh=True), "batch bytes")
    if sources:
        for item in [*m["arms"].values(), dict(source=j.HARNESS, **m["harness"])]:
            p.check_source(item["source"], item["pin"], item["files"])
        p.require(j.proof(j.SOURCES) == m["shared_fix"], "shared fix proof")
    for arm, item in m["arms"].items():
        p.require(not Path(item["build"]).exists() and not Path(item["build"]).is_symlink(), "new build root exists")
    for name in ["STARTED.json", "build-baseline", "build-candidate", "raw"]:
        p.require(not (root/name).exists(), "bundle already started")
    pre = p.read_json(root/"logs/preflight.json")
    p.require(pre == p.read_json(CACHE/"preflight.json"), "preflight bytes")
    p.require(p.sha(pre["historical_import_receipt"]) == pre["historical_import_sha256"], "historical evidence changed")
    return m, pre


def stage(pin):
    payload = json.load(sys.stdin)
    p.require(set(payload) == {"manifest.json", "campaign.sbatch", "bundle-receipt.json"}, "generated-data payload")
    root = Path(j.RUN)
    p.require(not root.exists(), "new namespace exists")
    m = json.loads(payload["manifest.json"])
    p.validate_manifest(m)
    p.require(m["harness"]["pin"] == pin, "staging harness pin")
    root.mkdir()
    (root/"logs").mkdir()
    for name, text in payload.items():
        durable(root/name, text)
    durable(root/"logs/preflight.json", p.read_json(CACHE/"preflight.json"))
    receipt = json.loads(payload["bundle-receipt.json"])
    verify_bundle(pin, receipt["manifest_sha256"], receipt["batch_sha256"])
    print(json.dumps(dict(staged=True, no_submission=True, **receipt)))


def audit(pin, manifest_sha, batch_sha):
    p.require(not (Path(j.RUN)/"logs/submission.lock").exists(), "permanent submission lock already exists")
    m, pre = verify_bundle(pin, manifest_sha, batch_sha)
    preserved = historical()
    for key in ["captured_full_snapshot_sha256", "old_runs", "build_sentinel_stats"]:
        p.require(preserved[key] == pre["preservation"][key], "preservation changed: "+key)
    # Small coherent runtime/config files are current; full same-allocation
    # runtime/toolchain/source checks in Engine are still mandatory.
    p.check_runtime_aliases()
    p.einsums_coherence()
    subprocess.run(["bash", "-n", str(Path(j.RUN)/"campaign.sbatch")], check=True, timeout=10)
    rows = queue()
    test = subprocess.run(["sbatch", "--test-only", str(Path(j.RUN)/"campaign.sbatch")],
                          capture_output=True, text=True, timeout=35)
    p.require(test.returncode == 0 and "gpu-h200" in test.stderr+test.stdout, "H200 test-only route")
    print(json.dumps(dict(harness_pin=pin, manifest_sha256=manifest_sha, batch_sha256=batch_sha,
                         preflight_sha256=p.sha(CACHE/"preflight.json"), shared_fix=m["shared_fix"],
                         pristine_staging=True, old_run_unchanged=True, old_build_sentinels_unchanged=True,
                         queue=rows, test_only_stdout=test.stdout, test_only_stderr=test.stderr,
                         no_submission=True), indent=2))


def submit(pin, manifest_sha, batch_sha):
    """Called ONLY by reviewed parent-finalize. A failed lock is never reusable."""
    root = Path(j.RUN)
    lock = root/"logs/submission.lock"
    lock.mkdir()
    m, pre = verify_bundle(pin, manifest_sha, batch_sha)
    gates = p.read_json(root/"logs/pre-submit-gates.json")
    p.require(gates["harness_pin"] == pin and gates["manifest_sha256"] == manifest_sha and
              gates["batch_sha256"] == batch_sha and gates["preflight_sha256"] == p.sha(CACHE/"preflight.json") and
              gates["pristine_staging"] and gates["old_run_unchanged"] and gates["old_build_sentinels_unchanged"],
              "parent audit gates")
    p.require("gpu-h200" in gates["test_only_stdout"]+gates["test_only_stderr"], "test-only missing")
    durable(lock/"controller-before.json", queue())
    argv = ["sbatch", "--parsable", str(root/"campaign.sbatch")]
    durable(lock/"attempt.json", dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            argv=argv, no_retry=True, replacement_for="14164895", preserved_failed_jobs=FAILED_JOBS,
            compiled_in_new_job=True, build_mode=j.MODE, manifest_sha256=manifest_sha))
    try:
        result = subprocess.run(argv, cwd=root, text=True, capture_output=True, timeout=35)
    except BaseException as exc:
        durable(lock/"AMBIGUOUS.json", dict(error=str(exc), no_retry=True))
        raise
    # Persist the ID before *any* acceptance assertions or controller queries.
    job = result.stdout.strip().split(";")[0]
    if re.fullmatch(r"\d+", job):
        durable(lock/"job-id.txt", job+"\n")
    durable(lock/"sbatch.stdout", result.stdout)
    durable(lock/"sbatch.stderr", result.stderr)
    durable(lock/"sbatch-exit.txt", str(result.returncode)+"\n")
    p.require(result.returncode == 0 and re.fullmatch(r"\d+", job), "never retry; reconcile durable raw receipt")
    info = output("scontrol", "show", "job", "-o", job)
    durable(lock/"acceptance.scontrol.txt", info+"\n")
    f = dict(re.findall(r"(\w+)=([^\s]+)", info))
    expected = dict(JobId=job, JobName="major2-gpu-ab", Account="gts-cs207-chemx", QOS="embers",
                    Features="H200", NumCPUs="8", NumTasks="1", MinMemoryNode="128G",
                    TimeLimit="04:00:00", Requeue="0", Restarts="0", Partition="gpu-h200",
                    TresPerNode="gres/gpu:h200:1", Command=str(root/"campaign.sbatch"), WorkDir=str(root),
                    StdOut=str(root/"logs"/("major2-gpu-ab-"+job+".out")),
                    StdErr=str(root/"logs"/("major2-gpu-ab-"+job+".err")))
    for key, value in expected.items():
        p.require(f.get(key) == value, "accepted route mismatch: "+key)
    p.require(f["NumNodes"] in ["1", "1-1"], "one node")
    for token in ["cpu=8", "mem=128G", "node=1", "gres/gpu=1", "gres/gpu:h200=1"]:
        p.require(token in f["ReqTRES"].split(","), "accepted resource mismatch")
    accepted = dict(job=job, controller=f, manifest_sha256=manifest_sha, submissions=1)
    durable(lock/"ACCEPTED.json", accepted)
    print(json.dumps(accepted, indent=2))


def main():
    p.require(os.environ["USER"] == "awallace43", "remote owner")
    mode, pin, *hashes = sys.argv[1:]
    p.require(p.git(j.HARNESS, "rev-parse", "HEAD") == pin, "handoff harness pin")
    if mode == "preflight":
        preflight()
    elif mode == "stage":
        stage(pin)
    elif mode == "audit":
        audit(pin, *hashes)
    elif mode == "submit":
        submit(pin, *hashes)
    else:
        raise RuntimeError("unknown handoff mode")


if __name__ == "__main__":
    main()
