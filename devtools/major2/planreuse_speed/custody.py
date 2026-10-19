"""Bounded login preparation and parent-only one-shot submission. No device API."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import config as s

legacy = s.native_module("handoff")
FAILED = legacy.FAILED
COMPLETED = legacy.COMPLETED+["14178506"]
preserve = legacy.preserve
queue = legacy.queue


def accounting():
    rows = [r.split("|") for r in s.n.output("sacct", "-n", "-P", "-j",
        ",".join(FAILED+COMPLETED), "--format=JobIDRaw,State,ExitCode").splitlines()]
    for jobs, state, code in ((FAILED, "FAILED", "1:0"), (COMPLETED, "COMPLETED", "0:0")):
        for job in jobs:
            for ident in (job, job+".batch"):
                s.require(any(r[:3] == [ident, state, code] for r in rows),
                          "historical accounting: "+ident)
    return rows


def collect():
    s.n.check_env(os.environ)
    s.require(not Path(s.RUN).exists(), "new run already exists")
    native = s.load(Path(s.NATIVE_RUN)/"manifest.json")
    s.require(native["pin"] == s.PARENT, "completed native source identity")
    pres = native["preservation"]
    preserve(pres)
    s.require(len(pres["roots"]) == 15, "fifteen inherited actual roots")
    native_root = Path(s.NATIVE_RUN)
    s.require(s.load(native_root/"STARTED.json") == dict(job="14178506", pin=s.PARENT) and
              s.load(native_root/"ENDPOINT-COMPLETE.json") == dict(
                  processes=3, outputs=18, pairs=153, accepted_speedup_data=False,
                  compiled_diagnostic_only=True) and
              s.load(native_root/"batch-exit.json") == dict(exit=0, signalled=0) and
              (native_root/"COMPLETED").is_file() and not (native_root/"FAILED").exists(),
              "completed native job, not failed/unsubmitted staging")
    pres["roots"][s.NATIVE_RUN] = legacy.snapshot(native_root)
    for name in ("manifest.json", "campaign.sbatch", "STARTED.json", "ENDPOINT-COMPLETE.json",
                 "COMPLETED", "batch-exit.json", "logs/job-id.txt"):
        pres["small_hashes"][str(native_root/name)] = legacy.small(native_root/name)
    # A metadata-only preliminary bundle may be archived, never overwritten.
    # This fixed owned archive is NOT a jobroot and can never count as a result.
    archive = Path(s.RUN+"-preparation-v1")
    if archive.exists():
        staged = legacy.unsubmitted_stage(archive)
        receipt = s.load(archive/"bundle-receipt.json")
        s.require(receipt["pin"] == "875e22bfb2d15ca03ed159da6d9b58843fb49dae" and
                  receipt["no_submission"] is True and
                  receipt["manifest_sha256"] == legacy.small(archive/"manifest.json") and
                  receipt["batch_sha256"] == legacy.small(archive/"campaign.sbatch"),
                  "exact unsubmitted preliminary speed preparation archive")
        pres["unsubmitted_staging"][str(archive)] = staged
        for rel in staged:
            pres["small_hashes"][str(archive/rel)] = legacy.small(archive/rel)
    s.require(len(pres["roots"]) == 16, "sixteen old roots: thirteen failed, three completed")
    preserve(pres)
    old = s.load(Path(s.n.OLD)/"manifest.json")  # policy metadata ONLY, no results/timers
    import repeatability as r
    r.validate_pair(old["diagnostic_pair"])
    return dict(preservation=pres, historical_accounting=accounting(), queue=queue(),
        diagnostic_pair=old["diagnostic_pair"], arms=old["arms"],
        suite=old["suite"], suite_sha256=old["suite_sha256"],
        identity_helper=old["identity_helper"], numerical_overrides={},
        compiler=native["compiler"], native_libraries=native["native_libraries"],
        sdk_headers=native["headers"],
        metadata_only=True, no_submission=True)


def batch_text(mh):
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=major2-planreuse-speed
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --constraint=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@300
#SBATCH --no-requeue
#SBATCH --chdir={s.RUN}
#SBATCH --output={s.RUN}/logs/%x-%j.out
#SBATCH --error={s.RUN}/logs/%x-%j.err
set -euo pipefail
exec bash {s.SOURCE}/devtools/major2/planreuse_speed/supervise.sh {s.RUN} \\
  {s.E}/bin/python -B {s.SOURCE}/devtools/major2/planreuse_speed/controller.py \\
  {s.RUN}/manifest.json {mh}
"""


def freeze(collection, directory):
    import fixed_jk as f
    import policy as p
    root = s.HERE.parents[2]
    pin = p.git(root, "rev-parse", "HEAD")
    names = p.git(root, "diff", "--name-status", s.PARENT, pin).splitlines()
    s.require(names and all(v.startswith("A\tdevtools/major2/planreuse_speed/") for v in names),
              "owned additions only")
    helpers = {v.split("\t")[1]:s.sha(root/v.split("\t")[1]) for v in names}
    for name in ("supervise.sh", "parent-finalize.sh", "run-tests.sh", "runner.py"):
        s.require("devtools/major2/planreuse_speed/"+name in helpers, "required tracked helper: "+name)
    files = p.file_manifest(root)
    m = dict(collection, source=s.SOURCE, run_root=s.RUN, parent=s.PARENT, pin=pin,
        helpers=helpers, policy=s.POLICY, frozen_factor=f.FROZEN_FACTOR,
        settings=dict(f.SETTINGS, computes_per_process=11), sequence=s.SEQUENCE,
        route=p.ROUTE, gates=p.GATES, scratch_parent=s.SCRATCH,
        harness=dict(path=s.SOURCE, pin=pin, files=files, files_digest=p.digest(files)),
        build_budget_seconds=[0, 0], accepted_speedup_data=False)
    s.source_check(m, root)
    out = Path(directory)
    out.mkdir()
    s.durable(out/"manifest.json", m)
    mh = s.sha(out/"manifest.json")
    s.manifest(out/"manifest.json", mh)
    s.durable(out/"campaign.sbatch", batch_text(mh))
    s.durable(out/"bundle-receipt.json", dict(pin=pin, manifest_sha256=mh,
        batch_sha256=s.sha(out/"campaign.sbatch"), no_submission=True))
    print(json.dumps(s.load(out/"bundle-receipt.json"), sort_keys=True))


def verify(pin, mh, bh, pristine=True):
    root = Path(s.RUN)
    m = s.manifest(root/"manifest.json", mh)
    s.require(m["pin"] == pin and s.sha(root/"campaign.sbatch") == bh and
              (root/"campaign.sbatch").read_text() == batch_text(mh) and
              s.load(root/"bundle-receipt.json") == dict(pin=pin, manifest_sha256=mh,
                  batch_sha256=bh, no_submission=True), "exact ready bundle")
    s.source_check(m)
    for item in m["arms"].values():
        s.require(s.n.output("git", "-C", item["source"], "rev-parse", "HEAD") == item["pin"] and
                  not s.n.output("git", "-C", item["source"], "status", "--porcelain",
                                 "--untracked-files=no"), "scientific source Git identity")
    if pristine:
        s.require(set(p.name for p in root.iterdir()) ==
                  {"manifest.json", "campaign.sbatch", "bundle-receipt.json", "logs"} and
                  not list((root/"logs").iterdir()), "pristine ready root; never resume")
    for file in (root/"manifest.json", root/"campaign.sbatch", root/"bundle-receipt.json"):
        s.require(not file.is_symlink() and not file.stat().st_mode & 0o222, "readonly ready data")
    preserve(m["preservation"])
    return m


def stage(pin):
    payload = json.load(sys.stdin)
    s.require(set(payload) == {"manifest.json", "campaign.sbatch", "bundle-receipt.json"},
              "generated data only, not tracked code transport")
    root = Path(s.RUN)
    root.mkdir()
    (root/"logs").mkdir()
    for name, text in payload.items():
        s.durable(root/name, text)
    receipt = s.load(root/"bundle-receipt.json")
    verify(pin, receipt["manifest_sha256"], receipt["batch_sha256"])
    print(json.dumps(dict(receipt, staged=True), sort_keys=True))


def audit(pin, mh, bh):
    s.n.check_env(os.environ)
    verify(pin, mh, bh)
    rows, acct = queue(), accounting()
    result = subprocess.run(["sbatch", "--test-only", s.RUN+"/campaign.sbatch"],
                            text=True, capture_output=True, timeout=35)
    s.require(result.returncode == 0 and "gpu-h200" in result.stdout+result.stderr,
              "exact H200 test-only route failed")
    return dict(pin=pin, manifest_sha256=mh, batch_sha256=bh,
        old_roots=16, failed_roots=13, completed_roots=3, queue=rows, accounting=acct,
        no_submission=True, test_only_stdout=result.stdout, test_only_stderr=result.stderr)


def permanent_lock(path):
    Path(path).mkdir()
    s.n.fsync_dir(Path(path).parent)


accept_id = legacy.accept_id


def parent_submit(pin, mh, bh):
    s.require(os.environ.get("MAJOR2_SPEED_PARENT_REVIEW") == mh, "parent review approval required")
    gates = audit(pin, mh, bh)
    permanent_lock(Path(s.RUN)/"logs/parent-attempt")
    s.durable(Path(s.RUN)/"logs/pre-submit-gates.json", gates)
    # Permanent attempt means even timeout, lost SSH or rejected response NEVER retries.
    result = subprocess.run(["sbatch", "--parsable", s.RUN+"/campaign.sbatch"],
                            capture_output=True, text=True, timeout=35)
    s.durable(Path(s.RUN)/"logs/sbatch-response.json", dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    s.require(result.returncode == 0, "uncertain/rejected submission: STOP, never retry")
    job = accept_id(result.stdout)
    s.durable(Path(s.RUN)/"logs/job-id.txt", job+"\n")  # fsynced BEFORE querying
    controller = s.n.output("scontrol", "show", "job", "-o", job)
    s.durable(Path(s.RUN)/"logs/accepted-controller.txt", controller+"\n")
    fields = dict(re.findall(r"(\w+)=([^\s]+)", controller))
    s.require(fields.get("JobId") == job and fields.get("QOS") == "embers" and
              fields.get("Account") == "gts-cs207-chemx" and
              "gpu-h200" in fields.get("Partition", "") and
              fields.get("Command") == s.RUN+"/campaign.sbatch", "accepted controller route")
    print(json.dumps(dict(job=job, no_retry=True)))


def parent_local(bundle, phx):
    bundle = Path(bundle)
    receipt = s.load(bundle/"bundle-receipt.json")
    pin, mh, bh = (receipt[k] for k in ("pin", "manifest_sha256", "batch_sha256"))
    s.require(os.environ.get("MAJOR2_SPEED_PARENT_REVIEW") == mh, "explicit parent local review")
    m = s.manifest(bundle/"manifest.json", mh)
    s.require(s.sha(bundle/"campaign.sbatch") == bh and
              (bundle/"campaign.sbatch").read_text() == batch_text(mh) and m["pin"] == pin,
              "local frozen bundle")
    s.source_check(m, s.HERE.parents[2])
    permanent_lock(bundle.parent/"parent-finalizer.attempt")
    command = (f"MAJOR2_SPEED_PARENT_REVIEW={mh} {s.E}/bin/python -B "
        f"{s.SOURCE}/devtools/major2/planreuse_speed/custody.py parent-submit {pin} {mh} {bh}")
    try:
        result = subprocess.run([phx, command], env=dict(os.environ, PHX_TIMEOUT="180"),
                                capture_output=True, text=True, timeout=190)
    except BaseException:
        s.durable(bundle.parent/"local-submit-ambiguous.json", dict(ambiguous=True, retry=False))
        raise
    s.durable(bundle.parent/"local-submit-response.json", dict(returncode=result.returncode,
              stdout=result.stdout, stderr=result.stderr))
    s.require(result.returncode == 0, "remote acceptance unknown/rejected: STOP, never retry")
    job = accept_id(json.loads(result.stdout)["job"])
    s.durable(bundle.parent/"job-id.txt", job+"\n")
    print(json.dumps(dict(job=job, no_retry=True)))


def main():
    s.n.check_env(os.environ)
    mode, *args = sys.argv[1:]
    if mode == "collect":
        print(json.dumps(collect(), sort_keys=True))
    elif mode == "freeze":
        freeze(s.load(args[0]), args[1])
    elif mode == "stage":
        stage(*args)
    elif mode == "audit":
        print(json.dumps(audit(*args), sort_keys=True))
    elif mode == "parent-submit":
        parent_submit(*args)
    elif mode == "parent-local":
        parent_local(*args)
    else:
        raise RuntimeError("unknown custody mode")


if __name__ == "__main__":
    main()
