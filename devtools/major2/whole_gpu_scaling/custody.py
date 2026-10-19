"""Stdlib login metadata/staging only; separately reviewed parent-only submit path."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import config as s

sys.modules.setdefault("common", s.n)
legacy = s.inherited("native_plan", "handoff")
FAILED = legacy.FAILED+["14182594"]
COMPLETED = legacy.COMPLETED+["14178506", "14179309"]
accept_id = legacy.accept_id
from admission import queue


# Independent saved first-case retrieval pins, not newly accepted science.
FAILED_ARTIFACT_HASHES = {
    "WHOLE-FAILED.json": "2dec06ae8c05830b6164d5141a3611a86c01a1c1da5d48156ab9fe853339cf6d",
    "FAILED": "0630b6557bb49b2952604d10aa49db0d7439afd726b940439f884d95e4126c2c",
    "batch-exit.json": "6fd4e5dff04355eccecfab0c3cf771a49c83977942fd64716c9181db4a31632c",
    "STARTED.json": "ce1cf84c29b0dfd967467446243a0e4dd20c1f49abac87467e9fc8ac42b34e62",
    "raw/scheduler.txt": "fac1597ed2fac549e510b73ab2a09b8d51ef825eefbb234c7394eeba4d6f83aa",
    "raw/pair-VERIFIED.json": "4b7aba2bda33b526638a426bd74df6693f510f18341fcfa4b873e2f6f2d2bf23",
    "raw/gpu-identity.json": "40e8f9c57f6d76300e28e57123c38fc209c54a0da3e3c3305cbd05f44ebc4e61",
    "raw/whole-baseline-peptide-r1.COMPLETE.json": "9246b162107cf0a7ff9685d6de367064b027a03c621336abe4f1fe934a5be97e",
    "raw/whole-baseline-peptide-r1.ENDPOINT-PRE-EXIT.json": "d6338aa3bdcb71094736c14948acee3172d324469465499e593208f3f90fd894",
    "raw/whole-baseline-peptide-r1.exit.json": "8bca810dc60387de807cce5c01160c88ac57d54d1e848d73b85c086806833cb1",
    "raw/whole-baseline-peptide-r1.log": "7a460b1f1165d926a09f16114153950682f6bc5b8155624df557cc08c71679b3",
    "logs/major2-whole-gpu-scaling-14182594.out": "f549e0f426d9815f4302e52fc11bc5e4313013914340a95230d5d2b55a20fe2f",
    "logs/major2-whole-gpu-scaling-14182594.err": "6c4a0d12e5a1a9c2373368b666321e301e4b2464ce6ca9551a044a71a6ce3001",
    "raw/sessions/whole-baseline-peptide-r1/result.json": "674aabc204343050f53e8db14498d1ac784f2868e25c416d38cc89a6e351736e",
    "raw/sessions/whole-baseline-peptide-r1/psi4.out": "f5a09c362b09a0c0878afc31eb0bf5e807d4fbd1777e41ebdbb99b7df268e6c9",
    "raw/sessions/whole-baseline-peptide-r1/timer.dat": "1d2364d21ea4c8282fcf46099092e3037b482abafb21c3551a7a738cf3f17313",
    "raw/sessions/whole-baseline-peptide-r1/profile.txt": "1111c0e9c572c4e15868e792f4cf877b68cb5c627f0e69dba803908d30c5b8d3",
}


def preserve(value):
    legacy.preserve(value)
    s.require(s.p.git(s.FAILED_SOURCE, "rev-parse", "HEAD") == s.FAILED_PIN and
              not s.p.git(s.FAILED_SOURCE, "status", "--porcelain"),
              "immutable failed whole source pin/clean tree")


def preserve_failed_whole(pres):
    """Add the actual FAILED allocation, including its completed first child."""
    root = Path(s.FAILED_RUN)
    stats = legacy.snapshot(root)
    s.require(sum(v[0] for v in stats.values()) <= 16*1024**2,
              "bounded failed whole metadata/output tree")
    hashes = {rel:legacy.small(root/rel) for rel in stats}
    s.require(all(hashes.get(rel) == digest for rel,digest in FAILED_ARTIFACT_HASHES.items()),
              "failed whole saved first-case artifact pins")
    mh = "a9318198cc73a82588ba46523f7dbecc06362d1838719be4b25461b913a4d7f5"
    bh = "3a9dec1242dc3f61fb9e0cccdd738ed364b9939a1bd60c06ac7da58f4ce1b46f"
    old = s.load(root/"manifest.json")
    s.require(old["pin"] == s.FAILED_PIN and old["source"] == s.FAILED_SOURCE and
              old["run_root"] == s.FAILED_RUN and hashes["manifest.json"] == mh and
              hashes["campaign.sbatch"] == bh and
              s.load(root/"bundle-receipt.json") == dict(pin=s.FAILED_PIN,
                  manifest_sha256=mh, batch_sha256=bh, no_submission=True),
              "failed whole original bundle/source identity")
    s.require(s.load(root/"STARTED.json") == dict(job="14182594", pin=s.FAILED_PIN,
                  manifest_sha256=mh) and
              s.load(root/"batch-exit.json") == dict(exit=1, signalled=0) and
              s.load(root/"WHOLE-FAILED.json") == dict(
                  error="missing backend/options print: Spherical Harmonics?: true",
                  error_type="RuntimeError", individually_valid_records=0, signalled=False) and
              (root/"FAILED").is_file() and not (root/"COMPLETED").exists() and
              not (root/"WHOLE-COMPLETE.json").exists() and
              (root/"logs/job-id.txt").read_text() == "14182594\n" and
              (root/"logs/parent-attempt").is_dir() and
              s.load(root/"logs/sbatch-response.json") ==
                  dict(returncode=0, stdout="14182594\n", stderr=""),
              "actual failed whole receipts/lock retained; not successful")
    label = "whole-baseline-peptide-r1"
    complete = s.load(root/("raw/"+label+".COMPLETE.json"))
    s.require(complete["job"] == "14182594" and complete["label"] == label and
              complete["child_exit"] == s.load(root/("raw/"+label+".exit.json")) and
              complete["child_exit"]["exit"] == 0 and
              complete["child_exit"]["interrupted"] is False and
              complete["evidence_kind"] == "parent post-exit completed output custody" and
              complete["output_hashes"] == {name:hashes["raw/sessions/"+label+"/"+name]
                  for name in ("result.json","psi4.out","timer.dat","profile.txt")},
              "first child completed custody only, parent failed")
    pres["roots"][s.FAILED_RUN] = stats
    pres["small_hashes"].update({str(root/rel):digest for rel,digest in hashes.items()})
    s.require(len(pres["roots"]) == 18 and len(pres["unsubmitted_staging"]) == 3,
              "18 actual roots and three unsubmitted stages")
    preserve(pres)


def accounting():
    rows = [v.split("|") for v in s.n.output("sacct", "-n", "-P", "-j",
        ",".join(FAILED+COMPLETED), "--format=JobIDRaw,State,ExitCode").splitlines()]
    for jobs,state,code in ((FAILED,"FAILED","1:0"), (COMPLETED,"COMPLETED","0:0")):
        for job in jobs:
            for ident in (job, job+".batch"):
                s.require(any(v[:3] == [ident,state,code] for v in rows), "old accounting: "+ident)
    return rows


def collect():
    s.n.check_env(os.environ)
    s.require(not Path(s.RUN).exists(), "new run exists; do not overwrite")
    root = Path(s.SPEED_RUN)
    m = s.load(root/"manifest.json")  # Policy metadata, never historical timings/matrices.
    s.require(m["pin"] == s.PARENT, "latest completed speed source")
    pres = m["preservation"]
    preserve(pres)
    s.require(len(pres["roots"]) == 16, "sixteen inherited roots")
    s.require(s.load(root/"STARTED.json") == dict(job="14179309", pin=s.PARENT) and
              s.load(root/"SPEED-COMPLETE.json") == dict(processes=6, computes=66,
                stored_arrays=24, matched_actual_D=True, plan_reuse_benchmark_only=True) and
              s.load(root/"batch-exit.json") == dict(exit=0, signalled=0) and
              (root/"COMPLETED").is_file() and not (root/"FAILED").exists() and
              (root/"logs/job-id.txt").read_text() == "14179309\n" and
              (root/"logs/parent-attempt").is_dir(), "completed speed receipt identity")
    pres["roots"][s.SPEED_RUN] = legacy.snapshot(root)
    for name in ("manifest.json","campaign.sbatch","STARTED.json","SPEED-COMPLETE.json",
                 "COMPLETED","batch-exit.json","logs/job-id.txt"):
        pres["small_hashes"][str(root/name)] = legacy.small(root/name)
    s.require(len(pres["roots"]) == 17 and len(FAILED) == 14 and len(COMPLETED) == 4,
              "17 inherited actual roots before adding failed14182594")
    # Rejected whole v1 is unsubmitted staging, not an actual jobroot.
    rejected = Path(s.REJECTED_RUN)
    staged = legacy.unsubmitted_stage(rejected)
    previous = s.load(rejected/"manifest.json")
    receipt = s.load(rejected/"bundle-receipt.json")
    s.require(previous["pin"] == receipt["pin"] ==
              "82d95144bfb5e7a00b0fdd9a2841b43e02836c19" and
              previous["run_root"] == s.REJECTED_RUN and receipt["no_submission"] is True and
              receipt["manifest_sha256"] == legacy.small(rejected/"manifest.json") and
              receipt["batch_sha256"] == legacy.small(rejected/"campaign.sbatch"),
              "rejected unsubmitted whole bundle identity")
    pres["unsubmitted_staging"][s.REJECTED_RUN] = staged
    for rel in staged:
        pres["small_hashes"][str(rejected/rel)] = legacy.small(rejected/rel)
    preserve(pres)
    preserve_failed_whole(pres)
    import repeatability as r
    r.validate_pair(m["diagnostic_pair"])
    return dict(preservation=pres, historical_accounting=accounting(), queue=queue(),
        diagnostic_pair=m["diagnostic_pair"], arms=m["arms"], suite=m["suite"],
        suite_sha256=m["suite_sha256"], identity_helper=m["identity_helper"],
        compiler=m["compiler"], native_libraries=m["native_libraries"],
        sdk_headers=m["sdk_headers"], metadata_only=True, no_submission=True)


def batch_text(mh):
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=major2-whole-gpu-scaling
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --constraint=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@300
#SBATCH --no-requeue
#SBATCH --chdir={s.RUN}
#SBATCH --output={s.RUN}/logs/%x-%j.out
#SBATCH --error={s.RUN}/logs/%x-%j.err
set -euo pipefail
exec bash {s.SOURCE}/devtools/major2/whole_gpu_scaling/supervise.sh {s.RUN} \\
  {s.E}/bin/python -B {s.SOURCE}/devtools/major2/whole_gpu_scaling/controller.py \\
  {s.RUN}/manifest.json {mh}
"""


def freeze(collection, directory):
    from provenance import verify_runner
    p = s.p
    pin = p.git(s.ROOT, "rev-parse", "HEAD")
    changes = p.git(s.ROOT, "diff", "--name-status", s.PARENT, pin).splitlines()
    s.require(changes and all(v.startswith("A\tdevtools/major2/whole_gpu_scaling/")
                             for v in changes), "only owned additions")
    helpers = {v.split("\t")[1]:s.sha(s.ROOT/v.split("\t")[1]) for v in changes}
    for name in ("runner.py", "execute.py", "controller.py", "validation.py", "supervise.sh", "parent-finalize.sh", "run-tests.sh"):
        s.require("devtools/major2/whole_gpu_scaling/"+name in helpers, "required owned helper")
    files = p.file_manifest(s.ROOT)
    m = dict(collection, source=s.SOURCE, run_root=s.RUN, parent=s.PARENT, pin=pin,
        helpers=helpers, policy=s.POLICY, route=s.ROUTE, gates=p.GATES,
        systems=p.SYSTEMS, sequence=s.SEQUENCE, scratch_parent=s.SCRATCH,
        harness=dict(path=s.SOURCE, pin=pin, files=files, files_digest=p.digest(files)),
        runner_provenance=verify_runner(s.ROOT), build_budget_seconds=[0,0])
    s.source_check(m, s.ROOT)
    out = Path(directory)
    out.mkdir()
    s.durable(out/"manifest.json", m)
    mh = s.sha(out/"manifest.json")
    s.manifest(out/"manifest.json", mh)
    s.durable(out/"campaign.sbatch", batch_text(mh))
    s.durable(out/"bundle-receipt.json", dict(pin=pin, manifest_sha256=mh,
        batch_sha256=s.sha(out/"campaign.sbatch"), no_submission=True))
    print(json.dumps(s.load(out/"bundle-receipt.json"), sort_keys=True))


def verify(pin, mh, bh):
    root = Path(s.RUN)
    m = s.manifest(root/"manifest.json", mh)
    s.require(m["pin"] == pin and s.sha(root/"campaign.sbatch") == bh and
              (root/"campaign.sbatch").read_text() == batch_text(mh) and
              s.load(root/"bundle-receipt.json") == dict(pin=pin, manifest_sha256=mh,
                  batch_sha256=bh, no_submission=True), "ready bundle identity")
    s.source_check(m)
    for v in m["arms"].values():
        s.require(s.p.git(v["source"], "rev-parse", "HEAD") == v["pin"] and
                  not s.p.git(v["source"], "status", "--porcelain", "--untracked-files=no"),
                  "scientific Git identity")
    s.require(set(v.name for v in root.iterdir()) ==
              {"manifest.json","campaign.sbatch","bundle-receipt.json","logs"} and
              not list((root/"logs").iterdir()), "pristine new run; no retries/resume")
    for name in ("manifest.json","campaign.sbatch","bundle-receipt.json"):
        f = root/name
        s.require(not f.is_symlink() and not f.stat().st_mode & 0o222, "readonly bundle")
    preserve(m["preservation"])
    return m


def stage(pin):
    payload = json.load(sys.stdin)
    s.require(set(payload) == {"manifest.json","campaign.sbatch","bundle-receipt.json"},
              "generated bundle data only; tracked code by Git")
    root = Path(s.RUN)
    root.mkdir()
    (root/"logs").mkdir()
    for name,text in payload.items():
        s.durable(root/name, text)
    receipt = s.load(root/"bundle-receipt.json")
    verify(pin, receipt["manifest_sha256"], receipt["batch_sha256"])
    print(json.dumps(dict(receipt, staged=True), sort_keys=True))


def audit(pin, mh, bh):
    s.n.check_env(os.environ)
    verify(pin, mh, bh)
    rows, acct = queue(), accounting()
    result = subprocess.run(["sbatch","--test-only",s.RUN+"/campaign.sbatch"],
                            text=True, capture_output=True, timeout=35)
    s.require(result.returncode == 0 and "gpu-h200" in result.stdout+result.stderr,
              "exact H200 test-only route")
    return dict(pin=pin, manifest_sha256=mh, batch_sha256=bh, old_roots=18,
        failed_roots=14, completed_roots=4, queue=rows, accounting=acct,
        no_submission=True, test_only_stdout=result.stdout, test_only_stderr=result.stderr)


def permanent_lock(path):
    Path(path).mkdir()
    s.n.fsync_dir(Path(path).parent)


def parent_submit(pin, mh, bh):
    s.require(os.environ.get("MAJOR2_WHOLE_PARENT_REVIEW") == mh, "parent review required")
    gates = audit(pin, mh, bh)
    permanent_lock(Path(s.RUN)/"logs/parent-attempt")
    s.durable(Path(s.RUN)/"logs/pre-submit-gates.json", gates)
    # Permanent attempt survives timeout/lost SSH/rejection; no retry ambiguity.
    result = subprocess.run(["sbatch","--parsable",s.RUN+"/campaign.sbatch"],
                            capture_output=True, text=True, timeout=35)
    s.durable(Path(s.RUN)/"logs/sbatch-response.json", dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    s.require(result.returncode == 0, "uncertain/rejected submission: STOP, never retry")
    job = accept_id(result.stdout)
    s.durable(Path(s.RUN)/"logs/job-id.txt", job+"\n")  # fsynced before controller query
    controller = s.n.output("scontrol","show","job","-o",job)
    s.durable(Path(s.RUN)/"logs/accepted-controller.txt", controller+"\n")
    fields = dict(re.findall(r"(\w+)=([^\s]+)", controller))
    s.require(fields.get("JobId") == job and fields.get("QOS") == "embers" and
              fields.get("Account") == "gts-cs207-chemx" and
              fields.get("TimeLimit") == "06:00:00" and
              "gpu-h200" in fields.get("Partition","") and
              fields.get("Command") == s.RUN+"/campaign.sbatch", "accepted route")
    print(json.dumps(dict(job=job, no_retry=True)))


def parent_local(bundle, phx):
    bundle = Path(bundle)
    receipt = s.load(bundle/"bundle-receipt.json")
    pin,mh,bh = (receipt[k] for k in ("pin","manifest_sha256","batch_sha256"))
    s.require(os.environ.get("MAJOR2_WHOLE_PARENT_REVIEW") == mh, "explicit parent local review")
    m = s.manifest(bundle/"manifest.json", mh)
    s.require(s.sha(bundle/"campaign.sbatch") == bh and
              (bundle/"campaign.sbatch").read_text() == batch_text(mh) and
              m["pin"] == pin, "local bundle")
    s.source_check(m, s.ROOT)
    permanent_lock(bundle.parent/"parent-finalizer.attempt")
    command = (f"MAJOR2_WHOLE_PARENT_REVIEW={mh} {s.E}/bin/python -B "
        f"{s.SOURCE}/devtools/major2/whole_gpu_scaling/custody.py parent-submit {pin} {mh} {bh}")
    try:
        result = subprocess.run([phx,command], env=dict(os.environ, PHX_TIMEOUT="180"),
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
    mode,*args = sys.argv[1:]
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
