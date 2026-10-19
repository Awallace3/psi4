"""Stdlib-only bounded login preparation; parent-submit is separately gated."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import common as n

PARENT = "beaaa33f5a23d7a1906e14ad3dc1fbf983fba4ce"
FAILED = ["14158602","14159755","14160001","14160573","14160948","14161975",
          "14164895","14165998","14169328","14170562","14173808","14174849"]
COMPLETED = ["14175243","14176359"]


def stat(path):
    s=Path(path).stat()
    return [s.st_size,s.st_mtime_ns,s.st_mode]


def small(path):
    p=Path(path)
    n.require(p.stat().st_size <= 16*1024**2,"small metadata only")
    return n.sha(p)


def snapshot(root):
    root=Path(root)
    files=[p for p in root.rglob("*") if p.is_file()]
    n.require(len(files)<1000 and all(not p.is_symlink() for p in files),"bounded old root")
    return {str(p.relative_to(root)):stat(p) for p in files}


def preserve(value):
    for root,expected in value["roots"].items():
        n.require(snapshot(root)==expected,"immutable old jobroot: "+root)
    for root,files in value["builds"].items():
        for name,expected in files.items():
            n.require(stat(Path(root)/name)==expected,"immutable scientific build sentinel")
    for path,digest in value["small_hashes"].items():
        n.require(small(path)==digest,"immutable metadata/staging")


def accounting():
    rows=[v.split("|") for v in n.output("sacct","-n","-P","-j",",".join(FAILED+COMPLETED),
             "--format=JobIDRaw,State,ExitCode").splitlines()]
    for jobs,state,code in [(FAILED,"FAILED","1:0"),(COMPLETED,"COMPLETED","0:0")]:
        for j in jobs:
            for ident in (j,j+".batch"):
                n.require(any(r[:3]==[ident,state,code] for r in rows),"historical accounting: "+ident)
    return rows


def queue():
    rows=n.output("squeue","-h","-r","-u","awallace43","-o","%i|%q|%j|%P").splitlines()
    n.require(sum(x.split("|")[1]=="embers" for x in rows)<50,"free embers headroom")
    for row in rows:
        fields=dict(re.findall(r"(\w+)=([^\s]+)",n.output("scontrol","show","job","-o",row.split("|")[0])))
        n.require(fields.get("UserId","").startswith("awallace43("),"queue identity")
        n.require(not any("major2" in fields.get(k,"").lower() for k in
                         ["JobName","Command","WorkDir"]) and
                  "gpu-h200" not in fields.get("Partition",""),"active duplicate/H200 holder")
    return rows


def collect():
    n.require(not Path(n.RUN).exists(),"new run already exists")
    old=n.load(Path(n.OLD)/"manifest.json")
    n.require(old["harness"]["pin"]==PARENT and
              old["frozen_factor"]["files"]==n.FACTOR_FILES,"old fixed factor policy")
    roots=old["preservation"]["failed_root_stats"]
    n.require(len(roots)==12,"12 failed roots")
    # Validate recorded prior snapshots before extending, without array reads.
    for root,expected in roots.items():
        n.require(snapshot(root)==expected,"prior failed root preservation")
    roots=dict(roots)
    roots[old["completed_control"]["root"]]=snapshot(old["completed_control"]["root"])
    roots[n.OLD]=snapshot(n.OLD)
    n.require(len(roots)==14,"14 total old jobroots")
    hashes={}
    for root in (old["completed_control"]["root"],n.OLD):
        n.require(not (Path(root)/"FAILED").exists(),"completed root must not be FAILED")
        marker=n.load(Path(root)/"ENDPOINT-COMPLETE.json")
        n.require(marker==dict(outputs=24,computes=24,accepted_speedup_data=False),"completed endpoint")
        n.require(len(list((Path(root)/"raw").glob("fixed-*/*.npz")))==24 and
                  len(list((Path(root)/"raw").glob("fixed-*/COMPLETE.json")))==6,
                  "completed 6-process/24-output root")
        for rel in ["manifest.json","campaign.sbatch","ENDPOINT-COMPLETE.json","STARTED.json"]:
            hashes[str(Path(root)/rel)]=small(Path(root)/rel)
    for item in old["prior_staging"].values():
        for rel in ("manifest.json","campaign.sbatch"):
            hashes[str(Path(item["root"])/rel)]=small(Path(item["root"])/rel)
    # Mandatory controlled-cohort environment receipts, no array read or numerical unblinding.
    receipts=list((Path(n.OLD)/"raw").glob("fixed-*.effective-env.json"))
    n.require(len(receipts)==6 and all(n.load(p)["observed"]=={"CUBLAS_WORKSPACE_CONFIG":":4096:8"}
                                      for p in receipts),"completed control ENV proofs")
    for p in receipts: hashes[str(p)]=small(p)
    pres=dict(roots=roots,builds=old["preservation"]["build_sentinel_stats"],small_hashes=hashes)
    preserve(pres)
    # Only tiny compiler wrapper read and version/frontend query on login; no compilation/import.
    wrapper=Path(n.CXX).read_text()
    n.require(n.CXX_REAL in wrapper or "x86_64-conda-linux-gnu" in wrapper,"compiler wrapper identity")
    env=dict(os.environ)
    for k in ("GCC_ROOT","CPATH","C_INCLUDE_PATH","CPLUS_INCLUDE_PATH","OBJC_INCLUDE_PATH","LIBRARY_PATH"):
        env.pop(k,None)
    version=n.output(n.CXX,"--version",env=env)
    frontend=n.output(n.CXX,"-print-prog-name=cc1plus",env=env)
    n.require("14.3.0" in version and frontend==n.CC1PLUS and os.access(frontend,os.X_OK),
              "actual GCC14 frontend; no site compiler fallback")
    completed_process=n.load(Path(n.OLD)/"raw/fixed-baseline-r1/COMPLETE.json")
    maps=completed_process["runtime_maps"]
    libraries={p:h for p,h in maps.items() if Path(p).name.startswith(
        ("libcuest.so","libcudart.so","libstdc++.so","libcublas","libcusolver","libcusparse","libnvJitLink","libnvrtc"))}
    n.require(all(any(Path(p).name.startswith(k) for p in libraries) for k in
                  ("libcuest.so","libcudart.so","libstdc++.so","libcublas.so","libcusolver.so")),
              "historical native runtime identities")
    n.require(libraries.get(str((Path(n.SDK)/"lib/13/libcuest.so.0").resolve()))==n.SDK_SHA,
              "exact pinned SDK binary identity")
    headers=list((Path(n.SDK)/"include").rglob("*.h"))
    n.require(len(headers)<200 and sum(p.stat().st_size for p in headers)<8*1024**2,"bounded public headers")
    return dict(preservation=pres,accounting=accounting(),queue=queue(),suite=old["suite"],
        suite_sha256=old["suite_sha256"],geometry_sha256=completed_process["input_geometry_sha256"],
        runtime_file=str(Path(n.OLD)/"build-baseline/runtime.env"),
        runtime_file_sha256=small(Path(n.OLD)/"build-baseline/runtime.env"),
        native_libraries=libraries,headers={str(p):n.sha(p) for p in headers},
        identity=old["identity_helper"],compiler=dict(wrapper=n.CXX,wrapper_sha256=small(n.CXX),
        actual=n.CXX_REAL,actual_resolved=str(Path(n.CXX_REAL).resolve(strict=True)),
        actual_sha256=n.CXX_REAL_SHA,frontend=frontend,frontend_sha256=n.CC1PLUS_SHA,
        version=version),base_pin=n.BASE_PIN,base_stage=n.STAGE,
        metadata_only=True,no_submission=True)


def batch_text(mh):
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=major2-native-plan
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
#SBATCH --chdir={n.RUN}
#SBATCH --output={n.RUN}/logs/%x-%j.out
#SBATCH --error={n.RUN}/logs/%x-%j.err
set -euo pipefail
exec bash {n.SOURCE}/devtools/major2/native_plan/supervise.sh {n.RUN} \\
  {n.E}/bin/python -B {n.SOURCE}/devtools/major2/native_plan/run.py \\
  {n.RUN}/manifest.json {mh}
"""


def freeze(collection, directory):
    root=n.HERE.parents[2]
    pin=n.output("git","-C",root,"rev-parse","HEAD")
    changes=n.output("git","-C",root,"diff","--name-status",PARENT,pin).splitlines()
    n.require(changes and all(x.startswith("A\tdevtools/major2/native_plan/") for x in changes),
              "only owned additions allowed")
    helpers={x.split("\t")[1]:n.sha(root/x.split("\t")[1]) for x in changes}
    m=dict(collection,policy=n.POLICY,source=n.SOURCE,run=n.RUN,parent=PARENT,pin=pin,
           helpers=helpers)
    n.source_check(m,root)
    directory=Path(directory); directory.mkdir()
    n.durable(directory/"manifest.json",m)
    mh=n.sha(directory/"manifest.json")
    n.durable(directory/"campaign.sbatch",batch_text(mh))
    n.durable(directory/"bundle-receipt.json",dict(pin=pin,manifest_sha256=mh,
        batch_sha256=n.sha(directory/"campaign.sbatch"),no_submission=True))
    print(json.dumps(n.load(directory/"bundle-receipt.json"),sort_keys=True))


def verify(pin,mh,bh,pristine=True):
    root=Path(n.RUN)
    m=n.manifest(root/"manifest.json",mh)
    n.require(m["pin"]==pin and n.sha(root/"campaign.sbatch")==bh and
              (root/"campaign.sbatch").read_text()==batch_text(mh),"exact ready bundle")
    n.source_check(m)
    n.require(n.output("git","-C",n.BASE,"rev-parse","HEAD")==n.BASE_PIN and
              not n.output("git","-C",n.BASE,"status","--porcelain","--untracked-files=no"),"BASE Git identity")
    if pristine:
        n.require(set(p.name for p in root.iterdir())==
                  {"manifest.json","campaign.sbatch","bundle-receipt.json","logs"} and
                  not list((root/"logs").iterdir()),"pristine new run; no resume")
    preserve(m["preservation"])
    return m


def stage(pin):
    payload=json.load(sys.stdin)
    n.require(set(payload)=={"manifest.json","campaign.sbatch","bundle-receipt.json"},"generated data only")
    root=Path(n.RUN); root.mkdir(); (root/"logs").mkdir()
    for name,value in payload.items(): n.durable(root/name,value)
    receipt=n.load(root/"bundle-receipt.json")
    verify(pin,receipt["manifest_sha256"],receipt["batch_sha256"])
    print(json.dumps(dict(staged=True,no_submission=True,**receipt),sort_keys=True))


def audit(pin,mh,bh,pristine=True):
    verify(pin,mh,bh,pristine)
    rows=queue(); acct=accounting()
    r=subprocess.run(["sbatch","--test-only",n.RUN+"/campaign.sbatch"],text=True,
                     capture_output=True,timeout=35)
    n.require(r.returncode==0 and "gpu-h200" in r.stdout+r.stderr,"H200 test-only failed")
    return dict(pin=pin,manifest_sha256=mh,batch_sha256=bh,old_roots=14,failed_roots=12,
                completed_roots=2,queue=rows,accounting=acct,no_submission=True,
                test_only_stdout=r.stdout,test_only_stderr=r.stderr)


def permanent_lock(path):
    Path(path).mkdir()
    n.fsync_dir(Path(path).parent)


def accept_id(text):
    n.require(bool(re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_-]+)?\n?",text)),
              "ambiguous sbatch response: STOP, never retry")
    return text.strip().split(";")[0]


def parent_submit(pin,mh,bh):
    n.require(os.environ.get("MAJOR2_NATIVE_PARENT_REVIEW")==mh,"parent review approval required")
    # Gate before lock, then immutable permanent attempt, then revalidate with lock present.
    gates=audit(pin,mh,bh)
    permanent_lock(Path(n.RUN)/"logs/parent-attempt")
    n.durable(Path(n.RUN)/"logs/pre-submit-gates.json",gates)
    # From this point no failures, timeouts or ambiguous responses permit another attempt.
    result=subprocess.run(["sbatch","--parsable",n.RUN+"/campaign.sbatch"],
                          capture_output=True,text=True,timeout=35)
    n.durable(Path(n.RUN)/"logs/sbatch-response.json",
              dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    n.require(result.returncode==0,"submission uncertain/rejected: STOP, no retry")
    job=accept_id(result.stdout)
    # Numeric accepted ID fsynced BEFORE any controller query.
    n.durable(Path(n.RUN)/"logs/job-id.txt",job+"\n")
    controller=n.output("scontrol","show","job","-o",job)
    n.durable(Path(n.RUN)/"logs/accepted-controller.txt",controller+"\n")
    fields=dict(re.findall(r"(\w+)=([^\s]+)",controller))
    n.require(fields.get("JobId")==job and fields.get("QOS")=="embers" and
              fields.get("Account")=="gts-cs207-chemx" and "gpu-h200" in fields.get("Partition","") and
              fields.get("Command")==n.RUN+"/campaign.sbatch","accepted controller route")
    print(json.dumps(dict(job=job,no_retry=True)))


def main():
    n.check_env(os.environ)
    mode,*args=sys.argv[1:]
    if mode=="collect": print(json.dumps(collect(),sort_keys=True))
    elif mode=="freeze": freeze(n.load(args[0]),args[1])
    elif mode=="stage": stage(*args)
    elif mode=="audit": print(json.dumps(audit(*args),sort_keys=True))
    elif mode=="parent-submit": parent_submit(*args)
    else: raise RuntimeError("unknown handoff mode")


if __name__=="__main__": main()
