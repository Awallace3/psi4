"""Prepare separated jobs; each explicit submit-one is one scheduler call."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from full_inputs import load_point
from gpu_runtime import digest,read_config
from run_full import require_release
from split_jobs import SYSTEMS,admit_cpu_producer
from submit_full import LIMITS


def command(source,run,kind,account,qos,parent=None,*,completed_producer=False,wall_seconds=28800):
    if kind not in ("cpu","gpu") or qos not in ("embers","inferno"):
        raise ValueError("Explicit separated resource class and QoS required")
    gpu=kind=="gpu"
    if wall_seconds not in ((28800,57600) if gpu else (28800,86400)):
        raise ValueError("Explicit bounded single/grouped wall limit required")
    wall_time=f"{wall_seconds//3600:02d}:00:00"
    args=["sbatch","--parsable","--job-name=disp0-split-"+kind,
        "--account="+account,"--partition="+("gpu-h200" if gpu else "cpu-small"),
        "--qos="+qos,"--nodes=1","--ntasks=1",
        "--cpus-per-task="+("8" if gpu else "24"),"--mem=160G","--time="+wall_time,
        "--no-requeue","--signal=TERM@120","--chdir="+str(run),
        "--output="+str(run/"logs/%x-%j.out"),"--error="+str(run/"logs/%x-%j.err"),
        "--export=ALL"]
    if gpu:
        if parent is not None or completed_producer:
            raise ValueError("GPU jobs must never wait for CPU references")
        args.append("--gres=gpu:h200:1")
    else:
        args.extend(("--exclusive","--constraint=core24&gold6226&localNVMe"))
        if completed_producer:
            if parent is not None:raise ValueError("Completed artifact mode has no scheduler dependency")
        else:
            if not parent or not str(parent).isdecimal():raise ValueError("Exact GPU producer ID required")
            args.extend(("--dependency=afterany:"+str(parent),"--kill-on-invalid-dep=yes"))
    return [*args,str(source/"devtools/disp0/phoenix_validation/split_job.sbatch")]


def save_new(path,value):
    with path.open("x") as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)


def source_pin(source):
    pin=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Clean immutable helper checkout required")
    return pin


def prepare(args):
    source=args.source.resolve(strict=True)
    pin=source_pin(source)
    config,build=read_config(args.config,args.config_sha256)
    gpu_config={**config,**LIMITS,"qualified_small_config_sha256":args.config_sha256}
    require_release(args.release,args.release_sha256,gpu_config)
    for system in SYSTEMS:load_point(args.inputs,args.inputs_sha256,system)
    root=args.root
    if not root.is_absolute():raise ValueError("Absolute fresh plan root required")
    root.mkdir(mode=0o700)
    plan={"status":"PREPARED_NOT_SUBMITTED","source":str(source),"helper_pin":pin,
          "scientific_pin":config["source_pin"],"jobs":{},"preserve_peptide_job":"14396956",
          "automatic_retry":False,"new_gpu_jobs":2,"new_cpu_jobs":6}
    for system in SYSTEMS:
        parent=system+"-gpu"
        parent_root=root/parent
        gpu_config_path=parent_root/"runtime-config.json"
        for kind,repeats in (("gpu",(None,)),("cpu",range(3))):
            for repeat in repeats:
                label=parent if kind=="gpu" else f"{system}-cpu-r{repeat}"
                run=root/label;run.mkdir();(run/"logs").mkdir()
                selected={**gpu_config,"threads":8 if kind=="gpu" else 24}
                save_new(run/"runtime-config.json",selected)
                draft={"kind":kind,"system":system,"repeat":repeat,"helper_pin":pin,
                    "science_pin":config["source_pin"],
                    "source":str(source),"config":str(run/"runtime-config.json"),
                    "config_sha256":digest(run/"runtime-config.json"),
                    "inputs":str(args.inputs.resolve(strict=True)),"inputs_sha256":args.inputs_sha256,
                    "release":str(args.release.resolve(strict=True)),"release_sha256":args.release_sha256,
                    "cpu_threads":8 if kind=="gpu" else 24,"memory_GiB":160,"wall_seconds":28800}
                if kind=="cpu":
                    draft.update(gpu_parent_label=parent,gpu_config=str(gpu_config_path),
                                 gpu_config_sha256=digest(gpu_config_path))
                save_new(run/"draft-manifest.json",draft)
                plan["jobs"][label]={"root":str(run),
                    "draft_sha256":digest(run/"draft-manifest.json"),
                    "config_sha256":draft["config_sha256"],"kind":kind}
    save_new(root/"plan.json",plan)
    print(json.dumps({"status":plan["status"],"plan":str(root/"plan.json"),
        "plan_sha256":digest(root/"plan.json"),"jobs":list(plan["jobs"])},indent=2))


def submit_one(args):
    plan=json.loads((args.root/"plan.json").read_text())
    source=Path(plan["source"])
    if source_pin(source)!=plan["helper_pin"]:raise RuntimeError("Prepared helper pin changed")
    item=plan["jobs"][args.submit_one];run=Path(item["root"])
    if digest(run/"draft-manifest.json")!=item["draft_sha256"]:
        raise RuntimeError("Prepared split manifest changed")
    draft=json.loads((run/"draft-manifest.json").read_text())
    if digest(Path(draft["config"]))!=item["config_sha256"]:
        raise RuntimeError("Prepared split runtime configuration changed")
    config,build=read_config(Path(draft["config"]),draft["config_sha256"])
    require_release(Path(draft["release"]),draft["release_sha256"],config)
    load_point(Path(draft["inputs"]),draft["inputs_sha256"],draft["system"])
    parent=None
    completed_producer=bool(args.completed_producer)
    if draft["kind"]=="cpu":
        producer=Path(plan["jobs"][draft["gpu_parent_label"]]["root"])
        parent=(producer/"job-id.txt").read_text().strip()
        if not parent.isdecimal():raise RuntimeError("Exact accepted GPU job ID required")
        draft["gpu_job_root"]=str(producer/("job-"+parent))
        if completed_producer:
            # Explicit, artifact-gated late validation; never resubmit the GPU
            # merely because its controller record has been purged.
            admit_cpu_producer(draft,config)
            draft["completed_gpu_producer_job_id"]=parent
            parent=None
        else:
            registered=subprocess.run(["scontrol","show","job","-o",parent],
                capture_output=True,text=True,timeout=30)
            if registered.returncode:
                raise RuntimeError("GPU controller record unavailable; inspect query failure or "
                    "explicitly use --completed-producer after validating finished artifacts: "
                    +registered.stderr[:1000])
    elif completed_producer:
        raise ValueError("Completed producer mode is only for CPU validation jobs")
    manifest={**draft,"account":args.account,"qos":args.qos}
    save_new(run/"manifest.json",manifest)
    argv=command(source,run,draft["kind"],args.account,args.qos,parent,
                 completed_producer=completed_producer)
    env={k:v for k,v in os.environ.items() if not k.startswith("SBATCH_")}
    env.update(DISP0_SOURCE=str(source),DISP0_PIN=plan["helper_pin"],
               DISP0_RUN_ROOT=str(run),DISP0_GATE_ENV=config["dependency_environment"],
               DISP0_SPLIT_THREADS=str(draft["cpu_threads"]),DISP0_SPLIT_KIND=draft["kind"])
    trial=subprocess.run([argv[0],"--test-only",*argv[1:]],env=env,capture_output=True,
                         text=True,timeout=60)
    save_new(run/"test-only.json",{"returncode":trial.returncode,"stdout":trial.stdout,"stderr":trial.stderr})
    print(trial.stdout+trial.stderr,flush=True)
    if trial.returncode:raise RuntimeError("Exact split route rejected; not submitted")
    save_new(run/"SUBMISSION_ATTEMPTED.json",{"command":argv})
    response=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
    save_new(run/"submission.json",{"returncode":response.returncode,
        "stdout":response.stdout,"stderr":response.stderr})
    print(response.stdout+response.stderr,flush=True)
    if response.returncode:raise RuntimeError("Ambiguous/failed submission; never automatically retry")
    job=response.stdout.strip().split(";")[0]
    if not job.isdecimal():raise RuntimeError("Unparseable accepted job ID")
    with (run/"job-id.txt").open("x") as stream:stream.write(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(run),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--submit-one")
    parser.add_argument("--completed-producer",action="store_true")
    parser.add_argument("--account",choices=("gts-cs207","gts-cs207-chemx"))
    parser.add_argument("--qos",choices=("embers","inferno"))
    for name in ("source","config","inputs","release"):
        parser.add_argument("--"+name,type=Path)
    for name in ("config","inputs","release"):
        parser.add_argument("--"+name+"-sha256")
    args=parser.parse_args()
    if args.submit_one:
        if not args.account or not args.qos:
            parser.error("Submit-one requires explicit current account selection and QoS approval")
        submit_one(args)
    else:
        if args.completed_producer:parser.error("Completed producer mode requires explicit submit-one")
        for name in ("source","config","inputs","release","config_sha256","inputs_sha256","release_sha256"):
            if getattr(args,name) is None:parser.error("Preparation requires "+name)
        prepare(args)


if __name__=="__main__":main()
