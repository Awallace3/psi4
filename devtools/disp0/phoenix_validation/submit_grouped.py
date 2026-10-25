"""User-requested regrouping: one GPU job, small shared CPU, protein CPU job."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from fragment_gate import require_oracle_config
from full_inputs import load_point
from gpu_runtime import read_config,digest
from run_full import require_release
from split_jobs import sealed_gpu
from submit_full import LIMITS
from submit_split import command,save_new,source_pin


def job_specs():
    return {"gpu":{"kind":"gpu","systems":["nanotube","protein83"],"allocation_wall_seconds":57600},
            "cpu-small":{"kind":"cpu","systems":["nanotube"],"allocation_wall_seconds":86400},
            "cpu-protein83":{"kind":"cpu","systems":["protein83"],"allocation_wall_seconds":86400}}


def prepare(args):
    source=args.source.resolve(strict=True);pin=source_pin(source)
    config,_=read_config(args.config,args.config_sha256)
    gpu={**config,**LIMITS,"qualified_small_config_sha256":args.config_sha256}
    require_release(args.release,args.release_sha256,gpu)
    for system in ("nanotube","protein83"):load_point(args.inputs,args.inputs_sha256,system)
    root=args.root
    if not root.is_absolute():raise ValueError("Absolute fresh grouped root required")
    root.mkdir(mode=0o700)
    plan={"status":"PREPARED_NOT_SUBMITTED","source":str(source),"helper_pin":pin,"jobs":{},
          "preserve_peptide_job":"14396956","resource_policy":"24core CPU vs8core/H200; diagnostics",
          "user_regrouping":"ONE_GPU_SMALL_CPU_SHARED_PROTEIN_CPU_SEPARATE"}
    for label,spec in job_specs().items():
        run=root/label;run.mkdir();(run/"logs").mkdir()
        selected={**gpu,"threads":8 if spec["kind"]=="gpu" else 24}
        save_new(run/"runtime-config.json",selected)
        draft={**spec,"source":str(source),"helper_pin":pin,"science_pin":config["source_pin"],
            "config":str(run/"runtime-config.json"),"config_sha256":digest(run/"runtime-config.json"),
            "inputs":str(args.inputs.resolve(strict=True)),"inputs_sha256":args.inputs_sha256,
            "release":str(args.release.resolve(strict=True)),"release_sha256":args.release_sha256,
            "cpu_threads":selected["threads"],"memory_GiB":160,"repeats":3}
        if spec["kind"]=="cpu":
            draft.update(gpu_config=str(root/"gpu/runtime-config.json"),
                         gpu_config_sha256=digest(root/"gpu/runtime-config.json"))
        save_new(run/"draft-manifest.json",draft)
        plan["jobs"][label]={"root":str(run),"draft_sha256":digest(run/"draft-manifest.json"),
                            "config_sha256":draft["config_sha256"]}
    save_new(root/"plan.json",plan)
    print(json.dumps({"status":plan["status"],"plan":str(root/"plan.json"),
        "plan_sha256":digest(root/"plan.json"),"jobs":list(plan["jobs"])},indent=2))


def submit_one(args):
    plan=json.loads((args.root/"plan.json").read_text())
    source=Path(plan["source"])
    if source_pin(source)!=plan["helper_pin"]:raise RuntimeError("Grouped helper pin changed")
    item=plan["jobs"][args.submit_one];run=Path(item["root"])
    if digest(run/"draft-manifest.json")!=item["draft_sha256"]:
        raise RuntimeError("Prepared grouped manifest changed")
    draft=json.loads((run/"draft-manifest.json").read_text())
    if digest(Path(draft["config"]))!=item["config_sha256"]:
        raise RuntimeError("Grouped configuration changed")
    config,_=read_config(Path(draft["config"]),draft["config_sha256"])
    require_release(Path(draft["release"]),draft["release_sha256"],config)
    for system in draft["systems"]:load_point(Path(draft["inputs"]),draft["inputs_sha256"],system)
    parent=None
    if draft["kind"]=="cpu":
        producer=Path(plan["jobs"]["gpu"]["root"])
        parent=(producer/"job-id.txt").read_text().strip()
        if not parent.isdecimal():raise RuntimeError("Accepted one-GPU parent ID required")
        draft["gpu_group_job_root"]=str(producer/("job-"+parent))
        peer,_=read_config(Path(draft["gpu_config"]),draft["gpu_config_sha256"])
        require_oracle_config(config,peer)
        if args.completed_producer:
            for system in draft["systems"]:
                for repeat in range(3):
                    selected={**draft,"system":system,"repeat":repeat}
                    sealed_gpu(Path(draft["gpu_group_job_root"])/system,selected,
                               draft["gpu_config_sha256"],repeat)
            draft["completed_gpu_producer_job_id"]=parent;parent=None
        else:
            probe=subprocess.run(["scontrol","show","job","-o",parent],
                                 capture_output=True,text=True,timeout=30)
            if probe.returncode:
                raise RuntimeError("Parent unavailable; inspect or explicitly use completed-producer "
                                   "artifact mode. No new job submitted. "+probe.stderr[:1000])
    elif args.completed_producer:raise ValueError("Completed producer mode is CPU-only")
    manifest={**draft,"account":args.account,"qos":args.qos}
    save_new(run/"manifest.json",manifest)
    argv=command(source,run,draft["kind"],args.account,args.qos,parent,
        completed_producer=args.completed_producer,wall_seconds=draft["allocation_wall_seconds"])
    argv=[("--job-name=disp0-"+args.submit_one) if value.startswith("--job-name=") else value
          for value in argv]
    env={k:v for k,v in os.environ.items() if not k.startswith(("SBATCH_","SLURM_"))}
    env.update(DISP0_SOURCE=str(source),DISP0_PIN=plan["helper_pin"],DISP0_RUN_ROOT=str(run),
        DISP0_GATE_ENV=config["dependency_environment"],DISP0_SPLIT_THREADS=str(draft["cpu_threads"]),
        DISP0_SPLIT_KIND=draft["kind"],DISP0_SPLIT_DRIVER="grouped_jobs.py")
    trial=subprocess.run([argv[0],"--test-only",*argv[1:]],env=env,capture_output=True,
                         text=True,timeout=60)
    save_new(run/"test-only.json",{"returncode":trial.returncode,"stdout":trial.stdout,"stderr":trial.stderr})
    print(trial.stdout+trial.stderr,flush=True)
    if trial.returncode:raise RuntimeError("Grouped route rejected; not submitted")
    save_new(run/"SUBMISSION_ATTEMPTED.json",{"command":argv})
    response=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
    save_new(run/"submission.json",{"returncode":response.returncode,
                                  "stdout":response.stdout,"stderr":response.stderr})
    print(response.stdout+response.stderr,flush=True)
    if response.returncode:raise RuntimeError("Submission failed/ambiguous; never automatically retry")
    job=response.stdout.strip().split(";")[0]
    if not job.isdecimal():raise RuntimeError("Unparseable job ID; inspect before retry")
    with (run/"job-id.txt").open("x") as stream:stream.write(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(run),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--submit-one",choices=tuple(job_specs()))
    parser.add_argument("--account",choices=("gts-cs207","gts-cs207-chemx"))
    parser.add_argument("--qos",choices=("embers","inferno"))
    parser.add_argument("--completed-producer",action="store_true")
    for name in ("source","config","inputs","release"):
        parser.add_argument("--"+name,type=Path)
    for name in ("config","inputs","release"):parser.add_argument("--"+name+"-sha256")
    args=parser.parse_args()
    if args.submit_one:
        if not args.account or not args.qos:parser.error("Explicit account and QoS approval required")
        submit_one(args)
    else:
        if args.completed_producer:parser.error("Completed producer mode requires submit-one")
        for name in ("source","config","inputs","release","config_sha256","inputs_sha256","release_sha256"):
            if getattr(args,name) is None:parser.error("Preparation requires "+name)
        prepare(args)


if __name__=="__main__":main()
