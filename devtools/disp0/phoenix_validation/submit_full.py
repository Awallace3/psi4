"""Submit exactly one user-requested bounded full-system H200 allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from full_inputs import load_point
from gpu_runtime import digest,read_config
from prepare_inputs import SYSTEMS
from run_full import require_release

GIB=1024**3
LIMITS={"threads":8,"host_memory":"96 GiB","device_memory_limit_bytes":120*GIB,
        "native_persistent_bytes":12*GIB}


def command(source,run,account):
    return ["sbatch","--parsable","--job-name=disp0-full-h200",
        "--account="+account,"--partition=gpu-h200","--qos=inferno",
        "--gres=gpu:h200:1","--nodes=1","--ntasks=1","--cpus-per-task=8",
        "--mem=160G","--time=08:00:00","--no-requeue","--signal=TERM@120",
        "--chdir="+str(run),"--output="+str(run/"logs/%x-%j.out"),
        "--error="+str(run/"logs/%x-%j.err"),"--export=ALL",
        str(source/"devtools/disp0/phoenix_validation/full_job.sbatch")]


def main():
    parser=argparse.ArgumentParser()
    for name in ("source","config","inputs","release","run-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("config","inputs","release"):
        parser.add_argument("--"+name+"-sha256",required=True)
    parser.add_argument("--account",choices=("gts-cs207","gts-cs207-chemx"),required=True)
    args=parser.parse_args()
    source=args.source.resolve(strict=True)
    config=args.config.resolve(strict=True);inputs=args.inputs.resolve(strict=True)
    release=args.release.resolve(strict=True)
    original,build=read_config(config,args.config_sha256)
    if original["device_family"]!="H200":raise RuntimeError("Only qualified H200 runtime admitted")
    derived={**original,**LIMITS,"qualified_small_config_sha256":args.config_sha256}
    require_release(release,args.release_sha256,derived)
    for system in SYSTEMS:load_point(inputs,args.inputs_sha256,system)
    pin=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Committed clean helper checkout required")
    run=args.run_root
    if not run.is_absolute():raise RuntimeError("Absolute fresh output root required")
    run.mkdir(mode=0o700);(run/"logs").mkdir()
    def save(name,value):
        with (run/name).open("x") as stream:json.dump(value,stream,indent=2,sort_keys=True)
    save("runtime-config.json",derived)
    checksum=digest(run/"runtime-config.json")
    # No library/module/runtime changes: only declared thread/memory policies.
    env={k:v for k,v in os.environ.items() if not k.startswith("SBATCH_")}
    env.update(DISP0_SOURCE=str(source),DISP0_PIN=pin,DISP0_RUN_ROOT=str(run),
        DISP0_GATE_ENV=original["dependency_environment"],DISP0_FULL_CONFIG=str(run/"runtime-config.json"),
        DISP0_FULL_CONFIG_SHA256=checksum,DISP0_FULL_INPUTS=str(inputs),
        DISP0_FULL_INPUTS_SHA256=args.inputs_sha256,DISP0_FULL_RELEASE=str(release),
        DISP0_FULL_RELEASE_SHA256=args.release_sha256)
    argv=command(source,run,args.account)
    save("manifest.json",{"scope":"FULL_SYSTEM_EXACT_RESPONSE_DIAGNOSTIC_NOT_SPEEDUP",
        "helper_pin":pin,"science_pin":original["source_pin"],"command":argv,
        "systems":list(SYSTEMS),"repeats_per_arm":3,"independent_oracles_per_system":3,
        "resource_policy":LIMITS,"runtime_config_sha256":checksum,
        "qualified_small_config":str(config),"qualified_small_config_sha256":args.config_sha256,
        "inputs":str(inputs),"inputs_sha256":args.inputs_sha256,
        "release":str(release),"release_sha256":args.release_sha256,
        "account":args.account,"qos":"inferno","memory_GiB":160,"wall_seconds":28800,
        "automatic_retry":False,"stop_on_first_failure":True})
    trial=subprocess.run([argv[0],"--test-only",*argv[1:]],env=env,capture_output=True,
                         text=True,timeout=60)
    save("test-only.json",{"returncode":trial.returncode,"stdout":trial.stdout,"stderr":trial.stderr})
    print(trial.stdout+trial.stderr,flush=True)
    if trial.returncode:raise RuntimeError("Exact route rejected; no job submitted")
    with (run/"SUBMISSION_ATTEMPTED").open("x") as stream:stream.write("One scheduler call only\n")
    response=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
    save("submission.json",{"returncode":response.returncode,
                           "stdout":response.stdout,"stderr":response.stderr})
    print(response.stdout+response.stderr,flush=True)
    if response.returncode:raise RuntimeError("Submission failed/ambiguous; inspect before retry")
    job=response.stdout.strip().split(";")[0]
    if not job.isdecimal():raise RuntimeError("Ambiguous job ID; do not resubmit")
    with (run/"job-id.txt").open("x") as stream:stream.write(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(run),flush=True)


if __name__=="__main__":main()
