"""One bounded H200 numerical gate, explicit QoS/account, no automatic retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(source, run, *, qos="embers", account="gts-cs207-chemx"):
    return ["sbatch","--parsable","--job-name=disp0-fragment-gate",
        "--account="+account,"--partition=gpu-h200","--qos="+qos,
        "--gres=gpu:h200:1","--nodes=1","--ntasks=1","--cpus-per-task=4",
        "--mem=32G","--time=01:00:00","--no-requeue","--signal=TERM@90",
        "--chdir="+str(run),"--output="+str(run/"logs/%x-%j.out"),
        "--error="+str(run/"logs/%x-%j.err"),"--export=ALL",
        str(source/"devtools/disp0/phoenix_validation/gate_job.sbatch")]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--qos",choices=("embers","inferno"),default="embers")
    parser.add_argument("--account",choices=("gts-cs207","gts-cs207-chemx"))
    args=parser.parse_args()
    if args.qos=="inferno" and args.account is None:
        raise RuntimeError("Fresh inferno approval and explicit live-selected account required")
    source=args.source.resolve(strict=True);config=args.config.resolve(strict=True)
    if digest(config)!=args.config_sha256:raise RuntimeError("Frozen runtime config differs")
    value=json.loads(config.read_text())
    preparation=config.parent/"preparation-result.json"
    prepared=json.loads(preparation.read_text())
    if (prepared["status"]!="PREPARED_NOT_GPU_VALIDATED"
            or prepared["runtime_config_sha256"]!=args.config_sha256):
        raise RuntimeError("CPU runtime preparation did not finish")
    # Require both actual native public CPU modes BEFORE allocating a GPU.
    for name in ("unscaled","scaled"):
        result=json.loads((config.parents[1]/("cpu-fragment-"+name)/"result.json").read_text())
        if (result["status"]!="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                or result["source_pin"]!=value["source_pin"]):
            raise RuntimeError("Matched native CPU fragment preflight incomplete")
    pin=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Gate helpers must belong to a clean immutable source")
    for filename in ("gate_job.sbatch","gate_payload.sh","run_gate.py","fragment_gate.py"):
        if not (source/"devtools/disp0/phoenix_validation"/filename).is_file():
            raise RuntimeError("Missing committed H200 gate payload: "+filename)
    run=args.run_root
    if not run.is_absolute():raise RuntimeError("Absolute fresh output root required")
    run.mkdir(mode=0o700);(run/"logs").mkdir()
    account=args.account or "gts-cs207-chemx"
    argv=command(source,run,qos=args.qos,account=account)
    env={k:v for k,v in os.environ.items() if not k.startswith("SBATCH_")}
    env.update(DISP0_SOURCE=str(source),DISP0_PIN=pin,DISP0_RUN_ROOT=str(run),
        DISP0_GATE_CONFIG=str(config),DISP0_GATE_CONFIG_SHA256=args.config_sha256,
        DISP0_GATE_ENV=value["dependency_environment"])
    manifest={"scope":"SMALL_H200_DIAGNOSTIC_GATE_NOT_TIMING_CAMPAIGN",
        "helper_pin":pin,"scientific_pin":value["source_pin"],"command":argv,
        "runtime_config":str(config),"runtime_config_sha256":args.config_sha256,
        "qos":args.qos,"account":account,"gpu_count":1,"gpu_type":"H200",
        "cpu_count":4,"memory_GiB":32,"time_limit":"01:00:00"}
    def save(name,value):
        with (run/name).open("x") as stream:json.dump(value,stream,indent=2,sort_keys=True)
    save("manifest.json",manifest)
    trial=subprocess.run([argv[0],"--test-only",*argv[1:]],env=env,capture_output=True,
                         text=True,timeout=60)
    save("test-only.json",{"returncode":trial.returncode,"stdout":trial.stdout,"stderr":trial.stderr})
    print(trial.stdout+trial.stderr,flush=True)
    if trial.returncode:raise RuntimeError("Exact-route preflight rejected; not submitted")
    with (run/"SUBMISSION_ATTEMPTED").open("x") as stream:stream.write("One scheduler call only\n")
    submitted=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
    save("submission.json",{"returncode":submitted.returncode,
                           "stdout":submitted.stdout,"stderr":submitted.stderr})
    print(submitted.stdout+submitted.stderr,flush=True)
    if submitted.returncode:raise RuntimeError("Submission failed/ambiguous; do not automatically retry")
    job=submitted.stdout.strip().split(";")[0]
    if not job.isdecimal():raise RuntimeError("Unparseable job ID; inspect saved response")
    with (run/"job-id.txt").open("x") as stream:stream.write(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(run),flush=True)


if __name__=="__main__":main()
