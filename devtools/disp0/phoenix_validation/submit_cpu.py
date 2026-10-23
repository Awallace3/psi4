"""Submit exactly one explicitly selected production CPU comparison allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

SHAPES={"peptide":("01:00:00",64,"40 GiB",3600),
        "nanotube":("02:00:00",64,"40 GiB",7200),
        "protein83":("08:00:00",160,"96 GiB",28800)}
CONSTRAINT="core24&gold6226&localNVMe"


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_pilot(outcome,status,*,helper_pin,source_pin,inputs_sha256,config_sha256):
    if (outcome["status"]!="PASS_PRODUCTION_CPU_COMPARISON" or outcome["system"]!="peptide"
            or outcome["accepted_cpu_sample_count"]!=6 or outcome["helper_pin"]!=helper_pin
            or outcome["source_pin"]!=source_pin
            or outcome["input_manifest_sha256"]!=inputs_sha256
            or outcome["dependency_proof_sha256"]!=config_sha256
            or "exit_code=0\n" not in status or "term_received=0\n" not in status):
        raise RuntimeError("Peptide pilot is incomplete or differs from this exact CPU comparison")


def validate_accounting(text,job):
    rows={v[0]:v[1:3] for line in text.splitlines() if len(v:=line.split("|"))>=3}
    if any(rows.get(key)!=["COMPLETED","0:0"] for key in (job,job+".batch",job+".0")):
        raise RuntimeError("Peptide job/step successful terminal accounting not established")


def command(source,run,system,*,qos,account):
    wall,mem,_,_=SHAPES[system]
    return ["sbatch","--parsable","--job-name=fsapt-cpu-"+system,
        "--account="+account,"--partition=cpu-small","--qos="+qos,
        "--nodes=1","--ntasks=1","--cpus-per-task=24","--mem="+str(mem)+"G",
        "--exclusive","--constraint="+CONSTRAINT,"--tmp=512G","--time="+wall,
        "--no-requeue","--signal=TERM@90","--chdir="+str(run),
        "--output="+str(run/"logs/%x-%j.out"),"--error="+str(run/"logs/%x-%j.err"),
        "--export=ALL",str(source/"devtools/disp0/phoenix_validation/cpu_job.sbatch")]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--inputs-sha256",required=True)
    parser.add_argument("--system",choices=SHAPES,required=True)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--pilot-job-run",type=Path,
                        help="Successful terminal peptide allocation required for larger systems")
    parser.add_argument("--qos",choices=("embers","inferno"),required=True)
    parser.add_argument("--account",choices=("gts-cs207","gts-cs207-chemx"),required=True)
    args=parser.parse_args()
    source=args.source.resolve(strict=True);config=args.config.resolve(strict=True)
    inputs=args.inputs.resolve(strict=True);run=args.run_root
    if digest(config)!=args.config_sha256 or digest(inputs)!=args.inputs_sha256:
        raise RuntimeError("Frozen dependency proof or inputs changed")
    value=json.loads(config.read_text())
    proof=Path(value["build_proof"])
    if digest(proof)!=value["build_proof_sha256"] or json.loads(proof.read_text())["status"]!="PASS":
        raise RuntimeError("Missing exact completed whole-stage build proof")
    pin=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip():
        raise RuntimeError("CPU helpers require a clean pinned worktree")
    pilot_proof=None
    if args.system!="peptide":
        if args.pilot_job_run is None:raise RuntimeError("A successful terminal peptide pilot is required")
        pilot=args.pilot_job_run.resolve(strict=True)
        outcome=json.loads((pilot/"comparison-result.json").read_text())
        status=(pilot/"job-status.env").read_text()
        validate_pilot(outcome,status,helper_pin=pin,source_pin=value["source_pin"],
                       inputs_sha256=args.inputs_sha256,config_sha256=args.config_sha256)
        job=pilot.name.removeprefix("job-")
        if not job.isdecimal():raise RuntimeError("Invalid pilot job identity")
        accounting=subprocess.check_output(["sacct","-n","-P","-j",job,
            "--format=JobIDRaw,State,ExitCode"],text=True,timeout=30)
        validate_accounting(accounting,job)
        pilot_proof={"run":str(pilot),"comparison_sha256":digest(pilot/"comparison-result.json"),
                     "accounting":accounting}
    for name in ("cpu_job.sbatch","cpu_payload.sh","cpu_benchmark.py","compare_cpu.py",
                 "validate_cpu_allocation.py"):
        if not (source/"devtools/disp0/phoenix_validation"/name).is_file():
            raise RuntimeError("Missing committed CPU helper: "+name)
    if not run.is_absolute():raise RuntimeError("Absolute fresh output root required")
    run.parent.mkdir(mode=0o700,exist_ok=True)
    run.mkdir(mode=0o700);(run/"logs").mkdir()
    wall,mem,psi_mem,seconds=SHAPES[args.system]
    env={k:v for k,v in os.environ.items() if not k.startswith("SBATCH_")}
    env.update(DISP0_SOURCE=str(source),DISP0_PIN=pin,DISP0_RUN_ROOT=str(run),
        DISP0_GATE_ENV=value["dependency_environment"],DISP0_GATE_CONFIG=str(config),
        DISP0_GATE_CONFIG_SHA256=args.config_sha256,DISP0_INPUTS=str(inputs),
        DISP0_INPUTS_SHA256=args.inputs_sha256,DISP0_CPU_SYSTEM=args.system,
        DISP0_CPU_MEMORY=psi_mem,DISP0_CPU_WALL_SECONDS=str(seconds))
    argv=command(source,run,args.system,qos=args.qos,account=args.account)
    manifest={"scope":"PRODUCTION_SCALED_CPU_V2_CPP_COMPARISON",
        "helper_pin":pin,"scientific_pin":value["source_pin"],"system":args.system,
        "command":argv,"account":args.account,"qos":args.qos,"cpus":24,"gpu_count":0,
        "memory_GiB":mem,"psi4_memory":psi_mem,"wall_time":wall,
        "input_manifest":str(inputs),"input_sha256":args.inputs_sha256,
        "dependency_proof_config":str(config),"dependency_proof_sha256":args.config_sha256,
        "repeats_per_arm":3,"arms":["einsums","cpp"],"preflight":"water pair, not timed samples",
        "ind_response":False,"ind_scale":True,"exchange_scale":True,
        "note":"Production scalar-rescaled fragment response. Not comparable to exact GPU "
               "fragment timings until a matching GPU production route is qualified."}
    def save(name,data):
        with (run/name).open("x") as stream:json.dump(data,stream,indent=2,sort_keys=True)
    save("manifest.json",manifest)
    if pilot_proof is not None:save("peptide-pilot-proof.json",pilot_proof)
    trial=subprocess.run([argv[0],"--test-only",*argv[1:]],env=env,capture_output=True,text=True,timeout=60)
    save("test-only.json",{"returncode":trial.returncode,"stdout":trial.stdout,"stderr":trial.stderr})
    print(trial.stdout+trial.stderr,flush=True)
    if trial.returncode or "partition cpu-small" not in trial.stdout+trial.stderr:
        raise RuntimeError("Expected exact CPU-small route not admitted; no submission")
    with (run/"SUBMISSION_ATTEMPTED").open("x") as stream:stream.write("One scheduler call only\n")
    submitted=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
    save("submission.json",{"returncode":submitted.returncode,"stdout":submitted.stdout,"stderr":submitted.stderr})
    print(submitted.stdout+submitted.stderr,flush=True)
    if submitted.returncode:raise RuntimeError("Submission failed/ambiguous; never retry automatically")
    job=submitted.stdout.strip().split(";")[0]
    if not job.isdecimal():raise RuntimeError("Unparseable job receipt")
    with (run/"job-id.txt").open("x") as stream:stream.write(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(run),flush=True)


if __name__=="__main__":main()
