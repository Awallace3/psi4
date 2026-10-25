"""One bounded, fail-closed full-system H200 qualification, never auto-resubmit.

These instrumented calls retain matrices/raw HF and independent native CPU
oracles. Report diagnostic elapsed times, NOT accepted production speedups.
The prior scalar-rescaled production CPU batch is deliberately not consumed.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import signal
import numpy as np

from full_inputs import load_point
from gpu_runtime import digest,read_config
from prepare_inputs import SYSTEMS
from repeat_gauge import compare_repeat
from run_gate import compare,environment
from validate_cpu_allocation import fields,memory_bytes,seconds
from owned_process import subreaper,group_alive,unwind,termination_signals


def save(path,value):
    temp=path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
    temp.replace(path)


def require_release(path,checksum,config):
    if digest(path)!=checksum:
        raise RuntimeError("Small H200 release proof changed")
    proof=json.loads(path.read_text())
    if (proof["status"]!="PASS_SMALL_H200_ONLY"
            or not proof["gpu_small_case_correctness_verified"]
            or proof["source_pin"]!=config["source_pin"]
            or proof["runtime_config_sha256"]!=config["qualified_small_config_sha256"]
            or len(proof["cases"])!=10
            or any(case["returncode"] for case in proof["cases"])):
        raise RuntimeError("Complete exact-source small H200 qualification required")
    return proof


def validate_allocation(job,manifest):
    if (job.get("Account")!=manifest["account"] or job.get("QOS")!="inferno"
            or job.get("Partition")!="gpu-h200" or job.get("NumNodes")!="1"
            or job.get("NumCPUs")!="8" or job.get("CPUs/Task")!="8"
            or job.get("NumTasks")!="1" or seconds(job["TimeLimit"])!=28800
            or memory_bytes(job["MinMemoryNode"])!=160*1024**3
            or "gres/gpu:h200=1" not in job.get("AllocTRES","").split(",")):
        raise RuntimeError("Effective allocation differs from declared H200 resource/charge limits")


def execute_owned(command,env,out,err,timeout,started):
    """Supervise/reap only this fresh process group; no local GPU lease logic."""
    with termination_signals():
        subreaper()
        child=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
        try:
            started(child.pid)
            code=child.wait(timeout=timeout)
            if group_alive(child.pid):
                raise RuntimeError("Full diagnostic left surviving owned descendants")
            return code
        except BaseException:
            selected=(signal.SIGTERM,signal.SIGHUP,signal.SIGINT)
            old={sig:signal.signal(sig,signal.SIG_IGN) for sig in selected}
            try:unwind(child)
            finally:
                for sig,handler in old.items():signal.signal(sig,handler)
            raise


def compare_energy(cpu,gpu):
    left,right=float(cpu["energy_Eh"]),float(gpu["energy_Eh"])
    if not np.isfinite([left,right]).all():
        raise ValueError("Finite signed CPU/GPU whole energies required")
    np.testing.assert_allclose(right,left,atol=2e-8,rtol=1e-8,
                               err_msg="Signed full interaction energy CPU/GPU")
    return {"cpu_Eh":left,"gpu_Eh":right,"difference_Eh":right-left}


def main():
    parser=argparse.ArgumentParser()
    for name in ("config","inputs","release","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("config","inputs","release"):
        parser.add_argument("--"+name+"-sha256",required=True)
    args=parser.parse_args()
    root=args.output.resolve(strict=True)
    manifest=json.loads((root.parent/"manifest.json").read_text())
    allocation={"status":"STARTED","job_id":os.environ["SLURM_JOB_ID"]}
    try:
        raw=subprocess.check_output(["scontrol","show","job","-o",allocation["job_id"]],
                                    text=True,timeout=30)
        allocation["scheduler_job"]=raw
        validate_allocation(fields(raw),manifest)
        allocation["cpu_affinity"]=sorted(os.sched_getaffinity(0))
        if len(allocation["cpu_affinity"])<8:
            raise RuntimeError("Fewer than8 allowed CPU threads")
        allocation["status"]="PASS_SCHEDULER_ALLOCATION_ONLY"
    except BaseException as error:
        allocation.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:save(root/"allocation.json",allocation)
    config,build=read_config(args.config,args.config_sha256)
    require_release(args.release,args.release_sha256,config)
    if (config["threads"]!=8 or config["host_memory"]!="96 GiB"
            or config["device_family"]!="H200"):
        raise RuntimeError("Declared full-system resource configuration required")
    helper=Path(__file__).resolve().parent
    scratch=Path(os.environ["TMPDIR"]).resolve(strict=True)
    deadline=time.monotonic()+8*3600-180
    result={"status":"STARTED","scope":"FULL_SYSTEM_EXACT_RESPONSE_DIAGNOSTIC",
        "helper_pin":os.environ["DISP0_PIN"],"source_pin":config["source_pin"],
        "runtime_config_sha256":args.config_sha256,
        "allocation_sha256":digest(root/"allocation.json"),
        "input_manifest_sha256":args.inputs_sha256,"release_sha256":args.release_sha256,
        "systems":{},"processes":[],"accepted_timing_count":0,"accepted_speedup_count":0,
        "note":"Instrumented exact response, not production scalar-rescaled CPU baseline. "
               "No speedup ratio is accepted from these diagnostics."}
    for system in SYSTEMS:load_point(args.inputs,args.inputs_sha256,system)
    def checkpoint():save(root/"full-result.json",result)
    checkpoint()

    def child(system,label,arm,oracle=None):
        output=root/(system+"-"+label)
        work=scratch/(system+"-"+label);work.mkdir()
        env=environment(config,gpu=arm=="gpu")
        env.update(TMPDIR=str(work),SCRATCH=str(work),PSI_SCRATCH=str(work))
        argv=[str(Path(config["stage"])/"bin/psi4"),config["dependency_environment"]+"/bin/python",
            str(helper/"fragment_gate.py"),"--config",str(args.config),
            "--config-sha256",args.config_sha256,"--inputs",str(args.inputs),
            "--inputs-sha256",args.inputs_sha256,"--system",system,
            "--arm",arm,"--scaled","--block-size","24","--output",str(output)]
        if oracle:argv.extend(("--oracle-from",str(oracle)))
        command=["/bin/bash","-c",
            'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
            "full-fsapt",*argv]
        row={"system":system,"label":label,"arm":arm,"argv":command}
        result["processes"].append(row);checkpoint()
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError("Eight-hour full-run bound exhausted")
        started=time.perf_counter()
        with (root/(output.name+".stdout")).open("x") as out,(
                root/(output.name+".stderr")).open("x") as err:
            try:
                def launched(pid):
                    row["pid"]=pid
                    checkpoint()
                code=execute_owned(command,env,out,err,remaining,launched)
            except BaseException as error:
                row.update(status="FAILED",error=f"{type(error).__name__}: {error}")
                raise
            finally:
                row["subprocess_seconds"]=time.perf_counter()-started
                checkpoint()
        row["returncode"]=code;checkpoint()
        if code:raise RuntimeError("Full-system diagnostic failed: "+output.name)
        document=json.loads((output/"result.json").read_text())
        if (document["status"]!="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                or document["system"]!=system
                or document["input_manifest_sha256"]!=args.inputs_sha256):
            raise RuntimeError("Incomplete/mismatched full-system result")
        row["result_sha256"]=digest(output/"result.json");checkpoint()
        return output,document
    try:
        for system in SYSTEMS:
            point,dimensions=load_point(args.inputs,args.inputs_sha256,system)
            summary={"status":"RUNNING","dimensions":dimensions,"pairs":[],"repeat_checks":[]}
            result["systems"][system]=summary;checkpoint()
            samples={"cpu":[],"gpu":[]}
            baselines={}
            for repeat in range(3):
                pair={}
                for arm in (("cpu","gpu") if repeat%2==0 else ("gpu","cpu")):
                    path,document=child(system,f"r{repeat}-{arm}",arm)
                    pair[arm]=(path,document)
                    samples[arm].append(document["diagnostic_whole_energy_seconds"])
                    if repeat:
                        summary["repeat_checks"].append({
                            "arm":arm,"repeat":repeat,"check":compare_repeat(
                                baselines[arm],path,dimensions=dimensions,arm=arm)})
                    else:baselines[arm]=path
                checks=compare(pair["cpu"][1],pair["gpu"][1])
                energy_check=compare_energy(pair["cpu"][1],pair["gpu"][1])
                for key in ("retained_orbital_rank","virtual_orbitals"):
                    if pair["cpu"][1][key]!=pair["gpu"][1][key]:
                        raise RuntimeError("Independent full CPU/GPU orbital spaces differ: "+key)
                # Every GPU call gets an independently rebuilt native CPU
                # all-field oracle, not just scalar agreement or a small gate.
                oracle_path,oracle=child(system,f"r{repeat}-oracle","cpu",oracle=pair["gpu"][0])
                summary["pairs"].append({"repeat":repeat,"signed_checks":checks,
                    "whole_energy_check":energy_check,
                    "same_gauge_fields":oracle["same_gauge_compared_fields"],
                    "same_gauge_errors":oracle["same_gauge_errors"],
                    "oracle_result_sha256":digest(oracle_path/"result.json")})
                checkpoint()
            summary.update(status="PASS_FULL_SYSTEM_DIAGNOSTIC",
                diagnostic_elapsed={arm:{"n":len(values),"median_seconds":statistics.median(values),
                    "minimum_seconds":min(values),"maximum_seconds":max(values)}
                    for arm,values in samples.items()})
            checkpoint()
        result["status"]="PASS_THREE_FULL_SYSTEM_DIAGNOSTICS"
    except BaseException as error:
        result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:checkpoint()


if __name__=="__main__":
    with termination_signals():main()
