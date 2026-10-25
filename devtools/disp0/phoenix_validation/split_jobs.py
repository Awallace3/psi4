"""Separated GPU calculation / CPU validation jobs; never wait holding a GPU."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from full_inputs import load_point
from fragment_gate import require_oracle_config
from gpu_runtime import digest,read_config
from repeat_gauge import compare_repeat
from run_full import save,require_release,execute_owned,compare_energy
from run_gate import environment,compare
from validate_cpu_allocation import fields,memory_bytes,seconds
from split_scratch import OwnedScratch

SYSTEMS=("nanotube","protein83")  # Peptide already has three qualified pairs.


def validate_allocation(job,manifest):
    gpu=manifest["kind"]=="gpu"
    threads=8 if gpu else 24
    partition="gpu-h200" if gpu else "cpu-small"
    allocated=job.get("AllocTRES","").split(",")
    if (job.get("Account")!=manifest["account"] or job.get("QOS")!=manifest["qos"]
            or job.get("Partition")!=partition or job.get("NumNodes")!="1"
            or job.get("NumCPUs")!=str(threads) or job.get("CPUs/Task")!=str(threads)
            or job.get("NumTasks")!="1"
            or seconds(job["TimeLimit"])!=manifest.get("allocation_wall_seconds",28800)
            or memory_bytes(job["MinMemoryNode"])!=160*1024**3):
        raise RuntimeError("Split allocation differs from its declared CPU/GPU route")
    if gpu:
        if "gres/gpu:h200=1" not in allocated:
            raise RuntimeError("GPU-only job requires exactly one H200")
    elif (any(key.startswith("gres/gpu") for key in allocated)
          or os.environ.get("SLURM_JOB_GPUS")):
        raise RuntimeError("CPU references/oracles must not reserve any GPU")


def sealed_gpu(root,manifest,config_sha,repeat):
    """Consume only a finalized successful producer; no scheduler wait loop."""
    if (root/"job-status.env").read_text().splitlines()!=["exit_code=0","term_received=0"]:
        raise RuntimeError("GPU producer job did not complete cleanly")
    proof=json.loads((root/"split-result.json").read_text())
    if (proof["status"]!="PASS_GPU_EXECUTION_REPEATS_AWAITING_CPU"
            or proof["system"]!=manifest["system"]
            or proof["input_manifest_sha256"]!=manifest["inputs_sha256"]
            or proof["runtime_config_sha256"]!=config_sha
            or proof["source_pin"]!=manifest["science_pin"]
            or proof["helper_pin"]!=manifest["helper_pin"]
            or len(proof["gpu_outputs"])!=3):
        raise RuntimeError("GPU producer is incomplete/failed/mismatched; CPU job will not run")
    row=proof["gpu_outputs"][repeat]
    path=root/f"r{repeat}-gpu"
    if row["repeat"]!=repeat or digest(path/"result.json")!=row["result_sha256"]:
        raise RuntimeError("GPU producer result seal changed")
    doc=json.loads((path/"result.json").read_text())
    if (doc["status"]!="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
            or doc["system"]!=manifest["system"] or doc["arm"]!="gpu"
            or doc["source_pin"]!=manifest["science_pin"]
            or doc["runtime_config_sha256"]!=config_sha):
        raise RuntimeError("Completed GPU diagnostic required")
    for name in ("raw-hf.npz","fragment-matrices.npz"):
        if digest(path/name)!=doc["artifact_sha256"][name]:
            raise RuntimeError("GPU raw/matrix artifact seal changed")
    return path,doc


def worker_command(helper,config,manifest,output,arm,oracle=None):
    # This guard makes it impossible for an H200 job's process plan to include
    # a CPU reference or fixed-gauge CPU oracle.
    if manifest["kind"]=="gpu" and (arm!="gpu" or oracle is not None):
        raise RuntimeError("GPU allocations cannot execute CPU references/oracles")
    if manifest["kind"]=="cpu" and arm!="cpu":
        raise RuntimeError("CPU allocations cannot execute GPU calculations")
    argv=[str(Path(config["stage"])/"bin/psi4"),config["dependency_environment"]+"/bin/python",
        str(helper/"fragment_gate.py"),"--config",manifest["config"],
        "--config-sha256",manifest["config_sha256"],"--inputs",manifest["inputs"],
        "--inputs-sha256",manifest["inputs_sha256"],"--system",manifest["system"],
        "--arm",arm,"--scaled","--block-size","24","--output",str(output)]
    if oracle:
        argv.extend(("--oracle-from",str(oracle),
            "--oracle-config",manifest["gpu_config"],
            "--oracle-config-sha256",manifest["gpu_config_sha256"]))
    return ["/bin/bash","-c",
        'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
        "split-fsapt",*argv]


def admit_cpu_producer(manifest,config):
    peer,_=read_config(Path(manifest["gpu_config"]),manifest["gpu_config_sha256"])
    require_oracle_config(config,peer)
    producer=Path(manifest["gpu_job_root"]).resolve(strict=True)
    return sealed_gpu(producer,manifest,manifest["gpu_config_sha256"],manifest["repeat"])


def run(manifest,root):
    root=Path(root).resolve(strict=True)
    result={"status":"STARTED","scope":"SEPARATED_EXACT_FSAPT_DIAGNOSTIC_NOT_SPEEDUP",
        "kind":manifest["kind"],"system":manifest["system"],"repeat":manifest.get("repeat"),
        "helper_pin":os.environ["DISP0_PIN"],"processes":[],
        "input_manifest_sha256":manifest["inputs_sha256"],
        "runtime_config_sha256":manifest["config_sha256"],
        "accepted_timing_count":0,"accepted_speedup_count":0}
    checkpoint=lambda:save(root/"split-result.json",result)
    checkpoint()
    helper=Path(__file__).resolve().parent
    case_cap=manifest.get("case_wall_seconds",28800)
    if not 180<case_cap<=28800:raise ValueError("Bounded per-case execution cap required")
    deadline=time.monotonic()+case_cap-180
    scratch_owner=None
    live_groups=set()
    try:
        scratch=Path(os.environ["TMPDIR"]).resolve(strict=True)
        scratch_owner=OwnedScratch(scratch,os.environ["DISP0_SPLIT_SCRATCH_CREATED"],root)
        raw=subprocess.check_output(["scontrol","show","job","-o",os.environ["SLURM_JOB_ID"]],
                                    text=True,timeout=30)
        validate_allocation(fields(raw),manifest)
        result["scheduler_job"]=raw
        threads=8 if manifest["kind"]=="gpu" else 24
        if len(os.sched_getaffinity(0))<threads:
            raise RuntimeError("Insufficient actual CPU affinity")
        config,build=read_config(Path(manifest["config"]),manifest["config_sha256"])
        if config["threads"]!=threads or config["host_memory"]!="96 GiB":
            raise RuntimeError("Declared per-role thread/memory policy changed")
        require_release(Path(manifest["release"]),manifest["release_sha256"],config)
        _,dimensions=load_point(Path(manifest["inputs"]),manifest["inputs_sha256"],manifest["system"])
        if manifest["kind"]=="cpu":
            # DiskDFJK must not spill to shared Lustre as in the mixed job.
            mount=fields(subprocess.check_output(["findmnt","-T",str(scratch),"-n","-P",
                                                 "-o","SOURCE,FSTYPE,TARGET"],text=True,timeout=30))
            if mount.get("FSTYPE") not in ("xfs","ext4","ext3","btrfs"):
                raise RuntimeError("CPU references require actual node-local disk scratch")
            stat=os.statvfs(scratch)
            if stat.f_bavail*stat.f_frsize<256*1024**3:
                raise RuntimeError("Insufficient node-local CPU scratch headroom")
            result["scratch_mount"]=mount
        result["source_pin"]=config["source_pin"]
        checkpoint()

        def child(label,arm,oracle=None):
            output=root/label
            work=scratch/label;work.mkdir()
            env=environment(config,gpu=arm=="gpu")
            env.update(TMPDIR=str(work),SCRATCH=str(work),PSI_SCRATCH=str(work))
            argv=worker_command(helper,config,manifest,output,arm,oracle)
            row={"label":label,"arm":arm,"argv":argv}
            result["processes"].append(row);checkpoint()
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError("Eight-hour split-job cap exhausted")
            start=time.perf_counter()
            with (root/(label+".stdout")).open("x") as out,(root/(label+".stderr")).open("x") as err:
                try:
                    def launched(pid):
                        live_groups.add(pid)
                        row["pid"]=pid;checkpoint()
                    def reaped(pid):
                        live_groups.discard(pid)
                        row["owned_group_reaped"]=True
                    code=execute_owned(argv,env,out,err,remaining,launched,reaped=reaped)
                except BaseException as error:
                    row.update(status="FAILED",error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    row["subprocess_seconds"]=time.perf_counter()-start;checkpoint()
            row["returncode"]=code;checkpoint()
            if code:raise RuntimeError("Split scientific process failed: "+label)
            doc=json.loads((output/"result.json").read_text())
            if (doc["status"]!="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                    or doc["arm"]!=arm or doc["system"]!=manifest["system"]
                    or doc["runtime_config_sha256"]!=manifest["config_sha256"]
                    or doc["input_manifest_sha256"]!=manifest["inputs_sha256"]):
                raise RuntimeError("Incomplete/mismatched split child")
            row["result_sha256"]=digest(output/"result.json");checkpoint()
            return output,doc

        if manifest["kind"]=="gpu":
            outputs=[];times=[];baseline=None
            for repeat in range(3):
                path,doc=child(f"r{repeat}-gpu","gpu")
                row={"repeat":repeat,"result_sha256":digest(path/"result.json")}
                if baseline:
                    row["repeat_check"]=compare_repeat(baseline,path,dimensions=dimensions)
                else:baseline=path
                outputs.append(row);times.append(doc["diagnostic_whole_energy_seconds"])
                result["gpu_outputs"]=outputs;checkpoint()
            result.update(status="PASS_GPU_EXECUTION_REPEATS_AWAITING_CPU",
                diagnostic_elapsed={"n":3,"median_seconds":statistics.median(times),
                                    "minimum_seconds":min(times),"maximum_seconds":max(times)},
                independent_cpu_validation_complete=False)
        else:
            repeat=manifest["repeat"]
            gpu_path,gpu=admit_cpu_producer(manifest,config)
            cpu_path,cpu=child(f"r{repeat}-cpu","cpu")
            oracle_path,oracle=child(f"r{repeat}-oracle","cpu",gpu_path)
            for key in ("retained_orbital_rank","virtual_orbitals"):
                if cpu[key]!=gpu[key]:
                    raise RuntimeError("Independent CPU/GPU retained orbital spaces differ")
            cpu_options,gpu_options=dict(cpu["options"]),dict(gpu["options"])
            if cpu_options.pop("use_cuest") or not gpu_options.pop("use_cuest"):
                raise RuntimeError("Split CPU/GPU native engine selection differs")
            for selected in (cpu_options,gpu_options):selected.pop("fisapt_fsapt_filepath")
            if cpu_options!=gpu_options:
                raise RuntimeError("Split reference and GPU scientific options differ")
            # Science identical; thread policy is deliberately recorded as
            # 24-core CPU node versus8-core hybrid/H200, not matched-thread timing.
            result.update(whole_energy_check=compare_energy(cpu,gpu),
                signed_checks=compare(cpu,gpu),
                same_gauge_fields=oracle["same_gauge_compared_fields"],
                same_gauge_errors=oracle["same_gauge_errors"],
                gpu_result_sha256=digest(gpu_path/"result.json"),
                oracle_result_sha256=digest(oracle_path/"result.json"),
                thread_policy={"cpu":24,"gpu_host":8},
                status="PASS_CPU_REFERENCE_AND_INDEPENDENT_ORACLE")
    except BaseException as error:
        result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if scratch_owner is not None:
                result["scratch_cleanup"]=scratch_owner.cleanup(live_groups)
        except BaseException as error:
            result.update(status="FAILED",scratch_cleanup={"complete":False,
                          "error":f"{type(error).__name__}: {error}"})
            raise
        finally:checkpoint()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve(strict=True)
    run(json.loads((root.parent/"manifest.json").read_text()),root)


if __name__=="__main__":
    from owned_process import termination_signals
    with termination_signals():main()
