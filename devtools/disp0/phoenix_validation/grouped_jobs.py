"""One GPU-only allocation; small CPU cases grouped, proteins separate."""
import argparse
import json
import os
from pathlib import Path
import time

from full_inputs import load_point
from gpu_runtime import digest
from repeat_gauge import compare_repeat
from run_full import save
from split_jobs import run
from split_scratch import OwnedScratch
from owned_process import termination_signals


def entries(kind,systems):
    if kind not in ("gpu","cpu") or not systems or len(set(systems))!=len(systems):
        raise ValueError("Explicit distinct grouped cases required")
    return [(system,repeat) for system in systems
            for repeat in ((None,) if kind=="gpu" else range(3))]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve(strict=True)
    manifest=json.loads((root.parent/"manifest.json").read_text())
    wall=manifest["allocation_wall_seconds"]
    if wall!=(57600 if manifest["kind"]=="gpu" else 86400):
        raise RuntimeError("Declared16h GPU /24h CPU grouped caps required")
    parent=Path(os.environ["TMPDIR"]).resolve(strict=True)
    owner=OwnedScratch(parent,os.environ["DISP0_SPLIT_SCRATCH_CREATED"],root)
    deadline=time.monotonic()+wall-180
    result={"status":"STARTED","kind":manifest["kind"],"systems":manifest["systems"],
            "helper_pin":manifest["helper_pin"],"source_pin":manifest["science_pin"],
            "cases":[],"accepted_timing_count":0,"accepted_speedup_count":0}
    safe=True
    failed_systems=set()
    checkpoint=lambda:save(root/"group-result.json",result)
    checkpoint()
    try:
        for system,repeat in entries(manifest["kind"],manifest["systems"]):
            if system in failed_systems:continue
            remaining=deadline-time.monotonic()
            if remaining<=181:raise TimeoutError("Grouped allocation deadline exhausted")
            name=system if repeat is None else f"{system}-r{repeat}"
            output=root/name;output.mkdir()
            scratch=parent/name;scratch.mkdir(mode=0o700)
            selected={**manifest,"system":system,"repeat":repeat,
                      "case_wall_seconds":min(28800,int(remaining))}
            if manifest["kind"]=="cpu":
                selected["gpu_job_root"]=str(Path(manifest["gpu_group_job_root"])/system)
            save(output/"case-manifest.json",selected)
            row={"system":system,"repeat":repeat,"path":str(output)}
            result["cases"].append(row);checkpoint()
            os.environ["TMPDIR"]=os.environ["PSI_SCRATCH"]=os.environ["SCRATCH"]=str(scratch)
            os.environ["DISP0_SPLIT_SCRATCH_CREATED"]=str(scratch)
            error=None
            try:
                # Same-process orchestration, not a nested child supervisor:
                # run() owns and reaps each fresh scientific worker directly.
                run(selected,output)
                row["status"]="PASS"
                with (output/"job-status.env").open("x") as stream:
                    stream.write("exit_code=0\nterm_received=0\n")
                row["result_sha256"]=digest(output/"split-result.json")
            except BaseException as caught:
                error=caught
                row.update(status="FAILED",error=f"{type(caught).__name__}: {caught}")
                failed_systems.add(system)
                document=json.loads((output/"split-result.json").read_text()) if (
                    output/"split-result.json").exists() else {}
                safe=document.get("scratch_cleanup",{}).get("complete") is True
            finally:
                os.environ["TMPDIR"]=os.environ["PSI_SCRATCH"]=os.environ["SCRATCH"]=str(parent)
                os.environ["DISP0_SPLIT_SCRATCH_CREATED"]=str(parent)
                checkpoint()
            if error is not None and (isinstance(error,(InterruptedError,KeyboardInterrupt)) or not safe):
                raise error
            # A failed case is not retried. Distinct GPU systems may still use
            # their already-approved part of the one allocation.
        if manifest["kind"]=="cpu":
            for system in manifest["systems"]:
                if system in failed_systems:continue
                _,dimensions=load_point(Path(manifest["inputs"]),manifest["inputs_sha256"],system)
                baseline=root/f"{system}-r0/r0-cpu"
                for repeat in (1,2):
                    check=compare_repeat(baseline,root/f"{system}-r{repeat}/r{repeat}-cpu",
                                         dimensions=dimensions,arm="cpu")
                    result.setdefault("CPU_repeat_checks",[]).append(
                        {"system":system,"repeat":repeat,"check":check})
        if failed_systems:
            result.update(status="FAILED_SOME_CASES",failed_systems=sorted(failed_systems))
            raise RuntimeError("Grouped job has failed cases; completed outputs are preserved")
        result["status"]=("PASS_GPU_GROUP_AWAITING_CPU" if manifest["kind"]=="gpu"
                          else "PASS_CPU_GROUP_REFERENCES_AND_ORACLES")
    except BaseException as error:
        result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            result["scratch_cleanup"]=owner.cleanup(set() if safe else {"unproved-case-cleanup"})
        except BaseException as error:
            result.update(status="FAILED",scratch_cleanup={"complete":False,
                          "error":f"{type(error).__name__}: {error}"})
            raise
        finally:checkpoint()


if __name__=="__main__":
    with termination_signals():main()
