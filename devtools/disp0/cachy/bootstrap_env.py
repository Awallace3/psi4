"""Create NEW private Cachy dependency prefixes; never touch existing envs.

Run from the pinned source after Git staging. No GPU computation or library
import, global installation, retry, source overlay or scientific build.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(command, folder, name, env, timeout):
    record={"argv":command,"started_unix":time.time(),"timeout_seconds":timeout}
    (folder/f"{name}.intent.json").write_text(json.dumps(record,indent=2)+"\n")
    with (folder/f"{name}.stdout").open("xb") as out,(folder/f"{name}.stderr").open("xb") as err:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                 env=env,start_new_session=True)
        interrupted=[]
        deadline=time.monotonic()+timeout
        def terminate(signum,frame):
            interrupted.append(signum)
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        watched=(signal.SIGTERM,signal.SIGINT,signal.SIGHUP)
        previous={sig:signal.signal(sig,terminate) for sig in watched}
        try:
            while process.poll() is None:
                if interrupted or time.monotonic()>=deadline:
                    raise subprocess.TimeoutExpired(command,timeout)
                try:process.wait(timeout=min(0.5,max(0.01,deadline-time.monotonic())))
                except subprocess.TimeoutExpired:pass
            code=process.returncode
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=10)
            code=124
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
        if interrupted:code=128+interrupted[0]
        out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
    record.update(exit=code,finished_unix=time.time())
    (folder/f"{name}.result.json").write_text(json.dumps(record,indent=2)+"\n")
    print(name,"exit",code,flush=True)
    if code:
        for suffix in ("stdout","stderr"):
            with (folder/f"{name}.{suffix}").open("rb") as stream:
                stream.seek(0,2);stream.seek(max(0,stream.tell()-6000))
                print(stream.read().decode(errors="replace"),flush=True)
        raise RuntimeError(f"{name} failed; preserve attempt, no automatic retry")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--base-env",type=Path,required=True)
    parser.add_argument("--conda",type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve(strict=True);source=root/"source";metadata=root/"metadata"
    pin=(metadata/"source-pin.txt").read_text().strip()
    head=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    status=subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True)
    if head!=pin or status.strip():
        raise RuntimeError("Source pin/cleanliness differs")
    runtime,clang=root/"environment",root/"clang23"
    if runtime.exists() or clang.exists():
        raise RuntimeError("New absent environment roots required; do not adopt failed/shared prefixes")
    logs=metadata/"bootstrap-01";logs.mkdir(mode=0o700)
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    env=dict(os.environ)
    for key in list(env):
        if key.startswith(("CONDA_","MAMBA_")) or key in (
                "PYTHONPATH","PYTHONHOME","LD_PRELOAD","LD_LIBRARY_PATH","CPATH","GCC_ROOT","GCC_EXEC_PREFIX",
                "C_INCLUDE_PATH","CPLUS_INCLUDE_PATH","OBJC_INCLUDE_PATH","LIBRARY_PATH",
                "CUDA_HOME","CUDA_PATH","CUDA_ROOT","CUDACXX","CMAKE_PREFIX_PATH",
                "CC","CXX","FC","CFLAGS","CXXFLAGS","FFLAGS","LDFLAGS","CPPFLAGS"):
            env.pop(key,None)
    env.update(CONDA_PKGS_DIRS=str(root/"conda-pkgs"),
               CONDARC=str(source/"devtools/disp0/condarc"),
               OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
               MAMBA_NUM_THREADS="2",MAMBA_EXTRACT_THREADS="2",
               PATH=str(args.conda.parent)+":/usr/bin:/bin",
               PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
               PIP_CACHE_DIR=str(root/"pip-cache"))
    phase="source-version"
    status_file=metadata/"SETUP_STATUS.md"
    status_file.write_text("# Cachy setup\n\nPrivate dependency bootstrap IN PROGRESS.\n"
                           "No compiled scientific/GPU readiness is claimed.\n")
    try:
        run([sys.executable,"-B",str(source/"devtools/disp0/optimized_build/source_version_guard.py"),
             "--source",str(source),"--out",str(metadata/"source-version-proof")],
            logs,phase,env,45)
        phase="baseline-record"
        run([str(args.conda),"list","--explicit","--prefix",str(args.base_env)],
            logs,phase,env,60)
        raw=(logs/f"{phase}.stdout").read_text().splitlines()
        excluded=("psi4-","einsums-","pyeinsums-")
        filtered=[]
        for line in raw:
            if line.startswith("#") or not line.strip() or line=="@EXPLICIT":
                filtered.append(line);continue
            name=line.split("/")[-1]
            if not name.startswith(excluded):filtered.append(line)
        lock=metadata/"baseline-filtered-explicit.txt";lock.write_text("\n".join(filtered)+"\n")
        phase="baseline-create"
        run([str(args.conda),"create","-y","--prefix",str(runtime),"--file",str(lock)],
            logs,phase,env,1800)
        specs=["python=3.13","gcc_linux-64=14.3.0","gxx_linux-64=14.3.0",
               "gfortran_linux-64=14.3.0","cuda-version=13.3","cuda-toolkit=13.3",
               "cuda-nvcc=13.3.73","cuda-bindings>=13,<14"]
        command=[str(args.conda),"install","-y","--prefix",str(runtime),
                 "--override-channels","-c","conda-forge","--strict-channel-priority",*specs]
        phase="cuda-dry-run"
        run([*command,"--dry-run","--json"],logs,phase,env,600)
        proposal=json.loads((logs/f"{phase}.stdout").read_text())
        if not proposal.get("success"):
            raise RuntimeError("CUDA dependency solver did not provide a successful proposal")
        phase="cuda-install"
        run(command,logs,phase,env,2400)
        phase="clang23-create"
        run([str(args.conda),"create","-y","--prefix",str(clang),"--override-channels",
             "-c","conda-forge","--strict-channel-priority",
             "clangdev=23.1.2","clangxx=23.1.2","llvmdev=23.1.2"],
            logs,phase,env,1800)
        for label,prefix in (("runtime",runtime),("clang23",clang)):
            phase=f"{label}-explicit"
            run([str(args.conda),"list","--explicit","--prefix",str(prefix)],
                logs,phase,env,60)
        phase="compiler-version"
        run([str(runtime/"bin/x86_64-conda-linux-gnu-c++"),"-dumpfullversion"],
            logs,phase,env,30)
        if (logs/f"{phase}.stdout").read_text().strip()!="14.3.0":
            raise RuntimeError("Actual compiler is not GNU14.3.0")
        phase="nvcc-version"
        run([str(runtime/"bin/nvcc"),"--version"],logs,phase,env,30)
        if "V13.3.73" not in (logs/f"{phase}.stdout").read_text():
            raise RuntimeError("Actual NVCC is not the pinned13.3.73")
        phase="python-abi"
        run([str(runtime/"bin/python"),"-B","-c",
             "import sys,sysconfig; assert sys.version_info[:2]==(3,13); "
             "assert not sysconfig.get_config_var('Py_GIL_DISABLED'); print(sys.version)"],
            logs,phase,env,30)
        status_file.write_text("# Cachy setup\n\nPrivate runtime/LLVM dependency environments created.\n"
            "SDK/sources/build receipts are separate. GPU LibXC/E2/Psi4 compilation,\n"
            "native imports, handle/plan factory and GPU numerical gate remain pending.\n"
            "No global environment or desktop GPU process was changed.\n")
    except BaseException as error:
        status_file.write_text("# Cachy setup\n\n**Dependency bootstrap FAILED/INCOMPLETE** at "
                              f"`{phase}`: {type(error).__name__}: {error}\n\n"
                              "Preserve logs/prefixes. No automatic retry/fallback/toolkit downgrade.\n"
                              "No compiled scientific or GPU readiness is established.\n")
        raise


if __name__=="__main__":
    main()
