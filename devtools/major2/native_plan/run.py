"""Allocation controller: isolated export, owned diagnostic compile, exactly 3 native processes."""
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import common as n


def native_maps(text, pins):
    paths={line.split()[-1] for line in text.splitlines() if line.split()[-1].startswith("/")}
    n.require(not any("libpsi4" in p or "/psi4/core." in p or "libpython" in p or
                      "libEinsums" in p for p in paths),"native process must not load Psi4/Python/Einsums")
    found={}
    prefixes=("libcuest.so","libcudart.so","libstdc++.so","libcublas","libcusolver",
              "libcusparse","libnvJitLink","libnvrtc")
    for path in paths:
        if Path(path).name.startswith(prefixes):
            n.require(Path(path).is_file(),"missing mapped runtime: "+path)
            resolved=str(Path(path).resolve(strict=True))
            n.require(resolved in pins and n.sha(resolved)==pins[resolved],"native runtime mapping: "+path)
            found[resolved]=pins[resolved]
    n.require(all(any(Path(p).name.startswith(k) for p in found) for k in
                  ("libcuest.so","libcudart.so","libstdc++.so","libcublas.so","libcusolver.so")),
              "required native library mapping")
    return found


def link_command(exe):
    # No -lpsi4, no stage linkage, no Python, no CMake/Psi4/SDK build.
    return ["bash",str(n.HERE.parent/"build.sh"),n.CXX,"-std=c++17","-O0",
        "-I"+n.CUDA+"/targets/x86_64-linux/include","-I"+n.SDK+"/include",
        str(n.HERE/"native.cc"),n.SDK+"/lib/13/libcuest.so",
        n.E+"/lib/libcudart.so",n.E+"/lib/libcublas.so",n.E+"/lib/libcusolver.so",
        "-Wl,-rpath,"+n.SDK+"/lib/13:"+n.E+"/lib","-o",str(exe)]


class Runner:
    def __init__(self,m,path,mh):
        self.m=m; self.path=Path(path); self.mh=mh
        self.root=Path(n.RUN); self.raw=self.root/"raw"
        self.scratch=Path(n.SCRATCH)/os.environ["SLURM_JOB_ID"]
        self.active=None; self.signalled=False
        self.end=min(int(os.environ["SLURM_JOB_END_TIME"]),int(os.environ["SLURM_JOB_START_TIME"])+14400)

    def stop(self,*_):
        self.signalled=True
        raise RuntimeError("TERM/INT/HUP: no success")

    def command(self,argv,name,timeout,env=None):
        n.require(not self.signalled,"signal latched")
        remaining=self.end-time.time()-300
        n.require(remaining>=timeout,"admission budget exhausted")
        n.durable(self.raw/(name+".argv.json"),dict(argv=list(map(str,argv)),cwd=str(self.scratch)))
        with (self.raw/(name+".log")).open("xb") as f:
            p=subprocess.Popen(list(map(str,argv)),cwd=self.scratch,env=env or self.env,
                               stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            self.active=p
            try:
                code=p.wait(timeout=timeout)
            except BaseException:
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL); p.wait()
                n.durable(self.raw/(name+".exit.json"),dict(exit=p.returncode,interrupted=True))
                raise
            finally:
                self.active=None
        n.durable(self.raw/(name+".exit.json"),dict(exit=code,interrupted=False,pid=p.pid))
        n.require(code==0 and not self.signalled,"command failed: "+name)
        n.require((self.raw/(name+".log")).stat().st_size<8*1024**2,"bounded diagnostic log")
        return p.pid

    def verify_native(self,directory,pid,uuid):
        v=n.load(directory/"native-complete.json")
        n.require(v==dict(outputs=6,plans=3,handles=1,upstream_builds=2,sdk="0.2.50",
                  cudart=13030,pid=pid,uuid=uuid,job=os.environ["SLURM_JOB_ID"]),"native lifecycle receipt")
        n.require(n.load(directory/"effective-env.json")==dict(numerical_overrides={},pid=pid),
                  "pre-device native ENV receipt")
        schedule=(directory/"schedule.txt").read_text().splitlines()
        sizes=[s.split() for s in schedule if s.startswith("workspace ")]
        n.require(len(sizes)==3 and all(len(s)==5 and all(x.isdigit() for x in s[1:]) for s in sizes),
                  "native workspace sizing receipts")
        schedule=[s for s in schedule if not s.startswith("workspace ")]
        n.require(schedule==["up+","A+","A1","A2","plan-","B+","B1","B2","plan-",
                            "up-","up+","C+","C1","C2","plan-","up-"],"native schedule counts/order")
        n.require(set(p.name for p in directory.iterdir())==
                  {s+".bin" for s in n.STATES} |
                  {"schedule.txt","native-complete.json","effective-env.json","maps.txt"},"native artifacts")
        maps=native_maps((directory/"maps.txt").read_text(),self.m["native_libraries"])
        n.durable(self.raw/(directory.name+"-verified.json"),dict(pid=pid,uuid=uuid,maps=maps,
            inputs={k:n.sha(self.raw/"input"/k) for k in ("basis.bin","D.bin")}))

    def execute(self):
        n.allocation_only()
        n.source_check(self.m)
        # New allocation attempt is irreversible; no retry within this bundle.
        n.durable(self.root/"STARTED.json",dict(job=os.environ["SLURM_JOB_ID"],pin=self.m["pin"]))
        self.raw.mkdir(); self.scratch.mkdir(parents=True,exist_ok=False)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,self.stop)
        sys.path.insert(0,str(n.HERE.parent))
        import policy
        import campaign
        self.env=policy.runtime_env(os.environ,self.scratch)
        n.check_env(self.env)
        self.env["CUDA_CACHE_PATH"]=str(self.scratch/"cuda-cache")
        Path(self.env["CUDA_CACHE_PATH"]).mkdir()
        scheduler=n.output("scontrol","show","job","-o",os.environ["SLURM_JOB_ID"])
        campaign.allocation(scheduler,self.env)
        n.durable(self.raw/"scheduler.txt",scheduler+"\n")
        n.durable(self.raw/"environment.json",{k:v for k,v in self.env.items() if k.startswith(
            ("SLURM_","CUDA_","OMP_","MKL_","OPENBLAS_")) or k in
            ("LD_LIBRARY_PATH","PATH","TMPDIR","SCRATCH","PSI_SCRATCH")})
        # Heavy hashes only on the allocated host.
        allpins=dict(self.m["headers"],**self.m["native_libraries"])
        allpins.update({self.m["compiler"]["wrapper"]:self.m["compiler"]["wrapper_sha256"],
            n.CXX_REAL:n.CXX_REAL_SHA,n.CC1PLUS:n.CC1PLUS_SHA,
            self.m["identity"]["path"]:self.m["identity"]["sha256"],
            self.m["runtime_file"]:self.m["runtime_file_sha256"]})
        n.require(str(Path(n.CXX_REAL).resolve(strict=True))==self.m["compiler"]["actual_resolved"],
                  "compiler binary alias resolution")
        for path,h in allpins.items(): n.require(n.sha(path)==h,"allocated input identity: "+path)
        n.durable(self.raw/"input-identities.json",allpins)
        self.command(["bash",n.HERE.parent/"build.sh",n.CXX,"--version"],"compiler-version",30)
        self.command(["bash",n.HERE.parent/"build.sh",n.CXX,"-print-prog-name=cc1plus"],"compiler-frontend",30)
        n.require(n.CC1PLUS==(self.raw/"compiler-frontend.log").read_text().splitlines()[-1],"resolved frontend")
        n.require("14.3.0" in (self.raw/"compiler-version.log").read_text(),"actual compiler version")
        self.command([self.m["identity"]["path"]],"gpu-identity",30)
        proof=n.load(self.raw/"gpu-identity.log")
        n.require(proof["count"]==1 and proof["logical_device"]==0 and
                  proof["header_version"]==proof["runtime_version"]==13030 and
                  "H200" in proof["name"] and proof["memory_bytes"]>=130000*1048576,"CUDA13/H200")
        uuid=proof["uuid"]
        n.require(re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",uuid),"UUID")
        for selector in (self.env.get("CUDA_VISIBLE_DEVICES",""),self.env.get("SLURM_JOB_GPUS","")):
            n.require(re.fullmatch(r"(?:\d+|GPU-[0-9a-fA-F-]+)",selector),"one scheduler/visible GPU")
            if selector.startswith("GPU-"): n.require(selector.lower()==uuid,"GPU selector")
        gpu=campaign.gpu_line(n.output("nvidia-smi","--id="+uuid,
            "--query-gpu=name,uuid,memory.total,driver_version,index","--format=csv,noheader,nounits"))
        n.require(gpu["uuid"]==uuid,"NVML UUID")
        n.durable(self.raw/"gpu.json",dict(cuda=proof,nvml=gpu))
        inp=self.raw/"input"; inp.mkdir()
        self.command(["bash",n.HERE.parent/"arm.sh",n.STAGE,self.m["runtime_file"],
            n.E+"/bin/python","-B",n.HERE/"export.py",self.path,self.mh,inp],"basis-export",600)
        inputs={k:n.sha(inp/k) for k in ("basis.bin","D.bin")}
        n.require(inputs["D.bin"]==n.D_SHA and n.load(inp/"INPUT-COMPLETE.json")["density_sha256"]==n.D_SHA,
                  "exact readonly original density")
        exe=self.scratch/"native-plan"
        self.command(link_command(exe),"compile-owned-native",120)
        self.command(["readelf","-d",exe],"native-elf-dynamic",30)
        dynamic=(self.raw/"native-elf-dynamic.log").read_text()
        n.require(all(s not in dynamic.lower() for s in ("libpsi4","libpython","einsums")) and
                  all(s in dynamic for s in ("libcuest.so","libcudart.so.13","libcublas.so","libcusolver.so")),
                  "native-only ELF link set")
        n.durable(self.raw/"compiled.json",dict(elf=str(exe),elf_sha256=n.sha(exe),
            source={p:n.sha(n.HERE/p) for p in ("native.cc","input.h")},compile_job=os.environ["SLURM_JOB_ID"]))
        pids=[]
        for index in range(1,4):
            n.check_env(self.env); n.source_check(self.m)
            n.require({k:n.sha(inp/k) for k in inputs}==inputs,"input mutation")
            n.require(not n.output("nvidia-smi","--id="+uuid,"--query-compute-apps=gpu_uuid,pid",
                                  "--format=csv,noheader,nounits"),"GPU already held before native process")
            directory=self.raw/f"p{index}"; directory.mkdir()
            pid=self.command([exe,inp/"basis.bin",inp/"D.bin",directory,uuid],f"native-p{index}",1800)
            pids.append(pid); self.verify_native(directory,pid,uuid)
        n.require(len(set(pids))==3,"three fresh native processes")
        # At most 153 matrix pairs, no historical output read, no timings.
        import compare
        def deadline(*_): raise RuntimeError("analysis budget")
        signal.signal(signal.SIGALRM,deadline); signal.alarm(1200)
        try: report=compare.analyze(self.raw,inp/"D.bin")
        finally: signal.alarm(0)
        n.source_check(self.m)
        n.require({k:n.sha(inp/k) for k in inputs}==inputs,"final input mutation")
        for path,h in allpins.items(): n.require(n.sha(path)==h,"final shared identity")
        # Check preservation once more without touching any old matrix bytes.
        import importlib.util
        spec=importlib.util.spec_from_file_location("native_handoff",n.HERE/"handoff.py")
        handoff=importlib.util.module_from_spec(spec); spec.loader.exec_module(handoff)
        handoff.preserve(self.m["preservation"])
        n.durable(self.root/"NATIVE-DIAGNOSTIC.json",report)
        n.durable(self.root/"ENDPOINT-COMPLETE.json",dict(processes=3,outputs=18,pairs=153,
                  accepted_speedup_data=False,compiled_diagnostic_only=True))


def main():
    n.allocation_only()
    path,mh=sys.argv[1:]
    m=n.manifest(path,mh)
    Runner(m,path,mh).execute()


if __name__=="__main__": main()
