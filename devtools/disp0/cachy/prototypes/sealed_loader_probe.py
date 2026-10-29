"""THROWAWAY: can sealed ELF snapshots preserve loader behavior?

This probes real kernel seals, $ORIGIN and exact cuEST/CUDA symbol ownership.
Default CPU-only; --gpu adds an explicitly approved compatibility gate.
It is not a state simulation, Psi4 overlay or production admission.
Run --toy-only locally; --root selects the existing private Cachy inputs.
"""
import argparse
import ast
import ctypes as ct
import errno
import fcntl
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import shutil
import subprocess
import sys

REFERENCE="3ea43ee66ebb2920138c9dfc89c1951d6fbb3673"
CONFIG_SHA="4f3fbda55b461b75550094466d03392570270afc57ee620f483a915a29ea8ef1"
# Conda's Python may be built against headers lacking these Python attributes.
# Linux UAPI values verified against Cachy's6.18 linux/fcntl.h and memfd.h.
F_ADD_SEALS=getattr(fcntl,"F_ADD_SEALS",1033)
F_GET_SEALS=getattr(fcntl,"F_GET_SEALS",1034)
REQUIRED=0x0008|0x0004|0x0002|0x0001
# Linux UAPI since6.3. Unsupported kernels fail, never silently downgrade.
EXEC_FLAG=getattr(os,"MFD_EXEC",0x0010)
EXEC_SEAL=getattr(fcntl,"F_SEAL_EXEC",0x0020)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(stage,**values):
    print("SEALED_PROTOTYPE",json.dumps({"stage":stage,**values},sort_keys=True),flush=True)


def fingerprint(s):
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,
            s.st_mode,s.st_uid,s.st_nlink)


def create_memfd(name):
    flags=0x0002|0x0001|EXEC_FLAG  # ALLOW_SEALING|CLOEXEC|EXEC.
    if hasattr(os,"memfd_create"):
        return os.memfd_create(name,flags)
    libc=ct.CDLL(None,use_errno=True)
    create=libc.memfd_create
    create.argtypes,create.restype=[ct.c_char_p,ct.c_uint],ct.c_int
    fd=create(name.encode(),flags)
    if fd<0:raise OSError(ct.get_errno(),"libc memfd_create")
    return fd


def snapshot(item,aliases):
    source=Path(item["source"])
    fd=create_memfd("sealed-prototype-"+item["name"])
    checksum=hashlib.sha256()
    with source.open("rb") as stream:
        before=os.fstat(stream.fileno())
        while block:=stream.read(1024**2):
            checksum.update(block)
            assert os.write(fd,block)==len(block)
        assert fingerprint(before)==fingerprint(os.fstat(stream.fileno()))
    assert checksum.hexdigest()==item["sha256"]
    assert os.fstat(fd).st_size==item["bytes"]
    # Keep write permission: rejection below must come from seals, not chmod.
    os.fchmod(fd,0o700)
    fcntl.fcntl(fd,F_ADD_SEALS,REQUIRED|EXEC_SEAL)
    seals=fcntl.fcntl(fd,F_GET_SEALS)
    assert seals&REQUIRED==REQUIRED and not os.get_inheritable(fd)
    os.lseek(fd,0,os.SEEK_SET)
    after=hashlib.sha256()
    while block:=os.read(fd,1024**2):after.update(block)
    assert after.hexdigest()==item["sha256"]
    names=[item["name"]]
    if item["name"].startswith("libcuest.so."):
        names.append("libcuest.so.0")
    for name in names:
        (aliases/name).symlink_to(f"/proc/{os.getpid()}/fd/{fd}")
    info=os.fstat(fd)
    record={**item,"fd":fd,"seals":seals,"inode":info.st_ino,
            "device":[os.major(info.st_dev),os.minor(info.st_dev)],
            "kernel_path":os.readlink(f"/proc/self/fd/{fd}")}
    state("sealed",name=item["name"],fd=fd,seals=seals,sha256=item["sha256"])
    return record


def cannot_write(record,alias):
    fd,size=record["fd"],record["bytes"]
    actions=(lambda:os.pwrite(fd,b"X",0),lambda:os.ftruncate(fd,size+1),
             lambda:os.ftruncate(fd,size-1),
             lambda:mmap.mmap(fd,0,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE),
             lambda:fcntl.fcntl(fd,F_ADD_SEALS,0))
    for action in actions:
        try:action()
        except OSError as error:assert error.errno==errno.EPERM,error
        else:raise AssertionError("Kernel allowed modification of sealed data")
    reopened=os.open(alias,os.O_RDWR)
    try:
        try:os.pwrite(reopened,b"X",0)
        except OSError as error:assert error.errno==errno.EPERM,error
        else:raise AssertionError("Reopening bypassed the inode's seals")
    finally:os.close(reopened)


def symbol_owner(function,record):
    address=ct.cast(function,ct.c_void_p).value
    found=[]
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6:continue
        lo,hi=(int(x,16) for x in fields[0].split("-"))
        if lo<=address<hi:
            found.append({"kernel_path":fields[5],"inode":int(fields[4]),
                          "device":[int(x,16) for x in fields[3].split(":")]})
    expected={k:record[k] for k in ("kernel_path","inode","device")}
    assert found==[expected],(found,expected)
    assert record["kernel_path"].startswith("/memfd:") and record["kernel_path"].endswith(" (deleted)")
    return expected


def compile_ptx_cpu(native,record,ptx_output=None):
    """NVRTC is a CPU compiler. Disable its optional cuInit-backed cache."""
    assert os.environ.get("CUDA_CACHE_DISABLE")=="1"
    V,P,C,Z=ct.c_void_p,ct.POINTER,ct.c_char_p,ct.c_size_t
    specs={"nvrtcVersion":[P(ct.c_int),P(ct.c_int)],
        "nvrtcCreateProgram":[P(V),C,C,ct.c_int,P(C),P(C)],
        "nvrtcCompileProgram":[V,ct.c_int,P(C)],
        "nvrtcGetProgramLogSize":[V,P(Z)],"nvrtcGetProgramLog":[V,ct.c_void_p],
        "nvrtcGetPTXSize":[V,P(Z)],"nvrtcGetPTX":[V,ct.c_void_p],
        "nvrtcDestroyProgram":[P(V)]}
    for name,types in specs.items():
        fn=getattr(native,name);symbol_owner(fn,record)
        fn.argtypes,fn.restype=types,ct.c_int
    major,minor=ct.c_int(),ct.c_int()
    assert native.nvrtcVersion(ct.byref(major),ct.byref(minor))==0
    assert (major.value,minor.value)==(13,3)
    program=V()
    source=b'extern "C" __global__ void sealed_probe(double *x) { x[0] += 1.0; }'
    assert native.nvrtcCreateProgram(ct.byref(program),source,b"sealed_probe.cu",0,None,None)==0
    options=(C*4)(b"--std=c++14",b"--gpu-architecture=compute_86",b"--fmad=false",b"-no-cache")
    code=native.nvrtcCompileProgram(program,4,options)
    size=Z()
    assert native.nvrtcGetProgramLogSize(program,ct.byref(size))==0
    log=ct.create_string_buffer(size.value)
    assert native.nvrtcGetProgramLog(program,log)==0
    assert code==0,log.value.decode(errors="replace")
    assert native.nvrtcGetPTXSize(program,ct.byref(size))==0
    assert 0<size.value<1024**2
    ptx=ct.create_string_buffer(size.value)
    assert native.nvrtcGetPTX(program,ptx)==0
    assert native.nvrtcDestroyProgram(ct.byref(program))==0
    if ptx_output is not None:ptx_output.write_bytes(ptx.raw)
    return {"version":[major.value,minor.value],"ptx_bytes":len(ptx.raw),
            "ptx_sha256":hashlib.sha256(ptx.raw).hexdigest(),"explicit_compiler_calls":8,
            "cache_disabled":True,"kernel_loaded_or_executed":False}


def child(input_path):
    plan=json.loads(input_path.read_text())
    if plan.get("gpu"):
        from sealed_gpu_probe import gpu_policy
        gpu_policy(plan)
    else:assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    aliases=Path(plan["aliases"]);toy_aliases=Path(plan["toy_aliases"])
    assert os.environ["LD_LIBRARY_PATH"]==plan["startup_library_path"]
    records={}
    for item in plan["toy"]:
        records[item["name"]]=snapshot(item,toy_aliases)
    # Only load the parent. The child is neither preloaded nor on LD_LIBRARY_PATH:
    # its sealed snapshot must be found via the parent's real $ORIGIN RUNPATH.
    assert str(toy_aliases) not in os.environ["LD_LIBRARY_PATH"].split(":")
    parent=ct.CDLL(str(toy_aliases/"libsealed_parent.so"),mode=os.RTLD_NOW|os.RTLD_GLOBAL)
    parent.parent.argtypes=[];parent.parent.restype=ct.c_int
    assert parent.parent()==7
    symbol_owner(parent.parent,records["libsealed_parent.so"])
    symbol_owner(parent.child,records["libsealed_child.so"])
    for item in plan["toy"]:Path(item["source"]).unlink()
    assert parent.parent()==7
    state("origin_passed",parent_value=7,source_files_removed=True)
    for item in plan["vendor"]:
        records[item["name"]]=snapshot(item,aliases)
    loaded=[]
    for name in plan["preload_order"]:
        loaded.append(ct.CDLL(str(aliases/name),mode=os.RTLD_NOW|os.RTLD_GLOBAL))
        state("vendor_loaded",name=name,loaded_count=len(loaded))
    ownership={};compiler={}
    if plan["vendor"]:
        native_name="libcuest.so.0.2.50"
        native=loaded[plan["preload_order"].index(native_name)]
        # This helper only configures checked C signatures; no cuEST call occurs.
        spec=importlib.util.spec_from_file_location("_sealed_probe_identity",plan["identity_helper"])
        identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
        names=list(identity.configure_abi(native))+plan["factory_symbols"]
        ownership={name:symbol_owner(getattr(native,name),records[native_name]) for name in names}
        state("vendor_symbols_passed",symbols=len(ownership),explicit_vendor_calls=0)
        nvrtc=loaded[plan["preload_order"].index("libnvrtc.so.13")]
        ptx_output=input_path.parent/"probe.ptx" if plan.get("gpu") else None
        compiler=compile_ptx_cpu(nvrtc,records["libnvrtc.so.13"],ptx_output)
        state("nvrtc_cpu_compile_passed",**compiler)
    for name,record in records.items():
        cannot_write(record,(toy_aliases if name.startswith("libsealed_") else aliases)/name)
    assert "psi4" not in sys.modules and "cuda" not in sys.modules
    report={"status":"PASS_CPU_SEALED_LOADER_PROTOTYPE_ONLY",
        "scope":"THROWAWAY_NOT_GPU_OR_PSI4_QUALIFICATION",
        "explicit_cuda_driver_calls":0,"explicit_cuda_runtime_calls":0,
        "explicit_cuest_api_calls":0,"origin_parent_value":7,"source_files_removed":True,
        "nvrtc_cpu_compile":compiler,
        "memfd_provider":"os.memfd_create" if hasattr(os,"memfd_create") else "libc.memfd_create",
        "records":records,"preloaded":plan["preload_order"],"symbol_ownership":ownership,
        "driver_sha256":digest(__file__),"input_sha256":digest(input_path)}
    if plan.get("gpu"):
        from sealed_gpu_probe import probe
        report["gpu"]=probe(plan,records,loaded)
        report["status"]="PASS_GPU_SEALED_LOADER_PROTOTYPE_ONLY"
        report["scope"]="THROWAWAY_NOT_PSI4_NUMERICAL_OR_PERFORMANCE_QUALIFICATION"
        for field in ("explicit_cuda_driver_calls","explicit_cuda_runtime_calls",
                      "explicit_cuest_api_calls"):
            report.pop(field)
    Path(plan["result"]).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    state("passed",sealed_files=len(records),status=report["status"])
    # Raw FDs and CDLL references deliberately last until this worker exits.


def item(path):
    return {"name":path.name,"source":str(path),"sha256":digest(path),"bytes":path.stat().st_size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--toy-only",action="store_true")
    parser.add_argument("--gpu",action="store_true",help="Approved bounded compatibility probe only")
    parser.add_argument("--child",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.child is not None:
        child(args.child)
        return
    if not args.toy_only and args.root is None:
        parser.error("--root is required unless --toy-only is selected")
    assert not args.gpu or not args.toy_only
    if not args.gpu:assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    assert args.output is not None
    args.output=args.output.resolve()
    assert not args.output.exists()
    args.output.mkdir(mode=0o700)
    scratch=args.output/"scratch";scratch.mkdir()
    aliases=args.output/"vendor-aliases";aliases.mkdir(mode=0o700)
    toy_aliases=args.output/"toy-origin-aliases";toy_aliases.mkdir(mode=0o700)
    cc=shutil.which("cc");assert cc
    source=scratch/"child.c";source.write_text("int child(void) { return 7; }\n")
    child_lib=scratch/"libsealed_child.so"
    subprocess.run([cc,"-shared","-fPIC",str(source),"-Wl,-soname,libsealed_child.so",
                    "-o",str(child_lib)],check=True)
    source=scratch/"parent.c";source.write_text("extern int child(void); int parent(void) { return child(); }\n")
    parent_lib=scratch/"libsealed_parent.so"
    subprocess.run([cc,"-shared","-fPIC",str(source),"-L"+str(scratch),"-lsealed_child",
                    "-Wl,-rpath,$ORIGIN","-o",str(parent_lib)],check=True)
    plan={"toy":[item(child_lib),item(parent_lib)],"vendor":[],"preload_order":[],
          "reference_pin":REFERENCE,
          "aliases":str(aliases),"toy_aliases":str(toy_aliases),
          "startup_library_path":str(aliases),"result":str(args.output/"result.json")}
    if not args.toy_only:
        root=args.root.resolve(strict=True)
        assert root==Path.home()/"gits/psi4-disp0-perf-cachy"
        assert subprocess.check_output(["git","-C",str(root/"source"),"rev-parse","HEAD"],
                                       text=True).strip()==REFERENCE
        assert not subprocess.check_output(["git","-C",str(root/"source"),"status","--porcelain"],
                                           text=True).strip()
        config_path=root/"runs/hot-owned-cpu-01/runtime-config.json"
        assert digest(config_path)==CONFIG_SHA
        config=json.loads(config_path.read_text())
        manifest=json.loads(Path(config["cuda_manifest"]).read_text())
        canonical=json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        assert hashlib.sha256(canonical).hexdigest()==config["cuda_manifest_sha256"]
        for row in manifest["components"]:
            path=root/"runtime/lib"/Path(row["path"]).name
            assert digest(path)==row["sha256"]
            plan["vendor"].append(item(path))
        native=root/"runtime/cuest/lib/13/libcuest.so.0.2.50"
        assert digest(native)=="65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c"
        plan["vendor"].append(item(native))
        plan["preload_order"]=[Path(p).name for p in config["native_preload_paths"]]
        plan["identity_helper"]=str(root/"source/psi4/driver/procrouting/sapt/sapt_cuest250_identity.py")
        tree=ast.parse((root/"source/psi4/driver/procrouting/sapt/sapt_cuest_factory_abi.py").read_text())
        table=next(n.value for n in tree.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=="FACTORY_ABI" for t in n.targets))
        plan["factory_symbols"]=[ast.literal_eval(k) for k in table.keys]
        plan["startup_library_path"]+=":"+config["dependency_environment"]+"/lib"
        if args.gpu:
            spec=importlib.util.spec_from_file_location("_prototype_header_identity",
                                                       plan["identity_helper"])
            identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
            plan["abi_header_sha256"]={}
            for name,sha in identity.REQUIRED_HEADERS.items():
                assert digest(root/"runtime/cuest"/name)==sha
                plan["abi_header_sha256"][name]=sha
            plan.update(gpu=True,root=str(root),optional_python_path=config["optional_python_path"],
                        expected_uuid=config["local_lease"]["expected_uuid"],
                        gpu_probe_sha256=digest(Path(__file__).with_name("sealed_gpu_probe.py")))
    assert sum(r["bytes"] for r in plan["vendor"])<=2*1024**3
    input_path=args.output/"input.json";input_path.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
    env=dict(os.environ,LD_LIBRARY_PATH=plan["startup_library_path"],CUDA_VISIBLE_DEVICES="",
             CUDA_CACHE_DISABLE="1")
    assert not env.get("LD_PRELOAD")
    state("starting",vendor_files=len(plan["vendor"]),vendor_preloads=len(plan["preload_order"]))
    if args.gpu:
        # The immutable scientific checkout supplies orchestration only.
        sys.path.insert(0,str(root/"source/devtools/disp0/cachy"))
        from coherent_stage11 import lease,run_command
        from response_reuse_probe import resource_scope
        resource_scope()
        home=args.output/"private-home";home.mkdir(mode=0o700)
        env.update(CUDA_VISIBLE_DEVICES=plan["expected_uuid"],HOME=str(home),
                   PYTHONPATH=plan["optional_python_path"],PYTHONNOUSERSITE="1")
        with lease(root) as fd:
            env["DISP0_LOCAL_LEASE_FD"]=str(fd)
            run_command([sys.executable,"-B",__file__,"--child",str(input_path)],
                        root,args.output,"worker",env,fd,300)
        print((args.output/"worker.stdout").read_text(),end="",flush=True)
        return
    with (args.output/"worker.stdout").open("x") as out,(args.output/"worker.stderr").open("x") as err:
        subprocess.run([sys.executable,"-B",__file__,"--child",str(input_path)],
                       env=env,stdout=out,stderr=err,check=True,timeout=240)
    print((args.output/"worker.stdout").read_text(),end="",flush=True)


if __name__=="__main__":main()
