"""Compile ctypes wire signatures against canonical headers; never run cuEST."""
import argparse
import ctypes as ct
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir()  # NEW evidence only.
    sapt = Path(__file__).resolve().parents[3]/"psi4/driver/procrouting/sapt"
    package = types.ModuleType("_disp0_factory_header_probe")
    package.__path__ = [str(sapt)]
    sys.modules[package.__name__] = package
    api = importlib.import_module(package.__name__+".sapt_cuest_factory_abi")
    proof = api.identity.verify_files(args.sdk_root,args.archive)
    compiler = args.compiler.resolve(strict=True)
    version = subprocess.check_output(
        [str(compiler),"-dumpfullversion"],text=True,timeout=10).strip()
    if version != "14.3.0":
        raise RuntimeError("Factory header ABI probe requires reviewed GNU14.3.0")
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    wire_types = {
        ct.c_int:"int", ct.c_uint64:"std::uint64_t", ct.c_uint32:"std::uint32_t",
        ct.c_double:"double", api.H:"void*", api.PH:"void**",
        api.PU64:"std::uint64_t*", api.PD:"double*",
        api.PQ:"cuestWorkspaceDescriptor_t*", api.PW:"cuestWorkspace_t*",
    }
    preamble = r"""
#include <cuest.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
template<class T> struct Norm;
template<class B, bool P, bool E> struct Select { using type = B; };
template<class B, bool E> struct Select<B,true,E> {
    using type = typename Norm<std::remove_pointer_t<B>>::type*;
};
template<class B> struct Select<B,false,true> {
    static_assert(sizeof(B)==sizeof(int), "native enum width");
    using type = int;
};
template<class T> struct Norm {
    using B = std::remove_cv_t<T>;
    using type = typename Select<B,std::is_pointer_v<B>,std::is_enum_v<B>>::type;
};
template<class T> struct Sig;
template<class R, class... A> struct Sig<R(*)(A...)> {
    using result = typename Norm<R>::type;
    using args = std::tuple<typename Norm<A>::type...>;
};
static_assert(sizeof(void*)==8 && sizeof(size_t)==8 && sizeof(int)==4, "host ABI");
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559, "FP64");
static_assert(sizeof(cuestWorkspaceDescriptor_t)==16, "descriptor size");
static_assert(offsetof(cuestWorkspaceDescriptor_t,hostBufferSizeInBytes)==0, "descriptor host");
static_assert(offsetof(cuestWorkspaceDescriptor_t,deviceBufferSizeInBytes)==8, "descriptor device");
static_assert(sizeof(cuestWorkspace_t)==32, "workspace size");
static_assert(offsetof(cuestWorkspace_t,hostBuffer)==0, "workspace host");
static_assert(offsetof(cuestWorkspace_t,hostBufferSizeInBytes)==8, "workspace host bytes");
static_assert(offsetof(cuestWorkspace_t,deviceBuffer)==16, "workspace device");
static_assert(offsetof(cuestWorkspace_t,deviceBufferSizeInBytes)==24, "workspace device bytes");
"""
    constants = {
        "CUEST_HANDLE_PARAMETERS":0, "CUEST_AOSHELL_PARAMETERS":1,
        "CUEST_AOBASIS_PARAMETERS":2, "CUEST_AOPAIRLIST_PARAMETERS":3,
        "CUEST_DFINTPLAN_PARAMETERS":5, "CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS":49,
        "CUEST_NATIVE_FP64_MATH_MODE":1,
        "CUEST_HANDLE_PARAMETERS_CUDASTREAM":0, "CUEST_HANDLE_PARAMETERS_CUBLAS":1,
        "CUEST_HANDLE_PARAMETERS_CUSOLVER":2, "CUEST_HANDLE_PARAMETERS_JIT_COMPILE_THREADS":8,
        "CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF":0,
        "CUEST_DFINTPLAN_PARAMETERS_FITTING_RELATIVE_CONDITIONING":1,
        "CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION":2,
        "CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM":3,
        "CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION":4,
        "CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA":5,
        "CUEST_DFINTPLAN_PARAMETERS_THREE_INDEX_INTEGRAL_DIRECT":6,
        "CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM_MATRIXPOWER":1,
        "CUEST_AOBASIS":2, "CUEST_AOBASIS_IS_PURE":0,
        "CUEST_AOBASIS_IS_CART":1, "CUEST_AOBASIS_IS_MIXED":2,
        "CUEST_AOBASIS_NUM_ATOM":3, "CUEST_AOBASIS_NUM_SHELL":4,
        "CUEST_AOBASIS_NUM_AO":5, "CUEST_AOBASIS_NUM_CART":6,
        "CUEST_AOBASIS_NUM_PRIMITIVE":7, "CUEST_AOBASIS_MAX_L":8,
    }
    preamble += "\n".join(f'static_assert({name}=={value}, "{name}");'
                          for name,value in constants.items())+"\n"
    def source(negative):
        signatures = dict(api.FACTORY_ABI)
        if negative:
            wrong = list(signatures["cuestAOShellCreate"])
            wrong[3] = ct.c_uint32  # Intentional primitive-count truncation.
            signatures["cuestAOShellCreate"] = tuple(wrong)
        text = preamble
        for name, types_ in signatures.items():
            expected = ",".join(wire_types[t] for t in types_)
            text += (f'static_assert(std::is_same_v<Sig<decltype(&{name})>::args,'
                     f'std::tuple<{expected}>>, "{name} wire signature mismatch");\n')
            text += f'static_assert(std::is_same_v<Sig<decltype(&{name})>::result,int>, "{name} status");\n'
        return text
    receipt = {"scope":"Canonical-header compile/type/layout proof ONLY; no native calls or GPU execution",
               "compiler":str(compiler), "compiler_version":version,
               "compiler_sha256":hashlib.sha256(compiler.read_bytes()).hexdigest(),
               "sdk_file_proof":proof, "factory_function_count":len(api.FACTORY_ABI),
               "abi_source_sha256":hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest(),
               "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "commands":[], "status":"running"}
    receipt_path = args.out/"receipt.json"
    def persist():
        receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
    persist()
    try:
        for label,negative in (("negative-count",True),("positive",False)):
            cpp = args.out/(label+".cc")
            cpp.write_text(source(negative))
            dependency = args.out/(label+".d")
            command = [str(compiler),"-std=c++17",f"-I{args.sdk_root.resolve()}/include",
                       f"-I{args.cuda_root.resolve()}/targets/x86_64-linux/include",
                       "-MMD","-MF",str(dependency),"-c",str(cpp),"-o",str(args.out/(label+".o"))]
            env = {k:os.environ[k] for k in ("HOME","USER") if k in os.environ}
            env.update(PATH=f"{compiler.parent}:/usr/bin:/bin",PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(command,capture_output=True,text=True,timeout=45,
                                    cwd=args.out,env=env)
            (args.out/(label+".stdout")).write_text(result.stdout)
            (args.out/(label+".stderr")).write_text(result.stderr)
            receipt["commands"].append({"label":label,"argv":command,"exit":result.returncode})
            persist()
            if negative:
                if result.returncode == 0 or "cuestAOShellCreate wire signature mismatch" not in result.stderr:
                    raise RuntimeError("Wrong-count negative control did not fail at intended signature")
            elif result.returncode != 0:
                raise RuntimeError("Canonical-header factory ABI compile failed")
        consumed = shlex.split((args.out/"positive.d").read_text().replace("\\\n"," ").split(":",1)[1])
        headers = {}
        for name in consumed:
            path = Path(name).resolve(strict=True)
            if path.is_relative_to(args.sdk_root.resolve()/"include"):
                relative = str(path.relative_to(args.sdk_root.resolve()))
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if proof["header_files"]["cuest_PID/"+relative]["sha256"] != actual:
                    raise RuntimeError("Consumed canonical header changed")
                headers[relative] = actual
            elif path.name == "cuest.h" or "/include/cuest/" in str(path):
                raise RuntimeError("Compiler consumed a foreign cuEST header")
        if not headers:
            raise RuntimeError("No consumed canonical cuEST headers recorded")
        receipt.update(status="passed-header-abi-positive-and-negative-control",
                       consumed_cuest_headers=headers)
    except BaseException as error:
        receipt.update(status="failed",error=str(error))
        raise
    finally:
        persist()
        print(receipt["status"],"functions",len(api.FACTORY_ABI),"receipt",receipt_path)


if __name__ == "__main__":
    main()
