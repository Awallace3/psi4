#!/usr/bin/env python3
"""Compile/link-only CUDA13 ABI gate in the actual job environment; no GPU calls."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

SOURCE = """#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuest.h>
extern "C" cudaError_t psi4_cuda_abi_probe(cudaDeviceProp* p) {
    return cudaGetDeviceProperties(p, 0);
}
"""


def header_flags(command):
    """Preserve ordered includes/defines and compiler search options, not LTO."""
    args = shlex.split(command)
    flags = []
    index = 1
    separate = {"-I", "-isystem", "-iquote", "-idirafter", "-D", "-U", "-B", "--sysroot"}
    while index < len(args):
        arg = args[index]
        if arg in separate:
            flags.extend(args[index:index+2])
            index += 2
            continue
        if arg.startswith(("-I", "-isystem", "-iquote", "-idirafter", "-D", "-U", "-B",
                           "--sysroot=", "-std=")):
            flags.append(arg)
        index += 1
    return args[0], flags


def runtime_command(database):
    for item in json.loads(Path(database).read_text()):
        path = Path(item["file"])
        if path.name == "cuest_runtime.cc" or (
                "/core.dir/" in str(path) and path.name.startswith("unity_")
                and "cuest_runtime.cc" in path.read_text()):
            return item
    raise ValueError("No cuest_runtime compilation command found")


def probe(compiler, flags, runtime, out, cwd=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(compiler=compiler, header_flags=flags, runtime=str(runtime), ok=False,
                    no_gpu_calls=True)
    def execute(command, stdin=None):
        proc = subprocess.run(command, input=stdin, capture_output=True, text=True,
                              cwd=cwd, timeout=30)
        return proc
    try:
        pp = execute([compiler, *flags, "-E", "-H", "-dM", "-x", "c++", "-"], SOURCE)
        (out/"headers.log").write_text(pp.stderr)
        version = re.search(r"^#define CUDART_VERSION (\d+)$", pp.stdout, re.M)
        metadata["cudart_version"] = int(version[1]) if version else None
        if pp.returncode or not version or int(version[1]) // 1000 != 13:
            raise RuntimeError(f"Expected CUDA13 headers; selected {metadata['cudart_version']}")
        obj = out/"probe.o"
        compile = execute([compiler, *flags, "-fPIC", "-fno-lto", "-x", "c++",
                           "-c", "-o", str(obj), "-"], SOURCE)
        (out/"compile.log").write_text(compile.stdout + compile.stderr)
        if compile.returncode:
            raise RuntimeError("ABI probe compilation failed")
        # No symbol-version alias/shim: require the selected runtime to actually
        # export the symbol chosen by these headers.
        link = execute([compiler, "-shared", str(obj), str(Path(runtime).resolve()),
                        "-Wl,--no-undefined", "-o", str(out/"probe.so")])
        (out/"link.log").write_text(link.stdout + link.stderr)
        if link.returncode:
            raise RuntimeError("CUDA header/runtime ABI probe failed to link")
        metadata["ok"] = True
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    (out/"result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-database", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    item = runtime_command(args.compile_database)
    compiler, flags = header_flags(item["command"])
    result = probe(compiler, flags, args.runtime, args.output, item["directory"])
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
