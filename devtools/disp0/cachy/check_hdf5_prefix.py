"""Reproduce foreign HDF5 discovery, then verify one-variable private-root fix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--attempt",required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if not args.attempt.replace("-","").isalnum():
        raise ValueError("Invalid probe attempt")
    runtime,source = root/"environment-v2",root/"deps-src/einsums"
    expected_pin = "06404fa69ce9346443d7d4f19a250f6db9f0f62c"
    if subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() != expected_pin:
        raise RuntimeError("Wrong Einsums source")
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Dirty Einsums source")
    run = root/"runs"/args.attempt
    run.mkdir()
    env = {k:os.environ[k] for k in ("HOME","USER","TMPDIR") if k in os.environ}
    env.update(PATH=f"{runtime}/bin:/usr/bin:/bin",
               CC=f"{runtime}/bin/x86_64-conda-linux-gnu-cc",
               CXX=f"{runtime}/bin/x86_64-conda-linux-gnu-c++",
               CMAKE_BUILD_PARALLEL_LEVEL="1")
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    records = [json.loads(p.read_text()) for p in (runtime/"conda-meta").glob("hdf5-*.json")]
    if len(records) != 1:
        raise RuntimeError("Require one pinned private HDF5 package record")
    expected_version = records[0]["version"]
    cmake = str(runtime/"bin/cmake")
    receipt = {"scope":"Private HDF5 discovery/link/runtime probe, no CUDA/native cuEST",
               "expected_private_hdf5":expected_version,"einsums_pin":expected_pin,
               "commands":[],"status":"running",
               "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def persist():
        (run/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    def command(label,argv):
        print(label," ".join(argv),flush=True)
        start = time.monotonic()
        result = subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True,timeout=45)
        (run/(label+".stdout")).write_text(result.stdout)
        (run/(label+".stderr")).write_text(result.stderr)
        receipt["commands"].append({"label":label,"argv":argv,"exit":result.returncode,
                                    "seconds":time.monotonic()-start})
        persist()
        return result
    persist()
    try:
        probe_source = Path(__file__).resolve().parent/"hdf5_probe"
        for label,private_root in (("red",False),("green",True)):
            build = root/"build"/(args.attempt+"-"+label)
            if build.exists():
                raise RuntimeError("Probe build must be NEW")
            argv = [cmake,"-S",str(probe_source),"-B",str(build),"-GNinja",
                    "-DCMAKE_BUILD_TYPE=Release",f"-DCMAKE_PREFIX_PATH={runtime}",
                    f"-DEINSUMS_SOURCE={source}",f"-DPRIVATE_PREFIX={runtime}"]
            if private_root:
                argv.append(f"-DHDF5_ROOT={runtime}")  # Sole causal discovery change.
            configured = command(label+"-configure",argv)
            if configured.returncode:
                raise RuntimeError("Probe configure failed before intended link signal")
            receipt[label+"_selection"] = (build/"selection.txt").read_text()
            built = command(label+"-build",[cmake,"--build",str(build),"--parallel","1"])
            if not private_root:
                if built.returncode == 0:
                    executed = command(label+"-run",[str(build/"hdf5_probe")])
                    selected = dict(line.split("=",1) for line in executed.stdout.splitlines() if "=" in line)
                    if executed.returncode == 0 and Path(selected["library"]).resolve().is_relative_to(runtime):
                        raise RuntimeError("Negative control did not reproduce foreign discovery")
                elif "__isoc23_" not in built.stdout+built.stderr:
                    raise RuntimeError("Negative link failed for a different reason")
                receipt["negative_control"] = "foreign HDF5 selection reproduced"
            else:
                if built.returncode:
                    raise RuntimeError("Private-root HDF5 link failed")
                executed = command(label+"-run",[str(build/"hdf5_probe")])
                if executed.returncode:
                    raise RuntimeError("Private HDF5 header/runtime mismatch")
                selected = dict(line.split("=",1) for line in executed.stdout.splitlines() if "=" in line)
                library = Path(selected["library"]).resolve(strict=True)
                if (not library.is_relative_to(runtime)
                        or selected["runtime"] != expected_version
                        or selected["header"] != expected_version):
                    raise RuntimeError("Private HDF5 provenance/version differs")
                receipt["private_library"] = str(library)
                receipt["private_library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        receipt["status"] = "passed-private-root-discovery-link-and-runtime"
    except BaseException as error:
        receipt.update(status="failed",error=str(error))
        raise
    finally:
        persist()
        print(receipt["status"],run,flush=True)


if __name__ == "__main__":
    main()
