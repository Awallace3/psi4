"""Early sealed loading for a whole-prefix Phoenix worker, never an overlay."""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys

from gpu_runtime import AuditedRuntime, read_config, require_cuda_manifest, verify_qualified_imports


def cpu_admission(config, config_path, checksum, output):
    """Exercise the GPU adapter's file/binding admission, without a CUDA session."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("SLURM_JOB_GPUS"):
        raise RuntimeError("CPU-only runtime admission must hide/allocate no GPUs")
    runtime = AuditedRuntime(config_path, checksum)
    if (runtime.identity_owner is None
            or runtime.identity_owner.contract != "process-owned-kernel-sealed-snapshots-v1"):
        raise RuntimeError("Exact sealed runtime adoption required")
    bindings = []
    for stem, symbols in (("libcublas.so.", ("cublasCreate_v2","cublasDestroy_v2")),
                          ("libcusolver.so.", ("cusolverDnCreate","cusolverDnDestroy"))):
        rows = [row for row in runtime.manifest["components"]
                if Path(row["path"]).name.startswith(stem)]
        if len(rows) != 1:
            raise RuntimeError("One explicit admitted library required: "+stem)
        row = rows[0]
        # Same cold filesystem-locator binding used by CUDA/cuSOLVER consumers;
        # no CUDASession, native handle creation, allocation or kernel call.
        native, canonical = runtime.cuda.library(row["path"], row["sha256"],
                                                owner=runtime.identity_owner)
        runtime.identity_owner.verify_symbols(canonical, symbols)
        bindings.append({"declared_locator": row["path"], "canonical_source": canonical,
                         "sha256": row["sha256"], "symbols": list(symbols)})
    runtime.report["immutable_runtime"] = runtime.identity_owner.report()
    import json
    with Path(output).open("x") as stream:
        json.dump({"status": "PASS", "scope": "CPU_RUNTIME_ADMISSION_ONLY_NOT_GPU_QUALIFICATION",
                   "source_pin": config["source_pin"], "runtime_config_sha256": checksum,
                   "gpu_correctness_verified": False, "cuda_session_created": False,
                   "bindings": bindings, "runtime": runtime.report},
                  stream, indent=2, sort_keys=True)


def bootstrap(config, build):
    if ((config["schema"], config["device_family"]) != (1, "H200")
            or config.get("identity_contract") !=
            "process-owned-kernel-sealed-snapshots-v1"):
        raise RuntimeError("Explicit Phoenix sealed configuration required")
    if "psi4" in sys.modules or "einsums" in sys.modules:
        raise RuntimeError("Sealed bootstrap must precede scientific imports")
    # Reuse the authenticated loader implementation, not a copied science file.
    folder = Path(__file__).resolve().parents[1] / "cachy"
    sys.path.insert(0, str(folder))
    try:
        from bootstrap_gate import prepare_sealed_owner, native_symbols, ConfigureEinsums
    finally:
        sys.path.pop(0)
    owner = prepare_sealed_owner(config, build, require_cuda_manifest(config))
    observer = ConfigureEinsums(config)
    sys.meta_path.insert(0, observer)
    try:
        importlib.import_module("psi4")
        importlib.import_module("psi4.core")
        importlib.import_module("einsums")
        if not observer.configured:
            raise RuntimeError("Einsums CPU-only upstream policy was not configured")
        verify_qualified_imports(config, build)
        native = str((Path(config["cuest_root"]) /
                      "lib/13/libcuest.so.0.2.50").resolve(strict=True))
        owner.verify_symbols(native, native_symbols(build["python_package"]))
    except BaseException:
        owner.invalidate("Phoenix sealed bootstrap import/admission failed")
        raise
    finally:
        sys.meta_path.remove(observer)
    return owner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--runtime-admission-only", action="store_true")
    parser.add_argument("--admission-output", type=Path)
    args, remaining = parser.parse_known_args()
    config, build = read_config(args.config, args.config_sha256)
    owner = bootstrap(config, build)  # Retain through scientific cleanup.
    if args.runtime_admission_only:
        if args.admission_output is None or remaining:
            parser.error("CPU admission requires its own output and no scientific arguments")
        cpu_admission(config, args.config, args.config_sha256, args.admission_output)
        return
    if args.admission_output is not None:
        parser.error("Admission output requires the CPU-only admission mode")
    script = Path(__file__).resolve().with_name("fragment_gate.py")
    sys.argv = [str(script), "--config", str(args.config),
                "--config-sha256", args.config_sha256, *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
