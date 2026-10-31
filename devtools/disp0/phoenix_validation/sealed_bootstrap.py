"""Early sealed loading for a whole-prefix Phoenix worker, never an overlay."""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys

from gpu_runtime import read_config, require_cuda_manifest, verify_qualified_imports


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
    args, remaining = parser.parse_known_args()
    config, build = read_config(args.config, args.config_sha256)
    owner = bootstrap(config, build)  # Retain through scientific cleanup.
    script = Path(__file__).resolve().with_name("fragment_gate.py")
    sys.argv = [str(script), "--config", str(args.config),
                "--config-sha256", args.config_sha256, *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
