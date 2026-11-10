"""CPU-only private CUDA Python/loader preparation; no CUDA imports or calls."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

BINDINGS_VERSION = "13.3.0"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def identity_module(source):
    spec = importlib.util.spec_from_file_location(
        "phoenix_cuest_file_identity",
        source / "psi4/driver/procrouting/sapt/sapt_cuest250_identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loader_manifest(source_sdk, target, identity):
    """Fresh logical locator; all genuine DSOs remain read-only, hash-bound inputs."""
    target.mkdir()
    for name in ("include", "bin", "nvvm"):
        (target / name).symlink_to(source_sdk / name, target_is_directory=True)
    lib = target / "lib64"
    lib.mkdir()
    unique = {}
    entries = [p for p in sorted((source_sdk / "lib64").iterdir()) if p.exists()]
    for path in entries:
        (lib / path.name).symlink_to(path.resolve(strict=True), target_is_directory=path.is_dir())
    targets = target / "targets/x86_64-linux"
    targets.mkdir(parents=True)
    (targets / "include").symlink_to(target / "include", target_is_directory=True)
    (targets / "lib").symlink_to(lib, target_is_directory=True)
    for path in entries:
        if path.is_file() and ".so" in path.name and path.name.startswith(identity.CUDA_PREFIXES):
            real = path.resolve(strict=True)
            if str(real) in unique:
                continue
            # Prefer each real DSO's filename, not a potentially ambiguous alias.
            candidate = lib / real.name
            if not candidate.exists():
                candidate.symlink_to(real)
            elif candidate.resolve(strict=True) != real:
                raise RuntimeError("Conflicting frozen CUDA DSO locator: " + str(candidate))
            unique[str(real)] = {"path": str(candidate), "sha256": digest(real)}
    if not unique:
        raise RuntimeError("No CUDA loader closure components")
    manifest = {"cuda_root": str(target), "cuda_runtime_version": 13030,
                "nvrtc_version": [13, 3],
                "components": sorted(unique.values(), key=lambda r: r["path"])}
    for record in manifest["components"]:
        identity.check_cuda13_name(record["path"])
    names = [Path(record["path"]).name for record in manifest["components"]]
    for stem in ("libcudart", "libcublas", "libcublasLt", "libcusolver", "libcusparse",
                 "libnvJitLink", "libnvrtc", "libnvrtc-builtins"):
        if not any(name.startswith(stem + ".so.") for name in names):
            raise RuntimeError("Missing explicit execution closure family: " + stem)
    return manifest


def prepare(source, run):
    build_path = run / "result.json"
    build = json.loads(build_path.read_text())
    if build["status"] != "PASS" or build["pin"] != os.environ["DISP0_PIN"]:
        raise RuntimeError("Exact completed CPU build proof required")
    root = run / "gpu-runtime"
    root.mkdir()
    result = {"status": "PREPARING", "scope": "PRIVATE_DEPENDENCY_PREPARATION_ONLY",
              "source_pin": build["pin"], "build_proof_sha256": digest(build_path),
              "gpu_correctness_verified": False}
    try:
        identity = identity_module(source)
        sdk = Path(os.environ["DISP0_CUEST_ROOT"]).resolve()
        archive = sdk / "cuest_v0.2.50.tar.gz"
        result["cuest_files"] = identity.verify_files(sdk, archive)
        wheels = root / "wheels"
        wheels.mkdir()

        def command(label, args):
            record = {"argv": args}
            try:
                completed = subprocess.run(args, capture_output=True, text=True, timeout=240)
                record.update(returncode=completed.returncode, stdout=completed.stdout,
                              stderr=completed.stderr)
            except subprocess.TimeoutExpired as error:
                text = lambda x: x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
                record.update(timeout=True, stdout=text(error.stdout), stderr=text(error.stderr))
                raise
            finally:
                save(root / (label + ".json"), record)
            if completed.returncode:
                raise RuntimeError("Private dependency command failed: " + label)

        pip = [sys.executable, "-B", "-m", "pip", "--isolated",
               "--disable-pip-version-check", "--no-cache-dir"]
        command("wheel-download", [*pip, "download", "--only-binary=:all:",
                "--index-url", "https://pypi.org/simple", "--dest", str(wheels),
                "--progress-bar", "off", "--no-input", "--timeout", "20", "--retries", "0",
                "cuda-bindings==" + BINDINGS_VERSION])
        wheel_records = [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                         for p in sorted(wheels.glob("*.whl"))]
        if not wheel_records:
            raise RuntimeError("No frozen Python CUDA artifacts")
        save(root / "wheel-lock.json", wheel_records)
        optional = root / "python"
        command("wheel-install", [*pip, "install", "--no-index", "--find-links", str(wheels),
                "--only-binary=:all:", "--target", str(optional), "--no-compile", "--no-input",
                "--report", str(root / "install-report.json"), "cuda-bindings==" + BINDINGS_VERSION])
        versions = {d.metadata["Name"].lower().replace("_", "-"): d.version
                    for d in importlib.metadata.distributions(path=[str(optional)])}
        if versions.get("cuda-bindings") != BINDINGS_VERSION:
            raise RuntimeError("Wrong installed optional CUDA Python version")
        result["optional_package_versions"] = versions
        manifest = loader_manifest(run / "cuda13", root / "cuda13", identity)
        manifest_path = root / "cuda-manifest.json"
        save(manifest_path, manifest)
        config = {"schema": 1, "scope": "SMALL_H200_NUMERICAL_GATE_ONLY",
            "source": str(source), "source_pin": build["pin"], "stage": build["stage"],
            "build_proof": str(build_path), "build_proof_sha256": digest(build_path),
            "dependency_environment": build["dependency_environment"],
            "einsums_prefix": build["read_only_einsums_dependency"],
            "optional_python_path": str(optional), "optional_package_versions": versions,
            "optional_files": {p.relative_to(optional).as_posix(): digest(p)
                               for p in optional.rglob("*")
                               if p.is_file() and "__pycache__" not in p.parts},
            "wheel_lock": str(root / "wheel-lock.json"),
            "wheel_lock_sha256": digest(root / "wheel-lock.json"),
            "cuda_manifest": str(manifest_path), "cuda_manifest_sha256": canonical_sha(manifest),
            "cuest_root": str(sdk), "cuest_archive": str(archive),
            "device_family": "H200", "threads": 4, "host_memory": "8 GiB",
            "device_memory_limit_bytes": 8*1024**3, "native_persistent_bytes": 512*1024**2}
        config_path = root / "runtime-config.json"
        save(config_path, config)
        result.update(status="PREPARED_NOT_GPU_VALIDATED", runtime_config=str(config_path),
                      runtime_config_sha256=digest(config_path),
                      cuda_manifest_sha256=canonical_sha(manifest))
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(root / "preparation-result.json", result)
    print(json.dumps({k: result[k] for k in
                      ("status", "runtime_config", "runtime_config_sha256")}), flush=True)


if __name__ == "__main__":
    prepare(Path(os.environ["DISP0_SOURCE"]).resolve(),
            Path(os.environ["DISP0_JOB_RUN"]).resolve())
