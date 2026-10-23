"""Explicit frozen whole-stage/runtime admission and owned GPU plan lifetime."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_modules(prefixes, root, inventory):
    """Bind actual imports, including preloaded extensions and namespace paths."""
    root = Path(root).resolve()
    for name, module in list(sys.modules.items()):
        if module is None or not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else None
            if relative not in inventory or digest(path) != inventory[relative]:
                raise RuntimeError("Foreign/changed loaded runtime module: " + name)
        for directory in getattr(module, "__path__", ()):
            if not Path(directory).resolve().is_relative_to(root):
                raise RuntimeError("Foreign runtime namespace path: " + name)


def read_config(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    if digest(path) != expected_sha256:
        raise RuntimeError("Frozen runtime configuration changed")
    value = json.loads(path.read_text())
    if value.get("schema") != 1 or value.get("device_family") != "H200":
        raise RuntimeError("Explicit H200 runtime schema required")
    proof = Path(value["build_proof"])
    if digest(proof) != value["build_proof_sha256"]:
        raise RuntimeError("Selected whole-build proof changed")
    build = json.loads(proof.read_text())
    if (build["status"] != "PASS" or build["pin"] != value["source_pin"]
            or build["stage"] != value["stage"]):
        raise RuntimeError("Runtime differs from the completed exact whole stage")
    if digest(Path(build["stage"]) / "bin/psi4") != build["launcher_sha256"]:
        raise RuntimeError("Whole-stage launcher changed")
    if digest(value["wheel_lock"]) != value["wheel_lock_sha256"]:
        raise RuntimeError("Optional dependency artifact lock changed")
    optional = Path(value["optional_python_path"]).resolve()
    actual_optional = {p.relative_to(optional).as_posix(): digest(p)
                       for p in optional.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if actual_optional != value["optional_files"]:
        raise RuntimeError("Installed optional dependency files changed")
    package = Path(build["python_package"]).resolve()
    if not package.is_relative_to(Path(build["stage"]).resolve()):
        raise RuntimeError("Scientific Python package is outside the selected stage")
    actual_science = {p.relative_to(package).as_posix(): digest(p)
                      for p in package.rglob("*.py")}
    if actual_science != build["scientific_python_sha256"]:
        raise RuntimeError("Whole installed scientific Python inventory changed")
    for filename, checksum in build["dependency_python_sha256"].items():
        if digest(filename) != checksum:
            raise RuntimeError("CPU-qualified dependency Python changed: " + filename)
    return value, build


class AuditedRuntime:
    """No implicit paths, hidden plan, library substitution or execution retry."""
    def __init__(self, config_path, config_sha256):
        self.config, self.build = read_config(config_path, config_sha256)
        config = self.config
        self.manifest = json.loads(Path(config["cuda_manifest"]).read_text())
        canonical = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != config["cuda_manifest_sha256"]:
            raise RuntimeError("Frozen CUDA closure changed")
        if os.environ.get("LD_PRELOAD"):
            raise RuntimeError("Scientific runtime cannot use LD_PRELOAD")
        import psi4
        from psi4 import core
        stage = Path(config["stage"]).resolve()
        for module in (psi4, core):
            if not Path(module.__file__).resolve().is_relative_to(stage):
                raise RuntimeError("Scientific imports do not belong to the whole selected stage")
            if str(Path(module.__file__).resolve()) != self.build["module_files"][module.__name__]:
                raise RuntimeError("Imported scientific module differs from CPU-qualified module")
        # Verify the same CPU-qualified core and dependency native worlds.
        for path, checksum in self.build["binary_sha256"].items():
            if digest(path) != checksum:
                raise RuntimeError("CPU-qualified binary changed: " + path)
        import einsums as ein
        import einsums.sealed as sealed
        if (sealed.world()["git_commit"] != "06404fa69ce9346443d7d4f19a250f6db9f0f62c"
                or len(sealed.mapped_einsums_libraries()) != 1):
            raise RuntimeError("Wrong Python/native Einsums world")
        prefix = Path(config["einsums_prefix"]).resolve()
        if not Path(ein.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError("Foreign Python Einsums package")
        dependency_inventory = {str(Path(p).relative_to(prefix)): checksum for p, checksum in
            {**self.build["dependency_python_sha256"], **self.build["binary_sha256"]}.items()
            if Path(p).is_relative_to(prefix)}
        verify_modules(("einsums", "waggle"), prefix, dependency_inventory)
        self.psi4 = psi4
        self.cuda = importlib.import_module("psi4.driver.procrouting.sapt.sapt_cuda_runtime")
        self.abi = importlib.import_module("psi4.driver.procrouting.sapt.sapt_cuest_factory_abi")
        self.plans = importlib.import_module("psi4.driver.procrouting.sapt.sapt_cuest_plan_owner")
        self.dfmo = importlib.import_module("psi4.driver.procrouting.sapt.sapt_cuest_dfmo_cuda")
        # Import cuda-bindings metadata only, never a vendor Python cuEST binding.
        for name, version in config["optional_package_versions"].items():
            distribution = importlib.metadata.distribution(name)
            if (distribution.version != version or not
                    Path(distribution.locate_file("")).resolve().is_relative_to(
                        Path(config["optional_python_path"]).resolve())):
                raise RuntimeError("Optional runtime package version changed: " + name)
        for name in ("cuda.bindings.runtime", "cuda.bindings.driver", "cuda.bindings.nvrtc"):
            importlib.import_module(name)
        verify_modules(("cuda",), config["optional_python_path"], config["optional_files"])
        matches = [v for v in self.manifest["components"]
                   if Path(v["path"]).name.startswith("libcublas.so.")]
        if len(matches) != 1:
            raise RuntimeError("One frozen cuBLAS DSO required")
        self.blas = matches[0]
        self.report = {"source_pin": config["source_pin"],
                       "runtime_config_sha256": config_sha256, "plans": []}

    def device_identity(self, session):
        """Correlate the actual owning PID/context UUID with allocation hardware."""
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
        rows = [[v.strip() for v in line.split(",")] for line in processes.splitlines()]
        own = [row for row in rows if len(row) == 3 and row[0] == str(os.getpid())]
        uuids = {row[1] for row in own}
        if len(uuids) != 1:
            raise RuntimeError("Expected one actual CUDA UUID for the current PID")
        uuid = next(iter(uuids))
        foreign = [row for row in rows if len(row) == 3 and row[1] == uuid
                   and row[0] != str(os.getpid())]
        if foreign:
            raise RuntimeError("Foreign compute contexts on the allocated GPU UUID")
        hardware = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
        devices = [[v.strip() for v in line.split(",")] for line in hardware.splitlines()]
        device = next((row for row in devices if row[0] == uuid), None)
        if device is None or "H200" not in device[1] or session.arch != "compute_90":
            raise RuntimeError("Actual CUDA/NVML device is not the requested H200")
        return {"uuid": uuid, "name": device[1], "driver": device[2],
                "memory_MiB": int(device[3]), "pid": os.getpid(),
                "nvrtc_architecture": session.arch, "ptx_sha256": session.ptx_sha256}

    @contextmanager
    def provider(self, primary, auxiliary):
        config = self.config
        record = {}
        self.report["plans"].append(record)
        session = owner = None
        primary_error = None
        try:
            with self.cuda.CUDASession(
                cublas_library=self.blas["path"], cublas_sha256=self.blas["sha256"],
                memory_limit_bytes=config["device_memory_limit_bytes"],
                native_persistent_bytes=config["native_persistent_bytes"]) as session:
                record["device"] = self.device_identity(session)
                api = self.abi.admit_factory_api(
                session=session, sdk_root=Path(config["cuest_root"]),
                archive=Path(config["cuest_archive"]), cuda_manifest=self.manifest,
                cuda_manifest_sha256=config["cuda_manifest_sha256"])
                owner = self.plans.DiagnosticPlanOwner(
                api, primary, auxiliary, coefficient_candidate="UNVALIDATED_PSI4_COEF",
                host_workspace_limit=256*1024**2)
                with owner:
                    provider = self.dfmo.CuESTDFMOProvider(
                        session=session, sdk_root=Path(config["cuest_root"]),
                        archive=Path(config["cuest_archive"]), cuda_manifest=self.manifest,
                        cuda_manifest_sha256=config["cuda_manifest_sha256"],
                        handle=owner.handle, plan=owner.plan, handle_pid=owner.pid,
                        primary_basis=primary, auxiliary_basis=auxiliary, keepalive=(owner,),
                        plan_policy=dict(self.dfmo.POLICY), output_order="C",
                        variable_bytes=256*1024**2)
                    yield provider
                    # Public entrypoints own the provider's inner context.
                    if not provider.closed:
                        raise RuntimeError("Public GPU entrypoint did not close its provider")
                    record["producer"] = provider.report()
        except BaseException as error:
            primary_error = error
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if owner is not None:
                record["owner_cleanup"] = owner.report()
            if session is not None:
                record["session_cleanup"] = {"complete": session.cleanup_complete,
                    "live_bytes": session.live_bytes, "native_owners": len(session.native_owners),
                    "peak_explicit_bytes": session.peak_live_bytes,
                    "successful_blas_calls": dict(session.blas_calls),
                    "successful_kernel_launches": dict(session.kernel_launches)}
                if session.live_bytes or session.native_owners or not session.cleanup_complete:
                    if primary_error is None:
                        raise RuntimeError("GPU runtime teardown is incomplete")
                    primary_error.add_note("GPU runtime teardown is incomplete; see cleanup report")
