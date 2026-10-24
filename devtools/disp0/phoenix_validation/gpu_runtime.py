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
import stat
import re
import uuid as uuid_module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_modules(prefixes, root, inventory, namespace_roots=None):
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
            resolved = Path(directory).resolve()
            if (not resolved.is_relative_to(root) or namespace_roots is not None
                    and not any(resolved.is_relative_to(root / p) for p in namespace_roots)):
                raise RuntimeError("Foreign runtime namespace path: " + name)


def require_local_lease(policy):
    """Verify an inherited, already-held cooperative lease; never acquire one.

    This inspects only our own descriptor. It neither queries other agents'
    processes nor treats a desktop GPU as an exclusive SLURM allocation.
    """
    lock, hold = Path(policy["lock_path"]), Path(policy["hold_path"])
    if (not lock.is_absolute() or not hold.is_absolute() or lock.name!="gpu0.lock"
            or hold.name!="gpu0.HOLD" or lock.parent!=hold.parent or lock.is_symlink()):
        raise RuntimeError("Explicit common local GPU lock/HOLD paths required")
    if hold.exists():
        raise RuntimeError("Local GPU HOLD is active")
    try:
        fd=int(os.environ["DISP0_LOCAL_LEASE_FD"])
        if fd<3:raise ValueError("Not an owned inherited descriptor")
        opened,current=os.fstat(fd),lock.stat()
        info=Path(f"/proc/self/fdinfo/{fd}").read_text()
    except (KeyError,ValueError,OSError) as error:
        raise RuntimeError("An inherited held local GPU lease is required") from error
    after=lock.stat()
    if (not stat.S_ISREG(opened.st_mode) or opened.st_uid!=os.getuid()
            or current.st_uid!=os.getuid()
            or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino)
            or re.search(r"lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+",info) is None
            or (after.st_dev,after.st_ino)!=(opened.st_dev,opened.st_ino)):
        raise RuntimeError("Inherited descriptor is not the held common GPU flock")
    if re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",policy["expected_uuid"]) is None:
        raise RuntimeError("An exact local CUDA UUID is required")
    return fd


def optional_inventory(config):
    root=Path(config["optional_python_path"]).resolve()
    names=config.get("optional_inventory_roots")
    if names is None:  # Original private H200 wheel installation, unchanged.
        roots=[root]
    else:
        if (config.get("schema")!=2 or config.get("device_family")!="RTX3090"
                or not isinstance(names,list) or len(set(names))!=len(names)
                or "cuda" not in names or any(
                    n!="cuda" and re.fullmatch(r"[A-Za-z0-9_.-]+\.dist-info",n) is None
                    for n in names)):
            raise RuntimeError("Explicit complete local CUDA namespace/distribution roots required")
        roots=[root/name for name in names]
        if any(not p.is_dir() or not p.resolve().is_relative_to(root) for p in roots):
            raise RuntimeError("Optional namespace inventory escaped its selected installation")
    result={}
    for selected in roots:
        for p in selected.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                if not p.resolve().is_relative_to(selected.resolve()):
                    raise RuntimeError("Optional dependency file escaped its inventory root")
                result[p.relative_to(root).as_posix()]=digest(p)
    return result


def read_config(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    if digest(path) != expected_sha256:
        raise RuntimeError("Frozen runtime configuration changed")
    value = json.loads(path.read_text())
    if (value.get("schema"),value.get("device_family"))==(1,"H200"):
        pass
    elif (value.get("schema"),value.get("device_family"))==(2,"RTX3090"):
        require_local_lease(value["local_lease"])
    else:
        raise RuntimeError("Explicit H200 or leased RTX3090 runtime schema required")
    proof = Path(value["build_proof"])
    if digest(proof) != value["build_proof_sha256"]:
        raise RuntimeError("Selected whole-build proof changed")
    build = json.loads(proof.read_text())
    if (build["status"] != "PASS" or build["pin"] != value["source_pin"]
            or build["stage"] != value["stage"]):
        raise RuntimeError("Runtime differs from the completed exact whole stage")
    if value["schema"]==2:
        if build.get("scope")!="WHOLE_BUILD_CPU_SMOKE_ONLY_NOT_GPU_QUALIFICATION":
            raise RuntimeError("Explicit completed whole local CPU build proof required")
        for filename,checksum in build["provenance_sha256"].items():
            if digest(filename)!=checksum:
                raise RuntimeError("Local whole-build/helper proof changed: "+filename)
        require_cuda_manifest(value)
        required={"psi4","psi4.core","einsums","einsums._core","waggle","qcelemental","qcengine","optking"}
        if not required.issubset(build["module_files"]):
            raise RuntimeError("Incomplete qualified module identity closure")
        bootstrap=Path(__file__).resolve().parents[1]/"cachy/bootstrap_gate.py"
        for code in (bootstrap,Path(__file__).resolve(),Path(value["metadata_path"])/"import_order.py"):
            if str(code) not in build["provenance_sha256"]:
                raise RuntimeError("Missing authenticated local bootstrap/admission helper")
    if digest(Path(build["stage"]) / "bin/psi4") != build["launcher_sha256"]:
        raise RuntimeError("Whole-stage launcher changed")
    if digest(value["wheel_lock"]) != value["wheel_lock_sha256"]:
        raise RuntimeError("Optional dependency artifact lock changed")
    actual_optional = optional_inventory(value)
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
    for filename, checksum in build["binary_sha256"].items():
        if digest(filename) != checksum:
            raise RuntimeError("CPU-qualified binary changed: " + filename)
    return value, build


def require_cuda_manifest(config):
    manifest=json.loads(Path(config["cuda_manifest"]).read_text())
    canonical=json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=config["cuda_manifest_sha256"]:
        raise RuntimeError("Frozen CUDA closure changed before native loading")
    return manifest


def verify_qualified_imports(config,build):
    """Check actual loaded dependencies in both local scientific arms."""
    for name,filename in build["module_files"].items():
        module=importlib.import_module(name)
        if Path(module.__file__).resolve()!=Path(filename).resolve():
            raise RuntimeError("Actual qualified module path changed: "+name)
    root=Path(config["dependency_environment"]).resolve()
    inventory={str(Path(p).relative_to(root)):checksum for p,checksum in
               {**build["dependency_python_sha256"],**build["binary_sha256"]}.items()
               if Path(p).is_relative_to(root)}
    verify_modules(("einsums","waggle","qcelemental","qcengine","optking"),root,inventory)


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
            if config.get("optional_inventory_roots") is not None:
                metadata_roots={Path(str(f)).parts[0] for f in (distribution.files or ())
                                if Path(str(f)).parts and Path(str(f)).parts[0].endswith(".dist-info")}
                if (not metadata_roots or not metadata_roots.issubset(config["optional_inventory_roots"])):
                    raise RuntimeError("Optional distribution metadata is outside the frozen inventory")
        for name in ("cuda.bindings.runtime", "cuda.bindings.driver", "cuda.bindings.nvrtc"):
            importlib.import_module(name)
        verify_modules(("cuda",), config["optional_python_path"], config["optional_files"],
                       ("cuda",) if config.get("schema")==2 else None)
        matches = [v for v in self.manifest["components"]
                   if Path(v["path"]).name.startswith("libcublas.so.")]
        if len(matches) != 1:
            raise RuntimeError("One frozen cuBLAS DSO required")
        self.blas = matches[0]
        self.report = {"source_pin": config["source_pin"],
                       "runtime_config_sha256": config_sha256, "plans": []}

    def require_device_admission(self):
        if self.config["device_family"]=="RTX3090":
            require_local_lease(self.config["local_lease"])

    def verify_optional_modules(self):
        config=self.config
        if optional_inventory(config)!=config["optional_files"]:
            raise RuntimeError("Optional runtime inventory changed during scientific execution")
        verify_modules(("cuda",),config["optional_python_path"],config["optional_files"],
                       ("cuda",) if config.get("schema")==2 else None)

    def device_identity(self, session):
        """Correlate the actual owning PID/context UUID with allocation hardware."""
        local=self.config["device_family"]=="RTX3090"
        if local:
            require_local_lease(self.config["local_lease"])
            device=session.rt(session.driver.cuCtxGetDevice())
            raw=session.rt(session.driver.cuDeviceGetUuid(device))
            value=bytes(raw.bytes)
            if len(value)!=16:raise RuntimeError("Actual local CUDA UUID has the wrong width")
            uuid="GPU-"+str(uuid_module.UUID(bytes=value))
            if uuid!=self.config["local_lease"]["expected_uuid"]:
                raise RuntimeError("Actual CUDA context is not the leased local UUID")
        else:
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
        expected_name,expected_arch=("RTX 3090","compute_86") if local else ("H200","compute_90")
        if device is None or expected_name not in device[1] or session.arch != expected_arch:
            raise RuntimeError("Actual CUDA/NVML device is not the requested "+self.config["device_family"])
        return {"uuid": uuid, "name": device[1], "driver": device[2],
                "memory_MiB": int(device[3]), "pid": os.getpid(),
                "nvrtc_architecture": session.arch, "ptx_sha256": session.ptx_sha256,
                "isolation_policy":"cooperative local lease; desktop co-tenancy allowed" if local
                                   else "exclusive allocation; foreign compute contexts rejected"}

    @contextmanager
    def provider(self, primary, auxiliary):
        config = self.config
        record = {}
        self.report["plans"].append(record)
        session = owner = None
        primary_error = None
        try:
            self.require_device_admission()
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
