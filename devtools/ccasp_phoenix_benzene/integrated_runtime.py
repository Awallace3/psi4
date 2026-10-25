# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Reviewed composition/admission inputs, not a build/install or an activation grant."""
import ctypes
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = json.loads(Path(globals().get("PLAN_PATH", HERE/"integrated_plan.json")).read_text())


def for_plan(path):
    """Explicit operational plan-path bootstrap; default capture users unchanged.

    A separate module namespace prevents changing any existing launcher's plan.
    No scientific import, package composition or numerical operation occurs here.
    """
    path = Path(path).resolve()
    if path.parent != HERE or path.suffix != ".json":
        raise ValueError("operational plan must be a tools-directory JSON file")
    spec = importlib.util.spec_from_file_location("_ccasp_runtime_"+path.stem, __file__)
    item = importlib.util.module_from_spec(spec)
    item.PLAN_PATH = path
    sys.modules[spec.name] = item
    spec.loader.exec_module(item)
    return item


def module(name, file):
    spec = importlib.util.spec_from_file_location(name,HERE/file)
    item = importlib.util.module_from_spec(spec)
    sys.modules[name] = item
    spec.loader.exec_module(item)
    return item


OLD = module("_ccasp_accepted_runtime","benzene_job.py")
BASIS = module("_ccasp_accepted_basis","basis_admission.py")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write(path, value):
    data = json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    if len(data) > 4*1024**2:
        raise ValueError("runtime receipt metadata bound")
    fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def pinned_json(path, pin):
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4*1024**2:
            raise ValueError("pinned JSON hash/type/size mismatch")
        data = stream.read(info.st_size+1)
    if len(data) != info.st_size or hashlib.sha256(data).hexdigest() != pin:
        raise ValueError("pinned JSON hash/type/size mismatch")
    return json.loads(data)


def compute_guard():
    host = socket.gethostname().split(".")[0]
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("USER") != "awallace43"
            or not os.environ.get("HOME","").startswith("/storage/home/")
            or host != PLAN["allocation"]["node"]
            or host != os.environ.get("SLURMD_NODENAME","").split(".")[0]
            or not sys.flags.isolated or not sys.flags.no_site
            or Path(sys.executable).resolve() != (OLD.ENV/"bin/python3.13").resolve()):
        raise ValueError("exact Phoenix compute node and isolated admitted Python required")


def effective_profile():
    """Optional explicit derivative profile; never replace the original probe profile."""
    file = PLAN.get("effective_profile_module")
    if file is None:
        return None
    return module("_ccasp_profile_"+Path(file).stem,file).configuration(sys.modules[__name__])[1]


def source_binding(commit, tree):
    OLD.identity(commit,tree)  # NEW driver root + unchanged native15cc/build receipts
    if ROOT != Path(PLAN["source"]):
        raise ValueError("unexpected new diagnostic source root")
    changed = OLD.git(ROOT,"diff","--name-only",OLD.BUILD_COMMIT,commit).splitlines()
    if any(not p.startswith("devtools/ccasp_phoenix_benzene/") and p not in PLAN["routing"] for p in changed):
        raise ValueError("source outside exact diagnostic routing allowlist")
    for subdir in ("psi4/src","psi4/share/psi4/basis"):
        if OLD.git(ROOT,"rev-parse",commit+":"+subdir) != OLD.git(ROOT,"rev-parse",OLD.BUILD_COMMIT+":"+subdir):
            raise ValueError("native/GBS source tree changed")
    for name,pin in PLAN["prior_driver_files"].items():
        if sha(HERE/name) != pin:
            raise ValueError("frozen scientific recipe/basis helper changed: "+name)
    binding = dict(source=str(ROOT),commit=commit,tree=tree,
        tools_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file()},
        routing_sha256={name:sha(ROOT/name) for name in PLAN["routing"]},
        native=dict(source=str(OLD.SOURCE),commit=OLD.BUILD_COMMIT,tree=OLD.BUILD_TREE,
                    stage=str(OLD.STAGE),core_sha256=OLD.CORE_SHA))
    profile = effective_profile()
    if profile is not None:
        binding["effective_profile"] = profile
    return binding


def regular_files(root, limit=5000, *, ignore_bytecode=True):
    files = []
    def failed(error):
        raise error
    for parent, dirs, names in os.walk(root,followlinks=False,onerror=failed):
        if any((Path(parent)/d).is_symlink() for d in dirs):
            raise ValueError("unexpected runtime directory symlink")
        if ignore_bytecode and "__pycache__" in dirs:
            dirs.remove("__pycache__")
        for name in names:
            path = Path(parent)/name
            if ignore_bytecode and path.suffix == ".pyc":
                continue
            files.append(path)
            if len(files) > limit:
                raise ValueError("runtime file count bound")
    return sorted(files)


def committed_driver_inventory(source, commit):
    """Independent Git object inventory, never inferred from walk or manifest."""
    data = subprocess.check_output(["git","-C",str(source),"ls-tree","-rz",commit,"--","psi4/driver"])
    if len(data) > 1024**2 or not data.endswith(b"\0"):
        raise ValueError("committed driver inventory metadata bound/format")
    records = {}
    for record in data.split(b"\0")[:-1]:
        header, path = record.split(b"\t",1)
        mode, kind, oid = header.decode("ascii").split()
        path = path.decode("utf-8")
        if not path.endswith(".py"):
            continue  # deliberately only Python source; bytecode is not source
        if (mode not in ("100644","100755") or kind != "blob"
                or not re.fullmatch("[0-9a-f]{40}",oid)
                or not path.startswith("psi4/driver/") or ".." in Path(path).parts):
            raise ValueError("committed driver inventory object/path mismatch")
        relative = Path(path.removeprefix("psi4/"))
        if relative in records or len(records) >= 5000:
            raise ValueError("committed driver inventory duplicate/count bound")
        records[relative] = oid
    if not records:
        raise ValueError("empty committed driver Python inventory")
    return records


def driver_sources(source, inventory):
    root = Path(source)/"psi4/driver"
    actual = {Path("driver")/file.relative_to(root):file
              for file in regular_files(root) if file.suffix==".py"}
    if set(actual) != set(inventory):
        raise ValueError("committed driver Python inventory differs from source files")
    return actual


def small_regular(path, remaining, *, destination=None):
    """Bounded no-follow/nonblocking regular read/copy, including configured launcher.

    No destination is opened until descriptor type/size is admitted. A changing
    input fails closed and leaves any partial exclusive output unreused.
    """
    with ExitStack() as stack:
        fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        inp = stack.enter_context(os.fdopen(fd,"rb"))
        info = os.fstat(inp.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("small runtime input must be a regular file")
        if info.st_size > remaining or remaining < 0:
            raise ValueError("small Python runtime composition byte bound")
        out = None if destination is None else stack.enter_context(Path(destination).open("xb"))
        digest = hashlib.sha256()
        blob = hashlib.sha1(b"blob "+str(info.st_size).encode()+b"\0")
        count = 0
        while chunk := inp.read(65536):
            count += len(chunk)
            if count > info.st_size:
                raise ValueError("small runtime input grew during read/copy")
            digest.update(chunk); blob.update(chunk)
            if out is not None:
                out.write(chunk)
        after = os.fstat(inp.fileno())
        if (count != info.st_size or after.st_size != info.st_size
                or after.st_mtime_ns != info.st_mtime_ns):
            raise ValueError("small runtime input changed during read/copy")
        if out is not None:
            out.flush(); os.fsync(out.fileno())
        return dict(nbytes=count,sha256=digest.hexdigest(),git_blob_sha1=blob.hexdigest())


def compose(source, accepted, stage, binding):
    """Small Python/configured-file composition only. No core copy/hash/import."""
    source,accepted,stage = map(Path,(source,accepted,stage))
    stage.mkdir(mode=0o700,parents=True,exist_ok=False)
    package = stage/"lib/psi4"
    inherited = accepted/"lib/psi4"
    inventory = committed_driver_inventory(source,binding["commit"])
    current_sources = driver_sources(source,inventory)
    core_name = "core.cpython-313-x86_64-linux-gnu.so"
    records, total = {}, 0
    candidates = {}
    for old in regular_files(inherited):
        relative = old.relative_to(inherited)
        if str(relative).startswith("driver/") and old.suffix == ".py":
            continue
        candidates[relative] = (old,"inherited configured package")
    for relative,current in current_sources.items():
        candidates[relative] = (current,"new committed Python tree")
    for relative,(origin,role) in sorted(candidates.items()):
        destination = package/relative
        destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if str(relative) == core_name:
            destination.symlink_to(origin)
            records[str(destination.relative_to(stage))] = dict(role="accepted native15cc core link",
                link=str(origin),sha256=binding["native"]["core_sha256"])
            continue
        if ".so" in origin.name:
            raise ValueError("unexpected inherited runtime object: "+str(origin))
        copied = small_regular(origin,32*1024**2-total,destination=destination)
        total += copied["nbytes"]
        if relative in inventory and copied["git_blob_sha1"] != inventory[relative]:
            raise ValueError("committed driver Python blob identity differs")
        destination.chmod(0o400)
        records[str(destination.relative_to(stage))] = dict(role=role,source=str(origin),
            sha256=copied["sha256"],nbytes=copied["nbytes"])
        if sha(destination) != copied["sha256"]:
            raise ValueError("runtime copy differs")
    if not (package/core_name).is_symlink():
        raise ValueError("accepted core missing")
    (stage/"bin").mkdir()
    copied = small_regular(accepted/"bin/psi4",32*1024**2-total,destination=stage/"bin/psi4")
    total += copied["nbytes"]
    (stage/"bin/psi4").chmod(0o700)
    records["bin/psi4"] = dict(role="inherited configured relocatable launcher",
        source=str(accepted/"bin/psi4"),sha256=copied["sha256"],nbytes=copied["nbytes"])
    (stage/"share").mkdir()
    (stage/"share/psi4").symlink_to(accepted/"share/psi4",target_is_directory=True)
    return dict(schema="ccasp-composed-python-runtime/1",stage=str(stage),binding=binding,
        records=records,data_link=str(accepted/"share/psi4"),copied_bytes=total,
        native_compiled_from_new_commit=False,
        core_hash_verified_during_composition=False,
        scope="configured root inherited; complete new driver Python; accepted native core/data separate")


def verify_runtime(path, pin, binding, *, hash_core=False):
    manifest = pinned_json(path,pin)
    if manifest["schema"] != "ccasp-composed-python-runtime/1" or manifest["binding"] != binding:
        raise ValueError("runtime source/driver binding differs")
    stage = Path(manifest["stage"])
    if stage != Path(PLAN["runtime"])/"stage":
        raise ValueError("wrong composed stage")
    expected = manifest["records"]
    inventory = committed_driver_inventory(Path(binding["source"]),binding["commit"])
    driver_sources(binding["source"],inventory)  # independent of the composed manifest
    if (set(p.name for p in stage.iterdir()) != {"lib","bin","share"}
            or set(p.name for p in (stage/"lib").iterdir()) != {"psi4"}
            or set(p.name for p in (stage/"bin").iterdir()) != {"psi4"}
            or set(p.name for p in (stage/"share").iterdir()) != {"psi4"}):
        raise ValueError("unexpected composed prefix content")
    actual = {str(p.relative_to(stage)) for p in regular_files(stage/"lib/psi4",ignore_bytecode=False)}
    actual.add("bin/psi4")
    expected_python = {"lib/psi4/"+str(rel) for rel in inventory}
    actual_python = {rel for rel in actual if rel.startswith("lib/psi4/driver/") and rel.endswith(".py")}
    manifest_python = {rel for rel in expected if rel.startswith("lib/psi4/driver/") and rel.endswith(".py")}
    if actual_python != expected_python or manifest_python != expected_python:
        raise ValueError("committed driver Python inventory differs from runtime/manifest")
    if actual != set(expected):
        raise ValueError("unexpected/missing runtime package file")
    total = 0
    for rel,record in expected.items():
        file = stage/rel
        if "link" in record:
            if record["sha256"] != binding["native"]["core_sha256"] or record["sha256"] != OLD.CORE_SHA:
                raise ValueError("inconsistent native core hash declaration")
            if not file.is_symlink() or os.readlink(file) != record["link"]:
                raise ValueError("wrong accepted core link")
            if file.resolve() != (OLD.STAGE/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so").resolve():
                raise ValueError("wrong resolved native core")
            if hash_core and sha(file) != OLD.CORE_SHA:
                raise ValueError("accepted core hash differs")
        else:
            staged = small_regular(file,32*1024**2-total)
            original = small_regular(record["source"],32*1024**2-total)
            total += staged["nbytes"]
            if (staged["sha256"] != record["sha256"] or original["sha256"] != record["sha256"]
                    or staged["nbytes"] != record["nbytes"] or original["nbytes"] != record["nbytes"]):
                raise ValueError("runtime Python/configured file hash differs")
            if (Path(record["source"]) != (
                    Path(binding["source"])/rel.removeprefix("lib/")
                    if rel.startswith("lib/psi4/driver/") and rel.endswith(".py")
                    else OLD.STAGE/rel)):
                raise ValueError("unexpected runtime source role")
            relative = Path(rel.removeprefix("lib/psi4/"))
            if relative in inventory and staged["git_blob_sha1"] != inventory[relative]:
                raise ValueError("committed driver Python blob identity differs")
    if total != manifest["copied_bytes"]:
        raise ValueError("runtime copied byte declaration differs")
    data = stage/"share/psi4"
    if (not data.is_symlink() or os.readlink(data) != manifest["data_link"]
            or data.resolve() != (OLD.STAGE/"share/psi4").resolve()):
        raise ValueError("wrong accepted data root")
    return manifest


def prior_probe():
    old = pinned_json(PLAN["prior_probe"],PLAN["prior_probe_sha256"])
    cfg = json.loads((HERE/"pilot.json").read_text())
    expected = dict(native=dict(source=str(OLD.SOURCE),commit=OLD.BUILD_COMMIT,tree=OLD.BUILD_TREE,
                               stage=str(OLD.STAGE),core_sha256=OLD.CORE_SHA),
        driver=dict(source=PLAN["prior_driver_source"],commit=PLAN["prior_driver_commit"],
                    tree=PLAN["prior_driver_tree"],files_sha256=PLAN["prior_driver_files"]),
        fixture_sha256=cfg["fixture_sha256"],config_sha256=sha(HERE/"pilot.json"))
    if (old.get("schema") != "ccasp-benzene-native-basis-admission/1"
            or old.get("status") != "basis_metadata_only_complete" or old.get("no_scf") is not True
            or old["binding"] != expected):
        raise ValueError("entire original private probe binding differs")
    for role,pin in PLAN["fingerprints"].items():
        snap = old["admission"][role]
        if (snap["fingerprint_sha256"] != pin
                or BASIS.digest({k:v for k,v in snap.items() if k!="fingerprint_sha256"}) != pin):
            raise ValueError("corrupt original normalized shell fingerprint")
    return old


def derived_document(binding, runtime_sha):
    old = prior_probe()
    document = dict(schema="ccasp-benzene-derived-basis-admission/1",
        original_probe=dict(path=PLAN["prior_probe"],sha256=PLAN["prior_probe_sha256"],
                            binding=old["binding"],fingerprints=PLAN["fingerprints"]),
        new_binding=binding,runtime_manifest_sha256=runtime_sha,
        status="derivation_from_accepted_measurement_NOT_a_new_probe",
        semantics="same native/core/GBS/fixture/config/basis helper; new Python diagnostics and driver explicit")
    profile = effective_profile()
    if profile is not None:
        if binding.get("effective_profile") != profile:
            raise ValueError("effective profile source binding differs")
        document.update(effective_profile=profile,
            semantics="same native/core/GBS/fixture/bases/frame/helper; original measurement identity retained; "
                      "explicit Q-grid and numerical-memory overrides are NOT a new basis measurement")
    return document


def admit_basis(core, options, molecule, receipt, binding, runtime_sha, derivative_path,
                derivative_sha, recipe_builder):
    expected = derived_document(binding,runtime_sha)
    if pinned_json(derivative_path,derivative_sha) != expected:
        raise ValueError("reviewed derivative admission binding differs")
    old = prior_probe()
    BASIS.basis_files(OLD.SOURCE,OLD.STAGE,receipt)  # preserve original physical data records
    main,aux = BASIS.build_bases(core,options,molecule,receipt)
    BASIS.compare_manifest(receipt,old,old["binding"])
    receipt["derivative_admission"] = dict(original_probe_sha256=PLAN["prior_probe_sha256"],
        derivative_sha256=derivative_sha,new_binding=binding,runtime_manifest_sha256=runtime_sha,
        basis_measurement_repeated_here_before_scf=True, no_extra_probe_job=True)
    return main,recipe_builder(aux)


def initialize(binding, manifest_path, manifest_sha, run, retained_handles):
    compute_guard()
    manifest = verify_runtime(manifest_path,manifest_sha,binding,hash_core=True)
    stage = Path(manifest["stage"])
    if sha(OLD.OLD/"lib/libEinsums.so.1.1.3") != OLD.PLAN["einsums_library_sha256"]:
        raise ValueError("accepted Einsums hash differs")
    # Retain this exact handle through final closure in the SAME interpreter.
    handle = ctypes.CDLL(str(OLD.OLD/"lib/libEinsums.so.1.1.3"),mode=os.RTLD_NOW|os.RTLD_GLOBAL)
    retained_handles.append(handle)  # retained even if a later initialization guard refuses
    OLD.snapshot("preloaded",full=False,receipt_run=run)
    command = [str(OLD.ENV/"bin/python3.13"),"-I","-S",str(stage/"bin/psi4"),"--psiapi"]
    text = subprocess.check_output(command,text=True).strip()
    expected = f"export PATH={stage/'bin'}:$PATH PYTHONPATH={stage/'lib'}:$PYTHONPATH"
    if text != expected:
        raise ValueError("composed staged --psiapi output differs")
    os.environ["PYTHONPATH"] = str(stage/"lib")+":"
    os.environ["PATH"] = str(stage/"bin")+":"+os.environ["PATH"]
    os.environ["PSIDATADIR"] = str(OLD.STAGE/"share/psi4")
    sys.path[:0] = [str(stage/"lib"),str(OLD.DEPS/"lib"),str(OLD.DEPS/"python"),
                   str(OLD.ENV/"lib/python3.13/site-packages")]
    import numpy, adcc, einsums, psi4
    if (Path(psi4.__file__).resolve() != (stage/"lib/psi4/__init__.py").resolve()
            or Path(psi4.core.__file__).resolve() != (OLD.STAGE/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so").resolve()
            or Path(psi4.core.get_datadir()).resolve() != (OLD.STAGE/"share/psi4").resolve()
            or Path(einsums.__file__).resolve() != (OLD.OLD/"lib/einsums/__init__.py").resolve()
            or Path(einsums.core.__file__).resolve() != (OLD.OLD/"lib/einsums/core.cpython-313-x86_64-linux-gnu.so").resolve()):
        raise ValueError("composed package/native/data/dependency role differs")
    pins = json.loads((OLD.BUILD_RUN/"metadata/dependency-hashes.json").read_text())
    for path,digest in pins.items():
        if sha(path) != digest:
            raise ValueError("accepted dependency changed")
    import importlib
    for name,version in OLD.PLAN["python_dependencies"].items():
        package = importlib.import_module(name)
        if (package.__version__ != version
                or not Path(package.__file__).resolve().is_relative_to((OLD.OLD/"lib"/name).resolve())):
            raise ValueError("isolated QC dependency differs")
    for rel,digest in binding["routing_sha256"].items():
        name = rel.removesuffix(".py").replace("/",".")
        package = importlib.import_module(name)
        expected_path = stage/"lib"/rel
        if Path(package.__file__).resolve() != expected_path.resolve() or sha(package.__file__) != digest:
            raise ValueError("loaded diagnostic routing identity differs")
    adcc.set_n_threads(1); psi4.set_num_threads(1)
    if adcc.get_n_threads()!=1 or psi4.core.get_num_threads()!=1:
        raise ValueError("single numerical thread admission failed")
    OLD.snapshot("before-capture",receipt_run=run)
    write(Path(run)/"metadata/runtime.json",dict(binding=binding,stage=str(stage),
        runtime_manifest_sha256=manifest_sha,psiapi_argv=command,psiapi_output=text,
        package=psi4.__file__,core=psi4.core.__file__,data=psi4.core.get_datadir(),
        python=sys.executable,numpy=numpy.__version__,adcc=adcc.__version__,
        node=socket.gethostname(),einsums_handle_retained=True))
    return psi4,handle
