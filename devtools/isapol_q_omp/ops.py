#!/usr/bin/env python3
"""Private oracle/build operations only. No Psi4 import in this controller."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

OLD = Path("/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-build001")
BASE = Path("/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt")
PATCH = "5ae597cd9305588d14681d0da435834ef761c264"
OLD_MANIFEST = "d2b4049c2e5818e462ad3c6ac36f93d6a9623a6a205c701b885887ec2a9f3bfd"
OLD_ID = dict(core="ef69cddb2dd7b675688a733e4c5dd68b78fc78ac35b77b363114d638ac9dae44",
              isapol_python="e1476e2d0b8e395c40b48219ca8bc8def9e69035cab3ab970f6b88940bebc74b")
FROZEN = {
    "psi4/src/psi4/libisapol/partitioned_response.cc": "8bae8f9d90ef3544b10a6550443f2cd7ce67fd9fb4e3da1a3d51ed03e62718ab",
    "psi4/driver/procrouting/isapol_distribution.py": "0a0ae151df375f4317c5575f8f2c05856c13c8249e2d5371b244e6490fd2be6c",
    "tests/pytests/test_isapol_q_parallel.py": "af35ec5b40041c6d17cb5e4a1dd01301416e58ee5b5c2f0251da0c079d4c085c",
    "devtools/isapol_q_omp/README.md": "2b9fa27ba84e1de29df45ddd13f07919c87fdcab98ac43f49014b40b9155806b",
    "devtools/isapol_q_omp/test_workspace.py": "1789984f7c88c6740400913f50ab356f2ac5594c896ab26d3d8e59a07ab06be4",
}
OPS = {"devtools/isapol_q_omp/" + name for name in
       ("ops.py", "oracle.sbatch", "build.sbatch", "test_ops.py", "OPS.md")}
TEST = "tests/pytests/test_isapol_q_parallel.py"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with pending.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)  # Atomic, no-clobber publication on the same filesystem.
    pending.unlink()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def source_gate(source, commit, tree):
    need(git(source, "rev-parse", "HEAD") == commit, "source commit differs")
    need(git(source, "rev-parse", "HEAD^{tree}") == tree, "source tree differs")
    need(git(source, "rev-parse", "--show-toplevel") == str(source), "source must be Git root")
    need(not git(source, "status", "--porcelain", "--untracked-files=all"), "source must be clean")
    changed = set(git(source, "diff", "--name-only", PATCH, commit).splitlines())
    need(changed <= OPS, "ops child changed files outside the allowed new operations")
    for name, expected in FROZEN.items():
        need(sha(source / name) == expected, "frozen review file changed: " + name)


def fresh_paths(root, scratch, protected):
    need(not root.exists() and not scratch.exists(), "fresh nonexistent root/scratch required")
    paths = [root, scratch]
    for a in paths:
        need(a.is_absolute() and a != Path("/"), "absolute dedicated roots required")
        for b in protected + [p for p in paths if p != a]:
            need(not a.is_relative_to(b) and not b.is_relative_to(a), "protected/owned path overlap")


def old_seal():
    path = OLD / "build-identity.json"
    need(sha(path) == OLD_MANIFEST, "qualified old manifest changed")
    manifest = read(path)
    for name, expected in manifest["runtime_files"].items():
        need(sha(OLD / "env" / name) == expected, "qualified old runtime changed: " + name)
    for name, expected in manifest["external_runtime_files"].items():
        need(sha(name) == expected, "sealed original dependency changed: " + name)
    for name, expected in manifest["installed_files"].items():
        need(sha(Path(manifest["installed_package"]) / name) == expected, "old science changed: " + name)
    return manifest


def environment(prefix, scratch):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("PIP_") or key in (
            "PYTHONPATH", "PYTHONHOME", "PSIPATH", "PSIDATADIR", "LD_PRELOAD", "GCC_ROOT",
            "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH",
            "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CC", "CXX", "CMAKE_PREFIX_PATH",
            "CMAKE_TOOLCHAIN_FILE", "CMAKE_ARGS"):
            env.pop(key, None)
    env.update(PATH=f"{prefix}/bin:{BASE}/bin:/usr/bin:/bin",
               LD_LIBRARY_PATH=f"{prefix}/lib:{BASE}/lib",
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPYCACHEPREFIX=str(scratch / "cache"), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
               TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch))
    return env


class Runner:
    def __init__(self, root, cwd, env, seconds):
        self.root, self.cwd, self.env = root, cwd, env
        self.deadline, self.child, self.interrupted = time.monotonic()+seconds, None, False

    def stop(self, *unused):
        self.interrupted = True
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def check(self):
        if self.interrupted:
            raise InterruptedError("TERM/INT received")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("finite operation deadline exhausted")

    def run(self, label, command):
        self.check()
        write(self.root / (label + "-command.json"), list(map(str, command)))
        with (self.root / (label + ".stdout")).open("x") as out, \
             (self.root / (label + ".stderr")).open("x") as err:
            self.child = subprocess.Popen(list(map(str, command)), cwd=self.cwd, env=self.env,
                                          stdout=out, stderr=err, start_new_session=True)
            try:
                while True:
                    try:
                        code = self.child.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        self.check()
                self.check()
                need(code == 0, f"{label} exited {code}; inspect retained logs")
            except BaseException:
                self.stop()
                try:
                    self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    self.child.wait()
                raise
            finally:
                self.child = None
        return (self.root / (label + ".stdout")).read_text()


def clone_runtime(old, new, base):
    """Offline copy; no old writes, bytecode execution or install-through links."""
    need(not new.exists(), "clone prefix already exists")
    def ignore(directory, names):
        return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]
    shutil.copytree(old, new, symlinks=True, ignore=ignore)
    changes = []
    for path in sorted(new.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        if target.is_relative_to(old):
            replacement = new / target.relative_to(old)
            path.unlink()  # Newly owned clone only, never old/evidence.
            path.symlink_to(replacement, target_is_directory=target.is_dir())
            changes.append(dict(path=str(path), action="retarget_old_env", target=str(replacement)))
        elif not target.is_relative_to(new):
            need(target.is_relative_to(base), "clone link leaves original BASE dependencies")
            need(target.is_file(), "external directory link requires explicit parent review/materialization")
            path.unlink()
            shutil.copy2(target, path)  # Same dependency bytes, private destination.
            changes.append(dict(path=str(path), action="materialize_BASE_file", target=str(target)))
    for path in new.rglob("*"):
        if path.is_symlink():
            need(path.resolve(strict=True).is_relative_to(new), "clone retained external link")
    for path in new.rglob("*.pth"):
        need(str(old) not in path.read_text(), "old-env path injection in cloned .pth")
    return changes


def configure(record, source, root):
    args = list(record["configure"])
    required = ("-DCMAKE_BUILD_TYPE=Release", "-DENABLE_OPENMP=ON", "-DENABLE_XHOST=OFF",
                "-DBLAS_TYPE=MKL", "-DBUILD_Libint2_GENERATOR=OFF")
    need(all(flag in args for flag in required), "original configuration stack differs")
    for package in ("pybind11", "gau2grid", "Libint2", "Libxc", "qcelemental", "qcengine", "optking"):
        need("-DCMAKE_INSIST_FIND_PACKAGE_"+package+"=ON" in args, "dependency fallback/download forbidden")
    need(Path(args[0]).resolve() == (BASE / "bin/cmake").resolve(), "original CMake must use BASE")
    args[args.index("-S")+1], args[args.index("-B")+1] = str(source), str(root / "build")
    args = [arg.replace(str(OLD / "env"), str(root / "env")) for arg in args]
    need("-DCMAKE_INSTALL_PREFIX="+str(root / "env") in args, "candidate install prefix differs")
    need("-DPython_EXECUTABLE="+str(root / "env/bin/python") in args, "candidate Python differs")
    args += ["-DFETCHCONTENT_FULLY_DISCONNECTED=ON", "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    return args


def junit_gate(path, oracle):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    need(cases and not any(case.find(tag) is not None for case in cases
                          for tag in ("failure", "error", "skipped")), "native tests failed/skipped")
    qcases = [c for c in cases if "test_isapol_q_parallel" in c.get("classname", "").split(".")]
    need(len(qcases) == 32, "exact32 frozen-Q cases required")
    if oracle:
        need(len(cases) == 32, "oracle runs ONLY frozen-Q cases")
    else:
        for name in ("test_isapol_native_performance", "test_isapol_partitioned_response"):
            need(any(name in c.get("classname", "").split(".") for c in cases), "missing native regression: "+name)
    return dict(cases=len(cases), frozen_q_cases=32, failures=0, errors=0, skipped=0, sha256=sha(path))


def oracle_gate(path, expected_sha, source):
    need(sha(path) == expected_sha, "parent-bound oracle receipt digest differs")
    value = read(path)
    marker = read(Path(path).parent / "COMPLETED")
    need(marker == dict(exit_code=0, acceptance=False, mode="oracle"),
         "oracle wrapper did not complete successfully")
    need(value["status"] == "READY_ORACLE_NOT_ACCEPTANCE"
         and value["source"] == source and value["old_manifest_sha256"] == OLD_MANIFEST
         and value["old_runtime_unchanged"] is True and value["identity"]["implementation"] == OLD_ID
         and value["tests"]["frozen_q_cases"] == value["tests"]["cases"] == 32
         and all(value["tests"][k] == 0 for k in ("failures", "errors", "skipped")),
         "passing original fixed-operand oracle required")
    return value


def science_owned(name):
    parts = Path(name).parts
    return ("psi4" in parts or any(p.startswith("psi4-") and p.endswith(".dist-info") for p in parts)
            or name == "bin/psi4" or Path(name).name.startswith("libpsi4"))


def dependency_gate(prefix, manifest):
    for name, expected in manifest["runtime_files"].items():
        if not science_owned(name):
            need(sha(prefix / name) == expected, "candidate changed an original dependency: "+name)


IDENTITY = """import json,pathlib,psi4
from psi4.driver.procrouting.isapol_schema import implementation_identity
print('QOMP_IDENTITY='+json.dumps(dict(package=str(pathlib.Path(psi4.__file__).resolve()),
core=str(pathlib.Path(psi4.core.__file__).resolve()),version=str(psi4.__version__),
implementation=implementation_identity()['sha256'])))"""


def identity(runner, python, prefix, scratch, label):
    text = runner.run(label, [python, "-I", "-B", "-X", f"pycache_prefix={scratch}/cache", "-c", IDENTITY])
    rows = [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("QOMP_IDENTITY=")]
    need(len(rows) == 1, "identity child did not return exactly one record")
    value = json.loads(rows[0])
    need(Path(value["package"]).is_relative_to(prefix) and Path(value["core"]).is_relative_to(prefix),
         "foreign science interpreter/package/core")
    return value


def test_command(python, source, root, scratch, oracle):
    names = [TEST] if oracle else [TEST, "tests/pytests/test_isapol_native_performance.py",
                                  "tests/pytests/test_isapol_partitioned_response.py"]
    return [python, "-I", "-B", "-X", f"pycache_prefix={scratch}/cache", "-m", "pytest", "-q",
            f"--rootdir={source}", f"--basetemp={scratch}/pytest", "-o", f"cache_dir={root}/pytest-cache",
            f"--junitxml={root}/junit.xml"] + [source / name for name in names]


def allocation(mode):
    text = subprocess.check_output(["/opt/slurm/current/bin/scontrol", "show", "job", "-o",
                                    os.environ["SLURM_JOB_ID"]], text=True)
    row = dict(v.split("=", 1) for v in text.split() if "=" in v)
    expected = dict(Account="gts-cs207-chemx", QOS="embers", NumNodes="1", NumTasks="1",
                    NumCPUs="8", **{"CPUs/Task": "8"}, MinMemoryNode="32G",
                    TimeLimit="02:00:00" if mode == "oracle" else "08:00:00", Requeue="0", Restarts="0")
    need(all(row.get(k) == v for k, v in expected.items()), "actual allocation differs from finite contract")
    need(row.get("Partition") == "cpu-small", "actual partition differs")
    return row


def perform(args):
    source = dict(path=str(args.source), commit=args.commit, tree=args.tree, frozen_patch=PATCH)
    source_gate(args.source, args.commit, args.tree)
    if args.mode == "build":
        need(args.oracle_report and args.oracle_sha256, "build requires parent-bound passing oracle")
        oracle_gate(args.oracle_report, args.oracle_sha256, source)
    protected = [args.source, OLD, BASE]
    if args.oracle_report:
        protected.append(args.oracle_report.parent)
    fresh_paths(args.root, args.scratch_root, protected)
    need(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK") == "8",
         "parent's fresh eight-CPU allocation required")
    row = allocation(args.mode)
    args.root.mkdir(parents=True, mode=0o700)
    write(args.root / "owner.json", dict(token=args.launch_token, mode=args.mode))
    args.scratch_root.mkdir(parents=True, mode=0o700)
    cwd = args.root / "cwd"
    cwd.mkdir(mode=0o700)
    os.chdir(cwd)  # Includes all import-only Psi4 children and their finalizers.
    prefix = OLD / "env" if args.mode == "oracle" else args.root / "env"
    runner = Runner(args.root, cwd, environment(prefix, args.scratch_root),
                    6900 if args.mode == "oracle" else 28200)
    handlers = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    result = dict(schema="q-omp-ops/1", status="FAILED", acceptance=False, source=source,
                  old_manifest_sha256=OLD_MANIFEST, allocation=row, mode=args.mode,
                  root=str(args.root), scratch=str(args.scratch_root), launch_token=args.launch_token)
    manifest = None
    try:
        manifest = old_seal()
        runner.check()
        if args.mode == "build":
            write(args.root / "clone-links.json", clone_runtime(OLD / "env", prefix, BASE))
            dependency_gate(prefix, manifest)
            record_path = OLD / "build-record.json"
            record_sha = sha(record_path)
            command = configure(read(record_path), args.source, args.root)
            runner.run("configure", command)
            runner.run("build-install", [BASE / "bin/cmake", "--build", args.root / "build",
                                         "--target", "install", "-j", "2"])
            need(sha(record_path) == record_sha, "original configure record changed")
            result.update(original_configure_sha256=record_sha, configure=command,
                          oracle_receipt_sha256=args.oracle_sha256)
            dependency_gate(prefix, manifest)
            package = prefix / "lib/python3.13/site-packages/psi4"
            for name in manifest["installed_files"]:
                original_source = args.source / "psi4" / name
                if name.endswith(".py") and original_source.is_file():
                    need(sha(package / name) == sha(original_source), "candidate Python install differs: "+name)
        ident = identity(runner, prefix / "bin/python", prefix, args.scratch_root, "identity-before")
        if args.mode == "oracle":
            need(ident["implementation"] == OLD_ID, "original qualified implementation differs")
        else:
            need(all(ident["implementation"][k] != OLD_ID[k] for k in OLD_ID),
                 "candidate did not install new matching core/Python")
        result["identity"] = ident
        runner.run("native-tests", test_command(prefix / "bin/python", args.source, args.root,
                                               args.scratch_root, args.mode == "oracle"))
        result["tests"] = junit_gate(args.root / "junit.xml", args.mode == "oracle")
        need(identity(runner, prefix / "bin/python", prefix, args.scratch_root, "identity-after") == ident,
             "imported science changed during tests")
        if args.mode == "build":
            result["installed_science_sha256"] = {
                str(p.relative_to(prefix)): sha(p) for p in package.rglob("*")
                if p.is_file() and p.suffix in (".py", ".so")}
        runner.check()
        result["status"] = "READY_ORACLE_NOT_ACCEPTANCE" if args.mode == "oracle" else "READY_BUILD_NOT_QUALIFIED"
    except BaseException as error:
        result["exception"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            old_seal()
            result["old_runtime_unchanged"] = True
            source_gate(args.source, args.commit, args.tree)
            if args.mode == "build" and manifest is not None and prefix.exists():
                dependency_gate(prefix, manifest)
            if runner.interrupted:
                result["status"] = "FAILED"
        except BaseException as error:
            result.update(status="FAILED", integrity_error=repr(error))
            raise
        finally:
            write(args.root / "result.json", result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    need(not runner.interrupted, "interrupted operation cannot succeed")


def finalize(args):
    """Wrapper alone owns COMPLETED; never write into a preexisting foreign root."""
    status = args.exit_code
    try:
        owner = read(args.root / "owner.json")
        need(owner == dict(token=args.launch_token, mode=args.mode), "root owner/token differs")
        need(args.root.stat().st_uid == os.getuid(), "root not owned by this user")
    except Exception:
        return status or 1  # No writes when helper did not create our root.
    if status == 0:
        try:
            source_gate(args.source, args.commit, args.tree)
            value = read(args.root / "result.json")
            expected = "READY_ORACLE_NOT_ACCEPTANCE" if args.mode == "oracle" else "READY_BUILD_NOT_QUALIFIED"
            need(value["status"] == expected and value["old_runtime_unchanged"] is True
                 and value["launch_token"] == args.launch_token, "helper did not finish successfully")
        except Exception:
            status = 1
    marker = args.root / ("COMPLETED" if status == 0 else "FAILED")
    write(marker, dict(exit_code=status, acceptance=False, mode=args.mode))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("oracle", "build", "oracle-finalize", "build-finalize"))
    for name in ("source", "commit", "tree", "root", "scratch-root", "launch-token"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--oracle-report")
    p.add_argument("--oracle-sha256")
    p.add_argument("--exit-code", type=int)
    args = p.parse_args()
    args.mode = args.operation.split("-")[0]
    for name in ("source", "root", "scratch_root", "oracle_report"):
        if getattr(args, name):
            need(Path(getattr(args, name)).is_absolute(), "absolute CLI paths required")
            setattr(args, name, Path(getattr(args, name)).resolve())
    if args.operation.endswith("-finalize"):
        return finalize(args)
    perform(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
