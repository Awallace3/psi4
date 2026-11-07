#!/usr/bin/env python3
"""Read-only validation of build001; never configure, install or repair it."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

_spec = importlib.util.spec_from_file_location("q_ops", Path(__file__).with_name("ops.py"))
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)

COMPILED_COMMIT = "5f83eb146986cb0e6a7113d9a9374d8ca3e5e610"
COMPILED_TREE = "8a8a18d326342d38214a1fde488a95abfe013955"
CANDIDATE = ops.OLD.parent / "ccasp-native-q-omp-build001"
ORACLE_SHA = "99aaebf6b5be0be816e84fbde521503c9b9f323dca28503a1b9521d3d09538da"
METADATA_SHA = "31a3fec8334714b53fed8b21fb8e9c15c65620f58d7b8b253c523cbe2a07e6a8"
BASIS = "source_control_flow_after_runner_checks_not_stored_exit_codes"
ARTIFACTS = ("result.json", "FAILED", "configure-command.json", "configure.stdout",
             "configure.stderr", "build-install-command.json", "build-install.stdout",
             "build-install.stderr")
KNOWN_ARTIFACTS = {
    "result.json": "00e8625149499f2e13256b8d1fd0165f7ca0768e415b02a2dbaf1d7e554b90d2",
    "FAILED": "aad057a9ed4547be11cd6a2119f93b5a43f36b6b527a306586d59f76cd030ea8",
    "configure-command.json": "51aafb81d98b6b23824f209743934866f4c1040286ed74a7ae2218ab8b04a424",
    "build-install-command.json": "3aa97853278df1b11727a99b49a6fbba6aae126ba9e1daae54aea59b040e05f0",
}
GENERATED = ("__init__.py", "extras.py", "metadata.py")
PACKAGE = Path("lib/python3.13/site-packages/psi4")


def owned_file(root, name):
    rel = Path(name)
    ops.need(not rel.is_absolute() and ".." not in rel.parts, "unsafe artifact relative path")
    path = root / rel
    ops.need(path.is_file() and path.resolve().is_relative_to(root.resolve()),
             "missing/outside compiled artifact: "+name)
    return path


def compiled_source(args):
    return dict(path=str(args.compiled_source), commit=COMPILED_COMMIT, tree=COMPILED_TREE,
                frozen_patch=ops.PATCH)


def evidence_gate(args):
    """Historical failure remains a failure. No command exit receipt existed."""
    ops.need(args.candidate_root == CANDIDATE, "only existing compiled build001 is authorized")
    ops.source_gate(args.compiled_source, COMPILED_COMMIT, COMPILED_TREE)
    ops.need(ops.sha(args.evidence) == args.evidence_sha256, "parent-bound evidence digest differs")
    proof = ops.read(args.evidence)
    ops.need(proof["schema"] == "q-omp-compiled-evidence/1"
             and proof["candidate_root"] == str(args.candidate_root)
             and proof["compiled_source"] == compiled_source(args)
             and proof["historical_job"] == dict(id="15280334", status="FAILED", exit_code="1:0")
             and proof["command_completion_basis"] == BASIS, "compiled evidence attribution differs")
    ops.need(set(proof["artifact_sha256"]) == set(ARTIFACTS), "incomplete build artifact binding")
    ops.need(all(proof["artifact_sha256"][name] == digest
                 for name, digest in KNOWN_ARTIFACTS.items()), "known historical artifact pin differs")
    for name, digest in proof["artifact_sha256"].items():
        ops.need(ops.sha(owned_file(args.candidate_root, name)) == digest,
                 "historical artifact changed: "+name)
    ops.need(not (args.candidate_root/"COMPLETED").exists(), "historical failed build relabeled")
    failed = ops.read(args.candidate_root/"FAILED")
    ops.need(failed == dict(exit_code=1, acceptance=False, mode="build"), "original FAILED differs")
    result = ops.read(args.candidate_root/"result.json")
    ops.need(result["status"] == "FAILED" and result["mode"] == "build"
             and result["acceptance"] is False and result["source"] == compiled_source(args)
             and result["root"] == str(args.candidate_root)
             and result["old_manifest_sha256"] == ops.OLD_MANIFEST
             and result["original_configure_sha256"] == ops.APPROVED_RECORD_SHA
             and result["oracle_receipt_sha256"] == ORACLE_SHA
             and result["exception"] == dict(type="ValueError",
                                            message="candidate Python install differs: metadata.py")
             and "integrity_error" not in result and "tests" not in result,
             "historical failure is not the reviewed post-install metadata check")
    record = ops.approved_record(ops.OLD/"build-record.json")
    command = ops.configure(record, args.compiled_source, args.candidate_root)
    ops.need(result["configure"] == command
             and ops.read(args.candidate_root/"configure-command.json") == command,
             "historical configure argv differs")
    install = [str(ops.BASE/"bin/cmake"), "--build", str(args.candidate_root/"build"),
               "--target", "install", "-j", "2"]
    ops.need(ops.read(args.candidate_root/"build-install-command.json") == install,
             "historical build/install argv differs")
    trace = Path(proof["traceback"]["path"])
    ops.need(trace.is_absolute() and ops.sha(trace) == proof["traceback"]["sha256"],
             "parent-bound captured traceback differs")
    text = trace.read_text()
    ops.need(str(args.compiled_source/"devtools/isapol_q_omp/ops.py") in text
             and "in installed_python_gate" in text
             and "ValueError: candidate Python install differs: metadata.py" in text,
             "captured traceback does not establish reviewed control-flow location")
    # In pinned5f83, this precise exception is after both Runner.run calls;
    # each checks code == 0 before returning. These are inferred zeros, NOT
    # stored command-exit receipts and NOT a successful historical build job.
    return proof


def tree_snapshot(prefix, check, link_roots=()):
    """Entire historical tree namespace/bytes, not just selected runtime files."""
    snapshot = {}
    for path in sorted(prefix.rglob("*")):
        check()
        name = str(path.relative_to(prefix))
        if path.is_symlink():
            target = path.resolve(strict=True)
            ops.need(any(target.is_relative_to(root) for root in (prefix, *link_roots)),
                     "candidate link leaves protected trees")
            snapshot[name] = dict(link=os.readlink(path), resolved=str(target))
            if target.is_file():
                snapshot[name]["sha256"] = ops.sha(target)
        elif path.is_file():
            snapshot[name] = dict(sha256=ops.sha(path))
        elif path.is_dir():
            snapshot[name] = dict(directory=True)
        else:
            raise ValueError("unsupported prefix entry: "+name)
    return snapshot


def candidate_gate(args, proof, manifest):
    root = args.candidate_root
    package = root/"env"/PACKAGE
    for path in (root/"env").rglob("*"):
        if path.is_symlink():
            ops.need(path.resolve(strict=True).is_relative_to(root/"env"), "candidate link leaves prefix")
    ops.dependency_gate(root/"env", manifest)
    names = set(manifest["installed_files"]) | {
        str(p.relative_to(package)) for p in package.rglob("*.py")} | {
        str(p.relative_to(args.compiled_source/"psi4"))
        for p in (args.compiled_source/"psi4/driver/procrouting").glob("isapol*.py")}
    checks = ops.installed_python_gate(args.compiled_source, root/"build", package, names)
    inner = root/"build/psi4-core-prefix/src/psi4-core-build"
    for name in GENERATED:
        digest = ops.sha(owned_file(inner, name))
        staged = owned_file(root/"build/stage"/PACKAGE, name)
        ops.need(ops.sha(staged) == digest == ops.sha(owned_file(package, name)),
                 "generated/staged/installed bytes differ: "+name)
    ops.need(ops.sha(inner/"metadata.py") == METADATA_SHA, "reviewed metadata bytes differ")
    core = proof["core"]
    installed = owned_file(root/"env", core["env_relative"])
    staged = owned_file(root/"build/stage", core["env_relative"])
    built = owned_file(inner, core["inner_build_relative"])
    ops.need(installed.parent == package and installed.name.startswith("core.")
             and installed.suffix == ".so" and core["sha256"] != ops.OLD_ID["core"],
             "candidate core path/identity differs")
    ops.need(ops.sha(installed) == ops.sha(staged) == ops.sha(built) == core["sha256"],
             "compiled/staged/installed core differs")
    for cache, source, prefix in (
        (root/"build/CMakeCache.txt", args.compiled_source, root/"env"),
        (inner/"CMakeCache.txt", args.compiled_source/"psi4", root/"build/stage"),
    ):
        values = {}
        for line in cache.read_text().splitlines():
            if not line or line.startswith(("//", "#")):
                continue
            key, value = line.split("=", 1)
            key = key.split(":", 1)[0]
            ops.need(key not in values, "duplicate CMake cache key")
            values[key] = value
        required = dict(CMAKE_HOME_DIRECTORY=str(source), CMAKE_INSTALL_PREFIX=str(prefix),
                        CMAKE_CXX_COMPILER=str(ops.BASE/"bin/x86_64-conda-linux-gnu-c++"),
                        ENABLE_OPENMP="ON")
        if cache == root/"build/CMakeCache.txt":
            required["ENABLE_XHOST"] = "OFF"  # Outer-only option; not forwarded to core.
        ops.need(all(values.get(k) == v for k, v in required.items()),
                 "compiled CMake cache source/prefix/compiler/OpenMP differs")
        ops.need("-march=x86-64-v3" in values.get("CMAKE_CXX_FLAGS", "").split(),
                 "compiled CPU portability flags differ")
    # All installation-bound Python must come from source or the explicit
    # generated whitelist, rather than a Python digest merely differing from OLD.
    for path in package.rglob("*.py"):
        rel = str(path.relative_to(package))
        if rel.startswith("tests/"):
            continue  # Installed tests are not the executed frozen source tests.
        ops.need(rel in checks, "unvalidated installed Python: "+rel)
    return checks, installed


def source_python_sha(source):
    """Mirror schema implementation_identity's file list without importing it."""
    driver = source/"psi4/driver"
    routing = driver/"procrouting"
    paths = list(routing.glob("isapol_*.py")) + [
        driver/"schema_wrapper.py", driver/"driver.py", driver/"task_planner.py",
        driver/"p4util/util.py", routing/"proc.py"]
    modules = {str(p.relative_to(driver)): ops.sha(p) for p in sorted(paths)}
    ops.need("procrouting/isapol_schema.py" in modules, "source identity is incomplete")
    return hashlib.sha256(json.dumps(modules, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def perform(args):
    ops.source_gate(args.source, args.commit, args.tree)
    # Before creating outputs, protect the proof/trace, compiled source/prefix
    # and historical artifacts against overlap with any newly writable roots.
    proof = evidence_gate(args)
    trace = Path(proof["traceback"]["path"]).resolve()
    ops.fresh_paths(args.root, args.scratch_root,
                    [args.source, args.compiled_source, args.candidate_root, ops.OLD, ops.BASE,
                     args.evidence.parent, trace.parent])
    ops.need(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK") == "8",
             "fresh eight-CPU allocation required")
    allocation = ops.allocation("validate")
    args.root.mkdir(parents=True, mode=0o700)
    ops.write(args.root/"owner.json", dict(token=args.launch_token, mode="validate"))
    args.scratch_root.mkdir(parents=True, mode=0o700)
    cwd = args.root/"cwd"
    cwd.mkdir(mode=0o700)
    os.chdir(cwd)
    prefix = args.candidate_root/"env"
    runner = ops.Runner(args.root, cwd, ops.environment(prefix, args.scratch_root), 6900)
    handlers = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    result = dict(schema="q-omp-diagnostic-validation/1", status="FAILED", mode="validate",
                  acceptance=False, adoption=False, launch_token=args.launch_token,
                  source=dict(path=str(args.source), commit=args.commit, tree=args.tree),
                  compiled_source=compiled_source(args), candidate_root=str(args.candidate_root),
                  allocation=allocation, evidence_sha256=args.evidence_sha256,
                  historical_build=dict(status="FAILED", job="15280334", exit_code="1:0",
                                        command_completion_basis=BASIS,
                                        inferred_command_exit_codes={"configure": 0, "build-install": 0}))
    before = None
    try:
        manifest = ops.old_seal()  # Actual independent precheck, not historical assertion.
        result["old_runtime_prechecked"] = True
        before = tree_snapshot(args.candidate_root, runner.check, (args.compiled_source, ops.BASE))
        ops.write(args.root/"compiled-tree-before.json", before)
        result["installed_python_checks"], core_path = candidate_gate(args, proof, manifest)
        ident = ops.identity(runner, prefix/"bin/python", prefix, args.scratch_root, "identity-before")
        ops.need(Path(ident["core"]).resolve() == core_path.resolve()
                 and ident["implementation"]["core"] == proof["core"]["sha256"]
                 and ident["implementation"]["isapol_python"] == source_python_sha(args.compiled_source)
                 and ident["implementation"]["isapol_python"] != ops.OLD_ID["isapol_python"],
                 "imported candidate identity differs")
        result["identity"] = ident
        runner.run("native-tests", ops.test_command(prefix/"bin/python", args.compiled_source,
                                                   args.root, args.scratch_root, False))
        result["tests"] = ops.junit_gate(args.root/"junit.xml", False)
        ops.need(ops.identity(runner, prefix/"bin/python", prefix, args.scratch_root,
                              "identity-after") == ident, "candidate identity changed during tests")
        runner.check()
        result["status"] = "READY_DIAGNOSTIC_NOT_ACCEPTANCE"
    except BaseException as error:
        result["exception"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            try:
                if before is not None:
                    # Cleanup hashes remain necessary on interruption; no new child.
                    after = tree_snapshot(args.candidate_root, lambda: None,
                                          (args.compiled_source, ops.BASE))
                    ops.write(args.root/"compiled-tree-after.json", after)
                    ops.need(before == after, "read-only compiled tree changed")
                    result["candidate_prefix_unchanged"] = True
                    result["compiled_tree_unchanged"] = True
            finally:
                # Attempt the qualified-original postcheck even if candidate
                # evidence changed or the native child failed.
                manifest = ops.old_seal()
                result["old_runtime_unchanged"] = True
                ops.source_gate(args.source, args.commit, args.tree)
                evidence_gate(args)
                if before is not None:
                    candidate_gate(args, proof, manifest)
            if runner.interrupted:
                result["status"] = "FAILED"
        except BaseException as error:
            result.update(status="FAILED", integrity_error=repr(error))
            raise
        finally:
            ops.write(args.root/"result.json", result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    ops.need(not runner.interrupted, "interrupted validation cannot succeed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("validate", "validate-finalize"))
    for name in ("source", "commit", "tree", "root", "scratch-root", "launch-token",
                 "compiled-source", "candidate-root", "evidence", "evidence-sha256"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--exit-code", type=int)
    args = p.parse_args()
    args.mode = "validate"
    for name in ("source", "root", "scratch_root", "compiled_source", "candidate_root", "evidence"):
        path = Path(getattr(args, name))
        ops.need(path.is_absolute(), "absolute CLI paths required")
        setattr(args, name, path.resolve())
    if args.operation == "validate-finalize":
        return ops.finalize(args)
    perform(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
