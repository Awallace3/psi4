"""Offline metadata/configure fixtures only. No CUDA, native build or allocation."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "devtools/disp0"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    old = list(sys.path)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    module = load("runtime_contract")
    source = tmp_path / "source13"
    for name in ("bin", "include", "lib64", "nvvm/bin", "nvvm/libdevice"):
        (source / name).mkdir(parents=True, exist_ok=True)
    for name in module.HEADERS:
        (source / "include" / name).parent.mkdir(parents=True, exist_ok=True)
        (source / "include" / name).write_text("// fixture header\n")
    (source / "include/cuda.h").write_text("#define CUDA_VERSION 13030\n")
    (source / "include/cuda_runtime_api.h").write_text("#define CUDART_VERSION 13030\n")
    for name in module.SUPPORT:
        path = source / name
        path.write_text("#!/bin/sh\nprintf 'fixture V13.3.73\\n'\n")
        path.chmod(0o700)
    for family, major in module.COMPONENTS.items():
        info = {"soname": f"lib{family}.so.{major}", "needed": [],
                "rpath": [], "runpath": []}
        (source / "lib64" / f"lib{family}.so.{major}.1").write_text(json.dumps(info))
    for name in ("libcudart_static.a", "libcudadevrt.a"):
        (source / "lib64" / name).write_text("fixture archive, never linked\n")
    # Explicit unit-test injection, NOT a production CLI bypass/mock mode.
    monkeypatch.setattr(module, "elf", lambda path: json.loads(Path(path).read_text()))
    manifest = module.inspect_sdk(source)
    return module, source, manifest


def test_complete_private_sdk_supplies_only_correct_developer_aliases(sdk, tmp_path):
    module, source, manifest = sdk
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    private = tmp_path / "private13"
    record = module.prepare_sdk(source, private, manifest)
    assert record["private_prefix"] == str(private)
    assert (private / "sdk-contract.json").is_file()
    for family, item in manifest["libraries"].items():
        assert (private / "lib64" / f"lib{family}.so").resolve() == Path(item["path"])
        assert not (source / "lib64" / f"lib{family}.so").exists()
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        module.prepare_sdk(source, private, manifest)


@pytest.mark.parametrize("corrupt", ["missing-header", "missing-lib", "ambiguous", "cuda12-needed",
                                    "wrong-soname", "mixed-inventory", "wrong-version",
                                    "conflicting-header", "changed-input", "site12", "missing-transitive"])
def test_incomplete_mixed_or_changed_sdk_fails_before_materialization(sdk, tmp_path, corrupt):
    module, source, manifest = sdk
    family = source / "lib64/libnvJitLink.so.13.1"
    if corrupt == "missing-header":
        (source / "include/cusparse.h").unlink()
    elif corrupt == "missing-lib":
        family.unlink()
    elif corrupt == "ambiguous":
        (source / "lib64/libnvJitLink.so.13.2").write_bytes(family.read_bytes())
    elif corrupt == "cuda12-needed":
        path = source / "lib64/libcusparse.so.12.1"
        info = json.loads(path.read_text())
        info["needed"] = ["libnvJitLink.so.12"]
        path.write_text(json.dumps(info))
    elif corrupt == "wrong-soname":
        info = json.loads(family.read_text())
        info["soname"] = "libnvJitLink.so.12"
        family.write_text(json.dumps(info))
    elif corrupt == "mixed-inventory":
        info = json.loads(family.read_text())
        info["soname"] = "libnvJitLink.so.12"
        (source / "lib64/libnvJitLink.so.12.1").write_text(json.dumps(info))
    elif corrupt == "wrong-version":
        (source / "include/cuda.h").write_text("#define CUDA_VERSION 12060\n")
    elif corrupt == "conflicting-header":
        (source / "include/driver_types.h").write_text("#define CUDART_VERSION 12060\n")
    elif corrupt == "changed-input":
        (source / "include/cusparse.h").write_text("// changed but plausible header\n")
    elif corrupt == "missing-transitive":
        (source / "include/crt/host_config.h").unlink()
    else:
        target = tmp_path / "cuda-12.6.1"
        source.rename(target)
        source = target
    destination = tmp_path / "never-created"
    with pytest.raises((RuntimeError, FileNotFoundError)):
        module.prepare_sdk(source, destination, manifest)
    assert not destination.exists()


@pytest.mark.parametrize("tree", ["include", "bin", "nvvm"])
def test_full_transitive_sdk_trees_are_bound_not_only_named_inputs(sdk, tmp_path, tree):
    module, source, _ = sdk
    extra = source / tree / "extra/deep/transitive.hpp"
    extra.parent.mkdir(parents=True)
    extra.write_text("// additional required input\n")
    manifest = module.inspect_sdk(source)
    extra.unlink()
    with pytest.raises(RuntimeError, match="changed"):
        module.prepare_sdk(source, tmp_path / "never", manifest)
    assert not (tmp_path / "never").exists()


def test_sdk_directory_cycle_is_rejected(sdk):
    module, source, _ = sdk
    (source / "include/cycle").symlink_to(source / "include", target_is_directory=True)
    with pytest.raises(RuntimeError, match="Cyclic"):
        module.inspect_sdk(source)


@pytest.mark.parametrize("corrupt", [None, "empty", "rpath", "old-build", "site12", "absolute-needed", "cuda12"])
def test_installed_loader_policy(sdk, tmp_path, corrupt):
    module, _, _ = sdk
    path = tmp_path / "fixture.so"
    expected = ["$ORIGIN", str(tmp_path / "private13/lib64"), str(tmp_path / "environment/lib")]
    info = {"soname": "libEinsums-abi1.so.1", "needed": ["libcudart.so.13"],
            "rpath": [], "runpath": list(expected)}
    if corrupt == "empty":
        info["runpath"].append("")
    elif corrupt == "rpath":
        info["rpath"], info["runpath"] = info["runpath"], []
    elif corrupt == "old-build":
        info["runpath"][0] = "/old/einsums-build/lib"
    elif corrupt == "site12":
        info["runpath"][1] = "/site/cuda-12.6.1/lib64"
    elif corrupt == "absolute-needed":
        info["needed"] = ["/old/einsums-build/lib/libEinsums-abi1.so.1"]
    elif corrupt == "cuda12":
        info["needed"] = ["libnvJitLink.so.12"]
    path.write_text(json.dumps(info))
    if corrupt:
        with pytest.raises(RuntimeError):
            module.audit_elf(path, expected)
    else:
        assert module.audit_elf(path, expected)["sha256"] == module.digest(path)


@pytest.mark.parametrize("corrupt", [None, "missing", "foreign-cuda", "site12"])
def test_actual_loader_closure_is_bound_to_selected_sdk(sdk, tmp_path, monkeypatch, corrupt):
    module, source, manifest = sdk
    prefix, environment = tmp_path / "installed", tmp_path / "environment"
    (prefix / "lib").mkdir(parents=True)
    (environment / "lib").mkdir(parents=True)
    manifest["private_prefix"] = str(tmp_path / "private13")
    lines = [f"{item['soname']} => {item['path']} (0x1)" for item in manifest["libraries"].values()]
    if corrupt == "missing":
        lines.append("libmissing.so => not found")
    elif corrupt == "foreign-cuda":
        other = tmp_path / "other/libnvJitLink.so.13"
        other.parent.mkdir()
        other.write_text("foreign")
        lines = [l for l in lines if not l.startswith("libnvJitLink")]
        lines.append(f"libnvJitLink.so.13 => {other} (0x1)")
    elif corrupt == "site12":
        other = tmp_path / "cuda-12.6.1/libnvJitLink.so.12"
        other.parent.mkdir()
        other.write_text("old")
        lines.append(f"libnvJitLink.so.12 => {other} (0x1)")
    monkeypatch.setattr(module, "command", lambda *args, **kwargs: "\n".join(lines))
    if corrupt:
        with pytest.raises(RuntimeError):
            module.audit_closure(prefix / "unused.so", manifest, prefix, environment)
    else:
        assert len(module.audit_closure(prefix / "unused.so", manifest, prefix, environment)) == 6


@pytest.mark.parametrize("corrupt", [None, "missing", "interface", "foreign", "changed-sha", "no-python",
                                    "no-stubs", "static-native", "skip-rpath", "bad-layout"])
def test_deferred_cmake_hook_configures_install_without_compiling(sdk, tmp_path, corrupt):
    module, source, manifest = sdk
    private = tmp_path / "private13"
    module.prepare_sdk(source, private, manifest)
    build = tmp_path / "configure-only"
    contract = private / "sdk-contract.json"
    args = ["cmake", "-S", str(TOOLS / "contract_config_fixture"), "-B", str(build),
            "-DCMAKE_PROJECT_Einsums_INCLUDE=" + str(TOOLS / "revised_einsums_hook.cmake"),
            "-DDISP0_SDK_CONTRACT=" + str(contract),
            "-DDISP0_SDK_CONTRACT_SHA256=" + module.digest(contract),
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DDISP0_DEPENDENCY_PREFIX=" + str(tmp_path / "environment")]
    if corrupt == "missing":
        args.append("-DFIXTURE_MISSING=nvJitLink")
    elif corrupt == "interface":
        args.append("-DFIXTURE_INTERFACE=nvJitLink")
    elif corrupt == "foreign":
        foreign = tmp_path / "outside/libnvJitLink.so"
        foreign.parent.mkdir()
        foreign.write_text("unattested")
        args += ["-DFIXTURE_FOREIGN=nvJitLink", "-DFIXTURE_FOREIGN_PATH=" + str(foreign)]
    elif corrupt == "changed-sha":
        Path(manifest["libraries"]["nvJitLink"]["path"]).write_text("changed")
    elif corrupt == "no-python":
        args.append("-DFIXTURE_MISSING_PYTHON=ON")
    elif corrupt == "no-stubs":
        args.append("-DFIXTURE_MISSING_STUBS=ON")
    elif corrupt == "static-native":
        args.append("-DFIXTURE_STATIC_NATIVE=ON")
    elif corrupt == "skip-rpath":
        args.append("-DCMAKE_SKIP_RPATH=ON")
    elif corrupt == "bad-layout":
        args.append("-DCMAKE_INSTALL_LIBDIR=lib64")
    process = subprocess.run(args, capture_output=True, text=True, timeout=15)
    if corrupt:
        assert process.returncode != 0, process.stdout
    else:
        assert process.returncode == 0, process.stdout + process.stderr
        properties = (build / "fixture-properties.txt").read_text()
        assert "$ORIGIN/.." in properties and "--enable-new-dtags" in properties
        assert (build / "disp0-cuda-targets.tsv").read_text().count("CUDA::") == 6
        install = (build / "cmake_install.cmake").read_text()
        assert "lib/einsums" in install and "_core" in install
        assert not list(build.rglob("*.so")) and not list(build.rglob("*.o"))


def write_physical_xml(path, *, bad_delta=False, skip_physical=False):
    root = ET.Element("testsuite")
    for i in range(26):
        case = ET.SubElement(root, "testcase", name=f"physical-{i}")
        if skip_physical and i == 1:
            ET.SubElement(case, "skipped")
        if i == 0:
            properties = ET.SubElement(case, "properties")
            rows = [{"label": label, "actual": -0.001, "reference": -0.001,
                     "delta": 0.01 if bad_delta else 0.0}
                    for _ in range(32) for label in ("Disp20,u", "Exch-Disp20,u")]
            ET.SubElement(properties, "property", name="signed_comparisons", value=json.dumps(rows))
    ET.ElementTree(root).write(path)


@pytest.mark.parametrize("corrupt", [None, "delta", "skipped"])
def test_physical_admission_requires_actual_scope_and_retained_signed_results(tmp_path, corrupt):
    module = load("revised_admission")
    xml = tmp_path / "results.xml"
    write_physical_xml(xml, bad_delta=corrupt == "delta", skip_physical=corrupt == "skipped")
    if corrupt:
        with pytest.raises(RuntimeError):
            module.physical_gate(xml)
    else:
        assert len(module.physical_gate(xml)["signed_comparisons"]) == 64


def test_original_failed_origin_is_never_accepted_by_new_admission(tmp_path):
    module = load("revised_admission")
    with pytest.raises(RuntimeError, match="ineligible"):
        module.admit(tmp_path, "14196330", tmp_path / "unused", tmp_path, "a" * 40)


@pytest.fixture
def prepared(sdk, tmp_path, monkeypatch):
    contract, sdk_source, sdk_input = sdk
    module = load("revised_admission")
    pin, job = "a" * 40, "24680"
    run = tmp_path / f"revised-build-{pin}-{job}"
    run.mkdir()
    sdk_record = contract.prepare_sdk(sdk_source, run / "cuda13", sdk_input)
    input_sdk = tmp_path / "input-sdk.json"
    input_sdk.write_text(json.dumps(sdk_input))
    dependency_lock = tmp_path / "dependencies.lock"
    dependency_lock.write_text("fixture exact dependency input\n")
    (run / "input-receipts.json").write_text(json.dumps({
        "sdk": {"path": str(input_sdk), "sha256": contract.digest(input_sdk)},
        "dependencies": {"path": str(dependency_lock), "sha256": contract.digest(dependency_lock)}}))
    source = tmp_path / "source"
    source.mkdir()
    (run / "einsums-source").mkdir()
    package = run / "einsums-build/lib/einsums"
    installed = run / "einsums-stage/lib/einsums"
    files = {"__init__.py", "rc.py", "graph.py", "sealed.py", "profile.py", "_core.pyi", "py.typed"}
    files |= {f"helper{i}.py" for i in range(20)}
    files.add("_core.fixture.so")
    for directory in (package, installed):
        directory.mkdir(parents=True)
        for name in files:
            (directory / name).write_text(f"complete fixture {name}\n")
    native = run / "einsums-stage/lib/libEinsums-abi1.so.2.0.0"
    native.write_text("fixture native\n")
    core = run / "psi4-build/stage/lib/psi4/core.fixture.so"
    core.parent.mkdir(parents=True)
    core.write_text("fixture scientific core\n")
    (core.parent / "__init__.py").write_text("# matching package fixture\n")
    for name in module.MODULES:
        relative = Path("psi4/driver/procrouting/sapt") / name
        p = source / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# matching scientific fixture\n")
        p = run / "psi4-build/stage/lib" / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# matching scientific fixture\n")
    binaries = {str(p.resolve()): contract.digest(p) for p in (core, installed / "_core.fixture.so", native)}
    maps = list(binaries) + [item["path"] for item in sdk_record["libraries"].values()]
    common = {"status": "PASS", "psi4": str(core.parent / "__init__.py"),
              "einsums": str(installed / "__init__.py")}
    metadata = {**common, "pid": 111, "source_pin": pin, "job": job, "run": str(run),
                "world": {"git_commit": contract.E2_PIN, "version_major": 2,
                                   "compiler_major": 14, "library_path": str(native)},
                "binary_sha256": binaries, "mapped_libraries": maps,
                "mapped_library_sha256": {p: contract.digest(p) for p in maps},
                "actual_gpu_correctness_verified": False,
                "elf_audit": {p: {"sha256": checksum} for p, checksum in binaries.items()},
                "closure": {str(binary): {item["soname"]: {"path": item["path"], "sha256": item["sha256"]}
                                          for item in sdk_record["libraries"].values()}
                            for binary in (installed / "_core.fixture.so", native, core)},
                "sdk_contract_sha256": contract.digest(run / "cuda13/sdk-contract.json")}
    api = {**common, "tolerances": {"rtol": 1e-11, "atol": 1e-12}, "checks": sorted(module.API_CHECKS)}
    (run / "runtime-provenance.json").write_text(json.dumps(metadata))
    (run / "api-probe.json").write_text(json.dumps(api))
    xml = run / "psi4-build/reference-check-fixture/results.xml"
    xml.parent.mkdir()
    write_physical_xml(xml)
    process = {"exit_status": 0, "pid": 222, "source_pin": pin, "run": str(run), "job": job,
               "binary_sha256": binaries, "metadata_sha256": contract.digest(run / "runtime-provenance.json"),
               "xml": str(xml), "xml_sha256": contract.digest(xml)}
    components = {family: item["path"] for family, item in sdk_record["libraries"].items()}
    process["cuda_component_maps"] = components
    process["mapped_library_sha256"] = {p: contract.digest(p) for p in maps}
    api["runtime_observation"] = {k: v for k, v in process.items()
                                  if k not in {"exit_status", "xml", "xml_sha256"}}
    (run / "api-probe.json").write_text(json.dumps(api))
    (run / "physical-process-proof.json").write_text(json.dumps(process))
    target_rows = [f"CUDA::{family}\tIMPORTED_LOCATION\t{run}/cuda13/lib64/lib{family}.so\t"
                   f"{item['path']}\t{item['sha256']}\n"
                   for family, item in sdk_record["libraries"].items()]
    (run / "einsums-build/disp0-cuda-targets.tsv").write_text("".join(target_rows))
    (run / "einsums-build/disp0-install-targets.tsv").write_text(
        f"Einsums\t$ORIGIN;{run}/cuda13/lib64;{run}/environment/lib\n"
        f"PyEinsums\t$ORIGIN/..;{run}/cuda13/lib64;{run}/environment/lib\n")
    (run / "environment-explicit.txt").write_text("fixture explicit dependency input\n")
    compiler = run / "environment/bin/x86_64-conda-linux-gnu-c++"
    cache = (f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n"
             f"CMAKE_CUDA_HOST_COMPILER:FILEPATH={compiler}\n"
             f"CMAKE_CUDA_COMPILER:FILEPATH={run}/cuda13/bin/nvcc\n"
             f"CMAKE_INSTALL_PREFIX:PATH={run}/einsums-stage\n"
             "EINSUMS_WITH_CUDA:BOOL=ON\nEINSUMS_WITH_CUDA_ARCHITECTURES:STRING=90\n"
             "EINSUMS_BUILD_PYTHON:BOOL=ON\nEINSUMS_WITH_GPU_MOCK_DISCRETE:BOOL=OFF\n"
             "EINSUMS_WITH_PROFILER:BOOL=ON\n")
    cache += "".join(f"CUDA_{family}_LIBRARY:FILEPATH={run}/cuda13/lib64/lib{family}.so\n"
                     for family in sdk_record["libraries"])
    for name in ("einsums-build", "psi4-build"):
        (run / name / "CMakeCache.txt").write_text(cache)
    (run / "compiler-version.txt").write_text("GNU14.3.0 fixture\n")
    (run / "nvcc-version.txt").write_text("V13.3.73 fixture\n")
    (run / "cmake-version.txt").write_text("cmake4.3.2 fixture\n")
    (run / "configuration-source-sha256.txt").write_text("fixture tracked hook identity\n")
    (run / "compiler-identities.txt").write_text("fixture compiler/frontend identities\n")
    evidence = {}
    databases = []
    for label in ("einsums-build", "psi4-build"):
        db = run / label / "compile_commands.json"
        db.write_text("fixture command database; no compiler executed\n")
        evidence[str(db)] = contract.digest(db)
        databases.append(str(db))
    audit = {"status": "EFFECTIVE_BUILD_INPUTS_AUDITED", "source": str(source), "run": str(run),
             "counts": {"compile": 2, "dependencies": 2, "links": 2},
             "observed_cuda_headers": ["cuda_runtime_api.h", "cublas_v2.h"],
             "file_sha256": evidence, "compile_databases": databases}
    (run / "effective-build-audit.json").write_text(json.dumps(audit))
    def fake_command(args):
        if args[0] == "sacct":
            return (f"{job}|COMPLETED|0:0|gts-cs207-chemx|embers\n"
                    f"{job}.batch|COMPLETED|0:0|gts-cs207-chemx|\n")
        if args[3:5] == ["rev-parse", "HEAD"]:
            return contract.E2_PIN if args[2].endswith("einsums-source") else pin
        return ""
    monkeypatch.setattr(module, "command", fake_command)
    return module, run, source, pin, xml, metadata, process, api


def test_new_build_admission_binds_actual_process_and_clean_allocation(prepared):
    module, run, source, pin, xml, *_ = prepared
    document = module.collect(run, source, pin, xml)
    assert document["status"] == "CPU_PREPARATION_PASS_PENDING_JOB_EXIT"
    assert document["actual_gpu_correctness_verified"] is False
    (run / "preparation-result.json").write_text(json.dumps(document))
    status = run / "status"
    status.write_text(f"exit_code=0\nterm_received=0\nrun={run}\nCPU_PREPARATION_EXIT_ZERO\n")
    admitted = module.admit(run, "24680", status, source, pin)
    assert admitted["status"] == "CPU_RUNTIME_PREPARATION_ONLY"
    assert len(admitted["physical_cpu_gate"]["signed_comparisons"]) == 64


@pytest.mark.parametrize("corrupt", ["foreign-stage", "incomplete-binaries", "old-tuple", "historical-xml",
                                    "no-process-proof", "physical-exit", "foreign-job", "extra-package",
                                    "bad-target", "empty-loader", "missing-api", "bad-sdk-map",
                                    "failed-allocation", "signaled", "stale-package",
                                    "extra-native", "empty-closure", "changed-closure",
                                    "wrong-compiler", "wrong-arch", "site-cache",
                                    "changed-dependency-input", "changed-sdk-input",
                                    "foreign-extra-map", "foreign-physical-map", "no-command-proof"])
def test_new_admission_rejects_stale_partial_or_failed_evidence(prepared, corrupt, monkeypatch):
    module, run, source, pin, xml, metadata, process, api = prepared
    if corrupt == "foreign-stage":
        metadata["psi4"] = "/installed-release/psi4/__init__.py"
    elif corrupt == "incomplete-binaries":
        metadata["binary_sha256"].pop(next(iter(metadata["binary_sha256"])))
    elif corrupt == "old-tuple":
        metadata["binary_sha256"] = dict(zip(metadata["binary_sha256"], sorted(module.OLD_TUPLE)))
    elif corrupt == "historical-xml":
        process["xml"] = "/historical/results.xml"
    elif corrupt == "no-process-proof":
        (run / "physical-process-proof.json").unlink()
    elif corrupt == "physical-exit":
        process["exit_status"] = 1
    elif corrupt == "foreign-job":
        process["job"] = "14196330"
    elif corrupt == "extra-package":
        (run / "einsums-stage/lib/einsums/overlay.py").write_text("# unexpected\n")
    elif corrupt == "bad-target":
        (run / "einsums-build/disp0-cuda-targets.tsv").write_text("")
    elif corrupt == "empty-loader":
        p = run / "einsums-build/disp0-install-targets.tsv"
        p.write_text(p.read_text().replace("$ORIGIN;", "$ORIGIN;;"))
    elif corrupt == "missing-api":
        api["checks"] = []
    elif corrupt == "bad-sdk-map":
        metadata["mapped_libraries"] = []
    elif corrupt == "stale-package":
        (run / "einsums-stage/lib/einsums/graph.py").write_text("# changed\n")
    elif corrupt == "extra-native":
        (run / "einsums-stage/lib/libforeign.so").write_text("unlisted")
    elif corrupt == "empty-closure":
        metadata["closure"][next(iter(metadata["closure"]))] = {}
    elif corrupt == "changed-closure":
        closure = next(iter(metadata["closure"].values()))
        closure[next(iter(closure))]["sha256"] = "0" * 64
    elif corrupt in {"wrong-compiler", "wrong-arch", "site-cache"}:
        path = run / "einsums-build/CMakeCache.txt"
        text = path.read_text()
        if corrupt == "wrong-compiler":
            text = text.replace(str(run / "environment/bin/x86_64-conda-linux-gnu-c++"), "/usr/bin/g++")
        elif corrupt == "wrong-arch":
            text = text.replace("ARCHITECTURES:STRING=90", "ARCHITECTURES:STRING=80")
        else:
            text = text.replace(f"{run}/cuda13/lib64/libcusparse.so", "/site/cuda-12.6/lib64/libcusparse.so")
        path.write_text(text)
    elif corrupt in {"changed-dependency-input", "changed-sdk-input"}:
        inputs = json.loads((run / "input-receipts.json").read_text())
        key = "dependencies" if corrupt == "changed-dependency-input" else "sdk"
        Path(inputs[key]["path"]).write_text("changed pinned input\n")
    elif corrupt in {"foreign-extra-map", "foreign-physical-map"}:
        outside = run.parent / "outside/libforeign.so"
        outside.parent.mkdir(exist_ok=True)
        outside.write_text("unapproved actual mapping")
        if corrupt == "foreign-extra-map":
            metadata["mapped_libraries"].append(str(outside))
            metadata["mapped_library_sha256"][str(outside)] = module.digest(outside)
        else:
            process["mapped_library_sha256"][str(outside)] = module.digest(outside)
    elif corrupt == "no-command-proof":
        (run / "effective-build-audit.json").unlink()
    if corrupt not in {"failed-allocation", "signaled"}:
        (run / "runtime-provenance.json").write_text(json.dumps(metadata))
        (run / "api-probe.json").write_text(json.dumps(api))
        if corrupt != "no-process-proof":
            (run / "physical-process-proof.json").write_text(json.dumps(process))
        with pytest.raises((RuntimeError, FileNotFoundError)):
            module.collect(run, source, pin, xml)
    else:
        document = module.collect(run, source, pin, xml)
        (run / "preparation-result.json").write_text(json.dumps(document))
        status = run / "status"
        status.write_text(f"exit_code=0\nterm_received={1 if corrupt == 'signaled' else 0}\n"
                          f"run={run}\nCPU_PREPARATION_EXIT_ZERO\n")
        if corrupt == "failed-allocation":
            monkeypatch.setattr(module, "command", lambda args:
                                "24680|FAILED|1:0|gts-cs207-chemx|embers\n"
                                "24680.batch|FAILED|1:0|gts-cs207-chemx|\n")
        with pytest.raises(RuntimeError):
            module.admit(run, "24680", status, source, pin)


def test_revised_batch_is_only_a_fixed_cpu_proposal():
    script = (TOOLS / "revised_cpu_build.sbatch").read_text()
    for flag in ("--cpus-per-task=8", "--mem=48G", "--time=02:00:00",
                 "--qos=embers", "--account=gts-cs207-chemx", "--no-requeue"):
        assert flag in script
    assert "--gres=" not in script
    assert "DISP0_NEW_CPU_BUILD_APPROVAL_REFERENCE:?" in script
    payload = (TOOLS / "revised_cpu_payload.sh").read_text()
    assert "DISP0_SDK_INPUT_SHA256:?" in payload and "DISP0_DEPENDENCY_LOCK_SHA256:?" in payload
    assert "d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62" in payload
    assert "587e172a33afaadb00eb7f574a5089533f7fc755" in payload
    assert "git apply" not in payload and "patchelf" not in payload and "LD_PRELOAD=" not in payload


@pytest.mark.parametrize("corrupt", [None, "compiler", "include", "link", "env-flags", "response",
                                    "version-script", "linker-script", "relative-script", "packed-script",
                                    "xlinker-script"])
def test_effective_compile_and_link_guard(sdk, tmp_path, corrupt):
    contract, source_sdk, manifest = sdk
    module = load("build_audit")
    run, source, scratch = tmp_path / "run", tmp_path / "science", tmp_path / "scratch"
    for p in (run, source, scratch):
        p.mkdir()
    sdk_data = {**manifest, "private_prefix": str(run / "cuda13")}
    compiler = run / "environment/bin/x86_64-conda-linux-gnu-c++"
    command = f"{compiler} -I{source_sdk}/include -c {source}/fixture.cc -o {run}/fixture.o"
    if corrupt == "compiler":
        command = command.replace(str(compiler), "/usr/bin/g++")
    elif corrupt == "include":
        command += " -isystem/foreign/include"
    elif corrupt == "link":
        command += " -Wl,/foreign/libwrong.so"
    elif corrupt == "env-flags":
        command += " -Xcompiler=-I/site/cuda-12.6/include"
    elif corrupt == "response":
        command += " @missing.rsp"
    elif corrupt == "version-script":
        command += " -Wl,--version-script=/foreign/exports.map"
    elif corrupt == "linker-script":
        command += " -Wl,-T,/foreign/layout.ld"
    elif corrupt == "relative-script":
        command += " -Wl,--version-script=../../foreign/exports.map"
    elif corrupt == "packed-script":
        command += " -Wl,-T/foreign/layout.ld"
    elif corrupt == "xlinker-script":
        command += " -Xlinker --version-script=/foreign/exports.map"
    entries = [{"directory": str(run), "file": str(source / "fixture.cc"), "command": command}]
    if corrupt:
        with pytest.raises(RuntimeError):
            module.audit_compile_database(entries, run, source, sdk_data, scratch)
    else:
        assert module.audit_compile_database(entries, run, source, sdk_data, scratch) == 1
        assert module.audit_link_commands(
            f"{compiler} -shared -Wl,-rpath,{run}/lib -o {run}/libfixture.so",
            run, run, source, sdk_data, scratch) == 1
        assert module.audit_link_commands(
            f"{compiler} -shared -Wl,--version-script={run}/exports.map,-T,{run}/layout.ld",
            run, run, source, sdk_data, scratch) == 1


@pytest.mark.parametrize("corrupt", [None, "foreign-cuda", "foreign-header", "changed-sdk",
                                    "driver_types.h", "vector_types.h", "builtin_types.h", "nv/target"])
def test_actual_ninja_dependency_paths_are_audited(sdk, tmp_path, corrupt):
    contract, source_sdk, manifest = sdk
    module = load("build_audit")
    run, source, scratch = tmp_path / "run", tmp_path / "science", tmp_path / "scratch"
    for p in (run, source, scratch):
        p.mkdir()
    paths = [source_sdk / "include/cuda_runtime_api.h", source_sdk / "include/cublas_v2.h"]
    if corrupt in {"foreign-cuda", "foreign-header"}:
        p = tmp_path / "foreign" / ("cuda_runtime_api.h" if corrupt == "foreign-cuda" else "foreign.hpp")
        p.parent.mkdir()
        p.write_text("// not an attested compiler include\n")
        paths.append(p)
    elif corrupt == "changed-sdk":
        paths[0].write_text("// changed after input inspection\n")
    elif corrupt in {"driver_types.h", "vector_types.h", "builtin_types.h", "nv/target"}:
        p = run / "environment/include" / corrupt
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("// matching-looking CUDA header from the wrong allowed root\n")
        paths.append(p)
    text = "fixture.o: #deps2 (VALID)\n" + "".join(f"    {p}\n" for p in paths)
    if corrupt:
        with pytest.raises(RuntimeError):
            module.audit_dependencies(text, run, run, source, manifest, scratch)
    else:
        count, names = module.audit_dependencies(text, run, run, source, manifest, scratch)
        assert count == 2 and names == {"cuda_runtime_api.h", "cublas_v2.h"}


def test_ninja_expands_response_metadata_without_compiling(tmp_path):
    # Deliberately nonexistent compiler. Only -t compdb runs, not a build.
    source = tmp_path / "fixture.cc"
    source.write_text("// metadata fixture only\n")
    (tmp_path / "build.ninja").write_text(
        "rule compile\n"
        "  command = /nonexistent/never-executed-c++ @fixture.rsp -c $in -o $out\n"
        "  rspfile = fixture.rsp\n"
        "  rspfile_content = -I/explicit/private/include\n"
        "rule link\n"
        "  command = /nonexistent/never-executed-c++ -shared $in -o $out\n"
        "build fixture.o: compile fixture.cc\n"
        "build fixture.so: link fixture.o\n")
    result = subprocess.run(["ninja", "-C", str(tmp_path), "-t", "compdb", "-x"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    entries = json.loads(result.stdout)
    assert any("-I/explicit/private/include" in e["command"] and "@fixture.rsp" not in e["command"] for e in entries)
    assert any("-shared" in e["command"] for e in entries)
    assert not (tmp_path / "fixture.o").exists() and not (tmp_path / "fixture.so").exists()


@pytest.mark.parametrize("path,expected", [
    ("/usr/lib64/libc.so.6", True), ("/lib64/ld-linux-x86-64.so.2", True),
    ("/usr/lib/x86_64-linux-gnu/libm.so.6", True), ("/usr/local/cuda-12.6/lib64/libcudart.so", False)])
def test_rhel_and_debian_system_library_roots(path, expected):
    assert load("runtime_contract").system_library_location(path) is expected


def test_deleted_actual_native_mapping_is_rejected():
    module = load("runtime_contract")
    with pytest.raises(RuntimeError, match="Deleted"):
        module.mapped_libraries("100-200 r-xp 0000 00:00 123 /foreign/libwrong.so (deleted)\n")
