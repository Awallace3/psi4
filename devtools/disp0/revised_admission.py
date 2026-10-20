"""New successful-build admission. FAILED14196330/fixed old tuple stay rejected."""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from runtime_contract import (COMPONENTS, E2_PIN, SCIENCE_PIN, audit_install_report,
                              audit_mapped_libraries, audit_target_report, command, digest, require, write_new)

LABELS = {"Disp20,u", "Exch-Disp20,u"}
MODULES = ("sapt_jk_terms_ein.py", "sapt_mp2_terms_ein.py", "sapt_dispersion_gpu.py", "sapt_proc.py")
API_CHECKS = {"gemm-denominator-division-reduction-1x1", "gemm-denominator-division-reduction-3x4",
              "gemm-denominator-division-reduction-2x1",
              "default-graph-replay-strided-view-mutation", "pool-view-keepalive-reset"}
OLD_TUPLE = {
    "b3cfe2f2261533df9a9018731d9fe2bffea41cfe70f53f14195caaee7de8697d",
    "8f23c69b79613f9b94eee95d4735e8f88f67be7c98867290f7546ca56bdc6833",
    "9e31b5ff8cf28e796b04ee74853c4923f06acf03f1af24dfddd75b79de583cb7",
}


def configuration_gate(run, sdk):
    run = Path(run)
    caches = {}
    for name in ("einsums-build", "psi4-build"):
        path = run / name / "CMakeCache.txt"
        parsed = {}
        for line in path.read_text().splitlines():
            if ":" in line and "=" in line and not line.startswith(("#", "//")):
                key, value = line.split("=", 1)
                parsed[key.split(":", 1)[0]] = value
        caches[name] = parsed
        require(parsed["CMAKE_CXX_COMPILER"] == str(run / "environment/bin/x86_64-conda-linux-gnu-c++"),
                "Actual compiler cache differs from the fresh GNU14 environment")
    e2 = caches["einsums-build"]
    for key, expected in {"EINSUMS_WITH_CUDA": "ON", "EINSUMS_WITH_CUDA_ARCHITECTURES": "90",
                          "EINSUMS_BUILD_PYTHON": "ON", "EINSUMS_WITH_GPU_MOCK_DISCRETE": "OFF",
                          "EINSUMS_WITH_PROFILER": "ON"}.items():
        require(e2.get(key) == expected, f"Actual E2 configuration differs: {key}")
    require(e2["CMAKE_CUDA_COMPILER"] == str(run / "cuda13/bin/nvcc")
            and e2["CMAKE_CUDA_HOST_COMPILER"] == caches["einsums-build"]["CMAKE_CXX_COMPILER"],
            "Actual CUDA compiler/host mismatch")
    require(e2["CMAKE_INSTALL_PREFIX"] == str(run / "einsums-stage"), "Wrong installed E2 prefix")
    for family, item in sdk["libraries"].items():
        locator = e2[f"CUDA_{family}_LIBRARY"]
        require(locator == str(run / f"cuda13/lib64/lib{family}.so")
                and str(Path(locator).resolve(strict=True)) == item["path"], "Wrong actual CUDA library cache")
    require("14.3.0" in (run / "compiler-version.txt").read_text(), "Wrong actual GNU compiler version")
    require("V13.3.73" in (run / "nvcc-version.txt").read_text(), "Wrong actual nvcc version")
    return {"cache_sha256": {name: digest(run / name / "CMakeCache.txt") for name in caches},
            "cuda_compiler": e2["CMAKE_CUDA_COMPILER"], "gnu_compiler": e2["CMAKE_CXX_COMPILER"]}


def physical_gate(xml):
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    require(len(cases) == 26, "Wrong physical CPU gate scope")
    records, skipped = [], []
    for case in cases:
        require(case.find("failure") is None and case.find("error") is None, "CPU gate failure")
        if case.find("skipped") is not None:
            skipped.append(case.attrib["name"])
        for prop in case.findall("./properties/property"):
            if prop.attrib["name"] == "signed_comparisons":
                records.extend({**row, "testcase": case.attrib["name"]}
                               for row in json.loads(prop.attrib["value"]))
    require(all(name == "test_disp0_cuda_request_rejects_mock" for name in skipped),
            "An actual physical/API test was skipped")
    require(len(records) == 64 and {r["label"] for r in records} == LABELS,
            "Missing signed physical comparisons")
    require(all(sum(r["label"] == label for r in records) == 32 for label in LABELS),
            "Signed component inventory differs")
    for row in records:
        a, b, delta = (float(row[k]) for k in ("actual", "reference", "delta"))
        require(all(math.isfinite(v) for v in (a, b, delta)), "Nonfinite physical comparison")
        require(math.isclose(a - b, delta, rel_tol=1e-12, abs_tol=1e-16), "Invalid retained discrepancy")
        require(abs(a - b) <= 1e-8 + 1e-8 * abs(b), "Signed CPU component mismatch")
    return {"test_count": len(cases), "skipped": skipped, "signed_comparisons": records,
            "tolerances": {"rtol": 1e-8, "atol_Eh": 1e-8}, "xml_sha256": digest(xml)}


def collect(run, source, pin, xml):
    run, source = Path(run).resolve(), Path(source).resolve()
    require(command(["git", "-C", str(source), "rev-parse", "HEAD"]).strip() == pin, "Wrong source pin")
    for args in (["diff", "HEAD", "--name-only"], ["diff", "--cached", "--name-only"],
                 ["diff", SCIENCE_PIN, pin, "--name-only", "--", "psi4/"]):
        require(not command(["git", "-C", str(source), *args]).strip(), "Scientific/source tree changed")
    e2_source = run / "einsums-source"
    require(command(["git", "-C", str(e2_source), "rev-parse", "HEAD"]).strip() == E2_PIN, "Wrong E2 source")
    require(not command(["git", "-C", str(e2_source), "status", "--porcelain"]).strip(),
            "E2 source was patched or dirtied")
    metadata_path, api_path = run / "runtime-provenance.json", run / "api-probe.json"
    metadata, api = json.loads(metadata_path.read_text()), json.loads(api_path.read_text())
    require(metadata["status"] == api["status"] == "PASS", "Missing installed metadata/API pass")
    require(metadata["source_pin"] == pin and metadata["run"] == str(run)
            and isinstance(metadata["pid"], int) and metadata["pid"] > 0, "Foreign metadata process proof")
    require(metadata.get("actual_gpu_correctness_verified") is False, "CPU proof mislabeled device execution")
    require(metadata["world"]["git_commit"] == E2_PIN
            and metadata["world"]["version_major"] == 2
            and metadata["world"]["compiler_major"] == 14, "Wrong mapped native revision/ABI")
    sdk_path = run / "cuda13/sdk-contract.json"
    sdk = json.loads(sdk_path.read_text())
    require(sdk["private_prefix"] == str(run / "cuda13") and set(sdk["libraries"]) == set(COMPONENTS),
            "Foreign/incomplete private SDK")
    receipts_path = run / "input-receipts.json"
    receipts = json.loads(receipts_path.read_text())
    for item in receipts.values():
        require(digest(item["path"]) == item["sha256"], "Pinned build input changed")
    source_sdk = json.loads(Path(receipts["sdk"]["path"]).read_text())
    require(sdk == {**source_sdk, "private_prefix": str(run / "cuda13")}, "Private SDK is not the pinned input")
    build_audit_path = run / "effective-build-audit.json"
    build_audit = json.loads(build_audit_path.read_text())
    require(build_audit["status"] == "EFFECTIVE_BUILD_INPUTS_AUDITED"
            and build_audit["source"] == str(source) and build_audit["run"] == str(run),
            "Missing/foreign effective build-command proof")
    require(all(build_audit["counts"][kind] > 0 for kind in ("compile", "dependencies", "links")),
            "Incomplete effective compile/include/link audit")
    require(build_audit["file_sha256"] and len(build_audit["compile_databases"]) >= 2
            and set(build_audit["observed_cuda_headers"]) & {"cuda_runtime.h", "cuda_runtime_api.h"}
            and set(build_audit["observed_cuda_headers"]) & {"cublas_v2.h", "cublas_api.h"},
            "Missing actual compile database/CUDA include evidence")
    for path, expected in build_audit["file_sha256"].items():
        require(Path(path).resolve().is_relative_to(run) and digest(path) == expected,
                "Effective build evidence changed")
    for proof in (metadata, api):
        require(Path(proof["psi4"]).resolve(strict=True).is_relative_to(run / "psi4-build/stage"),
                "Proof imported a foreign scientific stage")
        require(Path(proof["einsums"]).resolve(strict=True).is_relative_to(run / "einsums-stage"),
                "Proof imported another Python E2")
    for item in sdk["libraries"].values():
        require(item["path"] in metadata["mapped_libraries"]
                and digest(item["path"]) == item["sha256"], "Missing/changed actual CUDA map")
    mapped = audit_mapped_libraries(metadata["mapped_libraries"], sdk, run / "einsums-stage",
                                    run / "psi4-build/stage", run / "environment")
    require(metadata["mapped_library_sha256"] == mapped, "Actual mapped-library audit changed/incomplete")
    require(api["tolerances"] == {"rtol": 1e-11, "atol": 1e-12}
            and len(api["checks"]) == 5 and set(api["checks"]) == API_CHECKS, "Incomplete tiny API proof")
    prefix = run / "einsums-stage/lib/einsums"
    package = run / "einsums-build/lib/einsums"
    def package_files(root):
        return {str(p.relative_to(root)): p for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts}
    built, installed = package_files(package), package_files(prefix)
    require(set(built) == set(installed) and len(installed) >= 20, "Partial/extra installed E2 package")
    require({"__init__.py", "rc.py", "graph.py", "profile.py", "sealed.py", "_core.pyi", "py.typed"} <= set(installed),
            "Missing actual E2 package entrypoints")
    for relative in built:
        if not (relative.startswith("_core.") and relative.endswith(".so")):
            require(digest(built[relative]) == digest(installed[relative]), "Installed Python overlay/omission")
    cores = list((run / "psi4-build/stage/lib/psi4").glob("core.*.so"))
    extensions = list(prefix.glob("_core.*.so"))
    require(len(cores) == len(extensions) == 1, "Missing/ambiguous native/Python core")
    native = run / "einsums-stage/lib/libEinsums-abi1.so.2.0.0"
    expected_paths = {str(p.resolve()) for p in (cores[0], extensions[0], native)}
    require(set(metadata["binary_sha256"]) == expected_paths, "Incomplete/foreign binary tuple")
    require(Path(metadata["world"]["library_path"]).resolve() == native.resolve()
            and {str(Path(p).resolve()) for p in metadata["mapped_libraries"] if "libEinsums" in Path(p).name}
            == {str(native.resolve())}, "Wrong/mixed native library mapping")
    require(set(metadata["binary_sha256"].values()) != OLD_TUPLE, "Rejected fixed failed-build tuple")
    for binary, expected in metadata["binary_sha256"].items():
        require(Path(binary).resolve().is_relative_to(run) and digest(binary) == expected,
                "Different prepared binary")
    prefix_libraries = {str(p.resolve()) for p in (run / "einsums-stage/lib").glob("*.so*") if p.is_file()}
    audited = prefix_libraries | {str(cores[0].resolve()), str(extensions[0].resolve())}
    require(audited == set(metadata["elf_audit"])
            and expected_paths == set(metadata["closure"]),
            "Missing installed ELF/actual closure evidence")
    for path, record in metadata["elf_audit"].items():
        require(digest(path) == record["sha256"], "Audited installed library changed")
    for binary, closure in metadata["closure"].items():
        if binary != str(cores[0].resolve()):
            require({item["soname"] for item in sdk["libraries"].values()} <= set(closure),
                    "Partial actual CUDA dependency closure")
        for item in closure.values():
            require(digest(item["path"]) == item["sha256"], "Actual dependency changed after closure audit")
    physical_path = run / "physical-process-proof.json"
    process = json.loads(physical_path.read_text())
    require(process["exit_status"] == 0 and process["source_pin"] == pin and process["run"] == str(run),
            "Physical process did not pass on this source/run")
    require(str(process["job"]).isdigit() and run.name.endswith("-" + str(process["job"])),
            "Physical process proof belongs to another job")
    require(metadata["job"] == process["job"] and isinstance(process["pid"], int) and process["pid"] > 0,
            "Metadata/physical process identity mismatch")
    require(process["binary_sha256"] == metadata["binary_sha256"]
            and process["metadata_sha256"] == digest(metadata_path)
            and process["xml"] == str(Path(xml).resolve())
            and process["xml_sha256"] == digest(xml), "Historical/foreign physical process proof")
    expected_components = {family: item["path"] for family, item in sdk["libraries"].items()}
    require(process["cuda_component_maps"] == expected_components, "Foreign physical CUDA components")
    observation = api["runtime_observation"]
    require(observation["source_pin"] == pin and observation["run"] == str(run)
            and observation["job"] == process["job"]
            and observation["binary_sha256"] == metadata["binary_sha256"]
            and observation["metadata_sha256"] == digest(metadata_path)
            and isinstance(observation["pid"], int) and observation["pid"] > 0
            and observation["cuda_component_maps"] == expected_components,
            "API calculations did not prove the actual new runtime")
    for proof in (process, observation):
        actual = audit_mapped_libraries(proof["mapped_library_sha256"], sdk, run / "einsums-stage",
                                       run / "psi4-build/stage", run / "environment")
        require(actual == proof["mapped_library_sha256"], "Foreign/changed native map in actual test process")
    require(Path(xml).resolve().is_relative_to(run / "psi4-build"), "Historical/foreign physical XML")
    modules = {}
    for name in MODULES:
        relative = Path("psi4/driver/procrouting/sapt") / name
        staged = run / "psi4-build/stage/lib" / relative
        require(digest(staged) == digest(source / relative), "Different staged scientific source")
        modules[str(staged)] = digest(staged)
    proofs = {str(path): digest(path) for path in
              (metadata_path, api_path, physical_path, receipts_path, build_audit_path, Path(xml), sdk_path,
               run / "einsums-build/disp0-cuda-targets.tsv",
               run / "einsums-build/disp0-install-targets.tsv", run / "environment-explicit.txt")}
    require(metadata["sdk_contract_sha256"] == proofs[str(run / "cuda13/sdk-contract.json")],
            "SDK contract changed after imported closure proof")
    configuration = configuration_gate(run, sdk)
    for name in ("einsums-build/CMakeCache.txt", "psi4-build/CMakeCache.txt",
                 "compiler-version.txt", "nvcc-version.txt", "cmake-version.txt",
                 "configuration-source-sha256.txt", "compiler-identities.txt"):
        proofs[str(run / name)] = digest(run / name)
    targets = audit_target_report(run / "einsums-build/disp0-cuda-targets.tsv", sdk)
    installs = audit_install_report(run / "einsums-build/disp0-install-targets.tsv", sdk, run / "environment")
    return {"status": "CPU_PREPARATION_PASS_PENDING_JOB_EXIT", "source_pin": pin, "job": str(process["job"]),
            "scientific_baseline": SCIENCE_PIN, "einsums_pin": E2_PIN, "run": str(run),
            "proof_sha256": proofs, "binary_sha256": metadata["binary_sha256"],
            "scientific_module_sha256": modules, "physical_cpu_gate": physical_gate(xml),
            "cuda_target_identities": targets,
            "installed_loader_intent": installs,
            "actual_configuration": configuration,
            "effective_build_inputs": build_audit,
            "actual_gpu_correctness_verified": False,
            "scope": "New installed CPU/API/physical preparation, not GPU numerical correctness"}


def admit(run, job, status_path, source, pin):
    run = Path(run).resolve()
    require(job != "14196330", "Fixed failed origin is ineligible for new-build admission")
    document = json.loads((run / "preparation-result.json").read_text())
    require(document["status"] == "CPU_PREPARATION_PASS_PENDING_JOB_EXIT", "Incomplete build preparation")
    require(document["job"] == job, "Different CPU job preparation document")
    raw = Path(status_path).read_text()
    fields = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(fields.get("exit_code") == "0" and fields.get("term_received") == "0"
            and fields.get("run") == str(run) and "CPU_PREPARATION_EXIT_ZERO" in raw,
            "Failed/signaled CPU build remains inadmissible")
    accounting = command(["sacct", "-j", job, "-n", "-P", "-o", "JobID,State,ExitCode,Account,QOS"])
    rows = {line.split("|")[0]: line.split("|") for line in accounting.splitlines()}
    for name in (job, job + ".batch"):
        require(name in rows and rows[name][1:3] == ["COMPLETED", "0:0"], "CPU allocation not cleanly completed")
    require(rows[job][3:5] == ["gts-cs207-chemx", "embers"], "Wrong approved CPU preparation route")
    # Re-run read-only source/package/proof binding, not calculations/builds.
    xml = next(Path(p) for p in document["proof_sha256"] if p.endswith("results.xml"))
    require(collect(run, source, pin, xml) == document, "Stale or changed preparation evidence")
    return {**document, "status": "CPU_RUNTIME_PREPARATION_ONLY", "job": job,
            "job_status_sha256": digest(status_path), "accounting": accounting,
            "gpu_gate": "Still requires explicit resource/capacity checkpoint and actual numerical/device gate"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "admit"))
    for name in ("run", "source", "pin", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--xml")
    parser.add_argument("--job")
    parser.add_argument("--status")
    args = parser.parse_args()
    if args.action == "collect":
        require(args.xml, "Actual new-stage physical XML required")
        result = collect(args.run, args.source, args.pin, args.xml)
    else:
        require(args.job and args.status, "Actual allocation/status evidence required")
        result = admit(args.run, args.job, args.status, args.source, args.pin)
    write_new(args.out, result)
    print(json.dumps({"status": result["status"], "actual_gpu_correctness_verified": False}))


if __name__ == "__main__":
    main()
