"""Allocated-GPU validation of a pinned direct-JK build; retain per-test output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu_compare"))
from cuest_sdk import verify_build_manifest, runtime_identity


RKS_CASES = tuple(
    [f"test_cuest_direct_dft_energy[{functional}]"
     for functional in ("svwn", "pbe", "m06-l", "b3lyp", "wb97x", "lc-wpbe")]
    + [f"test_cuest_direct_cartesian_and_symmetry[{puream}]" for puream in ("True", "False")]
    + ["test_cuest_direct_with_gpu_xc"])


def test_targets(repo, suite):
    direct = repo / "tests/pytests/test_cuest_direct_jk.py"
    if suite == "rks":
        return [f"{direct}::{name}" for name in RKS_CASES]
    if suite == "full":
        return [str(direct), str(repo / "tests/pytests/test_cuest_jk.py")]
    raise ValueError(f"Unknown validation suite: {suite}")


def validation_success(result, cases, suite):
    if result != 0 or any(any(c.find(tag) is not None for tag in
                              ("failure", "error", "skipped")) for c in cases):
        return False
    if suite == "rks":
        names = [c.attrib["name"] for c in cases]
        return len(names) == len(RKS_CASES) and set(names) == set(RKS_CASES)
    if suite == "full":
        return len(cases) >= 18
    raise ValueError(f"Unknown validation suite: {suite}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--compiled-commit", help="Pinned core source, defaults to harness commit")
    parser.add_argument("--expected-gpu", default="H200", choices=["H200", "H100"],
                        help="Hopper device expected for this correctness allocation")
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=["full", "rks"], default="full",
                        help="RKS excludes UKS and unrelated DF regression tests")
    args = parser.parse_args()
    manifest = json.loads((args.build / "build-manifest.json").read_text())
    verify_build_manifest(manifest)
    compiled_commit = args.compiled_commit or args.commit
    assert manifest["commit"] == compiled_commit
    assert manifest["cuest_direct_jk_available"], "Direct API compile/link probe failed"
    assert subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                                   text=True).strip() == args.commit
    assert not subprocess.check_output(
        ["git", "-C", str(args.repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    args.out.mkdir(parents=True, exist_ok=False)

    import psi4
    import pytest

    package = Path(manifest["package"]).resolve()
    assert Path(psi4.__file__).resolve().parent == package
    import_identity = runtime_identity("psi4", psi4.core, package, require_version=False)
    core = next(package.glob("core*.so"))
    assert hashlib.sha256(core.read_bytes()).hexdigest() == manifest["core_sha256"]
    library = Path(manifest["cuest_library"])
    assert hashlib.sha256(library.read_bytes()).hexdigest() == manifest["cuest_library_sha256"]
    linkage = subprocess.check_output(["ldd", str(core)], text=True)
    assert "not found" not in linkage
    cuest_line = next(line for line in linkage.splitlines() if line.strip().startswith("libcuest.so"))
    assert str(library.parent) in cuest_line, linkage
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"], text=True)
    assert args.expected_gpu in gpu, gpu
    psi4.set_memory("4 GiB")
    psi4.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    out = args.out.resolve()

    class Evidence:
        def pytest_runtest_setup(self, item):
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.core.clean_timers()
            filename = item.nodeid.split("::", 1)[-1].replace("/", "_")
            psi4.set_output_file(str(out / (filename + ".psi4.out")), False)

        def pytest_runtest_teardown(self, item, nextitem):
            if item.name == "test_cuest_direct_with_gpu_xc":
                paths = ["/".join(t["timer_path"]) for t in psi4.core.get_timer_records().values()]
                # This particular Phoenix build explicitly selects CUDA LibXC.
                assert any("CUDA LibXC Functional" in p for p in paths), paths
            psi4.core.close_outfile()

    # Do not load the generic conftest: its finalizer deletes chemistry output.
    result = pytest.main([
        "--noconftest", "-v", "-ra", "--tb=short",
        "--basetemp", str(out / "pytest-temp"),
        "--junitxml", str(out / "junit.xml"),
        *test_targets(args.repo, args.suite),
    ], plugins=[Evidence()])
    root = ET.parse(out / "junit.xml").getroot()
    cases = root.findall(".//testcase")
    failures = [c.attrib["name"] for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skips = [c.attrib["name"] for c in cases if c.find("skipped") is not None]
    native_identity = runtime_identity("psi4", psi4.core, package)
    success = validation_success(result, cases, args.suite)
    report = dict(
        success=success, suite=args.suite, pytest_exit=int(result), tests=len(cases), failures=failures, skips=skips,
        harness_commit=args.commit, compiled_commit=compiled_commit,
        build_job=manifest["job_id"], job=os.environ["SLURM_JOB_ID"],
        qos=os.environ["SLURM_JOB_QOS"], gpu=gpu, expected_gpu=args.expected_gpu,
        core_sha256=manifest["core_sha256"],
        cuest_library_sha256=manifest["cuest_library_sha256"], linkage=linkage,
        sdk_import_identity=import_identity, sdk_runtime_identity=native_identity,
        protocol="CPU DIRECT vs cuEST four-center DIRECT; CPU XC isolates JK numerics. "
                 "Separate CUDA-XC smoke; DF-JK and UKS tests only in full suite. Native CPU/device grids "
                 "are NOT claimed identical by the GPU-XC smoke.",
    )
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
