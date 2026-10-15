"""Allocated-GPU validation of a pinned direct-JK build; retain per-test output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.build / "build-manifest.json").read_text())
    assert manifest["commit"] == args.commit
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
    assert "H200" in gpu, gpu
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
        str(args.repo / "tests/pytests/test_cuest_direct_jk.py"),
        str(args.repo / "tests/pytests/test_cuest_jk.py"),
    ], plugins=[Evidence()])
    root = ET.parse(out / "junit.xml").getroot()
    cases = root.findall(".//testcase")
    failures = [c.attrib["name"] for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skips = [c.attrib["name"] for c in cases if c.find("skipped") is not None]
    success = result == 0 and len(cases) >= 18 and not failures and not skips
    report = dict(
        success=success, pytest_exit=int(result), tests=len(cases), failures=failures, skips=skips,
        commit=args.commit, build_job=manifest["job_id"], job=os.environ["SLURM_JOB_ID"],
        qos=os.environ["SLURM_JOB_QOS"], gpu=gpu, core_sha256=manifest["core_sha256"],
        cuest_library_sha256=manifest["cuest_library_sha256"], linkage=linkage,
        protocol="CPU DIRECT vs cuEST four-center DIRECT; CPU XC isolates JK numerics. "
                 "Separate CUDA-XC smoke and existing DF-JK regressions. Native CPU/device grids "
                 "are NOT claimed identical by the GPU-XC smoke.",
    )
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
