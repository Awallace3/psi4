"""Separate failed-build artifact recovery admission, never a GPU/physics pass.

The original successful-build admission is deliberately unchanged. This route
accepts only the explicitly approved job/artifacts, a whole fresh deployment,
and new proofs from that deployed prefix. It never edits the compilation run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

COMPILE_PIN = "587e172a33afaadb00eb7f574a5089533f7fc755"
RECIPE_PIN = "14e135d23c9e9de3a68ca5ead0eed43413cc86c8"
JOB = "14196330"
E2_PIN = "d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62"
P = Path("/storage/project/r-cs207-0/awallace43")
ORIGIN = P / f"runs/saptdft-cuest-disp0/build-{COMPILE_PIN}-{JOB}"
EXPECTED = {
    "psi4-build/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so":
        "b3cfe2f2261533df9a9018731d9fe2bffea41cfe70f53f14195caaee7de8697d",
    "einsums-stage/lib/einsums/_core.cpython-313-x86_64-linux-gnu.so":
        "8f23c69b79613f9b94eee95d4735e8f88f67be7c98867290f7546ca56bdc6833",
    "einsums-stage/lib/libEinsums-abi1.so.2.0.0":
        "9e31b5ff8cf28e796b04ee74853c4923f06acf03f1af24dfddd75b79de583cb7",
}
MODULES = ("sapt_jk_terms_ein.py", "sapt_mp2_terms_ein.py",
           "sapt_dispersion_gpu.py", "sapt_proc.py")
CHECKS = {
    "gemm-denominator-division-reduction-1x1",
    "gemm-denominator-division-reduction-3x4",
    "gemm-denominator-division-reduction-2x1",
    "default-graph-replay-strided-view-mutation",
    "pool-view-keepalive-reset",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args],
                                   text=True, timeout=10).strip()


def verify_sources(science, harness, pin):
    require(git(science, "rev-parse", "HEAD") == COMPILE_PIN, "Wrong original compile source")
    require(git(harness, "rev-parse", "HEAD") == pin and len(pin) == 40, "Wrong admission source pin")
    for source in (science, harness):
        require(not git(source, "diff", "HEAD", "--name-only"), "Uncommitted source changes")
        require(not git(source, "diff", "--cached", "--name-only"), "Staged source changes")
    require(not git(harness, "diff", COMPILE_PIN, pin, "--name-only", "--", "psi4/"),
            "Recovered science differs from reviewed compilation baseline")
    script = Path(__file__).resolve()
    require(script == harness / "devtools/disp0/recovered_runtime.py",
            "Executing admission is not from its immutable source")
    require(git(harness, "hash-object", str(script)) ==
            git(harness, "rev-parse", f"{pin}:devtools/disp0/recovered_runtime.py"),
            "Executing admission differs from its pinned Git blob")


def compilation_provenance():
    names = ("environment-explicit.txt", "nvcc-version.txt", "cuda-header-runtime.txt",
             "einsums-build/CMakeCache.txt", "psi4-build/CMakeCache.txt")
    result = {name: {"path": str(ORIGIN / name), "sha256": digest(ORIGIN / name)}
              for name in names}
    require("V13.3.73" in (ORIGIN / "nvcc-version.txt").read_text(), "Wrong original CUDA compiler")
    require("13030" in (ORIGIN / "cuda-header-runtime.txt").read_text(),
            "Wrong original CUDA header/link proof (not device evidence)")
    caches = {}
    for label in ("einsums-build", "psi4-build"):
        parsed = {}
        for line in (ORIGIN / label / "CMakeCache.txt").read_text().splitlines():
            if "=" in line and ":" in line and not line.startswith(("#", "//")):
                key, value = line.split("=", 1)
                parsed[key.split(":", 1)[0]] = value
        caches[label] = parsed
        require(parsed["CMAKE_CXX_COMPILER"].startswith(str(ORIGIN / "environment/bin") + "/"),
                "Original compiler was not isolated")
    e2 = caches["einsums-build"]
    require(e2["EINSUMS_WITH_CUDA"] == "ON" and e2["EINSUMS_WITH_CUDA_ARCHITECTURES"] == "90",
            "Wrong original GPU-enabled/sm90 build configuration")
    require(e2["EINSUMS_BUILD_PYTHON"] == "ON" and e2["EINSUMS_WITH_GPU_MOCK_DISCRETE"] == "OFF",
            "Wrong original native/Python or mock build")
    receipt = ORIGIN.parent / "logs" / f"submission-{COMPILE_PIN[:10]}.receipt"
    text = receipt.read_text()
    require(COMPILE_PIN in text and JOB in text, "Original source/job receipt mismatch")
    result["submission_receipt"] = {"path": str(receipt), "sha256": digest(receipt)}
    accounting = subprocess.check_output(
        ["sacct", "-j", JOB, "-n", "-P", "-o",
         "JobID,State,ExitCode,Elapsed,Account,QOS,AllocCPUS,ReqMem,NodeList"],
        text=True, timeout=10)
    result["original_accounting"] = verify_accounting(accounting)
    result["cache_configuration"] = caches
    result["toolchain_drift"] = "Preserve actual GNU14.3 build20 versus CPU-oracle build18; "
    result["toolchain_drift"] += "CMake versions/configuration retained verbatim, not assumed equal"
    return result


def verify_accounting(text):
    rows = {fields[0]: fields for line in text.splitlines()
            if len(fields := line.split("|")) >= 9}
    for name in (JOB, JOB + ".batch"):
        require(name in rows and rows[name][1:3] == ["FAILED", "1:0"],
                "Original job/batch failed-exit accounting mismatch")
    job = rows[JOB]
    require(job[4:8] == ["gts-cs207-chemx", "embers", "8", "48G"],
            "Wrong original allocation provenance")
    return {"raw": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "job_and_batch_remain_failed": True}


def validate(prefix, science, harness, pin):
    """Read-only, fail-closed; historical local CPU tests are not admission evidence."""
    prefix = Path(prefix).resolve(strict=True)
    science, harness = Path(science).resolve(strict=True), Path(harness).resolve(strict=True)
    verify_sources(science, harness, pin)
    manifest_path = prefix / "deployment-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "ARTIFACT_BYTES_VERIFIED", "Incomplete artifact deployment")
    require(manifest["recipe_pin"] == RECIPE_PIN, "Unreviewed deployment recipe")
    require(manifest["source_run"] == str(ORIGIN), "Wrong origin compilation run")
    require(manifest["source_job_status"] == "FAILED; never relabelled", "Failure was relabelled")
    require(Path(manifest["destination"]) == prefix, "Wrong deployed prefix")
    require(manifest["selected_binary_sha256"] == EXPECTED, "Unapproved binary tuple")
    anchor = Path(manifest["private_recipe_anchor"]).resolve(strict=True)
    require(prefix.is_relative_to(anchor) and prefix.name == "image", "Nonisolated deployment")
    require(anchor.stat().st_uid == os.geteuid() and anchor.stat().st_mode & 0o777 == 0o700,
            "Deployment recipe anchor is not private")
    require(git(anchor, "rev-parse", "HEAD") == RECIPE_PIN, "Deployment source pin moved")
    require(not git(anchor, "diff", "HEAD", "--name-only"), "Deployment recipe changed")
    require(not git(anchor, "diff", "--cached", "--name-only"), "Deployment recipe index changed")

    failure = ORIGIN.parent / f"logs/{JOB}.status"
    failure_text = failure.read_text()
    fields = dict(line.split("=", 1) for line in failure_text.splitlines() if "=" in line)
    require(fields.get("exit_code") == "1" and fields.get("term_received") == "0"
            and "FAILED_OR_SIGNALLED" in failure_text
            and "CPU_PREPARATION_EXIT_ZERO" not in failure_text, "Original failed status changed")
    require(fields.get("run") == str(ORIGIN), "Original failure names another source run")
    provenance = compilation_provenance()
    # Recheck ALL deployed files, modes and links, not only three selected ELF files.
    records = manifest["files_and_links"]
    selected = {}
    for root, target in ((ORIGIN / "psi4-build/stage", Path("psi4-build/stage")),
                         (ORIGIN / "einsums-build/lib/einsums", Path("einsums-stage/lib/einsums"))):
        require(root.is_dir() and not root.is_symlink(), "Missing complete origin tree")
        for path in (root, *root.rglob("*")):
            selected[str(target / path.relative_to(root))] = str(path)
    for name in ("libEinsums-abi1.so", "libEinsums-abi1.so.1", "libEinsums-abi1.so.2.0.0"):
        selected[f"einsums-stage/lib/{name}"] = str(ORIGIN / "einsums-build/lib" / name)
    require(set(records) == set(selected), "Partial artifact inventory")
    # Proof files are outside both import trees. No extra importable module,
    # bytecode, library or overlay may be hidden by a manifest-only traversal.
    actual = set()
    for relative, include_root in (("psi4-build/stage", True), ("einsums-stage/lib", False)):
        root = prefix / relative
        for path in ((root, *root.rglob("*")) if include_root else root.rglob("*")):
            actual.add(str(path.relative_to(prefix)))
    require(actual == set(records), "Unlisted or missing destination artifact")
    for relative, record in records.items():
        name = Path(relative)
        require(not name.is_absolute() and ".." not in name.parts, "Escaping inventory path")
        path = prefix / name
        require(record["destination"] == str(path), "Foreign destination in inventory")
        origin = Path(record["origin"])
        require(str(origin) == selected[relative], "Foreign origin in inventory")
        require(path.lstat().st_mode & 0o7777 == record["mode"], "Changed artifact mode")
        require(origin.lstat().st_mode & 0o7777 == record["mode"], "Changed origin mode")
        if record["kind"] == "file":
            require(not path.is_symlink() and path.is_file(), "Missing or replaced artifact file")
            require(origin.is_file() and not origin.is_symlink()
                    and origin.stat().st_size == record["bytes"]
                    and digest(origin) == record["sha256"], f"Changed origin bytes: {relative}")
            require(path.stat().st_size == record["bytes"]
                    and digest(path) == record["sha256"] == record["destination_sha256"],
                    f"Changed artifact bytes: {relative}")
        elif record["kind"] == "symlink":
            require(path.is_symlink() and os.readlink(path) == record["target"]
                    and path.resolve().is_relative_to(prefix), "Changed artifact link")
            require(origin.is_symlink() and os.readlink(origin) == record["target"]
                    and origin.resolve().is_relative_to(ORIGIN), "Changed origin link")
        else:
            require(record["kind"] == "directory" and path.is_dir() and not path.is_symlink(),
                    "Changed artifact directory")
            require(origin.is_dir() and not origin.is_symlink(), "Changed origin directory")
    require(sum(r.get("bytes", 0) for r in records.values()) == manifest["artifact_bytes"],
            "Incomplete byte inventory")
    for relative, expected in EXPECTED.items():
        require(records[relative]["sha256"] == expected, "Wrong approved artifact")
        require(digest(records[relative]["origin"]) == expected, "Original artifact changed")
        abi = manifest["abi"][relative]
        require(abi["origin"] == abi["destination"], "Changed recorded ABI/dependency closure")

    proof_paths = {name: prefix / "recovery-proofs" / name
                   for name in ("runtime-provenance.json", "api-probe.json")}
    proofs = {}
    for name, path in proof_paths.items():
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_mtime_ns >= manifest_path.stat().st_mtime_ns,
                "Proof missing, reused or older than deployment")
        proofs[name] = json.loads(path.read_text())
        require(proofs[name]["status"] == "PASS", f"Failed recovered-prefix proof: {name}")
    metadata, api = proofs["runtime-provenance.json"], proofs["api-probe.json"]
    require(metadata["binary_sha256"] ==
            {str((prefix / relative).resolve()): value for relative, value in EXPECTED.items()},
            "Metadata proves different binaries")
    world = metadata["world"]
    require(world["git_commit"] == E2_PIN and world["version_major"] == 2
            and world["compiler_major"] == 14, "Wrong E2/GNU world")
    require(Path(world["library_path"]).resolve() ==
            (prefix / "einsums-stage/lib/libEinsums-abi1.so.2.0.0").resolve(), "Wrong native world")
    for proof in (metadata, api):
        require(Path(proof["psi4"]).resolve().is_relative_to(prefix / "psi4-build/stage"),
                "Proof imported another Psi4 stage")
        require(Path(proof["einsums"]).resolve().is_relative_to(prefix / "einsums-stage"),
                "Proof imported another E2 Python package")
    native_maps = {str(Path(p).resolve()) for p in metadata["mapped_libraries"] if "libEinsums" in p}
    require(native_maps == {str(Path(world["library_path"]).resolve())}, "Mixed native E2 world")
    require(any("libcudart.so.13" in p for p in metadata["mapped_libraries"])
            and any("libcublas.so.13" in p for p in metadata["mapped_libraries"]),
            "Missing CUDA13 link/import maps; this is not device evidence")
    require(set(api["checks"]) == CHECKS
            and api["tolerances"] == {"rtol": 1e-11, "atol": 1e-12}, "Incomplete tiny API proof")
    module_hashes = {}
    for name in MODULES:
        relative = Path("psi4/driver/procrouting/sapt") / name
        staged = prefix / "psi4-build/stage/lib" / relative
        require(digest(staged) == digest(science / relative), "Scientific module overlay")
        module_hashes[str(staged)] = digest(staged)
    return {
        "status": "RECOVERED_RUNTIME_PREPARATION_ONLY",
        "admission_pin": pin, "compile_pin": COMPILE_PIN, "origin_compilation_job": JOB,
        "original_failed_status": {"path": str(failure), "sha256": digest(failure),
                                   "contents": failure_text},
        "original_compilation_provenance": provenance,
        "source_run": str(ORIGIN), "prefix": str(prefix),
        "deployment_manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
        "new_prefix_proofs": {name: {"path": str(path), "sha256": digest(path)}
                             for name, path in proof_paths.items()},
        "binary_sha256": metadata["binary_sha256"], "imported_scientific_modules": module_hashes,
        "world": world, "mapped_libraries": metadata["mapped_libraries"],
        "gate_requirement": "Actual recovered-stage native/eager/defaultCPU physical signed references "
                            "and actual CUDA numerical/device evidence before GPU physics acceptance",
        "historical_cpu_baseline": "26 tests /64 signed comparisons; support only, not a new runtime pass",
        "actual_gpu_correctness_verified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    for flag in ("prefix", "science-source", "admission-source", "admission-pin", "out"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    def timeout(signum, frame):
        raise TimeoutError("Recovered-runtime admission exceeded90s; preserve evidence, no retry")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    result = validate(args.prefix, args.science_source, args.admission_source, args.admission_pin)
    require(Path(args.out).parent.resolve() == Path(args.prefix).resolve() / "recovery-proofs",
            "Admission output must be in the new prefix proof directory")
    with open(args.out, "x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({"status": result["status"], "prefix": result["prefix"], "out": args.out}))


if __name__ == "__main__":
    main()
