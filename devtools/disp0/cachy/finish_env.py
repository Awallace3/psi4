"""Install the reviewed development-only proposal into a NEW private prefix.

Preserves bootstrap-01, its successful baseline clone and failed GUI-toolkit
proposal. Never reruns that proposal or modifies the original baseline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

from bootstrap_env import run


def identity(record):
    return (record["name"], record["version"],
            record.get("build_string", record.get("build")))


def freeze(baseline, actions, cached):
    records = {r["name"]: r for r in baseline}
    original = dict(records)
    for record in actions.get("UNLINK", []):
        records.pop(record["name"], None)
    available = {identity(r): r for r in cached}
    available.update({identity(r): r for r in actions.get("FETCH", [])})
    for record in actions.get("LINK", []):
        complete = dict(record)
        complete.update(available.get(identity(record), {}))
        records[record["name"]] = complete
    lines = ["@EXPLICIT"]
    for name, record in sorted(records.items()):
        url = record.get("url")
        # Use the longstanding @EXPLICIT URL#md5 syntax. Preserve the stronger
        # repodata SHA256 alongside it in expected-packages.json, not a guessed
        # URL fragment encoding for different conda versions.
        checksum = record.get("md5")
        if not url or not checksum:
            raise RuntimeError(f"Missing exact archive identity for {name}; do not solve again")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "conda.anaconda.org" or parsed.query:
            raise RuntimeError(f"Unexpected package origin for {name}")
        if not parsed.path.startswith("/conda-forge/"):
            # Baseline doc/theme artifacts can originate in Psi4's channel.
            # Permit only an EXACT unchanged existing URL/hash/build, never a
            # newly solved package or a channel substitution.
            old = original.get(name)
            if (not old or identity(old) != identity(record) or
                    old.get("url") != url or old.get("md5") != checksum):
                raise RuntimeError(f"Unreviewed new package origin for {name}")
        lines.append(url.split("#")[0] + "#" + checksum)
    return "\n".join(lines) + "\n", records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", choices=("development-install-02", "development-install-03"),
                        default="development-install-02")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    source, metadata = root / "source", root / "metadata"
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Source must be clean")
    # NEW receipt/prefix only. The failed first recipe remains immutable.
    logs = metadata / args.attempt
    logs.mkdir(mode=0o700)
    runtime, clang = root / "environment-v2", root / "clang23"
    if runtime.exists() or clang.exists():
        raise RuntimeError("New absent environment prefixes required")
    proposal_bytes = (metadata / "cuda-development-proposal/stdout.json").read_bytes()
    proposal = json.loads(proposal_bytes)
    if not proposal.get("success"):
        raise RuntimeError("Development-only proposal did not pass")
    baseline = [json.loads(p.read_text()) for p in (root / "environment/conda-meta").glob("*.json")]
    cached = [json.loads(p.read_text()) for p in (root / "conda-pkgs").glob("*/info/repodata_record.json")]
    lock, expected = freeze(baseline, proposal["actions"], cached)
    versions = {name: record["version"] for name, record in expected.items()}
    if versions.get("cuda-nvcc") != "13.3.73" or versions.get("cuda-version") != "13.3":
        raise RuntimeError("CUDA compiler/ABI proposal drift")
    if any(name.startswith(("nsight", "cuda-visual-tools")) or name in ("psi4", "einsums", "pyeinsums")
           for name in expected):
        raise RuntimeError("Unexpected GUI or old scientific package")
    lockfile = logs / "runtime-explicit.txt"
    lockfile.write_text(lock)
    (logs / "expected-packages.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    (logs / "input-identity.json").write_text(json.dumps({
        "proposal_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
        "runtime_lock_sha256": hashlib.sha256(lock.encode()).hexdigest(),
        "source_pin": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2) + "\n")
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("CONDA_", "MAMBA_")) or key in (
            "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "CPATH", "GCC_ROOT",
            "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH",
            "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CC", "CXX", "FC", "CFLAGS", "CXXFLAGS",
            "LDFLAGS", "CMAKE_PREFIX_PATH",
        ):
            env.pop(key, None)
    conda = Path.home() / "miniconda3/bin/conda"
    env.update(CONDA_PKGS_DIRS=str(root / "conda-pkgs"),
               CONDARC=str(source / "devtools/disp0/condarc"),
               PATH=str(conda.parent) + ":/usr/bin:/bin",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MAMBA_NUM_THREADS="2", MAMBA_EXTRACT_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    status = metadata / "SETUP_STATUS.md"
    phase = "runtime-create"
    status.write_text("# Cachy setup\n\nDevelopment-only private installation IN PROGRESS.\n"
                      "Do not run another installer. Prior failed bootstrap retained.\n")
    try:
        run([str(conda), "create", "-y", "--prefix", str(runtime), "--file", str(lockfile)],
            logs, phase, env, 2700)
        installed = {r["name"]: r for r in
                     (json.loads(p.read_text()) for p in (runtime / "conda-meta").glob("*.json"))}
        if {identity(r) for r in installed.values()} != {identity(r) for r in expected.values()}:
            raise RuntimeError("Installed package identities differ from frozen proposal")
        phase = "clang23-create"
        run([str(conda), "create", "-y", "--prefix", str(clang), "--override-channels", "-c",
             "conda-forge", "--strict-channel-priority", "clangdev=23.1.2",
             "clangxx=23.1.2", "llvmdev=23.1.2"], logs, phase, env, 1800)
        for label, prefix in (("runtime", runtime), ("clang23", clang)):
            phase = label + "-explicit"
            run([str(conda), "list", "--explicit", "--prefix", str(prefix)], logs, phase, env, 60)
        for label, command, required in (
            ("compiler-version", [str(runtime / "bin/x86_64-conda-linux-gnu-c++"), "-dumpfullversion"], "14.3.0"),
            ("nvcc-version", [str(runtime / "bin/nvcc"), "--version"], "V13.3.73"),
        ):
            phase = label
            run(command, logs, phase, env, 30)
            if required not in (logs / (phase + ".stdout")).read_text():
                raise RuntimeError(f"Actual {label} disagrees")
        phase = "python-abi"
        run([str(runtime / "bin/python"), "-B", "-c",
             "import sys,sysconfig; assert sys.version_info[:2]==(3,13); "
             "assert not sysconfig.get_config_var('Py_GIL_DISABLED'); print(sys.version)"],
            logs, phase, env, 30)
        with (metadata / "setup.env").open("a") as stream:
            stream.write(f"DISP0_ENV={runtime}\n")
        status.write_text("# Cachy setup\n\nPrivate CUDA development runtime and LLVM23 environments installed.\n"
                          "Use DISP0_ENV from setup.env (environment-v2), NOT the preserved baseline.\n"
                          "cuEST canonical artifact/source receipts are separate. GPU LibXC7.1.2,\n"
                          "Einsums064 and whole Psi4 still need private compilation/import validation.\n"
                          "No GPU numerical readiness claimed. User will launch the local agent.\n")
    except BaseException as error:
        status.write_text(f"# Cachy setup\n\nDevelopment installation FAILED/INCOMPLETE at {phase}:\n"
                          f"{type(error).__name__}: {error}\n\nPreserve all attempts. No automatic retry.\n")
        raise


if __name__ == "__main__":
    main()
