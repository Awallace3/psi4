"""Bind benchmark approval to staged binaries, metadata, and build configuration."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def capture(package, source, built_commit):
    package, source = Path(package).resolve(), Path(source).resolve()
    # Harness-only changes need not invalidate the compiled scientific sources.
    trees = {name: git(source, "rev-parse", f"{built_commit}:{name}")
             for name in ("psi4", "cmake", "CMakeLists.txt", "external")}
    for name, tree in trees.items():
        assert git(source, "rev-parse", f"HEAD:{name}") == tree, name
    metadata = {}
    for node in ast.parse((package / "metadata.py").read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    metadata[target.id] = node.value.value
    version = metadata["__version_long"]
    assert built_commit.startswith(version.rsplit("+", 1)[1]), (version, built_commit)
    assert str(metadata["__version_is_clean"]).lower() == "true", metadata
    core, = package.glob("core*.so")
    cache = package.parents[2] / "CMakeCache.txt"
    assert cache.exists(), cache
    return {"source": str(source), "package": str(package), "built_commit": built_commit,
            "source_trees": trees, "version": version,
            "files": {str(f): sha256(f) for f in (core, package / "metadata.py", cache)},
            "note": "Commit checked against staged version metadata and source trees; file hashes bind smoke approval."}


def verify(receipt):
    actual = capture(receipt["package"], receipt["source"], receipt["built_commit"])
    assert actual == receipt, "Build artifacts changed since smoke validation"
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--built-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(capture(args.package, args.source, args.built_commit),
                                      indent=2, allow_nan=False) + "\n")
