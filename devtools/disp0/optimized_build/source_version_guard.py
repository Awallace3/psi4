"""Fail before configuration if Psi4's real release history cannot version it.

This generates preparation evidence in a NEW run only. It never patches the
source, installed metadata, import checks, or a preserved failed build.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

TAG = "v1.11"
TAG_OBJECT = "75fa0e14aa16a868a00eced50eaaa2f8658aa806"
TAG_COMMIT = "f16f4ea2ff2351cc2ab877d9b3887fcf81c2a670"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate(evidence):
    require(evidence["shallow"] == "false", "Shallow Psi4 history is not admissible")
    require(evidence["tag_object"] == TAG_OBJECT and evidence["tag_commit"] == TAG_COMMIT,
            "Wrong annotated Psi4 release identity")
    require(evidence["tag_type"] == "tag" and evidence["ancestor_exit"] == 0,
            "Release must be annotated and an ancestor of exact source")
    require(evidence["describe"].startswith(TAG + "-"), "Psi4 describe disagrees with fallback release")
    require(evidence["clean"], "Source is dirty")
    metadata = evidence["metadata"]
    require(metadata["__version_last_release"] == "1.11", "Wrong generated release series")
    version = metadata["__version__"]
    require(version.startswith("1.12a1.dev") and version.removeprefix("1.12a1.dev").isdigit(),
            "Missing/undefined generated development version")
    require(metadata["__version_long"] == version + "+" + evidence["head"][:7]
            and metadata["__version_is_clean"] == "True",
            "Generated metadata does not bind clean source HEAD")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    args.out.mkdir()  # NEW evidence only, no adoption/overwrite.

    def git(*argv):
        return subprocess.run(["git", "-C", str(source), *argv],
                              capture_output=True, text=True, timeout=15, check=True).stdout.strip()

    ancestor = subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", TAG, "HEAD"],
                              capture_output=True, text=True, timeout=15)
    evidence = {
        "shallow": git("rev-parse", "--is-shallow-repository"),
        "tag_object": git("rev-parse", f"refs/tags/{TAG}"),
        "tag_commit": git("rev-parse", f"refs/tags/{TAG}^{{}}"),
        "tag_type": git("cat-file", "-t", f"refs/tags/{TAG}"),
        "ancestor_exit": ancestor.returncode, "ancestor_stderr": ancestor.stderr,
        "describe": git("describe", "--abbrev=7", "--long", "--always", "HEAD"),
        "head": git("rev-parse", "HEAD"), "clean": not git("status", "--porcelain"),
    }
    command = [sys.executable, "-B", str(source / "psi4/versioner.py"),
               "--metaout", str(args.out / "metadata.py"),
               "--cmakeout", str(args.out / "metadata.cmake"), "--formatonly"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    evidence["generator"] = {"argv": command, "exit": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr}
    with (args.out / "raw-evidence.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
    require(result.returncode == 0, "Upstream version generator failed")
    evidence["metadata"] = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse((args.out / "metadata.py").read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    validate(evidence)
    with (args.out / "validated.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
