#!/usr/bin/env python3
"""Freeze approved dirty code on a NEW branch without touching HEAD or index.

No push or remote operations. Only tracked changes and explicitly listed
untracked runtime/tests/tools are included; unrelated handoff docs are excluded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

EXTRA_FILES = [
    "psi4/driver/p4util/timing.py",
    "psi4/src/psi4/libfunctional/xc_timing.h",
    "tests/pytests/test_basis_parse_reuse.py",
    "tests/pytests/test_cuest_options.py",
    "tests/pytests/test_cuest_sad.py",
    "tests/pytests/test_dft_grid_switch.py",
    "tests/pytests/test_scf_lifecycle.py",
    "devtools/saptdft_grid_sweep.py",
    "devtools/saptdft_xc_profile.py",
    "devtools/saptdft_grid_gpu_validate.sh",
    "devtools/saptdft_grid_freeze.py",
    "devtools/saptdft_grid_job.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base", help="Optional checkpoint parent; original HEAD still remains untouched")
    parser.add_argument("--include", action="append", default=[], help="Additional explicit runtime/input files")
    parser.add_argument("--message", default="Checkpoint opt-in XC grid switching and diagnostic microtiming")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    worktree = Path(args.worktree).resolve()
    manifest = Path(args.manifest).resolve()
    if worktree.exists() or manifest.exists():
        raise SystemExit("Refusing to overwrite an existing worktree or manifest")

    def git(*command, env=None):
        environment = dict(os.environ if env is None else env, GIT_OPTIONAL_LOCKS="0")
        return subprocess.check_output(["git", "-C", str(root), *command],
                                       env=environment).decode().strip()

    subprocess.run(["git", "-C", str(root), "check-ref-format", "--branch", args.branch], check=True,
                   stdout=subprocess.DEVNULL)
    ref = "refs/heads/" + args.branch
    exists = subprocess.run(["git", "-C", str(root), "show-ref", "--verify", "--quiet", ref])
    if exists.returncode != 1:
        raise SystemExit("Destination branch already exists or could not be checked")
    index = Path(git("rev-parse", "--path-format=absolute", "--git-path", "index"))

    def state():
        return dict(head=git("rev-parse", "HEAD"), branch=git("symbolic-ref", "HEAD"),
                    index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                    diff_sha256=hashlib.sha256(git("diff", "--binary", "HEAD").encode()).hexdigest(),
                    status=git("status", "--porcelain"))

    before = state()
    base = git("rev-parse", args.base or before["head"])
    with tempfile.TemporaryDirectory(prefix="psi4-grid-freeze-") as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git("read-tree", base, env=env)
        git("add", "-u", env=env)
        git("add", "-f", "--", *EXTRA_FILES, *args.include, env=env)
        tree = git("write-tree", env=env)
        files = git("diff", "--name-only", "--cached", base, env=env).splitlines()
        pin = git("commit-tree", tree, "-p", base, "-m", args.message, env=env)
    assert state() == before, "Source worktree changed while freezing; do not publish"
    # All-zero expected old object atomically refuses an existing destination.
    git("update-ref", ref, pin, "0" * len(pin))
    git("worktree", "add", str(worktree), args.branch)
    assert state() == before, "Original HEAD/index/dirty files changed"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(dict(
        source=str(root), worktree=str(worktree), branch=args.branch,
        base_commit=base, commit=pin, tree=tree,
        source_state=before, checkpoint_files=files,
        excluded_untracked=["cphf-check.md", "mgga-check.md"],
        preserved_source_state=True,
    ), indent=2) + "\n")
    print(json.dumps(dict(commit=pin, branch=args.branch, manifest=str(manifest)), indent=2))


if __name__ == "__main__":
    main()
