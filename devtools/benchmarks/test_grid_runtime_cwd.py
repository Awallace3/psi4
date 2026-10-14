"""Exercise the real wrapper provenance block from a shadowing source cwd."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.mark.parametrize("wrapper", ["grid_evidence_run.sh", "grid_memory_run.sh"])
def test_provenance_uses_staged_package_from_source_checkout(tmp_path, wrapper):
    # A source-only psi4 in cwd wins over --psiapi PYTHONPATH for python stdin.
    source = tmp_path / "source"
    (source / "psi4").mkdir(parents=True)
    (source / "psi4/__init__.py").write_text(
        "raise ImportError('SOURCE_CHECKOUT_SHADOWED_STAGED_PSI4')\n")
    science = tmp_path / "science"
    stage = science / "build_grid_host/stage"
    package = stage / "lib/psi4"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from . import core\n")
    core = package / "core.py"
    core.write_text("# stand-in for the validated native core\n")
    (stage / "bin").mkdir()
    psiapi = stage / "bin/psi4"
    psiapi.write_text("#!/bin/sh\nprintf '%s\\n' " +
                      shlex.quote("export PYTHONPATH=" + shlex.quote(str(stage / "lib"))) + "\n")
    psiapi.chmod(0o755)
    env_root = tmp_path / "env"
    (env_root / "bin").mkdir(parents=True)
    (env_root / "bin/python").symlink_to(sys.executable)
    validation = tmp_path / "validation"
    (validation / "host").mkdir(parents=True)
    receipt = dict(core_sha256=hashlib.sha256(core.read_bytes()).hexdigest())
    (validation / "host/import.json").write_text(json.dumps(receipt))
    out = tmp_path / "out"
    out.mkdir()

    # Execute the actual production shell block, not a rewritten approximation.
    text = Path(__file__).with_name(wrapper).read_text()
    start = text.index('        build="$science/build_grid_$variant"')
    end = text.index("\nPY\n", start) + len("\nPY\n")
    block = text[start:end]
    result = subprocess.run(["bash", "-eu", "-c", block], cwd=source, text=True,
                            capture_output=True, env=dict(os.environ,
                                science=str(science), env=str(env_root), variant="host",
                                validation=str(validation), out=str(out)))
    assert result.returncode == 0, result.stderr
    assert json.loads((out / "binary-host.json").read_text()) == receipt
