# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Compute-only explicit import/version gate; no Psi4, SCF or benchmark."""
import importlib
import json
import os
from pathlib import Path
import socket
import sys


def main():
    host = socket.gethostname().split(".")[0]
    if (not os.environ.get("SLURM_JOB_ID") or host.startswith("login")
            or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]):
        raise ValueError("compute allocation only")
    plan = json.loads(Path(__file__).with_name("build_smoke_plan.json").read_text())
    old = Path(plan["dependency_source"])
    env = Path(plan["environment"])
    run = Path(plan["run"])
    sys.path[:0] = [str(run / "deps/python"),
                    str(env / "lib/python3.13/site-packages")]
    records = {}
    for name, version in plan["python_dependencies"].items():
        module = importlib.import_module(name)
        if module.__version__ != version:
            raise ValueError(f"{name}: {module.__version__} != {version}")
        if not Path(module.__file__).resolve().is_relative_to((old / "lib" / name).resolve()):
            raise ValueError("wrong Python dependency path: " + name)
        records[name] = dict(version=module.__version__, path=module.__file__)
    with (run / "metadata/python-dependencies.json").open("x") as stream:
        json.dump(records, stream, indent=2)
        stream.write("\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
