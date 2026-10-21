# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit source-only test loader. Never import a real quantum package."""
import importlib.util
from pathlib import Path
import sys
import types

SOURCE = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting"


def load_source(name, core=None):
    if "psi4" in sys.modules and not getattr(sys.modules["psi4"], "_source_only_fake", False):
        raise RuntimeError("source-only tests must run in a fresh no-quantum process")
    if "psi4" not in sys.modules:
        fake = types.ModuleType("psi4")
        fake._source_only_fake = True
        fake.core = types.SimpleNamespace() if core is None else core
        sys.modules["psi4"] = fake
        package = types.ModuleType("_isapol_source")
        package.__path__ = [str(SOURCE)]
        sys.modules[package.__name__] = package
    elif core is not None:
        sys.modules["psi4"].core = core
    fullname = "_isapol_source." + name
    if fullname not in sys.modules:
        spec = importlib.util.spec_from_file_location(fullname, SOURCE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    return sys.modules[fullname]
