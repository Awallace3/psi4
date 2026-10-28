# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit source-only test loader. Never import a real quantum package."""
import importlib.util
from pathlib import Path
import sys
import types

SOURCE = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting"
_FAKE = types.ModuleType("psi4")
_FAKE._source_only_fake = True
_FAKE.core = types.SimpleNamespace()


def load_source(name, core=None):
    previous = sys.modules.get("psi4")
    if core is not None:
        _FAKE.core = core
    sys.modules["psi4"] = _FAKE
    if "_isapol_source" not in sys.modules:
        package = types.ModuleType("_isapol_source")
        package.__path__ = [str(SOURCE)]
        sys.modules[package.__name__] = package
        # _scf_seal is pure definitions. Bypass scf_proc.__init__, which loads
        # the SCF driver, while retaining the actual seal module under test.
        scf = types.ModuleType("_isapol_source.scf_proc")
        scf.__path__ = [str(SOURCE / "scf_proc")]
        sys.modules[scf.__name__] = scf
    try:
        fullname = "_isapol_source." + name
        if fullname not in sys.modules:
            spec = importlib.util.spec_from_file_location(fullname, SOURCE / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[fullname]
                raise
        return sys.modules[fullname]
    finally:
        # Native pytest may already have imported Psi4. Never replace it there.
        # Standalone source-only processes retain ONLY the marked fake instead.
        if previous is not None:
            sys.modules["psi4"] = previous
