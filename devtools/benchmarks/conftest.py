"""Make the benchmark scripts importable by their own tests.

These are standalone scripts, not an installed package, and the tests import
them by bare module name. Psi4's root `pytest.ini` sets
`--import-mode=importlib`, which deliberately does not put a test file's
directory on `sys.path`, so a run started anywhere but this directory fails at
collection with `ModuleNotFoundError`. Adding the directory here keeps the
scripts free of packaging boilerplate while letting the tests run from the
repository root like every other suite.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
