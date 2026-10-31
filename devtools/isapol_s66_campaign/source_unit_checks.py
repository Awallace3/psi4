"""Explicit source-under-test overlay on an installed matching native core.

This is a unit-test harness, not installed execution or molecular provenance.
No campaign/acceptance result may be produced through it.
"""
import importlib
import importlib.util
from pathlib import Path
import sys

import psi4
import pytest

def main():
    root = Path(__file__).resolve().parents[2]
    for name in ('isapol_pfit_stream', 'isapol_bounded_response', 'isapol_bounded',
                 'isapol_bounded_oeprop', 'isapol_recipes', 'isapol_molecular',
                 'isapol_schema'):
        qualified = 'psi4.driver.procrouting.'+name
        spec = importlib.util.spec_from_file_location(
            qualified, root/'psi4/driver/procrouting'/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        setattr(importlib.import_module('psi4.driver.procrouting'), name, module)

    qualified = 'psi4.driver.schema_wrapper'
    spec = importlib.util.spec_from_file_location(qualified, root/'psi4/driver/schema_wrapper.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    setattr(importlib.import_module('psi4.driver'), 'schema_wrapper', module)
    psi4.schema_wrapper = module

    raise SystemExit(pytest.main([
        '--noconftest', '-c', '/dev/null', '-p', 'no:cacheprovider', '-q', '--tb=short',
        *sys.argv[1:],
    ]))


if __name__ == '__main__':
    main()
