"""Closed fb11 source adapter. Arithmetic stays explicitly UNQUALIFIED.

Installed producer qualification is separate receipt evidence. No old policy,
FILES, source guard, ledger result or cap is reinterpreted.
"""
import hashlib
import importlib.util
from pathlib import Path

SCIENTIFIC_COMMIT = 'fb11b78d1f44658d3a9d62365b77f3d217f53129'
_path = Path(__file__).with_name('resource_plan_isa_grid_v1.py')
# The enclosing reviewed source inventory also binds this pure seam. Verify
# its exact Git-derived bytes, never import scientific Python/native Psi4.
_EXPECTED = '552dcf6eef5c4f8a6f731668f324bfe9ace5bfb3832c2f9e7f88cd939e41a7d1'
_raw = _path.read_bytes()
if hashlib.sha256(_raw).hexdigest() != _EXPECTED:
    raise ValueError('fb11 pure ISA accounting seam differs')
_spec = importlib.util.spec_from_file_location('_qomp_isa_accounting', _path)
_inner = importlib.util.module_from_spec(_spec)
exec(compile(_raw, str(_path), 'exec'), _inner.__dict__)
Dimensions, FILES, POLICY = _inner.Dimensions, dict(_inner.FILES), _inner.POLICY
plan = _inner.plan
source_binding = _inner.source_binding
shared_pfit_plan = _inner._legacy.shared_pfit_plan
