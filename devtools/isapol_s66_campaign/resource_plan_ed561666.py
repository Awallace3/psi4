"""Closed cap32 arithmetic profile; no installed or molecular qualification.

Use the same whole-chain arithmetic as the historical cb63305 calculator,
but charge this producer's supported Drho-C budget. No basis, grid, fit or
response dimension is reduced. Receipt pins are a separate acceptance gate.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

_path = Path(__file__).with_name('resource_plan_cb63305.py')
_spec = importlib.util.spec_from_file_location('_drho32_shared_ledger', _path)
_shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _shared
exec(compile(_path.read_bytes(), str(_path), 'exec'), _shared.__dict__)

SCIENTIFIC_COMMIT = 'ed561666fcac212fffa774429fc3bd8f8cfc903d'
POLICY = 'ed561666-reference-h2h1-ledger-v3'
DRHO_REFINEMENT_ITERATIONS = 32
FILES = dict(_shared.FILES)
FILES['psi4/driver/procrouting/isapol_native_partition.py'] = (
    '2b539fa8d539ef571ed3265fdc11168eb5bb6212efff42cac24d952244b6a643')
Dimensions = _shared.Dimensions
shared_pfit_plan = _shared.shared_pfit_plan


def source_binding(root):
    observed = {name: hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() for name in FILES}
    if observed != FILES:
        raise ValueError('scientific planning sources differ from ed561666 policy')
    return dict(scientific_commit=SCIENTIFIC_COMMIT, policy_id=POLICY, files_sha256=observed)


def plan(dimensions, resources, *, max_replay_work, pfit_maximum_work_bytes):
    """Pure prediction, never a passing descriptor/installation/model receipt."""
    return _shared._ledger(dimensions, resources, max_replay_work=max_replay_work,
        pfit_maximum_work_bytes=pfit_maximum_work_bytes,
        drho_refinement_iterations=DRHO_REFINEMENT_ITERATIONS,
        scientific_commit=SCIENTIFIC_COMMIT, policy=POLICY)
