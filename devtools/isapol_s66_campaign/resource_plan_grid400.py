"""Closed 400/2030 integration ledger, not an installed/model qualification.

The scientific candidate changes only response-AUX integration resolution.
Reuse the complete cap32 ledger with every new point charged. This arithmetic
profile has no receipt pins and cannot itself authorize a native calculation.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

_path = Path(__file__).with_name('resource_plan_ed561666.py')
_spec = importlib.util.spec_from_file_location('_grid400_legacy_ledger', _path)
_legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _legacy
exec(compile(_path.read_bytes(), str(_path), 'exec'), _legacy.__dict__)

SCIENTIFIC_COMMIT = '1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2'
POLICY = '1aaa9e0a30-reference-h2h1-grid400-ledger-v4'
FILES = dict(_legacy.FILES)
FILES['psi4/driver/procrouting/isapol_molecular.py'] = (
    '2ffc871d30dab088fae5b4bf1b0a88de1c616dfb948136a69dd04ce93b0492a0')
Dimensions = _legacy.Dimensions
shared_pfit_plan = _legacy.shared_pfit_plan


def source_binding(root):
    observed = {name: hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
                for name in FILES}
    if observed != FILES:
        raise ValueError('scientific planning sources differ from grid400 policy')
    return dict(scientific_commit=SCIENTIFIC_COMMIT, policy_id=POLICY, files_sha256=observed)


def plan(dimensions, resources, *, max_replay_work, pfit_maximum_work_bytes):
    """Predict finite costs only; never promote a resource declaration."""
    return _legacy._shared._ledger(dimensions, resources, max_replay_work=max_replay_work,
        pfit_maximum_work_bytes=pfit_maximum_work_bytes,
        drho_refinement_iterations=32, integration_grid=(400, 2030),
        scientific_commit=SCIENTIFIC_COMMIT, policy=POLICY)
