"""Synthetic streaming quadrature and frozen algebra; no quantum calculations."""
import importlib.util
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('grid_charge_diag',
    Path(__file__).with_name('grid_charge_diagnostic_ed561666.py'))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_streaming_visits_every_point_and_keeps_signed_weights():
    points = np.arange(33, dtype=float).reshape(11, 3)
    weights = np.array([1., -2., 3., -4., 5., 6., 7., -8., 9., 10., -11.])
    blocks = []
    def evaluate(p):
        blocks.append(p.copy())
        return p[:, :2]
    x, extended, error = diag.integrate_grid(points, weights, evaluate, 2, block_points=4)
    np.testing.assert_array_equal(np.concatenate(blocks), points)
    np.testing.assert_allclose(x, weights @ points[:, :2])
    np.testing.assert_allclose(extended, weights @ points[:, :2])
    assert error == 0.


@pytest.mark.parametrize('bad', ['missing_column', 'missing_row', 'nonfinite', 'float32'])
def test_partial_or_invalid_collocation_never_counts_as_complete_grid(bad):
    points, weights = np.ones((5, 3)), np.ones(5)
    def evaluate(p):
        value = np.ones((len(p), 2))
        if bad == 'missing_column': value = value[:, :1]
        if bad == 'missing_row': value = value[:-1]
        if bad == 'nonfinite': value[0, 0] = np.nan
        if bad == 'float32': value = value.astype(np.float32)
        return value
    with pytest.raises(ValueError, match='complete'):
        diag.integrate_grid(points, weights, evaluate, 2)


def test_projected_grid_error_is_signed_and_explicitly_not_a_new_response():
    rng = np.random.default_rng(10)
    c = rng.normal(size=(4, 3))
    x = rng.normal(size=(4, 50))
    analytic = np.array([1., 2., 3.])
    grid = analytic + np.array([1e-6, -2e-6, 3e-6])
    actual = -x.T @ (c @ grid)
    result = diag.frozen_projection(grid, analytic, c, x, 2)
    assert result['q_charge_row_error'] == pytest.approx(3e-6)
    assert result['frozen_transpose_charge_maximum'] == pytest.approx(
        np.max(np.abs(actual.reshape(2, 25)[:, :9])))
    point = result['at_maximum']
    assert point['fit']+point['integration'] == pytest.approx(point['signed_sum'])
    assert 'NOT a new physical response' in result['qualification']


def test_admission_refuses_oversized_grids_before_any_collocation(monkeypatch):
    monkeypatch.setattr(diag, 'MAX_POINTS', 1)
    with pytest.raises(ValueError, match='bounded'):
        diag.integrate_grid(np.ones((2, 3)), np.ones(2),
                            lambda _: pytest.fail('must not evaluate'), 1)


def test_native_point_formula_and_budget_are_declared_before_allocation():
    expected = [2325912, 5216952, 9719640]
    assert [diag.grid_plan(12, r, a)['points'] for r, a in diag.GRIDS] == expected
    assert max(diag.grid_plan(12, r, a)['planned_numeric_bytes']
               for r, a in diag.GRIDS) < 4*1024**3
    for args in [(11, 200, 974), (12, 400, 9999), (True, 200, 974)]:
        with pytest.raises(ValueError, match='closed'):
            diag.grid_plan(*args)


@pytest.mark.parametrize('bad_grid', [False, True])
def test_native_grid_adapter_visits_full_grid_and_owns_matrix_views(monkeypatch, bad_grid):
    calls = []
    monkeypatch.setattr(diag, 'grid_plan', lambda *args: dict(points=5))
    class Grid:
        def __init__(self, molecule, options):
            assert options.becke_smoothing == 3 and options.radius_scaling == 1.
            self.options = options
        def npoints(self): return 5
        def natom(self): return 12
        def radial_points(self): return self.options.radial_points
        def spherical_points(self): return 999 if bad_grid else self.options.spherical_points
        def atom_start(self, n): assert n == 12; return 5
        def x(self): return np.arange(5, dtype=float)
        def y(self): return np.zeros(5)
        def z(self): return np.zeros(5)
        def w(self): return np.ones(5)/5
    def evaluate(points):
        calls.extend(points)
        return NS(np=np.ones((len(points), 1065)))
    core = NS(IsaGridOptions=NS, IsaGrid=Grid)
    mol = NS(natom=lambda: 12, clone=lambda: 'SYNTHETIC_MOLECULE')
    aux = NS(nfunction=1065, evaluate=evaluate)
    if bad_grid:
        with pytest.raises(ValueError, match='actual complete'):
            diag.sample_native_grid(core, mol, aux, 200, 974)
        assert calls == []
    else:
        vectors, record = diag.sample_native_grid(core, mol, aux, 200, 974)
        assert len(calls) == 5 and record['plan']['points'] == 5
        np.testing.assert_allclose(vectors['direct_extended'], 1.)
        np.testing.assert_allclose(vectors['absolute_sum'], 1.)
        assert record['reduction']['longdouble_mantissa_bits'] >= 52


def test_vector_checkpoint_is_exclusive_and_roundtrip_checked(tmp_path):
    arrays = {k: np.ones(1065) for k in (
        'direct_binary64', 'direct_extended', 'absolute_sum')}
    path = tmp_path/'grid-vectors.npz'
    result = diag.save_vectors(path, arrays)
    assert result == diag.operations().file_sha(path)
    with pytest.raises(FileExistsError):
        diag.save_vectors(path, arrays)
    with pytest.raises(ValueError, match='roles'):
        diag.save_vectors(tmp_path/'bad.npz', {'direct_extended': np.ones(1065)})


def test_grid_study_help_is_lightweight_and_no_scientific_production_is_called():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', diag.__file__, '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0 and '--worker' in result.stdout
    tree = ast.parse(Path(diag.__file__).read_text())
    calls = {ast.unparse(node.func).split('.')[-1] for node in ast.walk(tree)
             if isinstance(node, ast.Call)}
    assert not calls & {'run_qcschema', 'properties', 'energy', 'oeprop', 'isa_moments',
        'native_partition', 'run_molecular_properties', 'isa_localize_lw',
        'PortalClient', 'ComputeManager', 'reset_records', 'submit', 'compute'}
    batch = Path(diag.__file__).with_suffix('.sbatch').read_text()
    assert '#SBATCH --qos=embers' in batch and '#SBATCH --mem=16G' in batch
    assert '#SBATCH --time=04:00:00' in batch and '#SBATCH --no-requeue' in batch
    assert batch.index('ccasp_wait_child "$child" 20') < batch.index('GRID_STUDY_COMPLETE')


def test_frozen_snapshot_cannot_follow_file_mutation_or_reenable_writes(tmp_path):
    path = tmp_path/'q.npy'
    original = np.array([1., np.nextafter(.5, 1.)])
    with path.open('xb') as stream:
        np.save(stream, original, allow_pickle=False)
    record = dict(file='q.npy', numeric_bytes=original.nbytes, shape=[2],
                  dtype='<f8', file_sha256=diag.operations().file_sha(path))
    snapshot = diag.owned_frozen_array(path, record)
    with path.open('wb') as stream:
        np.save(stream, np.zeros(2), allow_pickle=False)
    np.testing.assert_array_equal(snapshot, original)
    assert not snapshot.flags.writeable
    with pytest.raises(ValueError):
        snapshot.flags.writeable = True
    with pytest.raises(ValueError, match='hash changed'):
        diag.owned_frozen_array(path, record)


@pytest.mark.parametrize('mutate_operand', [False, True])
def test_parent_rechecks_all_frozen_operands_after_child_before_receipt(
        tmp_path, monkeypatch, mutate_operand):
    real = diag.operations()
    runs = tmp_path/'runs'
    runs.mkdir()
    run = runs/'ccasp-native-s66-ed561666fc-grid-study001'
    (run/'logs').mkdir(parents=True)
    frozen = runs/'ccasp-native-s66-ed561666fc-charge-diagnostic001/payload'
    frozen.mkdir(parents=True)
    records = {}
    for role in ('q', 'analytic_charge', 'anchor_coefficients', 'solved_anchor',
                 'anchor_legs', 'nonlocal_response'):
        path = frozen/(role+'.npy')
        with path.open('xb') as stream:
            np.save(stream, np.ones(2), allow_pickle=False)
        records[role] = dict(file=path.name, shape=[2], numeric_bytes=16,
                             dtype='<f8', file_sha256=real.file_sha(path))
    runtime_sha = real.sha(b'{}')
    original = dict(captured_arrays=records, captured_numeric_bytes=96,
        operational_commit=diag.FROZEN_OPERATIONAL_COMMIT, runtime_input_sha256=runtime_sha,
        production_tolerance=1e-6, discriminator='FIRST_NODE_PRODUCTION_LW_REFUSAL_CAPTURED')
    raw = json.dumps(original).encode()
    (frozen/'charge-diagnostic.json').write_bytes(raw)
    report_sha = real.sha(raw)
    receipt_raw = json.dumps(dict(report_sha256=report_sha)).encode()
    (frozen/'diagnostic-receipt.json').write_bytes(receipt_raw)
    monkeypatch.setattr(diag, 'FROZEN_REPORT_SHA256', report_sha)
    monkeypatch.setattr(diag, 'FROZEN_RECEIPT_SHA256', real.sha(receipt_raw))
    build = runs/'build'
    build.mkdir()
    manifest = build/'build-identity.json'
    manifest.write_text('{"synthetic":true}')
    case = tmp_path/'case'
    case.write_bytes(b'SYNTHETIC_CASE')
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    monkeypatch.setenv('SLURM_JOB_ID', 'SYNTHETIC')
    checks = NS(allocation=lambda: dict(Account='gts-cs207-chemx', QOS='embers',
        Partition='cpu-small', NumCPUs='12', NumNodes='1', MinMemoryNode='16G',
        TimeLimit='04:00:00'), verify_runtime=lambda *args: None,
        installed_python_command=lambda *args: ['SYNTHETIC_CHILD'])
    helpers = {'phoenix_installed_checks': checks, 'descriptor_scratch_cb63305': None,
        'phoenix_pilot_cb63305': NS(scratch_snapshot=lambda *args, **kw: {}),
        'phoenix_descriptor_bank_cb63305': NS(child_environment=lambda *args: {})}
    common = NS(MANAGER=Path(sys.executable).resolve().parents[1],
        RUNS=runs, BUILD=build, BUILD_SHA256=real.file_sha(manifest), CASE=case,
        CASE_SHA256=real.file_sha(case), SCRATCH=scratch, RUNTIME_SHA256=runtime_sha,
        SCIENTIFIC_COMMIT=real.SCIENTIFIC_COMMIT, sha=real.sha, file_sha=real.file_sha,
        write_json=real.write_json, source_identities=lambda *args: {'synthetic': 'same'},
        qualified_receipts=lambda: {'synthetic': 'qualified'},
        helper=lambda name: helpers[name], bound_runtime=lambda raw: b'{}')
    monkeypatch.setattr(diag, 'operations', lambda: common)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        common.write_json(run/'payload/grid-study.json', dict(
            status='COMPLETE_GRID_STUDY_NOT_A_MODEL', operational_commit='SYNTHETIC_OPS',
            runtime_sha256=runtime_sha, production_tolerance_unchanged=1e-6,
            frozen_operands_owned_immutable=True,
            frozen_operand_sha256={k: r['file_sha256'] for k, r in records.items()},
            grids=[dict(status='COMPLETE_GRID_QUADRATURE_NOT_A_RESPONSE',
                        radial_points=r, spherical_points=a) for r, a in diag.GRIDS],
            baseline_charge_sums_numerically_agree=False))
        if mutate_operand:
            with (frozen/'q.npy').open('wb') as stream:
                np.save(stream, np.zeros(2), allow_pickle=False)
    monkeypatch.setattr(diag.subprocess, 'run', child)
    if mutate_operand:
        with pytest.raises(ValueError, match='frozen operand changed'):
            diag.parent(NS(run=run, commit='SYNTHETIC_OPS'))
        assert not (run/'payload/study-receipt.json').exists()
    else:
        diag.parent(NS(run=run, commit='SYNTHETIC_OPS'))
        receipt = json.loads((run/'payload/study-receipt.json').read_bytes())
        assert receipt['frozen_operand_sha256'] == {k: r['file_sha256'] for k, r in records.items()}
    assert calls == [['SYNTHETIC_CHILD']]  # one child; never a retry
