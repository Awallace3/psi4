"""Offline bridge/configuration/comparison guards; no private config or HTTP."""
import importlib.util
from pathlib import Path
import sys

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('uri', [
    'http://0.0.0.0:17777', 'http://localhost:17777',
    'http://127.0.0.1:17777/path', 'http://user@127.0.0.1:17777',
    'http://127.0.0.1:17777?override=yes', 'http://127.0.0.1:17777#x',
    'https://127.0.0.1:17777', 'http://127.0.0.1',
])
def test_only_explicit_loopback_http_is_admitted(uri, tmp_path, monkeypatch):
    bridge = load('qca071_bridge')
    monkeypatch.setattr(bridge, 'CONFIG', tmp_path/'must-not-be-read')
    with pytest.raises(ValueError, match='loopback'):
        bridge.server_config(uri)


def test_existing_config_is_not_changed_or_serialized(tmp_path, monkeypatch):
    bridge = load('qca071_bridge')
    path = tmp_path/'config.yml'
    original = ('server:\n  fractal_uri: http://128.61.254.96:7777\n'
                '  username: awallace43\n  password: synthetic-fixture\n  verify: false\n')
    path.write_text(original)
    monkeypatch.setattr(bridge, 'CONFIG', path)
    server = bridge.server_config('http://127.0.0.1:17777')
    assert server['fractal_uri'] == 'http://127.0.0.1:17777'
    assert path.read_text() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ['config.yml']


@pytest.mark.parametrize('name,version', [
    ('CDS QCFractal Server', '0.70'), ('other server', '0.71'),
    ('CDS QCFractal Server', '0.72'),
])
def test_server_identity_is_exact(name, version):
    bridge = load('qca071_bridge')
    with pytest.raises(ValueError, match='0.71'):
        bridge.validate_information(dict(name=name, version=version))


def test_server_declares_supported_existing_client_and_manager():
    bridge = load('qca071_bridge')
    info = dict(name='CDS QCFractal Server', version='0.71',
                client_version_lower_limit='0.50', client_version_upper_limit='1.00',
                manager_version_lower_limit='0.50', manager_version_upper_limit='1.00')
    report = bridge.validate_information(info)
    assert report['version'] == '0.71'
    assert 'password' not in report
    info['manager_version_lower_limit'] = '999'
    with pytest.raises(ValueError, match='compatibility'):
        bridge.validate_information(info)


def test_dimer_comparison_rejects_actual_coefficient_changes(monkeypatch):
    import numpy as np
    check = load('qca071_water_check')
    class Numeric:
        @staticmethod
        def tensor_arrays(raw, ns):
            assert ns == 3
            tensor = np.zeros((11, 3, 9, 9))
            for i in range(9):
                tensor[:, :, i, i] = 1. if raw == b'a' else 2.
            return dict(refined=tensor)
    monkeypatch.setattr(check, 'helper', lambda name: Numeric)
    import base64
    def envelope(raw):
        return dict(artifacts={'tensors.npz':dict(data=base64.b64encode(raw).decode())},
                    result=dict(cp_weights=[0.]+[.1]*10))
    rows = [dict(runtime_input=dict(molecule=dict(geometry=geometry)))
            for geometry in ([0.,0.,0., 0.,1.,0., 0.,0.,1.],
                             [5.,0.,0., 5.,1.,0., 5.,0.,1.])]
    pairs = [dict(direct=envelope(b'a'), qca=envelope(b'a')) for _ in range(2)]
    report = check.compare_dimer(rows, pairs)
    assert report['status'] == 'PASS_WATER_DIMER_ISOTROPIC_COMPARISON'
    assert len(report['direct']['cross_fragment_coefficients']) == 27
    pairs[1]['qca'] = envelope(b'b')
    with pytest.raises(ValueError, match='cross-fragment'):
        check.compare_dimer(rows, pairs)


def test_manager_is_constructed_inside_one_shot_client_factory(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace as NS
    import qcfractalcompute.compute_manager as compute
    check = load('qca071_water_check')
    claim = load('single_claim')
    original = compute.ManagerClient
    events = []
    class Client:
        pass
    class Base:
        def __init__(self, config):
            assert compute.ManagerClient is Client
            events.append('construct')
            self.client = Client()
            self.manager_config = config
            self.name = 'synthetic-manager'
            self.dflow_kernel = None
        def start(self):
            self.local_native = {'synthetic': True}
            self.delivery_outcome = 'accepted'
            self.target_task_id = 7
    @contextmanager
    def factory():
        events.append('enter')
        monkeypatch.setattr(compute, 'ManagerClient', Client)
        try:
            yield Client
        finally:
            monkeypatch.setattr(compute, 'ManagerClient', original)
            events.append('exit')
    monkeypatch.setattr(compute, 'ComputeManager', Base)
    def helper(name):
        if name == 'single_claim':
            return claim
        if name == 'dataset_fresh_cb63305':
            return NS(grid400_manager_client_factory=factory)
        raise AssertionError(name)
    monkeypatch.setattr(check, 'helper', helper)
    config = NS(executors={'pilot': NS(compute_tags=['test-tag'], max_workers=1)})
    name, native = check.manager_session(config, 1, 'test-tag', {}, tmp_path)
    assert name == 'synthetic-manager' and native == {'synthetic': True}
    assert events == ['enter', 'construct', 'exit']
    assert compute.ManagerClient is original
