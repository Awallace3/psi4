"""Production JSON identity seam; pure synthetic values, no scientific work."""
import copy
import importlib.util
from pathlib import Path

import pytest


def verifier():
    path = Path(__file__).with_name('envelope_identity.py')
    spec = importlib.util.spec_from_file_location('_production_envelope_identity_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_native_envelope_identity


def envelopes():
    local = dict(schema='native-monomer-artifacts/2',
        request={'exact': 'unit-only', 'budget': 48},
        implementation={'sha256': {'core': 'a'*64, 'isapol_python': 'b'*64}},
        artifacts={'tensors.npz': {'data': 'SYNTHETIC_UNCHANGED', 'sha256': 'c'*64}},
        result=dict(provenance=dict(partition=dict(diagnostics=dict(
            drho_metric_condition=5.141176000843883e16, other=1.25)))))
    server = copy.deepcopy(local)
    server['result']['provenance']['partition']['diagnostics']['drho_metric_condition'] = 51411760008438830
    return local, server


def test_exact_json_number_representation_is_not_a_calculation_failure():
    local, server = envelopes()
    before = copy.deepcopy((local, server))
    assert local != server
    result = verifier()(local, server)
    assert result['status'] == 'PASS_NATIVE_ENVELOPE_IDENTITY'
    assert result['policy'] == 'exact-json-diagnostic-number/1'
    assert result['epsilon_tolerance_used'] is False
    assert result['diagnostic_json_number_equivalence']['decimal_value'] == '51411760008438830'
    assert (local, server) == before


def test_identical_envelopes_need_no_normalization():
    local, _ = envelopes()
    result = verifier()(local, copy.deepcopy(local))
    assert result['diagnostic_json_number_equivalence'] is None
    assert result['local_json_sha256'] == result['server_json_sha256']


@pytest.mark.parametrize('change', ['diagnostic', 'request', 'artifact', 'implementation',
    'other_diagnostic', 'bool', 'schema', 'missing_path', 'nonfinite', 'other_number_type'])
def test_no_tolerance_or_normalization_of_other_data(change):
    local, server = envelopes()
    diag = server['result']['provenance']['partition']['diagnostics']
    if change == 'diagnostic':
        diag['drho_metric_condition'] += 1
    elif change == 'request':
        server['request']['exact'] = 'changed'
    elif change == 'artifact':
        server['artifacts']['tensors.npz']['data'] += 'changed'
    elif change == 'implementation':
        server['implementation']['sha256']['core'] = 'd'*64
    elif change == 'other_diagnostic':
        diag['other'] += 1e-12
    elif change == 'bool':
        diag['drho_metric_condition'] = True
    elif change == 'schema':
        local['schema'] = server['schema'] = 'foreign'
    elif change == 'missing_path':
        del diag['drho_metric_condition']
    elif change == 'nonfinite':
        diag['drho_metric_condition'] = float('inf')
    else:
        local['request']['budget'], server['request']['budget'] = 48., 48
    with pytest.raises(ValueError):
        verifier()(local, server)


@pytest.mark.parametrize('x,y', [(1e20, 100000000000000000000), (1e18, 1000000000000000000)])
def test_representation_rule_is_not_hardcoded_to_incident_value(x, y):
    local, server = envelopes()
    local['result']['provenance']['partition']['diagnostics']['drho_metric_condition'] = x
    server['result']['provenance']['partition']['diagnostics']['drho_metric_condition'] = y
    assert verifier()(local, server)['status'] == 'PASS_NATIVE_ENVELOPE_IDENTITY'


def caller(filename):
    path = Path(__file__).with_name(filename+'.py')
    spec = importlib.util.spec_from_file_location('_identity_seam_'+filename, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_launcher_to_ordinary_audit_identity_seam(tmp_path):
    import json
    local, server = envelopes()
    launcher, audit = caller('phoenix_pilot_cb63305'), caller('runtime_audit_cb63305')
    witness = launcher.verify_delivery_envelope(local, server, local['request'],
        local['implementation']['sha256'], producer='grid400')
    path = tmp_path/'envelope-identity.json'
    path.write_text(json.dumps(witness))
    result = dict(success=True, extras={'psi4:native_monomer': local})
    assert audit.verify_delivery_identity(result, server, json.loads(path.read_text()),
                                          producer='grid400') == witness
    worker = caller('phoenix_worker_cb63305')
    assert 'envelope_identity.py' in worker.source_names('grid400')
    assert 'envelope_identity.py' not in worker.source_names('ed561666')


@pytest.mark.parametrize('change', ['missing', 'policy', 'local_hash', 'server_hash', 'value'])
def test_ordinary_audit_rejects_missing_or_tampered_identity_witness(change):
    local, server = envelopes()
    launcher, audit = caller('phoenix_pilot_cb63305'), caller('runtime_audit_cb63305')
    witness = launcher.verify_delivery_envelope(local, server, local['request'],
        local['implementation']['sha256'], producer='grid400')
    if change == 'missing':
        witness = None
    elif change == 'value':
        witness['diagnostic_json_number_equivalence']['decimal_value'] = 'changed'
    else:
        witness[{'policy': 'policy', 'local_hash': 'local_json_sha256',
                 'server_hash': 'server_json_sha256'}[change]] = 'changed'
    with pytest.raises(ValueError, match='witness differs'):
        audit.verify_delivery_identity(dict(success=True, extras={'psi4:native_monomer': local}),
                                       server, witness, producer='grid400')


@pytest.mark.parametrize('producer', ['cb63305', 'ed561666'])
def test_legacy_launcher_and_audit_keep_their_strict_comparison(producer):
    local, server = envelopes()
    launcher, audit = caller('phoenix_pilot_cb63305'), caller('runtime_audit_cb63305')
    with pytest.raises(RuntimeError):
        launcher.verify_delivery_envelope(local, server, local['request'],
            local['implementation']['sha256'], producer=producer)
    with pytest.raises(ValueError):
        audit.verify_delivery_identity(dict(success=True, extras={'psi4:native_monomer': local}),
                                       server, None, producer=producer)
