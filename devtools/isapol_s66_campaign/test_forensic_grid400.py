"""Pure incident/evidence gates; no QCA, native imports or scientific work."""
import copy
import importlib.util
from pathlib import Path
import subprocess

import pytest

path = Path(__file__).with_name('forensic_grid400.py')
spec = importlib.util.spec_from_file_location('_forensic_grid400_test', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def envelopes():
    local = dict(request={'exact': 'synthetic'}, artifacts={'payload': 'exact synthetic bytes'},
        result=dict(provenance=dict(partition=dict(diagnostics=dict(
            drho_metric_condition=5.141176000843883e16, other=1.25)))))
    server = copy.deepcopy(local)
    server['result']['provenance']['partition']['diagnostics']['drho_metric_condition'] = 51411760008438830
    return local, server


def test_measured_python_comparison_failure_has_exact_decimal_json_equivalence():
    local, server = envelopes()
    before = copy.deepcopy((local, server))
    assert local != server
    report = audit.compare_incident_envelopes(local, server)
    assert report['status'] == 'EXACT_JSON_DECIMAL_EQUIVALENCE_AT_MEASURED_DIAGNOSTIC_ONLY'
    assert report['epsilon_tolerance_used'] is False and report['every_other_field_exact'] is True
    assert (local, server) == before


@pytest.mark.parametrize('change', ['diagnostic', 'request', 'artifact', 'other_metadata',
                                   'other_numeric_type', 'missing_path', 'bool', 'nonfinite',
                                   'reversed_types', 'already_equal'])
def test_no_other_difference_or_general_float_tolerance_is_permitted(change):
    local, server = envelopes()
    diagnostics = server['result']['provenance']['partition']['diagnostics']
    if change == 'diagnostic':
        diagnostics['drho_metric_condition'] += 1
    elif change == 'request':
        server['request']['exact'] = 'changed'
    elif change == 'artifact':
        server['artifacts']['payload'] = 'changed'
    elif change == 'other_metadata':
        diagnostics['other'] += 1e-12
    elif change == 'other_numeric_type':
        local['request']['number'], server['request']['number'] = 1., 1
    elif change == 'missing_path':
        del diagnostics['drho_metric_condition']
    elif change == 'bool':
        diagnostics['drho_metric_condition'] = True
    elif change == 'nonfinite':
        local['result']['provenance']['partition']['diagnostics']['drho_metric_condition'] = float('inf')
    elif change == 'reversed_types':
        local, server = server, local
    else:
        server = copy.deepcopy(local)
    with pytest.raises(ValueError):
        audit.compare_incident_envelopes(local, server)


def chain():
    started = dict(status='STARTED_NOT_QUALIFIED', producer_profile='grid400', dataset_mode='fresh',
        monomer_id='monomer-0008', operational_commit=audit.CASE_COMMIT,
        allocation={'JobId': '15090188'}, find_existing=False, maximum_records=1,
        qcengine_internal_retries=0, compute_tag='synthetic-fixed-tag')
    created = dict(dataset_id=356, record_id=5234327, n_inserted=1, n_existing=0,
                   compute_tag=started['compute_tag'])
    claim = dict(record_id=5234327, task_id=4722332, manager='synthetic-owned-manager')
    delivered = claim | dict(outcome='accepted', interrupted=False, claim_error_type=None,
        submission_error_type=None, local_statistics=dict(total_successful_tasks=1,
            total_failed_tasks=0, total_rejected_tasks=0, active_tasks=0))
    server = dict(record_id=5234327, status='complete', history_count=1,
                  history_managers=[claim['manager']])
    return started, created, claim, delivered, server, copy.deepcopy(server)


def test_start_evidence_is_not_relabelled_as_a_successful_outcome():
    audit.validate_chain(*chain())
    values = chain()
    values[0]['status'] = 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
    with pytest.raises(ValueError):
        audit.validate_chain(*values)


@pytest.mark.parametrize('index,field,value', [
    (0, 'operational_commit', 'changed'), (0, 'dataset_mode', 'append'),
    (1, 'dataset_id', 355), (1, 'record_id', 5234326), (1, 'n_existing', 1),
    (2, 'task_id', 1), (3, 'outcome', 'rejected'), (3, 'interrupted', True),
    (4, 'status', 'error'), (5, 'history_count', 2), (5, 'history_managers', ['foreign'])])
def test_misbound_delivery_or_live_server_state_fails(index, field, value):
    values = chain()
    values[index][field] = value
    with pytest.raises(ValueError):
        audit.validate_chain(*values)


def test_original_numeric_fd_and_ordinary_audit_policies_are_unchanged():
    root = Path(__file__).resolve().parents[2]
    for relative in ('devtools/isapol_native_acceptance/trace_gate.py',
        'devtools/isapol_s66_campaign/validate_numeric.py',
        'devtools/isapol_s66_campaign/runtime_audit.py',
        'devtools/isapol_s66_campaign/runtime_audit_cb63305.py',
        'devtools/isapol_s66_campaign/phoenix_pilot_cb63305.py'):
        original = subprocess.check_output(['git', '-C', str(root), 'show',
                                            audit.CASE_COMMIT+':'+relative])
        assert (root/relative).read_bytes() == original


def test_forensic_report_cannot_be_a_model_acceptance_or_old_success_receipt():
    import ast
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    reports = [node for node in calls if any(
        keyword.arg == 'schema' and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == 'native-grid400-forensic-components/1'
        for keyword in node.keywords)]
    assert len(reports) == 1
    values = {keyword.arg: keyword.value for keyword in reports[0].keywords}
    for name in ('model_accepted', 'original_protocol_passed',
                 'original_completion_receipt_present', 'reconstructed_completion_receipt'):
        assert isinstance(values[name], ast.Constant) and values[name].value is False
    source = path.read_text()
    assert "with (output/'forensic-report.json').open('x')" in source
    assert "with (data/'pilot-outcome.json')" not in source


def intents():
    started = chain()[0] | dict(dataset_name='synthetic-dataset', specification_name='synthetic-spec')
    draft = dict(dataset_name='synthetic-dataset', specification_name='synthetic-spec',
                 entry={'name': 'synthetic-entry'})
    created = dict(dataset_id=356, name='synthetic-dataset')
    submitted = dict(dataset_id=356, entry='synthetic-entry', specification='synthetic-spec',
                     compute_tag=started['compute_tag'], find_existing=False, transport_retries=False)
    return started, draft, created, submitted


def test_original_creation_and_submission_intents_are_required():
    audit.validate_intents(*intents())


@pytest.mark.parametrize('index,field,value', [
    (0, 'dataset_name', 'foreign'), (0, 'specification_name', 'foreign'),
    (2, 'dataset_id', 355), (2, 'name', 'foreign'),
    (3, 'entry', 'foreign'), (3, 'specification', 'foreign'),
    (3, 'compute_tag', 'foreign'), (3, 'find_existing', True),
    (3, 'transport_retries', True), (3, 'dataset_id', 355)])
def test_contradictory_original_intents_cannot_pass(index, field, value):
    values = intents()
    values[index][field] = value
    with pytest.raises(ValueError):
        audit.validate_intents(*values)


def test_entire_draft_is_reconstructed_before_and_after_component_analysis():
    import ast
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'prepare_draft']
    assert len(calls) == 2
