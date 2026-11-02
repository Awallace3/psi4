"""Synthetic forecast declarations, not installed descriptors or molecules."""
import copy
import importlib.util
from pathlib import Path

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proposal = load('propose_descriptor_resources_cb63305', 'propose_descriptor_resources_cb63305.py')
fixtures = load('bank_input_fixtures', 'test_prepare_descriptor_bank_cb63305.py')


def inputs(monkeypatch, *, pentane=False, symbols=None):
    docs, _ = fixtures.fixture()
    for row in docs['inventory']['monomers']:
        row.update(symbols=['O', 'H', 'H'], charge=0, multiplicity=1)
    if pentane:
        symbols = ['C']*5+['H']*12
    if symbols is not None:
        docs['inventory']['monomers'][0]['symbols'] = symbols
        entry = docs['entries']['entries'][0]
        entry['molecule']['symbols'] = symbols
        entry['molecule']['geometry'] = [v for i in range(len(symbols)) for v in (float(i), 0., 0.)]
        geometry = fixtures.prepare.json_digest(entry['molecule'])
        entry['attributes']['geometry_sha256'] = geometry
        declared = docs['declarations']['monomers'][0]
        declared['geometry_sha256'] = geometry
        declared['declaration']['frames'] = [copy.deepcopy(
            declared['declaration']['frames'][0]) for _ in symbols]
        declared['declaration']['bonds'] = [[i, i+1] for i in range(len(symbols)-1)]
        declared['policy']['independent_parameters'] = 36*len(symbols)
    raw = fixtures.bind_inputs(monkeypatch, docs)
    original = proposal.helper
    monkeypatch.setattr(proposal, 'helper', lambda name: fixtures.prepare
        if name == 'prepare_descriptor_bank_cb63305' else original(name))
    return {k: raw[k] for k in ('inventory', 'entries', 'declarations')}


def test_small_cases_get_closed_small_declarations_not_maximum_for_everything(monkeypatch):
    raw = inputs(monkeypatch)
    manifest, summary = proposal.propose(raw)
    assert manifest['status'] == 'DECLARED_NOT_ADMITTED'
    assert summary['status'] == 'FORECAST_DECLARATIONS_ONLY'
    assert len(manifest['monomers']) == len(summary['monomers']) == 131
    for row in manifest['monomers'].values():
        assert row['resources'] == dict(max_bytes=12*1024**3, max_work=6*10**12,
                                       max_io_bytes=16*1024**3, capacity='large')
        assert row['pfit_maximum_work_bytes'] == 256*1024**2
        assert row['intended_memory_bytes'] == 20*1024**3
    assert 'not actual installed shell descriptors' in summary['qualification']


def test_large_case_is_explicit_extended_full_model_and_budget_not_work_discount(monkeypatch):
    manifest, summary = proposal.propose(inputs(monkeypatch, pentane=True))
    contract = manifest['monomers']['monomer-0001']
    predicted = summary['monomers'][0]
    assert contract['resources']['capacity'] == 'extended'
    assert contract['resources']['max_work'] == 128*10**12
    assert contract['resources']['max_bytes'] == 64*1024**3
    assert contract['pfit_maximum_work_bytes'] == 2*1024**3
    assert predicted['charged_work'] == 105_124_176_150_908
    assert predicted['dimensions']['parameters'] == 612
    assert predicted['dimensions']['fit_points'] == 2000
    assert predicted['dimensions']['nodes'] == 11
    assert predicted['memory_class'] == dict(numeric_bytes=64*1024**3,
        psi4_bytes=80*1024**3, job_bytes=96*1024**3, cpus=16)


def test_no_larger_class_or_partial_result_is_inferred_after_refusal(monkeypatch):
    raw = inputs(monkeypatch)
    monkeypatch.setattr(proposal, 'WORK_CLASSES', (1,))
    with pytest.raises(ValueError, match='closed finite resource classes'):
        proposal.propose(raw)


def test_raw_input_mutation_is_not_a_new_approved_forecast_input(monkeypatch):
    raw = inputs(monkeypatch)
    raw['inventory'] += b' '
    with pytest.raises(ValueError, match='input hash differs'):
        proposal.propose(raw)


@pytest.mark.parametrize('numeric,work,expected_memory,expected_work,capacity', [
    (12*1024**3, 6*10**12, 12, 6, 'large'),
    (12*1024**3+1, 6*10**12+1, 32, 12, 'large'),
    (12*1024**3, 12*10**12, 12, 12, 'large'),
    (12*1024**3, 12*10**12+1, 12, 15, 'large'),
    (32*1024**3, 15*10**12, 32, 15, 'large'),
    (12*1024**3, 15*10**12+1, 12, 25, 'large'),
    (32*1024**3+1, 25*10**12, 64, 25, 'large'),
    (12*1024**3, 25*10**12+1, 12, 50, 'large'),
    (12*1024**3, 50*10**12, 12, 50, 'large'),
    (12*1024**3, 50*10**12+1, 12, 75, 'large'),
    (12*1024**3, 75*10**12, 12, 75, 'large'),
    (12*1024**3, 75*10**12+1, 12, 100, 'large'),
    (64*1024**3, 100*10**12, 64, 100, 'large'),
    (64*1024**3, 100*10**12+1, 64, 128, 'extended'),
    (64*1024**3, 128*10**12, 64, 128, 'extended'),
])
def test_closed_class_exact_and_just_over_boundaries(
        monkeypatch, numeric, work, expected_memory, expected_work, capacity):
    raw = inputs(monkeypatch)
    original = proposal.helper
    def helper(name):
        module = original(name)
        if name == 'resource_plan_cb63305':
            module.plan = lambda d, r, **kwargs: dict(
                status='WITHIN_PREDICTED_LEDGER_LIMITS', charged_work=work,
                admission_numeric_bytes=numeric, checkpoint_io_bytes=1,
                dimensions=dict(parameters=36*d.sites, fit_points=2000, nodes=11),
                native_pfit=dict(base_bytes=1))
        return module
    monkeypatch.setattr(proposal, 'helper', helper)
    manifest, _ = proposal.propose(raw)
    first = manifest['monomers']['monomer-0001']['resources']
    assert first['max_bytes'] == expected_memory*1024**3
    assert first['max_work'] == expected_work*10**12
    assert first['capacity'] == capacity


@pytest.mark.parametrize('phase,index', [('envelope', 4), ('selected', 5)])
def test_later_case_refusal_stops_the_whole_proposal_without_dropping_or_retry(monkeypatch, phase, index):
    raw = inputs(monkeypatch)
    original = proposal.helper
    calls = []
    def helper(name):
        module = original(name)
        if name == 'resource_plan_cb63305':
            real = module.plan
            def plan(*args, **kwargs):
                current = len(calls)
                calls.append((args[1], kwargs))
                if current == index:
                    return dict(status='REFUSED')
                return real(*args, **kwargs)
            module.plan = plan
        return module
    monkeypatch.setattr(proposal, 'helper', helper)
    match = 'explicit new review' if phase == 'envelope' else 'no automatic promotion'
    with pytest.raises(ValueError, match=match):
        proposal.propose(raw)
    assert len(calls) == index+1


@pytest.mark.parametrize('symbols,cap', [
    (['C', 'N']+['H']*5, 256*1024**2),
    (['C', 'C']+['H']*6, 2*1024**3),
])
def test_native_cap_declaration_boundary_is_explicit_before_execution(monkeypatch, symbols, cap):
    manifest, summary = proposal.propose(inputs(monkeypatch, symbols=symbols))
    assert manifest['monomers']['monomer-0001']['pfit_maximum_work_bytes'] == cap
    assert summary['monomers'][0]['dimensions']['parameters'] == 36*len(symbols)


@pytest.mark.parametrize('numeric,work', [
    (64*1024**3+1, 128*10**12),
    (64*1024**3, 128*10**12+1),
])
def test_one_over_terminal_envelope_is_refused_without_larger_retry(monkeypatch, numeric, work):
    raw = inputs(monkeypatch)
    original = proposal.helper
    calls = []
    def helper(name):
        module = original(name)
        if name == 'resource_plan_cb63305':
            def plan(dimensions, resources, **kwargs):
                calls.append(copy.deepcopy(resources))
                assert numeric > resources['max_bytes'] or work > resources['max_work']
                return dict(status='REFUSED', refusals=['terminal_envelope'])
            module.plan = plan
        return module
    monkeypatch.setattr(proposal, 'helper', helper)
    with pytest.raises(ValueError, match='forecast exceeds closed proposal envelope'):
        proposal.propose(raw)
    assert calls == [dict(max_bytes=64*1024**3, max_work=128*10**12,
                          max_io_bytes=16*1024**3, capacity='extended')]
