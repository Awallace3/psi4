"""Experimental schema8 state-machine tests, not production capture evidence."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('cache_replay_test', Path(__file__).parent /
                                               'data_isapol/oracle/replay_response.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
FIT = ((1, 0, 1, 1), (0., 0., 0., 0.), (0, 0))
POLICY = (1, 1e-12, 1e-18)


class Scenario:
    def __init__(self, slots=(7, 11)):
        self.events = []
        self.slots = slots
        self.rid = self.bid = self.tid = self.sid = 0
        self.j = np.array([[2., .3], [-.1, 1.]])
        self.data = dict(OO=np.array([[.2, -.1], [.15, .2], [-.1, .4]]),
                         VV=np.array([[.3, .4], [-.2, .1], [.2, -.4]]),
                         OV=np.array([[.1, .2], [-.2, .3], [.4, -.1], [.2, .1]]))
        self.fit = dict.fromkeys(self.data, FIT)
        self.built = {}
        self.add('CACHE_CHECK', slots=slots, policy=POLICY, initialized=False, same_geometry=False)

    def add(self, kind, **kw):
        e = dict(version=8, header=(len(self.events)+1, 7, 4, 2, 2, 2, 4),
                 kind=kind, name='Water', scf='dalton', **copy.deepcopy(kw))
        self.events.append(e)
        return e

    def metric_metadata(self):
        side = 'A' if self.slots[0] == 7 else 'B'
        return dict(metric_identity=('J', 'j'), metric_type='S___', metric_description=f'_{side}_{side}',
                    metric_shape=(2, 2), metric_flags=(1, 0, 0, 0), policy=POLICY)

    def metric(self, route='DF_S', cached=False):
        self.rid += 1
        owner = ('J', 'j') if route == 'DF_S' else ('Sc', 'sc')
        self.add('METRIC_REQUEST', request_id=self.rid, slots=self.slots, route=route,
                 owner_identity=owner, cached=cached, s_done=cached or route == 'DF_SC',
                 **self.metric_metadata())
        if cached:
            return
        self.bid += 1
        self.add('METRIC_BUILD_BEGIN', build_id=self.bid, owner_id=self.rid, identity=('J', 'j'),
                 shape=(2, 2), recalculate=(1, 1), policy=POLICY)
        j = self.add('J', identity=('J', 'j'), extra=np.array(POLICY[1:]), matrix=self.j)
        self.add('METRIC_J_LINK', build_id=self.bid, producer_serial=j['header'][0])
        self.add('METRIC_BUILD_END', build_id=self.bid, info=0, identity=('J', 'j'), shape=(2, 2))
        self.add('METRIC_END', request_id=self.rid, info=0, done=True,
                 identity=owner, metric_identity=('J', 'j'))

    def subsets(self):
        self.sid += 1
        self.add('SOLVE_BEGIN', fit_control=(self.sid, 4, 10, 1, 0, 0),
                 x_identity=('NN', 'nn'), fit=FIT)
        self.add('SOLVE_END', solve_end=(self.sid, 0))
        for kind, values in self.data.items():
            for row, value in enumerate(values, 1):
                self.add(kind+'_ROW', destination=(kind, kind.lower()), parent=('NN', 'nn'),
                         ar=row, ij=r.subset_parent_index(kind, row, 2, 2), row=value)
            self.fit[kind] = FIT
            self.add('FIT_METADATA', identity=(kind, kind.lower()), shape=values.shape, fit=FIT)

    def density(self):
        d = self.add('KERNEL_DENSITY', density_identity=('OO', 'oo'),
                     coefficients=2*(self.data['OO'][0]+self.data['OO'][2]))
        self.add('KERNEL_SOURCE', identity=('K', 'k'), density_identity=('OO', 'oo'),
                 density_serial=d['header'][0], constrained=0)

    def retag(self):
        self.fit['OV'] = ((1, 0, 3, 1), FIT[1], FIT[2])
        self.add('FIT_METADATA', identity=('OV', 'ov'), shape=(4, 2), fit=self.fit['OV'])

    def tensor(self, kind='OVOV', cached=False, hessian='h1'):
        self.tid += 1
        left, right = ('OV', 'OV') if kind == 'OVOV' else ('VV', 'OO')
        def operand(role):
            return dict(identity=(role, role.lower()), shape=self.data[role].shape, fit=self.fit[role])
        meta = dict(slots=self.slots, tensor_type=kind, description=('A' if self.slots[0] == 7 else 'B')*4,
                    identity=(kind, kind.lower()), shape=(len(self.data[left]), len(self.data[right])),
                    mapping=(2, 2, 2, 2) if kind == 'OVOV' else (1, 1, 0, 0), flags=(1, 0, 0, 0, 0))
        self.add('DSD_REQUEST', request_id=self.tid, cached=cached, s_done=True, transpose=(0, 0),
                 request_switch=0, left=operand(left), right=operand(right), **meta, **self.metric_metadata())
        if not cached:
            self.add('DSD_COMPLETE', request_id=self.tid, info=0, done=True, **meta)
            self.built[kind] = self.data[left] @ (self.j @ self.data[right].T)
        self.add('HESSIAN_TENSOR', hessian=hessian, tensor_type=kind, description=meta['description'],
                 identity=meta['identity'], mapping=meta['mapping']+(1, 0, 1, 0), exchange=.25,
                 matrix=self.built.get(kind, np.zeros(meta['shape'])))


def scenario(overwrite=False, slots=(7, 11)):
    s = Scenario(slots)
    s.metric()
    s.subsets()
    s.density()
    s.retag()
    s.tensor()
    s.tensor('VVOO')
    if overwrite:
        s.data = {k: v*2 for k, v in s.data.items()}
        s.subsets()
        s.j *= 3
        s.metric('DF_SC')
    s.metric(cached=True)
    s.tensor(cached=True, hessian='h2')
    s.tensor('VVOO', cached=True, hessian='h2')
    return s


def report(s):
    return r.validate_subset_events(s.events)['integral_cache_provenance']


@pytest.mark.parametrize('slots', [(7, 11), (11, 7)])
@pytest.mark.parametrize('overwrite', [False, True])
def test_fresh_cache_preserves_original_successful_dependencies(slots, overwrite):
    s = scenario(overwrite, slots)
    result = report(s)
    assert result['complete_integral_cache_provenance'] and result['numerical_passed']
    assert result['tensor_generations'] == 2
    for a, b in zip(result['consumers'][:2], result['consumers'][2:]):
        assert b['cached'] and not a['cached']
        for key in ('generation', 'completion', 'left', 'right', 'metric_generation', 'producer_j'):
            assert a[key] == b[key]
        assert b['left']['parent_solve_id'] == 1
        assert b['left']['fit'][0][2] == 1  # not the current retag
    if overwrite:
        s.tensor()
        s.tensor('VVOO')
        later = report(s)
        assert later['tensor_generations'] == 4
        assert later['consumers'][-1]['left']['parent_solve_id'] == 2
        assert later['consumers'][-1]['producer_j'] != result['consumers'][-1]['producer_j']
        assert later['consumers'][-1]['metric_route'] == 'DF_SC'


def event(s, kind, n=0):
    return [e for e in s.events if e['kind'] == kind][n]


def renumber(events):
    mapping = {e['header'][0]: i for i, e in enumerate(events, 1)}
    for i, e in enumerate(events, 1):
        e['header'] = (i,) + e['header'][1:]
        for field in ('producer_serial', 'density_serial'):
            if field in e:
                e[field] = mapping.get(e[field], -1)


@pytest.mark.parametrize('fault', [
    'unknown_cache', 'missing_completion', 'failed_completion', 'wrong_completion_id',
    'duplicate_completion', 'nested_request', 'missing_request', 'consumer_before_completion',
    'wrong_consumer_map', 'wrong_consumer_name', 'wrong_consumer_molecule', 'wrong_consumer_shape',
    'wrong_stored_fit', 'wrong_operand', 'output_alias', 'renamed_backing_alias',
    'wrong_metric', 'wrong_slots', 'wrong_descriptor', 'wrong_transpose', 'absent_true',
    'generalized', 'rotated', 'force_rotated', 'switch', 'missing_parent_end', 'pending_subset',
    'wrong_request_map', 'wrong_completion_map', 'completion_not_done', 'changed_completion_name',
    'equal_without_provenance', 'late_coefficient_output_alias', 'pending_eof'])
def test_tensor_state_rejects_unproven_or_ambiguous_consumption(fault):
    s = scenario()
    request, complete, consumer = (event(s, k) for k in ('DSD_REQUEST', 'DSD_COMPLETE', 'HESSIAN_TENSOR'))
    if fault in ('unknown_cache', 'equal_without_provenance'):
        request['cached'] = True
        s.events.remove(complete)
    elif fault == 'missing_completion':
        s.events.remove(complete)
    elif fault == 'failed_completion':
        complete['info'] = -1
    elif fault == 'wrong_completion_id':
        complete['request_id'] = 100
    elif fault == 'duplicate_completion':
        s.events.insert(s.events.index(consumer), copy.deepcopy(complete))
    elif fault == 'nested_request':
        s.events.insert(s.events.index(complete), copy.deepcopy(request))
    elif fault == 'missing_request':
        s.events.remove(request)
    elif fault == 'consumer_before_completion':
        i, j = s.events.index(complete), s.events.index(consumer)
        s.events[i], s.events[j] = s.events[j], s.events[i]
    elif fault == 'wrong_request_map':
        request['mapping'] = (1, 1, 0, 0)
    elif fault == 'wrong_completion_map':
        complete['mapping'] = (1, 1, 0, 0)
    elif fault == 'completion_not_done':
        complete['done'] = False
    elif fault == 'changed_completion_name':
        complete['identity'] = ('Other', 'other')
    elif fault == 'wrong_consumer_map':
        consumer['mapping'] = (1, 1, 0, 0, 1, 0, 1, 0)
    elif fault == 'wrong_consumer_name':
        consumer['identity'] = ('Other', 'other')
    elif fault == 'wrong_consumer_molecule':
        consumer['description'] = 'BBBB'
    elif fault == 'wrong_consumer_shape':
        consumer['matrix'] = np.eye(2)
    elif fault == 'wrong_stored_fit':
        request['left']['fit'] = FIT  # current observed metadata is NN->OV retag
    elif fault == 'wrong_operand':
        request['left']['identity'] = ('VV', 'vv')
    elif fault == 'output_alias':
        request['identity'] = ('OV', 'ov')
    elif fault == 'renamed_backing_alias':
        request['identity'] = ('other', 'ov')
    elif fault == 'wrong_metric':
        request['metric_identity'] = ('Other', 'other')
    elif fault == 'wrong_slots':
        request['slots'] = (11, 7)
    elif fault == 'wrong_descriptor':
        request['metric_description'] = '_B_B'
    elif fault == 'wrong_transpose':
        request['transpose'] = (1, 1)
    elif fault == 'absent_true':
        request['transpose'] = (0, 1)
    elif fault in ('generalized', 'rotated', 'force_rotated', 'switch'):
        flags = list(request['flags'])
        flags[dict(generalized=1, rotated=2, force_rotated=3, switch=4)[fault]] = 1
        request['flags'] = tuple(flags)
    elif fault == 'missing_parent_end':
        s.events.remove(event(s, 'SOLVE_END'))
    elif fault == 'pending_subset':
        s.events.remove(next(e for e in s.events if e['kind'] == 'FIT_METADATA' and e['identity'] == ('OV', 'ov')))
        s.events.remove(next(e for e in s.events if e['kind'] == 'FIT_METADATA' and e['identity'] == ('OV', 'ov')))
    elif fault == 'late_coefficient_output_alias':
        s.add('FIT_METADATA', identity=('OVOV', 'ovov'), shape=(4, 2), fit=FIT)
    else:
        s.events = s.events[:s.events.index(complete)]
    renumber(s.events)
    with pytest.raises(ValueError):
        report(s)


@pytest.mark.parametrize('fault', [
    'unknown_metric_cache', 'missing_owner', 'wrong_owner', 'missing_build_end', 'failed_build_end',
    'duplicate_build_end', 'missing_wrapper_end', 'failed_wrapper_end', 'missing_j_link', 'wrong_j_link',
    'duplicate_j_link', 'two_equal_js', 'wrong_j_policy', 'wrong_build_identity', 'wrong_wrapper_identity',
    'false_recalculate', 'absent_recalculate', 'wrong_norm', 'constrained_failure', 'geometry_mutation',
    'norm_mutation', 'dimer_write', 'rotation', 'unowned_j', 'schema_marker_only'])
def test_metric_two_route_state_rejects_incomplete_provenance(fault):
    s = scenario(overwrite=True)
    request = event(s, 'METRIC_REQUEST')
    begin, end = event(s, 'METRIC_BUILD_BEGIN'), event(s, 'METRIC_BUILD_END')
    if fault == 'unknown_metric_cache':
        request['cached'] = request['s_done'] = True
    elif fault == 'missing_owner':
        s.events.remove(request)
    elif fault == 'wrong_owner':
        begin['owner_id'] = 999
    elif fault == 'missing_build_end':
        s.events.remove(end)
    elif fault == 'failed_build_end':
        end['info'] = -1
    elif fault == 'duplicate_build_end':
        s.events.insert(s.events.index(end), copy.deepcopy(end))
    elif fault == 'missing_wrapper_end':
        s.events.remove(event(s, 'METRIC_END'))
    elif fault == 'failed_wrapper_end':
        event(s, 'METRIC_END')['info'] = -2
    elif fault == 'missing_j_link':
        s.events.remove(event(s, 'METRIC_J_LINK'))
    elif fault == 'wrong_j_link':
        event(s, 'METRIC_J_LINK')['producer_serial'] = 1
    elif fault == 'duplicate_j_link':
        link = event(s, 'METRIC_J_LINK')
        s.events.insert(s.events.index(link), copy.deepcopy(link))
    elif fault == 'two_equal_js':
        j = event(s, 'J')
        s.events.insert(s.events.index(j), copy.deepcopy(j))
    elif fault == 'wrong_j_policy':
        event(s, 'J')['extra'][0] = .1
    elif fault == 'wrong_build_identity':
        begin['identity'] = ('Other', 'other')
    elif fault == 'wrong_wrapper_identity':
        event(s, 'METRIC_END')['identity'] = ('Other', 'other')
    elif fault == 'false_recalculate':
        begin['recalculate'] = (1, 0)
    elif fault == 'absent_recalculate':
        begin['recalculate'] = (0, 1)
    elif fault == 'wrong_norm':
        request['policy'] = (2, 1e-12, 1e-18)
    elif fault == 'constrained_failure':
        event(s, 'METRIC_END', 1)['info'] = -1
    elif fault == 'geometry_mutation':
        s.add('CACHE_CHECK', slots=s.slots, policy=POLICY, initialized=True, same_geometry=False)
    elif fault == 'norm_mutation':
        s.add('CACHE_CHECK', slots=s.slots, policy=(1, .1, 1e-18), initialized=True, same_geometry=True)
    elif fault in ('dimer_write', 'rotation'):
        s.add('CACHE_UNSUPPORTED', reason=fault)
    elif fault == 'unowned_j':
        s.add('J', identity=('J', 'j'), extra=np.array(POLICY[1:]), matrix=s.j)
    else:
        s.events = [e for e in s.events if e['kind'] not in r.cache.CACHE_KINDS]
    renumber(s.events)
    with pytest.raises(ValueError):
        report(s)


def test_numerically_bad_tensor_does_not_destroy_structural_diagnostic():
    s = scenario()
    event(s, 'HESSIAN_TENSOR')['matrix'][0, 1] += 1
    result = report(s)
    assert result['complete_integral_cache_provenance']
    assert not result['numerical_passed']
    assert result['scaled_tolerance'] == 1e-9


def test_numerical_equality_never_collapses_successful_metric_generations():
    s = scenario(overwrite=True)
    original = event(s, 'J')['matrix'].copy()
    event(s, 'J', 1)['matrix'] = original.copy()
    s.j = original.copy()
    s.tensor()
    result = report(s)
    assert result['metric_generations'] == 2
    assert result['consumers'][0]['producer_j'] != result['consumers'][-1]['producer_j']
    assert result['consumers'][0]['metric_generation'] != result['consumers'][-1]['metric_generation']
    assert result['numerical_passed']


def test_snapshots_are_owned_and_retag_is_not_generation():
    s = scenario(overwrite=True)
    state = r.cache.CacheState()
    result = r.validate_subset_events(s.events, cache_state=state)
    original = state.consumers[0]['generation']['left']['matrix'].copy()
    event(s, 'OV_ROW')['row'][:] = 200
    event(s, 'J')['matrix'][:] = 400
    assert np.array_equal(state.consumers[0]['generation']['left']['matrix'], original)
    assert state.consumers[0]['generation']['metric']['j']['matrix'][0, 0] == 2
    assert result['complete_parent_solve_and_cache_provenance']


def test_cached_current_inputs_are_context_not_original_dependencies():
    s = scenario(overwrite=True)
    # A later successful parent/subset replacement now has lambda1. The cached
    # lambda0 tensor still uses its original operands, not this current fit.
    new_fit = ((1, 0, 1, 1), (1., 0., 0., 0.), (0, 1))
    parent = event(s, 'SOLVE_BEGIN', 1)
    parent['fit'] = new_fit
    for e in s.events:
        if e['header'][0] > parent['header'][0]:
            if e['kind'] == 'FIT_METADATA':
                e['fit'] = new_fit
            elif e['kind'] == 'DSD_REQUEST':
                e['left']['fit'] = e['right']['fit'] = new_fit
    result = report(s)
    assert result['consumers'][2]['left']['fit'] == FIT
    assert result['consumers'][2]['request_context']['left']['fit'] == new_fit
    assert result['consumers'][2]['request_context']['differs_from_original']
    assert result['numerical_passed']


def lifecycle_body(e):
    """Independent test serializer for the provisional scalar wire contract."""
    lines = []
    def put(*values):
        lines.append(' '.join(str(int(x)) if isinstance(x, bool) else str(x) for x in values))
    def identity(value):
        lines.extend(value)
    def policy(value):
        put(value[0]); lines.extend(('METRIC_POLICY', '2')); put(*value[1:])
    def fit(value):
        put(*value[0]); lines.extend(('FIT_PARAMETERS', '4')); put(*value[1]); put(*value[2])
    def metric():
        put(e['metric_type']); put(e['metric_description']); identity(e['metric_identity'])
        put(*e['metric_shape']); put(*e['metric_flags']); policy(e['policy'])
    def output():
        put(*e['slots']); put(e['tensor_type']); put(e['description']); identity(e['identity'])
        put(*e['shape']); put(*e['mapping']); put(*e['flags'])
    kind = e['kind']
    if kind == 'CACHE_CHECK':
        put(*e['slots']); policy(e['policy']); put(e['initialized']); put(e['same_geometry'])
    elif kind == 'CACHE_UNSUPPORTED':
        put(e['reason'])
    elif kind == 'METRIC_REQUEST':
        put(e['request_id']); put(*e['slots']); put(e['route']); identity(e['owner_identity'])
        put(e['cached']); put(e['s_done']); metric()
    elif kind == 'METRIC_BUILD_BEGIN':
        put(e['build_id'], e['owner_id']); identity(e['identity']); put(*e['shape'])
        put(*e['recalculate']); policy(e['policy'])
    elif kind == 'METRIC_J_LINK':
        put(e['build_id'], e['producer_serial'])
    elif kind == 'METRIC_BUILD_END':
        put(e['build_id'], e['info']); identity(e['identity']); put(*e['shape'])
    elif kind == 'METRIC_END':
        put(e['request_id'], e['info']); put(e['done']); identity(e['identity']); identity(e['metric_identity'])
    elif kind == 'DSD_REQUEST':
        put(e['request_id']); output(); put(e['cached']); put(e['s_done'])
        put(*e['transpose']); put(e['request_switch']); metric()
        for operand in (e['left'], e['right']):
            identity(operand['identity']); put(*operand['shape']); fit(operand['fit'])
    elif kind == 'DSD_COMPLETE':
        put(e['request_id'], e['info']); put(e['done']); output()
    else:
        raise AssertionError(kind)
    return '\n'.join(lines)+'\n'


@pytest.mark.parametrize('kind', sorted(r.cache.CACHE_KINDS))
def test_schema8_scalar_lifecycle_roundtrip_and_legacy_rejection(tmp_path, kind):
    s = scenario(overwrite=True)
    s.add('CACHE_UNSUPPORTED', reason='dimer subset write')
    e = event(s, kind)
    body = lifecycle_body(e)
    text = ('ISAPOL_RESPONSE_EVENT 8\n'+' '.join(map(str, e['header']))+
            '\n'+kind+'\nWater\ndalton\n'+body+'END_RESPONSE_EVENT\n')
    path = tmp_path/'event.dat'
    path.write_text(text)
    assert r.read_event(path) == e
    path.write_text(text.replace('EVENT 8', 'EVENT 7'))
    with pytest.raises(ValueError, match='requires schema8'):
        r.read_event(path)
    path.write_text(text+'trailing\n')
    with pytest.raises(ValueError, match='Trailing'):
        r.read_event(path)


def test_scalar_parser_rejects_boolean_not_zero_or_one(tmp_path):
    s = scenario()
    e = event(s, 'CACHE_CHECK')
    body = lifecycle_body(e)
    path = tmp_path/'event.dat'
    path.write_text('ISAPOL_RESPONSE_EVENT 8\n1 7 4 2 2 2 4\nCACHE_CHECK\nWater\ndalton\n'+
                    body[:-4]+'2\n0\nEND_RESPONSE_EVENT\n')
    with pytest.raises(ValueError, match='boolean'):
        r.read_event(path)


def full_wire_fixture():
    """Upgrade the independent legacy Hessian fixture with explicit scalar epochs."""
    spec = importlib.util.spec_from_file_location('cache_hessian_fixture',
        Path(__file__).with_name('test_isapol_hessian_replay.py'))
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    f = h.f
    events = []
    def add(kind, body):
        events.append((kind, body)); return len(events)
    def lifecycle(kind, **fields):
        return add(kind, lifecycle_body(dict(kind=kind, **fields)))
    def metadata(identity, count):
        return '\n'.join(identity)+'\n'+f'{count} 1\n1 0 1 1\n'+f.vector('FIT_PARAMETERS', [0]*4)+'0 0\n'
    metric = dict(metric_type='S___', metric_description='_A_A', metric_identity=('Coulomb', 'Jfile'),
                  metric_shape=(1, 1), metric_flags=(1, 0, 0, 0), policy=POLICY)
    lifecycle('CACHE_CHECK', slots=(1, 2), policy=POLICY, initialized=False, same_geometry=False)
    raw_begin = next(body for kind, body in h.hessian_fixture() if kind == 'FIT_BEGIN')
    ordinary_started = False
    for kind, body in h.hessian_fixture():
        if kind == 'J':
            lifecycle('METRIC_REQUEST', request_id=1, slots=(1, 2), route='DF_S',
                      owner_identity=('Coulomb', 'Jfile'), cached=False, s_done=False, **metric)
            lifecycle('METRIC_BUILD_BEGIN', build_id=1, owner_id=1, identity=('Coulomb', 'Jfile'),
                      shape=(1, 1), recalculate=(1, 1), policy=POLICY)
            sid = add(kind, body)
            lifecycle('METRIC_J_LINK', build_id=1, producer_serial=sid)
            lifecycle('METRIC_BUILD_END', build_id=1, info=0, identity=('Coulomb', 'Jfile'), shape=(1, 1))
            lifecycle('METRIC_END', request_id=1, info=0, done=True,
                      identity=('Coulomb', 'Jfile'), metric_identity=('Coulomb', 'Jfile'))
            continue
        if kind == 'FIT_BEGIN':
            add('SOLVE_BEGIN', body)
        if kind == 'OV_ROW' and 'Dov0\n' in body and not ordinary_started:
            ordinary_started = True
            begin = (raw_begin.replace('1 4 10 1 0 0', '2 4 10 1 0 0').replace('Sc__', 'S___')
                     .replace('Full coefficients\nDfull', 'Full zero\nDfull0')
                     .replace('FIT_PARAMETERS\n4\n1 0 0 0', 'FIT_PARAMETERS\n4\n0 0 0 0'))
            assert begin.endswith('0 1\n')
            begin = begin[:-4]+'0 0\n'
            add('SOLVE_BEGIN', begin); add('SOLVE_END', '2 0\n')
        if kind == 'KERNEL_SOURCE':
            sid = add('KERNEL_DENSITY', 'Density fit\nDoo0\nRho\nAUX1\nRho expansion\nrho\n1 0 1\n'+
                      f.vector('COEFFICIENTS', [.2]))
            body += str(sid)+'\n'
        if kind == 'HESSIAN_TENSOR':
            parts = body.splitlines()
            hessian, tensor, description, name, filename = parts[:5]
            is_ov = tensor == 'OVOV'
            tid = 1 + (0 if hessian == 'h1' else 2) + (0 if is_ov else 1)
            out = dict(slots=(1, 2), tensor_type=tensor, description=description, identity=(name, filename),
                       shape=(4, 4) if is_ov else (3, 3), mapping=(2, 2, 2, 2) if is_ov else (1, 1, 0, 0),
                       flags=(1, 0, 0, 0, 0))
            ov = dict(identity=('Ordinary OV', 'Dov0'), shape=(4, 1), fit=FIT)
            left = ov if is_ov else dict(identity=('VV coeff', 'Dvv0'), shape=(3, 1), fit=FIT)
            right = ov if is_ov else dict(identity=('Density fit', 'Doo0'), shape=(3, 1), fit=FIT)
            lifecycle('DSD_REQUEST', request_id=tid, cached=hessian == 'h2', s_done=True,
                      transpose=(0, 0), request_switch=0, left=left, right=right, **metric, **out)
            if hessian == 'h1':
                lifecycle('DSD_COMPLETE', request_id=tid, info=0, done=True, **out)
        add(kind, body)
        if kind == 'FIT_END':
            add('SOLVE_END', '1 0\n')
        if kind == 'FIT_METADATA' and 'Dov0\n' in body:
            for role, identity, values in [('OO', ('Density fit', 'Doo0'), [-.2, .15, .3]),
                                           ('VV', ('VV coeff', 'Dvv0'), [.11, .21, .13])]:
                for row, value in enumerate(values, 1):
                    add(role+'_ROW', '\n'.join(identity)+'\nFull zero\nDfull0\n'+
                        f'{row} {r.subset_parent_index(role, row, 2, 2)}\n'+f.vector('ROW', [value]))
                add('FIT_METADATA', metadata(identity, 3))
    return events, f


def test_complete_schema8_wire_replay_and_independent_tensor_gate(tmp_path):
    events, f = full_wire_fixture()
    f.write_events(tmp_path, events, version=8)
    with pytest.raises(ValueError, match='CDF event count'):
        r.replay(tmp_path)  # default still requires all eleven frequencies
    parsed = [r.read_event(p) for p in sorted(tmp_path.glob('isapol-response-*.dat'))]
    h1, h2 = (next(e['matrix'] for e in parsed if e['kind'] == kind) for kind in ('H1', 'H2'))
    d = np.array([[.2], [.3], [-.1], [.25]])
    policy = next(body for kind, body in events if kind == 'POLICY')
    for i in range(2, 11):
        w = -float(i)
        cdf = d.T @ np.linalg.solve(h2 @ h1 - w*np.eye(4), -4*(h2 @ d))
        events.extend([('POLICY', policy), ('CDF', 'Response\nCDFfile\n'+
                      f.vector('EXTRA', [w, 0])+f.matrix('MATRIX', cdf))])
    f.write_events(tmp_path, events, version=8)
    result = r.replay(tmp_path)
    assert result['passed']
    provenance = result['subset_density_provenance']
    assert provenance['complete_parent_solve_and_cache_provenance']
    assert provenance['integral_cache_provenance']['numerical_passed']
    assert len(provenance['integral_cache_provenance']['consumers']) == 4
    assert result['hessian_reconstruction']['response_D_J_D_difference']['max_scaled'] > 1e-3
    # Corrupt only the supplied ordinary OO coefficient snapshot. Hessian/CDF and
    # raw-fit controls remain valid; the new VVOO construction gate must fail.
    for i, (kind, body) in enumerate(events, 1):
        if kind == 'OO_ROW' and '1 1\nROW\n' in body:
            path = tmp_path/f'isapol-response-{i:06d}.dat'
            path.write_text(path.read_text().replace(f.vector('ROW', [-.2]), f.vector('ROW', [-.25])))
    bad = r.replay(tmp_path)
    assert not bad['passed']
    assert bad['subset_density_provenance']['complete_integral_cache_provenance']
    assert not bad['subset_density_provenance']['integral_cache_provenance']['numerical_passed']
    assert bad['hessian_reconstruction']['h1_error']['max_scaled'] < 1e-14
    assert all(x['cdf_error']['max_scaled'] < 1e-14 for x in bad['frequencies'])


def test_parent_solve_cannot_overwrite_subset_storage():
    s = scenario()
    s.add('SOLVE_BEGIN', fit_control=(2, 4, 10, 1, 0, 0),
          x_identity=('OV', 'ov'), fit=FIT)
    s.add('SOLVE_END', solve_end=(2, 0))
    s.tensor()
    with pytest.raises(ValueError, match='storage role alias'):
        report(s)


def test_subset_parent_cannot_alias_destination():
    s = scenario()
    event(s, 'OV_ROW')['parent'] = ('OV', 'ov')
    with pytest.raises(ValueError):
        report(s)


def test_constrained_metric_requires_observed_ordinary_wrapper():
    s = Scenario()
    s.metric('DF_SC')
    s.subsets(); s.density(); s.tensor(); s.tensor('VVOO')
    with pytest.raises(ValueError, match='observed completed ordinary wrapper'):
        report(s)


@pytest.mark.parametrize('fault', ['parent_overwrite', 'unobserved_ordinary'])
def test_wire_rejects_parent_alias_and_unobserved_ordinary(tmp_path, fault):
    events, f = full_wire_fixture()
    if fault == 'parent_overwrite':
        body = next(body for kind, body in events if kind == 'SOLVE_BEGIN')
        body = body.replace('1 4 10 1 0 0', '3 4 10 1 0 0')
        body = body.replace('Full coefficients\nDfull', 'Ordinary OV\nDov0')
        events.extend([('SOLVE_BEGIN', body), ('SOLVE_END', '3 0\n')])
        message = 'storage role alias'
    else:
        f.write_events(tmp_path, events, version=8)
        index = next(i for i, (kind, _) in enumerate(events) if kind == 'METRIC_REQUEST')
        e = r.read_event(tmp_path/f'isapol-response-{index+1:06d}.dat')
        e.update(route='DF_SC', owner_identity=('Sc', 'sc'), s_done=True)
        events[index] = ('METRIC_REQUEST', lifecycle_body(e))
        message = 'observed completed ordinary wrapper'
    f.write_events(tmp_path, events, version=8)
    parsed = [r.read_event(p) for p in sorted(tmp_path.glob('isapol-response-*.dat'))]
    with pytest.raises(ValueError, match=message):
        r.validate_subset_events(parsed)


def test_structural_completeness_is_not_observer_validation():
    result = r.validate_subset_events(scenario().events)
    assert result['structural_cache_provenance_complete']
    assert not result['cache_observer_validated']
    assert result['integral_cache_provenance']['structural_provenance_complete']
    assert not result['integral_cache_provenance']['observer_validated']


def test_rebuilt_h2_uses_its_own_consumed_tensors(tmp_path):
    events, f = full_wire_fixture()
    f.write_events(tmp_path, events, version=8)
    parsed = [r.read_event(p) for p in sorted(tmp_path.glob('isapol-response-*.dat'))]
    h1, old_h2 = (next(e['matrix'] for e in parsed if e['kind'] == kind) for kind in ('H1', 'H2'))
    energy = np.array([-1., -.7, .2, 1.])
    delta = np.diag((energy[2:, None]-energy[:2]).ravel())
    # Rebuilding the metric with J=2*J_old doubles BOTH actual tensor inputs.
    # H2's exchange part is linear in those tensors; H1 is left unchanged.
    h2 = delta+2*(old_h2-delta)
    d = np.array([[.2], [.3], [-.1], [.25]])
    updated = []
    inserted = False
    def add_lifecycle(kind, **fields):
        updated.append((kind, lifecycle_body(dict(kind=kind, **fields))))
    for (kind, body), e in zip(events, parsed):
        if kind == 'DSD_REQUEST' and e['cached']:
            if not inserted:
                inserted = True
                meta = {key: e[key] for key in ('metric_type', 'metric_description', 'metric_identity',
                    'metric_shape', 'metric_flags', 'policy')}
                add_lifecycle('METRIC_REQUEST', request_id=2, slots=(1, 2), route='DF_SC',
                    owner_identity=('Sc', 'Scfile'), cached=False, s_done=True, **meta)
                add_lifecycle('METRIC_BUILD_BEGIN', build_id=2, owner_id=2,
                    identity=meta['metric_identity'], shape=(1, 1), recalculate=(1, 1), policy=POLICY)
                updated.append(('J', 'Coulomb\nJfile\n'+f.vector('EXTRA', POLICY[1:])+
                                f.matrix('MATRIX', [[2.]])))
                add_lifecycle('METRIC_J_LINK', build_id=2, producer_serial=len(updated))
                add_lifecycle('METRIC_BUILD_END', build_id=2, info=0,
                              identity=meta['metric_identity'], shape=(1, 1))
                add_lifecycle('METRIC_END', request_id=2, info=0, done=True,
                    identity=('Sc', 'Scfile'), metric_identity=meta['metric_identity'])
            request = copy.deepcopy(e); request['cached'] = False
            updated.append((kind, lifecycle_body(request)))
            complete = {key: e[key] for key in ('request_id', 'slots', 'tensor_type', 'description',
                                               'identity', 'shape', 'mapping', 'flags')}
            add_lifecycle('DSD_COMPLETE', info=0, done=True, **complete)
            continue
        if kind == 'HESSIAN_TENSOR' and e['hessian'] == 'h2':
            body = body.split('MATRIX\n')[0]+f.matrix('MATRIX', 2*e['matrix'])
        elif kind == 'H2':
            body = body.split('MATRIX\n')[0]+f.matrix('MATRIX', h2)
        elif kind == 'CDF':
            w = e['extra'][0]
            value = d.T@np.linalg.solve(h2@h1-w*np.eye(4), -4*(h2@d))
            body = body.split('MATRIX\n')[0]+f.matrix('MATRIX', value)
        updated.append((kind, body))
    f.write_events(tmp_path, updated, version=8)
    result = r.replay(tmp_path, expected_frequencies=2)
    assert result['passed']
    consumers = result['subset_density_provenance']['integral_cache_provenance']['consumers']
    assert consumers[0]['generation'] != consumers[2]['generation']
    assert consumers[0]['producer_j'] != consumers[2]['producer_j']
    assert result['hessian_reconstruction']['h2_error']['max_scaled'] < 1e-14
    # Negative reference: H2/CDF internally consistent with old H1-era tensors.
    # Only the independently consumed new-H2 tensor reconstruction should reject it.
    for index, (kind, body) in enumerate(updated):
        if kind == 'H2':
            updated[index] = (kind, body.split('MATRIX\n')[0]+f.matrix('MATRIX', old_h2))
        elif kind == 'CDF':
            e = r.read_event(tmp_path/f'isapol-response-{index+1:06d}.dat')
            value = d.T@np.linalg.solve(old_h2@h1-e['extra'][0]*np.eye(4), -4*(old_h2@d))
            updated[index] = (kind, body.split('MATRIX\n')[0]+f.matrix('MATRIX', value))
    f.write_events(tmp_path, updated, version=8)
    rejected = r.replay(tmp_path, expected_frequencies=2)
    assert not rejected['passed']
    assert rejected['hessian_reconstruction']['h2_error']['max_scaled'] > 1e-3
    assert all(x['cdf_error']['max_scaled'] < 1e-14 for x in rejected['frequencies'])


def test_wire_distinct_coefficient_storage_can_share_source_display_names(tmp_path):
    # define_D_matrix uses the same 'D Full...' name for ordinary D and D_c;
    # fill_Aov derives the same display label from each distinct parent.
    events, f = full_wire_fixture()
    events = [(kind, body.replace('Full zero\n', 'Full coefficients\n')
                         .replace('Ordinary OV\n', 'OV coefficients\n')) for kind, body in events]
    f.write_events(tmp_path, events, version=8)
    result = r.replay(tmp_path, expected_frequencies=2)
    assert result['passed']
    consumers = result['subset_density_provenance']['integral_cache_provenance']['consumers']
    assert all(c['left']['parent_solve_id'] == 2 for c in consumers)
    assert result['hessian_reconstruction']['response_D_J_D_difference']['max_scaled'] > 1e-3
    # Reusing a display label alone cannot transfer a committed generation.
    events = [(kind, body.replace('Dov0\n', 'unobserved-file\n') if kind == 'DSD_REQUEST' else body)
              for kind, body in events]
    f.write_events(tmp_path, events, version=8)
    with pytest.raises(ValueError, match='Unexplained storage rename'):
        r.replay(tmp_path, expected_frequencies=2)


@pytest.mark.parametrize('role', ['coefficient_parent', 'coefficient_subset'])
def test_repeated_display_name_requires_explicit_same_role_write(role):
    s = r.cache.CacheState()
    s.identity(('display', 'first'), role, coefficient_write=True)
    with pytest.raises(ValueError, match='Unexplained storage rename'):
        s.identity(('display', 'second'), role)
    s.identity(('display', 'second'), role, coefficient_write=True)
    assert ('display', 'first') != ('display', 'second')
    assert not s.tensors and not s.metrics  # registration is never commitment
    other = 'coefficient_subset' if role == 'coefficient_parent' else 'coefficient_parent'
    with pytest.raises(ValueError, match='Unexplained storage rename'):
        s.identity(('display', 'third'), other, coefficient_write=True)
    with pytest.raises(ValueError, match='Backing-file alias'):
        s.identity(('different display', 'first'), role, coefficient_write=True)


def test_present_false_transpose_and_cached_constrained_wrapper():
    s = scenario(overwrite=True)
    event(s, 'DSD_REQUEST')['transpose'] = (1, 0)
    s.metric('DF_SC', cached=True)
    assert report(s)['numerical_passed']
