"""Schema8 cache reader for the scalar observer in capture_response.py.

No identity is inferred from numerical equality. Inputs are normalized scalar
lifecycle records plus the existing resident J/tensor and committed subset events.
Only the owning chronological subset pass may supply the live subset registry.
This module does not instrument CamCASP or certify an unobserved mutation route.
"""
from copy import deepcopy
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def error(x, y):
    require(x.shape == y.shape and np.all(np.isfinite(x)) and np.all(np.isfinite(y)),
            'Nonfinite/mismatched cache reconstruction')
    delta = float(np.max(np.abs(x - y)))
    return dict(max_absolute=delta, max_scaled=delta / max(1., float(np.max(np.abs(y)))))


CACHE_KINDS = frozenset(('CACHE_CHECK', 'CACHE_UNSUPPORTED', 'METRIC_REQUEST',
    'METRIC_BUILD_BEGIN', 'METRIC_J_LINK', 'METRIC_BUILD_END', 'METRIC_END',
    'DSD_REQUEST', 'DSD_COMPLETE'))


def read_fields(r, e, fit_parameters):
    """Fortran-list-output-friendly schema8 scalar wire contract.

    Existing resident J and HESSIAN_TENSOR records remain unchanged. Lifecycle
    records carry no numerical operands. Every boolean is encoded as 0 or 1.
    This reader alone cannot authenticate an emitter or its noninterference.
    """
    def ints(n):
        return tuple(map(int, r.numbers(n, integer=True)))
    def identity():
        return r.line(), r.line()
    def boolean():
        value, = ints(1)
        require(value in (0, 1), 'Invalid cache boolean')
        return bool(value)
    def policy():
        norm, = ints(1)
        return (norm,) + tuple(map(float, r.vector('METRIC_POLICY', 2)))
    def metric():
        e['metric_type'], e['metric_description'] = r.line(), r.line()
        e['metric_identity'], e['metric_shape'] = identity(), ints(2)
        e['metric_flags'], e['policy'] = ints(4), policy()
    def output():
        e['slots'] = ints(2)
        e['tensor_type'], e['description'] = r.line(), r.line()
        e['identity'], e['shape'] = identity(), ints(2)
        e['mapping'], e['flags'] = ints(4), ints(5)
    def operand():
        return dict(identity=identity(), shape=ints(2), fit=fit_parameters(r))
    kind = e['kind']
    if kind == 'CACHE_UNSUPPORTED':
        e['reason'] = r.line()
        require(bool(e['reason']), 'Empty unsupported cache route')
    elif kind == 'CACHE_CHECK':
        e['slots'], e['policy'] = ints(2), policy()
        e['initialized'], e['same_geometry'] = boolean(), boolean()
    elif kind == 'METRIC_REQUEST':
        e['request_id'], = ints(1)
        e['slots'], e['route'], e['owner_identity'] = ints(2), r.line(), identity()
        e['cached'], e['s_done'] = boolean(), boolean()
        metric()
    elif kind == 'METRIC_BUILD_BEGIN':
        e['build_id'], e['owner_id'] = ints(2)
        e['identity'], e['shape'] = identity(), ints(2)
        e['recalculate'], e['policy'] = ints(2), policy()
    elif kind == 'METRIC_J_LINK':
        e['build_id'], e['producer_serial'] = ints(2)
    elif kind == 'METRIC_BUILD_END':
        e['build_id'], e['info'] = ints(2)
        e['identity'], e['shape'] = identity(), ints(2)
    elif kind == 'METRIC_END':
        e['request_id'], e['info'] = ints(2)
        e['done'], e['identity'], e['metric_identity'] = boolean(), identity(), identity()
    elif kind == 'DSD_REQUEST':
        e['request_id'], = ints(1)
        output()
        e['cached'], e['s_done'] = boolean(), boolean()
        e['transpose'], e['request_switch'] = ints(2), ints(1)[0]
        metric()
        e['left'], e['right'] = operand(), operand()
    elif kind == 'DSD_COMPLETE':
        e['request_id'], e['info'] = ints(2)
        e['done'] = boolean()
        output()
    else:
        raise ValueError('Unknown cache lifecycle event')


class CacheState:
    """One selected, fixed-geometry monomer; no recovery from failed writes.

    Metric construction has two owners (DF_S and DF_SC). A linked resident J is
    pending until successful common construction. Ordinary construction additionally
    waits for wrapper success; constrained construction replaces ordinary S at the
    inner success even when S.done was already true. A subsequent wrapper failure
    invalidates the entire trace, not just that generation.
    """

    def __init__(self):
        self.serial = 0
        self.header = None
        self.aliases = {}
        self.names = {}
        self.storage_roles = {}
        self.pending_coefficients = set()
        self.policy = None
        self.slots = None
        self.checked = False
        self.owners = {}
        self.owner_history = {}
        self.owner_ids = set()
        self.builds = {}
        self.build_ids = set()
        self.js = {}
        self.linked_js = set()
        self.metrics = {}
        self.metric_history = []
        self.pending = {}
        self.request_ids = set()
        self.tensors = {}
        self.ready = {}
        self.tensor_history = []
        self.consumers = []

    def identity(self, identity, role=None, *, coefficient_write=False):
        require(isinstance(identity, tuple) and len(identity) == 2 and
                all(isinstance(x, str) and x.strip() == x and x for x in identity),
                'Invalid cache storage identity')
        name, filename = identity
        # Reject path spelling aliases too: no normalization may invent identity.
        from pathlib import PurePosixPath
        require(str(PurePosixPath(filename)) == filename and '..' not in PurePosixPath(filename).parts,
                'Unsupported cache backing path')
        require(self.aliases.get(filename, name) == name, 'Backing-file alias')
        files = self.names.get(name, set())
        if files and filename not in files:
            # Pinned define_D_matrix assigns the SAME display name to D and D_c;
            # fill_Aoo/Aov/Avv inherit that display name. Only an explicit new
            # coefficient write may introduce another backing file under it.
            # Never transfer a generation by display name; all bindings remain
            # keyed by (name, filename), with disjoint parent/subset storage roles.
            require(coefficient_write and role in ('coefficient_parent', 'coefficient_subset') and
                    all(self.storage_roles.get(f) == role for f in files),
                    'Unexplained storage rename')
        self.aliases[filename] = name
        self.names.setdefault(name, set()).add(filename)
        if role is not None:
            require(self.storage_roles.get(filename, role) == role, 'Output/input storage role alias')
            self.storage_roles[filename] = role
        return identity

    def geometry(self, e):
        slots = e['slots']
        require(len(slots) == 2 and all(type(x) is int and x > 0 for x in slots) and slots[0] != slots[1],
                'Invalid explicit A/B slots')
        require(self.header[0] in slots, 'Selected molecule absent from A/B slots')
        require(self.slots is None or self.slots == slots, 'A/B mapping mutation')
        self.slots = slots
        return 'A' if self.header[0] == slots[0] else 'B'

    def check_policy(self, policy):
        require(len(policy) == 3 and policy[0] == 1 and np.all(np.isfinite(policy)) and
                policy[1] >= 0 and policy[2] > 0, 'Unsupported metric norm/policy')
        require(self.policy is None or self.policy == policy, 'Metric policy mutation')
        self.policy = policy

    def metric_metadata(self, e):
        side = self.geometry(e)
        require(e['metric_type'] == 'S___' and e['metric_description'] == f'_{side}_{side}',
                'Wrong mapped monomer metric')
        require(e['metric_shape'] == (self.header[2], self.header[2]), 'Metric shape mismatch')
        require(e['metric_flags'] == (1, 0, 0, 0), 'Unsupported metric definition/rotation/switch')
        self.check_policy(e['policy'])
        return self.identity(e['metric_identity'], 'metric')

    def output(self, e):
        side = self.geometry(e)
        _, _, m, o, v, _ = self.header
        require(e['description'] == side * 4, 'Wrong mapped tensor descriptor')
        kind = e['tensor_type']
        require(kind in ('OVOV', 'VVOO'), 'Unsupported tensor role')
        shape, mapping = ((o*v, o*v), (2, 2, o, o)) if kind == 'OVOV' else (
            (v*(v+1)//2, o*(o+1)//2), (1, 1, 0, 0))
        require(e['shape'] == shape and e['mapping'] == mapping, 'Wrong tensor shape/map')
        require(e['flags'] == (1, 0, 0, 0, 0), 'Unsupported tensor definition/generalized/rotation/switch')
        dest = self.identity(e['identity'], 'tensor')
        return dest, (kind, e['description'], dest, shape, mapping, self.slots)

    def subset(self, operand, role, latest):
        dest = self.identity(operand['identity'], 'coefficient_subset')
        require(dest in latest, 'Tensor request without committed subset')
        g = latest[dest]
        require(g['kind'] == role and operand['shape'] == g['metadata']['shape'], 'Wrong tensor operand role/shape')
        require(operand['fit'] == g['observed_metadata']['fit'], 'Tensor stored-parameter mismatch')
        require(g['metadata']['fit'][0][0] == 1 and g['metadata']['fit'][0][2:] == (1, 1) and
                g['metadata']['fit'][1] == (0., 0., 0., 0.), 'Tensor requires original ordinary NN fit')
        require(g['parent_solve'] is not None, 'Tensor operand without successful parent solve')
        return deepcopy(g)

    def consume(self, e, latest):
        serial = e['header'][0]
        require(serial == self.serial + 1, 'Nonsequential cache event stream')
        self.serial = serial
        if self.header is None:
            self.header = e['header'][1:]
        require(self.header == e['header'][1:] and e.get('version') == 8, 'Cache molecule/schema changed')
        kind = e['kind']
        if self.pending:
            require(kind == 'DSD_COMPLETE', 'Interleaved event during pending tensor construction')
        # Track all observed coefficient writes, including outputs not yet committed.
        if kind in ('OO_ROW', 'OV_ROW', 'VV_ROW'):
            self.identity(e['destination'], 'coefficient_subset', coefficient_write=True)
            self.identity(e['parent'], 'coefficient_parent')
            self.pending_coefficients.add(e['destination'])
        elif kind == 'SOLVE_BEGIN':
            # A full-parent solve may not reuse a subset's backing storage in
            # this bounded protocol, even after successful solve completion.
            self.identity(e['x_identity'], 'coefficient_parent', coefficient_write=True)
        elif kind == 'FIT_METADATA':
            dest = self.identity(e['identity'])
            require(self.storage_roles.get(dest[1]) in (None, 'coefficient_parent', 'coefficient_subset'),
                    'Coefficient metadata aliases non-coefficient storage')
            self.pending_coefficients.discard(e['identity'])
        elif kind == 'CACHE_UNSUPPORTED':
            raise ValueError('Unobserved cache mutation route: ' + e['reason'])
        elif kind == 'CACHE_CHECK':
            self.geometry(e)
            self.check_policy(e['policy'])
            require(type(e['initialized']) is bool and type(e['same_geometry']) is bool,
                    'Invalid cache comparison flags')
            if self.checked:
                require(e['initialized'] and e['same_geometry'], 'Geometry/cache initialization mutation')
            else:
                require(not self.metric_history and not self.owners, 'Late initial cache check')
                require(not e['initialized'] or e['same_geometry'], 'Initial geometry mismatch')
            self.checked = True
        elif kind == 'METRIC_REQUEST':
            require(self.checked, 'Metric before observed cache check')
            dest = self.metric_metadata(e)
            owner = self.identity(e['owner_identity'], 'metric')
            route = e['route']
            require(route in ('DF_S', 'DF_SC'), 'Unsupported metric wrapper owner')
            require((owner == dest) == (route == 'DF_S'), 'Wrong metric wrapper storage')
            require(type(e['cached']) is bool and type(e['s_done']) is bool, 'Invalid metric cache flags')
            if route == 'DF_S':
                require(e['cached'] == e['s_done'], 'Ordinary metric cache predicate mismatch')
            else:
                require(e['s_done'] and dest in self.metrics and ('DF_S', dest) in self.owner_history,
                        'Constrained metric without observed completed ordinary wrapper')
            rid = e['request_id']
            require(type(rid) is int and rid > 0 and rid not in self.owner_ids, 'Duplicate/invalid metric request ID')
            self.owner_ids.add(rid)
            require(not self.owners and not self.builds, 'Nested/pending metric owner')
            if e['cached']:
                require(dest in self.metrics, 'Unknown metric cache')
                require((route, owner) in self.owner_history, 'Unknown wrapper cache')
                require(self.owner_history[route, owner]['metric_identity'] == dest, 'Wrapper cache identity changed')
            else:
                self.metrics.pop(dest, None)
                self.owners[rid] = dict(event=deepcopy(e), build=None)
        elif kind == 'METRIC_BUILD_BEGIN':
            bid, rid = e['build_id'], e['owner_id']
            require(type(bid) is int and bid > 0 and bid not in self.build_ids, 'Duplicate/invalid metric build ID')
            require(rid in self.owners and not self.builds, 'Metric build without unique owner')
            owner = self.owners[rid]
            require(owner['build'] is None, 'Repeated construction under metric owner')
            require(e['identity'] == owner['event']['metric_identity'] and
                    e['shape'] == owner['event']['metric_shape'], 'Metric build identity/shape mismatch')
            require(e['recalculate'] == (1, 1), 'Metric build requires present true recalculation')
            require(e['policy'] == owner['event']['policy'], 'Metric build policy mismatch')
            self.build_ids.add(bid)
            self.builds[bid] = dict(event=deepcopy(e), owner=deepcopy(owner['event']), j=None)
        elif kind == 'J':
            self.identity(e['identity'], 'metric')
            require(len(self.builds) == 1, 'Resident J without common metric construction')
            b = next(iter(self.builds.values()))
            require(e['identity'] == b['event']['identity'] and e['matrix'].shape == b['event']['shape'],
                    'Resident J identity/shape mismatch')
            require(tuple(e['extra']) == b['event']['policy'][1:], 'Resident J policy mismatch')
            require(np.all(np.isfinite(e['matrix'])), 'Nonfinite resident J')
            self.js[serial] = deepcopy(e)
        elif kind == 'METRIC_J_LINK':
            bid, sid = e['build_id'], e['producer_serial']
            require(bid in self.builds and sid in self.js and sid not in self.linked_js, 'Unknown/reused J link')
            b = self.builds[bid]
            require(b['j'] is None and b['event']['header'][0] < sid < serial, 'Duplicate/wrong-epoch J link')
            require(self.js[sid]['identity'] == b['event']['identity'], 'Wrong J link identity')
            b['j'] = self.js[sid]
            self.linked_js.add(sid)
        elif kind == 'METRIC_BUILD_END':
            bid = e['build_id']
            require(bid in self.builds, 'Metric completion without pending build')
            b = self.builds.pop(bid)
            require(e['info'] == 0 and e['identity'] == b['event']['identity'] and
                    e['shape'] == b['event']['shape'], 'Failed/wrong metric construction completion')
            require(b['j'] is not None, 'Metric construction without linked J')
            require(sum(b['event']['header'][0] < sid < serial for sid in self.js) == 1,
                    'Metric construction requires exactly one resident J')
            b['end'] = deepcopy(e)
            self.owners[b['event']['owner_id']]['build'] = b
            if b['owner']['route'] == 'DF_SC':
                self.metrics[e['identity']] = b
        elif kind == 'METRIC_END':
            rid = e['request_id']
            require(rid in self.owners and not self.builds, 'Wrapper completion without finished owner')
            owner = self.owners.pop(rid)
            b = owner['build']
            require(b is not None and e['info'] == 0 and e['done'] is True, 'Failed/incomplete metric wrapper')
            require(e['identity'] == owner['event']['owner_identity'] and
                    e['metric_identity'] == owner['event']['metric_identity'], 'Wrong metric wrapper completion')
            b['wrapper_end'] = deepcopy(e)
            self.metrics[e['metric_identity']] = b
            self.metric_history.append(b)
            self.owner_history[owner['event']['route'], e['identity']] = owner['event']
        elif kind == 'DSD_REQUEST':
            dest, signature = self.output(e)
            require(type(e['cached']) is bool, 'Invalid tensor cache flag')
            require(e['transpose'] in ((0, 0), (1, 0)) and e['request_switch'] == 0,
                    'Unsupported tensor transpose/request switch')
            metric = self.metric_metadata(e)
            require(e['s_done'] is True, 'Tensor requires completed metric wrapper')
            rid = e['request_id']
            require(type(rid) is int and rid > 0 and rid not in self.request_ids, 'Duplicate/invalid tensor request ID')
            self.request_ids.add(rid)
            require(not self.pending and dest not in self.ready, 'Pending/unconsumed tensor request')
            require(not self.owners and not self.builds, 'Tensor during pending metric construction')
            require(not self.pending_coefficients, 'Tensor during pending coefficient generation')
            require(metric in self.metrics, 'Tensor request with unknown current metric')
            roles = ('OV', 'OV') if e['tensor_type'] == 'OVOV' else ('VV', 'OO')
            m, o, v = self.header[2:5]
            counts = dict(OV=o*v, VV=v*(v+1)//2, OO=o*(o+1)//2)
            for operand, role in zip((e['left'], e['right']), roles):
                self.identity(operand['identity'], 'coefficient_subset')
                require(operand['shape'] == (counts[role], m), 'Mapped operand dimensions')
                require(operand['identity'] in latest and latest[operand['identity']]['kind'] == role,
                        'Unknown/wrong current mapped subset')
                require(operand['fit'] == latest[operand['identity']]['observed_metadata']['fit'],
                        'Unexplained current stored parameters')
            require(dest[1] not in (metric[1], e['left']['identity'][1], e['right']['identity'][1]),
                    'Tensor output/input backing alias')
            require(metric[1] not in (e['left']['identity'][1], e['right']['identity'][1]), 'Metric/coefficient backing alias')
            require((e['left']['identity'] == e['right']['identity']) == (roles[0] == 'OV'), 'Wrong mapped operand alias')
            if e['cached']:
                require(dest in self.tensors and self.tensors[dest]['signature'] == signature, 'Unknown/wrong tensor cache')
                self.ready[dest] = dict(request=deepcopy(e), generation=self.tensors[dest],
                    current_metric=self.metrics[metric]['event']['header'][0],
                    current_subsets=tuple(latest[x['identity']]['start'] if x['identity'] in latest else None
                                          for x in (e['left'], e['right'])))
            else:
                self.tensors.pop(dest, None)
                require(metric in self.metrics, 'Tensor without successful metric generation')
                self.pending[dest] = dict(request=deepcopy(e), signature=signature,
                    left=self.subset(e['left'], roles[0], latest), right=self.subset(e['right'], roles[1], latest),
                    metric=deepcopy(self.metrics[metric]))
        elif kind == 'DSD_COMPLETE':
            dest, signature = self.output(e)
            require(dest in self.pending, 'Tensor completion without pending request')
            g = self.pending.pop(dest)
            require(e['request_id'] == g['request']['request_id'] and signature == g['signature'] and
                    e['info'] == 0 and e['done'] is True, 'Failed/wrong tensor completion')
            g['completion'] = deepcopy(e)
            self.tensors[dest] = g
            self.tensor_history.append(g)
            self.ready[dest] = dict(request=g['request'], generation=g,
                current_metric=g['metric']['event']['header'][0],
                current_subsets=(g['left']['start'], g['right']['start']))
        elif kind == 'HESSIAN_TENSOR':
            dest = self.identity(e['identity'])
            require(dest in self.ready and dest not in self.pending, 'Tensor consumer without ready request')
            use = self.ready.pop(dest)
            g = use['generation']
            require(self.tensors.get(dest) is g, 'Tensor overwritten before consumer')
            require((e['tensor_type'], e['description'], dest, e['matrix'].shape, e['mapping'][:4], self.slots) == g['signature']
                    and e['mapping'][4:] == (1, 0, 1, 0), 'Tensor consumer identity/shape/map mismatch')
            require(e['hessian'] in ('h1', 'h2'), 'Unknown Hessian consumer')
            calculated = g['left']['matrix'] @ (g['metric']['j']['matrix'] @ g['right']['matrix'].T)
            self.consumers.append(dict(event=deepcopy(e), request=use['request'], generation=g,
                                       current_metric=use['current_metric'], current_subsets=use['current_subsets'],
                                       reconstruction=error(calculated, e['matrix'])))

    def finish(self):
        require(self.checked and not self.owners and not self.builds and not self.pending and not self.ready,
                'Unfinished cache lifecycle')
        require(self.linked_js == set(self.js), 'Unlinked resident metric producer')
        require(self.metric_history and self.tensor_history and self.consumers, 'Missing cache provenance chain')
        def subset(g):
            return dict(start=g['start'], setter=g['metadata']['header'][0], fit=g['metadata']['fit'],
                        parent_solve_id=g['parent_solve']['begin']['fit_control'][0])
        consumers = []
        for use in self.consumers:
            g = use['generation']
            consumers.append(dict(consumer=use['event']['header'][0], tensor_type=use['event']['tensor_type'],
                request=use['request']['header'][0], cached=use['request']['cached'],
                generation=g['request']['header'][0], completion=g['completion']['header'][0],
                left=subset(g['left']), right=subset(g['right']),
                metric_generation=g['metric']['event']['header'][0],
                metric_completion=g['metric']['end']['header'][0],
                metric_owner=g['metric']['owner']['request_id'], metric_route=g['metric']['owner']['route'],
                producer_j=g['metric']['j']['header'][0], reconstruction=use['reconstruction'],
                request_context=dict(current_metric_generation=use['current_metric'],
                    current_subset_starts=use['current_subsets'],
                    left=use['request']['left'], right=use['request']['right'],
                    differs_from_original=(use['current_metric'] != g['metric']['event']['header'][0] or
                        use['current_subsets'] != (g['left']['start'], g['right']['start']) or
                        use['request']['left']['fit'] != g['left']['metadata']['fit'] or
                        use['request']['right']['fit'] != g['right']['metadata']['fit']))))
        return dict(complete_integral_cache_provenance=True,
                    structural_provenance_complete=True, observer_validated=False,
                    numerical_passed=all(c['reconstruction']['max_scaled'] <= 1e-9 for c in consumers),
                    scaled_tolerance=1e-9, consumers=consumers,
                    metric_generations=len(self.metric_history), tensor_generations=len(self.tensor_history),
                    limitations=['Reader structural checks do not authenticate an observer; require separate fresh-run evidence',
                                 'Bounded fixed-geometry monomer routes only; not universal storage mutation tracking'])
