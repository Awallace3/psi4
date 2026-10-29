# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Synthetic contract tests only; never a local benzene quantum calculation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from psi4.driver.procrouting import isapol_benzene_acceptance as acceptance
from psi4.driver.procrouting.isapol_recipes import benzene_declared_variables
from psi4.driver.procrouting.isapol_refine import RefinementSite
from psi4.driver.procrouting.isapol_bounded import refinement_models


def test_compute_guard_refuses_before_state_or_output(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Phoenix compute allocation'):
        acceptance.run_benzene(tmp_path/'uncreated', {}, {}, geometry_sha256='0'*64)
    assert not (tmp_path/'uncreated').exists()


def test_forged_slurm_environment_on_ordinary_hostname_is_refused(monkeypatch):
    monkeypatch.setattr(acceptance.socket, 'gethostname', lambda: 'cos-workstation')
    for name, value in dict(SLURM_JOB_ID='123', SLURMD_NODENAME='cos-workstation',
                            USER='awallace43', HOME='/storage/home/awallace43',
                            SLURM_CLUSTER_NAME='phoenix-slurm').items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match='Phoenix compute allocation'):
        acceptance._compute_allocation()


def test_attributed_geometry_is_data_not_historical_molecular_products():
    record = dict(schema='native-benzene-geometry/1', units='bohr',
                  symbols=['C']*6+['H']*6, charge=0, multiplicity=1,
                  source='synthetic contract test; not a physical input',
                  source_sha256='0'*64, geometry=np.zeros((12, 3)).tolist())
    assert acceptance._geometry(record).shape == (12, 3)
    for changes in (dict(units='angstrom'), dict(source=''), dict(charge=1),
                    dict(symbols=['H']*12), dict(geometry=[[float('nan')]*3]*12)):
        with pytest.raises(ValueError):
            acceptance._geometry(dict(record, **changes))


def test_full72_declaration_and_static_reference_inventory_crossings():
    declared = benzene_declared_variables()
    assert len(declared) == len(set(declared)) == 72
    assert declared[0] == 'C1_10_10_A' and declared[36] == 'H1_10_10_A'
    assert all('_00_' not in label for label in declared)
    sites = tuple(
        RefinementSite(f'{kind}{i+1}', kind, (float(offset+i), 0., 0.), np.eye(3), 2)
        for kind, offset in (('C', 0), ('H', 6)) for i in range(6))
    first = [np.pad(np.full((8, 8), .02), ((1, 0), (1, 0))) for _ in sites]
    first[6][3, 5] = first[6][5, 3] = 4e-5
    nodes = []
    for node in range(11):
        tensors = [a.copy() for a in first]
        if node:
            tensors[0][1, 1] = 3e-5  # downward crossing remains penalized
            tensors[6][3, 5] = tensors[6][5, 3] = 2e-4  # upward remains free
        nodes.append((node*.03, tensors))
    models = refinement_models(
        sites, nodes, cutoff=1e-4, weight_type=3, weight_coefficient=.001,
        declared_variables=declared)
    from psi4 import core
    from psi4.driver.procrouting.isapol_lw import ArraySnapshot
    local = []
    for _, tensors in nodes:
        packed = np.zeros((1, 12, 15, 15))
        packed[0, :, :8, :8] = np.asarray(tensors)[:, 1:, 1:]
        local.append(ArraySnapshot.of(packed))
    refinements = tuple(SimpleNamespace(
        model=m, parameters=(0.,)*72, refined_tensors=np.zeros((12, 9, 9)),
        status='synthetic; no solve', diagnostics=core.IsaPfitDiagnostics(),
        anchor_shift_maxabs=.02) for m in models)
    result = SimpleNamespace(refinements=refinements, local_tensors=tuple(local),
                             frequencies=tuple(f for f, _ in nodes))
    inventory = acceptance._model_inventory(result, declared, expected_sites=sites)
    assert len(inventory) == 11
    assert all('H1_11s_21c_A' in m['unpenalized_variables'] for m in inventory)
    assert len({m['selection_sha256'] for m in inventory}) == 1
    assert all(len(m['labels']) == 72 for m in inventory)
    assert all(len(entries) == 6 for entries in inventory[0]['entries'])
    broken = replace(models[-1], static_reference_sha256='f'*64)
    with pytest.raises(RuntimeError, match='inventory'):
        acceptance._model_inventory(
            SimpleNamespace(**dict(vars(result), refinements=(*refinements[:-1],
                SimpleNamespace(**dict(vars(refinements[-1]), model=broken))))),
            declared, expected_sites=sites)
    with pytest.raises(RuntimeError, match='72-variable'):
        acceptance._model_inventory(result, declared[:-1], expected_sites=sites)
    mutations = [
        dict(parameter_entries=tuple((entries[0],) for entries in models[0].parameter_entries)),
        dict(equivalent_sites=((0,), (6,))),
        dict(reference_sites=(1, 7)),
        dict(declared_variables=declared[::-1]),
        dict(strengths=(0.,)*72),
        dict(anchors=(0.,)*72),
        dict(unpenalized_variables=()),
        dict(selection_sha256='e'*64),
        dict(frequency_au=.123),
    ]
    # Consistent corruption across ALL nodes must not pass by self-consistency.
    for changes in mutations:
        changed = tuple(SimpleNamespace(**dict(vars(r), model=replace(r.model, **changes)))
                        for r in refinements)
        with pytest.raises(RuntimeError):
            acceptance._model_inventory(
                SimpleNamespace(**dict(vars(result), refinements=changed)),
                declared, expected_sites=sites)
    charged = np.zeros((12, 9, 9))
    charged[0, 0, 0] = 1.
    bad = SimpleNamespace(**dict(vars(refinements[0]), refined_tensors=charged))
    with pytest.raises(RuntimeError, match='tensor inventory'):
        acceptance._model_inventory(
            SimpleNamespace(**dict(vars(result), refinements=(bad, *refinements[1:]))),
            declared, expected_sites=sites)


def test_numeric12site_export_reuses_same_native_layout(tmp_path):
    from psi4.driver.procrouting.isapol_acceptance import _write_tensor_artifact
    from psi4.driver.procrouting.isapol_lw import ArraySnapshot
    local = np.zeros((11, 12, 15, 15))
    refined = np.zeros((11, 12, 9, 9))
    result = SimpleNamespace(
        frequencies=tuple(float(i) for i in range(11)),
        local_tensors=tuple(ArraySnapshot.of(a[None]) for a in local),
        refined_tensors=refined,
        refinements=(SimpleNamespace(model=SimpleNamespace(sites=(None,)*12)),)*11)
    layout = _write_tensor_artifact(tmp_path/'benzene-synthetic.npz', result,
                                    localization_rank_limit=2)
    assert layout['local_shape'] == (11, 12, 15, 15)
    assert layout['refined_shape'] == (11, 12, 9, 9)
    with np.load(tmp_path/'benzene-synthetic.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['refined'], refined)
