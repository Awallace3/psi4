"""Synthetic format tests for whole-sweep capture, not molecular controller parity."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

TOOLS = Path(__file__).parent/'data_isapol/oracle'
spec = importlib.util.spec_from_file_location('sweep_replay', TOOLS/'replay_isa_sweep.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def synthetic():
    def phase(name):
        text = name+'\n0 0 0 0\n'
        for a in range(1, 4):
            text += f'ATOM\n{a} 1\nH{a}\n{a} 0 0\n'
            for vector in ['D', 'D0', 'W', 'W0']:
                text += f'{vector}\n1\n1\n'
            text += 'CHARGES\n1 1 1\nFLAGS\n0 1 1\nTAIL\n1.5 2.5 1 2\n'
            text += 'SHAPE_NEIGHBOURS\n3\n1 2 3\n'
        return text
    return ('ISAPOL_SWEEP_STATE 1\n1 3\nCONFIG_FLOATS\n.17 .001 0 .2 1e-9 1e-5 1e-5 1e-5 0\n'
            'CONFIG_INTS\n1 1 120 20 20 1 0 1\n'+phase('PRE')+
            'DELTAS\n.1 .2 .3\nMAX_DELTA\n.3\nCALLS\n1 2 3\n'+phase('POST')+'END_SWEEP\n')


def read(tmp_path, text):
    p = tmp_path/'state.dat'
    p.write_text(text)
    return tool.read_state(p)


def test_sweep_state_roundtrip(tmp_path):
    s = read(tmp_path, synthetic())
    assert s['iteration'] == 1
    assert s['pre']['atoms'][2]['name'] == 'H3'
    np.testing.assert_array_equal(s['calls'], [1, 2, 3])
    np.testing.assert_array_equal(s['post']['atoms'][0]['vectors']['W0'], [1])


@pytest.mark.parametrize('old,new,match', [
    ('STATE 1', 'STATE 2', 'Expected'), ('1 3\nCONFIG', '1 2\nCONFIG', 'three-site'),
    ('1e-9 1e-5', '-1e-9 1e-5', 'settings'), ('120 20 20', '0 20 20', 'limits'),
    ('PRE\n0 0 0 0', 'PRE\n0 0 2 0', 'controls'),
    ('ATOM\n1 1', 'ATOM\n0 1', 'identity'), ('\nH1\n', '\n\n', 'Empty'),
    ('CHARGES\n1 1 1', 'CHARGES\n1 nan 1', 'Nonfinite'),
    ('FLAGS\n0 1 1', 'FLAGS\n0 2 1', 'flags'),
    ('TAIL\n1.5 2.5 1 2', 'TAIL\n1.5 2.5 1 0', 'defined tail'),
    ('SHAPE_NEIGHBOURS\n3\n1 2 3', 'SHAPE_NEIGHBOURS\n3\n1 1 3', 'neighbours'),
    ('DELTAS\n.1 .2 .3', 'DELTAS\n.1 -.2 .3', 'deltas'),
    ('CALLS\n1 2 3', 'CALLS\n1 3 4', 'associations'),
    ('END_SWEEP\n', '', 'Truncated'), ('END_SWEEP\n', 'END_SWEEP\njunk', 'Trailing'),
])
def test_sweep_state_rejects_malformed(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read(tmp_path, synthetic().replace(old, new))


def test_sweep_state_requires_committed_post_state(tmp_path):
    text = synthetic()
    pre, post = text.split('POST\n')
    with pytest.raises(ValueError, match='committed'):
        read(tmp_path, pre+'POST\n'+post.replace('W0\n1\n1', 'W0\n1\n2', 1))
    with pytest.raises(ValueError, match='identity'):
        read(tmp_path, pre+'POST\n'+post.replace('H1\n', 'changed\n', 1))


def test_capture_requires_live_controller_dispatch():
    spec = importlib.util.spec_from_file_location('sweep_capture', TOOLS/'capture_isa_sweep.py')
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    call = '  call Iterative_Stockholder_Atoms_restart(mol,0,4,.false.,info)\n'
    assert capture.controller_routine(call) == 'Iterative_Stockholder_Atoms_restart'
    for text in ['', call+call, call.replace('_restart', '')]:
        with pytest.raises(ValueError, match='dispatch'):
            capture.controller_routine(text)


def test_initial_legacy_isacharge_is_diagnostic_not_controller_input(tmp_path):
    text = synthetic().replace('CHARGES\n1 1 1', 'CHARGES\n0 0 NaN', 1)
    text = text.replace('FLAGS\n0 1 1', 'FLAGS\n0 0 -1', 1)
    s = read(tmp_path, text)
    atom = s['pre']['atoms'][0]
    assert atom['legacy_isacharge'] is None and atom['legacy_isacharge_token'] == 'NaN'
    np.testing.assert_array_equal(atom['charges'], [0, 0])
    assert atom['flags'].tolist() == [0, 0, -1]
    with pytest.raises(ValueError, match='defined tail'):
        read(tmp_path, text.replace('FLAGS\n0 0 -1', 'FLAGS\n0 1 -1', 1))
