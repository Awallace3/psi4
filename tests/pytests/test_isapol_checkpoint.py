"""Production capture FORMAT tests with synthetic streams, not molecular parity."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).parent / 'data_isapol' / 'oracle'


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load_tool('capture_isa_checkpoint')
checkpoint = load_tool('replay_isa_checkpoint')
water = load_tool('prepare_isa_water_run')
pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

# One normalized algebraic sample, deliberately NOT a physical Drho-C fixture.
STREAM = '''ISAPOL_CHECKPOINT 1
1 1 1 1
X
S
0 0 0
0 0 0 0.2
1 1
0.5
0 1 1 1
1 1
OVERLAP
1 1
2
METRIC
1 1
2
DENSITY
synthetic format test
1D-36
0 0
BATCH
1 1 1
0 0 0 1 2 1 1 1
RHS
1 1
2
POPULATION
2
COEFFICIENTS
1 1
1
END
'''


def read(tmp_path, text=STREAM):
    path = tmp_path / 'checkpoint.dat'
    path.write_text(text)
    return checkpoint.read_checkpoint(path)


def test_checkpoint_roundtrip_and_replay(tmp_path):
    c = read(tmp_path)
    assert c['batches'] == [dict(site=1, start=1, count=1)]
    assert c['shells'][0]['components'] == 1
    np.testing.assert_array_equal(c['basis_values'], [[1.]])
    assert c['options']['density_cutoff'] == 1e-36
    report = checkpoint.replay(c)
    assert all(e['max_absolute'] == 0 for e in report['errors'].values())
    assert report['relative_residual'] == 0


@pytest.mark.parametrize('old,new,match', [
    ('CHECKPOINT 1', 'CHECKPOINT 2', 'Expected'),
    ('1 1 1 1\nX', '1 1 0 1\nX', 'counts'),
    ('\nS\n', '\nQ\n', 'representation'),
    ('0 0 0 0.2', '0 0 nan 0.2', 'Nonfinite'),
    ('0 0 0 0.2', '0 -1 0 0.2', 'Negative'),
    ('1 1\n0.5', '2 1\n0.5', 'flags'),
    ('0 1 1 1', '0 1 1 2', 'contracted'),
    ('0 1 1 1', '1 1 1 1', 'shell'),
    ('0 1 1 1\n1 1', '0 1 1 1\n0 1', 'exponent'),
    ('OVERLAP\n1 1', 'OVERLAP\n2 1', 'dimensions'),
    ('BATCH\n1 1 1', 'BATCH\n0 1 1', 'batch'),
    ('1D-36\n0 0', '1D-36\n0 2', 'tail flags'),
    ('1D-36', '-1D-36', 'cutoff'),
    ('RHS\n1 1', 'RHS\n1 2', 'RHS dimensions'),
    ('\nEND\n', '\n', 'Truncated'),
    ('\nEND\n', '\nEND\njunk', 'trailing'),
    ('0 0 0 1 2 1 1 1', '0 0 0 1 2 1 1', 'numbers'),
])
def test_checkpoint_rejects_malformed(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read(tmp_path, STREAM.replace(old, new))


def version2_stream():
    def basis(label):
        return f'{label}\n1 1 1 1 1\nS\nX\n1 0 0 0\n1 1\n1 0 1 1\n'
    descriptors = ('DESCRIPTORS\n' + basis('ATOMIC_BASIS') + basis('DENSITY_BASIS')
                   + 'DENSITY_COEFFICIENTS\n1\n2\nDENSITY_NEIGHBOURS\n1\n1\n'
                   + basis('SHAPE_BASIS') + 'SHAPE_MAP\n1\n1\nSHAPE_OLD\n1\n1\n')
    return (STREAM.replace('CHECKPOINT 1', 'CHECKPOINT 2').replace('BATCH\n', descriptors + 'BATCH\n')
            .replace('END\n', 'SHAPE_NEW_RAW\n1\n1\nEND\n'))


def test_checkpoint_v2_descriptors(tmp_path):
    c = read(tmp_path, version2_stream())
    assert c['schema_version'] == 2
    d = c['descriptors']
    np.testing.assert_array_equal(d['density_coefficients'], [2])
    np.testing.assert_array_equal(d['density_basis']['centres'], [[0, 0, 0]])
    np.testing.assert_array_equal(d['atomic_basis']['shells'], [[1, 0, 1, 1]])
    np.testing.assert_array_equal(d['shape_map'], [1])
    np.testing.assert_array_equal(d['shape_new_raw'], [1])
    assert checkpoint.replay(c)['relative_residual'] == 0
    padded = read(tmp_path, version2_stream().replace('DENSITY_NEIGHBOURS\n1\n1',
                                                     'DENSITY_NEIGHBOURS\n3\n1 0 0'))
    np.testing.assert_array_equal(padded['descriptors']['density_neighbours'], [1, 0, 0])
    padded_map = read(tmp_path, version2_stream().replace('SHAPE_MAP\n1\n1', 'SHAPE_MAP\n3\n1 0 0'))
    np.testing.assert_array_equal(padded_map['descriptors']['shape_map'], [1])
    np.testing.assert_array_equal(padded_map['descriptors']['shape_map_storage'], [1, 0, 0])


@pytest.mark.parametrize('old,new,match', [
    ('DESCRIPTORS', 'MISSING', 'Expected'),
    ('ATOMIC_BASIS\n1 1 1 1 1', 'ATOMIC_BASIS\n0 1 1 1 1', 'counts'),
    ('DENSITY_COEFFICIENTS\n1', 'DENSITY_COEFFICIENTS\n2', 'size'),
    ('DENSITY_NEIGHBOURS\n1\n1', 'DENSITY_NEIGHBOURS\n1\n2', 'neighbour'),
    ('DENSITY_NEIGHBOURS\n1\n1', 'DENSITY_NEIGHBOURS\n3\n1 0 1', 'padding'),
    ('DENSITY_NEIGHBOURS\n1\n1', 'DENSITY_NEIGHBOURS\n1\n0', 'neighbour'),
    ('DENSITY_NEIGHBOURS\n1\n1', 'DENSITY_NEIGHBOURS\n1\n-1', 'neighbour'),
    ('SHAPE_MAP\n1\n1', 'SHAPE_MAP\n1\n2', 'shape shell map'),
    ('SHAPE_MAP\n1\n1', 'SHAPE_MAP\n3\n1 1 0', 'padding'),
    ('SHAPE_NEW_RAW\n1\n1\n', '', 'Expected'),
    ('1 0 1 1\n', '2 0 1 1\n', 'shell index'),
    ('1 0 1 1\n', '1 0 2 1\n', 'shell index'),
])
def test_checkpoint_v2_rejects_malformed(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read(tmp_path, version2_stream().replace(old, new))


def test_checkpoint_multiple_batches_preserve_order(tmp_path):
    sample = 'BATCH\n1 1 1\n0 0 0 1 2 1 1 1\n'
    second = 'BATCH\n2 7 1\n1 0 0 0.5 -1 -2 -3 4\n'
    c = read(tmp_path, STREAM.replace(sample, sample + second))
    np.testing.assert_array_equal(c['points'], [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(c['density'], [2, -1])
    assert c['batches'][1] == dict(site=2, start=7, count=1)


def test_source_patching_is_fail_closed():
    with pytest.raises(ValueError, match='source anchor'):
        capture.replace_once('abc abc', 'abc', 'def')
    with pytest.raises(ValueError, match='source anchor'):
        capture.replace_once('abc', 'missing', 'def')
    with pytest.raises(ValueError, match='subroutine'):
        capture.patch_routine('', 'foo', [])
    original = 'subroutine foo(x)\nimplicit none\nend subroutine foo\n'
    assert 'use bar' in capture.patch_routine(original, 'foo', [('implicit none', 'use bar\nimplicit none')])
    with pytest.raises(ValueError, match='already instrumented'):
        capture.instrument('module isapol_checkpoint_capture', '')


def test_water_preparation_is_explicit_and_preserves_sources(tmp_path):
    archive, source = tmp_path / 'archive', tmp_path / 'source'
    archive.mkdir()
    source.mkdir()
    sites = ('O 8.0 0 0 0 TYPE O\nH1 1.0 -1.45365196 0 -1.12168732 TYPE H\n'
             'H2 1.0 1.45365196 0 -1.12168732 TYPE H\n')
    original = (sites * 2 + '#include H2O-A.basis\n#include-camcasp aux\nAngular 200\nRadial  100\n'
                'SET Lattice\nunused response\nSet Num-Int-Pars\nBegin ISA\nDF = Doo-C\n'
                'Solver BVLS\nISA-Algorithm A\nW-TAILS\nFIX = ON\nEnd\nEnd\nBEGIN Polarizability\nunused downstream\n')
    (archive / 'H2O.cks').write_text(original)
    (archive / 'H2O-A.basis').write_text('main basis\n')
    (archive / 'H2O-A-asc.movecs').write_text('orbitals\n')
    (source / 'aux').write_text('aux basis\n')
    out = tmp_path / 'run'
    metadata = water.prepare(archive, source, out)
    expanded = (out / 'H2O-expanded.cks').read_text()
    assert 'DF = Drho-C' in expanded and 'Solver LU' in expanded
    assert 'BVLS' not in expanded and 'Doo-C' not in expanded
    assert '#include' not in expanded and 'main basis' in expanded and 'aux basis' in expanded
    assert 'unused' not in expanded
    assert 'Set Num-Int-Pars' not in expanded and 'FIX = ON' not in expanded
    assert metadata['settings']['tail_fix'].startswith('source default ON')
    assert metadata['protocol'] == 'adapted-archive-Drho-C-LU'
    assert (archive / 'H2O.cks').read_text() == original
    assert len(metadata['source_sha256']) == 4
    with pytest.raises(ValueError, match='Run directory'):
        water.prepare(archive, source, out)
    (archive / 'H2O.cks').write_text(original.replace('-1.45365196', '-1.4'))
    with pytest.raises(ValueError, match='geometry'):
        water.prepare(archive, source, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_source_preparation_copies_build_helper(tmp_path, monkeypatch):
    source, destination = tmp_path / 'upstream', tmp_path / 'scratch'
    for name in ('src/num_integrals.F90', 'src/stockholder.F90', 'Makefile', 'Makefile_body',
                 'VERSION', 'bin/version.py', 'x86-64/gfortran/exe/Flags'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + '\n')
    monkeypatch.setattr(capture, 'instrument', lambda num, stock: ('patched num', 'patched stock'))
    metadata = capture.prepare(source, destination)
    assert (destination / 'bin/version.py').read_text() == 'bin/version.py\n'
    assert (destination / 'src/num_integrals.F90').read_text() == 'patched num'
    assert (source / 'src/num_integrals.F90').read_text() == 'src/num_integrals.F90\n'
    assert metadata['source_sha256'] != metadata['instrumented_sha256']


def test_prepare_refuses_existing_destination(tmp_path):
    with pytest.raises(ValueError, match='Destination'):
        capture.prepare(tmp_path, tmp_path)
    with pytest.raises(ValueError, match='Destination'):
        capture.prepare(tmp_path, tmp_path / 'child')


def test_provider_reconstruction_rejects_v1(tmp_path):
    with pytest.raises(ValueError, match='v2 descriptors'):
        checkpoint.replay(read(tmp_path), reconstruct_providers=True)


def test_provider_reconstruction_v2_algebraic_replay(tmp_path):
    # One unnormalized exp(-r^2) at the origin; analytic integral, not molecular evidence.
    c = read(tmp_path, version2_stream())
    overlap = (np.pi/2.)**1.5
    c['overlap'][:] = c['metric'][:] = overlap
    c['coefficients'][:] = 2./overlap
    c['descriptors']['shape_new_raw'][:] = 2./overlap
    report = checkpoint.replay(c, reconstruct_providers=True)
    assert report['evidence_class'] == 'full exported-input provider reconstruction with supplied shape samples'
    assert 'captured shapes' in report['limitations']
    assert all(e['max_absolute'] < 2e-15 for e in report['errors'].values())
    assert 'provider_raw_shape' in report['errors']
    assert report['relative_residual'] < 1e-15
    c['descriptors']['atomic_basis']['nfunction'] = 2
    with pytest.raises(ValueError, match='function count'):
        checkpoint.replay(c, reconstruct_providers=True)
