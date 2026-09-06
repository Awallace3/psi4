"""Synthetic native-DF observer preparation checks, not integral/DF parity."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('native_capture', Path(__file__).parent/'data_isapol/oracle/capture_native_df.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def sources():
    smat = '''module df_Smat
use precision
contains
subroutine make_s_matrix_coulomb()
if (debug_matrices) call matout_types(S)
end subroutine make_s_matrix_coulomb
subroutine make_s_matrix_constraints()
call close_type(Sc,info,transpose=.false.,forcewrite=.true.,release=.false.,debug=debug2)
end subroutine make_s_matrix_constraints
end module df_Smat
'''
    monomer = '''module df_monomer_module
contains
  subroutine do_DFrho_monomer()
  implicit none
  if (same) then
  endif
    call write_type(Trho,info,release=.false.)
    call solve_df_equations(S%int,Trho,Drhotmp,info)
    Drho%vector(1:naux) = Drhotmp%matrix(1,1:naux)
  end subroutine do_DFrho_monomer
end module df_monomer_module
'''
    return smat, monomer, 'all:\n\ttrue\n'


def test_native_hooks_read_before_solver_and_after_assignment():
    smat, monomer, makefile = tool.instrument(*sources())
    assert smat.startswith('module isapol_native_df_capture')
    assert monomer.index('call ndc_rhs') < monomer.index('call solve_df_equations')
    assert monomer.index('Drho%vector(1:naux) =') < monomer.index('call ndc_solution')
    assert 'df_monomer.o: df_Smat.o' in makefile
    for mutation in ['call open_type(', 'call close_type(', 'call release_type(', 'call write_type(']:
        assert mutation not in tool.MODULE
    assert "status='new'" in tool.MODULE
    assert 'seen_j.and.seen_a' in tool.MODULE
    assert 'OCCUPATIONS_ASSUMED_BY_CLOSED_SHELL_ROUTINE' in tool.MODULE


@pytest.mark.parametrize('name', ['make_s_matrix_coulomb', 'make_s_matrix_constraints', 'do_DFrho_monomer'])
def test_native_preparation_fails_closed_on_missing_routine(name):
    texts = [s.replace(name, 'wrong_name') for s in sources()]
    with pytest.raises(ValueError, match='exactly one routine'):
        tool.instrument(*texts)


def test_native_module_import_anchor_ignores_routine_precision_imports():
    smat, monomer, makefile = sources()
    smat = smat.replace('subroutine make_s_matrix_coulomb()\n',
                        'subroutine make_s_matrix_coulomb()\nuse precision\n')
    patched, _, _ = tool.instrument(smat, monomer, makefile)
    assert patched.count('use isapol_native_df_capture, only: ndc_metric') == 1


def test_native_patch_rejects_duplicate_anchor():
    smat, monomer, makefile = sources()
    monomer = monomer.replace('  if (same) then', '  if (same) then\n  if (same) then')
    with pytest.raises(ValueError, match='one source anchor'):
        tool.instrument(smat, monomer, makefile)


def test_native_prepare_preserves_input_and_refuses_overwrite(tmp_path):
    source, dest = tmp_path/'source', tmp_path/'isolated'
    for rel,text in zip(['src/df_Smat.F90', 'src/df_monomer.F90', 'Makefile_body'], sources()):
        p=source/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)
    for rel in ['Makefile', 'VERSION', 'bin/version.py', 'x86-64/gfortran/exe/Flags']:
        p=source/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('fixture')
    before=(source/'src/df_monomer.F90').read_bytes()
    metadata=tool.prepare(source,dest)
    assert (source/'src/df_monomer.F90').read_bytes()==before
    assert metadata['source_sha256']['src/df_monomer.F90'] != metadata['instrumented_sha256']['src/df_monomer.F90']
    with pytest.raises(ValueError, match='new and outside'):
        tool.prepare(source,dest)
    with pytest.raises(ValueError, match='new and outside'):
        tool.prepare(source,source/'nested')


def native_fixture(tmp_path):
    spec = importlib.util.spec_from_file_location('native_df_replay', Path(tool.__file__).with_name('replay_native_df.py'))
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    text = ('ISAPOL_NATIVE_DF 2\n1 1 1 1 2\nsynthetic\ndalton\ndalton\n'
            'Sc__ _A_A T_MO ___A\n1 0 0 1\n1000 0 0\nMETRIC_COUNTS\n1 1\n')
    for role in ['MAIN', 'AUX']:
        text += (f'{role}_METADATA\nsynthetic basis\n{role}\nMC\n0\n'
                 f'{role}_BASIS\n1 1 1 1 1\nS\nHe\n2 0 0 0\n1 1\n1 0 1 1\n')
    text += ('C_OCC\n1 1\n1\nOCCUPATIONS_ASSUMED_BY_CLOSED_SHELL_ROUTINE\n1\n2\n'
             'Q\n1\n1\nRHS_CONSTRAINED\n1\n2004\nDRHO\n1\n2\nEND_NATIVE_DF\n')
    (tmp_path/'isapol-native-df-state.dat').write_text(text)
    for label,value in [('J', 2), ('A', 1002)]:
        (tmp_path/f'isapol-native-df-{label}-000001.dat').write_text(
            f'ISAPOL_NATIVE_METRIC 1\n1\n{label}\n1e-10 1e-18\nMETRIC\n1 1\n{value}\nEND_METRIC\n')
    return reader


def test_native_exported_equation_replay_synthetic(tmp_path):
    reader = native_fixture(tmp_path)
    report = reader.replay(tmp_path)
    assert report['passed'] and report['reference_solve_residual'] == 0
    assert report['fitted_electrons'] == 2


@pytest.mark.parametrize('old,new', [
    ('1 1 1 1 2', '1 1 1 1 3'), ('dalton', 'nwchem'),
    ('1000 0 0', '0 0 0'), ('Q\n1\n1', 'Q\n1\nnan'),
    ('END_NATIVE_DF\n', 'END_NATIVE_DF\ntrailing\n'),
])
def test_native_reader_rejects_invalid_state(tmp_path, old, new):
    reader = native_fixture(tmp_path)
    path = tmp_path/'isapol-native-df-state.dat'
    path.write_text(path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError): reader.replay(tmp_path)


def test_native_equation_mismatch_is_not_accepted(tmp_path):
    reader = native_fixture(tmp_path)
    path = tmp_path/'isapol-native-df-A-000001.dat'
    path.write_text(path.read_text().replace('1002', '1003'))
    assert not reader.replay(tmp_path)['passed']


@pytest.mark.parametrize('change', ['identical', 'different', 'missing'])
def test_repeated_metrics_must_be_complete_and_identical(tmp_path, change):
    reader = native_fixture(tmp_path)
    state = tmp_path/'isapol-native-df-state.dat'
    state.write_text(state.read_text().replace('METRIC_COUNTS\n1 1', 'METRIC_COUNTS\n2 1'))
    if change != 'missing':
        text = (tmp_path/'isapol-native-df-J-000001.dat').read_text()
        if change == 'different': text = text.replace('\n2\nEND_METRIC', '\n3\nEND_METRIC')
        (tmp_path/'isapol-native-df-J-000002.dat').write_text(text)
    if change == 'identical':
        assert reader.replay(tmp_path)['passed']
    else:
        with pytest.raises(ValueError): reader.replay(tmp_path)


def test_native_bad_source_does_not_create_destination(tmp_path):
    source, dest=tmp_path/'source',tmp_path/'isolated'
    (source/'src').mkdir(parents=True)
    for rel in ['src/df_Smat.F90', 'src/df_monomer.F90', 'Makefile_body']:
        (source/rel).write_text('invalid source')
    with pytest.raises(ValueError): tool.prepare(source,dest)
    assert not dest.exists()
