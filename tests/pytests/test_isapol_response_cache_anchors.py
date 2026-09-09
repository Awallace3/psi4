"""Pinned source and inserted schema8 hooks; not runtime acceptance evidence."""
from pathlib import Path
import re
import subprocess
import pytest

SOURCE = Path('/home/awallace43/gits/CamCASP')
PIN = '63b16a22b9bae597fe81ecdb8b8d91c21868c814'


@pytest.fixture(scope='module')
def source():
    if not (SOURCE/'src/df_integrals.F90').is_file():
        pytest.skip('Optional pinned reference source not present')
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    return {name: (SOURCE/'src'/name).read_text() for name in
            ('df_integrals.F90', 'df_integral_utilities.F90', 'df_Smat.F90', 'df_integrals_for_df.F90')}


def routine(source, name):
    found = re.findall(r'(?ims)^\s*subroutine\s+'+name+r'\(.*?^\s*end subroutine '+name+r'\b', source)
    assert len(found) == 1
    return found[0]


def test_dsd_entry_completion_and_released_operands(source):
    text = routine(source['df_integrals.F90'], 'make_D_S_D')
    assert text.count('if (I%done) return') == 1
    assert text.count('I%done = .true.') == 1
    assert text.index('if (I%done) return') < text.index('call matmult_types(S%int,Db,TMP')
    assert text.index('call matmult_types(S%int,Db,TMP') < text.index('call matmult_types(Da,TMP,I%int')
    for operand in ('S%int', 'Da', 'Db', 'I%int'):
        assert text.index('call release_type('+operand+')') < text.index('I%done = .true.')
    assert text.index('call destroy_type(TMP)') < text.index('I%done = .true.')
    assert 'if (present(transpose_S)) then' in text


@pytest.mark.parametrize('side', ['A', 'B'])
def test_actual_ordinary_mapped_slots_not_constrained_response(source, side):
    text = source['df_integrals.F90']
    suffix = side.lower()
    for role in ('Dov', 'Dvv', 'Doo'):
        assert f'{role}_{suffix} => {role}(indx{side})' in text
    assert f'call make_D_S_D(Iovov_{suffix*4},Dov_{suffix},S_{suffix}_{suffix},Dov_{suffix},info)' in text
    assert f'call make_D_S_D(Ivvoo_{suffix*4},Dvv_{suffix},S_{suffix}_{suffix},Doo_{suffix},info)' in text
    assert f'call df_make_s_matrix(mol{side},S_{suffix}_{suffix},S_file(indx{side}),info)' in source['df_integrals_for_df.F90']


def test_both_metric_wrapper_owners_reach_common_constructor(source):
    util = source['df_integral_utilities.F90']
    ordinary = routine(util, 'df_make_s_matrix')
    constrained = routine(util, 'df_make_s_matrix_constraints')
    assert ordinary.index('if (S%done) return') < ordinary.index('call make_s_matrix(') < ordinary.index('S%done = .true.')
    assert constrained.index('if (Sc%done) return') < constrained.index('call make_s_matrix_constraints(')
    assert 'recalculate=.true.,info=info)' in ordinary and 'recalculate=.true.,info=info)' in constrained
    inner = routine(source['df_Smat.F90'], 'make_s_matrix_constraints')
    assert inner.index('call make_s_matrix(mol,S,S%filename,recalculate,info)') < inner.index('Sc = S')
    common = routine(source['df_Smat.F90'], 'make_s_matrix')
    assert common.index('select case(DFNormIndx)') < common.index('call make_s_matrix_coulomb(')
    assert 'logical, intent(in), optional :: recalculate' in common
    j = routine(source['df_Smat.F90'], 'make_s_matrix_coulomb')
    assert j.index('if (debug_matrices) call matout_types(S)') < j.index('call write_type(S,info,transpose=.false.,forcewrite=.true.')


def test_dimer_and_rotation_bypasses_must_get_fail_closed_guards(source):
    text = source['df_integrals_for_df.F90']
    start = text.index('if (.not.(S_a_a%done.and.S_b_b%done.and.S_a_b%done.and.S_a_ab%done.and.S_b_ab%done)) then')
    assert start < text.index('call fill_Smat_subsets(dimAB,S_ab_ab%int,S_a_a%int,S_b_b%int,') < text.index('S_a_a%done = .true.')
    for name, call in [('rotate_twoe2indx', 'rotate_2indx_int'), ('rotate_twoe2indx_left', 'rotate_left_indx')]:
        body = routine(source['df_integral_utilities.F90'], name)
        assert 'I%ForceRotate' in body
        assert body.index('if (.not.rotate) return') < body.index('call '+call+'(')
    body = routine(source['df_integral_utilities.F90'], 'check_if_integrals_are_done')
    assert body.index('same_df_norm   = (df_DFNormIndx==int_par%df_DFNormIndx)') < body.index('S_a_a%done  = .false.')


@pytest.fixture(scope='module')
def inserted(source):
    import importlib.util
    path = Path(__file__).parent/'data_isapol/oracle/capture_response.py'
    spec = importlib.util.spec_from_file_location('cache_emitter_test', path)
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    paths = ['src/df_Smat.F90','src/densfit_prop.F90','src/prop_utilities.F90','src/polarizability.F90',
             'Makefile_body','src/df_data.f90','src/df_utilities.F90','src/df_monomer.F90',
             'src/df_integrals.F90','src/molecular_orbitals.F90','src/matrix_operations_types.F90',
             'src/num_integrals.F90','src/function_expansion_operations.F90',
             'src/df_integral_utilities.F90','src/df_integrals_for_df.F90','src/df_int_operations.F90']
    original = [(SOURCE/p).read_text() for p in paths]
    result = capture.instrument(*original)
    assert len(result) == len(paths)
    return capture, dict(zip(paths, result))


def test_coefficient_display_names_are_not_unique_storage_keys(source):
    monomer = (SOURCE/'src/df_monomer.F90').read_text()
    definition = routine(monomer, 'define_D_matrix')
    assert "typename = 'D  Full DF sol for '//trim(mol%name)" in definition
    assert 'call renamefile_type(D,Dfile,info)' in definition
    for side in ('A','B'):
        assert f'D(indx{side}),D_file(indx{side}),D_par(indx{side})' in monomer
        assert f'D_c(indx{side}),D_c_file(indx{side}),D_c_par(indx{side})' in monomer
    util = (SOURCE/'src/df_utilities.F90').read_text()
    for role, routine_name in [('OO','fill_Aoo'),('OV','fill_Aov'),('VV','fill_Avv')]:
        assert f"mat_name = '{role} part of '//trim(Afull%name)" in routine(util, routine_name)


def test_inserted_metric_lifecycle_and_raw_result(inserted):
    capture, files = inserted
    for name, owner in [('df_make_s_matrix','S'), ('df_make_s_matrix_constraints','Sc')]:
        body = routine(files['src/df_integral_utilities.F90'], name)
        assert body.index('call nrc_metric_request(') < body.index(f'if ({owner}%done) return')
        assert body.index(f'{owner}%done = .true.') < body.rindex('call nrc_metric_end(')
        assert 'if (info.lt.0) then; info = -1; return; endif' in body
    body = routine(files['src/df_Smat.F90'], 'make_s_matrix')
    assert body.index('call nrc_metric_begin(') < body.index('select case(DFNormIndx)')
    assert body.index('if (present(recalculate)) then') < body.index('nrc_rv=merge(1,0,recalculate)')
    assert body.index('call make_s_matrix_coulomb(') < body.index('call nrc_metric_result(')
    assert body.index('call nrc_metric_result(') < body.index("call check_info(info,'make_s_matrix_coulomb'")
    body = routine(files['src/df_Smat.F90'], 'make_s_matrix_coulomb')
    assert body.count('call nrc_metric_link(') == 1
    assert body.index("call nrc_matrix(mol,'J'") < body.index('call nrc_metric_link(')
    assert body.index('call nrc_metric_link(') < body.index('call write_type(S,info,transpose=.false.,forcewrite=.true.')


def test_inserted_dsd_observes_actual_operands_before_cache_return(inserted):
    _, files = inserted
    body = routine(files['src/df_integrals.F90'], 'make_D_S_D')
    assert body.rindex('call nrc_dsd_request(') < body.index('if (I%done) return')
    assert body.index('I%done = .true.') < body.index('call nrc_dsd_complete(')
    for side in ('A','B'):
        for role in ('Dov','Dvv','Doo'):
            for field in ('DFNormIndx','option','df_type','done','lambda','eta','gamma','gamma_DeltaZ'):
                assert f'{role}_par(indx{side})%{field}' in body
    assert 'Dov_c_par' not in body
    for side in ('A','B'):
        assert body.index(f'if (nrc_selected(mol{side}%indx)) then') < body.index(f'Dov_par(indx{side})%defined')
    assert 'if (nrc_rid>0) then' in body
    assert body.index('if (present(transpose_S)) then') < body.index('nrc_tv=merge(1,0,transpose_S)')
    original = routine((SOURCE/'src/df_integrals.F90').read_text(), 'make_D_S_D')
    for line in original.splitlines():
        if any(token in line for token in ['call matmult_types(', 'call release_type(', 'call destroy_type(',
                                          'if (info<0)', 'if (I%done) return', 'I%done = .true.']):
            assert line in body


def test_inserted_guards_dependencies_and_line_lengths(inserted):
    capture, files = inserted
    body = routine(files['src/df_integral_utilities.F90'], 'check_if_integrals_are_done')
    assert body.index('nrc_initialized=int_par%defined') < body.index('call create_type(int_par,')
    assert body.index('same_df_norm   =') < body.index('call nrc_cache_check(') < body.index('S_a_a%done  = .false.')
    normalized = re.sub(r'[\s&]', '', body)
    assert 'nrc_initialized,sameA' in normalized and 'nrc_initialized,sameB' in normalized
    body = routine(files['src/df_integrals_for_df.F90'], 'make_integrals_for_df')
    assert body.index('unobserved dimer metric subset write') < body.index('call fill_Smat_subsets(')
    body = routine(files['src/df_int_operations.F90'], 'open_df_int_2e')
    assert body.index('generalized/nonzero-switch') < body.index('I%generalized = .false.')
    for name, mols in [('rotate_onee2indx', ('molA','molB')), ('rotate_oneeint', ('mol',)),
                       ('rotate_twoe2indx', ('molA','molB')), ('rotate_twoe2indx_left', ('mol',))]:
        body = routine(files['src/df_integral_utilities.F90'], name)
        for mol in mols:
            assert capture.cache_guard(mol, 'rotation/ForceRotate', 'rotate') in body
        assert body.index('call nrc_guard(') < body.index('if (.not.rotate) return')
    for stem in ('df_integral_utilities','df_integrals_for_df','df_int_operations'):
        assert f'{stem}.o {stem}.mod: isapol_response_capture.mod' in files['Makefile_body']
    for path, text in files.items():
        if path == 'Makefile_body':
            continue
        old = set((SOURCE/path).read_text().splitlines())
        assert all(len(line) <= 132 for line in text.splitlines() if line not in old)
    declarations = capture.MODULE.split('contains', 1)[0]
    assert not re.search(r'type\s*\(|pointer|allocatable', declarations, re.I)


def test_new_scalar_observers_return_before_selection_side_effects(inserted):
    capture, _ = inserted
    names = ('nrc_guard','nrc_cache_check','nrc_metric_request','nrc_metric_end','nrc_metric_begin',
             'nrc_metric_link','nrc_metric_result','nrc_dsd_request','nrc_dsd_complete')
    for name in names:
        body = capture.MODULE.split('subroutine '+name+'(', 1)[1].split('end subroutine', 1)[0]
        assert '%matrix' not in body and '%vector' not in body and 'call resident(' not in body
        assert body.index('if (.not.selected(header(1))) return') < body.index('call scalar_event(')
        assert not re.search(r'\btype\s*\(|pointer|allocatable', body, re.I)
        assert not re.search(r'\bpresent\([^)]*\)\s*\.and\.|merge\([^\n]*present\(', body, re.I)
    for name in ('cache_metric','cache_output','cache_operand'):
        body = capture.MODULE.split('subroutine '+name+'(', 1)[1].split('end subroutine', 1)[0]
        assert not re.search(r'\btype\s*\(|pointer|allocatable|%matrix|%vector', body, re.I)
