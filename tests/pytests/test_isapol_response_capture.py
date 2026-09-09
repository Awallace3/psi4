"""Observer safety/patching tests; no reference execution or parity claim."""
import importlib.util
from pathlib import Path
import re
import pytest

path = Path(__file__).parent/'data_isapol/oracle/capture_response.py'
spec = importlib.util.spec_from_file_location('response_capture_test', path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def routine(name):
    return capture.MODULE.split('subroutine '+name+'(', 1)[1].split('end subroutine', 1)[0]


def test_response_observer_has_no_reference_object_io():
    assert not re.search(r'\bcall\s+(?:open_type\w*|close_type\w*|release_type\w*|reclaim_type\w*|'
                         r'write_type\w*|matrix_read\w*|matrix_write\w*)\s*\(', capture.MODULE, re.I)
    assert "status='new'" in capture.MODULE
    assert 'ISAPOL_RESPONSE_EVENT 8' in capture.MODULE
    assert max(map(len, capture.MODULE.splitlines())) <= 132


def test_projection_reads_only_metadata():
    body = routine('nrc_projection')
    assert 'D%matrix' not in body and 'Ker%matrix' not in body
    assert 'call resident(' not in body
    assert 'call cp_fit_parameters(par)' in body
    assert 'trim(D%filename)' in body and 'trim(Ker%filename)' in body


def test_orbitals_and_diagonal_use_are_separate():
    assert "call resident(mol%C,'ORBITALS C')" in routine('nrc_orbitals')
    assert 'mol%C' not in routine('nrc_energies')
    assert "cp_vector('DIAGONAL_FRACTION',(/frac/))" in routine('nrc_energies')
    assert "cp_vector('ROW',parent%matrix(1,:))" in routine('nrc_ov_row')
    assert 'dest%matrix' not in routine('nrc_ov_row')


def test_nested_routine_patch_requires_unique_anchor():
    text = '''subroutine outer(mol)
implicit none
type(molecule), intent(inout) :: mol
contains
subroutine inner()
implicit none
end subroutine inner
end subroutine outer
'''
    with pytest.raises(ValueError):
        capture.scaffold.patch_routine(text, 'outer', [('implicit none', 'use observer\nimplicit none')])
    anchor = 'implicit none\ntype(molecule), intent(inout) :: mol'
    patched = capture.scaffold.patch_routine(text, 'outer', [(anchor, 'use observer\n'+anchor)])
    assert patched.count('use observer') == 1
    assert 'subroutine inner()\nimplicit none' in patched


@pytest.mark.parametrize('kind', ['same', 'existing', 'child'])
def test_response_preparation_refuses_unsafe_destination(tmp_path, kind, monkeypatch):
    source = tmp_path/'source'; source.mkdir()
    destination = {'same': source, 'existing': tmp_path/'existing', 'child': source/'child'}[kind]
    if kind == 'existing': destination.mkdir()
    def unexpected(*args, **kwargs):
        raise AssertionError('Git/source access before destination guard')
    monkeypatch.setattr(capture.subprocess, 'check_output', unexpected)
    with pytest.raises(ValueError, match='Destination'):
        capture.prepare(source, destination)


def test_fit_observer_owns_only_metadata_and_reads_valid_rhs_slice():
    declarations=capture.MODULE.split('contains',1)[0].lower()
    assert 'type(molecule)' not in declarations and 'type(real_matrix)' not in declarations
    assert 'pointer' not in declarations
    block=routine('nrc_fit_block')
    assert 'ieee_is_finite(B%matrix(:,1:width))' in block
    assert "cp_matrix('MATRIX',B%matrix(:,1:width))" in block
    forbidden=r'^\s*call\s+resident\s*\(\s*B\s*[,)]'
    assert not re.search(forbidden,block,re.I|re.M)
    assert re.search(forbidden,"  call resident(B,'RHS')",re.I|re.M)
    assert not re.search(forbidden,"! Do not call resident(B)",re.I|re.M)
    assert 'first==fit_next' in block
    assert 'fit_active=.false.' in routine('nrc_fit_end')


def test_paired_rows_and_density_observe_only_live_inputs():
    assert 'dest%matrix' not in routine('nrc_pair_row')
    assert "cp_vector('ROW',parent%matrix(1,:))" in routine('nrc_pair_row')
    assert "cp_vector('COEFFICIENTS',Rho%D%vector)" in routine('nrc_density')
    assert 'Rho%DFpar' not in routine('nrc_density')
    assert 'if (.not.density_active) return' in routine('nrc_density')
    for name in ('nrc_density_begin','nrc_density_end'):
        assert '%matrix' not in routine(name) and '%vector' not in routine(name)
    assert 'density_serial' in routine('nrc_kernel')


def test_parent_solve_hooks_are_metadata_only():
    for name in ('nrc_solve_begin','nrc_solve_end'):
        body=routine(name)
        assert '%matrix' not in body and '%vector' not in body
        assert 'call resident(' not in body
    assert 'par%lambda==0.0_dp.or.par%lambda==1.0_dp' in routine('nrc_solve_begin')
    assert "require(info==0,'failed parent solve')" in routine('nrc_solve_end')


@pytest.fixture(scope='module')
def scalar_guard_executable(tmp_path_factory):
    """Compile exact scalar observer routines, without any reference types/runtime."""
    import shutil
    import subprocess
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('Optional scalar Fortran guard execution requires gfortran')
    folder = tmp_path_factory.mktemp('response-cache-scalar-driver')
    routines = capture.MODULE.split('! New lifecycle APIs', 1)[1].split('subroutine require(', 1)[0]
    routines = '! New lifecycle APIs'+routines
    for name in ('require','scalar_event','end_event','cp_vector'):
        routines += 'subroutine '+name+'('+routine(name)+'end subroutine\n'
    routines += 'logical function selected(index)'+capture.MODULE.split(
        'logical function selected(index)', 1)[1].split('end function', 1)[0]+'end function\n'
    source = '''module scalar_cache_test
use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
implicit none
integer, parameter :: dp=kind(1.0d0)
integer :: cp_unit,serial=0,metric_id=0,build_id=0,dsd_id=0
logical :: metric_active=.false.,build_active=.false.
character(*), parameter :: cp_fmt='(*(es26.17e3,1x))'
contains
'''+routines+'end module\n'+'''program driver
use scalar_cache_test
implicit none
integer :: fault,rid,tv,sw,mapping(4),flags(5),mflags(4)
character(20) :: tf,ln,rn
character(20) :: arg
call get_command_argument(1,arg)
read(arg,*) fault
mapping=(/2,2,2,2/)
flags=(/1,0,0,0,0/)
mflags=(/1,0,0,0/)
tf='tensor'
ln='OV'
rn='OV'
tv=0
sw=0
select case(fault)
case(1)
  tf='metric'
case(2)
  mapping(1)=1
case(3)
  tv=1
case(4)
  flags(2)=1
case(5)
  mflags(3)=1
case(6)
  sw=2
case(7)
  rn='aliased'
end select
call nrc_dsd_request((/7,4,2,2,2,4/),'Water','dalton',7,9, &
  'OVOV','AAAA','tensor',tf,(/4,4/),mapping,flags,.false.,.true., &
  'S___','_A_A','metric','metric',(/2,2/),mflags, &
  ln,'ov',(/4,2/),(/1,0,1,1/),(/0.0_dp,0.0_dp,0.0_dp,0.0_dp/),(/0,0/), &
  rn,'ov',(/4,2/),(/1,0,1,1/),(/0.0_dp,0.0_dp,0.0_dp,0.0_dp/),(/0,0/), &
  sw,1,1.0e-12_dp,1.0e-18_dp,rid,0,tv)
print *,rid
end program
'''
    assert not re.search(r'\btype\s*\(|pointer|allocatable|%matrix|%vector', source, re.I)
    assert max(map(len, source.splitlines())) <= 132
    file = folder/'driver.F90'
    file.write_text(source)
    exe = folder/'driver'
    flags = ('-O2 -DG77 -DCADPAC -DGAMESS -DSAPT2002 -DSIGNED_INTEGER -DERF -DF2003 '
             '-fno-backslash -fimplicit-none -fallow-argument-mismatch -ffp-contract=off').split()
    completed = subprocess.run([compiler]+flags+[str(file),'-o',str(exe)],cwd=folder,capture_output=True,text=True)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    return exe


@pytest.mark.parametrize('enabled', [False,True])
@pytest.mark.parametrize('fault', range(8))
def test_scalar_fortran_guards_and_disabled_selection(scalar_guard_executable, tmp_path, enabled, fault):
    import os
    import subprocess
    env = dict(os.environ)
    for key in ('ISAPOL_RESPONSE_MOLECULE','ISAPOL_NATIVE_DF_MOLECULE',
                'ISAPOL_CHECKPOINT_CALL','ISAPOL_CHECKPOINT_SWEEP'):
        env.pop(key,None)
    if enabled:
        env['ISAPOL_RESPONSE_MOLECULE']='7'
    result = subprocess.run([str(scalar_guard_executable),str(fault)],cwd=tmp_path,env=env,
                            capture_output=True,text=True)
    events = list(tmp_path.glob('isapol-response-*.dat'))
    if not enabled:
        assert result.returncode == 0 and result.stdout.strip() == '0'
        assert not events
    else:
        assert len(events) == 1
        content = events[0].read_text()
        assert content.startswith('ISAPOL_RESPONSE_EVENT 8\n')
        if fault:
            assert result.returncode == 92 and '\nCACHE_UNSUPPORTED\n' in content
            assert '\nDSD_REQUEST\n' not in content
        else:
            assert result.returncode == 0 and '\nDSD_REQUEST\n' in content
