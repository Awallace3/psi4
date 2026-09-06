#!/usr/bin/env python3
"""Prepare a NEW private CamCASP tree for native Drho-C input capture.

Select one molecule with ISAPOL_NATIVE_DF_MOLECULE=1. Observers read only naturally
resident arrays; they never open/close/release reference matrix objects. Raw AO B
is deliberately absent: this first gate captures J/A, bases/C/q/RHS/Drho instead.
No reference sources or executables are distributed here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

spec = importlib.util.spec_from_file_location('native_capture_scaffold', Path(__file__).with_name('capture_isa_checkpoint.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MODULE = '''module isapol_native_df_capture
use precision, only: dp
use types, only: basis_set, molecule, real_matrix, real_vector
implicit none
private
public :: ndc_selected, ndc_metric, ndc_begin, ndc_rhs, ndc_solution
integer :: cp_unit, j_count=0, a_count=0
logical :: seen_j=.false., seen_a=.false., begun=.false., rhs_seen=.false.
character(*), parameter :: cp_fmt='(*(es26.17e3,1x))'
contains
subroutine require(ok,message)
logical, intent(in) :: ok
character(*), intent(in) :: message
if (.not.ok) then
  print *, 'ISAPOL_NATIVE_DF: ',message
  stop 91
endif
end subroutine
logical function ndc_selected(index)
integer, intent(in) :: index
integer :: wanted, stat, n
character(64) :: value
ndc_selected=.false.
call get_environment_variable('ISAPOL_NATIVE_DF_MOLECULE',value,status=stat)
if (stat==1) return
call require(stat==0,'invalid or oversized molecule selector')
read(value,*,iostat=stat) wanted
call require(stat==0,'invalid molecule selector')
call require(wanted>0,'molecule selector must be positive')
call get_environment_variable('ISAPOL_CHECKPOINT_CALL',length=n,status=stat)
call require(n==0,'simultaneous atom selector')
call get_environment_variable('ISAPOL_CHECKPOINT_SWEEP',length=n,status=stat)
call require(n==0,'simultaneous sweep selector')
ndc_selected=index==wanted
end function
subroutine create_file(filename)
character(*), intent(in) :: filename
integer :: stat
open(newunit=cp_unit,file=filename,status='new',action='write',iostat=stat)
call require(stat==0,'cannot create fresh capture file')
end subroutine
subroutine ndc_metric(index,label,m,cutoff,dummy)
integer, intent(in) :: index
character(*), intent(in) :: label
 type(real_matrix), intent(in) :: m
real(dp), intent(in) :: cutoff,dummy
integer :: serial
character(128) :: filename
if (.not.ndc_selected(index)) return
call require(.not.begun,'metric appeared after density entry')
call require(m%in_memory.and.allocated(m%matrix),'metric not naturally resident')
call require(m%rows==m%cols,'metric is not square')
call require(size(m%matrix,1)==m%rows.and.size(m%matrix,2)==m%cols,'metric dimensions')
if (label=='J') then
  j_count=j_count+1
  serial=j_count
  seen_j=.true.
else if (label=='A') then
  a_count=a_count+1
  serial=a_count
  seen_a=.true.
else
  call require(.false.,'unknown metric label')
endif
call require(serial<1000000,'too many metric records')
write(filename,'(a,a,a,i6.6,a)') 'isapol-native-df-',label,'-',serial,'.dat'
call create_file(trim(filename))
write(cp_unit,'(a)') 'ISAPOL_NATIVE_METRIC 1'
write(cp_unit,*) index
write(cp_unit,'(a)') label
write(cp_unit,cp_fmt) cutoff,dummy
call cp_matrix('METRIC',m%matrix)
write(cp_unit,'(a)') 'END_METRIC'
close(cp_unit)
end subroutine
subroutine ndc_begin(mol,st,sd,tt,td,same,restart,norm,solver,iterations,constraint,lambda,eta,gamma,scf)
type(molecule), intent(in) :: mol
character(*), intent(in) :: st,sd,tt,td,scf
logical, intent(in) :: same,restart
integer, intent(in) :: norm,solver,iterations,constraint
integer :: i
real(dp), intent(in) :: lambda,eta,gamma
if (.not.ndc_selected(mol%indx)) return
call require(.not.begun,'duplicate density fit')
call require(.not.same.and..not.restart,'fresh density fit required')
call require(norm==1.and.solver==0,'only Coulomb norm and LU supported')
call require(iterations>=0,'invalid LU refinement count')
call require(constraint==1.or.constraint==2,'unsupported constraint type')
call require(lambda==1000.0_dp.and.eta==0.0_dp.and.gamma==0.0_dp,'unsupported penalty')
call require(st=='Sc__'.and.tt=='T_MO','constrained NN route required')
call require((sd=='_A_A'.and.td=='___A').or.(sd=='_B_B'.and.td=='___B'),'NN descriptors')
call require(trim(mol%SCFcode)=='dalton'.and.trim(scf)=='dalton','DALTON MAIN required')
call require(mol%nocc>0.and.mol%nelectrons==2*mol%nocc,'closed-shell count mismatch')
call require(mol%main%size==mol%ndim.and.mol%aux%size==mol%naux,'basis dimensions')
call require(mol%C%in_memory.and.allocated(mol%C%matrix),'MO coefficients not resident')
call require(size(mol%C%matrix,1)==mol%ndim.and.size(mol%C%matrix,2)>=mol%nocc,'MO dimensions')
! J/A can be constructed earlier in the pair fit, before this entry hook.
call require(seen_j.and.seen_a,'missing earlier raw metric records')
begun=.true.
call create_file('isapol-native-df-state.dat')
write(cp_unit,'(a)') 'ISAPOL_NATIVE_DF 2'
write(cp_unit,*) mol%indx,mol%ndim,mol%naux,mol%nocc,mol%nelectrons
write(cp_unit,'(a)') trim(mol%name)
write(cp_unit,'(a)') trim(mol%SCFcode)
write(cp_unit,'(a)') trim(scf)
write(cp_unit,'(a)') st//' '//sd//' '//tt//' '//td
write(cp_unit,*) norm,solver,iterations,constraint
write(cp_unit,cp_fmt) lambda,eta,gamma
write(cp_unit,'(a)') 'METRIC_COUNTS'
write(cp_unit,*) j_count,a_count
call basis_metadata('MAIN_METADATA',mol%main)
call cp_basis('MAIN_BASIS',mol%main)
call basis_metadata('AUX_METADATA',mol%aux)
call cp_basis('AUX_BASIS',mol%aux)
call cp_matrix('C_OCC',mol%C%matrix(1:mol%ndim,1:mol%nocc))
write(cp_unit,'(a)') 'OCCUPATIONS_ASSUMED_BY_CLOSED_SHELL_ROUTINE'
write(cp_unit,*) mol%nocc
write(cp_unit,cp_fmt) (2.0_dp,i=1,mol%nocc)
end subroutine
subroutine basis_metadata(label,b)
character(*), intent(in) :: label
type(basis_set), intent(in) :: b
call require(b%defined,'undefined basis')
call require(b%cartspher=='C'.or.b%cartspher=='S','unsupported representation')
call require(allocated(b%symm).and.allocated(b%expon).and.allocated(b%coeff),'missing basis arrays')
call require(all(b%symm>=0).and.all(b%symm<=4),'only S through G supported')
call require(all(b%expon>0),'nonpositive exponent')
call require(all(b%first>=1).and.all(b%last<=b%nprims).and.all(b%first<=b%last),'primitive ranges')
write(cp_unit,'(a)') label
write(cp_unit,'(a)') trim(b%name)
write(cp_unit,'(a)') trim(b%MainAux)
write(cp_unit,'(a)') trim(b%BasisType)
write(cp_unit,*) b%mol_charge
end subroutine
subroutine cp_vector(label,values)
character(*), intent(in) :: label
real(dp), intent(in) :: values(:)
write(cp_unit,'(a)') label
write(cp_unit,*) size(values)
write(cp_unit,cp_fmt) values
end subroutine
subroutine ndc_rhs(mol,q,rhs)
type(molecule), intent(in) :: mol
type(real_vector), intent(in) :: q
type(real_matrix), intent(in) :: rhs
if (.not.ndc_selected(mol%indx)) return
call require(begun.and..not.rhs_seen,'RHS observer lifecycle')
call require(q%in_memory.and.allocated(q%vector),'q not naturally resident')
call require(rhs%in_memory.and.allocated(rhs%matrix),'RHS not naturally resident')
call require(size(q%vector)==mol%naux,'q dimensions')
call require(size(rhs%matrix,1)==1.and.size(rhs%matrix,2)==mol%naux,'RHS dimensions')
call cp_vector('Q',q%vector)
call cp_vector('RHS_CONSTRAINED',rhs%matrix(1,:))
rhs_seen=.true.
end subroutine
subroutine ndc_solution(index,d)
integer, intent(in) :: index
type(real_vector), intent(in) :: d
if (.not.ndc_selected(index)) return
call require(begun.and.rhs_seen,'solution observer lifecycle')
call require(d%in_memory.and.allocated(d%vector),'Drho not naturally resident')
call cp_vector('DRHO',d%vector)
write(cp_unit,'(a)') 'END_NATIVE_DF'
close(cp_unit)
end subroutine
'''
# Reuse only descriptor serialization, not any numerical oracle.
MODULE += 'subroutine cp_matrix'+base.CAPTURE_MODULE.split('subroutine cp_matrix', 1)[1].split('end module', 1)[0]
MODULE += 'end module isapol_native_df_capture\n\n'


def patch_routine(text, name, edits):
    pattern = rf'(?ms)^[ \t]*subroutine {name}\b.*?^[ \t]*end subroutine {name}\b[^\n]*'
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one routine {name}')
    match = matches[0]
    body = match.group()
    for old, new in edits:
        body = base.replace_once(body, old, new)
    return text[:match.start()]+body+text[match.end():]


def instrument(smat, monomer, makefile):
    module_pattern = r'(?m)^[ \t]*module[ \t]+df_Smat[ \t]*$'
    if len(re.findall(module_pattern, smat)) != 1:
        raise ValueError('Expected exactly one df_Smat module declaration')
    smat = re.sub(module_pattern, r'\g<0>\nuse isapol_native_df_capture, only: ndc_metric', smat, count=1)
    smat = patch_routine(smat, 'make_s_matrix_coulomb', [
        ('if (debug_matrices) call matout_types(S)',
         "call ndc_metric(mol%indx,'J',S,integral_cutoff,dummy_s_exponent)\nif (debug_matrices) call matout_types(S)")])
    anchor = 'call close_type(Sc,info,transpose=.false.,forcewrite=.true.,release=.false.,debug=debug2)'
    smat = patch_routine(smat, 'make_s_matrix_constraints', [
        (anchor, "call ndc_metric(mol%indx,'A',Sc,integral_cutoff,dummy_s_exponent)\n"+anchor)])
    monomer = patch_routine(monomer, 'do_DFrho_monomer', [
        ('implicit none', '''use isapol_native_df_capture, only: ndc_begin,ndc_rhs,ndc_solution
  use global_data, only: g_scf_code
  use df_parameter_module, only: ndc_norm=>DFNormIndx,ndc_solver=>solver, &
    ndc_iterations=>iterations,ndc_constraint=>ConstraintType
  implicit none'''),
        ('  if (same) then', '''  call ndc_begin(mol,Stype,Sdesc,Ttype,Tdesc,same,restart, &
    ndc_norm,ndc_solver,ndc_iterations,ndc_constraint,lambda,eta,gamma,g_scf_code)
  if (same) then'''),
        ('    call write_type(Trho,info,release=.false.)',
         '    call ndc_rhs(mol,Iint%int,Trho)\n    call write_type(Trho,info,release=.false.)'),
        ('    Drho%vector(1:naux) = Drhotmp%matrix(1,1:naux)',
         '    Drho%vector(1:naux) = Drhotmp%matrix(1,1:naux)\n    call ndc_solution(mol%indx,Drho)')])
    return MODULE+smat, monomer, makefile+'\n# Observer module is emitted by df_Smat.o.\ndf_monomer.o: df_Smat.o\n'


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError('Destination must be new and outside reference source')
    paths = ['src/df_Smat.F90', 'src/df_monomer.F90', 'Makefile_body']
    patched = instrument(*[(source/p).read_text() for p in paths])
    destination.mkdir(parents=True)
    shutil.copytree(source/'src', destination/'src')
    for name in ['Makefile', 'Makefile_body', 'VERSION']:
        shutil.copy2(source/name, destination/name)
    shutil.copytree(source/'x86-64/gfortran/exe', destination/'x86-64/gfortran/exe',
                    ignore=shutil.ignore_patterns('*.o', '*.mod', 'camcasp', 'casimir', 'cluster', 'process'))
    (destination/'bin').mkdir()
    shutil.copy2(source/'bin/version.py', destination/'bin/version.py')
    for path,text in zip(paths,patched):
        (destination/path).write_text(text)
    report = dict(schema_version=1, purpose=__doc__, source_root=str(source),
                  source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in paths},
                  instrumented_sha256={p:hashlib.sha256((destination/p).read_bytes()).hexdigest() for p in paths},
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  serializer_harness_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                  limitations=['No raw AO three-centre B export', 'Closed-shell occupations assumed, not read',
                               'No native Libint2 or end-to-end certification'])
    (destination/'capture-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camcasp', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.camcasp,args.destination),indent=2))
