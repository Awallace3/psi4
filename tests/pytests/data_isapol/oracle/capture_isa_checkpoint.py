#!/usr/bin/env python3
"""Instrument a PRIVATE CamCASP source copy for one production ISA-A update.

This script never edits its source input. Build the generated copy with CamCASP's
own makefile. Run in a fresh work directory with ISAPOL_CHECKPOINT_CALL=N (1-based
make_Stilde_stockholder_A call number). Output is isapol-checkpoint.dat, created
exclusively. No upstream source or executable is distributed by this harness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


# Independent diagnostic I/O only; no replacement of reference arithmetic.
CAPTURE_MODULE = '''module isapol_checkpoint_capture
use precision, only: dp
use types, only: basis_set
implicit none
integer :: cp_count=0, cp_unit, cp_atom=0
logical :: cp_active=.false.
character(*), parameter :: cp_fmt='(*(es26.17e3,1x))'
contains
subroutine cp_begin(atom)
integer, intent(in) :: atom
integer :: wanted, stat
character(64) :: value
if (cp_active) stop 'ISAPOL: unfinished preceding checkpoint'
cp_count=cp_count+1
call get_environment_variable('ISAPOL_CHECKPOINT_CALL',value,status=stat)
if (stat/=0) return
wanted=0
read(value,*,iostat=stat) wanted
if (stat/=0.or.wanted<1) stop 'ISAPOL: invalid checkpoint call'
cp_active=cp_count==wanted
if (.not.cp_active) return
cp_atom=atom
open(newunit=cp_unit,file='isapol-checkpoint.dat',status='new',action='write',iostat=stat)
if (stat/=0) stop 'ISAPOL: cannot create checkpoint (already exists?)'
write(cp_unit,'(a)') 'ISAPOL_CHECKPOINT 2'
end subroutine
subroutine cp_matrix(label,values)
character(*), intent(in) :: label
real(dp), intent(in) :: values(:,:)
integer :: row
write(cp_unit,'(a)') label
write(cp_unit,*) size(values,1),size(values,2)
do row=1,size(values,1)
  write(cp_unit,cp_fmt) values(row,:)
enddo
end subroutine
subroutine cp_basis(label,b)
character(*), intent(in) :: label
type(basis_set), intent(in) :: b
integer :: i
write(cp_unit,'(a)') label
write(cp_unit,*) b%nprims,b%nshells,b%nsites,b%size,b%max_l
write(cp_unit,'(a)') b%cartspher
do i=1,b%nsites
  write(cp_unit,'(a)') trim(b%label(i))
  write(cp_unit,cp_fmt) b%charge(i),b%coord(:,i)
enddo
do i=1,b%nprims
  write(cp_unit,cp_fmt) b%expon(i),b%coeff(i,:)
enddo
do i=1,b%nshells
  write(cp_unit,*) b%site(i),b%symm(i),b%first(i),b%last(i)
enddo
end subroutine
end module isapol_checkpoint_capture

'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor, found {text.count(old)}: {old!r}")
    return text.replace(old, new, 1)


def patch_routine(text, name, edits):
    pattern = rf"^subroutine {name}\(.*?^end subroutine {name}\b"
    matches = list(re.finditer(pattern, text, re.I | re.M | re.S))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one subroutine {name}")
    match = matches[0]
    body = match.group()
    for old, new in edits:
        body = replace_once(body, old, new)
    return text[:match.start()] + body + text[match.end():]


def instrument(num, stock):
    if 'module isapol_checkpoint_capture' in num:
        raise ValueError('Source already instrumented')
    stock = patch_routine(stock, 'make_Stilde_stockholder_A', [
        ('implicit none', 'use isapol_checkpoint_capture\nimplicit none'),
        ('integer :: nauxA', 'integer :: cp_shell, cp_comp, cp_first, cp_last, cp_l\ninteger :: nauxA'),
        ('!Now find indices', '''call cp_begin(atomindx)
if (cp_active) then
  if (trim(ISA_algorithm)/='A') stop 'ISAPOL: only Algorithm A capture supported'
  if (solver/=0) stop 'ISAPOL: LU solver required'
  if (.not.ISA_restart_perform_iterations) stop 'ISAPOL: iterative shape update required'
  if (DFtype/=4.or.ISAtype/=0.or.Add_RHO1) stop 'ISAPOL: unperturbed Drho-C required'
  write(cp_unit,*) cp_count,atomindx,nauxA,mol%atoms(atomindx)%aux%nshells
  write(cp_unit,'(a)') trim(mol%atoms(atomindx)%name)
  write(cp_unit,'(a)') mol%atoms(atomindx)%aux%cartspher
  write(cp_unit,cp_fmt) mol%atoms(atomindx)%coord
  write(cp_unit,cp_fmt) wEps,EtaDamp,PositiveW_Lambda,PositiveW_MaxAlpha
  write(cp_unit,*) merge(1,0,wEps_S_block_only),merge(1,0,PositiveW_Auto)
  write(cp_unit,cp_fmt) mol%atoms(atomindx)%D%vector
  do cp_shell=1,mol%atoms(atomindx)%aux%nshells
    cp_first=mol%atoms(atomindx)%aux%first(cp_shell)
    cp_last=mol%atoms(atomindx)%aux%last(cp_shell)
    cp_l=mol%atoms(atomindx)%aux%symm(cp_shell)
    call inquire_shell_size(mol%atoms(atomindx)%aux,cp_shell,cp_comp)
    if (cp_first/=cp_last) stop 'ISAPOL: contracted atomic basis unsupported'
    write(cp_unit,*) cp_l,cp_comp,cp_first,cp_last
    write(cp_unit,cp_fmt) mol%atoms(atomindx)%aux%expon(cp_first), &
      mol%atoms(atomindx)%aux%coeff(cp_first,cp_l+1)
  enddo
  call cp_matrix('OVERLAP',Stilde%matrix)
endif
!Now find indices'''),
        ('Stilde%onfile = .false.', "if (cp_active) call cp_matrix('METRIC',Stilde%matrix)\nStilde%onfile = .false."),
    ])
    num = patch_routine(num, 'Chi_rho_w_overlap', [
        ('implicit none', 'use isapol_checkpoint_capture\nimplicit none'),
        ('real(dp) :: L_wSum_eps', 'real(dp), allocatable :: cp_phi(:,:)\nlogical :: cp_described=.false.\nreal(dp) :: L_wSum_eps'),
        ('!Calculation of S:', '''if (cp_active) then
  if (atomindx/=cp_atom) stop 'ISAPOL: RHS atom mismatch'
  cp_described=.false.
  allocate(cp_phi(MaxPointsBatch,nauxA))
  write(cp_unit,'(a)') 'DENSITY'
  write(cp_unit,'(a)') trim(Rho%name)
  write(cp_unit,cp_fmt) L_wSum_eps
  write(cp_unit,*) merge(1,0,ApplyTailFix),merge(1,0,ApplyTailFix_DummySites)
endif
!Calculation of S:'''),
        ('    call calculate_stockholder_weights(WaValues,WsumValues,r,atomindx,mol,info,   &', '''    if (cp_active.and..not.cp_described) then
      write(cp_unit,'(a)') 'DESCRIPTORS'
      call cp_basis('ATOMIC_BASIS',mol%atoms(atomindx)%aux)
      call cp_basis('DENSITY_BASIS',Rho%Basis)
      write(cp_unit,'(a)') 'DENSITY_COEFFICIENTS'
      write(cp_unit,*) size(Rho%D%vector)
      write(cp_unit,cp_fmt) Rho%D%vector
      write(cp_unit,'(a)') 'DENSITY_NEIGHBOURS'
      write(cp_unit,*) size(mol%atoms(atomindx)%NeighbourList_aux1)
      write(cp_unit,*) mol%atoms(atomindx)%NeighbourList_aux1
      call cp_basis('SHAPE_BASIS',mol%atoms(atomindx)%w0%Basis)
      write(cp_unit,'(a)') 'SHAPE_MAP'
      write(cp_unit,*) size(mol%atoms(atomindx)%wShellMap)
      write(cp_unit,*) mol%atoms(atomindx)%wShellMap
      write(cp_unit,'(a)') 'SHAPE_OLD'
      write(cp_unit,*) size(mol%atoms(atomindx)%w0%D%vector)
      write(cp_unit,cp_fmt) mol%atoms(atomindx)%w0%D%vector
      cp_described=.true.
    endif
    call calculate_stockholder_weights(WaValues,WsumValues,r,atomindx,mol,info,   &'''),
        ('      l_a = mol%atoms(atomindx)%aux%symm(ashell)', '''      l_a = mol%atoms(atomindx)%aux%symm(ashell)
      if (cp_active) cp_phi(1:npoints_in_batch,st_a:end_a)=Chi_a(1:npoints_in_batch,1:NumComp_a)'''),
        ('    st_pt = end_pt + 1', '''    if (cp_active) then
      write(cp_unit,'(a)') 'BATCH'
      write(cp_unit,*) site,st_pt,npoints_in_batch
      do pt=1,npoints_in_batch
        write(cp_unit,cp_fmt) r(:,pt),wt(pt),RhoValues(pt),WaValues(pt),WsumValues(pt),cp_phi(pt,:)
      enddo
    endif
    st_pt = end_pt + 1'''),
        ('if (writemat) then', '''if (cp_active) then
  call cp_matrix('RHS',S%matrix)
  write(cp_unit,'(a)') 'POPULATION'
  write(cp_unit,cp_fmt) tot_rho
  deallocate(cp_phi)
endif
if (writemat) then'''),
    ])
    stock = patch_routine(stock, 'update_D_one_atom', [
        ('implicit none', 'use isapol_checkpoint_capture\nimplicit none'),
        ('mol%atoms(atomindx)%D%vector(1:naux) = C%matrix(:,1)', '''if (cp_active) then
  if (atomindx/=cp_atom) stop 'ISAPOL: solution atom mismatch'
  call cp_matrix('COEFFICIENTS',C%matrix)
endif
mol%atoms(atomindx)%D%vector(1:naux) = C%matrix(:,1)'''),
    ])
    stock = patch_routine(stock, 'update_w', [
        ('implicit none', 'use isapol_checkpoint_capture\nimplicit none'),
        ('if (n_diis > 0) then', '''if (cp_active) then
  if (atomindx/=cp_atom) stop 'ISAPOL: shape atom mismatch'
  write(cp_unit,'(a)') 'SHAPE_NEW_RAW'
  write(cp_unit,*) sizeW
  write(cp_unit,cp_fmt) mol%atoms(atomindx)%w%D%vector(1:sizeW)
  write(cp_unit,'(a)') 'END'
  close(cp_unit)
  cp_active=.false.
endif
if (n_diis > 0) then'''),
    ])
    return CAPTURE_MODULE + num, stock


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError('Destination must be new and outside the reference source tree')
    paths = ['src/num_integrals.F90', 'src/stockholder.F90']
    original = [(source / p).read_text() for p in paths]
    patched = instrument(*original)  # validate every anchor before any writes
    destination.mkdir(parents=True)
    shutil.copytree(source / 'src', destination / 'src')
    for name in ['Makefile', 'Makefile_body', 'VERSION']:
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / 'x86-64/gfortran/exe', destination / 'x86-64/gfortran/exe',
                    ignore=shutil.ignore_patterns('*.o', '*.mod', 'camcasp', 'casimir', 'cluster', 'process'))
    (destination / 'bin').mkdir()
    shutil.copy2(source / 'bin/version.py', destination / 'bin/version.py')
    for path, text in zip(paths, patched):
        (destination / path).write_text(text)
    metadata = {'schema_version': 1, 'source_root': str(source),
                'purpose': 'production Algorithm A frozen-update capture; not a converged-property oracle',
                'source_sha256': {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in paths},
                'instrumented_sha256': {p: hashlib.sha256((destination / p).read_bytes()).hexdigest() for p in paths},
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (destination / 'capture-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camcasp', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.camcasp, args.destination), indent=2))
