#!/usr/bin/env python3
"""Opt-in isolated-source whole-sweep capture for the three-site Drho-C/LU track.

Select an ordinary iteration with ISAPOL_CHECKPOINT_SWEEP. Existing single-call
capture remains available, but simultaneous selectors are rejected. Never edits
reference source or archives; prepare into a new destination and build serially.
"""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location('capture_atom', Path(__file__).with_name('capture_isa_checkpoint.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base_instrument = base.instrument

HELPERS = '''subroutine cs_config()
integer :: stat
character(64) :: v, call_v
cs_wanted=0
call get_environment_variable('ISAPOL_CHECKPOINT_SWEEP',v,status=stat)
if (stat/=0) return
read(v,*,iostat=stat) cs_wanted
if (stat/=0.or.cs_wanted<1) error stop 'ISAPOL: invalid sweep selector'
call get_environment_variable('ISAPOL_CHECKPOINT_CALL',call_v,status=stat)
if (stat==0) error stop 'ISAPOL: simultaneous call and sweep selectors'
end subroutine
subroutine cs_begin(iteration,n)
integer, intent(in) :: iteration,n
integer :: stat
cs_active=iteration==cs_wanted
if (.not.cs_active) return
if (n/=3.or.cs_done) error stop 'ISAPOL: only one three-site sweep supported'
cs_closed=0
cs_calls=0
open(newunit=cs_unit,file='isapol-sweep-state.dat',status='new',action='write',iostat=stat)
if (stat/=0) error stop 'ISAPOL: cannot create sweep state'
write(cs_unit,'(a)') 'ISAPOL_SWEEP_STATE 1'
write(cs_unit,*) iteration,n
end subroutine
subroutine cs_vector(label,v)
character(*), intent(in) :: label
real(dp), intent(in) :: v(:)
write(cs_unit,'(a)') label
write(cs_unit,*) size(v)
write(cs_unit,cp_fmt) v
end subroutine
subroutine cs_state(label,mol,eps,lambda,apply,converged)
character(*), intent(in) :: label
type(molecule), intent(in) :: mol
real(dp), intent(in) :: eps,lambda
logical, intent(in) :: apply,converged
integer :: a
write(cs_unit,'(a)') label
write(cs_unit,*) eps,lambda,merge(1,0,apply),merge(1,0,converged)
do a=1,mol%nTypeAtom
  if (.not.mol%atoms(a)%D%in_memory.or..not.mol%atoms(a)%D0%in_memory) &
    error stop 'ISAPOL: D vectors not in memory'
  if (.not.mol%atoms(a)%w%D%in_memory.or..not.mol%atoms(a)%w0%D%in_memory) &
    error stop 'ISAPOL: shape vectors not in memory'
  if (.not.allocated(mol%atoms(a)%D%vector)) error stop 'ISAPOL: missing D'
  if (.not.allocated(mol%atoms(a)%D0%vector)) error stop 'ISAPOL: missing D0'
  if (.not.allocated(mol%atoms(a)%w%D%vector)) error stop 'ISAPOL: missing w'
  if (.not.allocated(mol%atoms(a)%w0%D%vector)) error stop 'ISAPOL: missing w0'
  write(cs_unit,'(a)') 'ATOM'
  write(cs_unit,*) a,mol%atoms(a)%Z
  write(cs_unit,'(a)') trim(mol%atoms(a)%name)
  write(cs_unit,cp_fmt) mol%atoms(a)%coord
  call cs_vector('D',mol%atoms(a)%D%vector)
  call cs_vector('D0',mol%atoms(a)%D0%vector)
  call cs_vector('W',mol%atoms(a)%w%D%vector)
  call cs_vector('W0',mol%atoms(a)%w0%D%vector)
  write(cs_unit,'(a)') 'CHARGES'
  write(cs_unit,cp_fmt) mol%atoms(a)%w_charge,mol%atoms(a)%w0_charge,mol%atoms(a)%ISAcharge
  write(cs_unit,'(a)') 'FLAGS'
  write(cs_unit,*) merge(1,0,mol%atoms(a)%StockholderConverged), &
    merge(1,0,mol%atoms(a)%w_Tail_Defined),mol%atoms(a)%w_Tail_FuncIndx
  write(cs_unit,'(a)') 'TAIL'
  write(cs_unit,cp_fmt) mol%atoms(a)%w_Tail_Cutoffs(1:2),mol%atoms(a)%w_Tail_FuncParams(1:2)
  write(cs_unit,'(a)') 'SHAPE_NEIGHBOURS'
  write(cs_unit,*) mol%atoms(a)%NumNeighbours
  write(cs_unit,*) mol%atoms(a)%NeighbourList(1:mol%atoms(a)%NumNeighbours)
enddo
end subroutine
'''


def controller_routine(stock):
    name = 'Iterative_Stockholder_Atoms_restart'
    calls = re.findall(r'^\s*call\s+'+name+r'\s*\(', stock, re.I | re.M)
    if len(calls) != 1:
        raise ValueError('Expected exactly one live restart-capable controller dispatch')
    return name


def instrument(num, stock):
    routine = controller_routine(stock)
    num, stock = base_instrument(num, stock)
    num = base.replace_once(num, 'use types, only: basis_set', 'use types, only: basis_set,molecule')
    num = base.replace_once(num, 'logical :: cp_active=.false.', '''logical :: cp_active=.false.
logical :: cs_active=.false.,cs_done=.false.
integer :: cs_wanted=0,cs_unit,cs_closed=0,cs_calls(3)=0
real(dp) :: cs_config_lambda=0.0_dp''')
    num = base.replace_once(num, 'end subroutine\nsubroutine cp_matrix', 'end subroutine cp_begin\nsubroutine cp_matrix')
    num = base.patch_routine(num, 'cp_begin', [
        ('character(64) :: value', 'character(64) :: value,filename'),
        ("call get_environment_variable('ISAPOL_CHECKPOINT_CALL',value,status=stat)", '''if (cs_active) then
  if (atom/=cs_closed+1.or.atom<1.or.atom>3) error stop 'ISAPOL: sweep atom order'
  cp_active=.true.
  wanted=cp_count
  cs_calls(atom)=cp_count
else
call get_environment_variable('ISAPOL_CHECKPOINT_CALL',value,status=stat)'''),
        ('cp_active=cp_count==wanted', 'cp_active=cp_count==wanted\nendif'),
        ("open(newunit=cp_unit,file='isapol-checkpoint.dat',status='new',action='write',iostat=stat)", '''filename='isapol-checkpoint.dat'
if (cs_active) write(filename,'(a,i0,a)') 'isapol-atom-',atom,'.dat'
open(newunit=cp_unit,file=trim(filename),status='new',action='write',iostat=stat)'''),
    ])
    num = base.replace_once(num, 'end module isapol_checkpoint_capture', HELPERS+'end module isapol_checkpoint_capture')
    stock = base.patch_routine(stock, 'update_w', [
        ('  cp_active=.false.', '  cp_active=.false.\n  if (cs_active) cs_closed=cs_closed+1'),
    ])
    # The live driver calls this restart-capable routine even for FRESH ISA runs.
    # The unsuffixed historical routine is not dispatched by this source revision.
    stock = base.patch_routine(stock, routine, [
        ('implicit none\ntype(molecule), intent(inout) :: mol\ninteger, intent(in) :: ISAtype',
         'use isapol_checkpoint_capture\nimplicit none\ntype(molecule), intent(inout) :: mol\ninteger, intent(in) :: ISAtype'),
        ('if (Decouple_wEps_and_TailFix) then', '''call cs_config()
cs_config_lambda=PositiveW_Lambda
if (Decouple_wEps_and_TailFix) then'''),
        ('    MaxDelta = 0.0_dp\n    !\n    do atomindx = 1, mol%nTypeAtom', '''    MaxDelta = 0.0_dp
    !
    call cs_begin(iteration,mol%nTypeAtom)
    if (cs_active) then
      if (ISA_restart.or..not.ISA_restart_perform_iterations) error stop 'ISAPOL: fresh iterative run required'
      if (ISA_algorithm/='A'.or.ConvergenceType/=0.or.n_diis/=0.or.symmetrize) &
        error stop 'ISAPOL: sweep requires ordinary A/W, no DIIS or symmetry'
      if (Decouple_wEps_and_TailFix.or.SelfConsistentTail) error stop 'ISAPOL: unsupported tail loops'
      if (ShapeFuncType/=1.or.ShapeFuncFitType/=3) error stop 'ISAPOL: Func1 Fit3 required'
      if (.not.wEps_S_block_only) error stop 'ISAPOL: only s-block weighted source path supported'
      write(cs_unit,'(a)') 'CONFIG_FLOATS'
      write(cs_unit,cp_fmt) wEps,cs_config_lambda,EtaDamp,PositiveW_MaxAlpha,conv_eps_norm, &
        wEps_EpsNorm,PositiveW_EpsNorm,TailFix_EpsNorm,w_mix_fraction
      write(cs_unit,'(a)') 'CONFIG_INTS'
      write(cs_unit,*) merge(1,0,wEps_S_block_only),merge(1,0,PositiveW_Auto),MaxIterations, &
        w_mix_skip_iterations,TailFix_IterMax,merge(1,0,FixShapeFuncTails), &
        merge(1,0,FixShapeFuncTails_DummySites),merge(1,0,Convergence_Skip_Dummy_Sites)
      call cs_state('PRE',mol,LwEps,PositiveW_Lambda,ApplyTailFix,StockholderConverged)
    endif
    do atomindx = 1, mol%nTypeAtom'''),
        ('  call flush(6)\nenddo', '''  if (cs_active) then
    if (cs_closed/=mol%nTypeAtom.or.cp_active) error stop 'ISAPOL: incomplete atom sweep'
    write(cs_unit,'(a)') 'DELTAS'
    write(cs_unit,cp_fmt) delta_a
    write(cs_unit,'(a)') 'MAX_DELTA'
    write(cs_unit,cp_fmt) MaxDelta
    write(cs_unit,'(a)') 'CALLS'
    write(cs_unit,*) cs_calls
    call cs_state('POST',mol,LwEps,PositiveW_Lambda,ApplyTailFix,StockholderConverged)
    write(cs_unit,'(a)') 'END_SWEEP'
    close(cs_unit)
    cs_active=.false.
    cs_done=.true.
  endif
  call flush(6)
enddo
if (cs_wanted>0.and..not.cs_done) error stop 'ISAPOL: requested sweep not reached' ''')
    ])
    return num, stock


def prepare(source, destination):
    base.instrument = instrument
    try:
        metadata = base.prepare(source, destination)
    finally:
        base.instrument = base_instrument
    metadata['sweep_harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['purpose'] = 'ordinary three-site A/W controller sweep transition capture; not end-to-end parity'
    (Path(destination)/'capture-provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camcasp', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.camcasp, args.destination), indent=2))
