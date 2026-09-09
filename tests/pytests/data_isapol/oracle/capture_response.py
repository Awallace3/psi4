#!/usr/bin/env python3
"""Prepare an isolated observer for the pinned OLD/internal CamCASP response.

Numbered, read-only resident events; no observer reference-object I/O. Captures
DF-produced Hessian tensors, OO/OV/VV subset rows and actual kernel density.
Schema8 adds scalar-only metric/tensor cache lifecycles and bounded mutation guards.
Preparation alone does not validate an observer; fresh build/run/replay evidence is
required. No quadrature-weight, native generation or end-to-end parity claim.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

spec = importlib.util.spec_from_file_location('response_scaffold', Path(__file__).with_name('capture_native_df.py'))
scaffold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaffold)
base = scaffold.base
PIN = '63b16a22b9bae597fe81ecdb8b8d91c21868c814'

MODULE = '''module isapol_response_capture
use precision, only: dp
use types, only: molecule, real_matrix, basis_set, df_parameters, twoeint, FuncExpansion
use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
implicit none
private
public :: nrc_matrix,nrc_orbitals,nrc_energies,nrc_projection,nrc_policy,nrc_result
public :: nrc_ov_row,nrc_pair_row,nrc_fit_metadata,nrc_kernel,nrc_direct_guard
public :: nrc_density_begin,nrc_density_end,nrc_density,nrc_solve_begin,nrc_solve_end
public :: selected
public :: nrc_guard,nrc_cache_check,nrc_metric_request,nrc_metric_end,nrc_metric_begin
public :: nrc_metric_link,nrc_metric_result,nrc_dsd_request,nrc_dsd_complete
integer :: metric_id=0,build_id=0,dsd_id=0
logical :: metric_active=.false.,build_active=.false.
integer :: solve_id=0,solve_header(6)
logical :: solve_active=.false.
character(512) :: solve_name,solve_scf
logical :: density_active=.false.
integer :: density_serial=0
character(512) :: density_file=''
public :: nrc_fit_begin,nrc_fit_end,nrc_fit_block,nrc_hessian_tensor,nrc_kernel_policy
integer :: cp_unit, serial=0
integer :: fit_id=0,fit_header(6),fit_nmos,fit_pairs,fit_next,fit_blocks
logical :: fit_active=.false.
character(512) :: fit_name,fit_scf,fit_files(3)
logical :: old_seen=.false.
character(*), parameter :: cp_fmt='(*(es26.17e3,1x))'
contains
! New lifecycle APIs accept copied scalar metadata only, never reference objects.
subroutine nrc_guard(header,name,scf,reason,unsafe)
integer, intent(in) :: header(6)
character(*), intent(in) :: name,scf,reason
logical, intent(in) :: unsafe
if (.not.selected(header(1))) return
if (.not.unsafe) return
call scalar_event(header,name,scf,'CACHE_UNSUPPORTED')
write(cp_unit,'(a)') reason
call end_event()
call require(.false.,reason)
end subroutine
subroutine cache_policy(norm,cutoff,exponent)
integer, intent(in) :: norm
real(dp), intent(in) :: cutoff,exponent
write(cp_unit,*) norm
call cp_vector('METRIC_POLICY',(/cutoff,exponent/))
end subroutine
subroutine nrc_cache_check(header,name,scf,a,b,initialized,same,identity,normsame,norm,cutoff,exponent)
integer, intent(in) :: header(6),a,b,norm
character(*), intent(in) :: name,scf
logical, intent(in) :: initialized,same,identity,normsame
real(dp), intent(in) :: cutoff,exponent
if (.not.selected(header(1))) return
call nrc_guard(header,name,scf,'cache identity or norm mutation',initialized.and.(.not.identity.or..not.normsame))
call scalar_event(header,name,scf,'CACHE_CHECK')
write(cp_unit,*) a,b
call cache_policy(norm,cutoff,exponent)
write(cp_unit,*) merge(1,0,initialized)
write(cp_unit,*) merge(1,0,same)
call end_event()
end subroutine
subroutine cache_metric(mt,md,mn,mf,ms,flags,norm,cutoff,exponent)
character(*), intent(in) :: mt,md,mn,mf
integer, intent(in) :: ms(2),flags(4),norm
real(dp), intent(in) :: cutoff,exponent
write(cp_unit,'(a)') mt,md,trim(mn),trim(mf)
write(cp_unit,*) ms
write(cp_unit,*) flags
call cache_policy(norm,cutoff,exponent)
end subroutine
subroutine nrc_metric_request(header,name,scf,a,b,route,oname,ofile,cached,sdone, &
  mt,md,mn,mf,ms,flags,norm,cutoff,exponent)
integer, intent(in) :: header(6),a,b,ms(2),flags(4),norm
character(*), intent(in) :: name,scf,route,oname,ofile,mt,md,mn,mf
logical, intent(in) :: cached,sdone
real(dp), intent(in) :: cutoff,exponent
if (.not.selected(header(1))) return
call nrc_guard(header,name,scf,'unsupported metric flags/norm',any(flags/=(/1,0,0,0/)).or.norm/=1)
call nrc_guard(header,name,scf,'constrained/ordinary metric backing alias',route=='DF_SC'.and.ofile==mf)
call require(.not.metric_active,'nested metric owner')
metric_id=metric_id+1
call scalar_event(header,name,scf,'METRIC_REQUEST')
write(cp_unit,*) metric_id
write(cp_unit,*) a,b
write(cp_unit,'(a)') route,trim(oname),trim(ofile)
write(cp_unit,*) merge(1,0,cached)
write(cp_unit,*) merge(1,0,sdone)
call cache_metric(mt,md,mn,mf,ms,flags,norm,cutoff,exponent)
call end_event()
metric_active=.not.cached
end subroutine
subroutine nrc_metric_end(header,name,scf,oname,ofile,mn,mf,done,info)
integer, intent(in) :: header(6),info
character(*), intent(in) :: name,scf,oname,ofile,mn,mf
logical, intent(in) :: done
if (.not.selected(header(1))) return
call require(metric_active,'metric result without owner')
call scalar_event(header,name,scf,'METRIC_END')
write(cp_unit,*) metric_id,info
write(cp_unit,*) merge(1,0,done)
write(cp_unit,'(a)') trim(oname),trim(ofile),trim(mn),trim(mf)
call end_event()
metric_active=.false.
end subroutine
subroutine nrc_metric_begin(header,name,scf,mn,mf,ms,norm,cutoff,exponent,rp,rv)
integer, intent(in) :: header(6),ms(2),norm,rp,rv
character(*), intent(in) :: name,scf,mn,mf
real(dp), intent(in) :: cutoff,exponent
if (.not.selected(header(1))) return
call require(metric_active.and..not.build_active,'common metric without unique wrapper')
build_id=build_id+1
build_active=.true.
call scalar_event(header,name,scf,'METRIC_BUILD_BEGIN')
write(cp_unit,*) build_id,metric_id
write(cp_unit,'(a)') trim(mn),trim(mf)
write(cp_unit,*) ms
write(cp_unit,*) rp,rv
call cache_policy(norm,cutoff,exponent)
call end_event()
end subroutine
subroutine nrc_metric_link(header,name,scf)
integer, intent(in) :: header(6)
character(*), intent(in) :: name,scf
integer :: jserial
if (.not.selected(header(1))) return
call require(build_active,'resident J without build')
jserial=serial
call scalar_event(header,name,scf,'METRIC_J_LINK')
write(cp_unit,*) build_id,jserial
call end_event()
end subroutine
subroutine nrc_metric_result(header,name,scf,mn,mf,ms,info)
integer, intent(in) :: header(6),ms(2),info
character(*), intent(in) :: name,scf,mn,mf
if (.not.selected(header(1))) return
call require(build_active,'metric result without build')
call scalar_event(header,name,scf,'METRIC_BUILD_END')
write(cp_unit,*) build_id,info
write(cp_unit,'(a)') trim(mn),trim(mf)
write(cp_unit,*) ms
call end_event()
build_active=.false.
end subroutine
subroutine cache_output(a,b,tt,td,tn,tf,ts,mapping,flags)
integer, intent(in) :: a,b,ts(2),mapping(4),flags(5)
character(*), intent(in) :: tt,td,tn,tf
write(cp_unit,*) a,b
write(cp_unit,'(a)') tt,td,trim(tn),trim(tf)
write(cp_unit,*) ts
write(cp_unit,*) mapping
write(cp_unit,*) flags
end subroutine
subroutine cache_operand(n,f,s,par,values,controls)
character(*), intent(in) :: n,f
integer, intent(in) :: s(2),par(4),controls(2)
real(dp), intent(in) :: values(4)
write(cp_unit,'(a)') trim(n),trim(f)
write(cp_unit,*) s
write(cp_unit,*) par
call cp_vector('FIT_PARAMETERS',values)
write(cp_unit,*) controls
end subroutine
subroutine nrc_dsd_request(header,name,scf,a,b,tt,td,tn,tf,ts,mapping,tflags,cached,sdone, &
  mt,md,mn,mf,ms,mflags,ln,lf,ls,lp,lv,lc,rn,rf,rs,rpar,rvalues,rc, &
  switch,norm,cutoff,exponent,rid,tp,tv)
integer, intent(in) :: header(6),a,b,ts(2),mapping(4),tflags(5),ms(2),mflags(4)
integer, intent(in) :: ls(2),lp(4),lc(2),rs(2),rpar(4),rc(2),switch,norm,tp,tv
integer, intent(out) :: rid
character(*), intent(in) :: name,scf,tt,td,tn,tf,mt,md,mn,mf,ln,lf,rn,rf
real(dp), intent(in) :: lv(4),rvalues(4),cutoff,exponent
logical, intent(in) :: cached,sdone
rid=0
if (.not.selected(header(1))) return
call nrc_guard(header,name,scf,'unsupported tensor flags/switch/transpose', &
  any(tflags/=(/1,0,0,0,0/)).or.switch/=0.or.tv/=0.or.tp<0.or.tp>1)
call nrc_guard(header,name,scf,'unsupported tensor metric flags/norm',any(mflags/=(/1,0,0,0/)).or.norm/=1)
call nrc_guard(header,name,scf,'tensor output/input backing alias',tf==mf.or.tf==lf.or.tf==rf.or.mf==lf.or.mf==rf)
call nrc_guard(header,name,scf,'coefficient backing alias',lf==rf.and.ln/=rn)
select case(tt)
case('OVOV')
  call nrc_guard(header,name,scf,'unsupported OVOV map/shape/operand alias', &
    any(mapping/=(/2,2,header(4),header(4)/)).or.any(ts/=header(4)*header(5)).or.lf/=rf.or.ln/=rn)
case('VVOO')
  call nrc_guard(header,name,scf,'unsupported VVOO map/shape/operand alias', &
    any(mapping/=(/1,1,0,0/)).or.lf==rf.or. &
    ts(1)/=header(5)*(header(5)+1)/2.or.ts(2)/=header(4)*(header(4)+1)/2)
case default
  call nrc_guard(header,name,scf,'unsupported tensor type',.true.)
end select
dsd_id=dsd_id+1
rid=dsd_id
call scalar_event(header,name,scf,'DSD_REQUEST')
write(cp_unit,*) rid
call cache_output(a,b,tt,td,tn,tf,ts,mapping,tflags)
write(cp_unit,*) merge(1,0,cached)
write(cp_unit,*) merge(1,0,sdone)
write(cp_unit,*) tp,tv
write(cp_unit,*) switch
call cache_metric(mt,md,mn,mf,ms,mflags,norm,cutoff,exponent)
call cache_operand(ln,lf,ls,lp,lv,lc)
call cache_operand(rn,rf,rs,rpar,rvalues,rc)
call end_event()
end subroutine
subroutine nrc_dsd_complete(header,name,scf,a,b,tt,td,tn,tf,ts,mapping,flags,rid,info,done)
integer, intent(in) :: header(6),a,b,ts(2),mapping(4),flags(5),rid,info
character(*), intent(in) :: name,scf,tt,td,tn,tf
logical, intent(in) :: done
if (.not.selected(header(1))) return
call require(rid>0,'tensor result without request')
call scalar_event(header,name,scf,'DSD_COMPLETE')
write(cp_unit,*) rid,info
write(cp_unit,*) merge(1,0,done)
call cache_output(a,b,tt,td,tn,tf,ts,mapping,flags)
call end_event()
end subroutine
subroutine require(ok,message)
logical, intent(in) :: ok
character(*), intent(in) :: message
if (.not.ok) then
  print *, 'ISAPOL_RESPONSE: ',message
  stop 92
endif
end subroutine
logical function selected(index)
integer, intent(in) :: index
integer :: wanted,stat,n
character(64) :: value
selected=.false.
call get_environment_variable('ISAPOL_RESPONSE_MOLECULE',value,status=stat)
if (stat==1) return
call require(stat==0,'invalid molecule selector')
read(value,*,iostat=stat) wanted
call require(stat==0.and.wanted>0,'invalid molecule selector')
call get_environment_variable('ISAPOL_NATIVE_DF_MOLECULE',length=n,status=stat)
call require(n==0,'simultaneous native density capture')
call get_environment_variable('ISAPOL_CHECKPOINT_CALL',length=n,status=stat)
call require(n==0,'simultaneous atomic capture')
call get_environment_variable('ISAPOL_CHECKPOINT_SWEEP',length=n,status=stat)
call require(n==0,'simultaneous sweep capture')
selected=index==wanted
end function
subroutine begin_event(mol,label)
type(molecule), intent(in) :: mol
character(*), intent(in) :: label
call scalar_event((/mol%indx,mol%ndim,mol%naux,mol%nocc,mol%nvir,mol%nelectrons/),mol%name,mol%SCFcode,label)
end subroutine
subroutine scalar_event(header,name,scf,label)
integer, intent(in) :: header(6)
character(*), intent(in) :: name,scf,label
character(128) :: filename
integer :: stat
call require(header(4)>0.and.header(5)>0,'nonempty occupied/virtual spaces required')
call require(header(6)==2*header(4),'closed-shell count mismatch')
call require(trim(scf)=='dalton','only explicit DALTON convention supported')
serial=serial+1
call require(serial<1000000,'too many events')
write(filename,'(a,i6.6,a)') 'isapol-response-',serial,'.dat'
open(newunit=cp_unit,file=filename,status='new',action='write',iostat=stat)
call require(stat==0,'cannot create fresh event')
write(cp_unit,'(a)') 'ISAPOL_RESPONSE_EVENT 8'
write(cp_unit,*) serial,header
write(cp_unit,'(a)') label
write(cp_unit,'(a)') trim(name)
write(cp_unit,'(a)') trim(scf)
end subroutine
subroutine end_event()
write(cp_unit,'(a)') 'END_RESPONSE_EVENT'
close(cp_unit)
end subroutine
subroutine resident(m,context)
type(real_matrix), intent(in) :: m
character(*), intent(in) :: context
if (.not.m%in_memory.or..not.allocated(m%matrix)) then
  print *, 'ISAPOL_RESPONSE operand: ',trim(context),' : ',trim(m%name)
  print *, 'in_memory,allocated,rows,cols: ',m%in_memory,allocated(m%matrix),m%rows,m%cols
endif
call require(m%in_memory.and.allocated(m%matrix),'matrix not naturally resident')
call require(size(m%matrix,1)==m%rows.and.size(m%matrix,2)==m%cols,'matrix dimensions')
call require(all(ieee_is_finite(m%matrix)),'nonfinite matrix')
end subroutine
subroutine cp_vector(label,values)
character(*), intent(in) :: label
real(dp), intent(in) :: values(:)
call require(all(ieee_is_finite(values)),'nonfinite vector')
write(cp_unit,'(a)') label
write(cp_unit,*) size(values)
write(cp_unit,cp_fmt) values
end subroutine
subroutine nrc_solve_begin(mol,A,B,X,par,st,sd,tt,td,constraint,solver,iterations,restart)
type(molecule), intent(in) :: mol
type(real_matrix), intent(in) :: A,B,X
type(df_parameters), intent(in) :: par
character(*), intent(in) :: st,sd,tt,td
integer, intent(in) :: constraint,solver,iterations
logical, intent(in) :: restart
integer :: pairs
if (.not.selected(mol%indx)) return
call require(.not.solve_active,'nested parent solve')
call require(par%df_type==1,'only full NN parent solves supported')
call require(par%lambda==0.0_dp.or.par%lambda==1.0_dp,'unsupported parent lambda')
call require(par%DFNormIndx==1.and.par%defined,'unsupported parent norm')
call require(par%eta==0.0_dp.and.par%gamma==0.0_dp,'unsupported parent constraints')
call require(.not.restart.and.solver==0.and.iterations==0,'fresh unrefined parent LU required')
call require(constraint==1.or.constraint==2,'unsupported parent constraint type')
call require((par%lambda==0.0_dp.and.st=='S___').or. &
             (par%lambda==1.0_dp.and.st=='Sc__'),'parent metric type')
call require(tt=='T_MO','parent RHS type')
call require((sd=='_A_A'.and.td=='___A').or.(sd=='_B_B'.and.td=='___B'),'parent descriptors')
call require(mol%nmos==mol%nocc+mol%nvir,'incomplete parent orbital space')
call require(max(len_trim(mol%name),len_trim(mol%SCFcode))<=512,'parent context string too long')
pairs=mol%nmos*(mol%nmos+1)/2
call require(A%rows==mol%naux.and.A%cols==mol%naux,'parent metric shape')
call require(B%rows==pairs.and.B%cols==mol%naux,'parent RHS shape')
call require(X%rows==pairs.and.X%cols==mol%naux,'parent destination shape')
solve_id=solve_id+1
solve_header=(/mol%indx,mol%ndim,mol%naux,mol%nocc,mol%nvir,mol%nelectrons/)
solve_name=mol%name
solve_scf=mol%SCFcode
solve_active=.true.
call begin_event(mol,'SOLVE_BEGIN')
write(cp_unit,*) solve_id,mol%nmos,pairs,constraint,solver,iterations
write(cp_unit,'(a)') st,sd,tt,td
write(cp_unit,'(a)') trim(A%name),trim(A%filename),trim(B%name),trim(B%filename),trim(X%name),trim(X%filename)
write(cp_unit,*) A%rows,A%cols,B%rows,B%cols,X%rows,X%cols
call cp_fit_parameters(par)
call end_event()
end subroutine
subroutine nrc_solve_end(info)
integer, intent(in) :: info
if (.not.solve_active) return
call scalar_event(solve_header,solve_name,solve_scf,'SOLVE_END')
write(cp_unit,*) solve_id,info
call end_event()
solve_active=.false.
call require(info==0,'failed parent solve')
end subroutine
subroutine nrc_fit_begin(mol,A,B,X,par,st,sd,tt,td,constraint,solver,iterations,restart)
type(molecule), intent(in) :: mol
type(real_matrix), intent(in) :: A,B,X
type(df_parameters), intent(in) :: par
character(*), intent(in) :: st,sd,tt,td
integer, intent(in) :: constraint,solver,iterations
logical, intent(in) :: restart
call require(.not.fit_active,'nested fit context')
if (.not.selected(mol%indx)) return
! Other natural fits remain observed by existing metadata/row events, not this bounded LU hook.
if (par%df_type/=1.or.par%lambda/=1.0_dp) return
call require(par%eta==0.0_dp.and.par%gamma==0.0_dp,'unsupported NN constraint parameters')
call require(par%DFNormIndx==1.and.par%defined,'unsupported NN fit norm')
call require(.not.restart.and.solver==0.and.iterations==0,'fresh unrefined LU required')
call require(constraint==1.or.constraint==2,'unsupported constraint type')
call require(st=='Sc__'.and.tt=='T_MO','unexpected NN integral types')
call require(len_trim(sd)==4.and.len_trim(td)==4,'invalid integral descriptors')
call require((sd=='_A_A'.and.td=='___A').or.(sd=='_B_B'.and.td=='___B'),'NN descriptor mismatch')
call require(mol%nmos==mol%nocc+mol%nvir,'incomplete NN orbital space')
call require(max(len_trim(mol%name),len_trim(mol%SCFcode))<=512,'context string too long')
call require(max(len_trim(A%filename),len_trim(B%filename),len_trim(X%filename))<=512,'fit filename too long')
fit_header=(/mol%indx,mol%ndim,mol%naux,mol%nocc,mol%nvir,mol%nelectrons/)
fit_name=mol%name
fit_scf=mol%SCFcode
fit_files=(/A%filename,B%filename,X%filename/)
call require(all(len_trim(fit_files)>0),'empty fit filename')
fit_nmos=mol%nmos
fit_pairs=fit_nmos*(fit_nmos+1)/2
call require(A%rows==mol%naux.and.A%cols==mol%naux,'fit A dimensions')
call require(B%rows==fit_pairs.and.B%cols==mol%naux,'fit B dimensions')
fit_id=fit_id+1
fit_next=1
fit_blocks=0
fit_active=.true.
call scalar_event(fit_header,fit_name,fit_scf,'FIT_BEGIN')
write(cp_unit,*) fit_id,fit_nmos,fit_pairs,constraint,solver,iterations
write(cp_unit,'(a)') st,sd,tt,td
write(cp_unit,'(a)') trim(A%name),trim(A%filename),trim(B%name),trim(B%filename),trim(X%name),trim(X%filename)
write(cp_unit,*) A%rows,A%cols,B%rows,B%cols,X%rows,X%cols
call cp_fit_parameters(par)
call end_event()
end subroutine
subroutine nrc_fit_block(A,B,X,ta,tb,iterations,original_rows,original_cols,first,last,first_call)
type(real_matrix), intent(in) :: A,B,X
character(*), intent(in) :: ta,tb
integer, intent(in) :: iterations,original_rows,original_cols,first,last
logical, intent(in) :: first_call
integer :: width
if (.not.fit_active) return
call require(A%filename==fit_files(1).and.B%filename==fit_files(2),'LU input context mismatch')
call require(X%filename==fit_files(3),'LU output context mismatch')
call require(ta=='N'.and.tb=='T'.and.iterations==0,'LU policy mismatch')
call require(original_rows==fit_pairs.and.original_cols==fit_header(3),'original RHS dimensions')
call require(first==fit_next.and.last>=first.and.last<=fit_pairs,'noncontiguous RHS coverage')
call require(first_call.eqv.(first==1),'first-block flag mismatch')
width=last-first+1
call resident(A,'pre-LU A')
call require(A%rows==fit_header(3).and.A%cols==fit_header(3),'pre-LU A dimensions')
call require(B%in_memory.and.allocated(B%matrix),'RHS block not naturally resident')
call require(size(B%matrix,1)==fit_header(3).and.size(B%matrix,2)>=width,'RHS block dimensions')
! Deliberately do not call resident(B): the unused allocation tail may be undefined.
call require(all(ieee_is_finite(B%matrix(:,1:width))),'nonfinite valid RHS block')
fit_blocks=fit_blocks+1
call scalar_event(fit_header,fit_name,fit_scf,'FIT_A')
write(cp_unit,*) fit_id,fit_blocks
call cp_matrix('MATRIX',A%matrix)
call end_event()
call scalar_event(fit_header,fit_name,fit_scf,'FIT_RHS_BLOCK')
write(cp_unit,*) fit_id,fit_blocks,first,last,fit_pairs,size(B%matrix,2),width
write(cp_unit,'(a)') ta,tb
write(cp_unit,*) original_rows,original_cols
call cp_matrix('MATRIX',B%matrix(:,1:width))
call end_event()
fit_next=last+1
end subroutine
subroutine nrc_fit_end(info)
integer, intent(in) :: info
if (.not.fit_active) return
fit_active=.false.
call scalar_event(fit_header,fit_name,fit_scf,'FIT_END')
write(cp_unit,*) fit_id,info,fit_blocks,fit_next-1,fit_pairs
call end_event()
call require(info==0.and.fit_next==fit_pairs+1,'failed or incomplete captured NN solve')
end subroutine
subroutine nrc_hessian_tensor(mol,I,hessian,cx)
type(molecule), intent(in) :: mol
type(twoeint), intent(in) :: I
character(*), intent(in) :: hessian
real(dp), intent(in) :: cx
integer :: no,nv
if (.not.selected(mol%indx)) return
call require(I%defined.and.I%done.and.I%open,'Hessian integral not open and complete')
call require(.not.I%generalized,'generalized Hessian integrals unsupported')
call require(hessian=='h1'.or.hessian=='h2','unknown Hessian consumer')
call require(I%desc=='AAAA'.or.I%desc=='BBBB','nonmonomer Hessian tensor')
no=mol%nocc
nv=mol%nvir
call resident(I%int,'Hessian integral consumer')
select case(I%type)
case('OVOV')
  call require(all(I%map==2).and.all(I%alpha==no),'OVOV mapping mismatch')
  call require(I%int%rows==no*nv.and.I%int%cols==no*nv,'OVOV dimensions')
case('VVOO')
  call require(all(I%map==1).and.all(I%alpha==0),'VVOO mapping mismatch')
  call require(I%int%rows==nv*(nv+1)/2.and.I%int%cols==no*(no+1)/2,'VVOO dimensions')
case default
  call require(.false.,'unexpected Hessian integral type')
end select
call begin_event(mol,'HESSIAN_TENSOR')
write(cp_unit,'(a)') hessian,I%type,I%desc,trim(I%int%name),trim(I%int%filename)
write(cp_unit,*) I%map,I%alpha,merge(1,0,I%done),merge(1,0,I%generalized),merge(1,0,I%open),I%switch
call cp_vector('EXCHANGE',(/cx/))
call cp_matrix('MATRIX',I%int%matrix)
call end_event()
end subroutine
subroutine nrc_kernel_policy(mol,branch,cutoff,batch)
type(molecule), intent(in) :: mol
character(*), intent(in) :: branch
real(dp), intent(in) :: cutoff
integer, intent(in) :: batch
if (.not.selected(mol%indx)) return
call require(branch=='ALDA'.or.branch=='ALDAX','unknown numerical kernel branch')
call begin_event(mol,'NUMERICAL_KERNEL_POLICY')
write(cp_unit,'(a)') branch
write(cp_unit,*) batch
call cp_vector('KERNEL_INTEGRAL_CUTOFF',(/cutoff/))
call end_event()
end subroutine
subroutine nrc_matrix(mol,label,m,extra)
type(molecule), intent(in) :: mol
character(*), intent(in) :: label
type(real_matrix), intent(in) :: m
real(dp), intent(in), optional :: extra(:)
if (.not.selected(mol%indx)) return
call resident(m,label)
call begin_event(mol,label)
write(cp_unit,'(a)') trim(m%name)
write(cp_unit,'(a)') trim(m%filename)
if (present(extra)) then
  call require(size(extra)==2,'expected two matrix metadata values')
  call cp_vector('EXTRA',extra)
else
  call cp_vector('EXTRA',(/0.0_dp,0.0_dp/))
endif
call cp_matrix('MATRIX',m%matrix)
call end_event()
end subroutine
subroutine nrc_orbitals(mol)
type(molecule), intent(in) :: mol
if (.not.selected(mol%indx)) return
call resident(mol%C,'ORBITALS C')
call require(mol%E%in_memory.and.allocated(mol%E%vector),'energies not naturally resident')
call require(size(mol%C%matrix,1)==mol%ndim,'MAIN dimension mismatch')
call require(size(mol%C%matrix,2)>=mol%nocc+mol%nvir,'incomplete orbital coefficients')
call require(size(mol%E%vector)>=mol%nocc+mol%nvir,'incomplete energies')
call begin_event(mol,'ORBITALS')
call cp_matrix('C',mol%C%matrix(:,1:mol%nocc+mol%nvir))
call cp_vector('ENERGIES',mol%E%vector(1:mol%nocc+mol%nvir))
call end_event()
end subroutine
subroutine nrc_energies(mol,frac)
type(molecule), intent(in) :: mol
real(dp), intent(in) :: frac
if (.not.selected(mol%indx)) return
call require(mol%E%in_memory.and.allocated(mol%E%vector),'diagonal energies not naturally resident')
call require(size(mol%E%vector)>=mol%nocc+mol%nvir,'incomplete diagonal energies')
call begin_event(mol,'DIAGONAL_ENERGIES')
call cp_vector('DIAGONAL_FRACTION',(/frac/))
call cp_vector('ENERGIES',mol%E%vector(1:mol%nocc+mol%nvir))
call cp_basis('MAIN_BASIS',mol%main)
call cp_basis('AUX_BASIS',mol%aux)
call end_event()
end subroutine
subroutine nrc_ov_row(mol,dest,parent,ar,ij)
type(molecule), intent(in) :: mol
type(real_matrix), intent(in) :: dest,parent
integer, intent(in) :: ar,ij
if (.not.selected(mol%indx)) return
call resident(parent,'OV producer row')
call require(size(parent%matrix,1)==1.and.size(parent%matrix,2)==mol%naux,'OV row dimensions')
call require(ar>=1.and.ar<=mol%nocc*mol%nvir,'invalid OV destination row')
call begin_event(mol,'OV_ROW')
write(cp_unit,'(a)') trim(dest%name)
write(cp_unit,'(a)') trim(dest%filename)
write(cp_unit,'(a)') trim(parent%name)
write(cp_unit,'(a)') trim(parent%filename)
write(cp_unit,*) ar,ij
call cp_vector('ROW',parent%matrix(1,:))
call end_event()
end subroutine
subroutine nrc_pair_row(mol,kind,dest,parent,row,ij)
type(molecule), intent(in) :: mol
character(*), intent(in) :: kind
type(real_matrix), intent(in) :: dest,parent
integer, intent(in) :: row,ij
integer :: count
if (.not.selected(mol%indx)) return
call require(kind=='OO'.or.kind=='VV','unsupported paired subset')
count=mol%nocc
if (kind=='VV') count=mol%nvir
call require(row>=1.and.row<=count*(count+1)/2,'paired subset row range')
call resident(parent,'paired subset producer row')
call require(size(parent%matrix,1)==1.and.size(parent%matrix,2)==mol%naux,'paired row dimensions')
call begin_event(mol,kind//'_ROW')
write(cp_unit,'(a)') trim(dest%name),trim(dest%filename),trim(parent%name),trim(parent%filename)
write(cp_unit,*) row,ij
call cp_vector('ROW',parent%matrix(1,:))
call end_event()
end subroutine
subroutine nrc_density_begin(mol)
type(molecule), intent(in) :: mol
if (.not.selected(mol%indx)) return
call require(.not.density_active,'nested kernel density context')
density_active=.true.
density_serial=0
density_file=''
end subroutine
subroutine nrc_density(mol,Doo,Rho)
type(molecule), intent(in) :: mol
type(real_matrix), intent(in) :: Doo
type(FuncExpansion), intent(in) :: Rho
if (.not.selected(mol%indx)) return
if (.not.density_active) return
call require(density_serial==0,'repeated kernel density producer')
call resident(Doo,'kernel density source')
call require(Doo%rows==mol%nocc*(mol%nocc+1)/2.and.Doo%cols==mol%naux,'density OO shape')
call require(Rho%D%in_memory.and.allocated(Rho%D%vector),'kernel density not naturally resident')
call require(Rho%MolIndx==mol%indx.and.Rho%AtomIndx==0,'kernel density molecule identity')
call require(Rho%Basis%size==mol%naux.and.size(Rho%D%vector)==mol%naux,'kernel density AUX shape')
call require(trim(Rho%WhichBasis)=='AUX1','kernel density basis role')
call require(len_trim(Doo%filename)<=512,'density source filename too long')
call begin_event(mol,'KERNEL_DENSITY')
density_serial=serial
density_file=Doo%filename
write(cp_unit,'(a)') trim(Doo%name),trim(Doo%filename),trim(Rho%name),trim(Rho%WhichBasis)
write(cp_unit,'(a)') trim(Rho%D%name),trim(Rho%D%filename)
write(cp_unit,*) Rho%MolIndx,Rho%AtomIndx,Rho%Basis%size
call cp_vector('COEFFICIENTS',Rho%D%vector)
call end_event()
end subroutine
subroutine nrc_density_end(mol)
type(molecule), intent(in) :: mol
if (.not.selected(mol%indx)) return
call require(density_active.and.density_serial>0,'missing kernel density producer')
density_active=.false.
end subroutine
subroutine cp_fit_parameters(par)
type(df_parameters), intent(in) :: par
call require(par%defined,'undefined fit metadata')
write(cp_unit,*) par%DFNormIndx,par%option,par%df_type,merge(1,0,par%done)
call cp_vector('FIT_PARAMETERS',(/par%lambda,par%eta,par%gamma,par%gamma_DeltaZ/))
write(cp_unit,*) merge(1,0,par%gamma_H_only),merge(1,0,par%constrained)
end subroutine
subroutine nrc_fit_metadata(mol,D,par)
type(molecule), intent(in) :: mol
type(real_matrix), intent(in) :: D
type(df_parameters), intent(in) :: par
if (.not.selected(mol%indx)) return
call begin_event(mol,'FIT_METADATA')
write(cp_unit,'(a)') trim(D%name)
write(cp_unit,'(a)') trim(D%filename)
write(cp_unit,*) D%rows,D%cols
call cp_fit_parameters(par)
call end_event()
end subroutine
subroutine nrc_direct_guard(mol,desc,restart)
type(molecule), intent(in) :: mol
character(*), intent(in) :: desc
logical, intent(in) :: restart
if (.not.selected(mol%indx)) return
call require(.not.restart,'restart response capture unsupported')
call require(desc(1:2)/='OV','fresh direct OV solve needs a separate block producer hook')
end subroutine
subroutine nrc_kernel(mol,Ker,Doo,constrained)
type(molecule), intent(in) :: mol
type(real_matrix), intent(in) :: Ker,Doo
logical, intent(in) :: constrained
if (.not.selected(mol%indx)) return
call resident(Ker,'kernel producer')
call require(.not.density_active.and.density_serial>0,'missing completed kernel density context')
call require(Doo%filename==density_file,'kernel density source mismatch')
call begin_event(mol,'KERNEL_SOURCE')
write(cp_unit,'(a)') trim(Ker%name)
write(cp_unit,'(a)') trim(Ker%filename)
write(cp_unit,'(a)') trim(Doo%name)
write(cp_unit,'(a)') trim(Doo%filename)
write(cp_unit,*) merge(1,0,constrained)
write(cp_unit,*) density_serial
call end_event()
call nrc_matrix(mol,'KERNEL',Ker)
end subroutine
subroutine nrc_projection(mol,D,Ker,constrained,par)
type(molecule), intent(in) :: mol
type(real_matrix), intent(in) :: D,Ker
logical, intent(in) :: constrained
type(df_parameters), intent(in) :: par
if (.not.selected(mol%indx)) return
! Metadata only: matmult restores operand entry residency; do not open either operand.
call require(par%defined,'undefined fit metadata')
call require(D%rows==mol%nocc*mol%nvir.and.D%cols==mol%naux,'transition dimensions')
call require(Ker%rows==mol%naux.and.Ker%cols==mol%naux,'kernel dimensions')
call begin_event(mol,'PROJECTION')
write(cp_unit,*) merge(1,0,constrained)
write(cp_unit,'(a)') trim(D%name)
write(cp_unit,'(a)') trim(D%filename)
write(cp_unit,'(a)') trim(Ker%name)
write(cp_unit,'(a)') trim(Ker%filename)
call cp_fit_parameters(par)
call end_event()
end subroutine
subroutine nrc_policy(mol,kind,cx,solver,iterations,df,df_int,alda,functional_cx,hessian_code,constrained)
type(molecule), intent(in) :: mol
character(*), intent(in) :: kind,hessian_code
real(dp), intent(in) :: cx,functional_cx
integer, intent(in) :: solver,iterations
logical, intent(in) :: df,df_int,alda,constrained
if (.not.selected(mol%indx)) return
call require(trim(kind)=='cks'.and.trim(hessian_code)=='internal','internal CKS capture required')
call require(ieee_is_finite(cx).and.ieee_is_finite(functional_cx),'nonfinite exchange fractions')
call require(df.and.df_int,'DF response and DF integrals required')
call require(solver==0.and.iterations>=0,'LU response required')
old_seen=.true.
call begin_event(mol,'POLICY')
write(cp_unit,'(a)') trim(kind)
write(cp_unit,'(a)') trim(hessian_code)
write(cp_unit,*) solver,iterations,merge(1,0,df),merge(1,0,df_int),merge(1,0,alda),merge(1,0,constrained)
call cp_vector('EXCHANGE',(/cx,functional_cx/))
call end_event()
end subroutine
subroutine nrc_result(mol,m,omega2,new_driver)
type(molecule), intent(in) :: mol
type(real_matrix), intent(in) :: m
real(dp), intent(in) :: omega2
logical, intent(in) :: new_driver
if (.not.selected(mol%indx)) return
call require(old_seen.and..not.new_driver,'old/internal response dispatch required')
call require(ieee_is_finite(omega2).and.omega2<=0.0_dp,'imaginary or static frequency required')
call require(m%rows==mol%naux.and.m%cols==mol%naux,'response dimensions')
call nrc_matrix(mol,'CDF',m,(/omega2,0.0_dp/))
end subroutine
'''
MODULE += 'subroutine cp_matrix'+base.CAPTURE_MODULE.split('subroutine cp_matrix',1)[1].split('end module',1)[0]
MODULE += 'end module isapol_response_capture\n\n'


def instrument(smat, prop, utilities, polar, makefile, dfdata, dfutilities, dfmonomer, dfints, orbitals, matrixops, numerical, expansion,
               intutilities=None, intsfordf=None, intops=None):
    patch = scaffold.patch_routine
    smat = patch(smat, 'make_s_matrix_coulomb', [
        ('implicit none', 'use isapol_response_capture, only: nrc_matrix\nimplicit none'),
        ('if (debug_matrices) call matout_types(S)',
         "call nrc_matrix(mol,'J',S,(/integral_cutoff,dummy_s_exponent/))\nif (debug_matrices) call matout_types(S)")])
    for name, label in [('make_h1','H1'),('make_h2','H2')]:
        anchor = f'call close_type({label},info,transpose=.false.,forcewrite=.true.,release=.false.,debug=debug2)'
        prop = patch(prop, name, [
            ('implicit none','use isapol_response_capture, only: nrc_matrix\n implicit none'),
            (anchor, f"call nrc_matrix(mol,'{label}',{label})\n {anchor}")])
    prop = patch(prop, 'make_h1', [
        ('      H1%matrix = 4.0_dp*(1.0_dp-CxKernel)*H1%matrix',
         "      call nrc_matrix(mol,'KERNEL_OVOV_RAW',H1,(/CxKernel,merge(1.0_dp,0.0_dp,kernel_alda)/))\n"
         '      H1%matrix = 4.0_dp*(1.0_dp-CxKernel)*H1%matrix')])
    prop = patch(prop, 'add_2e_ints_to_Hessians', [
        ('implicit none','use isapol_response_capture, only: nrc_hessian_tensor\n implicit none'),
        (' !Add the terms H1: + 4 (ar|bs) - Cx*(br|as)',
         ' call nrc_hessian_tensor(mol,Iovov,h1orh2,Cx)\n !Add the terms H1: + 4 (ar|bs) - Cx*(br|as)'),
        ('   !Add the terms H1: - Cx*(rs|ab) == - Cx*(ab|rs)',
         '   call nrc_hessian_tensor(mol,Ivvoo,h1orh2,Cx)\n   !Add the terms H1: - Cx*(rs|ab) == - Cx*(ab|rs)')])
    numerical = patch(numerical, 'Chi_fxc_Chi_ALDAKernel', [
        ('implicit none','use isapol_response_capture, only: nrc_kernel_policy\nimplicit none'),
        ("call my_timer('enter',this_routine,debug=debug)",
         "call my_timer('enter',this_routine,debug=debug)\n"
         'call nrc_kernel_policy(mol,PropagatorType,kernel_integral_cutoff,MaxPointsBatch)')])
    prop = patch(prop, 'adddiag', [
        ('implicit none','use isapol_response_capture, only: nrc_energies\n implicit none'),
        ("call my_timer('enter',this_routine,debug=debug1)",
         "call my_timer('enter',this_routine,debug=debug1)\n call nrc_energies(mol,frac)")])
    prop = patch(prop, 'densfit_prop', [
        ('implicit none','use isapol_response_capture, only: nrc_policy\n implicit none'),
        ('call init_prop(mol)',
         'call init_prop(mol)\n call nrc_policy(mol,PropagatorType,CxKernel,PropSolver,LUiterations,useDF,use_df_int, &\n   kernel_alda,CxFunctional,dft_code,use_constraints)')])
    utilities = patch(utilities, 'make_kernel_OVOV_ALDA', [
        ('use df_data, only : Dov, Dov_c','use df_data, only : Dov, Dov_c, Dov_par, Dov_c_par'),
        ('implicit none','use isapol_response_capture, only: nrc_projection\nimplicit none'),
        ('nullify(D,Ker)\ncall destroy_type(TMP)',
         '''if (UseConstrainedDF) then
  call nrc_projection(mol,D,Ker,UseConstrainedDF,Dov_c_par(mol%indx))
else
  call nrc_projection(mol,D,Ker,UseConstrainedDF,Dov_par(mol%indx))
endif
nullify(D,Ker)
call destroy_type(TMP)''')])
    polar = patch(polar, 'run_polarizability', [
        ('implicit none','use isapol_response_capture, only: nrc_result\nimplicit none'),
        ("call open_type(DFprop,info)\ncall check_info(info,'open_type',this_routine,__LINE__)",
         "call open_type(DFprop,info)\ncall check_info(info,'open_type',this_routine,__LINE__)\ncall nrc_result(mol,DFprop,omega2,UseNewProp)")])
    dfutilities = patch(dfutilities, 'fill_Aov', [
        ('implicit none','use isapol_response_capture, only: nrc_ov_row\n  implicit none'),
        ('      !write this row of Afull into Aov%filename:',
         '      call nrc_ov_row(mol,Aov,Afull,ar,ij)\n      !write this row of Afull into Aov%filename:')])
    for kind, sub, row in [('OO','Aoo','ab'),('VV','Avv','rs')]:
        dfutilities = patch(dfutilities, 'fill_'+sub, [
            ('implicit none','use isapol_response_capture, only: nrc_pair_row\n  implicit none'),
            ('      !write this row of Afull into '+sub+'%filename:',
             f"      call nrc_pair_row(mol,'{kind}',{sub},Afull,{row},ij)\n      !write this row of Afull into {sub}%filename:")])
    for name in ('Doo','Dvv','Doo_c'):
        setter=f'call set_type_df_parameters({name}_par(mol%indx),mol,lambda,eta,gamma,gamma_DeltaZ,gamma_H_only,df_type)'
        dfmonomer=patch(dfmonomer,'df_monomer',[(setter,setter+f'\n    call nrc_fit_metadata(mol,{name}(mol%indx),{name}_par(mol%indx))')])
    dfmonomer = patch(dfmonomer, 'df_monomer', [
        ('implicit none\ntype(molecule), intent(inout) :: mol',
         'use isapol_response_capture, only: nrc_fit_metadata\nimplicit none\ntype(molecule), intent(inout) :: mol'),
        ('call set_type_df_parameters(Dov_par(mol%indx),mol,lambda,eta,gamma,gamma_DeltaZ,gamma_H_only,df_type)',
         'call set_type_df_parameters(Dov_par(mol%indx),mol,lambda,eta,gamma,gamma_DeltaZ,gamma_H_only,df_type)\n    call nrc_fit_metadata(mol,Dov(mol%indx),Dov_par(mol%indx))'),
        ('call set_type_df_parameters(Dov_c_par(mol%indx),mol,lambda,eta,gamma,gamma_DeltaZ,gamma_H_only,df_type)',
         'call set_type_df_parameters(Dov_c_par(mol%indx),mol,lambda,eta,gamma,gamma_DeltaZ,gamma_H_only,df_type)\n    call nrc_fit_metadata(mol,Dov_c(mol%indx),Dov_c_par(mol%indx))')])
    dfmonomer = patch(dfmonomer, 'do_DF_monomer', [
        ('implicit none','use isapol_response_capture, only: nrc_fit_metadata,nrc_direct_guard,nrc_fit_begin,nrc_fit_end\n'
         '  use isapol_response_capture, only: nrc_solve_begin,nrc_solve_end\n'
         '  use df_parameter_module, only: nrc_solver=>solver,nrc_iterations=>iterations,nrc_constraint=>ConstraintType\n  implicit none'),
        ('call solve_df_equations(S%int,T%int,D,info)',
         'call nrc_solve_begin(mol,S%int,T%int,D,par,Stype,Sdesc,Ttype,Tdesc, &\n'
         '       nrc_constraint,nrc_solver,nrc_iterations,restart)\n'
         '  call nrc_fit_begin(mol,S%int,T%int,D,par,Stype,Sdesc,Ttype,Tdesc, &\n'
         '       nrc_constraint,nrc_solver,nrc_iterations,restart)\n'
         '  call solve_df_equations(S%int,T%int,D,info)\n  call nrc_fit_end(info)\n  call nrc_solve_end(info)'),
        ('    !Do the DF:', '    call nrc_direct_guard(mol,Tdesc,restart)\n    !Do the DF:'),
        ('call set_type_df_parameters(Dpar,mol,lambda,eta,gamma,gamma_DeltaZ,gamma_H_only,df_type)',
         'call set_type_df_parameters(Dpar,mol,lambda,eta,gamma,gamma_DeltaZ,gamma_H_only,df_type)\n  call nrc_fit_metadata(mol,D,Dpar)')])
    dfints = patch(dfints, 'kernel_integrals', [
        ('implicit none','use isapol_response_capture, only: nrc_kernel,nrc_density_begin,nrc_density_end\n    implicit none'),
        ('    call Rho_Doo2FuncExpansion(Doo,mol,Rho,info)',
         '    call nrc_density_begin(mol)\n    call Rho_Doo2FuncExpansion(Doo,mol,Rho,info)'),
        ("    call check_info(info,'Rho_Doo2FuncExpansion',this_routine,__LINE__,.false.)\n    if (info/=0) then; info = -2; return; endif",
         "    call check_info(info,'Rho_Doo2FuncExpansion',this_routine,__LINE__,.false.)\n    if (info/=0) then; info = -2; return; endif\n    call nrc_density_end(mol)"),
        ('    call close_type(KerInt%Int,info,transpose=.false.,release=.true.,&',
         '    call nrc_kernel(mol,KerInt%Int,Doo,use_df_with_constraints)\n    call close_type(KerInt%Int,info,transpose=.false.,release=.true.,&')])
    orbitals = patch(orbitals, 'read_mos_energies_ascii_2', [
        ('implicit none','use isapol_response_capture, only: nrc_orbitals\n  implicit none'),
        ('  call close_type_primary(mol%C,info,release=.false.,forcewrite=.true.)',
         '  call nrc_orbitals(mol)\n  call close_type_primary(mol%C,info,release=.false.,forcewrite=.true.)')])
    matrixops = patch(matrixops, 'lineq_solver_lu', [
        ('implicit none\n  type (real_matrix), intent(inout) :: A, B',
         'use isapol_response_capture, only: nrc_fit_block\n  implicit none\n  type (real_matrix), intent(inout) :: A, B'),
        ('    if (first_call) then',
         '    call nrc_fit_block(A,B,X,transA,transB,iterations,s_rowB,s_colB,start_col,end_col,first_call)\n'
         '    if (first_call) then')])
    expansion=patch(expansion,'Rho_Doo2FuncExpansion',[
        ('implicit none','use isapol_response_capture, only: nrc_density\nimplicit none'),
        ('Rho%D%vector = 2.0_dp*Rho%D%vector',
         'Rho%D%vector = 2.0_dp*Rho%D%vector\ncall nrc_density(mol,Doo,Rho)')])
    deps = ('\n# Both .o and .mod rules can compile source in this Makefile.\n'
            '# The low-level df_data.o emits the observer module as a side effect.\n'
            'isapol_response_capture.mod: df_data.o\n'
            'df_Smat.o densfit_prop.o prop_utilities.o polarizability.o df_utilities.o df_monomer.o '
            'df_integrals.o molecular_orbitals.o: isapol_response_capture.mod\n'
            'df_Smat.mod densfit_prop_parameters.mod densityfitted_propagator.mod PropUtilities.mod '
            'polarizability_module.mod df_utilities.mod df_monomer_module.mod df_integrals.mod '
            'molecular_orbitals.mod: isapol_response_capture.mod\n'
            'matrix_operations_types.o matrix_operations_types.mod: isapol_response_capture.mod\n'
            'num_integrals.o NumericalIntegrals.mod: isapol_response_capture.mod\n'
            'function_expansion_operations.o FunctionExpansionOperations.mod: isapol_response_capture.mod\n')
    if intutilities is not None:
        smat, dfints, intutilities, intsfordf = instrument_cache(smat, dfints, intutilities, intsfordf)
        deps += ('df_integral_utilities.o df_integral_utilities.mod: isapol_response_capture.mod\n'
                 'df_integrals_for_df.o df_integrals_for_df.mod: isapol_response_capture.mod\n')
    result = (smat, prop, utilities, polar, makefile+deps, MODULE+dfdata, dfutilities, dfmonomer,
              dfints, orbitals, matrixops, numerical, expansion)
    if intops is not None:
        intops = patch(intops, 'open_df_int_2e', [
            ('implicit none', 'use isapol_response_capture, only: nrc_guard\n'
             'use molecules, only: molA,molB\nimplicit none'),
            ('  if (I%generalized) then',
             cache_guard('molA','generalized/nonzero-switch tensor request','I%generalized.or.switch/=0')+'\n'+
             cache_guard('molB','generalized/nonzero-switch tensor request','I%generalized.or.switch/=0')+'\n'+
             '  if (I%generalized) then')])
        # Both suffix rules compile this source.
        result = result[:4] + (result[4] +
            'df_int_operations.o df_int_operations.mod: isapol_response_capture.mod\n',) + result[5:]
    if intutilities is None:
        return result
    return result + (intutilities, intsfordf) + (() if intops is None else (intops,))


def cache_array(items):
    return '(/' + ', &\n    '.join(items) + '/)'


def cache_header(m):
    return [cache_array([m+'%'+f for f in ('indx','ndim','naux','nocc','nvir','nelectrons')]),
            m+'%name', m+'%SCFcode']


def cache_call(name, args):
    return 'call '+name+'( &\n  '+', &\n  '.join(args)+')'


def cache_guard(m, reason, unsafe):
    return cache_call('nrc_guard', cache_header(m)+[repr(reason), unsafe])


def cache_storage(m):
    return [m+'%name', m+'%filename', cache_array([m+'%rows',m+'%cols'])]


def cache_metric_args(s):
    return [s+'%type',s+'%desc'] + cache_storage(s+'%int') + [cache_array([
        f'merge(1,0,{s}%{f})' for f in ('defined','Rotate','ForceRotate')]+[s+'%switch'])]


def cache_output_args(i):
    return ['indxA','indxB',i+'%type',i+'%desc'] + cache_storage(i+'%int') + [
        cache_array([i+'%map(1)',i+'%map(2)',i+'%alpha(1)',i+'%alpha(2)']),
        cache_array([f'merge(1,0,{i}%{f})' for f in ('defined','generalized','Rotate','ForceRotate')]+[i+'%switch'])]


def cache_operand_args(d, p):
    return cache_storage(d) + [
        cache_array([p+'%'+f for f in ('DFNormIndx','option','df_type')]+[f'merge(1,0,{p}%done)']),
        cache_array([p+'%'+f for f in ('lambda','eta','gamma','gamma_DeltaZ')]),
        cache_array([f'merge(1,0,{p}%{f})' for f in ('gamma_H_only','constrained')])]


def instrument_cache(smat, dfints, utilities, ford):
    """Exact routine anchors; new APIs take only scalar/string field values."""
    patch = scaffold.patch_routine
    policy_use = ('use integral_parameters, only: integral_cutoff,dummy_s_exponent\n'
                  'use df_parameter_module, only: nrc_norm=>DFNormIndx\n')
    policy = ['nrc_norm','integral_cutoff','dummy_s_exponent']
    for routine, owner, route in [('df_make_s_matrix','S','DF_S'),
                                  ('df_make_s_matrix_constraints','Sc','DF_SC')]:
        request = cache_call('nrc_metric_request',cache_header('mol')+['indxA','indxB',repr(route),
            owner+'%int%name',owner+'%int%filename',owner+'%done','S%done']+cache_metric_args('S')+policy)
        end = cache_call('nrc_metric_end',cache_header('mol')+[
            owner+'%int%name',owner+'%int%filename','S%int%name','S%int%filename',owner+'%done','info'])
        failure = 'if (info.lt.0) then; info = -1; return; endif'
        utilities = patch(utilities, routine, [
            ('implicit none','use isapol_response_capture, only: nrc_metric_request,nrc_metric_end\n'
             'use molecules, only: indxA,indxB\n'+policy_use+'implicit none'),
            (f'if ({owner}%done) return',request+f'\n  if ({owner}%done) return'),
            (failure,'if (info<0) then\n'+end+'\nendif\n  '+failure),
            (f'{owner}%done = .true.',f'{owner}%done = .true.\n'+end)])
    result = cache_call('nrc_metric_result',cache_header('mol')+cache_storage('S')+['info'])
    begin = cache_call('nrc_metric_begin',cache_header('mol')+cache_storage('S')+
                       ['DFNormIndx','integral_cutoff','dummy_s_exponent','nrc_rp','nrc_rv'])
    smat = patch(smat,'make_s_matrix',[
        ('implicit none','use isapol_response_capture, only: nrc_metric_begin,nrc_metric_result\nimplicit none'),
        ("character(len=*), parameter :: this_routine = 'make_s_matrix'",
         "integer :: nrc_rp,nrc_rv\ncharacter(len=*), parameter :: this_routine = 'make_s_matrix'"),
        ('select case(DFNormIndx)',
         'nrc_rp=0\nnrc_rv=0\nif (present(recalculate)) then\n'
         '  nrc_rp=1\n  nrc_rv=merge(1,0,recalculate)\nendif\n'+begin+'\nselect case(DFNormIndx)'),
        ("  call check_info(info,'make_s_matrix_coulomb'",result+"\n  call check_info(info,'make_s_matrix_coulomb'"),
        ("  call check_info(info,'make_s_matrix_overlap'",result+"\n  call check_info(info,'make_s_matrix_overlap'")])
    smat=patch(smat,'make_s_matrix_coulomb',[
        ('only: nrc_matrix','only: nrc_matrix,nrc_metric_link'),
        ("call nrc_matrix(mol,'J',S,(/integral_cutoff,dummy_s_exponent/))",
         "call nrc_matrix(mol,'J',S,(/integral_cutoff,dummy_s_exponent/))\n"+
         cache_call('nrc_metric_link',cache_header('mol')))])
    checks=[]
    for side in ('A','B'):
        checks += [f'if (mol{side}%defined) then',cache_call('nrc_cache_check',cache_header('mol'+side)+[
            'indxA','indxB','nrc_initialized','same'+side,'nrc_identity'+side,'same_df_norm']+policy),'endif']
    utilities=patch(utilities,'check_if_integrals_are_done',[
        ('implicit none','use isapol_response_capture, only: nrc_cache_check\n'
         'use molecules, only: indxA,indxB\n'+policy_use+'implicit none'),
        ('logical :: sameA,','logical :: nrc_initialized,nrc_identityA,nrc_identityB\n  logical :: sameA,'),
        ('  info = 0','  nrc_initialized=int_par%defined\n  nrc_identityA=.true.\n  nrc_identityB=.true.\n'
         '  if (nrc_initialized) then\n'
         '    if (molA%defined) nrc_identityA=int_par%molA_indx==molA%indx.and.int_par%molA_name==molA%name\n'
         '    if (molB%defined) nrc_identityB=int_par%molB_indx==molB%indx.and.int_par%molB_name==molB%name\n'
         '  endif\n  info = 0'),
        ("  if (debug) then\n    print *,' Flags in check_if_integrals_are_done: '",
         '\n'.join(checks)+"\n  if (debug) then\n    print *,' Flags in check_if_integrals_are_done: '")])
    for name,mols in [('rotate_twoe2indx',('molA','molB')),('rotate_twoe2indx_left',('mol',)),
                      ('rotate_onee2indx',('molA','molB')),('rotate_oneeint',('mol',))]:
        utilities=patch(utilities,name,[
            ('implicit none','use isapol_response_capture, only: nrc_guard\nimplicit none'),
            ('if (.not.rotate) return','\n'.join(cache_guard(m,'rotation/ForceRotate','rotate') for m in mols)+
             '\n  if (.not.rotate) return')])
    ford=patch(ford,'make_integrals_for_df',[
        ('implicit none','use isapol_response_capture, only: nrc_guard\nimplicit none'),
        ('        !Now fill the Smat subsets',cache_guard('molA','unobserved dimer metric subset write','.true.')+'\n'+
         cache_guard('molB','unobserved dimer metric subset write','.true.')+'\n        !Now fill the Smat subsets')])
    dispatch=['nrc_rid=0','nrc_tp=0','nrc_tv=0','if (present(transpose_S)) then',
              '  nrc_tp=1','  nrc_tv=merge(1,0,transpose_S)','endif','select case(I%desc)']
    complete=['select case(I%desc)']
    for side in ('A','B'):
        m='mol'+side
        dispatch += [f"case('{side*4}')",f'if (nrc_selected({m}%indx)) then','select case(I%type)']
        for kind,left,right in [('OVOV','Dov','Dov'),('VVOO','Dvv','Doo')]:
            pa,pb=f'{left}_par(indx{side})',f'{right}_par(indx{side})'
            dispatch += [f"case('{kind}')",cache_guard(m,'undefined stored tensor parameters',
                         f'.not.{pa}%defined.or..not.{pb}%defined'),
                cache_call('nrc_dsd_request',cache_header(m)+cache_output_args('I')+['I%done','S%done']+
                    cache_metric_args('S')+cache_operand_args('Da',pa)+cache_operand_args('Db',pb)+
                    ['switch']+policy+['nrc_rid','nrc_tp','nrc_tv'])]
        dispatch += ['case default',cache_guard(m,'unsupported DSD tensor role','.true.'),'end select','endif']
        complete += [f"case('{side*4}')",cache_call('nrc_dsd_complete',cache_header(m)+cache_output_args('I')+
                     ['nrc_rid','info','I%done'])]
    dispatch += ['case default',cache_guard('molA','unsupported DSD descriptor','.true.'),
                 cache_guard('molB','unsupported DSD descriptor','.true.'),'end select']
    complete += ['end select']
    dfints=patch(dfints,'make_D_S_D',[
        ('implicit none','use isapol_response_capture, only: nrc_dsd_request,nrc_dsd_complete,nrc_guard\n'
         'use isapol_response_capture, only: nrc_selected=>selected\n'
         'use df_data, only: Dov_par,Dvv_par,Doo_par\n'+policy_use+'implicit none'),
        ('    type(real_matrix) :: TMP','integer :: nrc_rid,nrc_tp,nrc_tv\n    type(real_matrix) :: TMP'),
        ('    if (I%done) return','\n'.join(dispatch)+'\n    if (I%done) return'),
        ('    I%done = .true.','    I%done = .true.\nif (nrc_rid>0) then\n'+'\n'.join(complete)+'\nendif')])
    return smat,dfints,utilities,ford


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError('Destination must be new and outside source')
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True)
    if commit != PIN or dirty:
        raise ValueError('Clean pinned tracked reference required')
    paths = ['src/df_Smat.F90','src/densfit_prop.F90','src/prop_utilities.F90','src/polarizability.F90','Makefile_body',
             'src/df_data.f90','src/df_utilities.F90','src/df_monomer.F90','src/df_integrals.F90','src/molecular_orbitals.F90',
             'src/matrix_operations_types.F90','src/num_integrals.F90','src/function_expansion_operations.F90',
             'src/df_integral_utilities.F90','src/df_integrals_for_df.F90','src/df_int_operations.F90']
    patched = instrument(*[(source/p).read_text() for p in paths])
    destination.mkdir(parents=True)
    shutil.copytree(source/'src',destination/'src')
    for name in ['Makefile','Makefile_body','VERSION']:
        shutil.copy2(source/name,destination/name)
    shutil.copytree(source/'x86-64/gfortran/exe',destination/'x86-64/gfortran/exe',
                    ignore=shutil.ignore_patterns('*.o','*.mod','camcasp','casimir','cluster','process'))
    (destination/'bin').mkdir()
    shutil.copy2(source/'bin/version.py',destination/'bin/version.py')
    for path,text in zip(paths,patched):
        (destination/path).write_text(text)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(schema_version=1,event_schema=8,observer_validated=False,
                  reference_commit=commit,source_root=str(source),
                  source_sha256={p:sha(source/p) for p in paths},
                  instrumented_sha256={p:sha(destination/p) for p in paths},
                  harness_sha256=sha(Path(__file__)),scaffold_sha256=sha(Path(scaffold.__file__)),
                  serializer_sha256=sha(Path(base.__file__)),
                  limitations=['Preparation is not observer validation; require fresh traced/untraced runs and strict replay',
                               'Bounded fixed-geometry ordinary monomer OVOV/VVOO cache routes only',
                               'No quadrature-weight export',
                               'NN-derived OV rows only; fresh direct OV solves rejected',
                               'Actual kernel D must still be checked against response D selection',
                               'No native or end-to-end acceptance'])
    (destination/'response-capture-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camcasp',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.camcasp,args.destination),indent=2))
