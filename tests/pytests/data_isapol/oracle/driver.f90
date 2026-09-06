! Reference dump of CamCASP's ISA integration grid and element table.
!
! Reads from stdin:
!     ns
!     n_r  n_a  k_mu  rscale
!     Z x y z            (ns lines, coordinates in bohr)
!
! Writes `radii.dat`  -- AtomProp for Z = 0 .. max_atoms (SPEC.md gate 5)
!        `grid.dat`   -- ns, ng; ns+1 atom offsets; ng lines of x y z w (gate 5b)
!
! See make_oracle.sh for how to build this.
program griddump
use precision, only : dp
use atoms, only : init_atoms, AtomProp, max_atoms
use atom_grids, only : grid, ng, make_grid, Lebedev, n_a, n_r, k_mu, start, rscale
implicit none
integer :: ns, i, info, u
integer, allocatable :: Z(:)
real(dp), allocatable :: c(:,:), radius(:)
character(200) :: arg

call init_atoms

! Element table, exactly as the grid sees it.  Written with 17 significant digits
! so that the float32 rounding of the Fortran literals (SPEC.md 3.5.3) is visible.
open(newunit=u, file='radii.dat', status='replace')
write(u,'(i0)') max_atoms
do i = 0, max_atoms
  write(u,'(i4,1x,a2,5es26.17e3)') i, AtomProp(i)%symbol, AtomProp(i)%Rslater, &
      AtomProp(i)%RvdwBondi, AtomProp(i)%RvdwGrimme, AtomProp(i)%C6Grimme,     &
      AtomProp(i)%Covalent
end do
close(u)

! read geometry from stdin: ns, then ns lines of "Z x y z" (bohr)
read *, ns
read *, n_r, n_a, k_mu, rscale
allocate(Z(ns), c(3,ns), radius(ns))
do i = 1, ns
  read *, Z(i), c(1,i), c(2,i), c(3,i)
  radius(i) = AtomProp(Z(i))%Rslater
end do
Lebedev = .true.

call make_grid(ns, Z, c, radius, info)
if (info /= 0) then
  print *, 'make_grid failed', info
  stop 1
end if

open(newunit=u, file='grid.dat', status='replace')
write(u,'(i0,1x,i0)') ns, ng
do i = 1, ns+1
  write(u,'(i0)') start(i)
end do
do i = 1, ng
  write(u,'(4es26.17e3)') grid(1,i), grid(2,i), grid(3,i), grid(4,i)
end do
close(u)
print '(a,i0)', 'ng = ', ng
end program
