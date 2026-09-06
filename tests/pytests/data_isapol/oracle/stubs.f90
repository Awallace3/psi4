module precision
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: real_bytes = 8
end module precision

module parameters
  use precision
  implicit none
  real(dp), parameter :: pi = 3.1415926535897932384626433832795028841968_dp
  ! Values copied from CamCASP src/parameters.f90; see SPEC.md 3.5.3 on why
  ! a_o must be CamCASP's constant and not the modern CODATA one.
  real(dp), parameter :: a_o = 0.529177249_dp
  real(dp), parameter :: au2kJ = 2625.49962_dp
end module parameters

module common_routines
contains
  subroutine general_error(a,b)
    character(*) :: a
    integer :: b
    print *, 'general_error: ', a, b
    stop 1
  end subroutine
  subroutine check_allocate(info,what,who,line,fatal)
    integer :: info, line
    character(*) :: what, who
    logical :: fatal
    if (info /= 0) then
      print *, 'alloc fail ', what, who, line
      stop 1
    end if
  end subroutine
  subroutine check_deallocate(info,what,who,line,fatal)
    integer :: info, line
    character(*) :: what, who
    logical :: fatal
    if (info /= 0) then
      print *, 'dealloc fail ', what, who, line
      stop 1
    end if
  end subroutine
  subroutine internal_error(who,line,msg,fatal)
    character(*) :: who, msg
    integer :: line
    logical :: fatal
    print *, 'internal_error ', who, line, msg
    stop 1
  end subroutine
end module common_routines
