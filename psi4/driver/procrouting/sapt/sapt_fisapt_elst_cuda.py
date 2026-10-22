"""Device-only fragment electrostatic algebra, not automatic FISAPT dispatch.

All occupied orbitals (including frozen) enter the matrix. Native JKFIT DFMO
factors and native point-charge potentials must be supplied by owned providers;
these routines neither generate CPU integrals nor copy factors to the host.
"""
from .sapt_dispersion_cuda_setup import DeviceMath


def _grid(size):
    return min(65535, (size+255)//256)


def diagonal_factors(session, value):
    """Extract C-order (Q,o,o) diagonal factors, preserving native metric gauge."""
    DeviceMath(session).require(value)
    if len(value.shape)!=3 or value.shape[1]!=value.shape[2] or min(value.shape)<1:
        raise ValueError("Positive square occupied DFMO factor blocks required")
    Q,O,_=value.shape
    output=session.empty((Q,O))
    session.launch("diagonal_factors",_grid(output.size),256,[Q,O,value,output])
    return output


def scatter_matrix(session, value, output, row0, col0, *, weight=1):
    """Add a dense device block into native atom/orbital slots on one stream."""
    m=DeviceMath(session);m.require(value);m.require(output)
    if (len(value.shape)!=2 or len(output.shape)!=2 or min(value.shape)<1
            or any(type(i) is not int for i in (row0,col0,weight))
            or min(row0,col0)<0 or abs(weight)>4
            or row0+value.shape[0]>output.shape[0]
            or col0+value.shape[1]>output.shape[1]
            or value.allocation is output.allocation):
        raise ValueError("Invalid fragment block placement/alias/weight")
    session.launch("scatter_matrix",_grid(value.size),256,
                   [*value.shape,output.shape[1],row0,col0,weight,value,output])


def sum_values(session,value):
    """GPU global sum; a subsequent scalar download is reporting, not arithmetic."""
    DeviceMath(session).require(value)
    output=session.empty((1,))
    session.launch("reduce_values",1,128,[value.size,value,output])
    return output


def nuclear_matrix(session,xyz,za,zb):
    m=DeviceMath(session)
    for value in (xyz,za,zb):m.require(value)
    if (len(xyz.shape)!=2 or xyz.shape[1]!=3 or xyz.shape[0]<1
            or za.shape!=(xyz.shape[0],) or zb.shape!=za.shape):
        raise ValueError("Full dimer coordinates/charge vectors required")
    N=xyz.shape[0];output=session.empty((N,N))
    session.launch("fragment_nuclear",_grid(N*N),256,[N,xyz,za,zb,output])
    return output


def electronic_pairs(session,qa,qb,output,natoms):
    """Add 4*(Q|aa).T@(Q|bb) without downloading reusable diagonal factors."""
    m=DeviceMath(session);m.require(qa);m.require(qb);m.require(output)
    if (type(natoms) is not int or natoms<1
            or len(qa.shape)!=2 or len(qb.shape)!=2
            or qa.shape[0]!=qb.shape[0] or min(*qa.shape,*qb.shape)<1
            or output.shape!=(natoms+qa.shape[1]+1,natoms+qb.shape[1]+1)):
        raise ValueError("Wrong JKFIT factor/matrix atom-orbital layout")
    with session.scope():
        block=session.gemm_at_b(qa,qb)
        scatter_matrix(session,block,output,natoms,natoms,weight=4)


def nuclear_electronic(session,potential,occupied,output,atom,*,side,natoms):
    """Add 2*diag(L.T V_atom L); native V must include negative nuclear charge."""
    m=DeviceMath(session)
    for value in (potential,occupied,output):m.require(value)
    if (side not in ("A","B") or type(atom) is not int or type(natoms) is not int
            or natoms<1 or not 0<=atom<natoms or len(occupied.shape)!=2
            or min(occupied.shape)<1
            or potential.shape!=(occupied.shape[0],occupied.shape[0])
            or len(output.shape)!=2 or min(output.shape)<natoms+2
            or output.shape[1 if side=="A" else 0]!=natoms+occupied.shape[1]+1):
        raise ValueError("Wrong nuclear/orbital placement or potential layout")
    with session.scope():
        product=session.gemm(potential,occupied)
        diagonal=diagonal_factors(session,session.gemm_at_b(occupied,product).view(
            (1,occupied.shape[1],occupied.shape[1])))
        if side=="A":
            scatter_matrix(session,diagonal,output,atom,natoms,weight=2)
        else:
            scatter_matrix(session,diagonal.view((occupied.shape[1],1)),
                           output,natoms,atom,weight=2)
