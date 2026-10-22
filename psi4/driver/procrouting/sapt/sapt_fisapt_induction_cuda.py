"""Uncoupled directional fragment induction primitives, not full find dispatch.

Nuclear/electronic source assembly and exchange-induction target construction
are separate operations. No coupled response/CPU amplitude fallback is present.
"""
from .sapt_dispersion_cuda_setup import DeviceMath


def uncoupled_source(session,source,energies,rotation,targets,outputs,*,column):
    """Divide a canonical source, localize amplitudes, reduce one source column.

    Caller validates full occupied rotation/gaps once before its source loop.
    `source` is canonical (occ,vir); localized `targets['ind']` is wL and
    optional `targets['exch']` is uL. Occupied space includes frozen orbitals.
    `outputs` has corresponding (localizedOcc,sourceCount) device matrices.
    No download or host arithmetic occurs. All returned effects are borrowed
    writes into caller-owned outputs; temporary amplitudes are scope bounded.
    """
    if (set(energies)!={"o","v"} or set(targets) not in ({"ind"},{"ind","exch"})
            or set(outputs)!=set(targets) or type(column) is not int or column<0):
        raise ValueError("Canonical energies and matching directional components required")
    m=DeviceMath(session)
    inputs=(source,rotation,*energies.values(),*targets.values())
    for value in (*inputs,*outputs.values()):m.require(value)
    if len(source.shape)!=2:
        raise ValueError("A canonical occupied-virtual source matrix is required")
    O,V=source.shape
    shape=outputs["ind"].shape
    if (rotation.shape!=(O,O) or energies["o"].shape!=(O,) or energies["v"].shape!=(V,)
            or any(value.shape!=(O,V) for value in targets.values())
            or len(shape)!=2 or shape[0]!=O or column>=shape[1]
            or any(value.shape!=shape for value in outputs.values())
            or any(value.allocation is output.allocation for value in inputs for output in outputs.values())
            or ("exch" in outputs and outputs["ind"].allocation is outputs["exch"].allocation)):
        raise ValueError("Wrong directional induction layout/placement/alias")
    with session.scope():
        canonical=session.empty(source.shape)
        session.launch("make_single_amplitudes",min(65535,(source.size+255)//256),256,
                       [O,V,source,energies["o"],energies["v"],canonical])
        localized=session.gemm_at_b(rotation,canonical)
        session.launch("reduce_induction_source",min(65535,O),128,
                       [O,V,shape[1],column,localized,targets["ind"],targets.get("exch"),
                        outputs["ind"],outputs.get("exch")])
