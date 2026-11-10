"""FP64 GPU normalization of final fragment matrices, never host arithmetic."""
import numpy as np
from .sapt_dispersion_cuda_setup import DeviceMath


def scale_factor(session, values, *, total=None):
    """Exchange target/summed S² matrix, or (Ind+ExchInd+dHF)/(Ind+ExchInd).

    ``values`` contains final upstream scalar results, not factors or response
    amplitudes. Every zero denominator (including zero/zero) and every
    nonfinite ratio fails closed. No identity fallback or denominator clamp.
    """
    values=np.asarray(values)
    if (values.dtype!=np.float64 or values.shape!=((3,) if total is None else (1,))
            or not np.isfinite(values).all()):
        raise ValueError("Explicit finite FP64 final scaling scalars required")
    if total is not None:
        DeviceMath(session).require(total)
        if total.shape!=(1,):raise ValueError("Device exchange total must be a scalar")
    factor=session.empty((1,))
    with session.scope():
        inputs=session.upload(values)
        error=session.empty((1,))
        session.launch("fragment_scale_factor",1,1,
                       [1 if total is None else 0,inputs,inputs if total is None else total,factor,error])
        if float(session.download(error)[0])!=0.:
            raise FloatingPointError("Undefined/nonfinite GPU fragment scaling ratio")
    return factor


def apply_scale(session, value, factor):
    m=DeviceMath(session);m.require(value);m.require(factor)
    if len(value.shape)!=2 or factor.shape!=(1,):
        raise ValueError("A device matrix and validated scalar factor are required")
    output=session.empty(value.shape)
    session.launch("apply_fragment_scale",min(65535,(value.size+255)//256),256,
                   [value.size,value,factor,output])
    return output
