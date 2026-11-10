"""Owned CUDA numerical setup for canonical HF Disp20/Exch-Disp20.

Inputs are raw upstream HF matrices, not CPU-prepared dispersion intermediates.
NumPy is used only for host input validation/transport. Numerical setup and final
reductions are cuBLAS/CUDA. This module does not create a context or a provider.
The caller owns a CUDA scope containing the returned device allocations.
"""
import numpy as np

from .sapt_cuda_runtime import DeviceArray


class DeviceMath:
    """Small dispersion-specific interface over owned C-order FP64 buffers."""
    def __init__(self, session):
        self.session = session

    def require(self, value):
        if (not isinstance(value, DeviceArray) or value.dtype != np.float64
                or value.order != "C" or value.allocation.session is not self.session):
            raise ValueError("Expected C-order FP64 buffer from this CUDA session")
        # Check the lifetime even before building a view or allocating output.
        value.pointer
        return value

    def window(self, value, start=0, columns=None, *, transpose=False):
        self.require(value)
        if len(value.shape) != 2:
            raise ValueError("Matrix window requires a rank-two buffer")
        rows, stride = value.shape
        columns = stride-start if columns is None else columns
        if (type(start) is not int or type(columns) is not int
                or start < 0 or columns < 1 or start+columns > stride):
            raise ValueError("Invalid matrix column window")
        out = self.session.empty((columns,rows) if transpose else (rows,columns))
        self.session.launch("matrix_window",min(65535,(out.size+255)//256),256,
                            [rows,stride,start,columns,bool(transpose),value,out])
        return out

    def combine(self, left, right, alpha=1, beta=1):
        self.require(left); self.require(right)
        if (left.shape != right.shape or type(alpha) is not int or type(beta) is not int
                or abs(alpha) > 4 or abs(beta) > 4):
            raise ValueError("Dispersion combination requires matching shapes and small integer weights")
        out = self.session.empty(left.shape)
        self.session.launch("linear_combination",min(65535,(out.size+255)//256),256,
                            [out.size,alpha,beta,left,right,out])
        return out

    def pack_coefficients(self, matrices):
        """Concatenate transposed AO coefficient matrices on device for DFMO."""
        matrices = tuple(matrices)
        if not matrices:
            raise ValueError("DFMO coefficient list cannot be empty")
        for value in matrices:
            self.require(value)
            if len(value.shape) != 2 or value.shape[0] != matrices[0].shape[0]:
                raise ValueError("DFMO coefficients require one common AO dimension")
        nao = matrices[0].shape[0]
        out = self.session.empty((sum(value.shape[1] for value in matrices),nao))
        offset = 0
        for value in matrices:
            rows,columns = value.shape
            target = out.view((columns,rows),offset=offset)
            self.session.launch("matrix_window",min(65535,(value.size+255)//256),256,
                                [rows,columns,0,columns,True,value,target])
            offset += value.nbytes
        return out

    def product(self, *values):
        if len(values) < 2:
            raise ValueError("Matrix product requires at least two factors")
        for value in values:
            self.require(value)
        out = values[0]
        for value in values[1:]:
            out = self.session.gemm(out,value)
        return out

    def terms(self, *terms):
        """Sum (integer weight, device matrix) terms; no host arithmetic."""
        if len(terms) < 2:
            raise ValueError("Combination requires at least two terms")
        (alpha,left),(beta,right),*rest = terms
        out = self.combine(left,right,alpha,beta)
        for beta,right in rest:
            out = self.combine(out,right,1,beta)
        return out


def prepare_hf(session, matrices, energies, *, frozen_a=0, frozen_b=0):
    """Upload raw HF inputs and form all twelve spaces/eight corrections on GPU.

    Frozen indices are metadata. Input finite checks are validation, not a
    numerical fallback. The caller supplies spherical basis admission separately.
    """
    square = ("S","D_A","P_A","V_A","J_A","K_A",
              "D_B","P_B","V_B","J_B","K_B","K_O")
    orbital = ("Cocc_A","Cocc_B","Cvir_A","Cvir_B")
    if set(matrices) != set(square+orbital) or set(energies) != {"a","b","r","s"}:
        raise ValueError("Complete raw HF matrices and four orbital energy arrays required")
    if type(frozen_a) is not int or type(frozen_b) is not int or min(frozen_a,frozen_b) < 0:
        raise ValueError("Frozen occupied counts must be nonnegative integers")
    raw = {name:np.asarray(value) for name,value in matrices.items()}
    first = raw["S"]
    if first.ndim != 2 or first.shape[0] != first.shape[1] or first.shape[0] < 1:
        raise ValueError("Nonempty square AO overlap required")
    nao = first.shape[0]
    for name,value in raw.items():
        if (value.dtype != np.float64 or value.ndim != 2 or value.shape[0] != nao
                or value.shape[1] < 1 or (name in square and value.shape[1] != nao)
                or not np.isfinite(value).all()):
            raise ValueError(f"Invalid native FP64 raw HF matrix {name}")
    sizes = dict(zip(("a","b","r","s"),(raw[name].shape[1] for name in orbital)))
    if frozen_a >= sizes["a"] or frozen_b >= sizes["b"]:
        raise ValueError("GPU setup requires nonempty active occupied spaces")
    ehost = {name:np.asarray(value) for name,value in energies.items()}
    for name,value in ehost.items():
        if value.dtype != np.float64 or value.shape != (sizes[name],) or not np.isfinite(value).all():
            raise ValueError(f"Invalid native FP64 raw HF energies {name}")
    # Validation only: reject nonpositive monomer gaps, never clamp denominators.
    if (np.max(ehost["a"][frozen_a:]) >= np.min(ehost["r"])
            or np.max(ehost["b"][frozen_b:]) >= np.min(ehost["s"])):
        raise ValueError("HF monomer occupied/virtual gaps must be positive")
    m = DeviceMath(session)
    d = {name:session.upload(value) for name,value in raw.items()}
    a = m.window(d["Cocc_A"],frozen_a)
    b = m.window(d["Cocc_B"],frozen_b)
    r,s = d["Cvir_A"],d["Cvir_B"]
    S,DA,DB = d["S"],d["D_A"],d["D_B"]
    # Shared products preserve the canonical equations, including nonsymmetric
    # K_O^T in Qbr. Density/projector conventions are inherited from upstream.
    DAS,DBS = m.product(DA,S),m.product(DB,S)
    DBr,DAs = m.product(DBS,r),m.product(DAS,s)
    at,bt = m.window(a,transpose=True),m.window(b,transpose=True)
    spaces = {
        "a":a,"b":b,"r":r,"s":s,
        "r1":m.combine(DBr,r,1,-1),
        "s1":m.combine(DAs,s,1,-1),
        "a2":m.product(DBS,a),
        "b2":m.product(DAS,b),
        "r3":m.combine(DBr,m.product(DAS,DBr),2,-2),
        "s3":m.combine(DAs,m.product(DBS,DAs),2,-2),
        "a4":m.combine(m.product(DAS,DBS,a),a,-2,0),
        "b4":m.combine(m.product(DBS,DAS,b),b,-2,0),
    }
    def p(*values):
        return m.product(*values)
    corrections = {
        "Qar":m.terms((4,p(at,d["J_B"],r)),(2,p(at,d["V_B"],r))),
        "Qbs":m.terms((4,p(bt,d["J_A"],s)),(2,p(bt,d["V_A"],s))),
        "Qas":m.terms(
            (2,p(at,d["J_B"],s)),(-1,p(at,d["K_B"],s)),(1,p(at,d["K_O"],s)),
            (-2,p(at,d["J_B"],DAs)),(-2,p(at,S,DB,d["J_A"],s)),
            (-1,p(at,S,DB,d["V_A"],s)),(1,p(at,d["V_B"],d["P_A"],S,s))),
        "Qbr":m.terms(
            (2,p(bt,d["J_A"],r)),(-1,p(bt,d["K_A"],r)),
            (1,p(bt,m.window(d["K_O"],transpose=True),r)),
            (-2,p(bt,S,DA,d["J_B"],r)),(-2,p(bt,d["J_A"],DBr)),
            (-1,p(bt,S,DA,d["V_B"],r)),(1,p(bt,d["V_A"],d["P_B"],S,r))),
        "Sas":p(at,S,s), "Sbr":p(bt,S,r),
        "SBar":p(at,S,DBr), "SAbs":p(bt,S,DAs),
    }
    # Raw energy upload/slicing is transport, not CPU dispersion arithmetic.
    energy_device = {name:session.upload(value[start:]) for name,value,start in (
        ("a",ehost["a"],frozen_a),("b",ehost["b"],frozen_b),
        ("r",ehost["r"],0),("s",ehost["s"],0))}
    return spaces,corrections,energy_device


def finish_components(session, disp, exch):
    """Return combined device pairs and three device global signed components."""
    m = DeviceMath(session)
    m.require(disp); m.require(exch)
    combined = m.combine(disp,exch)
    scalars = session.empty((3,))
    session.launch("reduce_components",1,128,[disp.size,disp,exch,scalars])
    return combined,scalars


def coefficient_pairs(session, spaces, side, start, columns):
    """Device-only counterpart of the canonical five-pair DFMO slab contract."""
    m = DeviceMath(session)
    if side == "r":
        r,r1,r3 = (m.window(spaces[name],start,columns) for name in ("r","r1","r3"))
        return {"Aar":(r,spaces["a"]), "Bbr":(r1,spaces["b"]),
                "Cbr":(r,spaces["b2"]), "Dar":(r3,spaces["a"]),
                "Ear":(r,spaces["a4"])}
    if side != "s":
        raise ValueError("DFMO side must be r or s")
    s,s1,s3 = (m.window(spaces[name],start,columns) for name in ("s","s1","s3"))
    return {"Abs":(s,spaces["b"]), "Bas":(s1,spaces["a"]),
            "Cas":(s,spaces["a2"]), "Dbs":(s3,spaces["b"]),
            "Ebs":(s,spaces["b4"])}
