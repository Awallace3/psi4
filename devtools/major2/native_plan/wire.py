"""Versioned little-endian IEEE754 transport; no native imports."""
import math
import struct

MAGIC = b"NDFPLAN1"
CAP = 128 * 1024**2


def encode(xyz, bases):
    """Serialize public BasisSet snapshots, preserving global shell/AO order."""
    out = bytearray(MAGIC)

    def u(*values):
        out.extend(struct.pack("<"+"Q"*len(values), *values))

    def d(values):
        values = list(values)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite basis input")
        out.extend(struct.pack("<"+"d"*len(values), *values))

    if not 0 < len(xyz) <= 512 or len(bases) != 2:
        raise ValueError("center/basis counts")
    u(len(xyz), *(b["nbf"] for b in bases))
    d(v for row in xyz for v in row)
    for b in bases:
        shells = b["shells"]
        u(len(shells))
        counts = [sum(s["center"] == a for s in shells) for a in range(len(xyz))]
        if not all(0 < c <= 512 for c in counts):
            raise ValueError("shell counts")
        u(*counts)
        offset = 0
        for i, s in enumerate(shells):
            if (s["index"] != i or s["offset"] != offset or s["pure"] != 1 or
                    not 0 <= s["am"] <= 10 or not 0 < len(s["exp"]) <= 256 or
                    len(s["exp"]) != len(s["coef"]) or
                    not all(v > 0 for v in s["exp"])):
                raise ValueError("invalid shell/order/primitive")
            u(i, s["center"], s["offset"], s["pure"], s["am"], len(s["exp"]))
            d(s["xyz"])
            d(s["exp"])
            d(s["coef"])
            offset += 2*s["am"]+1
        if offset != b["nbf"]:
            raise ValueError("basis function count")
    if len(out) > CAP:
        raise ValueError("basis transport cap")
    return bytes(out)


def snapshot(basis, natom):
    """No coefficient fallback: normalized coef(), never original_coef/erd_coef."""
    rows = []
    order = []
    for atom in range(natom):
        for q in range(basis.nshell_on_center(atom)):
            i = basis.shell_on_center(atom, q)
            order.append(i)
            s = basis.shell(i)
            if basis.shell_to_center(i) != atom or s.ncenter != atom:
                raise ValueError("shell center identity")
            rows.append(dict(index=i, center=atom, offset=s.function_index,
                pure=int(s.is_pure()), am=s.am, xyz=[s.coord(k) for k in range(3)],
                exp=[s.exp(k) for k in range(s.nprimitive)],
                coef=[s.coef(k) for k in range(s.nprimitive)]))
    if order != list(range(basis.nshell())):
        raise ValueError("center-grouped order differs from AO order")
    return dict(nbf=basis.nbf(), shells=rows)
