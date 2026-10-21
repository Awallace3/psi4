"""Host-input contracts shared by the experimental Python CUDA producer/kernel."""
import numpy as np


def coefficient_pairs(spaces, side, slab):
    if side == "r":
        return {"Aar": (spaces["r"][:, slab], spaces["a"]),
                "Bbr": (spaces["r1"][:, slab], spaces["b"]),
                "Cbr": (spaces["r"][:, slab], spaces["b2"]),
                "Dar": (spaces["r3"][:, slab], spaces["a"]),
                "Ear": (spaces["r"][:, slab], spaces["a4"])}
    if side != "s":
        raise ValueError("DFMO side must be r or s")
    return {"Abs": (spaces["s"][:, slab], spaces["b"]),
            "Bas": (spaces["s1"][:, slab], spaces["a"]),
            "Cas": (spaces["s"][:, slab], spaces["a2"]),
            "Dbs": (spaces["s3"][:, slab], spaces["b"]),
            "Ebs": (spaces["s"][:, slab], spaces["b4"])}


def validate_inputs(spaces, corrections, energies):
    needed = ("a", "r", "b", "s", "r1", "s1", "a2", "b2", "r3", "s3", "a4", "b4")
    if set(spaces) != set(needed) or set(energies) != set("arbs"):
        raise ValueError("Incomplete canonical orbital spaces/energies")
    for name, energy in energies.items():
        if energy.ndim != 1 or energy.dtype != np.float64 or not np.isfinite(energy).all():
            raise ValueError(f"Wrong FP64 energy vector {name}")
    if (any(energies[name].size == 0 for name in "arbs")
            or max(energies["a"]) >= min(energies["r"])
            or max(energies["b"]) >= min(energies["s"])):
        raise ValueError("Nonempty occupied/virtual spaces with positive HF gaps required")
    nao = spaces["a"].shape[0]
    for name in needed:
        matrix = spaces[name]
        if (matrix.ndim != 2 or matrix.shape != (nao, energies[name[0]].size)
                or matrix.dtype != np.float64 or not np.isfinite(matrix).all()):
            raise ValueError(f"Wrong FP64 orbital space {name}")
    shapes = {"Qar": ("a", "r"), "SBar": ("a", "r"), "Qbr": ("b", "r"), "Sbr": ("b", "r"),
              "SAbs": ("b", "s"), "Qbs": ("b", "s"), "Sas": ("a", "s"), "Qas": ("a", "s")}
    if set(corrections) != set(shapes):
        raise ValueError("Incomplete exchange-dispersion rank-one corrections")
    for name, (occ, vir) in shapes.items():
        array = corrections[name]
        if (array.shape != (energies[occ].size, energies[vir].size)
                or array.dtype != np.float64 or not np.isfinite(array).all()):
            raise ValueError(f"Wrong FP64 correction {name}")
