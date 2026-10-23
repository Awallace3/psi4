#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2024 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import time
from typing import List, Tuple

import numpy as np

from psi4 import core

from ...p4util import solvers
from .sapt_util import print_sapt_var
import einsums as ein
import einsums.graph as cg
import einsums.rc

# Turn off einsums' profiler unless the user chose a setting (``None`` is
# einsums' "unset").  Builds ship with EINSUMS_WITH_PROFILER=ON and record by
# default; the batch-dot einsums in fdisp0 open one zone per (a,b) pair, and the
# single consumer thread draining them slows the MKL GEMMs that follow.  At
# protein83 that is 42 ms of a 184 ms dispersion block.  The runtime reads
# einsums.rc once, on first use, so this must run at import.  Leave both alone
# when ``--einsums:profile:*`` was given on the command line: einsums rejects an
# rc setting and a flag that disagree.
if not any(o.startswith("--einsums:profile:") for o in getattr(ein, "cli_options", [])):
    if einsums.rc.profile_disable is None:
        einsums.rc.profile_disable = True
    if einsums.rc.profile_report is None:
        einsums.rc.profile_report = False


# --------------------------------------------------------------------------
# einsums v2 interop
#
# einsums v2 exposes no Python constructor that wraps foreign memory: a tensor
# owns its buffer, ``numpy.asarray(tensor)`` is a zero-copy view *out*, and
# anything going *in* (``einsums.asarray``) copies.  A psi4 ``core.Matrix``
# therefore cannot be handed to ``einsums.linalg`` as a view, the way v1's
# numpy-backed ``ein.core.*`` entry points were used in this module.
#
# The helpers below make every psi4 <-> einsums crossing explicit: a crossing
# is one O(N^2) copy, while the work that matters -- the gemm chains -- stays
# inside einsums tensors from the first factor to the last.  Hot accumulators
# are created as einsums tensors so their axpy traffic never crosses back.
# --------------------------------------------------------------------------

# Edge of the (r,s) compute block in fdisp0, in virtual orbitals.  The block
# GEMMs gain throughput up to about this edge (135 GF/s at nanotube dimensions
# on 24 threads, against 42 GF/s for a pair-at-a-time batched kernel), but the
# block's elementwise stages -- the denominator, the division, and the two
# passes of the localization -- are bandwidth-bound, and they set the optimum
# well below the point where the GEMMs stop improving.  Sweeping the edge over
# a whole block iteration at protein83 dimensions puts the minimum at 24 with a
# flat bottom from 20 to 32; 64 costs 6-17% more, and below 16 the GEMMs get
# too skinny to feed the cores and the cost climbs steeply.
FDISP_BLOCK = 24

# Size, in doubles, of one fdisp0 DF staging matrix.  The staging matrices only
# carry a slab of a DF tensor from disk into the packed Q-major buffers, so
# they want to be large enough that fill_tensor's per-call cost disappears and
# small enough to stay out of the resident high-water mark.  1 M doubles is
# 8 MB each, eight of them live at once.
STAGE_DOUBLES = 1_000_000

# Nuclear centers per batch in the F-SAPT nuclear-ESP transform.  The AO
# buffer is nbf * NESP_BLOCK * nbf doubles, so 24 keeps it around 60 MB at
# nbf ~ 600 while still amortizing the backtransform over enough centers.
NESP_BLOCK = 24

_EIN_TENSORS = (ein.RuntimeTensorD, ein.RuntimeTensorViewD)


def _is_ein(x) -> bool:
    """True for an einsums double-precision tensor or tensor view."""
    return isinstance(x, _EIN_TENSORS)


def _arr(x) -> np.ndarray:
    """A numpy view of a psi4 Matrix/Vector, einsums tensor, or ndarray."""
    if _is_ein(x) or isinstance(x, np.ndarray):
        return np.asarray(x)
    return x.np


def _ein(x, name: str = "tmp"):
    """``x`` as an einsums tensor, copying only if it is not one already."""
    if _is_ein(x):
        return x
    return ein.asarray(_arr(x), name=name)


def _ein_zeros(*dims, name: str = "zeros"):
    """A zeroed einsums tensor of the given shape."""
    return ein.create_zero_tensor(name, [int(d) for d in dims])


# Uninitialized scratch for fdisp0's per-block work buffers.  One pool for the
# life of the process, grown as needed and never rebuilt: its arenas are
# mimalloc's, which never hands one back and caps how many a process may hold,
# so a pool per call would leak an arena each time fdisp0 ran.  Unpinned, the
# pages of carves that have been freed purge back to the OS, so between calls
# it holds address space, not memory.  Carves of a few GB do not fit an arena,
# so this is for the block buffers only, not the packed DF tensors.
_FDISP_POOL = None


def _fdisp_pool(nbytes: int):
    """The fdisp0 scratch pool, with room for ``nbytes`` more of carves."""
    global _FDISP_POOL
    if _FDISP_POOL is None:
        _FDISP_POOL = ein.MemoryPool(nbytes, "fdisp0")
    else:
        _FDISP_POOL.reserve(_FDISP_POOL.bytes_used + nbytes)
    return _FDISP_POOL


# Smallest carve, in bytes, the pool has refused; larger ones skip the pool.
_FDISP_POOL_REFUSED = None


def _fdisp_carve(pool, dims, name: str):
    """An uninitialized ``dims`` tensor from ``pool``, or a zeroed one if the pool cannot carve it.

    mimalloc 3.5 will not carve a single block above about 224 MiB from a pool's
    arena ("could not carve ... even from a fresh arena"); 3.3 will.  A block
    buffer that large needs na * nb above about 51000 at the default block edge.
    The zeroed tensor gives the same results, without the first-touch placement.
    Each refusal also costs the pool an arena, of which mimalloc allows a
    process only so many, so a size that failed once is not tried again.
    """
    global _FDISP_POOL_REFUSED
    nbytes = 8 * dims[0] * dims[1]
    if _FDISP_POOL_REFUSED is None or nbytes < _FDISP_POOL_REFUSED:
        try:
            return pool.empty(dims, name=name)
        except RuntimeError as exc:
            if "could not carve" not in str(exc):
                raise
            if _FDISP_POOL_REFUSED is None:
                core.print_out(f"    fdisp0: einsums MemoryPool could not carve a {nbytes / 2**20:.0f} MiB block buffer; "
                               "using zeroed tensors for buffers that large.\n")
            _FDISP_POOL_REFUSED = nbytes if _FDISP_POOL_REFUSED is None else min(_FDISP_POOL_REFUSED, nbytes)
    return _ein_zeros(*dims, name=name)


def _dispersion_block_extents(nr, ns, max_r, max_s, blk_r, blk_s):
    """Align DF boundaries so only main/final-edge shapes can be cached.

    A non-aligned DF edge creates an additional short compute edge at every
    DF boundary, distinct from the final system edge. Aligning both axes
    limits their Cartesian product to four shapes, as charged by the planner.
    """
    blk_r, blk_s = min(blk_r, max_r), min(blk_s, max_s)
    if min(nr, ns, max_r, max_s, blk_r, blk_s) < 1:
        raise ValueError("Dispersion block extents must be positive")
    if max_r < nr:
        max_r = max(blk_r, max_r // blk_r * blk_r)
    if max_s < ns:
        max_s = max(blk_s, max_s // blk_s * blk_s)
    return max_r, max_s, blk_r, blk_s


def _ein_clone(x, name: str = "clone", scale: float = 1.0):
    """A fresh einsums tensor holding ``scale * x``, whatever ``x`` is."""
    out = ein.array(_arr(x), name=name)
    if scale != 1.0:
        ein.linalg.scale(scale, out)
    return out


def _mat(x) -> core.Matrix:
    """``x`` as a psi4 Matrix, copying only if it is not one already."""
    if isinstance(x, core.Matrix):
        return x
    return core.Matrix.from_array(_arr(x))


def _axpy(alpha: float, X, Y):
    """``Y += alpha * X`` through einsums, for any mix of Matrix/ndarray/tensor.

    A psi4-owned destination is written back explicitly, since einsums cannot
    accumulate into memory it does not own.  Pass einsums tensors on both
    sides (or a ``.T`` view of one) to keep the update copy-free.
    """
    if _is_ein(Y):
        ein.linalg.axpy(alpha, _ein(X, name="axpy_src"), Y)
        return Y
    dest = _arr(Y)
    acc = _ein(dest, name="axpy_dest")
    ein.linalg.axpy(alpha, _ein(X, name="axpy_src"), acc)
    dest[...] = np.asarray(acc)
    return Y


def _dot(X, Y) -> float:
    """Full inner product of two tensors/matrices/arrays, through einsums."""
    return ein.linalg.dot(_ein(X, name="dot_x"), _ein(Y, name="dot_y"))


# Equations come from https://doi.org/10.1063/5.0090688
def localization(
    cache: dict,
    dimer_wfn: core.Wavefunction,
    do_print: bool = True,
) -> None:
    r"""Localize dimer occupied orbitals via Intrinsic Bond Orbitals (IBO).

    Performs IBO localization on the dimer occupied orbitals, separating
    them into frozen-core and active localized subsets. The localized
    orbitals, rotation matrices, and IAO projectors are stored in *cache*.

    Parameters
    ----------
    cache : dict
        SAPT data cache. Must already contain ``'Cocc'``, ``'eps_occ'``.
        Updated in-place with ``'Locc'``, ``'Qocc'``, ``'IAO'``,
        ``'Lfocc'``, and ``'Laocc'``.
    dimer_wfn : core.Wavefunction
        Dimer supermolecular wavefunction (provides basis set and molecule).
    do_print : bool, optional
        Whether to print status output, by default True.
    """
    core.print_out("\n  ==> Localizing Orbitals 1 <== \n\n")
    # Extract monomers to compute frozen core counts
    mol = dimer_wfn.molecule()
    molA = mol.extract_subsets([1], [])
    molB = mol.extract_subsets([2], [])
    nfocc0A = dimer_wfn.basisset().n_frozen_core(
        core.get_option("GLOBALS", "FREEZE_CORE"), molA
    )
    nfocc0B = dimer_wfn.basisset().n_frozen_core(
        core.get_option("GLOBALS", "FREEZE_CORE"), molB
    )
    nfocc_dimer = nfocc0A + nfocc0B

    N_eps_occ = cache["eps_occ"].dimpi()[0]
    Focc = core.Matrix("Focc", N_eps_occ, N_eps_occ)
    for i in range(N_eps_occ):
        Focc.np[i, i] = cache["eps_occ"].np[i]
    ranges = [0, nfocc_dimer, N_eps_occ]  # Separate frozen and active orbitals
    minao = core.BasisSet.build(
        dimer_wfn.molecule(), "BASIS", core.get_global_option("MINAO_BASIS")
    )
    dimer_wfn.set_basisset("MINAO", minao)
    # pybind11 IBO location: ./psi4/src/export_wavefunction.cc
    IBO_loc = core.IBOLocalizer2(
        dimer_wfn.basisset(),
        dimer_wfn.get_basisset("MINAO"),
        cache["Cocc"],
    )
    IBO_loc.print_header()
    ret = IBO_loc.localize(
        cache["Cocc"],
        Focc,
        ranges,
    )
    cache["Locc"] = ret["L"]
    cache["Qocc"] = ret["Q"]
    cache["IAO"] = ret["A"]

    # Extract frozen and active localized orbitals separately
    nn = cache["Cocc"].shape[0]  # number of AO basis functions
    nf = nfocc_dimer
    na = N_eps_occ - nfocc_dimer  # number of active occupied orbitals

    if nf > 0:
        # Store frozen core localized orbitals
        Lfocc = core.Matrix("Lfocc", nn, nf)
        Lfocc.np[:, :] = ret["L"].np[:, :nf]
        cache["Lfocc"] = Lfocc

    # Store active occupied localized orbitals
    Laocc = core.Matrix("Laocc", nn, na)
    Laocc.np[:, :] = ret["L"].np[:, nf:]
    cache["Laocc"] = Laocc
    return


def flocalization(
    cache: dict,
    dimer_wfn: core.Wavefunction,
    do_print: bool = True,
) -> None:
    r"""Localize monomer occupied orbitals separately for F-SAPT partitioning.

    Performs IBO localization independently on monomer A and monomer B
    occupied orbitals. Separates frozen-core and active localized orbitals
    and stores them in *cache*. Handles link-orbital assignments for
    three-body (A-C-B) fragmentation schemes even though I-SAPT is not
    currently implemented for this module.

    Parameters
    ----------
    cache : dict
        SAPT data cache. Must already contain monomer orbital coefficients
        (``'Cocc_A'``, ``'Cocc_B'``) and orbital energies (``'eps_occ_A'``,
        ``'eps_occ_B'``). Updated in-place with ``'Locc_A'``, ``'Locc_B'``,
        ``'Uocc_A'``, ``'Uocc_B'``, ``'Qocc0A'``, ``'Qocc0B'``,
        ``'Lfocc0A'``, ``'Laocc0A'``, ``'Lfocc0B'``, ``'Laocc0B'``,
        ``'Caocc0A'``, and ``'Caocc0B'``.
    dimer_wfn : core.Wavefunction
        Dimer supermolecular wavefunction (provides basis set and molecule).
    do_print : bool, optional
        Whether to print status output, by default True.
    """
    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    core.print_out("  ==> F-SAPT Localization (IBO) <==\n\n")
    core.print_out("  ==> Local orbitals for Monomer A <==\n\n")
    mol = dimer_wfn.molecule()
    molA = mol.extract_subsets([1], [])
    molB = mol.extract_subsets([2], [])
    nfocc0A = dimer_wfn.basisset().n_frozen_core(
        core.get_option("GLOBALS", "FREEZE_CORE"), molA
    )
    nfocc0B = dimer_wfn.basisset().n_frozen_core(
        core.get_option("GLOBALS", "FREEZE_CORE"), molB
    )
    nn = cache["Cocc_A"].shape[0]
    nf = nfocc0A
    nm = cache["Cocc_A"].shape[1]  # total occupied orbitals (frozen + active)
    na = nm - nf  # active occupied orbitals only
    ranges = [0, nf, nm]
    N = cache["eps_occ_A"].shape[0]
    Focc = core.Matrix("Focc", N, N)
    for i in range(N):
        Focc.np[i, i] = cache["eps_occ_A"].np[i]
    IBO_loc = core.IBOLocalizer2(
        dimer_wfn.basisset(),
        dimer_wfn.get_basisset("MINAO"),
        core.Matrix.from_array(cache["Cocc_A"]),
    )
    IBO_loc.print_header()
    ret = IBO_loc.localize(
        core.Matrix.from_array(cache["Cocc_A"]),
        Focc,
        ranges,
    )

    Locc_A = ret["L"]
    Uocc_A = ret["U"]
    Qocc0A = ret["Q"]

    cache["Locc_A"] = Locc_A
    cache["Uocc_A"] = Uocc_A
    cache["Qocc0A"] = Qocc0A

    Lfocc0A = core.Matrix("Lfocc0A", nn, nf)
    Laocc0A = core.Matrix("Laocc0A", nn, na)
    Ufocc0A = core.Matrix("Ufocc0A", nf, nf)
    Uaocc0A = core.Matrix("Uaocc0A", na, na)

    Lfocc0A.np[:, :] = Locc_A.np[:, :nf]
    Laocc0A.np[:, :] = Locc_A.np[:, nf : nf + na]
    Ufocc0A.np[:, :] = Uocc_A.np[:nf, :nf]
    Uaocc0A.np[:, :] = Uocc_A.np[nf : nf + na, nf : nf + na]

    cache["Lfocc0A"] = Lfocc0A
    cache["Laocc0A"] = Laocc0A
    cache["Ufocc0A"] = Ufocc0A
    cache["Uaocc0A"] = Uaocc0A
    # Store active occupied orbitals for dispersion (Caocc0A = Cocc_A[:, nf:])
    Caocc0A = core.Matrix("Caocc0A", nn, na)
    Caocc0A.np[:, :] = cache["Cocc_A"].np[:, nf : nf + na]
    cache["Caocc0A"] = Caocc0A

    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        Locc_A = core.Matrix("Locc_A", nn, nm + 1)
        Locc_A.np[:, :nm] = Locc_A.np[:, :]
        Locc_A.np[:, nm] = cache["thislinkA"].np[:, 0]
        cache["Locc_A"] = Locc_A
    else:
        cache["Locc_A"] = Locc_A

    core.print_out("  ==> Local orbitals for Monomer B <==\n\n")

    nn = cache["Cocc_B"].shape[0]
    nf = nfocc0B
    nm = cache["Cocc_B"].shape[1]  # total occupied orbitals (frozen + active)
    na = nm - nf  # active occupied orbitals only
    ranges = [0, nf, nm]

    N = cache["eps_occ_B"].shape[0]
    Focc = core.Matrix("Focc", N, N)
    for i in range(N):
        Focc.np[i, i] = cache["eps_occ_B"].np[i]

    IBO_loc = core.IBOLocalizer2(
        dimer_wfn.basisset(),
        dimer_wfn.get_basisset("MINAO"),
        core.Matrix.from_array(cache["Cocc_B"]),
    )
    IBO_loc.print_header()
    ret = IBO_loc.localize(
        core.Matrix.from_array(cache["Cocc_B"]),
        Focc,
        ranges,
    )

    Locc_B = ret["L"]
    Uocc_B = ret["U"]
    Qocc0B = ret["Q"]

    cache["Locc_B"] = Locc_B
    cache["Uocc_B"] = Uocc_B
    cache["Qocc0B"] = Qocc0B

    Lfocc0B = core.Matrix("Lfocc0B", nn, nf)
    Laocc0B = core.Matrix("Laocc0B", nn, na)
    Ufocc0B = core.Matrix("Ufocc0B", nf, nf)
    Uaocc0B = core.Matrix("Uaocc0B", na, na)

    Lfocc0B.np[:, :] = Locc_B.np[:, :nf]
    Laocc0B.np[:, :] = Locc_B.np[:, nf : nf + na]
    Ufocc0B.np[:, :] = Uocc_B.np[:nf, :nf]
    Uaocc0B.np[:, :] = Uocc_B.np[nf : nf + na, nf : nf + na]

    cache["Lfocc0B"] = Lfocc0B
    cache["Laocc0B"] = Laocc0B
    cache["Ufocc0B"] = Ufocc0B
    cache["Uaocc0B"] = Uaocc0B
    # Store active occupied orbitals for dispersion (Caocc0B = Cocc_B[:, nf:])
    Caocc0B = core.Matrix("Caocc0B", nn, na)
    Caocc0B.np[:, :] = cache["Cocc_B"].np[:, nf : nf + na]
    cache["Caocc0B"] = Caocc0B

    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        Locc_B = core.Matrix("Locc_B", nn, nm + 1)
        Locc_B.np[:, :nm] = Locc_B.np[:, :]
        Locc_B.np[:, nm] = cache["thislinkB"].np[:, 0]
        cache["Locc_B"] = Locc_B
    else:
        cache["Locc_B"] = Locc_B

def partition(
    cache: dict,
    dimer_wfn: core.Wavefunction,
    do_print: bool = True,
) -> None:
    r"""Partition localized orbitals into monomer A, monomer B, and link fragments.

    Uses IBO charges to assign each localized orbital to a fragment (A, B,
    or linker C). Handles automatic and manual link-orbital selection and
    various link-assignment schemes (SAO, SIAO, etc.).

    Parameters
    ----------
    cache : dict
        SAPT data cache. Must contain ``'Locc'`` and ``'Qocc'`` from a prior
        :func:`localization` call. Updated in-place with fragment-partitioned
        orbitals and assignment vectors.
    dimer_wfn : core.Wavefunction
        Dimer supermolecular wavefunction (provides molecule and basis set).
    do_print : bool, optional
        Whether to print status output, by default True.
    """
    core.print_out("\n  ==> Partitioning <== \n\n")
    # Sizing
    mol = dimer_wfn.molecule()
    natoms = mol.natom()
    n_Locc = cache["Locc"].shape[1]

    # Monomer Atoms
    fragments = mol.get_fragments()
    indA = np.arange(*fragments[0], dtype=int)
    indB = np.arange(*fragments[1], dtype=int)
    indC = (
        np.arange(*fragments[2], dtype=int)
        if len(fragments) == 3
        else np.array([], dtype=int)
    )
    cache["FRAG"] = core.Vector(natoms)
    frag = cache["FRAG"].np
    frag[:] = 0.0
    frag[indA] = 1.0
    frag[indB] = 2.0
    if indC.size:
        frag[indC] = 3.0
    core.print_out("Fragment lookup table:\n")
    cache["FRAG"].print_out()
    core.print_out("   => Atomic Partitioning <= \n\n")
    core.print_out(f"    Monomer A: {len(indA)} atoms\n")
    core.print_out(f"    Monomer B: {len(indB)} atoms\n")
    core.print_out(f"    Monomer C: {len(indC)} atoms\n\n")
    np.set_printoptions(precision=14, suppress=True)

    # Fragment Orbital Charges
    Locc = cache["Locc"].np  # (n_ao x n_occ)
    Qocc = cache["Qocc"].np  # (n_atom x n_occ) orbital populations per atom

    n_ao, n_occ = Locc.shape

    QF = core.Matrix(3, n_Locc).np
    QF.fill(0.0)
    QF[0, :] = Qocc[indA, :].sum(axis=0)
    QF[1, :] = Qocc[indB, :].sum(axis=0)
    if indC.size:
        QF[2, :] = Qocc[indC, :].sum(axis=0)

    # --- link identification ---
    link_orbs: List[int] = []
    link_atoms: List[Tuple[int, int]] = []
    link_types: List[str] = []

    def top_two_atoms_for_orb(a: int) -> tuple[int, int]:
        A_sorted = np.argsort(Qocc[:, a])[::-1]
        A1, A2 = int(A_sorted[0]), int(A_sorted[1])
        return (A1, A2) if A1 < A2 else (A2, A1)

    link_sel = core.get_option("FISAPT", "FISAPT_LINK_SELECTION").upper()
    if link_sel == "AUTOMATIC":
        delta = float(core.get_option("FISAPT", "FISAPT_CHARGE_COMPLETENESS"))
        for a in range(n_occ):
            if np.any(QF[:, a] > delta):
                continue
            if QF[0, a] + QF[2, a] > delta:
                link_orbs.append(a)
                link_types.append("AC")
            elif QF[1, a] + QF[2, a] > delta:
                link_orbs.append(a)
                link_types.append("BC")
            elif QF[0, a] + QF[1, a] > delta:
                link_orbs.append(a)
                link_types.append("AB")
            else:
                raise ValueError(
                    "FISAPT: 3c-2e style bond encountered (no single/pair exceeds delta)."
                )
        for a in link_orbs:
            link_atoms.append(top_two_atoms_for_orb(a))
    elif link_sel == "MANUAL":
        if not core.get_option("FISAPT", "FISAPT_MANUAL_LINKS"):
            raise ValueError(
                "FISAPT: MANUAL selection requires manual_links (0-based atom pairs)."
            )
        S = set(indA.tolist())
        T = set(indB.tolist())
        U = set(indC.tolist())
        for A1, A2 in core.get_option("FISAPT", "FISAPT_MANUAL_LINKS"):
            prod = Qocc[A1, :] * Qocc[A2, :]
            a = int(np.argmax(prod))
            link_orbs.append(a)
            A1_, A2_ = (A1, A2) if A1 < A2 else (A2, A1)
            link_atoms.append((A1_, A2_))
            if (A1_ in S) and (A2_ in U):
                link_types.append("AC")
            elif (A1_ in T) and (A2_ in U):
                link_types.append("BC")
            elif (A1_ in S) and (A2_ in T):
                link_types.append("AB")
            else:
                raise ValueError("FISAPT: manual pair is not AB, AC, or BC.")
    else:
        raise ValueError("FISAPT: Unrecognized FISAPT_LINK_SELECTION.")
    link_orbs = np.array(link_orbs, dtype=int)

    # --- Z per fragment originals ---
    ZA = core.Vector(natoms)
    ZB = core.Vector(natoms)
    ZC = core.Vector(natoms)
    ZA.np[:] = 0.0
    ZB.np[:] = 0.0
    ZC.np[:] = 0.0

    Z_all = np.array([mol.Z(i) for i in range(natoms)], dtype=float)
    ZA.np[indA] = Z_all[indA]
    ZB.np[indB] = Z_all[indB]
    if indC.size:
        ZC.np[indC] = Z_all[indC]

    cache["ZA"] = ZA
    cache["ZB"] = ZB
    cache["ZC"] = ZC
    cache["ZA_orig"] = core.Vector.from_array(ZA.np.copy())
    cache["ZB_orig"] = core.Vector.from_array(ZB.np.copy())
    cache["ZC_orig"] = core.Vector.from_array(ZC.np.copy())

    # --- link assignment (C vs AB vs SAO*/SIAO*) ---
    orbsA: List[int] = []
    orbsB: List[int] = []
    orbsC: List[int] = []
    orbsL: List[int] = []
    typesL: List[str] = []

    la = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    valid = {"AB", "C", "SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}
    if la not in valid:
        raise ValueError("FISAPT: FISAPT_LINK_ASSIGNMENT not recognized.")

    # --- link assignment (C vs AB vs SAO*/SIAO*) ---
    orbsA: List[int] = []
    orbsB: List[int] = []
    orbsC: List[int] = []
    orbsL: List[int] = []
    typesL: List[str] = []

    la = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    valid = {"AB", "C", "SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}
    if la not in valid:
        raise ValueError("FISAPT: FISAPT_LINK_ASSIGNMENT not recognized.")

    if la in {"C", "SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}:
        for a, (A1, A2), t in zip(link_orbs, link_atoms, link_types):
            typesL.append(t)
            if t == "AC":
                ZA.np[A1] -= 1.0
                ZC.np[A1] += 1.0
                orbsC.append(a)
                orbsL.append(a)
            elif t == "BC":
                ZB.np[A1] -= 1.0
                ZC.np[A1] += 1.0
                orbsC.append(a)
                orbsL.append(a)
            elif t == "AB":
                ZA.np[A1] -= 1.0
                ZC.np[A1] += 1.0
                ZB.np[A2] -= 1.0
                ZC.np[A2] += 1.0
                orbsC.append(a)
                orbsL.append(a)
    elif la == "AB":
        for a, (A1, A2), t in zip(link_orbs, link_atoms, link_types):
            if t == "AC":
                ZA.np[A1] += 1.0
                ZC.np[A1] -= 1.0
                orbsA.append(a)
            elif t == "BC":
                ZB.np[A1] += 1.0
                ZC.np[A1] -= 1.0
                orbsB.append(a)
            elif t == "AB":
                raise ValueError(
                    "FISAPT: AB link requires LINK_ASSIGNMENT C in this scheme."
                )

    # --- electron counts per fragment; enforce closed-shell ---
    fragment_charges = mol.get_fragment_charges()
    qA, qB = int(fragment_charges[0]), int(fragment_charges[1])
    qC = int(fragment_charges[2]) if len(fragment_charges) == 3 else 0

    def i_round(x: float) -> int:
        # protect against bankers rounding
        return int(np.floor(x + 0.5))

    ZA2 = i_round(float(ZA.np.sum()))
    ZB2 = i_round(float(ZB.np.sum()))
    ZC2 = i_round(float(ZC.np.sum()))
    EA2, EB2, EC2 = ZA2 - qA, ZB2 - qB, ZC2 - qC

    if EA2 % 2 or EB2 % 2 or EC2 % 2:
        raise ValueError(
            "FISAPT: fragment charge incompatible with singlet (odd electron count)."
        )

    NA2, NB2, NC2 = EA2 // 2, EB2 // 2, EC2 // 2
    if (NA2 + NB2 + NC2) != n_occ:
        raise ValueError(
            "FISAPT: sum of fragment electrons incompatible with total electrons."
        )

    RA2 = NA2 - len(orbsA)
    RB2 = NB2 - len(orbsB)
    RC2 = NC2 - len(orbsC)

    # --- greedy fill using QF weights (C then A then B), excluding taken orbs ---
    taken = set(orbsA) | set(orbsB) | set(orbsC)

    def take_top(weights: np.ndarray, k: int, taken_set: set) -> list[int]:
        if k <= 0:
            return []
        order = np.argsort(weights)[::-1]
        picked: List[int] = []
        for a in order:
            if int(a) in taken_set:
                continue
            picked.append(int(a))
            taken_set.add(int(a))
            if len(picked) == k:
                break
        return picked

    orbsC += take_top(QF[2, :], RC2, taken)
    orbsA += take_top(QF[0, :], RA2, taken)
    orbsB += take_top(QF[1, :], RB2, taken)

    # --- sort & link ordering swap like C++ ---
    orbsA = np.array(sorted(set(orbsA)), dtype=int)
    orbsB = np.array(sorted(set(orbsB)), dtype=int)
    orbsC = np.array(sorted(set(orbsC)), dtype=int)
    orbsL = np.array(orbsL, dtype=int)
    if orbsL.size > 1 and orbsL[0] > orbsL[1]:
        orbsL[[0, 1]] = orbsL[[1, 0]]
        typesL[0], typesL[1] = typesL[1], typesL[0]

    # --- build LoccA/B/C/L as psi4 Matrices (column extracts) ---
    def cols(M_np: np.ndarray, idx: np.ndarray) -> core.Matrix:
        if idx.size == 0:
            return np.zeros(M_np.shape[0])
        return core.Matrix.from_array(M_np[:, idx])

    def extract_columns(cols, A: core.Matrix) -> core.Matrix:
        cols = np.asarray(cols, dtype=int)
        if cols.size == 0:
            return None
        A2 = A[:, cols]
        return core.Matrix.from_array(A2)

    cache["LoccA"] = extract_columns(orbsA, Locc)
    cache["LoccB"] = extract_columns(orbsB, Locc)
    cache["LoccC"] = extract_columns(orbsC, Locc)
    cache["LoccL"] = extract_columns(orbsL, Locc)

    cache["QF"] = QF
    # --- summary numbers  ---
    ZA_int, ZB_int, ZC_int = (
        i_round(ZA.np.sum()),
        i_round(ZB.np.sum()),
        i_round(ZC.np.sum()),
    )
    YA, YB, YC = int(2 * orbsA.size), int(2 * orbsB.size), int(2 * orbsC.size)

    core.print_out("   => Partition Summary <= \n\n")
    core.print_out(
        f"    Monomer A: {ZA_int - YA:2d} charge, {ZA_int:3d} protons, {YA:3d} electrons, {len(orbsA):3d} docc\n"
    )
    core.print_out(
        f"    Monomer B: {ZB_int - YB:2d} charge, {ZB_int:3d} protons, {YB:3d} electrons, {len(orbsB):3d} docc\n"
    )
    core.print_out(
        f"    Monomer C: {ZC_int - YC:2d} charge, {ZC_int:3d} protons, {YC:3d} electrons, {len(orbsC):3d} docc\n"
    )
    return cache


def build_sapt_jk_cache(
    wfn_dimer: core.Wavefunction,
    wfn_A: core.Wavefunction,
    wfn_B: core.Wavefunction,
    jk: core.JK,
    do_print: bool = True,
    external_potentials: dict = None,
) -> dict:
    r"""Construct the dimer-centered basis set (DCBS) cache of integrals and
    matrices for SAPT(DFT).

    Builds all one- and two-electron integrals needed for the electrostatics,
    exchange, and induction components of SAPT(DFT). Density matrices (Eq. 5,
    7), Coulomb/exchange matrices, nuclear potentials, and overlap integrals
    are computed and stored in the returned dictionary.

    .. math::

        \mathbf{P}^X = \mathbf{C}^{X,\text{occ}}(\mathbf{C}^{X,\text{occ}})^\dagger
        \quad (\text{Eq. 5})

    .. math::

        \mathbf{P}^{X,\text{vir}} = \mathbf{C}^{X,\text{vir}}(\mathbf{C}^{X,\text{vir}})^\dagger
        \quad (\text{Eq. 7})

    Parameters
    ----------
    wfn_dimer : core.Wavefunction
        Dimer supermolecular wavefunction.
    wfn_A : core.Wavefunction
        Monomer A wavefunction in the dimer-centered basis set (DCBS).
    wfn_B : core.Wavefunction
        Monomer B wavefunction in the dimer-centered basis set (DCBS).
    jk : core.JK
        Psi4 JK integral engine for computing Coulomb and exchange matrices.
    do_print : bool, optional
        Whether to print status output, by default True.
    external_potentials : dict, optional
        Dictionary of external potential objects keyed by ``'A'`` and ``'B'``.

    Returns
    -------
    dict
        Cache dictionary containing orbital coefficients, density matrices,
        Coulomb (``'J_A'``, ``'J_B'``, ``'J_O'``), exchange (``'K_A'``, ``'K_B'``,
        ``'K_O'``), nuclear potential (``'V_A'``, ``'V_B'``), overlap (``'S'``),
        orbital energies, and nuclear repulsion interaction energy.
    """
    core.print_out("\n  ==> Preparing SAPT Data Cache <== \n\n")
    jk.print_header()

    cache = {}
    cache["wfn_A"] = wfn_A
    cache["wfn_B"] = wfn_B

    # NOTE: scf_A from FISAPT0 and SAPT(DFT) wfn_A have slightly different
    # coefficients, so numerical exactness is not achieved everywhere, but
    # the pytests for final values are still quite robust

    # First grab the orbitals as psi4.core.Matrix objects
    cache["Cocc_A"] = wfn_A.Ca_subset("AO", "OCC")
    cache["Cocc_A"].name = "Cocc_A"
    cache["Cvir_A"] = wfn_A.Ca_subset("AO", "VIR")
    cache["Cvir_A"].name = "Cvir_A"

    cache["Cocc_B"] = wfn_B.Ca_subset("AO", "OCC")
    cache["Cocc_B"].name = "Cocc_B"
    cache["Cvir_B"] = wfn_B.Ca_subset("AO", "VIR")
    cache["Cvir_B"].name = "Cvir_B"

    cache["eps_occ_A"] = wfn_A.epsilon_a_subset("AO", "OCC")
    cache["eps_vir_A"] = wfn_A.epsilon_a_subset("AO", "VIR")
    cache["eps_occ_B"] = wfn_B.epsilon_a_subset("AO", "OCC")
    cache["eps_vir_B"] = wfn_B.epsilon_a_subset("AO", "VIR")

    # localization
    do_fsapt = core.get_option("SAPT", "SAPT_DFT_DO_FSAPT").upper() != "NONE"
    if do_fsapt:
        cache["Cfocc"] = wfn_dimer.Ca_subset("AO", "FROZEN_OCC")
        cache["eps_all"] = wfn_dimer.epsilon_a_subset("AO", "ALL")

        cache["Call"] = wfn_dimer.Ca_subset("AO", "ALL")
        cache["Cocc"] = wfn_dimer.Ca_subset("AO", "OCC")
        cache["Cvir"] = wfn_dimer.Ca_subset("AO", "VIR")

        cache["eps_occ"] = wfn_dimer.epsilon_a_subset("AO", "OCC")
        cache["eps_vir"] = wfn_dimer.epsilon_a_subset("AO", "VIR")

        cache["Caocc"] = wfn_dimer.Ca_subset("AO", "ACTIVE_OCC")
        cache["Cavir"] = wfn_dimer.Ca_subset("AO", "ACTIVE_VIR")
        cache["Cfvir"] = wfn_dimer.Ca_subset("AO", "FROZEN_VIR")

        cache["eps_focc"] = wfn_dimer.epsilon_a_subset("AO", "FROZEN_OCC")
        cache["eps_aocc"] = wfn_dimer.epsilon_a_subset("AO", "ACTIVE_OCC")
        cache["eps_avir"] = wfn_dimer.epsilon_a_subset("AO", "ACTIVE_VIR")
        cache["eps_fvir"] = wfn_dimer.epsilon_a_subset("AO", "FROZEN_VIR")

    # Build the densities as HF takes an extra "step", Eq. 5
    cache["D_A"] = chain_gemm_einsums([cache["Cocc_A"], cache["Cocc_A"]], ["N", "T"])
    cache["D_B"] = chain_gemm_einsums([cache["Cocc_B"], cache["Cocc_B"]], ["N", "T"])
    # Eq. 7
    cache["P_A"] = chain_gemm_einsums([cache["Cvir_A"], cache["Cvir_A"]], ["N", "T"])
    cache["P_B"] = chain_gemm_einsums([cache["Cvir_B"], cache["Cvir_B"]], ["N", "T"])

    # Potential ints - store as psi4.core.Matrix
    mints = core.MintsHelper(wfn_A.basisset())
    cache["V_A"] = mints.ao_potential()
    mints = core.MintsHelper(wfn_B.basisset())
    cache["V_B"] = mints.ao_potential()

    # External Potentials need to add to V_A and V_B. Preserve the
    # normalized metadata for downstream F-SAPT partitioning.
    cache["external_potentials"] = external_potentials
    if external_potentials:
        if external_potentials.get("A") is not None:
            ext_A = wfn_A.external_pot().computePotentialMatrix(wfn_A.basisset())
            cache["V_A"].add(ext_A)
        if external_potentials.get("B") is not None:
            ext_B = wfn_B.external_pot().computePotentialMatrix(wfn_B.basisset())
            cache["V_B"].add(ext_B)

    # Anything else we might need
    # S corresponds to the overlap matrix, S^{AO}
    cache["S"] = wfn_A.S().clone()
    cache["S"].name = "S"

    # J and K matrices
    jk.C_clear()

    # Normal J/K for Monomer A
    jk.C_left_add(wfn_A.Ca_subset("SO", "OCC"))
    jk.C_right_add(wfn_A.Ca_subset("SO", "OCC"))

    # Normal J/K for Monomer B
    jk.C_left_add(wfn_B.Ca_subset("SO", "OCC"))
    jk.C_right_add(wfn_B.Ca_subset("SO", "OCC"))

    DB_S_CA = chain_gemm_einsums([cache["D_B"], cache["S"], cache["Cocc_A"]])
    jk.C_left_add(DB_S_CA)
    jk.C_right_add(cache["Cocc_A"])

    jk.compute()

    # Clone them as the JK object will overwrite. Store as psi4.core.Matrix
    cache["J_A"] = jk.J()[0].clone()
    cache["K_A"] = jk.K()[0].clone()
    cache["J_B"] = jk.J()[1].clone()
    cache["K_B"] = jk.K()[1].clone()
    cache["J_O"] = jk.J()[2].clone()
    # K_O needs transpose
    K_O = jk.K()[2].clone().transpose()
    cache["K_O"] = core.Matrix.from_array(K_O.np)
    cache["K_O"].name = "K_O"

    monA_nr = wfn_A.molecule().nuclear_repulsion_energy()
    monB_nr = wfn_B.molecule().nuclear_repulsion_energy()
    dimer_nr = wfn_A.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy()

    cache["extern_extern_IE"] = 0.0
    if external_potentials:
        dimer_nr += wfn_dimer.external_pot().computeNuclearEnergy(wfn_dimer.molecule())
        if external_potentials.get("A") is not None:
            monA_nr += wfn_A.external_pot().computeNuclearEnergy(wfn_A.molecule())
        if external_potentials.get("B") is not None:
            monB_nr += wfn_B.external_pot().computeNuclearEnergy(wfn_B.molecule())
        if (
            external_potentials.get("A") is not None
            and external_potentials.get("B") is not None
        ):
            cache["extern_extern_IE"] = (
                wfn_A.external_pot().computeExternExternInteraction(
                    wfn_B.external_pot()
                )
            )

    cache["nuclear_repulsion_energy"] = dimer_nr - monA_nr - monB_nr
    return cache


def electrostatics(cache: dict, do_print: bool = True) -> tuple[dict, float]:
    r"""Compute the first-order electrostatic interaction energy :math:`E^{(1)}_{\text{elst}}`.

    Evaluates the Coulombic interaction between unperturbed monomer charge
    distributions (Eq. 4 of Xie et al. 2022):

    .. math::

        E^{(1)}_{\text{elst}} = 2\mathbf{P}^A \cdot \mathbf{V}^B
        + 2\mathbf{P}^B \cdot \mathbf{V}^A
        + 4\mathbf{P}^B \cdot \mathbf{J}^A + V_{\text{nuc}}

    Parameters
    ----------
    cache : dict
        SAPT data cache from :func:`build_sapt_jk_cache`.
    do_print : bool, optional
        Whether to print the result, by default True.

    Returns
    -------
    tuple[dict, float]
        A dictionary ``{'Elst10,r': float}`` and the extern-extern
        interaction energy (zero if no external potentials).
    """
    if do_print:
        core.print_out("\n  ==> E10 Electrostatics <== \n\n")

    # Eq. 4
    Elst10 = 2.0 * _dot(cache["D_A"], cache["V_B"])
    Elst10 += 2.0 * _dot(cache["D_B"], cache["V_A"])
    Elst10 += 4.0 * _dot(cache["D_B"], cache["J_A"])
    Elst10 += cache["nuclear_repulsion_energy"]

    if do_print:
        core.print_out(print_sapt_var("Elst10,r ", Elst10, short=True))
        core.print_out("\n")

    # External Potentials interacting with each other (V_A_ext, V_B_ext)
    extern_extern_ie = 0
    if cache.get("extern_extern_IE"):
        extern_extern_ie = cache["extern_extern_IE"]
        core.print_out(print_sapt_var("Extern-Extern ", extern_extern_ie, short=True))
        core.print_out("\n")

    return {"Elst10,r": Elst10}, extern_extern_ie


def felst(
    cache: dict,
    sapt_elst: dict,
    dimer_wfn: core.Wavefunction,
    wfn_A: core.Wavefunction,
    wfn_B: core.Wavefunction,
    jk: core.JK,
    do_print: bool = True,
) -> dict:
    r"""Compute F-SAPT partitioned electrostatic interaction energy.

    Decomposes the total :math:`E^{(1)}_{\text{elst}}` (Eq. 4) into
    atom-pair and orbital-pair contributions for functional-group
    analysis (F-SAPT). Nuclear-nuclear, nuclear-electron, and
    electron-electron terms are accumulated into the ``Elst_AB``
    breakdown matrix stored in *cache*.

    Parameters
    ----------
    cache : dict
        SAPT data cache with localized orbitals from :func:`flocalization`.
    sapt_elst : dict
        Total SAPT electrostatic energies from :func:`electrostatics`.
    dimer_wfn : core.Wavefunction
        Dimer supermolecular wavefunction.
    wfn_A : core.Wavefunction
        Monomer A wavefunction.
    wfn_B : core.Wavefunction
        Monomer B wavefunction.
    jk : core.JK
        JK integral engine.
    do_print : bool, optional
        Whether to print output, by default True.

    Returns
    -------
    dict
        Updated *cache* dictionary with ``'Elst_AB'`` breakdown matrix.
    """
    core.timer_on("F-SAPT Elst Setup")
    if do_print:
        core.print_out("  ==> F-SAPT Electrostatics <==\n\n")

    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT").upper()
    mol = dimer_wfn.molecule()  # dimer molecule
    dimer_basis = dimer_wfn.basisset()
    nA_atoms = mol.natom()
    nB_atoms = mol.natom()

    # Sizing
    L0A = (
        cache["Locc_A"]
        if link_assignment not in {"SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}
        else cache["Locc_A"]
    )
    L0B = (
        cache["Locc_B"]
        if link_assignment not in {"SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"}
        else cache["Locc_B"]
    )
    na = L0A.np.shape[1]
    nb = L0B.np.shape[1]

    # Initialize breakdown matrix (nA_atoms + na + 1, nB_atoms + nb + 1)
    Elst_AB = np.zeros((nA_atoms + na + 1, nB_atoms + nb + 1))

    # Terms for total
    Elst1_terms = np.zeros(4)  # [0]: a-B, [1]: A-b, [2]: a-b, [3]: nuc

    # Nuclear-nuclear interactions (A <-> B)
    ZA = cache["ZA"]
    ZB = cache["ZB"]

    # Vectorized nuclear-nuclear interactions
    # Build distance matrix
    for A in range(nA_atoms):
        for B in range(nB_atoms):
            if A == B:
                continue
            R = mol.xyz(A).distance(mol.xyz(B))
            if R == 0:
                continue
            E = ZA.np[A] * ZB.np[B] / R
            Elst_AB[A, B] = E
            Elst1_terms[3] += E

    # External A - atom B interactions
    if "A" in cache.get("external_potentials", {}):
        ext_pot_A = cache["external_potentials"]["A"]
        for B in range(nB_atoms):
            atom_mol = core.Molecule([core.Atom(ZB.np[B])])
            atom_mol.set_geometry([mol.xyz(B)])
            interaction = ext_pot_A.computeNuclearEnergy(atom_mol)
            Elst_AB[nA_atoms + na, B] = interaction
            Elst1_terms[3] += interaction

    # External B - atom A interactions
    if "B" in cache.get("external_potentials", {}):
        ext_pot_B = cache["external_potentials"]["B"]
        for A in range(nA_atoms):
            atom_mol = core.Molecule([core.Atom(ZA.np[A])])
            atom_mol.set_geometry([mol.xyz(A)])
            interaction = ext_pot_B.computeNuclearEnergy(atom_mol)
            Elst_AB[A, nB_atoms + nb] = interaction
            Elst1_terms[3] += interaction

    core.timer_off("F-SAPT Elst Setup")
    # => a <-> b (electron-electron interactions via DFHelper) <= //

    # Get auxiliary basis for density fitting
    aux_basis = dimer_wfn.get_basisset("DF_BASIS_SCF")

    # Create DFHelper object
    dfh = core.DFHelper(dimer_basis, aux_basis)

    # Set memory following fisapt.cc logic
    # Note: In C++, doubles_ is the total memory budget in doubles
    # Here we use a reasonable default or get from options if available
    memory_doubles = core.get_memory() // 8
    dfh.set_memory(memory_doubles)
    dfh.set_method("DIRECT_iaQ")
    dfh.set_nthreads(core.get_num_threads())
    dfh.initialize()
    dfh.print_header()

    # Create Matrix objects from numpy arrays for L0A and L0B
    L0A = core.Matrix.from_array(L0A.np)
    L0B = core.Matrix.from_array(L0B.np)

    # Add orbital spaces
    dfh.add_space("a", L0A)
    dfh.add_space("b", L0B)

    # Add transformations for diagonal blocks (a,a|Q) and (b,b|Q)
    dfh.add_transformation("Aaa", "a", "a")
    dfh.add_transformation("Abb", "b", "b")

    # Perform the transformation
    dfh.transform()

    # Extract diagonal 3-index integrals (vectorized)
    nQ = aux_basis.nbf()
    QaC = np.zeros((na, nQ))
    QbC = np.zeros((nb, nQ))

    # Process in batches for better memory efficiency
    batch_size = min(100, max(na, nb))

    # Extract Aaa diagonal elements
    for start_a in range(0, na, batch_size):
        end_a = min(start_a + batch_size, na)
        for a in range(start_a, end_a):
            tensor = dfh.get_tensor("Aaa", [a, a + 1], [a, a + 1], [0, nQ])
            QaC[a, :] = tensor.np.flatten()

    # Extract Abb diagonal elements
    for start_b in range(0, nb, batch_size):
        end_b = min(start_b + batch_size, nb)
        for b in range(start_b, end_b):
            tensor = dfh.get_tensor("Abb", [b, b + 1], [b, b + 1], [0, nQ])
            QbC[b, :] = tensor.np.flatten()

    # Compute electrostatic interaction: Elst10_3 = 4.0 * QaC @ QbC.T
    Elst10_3 = 4.0 * np.dot(QaC, QbC.T)

    # Store in breakdown matrix and accumulate total
    Elst1_terms[2] += np.sum(Elst10_3)
    Elst_AB[nA_atoms : nA_atoms + na, nB_atoms : nB_atoms + nb] += Elst10_3

    # Store QaC and QbC in cache for reuse in f-induction
    cache["Vlocc0A"] = core.Matrix.from_array(QaC)
    cache["Vlocc0B"] = core.Matrix.from_array(QbC)

    # Clear DFHelper spaces for next use
    dfh.clear_spaces()

    core.timer_on("F-SAPT Elst Final")
    # => A <-> b (nuclei A interacting with orbitals b) <= //
    L0B_mat = core.Matrix.from_array(L0B.np)
    L0B_mat.name = "L0B_mat"

    L0A_mat = core.Matrix.from_array(L0A.np)
    ext_pot = core.ExternalPotential()
    for A in range(nA_atoms):
        if ZA.np[A] == 0.0:
            continue

        ext_pot.clear()
        atom_pos = mol.xyz(A)
        ext_pot.addCharge(ZA.np[A], atom_pos[0], atom_pos[1], atom_pos[2])

        Vtemp = ext_pot.computePotentialMatrix(dimer_basis)
        Vtemp_mat = Vtemp.clone()
        Vtemp_mat.name = "Vtemp_mat"

        Vbb = chain_gemm_einsums([L0B_mat, Vtemp_mat, L0B_mat], ["T", "N", "N"])
        Vbb.name = "Vbb"

        # Vectorized diagonal extraction
        diag_Vbb = np.diag(Vbb.np)
        E_vec = 2.0 * diag_Vbb
        Elst1_terms[1] += np.sum(E_vec)
        Elst_AB[A, nB_atoms : nB_atoms + nb] += E_vec

    # Add external-A <-> orbital b interaction
    if "A" in cache.get("external_potentials", {}):
        ext_pot_A = cache["external_potentials"]["A"]
        Vtemp = ext_pot_A.computePotentialMatrix(dimer_basis)

        Vtemp_mat = Vtemp.clone()
        Vbb = chain_gemm_einsums([L0B_mat, Vtemp_mat, L0B_mat], ["T", "N", "N"])

        # Vectorized diagonal extraction
        diag_Vbb = np.diag(Vbb.np)
        E_vec = 2.0 * diag_Vbb
        Elst1_terms[1] += np.sum(E_vec)
        Elst_AB[nA_atoms + na, nB_atoms : nB_atoms + nb] += E_vec

    # => a <-> B (orbitals a interacting with nuclei B) <= //

    for B in range(nB_atoms):
        if ZB.np[B] == 0.0:
            continue

        ext_pot.clear()
        atom_pos = mol.xyz(B)
        ext_pot.addCharge(ZB.np[B], atom_pos[0], atom_pos[1], atom_pos[2])

        Vtemp = ext_pot.computePotentialMatrix(dimer_basis)

        Vtemp_mat = Vtemp.clone()
        Vaa = chain_gemm_einsums([L0A_mat, Vtemp_mat, L0A_mat], ["T", "N", "N"])

        # Vectorized diagonal extraction
        diag_Vaa = np.diag(Vaa.np)
        E_vec = 2.0 * diag_Vaa
        Elst1_terms[0] += np.sum(E_vec)
        Elst_AB[nA_atoms : nA_atoms + na, B] += E_vec

    # Add orbital a <-> external-B interaction
    if "B" in cache.get("external_potentials", {}):
        ext_pot_B = cache["external_potentials"]["B"]
        Vtemp = ext_pot_B.computePotentialMatrix(dimer_basis)

        Vtemp_mat = Vtemp.clone()
        Vaa = chain_gemm_einsums([L0A_mat, Vtemp_mat, L0A_mat], ["T", "N", "N"])

        # Vectorized diagonal extraction
        diag_Vaa = np.diag(Vaa.np)
        E_vec = 2.0 * diag_Vaa
        Elst1_terms[0] += np.sum(E_vec)
        Elst_AB[nA_atoms : nA_atoms + na, nB_atoms + nb] += E_vec

    # Clear DFHelper for next use
    dfh.clear_spaces()
    cache["dfh"] = dfh  # Store DFHelper in cache for potential reuse
    Elst10 = np.sum(Elst1_terms)
    core.print_out(f"    Elst10,r            = {Elst10 * 1000:.8f} [mEh]\n")
    # Ensure the partition reproduces the SAPT elst energy. The two are the
    # same quantity computed through different fitted paths -- this partition
    # through DFHelper, the total through the JK object -- so they agree only
    # to DF consistency, a few times 1e-7 [Eh] on a 24-atom dimer and growing
    # with system size. Check on a relative scale and warn below 1e-8.
    elst_gap = abs(Elst10 - sapt_elst)
    if elst_gap > 1e-8:
        core.print_out(
            "    Warning: localized Elst10,r and SAPT Elst10,r differ by "
            f"{elst_gap * 1000:.3e} [mEh]; the partition is density-fitted "
            "through DFHelper while the total comes from the JK object.\n"
        )
    assert elst_gap < max(1e-6, 1e-4 * abs(sapt_elst)), (
        f"FELST: Localized Elst10,r does not match SAPT Elst10,r!\n{Elst10 =}, {sapt_elst}"
    )

    # Add extern-extern contribution if both external potentials exist
    if "A" in cache.get("external_potentials", {}) and "B" in cache.get(
        "external_potentials", {}
    ):
        ext_pot_A = cache["external_potentials"]["A"]
        ext_pot_B = cache["external_potentials"]["B"]
        ext_ext = ext_pot_A.computeExternExternInteraction(ext_pot_B) * 2.0
        Elst_AB[nA_atoms + na, nB_atoms + nb] += ext_ext

    # Store breakdown matrix in cache
    cache["Elst_AB"] = core.Matrix.from_array(Elst_AB)
    core.timer_off("F-SAPT Elst Final")
    return cache


def fexch(
    cache: dict,
    sapt_exch10_s2: float,
    sapt_exch10: float,
    dimer_wfn: core.Wavefunction,
    wfn_A: core.Wavefunction,
    wfn_B: core.Wavefunction,
    jk: core.JK,
    do_print: bool = True,
) -> dict:
    r"""Compute the F-SAPT first-order exchange partitioning.

    Uses the Exch10(S^2) approximation with orbital partitioning and
    follows Eq. 6 of Xie et al. (2022). Note S = S^{AO}

    .. math::

        E^{(1)}_{\text{exch}}(S^2) = -2(\mathbf{P}^A \mathbf{S} \mathbf{P}^B \mathbf{S} \mathbf{P}^{A,\text{vir}}) \cdot \boldsymbol{\omega}^B
          - 2(\mathbf{P}^B \mathbf{S} \mathbf{P}^A \mathbf{S} \mathbf{P}^{B,\text{vir}}) \cdot \boldsymbol{\omega}^A
          - 2(\mathbf{P}^{A,\text{vir}} \mathbf{S} \mathbf{P}^B) \cdot \mathbf{K}[\mathbf{P}^A \mathbf{S} \mathbf{P}^{B,\text{vir}}]

    Parameters
    ----------
    cache : dict
        SAPT/F-SAPT cache containing localized orbitals and intermediates.
    sapt_exch10_s2 : float
        Total SAPT first-order exchange energy in the :math:`S^2` approximation.
    sapt_exch10 : float
        Total SAPT first-order exchange energy used for optional scaling.
    dimer_wfn : core.Wavefunction
        Dimer wavefunction used for basis and molecular metadata.
    wfn_A : core.Wavefunction
        Monomer A wavefunction.
    wfn_B : core.Wavefunction
        Monomer B wavefunction.
    jk : core.JK
        JK object for Coulomb/exchange intermediates.
    do_print : bool, optional
        Whether to print exchange diagnostics, by default True.

    Returns
    -------
    dict
        Updated cache with ``Exch_AB`` matrix.
    """
    if do_print:
        core.print_out("  ==> F-SAPT Exchange <==\n\n")

    mol = dimer_wfn.molecule()
    nA_atoms = nB_atoms = mol.natom()
    na = cache["Locc_A"].shape[1]
    nb = cache["Locc_B"].shape[1]
    nr = cache["Cvir_A"].shape[1]
    ns = cache["Cvir_B"].shape[1]

    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT")
    na1 = na
    nb1 = nb
    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        na1 = na + 1
        nb1 = nb + 1

    Exch10_2 = 0.0
    Exch10_2_terms = [0.0, 0.0, 0.0]

    Exch_AB = np.zeros((nA_atoms + na1 + 1, nB_atoms + nb1 + 1))

    S = cache["S"]
    V_A = cache["V_A"]
    J_A = cache["J_A"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]

    LoccA = cache["Locc_A"].clone()
    LoccA.name = "LoccA"
    LoccB = cache["Locc_B"].clone()
    LoccB.name = "LoccB"
    CvirA = cache["Cvir_A"]
    CvirB = cache["Cvir_B"]
    CvirA.name = "CvirA"
    CvirB.name = "CvirB"

    dfh = cache["dfh"]

    dfh.add_space("a", LoccA)
    dfh.add_space("r", CvirA)
    dfh.add_space("b", LoccB)
    dfh.add_space("s", CvirB)

    dfh.add_transformation("Aar", "a", "r")
    dfh.add_transformation("Abs", "b", "s")

    dfh.transform()

    W_A = _ein_clone(V_A, name="W_A")
    _axpy(2.0, J_A, W_A)
    W_B = _ein_clone(V_B, name="W_B")
    _axpy(2.0, J_B, W_B)

    # All eight transforms in one graph.  Locc_A^T S is shared by Sab and Sas,
    # Locc_B^T S by Sba and Sbr, and WAbs/WBar are consumed only by WAba/WBab,
    # so they stay graph-owned and never reach the host.
    e_S = _ein(S, "S")
    e_LoA = _ein(LoccA, "LoccA")
    e_LoB = _ein(LoccB, "LoccB")
    e_CvA = _ein(CvirA, "CvirA")
    e_CvB = _ein(CvirB, "CvirB")
    na_o, nb_o = e_LoA.shape[1], e_LoB.shape[1]
    nr_v, ns_v = e_CvA.shape[1], e_CvB.shape[1]

    Sab = _ein_zeros(na_o, nb_o, name="Sab")
    Sba = _ein_zeros(nb_o, na_o, name="Sba")
    Sas = _ein_zeros(na_o, ns_v, name="Sas")
    Sbr = _ein_zeros(nb_o, nr_v, name="Sbr")
    WBab = _ein_zeros(na_o, nb_o, name="WBab")
    WAba = _ein_zeros(nb_o, na_o, name="WAba")
    with graph_block("fexch_transforms") as blk:
        WAbs = blk.tensor(nb_o, ns_v, name="WAbs")
        WBar = blk.tensor(na_o, nr_v, name="WBar")
        chain_into(WAbs, [e_LoB, W_A, e_CvB], "TNN", beta=0.0, name="WAbs")
        chain_into(WBar, [e_LoA, W_B, e_CvA], "TNN", beta=0.0, name="WBar")
        chain_into(Sab, [e_LoA, e_S, e_LoB], "TNN", beta=0.0, name="Sab")
        chain_into(Sba, [e_LoB, e_S, e_LoA], "TNN", beta=0.0, name="Sba")
        chain_into(Sas, [e_LoA, e_S, e_CvB], "TNN", beta=0.0, name="Sas")
        chain_into(Sbr, [e_LoB, e_S, e_CvA], "TNN", beta=0.0, name="Sbr")
        chain_into(WBab, [WBar, Sbr], "NT", beta=0.0, name="WBab")
        chain_into(WAba, [WAbs, Sas], "NT", beta=0.0, name="WAba")

    Sab_np = np.asarray(Sab)
    Sba_np = np.asarray(Sba)
    Sas_np = np.asarray(Sas)
    Sbr_np = np.asarray(Sbr)
    WBab_np = np.asarray(WBab)
    WAba_np = np.asarray(WAba)

    E_exch1 = np.zeros((na, nb))
    E_exch2 = np.zeros((na, nb))

    for a in range(na):
        for b in range(nb):
            E_exch1[a, b] = -2.0 * Sab_np[a, b] * WBab_np[a, b]
            E_exch2[a, b] = -2.0 * Sba_np[b, a] * WAba_np[b, a]

    nQ = dimer_wfn.get_basisset("DF_BASIS_SCF").nbf()
    TrQ = core.Matrix("TrQ", nr, nQ)
    TsQ = core.Matrix("TsQ", ns, nQ)
    TbQ = core.Matrix("TbQ", nb, nQ)
    TaQ = core.Matrix("TaQ", na, nQ)

    dfh.add_disk_tensor("Bab", (na, nb, nQ))

    for a in range(na):
        TrQ.np[:, :] = dfh.get_tensor("Aar", [a, a + 1], [0, nr], [0, nQ]).np.reshape(
            nr, nQ
        )
        TbQ.np[:, :] = np.dot(Sbr_np, TrQ.np)
        dfh.write_disk_tensor("Bab", TbQ, [a, a + 1])

    dfh.add_disk_tensor("Bba", (nb, na, nQ))

    for b in range(nb):
        TsQ.np[:, :] = dfh.get_tensor("Abs", [b, b + 1], [0, ns], [0, nQ]).np.reshape(
            ns, nQ
        )
        TaQ.np[:, :] = np.dot(Sas_np, TsQ.np)
        dfh.write_disk_tensor("Bba", TaQ, [b, b + 1])

    E_exch3 = np.zeros((na, nb))

    for a in range(na):
        TbQ.np[:, :] = dfh.get_tensor("Bab", [a, a + 1], [0, nb], [0, nQ]).np.reshape(
            nb, nQ
        )
        for b in range(nb):
            TaQ_slice = dfh.get_tensor(
                "Bba", [b, b + 1], [a, a + 1], [0, nQ]
            ).np.reshape(nQ)
            E_exch3[a, b] = -2.0 * np.dot(TbQ.np[b, :], TaQ_slice)

    for a in range(na):
        for b in range(nb):
            Exch_AB[a + nA_atoms, b + nB_atoms] = (
                E_exch1[a, b] + E_exch2[a, b] + E_exch3[a, b]
            )
            Exch10_2_terms[0] += E_exch1[a, b]
            Exch10_2_terms[1] += E_exch2[a, b]
            Exch10_2_terms[2] += E_exch3[a, b]

    Exch10_2 = sum(Exch10_2_terms)

    if do_print:
        core.print_out(f"    Exch10(S^2)         = {Exch10_2 * 1000:18.10f} [mEh]\n")
        core.print_out(
            f"    Exch10(S^2)-true    = {sapt_exch10_s2 * 1000:18.10f} [mEh]\n"
        )
        core.print_out(f"    Exch10-true         = {sapt_exch10 * 1000:18.10f} [mEh]\n")
        core.print_out("\n")

    if core.get_option("FISAPT", "FISAPT_FSAPT_EXCH_SCALE"):
        scale = sapt_exch10 / Exch10_2
        Exch_AB *= scale
        if do_print:
            core.print_out(
                f"    Scaling F-SAPT Exch10(S^2) by {scale:11.3E} to match Exch10\n\n"
            )

    cache["Exch_AB"] = core.Matrix.from_array(Exch_AB)
    dfh.clear_spaces()
    return cache


# --------------------------------------------------------------------------
# einsums v2 ComputeGraph
#
# v1 had no choice but to walk a matrix chain left to right, one
# ``ein.core.gemm`` at a time, which is why the chains in this module used to
# carry hand-hoisted intermediates: a subproduct shared by several terms had
# to be spotted by hand or paid for twice, and the hoisted value then had to
# be threaded through the call sites.
#
# v2 can hand a whole block of chains over instead.  Inside a
# :class:`graph_block` every einsums op is recorded rather than run, and on
# exit the default pipeline gets to restructure the block: CSE collapses the
# subproducts the terms share, DistributiveFactoring folds the linear
# combinations that feed one accumulator into a single contraction, and
# ContractionPlanning re-parenthesizes what is left.  The terms can therefore
# be written the way the equations read, with no hoisting at all.
#
# The one rule is that a chain's interior values must be graph-owned and
# unaliased -- a value read outside the chain makes the eliminated write
# observable, so the passes decline the chain rather than change semantics.
# That is exactly what hand-hoisting violates, and why the hoisting has to go
# away for the graph to be able to do anything.
# --------------------------------------------------------------------------

_ACTIVE_BLOCK = None


class graph_block:
    """Capture a block of matrix-chain algebra into one einsums ComputeGraph.

    Nothing inside the ``with`` body executes until the block exits, so only
    tensor algebra may appear in it: reading a value back on the host
    (:func:`_mat`, :func:`_dot`, ``numpy.asarray``, a ``core.Matrix``
    constructor, a JK build) would see a buffer the graph has not written yet.
    Values needed after the block must be written into tensors created
    *outside* it, since graph-owned intermediates are freed after their last
    consumer.

    Entering a block while one is already active is a no-op that joins the
    outer graph, so a captured function may call another one.
    """

    def __init__(self, name: str, optimize: bool = True):
        self.name = name
        self._optimize = optimize
        self.g = None
        self._cap = None
        self._outer = False

    def tensor(self, *dims, name: str = "tmp"):
        """A graph-owned intermediate, or a plain tensor outside a block."""
        dims = [int(d) for d in dims]
        if self.g is None:
            return ein.create_zero_tensor(name, dims)
        return self.g.create_tensor(name, dims)

    def __enter__(self):
        global _ACTIVE_BLOCK
        if _ACTIVE_BLOCK is not None:
            return _ACTIVE_BLOCK
        self._outer = True
        self.g = cg.Graph(self.name)
        self._cap = cg.capture(self.g)
        self._cap.__enter__()
        _ACTIVE_BLOCK = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_BLOCK
        if not self._outer:
            return False
        self._cap.__exit__(exc_type, exc, tb)
        _ACTIVE_BLOCK = None
        if exc_type is None:
            if self._optimize:
                self.g.optimize()
            self.g.set_executor(cg.SequentialExecutor())
            self.g.execute()
        # Release the Graph as soon as it has run.  optimize()'s free pass
        # drops most intermediates at their last consumer, but the ones it
        # declines stay alive as long as the Graph does, which otherwise means
        # until the enclosing function returns.  Measured on peptide/aug-cc-
        # pVDZ this recovers 20 MB at fdisp0_setup and 7 MB at induction_jk_C.
        self.g = None
        return False


def _gemm_spec(trans_a: str, trans_b: str) -> str:
    """The einsum spec for ``C = op(A) @ op(B)`` with the given transposes."""
    return "ij <- %s ; %s" % ("ik" if trans_a == "N" else "ki",
                              "kj" if trans_b == "N" else "jk")


def chain_into(out, factors: list, transposes: str = None, coef: float = 1.0,
               beta: float = 1.0, name: str = "chain"):
    """``out = beta * out + coef * F1 F2 ... Fn``, as one matrix chain.

    Every factor must already be an einsums tensor.  Interior products go to
    graph-owned intermediates when a :class:`graph_block` is active, which is
    what lets the passes eliminate or re-order them; outside a block the chain
    is evaluated eagerly, left to right, into ordinary tensors.

    Parameters
    ----------
    out
        Destination tensor, accumulated into unless ``beta`` is 0.
    factors
        The chain, in order.
    transposes
        One flag per factor, ``"N"`` or ``"T"`` (default all ``"N"``).
    coef, beta
        ``out = beta * out + coef * (chain)``.
    """
    n = len(factors)
    tr = transposes if transposes is not None else "N" * n
    blk = _ACTIVE_BLOCK
    if n == 1:
        if beta != 1.0:
            ein.linalg.scale(beta, out)
        ein.linalg.axpy(coef, factors[0], out)
        return out
    A, at = factors[0], tr[0]
    for i in range(1, n):
        B, bt = factors[i], tr[i]
        if i == n - 1:
            dest, c_pf, ab_pf = out, beta, coef
        else:
            rows = A.shape[1] if at == "T" else A.shape[0]
            cols = B.shape[0] if bt == "T" else B.shape[1]
            dest = (blk.tensor(rows, cols, name="%s_%d" % (name, i)) if blk
                    else _ein_zeros(rows, cols, name="%s_%d" % (name, i)))
            c_pf, ab_pf = 0.0, 1.0
        if blk is not None:
            ein.einsum(_gemm_spec(at, bt), dest, A, B, c_pf=c_pf, ab_pf=ab_pf)
        else:
            ein.linalg.gemm(ab_pf, A, B, c_pf, dest,
                            trans_a=(at == "T"), trans_b=(bt == "T"))
        A, at = dest, "N"
    return out


def chain_sum(out, terms: list, beta: float = 0.0, name: str = "term"):
    """``out = beta * out + sum_t coef_t * (chain_t)`` over a list of chains.

    ``terms`` holds ``(coef, factors, transposes)`` triples.  Handed to a
    :class:`graph_block` the whole sum becomes one graph, and the subproducts
    the terms have in common are found by the passes rather than by hand.
    """
    for i, (coef, factors, tr) in enumerate(terms):
        chain_into(out, factors, tr, coef=coef,
                   beta=(beta if i == 0 else 1.0), name="%s%d" % (name, i))
    return out


def build_ind_pot(vars: dict) -> core.Matrix:
    r"""Build the induction potential in the MO basis for one monomer due to the other.

    Constructs :math:`\tilde{\boldsymbol{\omega}}^X` (Eq. 16 of Xie et al. 2022).
    By swapping A/B labels in ``vars``, the potential for either monomer can
    be computed.

    .. math::

        \tilde{\boldsymbol{\omega}}^X = (\mathbf{C}^{Y,\text{occ}})^\dagger \boldsymbol{\omega}^X \mathbf{C}^{Y,\text{vir}}

    Parameters
    ----------
    vars : dict
        Dictionary containing the matrices required for the induction
        potential build, including ``V_B``, ``J_B``, ``Cocc_A``, and ``Cvir_A``.

    Returns
    -------
    core.Matrix
        Induction potential in the occupied-virtual MO block.
    """
    w_B = _ein_clone(vars["V_B"], name="w_B")
    _axpy(2.0, vars["J_B"], w_B)
    return chain_gemm_einsums(
        [vars["Cocc_A"], w_B, vars["Cvir_A"]],
        ["T", "N", "N"],
    )


def build_exch_ind_pot_AB(vars: dict) -> core.Matrix:
    r"""Build the exchange-induction potential for monomer A due to monomer B.

    Constructs the exchange-induction operator in the MO basis following
    Eq. 17 of Xie et al. (2022), involving overlap-weighted density matrix
    products and Coulomb/exchange contractions.

    The :math:`S^2` exchange-induction energy for the A←B direction (Eq. 17):

    .. math::

        E^{(2)}_{\text{exch-ind}}(S^2)(A \leftarrow B)
          = 2\mathbf{x}^A \cdot \bigl(
            (\mathbf{C}^{A,\text{occ}})^\dagger
            \bigl[
              -\mathbf{K}^B
              - 2\mathbf{J}[\mathbf{O}]
              + \mathbf{K}[\mathbf{O}]
              + 2\mathbf{J}[\mathbf{P}^B \mathbf{S} \mathbf{O}]
              + \mathbf{S} \mathbf{P}^B
               \times (-\mathbf{h}^A
                 + \mathbf{S} \mathbf{P}^A \boldsymbol{\omega}^B
                 + \boldsymbol{\omega}^A \mathbf{P}^B \mathbf{S}
                 - \mathbf{K}[\mathbf{O}]^T)
              + (-\mathbf{h}^B
                 + \boldsymbol{\omega}^B \mathbf{P}^A \mathbf{S}
                 - \mathbf{K}[\mathbf{O}])
                \mathbf{P}^B \mathbf{S}
            \bigr]
            \mathbf{C}^{A,\text{vir}}
          \bigr)

    Parameters
    ----------
    vars : dict
        Dictionary of AO and MO intermediates required for the A<-B
        exchange-induction construction.

    Returns
    -------
    core.Matrix
        Exchange-induction potential for monomer A in the occupied-virtual
        MO block.
    """

    S = _ein(vars["S"], name="S")
    D_A = _ein(vars["D_A"], name="D_A")
    D_B = _ein(vars["D_B"], name="D_B")
    V_A = _ein(vars["V_A"], name="V_A")
    V_B = _ein(vars["V_B"], name="V_B")
    J_A = _ein(vars["J_A"], name="J_A")
    J_B = _ein(vars["J_B"], name="J_B")
    K_A = _ein(vars["K_A"], name="K_A")
    K_B = _ein(vars["K_B"], name="K_B")
    J_O = _ein(vars["J_O"], name="J_O")
    K_O = _ein(vars["K_O"], name="K_O")
    J_P_B = _ein(vars["J_P_B"], name="J_P_B")
    Cocc_A = _ein(vars["Cocc_A"], name="Cocc_A")
    Cvir_A = _ein(vars["Cvir_A"], name="Cvir_A")

    # Eq. 17, transcribed the way the equation reads, with no hoisting: S D_B
    # is shared by eight of these terms, D_B S by four and D_A S D_B S by two,
    # and finding those is the graph's job, not the caller's.
    terms = [
        (-1.0, [S, D_B, V_A], "NNN"),
        (-2.0, [S, D_B, J_A], "NNN"),
        (+1.0, [S, D_B, K_A], "NNN"),
        (+1.0, [S, D_B, S, D_A, V_B], "NNNNN"),
        (+2.0, [S, D_B, S, D_A, J_B], "NNNNN"),
        (+1.0, [S, D_B, V_A, D_B, S], "NNNNN"),
        (+2.0, [S, D_B, J_A, D_B, S], "NNNNN"),
        (-1.0, [S, D_B, K_O], "NNT"),
        (-1.0, [V_B, D_B, S], "NNN"),
        (-2.0, [J_B, D_B, S], "NNN"),
        (+1.0, [K_B, D_B, S], "NNN"),
        (+1.0, [V_B, D_A, S, D_B, S], "NNNNN"),
        (+2.0, [J_B, D_A, S, D_B, S], "NNNNN"),
        (-1.0, [K_O, D_B, S], "NNN"),
    ]

    # The result is written into a tensor allocated outside the block, since
    # the graph frees anything it owns once the last consumer has run.
    EX_A_MO = _ein_zeros(Cocc_A.shape[1], Cvir_A.shape[1], name="EX_A_MO")
    with graph_block("exch_ind_pot_AB") as blk:
        EX_A = blk.tensor(S.shape[0], S.shape[0], name="EX_A")
        chain_into(EX_A, [K_B], coef=-1.0, beta=0.0)
        _axpy(-2.0, J_O, EX_A)
        _axpy(1.0, K_O, EX_A)
        _axpy(2.0, J_P_B, EX_A)
        chain_sum(EX_A, terms, beta=1.0)
        chain_into(EX_A_MO, [Cocc_A, EX_A, Cvir_A], "TNN", beta=0.0, name="mo")
    return _mat(EX_A_MO)


def build_exch_ind_pot_BA(vars: dict) -> core.Matrix:
    r"""Build the exchange-induction potential for monomer B due to monomer A.

    Analogous to :func:`build_exch_ind_pot_AB` with A/B roles swapped,
    following Eq. 17 of Xie et al. (2022).

    Parameters
    ----------
    vars : dict
        Dictionary of AO and MO intermediates required for the B<-A
        exchange-induction construction.

    Returns
    -------
    core.Matrix
        Exchange-induction potential for monomer B in the occupied-virtual
        MO block.
    """

    S = _ein(vars["S"], name="S")
    D_A = _ein(vars["D_A"], name="D_A")
    D_B = _ein(vars["D_B"], name="D_B")
    V_A = _ein(vars["V_A"], name="V_A")
    V_B = _ein(vars["V_B"], name="V_B")
    J_A = _ein(vars["J_A"], name="J_A")
    J_B = _ein(vars["J_B"], name="J_B")
    K_A = _ein(vars["K_A"], name="K_A")
    K_B = _ein(vars["K_B"], name="K_B")
    J_O = _ein(vars["J_O"], name="J_O")
    K_O = _ein(vars["K_O"], name="K_O")
    J_P_A = _ein(vars["J_P_A"], name="J_P_A")
    Cocc_B = _ein(vars["Cocc_B"], name="Cocc_B")
    Cvir_B = _ein(vars["Cvir_B"], name="Cvir_B")

    # Eq. 17 with A and B swapped.  K_O is the mixed-occupied exchange matrix
    # and is not symmetric, so its two terms carry the transposes the A<-B
    # case does not.
    terms = [
        (-1.0, [S, D_A, V_B], "NNN"),
        (-2.0, [S, D_A, J_B], "NNN"),
        (+1.0, [S, D_A, K_B], "NNN"),
        (+1.0, [S, D_A, S, D_B, V_A], "NNNNN"),
        (+2.0, [S, D_A, S, D_B, J_A], "NNNNN"),
        (+1.0, [S, D_A, V_B, D_A, S], "NNNNN"),
        (+2.0, [S, D_A, J_B, D_A, S], "NNNNN"),
        (-1.0, [S, D_A, K_O], "NNN"),
        (-1.0, [V_A, D_A, S], "NNN"),
        (-2.0, [J_A, D_A, S], "NNN"),
        (+1.0, [K_A, D_A, S], "NNN"),
        (+1.0, [V_A, D_B, S, D_A, S], "NNNNN"),
        (+2.0, [J_A, D_B, S, D_A, S], "NNNNN"),
        (-1.0, [K_O, D_A, S], "TNN"),
    ]

    EX_B_MO = _ein_zeros(Cocc_B.shape[1], Cvir_B.shape[1], name="EX_B_MO")
    with graph_block("exch_ind_pot_BA") as blk:
        EX_B = blk.tensor(S.shape[0], S.shape[0], name="EX_B")
        chain_into(EX_B, [K_A], coef=-1.0, beta=0.0)
        _axpy(-2.0, J_O, EX_B)
        _axpy(1.0, K_O.T, EX_B)
        _axpy(2.0, J_P_A, EX_B)
        chain_sum(EX_B, terms, beta=1.0)
        chain_into(EX_B_MO, [Cocc_B, EX_B, Cvir_B], "TNN", beta=0.0, name="mo")
    return _mat(EX_B_MO)


def build_exch_ind_pot_avg(vars: dict) -> core.Matrix:
    r"""Build the averaged exchange-induction potential for link-orbital SAO/SIAO methods.

    Uses the older :func:`core.triplet` API for matrix triple products.
    This variant handles the SAO/SIAO link-orbital partitioning where
    average exchange-induction potentials are needed.

    Parameters
    ----------
    vars : dict
        Dictionary of AO and MO intermediates required to construct the
        averaged exchange-induction potential.

    Returns
    -------
    core.Matrix
        Averaged exchange-induction potential in the occupied-virtual
        MO block.
    """
    Ca = vars["Cocc_A"]
    Cr = vars["Cvir_A"]

    S = vars["S"]

    D_A = vars["D_A"]
    J_A = vars["J_A"]
    K_A = vars["K_A"]
    V_A = vars["V_A"]
    D_B = vars["D_B"]
    J_B = vars["J_B"]
    K_B = vars["K_B"]
    V_B = vars["V_B"]
    D_Y = vars["D_Y"]
    K_Y = vars["K_Y"]

    J_O = vars["J_O"]
    K_O = vars["K_O"]
    K_AOY = vars["K_AOY"]

    J_P = vars["J_P"]
    J_PYAY = vars["J_PYAY"]

    W = core.Matrix.from_array(-K_B.np)

    T = core.triplet(S, D_B, J_A, False, False, False)
    W.np[:] += -2.0 * T.np

    W.np[:] += K_O.np

    W.np[:] += -2.0 * J_O.np

    T = core.triplet(S, D_B, K_A, False, False, False)
    W.np[:] += T.np

    T = core.triplet(J_B, D_B, S, False, False, False)
    W.np[:] += -2.0 * T.np

    T = core.triplet(K_B, D_B, S, False, False, False)
    W.np[:] += T.np
    T = core.triplet(K_Y, D_Y, S, False, False, False)
    W.np[:] += T.np

    T1 = core.triplet(S, D_B, J_A, False, False, False)
    T = core.triplet(T1, D_B, S, False, False, False)
    W.np[:] += 2.0 * T.np
    T1 = core.triplet(S, D_Y, J_A, False, False, False)
    T = core.triplet(T1, D_Y, S, False, False, False)
    W.np[:] += 2.0 * T.np

    T1 = core.triplet(J_B, D_A, S, False, False, False)
    T = core.triplet(T1, D_B, S, False, False, False)
    W.np[:] += 2.0 * T.np

    T = core.triplet(K_O, D_B, S, False, False, False)
    W.np[:] += -1.0 * T.np
    T = core.triplet(K_AOY, D_Y, S, False, False, False)
    W.np[:] += -1.0 * T.np

    W.np[:] += 2.0 * J_P.np
    W.np[:] += 2.0 * J_PYAY.np

    T1 = core.triplet(S, D_B, S, False, False, False)
    T = core.triplet(T1, D_A, J_B, False, False, False)
    W.np[:] += 2.0 * T.np

    T = core.triplet(S, D_B, K_O, False, False, True)
    W.np[:] += -1.0 * T.np
    T = core.triplet(S, D_Y, K_AOY, False, False, True)
    W.np[:] += -1.0 * T.np

    T = core.triplet(S, D_B, V_A, False, False, False)
    W.np[:] += -1.0 * T.np

    T = core.triplet(V_B, D_B, S, False, False, False)
    W.np[:] += -1.0 * T.np

    T1 = core.triplet(S, D_B, V_A, False, False, False)
    T = core.triplet(T1, D_B, S, False, False, False)
    W.np[:] += T.np
    T1 = core.triplet(S, D_Y, V_A, False, False, False)
    T = core.triplet(T1, D_Y, S, False, False, False)
    W.np[:] += T.np

    T1 = core.triplet(V_B, D_A, S, False, False, False)
    T = core.triplet(T1, D_B, S, False, False, False)
    W.np[:] += T.np

    T1 = core.triplet(S, D_B, S, False, False, False)
    T = core.triplet(T1, D_A, V_B, False, False, False)
    W.np[:] += T.np

    return core.triplet(Ca, W, Cr, True, False, False)


def find(
    cache: dict,
    scalars: dict,
    dimer_wfn: core.Wavefunction,
    wfn_A: core.Wavefunction,
    wfn_B: core.Wavefunction,
    jk: core.JK,
    do_print: bool = True,
) -> dict:
    r"""Compute the F-SAPT induction partitioning.

    Partitions the second-order induction and exchange-induction energies into
    atomic pair contributions for F-SAPT analysis. Computes both uncoupled and
    (optionally) coupled induction using the CPSCF solver.

    Parameters
    ----------
    cache : dict
        SAPT data cache containing orbital coefficients, density matrices, and integrals.
    scalars : dict
        Reference scalar energies for validation.
    dimer_wfn : core.Wavefunction
        Dimer wavefunction.
    wfn_A : core.Wavefunction
        Monomer A wavefunction.
    wfn_B : core.Wavefunction
        Monomer B wavefunction.
    jk : core.JK
        JK integral engine.
    do_print : bool, optional
        Whether to print results, by default True.

    Returns
    -------
    dict
        Updated cache with ``Ind_AB`` and ``IndAB_AB``/``IndBA_AB`` matrices.
    """
    if do_print:
        core.print_out("  ==> F-SAPT Induction <==\n\n")

    ind_scale = core.get_option("FISAPT", "FISAPT_FSAPT_IND_SCALE")
    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT")

    mol = dimer_wfn.molecule()
    nA = mol.natom()
    nB = mol.natom()
    na = cache["Locc_A"].shape[1]
    nb = cache["Locc_B"].shape[1]
    nr = cache["Cvir_A"].shape[1]
    ns = cache["Cvir_B"].shape[1]

    na1 = na
    nb1 = nb
    # for the SAOn/SIAOn variants, we sometimes need na1 = na+1 (with link
    # orbital) and sometimes na (without) - be careful with this!
    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        na1 = na + 1
        nb1 = nb + 1

    Locc_A = cache["Locc_A"].clone()
    Locc_A.name = "LoccA"
    Locc_B = cache["Locc_B"].clone()
    Locc_B.name = "LoccB"

    Uocc_A = cache["Uocc_A"]
    Uocc_B = cache["Uocc_B"]

    Cocc_A = cache["Cocc_A"]
    Cocc_B = cache["Cocc_B"]
    Cvir_A = cache["Cvir_A"]
    Cvir_B = cache["Cvir_B"]

    eps_occ_A = cache["eps_occ_A"]
    eps_occ_B = cache["eps_occ_B"]
    eps_vir_A = cache["eps_vir_A"]
    eps_vir_B = cache["eps_vir_B"]

    # Collect relevant variables
    S = cache["S"]
    D_A = cache["D_A"]
    V_A = cache["V_A"]
    J_A = cache["J_A"]
    K_A = cache["K_A"]
    D_B = cache["D_B"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]
    K_B = cache["K_B"]
    J_O = cache["J_O"]
    K_O = cache["K_O"]
    J_P_A = cache["J_P_A"]
    J_P_B = cache["J_P_B"]

    aux_basis = dimer_wfn.get_basisset("DF_BASIS_SCF")
    nQ = aux_basis.nbf()

    dfh = cache["dfh"]

    # ESPs - external potential entries
    dfh.add_disk_tensor("WBar", (nB + nb1 + 1, na, nr))
    dfh.add_disk_tensor("WAbs", (nA + na1 + 1, nb, ns))

    core.timer_on("FIND:nucESP")
    # Nuclear contribution to the ESPs, one batch of centers at a time.  Two
    # things differ from the naive per-center loop:
    #
    #   1. The integrals come from MintsHelper.ao_multipole_potential(0, R),
    #      whose order-0 component is minus the AO potential of a +1 point
    #      charge at R (verified against ExternalPotential to 9e-16).  It builds
    #      one integral object and one nbf x nbf matrix, where
    #      ExternalPotential::computePotentialMatrix rebuilds an IntegralFactory
    #      plus one PotentialInt and one nbf x nbf matrix per OpenMP thread on
    #      every call -- 2.2x more wall time per center in isolation, and 3.3x
    #      (nanotube) to 3.7x (peptide) inside find().  Centers with zero
    #      charge, which the link-atom bookkeeping can produce, are skipped
    #      instead of contributing an integral pass that scales to zero.
    #   2. The centers are accumulated in (n, A, m) layout, so the occ/vir
    #      backtransform for a whole batch is two GEMMs inside one graph_block
    #      instead of a chain_gemm_einsums triple product per center, and the
    #      batch reaches disk in a single write.
    mints = core.MintsHelper(dimer_wfn.basisset())
    nn = _arr(Cocc_A).shape[0]

    def nuclear_esp(ncenter, Z, Cocc, Cvir, no, nv, tensor_name):
        """ESP of each of the first *ncenter* nuclei in one monomer's occ/vir
        basis, written into the disk tensor *tensor_name*."""
        for A0 in range(0, ncenter, NESP_BLOCK):
            A1 = min(A0 + NESP_BLOCK, ncenter)
            nblk = A1 - A0

            core.timer_on("FIND:nucESP:int")
            VnAm = np.zeros((nn, nblk, nn))
            for A in range(A0, A1):
                if Z[A] == 0.0:
                    continue
                p = mol.xyz(A)
                V = _arr(mints.ao_multipole_potential(0, [p[0], p[1], p[2]])[0])
                np.multiply(V, -Z[A], out=VnAm[:, A - A0, :])
            core.timer_off("FIND:nucESP:int")

            core.timer_on("FIND:nucESP:xform")
            T_V = _ein(VnAm, name="VnAm")
            T_Co = _ein(Cocc, name="Cocc")
            T_Cv = _ein(Cvir, name="Cvir")
            T_W = _ein_zeros(no, nblk, nv, name="Wblk")
            with graph_block("find_nucesp") as blk:
                T = blk.tensor(no, nblk, nn, name="T")
                # Merging a tensor's *trailing* axes into a reshape_view gives a
                # faithful GEMM operand; merging the leading ones does not (the
                # same leading-index view defect as einsums' batched_gemm), so
                # the second contraction keeps the center axis explicit and lets
                # einsums batch it.
                ein.einsum("ij <- ki ; kj", T.reshape_view([no, nblk * nn]),
                           T_Co, T_V.reshape_view([nn, nblk * nn]))
                ein.einsum("bAs <- bAm ; ms", T_W, T, T_Cv)
            Wblk = np.ascontiguousarray(
                np.transpose(np.asarray(T_W), (1, 0, 2))
            ).reshape(nblk * no, nv)
            core.timer_off("FIND:nucESP:xform")

            core.timer_on("FIND:nucESP:write")
            dfh.write_disk_tensor(tensor_name, core.Matrix.from_array(Wblk), (A0, A1))
            core.timer_off("FIND:nucESP:write")

    nuclear_esp(nA, cache["ZA"].np, Cocc_B, Cvir_B, nb, ns, "WAbs")
    nuclear_esp(nB, cache["ZB"].np, Cocc_A, Cvir_A, na, nr, "WBar")
    core.timer_off("FIND:nucESP")
    core.timer_on("FIND:dfhxform")
    dfh.add_space("a", core.Matrix.from_array(Cocc_A))
    dfh.add_space("r", core.Matrix.from_array(Cvir_A))
    dfh.add_space("b", core.Matrix.from_array(Cocc_B))
    dfh.add_space("s", core.Matrix.from_array(Cvir_B))

    dfh.add_transformation("Aar", "a", "r")
    dfh.add_transformation("Abs", "b", "s")

    dfh.transform()

    core.timer_off("FIND:dfhxform")
    core.timer_on("FIND:elecESP")
    RaC = cache["Vlocc0A"]  # na x nQ
    RbD = cache["Vlocc0B"]  # nb x nQ

    TsQ = core.Matrix("TsQ", ns, nQ)
    T1As = core.Matrix("T1As", na1, ns)
    for B in range(nb):
        dfh.fill_tensor("Abs", TsQ, [B, B + 1], [0, ns], [0, nQ])
        TsQ = core.Matrix.from_array(TsQ.np[0, :, :])
        T1As.gemm(False, True, 2.0, RaC, TsQ, 0.0)
        for A in range(na1):
            row_view = core.Matrix.from_array(T1As.np[A : A + 1, :])
            dfh.write_disk_tensor("WAbs", row_view, (nA + A, nA + A + 1), (B, B + 1))

    TrQ = core.Matrix("TrQ", nr, nQ)
    T1Br = core.Matrix("T1Br", nb1, nr)
    for A in range(na):
        dfh.fill_tensor("Aar", TrQ, [A, A + 1], [0, nr], [0, nQ])
        TrQ = core.Matrix.from_array(TrQ.np[0, :, :])
        T1Br.gemm(False, True, 2.0, RbD, TrQ, 0.0)
        for B in range(nb1):
            row_view = core.Matrix.from_array(T1Br.np[B : B + 1, :])
            dfh.write_disk_tensor("WBar", row_view, (nB + B, nB + B + 1), (A, A + 1))

    core.timer_off("FIND:elecESP")
    core.timer_on("FIND:pots")
    uAT = core.Matrix("uAT", nb, ns)
    wAT = core.Matrix("wAT", nb, ns)
    uBT = core.Matrix("uBT", na, nr)
    wBT = core.Matrix("wBT", na, nr)

    if link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        D_X = core.doublet(cache["thislinkA"], cache["thislinkA"], False, True)
        D_Y = core.doublet(cache["thislinkB"], cache["thislinkB"], False, True)
        J_X = cache["JLA"]
        K_X = cache["KLA"]
        J_Y = cache["JLB"]
        K_Y = cache["KLB"]

        K_AOY = cache["K_AOY"]
        K_XOB = core.Matrix.from_array(cache["K_XOB"].np.T)
        J_P_YAY = cache["J_P_YAY"]
        J_P_XBX = cache["J_P_XBX"]

        mapA = {
            "Cocc_A": Locc_A,
            "Cvir_A": Cvir_A,
            "S": S,
            "D_A": D_A,
            "V_A": V_A,
            "J_A": J_A,
            "K_A": K_A,
            "D_B": D_B,
            "V_B": V_B,
            "J_B": J_B,
            "K_B": K_B,
            "D_X": D_X,
            "J_X": J_X,
            "K_X": K_X,
            "D_Y": D_Y,
            "J_Y": J_Y,
            "K_Y": K_Y,
            "J_O": J_O,
            "K_O": K_O,
            "K_AOY": K_AOY,
            "J_P": J_P_A,
            "J_PYAY": J_P_YAY,
        }

        raise NotImplementedError("find() not ready yet for link orbitals")
        wBT = build_ind_pot(mapA)
        uBT = build_exch_ind_pot_avg(mapA)

        K_O.np = K_O.np
        K_O.np[:] = K_O.np.T

        mapB = {
            "Cocc_A": Locc_B,
            "Cvir_A": Cvir_B,
            "S": S,
            "D_A": D_B,
            "V_A": V_B,
            "J_A": J_B,
            "K_A": K_B,
            "D_B": D_A,
            "V_B": V_A,
            "J_B": J_A,
            "K_B": K_A,
            "D_X": D_Y,
            "J_X": J_Y,
            "K_X": K_Y,
            "D_Y": D_X,
            "J_Y": J_X,
            "K_Y": K_X,
            "J_O": J_O,
            "K_O": K_O,
            "K_AOY": K_XOB,
            "J_P": J_P_B,
            "J_PYAY": J_P_XBX,
        }

        wAT = build_ind_pot(mapB)
        uAT = build_exch_ind_pot_avg(mapB)

        K_O.np[:] = K_O.np.T

    else:
        mapA = {
            "S": S,
            "J_O": J_O,
            "K_O": K_O,
            "Cocc_A": Locc_A,
            "Cvir_A": Cvir_A,
            "D_A": D_A,
            "V_A": V_A,
            "J_A": J_A,
            "K_A": K_A,
            "J_P_A": J_P_A,
            "Cocc_B": Locc_B,
            "Cvir_B": Cvir_B,
            "D_B": D_B,
            "V_B": V_B,
            "J_B": J_B,
            "K_B": K_B,
            "J_P_B": J_P_B,
        }
        wBT = build_ind_pot(
            {
                "V_B": V_B,
                "J_B": J_B,
                "Cocc_A": Locc_A,
                "Cvir_A": Cvir_A,
            }
        )
        wAT = build_ind_pot(
            {
                "V_B": V_A,
                "J_B": J_A,
                "Cocc_A": Locc_B,
                "Cvir_A": Cvir_B,
            }
        )
        uBT = build_exch_ind_pot_AB(mapA)
        uAT = build_exch_ind_pot_BA(mapA)

    core.timer_off("FIND:pots")
    wBT.name = "wBT"
    uBT.name = "uBT"
    wAT.name = "wAT"
    uAT.name = "uAT"
    V_B.name = "V_B"
    J_B.name = "J_B"

    Ind20u_AB_terms = core.Matrix("Ind20 [A<-B] (a x B)", na, nB + nb1 + 1)
    Ind20u_BA_terms = core.Matrix("Ind20 [B<-A] (A x b)", nA + na1 + 1, nb)
    Ind20u_AB_termsp = Ind20u_AB_terms.np
    Ind20u_BA_termsp = Ind20u_BA_terms.np

    Ind20u_AB = 0.0
    Ind20u_BA = 0.0

    ExchInd20u_AB_terms = core.Matrix("ExchInd20 [A<-B] (a x B)", na, nB + nb1 + 1)
    ExchInd20u_BA_terms = core.Matrix("ExchInd20 [B<-A] (A x b)", nA + na1 + 1, nb)
    ExchInd20u_AB_termsp = ExchInd20u_AB_terms.np
    ExchInd20u_BA_termsp = ExchInd20u_BA_terms.np

    ExchInd20u_AB = 0.0
    ExchInd20u_BA = 0.0

    # sna = snB = snb = snA = 0
    # sExchInd20u_AB_terms = core.Matrix("sExchInd20 [A<-B] (a x B)", sna, snB + snb + 1)
    # sExchInd20u_BA_terms = core.Matrix("sExchInd20 [B<-A] (A x b)", snA + sna + 1, snb)
    # sExchInd20u_AB_termsp = sExchInd20u_AB_terms.np
    # sExchInd20u_BA_termsp = sExchInd20u_BA_terms.np
    #
    # sExchInd20u_AB = 0.0
    # sExchInd20u_BA = 0.0

    Indu_AB_terms = core.Matrix("Ind [A<-B] (a x B)", na, nB + nb1 + 1)
    Indu_BA_terms = core.Matrix("Ind [B<-A] (A x b)", nA + na1 + 1, nb)
    Indu_AB = 0.0
    Indu_BA = 0.0

    # Commented out terms are for sSAPT0 scaling... do we really want sSAPT0 support here?
    # sIndu_AB_terms = core.Matrix("sInd [A<-B] (a x B)", sna, snB + snb + 1)
    # sIndu_BA_terms = core.Matrix("sInd [B<-A] (A x b)", snA + sna + 1, snb)
    # sIndu_AB_termsp = sIndu_AB_terms.np
    # sIndu_BA_termsp = sIndu_BA_terms.np
    # sIndu_AB = 0.0
    # sIndu_BA = 0.0

    core.timer_on("FIND:uncAB")
    # ==> A <- B Uncoupled <==
    if dimer_wfn.has_potential_variable("B"):
        Var = core.triplet(Cocc_A, cache["VB_extern"], Cvir_A, True, False, False)
        dfh.write_disk_tensor("WBar", Var, (nB + nb1, nB + nb1 + 1))
    else:
        Var = core.Matrix("zero", na, nr)
        Var.zero()
        dfh.write_disk_tensor("WBar", Var, (nB + nb1, nB + nb1 + 1))

    # Every (a, r) amplitude for every ESP source B at once.  The ESP tensor
    # comes off disk in one read instead of one read per B, the orbital-energy
    # denominator is one elementwise pass, the backtransform by Uocc_A is one
    # GEMM over the whole B axis, and the two "zip up" dots become one
    # contraction each -- in place of nBt * (na * nr) scalar divisions and
    # nBt * 2 * na python-level dots.
    nBt = nB + nb1 + 1
    WBar_all = core.Matrix("WBar_all", nBt * na, nr)
    dfh.fill_tensor("WBar", WBar_all)
    # (B, a, r) -> (a, B, r), so a is the leading (GEMM-contracted) axis
    WaBr = np.ascontiguousarray(np.transpose(WBar_all.np, (1, 0, 2)))
    denomA = 1.0 / (_arr(eps_occ_A)[:, None] - _arr(eps_vir_A)[None, :])

    T_WaBr = _ein(WaBr, name="WaBr")
    T_denA = _ein(denomA, name="denomA")
    T_UoccA = _ein(Uocc_A, name="UoccA")
    T_wBT = _ein(wBT, name="wBT")
    T_uBT = _ein(uBT, name="uBT")
    T_JAB = _ein_zeros(na, nBt, name="JAB")
    T_KAB = _ein_zeros(na, nBt, name="KAB")

    with graph_block("find_uncAB") as blk:
        xA = blk.tensor(na, nBt, nr, name="xA")
        x2A = blk.tensor(na, nBt, nr, name="x2A")
        ein.einsum("aBr <- aBr ; ar", xA, T_WaBr, T_denA)
        ein.einsum("ij <- ki ; kj", x2A.reshape_view([na, nBt * nr]),
                   T_UoccA, xA.reshape_view([na, nBt * nr]))
        ein.einsum("aB <- aBr ; ar", T_JAB, x2A, T_wBT, ab_pf=2.0, c_pf=0.0)
        ein.einsum("aB <- aBr ; ar", T_KAB, x2A, T_uBT, ab_pf=2.0, c_pf=0.0)

    Jmat, Kmat = np.asarray(T_JAB), np.asarray(T_KAB)
    Ind20u_AB_termsp[:, :] = Jmat
    ExchInd20u_AB_termsp[:, :] = Kmat
    Indu_AB_terms.np[:, :] = Jmat + Kmat
    Ind20u_AB = float(Jmat.sum())
    ExchInd20u_AB = float(Kmat.sum())
    Indu_AB = Ind20u_AB + ExchInd20u_AB

    core.timer_off("FIND:uncAB")
    core.timer_on("FIND:uncBA")
    # ==> B <- A Uncoupled <==
    if dimer_wfn.has_potential_variable("A"):
        Vbs = core.triplet(Cocc_B, cache["VA_extern"], Cvir_B, True, False, False)
        dfh.write_disk_tensor("WAbs", Vbs, (nA + na1, nA + na1 + 1))
    else:
        Vbs = core.Matrix("zero", nb, ns)
        Vbs.zero()
        dfh.write_disk_tensor("WAbs", Vbs, (nA + na1, nA + na1 + 1))

    # Same batched form as A <- B, with the monomer labels swapped.  This is
    # the larger of the two at F-SAPT sizes (nAt * nb * ns amplitudes).
    nAt = nA + na1 + 1
    WAbs_all = core.Matrix("WAbs_all", nAt * nb, ns)
    dfh.fill_tensor("WAbs", WAbs_all)
    WbAs = np.ascontiguousarray(np.transpose(WAbs_all.np, (1, 0, 2)))
    denomB = 1.0 / (_arr(eps_occ_B)[:, None] - _arr(eps_vir_B)[None, :])

    T_WbAs = _ein(WbAs, name="WbAs")
    T_denB = _ein(denomB, name="denomB")
    T_UoccB = _ein(Uocc_B, name="UoccB")
    T_wAT = _ein(wAT, name="wAT")
    T_uAT = _ein(uAT, name="uAT")
    T_JBA = _ein_zeros(nb, nAt, name="JBA")
    T_KBA = _ein_zeros(nb, nAt, name="KBA")

    with graph_block("find_uncBA") as blk:
        xB = blk.tensor(nb, nAt, ns, name="xB")
        x2B = blk.tensor(nb, nAt, ns, name="x2B")
        ein.einsum("bAs <- bAs ; bs", xB, T_WbAs, T_denB)
        ein.einsum("ij <- ki ; kj", x2B.reshape_view([nb, nAt * ns]),
                   T_UoccB, xB.reshape_view([nb, nAt * ns]))
        ein.einsum("bA <- bAs ; bs", T_JBA, x2B, T_wAT, ab_pf=2.0, c_pf=0.0)
        ein.einsum("bA <- bAs ; bs", T_KBA, x2B, T_uAT, ab_pf=2.0, c_pf=0.0)

    Jmat, Kmat = np.asarray(T_JBA).T, np.asarray(T_KBA).T
    Ind20u_BA_termsp[:, :] = Jmat
    ExchInd20u_BA_termsp[:, :] = Kmat
    Indu_BA_terms.np[:, :] = Jmat + Kmat
    Ind20u_BA = float(Jmat.sum())
    ExchInd20u_BA = float(Kmat.sum())
    Indu_BA = Ind20u_BA + ExchInd20u_BA

    core.timer_off("FIND:uncBA")
    if do_print:
        core.print_out(
            f"    Ind20,u (A<-B)          = {Ind20u_AB * 1000:18.8f} [mEh]\n"
        )
        core.print_out(
            f"    Ind20,u (B<-A)          = {Ind20u_BA * 1000:18.8f} [mEh]\n"
        )
        assert (
            abs(scalars["Ind20,u (A<-B)"] - Ind20u_AB) < max(1e-6, 1e-4 * abs(scalars["Ind20,u (A<-B)"]))
        ), f"Ind20u_AB mismatch: {1000 * scalars['Ind20,u (A<-B)']:.8f} vs {1000 * Ind20u_AB:.8f}"
        assert (
            abs(scalars["Ind20,u (A->B)"] - Ind20u_BA) < max(1e-6, 1e-4 * abs(scalars["Ind20,u (A->B)"]))
        ), f"Ind20u_BA mismatch: {1000 * scalars['Ind20,u (A->B)']:.8f} vs {1000 * Ind20u_BA:.8f}"
        core.print_out(
            f"    Ind20,u                 = {Ind20u_AB + Ind20u_BA * 1000:18.8f} [mEh]\n"
        )
        core.print_out(
            f"    Exch-Ind20,u (A<-B)     = {ExchInd20u_AB * 1000:18.8f} [mEh]\n"
        )
        core.print_out(
            f"    Exch-Ind20,u (B<-A)     = {ExchInd20u_BA * 1000:18.8f} [mEh]\n"
        )
        assert (
            abs(scalars["Exch-Ind20,u (A<-B)"] - ExchInd20u_AB) < max(1e-6, 1e-4 * abs(scalars["Exch-Ind20,u (A<-B)"]))
        ), f"ExchInd20u_AB mismatch: {1000 * scalars['Exch-Ind20,u (A<-B)']:.8f} vs {1000 * ExchInd20u_AB:.8f}"
        assert (
            abs(scalars["Exch-Ind20,u (A->B)"] - ExchInd20u_BA) < max(1e-6, 1e-4 * abs(scalars["Exch-Ind20,u (A->B)"]))
        ), f"ExchInd20u_BA mismatch: {1000 * scalars['Exch-Ind20,u (A->B)']:.8f} vs {1000 * ExchInd20u_BA:.8f}"
        core.print_out(
            f"    Exch-Ind20,u            = {ExchInd20u_AB + ExchInd20u_BA * 1000:18.8f} [mEh]\n\n"
        )

    # The opt-in HF coupled fragment flag must solve each physical source.
    # Scaling an uncoupled partition by a canonical scalar ratio is not the
    # same calculation. Default (flag false) and non-HF paths are unchanged.
    exact_hf_response = (
        core.get_option("FISAPT", "FISAPT_FSAPT_IND_RESPONSE")
        and str(core.get_option("SAPT", "SAPT_DFT_FUNCTIONAL")).upper() == "HF"
    )
    if exact_hf_response:
        from .sapt_fisapt_cpu_response import partition_response
        if core.get_global_option("USE_CUEST"):
            raise ValueError("CPU coupled fragment continuation cannot substitute for CUDA")
        if (link_assignment != "C" or len(mol.get_fragments()) != 2 or dimer_wfn.nirrep() != 1
                or any(dimer_wfn.has_potential_variable(s) for s in ("A","B","C"))
                or core.get_option("FISAPT","SSAPT0_SCALE")
                or core.get_option("FISAPT","FISAPT_EXCH_PARPERP")):
            raise NotImplementedError("Exact CPU HF fragment reference requires two unlinked C1 C-assignment fragments, no external potentials/SSAPT0/parallel-perpendicular scaling")
        core.timer_on("FIND:coupled")
        try:
            coupled = {}
            for side,co,cv,eo,ev,sources,rotation,target,target_x in (
                    ("AB", Cocc_A, Cvir_A, eps_occ_A, eps_vir_A,
                     np.asarray(WBar_all.np).reshape(nBt,na,nr), Uocc_A, wBT, uBT),
                    ("BA", Cocc_B, Cvir_B, eps_occ_B, eps_vir_B,
                     np.asarray(WAbs_all.np).reshape(nAt,nb,ns), Uocc_B, wAT, uAT)):
                ind,exind,diagnostics = partition_response(
                    core,dimer_wfn.basisset(),aux_basis,_arr(co),_arr(cv),_arr(eo),_arr(ev),
                    sources,_arr(rotation),_arr(target),_arr(target_x),
                    memory_doubles=max(1,int(core.get_memory())//16))
                coupled[side] = ind,exind
                direction = "A<-B" if side=="AB" else "A->B"
                for prefix,values in (("Ind20,r",ind),("Exch-Ind20,r",exind)):
                    expected=float(scalars[f"{prefix} ({direction})"])
                    observed=float(np.sum(values))
                    if (not np.isfinite(expected) or not np.isfinite(observed)
                            or abs(observed-expected)>max(2e-7,1e-5*abs(expected))):
                        raise RuntimeError(
                            f"CPU coupled fragment {prefix} {direction} sum disagrees: "
                            f"{observed:.15g} vs canonical {expected:.15g} Eh")
                if do_print:
                    core.print_out(
                        f"    CPU coupled fragment {side}: {len(diagnostics)} sources, "
                        f"max iterations {max((d['iterations'] for d in diagnostics),default=0)}, "
                        f"max true residual {max((d['absolute_residual'] for d in diagnostics),default=0.):.3E}\n")
            def placed(values,side):
                output=core.Matrix("CPU fragment "+side,nA+na1+1,nB+nb1+1)
                output.zero()
                if side=="AB":output.np[nA:nA+na,:nB+nb1+1]=values
                else:output.np[:nA+na1+1,nB:nB+nb]=values.T
                return output
            for side,ind,exind in (("AB",Ind20u_AB_terms,ExchInd20u_AB_terms),
                                   ("BA",Ind20u_BA_terms,ExchInd20u_BA_terms)):
                cache["Ind20u_"+side]=placed(_arr(ind) if side=="AB" else _arr(ind).T,side)
                cache["ExchInd20u_"+side]=placed(_arr(exind) if side=="AB" else _arr(exind).T,side)
                cache["Ind20r_"+side]=placed(coupled[side][0],side)
                cache["ExchInd20r_"+side]=placed(coupled[side][1],side)
            Indu_AB_terms=core.Matrix.from_array(coupled["AB"][0]+coupled["AB"][1])
            # Existing BA local storage is [source A, occupied B].
            Indu_BA_terms=core.Matrix.from_array((coupled["BA"][0]+coupled["BA"][1]).T)
        finally:
            core.timer_off("FIND:coupled")

    # Induction scaling
    if ind_scale:
        dHF = scalars.get("Delta HF Correction", 0.0)
        IndHF = scalars["Ind20,r"] + scalars["Exch-Ind20,r"] + dHF
        IndSAPT0 = scalars["Ind20,r"] + scalars["Exch-Ind20,r"]

        Sdelta = IndHF / IndSAPT0
        if exact_hf_response and not np.isfinite(Sdelta):
            raise FloatingPointError("Nonfinite exact CPU fragment delta-HF scaling")

        # NOTE: if doing ind_resp, logic below needs adjusted
        SrAB = 1.0 if exact_hf_response else (scalars["Ind20,r (A<-B)"] + scalars["Exch-Ind20,r (A<-B)"]) / (
            scalars["Ind20,u (A<-B)"] + scalars["Exch-Ind20,u (A<-B)"]
        )
        SrBA = 1.0 if exact_hf_response else (scalars["Ind20,r (A->B)"] + scalars["Exch-Ind20,r (A->B)"]) / (
            scalars["Ind20,u (A->B)"] + scalars["Exch-Ind20,u (A->B)"]
        )

        if do_print:
            core.print_out(f"    Scaling for delta HF        = {Sdelta:11.3E}\n")
            core.print_out(f"    Scaling for response (A<-B) = {SrAB:11.3E}\n")
            core.print_out(f"    Scaling for response (A->B) = {SrBA:11.3E}\n")
            core.print_out(f"    Scaling for total (A<-B)    = {Sdelta * SrAB:11.3E}\n")
            core.print_out(f"    Scaling for total (A->B)    = {Sdelta * SrBA:11.3E}\n")
            core.print_out("\n")

        # Apply scaling to all terms
        Indu_AB_terms.scale(Sdelta * SrAB)
        Indu_BA_terms.scale(Sdelta * SrBA)
        if not exact_hf_response:
            Ind20u_AB_terms.scale(Sdelta * SrAB)
            ExchInd20u_AB_terms.scale(Sdelta * SrAB)
            Ind20u_BA_terms.scale(Sdelta * SrBA)
            ExchInd20u_BA_terms.scale(Sdelta * SrBA)

        # Apply SSAPT0 scaling if enabled
        # if "sExch-Ind20,r" in scalars:
        #     sIndu_AB_terms.scale(sSdelta * sSrAB)
        #     sIndu_BA_terms.scale(sSdelta * sSrBA)

    IndAB_AB = core.Matrix("IndAB_AB", nA + na1 + 1, nB + nb1 + 1)
    IndBA_AB = core.Matrix("IndBA_AB", nA + na1 + 1, nB + nb1 + 1)

    # Assemble from the total induction matrices, matching FISAPT::find().
    # In particular, use the Matrix objects after any induction scaling has
    # been applied.
    Indu_AB_termsp = Indu_AB_terms.np
    Indu_BA_termsp = Indu_BA_terms.np
    for a in range(na):
        for B in range(nB + nb1 + 1):
            IndAB_AB.np[a + nA, B] = Indu_AB_termsp[a, B]
    for A in range(nA + na1 + 1):
        for b in range(nb):
            IndBA_AB.np[A, b + nB] = Indu_BA_termsp[A, b]

    cache["IndAB_AB"] = IndAB_AB
    cache["IndBA_AB"] = IndBA_AB

    # if core.get_option("SAPT", "SSAPT0_SCALE"):
    #     cache["sExchInd20u_AB"] = sExchInd20u_AB
    #     cache["sExchInd20u_BA"] = sExchInd20u_BA
    #     cache["sIndu_AB"] = sIndu_AB
    #     cache["sIndu_BA"] = sIndu_BA

    dfh.clear_all()
    return cache


def _store_dispersion0_components(cache, disp, exch, nA, nB, nfa, nfb, na1, nb1):
    """Common original matrix placement/sum rules for canonical and fragment paths."""
    na, nb = disp.shape
    E_disp20 = core.Matrix("E_disp20", nA + nfa + na1 + 1, nB + nfb + nb1 + 1)
    E_exch_disp20 = core.Matrix("E_exch_disp20", nA + nfa + na1 + 1, nB + nfb + nb1 + 1)
    ablock = slice(nfa + nA, nfa + nA + na)
    bblock = slice(nfb + nB, nfb + nB + nb)
    E_disp20.np[ablock, bblock] = disp
    E_exch_disp20.np[ablock, bblock] = exch
    Disp_AB = core.Matrix("Disp_AB", nA + nfa + na1 + 1, nB + nfb + nb1 + 1)
    Disp_AB.np[:, :] = E_disp20.np + E_exch_disp20.np
    cache["Disp_AB"] = Disp_AB
    cache["Disp20,u"] = np.sum(E_disp20.np)
    cache["Exch-Disp20,u"] = np.sum(E_exch_disp20.np)


def dispersion0(cache, dimer_wfn, nfrozen_A=0, nfrozen_B=0, *, block_size=None, gpu=False,
                do_print=True, _graph_device=None, _df_block_size=None, _kernel_callback=None):
    """Canonical SAPT0 dispersion using the blocked Python Einsums equations.

    No localization or fragment cache is required. The caller's cache is not
    mutated. Both signed components are returned, in Hartree. ``gpu=True``
    offloads eligible block contractions only; elementwise work/reductions
    remain on the host. The native SAPT/FISAPT algorithms remain CPU oracles.
    """
    work = dict(cache)
    # A zero-dimensional call never executes a producer/graph. Do not inherit
    # an earlier call's device/timing provenance through the shallow cache copy.
    work.pop("Disp0 execution", None)
    fdisp0(work, {}, dimer_wfn, None, None, None, do_print,
           _canonical=True, _nfrozen=(nfrozen_A, nfrozen_B),
           _block_size=block_size, _gpu=gpu, _graph_device=_graph_device,
           _df_block_size=_df_block_size, _kernel_callback=_kernel_callback)
    result = {key: work[key] for key in ("Disp20,u", "Exch-Disp20,u")}
    if "Disp0 execution" in work:
        result["Disp0 execution"] = work["Disp0 execution"]
    return result


def dispersion0_cuda(cache, dimer_wfn, *, provider_factory, nfrozen_A=0, nfrozen_B=0,
                     block_size=24, do_print=True, _localized=None, _return_pairs=False):
    """Experimental explicit Python entrypoint; not an automatic driver option.

    provider_factory(primary_basis, auxiliary_basis) must be a context manager
    yielding the audited cuEST DFMO adapter. Raw upstream HF matrices are
    uploaded once; dispersion setup, DFMO packing, contractions and signed
    reductions execute on GPU. No E2 CPU setup or host factor path is used.
    """
    if (str(core.get_option("SAPT", "SAPT_DFT_FUNCTIONAL")).upper() != "HF"
            or not core.get_option("SAPT", "SAPT_DFT_DO_DISP")
            or core.get_option("SAPT", "SAPT_DFT_D3_IE")
            or core.get_option("SAPT", "SAPT_DFT_D4_IE")):
        raise ValueError("Experimental CUDA dispersion requires ordinary HF SAPT0, not DFT/D3/D4")
    fragment_mode=str(core.get_option("SAPT", "SAPT_DFT_DO_FSAPT")).upper()
    if ((fragment_mode!="NONE" if _localized is None else fragment_mode not in ("SAPTDFT","FISAPT"))
            or core.get_option("SAPT", "DO_DISP_EXCH_SINF")):
        raise NotImplementedError("CUDA fragment/F-SAPT and S-inf dispersion are unsupported")
    if core.get_option("SAPT", "SAPT_DFT_EINSUMS_GPU"):
        raise ValueError("Explicit CUDA and E2 GPU dispersion selections conflict")
    if not callable(provider_factory):
        raise TypeError("An explicit audited cuEST plan/provider factory is required")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("CUDA compute block must be a positive integer")
    if dimer_wfn.nirrep() != 1:
        raise NotImplementedError("Experimental CUDA dispersion currently requires C1 symmetry")
    from .sapt_cuest_basis import require_spherical_basis
    require_spherical_basis(dimer_wfn.basisset(),role="primary")

    if (type(nfrozen_A) is not int or type(nfrozen_B) is not int
            or min(nfrozen_A,nfrozen_B) < 0):
        raise ValueError("Frozen occupied counts must be nonnegative integers")
    counts = {name:cache[key].np.shape[1] for name,key in
              (("a","Cocc_A"),("b","Cocc_B"),("r","Cvir_A"),("s","Cvir_B"))}
    if nfrozen_A > counts["a"] or nfrozen_B > counts["b"]:
        raise ValueError("Frozen occupied count exceeds upstream HF space")
    if (nfrozen_A == counts["a"] or nfrozen_B == counts["b"]
            or counts["r"] == 0 or counts["s"] == 0):
        return {"Disp20,u":0.0,"Exch-Disp20,u":0.0}
    if _localized is not None:
        if not isinstance(_localized,(tuple,list)) or len(_localized)!=2:
            raise ValueError("Two active occupied localization rotations required")
        for value,size in zip(_localized,(counts["a"]-nfrozen_A,counts["b"]-nfrozen_B)):
            if (not isinstance(value,np.ndarray) or value.dtype!=np.float64
                    or value.shape!=(size,size) or not np.isfinite(value).all()):
                raise ValueError("Invalid raw active occupied localization rotation")

    from .sapt_dispersion_cuda import blocked_device_dispersion
    from .sapt_dispersion_cuda_setup import prepare_hf
    from .sapt_cuest_dfmo_cuda import CuESTDFMOProvider
    started = time.perf_counter()
    primary = dimer_wfn.basisset()
    auxiliary = core.BasisSet.build(
        dimer_wfn.molecule(),"DF_BASIS_MP2",core.get_option("DFMP2","DF_BASIS_MP2"),
        "RIFIT",core.get_global_option("BASIS"))
    require_spherical_basis(auxiliary,role="auxiliary")
    names = ("S","D_A","P_A","V_A","J_A","K_A","D_B","P_B","V_B","J_B","K_B","K_O",
             "Cocc_A","Cocc_B","Cvir_A","Cvir_B")
    matrices = {name:np.asarray(cache[name].np) for name in names}
    energies = {name:np.asarray(cache[key].np) for name,key in
                (("a","eps_occ_A"),("b","eps_occ_B"),("r","eps_vir_A"),("s","eps_vir_B"))}
    with provider_factory(primary,auxiliary) as provider:
        if not isinstance(provider,CuESTDFMOProvider):
            raise TypeError("The CUDA entrypoint requires strict cuEST0.2.50 native admission")
        # Provider parameters are owned/closed before the factory may destroy
        # its borrowed plan; construction must precede any temporary scope.
        with provider:
            provider.validate_basis(primary, auxiliary)
            provider.prepare_device_compute()
            with provider.session.scope():
                spaces,corrections,e = prepare_hf(
                    provider.session,matrices,energies,frozen_a=nfrozen_A,frozen_b=nfrozen_B)
                extra={}
                if _localized is not None:
                    extra["rotations"]=tuple(provider.session.upload(value) for value in _localized)
                result = blocked_device_dispersion(
                    provider,spaces=spaces,corrections=corrections,energies=e,block_size=block_size,**extra)
    execution = dict(result["Disp0 execution"])
    execution["entrypoint_seconds_including_factory_setup_cleanup"] = time.perf_counter()-started
    execution["raw_hf_numerical_setup"] = "owned cuBLAS/custom CUDA"
    execution["scope"] = ("Spherical HF dispersion from upstream matrices, including raw upload, "
                          "GPU setup/device DFMO/packing/contractions/pair/global reductions "
                          "and owned cleanup; excludes upstream SCF/elst/exch/induction")
    if do_print:
        core.print_out(f'    CUDA Disp20,u = {result["Disp20,u"]:.12e} Eh\n'
                       f'    CUDA Exch-Disp20,u = {result["Exch-Disp20,u"]:.12e} Eh\n')
    output={"Disp20,u":result["Disp20,u"],"Exch-Disp20,u":result["Exch-Disp20,u"],
            "Disp0 execution":execution}
    if _return_pairs:
        output.update({name:result[name] for name in (
            "Disp20 occupied pairs","Exch-Disp20 occupied pairs","Disp0 occupied pairs","Disp0,u")})
    return output


def fisapt_cuda(cache,dimer_wfn,wfn_A,wfn_B,*,provider_factory,block_size=24,scalars=None):
    """Bounded automatic GPU fragment continuation; never CPU localization/factors."""
    from .sapt_cuest_basis import require_spherical_basis
    from .sapt_fisapt_workflow_cuda import fragment_terms, HF_FIELDS
    mol=dimer_wfn.molecule()
    # Molecule::get_fragments returns [first,last_exclusive) pairs, NOT atom lists.
    fragments=[list(range(first,last)) for first,last in mol.get_fragments()]
    if (dimer_wfn.nirrep()!=1 or len(fragments)!=2
            or sorted(i for f in fragments for i in f)!=list(range(mol.natom()))):
        raise NotImplementedError("GPU FISAPT requires exactly two unlinked C1 fragments")
    primary=dimer_wfn.basisset()
    require_spherical_basis(primary,role="primary")
    jkfit=core.BasisSet.build(mol,"DF_BASIS_SCF",core.get_option("SCF","DF_BASIS_SCF"),
                            "JKFIT",core.get_global_option("BASIS"),puream=1)
    rifit=core.BasisSet.build(mol,"DF_BASIS_MP2",core.get_option("DFMP2","DF_BASIS_MP2"),
                            "RIFIT",core.get_global_option("BASIS"),puream=1)
    minimal=core.BasisSet.build(mol,"BASIS",core.get_global_option("MINAO_BASIS"),puream=1)
    for value in (jkfit,rifit,minimal):require_spherical_basis(value,role="auxiliary")
    matrices={n:np.asarray(cache[n].np) for n in (*HF_FIELDS,"J_O")}
    energies={n:np.asarray(cache[key].np) for n,key in zip("abrs",
              ("eps_occ_A","eps_occ_B","eps_vir_A","eps_vir_B"))}
    frozen=tuple(w.basisset().n_frozen_core(core.get_global_option("FREEZE_CORE"),w.molecule())
                 for w in (wfn_A,wfn_B))
    xyz=np.array([[mol.x(i),mol.y(i),mol.z(i)] for i in range(mol.natom())],dtype=np.float64)
    charges=[np.zeros(mol.natom(),dtype=np.float64) for _ in fragments]
    for values,fragment in zip(charges,fragments):
        for i in fragment:values[i]=mol.Z(i)
    scaling={}
    if core.get_option("FISAPT","FISAPT_FSAPT_EXCH_SCALE"):
        if scalars is None or "Exch10" not in scalars:
            raise ValueError("GPU fragment exchange scaling requires canonical Exch10")
        scaling["exchange"]=np.array([scalars["Exch10"]],dtype=np.float64)
    if core.get_option("FISAPT","FISAPT_FSAPT_IND_SCALE"):
        if scalars is None or any(n not in scalars for n in ("Ind20,r","Exch-Ind20,r")):
            raise ValueError("GPU fragment induction scaling requires canonical coupled HF scalars")
        scaling["induction"]=np.array([scalars["Ind20,r"],scalars["Exch-Ind20,r"],
                                       scalars.get("Delta HF Correction",0.)],dtype=np.float64)
    started=time.perf_counter()
    with provider_factory(primary,jkfit) as provider:
        with provider:
            provider.validate_basis(primary,jkfit)
            arrays,data=fragment_terms(provider,rifit,minimal,matrices,energies,xyz,*charges,
                                       frozen=frozen,block_size=block_size,scaling=scaling)
    work=dict(cache)
    work.update({name:core.Matrix.from_array(value) for name,value in arrays.items()})
    for side,values in zip(("A","B"),charges):
        work["Z"+side]=core.Vector.from_array(values)
        work["Z"+side].name="Z"+side
    data["FISAPT execution"]["entrypoint_seconds_including_factory_cleanup"]=time.perf_counter()-started
    return work,data


def fisapt_cuda_drop(cache,dimer_wfn):
    """Serialize final GPU results with the existing writer, no native FISAPT.

    The writer only needs molecule/vectors/matrices accessors. A data-only
    adapter avoids constructing a CPU FISAPT/DFHelper object just for output.
    Scalar/wavefunction variables are already set by the public SAPT driver.
    """
    from types import SimpleNamespace
    from ... import p4util
    from . import fisapt_proc
    adapter=SimpleNamespace(molecule=dimer_wfn.molecule,
        vectors=lambda:{name:cache[name] for name in ("ZA","ZB")},
        matrices=lambda:cache)
    stash=p4util.OptionsState(["FISAPT","FISAPT_DO_FSAPT_DISP"])
    try:
        core.set_local_option("FISAPT","FISAPT_DO_FSAPT_DISP",core.get_option("SAPT","SAPT_DFT_DO_DISP"))
        fisapt_proc.fisapt_fdrop(adapter)
    finally:
        stash.restore()


def fdisp0_cuda(cache,dimer_wfn,*,provider_factory,block_size=24,do_print=True):
    """Explicit unscaled two-fragment HF LO dispersion; return a new cache.

    Localization is an upstream input, NOT certified GPU work by this call.
    C-link assignment with exactly two fragments/no external potentials only.
    Atom/frozen/external slots are zero; the active pair sum is formed on GPU.
    """
    if (str(core.get_option("FISAPT","FISAPT_LINK_ASSIGNMENT")).upper()!="C"
            or core.get_option("FISAPT","SSAPT0_SCALE")
            or core.get_option("FISAPT","FISAPT_EXCH_PARPERP")
            or cache.get("external_potentials")):
        raise NotImplementedError("CUDA fragment dispersion link/scaling/external cases are not implemented")
    mol=dimer_wfn.molecule()
    if len(mol.get_fragments())!=2:
        raise NotImplementedError("CUDA fragment dispersion currently requires two fragments")
    needed=("Lfocc0A","Lfocc0B","Caocc0A","Caocc0B","Uaocc0A","Uaocc0B")
    if any(name not in cache for name in needed):
        raise ValueError("Complete active/frozen/localized occupied cache required")
    nf=tuple(cache["Lfocc0"+side].np.shape[1] for side in ("A","B"))
    for side,frozen in zip(("A","B"),nf):
        full=np.asarray(cache["Cocc_"+side].np)
        if (frozen>full.shape[1] or cache["Lfocc0"+side].np.shape[0]!=full.shape[0]
                or not np.array_equal(cache["Caocc0"+side].np,full[:,frozen:])):
            raise ValueError("Fragment canonical active/frozen cache differs from raw HF orbitals")
    work=dict(cache);work.pop("Disp0 execution",None)
    occupied=tuple(cache["Cocc_"+side].np.shape[1] for side in ("A","B"))
    matrix=core.Matrix("Disp_AB",mol.natom()+occupied[0]+1,mol.natom()+occupied[1]+1)
    result=dispersion0_cuda(cache,dimer_wfn,provider_factory=provider_factory,
        nfrozen_A=nf[0],nfrozen_B=nf[1],block_size=block_size,do_print=do_print,
        _localized=tuple(np.asarray(cache["Uaocc0"+side].np) for side in ("A","B")),_return_pairs=True)
    if "Disp0 occupied pairs" in result:
        pairs=result["Disp0 occupied pairs"]
        matrix.np[mol.natom()+nf[0]:mol.natom()+nf[0]+pairs.shape[0],
                  mol.natom()+nf[1]:mol.natom()+nf[1]+pairs.shape[1]]=pairs
    work.update({"Disp_AB":matrix,"Disp20,u":result["Disp20,u"],"Exch-Disp20,u":result["Exch-Disp20,u"],
                 "sDisp_AB":core.Matrix("sDisp_AB",1,1)})
    if "Disp0 execution" in result:work["Disp0 execution"]=result["Disp0 execution"]
    return work


def fdisp0(
    cache: dict,
    scalars: dict,
    dimer_wfn: core.Wavefunction,
    wfn_A: core.Wavefunction,
    wfn_B: core.Wavefunction,
    jk: core.JK,
    do_print: bool = True,
    *,
    _canonical=False,
    _nfrozen=(0, 0),
    _block_size=None,
    _gpu=False,
    _graph_device=None,
    _df_block_size=None,
    _kernel_callback=None,
) -> dict:
    r"""Compute the F-SAPT0 dispersion partitioning.

    Partitions the second-order dispersion and exchange-dispersion
    energies into atomic pair contributions for F-SAPT analysis. Note,
    this does not use DFT energies and is a Hartree-Fock (SAPT0)
    dispersion energy only. Ideally, this function should be optimized for
    python (considerably slower than C++).

    Parameters
    ----------
    cache : dict
        SAPT data cache containing orbital coefficients, density matrices, and integrals.
    scalars : dict
        Reference scalar energies for validation.
    dimer_wfn : core.Wavefunction
        Dimer wavefunction.
    wfn_A : core.Wavefunction
        Monomer A wavefunction.
    wfn_B : core.Wavefunction
        Monomer B wavefunction.
    jk : core.JK
        JK integral engine.
    do_print : bool, optional
        Whether to print results, by default True.

    Returns
    -------
    dict
        Updated cache with ``Disp_AB`` matrix and ``Disp20,u``/``Exch-Disp20,u`` scalar energies.
    """
    started = time.perf_counter()
    block_edge = FDISP_BLOCK if _block_size is None else int(_block_size)
    if block_edge < 1:
        raise ValueError("Dispersion compute block edge must be positive")
    if _df_block_size is not None and _df_block_size < 1:
        raise ValueError("Dispersion DF test block cap must be positive")
    if _gpu and not _canonical:
        raise NotImplementedError("GPU fragment dispersion is not yet supported; use the explicit CPU fragment route")
    if _kernel_callback is not None and (
            not _canonical or _gpu or _graph_device is not None or _df_block_size is not None):
        raise ValueError("Canonical kernel callback conflicts with fragment/E2-GPU/DF-block selections")
    use_graph = _gpu or _graph_device is not None
    if use_graph:
        from .sapt_dispersion_gpu import BlockContractions
        contractions = BlockContractions(device="cuda" if _gpu else _graph_device)
    else:
        contractions = None
    if do_print:
        core.print_out("  ==> Python SAPT0 Dispersion <==\n\n" if _canonical
                       else "  ==> F-SAPT0 Dispersion <==\n\n")

    core.timer_on("F-SAPT Disp Setup")
    # ind_scale = core.get_option("FISAPT", "FISAPT_FSAPT_IND_SCALE")
    link_assignment = core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT")

    mol = dimer_wfn.molecule()
    dimer_basis = dimer_wfn.basisset()
    nA = mol.natom()
    nB = mol.natom()
    nfa, nfb = _nfrozen if _canonical else (cache["Lfocc0A"].shape[1], cache["Lfocc0B"].shape[1])
    if _canonical:
        if not (0 <= nfa <= cache["Cocc_A"].shape[1] and 0 <= nfb <= cache["Cocc_B"].shape[1]):
            raise ValueError("Invalid frozen occupied dimensions for dispersion")
        Cocc_A = core.Matrix.from_array(cache["Cocc_A"].np[:, nfa:])
        Cocc_B = core.Matrix.from_array(cache["Cocc_B"].np[:, nfb:])
    else:
        Cocc_A, Cocc_B = cache["Caocc0A"], cache["Caocc0B"]
    # Use active occupied dimensions (excluding frozen core) to match C++ FISAPT fdisp
    na = Cocc_A.shape[1]
    nb = Cocc_B.shape[1]
    nr = cache["Cvir_A"].shape[1]
    ns = cache["Cvir_B"].shape[1]
    nn = cache["Cocc_A"].shape[0]  # number of AO basis functions

    na1 = na
    nb1 = nb
    snA = 0
    snfa = 0
    sna = 0
    snB = 0
    snfb = 0
    snb = 0
    # if options_.get_bool("FISAPT", "FISAPT_SSAPT0_SCALE"):
    #     snA = nA
    #     snfa = nfa
    #     sna = na
    #     snB = nB
    #     snfb = nfb
    #     snb = nb

    if not _canonical and link_assignment in ["SAO0", "SAO1", "SAO2", "SIAO0", "SIAO1", "SIAO2"]:
        na1 = na + 1
        nb1 = nb + 1

    if min(na, nb, nr, ns) == 0:
        cache["Disp_AB"] = core.Matrix("Disp_AB", nA + nfa + na1 + 1, nB + nfb + nb1 + 1)
        cache["Disp20,u"] = cache["Exch-Disp20,u"] = 0.0
        core.timer_off("F-SAPT Disp Setup")
        return cache
    Cvir_A = cache["Cvir_A"]
    Cvir_B = cache["Cvir_B"]

    # Use only active occupied orbital energies (skip frozen core)
    # nfa and nfb are already defined at the start of fdisp0
    eps_occ_A = core.Vector.from_array(cache["eps_occ_A"].np[nfa:])
    eps_occ_B = core.Vector.from_array(cache["eps_occ_B"].np[nfb:])
    eps_vir_A = cache["eps_vir_A"]
    eps_vir_B = cache["eps_vir_B"]

    # Collect relevant variables
    S = cache["S"]
    D_A = cache["D_A"]
    P_A = cache["P_A"]
    V_A = cache["V_A"]
    J_A = cache["J_A"]
    K_A = cache["K_A"]
    D_B = cache["D_B"]
    P_B = cache["P_B"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]
    K_B = cache["K_B"]
    K_O = cache["K_O"]

    # Dispersion fits an MP2-like (ar|bs) amplitude, so it takes the RI
    # auxiliary basis that FISAPT::fdisp takes, not the JKFIT DF_BASIS_SCF set
    # that the electrostatics, exchange and induction terms above share.  Built
    # exactly as the FISAPT branch of the SAPT(DFT) driver builds it, so the two
    # F-SAPT(DFT) dispersion routes fit in the same basis and their energies are
    # comparable.  Aside from being the right fitting set, RIFIT is roughly a
    # third smaller than JKFIT, and that factor lands on every DF transformation
    # below as well as on the (r,s) blocking -- at 1555 primary functions it is
    # the difference between reading the DF tensors once and reading them four
    # times.
    #
    # Leave puream alone.  BasisSet.build defaults to -1, "use the fitting set's
    # own angular momentum convention", which is what the driver's DF_BASIS_SAPT
    # build does and therefore what FISAPT::fdisp fits in.  Forcing the RI set to
    # follow the orbital basis instead silently rebuilds it in the orbital
    # basis's convention: with a Cartesian orbital basis such as 6-31G** that
    # turns the spherical cc-pvdz-ri into a Cartesian one, 2800 functions into
    # 3234, which both perturbs the fitted dispersion and puts 15% more work
    # through every DF transformation and block GEMM than the C++ route does.
    aux_basis = core.BasisSet.build(
        mol,
        "DF_BASIS_MP2",
        core.get_option("DFMP2", "DF_BASIS_MP2"),
        "RIFIT",
        core.get_global_option("BASIS"),
    )
    nQ = aux_basis.nbf()

    # => Auxiliary C and V matrices <= #
    #
    # One graph for the whole setup block.  These two dozen chains are written
    # exactly as the equations read, with nothing hoisted: D_B S, D_A S,
    # D_A S D_B S, Cocc_A^T S D_B and their partners recur all through them
    # (Cr1 and Cr3 begin with the same three-factor product, and so do Cs1 and
    # Cs3), and spotting that is the graph's job.  Only the values DFHelper
    # and the r,s loop read afterwards are allocated outside the block.
    e_S = _ein(S, "S")
    e_DA = _ein(D_A, "D_A")
    e_DB = _ein(D_B, "D_B")
    e_PA = _ein(P_A, "P_A")
    e_PB = _ein(P_B, "P_B")
    e_VA = _ein(V_A, "V_A")
    e_VB = _ein(V_B, "V_B")
    e_JA = _ein(J_A, "J_A")
    e_JB = _ein(J_B, "J_B")
    e_KA = _ein(K_A, "K_A")
    e_KB = _ein(K_B, "K_B")
    e_KO = _ein(K_O, "K_O")
    e_CoA = _ein(Cocc_A, "Cocc_A")
    e_CoB = _ein(Cocc_B, "Cocc_B")
    e_CvA = _ein(Cvir_A, "Cvir_A")
    e_CvB = _ein(Cvir_B, "Cvir_B")

    nso = e_S.shape[0]
    nao, nbo = e_CoA.shape[1], e_CoB.shape[1]
    nrv, nsv = e_CvA.shape[1], e_CvB.shape[1]

    Cr1 = _ein_zeros(nso, nrv, name="Cr1")
    Cs1 = _ein_zeros(nso, nsv, name="Cs1")
    Ca2 = _ein_zeros(nso, nao, name="Ca2")
    Cb2 = _ein_zeros(nso, nbo, name="Cb2")
    Cr3 = _ein_zeros(nso, nrv, name="Cr3")
    Cs3 = _ein_zeros(nso, nsv, name="Cs3")
    Ca4 = _ein_zeros(nso, nao, name="Ca4")
    Cb4 = _ein_zeros(nso, nbo, name="Cb4")
    Qar = _ein_zeros(nao, nrv, name="Qar")
    Qbs = _ein_zeros(nbo, nsv, name="Qbs")
    Qas = _ein_zeros(nao, nsv, name="Qas")
    Qbr = _ein_zeros(nbo, nrv, name="Qbr")
    Sas = _ein_zeros(nao, nsv, name="Sas")
    Sbr = _ein_zeros(nbo, nrv, name="Sbr")
    SBar = _ein_zeros(nao, nrv, name="SBar")
    SAbs = _ein_zeros(nbo, nsv, name="SAbs")

    with graph_block("fdisp0_setup"):
        # Cr1 = (D_B S - I) Cvir_A ; Cs1 = (D_A S - I) Cvir_B
        chain_sum(Cr1, [(1.0, [e_DB, e_S, e_CvA], "NNN"),
                        (-1.0, [e_CvA], "N")], name="Cr1")
        chain_sum(Cs1, [(1.0, [e_DA, e_S, e_CvB], "NNN"),
                        (-1.0, [e_CvB], "N")], name="Cs1")

        # Ca2 = D_B S Cocc_A ; Cb2 = D_A S Cocc_B
        chain_into(Ca2, [e_DB, e_S, e_CoA], "NNN", beta=0.0, name="Ca2")
        chain_into(Cb2, [e_DA, e_S, e_CoB], "NNN", beta=0.0, name="Cb2")

        # Cr3 = 2 (D_B S - D_A S D_B S) Cvir_A
        chain_sum(Cr3, [(2.0, [e_DB, e_S, e_CvA], "NNN"),
                        (-2.0, [e_DA, e_S, e_DB, e_S, e_CvA], "NNNNN")],
                  name="Cr3")
        # Cs3 = 2 (D_A S - D_B S D_A S) Cvir_B
        chain_sum(Cs3, [(2.0, [e_DA, e_S, e_CvB], "NNN"),
                        (-2.0, [e_DB, e_S, e_DA, e_S, e_CvB], "NNNNN")],
                  name="Cs3")

        # Ca4 = -2 D_A S D_B S Cocc_A ; Cb4 = -2 D_B S D_A S Cocc_B
        chain_into(Ca4, [e_DA, e_S, e_DB, e_S, e_CoA], "NNNNN",
                   coef=-2.0, beta=0.0, name="Ca4")
        chain_into(Cb4, [e_DB, e_S, e_DA, e_S, e_CoB], "NNNNN",
                   coef=-2.0, beta=0.0, name="Cb4")

        # Get your signs right Hesselmann!
        # Qar = 4 Cocc_A^T J_B Cvir_A + 2 Cocc_A^T V_B Cvir_A
        chain_sum(Qar, [(4.0, [e_CoA, e_JB, e_CvA], "TNN"),
                        (2.0, [e_CoA, e_VB, e_CvA], "TNN")], name="Qar")
        # Qbs = 4 Cocc_B^T J_A Cvir_B + 2 Cocc_B^T V_A Cvir_B
        chain_sum(Qbs, [(4.0, [e_CoB, e_JA, e_CvB], "TNN"),
                        (2.0, [e_CoB, e_VA, e_CvB], "TNN")], name="Qbs")

        # Qas = Jas + Kas + KOas + JAas + JBas + VBas + VRas
        chain_sum(Qas, [(2.0, [e_CoA, e_JB, e_CvB], "TNN"),
                        (-1.0, [e_CoA, e_KB, e_CvB], "TNN"),
                        (1.0, [e_CoA, e_KO, e_CvB], "TNN"),
                        (-2.0, [e_CoA, e_JB, e_DA, e_S, e_CvB], "TNNNN"),
                        (-2.0, [e_CoA, e_S, e_DB, e_JA, e_CvB], "TNNNN"),
                        (-1.0, [e_CoA, e_S, e_DB, e_VA, e_CvB], "TNNNN"),
                        (1.0, [e_CoA, e_VB, e_PA, e_S, e_CvB], "TNNNN")],
                  name="Qas")

        # Qbr = Jbr + Kbr + KObr + JAbr + JBbr + VAbr + VSbr
        # K_O is not symmetric, hence the transpose in the KObr term.
        chain_sum(Qbr, [(2.0, [e_CoB, e_JA, e_CvA], "TNN"),
                        (-1.0, [e_CoB, e_KA, e_CvA], "TNN"),
                        (1.0, [e_CoB, e_KO, e_CvA], "TTN"),
                        (-2.0, [e_CoB, e_S, e_DA, e_JB, e_CvA], "TNNNN"),
                        (-2.0, [e_CoB, e_JA, e_DB, e_S, e_CvA], "TNNNN"),
                        (-1.0, [e_CoB, e_S, e_DA, e_VB, e_CvA], "TNNNN"),
                        (1.0, [e_CoB, e_VA, e_PB, e_S, e_CvA], "TNNNN")],
                  name="Qbr")

        # Sas = Cocc_A^T S Cvir_B ; Sbr = Cocc_B^T S Cvir_A
        chain_into(Sas, [e_CoA, e_S, e_CvB], "TNN", beta=0.0, name="Sas")
        chain_into(Sbr, [e_CoB, e_S, e_CvA], "TNN", beta=0.0, name="Sbr")

        # SBar = Cocc_A^T S D_B S Cvir_A ; SAbs = Cocc_B^T S D_A S Cvir_B
        chain_into(SBar, [e_CoA, e_S, e_DB, e_S, e_CvA], "TNNNN",
                   beta=0.0, name="SBar")
        chain_into(SAbs, [e_CoB, e_S, e_DA, e_S, e_CvB], "TNNNN",
                   beta=0.0, name="SAbs")

    # An explicitly selected experimental producer/kernel shares the canonical
    # setup and final placement, but never constructs or stages host DFHelper.
    if _kernel_callback is not None:
        spaces = dict(a=np.asarray(e_CoA), r=np.asarray(e_CvA),
                      b=np.asarray(e_CoB), s=np.asarray(e_CvB),
                      r1=np.asarray(Cr1), s1=np.asarray(Cs1), a2=np.asarray(Ca2),
                      b2=np.asarray(Cb2), r3=np.asarray(Cr3), s3=np.asarray(Cs3),
                      a4=np.asarray(Ca4), b4=np.asarray(Cb4))
        corrections = {name: np.asarray(value) for name, value in (
            ("Qar", Qar), ("SBar", SBar), ("Qbs", Qbs), ("SAbs", SAbs),
            ("Qas", Qas), ("Sas", Sas), ("Qbr", Qbr), ("Sbr", Sbr))}
        energies = dict(a=eps_occ_A.np, b=eps_occ_B.np,
                        r=eps_vir_A.np, s=eps_vir_B.np)
        core.timer_off("F-SAPT Disp Setup")
        core.timer_on("F-SAPT Disp Compute")
        try:
            disp, exch, execution = _kernel_callback(
                primary_basis=dimer_basis, auxiliary_basis=aux_basis,
                spaces=spaces, corrections=corrections, energies=energies,
                block_size=block_edge)
            for component in (disp, exch):
                if (not isinstance(component, np.ndarray) or component.dtype != np.float64
                        or component.shape != (na, nb) or not np.isfinite(component).all()):
                    raise ValueError("Canonical kernel must return finite host FP64 occupied-pair components")
            _store_dispersion0_components(cache, disp, exch, nA, nB, nfa, nfb, na1, nb1)
            cache["Disp0 execution"] = dict(execution)
            cache["Disp0 execution"]["python_dispersion_call_seconds"] = time.perf_counter() - started
        finally:
            core.timer_off("F-SAPT Disp Compute")
        return cache

    # => Integrals from DFHelper <= #

    # Build list of orbital space matrices for DF transformations
    # Order: Cocc_A, Cvir_A, Cocc_B, Cvir_B, Cr1, Cs1, Ca2, Cb2, Cr3, Cs3, Ca4, Cb4
    # The Cr1..Cb4 spaces come out of the setup graph as einsums tensors, so
    # hand them to DFHelper through _mat().
    orbital_spaces = [
        _mat(Cocc_A),  # 0: 'a'
        _mat(Cvir_A),  # 1: 'r'
        _mat(Cocc_B),  # 2: 'b'
        _mat(Cvir_B),  # 3: 's'
        _mat(Cr1),  # 4: 'r1'
        _mat(Cs1),  # 5: 's1'
        _mat(Ca2),  # 6: 'a2'
        _mat(Cb2),  # 7: 'b2'
        _mat(Cr3),  # 8: 'r3'
        _mat(Cs3),  # 9: 's3'
        _mat(Ca4),  # 10: 'a4'
        _mat(Cb4),  # 11: 'b4'
    ]

    # Calculate total columns for memory allocation
    ncol = sum(mat.shape[1] for mat in orbital_spaces)
    # All should have same number of rows (AO basis)
    nrows = orbital_spaces[0].shape[0]

    # Initialize DFHelper on the RI basis selected above
    dfh = core.DFHelper(dimer_basis, aux_basis)

    # Set memory: total available minus space needed for orbital matrices
    # Note: In C++, doubles_ is the total memory budget in doubles
    # Here we use a reasonable default or get from options if available
    memory_doubles = core.get_memory() // 8
    orbital_memory = nrows * ncol
    dfh.set_memory(memory_doubles - orbital_memory)
    # print set memory in GB
    core.print_out(
        f"    Setting DFHelper memory to {(memory_doubles - orbital_memory) * 8 / 1e9:.3f} GB\n"
    )

    dfh.set_method("DIRECT_iaQ")
    dfh.set_nthreads(core.get_num_threads())
    dfh.initialize()
    dfh.print_header()

    # Add orbital spaces
    dfh.add_space("a", orbital_spaces[0])  # Cocc_A
    dfh.add_space("r", orbital_spaces[1])  # Cvir_A
    dfh.add_space("b", orbital_spaces[2])  # Cocc_B
    dfh.add_space("s", orbital_spaces[3])  # Cvir_B
    dfh.add_space("r1", orbital_spaces[4])  # Cr1
    dfh.add_space("s1", orbital_spaces[5])  # Cs1
    dfh.add_space("a2", orbital_spaces[6])  # Ca2
    dfh.add_space("b2", orbital_spaces[7])  # Cb2
    dfh.add_space("r3", orbital_spaces[8])  # Cr3
    dfh.add_space("s3", orbital_spaces[9])  # Cs3
    dfh.add_space("a4", orbital_spaces[10])  # Ca4
    dfh.add_space("b4", orbital_spaces[11])  # Cb4

    # Add DF transformations
    # Format: (name, left_space, right_space) -> computes (left|right) integrals
    dfh.add_transformation("Aar", "r", "a")  # (r|a) virtuals_A x occupied_A
    dfh.add_transformation("Abs", "s", "b")  # (s|b) virtuals_B x occupied_B
    dfh.add_transformation("Bas", "s1", "a")  # (s1|a) Cs1 x occupied_A
    dfh.add_transformation("Bbr", "r1", "b")  # (r1|b) Cr1 x occupied_B
    dfh.add_transformation("Cas", "s", "a2")  # (s|a2) virtuals_B x Ca2
    dfh.add_transformation("Cbr", "r", "b2")  # (r|b2) virtuals_A x Cb2
    dfh.add_transformation("Dar", "r3", "a")  # (r3|a) Cr3 x occupied_A
    dfh.add_transformation("Dbs", "s3", "b")  # (s3|b) Cs3 x occupied_B
    dfh.add_transformation("Ear", "r", "a4")  # (r|a4) virtuals_A x Ca4
    dfh.add_transformation("Ebs", "s", "b4")  # (s|b4) virtuals_B x Cb4

    # Perform DF transformations
    dfh.transform()

    # Clear spaces now that transformations are done
    dfh.clear_spaces()

    # => Memory blocking setup

    # Number of threads (single-threaded in Python)
    nT = 1

    # Calculate overhead for work arrays
    overhead = 0
    overhead += 2 * na * ns + 2 * nb * nr + 2 * na * nr + 2 * nb * ns  # S and Q
    # E_disp20 and E_exch_disp20 thread work and final
    overhead += 2 * na * nb * (nT + 1)
    # sE_exch_disp20 thread work and final
    overhead += 1 * sna * snb * (nT + 1)
    overhead += 1 * (nA + nfa + na) * (nB + nfb + nb)  # Disp_AB
    overhead += 1 * (snA + snfa + sna) * (snB + snfb + snb)  # sDisp_AB
    overhead += 12 * nn * nn  # D, V, J, K, P, C matrices for A and B

    total_memory = core.get_memory() // 8  # Convert bytes to doubles
    # Explicitly charge all eight host staging matrices. With a small budget,
    # shrink staging before consuming all remaining memory with packed DF
    # factors. One virtual orbital is the irreducible staging floor.
    stage_target = max(1, min(STAGE_DOUBLES, (total_memory - overhead) // 32))
    stage_r_cap = max(1, min(nr, stage_target // (nQ * max(na, nb))))
    stage_s_cap = max(1, min(ns, stage_target // (nQ * max(na, nb))))
    overhead += 2 * nQ * (na + nb) * (stage_r_cap + stage_s_cap)

    # The (r,s) pairs of a compute block are contracted as one matrix
    # V[(r,a),(s,b)] (see the main loop below), which needs seven work arrays
    # of nrb*na x nsb*nb doubles: V, T, I, T2, V2 and the energy denominator,
    # plus W2 for the (s,a) x (r,b) half of the exchange term.  Those GEMMs
    # are only part of the block's cost, so FDISP_BLOCK caps the compute block
    # at the whole-kernel optimum, independently of how much of the DF tensors
    # memory lets us hold at once.
    blk_r = min(block_edge, nr)
    blk_s = min(block_edge, ns)
    # Charge all four possible edge shapes and the graph input packs, not just
    # the main block. Shrink compute blocks before reducing DF staging.
    def compute_overhead(br, bs):
        work_arrays = 8 if use_graph else 7
        charge = 4 * work_arrays * br * bs * na * nb
        if use_graph:
            nk = 2 * nQ + 2
            charge += 4 * ((nk + nQ) * (br * na + bs * nb) + nk * (bs * na + br * nb))
        return charge

    while (blk_r > 1 or blk_s > 1) and 4 * compute_overhead(blk_r, blk_s) > total_memory - overhead:
        blk_r, blk_s = max(1, blk_r // 2), max(1, blk_s // 2)
    overhead += compute_overhead(blk_r, blk_s)
    # The slab _localize_columns stages each packed buffer through, sized to
    # fit in the two work arrays (W, IW) that localizing the factors made
    # unnecessary, so the total stays within what nine arrays used to take.
    # The exception is the one-orbital floor: when a block's two arrays hold
    # less than one orbital's columns (2*blk_r*blk_s*na*nb < nk*max(na, nb)),
    # the slab is one orbital, nk*max(na, nb) doubles.
    lo_slab = max(1, min(block_edge, max(nr, ns),
                         2 * blk_r * blk_s * na * nb // ((2 * nQ + 2) * max(na, nb))))
    if not _canonical:
        overhead += (2 * nQ + 2) * lo_slab * max(na, nb)

    # Available memory for dispersion calculation
    rem = total_memory - overhead

    core.print_out(
        f"    {total_memory} doubles - {overhead} overhead leaves {rem} for dispersion\n"
    )

    if rem < 0:
        raise Exception("Too little static memory for fdisp0")

    # Calculate cost per r or s virtual orbital
    # Each r contributes 2*nQ+2 Q/rank-one rows to AFar (na wide) and BCbr
    # (nb wide), i.e. (2*nQ+2)*(na+nb) doubles; likewise for each s. The DF
    # tensors are read into small staging matrices (STAGE_DOUBLES below) and
    # copied straight into those columns, so there is no second full-size copy
    # to pay for -- hence the factor of 2, for the r side and the s side.
    cost_r = (2 * nQ + 2) * (na + nb)
    max_r_l = rem // (2 * cost_r)
    max_s_l = max_r_l
    max_r = min(max_r_l, nr)
    max_s = min(max_s_l, ns)
    if _df_block_size is not None:
        max_r, max_s = min(max_r, _df_block_size), min(max_s, _df_block_size)

    if max_r < 1 or max_s < 1:
        raise Exception("Too little dynamic memory for fdisp0")

    # The compute block never exceeds the DF block that feeds it.
    max_r, max_s, blk_r, blk_s = _dispersion_block_extents(
        nr, ns, max_r, max_s, blk_r, blk_s)
    STAGE_R, STAGE_S = min(stage_r_cap, max_r), min(stage_s_cap, max_s)

    nrblocks = (nr + max_r - 1) // max_r  # Ceiling division
    nsblocks = (ns + max_s - 1) // max_s

    core.print_out(
        f"    Processing a single (r,s) pair requires {cost_r * 2} doubles\n"
    )
    core.print_out(f"    {nr} values of r processed in {nrblocks} blocks of {max_r}\n")
    core.print_out(
        f"    {ns} values of s processed in {nsblocks} blocks of {max_s}\n"
    )
    core.print_out(
        f"    (r,s) contracted in compute blocks of {blk_r} x {blk_s}\n\n"
    )
    # => Compute Far = Dar + Ear and Fbs = Dbs + Ebs
    # These represent combined D and E DF integrals that will be reused in the main loop

    # Add disk tensor for Far
    dfh.add_disk_tensor("Far", (nr, na, nQ))

    # Loop over r blocks to compute Far = Dar + Ear
    for rstart in range(0, nr, STAGE_R):
        nrblock = min(STAGE_R, nr - rstart)

        # Allocate matrices to hold the tensor slices
        Dar = core.Matrix("Dar block", nrblock * na, nQ)
        Ear = core.Matrix("Ear block", nrblock * na, nQ)

        # Fill Dar and Ear from disk tensors
        dfh.fill_tensor("Dar", Dar, [rstart, rstart + nrblock], [0, na], [0, nQ])
        dfh.fill_tensor("Ear", Ear, [rstart, rstart + nrblock], [0, na], [0, nQ])

        # Compute Far = Dar + Ear (element-wise addition)
        Dar.np[:, :] += Ear.np[:, :]

        # Write Far back to disk (Dar now contains Dar + Ear)
        dfh.write_disk_tensor("Far", Dar, (rstart, rstart + nrblock))
    del Dar, Ear

    # Add disk tensor for Fbs
    dfh.add_disk_tensor("Fbs", (ns, nb, nQ))

    # Loop over s blocks to compute Fbs = Dbs + Ebs
    for sstart in range(0, ns, STAGE_S):
        nsblock = min(STAGE_S, ns - sstart)

        # Allocate matrices to hold the tensor slices
        Dbs = core.Matrix("Dbs block", nsblock * nb, nQ)
        Ebs = core.Matrix("Ebs block", nsblock * nb, nQ)

        # Fill Dbs and Ebs from disk tensors
        dfh.fill_tensor("Dbs", Dbs, [sstart, sstart + nsblock], [0, nb], [0, nQ])
        dfh.fill_tensor("Ebs", Ebs, [sstart, sstart + nsblock], [0, nb], [0, nQ])

        # Compute Fbs = Dbs + Ebs (element-wise addition)
        Dbs.np[:, :] += Ebs.np[:, :]

        # Write Fbs back to disk (Dbs now contains Dbs + Ebs)
        dfh.write_disk_tensor("Fbs", Dbs, (sstart, sstart + nsblock))
    del Dbs, Ebs

    E_disp20_comp = _ein_zeros(na, nb, name="E_disp20")
    E_exch_disp20_comp = _ein_zeros(na, nb, name="E_exch_disp20")

    # => MO to LO Transformation
    UA = None if _canonical else _ein(cache["Uaocc0A"], name="Uaocc0A")
    UB = None if _canonical else _ein(cache["Uaocc0B"], name="Uaocc0B")

    # In the dispersion formula: indices a,b are occupied and r,s are virtual.
    ean = _arr(eps_occ_A)  # occupied energies of A (index a)
    ebn = _arr(eps_occ_B)  # occupied energies of B (index b)
    ern = eps_vir_A.np  # virtual energies for monomer A (index r)
    esn = eps_vir_B.np  # virtual energies for monomer B (index s)
    # The same four vectors as einsums tensors: the block energy denominator
    # is built straight out of them by linalg.outer_sum below.  outer_sum
    # takes its vectors as one homogeneous sequence, so the two full-range
    # operands are sliced into views to match the blocked r and s ones.
    ea_t, eb_t = _ein(ean, name="eps_a"), _ein(ebn, name="eps_b")
    er_t, es_t = _ein(ern, name="eps_r"), _ein(esn, name="eps_s")
    ea_v, eb_v = ea_t[:], eb_t[:]

    # => Work arrays for the blocked (r,s) kernel <= //
    #
    # A single (r,s) pair does only O(na*nb*nQ) flops, with na and nb in the
    # tens, so pair-at-a-time BLAS spends most of its time on call overhead.
    # The whole compute block is therefore held as one matrix,
    #
    #     V[(r,a),(s,b)] = sum_Q  X[(r,a),Q] Y[(s,b),Q] ,
    #
    # which turns a block's worth of tiny GEMMs into one big one.  The four
    # exchange DF terms and the four rank-1 (V,J,K) updates ride in the same
    # GEMM by concatenating them along Q with two extra columns:
    #
    #     AFar = [Aar | Far | Qar | SBar]   FAbs = [Fbs | Abs | SAbs | Qbs]
    #     BCas = [Bas | Cas | Sas | Qas]    BCbr = [Bbr | Cbr | Qbr | Sbr]
    #
    # so Disp20 costs one GEMM per block and Exch-Disp20 two.  einsums v2
    # tensors are column-major, so these are stored Q-major, (nk, n) with the
    # orbital index running fastest along a column; the transpose of the numpy
    # view is then a plain C-ordered (n, nk) buffer and every pack is a memcpy.
    nk = 2 * nQ + 2
    AFar = _ein_zeros(nk, max_r * na, name="AFar")
    BCbr = _ein_zeros(nk, max_r * nb, name="BCbr")
    FAbs = _ein_zeros(nk, max_s * nb, name="FAbs")
    BCas = _ein_zeros(nk, max_s * na, name="BCas")
    AFarn, BCbrn = np.asarray(AFar).T, np.asarray(BCbr).T
    FAbsn, BCasn = np.asarray(FAbs).T, np.asarray(BCas).T

    Sasn, Qasn = _arr(Sas), _arr(Qas)
    SAbsn, Qbsn = _arr(SAbs), _arr(Qbs)
    Qarn, SBarn = _arr(Qar), _arr(SBar)
    Qbrn, Sbrn = _arr(Qbr), _arr(Sbr)

    _bufs = {}

    def _work(nrb, nsb):
        """Work arrays for an nrb x nsb compute block, allocated once.

        Keyed on the block shape so that the short trailing blocks get their
        own correctly shaped tensors: the rank-4 reshapes below are views, and
        a view is only valid for the shape its tensor was allocated with.
        """
        key = (nrb, nsb)
        if key not in _bufs:
            M, N = nrb * na, nsb * nb
            Ms, Nr = nsb * na, nrb * nb
            # Every one of these is written in full before it is read (a GEMM
            # or elementwise kernel with beta = 0), so they are carved
            # uninitialized rather than zeroed.  That skips a serial memset,
            # and it leaves first touch to the threaded kernel that fills each
            # buffer, which spreads its pages over both sockets the way
            # numpy's lazily faulted np.zeros did for the v1 port.  A zeroed
            # tensor is first touched by this thread alone, and the block's
            # GEMMs then all read from one socket's memory.
            pool = _fdisp_pool(8 * ((7 if use_graph else 6) * M * N + Ms * Nr))
            b = dict(
                V=_fdisp_carve(pool, [M, N], "Vrs"),
                T=_fdisp_carve(pool, [M, N], "Trs"),
                I=_fdisp_carve(pool, [M, N], "Irs"),
                T2=_fdisp_carve(pool, [M, N], "T2rs"),
                V2=_fdisp_carve(pool, [M, N], "V2rs"),
                D=_fdisp_carve(pool, [M, N], "Drs"),
                W2=_fdisp_carve(pool, [Ms, Nr], "W2sr"),
            )
            if use_graph:
                b["X2"] = _fdisp_carve(pool, [M, N], "X2rs")
                b["X2v"] = b["X2"].reshape_view([na, nrb, nb, nsb])
            b["Dv"] = b["D"].reshape_view([na, nrb, nb, nsb])
            b["T2v"] = b["T2"].reshape_view([na, nrb, nb, nsb])
            b["V2v"] = b["V2"].reshape_view([na, nrb, nb, nsb])
            b["W2v"] = b["W2"].reshape_view([na, nsb, nb, nrb])
            # The contract-b operands of _to_lo and _from_lo with the column
            # index split into (b, y); b is the faster half of a column-major
            # column index.
            for k, rows, ncol in (("T", M, nsb), ("V2", M, nsb), ("I", M, nsb)):
                b[k + "3"] = b[k].reshape_view([rows, nb, ncol])
            _bufs[key] = b
        return _bufs[key]

    def _to_lo(b, x, i, y, nrow_blk, ncol_blk):
        """Y[(x,a),(y,b)] = sum_a' UA[a',a] sum_b' X[(x,a'),(y,b')] UB[b',b].

        The b' contraction is one einsum over all column blocks: Einsums runs
        it as a single gemm_batch, one (rows x b')(b' x b) GEMM per y against
        the same UB, which costs one Python dispatch instead of ncol_blk.  The
        a' contraction is one GEMM over the whole block, since a' is the
        fastest index of the column-major buffer.
        """
        ein.einsum("mcy <- mby ; bc", b[i + "3"], b[x + "3"], UB, c_pf=0.0, ab_pf=1.0)
        ncol = nrow_blk * ncol_blk * nb
        ein.linalg.gemm(1.0, UA, b[i].reshape_view([na, ncol]), 0.0,
                        b[y].reshape_view([na, ncol]), trans_a=True)

    def _from_lo(b, x, i, y, nrow_blk, ncol_blk):
        """The inverse of _to_lo: Y = UA X UB^T, back from the LO basis to MO.

        UA and UB are orthogonal (IBO rotations within the active occupied
        space), so the inverse is the transpose.
        """
        ein.einsum("mcy <- mby ; cb", b[i + "3"], b[x + "3"], UB, c_pf=0.0, ab_pf=1.0)
        ncol = nrow_blk * ncol_blk * nb
        ein.linalg.gemm(1.0, UA, b[i].reshape_view([na, ncol]), 0.0,
                        b[y].reshape_view([na, ncol]))

    # Localizing the DF factors rather than the products:
    #
    #     UA^T (AF^T FA) UB  =  (AF UA)^T (FA UB) ,
    #
    # so transforming each packed buffer's orbital columns once, as it is
    # staged, lets their GEMMs produce V2, X2 and W2 directly, and no MO copy
    # of the factors is kept.  Only the amplitudes need the MO basis, where the
    # denominator is diagonal: V is rotated back (_from_lo) and T forward
    # (_to_lo), two transforms per block where there were four, at no memory
    # cost.  This is the reassociation ComputeGraph's
    # ContractionPlanning reports for the captured block, which it cannot
    # apply itself because the chain runs through rank-4 views.
    lo_tmp = None if _canonical else _ein_zeros(nk, lo_slab * max(na, nb), name="LO slab")

    def _localize_columns(buf, n_o, nblk, U):
        """buf[:, (o,x)] <- sum_o' buf[:, (o',x)] U[o',o] for x < nblk, in place.

        Done a slab of columns at a time through lo_tmp: a column range of a
        column-major buffer is contiguous, so each slab is one memcpy out and
        one batched GEMM back.
        """
        if _canonical:
            return
        tmpn = np.asarray(lo_tmp)
        bufn = np.asarray(buf)
        for x0 in range(0, nblk, lo_slab):
            nc = min(lo_slab, nblk - x0)
            c0, c1 = x0 * n_o, (x0 + nc) * n_o
            tmpn[:, :c1 - c0] = bufn[:, c0:c1]
            ein.einsum("Qcx <- Qox ; oc", buf[:, c0:c1].reshape_view([nk, n_o, nc]),
                       lo_tmp[:, :c1 - c0].reshape_view([nk, n_o, nc]), U, c_pf=0.0, ab_pf=1.0)

    def _np2(m):
        """A DF block's numpy buffer as (nrow, nQ); fill_tensor may leave it 3-D."""
        a = m.np
        return a if a.ndim == 2 else a.reshape(-1, a.shape[-1])

    # => Main r,s loop <= //
    # The DF tensors are read off disk into staging matrices and copied
    # immediately into the packed buffers above.  Staging a whole max_r block
    # at once would hold a second, full-size copy of every DF tensor for the
    # entire loop -- at nanotube/aug-cc-pVDZ that is ~5 GB of resident memory
    # that is dead the moment the pack finishes.  Reading STAGE_R virtuals at
    # a time bounds it at a few tens of MB instead, at no measurable cost:
    # fill_tensor's per-call overhead is negligible against a slab this size.
    Aar = core.Matrix("Aar stage", STAGE_R * na, nQ)
    Far = core.Matrix("Far stage", STAGE_R * na, nQ)
    Bbr = core.Matrix("Bbr stage", STAGE_R * nb, nQ)
    Cbr = core.Matrix("Cbr stage", STAGE_R * nb, nQ)

    Abs = core.Matrix("Abs stage", STAGE_S * nb, nQ)
    Fbs = core.Matrix("Fbs stage", STAGE_S * nb, nQ)
    Bas = core.Matrix("Bas stage", STAGE_S * na, nQ)
    Cas = core.Matrix("Cas stage", STAGE_S * na, nQ)
    core.timer_off("F-SAPT Disp Setup")

    core.timer_on("F-SAPT Disp Compute")
    for rstart in range(0, nr, max_r):
        nrblock = min(max_r, nr - rstart)
        rsl = slice(rstart, rstart + nrblock)

        # Pack the r side of the block once per DF block.
        M, Nr = nrblock * na, nrblock * nb
        for c0 in range(0, nrblock, STAGE_R):
            c1 = min(c0 + STAGE_R, nrblock)
            r0, r1, nc = rstart + c0, rstart + c1, c1 - c0
            dfh.fill_tensor("Aar", Aar, [r0, r1], [0, na], [0, nQ])
            dfh.fill_tensor("Far", Far, [r0, r1], [0, na], [0, nQ])
            dfh.fill_tensor("Bbr", Bbr, [r0, r1], [0, nb], [0, nQ])
            dfh.fill_tensor("Cbr", Cbr, [r0, r1], [0, nb], [0, nQ])
            AFarn[c0 * na:c1 * na, 0:nQ] = _np2(Aar)[:nc * na]
            AFarn[c0 * na:c1 * na, nQ:2 * nQ] = _np2(Far)[:nc * na]
            BCbrn[c0 * nb:c1 * nb, 0:nQ] = _np2(Bbr)[:nc * nb]
            BCbrn[c0 * nb:c1 * nb, nQ:2 * nQ] = _np2(Cbr)[:nc * nb]
        AFarn[:M, 2 * nQ] = Qarn[:, rsl].T.reshape(-1)
        AFarn[:M, 2 * nQ + 1] = SBarn[:, rsl].T.reshape(-1)
        BCbrn[:Nr, 2 * nQ] = Qbrn[:, rsl].T.reshape(-1)
        BCbrn[:Nr, 2 * nQ + 1] = Sbrn[:, rsl].T.reshape(-1)
        _localize_columns(AFar, na, nrblock, UA)
        _localize_columns(BCbr, nb, nrblock, UB)

        for sstart in range(0, ns, max_s):
            nsblock = min(max_s, ns - sstart)
            ssl = slice(sstart, sstart + nsblock)

            N, Ms = nsblock * nb, nsblock * na
            for c0 in range(0, nsblock, STAGE_S):
                c1 = min(c0 + STAGE_S, nsblock)
                s0, s1, nc = sstart + c0, sstart + c1, c1 - c0
                dfh.fill_tensor("Abs", Abs, [s0, s1], [0, nb], [0, nQ])
                dfh.fill_tensor("Fbs", Fbs, [s0, s1], [0, nb], [0, nQ])
                dfh.fill_tensor("Bas", Bas, [s0, s1], [0, na], [0, nQ])
                dfh.fill_tensor("Cas", Cas, [s0, s1], [0, na], [0, nQ])
                FAbsn[c0 * nb:c1 * nb, 0:nQ] = _np2(Fbs)[:nc * nb]
                FAbsn[c0 * nb:c1 * nb, nQ:2 * nQ] = _np2(Abs)[:nc * nb]
                BCasn[c0 * na:c1 * na, 0:nQ] = _np2(Bas)[:nc * na]
                BCasn[c0 * na:c1 * na, nQ:2 * nQ] = _np2(Cas)[:nc * na]
            FAbsn[:N, 2 * nQ] = SAbsn[:, ssl].T.reshape(-1)
            FAbsn[:N, 2 * nQ + 1] = Qbsn[:, ssl].T.reshape(-1)
            BCasn[:Ms, 2 * nQ] = Sasn[:, ssl].T.reshape(-1)
            BCasn[:Ms, 2 * nQ + 1] = Qasn[:, ssl].T.reshape(-1)
            _localize_columns(FAbs, nb, nsblock, UB)
            _localize_columns(BCas, na, nsblock, UA)

            # => RS inner loop <= //
            for r0 in range(0, nrblock, blk_r):
                nrb = min(blk_r, nrblock - r0)
                ra0, ra1 = r0 * na, (r0 + nrb) * na
                rb0, rb1 = r0 * nb, (r0 + nrb) * nb
                rr = slice(rstart + r0, rstart + r0 + nrb)

                for s0 in range(0, nsblock, blk_s):
                    nsb = min(blk_s, nsblock - s0)
                    sb0, sb1 = s0 * nb, (s0 + nsb) * nb
                    sa0, sa1 = s0 * na, (s0 + nsb) * na
                    ss = slice(sstart + s0, sstart + s0 + nsb)

                    b = _work(nrb, nsb)
                    V, T, I = b["V"], b["T"], b["I"]
                    T2, V2, D = b["T2"], b["V2"], b["D"]
                    W2 = b["W2"]

                    # => Amplitudes, Disp20 <= //

                    # V2[(r,a),(s,b)] = sum_Q Aar[(r,a),Q] Abs[(s,b),Q], from the
                    # localized factors, so in the LO basis; V is its MO image.
                    if contractions is not None:
                        # Three independent DF products in one replay. The
                        # GPU helper packs owning column-major inputs and uses
                        # rank-2 einsum descriptors, never linalg.gemm.
                        contractions.products(
                            AFar[:, ra0:ra1], FAbs[:, sb0:sb1],
                            BCas[:, sa0:sa1], BCbr[:, rb0:rb1],
                            nQ, V2, b["X2"], W2)
                    else:
                        ein.linalg.gemm(1.0, AFar[0:nQ, ra0:ra1],
                                        FAbs[nQ:2 * nQ, sb0:sb1], 0.0, V2,
                                        trans_a=True)
                    if not _canonical:
                        _from_lo(b, "V2", "I", "V", nrb, nsb)

                    # Amplitudes T = V / (ea + eb - er - es).  outer_sum is
                    # einsums' own MP2-denominator primitive and builds the
                    # rank-4 denominator in one threaded pass; numpy's
                    # broadcast form has to materialise a full-block temporary
                    # first and runs single-threaded, which cost twice as much.
                    ein.linalg.outer_sum(b["Dv"], [ea_v, er_t[rr], eb_v, es_t[ss]],
                                         [1.0, -1.0, 1.0, -1.0])
                    ein.linalg.direct_division(1.0, V2 if _canonical else V,
                                               D, 0.0, T2 if _canonical else T)

                    # Transform to localized orbital basis and accumulate
                    if not _canonical:
                        _to_lo(b, "T", "I", "T2", nrb, nsb)
                    ein.einsum("ab <- arbs ; arbs", E_disp20_comp,
                               b["T2v"], b["V2v"], c_pf=1.0, ab_pf=4.0)

                    # => Exch-Disp20 <= //

                    # (r,a) x (s,b) half: Aar.Fbs + Far.Abs + Qar.SAbs + SBar.Qbs,
                    # from the localized factors, so it lands in V2 already in
                    # the LO basis (see _localize_columns).
                    if contractions is None:
                        ein.linalg.gemm(1.0, AFar[:, ra0:ra1], FAbs[:, sb0:sb1],
                                        0.0, V2, trans_a=True)
                    ein.einsum("ab <- arbs ; arbs", E_exch_disp20_comp,
                               b["T2v"], b["X2v"] if use_graph else b["V2v"],
                               c_pf=1.0, ab_pf=-2.0)

                    # (s,a) x (r,b) half: Bas.Bbr + Cas.Cbr + Sas.Qbr + Qas.Sbr.
                    # The localization is linear and the energy contraction
                    # elementwise, so this half stays in its own layout and is
                    # reduced against T2 through a transposed index map rather
                    # than permuted into the (r,a) x (s,b) one.
                    if contractions is None:
                        ein.linalg.gemm(1.0, BCas[:, sa0:sa1], BCbr[:, rb0:rb1],
                                        0.0, W2, trans_a=True)
                    ein.einsum("ab <- arbs ; asbr", E_exch_disp20_comp,
                               b["T2v"], b["W2v"], c_pf=1.0, ab_pf=-2.0)

    core.timer_off("F-SAPT Disp Compute")
    # => Accumulate thread results <= //
    _store_dispersion0_components(cache, np.asarray(E_disp20_comp), np.asarray(E_exch_disp20_comp),
                                  nA, nB, nfa, nfb, na1, nb1)
    if contractions is not None:
        cache["Disp0 execution"] = contractions.report(time.perf_counter() - started)
        cache["Disp0 execution"].update({
            "df_block_extents": [max_r, max_s], "compute_block_extents": [blk_r, blk_s],
            "cached_work_shapes": len(_bufs),
            "staging_extents": [STAGE_R, STAGE_S],
            "planned_overhead_doubles": overhead,
        })
    # if do_print:
    #     core.print_out(f"    Disp20              = {Disp20 * 1000:.8f} [mEh]\n")
    #     core.print_out(f"    Exch-Disp20         = {ExchDisp20 * 1000:.8f} [mEh]\n")
    #     core.print_out("\n")
    #     assert abs(scalars['Disp20,u'] - Disp20) < 1e-6, f"Disp20 scalar mismatch! {scalars['Disp20,u'] = } {Disp20 = }"
    #     assert abs(scalars['Exch-Disp20,u'] - ExchDisp20) < 1e-6, f"ExchDisp20 scalar mismatch!\nRef: {scalars['Exch-Disp20,u']:.4e}\nAct: {ExchDisp20:.4e}"
    return cache


# Present only in a cuEST-enabled build; see psi4/src/cuest_gemm.cc.
_HAVE_CUEST_GEMM = hasattr(core, "cuest_chain_gemm")


def _chain_gemm_work(tensors: list, transposes: list[str]) -> int:
    """Return the largest ``m * k * n`` among the links of a gemm chain.

    Follows the same shape rules as :func:`chain_gemm_einsums` itself: only the
    first left operand may be transposed, and the row count of the running
    product never changes after the first link.
    """
    rows, cols = tensors[0].shape
    if transposes[0] == "T":
        rows, cols = cols, rows
    largest = 0
    for i in range(len(tensors) - 1):
        b_rows, b_cols = tensors[i + 1].shape
        if transposes[i + 1] == "T":
            b_rows, b_cols = b_cols, b_rows
        largest = max(largest, rows * cols * b_cols)
        cols = b_cols
    return largest


def _use_cuest_gemm(tensors: list, transposes: list[str], prefactors_C: list[float]) -> bool:
    """Whether a gemm chain is worth sending to cuBLAS.

    A chain has to earn the two PCIe transfers that bracket it, so small ones
    stay on the CPU; ``CUEST_GEMM_MIN_DIM`` sets the crossover, expressed as the
    dimension of a square product.  ``prefactors_C`` is the other bar: the GPU
    path always accumulates into a fresh matrix, so a caller that actually wants
    a non-zero beta gets the einsums path.
    """
    if not _HAVE_CUEST_GEMM or len(tensors) < 2:
        return False
    # The native binding takes SharedMatrix, not an Einsums tensor. The v2
    # out="tensor" continuation must keep its existing Einsums-owned path.
    if any(not isinstance(value, core.Matrix) for value in tensors):
        return False
    if not core.get_global_option("USE_CUEST"):
        return False
    if prefactors_C is not None and any(prefactors_C):
        return False
    min_dim = core.get_global_option("CUEST_GEMM_MIN_DIM")
    return _chain_gemm_work(tensors, transposes) >= min_dim**3


def chain_gemm_einsums(
    tensors: list,
    transposes: list[str] = None,
    prefactors_C: list[float] = None,
    prefactors_AB: list[float] = None,
    return_tensors: list[bool] = None,
    out: str = "matrix",
):
    """
    Computes a chain of matrix multiplications with einsums.

    The chain is evaluated entirely inside einsums tensors: each input is
    copied in at most once and every intermediate stays einsums-owned, so an
    N-factor chain crosses the psi4/einsums boundary N times instead of once
    per gemm.

    Parameters
    ----------
    tensors : list
        Factors of the chain, as psi4 Matrices, numpy arrays, or einsums
        tensors (any mix).
    transposes : list[str], optional
        List of transpose operations for each tensor, where "N" means no transpose and "T" means transpose.
    prefactors_C : list[float], optional
        List of prefactors for the resulting tensors in the chain.
    prefactors_AB : list[float], optional
        List of prefactors for the tensors being multiplied in the chain.
    return_tensors : list[bool], optional
        List indicating which intermediate tensors should be returned. If None,
        only the final tensor is returned. Note that these are only
        intermediate tensors and final tensor; hence, the length of this list
        should be one less than the number of tensors.
    out : {"matrix", "tensor"}, optional
        Return psi4 Matrices (default, so call sites are unchanged) or the
        einsums tensors themselves. Use ``"tensor"`` when the result feeds
        straight back into einsums, to skip the copy out.
    """
    N = len(tensors)
    if transposes is None:
        transposes = ["N"] * N
    if prefactors_C is None:
        prefactors_C = [0.0] * (N - 1)
    if prefactors_AB is None:
        prefactors_AB = [1.0] * (N - 1)
    if _use_cuest_gemm(tensors, transposes, prefactors_C):
        # One upload, one download: the running product stays on the device for
        # the whole chain rather than round-tripping through host memory at
        # every link.  prefactors_C is deliberately not forwarded -- C is
        # allocated and zeroed immediately before each gemm below, so beta only
        # ever multiplies zero, and _use_cuest_gemm has already refused any
        # chain that asks for something else.
        convert = _ein if out == "tensor" else (lambda value: value)
        if return_tensors is None:
            flags = [False] * (N - 2) + [True]
            return convert(core.cuest_chain_gemm(tensors, transposes, prefactors_AB, flags)[0])
        flags = [bool(r) for r in return_tensors][: N - 1]
        flags += [False] * (N - 1 - len(flags))
        return [convert(value) for value in
                core.cuest_chain_gemm(tensors, transposes, prefactors_AB, flags)]

    # one boundary crossing per input factor, then stay in einsums
    ein_inputs = [_ein(t, name=f"chain_in{i}") for i, t in enumerate(tensors)]
    computed_tensors = [ein_inputs[0]]
    try:
        for i in range(N - 1):
            A = computed_tensors[-1]
            B = ein_inputs[i + 1]

            # For intermediate results (i > 0), always use 'N' for T1
            # since A is a computed intermediate
            T1 = transposes[i] if i == 0 else "N"
            T2 = transposes[i + 1]
            A_size = A.shape[1] if T1 == "T" else A.shape[0]
            B_size = B.shape[0] if T2 == "T" else B.shape[1]

            C = _ein_zeros(A_size, B_size, name=f"chain_out{i}")
            ein.linalg.gemm(
                prefactors_AB[i], A, B, prefactors_C[i], C,
                trans_a=(T1 == "T"), trans_b=(T2 == "T"),
            )
            computed_tensors.append(C)
    except Exception as e:
        raise ValueError(
            f"Error in einsum_chain_gemm: {e}\n{i=}\n{A=}\n{B=}\n{T1=}\n{T2=}"
        )
    convert = (lambda t: t) if out == "tensor" else _mat
    if return_tensors is None:
        return convert(computed_tensors[-1])
    returned_tensors = []
    for i, r in enumerate(return_tensors):
        if r:
            returned_tensors.append(convert(computed_tensors[i + 1]))
    return returned_tensors


def exchange(cache: dict, jk: core.JK, do_print: bool = True) -> dict:
    r"""Compute the first-order exchange energy :math:`E^{(1)}_{\text{exch}}`.

    Evaluates both the :math:`S^2` approximation (Eq. 6) and the
    :math:`S^\infty` (Eq. 9) first-order exchange energies from
    Xie et al. (2022).

    The :math:`S^2` approximation is:

    .. math::

        E^{(1)}_{\text{exch}}(S^2) = -2(\mathbf{P}^A \mathbf{S} \mathbf{P}^B \mathbf{S} \mathbf{P}^{A,\text{vir}}) \cdot \boldsymbol{\omega}^B
          - 2(\mathbf{P}^B \mathbf{S} \mathbf{P}^A \mathbf{S} \mathbf{P}^{B,\text{vir}}) \cdot \boldsymbol{\omega}^A
          - 2(\mathbf{P}^{A,\text{vir}} \mathbf{S} \mathbf{P}^B) \cdot \mathbf{K}[\mathbf{P}^A \mathbf{S} \mathbf{P}^{B,\text{vir}}]

    The :math:`S^\infty` exchange energy uses the full inverse overlap metric (Eq. 9):

    .. math::

        E^{(1)}_{\text{exch}} = -2\mathbf{P}^A \cdot \mathbf{K}^B
          + 2\mathbf{T}^{AB} \cdot (\mathbf{h}^A + \mathbf{h}^B)
          + 2\mathbf{T}^{AA} \cdot \mathbf{h}^B
          + 2\mathbf{T}^{BB} \cdot \mathbf{h}^A
          + 2\mathbf{T}^{BB} \cdot \mathbf{W}^{AB}
          + 2\mathbf{T}^{AA} \cdot \mathbf{W}^{AB}
          + 2\mathbf{T}^{BB} \cdot \mathbf{W}^{AA}
          + 2\mathbf{T}^{AB} \cdot \mathbf{W}^{AB}

    where the intermediates are (Eq. 8, 10):

    .. math::

        \boldsymbol{\omega}^X = 2\mathbf{J}^X + \mathbf{V}^X

        \mathbf{h}^X = \mathbf{V}^X + 2\mathbf{J}^X - \mathbf{K}^X

    Parameters
    ----------
    cache : dict
        SAPT data cache from :func:`build_sapt_jk_cache`.
    jk : core.JK
        JK integral engine for computing Coulomb and exchange matrices.
    do_print : bool, optional
        Whether to print the result, by default True.

    Returns
    -------
    dict
        Dictionary with keys ``'Exch10(S^2)'`` and ``'Exch10'`` mapping to
        the :math:`S^2` and :math:`S^\infty` exchange energies, respectively.
    """

    if do_print:
        core.print_out("\n  ==> E10 Exchange Einsums <== \n\n")

    # Eq. 10: h^A = V^A + 2*J^A - K^A
    h_A = _ein_clone(cache["V_A"], name="h_A")
    _axpy(2.0, cache["J_A"], h_A)
    _axpy(-1.0, cache["K_A"], h_A)

    # Eq. 10: h^B = V^B + 2*J^B - K^B
    h_B = _ein_clone(cache["V_B"], name="h_B")
    _axpy(2.0, cache["J_B"], h_B)
    _axpy(-1.0, cache["K_B"], h_B)

    # Eq. 8: omega^A = V^A + 2*J^A
    w_A = _ein_clone(cache["V_A"], name="w_A")
    _axpy(2.0, cache["J_A"], w_A)

    # Eq. 8: omega^B = V^B + 2*J^B
    w_B = _ein_clone(cache["V_B"], name="w_B")
    _axpy(2.0, cache["J_B"], w_B)

    # Build inverse exchange metric
    nocc_A = cache["Cocc_A"].shape[1]
    nocc_B = cache["Cocc_B"].shape[1]
    SAB = chain_gemm_einsums(
        [cache["Cocc_A"], cache["S"], cache["Cocc_B"]],
        ["T", "N", "N"],
    )

    num_occ = nocc_A + nocc_B

    Sab = core.Matrix(num_occ, num_occ)
    Sab.np[:nocc_A, nocc_A:] = SAB.np
    Sab.np[nocc_A:, :nocc_A] = SAB.np.T
    Sab.np[np.diag_indices_from(Sab.np)] += 1
    Sab.power(-1.0, 1.0e-14)
    Sab.np[np.diag_indices_from(Sab.np)] -= 1.0

    Tmo_AA = core.Matrix.from_array(Sab.np[:nocc_A, :nocc_A])
    Tmo_BB = core.Matrix.from_array(Sab.np[nocc_A:, nocc_A:])
    Tmo_AB = core.Matrix.from_array(Sab.np[:nocc_A, nocc_A:])

    T_AA = chain_gemm_einsums(
        [cache["Cocc_A"], Tmo_AA, cache["Cocc_A"]], ["N", "N", "T"], out="tensor"
    )
    T_BB = chain_gemm_einsums(
        [cache["Cocc_B"], Tmo_BB, cache["Cocc_B"]], ["N", "N", "T"], out="tensor"
    )
    T_AB = chain_gemm_einsums(
        [cache["Cocc_A"], Tmo_AB, cache["Cocc_B"]], ["N", "N", "T"], out="tensor"
    )

    S = cache["S"]
    D_A = cache["D_A"]
    P_A = cache["P_A"]
    D_B = cache["D_B"]
    P_B = cache["P_B"]

    # Compute the J and K matrices
    jk.C_clear()

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_A"]))
    jk.C_right_add(chain_gemm_einsums([cache["Cocc_A"], Tmo_AA]))

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_B"]))
    jk.C_right_add(chain_gemm_einsums([cache["Cocc_A"], Tmo_AB]))

    jk.C_left_add(core.Matrix.from_array(cache["Cocc_A"]))
    jk.C_right_add(chain_gemm_einsums([P_B, S, cache["Cocc_A"]]))
    # This also works... you can choose to form the density-like matrix either
    # way..., just remember that the C_right_add has an adjoint (transpose, and switch matmul order)
    # jk.C_left_add(core.Matrix.from_array(einsum_chain_gemm([D_A, S, cache['Cvir_B']])))
    # jk.C_right_add(core.Matrix.from_array(cache['Cvir_B']))
    jk.compute()

    JT_A, JT_AB, Jij = jk.J()
    KT_A, KT_AB, Kij = jk.K()

    # Eq. 6: E^(1)_exch(S^2) — three-term S^2 exchange
    Exch_s2 = 0.0

    # Save some intermediate tensors to avoid recomputation in the next
    # steps
    DA_S_DB_S_PA = chain_gemm_einsums([D_A, S, D_B, S, P_A], out="tensor")
    Exch_s2 -= 2.0 * _dot(w_B, DA_S_DB_S_PA)

    DB_S_DA_S_PB = chain_gemm_einsums([D_B, S, D_A, S, P_B], out="tensor")
    Exch_s2 -= 2.0 * _dot(w_A, DB_S_DA_S_PB)
    Exch_s2 -= 2.0 * _dot(Kij, chain_gemm_einsums([P_A, S, D_B], out="tensor"))

    if do_print:
        core.print_out(print_sapt_var("Exch10(S^2) ", Exch_s2, short=True))
        core.print_out("\n")

    # Eq. 9: E^(1)_exch(S^inf) — full inverse-overlap exchange
    Exch10 = 0.0
    h_AB = _ein_clone(h_A, name="h_A+h_B")
    _axpy(1.0, h_B, h_AB)

    JT_AB_e = _ein(JT_AB, name="JT_AB")
    KT_AB_e = _ein(KT_AB, name="KT_AB")
    G_AB = _ein_clone(JT_AB_e, name="JT_AB-KT_AB/2")
    _axpy(-0.5, KT_AB_e, G_AB)
    G_ABt = _ein_clone(JT_AB_e, name="JT_AB-KT_AB^T/2")
    _axpy(-0.5, KT_AB_e.T, G_ABt)
    G_A = _ein_clone(JT_A, name="JT_A-KT_A/2")
    _axpy(-0.5, KT_A, G_A)

    Exch10 -= 2.0 * _dot(D_A, cache["K_B"])
    Exch10 += 2.0 * _dot(T_AA, h_B)
    Exch10 += 2.0 * _dot(T_BB, h_A)
    Exch10 += 2.0 * _dot(T_AB, h_AB)
    Exch10 += 4.0 * _dot(T_BB, G_AB)
    Exch10 += 4.0 * _dot(T_AA, G_ABt)
    Exch10 += 4.0 * _dot(T_BB, G_A)
    Exch10 += 4.0 * _dot(T_AB, G_ABt)

    if do_print:
        core.set_variable("Exch10", Exch10)
        core.print_out(print_sapt_var("Exch10", Exch10, short=True))
        core.print_out("\n")

    return {"Exch10(S^2)": Exch_s2, "Exch10": Exch10}

def induction(
    cache: dict,
    jk: core.JK,
    do_print: bool = True,
    maxiter: int = 12,
    conv: float = 1.0e-8,
    do_response: bool = True,
    Sinf: bool = False,
    sapt_jk_B: core.JK | None = None,
) -> dict:
    r"""Compute second-order induction and exchange-induction energies.

    Evaluates the uncoupled and (optionally) coupled second-order induction
    energy :math:`E^{(2)}_{\text{ind}}` and exchange-induction energy
    :math:`E^{(2)}_{\text{exch-ind}}` from Xie et al. (2022).

    The induction energy for monomer A polarized by B is (Eq. 14):

    .. math::

        E^{(2)}_{\text{ind}}(A \leftarrow B) = 2\mathbf{x}^A \cdot \tilde{\boldsymbol{\omega}}^B

    where the induction potential :math:`\tilde{\omega}^B` is given by (Eq. 16):

    .. math::

        \tilde{\boldsymbol{\omega}}^A = (\mathbf{C}^{B,\text{occ}})^\dagger \boldsymbol{\omega}^A \mathbf{C}^{B,\text{vir}}

    and the uncoupled response amplitudes are (Eq. 20):

    .. math::

        (x^A)^a_r = -(\tilde{\omega}^B)^a_r / (\epsilon_r - \epsilon_a)

    For monomer B polarized by A, the formulas are analogous with A and B swapped.

    Parameters
    ----------
    cache : dict
        SAPT data cache from :func:`build_sapt_jk_cache`.
    jk : core.JK
        JK integral engine for Coulomb and exchange matrices.
    do_print : bool, optional
        Whether to print results, by default True.
    maxiter : int, optional
        Maximum CPSCF iterations for coupled induction, by default 12.
    conv : float, optional
        Convergence threshold for CPSCF solver, by default 1.0e-8.
    do_response : bool, optional
        Whether to compute coupled (CPSCF) induction, by default True.
    Sinf : bool, optional
        Whether to include :math:`S^\infty` exchange-induction, by default False.
    sapt_jk_B : core.JK or None, optional
        Separate JK object for monomer B, by default None (uses same as A).

    Returns
    -------
    dict
        Dictionary containing induction energies with keys such as
        ``'Ind20,u (A<-B)'``, ``'Ind20,u (A->B)'``, ``'Ind20,u'``,
        ``'Exch-Ind20,u (A<-B)'``, ``'Exch-Ind20,u (A->B)'``,
        ``'Exch-Ind20,u'``, and coupled variants (``'Ind20,r'``, etc.)
        when ``do_response=True``.
    """

    if do_print:
        core.print_out("\n  ==> E20 Induction Einsums <== \n\n")

    # Build Induction and Exchange-Induction potentials
    S = cache["S"]

    D_A = cache["D_A"]
    V_A = cache["V_A"]

    J_A = cache["J_A"]
    K_A = cache["K_A"]

    D_B = cache["D_B"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]
    K_B = cache["K_B"]

    K_O = cache["K_O"]
    J_O = cache["J_O"]

    # Prepare JK calculations.  The three left-hand C matrices share the
    # D_B S and D_A S D_B S products, so they go into one graph and the
    # sharing is left to CSE; JK only sees them once the block has run.
    e_S = _ein(S, "S")
    e_DA = _ein(D_A, "D_A")
    e_DB = _ein(D_B, "D_B")
    e_CoA = _ein(cache["Cocc_A"], "Cocc_A")
    e_CoB = _ein(cache["Cocc_B"], "Cocc_B")

    DB_S_CA = _ein_zeros(e_S.shape[0], e_CoA.shape[1], name="DB_S_CA")
    DB_S_DA_S_CB = _ein_zeros(e_S.shape[0], e_CoB.shape[1], name="DB_S_DA_S_CB")
    DA_S_DB_S_CA = _ein_zeros(e_S.shape[0], e_CoA.shape[1], name="DA_S_DB_S_CA")
    with graph_block("induction_jk_C"):
        chain_into(DB_S_CA, [e_DB, e_S, e_CoA], "NNN", beta=0.0, name="DB_S_CA")
        chain_into(DB_S_DA_S_CB, [e_DB, e_S, e_DA, e_S, e_CoB], "NNNNN",
                   beta=0.0, name="DB_S_DA_S_CB")
        chain_into(DA_S_DB_S_CA, [e_DA, e_S, e_DB, e_S, e_CoA], "NNNNN",
                   beta=0.0, name="DA_S_DB_S_CA")

    jk.C_clear()

    jk.C_left_add(_mat(DB_S_CA))
    jk.C_right_add(_mat(cache["Cocc_A"]))

    jk.C_left_add(_mat(DB_S_DA_S_CB))
    jk.C_right_add(_mat(cache["Cocc_B"]))

    jk.C_left_add(_mat(DA_S_DB_S_CA))
    jk.C_right_add(_mat(cache["Cocc_A"]))

    jk.compute()

    J_Ot, J_P_B, J_P_A = jk.J()
    K_Ot, K_P_B, K_P_A = jk.K()

    # Save for later usage in find()
    cache["J_P_A"] = J_P_A
    cache["J_P_B"] = J_P_B

    # Eq. 17: exchange-induction potential for A due to B
    EX_A = _ein_clone(K_B, name="EX_A", scale=-1.0)
    _axpy(-2.0, J_O, EX_A)
    _axpy(1.0, K_O, EX_A)
    _axpy(2.0, J_P_B, EX_A)

    # Apply all the axpy operations to EX_A
    S_DB, S_DB_VA, S_DB_VA_DB_S = chain_gemm_einsums(
        [S, D_B, V_A, D_B, S], return_tensors=[True, True, False, True]
    )
    S_DB_JA, S_DB_JA_DB_S = chain_gemm_einsums(
        [S_DB, J_A, D_B, S], return_tensors=[True, False, True]
    )
    S_DB_S_DA, S_DB_S_DA_VB = chain_gemm_einsums(
        [S_DB, S, D_A, V_B],
        return_tensors=[False, True, True],
    )
    _axpy(-1.0, S_DB_VA, EX_A)
    _axpy(-2.0, S_DB_JA, EX_A)
    _axpy(1.0, chain_gemm_einsums([S_DB, K_A], out="tensor"), EX_A)
    _axpy(1.0, S_DB_S_DA_VB, EX_A)
    _axpy(2.0, chain_gemm_einsums([S_DB_S_DA, J_B], out="tensor"), EX_A)
    _axpy(1.0, S_DB_VA_DB_S, EX_A)
    _axpy(2.0, S_DB_JA_DB_S, EX_A)
    _axpy(-1.0, chain_gemm_einsums([S_DB, K_O], ["N", "T"], out="tensor"), EX_A)
    _axpy(-1.0, chain_gemm_einsums([V_B, D_B, S], out="tensor"), EX_A)
    _axpy(-2.0, chain_gemm_einsums([J_B, D_B, S], out="tensor"), EX_A)
    _axpy(1.0, chain_gemm_einsums([K_B, D_B, S], out="tensor"), EX_A)
    _axpy(1.0, chain_gemm_einsums([V_B, D_A, S, D_B, S], out="tensor"), EX_A)
    _axpy(2.0, chain_gemm_einsums([J_B, D_A, S, D_B, S], out="tensor"), EX_A)
    _axpy(-1.0, chain_gemm_einsums([K_O, D_B, S], out="tensor"), EX_A)

    EX_A_MO_1 = chain_gemm_einsums(
        [cache["Cocc_A"], EX_A, cache["Cvir_A"]],
        ["T", "N", "N"],
    )
    mapA = {
        "S": S,
        "J_O": J_O,
        "K_O": K_O,
        "Cocc_A": cache["Cocc_A"],
        "Cvir_A": cache["Cvir_A"],
        "D_A": D_A,
        "V_A": V_A,
        "J_A": J_A,
        "K_A": K_A,
        "J_P_A": J_P_A,
        "Cocc_B": cache["Cocc_B"],
        "Cvir_B": cache["Cvir_B"],
        "D_B": D_B,
        "V_B": V_B,
        "J_B": J_B,
        "K_B": K_B,
        "J_P_B": J_P_B,
    }
    EX_A_MO = build_exch_ind_pot_AB(mapA)
    assert np.allclose(EX_A_MO, EX_A_MO_1), "EX_A_MO and EX_A_MO_1 do not match!"

    # Eq. 17: exchange-induction potential for B due to A
    EX_B = _ein_clone(K_A, name="EX_B", scale=-1.0)
    _axpy(-2.0, J_O, EX_B)
    _axpy(1.0, K_O, EX_B.T)
    _axpy(2.0, J_P_A, EX_B)
    cache["J_P_A"] = J_P_A
    cache["J_P_B"] = J_P_B

    S_DA, S_DA_VB, S_DA_VB_DA_S = chain_gemm_einsums(
        [S, D_A, V_B, D_A, S], return_tensors=[True, True, False, True]
    )
    S_DA_JB, S_DA_JB_DA_S = chain_gemm_einsums(
        [S_DA, J_B, D_A, S], return_tensors=[True, False, True]
    )
    S_DA_S_DB, S_DA_S_DB_VA = chain_gemm_einsums(
        [S_DA, S, D_B, V_A],
        return_tensors=[False, True, True],
    )

    # Apply all the axpy operations to EX_B
    _axpy(-1.0, S_DA_VB, EX_B)
    _axpy(-2.0, S_DA_JB, EX_B)
    _axpy(1.0, chain_gemm_einsums([S_DA, K_B], out="tensor"), EX_B)
    _axpy(1.0, S_DA_S_DB_VA, EX_B)
    _axpy(2.0, chain_gemm_einsums([S_DA_S_DB, J_A], out="tensor"), EX_B)
    _axpy(1.0, S_DA_VB_DA_S, EX_B)
    _axpy(2.0, S_DA_JB_DA_S, EX_B)
    _axpy(-1.0, chain_gemm_einsums([S_DA, K_O], out="tensor"), EX_B)
    _axpy(-1.0, chain_gemm_einsums([V_A, D_A, S], out="tensor"), EX_B)
    _axpy(-2.0, chain_gemm_einsums([J_A, D_A, S], out="tensor"), EX_B)
    _axpy(1.0, chain_gemm_einsums([K_A, D_A, S], out="tensor"), EX_B)
    _axpy(1.0, chain_gemm_einsums([V_A, D_B, S, D_A, S], out="tensor"), EX_B)
    _axpy(2.0, chain_gemm_einsums([J_A, D_B, S, D_A, S], out="tensor"), EX_B)
    _axpy(-1.0, chain_gemm_einsums([K_O, D_A, S], ["T", "N", "N"], out="tensor"), EX_B)

    EX_B_MO_1 = chain_gemm_einsums(
        [cache["Cocc_B"], EX_B, cache["Cvir_B"]],
        ["T", "N", "N"],
    )
    EX_B_MO = build_exch_ind_pot_BA(mapA)
    assert np.allclose(EX_B_MO, EX_B_MO_1), "EX_B_MO and EX_B_MO_1 do not match!"

    # Eq. 8: omega^A = V^A + 2*J^A
    w_A = V_A.clone()
    w_A.name = "w_A"
    _axpy(2.0, J_A, w_A)

    # Eq. 8: omega^B = V^B + 2*J^B
    w_B = V_B.clone()
    w_B.name = "w_B"
    _axpy(2.0, J_B, w_B)

    w_B_MOA_1 = chain_gemm_einsums(
        [cache["Cocc_A"], w_B, cache["Cvir_A"]],
        ["T", "N", "N"],
    )
    w_A_MOB_1 = chain_gemm_einsums(
        [cache["Cocc_B"], w_A, cache["Cvir_B"]],
        ["T", "N", "N"],
    )

    # Eq. 16: induction potential omega_B in MO basis of A
    w_B_MOA = build_ind_pot(
        {
            "V_B": V_B,
            "J_B": J_B,
            "Cocc_A": cache["Cocc_A"],
            "Cvir_A": cache["Cvir_A"],
        }
    )
    w_B_MOA.name = "w_B_MOA"
    # Eq. 16: induction potential omega_A in MO basis of B
    w_A_MOB = build_ind_pot(
        {
            "V_B": V_A,
            "J_B": J_A,
            "Cocc_A": cache["Cocc_B"],
            "Cvir_A": cache["Cvir_B"],
        }
    )
    w_A_MOB.name = "w_A_MOB"
    assert np.allclose(w_B_MOA, w_B_MOA_1), "w_B_MOA and w_B_MOA_1 do not match!"
    assert np.allclose(w_A_MOB, w_A_MOB_1), "w_A_MOB and w_A_MOB_1 do not match!"

    # Do uncoupled induction calculations
    core.print_out("   => Uncoupled Induction <= \n\n")

    # Create uncoupled response vectors by element-wise division
    unc_x_B_MOA = w_B_MOA.clone()
    unc_x_A_MOB = w_A_MOB.clone()

    eps_occ_A = cache["eps_occ_A"]
    eps_vir_A = cache["eps_vir_A"]
    eps_occ_B = cache["eps_occ_B"]
    eps_vir_B = cache["eps_vir_B"]

    # Eq. 20
    for r in range(unc_x_B_MOA.shape[0]):
        for a in range(unc_x_B_MOA.shape[1]):
            unc_x_B_MOA.np[r, a] /= eps_occ_A.np[r] - eps_vir_A.np[a]

    # Eq. 20
    for r in range(unc_x_A_MOB.shape[0]):
        for a in range(unc_x_A_MOB.shape[1]):
            unc_x_A_MOB.np[r, a] /= eps_occ_B.np[r] - eps_vir_B.np[a]

    # Eq. 14: E^(2)_ind(A<-B) = 2 * x^A . omega_tilde^B
    unc_ind_ab = 2.0 * _dot(unc_x_B_MOA, w_B_MOA)
    unc_ind_ba = 2.0 * _dot(unc_x_A_MOB, w_A_MOB)
    unc_indexch_ab = 2.0 * _dot(unc_x_B_MOA, EX_A_MO)
    unc_indexch_ba = 2.0 * _dot(unc_x_A_MOB, EX_B_MO)

    ret = {}
    ret["Ind20,u (A<-B)"] = unc_ind_ab
    ret["Ind20,u (A->B)"] = unc_ind_ba
    ret["Ind20,u"] = unc_ind_ab + unc_ind_ba
    ret["Exch-Ind20,u (A<-B)"] = unc_indexch_ab
    ret["Exch-Ind20,u (A->B)"] = unc_indexch_ba
    ret["Exch-Ind20,u"] = unc_indexch_ba + unc_indexch_ab

    plist = [
        "Ind20,u (A<-B)",
        "Ind20,u (A->B)",
        "Ind20,u",
        "Exch-Ind20,u (A<-B)",
        "Exch-Ind20,u (A->B)",
        "Exch-Ind20,u",
    ]

    if do_print:
        for name in plist:
            core.print_out(print_sapt_var(name, ret[name], short=True))
            core.print_out("\n")

    # Exch-Ind without S^2 (Sinf calculations)
    if Sinf:
        nocc_A = cache["Cocc_A"].shape[1]
        nocc_B = cache["Cocc_B"].shape[1]
        SAB = core.triplet(
            cache["Cocc_A"], cache["S"], cache["Cocc_B"], True, False, False
        )
        num_occ = nocc_A + nocc_B

        Sab = core.Matrix(num_occ, num_occ)
        Sab.np[:nocc_A, nocc_A:] = SAB.np
        Sab.np[nocc_A:, :nocc_A] = SAB.np.T
        Sab.np[np.diag_indices_from(Sab.np)] += 1
        Sab.power(-1.0, 1.0e-14)

        Tmo_AA = core.Matrix.from_array(Sab.np[:nocc_A, :nocc_A])
        Tmo_BB = core.Matrix.from_array(Sab.np[nocc_A:, nocc_A:])
        Tmo_AB = core.Matrix.from_array(Sab.np[:nocc_A, nocc_A:])

        T_A = core.triplet(cache["Cocc_A"], Tmo_AA, cache["Cocc_A"], False, False, True)
        T_B = core.triplet(cache["Cocc_B"], Tmo_BB, cache["Cocc_B"], False, False, True)
        T_AB = core.triplet(
            cache["Cocc_A"], Tmo_AB, cache["Cocc_B"], False, False, True
        )

        sT_A = core.Matrix.chain_dot(
            cache["Cvir_A"],
            unc_x_B_MOA,
            Tmo_AA,
            cache["Cocc_A"],
            trans=[False, True, False, True],
        )
        sT_B = core.Matrix.chain_dot(
            cache["Cvir_B"],
            unc_x_A_MOB,
            Tmo_BB,
            cache["Cocc_B"],
            trans=[False, True, False, True],
        )
        sT_AB = core.Matrix.chain_dot(
            cache["Cvir_A"],
            unc_x_B_MOA,
            Tmo_AB,
            cache["Cocc_B"],
            trans=[False, True, False, True],
        )
        sT_BA = core.Matrix.chain_dot(
            cache["Cvir_B"],
            unc_x_A_MOB,
            Tmo_AB,
            cache["Cocc_A"],
            trans=[False, True, True, True],
        )

        jk.C_clear()

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_A"], Tmo_AA))
        jk.C_right_add(cache["Cocc_A"])

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_B"], Tmo_BB))
        jk.C_right_add(cache["Cocc_B"])

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_A"], Tmo_AB))
        jk.C_right_add(cache["Cocc_B"])

        jk.compute()

        J_AA_inf, J_BB_inf, J_AB_inf = jk.J()
        K_AA_inf, K_BB_inf, K_AB_inf = jk.K()

        # A <- B
        EX_AA_inf = V_B.clone()
        EX_AA_inf.axpy(
            -1.00, core.Matrix.chain_dot(S, T_AB, V_B, trans=[False, True, False])
        )
        EX_AA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_B, V_B))
        EX_AA_inf.axpy(2.00, J_AB_inf)
        EX_AA_inf.axpy(
            -2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf, trans=[False, True, False])
        )
        EX_AA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AB_inf))
        EX_AA_inf.axpy(2.00, J_BB_inf)
        EX_AA_inf.axpy(
            -2.00, core.Matrix.chain_dot(S, T_AB, J_BB_inf, trans=[False, True, False])
        )
        EX_AA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_BB_inf))
        EX_AA_inf.axpy(-1.00, K_AB_inf.transpose())
        EX_AA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, True, True])
        )
        EX_AA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_B, K_AB_inf, trans=[False, False, True])
        )
        EX_AA_inf.axpy(-1.00, K_BB_inf)
        EX_AA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_BB_inf, trans=[False, True, False])
        )
        EX_AA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_BB_inf))

        EX_AB_inf = V_A.clone()
        EX_AB_inf.axpy(
            -1.00, core.Matrix.chain_dot(S, T_AB, V_A, trans=[False, True, False])
        )
        EX_AB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_B, V_A))
        EX_AB_inf.axpy(2.00, J_AA_inf)
        EX_AB_inf.axpy(
            -2.00, core.Matrix.chain_dot(S, T_AB, J_AA_inf, trans=[False, True, False])
        )
        EX_AB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AA_inf))
        EX_AB_inf.axpy(2.00, J_AB_inf)
        EX_AB_inf.axpy(
            -2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf, trans=[False, True, False])
        )
        EX_AB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AB_inf))
        EX_AB_inf.axpy(-1.00, K_AA_inf)
        EX_AB_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_AA_inf, trans=[False, True, False])
        )
        EX_AB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_AA_inf))
        EX_AB_inf.axpy(-1.00, K_AB_inf)
        EX_AB_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, True, False])
        )
        EX_AB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_AB_inf))

        # B <- A
        EX_BB_inf = V_A.clone()
        EX_BB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_AB, V_A))
        EX_BB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_A, V_A))
        EX_BB_inf.axpy(2.00, J_AB_inf)
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf))
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AB_inf))
        EX_BB_inf.axpy(2.00, J_AA_inf)
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AA_inf))
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AA_inf))
        EX_BB_inf.axpy(-1.00, K_AB_inf)
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf))
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_AB_inf))
        EX_BB_inf.axpy(-1.00, K_AA_inf)
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AA_inf))
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_AA_inf))

        EX_BA_inf = V_B.clone()
        EX_BA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_AB, V_B))
        EX_BA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_A, V_B))
        EX_BA_inf.axpy(2.00, J_BB_inf)
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_BB_inf))
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_BB_inf))
        EX_BA_inf.axpy(2.00, J_AB_inf)
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf))
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AB_inf))
        EX_BA_inf.axpy(-1.00, K_BB_inf)
        EX_BA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_BB_inf))
        EX_BA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_BB_inf))
        EX_BA_inf.axpy(-1.00, K_AB_inf.transpose())
        EX_BA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, False, True])
        )
        EX_BA_inf.axpy(
            1.00, core.Matrix.chain_dot(S, T_A, K_AB_inf, trans=[False, False, True])
        )

        unc_ind_ab_total = 2.0 * (
            sT_A.vector_dot(EX_AA_inf) + sT_AB.vector_dot(EX_AB_inf)
        )
        unc_ind_ba_total = 2.0 * (
            sT_B.vector_dot(EX_BB_inf) + sT_BA.vector_dot(EX_BA_inf)
        )
        unc_indexch_ab_inf = unc_ind_ab_total - unc_ind_ab
        unc_indexch_ba_inf = unc_ind_ba_total - unc_ind_ba

        ret["Exch-Ind20,u (A<-B) (S^inf)"] = unc_indexch_ab_inf
        ret["Exch-Ind20,u (A->B) (S^inf)"] = unc_indexch_ba_inf
        ret["Exch-Ind20,u (S^inf)"] = unc_indexch_ba_inf + unc_indexch_ab_inf

        if do_print:
            for name in plist[3:]:
                name = name + " (S^inf)"

                core.print_out(print_sapt_var(name, ret[name], short=True))
                core.print_out("\n")

    # Do coupled induction calculations
    if do_response:
        core.print_out("\n   => Coupled Induction <= \n\n")

        cphf_r_convergence = core.get_option("SAPT", "CPHF_R_CONVERGENCE")
        x_B_MOA, x_A_MOB = _sapt_cpscf_solve(
            cache,
            jk,
            w_B_MOA.np,
            w_A_MOB.np,
            maxiter,
            cphf_r_convergence,
            sapt_jk_B=sapt_jk_B,
        )
        x_B_MOA = core.Matrix.from_array(x_B_MOA)
        x_A_MOB = core.Matrix.from_array(x_A_MOB)

        ind_ab = 2.0 * _dot(x_B_MOA, w_B_MOA)
        ind_ba = 2.0 * _dot(x_A_MOB, w_A_MOB)
        indexch_ab = 2.0 * _dot(x_B_MOA, EX_A_MO)
        indexch_ba = 2.0 * _dot(x_A_MOB, EX_B_MO)

        ret["Ind20,r (A<-B)"] = ind_ab
        ret["Ind20,r (A->B)"] = ind_ba
        ret["Ind20,r"] = ind_ab + ind_ba
        ret["Exch-Ind20,r (A<-B)"] = indexch_ab
        ret["Exch-Ind20,r (A->B)"] = indexch_ba
        ret["Exch-Ind20,r"] = indexch_ba + indexch_ab

        if do_print:
            core.print_out("\n")
            for name in plist:
                name = name.replace(",u", ",r")
                core.print_out(print_sapt_var(name, ret[name], short=True))
                core.print_out("\n")

        # Exch-Ind without S^2
        if Sinf:
            cT_A = core.Matrix.chain_dot(
                cache["Cvir_A"],
                x_B_MOA,
                Tmo_AA,
                cache["Cocc_A"],
                trans=[False, True, False, True],
            )
            cT_B = core.Matrix.chain_dot(
                cache["Cvir_B"],
                x_A_MOB,
                Tmo_BB,
                cache["Cocc_B"],
                trans=[False, True, False, True],
            )
            cT_AB = core.Matrix.chain_dot(
                cache["Cvir_A"],
                x_B_MOA,
                Tmo_AB,
                cache["Cocc_B"],
                trans=[False, True, False, True],
            )
            cT_BA = core.Matrix.chain_dot(
                cache["Cvir_B"],
                x_A_MOB,
                Tmo_AB,
                cache["Cocc_A"],
                trans=[False, True, True, True],
            )

            ind_ab_total = 2.0 * (
                cT_A.vector_dot(EX_AA_inf) + cT_AB.vector_dot(EX_AB_inf)
            )
            ind_ba_total = 2.0 * (
                cT_B.vector_dot(EX_BB_inf) + cT_BA.vector_dot(EX_BA_inf)
            )
            indexch_ab_inf = ind_ab_total - ind_ab
            indexch_ba_inf = ind_ba_total - ind_ba

            ret["Exch-Ind20,r (A<-B) (S^inf)"] = indexch_ab_inf
            ret["Exch-Ind20,r (A->B) (S^inf)"] = indexch_ba_inf
            ret["Exch-Ind20,r (S^inf)"] = indexch_ba_inf + indexch_ab_inf

            if do_print:
                for name in plist[3:]:
                    name = name.replace(",u", ",r") + " (S^inf)"

                    core.print_out(print_sapt_var(name, ret[name], short=True))
                    core.print_out("\n")

    return ret


def _sapt_cpscf_solve(
    cache: dict,
    jk: core.JK,
    rhsA: np.ndarray,
    rhsB: np.ndarray,
    maxiter: int,
    conv: float,
    sapt_jk_B: core.JK | None = None,
) -> list:
    r"""Solve the coupled-perturbed SCF (CPSCF) equations for SAPT induction.

    Implements the coupled-perturbed Kohn-Sham (CPKS) or Hartree-Fock (CPHF)
    equations using a conjugate-gradient solver. The CPSCF response
    vectors :math:`\mathbf{x}^A` and :math:`\mathbf{x}^B` satisfy (Eq. 21-26
    of Xie et al. 2022):

    .. math::

        \mathbf{H}^{(1)} \mathbf{x}^A = \mathbf{\omega}^{B}

    and
    .. math::

        \mathbf{H}^{(1)} \mathbf{x}^B = \mathbf{\omega}^{A}

    where :math:`\mathbf{H}^{(1)}` includes exchange-correlation
    and exact exchange contributions.

    Parameters
    ----------
    cache : dict
        SAPT data cache containing wavefunctions and orbital energies.
    jk : core.JK
        JK integral engine for monomer A.
    rhsA : np.ndarray
        Right-hand side vector for monomer A response (:math:`-\tilde{\omega}^B`).
    rhsB : np.ndarray
        Right-hand side vector for monomer B response (:math:`-\tilde{\omega}^A`).
    maxiter : int
        Maximum number of CPSCF iterations.
    conv : float
        Convergence threshold (relative residual norm).
    sapt_jk_B : core.JK or None, optional
        Separate JK object for monomer B, by default None (uses same as A).

    Returns
    -------
    list
        Converged response vectors ``[x_A, x_B]`` as numpy arrays.
    """

    cache["wfn_A"].set_jk(jk)
    if sapt_jk_B:
        cache["wfn_B"].set_jk(sapt_jk_B)
    else:
        cache["wfn_B"].set_jk(jk)

    def setup_P_X(eps_occ, eps_vir, name="P_X"):
        # P_X[i, a] = eps_occ[i] - eps_vir[a]: a single einsums outer sum,
        # where v1 needed two tensor contractions against vectors of ones.
        P_X = _ein_zeros(eps_occ.shape[0], eps_vir.shape[0], name=name)
        ein.linalg.outer_sum(
            P_X,
            [_ein(eps_occ, name="eps_occ"), _ein(eps_vir, name="eps_vir")],
            [1.0, -1.0],
        )
        return P_X

    # Make a preconditioner function
    P_A = setup_P_X(cache["eps_occ_A"], cache["eps_vir_A"])
    P_B = setup_P_X(cache["eps_occ_B"], cache["eps_vir_B"])

    # Preconditioner function
    def apply_precon(x_vec, act_mask):
        if act_mask[0]:
            pA = x_vec[0].copy()
            pA /= P_A
        else:
            pA = False

        if act_mask[1]:
            pB = x_vec[1].copy()
            pB /= P_B
        else:
            pB = False
        return [pA, pB]

    # Hx function
    def hessian_vec(x_vec, act_mask):
        # NOTE: to fully convert induction to einsums here, would need to
        # re-write cphf_HX, onel_Hx, and twoel_Hx functions in
        # libscf_solver/uhf.cc
        if act_mask[0]:
            xA = cache["wfn_A"].cphf_Hx([core.Matrix.from_array(x_vec[0])])[0].np
        else:
            xA = False

        if act_mask[1]:
            xB = cache["wfn_B"].cphf_Hx([core.Matrix.from_array(x_vec[1])])[0].np
        else:
            xB = False

        return [xA, xB]

    # Manipulate the printing
    sep_size = 51
    core.print_out("   " + ("-" * sep_size) + "\n")
    core.print_out("   " + "SAPT Coupled Induction Solver".center(sep_size) + "\n")
    core.print_out("   " + ("-" * sep_size) + "\n")
    core.print_out("    Maxiter             = %11d\n" % maxiter)
    core.print_out("    Convergence         = %11.3E\n" % conv)
    core.print_out("   " + ("-" * sep_size) + "\n")

    tstart = time.time()
    core.print_out(
        "     %4s %12s     %12s     %9s\n" % ("Iter", "(A<-B)", "(B->A)", "Time [s]")
    )
    core.print_out("   " + ("-" * sep_size) + "\n")

    start_resid = [_dot(rhsA, rhsA), _dot(rhsB, rhsB)]

    def pfunc(niter, x_vec, r_vec):
        if niter == 0:
            niter = "Guess"
        else:
            niter = "%5d" % niter
        # Compute IndAB
        valA = (_dot(r_vec[0], r_vec[0]) / start_resid[0]) ** 0.5
        if valA < conv:
            cA = "*"
        else:
            cA = " "

        # Compute IndBA
        valB = (_dot(r_vec[1], r_vec[1]) / start_resid[1]) ** 0.5
        if valB < conv:
            cB = "*"
        else:
            cB = " "

        core.print_out(
            "    %5s %15.6e%1s %15.6e%1s %9d\n"
            % (niter, valA, cA, valB, cB, time.time() - tstart)
        )
        return [valA, valB]

    # Compute the solver
    vecs, resid = solvers.cg_solver_ein(
        [rhsA, rhsB],
        hessian_vec,
        apply_precon,
        maxiter=maxiter,
        rcond=conv,
        printlvl=0,
        printer=pfunc,
    )
    core.print_out("   " + ("-" * sep_size) + "\n")

    return vecs
