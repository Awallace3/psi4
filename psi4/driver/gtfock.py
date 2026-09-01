#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2026 The Psi4 Developers.
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
"""Python entry point for Psi4's optional GTFock MPI J/K engine.

GTFock is an MPI-distributed Fock-build engine that Psi4 links only when
configured with ``-DENABLE_GTFock=ON``. It talks exclusively to
``MPI_COMM_WORLD``, so somebody has to call ``MPI_Init`` before Psi4 touches it;
under ``mpirun`` that somebody is mpi4py, imported here.

Nothing in this module is imported at ``import psi4`` time and mpi4py is
imported lazily inside :func:`initialize`, so a default Psi4 build never needs
MPI, mpi4py, or GTFock.

Typical use, from a script launched as ``mpirun -n 2 python script.py``::

    import psi4
    from psi4.driver import gtfock

    info = gtfock.initialize()          # imports mpi4py, starts MPI, cross-checks
    psi4.set_options({"scf_type": "gtfock", "puream": False})
    energy = psi4.energy("scf")         # J/K come from GTFock on every rank
    assert gtfock.fock_builds() > 0     # GTFock really ran

For the density-fitted engine, set ``scf_type`` to ``gtfock_df`` instead and
assert on :func:`df_jk_builds`; :func:`df_partition` then reports how the fitted
tensor was split over the ranks.
"""

__all__ = [
    "GTFockNotAvailable",
    "available",
    "decomposition",
    "df_available",
    "df_jk_builds",
    "df_partition",
    "fock_builds",
    "initialize",
    "mpi_info",
]

from typing import Any, Dict

from psi4 import core


class GTFockNotAvailable(RuntimeError):
    """Raised when the GTFock path is asked for but is not usable in this process."""


def available() -> bool:
    """Whether this Psi4 was compiled with ``-DENABLE_GTFock=ON``."""
    return core.gtfock_enabled()


def fock_builds() -> int:
    """How many GTFock Fock builds this process has run.

    Zero after a calculation that claimed to use GTFock means Psi4 fell back to
    its own integrals, so tests assert on this rather than on timings.
    """
    return core.gtfock_fock_builds()


def df_available() -> bool:
    """Whether this Psi4 links GTFock's *distributed density-fitting* engine.

    That engine (``libgtfockdf``) is a later addition to gtfock_psi4 and is
    optional within ``-DENABLE_GTFock=ON``, so :func:`available` can be true
    while this is false. ``SCF_TYPE GTFOCK_DF`` needs this one.
    """
    return core.gtfock_df_enabled()


def df_jk_builds() -> int:
    """How many distributed density-fitted J/K builds this process has run.

    The DF engine is separate from the exact one, so this counter is separate
    from :func:`fock_builds`; a ``GTFOCK_DF`` calculation moves this one.
    """
    return core.gtfock_df_jk_builds()


def df_partition() -> Dict[str, Any]:
    """How the most recent DF engine split the fitted tensor over this rank.

    ``nlocal_aux`` is the number of auxiliary functions this rank owns: on more
    than one rank these differ between ranks and sum to ``naux``, which is what
    shows the tensor was distributed rather than replicated. ``nmetric_null``
    counts auxiliary functions the fitting condition dropped, ``nlocal_pairs``
    the AO-pair elements this rank computed integrals for before redistribution
    (legitimately zero when there are more ranks than shell pairs), and
    ``local_tensor_doubles`` the size of this rank's slice. Every entry is
    ``-1`` (or ``0``) before any engine is built.
    """
    nbf, naux, nlocal_aux, nmetric_null, nlocal_pairs = core.gtfock_df_partition()
    return {
        "nbf": nbf,
        "naux": naux,
        "nlocal_aux": nlocal_aux,
        "nmetric_null": nmetric_null,
        "nlocal_pairs": nlocal_pairs,
        "local_tensor_doubles": core.gtfock_df_local_tensor_doubles(),
    }


def mpi_info() -> Dict[str, int]:
    """Rank and size of ``MPI_COMM_WORLD`` as the *linked GTFock MPI library*
    sees them, without importing mpi4py. Both are ``-1`` before ``MPI_Init``."""
    return {
        "rank": core.gtfock_world_rank(),
        "size": core.gtfock_world_size(),
        "initialized": core.gtfock_mpi_initialized(),
    }


def decomposition() -> Dict[str, Any]:
    """How the most recent GTFock engine split the AO matrix over this rank.

    ``grid`` is the ``[nprow, npcol]`` process grid, ``block`` the
    ``[start_row, end_row, start_col, end_col]`` AO panel this rank owns, and
    ``task_shape`` the ``[nblks_row, nblks_col, ntasks]`` blocking GTFock chose
    within that panel. Blocks differing across ranks show the build was really
    distributed; ``nblks_row``/``nblks_col`` above one show the panel was large
    enough for GTFock to subdivide. Every entry is ``-1`` before any Fock build.
    """
    return {
        "grid": list(core.gtfock_process_grid()),
        "block": list(core.gtfock_local_block()),
        "task_shape": list(core.gtfock_local_task_shape()),
    }


def initialize() -> Dict[str, Any]:
    """Start MPI through mpi4py and confirm Psi4/GTFock share that MPI runtime.

    Importing :mod:`mpi4py.MPI` calls ``MPI_Init_thread``. Psi4's compiled
    GTFock shim then queries ``MPI_COMM_WORLD`` through its own linked MPI
    library; if Python and Psi4 had ended up bound to different MPI libraries
    the two views would disagree, and this function raises rather than letting a
    calculation proceed on a broken communicator.

    Returns a dict describing the MPI world; raises :class:`GTFockNotAvailable`
    if Psi4 has no GTFock support or mpi4py is missing.
    """
    if not available():
        raise GTFockNotAvailable(
            "This Psi4 was built without GTFock. Reconfigure with -DENABLE_GTFock=ON "
            "and -DGTFock_ROOT=<gtfock_psi4>/_install.")

    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise GTFockNotAvailable(
            "The GTFock path drives MPI from Python and needs mpi4py: `conda install mpi4py`.") from exc

    if not MPI.Is_initialized():  # pragma: no cover - mpi4py initializes on import
        MPI.Init()

    comm = MPI.COMM_WORLD
    py_rank, py_size = comm.Get_rank(), comm.Get_size()
    core_info = mpi_info()

    if not core_info["initialized"]:
        raise GTFockNotAvailable(
            "mpi4py reports MPI is running but Psi4's GTFock shim does not see it. Psi4 and mpi4py are "
            "probably linked against different MPI libraries; build both against the same one.")
    if (core_info["rank"], core_info["size"]) != (py_rank, py_size):
        raise GTFockNotAvailable(
            f"mpi4py sees rank {py_rank}/{py_size} but Psi4's GTFock shim sees "
            f"rank {core_info['rank']}/{core_info['size']}. Psi4 and mpi4py are linked against "
            "different MPI libraries.")

    return {
        "rank": py_rank,
        "size": py_size,
        "thread_level": MPI.Query_thread(),
        "mpi_library": MPI.Get_library_version().strip().splitlines()[0],
        "processor": MPI.Get_processor_name(),
    }
