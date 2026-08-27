.. #
.. # @BEGIN LICENSE
.. #
.. # Psi4: an open-source quantum chemistry software package
.. #
.. # Copyright (c) 2007-2026 The Psi4 Developers.
.. #
.. # The copyrights for code used from other parties are included in
.. # the corresponding files.
.. #
.. # This file is part of Psi4.
.. #
.. # Psi4 is free software; you can redistribute it and/or modify
.. # it under the terms of the GNU Lesser General Public License as published by
.. # the Free Software Foundation, version 3.
.. #
.. # Psi4 is distributed in the hope that it will be useful,
.. # but WITHOUT ANY WARRANTY; without even the implied warranty of
.. # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. # GNU Lesser General Public License for more details.
.. #
.. # You should have received a copy of the GNU Lesser General Public License along
.. # with Psi4; if not, write to the Free Software Foundation, Inc.,
.. # 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
.. #
.. # @END LICENSE
.. #

.. include:: autodoc_abbr_options_c.rst

.. index:: GTFock, MPI, mpi4py, integrals

.. _`sec:gtfock`:

Interface to GTFock by E. Chow *et al.*
=======================================

.. codeauthor:: Edmond Chow, Xing Liu, Hua Huang, Sanchit Misra
.. sectionauthor:: Psi4 Developers

.. image:: https://img.shields.io/badge/home-GTFock-5077AB.svg
   :target: https://github.com/gtfock-chem/gtfock

GTFock is a distributed-memory Fock-build engine: it partitions the AO matrix
across MPI ranks, evaluates two-electron integrals with
:ref:`Simint <sec:simint>`, and accumulates the Coulomb and exchange
contributions into distributed matrices. |PSIfour| can use it as an alternative
:term:`SCF_TYPE <SCF_TYPE (SCF)>`, ``GTFOCK``, in place of its own Libint2-based
J/K builders.

.. warning:: This is a **prototype**. It proves the Python |w---w| MPI |w---w|
   GTFock path end to end; it is not a production distributed SCF driver. The
   restrictions in `Prototype scope`_ are enforced with explicit errors rather
   than left to produce wrong numbers.

GTFock is entirely optional. A default |PSIfour| configure, build, ``import
psi4``, and test run require none of GTFock, Simint, MPI, or mpi4py: a default
``core.so`` links no MPI library and defines no GTFock symbol,
:py:mod:`psi4.driver.gtfock` imports mpi4py lazily inside
:py:func:`psi4.driver.gtfock.initialize` and nowhere else, and the GTFock tests
skip rather than fail. (``import psi4`` will still pick up an installed mpi4py
through the unrelated, and equally optional, MDI interface.)

Prototype scope
~~~~~~~~~~~~~~~

* **One density matrix per Fock build**, i.e. closed-shell RHF and closed-shell
  hybrid DFT. GTFock's GTMatrix-backed engine keeps a single global density
  matrix, so open-shell and multi-density work (UHF, ROHF, SOSCF, response) raise
  rather than run.
* **Hybrid DFT works; range-separated DFT does not.** A global hybrid such as
  ``b3lyp`` or ``pbe0`` needs exactly one J and one K per iteration, which is the
  one shape GTFock answers, and the exchange fraction is applied by
  ``RHF::form_G`` (:math:`G = J - \alpha K + V_{xc}`) after the J/K engine has
  returned |w---w| so it is engine-independent and needs nothing from GTFock.
  |PSIfours| exchange-correlation quadrature is unchanged and is *not*
  distributed; it is recomputed redundantly on every rank.
* **Cartesian basis sets only** (``puream false``). GTFock's Simint driver fills
  Cartesian shell blocks while GTFock sizes a basis it labels spherical as
  :math:`2l+1` per shell. The counts diverge above ``l = 1``, and even at
  ``l = 1`` the orderings differ |w---w| Simint lays a ``p`` shell out as
  ``px, py, pz`` while |PSIfour| orders pure shells by :math:`m` |w---w| so a
  spherical basis would give a permuted J/K. Any spherical basis raises,
  including an ``s``/``p``-only one such as the default spherical ``sto-3g``.
* **Maximum angular momentum bounded by the linked Simint.** libcint indexes
  GTFock's shell-pair work lists as :math:`l_P (l_{max} + 1) + l_Q` into a table
  sized for the maximum angular momentum the linked Simint was generated for,
  without checking the bound, so a higher shell would corrupt memory rather than
  fail. |PSIfour| refuses any shell above the ceiling, naming the offending
  shell and the ceiling itself. GTFock must therefore be built against a Simint
  generated for at least the angular momentum in use; ``gtfock_psi4``'s pinned
  build supplies :math:`l_{max} = 4` (through ``g`` functions), which matches
  the value libcint compiles against, so a basis such as Cartesian ``cc-pV5Z``,
  which carries ``h`` functions, raises.
* **No range-separated exchange.** ``wK`` is unavailable from GTFock, so a
  range-separated functional (``wb97x``, ``cam-b3lyp``, ...) raises. The refusal
  lands in |PSIfours| superfunctional builder, before a JK object is constructed,
  and names both the functional class and ``SCF_TYPE = GTFOCK``;
  ``GTFockJK::compute_JK`` refuses ``do_wK`` again for callers that reach ``JK``
  directly without a superfunctional. Neither path can return a hybrid-shaped
  energy with the long-range exchange silently dropped.
* **One engine per process.** GTFock caches the basis, the Simint handle, and
  its screening and blocking buffers in global state that it fills once and
  never refreshes, so |PSIfour| builds a single GTFock engine and reuses it.
  Asking for a second engine with a different molecule, basis, screening
  tolerance, or task shape in the same process raises; run that case in a fresh
  process.
* **Screening is density-weighted.** ``INTS_TOLERANCE`` (equivalently
  ``jk.set_cutoff()``) is handed to GTFock verbatim as its ``tolscr``, which is
  the right mapping: GTFock and |PSIfour| share one convention, each storing a
  shell pair's largest diagonal integral :math:`(MN|MN)` without taking its
  square root, then testing the product of two of those against the squared
  threshold. GTFock does, however, fold the largest relevant density element
  into that product, which |PSIfours| default Schwarz/CSAM screening does not,
  so at the same ``INTS_TOLERANCE`` GTFock screens somewhat more aggressively.
* **Integral agreement with the built-in engines is not uniform.** It is also
  not controlled by ``INTS_TOLERANCE``. For a single compact molecule the two
  engines agree to roundoff |w---w| measured around :math:`10^{-14}\ E_h` for one
  water in ``sto-3g``, ``6-31G`` or ``6-31G*``. For a six-water cluster in the
  same bases the gap grows to a few times :math:`10^{-6}\ E_h`. Sweeping
  ``INTS_TOLERANCE`` from ``1e-8`` to ``1e-16`` moves the cluster energy by less
  than :math:`10^{-7}\ E_h`, so this is not GTFock's shell-quartet screening.
  It comes from below GTFock: its libcint layer announces its own settings at
  engine creation (``Screen method: 2``, ``Screen tol: 1.000000e-14``) and prunes
  Simint *primitive* pairs at that fixed tolerance, which |PSIfours| threshold
  does not reach. The truncation is therefore per primitive pair and accumulates
  with the number of well-separated centres, which is why it is invisible on one
  water and visible on six. It is an accuracy property of the GTFock stack,
  present at a single rank, and independent of the rank count |w---w| the tests
  hold the cross-engine comparison and the rank-count comparison to separate,
  separately stated tolerances for exactly this reason (see `Testing`_). Anyone
  wanting |PSIfour|-grade agreement on an extended system needs a GTFock built
  with a tighter primitive screen; that tolerance is not currently exposed.
* **The Fock build is distributed; the SCF is not.** J and K are gathered on
  rank 0 and broadcast, so every rank holds the full matrices and then runs an
  identical replicated SCF: diagonalization, DIIS, and the DFT quadrature are
  duplicated work, and peak memory per rank is that of a serial run rather than
  :math:`1/N` of it. Only the two-electron integral work is divided. That is
  also why the energy is rank-count invariant (see `Testing`_), and why the
  measured speedup stays below the rank count (see `Measured rank scaling`_)
  |w---w| the replicated remainder is Amdahl's serial fraction, and it grows as a
  share of the run every time a rank is added. Distributing the SCF itself, so
  that each rank keeps only its own AO panel, is later work.

Installation
~~~~~~~~~~~~

**Source only.** GTFock has no conda package. Build it from the ``gtfock_psi4``
superproject, which pins GTFock, GTMatrix, libcint, OptErd, and the Simint
generator, and builds them with conda-forge IntelLLVM (``icx``/``icpx``) plus
OpenMPI.

.. note:: Build |PSIfour| with the **same OpenMP runtime** GTFock uses.
   conda-forge's ``icx`` links ``libomp``; if |PSIfour| were built with GCC it
   would link ``libgomp``, and GTFock's ``omp_get_thread_num()`` could bind to
   the wrong runtime and corrupt its per-thread integral buffers. Building
   |PSIfour| with the same IntelLLVM compilers, as below, keeps one OpenMP
   runtime in the process.

.. code-block:: bash

    # (1) environment: Psi4's build/run dependencies plus gtfock_psi4's build
    #     contract (IntelLLVM, OpenMPI, MKL) plus mpi4py.
    >>> conda env create -n p4gtf -f devtools/conda-envs/linux-64-gtfock.yaml --solver libmamba
    >>> conda activate p4gtf

    # (2) build GTFock. GTF_COMBINED_JK=OFF is required: with the default ON,
    #     GTFock folds exchange into its Fock matrix and Psi4's JK object can
    #     never recover K on its own.
    >>> git clone --recursive https://github.com/Awallace3/gtfock_psi4.git
    >>> cd gtfock_psi4
    >>> CMAKE_BUILD_PARALLEL_LEVEL=12 ./build_deps.sh --clean
    >>> cmake -S . -B _build/gtfock -DGTF_COMBINED_JK=OFF
    >>> cmake --build _build/gtfock --parallel 12
    >>> cmake --install _build/gtfock          # populates ./_install
    >>> cd ..

    # (3) configure and build Psi4 against that install
    >>> conda/psi4-path-advisor.py cache --compiler IntelLLVM --lapack mkl --objdir objdir_gtfock
    >>> # comment out the CMAKE_Fortran_COMPILER line in the generated cache file:
    >>> #   Psi4's superbuild forwards it verbatim, and the conda ifx wrapper is a
    >>> #   multi-argument list. No Fortran add-on is enabled here, so it is unused.
    >>> cmake -S. -GNinja -C"cache_p4gtf@gtfock.cmake" -Bobjdir_gtfock \
    ...       -DENABLE_GTFock=ON \
    ...       -DGTFock_ROOT=/path/to/gtfock_psi4/_install \
    ...       -DCMAKE_PREFIX_PATH="/path/to/gtfock_psi4/_install;${CONDA_PREFIX}" \
    ...       -DLAPACK_LIBRARIES="${CONDA_PREFIX}/lib/libmkl_intel_lp64.so;${CONDA_PREFIX}/lib/libmkl_intel_thread.so;${CONDA_PREFIX}/lib/libmkl_core.so" \
    ...       -DLAPACK_INCLUDE_DIRS="${CONDA_PREFIX}/include" \
    ...       -DCMAKE_INSTALL_PREFIX=/path/to/install-psi4-gtfock
    >>> cmake --build objdir_gtfock -j 12

``-DGTFock_ROOT`` alone is enough to resolve GTFock: libcint, GTMatrix, and the
Simint that GTFock was built against are all found as plain libraries under that
same prefix, alongside ``libgtfock``. Simint is deliberately *not* located
through its CMake package config, so enabling GTFock never also switches on
|PSIfours| own :ref:`Simint <sec:simint>` ERI engine, and it is searched *only*
inside the GTFock prefix: the angular-momentum ceiling above is read out of
GTFock's own ``CInt.h``, so any other ``libsimint`` on the system would leave
that guard describing a library the build does not link. The GTFock install must
therefore carry its own ``libsimint``; configure fails if it does not, however
many Simints the surrounding environment provides. Naming the GTFock prefix in
``-DCMAKE_PREFIX_PATH`` as well, as above, is harmless; the ``${CONDA_PREFIX}``
entry there is what the rest of the build environment needs.

``-DLAPACK_LIBRARIES`` must name MKL's *layered* libraries rather than letting
|PSIfour| pick ``libmkl_rt``. GTFock links MKL's BLACS and ScaLAPACK, whose
internal symbols (``mkl_serv_verbose_mode`` and friends) live in
``libmkl_core`` and are not exported by the ``libmkl_rt`` dispatcher. A plain
executable gets away with that through lazy binding, but Python dlopens
``core.so`` with ``RTLD_NOW``, so every symbol must resolve at ``import psi4``.

Running
~~~~~~~

GTFock only knows ``MPI_COMM_WORLD``, so MPI has to be up before |PSIfour|
touches it. :py:mod:`psi4.driver.gtfock` does that by importing mpi4py, and then
cross-checks that mpi4py and |PSIfours| linked MPI agree on rank and size |w---w|
if the two were bound to different MPI libraries, it raises instead of running
on a broken communicator.

.. code-block:: python

    # scf_gtfock.py
    import psi4
    from psi4.driver import gtfock

    info = gtfock.initialize()           # imports mpi4py; MPI_Init happens here
    psi4.core.set_output_file(f"out.rank{info['rank']}", False)

    psi4.geometry("""
    O
    H 1 0.96
    H 1 0.96 2 104.5
    """)
    psi4.set_options({"basis": "sto-3g", "puream": False, "scf_type": "gtfock"})
    energy = psi4.energy("scf")                 # or psi4.energy("b3lyp")

    assert psi4.core.gtfock_fock_builds() > 0   # GTFock really ran
    print(gtfock.decomposition())               # how the AO matrix was split

.. code-block:: bash

    >>> eval $(objdir_gtfock/stage/bin/psi4 --psiapi)
    >>> OMP_NUM_THREADS=1 mpirun -n 2 python scf_gtfock.py

Every rank runs the whole |PSIfour| driver and calls the same collective GTFock
routines; GTFock partitions the J/K work and each rank ends up with identical
matrices. Keep ``OMP_NUM_THREADS`` modest when oversubscribing a test machine.

Introspection helpers, all usable without mpi4py:

* :py:func:`psi4.core.gtfock_enabled` |w---w| was |PSIfour| compiled with GTFock
* :py:func:`psi4.core.gtfock_fock_builds` |w---w| GTFock Fock builds this process
  ran; ``0`` after a calculation means it fell back to |PSIfours| own integrals
* :py:func:`psi4.core.gtfock_world_rank` / ``gtfock_world_size`` |w---w|
  ``MPI_COMM_WORLD`` as |PSIfours| linked MPI sees it
* :py:func:`psi4.core.gtfock_process_grid` / ``gtfock_local_block`` /
  ``gtfock_local_task_shape`` |w---w| the process grid, the AO block GTFock gave
  this rank, and the ``[nblks_row, nblks_col, ntasks]`` blocking GTFock chose
  inside that block. :py:func:`psi4.driver.gtfock.decomposition` returns all
  three as a dict. Blocks that differ across ranks show the build really was
  distributed; block counts above one show the rank's panel was large enough for
  GTFock to subdivide.

Testing
~~~~~~~

:source:`tests/pytests/test_gtfock.py` covers the opt-in path and skips cleanly
when GTFock is absent, except for the two optionality guards, which always run:

.. code-block:: bash

    >>> pytest -v tests/pytests/test_gtfock.py

Without GTFock this is ``2 passed, 16 skipped``; the two that run assert that
:py:mod:`psi4.driver.gtfock` imports, reports itself unavailable, raises a
GTFock-specific error rather than a stray ``ImportError``, and needs no mpi4py to
do any of it.

The multi-rank cases launch :source:`tests/pytests/gtfock_mpi_driver.py` under
``mpirun`` and assert per-rank evidence: mpi4py and |PSIfours| MPI agree, each
rank owns a distinct AO block, ``jk.name()`` is ``GTFockJK``, GTFock's Fock build
counter advanced, and the energy agrees with |PSIfours| own PK result.

* ``test_gtfock_multirank_mpirun`` is the smoke case |w---w| STO-3G water RHF on
  two ranks, cheap enough to leave in an ordinary test run.

* ``test_gtfock_rank_count_invariance`` is the correctness case. It runs ``scf``
  and ``b3lyp`` on a water hexamer in Cartesian 6-31G* (60 shells, 114 basis
  functions) at one, two and four ranks. Four ranks give a 2x2 process grid, four
  distinct 55x55 AO panels, and a 4x4 task blocking *inside* each panel; the test
  asserts all three, because a system small enough to hand each rank a single
  block would say nothing about the decomposition.

  It then makes two separate comparisons at two separate, separately stated
  tolerances, because they are two different claims:

  - **GTFock against GTFock across rank counts**, at 1.0e-9 :math:`E_h`. This is
    the distributed-correctness claim, and it is the tight one. Measured spread
    over all ranks of all three rank counts is 4.5e-13 :math:`E_h` for RHF and
    9.1e-13 :math:`E_h` for B3LYP |w---w| reordered-summation noise, three orders
    below the tolerance. The iteration count is identical at every rank count.
  - **GTFock against a single-process PK reference** computed by |PSIfour|
    itself, at 1.0e-5
    :math:`E_h`. Measured disagreement is 2.4e-6 :math:`E_h` for RHF and 3.5e-6
    :math:`E_h` for B3LYP, identical at one, two and four ranks. This is the
    primitive-screening difference described under `Prototype scope`_, not a
    distribution artifact: it is fully present at a single rank, and it does not
    move when ranks are added. The loose tolerance is here so the test pins that
    behaviour instead of hiding it; a compact single molecule is still held to
    1.0e-9 :math:`E_h` against PK, and ``test_gtfock_multirank_mpirun`` and the
    in-process cases do exactly that.

  The reference comes from |PSIfours| own integrals rather than from a one-rank
  GTFock run, so agreement between rank counts cannot be agreement on a shared
  error.

* ``test_gtfock_hybrid_dft_energy_matches_reference`` pins B3LYP against PK in
  process. ``test_gtfock_refuses_range_separated_functionals`` and
  ``test_gtfock_refuses_wk_directly`` pin the range-separated refusal at both
  places it can be reached, the superfunctional builder and ``compute_JK``, so
  that a range-separated request stays an error rather than becoming a wrong
  number.

DFT is swept over rank counts alongside RHF even though the engine code is the
same for both, because what DFT changes is the caller: the exchange scaling and
the ``do_wK`` request are decided in |PSIfours| SCF, not in GTFock.

Measured rank scaling
~~~~~~~~~~~~~~~~~~~~~

:source:`tests/pytests/gtfock_benchmark.py` is a script rather than a test,
because a useful case takes minutes. It launches the same driver at each rank
count and prints a table:

.. code-block:: bash

    >>> python tests/pytests/gtfock_benchmark.py --ranks 1,2,4 \
    ...     --molecule water6 --basis cc-pVTZ --method scf

The numbers below are one such run: the water hexamer in Cartesian cc-pVTZ, 390
basis functions in 132 shells, on one 24-core AMD Ryzen Threadripper 3960X, with
a single OpenMP thread per rank so that the rank count is the only variable and
no rank count oversubscribes a core. ``wall (s)`` is the slowest rank's SCF,
excluding |PSIfour| startup and the one-shot GTFock/Simint setup. ``dE`` is
against the same method's one-rank energy.

======  =====  ====  ======  =====  ========  =======  =======
method  ranks  grid  blocks  iters  wall (s)  speedup  dE (Eh)
======  =====  ====  ======  =====  ========  =======  =======
RHF     1      1x1   4x4     16     227.6     1.00     ---
RHF     2      1x2   4x4     16     124.4     1.83     2.3e-13
RHF     4      2x2   4x4     16     65.2      3.49     2.3e-13
B3LYP   1      1x1   4x4     17     299.9     1.00     ---
B3LYP   2      1x2   4x4     17     191.0     1.57     1.5e-12
B3LYP   4      2x2   4x4     17     128.3     2.34     1.1e-13
======  =====  ====  ======  =====  ========  =======  =======

Both methods land on the same energy at every rank count: the largest spread is
1.5e-12 :math:`E_h`, which is reordered-summation noise, three orders below the
1.0e-9 :math:`E_h` the tests demand. The iteration counts are identical across
rank counts too, so the runs really are the same SCF and not merely two SCFs that
happened to converge to the same place.

The speedup is real but sublinear, and it is worse for B3LYP than for RHF |w---w|
3.5x against 2.3x at four ranks. That gap is the replicated remainder described
under `Prototype scope`_: GTFock divides the two-electron work, while the
diagonalization, DIIS, and |w---w| for DFT |w---w| the exchange-correlation
quadrature are recomputed in full on every rank, and B3LYP has more of that
replicated work per iteration than RHF does. Adding ranks shrinks only the part
being divided, so parallel efficiency falls from 91% at two ranks to 87% at four
for RHF, and from 79% to 58% for B3LYP. The ceiling here is the serial fraction,
not the integral engine; raising it means distributing the SCF, not tuning the
J/K.

These are measurements from one node of one machine at one problem size, reported
as measured and not extrapolated. The script is in the tree so they can be
re-measured rather than taken on faith.

.. _`cmake:gtfock`:

How to configure GTFock for building Psi4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Role and Dependencies**

* Role |w---w| In |PSIfour|, GTFock provides an MPI-distributed J/K builder
  selected by :term:`SCF_TYPE <SCF_TYPE (SCF)>` ``GTFOCK``.

* Downstream Dependencies |w---w| |PSIfour| (\ |dr| optional) GTFock

* Upstream Dependencies |w---w| GTFock |dr| Simint, MPI, OpenMP; and mpi4py at
  runtime

**CMake Variables**

* :makevar:`ENABLE_GTFock` |w---w| CMake variable toggling whether |PSIfour|
  builds with GTFock. Default ``OFF``.
* :makevar:`GTFock_ROOT` |w---w| CMake variable to specify where the pre-built
  GTFock can be found. Set to the installation directory containing
  ``include/pfock.h`` and ``lib/libgtfock.so``. libcint, GTMatrix, and Simint
  are located under the same prefix, and Simint is looked for *only* there.
* :makevar:`CMAKE_PREFIX_PATH` |w---w| an alternative to :makevar:`GTFock_ROOT`;
  add the same prefix to it.

There is no internal build: GTFock has no conda package and no CMake package
config, so |PSIfour| only ever links a GTFock you built yourself.

**Examples**

A. Build *without* GTFock (default)

  .. code-block:: bash

    >>> cmake

B. Link against pre-built

  .. code-block:: bash

    >>> cmake -DENABLE_GTFock=ON -DGTFock_ROOT=/path/to/gtfock_psi4/_install \
    ...       -DCMAKE_PREFIX_PATH=/path/to/gtfock_psi4/_install
