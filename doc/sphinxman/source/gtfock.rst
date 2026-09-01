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

``gtfock_psi4`` also carries a second, separate engine: a distributed
density-fitted J/K built on the same Simint driver, exposed as
:term:`SCF_TYPE <SCF_TYPE (SCF)>` ``GTFOCK_DF``. It distributes the fitted
three-index tensor over ranks by auxiliary function rather than the AO matrix by
block, it lifts three of the restrictions below, and it is optional *within*
``ENABLE_GTFock`` |w---w| a GTFock install predating ``libgtfockdf`` supplies
``GTFOCK`` and not ``GTFOCK_DF``. See `Distributed density fitting`_. Everything
between here and that section describes the exact four-centre path.

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
* **Diffuse functions on a large system are not supported.** On a 157-atom
  fragment pair the SCF converges through GTFock in Cartesian ``6-31G**``
  (1555 basis functions) and does not converge in Cartesian ``6-31+G**``
  (1863), running the full 100 iterations. The failure is the engine's and
  not the system's: |PSIfours| own ``DirectJK`` and its density fitting each
  converged the identical 1863-function input in 12 monotone iterations from
  the same build on the same node type. Distribution is excluded by
  measurement |w---w| every diagnostic quantity is identical at 1, 2 and 4
  ranks and at 1, 4, 8 and 24 threads within a rank |w---w| and so is any
  suggestion that GTFock is simply inaccurate on this basis: on a physical
  density for this molecule in ``6-31+G**``, GTFock's J agrees with
  ``DirectJK`` to 3e-06 relative with a purely scattered residual, which is
  the same signature |PSIfours| own screening options give.

  The mechanism is a discrete occupation switch in the SCF's very first
  diagonalization. The 1863-function overlap matrix is nearly singular
  |w---w| smallest eigenvalue 1.09e-07, and 98 of 1863 directions below
  1e-03 |w---w| so symmetric orthogonalization divides by
  :math:`\sqrt{s}` and amplifies any difference in the Fock build by up to
  9e+06. Both engines were made to build F on the bit-identical SAD guess
  density and the two matrices diagonalized side by side. They differ by
  7e-05 relative, and that difference is not even aimed at the near-singular
  block |w---w| it is *depleted* there by a factor of two against an
  even-spread baseline. It does not need to be aimed. Aufbau fills the
  lowest 290 eigenvectors whatever they are made of, and 14 near-singular
  states are driven down past the physical ones, arriving deep at
  :math:`-7.8` to :math:`-1.3\ E_h` rather than jostling at the Fermi
  level. ``DirectJK``'s occupied space contains no such state at all; its
  worst orbital sits at an effective overlap eigenvalue of 1.6e-02.
  The principal angles between the two occupied subspaces are decisive: 276
  of the 290 orbitals agree to 0.03 degrees and 14 are rotated past 45
  degrees, the largest at 89.998. Nothing is bent; fourteen states are
  replaced. Fourteen doubly-occupied spurious states at that depth is also
  the right size for the 57 :math:`E_h` by which the first iterate
  undershoots. There is no smooth-response account available, and none is
  needed: ``DirectJK``'s guess Fock has a 0.22 :math:`E_h` HOMO-LUMO gap, so
  first-order theory cannot move an orbital appreciably for a 7e-05
  perturbation, and 276 of them indeed do not move.

  A control settles which quantity is the operative one, because size is not
  it. Density fitting's guess Fock differs from ``DirectJK``'s by 3.7e-05
  relative, half of GTFock's 7.4e-05 and so the same size to within a factor of
  two, yet DF converges this input in 12 iterations. What differs is where the
  two differences land. Of its squared norm, DF puts 2e-08 into the 22
  directions below :math:`s = 10^{-5}` where GTFock puts 1e-02, a ratio of
  5e+05, so symmetric orthogonalization magnifies DF's difference threefold and
  GTFock's fifteen-hundred-fold. DF's first diagonalization is consequently
  ``DirectJK``'s to five figures: maximum principal angle 0.003 degrees, not
  one orbital rotated past 1 degree, and no occupied orbital anywhere near the
  singular region. Note that DF is by far the cruder engine here |w---w| its
  energy at the guess density is wrong by 3e-03 :math:`E_h`, some 2700 times
  GTFock's 1e-06 |w---w| and it is entirely safe. How large the error is does
  not matter; how much of it survives orthogonalization does.

  Two further controls put a floor under those numbers and a bound over them.
  ``DirectJK`` run against itself at a different thread count yields a
  difference containing no engine information whatever, yet by both obvious
  placement statistics it looks worse than GTFock: symmetric orthogonalization
  amplifies it 3e+05-fold against GTFock's 1.5e+03, and it is *enriched* in the
  near-singular directions 2.5-fold where GTFock is depleted. It changes the
  identity of no orbital at all. GTFock's own threading roundoff is more
  extreme still on both counts and likewise changes none, while reproducing the
  20-thread first diagonalization to every digit printed |w---w| the same
  collapsed gap, the same fourteen orbitals, the same 89.998 degrees |w---w| so
  the failure is deterministic and is not a race. Neither statistic can serve
  as a screen; both rank the failing engine third of four.

  What discriminates, and admits a bound, is the absolute size of the
  difference in the orthogonalized basis. Writing :math:`E = X^{T}(\Delta F)X`
  and :math:`\delta` for the reference HOMO-LUMO gap, the Davis-Kahan sine
  theorem bounds the rotation of the occupied space by
  :math:`\|E\|_2/(\delta-\|E\|_2)` whenever :math:`\|E\|_2<\delta`. That ratio
  is 1.6e-06 and 1.4e-05 for the two threading controls and 1.1e-02 for DF, so
  all three are certified against any change of occupation and all three are
  measured at none. For GTFock it is 38: the guarantee lapses, and fourteen
  orbitals change. Measured for GTFock in ``6-31G**``, where this molecule
  converges through every engine, it is 0.27, certified, and no orbital
  changes. Note the margin there: a factor of four below the gap, not orders
  of magnitude. The certificate runs one way only |w---w| exceeding the gap
  proves nothing |w---w| but it does put a number on the failure, and it says
  in one line why the remedy below works. Discarding the near-singular
  directions discards the directions :math:`\|E\|_2` is built from.

  Diffuse shells are not what creates that difference; they are what makes it
  matter. Repeating the guess-density comparison on the same molecule in
  ``6-31G**`` gives 3.0e-05 relative against the 7.4e-05 above, a factor of
  2.4, and GTFock's error in the guess energy grows by 8, from 1.4e-07 to
  1.2e-06 :math:`E_h` |w---w| while the amplification available to act on it
  grows by 3048 and the screen above by 144. Two to eight against three
  thousand is not a close call. The accuracy of the working case is likewise
  unremarkable: in ``6-31G**`` GTFock converges to -3642.668784 :math:`E_h`
  against ``DirectJK``'s -3642.668730, both in 12 iterations, and density
  fitting sits 93 times further from ``DirectJK`` than that and converges as
  well. What fails in the larger basis is the conditioning, which GTFock's
  error is merely large enough to trigger.

  What that leaves open is the origin of the 7e-05 difference itself, now
  known to be generic to the GTFock stack rather than provoked by diffuse
  functions, since it is present at full strength in a basis that has none.
  The hardcoded 1e-14 primitive-pair screen described in the bullet above is
  the standing suspect and has not been shown to be the cause.

  There is a measured way through, on this system. Because the switch is a
  property of the orthogonalizer, canonical orthogonalization removes it once
  ``S_TOLERANCE`` is high enough to discard the directions the spurious states
  are built from. Rediagonalizing the saved guess Fock matrices at each rung,
  the number of orbitals that change identity falls monotonically |w---w| 14,
  9, 1, 0, 0 for the symmetric case and for ``S_TOLERANCE`` 1e-05, 1e-04, 1e-03
  and 1e-02 |w---w| and that count predicts every SCF that has been run.
  Symmetric, 1e-05 and 1e-04 all fail, ending 57, 27 and 0.8 :math:`E_h` below
  the reference; one orbital that changes identity is evidently enough, so do
  not read the shrinking energy error as partial success. Density fitting at
  that same 1e-04 rung converges normally, which places that failure on the
  engine rather than on the truncation.

  ``S_TOLERANCE 1e-3``, the first rung at which the count reaches zero, does
  converge: 10 iterations from the SAD guess to -3642.742855 :math:`E_h`, which
  is 0.0099 :math:`E_h` ABOVE the full-space reference and so on the side a
  variational restriction has to be on. It was named in the diagnostics file as
  the rung that should work before the run started. That 0.0099 :math:`E_h` is
  the price of dropping 98 directions rather than GTFock's own residual error.
  Running ``DirectJK`` at the same rung settles that by subtraction alone: it
  converges to -3642.742429 :math:`E_h`, which puts the cost of dropping the 98
  directions at 0.0103 :math:`E_h` and leaves GTFock 0.0004 :math:`E_h` BELOW
  the correctly truncated answer rather than above it. Density fitting's own
  ladder independently agrees on 0.0103 :math:`E_h`, as a difference of two of
  its own energies in which its fitting error cancels. Do not read density
  fitting's raw figure at the rung instead: it converges the same restricted
  problem 0.0048 :math:`E_h` worse than GTFock, because it also sits 0.0043
  :math:`E_h` above the reference in the full space. Comparing two engines at a
  restricted rung needs both of them at the unrestricted one as well.

  Treat this as a workaround for this system rather than a general
  prescription. The rung was chosen because a diagnostic count reached zero on
  this molecule in this basis; that count is cheap to compute and is the
  transferable part, the number is not. Nudging ``S_TOLERANCE`` to 1e-6 in
  particular rescues nothing: it discards 8 of 1863 directions and leaves the
  switch intact.

  Worth knowing when reading a log: this system misses |PSIfours| own automatic
  protection by 9%. ``S_TOLERANCE`` defaults to 1e-7 and the smallest overlap
  eigenvalue here is 1.0895e-07, so the SCF prints ``Using symmetric
  orthogonalization`` and eliminates no MOs. Had the geometry been slightly
  different, canonical orthogonalization would have engaged on its own.

  So: on the default settings, treat a diffuse-augmented basis on a system of
  this size as unsupported rather than merely slow, and check any such result
  against a non-GTFock ``SCF_TYPE``. Raising ``S_TOLERANCE`` to 1e-3 does
  converge this particular case, as described above, but it was tuned on this
  one system and it changes the answer by 0.01 :math:`E_h` for reasons not yet
  fully attributed, so it is a workaround to verify rather than a setting to
  adopt blindly.

  The probes, their scripts and their verbatim output are in
  :source:`tests/pytests/gtfock_diffuse_j_diagnostics.txt`; that file also
  records two earlier attributions of this failure that later controls
  withdrew |w---w| a 0.31% deficit in GTFock's J, and a claim that
  ``DirectJK`` falls into the same near-singular subspace and merely lands
  shallower |w---w| together with why each was wrong.
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

Distributed density fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SCF_TYPE GTFOCK_DF`` is a second engine, sharing nothing with the one above but
the Simint it computes integrals with. Rather than partitioning the AO matrix and
evaluating four-centre quartets, it builds the fitted three-index tensor

.. math:: B_Q^{mn} = \sum_P \left[\mathbf{J}^{-1/2}\right]_{PQ} (P|mn)

once at setup, distributes it over ranks **by auxiliary function** :math:`Q`, and
then contracts it against a density each iteration:
:math:`c_Q = \sum_{rs} B_Q^{rs} D_{rs}`, :math:`J_{mn} = \sum_Q B_Q^{mn} c_Q`, and
:math:`K_{mn} = \sum_Q \sum_i B_Q^{mi} B_Q^{ni}`. :math:`Q` is the distribution
axis because K needs every AO pair of a given :math:`Q` on one rank; the AO pairs,
which the three-centre integral phase is itself partitioned over, are
redistributed once at setup and never again.

Each build costs one ``MPI_Allreduce`` of :math:`2 n_{bf}^2` doubles, so J and K
come back fully replicated, exactly as from the exact path. What is distributed is
the tensor, and that is the quantity worth distributing: it is
:math:`\mathcal{O}(n_{aux} n_{bf}^2)`, each rank holds only its own slice, and it
is the one thing here that does fall as :math:`1/N`.

The |PSIfour| side is ``GTFockDFJK``
(:source:`psi4/src/psi4/libfock/GTFockDFJK.cc`) over an RAII shim
(:source:`psi4/src/psi4/libfock/gtfock_df_interface.cc`) around ``gtfock_psi4``'s
``PDF_t``. It takes the fitting basis from ``DF_BASIS_SCF``, the Coulomb metric's
relative eigenvalue cutoff from ``DF_FITTING_CONDITION``, and its thread count
from ``DF_INTS_NUM_THREADS``.

What it lifts
.............

* **Open-shell works.** ``PDF_computeJK`` takes one density and one occupied
  block per call and keeps no per-density state, so ``GTFockDFJK`` loops over the
  densities ``JK`` handed it. UHF runs; the exact path's single global density
  matrix is not in the way.
* **The engine is re-entrant.** ``PDF_t`` keeps no file-scope state, unlike
  ``fock_task.c``, so the engine is an ordinary member rather than a
  process-lifetime singleton: ``GTFockDFJK`` builds it in ``preiterations()`` and
  releases it in ``postiterations()``, several may be alive at once over different
  basis pairs, and a released one may be rebuilt. There is no "one engine per
  process" rule and no need to run a second case in a fresh process. Note that
  ``HF::finalize()`` does not call ``JK::finalize()``, so in an ordinary SCF the
  engine is released when the ``JK`` object is, not at the end of the iterations.
* **The angular-momentum ceiling is Simint's own.** The DF driver calls Simint
  directly instead of going through libcint's shell-pair work lists, so its bound
  is ``SIMINT_OSTEI_MAXAM`` read out of Simint |w---w| :math:`l_{max} = 5`, through
  ``h``, in ``gtfock_psi4``'s pinned build, one higher than libcint's separately
  hardcoded 4. Cartesian ``cc-pV5Z`` therefore runs here and raises on the exact
  path. |PSIfour| screens against ``GTFDF_maxSupportedAM()`` rather than a copied
  constant, so the two cannot drift apart.

What it keeps
.............

* **Cartesian basis sets only, and now that is two basis sets.** The orbital and
  the fitting basis are screened separately and each raises on its own. ``puream
  false`` in the input covers both.
* **No range-separated exchange.** ``do_wK`` raises in ``compute_JK``.
* **Symmetric densities only.** ``C_left != C_right`` raises rather than returning
  a K built from the wrong pair.
* **The Fock build is distributed; the SCF is not.** Unchanged from the exact
  path: diagonalization, DIIS and any DFT quadrature stay replicated on every
  rank, and J and K are replicated after the reduction.

One caveat from `What the J/K timer does not cover`_ applies here with full force,
because this engine is a density-fitting builder: the tensor is built in
``preiterations()``, which ``JK::initialize()`` runs *before*
``timer_on("JK: JK")`` is ever opened. The ``JK: JK`` line in a ``GTFOCK_DF``
output therefore reports the per-iteration contractions only and omits the setup
that dominates a short run.

Unlike ``MemDFJK``, which is legible through ``DFHelper``'s own timers, the
GTFock DF engine is C and brackets nothing, so a tool that summed the known
density-fitting setup timers would report a confident zero for this builder
|w---w| a measured absence, which is the one wrong answer instrumentation can
give. ``GTFockDFJK::preiterations()`` therefore opens a top-level timer of its
own, ``JK: GTFock DF setup``, a sibling of ``HF: Form G`` rather than a child of
``JK: JK``, covering the three-centre integrals, the metric inverse square root
and the redistribution together. The name is defined once in
:source:`psi4/src/psi4/libfock/gtfock_df_interface.h` and read back through
:py:func:`psi4.driver.gtfock.df_setup_timer`, so a benchmark asks the module for
it rather than carrying a copy that can drift. Add that line to ``JK: JK``, or
read ``Total time``, before comparing against a ``direct`` or ``gtfock`` row.

Measured behaviour
..................

One water in Cartesian ``cc-pVDZ`` with ``cc-pVDZ-JKFIT``, against |PSIfours| own
``MemDFJK`` on the same basis pair and the same densities: ``max|dJ|`` 1.1e-12 and
``max|dK|`` 2.1e-13, and a full RHF landing 5.2e-12 :math:`E_h` from the
``mem_df`` energy. The two builders solve the same fitting equations, so this is
roundoff and is expected to stay roundoff |w---w| it is not the loose cross-engine
agreement the exact path has against PK.

The water hexamer in the same basis pair (150 basis functions, 786 auxiliary
functions, 6 auxiliary directions dropped by the fitting condition, a
9,028,782-double tensor) splits exactly at four ranks: 197 + 197 + 196 + 196
auxiliary functions, AO-pair-element counts summing to the whole, and tensor
slices tiling the whole tensor with no rank holding all of it. Every rank reports
the same null-space dimension, since the metric is inverted redundantly from a
replicated :math:`(P|Q)`. The energy spread over one, two, three and four ranks is
around 4e-14 :math:`E_h`, and wall time goes from 6.5 s at one rank to 4.3 s at
four |w---w| a real but sublinear gain, for the same replicated-remainder reason as
the exact path.

Introspection, all usable without mpi4py and all separate from the exact path's
counters:

* :py:func:`psi4.core.gtfock_df_enabled` |w---w| was |PSIfour| compiled against
  ``libgtfockdf``. Equivalently ``psi4.addons("gtfock_df")`` and
  :py:func:`psi4.driver.gtfock.df_available`.
* :py:func:`psi4.core.gtfock_df_jk_builds` |w---w| DF J/K builds this process ran
* :py:func:`psi4.core.gtfock_df_partition` |w---w| the most recently created
  engine's ``nbf``, ``naux``, auxiliary functions on this rank, metric null-space
  dimension, and local AO-pair elements.
  :py:func:`psi4.driver.gtfock.df_partition` returns those and the local tensor
  size as a dict. All are ``-1`` before any engine has been built.

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

The density-fitted engine is detected separately, and needs no extra configure
flag. ``FindGTFock`` looks for ``gtfock_pdf.h``, ``gtfock_df.h`` and
``libgtfockdf`` under the same prefix and sets ``GTFock_DF_FOUND`` only when all
three turn up alongside ``libgtfock``; configure prints a status line when it
finds GTFock without them. Such a build still gets ``SCF_TYPE GTFOCK``, and
``psi4.addons("gtfock_df")`` is False. The recipe above installs the DF library
by default, so a GTFock install predating it is the only way to land on that
side.

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

``SCF_TYPE GTFOCK_DF`` is driven the same way, since ``PDF_create`` is collective
and so also needs MPI up before |PSIfour| touches it. It adds a fitting basis,
which must be Cartesian too |w---w| ``puream false`` covers both:

.. code-block:: python

    psi4.set_options({"basis": "cc-pvdz", "puream": False,
                      "df_basis_scf": "cc-pvdz-jkfit",
                      "scf_type": "gtfock_df"})
    energy = psi4.energy("scf")               # or psi4.energy("uhf")

    assert psi4.core.gtfock_df_jk_builds() > 0
    print(gtfock.df_partition())              # how the fitted tensor was split

Its counters are separate from the exact path's, because the two engines share no
state; they are listed under `Distributed density fitting`_.

Testing
~~~~~~~

:source:`tests/pytests/test_gtfock.py` covers both opt-in paths and skips cleanly
when GTFock is absent, except for the three optionality guards and the six
reducer tests, which always run:

.. code-block:: bash

    >>> pytest -v tests/pytests/test_gtfock.py

Without GTFock this is ``9 passed, 29 skipped``. Two of the optionality guards
assert that :py:mod:`psi4.driver.gtfock` imports, reports itself unavailable,
raises a GTFock-specific error rather than a stray ``ImportError``, and needs no
mpi4py to do any of it. The third, ``test_gtfock_df_is_optional``, is separate
from them on purpose: ``libgtfockdf`` is optional *within* ``ENABLE_GTFock``, so
the two add-on flags can legitimately disagree and each needs its own guard. It
checks that ``psi4.addons``, ``psi4.core.gtfock_df_enabled`` and
:py:func:`psi4.driver.gtfock.df_available` agree, and that the DF flag is never
set without the GTFock one under it. The six reducer tests drive
:source:`tests/pytests/gtfock_hpc_collect.py` over synthesized per-rank records
and need neither GTFock nor MPI: they check that one run collapses to one row
with the slowest rank's wall clock, the worst rank's memory and the node's summed
memory, that a job spanning several nodes is still one point, and that records
from two jobs, one directory passed twice, two runs with no job id at all, or a
point missing some of the ranks it declares are refused rather than silently
merged.

The multi-rank cases launch :source:`tests/pytests/gtfock_mpi_driver.py` under
``mpirun`` and assert per-rank evidence: mpi4py and |PSIfours| MPI agree, each
rank owns a distinct AO block, ``jk.name()`` is ``GTFockJK``, GTFock's Fock build
counter advanced, and the energy agrees with |PSIfours| own PK result.

* ``test_gtfock_multirank_mpirun`` is the smoke case |w---w| STO-3G water RHF on
  two ranks, cheap enough to leave in an ordinary test run.

* ``test_gtfock_rank_count_invariance`` is the correctness case. It runs ``scf``
  and ``b3lyp`` on a water hexamer in Cartesian 6-31G* (60 shells, 114 basis
  functions) at one, two and four ranks. Four ranks give a 2x2 process grid, and
  each of the four ranks owns a strict sub-block of the 114-function matrix that
  GTFock has itself split into more than one task block in each dimension; the
  test asserts all three, because a system small enough to hand each rank a
  single block would say nothing about the decomposition.

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

The fourteen ``test_gtfock_df_*`` cases carry their own ``gtfock_df`` marker and,
unlike the exact path's, need no subprocess isolation: the engine is re-entrant,
so they build and drop engines freely in one process. They assert, in order: that
J and K match |PSIfours| own ``MemDFJK`` on the same basis pair to 1.0e-9 (the two
solve the same fitting equations, so anything looser would be hiding a bug rather
than pinning one); that RHF and UHF energies match ``mem_df``; that an empty
occupied block, i.e. a hydrogen atom's beta space, is handled rather than passed
to ``PDF_computeJK`` as ``nocc = 0`` work; that the engine really is built in
``initialize()`` and not on first use, which is the claim
`What the J/K timer does not cover`_ turns on; that two engines may be alive at
once and a released one rebuilt; that a spherical orbital basis and a spherical
fitting basis each raise on their own; that a basis above
``GTFDF_maxSupportedAM()`` raises, which needs Cartesian ``cc-pV6Z`` rather than
``cc-pV5Z`` because this path's ceiling is one higher than the exact path's; that
``wK`` and a non-symmetric density each raise; and that J alone can be requested
without K.

``test_gtfock_df_rank_count_invariance`` is the distributed case. It drives
:source:`tests/pytests/gtfock_mpi_driver.py` with ``--scf-type gtfock_df`` and
``--df-basis`` at one, two, three and four ranks on the water hexamer, and checks
both halves of the claim: that the per-rank auxiliary-function counts, AO-pair
element counts and tensor slice sizes each sum to the whole-tensor totals with no
rank holding all of it, and that the energy is invariant to 1.0e-9 :math:`E_h`
across rank counts. Summing to the total is what distinguishes a real partition
from a replicated tensor that happens to give the right answer. Three ranks is in
the sweep because ``naux`` does not divide evenly by it, which is the case the
remainder handling in the partition exists for.

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

Rank scaling at a fixed core count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table above answers "does adding cores help?" |w---w| each rank there had a
single thread, so four ranks used four times the hardware. A user with a fixed
allocation asks a different and harder question: given one node, is it better to
split its cores across MPI ranks or to leave them in one threaded process? That
comparison holds the total core count constant, so nothing is hidden by the extra
hardware, and it is the comparison that puts the replicated remainder of the SCF
under a spotlight.

:source:`tests/pytests/gtfock_hpc_benchmark.py` measures one point per
invocation and :source:`tests/pytests/gtfock_hpc_phoenix.slurm` sweeps a whole
system inside a single exclusive allocation, so no two points in a sweep can land
on different hardware. :source:`tests/pytests/gtfock_hpc_collect.py` reduces the
per-rank JSON to the tables below and to
:source:`tests/pytests/gtfock_hpc_results.csv`, which carries every number
measured, including the ones not shown here.

The systems are two dimers from a live SAPT(DFT) study, run as single
closed-shell molecules with both fragments present |w---w| this is a Fock-build
measurement, not an interaction energy. The basis is 6-31+G**, chosen because
|PSIfour| ships it Cartesian and the GTFock engine currently supports Cartesian
basis sets only; that is a statement about the engine's present envelope, not
about the basis. The benchmark asserts the loaded basis is Cartesian and has the
expected function count, so a basis change cannot quietly turn the two arms into
two different computations.

The hardware is one exclusive node of Georgia Tech's Phoenix cluster,
``cpu-small`` partition with the ``core24`` feature: two 12-core Intel Xeon Gold
6226 sockets at 2.70 GHz, 24 physical cores, 187 GB, no hyperthreading in the
allocation. Every point in a sweep gets all 24 cores. The reference arms run as
one process with 24 OpenMP threads; GTFock runs as 1 rank x 24 threads, 2 x 12,
and 4 x 6. Both reference arms are |PSIfours| own builders on the same node:
``DirectJK`` (``scf_type direct``) is the algorithmic apples-to-apples baseline,
since it evaluates the same exact ERIs GTFock does, and ``MemDFJK``
(``scf_type df``) is recorded alongside it as a *different* algorithm, useful for
context but not a like-for-like comparison. The submission script carries the
account, QOS, and partition the runs actually used |w---w| ``gts-cs207-chemx``,
``inferno``, ``cpu-small`` |w---w| so the queue is part of the record rather than
something to guess at. The two systems ran in two jobs on two nodes of that same
partition and feature; every comparison below is within one system, and none is
drawn across them.

Two details of the protocol are worth stating because they cut against the
conclusion rather than for it. First, a ``core24`` node is two sockets, and
OpenMPI refuses to bind one rank across both packages, so the one-rank GTFock
point runs unbound while the 2- and 4-rank points come out socket-local
(``--report-bindings`` recorded ``package[0][core:0-11]`` and
``package[1][core:12-23]`` at two ranks, and two ranks per package at four): the
NUMA advantage sits on the multi-rank side. Second, the SAD guess and its atomic
J/K are pinned to density fitting on every point (``sad_scf_type df``), and
``df_scf_guess`` is off, so the guess is identical work in all arms and no DF
pre-pass hides inside the measured SCF. Convergence thresholds, integral
screening, and the maximum iteration count are the same everywhere.

Peak memory per point is the per-rank high-water mark read from
``/proc/self/status`` (``VmHWM``). ``sacct`` reports no ``MaxRSS`` on this
cluster, so the only job-level figure is the ``mem=`` field of the SLURM
epilogue's ``Rsrc Used:`` line |w---w| 1.8 GB for the peptide job, 10.4 GB for
the nanotube one |w---w| and all five points of a system share one job, so that
figure belongs to the whole sweep and cannot be attributed to a point. Both
jobs' provenance, ``sacct`` output and epilogue are copied verbatim into
:source:`tests/pytests/gtfock_hpc_provenance.txt`. The per-rank measurement is
reported two ways, because they answer different questions: ``RSS/rank`` is what
one process needed, ``RSS node`` is the sum over ranks, which is what the node
had to supply.

.. peptide backbone dimer, 24 atoms, 6-31+G** (260 basis functions in 122
   shells), on atl1-1-02-006-1-1

======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
arm     ranks  thr  grid  iters  SCF (s)  J/K (s)  speedup  RSS/rank (MB)  RSS node (MB)  dE (Eh)
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
direct  1      24   ---   11     20.1     16.1     ---      801            801            ---
df      1      24   ---   11     3.3      0.6      ---      1492           1492           6.0e-04
gtfock  1      24   1x1   11     6.8      5.1      1.00     643            643            -1.3e-07
gtfock  2      12   1x2   11     6.1      4.6      1.12     499            995            -1.3e-07
gtfock  4      6    2x2   11     6.3      4.8      1.08     474            1891           -1.3e-07
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========

.. ethene plus nanotube fragment, 42 atoms, 6-31+G** (574 basis functions in 256
   shells), on atl1-1-02-006-18-2

======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
arm     ranks  thr  grid  iters  SCF (s)  J/K (s)  speedup  RSS/rank (MB)  RSS node (MB)  dE (Eh)
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
direct  1      24   ---   11     302.9    294.1    ---      1105           1105           ---
df      1      24   ---   11     32.7     7.7      ---      10360          10360          1.4e-03
gtfock  1      24   1x1   11     94.9     87.6     1.00     803            803            -8.4e-06
gtfock  2      12   1x2   11     94.5     84.5     1.00     699            1397           -8.4e-06
gtfock  4      6    2x2   11     88.7     84.2     1.07     605            2403           -8.4e-06
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========

``speedup`` is total SCF against the one-rank GTFock point, ``dE`` is against the
``direct`` arm on the same system, and ``J/K (s)`` is the ``JK: JK`` timer that
``JK::compute()`` wraps around every builder's ``compute_JK()``, so it is the
same clock in all three arms. It is not the same *fraction* of each arm's work:
on ``df`` it excludes the three-index construction, which happens before that
timer opens, so every ``df`` row's ``J/K (s)`` in this document is a partial
figure. `What the J/K timer does not cover`_ measures what it leaves out, at all
four sizes. Every point converged in 11 iterations, and the
GTFock energy is identical across ranks to every digit printed |w---w| the
spread within a rank count is exactly zero, not merely small. Rank-count
invariance survives at production size, not just in the tests.

The engine itself is a clear win at equal hardware. GTFock computes the same
exact ERIs as ``DirectJK`` and does it 3.0x faster on the peptide and 3.2x faster
on the nanotube, in less memory per process (643 against 801 MB, 803 against
1105 MB). That is Simint plus GTFock's own task scheduling, and it is available
at one rank.

Splitting a fixed core count across ranks, however, buys essentially nothing:
1.00, 1.12, 1.08 on the peptide and 1.00, 1.00, 1.07 on the nanotube. This is a
negative result and it is the honest one. It is also not simply Amdahl's law
acting on the replicated remainder: the J/K build *alone* barely moves either
(5.1 to 4.8 s, and 87.6 to 84.2 s), and on the nanotube the remainder is only 8%
of the SCF, which would still allow 3.3x at four ranks. GTFock's OpenMP threading
inside a single rank already saturates the 24 cores, so re-partitioning the same
cores into MPI ranks adds a gather-and-broadcast round per Fock build without
adding any compute to divide it over.

Memory moves the wrong way while that happens. Each rank's own footprint does
fall |w---w| 643 to 474 MB and 803 to 605 MB |w---w| so the distributed AO blocks
are real and not a bookkeeping fiction. But the replicated J, K, density, and
Fock matrices dominate, so the node total grows roughly with the rank count:
2.9x on the peptide and 3.0x on the nanotube at four ranks. Paying three times
the memory for a 1.08x speedup is not a trade worth making.

The practical consequence for a user with one node is short: run GTFock with one
rank and all the cores. Ranks earn their keep when they bring *more* cores, which
is what the previous section measures at one thread per rank and what
`Scaling out: one rank per node`_ measures at twenty-four; they do not earn it by
subdividing a fixed set. Distributing the rest of the SCF is what would change that, and it is the
same conclusion the fixed-thread table reaches from the other direction.

The density-fitting line is included for orientation and should not be read as a
loss. It is faster in wall clock, but it is a different algorithm: it approximates
the ERIs, and it lands 6.0e-04 :math:`E_h` (peptide) and 1.4e-03 :math:`E_h`
(nanotube) away from the exact-ERI answer, while needing 10.1 GB for the nanotube
|w---w| more than four times the whole four-rank GTFock node footprint. GTFock's
own offset from ``direct``, -1.3e-07 and -8.4e-06 :math:`E_h`, is the Simint
primitive-screening floor described under `Prototype scope`_: it grows with the
number of well-separated centres and is identical at every rank count, which is
what distinguishes it from a distribution bug.

These are measurements from one cluster at two problem sizes, reported as
measured and not extrapolated. The driver, the submission script, the reducer,
and the raw numbers are all in the tree.

Scaling out: one rank per node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two tables above hold the hardware fixed and vary how it is used. This one
does the opposite: every rank owns a whole 24-core node, so the rank count and
the core count grow together, 24 to 96 cores across four nodes. It is the
question a user with a queue allocation actually asks |w---w| "if I request four
nodes instead of one, what do I get?" |w---w| and it is the only one of the three
sweeps in which the ranks have to talk across a network.

:source:`tests/pytests/gtfock_hpc_phoenix_multinode.slurm` runs it, with the same
driver and the same reducer as the fixed-core sweep. All four GTFock points, both
reference arms, and a control point come out of a *single* four-node allocation,
so a wider point cannot be faster for having landed on quieter hardware, and
``SLURM_JOB_NODELIST`` is identical across the points. The four nodes were checked
identical from ``scontrol show node`` |w---w| two-socket Xeon Gold 6226,
``CPUTot=24``, ``RealMemory=191000``, same feature list |w---w| and the nodes are
InfiniBand-connected: UCX 1.19.1 in this environment reports ``rc_mlx5``,
``dc_mlx5`` and ``ud_mlx5`` transports on a Mellanox ``mlx5_0:1`` HCA, so the
cross-node traffic has a real IB path rather than falling back to Ethernet. Each
rank gets 24 cores and 120 GB. A core set spanning all 24 cores crosses both
sockets, which OpenMPI refuses to bind, so every point here runs unbound with
``OMP_PLACES=cores``, the one-rank point included; unlike the fixed-core sweep,
the placement is uniform across the rank counts rather than varying with them.

Before spending any SCF time the script launches a ``hostname``-only MPI job and
aborts unless the ranks land on as many distinct hosts as there are ranks. A sweep
that quietly oversubscribed one node would measure the wrong thing while looking
entirely plausible; both jobs' checks are recorded in
:source:`tests/pytests/gtfock_hpc_multinode_provenance.txt`. Every per-point
number is in :source:`tests/pytests/gtfock_hpc_multinode_results.csv`, kept
separate from the fixed-core CSV rather than merged into it, because the reducer
keys a point on (system, arm, ranks) and the two sweeps share rank counts while
meaning different things by them.

.. peptide backbone dimer, 24 atoms, 6-31+G** (260 basis functions in 122
   shells), on atl1-1-02-010-23-2 and three more nodes, job 12433686

======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
arm     ranks  thr  grid  iters  SCF (s)  J/K (s)  speedup  RSS/rank (MB)  RSS node (MB)  dE (Eh)
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
direct  1      24   ---   11     17.8     16.1     ---      806            806            ---
df      1      24   ---   11     3.8      0.6      ---      1482           1482           6.0e-04
gtfock  1      24   1x1   11     7.1      4.9      1.00     643            643            -1.3e-07
gtfock  2      24   1x2   11     5.0      3.2      1.43     647            1291           -1.3e-07
gtfock  3      24   1x3   11     4.3      2.5      1.66     649            1936           -1.3e-07
gtfock  4      24   2x2   11     3.9      2.1      1.84     647            2580           -1.3e-07
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========

.. ethene plus nanotube fragment, 42 atoms, 6-31+G** (574 basis functions in 256
   shells), on the same four nodes, job 12433617

======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
arm     ranks  thr  grid  iters  SCF (s)  J/K (s)  speedup  RSS/rank (MB)  RSS node (MB)  dE (Eh)
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
direct  1      24   ---   11     300.9    293.2    ---      1124           1124           ---
df      1      24   ---   11     44.5     7.8      ---      10342          10342          1.4e-03
gtfock  1      24   1x1   11     92.8     88.7     1.00     803            803            -8.4e-06
gtfock  2      24   1x2   11     49.7     45.9     1.87     803            1592           -8.4e-06
gtfock  3      24   1x3   11     35.5     31.9     2.62     791            2354           -8.4e-06
gtfock  4      24   2x2   11     27.9     24.3     3.33     807            3205           -8.4e-06
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========

Here ``ranks`` is also the node count, and ``thr`` stays at 24 rather than falling
as it does in the fixed-core table, so the total core count is 24 x ``ranks``.
``speedup`` is again total SCF against the one-rank GTFock point, which for this
sweep means one node.

Rank-count invariance holds across nodes at production size, and the tables
understate it. Within one rank count every rank returns a bitwise identical
energy, so the reducer's spread column is exactly zero. Across rank counts the
spread is 4.5e-13 :math:`E_h` on the peptide and 1.4e-11 on the nanotube |w---w|
factors of 2200 and 73 inside the 1e-09 tolerance the in-tree invariance test
enforces, and four to five orders of magnitude inside the -1.3e-07 and -8.4e-06
:math:`E_h` offsets from ``direct``. Those offsets are in turn the same to six
significant figures as the ones the fixed-core sweep measured in a different job
on different nodes with a different partitioning, which is a cross-check the two
sweeps get for free by sharing a driver. Every point converged in 11 iterations
with 12 J/K builds. Three nodes are in the table deliberately: ``split_procs()``
factors a prime rank count as 1x3 rather than refusing it, and this is the
measurement that says so |w---w| the in-tree invariance test now pins that grid
as well.

The speedup is real. On the nanotube, four nodes cut the SCF from 92.8 s to
27.9 s (1.00, 1.87, 2.62, 3.33), and the J/K build itself from 88.7 s to 24.3 s
(1.00, 1.93, 2.78, 3.64) |w---w| 97%, 93% and 91% parallel efficiency in the part
of the SCF that GTFock actually distributes. The peptide gains less: 1.00, 1.43,
1.66, 1.84 on the SCF and 1.00, 1.55, 1.97, 2.32 on the J/K, for 78%, 66% and 58%
efficiency. Both are a different world from the 1.07x and 1.08x the same rank
counts bought when they were subdividing one node's cores, and the difference is
not the engine: it is that these ranks brought hardware with them.

The gap between the two systems is a size effect and it is measurable rather than
asserted. Per Fock build, the nanotube is 18x more J/K work than the peptide
(7.39 s against 0.41 s at one node) but only 4.9x more matrix to gather and
broadcast, since the communicated J and K scale as the square of the basis size
while the integral work scales much more steeply. Subtracting an ideal 1/N from
the measured four-node build leaves 0.07 s per build on the peptide and 0.18 s on
the nanotube |w---w| both small in absolute terms, and decisive only when the
build being divided is itself 0.4 s.

What does not scale is everything else. The SCF total minus the J/K timer is
2.18, 1.80, 1.82, 1.86 s across the peptide's four points and 4.15, 3.72, 3.57,
3.55 s across the nanotube's: flat, because it is replicated on every rank by
design |w---w| each rank holds the full J and K, so diagonalization, DIIS, and the
DFT quadrature are done four times over rather than divided. That remainder is
the ceiling. Even a J/K build driven to zero could not take the peptide's
four-node SCF below 1.86 s (3.8x on 7.1 s) or the nanotube's below 3.55 s (26x on
92.8 s), and on the peptide the remainder is already 48% of the four-node wall
clock against 13% on the nanotube. Distributing the rest of the SCF, not tuning
the Fock build, is what would move these numbers further.

Memory behaves the way the design predicts, and the reading is the opposite of
the fixed-core sweep's. Per-rank RSS is flat here |w---w| 643 to 647 MB on the
peptide, 803 to 807 MB on the nanotube |w---w| because per-rank memory is exactly
what this protocol holds constant, and because the replicated matrices dominate
the distributed AO blocks at these sizes. The summed footprint still grows about
linearly with the rank count, 4.0x at four nodes in both systems, but each node is
now supplying under 1 GB of its own 187 GB rather than four ranks competing for
one node's budget. The same growth that made the fixed-core result a bad trade is
close to free here.

Two comparisons in the CSV need reading carefully. ``speedup_vs_direct`` reaches
4.61x on the peptide and 10.79x on the nanotube at four nodes, but that ratio
compares 96 cores against 24: it is a throughput number, not an algorithmic one.
The like-for-like engine comparison is the one-node row, 2.5x and 3.2x against
``DirectJK`` on identical hardware, consistent with the 3.0x and 3.2x the
fixed-core sweep measured. The density-fitting line also moved between the two
sweeps on the nanotube, 32.7 s there against 44.5 s here, while its ``J/K: JK``
timer did not budge (7.7 against 7.8 s), so the whole difference is outside
``JK::compute()``. It is not the three-index integral construction, which is
where an earlier version of this paragraph put it: two dedicated re-runs of this
point (jobs 12638902 and 12639820) time that construction at 6.5 and 6.3 s and
land the whole SCF at 17.9 and 18.0 s, under either sweep figure. The swing sits
in the non-J/K remainder instead, and the likeliest reading is the one the
peptide re-run makes visible |w---w| 21.172 s in ``HF: DIIS`` on a cold node
against 0.598 s on a warm one, which is PSIO scratch I/O on shared storage and
not work the arm does. It is recorded rather than smoothed, and it
is one more reason to read the DF row as orientation only. What is worth noting is
that GTFock's four-node exact-ERI SCF, 27.9 s, is faster than either
density-fitting measurement in this document while carrying no fitting error and
using 807 MB per process against 10.3 GB in one.

The one-sided component was checked rather than assumed. Both submission scripts
export ``OMPI_MCA_osc=^ucx`` to silence a per-rank OpenMPI log line, and GTFock's
``GTMatrix`` layer genuinely does use MPI one-sided RMA (``MPI_Win_create``,
``MPI_Get``, ``MPI_Accumulate``, window lock/unlock), so across a network that
export is an assumption about the component carrying real traffic and not a
cosmetic setting. The multi-node script therefore re-runs its widest point with
the variable unset, into a subdirectory the reducer will not merge with the point
it controls. The wall clock is unchanged |w---w| 4.1 s / 2.3 s against 3.9 / 2.1
on the peptide, 27.9 / 24.3 against 27.9 / 24.3 on the nanotube |w---w| so nothing
in the tables above depends on it. Per-rank memory is not: letting OpenMPI choose
the UCX one-sided component roughly doubles it, to 1364 MB and 1523 MB.

One caveat belongs on the peptide numbers specifically. A first attempt at that
sweep returned a one-node baseline of 33.4 s SCF against a 5.4 s J/K build, a 28 s
remainder where every other point in the same job had 1.6 to 2.7 s; taken at face
value it would have reported 8.55x at four nodes against a J/K improvement of only
2.49x. It was diagnosed as a transient stall rather than a systematic effect |w---w|
that process burned 215 s of user time in 33 s of wall on 24 cores while an
equivalent point elsewhere did the same work in 6 s |w---w| the sweep was
resubmitted with a byte-identical script, and the discarded job's records are kept
in the provenance file instead of deleted. The lesson is in the numbers: a
four-second SCF on a shared cluster sits close enough to the noise floor that one
point can be wrong by a factor of five, which is the second reason to treat the
peptide's 1.84x as the weaker of the two results.

Twelve measured points and two controls, from two jobs on one cluster, reported as
measured and not extrapolated. What they support is narrow and worth stating
plainly: given a system large enough to give each node real work, GTFock turns
additional nodes into a proportionally faster exact-ERI Fock build |w---w| 3.33x
on four nodes, 91% efficiency in the J/K itself |w---w| and the SCF wrapped around
it does not distribute at all. The section below runs the same protocol on a
system 2.7x wider, where both halves of that sentence get sharper and the second
one starts to cost more than it does here.

At production size: 1555 basis functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sweeps above top out at 574 basis functions. That is large enough to show the
distribution working and too small to say what it is worth: the peptide's
four-second SCF sits near the cluster's noise floor, and even the nanotube's
one-node J/K build is only 88.7 s. This section repeats the one-rank-per-node
protocol on a system 2.7x wider |w---w| a 157-atom fragment pair carved out of PDB
``3acx`` (a 118-atom peptide and a 39-atom ligand fragment, run as one
closed-shell molecule) in Cartesian ``6-31G**``: 1555 basis functions in 702
shells, 12 iterations and 13 Fock builds at every point.

:source:`tests/pytests/gtfock_hpc_phoenix_protein.slurm` runs it, submitted with
``FM_BASIS=6-31G**``. Job 12445512 took 57 minutes of a four-node allocation and
used the same driver, the same reducer and the same pre-flight placement check as
the two sweeps above, on four nodes of the same type but not the same four nodes.
Two things in that script differ on purpose: a twelve-hour walltime, and 160 GB
rather than 120 GB for the single-process reference arms, so that density fitting
keeps its three-index tensor in core instead of silently becoming ``DiskDFJK`` and
reporting a third algorithm under the second one's name. It did stay in core at
this width |w---w| the ``df`` row below is ``MemDFJK``, peaking at the 94750 MB
in the table. It does not at 1863 basis functions, which the basis-set boundary
at the end of this section reports.

.. 157-atom fragment pair from PDB 3acx, 6-31G** (1555 basis functions in 702
   shells), on atl1-1-02-006-16-2 and three more nodes, job 12445512

======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
arm     ranks  thr  grid  iters  SCF (s)  J/K (s)  speedup  RSS/rank (MB)  RSS node (MB)  dE (Eh)
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========
direct  1      24   ---   12     1674.2   1606.4   ---      3371           3371           ---
df      1      24   ---   12     393.5    244.1    ---      94750          94750          5.0e-03
gtfock  1      24   1x1   12     403.9    359.9    1.00     2407           2407           -5.4e-05
gtfock  2      24   1x2   12     224.1    179.9    1.80     2299           4568           -5.4e-05
gtfock  3      24   1x3   12     165.5    123.9    2.44     2215           6618           -5.4e-05
gtfock  4      24   2x2   12     129.5    94.1     3.12     2170           8650           -5.4e-05
======  =====  ===  ====  =====  =======  =======  =======  =============  =============  ========

The Fock build distributes better here than anywhere else in this document: 1.00,
2.00, 2.91, 3.82, which is 100%, 97% and 96% parallel efficiency at two, three
and four nodes, against the nanotube's 97/93/91% and the peptide's 78/66/58%. Size is the whole reason. One
node spends 27.68 s per Fock build here against 7.39 s on the nanotube, and
subtracting an ideal 1/N from the measured four-node build leaves 0.32 s per build
against the nanotube's 0.18 s |w---w| 1.8x the overhead to hide 3.7x the work. The
SCF total follows at 1.00, 1.80, 2.44, 3.12, cutting 403.9 s to 129.5 s.

That 3.12x is *lower* than the nanotube's 3.33x, and the reason is exactly what
running a bigger system was supposed to expose. The replicated remainder |w---w| SCF
total minus the J/K timer |w---w| is 44.0, 44.2, 41.7 and 35.5 s here against the
nanotube's 4.15, 3.72, 3.57 and 3.55. Going from 574 to 1555 basis functions
multiplied the one-node J/K build by 4.1 and that remainder by 10.6, so the
fraction of the SCF GTFock does not distribute went *up* with system size: 4.5% to
10.9% of the one-node wall clock, and 13% to 27% at four nodes. Diagonalization,
DIIS and the density build are :math:`O(N^3)` and replicated on every rank, while
a screened Fock build is not, and over this range the replicated part is growing
faster. Even a free J/K could not take this four-node SCF below about 35 s, an 11x
ceiling on the one-node 403.9 s and a lower one than the 26x the nanotube had.
Distributing the rest of the SCF is what would move this number, and the larger
the system the more that is true.

The remainder is also the one quantity here that is not quite flat: it sheds 8.5 s
across the sweep while the J/K sheds 266 s. Nothing in the record decomposes that
8.5 s, and the component of it that could most plausibly shrink with rank count
does not |w---w| GTFock prints the cost of its Schwarz screening setup, and that
is 0.87, 0.86, 0.81 and 0.84 s at one through four nodes. It is reported rather
than explained.

Rank-count invariance holds at production size. Within a rank count every rank
returns a bitwise identical energy; across one, two, three and four nodes the
spread is 1.1e-11 :math:`E_h`, two orders of magnitude inside the 1e-09 tolerance
the in-tree invariance test enforces and six inside the offset from ``direct``.
Every point converged in 12 iterations with 13 J/K builds, on the 1x1, 1x2, 1x3
and 2x2 grids.

That offset, -5.4e-05 :math:`E_h`, is the primitive-pair screening difference
described under `Testing`_, and it behaves as that section says it does: it grows
with the number of well-separated centres |w---w| -1.3e-07 :math:`E_h` at 24
atoms, -8.4e-06 at 42, -5.4e-05 here at 157. It is the same to three significant
figures at all four rank counts, so it is a property of the engine and not of the
distribution.

The reference arms support three different comparisons and they are worth keeping
apart. Like for like on one node, GTFock's 403.9 s against ``DirectJK``'s 1674.2 s
is 4.15x, or 4.46x on the J/K timer alone |w---w| a wider margin than the 2.5x and
3.2x the smaller systems gave, since a larger molecule is what GTFock's blocking
and screening are built for. Against density fitting, |PSIfour|'s default and the
algorithm most users actually run, the one-node points are effectively tied at
403.9 s to 393.5 s, and four nodes make the exact-ERI engine 3.0x faster than
density fitting on the same molecule |w---w| at 2170 MB per rank against 94750 in
one process, and 93x closer to the exact-ERI answer (-5.4e-05 against +5.0e-03
:math:`E_h` of fitting error). Both of those are SCF totals and both stand. The
comparison that does not is the one the ``J/K (s)`` column invites: 1606.4 s
against 244.1 s is 6.6x, but 91.6 s of this ``df`` point's integral work happens
before that timer opens, and the honest ratio of integral work is 4.7x.
`What the J/K timer does not cover`_ has the split. The CSV's
``speedup_vs_direct`` column reaches 12.9x at four nodes; that compares 96 cores
against 24 and is a throughput number, not an algorithmic one.

Memory finally shows the distribution doing something. Per-rank RSS *falls* with
rank count here |w---w| 2407, 2299, 2215, 2170 MB |w---w| rather than staying flat
as it did at 574 basis functions, and the summed footprint grows 3.59x over a 4x
increase in ranks instead of 4.0x. The AO blocks GTFock actually partitions are
large enough at 1555 basis functions to show against the replicated matrices,
though the replicated part still dominates: four nodes hold 8650 MB between them
to do what one node did in 2407.

The one-sided control was repeated at this size rather than carried over, since
the RMA volume per Fock build grows with the square of the basis and the previous
check was made at 574 basis functions. With ``OMPI_MCA_osc`` unset the four-node
point runs 128.4 s SCF and 93.8 s J/K against 129.5 and 94.1, and the two energies
agree to 7.3e-12 :math:`E_h`, so the exclusion is cosmetic here too. Per-rank
memory is again not: 2870 MB against 2170, an increment within 17 MB of the one
both smaller systems showed.

One boundary belongs on this result: it is measured in Cartesian ``6-31G**``. The
same molecule in ``6-31+G**`` is 1863 basis functions, and there the GTFock SCF
ran 100 iterations without converging |w---w| job 12434225, which then died before
either reference arm started and is the failure the point-level error handling in
the script was written for. Job 12441235 supplied the reference arms that job
never reached, on one node of the same type: ``DirectJK`` converged that input in
12 iterations to -3642.752751 :math:`E_h` in 8470 s, and density fitting in 12 to
-3642.748403 in 829 s, both monotonically and both with the DIIS error falling to
1.6e-08. That density-fitting figure is ``DiskDFJK``, not ``MemDFJK``. At 1863
basis functions the 160 GB this script gives the reference arms reaches MemDFJK's
estimator as 111.759 GiB against the 127.825 GiB of AOs it needs, so |PSIfour|
falls back, and the in-core guarantee that holds at 1555 does not extend here.
Job 12638902 reproduced the fallback, and job 12639820 held the same point in
core at 185 GB, where it took 778.3 s against the disk algorithm's 716.2 s on a
node of the same type |w---w| in core was not the faster arrangement.

So the non-convergence is GTFock's and not the system's, which is why
this section reports the smaller basis and why the restriction is stated under
`Prototype scope`_ rather than treated as a scaling limit. Nothing about
distribution is implicated either: rebuilding J on one fixed density at 1, 2, and
4 ranks reproduces the same error to every digit carried.

Six more measured points and a third control, from a third job on the same
cluster, again reported as measured and not extrapolated. What they add to the two
smaller systems is a size dependence that runs in two directions at once: the Fock
build distributes better the larger the system gets, reaching 96% parallel
efficiency on four nodes, while the replicated remainder around it grows faster
than the Fock build does, so the speedup of the SCF as a whole does not keep
improving |w---w| 1.84x at 260 basis functions, 3.33x at 574, 3.12x at 1555. Both
halves of that point at the same next step, which is distributing the rest of the
SCF rather than tuning the part that already scales.

What the J/K timer does not cover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every table above sets a ``df`` row's ``J/K (s)`` beside a ``direct`` or
``gtfock`` row's, and the quotient invites being read as a ratio of Fock-build
speed. It is not one, and the reason is structural rather than statistical.
``JK::initialize()`` calls the subclass's ``preiterations()``
(:source:`psi4/src/psi4/libfock/jk.cc`), and ``timer_on("JK: JK")`` is opened
later, inside ``JK::compute()``. Whatever a builder does in ``preiterations()``
is invisible to that clock. ``DirectJK::preiterations()`` does essentially
nothing beyond a BrianQC hook, and GTFock's one-shot Simint and screening setup
is reported separately above. But ``MemDFJK::preiterations()`` ends in
``dfh_->initialize()`` with method ``STORE``
(:source:`psi4/src/psi4/libfock/MemDFJK.cc`): it builds the entire three-index
tensor and keeps it. That is most of density fitting's integral work, and none
of it is in the column.

``J/K (s)`` is therefore the same clock in every arm, as those sections say, but
it does not cover the same *fraction* of each arm's work. On the exact-ERI arms
it covers all the integral work. On ``df`` it covers the contraction and not the
construction, and the ratio between the two columns is correspondingly
flattered.

Measuring the omission needed no new C++. ``DFH: sparsity prep`` and
``DFH: initialize()`` have been timers in
:source:`psi4/src/psi4/lib3index/dfhelper.cc` all along, and ``DiskDFJK``
brackets its own setup under ``JK: (A|mn)``, ``JK: (A|Q)^-1/2`` and
``JK: (Q|mn)`` (:source:`psi4/src/psi4/libfock/DiskDFJK.cc`) |w---w| all of them
top-level timers, siblings of ``HF: Form G`` rather than children of ``JK: JK``.
Both vocabularies have to be watched, because both builders answer to
``scf_type df`` and only one of them uses ``DFHelper``.
:source:`tests/pytests/gtfock_hpc_phoenix_df_setup.slurm` re-runs the ``df`` arm
alone at all four documented sizes, one process on one exclusive ``core24``
node, and :source:`tests/pytests/gtfock_hpc_phoenix_df_setup2.slurm` repeats the
three points that needed a warm node or a larger budget. Wall seconds, ``setup``
being the sum of those top-level timers and ``rest`` the SCF with both the setup
and ``JK: JK`` removed:

.. density fitting alone, one process x 24 threads, exclusive core24 nodes,
   jobs 12638902 (160 GB) and 12639820 (120 GB, 185 GB at the last row)

==========  ====  ========  =========  =======  ========  =======  ========  ========
system      nbf   builder   setup (s)  J/K (s)  rest (s)  SCF (s)  RSS (MB)  job
==========  ====  ========  =========  =======  ========  =======  ========  ========
peptide     260   MemDFJK   0.84       0.62     1.86      3.3      1494      12639820
nanotube    574   MemDFJK   6.33       8.32     3.38      18.0     10362     12639820
protein157  1555  MemDFJK   91.58      248.38   50.75     390.7    94736     12638902
protein157  1863  DiskDFJK  167.11     479.16   69.92     716.2    128048    12638902
protein157  1863  MemDFJK   285.53     426.78   65.98     778.3    194145    12639820
==========  ====  ========  =========  =======  ========  =======  ========  ========

Setup is between a quarter and better than a third of the density-fitting SCF at
every size measured |w---w| 25%, 35%, 23%, 23% and 37% of the SCF column |w---w|
and at every size but the smallest it is larger than the whole non-J/K remainder
left after it is removed. It
does not amortise away with system size either: it grows from 0.84 s to 285.5 s
across a 7.2x widening of the basis, and at 1863 basis functions in core it is
two thirds the size of all thirteen Fock builds put together.

The 1555-basis-function point is where this changes a published number, because
it is the only size at which the re-run reproduces the in-job measurement. Job
12638902 returns 390.70 s SCF and 248.38 s in ``JK: JK`` against job 12445512's
393.5 and 244.1, 0.7% and 1.7% apart on a different node of the same type. That
licenses putting the three ways of comparing it against the same job's
``direct`` arm (1674.2 s SCF, 1606.4 s J/K) side by side:

* ``JK: JK`` against ``JK: JK``, which is what the two table columns give:
  **6.47x**.
* integral work against integral work, the DF setup added back:
  1606.4 / (248.38 + 91.58) = **4.73x**.
* SCF total against SCF total: 1674.2 / 390.70 = **4.28x**.

The first overstates the last by 51%. The third is the one that answers "how
much sooner do I get an energy", the second is the one that answers "how much
cheaper are the integrals", and the first answers neither.

The 574-basis-function point does not reproduce, and the split is published
there with no ratio attached. Two re-runs give 18.0 and 17.9 s SCF where the two
sweeps gave 44.5 and 32.7. Their 6.3 and 6.5 s of setup against 7.8 and 8.3 s of
J/K is a real measurement of this build on this hardware, and the fraction it
implies stands; a speedup against a ``direct`` arm measured in a different job at a
different hour is not something these runs license, so none is quoted. The
260-basis-function point is in the same position for a smaller reason, its whole
SCF being 3.3 s, and is worth reporting only for the shape: 0.62 s inside
``JK: JK``, 0.84 s building the tensor and 1.86 s in everything else, so at this
size density fitting's advantage over exact ERIs is not a Fock-build story at
all.

The two 1863-basis-function rows are the one place both builders ran the same
input, and they divide the work differently rather than one being a slow copy of
the other. ``DiskDFJK`` spends 167.1 s in setup |w---w| ``JK: (A|mn)`` 19.9,
``JK: (A|Q)^-1/2`` 40.1, ``JK: (Q|mn)`` 102.7 |w---w| and 479.2 s in J/K.
``MemDFJK`` at a 185 GB budget spends 285.5 s in setup and 426.8 s in J/K. Going
in core buys 52 s of Fock build and pays 118 s for it. Its
``DFH: AO Construction`` and ``DFH: AO-Met. Contraction`` each report 38 calls
against 1 at 1555 basis functions, so the tensor was assembled in 38 Q-shell
blocks: in core by the estimator's standard, and not comfortably. Peak RSS was
194145 MB against the 185 GB declared, which is worth knowing before reading
that budget as a cap.

With both rows in hand the diffuse-function comparison finally has a
same-algorithm form. ``MemDFJK`` goes 390.7 s to 778.3 s from 1555 to 1863 basis
functions, 1.99x, against ``DirectJK``'s 1674.2 s to 8470 s, 5.06x |w---w| so
the diffuse set costs the exact-ERI arm about 2.5x what it costs density
fitting, which is the screening degradation described under `Testing`_ showing
up in wall clock. The 2.11x that 393.5 s and 829 s appear to give is not one
algorithm's inflation, since those two points are ``MemDFJK`` and ``DiskDFJK``;
that it lands near the right answer anyway is a coincidence and not a
confirmation.

None of this moves a GTFock number, and it does not disturb any comparison in
this document that was made on SCF totals: the one-node tie at 1555 basis
functions, the 3.0x at four nodes, the per-process memory contrast and the
fitting-error contrast all stand as written. What it removes is the use of a
``df`` row's ``J/K (s)`` as a proxy for how fast density fitting builds a Fock
matrix. Two things it does not settle. Every point here is one process, so
nothing in this section speaks to how a *distributed* density fitting would
compare against distributed GTFock, which is the comparison the prototype
actually needs. And every figure is a single measurement on shared storage, with
the 260-basis-function point's own history |w---w| 27.5 s and 26.4 s on two cold
nodes, then 3.3 s on the second of them once a discarded warm-up point had run
|w---w| as the standing reminder of what one such measurement is worth.

Both jobs' provenance blocks, ``sacct`` records, epilogues and full timer trees
are copied verbatim into
:source:`tests/pytests/gtfock_df_setup_provenance.txt`, together with the two
lines in which |PSIfour| announces the fallback.
:source:`tests/pytests/gtfock_hpc_benchmark.py` now reports the setup total, its
constituent timer keys and the SCF remainder next to the J/K figure it already
reported, so a future sweep records the split without a second job.

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
