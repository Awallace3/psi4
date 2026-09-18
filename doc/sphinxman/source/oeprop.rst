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

.. index::
   single: OEProp
   pair: OEProp; theory


.. _`sec:oeprop`:

Evaluation of One-Electron Properties |w---w| :py:func:`~psi4.driver.oeprop`
============================================================================

.. codeauthor:: Robert M. Parrish and Andrew C. Simmonett
.. sectionauthor:: Andrew C. Simmonett

.. autofunction:: psi4.oeprop(wfn, \*args[, title])
   :noindex:

|PSIfour| is capable of computing a number of one-electron properties
summarized in the table below. 

.. _`table:oe_features`:

.. table:: Current one-electron property capabilities of |PSIfour|

   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Feature                            | Keyword               | Notes                                                                             |
   +====================================+=======================+===================================================================================+
   | Electric dipole moment             | DIPOLE                |                                                                                   |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Electric quadrupole moment         | QUADRUPOLE            | Raw (traced) moments and traceless multipoles                                     |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | All moments up order N             | MULTIPOLE(N)          | Only raw (traced) moments. Sets global variables e.g. "DIPOLE", "32-POLE"         |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Electrostatic potential, at nuclei | ESP_AT_NUCLEI         | Sets global variables "ESP AT CENTER n", n = 1 to natoms                          |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Electrostatic potential, on grid   | GRID_ESP              | Generates V at each point in grid_esp.dat. See :ref:`sec:oeprop_grid`             |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Electric field, on grid            | GRID_FIELD            | Generates {Ex,Ey,Ez} at each point grid_field.dat. See :ref:`sec:oeprop_grid`     |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Molecular orbital extents          | MO_EXTENTS            |                                                                                   |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Mulliken atomic charges            | MULLIKEN_CHARGES      |                                                                                   |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | L\ |o_dots|\ wdin atomic charges   | LOWDIN_CHARGES        |                                                                                   |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | L\ |o_dots|\ wdin atomic spins     | LOWDIN_SPINS          | Scalar spin population: the fractional number of unpaired electrons [e].          |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Wiberg bond indices                | WIBERG_LOWDIN_INDICES | Uses (L\ |o_dots|\ wdin) symmetrically orthogonalized orbitals                    |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Mayer bond indices                 | MAYER_INDICES         |                                                                                   |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Natural orbital occupations        | NO_OCCUPATIONS        |                                                                                   |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Stockholder Atomic Multipoles      | MBIS_CHARGES          | Generates atomic charges, dipoles, etc. See :ref:`sec:oeprop_mbis`                |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
   | Hirshfeld volume ratios            | MBIS_VOLUME_RATIOS    | Generate the AIM to free atom volume ratios                                       |
   +------------------------------------+-----------------------+-----------------------------------------------------------------------------------+

There are two ways the computation of one-electron properties can be requested. 
Firstly, the properties can be evaluated from the last
computed one-particle density, using the following syntax::

  oeprop("MO_EXTENTS", "MULTIPOLE(4)", title = "hello!")

Note that it is the user's responsibility to ensure that the relaxed density
matrix is computed using the method of interest, which may require setting
additional keywords (see the method's manual section for details). The named
argument, *title*, is completely optional and is prepended to any
globals variables set during the computation.  The unnamed arguments are the
properties to be computed.  These can appear in any order, and multiple
properties may be requested, as in the example above.  Note that, due to Python
syntax restrictions, the title argument must appear after the list of
properties to compute.  The available properties are shown in the table above.

The syntax above works well for computing properties using the SCF
wavefunction, however, may be difficult (or impossible) to use for some of the
correlated levels of theory. Alternatively, one-electron properties can be
computed using the built-in properties() function, e.g.::

  properties('ccsd', properties=['dipole'])

The :py:func:`~psi4.driver.properties` function provides limited functionality, but is a lot easier to
use for correlated methods. For capabilities of :py:func:`~psi4.driver.properties` see the
corresponding section of the manual.


Basic Keywords
^^^^^^^^^^^^^^

Multipole moments may be computed at any origin, which is controlled by the
global |globals__properties_origin| keyword.  The keyword takes an array with
the following possible values:

.. _`table:oe_origin`:

.. table:: Allowed origin specifications

   +-------------------------------+-------------------------------------------------------------------------------+
   | Keyword                       | Interpretation                                                                |
   +===============================+===============================================================================+
   | [x, y, z]                     | Origin is at the coordinates, in the same units as the geometry specification |
   +-------------------------------+-------------------------------------------------------------------------------+
   | ["COM"]                       | Origin is at the center of mass                                               |
   +-------------------------------+-------------------------------------------------------------------------------+
   | ["NUCLEAR_CHARGE"]            | Origin is at the center of nuclear charge                                     |
   +-------------------------------+-------------------------------------------------------------------------------+


.. _`sec:oeprop_grid`:


Properties evaluated on a grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Certain properties may be evaluated a user-specified grid points.  The grid
points are completely arbitrary and are specified by providing a file called
grid.dat containing the x,y,z values separated with spaces for each point in order::

    x1 y1 z1
    x2 y2 z2
    ..........
    xn yn zn

The grid.dat file is completely free form; any number of spaces and/or newlines
between entries is permitted.  The units of the coordinates in grid.dat are the
same as those used to specify the molecule's geometry, and the output
quantities are always in atomic units.  The requested properties will be
written out in the same order as the grid point specification in grid.dat; see
the above table for the format and file name of the output.

The grid may be generated in the input file using standard Python loops.  By
capturing the wavefunction used to evaluate the one-electron properties, the
values at each grid point may be captured as Python arrays in the input file::

    E, wfn = prop('scf', properties=["GRID_ESP", "GRID_FIELD"], return_wfn=True)
    Vvals = wfn.oeprop.Vvals()
    Exvals = wfn.oeprop.Exvals()
    Eyvals = wfn.oeprop.Eyvals()
    Ezvals = wfn.oeprop.Ezvals()

In this example, the *Vvals* array contains the electrostatic potential at each
grid point, in the order that the grid was specified, while the *Exvals*,
*Eyvals* and *Ezvals* arrays contain the *x*, *y* and *z* components of the
electric field, respectively; all of these arrays can be iterated and
manipulated using standard Python syntax.  For a complete demonstration of this
utility, see the :srcsample:`props4` test case.


.. index:: ISA; MBIS

.. _`sec:oeprop_mbis`:

Minimal Basis Iterative Stockholder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Minimal Basis Iterative Stockholder (MBIS) method is one of many procedures
that partitions a molecular one-particle density matrix into atomic electron densities.
Running MBIS in |PSIfour| will calculate atomic valence charge widths, volume ratios,
atomic charges, as well as dipoles, quadrupoles, and octupoles. 
Additionally, all expectation values of radial moments of n-th order (:math:`<r^n>`) 
are computed up to fourth order. Higher moments can be computed by specifying |globals__max_radial_moment|.
The volume ratios are computed as the ratio between the volume of the atomic density
(:math:`<r^3>`) and the volume of the free atom computed using the same level
of theory, but with a potentially unrestricted reference.

The allowed number of iterations and convergence criteria for the stockholder 
algorithm is controlled by |globals__mbis_maxiter| and |globals__mbis_d_convergence|. Note 
that the density is partitioned on a molecular quadrature grid, the details of which can be
controlled with the keywords |globals__mbis_radial_points|, |globals__mbis_spherical_points|, and 
|globals__mbis_pruning_scheme|. (Associated Paper: [Verstraelen:2016]_)

.. note::
   MBIS is not supported for basis sets that use effective core potentials (ECPs).
   Please use all-electron basis sets for MBIS calculations. See `this issue at denspart <https://github.com/theochem/denspart/issues/19>`_

.. _`sec:oeprop_mbis_free_atom_cache`:

Caching the free-atom volumes
"""""""""""""""""""""""""""""

Asking for ``MBIS_VOLUME_RATIOS`` runs one extra SCF per distinct element, on an isolated atom, to
get the volume the ratio divides by. That cost depends on how many elements are present and not at
all on how large the molecule is, so it is a small fraction of a big calculation and a large one of
a small calculation --- 8% of a seven-atom PBE0/aug-cc-pVTZ job, 25% of CH\ :sub:`3`\ Br at
HF/aug-cc-pVTZ. Repeated across a dataset, a finite-difference frequency, or a geometry
optimization, it is the same handful of numbers computed over and over.

A free-atom volume is a property of an element, a level of theory, the basis that element is given,
and the grid and convergence settings. It is never a property of the molecule, so |PSIfour| keeps
the ones it has computed in ``~/.cache/psi4/free_atom_volumes`` and reuses them. This is on by
default and needs no attention; the controls exist for the cases where it does.

Set |globals__mbis_free_atom_cache_path| (or :envvar:`PSI4_FREE_ATOM_CACHE_PATH`) to put the cache
somewhere shared, such as a project directory on a cluster filesystem. Entries are one small JSON
file each, written atomically, so any number of jobs may read and write the directory at once
without locking. Set |globals__mbis_free_atom_cache| (or :envvar:`PSI4_FREE_ATOM_CACHE`) to
``READ`` for workers that should consult a prepared cache but not add to it, to ``WRITE`` to
recompute and refresh every entry, or to ``OFF`` to disable reuse entirely.

Each entry records the full provenance of its number --- element, spin state, reference, method, a
hash of the contracted functions the element was given, every grid and convergence setting, and
every option the input changed --- and a cached value is used only when all of it matches the
current calculation. Nothing is keyed on the *name* of a basis, because two ``basis {}`` blocks can
share a name and differ; a run whose basis cannot be identified this way (an ECP-bearing basis,
which MBIS rejects anyway) simply recomputes. A damaged, truncated, or hand-edited entry is
likewise ignored rather than trusted or raised on.

.. warning::
   Reusing a free-atom volume changes results by the amount to which the reference SCF was itself
   converged --- typically :math:`10^{-11}` in a volume ratio at |globals__d_convergence| of
   :math:`10^{-8}`, since the cached atom was not converged from the same starting guess. If that
   matters, or to guarantee that a whole dataset was divided by identical references, prewarm the
   cache once and run the campaign against it:

   .. code-block:: python

      from psi4.driver.p4util import free_atom_cache

      psi4.set_options({"basis": "aug-cc-pvtz", "scf_type": "df", "d_convergence": 8})
      free_atom_cache.prewarm(["H", "C", "N", "O", "S", "Cl"], "pbe0")

   ``free_atom_cache`` also offers ``list_entries()``, ``clear()``, and ``path()`` for inspecting
   and managing the directory.

