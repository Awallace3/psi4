.. #
.. # @BEGIN LICENSE
.. #
.. # Psi4: an open-source quantum chemistry software package
.. #
.. # Copyright (c) 2007-2025 The Psi4 Developers.
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

.. index:: XDM
.. _`sec:xdm`:

XDM: Exchange-Hole Dipole Moment Dispersion Model
==================================================

.. codeauthor:: Alistair Price, Austin M. Wallace, Opus-4.6 and GPT-5.3-Codex
.. sectionauthor:: Alistair Price and Austin M. Wallace

*Module:* :ref:`Samples <apdx:testSuitexdm>`

Theory
~~~~~~

The exchange-hole dipole moment (XDM) model [Becke:2005:154104]_
[Becke:2007:154108]_ is a nonempirical approach to computing dispersion
interactions from first principles.  Unlike Grimme's -D2/-D3/-D4 methods, which
rely on precomputed atomic parameters and geometric coordination numbers, XDM
derives pairwise dispersion coefficients directly from the electron density of
the system at hand.

The key insight is that the instantaneous dipole moment arising between a
reference electron and its exchange hole provides a natural source of the
London dispersion interaction.  By computing the second moment of this
exchange-hole dipole (and its higher multipole generalizations) from the
converged wavefunction, one obtains system-specific :math:`C_6`,
:math:`C_8`, and :math:`C_{10}` coefficients that respond to the chemical
environment without fitted elemental parameters.

The XDM dispersion energy takes the form

.. math:: E_{\text{disp}} = -\sum_{n=6,8,10} \sum_{i<j}
   \frac{C_{n,ij}\, f_n(R_{ij})}{R_{ij}^n}
   :label: XDMdisp

where :math:`f_n(R_{ij})` is the Becke--Johnson (BJ) damping function
[Johnson:2006:174104]_

.. math:: f_n(R_{ij}) = \frac{R_{ij}^n}{R_{ij}^n + (a_1\, R_{c,ij} + a_2)^n}
   :label: BJdamp

with the critical interatomic distance

.. math:: R_{c,ij} = \left(\frac{C_{8,ij}}{C_{6,ij}}\right)^{1/2}\!,\quad
   \left(\frac{C_{10,ij}}{C_{8,ij}}\right)^{1/2}\!,\quad
   \left(\frac{C_{10,ij}}{C_{6,ij}}\right)^{1/4}
   \qquad\text{(averaged)}

The dispersion coefficients are constructed from Hirshfeld-partitioned
[Hirshfeld:1977:129]_ atomic polarizabilities and multipole moments of the
exchange hole.  The atomic polarizabilities are obtained by scaling free-atom
values by the ratio of the atom-in-molecule volume to the free-atom volume
[Becke:2006:014104]_.  The moments :math:`\langle M_l^2\rangle_i` are
integrated from the exchange-hole dipole via the Becke--Roussel (BR) model
[Becke:1989:3761]_.

The only adjustable parameters are :math:`a_1` and :math:`a_2` in the damping
function (Eq. :eq:`BJdamp`), which are fitted to benchmark noncovalent
interaction data for each combination of exchange--correlation functional
and basis set. The XDM implementation in |PSIfour| stores fitted
:math:`(a_1, a_2)` values for several functional/basis combinations
(see :ref:`table:xdmparams`).

For further details on the theoretical derivation, see
[Becke:2005:154104]_, [Becke:2007:154108]_, and [Johnson:2006:174104]_.

.. rubric:: Key references

.. [Becke:2005:154104] A. D. Becke and E. R. Johnson,
   *J. Chem. Phys.* **122**, 154104 (2005).
.. [Becke:2007:154108] A. D. Becke and E. R. Johnson,
   *J. Chem. Phys.* **127**, 154108 (2007).
.. [Becke:2006:014104] A. D. Becke and E. R. Johnson,
   *J. Chem. Phys.* **124**, 014104 (2006).
.. [Johnson:2006:174104] E. R. Johnson and A. D. Becke,
   *J. Chem. Phys.* **124**, 174104 (2006).
.. [Becke:1989:3761] A. D. Becke and M. R. Roussel,
   *Phys. Rev. A* **39**, 3761 (1989).
.. [Hirshfeld:1977:129] F. L. Hirshfeld,
   *Theor. Chim. Acta* **44**, 129 (1977).


Differences from DFTD3/DFTD4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

XDM differs from the Grimme family of corrections (see :ref:`sec:dftd3`) in
several important ways:

* **Density-dependent.** XDM requires a converged electron density and is
  therefore computed *post-SCF*, after the self-consistent field procedure
  finishes.  In contrast, -D2/-D3/-D4 corrections depend only on the
  molecular geometry and can be evaluated before or during the SCF.

* **No external program.** The XDM correction is computed entirely within
  |PSIfours| own C++ codebase (``libxdm``).  No external executables or
  Python packages are required.

* **Basis-set-dependent parameters.** Because the electron density (and thus
  the exchange-hole moments) depend on the basis set, the BJ damping
  parameters :math:`(a_1, a_2)` are fitted per functional *and* per basis
  set.  Separate parameter sets are available for counterpoise (CP) and
  non-counterpoise (noCP) workflows.

* **Higher-order terms.** The XDM series includes :math:`C_6`, :math:`C_8`,
  *and* :math:`C_{10}` contributions, whereas DFT-D3 uses only :math:`C_6`
  and :math:`C_8`.


Running XDM
~~~~~~~~~~~

An XDM dispersion correction is requested by appending ``-XDM`` to a
supported DFT functional name, analogous to how ``-D3`` or ``-D3BJ``
appends a Grimme correction.  For example::

   molecule h2o {
       O
       H 1 1.0
       H 1 1.0 2 104.5
   }
   set basis aug-cc-pvtz
   energy('b3lyp-xdm')

The dispersion correction is available after a calculation in the
PSI variable :psivar:`DISPERSION CORRECTION ENERGY`.

.. _`table:xdmfuncs`:

.. table:: DFT functionals available with the -XDM suffix

   +-------------------+-------------+
   | Functional        | Type        |
   +===================+=============+
   | B3LYP-XDM        | Hybrid GGA  |
   +-------------------+-------------+
   | PBE0-XDM         | Hybrid GGA  |
   +-------------------+-------------+
   | BHandHLYP-XDM    | Hybrid GGA  |
   +-------------------+-------------+
   | CAM-B3LYP-XDM    | Range-sep.  |
   +-------------------+-------------+
   | LC-wPBE-XDM      | Range-sep.  |
   +-------------------+-------------+
   | B97-1-XDM        | Hybrid GGA  |
   +-------------------+-------------+
   | BLYP-XDM         | GGA         |
   +-------------------+-------------+
   | PW86PBE-XDM      | GGA         |
   +-------------------+-------------+
   | PBE-XDM          | GGA         |
   +-------------------+-------------+


Damping Parameters
~~~~~~~~~~~~~~~~~~

The BJ damping parameters :math:`(a_1, a_2)` are looked up automatically from
an internal table keyed by ``functional/basis``.  Because XDM coefficients are
density-derived, :math:`(a_1, a_2)` vary with both functional and basis set.

.. _`table:xdmparams`:

.. table:: Fitted XDM BJ damping parameters (a1, a2 in angstrom) for selected functional/basis combinations

   +-------------------------------+------+-------+-----------+
   | Functional / Basis            | CP?  | a1    | a2 (Ang)  |
   +===============================+======+=======+===========+
   | B3LYP / aug-cc-pVTZ           | CP   | 0.539 | 1.707     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / aug-cc-pVTZ           | noCP | 0.299 | 2.455     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / aug-cc-pVDZ           | CP   | 0.541 | 1.708     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / aug-cc-pVDZ           | noCP | 0.594 | 1.708     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / cc-pVTZ               | CP   | 0.533 | 1.709     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / cc-pVTZ               | noCP | 0.475 | 2.034     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / cc-pVDZ               | CP   | 0.503 | 1.705     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / cc-pVDZ               | noCP | 0.200 | 3.125     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / 6-31+G*               | CP   | 0.539 | 1.707     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / 6-31+G*               | noCP | 0.249 | 2.691     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / 6-31+G**              | CP   | 0.544 | 1.707     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / 6-31+G**              | noCP | 0.226 | 2.762     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / 6-311+G(2d,2p)        | CP   | 0.541 | 1.706     |
   +-------------------------------+------+-------+-----------+
   | B3LYP / 6-311+G(2d,2p)        | noCP | 0.565 | 1.708     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / aug-cc-pVDZ            | CP   | 0.676 | 1.709     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / aug-cc-pVDZ            | noCP | 0.757 | 1.710     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / cc-pVTZ                | CP   | 0.662 | 1.709     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / cc-pVTZ                | noCP | 0.337 | 2.867     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / cc-pVDZ                | CP   | 0.648 | 1.707     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / cc-pVDZ                | noCP | 0.000 | 4.226     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / 6-31+G*                | CP   | 0.666 | 1.708     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / 6-31+G*                | noCP | 0.000 | 3.942     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / 6-31+G**               | CP   | 0.673 | 1.709     |
   +-------------------------------+------+-------+-----------+
   | PBE0 / 6-31+G**               | noCP | 0.000 | 3.958     |
   +-------------------------------+------+-------+-----------+
   | PBE / aug-cc-pVTZ             | CP   | 0.641 | 1.707     |
   +-------------------------------+------+-------+-----------+
   | PBE / aug-cc-pVTZ             | noCP | 0.659 | 1.708     |
   +-------------------------------+------+-------+-----------+
   | PBE / aug-cc-pVDZ             | CP   | 0.646 | 1.707     |
   +-------------------------------+------+-------+-----------+
   | PBE / aug-cc-pVDZ             | noCP | 0.719 | 1.705     |
   +-------------------------------+------+-------+-----------+
   | PBE / 6-31+G*                 | CP   | 0.638 | 1.706     |
   +-------------------------------+------+-------+-----------+
   | PBE / 6-31+G*                 | noCP | 0.699 | 1.710     |
   +-------------------------------+------+-------+-----------+
   | PBE / 6-31+G**                | CP   | 0.645 | 1.707     |
   +-------------------------------+------+-------+-----------+
   | PBE / 6-31+G**                | noCP | 0.704 | 1.709     |
   +-------------------------------+------+-------+-----------+

Not all functional/basis combinations are tabulated.  For unlisted
combinations, supply :math:`(a_1, a_2)` manually through
|scf__xdm_dispersion_parameters|.

The full parameter table is maintained in the source file
``psi4/driver/procrouting/empirical_disp/xdm_params.py``.


Counterpoise and Non-Counterpoise Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An important distinction in the XDM parameter table is between
counterpoise-corrected (CP) and non-counterpoise (noCP) parameterizations.
Because BSSE affects the electron density (and hence XDM moments and
coefficients), the optimal :math:`(a_1, a_2)` values differ between
the two workflows.

When you request a BSSE-corrected interaction energy with ``bsse_type='cp'``,
|PSIfour| automatically selects the CP-fitted parameters for the XDM
damping.  For a non-counterpoise calculation (``bsse_type='nocp'``), the noCP
parameters are used.  For a standalone single-molecule calculation (no
``bsse_type``), the noCP parameters are used by default.

For example, a CP-corrected interaction energy::

   dimer = psi4.geometry("""
   0 1
   N -1.578718 -0.046611  0.000000
   H -2.158621  0.136396 -0.809565
   H -2.158621  0.136396  0.809565
   H -0.849471  0.658193  0.000000
   --
   0 1
   O  2.350625  0.111469  0.000000
   H  2.680398 -0.373741 -0.758561
   H  2.680398 -0.373741  0.758561
   units angstrom
   """)
   set basis sto-3g
   energy('b3lyp-xdm', bsse_type='cp')

And the noCP counterpart::

   energy('b3lyp-xdm', bsse_type='nocp')


Custom Parameters
~~~~~~~~~~~~~~~~~

If your functional/basis combination is not in the built-in table, or if you
want to override the default parameters, use the |scf__xdm_dispersion_parameters|
keyword.  This option accepts a two-element array ``[a1, a2]`` where
:math:`a_2` is in angstroms::

   set xdm_dispersion_parameters [0.5, 1.0]
   energy('b3lyp-xdm')

When |scf__xdm_dispersion_parameters| is set, it overrides any automatic
functional/basis lookup.


Ghost Atoms
~~~~~~~~~~~

XDM correctly handles ghost atoms (used in counterpoise calculations).
The dispersion coefficients and pairwise energies are computed only for
real atoms; ghost atoms contribute to the SCF density but are excluded
from the XDM summation.  The ``XDM C6 COEFFICIENTS`` matrix returned
in the wavefunction has shape ``(N_real, N_real)`` where ``N_real`` is
the number of non-ghost atoms::

   mol = psi4.geometry("""
   0 1
   Gh(O)  -1.551  -0.115  0.000
   Gh(H)  -1.934   0.763  0.000
   Gh(H)  -0.600   0.041  0.000
   --
   0 1
   O   1.351   0.111  0.000
   H   1.680  -0.374 -0.759
   H   1.680  -0.374  0.759
   units angstrom
   """)
   set basis sto-3g
   set xdm_dispersion_parameters [0.5, 1.0]
   e, wfn = energy('b3lyp-xdm', return_wfn=True)
   # XDM C6 matrix is (3, 3) for the 3 real atoms
   print(wfn.variable('XDM C6 COEFFICIENTS').shape)


PSI Variables
~~~~~~~~~~~~~

After an XDM-corrected computation, the following PSI variables are set:

.. table:: PSI variables set by the XDM module

   +--------------------------------------+-----------------------------------------------------------+
   | Variable                             | Description                                               |
   +======================================+===========================================================+
   | :psivar:`DISPERSION CORRECTION       | Total XDM dispersion energy [Eh]                          |
   | ENERGY`                              |                                                           |
   +--------------------------------------+-----------------------------------------------------------+
   | ``XDM ENERGY``                       | Same as above (alias)                                     |
   +--------------------------------------+-----------------------------------------------------------+
   | ``XDM C6 COEFFICIENTS``              | Pairwise :math:`C_6` coefficients (Matrix, N x N)         |
   +--------------------------------------+-----------------------------------------------------------+
   | ``XDM C8 COEFFICIENTS``              | Pairwise :math:`C_8` coefficients (Matrix, N x N)         |
   +--------------------------------------+-----------------------------------------------------------+
   | ``XDM C10 COEFFICIENTS``             | Pairwise :math:`C_{10}` coefficients (Matrix, N x N)      |
   +--------------------------------------+-----------------------------------------------------------+
   | ``XDM RC COEFFICIENTS``              | Pairwise critical radii :math:`R_{c,ij}` (Matrix, N x N)  |
   +--------------------------------------+-----------------------------------------------------------+
   | ``XDM PAIRWISE ENERGY``              | Pairwise dispersion energies (Matrix, N x N)              |
   +--------------------------------------+-----------------------------------------------------------+

These can be accessed from the wavefunction object::

   e, wfn = energy('b3lyp-xdm', return_wfn=True)
   disp = wfn.variable('DISPERSION CORRECTION ENERGY')
   c6   = wfn.variable('XDM C6 COEFFICIENTS')


Options
~~~~~~~

.. include:: autodir_options_c/scf__xdm_dispersion_parameters.rst

Keywords summarized:

.. table:: XDM-related options

   +------------------------------------------+-----------------------------------------------------+
   | Keyword                                  | Description                                         |
   +==========================================+=====================================================+
   | |scf__xdm_dispersion_parameters|         | ``[a1, a2_angstrom]``. Override automatic            |
   |                                          | BJ parameter lookup. Two-element array.              |
   +------------------------------------------+-----------------------------------------------------+
   | |scf__dft_spherical_points|               | Number of angular Lebedev points for the DFT         |
   |                                          | grid.  Affects the XDM integration accuracy.         |
   +------------------------------------------+-----------------------------------------------------+
   | |scf__dft_radial_points|                  | Number of radial points for the DFT grid.            |
   |                                          | Affects the XDM integration accuracy.                |
   +------------------------------------------+-----------------------------------------------------+
   | |scf__basis|                              | Orbital basis set. Used together with the functional |
   |                                          | name to look up fitted :math:`(a_1, a_2)` values.   |
   +------------------------------------------+-----------------------------------------------------+


How It Works Internally
~~~~~~~~~~~~~~~~~~~~~~~

1. The SCF converges normally (without dispersion).
2. After convergence, the XDM module integrates Hirshfeld-weighted atomic
   properties (exchange-hole multipole moments and effective volumes) from
   the converged density on the DFT grid.
3. Pairwise :math:`C_6`, :math:`C_8`, :math:`C_{10}` dispersion coefficients
   are assembled from these atomic properties.
4. The BJ-damped dispersion energy (Eq. :eq:`XDMdisp`) is evaluated and added
   to the SCF total energy.
5. Gradients are also available: the XDM gradient is computed from the
   pairwise energy expression at fixed coefficients and added to the SCF
   gradient.

Because XDM is a post-SCF correction, the SCF cycles themselves are
unaffected by the dispersion correction.  The XDM energy is added once
after convergence.
