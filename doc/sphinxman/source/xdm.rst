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

.. index:: XDM
.. _`sec:xdm`:

XDM: Exchange-Hole Dipole Moment Dispersion Model
==================================================

.. codeauthor:: Alastair Price and Austin M. Wallace
.. sectionauthor:: Alastair Price and Austin M. Wallace

*Module:* :source:`LIBXDM <psi4/src/psi4/libxdm>`

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
[Becke:2005:154104]_, [Becke:2007:154108]_, and [Johnson:2006:174104]_. The
exchange-hole model is that of [Becke:1989:3761]_, and the atomic partitioning
follows [Hirshfeld:1977:129]_; see also [Becke:2006:014104]_. All citations are
defined in :ref:`apdx:bib`.


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

* **Basis-set-dependent parameters.** Because the electron density depends on
  the basis set, the BJ damping parameters :math:`(a_1, a_2)` are fitted per
  functional and basis.

* **Higher-order terms.** The XDM series includes :math:`C_6`, :math:`C_8`,
  *and* :math:`C_{10}` contributions, whereas DFT-D3 uses only :math:`C_6`
  and :math:`C_8`.


Running XDM
~~~~~~~~~~~

An XDM dispersion correction is requested by appending ``-XDM`` to a
supported functional name, analogous to how ``-D3`` or ``-D3BJ`` appends a
Grimme correction.  For example::

   molecule h2o {
       O
       H 1 1.0
       H 1 1.0 2 104.5
   }
   set basis aug-cc-pvtz
   energy('b3lyp-xdm')

``-XDM`` selects the KB49 damping model. Use ``-XDM(KB49)`` explicitly or
``-XDM(LOS-II)`` for the LoS-II refit, which is fitted only for the
functional/basis combinations in :ref:`table:xdmlosiiparams` and
:ref:`table:xdmlosiicpparams`. Each model keeps separate no-CP and
counterpoise fits; the counterpoise set is selected automatically for
``bsse_type='cp'`` and ``'vmfc'`` (see below). Any other combination
requires |scf__xdm_dispersion_parameters|.

``wB97M-XDM(LOS-II)`` is wB97M-V with its VV10 nonlocal correlation turned
off (as in wB97M-D3BJ), and ``wB97X-XDM(LOS-II)`` is the dispersion-free
wB97X. Both have their own free-atom volumes (H--Ar, Ga--Kr; other elements
use the LSDA volumes) but no fitted damping parameters yet, so they require
|scf__xdm_dispersion_parameters|.

The dispersion correction is available after a calculation in the
PSI variable :psivar:`DISPERSION CORRECTION ENERGY`.

.. _`table:xdmfuncs`:

.. table:: Methods available with the -XDM suffix

   +---------------+------------+
   | Method        | Type       |
   +===============+============+
   | B3LYP-XDM     | Hybrid GGA |
   +---------------+------------+
   | B3P86-XDM     | Hybrid GGA |
   +---------------+------------+
   | B3PW91-XDM    | Hybrid GGA |
   +---------------+------------+
   | PBE0-XDM      | Hybrid GGA |
   +---------------+------------+
   | BHandHLYP-XDM | Hybrid GGA |
   +---------------+------------+
   | CAM-B3LYP-XDM | Range-sep. |
   +---------------+------------+
   | HSE06-XDM     | Range-sep. |
   +---------------+------------+
   | LC-wPBE-XDM   | Range-sep. |
   +---------------+------------+
   | B97-1-XDM     | Hybrid GGA |
   +---------------+------------+
   | BLYP-XDM      | GGA        |
   +---------------+------------+
   | B86BPBE-XDM   | GGA        |
   +---------------+------------+
   | BP86-XDM      | GGA        |
   +---------------+------------+
   | PW86PBE-XDM   | GGA        |
   +---------------+------------+
   | PBE-XDM       | GGA        |
   +---------------+------------+
   | TPSS-XDM      | Meta-GGA   |
   +---------------+------------+
   | HF-XDM        | HF         |
   +---------------+------------+


Damping Parameters
~~~~~~~~~~~~~~~~~~

The BJ damping parameters :math:`(a_1, a_2)` are looked up automatically from
an internal table keyed by ``functional/basis``.  Because XDM coefficients are
density-derived, :math:`(a_1, a_2)` vary with both functional and basis set.

.. _`table:xdmparams`:

.. table:: Selected no-CP XDM BJ damping parameters (a1, a2 in angstrom)

   +-------------------------------+--------+-----------+
   | Functional / Basis            | a1     | a2 (Ang)  |
   +===============================+========+===========+
   | B3LYP / aug-cc-pVTZ           | 0.5049 | 1.8215    |
   +-------------------------------+--------+-----------+
   | B3LYP / aug-cc-pVDZ           | 0.7259 | 1.3140    |
   +-------------------------------+--------+-----------+
   | PBE0 / aug-cc-pVTZ            | 0.5884 | 2.0642    |
   +-------------------------------+--------+-----------+
   | PBE0 / aug-cc-pVDZ            | 0.4681 | 2.7183    |
   +-------------------------------+--------+-----------+
   | PBE / aug-cc-pVTZ             | 0.7110 | 1.6463    |
   +-------------------------------+--------+-----------+
   | PBE / aug-cc-pVDZ             | 0.5556 | 2.3610    |
   +-------------------------------+--------+-----------+

.. _`table:xdmlosiiparams`:

.. table:: Complete LoS-II XDM BJ damping parameters (a1, a2 in angstrom)

   +-------------------------------+----------+-----------+
   | Functional / Basis            | a1       | a2 (Ang)  |
   +===============================+==========+===========+
   | B3LYP / aug-cc-pVTZ           | 0.344500 | 2.304241  |
   +-------------------------------+----------+-----------+
   | B3LYP / aug-cc-pVDZ           | 0.362043 | 2.452093  |
   +-------------------------------+----------+-----------+
   | PBE0 / aug-cc-pVDZ            | 0.068267 | 3.963874  |
   +-------------------------------+----------+-----------+

.. _`table:xdmlosiicpparams`:

.. table:: LoS-II counterpoise XDM BJ damping parameters, used with ``bsse_type='cp'`` (a1, a2 in angstrom)

   +-------------------------------+----------+-----------+
   | Functional / Basis            | a1       | a2 (Ang)  |
   +===============================+==========+===========+
   | B3LYP / aug-cc-pVDZ           | 0.3880   | 2.1781    |
   +-------------------------------+----------+-----------+
   | PBE0 / aug-cc-pVDZ            | 0.1592   | 3.3865    |
   +-------------------------------+----------+-----------+

For unlisted combinations, supply :math:`(a_1, a_2)` through
|scf__xdm_dispersion_parameters|. The full table is in
``psi4/driver/procrouting/empirical_disp/xdm_params.py``.

Counterpoise-corrected XDM energies are supported with ``bsse_type='cp'``
and ``'vmfc'``; see **Ghost Atoms** for how ghost centres are treated::

   energy('b3lyp-xdm', bsse_type=['cp', 'nocp'])

Counterpoise-corrected XDM gradients, and counterpoise XDM energies through a
CBS specification, raise ``NotImplementedError``.

Counterpoise and no-CP interaction energies use different damping
parameters. For ``bsse_type='cp'`` or ``'vmfc'`` the many-body driver sets
|scf__xdm_counterpoise| on every task, and the parameters come from the
model's counterpoise fit (:ref:`table:xdmlosiicpparams`), made against
counterpoise-corrected LoS-II interaction energies with ghost-centre
superposition::

   energy('b3lyp-xdm(los-ii)', bsse_type='cp')

KB49 has no counterpoise fit, so counterpoise ``-XDM`` and ``-XDM(KB49)``
energies require |scf__xdm_dispersion_parameters|, as does ``dft-xdm(sapt)``,
whose delta-DFT monomers are counterpoise subsystems. Because the cluster
tasks are shared, requesting ``'nocp'`` together with ``'cp'`` or ``'vmfc'``
also requires explicit parameters; otherwise run them separately. When
running counterpoise subsystem jobs outside the many-body driver (for
example, as individual QCFractal singlepoints), set
|scf__xdm_counterpoise| yourself.


Custom Parameters
~~~~~~~~~~~~~~~~~

If your functional/basis combination is not in the built-in table, or if you
want to override the default parameters, use the |scf__xdm_dispersion_parameters|
keyword.  This option accepts a two-element array ``[a1, a2]`` where
:math:`a_2` is in angstroms::

   set xdm_dispersion_parameters [0.5, 1.0]
   energy('b3lyp-xdm')

When |scf__xdm_dispersion_parameters| is set, it overrides any automatic
functional/basis lookup. Automatic lookup requires the fitted functional's
original exact-exchange fraction. With explicit parameters, modified global
exact exchange is supported only for systems whose atoms have
:math:`Z \leq 10`; modified range-separation parameters and unknown
range-separated functionals are not supported.

XDM requires an all-electron density, so calculations using effective-core
potentials are rejected. XDM also cannot be combined with another empirical
dispersion correction or a functional that includes VV10 nonlocal correlation.


Ghost Atoms
~~~~~~~~~~~

Ghost atoms are part of the Hirshfeld promolecule but not of the pair sum.
Each ghost centre keeps the share of the density its free atom would claim, so
a counterpoise monomer partitions its own density the way the parent cluster
does, and its real atoms do not absorb the density that ghost basis functions
place on the partner. (Dropping ghosts from the promolecule hands that tail to
the real atoms, where the :math:`r^4`- and :math:`r^6`-weighted moments inflate
:math:`C_{10}` and :math:`R_c`. The resulting XDM interaction energy grows too
attractive with separation.)

That alone does not make a counterpoise interaction energy decay correctly. In
the cluster, each atom of monomer A also picks up part of the partner's density
:math:`\int h_a \rho_B`. That pickup is weighted by up to :math:`r^6` in
:math:`\langle M_3^2 \rangle` and changes the damping radius of every A--A
pair. It is absent from the counterpoise monomer, so
:math:`E_{AB} - E_A - E_B` keeps an intramonomer term that decays much more
slowly than :math:`R^{-6}`. With large-:math:`a_1` parameter sets it changes
the sign of the dispersion interaction energy at long range.

The many-body driver therefore superposes fragment moments. For a subsystem
computed with ghost fragments, it adds to each real atom the moments that each
ghost fragment's own counterpoise job deposited on that atom's centre. It then
re-evaluates the pair sum, so the subtraction leaves only intermolecular pairs
and real density deformation. On benzene-dimer and water-dimer scans to 15 Å,
this counterpoise dispersion interaction energy tracks the intermolecular
pair sum and decays as :math:`R^{-6}`. The corrections appear in the output
under ``XDM Counterpoise: Superposed Ghost-Centre Moments``.

A single calculation with ``Gh()`` atoms has no partner density to superpose,
so it reports the bare counterpoise-monomer energy. The ``XDM C6 COEFFICIENTS``
matrix returned in the wavefunction has shape ``(N_real, N_real)``, where
``N_real`` is the number of non-ghost atoms::

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

After an XDM-corrected computation, the following PSI variables are set. The
coefficient and pair-energy matrices have shape ``(N_real, N_real)`` as
described under **Ghost Atoms**.

.. table:: PSI variables set by the XDM module

   +---------------------------------------+-----------------------------------------------------------+
   | Variable                              | Description                                               |
   +=======================================+===========================================================+
   | :psivar:`DISPERSION CORRECTION        | Total XDM dispersion energy [Eh]                          |
   | ENERGY`                               |                                                           |
   +---------------------------------------+-----------------------------------------------------------+
   | ``XDM ENERGY``                        | Same as above (alias)                                     |
   +---------------------------------------+-----------------------------------------------------------+
   | ``XDM C6 COEFFICIENTS``               | Pairwise :math:`C_6` coefficients                         |
   +---------------------------------------+-----------------------------------------------------------+
   | ``XDM C8 COEFFICIENTS``               | Pairwise :math:`C_8` coefficients                         |
   +---------------------------------------+-----------------------------------------------------------+
   | ``XDM C10 COEFFICIENTS``              | Pairwise :math:`C_{10}` coefficients                      |
   +---------------------------------------+-----------------------------------------------------------+
   | ``XDM RC COEFFICIENTS``               | Pairwise critical radii :math:`R_{c,ij}`                  |
   +---------------------------------------+-----------------------------------------------------------+
   | ``XDM PAIRWISE ENERGY``               | Symmetric pair energies; matrix sums to total energy      |
   +---------------------------------------+-----------------------------------------------------------+
   | ``XDM ATOMIC PROPERTIES``             | ``(N_atom, 8)``, ghosts included: true Z, real flag,      |
   |                                       | charge, volume, :math:`\langle M_1^2\rangle`,             |
   |                                       | :math:`\langle M_2^2\rangle`,                             |
   |                                       | :math:`\langle M_3^2\rangle`, polarizability              |
   +---------------------------------------+-----------------------------------------------------------+
   | ``XDM DAMPING PARAMETERS``            | ``(1, 2)``: :math:`a_1` and :math:`a_2` [bohr]            |
   +---------------------------------------+-----------------------------------------------------------+

These can be accessed from the wavefunction object::

   e, wfn = energy('b3lyp-xdm', return_wfn=True)
   disp = wfn.variable('DISPERSION CORRECTION ENERGY')
   c6   = wfn.variable('XDM C6 COEFFICIENTS')


Options
~~~~~~~

.. include:: autodir_options_c/scf__xdm_dispersion_parameters.rst

Keywords summarized:

.. table:: XDM-related options

   +----------------------------------+-----------------------------------------------------------------------------------------------+
   | Keyword                          | Description                                                                                   |
   +==================================+===============================================================================================+
   | |scf__xdm_dispersion_parameters| | ``[a1, a2_angstrom]``. Override automatic BJ parameter lookup. Two-element array.             |
   +----------------------------------+-----------------------------------------------------------------------------------------------+
   | |scf__dft_spherical_points|      | Number of angular Lebedev points for the DFT grid. Affects the XDM integration accuracy.      |
   +----------------------------------+-----------------------------------------------------------------------------------------------+
   | |scf__dft_radial_points|         | Number of radial points for the DFT grid. Affects the XDM integration accuracy.               |
   +----------------------------------+-----------------------------------------------------------------------------------------------+
   | |scf__basis|                     | Orbital basis set. Used with the functional name to look up fitted :math:`(a_1, a_2)` values. |
   +----------------------------------+-----------------------------------------------------------------------------------------------+


How It Works Internally
~~~~~~~~~~~~~~~~~~~~~~~

1. The SCF converges normally (without dispersion).
2. After convergence, the XDM module integrates Hirshfeld-weighted atomic
   properties (exchange-hole multipole moments and effective volumes) from
   the converged density on the DFT grid. When the SCF's own XC grid is the
   same grid (same basis and grid options), XDM integrates on it instead of
   building another; |scf__xdm_grid_reuse| turns this off.
3. Pairwise :math:`C_6`, :math:`C_8`, :math:`C_{10}` dispersion coefficients
   are assembled from these atomic properties.
4. The BJ-damped dispersion energy (Eq. :eq:`XDMdisp`) is evaluated and added
   to the SCF total energy.

Because XDM is a post-SCF correction, the SCF cycles themselves are
unaffected by the dispersion correction.  The XDM energy is added once
after convergence.


Derivatives
~~~~~~~~~~~

XDM gradients are available and are evaluated by finite differences of the
complete XDM-corrected energy::

   grad = psi4.gradient("b3lyp-xdm")

In mixed many-body ``levels`` and CBS specifications, only XDM levels or CBS
components use finite differences; non-XDM components retain analytic
gradients when available. A structured or callable ``dft_functional`` whose
dispersion metadata has ``type: "xdm"`` follows the same routing. Explicitly
requesting an analytic derivative with ``dertype=1`` is rejected; omit
``dertype`` or use ``dertype=0``.

Finite-difference XDM gradients automatically use Psi4's standard ``(1, 0)``
SCF convergence criteria. Each XDM CBS component receives the same criteria
before its finite-difference tasks are constructed. An explicit
module-qualified CBS stage option such as ``scf__e_convergence`` takes
precedence; an unqualified ``e_convergence`` does not suppress the tighter
SCF-local finite-difference criterion.

Analytic XDM gradients are *not* implemented.  Because every XDM ingredient
--- the exchange-hole multipole moments, the Hirshfeld atomic volumes, the
polarizabilities, the :math:`C_6`/:math:`C_8`/:math:`C_{10}` coefficients, and
the damping radius :math:`R_{\mathrm{vdW}}` --- is a functional of the
converged density, an analytic derivative requires the response of all of
those quantities to nuclear displacement.  Finite differences of the energy
capture these terms by construction.

.. warning:: Do not approximate the XDM gradient by differentiating only the
   pair distances :math:`R_{ij}` while holding the dispersion coefficients and
   damping radii fixed.  That form is accurate to about 0.1% between
   well-separated closed-shell fragments, but for bonded atom pairs the
   coefficient-response terms dominate: the frozen-coefficient derivative can
   have the wrong sign and an error of order 100% of the true value.

Because the gradient is a finite difference of energies, the cost is
:math:`6N_{\mathrm{atom}}` (or fewer, when point-group symmetry reduces the
number of displacements) XDM-corrected SCF calculations.  Since XDM quantities
are integrated on the DFT grid, use a dense grid and tight SCF convergence so
that grid noise does not contaminate the difference, for example::

   psi4.set_options({
       "dft_spherical_points": 590,
       "dft_radial_points": 99,
       "scf__e_convergence": 1e-10,
       "scf__d_convergence": 1e-8,
   })

XDM Hessians and property calculations are not implemented and raise
``NotImplementedError``.  Counterpoise-corrected gradients are blocked
because the counterpoise superposition of ghost-centre moments has no
derivative implementation.
