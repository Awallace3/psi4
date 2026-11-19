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

.. index:: DFTNL
.. _`sec:dftnl`:


DFT-NL
======

.. codeauthor:: Daniel G. A. Smith and Holger Kruse
.. sectionauthor:: Holger Kruse

Non-local (NL), density based correlation energy from the VV10 kernel can be added
to arbitrary functionals.

.. math:: E_{DFT-NL}=E_{DFT}+E_{NL}

For pre-defined functionals (see Functional overview in :ref:`this Table <table:dft_all>` ) it is sufficient to add `-NL` to
the functional name::

    energy('b3lyp-nl')

Modification of the parameters `b` and `C` is done setting |scf__dft_vv10_b| and |scf__dft_vv10_c|. The `C` is usually left unchanged and the originally proposed
value of `C=0.0093` is used.

Adding |scf__dft_vv10_b| to any functional activates the calculation of the VV10 kernel. A BLYP-NL calculation can be set as follows::

    set DFT_VV10_B 4.0
    energy('blyp')

The default `C` parameter will be used.

Similar to |scf__dft_dispersion_parameters| the tuple |scf__nl_dispersion_parameters| can used::

    set NL_DISPERSION_PARAMTERS [4.0]
    energy('blyp')

which is equivalent to the example above.

Further examples can be found in the respective :source:`regression test <tests/dft-vv10/input.dat>`

post-SCF time savings
~~~~~~~~~~~~~~~~~~~~~

Substantial time-savings for energy calculations are available by evaluating the VV10 kernel only at the converged electron density, i.e. in a post-SCF fashion.
The deviations from the fully self-consistent treatment are usually minimal. To activate this set |scf__dft_vv10_postscf| to `true`.

Analytic gradients
~~~~~~~~~~~~~~~~~~

Analytic nuclear gradients are available for DFT-NL with RKS and UKS
references, including meta-GGA hosts such as wB97M-V. The gradient
includes the response of the VV10 kernel to the density; the response of
the NL grid points and partition weights to nuclear displacement is
neglected. The resulting finite-grid error depends on the molecule and grid;
this is not an exact derivative of the moving-grid energy. The gradient assumes a self-consistent
VV10 treatment; |scf__dft_vv10_postscf| is therefore rejected by the VV10
gradient evaluator. Analytic
Hessians and TDDFT are not available for VV10 functionals.

Optional dimer gradient in DFT-VV10(SAPT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With |sapt__sapt_dft_ddft_gradient| enabled, ``energy('dft-vv10(sapt)')``
also stores ``SAPT(DFT) DFT DIMER GRADIENT`` on the returned wavefunction
and in the global QCVariables. This is the unshifted, supermolecular DFT
dimer gradient, in hartree/bohr; a force is its negative. It is **not** the
gradient of the CP-corrected interaction energy, a SAPT component, or the
delta-DFT correction. No monomer or ghost-center gradient is subtracted.
For VV10 it has the same density-response-only approximation described above.
Post-SCF VV10 is rejected for this option.

The option is off by default, does not change the energy decomposition, and
requires |sapt__sapt_dft_do_ddft| with a non-HF functional.
``DFT-VV10(SAPT)`` already enables that dimer SCF.
