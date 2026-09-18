# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""The public option front end of the DECLARED_MULTPOLE_AC correction producer.

``isapol_native_ac`` deliberately reads no options: its docstring promises that
nothing in it is inferred from an option name, a method name or a wavefunction,
and a declaration reaches it only as an explicit :class:`AcDeclaration`. That
promise is what makes the module's refusals meaningful, so the option surface is
resolved HERE and the producer is left alone.

What this module does, and what it deliberately does not
--------------------------------------------------------
* :func:`declaration` resolves the ambient ``ATOMIC_AC_*`` options into ONE frozen
  ``AcDeclaration``, validated by that class, BEFORE any orbital is produced. A
  string option cannot carry a declaration, so the eleven fields that make up the
  model's identity each have their own validated option and the resolution is a
  single point of truth shared by the producer and by property admission.
* :func:`run` drives the ALREADY EXPLICIT POST-SCF PRODUCER. It does not integrate
  the correction potential into Psi4's own SCF, and it is not reachable from a
  property request: ``isapol_oeprop`` only ever *admits* orbitals that a prior
  explicit call produced, which is what keeps "no hidden correction iterations
  inside property requests" literally true rather than merely intended.
* The choice is recorded rather than left implicit: the front end drives the
  post-SCF producer. Integrating the potential into core SCF would put a
  non-variational energy behind Psi4's own SCF seal, and the seal is the evidence
  every native property request checks. The post-SCF route instead VERIFIES that
  seal, then invalidates it by an explicitly named mutation and publishes its own
  convergence record, so a corrected state can never be mistaken for a converged
  Kohn-Sham minimum.
* ``shift_damping`` and ``diis_subspace`` are producer solver details that do not
  change the admitted model -- the variational Tozer-Handy constant is re-checked
  at its fixed point to 1e-6 at admission -- so they are NOT given options here.
  Callers who need them keep using ``isapol_native_ac`` directly.

Nothing here computes an ionization potential. ``ATOMIC_AC_IONIZATION_POTENTIAL``
is a declared input in Hartree whose zero default is an undeclared sentinel, the
same convention ``ATOMIC_SCF_EXPECTED_GRAC_SHIFT`` already uses; it is refused,
never replaced by a HOMO eigenvalue or a Delta-SCF estimate.
"""
from numbers import Integral
from psi4 import core
from .isapol_native_ac import (AC_POLICY, AcDeclaration, apply_declared_ac,
                               declared_ac_orbitals)
from . import isapol_logging as _lg

POLICY_OPTION = 'ATOMIC_SCF_ASYMPTOTIC_CORRECTION'


def _lower(value):
    if not isinstance(value, str):
        raise ValueError('declared asymptotic-correction keyword must be a string')
    return value.lower()


def _exact_int(value):
    # A truncating int() here would silently turn a mis-declared 2.7 into 2, and
    # the multipole order is part of the model's identity.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError('declared asymptotic-correction count must be an integer')
    return int(value)


#: ``(option, AcDeclaration field, converter)``. Every identity-bearing field of
#: the declaration appears exactly once, so a new field cannot become reachable
#: without becoming declarable; ``test_isapol_ac_options`` pins both directions
#: and pins each option's built-in default against the field's own default, which
#: is the only thing stopping the C++ surface and the dataclass from drifting into
#: two different models that share a name.
DECLARATION_OPTIONS = (
    ('ATOMIC_AC_IONIZATION_POTENTIAL', 'ionization_potential', float),
    ('ATOMIC_AC_JOIN', 'join', _lower),
    ('ATOMIC_AC_SPLICE_LOWER', 'b1', float),
    ('ATOMIC_AC_SPLICE_UPPER', 'b2', float),
    ('ATOMIC_AC_TANH_SHARPNESS', 'tanh_k', float),
    ('ATOMIC_AC_BRAGG_TABLE', 'bragg_table', _lower),
    ('ATOMIC_AC_FERMI_AMALDI_SCALE', 'fa_scale', float),
    ('ATOMIC_AC_MULTIPOLE_ORDER', 'multipole_order', _exact_int),
    ('ATOMIC_AC_MULTIPOLE_ORIGIN', 'origin', _lower),
    ('ATOMIC_AC_SHIFT_MODE', 'shift_mode', _lower),
    ('ATOMIC_AC_SHIFT_VALUE', 'shift_value', float),
)

#: Producer stopping controls. These are not part of the declaration: two runs
#: that differ only here are the same model, and ``validate_declared_ac`` refuses
#: a record converged more loosely than its own admission limits either way.
PRODUCER_OPTIONS = (
    ('ATOMIC_AC_MAXITER', 'maxiter', _exact_int),
    ('ATOMIC_AC_E_CONVERGENCE', 'energy_threshold', float),
    ('ATOMIC_AC_D_CONVERGENCE', 'gradient_threshold', float),
)

#: Recorded by ``isapol_oeprop`` only under DECLARED_MULTPOLE_AC. Listing them on
#: a NONE or FIXED_GRAC request would read as though they had applied to it.
KEYS = tuple(name for name, _, _ in DECLARATION_OPTIONS + PRODUCER_OPTIONS)


def declaration():
    """Resolve the ambient ``ATOMIC_AC_*`` options into one frozen declaration.

    Raises rather than returning a partial or defaulted model. The undeclared
    ionization potential is refused by name here, before ``AcDeclaration``'s own
    generic positivity check, because "you did not declare an IP" and "you
    declared a nonpositive IP" are different mistakes.
    """
    values = {}
    for name, field, convert in DECLARATION_OPTIONS:
        try:
            values[field] = convert(core.get_global_option(name))
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from None
    if values['ionization_potential'] == 0.:
        raise ValueError(
            f'{AC_POLICY} requires an explicit ATOMIC_AC_IONIZATION_POTENTIAL in Hartree. '
            'Zero is the undeclared sentinel: the correction is ionization-potential '
            'driven and the IP is never computed from a HOMO eigenvalue or a Delta-SCF.')
    return AcDeclaration(**values)


def producer_controls():
    """Resolve the producer's stopping controls; admission bounds them, not this."""
    controls = {}
    for name, field, convert in PRODUCER_OPTIONS:
        try:
            controls[field] = convert(core.get_global_option(name))
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from None
    return controls


def run(wfn):
    """Produce and apply the declared asymptotic correction named by the options.

    This is the explicit producer call. It VERIFIES ``wfn``'s successful-SCF seal,
    iterates the declared correction from that sealed canonical state, then
    REPLACES ``wfn``'s orbitals, density, Fock matrix, eigenvalues and energy and
    invalidates the seal -- because the state is no longer the one Psi4's SCF
    converged, and the reported energy is the plain functional at the corrected
    density, which is not a variational minimum.

    The ``ATOMIC_SCF_ASYMPTOTIC_CORRECTION`` policy must already name
    DECLARED_MULTPOLE_AC. That is not red tape: production and property admission
    then read the SAME resolved declaration from the same options, so a run cannot
    produce one model and later admit its orbitals as another.
    """
    policy = core.get_global_option(POLICY_OPTION)
    if policy != AC_POLICY:
        raise ValueError(
            f'{POLICY_OPTION} is {policy}; producing a declared asymptotic correction '
            f'requires declaring it. Set {POLICY_OPTION} to {AC_POLICY} first, so that '
            'the model produced here is the model a later property request admits.')
    # One log for both stages, at the same verbosity the property request uses.
    # Narration is gated by it; no number below depends on it.
    log = _lg.StageLog(int(core.get_global_option('ATOMIC_PROPERTY_PRINT')))
    record = declared_ac_orbitals(wfn, declaration(), **producer_controls(), log=log)
    return apply_declared_ac(wfn, record, log=log)
