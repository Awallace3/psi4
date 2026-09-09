# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit SCF correction admission. Never sets options or changes a wavefunction."""
from dataclasses import dataclass
import math
from numbers import Real
from psi4 import core


@dataclass(frozen=True)
class CorrectionProvenance:
    """Owned scalar/tuple snapshot, not a live functional or a convergence claim."""
    policy: str
    shift: float
    alpha: float
    beta: float
    components: tuple

    @property
    def response_description(self):
        return ('ALDA response using fixed-GRAC SCF orbitals; no GRAC kernel derivative'
                if self.policy == 'FIXED_GRAC' else 'no SCF asymptotic correction')


def component_definition(component, *, include_cutoff=False):
    """Include overrides: mix data alone cannot detect nonhybrid LibXC tweaks.

    Generic components remain sealable, but are not admitted as canonical PBE0.
    Density cutoff is a numerical grid control, not a canonical functional scale.
    """
    if component is None:
        return None
    libxc = isinstance(component, core.LibXCFunctional)
    definition = (component.name(), component.alpha(), component.omega(),
            component.is_gga(), component.is_meta(), component.is_lrc(), component.is_unpolarized(), libxc,
            tuple(component.get_mix_data()) if libxc else (),
            tuple(sorted(component.query_libxc('XC_HYB_CAM_COEF').items())) if libxc else (),
            tuple(sorted(component.get_tweak().items())) if libxc else ())
    # Canonical admission ignores numerical grid controls; state seals must not.
    return (definition, component.density_cutoff() if libxc else None) if include_cutoff else definition


def correction_state(functional):
    """Read actual attachments even when needs_grac is false; no ambient DFT options."""
    return (functional.grac_shift(), functional.grac_alpha(), functional.grac_beta(),
            (grac_component_definition(functional.grac_x_functional()),
             grac_component_definition(functional.grac_c_functional())))


def grac_component_definition(component):
    if component is None:
        return None
    return (component_definition(component),
            component.density_cutoff() if isinstance(component, core.LibXCFunctional) else None)


SCALAR_KEYS = ('x_alpha', 'x_beta', 'x_omega', 'c_alpha', 'c_omega',
               'c_os_alpha', 'c_ss_alpha', 'vv10_b', 'vv10_c',
               'grac_shift', 'grac_alpha', 'grac_beta', 'ansatz',
               'is_libxc_func', 'needs_vv10', 'needs_grac', 'is_x_lrc', 'is_c_lrc')


def functional_definition(functional, *, include_cutoffs=False):
    return (tuple(getattr(functional, key)() for key in SCALAR_KEYS),
            tuple(tuple(component_definition(c, include_cutoff=include_cutoffs) for c in group)
                  for group in (functional.x_functionals(), functional.c_functionals())),
            correction_state(functional)[3])


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validate_correction(wfn, *, scf_correction='NONE', expected_grac_shift=None,
                        require_canonical=False):
    """Accept only NONE or canonical fixed GRAC (.5,40, LB*.75,VWN*1).

    FIXED_GRAC always requires underlying canonical PBE0 and a current SCF seal.
    NONE preserves the expert producer's non-PBE0 support, not GRAC admission.
    Expected shift is a positive finite Hartree value, compared exactly. It is
    never calculated from an IP/HOMO or inferred from a method/option name.
    """
    if scf_correction not in ('NONE', 'FIXED_GRAC'):
        raise ValueError('unsupported native SCF asymptotic correction policy')
    if scf_correction == 'NONE':
        if expected_grac_shift is not None:
            raise ValueError('NONE requires no expected GRAC shift declaration')
    elif not _finite(expected_grac_shift) or expected_grac_shift <= 0:
        raise ValueError('FIXED_GRAC requires an explicit positive finite expected GRAC shift')
    if not callable(getattr(wfn, 'functional', None)):
        raise ValueError('actual SCF functional state is required for native correction admission')
    functional = wfn.functional()
    if functional is None:
        raise ValueError('actual SCF functional state is required for native correction admission')
    shift, alpha, beta, components = correction_state(functional)
    if not all(_finite(v) for v in (shift, alpha, beta)):
        raise ValueError('actual GRAC controls must be finite')
    if scf_correction == 'NONE':
        if (shift, alpha, beta, components) != (0., .5, 40., (None, None)) or functional.needs_grac():
            raise ValueError('unmodified canonical PBE0/native NONE policy rejects GRAC state')
    else:
        x = core.LibXCFunctional('XC_GGA_X_LB', True)
        x.set_alpha(.75)  # VBase::set_grac_shift for global PBE0, not ambient options
        c = core.LibXCFunctional('XC_LDA_C_VWN', True)
        expected_components = (grac_component_definition(x), grac_component_definition(c))
        if ((shift, alpha, beta) != (float(expected_grac_shift), .5, 40.)
                or components != expected_components or not functional.needs_grac()):
            raise ValueError('fixed GRAC correction mismatch: expected shift/.5/40 and canonical LB*.75/VWN*1')
    if require_canonical or scf_correction == 'FIXED_GRAC':
        canonical = core.SuperFunctional.XC_build('XC_HYB_GGA_XC_PBEH', True)
        scalars, groups, attachments = functional_definition(canonical)
        if scf_correction == 'FIXED_GRAC':
            expected = dict(zip(SCALAR_KEYS, scalars))
            expected.update(grac_shift=float(expected_grac_shift), grac_alpha=.5,
                            grac_beta=40., needs_grac=True)
            scalars = tuple(expected[k] for k in SCALAR_KEYS)
            attachments = expected_components
        if (functional.name().upper() != 'PBE0'
                or functional_definition(functional) != (scalars, groups, attachments)
                or getattr(wfn, '_disp_functor', None) is not None):
            raise ValueError('Atomic response requires unmodified canonical PBE0 underlying the declared correction')
    if scf_correction == 'FIXED_GRAC':
        require_scf_seal(wfn)
    return CorrectionProvenance(scf_correction, float(shift), float(alpha), float(beta), components)


def require_scf_seal(wfn):
    """Verify the existing successful-SCF seal; never manufacture one."""
    from .scf_proc.scf_iterator import _scf_state_signature
    evidence = getattr(wfn, '_scf_convergence_evidence', None)
    if evidence is None:
        raise ValueError('Successful SCF convergence evidence is required; native properties never run SCF')
    diagnostics, signature, basis = evidence
    iteration, delta_e, gradient_norm, e_threshold, d_threshold, rms_norm = diagnostics
    if (iteration < 0 or not all(_finite(v) for v in (delta_e, gradient_norm, e_threshold, d_threshold))
            or not (abs(delta_e) < e_threshold and 0 <= gradient_norm < d_threshold)):
        raise ValueError('SCF convergence stopping diagnostics are inconsistent')
    if basis != wfn.basisset() or signature != _scf_state_signature(wfn):
        raise ValueError('SCF convergence evidence is stale: wavefunction state has changed')
