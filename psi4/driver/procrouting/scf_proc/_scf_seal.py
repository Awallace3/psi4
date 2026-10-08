# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Functional definitions hashed into the restricted-C1 SCF seal.

Kept beside ``scf_iterator`` so the SCF finalize step never imports ISA-Pol modules;
``isapol_native_correction`` re-exports these helpers unchanged.
"""
from psi4 import core


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
