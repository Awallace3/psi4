# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Owned distributed moments, independent of the response solver and LW.

Q has site-major rows, real Racah components 00,10,11c,11s,... (z,x,y
for dipoles), global Cartesian axes and bohr origins. Columns are the exact
ordered effective functions of a declared response BasisRecipe, not necessarily
ISA's density-fit AUX. Q maps fitted *density* coefficients to moments; it
contains neither an electronic minus sign nor a response/neutrality correction.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from .isapol_basis import BasisRecipe, _owned
from .isapol_df_multipoles import df_centre_multipoles

CONVENTION = 'site-major real Racah 00,10,11c,11s,...; global axes; bohr; density moments'


def basis_identity(recipe):
    """Conservative identity including representation, ordered shells and origins."""
    if not isinstance(recipe, BasisRecipe):
        raise TypeError('explicit response AUX BasisRecipe required')
    return hashlib.sha256(repr(recipe).encode()).hexdigest()


def _width(recipe):
    return sum((s.l+1)*(s.l+2)//2 if recipe.representation == 'Cartesian' else 2*s.l+1
               for s in recipe.shells)


@dataclass(frozen=True)
class DistributedMoments:
    """Immutable numeric snapshot with explicit axes and successful model status.

    ``state_sha256`` is None only for a density-independent producer (DF-centre).
    Diagnostics are immutable scalar/tuple pairs, not live native controller
    records. Convergence is a provider declaration, not an accuracy certificate.
    """
    values: np.ndarray
    labels: tuple
    origins_bohr: tuple
    rank: int
    auxiliary: BasisRecipe
    model: str
    provenance: str
    converged: bool
    state_sha256: str | None = None
    diagnostics: tuple = ()
    convention: str = CONVENTION

    def __post_init__(self):
        labels = tuple(self.labels)
        origins = tuple(tuple(float(x) for x in p) for p in self.origins_bohr)
        if (not labels or any(not isinstance(s, str) or not s.strip() for s in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError('unique nonempty site labels required')
        if np.asarray(origins).shape != (len(labels), 3) or not np.isfinite(origins).all():
            raise ValueError('finite bohr origins in site order required')
        if type(self.rank) is not int or not 0 <= self.rank <= 4:
            raise ValueError('distributed moment rank must be 0..4')
        basis_identity(self.auxiliary)
        if origins != self.auxiliary.centres:
            raise ValueError('site order/origins must match response AUX centres')
        if self.convention != CONVENTION:
            raise ValueError('unsupported distributed moment convention')
        if self.converged is not True:
            raise ValueError('distributed moments require a converged model')
        if any(not isinstance(s, str) or not s.strip() for s in (self.model, self.provenance)):
            raise ValueError('model and provenance required')
        if self.state_sha256 is not None and (not isinstance(self.state_sha256, str) or not self.state_sha256):
            raise ValueError('invalid density state identity')
        if np.asarray(self.values).dtype.kind not in 'fiu':
            raise ValueError('Q must contain real numeric values')
        values = _owned(self.values)
        if values.shape != (len(labels)*(self.rank+1)**2, _width(self.auxiliary)):
            raise ValueError('Q shape must match site/rank rows and response AUX columns')
        def immutable(value):
            if isinstance(value, (tuple, list)):
                return tuple(immutable(v) for v in value)
            if isinstance(value, (str, bool, int, float, type(None))):
                return value
            raise TypeError('diagnostics must contain scalar/tuple values')
        diagnostics = tuple((str(k), immutable(v)) for k, v in self.diagnostics)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'labels', labels)
        object.__setattr__(self, 'origins_bohr', origins)
        object.__setattr__(self, 'diagnostics', diagnostics)

    @property
    def auxiliary_sha256(self):
        return basis_identity(self.auxiliary)

    def validate_for(self, auxiliary, sites, rank, state_sha256):
        if (auxiliary != self.auxiliary or tuple(s.label for s in sites) != self.labels
                or tuple(tuple(s.origin) for s in sites) != self.origins_bohr or rank != self.rank):
            raise ValueError('distributed moment site/rank/response AUX identity mismatch')
        if self.state_sha256 is not None and state_sha256 != self.state_sha256:
            raise ValueError('distributed moment density state identity mismatch')

    def anchor_legs(self, coefficients):
        """The sole provider-independent replacement for fit.coefficients @ Q.T."""
        if np.asarray(coefficients).dtype.kind not in 'fiu':
            raise ValueError('real numeric fitted response AUX coefficients required')
        coefficients = np.asarray(coefficients, dtype=float)
        if (coefficients.ndim != 2 or coefficients.shape[1] != self.values.shape[1]
                or not np.isfinite(coefficients).all()):
            raise ValueError('finite fitted response AUX coefficient matrix required')
        legs = coefficients @ self.values.T
        if not np.isfinite(legs).all():
            raise ValueError('nonfinite distributed anchor legs')
        return legs


def analytic_df_moments(auxiliary, sites, rank):
    """Wrap the existing analytic producer without altering a single Q element."""
    result = df_centre_multipoles('df_centre_analytic', auxiliary, sites, rank)
    return DistributedMoments(result.values, result.labels, result.origins, rank, auxiliary,
                              'df_centre_analytic', result.provenance, True,
                              diagnostics=tuple(result.diagnostics.items()))
