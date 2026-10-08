# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Owned restricted-C1 state admission for the native atomic-property backends.

No SCF, operator construction or frequency solve happens here.
"""
import numpy as np

from psi4 import core


def native_restricted_state_from_wavefunction(wavefunction, *, caller_converged,
                                            max_bytes=512*1024**2):
    """Validate and own a restricted C1 state without constructing operators.

    State-only admission, not SCF convergence verification or correction/kernel
    policy certification. Native basis/density/overlap checks remain mandatory.
    Existing dense response limits are neither invoked nor relaxed.
    """
    if not isinstance(caller_converged, (bool, np.bool_)) or not caller_converged:
        raise ValueError("explicit true caller_converged declaration required")
    if (isinstance(max_bytes, (bool, np.bool_))
            or not isinstance(max_bytes, (int, np.integer))
            or max_bytes <= 0):
        raise ValueError("max_bytes must be a positive integer")
    return core.NativeRestrictedState(wavefunction, True, int(max_bytes))
