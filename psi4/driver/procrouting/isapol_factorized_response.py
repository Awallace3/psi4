# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Bounded plain-DF MO factors, not a CamCASP preset.

In plain Coulomb DF, ``oo`` and ``ov`` are MO three-centre integrals and
``dual_ov``/``dual_vv`` are solved against the Coulomb metric. Keeping
left/right factors separate does not assume that other fitting models are a
single whitened Gram matrix. No metric inversion, constrained fitting, kernel
construction, or native admission is performed here.
"""
import numpy as np


def mo_three_center_shell(coulomb, main, coefficients, shell_index, *,
                          max_bytes=512*1024**2):
    """Return one AUX shell's raw MO integrals, shaped (function,MO,MO).

    ``coefficients`` must already be in the explicit spherical MAIN order;
    no Psi4-to-MAIN conversion, metric solve, or state certification is inferred.
    The selected AUX shell retains its original Cartesian/spherical ordering.
    Only its AO integrals are generated, never the full AUX-by-AO-pair tensor.

    Admission conservatively reserves 15 rows (the largest admitted S--G
    shell), native spherical accumulation (at most nine rows), coefficient
    storage, output, validation buffers and sequential two-matrix-product
    temporaries. Caller-retained earlier outputs, basis
    descriptors and implementation-specific Libint/BLAS workspace are excluded:
    a collecting consumer must maintain its own aggregate allocation ledger.
    The returned array is independent of inputs. No symmetrization is applied.
    """
    from psi4 import core

    if not isinstance(coulomb, core.IsaAuxCoulomb) or not isinstance(main, core.IsaExplicitBasis):
        raise ValueError("explicit Coulomb provider and MAIN basis required")
    if main.role != core.IsaBasisRole.Orbital:
        raise ValueError("MAIN must have Orbital role")
    if type(shell_index) is not int or shell_index < 0:
        raise ValueError("shell_index must be a nonnegative integer")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    nao = main.nfunction
    if (not isinstance(coefficients, np.ndarray) or coefficients.dtype != np.float64
            or coefficients.ndim != 2 or coefficients.shape[0] != nao
            or coefficients.shape[1] < 1):
        raise ValueError("coefficients must be nonempty float64 (nmain,nmo)")
    nmo = coefficients.shape[1]
    # Native generation holds its result and spherical accumulation together.
    # Include room for noncontiguous coefficient packing and finite masks.
    needed = 8*(24*nao**2 + 18*nmo**2 + 4*nao*nmo)
    if needed > max_bytes:
        raise ValueError("MO shell transform byte resource limit exceeded")
    work = 15*(2*nao**2*nmo + 2*nao*nmo**2)
    if work > FactorizedDFOperators.MAX_WORK:
        raise ValueError("MO shell transform work resource limit exceeded")
    if not np.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite")
    ao = np.asarray(coulomb.three_center_shell_block(
        main, shell_index, 1, 15*nao**2*8)).reshape(-1, nao, nao)
    result = np.empty((len(ao), nmo, nmo))
    with np.errstate(over="raise", invalid="raise"):
        for row in range(len(ao)):
            result[row] = (coefficients.T @ ao[row]) @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("nonfinite MO shell integrals")
    return result


class FactorizedDFOperators:
    """Owned, immutable occupied-fast plain-DF factors built by ``native_plain_df_operators``.

    ``_oo``/``_ov`` have shapes (naux,nocc,nocc)/(naux,nocc,nvir); ``_dual_ov``
    and ``_dual_vv`` are tuples of solved column tiles of the flattened
    (naux,nocc*nvir) and (naux,nvir*nvir) duals. Gaps have shape (nocc*nvir,),
    indexed by ``a*nocc+i``. The response runner assembles H1/H2 from them.
    """
    MAX_WORK = 64_000_000_000
