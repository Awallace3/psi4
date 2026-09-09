# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Pure dimension guards, not a memory, SCF, quadrature or parity certificate.

Mirrors native_response.cc's OV, direct-JK and ALDA guard order. Python integers
are exact (no machine wrap); C++-int input bounds also bound the cost of this
calculation. C++ remains authoritative for shell, workspace and state checks.
Zero grid rows explicitly means gridless/no_local, not an estimated ALDA grid.
"""
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class ResponseWorkEstimate:
    nbf: int
    nmo: int
    nocc: int
    grid_rows: int
    nov: int
    ao_work: int
    alda_work: int
    max_grid_rows: int
    max_nov: int
    failures: tuple
    provenance: str
    nbf_limit: int = 256
    native_nov_limit: int = 512
    ao_work_limit: int = 64_000_000_000
    alda_work_limit: int = 2_000_000_000
    grid_rows_limit: int = 1_000_000
    unchecked: tuple = ("shell max_am<=4 / max_nprimitive<=64", "workspace max_bytes",
                        "restricted C1 state, occupations, density and overlap",
                        "grid values and scientific integration accuracy")

    @property
    def passes(self):
        return not self.failures

    def require_pass(self):
        """Use the native invalid_argument/ValueError message contract."""
        if self.failures:
            raise ValueError("NativeResponseProvider: " + self.failures[0])


def estimate_response_work(nbf, nmo, nocc, grid_rows, *, max_nov=512):
    """Assess explicit dimensions without allocating grids, matrices or integrals.

    ``grid_rows`` is caller-supplied, not inferred from options. Use zero ONLY
    for a gridless policy. All dimensions must be non-bool integers representable
    by native C++ int, with 0 < nocc < nmo <= nbf. Products use bounded arbitrary
    precision integers: even rejected cases never overflow or silently wrap.
    Failures are ordered as in C++; later failures are diagnostic only (native
    short-circuits). Guard-passing products fit even a 32-bit size_t except AO
    work, whose production native limit requires a 64-bit build.
    """
    values = []
    for name, value in (("nbf", nbf), ("nmo", nmo), ("nocc", nocc), ("grid_rows", grid_rows)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2**31-1:
            raise ValueError(f"{name} must be a nonnegative native-int dimension (not bool)")
        values.append(int(value))
    nbf, nmo, nocc, grid_rows = values
    if not 0 < nocc < nmo <= nbf:
        raise ValueError("NativeResponseProvider: invalid integer Aufbau occupations or empty OV space")
    if isinstance(max_nov, bool) or not isinstance(max_nov, Integral) or max_nov <= 0:
        raise ValueError("max_nov must be a positive integer")
    nov = nocc * (nmo - nocc)
    ao_work = nov * nbf**4
    alda_work = grid_rows * nov**2
    failures = []
    if nov > min(max_nov, 512):
        failures.append("dense OV resource limit (maximum 512)")
    if nbf > 256 or ao_work > 64_000_000_000:
        failures.append("direct JK work resource limit")
    if grid_rows > 1_000_000 or alda_work > 2_000_000_000:
        failures.append("ALDA work resource limit")
    return ResponseWorkEstimate(nbf, nmo, nocc, grid_rows, nov, ao_work, alda_work,
        min(1_000_000, 2_000_000_000 // nov**2), int(max_nov), tuple(failures),
        "explicit caller dimensions and supplied row count; exact integer arithmetic; "
        "native_response.cc dimension guards only; zero rows means gridless")
