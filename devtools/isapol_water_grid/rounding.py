"""Offline e15.8 print intervals, not a machine-equality grid test."""
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import re


def e15_8_interval(token):
    """Closed nearest-eight-significant-digit midpoint bounds.

    Accept whitespace, optional sign, normalized ``0.dddddddd``, uppercase E/D
    and a signed two/three-digit exponent. Nonzero mantissas must start at1--9.
    Zero is explicitly exact, accepted only as signed/unsigned
    ``0.00000000E+00`` (or D): no arbitrary Decimal-Emin/underflow neighborhood.

    At a decade, distinct representables are unequally spaced. E.g.1 has bounds
    [0.999999995,1.00000005], not1 +/-5e-8. Both endpoints are retained because
    the writer's exact binary halfway/tie behavior is not established. These
    closed bounds do not assert which adjacent token owns a true exact tie.
    Arithmetic is independent of the caller's Decimal precision/range/traps.
    """
    match = re.fullmatch(r"([+-]?0\.([0-9]{8}))[ED]([+-][0-9]{2,3})", token.strip())
    if match is None:
        raise ValueError(f"not a finite e15.8 token: {token!r}")
    digits, exponent = match[2], match[3]
    if digits == "00000000":
        if exponent != "+00":
            raise ValueError("canonical zero requires exponent +00; zero is exact")
        return Decimal(0), Decimal(0)
    if digits[0] == "0":
        raise ValueError("nonzero e15.8 token must be normalized")
    centre = Decimal(token.strip().replace("D", "E"))
    options = dict(rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
                   capitals=1, clamp=0, flags=[], traps=[])
    precision8 = Context(prec=8, **options)
    below = centre.next_minus(context=precision8)
    above = centre.next_plus(context=precision8)
    with localcontext(Context(prec=32, **options)):
        return (below+centre)/2, (centre+above)/2
