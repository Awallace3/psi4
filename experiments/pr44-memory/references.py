"""Frozen protein157 full-run references; absolute energy differences in Eh."""
import math

REFERENCES = {
    "old-1200GB-job15521907": {
        "SAPT ELST ENERGY": -0.0075476756937860046,
        "SAPT EXCH ENERGY": 0.02114822712216955,
        "SAPT IND ENERGY": -0.0030370767602629365,
        "SAPT DISP ENERGY": -0.024184990657185188,
        "SAPT TOTAL ENERGY": -0.013621515989064577,
    },
    "new-1200GB-job15521908": {
        "SAPT ELST ENERGY": -0.0075476756846910575,
        "SAPT EXCH ENERGY": 0.021148227122191283,
        "SAPT IND ENERGY": -0.0030370768130164056,
        "SAPT DISP ENERGY": -0.024184984850164085,
        "SAPT TOTAL ENERGY": -0.013621510225680264,
    },
}


def compare(values, tolerance=1e-7):
    assert math.isfinite(tolerance) and tolerance > 0
    comparisons = {}
    for name, reference in REFERENCES.items():
        assert all(math.isfinite(values[k]) for k in reference)
        delta = {k: values[k] - v for k, v in reference.items()}
        comparisons[name] = {
            "delta_Eh": delta,
            "max_abs_delta_Eh": max(abs(v) for v in delta.values()),
            "passed": all(abs(v) <= tolerance for v in delta.values()),
        }
    return {"tolerance_Eh": tolerance, "references": comparisons,
            "passed": all(row["passed"] for row in comparisons.values())}
