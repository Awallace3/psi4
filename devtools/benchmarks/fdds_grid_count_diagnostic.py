"""Local, unchanged-science state snapshots for the CPHF grid-count failure.

Run outside the source tree after selecting a staged psiapi. This does not
change pruning, geometry, fitting settings or assertions. The fresh-grid
counts are diagnostic reconstruction, not a substitute for the live grid.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import psi4

    package = Path(psi4.__file__).resolve().parent
    if package != args.package.resolve():
        raise RuntimeError(f"Unexpected staged import: {package}")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source / "tests/pytests"))
    import test_cuest_cphf as tests

    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.core.clean_options()
    psi4.set_memory("500 MiB")
    keys = """
        DFT_GRID_NAME DFT_RADIAL_POINTS DFT_SPHERICAL_POINTS
        DFT_RADIAL_SCHEME DFT_PRUNING_SCHEME DFT_PRUNING_ALPHA
        DFT_NUCLEAR_SCHEME DFT_BS_RADIUS_ALPHA DFT_WEIGHTS_TOLERANCE
        DFT_BASIS_TOLERANCE DFT_REMOVE_DISTANT_POINTS DFT_BLOCK_SCHEME
        DFT_BLOCK_MAX_POINTS DFT_BLOCK_MIN_POINTS DFT_BLOCK_MAX_RADIUS
    """.split()
    receipt = {
        "scope": "unchanged CPHF setup and post-SCF grid reconstruction only",
        "package": str(package),
        "core_sha256": hashlib.sha256(Path(psi4.core.__file__).read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(tests.__file__).read_bytes()).hexdigest(),
    }

    def snapshot(wfn):
        basis = wfn.basisset()
        grid = wfn.V_potential().grid()
        arrays = {
            "geometry": np.asarray(wfn.molecule().geometry()),
            "basis_geometry": np.asarray(basis.molecule().geometry()),
            "orientation": np.asarray(grid.orientation()),
        }
        return {
            "npoints": grid.npoints(),
            "arrays_hex": {k: [[float(x).hex() for x in row] for row in a]
                           for k, a in arrays.items()},
            "options": {k: psi4.core.get_option("SCF", k) for k in keys},
        }

    def save():
        (args.output / "snapshots.json").write_text(json.dumps(receipt, indent=2) + "\n")

    fixed, _ = tests._scf("svwn", False, "host", None)
    receipt["fixed_before_polarizability"] = snapshot(fixed)
    save()
    tests._polarizability(fixed, False)
    receipt["fixed_after_polarizability"] = snapshot(fixed)
    save()
    switched, _ = tests._scf("svwn", False, "host", (25, 110))
    receipt["switched"] = snapshot(switched)
    save()
    for name, wfn in (("fixed", fixed), ("switched", switched)):
        basis = wfn.basisset()
        receipt[name + "_rebuilt_counts"] = [
            psi4.core.DFTGrid.build(basis.molecule(), basis).npoints() for _ in range(3)]
    receipt["same_live_count"] = (receipt["fixed_before_polarizability"]["npoints"]
                                  == receipt["switched"]["npoints"])
    save()
    psi4.core.close_outfile()
    return 0 if receipt["same_live_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
