"""Run only the supplier oracle against a pre-pseudoinverse CPU stage.

This compatibility driver is NOT a candidate admission gate. It removes only
the irrelevant, disabled CUEST_FDDS_PINV fixture option from calls to an older
binary that has no such option. Integral inputs, assertions and math are not
patched. Run outside the source tree with an explicitly selected staged import.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import psi4
    import pytest

    package = Path(psi4.__file__).resolve().parent
    if package != args.package.resolve():
        raise RuntimeError(f"Unexpected staged import: {package}")
    try:
        psi4.core.get_global_option("CUEST_FDDS_PINV")
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Historical adapter is forbidden on a current binary")
    args.output.mkdir(parents=True, exist_ok=False)
    test = args.source / "tests/pytests/test_fdds_tensor_provider.py"
    fixture = args.source / "tests/pytests/test_cuest_fdds.py"
    original = psi4.set_options
    dropped = []

    def compatible_options(options, *positional, **keywords):
        options = dict(options)
        for key in list(options):
            if key.lower() == "cuest_fdds_pinv":
                if options[key] is not False:
                    raise RuntimeError("Only a disabled, unused pinv option may be removed")
                dropped.append(key)
                del options[key]
        return original(options, *positional, **keywords)

    receipt = {
        "scope": "historical native tensor oracle; not candidate admission",
        "package": str(package),
        "core_sha256": hashlib.sha256(Path(psi4.core.__file__).read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "compatibility": "remove only disabled CUEST_FDDS_PINV from old fixture",
        "removed_options": dropped,
    }
    psi4.set_options = compatible_options
    try:
        result = pytest.main([str(test), "-q", "-s",
                              f"--junitxml={args.output / 'oracle.xml'}"])
    finally:
        psi4.set_options = original
        (args.output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
