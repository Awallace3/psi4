#!/usr/bin/env python3
"""Run one GPU validation worker with SDK/map admission in that same process.

Does not create a cuEST handle. The payload must actually initialize/use Psi4's
cuEST route, after which the checked native-version provider is queried.
Compile-only/CPU-only validation must not use this GPU worker launcher.
"""
import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys


def check_provider(core):
    provider = getattr(core, "_cuest_runtime_version", None)
    if provider is None:
        raise ValueError("staged core has no approved cuEST native-version provider")
    if not inspect.isbuiltin(provider) or provider.__module__ != core.__name__:
        raise ValueError("cuEST native-version provider is not a core builtin")
    return provider


def native_receipt(core):
    provider = check_provider(core)
    version = provider()
    if (type(version) is not tuple or len(version) != 3
            or any(type(part) is not int for part in version) or version != (0, 2, 50)):
        raise ValueError(f"native cuEST version receipt is not tuple(0,2,50): {version!r}")
    return version


def run(module, argv, receipt, identity_file, contract_file, script=False):
    receipt = Path(receipt)
    if receipt.exists():
        raise ValueError("refusing to overwrite a runtime admission receipt")
    record = {"status": "failed", "module": module, "argv": argv}
    status = 1
    old_argv = sys.argv
    try:
        spec = importlib.util.spec_from_file_location("cuest_contract", contract_file)
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        expected = json.loads(Path(identity_file).read_text())
        import numpy  # noqa: F401 -- consistent NumPy-before-Psi4 BLAS loading
        sys.argv.append("--einsums:no-attach-debugger")
        try:
            import psi4
        finally:
            sys.argv.pop()
        # Presence only before GPU work: do not call the provider on a null handle.
        check_provider(psi4.core)
        record["before"] = contract.admit_runtime(psi4.core.__file__, expected)
        sys.argv = [module, *argv]
        try:
            if script:
                runpy.run_path(module, run_name="__main__")
            else:
                runpy.run_module(module, run_name="__main__", alter_sys=True)
            status = 0
        except SystemExit as exit:
            status = exit.code if type(exit.code) is int else (0 if exit.code is None else 1)
        record["payload_exit"] = status
        record["native_version"] = native_receipt(psi4.core)
        record["after"] = contract.admit_runtime(psi4.core.__file__, expected)
        if status == 0:
            record["status"] = "passed"
    except BaseException as error:
        record["admission_error"] = f"{type(error).__name__}: {error}"
        status = status or 1
    finally:
        sys.argv = old_argv
        # O_EXCL prevents accidental evidence reuse, even on overlapping launches.
        with receipt.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--module")
    entry.add_argument("--script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.args[1:] if args.args[:1] == ["--"] else args.args
    return run(args.module or args.script, argv, args.receipt, os.environ["CUEST_CORE_IDENTITY"],
               os.environ["CUEST_CONTRACT"], script=args.script is not None)


if __name__ == "__main__":
    raise SystemExit(main())
