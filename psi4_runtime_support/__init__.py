"""Pre-native bootstrap support, installed with the whole Psi4 distribution.

Importing this package does not import Psi4, CUDA bindings, or native libraries.
Only explicit dedicated-worker preparation allocates executable snapshots.
The selected scientific runtime must adopt the exact retained preparation.
"""
import os
from pathlib import Path
import re
import sys
import threading

_PREPARED = None
_PREPARING = False
_LOCK = threading.Lock()


def _after_fork():
    global _LOCK
    _LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def sealed_preparation_active():
    """Selection guard, not a byte/mapping receipt."""
    return _PREPARING or _PREPARED is not None


def selection_lock():
    """The selector/preparation seam shares one admission lock."""
    return _LOCK


def prepared_runtime():
    """Return only the actual process-retained object, never a supplied receipt."""
    with _LOCK:
        if _PREPARING:
            raise RuntimeError("Sealed preparation is still in progress")
        if _PREPARED is not None:
            _PREPARED.check_live()
        return _PREPARED


def prepare_sealed_runtime(pins, *, alias_directory, dedicated_worker=False):
    """Prepare once before Psi4 import, or confirm the exact retained preparation.

    This support package and its bytes must be authenticated by the whole-build
    bootstrap. The diagnostic directory must be excluded from the startup
    loader path; only sealed descriptors/SONAME bindings load toolkit images.
    No caller-provided dictionary is an adoption proof.
    """
    global _PREPARED, _PREPARING
    if dedicated_worker is not True:
        raise ValueError("Explicit dedicated_worker=True is required")
    from .sealed import _SealedRuntime
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Explicit nonempty path/SHA256 pins required")
    normalized = {}
    for name, checksum in pins.items():
        path = Path(name)
        if (not path.is_absolute() or not isinstance(checksum, str)
                or re.fullmatch(r"[0-9a-f]{64}", checksum) is None):
            raise ValueError("Absolute paths and lowercase SHA256 pins required")
        canonical = str(path.resolve(strict=True))
        if canonical in normalized:
            raise ValueError("Duplicate canonical snapshot path")
        normalized[canonical] = checksum
    snapshot = tuple(sorted(normalized.items()))
    directory = os.fspath(alias_directory)
    with _LOCK:
        if _PREPARED is not None:
            _PREPARED.check_live()
            if snapshot != _PREPARED.snapshot or directory != _PREPARED.alias_directory:
                raise RuntimeError("A different sealed runtime is already prepared")
            return _PREPARED
        if _PREPARING:
            raise RuntimeError("Sealed preparation is already in progress")
        if "psi4" in sys.modules:
            raise RuntimeError("Sealed preparation must precede Psi4 import")
        # Selection can only be initiated by the admitting main thread.
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Sealed preparation requires the main thread")
        # Also protect a stdlib-only test/embedding import of the selector.
        # No import of a parent scientific package occurs here.
        for name, module in tuple(sys.modules.items()):
            if name.endswith(".sapt_cuest_owned_runtime") and module is not None:
                if (getattr(module, "_PROCESS_OWNER", None) is not None
                        or getattr(module, "_LEGACY_SESSION_STARTED", False)
                        or getattr(module, "_ADMISSION_IN_PROGRESS", False)):
                    raise RuntimeError("Cannot prepare after runtime/session selection")
        _PREPARING = True
    owner = _SealedRuntime()
    try:
        try:
            owner._initialize(snapshot, directory)
        except BaseException as primary:
            try:
                owner._discard_unpublished()
            except BaseException as error:
                if not getattr(owner, "_rollback_complete", False):
                    os._exit(86)
                primary.add_note("Snapshot rollback also failed: " + str(error))
            raise
        with _LOCK:
            owner.published = True
            _PREPARED = owner
    finally:
        with _LOCK:
            _PREPARING = False
    owner.check_live()
    return owner
