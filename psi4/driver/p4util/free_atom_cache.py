#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2026 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#
"""Reuse of the MBIS free-atom reference volumes, in process and on disk.

A free-atom volume -- the :math:`\\langle r^3 \\rangle` radial moment of an isolated neutral atom,
which :psivar:`MBIS VOLUME RATIOS` divides by -- is a property of an element, a level of theory,
the basis that element is given, and the grid and convergence settings.  It is not a property of
the molecule.  So every job in a dataset recomputes the same handful of numbers: for small
molecules the free-atom references run 8-25% of the total wall time, a finite-difference frequency
repeats them at every displacement, and QCFractal, which runs one molecule per process, gets no
reuse at all.  The last of those is why an in-memory dict is not enough on its own.

The governing asymmetry, and the reason the key below is as paranoid as it is: a spurious miss
costs one atomic SCF, whereas a wrong hit biases every volume ratio in a dataset the same way.
Volume ratios feed dispersion coefficients, so such a bias is systematic, not noise.  Every
judgement call here is therefore resolved towards a miss -- unreadable file, unparseable record,
key that disagrees by one field, basis whose contents cannot be hashed: all of them miss, and none
of them may raise, because a cache must never be able to fail a calculation.
"""

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from psi4 import core

__all__ = [
    "basis_content_hash",
    "build_key",
    "clear",
    "key_hash",
    "list_entries",
    "lookup",
    "mode",
    "path",
    "prewarm",
    "store",
]

#: Bump whenever the *construction* of the key changes in a way that should invalidate existing
#: entries -- a field added, removed, renamed, or canonicalized differently.  Old entries then
#: simply stop matching rather than having to be found and deleted.
CACHE_FORMAT_VERSION = 1

#: Bump whenever the on-disk record layout changes.  Records carrying any other value are ignored.
SCHEMA = 1

_VALID_MODES = ("OFF", "READ", "WRITE", "READWRITE")

# Layer 0: process-local memo, keyed by the same hash as the files.  Serves driver loops (findif,
# geometry optimization, a scripted scan) which the disk layer would also serve, but without the
# json round trip.
_MEMO: Dict[str, Dict[str, Any]] = {}

# Options whose effective value always enters the key.  Section 3.1 of the design measured
# free-atom volume shifts of 1e-7 to 3e-4 for SCF_TYPE, DF_BASIS_SCF, PUREAM, D_CONVERGENCE and the
# MBIS grid; the rest are here because they plainly could move a converged atomic density.
#
# The DFT entries are recorded even when the method is HF, where they are inert.  A constant extra
# field costs nothing, and the alternative -- deciding per method which knobs are live -- is one
# more thing to get wrong every time psi4 grows a functional.
_KEY_OPTIONS = (
    # Reference determination and integrals.
    "SCF_TYPE",
    "DF_BASIS_SCF",
    "PUREAM",
    "E_CONVERGENCE",
    "D_CONVERGENCE",
    "INTS_TOLERANCE",
    "MAXITER",
    "STABILITY_ANALYSIS",
    "DAMPING_PERCENTAGE",
    "SOSCF",
    "PERTURB_H",
    "RELATIVISTIC",
    "FREEZE_CORE",
    # MBIS partitioning itself and the grid it is evaluated on.
    "MBIS_RADIAL_POINTS",
    "MBIS_SPHERICAL_POINTS",
    "MBIS_PRUNING_SCHEME",
    "MBIS_MAXITER",
    "MBIS_D_CONVERGENCE",
    "MAX_RADIAL_MOMENT",
    # MBIS_SCREENING_THRESHOLD drops an atom from a grid block once its pro-atom density falls
    # below it, so it is an approximation control and moves the number outright.  MBIS_ANDERSON
    # only accelerates the iteration towards the same fixed point, and so should reach the same
    # answer to within MBIS_D_CONVERGENCE -- "should" is not "does", and an entry is meant to
    # record the settings that produced it, so it is keyed on too.
    "MBIS_SCREENING_THRESHOLD",
    "MBIS_ANDERSON",
    # DFT grid and range-separation/dispersion knobs.
    "DFT_SPHERICAL_POINTS",
    "DFT_RADIAL_POINTS",
    "DFT_PRUNING_SCHEME",
    "DFT_NUCLEAR_SCHEME",
    "DFT_RADIAL_SCHEME",
    "DFT_BLOCK_SCHEME",
    "DFT_BASIS_TOLERANCE",
    "DFT_DENSITY_TOLERANCE",
    "DFT_WEIGHTS_TOLERANCE",
    "DFT_OMEGA",
    "DFT_ALPHA",
    "DFT_OMEGA_C",
    "DFT_ALPHA_C",
    "DFT_VV10_B",
    "DFT_VV10_C",
)

# Options read from the *global* scope rather than as the SCF module sees them, because the SCF
# module's copy is not the user's setting.  ``GUESS`` is the case that forced this: scf_helper
# resolves ``AUTO`` to ``SAD`` or ``CORE`` depending on how many atoms the molecule has and leaves
# the resolved value behind in the SCF module while free_atom_volumes runs, so keying on it would
# have made oxygen's free-atom reference depend on the size of the molecule that asked for it --
# measured, and it did: water and neon-dimer parents landed on different entries than a lone neon.
# The user's own ``set guess sad`` is global and is still honoured.  ``DF_BASIS_MP2`` is here for
# the duller reason that it is not an SCF-module option at all.
_KEY_OPTIONS_GLOBAL = (
    "GUESS",
    "DF_BASIS_MP2",
)

# The whitelist above will go stale as psi4 gains options, so it is backed by a sweep over every
# option the user actually changed (:func:`_changed_options`).  These are the ones that sweep may
# skip.  Keep this set small and justify every entry: anything omitted here merely causes a
# spurious miss, which is the safe direction, while anything wrongly added risks a wrong hit.
_SWEEP_IGNORE = frozenset({
    # Printing, logging, and file naming: cannot reach a number.
    "PRINT",
    "DEBUG",
    "OUTPUT",
    "WRITER_FILE_LABEL",
    "PSI_SCRATCH",
    "PRINT_BASIS",
    # Resources: change how long the answer takes, not what it is.
    "NUM_THREADS",
    "MEMORY",
    "INTS_NUM_THREADS",
    # Carried by the key already, or deliberately superseded by it.
    "BASIS",  # a *name*; the key carries a hash of the resolved shells instead
    "REFERENCE",  # resolved per element and stored explicitly
    "SCF_PROPERTIES",  # stripped from the atomic computations by free_atom_volumes
    "MBIS_FREE_ATOM_CACHE",  # the cache configuring itself
    "MBIS_FREE_ATOM_CACHE_PATH",
    # Geometry-optimization and finite-difference machinery.  A free-atom reference is a single
    # atom at the origin computed by a bare energy() call; none of this is reachable from there.
    "GEOM_MAXITER",
    "FULL_HESS_EVERY",
    "G_CONVERGENCE",
    "OPT_TYPE",
    "STEP_TYPE",
})

# Ignored in the *module-local* sweep only: psi4 rewrites its own copy of these mid-run (see
# _KEY_OPTIONS_GLOBAL), so the module-local value reports the driver's bookkeeping rather than
# anything the user asked for.  The global value of each is still swept and still keyed on.
_SWEEP_IGNORE_LOCAL = frozenset({
    "GUESS",
    "DF_INTS_IO",
    "ORBITALS_WRITE",
})

# Options whose effective value the key already records verbatim in ``options``.  Sweeping them a
# second time for *changed-ness* adds nothing and actively hurts: psi4 sets several of them on its
# own behalf mid-run (scf_helper's optstash covers INTS_TOLERANCE, DF_BASIS_SCF, PUREAM, SCF_TYPE),
# so after one molecular SCF has run in the process they report "changed" at the same value they
# always had.  That made an entry written by prewarm() unreachable from the very jobs it was meant
# to serve.  Keying on the value and not on how it got there is both stricter and stabler.
_SWEEP_IGNORE_KEYED = frozenset(_KEY_OPTIONS) | frozenset(_KEY_OPTIONS_GLOBAL)

# Whole families that cannot touch a one-atom SCF, ignored by prefix.
_SWEEP_IGNORE_PREFIXES = (
    "MOLDEN_",  # output writers
    "CUBEPROP_",
    "SAPT",  # interaction-energy machinery; needs >= 2 fragments
    "FISAPT",
    "FINDIF_",  # displacement generation
    "OPTKING",
)

# Modules swept for *locally* set options.  `set scf reference uhf` is module-local and does not
# show up in the global sweep, so SCF has to be checked explicitly.  Only SCF is swept: it is the
# module that runs the free-atom computation, and each additional module costs a full pass over
# the ~1100-entry option list.  A module-local set on some *other* module that nonetheless changes
# a free-atom volume would be missed -- no such option is known, and the whitelist above covers
# the plausible candidates by value regardless of where they were set.
_SWEEP_MODULES = ("SCF", )

# Filled on first use: which option names are valid in a given module.  Probing this costs a pass
# over the whole option list, so it is done once per process rather than once per molecule.
_MODULE_OPTIONS: Dict[str, List[str]] = {}

_warned: set = set()


def _warn_once(message: str) -> None:
    """Print `message` to the output file the first time it is seen in this process."""
    if message not in _warned:
        _warned.add(message)
        core.print_out(f"  Warning: {message}\n")


# ==> Configuration <==


def mode() -> str:
    """Return the active cache mode, one of ``OFF``, ``READ``, ``WRITE``, ``READWRITE``.

    :envvar:`PSI4_FREE_ATOM_CACHE` wins over the |globals__mbis_free_atom_cache| option, so a
    cluster job can be forced read-only without touching any input file.  ``WRITE`` recomputes
    unconditionally and refreshes the entry; ``OFF`` disables the in-process memo as well, which
    is what makes it usable for a controlled timing comparison.
    """
    candidate = os.environ.get("PSI4_FREE_ATOM_CACHE")
    if candidate is None:
        try:
            candidate = core.get_option("SCF", "MBIS_FREE_ATOM_CACHE")
        except Exception:
            return "READWRITE"
    candidate = str(candidate).strip().upper()
    if candidate not in _VALID_MODES:
        _warn_once(f"ignoring unrecognized free-atom cache mode {candidate!r}; using READWRITE")
        return "READWRITE"
    return candidate


def path() -> Path:
    """Return the directory holding the cache, without creating it.

    :envvar:`PSI4_FREE_ATOM_CACHE_PATH`, else |globals__mbis_free_atom_cache_path|, else
    :func:`_install_cache_dir`.
    """
    candidate = os.environ.get("PSI4_FREE_ATOM_CACHE_PATH")
    if not candidate:
        try:
            candidate = core.get_option("SCF", "MBIS_FREE_ATOM_CACHE_PATH")
        except Exception:
            candidate = ""
    if candidate:
        return Path(candidate).expanduser()
    return _install_cache_dir()


def _install_cache_dir() -> Path:
    """``<PSIDATADIR>/free_atom_volumes``: inside the installation that wrote the entries.

    An entry is only ever reused by a psi4 of the same series (the key records it), and it is
    derived data that costs one atomic SCF to regenerate, so it belongs inside the installation it
    came from -- under the conda prefix for a conda install, under ``<objdir>/stage`` for a build
    -- where removing psi4 removes it too.  A user-wide location such as ``~/.cache/psi4`` would
    outlive every psi4 that ever wrote to it, and nothing would ever clean it up.

    An installation nobody can write to therefore gets no cache rather than one somewhere
    unaffiliated: :func:`store` says so once and the calculation proceeds unaffected, and a shared
    or writable directory is one option away.
    """
    try:
        datadir = core.get_datadir()
    except Exception:
        datadir = ""
    if not datadir:
        # Nothing to anchor on; this file's own package directory is the same installation.
        datadir = str(Path(__file__).resolve().parents[2])
    return Path(datadir) / "free_atom_volumes"


# ==> Key construction <==


def basis_content_hash(basisset: core.BasisSet, center: int) -> Optional[str]:
    """Hash the shells `center` actually carries, or return None if they cannot be hashed.

    The obvious identifier, :py:meth:`~psi4.core.Molecule.basis_on_atom`, is the *block name* and
    not the resolved basis: ``set basis aug-cc-pvdz`` and a ``basis {}`` block named ``mixed``
    both report one string for every atom, whatever each atom was actually assigned.  Within a
    single run that is harmless, because the name is re-resolved per element downstream.  For a
    cache that outlives the run it is poison -- two unrelated inputs whose blocks happen to share
    a name would collide silently -- so identity is taken from the contracted functions
    themselves.  As a side benefit, anonymous blocks, which psi4 names with a fresh random suffix
    every run and which therefore could never hit a name-keyed cache, hash stably.

    Shell descriptors are sorted, so the hash depends on the set of contracted functions and not
    on the order the basis file happened to list them in.

    Returns None when the center carries an ECP: :py:meth:`~psi4.core.BasisSet.ecp_shell` hands
    back a type with no Python bindings, so the ECP's contents cannot be folded into the hash, and
    a hash that silently ignores part of the basis is exactly the wrong kind of wrong.  MBIS
    rejects ECP-bearing basis sets today, so this is defensive only.
    """
    try:
        if basisset.n_ecp_shell_on_center(center) > 0 or basisset.n_ecp_core() > 0:
            return None

        shells = []
        for i in range(basisset.nshell_on_center(center)):
            shell = basisset.shell(basisset.shell_on_center(center, i))
            primitives = tuple((repr(shell.exp(p)), repr(shell.original_coef(p)))
                               for p in range(shell.nprimitive))
            shells.append((int(shell.am), bool(shell.is_pure()), primitives))

        digest = hashlib.sha256()
        digest.update(b"psi4-basis-content-v1\n")
        digest.update(json.dumps(sorted(shells), separators=(",", ":")).encode())
        return digest.hexdigest()
    except Exception as exc:
        _warn_once(f"could not hash the basis on center {center} ({exc}); free-atom volumes will not be cached")
        return None


def _canonical(value: Any) -> Any:
    """Reduce an option value to something json can store and compare exactly."""
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return repr(value)


def _option_values() -> Dict[str, Any]:
    """Effective value of every whitelisted option."""
    values = {}
    for name, getter in ([(n, lambda n: core.get_option("SCF", n)) for n in _KEY_OPTIONS] +
                         [(n, core.get_global_option) for n in _KEY_OPTIONS_GLOBAL]):
        try:
            values[name] = _canonical(getter(name))
        except Exception:
            # An option this build does not have.  Recording its absence still discriminates
            # between builds that do and do not have it.
            values[name] = "<absent>"
    return values


def _ignored(name: str) -> bool:
    return (name in _SWEEP_IGNORE or name in _SWEEP_IGNORE_KEYED
            or name.startswith(_SWEEP_IGNORE_PREFIXES))


def _changed_options() -> List[List[Any]]:
    """Every option the user changed that is not provably irrelevant.

    This is the anti-staleness half of the key.  :data:`_KEY_OPTIONS` is a hand-maintained list and
    will go stale; this sweep means the default behaviour when psi4 gains an option that matters is
    a spurious miss rather than a wrong hit.  It is cheap: the global pass is a millisecond, and
    the per-module pass is a millisecond after the first call in a process pays to learn which
    option names each module has.
    """
    changed: Dict[str, List[Any]] = {}

    for name in core.get_global_option_list():
        if _ignored(name):
            continue
        try:
            if core.has_global_option_changed(name):
                changed[f"GLOBAL::{name}"] = ["GLOBAL", name, _canonical(core.get_global_option(name))]
        except Exception:
            continue

    for module in _SWEEP_MODULES:
        if module not in _MODULE_OPTIONS:
            try:
                _MODULE_OPTIONS[module] = [
                    name for name in core.get_global_option_list() if core.option_exists_in_module(module, name)
                ]
            except Exception:
                _MODULE_OPTIONS[module] = []
        for name in _MODULE_OPTIONS[module]:
            if _ignored(name) or name in _SWEEP_IGNORE_LOCAL:
                continue
            try:
                if core.has_option_changed(module, name):
                    changed[f"{module}::{name}"] = [module, name, _canonical(core.get_option(module, name))]
            except Exception:
                continue

    return [changed[k] for k in sorted(changed)]


def _psi4_series() -> str:
    """``major.minor`` of the running psi4.

    Deliberately not the full version: keying on the git-describe string would invalidate the
    whole cache on every rebuild, which in a development worktree means never getting a hit.  The
    trade is that a patch-level change to the MBIS code in oeprop.cc would not invalidate entries;
    bump :data:`CACHE_FORMAT_VERSION` by hand if that ever happens.
    """
    import psi4

    # Only the leading integers: a development build calls itself "1.12a1.dev585", and keying on
    # the pre-release tag would give every alpha its own cache for no benefit.
    matched = re.match(r"(\d+)\.(\d+)", str(psi4.__version__))
    return f"{matched.group(1)}.{matched.group(2)}" if matched else str(psi4.__version__)


def build_key(
    symbol: str,
    Z: int,
    multiplicity: int,
    reference: str,
    theory: str,
    basis_hash: Optional[str],
    salt: str = "",
) -> Optional[Dict[str, Any]]:
    """Assemble the full provenance dict identifying one free-atom volume.

    Note what is *not* here: the atom's input label, and the name of its basis.  A free-atom volume
    depends on the element and on which contracted functions it is given, so ``O1`` and ``O2``
    share an entry exactly when they share a basis -- which is the whole point of hashing contents
    rather than names.

    Returns None when `basis_hash` is None, i.e. when the basis could not be identified with
    confidence.  Callers must treat that as "do not cache", not as "cache under a partial key".
    """
    if basis_hash is None:
        return None
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "quantity": "MBIS FREE ATOM VOLUME",
        "psi4_version": _psi4_series(),
        "element": str(symbol).upper(),
        "Z": int(Z),
        "multiplicity": int(multiplicity),
        "reference": str(reference).upper(),
        "theory": str(theory).upper(),
        "basis_content_hash": basis_hash,
        "options": _option_values(),
        "changed_options": _changed_options(),
        "salt": str(salt),
    }


def _canonical_json(key: Dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def key_hash(key: Dict[str, Any]) -> str:
    """The 16 hex characters naming the file for `key`."""
    return hashlib.sha256(_canonical_json(key).encode()).hexdigest()[:16]


# ==> Store <==


def _entry_path(key: Dict[str, Any]) -> Path:
    return path() / f"{key_hash(key)}.json"


def lookup(key: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the cached record for `key`, or None on any kind of miss.

    "Any kind" is meant literally: no file, a truncated or unparseable file, a record from a
    future schema, or a record whose embedded key disagrees with `key` in a single field all
    return None, and none of them raise.  The field-by-field comparison is what makes a filename
    collision, a hand-edited file, or a stale entry from a different psi4 harmless.
    """
    if key is None:
        return None
    active = mode()
    if active in ("OFF", "WRITE"):
        return None

    digest = key_hash(key)
    record = _MEMO.get(digest)
    if record is not None and record.get("key") == key:
        return record

    entry = _entry_path(key)
    try:
        with open(entry) as handle:
            record = json.load(handle)
    except Exception:
        return None

    try:
        if record.get("schema") != SCHEMA:
            return None
        if record.get("key") != key:
            return None
        value = float(record["value"])
    except Exception:
        return None

    record["value"] = value
    _MEMO[digest] = record
    core.print_out(f"  MBIS free-atom volume for {key['element']} taken from cache {entry}\n")
    return record


def store(key: Optional[Dict[str, Any]], value: float, extras: Optional[Dict[str, Any]] = None,
          walltime_s: Optional[float] = None) -> None:
    """Record `value` for `key`, in process and (in a writing mode) on disk.

    The write goes to a uniquely named temporary file in the destination directory, is fsynced,
    and is then :func:`os.replace`\\ d onto its final name.  That is atomic on POSIX and needs no
    lock, which matters because the intended workload is hundreds of cluster workers computing the
    same few elements at once.  They race, but they race to write byte-identical content, so
    last-writer-wins is benign.
    """
    if key is None:
        return
    active = mode()
    if active == "OFF":
        return

    record = {
        "schema": SCHEMA,
        "value": float(value),
        "extras": extras or {},
        "key": key,
        "psi4_version": _psi4_series(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "walltime_s": walltime_s,
    }
    _MEMO[key_hash(key)] = record

    if "WRITE" not in active:
        return

    entry = _entry_path(key)
    temporary = entry.with_name(f"{entry.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        with open(temporary, "w") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, entry)
    except Exception as exc:
        _warn_once(f"could not write the free-atom cache entry {entry} ({exc}); "
                   "set MBIS_FREE_ATOM_CACHE_PATH to a writable directory to keep one")
        try:
            temporary.unlink()
        except Exception:
            pass


# ==> Inspection <==


def list_entries(cache_path: Optional[os.PathLike] = None) -> List[Dict[str, Any]]:
    """Every readable record in the cache, each with the ``file`` it came from.

    Unreadable files are skipped rather than reported, on the same principle as :func:`lookup`.
    """
    directory = Path(cache_path) if cache_path is not None else path()
    entries = []
    try:
        candidates = sorted(directory.glob("*.json"))
    except Exception:
        return entries
    for candidate in candidates:
        try:
            with open(candidate) as handle:
                record = json.load(handle)
        except Exception:
            continue
        record["file"] = str(candidate)
        entries.append(record)
    return entries


def clear(cache_path: Optional[os.PathLike] = None, memory_only: bool = False) -> int:
    """Drop the process-local memo and, unless `memory_only`, delete the cache files.

    Only ``*.json`` and leftover ``*.tmp`` directly in the cache directory are removed; the
    directory itself and anything else in it are left alone.  Returns the number of files deleted.
    """
    _MEMO.clear()
    if memory_only:
        return 0

    directory = Path(cache_path) if cache_path is not None else path()
    removed = 0
    for pattern in ("*.json", "*.tmp"):
        try:
            candidates = list(directory.glob(pattern))
        except Exception:
            continue
        for candidate in candidates:
            try:
                candidate.unlink()
                removed += 1
            except Exception:
                continue
    return removed


def prewarm(elements: Sequence[str], method: str, basis: Optional[str] = None,
            salt: str = "") -> Dict[str, float]:
    """Compute and cache the free-atom volume for each of `elements` up front.

    Run this once before launching a large campaign and ship the cache directory to the cluster,
    so the workers are read-mostly.  Beyond the speed, this is how to guarantee that every record
    in a dataset was divided by the *same* free-atom reference -- a consistency property worth
    having on its own.

    Options are read from the current psi4 state, so set them exactly as the campaign will, then
    call this.  Returns the volume computed for each element.
    """
    from .prop_util import compute_free_atom_volume

    if basis is not None:
        core.set_global_option("BASIS", basis)

    volumes = {}
    for element in elements:
        volumes[element] = compute_free_atom_volume(element, method, salt=salt)
    return volumes
