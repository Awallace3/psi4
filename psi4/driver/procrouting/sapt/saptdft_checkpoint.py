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

import hashlib
import importlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import qcelemental as qcel

from psi4 import core
from psi4.metadata import __version__, __version_long

from ... import p4util
from ...p4util.exceptions import ValidationError

__all__ = [
    "SAPTDFTCheckpoint",
    "SAPTDFT_STAGE_DEFINITIONS",
    "StageDefinition",
    "build_saptdft_job_identity",
    "capture_scf_snapshot",
    "rehydrate_scf_wavefunction",
]

SAPTDFT_CHECKPOINT_SCHEMA_VERSION = 1
SAPTDFT_STAGE_DEFINITION_VERSION = 1
SAPTDFT_SCF_SNAPSHOT_VERSION = 1
SAPTDFT_MANIFEST_FILENAME = "saptdft_state.json"
SAPTDFT_LOCK_FILENAME = "saptdft_state.lock"
_ALLOWED_ARTIFACT_KINDS = {"array", "scf_snapshot"}
_SCF_SNAPSHOT_METADATA_KEY = "scf_snapshot"
_SCF_SNAPSHOT_DIMENSION_KEYS = (
    "doccpi",
    "frzcpi",
    "frzvpi",
    "nalphapi",
    "nbetapi",
    "nmopi",
    "nsopi",
    "soccpi",
)
_SCF_SNAPSHOT_FACTORY_BASIS_KEYS = ("DF_BASIS_SCF", "BASIS_RELATIVISTIC", "SAPGAU")
_SCF_SNAPSHOT_REQUIRED_FIELDS = {
    "matrix": ("Ca", "Cb", "Da", "Db", "Fa", "Fb"),
    "vector": ("epsilon_a", "epsilon_b"),
}
_SCF_SNAPSHOT_REQUIRED_FLOATVARS = {
    "RHF": ("CURRENT ENERGY", "CURRENT REFERENCE ENERGY", "SCF TOTAL ENERGY", "HF TOTAL ENERGY"),
    "RKS": ("CURRENT ENERGY", "CURRENT REFERENCE ENERGY", "SCF TOTAL ENERGY", "DFT TOTAL ENERGY"),
}
_RUNTIME_CONTROL_KEYS = {
    "checkpoint_dir",
    "checkpoint_directory",
    "checkpoint_stop_after",
    "memory",
    "memory_bytes",
    "nthreads",
    "num_threads",
    "output",
    "output_file",
    "print",
    "psi4_checkpoint_dir",
    "psi4_checkpoint_stop_after",
    "threads",
    "timer",
    "timing",
    "verbose",
    "verbosity",
}


@dataclass(frozen=True)
class StageDefinition:
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    version: int = SAPTDFT_STAGE_DEFINITION_VERSION


SAPTDFT_STAGE_DEFINITIONS: dict[str, StageDefinition] = {
    "grac_monomer_a": StageDefinition(),
    "grac_monomer_b": StageDefinition(),
    "hf_dimer_scf": StageDefinition(dependencies=("grac_monomer_a", "grac_monomer_b")),
    "hf_monomer_a_scf": StageDefinition(dependencies=("hf_dimer_scf",)),
    "hf_monomer_b_scf": StageDefinition(dependencies=("hf_monomer_a_scf",)),
    "hf_sapt_elst": StageDefinition(dependencies=("hf_monomer_b_scf",)),
    "hf_sapt_exch": StageDefinition(dependencies=("hf_sapt_elst",)),
    "hf_sapt_ind": StageDefinition(dependencies=("hf_sapt_exch",)),
    "dimer_localization_scf": StageDefinition(dependencies=("hf_monomer_b_scf",)),
    "monomer_a_dft_scf": StageDefinition(dependencies=("dimer_localization_scf",)),
    "monomer_b_dft_scf": StageDefinition(dependencies=("monomer_a_dft_scf",)),
    "delta_dft_dimer_scf": StageDefinition(dependencies=("monomer_b_dft_scf",)),
    "delta_dft_monomer_a_scf": StageDefinition(dependencies=("delta_dft_dimer_scf",)),
    "delta_dft_monomer_b_scf": StageDefinition(dependencies=("delta_dft_monomer_a_scf",)),
    "delta_dft": StageDefinition(dependencies=("delta_dft_monomer_b_scf",)),
    "elst": StageDefinition(dependencies=("monomer_b_dft_scf",)),
    "exch": StageDefinition(dependencies=("elst",)),
    "ind": StageDefinition(dependencies=("exch",)),
    "disp": StageDefinition(dependencies=("ind",)),
    "d3": StageDefinition(dependencies=("disp",)),
    "d4": StageDefinition(dependencies=("disp",)),
    "fsapt_setup": StageDefinition(dependencies=("monomer_b_dft_scf",)),
    "fsapt_elst": StageDefinition(dependencies=("fsapt_setup",)),
    "fsapt_exch": StageDefinition(dependencies=("fsapt_elst",)),
    "fsapt_ind": StageDefinition(dependencies=("fsapt_exch",)),
    "fsapt_disp": StageDefinition(dependencies=("fsapt_ind",)),
    "fsapt_final": StageDefinition(dependencies=("fsapt_disp",)),
    "final": StageDefinition(dependencies=("disp",)),
}


def _validate_stage_definitions() -> None:
    for stage, definition in SAPTDFT_STAGE_DEFINITIONS.items():
        for dependency in definition.dependencies:
            if dependency not in SAPTDFT_STAGE_DEFINITIONS:
                raise ValidationError(f"SAPT(DFT) checkpoint stage {stage} depends on unknown stage {dependency}.")

    visiting = set()
    visited = set()

    def walk(stage: str) -> None:
        if stage in visited:
            return
        if stage in visiting:
            raise ValidationError(f"SAPT(DFT) checkpoint stage graph contains a cycle at {stage}.")
        visiting.add(stage)
        for dependency in SAPTDFT_STAGE_DEFINITIONS[stage].dependencies:
            walk(dependency)
        visiting.remove(stage)
        visited.add(stage)

    for stage in SAPTDFT_STAGE_DEFINITIONS:
        walk(stage)


_validate_stage_definitions()


def _json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _normalize_jsonable(value: Any, *, lower_dict_keys: bool = False, lower_strings: bool = False) -> Any:
    value = _normalize_scalar(value)
    if isinstance(value, Mapping):
        items = []
        for key, item_value in value.items():
            key = str(key)
            if lower_dict_keys:
                key = key.lower()
            items.append((key, _normalize_jsonable(item_value, lower_dict_keys=lower_dict_keys, lower_strings=lower_strings)))
        return {key: item for key, item in sorted(items, key=lambda kv: kv[0])}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_normalize_jsonable(item, lower_dict_keys=lower_dict_keys, lower_strings=lower_strings) for item in value]
    if isinstance(value, list):
        return [_normalize_jsonable(item, lower_dict_keys=lower_dict_keys, lower_strings=lower_strings) for item in value]
    if isinstance(value, str) and lower_strings:
        return value.lower()
    return value


def _strip_runtime_controls(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned = {}
        for key, item_value in value.items():
            lowered_key = str(key).lower()
            if lowered_key in _RUNTIME_CONTROL_KEYS:
                continue
            nested = _strip_runtime_controls(item_value)
            if isinstance(nested, Mapping) and not nested:
                continue
            cleaned[key] = nested
        return cleaned
    if isinstance(value, tuple):
        return [_strip_runtime_controls(item) for item in value]
    if isinstance(value, list):
        return [_strip_runtime_controls(item) for item in value]
    return value


def _first_difference(left: Any, right: Any, path: str = "") -> Optional[str]:
    if type(left) != type(right):
        return f"{path or '<root>'}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, Mapping):
        left_keys = sorted(str(key) for key in left)
        right_keys = sorted(str(key) for key in right)
        if left_keys != right_keys:
            return f"{path or '<root>'}: keys {left_keys} != {right_keys}"
        for key in left_keys:
            diff = _first_difference(left[key], right[key], f"{path}.{key}" if path else key)
            if diff:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path or '<root>'}: lengths {len(left)} != {len(right)}"
        for index, (lval, rval) in enumerate(zip(left, right)):
            diff = _first_difference(lval, rval, f"{path}[{index}]")
            if diff:
                return diff
        return None
    if left != right:
        return f"{path or '<root>'}: {left!r} != {right!r}"
    return None


def _optional_module_version(name: str) -> Optional[str]:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None) or "present"


def _saptdft_einsums_bundle_available() -> bool:
    try:
        importlib.import_module("einsums")
        importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")
        importlib.import_module("psi4.driver.procrouting.sapt.sapt_mp2_terms_ein")
    except ImportError:
        return False
    return True



def _select_saptdft_backend() -> str:
    use_einsums = core.get_option("SAPT", "SAPT_DFT_USE_EINSUMS")
    if use_einsums and _saptdft_einsums_bundle_available():
        return "einsums"
    return "numpy"



def _selected_addon_versions(canonical_input: Mapping[str, Any]) -> dict[str, Any]:
    method = str(
        canonical_input.get("specification", {})
        .get("model", {})
        .get("method", "")
    ).lower()
    addon_versions = {}
    if "-d3" in method or "dft-d3" in method:
        addon_versions["dftd3_version"] = _optional_module_version("dftd3")
    if "-d4" in method or "dft-d4" in method:
        addon_versions["dftd4_version"] = _optional_module_version("dftd4")
    return addon_versions



def _build_execution_fingerprint(canonical_input: Mapping[str, Any]) -> dict[str, Any]:
    selected_backend = _select_saptdft_backend()
    fingerprint = {
        "checkpoint_schema_version": SAPTDFT_CHECKPOINT_SCHEMA_VERSION,
        "stage_definition_version": SAPTDFT_STAGE_DEFINITION_VERSION,
        "psi4_version": __version__,
        "psi4_version_long": __version_long,
        "qcelemental_version": qcel.__version__,
        "qcengine_version": _optional_module_version("qcengine"),
        "qcschema_model": "AtomicInput_v2",
        "selected_backend": selected_backend,
        "numpy_version": np.__version__,
    }
    if selected_backend == "einsums":
        fingerprint["einsums_version"] = _optional_module_version("einsums")
    fingerprint.update(_selected_addon_versions(canonical_input))
    return fingerprint


def _coerce_atomic_input_dict(atomic_input: Any) -> dict[str, Any]:
    data = p4util.qcmodel_to_jsonable(atomic_input)
    if not isinstance(data, Mapping):
        raise ValidationError("SAPT(DFT) checkpoint identity requires a QCSchema AtomicInput or mapping.")
    data = dict(data)
    if "specification" in data:
        specification = dict(data.get("specification") or {})
        molecule = data.get("molecule")
    else:
        specification = {
            "driver": data.get("driver"),
            "model": data.get("model"),
            "keywords": data.get("keywords", {}),
            "protocols": data.get("protocols", {}),
            "extras": data.get("extras", {}),
        }
        molecule = data.get("molecule")
    return {"molecule": molecule, "specification": specification}


def _canonicalize_atomic_input(atomic_input: Any, *, name: str) -> dict[str, Any]:
    coerced = _coerce_atomic_input_dict(atomic_input)
    specification = dict(coerced.get("specification") or {})
    model = dict(specification.get("model") or {})
    keywords = dict(specification.get("keywords") or {})
    canonical = {
        "schema_name": "qcschema_input",
        "schema_version": 2,
        "molecule": _normalize_jsonable(coerced.get("molecule")),
        "specification": {
            "driver": str(specification.get("driver") or "energy").lower(),
            "model": {
                "method": str(model.get("method") or name).lower(),
                "basis": _normalize_jsonable(model.get("basis"), lower_strings=True),
            },
            "keywords": _normalize_jsonable(
                _strip_runtime_controls(keywords),
                lower_dict_keys=True,
                lower_strings=True,
            ),
            "protocols": _normalize_jsonable(specification.get("protocols") or {}),
            "extras": _normalize_jsonable(specification.get("extras") or {}),
        },
    }
    return canonical


def build_saptdft_job_identity(
    *,
    name: str,
    molecule,
    function_kwargs: Optional[dict[str, Any]] = None,
    atomic_input: Any = None,
) -> dict[str, Any]:
    if atomic_input is None:
        atomic_input = p4util.state_to_atomicinput(
            dtype=2,
            driver="energy",
            method=name,
            molecule=molecule,
            function_kwargs=function_kwargs,
        )
    canonical_input = _canonicalize_atomic_input(atomic_input, name=name)
    execution_fingerprint = _build_execution_fingerprint(canonical_input)
    sha256 = hashlib.sha256(
        _json_dumps(
            {
                "canonical_input": canonical_input,
                "execution_fingerprint": execution_fingerprint,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "canonical_input": canonical_input,
        "execution_fingerprint": execution_fingerprint,
        "sha256": sha256,
    }


class SAPTDFTCheckpoint:
    def __init__(self, path: Path, identity: dict[str, Any]):
        self.path = Path(path)
        self.identity = identity
        self.manifest_path = self.path / SAPTDFT_MANIFEST_FILENAME
        self.lock_path = self.path / SAPTDFT_LOCK_FILENAME
        self._lock_acquired = False
        self._manifest = self._empty_manifest()

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SAPTDFT_CHECKPOINT_SCHEMA_VERSION,
            "job_identity": self.identity,
            "completed_stages": {},
            "scalars": {},
            "artifacts": {},
        }

    def open(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            if self.manifest_path.exists():
                self._manifest = self._read_manifest()
            else:
                self._manifest = self._empty_manifest()
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        if self._lock_acquired and self.lock_path.exists():
            try:
                metadata = json.loads(self.lock_path.read_text())
            except Exception:
                metadata = {}
            if metadata.get("pid") == os.getpid():
                self.lock_path.unlink(missing_ok=True)
        self._lock_acquired = False

    def is_complete(self, stage):
        self._require_known_stage(stage)
        return self._stage_is_complete(stage)

    def restore_scalars(self, keys: Sequence[str]):
        restored = {}
        for key in keys:
            if key not in self._manifest["scalars"]:
                raise ValidationError(f"SAPT(DFT) checkpoint scalar {key} is not available in {self.path}.")
            restored[key] = self._manifest["scalars"][key]
        return restored

    def restore_array(self, name: str):
        artifact = self._manifest["artifacts"].get(name)
        if artifact is None:
            raise ValidationError(f"SAPT(DFT) checkpoint array artifact {name} is not available in {self.path}.")
        if artifact.get("kind") != "array":
            raise ValidationError(f"SAPT(DFT) checkpoint artifact {name} is not an array artifact.")
        artifact_path = self._validate_artifact(name, artifact)
        with artifact_path.open("rb") as handle:
            return np.load(handle, allow_pickle=False)

    def commit_stage(self, stage, *, scalars=None, arrays=None, wavefunctions=None):
        self._require_known_stage(stage)
        missing_dependencies = [dependency for dependency in SAPTDFT_STAGE_DEFINITIONS[stage].dependencies if not self._stage_is_complete(dependency)]
        if missing_dependencies:
            raise ValidationError(
                f"SAPT(DFT) checkpoint stage {stage} in {self.path} is missing completed dependencies {missing_dependencies}."
            )

        next_manifest = json.loads(_json_dumps(self._manifest))
        stage_scalars = self._normalize_scalars(scalars or {})
        next_manifest["scalars"].update(stage_scalars)

        artifact_names = []
        for name, array in (arrays or {}).items():
            next_manifest["artifacts"][name] = self._write_array_artifact(name, array)
            artifact_names.append(name)
        for name, wavefunction in (wavefunctions or {}).items():
            next_manifest["artifacts"][name] = self._write_wavefunction_artifact(name, wavefunction)
            artifact_names.append(name)

        next_manifest["completed_stages"][stage] = {
            "artifacts": sorted(artifact_names),
            "dependencies": list(SAPTDFT_STAGE_DEFINITIONS[stage].dependencies),
            "scalars": sorted(stage_scalars),
            "version": SAPTDFT_STAGE_DEFINITION_VERSION,
        }
        self._write_manifest_atomic(next_manifest)
        self._manifest = next_manifest

    def _require_known_stage(self, stage: str) -> None:
        if stage not in SAPTDFT_STAGE_DEFINITIONS:
            raise ValidationError(f"Unknown stage {stage!r} for SAPT(DFT) checkpoint store.")

    def _stage_is_complete(self, stage: str) -> bool:
        entry = self._manifest["completed_stages"].get(stage)
        if entry is None:
            return False
        for dependency in SAPTDFT_STAGE_DEFINITIONS[stage].dependencies:
            if not self._stage_is_complete(dependency):
                return False
        for scalar_name in entry.get("scalars", []):
            if scalar_name not in self._manifest["scalars"]:
                return False
        for artifact_name in entry.get("artifacts", []):
            artifact = self._manifest["artifacts"].get(artifact_name)
            if artifact is None:
                return False
            self._validate_artifact(artifact_name, artifact)
        return True

    def _acquire_lock(self) -> None:
        metadata = {
            "created_at": time.time(),
            "job_identity_sha256": self.identity["sha256"],
            "pid": os.getpid(),
        }
        try:
            file_descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = json.loads(self.lock_path.read_text())
            except Exception:
                existing = {}
            existing_pid = existing.get("pid")
            if isinstance(existing_pid, int) and _pid_exists(existing_pid):
                raise ValidationError(
                    f"SAPT(DFT) checkpoint lock at {self.lock_path} is held by live PID {existing_pid}."
                )
            raise ValidationError(f"SAPT(DFT) checkpoint lock at {self.lock_path} is stale and must be removed explicitly.")
        with os.fdopen(file_descriptor, "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        self._lock_acquired = True

    def _read_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except Exception as exc:
            raise ValidationError(f"SAPT(DFT) checkpoint manifest {self.manifest_path} is not valid JSON: {exc}") from exc
        self._validate_manifest(manifest)
        return manifest

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        schema_version = manifest.get("schema_version")
        if schema_version != SAPTDFT_CHECKPOINT_SCHEMA_VERSION:
            raise ValidationError(
                f"SAPT(DFT) checkpoint manifest {self.manifest_path} has unsupported schema version {schema_version}."
            )
        if not isinstance(manifest.get("completed_stages"), Mapping):
            raise ValidationError(f"SAPT(DFT) checkpoint manifest {self.manifest_path} is missing completed_stages.")
        if not isinstance(manifest.get("scalars"), Mapping):
            raise ValidationError(f"SAPT(DFT) checkpoint manifest {self.manifest_path} is missing scalars.")
        if not isinstance(manifest.get("artifacts"), Mapping):
            raise ValidationError(f"SAPT(DFT) checkpoint manifest {self.manifest_path} is missing artifacts.")

        job_identity = manifest.get("job_identity")
        if not isinstance(job_identity, Mapping):
            raise ValidationError(f"SAPT(DFT) checkpoint manifest {self.manifest_path} is missing job_identity.")
        difference = _first_difference(job_identity.get("canonical_input"), self.identity.get("canonical_input"))
        if difference:
            raise ValidationError(f"SAPT(DFT) checkpoint identity mismatch for {self.path}: {difference}")
        difference = _first_difference(job_identity.get("execution_fingerprint"), self.identity.get("execution_fingerprint"))
        if difference:
            raise ValidationError(f"SAPT(DFT) checkpoint execution fingerprint mismatch for {self.path}: {difference}")
        if job_identity.get("sha256") != self.identity.get("sha256"):
            raise ValidationError(f"SAPT(DFT) checkpoint digest mismatch for {self.path}.")

        for stage, entry in manifest["completed_stages"].items():
            self._require_known_stage(stage)
            if not isinstance(entry, Mapping):
                raise ValidationError(f"SAPT(DFT) checkpoint stage entry {stage} in {self.manifest_path} must be a mapping.")
            entry_dependencies = tuple(entry.get("dependencies", []))
            for dependency in entry_dependencies:
                self._require_known_stage(dependency)
            if entry_dependencies != SAPTDFT_STAGE_DEFINITIONS[stage].dependencies:
                raise ValidationError(
                    f"SAPT(DFT) checkpoint stage {stage} in {self.manifest_path} has dependency metadata {entry_dependencies} but expected {SAPTDFT_STAGE_DEFINITIONS[stage].dependencies}."
                )
            if entry.get("version") != SAPTDFT_STAGE_DEFINITION_VERSION:
                raise ValidationError(
                    f"SAPT(DFT) checkpoint stage {stage} in {self.manifest_path} has unsupported definition version {entry.get('version')}."
                )
            missing_dependencies = [dependency for dependency in SAPTDFT_STAGE_DEFINITIONS[stage].dependencies if dependency not in manifest["completed_stages"]]
            if missing_dependencies:
                raise ValidationError(
                    f"SAPT(DFT) checkpoint stage {stage} in {self.manifest_path} is missing dependencies {missing_dependencies}."
                )
            for scalar_name in entry.get("scalars", []):
                if scalar_name not in manifest["scalars"]:
                    raise ValidationError(
                        f"SAPT(DFT) checkpoint stage {stage} in {self.manifest_path} references missing scalar {scalar_name}."
                    )
            for artifact_name in entry.get("artifacts", []):
                if artifact_name not in manifest["artifacts"]:
                    raise ValidationError(
                        f"SAPT(DFT) checkpoint stage {stage} in {self.manifest_path} references missing artifact {artifact_name}."
                    )
                self._validate_artifact(artifact_name, manifest["artifacts"][artifact_name])

    def _validate_artifact(self, name: str, artifact: Mapping[str, Any]) -> Path:
        kind = artifact.get("kind")
        if kind not in _ALLOWED_ARTIFACT_KINDS:
            raise ValidationError(f"SAPT(DFT) checkpoint artifact {name} in {self.manifest_path} has unknown kind {kind!r}.")
        relpath = artifact.get("path")
        if not isinstance(relpath, str) or not relpath:
            raise ValidationError(f"SAPT(DFT) checkpoint artifact {name} in {self.manifest_path} is missing a valid path.")
        artifact_path = self._resolve_artifact_path(name, relpath)
        if not artifact_path.exists():
            raise ValidationError(f"SAPT(DFT) checkpoint artifact {name} is missing: {artifact_path}")
        sha256, size = _file_digest_and_size(artifact_path)
        if sha256 != artifact.get("sha256"):
            raise ValidationError(f"SAPT(DFT) checkpoint artifact {name} in {artifact_path} failed checksum validation.")
        if size != artifact.get("size"):
            raise ValidationError(f"SAPT(DFT) checkpoint artifact {name} in {artifact_path} failed size validation.")
        return artifact_path

    def _resolve_artifact_path(self, name: str, relpath: str) -> Path:
        artifact_relpath = Path(relpath)
        if artifact_relpath.is_absolute():
            raise ValidationError(
                f"SAPT(DFT) checkpoint artifact {name} in {self.manifest_path} must stay inside the checkpoint directory; absolute path {relpath!r} is not allowed."
            )
        if ".." in artifact_relpath.parts:
            raise ValidationError(
                f"SAPT(DFT) checkpoint artifact {name} in {self.manifest_path} must stay inside the checkpoint directory; traversal path {relpath!r} is not allowed."
            )
        base_path = self.path.resolve()
        artifact_path = (self.path / artifact_relpath).resolve(strict=False)
        try:
            artifact_path.relative_to(base_path)
        except ValueError as exc:
            raise ValidationError(
                f"SAPT(DFT) checkpoint artifact {name} in {self.manifest_path} resolves outside the checkpoint directory: {relpath!r}."
            ) from exc
        return artifact_path

    def _normalize_scalars(self, scalars: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, value in scalars.items():
            value = _normalize_scalar(value)
            if isinstance(value, (bool, float, int, str)) or value is None:
                normalized[str(key)] = value
                continue
            raise ValidationError(
                f"SAPT(DFT) checkpoint scalar {key} in {self.path} must be a JSON scalar, not {type(value).__name__}."
            )
        return normalized

    def _write_array_artifact(self, name: str, array: Any) -> dict[str, Any]:
        array = np.asarray(array)
        suffix = f"{_safe_artifact_stem(name)}--{uuid.uuid4().hex}.npy"
        tmp_path = self.path / f".{suffix}.tmp"
        with tmp_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        sha256, size = _file_digest_and_size(tmp_path)
        final_path = self.path / suffix
        os.replace(tmp_path, final_path)
        return {"kind": "array", "path": final_path.name, "sha256": sha256, "size": size}

    def _write_wavefunction_artifact(self, name: str, wavefunction: Any) -> dict[str, Any]:
        base_name = f"{_safe_artifact_stem(name)}--{uuid.uuid4().hex}"
        tmp_base = self.path / f".{base_name}.tmp"
        wavefunction.to_file(str(tmp_base))
        tmp_path = Path(f"{tmp_base}.npy")
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        sha256, size = _file_digest_and_size(tmp_path)
        final_path = self.path / f"{base_name}.npy"
        os.replace(tmp_path, final_path)
        return {"kind": "scf_snapshot", "path": final_path.name, "sha256": sha256, "size": size}

    def _write_manifest_atomic(self, manifest: Mapping[str, Any]) -> None:
        tmp_path = self.path / f".{SAPTDFT_MANIFEST_FILENAME}.{uuid.uuid4().hex}.tmp"
        with tmp_path.open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.manifest_path)


def _safe_artifact_stem(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return safe or "artifact"


def _file_digest_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _normalize_scf_reference(reference: str) -> str:
    normalized = str(reference).upper()
    if normalized not in _SCF_SNAPSHOT_REQUIRED_FLOATVARS:
        raise ValidationError(f"SCF snapshot support is currently limited to RHF and RKS, not {reference!r}.")
    return normalized


def _expected_scf_snapshot_required_fields(reference: str) -> dict[str, list[str]]:
    normalized = _normalize_scf_reference(reference)
    return {
        "matrix": list(_SCF_SNAPSHOT_REQUIRED_FIELDS["matrix"]),
        "vector": list(_SCF_SNAPSHOT_REQUIRED_FIELDS["vector"]),
        "floatvar": list(_SCF_SNAPSHOT_REQUIRED_FLOATVARS[normalized]),
    }


def _load_scf_snapshot_data(snapshot: Any) -> dict[str, Any]:
    if isinstance(snapshot, Mapping):
        return dict(snapshot)

    snapshot_path = os.fspath(snapshot)
    if isinstance(snapshot_path, str) and not snapshot_path.endswith(".npy"):
        snapshot_path = snapshot_path + ".npy"

    try:
        loaded = np.load(snapshot_path, allow_pickle=True).item()
    except Exception as exc:
        raise ValidationError(f"SCF snapshot {snapshot!r} could not be loaded: {exc}") from exc

    if not isinstance(loaded, Mapping):
        raise ValidationError(f"SCF snapshot {snapshot!r} must deserialize to a mapping.")

    return dict(loaded)


def _validate_scf_snapshot_required_fields(snapshot_data: Mapping[str, Any], required_fields: Mapping[str, Sequence[str]]) -> None:
    for section_name, field_names in required_fields.items():
        section = snapshot_data.get(section_name)
        if not isinstance(section, Mapping):
            raise ValidationError(f"SCF snapshot is missing required {section_name} section.")
        for field_name in field_names:
            if field_name not in section or section[field_name] is None:
                raise ValidationError(f"SCF snapshot is missing required {section_name} field {field_name}.")


def _maybe_get_basisset(wfn: core.Wavefunction, key: str):
    try:
        return wfn.get_basisset(key)
    except RuntimeError:
        return None


def _capture_factory_basissets(wfn: core.Wavefunction) -> dict[str, Any]:
    basissets = {}
    for key in _SCF_SNAPSHOT_FACTORY_BASIS_KEYS:
        if (basis := _maybe_get_basisset(wfn, key)) is not None:
            basissets[key] = basis
    return basissets


def capture_scf_snapshot(wfn, *, reference: str, method: str) -> dict[str, object]:
    normalized_reference = _normalize_scf_reference(reference)
    snapshot = wfn.to_file()
    required_fields = _expected_scf_snapshot_required_fields(normalized_reference)
    _validate_scf_snapshot_required_fields(snapshot, required_fields)
    snapshot[_SCF_SNAPSHOT_METADATA_KEY] = {
        "basis": {
            "name": snapshot["string"]["basisname"],
            "puream": snapshot["boolean"]["basispuream"],
        },
        "dimensions": {key: list(snapshot["dimension"][key]) for key in _SCF_SNAPSHOT_DIMENSION_KEYS},
        "functional": wfn.functional().name(),
        "method": str(method).lower(),
        "molecule": _normalize_jsonable(snapshot["molecule"]),
        "reference": normalized_reference,
        "required_fields": required_fields,
        "version": SAPTDFT_SCF_SNAPSHOT_VERSION,
    }
    return snapshot


def _validate_scf_snapshot_metadata(snapshot_data: Mapping[str, Any], loaded: core.Wavefunction, *, method: str, reference: str) -> dict[str, Any]:
    metadata = snapshot_data.get(_SCF_SNAPSHOT_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        raise ValidationError("SCF snapshot metadata is missing.")

    if metadata.get("version") != SAPTDFT_SCF_SNAPSHOT_VERSION:
        raise ValidationError(f"SCF snapshot has unsupported version {metadata.get('version')}.")

    normalized_reference = _normalize_scf_reference(reference)
    if str(metadata.get("reference", "")).upper() != normalized_reference:
        raise ValidationError(
            f"SCF snapshot reference mismatch: expected {normalized_reference}, got {metadata.get('reference')!r}."
        )

    normalized_method = str(method).lower()
    if str(metadata.get("method", "")).lower() != normalized_method:
        raise ValidationError(f"SCF snapshot method mismatch: expected {normalized_method!r}.")

    expected_required_fields = _expected_scf_snapshot_required_fields(normalized_reference)
    metadata_required_fields = metadata.get("required_fields")
    if not isinstance(metadata_required_fields, Mapping):
        raise ValidationError("SCF snapshot required field metadata is missing.")
    if _first_difference(_normalize_jsonable(metadata_required_fields), _normalize_jsonable(expected_required_fields)):
        raise ValidationError("SCF snapshot required field metadata does not match the expected schema.")
    _validate_scf_snapshot_required_fields(snapshot_data, expected_required_fields)

    expected_molecule = _normalize_jsonable(snapshot_data.get("molecule"))
    metadata_molecule = _normalize_jsonable(metadata.get("molecule"))
    molecule_difference = _first_difference(metadata_molecule, expected_molecule)
    if molecule_difference:
        raise ValidationError(f"SCF snapshot molecule metadata mismatch: {molecule_difference}")
    loaded_molecule = _normalize_jsonable(loaded.molecule().to_dict())
    molecule_difference = _first_difference(metadata_molecule, loaded_molecule)
    if molecule_difference:
        raise ValidationError(f"SCF snapshot molecule does not match the deserialized wavefunction: {molecule_difference}")

    metadata_basis = metadata.get("basis")
    if not isinstance(metadata_basis, Mapping):
        raise ValidationError("SCF snapshot basis metadata is missing.")
    expected_basis = {
        "name": snapshot_data.get("string", {}).get("basisname"),
        "puream": snapshot_data.get("boolean", {}).get("basispuream"),
    }
    basis_difference = _first_difference(_normalize_jsonable(metadata_basis), _normalize_jsonable(expected_basis))
    if basis_difference:
        raise ValidationError(f"SCF snapshot basis metadata mismatch: {basis_difference}")
    if loaded.basisset().name() != metadata_basis.get("name"):
        raise ValidationError(
            f"SCF snapshot basis mismatch: expected {metadata_basis.get('name')!r}, got {loaded.basisset().name()!r}."
        )
    if loaded.basisset().has_puream() != metadata_basis.get("puream"):
        raise ValidationError("SCF snapshot basis puream metadata mismatch.")

    metadata_dimensions = metadata.get("dimensions")
    if not isinstance(metadata_dimensions, Mapping):
        raise ValidationError("SCF snapshot dimensions metadata is missing.")
    expected_dimensions = snapshot_data.get("dimension")
    if not isinstance(expected_dimensions, Mapping):
        raise ValidationError("SCF snapshot dimension section is missing.")
    for key in _SCF_SNAPSHOT_DIMENSION_KEYS:
        expected_dimension = list(expected_dimensions.get(key) or [])
        if list(metadata_dimensions.get(key) or []) != expected_dimension:
            raise ValidationError(f"SCF snapshot dimensions mismatch for {key}.")
        if list(getattr(loaded, key)().to_tuple()) != expected_dimension:
            raise ValidationError(f"SCF snapshot deserialized dimensions mismatch for {key}.")

    return dict(metadata)


def rehydrate_scf_wavefunction(snapshot, *, method: str, reference: str) -> core.HF:
    snapshot_data = _load_scf_snapshot_data(snapshot)
    loaded = core.Wavefunction.from_file(snapshot_data)
    metadata = _validate_scf_snapshot_metadata(snapshot_data, loaded, method=method, reference=reference)

    from ..proc import scf_wavefunction_factory

    fresh_base = core.Wavefunction.build(loaded.molecule(), loaded.basisset())
    rehydrated = scf_wavefunction_factory(method, fresh_base, _normalize_scf_reference(reference))
    factory_basissets = _capture_factory_basissets(rehydrated)
    rehydrated.deep_copy(loaded)
    for key, basis in factory_basissets.items():
        rehydrated.set_basisset(key, basis)

    if rehydrated.functional().name() != metadata.get("functional"):
        raise ValidationError(
            f"SCF snapshot functional mismatch: expected {metadata.get('functional')!r}, got {rehydrated.functional().name()!r}."
        )

    for key in _SCF_SNAPSHOT_DIMENSION_KEYS:
        expected_dimension = tuple(snapshot_data["dimension"][key])
        if getattr(rehydrated, key)().to_tuple() != expected_dimension:
            raise ValidationError(f"SCF snapshot rehydrated dimensions mismatch for {key}.")

    rehydrated.set_energy(loaded.energy())
    return rehydrated


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
