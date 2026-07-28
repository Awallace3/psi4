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
    "selected_stage_dependencies",
    "selected_stages",
]

SAPTDFT_CHECKPOINT_SCHEMA_VERSION = 1
SAPTDFT_STAGE_DEFINITION_VERSION = 2
SAPTDFT_SCF_SNAPSHOT_VERSION = 1
SAPTDFT_MANIFEST_FILENAME = "saptdft_state.json"
SAPTDFT_LOCK_FILENAME = "saptdft_state.lock"
_ALLOWED_ARTIFACT_KINDS = {"array", "scf_snapshot", "wavefunction"}
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
_SCF_SNAPSHOT_SERIALIZED_SECTIONS = {
    "molecule": Mapping,
    "matrix": Mapping,
    "vector": Mapping,
    "dimension": Mapping,
    "int": Mapping,
    "string": Mapping,
    "boolean": Mapping,
    "float": Mapping,
    "floatvar": Mapping,
    "matrixarr": Mapping,
    _SCF_SNAPSHOT_METADATA_KEY: Mapping,
}
_SCF_SNAPSHOT_REQUIRED_SECTION_KEYS = {
    "string": ("basisname",),
    "boolean": ("basispuream",),
    "dimension": _SCF_SNAPSHOT_DIMENSION_KEYS,
    _SCF_SNAPSHOT_METADATA_KEY: ("basis", "dimensions", "functional", "method", "molecule", "reference", "required_fields", "version"),
}
_SCF_SNAPSHOT_REQUIRED_FLOATVARS = {
    "RHF": ("CURRENT ENERGY", "CURRENT REFERENCE ENERGY", "SCF TOTAL ENERGY", "HF TOTAL ENERGY"),
    "RKS": ("CURRENT ENERGY", "CURRENT REFERENCE ENERGY", "SCF TOTAL ENERGY", "DFT TOTAL ENERGY"),
}
_IDENTITY_ATOMIC_INPUT_KEY = "_saptdft_checkpoint_atomic_input"

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
    _IDENTITY_ATOMIC_INPUT_KEY,
    "psi4_checkpoint_dir",
    "psi4_checkpoint_stop_after",
    "threads",
    "timer",
    "timing",
    "verbose",
    "verbosity",
}

_MOLECULE_IDENTITY_FIELDS = (
    "atom_labels",
    "atomic_numbers",
    "fix_com",
    "fix_orientation",
    "fix_symmetry",
    "fragment_charges",
    "fragment_multiplicities",
    "fragments",
    "geometry",
    "mass_numbers",
    "masses",
    "molecular_charge",
    "molecular_multiplicity",
    "real",
    "symbols",
)

_QCSCHEMA_PROTOCOL_DEFAULTS = {
    "error_correction": {"default_policy": True, "policies": None},
    "native_files": "none",
    "stdout": True,
    "wavefunction": "none",
}
_QCSCHEMA_PROTOCOL_NOISE_KEYS = {"schema_name", "schema_version"}
_QCSCHEMA_RUNTIME_EXTRA_KEYS = {"current_qcvars_only", "extra_infiles", "wfn_qcvars_only"}


@dataclass(frozen=True)
class StageDefinition:
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    version: int = SAPTDFT_STAGE_DEFINITION_VERSION


SAPTDFT_STAGE_DEFINITIONS: dict[str, StageDefinition] = {
    "grac_monomer_a": StageDefinition(),
    "grac_monomer_b": StageDefinition(),
    "hf_dimer_scf": StageDefinition(),
    "hf_monomer_a_scf": StageDefinition(dependencies=("hf_dimer_scf",)),
    "hf_monomer_b_scf": StageDefinition(dependencies=("hf_monomer_a_scf",)),
    "hf_sapt_elst": StageDefinition(dependencies=("hf_monomer_b_scf",)),
    "hf_sapt_exch": StageDefinition(dependencies=("hf_sapt_elst",)),
    "hf_sapt_ind": StageDefinition(dependencies=("hf_sapt_exch",)),
    "dimer_localization_scf": StageDefinition(),
    "monomer_a_dft_scf": StageDefinition(),
    "monomer_b_dft_scf": StageDefinition(dependencies=("monomer_a_dft_scf",)),
    "delta_dft_dimer_scf": StageDefinition(dependencies=("monomer_b_dft_scf",)),
    "delta_dft_monomer_a_scf": StageDefinition(dependencies=("delta_dft_dimer_scf",)),
    "delta_dft_monomer_b_scf": StageDefinition(dependencies=("delta_dft_monomer_a_scf",)),
    "delta_dft": StageDefinition(dependencies=("delta_dft_monomer_b_scf",)),
    "elst": StageDefinition(),
    "exch": StageDefinition(dependencies=("elst",)),
    "ind": StageDefinition(dependencies=("exch",)),
    "disp": StageDefinition(dependencies=("ind",)),
    "d3": StageDefinition(),
    "d4": StageDefinition(),
    "fsapt_setup": StageDefinition(),
    "fsapt_elst": StageDefinition(dependencies=("fsapt_setup",)),
    "fsapt_exch": StageDefinition(dependencies=("fsapt_elst",)),
    "fsapt_ind": StageDefinition(dependencies=("fsapt_exch",)),
    "fsapt_disp": StageDefinition(dependencies=("fsapt_ind",)),
    "fsapt_final": StageDefinition(dependencies=("fsapt_disp",)),
    "final": StageDefinition(dependencies=("ind",)),
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


@dataclass(frozen=True)
class FSAPTArtifactSpec:
    cache_key: str
    artifact_name: str
    value_type: str


_FSAPT_SETUP_ARTIFACT_SPECS: tuple[FSAPTArtifactSpec, ...] = (
    FSAPTArtifactSpec("Qocc0A", "fsapt_setup.Qocc0A", "matrix"),
    FSAPTArtifactSpec("Qocc0B", "fsapt_setup.Qocc0B", "matrix"),
    FSAPTArtifactSpec("Locc_A", "fsapt_setup.Locc_A", "matrix"),
    FSAPTArtifactSpec("Locc_B", "fsapt_setup.Locc_B", "matrix"),
    FSAPTArtifactSpec("Uocc_A", "fsapt_setup.Uocc_A", "matrix"),
    FSAPTArtifactSpec("Uocc_B", "fsapt_setup.Uocc_B", "matrix"),
    FSAPTArtifactSpec("Lfocc0A", "fsapt_setup.Lfocc0A", "matrix"),
    FSAPTArtifactSpec("Lfocc0B", "fsapt_setup.Lfocc0B", "matrix"),
    FSAPTArtifactSpec("Laocc0A", "fsapt_setup.Laocc0A", "matrix"),
    FSAPTArtifactSpec("Laocc0B", "fsapt_setup.Laocc0B", "matrix"),
    FSAPTArtifactSpec("Uaocc0A", "fsapt_setup.Uaocc0A", "matrix"),
    FSAPTArtifactSpec("Uaocc0B", "fsapt_setup.Uaocc0B", "matrix"),
    FSAPTArtifactSpec("Caocc0A", "fsapt_setup.Caocc0A", "matrix"),
    FSAPTArtifactSpec("Caocc0B", "fsapt_setup.Caocc0B", "matrix"),
    FSAPTArtifactSpec("ZA", "fsapt_setup.ZA", "vector"),
    FSAPTArtifactSpec("ZA_orig", "fsapt_setup.ZA_orig", "vector"),
    FSAPTArtifactSpec("ZB", "fsapt_setup.ZB", "vector"),
    FSAPTArtifactSpec("ZB_orig", "fsapt_setup.ZB_orig", "vector"),
    FSAPTArtifactSpec("ZC", "fsapt_setup.ZC", "vector"),
    FSAPTArtifactSpec("ZC_orig", "fsapt_setup.ZC_orig", "vector"),
)
_FSAPT_STAGE_ARTIFACT_SPECS: dict[str, tuple[FSAPTArtifactSpec, ...]] = {
    "fsapt_setup": _FSAPT_SETUP_ARTIFACT_SPECS,
    "fsapt_elst": (
        FSAPTArtifactSpec("Elst_AB", "fsapt_elst.Elst_AB", "matrix"),
        FSAPTArtifactSpec("Vlocc0A", "fsapt_elst.Vlocc0A", "matrix"),
        FSAPTArtifactSpec("Vlocc0B", "fsapt_elst.Vlocc0B", "matrix"),
    ),
    "fsapt_exch": (FSAPTArtifactSpec("Exch_AB", "fsapt_exch.Exch_AB", "matrix"),),
    "fsapt_ind": (
        FSAPTArtifactSpec("IndAB_AB", "fsapt_ind.IndAB_AB", "matrix"),
        FSAPTArtifactSpec("IndBA_AB", "fsapt_ind.IndBA_AB", "matrix"),
        FSAPTArtifactSpec("Disp_AB", "fsapt_ind.Disp_AB", "matrix"),
    ),
    "fsapt_disp": (FSAPTArtifactSpec("Disp_AB", "fsapt_disp.Disp_AB", "matrix"),),
}


def _fsapt_stage_artifact_specs(stage: str) -> tuple[FSAPTArtifactSpec, ...]:
    return _FSAPT_STAGE_ARTIFACT_SPECS.get(stage, ())


def _fsapt_payload_array(value: Any) -> np.ndarray:
    if hasattr(value, "np"):
        return np.asarray(value.np)
    return np.asarray(value)


def fsapt_stage_arrays(stage: str, cache: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = {}
    for spec in _fsapt_stage_artifact_specs(stage):
        if spec.cache_key in cache:
            arrays[spec.artifact_name] = _fsapt_payload_array(cache[spec.cache_key])
    return arrays


def restore_fsapt_stage_cache(stage: str, checkpoint, cache: dict[str, Any]) -> dict[str, Any]:
    for spec in _fsapt_stage_artifact_specs(stage):
        artifact = checkpoint._manifest["artifacts"].get(spec.artifact_name)
        if artifact is None:
            continue
        array = checkpoint.restore_array(spec.artifact_name)
        if spec.value_type == "matrix":
            restored = core.Matrix.from_array(array)
        elif spec.value_type == "vector":
            restored = core.Vector.from_array(np.asarray(array).reshape(-1))
        else:
            raise ValidationError(f"Unsupported F-SAPT checkpoint payload type {spec.value_type!r} for {spec.cache_key}.")
        restored.name = spec.cache_key
        cache[spec.cache_key] = restored
    return cache


def _option_is_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper() not in {"", "0", "FALSE", "OFF", "NO", "NONE"}
    return bool(value)


def _identity_keywords(identity: Mapping[str, Any]) -> dict[str, Any]:
    specification = identity.get("canonical_input", {}).get("specification", {})
    keywords = specification.get("keywords", {})
    if not isinstance(keywords, Mapping):
        return {}
    return {str(key).lower(): value for key, value in keywords.items()}


def _identity_method(identity: Mapping[str, Any]) -> str:
    canonical_input = identity.get("canonical_input", {})
    if not isinstance(canonical_input, Mapping):
        return ""

    specification = canonical_input.get("specification", {})
    if isinstance(specification, Mapping):
        model = specification.get("model", {})
        if isinstance(model, Mapping) and model.get("method") is not None:
            return str(model.get("method")).lower()

    model = canonical_input.get("model", {})
    if isinstance(model, Mapping) and model.get("method") is not None:
        return str(model.get("method")).lower()

    return ""


def _selected_stage_options(identity: Mapping[str, Any]) -> dict[str, Any]:
    keywords = _identity_keywords(identity)
    method = _identity_method(identity)
    functional = str(keywords.get("sapt_dft_functional", "pbe0")).upper()
    do_dft = functional != "HF"
    do_delta_hf = _option_is_enabled(keywords.get("sapt_dft_do_dhf", True))
    do_delta_dft = do_dft and _option_is_enabled(keywords.get("sapt_dft_do_ddft", False))
    do_disp = _option_is_enabled(keywords.get("sapt_dft_do_disp", True))
    do_d3 = _option_is_enabled(keywords.get("sapt_dft_d3_ie", False))
    do_d4 = _option_is_enabled(keywords.get("sapt_dft_d4_ie", False))
    method_upper = method.upper()
    if "-D4" in method_upper:
        do_d4 = True
        do_d3 = False
        do_disp = False
        do_delta_dft = do_dft and ("DFT-D4" in method_upper)
    elif "-D3" in method_upper:
        do_d3 = True
        do_d4 = False
        do_disp = False
        do_delta_dft = do_dft and ("DFT-D3" in method_upper)
    fsapt_mode = str(keywords.get("sapt_dft_do_fsapt", "none")).upper()
    do_fsapt = fsapt_mode != "NONE"
    do_grac = do_dft and str(keywords.get("sapt_dft_grac_compute", "none")).upper() != "NONE"
    localization_path = do_fsapt and not do_delta_hf
    return {
        "do_dft": do_dft,
        "do_delta_hf": do_delta_hf,
        "do_delta_dft": do_delta_dft,
        "do_disp": do_disp,
        "do_d3": do_d3,
        "do_d4": do_d4,
        "do_fsapt": do_fsapt,
        "fsapt_mode": fsapt_mode,
        "do_grac": do_grac,
        "localization_path": localization_path,
        "method": method,
    }


def selected_stages(identity: Mapping[str, Any]) -> tuple[str, ...]:
    options = _selected_stage_options(identity)
    stages = []

    if options["do_grac"]:
        stages.extend(["grac_monomer_a", "grac_monomer_b"])

    if options["do_delta_hf"]:
        stages.extend(["hf_dimer_scf", "hf_monomer_a_scf", "hf_monomer_b_scf"])
        if options["do_dft"]:
            stages.extend(["hf_sapt_elst", "hf_sapt_exch", "hf_sapt_ind"])

    if options["localization_path"]:
        stages.append("dimer_localization_scf")

    if options["do_dft"] or not options["do_delta_hf"]:
        stages.extend(["monomer_a_dft_scf", "monomer_b_dft_scf"])

    if options["do_delta_dft"]:
        stages.extend(
            [
                "delta_dft_dimer_scf",
                "delta_dft_monomer_a_scf",
                "delta_dft_monomer_b_scf",
                "delta_dft",
            ]
        )

    stages.extend(["elst", "exch", "ind"])

    if options["do_disp"]:
        stages.append("disp")
    if options["do_d3"]:
        stages.append("d3")
    if options["do_d4"]:
        stages.append("d4")

    if options["do_fsapt"]:
        stages.extend(["fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind"])
        if options["do_disp"]:
            stages.append("fsapt_disp")
        stages.append("fsapt_final")

    stages.append("final")
    return tuple(stages)


def selected_stage_dependencies(identity: Mapping[str, Any], stage: str) -> tuple[str, ...]:
    options = _selected_stage_options(identity)
    grac_dependencies = ("grac_monomer_a", "grac_monomer_b") if options["do_grac"] else ()

    if stage == "hf_dimer_scf":
        return grac_dependencies
    if stage == "hf_monomer_a_scf":
        return ("hf_dimer_scf",)
    if stage == "hf_monomer_b_scf":
        return ("hf_monomer_a_scf",)
    if stage == "hf_sapt_elst":
        return ("hf_monomer_b_scf",)
    if stage == "hf_sapt_exch":
        return ("hf_sapt_elst",)
    if stage == "hf_sapt_ind":
        return ("hf_sapt_exch",)
    if stage == "dimer_localization_scf":
        return grac_dependencies if options["localization_path"] else ()
    if stage == "monomer_a_dft_scf":
        if options["localization_path"]:
            return ("dimer_localization_scf",)
        return grac_dependencies
    if stage == "monomer_b_dft_scf":
        return ("monomer_a_dft_scf",)
    if stage == "delta_dft_dimer_scf":
        return ("monomer_b_dft_scf",)
    if stage == "delta_dft_monomer_a_scf":
        return ("delta_dft_dimer_scf",)
    if stage == "delta_dft_monomer_b_scf":
        return ("delta_dft_monomer_a_scf",)
    if stage == "delta_dft":
        return ("delta_dft_monomer_b_scf",)
    if stage == "elst":
        if options["do_delta_dft"]:
            return ("delta_dft",)
        if options["do_dft"] or not options["do_delta_hf"]:
            return ("monomer_b_dft_scf",)
        return ("hf_monomer_b_scf",)
    if stage == "exch":
        return ("elst",)
    if stage == "ind":
        return ("exch",)
    if stage == "disp":
        return ("ind",)
    if stage == "d3":
        if options["do_delta_dft"]:
            return ("delta_dft",)
        if options["do_dft"] or not options["do_delta_hf"]:
            return ("monomer_b_dft_scf",)
        return ("hf_monomer_b_scf",)
    if stage == "d4":
        if options["do_delta_dft"]:
            return ("delta_dft",)
        if options["do_dft"] or not options["do_delta_hf"]:
            return ("monomer_b_dft_scf",)
        return ("hf_monomer_b_scf",)
    if stage == "fsapt_setup":
        return ("ind",)
    if stage == "fsapt_elst":
        return ("fsapt_setup",)
    if stage == "fsapt_exch":
        return ("fsapt_elst",)
    if stage == "fsapt_ind":
        return ("fsapt_exch",)
    if stage == "fsapt_disp":
        return ("fsapt_ind",)
    if stage == "fsapt_final":
        return ("fsapt_disp",) if options["do_disp"] else ("fsapt_ind",)
    if stage == "final":
        final_dependencies = []
        if options["do_fsapt"]:
            final_dependencies.append("fsapt_final")
        else:
            final_dependencies.append("ind")
        if options["do_d4"]:
            final_dependencies.append("d4")
        elif options["do_d3"]:
            final_dependencies.append("d3")
        elif options["do_disp"]:
            final_dependencies.append("disp")
        return tuple(final_dependencies)
    return SAPTDFT_STAGE_DEFINITIONS[stage].dependencies


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


def _normalize_identity_jsonable(value: Any) -> Any:
    value = _normalize_scalar(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_identity_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_identity_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_normalize_identity_jsonable(item) for item in value]
    if isinstance(value, float):
        if abs(value) < 1.0e-14:
            return 0.0
        rounded = round(value, 12)
        if rounded.is_integer():
            return int(rounded)
        return rounded
    return value


def _canonicalize_molecule(molecule: Any) -> Any:
    normalized = _normalize_identity_jsonable(_normalize_jsonable(molecule))
    if not isinstance(normalized, Mapping):
        return normalized
    return {key: normalized[key] for key in _MOLECULE_IDENTITY_FIELDS if key in normalized}


def _canonicalize_identity_keywords(keywords: Mapping[str, Any]) -> dict[str, Any]:
    stripped_keywords = _strip_runtime_controls(keywords)
    if not isinstance(stripped_keywords, Mapping):
        return {}

    normalized_keywords = _normalize_jsonable(
        stripped_keywords,
        lower_dict_keys=True,
        lower_strings=True,
    )
    keyword_defaults = _normalize_jsonable(
        p4util.saptdft_identity_defaults_for_keys(normalized_keywords.keys()),
        lower_dict_keys=True,
        lower_strings=True,
    )
    return {
        key: value
        for key, value in normalized_keywords.items()
        if key not in keyword_defaults or value != keyword_defaults[key]
    }



def _canonicalize_identity_protocols(protocols: Mapping[str, Any]) -> dict[str, Any]:
    normalized_protocols = _normalize_identity_jsonable(_normalize_jsonable(protocols))
    if not isinstance(normalized_protocols, Mapping):
        return {}

    defaults = _normalize_identity_jsonable(_QCSCHEMA_PROTOCOL_DEFAULTS)
    canonical = {}
    for key, value in normalized_protocols.items():
        key_str = str(key)
        if key_str.lower() in _QCSCHEMA_PROTOCOL_NOISE_KEYS:
            continue
        if key_str in defaults and _first_difference(value, defaults[key_str]) is None:
            continue
        canonical[key_str] = value
    return canonical



def _canonicalize_identity_extras(extras: Mapping[str, Any]) -> dict[str, Any]:
    normalized_extras = _normalize_identity_jsonable(_normalize_jsonable(extras))
    if not isinstance(normalized_extras, Mapping):
        return {}

    canonical = {}
    for key, value in normalized_extras.items():
        if str(key).lower() in _QCSCHEMA_RUNTIME_EXTRA_KEYS:
            continue
        canonical[str(key)] = value
    return canonical



def _canonicalize_atomic_input(atomic_input: Any, *, name: str) -> dict[str, Any]:
    coerced = _coerce_atomic_input_dict(atomic_input)
    specification = dict(coerced.get("specification") or {})
    model = dict(specification.get("model") or {})
    canonical = {
        "schema_name": "qcschema_input",
        "schema_version": 2,
        "molecule": _canonicalize_molecule(coerced.get("molecule")),
        "specification": {
            "driver": str(specification.get("driver") or "energy").lower(),
            "model": {
                "method": str(model.get("method") or name).lower(),
                "basis": _normalize_jsonable(model.get("basis"), lower_strings=True),
            },
            "keywords": _canonicalize_identity_keywords(specification.get("keywords") or {}),
            "protocols": _canonicalize_identity_protocols(specification.get("protocols") or {}),
            "extras": _canonicalize_identity_extras(specification.get("extras") or {}),
        },
    }
    return canonical


def _merge_identity_atomic_inputs(resolved_atomic_input: Any, provided_atomic_input: Any) -> dict[str, Any]:
    resolved = _coerce_atomic_input_dict(resolved_atomic_input)
    provided = _coerce_atomic_input_dict(provided_atomic_input)

    resolved_specification = dict(resolved.get("specification") or {})
    provided_specification = dict(provided.get("specification") or {})
    resolved_model = dict(resolved_specification.get("model") or {})
    provided_model = dict(provided_specification.get("model") or {})
    resolved_keywords = dict(resolved_specification.get("keywords") or {})
    provided_keywords = dict(provided_specification.get("keywords") or {})
    resolved_protocols = dict(resolved_specification.get("protocols") or {})
    provided_protocols = dict(provided_specification.get("protocols") or {})
    resolved_extras = dict(resolved_specification.get("extras") or {})
    provided_extras = dict(provided_specification.get("extras") or {})

    return {
        "molecule": resolved.get("molecule"),
        "specification": {
            "driver": resolved_specification.get("driver") or provided_specification.get("driver"),
            "model": {
                "method": resolved_model.get("method") or provided_model.get("method"),
                "basis": resolved_model.get("basis") or provided_model.get("basis"),
            },
            "keywords": {**provided_keywords, **resolved_keywords},
            "protocols": {**resolved_protocols, **provided_protocols},
            "extras": {**resolved_extras, **provided_extras},
        },
    }


def build_saptdft_job_identity(
    *,
    name: str,
    molecule,
    function_kwargs: Optional[dict[str, Any]] = None,
    atomic_input: Any = None,
) -> dict[str, Any]:
    resolved_atomic_input = p4util.state_to_atomicinput(
        dtype=2,
        driver="energy",
        method=name,
        molecule=molecule,
        function_kwargs=function_kwargs,
    )
    if atomic_input is None:
        atomic_input = resolved_atomic_input
    else:
        atomic_input = _merge_identity_atomic_inputs(resolved_atomic_input, atomic_input)
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

    def next_unfinished_stage(self, stages: Optional[Sequence[str]] = None) -> Optional[str]:
        stage_order = tuple(stages) if stages is not None else self._selected_stages()
        for stage in stage_order:
            self._require_known_stage(stage)
            self._require_selected_stage(stage)
            if not self._stage_is_complete(stage):
                return stage
        return None

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

    def restore_scf_snapshot(self, name: str) -> dict[str, Any]:
        artifact = self._manifest["artifacts"].get(name)
        if artifact is None:
            raise ValidationError(f"SAPT(DFT) checkpoint SCF snapshot artifact {name} is not available in {self.path}.")
        if artifact.get("kind") != "scf_snapshot":
            raise ValidationError(f"SAPT(DFT) checkpoint artifact {name} is not an SCF snapshot artifact.")
        artifact_path = self._validate_artifact(name, artifact)
        snapshot_data = _load_scf_snapshot_data(artifact_path)
        _prevalidate_scf_snapshot_structure(snapshot_data)
        return snapshot_data

    def _selected_stages(self) -> tuple[str, ...]:
        return selected_stages(self.identity)

    def _require_selected_stage(self, stage: str) -> None:
        if stage not in self._selected_stages():
            raise ValidationError(f"SAPT(DFT) checkpoint stage {stage} is not selected for identity at {self.path}.")

    def _stage_dependencies(self, stage: str) -> tuple[str, ...]:
        self._require_known_stage(stage)
        self._require_selected_stage(stage)
        return selected_stage_dependencies(self.identity, stage)

    def commit_stage(self, stage, *, scalars=None, arrays=None, wavefunctions=None, scf_snapshots=None):
        self._require_known_stage(stage)
        self._require_selected_stage(stage)
        stage_dependencies = self._stage_dependencies(stage)
        missing_dependencies = [dependency for dependency in stage_dependencies if not self._stage_is_complete(dependency)]
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
        for name, snapshot_input in (scf_snapshots or {}).items():
            next_manifest["artifacts"][name] = self._write_scf_snapshot_artifact(name, snapshot_input)
            artifact_names.append(name)

        next_manifest["completed_stages"][stage] = {
            "artifacts": sorted(artifact_names),
            "dependencies": list(stage_dependencies),
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
        for dependency in self._stage_dependencies(stage):
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

        selected = set(self._selected_stages())
        for stage, entry in manifest["completed_stages"].items():
            self._require_known_stage(stage)
            if stage not in selected:
                raise ValidationError(
                    f"SAPT(DFT) checkpoint stage {stage} in {self.manifest_path} is not selected for this identity."
                )
            if not isinstance(entry, Mapping):
                raise ValidationError(f"SAPT(DFT) checkpoint stage entry {stage} in {self.manifest_path} must be a mapping.")
            entry_dependencies = tuple(entry.get("dependencies", []))
            for dependency in entry_dependencies:
                self._require_known_stage(dependency)
            expected_dependencies = selected_stage_dependencies(self.identity, stage)
            if entry_dependencies != expected_dependencies:
                raise ValidationError(
                    f"SAPT(DFT) checkpoint stage {stage} in {self.manifest_path} has dependency metadata {entry_dependencies} but expected {expected_dependencies}."
                )
            if entry.get("version") != SAPTDFT_STAGE_DEFINITION_VERSION:
                raise ValidationError(
                    f"SAPT(DFT) checkpoint stage {stage} in {self.manifest_path} has unsupported definition version {entry.get('version')}."
                )
            missing_dependencies = [dependency for dependency in expected_dependencies if dependency not in manifest["completed_stages"]]
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
        return {"kind": "wavefunction", "path": final_path.name, "sha256": sha256, "size": size}

    def _write_scf_snapshot_artifact(self, name: str, snapshot_input: Any) -> dict[str, Any]:
        if isinstance(snapshot_input, Mapping) and "wavefunction" in snapshot_input:
            wavefunction = snapshot_input.get("wavefunction")
            reference = snapshot_input.get("reference")
            method = snapshot_input.get("method")
            if wavefunction is None or reference is None or method is None:
                raise ValidationError(
                    f"SAPT(DFT) checkpoint SCF snapshot artifact {name} requires wavefunction, reference, and method."
                )
            snapshot_data = capture_scf_snapshot(wavefunction, reference=reference, method=method)
        elif isinstance(snapshot_input, Mapping):
            snapshot_data = dict(snapshot_input)
            metadata = snapshot_data.get(_SCF_SNAPSHOT_METADATA_KEY)
            if not isinstance(metadata, Mapping):
                raise ValidationError(
                    f"SAPT(DFT) checkpoint SCF snapshot artifact {name} must include scf_snapshot metadata."
                )
        else:
            raise ValidationError(
                f"SAPT(DFT) checkpoint SCF snapshot artifact {name} must be a snapshot mapping or a mapping with wavefunction/reference/method."
            )

        _prevalidate_scf_snapshot_structure(snapshot_data)

        suffix = f"{_safe_artifact_stem(name)}--{uuid.uuid4().hex}.npy"
        tmp_path = self.path / f".{suffix}.tmp"
        with tmp_path.open("wb") as handle:
            np.save(handle, snapshot_data, allow_pickle=True)
            handle.flush()
            os.fsync(handle.fileno())
        sha256, size = _file_digest_and_size(tmp_path)
        final_path = self.path / suffix
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


def _prevalidate_scf_snapshot_structure(snapshot_data: Mapping[str, Any]) -> None:
    if not isinstance(snapshot_data, Mapping):
        raise ValidationError("SCF snapshot must be a mapping.")

    for section_name, section_type in _SCF_SNAPSHOT_SERIALIZED_SECTIONS.items():
        section = snapshot_data.get(section_name)
        if not isinstance(section, section_type):
            raise ValidationError(
                f"SCF snapshot top-level section {section_name} is missing or has invalid type {type(section).__name__}."
            )

    for section_name, required_keys in _SCF_SNAPSHOT_REQUIRED_SECTION_KEYS.items():
        section = snapshot_data[section_name]
        for key in required_keys:
            if key not in section:
                raise ValidationError(f"SCF snapshot section {section_name} is missing required key {key}.")


def _deserialize_scf_snapshot(snapshot_data: Mapping[str, Any]) -> core.Wavefunction:
    try:
        loaded = core.Wavefunction.from_file(snapshot_data)
    except Exception as exc:
        raise ValidationError(f"SCF snapshot deserialization failed: {exc}") from exc
    return loaded


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


def _initialize_rehydrated_scf_state(target: core.HF) -> None:
    target.form_H()
    target.form_Shalf()
    target.form_initial_F()
    target.form_initial_C()
    target.form_D()


def _copy_rehydrated_matrix_fields(target: core.HF, source: core.Wavefunction) -> None:
    for field_name in _SCF_SNAPSHOT_REQUIRED_FIELDS["matrix"]:
        target_matrix = getattr(target, field_name)()
        source_matrix = getattr(source, field_name)()
        if target_matrix is None or source_matrix is None:
            raise ValidationError(f"SCF snapshot matrix field {field_name} is not available for rehydration.")
        target_matrix.copy(source_matrix)


def _copy_rehydrated_vector_fields(target: core.HF, source: core.Wavefunction) -> None:
    for field_name in _SCF_SNAPSHOT_REQUIRED_FIELDS["vector"]:
        target_vector = getattr(target, field_name)()
        source_vector = getattr(source, field_name)()
        if target_vector is None or source_vector is None:
            raise ValidationError(f"SCF snapshot vector field {field_name} is not available for rehydration.")
        target_vector.copy(source_vector)


def _copy_rehydrated_qcvariables(target: core.HF, source: core.Wavefunction) -> None:
    for key, value in source.variables().items():
        if target.has_variable(key):
            target.del_variable(key)
        target.set_variable(key, value)


def rehydrate_scf_wavefunction(
    snapshot,
    *,
    method: str,
    reference: str,
    molecule: core.Molecule | None = None,
) -> core.HF:
    snapshot_data = _load_scf_snapshot_data(snapshot)
    _prevalidate_scf_snapshot_structure(snapshot_data)
    loaded = _deserialize_scf_snapshot(snapshot_data)
    metadata = _validate_scf_snapshot_metadata(snapshot_data, loaded, method=method, reference=reference)

    from ..proc import scf_wavefunction_factory

    if molecule is not None:
        supplied_molecule = _normalize_jsonable(molecule.to_dict())
        molecule_difference = _first_difference(_normalize_jsonable(metadata.get("molecule")), supplied_molecule)
        if molecule_difference:
            raise ValidationError(
                f"SCF snapshot supplied molecule does not match snapshot identity: {molecule_difference}"
            )
        fresh_molecule = molecule
    else:
        fresh_molecule = loaded.molecule()

    fresh_base = core.Wavefunction.build(fresh_molecule, loaded.basisset())
    rehydrated = scf_wavefunction_factory(method, fresh_base, _normalize_scf_reference(reference))
    _initialize_rehydrated_scf_state(rehydrated)
    factory_basissets = _capture_factory_basissets(rehydrated)

    if rehydrated.functional().name() != metadata.get("functional"):
        raise ValidationError(
            f"SCF snapshot functional mismatch: expected {metadata.get('functional')!r}, got {rehydrated.functional().name()!r}."
        )

    for key in _SCF_SNAPSHOT_DIMENSION_KEYS:
        expected_dimension = tuple(snapshot_data["dimension"][key])
        if getattr(rehydrated, key)().to_tuple() != expected_dimension:
            raise ValidationError(f"SCF snapshot rehydrated dimensions mismatch for {key}.")

    _copy_rehydrated_matrix_fields(rehydrated, loaded)
    _copy_rehydrated_vector_fields(rehydrated, loaded)
    _copy_rehydrated_qcvariables(rehydrated, loaded)
    rehydrated.set_energy(loaded.energy())

    vpot = rehydrated.V_potential()
    if vpot is not None:
        vpot.set_D([rehydrated.Da()])

    for key, basis in factory_basissets.items():
        rehydrated.set_basisset(key, basis)

    return rehydrated


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
