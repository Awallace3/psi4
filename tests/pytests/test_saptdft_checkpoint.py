"""SAPT(DFT) checkpoint and restart.

Split out of test_fsaptdft.py: checkpointing is its own feature with its own
stage model, on-disk format, and failure modes, and none of it depends on the
F-SAPT(DFT) energy regressions that file covers.

Restarts that must prove no SCF was replayed run in a fresh interpreter through
fsaptdft_checkpoint_worker.py -- an in-process test cannot tell a restored
wavefunction apart from one still sitting in Psi4's global state.
"""

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import psi4
import pytest
import qcelemental as qcel
from addons import using
from fsaptdft_checkpoint_worker import run as _run_fsaptdft_checkpoint_worker
from psi4 import compare_values
from psi4 import core

pytestmark = [pytest.mark.psi, pytest.mark.api]

# ---------------------------------------------------------------------------
# SAPT(DFT) checkpoint / restart
#
# Restarts that must prove "no SCF was replayed" run in a fresh interpreter via
# fsaptdft_checkpoint_worker.py -- an in-process test cannot distinguish a
# restored wavefunction from one still sitting in Psi4's global state.
# ---------------------------------------------------------------------------

VE = psi4.driver.p4util.exceptions.ValidationError

_CHECKPOINT_BASE_OPTIONS = {
    "basis": "sto-3g",
    "scf_type": "df",
    "guess": "sad",
    "freeze_core": False,
    "orbital_optimizer_package": "internal",
    "sapt_dft_functional": "svwn",
    "sapt_dft_do_dhf": True,
    "sapt_dft_do_ddft": True,
    "sapt_dft_do_disp": False,
    "sapt_dft_do_fsapt": "none",
    "sapt_dft_do_hybrid": False,
    "sapt_dft_grac_shift_a": 0.0,
    "sapt_dft_grac_shift_b": 0.0,
    "sapt_dft_use_einsums": False,
}

# Option overrides per worker scenario; kept in step with _configure() in
# fsaptdft_checkpoint_worker.py so stage expectations can be derived here.
_CHECKPOINT_SCENARIOS = {
    "default": {},
    "disp": {"sapt_dft_do_ddft": False, "sapt_dft_do_disp": True},
    "lrc": {"sapt_dft_functional": "wb97x"},
    "localization": {
        "sapt_dft_do_dhf": False,
        "sapt_dft_do_ddft": False,
        "sapt_dft_do_fsapt": "FISAPT",
    },
    "fsapt_einsums": {
        "sapt_dft_do_dhf": False,
        "sapt_dft_do_ddft": False,
        "sapt_dft_do_disp": True,
        "sapt_dft_do_fsapt": "SAPTDFT",
        "sapt_dft_functional": "HF",
        "sapt_dft_use_einsums": True,
    },
    "fsapt_fisapt": {
        "sapt_dft_do_dhf": False,
        "sapt_dft_do_ddft": False,
        "sapt_dft_do_disp": True,
        "sapt_dft_do_fsapt": "FISAPT",
        "sapt_dft_functional": "HF",
    },
}

# Banner each SCF stage prints. Stages absent here (delta-DFT, the scalar SAPT
# terms) go through run_scf or a component routine instead.
_STAGE_BANNERS = {
    "hf_dimer_scf": "SAPT(DFT): delta HF Dimer",
    "hf_monomer_a_scf": "SAPT(DFT): delta HF Monomer A",
    "hf_monomer_b_scf": "SAPT(DFT): delta HF Monomer B",
    "dimer_localization_scf": "SAPT(DFT): Dimer for Localization",
    "monomer_a_dft_scf": "SAPT(DFT): DFT Monomer A",
    "monomer_b_dft_scf": "SAPT(DFT): DFT Monomer B",
}

_JK_PAYLOAD_ARTIFACTS = ["exch.J_P_A", "exch.J_P_B"]


def _saptdft_checkpoint_module():
    from psi4.driver.procrouting.sapt import saptdft_checkpoint

    return saptdft_checkpoint


def _saptdft_checkpoint_molecule(distance=3.0):
    return psi4.geometry(
        f"""
0 1
Ne 0.0 0.0 0.0
--
0 1
Ne 0.0 0.0 {distance}
units angstrom
symmetry c1
no_reorient
no_com
"""
    )


def _saptdft_checkpoint_identity_inputs(molecule, function_kwargs=None, method="sapt(dft)"):
    return psi4.driver.p4util.state_to_atomicinput(
        dtype=2,
        driver="energy",
        method=method,
        molecule=molecule,
        function_kwargs=function_kwargs,
    )


def _build_checkpoint_identity(*, name="sapt(dft)", options=None, molecule=None, function_kwargs=None):
    """Configure options and return (molecule, function_kwargs, atomic_input, identity)."""
    core.clean_options()
    psi4.set_options(options if options is _IDENTITY_OPTIONS else {**_CHECKPOINT_BASE_OPTIONS, **(options or {})})
    molecule = molecule if molecule is not None else _saptdft_checkpoint_molecule()
    function_kwargs = (
        function_kwargs
        if function_kwargs is not None
        else {"checkpoint_dir": "identity-dir", "checkpoint_stop_after": "final"}
    )
    atomic_input = _saptdft_checkpoint_identity_inputs(molecule, function_kwargs=function_kwargs, method=name)
    identity = _saptdft_checkpoint_module().build_saptdft_job_identity(
        name=name,
        molecule=molecule,
        function_kwargs=function_kwargs,
        atomic_input=atomic_input,
    )
    return molecule, function_kwargs, atomic_input, identity


# Identity tests deliberately run with einsums requested and F-SAPT on, so the
# execution fingerprint has a non-default backend and the full keyword surface.
_IDENTITY_OPTIONS = {
    "basis": "sto-3g",
    "scf_type": "df",
    "guess": "sad",
    "sapt_dft_functional": "hf",
    "sapt_dft_do_dhf": True,
    "sapt_dft_do_hybrid": False,
    "sapt_dft_do_disp": True,
    "sapt_dft_mp2_disp_alg": "fisapt",
    "sapt_dft_do_fsapt": "fisapt",
    "sapt_dft_use_einsums": True,
    "fisapt_fsapt_filepath": "none",
    "orbital_optimizer_package": "internal",
}


@pytest.fixture
def checkpoint_identity():
    """(module, molecule, function_kwargs, atomic_input, identity) for the default job."""
    molecule, function_kwargs, atomic_input, identity = _build_checkpoint_identity(
        options=_IDENTITY_OPTIONS,
        function_kwargs={
            "checkpoint_dir": "first-dir",
            "checkpoint_stop_after": "elst",
            "output": "first.out",
            "memory": "1 GiB",
            "threads": 1,
            "timer": False,
            "verbosity": 1,
        },
    )
    yield _saptdft_checkpoint_module(), molecule, function_kwargs, atomic_input, identity
    core.clean_options()


def _seed_checkpoint(mod, path, identity, **commit):
    """Open a checkpoint, commit one stage, close, and return the manifest path."""
    checkpoint = mod.SAPTDFTCheckpoint(path, identity)
    checkpoint.open()
    checkpoint.commit_stage(commit.pop("stage", "hf_dimer_scf"), **commit)
    checkpoint.close()
    return Path(path) / mod.SAPTDFT_MANIFEST_FILENAME


# Order in which run_sapt_dft/sapt_dft actually reach the stages. This is not the
# order selected_stages() reports: that one is grouped by dependency, whereas the
# driver computes -D3/-D4 before the scalar SAPT terms and finishes the F-SAPT
# partition before dispersion.
_EXECUTION_ORDER = (
    "grac_monomer_a", "grac_monomer_b",
    "hf_dimer_scf", "hf_monomer_a_scf", "hf_monomer_b_scf",
    "hf_sapt_elst", "hf_sapt_exch", "hf_sapt_ind",
    "dimer_localization_scf",
    "monomer_a_dft_scf", "monomer_b_dft_scf",
    "delta_dft_dimer_scf", "delta_dft_monomer_a_scf", "delta_dft_monomer_b_scf", "delta_dft",
    "d3", "d4",
    "elst", "exch", "ind",
    "fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind",
    "disp", "fsapt_disp", "fsapt_final",
    "final",
)


# Shorthand for the stage groups that travel together, so a selection can be
# written as one readable string instead of a fifteen-element list.
_STAGE_GROUPS = {
    "hf*": ["hf_dimer_scf", "hf_monomer_a_scf", "hf_monomer_b_scf"],
    "hfsapt*": ["hf_sapt_elst", "hf_sapt_exch", "hf_sapt_ind"],
    "loc": ["dimer_localization_scf"],
    "dft*": ["monomer_a_dft_scf", "monomer_b_dft_scf"],
    "ddft*": ["delta_dft_dimer_scf", "delta_dft_monomer_a_scf", "delta_dft_monomer_b_scf", "delta_dft"],
    "scalar": ["elst", "exch", "ind"],
    "fsapt*": ["fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind"],
}


def _expand_stage_shorthand(spec):
    stages = []
    for token in spec.split():
        stages.extend(_STAGE_GROUPS.get(token, [token]))
    return stages


def _selected_stages(*, scenario=None, name="sapt(dft)", options=None):
    """Ordered stages the checkpoint selects for a worker scenario or option set."""
    if scenario is not None:
        options = _CHECKPOINT_SCENARIOS[scenario]
    _, _, _, identity = _build_checkpoint_identity(name=name, options=options)
    return list(_saptdft_checkpoint_module().selected_stages(identity))


def _stages_through(stop_stage, **kwargs):
    """Stages the checkpoint should hold after stopping at ``stop_stage``."""
    selected = set(_selected_stages(**kwargs))
    reached = [stage for stage in _EXECUTION_ORDER if stage in selected]
    return reached[: reached.index(stop_stage) + 1]


def _banners_through(stop_stage, **kwargs):
    """SCF banners that must not reappear on a restart from ``stop_stage``."""
    return [_STAGE_BANNERS[s] for s in _stages_through(stop_stage, **kwargs) if s in _STAGE_BANNERS]



# --- job identity ----------------------------------------------------------


def _raw_qcschema_molecule():
    geometry_angstrom = np.array([[1.25, -0.40, 0.30], [3.10, 1.70, 2.80]])
    return {
        "symbols": ["He", "Ne"],
        "geometry": (geometry_angstrom / qcel.constants.bohr2angstroms).ravel().tolist(),
        "molecular_charge": 0,
        "molecular_multiplicity": 1,
        "fragments": [[0], [1]],
        "fragment_charges": [0, 0],
        "fragment_multiplicities": [1, 1],
    }


def _qcschema_input(molecule, *, function_kwargs=None, method="sapt(dft)", keyword_overrides=None,
                    protocols=None, extras=None, raw_molecule=None):
    keywords = {k.lower(): v for k, v in psi4.driver.p4util.prepare_options_for_set_options().items()}
    basis = keywords.pop("basis", core.get_global_option("BASIS"))
    if function_kwargs is not None:
        keywords["function_kwargs"] = dict(function_kwargs)
    keywords.update(keyword_overrides or {})
    return {
        "schema_name": "qcschema_atomic_input",
        "schema_version": 2,
        "molecule": raw_molecule if raw_molecule is not None else molecule.to_schema(dtype=3),
        "specification": {
            "driver": "energy",
            "model": {"method": method, "basis": basis},
            "keywords": keywords,
            "protocols": dict(protocols or {}),
            "extras": dict(extras or {}),
        },
    }


_DEFAULT_QCSCHEMA_PROTOCOLS = {
    "schema_name": "qcschema_atomic_protocols",
    "error_correction": {"default_policy": True, "policies": None},
    "native_files": "none",
    "stdout": True,
    "wavefunction": "none",
}
_RUNTIME_ONLY_QCSCHEMA_EXTRAS = {"current_qcvars_only": False, "extra_infiles": {}, "wfn_qcvars_only": False}


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    "variant",
    ["raw_molecule", "default_omissions", "default_protocols_and_runtime_extras"],
)
def test_saptdft_checkpoint_identity_matches_equivalent_qcschema(variant):
    """A QCSchema job and the equivalent psi4.energy() job must hash identically."""
    mod = _saptdft_checkpoint_module()
    core.clean_options()
    psi4.set_options({**_CHECKPOINT_BASE_OPTIONS, "sapt_dft_functional": "hf", "sapt_dft_do_ddft": False})
    function_kwargs = {"checkpoint_dir": "first-dir", "output": "first.out"}

    if variant == "raw_molecule":
        molecule = psi4.geometry("0 1\nHe 1.25 -0.40 0.30\n--\n0 1\nNe 3.10 1.70 2.80\nunits angstrom\n")
        molecule.update_geometry()
        kwargs = dict(raw_molecule=_raw_qcschema_molecule())
    elif variant == "default_omissions":
        molecule = _saptdft_checkpoint_molecule()
        kwargs = dict(keyword_overrides={"orbital_optimizer_package": "internal"})
    else:
        molecule = _saptdft_checkpoint_molecule()
        kwargs = dict(protocols=_DEFAULT_QCSCHEMA_PROTOCOLS, extras=_RUNTIME_ONLY_QCSCHEMA_EXTRAS)

    common = dict(name="sapt(dft)", molecule=molecule, function_kwargs=function_kwargs)
    direct = mod.build_saptdft_job_identity(**common)
    from_qcschema = mod.build_saptdft_job_identity(
        **common, atomic_input=_qcschema_input(molecule, function_kwargs=function_kwargs, **kwargs)
    )

    assert direct == from_qcschema
    spec = direct["canonical_input"]["specification"]
    assert spec["protocols"] == {}
    assert spec["extras"] == {}
    if variant == "raw_molecule":
        # The resolved (reoriented) molecule wins over the raw QCSchema geometry.
        assert direct["canonical_input"]["molecule"]["geometry"] != _raw_qcschema_molecule()["geometry"]
    if variant == "default_omissions":
        assert "orbital_optimizer_package" not in spec["keywords"]
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("protocols", "extras", "expected_message"),
    [
        pytest.param({"stdout": False}, None, r"specification\.protocols", id="protocols"),
        pytest.param(None, {"user_tag": "alpha"}, r"specification\.extras", id="extras"),
    ],
)
def test_saptdft_checkpoint_mismatch_reports_qcschema_protocols_and_extras(
    tmp_path, protocols, extras, expected_message
):
    mod = _saptdft_checkpoint_module()
    core.clean_options()
    psi4.set_options({**_CHECKPOINT_BASE_OPTIONS, "sapt_dft_functional": "hf", "sapt_dft_do_ddft": False})
    molecule = _saptdft_checkpoint_molecule()
    function_kwargs = {"checkpoint_dir": "first-dir", "output": "first.out"}
    common = dict(name="sapt(dft)", molecule=molecule, function_kwargs=function_kwargs)

    identity = mod.build_saptdft_job_identity(
        **common,
        atomic_input=_qcschema_input(
            molecule, function_kwargs=function_kwargs, protocols=protocols, extras=extras
        ),
    )
    _seed_checkpoint(mod, tmp_path, identity)

    with pytest.raises(VE, match=expected_message):
        mod.SAPTDFTCheckpoint(tmp_path, mod.build_saptdft_job_identity(**common)).open()
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_identity_is_deterministic(checkpoint_identity):
    mod, molecule, function_kwargs, atomic_input, _ = checkpoint_identity
    from_atomic = mod.build_saptdft_job_identity(
        name="SAPT(DFT)", molecule=molecule, function_kwargs=function_kwargs, atomic_input=atomic_input
    )
    from_state = mod.build_saptdft_job_identity(
        name="sapt(dft)", molecule=molecule, function_kwargs=dict(function_kwargs)
    )
    repeat = mod.build_saptdft_job_identity(
        name="sapt(dft)", molecule=molecule, function_kwargs=dict(function_kwargs), atomic_input=atomic_input
    )

    assert from_atomic == from_state == repeat
    assert len(from_atomic["sha256"]) == 64


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_identity_excludes_runtime_controls(checkpoint_identity):
    """Where the job writes, how long it runs, and how loud it is are not part of its identity."""
    mod, molecule, first_kwargs, _, _ = checkpoint_identity
    second_kwargs = {
        "checkpoint_dir": "second-dir",
        "checkpoint_stop_after": "disp",
        "output": "second.out",
        "memory": "2 GiB",
        "threads": 8,
        "timer": True,
        "verbosity": 5,
    }

    identities = [
        mod.build_saptdft_job_identity(
            name="sapt(dft)",
            molecule=molecule,
            function_kwargs=kwargs,
            atomic_input=_saptdft_checkpoint_identity_inputs(molecule, function_kwargs=kwargs),
        )
        for kwargs in (first_kwargs, second_kwargs)
    ]

    assert identities[0] == identities[1]
    serialized = json.dumps(identities[0]["canonical_input"], sort_keys=True)
    for forbidden in ["checkpoint_dir", "checkpoint_stop_after", "first.out", "second.out",
                      "threads", "verbosity", "timer"]:
        assert forbidden not in serialized


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize("helper_available", [True, False], ids=["helper-present", "helper-missing"])
def test_saptdft_checkpoint_identity_selected_backend(monkeypatch, checkpoint_identity, helper_available):
    """The identity records the backend that will actually run, not the one requested.

    Asking for einsums is not enough: without the SAPT einsums helper module the
    run falls back to numpy, and two checkpoints must not be interchangeable.
    """
    mod, molecule, function_kwargs, _, _ = checkpoint_identity

    if not helper_available:
        original_import_module = mod.importlib.import_module

        def fake_import_module(name):
            if name == "einsums":
                return types.SimpleNamespace(__version__="test-einsums")
            if name == "psi4.driver.procrouting.sapt.sapt_jk_terms_ein":
                raise ImportError("missing SAPT einsums helper")
            return original_import_module(name)

        monkeypatch.setattr(mod.importlib, "import_module", fake_import_module)

    def fingerprint():
        return mod.build_saptdft_job_identity(
            name="sapt(dft)",
            molecule=molecule,
            function_kwargs=function_kwargs,
            atomic_input=_saptdft_checkpoint_identity_inputs(molecule, function_kwargs=function_kwargs),
        )["execution_fingerprint"]

    psi4.set_options({"sapt_dft_use_einsums": False})
    assert fingerprint()["selected_backend"] == "numpy"

    psi4.set_options({"sapt_dft_use_einsums": True})
    selected = fingerprint()
    expected = "einsums" if mod._saptdft_einsums_bundle_available() else "numpy"
    assert expected == ("einsums" if helper_available and mod._saptdft_einsums_bundle_available() else "numpy")
    assert selected["selected_backend"] == expected
    assert ("einsums_version" in selected) is (expected == "einsums")
    # Empirical dispersion versions only matter for -D3/-D4 methods.
    assert "dftd3_version" not in selected
    assert "dftd4_version" not in selected


# --- manifest store, validation, and corruption ----------------------------


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_manifest_schema(tmp_path, checkpoint_identity):
    mod, _, _, _, identity = checkpoint_identity
    array = np.arange(4.0).reshape(2, 2)
    checkpoint = mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    checkpoint.commit_stage("hf_dimer_scf", scalars={"SAPT ELST ENERGY": -0.125}, arrays={"Elst_AB": array})
    np.testing.assert_allclose(checkpoint.restore_array("Elst_AB"), array)
    assert checkpoint.restore_scalars(["SAPT ELST ENERGY"]) == {"SAPT ELST ENERGY": -0.125}
    checkpoint.close()

    manifest = json.loads((tmp_path / "saptdft_state.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["job_identity"]["sha256"] == identity["sha256"]
    assert manifest["completed_stages"]["hf_dimer_scf"]["artifacts"] == ["Elst_AB"]
    assert manifest["completed_stages"]["hf_dimer_scf"]["scalars"] == ["SAPT ELST ENERGY"]
    assert manifest["artifacts"]["Elst_AB"]["kind"] == "array"
    assert manifest["artifacts"]["Elst_AB"]["path"].endswith(".npy")
    assert manifest["artifacts"]["Elst_AB"]["size"] > 0


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    "mutator, expected_message",
    [
        (lambda i: i["canonical_input"]["molecule"].__setitem__("fragments", [[0, 1]]), "fragments"),
        (lambda i: i["canonical_input"]["molecule"].__setitem__("molecular_charge", 1), "molecular_charge"),
        (lambda i: i["canonical_input"]["molecule"].__setitem__("molecular_multiplicity", 3), "molecular_multiplicity"),
        (lambda i: i["canonical_input"]["specification"]["model"].__setitem__("method", "sapt(dft)-d3(s)"), "method"),
        (lambda i: i["canonical_input"]["specification"]["model"].__setitem__("basis", "cc-pvdz"), "basis"),
        (lambda i: i["canonical_input"]["specification"]["keywords"].__setitem__("sapt_dft_functional", "pbe0"),
         "sapt_dft_functional"),
        (lambda i: i["execution_fingerprint"].__setitem__("selected_backend", "numpy"), "selected_backend"),
        (lambda i: i["execution_fingerprint"].__setitem__("psi4_version", "0.0-test"), "psi4_version"),
    ],
)
def test_saptdft_checkpoint_mismatch_rejects_identity_mismatches(
    tmp_path, checkpoint_identity, mutator, expected_message
):
    """A checkpoint from a different calculation must fail loudly, never silently recompute."""
    mod, _, _, _, identity = checkpoint_identity
    _seed_checkpoint(mod, tmp_path, identity, scalars={"SAPT ELST ENERGY": -0.125})

    mismatched = json.loads(json.dumps(identity))
    mutator(mismatched)
    mismatched["sha256"] = "0" * 64

    with pytest.raises(VE, match=expected_message):
        mod.SAPTDFTCheckpoint(tmp_path, mismatched).open()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_rejects_changed_geometry(tmp_path, checkpoint_identity):
    mod, _, function_kwargs, _, identity = checkpoint_identity
    _seed_checkpoint(mod, tmp_path, identity, scalars={"SAPT ELST ENERGY": -0.125})

    other = _saptdft_checkpoint_molecule(distance=3.4)
    other_identity = mod.build_saptdft_job_identity(
        name="sapt(dft)",
        molecule=other,
        function_kwargs=function_kwargs,
        atomic_input=_saptdft_checkpoint_identity_inputs(other, function_kwargs=function_kwargs),
    )

    with pytest.raises(VE, match="geometry"):
        mod.SAPTDFTCheckpoint(tmp_path, other_identity).open()


def _corrupt_payload(mod, tmp_path, manifest_path):
    """Overwrite the artifact bytes without touching the manifest."""
    manifest = json.loads(manifest_path.read_text())
    (tmp_path / manifest["artifacts"]["Elst_AB"]["path"]).write_bytes(b"corrupt")


def _corrupt_truncate(mod, tmp_path, manifest_path):
    """Truncate the artifact and re-checksum it, leaving the recorded size stale."""
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["artifacts"]["Elst_AB"]
    artifact_path = tmp_path / entry["path"]
    original_size = entry["size"]
    artifact_path.write_bytes(artifact_path.read_bytes()[: max(1, original_size // 2)])
    sha256, size = mod._file_digest_and_size(artifact_path)
    assert size != original_size
    entry["sha256"] = sha256
    entry["size"] = original_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _corrupt_schema_version(mod, tmp_path, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _corrupt_path_escape(mod, tmp_path, manifest_path):
    """Repoint an artifact outside the checkpoint directory, checksums intact."""
    outside_path = tmp_path.parent / "outside.npy"
    with outside_path.open("wb") as handle:
        np.save(handle, np.arange(2.0), allow_pickle=False)
    sha256, size = mod._file_digest_and_size(outside_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["Elst_AB"].update({"path": "../outside.npy", "sha256": sha256, "size": size})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    "corrupt, expected_message",
    [
        pytest.param(_corrupt_payload, "checksum", id="checksum"),
        pytest.param(_corrupt_truncate, "size validation", id="truncated"),
        pytest.param(_corrupt_schema_version, "schema version", id="schema-version"),
        pytest.param(_corrupt_path_escape, "checkpoint directory", id="path-escape"),
    ],
)
def test_saptdft_checkpoint_corruption_is_rejected(tmp_path, checkpoint_identity, corrupt, expected_message):
    mod, _, _, _, identity = checkpoint_identity
    manifest_path = _seed_checkpoint(
        mod, tmp_path, identity, arrays={"Elst_AB": np.arange(16.0).reshape(4, 4)}
    )
    corrupt(mod, tmp_path, manifest_path)

    with pytest.raises(VE, match=expected_message):
        mod.SAPTDFTCheckpoint(tmp_path, identity).open()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_rejects_unknown_stage(tmp_path, checkpoint_identity):
    mod, _, _, _, identity = checkpoint_identity
    checkpoint = mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    with pytest.raises(VE, match="Unknown stage"):
        checkpoint.commit_stage("not_a_stage", scalars={"VALUE": 1.0})
    checkpoint.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_artifact_first_interruption(tmp_path, monkeypatch, checkpoint_identity):
    """Artifacts land before the manifest, so a crash mid-commit leaves the stage incomplete."""
    mod, _, _, _, identity = checkpoint_identity
    checkpoint = mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()

    def boom(_manifest):
        raise RuntimeError("manifest boom")

    monkeypatch.setattr(checkpoint, "_write_manifest_atomic", boom)
    with pytest.raises(RuntimeError, match="manifest boom"):
        checkpoint.commit_stage("hf_dimer_scf", arrays={"Elst_AB": np.arange(4.0)})
    checkpoint.close()

    assert not (tmp_path / "saptdft_state.json").exists()
    assert any(path.suffix == ".npy" for path in tmp_path.iterdir())

    reopened = mod.SAPTDFTCheckpoint(tmp_path, identity)
    reopened.open()
    assert not reopened.is_complete("hf_dimer_scf")
    reopened.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_store_wavefunction_artifact_smoke(tmp_path, checkpoint_identity):
    mod, molecule, _, _, identity = checkpoint_identity
    _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
    _seed_checkpoint(mod, tmp_path, identity, wavefunctions={"dimer_wfn": wfn})

    reopened = mod.SAPTDFTCheckpoint(tmp_path, identity)
    reopened.open()
    assert reopened.is_complete("hf_dimer_scf")
    artifact = reopened._manifest["artifacts"]["dimer_wfn"]
    assert artifact["kind"] == "wavefunction"
    restored = core.Wavefunction.from_file(str(reopened._validate_artifact("dimer_wfn", artifact)))
    compare_values(wfn.energy(), restored.energy(), 10, "checkpoint wavefunction energy")
    reopened.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_lock_contention(tmp_path, checkpoint_identity):
    mod, _, _, _, identity = checkpoint_identity
    first = mod.SAPTDFTCheckpoint(tmp_path, identity)
    first.open()
    second = mod.SAPTDFTCheckpoint(tmp_path, identity)
    with pytest.raises(VE, match="lock"):
        second.open()
    first.close()

    second.open()
    second.close()



# --- SCF snapshot capture and rehydration ----------------------------------


def _scf_snapshot_case(method, reference, molecule=None):
    """Converge a small SCF whose state a snapshot must reproduce exactly."""
    core.clean_options()
    psi4.set_options({"basis": "sto-3g", "scf_type": "df", "reference": reference.lower()})
    if molecule is None:
        molecule = psi4.geometry("0 1\nH\nH 1 0.74\nsymmetry c1\nno_reorient\nno_com\n")
    _, wfn = psi4.energy(method, molecule=molecule, return_wfn=True)
    return molecule, wfn


def _guard_scf_rehydrate(monkeypatch):
    """Rehydration must never re-enter an SCF convergence path."""

    def boom(*args, **kwargs):
        raise AssertionError("SCF convergence entry point called during checkpoint rehydration")

    for attr in ["compute_energy", "guess", "diis"]:
        monkeypatch.setattr(core.HF, attr, boom)


def _assert_rehydrated_matches_loaded(restored, loaded, wfn, snapshot, expected_variable):
    assert type(restored) is type(wfn)
    assert restored.functional().name() == wfn.functional().name()
    assert restored.energy() == loaded.energy()
    assert restored.has_variable(expected_variable)

    for name in ["Ca", "Cb", "Da", "Db", "Fa", "Fb", "epsilon_a", "epsilon_b"]:
        np.testing.assert_array_equal(getattr(restored, name)().np, getattr(loaded, name)().np)

    for name in ["doccpi", "frzcpi", "frzvpi", "nalphapi", "nbetapi", "nmopi", "nsopi", "soccpi"]:
        assert getattr(restored, name)().to_tuple() == getattr(loaded, name)().to_tuple()

    for name in snapshot["scf_snapshot"]["required_fields"]["floatvar"]:
        assert restored.variable(name) == loaded.variable(name)

    # A rehydrated wavefunction has to be usable, not just structurally equal:
    # CPHF needs a working JK object and consistent orbital dimensions.
    jk = core.JK.build(restored.basisset(), restored.get_basisset("DF_BASIS_SCF"))
    jk.initialize()
    restored.set_jk(jk)
    trial = core.Matrix("trial", restored.doccpi(), restored.nmopi() - restored.doccpi())
    hx = restored.cphf_Hx([trial])
    assert len(hx) == 1
    np.testing.assert_array_equal(hx[0].np, np.zeros_like(trial.np))


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    "method, reference, expected_variable, via_checkpoint",
    [
        ("hf", "RHF", "HF TOTAL ENERGY", False),
        ("svwn", "RKS", "DFT TOTAL ENERGY", False),
        ("hf", "RHF", "HF TOTAL ENERGY", True),
    ],
    ids=["rhf", "rks", "rhf-through-checkpoint"],
)
def test_saptdft_checkpoint_rehydrate_roundtrip(
    tmp_path, monkeypatch, checkpoint_identity, method, reference, expected_variable, via_checkpoint
):
    """A stored SCF must come back byte-identical, whether read straight off disk or via the store."""
    mod, _, _, _, identity = checkpoint_identity
    _, wfn = _scf_snapshot_case(method, reference)

    if via_checkpoint:
        _seed_checkpoint(
            mod, tmp_path, identity,
            scf_snapshots={"dimer_scf": {"wavefunction": wfn, "reference": reference, "method": method}},
        )
        reopened = mod.SAPTDFTCheckpoint(tmp_path, identity)
        reopened.open()
        artifact = reopened._manifest["artifacts"]["dimer_scf"]
        assert artifact["kind"] == "scf_snapshot"
        source = reopened.restore_scf_snapshot("dimer_scf")
        loaded = core.Wavefunction.from_file(str(reopened._validate_artifact("dimer_scf", artifact)))
    else:
        source = mod.capture_scf_snapshot(wfn, reference=reference, method=method)
        snapshot_path = tmp_path / f"{reference.lower()}_snapshot.npy"
        np.save(snapshot_path, source, allow_pickle=True)
        loaded = core.Wavefunction.from_file(snapshot_path)
        source = snapshot_path

    snapshot = mod.capture_scf_snapshot(wfn, reference=reference, method=method)
    _guard_scf_rehydrate(monkeypatch)
    restored = mod.rehydrate_scf_wavefunction(source, method=method, reference=reference)

    _assert_rehydrated_matches_loaded(restored, loaded, wfn, snapshot, expected_variable)
    if via_checkpoint:
        reopened.close()
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    "mutator, expected_message, prevalidated",
    [
        # Metadata mismatches: caught after the payload deserializes.
        (lambda s: s["scf_snapshot"].__setitem__("version", 999), "version", False),
        (lambda s: s["scf_snapshot"]["molecule"]["geom"].__setitem__(0, 9.99), "molecule", False),
        (lambda s: s["scf_snapshot"]["basis"].__setitem__("name", "cc-pvdz"), "basis", False),
        (lambda s: s["scf_snapshot"].__setitem__("reference", "RKS"), "reference", False),
        (lambda s: s["scf_snapshot"].__setitem__("functional", "bogus"), "functional", False),
        (lambda s: s["scf_snapshot"]["dimensions"].__setitem__("doccpi", [0]), "dimensions", False),
        (lambda s: s["matrix"].__setitem__("Ca", None), "required", False),
        # Structural damage: must be rejected before from_file is even reached.
        (lambda s: s.__delitem__("matrix"), "top-level section matrix", True),
        (lambda s: s.__setitem__("matrix", []), "top-level section matrix", True),
        (lambda s: s["string"].__delitem__("basisname"), "section string", True),
        (lambda s: s["scf_snapshot"].__delitem__("reference"), "section scf_snapshot", True),
    ],
)
def test_saptdft_checkpoint_rehydrate_rejects_bad_snapshot(
    monkeypatch, mutator, expected_message, prevalidated
):
    mod = _saptdft_checkpoint_module()
    _, wfn = _scf_snapshot_case("hf", "RHF")
    snapshot = mod.capture_scf_snapshot(wfn, reference="RHF", method="hf")
    mutator(snapshot)

    if prevalidated:
        def should_not_deserialize(_snapshot):
            raise AssertionError("from_file should not be reached for malformed structure")

        monkeypatch.setattr(core.Wavefunction, "from_file", should_not_deserialize)

    with pytest.raises(VE, match=expected_message):
        mod.rehydrate_scf_wavefunction(snapshot, method="hf", reference="RHF")
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_corruption_rejects_unsupported_snapshot_version_from_checkpoint(
    tmp_path, checkpoint_identity
):
    """An old snapshot format reached through the store is rejected, not silently reused."""
    mod, _, _, _, identity = checkpoint_identity
    _, wfn = _scf_snapshot_case("hf", "RHF")
    _seed_checkpoint(
        mod, tmp_path, identity,
        scf_snapshots={"dimer_scf": {"wavefunction": wfn, "reference": "RHF", "method": "hf"}},
    )

    manifest = _checkpoint_manifest(tmp_path)
    artifact = manifest["artifacts"]["dimer_scf"]
    artifact_path = Path(tmp_path) / artifact["path"]
    payload = np.load(artifact_path, allow_pickle=True).item()
    payload["scf_snapshot"]["version"] = 999
    with artifact_path.open("wb") as handle:
        np.save(handle, payload, allow_pickle=True)
    artifact["sha256"], artifact["size"] = mod._file_digest_and_size(artifact_path)
    (Path(tmp_path) / "saptdft_state.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    reopened = mod.SAPTDFTCheckpoint(tmp_path, identity)
    reopened.open()
    with pytest.raises(VE, match="version"):
        mod.rehydrate_scf_wavefunction(reopened.restore_scf_snapshot("dimer_scf"), method="hf", reference="RHF")
    reopened.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_rehydrate_translates_deserialization_failure(monkeypatch):
    mod = _saptdft_checkpoint_module()
    _, wfn = _scf_snapshot_case("hf", "RHF")
    snapshot = mod.capture_scf_snapshot(wfn, reference="RHF", method="hf")

    def boom(_snapshot):
        raise RuntimeError("bad payload")

    monkeypatch.setattr(core.Wavefunction, "from_file", boom)
    with pytest.raises(VE, match="deserialization failed"):
        mod.rehydrate_scf_wavefunction(snapshot, method="hf", reference="RHF")
    core.clean_options()


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_rehydrate_never_uses_unsafe_direct_wrap(tmp_path):
    """The SCF subclass must be built on a fresh Wavefunction, not wrapped around the loaded one.

    Runs out-of-process because it monkeypatches core.Wavefunction.build globally.
    """
    script = textwrap.dedent(
        f"""
        import numpy as np
        import psi4
        from psi4 import core
        from psi4.driver.procrouting import proc
        from psi4.driver.procrouting.sapt import saptdft_checkpoint as checkpoint_mod

        core.clean_options()
        psi4.set_options({{"basis": "sto-3g", "scf_type": "df", "reference": "rhf"}})
        molecule = psi4.geometry("0 1\\nH\\nH 1 0.74\\nsymmetry c1\\nno_reorient\\nno_com\\n")
        _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
        snapshot = checkpoint_mod.capture_scf_snapshot(wfn, reference="RHF", method="hf")
        snapshot_path = r"{tmp_path / 'unsafe_regression.npy'}"
        np.save(snapshot_path, snapshot, allow_pickle=True)

        original_build = core.Wavefunction.build
        def tagged_build(*args, **kwargs):
            fresh = original_build(*args, **kwargs)
            fresh._rehydrate_fresh = True
            return fresh
        core.Wavefunction.build = tagged_build

        original_factory = proc.scf_wavefunction_factory
        def guarded_factory(name, ref_wfn, reference, **kwargs):
            if not getattr(ref_wfn, "_rehydrate_fresh", False):
                raise RuntimeError("unsafe direct wrap")
            return original_factory(name, ref_wfn, reference, **kwargs)
        proc.scf_wavefunction_factory = guarded_factory

        restored = checkpoint_mod.rehydrate_scf_wavefunction(snapshot_path, method="hf", reference="RHF")
        assert restored.energy() == wfn.energy()
        print("safe")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, env=dict(os.environ)
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "safe" in completed.stdout


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_rehydrate_preserves_prepared_molecule_subset_nre(monkeypatch):
    """Ghost-atom fragmentation survives the round trip; the serialized molecule alone loses it."""
    from psi4.driver.procrouting import proc_util

    mod = _saptdft_checkpoint_module()
    dimer = _saptdft_checkpoint_molecule()
    _, monomerA, _ = proc_util.prepare_sapt_molecule(dimer, "dimer")
    _, wfn = _scf_snapshot_case("svwn", "RKS", molecule=monomerA)
    snapshot = mod.capture_scf_snapshot(wfn, reference="RKS", method="svwn")
    loaded = core.Wavefunction.from_file(snapshot)
    _guard_scf_rehydrate(monkeypatch)

    restored = mod.rehydrate_scf_wavefunction(snapshot, method="svwn", reference="RKS", molecule=monomerA)

    assert loaded.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy() == 0.0
    compare_values(
        wfn.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy(),
        restored.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy(),
        12,
        "prepared molecule subset nuclear repulsion",
    )
    core.clean_options()



# --- stage graph -----------------------------------------------------------


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_stage_dependencies():
    mod = _saptdft_checkpoint_module()
    from psi4.driver.procrouting.sapt import sapt_proc as sapt_proc_mod

    assert not hasattr(sapt_proc_mod, "_SAPTDFT_CHECKPOINT_STAGES")
    for stage in ["elst", "exch", "ind", "disp", "delta_dft", "d3", "d4", "final"]:
        assert stage in mod.SAPTDFT_STAGE_DEFINITIONS
    # The JK object and SAPT cache are rebuilt on every restart, never stored.
    assert "build_jk" not in mod.SAPTDFT_STAGE_DEFINITIONS
    assert "hf_sapt_jk" not in mod.SAPTDFT_STAGE_DEFINITIONS


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("name", "options", "expected"),
    [
        pytest.param("sapt(dft)", None, "hf* hfsapt* dft* ddft* scalar final", id="default"),
        pytest.param("sapt(dft)", {"sapt_dft_do_ddft": False},
                     "hf* hfsapt* dft* scalar final", id="no-ddft"),
        pytest.param("sapt(dft)", {"sapt_dft_do_dhf": False, "sapt_dft_do_ddft": False,
                                   "sapt_dft_do_fsapt": "fisapt"},
                     "loc dft* scalar fsapt* fsapt_final final", id="localization"),
        pytest.param("sapt(dft)", {"sapt_dft_functional": "hf", "sapt_dft_do_ddft": False},
                     "hf* scalar final", id="hf-do-dhf"),
        pytest.param("sapt(dft)", {"sapt_dft_do_ddft": False, "sapt_dft_do_disp": True},
                     "hf* hfsapt* dft* scalar disp final", id="disp"),
        pytest.param("sapt(dft)-d3(s)", {"sapt_dft_functional": "pbe0"},
                     "hf* hfsapt* dft* scalar d3 final", id="d3-method-selected"),
        pytest.param("sapt(dft)-d4(s)", {"sapt_dft_functional": "pbe0"},
                     "hf* hfsapt* dft* scalar d4 final", id="d4-method-selected"),
        pytest.param("sapt(dft)",
                     {"sapt_dft_functional": "hf", "sapt_dft_do_dhf": False, "sapt_dft_do_ddft": False,
                      "sapt_dft_do_disp": True, "sapt_dft_do_fsapt": "fisapt"},
                     "loc dft* scalar disp fsapt* fsapt_disp fsapt_final final",
                     id="fsapt-disp-conditional"),
    ],
)
def test_saptdft_checkpoint_selected_stages(tmp_path, name, options, expected):
    """Which stages exist depends on the requested SAPT(DFT) flavour, and all of them must commit."""
    mod = _saptdft_checkpoint_module()
    _, _, _, identity = _build_checkpoint_identity(name=name, options=options)
    selected = mod.selected_stages(identity)

    assert sorted(selected) == sorted(_expand_stage_shorthand(expected))

    checkpoint_dir = tmp_path / name.replace("(", "_").replace(")", "_").replace("-", "_")
    checkpoint = mod.SAPTDFTCheckpoint(checkpoint_dir, identity)
    checkpoint.open()
    for stage in selected:
        checkpoint.commit_stage(stage)
    checkpoint.close()

    reopened = mod.SAPTDFTCheckpoint(checkpoint_dir, identity)
    reopened.open()
    assert set(reopened._manifest["completed_stages"]) == set(selected)
    reopened.close()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("options", "offpath_stages"),
    [
        pytest.param(None, ["dimer_localization_scf", "d3", "d4"], id="default-rejects-localization-and-d3d4"),
        pytest.param({"sapt_dft_do_ddft": False},
                     ["delta_dft_dimer_scf", "delta_dft_monomer_a_scf", "delta_dft_monomer_b_scf", "delta_dft"],
                     id="no-ddft-rejects-delta"),
        pytest.param(None, ["fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind", "fsapt_disp", "fsapt_final"],
                     id="non-fsapt-rejects-fsapt"),
    ],
)
def test_saptdft_checkpoint_rejects_offpath_stages(tmp_path, options, offpath_stages):
    """A stage this job never runs can be neither committed nor read back from a manifest."""
    mod = _saptdft_checkpoint_module()
    _, _, _, identity = _build_checkpoint_identity(options=options)

    commit_checkpoint = mod.SAPTDFTCheckpoint(tmp_path / "commit", identity)
    commit_checkpoint.open()
    for stage in offpath_stages:
        with pytest.raises(VE, match=stage):
            commit_checkpoint.commit_stage(stage)
    commit_checkpoint.close()

    for index, stage in enumerate(offpath_stages):
        checkpoint_dir = tmp_path / f"manifest-{index}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / mod.SAPTDFT_MANIFEST_FILENAME).write_text(json.dumps({
            "schema_version": mod.SAPTDFT_CHECKPOINT_SCHEMA_VERSION,
            "job_identity": identity,
            "completed_stages": {
                stage: {"artifacts": [], "dependencies": [], "scalars": [],
                        "version": mod.SAPTDFT_STAGE_DEFINITION_VERSION}
            },
            "scalars": {},
            "artifacts": {},
        }))
        with pytest.raises(VE, match=stage):
            mod.SAPTDFTCheckpoint(checkpoint_dir, identity).open()


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize("name, stage", [("sapt(dft)-d3(s)", "d3"), ("sapt(dft)-d4(s)", "d4")])
def test_saptdft_checkpoint_stage_dependencies_selected_method_dispersion_path(tmp_path, name, stage):
    """"final" cannot close until the empirical dispersion stage the method implies is done."""
    mod = _saptdft_checkpoint_module()
    _, _, _, identity = _build_checkpoint_identity(name=name, options={"sapt_dft_functional": "pbe0"})
    assert stage in mod.selected_stages(identity)
    assert mod.selected_stage_dependencies(identity, "final") == ("ind", stage)

    checkpoint = mod.SAPTDFTCheckpoint(tmp_path, identity)
    checkpoint.open()
    for done in _stages_through("ind", name=name, options={"sapt_dft_functional": "pbe0"}):
        if done != stage:  # leave the dispersion stage outstanding
            checkpoint.commit_stage(done)
    with pytest.raises(VE, match=stage):
        checkpoint.commit_stage("final")
    checkpoint.close()



# --- fresh-process restart (worker-driven) ---------------------------------
#
# Every test below runs SAPT(DFT) in a subprocess so a restart cannot be
# satisfied by leftover in-memory Psi4 state, and installs guards that raise if
# any completed stage's SCF or component routine is entered a second time.


def _worker(expect="ok", **kwargs):
    """Run the worker and assert it reached the expected terminal status."""
    proc, payload = _run_fsaptdft_checkpoint_worker(**kwargs)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert payload["status"] == expect, payload.get("traceback") or payload.get("error")
    return payload


def _saptdft_checkpoint_einsums_available():
    return _saptdft_checkpoint_module()._saptdft_einsums_bundle_available()


def _checkpoint_manifest(checkpoint_dir):
    return json.loads((Path(checkpoint_dir) / "saptdft_state.json").read_text())


def _checkpoint_array_payloads(checkpoint_dir, manifest):
    base = Path(checkpoint_dir)
    return {
        name: np.load(base / artifact["path"], allow_pickle=False)
        for name, artifact in manifest["artifacts"].items()
        if artifact.get("kind") == "array"
    }


def _walk_mapping_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_mapping_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_mapping_keys(item)


def _walk_mapping_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_mapping_values(item)
    elif isinstance(value, (list, tuple)) and not isinstance(value, np.ndarray):
        for item in value:
            yield from _walk_mapping_values(item)
    else:
        yield value


def _assert_checkpoint_has_no_jk(checkpoint_dir):
    """JK objects can be enormous; nothing resembling one may reach disk."""
    manifest = _checkpoint_manifest(checkpoint_dir)
    base = Path(checkpoint_dir)
    assert manifest["artifacts"]
    assert all("jk" not in str(name).lower() for name in manifest["artifacts"])

    for name, artifact in manifest["artifacts"].items():
        if artifact.get("kind") not in {"scf_snapshot", "wavefunction"}:
            continue
        payload = np.load(base / artifact["path"], allow_pickle=True).item()
        assert all("jk" not in key.lower() for key in _walk_mapping_keys(payload))
        if artifact["kind"] == "wavefunction":
            # A serialized wavefunction is plain data: named sections of arrays and
            # scalars, with no live Psi4 objects hiding in it.
            assert set(payload) == {"molecule", "matrix", "vector", "dimension", "int",
                                    "string", "boolean", "float", "floatvar", "matrixarr"}
            for value in _walk_mapping_values(payload):
                assert isinstance(value, (np.ndarray, np.generic, str, bool, int, float, type(None)))
    return manifest


_FSAPT_BASE_ARTIFACT_KINDS = {
    "dimer_localization_scf": "scf_snapshot",
    "monomer_a_dft_scf": "scf_snapshot",
    "monomer_b_dft_scf": "scf_snapshot",
    **{name: "array" for name in _JK_PAYLOAD_ARTIFACTS},
}
_FSAPT_STAGE_ARTIFACTS = {
    "fsapt_setup": [f"fsapt_setup.{key}" for key in (
        "Caocc0A", "Caocc0B", "Laocc0A", "Laocc0B", "Lfocc0A", "Lfocc0B", "Locc_A", "Locc_B",
        "Qocc0A", "Qocc0B", "Uaocc0A", "Uaocc0B", "Uocc_A", "Uocc_B",
        "ZA", "ZA_orig", "ZB", "ZB_orig", "ZC", "ZC_orig")],
    "fsapt_elst": ["fsapt_elst.Elst_AB", "fsapt_elst.Vlocc0A", "fsapt_elst.Vlocc0B"],
    "fsapt_exch": ["fsapt_exch.Exch_AB"],
    "fsapt_ind": ["fsapt_ind.IndAB_AB", "fsapt_ind.IndBA_AB"],
    "fsapt_disp": ["fsapt_disp.Disp_AB"],
}


def _assert_fsapt_artifacts_exact(checkpoint_dir, completed_fsapt_stages):
    """The manifest must hold exactly the payloads the completed F-SAPT stages own."""
    expected = dict(_FSAPT_BASE_ARTIFACT_KINDS)
    for stage in completed_fsapt_stages:
        expected.update({name: "array" for name in _FSAPT_STAGE_ARTIFACTS[stage]})
    manifest = _assert_checkpoint_has_no_jk(checkpoint_dir)
    assert {name: artifact["kind"] for name, artifact in manifest["artifacts"].items()} == expected
    return manifest


def _assert_stop_result(stopped, *, stop_stage, expected_stages):
    mod = _saptdft_checkpoint_module()
    assert stopped["completed_stages"] == sorted(expected_stages)
    assert stopped["manifest"]["completed_stages"][stop_stage]["dependencies"] == list(
        mod.selected_stage_dependencies(stopped["manifest"]["job_identity"], stop_stage)
    )


def _assert_energies_match(reference, restarted, label, keys=("sapt_total_energy",)):
    for key in keys:
        compare_values(reference[key], restarted[key], 8, f"{label} {key}")


def _assert_scalar_qcvars_match(reference, restarted, label):
    assert set(reference["qcvars"]) == set(restarted["qcvars"])
    for key, ref_value in reference["qcvars"].items():
        if isinstance(ref_value, bool):
            assert restarted["qcvars"][key] is ref_value, f"{label} qcvar {key}"
        else:
            compare_values(ref_value, restarted["qcvars"][key], 8, f"{label} qcvar {key}")


def _assert_fsapt_variables_match(reference, restarted):
    assert set(reference["fsapt_variables"]) == set(restarted["fsapt_variables"])
    for label, ref_value in reference["fsapt_variables"].items():
        np.testing.assert_allclose(
            np.asarray(restarted["fsapt_variables"][label]), np.asarray(ref_value),
            atol=1.0e-10, rtol=1.0e-10, err_msg=label,
        )


def _assert_array_artifacts_match(reference_dir, comparison_dir, artifact_names):
    reference_arrays = _checkpoint_array_payloads(reference_dir, _checkpoint_manifest(reference_dir))
    comparison_arrays = _checkpoint_array_payloads(comparison_dir, _checkpoint_manifest(comparison_dir))
    assert set(artifact_names) <= set(reference_arrays)
    assert set(artifact_names) <= set(comparison_arrays)
    for name in artifact_names:
        np.testing.assert_allclose(comparison_arrays[name], reference_arrays[name],
                                   atol=1.0e-10, rtol=1.0e-10, err_msg=name)


@pytest.fixture(scope="module")
def checkpoint_reference(tmp_path_factory):
    """Uninterrupted worker runs per scenario, computed once and reused as the baseline."""
    cache = {}

    def get(scenario="default", **kwargs):
        key = (scenario, tuple(sorted(kwargs.items())))
        if key not in cache:
            checkpoint_dir = tmp_path_factory.mktemp(f"saptdft-{scenario}-reference")
            payload = _worker(checkpoint_dir=checkpoint_dir, mode="reference", scenario=scenario, **kwargs)
            cache[key] = (checkpoint_dir, payload)
        return cache[key]

    return get



@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize("stop_stage", [
    "hf_dimer_scf", "hf_monomer_a_scf", "hf_monomer_b_scf", "monomer_a_dft_scf",
    "monomer_b_dft_scf", "delta_dft_dimer_scf", "delta_dft_monomer_a_scf", "delta_dft_monomer_b_scf",
])
def test_saptdft_checkpoint_restart_skips_scf(tmp_path, checkpoint_reference, stop_stage):
    """Restarting after any SCF stage must reuse it, not converge it again."""
    _, reference = checkpoint_reference()

    checkpoint_dir = tmp_path / stop_stage
    stopped = _worker(expect="stopped", checkpoint_dir=checkpoint_dir, mode="stop", stop_after=stop_stage)
    _assert_stop_result(stopped, stop_stage=stop_stage, expected_stages=_stages_through(stop_stage, scenario="default"))

    restarted = _worker(
        checkpoint_dir=checkpoint_dir,
        mode="restart_with_guards",
        forbid_banners=_banners_through(stop_stage, scenario="default"),
    )
    _assert_energies_match(reference, restarted, f"restart {stop_stage}", ("elst10_r", "sapt_total_energy"))


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_restart_skips_localization_scf(tmp_path):
    """The dimer-for-localization SCF is a restartable stage in its own right."""
    reference = _worker(
        expect="stopped", checkpoint_dir=tmp_path / "localization-reference", mode="stop",
        stop_after="monomer_a_dft_scf", scenario="localization",
    )

    checkpoint_dir = tmp_path / "dimer_localization_scf"
    stopped = _worker(
        expect="stopped", checkpoint_dir=checkpoint_dir, mode="stop",
        stop_after="dimer_localization_scf", scenario="localization",
    )
    _assert_stop_result(stopped, stop_stage="dimer_localization_scf", expected_stages=["dimer_localization_scf"])

    restarted = _worker(
        expect="stopped", checkpoint_dir=checkpoint_dir, mode="restart_with_guards", scenario="localization",
        stop_after="monomer_a_dft_scf", forbid_banners=["SAPT(DFT): Dimer for Localization"],
    )
    _assert_stop_result(
        restarted, stop_stage="monomer_a_dft_scf",
        expected_stages=["dimer_localization_scf", "monomer_a_dft_scf"],
    )
    compare_values(reference["current_energy"], restarted["current_energy"], 8,
                   "checkpoint restart localization monomer energy")


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("scenario", "stop_stage", "array_artifacts"),
    [
        ("default", "hf_sapt_elst", []),
        ("default", "hf_sapt_exch", []),
        ("default", "hf_sapt_ind", []),
        ("default", "delta_dft", []),
        ("default", "elst", []),
        ("default", "exch", _JK_PAYLOAD_ARTIFACTS),
        ("default", "ind", _JK_PAYLOAD_ARTIFACTS),
        ("disp", "disp", _JK_PAYLOAD_ARTIFACTS),
    ],
)
def test_saptdft_checkpoint_persistent_stage_restart_matches_reference(
    tmp_path, checkpoint_reference, scenario, stop_stage, array_artifacts
):
    """Every persistent stage boundary must resume to bit-comparable results."""
    reference_dir, reference = checkpoint_reference(scenario)

    checkpoint_dir = tmp_path / f"{scenario}-{stop_stage}"
    stopped = _worker(expect="stopped", checkpoint_dir=checkpoint_dir, mode="stop",
                      stop_after=stop_stage, scenario=scenario)
    _assert_stop_result(stopped, stop_stage=stop_stage,
                        expected_stages=_stages_through(stop_stage, scenario=scenario))

    restarted = _worker(checkpoint_dir=checkpoint_dir, mode="restart_with_guards", scenario=scenario)
    assert restarted["guarded_call_sentinel"] is None
    assert restarted["completed_stages"] == sorted(_selected_stages(scenario=scenario))
    _assert_scalar_qcvars_match(reference, restarted, f"{scenario} {stop_stage}")
    if array_artifacts:
        _assert_array_artifacts_match(reference_dir, checkpoint_dir, array_artifacts)


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_lrc_restart_rebuilds_functional_jk(tmp_path, checkpoint_reference):
    """A restored range-separated monomer needs its omega-aware JK rebuilt, exactly once."""
    reference_dir, reference = checkpoint_reference("lrc", count_jk_builds=True, capture_jk_settings=True)
    checkpoint_dir = tmp_path / "lrc-monomer-b"

    stopped = _worker(expect="stopped", checkpoint_dir=checkpoint_dir, mode="stop",
                      stop_after="monomer_b_dft_scf", scenario="lrc")
    _assert_stop_result(stopped, stop_stage="monomer_b_dft_scf",
                        expected_stages=_stages_through("monomer_b_dft_scf", scenario="lrc"))

    restarted = _worker(
        checkpoint_dir=checkpoint_dir, mode="restart_with_guards", scenario="lrc",
        count_jk_builds=True, capture_jk_settings=True,
        forbid_banners=_banners_through("monomer_b_dft_scf", scenario="lrc"),
    )
    assert restarted["jk_build_count"] == 1
    built = [state for state in restarted["jk_builds"] if state.get("built")]
    assert len(built) == 1
    jk_state = built[0]
    assert jk_state["context"] == "restore:wb97x"
    assert jk_state["orbital_basis"] == "STO-3G"
    assert jk_state["aux_basis"] not in (None, "")
    assert jk_state["do_K"] is True
    assert jk_state["do_wK"] is True
    assert jk_state["initialized"] is True
    assert jk_state["memory"] > 0
    compare_values(0.3, jk_state["omega"], 8, "LRC restored JK omega")
    compare_values(0.157706, jk_state["omega_alpha"], 6, "LRC restored JK omega alpha")
    compare_values(0.842294, jk_state["omega_beta"], 6, "LRC restored JK omega beta")
    _assert_energies_match(reference, restarted, "lrc restart", ("current_energy", "sapt_total_energy"))
    _assert_checkpoint_has_no_jk(checkpoint_dir)
    _assert_array_artifacts_match(reference_dir, checkpoint_dir, _JK_PAYLOAD_ARTIFACTS)


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize("scenario", ["default", "lrc"])
def test_saptdft_checkpoint_final_restart_returns_before_scf_and_jk(tmp_path, checkpoint_reference, scenario):
    """A completed job replays from the manifest: no SCF, and JK.build must never be called."""
    reference_kwargs = {"count_jk_builds": True, "capture_jk_settings": True} if scenario == "lrc" else {}
    _, reference = checkpoint_reference(scenario, **reference_kwargs)

    checkpoint_dir = tmp_path / f"{scenario}-final"
    _worker(expect="stopped", checkpoint_dir=checkpoint_dir, mode="stop", stop_after="final", scenario=scenario)
    manifest = _assert_checkpoint_has_no_jk(checkpoint_dir)
    assert manifest["artifacts"]["dimer_wfn"]["kind"] == "wavefunction"

    restarted = _worker(checkpoint_dir=checkpoint_dir, mode="restart_with_guards",
                        scenario=scenario, guard_jk=True)
    _assert_energies_match(reference, restarted, f"{scenario} final restart",
                           ("current_energy", "sapt_total_energy"))


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("name", "stage"),
    [
        pytest.param("sapt(dft)-d3(s)", "d3", marks=[*using("s-dftd3")], id="d3"),
        pytest.param("sapt(dft)-d4(s)", "d4", marks=[*using("dftd4")], id="d4"),
    ],
)
def test_saptdft_checkpoint_empirical_dispersion_restart(tmp_path, name, stage):
    """Empirical -D3/-D4 interaction energies are their own restartable stage."""
    reference = _worker(checkpoint_dir="", mode="reference", name=name)

    checkpoint_dir = tmp_path / stage
    stopped = _worker(expect="stopped", checkpoint_dir=checkpoint_dir, mode="stop",
                      stop_after=stage, name=name)
    _assert_stop_result(stopped, stop_stage=stage,
                        expected_stages=_stages_through(stage, name=name, options={"sapt_dft_functional": "pbe0"}))

    restarted = _worker(
        checkpoint_dir=checkpoint_dir, mode="restart_with_guards", name=name,
        forbid_banners=_banners_through(stage, name=name, options={"sapt_dft_functional": "pbe0"}),
    )
    _assert_energies_match(reference, restarted, f"checkpoint restart {stage}")


_FSAPT_RESTART_BANNERS = [
    "SAPT(DFT): Dimer for Localization",
    "SAPT(DFT): DFT Monomer A",
    "SAPT(DFT): DFT Monomer B",
]


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.parametrize(
    ("scenario", "stop_stage", "completed_fsapt_stages", "expect_jk_build"),
    [
        ("fsapt_einsums", "fsapt_exch", ["fsapt_setup", "fsapt_elst", "fsapt_exch"], True),
        ("fsapt_einsums", "fsapt_ind", ["fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind"], True),
        ("fsapt_fisapt", "fsapt_ind", ["fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind"], True),
        ("fsapt_einsums", "fsapt_final",
         ["fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind", "fsapt_disp"], False),
        ("fsapt_fisapt", "fsapt_final",
         ["fsapt_setup", "fsapt_elst", "fsapt_exch", "fsapt_ind", "fsapt_disp"], False),
    ],
)
def test_saptdft_checkpoint_fsapt_restart_reuses_artifacts(
    tmp_path, scenario, stop_stage, completed_fsapt_stages, expect_jk_build
):
    """F-SAPT partition matrices are restored from disk; their routines must not run again.

    Restarting at ``fsapt_final`` has no work left that needs a JK object, so
    JK.build is guarded outright rather than counted.
    """
    if scenario == "fsapt_einsums" and not _saptdft_checkpoint_einsums_available():
        pytest.skip("einsums bundle unavailable")

    _, reference = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir="", mode="reference", scenario=scenario, capture_fsapt=True
    )
    assert reference["status"] == "ok"

    checkpoint_dir = tmp_path / f"{scenario}-{stop_stage}"
    stopped = _worker(expect="stopped", checkpoint_dir=checkpoint_dir, mode="stop",
                      stop_after=stop_stage, scenario=scenario)
    _assert_stop_result(stopped, stop_stage=stop_stage,
                        expected_stages=_stages_through(stop_stage, scenario=scenario))
    _assert_fsapt_artifacts_exact(checkpoint_dir, completed_fsapt_stages)

    restarted = _worker(
        checkpoint_dir=checkpoint_dir, mode="restart_with_guards", scenario=scenario,
        capture_fsapt=True, guard_jk=not expect_jk_build, count_jk_builds=expect_jk_build,
        forbid_banners=_FSAPT_RESTART_BANNERS,
        forbid_fsapt_stages=[stage.removeprefix("fsapt_") for stage in completed_fsapt_stages],
    )
    if expect_jk_build:
        assert restarted["jk_build_count"] == 1
    label = f"{scenario} {stop_stage} restart"
    _assert_energies_match(reference, restarted, label,
                           ("current_energy", "sapt_total_energy", "saptdft_total_energy"))
    _assert_fsapt_variables_match(reference, restarted)


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_corruption_fresh_process_never_recomputes(tmp_path):
    """A damaged checkpoint aborts the job; it is never treated as a cache miss."""
    checkpoint_dir = tmp_path / "corrupt-guarded-restart"
    _worker(expect="stopped", checkpoint_dir=checkpoint_dir, mode="stop", stop_after="hf_dimer_scf")

    artifact_path = checkpoint_dir / _checkpoint_manifest(checkpoint_dir)["artifacts"]["hf_dimer_scf"]["path"]
    artifact_path.write_bytes(artifact_path.read_bytes()[:128])

    restarted_proc, restarted = _run_fsaptdft_checkpoint_worker(
        checkpoint_dir=checkpoint_dir, mode="restart_with_guards", guard_jk=True,
        count_jk_builds=True, forbid_banners=["SAPT(DFT): delta HF Dimer"],
    )
    assert restarted_proc.returncode != 0
    assert restarted["status"] == "error"
    assert restarted["error_type"] == "ValidationError"
    assert "checksum" in restarted["error"] or "size validation" in restarted["error"]
    for counter in ["jk_build_count", "scf_helper_call_count", "run_scf_call_count", "guarded_call_count"]:
        assert restarted[counter] == 0
    assert restarted.get("guarded_call_sentinel") is None


@pytest.mark.saptdft
@pytest.mark.fsapt
def test_saptdft_checkpoint_unexpected_exception_closes_lock_same_process(tmp_path, monkeypatch):
    """A crash mid-run must still release the checkpoint lock for the next process."""
    from psi4.driver.procrouting import proc
    from psi4.driver.procrouting.sapt import sapt_proc as sapt_proc_mod

    molecule = _saptdft_checkpoint_molecule()
    core.clean_options()
    psi4.set_options(_CHECKPOINT_BASE_OPTIONS)

    def boom(*args, **kwargs):
        raise RuntimeError("checkpoint crash probe")

    with monkeypatch.context() as ctx:
        ctx.setattr(proc, "scf_helper", boom)
        ctx.setattr(sapt_proc_mod, "scf_helper", boom)
        with pytest.raises(RuntimeError, match="checkpoint crash probe"):
            psi4.energy("sapt(dft)", molecule=molecule, checkpoint_dir=str(tmp_path))

    _worker(expect="stopped", checkpoint_dir=tmp_path, mode="stop", stop_after="hf_dimer_scf")



# --- in-process stop/restart via the environment-variable controls ---------
#
# The worker suite above covers the stage matrix with stronger guarantees (fresh
# process, guards proving nothing is recomputed). What is only reachable here is
# the PSI4_CHECKPOINT_DIR / PSI4_CHECKPOINT_STOP_AFTER entry point, and the GRAC
# stages, which no worker scenario enables.

_INPROCESS_OPTIONS = {
    "basis": "sto-3g",
    "scf_type": "df",
    "reference": "rhf",
    "scf__reference": "rhf",
    "SAPT_DFT_FUNCTIONAL": "HF",
    "SAPT_DFT_DO_DHF": True,
    "SAPT_DFT_DO_DDFT": False,
    "SAPT_DFT_DO_DISP": True,
    "SAPT_DFT_MP2_DISP_ALG": "FISAPT",
    "SAPT_DFT_DO_FSAPT": "NONE",
    "SAPT_DFT_DO_HYBRID": False,
    "SAPT_DFT_USE_EINSUMS": True,
    "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
}
_GRAC_OPTIONS = {"SAPT_DFT_FUNCTIONAL": "PBE0", "SAPT_DFT_DO_DHF": False,
                 "SAPT_DFT_GRAC_COMPUTE": "SINGLE", "SAPT_DFT_DO_DISP": False}


@pytest.mark.saptdft
@pytest.mark.fsapt
@pytest.mark.quick
@pytest.mark.parametrize(
    ("stop_stage", "option_updates"),
    [
        pytest.param("hf_dimer_scf", {}, id="scf"),
        pytest.param("disp", {}, id="disp"),
        pytest.param("final", {}, id="final"),
        pytest.param("grac_monomer_a", _GRAC_OPTIONS, id="grac-a"),
        pytest.param("grac_monomer_b", _GRAC_OPTIONS, id="grac-b"),
    ],
)
def test_saptdft_checkpoint_env_var_stop_and_restart(tmp_path, monkeypatch, stop_stage, option_updates):
    """PSI4_CHECKPOINT_DIR/_STOP_AFTER must drive a restart that lands on the same numbers."""
    compare_vars = ["SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY",
                    "SAPT DISP ENERGY", "SAPT TOTAL ENERGY"]

    def run():
        psi4.core.clean()
        psi4.core.clean_variables()
        psi4.core.clean_timers()
        # SAPT(DFT) sets local SCF options as a side effect; without resetting them the
        # next in-process run would present a different job identity than the checkpoint
        # it is restarting from.
        psi4.core.clean_options()
        psi4.set_options({**_INPROCESS_OPTIONS, **option_updates})
        psi4.energy("sapt(dft)", molecule=_saptdft_checkpoint_molecule(distance=3.2))

    monkeypatch.delenv("PSI4_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("PSI4_CHECKPOINT_STOP_AFTER", raising=False)
    run()
    reference = {var: psi4.core.variable(var) for var in compare_vars if psi4.core.has_variable(var)}
    assert reference, "reference run published no SAPT components"

    checkpoint_dir = tmp_path / stop_stage
    monkeypatch.setenv("PSI4_CHECKPOINT_DIR", str(checkpoint_dir))
    monkeypatch.setenv("PSI4_CHECKPOINT_STOP_AFTER", stop_stage)
    with pytest.raises(RuntimeError, match=f"SAPT\\(DFT\\) checkpoint stop after {stop_stage}"):
        run()
    assert (checkpoint_dir / "saptdft_state.json").exists()

    monkeypatch.delenv("PSI4_CHECKPOINT_STOP_AFTER", raising=False)
    run()
    for var, expected in reference.items():
        assert compare_values(expected, psi4.core.variable(var), 8, var)
    psi4.core.clean_options()


