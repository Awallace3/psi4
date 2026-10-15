"""Explicit overlap and DF policies for the controlled comparison."""
import contextlib
import hashlib
import inspect
import json
import math
from pathlib import Path
import time


def molecular_geometry(atom, config):
    """Explicit shared Bohr coordinates; do not use engine-dependent constants."""
    factor = config.get("geometry_angstrom_per_bohr")
    if factor is None:
        return atom, "Angstrom"  # Historical replay only.
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Geometry conversion factor must be finite and positive")
    lines = []
    for line in atom.splitlines():
        symbol, *values = line.split()
        coords = [float(value) / factor for value in values]
        if len(coords) != 3 or not all(math.isfinite(x) for x in coords):
            raise ValueError("Invalid geometry coordinates")
        lines.append(symbol + " " + " ".join(format(x, ".17g") for x in coords))
    return "\n".join(lines), "Bohr"


def capture_geometry(coordinates, atom, record):
    """Verify live engine coordinates, not only the hash of an Angstrom file."""
    if record["config"].get("geometry_angstrom_per_bohr") is None:
        return
    import numpy as np
    converted, _ = molecular_geometry(atom, record["config"])
    expected = np.array([[float(x) for x in line.split()[1:]]
                         for line in converted.splitlines()], dtype="<f8")
    actual = np.array(coordinates, dtype="<f8", copy=True)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise RuntimeError("Live Bohr coordinates differ from the shared geometry")
    actual[actual == 0] = 0.0
    record["geometry_bohr_sha256"] = hashlib.sha256(actual.tobytes(order="C")).hexdigest()
    record["geometry_policy"] = dict(input_unit="Bohr",
        angstrom_per_bohr=record["config"]["geometry_angstrom_per_bohr"],
        live_coordinates_verified=True)


def verify_compiled_sources(repo, compiled_commit, harness_commit):
    """Allow harness-only evolution, never silently change compiled Psi4 inputs."""
    import subprocess
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != harness_commit:
        raise RuntimeError("Harness checkout changed")
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code",
        compiled_commit, harness_commit, "--", "psi4/", "cmake/", "external/",
        "CMakeLists.txt", "tests/pytests/test_cuest_grid_export.py"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def overlap_cutoff(config):
    value = config.get("overlap_eigenvalue_cutoff")
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError("Overlap cutoff must be finite and positive")
    return value


def configure_pyscf_overlap(config, record):
    value = overlap_cutoff(config)
    if value is None:
        return
    from gpu4pyscf.scf import hf
    hf.remove_overlap_zero_eigenvalue = True
    hf.overlap_zero_eigenvalue_threshold = value
    record["overlap_policy"] = dict(method="canonical", absolute_eigenvalue_cutoff=value)


def capture_overlap(matrix, nmo, record, diagnostic):
    """Actual retained dimension always; extra eigensolve only in diagnostics."""
    import numpy as np
    cutoff = overlap_cutoff(record["config"])
    nbf = record["nbf"]
    if not 0 < nmo <= nbf:
        raise RuntimeError("Invalid retained orbital dimension")
    record.update(nmo=int(nmo), removed_overlap_directions=int(nbf-nmo))
    if cutoff is None or not diagnostic:
        return
    matrix = np.asarray(matrix)
    if (matrix.shape != (nbf, nbf) or not np.isfinite(matrix).all() or
            not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)):
        raise RuntimeError("Invalid overlap matrix")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if int(np.count_nonzero(eigenvalues > cutoff)) != nmo:
        raise RuntimeError("Retained dimension does not implement the prescribed overlap cutoff")
    info = dict(engine=record["engine"], basis=record["config"]["basis"],
                system=record["system"], functional=record["functional"],
                geometry_sha256=record["geometry_sha256"], config_sha256=record["config_sha256"],
                nbf=nbf, nmo=int(nmo), cutoff=cutoff,
                minimum_eigenvalue=float(eigenvalues[0]),
                cutoff_distance=float(np.min(np.abs(eigenvalues-cutoff))),
                rule="absolute overlap eigenvalue > cutoff; no auxiliary-metric truncation implied")
    if record["config"].get("geometry_angstrom_per_bohr") is not None:
        info["geometry_bohr_sha256"] = record["geometry_bohr_sha256"]
    with Path("overlap-spectrum.npz").open("xb") as stream:
        np.savez(stream, eigenvalues=eigenvalues)
    with Path("overlap-spectrum.json").open("x") as stream:
        json.dump(info, stream, indent=2)
    record["overlap_audit"] = info


def compare_overlap_exports(left, right):
    """Rank and complete eigenvalue spectra, not just declared AO counts."""
    import numpy as np
    values, metadata = [], []
    for path in map(Path, (left, right)):
        info = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as artifact:
            spectrum = artifact["eigenvalues"]
        if (spectrum.shape != (info["nbf"],) or not np.isfinite(spectrum).all() or
                np.any(np.diff(spectrum) < 0) or
                np.count_nonzero(spectrum > info["cutoff"]) != info["nmo"]):
            raise ValueError("Invalid overlap spectrum/rank artifact")
        values.append(spectrum)
        metadata.append(info)
    if {info["engine"] for info in metadata} != {"psi4", "pyscf"}:
        raise ValueError("Overlap audit requires both engines")
    for key in ("basis", "system", "functional", "geometry_sha256", "nbf", "nmo", "cutoff"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f"Overlap comparison mismatch: {key}")
    if any("geometry_bohr_sha256" in info for info in metadata):
        if (not metadata[0].get("geometry_bohr_sha256") or
                metadata[0].get("geometry_bohr_sha256") != metadata[1].get("geometry_bohr_sha256")):
            raise ValueError("Overlap comparison mismatch: live Bohr coordinates")
    if not np.allclose(values[0], values[1], atol=1e-9, rtol=1e-10):
        raise ValueError("Cross-engine overlap eigenvalue spectra differ")
    return dict(verified=True, nbf=metadata[0]["nbf"], nmo=metadata[0]["nmo"],
                cutoff=metadata[0]["cutoff"],
                maximum_spectrum_difference=float(np.max(np.abs(values[0]-values[1]))),
                scope="Equal rank/cutoff and matching full overlap spectra; not an AO-projector or density certification")


def stored_tensor_info(dfobj, cp):
    tensors = getattr(dfobj, "_cderi", None)
    if not isinstance(tensors, (dict, list, tuple)) or not tensors:
        raise RuntimeError("Materialized DF tensor is missing")
    result = []
    # Current GPU4PySCF returns a per-device list from multi_gpu.run;
    # older implementations can expose an indexed mapping.
    entries = tensors.items() if isinstance(tensors, dict) else enumerate(tensors)
    for key, tensor in entries:
        if (not isinstance(tensor, cp.ndarray) or tensor.dtype != cp.float64 or
                tensor.ndim != 2 or tensor.size == 0 or int(tensor.device.id) != 0):
            raise RuntimeError("Stored-DF control requires FP64 tensors on the selected GPU; no host spill")
        result.append(dict(device=int(tensor.device.id), shape=list(tensor.shape), bytes=int(tensor.nbytes)))
    return result


@contextlib.contextmanager
def pyscf_df_policy(mf, config, record, sync):
    """Verify the *consumed* DF route, including pure-functional J contractions."""
    import cupy as cp
    from gpu4pyscf.df import df_jk
    mode = config.get("df_mode", "native-strategy")
    if mode not in ("native-strategy", "stored-df"):
        raise ValueError("Unknown DF strategy")
    direct, stored = df_jk.get_j, df_jk.get_jk
    stored_signature = inspect.signature(stored)
    counts = dict(integral_direct_df_j_calls=0, stored_df_jk_calls=0)
    record["df_route_calls"] = counts
    def get_j(*args, **kwargs):
        counts["integral_direct_df_j_calls"] += 1
        if mode == "stored-df":
            raise RuntimeError("Integral-direct DF-J attempted during stored-DF control")
        return direct(*args, **kwargs)
    def get_jk(dfobj, *args, **kwargs):
        if mode == "stored-df":
            if getattr(dfobj, "_cderi", None) is None:
                # RSH DF objects are created lazily by DF.range_coulomb.
                # Preserve the original contraction's explicit omega, including
                # positional calls; mol.omega alone is not a build argument.
                bound = stored_signature.bind(dfobj, *args, **kwargs)
                omega = bound.arguments.get("omega")
                dfobj.use_gpu_memory = True
                dfobj.build(omega=omega)
                sync()
                record.setdefault("lazy_df_builds", []).append(dict(omega=omega))
            stored_tensor_info(dfobj, cp)
        counts["stored_df_jk_calls"] += 1
        return stored(dfobj, *args, **kwargs)
    df_jk.get_j, df_jk.get_jk = get_j, get_jk
    try:
        if mode == "stored-df":
            mf.with_df.use_gpu_memory = True
            start = time.perf_counter()
            mf.with_df.build()
            sync()
            record["df_prebuild_wall_s"] = time.perf_counter()-start
            record["df_tensor_at_scf_entry"] = stored_tensor_info(mf.with_df, cp)
        yield
        if mode == "stored-df":
            if counts["integral_direct_df_j_calls"]:
                raise RuntimeError("Integral-direct DF-J was attempted during stored-DF control")
            if not counts["stored_df_jk_calls"]:
                raise RuntimeError("No stored-tensor DF contractions were observed")
            objects = [mf.with_df] + list(mf.with_df._rsh_df.values())
            record["df_tensors_at_exit"] = [stored_tensor_info(obj, cp) for obj in objects]
            record["stored_df_verified"] = True
        else:
            record["stored_df_verified"] = False
    finally:
        df_jk.get_j, df_jk.get_jk = direct, stored
