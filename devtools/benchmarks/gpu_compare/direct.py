"""Conventional four-center JK policy, distinct from integral-direct DF-J."""
import contextlib
import hashlib
import inspect
from pathlib import Path
from unittest.mock import patch


def jk_mode(config):
    mode = config.get("jk_mode", "density-fitted")
    if mode not in ("density-fitted", "conventional-direct"):
        raise ValueError(f"Unknown JK mode: {mode}")
    if mode == "conventional-direct" and config.get("df_mode") is not None:
        raise ValueError("A conventional-direct cohort cannot select a DF strategy")
    return mode


def is_direct(config):
    return jk_mode(config) == "conventional-direct"


def make_rks(rks, mol, xc, config):
    mf = rks.RKS(mol, xc=xc)
    if not is_direct(config):
        mf = mf.density_fit(auxbasis=config["auxbasis"], only_dfj=False)
    return mf


@contextlib.contextmanager
def audit_direct_contractions(mf, record, cp, j_engine, jk, forbidden, kernels=()):
    """Audit the consumed four-center functions; restore every hook on failure."""
    if getattr(mf, "with_df", None) is not None or getattr(mf, "_eri", None) is not None:
        raise RuntimeError("Direct comparison forbids DF or materialized ERIs")
    counts = dict(j_attempted=0, j_completed=0, k_attempted=0, k_completed=0,
                  rsh_k_completed=0, forbidden_attempts=0,
                  gpu_j_kernel_calls=0, gpu_k_kernel_calls=0)
    record["direct_route_calls"] = counts
    record["direct_requests"] = requests = []
    record["direct_verified"] = False
    with contextlib.ExitStack() as stack:
        for library, name, kind in kernels:
            original = getattr(library, name)
            def launched(*args, _fn=original, _kind=kind, **kwargs):
                status = _fn(*args, **kwargs)
                if status == 0:
                    counts[f"gpu_{_kind}_kernel_calls"] += 1
                return status
            stack.enter_context(patch.object(library, name, launched))
        for module, name in forbidden:
            if not hasattr(module, name):
                raise RuntimeError(f"Missing required fallback guard: {name}")
            def reject(*args, _name=name, **kwargs):
                counts["forbidden_attempts"] += 1
                raise RuntimeError(f"Forbidden CPU/DF contraction in direct cohort: {_name}")
            stack.enter_context(patch.object(module, name, reject))
        for module, name, kind in ((j_engine, "get_j", "j"), (jk, "get_k", "k")):
            original = getattr(module, name)
            signature = inspect.signature(original)
            def audited(*args, _fn=original, _sig=signature, _kind=kind, **kwargs):
                bound = _sig.bind(*args, **kwargs)
                bound.apply_defaults()
                if not isinstance(bound.arguments["dm"], cp.ndarray):
                    raise RuntimeError("Four-center contraction must consume a GPU density")
                counts[f"{_kind}_attempted"] += 1
                omega = bound.arguments.get("omega")
                if omega is None:
                    omega = bound.arguments["mol"].omega
                event = dict(kind=_kind, omega=float(omega),
                             lr_factor=bound.arguments.get("lr_factor"),
                             sr_factor=bound.arguments.get("sr_factor"))
                requests.append(event)
                result = _fn(*args, **kwargs)
                if not isinstance(result, cp.ndarray):
                    raise RuntimeError("Four-center contraction did not return a GPU matrix")
                counts[f"{_kind}_completed"] += 1
                if _kind == "k" and abs(event["omega"]) > 0:
                    counts["rsh_k_completed"] += 1
                return result
            stack.enter_context(patch.object(module, name, audited))
        yield
        if getattr(mf, "with_df", None) is not None or getattr(mf, "_eri", None) is not None:
            raise RuntimeError("Direct worker switched to DF or stored ERIs")
        meta = record["functional_metadata"]
        need_k = bool(meta["short_range_exchange"] or meta["long_range_exchange"])
        if (counts["forbidden_attempts"] or not counts["j_completed"] or
                (need_k and not counts["k_completed"]) or
                (need_k and meta["omega"] and not counts["rsh_k_completed"]) or
                (kernels and (not counts["gpu_j_kernel_calls"] or
                              (need_k and not counts["gpu_k_kernel_calls"])))):
            raise RuntimeError("Required conventional GPU direct contractions were not verified")
        record["direct_verified"] = True


@contextlib.contextmanager
def pyscf_direct_policy(mf, record):
    import cupy as cp
    from gpu4pyscf.dft import rks
    from gpu4pyscf.scf import hf, j_engine, jk
    from gpu4pyscf.df import df, df_jk
    from pyscf.scf import hf as cpu_hf, jk as cpu_jk, _vhf
    if type(mf) is not rks.RKS:
        raise RuntimeError("Conventional direct requires undecorated GPU4PySCF RKS")
    for method in ("get_j", "get_k"):
        if getattr(mf, method).__func__ is not getattr(hf.SCF, method):
            raise RuntimeError(f"Unexpected GPU4PySCF direct dispatch: {method}")
    # These are the consumed entry points in the pinned GPU4PySCF installation.
    paths = {Path(inspect.getsourcefile(module)).resolve() for module in (hf, j_engine, jk)}
    record["direct_source_files"] = {}
    for path in sorted(paths):
        with path.open("rb") as stream:
            record["direct_source_files"][str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    record["jk_backend"] = "gpu4pyscf.scf.j_engine.get_j + gpu4pyscf.scf.jk.get_k"
    forbidden = [(df.DF, "build"), (df_jk, "get_j"), (df_jk, "get_jk"),
                 (cpu_hf, "get_jk"), (cpu_jk, "get_jk"),
                 (_vhf, "direct"), (_vhf, "direct_mapdm"), (_vhf, "direct_bindm")]
    kernels = [(j_engine.libvhf_md, "MD_build_j", "j"),
               (jk.libvhf_rys, "RYS_build_k", "k")]
    with audit_direct_contractions(mf, record, cp, j_engine, jk, forbidden, kernels):
        yield
