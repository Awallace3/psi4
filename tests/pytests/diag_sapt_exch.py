"""Diagnostic: dump every SAPT(DFT) exchange intermediate for CPU vs cuEST.

Not a test.  Run as:  python diag_sapt_exch.py cpu   /  python diag_sapt_exch.py gpu
then:                 python diag_sapt_exch.py diff

Monkeypatches sapt_jk_terms_ein.exchange so we capture the cache *and* the
second J/K call's outputs exactly as the exchange formula sees them.  Whatever
diverges first between the two dumps is the bug.
"""

import sys
import numpy as np
import psi4
import psi4.driver.procrouting.sapt.sapt_jk_terms_ein as ein_mod

_water_dimer = """
0 1
O   -0.702196054   -0.056060256   0.009942262
H   -1.022193224    0.846775782   -0.011488714
H    0.257521062    0.042121496    0.005218999
--
0 1
O    2.268880784    0.026340101    0.000508029
H    2.645502399   -0.412039965    0.766632411
H    2.641145101   -0.449872874   -0.744894473
units angstrom
"""

_CACHE_KEYS = ["Cocc_A", "Cocc_B", "Cvir_A", "Cvir_B", "D_A", "D_B", "P_A", "P_B",
               "S", "V_A", "V_B", "J_A", "J_B", "K_A", "K_B", "J_O", "K_O"]


def _np(x):
    return np.asarray(x.np if hasattr(x, "np") else x, dtype=float)


def run(tag, use_cuest):
    dump = {}

    orig = ein_mod.exchange

    def patched(cache, jk, do_print=True):
        for k in _CACHE_KEYS:
            if k in cache:
                dump["cache/" + k] = _np(cache[k])
        res = orig(cache, jk, do_print)
        # jk still holds the second compute()'s J/K when exchange() returns
        for n, name in enumerate(["JT_A", "JT_AB", "Jij"]):
            dump["jk2/" + name] = _np(jk.J()[n])
        for n, name in enumerate(["KT_A", "KT_AB", "Kij"]):
            dump["jk2/" + name] = _np(jk.K()[n])
        dump["res/Exch10"] = np.array([res["Exch10"]])
        dump["res/Exch10(S^2)"] = np.array([res["Exch10(S^2)"]])
        return res

    ein_mod.exchange = patched
    try:
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.set_output_file(f"diag_{tag}.out", False)
        psi4.geometry(_water_dimer)
        psi4.set_options({
            "basis": "cc-pvdz",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "pbe0",
            "SAPT_DFT_GRAC_SHIFT_A": 0.0,
            "SAPT_DFT_GRAC_SHIFT_B": 0.0,
            "SAPT_DFT_INDUCTION_TYPE": "NONE",
            "SAPT_DFT_DO_DHF": True,
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
            "USE_CUEST": use_cuest,
            "CUEST_MIXED_PRECISION": False,
        })
        psi4.energy("sapt(dft)-d4(i)")
    finally:
        ein_mod.exchange = orig
        psi4.core.close_outfile()

    np.savez(f"diag_{tag}.npz", **dump)
    print(f"[{tag}] wrote diag_{tag}.npz with {len(dump)} arrays")
    print(f"[{tag}] Exch10       = {dump['res/Exch10'][0]:.10f}")
    print(f"[{tag}] Exch10(S^2)  = {dump['res/Exch10(S^2)'][0]:.10f}")


def diff():
    a = np.load("diag_cpu.npz")
    b = np.load("diag_gpu.npz")
    keys = sorted(set(a.files) | set(b.files))
    print(f"{'key':22s} {'shape':14s} {'max|d|':>12s} {'rel':>10s}")
    for k in keys:
        if k not in a.files or k not in b.files:
            print(f"{k:22s} MISSING in {'gpu' if k not in b.files else 'cpu'}")
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"{k:22s} SHAPE {x.shape} vs {y.shape}")
            continue
        d = np.max(np.abs(x - y))
        scale = max(np.max(np.abs(x)), 1e-30)
        flag = "   <-- DIFFERS" if d > 1e-8 else ""
        print(f"{k:22s} {str(x.shape):14s} {d:12.3e} {d/scale:10.2e}{flag}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cpu":
        run("cpu", False)
    elif mode == "gpu":
        run("gpu", True)
    elif mode == "diff":
        diff()
    else:
        raise SystemExit("mode must be cpu|gpu|diff")
