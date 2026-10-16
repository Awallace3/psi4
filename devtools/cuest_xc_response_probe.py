#!/usr/bin/env python
"""GPU evidence and timings for the cuEST-grid XC response kernel.

Run inside a GPU allocation, outside the source tree, with the staged Psi4 first on
PYTHONPATH:

    python cuest_xc_response_probe.py OUT_DIR [contract] [timing] [sapt]

contract  SDK normalization/signed-density contract on the cuEST grid, central
          differences of the grid's own compute_V, the CUDA-vs-host LibXC derivative
          routes, and the nonfinite-reference behavior. Writes contract.json.
timing    Cold setup, repeated actions and complete CPKS solves, cuEST-grid XC vs
          host-grid XC (both with cuEST J/K). Writes timing.json.
sapt      SAPT(DFT)-D4(I) with CPKS induction, CUEST_XC true vs false. Writes sapt.json.
quick     Restrict timing and sapt to the smallest system.

Numbers only; tests/pytests/test_cuest_cphf.py holds the pass/fail gates.
PROBE_CPU_SMOKE=1 runs the same logic with every potential on the host grid (no GPU),
to check the script itself.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psi4

if os.environ.get("EXPECTED_PSI4_PACKAGE"):
    found = Path(psi4.__file__).resolve().parent
    assert found == Path(os.environ["EXPECTED_PSI4_PACKAGE"]).resolve(), found
OUT = Path(sys.argv[1]).resolve()
SECTIONS = set(sys.argv[2:]) - {"quick"} or {"contract", "timing", "sapt"}
QUICK = "quick" in sys.argv[2:]
OUT.mkdir(parents=True, exist_ok=True)
psi4.core.set_output_file(str(OUT / "probe.out"), False)
psi4.set_memory("24 GB")
GPU = not os.environ.get("PROBE_CPU_SMOKE")


def M(a):
    return psi4.core.Matrix.from_array(np.ascontiguousarray(a))


def maxabs(a, b):
    return float(max(np.abs(x - y).max() for x, y in zip(a, b)))


def scale(a):
    return float(max(np.abs(x).max() for x in a))


WATER = """
{charge} {mult}
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
"""

BENZENE_DIMER = """
0 1
C   0.000000  1.396792  0.000000
C   1.209657  0.698396  0.000000
C   1.209657 -0.698396  0.000000
C   0.000000 -1.396792  0.000000
C  -1.209657 -0.698396  0.000000
C  -1.209657  0.698396  0.000000
H   0.000000  2.484212  0.000000
H   2.151390  1.242106  0.000000
H   2.151390 -1.242106  0.000000
H   0.000000 -2.484212  0.000000
H  -2.151390 -1.242106  0.000000
H  -2.151390  1.242106  0.000000
--
C   0.000000  1.396792  3.700000
C   1.209657  0.698396  3.700000
C   1.209657 -0.698396  3.700000
C   0.000000 -1.396792  3.700000
C  -1.209657 -0.698396  3.700000
C  -1.209657  0.698396  3.700000
H   0.000000  2.484212  3.700000
H   2.151390  1.242106  3.700000
H   2.151390 -1.242106  3.700000
H   0.000000 -2.484212  3.700000
H  -2.151390 -1.242106  3.700000
H  -2.151390  1.242106  3.700000
symmetry c1
no_reorient
no_com
"""

WATER_DIMER = """
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


def base_options(**extra):
    psi4.core.clean_options()
    options = {
        "basis": "cc-pvdz", "df_basis_scf": "cc-pvdz-jkfit", "scf_type": "df",
        "save_jk": True, "e_convergence": 1.e-10, "d_convergence": 1.e-9,
        "cuest_mixed_precision": False, "dft_radial_points": 75, "dft_spherical_points": 302,
    }
    options.update(extra)
    psi4.set_options(options)


def scf(geometry, method, unrestricted, gpu_xc, **extra):
    psi4.core.clean()
    base_options(reference="uhf" if unrestricted else "rhf", use_cuest=GPU, cuest_xc=gpu_xc and GPU, **extra)
    mol = psi4.geometry(geometry)
    t0 = time.perf_counter()
    wfn = psi4.energy(method, molecule=mol, return_wfn=True)[1]
    return wfn, time.perf_counter() - t0


def potential(wfn, gpu, name=None, polarized=False):
    gpu = gpu and GPU
    psi4.set_options({"use_cuest": GPU, "cuest_xc": gpu})
    functional = wfn.functional()
    restricted = wfn.same_a_b_orbs() and not polarized
    if name is not None:
        functional = psi4.driver.dft.build_superfunctional(name, restricted)[0]
    V = psi4.core.VBase.build(wfn.basisset(), functional, "RV" if restricted else "UV")
    V.initialize()
    assert V.uses_cuest_xc() == gpu
    return V


def set_reference(V, Cs):
    V.set_D([M(c @ c.T) for c in Cs])
    V.set_Cocc([M(c) for c in Cs])


def potential_at(V, Cs):
    set_reference(V, Cs)
    out = [psi4.core.Matrix(c.shape[0], c.shape[0]) for c in Cs]
    V.compute_V(out)
    return [o.np.copy() for o in out]


def apply(V, kind, Dx):
    n = Dx[0].shape[0]
    ret = [psi4.core.Matrix(n, n) for _ in Dx]
    if kind == "triplet":
        V.compute_Vx_full([M(d) for d in Dx], ret, False)
    else:
        V.compute_Vx([M(d) for d in Dx], ret)
    return [r.np.copy() for r in ret]


def apply_factored(V, kind, left, right):
    n = left[0].shape[0]
    ret = [psi4.core.Matrix(n, n) for _ in left]
    L, R = [M(x) for x in left], [M(x) for x in right]
    if kind == "triplet":
        V.compute_Vx_full_factored(L, R, ret, False)
    else:
        V.compute_Vx_factored(L, R, ret)
    return [r.np.copy() for r in ret]


def orbitals(wfn, unrestricted, space):
    Cs = [wfn.Ca_subset("AO", space).np.copy()]
    if unrestricted:
        Cs.append(wfn.Cb_subset("AO", space).np.copy())
    return Cs


def timer_totals(prefixes):
    out = {}
    for key, value in psi4.core.get_timer_dict().items():
        label = key.split(";")[-1]
        if any(label.startswith(p) for p in prefixes):
            entry = out.setdefault(label, {"wall_time": 0.0, "n_calls": 0})
            entry["wall_time"] += value["wall_time"]
            entry["n_calls"] += value["n_calls"]
    return out


def libxc_route_from_scf():
    timers = psi4.core.get_timer_dict()
    return {"cuda_libxc_scf_timer": any("CUDA LibXC Functional" in k for k in timers),
            "host_functional_scf_timer": any("cuEST XC: Host Functional" in k for k in timers)}


# => Contract <= #


def contract():
    results = {"cases": []}
    for method in ("svwn", "blyp", "pbe", "pbe0", "b3lyp", "lrc-wpbe"):
        for kind in ("singlet", "triplet", "uks"):
            unrestricted = kind == "uks"
            psi4.core.clean_timers()
            wfn, _ = scf(WATER.format(charge=1 if unrestricted else 0, mult=2 if unrestricted else 1),
                         method, unrestricted, gpu_xc=True)
            route = libxc_route_from_scf()
            case = {"method": method, "kind": kind, **route}
            V = potential(wfn, True)
            case["V_on_cuest_grid"] = V.uses_cuest_xc()
            Cs, Cv = orbitals(wfn, unrestricted, "OCC"), orbitals(wfn, unrestricted, "VIR")
            set_reference(V, Cs)
            rng = np.random.default_rng(23)
            xs = [rng.normal(size=(o.shape[1], v.shape[1])) * 0.1 for o, v in zip(Cs, Cv)]
            L = Cs
            R = [v @ x.T for v, x in zip(Cv, xs)]
            Dx = [l @ r.T for l, r in zip(L, R)]

            for host_libxc in (False, True):
                psi4.set_options({"cuest_xc_response_host_libxc": host_libxc})
                tag = "host_libxc" if host_libxc else "default_libxc"
                set_reference(V, Cs)
                ref = apply(V, kind, Dx)
                case[f"{tag}_scale"] = scale(ref)
                case[f"{tag}_factored_vs_dense"] = maxabs(ref, apply_factored(V, kind, L, R))
                case[f"{tag}_transpose"] = maxabs(ref, apply_factored(V, kind, R, L))
                case[f"{tag}_sign"] = maxabs([-r for r in ref], apply_factored(V, kind, L, [-r for r in R]))
                case[f"{tag}_antisymmetric"] = scale(apply(V, kind, [d - d.T for d in Dx]))
                # L = R: the nonsymmetric SDK density with equal factors.
                sym = apply_factored(V, kind, Cs, Cs)
                case[f"{tag}_L_eq_R_vs_dense"] = maxabs(sym, apply(V, kind, [c @ c.T for c in Cs]))
                # Reference from the ordinary SDK density (occupied factors) vs the
                # nonsymmetric identity-tile density (dense D, factors made stale).
                V.set_Cocc([M(c * 1.01) for c in Cs])
                case[f"{tag}_occupied_vs_dense_reference"] = maxabs(ref, apply(V, kind, Dx))
                set_reference(V, Cs)
                case[f"{tag}_result"] = ref
            psi4.set_options({"cuest_xc_response_host_libxc": False})
            case["cuda_vs_host_libxc"] = maxabs(case.pop("default_libxc_result"), case.pop("host_libxc_result"))

            # Central differences of this grid's compute_V (normalization, weights once).
            Us = [rng.normal(size=c.shape) * 0.1 for c in Cs]
            eps = 1.e-4
            if kind == "triplet":
                oracle = potential(wfn, True, name=method, polarized=True)
                plus = potential_at(oracle, [Cs[0] + eps * Us[0], Cs[0] - eps * Us[0]])[0]
                minus = potential_at(oracle, [Cs[0] - eps * Us[0], Cs[0] + eps * Us[0]])[0]
                expected = [0.25 * (plus - minus) / eps]
                oracle.finalize()
            else:
                plus = potential_at(V, [c + eps * u for c, u in zip(Cs, Us)])
                minus = potential_at(V, [c - eps * u for c, u in zip(Cs, Us)])
                factor = 0.5 if kind == "singlet" else 1.0
                expected = [factor * (p - m) / (2 * eps) for p, m in zip(plus, minus)]
            set_reference(V, Cs)
            got = apply(V, kind, [u @ c.T + c @ u.T for c, u in zip(Cs, Us)])
            case["finite_difference"] = maxabs(expected, got)
            case["finite_difference_scale"] = scale(expected)

            # Host grid (native kernel) for the same orbitals: quadrature-level only.
            host = potential(wfn, False)
            set_reference(host, Cs)
            case["host_grid_vs_cuest_grid"] = maxabs(apply(host, kind, Dx), apply(V, kind, Dx))
            host.finalize()

            # A nonphysical (negative) reference density: error or finite output?
            V.set_D([M(-c @ c.T) for c in Cs])
            try:
                out = apply(V, kind, Dx)
                case["negative_reference"] = "finite" if all(np.isfinite(o).all() for o in out) else "nonfinite"
            except RuntimeError as err:
                case["negative_reference"] = "raised: " + str(err).strip().splitlines()[-1][:200]
            V.finalize()
            print(json.dumps(case), flush=True)
            results["cases"].append(case)
    (OUT / "contract.json").write_text(json.dumps(results, indent=2) + "\n")


# => Timing <= #


def rotations(wfn, unrestricted, count, seed=29):
    rng = np.random.default_rng(seed)
    ret = []
    for _ in range(count):
        for spin in (("a", "b") if unrestricted else ("a",)):
            subset = getattr(wfn, f"C{spin}_subset")
            occ, vir = subset("SO", "OCC"), subset("SO", "VIR")
            ret.append(M(rng.normal(size=(occ.coldim()[0], vir.coldim()[0])) * 0.01))
    return ret


def dipole_rhs(wfn, unrestricted):
    mints = psi4.core.MintsHelper(wfn.basisset())
    rhs = []
    for mu in mints.so_dipole():
        for spin in (("a", "b") if unrestricted else ("a",)):
            subset = getattr(wfn, f"C{spin}_subset")
            rhs.append(psi4.core.triplet(subset("SO", "OCC"), mu, subset("SO", "VIR"), True, False, False))
    return rhs


def time_actions(wfn, unrestricted, nrhs, repeats):
    label = ("UV" if unrestricted else "RV") + ": Form Vx"
    rot = rotations(wfn, unrestricted, nrhs)
    V = wfn.V_potential()
    gpu = V.uses_cuest_xc()
    record = {"nrhs": nrhs}
    if gpu and nrhs == 1:
        # Cold setup: the first factored response call of a freshly initialized potential
        # (device workspace allocation and SDK workspace queries), then the same call warm.
        fresh = potential(wfn, True)
        Cs = orbitals(wfn, unrestricted, "OCC")
        Cv = orbitals(wfn, unrestricted, "VIR")
        set_reference(fresh, Cs)
        right = [v @ x.np.T for v, x in zip(Cv, rot)]
        for key in ("cold_first_factored_call_s", "warm_factored_call_s"):
            t0 = time.perf_counter()
            apply_factored(fresh, "uks" if unrestricted else "singlet", Cs, right)
            record[key] = time.perf_counter() - t0
        fresh.finalize()
    psi4.core.clean_timers()
    wfn.cphf_Hx(rot)  # warm
    psi4.core.clean_timers()
    t0 = time.perf_counter()
    for _ in range(repeats):
        wfn.cphf_Hx(rot)
    record["cphf_Hx_mean_s"] = (time.perf_counter() - t0) / repeats
    stages = timer_totals((label, "cuEST Vx:"))
    record["stages_per_action_s"] = {k: v["wall_time"] / repeats for k, v in stages.items()}
    record["vx_calls_per_action"] = {k: v["n_calls"] / repeats for k, v in stages.items()
                                     if k.startswith(label)}
    return record


def time_solve(wfn, unrestricted):
    label = ("UV" if unrestricted else "RV") + ": Form Vx"
    rhs = dipole_rhs(wfn, unrestricted)
    psi4.core.clean_timers()
    t0 = time.perf_counter()
    solved = wfn.cphf_solve(rhs, 1.e-8, 100, 0)
    elapsed = time.perf_counter() - t0
    residual = max(float(np.abs(ax.np - b.np).max()) for b, ax in zip(rhs, wfn.cphf_Hx(solved)))
    nspin = 2 if unrestricted else 1
    factor = 2.0 if unrestricted else 4.0
    alpha = [factor * sum(rhs[nspin * i + s].vector_dot(solved[nspin * i + s]) for s in range(nspin))
             for i in range(3)]
    stages = timer_totals((label, "cuEST Vx:"))
    return {"solve_s": elapsed, "converged": bool(wfn.cphf_converged()), "max_residual": residual,
            "alpha_diag": alpha, "vx_timers": stages}


def timing():
    systems = [("water", WATER.format(charge=0, mult=1), "cc-pvdz"),
               ("water", WATER.format(charge=0, mult=1), "aug-cc-pvtz"),
               ("benzene_dimer", BENZENE_DIMER, "cc-pvdz"),
               ("benzene_dimer", BENZENE_DIMER, "aug-cc-pvdz")]
    if QUICK:
        systems = systems[:1]
    results = []
    for name, geometry, basis in systems:
        for method in ("pbe", "pbe0"):
            for unrestricted in (False,) if name != "water" else (False, True):
                mol = WATER.format(charge=1, mult=2) if unrestricted else geometry
                for gpu_xc in (True, False):
                    psi4.core.clean_timers()
                    wfn, scf_s = scf(mol, method, unrestricted, gpu_xc, basis=basis, df_basis_scf=basis + "-jkfit")
                    entry = {"system": name, "basis": basis, "nbf": wfn.basisset().nbf(), "method": method,
                             "reference": "UKS" if unrestricted else "RKS",
                             "xc_grid": "cuest" if gpu_xc else "host", "scf_s": scf_s,
                             **libxc_route_from_scf()}
                    entry["actions"] = [time_actions(wfn, unrestricted, n, 5) for n in (1, 8)]
                    entry["solve"] = time_solve(wfn, unrestricted)
                    print(json.dumps(entry), flush=True)
                    results.append(entry)
                    (OUT / "timing.json").write_text(json.dumps(results, indent=2) + "\n")


# => SAPT(DFT) CPKS induction <= #


def sapt():
    import psi4.driver.procrouting.sapt.sapt_jk_terms_ein as terms

    results = []
    original = terms._sapt_cpscf_solve

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timed.elapsed += time.perf_counter() - t0

    terms._sapt_cpscf_solve = timed
    try:
        systems = (("water_dimer", WATER_DIMER, "cc-pvdz"),
                   ("water_dimer", WATER_DIMER, "aug-cc-pvdz"),
                   ("benzene_dimer", BENZENE_DIMER, "aug-cc-pvdz"))
        for name, geometry, basis in systems[:1] if QUICK else systems:
            for cuest_xc in (False, True):
                psi4.core.clean()
                psi4.core.clean_options()
                psi4.core.clean_timers()
                psi4.geometry(geometry)
                psi4.set_options({
                    "basis": basis, "scf_type": "df", "SAPT_DFT_FUNCTIONAL": "pbe0",
                    "SAPT_DFT_GRAC_SHIFT_A": 0.136, "SAPT_DFT_GRAC_SHIFT_B": 0.136,
                    "SAPT_DFT_INDUCTION_TYPE": "CPKS", "SAPT_DFT_DO_DHF": True,
                    "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL", "USE_CUEST": GPU, "CUEST_XC": cuest_xc and GPU,
                    "CUEST_MIXED_PRECISION": False,
                })
                timed.elapsed = 0.0
                t0 = time.perf_counter()
                psi4.energy("sapt(dft)-d4(i)")
                total = time.perf_counter() - t0
                entry = {"system": name, "basis": basis, "cuest_xc": cuest_xc, "total_s": total,
                         "cpks_solve_s": timed.elapsed,
                         "vx_timers": timer_totals(("RV: Form Vx", "cuEST Vx:")),
                         "values": {k: psi4.variable(k) for k in (
                             "SAPT ELST ENERGY", "SAPT EXCH ENERGY",
                             "SAPT IND ENERGY", "SAPT TOTAL ENERGY")}}
                print(json.dumps(entry), flush=True)
                results.append(entry)
                (OUT / "sapt.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        terms._sapt_cpscf_solve = original


if __name__ == "__main__":
    if "contract" in SECTIONS:
        contract()
    if "timing" in SECTIONS:
        timing()
    if "sapt" in SECTIONS:
        sapt()
    psi4.core.close_outfile()
