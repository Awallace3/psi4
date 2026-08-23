"""Collect every number state-of-camcasp-psi4.html reports. Dev-only."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

REPO = Path("/home/awallace43/gits/camcasp_psi4")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("wpa", REPO / "water-pes-aim.py")
wpa = importlib.util.module_from_spec(spec)
sys.modules["wpa"] = wpa          # @dataclass resolves via sys.modules[cls.__module__]
spec.loader.exec_module(wpa)
from devtools import camcasp_reference as cr

OUT = {}
ATOMS = ("O", "H1", "H2")
TYPES = {"O": "O", "H1": "H", "H2": "H"}
BLOCKS = {1: range(0, 3), 2: range(3, 8), 3: range(8, 15)}

# ---------------------------------------------------------------- PES curves
arms = wpa.load_arms()
distances = [2.5, 2.6, 2.7, 2.8, 2.912, 3.0, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0,
             4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
OUT["pes"] = {"distances": distances, "modes": {}}
for mode in ("r4r2", "c8c6"):
    per_arm = {}
    for key, arm in arms.items():
        rows = []
        for d, donor, acceptor in wpa.dimer_scan_geometries(np.array(distances)):
            r = wpa.evaluate(arm, donor, acceptor, mode)
            rows.append({"ind": r["induction"], "ind_iso": r["induction_isotropic"],
                         "c6": r["dispersion"].c6, "c8": r["dispersion"].c8,
                         "c10": r["dispersion"].c10,
                         "d3": r["dispersion"].d3_comparable})
        per_arm[key] = rows
    OUT["pes"]["modes"][mode] = per_arm
OUT["pes"]["labels"] = {k: v["label"] for k, v in arms.items()}
print("pes done")

# ------------------------------------------------- rank-resolved polarizability
ours_l3 = np.asarray(json.load(open("ours_full.json"))["ATOMIC ANISOTROPIC POLARIZABILITIES"]).reshape(3, 15, 15)
L2G = {"O": ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),
       "H1": ((-1.,0.,0.),(0.,-1.,0.),(0.,0.,1.)),
       "H2": ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))}
blocks = cr.parse_refined_polarizabilities(
    REPO / ".camcasp-reference/work/H2O-isagrid/H2O_ref_wt4_L3_0f10.pol", list(ATOMS), limit=3)
ref_l3 = np.stack([np.asarray(cr.l3_local_to_molecular(blocks[0].atoms[a], L2G[a])) for a in ATOMS])

rank = {}
for l, idx in BLOCKS.items():
    i = list(idx)
    o = sum(np.trace(ours_l3[s][np.ix_(i, i)]) / (2 * l + 1) for s in range(3))
    f = sum(np.trace(ref_l3[s][np.ix_(i, i)]) / (2 * l + 1) for s in range(3))
    per_site = {ATOMS[s]: {"ours": float(np.trace(ours_l3[s][np.ix_(i,i)])/(2*l+1)),
                           "ref": float(np.trace(ref_l3[s][np.ix_(i,i)])/(2*l+1))}
                for s in range(3)}
    rank[l] = {"ours": float(o), "ref": float(f), "ratio": float(o/f), "sites": per_site}
OUT["rank_invariants"] = rank
print("rank invariants done")

# ------------------------------------------------------- static alpha components
COMP = ("xx", "xy", "xz", "yy", "yz", "zz")
ours_a = np.asarray(json.load(open("ours_full.json"))["ATOMIC POLARIZABILITIES"])
OUT["static_alpha"] = {
    "components": COMP, "atoms": ATOMS,
    "ours": ours_a.tolist(),
    "isa_grid": wpa.CAMCASP["isa_grid"]["alpha"],
    "df": wpa.CAMCASP["df"]["alpha"],
}
print("static alpha done")

# ------------------------------------------------------------- isotropic Cn
full = json.load(open("ours_full.json"))
OUT["isotropic_cn"] = {"atoms": ATOMS, "orders": {}}
for order in ("C6", "C8", "C10", "C12"):
    OUT["isotropic_cn"]["orders"][order] = {
        "ours": np.asarray(full[f"ATOMIC {order}"]).tolist(),
        "isa_grid": wpa.CAMCASP["isa_grid"][order],
        "df": wpa.CAMCASP["df"][order],
    }
print("isotropic Cn done")

# ------------------------------------------------------- frequency grid + dynamic
OUT["frequencies"] = np.asarray(full["ATOMIC POLARIZABILITY FREQUENCIES"]).reshape(-1).tolist()
dyn = np.asarray(full["ATOMIC DYNAMIC POLARIZABILITIES"]).reshape(11, 3, 6)
OUT["dynamic_iso"] = {
    "ours": [[float((d[s][0]+d[s][3]+d[s][5])/3.0) for s in range(3)] for d in dyn],
}
ref_dyn = []
for b in blocks:
    row = []
    for a in ATOMS:
        m = np.asarray(cr.l3_local_to_molecular(b.atoms[a], L2G[a]))
        row.append(float(np.trace(m[np.ix_([0,1,2],[0,1,2])]) / 3.0))
    ref_dyn.append(row)
OUT["dynamic_iso"]["isa_grid"] = ref_dyn
print("dynamic done")

# ------------------------------------------------- anisotropic Cn vs CASIMIR
#
# Frame care. Our published table is in the MOLECULAR frame; CASIMIR's blocks are in each
# site's LOCAL frame. From H2O.axes the local frames are the identity for O and H2 and a
# 180 deg z-rotation for H1, so only site pairs drawn from {O, H2} are directly comparable
# with no rotation at all. Rows are `first * site_count + second` (verified in the C++).
# Both choices are computed so the effect of the choice is measured, not assumed.
labels = np.asarray(full["ATOMIC DISPERSION LABELS"], dtype=int)
coeffs = np.asarray(full["ATOMIC DISPERSION COEFFICIENTS"])

# The parser keys labels as (l1, k1, l2, k2, j) with k as an INTEGER index in the same
# l0, l1c, l1s, l2c, l2s, ... convention our ATOMIC DISPERSION LABELS uses, so the two are
# directly comparable with no name conversion.
ours_map = {}
for col in range(labels.shape[0]):
    n, l1, k1, l2, k2, j = (int(x) for x in labels[col])
    ours_map[(n, l1, k1, l2, k2, j)] = coeffs[:, col]

casimir = cr.parse_anisotropic_cn(
    REPO / ".camcasp-reference/work/H2O-isagrid/H2O_ref_wt4_L3_casimir.out", list(ATOMS), TYPES)

SITE_OF = {"identity": {"O": 0, "H": 2}, "h1": {"O": 0, "H": 1}}
results = {}
for choice, mapping in SITE_OF.items():
    comp = []
    for table in casimir.values():
        a, b = table.first_type, table.second_type
        flat = mapping[a] * 3 + mapping[b]
        for label in table.labels:
            l1, k1, l2, k2, j = label
            for n in range(6, 13):
                ref = table.coefficient(label, n)
                if ref == 0.0: continue
                key = (n, l1, k1, l2, k2, j)
                if key not in ours_map: continue
                comp.append({"pair": f"{a}{b}", "n": n, "l1": l1, "k1": k1, "l2": l2,
                             "k2": k2, "j": j, "ours": float(ours_map[key][flat]),
                             "ref": float(ref)})
    results[choice] = comp
    sign = sum(1 for c in comp if c["ours"] * c["ref"] < 0)
    print(f"  site choice {choice:9}: {len(comp)} comparisons, {sign} sign disagreements "
          f"({100*sign/len(comp):.1f}%)")
OUT["aniso_cn"] = results["identity"]
OUT["aniso_cn_h1"] = results["h1"]

# Does the exact CG factor 1/|<l1 0; l2 0|j 0>| reconcile the cosine sector at C6?
from math import sqrt
def cg000(l1, l2, j):
    """<l1 0; l2 0 | j 0> via the Racah formula, integer arguments only."""
    from math import factorial as f
    if (l1 + l2 + j) % 2: return 0.0
    if not (abs(l1 - l2) <= j <= l1 + l2): return 0.0
    g = (l1 + l2 + j) // 2
    pref = sqrt((2 * j + 1) * f(l1 + l2 - j) * f(l1 - l2 + j) * f(-l1 + l2 + j)
                / f(l1 + l2 + j + 1))
    return ((-1) ** (g - j) * f(g) / (f(g - l1) * f(g - l2) * f(g - j))) * pref

c6 = [c for c in OUT["aniso_cn"] if c["n"] == 6]
scan = {}
for c in c6:
    key = (c["pair"], c["l1"], c["k1"], c["l2"], c["k2"])
    cg = cg000(c["l1"], c["l2"], c["j"])
    if cg == 0.0: continue
    scan.setdefault(key, []).append((c["j"], (c["ours"] / c["ref"]) * abs(cg)))
spread = []
for key, vals in scan.items():
    if len(vals) < 2: continue
    v = [x for _, x in vals]
    spread.append((max(v) - min(v)) / abs(np.mean(v)))
OUT["cg_scan"] = {"groups": len(spread), "worst_relative_spread": float(max(spread)) if spread else None}
print(f"  CG j-scan: {len(spread)} multi-j groups at C6, worst relative spread "
      f"{max(spread):.3e}" if spread else "  CG j-scan: no multi-j groups")

json.dump(OUT, open("report-data.json", "w"))
print("wrote report-data.json")
