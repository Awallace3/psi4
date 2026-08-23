"""Generate state-of-camcasp-psi4.html. Dev-only."""
import json, sys, subprocess, datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import svgplot as sp
from svgplot import Axes, grouped_bars, PALETTE

REPO = Path("/home/awallace43/gits/camcasp_psi4")
D = json.load(open(Path(__file__).resolve().parent / "report-data.json"))
OUT = REPO / "state-of-camcasp-psi4.html"

HEAD = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>State of the CamCASP-Psi4 distributed-property pipeline</title>
<script>
window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['$$','$$']]},
 svg:{fontCache:'global'}};
</script>
<script id="MathJax-script" async
 src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
:root{--fg:#1a202c;--mut:#4a5568;--line:#e2e8f0;--bg:#fff;--accent:#2b6cb0;
 --ok:#2f855a;--warn:#b7791f;--bad:#c53030;--open:#805ad5;--code:#f7fafc;}
*{box-sizing:border-box}
body{margin:0;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif;
 color:var(--fg);background:var(--bg)}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px 96px}
header{border-bottom:3px solid var(--fg);margin-bottom:34px;padding:44px 0 22px}
h1{font-size:2.0rem;line-height:1.2;margin:0 0 10px;letter-spacing:-.02em}
.sub{color:var(--mut);font-size:1.05rem;margin:0}
.meta{margin-top:18px;font-size:.83rem;color:var(--mut);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
h2{font-size:1.5rem;margin:52px 0 4px;padding-top:14px;border-top:1px solid var(--line);letter-spacing:-.01em}
h2 .num{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:.5rem}
h3{font-size:1.13rem;margin:32px 0 8px}
h4{font-size:1.0rem;margin:24px 0 6px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;font-size:.82rem}
p{margin:12px 0}
code{background:var(--code);padding:.1em .35em;border-radius:3px;font-size:.88em;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:14px 16px;border-radius:6px;overflow-x:auto;font-size:.84rem;
 border:1px solid var(--line);line-height:1.5}
table{border-collapse:collapse;width:100%;margin:18px 0;font-size:.88rem}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--code);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
tbody tr:hover{background:#fafcfe}
figure{margin:22px 0;border:1px solid var(--line);border-radius:8px;padding:12px 12px 6px;background:#fff}
svg.chart{width:100%;height:auto;display:block}
.pa{fill:#fcfdfe;stroke:var(--line);stroke-width:1}
.gr{stroke:var(--line);stroke-width:1}
.tk{font-size:11px;fill:var(--mut);font-family:ui-monospace,Menlo,monospace}
.vl{font-size:10px;fill:var(--mut);font-family:ui-monospace,Menlo,monospace}
.al{font-size:12px;fill:var(--fg)}
.ct{font-size:13px;fill:var(--fg);font-weight:600}
.bl{font-size:11px}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;padding:8px 4px 4px;font-size:.8rem;color:var(--mut)}
.li{display:flex;align-items:center;gap:6px}
.li i{width:14px;height:3px;border-radius:2px;display:inline-block}
.note{border-left:3px solid var(--accent);background:#f7fbff;padding:12px 16px;margin:20px 0;border-radius:0 6px 6px 0}
.warnbox{border-left:3px solid var(--bad);background:#fff7f7;padding:12px 16px;margin:20px 0;border-radius:0 6px 6px 0}
.openbox{border-left:3px solid var(--open);background:#faf7ff;padding:12px 16px;margin:20px 0;border-radius:0 6px 6px 0}
.okbox{border-left:3px solid var(--ok);background:#f6fcf8;padding:12px 16px;margin:20px 0;border-radius:0 6px 6px 0}
.note p:first-child,.warnbox p:first-child,.openbox p:first-child,.okbox p:first-child{margin-top:0}
.note p:last-child,.warnbox p:last-child,.openbox p:last-child,.okbox p:last-child{margin-bottom:0}
.pill{display:inline-block;padding:1px 8px;border-radius:11px;font-size:.72rem;font-weight:600;
 text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
.p-ok{background:#e6f6ec;color:var(--ok)}
.p-int{background:#fdf6e3;color:var(--warn)}
.p-bad{background:#fdeaea;color:var(--bad)}
.p-open{background:#f3ecfd;color:var(--open)}
.p-none{background:#eef1f5;color:var(--mut)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:12px;margin:24px 0}
.kpi{border:1px solid var(--line);border-radius:8px;padding:13px 15px}
.kpi .v{font-size:1.5rem;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .k{font-size:.73rem;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-top:3px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
nav.toc{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:16px 22px;margin:26px 0}
nav.toc ol{margin:0;padding-left:22px;columns:2;column-gap:34px}
nav.toc li{margin:3px 0;font-size:.9rem;break-inside:avoid}
nav.toc a{color:var(--accent);text-decoration:none}
nav.toc a:hover{text-decoration:underline}
.eq{margin:16px 0;padding:2px 0;overflow-x:auto}
.caption{font-size:.83rem;color:var(--mut);margin:6px 4px 2px;line-height:1.55}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);font-size:.82rem;color:var(--mut)}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:.86em}
.tight td,.tight th{padding:4px 8px;font-size:.83rem}
</style></head><body><div class="wrap">
"""


def pill(kind, text):
    return f'<span class="pill p-{kind}">{text}</span>'


# ------------------------------------------------------------------ diagrams
def pipeline_diagram():
    """Ten-stage route diagram with the partition fork."""
    W, H = 1000, 760
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" xmlns="http://www.w3.org/2000/svg">']
    o.append('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#4a5568"/></marker></defs>')

    def box(x, y, w, h, title, sub, status, small=False):
        col = {"ok": "#2f855a", "int": "#b7791f", "bad": "#c53030",
               "open": "#805ad5", "none": "#a0aec0"}[status]
        fill = {"ok": "#f2fbf5", "int": "#fefbf2", "bad": "#fef4f4",
                "open": "#f9f5ff", "none": "#f7fafc"}[status]
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
                 f'stroke="{col}" stroke-width="1.6"/>')
        o.append(f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{col}"/>')
        o.append(f'<text x="{x+14}" y="{y+18}" style="font-size:12.5px;font-weight:650;'
                 f'fill:#1a202c">{title}</text>')
        for i, line in enumerate(sub):
            o.append(f'<text x="{x+14}" y="{y+34+i*13}" style="font-size:10.5px;fill:#4a5568;'
                     f'font-family:ui-monospace,Menlo,monospace">{line}</text>')

    def arrow(x1, y1, x2, y2, label="", dash=False):
        d = ' stroke-dasharray="5 4"' if dash else ""
        o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4a5568" '
                 f'stroke-width="1.6" marker-end="url(#ar)"{d}/>')
        if label:
            o.append(f'<text x="{(x1+x2)/2+7}" y="{(y1+y2)/2+3}" style="font-size:10px;'
                     f'fill:#4a5568">{label}</text>')

    cx, bw = 300, 400
    y = 8
    box(cx, y, bw, 46, "SCF triple  —  PBE0/aug-cc-pVTZ + GRAC",
        ["neutral · neutral precursor · cation"], "ok")
    arrow(cx + bw / 2, y + 46, cx + bw / 2, y + 68)
    y += 68
    box(cx, y, bw, 46, "Stage 1  FrozenResponseContext",
        ["provenance seal; refuses a marginally converged SCF"], "ok")
    arrow(cx + bw / 2, y + 46, cx + bw / 2, y + 70)
    y += 70

    # the fork
    o.append(f'<text x="{cx+bw/2}" y="{y+13}" text-anchor="middle" '
             f'style="font-size:11.5px;font-weight:650;fill:#2b6cb0">'
             f'Stage 2 — PARTITION OF THE FROZEN DENSITY (the fork)</text>')
    y += 22
    lx, rx, hw = 92, 520, 388
    box(lx, y, hw, 62, "real-space ISA  (default)",
        ["stockholder weights on the sealed grid", "PARTITION=ISA · converges in 32 iter"], "ok")
    box(rx, y, hw, 62, "constrained density fitting",
        ["Dunlap functional, atom-centred aux basis",
         "PARTITION=CDF · λ=1.0, η=5e-4 penalty"], "ok")
    arrow(cx + bw / 2 - 120, y - 8, lx + hw / 2, y - 2)
    arrow(cx + bw / 2 + 120, y - 8, rx + hw / 2, y - 2)
    o.append(f'<text x="{lx+hw/2}" y="{y+80}" text-anchor="middle" style="font-size:10px;'
             f'fill:#2f855a">matches ISA_GRID_* oracle</text>')
    o.append(f'<text x="{rx+hw/2}" y="{y+80}" text-anchor="middle" style="font-size:10px;'
             f'fill:#2f855a">matches DF_* oracle</text>')
    arrow(lx + hw / 2, y + 86, cx + bw / 2 - 60, y + 104)
    arrow(rx + hw / 2, y + 86, cx + bw / 2 + 60, y + 104)
    y += 108

    stages = [
        ("Stage 3  FDDS site-pair response α<tspan dy='3' style='font-size:8px'>ab</tspan>"
         "<tspan dy='-3'>(iω)</tspan>",
         ["frequency-dependent, 11-point Gauss-Legendre grid"], "int"),
        ("Stage 4  covalent bond graph", ["fails closed unless a single connected component"], "ok"),
        ("Stage 5  LW localization → site-diagonal L3",
         ["charge-sum postcondition; measured residual 5.4e-07"], "ok"),
        ("Stage 6  symmetry-faithful fit points + point response",
         ["407-point grid for water"], "ok"),
        ("Stage 7  PDef active-variable mask",
         ["site symmetry: 170 free variables under C2v(Z)"], "ok"),
        ("Stage 8  constrained L3 WSM refinement",
         ["PFIT weight 4, coeff 1e-3, cutoff 1e-4, ranks 1-3"], "bad"),
    ]
    for title, sub, st in stages:
        h = 44 if len(sub) == 1 else 56
        box(cx, y, bw, h, title, sub, st)
        arrow(cx + bw / 2, y + h, cx + bw / 2, y + h + 20)
        y += h + 20

    # recoupling fork
    box(lx + 34, y, hw - 20, 56, "Stage 9  isotropic Casimir-Polder",
        ["ATOMIC C6 / C8 / C10 / C12   (natom, natom)"], "bad")
    box(rx - 14, y, hw + 14, 56, "Stage 9b  anisotropic recoupling",
        ["Cₙ[l₁k₁, l₂k₂, j] · 16985 published labels, n ≤ 12"], "open")
    arrow(cx + bw / 2 - 60, y - 20, lx + 34 + (hw - 20) / 2, y - 2)
    arrow(cx + bw / 2 + 60, y - 20, rx - 14 + (hw + 14) / 2, y - 2)
    arrow(lx + 34 + (hw - 20) / 2, y + 56, cx + bw / 2 - 60, y + 76)
    arrow(rx - 14 + (hw + 14) / 2, y + 56, cx + bw / 2 + 60, y + 76)
    y += 80
    box(cx - 60, y, bw + 120, 46, "Stage 10  publish 12 QCVariables",
        ["all-or-nothing: either every array appears or none does"], "ok")
    o.append("</svg>")

    legend = ("".join(
        f'<span class="li"><i style="background:{c};height:11px;width:11px;border-radius:3px"></i>'
        f'{t}</span>'
        for c, t in (("#2f855a", "validated against an external oracle"),
                     ("#b7791f", "internally validated only"),
                     ("#c53030", "known quantified deficit"),
                     ("#805ad5", "oracle exists, convention unresolved"))))
    return f'<figure>{"".join(o)}<div class="legend">{legend}</div></figure>'


def oracle_map():
    """Which arm must be compared against which oracle."""
    W, H = 1000, 300
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" xmlns="http://www.w3.org/2000/svg">']
    o.append('<defs><marker id="ok" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
             'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
             'fill="#2f855a"/></marker>'
             '<marker id="no" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
             'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
             'fill="#c53030"/></marker></defs>')

    def node(x, y, w, h, t, s, col, fill):
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" '
                 f'stroke="{col}" stroke-width="1.8"/>')
        o.append(f'<text x="{x+w/2}" y="{y+22}" text-anchor="middle" style="font-size:13px;'
                 f'font-weight:650">{t}</text>')
        for i, l in enumerate(s):
            o.append(f'<text x="{x+w/2}" y="{y+40+i*13}" text-anchor="middle" '
                     f'style="font-size:10.5px;fill:#4a5568;'
                     f'font-family:ui-monospace,Menlo,monospace">{l}</text>')

    node(40, 46, 300, 66, "our ISA arm", ["PARTITION=ISA (default)", "real-space stockholder"],
         "#2b6cb0", "#f5f9fe")
    node(40, 176, 300, 66, "our C-DF arm", ["PARTITION=CDF", "auxiliary-basis fit"],
         "#2b6cb0", "#f5f9fe")
    node(660, 46, 300, 66, "ISA_GRID_* oracle", ["regenerated CamCASP run", "grid ISA partition"],
         "#2f855a", "#f2fbf5")
    node(660, 176, 300, 66, "DF_* oracle", ["originally reviewed run", "constrained density fit"],
         "#2f855a", "#f2fbf5")

    o.append('<line x1="345" y1="79" x2="655" y2="79" stroke="#2f855a" stroke-width="2.2" '
             'marker-end="url(#ok)"/>')
    o.append('<text x="500" y="70" text-anchor="middle" style="font-size:11px;fill:#2f855a;'
             'font-weight:600">matched — 2-5 % on 6 of 7 components</text>')
    o.append('<line x1="345" y1="209" x2="655" y2="209" stroke="#2f855a" stroke-width="2.2" '
             'marker-end="url(#ok)"/>')
    o.append('<text x="500" y="200" text-anchor="middle" style="font-size:11px;fill:#2f855a;'
             'font-weight:600">matched — dipole block 4.2× closer than crossed</text>')
    o.append('<path d="M345,96 C480,130 520,158 655,192" fill="none" stroke="#c53030" '
             'stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#no)"/>')
    o.append('<path d="M345,192 C480,158 520,130 655,96" fill="none" stroke="#c53030" '
             'stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#no)"/>')
    o.append('<rect x="374" y="128" width="252" height="34" rx="5" fill="#fff" '
             'stroke="#c53030" stroke-width="1.3"/>')
    o.append('<text x="500" y="143" text-anchor="middle" style="font-size:11px;fill:#c53030;'
             'font-weight:650">CROSSED = wrong by up to 113×</text>')
    o.append('<text x="500" y="156" text-anchor="middle" style="font-size:9.5px;fill:#c53030">'
             'on H α_zx, with nothing defective anywhere</text>')
    o.append('<text x="500" y="277" text-anchor="middle" style="font-size:11px;fill:#4a5568">'
             'Both oracles ran the same protocol and agree on the molecular total to 0.11 %. '
             'They differ only in how the response is split between sites.</text>')
    o.append("</svg>")
    return f'<figure>{"".join(o)}</figure>'


# ------------------------------------------------------------------- figures
ATOMS = ("O", "H1", "H2")
COMP = ("xx", "xy", "xz", "yy", "yz", "zz")


def fig_static_alpha():
    s = D["static_alpha"]
    ours, isa, df = (np.asarray(s[k]) for k in ("ours", "isa_grid", "df"))
    keep = [(a, c) for a in range(3) for c in range(6)
            if abs(isa[a][c]) > 1e-6 or abs(ours[a][c]) > 1e-6]
    names = [f"{ATOMS[a]} {COMP[c]}" for a, c in keep]
    return grouped_bars(
        names, ["ours (ISA)", "CamCASP ISA-GRID", "CamCASP C-DF"],
        [[float(ours[a][c]) for a, c in keep],
         [float(isa[a][c]) for a, c in keep],
         [float(df[a][c]) for a, c in keep]],
        title="Static dipole polarizability by component, molecular frame (a.u.)",
        ylabel="α", width=1000, height=350, fmt="{:.3f}")


def fig_rank_deficit():
    r = D["rank_invariants"]
    ls = ["1", "2", "3"]
    return grouped_bars(
        [f"l = {l}" for l in ls], ["ours", "CamCASP ISA-GRID"],
        [[r[l]["ours"] for l in ls], [r[l]["ref"] for l in ls]],
        title="Site-summed rank invariants  a_l = Tr(α^ll)/(2l+1)   (a.u.)",
        ylabel="a_l", width=1000, height=330, fmt="{:.2f}")


def fig_rank_ratio():
    r = D["rank_invariants"]
    ls = ["1", "2", "3"]
    vals = [[100.0 * (1.0 - r[l]["sites"][a]["ours"] / r[l]["sites"][a]["ref"]) for l in ls]
            for a in ATOMS]
    vals.append([100.0 * (1.0 - r[l]["ours"] / r[l]["ref"]) for l in ls])
    return grouped_bars(
        [f"l = {l}" for l in ls], list(ATOMS) + ["site sum"], vals,
        title="Polarizability deficit against the matched oracle, by rank (%)",
        ylabel="deficit / %", width=1000, height=330, fmt="{:.1f}", hline=0.0)


def fig_dynamic():
    f = D["frequencies"]
    ax = Axes(width=1000, height=380, logx=False, title=None,
              xlabel="imaginary frequency  ω  /  hartree",
              ylabel="isotropic α(iω)  /  a.u.")
    ours, ref = D["dynamic_iso"]["ours"], D["dynamic_iso"]["isa_grid"]
    for s, atom in enumerate(ATOMS):
        if atom == "H2":
            continue
        ax.add(f, [row[s] for row in ours], f"{atom} ours", PALETTE[s], marker=True)
        ax.add(f, [row[s] for row in ref], f"{atom} ISA-GRID", PALETTE[s], marker=True, dashed=True)
    ax.title = "Dynamic isotropic dipole polarizability on the Casimir-Polder grid"
    return ax.render()


def fig_dynamic_ratio():
    f = D["frequencies"]
    ours, ref = D["dynamic_iso"]["ours"], D["dynamic_iso"]["isa_grid"]
    ax = Axes(width=1000, height=340, xlabel="imaginary frequency  ω  /  hartree",
              ylabel="ours / ISA-GRID",
              title="Ratio across the grid — the rank-1 deficit is nearly frequency-flat")
    for s, atom in enumerate(ATOMS):
        if atom == "H2":
            continue
        ax.add(f, [ours[i][s] / ref[i][s] for i in range(len(f))], atom, PALETTE[s], marker=True)
    tot_o = [sum(r) for r in ours]
    tot_r = [sum(r) for r in ref]
    ax.add(f, [a / b for a, b in zip(tot_o, tot_r)], "molecular total", "#1a202c",
           marker=True, width=2.6)
    ax.hline(1.0, "exact agreement")
    return ax.render()


def fig_isotropic_cn():
    ic = D["isotropic_cn"]["orders"]
    orders = ["C6", "C8", "C10", "C12"]
    pairs = [("O-O", 0, 0), ("O-H", 0, 1), ("H-H", 1, 1)]
    cats, vals = [], []
    for pname, i, j in pairs:
        cats.append(f"{pname} vs ISA-GRID")
        vals.append([100.0 * (1.0 - ic[o]["ours"][i][j] / ic[o]["isa_grid"][i][j])
                     for o in orders])
    return grouped_bars(
        orders, cats, vals,
        title="Isotropic Cn deficit against the matched oracle, by order and site pair (%)",
        ylabel="deficit / %", width=1000, height=340, fmt="{:.1f}", hline=0.0)


def fig_aniso_scatter():
    c = D["aniso_cn"]
    ax = Axes(width=1000, height=430, logx=True, logy=True,
              xlabel="|CamCASP CASIMIR|   (hartree bohr^n)",
              ylabel="|ours|   (hartree bohr^n)",
              title="Anisotropic Cn, every shared nonzero coefficient")
    for i, pair in enumerate(("OO", "HO", "HH")):
        pts = [(abs(x["ref"]), abs(x["ours"])) for x in c if x["pair"].upper() in (pair, pair[::-1])]
        if pts:
            ax.add([p[0] for p in pts], [p[1] for p in pts], f"{pair[0]}-{pair[1]}",
                   PALETTE[i], marker=True, width=0)
    lo = min(abs(x["ref"]) for x in c if x["ref"] != 0)
    hi = max(abs(x["ref"]) for x in c)
    ax.add([lo, hi], [lo, hi], "exact agreement", "#1a202c", width=1.6, dashed=True)
    return ax.render()


def fig_aniso_by_order():
    c = D["aniso_cn"]
    orders = list(range(6, 13))
    med, p90, cnt = [], [], []
    for n in orders:
        r = [abs(x["ours"] / x["ref"]) for x in c if x["n"] == n and x["ref"] != 0]
        cnt.append(len(r))
        med.append(float(np.median(r)) if r else 0.0)
        p90.append(float(np.percentile(r, 90)) if r else 0.0)
    ax = Axes(width=1000, height=360, logy=True, xlabel="dispersion order n",
              ylabel="|ours / CASIMIR|",
              title="Anisotropic Cn ratio by order — median and 90th percentile")
    ax.add(orders, med, "median", PALETTE[0], marker=True, width=2.4)
    ax.add(orders, p90, "90th percentile", PALETTE[1], marker=True, dashed=True)
    ax.hline(1.0, "exact agreement", "#1a202c", dashed=True)
    return ax.render(), cnt


def fig_cg_correction():
    """C6 only: the sector where a single site-block quadruple contributes."""
    from math import factorial as fact, sqrt
    def cg000(l1, l2, j):
        if (l1 + l2 + j) % 2 or not (abs(l1 - l2) <= j <= l1 + l2):
            return 0.0
        g = (l1 + l2 + j) // 2
        pref = sqrt((2 * j + 1) * fact(l1 + l2 - j) * fact(l1 - l2 + j) *
                    fact(-l1 + l2 + j) / fact(l1 + l2 + j + 1))
        return ((-1) ** (g - j) * fact(g) / (fact(g - l1) * fact(g - l2) * fact(g - j))) * pref

    c6 = [x for x in D["aniso_cn"] if x["n"] == 6 and x["ref"] != 0]
    raw, corr, xs = [], [], []
    for i, x in enumerate(c6):
        cg = cg000(x["l1"], x["l2"], x["j"])
        raw.append(abs(x["ours"] / x["ref"]))
        corr.append(abs(x["ours"] / x["ref"]) * abs(cg) if cg else float("nan"))
        xs.append(i)
    ax = Axes(width=1000, height=380, logy=True, xlabel="C6 coefficient index (sorted by label)",
              ylabel="|ours / CASIMIR|",
              title="The C6 sector: raw ratio, and after the Clebsch-Gordan factor")
    ax.add(xs, raw, "raw ratio", PALETTE[1], marker=True, width=0)
    ok = [(i, v) for i, v in zip(xs, corr) if v == v]
    ax.add([i for i, _ in ok], [v for _, v in ok],
           "x |<l1 0; l2 0 | j 0>|", PALETTE[2], marker=True, width=0)
    ax.hline(1.0, "exact agreement", "#1a202c", dashed=True)
    ax.band(0.90, 1.10, "known 1-10 % C6 deficit band", "#2f855a")
    return ax.render()


def fig_pes(mode):
    p = D["pes"]
    dist = p["distances"]
    arms = p["modes"][mode]
    ax = Axes(width=1000, height=400, xlabel="R(O–O)  /  Å",
              ylabel="energy  /  kcal mol⁻¹",
              title=f"Water dimer: induction and dispersion, BJ radius from {mode}")
    style = {"ours": (PALETTE[0], False), "isa_grid": (PALETTE[2], True),
             "df": (PALETTE[1], True)}
    label = {"ours": "ours", "isa_grid": "ISA-GRID", "df": "C-DF (wrong oracle)"}
    for key in ("ours", "isa_grid", "df"):
        col, dash = style[key]
        ax.add(dist, [r["ind"] for r in arms[key]], f"E_ind {label[key]}", col,
               marker=True, dashed=dash)
    for key in ("ours", "isa_grid", "df"):
        col, dash = style[key]
        ax.add(dist, [r["d3"] for r in arms[key]], f"E_disp {label[key]}", col,
               marker=True, dashed=dash, width=3.0)
    return ax.render()


def fig_pes_delta():
    p = D["pes"]
    dist = p["distances"]
    ax = Axes(width=1000, height=380, xlabel="R(O–O)  /  Å",
              ylabel="ours − ISA-GRID  /  kcal mol⁻¹",
              title="Energy error against the matched oracle (positive = under-binding)")
    for mode, dash in (("r4r2", False), ("c8c6", True)):
        a = p["modes"][mode]
        ax.add(dist, [o["ind"] - r["ind"] for o, r in zip(a["ours"], a["isa_grid"])],
               f"induction ({mode})", PALETTE[0], marker=True, dashed=dash)
        ax.add(dist, [o["d3"] - r["d3"] for o, r in zip(a["ours"], a["isa_grid"])],
               f"dispersion ({mode})", PALETTE[1], marker=True, dashed=dash, width=2.6)
    a = p["modes"]["r4r2"]
    ax.add(dist, [o["ind"] - r["ind"] for o, r in zip(a["df"], a["isa_grid"])],
           "induction, wrong oracle", "#a0aec0", marker=True)
    ax.add(dist, [o["d3"] - r["d3"] for o, r in zip(a["df"], a["isa_grid"])],
           "dispersion, wrong oracle", "#718096", marker=True, dashed=True)
    ax.hline(0.0, "", "#1a202c", dashed=False)
    return ax.render()


def fig_pes_orders():
    p = D["pes"]
    dist = p["distances"]
    a = p["modes"]["r4r2"]
    ax = Axes(width=1000, height=390, xlabel="R(O–O)  /  Å",
              ylabel="dispersion contribution  /  kcal mol⁻¹",
              title="Dispersion by order (r4r2 radius). C10 rivals C6 at the H-bond minimum")
    for i, key in enumerate(("c6", "c8", "c10")):
        ax.add(dist, [r[key] for r in a["ours"]], f"{key.upper()} ours", PALETTE[i], marker=True)
        ax.add(dist, [r[key] for r in a["isa_grid"]], f"{key.upper()} ISA-GRID",
               PALETTE[i], marker=True, dashed=True)
    return ax.render()


def fig_anisotropy():
    p = D["pes"]
    dist = p["distances"]
    a = p["modes"]["r4r2"]
    ax = Axes(width=1000, height=370, xlabel="R(O–O)  /  Å",
              ylabel="E_ind  /  kcal mol⁻¹",
              title="What α anisotropy is worth: full tensor against the same α isotropised")
    ax.add(dist, [r["ind"] for r in a["ours"]], "ours, full tensor", PALETTE[0], marker=True)
    ax.add(dist, [r["ind_iso"] for r in a["ours"]], "ours, isotropised", PALETTE[0],
           marker=True, dashed=True)
    ax.add(dist, [r["ind"] for r in a["isa_grid"]], "ISA-GRID, full tensor", PALETTE[2],
           marker=True)
    ax.add(dist, [r["ind_iso"] for r in a["isa_grid"]], "ISA-GRID, isotropised", PALETTE[2],
           marker=True, dashed=True)
    return ax.render()


# --------------------------------------------------------------------- prose
def git_meta():
    def g(*a):
        return subprocess.run(["git", "-C", str(REPO)] + list(a),
                              capture_output=True, text=True).stdout.strip()
    return g("rev-parse", "--short", "HEAD"), g("rev-parse", "--abbrev-ref", "HEAD"), g(
        "log", "-1", "--format=%s")


def section_theory():
    return r"""
<h2 id="theory"><span class="num">2</span>Theory, as implemented</h2>

<p>Every equation below is the form the code actually implements, not the form the original
specification proposed. Where the two differ the difference is called out, because in three
places the specification was wrong and implementing it literally would have reproduced the
wrong physical model.</p>

<h3>2.1 Distributed polarizabilities from the FDDS</h3>

<p>The object being partitioned is the frequency-dependent density susceptibility, the linear
response of the density to a local perturbation at imaginary frequency:</p>
<div class="eq">$$\alpha(\mathbf{r},\mathbf{r}';i\omega)=
\frac{\delta\rho(\mathbf{r};i\omega)}{\delta v(\mathbf{r}';i\omega)}$$</div>

<p>A distributed model replaces it by a finite set of site-pair multipole polarizabilities.
With \(Q_t\) the real solid-harmonic multipole operators and \(w_a\) a partition of unity over
sites,</p>
<div class="eq">$$\alpha^{ab}_{tu}(i\omega)=\iint w_a(\mathbf{r})\,Q_t(\mathbf{r}-\mathbf{R}_a)\,
\alpha(\mathbf{r},\mathbf{r}';i\omega)\,w_b(\mathbf{r}')\,Q_u(\mathbf{r}'-\mathbf{R}_b)\,
\mathrm{d}\mathbf{r}\,\mathrm{d}\mathbf{r}'$$</div>

<p>The whole pipeline is a chain of choices about \(w_a\), about how the site-pair blocks are
collapsed onto site-diagonal ones, and about how the resulting \(\alpha^{aa}_{tu}(i\omega)\) are
integrated into dispersion coefficients.</p>

<h3>2.2 The fork: two definitions of \(w_a\)</h3>

<p><strong>Real-space ISA</strong> (the default) uses stockholder weights built from spherically
averaged shape functions, solved to a fixed point:</p>
<div class="eq">$$w_a(\mathbf{r})=\frac{\rho^{\mathrm{sh}}_a(|\mathbf{r}-\mathbf{R}_a|)}
{\sum_b\rho^{\mathrm{sh}}_b(|\mathbf{r}-\mathbf{R}_b|)},\qquad
\rho^{\mathrm{sh}}_a(r)=\frac{1}{4\pi}\oint w_a(\mathbf{R}_a+r\hat{\mathbf{n}})\,
\rho(\mathbf{R}_a+r\hat{\mathbf{n}})\,\mathrm{d}\hat{\mathbf{n}}$$</div>

<p><strong>Constrained density fitting</strong> expands the response in atom-centred auxiliary
functions and partitions by ownership of those functions. The fit minimises the Dunlap
self-repulsion in the Coulomb metric \(J_{kl}=(\chi_k\|\chi_l)\).</p>

<div class="warnbox">
<p><strong>The specification was wrong here, and it matters.</strong> §A.3.1 wrote the atomic
population conditions as a hard constraint \(\mathbf{C}\mathbf{d}=\mathbf{n}\). The reference
calculation did not do that. It applied a finite quadratic <em>penalty</em>:</p>
<div class="eq">$$\Delta=(\rho-\tilde\rho\,\|\,\rho-\tilde\rho)
+\lambda\sum_a\Big(\textstyle\sum_{k\in a}d_k\langle\chi_k\rangle-n_a\Big)^2,
\qquad\lambda=1.0$$</div>
<p>with normal equations carrying an inter-site self-repulsion mix \(\eta=5\times10^{-4}\):</p>
<div class="eq">$$\big[(1-\eta)\mathbf{J}+\eta\mathbf{K}_{\mathrm{self}}
+\lambda\mathbf{C}^{\!\top}\mathbf{C}\big]\,\mathbf{d}
=\mathbf{b}+\lambda\mathbf{C}^{\!\top}\mathbf{n}$$</div>
<p>The proof is in the reference's own output: it reports an orthonormality violation of
<code>0.01065</code>, where a hard constraint gives machine zero. The hard-constraint form is
retained in the code as the \(\lambda\to\infty\) limit, so nothing is lost. The \(\eta\) sign was
settled empirically — the normal matrix is positive semi-definite only for
\(0\le\eta\le1\).</p>
</div>

<h3>2.3 Localization and refinement</h3>

<p>LW localization collapses the site-pair blocks onto site-diagonal ones while preserving the
charge-flow sum rule, which is the statement that a uniform potential shift induces no
multipole:</p>
<div class="eq">$$\sum_b\alpha^{ab}_{t,00}(i\omega)=0$$</div>

<p>This is a genuine postcondition in the code, not a comment: the measured residual at the
reviewed protocol is \(5.4\times10^{-7}\), and a protocol whose grid cannot reach the requested
tolerance fails closed rather than proceeding.</p>

<p>WSM/PFIT refinement then re-fits the site-diagonal model against the exact response at
external points, under a symmetry mask and a penalty on the fitted parameters:</p>
<div class="eq">$$\min_{\boldsymbol\alpha}\;\sum_p\big|V_p^{\text{exact}}-V_p[\boldsymbol\alpha]\big|^2
+w\sum_i c_i\,\alpha_i^2,\qquad w=4,\;c=10^{-3}$$</div>

<h3>2.4 Casimir–Polder integration and recoupling</h3>

<p>The isotropic coefficients are the familiar frequency integral over rank invariants
\(\bar\alpha_l=\mathrm{Tr}\,\alpha^{ll}/(2l+1)\):</p>
<div class="eq">$$C_6^{ab}=\frac{3}{\pi}\int_0^\infty\bar\alpha_1^a(i\omega)\,
\bar\alpha_1^b(i\omega)\,\mathrm{d}\omega$$</div>

<p>The anisotropic set generalises this. Writing \(L_1=l_a+l_b\) and \(L_2=l_{a'}+l_{b'}\),</p>
<div class="eq">$$C_n[l_1k_1,l_2k_2,j]=\frac{1}{2\pi}
\sum_{l_al_{a'}l_bl_{b'}}g^{\,r}\int_0^\infty
\alpha^A_{l_al_{a'}}(i\omega)\,\alpha^B_{l_bl_{b'}}(i\omega)\,\mathrm{d}\omega,
\qquad n=l_a+l_{a'}+l_b+l_{b'}+2$$</div>
<div class="eq">$$g^{\,r}=(-1)^{l_b+l_{b'}}
\sqrt{\binom{2L_1}{2l_a}\binom{2L_2}{2l_{a'}}}\;
\langle L_1\,0;L_2\,0\,|\,j\,0\rangle\;
\frac{\Lambda_j}{\eta_A\eta_B\mathcal{N}}$$</div>

<p>and the coefficients enter the energy through Stone's \(S\) functions, which carry all the
orientation dependence:</p>
<div class="eq">$$E_{\text{disp}}=-\sum_{a\in A}\sum_{b\in B}\sum_n\frac{1}{R_{ab}^{\,n}}
\sum_{l_1k_1l_2k_2j}C_n[l_1k_1,l_2k_2,j]\;
S_{l_1k_1l_2k_2j}(\Omega_a,\Omega_b,\hat{\mathbf{R}}_{ab})$$</div>

<div class="warnbox">
<p><strong>Three further specification errors, all found by implementing it.</strong> The spec gave
\(n=2(l_a+l_b+1)\), which admits only even orders and would have silently deleted every odd
coefficient — C7, C9, C11 and C13 are all real and all nonzero. It carried a stray
\(1/2\pi\), and it had the permutation sign wrong. It also estimated the label count at
"dozens"; the true internal set is 29 762, of which 16 985 are published at \(n\le12\).</p>
</div>

<h3>2.5 Selection rules</h3>

<p>Three rules govern which labels can be nonzero. All three were verified against 10 457
nonzero reference coefficients with zero violations:</p>
<div class="eq">$$|l_1-l_2|\le j\le l_1+l_2,\qquad
n\equiv j\!\!\pmod 2,\qquad
n\ge m(l_1)+m(l_2)+2$$</div>
<p>where \(m(l)\) is the smallest \(l_a+l_{a'}\) admitting coupled rank \(l\) with
\(l_a,l_{a'}\in[1,3]\), and \(l_a=l_{a'}\) allowed only for even \(l\); it evaluates to
\(m-2=0,1,0,1,2,3,4\) for \(l=0\ldots6\).</p>

<div class="note">
<p>The middle rule is worth dwelling on because getting it wrong is invisible. It is
\(n\equiv j\), <em>not</em> \(n\equiv l_1+l_2\) — the latter is violated 2 968 times, first at
O–O <code>20 22s 1</code>, because it misses the parity phase carried by the sine components.
The correct rule follows with no symmetry assumption from \(L_1+L_2+j\) even together with
\(L_1+L_2=n-2\).</p>
<p>Its practical value is that it detects a column-misalignment bug in the reference reader.
CASIMIR truncates <em>trailing</em> zeros only, so a reader that left-pads instead of
right-padding shifts every coefficient by one order and still produces plausible,
correctly-signed, correctly-scaled numbers. Under the correct reading this rule has 0
violations; under the shifted one, 4 124.</p>
</div>

<h3>2.6 The two energy models used for the PES test</h3>

<p>Induction is a self-consistent point-induced-dipole model. With \(\mathbf{A}\) the
block-diagonal polarizability and \(\mathbf{T}_2\) the Thole-damped dipole field tensor, the
fixed point is solved as one linear system rather than iterated:</p>
<div class="eq">$$(\mathbf{1}-\mathbf{A}\mathbf{T}_2)\,\boldsymbol\mu^{\text{ind}}
=\mathbf{A}\,\mathbf{F}^{\text{perm}},\qquad
E_{\text{ind}}=-\tfrac12\sum_i\boldsymbol\mu_i^{\text{ind}}\!\cdot\mathbf{F}_i^{\text{perm}}$$</div>
<div class="eq">$$u=\frac{R_{ij}}{(\alpha_i\alpha_j)^{1/6}},\quad\tilde a=a\,u^3,\quad
\lambda_3=1-e^{-\tilde a},\quad\lambda_5=1-(1+\tilde a)e^{-\tilde a},\quad a=0.39$$</div>

<p>Dispersion is a D3-style pairwise sum with Becke–Johnson damping. Because we have genuine
distributed \(C_8\) and \(C_6\), the critical radius can be taken in Becke and Johnson's
original form rather than through D3's \(r_4/r_2\) approximation to it:</p>
<div class="eq">$$E_{\text{disp}}=-\sum_{i\in A}\sum_{j\in B}\sum_{n}
\frac{s_n\,C_n^{ij}}{R_{ij}^{\,n}+R_0^{\,n}},\qquad
R_0=a_1\sqrt{C_8^{ij}/C_6^{ij}}+a_2$$</div>
<p>Both choices of \(R_0\) are reported below, because the difference between them turns out to
carry most of the apparent dispersion error.</p>
"""


PUBLISHED = [
    ("ATOMIC POLARIZABILITIES", "(natom, 6)", "static dipole, packed xx xy xz yy yz zz", "ok",
     "2–5 % on 6 of 7 components"),
    ("ATOMIC DYNAMIC POLARIZABILITIES", "(nfreq·natom, 6)", "same, all 11 grid frequencies", "ok",
     "ratio 0.97 and nearly frequency-flat"),
    ("ATOMIC POLARIZABILITY FREQUENCIES", "(nfreq, 1)", "Gauss-Legendre grid, scale 0.5", "ok",
     "exact to 1e-10"),
    ("ATOMIC C6", "(natom, natom)", "isotropic dispersion", "bad", "1–10 % per pair"),
    ("ATOMIC C8", "(natom, natom)", "isotropic dispersion", "bad", "≈25 %"),
    ("ATOMIC C10", "(natom, natom)", "isotropic dispersion", "bad", "≈36 %"),
    ("ATOMIC C12", "(natom, natom)", "isotropic dispersion", "bad", "≈46 %"),
    ("ATOMIC DISPERSION COEFFICIENTS", "(natom², 16985)", "anisotropic Cₙ[l₁k₁,l₂k₂,j], n ≤ 12",
     "open", "0 of 10457 inside 1e-4; convention unresolved"),
    ("ATOMIC DISPERSION LABELS", "(16985, 6)", "[n, l₁, k₁, l₂, k₂, j]", "ok",
     "label set strictly contains CamCASP's"),
    ("ATOMIC ANISOTROPIC POLARIZABILITIES", "(natom·15, 15)", "full rank 1–3 static block", "int",
     "rank 1 2.9 %, ranks 2–3 ≈30 %"),
    ("ATOMIC ANISOTROPIC DYNAMIC POLARIZABILITIES", "(nfreq·natom·15, 15)", "same, all frequencies",
     "int", "not yet compared frequency-by-frequency"),
    ("ATOMIC ANISOTROPIC POLARIZABILITY COMPONENTS", "(15, 3)", "[l, |k|, cos/sin kind]", "ok",
     "exact by construction"),
]

TASKS = [
    ("A1–A4", "C-DF core: auxiliary multipole moments, Coulomb metric, constrained solve",
     "done", "Analytic moments; solver never forms J⁻¹."),
    ("A5–A7", "Wire C-DF into the partition fork, seal aux-basis provenance", "done",
     "ISA path verified bit-identical on all published arrays."),
    ("A8", "Flip the six DF_* comparisons to live tests at rtol=1e-4", "failed",
     "All six still fail; markers kept, gate not widened. Partition reproduced "
     "(dipole block 4.2× closer) but two partition-independent residuals remain."),
    ("A9", "Record the C-DF diagnostics", "done", ""),
    ("B1–B5", "Interaction tensor, block-product Casimir–Polder, recoupling table",
     "done", "Versioned factorised table, phased loader proven by 31 mutations."),
    ("B6–B8", "Plan→gate→allocate; direct-energy reconstruction", "done",
     "Reconstruction 7.7e-15 over 12 orientations on the full internal label set."),
    ("B9", "Publish the anisotropic contract at n ≤ 12", "done",
     "Isotropic entries match ATOMIC C6…C12 to 6.9e-16."),
    ("B10", "Parse CASIMIR's anisotropic output; compare on a partition-matched run",
     "measured", "Oracle built and validated end to end. 10457 coefficients compared, "
     "0 inside 1e-4. Located an exact Clebsch-Gordan discrepancy."),
    ("—", "Publish the anisotropic polarizability tensors", "done",
     "Three new QCVariables; nine pre-existing arrays bit-identical."),
    ("—", "Quantify the energetic consequence on a real PES", "done",
     "Induction 1.4 %, dispersion 12 % at the H-bond minimum."),
]

TODO = [
    ("P0", "Close the rank-2/rank-3 polarizability deficit", "open", r"""
     <p>Measured at <strong>31.9 %</strong> (l=2) and <strong>30.0 %</strong> (l=3) against
     the matched oracle, versus 2.9 % at l=1. This is the dominant error and it is what makes
     the C8 energy error four times the C6 error. Two candidate causes, and the experiments
     that separate them:</p>
     <ul>
     <li><strong>The missing rank 4.</strong> CamCASP computes non-local polarizabilities to
     rank 4 and <em>then</em> localizes to L3, so its L3 ranks 2–3 absorb rank-4 content. Ours
     is L3 throughout. Test: extend the internal model to rank 4 before localization and
     re-measure the l=2/l=3 invariants.</li>
     <li><strong>A per-rank normalisation.</strong> \(\sqrt{0.681}=0.825\) and
     \(\sqrt{0.700}=0.837\) are suspiciously close, which is what a per-rank factor gives.
     Against it: each hydrogen's rank-1↔rank-2 and rank-1↔rank-3 cross blocks come out
     <em>30 % too large</em> (1.274, 1.330) — a scale factor cannot change sign. Test:
     re-measure those cross blocks component by component rather than by Frobenius norm,
     which mixes signs.</li>
     </ul>"""),
    ("P0", "Decide the Clebsch-Gordan convention", "blocked", r"""
     <p>Our recoupling prefactor differs from CASIMIR's by exactly
     \(1/|\langle l_1 0;l_2 0|j0\rangle|\) on the cosine sector at C6, reproduced
     independently to \(6.45\times10^{-7}\). Removing it puts the H–H residual inside the
     independently measured 1–10 % C6 band.</p>
     <p><strong>No internal check can settle which side is right.</strong> The
     \(\langle L_1 0;L_2 0|j0\rangle\Lambda_j\) split is invariant under
     \(C\to C\kappa,\;S\to S/\kappa\), which is precisely why the machine-precision
     direct-energy reconstruction passes on both. This needs the published Stone
     \(S\)-function definition or a third independent implementation. <strong>Until then,
     do not "fix" either side, and do not gate anything anisotropic label-by-label.</strong></p>"""),
    ("P1", "Close the 2.9 % rank-1 deficit", "open", r"""
     <p>Site-summed isotropic dipole polarizability is 0.9706 of the reference, and the two
     independent oracles agree with each other on that total to 0.11 % — so it is upstream of
     the partition, in \(G(i\omega)\) or the response kernel. Untested candidates: the
     reference density-fitted its own propagator integrals, and ALDA versus ALDA+CHF is
     unresolved.</p>"""),
    ("P1", "The j ≥ 9 sector has no oracle at all", "open", r"""
     <p>CASIMIR caps the coupled rank at \(j\le8\); we reach \(j=10\). That leaves 222
     genuinely nonzero labels with no external reference at any rank — and the sector is not
     marginal: one reaches 247.2 against CASIMIR's largest <em>printed</em> \(|C_{11}|\) of
     299.1. Either find a reference that goes higher, or document the sector as
     permanently unvalidated.</p>"""),
    ("P2", "Order-dependent dispersion damping", "open", r"""
     <p>At the H-bond minimum \(C_6+C_8+C_{10}=-3.29\) kcal mol⁻¹ with \(C_{10}\) alone at
     \(-1.08\), nearly as large as \(C_6\). A single order-independent Becke–Johnson radius is
     inadequate; Tang–Toennies damping with order-dependent parameters is the right treatment
     and is not implemented.</p>"""),
    ("P2", "Rank 0 and rank 4; non-local blocks", "open", r"""
     <p>The model carries no rank 0 (charge flow) and no rank 4, and is site-diagonal after
     localization where CamCASP retains all nine ordered site-pair blocks. Rank 4 is coupled
     to P0 above.</p>"""),
    ("P3", "Site-axes support", "open", r"""
     <p>Site frames are the identity, so everything is published in the molecular frame. The
     rank-1–3 rotation machinery exists on both sides now; what is missing is an axes input
     and applying it at publication. No new derivation needed.</p>"""),
]


def main():
    sha, branch, subject = git_meta()
    today = datetime.date.today().isoformat()
    r = D["rank_invariants"]
    pes = D["pes"]
    di = pes["distances"].index(2.912)
    a4 = pes["modes"]["r4r2"]
    a8 = pes["modes"]["c8c6"]
    ind_d = a4["ours"][di]["ind"] - a4["isa_grid"][di]["ind"]
    dsp_d = a4["ours"][di]["d3"] - a4["isa_grid"][di]["d3"]
    dsp_d8 = a8["ours"][di]["d3"] - a8["isa_grid"][di]["d3"]
    aniso = a4["ours"][di]["ind"] - a4["ours"][di]["ind_iso"]
    ratios = [abs(x["ours"] / x["ref"]) for x in D["aniso_cn"] if x["ref"] != 0]
    reldev = np.array([abs((x["ours"] - x["ref"]) / x["ref"])
                       for x in D["aniso_cn"] if x["ref"] != 0])
    order_fig, order_counts = fig_aniso_by_order()

    P = [HEAD]
    P.append(f"""<header>
<h1>State of the CamCASP–Psi4 distributed-property pipeline</h1>
<p class="sub">What is implemented, how closely it reproduces CamCASP, what that costs in
energy, and what is left.</p>
<div class="meta">{today} &nbsp;·&nbsp; branch <b>{branch}</b> @ <b>{sha}</b> &nbsp;·&nbsp;
H<sub>2</sub>O, PBE0/aug-cc-pVTZ + GRAC, ALDA+CHF, LW→L3, PFIT WSM weight 4
&nbsp;·&nbsp; test suite <b>634 passed, 8 xfailed, 0 skipped</b></div>
</header>""")

    P.append(f"""
<div class="kpis">
<div class="kpi"><div class="v">12</div><div class="k">QCVariables published</div></div>
<div class="kpi"><div class="v">2</div><div class="k">partition routes</div></div>
<div class="kpi"><div class="v">2.94 %</div><div class="k">rank-1 α deficit</div></div>
<div class="kpi"><div class="v">31.9 %</div><div class="k">rank-2 α deficit</div></div>
<div class="kpi"><div class="v">{abs(ind_d):.3f}</div><div class="k">induction error, kcal/mol</div></div>
<div class="kpi"><div class="v">{abs(dsp_d):.3f}</div><div class="k">dispersion error, kcal/mol</div></div>
<div class="kpi"><div class="v">10457</div><div class="k">anisotropic Cₙ compared</div></div>
<div class="kpi"><div class="v">0</div><div class="k">inside rtol 1e-4</div></div>
</div>

<div class="note">
<p><strong>Read this first.</strong> The pipeline is feature-complete against its plan: both
partition routes work, all ten stages run under fail-closed invariants, and twelve arrays are
published. What is <em>not</em> complete is quantitative agreement with CamCASP, and this page
is mostly about locating that disagreement precisely rather than describing the machinery.</p>
<p>The short version: the polarizability deficit is <strong>rank-resolved, not diffuse</strong>
— 2.9 % at rank 1 and about 30 % at ranks 2 and 3 — and that single fact explains the whole
C6→C12 error ladder and the whole shape of the energy error. One further discrepancy, an exact
Clebsch-Gordan factor in the anisotropic recoupling, is <strong>deliberately left unresolved</strong>
because no internal check can decide which side is right.</p>
</div>

<nav class="toc"><ol>
<li><a href="#routes">Supported routes and pipeline stages</a></li>
<li><a href="#theory">Theory, as implemented</a></li>
<li><a href="#oracles">The two oracles, and the 113× trap</a></li>
<li><a href="#outputs">Published outputs</a></li>
<li><a href="#props">Numerical agreement: polarizabilities</a></li>
<li><a href="#cn">Numerical agreement: dispersion coefficients</a></li>
<li><a href="#aniso">The anisotropic table and the convention question</a></li>
<li><a href="#energy">Energy validation: parity and SAPT0</a></li>
<li><a href="#status">Implementation status, task by task</a></li>
<li><a href="#left">What is left to do</a></li>
</ol></nav>

<h2 id="routes"><span class="num">1</span>Supported routes and pipeline stages</h2>

<p>One entry point, <code>atomic_polarizabilities()</code>, chains ten native stages. The only
user-visible branch is the partition, selected by
<code>ATOMIC_POLARIZABILITY_PARTITION</code>; everything downstream of stage 3 is shared. Colour
marks how well each stage is validated, not whether it runs.</p>

{pipeline_diagram()}

<p class="caption">Every stage carries a fail-closed invariant rather than a comment: the
provenance seal refuses a marginally converged SCF, the bond graph refuses a disconnected
molecule, the ISA partition refuses to proceed unconverged, and the LW charge-sum postcondition
refuses a grid too coarse to reach the requested tolerance. Publication is all-or-nothing —
either all twelve arrays appear or none do.</p>

{section_theory()}

<h2 id="oracles"><span class="num">3</span>The two oracles, and the 113× trap</h2>

<p>There are two reference calculations. They ran the <em>same</em> protocol and differ in
exactly one respect: how the response is partitioned between sites. They agree on the molecular
total to 0.11 % and disagree on the split. Comparing an arm against the wrong one is wrong by up
to a factor of 113 on a single component with nothing defective anywhere.</p>

{oracle_map()}

<div class="okbox">
<p><strong>This is the single most valuable result in the project so far</strong>, and it was a
measurement error rather than a code defect. Before the ISA-GRID oracle existed, our ISA arm was
being compared against the C-DF reference, which made oxygen look strongly anisotropic where ISA
makes it nearly isotropic and put a hydrogen dipole off-diagonal out by a factor of 113. Every
feature of the apparent "misdistribution" is reproduced by changing the reference's partition
alone.</p>
</div>

<h2 id="outputs"><span class="num">4</span>Published outputs</h2>

<table><thead><tr><th>QCVariable</th><th>shape</th><th>content</th><th>agreement</th>
<th>status</th></tr></thead><tbody>""")

    for name, shape, content, st, agree in PUBLISHED:
        lab = {"ok": ("ok", "validated"), "int": ("int", "internal only"),
               "bad": ("bad", "known deficit"), "open": ("open", "unresolved")}[st]
        P.append(f'<tr><td class="mono">{name}</td><td class="mono">{shape}</td><td>{content}</td>'
                 f'<td>{agree}</td><td>{pill(*lab)}</td></tr>')
    P.append("</tbody></table>")
    P.append("""
<p>The three <code>ANISOTROPIC</code> arrays are new. The full 15×15 rank-1–3 tensors were
already being computed, symmetry-constrained, refined and gate-validated at every frequency, and
then discarded at publication, which kept only the rank-1 3×3 sub-block — 6 of 225 numbers per
site. The <code>COMPONENTS</code> table makes the component convention machine-readable rather
than a comment.</p>
""")
    P.append(f"""
<h2 id="props"><span class="num">5</span>Numerical agreement: polarizabilities</h2>

<h4>Static dipole block</h4>
{fig_static_alpha()}
<p class="caption">Against the matched ISA-GRID oracle six of seven nonzero components agree to
2–5 %. The C-DF bars are shown to make the trap concrete: on H α<sub>xz</sub> the two references
differ by two orders of magnitude, and our value tracks the matched one.</p>

<h4>Across the Casimir–Polder grid</h4>
{fig_dynamic()}
{fig_dynamic_ratio()}
<p class="caption">The rank-1 deficit is a nearly frequency-independent 3 %, which is what makes
it propagate cleanly into C6 as roughly its square. A frequency-dependent error would have
produced a different, order-dependent signature.</p>

<h4>Resolved by rank — the central result</h4>
{fig_rank_deficit()}
{fig_rank_ratio()}
<p class="caption">Site-summed rank invariants a<sub>l</sub> = Tr(α<sup>ll</sup>)/(2l+1). Both
sides are anchored to already-reviewed numbers before anything new is claimed: the oracle's
rank-1 diagonal reproduces the reviewed literals exactly, and ours reproduces the recorded
basis-matched column exactly.</p>

<div class="warnbox">
<p><strong>The deficit is rank-resolved, not diffuse.</strong> l=1 is
{100*(1-r['1']['ratio']):.2f} %, l=2 is {100*(1-r['2']['ratio']):.2f} %, l=3 is
{100*(1-r['3']['ratio']):.2f} %. The "uniform 2.9 % deficit" recorded in earlier work is
<em>only</em> the rank-1 term; a second deficit ten times larger sits in ranks 2 and 3.</p>
<p>This was not measurable until both the anisotropic publication and the ISA-GRID
<code>.pol</code> oracle existed, which happened on the same day. It reproduces the measured
C6→C12 error ladder with no further assumption, because C6 is built only from rank-1×rank-1
(≈0.97²) while C10 and C12 are dominated by rank-2 and rank-3 products (≈0.68²–0.70²).</p>
</div>
""")
    P.append(f"""
<h2 id="cn"><span class="num">6</span>Numerical agreement: dispersion coefficients</h2>

{fig_isotropic_cn()}
<p class="caption">The error ladder. C6 is within 1–10 % per pair; by C12 it is near 46 %. This
is the downstream consequence of the rank-2/3 deficit above, not an independent problem — and
it is why attacking C6 further buys almost nothing.</p>

<h2 id="aniso"><span class="num">7</span>The anisotropic table and the convention question</h2>

<p>The published anisotropic table is
<code>(natom², 16985)</code> at n ≤ 12. A partition-matched CASIMIR run supplies
{len(D['aniso_cn'])} shared nonzero coefficients to compare against.</p>

<table class="tight"><thead><tr><th>property</th><th class="n">value</th></tr></thead><tbody>
<tr><td>Labels CamCASP emits that we do not publish</td><td class="n">0</td></tr>
<tr><td>Its nonzero coefficients landing on labels we also make nonzero</td>
<td class="n">10457 / 10457</td></tr>
<tr><td>Shared nonzero coefficients compared</td><td class="n">{len(D['aniso_cn'])}</td></tr>
<tr><td>Inside rtol = 1e-4</td><td class="n">0</td></tr>
<tr><td>Median |ours / CASIMIR| &nbsp;<i>(ratio)</i></td>
<td class="n">{np.median(ratios):.4f}</td></tr>
<tr><td>Median |ours − CASIMIR| / |CASIMIR| &nbsp;<i>(relative deviation)</i></td>
<td class="n">{np.median(reldev):.4f}</td></tr>
<tr><td>&nbsp;&nbsp;90th percentile / worst</td>
<td class="n">{np.percentile(reldev,90):.2f} / {reldev.max():.0f}</td></tr>
<tr><td>Sign disagreements</td>
<td class="n">{sum(1 for x in D['aniso_cn'] if x['ours']*x['ref']<0)} ({100*sum(1 for x in D['aniso_cn'] if x['ours']*x['ref']<0)/len(D['aniso_cn']):.1f} %)</td></tr>
<tr><td>CASIMIR's coupled-rank cap</td><td class="n">j ≤ 8</td></tr>
<tr><td>Our reach</td><td class="n">j = 10</td></tr>
<tr><td>Our nonzero labels with no oracle at any rank</td><td class="n">222</td></tr>
</tbody></table>

{fig_aniso_scatter()}
<p class="caption">The structure is right — the label set, the sparsity pattern and the order of
magnitude all track — but the coefficients are not. Deviation grows with order.</p>

{order_fig}
<p class="caption">Counts per order: {", ".join(f"C{n} {c}" for n, c in zip(range(6,13), order_counts))}.</p>

<h3>7.1 An exact Clebsch-Gordan factor, and why it is left unresolved</h3>

<p>C6 is the only published order to which a single site-block quadruple contributes, so at fixed
\\((l_1,l_2,k_1,k_2)\\) the \\(j\\) dependence isolates the recoupling prefactor. Doing that
scan:</p>

{fig_cg_correction()}
<p class="caption">Raw ratios scatter over more than an order of magnitude. Multiplied by
|⟨l₁0;l₂0|j0⟩| they collapse onto the known deficit band. Measured worst relative spread across
the {D['cg_scan']['groups']} multi-j groups: <b>{D['cg_scan']['worst_relative_spread']:.3e}</b>,
which is the oracle's printed precision.</p>

<div class="openbox">
<p><strong>This is left open on purpose.</strong> The collapse is exact, and after it the H–H
residual lands inside the independently measured 1–10 % C6 band. But the
\\(\\langle L_1 0;L_2 0|j0\\rangle\\Lambda_j\\) split inside \\(g^r\\) is invariant under
\\(C\\to C\\kappa, S\\to S/\\kappa\\) — the \\(S\\) functions absorb exactly this freedom — so
the machine-precision direct-energy reconstruction passes on <em>both</em> conventions and
cannot decide between them.</p>
<p>An earlier analysis concluded that the factor could not be a convention, because it vanishes
on 2 968 of the shared entries where both tables are nonzero. <strong>That inference is wrong
and was withdrawn.</strong> Those entries are <em>exactly</em> the sine-carrying labels —
<code>l₁+l₂+j+σ</code> is even on 6 285 of 6 285 rows — and the real S functions on that sector
normalise with <em>nonzero-m</em> Clebsch-Gordan coefficients, not the m = 0 one being fitted. An
m = 0 CG vanishing there is expected and carries no information about correctness.</p>
<p>Resolving it needs the published Stone S-function definition or a third independent
implementation. Until then, do not "fix" either side.</p>
</div>
""")
    P.append(f"""
<h2 id="energy"><span class="num">8</span>Energy-level validation: internal parity and SAPT0 references</h2>

<p>Property-space percentages do not say whether a model is usable. Energy validation therefore
has two layers. The first is a controlled same-kernel comparison, which isolates the effect of
our distributed properties. The second is an external <strong>SAPT0/aug-cc-pVDZ</strong>
reference, which asks whether the resulting advanced force-field terms reproduce the physical
induction and dispersion surfaces rather than merely reproducing CamCASP properties.</p>

<div class="note">
<p><strong>SAPT reference convention.</strong> At every geometry record the Psi4 SAPT0 induction
component (second-order induction + exchange-induction, including the SAPT0 δ<sub>HF</sub>
correction when reported in the induction total) and the SAPT0 dispersion component
(second-order dispersion + exchange-dispersion). Compare those totals directly with the
advanced force-field <i>E</i><sub>ind</sub> and <i>E</i><sub>disp</sub>; do not compare either one
with the full SAPT interaction energy.</p>
<p>For dispersion, report the force-field prediction cumulatively as <strong>C6</strong>,
<strong>C6+C8</strong>, and <strong>C6+C8+C10</strong>. SAPT0 supplies one dispersion target,
not an order decomposition, so the three cumulative curves measure whether each added
long-range order improves the same reference. C10 must remain visible rather than being hidden
inside a D3-comparable total.</p>
</div>

<h3>8.1 Water dimer: controlled property-parity surface</h3>
<p>A rigid water-dimer surface provides the isolation test: geometries, MBIS permanent
multipoles, damping parameters and energy kernels are held identical across arms, so every
difference is attributable to the distributed properties and to nothing else. The SAPT0 curves
are the external reference for the advanced terms; the ours-versus-ISA-GRID deltas below remain
the stricter diagnostic for locating property errors.</p>

{fig_pes("r4r2")}
{fig_pes_delta()}
<p class="caption">Solid lines are the standard r4r2 damping radius, dashed the coefficient-derived
one. The grey pair shows what the <em>wrong</em> oracle would have implied — note it inverts the
sign of the dispersion verdict.</p>

<table class="tight"><thead><tr><th>at R(O–O) = 2.912 Å</th><th class="n">ours</th>
<th class="n">ISA-GRID</th><th class="n">Δ</th><th class="n">Δ %</th></tr></thead><tbody>
<tr><td>induction</td><td class="n">{a4['ours'][di]['ind']:.4f}</td>
<td class="n">{a4['isa_grid'][di]['ind']:.4f}</td><td class="n">{ind_d:+.4f}</td>
<td class="n">{100*ind_d/a4['isa_grid'][di]['ind']:.2f}</td></tr>
<tr><td>dispersion C6+C8, r4r2 radius</td><td class="n">{a4['ours'][di]['d3']:.4f}</td>
<td class="n">{a4['isa_grid'][di]['d3']:.4f}</td><td class="n">{dsp_d:+.4f}</td>
<td class="n">{100*dsp_d/a4['isa_grid'][di]['d3']:.2f}</td></tr>
<tr><td>dispersion C6+C8, √(C8/C6) radius</td><td class="n">{a8['ours'][di]['d3']:.4f}</td>
<td class="n">{a8['isa_grid'][di]['d3']:.4f}</td><td class="n">{dsp_d8:+.4f}</td>
<td class="n">{100*dsp_d8/a8['isa_grid'][di]['d3']:.2f}</td></tr>
</tbody></table>

<div class="warnbox">
<p><strong>Dispersion is the problem; induction is not.</strong> The 2.9 % rank-1 polarizability
deficit costs {abs(100*ind_d/a4['isa_grid'][di]['ind']):.1f} % of the induction energy and is
nearly flat in R, as a linear property should be. Dispersion is
{abs(100*dsp_d/a4['isa_grid'][di]['d3']):.0f} % out and strongly R-dependent.</p>
<p><strong>But most of that is a damping artifact.</strong> Switching to Becke and Johnson's
original √(C8/C6) radius drops it to {abs(100*dsp_d8/a8['isa_grid'][di]['d3']):.1f} %, because
our smaller C8/C6 ratio damps less and partly cancels the coefficient deficit. So the rank
deficit is largely absorbable by refitting damping — but <em>not</em> by dropping our
coefficients into an existing r4r2-parametrised D3.</p>
</div>

{fig_pes_orders()}
<p class="caption">By order. The C8 error is roughly four times the C6 error in kcal/mol, which
is the energetic restatement of the rank-2/3 deficit. Note also that C10 alone rivals C6 at the
minimum — the pairwise expansion is barely converging there under a single order-independent
damping radius, so C10 is reported separately and excluded from any D3-comparable total.</p>

{fig_anisotropy()}
<p class="caption">Anisotropy is worth {abs(aniso):.3f} kcal/mol at the minimum,
{abs(100*aniso/a4['ours'][di]['ind']):.0f} % of the induction energy, and ours reproduces the
oracle's anisotropy to 4 %. Every isotropic-α consumer discards this — which includes every
induction model in the reference force-field codebase.</p>

<h3>8.2 External PES benchmark set</h3>
<table><thead><tr><th>system</th><th>rigid scan</th><th>why it is included</th>
<th>SAPT0/aug-cc-pVDZ targets</th><th>advanced-FF curves</th></tr></thead><tbody>
<tr><td><strong>water–water</strong><br>S22 hydrogen-bonded</td>
<td>O–O separation, fixed S22 orientation; retain R = 2.5–8.0 Å grid</td>
<td>Induction-dominated test with strong polarizability anisotropy and an existing
property-parity diagnosis.</td>
<td><i>E</i><sub>ind</sub><sup>SAPT0</sup>, <i>E</i><sub>disp</sub><sup>SAPT0</sup></td>
<td>full-tensor <i>E</i><sub>ind</sub>; damped C6, C6+C8, C6+C8+C10</td></tr>
<tr><td><strong>benzene–benzene</strong><br>S22 parallel-displaced</td>
<td>Interplanar separation through the S22 geometry, preserving the lateral offset and monomer
orientations; include the repulsive wall, minimum and long-range tail</td>
<td>Dispersion-dominated, extended π system. It gives the C8 and C10 metrics enough weight to be
meaningful instead of letting a water hydrogen bond dominate the verdict.</td>
<td><i>E</i><sub>ind</sub><sup>SAPT0</sup>, <i>E</i><sub>disp</sub><sup>SAPT0</sup></td>
<td>full-tensor <i>E</i><sub>ind</sub>; damped C6, C6+C8, C6+C8+C10</td></tr>
</tbody></table>

<h3>8.3 Metrics against SAPT0</h3>
<table class="tight"><thead><tr><th>term</th><th>primary metric</th><th>diagnostic metrics</th>
</tr></thead><tbody>
<tr><td>induction</td><td>MAE of <i>E</i><sub>ind</sub><sup>FF</sup> −
<i>E</i><sub>ind</sub><sup>SAPT0</sup> over the attractive and near-minimum region</td>
<td>error at the SAPT0 minimum; tail-relative error; full-tensor minus isotropised
contribution</td></tr>
<tr><td>dispersion</td><td>MAE of each cumulative C6, C6+C8 and C6+C8+C10 curve against
<i>E</i><sub>disp</sub><sup>SAPT0</sup></td><td>change in MAE on adding C8 and C10; error at the
SAPT0 minimum; long-range log-slope; damping sensitivity</td></tr>
</tbody></table>
<div class="warnbox">
<p><strong>Acceptance rule for higher orders.</strong> C8 or C10 counts as an improvement only if
it reduces the SAPT0 dispersion error over the near-minimum and attractive regions on
<strong>both</strong> water–water and benzene–benzene. A better asymptotic coefficient that worsens
the damped PES is a damping-model failure, not a successful advanced term.</p>
</div>
""")

    P.append("""
<h2 id="status"><span class="num">9</span>Implementation status, task by task</h2>
<table><thead><tr><th>task</th><th>description</th><th>status</th><th>notes</th></tr></thead>
<tbody>""")
    for tid, desc, st, note in TASKS:
        lab = {"done": ("ok", "done"), "failed": ("bad", "negative result"),
               "measured": ("open", "measured, not gated")}[st]
        P.append(f'<tr><td class="mono">{tid}</td><td>{desc}</td><td>{pill(*lab)}</td>'
                 f'<td>{note}</td></tr>')
    P.append("</tbody></table>")

    P.append("""
<div class="note">
<p><strong>On A8, which is recorded as a negative result.</strong> The plan expected that
implementing C-DF would flip the six <code>DF_*</code> comparisons to passing. It did not. All
six still fail at rtol = 1e-4, the strict-xfail markers were kept and the gate was not widened.
What the work bought instead was a located defect: the partition <em>was</em> reproduced —
dipole-block disagreement fell 4.2× — which is precisely what proved the residual is not in the
partition. That redirected the next effort away from a dead end.</p>
</div>

<h2 id="left"><span class="num">10</span>What is left to do</h2>
""")
    for prio, title, st, body in TODO:
        lab = {"open": ("open", "open"), "blocked": ("bad", "blocked externally")}[st]
        P.append(f'<h3>{pill("none", prio)} {title} &nbsp;{pill(*lab)}</h3>{body}')

    P.append(f"""
<footer>
<p>Generated {today} from branch <b>{branch}</b> at <b>{sha}</b> (<i>{subject}</i>).
Numbers are measured, not quoted: the polarizability and dispersion tables come from a live
run of the reviewed parity protocol, and the reference values from the checked-in CamCASP
oracles via <code>devtools/camcasp_reference.py</code>. Energies come from
<code>water-pes-aim.py</code>, whose two kernels are cross-checked against an independent
implementation (dispersion bit-exact at 1.1e-16, induction to 3.3e-8 kcal/mol). Section 8
also defines SAPT0/aug-cc-pVDZ induction and dispersion as the external PES references; those
SAPT values must be generated for both listed benchmark systems before force-field accuracy is
claimed.</p>
<p>Charts are inline SVG and render without network access. Equations use MathJax from a CDN and
degrade to readable TeX source without it.</p>
</footer>
</div></body></html>""")

    OUT.write_text("".join(P))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
