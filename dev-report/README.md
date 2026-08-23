# Generator for `state-of-camcasp-psi4.html`

**Development-only. Drop this directory and the generated HTML before any upstream PR.**

Regenerate:

```bash
# 1. capture the published arrays from a live reviewed-protocol run (~10 min)
cd dev-report
PYTHONPATH=../build_camcasp/stage/lib PSI_SCRATCH=$PWD \
  ~/miniconda3/envs/p4_camcasp/bin/python3.13 -P ours_cn.py

# 2. collect every number the page reports
PYTHONPATH=.. ~/miniconda3/envs/p4_camcasp/bin/python3.13 gather.py

# 3. build the page
PYTHONPATH=.. ~/miniconda3/envs/p4_camcasp/bin/python3.13 gen_html.py
```

`ours_cn.py` is not checked in because it is three lines of protocol plus a dump; the
protocol it must use is `PARITY_PROTOCOL` in `tests/pytests/test_atomic_polarizabilities.py`,
copied verbatim. Anything cheaper is a different model and its deltas mean nothing.

- `svgplot.py` — dependency-free inline SVG charts, so the page renders with no network.
- `gather.py` — reads the live run plus the checked-in CamCASP oracles through
  `devtools/camcasp_reference.py` and writes `report-data.json`.
- `gen_html.py` — prose, equations, route diagrams and figures.

Every number on the page is measured by this chain. Nothing is transcribed by hand.
