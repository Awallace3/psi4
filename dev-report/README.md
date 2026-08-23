# Generator for `state-of-camcasp-psi4.html`

**Development-only. Drop this directory and the generated HTML before any upstream PR.**

Regenerate:

```bash
# 1. capture the published arrays from a live reviewed-protocol run (~10 min)
cd dev-report
PYTHONPATH=../build_camcasp/stage/lib PSI_SCRATCH=$PWD \
  ~/miniconda3/envs/p4_camcasp/bin/python3.13 -P ours_cn.py

# 2. generate the SAPT0/aug-cc-pVDZ water PES reference (parallel example)
export PYTHONPATH=../build_camcasp/stage/lib PSI_SCRATCH=/tmp
printf '%s\n' 2.5 2.6 2.7 2.8 2.912 3.0 3.1 3.2 3.4 3.6 3.8 4.0 4.5 5.0 5.5 6.0 7.0 8.0 | \
  xargs -P6 -I{} ~/miniconda3/envs/p4_camcasp/bin/python3.13 -P sapt0_pes.py --distance {}
~/miniconda3/envs/p4_camcasp/bin/python3.13 -P sapt0_pes.py --combine

# 3. collect every number the page reports
PYTHONPATH=.. ~/miniconda3/envs/p4_camcasp/bin/python3.13 gather.py

# 4. build the page
PYTHONPATH=.. ~/miniconda3/envs/p4_camcasp/bin/python3.13 gen_html.py
```

`ours_cn.py` is not checked in because it is three lines of protocol plus a dump; the
protocol it must use is `PARITY_PROTOCOL` in `tests/pytests/test_atomic_polarizabilities.py`,
copied verbatim. Anything cheaper is a different model and its deltas mean nothing.

- `svgplot.py` — dependency-free inline SVG charts, so the page renders with no network.
- `sapt0_pes.py` — computes the 18-point SAPT0/aug-cc-pVDZ water induction and dispersion
  references in `sapt0-water.tsv`.
- `gather.py` — reads the live run, SAPT0 PES, and checked-in CamCASP oracles through
  `devtools/camcasp_reference.py`, then writes `report-data.json`.
- `gen_html.py` — prose, equations, route diagrams and figures.

Every number on the page is measured by this chain. Nothing is transcribed by hand.
