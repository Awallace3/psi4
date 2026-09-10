# Phoenix cuEST GRAC timing and accuracy

Status: partial.
Wall time is the fresh-process `energy()` call, including backend initialization.
Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.

| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water | cc-pvdz | 24 | 24 | 48 | 3/3 | 17.87 [17.80–17.89] | 18.45 [18.41–18.55] | 0.97× | 5.843e-08 | PASS |
| water | aug-cc-pvdz | 41 | 41 | 82 | 3/3 | 22.89 [22.87–22.98] | 18.29 [18.16–18.32] | 1.25× | 5.051e-08 | PASS |
| benzene | cc-pvdz | 114 | 114 | 228 | 3/3 | 154.50 [153.45–154.66] | 50.67 [50.42–51.04] | 3.05× | 2.341e-06 | FAIL |
| benzene | aug-cc-pvdz | 192 | 192 | 384 | 3/3 | 389.06 [388.81–389.29] | 55.32 [54.99–55.78] | 7.03× | 2.077e-06 | FAIL |
| peptide | 6-31+g** | 125 | 125 | 250 | 3/3 | 202.34 [201.07–202.48] | 75.84 [75.78–76.21] | 2.67× | 4.754e-08 | PASS |
| nanotube | 6-31+g** | 56 | 492 | 548 | 1/2 | 1078.82 [1078.82–1078.82] | 113.32 [113.19–113.45] | 9.52× | 9.268e-08 | PASS |

Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.

Accuracy threshold: 1.0e-06 Eh for every component and paired repeat.

## Component accuracy

| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |
|---|---|---:|---:|---:|
| water / cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / cc-pvdz | SAPT ELST ENERGY | -0.012689512780 | -0.012689530381 | 1.760e-08 |
| water / cc-pvdz | SAPT EXCH ENERGY | 0.010830298358 | 0.010830257528 | 4.083e-08 |
| water / cc-pvdz | SAPT IND ENERGY | -0.002618063773 | -0.002618063773 | 3.607e-13 |
| water / cc-pvdz | SAPT TOTAL ENERGY | -0.008087035897 | -0.008087094328 | 5.843e-08 |
| water / aug-cc-pvdz | SAPT DISP ENERGY | -0.003609757702 | -0.003609757702 | 0.000e+00 |
| water / aug-cc-pvdz | SAPT ELST ENERGY | -0.011314630000 | -0.011314617647 | 1.235e-08 |
| water / aug-cc-pvdz | SAPT EXCH ENERGY | 0.010087438972 | 0.010087477131 | 3.816e-08 |
| water / aug-cc-pvdz | SAPT IND ENERGY | -0.002881098793 | -0.002881098793 | 1.429e-12 |
| water / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.007718047523 | -0.007717997012 | 5.051e-08 |
| benzene / cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / cc-pvdz | SAPT ELST ENERGY | -0.002987010440 | -0.002988513767 | 1.503e-06 |
| benzene / cc-pvdz | SAPT EXCH ENERGY | 0.011832611246 | 0.011834951820 | 2.341e-06 |
| benzene / cc-pvdz | SAPT IND ENERGY | -0.001350650879 | -0.001350650899 | 2.224e-11 |
| benzene / cc-pvdz | SAPT TOTAL ENERGY | -0.004962451275 | -0.004961614048 | 8.372e-07 |
| benzene / aug-cc-pvdz | SAPT DISP ENERGY | -0.012457401202 | -0.012457401202 | 0.000e+00 |
| benzene / aug-cc-pvdz | SAPT ELST ENERGY | -0.003435953786 | -0.003437433767 | 1.480e-06 |
| benzene / aug-cc-pvdz | SAPT EXCH ENERGY | 0.012199781315 | 0.012201858754 | 2.077e-06 |
| benzene / aug-cc-pvdz | SAPT IND ENERGY | -0.001451162542 | -0.001451161722 | 8.433e-10 |
| benzene / aug-cc-pvdz | SAPT TOTAL ENERGY | -0.005144736215 | -0.005144137936 | 5.983e-07 |
| peptide / 6-31+g** | SAPT DISP ENERGY | -0.007726268123 | -0.007726268123 | 0.000e+00 |
| peptide / 6-31+g** | SAPT ELST ENERGY | -0.015531989909 | -0.015532010979 | 2.134e-08 |
| peptide / 6-31+g** | SAPT EXCH ENERGY | 0.014757057567 | 0.014757031393 | 2.617e-08 |
| peptide / 6-31+g** | SAPT IND ENERGY | -0.004728319097 | -0.004728319113 | 1.744e-11 |
| peptide / 6-31+g** | SAPT TOTAL ENERGY | -0.013229519563 | -0.013229566801 | 4.754e-08 |
| nanotube / 6-31+g** | SAPT DISP ENERGY | -0.027897956630 | -0.027897956630 | 0.000e+00 |
| nanotube / 6-31+g** | SAPT ELST ENERGY | -0.024720720508 | -0.024720719056 | 1.272e-09 |
| nanotube / 6-31+g** | SAPT EXCH ENERGY | 0.057773743032 | 0.057773835831 | 9.268e-08 |
| nanotube / 6-31+g** | SAPT IND ENERGY | -0.006463905712 | -0.006463924514 | 1.882e-08 |
| nanotube / 6-31+g** | SAPT TOTAL ENERGY | -0.001308839817 | -0.001308764368 | 7.513e-08 |

## Failed or incomplete measurements

```json
[
  {
    "error": "missing result.json",
    "name": "nanotube-6-31+g**-cpu-2",
    "process_wall_s": 0.0,
    "returncode": 1
  },
  {
    "error": "missing result.json",
    "name": "nanotube-6-31+g**-cpu-3",
    "process_wall_s": 0.0,
    "returncode": 1
  },
  {
    "error": "missing result.json",
    "name": "nanotube-6-31+g**-gpu-3",
    "process_wall_s": 0.0,
    "returncode": 1
  }
]
```
