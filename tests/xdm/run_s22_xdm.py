"""
S22 benchmark: B3LYP-XDM/aug-cc-pVTZ binding energies vs S22B reference values.

Computes dimer and monomer energies for all 22 systems, calculates
interaction energies (counterpoise-uncorrected), and compares to
CCSD(T)/CBS reference values from Marshall et al. JCP 135, 194102 (2011).
"""

import sys
import os
import time

import psi4
import numpy as np

psi4.set_memory('4 GB')
psi4.set_num_threads(4)
psi4.core.set_output_file('s22_xdm_benchmark.out', False)

# S22 systems: (name, dimer_file, mono1_file, mono2_file, ref_kcal)
# Parsed from s22.din
S22_SYSTEMS = [
    ("NH3...NH3",                  "nh3_nh3",                  "nh3_nh3_1",                  "nh3_nh3_2",                   -3.133),
    ("H2O...H2O",                  "h2o_h2o",                  "h2o_h2o_1",                  "h2o_h2o_2",                   -4.989),
    ("HCOOH...HCOOH",              "h2co2_h2co2",              "h2co2_h2co2_1",              "h2co2_h2co2_2",              -18.753),
    ("Formamide...Formamide",      "formamide_formamide",      "formamide_formamide_1",      "formamide_formamide_2",      -16.062),
    ("Uracil...Uracil (HB)",       "uracil_uracil_hb",        "uracil_uracil_hb_1",         "uracil_uracil_hb_2",        -20.641),
    ("Pyridoxine...Aminopyridine", "pyridoxine_aminopyridine", "pyridoxine_aminopyridine_1", "pyridoxine_aminopyridine_2", -16.934),
    ("A...T (WC)",                 "adenine_thymine_wcc1",     "adenine_thymine_wcc1_1",     "adenine_thymine_wcc1_2",     -16.660),
    ("CH4...CH4",                  "ch4_ch4",                  "ch4_ch4_1",                  "ch4_ch4_2",                   -0.527),
    ("C2H4...C2H4",               "c2h4_c2h4",                "c2h4_c2h4_1",                "c2h4_c2h4_2",                 -1.472),
    ("C6H6...CH4",                 "c6h6_ch4",                 "c6h6_ch4_1",                 "c6h6_ch4_2",                  -1.448),
    ("C6H6...C6H6 (PD)",          "c6h6_c6h6_pd",             "c6h6_c6h6_pd_1",             "c6h6_c6h6_pd_2",             -2.654),
    ("Pyrazine...Pyrazine",        "pyrazine_pyrazine",        "pyrazine_pyrazine_1",        "pyrazine_pyrazine_2",         -4.255),
    ("Uracil...Uracil (stack)",    "uracil_uracil_stack",      "uracil_uracil_stack_1",      "uracil_uracil_stack_2",      -9.805),
    ("Indole...C6H6 (stack)",      "indole_c6h6_stack",        "indole_c6h6_stack_1",        "indole_c6h6_stack_2",         -4.524),
    ("A...T (stack)",              "adenine_thymine_stack",     "adenine_thymine_stack_1",     "adenine_thymine_stack_2",    -11.730),
    ("C2H4...C2H2",               "c2h4_c2h2",                "c2h4_c2h2_1",                "c2h4_c2h2_2",                 -1.496),
    ("C6H6...H2O",                "c6h6_h2o",                  "c6h6_h2o_1",                  "c6h6_h2o_2",                 -3.275),
    ("C6H6...NH3",                "c6h6_nh3",                  "c6h6_nh3_1",                  "c6h6_nh3_2",                 -2.312),
    ("C6H6...HCN",                "c6h6_hcn",                  "c6h6_hcn_1",                  "c6h6_hcn_2",                 -4.541),
    ("C6H6...C6H6 (T)",           "c6h6_c6h6_t",              "c6h6_c6h6_t_1",              "c6h6_c6h6_t_2",              -2.717),
    ("Indole...C6H6 (T)",         "indole_c6h6_t",             "indole_c6h6_t_1",             "indole_c6h6_t_2",            -5.627),
    ("Phenol...Phenol",            "phenol_phenol",             "phenol_phenol_1",             "phenol_phenol_2",            -7.097),
]

STRUCT_DIR = "/Users/albd/research/psi4/refdata/20_s22"
HA_TO_KCAL = 627.5094740631


def read_xyz_to_psi4(filepath):
    """Read an xyz file and return a psi4 geometry string."""
    with open(filepath) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    charge_mult = lines[1].strip()  # "0 1" typically
    coords = []
    for i in range(2, 2 + natom):
        coords.append(lines[i].strip())
    geom_str = f"{charge_mult}\n" + "\n".join(coords) + "\nsymmetry c1\nno_reorient\nno_com\n"
    return geom_str


def compute_energy(geom_str, label):
    """Run B3LYP-XDM/aug-cc-pVTZ and return total energy."""
    mol = psi4.geometry(geom_str)
    psi4.set_options({
        'basis': 'aug-cc-pVTZ',
        'scf_type': 'df',
        'd_convergence': 1e-8,
        'e_convergence': 1e-10,
    })
    e = psi4.energy('B3LYP-XDM')
    e_xdm = psi4.variable('DISPERSION CORRECTION ENERGY')
    psi4.core.clean()
    psi4.core.clean_variables()
    print(f"  {label}: E = {e:.10f} Eh, XDM disp = {e_xdm:.10f} Eh", flush=True)
    return e, e_xdm


# Run all S22 systems
print("=" * 80)
print("S22 Benchmark: B3LYP-XDM/aug-cc-pVTZ")
print("=" * 80)
print(flush=True)

results = []
t_start = time.time()

for idx, (name, dimer, mono1, mono2, ref) in enumerate(S22_SYSTEMS, 1):
    print(f"\n[{idx:2d}/22] {name}")
    print("-" * 60, flush=True)

    dimer_xyz = os.path.join(STRUCT_DIR, dimer + ".xyz")
    mono1_xyz = os.path.join(STRUCT_DIR, mono1 + ".xyz")
    mono2_xyz = os.path.join(STRUCT_DIR, mono2 + ".xyz")

    try:
        geom_d = read_xyz_to_psi4(dimer_xyz)
        geom_m1 = read_xyz_to_psi4(mono1_xyz)
        geom_m2 = read_xyz_to_psi4(mono2_xyz)

        e_d, xdm_d = compute_energy(geom_d, "Dimer")
        e_m1, xdm_m1 = compute_energy(geom_m1, "Mono1")
        e_m2, xdm_m2 = compute_energy(geom_m2, "Mono2")

        ie = (e_d - e_m1 - e_m2) * HA_TO_KCAL
        ie_xdm = (xdm_d - xdm_m1 - xdm_m2) * HA_TO_KCAL
        error = ie - ref

        results.append({
            'idx': idx, 'name': name, 'ref': ref,
            'ie': ie, 'ie_xdm': ie_xdm, 'error': error,
            'e_d': e_d, 'e_m1': e_m1, 'e_m2': e_m2,
            'xdm_d': xdm_d, 'xdm_m1': xdm_m1, 'xdm_m2': xdm_m2,
        })
        print(f"  IE = {ie:.3f} kcal/mol | XDM contrib = {ie_xdm:.3f} kcal/mol | Ref = {ref:.3f} | Error = {error:+.3f}")

    except Exception as ex:
        print(f"  ERROR: {ex}")
        results.append({
            'idx': idx, 'name': name, 'ref': ref,
            'ie': None, 'ie_xdm': None, 'error': None,
            'e_d': None, 'e_m1': None, 'e_m2': None,
            'xdm_d': None, 'xdm_m1': None, 'xdm_m2': None,
        })

t_elapsed = time.time() - t_start

# Summary table
print("\n\n")
print("=" * 100)
print(f"{'S22 B3LYP-XDM/aug-cc-pVTZ Results Summary':^100}")
print("=" * 100)
print(f"{'#':>3}  {'System':<30}  {'Ref':>8}  {'B3LYP-XDM':>10}  {'XDM disp':>10}  {'Error':>8}  {'|Error|':>8}")
print("-" * 100)

errors = []
for r in results:
    if r['ie'] is not None:
        errors.append(r['error'])
        print(f"{r['idx']:3d}  {r['name']:<30}  {r['ref']:8.3f}  {r['ie']:10.3f}  {r['ie_xdm']:10.3f}  {r['error']:+8.3f}  {abs(r['error']):8.3f}")
    else:
        print(f"{r['idx']:3d}  {r['name']:<30}  {r['ref']:8.3f}  {'FAILED':>10}  {'':>10}  {'':>8}  {'':>8}")

print("-" * 100)
errors = np.array(errors)
print(f"{'':>3}  {'Mean Error (ME):':<30}  {'':>8}  {'':>10}  {'':>10}  {np.mean(errors):+8.3f}")
print(f"{'':>3}  {'Mean Abs Error (MAE):':<30}  {'':>8}  {'':>10}  {'':>10}  {'':>8}  {np.mean(np.abs(errors)):8.3f}")
print(f"{'':>3}  {'RMS Error (RMSE):':<30}  {'':>8}  {'':>10}  {'':>10}  {'':>8}  {np.sqrt(np.mean(errors**2)):8.3f}")
print(f"{'':>3}  {'Max Abs Error:':<30}  {'':>8}  {'':>10}  {'':>10}  {'':>8}  {np.max(np.abs(errors)):8.3f}")
print(f"{'':>3}  {'Std Dev:':<30}  {'':>8}  {'':>10}  {'':>10}  {'':>8}  {np.std(errors):8.3f}")
print("=" * 100)

# Breakdown by category
hb_indices = [0, 1, 2, 3, 4, 5, 6]  # hydrogen-bonded
disp_indices = [7, 8, 9, 10, 11, 12, 13, 14]  # dispersion-dominated
mixed_indices = [15, 16, 17, 18, 19, 20, 21]  # mixed

print(f"\n{'Category Breakdown':^60}")
print("-" * 60)
for cat_name, indices in [("Hydrogen-bonded (1-7)", hb_indices),
                          ("Dispersion-dominated (8-15)", disp_indices),
                          ("Mixed (16-22)", mixed_indices)]:
    cat_errors = [errors[i] for i in indices if i < len(errors)]
    if cat_errors:
        cat_errors = np.array(cat_errors)
        print(f"  {cat_name:<35}  ME={np.mean(cat_errors):+6.3f}  MAE={np.mean(np.abs(cat_errors)):6.3f}  RMSE={np.sqrt(np.mean(cat_errors**2)):6.3f}")

print(f"\nTotal elapsed time: {t_elapsed:.1f} s ({t_elapsed/60:.1f} min)")
print("\nAll values in kcal/mol. Reference: S22B from Marshall et al. JCP 135, 194102 (2011).")
print("Interaction energies are NOT counterpoise-corrected.")
