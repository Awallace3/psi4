import numpy as np, psi4, json
from psi4.driver.procrouting import atomic_polarizability as nd
PROTO = {"basis":"aug-cc-pvtz","scf_type":"pk","e_convergence":1e-10,"d_convergence":1e-9,
 "dft_spherical_points":590,"dft_radial_points":99,"dft_density_tolerance":1e-12,
 "atomic_polarizability_isa_radial_points":100,"atomic_polarizability_isa_angular_polar_points":24,
 "atomic_polarizability_isa_angular_azimuthal_points":32,
 "atomic_polarizability_localization_tolerance":1e-6}
psi4.core.clean_variables(); psi4.core.be_quiet()
psi4.set_options({"atomic_polarizability_partition":"ISA", **PROTO})
mol = psi4.geometry("""
0 1
O  0.00000000  0.0  0.00000000
H -1.45365196  0.0 -1.12168732
H  1.45365196  0.0 -1.12168732
symmetry c1
no_com
no_reorient
units bohr
""")
wfn = nd.atomic_polarizabilities(molecule=mol)
names = ("ATOMIC DISPERSION COEFFICIENTS","ATOMIC DISPERSION LABELS",
         "ATOMIC ANISOTROPIC POLARIZABILITIES","ATOMIC POLARIZABILITIES",
         "ATOMIC C6","ATOMIC C8","ATOMIC C10","ATOMIC C12",
         "ATOMIC POLARIZABILITY FREQUENCIES","ATOMIC DYNAMIC POLARIZABILITIES")
out = {n: np.asarray(wfn.array_variable(n)).tolist() for n in names}
json.dump(out, open("ours_full.json","w"))
print("coeff shape", np.asarray(out["ATOMIC DISPERSION COEFFICIENTS"]).shape)
print("label shape", np.asarray(out["ATOMIC DISPERSION LABELS"]).shape)
