"""Login-safe import/API availability only. No factorization, JK, cuEST or device calls."""
import sys
from pathlib import Path

import fixed_jk as f
import fixed_jk_runner as runner
import observe as o
import policy as p


def main():
    arm, output = sys.argv[1:]
    output = Path(output)
    psi4, observed = runner.imported(arm,output.with_suffix(".import.json"))
    psi4.core.clean_options()
    psi4.set_options(f.OPTIONS)
    options = {key:psi4.core.get_option("SCF",key) for key in f.OPTIONS}
    p.require({k:v.lower() if isinstance(v,str) else v for k,v in options.items()} ==
              {k:v.lower() if isinstance(v,str) else v for k,v in f.OPTIONS.items()}, "effective host options")
    methods = ["partial_cholesky_factorize","from_array"]
    p.require(all(callable(getattr(psi4.core.Matrix,name,None)) for name in methods), "native factor API unavailable")
    o.atomic(output,dict(arm=arm,core_sha256=observed["core_sha256"],version=observed["psi4_version"],
        options=options,native_matrix_methods=methods,no_factorization=True,no_device_calls=True,
        no_JK_object=True,no_energy=True,not_GPU_correctness=True))


if __name__ == "__main__":
    main()
