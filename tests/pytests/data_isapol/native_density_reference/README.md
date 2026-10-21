# Native water finite-density reference

Small, portable, native-only numerical fixture. No CamCASP arrays, MO data,
grid samples, fields, wavefunction or executable/runtime objects are included.

`water.npz` contains only float64 `J` (246x246), `b`, `q` and the rounded
intended finite-density reference. `water.json` binds the archive and each
payload, gives all246 intended110-digit coefficient strings, and records
the originating native run/audit artifact hashes. It also retains the old
direct/rank-one coefficient-error summaries without copying their vectors.
All provenance uses experiment/artifact names, not absolute scratch paths.

Derivation: subset of the existing fresh native `density_001` export
(`ccasp-water-density-forward-v2`, original NPZ SHA256
`b0513c096a39b58a6925950027a437c96d23edc87fbdeaf1daecb262f1bb56f3`).
Reference: intended `(J+1000qqT)c=b+10000q`, from the accepted80/110-digit
audit and independent160-digit residual/identity checks. This is convergence
evidence, not a formal interval certificate or a field/property benchmark.
No SCF, integral or field regeneration was involved.

Tests round the supplied110-digit strings directly to binary64, then compare
**all246 coefficient bit patterns**, including weak and zero entries.
No `allclose`, residual-only replacement, cutoff or density repair is used.
The new module is explicitly opt-in; existing helpers/defaults are unchanged.

Safe independent validation from the checkout, with an output path that
does **not** yet exist:

```bash
PYTHON=python3 bash tests/pytests/run_stable_density_offline.bash /tmp/fresh-density-review
```

Use the requested pinned interpreter via `PYTHON` when available. The runner
sets8GiB AS/core0, numerical threads1 and disables bytecode; it imports only
the actual numerical source, never Psi4/core/adcc/LibXC. It runs all analytic/
refusal tests before the portable water test and writes fresh logs/identities.
It refuses an existing output directory rather than reusing sealed evidence.
