# `DECLARED_MULTPOLE_AC`: the reference's own asymptotic correction, as its own gate

This is the third and last SCF asymptotic-correction policy admitted by
`isapol_native_correction.py::validate_correction`. It exists because the
reference case's SCF input declares an asymptotic correction that is **not**
Psi4's gradient-regulated GRAC, and the rule in force throughout this work is
that a different correction form is a different model: it gets its own gate, its
own declaration object and its own provenance type, and is never reached by
widening an existing policy or by refitting a GRAC β.

## What is declared

`examples/properties/H2O/output_1/H2O_aTZ_A.dal` — the reference's own input,
read as input data:

```
.DFTAC
MULTPOLE
TANH
0.46380 0.46380 3.0 4.0
.DFTELS
0.01
```

`docs/users_guide.tex:538` identifies the pieces, citing Tozer and Handy (1998):
a Fermi–Amaldi long-range exchange potential, spliced onto the DFA potential by a
`tanh` switch over **3 → 4 Bragg–Slater radii**, with the asymptotic branch
shifted so that `v_xc(∞) = I − |ε_HOMO|`. The two equal numbers are the
ionization potential in hartree, `12.62063 eV / 27.21136 = 0.46380`, which is the
same I.P. the `.clt` files declare.

## The form

```
v_xc^AC(r) = (1 − f(x))·v_xc^DFA(r) + f(x)·[ c_FA·v^FA(r) + Δ ]
v^FA(r)    = −V_H^mult(r)/N          (multipole expansion of the Hartree potential)
x          = min_A |r − R_A| / R_A^BS
Δ          = I + ε_HOMO              (the Tozer–Handy shift; `shift_mode='variational'`)
c_FA       = 1 − a_x = 0.75          for PBE0
```

Three things about how it is applied, each of which was a wrong turn first:

1. **Pointwise, not on the GGA integrand.** Splicing inside the exchange energy
   density leaves a `∇f` surface term worth +0.22 Eh. The correction is added as
   a local `f·[v^asym − v^{DFA,pw}]` contribution on top of Psi4's unchanged XC
   matrix.
2. **`f` is identically zero inside b1**, for every join form, not merely small.
   The multipole branch has a pole at its origin, so a small-but-nonzero weight in
   the core is not a small perturbation. `splice_weight` enforces exact zero and
   `test_isapol_declared_ac.py` asserts it with `== 0.`, not a tolerance.
3. **There is no energy functional.** The reported energy is the plain functional
   evaluated at the corrected density, so it must lie *above* the plain SCF
   minimum. The producer refuses a record that does not, because a lower value
   would mean a mislabel rather than a better answer.

## What fixed each field

The declaration has eight fields and only some of them are read off the input
file. The rest were fixed by a 14-row bracket scored against DALTON's **own
printed eigenvalue spectrum** for this case (occupied −19.21343344, −1.03775901,
−0.54409489, −0.40707262, −0.33059656; virtual 0.01058160, 0.06554206,
0.11823923, 0.13153098, 0.13988589; `Final DFT energy −76.379643934519`), never
against a polarizability:

| varied field | r.m.s. vs printed spectrum (Eh) |
| --- | --- |
| uncorrected PBE0 | 0.020674 |
| **the declared form** | **0.003236** |
| `fa_scale = 1.00` instead of 0.75 | 0.028990 |
| no Tozer–Handy shift | 0.078100 |
| `bragg_table='camcasp'` | 0.004300 |
| `b = 3.5–4.7` (the He2 case's numbers) | 0.005050 |
| `join='linear'` / `'tanh_raw'` | 0.003250 / 0.003380 |
| `tanh_k = 3` | 0.003200 |
| `multipole_order` 0 / 1 / 3 | 0.003320 / 0.003260 / 0.003250 |
| `origin='com'` | 0.003240 |

So `fa_scale`, the shift, the Bragg table and `b1/b2` are **determined** by the
reference's own printed numbers, and the join form, `tanh_k`, the multipole order
and the origin are **not** — they move the r.m.s. by ≤ 0.0002, which is why the
surviving 0.003236 cannot be attributed to any undetermined field. A fourteenth
row, `shift_mode='fixed'` at +0.13320, scores 0.002890, but its shift was computed
from the oracle's own HOMO and it is a diagnostic only, never a declaration.

0.003236 Eh is a 6.39× reduction and accounts for 84% of the +0.02196 gap error.
It is **not** a match, and this file does not claim one. `.DFTELS 0.01` has no
counterpart in our implementation at all.

## The gates

- `AcDeclaration` is frozen, hashable, and refuses every undeclared or impossible
  field in `__post_init__` (unknown join/table/origin/shift mode, non-integer or
  out-of-range multipole order, non-positive or non-finite I.P., `b1 ≥ b2`,
  `fa_scale` outside [0,1], …). Any changed field is a different declaration with
  a different `label()`.
- `splice_weight` refuses an element absent from the declared Bragg table —
  "no fallback radius is substituted".
- `declared_ac_orbitals(wfn, declaration)` **verifies the uncorrected SCF seal
  and manufactures none**, refuses every bad argument before entering the
  iteration, returns nothing unless the iteration converged inside thresholds at
  least as tight as `1e-8`/`1e-6`, and **applies nothing**: the wavefunction's
  `_scf_state_signature` is bit-identical afterwards.
- `apply_declared_ac(wfn, record)` is the explicit, named mutation. It refuses a
  record from a different SCF state, a mismatched basis/occupation, and
  composition with an existing correction. It deliberately invalidates the SCF
  seal, which is correct: the state is no longer the one Psi4's SCF converged.
- `validate_declared_ac` re-derives the provenance from a *current* record and
  refuses a stale signature or a doctored convergence record (iterations,
  clamping, thresholds, sub-minimum energy, or a shift off its own fixed point).
- `NONE` and `FIXED_GRAC` both **refuse** a wavefunction carrying
  `_declared_ac_evidence`, so AC orbitals can never be reported as "no SCF
  asymptotic correction". This closes the one mislabel hazard the policy creates.
- The response factory requires the declaration to be passed explicitly and
  refuses `kernel='no_local'`: there is no asymptotic-correction kernel
  derivative, exactly as for `FIXED_GRAC`. Its convergence evidence reads
  "verified current declared asymptotic-correction record, **not an SCF seal**".
- The C++ option `ATOMIC_SCF_ASYMPTOTIC_CORRECTION` is still
  `"NONE FIXED_GRAC"`. It was not widened: a string option cannot carry an
  `AcDeclaration`, so the policy is reachable only through the explicit Python
  `ac_declaration` argument. `test_isapol_declared_ac.py` asserts this
  non-widening, and asserts by AST that the module never calls a GRAC setter, an
  option setter, a LibXC tweak or the `psi4.energy` driver.

## Reproducible anchors

PBE0/cc-pVDZ at the reference geometry (the test's own anchor, not a reference
value): 17 iterations, `shift 0.164581425171`, `homo −0.299218530996`,
`lumo 0.078224857057`, `E −76.338733436310` above the plain
`−76.338758964954`, `D − C_occ C_occ^T` exactly 0, `FDS − SDF` 1.5e-14.

PBE0/aug-cc-pVTZ, the reference case: 18 iterations in 15.0 s,
`shift 0.132531307327`, `homo −0.331268635758`, `lumo 0.006385306066`,
`gap 0.337653941824`, `E −76.379670832097` above `−76.379694207042`,
171396 grid points.

## What it does and does not close

Refined end to end on the reference's declared lattice, model, target and
anchors, ρ moves from 0.210440 (`NONE`) to **0.211698** against the refined
reference 0.213354 — a residual of −0.776% where `NONE` gives −1.366%, so 43% of
it is closed. The static molecular α excess falls from +4.147% to +2.144%, and
at the parameter level `max |our refined O1 diagonal − the reference's own printed
refined O1 diagonal|` falls from 2.7913 to 1.4913, 46.6% removed.

It does not close the residual, and SPEC.md records where the rest is: about 0.5
percentage points is the eigenvalue deficit the AC still leaves, and about 1.5
points is not an asymptotic-correction effect at all but the uncorrected
DALTON-versus-Psi4 PBE0 difference (a near-uniform +0.0021 Eh offset on every
occupied eigenvalue at ΔE = −5.03e-05) plus the uncounterparted `.DFTELS 0.01`.

`NONE`, `FIXED_GRAC` and `DECLARED_MULTPOLE_AC` are three separately declared
models. **No two of their rows may be quoted as agreeing with one another**, and
the first two remain the bracket they always were rather than becoming members of
a series with the third.
