# Local-axis recoupled dispersion: rank <= 3 first increment

## Scope and acceptance status

`IsaRecoupledModel(IsaAnisotropicModel)` transforms a supplied local tensor to
complex coupled components. `isa_recoupled_dispersion(a,b,cp_weights,max_order)`
then returns **C_n(t,u,J)** for n=6..12, including odd orders and J=9,10. These
are independently callable typed C++ operations with Python bindings, not a
relabeling of the existing orientation-resolved scalar contraction. There is no
energy, displacement, rotation to global axes, damping, or coincident-site ban.
The caller's local frames define the coefficients.

Parent build/staging and runtime validation PASSED: 31 focused tests, then
1741 combined ISA/FDDS regressions in24.74s after the copy-free preflight followup.
The archive test checks6285 exact rows,10457 nonzero values at rtol1e-6/atol0,
30791 written zero placeholders at abs<=1e-6, and omitted trailing fields at the
same threshold. A mutation test detects an erroneous visible omitted field.
Acceptance: **shipped rank <= 3 recoupled C6-C12 matches the archived L3 table to
write precision**. This is NOT full native SCF/PFIT/GRAC protocol parity. The
411 visible J9/10 rows are counted, not numerically certified by the archive.
Rank4 is rejected; rank3 C12 remains structurally partial.

Evidence: `.pi/audit/recoupled-parity-tests-v1.log` and
`recoupled-parity-regressions-v2.log`. The final integrated core also reproduces
all saved fresh-water tensors/Cn bitwise at one thread in29.90s/589516KiB RSS
(`native-water-post-recoupling-comparison.json`). Internal trusted read-only
access avoids electronic-tensor clones before resource preflight; public getters
still return owned copies.

## Inputs and ownership

The already validated owned `IsaAnisotropicModel` requires increasing explicit
ranks, complete blocks among those ranks, exactly reciprocal finite matrices,
proper explicit local-to-global frames, finite positions, unique site labels,
nonnegative increasing frequencies, and nonblank provenance. No symmetry repair
or response fitting is performed. Declared rank 4 is rejected even when its
entries are zero: upstream skips (3,4), (4,3), (4,4) initialization; that undefined
state is not a production convention. All nonempty subsets of {1,2,3} work;
compressed storage uses offsets in the declared ranks, never a global component
index as a matrix offset.

`IsaRecoupledModel.source` retains the complete supplied model, positions,
frames, labels, ranks, declaration and provenance. Tensor getters return fresh
copies; the C++ internal tensors cannot be mutated. `sites[site]` contains every
ordered declared rank pair. Each block identifies `la`, `lap`, inclusive
`first_component` / `last_component`, component labels, and flat **frequency-major**
complex `values`. `value(site,frequency,la,lap,t)` uses zero-based site/frequency
and one-based global component t, and returns zero for structural absence.
Indices outside the supported domain throw.

## Numerical contract

Real Racah, no Condon--Shortley phase: rank 1 is **z,x,y**. Public component indices
are t(l,0)=l*l+1, t(l,mc)=l*l+2*m, t(l,ms)=l*l+2*m+1.

The 1101 shipped nonzero first-stage numerical records cover all nine ordered
rank pairs <=3. A record `(la,lap,k,q,v,p,denominator,r,s)` has zero-based
rank-local k/q and global coupled v. Its value is `(p/denominator)*sqrt(r/s)`
for r>0, or `i*(p/denominator)*sqrt(-r/s)` for r<0. Thus

    ac(f;l,lp;t) = sum_kq G(l,lp,k,q,t-1) A(f;offset(l)+k,offset(lp)+q)

for `(l-lp)^2+1 <= t <= (l+lp+1)^2`. Addition order is t, then k, then q,
independently for each frequency, matching casimir's component loops.

For each shipped second-stage block (n,L1,L2,J), every t/u in those ranks is
computed using the existing exact 393-block / 4673-term `recoupling_tables`:

    Cn(t,u,J) = sum_terms cterm * REAL(i^ipow *
        sum_f CPweight[f] * acA(f;la,lap;t) * acB(f;lb,lbp;u))

There is **no conjugation**, extra spin/off-diagonal factor, or added prefactor.
CP weights already include 1/(2*pi). The grids must match exactly, each static
node must have zero weight, each dynamic node must have positive finite weight,
and at least one positive node is required. Zero weight is skipped **before**
CP multiplication. Static tensors remain available in standalone recoupling;
like all coupled tensors they must themselves be representable in double.
Frequency and table term order are fixed. The imaginary residue is checked
**per phased CP integral**, strictly `<1e-8`, as in the source. Nonfinite
arithmetic or a failed residue check raises deterministic invalid input rather
than returning an uninitialized value. No scientific tolerance is relaxed.

## Results, coverage and limits

The result owns both recoupled models, the exact grid and authoritative weights,
and every ordered site pair, including self pairs. Pair coefficient records
carry `(order,t,u,J,value)`. All structurally contributing rows are retained,
including numerical zeros; `coefficient(order,t,u,J)` returns zero for structural
absence. There is no numeric print threshold in the engine.

Per-order coverage reports included rank quadruples, missing shipped-table
quadruples and missing unrestricted positive-rank quadruples (sum=n-2).
`table_complete` means all shipped quadruples are available; it does not imply
unrestricted completeness. The unrestricted list also includes quadruples not
in the shipped tables. Rank-3 C12 is partial, not full multipole-rank coverage.

Before result allocation the engine checks <=4096 pairs, <=2,000,000 returned
coefficient records, and <=100,000,000 frequency-term operations. Recoupling
checks <=8,000,000 complex values and <=100,000,000 record-frequency operations;
the existing local-model limits also apply. Product/addition checks avoid
size_t overflow. Pairs are evaluated in order with sparse structural records,
never a giant dense `[n,81,81,J]` array per pair. Metadata/coverage is bounded by
the fixed tables and the pair ceiling. Python checks weight sequence length
before conversion. Oversize input, arithmetic overflow, invalid indices and
unsupported declared ranks fail explicitly.

## Independent runtime example (no CamCASP files)

```python
import numpy as np
import psi4
c = psi4.core
s = c.IsaAnisotropicSite()
s.label, s.origin, s.frame, s.ranks = 'analytic', [0.,0.,0.], np.eye(3), [1]
s.responses = [c.Matrix.from_array(np.diag([2.,3.,4.]))]
a = c.IsaAnisotropicModel([1.], [s], 'supplied_local_response',
                           'independent analytic local tensor; atomic units')
ac = c.IsaRecoupledModel(a)
result = c.isa_recoupled_dispersion(ac, ac, [1.], 12)
print(ac.value(0,0,1,1,1))  # -trace(A)/sqrt(3)
print(result.pairs[0].coefficient(6,1,1,0))
```

This is an external expert supplied-local path. No new OEPROP task, adapter,
existing Python-module modification or implied native protocol is introduced.

## Portable archive: do not mix water tracks

`tests/pytests/data_isapol/recoupled_h2o_isagrid_l3/` is a separately named,
hash-pinned, 103044-byte compressed literal numerical fixture from the **exact
work/H2O-isagrid audit deck** and its pot output. It is not the historical
`orient_local` work/H2O fixture: O-O scalar C6 is **26.48177**, not **17.25559**.
The archive stores literal decimal tokens and one-based input line ranges / pot
line numbers. Pot fields preserve the original fixed 15-column field tokens,
including zero placeholders and absent trailing fields. The generator does not
compute any expected coefficient, nor import the audit transcode. Missing input
deck entries are exact zero and the documented reciprocal reader convention is
applied. Because the deck has no geometry/frames, fixture identity frames and
zero origins are explicitly bookkeeping, not a claim about molecular geometry.

Grid nodes and weights come from native `CasimirGrid(10,.5)`, not fixture
frequencies. The audit's squared-frequency endpoints were
-4.3686833258996777e-05 and -1430.6369983255513; its worst archive relative value
deviation was 4.997966647993114e-7.

The archive test enforces the exact **6285** J<=8 rowset, **10457** nonzero
values at rtol=1e-6 (no absolute allowance), **30791** written zero placeholders
at abs<=1e-6, and zero unexplained extras. It counts **411** additional visible
J9/10 rows without claiming numerical parity for them: casimir's writer only
loops J=0..8. Only the test constructs the legacy `abs>1e-6`/J<=8 view; the
underlying API never drops high J or small values.

The development-only `oracle/generate_recoupled_portable.py` requires explicit
CamCASP/reference paths, pins the exact deck/pot SHA256, parses all nine full
CG files, verifies all exact second-stage records using the existing development
parser without rewriting them, and records source hashes. Production and pytest
never open a home CamCASP tree or invoke/import the generator. The raw ~697KB
pot file is not shipped. Decompression is bounded and the compressed hash is
checked before reading fixture data.

## Attribution and license

CamCASP conventions and numerical artifacts are by **Alston J. Misquitta and
Anthony J. Stone**, Copyright (c) 2019 Anthony Stone, MIT. The complete permission
and warranty notice is in `RECOUPLED_CAMCASP_LICENSE` and fixture `LICENSE`,
recovered exactly with `git -C ~/gits/CamCASP show b40ae4f^:LICENSE`.
`realcg_manifest.txt` (JSON-formatted, .txt to avoid repository JSON ignore rules)
records actual SHA256 of all nine numerical CG files,
casimir.f90, c6..c12 tables, and that full license. Fixture manifest records
actual deck, pot and compressed-payload hashes. These are numerical tables,
not extracted executable source binaries. No ORIENT source was consulted.
Psi4 additions use LGPL-3.0-only.
