# Lossless pre-LW capture and strict diagnostic replay

## Status and limits

Preparation only. `lw_capture.py` is **not connected to the pipeline**. It has
no scientific import at module import time. Tests and the example are literal
synthetic data, not P003 data or a molecular validation. Native replay is
implemented but **unrun**. No production package, gate, policy, source default,
existing driver, accepted stage, or previous artifact is modified.

P003's rounded residuals do not recover its tensors, solved response, or
wavefunction. This tooling does not establish the cause of its input defect.
It cannot replay P003 without a real pre-LW capture from a separately authorized
future calculation. There is no automatic retry, second response solve, PFIT,
projection, or molecular-property publication here.

## Owned artifact

`snapshot(NEW_DIRECTORY, ...)` accepts `Float64Array(shape, immutable_bytes)`
objects, with exact little-endian IEEE binary64 C-order bytes. Callers must
explicitly copy their live float64 arrays, checking dtype and shape first; no
object array, pickle, dtype inference, omitted-component padding, rotation,
normalization, or numerical repair is accepted. Cross-endian conversion must
be explicit and bit-preserving, never a floating-point recomputation.

Required arrays:

| Name | Shape | Meaning |
|---|---|---|
| `tensors` | `(1,n,n,25,25)` | frequency, response site, potential site, response component, potential component |
| `origins` | `(n,3)` | absolute supplied bohr coordinates |
| `frames` | `(n,3,3)` | proper local-to-global Cartesian columns |
| `sum_potential_site` | `(n,25)` | ordered `sum_b T[a,b,t,0]` |
| `sum_response_site` | `(n,25)` | ordered `sum_b T[b,a,0,t]` |
| `fsum_potential_site`, `fsum_response_site` | `(n,25)` each | compensated diagnostics only |

Optional `raw_site_major` has shape `(25*n,25*n)` and must agree **byte for
byte** with the correct pair mapping:

```text
pair[0,a,b,t,u] = raw[25*a+t,25*b+u]
# NumPy shape analogy, not an executable dependency:
raw.reshape(n,25,n,25).transpose(0,2,1,3)[None]
```

The full 25-component rank-4 source is preserved. The existing public route
supplies **16x16**, including charge, to native LW; its rank-2 working space
uses 9 components. A manufactured padded 9x9 input is not equivalent.
Components use real Racah/no-CS ordering, charge index 0, dipoles z,x,y.
Units are atomic units: origins bohr, imaginary frequency magnitude Eh,
response block `(l,lprime)` in `bohr^(l+lprime+1)`. Labels and bond
order/orientation are retained, not sorted or idealized.

Each `<name>.f64` is a raw byte file, not NPY. Exclusive directory creation
(0700), exclusive file creation (0600), flush/fsync, SHA-256, exact shapes,
dtype, byte counts, and `manifest.json` written **last** distinguish a complete
capture. Partial artifacts are preserved and never reused. Keep the returned
manifest SHA-256 in a separate trusted receipt; checksums alone are not
authenticity or binary/source attestation.

The manifest declares exact frequency using `float.hex`, production policy,
unchanged 1e-6 tolerance, input rank 4, explicit rank-4 truncation, localization
rank 2, frames/axes/units, core path/hash/source commit/tree/build node, Python
LW boundary path/hash/package/source identity, separate driver source identity,
and config/private basis-manifest/fixture hashes. These are caller identities,
not a claim that this module proves the entire runtime closure.

Bounds: 1–64 sites; one frequency; 64 MiB **aggregate** array bytes; 1 MiB
manifest; identity paths at most 1024 characters; labels at most 256 UTF-8
bytes and unique/nonblank. Optional raw bytes consume the same aggregate
budget. File sizes, manifest pin, duplicate keys, declarations, array names,
shapes, dtype, and total payload size are validated before binary data is
opened. Individual input files reject symlinks and nonregular files. This is
private diagnostic storage, not a hostile multiuser-directory sandbox.
Nonfinite values, corrupt hashes, inconsistent raw/pair mappings, invalid
frames, and inconsistent signed diagnostics fail closed. No invented numeric
tolerance is used: frame checks mirror the public existing 1e-12 rule.

All 25 signed component sums are retained; native-working argmax uses indices
0..8. Ordered `+=` follows native site iteration. `math.fsum` is reported
separately to expose cancellation, never fed into LW. Tie-breaking and signed
site/component/axis/value are explicit. Python scalar reductions do not by
themselves certify native compiler arithmetic equivalence.

## Engine-free commands

Run in this source worktree, using a **new** example directory:

```bash
python3 -I -S -B devtools/ccasp_phoenix_benzene/test_lw_capture.py -v
python3 -I -S -B -m unittest discover \
  -s devtools/ccasp_phoenix_benzene -p 'test_*.py' -v
python3 -I -S -B devtools/ccasp_phoenix_benzene/lw_capture.py example /NEW/toy
python3 -I -S -B devtools/ccasp_phoenix_benzene/lw_capture.py inspect /NEW/toy \
  --manifest-sha256 RETURNED_SHA256
```

The two-site toy has explicitly asymmetric signed sums -3 and +6, plus a
signed-zero element. Its residuals, placeholder source paths and identities
are synthetic. Do not relabel it as a real capture.

## Strict replay contract — unrun native adapter

`replay(capture, strict_boundary=...)` gives the boundary independent immutable
byte copies and calls it once. Tests use refusing stubs. Before and after
hashes detect operand mutation. Any unrelated exception is re-raised unchanged.
Only a `RuntimeError` containing exactly one native LW postcondition block
can qualify: all seven named fields must be finite/nonnegative. All four owned
off-site/reciprocity/molecular-sum/charge-transport **printed** fields must be
strictly below 1e-6: equality is refused as rounding-ambiguous, since an actual
above-gate value could print as 1e-6. The exact captured ordered input maximum
must exceed the unchanged 1e-6 gate and match the printed input field's native
six-significant-digit display. Other supplied residuals need not exceed the
gate. This conservative classification and display-token corroboration are
not new scientific tolerances. Ambiguity fails closed.

An expected refusal yields `EXPECTED_INPUT_SUM_RULE_REFUSAL`, diagnostic pass
only, **`scientific_properties_accepted=false`**. Unexpected success is a
diagnostic failure and its property return value is discarded. No exception
is swallowed to continue the campaign.

`native_replay()` delegates to the same public
`isapol_lw.supplied_nonlocal_properties` once, production policy and one
frequency, hence one actual `core.isa_localize_lw` call after public validation.
It requires a real `native_owned` artifact and a Phoenix allocation on the
captured native build node. Before any scientific import it refuses local or
login-node execution. It requires Psi4/NumPy **already initialized in the same
interpreter** by a separately reviewed exact stage/CDLL runtime bootstrap;
it never searches for or imports a fallback runtime. It checks captured
clean source commits/trees, core path/hash, package path and boundary file/hash
against that process. The external bootstrap must still verify the complete
accepted dependency/loader closure before and after replay.

CLI `replay-native CAPTURE --manifest-sha256 SHA --out NEW.json` provides an
exclusive result receipt. A fresh standalone Python process intentionally
fails unless the reviewed initializer has populated its runtime. Future
allocation code must import this module and call
`main(["replay-native", ...])` **in the initialized interpreter**, not launch a
second subprocess that loses admitted CDLL state. Expected diagnostic refusal
returns 0; unexpected strict success returns 2; unrelated failure writes
`REPLAY_FAILED` and propagates. No shell launcher/job is authorized here.

## Smallest future opt-in integration (proposal, NOT implemented)

1. An explicitly approved future source change may add an optional
   caller-injected snapshot callback immediately before
   `isapol_bounded.py`'s `lw.supplied_nonlocal_properties(...)` in its existing
   per-node loop. It already owns `raw`, the complete reshaped/transposed
   `tensors`, sites, frames, labels, bonds, frequency, and recipe identities.
   Default `None` must leave the production path unchanged.
2. The callback makes owned exact bytes and calls `snapshot` once with a new
   node-specific exclusive directory. Require strict production/rank/truncation
   declarations and pinned identities. Failure to capture must propagate
   **before** the LW call; never catch and continue without a complete capture.
   Preserve the existing strict LW call and its natural failure unchanged.
   Do not rebuild, re-solve, mutate inputs, retry, run PFIT, or publish
   properties after refusal.
3. Alternatively the explicit opt-in seam can sit in `isapol_lw.py`
   immediately before `core.isa_localize_lw(...)` inside the frequency loop.
   Capture full owned original 25x25 tensors *and* the actual supplied
   16x16 blocks if diagnosing that lower seam. Never infer omitted components
   from rank-2 output. Choose one seam, not double capture/interception.
4. Record source integration/review pins separately from the immutable accepted
   native15cc stage. Review and approve any changed Python routing before a
   future allocation; this preparation does not authorize editing the frozen
   runtime or injecting monkeypatches.
5. A later separately authorized real pre-LW capture would need the credible
   molecular allocation and its own native-runtime/basis admission; no P003
   checkpoint is assumed available. Once a complete real capture exists,
   strict replay alone needs no SCF, response solve, density, ISA iteration or
   PFIT. A short free-embers same-build-node CPU allocation may then be proposed,
   with explicit threads1, absolute logs, existing exact runtime bootstrap and
   measured memory/storage headroom. There is no submission, benchmark retry,
   resource grant, or auto-resubmission in this deliverable.

Signed attribution is a **separate future operand-lifetime seam**, described
below; a small LW tensor alone cannot recover its inputs.

## Signed attribution proposal — no new solve

Use names that do not confuse the solver's integer `q` (number of moment
columns) with the AUX charge vector below:

| Symbol | Shape / source |
|---|---|
| `Q` | `(m,p)`, `moments.values`, `m=25*nsite`, `p=naux` |
| `c` | `(m,)`, charge-row selector: 1 at `25*a`, otherwise 0 |
| `q` | `(p,)`, **actual** `charges` from the existing `IsaAuxCoulomb.charges()` call |
| `F` | `(nov,p)`, the **anchor** constrained fit's `fit.coefficients`; row `a*nocc+i` |
| `B` | `(nov,m)`, existing `anchor = F @ Q.T` |
| `X` | `(nov,m)`, existing solved **anchor RHS columns**, `solved[:,p:]` |
| `alpha` | `(m,m)`, existing `-B.T @ X`, before pair reshape |

The signed decomposition requested for the column-charge axis is

```text
deltaq = Q.T @ c - q          # AUX charge-row mismatch, no correction
ufit   = F @ q                # finite charge-penalty fit contribution
uQ     = F @ deltaq           # distributed-moment charge-row contribution
sfit   = -ufit.T @ X          # (m,)
sQ     = -uQ.T @ X            # (m,)
alpha.T @ c ≈ sfit + sQ       # signed; finite arithmetic regrouping differs
```

The last comparison is a diagnostic, not a new equality tolerance or gate.
Record the **actual** `alpha.T @ c` and `alpha @ c` separately; do not replace
one axis using reciprocity. The second axis can be attributed with existing
`B` and `X @ c` if needed; it is not licensed to assume `X` is symmetric.
Even `F @ (Q.T @ c)` and `(F @ Q.T) @ c` need not be bitwise identical.
Use the actual retained `B=self.legs[:,p:]` for direct contractions rather
than reconstructing it later and hiding the arithmetic regrouping.
Keep global real-Racah site/component order and both signed maxima. Rank-0
units differ from higher ranks; retain per-component ranks and atomic units,
never a dimensionless mixed-component norm as a physical error certificate.

Relevant immutable source locations:

- `isapol_native_factors.py:204–209,223–244` (`native_constrained_ov`) obtains `charges`, forms
  `J_eta + lambda*q*q.T` and returns only fitted coefficients and scalar
  diagnostics. `charges` is not currently part of its returned public record.
  A future opt-in diagnostic callback at this existing operand lifetime would
  be needed to preserve that exact vector, without another integral/fit solve.
- `isapol_bounded_response.py:668–688` (`BoundedResponse._prepare_reference`), anchor
  branch, has both `fit.coefficients` and `moments.values` immediately before
  `store.save(name, moments.anchor_legs(fit.coefficients))`. It then deletes
  `moments` and `fit`. Compute/retain only the signed `deltaq`, `ufit`, `uQ`
  (with exact operand identity hashes) there if separately authorized.
  Do not substitute the subsequent **eta=0 target fit** for the anchor fit
  whose eta is 5e-4 in this scientific profile.
- `isapol_distribution.py:110–121` (`DistributedMoments.anchor_legs`) defines the actual
  `F @ Q.T` contraction, with density moments and **no electronic minus sign**.
- `isapol_bounded_response.py:361–373` (`_H2H1Response.solve`) has the actual `solved`
  after `lu_solve` and the unchanged response-residual check. Immediately
  before its return, `self.legs[:,p:]` and `solved[:,p:]` give `B` and `X`.
  Reduce the already available `X` against retained vectors there, after the
  gate passes. `FrequencyResponse.target_response` is a contracted **p x p**
  object, **not X**; `nonlocal_response` is already contracted m x m. The
  outer pipeline cannot reconstruct X from either.

No such callback, retained vector, reduction or package change is implemented
in this preparation. This identifies genuine missing lifetime/export seams,
not a callable workaround against the frozen package. A future implementation
must admit and account for diagnostic live-byte/work overhead, keep one
existing response solve and natural failures, and never serialize full X
merely to evade the resource ledger.

For roundoff diagnosis, preserve exact input-vector bytes/hashes and report
ordinary signed reductions plus separately compensated dot products over the
**same ordered products**. Compensating `Q` charge rows, `F @ q`,
`F @ deltaq`, and `u.T @ X` tests summation sensitivity only: it does not
recover exact products, eliminate LU error, prove physical neutrality, or
identify a root cause. Do not invent a new normalization, tolerance, corrected
tensor or accepted-properties path.
