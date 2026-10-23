# Exact captured Q-grid AUX-only diagnostic

This is a new path, not a molecular launcher, replay, capture retry, gate
change or repaired Q. Author preparation is local and engine-free. Coordinator
review/publication/metadata admission precede **one** separately controlled
allocation. Old sources, launchers, run directories and evidence are not reused
as output destinations. No runtime/probe-manifest hash is fabricated here.

## Public-API feasibility proof (source inspection, no native execution)

The smallest route is public `core.IsaExplicitBasis.evaluate_screened`, followed
by bounded C-level NumPy reductions:

- `psi4/src/export_isapol.cc:720–761` exports `IsaBasisRole.MolecularAux`,
  Cartesian representation, effective-coefficient shells, immutable basis,
  `nfunction`, and public `evaluate` / `evaluate_screened`.
- `psi4/src/psi4/libisapol/explicit_basis.cc:24–38` specifies GAMINT Cartesian
  powers/factors through G. `:144–171` validates exact centres/shells/effective
  coefficients without renormalization. `:173–239` shows evaluation:
  all columns allocated, exact supplied unique centre list, no radial/shell
  cutoff, point-parallel native loops, effective contracted Gaussian radial
  factors times angular factors. Supplying `list(range(12))` includes **all**
  molecular AUX centres, not only one diagnostic origin.
- `psi4/driver/procrouting/isapol_distribution.py:202–219,250–268` confirms
  that captured Q uses this same explicit-AUX evaluator, all12 centres, and
  hashes `integration_grid.tobytes()` rather than the response-kernel grid.
- Public `IsaPartitionedMultipoles` is also sufficient:
  `partitioned_response.cc:43–94`, exported at `export_isapol.cc:690–719`.
  One rank0 site with all points/weights, shape=shape_sum=1, all12 AUX centres,
  cutoff0 gives `values[0,:]=U`; rank00 is exactly1 and exclusion is absent.
  It preserves native original point-order addition and internally bounds
  collocation to8MiB. **We do not use it** because the direct route also supplies
  absolute weighted sums, accumulation diagnostics and selected actual-point
  comparisons in the same collocation pass, without all-grid Python list
  samples or another expensive traversal.

No private binding, native monkeypatch, C++ modification/build, full collocation
matrix or Python scalar loop over1.6billion values is needed. Native and
production Python source are byte-identical to capture base c31ad7fb.
`integrated_runtime.py` has only an explicit operational plan-path factory;
its default old launcher behavior is unchanged. Each factory call creates a
separate module namespace, not a mutation of an existing launcher's plan.

## Exact inputs, grid and binding

`qgrid_plan.json` pins actual capture14291484's original marker, context,
anchor manifest, normalized-shell receipt and lossless q/S/deltaq bytes.
`verify_capture` validates links, fingerprints, shape/dtype/hash/finiteness
and exact saved `S-q=deltaq` before any allocated scientific import.
The remote root remains immutable capture-002. No missing mask, shapes, E,
V, F, Q or X are manufactured from hashes/tails.

Original fixture, original permutation and `pilot.make_molecule` retain the
absolute dimer frame, no reorientation/no COM shift. **`pilot.main` is never
called**. Within the same allocation, the original probe/new-driver derivative
and accepted GBS checks rebuild/admit MAIN and Cartesian AUX metadata, retaining
exact coefficients, shell order and all12 centres; no additional basis job.
The rebuilt AUX must equal the actual captured normalized-shell snapshot.
Only basis/PUREAM options are set; no density, orbitals or SCF is built.

Declared `IsaGrid` recipe: radial200 (199 shells), Lebedev590,
Becke smoothing3, radius scale1. Native `IsaGrid` retains atom-major,
radial-major, angular-major order, unfiltered Becke weights and translations
without spherical rotation (`isa_grid.h:49–113`).
Public grid bindings are at `export_isapol.cc:962–985`.
The exact native-core/geometry/options path matches captured
`integrated_driver.py:76–81`; all four knobs are now explicit.

Canonical grid: C-order little-endian binary64 `(x,y,z,w)` rows,
1,408,920×4 (45,085,440bytes). Recorded **partition Q integration** hash:

```
a01a17a4ab55dab608883cf5a184bfc13a42a15fae15efecf0256a9df383ab29
```

This comes from the pinned marker's
`source_repr.partition.diagnostics.integration_grid_sha256`, not top-level
`grid_sha256` (`06df...`, the different response grid). Grid count/layout and
canonical hash must pass **before collocation, U integration or comparison**.
A same-named/default grid is not admitted. Actual hash reproduction has not
yet been run.

## Bounded numerical work and outputs

-1,408,920points ×1146columns =1,614,622,320native values, 252 single-primitive
  shells (355,047,840native exponent evaluations); one traversal,1376tiles.
-1024point tiles; native matrix9,388,032bytes, products9,388,032bytes.
  One full grid only; no full point×AUX retained matrix or scratch spill.
  Point-list marshalling is only3072scalar values per tile. NumPy axis0
  reductions and vector Kahan addition are compiled operations. At most2160
  explicit independent scalar oracle checks, not billions of Python loops.
-This is a source-level work/memory feasibility argument, **not a measured
 30minute timing guarantee**. No benchmark/native run was performed locally.
  Deadline1500s from payload entry checked before/after grid and each tile,
  with TERM-safe failure and no retry. A native call/hash/loader operation
  cannot be interrupted by a Python deadline until it returns; SLURM's
 30minute wall cap still applies.
-Whole output budget8MiB, scratch64MiB, required free scratch256MiB,
  metadata JSON bound4MiB/file, payload file16MiB limit. Shell stdout/stderr
  files capped32MiB each. Runtime composition remains the accepted separate
 32MiB configured-Python bound; it is metadata staging, not numerical scratch.
  All numerical vectors/point diagnostics are finite and exclusive.
-Four lossless1146-entry `<f8` vectors: U, Kahan-across-tile-sums U,
  `sum(abs(w*chi))`, ordinary-U conditional summation-error diagnostic.
  The two U vectors share the same compiled per-tile reductions; Kahan is
  **not** an independent integration or collocation check. Record NumPy
  runtime identity, tile size, ordering and convention. The summation bound
  does not certify exp/collocation/product/underflow/quadrature error.
-Selected actual points: two per atom (zero-based radial100/150,
  angular137/389). All components of each most-diffuse shell S–G at C1/C2,
  S–F at H1 (90columns). Includes worst-delta column254, diffuse f x*z²,
  both even/odd Cartesian functions and both C/H. Point coordinates,
  column identities, native/oracle hex floats and measured differences
  are retained. Independent literal GAMINT powers/factors, `math.exp`,
  polynomial products and `math.fsum` use exact effective coefficients;
  no native producer is called twice and no tolerance forces admission.
-Signed U−q, S−U and captured deltaq have maxabs, signed peak/column,
  L1/L2/signed sums and decomposition residuals. A large U−q narrows to
  grid **or collocation**; it is not automatically a quadrature proof.
  S−U still combines original exclusions/unity/accumulation. No response
  projection, prediction, corrected tensor or physical-property acceptance.

## Future coordinator actions (not executed by author)

1. Independently review/test the new branch; verify committed tools-only diff,
   all132 prior cases, exact base, unchanged scientific trees and evidence.
2. Publish by committed Git into the new `qgrid_plan.json` source root.
   Never mutate old source/stage/run trees. Assert exact clean commit/tree.
3. Run metadata-only `qgrid_job.py prepare --commit <actual> --tree <actual>`
   under isolated `-I -S -B` Python. It imports no scientific packages.
   New exclusive runtime/source/run/log/scratch paths are all in the plan.
   Review actual composed runtime manifest, derivative admission, original
   probe and capture/grid pins; hashes exist only after that preparation.
4. Review returned **actual** test-only/future-submit argv and resource plan.
   Check exact scheduler test-only admission, fresh free capacity/storage/
   scratch/log paths, accepted build-node availability and no duplicate job.
5. One explicit `gts-cs207-chemx/cpu-small/embers` allocation,1node1task,
  2CPUs8G00:30:00, build node `atl1-1-02-007-23-1`, threads1, Psi4 memory4GiB.
   New `qgrid_probe.sbatch` dispatches only `qgrid_job.py run`, never the
   integrated/old molecular launcher. `--no-requeue`, no dependencies,
   paid/GPU route, install/download/fallback or automatic retry.
6. Same-interpreter reviewed Einsums1.1.3 `RTLD_NOW|RTLD_GLOBAL` handle
   retained before/after closure; existing runtime/core/data/dependency/
   mapped-library admission remains unchanged. `probe_complete` is distinct
   from `science_pass`/`properties_accepted`, which remain false. Batch
   complete requires actual unsignalled child exit0; signal/failure/closure
   refusal never publishes a completed probe.

Preparation does not authorize the author to publish, access remote systems,
invoke native code or submit. Stop after external report/ready + FIFO handover.
