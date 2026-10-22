# Python/Einsums GPU SAPT0 dispersion progress

## CURRENT native gate — cuEST S/P/D device integrals pass; DFMO/SAPT pending

Private dependencies now build/install/import coherently: exact Einsums064
CUDA86 with whole matching Waggle, private HDF51.14.6, CPU LibXC7.0 for the
oracle and separate GPU LibXC7.1.2. Whole reviewed factory-stage03 at58bd62caf
passes core/metadata/all896Python-file identity and native CPU smoke; no overlays.
E2's7used host-API fixtures pass. The separate CPU/native FISAPT suite passes
22tests/1expected mock-only skip with64signed comparisons, max1.003e-17Eh.
This CPU oracle's DFHelper factors never feed a production GPU calculation.

Real native attempt01 stopped at binding-version admission. The user approved
a scoped-prefix-lock correction of only cuda-bindings13.4.4→exact13.3.0.
All13855protected files/483other package records remained unchanged; old payload,
exact archives and corrected explicit lock are retained. Bound/direct canonical
CUDA versions now both13030; fresh whole-stage/E2/API checks pass.

Native attempt02 reached actual NVRTC13.3 compilation/module load on canonical
3090 UUID/PCI/cap8.6 (PTX SHA404cb7d2...287f0e2), but launched no kernel and
made no cuEST constructor call. It rejected a Btrfs identity discrepancy:
same inode, maps/mountinfo superblock device0:34 versus fstat subvolume0:51.
Session cleanup completed, no live allocations/native owners remained.

Live non-executable RED reproducer isolated that filesystem behavior. Review
rejected the proposed fdinfo/mountinfo/path-link fallback: two overmounted
Btrfs subvolumes can present the same path/inode/raw-device while the mapped
and opened files differ. A stable descriptor mount record identifies only the
opened file, not the VMA's backing mount. The adversarial negative went RED
against the proposal, which was withdrawn completely. **56cuEST+31CUDA guards
now PASS with the original strict runtime code**, byte-identical to installed03.
No Btrfs exception was installed. Prior positive probe evidence did not prove
safe alias admission.

Explicit byte-identical deployment to private executable tmpfs now PASSES:
2268files/1787961811bytes, original CUDA16-file native closure,2192CUDA includes,
canonical59cuEST headers/native. Destination hashes/inventory, owner/mode,
host/tmpfs/cgroup-ancestor headroom and original strict live executable-mmap
identity checks pass. Original ENV/SDK/E2/Psi4 remain unchanged; no RPATH,
Python-overlay, mount-namespace or runtime-guard patch.

Fresh-process loader-only probe PASSES: exact58bd core before deliberate
dependency callback, callback before first E2 request; real singleE2 world,
approved binding package, runtime13030/NVRTC13.3 and canonical3090 identity.
Actual toolkit/cuEST mapped-byte checks and NVRTC/driver resolver/cuBLAS
pointer ownership pass; canonical cuEST DSO/17factory symbols audited,
9DSO owners retained through process completion. No intentional cuEST
construction/version getter, DFMO or custom-kernel API was issued. ELF
initializers may execute. Core proof assumes a trusted frozen install using
pathname/file hash; system-driver proof uses canonical file hash and mapped
pointer path, not strict mapped-byte identity. Toolkit/cuEST proof is stronger.

Loader review conditionally cleared the controlled probe;27offline synthetic
guards pass. Tiny construction harness now accepts explicit paired staging/
loader proofs, binds completed prior result/report to the same whole stage,
and re-admits all live loader inputs in the fresh child. Combined38offline
tests pass. Construction handoff review conditionally cleared the controlled
attempt, then **native-plan-device-03 PASSED**52.940s plus NVML postflight.
Actual owned context reports0/2/50 and passes native FP64 checks. H2 STO-3G
2AO/cc-pVDZ-RI28aux bases, primary pair list and DF integral plan constructed.
Stream and full-Coulomb/MATRIXPOWER/relative-cutoff policy readback pass.
Reverse native and session cleanup complete,0live bytes/retained native
owners/ambiguous outcomes. Peak explicit allocations746752bytes; opaque native
memory and total peak remain unmeasured.9deployment DSO owners retained through
cleanup. NVRTC compile/module load succeeds; no custom kernel launch.
Coefficient candidate remains UNVALIDATED_PSI4_COEF. Construction success does
not admit normalization/components/units/metric rank/layout or DFMO numerics.

New independent semantic gate **PASS**: complete real cuEST device overlap and
kinetic matrices for two-center Cartesian20AO/pure18AO S/P/D fixtures. Off-center
allXYZ displacement, single primitives and unequal-exponent mixed-sign
contractions; known-input analytic Gaussian/Racah oracle and exactPsi4 reference
agree. Device maxabs vs analytic4.441e-15, vs Psi43.553e-15, unchanged
rtol1e-11/atol1e-12, no fitted rescaling/permutation. Seven added API signatures/
enums/workspace layout compiler-witnessed, intended wrong-count negative fails;
actual mapped-symbol ownership checked.50offline guard/oracle tests pass.
Actual native compute and full cleanup pass82.695s plus NVML postflight,
0live tracked allocations/native owners. Explicit allocation peaks613632bytes
Cartesian/611648bytes pure; opaque native peak remains unmeasured.
No CPU reference integral/factor uploaded. This supports coef()/normalization/
pure ordering/Bohr conventions for these S/P/D fixtures only. Higher-L/arbitrary
bases, RI metric/rank, storage orientation, DFMO output and signed SAPT remain
unqualified. Symmetric S/T do not establish row/column-major layout.

Receipts: `../runs/native-plan-device-01/`, `native-plan-device-02/`,
`cuda-runtime-version-probe-02/`, `binding-pin-apply-01/`,
`btrfs-identity-red-01/`, `btrfs-identity-green-01/`,
`mapped-identity-live-{red,green}-01/`, `btrfs-overmount-red-01/`,
`btrfs-overmount-strict-green-01/`, `cpu-signed-01/`.
New deployment receipts: `../runs/native-tmpfs-stage-01/`,
`native-loader-inputs-01/`, `native-loader-device-01/`.
Native construction receipt: `../runs/native-plan-device-03/`.
Semantic receipts: `../runs/basis-semantic-cpu-01/`, `basis-semantic-abi-01/`,
`native-plan-basis-semantic-01/`; plan: `../metadata/BASIS_SEMANTIC_GATE.md`.
Current full chronology: `../metadata/SETUP_STATUS.md`.
Production still BLOCKED: no general basis/layout/rank admission, DFMO compute,
custom-kernel numerical execution or signed GPU gate. No DFHelper fallback,
F-SAPT expansion or speedup claim. Following entries preserve earlier states.

## PRIOR dependency rebuild — private HDF5 discovery proved

Whole Einsums build04 FAILED at executable link346/347 after3303.60s; no
installation or runtime import proof. `calibrate_hardware` linked system
`/usr/lib/libhdf5.so.320.2.0` (config2.2.0), requiring GLIBC2.38 `__isoc23_*`
symbols unavailable in the private GNU14.3 sysroot. Failed artifacts retained.
The private environment already has HDF51.14.6 and its `h5cc`, but no HDF5
CMake config; default FindHDF5 preferred the system config before the wrapper.

Exact E2 `FindTargetHDF5` minimal discovery/link/runtime probe **PASS**.
RED reproduced the foreign2.2.0 selection and `__isoc23_` link failure.
The sole GREEN discovery change `HDF5_ROOT=<private environment>` selected
private1.14.6 headers/library; linking and actual `H5get_libversion`/`dladdr`
confirmed1.14.6 at `environment-v2/lib/libhdf5.so.310.5.1`, SHA256
`e687f5de4db1dff2e01c199ff4cdf9767e72c14b05b1fe297960a2522a165500`.
The new pre-build cache/library-origin guard rejected the RED cache and
accepted GREEN. No compiler/sysroot, E2/scientific pin, tolerance or validation
bypass. Fresh one-job build05 STARTED, with HDF5 admission required before
compilation; full build/install/import results remain pending. No failed04 reuse.
Receipts: `../runs/einsums-064-build-04/`, `../runs/hdf5-prefix-probe-01/`,
`../runs/einsums-064-build-05/`.

Whole native Psi402 build/install **PASS**,3824.25s/0.924s, source frozen clean
at a7204feec throughout. Independent installed `--psiapi`/core/Python inventory,
CPU LibXC maps and H2 PK/direct smoke are running at
`../runs/psi4-cpu-import-01/`; import result is not yet claimed. Native E1
remains OFF; whole Python E2 build and subsequent coherent E2 import are
separate pending milestones. This stage does not contain the unmerged factory
branch, and no new-module overlay is installed into it.

## CURRENT diagnostic native resource factory — implemented, not device-admitted

`sapt_cuest_plan_owner.py` now constructs an explicitly requested diagnostic
context, parameter owners, all AO shells, primary/RI bases, primary pair list
and one full-Coulomb MATRIXPOWER DF plan. It checks owned native version/FP64,
parameter readback, native basis dimensions and exact basis snapshots. Stream/
workspace/parent lifetimes are retained through reverse-order cleanup; creation
temporaries remain conservatively live for these initial small-case diagnostics.

Failed cleanup stops before parent release. Retry is explicit cleanup-only;
ambiguous creation/destruction outcomes never receive speculative destructor
calls. Post-destroy synchronization failure retains retired host inputs and
parents without re-destroying a completed native object. Known workspace
subtotals are distinguished from the declared allowance for unmeasured opaque
native internals. No total-VRAM or native-memory-certification claim.

The required coefficient candidate is explicitly `UNVALIDATED_PSI4_COEF`.
This class cannot yield a production factor provider and performs no DFMO
compute, DFHelper work or CPU integral/factor generation. Normalization/
component/unit/layout/metric/rank/signed physical admission remain unpassed.

**49 cuEST CPU structural/ABI/identity/owner tests PASS,0.202s;31 CUDA CPU
fixtures PASS,0.907s**, diff check PASS; `../runs/native-plan-owner-03/`.
Resource-graph review found a P1: an owner registered before a temporary
session scope could delay construction into that scope, whose exit would free
borrowed workspaces. Exact delayed-build/enter regressions went RED; construction
guards now reject at build/native-create/workspace boundaries before resource
creation. Existing-built native workspaces survive a later scope's cleanup
failure. Fake destruction now checks persistent AND creation-temporary device/
host buffers, and shell/basis/pair inputs survive forced GC through a failed
post-destroy synchronization, then release only after explicit successful retry.
**52 cuEST CPU tests PASS,0.238s;31 CUDA CPU tests PASS,0.872s**, diff check
PASS; `../runs/native-plan-scope-red-01/` and `native-plan-scope-green-01/`.
Narrow confirmation d46b8b **closed the P1 with no new scoped blocker**;
read-only review of owner/tests, not an independent test/native run.
Reviewed-state rerun **52 cuEST CPU tests PASS,0.231s;31 CUDA CPU tests
PASS,0.861s**, diff check PASS; `../runs/native-plan-reviewed-01/`.
Header compile probe02 **PASS**, including
the exact native basis query IDs used here, retaining17-signature/all59-header
and wrong-count negative controls. Native constructor/DFMO execution remains UNRUN.

## CURRENT native-owner session leases — lifecycle groundwork only

CUDA sessions now retain explicit native cleanup owners. Persistent owners
cannot be registered inside temporary allocation scopes. Session close must
finish native destruction/unregistration before freeing device allocations
or destroying its module/BLAS/borrowed stream; failed native close retains
all those dependencies, permanently disables execution and permits only
cleanup continuation. Primary computation exceptions remain primary.

A CPU negative control executed the old actual `close` body and observed it
free dependencies while a fake native owner remained live; the guarded body
rejects and retains them. **27 CUDA CPU tests PASS,1.113s;31 cuEST structural/
ABI/identity tests PASS,0.047s**. `../runs/native-owner-leases-01/`.
Scoped read-only lease review found no blocker. Its requested coverage gaps
are added: multiple-owner reverse cleanup/partial success/retry without
reclosing successful owners; completed-but-still-registered owner blocking;
nested scopes with computation/cleanup failure and depth restoration.
Final rerun: **31 CUDA CPU tests PASS,0.928s;31 cuEST structural/ABI/identity
tests PASS,0.050s**, diff check PASS; `../runs/native-owner-leases-02/`.
The diagnostic context/basis/pair-list/plan graph is implemented above;
actual native execution and all production numerical admission remain unpassed.

## CURRENT factory ABI/policy component — CPU fixtures only

`sapt_cuest_factory_abi.py` adds canonical constructor/workspace/destroy/
parameter ABI and mapped ownership checks for every factory symbol before
native use. It freezes the CUDA manifest, checks the session/cuBLAS closure,
and never creates a hidden handle merely to query version. Typed parameter
round trips explicitly request stream ownership, native-owned BLAS/solver,
one JIT thread, full Coulomb/MATRIXPOWER1/relative1/cutoff1e-12/LRC0/direct0.
An explicitly owned same-process context can be set/read back to native FP64.
Native/readback/map/device errors permanently disable further computation.

**9 new CPU ABI/policy tests PASS,0.025s;10 basis tests PASS,0.007s;
22 CUDA CPU fixtures PASS,1.173s;12 synthetic identity PASS,0.018s**,
diff check PASS; `../runs/factory-abi-01/`. Scoped reviewer `da870d...`
exceeded its24-tool-call limit without an answer; NOT a review pass.
An independent GNU14.3 C++ header compile probe is now running against exact
canonical headers, including workspace offsets, enum/policy IDs and all new
factory signatures, with an intentionally wrong uint32 primitive-count
negative control. It performs no link, native invocation or device work.
Artifacts: `../runs/factory-header-abi-01/`; **PASS**: GNU14.3 compiled all17
new factory wire signatures, workspace layouts and enum/policy IDs while
consuming all59 canonical headers. Positive compile0 with no diagnostics;
intentional uint32 primitive-count negative compile1 failed exactly at
`cuestAOShellCreate wire signature mismatch`. This is actual compiler/header
evidence, NOT invocation of native constructors/getters or CUDA execution.
This component does NOT construct/destroy live native objects, implement the
owner graph, admit normalization/layout/rank semantics, or yield a production
provider. It is NOT native ABI execution or GPU proof. Native ownership and
all numerical gates remain unfinished; no host-factor/DFHelper fallback.

## CURRENT factory development — separate owned Cachy worktree

Main compilation checkout stays frozen clean at `a7204feec` while native Psi4
builds. Factory development is isolated in `../factory-source` on owned branch
`cachy/disp0-native-factory`; no merge into another agent's branch or master.

New `sapt_cuest_basis.py` performs only structural public-API preflight: exact
shell centers/offsets/geometry, actual `nbf()` versus Cartesian `nao()`, separate
primary/RI representation flags, three uninterpreted coefficient variants and
immutable fingerprints. Rejects ECP, Cartesian RI, mixed representations,
angular-momentum holes, missing centers, unadmitted permutations, nonfinite
data and changed inputs. It never generates factors or initializes CUDA.
Native normalization/component/unit/layout policy remains unadmitted; this
module cannot create or admit a production provider.

**8 new CPU structural tests PASS,0.004s;22 existing CPU CUDA fixtures
PASS,1.036s;12 synthetic identity fixtures PASS,0.021s**.
Logs `../runs/factory-basis-01/`. Review confirmed the exported API and
structural-only scope, but found that owning-molecule coordinates could change
independently of stored shell coordinates. Snapshots now include both and
reject moved/stale disagreement; changed-input detection covers molecule-only,
shell-only and consistent dual movement plus all3 coefficient variants.
Rerun: **10 basis tests PASS,0.007s;22 CPU CUDA fixtures PASS,1.266s;
12 identity fixtures PASS,0.019s**, diff check PASS. Logs:
`../runs/factory-basis-02/`. Narrow confirmation delegate `ddccdb...` exceeded
its6-turn limit and produced no committed answer; NOT counted as a review pass.
The reported gap is addressed by direct source inspection and the explicit
geometry/mutation regressions, not by an unavailable follow-up opinion.
Native resource/basis/plan construction, native numerical conversion and signed
Disp0 GPU gate remain unfinished. Production still has no DFHelper fallback.

Dependency update: GPU LibXC7.1.2 actual device-buffer LDA diagnostic PASS,
21 FP64 EXC/VXC/FXC outputs versus analytic exchange and explicitly host-mode
same-build evaluation, maximum absolute discrepancy2.8421709430404007e-14.
PID66536/expected3090UUID/runtime13030/maps/native hashes retained in
`../runs/libxc-gpu-smoke-02/`. This is NOT cuEST/Disp0 execution evidence.
Whole E2/Waggle and native CPU Psi4 builds remain in progress. Psi4 build01
was stopped when inner Ninja lacked a job cap; isolated nested-build RED/GREEN
verified `CMAKE_BUILD_PARALLEL_LEVEL=1`, and fresh build02 uses it. Stopped
artifacts are retained, not adopted. Full receipts: `../metadata/SETUP_STATUS.md`.

## CURRENT directive — no DFHelper in production dispersion

Read mandatory `../metadata/NO_DFHELPER.md`. Production is strictly cuEST0.2.50
DFMO → device metric-transformed factors → cuBLAS/custom CUDA kernels.
No direct/indirect DFHelper, host-factor cache/disk transforms, NumPy producer
or silent fallback. Keep existing default/non-HF CPU paths and explicitly
separate correctness oracles unchanged. An unfinished factory means BLOCKED.

Actual source call-chain audit: `dispersion0_cuda` passes a canonical callback;
`fdisp0` invokes it and returns before constructing DFHelper. Only12 orbital
spaces/8 rank-one corrections/energies/basis objects cross into the callback,
not cached DF factors. The strict provider uploads coefficients, calls native
DFMO into owned device output, and the consumer rejects host/foreign-session
factor buffers before device packing/cuBLAS/reduction. No fallback was found.
Native basis/plan factory and real DFMO provenance remain unimplemented/unproven.

Added public-entrypoint CPU regressions with poisoned cached DFHelper and all
ten host-factor names, stale CPU provenance, factory/native failures and wrong
provider rejection. **22 CPU CUDA fixtures PASS,0.990s;12 identity fixtures
PASS,0.020s**, diff check PASS. Logs `../runs/no-dfhelper-audit-01/`.
These tests are explicitly CPU orchestration spies, NOT GPU execution proof.

Dependency progress: exact Catch2 and mimalloc private builds resolved two
Einsums fallback configure/export failures without modifying064. Fresh config03
PASS,25.12s. Whole Einsums/Waggle build04 is running with native Python prefix
installation, one job. Separate GPU LibXC7.1.2 build/install PASS
(691.93s build, sm86/GNU14.3/CUDA13.3); its tiny standalone device diagnostic
is pending, not a SAPT gate. CPU LibXC remains distinct. Exact commands/failure
logs and updated receipts are under `../metadata/SETUP_STATUS.md` and `../runs/`.

## CURRENT Cachy continuation — private native build started

Clean starting HEAD/source receipt `af13b713b14282d3b9e55301eff842002c78962b`.
Fresh source-version guard PASS, exact annotated v1.11 and complete history:
`../metadata/source-version-cachy-01/validated.json`. Existing `environment-v2`
and build-only `clang23` verified; GNU14.3, nvcc13.3.73, cp313,
cuda-bindings13.4.4. The two dependency source pins match and are clean.

Actual local device inventory matches setup: RTX3090, capability8.6,
driver615.78.08, UUID `GPU-d36fd194-bdd6-3d53-96c9-6a67e3225eac`.
Approximately1005MiB occupied at initial probe; no desktop contexts changed.
GPU LibXC7.1.2 source `f1eb21e8d6743ed5ba519b8950c1ae02c3f5acd9`
is building privately with two jobs and explicit CUDA architecture86.
New build/install attempt `libxc-gpu-01`; exact commands, package-record hashes,
phase logs and eventual outcome live at `../runs/libxc-gpu-01/receipt.json`.
Reproducible setup command is in `../metadata/SETUP_STATUS.md`.

Fresh private cp313 fixture rerun: **16 CPU serial-body/transport/setup PASS,
0.835s; 11 synthetic identity PASS,0.016s**. Logs:
`../runs/cpu-fixtures-01/`. These remain CPU fixtures, NOT CUDA/native proof.
The existing environment LibXC7.0.0 is CPU-only, not the required GPU artifact.
Einsums064 and whole Psi4 builds/import, native factory/admission, real
CUDA/cuEST/cuBLAS execution and signed physical gate remain unpassed.
No F-SAPT expansion or performance claim. Phoenix remains out of scope.

Read-only native-factory audit confirmed missing context/basis/pair-list/plan
construction and cleanup, native policy readback, basis normalization/ordering,
actual output layout and allocation accounting. Canonical headers require a
pure RI basis and disallow mixed AO representations; unsupported cases must
reject. cuEST default math mode may use mixed precision, independently of
the default QR fitting policy.

Added fail-closed **native FP64 mode1 readback** after symbol-ownership admission
and before each DFMO batch; changed mode/getter failure invalidates computation,
with no silent policy mutation. Canonical enum-pointer ABI joins mapped-symbol
checks. New CPU negatives went RED (5 failures), then **18 CPU CUDA fixtures
PASS,0.768s;11 identity fixtures PASS,0.016s**, diff check PASS.
Raw RED/GREEN logs remain in `../runs/cpu-fixtures-01/`.
Scoped read-only review found no blocker; its identified coverage gaps were
added (missing/foreign-owner getter, compute-time getter status/exception,
report mode/readback drift). Final rerun: **20 CPU CUDA fixtures PASS,0.849s;
12 identity fixtures PASS,0.019s;58 optimized-build guards PASS,0.534s**,
diff check PASS. Logs: `../runs/cpu-fixtures-02/`.
This fixes one admission gap, not the missing factory or physical GPU gate.
Exact shared references `a323afa...`/`9d6d3d4...` were initially absent locally;
an exact-object fetch succeeded without merging/changing any checkout.
Both commits were inspected; they provide shared SDK/version admission, not a
Python Disp0 plan factory. Detailed restrictions/ownership/proof gaps:
`../metadata/factory-audit.md`. Einsums064 CUDA configure-only attempt
`einsums-064-config-01` started (sm86/GNU14.3/LLVM23); no concurrent E2 build.

## Cachy migration — private setup in progress

User requested a dedicated worktree and local agent on `cachy:gits/psi4`
to avoid Phoenix GPU queues. Read-only probes identified RTX3090/24GiB
(UUID GPU-d36fd194-bdd6-3d53-96c9-6a67e3225eac), driver615.78.08,
Ryzen2700X/32GiB RAM. Existing desktop GPU contexts stay untouched.
Existing master and three sibling worktrees/environments are preserved.
Private setup target: `/home/amwalla3/gits/psi4-disp0-cachy/`.

`devtools/disp0/cachy/AGENT_HANDOFF.md` records exact scientific/dependency
pins, current native factory/import blockers and the unpassed GPU gate.
`bootstrap_env.py` creates NEW private cp313/GNU14.3/CUDA13.3 and LLVM23
prefixes, excluding installed Psi4/Einsums1 packages from the baseline.
It records raw solver/install failures; no implicit CUDA13.2 fallback.
Actual creation/compilation/import/device results will be separate receipts.
No Phoenix allocation or environment changes are part of this migration.

Cachy staging actually completed at source `ac75cab59ab4a5c7444a6ebceb917f114551129f`,
with clean private Git worktree and exact annotatedv1.11 release tag. Canonical
cuEST archive/native/all59headers verified and installed as private artifact
under `dependencies/cuest/cuest_PID`; no native/device execution. E2064 and
LibXC7.1.2/f1eb21e8d6743ed5ba519b8950c1ae02c3f5acd9 sources cloned clean.

First bootstrap: source-version/baseline clone PASS; full cuda-toolkit13.3
dry-run FAILED (optional Nsight GUI/alsa baseline conflict), no CUDA installation.
Preserved metadata/bootstrap-01 and baseline environment. Development-only
cuda-libraries-dev13.3 + nvcc13.3.73 proposal PASS,61 linked/3 unlinked packages;
includes genuine matching cusparse/nvJitLink headers/runtime without GUI tools.
`finish_env.py` freezes that proposal (raw hashes retained), creates NEW
environment-v2 and private LLVM23, rather than adopting/replaying a failed prefix.
Actual installation remains pending this record. User will launch the remote
agent; do not launch a duplicate. Remote Pi1.3 lists gpt-5.5 but lacks the newer
auth-check CLI; existing credentials are configured, not model-call validated.

## Current LOCAL work — CuPy-free CUDA replacement / strict0.2.50

User explicitly authorized custom CUDA kernels and dropping CuPy, superseding
the earlier prohibition for this experimental backend only. The old CuPy
prototype remains in Git history; working source replaces its modules/entrypoint
with `dispersion0_cuda`, Python cuda-bindings runtime/NVRTC, direct cuEST0.2.50
C-ABI, three native FP64 cuBLAS GEMMs and small packing/fused-denominator/
signed-reduction kernels. No new automatic SAPT option/default/E2 change.
No CuPy, `cuest.bindings`, `cuest_scf` or nvmath import/dependency in this path.

LOCAL archive independently rehashed:206185073 bytes,
SHA256a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d.
DFMO headerc93868fcc871fd1fafea374661ad9bc188a3b40646e54eb6b906b5353f06775b,
paramsheaderce1b4313eab84d3027d0ad5202ca17e49baf41573f9c700c9089c6535051d882;
contextheader eea9ba0be5a9f5739ed947f8b97d4819d7a4f019d52629a33832028d14c6a863
defines **status(handle,uint32_t*)** Major/Minor/Patch getters. No native call
or hidden handle creation occurred. Full streaming archive inventory task
bb9742d5b is separate read-only provenance, not an SDK install.

`sapt_cuest250_identity.py` requires exact canonical archive/whole header tree,
CUDA13 nativeSHA65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c,
native getter tuple0/2/50 from an explicit valid same-process handle, capability
symbols, actual mapped function ownership and native/CUDA file hash+inode/device,
and an explicitly rooted frozen-hash CUDA13.3 closure manifest. Python cuEST
bindings cannot bypass this direct-C-ABI contract. This is distinct from
the old CUDA/E2 build proof, which did not enable cuEST in Psi4.
An inode/device negative went RED, then the explicit mapped-file identity
comparison was added. **9 synthetic identity tests PASS,0.024s**.

The actual kernel source bodies execute through a LOCAL C++ qualifier shim
on one serial CPU thread in tests, with NumPy GEMM/factor stand-ins. **13 CPU
serial-body/transport/setup tests PASS,0.672s**, including layoutsC/F,
edges1/3/500, changed inputs, signed occupied-pair/total equations, cuBLAS
descriptor, parameter/workspace transport, failed cleanup/no retry and guards.
**This is NOT NVRTC compilation, CUDA parallel execution, native cuEST/cuBLAS,
or physical SAPT validation.** NVRTC/device/rank/eigenvalue/scientific gates
remain unpassed. Static reviewd28768ac is pending; new source not committed yet.

New plan contract explicitly rejects defaultQR: full Coulomb1/range-separated0,
MATRIXPOWER1/relative1/cutoff1e-12, no denominator clamp. These declarations
do not prove native rank/negative-eigenvalue semantics or basis/gauge parity.
Actual approved live-handle/basis/plan factory, complete correct Psi4 import,
optional dependency lock and mapped CUDA closure remain missing. No borrowed
window: owner14250361 is terminalTIMEOUT0:0 per its official watcher. No
remote probes, resource/extension/retry/cancellation, dependency solve/install,
old-root/stage/queued-job changes or speedup claim.

LOCAL manifest inventorybb9742d5b completed0 (no extraction): **59 headers**.
Full JSON receipt `.pi/tasks/01a154d8-a459-768d-83bd-c57c6a5dc42c-4065461/bb9742d5b.output`,
SHA256496f97fa3811c6704c31079c9414b94bb37192ec840bc5d20d24bb22bfb77ec1.
Header-tree compact sorted JSON with FULL archive-member keys:
SHA256644fb4d1b64862e91d9278153aaeb78273a38717209e9ff91d10f4f9f1cecc85.
CUDA13 shared member88186400 bytes/65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c;
static member80756594 bytes/3e4e26b2fb9c3f45360814d1f31314ca19a609de528e3c8ea6c401b515dddcf4.
Symlinks `.so -> .so.0 -> .so.0.2.50`. Full record/path/schema sent directly
to PySCF and FDDS; Python binding artifact record remains null (unused by
the new ctypes backend, not a verified external package).
This archive/header proof does not certify matching runtime CUDA closure,
actual native getter calls, rank/eigenvalue or GPU scientific correctness.

Peer header serialization independently reproduced from the same59 records:
sorted include-relative POSIX path +NUL+lowercase fileSHA+newline =>
**3d26a7ef607d02ee294f66f11f1374f3badbd83430f5be28ed0a05e6c8d73d18**.
Our644fb4d1... compactJSON values are `{bytes:int,sha256:str}`, not bareSHA.
Both peers independently reconciled both algorithms; no header-byte discrepancy
or cross-worktree permission bypass. No remote query occurred.

Reviewd28768ac raised four concrete source issues, now addressed locally:
mapped library hashing uses one openedFD/fstat and before/after fd+path identity;
mapped/function ownership is checked BEFORE native version getters receive a
handle; execution closure is separate from successful cleanup completion,
with failed allocations retained and successful destroys cleared for explicit
cleanup-only retry; negative nested subview offsets are rejected. Production
`cuLaunchKernel` now receives `ct.addressof(parameters)` while typed argument
holders stay alive. CPU ABI spies cover that actual marshalling function,
all11 DFMO arguments/uint64 counts/kind49/workspace16/32-byte layouts,
and denial before getters on map failure.

Additional mixed-toolkit negative went RED: a self-consistent CUDA12 cuBLAS
record with CUDART13 was initially accepted. Explicit CUDA13 major guards now
reject it even if declared in a hash-consistent inventory. Same-FD replacement/
inode/device negatives and cleanup/borrow lifecycle cases also pass.
Latest local runs: **16 CPU serial-body/transport/setup PASS,0.924s**;
**11 synthetic identity PASS,0.030s**; **58 build guards PASS,4.235s**;
**96 legacy runtime guards PASS,4.49s**; AST/diff checks PASS.
Scoped follow-upd52b40b1 pending. Exact cuda-bindings package/ABI lock, real
NVRTC compilation/parallel execution and native cuEST calls remain UNRUN.
FDDS's future builtin tuple/provider remains uncompiled here; we do not claim
or invoke it on frozen old cores or impose Disp0 policies on RKS/PySCF.

Scoped follow-upd52b40b1: **no scoped blocker found** on the four review
fixes and launch marshalling/transport tests; read-only, not execution proof.
The identity source now also requires all59 canonical headers and BOTH
documented aggregate encodings. Executing its real `canonical_headers()` on
the LOCAL archive passed with644fb4d1... and3d26a7ef... exactly, without
extraction or native library calls.
Final rerun: **16 CPU serial-body/transport/setup PASS,0.866s** and
**11 synthetic identity PASS,0.034s**. Previous58/96 guard results retained.

This source checkpoint replaces, rather than supplements, the optional CuPy
backend. Remaining unresolved execution prerequisites are unchanged: repaired
complete Psi4 installation/import, pinned cuda-bindings/library environment,
audited same-process live-handle/basis/plan factory, explicit MATRIXPOWER/
relative-cutoff/rank/negative-eigenvalue parity, real mapped CUDA closure,
NVRTC compilation and actual device signed-component gate. All GPU/SLURM
allocations remain terminal/unborrowed; no new resource action is authorized
or performed. Phase2 broader F-SAPT is still blocked by the physicalGPU gate.

## Latest checkpoint — 2026-10-21 12:27 Eastern

**Lease terminal update:** watcherb53cf45bd observed CPU14254131
**PREEMPTED0:0, elapsed02:00:05**, at2026-10-21T16:50:57Z.
This is allocation/accounting status, NOT a successful build/readiness marker.
Step0 FAILED1:0 after43:16; validation1 FAILED1:0 after52s;
validation2 FAILED1:0 after65s; read-only diagnostic3 COMPLETED0:0 after23s;
validation4 FAILED1:0 after81s (its effective audit and standalone E2 import
passed, but joint-stage identity did not). No active Disp0 CPU holder remains.
No retry, replacement allocation, GPU request or installation repair was
submitted. All original and later evidence stays preserved.

Existing FDDS14250361 reuse inquiry answered **LOCAL ONLY** via workmux to
owner `saptdft-cuest-disp` and parent: **BLOCKED/not READY; do not hold GPU
idle for Disp0**. No allocation probe/borrow/step requested. Owner must verify
its own live lease and assign any future serialized window.

Current integral seam in `sapt_jk_terms_ein.fdisp0`: host DFHelper
`DIRECT_iaQ`, ten orbital-space transformations (`Aar,Abs,Bas,Bbr,Cas,Cbr,
Dar,Dbs,Ear,Ebs`), host staging/packing, then three packed FP64 default-graph
DF products in `sapt_dispersion_gpu.BlockContractions`. Denominators and
reductions remain host operations. The special E2 branch is not an integral
engine and does not replace DFHelper. A future GPU-focused producer would
require an audited existing cuEST three-center/auxiliary-metric endpoint,
Python/E2 blocked orbital transforms and an explicit device-factor lifetime/
layout contract before feeding those contractions. That producer/residency
work is NOT implemented or validated. Preserve basis/metric/spin/exchange
conventions and independent signed component references. Greater-than10x
end-to-end speedup is an optimization goal, not a measured or promised result.

Local follow-up plan: `devtools/disp0/dfmo_device_plan.md` maps all ten
coefficient-pair factors, Far/Fbs combinations, packed rank-one corrections,
defaultDFHelper metric−1/2 and threshold settings, memory/gauge/layout and
validation requirements. The owner-reported0.2.50
`cuestDFMOIntegralsCompute` endpoint is promising; direct archive/header
identity inspection here is still pending the local path/hash handoff.

Read-only exact064 E2 auditdd2a4fd55fec87804bfc193d4393aefb1 finds **no
supported Python external CUDA-buffer import/ownership seam**.
`RuntimeGPUTensor` is native-only; DeviceShadowMap owns its allocations;
default replay starts with empty device validity and ends in D2H flushing.
No existing zero-copy producer contract was found. Direct inspection confirms
those storage/executor findings. Generic Python-visible device storage,
ownership, stream/freshness and safe-fallback plumbing would be needed;
no C++ dispersion algorithm, unsafe host-tensor pointer alias, global executor
change or implicit zero-copy claim is proposed. Parent/owner notified LOCAL
ONLY; not READY and no GPU window requested.

## User-authorized alternative CuPy prototype — LOCAL implementation in progress

User explicitly authorized trying CuPy-owned device buffers and CuPy Python
dispersion instead of E2-only for a new experimental backend. This does not
change default/E2 dispatch or authorize installing into shared environments,
new resources or a GPU-holder window.

New local building blocks `sapt_dispersion_cupy.py` and `sapt_cuest_dfmo.py`
implement blocked FP64 packing/V-X-W/negative denominators/occupied-pair
reductions and the existing caller-owned cuEST DFMO Python API. Ten coefficient
pair factors are produced in r/s slab batches; only coefficient slabs upload,
not host DF factors. Explicit native synchronization precedes workspace/owner
release. Host-returned producer factors are rejected rather than relabeled
GPU. Native plan/basis construction remains a caller-factory prerequisite,
and exact layout/metric/native-package/runtime provenance remains unverified.

CPU-only NumPy/ctypes fake-transport regression command:
`python3 -B -m unittest discover -s devtools/disp0 -p test_cupy_backend.py -v`:
**6PASS,0.063s**. Independent scalar signed-component and occupied-pair oracle,
blocks1/3/500, changed inputs, crossed W indices, C/F output-layout fixtures,
strided coefficient uploads, budget/device/native-error and missing-dependency/
host-fallback rejection exercised. These are NOT actual CuPy/CUDA/native cuEST
or physical SAPT tests. Source review and canonical setup integration pending.

Optional dependencies documented in `devtools/disp0/cupy_backend.md`:
CuPy (`cupy-core` from conda-forge or the `cupy-cuda13x` wheel, not both),
compatible `cuest.bindings` and CUDA Python dependencies. No solve/install or
new lock has occurred. Stable CuPy docs list CUDA13.2, not our13.3 combination,
so compatibility is a runtime gate, not assumed from the13x package name.

Canonical integration now adds explicit `dispersion0_cupy(...,provider_factory=...)`
and a private optional callback after unchanged orbital/correction setup,
before DFHelper construction. It shares the original occupied-pair placement
and signed-sum helper with the default path. HF/ordinary dispersion,
fragment/S-inf, E2-GPU conflict and C1 guards precede GPU/factory access.
Zero active spaces bypass the factory; normal default dispatch remains selected
unless this new programmatic entrypoint is explicitly called. No automatic
driver option or audited native plan constructor is supplied yet.

Expanded CPU-only suite: **10PASS,0.498s**, plus **58 existing optimized-build
guards PASS,1.849s**; Python AST syntax/diff whitespace checks PASS.
An initial independent Qbr boundary fixture failed (max231.525 in
synthetic matrix units) due to two incorrectly transcribed *fixture*
terms; corrected those two terms to the existing source expression, without
changing setup equations or tolerances. Noncommuting densities and nonsymmetric
K_O then pass at1e-12, with asymmetric freezing, caller-cache invariance and
DFHelper failure sentinel. Source reviewsd7669487 andd020e303 remain pending.

Reviewd7669487 confirms static V/X/W/sign/permutation consistency and identifies
three prototype issues, now fixed locally: cleanup no longer masks original
native/body errors (secondary failures become notes, failed adapters invalidate);
memory admission explicitly estimates live buffers and requires aggregate
native persistent allocation bytes, reports reserve separately and considers
cached pool reuse; timing renamed `blocked_execution_seconds`, excluding
upstream orbital setup and native-plan creation/cleanup. **13CPU/mock tests
PASS,0.518s**, including combined native/sync/destroy errors and realistic
used/reserved/native-memory fixtures. Follow-updf94c249 and callback review
d020e303 pending. No actual native/GPU tests, installation or allocation.

Callback reviewd020e303 found inherited execution metadata surviving a
zero-dimensional call. Fixed `dispersion0` to remove only the copied execution
report before dispatch, preserving the caller's original cache. Occupied-empty
and virtual-empty tests preload a stale GPU report and now return zero signed
energies without factory execution or stale provenance. Added original-default
DFHelper sentinel and direct fragment/callback conflict checks. The default
sentinel initially hit a missing `_mat` function in the CPU test namespace
(harness NameError, not native/scientific execution); added that fixture stub.
**14CPU/mock tests PASS,0.610s**. No tolerances changed; full native/physical
regressions remain unrun with the new source and cannot be replaced by fixtures.

Final scoped reviewdf94c249 identified an enclosing-`except` corner case:
`sys.exception()` could select an unrelated caller exception and swallow a
cleanup failure. Added a regression that went RED (cleanup not raised/caller
exception modified), then replaced ambient exception inspection with a local
primary-error variable populated only by this compute call. Both synchronization
and parameter-destruction cases now raise correctly; original native errors
still retain secondary cleanup notes. No native operation was executed.

Final local commands:
* `python3 -B -m unittest discover -s devtools/disp0 -p 'test_cupy*.py'`:
  **15PASS,0.581s**.
* Optimized-build guard discovery: **58PASS,3.105s**.
* Existing p4_einv2 Python, `-B -m pytest --noconftest -q
  tests/pytests/test_disp0_runtime_contract.py`: **96PASS,3.93s**.
* Python AST syntax on all three changed scientific modules and
  `git diff --check`: PASS.

This is a reviewed **experimental implementation checkpoint**, not a ready
CuPy/cuEST runtime. Native AO/RI basis/plan factory, exact dependency/artifact
lock and loaded-library admission, cuEST output strides/metric/gauge physical
proof, corrected complete Psi4 installation and the signed actualGPU gate
remain separate blockers (task27/22/4). No source pushed to cluster, no
dependency solve/install, no GPU/resource query/window and no performance
claim. Existing E2/CPU behavior still requires a matching-stage physical
regression after the installation blocker is fixed.

The optimized064 native/Python Einsums and whole Psi4 installation finished
inside CPU allocation14254131. Original build step0 remains FAILED in its
post-build audit; completion of compilation is not runtime readiness.

Read-only validation613511b0dfe914ed4bd241040dc6fc1a25277570 ran as
14254131.2, taskbd94438a1: **FAIL**. Independent status:
audit1, installed identity1, API SKIPPED125, snapshots0, binary invariance0.
The installed Psi4 refuses import because its generated version is undefined
(`header.py` explicitly requires valid Git tags). No version-check bypass,
Python metadata overlay or binary change was attempted. A separate audit
classification also rejected a compiler-looking auxiliary argument; a bounded
read-only same-allocation diagnosis is in progress. All raw errors retained in
`P/runs/saptdft-cuest-disp0/validation-613511b0dfe914ed4bd241040dc6fc1a25277570-14254131-2/`.

Current candidate is **not READY**. Actual installed API, physical signed
components and CUDA numerical/device gates remain unpassed; Phase2 has not
started. No GPU request or replacement CPU allocation follows automatically.
Existing CPU lease ends12:46:01 Eastern; previous failed roots stay unchanged.

Read-only diagnosticb12a679f0 (14254131.3) confirms the undefined installed
metadata (`undefined+770ecf9`), shallow source=true, Git describe=770ecf9,
and no tags. Source fallback requires release1.11. A future proper build must
fetch and verify real annotatedv1.11 identity/history BEFORE version generation,
not patch installed Python or disable `header.py`. Locally inspected tag object
75fa0e14aa16a868a00eced50eaaa2f8658aa806 peels to
f16f4ea2ff2351cc2ab877d9b3887fcf81c2a670.

The exact remaining audit false positive is the CMake configure rule's
`-DCMAKE_CXX_COMPILER=<approved GNU executable>`, not an executed wrapper.
Narrow recognition validates only four explicit CMake compiler-definition
keys against approved actual compiler identities; unknown wrappers/foreign
compiler settings remain rejected. Local optimized-build guards **55PASS,
1.852s**. This does not repair or accept the undefined installed Psi4 stage.

Actual validation5bcf985b455974f7a60efc7a924cad7f4156797d as14254131.4,
taskb54dedd95: **effective build/header/link audit PASS**; installed identity
FAIL solely at Psi4's undefined-version import check, API SKIPPED125;
binary snapshots/invariance PASS. Separate standalone E2 diagnostic imported
the installed cp313 Python core with one mapped installed
`libEinsums-abi1.so.1`; sealed world reports exact
06404fa69ce9346443d7d4f19a250f6db9f0f62c, GNU14, E2 major2.
RuntimeTensorD, RuntimeTensorViewD, MemoryPool and asarray are present.
This is standalone import/symbol evidence ONLY, not joint-stage identity,
operation/API semantics, physical correctness or CUDA execution.

Offline corrective contract4cf3eb280d51b47409369d4559d05b5f60d81378 adds
the version-history guard before SDK/dependency work; **58 local tests PASS,
3.346s**, Bash syntax PASS. Executing the real upstream version generator
against this clean LOCAL full-history checkout passed and generated
1.12a1.dev660+4cf3eb2 with the verified annotated1.11 identity. No remote
stage was changed. Next proper installation must regenerate metadata using
verified source history through the upstream build/install process, with
preserved old roots and complete-package/binary provenance; do not copy a
Python-only overlay, patch version strings or disable the import guard.

## Ownership and immutable launch pins

Dedicated worktree: `saptdft-cuest-disp0`; branch: `saptdft_cuest_disp0`.
No changes to sibling worktrees, coordinator builds, environments or run roots.

* Base `satpdft_einv2`: `1592851675cbd8ce6898eca969976a6e7becdb24`
  (verified HEAD and ref at start).
* Required reference `fisapt_fdisp_blocked`:
  `907f6d75690c071dd9b452e3f0a82abb7ba317ea` (verified).
* Optional plumbing `saptdft_cuest` and `saptdft-cuest-cpks`:
  both `ca21ef8645a73263281480454b730085506adca5` at inspection.
* Shared merge base with the two source histories:
  `288fe50fc673747944a77b08af250fb0ce557c8f`.

No tracked AGENTS.md/CLAUDE.md or ancestor guidance found at initial inspection.
`workmux send saptdft-cuest` succeeded with ownership/readiness notice.
No GPU allocation requested/submitted yet.

Parent runtime handoff received: Phoenix prefix
`P=/storage/project/r-cs207-0/awallace43`, env
`P/miniconda/envs/p4cuest_sapt`; cuEST 0.2.50
`P/software/cuest-0.2.50-a06e8148/lib/13`, explicit cuda13 toolkit
(13.3.73, runtime 13030), conda GCC/G++ 14.3 wrappers and CUDA LibXC 7.1.2.
Recipe pin `8af4a8bf034c16c0c01821b8442673db0f54514f`; scientific cuEST
pin `ca21ef8645` remains reference-only, not an ABI claim for this branch.
The parent's coherent dependency stage has native/Python Einsums 1.1.3.
This launch base actually calls `RuntimeTensorD`, `reshape_view`,
`MemoryPool`, `einsums.graph`, `outer_sum` and `direct_division`; check those
APIs explicitly rather than infer compatibility from the path's name.
Parent clarification: that 1.1.3 stage is **not applicable** to this branch.
Do not downgrade or mix native/Python major versions. Investigate a dedicated
coherent CUDA-enabled Einsums2 build, preferably at the CPU-oracle revision
`d17bb76e`, and test used operations, views, pools, graph replay, layout and
division semantics. Toolchain recipes are candidates, not validated Einsums2
compatibility. Readiness coordination remains mandatory before any GPU job.

Allocation constraint: **do not submit yet**. Proposed correctness gate is
one H200 / 4 CPUs / 32G host / 30min / embers / gts-cs207-chemx, subject to
another readiness checkpoint (CPU outcomes, dedicated GPU build/import/ABI
proof, comparisons/tolerances, memory/runtime). No parent holder is available.
A separate bounded build job, if needed, must first be proposed. Do not
repair workstation drivers or change other allocations.

## Plan and gates

1. Inspect actual HF dispatch, einsums_v2 APIs, blocked FISAPT equations,
   tests and available isolated runtimes.
2. Review and import the optimized blocked reference **before GPU work**.
   Prefer minimal faithful port/cherry-pick over broad divergent histories.
   Preserve pair blocking and the einsums_v2 memory order; record exact commits.
   Validate reference and commit this as an explicit prerequisite milestone.
3. Port only necessary committed GPU endpoints, then implement Python-level
   FP64 blocked SAPT0 dispersion and exchange-dispersion for
   SAPT_DFT_FUNCTIONAL=HF. Preserve non-HF and CPU paths.
4. Validate signed components, sums, dispatch, block/edge cases and multiple
   dimers/bases against CPU oracles; perform actual GPU checks with endpoint
   evidence and bounded memory. Record source/build/runtime pins and commands.
5. Only after that numerical gate passes, inventory all F-SAPT terms and
   implement/validate Python Einsums GPU coverage and fragment sum rules.
   If GPU validation is unavailable or fails, stop before Phase 2.

Initial tolerance policy (to be confirmed against oracle precision before
interpreting results): synthetic FP64 kernel comparisons rtol=1e-11,
atol=1e-12 Eh; physical signed energy components atol=1e-8 Eh and rtol=1e-8,
with discrepancies retained individually. Compare unrounded values where
available; printed reference precision cannot justify tighter assertions.
No tolerance relaxation to hide failures.

## Current status

* Lineage verified, worktree initially clean.
* Reference imported as `fb430094cc` via
  `git cherry-pick -x 907f6d75690c071dd9b452e3f0a82abb7ba317ea`.
  This exact, clean three-file cherry-pick imports the native CPU oracle,
  BLOCKED/PAIR options and frozen-core regression test. No new native GPU
  algorithm is being introduced.
* Reviewed equations: `V=Aar.Abs`; exchange `Aar.Fbs+Far.Abs`, plus
  permuted `Bas.Bbr+Cas.Cbr` and four rank-one overlap/potential terms.
  Denominator `ea+eb-er-es`, factors +4 (dispersion) and -2 (exchange).
  Pair fallback retained for parallel/perpendicular link-orbital terms.
* Deliberately excluded `3c06680fea` and `e8478e2064`: these touch the
  divergent v1 Python module. The launch base already has the v2 Q-major
  packed factors, bounded staging, localized factors and block reductions.
  Replacing that module would regress the required route.
* **Reference numerical gate PASSED**: 2 physical tests, frozen-core off/on,
  same localized ethane-water/cc-pVDZ wavefunction for PAIR vs BLOCKED edges
  1/5/500. Both signed Disp20 and Exch-Disp20, combined Disp_AB and partition
  sums agree at rtol=1e-11/atol=1e-12 Eh. No tolerance changes.
* Initial scalar Phase 1 implementation checkpoint `1e25b5e89a` is
  **CPU-validated: 26 passed**, 64 signed physical comparisons retained.
  CUDA numerical/device-shadow correctness remains unvalidated.
  opt-in `SAPT_DFT_MP2_DISP_ALG=EINSUMS`, with HF-only validation and
  `SAPT_DFT_EINSUMS_GPU=false` default. Existing SAPT/FISAPT defaults and
  non-HF paths remain selected as before. Canonical mode shares fdisp0's
  equations with active occupied slices, without localization caches.
* Kernel-local `sapt_dispersion_gpu.BlockContractions` uses three packed
  rank-2 einsum descriptors in default graph replay, cached per edge shape.
  No global executor changes and no new native algorithms. It reports
  placement as **planned**, never device proof, and accounts for packing,
  replay including transfers, other host work and total elapsed time.
* GPU implementation has not executed on CUDA. No GPU correctness claim.
  Fragment-GPU and S-infinity dispersion are explicitly rejected in this
  initial scalar checkpoint rather than silently taking a CPU route.
  Fragment dispersion/sum-rule coverage is still required before claiming
  the full Phase 1 gate; no broader F-SAPT terms have been implemented.
* Phase 2: gated, not started.
* GPU work is conditionally authorized but runtime-blocked. Failed CPU jobs:
  14182584(Catch2),14193865(Apiary resource headers),14196330(completed both
  builds but omitted E2 Python installation; post-install import failed).
  Same-job build-tree import/ELF metadata now PASS with one E2d17/GNU14 world;
  this is not an installed-runtime/numerical pass. Artifact-only installation
  into a fresh prefix passed whole-artifact byte/ELF audit in59.5s, but the
  **installed-prefix mapping proof FAILED**: E2 native resolved to the old
  build tree. Tiny new-prefix API probe did not run. Read-only ELF diagnosis
  pending; no loader bypass, binary patch, overlay or automatic retry.
  Preserve all failed roots/statuses and failed proof artifacts.
* No GPU job has been submitted. Available conditional approvals: one
  H200/4CPU/32G/30min signed/device gate, then one H200/8CPU/512G/up-to6h
  timing job, bothinferno/gts-cs207. Quoted combined ceiling$9.19 requires
  fresh capacity/rate/route checks. No additional CPU allocation or retry.
  Fixed-suite HF policy-controlled timings are distinct from parent PBE0-D4;
  actual timed dispatch remains unverified, diagnostics stay separate.
  Target2026-10-20 08:00 America/New_York is not guaranteed.

Parent scope agreement: Phase 1 is explicitly **GPU contractions with CPU
elementwise/reductions**, not fully GPU-resident dispersion. Use default
`Graph.execute` only in the new local kernel route; do not globally change
existing executor semantics. No new C++ dispersion or elementwise algorithms.
Test packed/strided layouts, pool/view lifetimes, repeated replay and shadow
freshness against signed components. Account for host work and transfers in
end-to-end timings. After the real GPU numerical gate, inventory remaining
F-SAPT GPU API gaps and propose a Python-level approach before expanding.

## Launch-base dispatch and GPU capability review

* At launch pin1592851675, `energy("sapt(dft)")` + `SAPT_DFT_FUNCTIONAL=HF` correctly skips
  FDDS and DFT exchange-dispersion scaling. Plain scalar dispersion still
  calls native `FISAPT.disp` or `core.sapt`, even from
  `sapt_mp2_terms_ein.py`. Only the SAPTDFT F-SAPT branch calls Python
  `sapt_jk_terms_ein.fdisp0`. There is no `einsums_v2` option.
* A scalar Python entrypoint needs active canonical occupied coefficients,
  separate frozen-core counts and identity occupied rotations (or an explicit
  canonical mode), not accidental dependence on fragment localization caches.
  Non-HF FDDS and explicit D3/D4 aliases must retain existing dispatch.
* Local Einsums source pin `d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62`
  (tracked files clean; untracked install/). Its current local build has
  CUDA/HIP/GPU support OFF; Python imports report 2.0.0.
* Reviewed GPU graph dispatcher: packed column-major rank-2
  `ein.einsum` descriptors and strided-batched GEMM can dispatch to FP64
  cuBLAS. Merely calling `ein.linalg.gemm` is **not** evidence: its
  GemmDescriptor is excluded by this GPUPlacement implementation.
  The current driver's custom SequentialExecutor also bypasses the default
  device-shadow replay. These must not be labelled GPU execution unchanged.
* Python exposes host RuntimeTensors, not explicit device arrays. Default graph
  replay can retain shadows across consecutive supported contractions, but
  final replay flushes to host. Denominator/division, permutations and general
  reductions lack GPU dispatch in the inspected library; an initial
  GPU-contraction/CPU-elementwise hybrid must be reported as such.
  Inspect cuBLAS trace/profiler `gpu_dispatch` rather than placement alone:
  replay can legitimately decline to CPU.
* Committed Psi4 generic `core.cuest_chain_gemm` is another candidate, from
  `f5ed28020cc5fb65e86a9391460bd827d00c59e1`, reviewed at `ca21ef8645`.
  It accepts host `core.Matrix` operands, uploads factors, retains chain
  intermediates and downloads requested results; it is not a public persistent
  device tensor API. It depends on the committed cuEST runtime/handle plumbing
  absent from this base. No broad history has been merged to obtain it.
* Einsums2 CUDA build audit at exact `d17bb76e`: CMake >=3.25.2, C++20,
  native CUDA toolkit; build shared native library and Python `_core`
  together. Dependencies include Apiary 1.x code generator, fmt 12,
  mimalloc >=3.3,<3.6 (excluding 3.4.0/3.4.1), spdlog 1.17 source,
  HDF5, OpenMP, BLAS/LAPACK and a Python-3.13-capable pybind11.
  `EINSUMS_BUILD_PYTHON=ON`, `EINSUMS_WITH_CUDA=ON`, explicit arch 90,
  HIP/MPS/mock OFF, split binding TUs for bounded memory are candidates.
  No source rule excludes CUDA13.3/GCC14.3, but compatibility is untested.
  No upstream source change is yet justified merely to import/build.
* Native E2 soname is `libEinsums-abi1.so.1`, not inferred from major version.
  `einsums.sealed.world()` after importing the dedicated Psi4 stage reports
  2.0.0-trunk, exact `d17bb76e`, compiler GNU14 and a single mapped library
  `$HOME/gits/Einsums/build/lib/libEinsums-abi1.so.1`. No 1.x library mapped.
  This validates CPU runtime coherence, not CUDA execution.
* Stock Python does not expose GPUPlacement's force/threshold parameters.
  `pm.enable("GPUPlacement")` does not force cost-model-rejected small
  blocks onto the device. Tests must retain and label CPU edge-block
  fallbacks and require actual eligible contractions plus cuBLAS trace.
* Pool `pinned=True` is host resident-page policy, **not CUDA-pinned memory**.
  `outer_sum` assumes contiguous vector elements in its inspected loop;
  energy vectors used by the kernel must be packed/stride-one. Do not
  generalize its semantics to arbitrary strided vectors or broadcasting.

## Execution log

* `git status --short`, `git branch --show-current`,
  `git rev-parse --verify` for HEAD and all named source refs: PASS.
* Read-only source-history diff/log inspection: prerequisite tip changes
  FISAPT native reference plus its tests; divergent earlier histories also
  contain einsums changes, so a blind broad merge is inappropriate.
* Runtime preflight: system `python` absent; system python3 has no Psi4.
  `$HOME/miniconda3/envs/p4_einv2/bin/python` imports release Psi4 1.10 and
  editable Einsums v2 from `$HOME/gits/Einsums/build/lib/einsums`.
  Release Psi4 lacks `FISAPT_FDISP_ALGORITHM`; it cannot validate the import.
* `bash devtools/disp0_build_cpu.sh`: background isolated build started in
  this worktree's `build_disp0_cpu`, reading p4_einv2 dependencies without
  installing into or modifying that environment. Native ENABLE_Einsums=OFF
  follows the base build (Python Einsums remains available independently).
* Local `nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader`:
  FAIL, NVML driver/library mismatch (library 580.178). No local GPU
  correctness evidence; Phoenix allocation coordination will be required.
* Isolated CPU build: PASS (background task b748623db, exit 0), native
  scientific source at `fb430094cc`; stage
  `build_disp0_cpu/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so`,
  SHA256 `cbc0ff81eadbab331193c4ad9fb35c0ba70b70e8edef0606713c87dd0188e72a`.
  Einsums2 Python native module SHA256
  `94a7f03f3ee0402de4b2ba43f16c3093dfd869ee7fc0184dfe8969638e6f6c02`.
* `bash devtools/disp0_test_reference.sh`: launched after successful build;
  first attempt FAIL before numerical execution (task b75d4fd27):
  generated `--psiapi` appends `$PYTHONPATH`, which was unset under nounset.
  Fixed runner to define an empty PYTHONPATH before evaluating --psiapi.
  Second attempt ba57c4f18 failed before numerical execution because source
  cwd shadowed the staged package (ImportError: no source-tree core).
  Runner now changes to its isolated run directory before Python/pytest.
  Both are harness failures, not numerical pass/fail. Full rerun launched as
  b2f6b3d0a: **PASS**, exit 0, `2 passed, 9 deselected in 9.08s`.
  Results: `build_disp0_cpu/reference-check-9PFhJqF7/results.xml`.
  Source scientific pin `fb430094cc48d55c3479dae0040276c6c6dc6dd1`,
  dedicated core/import proof printed before pytest, 2 CPU threads.
  This is the prerequisite CPU reference gate, **not the GPU gate**.
* `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1
  $HOME/miniconda3/envs/p4_einv2/bin/python devtools/disp0_einsums_api_probe.py`:
  PASS, exit 0 (task b693753c0). Tested FP64 packed contractions,
  occupied-fastest rank-4 denominator construction/division/reduction at
  shapes 1x1, 3x4 and 2x1; default graph replay with a strided submatrix
  and mutated input; view keepalive after dropping its pool-carve owner;
  rejection of reset with a live view and release after view destruction.
  This initial dependency probe imports installed release Psi4 first; it is
  not a scientific source-pinned calculation. Repeat with the dedicated
  scientific stage: PASS (b05a7e7ad), identical five checks, Psi4
  1.12a1.dev615 imported from `build_disp0_cpu/stage/lib/psi4`, Einsums2
  from the exact CPU-oracle dependency build.
* Additional runner isolation fix: run Python stdin/pytest from the new
  run directory, not the repository root. A direct audit command from the
  source root reproduced source `psi4/` shadowing the staged native package.
  The successful stage API probe and sealed-world probe use a non-source
  working directory. No scientific mismatch is attributed to this harness issue.
* Canonical checkpoint incremental dedicated build: PASS (bf9409911).
  Source is currently the uncommitted Phase 1 checkpoint above; no claim of
  a finalized source pin until its tests pass and it is committed.
* `bash devtools/disp0_test_reference.sh
  "$PWD/tests/pytests/test_saptdft_disp0.py"
  "$PWD/tests/pytests/test_saptdft.py::test_einsum_terms"
  "$PWD/tests/pytests/test_fisapt.py::test_fsapt_fdisp_blocked_matches_pair"`:
  b55c1dc04: **PASS, 15 passed in 18.50s**, 3 JUnit-schema warnings.
  Results: `build_disp0_cpu/reference-check-IvAuRbOB/results.xml`.
  XML parsing confirmed all 36 signed discrepancy records were retained.
  Runner now selects legacy JUnit schema to avoid record_property warnings.
  This run used the uncommitted checkpoint over `69e8299046`; it is not
  represented as a finalized source pin. Physical cases: water-Ne at two
  distinct geometries/cc-pVDZ spherical/unfrozen, ethane-water/6-31G*
  Cartesian/frozen. Each signed component is compared to native FISAPT for
  edges 1/5/500, eager and independent default CPU graph execution. Every
  discrepancy is retained in JUnit properties. Additional tests poison and
  mutate graph inputs across cache hits, pack strided views, verify pool
  reset fails with live replay borrows and succeeds after release.
  Maximum absolute discrepancies (Eh), Disp20 / Exch-Disp20:
  water-Ne geometry 1: 3.66e-19 / 4.40e-20;
  geometry 2: 1.42e-19 / 1.06e-20;
  ethane-water: 7.81e-18 / 9.13e-18.
  Both the non-HF regression and imported physical blocked-reference tests
  passed in this run. No GPU numerical/device-shadow test has run.

## Dedicated CUDA2 build proposal (not submitted)

Use exact upstream Git pin `d17bb76e` (present in local origin/main remote
tracking and pinned fork refs), a new environment without the conda Einsums1
package, separate build/install directories, and a fresh dedicated Psi4 stage.
The Python-only dispersion route needs no native cuEST algorithm or older
Einsums stage. SCF/DFHelper and non-dispersion work remain CPU in this first
implementation; do not describe it as a whole-run GPU implementation.

A CPU-only embers build job can configure CUDA arch 90 explicitly, compile
the CUDA-enabled E2 library/Python together, rebuild this branch's Psi4, and
run CPU/API probes without borrowing a GPU. Candidate request: 8 CPUs, 48G,
2h, gts-cs207-chemx, explicit embers; exact-resource test-only and parent
coordination still required. Then request the separately coordinated
one-H200/4CPU/32G/30min numerical gate only after build/import/ABI proof.
The known CUDA13.3/GCC14.3 prefix is a toolchain candidate, not a passing E2
configuration. No job or shared environment has been changed.

Parent approval received for **ONE CPU-only dependency/build job**, after
canonical CPU regressions are green and the source is committed/pinned:
8 CPUs / 48G / 2h / embers / gts-cs207-chemx, cpu-small subject to actual
test-only routing. This is not a benchmark rerun. Use isolated immutable
source/build/install/environment roots, explicit sm90/CUDA/GNU pins and
persistent job-owned scratch; TERM-safe honest outputs, no automatic retry,
inferno or shared mutation. Validate test-only/headroom, record the actual
QoS/account/job ID and send the parent a receipt. GPU-enabled import failure
on a CPU node must not be masked. CPU/API/header/ELF/import checks establish
preparation only. H200 GPU numerical gate still requires separate readiness.

Prepared `devtools/disp0/{environment.yml,condarc,cpu_build.sbatch,
cpu_build_payload.sh,cuda_header_abi.cc,runtime_provenance.py}`.
Syntax/AST/YAML/diff checks PASS; environment solve/build not yet attempted.
The recipe uses job-owned package/cache/env/build/install roots, no conda
Einsums1 or release Psi4 package, explicit native CUDA13 header/runtime check
(13030) and GCC14.3 frontends, E2 shared/Python build together, strict import
pairing and CPU-only tests. No native E2 unit-test suite is built in this
bounded recipe; used Python operations and physical signed comparisons are
the focused checks. TERM handling propagates failure and cannot write a
CPU preparation success marker after a signal. No retry or GPU submission.

## Read-only review follow-up before source pin

Review d0a5ab9b8243c1321ec536956c8a16bca found no signed-equation or
frozen-energy slicing defect, but identified three P2 gaps in the green
checkpoint. Retained here rather than treating the earlier tests as exhaustive:

1. Non-aligned DF/compute edges could retain nine shapes, exceeding a four-shape
   charge. DF extents now align to compute extents; a pure planner test includes
   the exact 93/47/24 reproducer. Both host work and graph packs charge all four
   possible main/final-edge shapes.
2. Final Far/Fbs construction matrices remained live with packed factors.
   Construction now uses bounded staging slabs, deletes its temporaries before
   packing, charges all eight stage arrays and includes the rank-one rows in
   packed-factor memory costs.
3. Direct exported `sapt_dft()` bypassed unsupported-mode validation. Both
   entrypoints now share validation using effective do_dft/do_disp/backend
   arguments, with direct-entrypoint fail-fast tests.

Expanded tests add main/edge/main reuse in one graph cache, forced non-aligned
DF caps, asymmetric frozen counts, zero-active/virtual spaces and A/B-reversed
geometry. Prior 15-test CPU pass remains valid for its checkpoint; revised
source must pass again before commit/pin and the single approved CPU job.
Incremental revised stage build launched as ba1079bc6. No submission yet.

Revised stage build ba1079bc6: PASS. Expanded CPU gate b1f7d8f24:
**26 passed in 21.17s, no warnings or skips**. Same focused command as above.
Results `build_disp0_cpu/reference-check-8JYWWupu/results.xml` retain 64 signed
comparison records across four physical cases, eager/default replay,
forced non-aligned DF cap and asymmetric freezing. Max |delta|:
Disp20 7.81e-18 Eh; Exch-Disp20 1.01e-17 Eh. A/B reversed native references
differ by 1.18e-17 / -8.05e-18 Eh. Also passed zero-active/virtual, one-cache
main/edge/main replay, lifetime/reset, dispatch guards, unchanged non-HF and
imported fragment reference tests. A separate deterministic planner property
check passed 1000 extent/cap cases with <=4 cached shapes.

Follow-up read-only review d8ebbed82e78c854667d91df47c1cc14c retrieved:
no source blockers in the four scoped files; all three prior findings appear
resolved. Advisory review is not a build/test or GPU correctness substitute.
No CUDA/device-shadow numerical evidence exists yet. CPU build submission
still waits for finalized committed source plus exact-route/headroom checks.

Finalized implementation/build-recipe source pin:
`1e25b5e89a0ec29e17e20c791d9f3d23f7ef3e86`, committed and pushed on
`saptdft_cuest_disp0` (be962a2cf, exit 0). No merge into a parent branch.
This pin represents the initial canonical scalar hybrid checkpoint, not a
passed GPU gate or F-SAPT GPU coverage. The implementation task is complete
only in that explicitly limited sense.

One-shot headroom/path check ba6538b83 and new detached-source/test-only
preparation b76687edd started. Planned immutable remote source:
`P/gits/psi4.saptdft-cuest-disp0-build-1e25b5e89a`; source moved by Git only,
without updating sibling checkouts or queued pins. No accepted job ID yet.
Live preflight ba6538b83: PASS, embers **29/50** slots used, main repository
resolves to `P/gits/psi4`, conda 24.11.3 available, new own source path free,
explicit CUDA13 nvcc present. Exact-route test-only still pending.

Exact-resource test-only b76687edd: PASS, route **cpu-small**, 8 processors,
2h/48G/embers/gts-cs207-chemx script. The test-only forecast number 14182567
is **not an accepted job receipt**. Immutable detached remote source verified
at `1e25b5e89a0ec29e17e20c791d9f3d23f7ef3e86`; only this new worktree created.
The one approved actual submission is now in flight as b15fcd7f7.
No GPU submission and no automatic retry.

### Accepted CPU build receipt

* **Job 14182584**, accepted by b15fcd7f7 (exit 0), submitted
  2026-10-19 12:35:23 Phoenix local time.
* Actual `scontrol`: account **gts-cs207-chemx**, partition **cpu-small**,
  QoS **embers**, 8 CPUs / 48G / 02:00:00, **no GPU TRES**.
  Initial state PENDING (Priority); no start-time estimate claimed.
* Immutable source:
  `P/gits/psi4.saptdft-cuest-disp0-build-1e25b5e89a`,
  exact commit `1e25b5e89a0ec29e17e20c791d9f3d23f7ef3e86`.
* Run root:
  `P/runs/saptdft-cuest-disp0/build-1e25b5e89a0ec29e17e20c791d9f3d23f7ef3e86-14182584`.
  Environment, package/cache, E2 source/build/install and Psi4 build all
  live under that new job-owned root.
* Persistent scratch:
  `/storage/scratch1/3/awallace43/saptdft-cuest-disp0/14182584`.
* Logs:
  `P/runs/saptdft-cuest-disp0/logs/disp0-e2-build-14182584.{out,err}`;
  terminal status file `logs/14182584.status`;
  accepted receipt `logs/submission-1e25b5e89a.receipt`.
* Parent receipt sent successfully with workmux. Exactly one actual
  submission; test-only forecast 14182567 is not a second allocation.
* Scheduler caveat: effective Requeue=1 default was observed. Payload never
  resubmits and refuses reuse of a partial build root. Asked parent permission
  to set Requeue=0 on this job only; **no job modification made yet**.
  Future recipes should explicitly use --no-requeue. Queued source is frozen.

This is CPU build/API/ABI preparation only. It does not establish actual CUDA
device, GPU shadow or cuBLAS numerical correctness. H200 gate and Phase 2 remain
unapproved/unpassed and no GPU job has been submitted.

### CPU build terminal failure (no retry)

Official state-change monitor b68b89e8d reports job **14182584 FAILED**,
job/.batch/.0 all exit **1:0**, elapsed **00:06:31** (step 00:06:26).
It waited pending until the 2026-10-19 23:04:45Z snapshot showed RUNNING;
23:09:48Z snapshot showed FAILED/ALL-TERMINAL. These are observation times,
not exact scheduler start/end timestamps. No requeue or job modification was
made by this agent, and no second build or GPU job has been submitted.

Bounded stdout/stderr/status inspection launched as bceb45947; exact cause
pending artifacts. The single build authorization is consumed; preserve
failed immutable source/environment/run roots and coordinate any next build
separately. CPU/API/ABI preparation has **not passed** merely because its
submission succeeded. GPU numerical/device-shadow gate and Phase 2 remain
blocked.

Bounded artifact inspection bceb45947: confirmed FAILED_OR_SIGNALLED with
`exit_code=1`, `term_received=0`; epilog confirms actual cpu-small / embers /
gts-cs207-chemx. Exact failure occurs in Einsums d17bb76e configure:
`cmake/Einsums_SetupCatch.cmake:37` calls `check_cxx_source_compiles`; its
try_compile cannot resolve `Catch2::Catch2`. This setup is unconditional even
when native tests are OFF. The fresh env omitted installed Catch2, selecting
the FetchContent source-backed alias. Local CPU baseline has installed
Catch2 3.16.0 and imported targets. The solve also selected CMake 4.4 rather
than local 4.3.2; this is recorded drift, not the sole assumed cause.

Local diagnostic:
`devtools/disp0/catch_config_probe` with GNU14.3/CMake4.3.2 reproduced the
same missing-target error for a source alias (exit 1) and linked the actual
installed Catch2 string-view symbol with imported target (exit 0).
This red/green result establishes the build-system issue independently of
CUDA. No scientific contraction or native dispersion algorithm was changed.

Prepared recipe-only correction: Catch2=3.16.0, explicit Catch2_DIR and
preflight link probe, CMake=4.3.2 for baseline comparability, and
`#SBATCH --no-requeue`. These changes affect only this local branch, **not**
failed immutable source/env/build/install roots. No new job submitted.
Any corrected dependency build requires fresh parent approval.

Additional failed-run provenance bafb39732 (read-only): header/link probe
reported `CUDART_VERSION=13030`, runtime version 13030, status 0; nvcc
13.3.73. Configure selected explicit CUDA13 compiler/toolkit, sm90 and
Python build ON. These are **CPU header/link preparation checks only**,
not CUDA device/runtime or numerical correctness.

Environment lock exists. Actual packages included Apiary 1.0.0,
pybind11 3.0.2, mimalloc 3.3.2, CMake **4.4.4**, GCC/G++ implementations
14.3.0 **build 20**; the local CPU oracle uses GCC/G++14.3.0 build18.
The version-only GCC pin did not freeze package build numbers; this drift is
retained rather than claiming identical toolchains. `Catch2_DIR` pointed to
CMake's `pkgRedirects`, confirming source fallback, not the installed imported
target used by the CPU oracle. No E2 native/Python install or fresh Psi4 stage
was completed.

Recipe-only correction pin `5245210a100b73c3c5b275d889ad2de3c51ec736`
was sent to parent with a fresh one-job CPU build approval request. No new
approval or allocation is assumed.

## Corrected CPU build authorization and preflight

Coordinator explicitly relayed fresh approval for **EXACTLY ONE** corrected
CPU-only build at `5245210a100b73c3c5b275d889ad2de3c51ec736`, 8 CPUs / 48G /
2h / free embers / gts-cs207-chemx. This is one authorization, not permission
for an extra submission if already acted upon. At receipt, this agent had
submitted only failed job14182584; **no corrected job had been submitted**.

Require fresh passing exact-route test-only, live headroom, explicit
--no-requeue and new detached source/env/build/run roots. Preserve all
14182584/1e25b5e89a evidence, including actual CMake4.4.4 / GNU14.3 build20
versus CPU-oracle build18 drift. No automatic retry, inferno or GPU submission.

Started corrected-branch publication b4eae7c65 and fresh headroom/duplicate
guard be1c8d608. Intended new source
`P/gits/psi4.saptdft-cuest-disp0-build-5245210a10`; receipt must not already
exist at `logs/submission-5245210a10.receipt` before any actual submission.
No corrected accepted job ID yet.

Corrected preflight be1c8d608: PASS, live embers **26/50** slots used.
Corrected source path/receipt were absent before staging, no active
`disp0-e2-build` job was reported, and the original failed receipt is present.
Branch publication b4eae7c65: PASS. Exact corrected test-only b2fe53fb3 pending.

Corrected exact-route test-only b2fe53fb3: PASS, **cpu-small**, 8 processors.
Detached source verified at the exact approved 5245210a full SHA and committed
--no-requeue verified. Forecast number 14193846 is test-only, **not an actual
allocation**. One corrected submission is now in flight as bb8c805ec, with
fresh receipt guard and a second active-build duplicate check immediately
before sbatch. Explicit --no-requeue also passed on the actual command line.

### Accepted corrected CPU build receipt

* **Job 14193865**, accepted by bb8c805ec (exit 0), submit timestamp
  2026-10-19 19:18:04 Phoenix local time.
* Exactly approved source pin:
  `5245210a100b73c3c5b275d889ad2de3c51ec736`,
  detached at `P/gits/psi4.saptdft-cuest-disp0-build-5245210a10`.
* Effective `scontrol`: **gts-cs207-chemx / cpu-small / embers**, 8 CPUs,
  48G, 02:00:00, **Requeue=0**, Restarts=0 and **no GPU TRES**.
  Initial PENDING (Priority); no start-time estimate claimed.
* New job-owned root:
  `P/runs/saptdft-cuest-disp0/build-5245210a100b73c3c5b275d889ad2de3c51ec736-14193865`.
  Source/env/build/install/cache roots are distinct from failed job14182584.
* Persistent scratch:
  `/storage/scratch1/3/awallace43/saptdft-cuest-disp0/14193865`.
* Logs `P/runs/saptdft-cuest-disp0/logs/disp0-e2-build-14193865.{out,err}`;
  terminal artifact `logs/14193865.status`;
  accepted receipt `logs/submission-5245210a10.receipt`.
* Parent receipt delivered. Exactly one corrected actual submission.
  Official state-change monitor b130435b3 started; no automatic retry.
  Failed14182584/1e25 roots and evidence remain untouched.

Acceptance is not build success or CUDA correctness. No H200/GPU numerical
job or broader F-SAPT work is authorized/started by this receipt.

## GPU-test authorization

User subsequently explicitly authorized the GPU test. Apply the previously
agreed bounded shape: one H200 / 4 CPUs / 32G / 30min, free embers /
gts-cs207-chemx. This does not waive corrected-build readiness, exact-route
test-only/headroom, matching native/Python import/ABI checks, actual device
identity or signed numerical/trace evidence. Do not finish an unready build
inside this allocation, borrow a holder, use inferno or start broader F-SAPT.
First inspect corrected job14193865 and its strict preparation artifacts.

Readiness inspection b56d9f49b: corrected build14193865 is **RUNNING** on
atl1-1-02-005-24-2, elapsed 00:05:35 at snapshot; job/.batch/.0 still running.
No terminal status, runtime-provenance.json, api-probe.json or staged Psi4
executable existed yet. The GPU gate is authorized but **not build-ready**;
do not allocate a GPU merely to wait or finish this build. Official watcher
b130435b3 remains responsible for terminal notification.

While the corrected build runs, prepared standalone GPU gate files
`devtools/disp0/{gpu_gate.py,cublas_trace.cc,prepare_gpu_gate.sh,
gpu_gate_payload.sh,gpu_gate.sbatch}` and parser-only fixtures
`tests/pytests/test_disp0_gate_evidence.py`. Scientific `psi4/` files remain
byte-for-byte unchanged from build pin5245210a; the harness must import only
that prepared stage, never a source overlay.

Gate plan: native FISAPT plus eager/default CPU replay vs CUDA requests at
edges1/5/24/500, forced DFcap7, asymmetric freezing, changed/reversed/repeated
geometries, spherical/Cartesian bases and a nontrivial benzene dimer to obtain
naturally eligible blocks without changing the cost model. Explicit larger
production block for that case remains subject to the existing memory planner.
Real GPU replay tests strided input packs, main/edge/main reuse and live pool
reset rejection/release. Native CPU fragment partition sum is an oracle, not
GPU fragment coverage; unsupported fragment/S-inf requests must still reject.
Unchanged non-HF regression is repeated with the real CUDA runtime present.

Evidence plan: exact prepared binary hashes/single-world import, header-ABI
identity function compiled against CUDA13, logical CUDA UUID matched to NVML
UUID and this process/exclusive context (no physical-index-zero assumption),
default graph's observed `gpu_dispatch` annotations AND forwarded native
`cublasDgemm_v2` API/status records within physical/production phases.
The diagnostic interposer implements no mathematics and forwards arguments
unchanged. It must be compiled/ELF-checked **before** the GPU allocation.
Packing, transfers, CPU work and diagnostic/oracle time are labelled; no
speedup/full-GPU/F-SAPT coverage claim.

Gate scripts pass AST/shell/diff checks. Parser-only CPU fixture tests launched
as b45d86fc5; review d61de3e3d264a3ad3977373062ef1950d pending. The tracer has
not yet been compiled and the gate has not run on CUDA. No GPU submitted.
* Read-only Phoenix access preflight: PASS, exit 0 (bfefc7153).
  No remote branch worktree/build has yet been created and no job submitted.
* Read-only Phoenix dependency discovery: PASS, exit 0 (b99607f10).
  Confirmed `P/miniconda/envs/p4cuest_sapt` (Python 3.13.11) and cuda13
  toolkit exist. No Einsums checkout found at either checked conventional
  `P/gits/Einsums` or `$HOME/gits/Einsums`, and no `P/miniconda/envs/p4_einv2`.
  This does not exclude custom locations; do not treat the parent's old
  1.x measurement stage as a compatible ready build. Dedicated source,
  environment and native/Python install preparation are still needed.

### Corrected CPU build failed; authorized GPU gate remains blocked

Official watcher b130435b3 reports corrected job **14193865 FAILED**, job,
.batch and .0 all exit **1:0**, elapsed **00:17:51** (step00:17:42).
The terminal observation was 2026-10-19T23:38:47Z. No retry or job mutation
was made. Both immutable failed roots remain preserved.

Bounded stdout/stderr/status inspection launched as b6ef30efe for the exact
new blocker. The user's GPU-test authorization is retained, but is not a
waiver of coherent E2/Psi4 build/import/API readiness. **No GPU submitted**;
do not repair/finish a full dependency build inside the 30-minute H200 gate.
No third CPU allocation is authorized by either consumed one-job approval.

Parser-only CPU fixture task b45d86fc5: **11 passed in 1.33s**, no warnings
or skips. Results:
`build_disp0_cpu/reference-check-lOGVr7MK/results.xml`.
It verifies acceptance/rejection logic for native API records and observed
profile annotations, including rejection of placement-only/CPU-fallback,
wrong-PID, failed-status and nonphysical fixtures. These are deliberately
synthetic parser fixtures, **not CUDA execution or GPU numerical evidence**.

GPU harness review d61de3e3d264a3ad3977373062ef1950d found seven proof/control
gaps; addressing before any allocation: explicit native cuBLAS loader/link
preflight, attesting currently imported/mapped binaries and scientific modules
instead of only prepared filenames, actual cuBLAS on each repeated synthetic
main replay, dropping independent pool-output references before ownership
checks, observed CPU edge execution (not placement alone), published driver
signed/sum QCVariables, and EXIT status handling before fallible preflight.

Native forwarding now explicitly links cuBLAS and exposes a resolve-only
identity probe, still no numerical algorithm. Larger bounded synthetic main
matrices exercise naturally eligible CUDA replays without forcing the cost
model; small edges remain separate. Physical CPU-edge proof is scoped to
actual `ComputeGraph::execute(sapt0-dispersion-products)` profile events.
Additional negative parser fixtures and fake-scheduler parameter/controller/
QoS/signal tests launched as b2210c4b4; CPU QCVariable wiring is also checked.
These tests do not substitute for native compilation, loader preflight or GPU
numerics. Underlying E2/Psi4 build readiness is still blocked by job14193865.

Corrected-build artifact inspection b6ef30efe confirms the new exact blocker:
Apiary/Clang cannot find `stddef.h` while parsing Options.hpp through GNU14.3
`<complex>/<cmath>` and the conda sysroot's stdlib.h. `generate_rc` then fails
"no option descriptors found". The native E2 shared library linked at build
step378/520, but installed Python `_core`, Psi4 stage, runtime proof and API
proof are all absent. Status records `exit_code=1`, `term_received=0`,
FAILED_OR_SIGNALLED; epilog confirms cpu-small/embers/gts-cs207-chemx.
This is not a device runtime failure that a GPU allocation can solve.

Source audit: Apiary's CMake helper only locates its Clang builtin resource
directory through a Clang native compiler or `CONDA_PREFIX/bin/clang++`.
Native compiler must remain GNU14.3. CPU oracle has clangxx/Clang22.1.4 and
`lib/clang/22/include/stddef.h`; original build recipe did not explicitly
install clangxx or bind CONDA_PREFIX to its new env. Read-only failed-cache/
package inspection b32b6a119 is pending before confirming this environment
discovery hypothesis.

Prepared a stricter isolated Apiary standard-library/annotated-AST preflight.
Initial negative resource probes unexpectedly exited0 and retained the
annotated declaration; direct diagnostics revealed **Apiary exits0 and emits
partial AST even after a fatal missing-stddef error**. These failed diagnostic
assumptions are retained. The strengthened preflight rejects fatal parser
stderr as well as exit failures/missing AST, removes prior output and excludes
ambient host include rescue. It now gives local RED (missing resource,
exit1 with fatal stddef) / GREEN (matched Clang resources, native GNU14.3,
nonempty AST, exit0).

Prospective recipe adds clangxx22.1.4 for AST resources only, job-local
CONDA_PREFIX binding, this fail-fast preflight and fatal-parser diagnostics
checking on the full build log. Native GNU/CUDA pins and scientific source
are unchanged. No failed root has been modified and no third CPU allocation
or GPU allocation has been submitted.

Strengthened CPU gate-control task b2210c4b4 completed: **22 passed in
2.11s**, no warnings/skips. Command:
`bash devtools/disp0_test_reference.sh "$PWD/tests/pytests/test_disp0_gate_evidence.py" "$PWD/tests/pytests/test_disp0_gate_batch.py"`.
JUnit:
`build_disp0_cpu/reference-check-iqa4RZ30/results.xml`.
This covers synthetic trace/profile acceptance and rejection, actual CPU
driver QCVariable names/component sum wiring, and fake-scheduler early
failure/QoS/controller/TERM handling. It is separate from the 26-pass physical
CPU checkpoint and provides **no CUDA execution or GPU numerical proof**.
Follow-up read-only review da307e05df0e1659daaa9052fa847294b is pending.

Read-only failed-environment inspection b32b6a119 confirms the Apiary resource
discovery failure, rather than merely suggesting it:
`APIARY_RESOURCE_DIR` is empty; the dedicated env has no `bin/clang++` and no
Clang builtin `stddef.h`; `APIARY_EXTRA_ISYSTEM` incorrectly names the base
`P/miniconda/include`, not this job's environment. Native C++ is correctly
the job-owned GNU14.3 compiler. Failed job14193865 has Apiary1.0.0,
Catch2 3.16.0, **CMake4.3.2** and libllvm22 22.1.8. This distinguishes its
corrected CMake pin from failed job14182584's actual CMake4.4.4. The only
partial native result found is
`einsums-build/lib/libEinsums-abi1.so.2.0.0`; it is not an installed coherent
native/Python runtime and will not be overlaid with an old Python extension.
All inspection was read-only. Proposed Clang resources/CONDA_PREFIX/preflight
fix still needs review and a new committed pin before requesting any fresh
CPU build approval.

Follow-up review da307e05df0e1659daaa9052fa847294b retrieved: no further
concrete GPU harness blockers; one P2 in the CPU launcher: required parameter,
controller and QoS checks preceded its EXIT trap. Moved trap installation
ahead of those checks, added unset-safe status fields and status-write failure
handling. The fake-scheduler early-failure/TERM suite now exercises both
CPU and GPU launchers. Final CPU control rerun bfd547bbb and fresh Apiary
red/green preflight b1743f6f0 are pending. Static bash/diff checks pass;
`psi4/` source remains byte-identical to5245210a. Native CUDA tracer compilation,
loader preflight and actual GPU numerical execution remain untested.

Final CPU gate-control rerun bfd547bbb: **26 passed in2.10s**, no warnings or
skips, including all eight CPU/GPU fake-scheduler early-error and signal cases.
Command is the same two-file invocation above; JUnit:
`build_disp0_cpu/reference-check-9hsDyCjK/results.xml`.
These 26 control tests are **distinct from** the earlier 26 physical
dispersion/API tests and64 signed comparisons; do not combine the counts into
a GPU pass claim.

Final fresh Apiary preflight b1743f6f0: **PASS**. Deliberately missing
resource headers give RED exit1 with the expected fatal missing-stddef
diagnostic; matched Clang22.1.4 resources with native GNU14.3 give GREEN
exit0 and the required nonempty annotated AST. Artifacts:
`build_disp0_cpu/apiary-final-review/{red,green,red.log,green.log}`.
No failed remote root/environment was touched. Review findings are addressed
and the build fix plus gate controls are ready to commit/pin. This is local
parser preparation only, not proof that a fresh Phoenix solve/build or CUDA
runtime will pass. Another CPU build requires fresh explicit approval;
the already authorized GPU test stays blocked until matching preparation
passes. No automatic retry and no broader F-SAPT transition.

## Approved single additional CPU preparation build

User explicitly approved exactly one additional CPU-only build at
`587e172a33afaadb00eb7f574a5089533f7fc755`, 8 CPUs/48G/2h,
free embers/gts-cs207-chemx, cpu-small subject fresh exact-route test-only and
live headroom, explicit no-requeue, fresh isolated source/environment/build/
run/persistent scratch. Failed14182584/14193865 remain untouched.
Push b70e9aa83 completed successfully to origin/saptdft_cuest_disp0.
Source staging b333efc2f uses a new independent filtered Git clone and exact
detached clean identity; it does not advance a shared checkout or queued pin.
Live headroom inspection b03e6706f launched. Approval has not yet been
consumed by an actual submission. This is CPU preparation only; GPU
authorization remains conditional on matching stage/runtime readiness.

Source staging b333efc2f completed: new independent
`P/gits/psi4.saptdft-cuest-disp0-build-587e172a33`, exact587e172a33 full pin,
detached and clean; no shared checkout advanced. Headroom b03e6706f:
27/50 embers slots. Fresh exact-route/no-duplicate inspection ba11331a6
is pending; no actual submission yet. Clang/Apiary resource and linked LLVM/
libclang provenance must be retained from the newly solved job environment,
not inferred from the local red/green probe.

Exact-route preflight ba11331a6 passed at2026-10-20T00:14:37Z: source
identity/cleanliness verified, 27/50 embers slots and zero active dispersion
build duplicates. `sbatch --test-only` pseudo-ID14196325 selected
**cpu-small**, 8 processors; its predicted start is not an accepted job
or scheduling guarantee. Invocation explicitly requests account
gts-cs207-chemx, QoSembers, 8CPU/48G/2h and no-requeue, identical to the
actual single-attempt submission launched as b7a547b83.
Actual accepted ID/effective route are pending that receipt. The receipt
is created exclusively before sbatch; an ambiguous failure must be inspected,
not automatically resubmitted.

### Accepted additional CPU build receipt

* Exactly one actual submission accepted: **14196330** (b7a547b83, exit0).
* Source pin: **587e172a33afaadb00eb7f574a5089533f7fc755**, detached clean
  independent clone `P/gits/psi4.saptdft-cuest-disp0-build-587e172a33`.
* Effective controller route: **gts-cs207-chemx / cpu-small / embers**,
  8CPUs/48G/02:00:00, **Requeue=0**, Restarts=0, no GPU TRES.
  Initially PENDING(Priority); no actual start-time guarantee.
* Fresh job-owned environment/source/build/install/cache run root:
  `P/runs/saptdft-cuest-disp0/build-587e172a33afaadb00eb7f574a5089533f7fc755-14196330`.
  Persistent scratch:
  `/storage/scratch1/3/awallace43/saptdft-cuest-disp0/14196330`.
* Logs: `P/runs/saptdft-cuest-disp0/logs/disp0-e2-build-587e-14196330.{out,err}`.
  Terminal artifact `logs/14196330.status`; submission receipt
  `logs/submission-587e172a33.receipt`; preflight receipt
  `logs/preflight-587e172a33.txt`.
* Official state-change monitor b7e91b661 is running. This approval is
  consumed; no retry or fourth CPU submission is authorized.
  Failed14182584/14193865 environments and evidence remain unchanged.

Acceptance is not successful preparation. At terminal state, require actual
clang/libclang/LLVM/resource paths and versions, full explicit environment,
header/ELF/import/API/physical checks and a coherent matching stage before
the separately authorized GPU gate. No GPU submission or F-SAPT expansion.

## User-authorized paid route and benchmark protocol

User explicitly requests inferno on **gts-cs207**, results by morning8am
(interpreted as Phoenix/US Eastern unless corrected), then specifies the
benchmarking under `~/docs/saptdft/cuest` with inputs from
`~/data/timing_test_suite/`. Do not silently fall back to chemx, promise a
speedup or bypass correctness. Inspect existing14196330 before moving its
route; do not create a duplicate. Original monitor remains active.
Read-only state bd6312b8b and corrected CPU account pricing b2a779317 pending.
An initial pricing-only invocation b97f173d9 used invalid `--gpus0`; it
submits nothing and is replaced by proper `--cpus8`.

Read-only protocol inspection: suite routine systems are peptide, nanotube,
protein83 and protein157; pla15_3pe2 is explicitly excluded without named
opt-in. Use fixed geometries/NBF checks, three independent fresh-process
repeats per arm, median whole-energy wall and actual selected-run phase
vectors, retained min/max rather than invented confidence intervals, fresh
timer directories and flat-section-only parsing. The published coupled
scaling cohort is PBE0 SAPT(DFT)-D4(I)/`6-31+G**`, spherical, not this branch's
HF canonical Disp20/Exch-Disp20 science. Reuse its inputs/protocol, not its
energies/timings as a matched HF baseline. Host threads, memory and library
differences must remain explicit. Its large protein memory budgets exceed
the bounded32G correctness gate; do not squeeze the full suite into it.
Parent notified of paid-route/benchmark request and overlap coordination.
No source/data/docs in the reference suite or coordinator trees modified.

Fresh state bd6312b8b:14196330 is already RUNNING on
atl1-1-02-005-26-2, cpu-small/embers, elapsed30:27, start
2026-10-19T20:15:10 Eastern; job/.batch/.0 still running. Do not cancel,
restart or change this running job's QoS/account: paid scheduling cannot
accelerate it and its safety guard expects embers. Parent confirms this
decision. Read-only running build-phase/resource inspection bae1ef0f0 pending.
Explicit deadline assumption: **2026-10-20 08:00 America/New_York**.
No guarantee of completed dependencies, signed GPU correctness or speedup
by then.

Parent14185006 remains pending(Priority), oneH200/8CPU/128G/6h and24 full
PBE0-D4(I) calls; no holder is available. Do not touch/duplicate that
allocation or its roots/data/docs. For our distinct HF-E2 work, fresh paid
account capacity, exact job count/resource/time charge bound and independent
protein memory sizing must be reported **before expanded allocations**.
Matching geometry/basis/fresh-process3repeats/wholewall/phase protocol does
not make different scientific methods a valid speedup pair. Conditional
one-GPU correctness approval does not imply unlimited benchmark spending.

Read-only running snapshot bae1ef0f0: native E2 build advanced to **517/520**
(Python binding shards); no installed E2 Python/Psi4 stage, runtime-provenance
or API JSON yet. Apiary's resource/include cache now selects this job's
environment, not base miniconda. clangxx/clang/clang-22/libclang-cpp22.1 are
22.1.4; actual libllvm22 is22.1.8 (retain patch-version drift versus local
oracle, not an exact dependency match). Apiary's ELF dependency resolves
`libLLVM.so.22.1` inside the job environment. Native compiler remains the
job-owned GNU toolchain; E2 config CUDA=ON, arch90.
Clang reports22.1.4 and dedicated `lib/clang/22/include/stddef.h` exists,
SHA256 `b3e62a046ddf28d09eedebe6eb9461cb7060e47cad2c69964b222c4eea10347e`.
Resource discovery is fixed; this does not yet prove completed Python
binding generation/install/import, scientific readiness or CUDA execution.
Parent received the phase/provenance checkpoint.

Correct CPU account selector b2a779317: gts-cs207 available$29.51, hypothetical
8CPU/cpu-small/2h charge bound$0.13; the running CPU job stays free embers.
H200 pricing b1fead507 and basis-only routine-suite sizing b30d30efd pending;
no expanded allocation submitted.

Basis-only suite inventory b30d30efd PASS, **no SCF/timing performed**.
Read-only systems.py SHA256
`72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb`.
Using spherical6-31+G**, def2-universal-jkfit and aug-cc-pVDZ-RI:

| System | NBF | JKaux | RIaux | Dense DFJK upper GiB | Transformed-factor disk upper GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| peptide | 250 | 1010 | 1042 | 0.482 | 0.429 |
| nanotube | 548 | 2238 | 2240 | 5.063 | 4.267 |
| protein83 | 922 | 3735 | 3820 | 23.811 | 21.164 |
| protein157 | 1786 | 7265 | 7384 | 173.242 | 155.782 |

Dense bound is8*(Naux*NBF^2+Naux^2+8*NBF^2), no screening assumption.
Disk bound is8*Nri*(4*na*nr+4*nb*ns+2*na*ns+2*nb*nr), using unfrozen
occupied spaces and dimer-centered virtuals, not measured disk/RSS/VRAM.
Other live arrays, integral/setup work, page cache and temporary disk copies
are additional. Spherical NBF differs intentionally from the suite's original
Cartesian RHF counts; verified against the number of heavy-atom d shells.
Do not assert that protein157 fits the32G gate or the parent's128G budget
for this different HF/E2 workload.

H200 selector b1fead507: current gts-cs207 available$29.51;
one30min GPU charge estimate$0.71; one6h GPU charge estimate$8.48.
These are GPU-weight estimates only; full CPU/memory billing bound
inspection b2ffbd4c3 pending. No paid allocation submitted.
Timing-design review dced54a2ddace409bff289adaddfde83d: gate must pass before
timing; do not use oracle-inclusive gate walls or replay-with-transfers as
kernel-only timing. Instrumentation-free fresh-process suite wholecall timing
must be distinct from diagnostic profiling, and wholecall versus dispersion
regions reported separately. GPU launcher now explicitly requests/validates
gts-cs207/inferno; CPU launcher remains unchanged chemx/embers. Updated
negative route/account/control tests b1851c000 pending.

User explicitly approves **one candidate H200 timing allocation up to6h**,
inferno charged specificallygts-cs207, conditional on coherent installed
E2/Psi4 readiness and the actual signed GPU correctness gate passing.
This is one approval, not an additional duplicate six-hour job; it does
not authorize CPU reruns, retry loops, parent14185006 changes or blanket
paid work. The separately approved bounded30min correctness allocation
remains available. Neither GPU approval has been consumed.
Before either submission, report exact CPU/memory/GRES/time/jobcount and
conservative charge using fresh gts-cs207 capacity, then stop if that account
cannot cover the work. No silent account fallback. Fresh roots/no-requeue,
no GPU builds/repairs, preserve failed and parent evidence. Timings require
uninstrumented fresh-process3repeats, matched HF science and wholewall/phase
cuts; no08:00 guarantee while prerequisites remain incomplete.

Paid GPU route/control regression b1851c000: **28 passed in2.18s**, no
warnings/skips, JUnit
`build_disp0_cpu/reference-check-cikiPNBp/results.xml`.
Command: same two-file gate-evidence/batch test invocation recorded above.
GPU guard accepts onlygts-cs207/inferno; CPU guard remains
gts-cs207-chemx/embers; wrong-account, wrong-QoS, early-parameter/controller
failure and TERM cases tested with fake scheduler. This is CPU-only harness
validation, not CUDA execution or GPU numerical correctness.

Complete billing inspection b2ffbd4c3: gpu-h200 has
`TRESBillingWeights=gres/GPU=52381` only (no CPU/memory weights),
MaxMemPerNode=UNLIMITED, PriorityFlags empty. Selector estimates therefore
bound the requested resources at$0.71 for1H200/30min and$8.48 for1H200/6h,
combined$9.19. Capacity/route must be refreshed before actual submissions.
Exact conditional proposal reported to user and parent and saved in
`devtools/disp0/benchmark_plan.md`: gate1H200/4CPU/32G/30min plus one
timing1H200/8CPU/512G/up-to6h, bothgts-cs207/inferno/no-requeue.
Timing uses one node for both arms, 24 timed fresh-process HF calls,
four routine systems × two arms ×3repeats, initial Psi4 budget128GiB;
resource margin considers independent protein157 bounds, other arrays,
DF/disk behavior and page cache. Sufficient scratch/quota and actual
memory behavior still require checks. No additional CPU allocation, no
parent PBE0-D4 duplicate and no GPU job submitted. Six hours is a ceiling,
not a validated runtime estimate or deadline guarantee.

Parent confirms the exact512G proposal lies within the two conditional
approvals. Clarified benchmark plan: CPU arm disables **GPUPlacement** for
the same E2 graph; GPU arm retains natural placement. Label8-CPU versus
hybridGPU, never historical24-CPU/PBE0 speedup or fully resident GPU.
Actual free RAM and persistent scratch/user+group quota must be sufficient
before protein launch; read-only quota inspection b18030c4e pending.

Preparing standalone `benchmark_run.py` and `benchmark_evidence.py`; not
ready/submitted and no scientific stage overlay. The per-process proof
design captures actual same-graph objects, asserts all CPU targets for the
CPU arm, then replays/logs them **after measured work**, with a separate
naturally eligible native positive control to avoid accepting a silent
logger as no GPU. Native cuBLAS logging is explicitly OFF during measured
work, no interposer/profiler there. Gate preflight must validate the logger
and both CPU/GPU controls before the six-hour timing job can be considered
ready. Native API/header/symbol inspection bb41a5dba pending; no claim this
diagnostic works on CUDA yet. CPU synthetic control/tamper tests b9c0fe764
pending. This does not relax the numerical gate or imply timing readiness.

## Additional CPU build14196330 failed; GPU approvals remain conditional

Official watcher b7e91b661 terminal observation2026-10-20T00:55:46Z:
**14196330 FAILED**, job/.batch/.0 exit **1:0**, elapsed **00:40:04**
(step39:57). No fourth CPU build/retry is authorized; no job or failed root
was altered. Parent notified immediately. Exact read-only log/status/install
artifact diagnosis b1f43b496 pending; do not infer the blocker from the
earlier517/520 snapshot. Neither GPU approval is consumed. Matching stage,
signed/device correctness and timings remain unpassed; no build/repair
inside a GPU job and no guaranteed08:00 outcome.

Native API inspection bb41a5dba confirms the installed CUDA13 header
declares `cublasLoggerConfigure(int,int,int,const char*)` and the selected
libcublas.so.13 exports it (also logger callback getters/setters). This
validates the proposed ctypes signature/symbol only; actual logging,
positive-control CUDA execution and untimed per-process proof have not run.
It does not resolve the failed build or authorize GPU allocation.

Persistent scratch inspection b18030c4e: filesystem has
1,264,549,264,683,008 free bytes; user usage4,838,809,280KiB versus
16,106,127,360KiB soft/hard limit, about **10.494TiB** quota headroom and
792,049 available inodes. Reported primary-group byte/inode limits are0
(no finite limits in that report). This snapshot is sufficient for the
proposed600GiB conservative protein scratch margin, but must be refreshed
in the actual allocation before protein launches; it is not proof of actual
future RAM availability or a waiver of project/filesystem restrictions.
No scratch data modified. Preparing fail-closed per-point RAM/free-space/
quota checks, with raw quota evidence retained. Runtime/build blocker remains.

Initial timing-control task b9c0fe764 completed **41passed in2.46s**,
JUnit `build_disp0_cpu/reference-check-ot373Eol/results.xml`. These are
synthetic proof/tamper/logger-toggle tests plus CPU route/QCVariable tests,
not CUDA proof or measured timings. Subsequent quota parsing and one small
Water-Ne full HF/CPHF+deltaHF native-vs-CPU-graph physical regression were
added; final frozen rerun b3da2a5a0 and read-only timing-worker review
da12090c4390f1dbfb7ee6d81962647de pending. The single-point timing worker
is preparatory; no multi-point campaign launcher is ready/submitted.

Exact failed14196330 artifact inspection b1f43b496: **E2 and Psi4 builds
completed; Psi4 install completed** (`[9/9] Completed psi4-core`).
The post-build `runtime_provenance.py` failed at `import einsums` with
**ModuleNotFoundError**. Expected `einsums-stage/lib/einsums` is absent;
staged Psi4 executable exists; runtime-provenance.json exists but is empty,
API JSON absent. No import/API/physical probe passed. This is not yet an
observed CUDA runtime or numerical failure. Status remains exit1/term0/
FAILED_OR_SIGNALLED, epilog cpu-small/embers/gts-cs207-chemx.
Read-only install-manifest/cache/actual module-path audit b237a48ef pending.
Do not rebuild or rewrite failed stage/environment/status. If matching
installed modules exist elsewhere, propose fresh CPU-only validation/proof
root (no rebuild) with explicit approval rather than silently relabeling
this failed job or repairing anything inside a GPU allocation.

Expanded timing controls b3da2a5a0: **46passed/1failed in5.12s**,
JUnit `build_disp0_cpu/reference-check-Nbr2YCLV/results.xml`.
The new full HF/CPHF test failed at QCVariable lookup before comparing
components: invented `SAPT DISP20 ENERGY` name is not published on this
route. Corrected both test and preparatory worker to the actual existing
`Disp20,u` / `Exch-Disp20,u` signed names already validated by prior CPU
gate tests. No numerical discrepancy inferred from that lookup error;
no tolerance changes. Rerun b0c706d55 pending. Keep this failure evidence.

Install audit b237a48ef: **no E2 Python package at any installed path**.
Manifest lists native libEinsums ABI1 libraries only; cache has
EINSUMS_INSTALL_PYMODDIR=naive. Complete built package, `_core.cpython-313`
and native libEinsums pair exist together under the same job's
`einsums-build/lib`. This is a genuine missing Python install, not an
alternate installed directory. No overlay/rebuild/failed-root edits made.
Bounded read-only same-build metadata/import/ELF/hash probe bf8b36bda launched,
using build-tree E2 plus installed same-job Psi4, bytecode writes disabled.
It performs no calculation and is not installed-runtime/API/GPU evidence.

Prospective recovery, not authorized/submitted: declaratively install the
complete same-build E2 Python/native artifacts and whole Psi4 stage into a
fresh runtime prefix (no selective scientific Python overlay), verify byte
identity/source lineage/library maps and run CPU API/physical probes in a
fresh proof root. No native recompilation should be needed if the metadata
audit passes. This would require a narrow fresh CPU-only approval; the
current no-extra-CPU allocation constraint is respected. Do not turn a GPU
gate into installation/repair or relabel failed14196330. Parent informed.

Timing review da12090c4390f1dbfb7ee6d81962647de found five blockers:
optimized-Python assert bypass (including native side-effect calls);
wrong/root cgroup limits; incorrect scratch group/project quota coverage;
retaining adapters excluded normal cleanup from measured wholewall;
and an incomplete scientific module manifest could pass.
Addressed provisionally: reject optimized Python before science imports,
native ctypes calls outside assert; resolve process membership/mounts and
tightest visible ancestor cgroup limits, fail closed if unprovable; inherited
setgid and project quota checks; require exactly four scientific module
names/hashes. Added targeted negative/synthetic tests.

Crucially, removed strong graph references from timed work. Per-process
dispatch proof now reconstructs an identical whole HF call **after** timing,
with logging/reference retention limited to that untimed pass. Signed/global
components and compute/DF/staging/placement policy must match the measured
call. Maximum campaign count is now explicitly **24timed +24untimed full
calls**, plus small native positive controls, in the same single6h allocation
and unchanged charge ceiling. No cleanup cost is intentionally moved outside
the measured call, and no extra allocation is implied. Parent notified of
the increased work and deadline risk; plan updated. Final CPU control run
bda04c116 and follow-up review dbf45f11ac926f5817187e64078bd26ec pending.
No GPU execution/coverage/readiness/timing claim.

Parent methodology clarification accepted: reconstructed untimed execution
**does not independently prove actual timed-call dispatch**. Worker/plan now
label measured results **policy-controlled8-CPU versus hybrid timing**,
`actual_timed_gpu_dispatch_verified=false`; diagnostic evidence is separately
named. Keep24timed calls in24fresh processes, at most24additional untimed
reconstructions plus small controls; diagnostic walls never enter medians.
48-call admission stays inside the same one6h/$8.48 ceiling and deadline,
with partial counts/failures retained, not manufactured3repeat ratios.
No new CPU approval exists.

Also removed pre-clock CUDA identity/synthetic warmup/forced GPU synchronization:
fresh whole-call first-use work remains within production timing where
applicable; CPU timed route does not explicitly call CUDA synchronization.
GPU end synchronization is included in both its wholecall and dispersion
boundaries. Native logger remains default-OFF in the fresh measured process
(reject logger environment flags); native metadata/logger diagnostics begin
only after the measured interval. These are preparatory methodology changes,
not executed native timing results.

Corrected full-HF protocol/control rerun b0c706d55: **47passed in7.58s**,
JUnit `build_disp0_cpu/reference-check-LMqLhDCA/results.xml`, no warnings/
skips. This precedes the later review-control additions; bda04c116 remains
the final pending run. The small Water-Ne fullHF/CPHF+deltaHF CPU graph
versus nativeSAPT comparison retains all seven component deltas:
ELST/EXCH/IND0; signedDisp20 -2.882622526095835e-17Eh;
signedExch-Disp20 +1.2894350812459136e-9Eh; combinedDisp/TOTAL
+1.2894350524247705e-9Eh. These pass the predeclared1e-8 physical tolerance
without loosening; they are not the near1e-17 agreement of the earlier
nativeFISAPT-specific oracle comparisons. No GPU result inferred. Parent
received the discrepancy and scope explicitly.

Same-build metadata bf8b36bda PASS (read-only, no calculation): Psi4
1.12a1.dev622 from completed installed stage; E2 Python/native both from
`einsums-build/lib`, exactd17bb76e/GNU14 and one mapped native world.
Approved source binary hashes:
Psi4core `b3cfe2f2261533df9a9018731d9fe2bffea41cfe70f53f14195caaee7de8697d`;
E2Python `_core` `8f23c69b79613f9b94eee95d4735e8f88f67be7c98867290f7546ca56bdc6833`;
native `9e31b5ff8cf28e796b04ee74853c4923f06acf03f1af24dfddd75b79de583cb7`.
CUDA maps resolve toolkit links to p4cuest_sapt's CUDA13 libraries
(libcublas13.6.0.2 and libcudart13.3.29), not that environment's old
Einsums1 stage. Actual CUDA ABI/device/numerics still untested. Bounded
tiny1thread API/layout probe b20c10275 pending.

Coordinator grants ARTIFACT-ONLY deployment **without any new allocation**:
reviewed immutable recipe, whole same-job E2 built Python+matching native
and completed whole Psi4 stage into a new absent isolated prefix; no
configure/compile/rebuild/old-core overlay/RPATH patch/shared mutations.
Preserve failed roots and statuses. Bound operation below2min with byte/
quota preflight, origin→destination file/hash/ABI/dependency manifest and
the exact three identities above. Then metadata/tinyAPI only, not GPU
readiness. Proposed stdlib installer `artifact_deploy.py` has90s I/O alarm/
deadline, absent-prefix guards, full inventory/destination hashes, symlink
escape rejection and paired readelf audit. Tiny fake-artifact tests
b86b0824f and review d900114d4a94fb4a63bdc9150deedb751 pending before
commit/push/deployment. Artifact count/byte audit bad7ba6d6 and destination
project quota b2669198f pending. No deployment or GPU allocation yet.

Review-control run bda04c116 completed **60passed in6.70s**, no warnings/
skips, JUnit `build_disp0_cpu/reference-check-kP0JNkN5/results.xml`.
Includes optimized-Python fail-fast, each omitted scientific-module manifest,
process cgroupv1/v2/namespace and tightest ancestor checks, inherited group,
quota parsing, prior route/evidence controls and small fullHF/CPHF physical
regression. These are local CPU tests; native logger, deployed-prefix imports,
actual GPU numerics and measured timings remain unvalidated. Follow-up review
and later methodology labels/untimed-work admission remain separately tracked.

Second timing review dbf45f11ac926f5817187e64078bd26ec found further
resource/provenance/preservation gaps. Provisionally fixed: reject unprovable
root-membership/partial bind mounts rather than guessing namespaces; allow
the real cgroupv2 hierarchy root to lack memory.max after a finite visible
descendant bound, but reject missing ordinary controls. Scratch project/group
ownership now comes from a private **new zero-byte inode** in job-owned scratch,
with inheritance flag checked and probe removed outside timed work; query
its actual user/group/project quota rather than assuming the parent ID.
Completed measured wall/components/timers are atomically checkpointed as
**MEASURED_PROOF_PENDING** before expensive diagnostics (not accepted PASS).
Scientific source guard now checks both worktree-vs-HEAD and index-vs-HEAD,
including staged-then-restored cases. Targeted fixtures added; final CPU
validation b6403af75 pending. No installed-runtime or GPU readiness inferred.

Same-build tiny1thread API probe b20c10275 PASS: FP64 GEMM/denominator/
division/reduction at edges1x1/3x4/2x1, default-graph strided-view changed-input
replay and pool/view keepalive/reset. Fixedrtol1e-11/atol1e-12. Psi4 is the
completed same-job installed1.12a1.dev622; E2 is same-job built2.0.0 package.
No SCF/SAPT/benchmark/device work, failed roots unchanged. This narrows the
blocker to clean Python/runtime installation and deployed-prefix validation,
not missing E2 API functionality. Parent informed. Repeat metadata and this
tiny probe on the fresh artifact prefix after deployment; no GPU readiness
claim yet.

Artifact byte audit bad7ba6d6 completed read-only: whole Psi4 stage4803
files/129,175,976bytes; complete built E2 package31files/24,001,971bytes;
native E2 real library37,327,600bytes plus two relative ABI/name aliases.
About190.5MB actual file data before metadata; the installer computes its
exact full inventory and requires2×bytes+1GiB free/quota margin.
This is a bounded artifact-I/O candidate, not a compilation. Project quota,
recipe review/tests and immutable pin remain prerequisites to the cleared
deployment; no copy has begun.

Destination preflight b2669198f PASS: project export8TiB total,
2,102,660,169,728 available bytes. Human-reviewed pace-quota research-group
row reports6233.8/8192GiB used/limit (no finite project file-count limit);
use conservative1900GiB quota headroom versus the installer's approximately
1.36GiB minimum. pace-quota explicitly warns not to script its display, so
do not automate parsing it. Deployment additionally checks live destination
free bytes itself. Quota output is retained in taskb2669198f; this does not
authorize spending or imply timing readiness. Account available at this
snapshot is$28.72; refresh the dedicated selector before any GPU submission.
No copy/deployment begun; recipe unit/review results still pending.

Artifact recipe unit task b86b0824f: **6passed in3.06s**, no warnings/skips,
JUnit `build_disp0_cpu/reference-check-lkoyv8qX/results.xml`. Fake artifacts
only: complete-tree copies/hash identity, absent-prefix/no overlay,
external symlink rejection, source tamper detection with partial preservation,
deadline and optimized-Python rejection. No deployment/native build/device
work. Await recipe review before immutable commit/push and actual cleared
artifact installation.

Tiny local CPU profiler API probe b3de6fda5 launched to check actual
E2 default-graph annotation schema (not only synthetic parser fixtures).
No GPU, physical SAPT or performance measurement; own local artifact directory.

Artifact review d900114d4a94fb4a63bdc9150deedb751 found four safety blockers:
relocated symlinks validated too late, pre-existing prefix manifest overwrite
on failed exclusive mkdir, untracked recipe could claim an unrelated pin,
and unbounded timeout finalization. Fixed provisionally: reject symlinked
directory roots/ancestors and validate the full relocated link graph before
writes; exclusive prefix creation returns device/inode ownership and only
that prefix can receive an exclusive-created manifest; executing installer
must match its tracked pinned Git blob;90s body plus separately hard-bounded
10s finalization (absolute cap110s, exit124 if finalization stalls).
Regular files retain explicit observed destination hashes/modes as well as
origin identities; no binary/RPATH patch. Added targeted race/link/blob/
deadline fixtures. Retest bca7cba61 and follow-up review
d2d20df2919fa617238812c312ef58fa4 pending before commit/deployment.
No copy, allocation or failed-root/status mutation has occurred.

Hardened artifact retest bca7cba61: **14passed in2.61s**, no warnings/
skips, JUnit `build_disp0_cpu/reference-check-kCD153yc/results.xml`.
All four review-specific negative paths are covered. Final frozen rerun
b354041f4 additionally checks the current recipe with explicit destination
hash/mode recording; follow-up review still pending. No deployment begun.

Follow-up artifact review d2d20df2919fa617238812c312ef58fa4 identified two
remaining blockers: pathname replacement between ownership check and writes,
and disarmed timeout before stdout/process exit. Reworked writes to retained
owned directory descriptors: exclusive/no-follow file opens, descriptor-based
copy/chmod/symlink/manifest operations, private new directories and no adoption
of unexpected existing directories. Added replacement-at-write fixtures that
verify foreign source/manifests remain unchanged.

Added a separate parent watchdog through worker output/termination: readiness
barrier before work, pidfd/private process group, waitid(WNOWAIT) to prevent PID
reuse, hard110s kill of only owned worker/readelf descendants. Worker still has
90s body and10s finalization caps; blocked-output fixture added. Retest
b4f02ed0c and final bounded review d020ac4f93e3a496abe2c63a0d1db6c95 pending.
No artifact deployment, compilation, allocation or failed-root mutation.

Root acquisition review d020ac4f93e3a496abe2c63a0d1db6c95 still found a
regular-directory acquisition gap before anchoring. Production recipe now
creates an unpredictable private0700 **unpublished namespace** under an
owner/non-group-writable parent; image and nested acquisition happen below
that trusted anchor. There is no caller/public destination to adopt.
Only the completed actual prefix is disclosed; old/public-name replacement
cannot redirect writes. CLI uses a label, not an exact destination path.
The copy is still whole-artifact, unchanged bytes, no build/RPATH patch.
Private-parent permission preflight b42749f74 pending.

Permission preflight b42749f74 completed: existing runs parent is
**775/UID3170273**, so trust guard correctly blocks it. No chmod or copy.
Updated placement uses a NEW recipe Git checkout created withumask077/
private0700, verified/retained by FD (Git pin/blob checks through
`/proc/<worker>/fd`). Its own unpublished namespace/image is the fresh
runtime prefix, on the same project filesystem/quota. This changes only
new-owned placement, not any existing shared/failed directory permissions.
Full local integration revalidated: **21passed in2.26s**,
`build_disp0_cpu/reference-check-uce6U0WA/results.xml`.
Private Git-anchor review df3cf338f4418293b1c4fcd3b2c302c80 pending.
Earlier f7be8f2d80 is superseded for this parent-selection change; require
an updated immutable pin before any deployment.

Private acquisition tests (regular parent replacement at root+nested gaps)
passed20/20; added a full supervised fake-ELF integration test exercising
tracked recipe verification, private prefix selection, copy/ABI manifest,
output and process termination. Final local run **21passed in2.23s**,
`build_disp0_cpu/reference-check-4boxeCra/results.xml`, no warnings/skips.
Read-only review d21d5f2ee456d6ce788d0aa72af84814d pending before actual
deployment/new immutable pin acceptance. No copy or allocation yet.

Static admission audit (not measured throughput/runtime): for full dimer-basis,
unfrozen spaces, the three product GEMMs require
`(10*Nri+8)*na*nb*nr*ns` real FP64 FLOP. Routine cases are0.221/3.549/
146.069/2995.439TFLOP respectively. Protein157 alone therefore has about
2.995PFLOP/call of product work; six CPU calls would consume about5h even
at a **hypothetical**, unmeasured1TFLOP/s, excluding SCF/CPHF/packing/I/O/
other cases/GPU calls. Do not assume the48-full-call plan fits6h.

Proposed to parent, **not yet adopted**: since full untimed reconstruction
does not establish timed dispatch, replace24 extra full-energy diagnostics
with24 small representative-block same-kernel synthetic post-timing probes
and native positive controls. Preserve24 fresh timed whole calls, matched
science, signed comparisons, policy-controlled/unverified-timed-dispatch
labels and the same single6h allocation/charge ceiling. Await protocol
clearance; otherwise preserve the explicit48 maximum and partial admission.

Protocol clearance subsequently granted: current plan is24 timed fullHF
calls in24fresh processes, plus24 bounded fresh SAME-kernel synthetic
post-timing probes and24 explicitly separate native positive controls.
The48-full-call proposal above is **superseded history**. No full untimed
HF reconstruction remains; no timed adapters/graphs/arrays/DF scratch
retained/reused, natural GPU policy unforced, diagnostic walls excluded
from medians. Probe caps are virtual edges4, occupied widths8, RI64
(≤32wide matrices), plus the separate fixed naturally eligible control.
Policy-controlled8-CPU/hybrid labels and unverified-timed-dispatch caveat
remain. One6h job/charge ceiling unchanged; no extra allocation or retry.
Harness/schema/tests are being updated and are not timing-ready.

Artifact descriptor/watchdog run b4f02ed0c initially **16passed/1failed**
in2.98s, JUnit `build_disp0_cpu/reference-check-xHB1P9v5/results.xml`.
Failure was a capability guard: this conda Python has no pidfd bindings.
Removed the unnecessary pidfd dependency: the independent supervisor uses
private child session/group and `waitid(WNOWAIT)`, retaining the child PID
until supervisor exit so it cannot target a recycled unrelated PID.
No unbounded fallback. Frozen retest via context-mode subprocess:
**17passed in2.10s**, no warnings/skips,
`build_disp0_cpu/reference-check-yGjzEGBp/results.xml`.
Includes actual copy/manifest replacement-window and blocked-stdout
fixtures. Final read-only review d020ac4f93e3a496abe2c63a0d1db6c95 pending;
no deployment before review and immutable recipe pin.

Artifact-only recipe and tests committed as
`aae3064d84683214636c6892d2937d0b15b98930`; Git push b7f16becc in progress.
Commit does not authorize deployment before the pending review; scientific
source remains unchanged from completed build587e172a33.

Updated bounded synthetic protocol/control suite: **70passed in4.75s**,
JUnit `build_disp0_cpu/reference-check-lvMN6sk6/results.xml`. Includes caps,
nonmutation and AST check for exactly one full energy call/no adapter
retention. No GPU/device logger or measured timing run. Protocol review
dc2b71ba92c83f5f5c009a983555d68b0 pending; no campaign launcher submitted.

Protocol review dc2b71ba92c83f5f5c009a983555d68b0 retrieved: **no concrete
blockers/new regressions**. Confirms one timed energy call, production cleanup
inside timing, fresh bounded post-timing probes, separated positive controls,
proof-pending checkpoints and diagnostic-versus-timed dispatch distinction.
Review does not establish installed runtime, native logger functionality or
GPU numerical correctness. Committing this as preparatory protocol/harness,
not a ready/submitted campaign or measured speedup.

Final acquisition review d020ac4f93e3a496abe2c63a0d1db6c95 found remaining
regular-directory replacement gaps before FD anchoring. **Do not deploy
the earlier aae3064d84 recipe.** Reworked acquisition into an unpredictable
private0700 nonce namespace below a verified trusted parent, not a
caller/public destination. Image/nested acquisitions are below the private
anchor before its path is disclosed. The actual fresh prefix is reported
only after complete verification; no predictable public target is adopted.
Added regular-parent replacement fixtures at root/nested acquisition gaps.
Local retest: **20passed in2.09s**, no warnings/skips,
`build_disp0_cpu/reference-check-9IPy4Izc/results.xml`.
Private namespace review d21d5f2ee456d6ce788d0aa72af84814d and read-only
parent permission preflight b42749f74 pending before a new immutable pin.
No artifact copy, compilation, allocation or failed-root changes.

Parent overlap update acknowledged:14185006 completed0:0, allocation
released, no holder to borrow. Its24PBE0-D4 records/results and unresolved
protein157 matched-work acceptance remain separate; do not use them as an
HF baseline or modify parent artifacts. Our readiness/correctness/capacity
gates remain unchanged.

Unpublished-namespace review d21d5f2ee456d6ce788d0aa72af84814d retrieved:
**no concrete functional or namespace-safety blockers** under the stated
threat model. Confirms private-before-publication acquisition and retained
directory descriptors prevent public-path replacement from redirecting
writes; root/nested regular-directory replacement fixtures are covered.
Read-only review, not a test rerun or runtime deployment proof. The final
private Git-anchor review and exact recipe push receipt remain prerequisites
before the first artifact copy.

Coordinator protocol clearance received for a **separate immutable recovered-
runtime admission**. It must bind original FAILED14196330/exit/provenance,
exact compilepin587e172a33afaadb00eb7f574a5089533f7fc755, fixed b3/8f/9e
whole-artifact identities, fresh complete deployment manifest/prefix and
new metadata/maps/tinyAPI proofs, plus scientific-source equality. Existing
failed-build admission must continue to reject failed status; no generic
bypass or old-root/status mutation. No new CPU allocation is authorized.
The local26/64 signed baseline is historical support only. Independent
native/eager/defaultCPU signed physical references must execute with the
actual recovered stage inside the authorized30min GPU gate before accepting
GPU physics or timing. Report admission pin/manifest/prefix/proofs and fresh
exact-route/gts-cs207 capacity/cost readiness before the single no-requeue
GPU submission. Artifact14e still requires final narrow review/push and
installed-prefix proofs before this transition.

Final narrow anchor review df3cf338f4418293b1c4fcd3b2c302c80 exhausted its
five-turn limit without a committed answer; this is **not review clearance**.
Replacement scoped review d45a0182cbb14280ba8e790ab826cfaae launched with
adequate answer budget; artifact installation remains unattempted.
Fresh read-only storage/quota task b435a7ee8 and original failed-status/
receipt/accounting evidence task bff3c2587 are running; no allocation.

Separate admission implementation drafted in `devtools/disp0/recovered_runtime.py`
without touching the original failed-build admission. Local fixture suite:
**18passed in1.66s**, JUnit
`build_disp0_cpu/reference-check-CMsZKLM7/results.xml`.
Checks complete file inventory/hashes/modes/links, exact approved tuple,
failed-origin identity, original toolchain/cache/header-link provenance,
fresh prefix proofs and scientific equality; explicitly reports preparation
only and historical CPU support, never GPU physics. Read-only review
d12d077bd68ee10ab815ffc4ca3ad76ac pending. Real recovered-prefix proofs and
actual original-provenance schema still need validation before use.

Exact14e135d23c push receipt bda5ab64e confirmed origin updatef7be→14e.
Staging a NEW umask077 private independent detached Git checkout through
bda0dc11b; no artifact copy until final review/quota clearance.
Timing-input discovery bb185c4cc found
`/storage/home/hcoda1/3/awallace43/data/timing_test_suite/geometries/systems.py`,
module SHAd949b063a052fce2c0f43c2bf7f50bccd68644b6917f6de2c9c40a28833df154,
which **differs** from local planned module72668390…. Do not silently
substitute it or assume matched geometries. Read-only AST literal comparison
of the four exact geometry-string hashes launched as b13b22dcb; no import,
SCF, source edits, or timing admission.

Fresh artifact storage preflight b435a7ee8 completed successfully. Live
project filesystem free2,073,195,184,128bytes; human-reviewed pace-quota
project6261.2/8192.0GiB, inode limit0(unlimited). No scripted parsing of
its unstable display. Conservative quota admission **1800GiB** comfortably
exceeds selected artifact bytes≈190.5MB and required2×bytes+1GiB≈1.36GiB.
Evidence: `.pi/tasks/01a154d8-a459-768d-83bd-c57c6a5dc42c-4065461/b435a7ee8.output`.
Snapshot gts-cs207available$29.35 is not the mandatory fresh pre-submission
capacity selector and does not authorize submission. No artifact copy yet.

Final narrow private Git-anchor review d45a0182cbb14280ba8e790ab826cfaae:
**no blockers** in main/verify_recipe/PrivateNamespace/private-anchor fixture.
Exact-pin, `/proc/<workerPID>/fd/<repoFD>`, namespace creation and CLI are
functionally consistent. Read-only review, no test rerun. Recipe14e push and
fresh byte/quota evidence are complete; awaiting the private Git staging
receipt before the first single bounded artifact installation.

Original-failure evidence bff3c2587 confirms unchanged14196330 status
exit_code1/term_received0, correct exact run/compilepin, submission receipt
accepted_job_id14196330/Requeue0, cpu-small/embers/gts-cs207-chemx/8CPU48G2h.
Live accounting job and.batch both **FAILED1:0,40m04s**; .0FAILED1:0
39m57s. Extern0:0 is not success. Receipt/status schema matches separate
admission expectations. No relabel, old-root changes or new allocation.

Separate-admission review d12d077bd68ee10ab815ffc4ca3ad76ac identified two
binding gaps: unlisted destination files and unchecked non-ELF origin bytes/
links. Fixed both: exact enumeration of deployed whole Psi4stage and E2lib,
plus every source/destination kind/mode/hash/size/link rechecked. Added extra
module/library and changed-origin Python/link fixtures. Also binds live
read-only sacct original job+.batch FAILED1:0 and original account/QoS/
CPU/memory. Retest: **27passed in1.81s**, no skips/warnings,
`build_disp0_cpu/reference-check-0eUQAqw4/results.xml`.
Scoped follow-up d4970cca274b40cc7d84c36316a1e1de2 pending; no runtime/
numerical admission or GPU submission yet. Original preparation script
remains unchanged and rejects the failed status.

Scoped recovery follow-up d4970cca274b40cc7d84c36316a1e1de2 retrieved:
**no blockers**. Confirms exact deployed-tree inventory, complete original/
destination kinds/modes/size/hash/link audits, and original job+.batch
FAILED1:0/account/QoS/CPU/memory binding. Tests inspected, not executed.
This clears the separate admission implementation for immutable pinning;
actual new-prefix proofs remain required and no GPU correctness is implied.

Remote input query b13b22dcb failed with `KeyError: protein83` in the initial
literal SYSTEMS dictionary. This is an input-schema/provenance failure,
not an energy/tolerance result. No geometry has been substituted or admitted.
Read-only AST structure discovery b93143610 launched; parent requested to
provide the immutable suite path/hash used for its finished cohort if known.
No parent input/source/data changes.

Private Git staging bda0dc11b completed: new checkout
`P/gits/psi4.saptdft-cuest-disp0-artifact-14e135d23c-resume1`,
mode0700/UID3170273, exact detached14e135d23c9e9de3a68ca5ead0eed43413cc86c8,
clean tracked tree. Remote receipt time2026-10-20T05:51:29-04:00.
All installation prerequisites now met. Exactly one artifact-only installer
launched as b76327e4b with110s watchdog, fresh unpublished prefix, conservative
1800GiB quota receipt, no compilation/allocation/retry. Actual outcome pending.
The dated08:00 deadline is not promised; no timing work has begun.

User supplied exact immutable input snapshot
`P/runs/saptdft-cuest-cpks/scal631-fc851a8346-20261016/geometries/systems.py`,
SHA72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb.
Read-only byte/geometry identity verification b1958e914 launched. This
resolves which source to inspect, not yet a verified timing admission.
Parent snapshot is read-only, not copied/edited or used as an HF energy baseline.

Read-only home-copy schema inspection b93143610 confirms an older three-key
SYSTEMS dictionary (peptide/nanotube/protein157, no protein83) and different
NBF metadata. It is excluded from campaign admission. Only the supplied
immutable72668390… snapshot will be considered; verification b1958e914
pending. No home-copy or parent-snapshot edits.

Supplied snapshot verification b1958e914 **INPUT_BYTES_VERIFIED**:
module72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb;
all four exact geometry-string hashes match the predeclared local inventory
(peptide14b4041b…, nanotube5010ea28…, protein83c4bc7429…,
protein157a1e9c894…). This is input-byte identity only, not matched energies,
runtime readiness, GPU correctness or timing. Parent snapshot unchanged.

Artifact installation b76327e4b **ARTIFACT_BYTES_VERIFIED**, exactly
190,505,547bytes,59.515536654s (within110s watchdog/<2min clearance).
Recipe14e135d23c9e9de3a68ca5ead0eed43413cc86c8; fresh prefix
`P/gits/psi4.saptdft-cuest-disp0-artifact-14e135d23c-resume1/runtime-587e172a33-14e135d23c-sxist_u6/image`.
Complete per-origin/destination/hash/mode/link/ELF manifest:
`<prefix>/deployment-manifest.json`. No compilation, old-root/status changes,
allocation or retries. New-prefix matching --psiapi/import/maps and tiny
one-thread API/layout/pool probe b9ec91c8f running; no physical/GPU claim.

Separate admission committedb4ffd6595044c3fc6a0a97176acbf3edb0927152,
pushb095c9076 pending. Explicit recovered gate integration drafted without
changing original successful-build admission/payload. It revalidates the
hash-bound recovered admission and records FAILED compilation origin, not a
successful CPU preparation. Gate retains actual recovered-stage native/
eager/defaultCPU signed oracles before GPU acceptance. Local focused
admission/evidence/batch suite **65passed in3.03s**, XML
`build_disp0_cpu/reference-check-cOQvE8NM/results.xml`.
Integration reviewdd650be73a5e54b3c13c9d995604bb79e pending.
Parent checkpoint sent with artifact prefix/result and request for separate
ONE bounded<2min compilation of the existing generic forwarding diagnostic
(no scientific/E2/Psi4 compilation or device operation). No CPU allocation
requested; artifact-only clearance is not treated as compile permission.

Separate diagnostic compile clearance received: ONE new isolated prep root,
one CPU thread, aggregate<120s, modest memory/byte limits, no retries/fallback,
read-only GNU14.3/CUDA13 inputs, no deployed-prefix/shared-env mutation or
E2/Psi4/scientific compilation, no CUDA/device/numerical execution. Script
updated for110s timeout(+5s forced termination),4GiB virtual-address ceiling
(large read-only NVIDIA ELF mappings),64MiB output-file bound, explicit
compiler/header/helper/command/output/library hashes, ELF and resolve-only
result, compiler resource usage. Admission creation remains a **separate**
operation; diagnostic prep now only reads its hash-bound record and never
writes the deployed prefix. No diagnostic compilation has executed yet.

**Installed-prefix proof FAILED** b9ec91c8f. Relocated `--psiapi` correctly
selected the new Psi4stage, but `sealed.mapped_einsums_libraries()` reported
`P/runs/saptdft-cuest-disp0/build-587e172a33afaadb00eb7f574a5089533f7fc755-14196330/einsums-build/lib/libEinsums-abi1.so.1`,
not the new prefix. Assertion in runtime_provenance.py:39 correctly stopped
the proof. Metadata output remains partial/empty; tiny API tests **not run**.
Do not treat the artifact byte/ELF copy pass as a relocatable runtime pass.
Stopped diagnostic compile and GPU preparation; no compile/GPU submission.
Read-only recorded/current ELF diagnosisb13607b43 launched and parent notified.
No RPATH patch, old-core overlay, prefix/failed-root repair, retry or version/
mapping-check bypass. Local pinned E2 sealed implementation uses actual
`dl_iterate_phdr` enumeration, not a compile-time path label; ELF evidence
is needed to identify the loader binding cause. Task9 records this blocker.

Recovered-gate integration reviewdd650be73a5e54b3c13c9d995604bb79e:
**no blockers** in scoped recovered admission/branch/payload. Confirms original
failed-build rejection and recorded failed origin, plus actual recovered
native/eager/defaultCPU references before GPU acceptance. Read-only review,
no tests rerun or runtime proof. New one-shot diagnostic timeout/resource/
read-only-scope guards are covered by local static fixture; focused suite
now **66passed in3.03s**, no skips/warnings, XML
`build_disp0_cpu/reference-check-4KQoZysn/results.xml`; all three shell files
individually pass `bash -n`. These are harness results only. Runtime remains
blocked by failed installed-prefix mapping proof. No diagnostic compile or
GPU submission has run; the one-shot compile authorization is unconsumed.

Read-only ELF diagnosisb13607b43 confirms actual **DT_RPATH**, not RUNPATH:
E2Python `_core` lists original14196330/einsums-build/lib first, overriding
new LD_LIBRARY_PATH and explaining the failed mapping proof. Both E2Python
and native include siteCUDA12.6/lib64 before explicitcuda13; native also has
an empty `::` path entry. Recorded origin/destination ELF is identical, so
this is retained build linkage, not a corrupted artifact copy. Actual
CUDA-component resolution is being inspected read-only as bf472f992; no
preload workaround, loader/version-check bypass or binary/path patch.
Parent notified that the fixed tuple is not relocatable under the cleared
contract. No extra CPU allocation, compilation or GPU submission authorized/
executed to resolve this failure.

Read-only exactd17 source audit (local E2HEAD also exactd17) found no Python
install() rules in `cmake/Einsums_FinalizePybind.cmake` and no packaging
pyproject/setup entrypoint. The historical EINSUMS_INSTALL_PYMODDIR=naive
argument did not provide a Python install target. README now explicitly
marks that configuration as historical/unready rather than recommending
another identical failed build. Any new install/loader contract or coherent
rebuild must first be reviewed and separately cleared; no repair executed.

Preparatory gate checkpointd7c1517d06cccce232582d75ab0bcdc76d514371 push
confirmed byb8df6762d (originb4ffd→d7c1517d06). This pin does not admit the
failed relocated runtime. Neither H200 authorization nor the separate
one-shot diagnostic compilation authorization has been consumed.

Read-only dynamic-loader tracebf472f992 confirms **mixed CUDA dependency
selection** under the intended LD_LIBRARY_PATH: cuSPARSE.so12 and
nvJitLink.so12 resolve to siteCUDA12.6, while cuBLAS13/cuBLASLt13/CUDART13/
cuSOLVER12 and an additional nvJitLink13 resolve through explicitcuda13.
No missing-library error does not imply ABI coherence. This is loader-trace
evidence only, not device execution. Parent notified. The failed prefix
cannot satisfy the current admission contract; no preload/RPATH/dependency
patch, mapping-check relaxation, retry or extra allocation.
One final read-only original CMake cache/explicit-toolkit identity audit
b2b62ea4e is running to specify a minimal corrected recipe proposal.

Final read-only auditb2b62ea4e confirms compiler/root explicitly point to
CUDA13 and sm90, yet original cache CUDA_cusparse_LIBRARY and
CUDA_nvJitLink_LIBRARY point to siteCUDA12.6 (54 site12.6 cached selections
in the inspected CUDA/configuration subset). Explicitcuda13 unversioned
`libcusparse.so` and `libnvJitLink.so` do not exist; cuBLAS/CUDART development
links resolve to genuine13.6.0.2/13.3.29 libraries. This is an incomplete
development-toolkit/component-selection contract, not merely a misleading
CUDA_ROOT locator. No configure/build/package/shared-prefix mutation occurred.

Bounded correction handoff: `devtools/disp0/runtime_correction_plan.md`.
It proposes a complete NEW private CUDA13 development prefix, configure-time
imported-target closure rejection, complete E2 Python/native install rules
and deliberate installed RPATH/RUNPATH, fresh coherent build/stage and a
new explicitly reviewed artifact tuple. No SONAME aliases, binary patch,
preload workaround, old-object repair or E1 downgrade. Candidate CPU shape
8CPU/48G/up-to2h/freeembers is planning only; no corrected executable recipe/
immutable build pin or new CPU approval exists. Existing diagnostic/GPU
authorizations remain unconsumed. No credible08:00 speedup promise.

Parent explicitly cleared OFFLINE executable preparation; no additional
planning approval is being awaited. Task10 active; runtime/task5 remains
blocked by task9. New tracked recipe draft comprises private CUDA SDK
identity/materialization/ELF/closure guards, deferred CMake configuration/
complete-package install hook, actual runtime probe, same-process API/
physical pytest proof, successful-new-build admission and one proposed CPU
SBATCH/payload. Only devtools/tests change; exact local E2HEADd17 and its
tracked source are unchanged, and `git diff587 -- psi4/` is empty.
The original failed-runtime admission/old tuple stay rejected.

Offline positive/negative fixtures cover missing/ambiguous/mixed CUDA
components/headers, private aliases without source mutation, actual closure,
bad loader paths, deferred install/configuration targets, whole-package/
binary/source/process/physical proof binding, failed/signaled allocations,
explicit CPU approval and resource-route guards. CMake fixtures use
LANGUAGES NONE and deliberately nonexecutable build commands; no compiler,
CUDA, SDK preparation, installation, full dependency/scientific rebuild,
diagnostic compilation or resource job has run. Combined current guard
suite: **127passed in5.69s**, no skips/warnings,
`build_disp0_cpu/reference-check-hs7vu57I/results.xml`.
An additional required cuBLASLt public-header presence check was then added
and will be included in the final retest.

Read-only upstream E2 packaging-seam investigationd1e572c41ab501a0befcdca7387a6ce33
and revised-contract reviewdbbb8422a7b400be2d62c1d288be6b490 pending.
`devtools/disp0/revised_build_contract.md` documents exact executable inputs/
guards and ONE proposed8CPU/48G/2h/freeembers/gts-cs207-chemx job. This is
not a CPU approval request yet: final review, immutable source pin and exact
guard evidence must be returned first. Complete real SDK/dependency input
identities remain required; fake/offline guard passes are not installed-
runtime readiness or GPU correctness.

Upstream packaging investigationd1e572c41ab501a0befcdca7387a6ce33 retrieved:
external root-deferred configuration/install hook is viable without an E2
source patch. Confirms native targetEinsums inlibs, Python targetPyEinsums
inroot, existing native install and absent Python install. Additional
constraints incorporated: require shared native/prefixlib layout, reject
skipped install-RPATH policy, explicitly build PyEinsumsStubs before install,
and require `_core.pyi`/py.typed. Build and installed paths remain distinct;
actual RUNPATH and indirect CUDA loader closure are checked rather than
assumed transitive. No upstream configure/build/install was executed.

Updated offline guard run: **136passed in6.77s**, no warnings/skips,
`build_disp0_cpu/reference-check-RcGadfuA/results.xml`. All three revised/
optional-proof shell files pass syntax checks individually. Coverage now
also includes native shared/stub-target/install-layout/RPATH requirements,
actual compiler/architecture/library cache binding, immutable input receipts
and actual API/physical process identities/components. No SDK inspection,
dependency install or native compilation was executed. A final mixed-
CUDA12/13 inventory rejection fixture was added afterward for next retest.
Main contract reviewdbbb8422a7b400be2d62c1d288be6b490 remains pending;
no final recipe pin or CPU approval request yet.

Reviewdbbb8422a7b400be2d62c1d288be6b490 found three concrete blockers:
missing transitive SDK inputs, incomplete actual mapped-library/Psi4 closure
checks, and missing effective compile/include/link audit. Fixed in the
offline draft: full include/bin/NVVM tree identities with required CRT header/
cudafe++ and cycle checks; actual all-native process maps plus Psi4 loader
closure; new build_audit.py captures compile databases and Ninja commands/
actual dependencies/response-file expansions, validates compiler/path/header/
link evidence before acceptance. NVCC flags already sanitized; launcher/
ambient-path cleanup and verbose source-build logs are explicit.

First expanded rerun retained **3failed/158passed**, XML
`build_disp0_cpu/reference-check-QaOWQ3G5/results.xml`: generic historical-
build path policy incorrectly rejected the intended NEW Psi4 installed
stage. Fixed with an exact explicit installed-stage exception only; historical
job/site12/build-tree mappings remain forbidden. No tolerance change or
runtime retry. Corrected expanded offline suite **161passed in7.75s**,
`build_disp0_cpu/reference-check-mBaCee31/results.xml`. Includes real Ninja
response METADATA expansion with a nonexistent compiler; no object/shared
library or actual build executes. Follow-up review
d9b943acc60714330d8a980540948b57d pending.

Shared GPU policy relayed to parent and FDDS owner through workmux; no active
off-registry CPHF/CPKS agent was found by parent, no arbitrary pane input.
FDDS is sole owner; disp0 has no immediate READY GPU need/allocation and must
not hold it idle. No duplicate/direct-srun request; current shared128GiB
cannot support512GiB proteins. Owner's first8e9/interactive-01 acquisition
failed before salloc; discard it for launch, preserve evidence. Corrected
controller9a026ca149/interactive-02 is only a future endpoint, **NOT READY**;
await actual manifestallocated/job/fullUUID/ownerGO+serialized window. Only
the owner controller may hold commonflock through exclusive step/cleanup.
Runtime correction remains OFFLINE with no new CPU approval, actual SDK
inspection/install, full rebuild, diagnostic compile or allocation executed.

Follow-up reviewd9b943acc60714330d8a980540948b57d found two remaining input
audit gaps: transitive CUDA headers under otherwise allowed roots, and
embedded linker script/version-map forms. Both addressed: critical
driver/vector/builtin/nv-target headers required; complete SDK root-name/
namespace identity classification catches foreign copies while standard
host headers remain legitimate. Linker argument parsing now validates
version scripts, -T scripts, packed/equal/relative and -Xlinker forms.
Negative fixtures reproduce every reported counterexample. Updated offline
suite **170passed in7.52s**, no warnings/skips,
`build_disp0_cpu/reference-check-IhfgnghE/results.xml`.
No real SDK, native build, diagnostic compilation, resource job or GPU
execution occurred. Original failures and old fixed tuple remain rejected.

Final targeted reviewdd3490bef162184f2a13a686f1b412629 confirmed the CUDA
header findings and reported one forwarding counterexample:
`-Xlinker --version-script -Xlinker ../../foreign/exports.map`.
Fixed by normalizing all -Xlinker/-Wl forwarding into one ordered linker
argument stream before path auditing; the marker can no longer be consumed
as a filename. Added the exact negative, truncated-forwarding negative,
own-root positive and mixed-forwarding positive. Final local suite:
**172passed in8.35s**, no warnings/skips,
`build_disp0_cpu/reference-check-FhEYT3uB/results.xml`.
Recipe commit6f5536c989 is superseded for execution by the corrective
normalization commit; do not submit the earlier pin. All prior evidence is
preserved. No new SDK operation, rebuild, diagnostic compilation or resource
allocation. CPU proposal remains one8CPU/48G/2h/freeembers/gts-cs207-chemx/
noGPU/no-requeue job only after fresh explicit approval and pinned input/
route/headroom checks; no approval requested before returning the final pin.

Final static confirmationd7c65a69eeeb89bf3228306f4bf3f0337 retrieved:
**specific final finding resolved**, no remaining counterexample within the
forwarding scope; fixtures inspected,172-test result not independently rerun.
Executable recipe pin is **89b6160126d876b2ad649e6cf2a22f01f4b300f7**,
origin push confirmedbee4dadc4 (6f5536→89b616). Working tree was clean,
Python AST/four shell syntax/whitespace checks pass; scientific587 equality
underpsi4/ and exact unchanged E2d17 tracked source were verified.
Exact changed-source scope from5da211 to89b616 is18 paths: devtools/disp0
configuration/packaging/audits/proofs/fixtures/docs, three existing devtools
build/API/reference helpers, two pytest files and this progress document.
No scientific source or shared/parent roots changed.

Offline task10 completed. Fresh CPU approval REQUESTED through parent,
not granted or submitted: ONE8CPU/48G/02:00:00/noGPU/no-requeue job,
cpu-small subject exact-route, freeembers/gts-cs207-chemx,16CPU-hours max/$0.
Complete real SDK input manifest and dependency-lock SHA remain UNPASSED
prerequisites; their preparation is not implied by local fake guard tests.
Source/push/inputs/test-only/headroom/quota/no-duplicate checks must precede
any submission. Task11 tracks approval/input receipts. Runtime/GPU physics
remain unpassed. No SDK operation, native build, diagnostic compile or new
CPU/GPU allocation ran. SharedFDDS-controller-only/no512GiB/noidlehold policy
continues; no immediate READY window requested.

This later progress-only record does not replace executable pin89b6160126.

Parent granted CONDITIONAL resources for exactly one CPU-only build at89b616:
1node/1task/8CPU/48G/2h/noGPU/freeembers/gts-cs207-chemx/no-requeue,
fresh roots/no retry. **NOT SUBMISSION GO**. Explicit parent INPUT-READINESS
CLEARANCE after real completeSDKmanifest+SHA/exactdependencylock+SHA is
mandatory before subsequent fresh source/route/headroom/quota/duplicate
checks. GPU requests remain held for FDDS-owner assigned shared02 window.

Separately cleared <=120s aggregate one-thread bounded read-only SDK/lock
inspection b611c8f6e completed in**0.401507902s**,1GiB address-space ceiling,
streaming hashes, no remote writes/materialization/compile/install/device/
numerical call. Result **INCOMPLETE_SDK_NO_BUILD**, not an input pass:
`P/software/toolchains/cuda13/include/cusparse.h` and `nvJitLink.h` missing.
Their unversioned development links are also absent. Versioned genuine
candidates are cusparse12.8.2.51(SONAME12) and nvJitLink13.3.33(SONAME13);
nvcc is13.3.73. Other inspected required bin/NVVM/CRT inputs exist. Full
SDK inventory was not attempted because mandatory inputs were missing;
partial identities are not a complete SDK manifest.

Exact immutable dependency lock:
`P/runs/saptdft-cuest-disp0/build-587e172a33afaadb00eb7f574a5089533f7fc755-14196330/environment-explicit.txt`,
11018bytes/@EXPLICIT,
SHA256 **3f9a508fd601306f548a9088f5cd304367a5fde3988e8bbd4fe00d9e6efb0205**.
Includes GNU14.3 implementationbuild20/wrappers28, CMake4.3.2,
Clang22.1.4/LLVM22.1.8. Raw numerical flags were empty; raw PMIX CPATH/
LIBRARY_PATH and LD_LIBRARY_PATH with empty entries were retained in the
task output, not adopted as effective build inputs. Future effective flags
remain unexecuted/unverified; unknown numerical drift is not approved.
Evidence: `.pi/tasks/01a154d8-a459-768d-83bd-c57c6a5dc42c-4065461/b611c8f6e.output`.
Parent received exact failure/lock digest; no input clearance or CPU job.

Read-only matching installed-package/cache metadata lookup bab0e76b0 is the
bounded continuation (85s maximum plus prior0.4s,1thread/512MiB), solely to
identify available matched development inputs for a separate private-input
preparation proposal. No download/extraction/install/materialization or
shared mutation is authorized/performed. Conditional build approval remains
unconsumed; all prior failed roots and fixed-tuple rejection remain intact.

Final timing safeguard test task b6403af75: **68passed in6.48s**, no
warnings/skips, JUnit
`build_disp0_cpu/reference-check-ctS1hjRJ/results.xml`. Includes staged/
unstaged/restored-index source pin rejection, realistic cgroupv2 root
without memory.max, namespace/bind fail-closed checks, actual scratch-probe
ownership/project inheritance and cleanup, prior guards and small fullHF/
CPHF signed/global regression. No actual timed process, native CUDA logger,
installed recovered runtime or GPU numerical gate has run.

Real local CPU profiler microprobe b3de6fda5 PASS: observed
`physical/cpu-api/edge-1` → `ComputeGraph::execute(sapt0-dispersion-products)`
with actual `op_kind=Einsum`, `device=CPU` annotations. This confirms the
CPU-edge parser schema beyond synthetic fixtures. Artifact:
`build_disp0_cpu/profile-api-y9bZgU0K/profile.json`.
Tiny matrix API work only; no GPU/physical SAPT/performance claim.

### Matched private developer input attempt (preparation)

Parent subsequently cleared ONE private INPUT provision, not another planning
roundtrip or CPU submission: new absent0700 prefix,120s aggregate/1thread/
1GiB RAM,128MiB download/512MiB written maximum, no retry/solve/compile/device/
shared mutation. Runtime cache lookup bab0e76b0 took0.2563s and found only
runtime packages, no developer headers.

Public conda-forge release metadata supplies exact matching developer inputs:

* `libcusparse-dev-12.8.2.51-hecca717_0.conda`,54443bytes,
  SHA256`c472d4f6687017a42b1a392ba4064dbe07ee499d77a61558c04bfd7e6b81b163`.
* `libnvjitlink-dev-13.3.33-hecca717_0.conda`,28126bytes,
  SHA256`4d0c356194dc359142147236efb56099c0f3d745e6239e51627fa8a97cfaf229`.

Metadata URLs are `https://api.anaconda.org/release/conda-forge/NAME/VERSION`;
exact download URLs use
`https://conda.anaconda.org/conda-forge/linux-64/ARCHIVE`.
Combined archive bytes82569, no latest/full toolkit/runtime replacement.
New tracked `provision_sdk_inputs.py` is a separately pinned INPUT tool,
not a change to conditionally approved CPU executable89b616. It validates
whole package SHA/index/header/link identities before materializing headers,
copies the complete include tree privately, retains read-only bin/NVVM/lib64
inputs and supplies genuine devlinks under private `lib/`. Complete89 SDK
inspection and read-only ELF/ldd closure follow within the same deadline.
The old explicit lock is a permitted read-only input, not a loader path.
No provision/build/GPU result is implied by this preparatory record.

Initial input guards b635d8517:12passed1.36s, JUnit
`build_disp0_cpu/reference-check-H0WpGeuz/results.xml` (fixture archives,
no download/compile). Read-only locator check b7eb0c16d took0.80054s:
2129 include files/22211755bytes; base Python zstandard0.19.0 available.
The existing nvcc shell wrapper delegates to
`P/miniconda/envs/p4cuest_sapt/bin/nvcc`, injecting original PATH/LIBRARIES
for verbose identification. Review d275c9028 confirms overlay compatibility
with89 inspection/preparation but **does not prove compiler discovery uses
the copied include root**; unchanged89 correctly rejects original imported
include directories outside that root. No wrapper edit/fabricated metadata/
compiler invocation is permitted. The provisioning receipt additionally binds
the real vendor nvcc/profile identities and retains discovery_ready=false.
This limitation must go to parent input-readiness review, not become an
implicit successful build admission.

Review db7f140b07ff249173fb4a6f623ce7395 requested `python -B` and protected
destination rejection; both are incorporated, plus a mocked successful
provision and nesting-negative tests. Its suggestion to replace input
`lib64` with a symlink-filled private directory is NOT adopted:89 canonicalizes
`library_root` to that directory and then rejects canonical imported targets
outside it. Instead original canonical `lib64` stays identity-bound, input
`lib/` developer links are separately hash/SONAME-bound in the receipt, and89
supplies its own private aliases. This matches the independent overlay review.
Package bytes/hashes are exact; no fallback/solve/compile/device work exists.

Final input safety guards bfa2f3b6d:16passed1.39s, no warnings/skips,
JUnit`build_disp0_cpu/reference-check-J2QIEGgw/results.xml`.
Compatibility guards bb5b5e236:108passed6.14s,
JUnit`build_disp0_cpu/reference-check-hL3IRFbM/results.xml`.
AST/whitespace PASS; approved89 runtime_contract/payload and scientific
sources unchanged. Input helper SHA256
`f47c959fa2bc0f50b3f4ba629cb51602c2ce548f681bb0d2bc89dcff0b036a00`.
Next execute the already-cleared single bounded attempt from its pushed
immutable tool pin; this is not an additional CPU approval request.

Tool pin`8b34329bee12f30fd5365bdb4666142f811ccaa4` pushed and verified by
`git ls-remote` (b04392027). ONE authorized input attempt b1e4d02d6 launched:
fresh0700 root`P/gits/psi4.disp0-sdk-inputs-8b34329bee-01`, Git-only shared
read-only object origin plus exact pinned fetch/sparse checkout; destination
`root/private-inputs`. The entire staging+provision command has outer110s
timeout/5s kill,1GiB address ceiling and one CPU affinity; no retry. Python-B
prevents import cache writes outside the new prefix. This launch record
does not assert successful provisioning/input readiness/build submission.

**Attempt b1e4d02d6 FAILED**, exit1 after3s aggregate. New root/source and
partial private-inputs prefix preserved; no retry. Staged exact tool pin8b34329bee
before execution, affinity CPU0/1GiB address bound/outer110s+5s guard.
First public cusparse developer archive54443bytes passed exact size/SHA
checks, then zstandard0.19.0 rejected its compressed frame with
`Frame requires too much memory for decoding` at configured
`ZstdDecompressor(max_window_size=128*1024)`. Decoder-window limit/units and
frame requirement need local diagnosis; this does NOT demonstrate1GiB actual
memory exhaustion. Unit fixtures used uncompressed stream injection and
therefore did not test this real decoder behavior.

Failure preceded all header copying/extraction, developer-link creation,
SDK inventory/closure or manifest/receipt writes. Second archive was not
requested. Git staging and fresh directories are the only persisted writes;
downloaded first archive was validated in memory before the failing decoder.
No complete SDK manifest exists, no INPUT-READINESS clearance, no CPU/GPU job
or scientific/diagnostic compilation. Parent received exact failure immediately.
The one input-provision attempt is consumed; any corrected remote operation
or bounded input-prep CPU work needs separate explicit authorization.
Conditional CPU89 approval remains unconsumed/NOT submission GO.
Evidence `.pi/tasks/01a154d8-a459-768d-83bd-c57c6a5dc42c-4065461/b1e4d02d6.output`.

LOCAL-only diagnosis (zstandard0.25.0, distinct from remote0.19.0):
an actual streaming frame with8MiB advertised window reproduces the same
error with raw max_window_size131072;134217728 succeeds on8192payload bytes.
The prior constant was a128KiB bound, not128MiB. Corrected local helper
names128MiB explicitly and adds a real streaming-frame regression; no remote
retry or fresh archive download was made. Old pinned code/source remains
unchanged. Compressed frame tests are preparation evidence, not validation
of actual remote package contents or a complete input pass.

Sent parent bounded alternative proposal (NOT approved/submitted): ONE
input-only1node/1task/1CPU/2G/10min freeembers/chemx job, cpu-small subject
exact-route, no-requeue/no retry,1/6CPUh/1/3GiB-hour/$0 maximum. Only the same
two exact archives/private input manifests/ELF closure/lock proof; no solve/
E2/Psi4/scientific build/SCF/SAPT/device. Keep1GiB address/512MiBnewwrite/
110s payload guards, fresh roots, source/storage/headroom/duplicate checks.
This is separate from conditional unconsumed CPU89 resources; no submission
GO exists for either. Shared FDDS GPU hold and no512GiB proteins unchanged.

Local corrected input guard suite bd5bff370: **17passed1.39s**, no
warnings/skips, including real compressed streaming-frame window regression.
JUnit`build_disp0_cpu/reference-check-nZhWZLxV/results.xml`.
This does not rerun remote0.19 or real developer archives. No corrected
provisioning/CPU preparation job was executed. Preserve the consumed failed
attempt as an explicit operational milestone; await fresh authorization.

Parent subsequently granted fresh approval for EXACTLY ONE INPUT-ONLY CPU
job:1node/1task/1CPU/2G/10min/freeembers/chemx/cpu-small subject exact-route,
no-requeue/no retry,110s payload/1GiB process, same TWO pinned82569B archives,
<=512MiBnewwritten. Does NOT consume or clear conditional CPU89. New tracked
`input_prep.sbatch`/`input_prep_payload.sh` preserve failed roots and require
frozen source, correct actual route/resources/approval, fresh run+scratch,
TERM-safe status. Before downloading/extracting anything, actual remote
zstandard0.19 must pass an8MiB-window frame negative131072/positive128MiB
control. No unbounded decoder/version fallback. Current preparation is local
and preflight only; no input-only job submitted yet. Complete SDKmanifest/
closure/lock evidence still must go to explicit parent INPUT-READINESS.

Input-only harness bf3591950: shell syntax PASS, **19passed1.41s**,
no warnings/skips, JUnit
`build_disp0_cpu/reference-check-7KjSzbmf/results.xml`.
Tests include real LOCAL0.25 codec frames and version-rejection/injected
unit metadata, NOT actual remote0.19 proof; only the new approved job can
supply that startup proof. Freeze/push this tracked input-only recipe before
fresh exact-route/source/quota/headroom/no-duplicate submission checks.

Frozen input-only source`e396e97215158830bad3fbee265a20ebeeb2bbd7`;
push verification bd12e76a1 pending at this record. Quota/capacity snapshot
b94f79888 PASS: embers21/50 slots, no disp0-sdk-input duplicates; project
6272.5/8192GB quota and37281129942 filesystem free inodes, scratch
3556.8/15360GB quota and208713/1000000 individual inodes. More than enough
for <=512MiB newly written input scope; no protein/run/build sizing inference.
Fresh exact source staging/test-only and final live duplicate/headroom check
still precede ONE approved submission. No GPU job/window requested.

Push bd12e76a1 verified exact remotee396e97215158830bad3fbee265a20ebeeb2bbd7.
Git-only source staging b5440ad3d PASS in NEW0700root
`P/gits/psi4.disp0-input-cpu-e396e97215-01/source`, detached/clean.
Helper SHA256`58068fe77cdf55148f0370f8ae255bd25c4c416662f7b5e975c64d3d59cfdd64`;
batch`0d46aef917ea16410e0638e125fe580d46572504956e9dfc167d207c5396fc30`;
payload`652d9142604b820ce72dbba3609ab35a456a5eba0cc4437a91d3011af43165ae`.
Exact test-only PASS cpu-small/1CPU; simulated14215760 is NOT an accepted job.
Single approved submission task b4eb8ff87 now launched, with final live
headroom/duplicate checks, byte/inode checks, cleared ambientSBATCH overrides,
fresh exact-route test-only and irreversible own-root submission-attempt
guard before one sbatch--parsable/--no-requeue call. Accepted receipt pending;
no automatic retry even on ambiguous scheduler/network failure.

**ACCEPTED input-only job14215786**, receipt b4eb8ff87, submitted
2026-10-20 08:28:57 Phoenix local(Eastern). Effective scontrol:
`gts-cs207-chemx /cpu-small /embers`,1node/1task/1CPU/2G/00:10:00,
Requeue0/no GPU; initially PENDING(Priority), no node/start assigned.
Final headroom21/50 andduplicates0; byte/inode/quota checks PASS.
Final test-only simulated14215785 is NOT a second real job.
The ONE fresh input-only CPU approval is consumed. No retry, no builds/
device work; conditional8CPUCPU89 approval remains unconsumed and NOT GO.

Source remains exact frozene396e97215158830bad3fbee265a20ebeeb2bbd7 in
`P/gits/psi4.disp0-input-cpu-e396e97215-01/source`.
Logs`P/runs/saptdft-cuest-disp0/logs/disp0-sdk-input-14215786.{out,err}`;
status`logs/14215786.input-status`; future run
`P/runs/saptdft-cuest-disp0/input-prep-e396e97215158830bad3fbee265a20ebeeb2bbd7-14215786`.
Official watcher bdebd125d uses300s state-change reporting, no manual polls.
Parent received accepted receipt/resource/source proof. Current test-only
forecast is next-day2026-10-21~10:41, not a guaranteed start; dated08:00
America/New_York objective is unmet, no speedup ETA promised. No GPU held/
requested or parent roots/allocations changed. Input manifest, actual remote
decoder proof, closure and parent INPUT-READINESS remain PENDING.

Watcher bdebd125d terminal report: input-only14215786 **FAILED1:0**
after00:00:03; `.batch`FAILED1:0/3s and `.0`FAILED1:0/2s. Parent notified.
No retry or new allocation. Bounded read-only log/artifact diagnosis
b6a4ed947 launched; precise blocker/manifest status pending inspection.
Conditional CPU89 remains unconsumed/NOT GO, no GPU readiness/step.

Read-only diagnosis b6a4ed947: actual0.19 decoder controls **PASS**
inside14215786 (8MiB frame,131072 rejected,134217728 decoded8192bytes).
Failure is specifically the developer-link immediate-target check; it does
NOT establish invalid packages. `term_received=0`; manifests/receipt ABSENT.
Nodeatl1-1-02-002-27-2; actual prolog/epilog confirm approved1CPU/2G/
cpu-small/embers/chemx. All failed roots/status/evidence preserved.

Parent cleared READ-ONLY LOCAL validation of BOTH exact public archives,
not another allocation/provision/retry. Both official hashes/sizes/indexes
were validated entirely in memory. Exact topology:
`lib/libcusparse.so -> ../targets/x86_64-linux/lib/libcusparse.so.12.8.2.51`,
and targetlib/libcusparse.so points directly to that same versioned filename;
nvJitLink analogously targets`libnvJitLink.so.13.3.33`. Full filenames differ
from their genuine ELF SONAME aliases. No header file was extracted.

Actual header SHAs from both approved archives:
cusparse.h`76c4ca9c08a101291c987776a42c03fa436202ac9fe970a02280452c8d1ee222`,
cusparse_v2.h`8e41f603d85873e4c4174bee43f4aeadb84f3441e460650846ec4a5e1140aa7c`,
nvJitLink.h`d1f43e48b34cb8bdc2ac32c192e60b1fec882c09c5dae691cb876aae4eafb48e`.
New LOCAL audit entrypoint`audit_sdk_archives.py` enforces both archives
and those header identities, emits virtual link graphs and approved external
binding expectations; never follows archive links as host paths.

Parent provided independently verified COMPONENT-FILE bindings from14185006
manifest`P/runs/saptdft-cuest-cpks/major2-whole-gpu-scaling-20261019-v3/manifest.json`,
SHA`3030cecda5562f1716a8df262f4f05386427ed4aab70cbda7d65000dc25a1ad1`.
At canonicalvendor targetlib: cusparse12.8.2.51 SHA
`cadbe1c78feae59b7448dde354bc3c1a60c3da32e0f5902780842f14d475d405`,
SONAME`libcusparse.so.12`; nvJitLink13.3.33 SHA
`f84abd240b428825c3ef16632deb1026b208252fe19cef225a9f4ad06a39d1aa`,
SONAME`libnvJitLink.so.13`. Those are component expectations only, NOT
SDK/E2/device readiness. Actual remote files were NOT rehashed locally.
Before any future use, the corrected helper separately requires actual regular
files at that exact approved canonical root, SHA and ELF SONAME match.

Virtual resolver allows<=16 links, versioned-filename and legitimate
multi-hop aliases; cycle/escape/broken/regular-intermediate/shadow/too-deep
negatives plus actual wronghash/wrongSONAME/filename/symlink binding tests
are running locally (ba505a815). First BOTH real-archive regression exit0;
durable tracked audit b4f162ba8 and narrow read-only review dea89bcb pending.
No new allocation/provisioning/step authorized or executed. Return reviewed
immutable pin, BOTH real-archive proof and narrow proposal before any new
resource discussion; CPU89 remains conditional/NOT GO.

Virtual-chain/component-binding guard task ba505a815: **35passed1.45s**,
no warnings/skips, JUnit
`build_disp0_cpu/reference-check-pgmJLAWQ/results.xml`.
Actual SHA checks use local fixture bytes; ELF is explicitly mocked in unit
tests. They do not reverify remote component files. AST/whitespace PASS;
approved89 runtime_contract/payload and all scientificpsi4/ sources unchanged.

Review dea89bcb found an explicit symlink/non-directory ancestor could shadow
the virtual terminal namespace even though no host path was followed.
Corrected resolver now rejects non-directory ancestors of EVERY traversed
path/terminal; added ancestor-link/ancestor-file negatives and explicit-directory
positive. Also added post-inventory runtime SHA/path/SONAME equality to the
approved preflight bindings so changed external bytes cannot enter a manifest.
Real-archive evidence remains separate from synthetic/unit coverage:
new durable BOTH actual archive audit b065224c4 runs the corrected helper;
guard suite b4fdb7478 and narrow follow-up d81d3e45 pending. No attempt/job
was retried and no new resource approval is inferred.

Corrected BOTH real-archive audit b065224c4 **PASS exit0**:
two original exact-SHA archives/82569bytes, real local zstd decoder, expected
header inventory/SHAs, both virtual devlink graphs and approved component
expectations. No filesystem extraction, remote component read, provisioning
or allocation. Helper SHA256
`c8c1b65fc9764212b524fdcc2479d171b9ec4cd3a4950be7d7b3390392b99d20`;
durable stdout SHA256
`7cb5bfe63e9ae3fa1e7688d0fdcfff7f0bc43e0bcf1e0f847b68681c768af71b`.
Command: local`p4_einv2/bin/python -B devtools/disp0/audit_sdk_archives.py`.
Evidence`.pi/tasks/01a154d8-a459-768d-83bd-c57c6a5dc42c-4065461/b065224c4.output`.
This real-archive proof is explicitly separate from unit fixtures; it does
not replace actual remote file/ELF revalidation, completeSDKmanifest,
parent INPUT-READINESS, CPU89 build or GPU numerical gate.

Final namespace guard suite b4fdb7478: **38passed1.49s**, no warnings/skips,
JUnit`build_disp0_cpu/reference-check-FR4phSpz/results.xml`. BOTH real
archive proof helper SHA matches current source byte-for-byte. AST/whitespace
and unchanged89 scientific/runtime/payload checks PASS. Follow-up review
d81d3e45 ended`child_turn_limit` with NO committed answer; not a pass and not
a numerical/code-test failure. Its failed diagnostic artifacts are preserved.
One narrowly scoped read-only final review d5d1f4cc is pending. No new HPC
attempt, allocation, provisioning or remote source/build/root mutation is authorized.

Final scoped review d5d1f4cc22b1b1c1c0789059e65e6aa1b completed: **no concrete
blocker**, prior ancestor-shadow and real-proof-separation findings addressed.
It read the three code/test files; supplied test/audit evidence was not
independently rerun by reviewer. This is source review, not runtime clearance.

Local milestone ready to freeze/push:38guardsPASS, BOTH original pinned
archive real-decoder auditsPASS, exact parent-approved component-file
expectations bound virtually, actual external regular-file/SHA/ELF checks
required separately before future use, post-inventory identity equality.
Changed scope: input helper, new local archive audit entrypoint, archive/
binding unit tests, private-input documentation and this progress record.
No scientificpsi4/ source, E2d17 source, approved89 runtime/payload, old
failed root or coordinator data/source/build/environment changed.

Narrow proposed next step (NOT APPROVED/SUBMITTED): exactlyONE INPUT-ONLY
1node/1task/1CPU/2G/10min freeembers/gts-cs207-chemx job, cpu-small subject
fresh exact-route; noGPU/no-requeue/no retry,110s/1GiB payload/128MiBdecoder,
same82569B TWOarchives/<=512MiBwritten. Fresh source/run/input/scratch and
source/headroom/quota/duplicates checks; actual0.19 proof + real approved
component-file SHA/ELF before use, complete privateSDKmanifest/closure/lock
proof only. No solve/native/scientific compilation/device/SCF/SAPT.
Request explicit fresh input-only approval after returning immutable pin
and this proof. CPU89 remains conditional/unconsumed/NO INPUT-READINESS;
compiler/copied-include discovery still unproven. No independent GPU request.

Parent granted fresh EXACTLY ONE INPUT-ONLY CPU job pinned
`3bb0c7d1d2ce21412d883ff9041c96900ef16277`:1node/1task/1CPU/2G/10min,
freeembers/chemx/cpu-small subject fresh route/headroom/quota/duplicates,
no-requeue/noGPU/no retry, NEW roots;110s/1GiB payload,128MiB decoder,
same82569B two archives/<=512MiBwritten. Actual remote decoder and both
canonical regular-file SHA/ELF checks BEFORE use; post-inventory equality,
package/header/devlink and original component-manifest provenance retained.
Only complete SDKmanifest/closure/exactlock, no solve/build/SCF/device.

Approval is currently unconsumed: fresh capacity/quota b1400e06d and exact
Git source staging/test-only b7a9b27f7 running independently. Planned NEW
0700root`P/gits/psi4.disp0-input-cpu-3bb0c7d1d2-01/source`. No submission
yet. Preserve failed14215786 and all earlier roots/manifestsABSENT. This
does not clear or consume CPU89; explicit parent INPUT-READINESS after
actual manifest remains mandatory. No GPU/shared allocation step requested.

Source/test-only preflight b7a9b27f7 FAILED transport timeout124 after75s
onphoenix, with NO remote command output. No real sbatch submission exists
in that preflight script (Git prep/test-only only); approved job remains
unsubmitted/approvalunconsumed. Do not infer root absence from silence or
blindly repeat mutations. Bounded READ-ONLY diagnosis b5b03a6cc uses configured
alternatephoenix2 to inspect planned root and live matching jobs; pending.
Original quota snapshot task b1400e06d terminal result still pending.
Parent notified; timeout evidence preserved, no provisioning/device claim.

Capacity task b1400e06d also timed out100s/no remote output, with a producer
warning that processgroup920568 might remain after termination. Targeted
localps inspection found ZERO processes in that exact group; no signals or
shared SSH mux changes were made. Fresh headroom/quota proof UNAVAILABLE,
not passed. Parent notified. Alternate read-only diagnosis remains pending;
no actual job submission/provisioning or approval consumption.

Alternatephoenix2 READ-ONLY diagnosis b5b03a6cc also FAILED timeout124:
phx40s/no output, background task50s. Both configured aliases unresponsive.
STOPPED remote attempts after the bounded configured fallback, no retry loop/
timeout stretching/shared SSH mux reset/cluster job modification. Planned
source-root presence remains UNKNOWN; do not infer absence or reuse a
possibly partial root. Fresh source/route/headroom/quota/no-duplicate proof
remains UNPASSED. Parent notified. Exactly-one approved3bb input-only job
is UNCONSUMED/UNSUBMITTED: no real sbatch command was issued, no provisioning/
device/shared step. Current local recipe/38guards/BOTH real-archive proof
remain unchanged; preserve failed14215786 and all timeout evidence.
Resume only after remote access recovers and fresh bounded state/preflight
proofs pass, still pinned3bb regardless of later progress-only commits.

Owner later reported shared14214205 stateUNKNOWN after its bounded status
check timed out; past queued receipt is not grant/failure/release evidence.
All GPU windows HOLD; no probing/cancellation/duplicate/mux reset on its
behalf. Our input pause remained unchanged.

User then reported connectivity-only defaultphoenix read-only login PASS
2.28s at2026-10-20 09:26:02Eastern. This proves neither job/SDK/GPU state nor
source-root presence. FDDS owner reconciling its actual allocation separately.
To avoid simultaneous probes, requested one bounded INPUT-only fresh
source/root/quota/headroom/route/duplicate preflight window after owner
reconciliation. No remote command/submission run from this report alone.
3bb input-job approval remains unconsumed/unsubmitted; GPU HOLD/CPU89noGO.

Owner granted a single READ-ONLY CPU metadata turn after its healthy snapshot.
Taskbe3b362fd captured2026-10-20 09:29:55Eastern onlogin-phoenix-gnr-3:
own planned3bb-01root EXISTS0700/uid3170273 but is NOT a complete Git checkout;
exact-pin/pristine source gateFAILED. No matching own input job, embers21/50;
project6273.3/8192GB andscratch3734.1/15360GB/209068of1Minodes. Current-login-node
own-root process references NONE, not a global quiescence claim. Resource-only
test-only completion marker exists but routing stderr was not retained, so
exact pinned-script route proof is NOT claimed.

An intervening HOLD superseded the earlier NOW signal; stopped only our local
read-only task, no Slurm job/controller/session changes. The captured output
already contains READONLY_CPU_INPUT_TURN_COMPLETE_NO_SUBMISSION; did not
duplicate probes on the renewed NOW signal. Preserved source partialroot,
no writes/provisioning/realsbatch or approval consumption. Snapshot stdout
SHA`90613b4132d0b218d79c36927e121d2e558f7216b180ca8a5e3be93cd4b07ac7`.
Owner and parent received STOPPED/source-gate failure.

Owner acknowledges turn complete; need for separately authorized bounded
Git-only staging into NEW root relayed to parent. Prior metadata NOW is NOT
write/submit approval. Remain offline/no re-probes pending that authorization.
Shared14214205 last verified PENDING/manifestpending, no GPU window held for
disp0. Exact-one3bb inputCPU approval unchanged/unconsumed; CPU89/GPUnoGO.

Parent separately authorized ONE Git-only staging into NEWabsent-02 root,
<=110s configuredSSH/Git, no retry/SDK/build/sbatch/test-only/GPU query;
owner assigned CPU_GIT_STAGE=DISP0_GIT_ONLY_02 NOW as the SAME one turn.
Executed exactlyONCE, taskb30423840 **exit0/elapsed1s**,
2026-10-20 09:35:50Eastern onlogin-phoenix-gnr-3.
NEW private0700 uid3170273 source:
`P/gits/psi4.disp0-input-cpu-3bb0c7d1d2-02/source`.
ExactHEAD`3bb0c7d1d2ce21412d883ff9041c96900ef16277`, pristine worktree/index,
helperSHAc8c1b65fc9764212b524fdcc2479d171b9ec4cd3a4950be7d7b3390392b99d20,
batch0d46aef917ea16410e0638e125fe580d46572504956e9dfc167d207c5396fc30,
payload652d9142604b820ce72dbba3609ab35a456a5eba0cc4437a91d3011af43165ae.
Gitstdout/stderr retained in
`.pi/tasks/01a154d8-a459-768d-83bd-c57c6a5dc42c-4065461/b30423840.output`.
Oldpartial-01 untouched/notadopted/reset/copied. No SDK/env/provision/
install/build/sbatch/test-only/GPUquery/lockborrow or duplicate turn.

Reported TURNCOMPLETE toowner+parent; owner acknowledged turn CLOSED.
STOPPED, no additional probe/submission/GPU window authorized by that ack.
Exactly-one approved3bb inputCPU job remains UNCONSUMED/UNSUBMITTED;
source proof nowPASS, remaining fresh exact-route(stderr retained)/quota/
headroom/duplicate gates still needed before the existing CPU approval can
execute. CPU89 build and all GPU readiness remain NO GO.

Parent/owner assigned SEPARATE one final CPU preflight+conditional-submit
turn. Used ONE bounded configured SSH session exec77157, no parallel probe.
Fresh09:42:19Eastern source/sha/path/pristine proofPASS at3bb/-02;
own duplicates0, embers19/50; byte/inode df gatesPASS, quota raw
project6273.7/8192GB andscratch3660.7/15360GB/208990of1Minodes.
Exact pinned batch test-only routePASScpu-small, bothstdout+stderr retained.
Simulated14217020 is NOT a real accepted job.

Permanent own attempt directory:
`P/gits/psi4.disp0-input-cpu-3bb0c7d1d2-02/final-cpu-attempt`.
Raw preflight/source/route/quota/df/duplicate evidence stored there. Session
was interactive so quota/route could be reviewed before actual submission.
Its hard100s turn clock expired before submission-phase acknowledgement.
This is a turn-wall-budget failure, NOT evidence of a new cluster outage.
The subsequent tool write of the actual-submit/durable-receipt phase returned
exit124 immediately, with no echoed phase/raw sbatch response/durable ID/
controller output observed. Input delivery/actual sbatch invocation is
UNKNOWN; do NOT infer accepted or unsubmitted. Intended phase permanently
marks invocation, retains raw response, fsyncs receipt+jobID BEFORE any
controller query, but its execution is not observed/proved.

Reported TURNFAILED/TIMEOUT toowner+parent and STOPPED/NEVERREPLAY. Existing
exactly-one CPU approval RESERVED pending proof, not reusable automatically.
No actual accepted job ID known; CPU89/GPUnoGO unchanged.
Owner separately assigned ONE READ-ONLY receipt-reconciliation turn; task
ba881acc1 inspects existing marker/raw receipt files and ONLY own
disp0-sdk-input queue/accounting (no sourcewrites/sbatch/test-only/SDK/GPU).
Empty/failed/delayed accounting is not proof of absence. Result pending;
no other query or submit loop authorized.

DISP0_CPU_RECEIPT_RECONCILE_01=ba881acc1 **TURNCOMPLETE exit0**,
2026-10-20T13:46:40Z, one noninteractive bounded READ-ONLY SSH command.
Attempt directory exists; source-pin.txt exactly3bb (41bytes, SHA
63fbad7d5c603f9169136ccf86c8da2442285e63dc4abee657cef8f3043ccad4).
ABSENT: SUBMISSION_INVOKED, preflight-review.txt, submission-response.json,
job-id.txt, sbatch.stdout/stderr, controller.stdout/stderr.
Retained test-only.stdout empty andtest-only.stderr125bytes/SHA
7db2ddd73d0f6c2885a1bbe517ad2fa6cd9f8e1fe51cd8bbd0c85d22378c4720;
simulated14217020 remains NOT an accepted ID.

OwnCPU squeue(name=disp0-sdk-input) exit0,empty stdout/empty stderr.
OwnCPU sacct(name filter/since09:40/WorkDir included) exit0,empty stdout/
empty stderr. Complete query arguments/exits/raw stdout+stderr preserved
`.pi/tasks/01a154d8-a459-768d-83bd-c57c6a5dc42c-4065461/ba881acc1.output`.
No accepted job ID/effective job route proved. Missing records/empty or
possibly delayed accounting are NOT conclusive absence proof: submission
delivery/execution remains UNCONFIRMED/UNKNOWN; exact-one CPU approval
RESERVED pending conclusive reconciliation, NEVERREPLAY. Reported exact
evidence and TURNCOMPLETE toowner+parent and STOPPED. No sourcewrites/
quota-refresh/sbatch/test-only/provision/build/GPU or ownerallocation query/
signals/jobchanges in this reconciliation. Original records untouched.
CPU89/GPUnoGO unchanged.

Parent authorized LOCAL original script/session/tool chronology analysis ONLY;
all remote actions remainSTOPPED, UNKNOWN/approvalRESERVED cannot be relaxed.
No original command executed/replayed. Evidence report:
`devtools/disp0/local_submission_chronology.md`.

Original session JSONL lines4185–4198 show exec-call journal13:41:56.699Z;
second write request13:43:47.321Z,110.622s later/10.622s after nominal100s
deadline. Second native function-call create_time13:43:47.146772Z is metadata,
not a transport ACK. Tool result13:43:47.323Z reports exit124 in0.000128128s,
only timeout text/no phase2 echo/receipt. Requested2618UTF8bytes SHA
33ed577ce46b2034979c2533a77fde79427fbef92572336a8a956e2f48dee4bf;
no byte-write counter/PTY/SSH delivery ACK appears in inspected result
details. Earlier assistant message timestamp is NOT delivery evidence.

Original phase2 requires review file(line1), invocation-directory(line2)
before actual sbatch(line4); response/jobID fsync precedes controller(line26).
However marker/review writes lack explicit pre-RPC fsync. Missing markers
therefore are supporting evidence, not a certified durable negative proof.
Chronology is consistent with a late call to an exited session, but does not
independently certify zero delivered bytes. No predeadline phase2 dispatch
is recorded. UNKNOWN/RESERVED/NEVERREPLAY retained; no remote queries,
additional turns, resource approval or submission inferred.

## 2026-10-21 — user requests optimized-branch build

User: **start the build**. Parent confirms proceed with one isolated CPU-only
build pinned to fork06404fa69ce9346443d7d4f19a250f6db9f0f62c, Psi4 scientific
587e172a33 unchanged. FDDS owner reports no active owner remote operations,
terminal14214205/no holder and clears a bounded CPU-preparation SSH turn.
This supersedes the general remote preparation hold for this NEW build only;
old input submission UNKNOWN/approvalRESERVED/NEVERREPLAY stays unchanged.
No GPU request/step or paid allocation.

Local comparison-agent consultation/source audit is recorded in
`devtools/disp0/optimized_einsums2_branch_review.md` (commit6926a49660).
The optimized target is fork`perf/fdisp0-hotpaths`064, not stockd17 nor
the misleading local branch alias10be. All supplied prior comparison evidence
is CPU-only; local064-like binary dirty, remote clean064 claim not remotely
reverified. No old binary/driver overlay is used for the new build.

Fresh executable recipe: `devtools/disp0/optimized_build/`.
Resources: ONE1node/1task/8CPU/48GiB/2h/freeembers/gts-cs207-chemx/noGPU/
no-requeue/no retry, maximum16CPUh. Complete private CUDA13 inputs are
prepared within the build job with existing exact-archive/runtime binding
guards, before any configuration. Original explicit runtime lock unchanged;
separate fresh build-only LLVM/Clang23.1.2 environment, explicit solved lock
retained; sourceApiary1.2 as required by exact064. Whole native+Python
installation uses upstream064's own rules with deliberate installed RUNPATH;
no d17 package-copy hook. New source/env/build/install/cache/scratch roots,
all old failures/prefixes/parent roots untouched.

Submission is a new self-contained one-shot noninteractive command: permanent
intent+file/directory fsync, fresh CPU route/stdout+stderr/quota/headroom/
duplicates, dispatch-intent fsync BEFORE sole sbatch, raw reply+real durable
jobID BEFORE controller query; reentry/reply-loss never permits replay.
13 local mock/metadata guards PASS1.203s; both Bash syntax checks PASS.
Complete-input guard rerun and narrow read-only build reviews pending.
No remote command, real submission or accepted job ID yet at this entry.
CPU build/import/ELF/API proof remains distinct from signed physical/device
correctness; historical26/64 is not a validation of this optimized stage.

Narrow reviews d90202093/d3bf5e8c1 retrieved. Whole Waggle native+Python
companion and generated clean064 GitInfo are now required. Resource inheritance
blocker fixed: removeSBATCH_* from bothscheduler-call environments and pass
all authorizedresources explicitly on bothCLIcalls. Exact064 auto-ccache
disabled withEINSUMS_WITH_CCACHE=OFF. Final21localguards PASS2.694s plus
Bashsyntax; 38existingSDKinputguards PASS0.27s with existingpytest-capable
interpreter. InitialsystemPython rerun failedbeforecollection(No pytest),
not an inputguardnumericalfailure; no install. Scientific psi4/ diff versus
587 empty. Narrow review-fix follow-up pending; execution not yet begun.

Follow-up d8335ebad confirms review fixes, **no scoped blocker** (read-only,
not a build/test). Freezing/publishing the executable optimized recipe and
proceeding to the owner-cleared bounded Git/source/quota preparation. No
actual accepted CPUjob yet; no GPUjob/request. ExistinginputUNKNOWN untouched.

Pre-push tracking check caught repository ignore rules `build*`, `*.sh`,
`install*` excluding the new batch/payload/hook from initial localc2c2611.
No remote/push/job used that incomplete pin. Explicitly force-adding the three
reviewed configuration scripts and adding a Git-tracking regression before
publishing the final complete pin. Content/reviewed scientific scope unchanged.

Complete executable source pin **d097512fe4b01b55397b0df0075a474121ca852a**
published and exact GitHub ref verified (bfbbe6ff7exit0). Final22local
guards PASS2.291s including executable-files-tracked regression.

CPU_PREPARATION turn **b392ad60a TURNCOMPLETE exit0**, no submission:
fresh root `P/gits/psi4.disp0-opt064-20261021`, mode0700/uid3170273;
source exactd097, pristine and scientificpsi4/ equal587. Both commits fetched
via Git only; oldpartial/failed/UNKNOWN roots untouched. Batch/payload/hook/
submit SHA256s retained in task output. Rawquota timestamp
**2026-10-21T10:08:53Z** /1792577333, pace-quotaexit0/stderrempty:
project6316.3/8192GB, unlimitedgroupfiles(0limit);
scratch4936.0/15360GB and212159/1000000files. Projectphysicalavailable
1966757888KiB and37280898762inodes; scratch1234121621144KiB and1325503256
inodes. Conservative human-reviewed admission1600GiBproject+1Mfiles,
9000GiBscratch+700kfiles; rawstdoutSHA
1e427d596b3fcbac1dc09db821bbdee463274a823aafbf8516ed3902698c9134.
This is manual interpretation of retained output, not parsing pace-quota's
unstable display. >=64GiB/50k threshold comfortably satisfied.

Next self-contained bounded submission **b9fd200c8** launched under fresh
user/parent build authorization: source/quota/route(stderrretained)/live
headroom/duplicate checks, durable newattempt+intent/dispatchmarker before
sole actualsbatch, rawresponse+realID beforecontroller. No interactive stdin
secondphase/model-review pause. New attempt:
`P/gits/psi4.disp0-opt064-20261021/submission-attempt`.
Actual acceptance/ID pending at this entry, NEVER infer/replay on ambiguity.
No GPUquery/allocation/step, no login solve/install/compile, no oldinputreplay.

**ACCEPTED CPU build14249189** — b9fd200c8exit0,
SubmitTime2026-10-21T06:10:25America/New_York.
Actualcontrollerexit0/stderrempty proves:
`Account=gts-cs207-chemx QOS=embers Partition=cpu-small Requeue=0
NumCPUs=8 NumTasks=1 ReqTRES=cpu=8,mem=48G,node=1 TimeLimit=02:00:00`.
NoGPUrequest; initialstatePENDING/ReasonPriority, StartTimeUnknown/noETA.
ExactlyONEactualsubmission. AcceptedID printed only after durable real-ID
file; rawresponse retained beforecontroller. ReviewquotaproofSHA
e7a37662ad1aceee7d4114765691dbb6e52ef1c1cbce8e09f5412ba76357f28e.
Exactrecipe d097512fe4/fork064/science587. No oldinputattempt replay/change.
Bothowner+parent received actualreceipt/effectiveroute/TURNCOMPLETE.

Outputs: `P/runs/saptdft-cuest-disp0/logs/disp0-opt064-14249189.{out,err}`,
status`14249189.opt064-status`, intended fresh run
`P/runs/saptdft-cuest-disp0/opt064-d097512fe4b01b55397b0df0075a474121ca852a-14249189`,
job-owned persistent scratch
`/storage/scratch1/3/awallace43/disp0-opt064-14249189`.
Event monitor **bc8981c53** uses installedwatch_jobs.sh -i300, including
batch accounting; no submit/retry/GPU/jobchange. Build/API/signedphysics/device
correctness remain unpassed pending actualartifacts and cleanbatch exit.

**14249189 FAILED1:0**, including.batch/.0, elapsed48s (start06:11:01,
epilog06:11:49Eastern), signalled0. Eventmonitorbc8981c53exit0 at10:16:34Z
reports negativeaccounting; not success. Bounded ownartifactinspection
b2fcb375cexit0 confirms actualprologchemx/cpu-small/embers, status exit1/
signalled0 and exactfresh run above.

Failure BEFOREconfiguration/compilation/environmentcreation:
`provision_sdk_inputs.provision -> closure -> runtime_contract.command`
`/usr/bin/ldd` on canonical
`P/miniconda/envs/p4cuest_sapt/targets/x86_64-linux/lib/libcusolver.so.12.2.6.9`
returned1. `check_output` exception did not retain the failedloader's stdout
in the log, so underlyingloaderdiagnostic is notproved. Do not classify the
hash-pinned headers/runtime invalid, or this as GPU/formula/numericalfailure.
Actualremotezstd0.19streamingwindownegative/positive proofPASS beforepackage
extraction. ABSENT: private-inputs/sdk-manifest.json, provision-receipt.json,
cuda13/sdk-contract.json, bothenvexplicitlocks, configure/buildlogs,
installedproof/APIresult. No SDK/build/runtime/device admission.

Oneadditional bounded **READ-ONLY ELF program-header metadata** check
b7ac30031 inspects PT_LOAD virtualspan versus inheritedprovisioner1GiB
RLIMIT_AS (a hypothesis, not yet confirmed). No ldd rerun/libraryload/CUDA
call/provision/source-envmutation/newallocation. No automaticretry. Preserve
failedjob/run/privateinputpartial/scratch and oldUNKNOWN attempt unchanged.

ELFmetadata b7ac30031exit0: cuSolverfile167967544B, twoPT_LOAD segments,
ownminimalmappedspan170224600B (<1GiB). This does NOT by itself establish
the aggregateclosurebudget or explain the failure. Therefore one bounded
load/resolve-only diagnostic **ba3e8f69a** captured stdout+stderr under the
SAME1GiBAS/10sCPU cap,15s timeout; no constructors/device/scientificwork,
no provisioning/newallocation/fallback. Actualexit1/stderrempty/stdout:
`libcusolver.so.12.2.6.9: error while loading shared libraries:
libcublasLt.so.13: failed to map segment from shared object`.
The failing check is inability to map a transitiveCUDA dependency under the
inputprocessaddress-spacebudget, not a missingheader/SONAME/numericalerror.
No unconstrainedtrace or highercap trial was run; completeSDK/runtime
correctness remains unproved.

Narrow correction proposal: preserve1GiB/110s/128MiBwindow/512MiBwrite
archive-decoder limits unchanged; perform actual captured load/resolve
closure audit in a separate explicitly bounded CPU-preparation process,
not inheriting the decoder'sAShardlimit. Keep both phases mandatory and
write finaladmission ONLY after both pass; preserve allfailurestdout/stderr.
No scientificE2/Psi4 source/flag/tolerance changes. Tests/review/freshpin
required; no actualrepair/build/retry/allocation authorized or performed
by this diagnostic. Next resourceproposal remains ONE8CPU48G2h/freeembers/
chemx CPUbuild, subject fresh approval/gates; do not replay14249189 or
the oldUNKNOWN inputattempt. Allfailedroots unchanged.

User **yes** grants ONE corrected same8CPU48G2hembers/chemx build. User then
explicitly selects **one interactiveCPU allocation for build/debug rather
than sbatching every attempt**, then GPU tests only after readiness. This
REPLACES unsubmittedcorrectedbatch, not an extraallocation. No new allocation
yet. Parent+FDDS informed; FDDS freshH200ownerpreflight was given precedence:
ourworkLOCAL/noCPU-preparationSSHturn active. No GPUaction/window/borrow.

Corrected SDK code preserves originaldefaultCLI/closurechecks and decoder
1GiBAS/110sCPU+wall/onecore/128MiBwindow. New optimized entrypoint forks one
8GiBvirtual-AS CPU loadertrace worker BEFORE decoderlimit; max6canonicalCUDA
component/SONAME-major-bound requests,10sCPU/10stimeout each,70sCPU/110swall
worker, rawstdout+stderr durablebeforereply. No retry/respawn/constructors/
device/scientificwork. Both finalSDKmanifest/receipt remain AFTERcomplete
mandatoryclosure+trace evidence. Audit1MiB reservedfrom512MiBwritecap.
Oldhelperwithoutnewentrypoint has no admission bypass/versionoverride.

Local redvirtualmapping regression failed as intended beforefix
(1536MiBmmap/1GiBAS,errno12,0.231s), then passed with separatelybounded
loadercapacity; no pages touched/CUDAlibraryloaded.39SDKguards PASS0.28s
(newnegative: closurefailure producesneithermanifest),33combinedmock/metadata/
configureguards PASS1.537s beforeadditionalIPCproof. BashsyntaxPASS.
Oneintermediatetest-only FileNotFound-vsRuntimeError normalization corrected,
not a numericalfailure. Two narrowread-only reviews active: SDKbudget seam
d01c2d55c and interactiveprotocol d5c46a79b. No remote/currentjob yet.

Actual localfork/IPC regression additionally observes decoder1GiBhardAS and
worker8GiBhardAS after workerstartedBEFOREdecoderlimit. Workerrejects fake
non-lddrequest, rawresponsecaptured; no libraryloaded. Combined34PASS4.070s.
FDDSownerpreflight b8b80efdb0 nowTURNCOMPLETE/noactiveownerremoteoperation;
weacknowledgedstillLOCAL and willnotifybeforeownboundedCPUprep.

SDKreview d01c2d55c found oneblocker: timeout/oversize genericexceptionhandler
discardedpartialloaderstreams. Fixed boundedstdout/stderr+explicitbytecounts/
truncation/timeoutstatus/fail-closedreturn, fsyncedrawbeforeack. Added actual
forkworker tests for timeoutpartial/oversize paths.36combinedguards
PASS1.669s and39SDKguardsPASS0.29s. Narrowconfirmationdd7ba9065 pending;
interactivecontrollerreviewd5c46a79b pending. No currentremoteCPUoperation.
FDDS Gitstageba02d3335complete and ownerfinaladmission nowinprogress;
ourCPUloginprep waits its shortadmission window, not its longGPUqueue client.
No GPUwindow/borrow/jobactions onownerbehalf.

FDDS14250361actual4hH200grant relayed; its exclusivegates active, but no
CPU-preparationSSHturn held and ownCPUprepcleared. No holderborrow/GPUprobe.
Independentfollowups d5c46a79b anddd7ba9065 hit artificialturnlimit and gave
NOcommittedanswer, NOT counted asreviewPASS. KnownSDKblockerfixed+36/39
testsPASS; directsourceinspectioncomplete. Freezingcurrentcode for Git
staging/freshCPUpreflight; advisorydf014bf6e reviewsfrozenSDK/allocator
withadequatebudget while independentpreparation proceeds. No allocationyet.

Frozen executable correction pin **6d1a0e9f138e250bbc832046f1b6c802e731288e**.
Narrow allocator review **de16fa371** retrieved: **no scopedblocker**, given
testedpreflightguarantees; oneRPC/durableintent+raw+ID/controllerCPUresource/
Requeue0/Restarts0 guards confirmed. Livecontroller acceptance stillunproved.
Publication b524b344e pendingterminalreceipt atthisentry; no allocationyet.

Publication6d b524b344ePASS/exactGitHubref. CPUstage02b61b89406 active.
Finaladvisorydf014bf6e retrieved: SDKtimeout/outputevidencefix and budget
separation have **noadditionalblocker**. Oneallocatorblocker: absentReqTRES
or GPU onlyinAllocTRES couldpass. TightenedcontrollerrequiresBOTHpresent/
parseable CPU8/node1/memexact48GiB, rejectsGPUinboth, NumNodes1 required;
memoryunitnormalization and missing/overalloc/GPU-only-allocated negatives.
38combinedguards PASS2.012s. Scopedfixconfirmationd428f3ea1 pending.
6dcontroller has NOTrequestedallocation; preserve staged02 and freezefixed
controllerintoNEW03/source (no reset/adoption). Quotaevidencefrom02 may
be reused whilefresh/hash-bound for SAMEproject+scratch filesystems.
No science/resource-budget change or extraallocation.

Final source pin **10ea8add2ab049d293d0ea088d9e2c282c752f82**.
Reviewd428f3ea1 confirms requested/allocatedTRESblockerresolved, **no scoped
blocker**; no commands/tests executedbyreviewer. Actuallocal38PASS2.012s,
SDK39PASS0.29s remain executed evidence. Pushb8c897f0b pendingreceipt.

Stage02 **b61b89406 TURNCOMPLETE exit0**: exact6d, clean0700/uid3170273,
science587unchanged; noallocation. ActualSlurm26.05.4; sallochelp records
--immediate[=secs] and no--no-requeue, rawhelpSHA
055a615afc836c1f783f84c180eca82608d1fc846d25b6b213998f27b2a2a1c3.
Freshquota2026-10-21T11:35:38Z/1792582538:
project6317.6/8192GB, fileslimit0(unlimited);
scratch4876.0/15360GB,212542/1Mfiles. Projectphysicalavailable1965267968KiB/
37280862699inodes; scratch1234220843848KiB/1325641389inodes.
RawquotaSHA b1e7421dd01537fa7a361c268e241cab9f5837dc3070a6a67038df01d53eb766,
stderrempty. Humanconservativequota1600GiBproject+1Mfiles and9000GiBscratch+
700kfiles comfortably >64GiB/50k thresholds. No display-format parsing.
NextNEW03Gitstage preserves02; samefreshhash-bound quota canbe used if
<600s old atfinalpreflight. Freshqueue/headroom/routechecks stillinside
durableone-shotacquisition. Actualallocationnotrequestedyet atthisentry.

10eapublicationb8c897f0bPASS/exactremoteSHA. FinalNEW03 stage
**bccf77c14TURNCOMPLETE exit0**: exact10ea/pristine/0700uid3170273,
science587equal. Noreset/adoptionof02. Freshquotatimestamp
**2026-10-21T11:44:20Z**/1792583060: project6318.1/8192GB;
scratch5054.8/15360GB and212658/1Mfiles; physbytes/inodesample.
RawstdoutSHA95e53faee50f40c3339a6b5147c93e3d007b545c3dba3fd6a9635c8d2d41e99d,
stderrempty. Sameconservative1600GiBproject/9000GiBscratch+1M/700kinodes
humanadmission; noquotaformatparser. ActualSALLOChelp/versionalready26.05.4.

**ONEinteractiveCPU acquisitionb40cff0bd launched**: freshsource/route/
headroom/duplicates/quota gates insidepermanent03/interactive-attempt,
then one `salloc --no-shell --immediate=600` with1node1task8CPU48G2h/
explicitfreeembers/gts-cs207-chemx/noGPU. Noextra correctedbatchjob,
no allocationretry. Self-containedconfiguredSSH PHX_TIMEOUT700/backend730,
not anopen stdin/model-review session. Everyaccepted/rawlinefsynced,
realIDdurableBEFOREcontroller; requireRUNNING+exactbothReqTRES/AllocTRES/
Requeue0/Restarts0 beforeCPUbuildGO. Request/grantreceipt pending atthisentry;
do not infer acceptedID/state. Finalreviewd428 scopedPASS. Owner+parent
notified CPUprepturnclosed/acquisitionactive; no GPUqueries/holderborrow.

**Actual14250580CPUgrant**, b40cff0bd: nodeatl1-1-02-006-8-2,
chemx/cpu-small/embers,8CPU48G2h,Req/AllocTRESexactnogpu,Restarts0,
BatchFlag0. AcceptedID/raw/controllerdurable. Controllerguardstopped on
site **Requeue1** default, so acquisitiontaskexit1 did NOT mean noallocation.
Allocationmanifest/CPUbuildGO was NOTwritten; no srunbuildstep launched.
JobStart07:46:19/EndTime09:46:21Eastern. User/coordinatorexplicitlyapproved
ONEold-IDflagupdateRequeue0; questionnaire resolved with that approval, no
newapprovalasked. Subsequent user says tryagain.

Atresume10:22Eastern, one correction **b3e625dbf** failed FIRST
`scontrol show job14250580`: exit1/Invalidjobid. Saved authorization+before
query; branch never reached update-intent/RPC. **No flagupdate performed**.
Neverreplaythisattempt. One boundedownaccounting **bed1afb83exit0** proves
terminal **PREEMPTED0:0**, elapsed02:00:11,07:46:19–09:46:30Eastern,
sameCPU node; queueempty. No currentCPUholder. No compilation/API/physical/
GPUvalidation occurred; leasewasheldunused whilepolicyguardblocked.
Oldsource03/attempt/correctionrecords/failed14249189/oldUNKNOWN unchanged.

Parent+ownerinformed exactactualterminalrecord, no update/newallocation.
FreshONEreplacementCPU8CPU48G2hembersdecision requested with bundled
permission ONEnew-own-IDRequeue0(ifsite1) and immediatebuildfollowup with
no morehumanquestion. **Replacementnotyetrequested/authorized atthisentry**.
Localcontrollerprep now has explicit opt-inflagcorrection requiring approval
reference, ownUID/jobname+completeTRES proof, onefsyncedupdate and actual
unchangedresource/start/end/timeproof beforeGO. Defaults remainstrict.
Newfreshnumberedsource namespace avoids codeeditingjusttoselect04; no
attemptreuse/adoption.42localguards PASS2.524s. No remoteactionfromtests.

Parent's explicit message, read in localworkmuxcapture, confirms fresh
replacementapproval **alreadygranted/valid; doNOTwaitanotherapproval**:
ONE8CPU48G2hembers/chemx, bundledONEnewownIDRequeue0(ifneeded), verify
unchangedactualresources/time, thenauto-startbuild. NoGPU/ambiguousretry.
FDDSowneragaincleared boundedCPU-onlyGit/quota turn; noownerprobeinflight
exceptitsown5minobserver; noGPUwindow/borrow. No morequestionnaire.

Reviewd132 found onlymissingStartTime/EndTime proof couldcompareNone.
Fixed mandatoryfullISOseconds/realcalendar/positiveinterval and direct
unchangedkeychecks. Addedmissing/invalid/backwards/changedstart/changedend
negatives;45localguards PASS4.772s. Scopedconfirmationd83e42a43 pending.
Actualresourcepermission nowVALID forONEfreshreplacement, notyetrequested.
Old14250580PREEMPTED/alloldroots/UNKNOWN remainunchanged; no old-IDretry.

Finalreplacementcodepin **770ecf90108aa7783da766bf0402d8c267dcbbed**.
Scopedtimestampfixreviewd83e42a43 confirmspriorblockerresolved/no scoped
blocker; read-onlyreview, notindependentlyexecutedtests. Actual45PASS4.772s.
Publicationbac804fe1 pendingreceipt. Nextfreshprivate04source/Git/quota
turn underexistingowner/parentclearance; nofurtherquestion, ONEreplacement
CPU8CPU48G2hemberschemx withpreapprovednew-own-IDflag0 incontroller, strict
postproofthenimmediatebuildfollowup. Noallocationrequest atthisentry.

Final04 staging **bb9e1ff67exit0**: exact770ecf9/pristine/0700uid3170273,
science587same. Quota2026-10-21T14:44:36Z/1792593876:
project6340.2/8192GB, scratch4835.5/15360GB/212096of1Mfiles,
bytes/inodesample. RawquotaSHA
32cacefa2b719ed69eb70a8b2d30e6891d7ab53f463f7f93051cd8c6eea08b9d,
stderrempty; humanconservative1600GiBproject/9000GiBscratch+1M/700kfiles.

**ACTUAL replacementCPU14254131GRANTED**, b0fb13320exit0.
Nodeatl1-1-02-005-10-2, Accountgts-cs207-chemx/QOSembers/Partitioncpu-small,
1node1task8CPU48G,TimeLimit2h,Start2026-10-21T10:46:00/
End12:46:01America/New_York. BothReqTRES+AllocTRES exactcpu8/mem48G/node1,
noGPU,BatchFlag0,Restarts0. PreapprovedONEnew-own-IDRequeue0correction
SUCCEEDED; strictpostcontroller verifiesflag0 andunchangedresources/account/
start/end/time. Noquestion/newallocationretry. RawacceptedID/flagintent/
response/before-after/controller/allocationmanifest durable in
`P/gits/psi4.disp0-opt064-20261021-04/interactive-attempt/`.
Quota-reviewSHAa6be658bf130c4a98406678e66b2aee88e52f06598cbbc858fca0909192a3feb.

**FirstactualCPUbuildclient launched immediately**, be4138c69,
exclusive8CPU48G SAMEallocation, source770/fork064/science587. Freshactual
controller/time budget checked; bounded<=6300s and remaininglease-minus90s.
No logincompilation: allbuild work inside exclusive srun invoking tracked
interactive_step.sh. No newallocation orGPUstep. Logs:
`.../interactive-attempt/build-step-01/{launch-intent.json,controller-before.json,
stdout,stderr,client-result.json}`. Actualnode-stepstart receiptcheck
bbd025732 pending; no build/API/numerical/devicePASS inferred yet.
Ownlease officialmonitor b53cf45bd -i300 and independentstepterminalevents
avoidqueryloops. Owner+parentreceivedactualgrant/correction/launch receipts.

Build-startreceipt **bbd025732exit0** proves actualnode srunprolog10:47:35,
CPU14254131/embers/Requeue0. RepairedSDKhelper **PASS49.297034s**,
23,420,568Bwritten; actualremotezstd0.19negative/positivewindowproofpassed,
mandatoryseparatelyboundedloaderclosure finished and helper cleanlyreturned.
SDKmanifestSHA **04c087d7d61535d6f0906bd871aff32024eb46914e4ba5e3b2af977e5d3f91d2**,
prefix`P/runs/saptdft-cuest-disp0/opt064-770ecf90108aa7783da766bf0402d8c267dcbbed-14254131-0/private-inputs`.
Stderrempty/client-resultabsent atthisone-timecheck; mainbuildcontinues,
notCPUAPI/physical/devicePASS. Inputsubmission_go=false correctlyremains
not an overallscientific/runtimeadmission. Parent+ownerreceivedthismilestone.

IndependentREAD-ONLYphase1readinessaudit **dd72e6f602exit0** completed while
buildruns; nocommands/mutations/GPUactions. Existinggpu_gate.sbatch/old
payloadsadmitd17/orfailedrecovery, notnew064opt064interactive step. Do NOT
submitstandaloneoldwrapper or relabeloldfailedbuild. Minimumfuturework:
separateoptimizedadmission withcompleteinstalledpackage/stage+binarytuple/
ELF/closure/actualprocessmaps, cleanjob.STEP/nonsignalledstatus(not.batch),
matching-stageAPIandindependentsignednative/eager/defaultCPUreferences
(26/64focus/tolerancesretained), CPU-preparedforwardlogger, thenowner-assigned
GPUwindowonly. PreserveexistingcuBLAS/naturalplacement/edgeCPUfallback/
strides/replay/pool/shadowfreshness and signedcomponentguards. Thisis a
proposalONLY; task21pendingblockedbybuild19. NotREADY/noPhase2expansion.

MainCPUstep **be4138c69exit1**, clientelapsed2596.564s,
node-status14254131-0 exit1/signalled0; originalfailure preserved.
Actualboundedinspection b67dad02dexit0 atRunTime45:17 shows SAMElease
14254131RUNNING/Requeue0/End12:46:01. Both E2064native/Python build+install
and Psi4whole-stageinstall completed before failure. Exactfailure is metadata
parser: `build_audit.safe_path` rejects literal Ninja-escaped `\\$ORIGIN/../`.
This is NOT a compilation/numerical/GPUerror. Runtime/API stillunexecuted.
Rawcompiler/installation logs, both explicitlocks, headerABI proof and
767372-byteprivateSDKcontract exist; no installedprefix repair performed.

Narrowparserfix recognizesanchoredexactescaped/unescapedORIGIN only;
unknownvars/lookalikes/foreignCUDAstillfail. Redexactrepro observed beforefix.
Externalnewartifact-dir option keepsfailedrun/partialaudit READ-ONLY;
defaultlegacybehavior preserved.48localguards PASS5.275s +96existingruntime
guards PASS4.60s; oneintermediatefixturefailedwrongcompilerallowedroot,
correctedfixtureonly. Newtrackedvalidate_install.sh performs ONLY same-node
read-onlyaudit/installedproof/API withbefore-afterbinaryhash equality,
newvalidation/scratch outputs, no configure/build/install/RPATHpatch/science
changes; oldfailedstep staysfailed. Scopedreviewdf9995ba8 active while
independentGit stagingproceeds. SAMECPUallocation, no newallocation/GPU.

Frozenvalidatorfix **45bc49eda03a556beb564ca68368685ab99ae6a2**.
Reviewdf9995ba8 confirms **no scopedblocker** in exactORIGIN/pathguard,
externalfreshoutputs andread-onlybinary-invariancecontinuation; no commands/
remotetest byreviewer. Publicationb637ded02 pendingreceipt. No recompile/
install/binarymodification performed; originalcompiledprefix andfailed
step0status preserved. Nextnewprivatevalidatorcheckout viaGit, sameCPUstep
with boundedremainingtime; runtime/API/signedphysics/GPUstillnotpassed.

Validationclient **bb8e709ccexit1**, elapsed52.8829s, actualstep1 status
exit1/signalled0. No installedproof/APIran: effective-auditstopped on
`Unexpanded compiler/link response file`. Originalstep0/build/install and
allpartialvalidation1outputs unchanged. Snapshotb63dfaa1a proves SAMElease
RUNNING/Requeue0 at11:56:51Eastern, End12:46:01; no newallocation.

BoundedREAD-ONLYmetadata b176b96ccexit0 identifies EXACToffendingrule:
`python .../apiary_aggregate_stubs.py @generated/pybind/.stubs.fragments
--overlay-class-regex ... && cmake -E touch ...`. Among2168top-level
Ninjaentries, oneunexpanded@entry is this auxiliaryPythonrule; compiler
responseexpansion is NOTweakened. Newparserseparatesshellsegments, checks
actualcompilerposition/approvedfrontend, validatescd, rejectsforeignflags/
caches/unknowncompilerwrappers; fullauxiliarycommands capturedRAW. Also
fixesfalsecompileridentificationofsource*.cc arguments.
ExactauxruleREDreproobserved;52localguardsPASS3.624s and96legacyruntime
guardsPASS4.87s. Scopedreviewd2fe2a235 pending whilenewGitvalidatorprepared.
No native/source/binary/reinstallchanges; continue SAMECPUlease only.

Reviewd2fe2a235 foundopaque `sh -c` couldhidecompilerinone token.
NewREDwrapperregression observed, then rejectopaque shells/env/cache/
inlinePython/CMakecompilerwrappers beforeauxfallback; no compiler/rsp
relaxation.54localguardsPASS5.610s+96legacyPASS3.89s+BashsyntaxPASS.
Validationcontinuation now collects audit+identity diagnostics together,
APIonlyifidentityPASS, overallANDallchecks mandatory; fullerrorfilesretained,
boundedtailsprintedatfailure toavoidanotherdiagnosticSSHroundtrip.
No compiledartifact/source/flagchange. ContinueCPU14254131withfreshmetadata
outputs only; sourcevalidationfix publication+Gitstage+nodeexecution chained
to reduceidle latency, not anotherresource job.
