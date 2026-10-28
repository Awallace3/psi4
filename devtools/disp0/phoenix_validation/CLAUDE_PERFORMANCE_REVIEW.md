# Independent Claude performance review

Claude Code, explicitly pinned to `claude-opus-4-6`, completed a read-only
review of HEAD `2155392a2e`. Tools were restricted to Read/Grep/Glob;
no shell, edits, subagents or GPU allocations. Two preceding Cursor-provider
attempts failed with connection errors and produced no review.

Full returned review and execution metadata:
`.pi/tasks/01a165d6-ab03-768d-83bd-c6172c2c1b1b-4065461/b27ef5581.output`.
Claude did not have remote result JSON or native library internals.
Its recommendations are source-level findings, not new GPU measurements.

## Principal findings

1. **Hardcoded single-thread native JIT.**
   `sapt_cuest_plan_owner.py:249` explicitly configures
   `jit_compile_threads=1`, despite eight allocated GPU-host CPUs.
   This is a confirmed configuration choice worth testing first. Its
   contribution to the observed 374/449-second construction windows is
   unmeasured; proportional speedup with more threads is not established.

2. **Separate native context for each auxiliary plan.**
   `sapt_cuest_dfmo_cuda.py:117-122` constructs a new `DiagnosticPlanOwner`
   for RIFIT; `sapt_cuest_plan_owner.py:251` creates its context.
   CUDA session/stream sharing is not native-context sharing.
   Repeated initialization/JIT is a plausible avoidable cost, but neither
   redundant compilation nor lack of a library-wide cache is proven.
   Sharing one owned context must first satisfy the vendor multi-plan,
   workspace and handle-lifetime contracts.

3. **Dispersion caching is too small for this fixture.**
   Five of fourteen S slabs fit the selected 64 MiB cache, leaving 145
   native producer calls. Full reuse predicts 28 calls, requiring about
   160.4 MiB for those slabs. This is a confirmed work-count opportunity,
   not a measured runtime gain or proof that increasing a budget is safe.
   Every changed budget still needs actual peak-memory admission.

4. **Producer timing combines too many operations.**
   The 82-second producer measurement includes audits, ABI binding,
   packing/allocation, workspace queries, native computation and device
   synchronization. Add subphase timers before claiming a native-kernel
   bottleneck. Do not remove identity checks on the basis of this review.

5. **Opaque reserve needs coexistence evidence.**
   Two native contexts share the single session's opaque reserve.
   The review identifies a possible underbudget, not demonstrated excess
   memory. Inspect actual coexistence memory before changing the ownership
   architecture or reserve.

## Corrections to the review itself

Claude labeled context/JIT findings P0. Their potential performance impact
is high, but source inspection does not establish their measured subcause
or justify a critical correctness severity.

Claude also called `raw_hf_upload_bytes` double-counting and proposed deleting
the second sum. **That proposed correction is incorrect.** The workflow
uploads the matrices/energies at lines85–86, then `prepare_hf` at lines109–110
uploads HF fields and energies again (`sapt_dispersion_cuda_setup.py:126,165`).
The counter reflects real duplicate transfers, although its second energy
sum counts full arrays rather than the frozen-occupied slices actually sent.
Reusing existing device inputs could remove duplicate work, but deleting
the second sum alone would underreport it. These small transfers
do not explain the measured setup windows, which precede them.

The review's suggested 82→16-second dispersion estimate assumes runtime
scales with producer call count. Retain it as a hypothesis only.
Its claim that per-call identity checks can simply be cached is likewise
not accepted without an explicit loader/identity lifetime contract.

## Methodology accountability and next experiment

The prior response-workspace/cache microbenchmarks excluded construction.
They did not establish whole-workflow readiness; end-to-end setup should have
been profiled on Cachy before spending further H200 time. The measured
whole-call regression must remain visible, not be removed by subtracting setup.

Next smallest experiment: a bounded, serialized Cachy construction-only test
with identical basis, FP64 policy and ownership/cleanup, recording wall and
process CPU time for admission, context, basis/pairs, workspace query and plan
creation. Compare JIT threads1 versus a host-budgeted value such as4.
Keep cache changes separate. Follow with unchanged numerical gates before
another H200 benchmark. No implementation change or GPU test was launched
as part of this review.

Follow-up implementation and local execution contract:
[`../cachy/CONSTRUCTION_PERFORMANCE.md`](../cachy/CONSTRUCTION_PERFORMANCE.md).
This adds instrumentation and an explicit thread-count experiment; it does not
replace the measured whole-call regression with an inferred speedup.

The [completed water construction diagnostic](../cachy/CONSTRUCTION_RESULTS.md)
found over 98.46% of owner-build time in per-object admission blocks.
JIT 1 versus 4 did not establish an improvement; native context creation was
milliseconds on that control. This is Cachy evidence, not a decomposition of
the earlier H200 peptide run. The next fix needs an explicit identity-lifetime
contract rather than deleting or blindly caching the existing checks.

The [opt-in owned-runtime follow-up](../cachy/OWNED_RUNTIME_RESULTS.md) now
implements that explicit lifetime contract and passes the small ten-process
water numerical gate. Matched Cachy construction/cleanup changed from147.1 to
33.4 seconds at JIT1, with unchanged tracked device memory. This is reduced
host-audit overhead, not an H200 kernel speedup. Whole-energy diagnostics still
take155.5–158.6 seconds; a CPU-only profile isolates repeated live checks and
pathname resolution as the next targets. The original fresh-byte path remains
the default, and no Phoenix run or broad production release is implied.
