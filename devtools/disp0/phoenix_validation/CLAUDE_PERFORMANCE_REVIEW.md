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
