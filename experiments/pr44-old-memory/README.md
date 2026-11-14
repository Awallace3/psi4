# Old-code protein157 memory floor

Use the actual frozen PR44 base `8f46e95cba`, NOT the new code's LEGACY selector.
Reuse its validated native build/hash; no new build or native modifications.
The source tree in this *harness* worktree is not the binary under test.
Keep it separate from the active new-code jobs' frozen checkout.

First run `submit.py CAMPAIGN --smoke`: one 2-core/8GiB/15min embers water-dimer
test of the **old binary** (forced disk failure, namespace-corrected completion,
and MEM_DF reference). Then use a new campaign for at most two full budgets:

```
submit.py CAMPAIGN --memory-gb BUDGET1 BUDGET2 --qos inferno --time 16:00:00 --test-only
```

**Full inferno runs require fresh explicit approval and credit checks.** The
approval for new-code jobs15602505/15602506 does not cover these old-code runs.
Do not submit speculative decades of budgets before reviewing source gates.

Source audit: old DFHelper defaults to INCORE **only when SCF_SUBTYPE is unset**,
with a dense round-2 admission requirement of471.945GB for this system.
Our original comparison config already explicitly sets `SCF_SUBTYPE=AUTO`,
so that default-policy restriction is not the matched baseline. The smoke
checks AUTO's changed-state on the actual old binary and records the auxiliary
max primitive/function counts; full runs also assert/record the explicit AUTO.
Old QR requires74.869GB declared; X/Y block down much further. Subject to
the runtime option audit,100GB and85GB are sensible first matched budgets.
Do NOT claim old code intrinsically needs472GB or call an admission error OOM.

Full cases retain192threads, exact geometry/options,1400GiB scheduler allocation
initially and2600GiB scratch filter. Apply the same process-local delta-HF
namespace correction to old and new; record hashes and disclose it in the PR.
Only budget changes; no SCF/grid/functional convergence changes.
All five components are compared to BOTH original1200GB references, using
the user-approved absolute1e-7Eh tolerance. Preserve original raw results.

`pr44-memory/compare_results.py RESULT_DIR ...` also evaluates historical and
active-new-code completed artifacts without changing their original1e-8 checks.
It checks geometry/options, refuses FDDS-only replay, and marks incomplete
results separately. It does not establish resource success by energy alone.

After finding completed old/new low-budget endpoints, lower the actual scheduler
cap (`--slurm-gib`) independently and bracket pass/OOM with page-cache headroom.
Report **tested intervals**, not exact minima. Admission errors, PSIO failures,
preemption, timeout and numerical failures are not OOM. Preserve partial samples
and Slurm `.batch`/step accounting when cgroup OOM prevents a result.json.
Compare scratch, RSS, cgroup peak, wall time and energy drift at each setting.
