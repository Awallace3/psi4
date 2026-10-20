# Bounded shared cuEST interactive allocation

One owning agent requests an eight-hour H200 allocation with `salloc --no-shell`
(8 CPUs, 128 GiB, explicit approved inferno). Other agents must not submit
competing requests. The owner publishes an atomic `allocation.json` containing
the actual job ID, hardware/resources, run directory and `status=allocated`
only after allocation grant. No GPU work is permitted while it says pending.

All participants run `cuest_shared_step.sh` **on Phoenix**, supplying this same
manifest. Its common `gpu-step.lock` covers the entire bounded `srun` command;
steps request all GPU/CPU/memory resources exclusively. Do not bypass that
lock with raw `srun`, SSH-to-node launches, background GPU consumers or another
process in the step. Concurrent GPU work invalidates benchmark isolation.
The payload records GPU processes before and after the command against the
allocation's actual full CUDA UUID. Missing cleanup proof quarantines the
allocation and blocks all further steps until the owner investigates; it never
kills unknown processes to clear the gate. No other participant should remove
QUARANTINED. A zero exit code without cleanup proof is not accepted.

The owner assigns windows through workmux before each step. Initial order:
FDDS short correctness/diagnostic gate; RKS bounded failure diagnosis; FDDS
small-system paired timing; other coordinated response diagnostics; then
longer final timing windows. A failed short gate gets bounded diagnosis before
large timing, not a blind full-suite rerun. Agents must not occupy hours while
waiting for instructions. Lock waits are bounded at 120 seconds; coordinate
again instead of polling/retrying. Command bounds are at most three hours.

Each agent retains its own pinned source, build, dependency environment and
output directories. No shared environment/stage mutation. Candidate changes
use new git pins and isolated builds; do not overwrite frozen failed or
validated stages. Final correctness and timings use a recorded final frozen
pin, with exact staged import/core/library hashes, actual GPU-route proof,
unchanged scientific settings, and fresh-process same-allocation CPU controls
when claiming speedup. Record GPU name/driver, CPU threads, memory, allocation
ID, step bounds and that other projects shared the allocation sequentially.
The helper's COMMAND_EXITED_ZERO is not a scientific pass certificate.

Write completion/failure evidence even for partial runs. The owner releases
the allocation promptly after all assigned work finishes; the eight-hour
limit is a ceiling, not a target. No auto-merge or cancellations of other jobs.
