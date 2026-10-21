# One Phoenix native benzene pilot

Build14254599 passed five native smoke tests with the v3 stage/core SHA
`b283901ddac01e6bf7210a7461fb71cd5b2333cf9e42b99d60271959b40ac7eb`.
The new launcher uses that immutable source15cc9a7a and its original pilot.py,
not a rebuilt or edited scientific implementation. Launcher and scientific
source commits are separately pinned. Existing read-only dependencies and
explicit1.1.3 bootstrap match the accepted Phoenix runtime.

The first pilot targets the SAME build nodeatl1-1-02-007-23-1 because the
binary was compiled-march=native; arbitrary cpu-small hardware is not assumed
ISA-compatible. No affinity policy or numerical algorithm is changed.

The committed benzene.sbatch requests one gts-cs207-chemx/cpu-small/embers
allocation,8CPUs48G,6h,one numerical thread. The accepted pilot.json sets
24GiB numerical memory and its unchanged work/I/O/LW/PFIT/scientific controls.
Scratch200GiB is a declared budget, not a hard quota; pilot requires400GiB
free. Output/scratch are new. The exact hash-pinned fixture is transferred
privately only, never committed.

Only one SCF and one public native property pipeline execute, through the
already-reviewed pilot.py. Success exports all11response nodes and optional
rank≤2 mixed/recoupled results. Strict production refusal stays nonzero and
preserves the original error/log/SCF if available; no gate relaxation,
automatic retry, old overlay or second reporting-only ISA calculation.
Full pre-throw tensor capture and independent anisotropic CamCASP comparison
are still absent. A useful refusal is NOT scientific PASS.

prepare mode creates new input/log/metadata/scratch parents only. After exact
private fixture transfer, local/remote stdlib tests, --test-only and fresh
submit-headroom checks, coordinator may grant/submit one pilot. No science
runs during those checks. TERM-safe batch and payload terminal receipts and
runtime before/after maps must be reviewed alongside SLURM accounting.
