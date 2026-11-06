# Wave core implementation status

Interface: `95f5c69784399c5167b6b8384b7bb362f5fbc0ad`, unchanged.

Initial integration commit includes pure whole-wave reduction, measured closed
campaign loading, wave-only record-body identity snapshots, separate one-shot
registration and read-only transports, shared serial/wave intent collision,
bounded held/released controller, and new metadata-only ops wrapper/tests.
Production activation remains disabled in `bank_wave_controller.py`.

Initial local verification:

* `/home/awallace43/miniconda3/envs/p4qca/bin/python -m pytest -q
  devtools/isapol_s66_campaign/test_bank_wave.py`: **57 passed**.
* Explicit p4qca `py_compile` for new wave Python modules: passed.
* `bash -n devtools/isapol_s66_campaign/bank_wave_ops.sbatch`: passed.

At this integration checkpoint, real runtime integration, SDK mockHTTP
failure-path coverage, full wave publication tests and full regression are
still pending. This commit is for concurrent integration/review, **not**
deployment approval or a scientific claim. No credentials, network, Phoenix,
SLURM jobs or quantum calculations have been accessed/executed.

Runtime agreed extra helper:
`bank_wave_transport.snapshot_dataset(client, *, binding)` enriches declared
wave records only (and previously wave-accepted records) with
`wave_identity_sha256`; serial prefix representations stay unchanged.
The stable full-record identity excludes the explicit result/lifecycle field
set in `LIFECYCLE_FIELDS`, retaining unknown/user/immutable fields.
`validate_capture` returns `snapshot_sha256`, `preserved_records`,
`peer_statuses`; `require_complete=True` is whole-wave-finalizer only.
